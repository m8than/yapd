from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from torch import nn
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False,
        voice_name: Optional[str] = None
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
    
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)


    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None
    
    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: Union[float, torch.FloatTensor] = 1.0,
        input_lengths: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        if input_lengths is None:
            input_lengths = torch.full(
                (input_ids.shape[0],), input_ids.shape[-1],
                device=input_ids.device, dtype=torch.long,
            )
        if ref_s.ndim == 2:
            ref_s = ref_s.unsqueeze(1)
        ref_s = ref_s.to(self.device)
        s = ref_s[:, :, 128:]
    
        max_len = input_ids.shape[1]
        text_mask = torch.arange(max_len, device=self.device).unsqueeze(0)
        sequence_mask = (text_mask.expand(input_ids.shape[0], -1) >= input_lengths.unsqueeze(1)).to(self.device) # b x seq_len
        # Convert to attention mask where 1 means "attend to this token" and 0 means "ignore this token"
        attention_mask = (~sequence_mask).to(dtype=ref_s.dtype)
        
        # Forward pass through BERT
        bert_dur = self.bert(input_ids, attention_mask=attention_mask) # b x seq_len x hidden
        d_en = self.bert_encoder(bert_dur) # b x seq_len x hidden
        
        # Pass through predictor
        d = self.predictor.text_encoder(
            d_en, s, input_lengths, sequence_mask,
        )
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x) # b x seq_len x max_dur
        speed_value = (
            speed.to(self.device).reshape(-1, 1)
            if torch.is_tensor(speed) else float(speed)
        )
        duration = torch.round(
            torch.sigmoid(duration).sum(dim=-1) * attention_mask / speed_value
        )
        updated_seq_lengths = torch.sum(duration, dim=-1) # b
        duration = duration.to(torch.float32)
        duration = duration.clamp(min=1).long()
        # Text batches are grouped by exact token length by the scheduler.
        # Decoder batches are further grouped by predicted frame length so no
        # waveform row observes padded normalization/convolution context.
        t_en = self.text_encoder(input_ids, input_lengths, attention_mask)
        groups = {}
        for row, frames in enumerate(updated_seq_lengths.cpu().tolist()):
            groups.setdefault(int(frames), []).append(row)
        decoded = []
        frame_lengths = torch.empty(
            input_ids.shape[0], dtype=torch.long, device=self.device,
        )
        max_audio = 0
        for frames, rows in groups.items():
            row_indices = torch.tensor(rows, device=self.device)
            duration_group = duration.index_select(0, row_indices)
            d_group = d.index_select(0, row_indices)
            s_group = s.index_select(0, row_indices)
            ref_group = ref_s.index_select(0, row_indices)
            text_group = t_en.index_select(0, row_indices)
            frame_indices = torch.arange(
                frames, device=self.device,
            ).view(1, 1, -1)
            duration_cumsum = duration_group.cumsum(dim=1).unsqueeze(-1)
            zeros = torch.zeros(
                len(rows), 1, 1,
                device=self.device, dtype=duration_cumsum.dtype,
            )
            alignment = (
                (duration_cumsum > frame_indices)
                & (
                    frame_indices >= torch.cat(
                        [zeros, duration_cumsum[:, :-1, :]], dim=1,
                    )
                )
            ).to(dtype=ref_s.dtype).transpose(1, 2)
            en = torch.bmm(alignment, d_group)
            frame_mask = torch.ones(
                len(rows), frames, device=self.device, dtype=ref_s.dtype,
            )
            group_lengths = torch.full(
                (len(rows),), frames,
                dtype=torch.long, device=self.device,
            )
            f0, noise, _ = self.predictor.F0Ntrain(
                en, s_group, group_lengths, frame_mask,
            )
            asr = torch.bmm(alignment, text_group)
            audio = self.decoder(
                asr, f0, noise, ref_group[:, :, :128], frame_mask,
            )
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            samples_per_frame = audio.shape[-1] // frames
            length = frames * samples_per_frame
            frame_lengths[row_indices] = length
            max_audio = max(max_audio, length)
            decoded.append((row_indices, audio[:, :length]))
        output = torch.zeros(
            input_ids.shape[0], max_audio,
            dtype=decoded[0][1].dtype, device=self.device,
        )
        for rows, audio in decoded:
            output[rows, :audio.shape[-1]] = audio
        return output, frame_lengths

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(
            input_ids, ref_s, speed,
            torch.tensor([input_ids.shape[1]], device=self.device),
        )
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
