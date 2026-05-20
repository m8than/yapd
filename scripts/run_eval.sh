#!/usr/bin/env bash
# Reproduce the paper's zero-shot TTS scores for Chatterbox-Flash.
#
# Pipeline (English-only):
#   stage 1: download eval datasets + models from HuggingFace (k2-fsa).
#   stage 2: LibriSpeech-PC test-clean  → SIM-o (omnivoice) + WER (HuBERT) + UTMOS.
#   stage 3: Seed-TTS test-en           → SIM-o (omnivoice) + WER (Whisper, seedtts-style) + UTMOS.
#
# Usage:
#   bash scripts/run_eval.sh                         # full pipeline
#   stage=2 stop_stage=3 bash scripts/run_eval.sh
#   CKPT_DIR=/path/to/chatterbox-flash-ckpt SKIP_EVAL=1 bash scripts/run_eval.sh
#
# Requirements:
#   pip install -e .
#   pip install omnivoice   # provides the eval scripts
#   hf auth login   # if any of the eval models require auth
set -euo pipefail

# ── repo root ─────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── stage selection ───────────────────────────────────────────────
stage="${stage:-1}"
stop_stage="${stop_stage:-3}"

# ── env ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── checkpoints / output ──────────────────────────────────────────
# Local directory containing the converted chatterbox-flash safetensors
# (t3_flash.safetensors, s3gen.safetensors, ve.safetensors, tokenizer.json).
# Or pass HF_REPO_ID instead to pull from the hub.
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/chatterbox-flash}"
# Default to pulling the released checkpoint from the Hugging Face Hub. Set
# CKPT_DIR to an existing local checkout (or HF_REPO_ID="" + a populated
# CKPT_DIR) to use local weights instead.
HF_REPO_ID="${HF_REPO_ID:-ResembleAI/chatterbox-flash}"

# The k2-fsa eval JSONLs hard-code ``ref_audio`` paths under a ``download/``
# prefix, resolved relative to the parent of DOWNLOAD_DIR (see REF_AUDIO_ROOT
# below). DOWNLOAD_DIR must therefore be named ``download`` for those relative
# paths to resolve — otherwise every row's ref_audio misses and 0 are generated.
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${REPO_ROOT}/download}"
TTS_EVAL_DATA_DIR="${DOWNLOAD_DIR}/tts_eval_datasets"
TTS_EVAL_MODEL_DIR="${DOWNLOAD_DIR}/tts_eval_models"

# JSONL ref_audio paths are relative to the parent of DOWNLOAD_DIR.
REF_AUDIO_ROOT="${REF_AUDIO_ROOT:-$(dirname "${DOWNLOAD_DIR}")}"

RES_ROOT="${RES_ROOT:-${REPO_ROOT}/eval_out}"
mkdir -p "${RES_ROOT}"

SKIP_EVAL="${SKIP_EVAL:-0}"

# ── inference knobs (defaults = paper config) ─────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bf16}"
DRF_BLOCK_SIZE="${DRF_BLOCK_SIZE:-16}"
NUM_STEPS="${NUM_STEPS:-10}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TIME_SHIFT_TAU="${TIME_SHIFT_TAU:-0.5}"
OMNIVOICE_SCHEDULE_T_SHIFT="${OMNIVOICE_SCHEDULE_T_SHIFT:-0.5}"   # early decoding
POSITION_TEMPERATURE="${POSITION_TEMPERATURE:-5.0}"
CFG_SCALE="${CFG_SCALE:-1.0}"
EXAGGERATION="${EXAGGERATION:-0.5}"

# ── eval throughput knobs ─────────────────────────────────────────
SIM_NJ_PER_GPU="${SIM_NJ_PER_GPU:-2}"
WER_BATCH_SIZE="${WER_BATCH_SIZE:-32}"
WER_NJ_PER_GPU="${WER_NJ_PER_GPU:-2}"
UTMOS_NJ_PER_GPU="${UTMOS_NJ_PER_GPU:-2}"

run_generate() {
    local test_jsonl="$1" res_dir="$2"
    echo "──────── generate ────────"
    echo "  test_list = ${test_jsonl}"
    echo "  res_dir   = ${res_dir}"
    mkdir -p "${res_dir}"
    python -m chatterbox_flash.eval.generate \
        --ckpt_dir "${CKPT_DIR}" \
        --test_list "${test_jsonl}" \
        --res_dir "${res_dir}" \
        --ref_audio_root "${REF_AUDIO_ROOT}" \
        --language_id_filter en \
        --batch_size "${BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --drf_block_size "${DRF_BLOCK_SIZE}" \
        --num_steps "${NUM_STEPS}" \
        --temperature "${TEMPERATURE}" \
        --time_shift_tau "${TIME_SHIFT_TAU}" \
        --omnivoice_schedule_t_shift "${OMNIVOICE_SCHEDULE_T_SHIFT}" \
        --position_temperature "${POSITION_TEMPERATURE}" \
        --cfg_scale "${CFG_SCALE}" \
        --exaggeration "${EXAGGERATION}" \
        --skip_existing
}

run_eval() {
    if [ "${SKIP_EVAL}" = "1" ]; then
        echo "  [skip-eval] $*"
        return 0
    fi
    python -m "$@"
}

# ════════════════════════════════════════════════════════════════
# Stage 1: download eval datasets + models
# ════════════════════════════════════════════════════════════════
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: download eval datasets + models"
    mkdir -p "${TTS_EVAL_DATA_DIR}" "${TTS_EVAL_MODEL_DIR}"

    hf_repo=k2-fsa/TTS_eval_datasets
    for file in \
        librispeech_pc_test_clean.jsonl \
        librispeech_pc_test_clean_transcript.jsonl \
        seedtts_test_en.jsonl ; do
        if [ ! -f "${TTS_EVAL_DATA_DIR}/${file}" ]; then
            echo "Downloading ${file}…"
            hf download --repo-type dataset \
                --local-dir "${TTS_EVAL_DATA_DIR}/" "${hf_repo}" "${file}"
        fi
    done

    for file in \
        librispeech_pc_testset.tar.gz \
        seedtts_testset.tar.gz ; do
        if [ ! -f "${TTS_EVAL_DATA_DIR}/${file}" ]; then
            echo "Downloading ${file}…"
            hf download --repo-type dataset \
                --local-dir "${TTS_EVAL_DATA_DIR}/" "${hf_repo}" "${file}"
        fi
        sentinel="${TTS_EVAL_DATA_DIR}/.${file%.tar.gz}.extracted"
        if [ ! -f "${sentinel}" ]; then
            echo "Extracting ${file}…"
            tar -xzf "${TTS_EVAL_DATA_DIR}/${file}" -C "${TTS_EVAL_DATA_DIR}/"
            touch "${sentinel}"
        fi
    done

    if [ ! -f "${TTS_EVAL_MODEL_DIR}/.downloaded" ]; then
        echo "Downloading eval models (k2-fsa/TTS_eval_models)…"
        hf download --local-dir "${TTS_EVAL_MODEL_DIR}" \
            k2-fsa/TTS_eval_models
        touch "${TTS_EVAL_MODEL_DIR}/.downloaded"
    fi

    if [ -n "${HF_REPO_ID}" ] && [ ! -d "${CKPT_DIR}" ]; then
        echo "Downloading Chatterbox-Flash from ${HF_REPO_ID}…"
        mkdir -p "${CKPT_DIR}"
        hf download --local-dir "${CKPT_DIR}" "${HF_REPO_ID}"
    fi
fi

# ════════════════════════════════════════════════════════════════
# Stage 2: LibriSpeech-PC test-clean
# ════════════════════════════════════════════════════════════════
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: LibriSpeech-PC test-clean"
    test_jsonl="${TTS_EVAL_DATA_DIR}/librispeech_pc_test_clean.jsonl"
    transcript_jsonl="${TTS_EVAL_DATA_DIR}/librispeech_pc_test_clean_transcript.jsonl"
    res_dir="${RES_ROOT}/librispeech_pc"

    run_generate "${test_jsonl}" "${res_dir}"

    run_eval omnivoice.eval.speaker_similarity.sim \
        --wav-path "${res_dir}" \
        --test-list "${test_jsonl}" \
        --decode-path "${res_dir}.sim.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${SIM_NJ_PER_GPU}"

    run_eval omnivoice.eval.wer.hubert \
        --wav-path "${res_dir}" \
        --test-list "${transcript_jsonl}" \
        --decode-path "${res_dir}.wer.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --batch-size "${WER_BATCH_SIZE}" \
        --nj-per-gpu "${WER_NJ_PER_GPU}"

    run_eval omnivoice.eval.mos.utmos \
        --wav-path "${res_dir}" \
        --test-list "${test_jsonl}" \
        --decode-path "${res_dir}.mos.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${UTMOS_NJ_PER_GPU}"
fi

# ════════════════════════════════════════════════════════════════
# Stage 3: Seed-TTS test-en
# ════════════════════════════════════════════════════════════════
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Seed-TTS test-en"
    test_jsonl="${TTS_EVAL_DATA_DIR}/seedtts_test_en.jsonl"
    res_dir="${RES_ROOT}/seedtts_en"

    run_generate "${test_jsonl}" "${res_dir}"

    run_eval omnivoice.eval.speaker_similarity.sim \
        --wav-path "${res_dir}" \
        --test-list "${test_jsonl}" \
        --decode-path "${res_dir}.sim.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${SIM_NJ_PER_GPU}"

    if [ "${SKIP_EVAL}" != "1" ]; then
        python -m chatterbox_flash.eval.wer_seedtts \
            --wav-path "${res_dir}" \
            --test-list "${test_jsonl}" \
            --decode-path "${res_dir}.wer.log" \
            --model-dir "${TTS_EVAL_MODEL_DIR}" \
            --lang en \
            --nj-per-gpu "${WER_NJ_PER_GPU}" \
            --text-norm "${WER_TEXT_NORM:-omnivoice}"
    fi

    run_eval omnivoice.eval.mos.utmos \
        --wav-path "${res_dir}" \
        --test-list "${test_jsonl}" \
        --decode-path "${res_dir}.mos.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${UTMOS_NJ_PER_GPU}"
fi

echo "──────── done ────────"
echo "Outputs   : ${RES_ROOT}"
echo "Eval logs : *.sim.log / *.wer.log / *.mos.log alongside each res_dir"

# ── final metric summary ──────────────────────────────────────────
_metric() {
    local log="$1" pat="$2" extract="$3"
    if [ -f "${log}" ]; then
        grep -E "${pat}" "${log}" | tail -1 | sed -E "${extract}" || echo "-"
    else
        echo "-"
    fi
}

_load_metrics() {
    local rd="$1"
    _sim=$(_metric "${rd}.sim.log" 'Average SIM-o:' \
        's/.*Average SIM-o:[[:space:]]*([0-9.]+).*/\1/')
    _wer=$(_metric "${rd}.wer.log" 'Seed-TTS WER \(Avg of WERs\):' \
        's/.*Seed-TTS WER \(Avg of WERs\):[[:space:]]*([0-9.]+)%.*/\1/')
    if [ -z "${_wer}" ] || [ "${_wer}" = "-" ]; then
        _wer=$(_metric "${rd}.wer.log" 'WER \(Weighted\):' \
            's/.*WER \(Weighted\):[[:space:]]*([0-9.]+)%.*/\1/')
    fi
    if [ -z "${_wer}" ] || [ "${_wer}" = "-" ]; then
        _wer=$(_metric "${rd}.wer.log" '^WER:' \
            's/.*^WER:[[:space:]]*([0-9.]+)%.*/\1/')
    fi
    [ -z "${_wer}" ] && _wer="-"
    _mos=$(_metric "${rd}.mos.log" 'Average UTMOS:' \
        's/.*Average UTMOS:[[:space:]]*([0-9.]+).*/\1/')
}

echo
echo "════════════ Results ════════════"
for label in librispeech_pc seedtts_en; do
    rd="${RES_ROOT}/${label}"
    if [ -d "${rd}" ]; then
        _load_metrics "${rd}"
        printf "%-18s  SIM-o=%s  WER=%s%%  UTMOS=%s\n" \
            "${label}" "${_sim}" "${_wer}" "${_mos}"
    fi
done
echo "════════════════════════════════"
