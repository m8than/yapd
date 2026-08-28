"""Native ROCm inference kernels for the serving path."""
from __future__ import annotations

import os
from functools import lru_cache
from types import MethodType

import torch
from torch.utils.cpp_extension import load_inline

_CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor snake_hip(torch::Tensor x, torch::Tensor alpha);
torch::Tensor snake_bias_hip(torch::Tensor x, torch::Tensor alpha, torch::Tensor bias);
torch::Tensor residual_bias_hip(torch::Tensor x, torch::Tensor residual, torch::Tensor bias);
torch::Tensor mean3_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);
std::vector<torch::Tensor> residual_bias_snake_hip(torch::Tensor x, torch::Tensor residual, torch::Tensor bias, torch::Tensor alpha);
std::vector<torch::Tensor> branch_start_hip(torch::Tensor x, torch::Tensor source, torch::Tensor alpha0, torch::Tensor alpha1, torch::Tensor alpha2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("snake", &snake_hip);
  m.def("snake_bias", &snake_bias_hip);
  m.def("residual_bias", &residual_bias_hip);
  m.def("mean3", &mean3_hip);
  m.def("residual_bias_snake", &residual_bias_snake_hip);
  m.def("branch_start", &branch_start_hip);
}
"""

_HIP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <cmath>

__device__ __forceinline__ float snake_value(float x, float alpha) {
  float s = sinf(x * alpha);
  return x + (s * s) / (alpha + 1.0e-9f);
}

__global__ void snake_kernel(const float* x, const float* alpha, float* out, int64_t n, int channels, int width) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { int c = (i / width) % channels; out[i] = snake_value(x[i], alpha[c]); }
}
__global__ void snake_bias_kernel(const float* x, const float* alpha, const float* bias, float* out, int64_t n, int channels, int width) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { int c = (i / width) % channels; out[i] = snake_value(x[i] + bias[c], alpha[c]); }
}
__global__ void residual_bias_kernel(const float* x, const float* residual, const float* bias, float* out, int64_t n, int channels, int width) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { int c = (i / width) % channels; out[i] = x[i] + residual[i] + bias[c]; }
}
__global__ void mean3_kernel(const float* a, const float* b, const float* c, float* out, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (a[i] + b[i] + c[i]) * (1.0f / 3.0f);
}
__global__ void residual_bias_snake_kernel(const float* x, const float* residual, const float* bias, const float* alpha, float* raw_out, float* snake_out, int64_t n, int channels, int width) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = (i / width) % channels; float raw = x[i] + residual[i] + bias[c];
  raw_out[i] = raw; snake_out[i] = snake_value(raw, alpha[c]);
}
__global__ void branch_start_kernel(const float* x, const float* source, const float* alpha0, const float* alpha1, const float* alpha2, float* raw_out, float* out0, float* out1, float* out2, int64_t n, int channels, int width) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = (i / width) % channels; float raw = x[i] + source[i];
  raw_out[i] = raw; out0[i] = snake_value(raw, alpha0[c]); out1[i] = snake_value(raw, alpha1[c]); out2[i] = snake_value(raw, alpha2[c]);
}

void check_tensor(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.dim() == 3, "contiguous fp32 CUDA BCT required");
}
void check_channel(torch::Tensor x, torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.scalar_type() == torch::kFloat32 && value.is_contiguous() && value.numel() == x.size(1), name);
}
void launch_shape(torch::Tensor x, int& blocks, int& threads) { threads = 256; blocks = (int)((x.numel() + threads - 1) / threads); }

torch::Tensor snake_hip(torch::Tensor x, torch::Tensor alpha) {
  check_tensor(x); check_channel(x, alpha, "alpha mismatch"); auto out=torch::empty_like(x); int b,t; launch_shape(x,b,t);
  snake_kernel<<<b,t,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),alpha.data_ptr<float>(),out.data_ptr<float>(),x.numel(),x.size(1),x.size(2)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor snake_bias_hip(torch::Tensor x, torch::Tensor alpha, torch::Tensor bias) {
  check_tensor(x); check_channel(x,alpha,"alpha mismatch"); check_channel(x,bias,"bias mismatch"); auto out=torch::empty_like(x); int b,t; launch_shape(x,b,t);
  snake_bias_kernel<<<b,t,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),alpha.data_ptr<float>(),bias.data_ptr<float>(),out.data_ptr<float>(),x.numel(),x.size(1),x.size(2)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor residual_bias_hip(torch::Tensor x, torch::Tensor residual, torch::Tensor bias) {
  check_tensor(x); check_tensor(residual); TORCH_CHECK(x.sizes()==residual.sizes(),"residual shape mismatch"); check_channel(x,bias,"bias mismatch"); auto out=torch::empty_like(x); int b,t; launch_shape(x,b,t);
  residual_bias_kernel<<<b,t,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),residual.data_ptr<float>(),bias.data_ptr<float>(),out.data_ptr<float>(),x.numel(),x.size(1),x.size(2)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor mean3_hip(torch::Tensor a, torch::Tensor b0, torch::Tensor c) {
  check_tensor(a); check_tensor(b0); check_tensor(c); TORCH_CHECK(a.sizes()==b0.sizes() && a.sizes()==c.sizes(),"mean3 shape mismatch"); auto out=torch::empty_like(a); int b,t; launch_shape(a,b,t);
  mean3_kernel<<<b,t,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(),b0.data_ptr<float>(),c.data_ptr<float>(),out.data_ptr<float>(),a.numel()); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
std::vector<torch::Tensor> residual_bias_snake_hip(torch::Tensor x, torch::Tensor residual, torch::Tensor bias, torch::Tensor alpha) {
  check_tensor(x); check_tensor(residual); TORCH_CHECK(x.sizes()==residual.sizes(),"residual shape mismatch"); check_channel(x,bias,"bias mismatch"); check_channel(x,alpha,"alpha mismatch"); auto raw=torch::empty_like(x); auto act=torch::empty_like(x); int b,t; launch_shape(x,b,t);
  residual_bias_snake_kernel<<<b,t,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),residual.data_ptr<float>(),bias.data_ptr<float>(),alpha.data_ptr<float>(),raw.data_ptr<float>(),act.data_ptr<float>(),x.numel(),x.size(1),x.size(2)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {raw,act};
}
std::vector<torch::Tensor> branch_start_hip(torch::Tensor x, torch::Tensor source, torch::Tensor a0, torch::Tensor a1, torch::Tensor a2) {
  check_tensor(x); check_tensor(source); TORCH_CHECK(x.sizes()==source.sizes(),"source shape mismatch"); check_channel(x,a0,"alpha0 mismatch"); check_channel(x,a1,"alpha1 mismatch"); check_channel(x,a2,"alpha2 mismatch"); auto raw=torch::empty_like(x); auto o0=torch::empty_like(x); auto o1=torch::empty_like(x); auto o2=torch::empty_like(x); int b,t; launch_shape(x,b,t);
  branch_start_kernel<<<b,t,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),source.data_ptr<float>(),a0.data_ptr<float>(),a1.data_ptr<float>(),a2.data_ptr<float>(),raw.data_ptr<float>(),o0.data_ptr<float>(),o1.data_ptr<float>(),o2.data_ptr<float>(),x.numel(),x.size(1),x.size(2)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {raw,o0,o1,o2};
}
"""

@lru_cache(maxsize=1)
def load_snake_extension():
    os.environ.setdefault("MAX_JOBS", "16")
    return load_inline(name="chatterbox_hift_fused_hip_v7", cpp_sources=_CPP_SOURCE, cuda_sources=_HIP_SOURCE, extra_cflags=["-O3"], extra_cuda_cflags=["-O3","-ffast-math"], verbose=False)

def _fused_snake_forward(module, x):
    bias=getattr(module,"pre_bias",None); ext=load_snake_extension()
    if module.alpha_logscale or x.dtype!=torch.float32 or not x.is_contiguous():
        alpha=module.alpha.unsqueeze(0).unsqueeze(-1); alpha=torch.exp(alpha) if module.alpha_logscale else alpha; x=x+(bias.unsqueeze(0).unsqueeze(-1) if bias is not None else 0); return x+torch.sin(x*alpha).pow(2)/(alpha+module.no_div_by_zero)
    return ext.snake_bias(x,module.alpha.contiguous(),bias.contiguous()) if bias is not None else ext.snake(x,module.alpha.contiguous())

def _run_fused_resblock(module,x,preactivated=None):
    ext=load_snake_extension()
    for i in range(len(module.convs1)):
        y=module.activations1[i](x) if preactivated is None else preactivated; y=module.convs1[i](y); y=module.activations2[i](y); y=module.convs2[i](y); bias=getattr(module,f"residual_bias_{i}"); can=y.dtype==torch.float32 and y.is_contiguous() and x.is_contiguous()
        if can and i+1<len(module.convs1): x,preactivated=ext.residual_bias_snake(y,x,bias.contiguous(),module.activations1[i+1].alpha.contiguous())
        elif can: x=ext.residual_bias(y,x,bias.contiguous()); preactivated=None
        else: x=y+x+bias.unsqueeze(0).unsqueeze(-1); preactivated=module.activations1[i+1](x) if i+1<len(module.convs1) else None
    return x

def _fused_resblock_forward(module,x): return _run_fused_resblock(module,x)

def _fused_hift_decode(module, x, s):
    import torch.nn.functional as functional
    s_real, s_imag = module._stft(s.squeeze(1))
    s_stft = torch.cat([s_real, s_imag], dim=1)
    extension = load_snake_extension()
    x = module.conv_pre(x)
    for i in range(module.num_upsamples):
        x = functional.leaky_relu(x, module.lrelu_slope)
        x = module.ups[i](x)
        if i == module.num_upsamples - 1:
            x = module.reflection_pad(x)
        source = module.source_resblocks[i](module.source_downs[i](s_stft))
        x = x + source
        outputs = [
            module.resblocks[i * module.num_kernels + j](x)
            for j in range(module.num_kernels)
        ]
        x = (
            extension.mean3(*outputs)
            if all(y.dtype == torch.float32 and y.is_contiguous() for y in outputs)
            else sum(outputs) / module.num_kernels
        )
    x = module.conv_post(functional.leaky_relu(x))
    split = module.istft_params["n_fft"] // 2 + 1
    return torch.clamp(
        module._istft(torch.exp(x[:, :split, :]), torch.sin(x[:, split:, :])),
        -module.audio_limit, module.audio_limit,
    )

def install_fused_snake(module):
    load_snake_extension(); blocks=[m for m in module.modules() if m.__class__.__name__=="ResBlock"]
    for block in blocks:
        for i,(c1,c2,a2) in enumerate(zip(block.convs1,block.convs2,block.activations2)):
            if c1.bias is not None: a2.pre_bias=c1.bias; c1.bias=None
            if c2.bias is not None: setattr(block,f"residual_bias_{i}",c2.bias); c2.bias=None
        block.forward=MethodType(_fused_resblock_forward,block)
    if module.__class__.__name__=="HiFTGenerator": module.decode=MethodType(_fused_hift_decode,module)
    n=0
    for child in module.modules():
        if child.__class__.__name__=="Snake": child.forward=MethodType(_fused_snake_forward,child); n+=1
    return n
