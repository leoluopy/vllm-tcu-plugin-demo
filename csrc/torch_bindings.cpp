#include <torch/extension.h>


#pragma once
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <torch/torch.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/EmptyTensor.h>

#include <iostream>
#include <ATen/ops/empty_strided.h>

#define LOG_ENTER(OP_NAME) std::cout << " VLLM_TCU BACKEND aten::" << OP_NAME << " called " << std::endl;
#define LOG_INFO(information)  std::cout << information << std::endl;
#define LOG_TENSOR_SIZE(tensor_name, tensor)  {\
auto sizes = tensor.sizes();\
std::cout << tensor_name << " vllm_tcu Tensor dimensions: ";\
for (size_t i = 0; i < sizes.size(); ++i) {\
std::cout << sizes[i];\
if (i < sizes.size() - 1) {\
std::cout << " x ";\
}\
}\
std::cout << std::endl;\
}
#define LOG_SIZES(tag, sizes_)  {\
auto sizes = sizes_;\
std::cout << tag << " sizes dimensions: ";\
for (size_t i = 0; i < sizes.size(); ++i) {\
std::cout << sizes[i];\
if (i < sizes.size() - 1) {\
std::cout << " x ";\
}\
}\
std::cout << std::endl;\
}

torch::Tensor rms_norm_impl( torch::Tensor input,torch::Tensor weight, double eps = 1e-8) {
    LOG_ENTER("rms_norm_impl");
    LOG_TENSOR_SIZE(" rms_norm_impl input",input);
    LOG_TENSOR_SIZE(" rms_norm_impl weight",weight);

    at::Tensor cpu_input = input.to(at::device(c10::kCPU));
    at::Tensor cpu_weight = weight.to(at::device(c10::kCPU));

    auto norm_shape = cpu_input.sizes().slice(input.dim() - 1);
    at::Tensor cpu_out = torch::rms_norm(cpu_input,norm_shape,cpu_weight,eps);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));

    LOG_TENSOR_SIZE(" rms_norm_impl output",tcu_out);
    return tcu_out;
}
torch::Tensor flash_attention_qlens(at::Tensor & query,  at::Tensor & key,  at::Tensor & value,at::Tensor & attn_mask,double scale) {

    LOG_ENTER("flash_attention_qlens");
    LOG_TENSOR_SIZE(" flash_attention_qlens query",query);
    LOG_TENSOR_SIZE(" flash_attention_qlens key",key);
    LOG_TENSOR_SIZE(" flash_attention_qlens value",value);

    at::Tensor cpu_query = query.to(at::device(c10::kCPU));
    at::Tensor cpu_key = key.to(at::device(c10::kCPU));
    at::Tensor cpu_value = value.to(at::device(c10::kCPU));
    at::Tensor cpu_mask = attn_mask.to(at::device(c10::kCPU));

    torch::Tensor cpu_output =torch::scaled_dot_product_attention(cpu_query,cpu_key, cpu_value, cpu_mask, 0.0, false, scale, false);
    at::Tensor tcu_out = cpu_output.to(at::device(c10::kPrivateUse1));

    LOG_TENSOR_SIZE(" flash_attention_qlens output",tcu_out);

    return tcu_out;
}
void reshape_and_cache(torch::Tensor key,torch::Tensor value,torch::Tensor& key_cache,torch::Tensor& value_cache, torch::Tensor slot_mapping) {

    LOG_ENTER("reshape_and_cache");
}

TORCH_LIBRARY(_C, m) {
    m.def("rms_norm(Tensor input, Tensor weight, float eps=1e-8) -> Tensor");
    m.def("reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, Tensor slot_indices) -> ()");
    m.def("flash_attention_qlens(Tensor! query, Tensor! key, Tensor! value, Tensor! mask,  float scale_value) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C, PrivateUse1, m) {
    m.impl("rms_norm", rms_norm_impl);
    m.impl("reshape_and_cache", reshape_and_cache);
    m.impl("flash_attention_qlens", flash_attention_qlens);
}

TORCH_LIBRARY_IMPL(_C, AutogradPrivateUse1, m) {
    m.impl("rms_norm", rms_norm_impl);
    m.impl("reshape_and_cache", reshape_and_cache);
    m.impl("flash_attention_qlens", flash_attention_qlens);
}

// 关键修改：使用明确的模块名称 _C
PYBIND11_MODULE(_C, m) {
    m.doc() = "vLLM custom operators (direct _C binding)";
}