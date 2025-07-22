
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

#define LOG_ENTER(OP_NAME) std::cout << "TCU BACKEND aten::" << OP_NAME << " called " << std::endl;
#define LOG_INFO(information)  std::cout << information << std::endl;
#define LOG_TENSOR_SIZE(tensor_name, tensor)  {\
auto sizes = tensor.sizes();\
std::cout << tensor_name << " Tensor dimensions: ";\
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


at::Tensor empty_strided(c10::ArrayRef<int64_t> size, c10::ArrayRef<int64_t> stride, c10::optional<at::ScalarType> dtype ,
                         c10::optional<at::Layout> layout ,
                         c10::optional<at::Device> device ,
                         c10::optional<bool> pin_memory );
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);
at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                                      c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                                      c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format);
at::Tensor view(const at::Tensor & self, at::IntArrayRef size);
at::Tensor as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef  stride,
                      c10::optional<int64_t> storage_offset);


at::Tensor & tcu_bmm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out);
at::Tensor & tcu_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out);
at::Tensor & div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
at::Tensor & addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2,
                       const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out);
at::Tensor & arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out);
at::Tensor & pow_out_scalar(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out);
at::Tensor & pow_out(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out);
at::Tensor & reciprocal_out(const at::Tensor & self, at::Tensor & out);
at::Tensor & mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
at::Tensor & sub_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out);

at::Tensor & gt_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);
at::Tensor where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other);
at::Tensor & cos_out(const at::Tensor & self, at::Tensor & out);
at::Tensor & sin_out(const at::Tensor & self, at::Tensor & out);
at::Tensor & cat_out(const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out);
at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);
at::Tensor & rsqrt_out(const at::Tensor & self, at::Tensor & out);
at::Tensor & zero_(at::Tensor & self);
at::Tensor & mm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out);
at::Tensor & silu_out(const at::Tensor & self, at::Tensor & out);
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending,
    at::Tensor & values, at::Tensor & indices);
at::Tensor & gather_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out);
at::Tensor & lt_out_tensor(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
at::Tensor & lt_out_scalar(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);

at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);
at::Tensor & cumsum_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out);
at::Tensor & le_out_tensor(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
at::Tensor & scatter_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out);
at::Tensor & _log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out);
at::Tensor & index_out(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, at::Tensor & out);
at::Tensor & exponential_(at::Tensor & self, double lambd, c10::optional<at::Generator> generator);
at::Tensor & argmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out);
at::Tensor & uniform_(at::Tensor & self, double from, double to, c10::optional<at::Generator> generator);


