#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <torch/csrc/Device.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/EmptyTensor.h>

#include <iostream>
#include <ATen/ops/empty_strided.h>

#include "op_implements.h"

static uint16_t CURR_DEVICE = 0;
#define LOG_CURRENT_DEVICE(TAG) std::cout << "     CURRENT DEVICE "<<TAG<< " :" << CURR_DEVICE << std::endl;
// #define LOG_CURRENT_DEVICE(TAG) {};
#define LOG_BASE_MEM_OP_LOG(TAG) std::cout << "     Call BASE MEM OP "<<TAG<< " called!" << std::endl;
//#define LOG_BASE_MEM_OP_LOG(TAG) {};

// Create and register a dummy device guard.
struct DummyDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
    DummyDeviceGuardImpl() {}
    explicit DummyDeviceGuardImpl(c10::DeviceType t) {
        TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
    }
    at::DeviceType type() const override {
        return at::DeviceType::PrivateUse1;
    }
    at::Device exchangeDevice(at::Device d) const override {
        TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
        TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
        at::Device old_device = getDevice();
        if (old_device.index() != d.index()) {
            // "set the active device"
            CURR_DEVICE = d.index();
        }
        return old_device;
    }
    at::Device getDevice() const override {
        LOG_CURRENT_DEVICE("GET");
        return at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE);
    }
    void setDevice(at::Device d) const override {
        LOG_CURRENT_DEVICE("SET OLD");
        TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
        TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
        at::Device current_device = getDevice();
        if (current_device != d) {
            CURR_DEVICE = d.index();
        }
    }
    void uncheckedSetDevice(at::Device d) const noexcept override {
            auto current_device = getDevice();
            if (current_device != d) {
                CURR_DEVICE = d.index();
            }
    }
    at::Stream getStream(at::Device d) const noexcept override {
            // no-op
            return at::Stream(at::Stream::DEFAULT, d);
    }
    // NB: These do NOT set the current device
    at::Stream exchangeStream(at::Stream) const noexcept override {
            // no-op
            return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE));
    }
    at::DeviceIndex deviceCount() const noexcept override {
            // Hardcoding the number of "valid" devices here at 2.
            return 2;
    }

    // Event-related functions
    void record(
            void** /*event*/,
            const at::Stream& /*stream*/,
            const at::DeviceIndex /*device_index*/,
            const c10::EventFlag /*flag*/) const override {
        TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.");
    }
    void block(void* /*event*/, const at::Stream& /*stream*/) const override {
        TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
    }
    bool queryEvent(void* /*event*/) const override {
        TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
    }
    void destroyEvent(void* /*event*/, const at::DeviceIndex /*device_index*/)
    const noexcept override {}

    // Stream-related functions
    bool queryStream(const at::Stream& /*stream*/) const override {
        return true;
    }
    void synchronizeStream(const at::Stream& /*stream*/) const override {
        // Don't wait for anything.
    }
};

struct DummyGuard {
    explicit DummyGuard() = delete;
    explicit DummyGuard(at::DeviceIndex device_index) : guard_(device_index) {}
    explicit DummyGuard(at::Device device) : guard_(device) {}
    DummyGuard(const DummyGuard&) = delete;
    DummyGuard& operator=(const DummyGuard&) = delete;
    DummyGuard(DummyGuard&& other) = delete;
    DummyGuard& operator=(DummyGuard&& other) = delete;

    void set_device(at::Device device) {
        guard_.set_device(device);
    }

    void reset_device(at::Device device) {
        guard_.reset_device(device);
    }

    void set_index(at::DeviceIndex device_index) {
        guard_.set_index(device_index);
    }

    at::Device original_device() const {
        return guard_.original_device();
    }

    at::Device current_device() const {
        return guard_.current_device();
    }

private:
    c10::impl::InlineDeviceGuard<DummyDeviceGuardImpl> guard_;
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, DummyDeviceGuardImpl);

// =====================================
// ========= Custom Allocators =========
// =====================================

// PyTorch provides an API for registering custom allocators for your device.
// You can create one by inheriting from the at::Allocator class,
// and registering your allocator for the particular device type
// (PrivateUse1 for open registration devices)

// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator {
    DummyCustomAllocator() = default;

    at::DataPtr allocate(size_t nbytes) override {
        LOG_BASE_MEM_OP_LOG("allocator's allocate() size:"+std::to_string(nbytes))
        void *data = c10::alloc_cpu(nbytes);
        return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE)};
    }

    static void ReportAndDelete(void *ptr) {
        if (!ptr) {
            return;
        }
        LOG_BASE_MEM_OP_LOG("allocator's delete()")
        c10::free_cpu(ptr);
    }

    at::DeleterFnPtr raw_deleter() const override {
        return &ReportAndDelete;
    }
    void copy_data(void* dest, const void* src, std::size_t count) const final {
        std::memcpy(dest, src, count);
    }
};

// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

at::Tensor empty_strided(c10::ArrayRef<int64_t> size, c10::ArrayRef<int64_t> stride,
                         c10::optional<at::ScalarType> dtype = c10::nullopt,
                         c10::optional<at::Layout> layout = c10::nullopt,
                         c10::optional<at::Device> device = c10::nullopt,
                         c10::optional<bool> pin_memory = c10::nullopt) {
    LOG_BASE_MEM_OP_LOG("aten::empty_strided()");
    LOG_INFO(dtype.value());
    LOG_SIZES("empty stried sizes ",size);
    const at::OptionalDeviceGuard device_guard(device);
    constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
    at::MemoryFormat memory_format;
    at::Tensor tensor = at::detail::empty_generic(size, &global_custom_alloc, private_use_ks,
                                                  c10::dtype_or_default(dtype), memory_format);

    return tensor;
}
at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    LOG_BASE_MEM_OP_LOG("aten::empty.memory_format()");
    const at::OptionalDeviceGuard device_guard(device);
    constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}
// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor &self, const at::Tensor &dst, bool non_blocking) {
    const at::OptionalDeviceGuard device_guard(at::device_of(self));
    LOG_BASE_MEM_OP_LOG("aten::_copy_from()")
    LOG_TENSOR_SIZE("self:",self);

    at::Tensor tmp_self;
    if (self.scalar_type() != dst.scalar_type())
    {
        std::cout << "           ####   type convert called  " << std::endl;
        at::Tensor cpu_self = self.to(at::device(c10::kCPU));
        at::Tensor cpu_dst = dst.to(at::device(c10::kCPU));
        tmp_self=cpu_self.type_as(cpu_dst);
        tmp_self = tmp_self.to(at::device(c10::kPrivateUse1));
    }else
    {
        tmp_self = self;
    }
    TORCH_CHECK(tmp_self.is_cpu() || tmp_self.device().type() == c10::DeviceType::PrivateUse1,
                "Dummy test only allows copy from cpu -> dummy device.");
    TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1,
                "Dummy test only allows copy from cpu -> dummy device.");
    // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
    TORCH_CHECK(tmp_self.sizes() == dst.sizes());
    TORCH_CHECK(tmp_self.scalar_type() == dst.scalar_type());
    TORCH_CHECK(tmp_self.is_contiguous() && dst.is_contiguous());

//    LOG_INFO("START COPY")
//    LOG_INFO(dst.storage().data_ptr().get())
//    LOG_INFO(tmp_self.storage().data_ptr().get())
//    LOG_INFO(abs(long(dst.storage().data_ptr().get()) - long(tmp_self.storage().data_ptr().get())))
//    LOG_INFO(tmp_self.storage().nbytes())

    std::memcpy(dst.storage().data_ptr().get(), tmp_self.storage().data_ptr().get(), tmp_self.storage().nbytes());
//    LOG_INFO("COPY DONE")
    return dst;
}
at::Tensor view(const at::Tensor & self, at::IntArrayRef size){
    LOG_BASE_MEM_OP_LOG("aten:view()")
    LOG_TENSOR_SIZE("view input",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor out = cpu_self.view(size);
    at::Tensor tcu_out = out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("view tcu out",tcu_out);
    return tcu_out;
}
bool hasZeroDimension(const at::IntArrayRef& size) {
    for (const auto& dim : size) {
        if (dim <= 0) {
            return true;
        }
    }
    return false;
}
at::Tensor as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef  stride,
                      c10::optional<int64_t> storage_offset=c10::nullopt)
{

    LOG_BASE_MEM_OP_LOG("aten::as_strided");
    LOG_TENSOR_SIZE("as_strided input",self);
    LOG_INFO(size);
    // if(hasZeroDimension(size)){
    //     LOG_INFO("return original zero empty");
    //     return self.clone();
    // }
    LOG_INFO(stride);
    std::cout << "self.is_contiguous():" << self.is_contiguous() << std::endl;
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor out = torch::as_strided(cpu_self,size,stride,storage_offset);
    at::Tensor out_contiguous = out.contiguous();

    at::Tensor tcu_out;
    if(self.sizes() != size){
        at::Tensor out_contiguous_new_allo = at::empty_like(out_contiguous);
        out_contiguous_new_allo.copy_(out_contiguous);
        tcu_out = out_contiguous_new_allo.to(at::device(c10::kPrivateUse1));
    }else{
        tcu_out = out_contiguous.to(at::device(c10::kPrivateUse1));;
    }
    LOG_TENSOR_SIZE("as_strided tcuout",tcu_out);
    std::cout << "tcu_out.is_contiguous():" << tcu_out.is_contiguous() << std::endl;
    return tcu_out;
}

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
    const at::OptionalDeviceGuard device_guard(at::device_of(self));
    // Not bothering to implement.
    // Should fill the tensor's data with "value".
    return self;
}

at::Tensor & normal_(at::Tensor & self, double mean, double std, c10::optional<torch::Generator> generator){
    std::cout << "Custom aten::normal_() called! size:"<< std::endl;
    return self;
}
at::Tensor triu(const at::Tensor & self, int64_t diagonal){
    std::cout << "Custom aten::triu() called! dtype :"<< self.dtype() << std::endl;

    return self;
}
at::Tensor bmm(const at::Tensor & self, const at::Tensor & mat2){
    std::cout << "Custom aten::bmm() called! "<< std::endl;
//    return torch::matmul(self,mat2);
}


// {"schema": "aten::convolution_overrideable(Tensor input, Tensor weight,
// Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
// "dispatch": "True", "default": "True"}
at::Tensor convolution_overrideable(const at::Tensor & input, const at::Tensor & weight,
                                const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding,
                                    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups)
{
    printf("Calling TCU OP:  convolution_overrideable \n");
    std::cout << input.device() << "," << weight.device() << ","  << std::endl;
    at::Tensor cpu_input = input.to(at::device(c10::kCPU));
    at::Tensor cpu_weight = weight.to(at::device(c10::kCPU));
//    at::Tensor cpu_bias = bias.value().to(at::device(c10::kCPU));
    std::cout << "cpu_input: d:" << cpu_input.device() << std::endl;

//    torch::nn::Conv2dOptions toptions(1, 1, 3);
    torch::Tensor out = torch::conv2d(cpu_input, cpu_weight, torch::Tensor(), stride, padding, dilation, groups);
    std::cout << " See torch::conv2d  out1: d:" << out.device() ;
    at::Tensor out_tcu = out.to(at::device(c10::kPrivateUse1));
    std::cout << "convolution_overrideable out2: d:" << out_tcu.device() << std::endl;
    return out_tcu ;
}
// // {"schema": "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var,
// bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input,
                                                                 const c10::optional<at::Tensor> & weight,
                                                                 const c10::optional<at::Tensor> & bias,
                                                                 const c10::optional<at::Tensor> & running_mean,
                                                                 const c10::optional<at::Tensor> & running_var,
                                                                 bool training, double momentum, double eps)
{
    printf("Calling TCU OP:  native_batch_norm \n");
    at::Tensor cpu_input = input.to(at::device(c10::kCPU));
    std::cout << " See torch::batch_norm  d: " << cpu_input.device() << std::endl;
    torch::Tensor cpu_weight;
    if (weight.has_value()) {
        cpu_weight = weight.value().to(at::device(c10::kCPU));
    }
    cpu_weight =cpu_weight.to(at::device(c10::kCPU));

    torch::Tensor cpu_bias;
    if (bias.has_value()) {
        cpu_bias = bias.value().to(at::device(c10::kCPU));
    }
    cpu_bias=cpu_bias.to(at::device(c10::kCPU));

    torch::Tensor cpu_running_mean;
    if (running_mean.has_value()) {
        cpu_running_mean = running_mean.value().to(at::device(c10::kCPU));
    }
    cpu_running_mean=cpu_running_mean.to(at::device(c10::kCPU));

    torch::Tensor cpu_running_var;
    if (running_var.has_value()) {
        cpu_running_var = running_var.value().to(at::device(c10::kCPU));
    }
    cpu_running_var=cpu_running_var.to(at::device(c10::kCPU));

    std::cout << cpu_weight.device() << cpu_bias.device() << cpu_running_mean.device() << cpu_running_var.device()
              << std::endl;
    auto output = torch::native_batch_norm(cpu_input, cpu_weight, cpu_bias,
                                           cpu_running_mean, cpu_running_var,
                                           training, momentum, eps);
    std::cout << " See torch::batch_norm  out1: d:" << (std::get<0>(output)).device();
    auto out_tcu1 = std::get<0>(output).to(at::device(c10::kPrivateUse1));
    auto out_tcu2 = std::get<1>(output).to(at::device(c10::kPrivateUse1));
    auto out_tcu3 = std::get<2>(output).to(at::device(c10::kPrivateUse1));

    std::cout << " See torch::batch_norm  out2: " << out_tcu1.device() << std::endl;
    std::tuple<at::Tensor, at::Tensor, at::Tensor> tupleOfTensors(out_tcu1, out_tcu2, out_tcu3);
    return tupleOfTensors;
}

at::Tensor &relu_(at::Tensor &self) {
    printf("Calling TCU OP:  relu \n");
    at::Tensor cpu_input = self.to(at::device(c10::kCPU));
    at::Tensor out = torch::relu(cpu_input);
    at::Tensor out_tcu = out.to(at::device(c10::kPrivateUse1));
    std::cout << "relu out " << out_tcu.device();
    return out_tcu;
}

//  torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
// {"schema": "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor", "dispatch": "False", "default": "True"}
at::Tensor max_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                      at::IntArrayRef dilation, bool ceil_mode)
{
    printf("Calling TCU OP:  max_pool2d \n");
    at::Tensor cpu_input = self.to(at::device(c10::kCPU));
    at::Tensor out = torch::max_pool2d(cpu_input, kernel_size, stride, padding, dilation, ceil_mode);

    at::Tensor out_tcu = out.to(at::device(c10::kPrivateUse1));
    std::cout << "max_pool2d out " << out_tcu.device();
    return out_tcu;
}

at::Tensor & add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out)
{
    LOG_ENTER("add_out");
    LOG_TENSOR_SIZE("add_out input",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_other = other.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::add_out(cpu_out,cpu_self,cpu_other,alpha);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("add_out tcu out",tcu_out);
    out = tcu_out;
    return tcu_out;
}
at::Tensor & mean_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim,
                      c10::optional<at::ScalarType> dtype, at::Tensor & out){
    printf("Calling TCU OP:  mean_out \n");

    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    printf("call torch cpu fun mean show ret shape next\n");
    at::Tensor ret = torch::mean_out(cpu_out,cpu_self,dim,keepdim,dtype);
    std::cout << ret.device();

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out ;
    return tcu_out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {

    m.impl("empty.memory_format", &custom_empty_memory_format);
    m.impl("empty_strided", &empty_strided);
    m.impl("fill_.Scalar", &custom_fill__scalar);
    m.impl("_copy_from", &custom__copy_from);

    m.impl("convolution_overrideable", &convolution_overrideable);
    m.impl("native_batch_norm", &native_batch_norm);
    m.impl("relu_", &relu_);
    m.impl("max_pool2d", &max_pool2d);
    m.impl("add.out", &add_out);
    m.impl("mean.out", &mean_out);
    m.impl("view", &view);
    m.impl("as_strided", &as_strided);
    m.impl("addmm.out", &addmm_out);
    m.impl("mul.out", &mul_out);
    m.impl("sub.out", &sub_out);

    m.impl("normal_", &normal_);
    m.impl("triu", &triu);
    m.impl("bmm.out", &tcu_bmm_out);
    m.impl("div.out", &div_out);
    m.impl("_softmax.out", &tcu_softmax_out);
    m.impl("arange.start_out", &arange_out);
    m.impl("pow.Scalar_out", &pow_out_scalar);
    m.impl("pow.Tensor_Scalar_out", &pow_out);
    m.impl("reciprocal.out", &reciprocal_out);
    m.impl("lt.Scalar_out", &lt_out_scalar);
    m.impl("lt.Tensor_out", &lt_out_tensor);

    m.impl("gt.Scalar_out", &gt_out);
    m.impl("where.self", &where);
    m.impl("cos.out", &cos_out);
    m.impl("sin.out", &sin_out);
    m.impl("cat.out", &cat_out);
    m.impl("index_select", &index_select);
    m.impl("rsqrt.out", &rsqrt_out);
    m.impl("zero_", &zero_);
    m.impl("mm.out", &mm_out);
    m.impl("silu.out", &silu_out);
    m.impl("sort.values_stable", &sort_out);
    m.impl("gather.out", &gather_out);
    m.impl("masked_fill_.Scalar", &masked_fill_);
    m.impl("cumsum.out", &cumsum_out);
    m.impl("le.Tensor_out", &le_out_tensor);
    m.impl("scatter.src_out", &scatter_out);
    m.impl("_log_softmax.out", &_log_softmax_out);
    m.impl("index.Tensor_out", &index_out);
    m.impl("exponential_", &exponential_);

    m.impl("argmax.out", &argmax_out);
    m.impl("uniform_", &uniform_);

}

c10::Device get_custom_device() {
    return c10::Device(c10::DeviceType::PrivateUse1, CURR_DEVICE);
}
c10::Device set_device(c10::Device device) {
    CURR_DEVICE = device.index();
    return get_custom_device();

}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("get_device", &get_custom_device, "get device");
m.def("set_device", &set_device, "set device idx");
}
