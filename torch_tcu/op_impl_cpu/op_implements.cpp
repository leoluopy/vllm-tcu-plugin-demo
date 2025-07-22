#include "op_implements.h"



///////////////////
// compute OP
///////////////////
at::Tensor &tcu_softmax_out(const at::Tensor &self, int64_t dim, bool half_to_float, at::Tensor &out) {

    LOG_ENTER("tcu_softmax_out");
    LOG_TENSOR_SIZE("tcu_softmax_out input",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    auto out_cpu  = torch::softmax(cpu_self,dim);
    at::Tensor tcu_out = out_cpu.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("tcu_softmax_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}

at::Tensor &tcu_bmm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out) {
    LOG_ENTER("bmm_out");
    LOG_TENSOR_SIZE("bmm_out input",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_mat2 = mat2.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));
    cpu_out = torch::matmul(cpu_self,cpu_mat2);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("bmm_out tcu out",tcu_out);
    out = tcu_out;
    return tcu_out;
}
at::Tensor & div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    LOG_ENTER("div_out");
    LOG_TENSOR_SIZE("div_out input",self);
    LOG_TENSOR_SIZE("div_out other in",other);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_other = other.to(at::device(c10::kCPU));
    auto out_cpu  = torch::div(cpu_self,cpu_other);
    at::Tensor tcu_out = out_cpu.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("div_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2,
                       const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out)
{
    LOG_ENTER("addmm_out");
    LOG_TENSOR_SIZE("addmm_out input",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_mat1 = mat1.to(at::device(c10::kCPU));
    at::Tensor cpu_mat2 = mat2.to(at::device(c10::kCPU));
    at::Tensor cpu_out = torch::addmm(cpu_self,cpu_mat1,cpu_mat2,beta,alpha);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("addmm_out tcu out",tcu_out);
    out = tcu_out;
    return tcu_out;
}
at::Tensor & arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out)
{
    LOG_ENTER("arange");
    at::Tensor cpu_out = torch::arange(start,end,step);
    LOG_TENSOR_SIZE("arange cpu_out",cpu_out);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("arange tcu out",out);
    return out;
}
at::Tensor & pow_out_scalar(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out){
    LOG_ENTER("pow_out_scalar");
    at::Tensor cpu_exp = exponent.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::pow_out(cpu_out,self,cpu_exp);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("pow_out_scalar tcu out",out);
    return out;
}
at::Tensor & pow_out(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out)
{
    LOG_ENTER("aten::pow.Tensor_Scalar_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::pow_out(cpu_out,cpu_self,exponent);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("aten::pow.Tensor_Scalar_out tcu out",out);
    return out;
}
at::Tensor & reciprocal_out(const at::Tensor & self, at::Tensor & out){
    LOG_ENTER("reciprocal_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::reciprocal_out(cpu_out,cpu_self);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("reciprocal_out tcu out",out);
    return out;
}
at::Tensor & mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    LOG_ENTER("mul_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_other = other.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::mul_out(cpu_out,cpu_self,cpu_other);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("mul_out tcu out",out);
    return out;
}
at::Tensor & sub_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    LOG_ENTER("sub_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_other = other.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::sub_out(cpu_out,cpu_self,cpu_other,alpha);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("sub_out tcu out",out);
    return out;
}
at::Tensor & gt_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    LOG_ENTER("gt_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::gt_out(cpu_out,cpu_self,other);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("gt_out tcu out",out);
    return out;
}
at::Tensor where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other){
    LOG_ENTER("where");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_condition = condition.to(at::device(c10::kCPU));
    at::Tensor cpu_other = other.to(at::device(c10::kCPU));

    at::Tensor cpu_out = torch::where(cpu_condition,cpu_self,cpu_other);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("where tcu out",tcu_out);
    return tcu_out;
}
at::Tensor & cos_out(const at::Tensor & self, at::Tensor & out){
    LOG_ENTER("cos_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::cos_out(cpu_out,cpu_self);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("cos_out tcu out",out);
    return out;
}
at::Tensor & sin_out(const at::Tensor & self, at::Tensor & out){
    LOG_ENTER("sin_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::sin_out(cpu_out,cpu_self);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("sin_out tcu out",out);
    return out;
}
at::Tensor & cat_out(const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out){
    LOG_ENTER("cat_out");
    std::vector<at::Tensor> holder;
    for (const auto& t : tensors) {
        holder.emplace_back(t.cpu());
    }
    at::ITensorListRef cpu_tensors =holder;

    at::Tensor cpu_out = out.to(at::device(c10::kCPU));
    torch::cat_out(cpu_out,cpu_tensors,dim);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("cat_out tcu out",out);
    return out;
}
at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index){
    LOG_ENTER("index_select");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_index = index.to(at::device(c10::kCPU));

    at::Tensor cpu_out = torch::index_select(cpu_self,dim,cpu_index);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("index_select tcu out",tcu_out);
    return tcu_out;
}
at::Tensor & rsqrt_out(const at::Tensor & self, at::Tensor & out)
{
    LOG_ENTER("rsqrt_out");
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));

    at::Tensor cpu_out = torch::rsqrt(cpu_self);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("rsqrt_out tcu out",tcu_out);
    out = tcu_out;
    return tcu_out;
}
at::Tensor & zero_(at::Tensor & self)
{
    LOG_ENTER("zero_");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));

    at::Tensor cpu_out = torch::zero_(cpu_self);
    self = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("zero_ tcu out",self);
    return self;
}
at::Tensor & mm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out)
{
    LOG_ENTER("mm_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_mat2 = mat2.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::mm_out(cpu_out,cpu_self,cpu_mat2);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("mm_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & silu_out(const at::Tensor & self, at::Tensor & out)
{
    LOG_ENTER("silu_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::silu_out(cpu_out,cpu_self);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("silu_out tcu out",tcu_out);
    out = tcu_out;
    return tcu_out;
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending,
    at::Tensor & values, at::Tensor & indices)
{
    LOG_ENTER("sort_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_values = values.to(at::device(c10::kCPU));
    at::Tensor cpu_indices = indices.to(at::device(c10::kCPU));

    auto tutple_cpu = torch::sort_out(cpu_values,cpu_indices,cpu_self,stable,dim,descending);

    at::Tensor tcu_values = cpu_values.to(at::device(c10::kPrivateUse1));
    at::Tensor tcu_indices = cpu_indices.to(at::device(c10::kPrivateUse1));

    values = tcu_values ;
    indices = tcu_indices ;
    ::std::tuple<at::Tensor &,at::Tensor &> tuple_tcu(values,indices);
    return tuple_tcu;
}
at::Tensor & gather_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out)
{
    LOG_ENTER("gather_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_index = index.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::gather_out(cpu_out,cpu_self,dim,cpu_index,sparse_grad);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("gather_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & lt_out_scalar(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    LOG_ENTER("lt_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    torch::lt_out(cpu_out,cpu_self,other);
    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    out = tcu_out;
    LOG_TENSOR_SIZE("lt_out tcu out",out);
    return out;
}
at::Tensor & lt_out_tensor(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    LOG_ENTER("lt_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_other = other.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::lt_out(cpu_out,cpu_self,cpu_other);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("lt_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value)
{
    LOG_ENTER("lt_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_mask = mask.to(at::device(c10::kCPU));

    torch::masked_fill(cpu_self,cpu_mask,value);

    at::Tensor self_tcu = cpu_self.to(at::device(c10::kPrivateUse1));
    self = self_tcu;
    LOG_TENSOR_SIZE("lt_out tcu out",self);
    return self;
}
at::Tensor & cumsum_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out)
{
    LOG_ENTER("cumsum_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::cumsum_out(cpu_out,cpu_self,dim,dtype);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("cumsum_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & le_out_tensor(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    LOG_ENTER("le_out_tensor");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_other = other.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::le_out(cpu_out,cpu_self,cpu_other);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("le_out_tensor tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & scatter_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out)
{
    LOG_ENTER("scatter_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_src = src.to(at::device(c10::kCPU));
    at::Tensor cpu_index = index.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::scatter_out(cpu_out,cpu_self,dim,cpu_index,cpu_src);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("scatter_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & _log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out)
{
    LOG_ENTER("_log_softmax_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::log_softmax_out(cpu_out,cpu_self,dim);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("_log_softmax_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & index_out(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, at::Tensor & out)
{
    LOG_ENTER("index_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));
    c10::List<c10::optional<at::Tensor>> cpu_indices;
    for (auto indi : indices)
    {
        at::Tensor cpu_tensor = indi.get().toTensor().to(at::device(c10::kCPU));
        cpu_indices.emplace_back(cpu_tensor);
    }

    cpu_out = torch::index_out(cpu_out,cpu_self,cpu_indices);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("index_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & exponential_(at::Tensor & self, double lambd, c10::optional<at::Generator> generator)
{
    LOG_ENTER("exponential_");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));

    at::Tensor cpu_out = torch::exponential(cpu_self,lambd,generator);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("exponential_ tcu out",tcu_out);
    self = tcu_out;
    return self;
}
at::Tensor & argmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out)
{
    LOG_ENTER("argmax_out");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));
    at::Tensor cpu_out = out.to(at::device(c10::kCPU));

    cpu_out = torch::argmax_out(cpu_out,cpu_self,dim,keepdim);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("argmax_out tcu out",tcu_out);
    out = tcu_out;
    return out;
}
at::Tensor & uniform_(at::Tensor & self, double from, double to, c10::optional<at::Generator> generator)
{
    LOG_ENTER("uniform_");
    LOG_TENSOR_SIZE("self: ",self);
    at::Tensor cpu_self = self.to(at::device(c10::kCPU));

    at::Tensor cpu_out = torch::uniform(cpu_self,from,to,generator);

    at::Tensor tcu_out = cpu_out.to(at::device(c10::kPrivateUse1));
    LOG_TENSOR_SIZE("uniform_ tcu out",tcu_out);
    self = tcu_out;
    return self;
}
