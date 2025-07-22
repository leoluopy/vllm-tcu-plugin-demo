
#include "iostream"
#include "op_implements.h"

int test_softmax() {
    torch::manual_seed(42);

    // 测试 2D 张量（dim=1）
    {
        auto self = torch::randn({3, 4});
        int64_t dim = 1;
        auto expected = torch::softmax(self, dim);
        auto out = torch::empty_like(expected);

        tcu_softmax_out(self, dim, false, out);

        TORCH_CHECK(out.sizes() == expected.sizes(), "2D shape mismatch");
        TORCH_CHECK(torch::allclose(out, expected, 1e-5, 1e-8), "2D value mismatch");
    }

    // 测试 3D 张量（dim=-1）
    {
        auto self = torch::randn({2, 3, 4});
        int64_t dim = -1;
        auto expected = torch::softmax(self, dim);
        auto out = torch::empty_like(expected);

        tcu_softmax_out(self, dim, false, out);

        TORCH_CHECK(out.sizes() == expected.sizes(), "3D shape mismatch");
        TORCH_CHECK(torch::allclose(out, expected, 1e-5, 1e-8), "3D value mismatch");
    }

    // 测试 half_to_float 转换
//    {
//        auto self = torch::randn({3, 4}, torch::kHalf);
//        int64_t dim = 1;
//        auto expected = torch::softmax(self.to(torch::kFloat), dim); // 转换为 float 计算
//        auto out = torch::empty_like(expected);
//
//        tcu_softmax_out(self, dim, true, out);
//
//        TORCH_CHECK(out.sizes() == expected.sizes(), "half_to_float shape mismatch");
//        TORCH_CHECK(torch::allclose(out, expected, 1e-5, 1e-8), "half_to_float value mismatch");
//    }

    return 0; // 返回 0 表示测试通过
}

int test_matmul() {
    torch::manual_seed(42);
    const int64_t batch_size = 3;
    auto self = torch::randn({batch_size, 2, 4});
    auto mat2 = torch::randn({batch_size, 4, 5});
    auto expected = torch::bmm(self, mat2);
    auto out = torch::empty_like(expected);

    tcu_bmm_out(self, mat2, out);

    TORCH_CHECK(out.sizes() == expected.sizes(), "Shape mismatch");
    TORCH_CHECK(torch::allclose(out, expected, 1e-5, 1e-8), "Value mismatch");
}

int main() {

    test_matmul();
    test_softmax();
    printf("ALL PASSED\n");
}

