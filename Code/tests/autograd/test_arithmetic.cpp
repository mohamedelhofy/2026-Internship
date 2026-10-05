#include "tensor/tensor.h"
#include <gtest/gtest.h>

TEST(AutogradArithmeticTest, ComplexExpression) {
    // z = (x + y) * (x - y) / x
    Tensor x({4.0f}, {1});
    Tensor y({2.0f}, {1});
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    Tensor z = (x + y) * (x - y) / x;
    z.backward();

    // z = 6 * 2 / 4 = 3
    EXPECT_FLOAT_EQ(z.data()[0], 3.0f);

    // Math: dz/dx = (2x*x - (x^2-y^2))/x^2 = (32 - 12)/16 = 1.25
    ASSERT_NE(x.grad_, nullptr);
    EXPECT_FLOAT_EQ(x.grad_->data()[0], 1.25f);

    // Math: dz/dy = -2y/x = -4/4 = -1
    ASSERT_NE(y.grad_, nullptr);
    EXPECT_FLOAT_EQ(y.grad_->data()[0], -1.0f);
}
