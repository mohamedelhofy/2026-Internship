#include "tensor/tensor.h"
#include <gtest/gtest.h>

TEST(AutogradCoreTest, RequiresGradInitialization) {
    Tensor t({1.0f, 2.0f}, {2});
    EXPECT_FALSE(t.requires_grad());
    EXPECT_EQ(t.grad_, nullptr);

    t.set_requires_grad(true);
    EXPECT_TRUE(t.requires_grad());
    ASSERT_NE(t.grad_, nullptr);
    EXPECT_FLOAT_EQ(t.grad_->data()[0], 0.0f);
}

TEST(AutogradCoreTest, ZeroGrad) {
    Tensor t({1.0f}, {1});
    t.set_requires_grad(true);

    // Simulate some gradient accumulation
    t.grad_->data()[0] = 5.0f;
    EXPECT_FLOAT_EQ(t.grad_->data()[0], 5.0f);

    t.zero_grad();

    EXPECT_EQ(t.grad_, nullptr);
}

TEST(AutogradCoreTest, ScalarBackwardBaseCase) {
    Tensor loss({10.0f}, {1});
    loss.set_requires_grad(true);

    // Call backward on a scalar without explicit gradient (should seed with 1.0)
    loss.backward();

    ASSERT_NE(loss.grad_, nullptr);
    EXPECT_FLOAT_EQ(loss.grad_->data()[0], 1.0f);
}

TEST(AutogradCoreTest, NonScalarBackwardThrows) {
    Tensor t({1.0f, 2.0f}, {2});
    t.set_requires_grad(true);

    // Calling backward on non-scalar without gradient should throw
    EXPECT_THROW(t.backward(), std::invalid_argument);
}
