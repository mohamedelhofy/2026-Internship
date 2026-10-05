#include "losses/cross_entropy_loss.h"
#include "losses/mse_loss.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>

TEST(AutogradIntegrationTest, ReLUGradient) {
    // x = [-2.0, 1.0, 3.0]
    Tensor x({-2.0f, 1.0f, 3.0f}, {3});
    x.set_requires_grad(true);

    Tensor y = x.relu(); // y = [0.0, 1.0, 3.0]

    // Default upstream gradient = [1, 1, 1]
    Tensor grad_out({1.0f, 1.0f, 1.0f}, {3});
    y.backward(grad_out);

    // dx = [0.0, 1.0, 1.0] (Derivative of ReLU)
    ASSERT_NE(x.grad_, nullptr);
    EXPECT_FLOAT_EQ(x.grad_->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(x.grad_->data()[1], 1.0f);
    EXPECT_FLOAT_EQ(x.grad_->data()[2], 1.0f);
}

TEST(AutogradIntegrationTest, SigmoidGradient) {
    Tensor x({0.0f}, {1}); // sigmoid(0) = 0.5
    x.set_requires_grad(true);

    Tensor y = x.sigmoid();
    y.backward(); // Scalar backward seeds with 1.0

    // ds/dx = s * (1 - s) = 0.5 * 0.5 = 0.25
    ASSERT_NE(x.grad_, nullptr);
    EXPECT_FLOAT_EQ(x.grad_->data()[0], 0.25f);
}

TEST(AutogradIntegrationTest, MSELossGradientMean) {
    Tensor pred({1.0f, 2.0f}, {2});
    Tensor target({1.0f, 3.0f}, {2});
    pred.set_requires_grad(true);

    MSELoss criterion(Reduction::Mean);
    Tensor loss = criterion.forward(pred, target);
    loss.backward();

    // loss = mean((pred - target)^2) = mean([0, 1]) = 0.5
    // grad_pred = 2 * (pred - target) / N = 2 * [0, -1] / 2 = [0, -1]
    ASSERT_NE(pred.grad_, nullptr);
    EXPECT_FLOAT_EQ(pred.grad_->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(pred.grad_->data()[1], -1.0f);
}
