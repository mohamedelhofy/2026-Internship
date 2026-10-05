#include "tensor/tensor.h"
#include <gtest/gtest.h>

TEST(AutogradMatrixTest, MatmulGradient) {
    // X = [1, 2] (Shape: 1x2)
    Tensor X({1.0f, 2.0f}, {1, 2});

    // W = [[3, 4], [5, 6]] (Shape: 2x2)
    Tensor W({3.0f, 4.0f, 5.0f, 6.0f}, {2, 2});

    X.set_requires_grad(true);
    W.set_requires_grad(true);

    Tensor Z = X.matmul(W); // Z = [13, 16] (Shape: 1x2)

    // Provide an upstream gradient of [1, 1]
    Tensor grad_out({1.0f, 1.0f}, {1, 2});
    Z.backward(grad_out);

    // Check X.grad = grad_out @ W.T = [1, 1] @ [[3, 5], [4, 6]] = [7, 11]
    ASSERT_NE(X.grad_, nullptr);
    EXPECT_FLOAT_EQ(X.grad_->data()[0], 7.0f);
    EXPECT_FLOAT_EQ(X.grad_->data()[1], 11.0f);

    // Check W.grad = X.T @ grad_out = [[1], [2]] @ [1, 1] = [[1, 1], [2, 2]]
    ASSERT_NE(W.grad_, nullptr);
    EXPECT_FLOAT_EQ(W.grad_->data()[0], 1.0f);
    EXPECT_FLOAT_EQ(W.grad_->data()[1], 1.0f);
    EXPECT_FLOAT_EQ(W.grad_->data()[2], 2.0f);
    EXPECT_FLOAT_EQ(W.grad_->data()[3], 2.0f);
}
