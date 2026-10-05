#include "tensor.h"
#include <gtest/gtest.h>

TEST(TensorMatMulTest, SquareMatrix) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    Tensor b({5, 6, 7, 8}, {2, 2});

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_EQ(c.size(), 4);

    // EXPECT_FLOAT_EQ(c(0, 0), 19.0f);
    // EXPECT_FLOAT_EQ(c(0, 1), 22.0f);
    // EXPECT_FLOAT_EQ(c(1, 0), 43.0f);
    // EXPECT_FLOAT_EQ(c(1, 1), 50.0f);
    EXPECT_FLOAT_EQ(c.data()[0], 19.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 22.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 43.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 50.0f);
}

TEST(TensorMatMulTest, RectangularMatrix) {
    Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor b({7, 8, 9, 10, 11, 12}, {3, 2});

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    // EXPECT_FLOAT_EQ(c(0, 0), 58.0f);
    // EXPECT_FLOAT_EQ(c(0, 1), 64.0f);
    // EXPECT_FLOAT_EQ(c(1, 0), 139.0f);
    // EXPECT_FLOAT_EQ(c(1, 1), 154.0f);
    EXPECT_FLOAT_EQ(c.data()[0], 58.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 64.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 139.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 154.0f);
}

TEST(TensorMatMulTest, IdentityMatrix) {
    Tensor identity({1, 0, 0, 1}, {2, 2});

    Tensor a({5, 6, 7, 8}, {2, 2});

    Tensor c = identity.matmul(a);

    EXPECT_FLOAT_EQ(c.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 6.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 7.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 8.0f);
}

TEST(TensorMatMulTest, IncompatibleShapes) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    Tensor b({1, 2, 3, 4, 5, 6}, {3, 2});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST(TensorMatMulTest, OneDimensionalTensor) {
    Tensor a({1, 2, 3}, {3});

    Tensor b({4, 5, 6}, {3});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST(TensorMatMulTest, OutOfBoundsAccess) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    // EXPECT_THROW(a(2, 0), std::out_of_range);

    // EXPECT_THROW(a(0, 2), std::out_of_range);
    EXPECT_ANY_THROW(void(a(Tensor::Shape{2, 0})));
    EXPECT_ANY_THROW(void(a(Tensor::Shape{0, 2})));
}
// add tests for ND tensors
TEST(TensorMatMulTest, NDimensionalTensor) {
    Tensor a({1, 2, 3, 4, 5, 6}, {1, 2, 3});

    Tensor b({7, 8, 9, 10, 11, 12}, {1, 3, 2});

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape()[0], 1);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_EQ(c.shape()[2], 2);

    EXPECT_FLOAT_EQ(c.data()[0], 58.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 64.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 139.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 154.0f);
}

TEST(TensorMatMulTest, BatchMatrixMultiplication) {
    Tensor a({1, 2, 3, 4, 5, 6}, {2, 1, 3});    // (2,1,3)
    Tensor b({7, 8, 9, 10, 11, 12}, {2, 3, 1}); // (2,3,1)

    Tensor c = a.matmul(b);

    ASSERT_EQ(c.shape().size(), 3);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 1);
    EXPECT_EQ(c.shape()[2], 1);

    // Batch 0: [1 2 3] x [[7],[8],[9]] = 50
    EXPECT_FLOAT_EQ(c.data()[0], 50.0f);

    // Batch 1: [4 5 6] x [[10],[11],[12]] = 167
    EXPECT_FLOAT_EQ(c.data()[1], 167.0f);
}

TEST(TensorMatMulTest, FourDimensionalTensor) {
    Tensor a(
        {// Batch (0,0)
         1, 2, 3, 4, 5, 6,

         // Batch (0,1)
         7, 8, 9, 10, 11, 12,

         // Batch (1,0)
         13, 14, 15, 16, 17, 18,

         // Batch (1,1)
         19, 20, 21, 22, 23, 24},
        {2, 2, 2, 3});

    Tensor b(
        {// Batch (0,0)
         1, 2, 3, 4, 5, 6,

         // Batch (0,1)
         7, 8, 9, 10, 11, 12,

         // Batch (1,0)
         13, 14, 15, 16, 17, 18,

         // Batch (1,1)
         19, 20, 21, 22, 23, 24},
        {2, 2, 3, 2});

    Tensor c = a.matmul(b);

    ASSERT_EQ(c.shape().size(), 4);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_EQ(c.shape()[2], 2);
    EXPECT_EQ(c.shape()[3], 2);

    std::vector<float> expected = {// Batch (0,0)
                                   22, 28, 49, 64,

                                   // Batch (0,1)
                                   220, 244, 301, 334,

                                   // Batch (1,0)
                                   634, 676, 769, 820,

                                   // Batch (1,1)
                                   1264, 1324, 1453, 1522};

    ASSERT_EQ(c.data().size(), expected.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], expected[i]);
    }
}
