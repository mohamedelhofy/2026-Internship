#include "tensor.h"
#include <gtest/gtest.h>

TEST(TensorTest, DefaultConstructor) {
    Tensor tensor;

    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.size(), 0);
    EXPECT_EQ(tensor.ndim(), 0);
    EXPECT_TRUE(tensor.shape().empty());
    EXPECT_TRUE(tensor.data().empty());
}

TEST(TensorTest, Constructor) {
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    EXPECT_FALSE(tensor.empty());

    EXPECT_EQ(tensor.size(), 4);
    EXPECT_EQ(tensor.ndim(), 2);

    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 2);

    EXPECT_EQ(tensor.data()[0], 1.0f);
    EXPECT_EQ(tensor.data()[1], 2.0f);
    EXPECT_EQ(tensor.data()[2], 3.0f);
    EXPECT_EQ(tensor.data()[3], 4.0f);

    EXPECT_EQ(tensor.dtype(), "float");
}

TEST(TensorTest, ElementAccess) {
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    EXPECT_FLOAT_EQ(tensor({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(tensor({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(tensor({1, 1}), 4.0f);

    // Modify an element
    tensor({1, 1}) = 10.0f;
    EXPECT_FLOAT_EQ(tensor({1, 1}), 10.0f);
}

TEST(TensorTest, NDimensionalElementAccess) {
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});

    EXPECT_FLOAT_EQ(tensor({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor({0, 0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(tensor({0, 1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(tensor({0, 1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(tensor({1, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(tensor({1, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(tensor({1, 1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(tensor({1, 1, 1}), 8.0f);
}

TEST(TensorTest, OutOfBoundsAccess) {
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_THROW(void(tensor({2, 0})), std::out_of_range);
    EXPECT_THROW(void(tensor({0, 2})), std::out_of_range);
    EXPECT_THROW(void(tensor({0, 0, 0})), std::invalid_argument);
}
TEST(TensorTest, OutputOperator) {
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    std::ostringstream oss;
    oss << tensor;

    std::string expected_output = "Tensor([\n"
                                  "  [   1.0000,    2.0000],\n"
                                  "  [   3.0000,    4.0000]\n"
                                  "])";

    EXPECT_EQ(oss.str(), expected_output);
}

TEST(TensorTest, EqualityOperator) {
    Tensor tensor1({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor tensor2({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor tensor3({1.0f, 2.0f, 3.0f, 5.0f}, {2, 2});

    EXPECT_TRUE(tensor1 == tensor2);
    EXPECT_FALSE(tensor1 == tensor3);
}
