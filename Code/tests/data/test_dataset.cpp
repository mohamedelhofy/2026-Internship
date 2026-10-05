#include "data/tensor_dataset.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(TensorDatasetTest, InitializationAndSize) {
    Tensor features({1, 2, 3, 4}, {2, 2}); // 2 samples, 2 features
    Tensor labels({0, 1}, {2});            // 2 samples

    TensorDataset dataset({features, labels});

    EXPECT_EQ(dataset.size(), 2);
}

TEST(TensorDatasetTest, MismatchedDimensionsThrows) {
    Tensor features({1, 2, 3, 4}, {2, 2}); // 2 samples
    Tensor labels({0, 1, 2}, {3});         // 3 samples (MISMATCH)

    EXPECT_THROW(TensorDataset({features, labels}), std::invalid_argument);
}

TEST(TensorDatasetTest, GetItem) {
    Tensor features({10, 20, 30, 40}, {2, 2});
    Tensor labels({0, 1}, {2});

    TensorDataset dataset({features, labels});

    auto sample_0 = dataset.get_item(0);
    ASSERT_EQ(sample_0.size(), 2);

    // Check first sample features
    EXPECT_EQ(sample_0[0].shape().size(), 1);
    EXPECT_EQ(sample_0[0].shape()[0], 2);
    EXPECT_FLOAT_EQ(sample_0[0].data()[0], 10);
    EXPECT_FLOAT_EQ(sample_0[0].data()[1], 20);

    // Check first sample label
    EXPECT_FLOAT_EQ(sample_0[1].data()[0], 0);
}
