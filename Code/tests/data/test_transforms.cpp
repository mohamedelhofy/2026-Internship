#include "data/transforms.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>

TEST(TransformsTest, RandomHorizontalFlipShapeAndValues) {
    // 1 channel, 2x2 image
    Tensor img({1, 2, 3, 4}, {1, 2, 2});

    // Force flip by setting probability to 1.0
    RandomHorizontalFlip flip(1.0f);
    Tensor flipped = flip(img);

    ASSERT_EQ(flipped.shape().size(), 3);
    EXPECT_EQ(flipped.shape()[1], 2);
    EXPECT_EQ(flipped.shape()[2], 2);

    // Row 1: [1, 2] -> [2, 1]
    EXPECT_FLOAT_EQ(flipped.data()[0], 2);
    EXPECT_FLOAT_EQ(flipped.data()[1], 1);
    // Row 2: [3, 4] -> [4, 3]
    EXPECT_FLOAT_EQ(flipped.data()[2], 4);
    EXPECT_FLOAT_EQ(flipped.data()[3], 3);
}

TEST(TransformsTest, RandomCropShape) {
    // 3 channels, 32x32 image
    Tensor img({3, 32, 32});

    // Crop to 28x28 with 4 padding
    RandomCrop crop({28, 28}, 4);
    Tensor cropped = crop(img);

    ASSERT_EQ(cropped.shape().size(), 3);
    EXPECT_EQ(cropped.shape()[0], 3);
    EXPECT_EQ(cropped.shape()[1], 28);
    EXPECT_EQ(cropped.shape()[2], 28);
}
