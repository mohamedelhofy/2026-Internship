#include "data/dataloader.h"
#include "data/tensor_dataset.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>

TEST(DataLoaderTest, BatchingLogic) {
    Tensor features({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 2}); // 5 samples
    Tensor labels({0, 1, 0, 1, 0}, {5});                      // 5 samples

    TensorDataset dataset({features, labels});

    // Batch size 2, should produce 3 batches (sizes: 2, 2, 1)
    DataLoader loader(dataset, 2, false);

    EXPECT_EQ(loader.size(), 3);

    auto it = loader.begin();

    // Batch 1
    auto batch1 = *it;
    EXPECT_EQ(batch1[0].shape()[0], 2);
    ++it;

    // Batch 2
    auto batch2 = *it;
    EXPECT_EQ(batch2[0].shape()[0], 2);
    ++it;

    // Batch 3 (Remaining 1 sample)
    auto batch3 = *it;
    EXPECT_EQ(batch3[0].shape()[0], 1);
}
