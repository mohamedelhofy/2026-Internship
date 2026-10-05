#include "losses/cross_entropy_loss.h"
#include "tensor/tensor.h"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
constexpr float kTolerance = 1e-4f;
}

// ================= Correct Numerical Values =================

TEST(CrossEntropyLossTest, UniformLogitsGiveLogNumClasses) {
    CrossEntropyLoss criterion;

    // Equal logits -> uniform distribution -> loss = -log(1 / C) = log(C)
    Tensor logits({0.0f, 0.0f, 0.0f}, {1, 3});

    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{0});

    ASSERT_EQ(loss.size(), 1);
    EXPECT_NEAR(loss[0], std::log(3.0f), kTolerance);
}

TEST(CrossEntropyLossTest, KnownValueSingleSample) {
    CrossEntropyLoss criterion;

    // logits {1, 2, 3}: log_sum_exp = 3 + log(1 + e^-1 + e^-2) = 3.4076059
    // log_softmax = {-2.4076059, -1.4076059, -0.4076059}
    Tensor logits({1.0f, 2.0f, 3.0f}, {1, 3});

    EXPECT_NEAR(criterion(logits, CrossEntropyLoss::IndexList{2})[0], 0.4076059f, kTolerance);
    EXPECT_NEAR(criterion(logits, CrossEntropyLoss::IndexList{0})[0], 2.4076059f, kTolerance);
}

TEST(CrossEntropyLossTest, ConfidentCorrectPredictionApproachesZero) {
    CrossEntropyLoss criterion;

    Tensor logits({0.0f, 30.0f}, {1, 2});

    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{1});

    EXPECT_GE(loss[0], 0.0f);
    EXPECT_LT(loss[0], 1e-5f);
}

TEST(CrossEntropyLossTest, LossIsAlwaysNonNegative) {
    CrossEntropyLoss criterion;

    Tensor logits({-3.0f, 0.5f, 7.0f, 2.0f}, {1, 4});

    for (Tensor::size_type target = 0; target < 4; ++target) {
        EXPECT_GE(criterion(logits, CrossEntropyLoss::IndexList{target})[0], 0.0f);
    }
}

TEST(CrossEntropyLossTest, MatchesManualNegativeLogSoftmax) {
    CrossEntropyLoss criterion;

    Tensor logits({0.3f, -1.2f, 2.5f}, {1, 3});

    // Reference value computed through the tensor primitive itself.
    const Tensor log_probabilities = logits.log_softmax(-1);
    const float expected = -log_probabilities[1];

    EXPECT_NEAR(criterion(logits, CrossEntropyLoss::IndexList{1})[0], expected, kTolerance);
}

// ================= Unbatched Input =================

TEST(CrossEntropyLossTest, OneDimensionalLogitsTreatedAsSingleSample) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f}, {3});

    EXPECT_NEAR(criterion(logits, CrossEntropyLoss::IndexList{2})[0], 0.4076059f, kTolerance);
}

TEST(CrossEntropyLossTest, OneDimensionalMatchesEquivalentBatchOfOne) {
    CrossEntropyLoss criterion;

    Tensor flat({0.7f, -0.4f, 1.9f}, {3});
    Tensor batched({0.7f, -0.4f, 1.9f}, {1, 3});

    EXPECT_NEAR(criterion(flat, CrossEntropyLoss::IndexList{0})[0],
                criterion(batched, CrossEntropyLoss::IndexList{0})[0], kTolerance);
}

// ================= Batch Inputs =================

TEST(CrossEntropyLossTest, BatchMeanOverSamples) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}, {2, 3});

    // per-sample losses: 0.4076059 (class 2) and 2.4076059 (class 0)
    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{2, 0});

    EXPECT_NEAR(loss[0], 1.4076059f, kTolerance);
}

TEST(CrossEntropyLossTest, BatchRowsAreIndependent) {
    CrossEntropyLoss criterion(Reduction::None);

    // The second row is much more confident than the first.
    Tensor logits({0.0f, 0.0f, 0.0f, 10.0f}, {2, 2});

    Tensor per_sample = criterion(logits, CrossEntropyLoss::IndexList{0, 1});

    ASSERT_EQ(per_sample.size(), 2);
    EXPECT_NEAR(per_sample[0], std::log(2.0f), kTolerance);
    EXPECT_LT(per_sample[1], 1e-3f);
}

TEST(CrossEntropyLossTest, LargeBatchAllCorrect) {
    CrossEntropyLoss criterion;

    Tensor::Storage data;
    CrossEntropyLoss::IndexList targets;

    // 8 samples, 4 classes, each strongly predicting its own class.
    for (Tensor::size_type sample = 0; sample < 8; ++sample) {
        const Tensor::size_type correct = sample % 4;
        for (Tensor::size_type klass = 0; klass < 4; ++klass) {
            data.push_back(klass == correct ? 20.0f : 0.0f);
        }
        targets.push_back(correct);
    }

    Tensor logits(data, {8, 4});
    Tensor loss = criterion(logits, targets);

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_LT(loss[0], 1e-5f);
}

// ================= Reduction Modes =================

TEST(CrossEntropyLossTest, SumReduction) {
    CrossEntropyLoss criterion(Reduction::Sum);

    Tensor logits({0.0f, 0.0f, 0.0f, 0.0f}, {2, 2});

    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{0, 1});

    EXPECT_NEAR(loss[0], 2.0f * std::log(2.0f), kTolerance);
}

TEST(CrossEntropyLossTest, NoneReductionHasBatchShape) {
    CrossEntropyLoss criterion(Reduction::None);

    Tensor logits({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {3, 2});

    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{0, 1, 0});

    ASSERT_EQ(loss.ndim(), 1);
    EXPECT_EQ(loss.shape()[0], 3);
}

// ================= Tensor-Valued Targets =================

TEST(CrossEntropyLossTest, TensorTargetsMatchIndexTargets) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor targets({2.0f, 0.0f}, {2});

    EXPECT_NEAR(criterion(logits, targets)[0],
                criterion(logits, CrossEntropyLoss::IndexList{2, 0})[0], kTolerance);
}

TEST(CrossEntropyLossTest, FractionalTensorTargetThrows) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f}, {1, 2});
    Tensor targets({0.5f}, {1});

    EXPECT_THROW(static_cast<void>(criterion(logits, targets)), std::invalid_argument);
}

TEST(CrossEntropyLossTest, NegativeTensorTargetThrows) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f}, {1, 2});
    Tensor targets({-1.0f}, {1});

    EXPECT_THROW(static_cast<void>(criterion(logits, targets)), std::invalid_argument);
}

TEST(CrossEntropyLossTest, MultiDimensionalTargetTensorThrows) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor targets({0.0f, 1.0f}, {2, 1});

    EXPECT_THROW(static_cast<void>(criterion(logits, targets)), std::invalid_argument);
}

// ================= Invalid Input =================

TEST(CrossEntropyLossTest, ClassIndexOutOfRangeThrows) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f}, {1, 3});

    EXPECT_THROW(static_cast<void>(criterion(logits, CrossEntropyLoss::IndexList{3})),
                 std::out_of_range);
}

TEST(CrossEntropyLossTest, ClassIndexOutOfRangeInTensorTargetThrows) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f}, {1, 3});
    Tensor targets({7.0f}, {1});

    EXPECT_THROW(static_cast<void>(criterion(logits, targets)), std::out_of_range);
}

TEST(CrossEntropyLossTest, TargetCountMismatchThrows) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    EXPECT_THROW(static_cast<void>(criterion(logits, CrossEntropyLoss::IndexList{0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(criterion(logits, CrossEntropyLoss::IndexList{0, 1, 0})),
                 std::invalid_argument);
}

TEST(CrossEntropyLossTest, ThreeDimensionalLogitsThrow) {
    CrossEntropyLoss criterion;

    Tensor logits({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2});

    EXPECT_THROW(static_cast<void>(criterion(logits, CrossEntropyLoss::IndexList{0})),
                 std::invalid_argument);
}

TEST(CrossEntropyLossTest, EmptyLogitsThrow) {
    CrossEntropyLoss criterion;

    Tensor logits({}, Tensor::Shape{0});

    EXPECT_THROW(static_cast<void>(criterion(logits, CrossEntropyLoss::IndexList{})),
                 std::invalid_argument);
}

// ================= Numerical Stability =================

TEST(CrossEntropyLossTest, VeryLargeLogitsStayFinite) {
    CrossEntropyLoss criterion;

    // exp(1000) overflows to +inf, so a naive log(softmax(x)) would yield NaN.
    Tensor logits({1000.0f, 1000.0f, 1000.0f}, {1, 3});

    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{0});

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_NEAR(loss[0], std::log(3.0f), kTolerance);
}

TEST(CrossEntropyLossTest, LargeLogitGapStaysExact) {
    CrossEntropyLoss criterion;

    Tensor logits({0.0f, 1000.0f}, {1, 2});

    // Predicting the dominant class costs essentially nothing...
    EXPECT_NEAR(criterion(logits, CrossEntropyLoss::IndexList{1})[0], 0.0f, kTolerance);

    // ...while predicting the other one costs the full logit gap, finitely.
    const float wrong = criterion(logits, CrossEntropyLoss::IndexList{0})[0];
    EXPECT_TRUE(std::isfinite(wrong));
    EXPECT_NEAR(wrong, 1000.0f, 0.1f);
}

TEST(CrossEntropyLossTest, VeryNegativeLogitsStayFinite) {
    CrossEntropyLoss criterion;

    // A naive softmax would underflow every entry to 0 and then take log(0).
    Tensor logits({-1000.0f, -1000.0f}, {1, 2});

    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{0});

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_NEAR(loss[0], std::log(2.0f), kTolerance);
}

TEST(CrossEntropyLossTest, ShiftInvariance) {
    CrossEntropyLoss criterion;

    // Softmax is invariant to a constant shift, so the loss must be too.
    Tensor logits({1.0f, 2.0f, 3.0f}, {1, 3});
    Tensor shifted({501.0f, 502.0f, 503.0f}, {1, 3});

    EXPECT_NEAR(criterion(logits, CrossEntropyLoss::IndexList{1})[0],
                criterion(shifted, CrossEntropyLoss::IndexList{1})[0], kTolerance);
}

// ================= Regression =================

TEST(CrossEntropyLossTest, RegressionKnownBatchValue) {
    CrossEntropyLoss criterion;

    Tensor logits({2.0f, 1.0f, 0.1f, 0.5f, 2.5f, 0.3f}, {2, 3});

    // sample 0, class 0: log_sum_exp = 2 + log(1 + e^-1 + e^-1.9) = 2.4170300
    //                    loss = 0.4170300
    // sample 1, class 1: log_sum_exp = 2.5 + log(e^-2 + 1 + e^-2.2) = 2.7200495
    //                    loss = 0.2200495
    // mean = 0.3185398
    Tensor loss = criterion(logits, CrossEntropyLoss::IndexList{0, 1});

    EXPECT_NEAR(loss[0], 0.3185398f, kTolerance);
}