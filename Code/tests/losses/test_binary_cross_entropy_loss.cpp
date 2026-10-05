#include "losses/binary_cross_entropy_loss.h"
#include "tensor/tensor.h"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
constexpr float kTolerance = 1e-5f;
}

// ================= Correct Numerical Values =================

TEST(BinaryCrossEntropyLossTest, MaximumUncertaintyGivesLogTwo) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({0.5f}, {1});
    Tensor targets({1.0f}, {1});

    Tensor loss = criterion(probabilities, targets);

    ASSERT_EQ(loss.size(), 1);
    EXPECT_NEAR(loss[0], std::log(2.0f), kTolerance);
}

TEST(BinaryCrossEntropyLossTest, KnownValueConfidentAndCorrect) {
    BinaryCrossEntropyLoss criterion;

    // p = 0.9 with y = 1, and p = 0.1 with y = 0: both cost -log(0.9)
    Tensor probabilities({0.9f, 0.1f}, {2});
    Tensor targets({1.0f, 0.0f}, {2});

    EXPECT_NEAR(criterion(probabilities, targets)[0], -std::log(0.9f), kTolerance);
}

TEST(BinaryCrossEntropyLossTest, ConfidentAndWrongCostsMore) {
    BinaryCrossEntropyLoss criterion;

    Tensor targets({1.0f}, {1});

    Tensor good({0.9f}, {1});
    Tensor bad({0.1f}, {1});

    EXPECT_LT(criterion(good, targets)[0], criterion(bad, targets)[0]);
}

TEST(BinaryCrossEntropyLossTest, LabelSymmetry) {
    BinaryCrossEntropyLoss criterion;

    // BCE(p, 1) must equal BCE(1 - p, 0)
    Tensor probabilities({0.73f}, {1});
    Tensor complement({0.27f}, {1});

    Tensor positive({1.0f}, {1});
    Tensor negative({0.0f}, {1});

    EXPECT_NEAR(criterion(probabilities, positive)[0], criterion(complement, negative)[0], 1e-4f);
}

TEST(BinaryCrossEntropyLossTest, LossIsAlwaysNonNegative) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({0.01f, 0.25f, 0.5f, 0.75f, 0.99f}, {5});
    Tensor targets({0.0f, 1.0f, 0.0f, 1.0f, 0.0f}, {5});

    EXPECT_GE(criterion(probabilities, targets)[0], 0.0f);
}

TEST(BinaryCrossEntropyLossTest, SoftTargetsAreAccepted) {
    BinaryCrossEntropyLoss criterion;

    // Targets anywhere in [0, 1] are valid, matching PyTorch's BCELoss.
    Tensor probabilities({0.5f}, {1});
    Tensor targets({0.3f}, {1});

    Tensor loss = criterion(probabilities, targets);

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_NEAR(loss[0], std::log(2.0f), kTolerance);
}

// ================= Batch Inputs =================

TEST(BinaryCrossEntropyLossTest, BatchAveragesOverAllElements) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({0.5f, 0.5f, 0.5f, 0.5f}, {2, 2});
    Tensor targets({1.0f, 0.0f, 1.0f, 0.0f}, {2, 2});

    EXPECT_NEAR(criterion(probabilities, targets)[0], std::log(2.0f), kTolerance);
}

TEST(BinaryCrossEntropyLossTest, MultiLabelShapeIsSupported) {
    BinaryCrossEntropyLoss criterion(Reduction::None);

    Tensor probabilities({0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}, {2, 3});
    Tensor targets({1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}, {2, 3});

    Tensor loss = criterion(probabilities, targets);

    ASSERT_EQ(loss.ndim(), 2);
    EXPECT_EQ(loss.shape()[0], 2);
    EXPECT_EQ(loss.shape()[1], 3);

    for (Tensor::size_type i = 0; i < loss.size(); ++i) {
        EXPECT_NEAR(loss[i], std::log(2.0f), kTolerance);
    }
}

// ================= Reduction Modes =================

TEST(BinaryCrossEntropyLossTest, SumReduction) {
    BinaryCrossEntropyLoss criterion(Reduction::Sum);

    Tensor probabilities({0.5f, 0.5f}, {2});
    Tensor targets({1.0f, 0.0f}, {2});

    EXPECT_NEAR(criterion(probabilities, targets)[0], 2.0f * std::log(2.0f), kTolerance);
}

TEST(BinaryCrossEntropyLossTest, NoneReductionKeepsShape) {
    BinaryCrossEntropyLoss criterion(Reduction::None);

    Tensor probabilities({0.5f, 0.25f}, {2});
    Tensor targets({1.0f, 1.0f}, {2});

    Tensor loss = criterion(probabilities, targets);

    ASSERT_EQ(loss.size(), 2);
    EXPECT_NEAR(loss[0], -std::log(0.5f), kTolerance);
    EXPECT_NEAR(loss[1], -std::log(0.25f), kTolerance);
}

// ================= Clipping and Numerical Stability =================

TEST(BinaryCrossEntropyLossTest, PerfectPredictionIsFiniteAndNearZero) {
    BinaryCrossEntropyLoss criterion;

    // Without clipping this is log(1) = 0, which is fine, but the complement
    // term log(1 - p) would be log(0) = -inf.
    Tensor probabilities({1.0f, 0.0f}, {2});
    Tensor targets({1.0f, 0.0f}, {2});

    Tensor loss = criterion(probabilities, targets);

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_NEAR(loss[0], 0.0f, 1e-4f);
}

TEST(BinaryCrossEntropyLossTest, CompletelyWrongPredictionIsClippedNotInfinite) {
    BinaryCrossEntropyLoss criterion;

    // p = 0 while y = 1 is exactly the log(0) case; clipping bounds it at
    // -log(eps) = -log(1e-7) = 16.118
    Tensor probabilities({0.0f}, {1});
    Tensor targets({1.0f}, {1});

    Tensor loss = criterion(probabilities, targets);

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_NEAR(loss[0], -std::log(BinaryCrossEntropyLoss::default_epsilon), 1e-3f);
}

TEST(BinaryCrossEntropyLossTest, ConfidentlyWrongOnBothEndsStaysFinite) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({0.0f, 1.0f}, {2});
    Tensor targets({1.0f, 0.0f}, {2});

    Tensor loss = criterion(probabilities, targets);

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_GT(loss[0], 10.0f);
}

TEST(BinaryCrossEntropyLossTest, OneBadElementDoesNotPoisonTheBatch) {
    BinaryCrossEntropyLoss criterion;

    // Without clipping the single -inf element would make the whole mean -inf.
    Tensor probabilities({0.9f, 0.9f, 0.9f, 0.0f}, {4});
    Tensor targets({1.0f, 1.0f, 1.0f, 1.0f}, {4});

    Tensor loss = criterion(probabilities, targets);

    EXPECT_TRUE(std::isfinite(loss[0]));
}

TEST(BinaryCrossEntropyLossTest, CustomEpsilonChangesTheClippingBound) {
    BinaryCrossEntropyLoss criterion(Reduction::Mean, 1e-3f);

    EXPECT_FLOAT_EQ(criterion.epsilon(), 1e-3f);

    Tensor probabilities({0.0f}, {1});
    Tensor targets({1.0f}, {1});

    // -log(1e-3) = 6.9077553
    EXPECT_NEAR(criterion(probabilities, targets)[0], -std::log(1e-3f), 1e-3f);
}

TEST(BinaryCrossEntropyLossTest, SmallerEpsilonGivesLargerWorstCaseLoss) {
    Tensor probabilities({0.0f}, {1});
    Tensor targets({1.0f}, {1});

    const float coarse = BinaryCrossEntropyLoss(Reduction::Mean, 1e-2f)(probabilities, targets)[0];
    const float fine = BinaryCrossEntropyLoss(Reduction::Mean, 1e-6f)(probabilities, targets)[0];

    EXPECT_GT(fine, coarse);
}

TEST(BinaryCrossEntropyLossTest, ClippingDoesNotDisturbInteriorValues) {
    BinaryCrossEntropyLoss criterion;

    // Values well inside (0, 1) must be untouched by the clipping step.
    Tensor probabilities({0.3f}, {1});
    Tensor targets({1.0f}, {1});

    EXPECT_NEAR(criterion(probabilities, targets)[0], -std::log(0.3f), kTolerance);
}

// ================= Invalid Input =================

TEST(BinaryCrossEntropyLossTest, MismatchedShapesThrow) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({0.5f, 0.5f}, {2});
    Tensor targets({1.0f}, {1});

    EXPECT_THROW(static_cast<void>(criterion(probabilities, targets)), std::invalid_argument);
}

TEST(BinaryCrossEntropyLossTest, ProbabilityAboveOneThrows) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({1.5f}, {1});
    Tensor targets({1.0f}, {1});

    EXPECT_THROW(static_cast<void>(criterion(probabilities, targets)), std::invalid_argument);
}

TEST(BinaryCrossEntropyLossTest, NegativeProbabilityThrows) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({-0.1f}, {1});
    Tensor targets({1.0f}, {1});

    EXPECT_THROW(static_cast<void>(criterion(probabilities, targets)), std::invalid_argument);
}

TEST(BinaryCrossEntropyLossTest, TargetOutsideUnitIntervalThrows) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({0.5f}, {1});
    Tensor targets({2.0f}, {1});

    EXPECT_THROW(static_cast<void>(criterion(probabilities, targets)), std::invalid_argument);
}

TEST(BinaryCrossEntropyLossTest, EmptyInputThrows) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({}, Tensor::Shape{0});
    Tensor targets({}, Tensor::Shape{0});

    EXPECT_THROW(static_cast<void>(criterion(probabilities, targets)), std::invalid_argument);
}

TEST(BinaryCrossEntropyLossTest, InvalidEpsilonThrows) {
    EXPECT_THROW(BinaryCrossEntropyLoss(Reduction::Mean, 0.0f), std::invalid_argument);
    EXPECT_THROW(BinaryCrossEntropyLoss(Reduction::Mean, -1e-3f), std::invalid_argument);
    EXPECT_THROW(BinaryCrossEntropyLoss(Reduction::Mean, 0.5f), std::invalid_argument);
    EXPECT_THROW(BinaryCrossEntropyLoss(Reduction::Mean, 1.0f), std::invalid_argument);

    // Too small to survive float32 rounding: 1 - eps would collapse back to 1.
    EXPECT_THROW(BinaryCrossEntropyLoss(Reduction::Mean, 1e-12f), std::invalid_argument);
}

// ================= Regression =================

TEST(BinaryCrossEntropyLossTest, RegressionKnownBatchValue) {
    BinaryCrossEntropyLoss criterion;

    Tensor probabilities({0.8f, 0.4f, 0.6f, 0.2f}, {2, 2});
    Tensor targets({1.0f, 0.0f, 1.0f, 0.0f}, {2, 2});

    // element costs: -log(0.8), -log(1 - 0.4), -log(0.6), -log(1 - 0.2)
    const float expected =
        -(std::log(0.8f) + std::log(0.6f) + std::log(0.6f) + std::log(0.8f)) / 4.0f;

    EXPECT_NEAR(criterion(probabilities, targets)[0], expected, kTolerance);
}