#include "losses/mse_loss.h"
#include "tensor/tensor.h"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
constexpr float kTolerance = 1e-5f;
}

// ================= Correct Numerical Values =================

TEST(MSELossTest, IdenticalTensorsGiveZero) {
    MSELoss criterion;

    Tensor prediction({1.0f, 2.0f, 3.0f}, {3});
    Tensor target({1.0f, 2.0f, 3.0f}, {3});

    Tensor loss = criterion(prediction, target);

    ASSERT_EQ(loss.size(), 1);
    EXPECT_NEAR(loss[0], 0.0f, kTolerance);
}

TEST(MSELossTest, KnownValueMeanReduction) {
    MSELoss criterion;

    Tensor prediction({1.0f, 2.0f, 3.0f}, {3});
    Tensor target({2.0f, 4.0f, 6.0f}, {3});

    // squared errors: 1, 4, 9 -> mean = 14 / 3
    Tensor loss = criterion(prediction, target);

    EXPECT_NEAR(loss[0], 14.0f / 3.0f, kTolerance);
}

TEST(MSELossTest, ScalarShapeIsOne) {
    MSELoss criterion;

    Tensor prediction({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor target({0.0f, 0.0f, 0.0f, 0.0f}, {2, 2});

    Tensor loss = criterion(prediction, target);

    ASSERT_EQ(loss.ndim(), 1);
    EXPECT_EQ(loss.shape()[0], 1);
}

TEST(MSELossTest, SymmetricInArguments) {
    MSELoss criterion;

    Tensor a({1.0f, -2.0f, 3.5f}, {3});
    Tensor b({0.5f, 2.0f, -1.0f}, {3});

    EXPECT_NEAR(criterion(a, b)[0], criterion(b, a)[0], kTolerance);
}

TEST(MSELossTest, NegativeErrorsAreSquared) {
    MSELoss criterion;

    // Errors of -2 and +2 must contribute equally.
    Tensor prediction({0.0f, 4.0f}, {2});
    Tensor target({2.0f, 2.0f}, {2});

    Tensor loss = criterion(prediction, target);

    EXPECT_NEAR(loss[0], 4.0f, kTolerance);
}

// ================= Batch Inputs =================

TEST(MSELossTest, BatchInputAveragesOverAllElements) {
    MSELoss criterion;

    Tensor prediction({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor target({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f}, {2, 3});

    // Only the last element differs, by 1 -> mean = 1 / 6
    Tensor loss = criterion(prediction, target);

    EXPECT_NEAR(loss[0], 1.0f / 6.0f, kTolerance);
}

TEST(MSELossTest, ThreeDimensionalInput) {
    MSELoss criterion;

    Tensor prediction({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {2, 2, 2});
    Tensor target({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {2, 2, 2});

    Tensor loss = criterion(prediction, target);

    EXPECT_NEAR(loss[0], 1.0f, kTolerance);
}

// ================= Reduction Modes =================

TEST(MSELossTest, SumReduction) {
    MSELoss criterion(Reduction::Sum);

    Tensor prediction({1.0f, 2.0f, 3.0f}, {3});
    Tensor target({2.0f, 4.0f, 6.0f}, {3});

    Tensor loss = criterion(prediction, target);

    ASSERT_EQ(loss.size(), 1);
    EXPECT_NEAR(loss[0], 14.0f, kTolerance);
}

TEST(MSELossTest, NoneReductionKeepsShape) {
    MSELoss criterion(Reduction::None);

    Tensor prediction({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor target({2.0f, 4.0f, 6.0f, 8.0f}, {2, 2});

    Tensor loss = criterion(prediction, target);

    ASSERT_EQ(loss.ndim(), 2);
    EXPECT_EQ(loss.shape()[0], 2);
    EXPECT_EQ(loss.shape()[1], 2);

    EXPECT_NEAR(loss[0], 1.0f, kTolerance);
    EXPECT_NEAR(loss[1], 4.0f, kTolerance);
    EXPECT_NEAR(loss[2], 9.0f, kTolerance);
    EXPECT_NEAR(loss[3], 16.0f, kTolerance);
}

TEST(MSELossTest, ReductionPropertyIsReported) {
    MSELoss criterion(Reduction::Sum);
    EXPECT_EQ(criterion.reduction(), Reduction::Sum);

    MSELoss default_criterion;
    EXPECT_EQ(default_criterion.reduction(), Reduction::Mean);
}

TEST(MSELossTest, SumEqualsMeanTimesCount) {
    Tensor prediction({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    Tensor target({0.0f, 0.0f, 0.0f, 0.0f}, {4});

    const float mean_loss = MSELoss(Reduction::Mean)(prediction, target)[0];
    const float sum_loss = MSELoss(Reduction::Sum)(prediction, target)[0];

    EXPECT_NEAR(sum_loss, mean_loss * 4.0f, 1e-4f);
}

// ================= Invalid Shapes and Edge Cases =================

TEST(MSELossTest, MismatchedShapesThrow) {
    MSELoss criterion;

    Tensor prediction({1.0f, 2.0f, 3.0f}, {3});
    Tensor target({1.0f, 2.0f}, {2});

    EXPECT_THROW(static_cast<void>(criterion(prediction, target)), std::invalid_argument);
}

TEST(MSELossTest, BroadcastableButUnequalShapesStillThrow) {
    MSELoss criterion;

    // {2, 1} and {2} would broadcast under plain tensor arithmetic; a loss
    // must refuse instead of silently producing a {2, 2} result.
    Tensor prediction({1.0f, 2.0f}, {2, 1});
    Tensor target({1.0f, 2.0f}, {2});

    EXPECT_THROW(static_cast<void>(criterion(prediction, target)), std::invalid_argument);
}

TEST(MSELossTest, EmptyInputThrows) {
    MSELoss criterion;

    Tensor prediction({}, Tensor::Shape{0});
    Tensor target({}, Tensor::Shape{0});

    EXPECT_THROW(static_cast<void>(criterion(prediction, target)), std::invalid_argument);
}

TEST(MSELossTest, SingleElementInput) {
    MSELoss criterion;

    Tensor prediction({3.0f}, {1});
    Tensor target({1.0f}, {1});

    EXPECT_NEAR(criterion(prediction, target)[0], 4.0f, kTolerance);
}

// ================= Numerical Behaviour =================

TEST(MSELossTest, LargeButRepresentableValuesStayFinite) {
    MSELoss criterion;

    Tensor prediction({1.0e6f, -1.0e6f}, {2});
    Tensor target({0.0f, 0.0f}, {2});

    Tensor loss = criterion(prediction, target);

    EXPECT_TRUE(std::isfinite(loss[0]));
    EXPECT_NEAR(loss[0], 1.0e12f, 1.0e7f);
}

TEST(MSELossTest, VerySmallErrorsDoNotUnderflowToZeroPrematurely) {
    MSELoss criterion;

    Tensor prediction({1.0e-3f, 1.0e-3f}, {2});
    Tensor target({0.0f, 0.0f}, {2});

    Tensor loss = criterion(prediction, target);

    EXPECT_GT(loss[0], 0.0f);
    EXPECT_NEAR(loss[0], 1.0e-6f, 1.0e-9f);
}

// ================= Regression =================

TEST(MSELossTest, RegressionKnownBatchValue) {
    MSELoss criterion;

    Tensor prediction({0.5f, -1.5f, 2.0f, 0.0f}, {2, 2});
    Tensor target({1.0f, -1.0f, 0.0f, 0.5f}, {2, 2});

    // squared errors: 0.25, 0.25, 4.0, 0.25 -> mean = 4.75 / 4 = 1.1875
    EXPECT_NEAR(criterion(prediction, target)[0], 1.1875f, kTolerance);
}