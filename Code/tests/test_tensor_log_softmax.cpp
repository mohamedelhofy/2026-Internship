#include "tensor/tensor.h"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
constexpr float kTolerance = 1e-4f;
}

// ================= Log-Softmax Correctness =================

TEST(TensorLogSoftmaxTest, UniformInputGivesMinusLogNumClasses) {
    Tensor logits({0.0f, 0.0f, 0.0f, 0.0f}, {4});

    Tensor result = logits.log_softmax(-1);

    ASSERT_EQ(result.size(), 4);
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], -std::log(4.0f), kTolerance);
    }
}

TEST(TensorLogSoftmaxTest, ExponentiatedResultSumsToOne) {
    Tensor logits({1.0f, -2.0f, 0.5f, 3.0f}, {4});

    Tensor result = logits.log_softmax(-1);

    float total = 0.0f;
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        total += std::exp(result[i]);
    }

    EXPECT_NEAR(total, 1.0f, kTolerance);
}

TEST(TensorLogSoftmaxTest, MatchesLogOfSoftmaxForModerateValues) {
    Tensor logits({0.3f, -1.2f, 2.5f, 0.0f}, {4});

    Tensor stable = logits.log_softmax(-1);
    Tensor probabilities = logits.softmax(-1);

    for (Tensor::size_type i = 0; i < stable.size(); ++i) {
        EXPECT_NEAR(stable[i], std::log(probabilities[i]), kTolerance);
    }
}

TEST(TensorLogSoftmaxTest, AllValuesAreNonPositive) {
    Tensor logits({5.0f, -5.0f, 0.0f}, {3});

    Tensor result = logits.log_softmax(-1);

    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        EXPECT_LE(result[i], 0.0f);
    }
}

TEST(TensorLogSoftmaxTest, ShapeIsPreserved) {
    Tensor logits({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});

    Tensor result = logits.log_softmax(-1);

    ASSERT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 3);
}

TEST(TensorLogSoftmaxTest, ShiftInvariance) {
    Tensor logits({1.0f, 2.0f, 3.0f}, {3});
    Tensor shifted({101.0f, 102.0f, 103.0f}, {3});

    Tensor a = logits.log_softmax(-1);
    Tensor b = shifted.log_softmax(-1);

    for (Tensor::size_type i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], kTolerance);
    }
}

// ================= Dimensions =================

TEST(TensorLogSoftmaxTest, NormalizesAlongLastDimensionByDefault) {
    Tensor logits({0.0f, 0.0f, 0.0f, 0.0f}, {2, 2});

    Tensor result = logits.log_softmax();

    // Each row of two equal logits -> log(0.5)
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], -std::log(2.0f), kTolerance);
    }
}

TEST(TensorLogSoftmaxTest, NormalizesAlongDimensionZero) {
    // Columns are {1, 3} and {2, 4}.
    Tensor logits({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    Tensor result = logits.log_softmax(0);

    // Column 0: log_sum_exp = 3 + log(1 + e^-2) = 3.126928
    EXPECT_NEAR(result({0, 0}), 1.0f - 3.126928f, kTolerance);
    EXPECT_NEAR(result({1, 0}), 3.0f - 3.126928f, kTolerance);

    // Each column must exponentiate back to 1.
    EXPECT_NEAR(std::exp(result({0, 0})) + std::exp(result({1, 0})), 1.0f, kTolerance);
    EXPECT_NEAR(std::exp(result({0, 1})) + std::exp(result({1, 1})), 1.0f, kTolerance);
}

TEST(TensorLogSoftmaxTest, NegativeDimensionIsResolved) {
    Tensor logits({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    Tensor by_negative = logits.log_softmax(-2);
    Tensor by_positive = logits.log_softmax(0);

    for (Tensor::size_type i = 0; i < by_negative.size(); ++i) {
        EXPECT_FLOAT_EQ(by_negative[i], by_positive[i]);
    }
}

TEST(TensorLogSoftmaxTest, ThreeDimensionalInput) {
    Tensor logits({1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f}, {2, 2, 2});

    Tensor result = logits.log_softmax(2);

    // Every lane holds two equal logits -> log(0.5)
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], -std::log(2.0f), kTolerance);
    }
}

TEST(TensorLogSoftmaxTest, OutOfRangeDimensionThrows) {
    Tensor logits({1.0f, 2.0f}, {2});

    EXPECT_THROW(static_cast<void>(logits.log_softmax(1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(logits.log_softmax(-2)), std::invalid_argument);
}

// ================= Numerical Stability =================

TEST(TensorLogSoftmaxTest, HugeLogitsDoNotOverflow) {
    // exp(1000) is +inf in float32, so log(softmax(x)) would be NaN here.
    Tensor logits({1000.0f, 1000.0f, 1000.0f}, {3});

    Tensor result = logits.log_softmax(-1);

    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result[i]));
        EXPECT_NEAR(result[i], -std::log(3.0f), kTolerance);
    }
}

TEST(TensorLogSoftmaxTest, HugeNegativeLogitsDoNotUnderflow) {
    // Naive softmax would give 0 for every entry and then log(0) = -inf.
    Tensor logits({-1000.0f, -1000.0f}, {2});

    Tensor result = logits.log_softmax(-1);

    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result[i]));
        EXPECT_NEAR(result[i], -std::log(2.0f), kTolerance);
    }
}

TEST(TensorLogSoftmaxTest, WideLogitGapRemainsExact) {
    Tensor logits({0.0f, 1000.0f}, {2});

    Tensor result = logits.log_softmax(-1);

    // The dominant class keeps a log-probability of ~0, and the crushed one
    // degrades linearly instead of saturating at -inf.
    EXPECT_NEAR(result[1], 0.0f, kTolerance);
    EXPECT_TRUE(std::isfinite(result[0]));
    EXPECT_NEAR(result[0], -1000.0f, 0.1f);
}

TEST(TensorLogSoftmaxTest, NaiveCompositionWouldFailWhereStableVersionSucceeds) {
    Tensor logits({0.0f, 1000.0f}, {2});

    // Documents precisely why log(softmax(x)) is forbidden. Even though this
    // softmax is itself stable, its output underflows to exactly 0 for the
    // crushed class, so composing a logarithm on top of it yields -inf and
    // destroys the information the fused form retains.
    Tensor probabilities = logits.softmax(-1);
    Tensor stable = logits.log_softmax(-1);

    EXPECT_FLOAT_EQ(probabilities[0], 0.0f);
    EXPECT_TRUE(std::isinf(std::log(probabilities[0])));

    EXPECT_TRUE(std::isfinite(stable[0]));
    EXPECT_NEAR(stable[0], -1000.0f, 0.1f);
}

TEST(TensorLogSoftmaxTest, SingleClassGivesZero) {
    Tensor logits({7.0f}, {1});

    Tensor result = logits.log_softmax(-1);

    EXPECT_NEAR(result[0], 0.0f, kTolerance);
}

// ================= Element-wise log =================

TEST(TensorLogTest, ComputesNaturalLogarithm) {
    Tensor t({1.0f, std::exp(1.0f), 10.0f}, {3});

    Tensor result = t.log();

    EXPECT_NEAR(result[0], 0.0f, kTolerance);
    EXPECT_NEAR(result[1], 1.0f, kTolerance);
    EXPECT_NEAR(result[2], std::log(10.0f), kTolerance);
}

TEST(TensorLogTest, ShapeIsPreserved) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    Tensor result = t.log();

    ASSERT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST(TensorLogTest, NonPositiveValuesThrow) {
    Tensor zero({0.0f}, {1});
    Tensor negative({-1.0f}, {1});

    EXPECT_THROW(static_cast<void>(zero.log()), std::domain_error);
    EXPECT_THROW(static_cast<void>(negative.log()), std::domain_error);
}

// ================= Element-wise clamp =================

TEST(TensorClampTest, BoundsValuesOnBothSides) {
    Tensor t({-5.0f, 0.25f, 5.0f}, {3});

    Tensor result = t.clamp(0.0f, 1.0f);

    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 0.25f);
    EXPECT_FLOAT_EQ(result[2], 1.0f);
}

TEST(TensorClampTest, InteriorValuesAreUntouched) {
    Tensor t({0.1f, 0.5f, 0.9f}, {3});

    Tensor result = t.clamp(0.0f, 1.0f);

    for (Tensor::size_type i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], t[i]);
    }
}

TEST(TensorClampTest, ShapeIsPreserved) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    Tensor result = t.clamp(2.0f, 3.0f);

    ASSERT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST(TensorClampTest, InvertedBoundsThrow) {
    Tensor t({1.0f}, {1});

    EXPECT_THROW(static_cast<void>(t.clamp(1.0f, 0.0f)), std::invalid_argument);
}

TEST(TensorClampTest, MakesLogarithmSafe) {
    Tensor probabilities({0.0f, 1.0f}, {2});

    // The pairing the BCE criterion relies on.
    Tensor safe = probabilities.clamp(1e-7f, 1.0f - 1e-7f);

    EXPECT_NO_THROW(static_cast<void>(safe.log()));
    EXPECT_NO_THROW(static_cast<void>((1.0f - safe).log()));
}