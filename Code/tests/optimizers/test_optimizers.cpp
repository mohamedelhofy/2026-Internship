#include "optimizers/adam.h"
#include "optimizers/adamw.h"
#include "optimizers/optimizer.h"
#include "optimizers/sgd.h"
#include "tensor/tensor.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr float kTol = 1e-4f;

// Build a simple parameter list from two 1-D tensors owned here.
struct Params {
    Tensor w;
    Tensor b;
    Optimizer::ParamList list;
    Params() : w({1.0f, 2.0f, 3.0f}, {3}), b({0.5f}, {1}), list{&w, &b} {}
};

void set_grads(Params& p) {
    p.w.set_requires_grad(true);
    p.b.set_requires_grad(true);
    p.w.grad_->fill(0.0f);
    p.b.grad_->fill(0.0f);
    (*p.w.grad_)[0] = 0.1f;
    (*p.w.grad_)[1] = 0.2f;
    (*p.w.grad_)[2] = -0.1f;
    (*p.b.grad_)[0] = 0.3f;
}

} // namespace

// ================= Base Optimizer =================

TEST(OptimizerTest, ZeroGradResetsBuffersWithoutNulling) {
    Params p;
    set_grads(p);
    SGD opt(p.list, 0.01f);
    opt.zero_grad();
    EXPECT_NEAR((*p.w.grad_)[0], 0.0f, kTol);
    EXPECT_NEAR((*p.w.grad_)[2], 0.0f, kTol);
    EXPECT_NEAR((*p.b.grad_)[0], 0.0f, kTol);
    // The gradient buffer is preserved, not deallocated.
    EXPECT_TRUE(p.w.grad_ != nullptr);
}

TEST(OptimizerTest, StepCountIncrements) {
    Params p;
    set_grads(p);
    SGD opt(p.list, 0.01f);
    EXPECT_EQ(opt.step_count(), 0u);
    opt.step();
    opt.step();
    opt.step();
    EXPECT_EQ(opt.step_count(), 3u);
}

// ================= SGD =================

TEST(SGDTest, PlainUpdate) {
    Params p;
    set_grads(p);
    SGD opt(p.list, 0.01f);
    opt.step();
    // theta -= lr * g
    EXPECT_NEAR(p.w[0], 1.0f - 0.01f * 0.1f, kTol);
    EXPECT_NEAR(p.w[1], 2.0f - 0.01f * 0.2f, kTol);
    EXPECT_NEAR(p.w[2], 3.0f - 0.01f * (-0.1f), kTol);
    EXPECT_NEAR(p.b[0], 0.5f - 0.01f * 0.3f, kTol);
}

TEST(SGDTest, WeightDecayAddsL2TermToGrad) {
    Params p;
    set_grads(p);
    SGD opt(p.list, 0.1f, 0.0f, 0.5f);
    opt.step();
    // g' = g + lambda * theta ; theta -= lr * g'
    const float expected_w0 = 1.0f - 0.1f * (0.1f + 0.5f * 1.0f);
    EXPECT_NEAR(p.w[0], expected_w0, kTol);
}

TEST(SGDTest, MomentumUsesVelocity) {
    Params p;
    set_grads(p);
    SGD opt(p.list, 0.01f, 0.9f);
    opt.step();
    // v = g; theta -= lr * v
    EXPECT_NEAR(p.w[0], 1.0f - 0.01f * 0.1f, kTol);
    // Second step: v = 0.9*v + g
    const float v0_prev = 0.1f;
    const float v0 = 0.9f * v0_prev + 0.1f;
    opt.step();
    EXPECT_NEAR(p.w[0], 1.0f - 0.01f * v0_prev - 0.01f * v0, kTol);
}

// ================= Adam =================

TEST(AdamTest, HandComputedSingleStep) {
    Params p;
    set_grads(p);
    Adam opt(p.list, 0.1f, 0.9f, 0.999f, 1e-8f);
    opt.step();

    // m = (1 - b1) * g ; v = (1 - b2) * g^2
    const float g0 = 0.1f;
    const float m0 = (1.0f - 0.9f) * g0;
    const float v0 = (1.0f - 0.999f) * g0 * g0;
    const float m_hat = m0 / (1.0f - 0.9f);
    const float v_hat = v0 / (1.0f - 0.999f);
    const float expected_w0 = 1.0f - 0.1f * m_hat / (std::sqrt(v_hat) + 1e-8f);
    EXPECT_NEAR(p.w[0], expected_w0, 1e-4f);
}

TEST(AdamTest, BiasCorrectionRecoversUnbiasedFirstMoment) {
    // At step 1 the bias-corrected m_hat equals the raw gradient g.
    Params p;
    set_grads(p);
    Adam opt(p.list, 0.1f, 0.9f, 0.999f, 1e-8f);
    opt.step();

    const float g0 = 0.1f;
    const float m_hat = g0; // after bias correction on the first step
    const float v0 = (1.0f - 0.999f) * g0 * g0;
    const float v_hat = v0 / (1.0f - 0.999f);
    const float expected = 1.0f - 0.1f * m_hat / (std::sqrt(v_hat) + 1e-8f);
    EXPECT_NEAR(p.w[0], expected, 1e-4f);
}

TEST(AdamTest, WeightDecayFoldedIntoGradient) {
    Params p;
    set_grads(p);
    Adam opt(p.list, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f);
    opt.step();

    const float g0 = 0.1f + 0.5f * 1.0f; // wrapped decay
    const float m0 = (1.0f - 0.9f) * g0;
    const float v0 = (1.0f - 0.999f) * g0 * g0;
    const float m_hat = m0 / (1.0f - 0.9f);
    const float v_hat = v0 / (1.0f - 0.999f);
    const float expected = 1.0f - 0.1f * m_hat / (std::sqrt(v_hat) + 1e-8f);
    EXPECT_NEAR(p.w[0], expected, 1e-4f);
}

// ================= AdamW =================

TEST(AdamWTest, UsesPureGradientForMomentUpdate) {
    Params p;
    set_grads(p);
    AdamW opt(p.list, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f);
    opt.step();

    // Moment update uses PURE gradient (weight decay NOT added).
    const float g0 = 0.1f;
    const float m0 = (1.0f - 0.9f) * g0;
    const float v0 = (1.0f - 0.999f) * g0 * g0;
    const float m_hat = m0 / (1.0f - 0.9f);
    const float v_hat = v0 / (1.0f - 0.999f);
    const float adam_term = 0.1f * m_hat / (std::sqrt(v_hat) + 1e-8f);

    // THEN decoupled decay: theta *= (1 - lr * lambda)
    const float decay = 1.0f - 0.1f * 0.5f;
    const float expected = (1.0f - adam_term) * decay;
    EXPECT_NEAR(p.w[0], expected, 1e-4f);
}

TEST(AdamWTest, NoWeightDecayMatchesAdam) {
    Params p_adamw;
    set_grads(p_adamw);
    AdamW optw(p_adamw.list, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
    optw.step();

    Params p_adam;
    set_grads(p_adam);
    Adam opta(p_adam.list, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
    opta.step();

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(p_adamw.w[i], p_adam.w[i], kTol);
    }
    EXPECT_NEAR(p_adamw.b[0], p_adam.b[0], kTol);
}
