#include "layer/linear.h"
#include "layer/sequential.h"
#include "losses/mse_loss.h"
#include "optimizers/sgd.h"
#include "tensor/tensor.h"
#include "training/clip_grad.h"
#include "training/scheduler.h"
#include "training/trainer.h"
#include <filesystem> // For removing the temp checkpoint file
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// ==========================================
// 1. Scheduler Tests
// ==========================================
TEST(SchedulerTest, CosineAnnealingDecay) {
    CosineSchedule scheduler(0.1f, 0.01f, 100);

    EXPECT_FLOAT_EQ(scheduler.get_lr(0), 0.1f);

    EXPECT_FLOAT_EQ(scheduler.get_lr(100), 0.01f);

    EXPECT_FLOAT_EQ(scheduler.get_lr(150), 0.01f);

    EXPECT_NEAR(scheduler.get_lr(50), 0.055f, 1e-4);
}

// ==========================================
// 2. Gradient Clipping Tests
// ==========================================
TEST(ClipGradTest, GlobalNormScaling) {
    Tensor t({0.0f, 0.0f}, {2});
    t.set_requires_grad(true);

    t.grad_->data()[0] = 3.0f;
    t.grad_->data()[1] = 4.0f;

    std::vector<Tensor*> params = {&t};

    float norm = clip_grad_norm(params, 1.0f);

    EXPECT_FLOAT_EQ(norm, 5.0f);
    EXPECT_FLOAT_EQ(t.grad_->data()[0], 0.6f); // 3.0 * 0.2
    EXPECT_FLOAT_EQ(t.grad_->data()[1], 0.8f); // 4.0 * 0.2
}

// ==========================================
// 3. Trainer Checkpointing Tests
// ==========================================
TEST(TrainerTest, BinaryCheckpointSaveLoad) {
    Sequential model;
    model.add(std::make_shared<Linear>(2, 2, false));

    for (Tensor* param : model.parameters()) {
        std::fill(param->data().begin(), param->data().end(), 7.5f);
    }

    SGD optimizer(model.parameters(), 0.01f);
    MSELoss loss_fn;
    Trainer trainer(model, optimizer, loss_fn);
    trainer.epoch = 10;
    trainer.step = 500;

    std::string ckpt_path = "test_checkpoint.bin";

    EXPECT_NO_THROW(trainer.save_checkpoint(ckpt_path));

    trainer.epoch = 0;
    trainer.step = 0;
    for (Tensor* param : model.parameters()) {
        std::fill(param->data().begin(), param->data().end(), 0.0f);
    }

    EXPECT_NO_THROW(trainer.load_checkpoint(ckpt_path));

    EXPECT_EQ(trainer.epoch, 10);
    EXPECT_EQ(trainer.step, 500);

    for (Tensor* param : model.parameters()) {
        for (float val : param->data()) {
            EXPECT_FLOAT_EQ(val, 7.5f);
        }
    }

    std::filesystem::remove(ckpt_path);
}