#pragma once
#include "data/dataloader.h"
#include "layer/sequential.h"
#include "losses/mse_loss.h"
#include "optimizers/optimizer.h"
#include "training/scheduler.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Trainer {
  public:
    Trainer(Sequential& model, Optimizer& optimizer, MSELoss& loss_fn,
            std::shared_ptr<CosineSchedule> scheduler = nullptr, float grad_clip_norm = -1.0f);

    // Orchestrates one complete pass over the dataset
    float train_epoch(DataLoader& dataloader, int accumulation_steps = 1);

    // Evaluates model without backward pass
    std::pair<float, float> evaluate(DataLoader& dataloader);

    // Fault tolerance: Binary serialization of training state
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    // State Tracking
    int epoch = 0;
    int step = 0;
    std::map<std::string, std::vector<float>> history;

  private:
    Sequential& model_;
    Optimizer& optimizer_;
    MSELoss& loss_fn_;
    std::shared_ptr<CosineSchedule> scheduler_;
    float grad_clip_norm_;
    bool training_mode_ = true;
};
