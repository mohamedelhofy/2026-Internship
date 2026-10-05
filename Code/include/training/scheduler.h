#pragma once

class CosineSchedule {
  public:
    CosineSchedule(float max_lr, float min_lr, int total_epochs);

    // Returns learning rate for a given epoch
    [[nodiscard]] float get_lr(int epoch) const;

  private:
    float max_lr_;
    float min_lr_;
    int total_epochs_;
};
