#include "training/scheduler.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CosineSchedule::CosineSchedule(float max_lr, float min_lr, int total_epochs)
    : max_lr_(max_lr), min_lr_(min_lr), total_epochs_(total_epochs) {}

float CosineSchedule::get_lr(int epoch) const {
    if (epoch >= total_epochs_) {
        return min_lr_;
    }
    // Cosine annealing formula
    float cosine_factor = (1.0f + std::cos(M_PI * epoch / total_epochs_)) / 2.0f;
    return min_lr_ + (max_lr_ - min_lr_) * cosine_factor;
}
