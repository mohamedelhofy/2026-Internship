#pragma once
#include "tensor/tensor.h"
#include <vector>

// Clips gradients by global norm to prevent exploding gradients
float clip_grad_norm(const std::vector<Tensor*>& parameters, float max_norm = 1.0f);
