#pragma once

#include "optimizers/adam.h"

// AdamW Optimizer

class AdamW : public Adam {
  public:
    // Default weight_decay is 0.01, matching PyTorch's recommended default.
    AdamW(const ParamList& params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.01f);

    void step() override;
};
