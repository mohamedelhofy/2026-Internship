#include "layer/linear.h"
#include <cmath>
#include <random>

// ============================================================================
// Linear Layer Implementation
// ============================================================================
Linear::Linear(Tensor::size_type in_features, Tensor::size_type out_features, bool bias,
               std::optional<unsigned int> seed)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias),
      weight_(Tensor::Shape{in_features, out_features}),
      bias_(bias ? Tensor::Shape{out_features} : Tensor::Shape{}) {

    // LeCun-style initialization
    Tensor::value_type scale = std::sqrt(1.0f / static_cast<Tensor::value_type>(in_features));

    // Initialize Random Number Generator (RNG)
    std::mt19937 gen;
    if (seed.has_value()) {
        gen.seed(seed.value()); // Deterministic initialization
    } else {
        std::random_device rd;
        gen.seed(rd()); // True random initialization
    }

    std::normal_distribution<Tensor::value_type> d(0.0f, 1.0f);

    // Populate weights
    for (Tensor::size_type i = 0; i < weight_.size(); ++i) {
        weight_.data()[i] = d(gen) * scale;
    }

    if (use_bias_) {
        bias_.fill(0.0f);
        bias_.set_requires_grad(true);
    }

    weight_.set_requires_grad(true);
}
Tensor Linear::forward(const Tensor& x, bool /*training*/) {
    if (use_bias_) {
        return x.matmul(weight_) + bias_;
    }
    return x.matmul(weight_);
}

Layer::ParamList Linear::parameters() {
    ParamList params;
    params.push_back(&weight_);
    if (use_bias_) {
        params.push_back(&bias_);
    }
    return params;
}