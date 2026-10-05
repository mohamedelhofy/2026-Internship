#pragma once

#include "tensor/tensor.h"
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

// ================= Base Transform =================
class Transform {
  public:
    virtual ~Transform() = default;
    [[nodiscard]] virtual Tensor forward(const Tensor& img) = 0;

    [[nodiscard]] Tensor operator()(const Tensor& img) { return forward(img); }
};

// ================= RandomHorizontalFlip =================
class RandomHorizontalFlip : public Transform {
  public:
    explicit RandomHorizontalFlip(Tensor::value_type p = 0.5f,
                                  std::optional<unsigned int> seed = std::nullopt);
    [[nodiscard]] Tensor forward(const Tensor& img) override;

  private:
    Tensor::value_type p_;
    std::mt19937 gen_;
};

// ================= RandomCrop =================
class RandomCrop : public Transform {
  public:
    // size: {Height, Width}, padding: pixels to pad on all sides
    RandomCrop(std::pair<Tensor::size_type, Tensor::size_type> size, Tensor::size_type padding = 4,
               std::optional<unsigned int> seed = std::nullopt);
    [[nodiscard]] Tensor forward(const Tensor& img) override;

  private:
    std::pair<Tensor::size_type, Tensor::size_type> size_;
    Tensor::size_type padding_;
    std::mt19937 gen_;
};

// ================= Compose =================
class Compose : public Transform {
  public:
    explicit Compose(const std::vector<std::shared_ptr<Transform>>& transforms);
    [[nodiscard]] Tensor forward(const Tensor& img) override;

  private:
    std::vector<std::shared_ptr<Transform>> transforms_;
};
