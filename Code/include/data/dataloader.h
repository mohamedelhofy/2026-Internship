#pragma once

#include "data/dataset.h"
#include <optional>
#include <random>
#include <vector>

class DataLoader {
  public:
    DataLoader(const Dataset& dataset, Tensor::size_type batch_size, bool shuffle = false,
               std::optional<unsigned int> seed = std::nullopt);

    // C++ Iterator for Python Generator bridging
    class Iterator {
      public:
        Iterator(const DataLoader& loader, Tensor::size_type batch_idx);
        std::vector<Tensor> operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
        bool operator==(const Iterator& other) const;

      private:
        const DataLoader& loader_;
        Tensor::size_type batch_idx_;
    };

    [[nodiscard]] Iterator begin() const;
    [[nodiscard]] Iterator end() const;
    [[nodiscard]] Tensor::size_type size() const; // Number of batches

  private:
    const Dataset& dataset_;
    Tensor::size_type batch_size_;
    bool shuffle_;
    mutable std::vector<Tensor::size_type> indices_;
    std::mt19937 gen_;

    void reset_indices() const;
    [[nodiscard]] std::vector<Tensor>
    collate_batch(const std::vector<std::vector<Tensor>>& batch_samples) const;
};
