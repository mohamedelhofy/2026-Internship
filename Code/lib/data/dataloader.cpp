#include "data/dataloader.h"
#include <algorithm>
#include <numeric>

DataLoader::DataLoader(const Dataset& dataset, Tensor::size_type batch_size, bool shuffle,
                       std::optional<unsigned int> seed)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) {

    indices_.resize(dataset_.size());
    std::iota(indices_.begin(), indices_.end(), 0);

    if (seed.has_value()) {
        gen_.seed(seed.value());
    } else {
        std::random_device rd;
        gen_.seed(rd());
    }
}

void DataLoader::reset_indices() const {
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), const_cast<std::mt19937&>(gen_));
    }
}

Tensor::size_type DataLoader::size() const {
    return (dataset_.size() + batch_size_ - 1) / batch_size_;
}

DataLoader::Iterator DataLoader::begin() const {
    reset_indices();
    return Iterator(*this, 0);
}

DataLoader::Iterator DataLoader::end() const { return Iterator(*this, size()); }

std::vector<Tensor>
DataLoader::collate_batch(const std::vector<std::vector<Tensor>>& batch_samples) const {
    if (batch_samples.empty())
        return {};

    Tensor::size_type current_batch_size = batch_samples.size();
    Tensor::size_type num_tensors = batch_samples[0].size();

    std::vector<Tensor> batched_tensors;
    batched_tensors.reserve(num_tensors);

    for (Tensor::size_type t = 0; t < num_tensors; ++t) {
        Tensor::Shape batched_shape = batch_samples[0][t].shape();
        batched_shape.insert(batched_shape.begin(), current_batch_size);

        Tensor::size_type single_sample_size = batch_samples[0][t].size();
        Tensor::Storage batched_data(current_batch_size * single_sample_size);

        for (Tensor::size_type b = 0; b < current_batch_size; ++b) {
            const auto& sample_data = batch_samples[b][t].data();
            std::copy(sample_data.begin(), sample_data.end(),
                      batched_data.begin() + (b * single_sample_size));
        }

        batched_tensors.emplace_back(batched_data, batched_shape);
    }
    return batched_tensors;
}

// ================= Iterator =================
DataLoader::Iterator::Iterator(const DataLoader& loader, Tensor::size_type batch_idx)
    : loader_(loader), batch_idx_(batch_idx) {}

std::vector<Tensor> DataLoader::Iterator::operator*() const {
    Tensor::size_type start_idx = batch_idx_ * loader_.batch_size_;
    Tensor::size_type end_idx = std::min(start_idx + loader_.batch_size_, loader_.dataset_.size());

    std::vector<std::vector<Tensor>> batch_samples;
    batch_samples.reserve(end_idx - start_idx);

    for (Tensor::size_type i = start_idx; i < end_idx; ++i) {
        Tensor::size_type real_idx = loader_.indices_[i];
        batch_samples.push_back(loader_.dataset_.get_item(real_idx));
    }

    return loader_.collate_batch(batch_samples);
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    batch_idx_++;
    return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return batch_idx_ != other.batch_idx_;
}

bool DataLoader::Iterator::operator==(const Iterator& other) const {
    return batch_idx_ == other.batch_idx_;
}
