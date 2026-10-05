#include "data/transforms.h"
#include <algorithm>
#include <stdexcept>

// ================= RandomHorizontalFlip =================
RandomHorizontalFlip::RandomHorizontalFlip(Tensor::value_type p, std::optional<unsigned int> seed)
    : p_(p) {
    if (seed.has_value()) {
        gen_.seed(seed.value());
    } else {
        std::random_device rd;
        gen_.seed(rd());
    }
}

Tensor RandomHorizontalFlip::forward(const Tensor& img) {
    std::uniform_real_distribution<Tensor::value_type> dist(0.0f, 1.0f);
    if (dist(gen_) >= p_) {
        return img; // Skip flip
    }

    if (img.shape().size() < 2) {
        throw std::invalid_argument("Image must have at least 2 dimensions for horizontal flip.");
    }

    Tensor::Shape shape = img.shape();
    Tensor result(shape);

    Tensor::size_type width = shape.back();
    Tensor::size_type height = shape[shape.size() - 2];
    Tensor::size_type channels = (shape.size() > 2) ? shape[shape.size() - 3] : 1;

    for (Tensor::size_type c = 0; c < channels; ++c) {
        for (Tensor::size_type h = 0; h < height; ++h) {
            for (Tensor::size_type w = 0; w < width; ++w) {
                Tensor::size_type src_idx = (c * height * width) + (h * width) + w;
                Tensor::size_type dst_idx = (c * height * width) + (h * width) + (width - 1 - w);
                result.data()[dst_idx] = img.data()[src_idx];
            }
        }
    }
    return result;
}

// ================= RandomCrop =================
RandomCrop::RandomCrop(std::pair<Tensor::size_type, Tensor::size_type> size,
                       Tensor::size_type padding, std::optional<unsigned int> seed)
    : size_(size), padding_(padding) {
    if (seed.has_value()) {
        gen_.seed(seed.value());
    } else {
        std::random_device rd;
        gen_.seed(rd());
    }
}

Tensor RandomCrop::forward(const Tensor& img) {
    Tensor::Shape shape = img.shape();
    if (shape.size() < 2)
        throw std::invalid_argument("Image must have at least 2 dimensions.");

    Tensor::size_type in_w = shape.back();
    Tensor::size_type in_h = shape[shape.size() - 2];
    Tensor::size_type channels = (shape.size() > 2) ? shape[shape.size() - 3] : 1;

    Tensor::size_type pad_w = in_w + 2 * padding_;
    Tensor::size_type pad_h = in_h + 2 * padding_;

    // Generate random top-left corner
    std::uniform_int_distribution<Tensor::size_type> dist_h(0, pad_h - size_.first);
    std::uniform_int_distribution<Tensor::size_type> dist_w(0, pad_w - size_.second);

    Tensor::size_type start_h = dist_h(gen_);
    Tensor::size_type start_w = dist_w(gen_);

    Tensor::Shape out_shape = shape;
    out_shape[out_shape.size() - 2] = size_.first;
    out_shape[out_shape.size() - 1] = size_.second;
    Tensor result(out_shape);

    for (Tensor::size_type c = 0; c < channels; ++c) {
        for (Tensor::size_type out_y = 0; out_y < size_.first; ++out_y) {
            for (Tensor::size_type out_x = 0; out_x < size_.second; ++out_x) {
                // Map to padded coordinates
                Tensor::size_type py = start_h + out_y;
                Tensor::size_type px = start_w + out_x;

                Tensor::value_type val = 0.0f; // Padding value (black)
                // Check if inside original image
                if (py >= padding_ && py < padding_ + in_h && px >= padding_ &&
                    px < padding_ + in_w) {
                    Tensor::size_type orig_y = py - padding_;
                    Tensor::size_type orig_x = px - padding_;
                    Tensor::size_type src_idx = (c * in_h * in_w) + (orig_y * in_w) + orig_x;
                    val = img.data()[src_idx];
                }

                Tensor::size_type dst_idx =
                    (c * size_.first * size_.second) + (out_y * size_.second) + out_x;
                result.data()[dst_idx] = val;
            }
        }
    }
    return result;
}

// ================= Compose =================
Compose::Compose(const std::vector<std::shared_ptr<Transform>>& transforms)
    : transforms_(transforms) {}

Tensor Compose::forward(const Tensor& img) {
    Tensor result = img;
    for (const auto& transform : transforms_) {
        result = transform->forward(result);
    }
    return result;
}
