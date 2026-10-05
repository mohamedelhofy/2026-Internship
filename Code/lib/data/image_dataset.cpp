#include "data/image_dataset.h"
#include <stdexcept>

ImageDataset::ImageDataset(const std::vector<std::string>& image_paths,
                           const std::vector<Tensor>& labels, std::shared_ptr<Transform> transform)
    : image_paths_(image_paths), labels_(labels), transform_(transform) {

    if (image_paths_.size() != labels_.size()) {
        throw std::invalid_argument("Number of images and labels must match.");
    }
}

Tensor::size_type ImageDataset::size() const { return image_paths_.size(); }

Tensor ImageDataset::load_jpeg(const std::string& /*path*/) const {
    // MOCK: Return a 3x32x32 tensor filled with 1.0f to simulate an image
    Tensor::Shape dummy_shape = {3, 32, 32};
    Tensor img(dummy_shape);
    img.fill(1.0f);
    return img;
}

std::vector<Tensor> ImageDataset::get_item(Tensor::size_type idx) const {
    if (idx >= image_paths_.size()) {
        throw std::out_of_range("ImageDataset index out of bounds.");
    }

    Tensor img = load_jpeg(image_paths_[idx]);

    if (transform_) {
        img = transform_->forward(img);
    }

    return {img, labels_[idx]};
}
