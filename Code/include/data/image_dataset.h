#pragma once

#include "data/dataset.h"
#include "data/transforms.h"
#include <memory>
#include <string>
#include <vector>

class ImageDataset : public Dataset {
  public:
    ImageDataset(const std::vector<std::string>& image_paths, const std::vector<Tensor>& labels,
                 std::shared_ptr<Transform> transform = nullptr);

    ~ImageDataset() override = default;

    [[nodiscard]] Tensor::size_type size() const override;
    [[nodiscard]] std::vector<Tensor> get_item(Tensor::size_type idx) const override;

  private:
    std::vector<std::string> image_paths_;
    std::vector<Tensor> labels_;
    std::shared_ptr<Transform> transform_;

    // Mock for loading images. In production, this uses OpenCV or stb_image.
    [[nodiscard]] Tensor load_jpeg(const std::string& path) const;
};
