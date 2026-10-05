<<<<<<< HEAD:Code/lib/tensor_properties.cpp
#include "tensor.h"
=======
#include "tensor/tensor.h"
#include <iomanip>
>>>>>>> origin/feature/final-demo:Code/lib/tensor/tensor_properties.cpp
#include <iostream>
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    if (tensor.data().empty()) {
        os << "Tensor([])";
        return os;
    }

    if (tensor.ndim() == 1) {
        os << "Tensor([";
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            os << std::defaultfloat << std::setprecision(4) << tensor.data()[i];
            if (i < tensor.shape()[0] - 1) {
                os << ", ";
            }
        }
        os << "])";
    } else if (tensor.ndim() == 2) {
        size_t rows = tensor.shape()[0];
        size_t cols = tensor.shape()[1];

        os << "Tensor([\n";
        for (size_t i = 0; i < rows; ++i) {
            os << "  [";
            for (size_t j = 0; j < cols; ++j) {
                os << std::setw(9) << std::fixed << std::setprecision(4)
                   << tensor.data()[i * cols + j];

                if (j < cols - 1) {
                    os << ", ";
                }
            }
            os << "]";
            if (i < rows - 1) {
                os << ",\n";
            }
        }
        os << "\n])";
    } else {
        os << "Tensor(shape=[";
        for (size_t i = 0; i < tensor.shape().size(); ++i) {
            os << tensor.shape()[i];
            if (i < tensor.shape().size() - 1) {
                os << ", ";
            }
        }
        os << "], data=...)";
    }

    return os;
}

bool Tensor::operator==(const Tensor& other) const noexcept {
    return shape_ == other.shape_ && data_ == other.data_;
}
