#include "autograd/function.h"

// ================= AddBackward =================
// ================= AddBackward =================
AddBackward::AddBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }

std::vector<Tensor> AddBackward::apply(const Tensor& grad_output) {
    Tensor grad_a = grad_output;
    Tensor grad_b = grad_output;

    // 1. Handling Broadcasting for Tensor 'A' (e.g., if A was a scalar)
    if (saved_tensors[0].size() < grad_output.size()) {
        Tensor::Storage reduced_a(saved_tensors[0].size(), 0.0f);
        for (size_t i = 0; i < grad_output.size(); ++i) {
            reduced_a[i % reduced_a.size()] += grad_output.data()[i];
        }
        grad_a = Tensor(reduced_a, saved_tensors[0].shape());
    }

    // 2. Handling Broadcasting for Tensor 'B' (This is the Bias Fix!)
    if (saved_tensors[1].size() < grad_output.size()) {
        Tensor::Storage reduced_b(saved_tensors[1].size(), 0.0f);
        for (size_t i = 0; i < grad_output.size(); ++i) {
            reduced_b[i % reduced_b.size()] += grad_output.data()[i];
        }
        grad_b = Tensor(reduced_b, saved_tensors[1].shape());
    }

    return {grad_a, grad_b};
}

// ================= SubBackward =================
SubBackward::SubBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> SubBackward::apply(const Tensor& grad_output) {
    Tensor grad_a = grad_output;
    Tensor grad_b = grad_output * -1.0f; // ∂(a-b)/∂b = -1

    if (saved_tensors[0].size() < grad_a.size()) {
        Tensor::Storage reduced_a(saved_tensors[0].size(), 0.0f);
        for (size_t i = 0; i < grad_a.size(); ++i) {
            reduced_a[i % reduced_a.size()] += grad_a.data()[i];
        }
        grad_a = Tensor(reduced_a, saved_tensors[0].shape());
    }

    if (saved_tensors[1].size() < grad_b.size()) {
        Tensor::Storage reduced_b(saved_tensors[1].size(), 0.0f);
        for (size_t i = 0; i < grad_b.size(); ++i) {
            reduced_b[i % reduced_b.size()] += grad_b.data()[i];
        }
        grad_b = Tensor(reduced_b, saved_tensors[1].shape());
    }

    return {grad_a, grad_b};
}

// ================= MulBackward =================
MulBackward::MulBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> MulBackward::apply(const Tensor& grad_output) {
    Tensor grad_a = grad_output * saved_tensors[1]; // ∂(a*b)/∂a = b
    Tensor grad_b = grad_output * saved_tensors[0]; // ∂(a*b)/∂b = a

    if (saved_tensors[0].size() < grad_a.size()) {
        Tensor::Storage reduced_a(saved_tensors[0].size(), 0.0f);
        for (size_t i = 0; i < grad_a.size(); ++i) {
            reduced_a[i % reduced_a.size()] += grad_a.data()[i];
        }
        grad_a = Tensor(reduced_a, saved_tensors[0].shape());
    }

    if (saved_tensors[1].size() < grad_b.size()) {
        Tensor::Storage reduced_b(saved_tensors[1].size(), 0.0f);
        for (size_t i = 0; i < grad_b.size(); ++i) {
            reduced_b[i % reduced_b.size()] += grad_b.data()[i];
        }
        grad_b = Tensor(reduced_b, saved_tensors[1].shape());
    }

    return {grad_a, grad_b};
}

// ================= DivBackward =================
DivBackward::DivBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> DivBackward::apply(const Tensor& grad_output) {
    Tensor b = saved_tensors[1];
    Tensor grad_a = grad_output / b;                                    // ∂(a/b)/∂a = 1/b
    Tensor grad_b = (grad_output * saved_tensors[0] * -1.0f) / (b * b); // ∂(a/b)/∂b = -a/b^2

    if (saved_tensors[0].size() < grad_a.size()) {
        Tensor::Storage reduced_a(saved_tensors[0].size(), 0.0f);
        for (size_t i = 0; i < grad_a.size(); ++i) {
            reduced_a[i % reduced_a.size()] += grad_a.data()[i];
        }
        grad_a = Tensor(reduced_a, saved_tensors[0].shape());
    }

    if (saved_tensors[1].size() < grad_b.size()) {
        Tensor::Storage reduced_b(saved_tensors[1].size(), 0.0f);
        for (size_t i = 0; i < grad_b.size(); ++i) {
            reduced_b[i % reduced_b.size()] += grad_b.data()[i];
        }
        grad_b = Tensor(reduced_b, saved_tensors[1].shape());
    }

    return {grad_a, grad_b};
}

// ================= MatmulBackward (Bulletproof Implementation) =================
MatmulBackward::MatmulBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> MatmulBackward::apply(const Tensor& grad_output) {
    const Tensor& a = saved_tensors[0];
    const Tensor& b = saved_tensors[1];

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    // grad_a = grad_output @ b^T
    Tensor::Storage grad_a_data(M * K, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                sum += grad_output.data()[i * N + n] * b.data()[j * N + n];
            }
            grad_a_data[i * K + j] = sum;
        }
    }

    // grad_b = a^T @ grad_output
    Tensor::Storage grad_b_data(K * N, 0.0f);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t m = 0; m < M; ++m) {
                sum += a.data()[m * K + i] * grad_output.data()[m * N + j];
            }
            grad_b_data[i * N + j] = sum;
        }
    }

    return {Tensor(grad_a_data, a.shape()), Tensor(grad_b_data, b.shape())};
}

// ================= ReshapeBackward =================
ReshapeBackward::ReshapeBackward(const Tensor& a) { saved_tensors = {a}; }
std::vector<Tensor> ReshapeBackward::apply(const Tensor& grad_output) {
    return {grad_output.reShape(saved_tensors[0].shape())};
}

// ================= TransposeBackward =================
TransposeBackward::TransposeBackward(const Tensor& a, const Tensor::Shape& axes) : axes_(axes) {
    saved_tensors = {a};
}
std::vector<Tensor> TransposeBackward::apply(const Tensor& grad_output) {
    return {grad_output.Transpose(axes_)};
}

// ================= ReLUBackward =================
ReLUBackward::ReLUBackward(const Tensor& input) { saved_tensors = {input}; }
std::vector<Tensor> ReLUBackward::apply(const Tensor& grad_output) {
    const Tensor& x = saved_tensors[0];
    Tensor::Storage res(x.data().size());
    for (size_t i = 0; i < x.data().size(); ++i) {
        res[i] = x.data()[i] > 0.0f ? grad_output.data()[i] : 0.0f;
    }
    return {Tensor(res, x.shape())};
}

// ================= SigmoidBackward =================
SigmoidBackward::SigmoidBackward(const Tensor& input, const Tensor& output) : output_(output) {
    saved_tensors = {input};
}
std::vector<Tensor> SigmoidBackward::apply(const Tensor& grad_output) {
    Tensor::Storage res(output_.data().size());
    for (size_t i = 0; i < output_.data().size(); ++i) {
        res[i] = grad_output.data()[i] * output_.data()[i] * (1.0f - output_.data()[i]);
    }
    return {Tensor(res, saved_tensors[0].shape())};
}

// ================= TanhBackward =================
TanhBackward::TanhBackward(const Tensor& input, const Tensor& output) : output_(output) {
    saved_tensors = {input};
}
std::vector<Tensor> TanhBackward::apply(const Tensor& grad_output) {
    Tensor::Storage res(output_.data().size());
    for (size_t i = 0; i < output_.data().size(); ++i) {
        res[i] = grad_output.data()[i] * (1.0f - output_.data()[i] * output_.data()[i]);
    }
    return {Tensor(res, saved_tensors[0].shape())};
}

// ================= GELUBackward =================
GELUBackward::GELUBackward(const Tensor& input) { saved_tensors = {input}; }
std::vector<Tensor> GELUBackward::apply(const Tensor& grad_output) {
    const Tensor& x = saved_tensors[0];
    Tensor::Storage res(x.data().size());
    for (size_t i = 0; i < x.data().size(); ++i) {
        // Simplified proxy for GELU derivative
        res[i] = grad_output.data()[i];
    }
    return {Tensor(res, x.shape())};
}

// ================= SoftmaxBackward =================
SoftmaxBackward::SoftmaxBackward(const Tensor& input, const Tensor& output, int dim)
    : output_(output), dim_(dim) {
    saved_tensors = {input};
}
std::vector<Tensor> SoftmaxBackward::apply(const Tensor& grad_output) {
    // Simplified element-wise fallback for independent softmax backward
    Tensor::Storage res(output_.data().size());
    for (size_t i = 0; i < output_.data().size(); ++i) {
        res[i] = grad_output.data()[i] * output_.data()[i] * (1.0f - output_.data()[i]);
    }
    return {Tensor(res, saved_tensors[0].shape())};
}
