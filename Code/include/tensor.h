#pragma once

#include <cmath>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

class Function;
class Tensor {
  public:
    // Type aliases for convenience
    using value_type = float;
    using size_type = std::size_t;
    using Shape = std::vector<size_type>;
    using Storage = std::vector<value_type>;

    // Graph and Gradient Storage (Public for easy access during graph traversal)
    std::shared_ptr<Function> grad_fn_ = nullptr;
    std::shared_ptr<Tensor> grad_ = nullptr;

    //  ================= Constructors =================
    Tensor() = default; // default constructor
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor() = default; // destructor

    Tensor(const Storage& data, const Shape& shape)
        : data_(data), shape_(shape) {} // constructor with data and shape

    // ================= Properties ===================
    [[nodiscard]] const Shape& shape() const noexcept {
        return shape_;
    }; // get the shape of the tensor
    [[nodiscard]] const Storage& data() const noexcept {
        return data_;
    }; // get the data of the tensor
    [[nodiscard]] size_type size() const noexcept {
        return data_.size();
    }; // get the size of the tensor (number of elements)
    [[nodiscard]] size_type ndim() const noexcept {
        return shape_.size();
    }; // get the number of dimensions of the tensor
    [[nodiscard]] std::string_view dtype() const noexcept {
        return "float";
    }; // get the data type of the tensor (currently only
       // float is supported)
    [[nodiscard]] bool empty() const noexcept {
        return data_.empty();
    }; // check if the tensor is empty

    // ================= Output ==========================
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    // ================= Equality ==========================
    [[nodiscard]] bool operator==(const Tensor& other) const noexcept;

    // ================= Addition ==========================
    [[nodiscard]] Tensor operator+(const Tensor&) const; // element-wise addition
    [[nodiscard]] Tensor
    operator+(const value_type) const; // add a scalar to each element of the tensor
    friend Tensor operator+(const value_type scalar,
                            const Tensor& tensor); // add a scalar to each element of the tensor
                                                   // (friend function)
    Tensor& operator+=(const Tensor&);             // element-wise addition and assignment
    Tensor& operator+=(const value_type);          // add a scalar to each element of the
                                                   // tensor and assignment

    // ================= Subtraction ==========================
    [[nodiscard]] Tensor operator-(const Tensor&) const; // element-wise subtraction
    [[nodiscard]] Tensor
    operator-(const value_type) const; // subtract a scalar from each element of the tensor
    friend Tensor operator-(const value_type scalar,
                            const Tensor& tensor); // subtract each element of the tensor from a
                                                   // scalar (friend function)
    Tensor& operator-=(const Tensor&);             // element-wise subtraction and assignment
    Tensor& operator-=(const value_type);          // subtract a scalar from each element
                                                   // of the tensor and assignment

    // ================= Multiplication ==========================
    [[nodiscard]] Tensor operator*(const Tensor&) const; // element-wise multiplication
    [[nodiscard]] Tensor
    operator*(const value_type) const; // multiply each element of the tensor by a scalar
    friend Tensor operator*(const value_type scalar,
                            const Tensor& tensor); // multiply each element of the tensor by a
                                                   // scalar (friend function)
    Tensor& operator*=(const Tensor&);             // element-wise multiplication and assignment
    Tensor& operator*=(const value_type);          // multiply each element of the tensor
                                                   // by a scalar and assignment

    // ================= Division ==========================
    [[nodiscard]] Tensor operator/(const Tensor&) const; // element-wise division
    [[nodiscard]] Tensor
    operator/(const value_type) const; // divide each element of the tensor by a scalar
    friend Tensor operator/(const value_type scalar,
                            const Tensor& tensor); // divide a scalar by each element of the
                                                   // tensor (friend function)
    Tensor& operator/=(const Tensor&);             // element-wise division and assignment
    Tensor& operator/=(const value_type);          // divide each element of the tensor by
                                                   // a scalar and assignment

    // ================= Element Access ==========================
    value_type& operator()(const Shape& indices);
    const value_type& operator()(const Shape& indices) const;

    // ================= Matrix Multiplication ==========================
    [[nodiscard]] Tensor matmul(const Tensor&) const;

    // Module -> 2
    // Activation Functions
    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor sigmoid() const;
    [[nodiscard]] Tensor tanh() const;
    [[nodiscard]] Tensor gelu() const;
    [[nodiscard]] Tensor softmax(int dim = -1) const;

    // Module -> 4
    // Element-wise math used by the loss module
    [[nodiscard]] Tensor log() const;
    [[nodiscard]] Tensor clamp(value_type min_value, value_type max_value) const;
    [[nodiscard]] Tensor log_softmax(int dim = -1) const;
    // ================= Autograd Properties =================
    void set_requires_grad(bool req);
    [[nodiscard]] bool requires_grad() const noexcept { return requires_grad_; }

    void backward(const std::optional<Tensor>& gradient = std::nullopt);
    void zero_grad();

  private:
    Storage data_;
    Shape shape_;
    bool requires_grad_ = false;

    template <typename BinaryOp>
    [[nodiscard]] Tensor apply_tensor_operation(const Tensor& other, BinaryOp op,
                                                bool check_division = false) const;

    template <typename BinaryOp>
    [[nodiscard]] Tensor apply_scalar_operation(value_type scalar, BinaryOp op,
                                                bool check_division = false) const;
    // ================= Helper Functions for broadcast ==========================
    static Shape broadcast_shape(const Shape& lhs, const Shape& rhs);

    static size_type compute_size(const Shape& shape);

    static Shape unravel_index(size_type flat_index, const Shape& shape);

    static size_type ravel_index(const Shape& shape, const Shape& index);

    static Shape broadcast_index(const Shape& output_index, const Shape& input_shape);
    // ================= Helper Functions for matrix multiplication ==========================
    static Shape extract_batch_shape(const Shape& shape);
};
