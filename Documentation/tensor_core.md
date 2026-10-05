# Tensor Core & Operations

<<<<<<< HEAD
The Tensor core is the foundation of TinyTorch. It provides a fully dynamic, PyTorch-like N-dimensional Tensor implementation written in modern C++17.

The Tensor class is responsible for:

* Dynamic N-dimensional data representation.
* Contiguous memory storage.
* Shape manipulation.
* Broadcasting.
* Element-wise arithmetic.
* Scalar arithmetic.
* Reduction operations.
* Matrix multiplication.
* Mathematical operations.
* Autograd integration.
* CPU-level performance optimizations.
=======
The **Tensor Core** is the backbone of TinyTorch. It provides a fully dynamic, PyTorch-like **N-dimensional tensor implementation** defined in `tensor.h`.

It handles memory management, mathematical operations, shape manipulation, broadcasting, matrix multiplication, and provides the foundational data structure required by the **Autograd engine**.
>>>>>>> origin/feature/final-demo

---

## Tensor Properties & Initialization

<<<<<<< HEAD
Tensors use a contiguous `std::vector<value_type>` as their underlying storage, with `float` as the default value type.

### Storage

Tensor data is stored in:

```cpp
std::vector<value_type>
```

This provides dynamic allocation and contiguous memory access for the Tensor data.

### Initialization

The Tensor core supports:

* Empty Tensor initialization.
* Shape-based initialization.
* Direct data and shape construction.
* Copy and move construction.
=======
Tensors are built on top of standard C++ vectors, providing dynamic sizing and contiguous memory storage.

### Memory & Storage

* Data is managed using `std::vector<value_type>`.
* `value_type` is `float` by default.
* Tensor elements are stored in a contiguous memory layout where applicable.

### Initialization

The Tensor class supports:

* Empty initialization.
* Shape-based initialization with zero-initialized elements.
* Direct data and shape injection.
>>>>>>> origin/feature/final-demo

### Utility Methods

| Method        | Description                                             |
| ------------- | ------------------------------------------------------- |
<<<<<<< HEAD
| `shape()`     | Returns the Tensor dimensions as `std::vector<size_t>`. |
| `size()`      | Returns the total number of elements.                   |
| `ndim()`      | Returns the number of dimensions.                       |
| `empty()`     | Checks whether the Tensor contains no elements.         |
| `dtype()`     | Returns the underlying data type.                       |
| `fill(value)` | Fills the entire Tensor with a scalar value.            |

---

# Shape Operations

TinyTorch supports dynamic N-dimensional shape manipulation.

## `reShape()`
=======
| `shape()`     | Returns the tensor dimensions as `std::vector<size_t>`. |
| `size()`      | Returns the total number of elements.                   |
| `ndim()`      | Returns the number of dimensions.                       |
| `empty()`     | Checks whether the tensor contains no elements.         |
| `dtype()`     | Returns the underlying data type (`float`).             |
| `fill(value)` | Fills the entire tensor with a specified scalar value.  |

---

## Shape Operations

TinyTorch supports complex tensor shape transformations natively.

### `reShape()`
>>>>>>> origin/feature/final-demo

```cpp
Tensor reShape(const Shape& new_shape);
```

<<<<<<< HEAD
Returns a Tensor with a new shape while preserving the original element ordering.

The total number of elements must remain unchanged.

For example:

```text
[1, 2, 3, 4, 5, 6]
```

can be reshaped from:

```text
[2, 3]
```

to:

```text
[3, 2]
```

without changing the underlying element order.

---

## `Transpose()`
=======
Returns a new tensor with the requested shape while preserving the original underlying element order.

### `Transpose()`
>>>>>>> origin/feature/final-demo

```cpp
Tensor Transpose(const Shape& axis);
```

<<<<<<< HEAD
Supports true N-dimensional transposition using arbitrary axis permutations.

The implementation remaps multidimensional indices between the input and output Tensor layouts.

For example, a 3D Tensor can be transformed using an axis permutation such as:

```text
[0, 1, 2] → [1, 2, 0]
```

The transpose implementation was generalized from the original 2D-only implementation to support arbitrary Tensor ranks.

---

# Reduction Operations

The Tensor core provides mathematical reduction operations.
=======
Performs a true **N-dimensional transpose** by remapping tensor dimensions and elements according to the provided axis permutation.

Unlike a 2D-only transpose, this implementation supports arbitrary tensor ranks.

---

## Reduction Operations

TinyTorch provides standard mathematical reduction operations both globally and along specific dimensions.
>>>>>>> origin/feature/final-demo

### Summation

```cpp
<<<<<<< HEAD
sum()
sum(axis)
```

* `sum()` reduces the complete Tensor to a scalar.
* `sum(axis)` reduces along a specific dimension.
=======
tensor.sum();
tensor.sum(axis);
```

* `sum()` reduces the entire tensor to a scalar.
* `sum(axis)` reduces the tensor along the specified dimension.
>>>>>>> origin/feature/final-demo

### Mean

```cpp
<<<<<<< HEAD
mean()
mean(axis)
```

* `mean()` calculates the global mean.
* `mean(axis)` calculates the mean along a selected dimension.
=======
tensor.mean();
tensor.mean(axis);
```

* `mean()` computes the global mean.
* `mean(axis)` computes the mean along a specific dimension.
>>>>>>> origin/feature/final-demo

### Extrema

```cpp
<<<<<<< HEAD
max()
min()
```

These return the global maximum and minimum values of the Tensor.

---

# Arithmetic Operations

TinyTorch supports element-wise Tensor arithmetic and scalar arithmetic.

## Tensor-to-Tensor Operations

```text
+
-
*
/
```

Each operation produces a new Tensor without directly modifying the input tensors.

Example:

```cpp
Tensor result = a + b;
```

## Tensor-to-Scalar Operations

```cpp
tensor + scalar
tensor - scalar
tensor * scalar
tensor / scalar
```

Scalar operations use a dedicated implementation path to avoid unnecessary Tensor construction and broadcasting overhead.

## In-Place Operations

The following operators are available:

```text
+=
-=
*=
/=
```

These provide convenient in-place-style updates by assigning the computed result back to the Tensor.

---

# ⚡ Arithmetic Performance Optimization

The Tensor arithmetic implementation contains specialized execution paths designed to reduce unnecessary overhead.

## Same-Shape Fast Path

When two tensors have exactly the same shape, broadcasting is unnecessary.

Instead of performing multidimensional index calculations for every element, TinyTorch directly accesses the contiguous memory:

```cpp
data_[i]
other.data_[i]
resultData[i]
```

This avoids repeated calls to:

```text
unravel_index()
broadcast_index()
ravel_index()
```

inside the main computational loop.

The result is a much cheaper execution path for common element-wise operations.

---

## Parallel SIMD Execution

Operations that can safely execute without exceptions use OpenMP parallelization combined with SIMD vectorization.

Example:

```cpp
#pragma omp parallel for simd schedule(static)
```

This allows independent elements to be processed across multiple CPU threads while also allowing the compiler to generate SIMD instructions.

This approach is used for suitable operations including:

* Same-shape arithmetic.
* Tensor-scalar arithmetic.
* `clamp()`.

---

# Broadcasting

TinyTorch implements NumPy-style broadcasting for element-wise Tensor operations.

Broadcasting supports:

* Different Tensor ranks.
* Singleton dimensions.
* Automatic output shape calculation.
* Strict shape compatibility validation.

### Internal Helpers

The broadcasting implementation relies on helper functions such as:
=======
tensor.max();
tensor.min();
```

Returns the maximum or minimum value of the tensor.

---

## Arithmetic Operations

The Tensor core supports element-wise and scalar arithmetic operations.

All standard arithmetic operations return new Tensor instances without modifying the original operands.

### Tensor-to-Tensor Operations

```cpp
tensor_a + tensor_b;
tensor_a - tensor_b;
tensor_a * tensor_b;
tensor_a / tensor_b;
```

### Tensor-to-Scalar Operations

```cpp
tensor + scalar;
tensor - scalar;
tensor * scalar;
tensor / scalar;
```

Scalar operations are also supported in the reverse direction where applicable:

```cpp
scalar + tensor;
scalar - tensor;
scalar * tensor;
scalar / tensor;
```

### In-Place Operations

For memory-efficient updates:

```cpp
tensor += other;
tensor -= other;
tensor *= other;
tensor /= other;
```

---

## Broadcasting

TinyTorch implements **NumPy-style broadcasting** for element-wise operations.

### Capabilities

The broadcasting system supports:

* Automatic broadcast shape calculation.
* Operations between tensors with different numbers of dimensions.
* Broadcasting singleton dimensions with size `1`.
* Strict shape validation to detect incompatible operations.

### Internal Helpers

The broadcasting implementation relies on several helper functions:
>>>>>>> origin/feature/final-demo

```text
broadcast_shape()
broadcast_index()
compute_size()
unravel_index()
ravel_index()
```

<<<<<<< HEAD
These functions handle shape calculation and mapping between flattened memory offsets and multidimensional indices.

---

## Broadcasting Execution Path

When Tensor shapes differ, TinyTorch uses a separate broadcasting path.

For operations that do not require exception handling, the output elements can be calculated independently and therefore the broadcasting loop is parallelized using OpenMP:

```cpp
#pragma omp parallel for schedule(static)
```

Each iteration:

1. Converts the output flat index into a multidimensional index.
2. Maps the output index to the corresponding input indices.
3. Calculates the input flat offsets.
4. Performs the arithmetic operation.
5. Stores the result.

This preserves the flexibility of broadcasting while still allowing parallel execution.

---

# Division Safety

Division operations perform explicit zero-division validation.

For example:

```cpp
if (other.data_[i] == 0) {
    throw std::invalid_argument(
        "Division by zero in tensor-tensor operation."
    );
}
```

Because these operations may throw exceptions during element processing, the implementation intentionally uses a sequential path for the validation loop.

This keeps error handling deterministic and avoids exception-related problems inside OpenMP parallel regions.

The same principle is applied to:

```text
tensor / scalar
scalar / tensor
```

---

# Scalar Operations

Scalar operations have their own specialized execution path:

```cpp
apply_scalar_operation(...)
```

Instead of constructing a temporary Tensor and performing broadcasting, the scalar is directly applied to every element.

For example:

```cpp
tensor * scalar
```

can directly execute:

```cpp
resultData[i] = data_[i] * scalar;
```

The loop is parallelized and vectorized when safe:

```cpp
#pragma omp parallel for simd schedule(static)
```

This reduces temporary allocations and avoids unnecessary broadcasting logic.

---

# Mathematical Operations

The Tensor core provides several mathematical primitives used by the neural network and loss modules.

## `log()`

Computes the natural logarithm element-wise.

Invalid values are rejected:

```text
x <= 0
```

results in a `std::domain_error`.

The current implementation intentionally uses a sequential loop to preserve straightforward exception handling.

---

## `clamp()`

```cpp
Tensor clamp(value_type min_value, value_type max_value);
```

Restricts every element to the closed interval:

```text
[min_value, max_value]
```

The implementation validates the bounds and then performs the element-wise operation using OpenMP parallelization and SIMD:

```cpp
#pragma omp parallel for simd schedule(static)
```

This makes `clamp()` suitable for high-throughput element-wise processing.

---

# Matrix Multiplication

TinyTorch provides an N-dimensional `matmul()` implementation supporting:

* 2D matrix multiplication.
* Batched matrix multiplication.
* N-dimensional batch dimensions.
* Broadcasting of batch dimensions.
* Dynamic result shape generation.

The implementation uses:
=======
These functions handle shape compatibility, index mapping, and conversion between flat and multidimensional indices.

---

## Matrix Multiplication

TinyTorch provides a `matmul()` implementation supporting both standard and batched matrix multiplication.

### Supported Operations

* 2D matrix multiplication.
* N-dimensional batched matrix multiplication.
* Automatic broadcasting of batch dimensions.
* Dynamic generation of the resulting tensor shape.

### Helper Functions
>>>>>>> origin/feature/final-demo

```text
extract_batch_shape()
```

<<<<<<< HEAD
to separate batch dimensions from the final matrix dimensions.

---

# ⚡ Matrix Multiplication Optimization

Matrix multiplication is one of the most computationally expensive operations in the Tensor core, so it contains several CPU-level optimizations.

## Direct Flat Indexing

The implementation calculates the base memory offsets of each matrix slice before entering the main matrix multiplication loops.

This avoids repeatedly calling multidimensional indexing functions inside the expensive `O(N × M × K)` computation.

The inner computation can therefore use simple flat offsets such as:

```cpp
a_base + n * K + k
```

and:

```cpp
b_base + m * K + k
```

---

## Cache-Friendly Access

The right-hand matrix is transposed into a temporary contiguous buffer before the multiplication.

Instead of repeatedly accessing a column with non-contiguous memory jumps, the computation reads the transposed matrix sequentially.

Conceptually:

```text
B[k][m]
```

becomes:

```text
B_transposed[m][k]
```

This improves memory locality and reduces cache misses during the dot-product computation.

---

## OpenMP Parallelization

The outer matrix multiplication loops are parallelized using OpenMP:

```cpp
#pragma omp parallel for collapse(2) schedule(static)
```

The independent output elements can therefore be computed concurrently across available CPU cores.

---

## SIMD Vectorization

The innermost dot-product loop uses:

```cpp
#pragma omp simd
```

This gives the compiler an explicit opportunity to vectorize the multiplication and accumulation operations.

Combined with native compiler optimization flags such as:

```text
-O3
-march=native
```

the compiler can utilize SIMD instructions supported by the target CPU, such as AVX/AVX2 when available.

---

# Performance Benchmark

The optimized implementation was benchmarked using the Python frontend.

Each benchmark performs five iterations after a warmup pass.

## 128 × 128

```text
Matrix Size  : 128x128
Total Time   : 8.1362 seconds
Average Time : 1627.24 ms per operation
```

## 1024 × 1024

```text
Matrix Size  : 1024x1024
Total Time   : 0.9590 seconds
Average Time : 191.81 ms per operation
```

| Matrix Size   | Iterations | Total Time | Average Time |
| ------------- | ---------: | ---------: | -----------: |
| `128 × 128`   |          5 | `8.1362 s` | `1627.24 ms` |
| `1024 × 1024` |          5 | `0.9590 s` |  `191.81 ms` |

> **Benchmark Note:** These are measurements for different matrix sizes and therefore should not be interpreted as a direct before/after speedup comparison.

The benchmark can be executed using:

```bash
cd Code
python3 python/benchmark.py
```

---

# Element Access

TinyTorch supports multidimensional element access.

Example:

```cpp
tensor({i, j, k});
```

This allows an element to be accessed using an N-dimensional index.

Flat storage can also be accessed through:

```cpp
tensor[i]
```

which operates directly on the underlying contiguous storage.

---

# Operators

The Tensor class provides:

* Arithmetic operators.
* Scalar operators.
* In-place arithmetic operators.
* Equality comparison.
* Copy constructors.
* Move constructors.
* Copy assignment.
* Move assignment.
* Stream output using `operator<<`.

The equality operator performs a deep comparison of Tensor contents.

---

# Autograd Integration

The Tensor class directly stores the information required by the TinyTorch reverse-mode Autograd engine.
=======
The implementation separates batch dimensions from the final two matrix dimensions and performs matrix multiplication across the resulting batches.

---

## Element Access & Operators

### N-Dimensional Access

Tensor elements can be accessed or modified using multidimensional indices:

```cpp
tensor({i, j, k, ...});
```

For example:

```cpp
Tensor tensor({2, 3, 4});

float value = tensor({1, 2, 3});
tensor({1, 2, 3}) = 10.0f;
```

### Flat Access

Standard `operator[]` provides direct flat-memory access:

```cpp
tensor[index];
```

### Comparison

The Tensor class provides:

```cpp
operator==
```

for deep equality checks between tensors.

### Copy & Move Semantics

The Tensor implementation supports standard C++:

* Copy constructor.
* Move constructor.
* Copy assignment.
* Move assignment.

### Stream Output

The `std::ostream` operator (`<<`) is implemented for convenient tensor printing and debugging:

```cpp
std::cout << tensor << std::endl;
```

---

## Autograd Integration

The Tensor object directly stores the metadata required by TinyTorch's **reverse-mode automatic differentiation engine**.
>>>>>>> origin/feature/final-demo

### Gradient Tracking

```cpp
<<<<<<< HEAD
set_requires_grad(bool)
requires_grad()
```

These methods enable or disable gradient tracking.

### Computation Graph

Each Tensor can store a pointer to the function that generated it:

```text
grad_fn_
```

This allows the Autograd engine to traverse the computation graph during backpropagation.

### Gradient Storage

Gradients are stored through:

```text
grad_
```

and accumulated during the backward pass.

### Backpropagation

The Tensor interface provides:

```cpp
backward(gradient)
zero_grad()
```

`backward()` initiates reverse-mode automatic differentiation, while `zero_grad()` clears accumulated gradients before another training iteration.

---

# Neural Network Primitives

The Tensor core also provides mathematical primitives required by higher-level neural network components.

### Activations

```text
relu()
sigmoid()
tanh()
gelu()
softmax(dim)
```

### Loss Support

The Tensor core provides mathematical operations required by loss functions, including:

```text
log()
clamp()
log_softmax(dim)
```

These primitives are used by the higher-level loss implementations to build numerically stable training operations.

---

# Development Notes

## Generic N-Dimensional Transpose

The original transpose implementation only supported **2D tensors** and rejected higher-rank tensors.

The implementation was generalized to support arbitrary N-dimensional tensors using axis permutations and multidimensional index remapping.

---

## `ravel_index` Mapping Bug

The first generic transpose implementation used an incorrect argument order when calling `ravel_index()`.

This caused incorrect element placement after transposition.

The index mapping was corrected and validated using higher-dimensional Tensor cases.

---

## 3D Transpose Tests

The expected results of the 3D transpose tests were updated after verifying the correct N-dimensional index mapping.

This ensured that the tests matched the mathematical definition of axis permutation rather than the behavior of the previous implementation.

---

# Future Improvements

The Tensor core can be further improved through:

* Blocked / tiled matrix multiplication.
* Better cache-aware algorithms.
* Memory reuse and reduced temporary allocations.
* Further SIMD optimization.
* More efficient broadcasting and index mapping.
* GPU acceleration using CUDA.
* Chained indexing using proxy classes.

For example, chained indexing could eventually support:

```python
tensor[i][j][k]
```

using proxy objects.

---

# Summary

The TinyTorch Tensor core combines a flexible N-dimensional Tensor abstraction with optimized CPU execution.

Its main capabilities include:

* Dynamic N-dimensional storage.
* Shape manipulation.
* N-dimensional transpose.
* Broadcasting.
* Tensor and scalar arithmetic.
* Reduction operations.
* Mathematical primitives.
* Optimized matrix multiplication.
* OpenMP parallel execution.
* SIMD vectorization.
* Cache-friendly memory access.
* Autograd integration.
* Python interoperability through PyBind11.

The implementation is designed both as a functional Tensor backend and as an educational exploration of the techniques used inside modern deep learning frameworks.
=======
tensor.set_requires_grad(true);
tensor.requires_grad();
```

These methods enable or query gradient tracking for a tensor.

### Computation Graph

Each tensor can store the function node responsible for generating it:

```cpp
grad_fn_
```

This pointer connects the tensor to the corresponding node in the computation graph.

### Gradient Storage

Accumulated gradients are stored through:

```cpp
grad_
```

### Backpropagation

The Tensor class provides:

```cpp
tensor.backward(gradient);
```

which triggers the reverse-mode backward pass starting from the tensor.

### Clearing Gradients

```cpp
tensor.zero_grad();
```

Clears accumulated gradients before the next training iteration.

---

## Neural Network & Loss Primitives

Although activations and losses belong logically to higher-level modules, the Tensor core provides the fundamental mathematical operations required to implement them efficiently.

### Activation Functions

The Tensor API provides:

```cpp
tensor.relu();
tensor.sigmoid();
tensor.tanh();
tensor.gelu();
tensor.softmax(dim);
```

These operations form the mathematical building blocks for neural network layers.

### Loss & Numerical Operations

The Tensor core also provides:

```cpp
tensor.log();
tensor.clamp(min, max);
tensor.log_softmax(dim);
```

These operations are particularly important for implementing numerically stable loss functions such as **Cross Entropy Loss**.

---

## Summary

The Tensor core provides the fundamental numerical infrastructure for TinyTorch:

* Dynamic N-dimensional tensor representation.
* Shape manipulation and N-dimensional transpose.
* NumPy-style broadcasting.
* Element-wise and scalar arithmetic.
* Global and axis-based reductions.
* 2D and batched matrix multiplication.
* Multidimensional and flat element access.
* Reverse-mode Autograd integration.
* Activation and loss-related mathematical primitives.

Together, these features form the foundation on which the higher-level **Neural Network**, **Loss**, **Data Pipeline**, and **Autograd** components of TinyTorch are built.
>>>>>>> origin/feature/final-demo
