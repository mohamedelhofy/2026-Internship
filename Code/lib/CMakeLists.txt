add_library(Tensor
<<<<<<< HEAD
    tensor_arithmetic.cpp
    tensor_access.cpp
    tensor_matrix.cpp
    tensor_broadcast.cpp
    tensor_properties.cpp
=======
    tensor/tensor_arithmetic.cpp
    tensor/tensor_access.cpp
    tensor/tensor_matrix.cpp
    tensor/tensor_broadcast.cpp
    tensor/tensor_properties.cpp
    tensor/tensor_reduction.cpp
    tensor/tensor_shapeOps.cpp
    tensor/Tensor_ActivationOps.cpp
    activations/activations.cpp
    layer/linear.cpp
    layer/dropout.cpp
    layer/sequential.cpp
    losses/loss.cpp
    losses/mse_loss.cpp
    losses/cross_entropy_loss.cpp
    losses/binary_cross_entropy_loss.cpp
    data/tensor_dataset.cpp
    data/transforms.cpp
    data/dataloader.cpp
    data/image_dataset.cpp
    tensor/tensor_autograd.cpp
    autograd/function.cpp
    autograd/loss_function.cpp
    optimizers/optimizer.cpp
    optimizers/sgd.cpp
    optimizers/adam.cpp
    optimizers/adamw.cpp
    training/trainer.cpp
    training/scheduler.cpp
    training/clip_grad.cpp
>>>>>>> origin/feature/final-demo
)
set_target_properties(Tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(Tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
<<<<<<< HEAD
)
=======
        ${PROJECT_SOURCE_DIR}/include/tensor
        ${PROJECT_SOURCE_DIR}/include/activations
        ${PROJECT_SOURCE_DIR}/include/layer
        ${PROJECT_SOURCE_DIR}/include/losses
        ${PROJECT_SOURCE_DIR}/include/data
        ${PROJECT_SOURCE_DIR}/include/autograd
        ${PROJECT_SOURCE_DIR}/include/optimizers
        ${PROJECT_SOURCE_DIR}/include/training
)

set_target_properties(Tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
>>>>>>> origin/feature/final-demo
