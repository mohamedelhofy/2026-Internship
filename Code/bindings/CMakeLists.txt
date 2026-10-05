include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(tinytorch
<<<<<<< HEAD
    bindings.cpp
=======
    losses/loss.cpp
    core/module.cpp
    core/tensor.cpp
    layer/layer.cpp
    data/data.cpp
    data/dataloader.cpp
    transforms/transforms.cpp
    optimizers/optimizers.cpp
    training/trainer.cpp
>>>>>>> origin/feature/final-demo
)

target_link_libraries(tinytorch PRIVATE Tensor)

target_include_directories(tinytorch PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)