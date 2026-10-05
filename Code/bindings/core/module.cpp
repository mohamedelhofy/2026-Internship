#include "bindings.h"

PYBIND11_MODULE(tinytorch, m) {
    m.doc() = "TinyTorch";

    bind_tensor(m);
    bind_layers(m);
    bind_losses(m);
    bind_data(m);
    bind_transforms(m);
    bind_dataloader(m);
    bind_optimizers(m);
    bind_training(m);
}
