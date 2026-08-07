#include "bindings.h"

PYBIND11_MODULE(tinytorch, m) {
    m.doc() = "TinyTorch";

    bind_tensor(m);
    bind_layers(m);
    bind_data(m);
    bind_transforms(m);
}
