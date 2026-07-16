#pragma once

#include "tensor/tensor.h"
#include <memory>
#include <vector>

// ============================================================================
// Base Layer Class
// ============================================================================
class Layer {
  public:
    // Type aliases for cleaner code
    using ParamList = std::vector<Tensor*>;
    using LayerPtr = std::shared_ptr<Layer>;
    using LayerList = std::vector<LayerPtr>;

    virtual ~Layer() = default;

    // ================= Forward Pass ==========================
    [[nodiscard]] virtual Tensor forward(const Tensor& x, bool training = true) = 0;

    // ================= Callable Operator ==========================
    [[nodiscard]] Tensor operator()(const Tensor& x, bool training = true) {
        return forward(x, training);
    }

    // ================= Parameters ==========================
    [[nodiscard]] virtual ParamList parameters() {
        return {}; // Default for layers without parameters
    }
};
