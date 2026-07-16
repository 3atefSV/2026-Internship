#pragma once

#include "layer/layer.h"
#include "tensor/tensor.h"
#include <optional>
#include <random>

// ============================================================================
// Dropout Layer
// ============================================================================
class Dropout : public Layer {
  public:
    // ================= Constructors =================
    // Added optional seed parameter for deterministic dropout masking
    explicit Dropout(Tensor::value_type p = 0.5f, std::optional<unsigned int> seed = std::nullopt);
    ~Dropout() override = default;

    // ================= Core Operations =================
    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;

    // ================= Properties ===================
    [[nodiscard]] Tensor::value_type p() const noexcept { return p_; }

  private:
    Tensor::value_type p_;
    std::mt19937 gen_; // RNG as a member so its state persists across forward passes!
};