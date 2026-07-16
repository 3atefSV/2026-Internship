#pragma once

#include "layer/layer.h"
#include "tensor/tensor.h"
#include <optional>
// ============================================================================
// Linear Layer (y = xW + b)
// ============================================================================
class Linear : public Layer {
  public:
    // ================= Constructors =================
    // Added optional seed parameter for deterministic initialization
    Linear(Tensor::size_type in_features, Tensor::size_type out_features, bool bias = true,
           std::optional<unsigned int> seed = std::nullopt);
    ~Linear() override = default;

    // ================= Core Operations =================
    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;
    [[nodiscard]] ParamList parameters() override;

    // ================= Properties ===================
    [[nodiscard]] Tensor::size_type in_features() const noexcept { return in_features_; }
    [[nodiscard]] Tensor::size_type out_features() const noexcept { return out_features_; }
    [[nodiscard]] bool has_bias() const noexcept { return use_bias_; }

  private:
    Tensor::size_type in_features_;
    Tensor::size_type out_features_;
    bool use_bias_;
    Tensor weight_;
    Tensor bias_;
};
