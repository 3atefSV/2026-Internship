#pragma once

#include "layer/layer.h"
#include "tensor/tensor.h"
#include <initializer_list>
#include <memory>
#include <vector>

// ============================================================================
// Sequential Container
// ============================================================================
class Sequential : public Layer {
  public:
    // ================= Constructors =================
    Sequential() = default;
    Sequential(std::initializer_list<LayerPtr> layers);
    ~Sequential() override = default;

    // ================= Container Operations =================
    void add(LayerPtr layer);
    [[nodiscard]] bool empty() const noexcept { return layers_.empty(); }
    [[nodiscard]] Tensor::size_type size() const noexcept { return layers_.size(); }

    // ================= Core Operations =================
    [[nodiscard]] Tensor forward(const Tensor& x, bool training = true) override;
    [[nodiscard]] ParamList parameters() override;

  private:
    LayerList layers_;
};