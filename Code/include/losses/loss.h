#pragma once

#include "tensor/tensor.h"
#include <string_view>

// Controls how the per-element (or per-sample) losses are collapsed into the final scalar valuess
enum class Reduction {
    Mean, // average of all values
    Sum,  // sum of all values
    None  // no reduction at all
};

class Loss {
  public:
    explicit Loss(Reduction reduction = Reduction::Mean) : reduction_(reduction) {}

    Loss(const Loss&) = default;

    Loss& operator=(const Loss&) = default;

    virtual ~Loss() = default;

    [[nodiscard]] virtual Tensor forward(const Tensor& prediction, const Tensor& target) const = 0;

    [[nodiscard]] Tensor operator()(const Tensor& prediction, const Tensor& target) const {
        return forward(prediction, target);
    }

    [[nodiscard]] Reduction reduction() const noexcept { return reduction_; }

  protected:
    [[nodiscard]] Tensor reduce(const Tensor& values) const;

    static void check_same_shape(const Tensor& prediction, const Tensor& target, const char* name);

    static void check_not_empty(const Tensor& tensor, std::string_view name);

  private:
    Reduction reduction_;
};
