#include "optimizers/optimizer.h"

#include <cstddef>

Optimizer::Optimizer(const ParamList& params) : params_(params) {}

void Optimizer::zero_grad() {
    for (Tensor* param : params_) {
        if (param && param->grad_) {
            // Zero the existing gradient buffer in place so that no memory is
            // freed and re-allocated on every mini-batch.
            std::fill(param->grad_->data().begin(), param->grad_->data().end(), 0.0f);
        }
    }
}
