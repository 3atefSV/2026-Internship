#include "layer/dropout.h"
#include <stdexcept>

// ============================================================================
// Dropout Layer Implementation
// ============================================================================
Dropout::Dropout(Tensor::value_type p, std::optional<unsigned int> seed) : p_(p) {
    if (p_ < 0.0f || p_ >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in range [0, 1).");
    }
    
    // Initialize RNG once in the constructor
    if (seed.has_value()) {
        gen_.seed(seed.value());
    } else {
        std::random_device rd;
        gen_.seed(rd());
    }
}

Tensor Dropout::forward(const Tensor& x, bool training) {
    if (!training || p_ == 0.0f) {
        return x;
    }

    Tensor::value_type keep_prob = 1.0f - p_;
    Tensor::value_type scale_factor = 1.0f / keep_prob;

    std::uniform_real_distribution<Tensor::value_type> dist(0.0f, 1.0f);

    Tensor result(x.shape());
    
    for (Tensor::size_type i = 0; i < x.size(); ++i) {
        // Use the member 'gen_' so the sequence continues across calls
        if (dist(gen_) < keep_prob) {
            result.data()[i] = x.data()[i] * scale_factor;
        } else {
            result.data()[i] = 0.0f;
        }
    }

    return result;
}