#include "layer/sequential.h"

// ============================================================================
// Sequential Container Implementation
// ============================================================================
Sequential::Sequential(std::initializer_list<LayerPtr> layers) : layers_(layers) {}

void Sequential::add(LayerPtr layer) { layers_.push_back(std::move(layer)); }

Tensor Sequential::forward(const Tensor& x, bool training) {
    Tensor current_output = x;
    for (const auto& layer : layers_) {
        current_output = layer->forward(current_output, training);
    }
    return current_output;
}

Layer::ParamList Sequential::parameters() {
    ParamList all_params;
    for (const auto& layer : layers_) {
        ParamList layer_params = layer->parameters();
        all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
    }
    return all_params;
}
