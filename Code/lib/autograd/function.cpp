#include "autograd/function.h"

// ================= AddBackward =================
AddBackward::AddBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> AddBackward::apply(const Tensor& grad_output) {
    return {grad_output, grad_output}; // ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
}

// ================= SubBackward =================
SubBackward::SubBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> SubBackward::apply(const Tensor& grad_output) {
    return {grad_output, grad_output * -1.0f}; // ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
}

// ================= MulBackward =================
MulBackward::MulBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> MulBackward::apply(const Tensor& grad_output) {
    Tensor grad_a = grad_output * saved_tensors[1]; // ∂(a*b)/∂a = b
    Tensor grad_b = grad_output * saved_tensors[0]; // ∂(a*b)/∂b = a
    return {grad_a, grad_b};
}

// ================= DivBackward =================
DivBackward::DivBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> DivBackward::apply(const Tensor& grad_output) {
    Tensor b = saved_tensors[1];
    Tensor grad_a = grad_output / b;                                    // ∂(a/b)/∂a = 1/b
    Tensor grad_b = (grad_output * saved_tensors[0] * -1.0f) / (b * b); // ∂(a/b)/∂b = -a/b^2
    return {grad_a, grad_b};
}
