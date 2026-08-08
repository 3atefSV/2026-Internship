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

// ================= MatmulBackward =================
MatmulBackward::MatmulBackward(const Tensor& a, const Tensor& b) { saved_tensors = {a, b}; }
std::vector<Tensor> MatmulBackward::apply(const Tensor& grad_output) {
    // For 2D matrices: ∂(A@B)/∂A = grad @ B.T, ∂(A@B)/∂B = A.T @ grad
    Tensor a_T = saved_tensors[0].Transpose({1, 0}); // Transpose 2D
    Tensor b_T = saved_tensors[1].Transpose({1, 0}); // Transpose 2D

    Tensor grad_a = grad_output.matmul(b_T); 
    Tensor grad_b = a_T.matmul(grad_output); 
    return {grad_a, grad_b};
}

// ================= ReshapeBackward =================
ReshapeBackward::ReshapeBackward(const Tensor& a) { saved_tensors = {a}; }
std::vector<Tensor> ReshapeBackward::apply(const Tensor& grad_output) {
    // ∂(X.reshape)/∂X = grad.reshape(X.shape)
    return {grad_output.reShape(saved_tensors[0].shape())};
}

// ================= TransposeBackward =================
TransposeBackward::TransposeBackward(const Tensor& a, const Tensor::Shape& axes) : axes_(axes) { 
    saved_tensors = {a}; 
}
std::vector<Tensor> TransposeBackward::apply(const Tensor& grad_output) {
    // For a simple 2D transpose {1, 0}, applying it again reverts it. 
    return {grad_output.Transpose(axes_)};
}
