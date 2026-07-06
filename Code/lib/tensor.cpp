#include "tensor.h"

Tensor::Tensor() 
{
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape) 
    : data(data), shape(shape) 
{
}

const std::vector<size_t>& Tensor::getShape() const 
{
    return shape;
}

const std::vector<float>& Tensor::getData() const 
{
    return data;
}

size_t Tensor::getSize() const 
{
    return data.size(); 
}

size_t Tensor::getNumDimensions() const 
{
    return shape.size(); 
}

std::string Tensor::dtype() const 
{
    return "float"; // currently only float is supported
}

bool Tensor::isEmpty() const 
{
    return data.empty(); 
}
