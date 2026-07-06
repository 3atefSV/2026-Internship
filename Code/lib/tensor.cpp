#include "tensor.h"
#include <stdexcept>

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

Tensor Tensor::operator+(const Tensor& other) const 
{
    if (shape != other.shape) 
    {
        throw std::invalid_argument("Shapes of tensors do not match for addition.");
    }

    std :: vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        resultData[i] = data[i] + other.data[i];
    }

    return Tensor(resultData, shape);
}
Tensor Tensor::operator-(const Tensor& other) const 
{
    if (shape != other.shape) 
    {
        throw std::invalid_argument("Shapes of tensors do not match for subtraction.");
    }

    std::vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        resultData[i] = data[i] - other.data[i];
    }

    return Tensor(resultData, shape);
}

Tensor Tensor::operator*(const Tensor& other) const 
{
    if (shape != other.shape) 
    {
        throw std::invalid_argument("Shapes of tensors do not match for multiplication.");
    }
    std::vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        resultData[i] = data[i] * other.data[i];
    }
    return Tensor(resultData, shape);
}

Tensor Tensor::operator/(const Tensor& other) const 
{
    if (shape != other.shape) 
    {
        throw std::invalid_argument("Shapes of tensors do not match for division.");
    }
    std::vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        if (other.data[i] == 0) 
        {
            throw std::invalid_argument("Division by zero in tensor division.");
        }
        resultData[i] = data[i] / other.data[i];
    }
    return Tensor(resultData, shape);
}

Tensor Tensor::operator+(const float scalar) const 
{
    std::vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        resultData[i] = data[i] + scalar;
    }
    return Tensor(resultData, shape);
}

Tensor Tensor::operator-(const float scalar) const 
{
    std::vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        resultData[i] = data[i] - scalar;
    }
    return Tensor(resultData, shape);
}

Tensor Tensor::operator*(const float scalar) const 
{
    std::vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        resultData[i] = data[i] * scalar;
    }
    return Tensor(resultData, shape);
}

Tensor Tensor::operator/(const float scalar) const 
{
    if (scalar == 0) 
    {
        throw std::invalid_argument("Division by zero in tensor-scalar division.");
    }
    std::vector<float> resultData(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        resultData[i] = data[i] / scalar;
    }
    return Tensor(resultData, shape);
}
