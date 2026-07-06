#pragma once

#include <vector>
#include <string>
#include <cstddef>

class Tensor { 
    public:
        // constructor
        Tensor(); // default constructor
        Tensor(const std::vector<float>& data, const std::vector<size_t>& shape); // constructor with data and shape

        // properties
        const std::vector<size_t>& getShape() const; // get the shape of the tensor
        const std::vector<float>& getData() const; // get the data of the tensor
        size_t getSize() const; // get the size of the tensor (number of elements)
        size_t getNumDimensions() const; // get the number of dimensions of the tensor
        std::string dtype() const; // get the data type of the tensor (currently only float is supported)
        bool isEmpty() const; // check if the tensor is empty
        
        // Arithmetic operations
        Tensor operator+(const Tensor& other) const; // element-wise addition
        Tensor operator+(const float scalar) const; // add a scalar to each element of the tensor

    private:
        std::vector<float> data; // data of the tensor
        std::vector<size_t> shape; // shape of the tensor

};
