#pragma once

#include <cstddef>
#include <string>
#include <vector>

class Tensor {
public:
    // Type aliases for convenience
    using value_type = float;
    using size_type  = std::size_t;
    using Shape      = std::vector<size_type>;
    using Storage    = std::vector<float>;
    
    // constructor
    Tensor() = default; // default constructor
    Tensor(const Storage &data, const Shape &shape); // constructor with data and shape

    // properties
    const Shape &getShape() const; // get the shape of the tensor
    const Storage &getData() const;   // get the data of the tensor
    size_t getSize() const; // get the size of the tensor (number of elements)
    size_t getNumDimensions() const; // get the number of dimensions of the tensor
    std::string dtype() const; // get the data type of the tensor (currently only
                                // float is supported)
    bool isEmpty() const;      // check if the tensor is empty

    // Arithmetic operations with Tensor
    Tensor operator+(const Tensor &other) const; // element-wise addition
    Tensor operator-(const Tensor &other) const; // element-wise subtraction
    Tensor operator*(const Tensor &other) const; // element-wise multiplication
    Tensor operator/(const Tensor &other) const; // element-wise division

    // Arithmetic operations with scalar
    Tensor operator+(
        const value_type scalar) const; // add a scalar to each element of the tensor
    Tensor operator-(const value_type scalar)
        const; // subtract a scalar from each element of the tensor
    Tensor operator*(const value_type scalar)
        const; // multiply each element of the tensor by a scalar
    Tensor operator/(const value_type scalar)
        const; // divide each element of the tensor by a scalar

    // element access (2d)
    value_type &operator()(size_type row,
                        size_type col); // access element by index (non-const)
    const value_type &operator()(size_type row,
                            size_type col) const; // for const access  function

    // matrix multiplication
    Tensor matmul(const Tensor &other) const;

    private:
    Storage data_;   // data of the tensor
    Shape shape_; // shape of the tensor
};
