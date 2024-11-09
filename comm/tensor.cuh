#pragma once

#include "assert.h"
// #include "spdlog/spdlog.h"
#include <algorithm>
#include <array>
#include <concepts> // For std::integral
#include <iostream>
#include <iterator>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Enums using enum class with string conversion functions
enum class PllmLayout {
	ROW_MAJOR,
	COL_MAJOR
};

inline std::string toString(PllmLayout layout) {
    switch (layout) {
        case PllmLayout::ROW_MAJOR: return "ROW_MAJOR";
        case PllmLayout::COL_MAJOR: return "COL_MAJOR";
        default: return "UNKNOWN_LAYOUT";
	}
}

enum class PllmDimension {
	ROW,
	COLUMN
};

inline std::string toString(PllmDimension dim) {
    switch (dim) {
        case PllmDimension::ROW: return "ROW";
        case PllmDimension::COLUMN: return "COLUMN";
        default: return "UNKNOWN_DIMENSION";
	}
}



template<typename T>
struct pllmTensor {
	T* ptr;
	size_t dim1;
	size_t dim2;
	size_t& dimC;
	PllmLayout layout;

	pllmTensor()
		: ptr(nullptr)
		, dim1(0)
		, dim2(0)
		, dimC(this->dim2)
		, layout(PllmLayout::ROW_MAJOR) { }
	pllmTensor(T* ptr, size_t dim1, size_t dim2, PllmLayout layout)
		: ptr(ptr)
		, dim1(dim1)
		, dim2(dim2)
		, dimC(this->dim2)
		, layout(layout) { }
	pllmTensor(T* ptr, int dim1, int dim2, PllmLayout layout)
		: ptr(ptr)
		, dim1(dim1)
		, dim2(dim2)
		, dimC(this->dim2)
		, layout(layout) { 
		assert(dim1 >= 0 && dim2 >= 0);
		}

	pllmTensor(T* ptr, size_t length)
		: ptr(ptr)
		, dim1(length)
		, dim2(1)
		, dimC(this->dim2)
		, layout(PllmLayout::ROW_MAJOR) {}
	
	pllmTensor(T* ptr, int length)
		: ptr(ptr)
		, dim1(length)
		, dim2(1)
		, dimC(this->dim2)
		, layout(PllmLayout::ROW_MAJOR) {
		assert(length >= 0);
	}

	size_t size() const {
		return static_cast<size_t>(dim1) * dim2;
	}
	size_t bytes() const {
		return size() * sizeof(T);
	}

	pllmTensor& operator=(const pllmTensor& other) {
		if(this == &other) {
			return *this;
		}

		ptr = other.ptr;
		dim1 = other.dim1;
		dim2 = other.dim2;
		layout = other.layout;

		return *this;
	}

	inline void clearContent() const {
		cudaMemset(ptr, 0, bytes());
	}
};
