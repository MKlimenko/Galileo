#pragma once

#include "galileo.h"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <numeric>

namespace common {
	inline auto& GetQueue(GALILEO_QUEUE queue) {
		return *reinterpret_cast<sycl::queue*>(queue);
	}

	inline void* TypeErasedAllocate(sycl::queue& queue, GALILEO_DATA_TYPE data_type, unsigned int size) {
		switch (data_type) {
		case GALILEO_UINT8:
			return sycl::malloc_shared<std::uint8_t>(size, queue);
		case GALILEO_UINT16:
			return sycl::malloc_shared<std::uint16_t>(size, queue);
		case GALILEO_UINT32:
			return sycl::malloc_shared<std::uint32_t>(size, queue);
		case GALILEO_UINT64:
			return sycl::malloc_shared<std::uint64_t>(size, queue);
		case GALILEO_INT8:
			return sycl::malloc_shared<std::int8_t>(size, queue);
		case GALILEO_INT16:
			return sycl::malloc_shared<std::int16_t>(size, queue);
		case GALILEO_INT32:
			return sycl::malloc_shared<std::int32_t>(size, queue);
		case GALILEO_INT64:
			return sycl::malloc_shared<std::int64_t>(size, queue);
		case GALILEO_FLOAT:
			return sycl::malloc_shared<float>(size, queue);
		case GALILEO_DOUBLE:
			return sycl::malloc_shared<double>(size, queue);
		case GALILEO_HALF:
			return sycl::malloc_shared<sycl::half>(size, queue);
		case GALILEO_BFLOAT16:
			return sycl::malloc_shared<sycl::ext::oneapi::experimental::bfloat16>(size, queue);
		default:
			throw GALILEO_RESULT::GALILEO_RESULT_UNEXPECTED_DATA_TYPE;
		}
	}

	inline bool VerifyPtr(sycl::queue& queue, const void* ptr) {
		auto device = sycl::get_pointer_type(ptr, queue.get_context());

		return device == sycl::usm::alloc::shared;
	}

	template <typename ... Args>
	bool VerifyQueryPtrs(const GALILEO_TENSOR& lhs, const GALILEO_TENSOR& rhs, Args&& ...args) {
		auto is_same_queue = lhs.associated_queue == rhs.associated_queue;
		if constexpr (sizeof...(Args) > 0) {
			is_same_queue &= VerifyQueryPtrs(rhs, args...);
		}
		return is_same_queue;
	}

	template <typename ... Args>
	bool VerifyDimensionsPtrs(const GALILEO_TENSOR& lhs, const GALILEO_TENSOR& rhs, Args&& ...args) {
		const auto& lhs_dimensions = lhs.dimensions;
		const auto& rhs_dimensions = rhs.dimensions;
		auto is_same = lhs_dimensions.tensor_dimensions_size == rhs_dimensions.tensor_dimensions_size;
		if (!is_same)
			return false;
		is_same &= std::equal(lhs_dimensions.tensor_dimensions, lhs_dimensions.tensor_dimensions + lhs_dimensions.tensor_dimensions_size,
			rhs_dimensions.tensor_dimensions);

		if constexpr (sizeof...(Args) > 0) {
			is_same &= VerifyDimensionsPtrs(rhs, args...);
		}
		return is_same;
	}

	inline unsigned int GetTotalSize(const GALILEO_TENSOR_DIMENSIONS& dimensions) {
		return std::accumulate(dimensions.tensor_dimensions,
			dimensions.tensor_dimensions + dimensions.tensor_dimensions_size, 0);
	}
}

