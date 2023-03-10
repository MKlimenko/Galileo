#pragma once

#include "galileo.h"
#include <sycl/sycl.hpp>

namespace common {
	inline auto& GetQueue(GALILEO_QUEUE queue) {
		return *reinterpret_cast<sycl::queue*>(queue);
	}

	inline void* TypeErasedAllocate(sycl::queue& queue, GALILEO_DATA_TYPE data_type, unsigned int size) {
		switch (data_type) {
		case GALILEO_FLOAT:
			return sycl::malloc_shared<float>(size, queue);
		case GALILEO_DOUBLE:
			return sycl::malloc_shared<double>(size, queue);
		default:
			throw GALILEO_RESULT::GALILEO_RESULT_UNEXPECTED_DATA_TYPE;
		}
	}

	inline bool VerifyPtr(sycl::queue& queue, const void* ptr) {
		auto device = sycl::get_pointer_type(ptr, queue.get_context());

		return device == sycl::usm::alloc::shared;
	}
}