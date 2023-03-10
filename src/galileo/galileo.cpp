#include "common.hpp"

#include <complex>

GALILEO_RESULT GALILEO_GetLibVersion(unsigned int* major, unsigned int* minor, unsigned int* patch) {
	if (!major || !minor || !patch)
		return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER;
	
    *major = GALILEO_VERSION_MAJOR;
    *minor = GALILEO_VERSION_MINOR;
	*patch = GALILEO_VERSION_PATCH;
    return GALILEO_RESULT::GALILEO_RESULT_OK;
}

GALILEO_RESULT GALILEO_GetQueueSize(unsigned int* size) {
	if (!size)
		return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER;
	
	*size = sizeof(sycl::queue);
	return GALILEO_RESULT::GALILEO_RESULT_OK;
}
GALILEO_RESULT GALILEO_InitQueue(GALILEO_QUEUE queue) {
	if (!queue)
		return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER;
	
	new (queue) sycl::queue();
	return GALILEO_RESULT::GALILEO_RESULT_OK;	
}

GALILEO_RESULT GALILEO_ReleaseQueue(GALILEO_QUEUE queue) {
	if (!queue)
		return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER;
	
	using T = sycl::queue;
	auto& typed_queue = common::GetQueue(queue);
	typed_queue.wait();
	typed_queue.~T();
	return GALILEO_RESULT::GALILEO_RESULT_OK;	
}

GALILEO_RESULT GALILEO_Allocate(GALILEO_QUEUE queue, GALILEO_DATA_TYPE data_type, unsigned size, void** ptr) {
	try {
		if (!queue|| !ptr)
			return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER;
		*ptr = common::TypeErasedAllocate(common::GetQueue(queue), data_type, size);
		return GALILEO_RESULT::GALILEO_RESULT_OK;
	}
	catch(GALILEO_RESULT result) {
		return result;
	}
	catch (...) {
		return GALILEO_RESULT::GALILEO_RESULT_UNKNOWN_ERROR;
	}
}

GALILEO_RESULT GALILEO_Deallocate(GALILEO_QUEUE queue, void* ptr) {
	if (!queue|| !ptr)
		return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER;

	sycl::free(ptr, common::GetQueue(queue));
	return GALILEO_RESULT::GALILEO_RESULT_OK;
}
