#include "gtest/gtest.h"
#include "galileo.h"
#include "unary.hpp"

#include <sycl/sycl.hpp>

inline namespace helpers {
	auto GetQueue() {
		unsigned int size = 0;
		auto result = GALILEO_GetQueueSize(&size);
		if (result != GALILEO_RESULT::GALILEO_RESULT_OK)
			throw result;

		auto queue_ptr = std::unique_ptr<std::byte[], std::function<GALILEO_RESULT(GALILEO_QUEUE)>>(new std::byte[size], GALILEO_ReleaseQueue);
		result = GALILEO_InitQueue(queue_ptr.get());
		if (result != GALILEO_RESULT::GALILEO_RESULT_OK)
			throw result;
		return queue_ptr;
	}
}

TEST(InfrastructureTests, GetLibVersion) {
	unsigned int major = 0;
	unsigned int minor = 0;
	unsigned int patch = 0;
	auto result = GALILEO_GetLibVersion(&major, &minor, &patch);
	ASSERT_EQ(result, GALILEO_RESULT::GALILEO_RESULT_OK);
	ASSERT_EQ(major, GALILEO_VERSION_MAJOR);
	ASSERT_EQ(minor, GALILEO_VERSION_MINOR);
	ASSERT_EQ(patch, GALILEO_VERSION_PATCH);
}

TEST(InfrastructureTests, GetQueueSize) {
	unsigned int size = 0;
	auto result = GALILEO_GetQueueSize(&size);
	ASSERT_EQ(result, GALILEO_RESULT::GALILEO_RESULT_OK);
	ASSERT_EQ(size, sizeof(sycl::queue));
}

TEST(InfrastructureTests, InitAndReleaseQueue) {
	auto queue_ptr = GetQueue();
}

TEST(InfrastructureTests, AllocateDeallocate) {
	auto queue_ptr = GetQueue();
	constexpr auto size = 1024;
	for (auto type = static_cast<int>(GALILEO_UINT8); type <= static_cast<int>(GALILEO_COMPLEX_HALF); ++type) {
		void* ptr = nullptr;
		auto result = GALILEO_Allocate(queue_ptr.get(), static_cast<GALILEO_DATA_TYPE>(type), size, &ptr);
		ASSERT_EQ(result, GALILEO_RESULT::GALILEO_RESULT_OK);
		result = GALILEO_Deallocate(queue_ptr.get(), ptr);
		ASSERT_EQ(result, GALILEO_RESULT::GALILEO_RESULT_OK);
	}
}
