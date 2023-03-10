#include "galileo.h"

#include <cstddef>
#include <memory>
#include <span>

int main() {
	unsigned int queue_size = 0;
	GALILEO_GetQueueSize(&queue_size);

	auto queue_ptr = std::unique_ptr<std::byte[]>(new std::byte[queue_size]);
	GALILEO_InitQueue(queue_ptr.get());

	void* ptr = nullptr;
	GALILEO_Allocate(queue_ptr.get(), GALILEO_DATA_TYPE::GALILEO_FLOAT, 1024, &ptr);
	auto data = std::span(reinterpret_cast<float*>(ptr), 1024);

	GALILEO_Abs(queue_ptr.get(), ptr, GALILEO_FLOAT, 1024, ptr, GALILEO_FLOAT);
	float qq[1024];
	GALILEO_Abs(queue_ptr.get(), qq, GALILEO_FLOAT, 1024, ptr, GALILEO_FLOAT);

	GALILEO_Deallocate(queue_ptr.get(), ptr);

	GALILEO_ReleaseQueue(queue_ptr.get());
	return 0;
}