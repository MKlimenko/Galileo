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

	auto in_out_tensor = GALILEO_TENSOR();
	GALILEO_Create1dTensor(queue_ptr.get(), ptr, GALILEO_FLOAT, 1024, &in_out_tensor);
	GALILEO_Abs(&in_out_tensor, &in_out_tensor);
	auto incorrect = in_out_tensor;
	float qq[1024];
	incorrect.tensor_data = qq;
	GALILEO_Abs(&incorrect, &in_out_tensor);

	GALILEO_Deallocate(queue_ptr.get(), ptr);

	GALILEO_ReleaseQueue(queue_ptr.get());
	return 0;
}