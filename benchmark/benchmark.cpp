#include <benchmark/benchmark.h>
#include "galileo.h"
#include "../external/type_map/include/type_map.hpp"

#include <complex>
#include <memory>

namespace helper {
	using type_map = mk::TypeMap<
		mk::TypeValuePair<std::uint8_t, GALILEO_DATA_TYPE::GALILEO_UINT8>,
		mk::TypeValuePair<std::uint16_t, GALILEO_DATA_TYPE::GALILEO_UINT16>,
		mk::TypeValuePair<std::uint32_t, GALILEO_DATA_TYPE::GALILEO_UINT32>,
		mk::TypeValuePair<std::uint64_t, GALILEO_DATA_TYPE::GALILEO_UINT64>,
		mk::TypeValuePair<std::int8_t, GALILEO_DATA_TYPE::GALILEO_INT8>,
		mk::TypeValuePair<std::int16_t, GALILEO_DATA_TYPE::GALILEO_INT16>,
		mk::TypeValuePair<std::int32_t, GALILEO_DATA_TYPE::GALILEO_INT32>,
		mk::TypeValuePair<std::int64_t, GALILEO_DATA_TYPE::GALILEO_INT64>,
		mk::TypeValuePair<float, GALILEO_DATA_TYPE::GALILEO_FLOAT>,
		mk::TypeValuePair<double, GALILEO_DATA_TYPE::GALILEO_DOUBLE>,
		//mk::TypeValuePair<std::uint8_t, GALILEO_DATA_TYPE::GALILEO_HALF>,
		//mk::TypeValuePair<std::uint8_t, GALILEO_DATA_TYPE::GALILEO_BFLOAT16>,
		mk::TypeValuePair<std::complex<float>, GALILEO_DATA_TYPE::GALILEO_COMPLEX_FLOAT>,
		mk::TypeValuePair<std::complex<double>, GALILEO_DATA_TYPE::GALILEO_COMPLEX_DOUBLE>
		//mk::TypeValuePair<std::uint8_t, GALILEO_DATA_TYPE::GALILEO_COMPLEX_HALF>
	>;

	auto GetQueue() {
		unsigned int queue_size = 0;
		GALILEO_GetQueueSize(&queue_size);

		auto queue_ptr = std::unique_ptr<std::byte[], GALILEO_RESULT(*)(void*)>(new std::byte[queue_size], GALILEO_ReleaseQueue);
		GALILEO_InitQueue(queue_ptr.get());

		return queue_ptr;
	}

	template <typename T>
	auto GetTensor(const GALILEO_QUEUE& queue, std::size_t size) {
		void* ptr = nullptr;
		GALILEO_Allocate(queue, type_map::GetValueByType<T>(), size, &ptr);

		auto deleter = [queue](GALILEO_TENSOR* tensor) {
			return GALILEO_Deallocate(queue, tensor->tensor_data);
		};
		auto tensor = std::unique_ptr<GALILEO_TENSOR, decltype(deleter)>(new GALILEO_TENSOR(), deleter);
		GALILEO_Create1dTensor(queue, ptr, type_map::GetValueByType<T>(), size, tensor.get());
		return tensor;
	}
}
#include <iostream>
#define UNARY_BENCHMARK_OPTIONS RangeMultiplier(2)->Range(16, 2048 << 6)->Complexity()
template <typename T, typename U>
static void ExpBenchmark(benchmark::State& state) {
	auto queue = helper::GetQueue();
	auto input = helper::GetTensor<T>(queue.get(), state.range());
	auto output = helper::GetTensor<U>(queue.get(), state.range());
    for (auto _ : state) {
		auto result = GALILEO_Exp(input.get(), output.get());
		if (result != GALILEO_RESULT_OK) {
			std::cout << "!!!!!!!!!!!!" << (int)result << std::endl;
			//throw std::runtime_error("Oops");
			break;
		}
        //ugsdr::SequentialMixer::Translate(input, 100.0, 1.0);
        //benchmark::DoNotOptimize(input);
    }
    state.SetComplexityN(state.range());
}
BENCHMARK_TEMPLATE(ExpBenchmark, float, float)->UNARY_BENCHMARK_OPTIONS;
BENCHMARK_TEMPLATE(ExpBenchmark, int, int)->UNARY_BENCHMARK_OPTIONS;
BENCHMARK_TEMPLATE(ExpBenchmark, double, double)->UNARY_BENCHMARK_OPTIONS;

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    system("pause");
    return 0;
}