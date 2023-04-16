#pragma once

#include "galileo.h"
#define SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <variant>

#include "../../external/type_map/include/type_map.hpp"

namespace galileo::common {
	#define CONSTIFY(T) std::conditional_t<is_const, const T, T>

	template <typename T>
	using complex = sycl::ext::oneapi::experimental::complex<T>;

	template <bool is_const, typename ... Args>
	constexpr std::variant<CONSTIFY(Args)*...> GetVariantFromTuple(std::tuple<Args...> t);

	template <bool is_const, typename ... Args>
	constexpr std::variant<CONSTIFY(Args)*...> GetPairVariantFromTuple(std::tuple<Args...> t);

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
		case GALILEO_COMPLEX_FLOAT:
			return sycl::malloc_shared<complex<float>>(size, queue);
		case GALILEO_COMPLEX_DOUBLE:
			return sycl::malloc_shared<complex<double>>(size, queue);
		case GALILEO_COMPLEX_HALF:
			return sycl::malloc_shared<complex<sycl::half>>(size, queue);
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

	template <typename From, typename To>
	concept is_narrowing_conversion = !requires(From from) {
		To{ from };
	};

	template <typename T> struct is_complex : std::false_type {};
	template <typename T> struct is_complex<const T> : is_complex<T> {};
	template <typename T> struct is_complex<complex<T>> : std::true_type {};
	template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

	template <typename T> struct underlying { using type = T; };
	template <typename T> struct underlying<complex<T>> { using type = T; };
	template <typename T> using underlying_t = typename underlying<T>::type;
	
	template <typename T1, typename T2, bool complex = false>
	struct TypeHelperImpl {
		using First = T1;
		using Second = T2;
		using RawFirst = First;
		using RawSecond = Second;
	};
	
	template <typename T1, typename T2>
	struct TypeHelperImpl<T1, T2, true> {
		using helper_underlying = TypeHelperImpl<underlying_t<T1>, underlying_t<T2>>;

		using common_type = decltype(std::declval<typename helper_underlying::First>() * std::declval<typename helper_underlying::Second>());
		using First = complex<common_type>;
		using Second = complex<common_type>;
		using RawFirst = std::conditional_t<!is_complex_v<T1>, common_type, First>;
		using RawSecond = std::conditional_t<!is_complex_v<T2>, common_type, Second>;
	};

	template <typename T1, typename T2>
	struct TypeHelper {
	private:
		using helper_impl = TypeHelperImpl<T1, T2, is_complex_v<T1> || is_complex_v<T2>>;

	public:
		using First = typename helper_impl::First;
		using Second = typename helper_impl::Second;
		using RawFirst = typename helper_impl::RawFirst;
		using RawSecond = typename helper_impl::RawSecond;
	};
}

