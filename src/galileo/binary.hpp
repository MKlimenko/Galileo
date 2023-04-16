#include "common.hpp"

namespace galileo {
	inline namespace detail {
		template <auto F>
		struct BinaryElementwiseOp {
		protected:
			using binary_eltwise_types = std::tuple<
				std::int8_t,
				std::int16_t,
				std::int32_t,
				std::int64_t,
				std::uint8_t,
				std::uint16_t,
				std::uint32_t,
				std::uint64_t,
				float,
				double,
				sycl::half,
				common::complex<float>,
				common::complex<double>,
				common::complex<sycl::half>
			>;

			template <bool is_const, typename T>
			static T GetVariantFromInput(CONSTIFY(void)* ptr, GALILEO_DATA_TYPE data_type) {
				switch (data_type) {
				case GALILEO_UINT8: return reinterpret_cast<CONSTIFY(std::uint8_t)*>(ptr);
				case GALILEO_UINT16: return reinterpret_cast<CONSTIFY(std::uint16_t)*>(ptr);
				case GALILEO_UINT32: return reinterpret_cast<CONSTIFY(std::uint32_t)*>(ptr);
				case GALILEO_UINT64: return reinterpret_cast<CONSTIFY(std::uint64_t)*>(ptr);
				case GALILEO_INT8: return reinterpret_cast<CONSTIFY(std::int8_t)*>(ptr);
				case GALILEO_INT16: return reinterpret_cast<CONSTIFY(std::int16_t)*>(ptr);
				case GALILEO_INT32: return reinterpret_cast<CONSTIFY(std::int32_t)*>(ptr);
				case GALILEO_INT64: return reinterpret_cast<CONSTIFY(std::int64_t)*>(ptr);
				case GALILEO_FLOAT: return reinterpret_cast<CONSTIFY(float)*>(ptr);
				case GALILEO_DOUBLE: return reinterpret_cast<CONSTIFY(double)*>(ptr);
				case GALILEO_HALF: return reinterpret_cast<CONSTIFY(sycl::half)*>(ptr);
				case GALILEO_COMPLEX_FLOAT: return reinterpret_cast<CONSTIFY(common::complex<float>)*>(ptr);
				case GALILEO_COMPLEX_DOUBLE: return reinterpret_cast<CONSTIFY(common::complex<double>)*>(ptr);
				case GALILEO_COMPLEX_HALF: return reinterpret_cast<CONSTIFY(common::complex<sycl::half>)*>(ptr);
				default:
					throw GALILEO_RESULT::GALILEO_RESULT_UNEXPECTED_DATA_TYPE;
				}
			}

			template <typename T, typename U, typename D>
			void Process(sycl::handler& h, const T* input_lhs_ptr, const U* input_rhs_ptr, D* output_ptr) {
				using type_helper = common::TypeHelper<T, U>;

				using Src1Type = typename type_helper::First;
				using Src2Type = typename type_helper::Second;
				using Src1RawType = typename type_helper::RawFirst;
				using Src2RawType = typename type_helper::RawSecond;
				using invoke_result = std::invoke_result_t<decltype(F), Src1Type, Src2Type>;
				//if constexpr (common::is_narrowing_conversion<invoke_result, D>)
				if constexpr (!std::is_same_v<invoke_result, D>) // todo: switch to narrowing_conversion, bypass is used to speedup the  build process
					throw std::runtime_error("Requested type requires narrowing conversion from the calculation result, add explicit cast or quantization");
				else {
					h.parallel_for(size, [=](auto i) {
						auto lhs = static_cast<Src1RawType>(input_lhs_ptr[i]);
						auto rhs = static_cast<Src2RawType>(input_rhs_ptr[i]);
						auto dst = F(lhs, rhs);
						output_ptr[i] = static_cast<D>(dst);
						});
				}
			}

		public:
			using InputType = decltype(common::GetVariantFromTuple<true>(std::declval<binary_eltwise_types>()));
			using OutputType = decltype(common::GetVariantFromTuple<false>(std::declval<binary_eltwise_types>()));

			InputType input_lhs;
			InputType input_rhs;
			OutputType output;
			unsigned int size; // todo: consider broadcasting

			BinaryElementwiseOp(const GALILEO_TENSOR& input_lhs, const GALILEO_TENSOR& input_rhs, GALILEO_TENSOR& output) :
				input_lhs(GetVariantFromInput<true, InputType>(input_lhs.tensor_data, input_lhs.data_type)),
				input_rhs(GetVariantFromInput<true, InputType>(input_rhs.tensor_data, input_rhs.data_type)),
				output(GetVariantFromInput<false, OutputType>(output.tensor_data, output.data_type)),
				size(common::GetTotalSize(input_lhs.dimensions)) {}

			void operator()(sycl::handler& h) {
				std::visit([&](const auto* input_lhs_ptr, const auto* input_rhs_ptr, auto* output_ptr) { Process(h, input_lhs_ptr, input_rhs_ptr, output_ptr); }, input_lhs, input_rhs, output);
			}
		};

	}
}

using Add = galileo::BinaryElementwiseOp < [](auto lhs, auto rhs) { return lhs + rhs; } > ;
using Mul = galileo::BinaryElementwiseOp < [](auto lhs, auto rhs) { return lhs * rhs; } > ;
using Div = galileo::BinaryElementwiseOp < [](auto lhs, auto rhs) { return lhs / rhs; } > ;
using Sub = galileo::BinaryElementwiseOp < [](auto lhs, auto rhs) { return lhs - rhs; } > ;
