#include "common.hpp"

namespace galileo {
	inline namespace detail {
		enum class TypesToUse {
			OnlyFp,
			FpWithSignedIntegers,
			FpWithIntegers
		};

		template <TypesToUse types_to_use, auto F>
		struct UnaryElementwiseOp {
		protected:
			using eltwise_types_integers = std::tuple<
				std::int8_t,
				std::int16_t,
				std::int32_t,
				std::int64_t
			>;
			using eltwise_types_uintegers = std::tuple<
				std::uint8_t,
				std::uint16_t,
				std::uint32_t,
				std::uint64_t
			>;
			using eltwise_types_fp = std::tuple<
				float,
				double,
				sycl::half,
				sycl::ext::oneapi::experimental::bfloat16
			>;

			using eltwise_types_with_signed_integers = decltype(std::tuple_cat(std::declval<eltwise_types_fp>(), std::declval<eltwise_types_integers>()));
			using eltwise_types_with_integers = decltype(std::tuple_cat(std::declval<eltwise_types_with_signed_integers>(), std::declval<eltwise_types_uintegers>()));
			using eltwise_types = std::conditional_t<types_to_use == TypesToUse::OnlyFp, eltwise_types_fp,
				std::conditional_t<types_to_use == TypesToUse::FpWithIntegers, eltwise_types_with_integers, eltwise_types_with_signed_integers>>;

			template <bool is_const, typename T>
			static T GetVariantFromInput(CONSTIFY(void)* ptr, GALILEO_DATA_TYPE data_type) {
				if constexpr (types_to_use == TypesToUse::FpWithIntegers) {
					switch (data_type) {
					case GALILEO_UINT8: return reinterpret_cast<CONSTIFY(std::uint8_t)*>(ptr);
					case GALILEO_UINT16: return reinterpret_cast<CONSTIFY(std::uint16_t)*>(ptr);
					case GALILEO_UINT32: return reinterpret_cast<CONSTIFY(std::uint32_t)*>(ptr);
					case GALILEO_UINT64: return reinterpret_cast<CONSTIFY(std::uint64_t)*>(ptr);
					default:
						break;
					}
				}
				if constexpr (types_to_use == TypesToUse::FpWithSignedIntegers) {
					switch (data_type) {
					case GALILEO_INT8: return reinterpret_cast<CONSTIFY(std::int8_t)*>(ptr);
					case GALILEO_INT16: return reinterpret_cast<CONSTIFY(std::int16_t)*>(ptr);
					case GALILEO_INT32: return reinterpret_cast<CONSTIFY(std::int32_t)*>(ptr);
					case GALILEO_INT64: return reinterpret_cast<CONSTIFY(std::int64_t)*>(ptr);
					default:
						break;
					}
				}
				switch (data_type) {
				case GALILEO_FLOAT: return reinterpret_cast<CONSTIFY(float)*>(ptr);
				case GALILEO_DOUBLE: return reinterpret_cast<CONSTIFY(double)*>(ptr);
				case GALILEO_HALF: return reinterpret_cast<CONSTIFY(sycl::half)*>(ptr);
				case GALILEO_BFLOAT16: return reinterpret_cast<CONSTIFY(sycl::ext::oneapi::experimental::bfloat16)*>(ptr);
				default:
					throw GALILEO_RESULT::GALILEO_RESULT_UNEXPECTED_DATA_TYPE;
				}
			}

			template <typename T, typename U>
			void Process(sycl::handler& h, const T* input_ptr, U* output_ptr) {
				h.parallel_for(size, [=](auto i) {
					auto src = static_cast<std::conditional_t<std::is_same_v<T, sycl::ext::oneapi::experimental::bfloat16>, float, T>>(input_ptr[i]);
					auto dst = F(src);
					output_ptr[i] = static_cast<U>(dst);
					});
			}

		public:
			using InputType = decltype(common::GetVariantFromTuple<true>(std::declval<eltwise_types>()));
			using OutputType = decltype(common::GetVariantFromTuple<false>(std::declval<eltwise_types>()));

			InputType input;
			OutputType output;
			unsigned int size;

			UnaryElementwiseOp(const GALILEO_TENSOR& input, GALILEO_TENSOR& output) :
				input(GetVariantFromInput<true, InputType>(input.tensor_data, input.data_type)),
				output(GetVariantFromInput<false, OutputType>(output.tensor_data, output.data_type)),
				size(common::GetTotalSize(input.dimensions)) {}

			void operator()(sycl::handler& h) {
				std::visit([&](const auto* input_ptr, auto* output_ptr) { Process(h, input_ptr, output_ptr); }, input, output);
			}
		};

		using Abs = UnaryElementwiseOp < TypesToUse::FpWithIntegers, [](auto v) { return sycl::abs(v); } > ;
		using Acos = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::acos(v); } > ;
		using Acosh = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::acosh(v); } > ;
		using Asin = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::asin(v); } > ;
		using Asinh = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::asinh(v); } > ;
		using Atan = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::atan(v); } > ;
		using Atanh = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::atanh(v); } > ;
		using Cos = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::cos(v); } > ;
		using Cosh = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::cosh(v); } > ;
		using Erf = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::erf(v); } > ;
		using Exp = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::exp(v); } > ;
		using Log = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::log(v); } > ;
		using Neg = UnaryElementwiseOp < TypesToUse::FpWithSignedIntegers, [](auto v) { return -v; } > ;
		using Sign = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::sign(v); } > ;
		using Sin = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::sin(v); } > ;
		using Sinh = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::sinh(v); } > ;
		using Sqrt = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::sqrt(v); } > ;
		using Tan = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::tan(v); } > ;
		using Tanh = UnaryElementwiseOp < TypesToUse::OnlyFp, [](auto v) { return sycl::tanh(v); } > ;
	}
}
