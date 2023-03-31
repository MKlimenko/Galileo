#include "common.hpp"
#include "unary.hpp"
#include <variant>

#define UNARY_ELTWISE_FUNCTION_DEF(EXT_NAME, KERNEL_NAME) GALILEO_RESULT EXT_NAME(const GALILEO_TENSOR* input, GALILEO_TENSOR* output) { \
	try { \
		if (!input || !output || !input->tensor_data || !output->tensor_data) \
			return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER; \
	 \
		if (!galileo::common::VerifyQueryPtrs(*input, *output)) \
			return GALILEO_RESULT::GALILEO_RESULT_TENSOR_QUEUE_MISMATCH; \
	 \
		if (!galileo::common::VerifyDimensionsPtrs(*input, *output)) \
			return GALILEO_RESULT::GALILEO_RESULT_TENSOR_DIMENSIONS_MISMATCH; \
	 \
		auto queue = input->associated_queue; \
	 \
		const auto input_ptr_state = galileo::common::VerifyPtr(galileo::common::GetQueue(queue), input->tensor_data); \
		const auto output_ptr_state = galileo::common::VerifyPtr(galileo::common::GetQueue(queue), output->tensor_data); \
		if (!input_ptr_state || !output_ptr_state) \
			return GALILEO_RESULT_NON_USM_POINTER; \
	\
		auto kernel = KERNEL_NAME(*input, *output); \
		auto& typed_queue = galileo::common::GetQueue(queue); \
		typed_queue.submit(kernel); \
	} \
	catch (GALILEO_RESULT res) { \
		return res; \
	} \
	catch (...) { \
		return GALILEO_RESULT::GALILEO_RESULT_UNKNOWN_ERROR; \
	} \
	return GALILEO_RESULT::GALILEO_RESULT_OK; \
}
#define CREATE_EXT_NAME( s ) GALILEO_ ## s
#define UNARY_ELTWISE_FUNCTION(NAME) UNARY_ELTWISE_FUNCTION_DEF(CREATE_EXT_NAME(NAME), NAME)

UNARY_ELTWISE_FUNCTION(Abs)
UNARY_ELTWISE_FUNCTION(Acos)
UNARY_ELTWISE_FUNCTION(Acosh)
UNARY_ELTWISE_FUNCTION(Asin)
UNARY_ELTWISE_FUNCTION(Asinh)
UNARY_ELTWISE_FUNCTION(Atan)
UNARY_ELTWISE_FUNCTION(Atanh)
UNARY_ELTWISE_FUNCTION(Conj)
UNARY_ELTWISE_FUNCTION(Cos)
UNARY_ELTWISE_FUNCTION(Cosh)
UNARY_ELTWISE_FUNCTION(Erf)
UNARY_ELTWISE_FUNCTION(Exp)
UNARY_ELTWISE_FUNCTION(Log)
UNARY_ELTWISE_FUNCTION(Neg)
UNARY_ELTWISE_FUNCTION(Sign)
UNARY_ELTWISE_FUNCTION(Sin)
UNARY_ELTWISE_FUNCTION(Sinh)
UNARY_ELTWISE_FUNCTION(Sqrt)
UNARY_ELTWISE_FUNCTION(Tan)
UNARY_ELTWISE_FUNCTION(Tanh)