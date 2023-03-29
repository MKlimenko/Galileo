#include "common.hpp"
#include "binary.hpp"

#define BINARY_BINARY_ELTWISE_FUNCTION_DEF(EXT_NAME, KERNEL_NAME) GALILEO_RESULT EXT_NAME(const GALILEO_TENSOR* input_lhs, const GALILEO_TENSOR* input_rhs, GALILEO_TENSOR* output) { \
	try { \
		if (!input_lhs || !input_rhs || !output || !input_lhs->tensor_data || !input_rhs->tensor_data || !output->tensor_data) \
			return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER; \
	 \
		if (!galileo::common::VerifyQueryPtrs(*input_lhs, *input_rhs, *output)) \
			return GALILEO_RESULT::GALILEO_RESULT_TENSOR_QUEUE_MISMATCH; \
	 \
		if (!galileo::common::VerifyDimensionsPtrs(*input_lhs, *input_rhs, *output)) \
			return GALILEO_RESULT::GALILEO_RESULT_TENSOR_DIMENSIONS_MISMATCH; \
	 \
		auto queue = input_lhs->associated_queue; \
	 \
		const auto input_lhs_ptr_state = galileo::common::VerifyPtr(galileo::common::GetQueue(queue), input_lhs->tensor_data); \
		const auto input_rhs_ptr_state = galileo::common::VerifyPtr(galileo::common::GetQueue(queue), input_rhs->tensor_data); \
		const auto output_ptr_state = galileo::common::VerifyPtr(galileo::common::GetQueue(queue), output->tensor_data); \
		if (!input_lhs_ptr_state || !input_rhs_ptr_state || !output_ptr_state) \
			return GALILEO_RESULT_NON_USM_POINTER; \
	\
		auto kernel = KERNEL_NAME(*input_lhs, *input_rhs, *output); \
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
#define BINARY_ELTWISE_FUNCTION(NAME) BINARY_BINARY_ELTWISE_FUNCTION_DEF(CREATE_EXT_NAME(NAME), NAME)

BINARY_ELTWISE_FUNCTION(Add)
BINARY_ELTWISE_FUNCTION(Div)
BINARY_ELTWISE_FUNCTION(Mul)
BINARY_ELTWISE_FUNCTION(Sub)