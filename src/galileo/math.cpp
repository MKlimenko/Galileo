#include "common.hpp"
#include "math.hpp"
#include <variant>

#define ELTWISE_FUNCTION_DEF(EXT_NAME, KERNEL_NAME) GALILEO_RESULT EXT_NAME(const GALILEO_TENSOR* input, GALILEO_TENSOR* output) { \
	try { \
		if (!input || !output || !input->tensor_data || !output->tensor_data) \
			return GALILEO_RESULT::GALILEO_RESULT_INVALID_FUNC_PARAMETER; \
	 \
		if (!common::VerifyQueryPtrs(*input, *output)) \
			return GALILEO_RESULT::GALILEO_RESULT_TENSOR_QUEUE_MISMATCH; \
	 \
		if (!common::VerifyDimensionsPtrs(*input, *output)) \
			return GALILEO_RESULT::GALILEO_RESULT_TENSOR_DIMENSIONS_MISMATCH; \
	 \
		auto queue = input->associated_queue; \
	 \
		const auto input_ptr_state = common::VerifyPtr(common::GetQueue(queue), input->tensor_data); \
		const auto output_ptr_state = common::VerifyPtr(common::GetQueue(queue), output->tensor_data); \
		if (!input_ptr_state || !output_ptr_state) \
			return GALILEO_RESULT_NON_USM_POINTER; \
	\
		auto kernel = KERNEL_NAME(*input, *output); \
		auto& typed_queue = common::GetQueue(queue); \
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
#define ELTWISE_FUNCTION(NAME) ELTWISE_FUNCTION_DEF(CREATE_EXT_NAME(NAME), NAME)


ELTWISE_FUNCTION(Abs)
ELTWISE_FUNCTION(Acos)
ELTWISE_FUNCTION(Acosh)
ELTWISE_FUNCTION(Asin)
ELTWISE_FUNCTION(Asinh)
ELTWISE_FUNCTION(Atan)
ELTWISE_FUNCTION(Atanh)
ELTWISE_FUNCTION(Cos)
ELTWISE_FUNCTION(Cosh)
ELTWISE_FUNCTION(Erf)
ELTWISE_FUNCTION(Exp)
ELTWISE_FUNCTION(Log)
ELTWISE_FUNCTION(Neg)
ELTWISE_FUNCTION(Sign)
ELTWISE_FUNCTION(Sin)
ELTWISE_FUNCTION(Sinh)
ELTWISE_FUNCTION(Sqrt)
ELTWISE_FUNCTION(Tan)
ELTWISE_FUNCTION(Tanh)
