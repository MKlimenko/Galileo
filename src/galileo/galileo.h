#ifndef GALILEO_H_
#define GALILEO_H_

typedef enum tagGALILEO_RESULT {
	GALILEO_RESULT_OK,
	GALILEO_RESULT_INVALID_FUNC_PARAMETER,
	GALILEO_RESULT_UNEXPECTED_DATA_TYPE,
	GALILEO_RESULT_NON_USM_POINTER,
	GALILEO_RESULT_UNKNOWN_ERROR
} GALILEO_RESULT;

typedef enum tagGALILEO_DATA_TYPE {
	GALILEO_UINT8 = 0,
	GALILEO_UINT16,
	GALILEO_UINT32,
	GALILEO_UINT64,
	GALILEO_INT8,
	GALILEO_INT16,
	GALILEO_INT32,
	GALILEO_INT64,
	GALILEO_FLOAT,
	GALILEO_DOUBLE,
	GALILEO_HALF,
	GALILEO_BFLOAT16
} GALILEO_DATA_TYPE;

typedef void* GALILEO_QUEUE;

#ifdef __cplusplus
extern "C" {
#endif

GALILEO_RESULT GALILEO_GetLibVersion(unsigned int* major, unsigned int* minor, unsigned int* patch);
GALILEO_RESULT GALILEO_GetQueueSize(unsigned int* size);
GALILEO_RESULT GALILEO_InitQueue(GALILEO_QUEUE queue);
GALILEO_RESULT GALILEO_ReleaseQueue(GALILEO_QUEUE queue);
GALILEO_RESULT GALILEO_Allocate(GALILEO_QUEUE queue, GALILEO_DATA_TYPE data_type, unsigned int size, void** ptr);
GALILEO_RESULT GALILEO_Deallocate(GALILEO_QUEUE queue, void* ptr);

GALILEO_RESULT GALILEO_Abs(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Acos(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Acosh(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Asin(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Asinh(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Atan(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Atanh(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Cos(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Cosh(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Erf(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Exp(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Neg(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Sign(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Sin(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Sinh(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Sqrt(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Tan(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);
GALILEO_RESULT GALILEO_Tanh(GALILEO_QUEUE queue, const void* input, GALILEO_DATA_TYPE input_data_type, unsigned int size, void* output, GALILEO_DATA_TYPE output_data_type);

#ifdef __cplusplus
}
#endif

#endif
