#include "CudaCommon.h"
#include "cuda_runtime_api.h"

bool CudaUtils::isCudaSupported()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		LOG_ERROR << "cudaGetDeviceCount returned ERROR. <" << static_cast<int>(error_id) << "> <" << cudaGetErrorString(error_id) << ">";		
		return false;
	}

	return deviceCount != 0;
}