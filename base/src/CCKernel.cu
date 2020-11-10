#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CCKernel.h"

__global__ void appYUV411ToYUV444(const Npp8u* src, int nSrcStep, Npp8u* dst_y, Npp8u* dst_u, Npp8u* dst_v, int rDstStep, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y >= height)
	{
		return;
	}	

	int x_ = 4 * x;
	int dst_offset = y * rDstStep + x_;
	int offset = y* nSrcStep + 6 * x;

	dst_y[dst_offset + 0] = src[offset + 1];
	dst_y[dst_offset + 1] = src[offset + 2];
	dst_y[dst_offset + 2] = src[offset + 4];
	dst_y[dst_offset + 3] = src[offset + 5];
	
	
	auto u_value = src[offset];
	dst_u[dst_offset + 0] = u_value;
	dst_u[dst_offset + 1] = u_value;
	dst_u[dst_offset + 2] = u_value;
	dst_u[dst_offset + 3] = u_value;

	auto v_value = src[offset + 3];
	dst_v[dst_offset + 0] = v_value;
	dst_v[dst_offset + 1] = v_value;
	dst_v[dst_offset + 2] = v_value;
	dst_v[dst_offset + 3] = v_value;
}

void lanuchAPPYUV411ToYUV444(const Npp8u* src, int nSrcStep, Npp8u* dst[3], int rDstStep, NppiSize oSizeROI, cudaStream_t stream)
{
	dim3 block(32, 32); 
	dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
	appYUV411ToYUV444 <<<grid, block, 0, stream>>> (src, nSrcStep, dst[0], dst[1], dst[2], rDstStep, oSizeROI.width >> 2, oSizeROI.height);
}