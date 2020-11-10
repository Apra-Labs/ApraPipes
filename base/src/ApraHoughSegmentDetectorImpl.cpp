/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <ApraHoughSegmentDetectorImpl.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace cv
{
	namespace cuda
	{
		namespace device
		{
			namespace apra_hough
			{
				int buildPointList_gpu(PtrStepSzb src, unsigned int *list, int* g_counter, cudaStream_t stream);
				void linesAccum_gpu(const unsigned int *list, int count, PtrStepSzi accum, float rho, float theta, size_t sharedMemPerBlock, bool has20, cudaStream_t stream);
				int houghLinesProbabilistic_gpu(PtrStepSzb mask, PtrStepSzi accum, int4 *out, int maxSize, float rho, float theta, int lineGap, int lineLength, int* g_counter, cudaStream_t stream);
			} // namespace apra_hough
		}	  // namespace device
	}		  // namespace cuda
} // namespace cv

using namespace cv;
using namespace cv::cuda;

ApraHoughSegmentDetectorImpl::ApraHoughSegmentDetectorImpl(float rho, float theta, int minLineLength, int maxLineGap, int maxLines) : rho_(rho), theta_(theta), minLineLength_(minLineLength), maxLineGap_(maxLineGap), maxLines_(maxLines)
{
	void* temp;
	cudaMalloc(&temp, 4);
	g_counter_buildpoints_ = static_cast<int*>(temp);

	cudaMalloc(&temp, 4);
	g_counter_houghsegments_ = static_cast<int*>(temp);
}

ApraHoughSegmentDetectorImpl::~ApraHoughSegmentDetectorImpl()
{
	if (g_counter_buildpoints_)
	{
		cudaFree(g_counter_buildpoints_);
		cudaFree(g_counter_houghsegments_);
		g_counter_buildpoints_ = nullptr;
		g_counter_houghsegments_ = nullptr;
	}
}

void ApraHoughSegmentDetectorImpl::init(int rows, int cols, int type)
{
	CV_Assert(type == CV_8UC1);
	CV_Assert(cols < std::numeric_limits<unsigned short>::max());
	CV_Assert(rows < std::numeric_limits<unsigned short>::max());

	ensureSizeIsEnough(1, cv::Size(cols, rows).area(), CV_32SC1, list_);

	const int numangle = cvRound(CV_PI / theta_);
	const int numrho = cvRound(((cols + rows) * 2 + 1) / rho_);
	CV_Assert(numangle > 0 && numrho > 0);

	ensureSizeIsEnough(numangle + 2, numrho + 2, CV_32SC1, accum_);

	DeviceInfo devInfo;
	has20_ = devInfo.supports(FEATURE_SET_COMPUTE_20);
	sharedMemPerBlock_ = devInfo.sharedMemPerBlock();
}

int ApraHoughSegmentDetectorImpl::detect(InputArray _src, OutputArray _lines, Stream &stream)
{
	auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);

	cv::cuda::GpuMat src = _src.getGpuMat();
	unsigned int *srcPoints = list_.ptr<unsigned int>();
	cv::cuda::GpuMat lines = _lines.getGpuMat();

	const int pointsCount = cv::cuda::device::apra_hough::buildPointList_gpu(src, srcPoints, g_counter_buildpoints_, cudaStream);
	if (pointsCount == 0)
	{
		return 0;
	}

	accum_.setTo(Scalar::all(0), stream);

	cv::cuda::device::apra_hough::linesAccum_gpu(srcPoints, pointsCount, accum_, rho_, theta_, sharedMemPerBlock_, has20_, cudaStream);

	auto linesCount = cv::cuda::device::apra_hough::houghLinesProbabilistic_gpu(src, accum_, lines.ptr<int4>(), maxLines_, rho_, theta_, maxLineGap_, minLineLength_, g_counter_houghsegments_, cudaStream);

	return linesCount;
}