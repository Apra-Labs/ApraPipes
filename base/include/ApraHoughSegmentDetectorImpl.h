#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector_types.h>

class ApraHoughSegmentDetectorImpl
{
public:
	ApraHoughSegmentDetectorImpl(float rho, float theta, int minLineLength, int maxLineGap, int maxLines);
	~ApraHoughSegmentDetectorImpl();

	void init(int rows, int cols, int type);

	int detect(cv::InputArray src, cv::OutputArray lines, cv::cuda::Stream& stream);

	void setRho(float rho) { rho_ = rho; }
	float getRho() const { return rho_; }

	void setTheta(float theta) { theta_ = theta; }
	float getTheta() const { return theta_; }

	void setMinLineLength(int minLineLength) { minLineLength_ = minLineLength; }
	int getMinLineLength() const { return minLineLength_; }

	void setMaxLineGap(int maxLineGap) { maxLineGap_ = maxLineGap; }
	int getMaxLineGap() const { return maxLineGap_; }

	void setMaxLines(int maxLines) { maxLines_ = maxLines; }
	int getMaxLines() const { return maxLines_; }

	

private:
	float rho_;
	float theta_;
	int minLineLength_;
	int maxLineGap_;
	int maxLines_;
	bool has20_;
	size_t sharedMemPerBlock_;

	int* g_counter_buildpoints_;
	int* g_counter_houghsegments_;

	cv::cuda::GpuMat accum_;
	cv::cuda::GpuMat list_;
};