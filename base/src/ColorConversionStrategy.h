#include "ColorConversionXForm.h"

class DetailAbstract
{
public:
	DetailAbstract() {}
	~DetailAbstract() {}
	virtual void convert(frame_container& inputFrame, frame_sp& outFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg) {};
};

class CpuInterleaved2Planar : public DetailAbstract
{
public:
	CpuInterleaved2Planar() {}
	~CpuInterleaved2Planar() {}

protected:
	void initMatImages(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat& inpImg, cv::Mat& outImg)
	{
		auto frame = Module::getFrameByType(inputFrame, FrameMetadata::RAW_IMAGE);

		inpImg.data = static_cast<uint8_t*>(frame->data());
		outImg.data = static_cast<uint8_t*>(outputFrame->data());
	}
};

class CpuRGB2YUV420Planar : public CpuInterleaved2Planar
{
public:
	CpuRGB2YUV420Planar() {}
	~CpuRGB2YUV420Planar() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_RGB2YUV_I420);
	}
};

class CpuInterleaved2Interleaved : public DetailAbstract
{
public:
	CpuInterleaved2Interleaved() {}
	~CpuInterleaved2Interleaved() {}
protected:
	void initMatImages(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat& inpImg, cv::Mat& outImg)
	{
		auto frame = Module::getFrameByType(inputFrame, FrameMetadata::RAW_IMAGE);

		inpImg.data = static_cast<uint8_t*>(frame->data());
		outImg.data = static_cast<uint8_t*>(outputFrame->data());
	}
};

class CpuRGB2BGR : public CpuInterleaved2Interleaved
{
public:
	CpuRGB2BGR() {}
	~CpuRGB2BGR() {}
	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_RGB2BGR);
	}
};

class CpuBGR2RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBGR2RGB() {}
	~CpuBGR2RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BGR2RGB);
	}
};

class CpuRGB2MONO : public CpuInterleaved2Interleaved
{
public:
	CpuRGB2MONO() {}
	~CpuRGB2MONO() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_RGB2GRAY);
	}
};

class CpuBGR2MONO : public CpuInterleaved2Interleaved
{
public:
	CpuBGR2MONO() {}
	~CpuBGR2MONO() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BGR2GRAY);
	}
};

class CpuBayerBG82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerBG82RGB() {}
	~CpuBayerBG82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerRG2RGB);
	}
};

class CpuBayerGB82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerGB82RGB() {}
	~CpuBayerGB82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerGR2RGB);
	}
};

class CpuBayerGR82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerGR82RGB() {}
	~CpuBayerGR82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerGB2RGB);
	}
};

class CpuBayerRG82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerRG82RGB() {}
	~CpuBayerRG82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerBG2RGB);
	}
};

class CpuBayerBG82Mono : public CpuInterleaved2Interleaved
{
public:
	CpuBayerBG82Mono() {}
	~CpuBayerBG82Mono() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerBG2GRAY);
	}
};

class CpuPlanar2Interleaved : public DetailAbstract
{
public:
	CpuPlanar2Interleaved() {}
	~CpuPlanar2Interleaved() {}
	void initMatImages(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat& inpImg, cv::Mat& outImg)
	{
		auto frame = Module::getFrameByType(inputFrame, FrameMetadata::RAW_IMAGE_PLANAR);

		inpImg.data = static_cast<uint8_t*>(frame->data());
		outImg.data = static_cast<uint8_t*>(outputFrame->data());
	}
};

class CpuYUV420Planar2RGB : public CpuPlanar2Interleaved
{
public:
	CpuYUV420Planar2RGB() {}
	~CpuYUV420Planar2RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata, cv::Mat inpImg, cv::Mat outImg)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata, inpImg, outImg);
		cv::cvtColor(inpImg, outImg, cv::COLOR_YUV420p2RGB);
	}
};