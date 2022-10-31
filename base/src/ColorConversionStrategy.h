#include "ColorConversionXForm.h"

class DetailAbstract
{
public:
	DetailAbstract() {}
	DetailAbstract(cv::Mat _inpImg, cv::Mat _outImg)
	{
		inpImg = _inpImg;
		outImg = _outImg;
	}
	~DetailAbstract() {}
	virtual void convert(frame_container& inputFrame, frame_sp& outFrame, framemetadata_sp outputMetadata) {};
	cv::Mat inpImg;
	cv::Mat outImg;
};

class CpuInterleaved2Planar : public DetailAbstract
{
public:
	CpuInterleaved2Planar(cv::Mat _inpImg, cv::Mat _outImg) : DetailAbstract(_inpImg, _outImg) {}
	~CpuInterleaved2Planar() {}

protected:
	void initMatImages(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		auto frame = Module::getFrameByType(inputFrame, FrameMetadata::RAW_IMAGE);

		inpImg.data = static_cast<uint8_t*>(frame->data());
		outImg.data = static_cast<uint8_t*>(outputFrame->data());
	}
};

class CpuRGB2YUV420Planar : public CpuInterleaved2Planar
{
public:
	CpuRGB2YUV420Planar(cv::Mat _inpImg, cv::Mat _outImg) :  CpuInterleaved2Planar(_inpImg,_outImg){}
	~CpuRGB2YUV420Planar() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_RGB2YUV_I420);
	}
};

class CpuInterleaved2Interleaved : public DetailAbstract
{
public:
	CpuInterleaved2Interleaved(cv::Mat _inpImg, cv::Mat _outImg) : DetailAbstract(_inpImg, _outImg) {}
	~CpuInterleaved2Interleaved() {}
protected:
	void initMatImages(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		auto frame = Module::getFrameByType(inputFrame, FrameMetadata::RAW_IMAGE);

		inpImg.data = static_cast<uint8_t*>(frame->data());
		outImg.data = static_cast<uint8_t*>(outputFrame->data());
	}
};

class CpuRGB2BGR : public CpuInterleaved2Interleaved
{
public:
	CpuRGB2BGR(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg,_outImg){}
	~CpuRGB2BGR() {}
	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_RGB2BGR);
	}
};

class CpuBGR2RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBGR2RGB(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuBGR2RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BGR2RGB);
	}
};

class CpuRGB2MONO : public CpuInterleaved2Interleaved
{
public:
	CpuRGB2MONO(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuRGB2MONO() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_RGB2GRAY);
	}
};

class CpuBGR2MONO : public CpuInterleaved2Interleaved
{
public:
	CpuBGR2MONO(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuBGR2MONO() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BGR2GRAY);
	}
};

class CpuBayerBG82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerBG82RGB(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuBayerBG82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerRG2RGB);
	}
};

class CpuBayerGB82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerGB82RGB(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuBayerGB82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerGR2RGB);
	}
};

class CpuBayerGR82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerGR82RGB(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuBayerGR82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerGB2RGB);
	}
};

class CpuBayerRG82RGB : public CpuInterleaved2Interleaved
{
public:
	CpuBayerRG82RGB(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuBayerRG82RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerBG2RGB);
	}
};

class CpuBayerBG82Mono : public CpuInterleaved2Interleaved
{
public:
	CpuBayerBG82Mono(cv::Mat _inpImg, cv::Mat _outImg) : CpuInterleaved2Interleaved(_inpImg, _outImg) {}
	~CpuBayerBG82Mono() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_BayerBG2GRAY);
	}
};

class CpuPlanar2Interleaved : public DetailAbstract
{
public:
	CpuPlanar2Interleaved(cv::Mat _inpImg, cv::Mat _outImg) : DetailAbstract(_inpImg, _outImg) {}
	~CpuPlanar2Interleaved() {}
	void initMatImages(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		auto frame = Module::getFrameByType(inputFrame, FrameMetadata::RAW_IMAGE_PLANAR);

		inpImg.data = static_cast<uint8_t*>(frame->data());
		outImg.data = static_cast<uint8_t*>(outputFrame->data());
	}
};

class CpuYUV420Planar2RGB : public CpuPlanar2Interleaved
{
public:
	CpuYUV420Planar2RGB(cv::Mat _inpImg, cv::Mat _outImg) : CpuPlanar2Interleaved(_inpImg,_outImg) {}
	~CpuYUV420Planar2RGB() {}

	void convert(frame_container& inputFrame, frame_sp& outputFrame, framemetadata_sp outputMetadata)
	{
		initMatImages(inputFrame, outputFrame, outputMetadata);
		cv::cvtColor(inpImg, outImg, cv::COLOR_YUV420p2RGB);
	}
};