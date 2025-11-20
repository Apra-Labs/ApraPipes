#pragma once

#include "Module.h"
#include "CudaCommon.h"
#include "FrameMetadata.h"

#include <boost/serialization/vector.hpp>

class GaussianBlurProps : public ModuleProps
{
public:
	GaussianBlurProps(cudastream_sp _stream) : ModuleProps()
	{
		stream = _stream;
		kernelSize = 11;
		roi = {0, 0, 0, 0};
	}

	GaussianBlurProps(cudastream_sp _stream, int _kernelSize) : ModuleProps()
	{
		stream = _stream;
		kernelSize = _kernelSize;
		roi = {0, 0, 0, 0};
	}

	cudastream_sp stream;
	int kernelSize;
	std::vector<int> roi;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(roi) + (4 * sizeof(int)) + sizeof(kernelSize);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &kernelSize;
		ar &roi;
	}
};

class GaussianBlur : public Module
{
public:
	GaussianBlur(GaussianBlurProps props);
	virtual ~GaussianBlur();

	bool init() override;
	bool term() override;

	void setProps(GaussianBlurProps &props);
	GaussianBlurProps getProps();

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, std::string_view pinId) override; // throws exception if validation fails
	bool shouldTriggerSOS() override;
	bool processEOS(std::string_view pinId) override;
	bool handlePropsChange(frame_sp &frame) override;

private:
	void setMetadata(framemetadata_sp& inputMetadata);

	class Detail;
	std::shared_ptr<Detail> mDetail;

	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	size_t mOutDataSize;
};
