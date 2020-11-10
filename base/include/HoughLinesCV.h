#pragma once

#include "Module.h"
#include "CudaCommon.h"
#include "FrameMetadata.h"

#include <boost/serialization/vector.hpp>

class HoughLinesCVProps : public ModuleProps
{
public:
	HoughLinesCVProps(cudastream_sp _stream) : ModuleProps()
	{
		stream = _stream;
		roi = {0, 0, 0, 0};

		minLineLength = 50;
		maxLineGap = 20;
	}

	cudastream_sp stream;
	std::vector<int> roi;

	int minLineLength;
	int maxLineGap;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() +
			   sizeof(roi) + (4 * sizeof(int)) +
			   sizeof(minLineLength) +
			   sizeof(maxLineGap);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &roi;
		ar &minLineLength;
		ar &maxLineGap;
	}
};

class HoughLinesCV : public Module
{
public:
	HoughLinesCV(HoughLinesCVProps props);
	virtual ~HoughLinesCV();

	virtual bool init();
	virtual bool term();

	void setProps(HoughLinesCVProps &props);
	HoughLinesCVProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId); // throws exception if validation fails
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);
	bool handlePropsChange(frame_sp &frame);

private:
	void setMetadata(framemetadata_sp &inputMetadata);

	class Detail;
	boost::shared_ptr<Detail> mDetail;

	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
};
