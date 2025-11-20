#pragma once

#include "Module.h"

class VirtualPTZProps : public ModuleProps
{
public:
	VirtualPTZProps()
	{
		roiHeight = roiWidth = 1;
		roiX = roiY = 0;
	}
	VirtualPTZProps(float _roiWidth, float _roiHeight, float _roiX, float _roiY) : roiWidth(_roiWidth), roiHeight(_roiHeight), roiX(_roiX), roiY(_roiY)
	{
	}

	float roiX;
	float roiY;
	float roiWidth;
	float roiHeight;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(float) * 4;
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &roiX &roiY &roiWidth &roiHeight;
	}
};

class VirtualPTZ : public Module
{

public:
	VirtualPTZ(VirtualPTZProps _props);
	virtual ~VirtualPTZ();
	bool init() override;
	bool term() override;
	void setProps(VirtualPTZProps &props);
	VirtualPTZProps getProps();

protected:
	bool process(frame_container &frames) override;
	bool processSOS(frame_sp &frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	void addInputPin(framemetadata_sp &metadata, std::string_view pinId) override;
	bool handlePropsChange(frame_sp &frame) override;

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	std::shared_ptr<Detail> mDetail;
};