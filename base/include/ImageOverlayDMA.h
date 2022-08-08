#pragma once
#include "Module.h"

class ImageOverlayProps : public ModuleProps
{
public:
	ImageOverlayProps(int _topLeft, int _topRight, float _alpha) : topLeft(_topLeft), topRight(_topRight), alpha(_alpha)
	{
	}

	int topLeft, topRight;
	float alpha;
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + 2 * sizeof(int) + sizeof(float);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &topRight &topLeft &alpha;
	}
};

class ImageOverlay : public Module
{

public:
	ImageOverlay(ImageOverlayProps _props);
	virtual ~ImageOverlay();
	bool init();
	bool term();
	void setProps(ImageOverlayProps &props);
	ImageOverlayProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp &metadata, string &pinId);
	void setProps(ImageOverlay);
	bool handlePropsChange(frame_sp &frame);

private:
	void setMetadata(framemetadata_sp &metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};