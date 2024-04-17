# pragma once

#include "Module.h"

class SceneDescriptorXFormProps : public ModuleProps {
public:
	enum SceneDescriptorStrategy {
		LLAVA = 0
	};
	
	SceneDescriptorXFormProps(SceneDescriptorStrategy _modelStrategyType) {
		modelStrategyType = _modelStrategyType;
	}

	size_t getSerializeSize() {
		return ModuleProps::getSerializeSize() + sizeof(modelStrategyType);
	}

	SceneDescriptorStrategy modelStrategyType;
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar & modelStrategyType;
	}
};

class SceneDescriptorXForm : public Module {
public:
	SceneDescriptorXForm(SceneDescriptorXFormProps _props);
	virtual ~SceneDescriptorXForm();
	bool init();
	bool term();
	void setProps(SceneDescriptorXFormProps& props);
	SceneDescriptorXFormProps getProps();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};