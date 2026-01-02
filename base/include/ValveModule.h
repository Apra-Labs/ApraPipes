#pragma once 
#include "Module.h"

class ValveModuleProps : public ModuleProps
{
public:
	ValveModuleProps() 
	{

	}

	ValveModuleProps(uint64 _noOfFramesToCapture)
	{
		noOfFramesToCapture = _noOfFramesToCapture;
	}
	uint64 noOfFramesToCapture = 10;
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(noOfFramesToCapture);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& noOfFramesToCapture;
	}
};

class ValveModule : public Module
{
public:
	ValveModule(ValveModuleProps _props);
	~ValveModule();
	bool init() override;
	bool term() override;
	/* We can set the number of frames property by passing as
	arguement to allowFrames else module props value is taken */
	bool allowFrames(int numframes); 
	void setProps(ValveModuleProps& props);
	ValveModuleProps getProps();
	bool setNext(std::shared_ptr<Module> next, bool open = true, bool sieve = false) override;
protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool validateInputOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override;
	bool handleCommand(Command::CommandType type, frame_sp& frame) override;
	bool handlePropsChange(frame_sp& frame) override;

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	std::shared_ptr<Detail> mDetail;
};