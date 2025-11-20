#pragma once
#include "Module.h"
#include "AIPExceptions.h"
#include <filesystem>

class ArchiveSpaceManagerProps : public ModuleProps
{
public:
	ArchiveSpaceManagerProps(uint64_t _lowerWaterMark, uint64_t _upperWaterMark, string _pathToWatch, int _samplingFreq)
	{
		lowerWaterMark = _lowerWaterMark;
		upperWaterMark = _upperWaterMark;
		pathToWatch = _pathToWatch;
		samplingFreq = _samplingFreq;
		fps = 0.001;

		auto totalSpace = std::filesystem::space(pathToWatch);
		if ((lowerWaterMark > upperWaterMark) || (upperWaterMark > totalSpace.capacity))
		{
			LOG_ERROR << "Please enter correct properties!";
			std::string errorMsg = "Incorrect properties set for Archive Manager. TotalDiskCapacity <" + std::to_string(totalSpace.capacity) + ">lowerWaterMark<" + std::to_string(lowerWaterMark) + "> UpperWaterMark<" + std::to_string(upperWaterMark) + ">";
			throw AIPException(AIP_FATAL, errorMsg);
		}
	}

	ArchiveSpaceManagerProps(uint64_t maxSizeAllowed, string _pathToWatch, int _samplingFreq)
	{
		lowerWaterMark = maxSizeAllowed - (maxSizeAllowed / 10);
		upperWaterMark = maxSizeAllowed;
		pathToWatch = _pathToWatch;
		samplingFreq = _samplingFreq;
		fps = 0.001;

		auto totalSpace = std::filesystem::space(pathToWatch);
		if ((lowerWaterMark > upperWaterMark) || (upperWaterMark > totalSpace.capacity))
		{
			LOG_ERROR << "Please enter correct properties!";
			std::string errorMsg = "Incorrect properties set for Archive Manager. TotalDiskCapacity <" + std::to_string(totalSpace.capacity) + ">lowerWaterMark<" + std::to_string(lowerWaterMark) + "> UpperWaterMark<" + std::to_string(upperWaterMark) + ">";
			throw AIPException(AIP_FATAL, errorMsg);
		}
	}


	uint64_t lowerWaterMark; // Lower disk space
	uint64_t upperWaterMark; // Higher disk space
	std::string pathToWatch;
	int samplingFreq;
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(lowerWaterMark) + sizeof(upperWaterMark) + sizeof(pathToWatch) + sizeof(samplingFreq);
	}
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& lowerWaterMark;
		ar& upperWaterMark;
		ar& pathToWatch;
		ar& samplingFreq;
	}
};


class ArchiveSpaceManager : public Module {
public:
	ArchiveSpaceManager(ArchiveSpaceManagerProps _props);

	virtual ~ArchiveSpaceManager() {
	}
	bool init() override;
	bool term() override;
	uint64_t finalArchiveSpace = 0;
	void setProps(ArchiveSpaceManagerProps& props);
	ArchiveSpaceManagerProps getProps();

protected:
	bool produce() override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool validateInputOutputPins() override;
	void addInputPin(framemetadata_sp& metadata, std::string_view pinId) override;
	bool handlePropsChange(frame_sp& frame) override;
private:

	class Detail;
	std::shared_ptr<Detail> mDetail;
	bool checkDirectory = true;
};