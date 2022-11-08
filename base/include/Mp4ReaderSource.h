#pragma once

#include "Module.h"
class Detail;
class DetailJpeg;
class DetailH264;

class Mp4ReaderSourceProps : public ModuleProps
{
public:
	Mp4ReaderSourceProps() : ModuleProps()
	{

	}

	Mp4ReaderSourceProps(std::string _videoPath, bool _parseFS, size_t _biggerFrameSize, size_t _biggerMetadataFrameSize) : ModuleProps()
	{
		biggerFrameSize = _biggerFrameSize;
		biggerMetadataFrameSize = _biggerMetadataFrameSize;
		videoPath = _videoPath;
		parseFS = _parseFS;
		if (parseFS)
		{
			skipDir = boost::filesystem::path(videoPath).parent_path().parent_path().parent_path().string();
		}

	}

	Mp4ReaderSourceProps(std::string _videoPath, bool _parseFS) : ModuleProps()
	{
		videoPath = _videoPath;
		parseFS = _parseFS;
		if (parseFS)
		{
			skipDir = boost::filesystem::path(videoPath).parent_path().parent_path().parent_path().string();
		}

	}


	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(videoPath) + sizeof(parseFS) + sizeof(skipDir);
	}

	std::string skipDir = "./data/mp4_videos";
	std::string videoPath = "";
	size_t biggerFrameSize = 600000;
	size_t biggerMetadataFrameSize = 60000;
	bool parseFS = true;
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& videoPath;
		ar& parseFS;
		ar& skipDir;
		ar& biggerFrameSize;
		ar& biggerMetadataFrameSize;
	}
};

class Mp4ReaderSource : public Module
{
public:
	Mp4ReaderSource(Mp4ReaderSourceProps _props);
	virtual ~Mp4ReaderSource();
	bool init();
	bool term();
	Mp4ReaderSourceProps getProps();
	void setProps(Mp4ReaderSourceProps& props);
	std::string addOutPutPin(framemetadata_sp& metadata);
	bool randomSeek(uint64_t skipTS);
protected:
	bool produce();
	bool validateOutputPins();
	bool handleCommand(Command::CommandType type, frame_sp& fame);
	bool handlePropsChange(frame_sp& frame);
private:
	std::string h264ImagePinId;
	std::string encodedImagePinId;
	std::string mp4FramePinId;
	int outImageFrameType;
	boost::shared_ptr<Detail> mDetail;
	Mp4ReaderSourceProps props;
	std::function<frame_sp(size_t size)> _makeFrame;
	std::function<framemetadata_sp(int type)> _getOutputMetadataByType;
};