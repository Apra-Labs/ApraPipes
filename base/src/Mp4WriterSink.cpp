#include <ctime>
#include <fstream>

#include "FrameMetadata.h"
#include "Frame.h"
#include "H264Utils.h"
#include "Mp4VideoMetadata.h"
#include "Mp4WriterSink.h"
#include "Mp4WriterSinkUtils.h"
#include "EncodedImageMetadata.h"
#include "Module.h"
#include "libmp4.h"
#include "PropsChangeMetadata.h"
#include "H264Metadata.h"

class DTSCalcStrategy
{
public:
	enum DTSCalcStrategyType
	{
		PASS_THROUGH = 0,
		FIXED_RATE
	};

	DTSCalcStrategy(DTSCalcStrategyType _type)
	{
		type = _type;
	}
	virtual int64_t getDTS(uint64_t& frameTS, uint64_t lastFrameTS, uint16_t fps) = 0;
	DTSCalcStrategyType type;
};

class DTSPassThroughStrategy : public DTSCalcStrategy
{
public:
	DTSPassThroughStrategy() : DTSCalcStrategy(DTSCalcStrategy::DTSCalcStrategyType::PASS_THROUGH)
	{
	}

	int64_t getDTS(uint64_t& frameTS, uint64_t lastFrameTS, uint16_t fps) override
	{
		int64_t diffInMsecs = frameTS - lastFrameTS;
		// half of the ideal duration of one frame i.e. (1/fps) secs
		int64_t halfDurationInMsecs = static_cast<int64_t>(1000 / (2 * fps));
		if (!diffInMsecs)
		{
			frameTS += halfDurationInMsecs;
		}
		else if (diffInMsecs < 0)
		{
			frameTS = lastFrameTS + halfDurationInMsecs;
		}
		diffInMsecs = frameTS - lastFrameTS;
		return diffInMsecs;
	}
};

class DTSFixedRateStrategy : public DTSCalcStrategy
{
public:
	DTSFixedRateStrategy() : DTSCalcStrategy(DTSCalcStrategy::DTSCalcStrategyType::FIXED_RATE)
	{
	}
	int64_t getDTS(uint64_t& frameTS, uint64_t lastFrameTS, uint16_t fps) override
	{
		// ideal duration of one frame i.e. (1/fps) secs
		int64_t idealDurationInMsecs = static_cast<int64_t>(1000 / fps);
		return idealDurationInMsecs;
	}
};

class DetailAbs
{
public:
	DetailAbs(Mp4WriterSinkProps &_props)
	{
		setProps(_props);
		mNextFrameFileName = "";
		mux = nullptr;
		mMetadataEnabled = false;
		/* DTS should be based on recorded timestamps of frames or on the fps prop entirely */
		if (_props.recordedTSBasedDTS)
		{
			mDTSCalc.reset(new DTSPassThroughStrategy);
		}
		else
		{
			mDTSCalc.reset(new DTSFixedRateStrategy);
		}
	};

	void setProps(Mp4WriterSinkProps& _props)
	{
		mProps.reset(new Mp4WriterSinkProps(_props.chunkTime, _props.syncTimeInSecs, _props.fps, _props.baseFolder, _props.recordedTSBasedDTS, _props.enableMetadata));
	}

	~DetailAbs()
	{
	};
	virtual bool set_video_decoder_config() = 0;
	virtual bool write(frame_container& frames) = 0;

	void setImageMetadata(framemetadata_sp& metadata)
	{
		mInputMetadata = metadata;
		mFrameType = mInputMetadata->getFrameType();
		if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE)
		{
			auto encodedImageMetadata = FrameMetadataFactory::downcast<EncodedImageMetadata>(metadata);
			mHeight = encodedImageMetadata->getHeight();
			mWidth = encodedImageMetadata->getWidth();
		}
		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
		{
			auto h264ImageMetadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
			mHeight = h264ImageMetadata->getHeight();
			mWidth = h264ImageMetadata->getWidth();
		}
	}

	bool enableMetadata(std::string& formatVersion)
	{
		mMetadataEnabled = true;
		mSerFormatVersion = formatVersion;
		return mMetadataEnabled;
	}

	void initNewMp4File(std::string& filename)
	{
		if (mux)
		{
			mp4_mux_close(mux);
		}
		syncFlag = false;
		lastFrameTS = 0;

		uint32_t timescale = 30000;
		now = std::time(nullptr);

		auto ret = mp4_mux_open(filename.c_str(), timescale, now, now, &mux);
		if (mMetadataEnabled)
		{
			/* \251too -> �too */
			std::string key = "\251too";
			std::string val = mSerFormatVersion.c_str();
			mp4_mux_add_file_metadata(mux, key.c_str(), val.c_str());
		}
		// track parameters
		params.type = MP4_TRACK_TYPE_VIDEO;
		params.name = "VideoHandler";
		params.enabled = 1;
		params.in_movie = 1;
		params.in_preview = 0;
		params.timescale = timescale;
		params.creation_time = now;
		params.modification_time = now;
		// add video track
		videotrack = mp4_mux_add_track(mux, &params);

		set_video_decoder_config();

		mp4_mux_track_set_video_decoder_config(mux, videotrack, &vdc);

		// METADATA stuff
		if (mMetadataEnabled)
		{
			metatrack_params = params;
			metatrack_params.type = MP4_TRACK_TYPE_METADATA;
			metatrack_params.name = "APRA_METADATA";
			metatrack = mp4_mux_add_track(mux, &metatrack_params);

			if (metatrack < 0)
			{
				LOG_ERROR << "Failed to add metadata track";
			}

			// https://www.rfc-editor.org/rfc/rfc4337.txt
			std::string content_encoding = "base64";
			std::string mime_format = "video/mp4";

			auto ret = mp4_mux_track_set_metadata_mime_type(
				mux,
				metatrack,
				content_encoding.c_str(),
				mime_format.c_str());
			if(ret != 0)
			{
				LOG_ERROR << "Failed to add metadata track mime type";
			}
			/* Add track reference */
			if (metatrack > 0)
			{
				LOG_INFO << "metatrack <" << metatrack << "> videotrack <" << videotrack << ">";
				ret = mp4_mux_add_ref_to_track(mux, metatrack, videotrack);
				if (ret != 0)
				{
					LOG_ERROR << "Failed to add metadata track as reference";
				}
			}
		}
	}

	bool attemptFileClose()
	{
		if (mux)
		{
			mp4_mux_close(mux);
			mux = nullptr;
		}
		return true;
	}

	bool shouldTriggerSOS()
	{
		return !mInputMetadata.get();
	}

	void addMetadataInVideoHeader(frame_sp inFrame)
	{
		if (!lastFrameTS)
		{
			/* \251sts -> ©sts */
			std::string key = "\251sts";
			std::string val = std::to_string(inFrame->timestamp);
			mp4_mux_add_file_metadata(mux, key.c_str(), val.c_str());
		}
	}

	boost::shared_ptr<Mp4WriterSinkProps> mProps;
	bool mMetadataEnabled = false;
	bool isKeyFrame;
	struct mp4_mux* mux;
	bool syncFlag = false;
protected:
	int videotrack;
	int metatrack;
	int audiotrack;
	int current_track;
	uint64_t now;
	struct mp4_mux_track_params params, metatrack_params;
	struct mp4_video_decoder_config vdc;
	struct mp4_mux_sample mux_sample;
	struct mp4_mux_prepend_buffer prepend_buffer;
	struct mp4_track_sample sample;

	int mHeight;
	int mWidth;
	short mFrameType;
	Mp4WriterSinkUtils mWriterSinkUtils;
	std::string mNextFrameFileName;
	std::string mSerFormatVersion;
	framemetadata_sp mInputMetadata;
	uint64_t lastFrameTS = 0;
	boost::shared_ptr<DTSCalcStrategy> mDTSCalc = nullptr;
};

class DetailJpeg : public DetailAbs
{
public:
	DetailJpeg(Mp4WriterSinkProps& _props) : DetailAbs(_props) {}
	bool set_video_decoder_config()
	{
		vdc.width = mWidth;
		vdc.height = mHeight;
		vdc.codec = MP4_VIDEO_CODEC_MP4V;
		return true;
	}
	bool write(frame_container& frames);
};

class DetailH264 : public DetailAbs
{
public:
	frame_sp m_headerFrame;
	const_buffer spsBuffer;
	const_buffer ppsBuffer;
	const_buffer spsBuff;
	const_buffer ppsBuff;

	DetailH264(Mp4WriterSinkProps& _props) : DetailAbs(_props)
	{
	}
	bool write(frame_container& frames);
	void modifyFrameOnNewSPSPPS(short naluType, frame_sp frame, uint8_t*& spsPpsdata, size_t& spsPpsSize, uint8_t*& frameData, size_t& frameSize);

	bool set_video_decoder_config()
	{
		vdc.width = mWidth;
		vdc.height = mHeight;
		vdc.codec = MP4_VIDEO_CODEC_AVC;
		vdc.avc.sps = reinterpret_cast<uint8_t*>(const_cast<void*>(spsBuffer.data()));
		vdc.avc.pps = reinterpret_cast<uint8_t*>(const_cast<void*>(ppsBuffer.data()));
		vdc.avc.pps_size = ppsBuffer.size();
		vdc.avc.sps_size = spsBuffer.size();
		return true;
	}
private:
};

bool DetailJpeg::write(frame_container& frames)
{
	auto inJpegImageFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
	auto inMp4MetaFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::MP4_VIDEO_METADATA);
	if (!inJpegImageFrame)
	{
		LOG_ERROR << "Image Frame is empty. Unable to write.";
		return true;
	}
	short naluType = 0;
	std::string _nextFrameFileName;
	mWriterSinkUtils.getFilenameForNextFrame(_nextFrameFileName, inJpegImageFrame->timestamp, mProps->baseFolder,
		mProps->chunkTime, mProps->syncTimeInSecs, syncFlag ,mFrameType, naluType);

	if (_nextFrameFileName == "")
	{
		LOG_ERROR << "Unable to get a filename for the next frame";
		return false;
	}

	if (mNextFrameFileName != _nextFrameFileName)
	{
		mNextFrameFileName = _nextFrameFileName;
		initNewMp4File(mNextFrameFileName);
	}

	if (syncFlag)
	{
		LOG_TRACE << "attempting to sync <" << mNextFrameFileName << ">";
		auto ret = mp4_mux_sync(mux);
		syncFlag = false;
	}

	addMetadataInVideoHeader(inJpegImageFrame);

	mux_sample.buffer = static_cast<uint8_t*>(inJpegImageFrame->data());
	mux_sample.len = inJpegImageFrame->size();
	mux_sample.sync = 0;
	int64_t diffInMsecs = 0;

	if (!lastFrameTS)
	{
		diffInMsecs = 0;
		mux_sample.dts = 0;
	}
	else
	{
		diffInMsecs = mDTSCalc->getDTS(inJpegImageFrame->timestamp, lastFrameTS, mProps->fps);
	}
	lastFrameTS = inJpegImageFrame->timestamp;
	mux_sample.dts = mux_sample.dts + static_cast<int64_t>((params.timescale / 1000) * diffInMsecs);

	mp4_mux_track_add_sample(mux, videotrack, &mux_sample);

	if (metatrack != -1 && mMetadataEnabled)
		{
			if (inMp4MetaFrame.get() && inMp4MetaFrame->fIndex == 0)
			{
				mux_sample.buffer = static_cast<uint8_t *>(inMp4MetaFrame->data());
				mux_sample.len = inMp4MetaFrame->size();
			}
			else
			{
				mux_sample.buffer = nullptr;
				mux_sample.len = 0;
			}
			
			mp4_mux_track_add_sample(mux, metatrack, &mux_sample);
		}
	return true;
}

void DetailH264::modifyFrameOnNewSPSPPS(short naluType, frame_sp inH264ImageFrame, uint8_t*& spsPpsBuffer, size_t& spsPpsSize, uint8_t*& frameData, size_t& frameSize)
{
	char NaluSeprator[3] = { 00 ,00, 00 };
	auto nalu = reinterpret_cast<uint8_t*>(NaluSeprator);
	spsPpsSize = spsBuffer.size() + ppsBuffer.size() + 8;
	if (naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		frameSize = inH264ImageFrame->size();
	}
	//Add the size of sps and pps to I frame - (First frame of the video)
	else if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)
	{
		frameSize = inH264ImageFrame->size() + spsPpsSize;
	}
	spsPpsBuffer = new uint8_t[spsPpsSize + 4];
	//add the size of sps to the 4th byte of sps's nalu seprator (00 00 00 SpsSize 67)
	memcpy(spsPpsBuffer, nalu, 3);
	spsPpsBuffer += 3;
	spsPpsBuffer[0] = spsBuffer.size();
	spsPpsBuffer += 1;
	memcpy(spsPpsBuffer, spsBuffer.data(), spsBuffer.size());
	spsPpsBuffer += spsBuffer.size();

	//add the size of sps to the 4th byte of pps's nalu seprator (00 00 00 PpsSize 68)
	memcpy(spsPpsBuffer, nalu, 3);
	spsPpsBuffer += 3;
	spsPpsBuffer[0] = ppsBuffer.size();
	spsPpsBuffer += 1;
	memcpy(spsPpsBuffer, ppsBuffer.data(), ppsBuffer.size());
	spsPpsBuffer += ppsBuffer.size();

	//add the size of I frame to the I frame's nalu seprator
	spsPpsBuffer[0] = (frameSize - spsPpsSize - 4 >> 24) & 0xFF;
	spsPpsBuffer[1] = (frameSize - spsPpsSize - 4 >> 16) & 0xFF;
	spsPpsBuffer[2] = (frameSize - spsPpsSize - 4 >> 8) & 0xFF;
	spsPpsBuffer[3] = frameSize - spsPpsSize - 4 & 0xFF;

	frameData = reinterpret_cast<uint8_t*>(inH264ImageFrame->data());
	if (naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		frameData = frameData + spsPpsSize + 4;
		frameSize = frameSize - spsPpsSize - 4;
	}
	else if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)
	{
		frameData = frameData + 4;
		frameSize -= 4;
	}
	spsPpsBuffer = spsPpsBuffer - spsPpsSize;
	spsPpsSize += 4;
}

bool DetailH264::write(frame_container& frames)
{
	auto inH264ImageFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::H264_DATA);
	auto inMp4MetaFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::MP4_VIDEO_METADATA);
	if (!inH264ImageFrame)
	{
		LOG_ERROR << "Image Frame is empty. Unable to write.";
		return true;
	}

	auto mFrameBuffer = const_buffer(inH264ImageFrame->data(), inH264ImageFrame->size());
	auto ret = H264Utils::parseNalu(mFrameBuffer);
	short typeFound;
	tie(typeFound, spsBuff, ppsBuff) = ret;

	if ((spsBuff.size() !=0 ) || (ppsBuff.size() != 0))
	{
		m_headerFrame = inH264ImageFrame; //remember this forever.
		spsBuffer = spsBuff;
		ppsBuffer = ppsBuff;
	}
	auto naluType = H264Utils::getNALUType((char*)mFrameBuffer.data());
	std::string _nextFrameFileName;
	mWriterSinkUtils.getFilenameForNextFrame(_nextFrameFileName,inH264ImageFrame->timestamp, mProps->baseFolder,
		mProps->chunkTime, mProps->syncTimeInSecs, syncFlag,mFrameType, naluType);

	if (_nextFrameFileName == "")
	{
		LOG_ERROR << "Unable to get a filename for the next frame";
		return false;
	}

	uint8_t* spsPpsBuffer = nullptr;
	size_t spsPpsSize;
	uint8_t* frameData = nullptr;
	size_t frameSize;
	if (mNextFrameFileName != _nextFrameFileName)
	{
		mNextFrameFileName = _nextFrameFileName;
		initNewMp4File(mNextFrameFileName);
		if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
		{
			// new video 
			modifyFrameOnNewSPSPPS(naluType, inH264ImageFrame, spsPpsBuffer, spsPpsSize, frameData, frameSize);
			prepend_buffer.buffer = spsPpsBuffer;
			prepend_buffer.len = spsPpsSize;
			mux_sample.buffer = frameData;
			mux_sample.len = frameSize;
		}
	}
	else if (naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		// new sps pps
		modifyFrameOnNewSPSPPS(naluType, inH264ImageFrame, spsPpsBuffer, spsPpsSize, frameData, frameSize);
		prepend_buffer.buffer = spsPpsBuffer;
		prepend_buffer.len = spsPpsSize;
		mux_sample.buffer = frameData;
		mux_sample.len = frameSize;
	}
	else
	{
		uint8_t* naluData = new uint8_t[4];
		// assign size of the frame to the NALU seperator for playability in default players
		naluData[0] = (inH264ImageFrame->size() - 4 >> 24) & 0xFF;
		naluData[1] = (inH264ImageFrame->size() - 4 >> 16) & 0xFF;
		naluData[2] = (inH264ImageFrame->size() - 4 >> 8) & 0xFF;
		naluData[3] = inH264ImageFrame->size() - 4 & 0xFF;

		prepend_buffer.buffer = naluData;
		prepend_buffer.len = 4;

		uint8_t* frameData = static_cast<uint8_t*>(inH264ImageFrame->data());
		mux_sample.buffer = frameData + 4;
		mux_sample.len = inH264ImageFrame->size() - 4;
	}

	if (syncFlag)
	{
		LOG_TRACE << "attempting to sync <" << mNextFrameFileName << ">";
		auto ret = mp4_mux_sync(mux);
		syncFlag = false;
	}

	isKeyFrame = false;

	if (naluType == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_SEQ_PARAM)
	{
		isKeyFrame = true;
	}

	addMetadataInVideoHeader(inH264ImageFrame);

	mux_sample.sync = isKeyFrame ? 1 : 0;
	int64_t diffInMsecs = 0;

	if (!lastFrameTS)
	{
		diffInMsecs = 0;
		mux_sample.dts = 0;
	}
	else
	{
		diffInMsecs = mDTSCalc->getDTS(inH264ImageFrame->timestamp, lastFrameTS, mProps->fps);
	}
	lastFrameTS = inH264ImageFrame->timestamp;
	mux_sample.dts = mux_sample.dts + static_cast<int64_t>((params.timescale / 1000) * diffInMsecs);

	mp4_mux_track_add_sample_with_prepend_buffer(mux, videotrack, &prepend_buffer, &mux_sample);

	if (metatrack != -1 && mMetadataEnabled)
	{
		// null check must happen here to ensure 0 sized entries in mp4 metadata track's stsz table
		// this will ensure equal entries in metadata and video tracks 
		if (inMp4MetaFrame.get() && inMp4MetaFrame->size())
		{
			mux_sample.buffer = static_cast<uint8_t*>(inMp4MetaFrame->data());
			mux_sample.len = inMp4MetaFrame->size();
		}
		else
		{
			mux_sample.buffer = nullptr;
			mux_sample.len = 0;
		}
		mp4_mux_track_add_sample(mux, metatrack, &mux_sample);
	}
	return true;
}

Mp4WriterSink::Mp4WriterSink(Mp4WriterSinkProps _props)
	: Module(SINK, "Mp4WriterSink", _props), mProp(_props)
{
}

Mp4WriterSink::~Mp4WriterSink() {}

bool Mp4WriterSink::init()
{
	bool enableVideoMetadata = false;
	framemetadata_sp mp4VideoMetadata;
	if (!Module::init())
	{
		return false;
	}
	auto inputPinIdMetadataMap = getInputMetadata();

	for (auto const& element : inputPinIdMetadataMap)
	{
		auto metadata = element.second;
		auto mFrameType = metadata->getFrameType();
		if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE)
		{
			mDetail.reset(new DetailJpeg(mProp));
		}

		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
		{
			mDetail.reset(new DetailH264(mProp));
		}

		else if (mFrameType == FrameMetadata::FrameType::MP4_VIDEO_METADATA && mProp.enableMetadata)
		{
			enableVideoMetadata = true;
			mp4VideoMetadata = metadata;
		}
	}
	if (enableVideoMetadata)
	{
		enableMp4Metadata(mp4VideoMetadata);
	}
	return Module::init();
}

bool Mp4WriterSink::validateInputOutputPins()
{
	if (getNumberOfInputsByType(FrameMetadata::H264_DATA) != 1 && getNumberOfInputsByType(FrameMetadata::ENCODED_IMAGE) != 1) 
	{
		LOG_ERROR << "<" << getId() << ">::validateInputOutputPins expected 1 pin of ENCODED_IMAGE. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}
	return true;
}

bool Mp4WriterSink::validateInputPins()
{
	if (getNumberOfInputPins() > 5)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	auto inputPinIdMetadataMap = getInputMetadata();
	for (auto const& element : inputPinIdMetadataMap)
	{
		auto& metadata = element.second;
		auto mFrameType = metadata->getFrameType();
		if (mFrameType != FrameMetadata::ENCODED_IMAGE && mFrameType != FrameMetadata::MP4_VIDEO_METADATA && mFrameType != FrameMetadata::H264_DATA)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be ENCODED_IMAGE or MP4_VIDEO_METADATA. Actual<" << mFrameType << ">";
			return false;
		}

		FrameMetadata::MemType memType = metadata->getMemType();
		if (memType != FrameMetadata::MemType::HOST)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
			return false;
		}
	}
	return true;
}
bool Mp4WriterSink::setMetadata(framemetadata_sp& inputMetadata)
{
	mDetail->setImageMetadata(inputMetadata);
	return true;
}

bool Mp4WriterSink::enableMp4Metadata(framemetadata_sp &inputMetadata)
{
	auto mp4VideoMetadata = FrameMetadataFactory::downcast<Mp4VideoMetadata>(inputMetadata);
	std::string formatVersion = mp4VideoMetadata->getVersion();
	if (formatVersion.empty())
	{
		LOG_ERROR << "Serialization Format Information missing from the metadata. Metadata writing will be disabled";
		return false;
	}
	mDetail->enableMetadata(formatVersion);
	return true;
}

void Mp4WriterSink::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
}

bool Mp4WriterSink::processSOS(frame_sp& frame)
{
	auto inputMetadata = frame->getMetadata();
	auto mFrameType = inputMetadata->getFrameType();
	if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE || mFrameType == FrameMetadata::FrameType::H264_DATA)
	{
		setMetadata(inputMetadata);
	}

	if (mFrameType == FrameMetadata::FrameType::MP4_VIDEO_METADATA)
	{
		auto mp4VideoMetadata = FrameMetadataFactory::downcast<Mp4VideoMetadata>(inputMetadata);
		std::string formatVersion = mp4VideoMetadata->getVersion();
		if (formatVersion.empty())
		{
			LOG_ERROR << "Serialization Format Information missing from the metadata. Metadata writing will be disabled";
			return true;
		}
		mDetail->enableMetadata(formatVersion);
	}
	return true;
}

bool Mp4WriterSink::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

bool Mp4WriterSink::term()
{
	mDetail->attemptFileClose();
	return true;
}

bool Mp4WriterSink::process(frame_container& frames)
{
	try
	{
		if (!mDetail->write(frames))
		{
			LOG_FATAL << "Error occured while writing mp4 file<>";
			return true;
		}
	}
	catch (const std::exception& e)
	{
		LOG_ERROR << e.what();
		// close any open file
		mDetail->attemptFileClose();
	}
	return true;
}

bool Mp4WriterSink::processEOS(string& pinId)
{
	return true;
}

Mp4WriterSinkProps Mp4WriterSink::getProps()
{
	auto tempProps = Mp4WriterSinkProps(mDetail->mProps->chunkTime, mDetail->mProps->syncTimeInSecs, mDetail->mProps->fps, mDetail->mProps->baseFolder);
	fillProps(tempProps);
	return tempProps;
}

bool Mp4WriterSink::handlePropsChange(frame_sp& frame)
{
	Mp4WriterSinkProps props;
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void Mp4WriterSink::setProps(Mp4WriterSinkProps& props)
{
	Module::addPropsToQueue(props, true);
}

bool Mp4WriterSink::doMp4MuxSync()
{
	auto ret = mp4_mux_sync(mDetail->mux);
	mDetail->syncFlag = false;
	return ret;
}