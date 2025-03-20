// #include <ctime>
// #include <fstream>
// #include <vector>

// #include "FrameMetadata.h"
// #include "Frame.h"
// #include "H264Utils.h"
// #include "Mp4VideoMetadata.h"
// #include "Mp4WriterSink.h"
// #include "Mp4WriterSinkUtils.h"
// #include "EncodedImageMetadata.h"
// #include "Module.h"
// #include "libmp4.h"
// #include "PropsChangeMetadata.h"
// #include "H264Metadata.h"
// #include <openssl/sha.h>

// class DetailAbs
// {
// public:
// 	DetailAbs(Mp4WriterSinkProps _module)
// 	{
// 		setProps(_module);
// 		mNextFrameFileName = "";
// 		mux = nullptr;
// 		mMetadataEnabled = false;
// 		m_frameNumber = 0;
		
// 	};

// 	void setProps(Mp4WriterSinkProps& _props)
// 	{
// 		mProps.reset(new Mp4WriterSinkProps(_props.chunkTime, _props.syncTimeInSecs, _props.fps, _props.baseFolder));
// 	}

// 	~DetailAbs()
// 	{
// 	};
// 	virtual bool set_video_decoder_config() = 0;
// 	virtual bool write(frame_container& frames) = 0;

// 	void setImageMetadata(framemetadata_sp& metadata)
// 	{
// 		mInputMetadata = metadata;
// 		mFrameType = mInputMetadata->getFrameType();
// 		if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE)
// 		{
// 			auto encodedImageMetadata = FrameMetadataFactory::downcast<EncodedImageMetadata>(metadata);
// 			mHeight = encodedImageMetadata->getHeight();
// 			mWidth = encodedImageMetadata->getWidth();
// 		}
// 		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
// 		{
// 			auto h264ImageMetadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
// 			mHeight = h264ImageMetadata->getHeight();
// 			mWidth = h264ImageMetadata->getWidth();
// 		}
// 	}

// 	bool enableMetadata(std::string& formatVersion)
// 	{
// 		mMetadataEnabled = true;
// 		mSerFormatVersion = formatVersion;
// 		return mMetadataEnabled;
// 	}

// 	void initNewMp4File(std::string& filename)
// 	{
// 		if (mux)
// 		{
// 			mp4_mux_close(mux);
// 		}
// 		syncFlag = false;
// 		lastFrameTS = 0;

// 		uint32_t timescale = 30000;
// 		now = std::time(nullptr);

// 		auto ret = mp4_mux_open(filename.c_str(), timescale, now, now, &mux);
// 		if (mMetadataEnabled)
// 		{
// 			LOG_ERROR << "mMetadataEnabled Called!\n";
// 			/* \251too -> �too */
// 			std::string key = "\251too";
// 			// std::string val = mSerFormatVersion.c_str();
// 			std::string val = "gibberish data";
// 			mp4_mux_add_file_metadata(mux, key.c_str(), val.c_str());
// 		}
// 		// track parameters
// 		params.type = MP4_TRACK_TYPE_VIDEO;
// 		params.name = "VideoHandler";
// 		params.enabled = 1;
// 		params.in_movie = 1;
// 		params.in_preview = 0;
// 		params.timescale = timescale;
// 		params.creation_time = now;
// 		params.modification_time = now;
// 		// add video track
// 		videotrack = mp4_mux_add_track(mux, &params);

// 		set_video_decoder_config();

// 		mp4_mux_track_set_video_decoder_config(mux, videotrack, &vdc);

// 		// METADATA stuff
// 		if (mMetadataEnabled)
// 		{
// 			metatrack_params = params;
// 			metatrack_params.type = MP4_TRACK_TYPE_METADATA;
// 			metatrack_params.name = "APRA_METADATA";
// 			metatrack = mp4_mux_add_track(mux, &metatrack_params);

// 			if (metatrack < 0)
// 			{
// 				LOG_ERROR << "Failed to add metadata track";
// 				// #Dec_24_Review - should we throw exception here ? This means that user sends metadata to this module but we don't write ?
// 			}

// 			// https://www.rfc-editor.org/rfc/rfc4337.txt
// 			std::string content_encoding = "base64";
// 			std::string mime_format = "video/mp4";

// 			// #Dec_24_Review - use return code from the below function call 
// 			mp4_mux_track_set_metadata_mime_type(
// 				mux,
// 				metatrack,
// 				content_encoding.c_str(),
// 				mime_format.c_str());

// 			/* Add track reference */
// 			if (metatrack > 0)
// 			{
// 				LOG_INFO << "metatrack <" << metatrack << "> videotrack <" << videotrack << ">";
// 				ret = mp4_mux_add_ref_to_track(mux, metatrack, videotrack);
// 				if (ret != 0)
// 				{
// 					LOG_ERROR << "Failed to add metadata track as reference";
// 					// #Dec_24_Review - should we throw exception here ? This means that user sends metadata to this module but we don't write ?
// 				}
// 			}
// 		}
// 	}

// 	bool attemptFileClose()
// 	{
// 		LOG_ERROR << "ATTEMPTING TO CLOSE FILE";
// 		if (mux)
// 		{
// 			LOG_ERROR << "Got Mux OPen Closing File";
// 			int status = mp4_mux_close(mux);
// 			LOG_ERROR << "Status of File CLosed " << status;
// 		}
// 		mux = NULL;
// 		return true;
// 	}

// 	bool shouldTriggerSOS()
// 	{
// 		return !mInputMetadata.get();
// 	}

// 	void addMetadataInVideoHeader(frame_sp inFrame)
// 	{
// 		if (!lastFrameTS)
// 		{
// 			/* \251sts -> ©sts */
// 			std::string key = "\251sts";
// 			std::string val = std::to_string(inFrame->timestamp);
// 			mp4_mux_add_file_metadata(mux, key.c_str(), val.c_str());
// 		}
// 	}

// 	void addMetadataInVideoHeaderAtEnd(const char* key, const char* data)
// 	{
// 		LOG_ERROR << "adding metadata <" << key << "> <" << data << ">";
// 		mp4_mux_add_file_metadata(mux, key, data);
// 	}

// 	boost::shared_ptr<Mp4WriterSinkProps> mProps;
// 	bool mMetadataEnabled = false;
// 	bool isKeyFrame;
// 	uint64_t m_frameNumber = 0;
// 	vector<uint64_t> queuedFrames;
// protected:
// 	int videotrack;
// 	int metatrack;
// 	int audiotrack;
// 	int current_track;
// 	uint64_t now;
// 	struct mp4_mux* mux;
// 	struct mp4_mux_track_params params, metatrack_params;
// 	struct mp4_video_decoder_config vdc;
// 	struct mp4_mux_sample mux_sample;
// 	struct mp4_track_sample sample;

// 	int mHeight;
// 	int mWidth;
// 	short mFrameType;
// 	bool syncFlag = false;
// 	Mp4WriterSinkUtils mWriterSinkUtils;
// 	std::string mNextFrameFileName;
// 	std::string mSerFormatVersion;
// 	framemetadata_sp mInputMetadata;
// 	uint64_t lastFrameTS = 0;
// };

// class DetailJpeg : public DetailAbs
// {
// public:
// 	DetailJpeg(Mp4WriterSinkProps& _props) : DetailAbs(_props) {}
// 	bool set_video_decoder_config()
// 	{
// 		vdc.width = mWidth;
// 		vdc.height = mHeight;
// 		vdc.codec = MP4_VIDEO_CODEC_MP4V;
// 		return true;
// 	}
// 	bool write(frame_container& frames);
// };

// class DetailH264 : public DetailAbs
// {
// public:
// 	frame_sp m_headerFrame;
// 	const_buffer spsBuffer;
// 	const_buffer ppsBuffer;
// 	const_buffer spsBuff;
// 	const_buffer ppsBuff;
// 	short typeFound;

// 	DetailH264(Mp4WriterSinkProps& _props) : DetailAbs(_props)
// 	{
// 	}
// 	bool write(frame_container& frames);
// 	uint8_t* AppendSizeInNaluSeprator(short naluType, frame_sp frame, size_t& frameSize);
// 	bool set_video_decoder_config()
// 	{
// 		vdc.width = mWidth;
// 		vdc.height = mHeight;
// 		vdc.codec = MP4_VIDEO_CODEC_AVC;
// 		vdc.avc.sps = reinterpret_cast<uint8_t*>(const_cast<void*>(spsBuffer.data()));
// 		vdc.avc.pps = reinterpret_cast<uint8_t*>(const_cast<void*>(ppsBuffer.data()));
// 		vdc.avc.pps_size = ppsBuffer.size();
// 		vdc.avc.sps_size = spsBuffer.size();
// 		return true;
// 	}
// private:
// };

// bool DetailJpeg::write(frame_container& frames)
// {
// 	auto inJpegImageFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
// 	auto inMp4MetaFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::MP4_VIDEO_METADATA);
// 	if (!inJpegImageFrame)
// 	{
// 		LOG_ERROR << "Image Frame is empty. Unable to write.";
// 		return true;
// 	}
// 	short naluType = 0;
// 	std::string _nextFrameFileName;
// 	mWriterSinkUtils.getFilenameForNextFrame(_nextFrameFileName, inJpegImageFrame->timestamp, mProps->baseFolder, mProps->chunkTime, mProps->syncTimeInSecs, syncFlag ,mFrameType, naluType);

// 	if (_nextFrameFileName == "")
// 	{
// 		LOG_ERROR << "Unable to get a filename for the next frame";
// 		return false;
// 	}

// 	if (mNextFrameFileName != _nextFrameFileName)
// 	{
// 		mNextFrameFileName = _nextFrameFileName;
// 		initNewMp4File(mNextFrameFileName);
// 	}

// 	if (syncFlag)
// 	{
// 		mp4_mux_sync(mux);
// 		syncFlag = false;
// 	}

// 	addMetadataInVideoHeader(inJpegImageFrame);

// 	mux_sample.buffer = static_cast<uint8_t*>(inJpegImageFrame->data());
// 	mux_sample.len = inJpegImageFrame->size();
// 	mux_sample.sync = 0;
// 	int64_t diffInMsecs = 0;

// 	if (!lastFrameTS)
// 	{
// 		diffInMsecs = 0;
// 		mux_sample.dts = 0;
// 	}
// 	else
// 	{
// 		diffInMsecs = inJpegImageFrame->timestamp - lastFrameTS;
// 		int64_t halfDurationInMsecs = static_cast<int64_t>(1000 / (2 * mProps->fps));
// 		if (!diffInMsecs)
// 		{
// 			inJpegImageFrame->timestamp += halfDurationInMsecs;
// 		}
// 		else if (diffInMsecs < 0)
// 		{
// 			inJpegImageFrame->timestamp = lastFrameTS + halfDurationInMsecs;
// 		}
// 		diffInMsecs = inJpegImageFrame->timestamp - lastFrameTS;
// 	}
// 	lastFrameTS = inJpegImageFrame->timestamp;
// 	mux_sample.dts = mux_sample.dts + static_cast<int64_t>((params.timescale / 1000) * diffInMsecs);

// 	mp4_mux_track_add_sample(mux, videotrack, &mux_sample);

// 	if (metatrack != -1 && mMetadataEnabled && inMp4MetaFrame.get())
// 	{
// 		mux_sample.buffer = static_cast<uint8_t*>(inMp4MetaFrame->data());
// 		mux_sample.len = inMp4MetaFrame->size();
// 		mp4_mux_track_add_sample(mux, metatrack, &mux_sample);
// 	}
// 	return true;
// }

// uint8_t* DetailH264::AppendSizeInNaluSeprator(short naluType, frame_sp inH264ImageFrame, size_t& frameSize)
// {
// 	char NaluSeprator[3] = { 00 ,00, 00 };
// 	auto nalu = reinterpret_cast<uint8_t*>(NaluSeprator);
// 	uint spsPpsSize = spsBuffer.size() + ppsBuffer.size() + 8;
// 	if (naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
// 	{
// 		frameSize = inH264ImageFrame->size();
// 	}
// 	//Add the size of sps and pps to I frame - (First frame of the video)
// 	else if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)
// 	{
// 		frameSize = inH264ImageFrame->size() + spsPpsSize;
// 	}
// 	uint8_t* newBuffer = new uint8_t[frameSize];
// 	//add the size of sps to the 4th byte of sps's nalu seprator (00 00 00 SpsSize 67)
// 	memcpy(newBuffer, nalu, 3);
// 	newBuffer += 3;
// 	newBuffer[0] = spsBuffer.size();
// 	newBuffer += 1;
// 	memcpy(newBuffer, spsBuffer.data(), spsBuffer.size());
// 	newBuffer += spsBuffer.size();

// 	//add the size of sps to the 4th byte of pps's nalu seprator (00 00 00 PpsSize 68)
// 	memcpy(newBuffer, nalu, 3);
// 	newBuffer += 3;
// 	newBuffer[0] = ppsBuffer.size();
// 	newBuffer += 1;
// 	memcpy(newBuffer, ppsBuffer.data(), ppsBuffer.size());
// 	newBuffer += ppsBuffer.size();

// 	//add the size of I frame to the I frame's nalu seprator
// 	newBuffer[0] = (frameSize - spsPpsSize - 4 >> 24) & 0xFF;
// 	newBuffer[1] = (frameSize - spsPpsSize - 4 >> 16) & 0xFF;
// 	newBuffer[2] = (frameSize - spsPpsSize - 4 >> 8) & 0xFF;
// 	newBuffer[3] = frameSize - spsPpsSize - 4 & 0xFF;
// 	newBuffer += 4;

// 	uint8_t* tempBuffer = reinterpret_cast<uint8_t*>(inH264ImageFrame->data());
// 	if (naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
// 	{
// 		tempBuffer = tempBuffer + spsPpsSize + 4;
// 	}
// 	else if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)
// 	{
// 		tempBuffer = tempBuffer + 4;
// 	}
// 	//copy I frame data to the buffer
// 	memcpy(newBuffer, tempBuffer, frameSize - spsPpsSize - 4);
// 	//set the pointer to the starting of frame
// 	newBuffer -= spsPpsSize + 4;
// 	return newBuffer;
// }

// bool DetailH264::write(frame_container& frames)
// {
// 	auto inH264ImageFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::H264_DATA);
// 	auto inMp4MetaFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::MP4_VIDEO_METADATA);
// 	if (!inH264ImageFrame)
// 	{
// 		LOG_ERROR << "Image Frame is empty. Unable to write.";
// 		return true;
// 	}

// 	auto mFrameBuffer = const_buffer(inH264ImageFrame->data(), inH264ImageFrame->size());
// 	auto ret = H264Utils::parseNalu(mFrameBuffer);
// 	tie(typeFound, spsBuff, ppsBuff) = ret;

// 	if ((spsBuff.size() !=0 ) || (ppsBuff.size() != 0))
// 	{
// 		m_headerFrame = inH264ImageFrame; //remember this forever.
// 		spsBuffer = spsBuff;
// 		ppsBuffer = ppsBuff;
// 	}

// 	std::string _nextFrameFileName;
// 	mWriterSinkUtils.getFilenameForNextFrame(_nextFrameFileName,inH264ImageFrame->timestamp, mProps->baseFolder,
// 		mProps->chunkTime, mProps->syncTimeInSecs, syncFlag,mFrameType, typeFound);

// 	if (_nextFrameFileName == "")
// 	{
// 		LOG_ERROR << "Unable to get a filename for the next frame";
// 		return false;
// 	}

// 	uint8_t* frameData = reinterpret_cast<uint8_t*>(inH264ImageFrame->data());
// 	// assign size of the frame to the last two bytes of the NALU seperator for playability in default players
// 	frameData[0] = (inH264ImageFrame->size() - 4 >> 24) & 0xFF;
// 	frameData[1] = (inH264ImageFrame->size() - 4 >> 16) & 0xFF;
// 	frameData[2] = (inH264ImageFrame->size() - 4 >> 8) & 0xFF;
// 	frameData[3] = inH264ImageFrame->size() - 4 & 0xFF;

// 	mux_sample.buffer = frameData;
// 	mux_sample.len = inH264ImageFrame->size();
// 	auto naluType = H264Utils::getNALUType((char*)mFrameBuffer.data());
// 	size_t frameSize;

// 	if (mNextFrameFileName != _nextFrameFileName)
// 	{
// 		mNextFrameFileName = _nextFrameFileName;
// 		initNewMp4File(mNextFrameFileName);
// 		if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)
// 		{
// 			//add sps and pps to I-frame and change the Nalu seprator according to Mp4 format
// 			auto newBuffer = AppendSizeInNaluSeprator(naluType, inH264ImageFrame, frameSize);
// 			mux_sample.buffer = newBuffer;
// 			mux_sample.len = frameSize;
// 		}
// 	}

// 	if (naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
// 	{
// 		//change the Nalu seprator according to Mp4 format
// 		auto newBuffer = AppendSizeInNaluSeprator(naluType, inH264ImageFrame, frameSize);
// 		mux_sample.buffer = newBuffer;
// 		mux_sample.len = frameSize;
// 	}

// 	if (syncFlag)
// 	{
// 		mp4_mux_sync(mux);
// 		LOG_ERROR << " media synced!";
// 		syncFlag = false;
// 	}

// 	if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
// 	{
// 		isKeyFrame = true;
// 	}
// 	else
// 	{
// 		isKeyFrame = false;
// 	}

// 	addMetadataInVideoHeader(inH264ImageFrame);

// 	mux_sample.sync = isKeyFrame ? 1 : 0;
// 	int64_t diffInMsecs = 0;

// 	if (!lastFrameTS)
// 	{
// 		diffInMsecs = 0;
// 		mux_sample.dts = 0;
// 	}
// 	else
// 	{
// 		diffInMsecs = inH264ImageFrame->timestamp - lastFrameTS;
// 		int64_t halfDurationInMsecs = static_cast<int64_t>(1000 / (2 * mProps->fps));
// 		if (!diffInMsecs)
// 		{
// 			inH264ImageFrame->timestamp += halfDurationInMsecs;
// 		}

// 		else if (diffInMsecs < 0)
// 		{
// 			inH264ImageFrame->timestamp = lastFrameTS + halfDurationInMsecs;
// 		}
// 		diffInMsecs = inH264ImageFrame->timestamp - lastFrameTS;
// 	}
// 	lastFrameTS = inH264ImageFrame->timestamp;
// 	mux_sample.dts = mux_sample.dts + static_cast<int64_t>((params.timescale / 1000) * diffInMsecs);

// 	mp4_mux_track_add_sample(mux, videotrack, &mux_sample);

// 	if (metatrack != -1 && mMetadataEnabled && inMp4MetaFrame.get())
// 	{
// 		mux_sample.buffer = static_cast<uint8_t*>(inMp4MetaFrame->data());
// 		mux_sample.len = inMp4MetaFrame->size();
// 		mp4_mux_track_add_sample(mux, metatrack, &mux_sample);
// 	}
// 	m_frameNumber++;
// 	return true;
// }

// Mp4WriterSink::Mp4WriterSink(Mp4WriterSinkProps _props)
// 	: Module(SINK, "Mp4WriterSink", _props), mProp(_props),
// 	m_hashFrameStartQueue(), m_hashFrameEndQueue(), m_hashFrameQueue(),
// 	m_shouldStopFileWrite(false)
// {
// 	LOG_ERROR << "FPS was set to " << mProp.fps;
// 	// TODO: Have separate cache limit for first and last buffer
// 	m_hashFrameStartQueue.clear();
// 	m_hashFrameEndQueue.clear();
// 	m_hashFrameQueue.clear();
// }

// Mp4WriterSink::~Mp4WriterSink() {}

// bool Mp4WriterSink::init()
// {
// 	if (!Module::init())
// 	{
// 		return false;
// 	}
// 	auto inputPinIdMetadataMap = getInputMetadata();

// 	for (auto const& element : inputPinIdMetadataMap)
// 	{
// 		auto& metadata = element.second;
// 		auto mFrameType = metadata->getFrameType();
// 		if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE)
// 		{
// 			mDetail.reset(new DetailJpeg(mProp));
// 		}

// 		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
// 		{
// 			mDetail.reset(new DetailH264(mProp));
// 		}
// 	}
// 	return Module::init();
// }

// bool Mp4WriterSink::validateInputOutputPins()
// {
// 	if (getNumberOfInputsByType(FrameMetadata::H264_DATA) != 1 && getNumberOfInputsByType(FrameMetadata::ENCODED_IMAGE) != 1) 
// 	{
// 		LOG_ERROR << "<" << getId() << ">::validateInputOutputPins expected 1 pin of ENCODED_IMAGE. Actual<" << getNumberOfInputPins() << ">";
// 		return false;
// 	}
// 	return true;
// }

// bool Mp4WriterSink::validateInputPins()
// {
// 	if (getNumberOfInputPins() > 2)
// 	{
// 		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
// 		return false;
// 	}

// 	auto inputPinIdMetadataMap = getInputMetadata();
// 	for (auto const& element : inputPinIdMetadataMap)
// 	{
// 		auto& metadata = element.second;
// 		auto mFrameType = metadata->getFrameType();
// 		if (mFrameType != FrameMetadata::ENCODED_IMAGE && mFrameType != FrameMetadata::MP4_VIDEO_METADATA && mFrameType != FrameMetadata::H264_DATA)
// 		{
// 			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be ENCODED_IMAGE or MP4_VIDEO_METADATA. Actual<" << mFrameType << ">";
// 			return false;
// 		}

// 		FrameMetadata::MemType memType = metadata->getMemType();
// 		if (memType != FrameMetadata::MemType::HOST)
// 		{
// 			LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
// 			return false;
// 		}
// 	}
// 	return true;
// }
// bool Mp4WriterSink::setMetadata(framemetadata_sp& inputMetadata)
// {
// 	// #Dec_24_Review - this function seems to do nothing
// 	mDetail->setImageMetadata(inputMetadata);
// 	return true;
// }

// bool Mp4WriterSink::processSOS(frame_sp& frame)
// {
// 	auto inputMetadata = frame->getMetadata();
// 	auto mFrameType = inputMetadata->getFrameType();
// 	if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE || mFrameType == FrameMetadata::FrameType::H264_DATA)
// 	{
// 		setMetadata(inputMetadata);
// 	}

// 	if (mFrameType == FrameMetadata::FrameType::MP4_VIDEO_METADATA)
// 	{
// 		auto mp4VideoMetadata = FrameMetadataFactory::downcast<Mp4VideoMetadata>(inputMetadata);
// 		std::string formatVersion = mp4VideoMetadata->getVersion();
// 		LOG_ERROR << "processSOS Called!\n";
// 		if (formatVersion.empty())
// 		{
// 			LOG_ERROR << "Serialization Format Information missing from the metadata. Metadata writing will be disabled";
// 			return true;
// 		}
// 		mDetail->enableMetadata(formatVersion);
// 	}
// 	return true;
// }

// bool Mp4WriterSink::shouldTriggerSOS()
// {
// 	return mDetail->shouldTriggerSOS();
// }

// bool Mp4WriterSink::term()
// {
// 	mDetail->attemptFileClose();
// 	return true;
// }

// bool Mp4WriterSink::process(frame_container& frames)
// {
// 	try
// 	{
// 		m_currFrame++;
// 		if (m_shouldStopFileWrite)
// 		{
// 			LOG_ERROR << "Frame can not come through";
// 			return true;
// 		}
// 		else
// 		{
// 			auto frame = getFrameByType(frames, FrameMetadata::H264_DATA);
// 			auto type =  H264Utils::getNALUType((char*)(frame->data()));
// 			if (!frame) 
// 			{
// 				LOG_ERROR << "Frame not found";
// 			}
// 			else if (((type == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE || type == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_SEQ_PARAM) && m_lastFrameStored < 0) || m_lastFrameStored >= 0)
// 			{
// 				m_lastFrameStored++;
// 				cacheFrames(5, 5, frame);
// 			}
// 		}
// 		if (!mDetail->write(frames))
// 		{
// 			LOG_FATAL << "Error occured while writing mp4 file<>";
// 			return true;
// 		}
// 	}
// 	catch (const std::exception& e)
// 	{
// 		LOG_ERROR << e.what();
// 		// close any open file
// 		mDetail->attemptFileClose();
// 	}
// 	return true;
// }

// vector<uint8_t> Mp4WriterSink::getFrameBytes(frame_sp frame)
// {
// 	uint8_t* rawBuffer = static_cast<uint8_t*>(frame->data());
// 	char* signedBuffer = static_cast<char*>(frame->data());
// 	vector<uint8_t> frameCopy;
// 	size_t frameLength = frame->size();
// 	size_t iFrameOffset = 0;
// 	int32_t naluSeparatorSize = 0;
// 	bool isKeyFrame = H264Utils::getNALUnitOffsetAndSizeBasedOnGivenType(signedBuffer, frameLength, 
// 		iFrameOffset, naluSeparatorSize, H264Utils::H264_NAL_TYPE_IDR_SLICE, true);
// 	if(!isKeyFrame || (isKeyFrame && !iFrameOffset))
// 	{
// 		frameCopy.insert(frameCopy.end(), rawBuffer + 4, rawBuffer + frameLength);
// 	}
// 	else
// 	{
// 		frameCopy.insert(frameCopy.end(), rawBuffer + iFrameOffset, rawBuffer + frameLength);
// 	}
// 	return frameCopy;
// }

// void Mp4WriterSink::cacheFrames(uint32_t firstLimit, uint32_t endLimit, frame_sp frame)
// {
// 	// uint8_t* frameCopy;
// 	vector<uint8_t> frameCopy = getFrameBytes(frame);
// 	if (m_hashFrameStartQueue.size() < firstLimit) 
// 	{
// 		m_hashFrameStartQueue.push_back(frameCopy);
// 		mDetail->queuedFrames.push_back(m_lastFrameStored);
// 	}
// 	else 
// 	{
// 		if (m_hashFrameEndQueue.size() >= endLimit) 
// 		{
// 			m_hashFrameEndQueue.erase(m_hashFrameEndQueue.begin());
// 		}
// 		m_hashFrameEndQueue.push_back(frameCopy);
// 		mDetail->queuedFrames.push_back(m_lastFrameStored);
// 	}
// 	// hashing(frameCopy.data(), frameCopy.size());
// }

// bool Mp4WriterSink::processEOS(string& pinId)
// {
// 	// #Dec_24_Review - generally you do opposite of what you do on SOS, so that after EOS, SOS is triggered
// 	// in current state after EOS, SOS is not triggered - is it by design ? 
// 	// Example EOS can be triggered if there is some resolution change in upstream module
// 	// so you want to do mDetail->mInputMetadata.reset() - so that SOS gets triggered
// 	return true;
// }

// Mp4WriterSinkProps Mp4WriterSink::getProps()
// {
// 	auto tempProps = Mp4WriterSinkProps(mDetail->mProps->chunkTime, mDetail->mProps->syncTimeInSecs, mDetail->mProps->fps, mDetail->mProps->baseFolder);
// 	fillProps(tempProps);
// 	return tempProps;
// }

// bool Mp4WriterSink::handlePropsChange(frame_sp& frame)
// {
// 	Mp4WriterSinkProps props;
// 	bool ret = Module::handlePropsChange(frame, props);
// 	mDetail->setProps(props);
// 	m_shouldStopFileWrite = false;
// 	return ret;
// }

// void Mp4WriterSink::setProps(Mp4WriterSinkProps& props)
// {
// 	Module::addPropsToQueue(props);
// }

// bool Mp4WriterSink::handleCommand(Command::CommandType type, frame_sp& frame)
// {
//     if (type == Command::CommandType::Mp4FileClose)
//     {
//         Mp4WriterSinkCloseFile cmd;
//         getCommand(cmd, frame);
// 		if (m_customMetadata.empty())
// 		{
// 			m_customMetadata = "dummy data";
// 		}
// 		mDetail->addMetadataInVideoHeaderAtEnd("\251too", m_customMetadata.c_str());
// 		mDetail->attemptFileClose();
// 		m_hashFrameQueue.clear();
// 		m_shouldStopFileWrite = false;

// 		m_lastFrameStored = -1;

// 		for (uint64_t val : mDetail->queuedFrames)
// 		{
// 			std::cout << val << ", ";
// 		}
// 		std::cout << std::endl;
// 		m_currFrame = -1;
// 		mDetail->queuedFrames.clear();
// 		LOG_ERROR << mProp.baseFolder;

//         return true;
//     }
// 	else  if (type == Command::CommandType::StopMp4Write)
//     {
//         Mp4StopWrite cmd;
//         getCommand(cmd, frame);
// 		m_shouldStopFileWrite = true;
// 		m_hashFrameQueue.clear();
// 		m_hashFrameQueue.insert(m_hashFrameQueue.end(), m_hashFrameStartQueue.begin(), m_hashFrameStartQueue.end());
// 		m_hashFrameStartQueue.clear();
// 		m_hashFrameQueue.insert(m_hashFrameQueue.end(), m_hashFrameEndQueue.begin(), m_hashFrameEndQueue.end());
// 		m_hashFrameEndQueue.clear();
// 		hashing();
// 		if (m_callbackFunction)
// 		{
// 			m_callbackFunction();
// 		}
//         return true;
//     }
//     return Module::handleCommand(type, frame);
// }

// bool Mp4WriterSink::closeFile()
// {
//     Mp4WriterSinkCloseFile cmd;
//     return queueCommand(cmd);
// }

// bool Mp4WriterSink::retortCallback()
// {
// 	Mp4StopWrite cmd;
//     return queueCommand(cmd);
// }

// void Mp4WriterSink::setCustomMetadata(std::string data)
// {
// 	m_customMetadata = data;
// }

// std::vector<std::vector<uint8_t>> Mp4WriterSink::getQueuedFrames()
// {
// 	return m_hashFrameQueue;
// 	// return mDetail->queuedFrames;
// }

// void Mp4WriterSink::hashing(uint8_t* frame, size_t frameSize)
// {
// 	unsigned char hash[SHA512_DIGEST_LENGTH] = { 0 }; // Buffer for SHA-512 hash
// 	char computedHash[SHA512_DIGEST_LENGTH * 2 + 1] = { 0 };
// 	// SHA512(vec[i], sizeof(vec[i]), hash); 
// 	SHA512(frame, frameSize, hash); 
// 	for (int j = 0; j < SHA512_DIGEST_LENGTH; j++)
// 	{
// 		sprintf(computedHash + (j * 2), "%02x", hash[j]);
// 	}
// 	printf("\tHex of frame at generation %lu -> %s with size %lu \n", m_lastFrameStored, computedHash, frameSize);
// }


// void Mp4WriterSink::hashing()
// {
// 	std::vector<std::vector<uint8_t>> vec = m_hashFrameQueue;
// 	for (size_t i = 0; i < vec.size(); ++i) 
// 	{
// 		unsigned char hash[SHA512_DIGEST_LENGTH]; // Buffer for SHA-512 hash
// 		char computedHash[SHA512_DIGEST_LENGTH * 2 + 1];
// 		SHA512(vec[i].data(), vec[i].size(), hash);
// 		for (int j = 0; j < SHA512_DIGEST_LENGTH; j++)
// 		{
// 			sprintf(computedHash + (j * 2), "%02x", hash[j]);
// 		}
// 		printf("\tHex of frame at close file %lu -> %s (%lu)  \n", i, computedHash, vec[i].size());
// 	}
// }