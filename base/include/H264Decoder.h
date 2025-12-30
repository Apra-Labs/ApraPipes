#pragma once

#include "Module.h"
#include <vector>
#include <array>
#include "declarative/Metadata.h"

class H264DecoderProps : public ModuleProps
{
public:
	H264DecoderProps(uint _lowerWaterMark = 300, uint _upperWaterMark = 350)
	{
		lowerWaterMark = _lowerWaterMark;
		upperWaterMark = _upperWaterMark;
	}
	uint lowerWaterMark;
	uint upperWaterMark;
};

class H264Decoder : public Module
{
public:
	// ============================================================
	// Declarative Pipeline Metadata
	// ============================================================
	struct Metadata {
		static constexpr std::string_view name = "H264Decoder";
		static constexpr apra::ModuleCategory category = apra::ModuleCategory::Transform;
		static constexpr std::string_view version = "1.0.0";
		static constexpr std::string_view description =
			"Decodes H.264/AVC encoded video frames to raw image frames. "
			"Supports bidirectional playback with GOP buffering.";

		static constexpr std::array<std::string_view, 4> tags = {
			"decoder", "h264", "video", "transform"
		};

		static constexpr std::array<apra::PinDef, 1> inputs = {
			apra::PinDef::create("input", "H264Frame", true, "H.264 encoded video frames")
		};

		static constexpr std::array<apra::PinDef, 1> outputs = {
			apra::PinDef::create("output", "RawImagePlanar", true, "Decoded raw image frames")
		};

		static constexpr std::array<apra::PropDef, 2> properties = {
			apra::PropDef::Integer("lowerWaterMark", 300, 0, 10000,
				"Lower watermark for frame buffering"),
			apra::PropDef::Integer("upperWaterMark", 350, 0, 10000,
				"Upper watermark for frame buffering")
		};
	};

	H264Decoder(H264DecoderProps _props);
	virtual ~H264Decoder();
	bool init();
	bool term();
	bool processEOS(string& pinId);

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();
	void flushQue();
	bool handleCommand(Command::CommandType type, frame_sp& frame);

private:
	void bufferDecodedFrames(frame_sp& frame);
	void bufferBackwardEncodedFrames(frame_sp& frame, short naluType);
	void bufferAndDecodeForwardEncodedFrames(frame_sp& frame, short naluType);

	class Detail;
	boost::shared_ptr<Detail> mDetail;
	bool mShouldTriggerSOS;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	H264DecoderProps mProps;


	/* Used to buffer multiple complete GOPs 
	note that we decode frames from this queue in reverse play*/
	std::deque<std::deque<frame_sp>> backwardGopBuffer;
	/* buffers the incomplete GOP */
	std::deque<frame_sp> latestBackwardGop;
	/* It buffers only one latest GOP 
	used in cases where partial GOP maybe in cache and rest of the GOP needs to be decoded
	note that since there is no buffering in forward play, we directly decode frames from module queue*/
	std::deque<frame_sp> latestForwardGop;
	std::map<uint64, frame_sp> decodedFramesCache;
	void sendDecodedFrame();
	bool mDirection;
	bool dirChangedToFwd = false;
	bool dirChangedToBwd = false;
	bool foundIFrameOfReverseGop = false;
	bool decodePreviousFramesOfTheForwardGop = false;
	bool prevFrameInCache = false;
	void decodeFrameFromBwdGOP();
	std::deque<uint64_t> incomingFramesTSQ;
	void clearIncompleteBwdGopTsFromIncomingTSQ(std::deque<frame_sp>& latestGop);
	void saveSpsPps(frame_sp frame);
	void* prependSpsPps(frame_sp& iFrame, size_t& spsPpsFrameSize);
	void dropFarthestFromCurrentTs(uint64_t ts);
	frame_sp mHeaderFrame;
	boost::asio::const_buffer spsBuffer;
	boost::asio::const_buffer ppsBuffer;
	std::mutex m;
	int framesToSkip = 0;
	int iFramesToSkip = 0;
	int currentFps = 24;
	int previousFps = 24;
	float playbackSpeed = 1;
	int gop;
	uint64_t lastFrameSent;
	bool resumeFwdPlayback = true;
	bool resumeBwdPlayback = true;
	bool resumePlayback = true;
	int incomingFramesTSQSize = 0;
};
