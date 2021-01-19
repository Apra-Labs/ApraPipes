#pragma once
#include <boost/asio/buffer.hpp>
#include "FrameContainerQueue.h"
using boost::asio::const_buffer;
using boost::asio::mutable_buffer;
class Frame;
class H264FrameDemuxer : public FrameContainerQueueAdapter{
	enum STATE {
		INITIAL,
		SPS_RCVD,
		PPS_RCVD,
		WAITING_FOR_IFRAME, // drops
		NORMAL
	};
	const_buffer parseNALU(mutable_buffer& input, short &typeFound);
	STATE myState;
	const_buffer sps, pps, sps_pps;
protected:
	FrameContainerQueueAdapter::PushType should_push(frame_container item);
	void on_failed_push(frame_container item);
	void on_push_success(frame_container item);
	frame_container on_pop_success(frame_container item);

public:
	H264FrameDemuxer(): myState(INITIAL) {}
		short getState() { return myState; }
	const_buffer getSPS() { return sps; }
	const_buffer getPPS() { return pps; }
	const_buffer getSPS_PPS() { return sps_pps; } //as is supplied by the source
};