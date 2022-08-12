#pragma once
#include <boost/asio/buffer.hpp>
#include "FrameContainerQueue.h"
#include "Module.h"
using boost::asio::const_buffer;
using boost::asio::mutable_buffer;
class Frame;
class H264FrameUtils : public FrameContainerQueueAdapter
{
	enum STATE {
		INITIAL,
		SPS_RCVD,
		PPS_RCVD,
		WAITING_FOR_IFRAME, // drops
		NORMAL
	};
	STATE myState;
	const_buffer sps, pps, sps_pps;
public:
	tuple<short, const_buffer, const_buffer, const_buffer> parseNalu(mutable_buffer& input);
};