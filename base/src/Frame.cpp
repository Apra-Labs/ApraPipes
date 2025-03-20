#include "stdafx.h"
#include "Frame.h"
#include "stdlib.h"
#include "ApraData.h"
#include "FrameMetadata.h"

Frame::Frame():mutable_buffer(0, 0),myOrig(0)
{
	setDefaultValues();
}
Frame::Frame(void *buff, size_t size, boost::shared_ptr<FrameFactory> mother):mutable_buffer(buff,size), myOrig(buff), myMother(mother)
{
	setDefaultValues();
}
Frame::~Frame() {
	myMother.reset();
}

void Frame::setDefaultValues()
{
	 mFrameType = 0;
	 mFStart = 0;
	 mFEnd = 0;
	 m_num = 0;
	 m_den = 0;
	 fIndex = 0;
	 pictureType = 255;
}

void* Frame::data() const BOOST_ASIO_NOEXCEPT
{	
	return boost::asio::mutable_buffer::data();
}

std::size_t Frame::size() const BOOST_ASIO_NOEXCEPT
{
	return boost::asio::mutable_buffer::size();
}

bool Frame::isPropsChange()
{
	return mMetadata->getFrameType() == FrameMetadata::FrameType::PROPS_CHANGE;
}

bool Frame::isPausePlay()
{
	return mMetadata->getFrameType() == FrameMetadata::FrameType::PAUSE_PLAY;
}

bool Frame::isCommand()
{
	return mMetadata->getFrameType() == FrameMetadata::FrameType::COMMAND;
}

void Frame::resetMemory()
{
	myOrig = NULL;
}

EoPFrame::EoPFrame() :Frame() {}
bool EoPFrame::isEoP() { return true; }

EoSFrame::EoSFrame():Frame(){}
bool EoSFrame::isEOS() { return true; }

EmptyFrame::EmptyFrame() : Frame() {}
bool EmptyFrame::isEmpty() { return true; }

EoSFrame::EoSFrameType EoSFrame::getEoSFrameType()
{
	return type;
}

EoSFrame::EoSFrame(EoSFrame::EoSFrameType _type, uint64_t _mp4TS) : Frame()
{
	type = _type;
	mp4TS = _mp4TS;
	timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

ExternalFrame::ExternalFrame(ApraData* data) : Frame()
{
	data->locked.fetch_add(1, memory_order_seq_cst);
	mData = data;
	fIndex = data->fIndex;	
}

ExternalFrame::~ExternalFrame()
{
	mData->locked.fetch_sub(1, memory_order_seq_cst);
}

void* ExternalFrame::data() const BOOST_ASIO_NOEXCEPT
{
	return mData->buffer;
}

std::size_t ExternalFrame::size() const BOOST_ASIO_NOEXCEPT
{
	return mData->size;
}