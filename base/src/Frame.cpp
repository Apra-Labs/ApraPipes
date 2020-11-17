#include "stdafx.h"
#include "Frame.h"
#include "stdlib.h"
#include "ApraData.h"
#include "FrameMetadata.h"

Buffer::Buffer(void *buff, size_t size, boost::shared_ptr<FrameFactory> mother) : mutable_buffer(buff, size), myOrig(buff), myMother(mother)
{

}
Buffer::~Buffer() {
	myMother.reset();
}

void Buffer::resetMemory()
{
	myOrig = NULL;
}

void* Buffer::data() const BOOST_ASIO_NOEXCEPT
{
	return boost::asio::mutable_buffer::data();
}

std::size_t Buffer::size() const BOOST_ASIO_NOEXCEPT
{
	return boost::asio::mutable_buffer::size();
}

Frame::Frame():mutable_buffer(0, 0),myOrig(0)
{
	setDefaultValues();
}
Frame::Frame(void *buff, size_t size, boost::shared_ptr<FrameFactory> mother):mutable_buffer(buff,size), myOrig(buff), myMother(mother)
{
	setDefaultValues();
}
Frame::Frame(void *buff, size_t size, framemetadata_sp& metadata):mutable_buffer(buff,size), myOrig(buff)
{
	setDefaultValues();
	mMetadata = metadata;
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

EoPFrame::EoPFrame() :Frame() {}
bool EoPFrame::isEoP() { return true; }

EoSFrame::EoSFrame():Frame(){}
bool EoSFrame::isEOS() { return true; }

EmptyFrame::EmptyFrame() : Frame() {}
bool EmptyFrame::isEmpty() { return true; }

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