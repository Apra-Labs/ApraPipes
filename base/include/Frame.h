#pragma once
#include <boost/asio.hpp>
#include "CommonDefs.h"

using namespace std;

class FrameFactory;
class ApraData;

class Buffer : public boost::asio::mutable_buffer
{
public:
	Buffer(void *buff, size_t size, boost::shared_ptr<FrameFactory> mother);
	virtual ~Buffer();

	virtual void* data() const BOOST_ASIO_NOEXCEPT;
	virtual std::size_t size() const BOOST_ASIO_NOEXCEPT;
private:
	void *myOrig;
	friend class FrameFactory;
	boost::shared_ptr<FrameFactory> myMother; //so that the mother does not get destroyed before children
	void resetMemory(); // used during resize frame
};

class Frame :public boost::asio::mutable_buffer {
public:
	Frame(void *buff, size_t size, boost::shared_ptr<FrameFactory> mother);
	Frame(void *buff, size_t size, framemetadata_sp& metadata);
	virtual ~Frame();
	short mFrameType;
	uint64_t mFStart, mFEnd;
	uint64_t fIndex;
	uint64_t fIndex2; // added for fileReaderModule to propagate currentIndex
	uint64_t timestamp;
	int         pictureType; // used for H264 Encoder
	int m_num, m_den;
	virtual bool isEoP() { return false; }
	virtual bool isEOS() { return false; }
	virtual bool isEmpty() { return false; }	
	virtual bool isPropsChange();
	virtual bool isPausePlay();
	virtual bool isCommand();
	void setMetadata(framemetadata_sp& _metadata) { mMetadata = _metadata; }
	framemetadata_sp getMetadata() { return mMetadata; }
	virtual void* data() const BOOST_ASIO_NOEXCEPT;
	virtual std::size_t size() const BOOST_ASIO_NOEXCEPT;
protected:
	Frame();
	framemetadata_sp mMetadata;
private:
	void setDefaultValues();
	void *myOrig;
	friend class FrameFactory;
	boost::shared_ptr<FrameFactory> myMother; //so that the mother does not get destroyed before children	
};

class EoPFrame : public Frame
{
public:
	EoPFrame();
	virtual ~EoPFrame() {}
	virtual bool isEoP();
};

class EoSFrame : public Frame {
public:
	EoSFrame();
	virtual ~EoSFrame() {}
	virtual bool isEOS();
};

class EmptyFrame :public Frame {
public:
	EmptyFrame();
	virtual ~EmptyFrame() {}
	virtual bool isEmpty();
};

class ExternalFrame : public Frame
{
public:
	ExternalFrame(ApraData* data);
	virtual ~ExternalFrame();

	void* data() const BOOST_ASIO_NOEXCEPT;
	std::size_t size() const BOOST_ASIO_NOEXCEPT;

private:
	ApraData* mData;
};