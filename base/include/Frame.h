#pragma once
#include <boost/asio.hpp>
#include "CommonDefs.h"

using namespace std;

class FrameFactory;
class ApraData;

class Frame :public boost::asio::mutable_buffer {
public:
	Frame(void *buff, size_t size, boost::shared_ptr<FrameFactory> mother);
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
	framemetadata_sp getMetadata() { return mMetadata; }
	// Make it private
	// If someone wants to use it, make that class a friend class
	void setMetadata(framemetadata_sp& _metadata) { mMetadata = _metadata; }
	virtual void* data() const BOOST_ASIO_NOEXCEPT;
	virtual std::size_t size() const BOOST_ASIO_NOEXCEPT;
protected:
	Frame();
	framemetadata_sp mMetadata;
private:
	void setDefaultValues();
	void resetMemory();
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
	enum EoSFrameType
	{
		GENERAL = 0,
		MP4_PLYB_EOS,
		MP4_SEEK_EOS,
	};
	EoSFrame();
	virtual ~EoSFrame() {}
	virtual bool isEOS();
	EoSFrameType getEoSFrameType();
	EoSFrame(EoSFrameType eosType, uint64_t mp4TS);
private:
	EoSFrameType type;
	uint64_t mp4TS;
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