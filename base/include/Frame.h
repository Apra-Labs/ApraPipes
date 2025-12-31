#pragma once
#include <memory>
#include <cstddef>
#include "CommonDefs.h"

using namespace std;

class FrameFactory;
class ApraData;

// Simple mutable buffer replacement
class mutable_buffer {
public:
	mutable_buffer(void* data, size_t size) : m_data(data), m_size(size) {}
	void* data() const { return m_data; }
	size_t size() const { return m_size; }
protected:
	mutable_buffer() : m_data(nullptr), m_size(0) {}
private:
	void* m_data;
	size_t m_size;
};

class Frame : public mutable_buffer {
public:
	Frame(void *buff, size_t size, std::shared_ptr<FrameFactory> mother);
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
	virtual bool isMp4ErrorFrame() { return false; }
	virtual bool isEmpty() { return false; }	
	virtual bool isPropsChange();
	virtual bool isPausePlay();
	virtual bool isCommand();	
	framemetadata_sp getMetadata() { return mMetadata; }
	// Make it private
	// If someone wants to use it, make that class a friend class
	void setMetadata(framemetadata_sp& _metadata) { mMetadata = _metadata; }
	virtual void* data() const;
	virtual std::size_t size() const;
protected:
	Frame();
	framemetadata_sp mMetadata;
private:
	void setDefaultValues();
	void resetMemory();
	void *myOrig;
	friend class FrameFactory;
	std::shared_ptr<FrameFactory> myMother; //so that the mother does not get destroyed before children
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

	void* data() const;
	std::size_t size() const;

private:
	ApraData* mData;
};