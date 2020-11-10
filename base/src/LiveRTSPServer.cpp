#include "LiveRTSPServer.h"

#include "FrameMetadata.h"
#include "Logger.h"
#include "Frame.h"
#include "ThreadSafeQue.h"

#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include "GroupsockHelper.hh"
#include "JPEGVideoSource.hh"
#include "JPEGFrameParser.h"
#include <GroupsockHelper.hh> // for "gettimeofday()"

#include <boost/thread/condition.hpp>
#include <thread>

#define LIVE_MEDIA_RELEASE(media) \
    if (media)                    \
    {                             \
        Medium::close(media);     \
        media = nullptr;          \
    }

class MJPEGVideoSource : public JPEGVideoSource
{
public:
	static MJPEGVideoSource *createNew(UsageEnvironment &env, std::function<bool(unsigned char *&buffer, size_t &length)> getNextFrame)
	{
		return new MJPEGVideoSource(env, getNextFrame);
	}

	virtual void doGetNextFrame()
	{
		if (!mGetNextFrame(mBuffer, mLength))
		{
			return;
		}

		bool parseRes = mFrameParser.Parse(mBuffer, static_cast<unsigned int>(mLength));
		if (!parseRes)
		{
			return;
		}

		auto buffer = const_cast<unsigned char*>(mFrameParser.GetScandata(fFrameSize));
		memcpy(fTo, buffer, fFrameSize);

		gettimeofday(&fPresentationTime, NULL);

		fNumTruncatedBytes = 0;
		fDurationInMicroseconds = 0U;
		afterGetting(this);
	}

	virtual u_int8_t type()
	{
		return mFrameParser.GetType();
	}

	virtual u_int8_t qFactor()
	{
		return mFrameParser.GetQFactor();
	}

	virtual u_int8_t width()
	{
		return static_cast<u_int8_t>(mFrameParser.GetWidth());
	}

	virtual u_int8_t height()
	{
		return static_cast<u_int8_t>(mFrameParser.GetHeight());
	}

protected:
	MJPEGVideoSource(UsageEnvironment &env, std::function<bool(unsigned char *&buffer, size_t &length)> getNextFrame) : JPEGVideoSource(env), mLength(0), mBuffer(nullptr)
	{
		mGetNextFrame = getNextFrame;
	}

	virtual ~MJPEGVideoSource()
	{
	}

	MJPEGVideoSource(const MJPEGVideoSource &other) = delete;
	MJPEGVideoSource &operator=(const MJPEGVideoSource &other) = delete;

protected:
	JpegFrameParser mFrameParser;
	std::function<bool(unsigned char *&buffer, size_t &length)> mGetNextFrame;
	unsigned char *mBuffer;
	size_t mLength;
};

class Session
{
public:
	Session() : mRTPGroupSock(nullptr),
		mRTCPGroupSock(nullptr),
		mRTCPInst(nullptr),
		mServerSession(nullptr),
		mVideoSource(nullptr)
	{
	}

	~Session()
	{
		ReleaseResources();
	}

	bool InitSession(RTSPServer *pServerToAddTo,
		UsageEnvironment *pUsageEnv,
		const char *pszStreamName,
		const char *pszDescription,
		const unsigned short rtpPort,
		void* buffer, size_t size,
		std::function<bool(unsigned char *&buffer, size_t &length)> getNextFrame)
	{
		bool initRes = false;

		if (nullptr != pServerToAddTo)
		{
			// Create 'groupsocks' for RTP and RTCP:
			struct in_addr destinationAddress;
			destinationAddress.s_addr = chooseRandomIPv4SSMAddress(*pUsageEnv);

			const unsigned short rtpPortNum = rtpPort; // 18888;
			const unsigned short rtcpPortNum = rtpPortNum + 1;
			const unsigned char ttl = 255;

			const Port rtpPort(rtpPortNum);
			const Port rtcpPort(rtcpPortNum);

			mRTPGroupSock = new Groupsock(*pUsageEnv, destinationAddress, rtpPort, ttl);
			mRTPGroupSock->multicastSendOnly(); // we're a SSM source

			mRTCPGroupSock = new Groupsock(*pUsageEnv, destinationAddress, rtcpPort, ttl);
			mRTCPGroupSock->multicastSendOnly(); // we're a SSM source

			JpegFrameParser parser;
			parser.Parse(static_cast<uint8_t*>(buffer), static_cast<uint32_t>(size));
			unsigned int maxBufferSize = (parser.GetWidth()*parser.GetHeight()*3) << 6; // width and height were divided by 8
			mVideoSink = JPEGVideoRTPSink::createNew(*pUsageEnv, mRTPGroupSock, maxBufferSize);

			// Create (and start) a 'RTCP instance' for this RTP sink:
			unsigned estimatedSessionBandwidth = OutPacketBuffer::maxSize; // in kbps; for RTCP b/w share			
			estimatedSessionBandwidth = maxBufferSize;

			const unsigned maxCNAMElen = 100;
			unsigned char CNAME[maxCNAMElen + 1];
			gethostname((char *)CNAME, maxCNAMElen);
			CNAME[maxCNAMElen] = '\0'; // just in case
			mRTCPInst = RTCPInstance::createNew(*pUsageEnv, mRTCPGroupSock,
				estimatedSessionBandwidth, CNAME,
				mVideoSink, NULL /* we're a server */,
				True /* we're a SSM source */);
			// Note: This starts RTCP running automatically

			mServerSession = ServerMediaSession::createNew(*pUsageEnv, pszStreamName, pszStreamName,
				pszDescription, True /*SSM*/);

			Boolean addSessionRes = mServerSession->addSubsession(PassiveServerMediaSubsession::createNew(*mVideoSink, mRTCPInst));

			if (addSessionRes)
			{
				pServerToAddTo->addServerMediaSession(mServerSession);

				char *pszUrl = pServerToAddTo->rtspURL(mServerSession);
				*pUsageEnv << "Play this stream using the URL \"" << pszUrl << "\"\n";

				mUrl = pszUrl;

				delete[] pszUrl;

				mVideoSource = MJPEGVideoSource::createNew(*pUsageEnv, getNextFrame);

				LOG_INFO << "LiveRTSPServer::Session::InitSession Success : Url " << mUrl;

				initRes = true;
			}
			else
			{
				*pUsageEnv << "InitSession failed : Port Used \"" << rtpPort << "\"\n";
				LOG_INFO << "LiveRTSPServer::Session::InitSession failed : Port Used " << rtpPort.num();
			}
		}

		return initRes;
	}

	void StartPlaying()
	{
		if ((nullptr != mVideoSink) && (nullptr != mVideoSource))
		{
			mVideoSink->startPlaying(*mVideoSource, NULL, NULL);
		}
	}

	void ReleaseResources()
	{
		//Will be deleted by RTSPServer::removeServerMediaSession
		//LIVE_MEDIA_RELEASE(mServerSession);
		LIVE_MEDIA_RELEASE(mRTCPInst);
		LIVE_MEDIA_RELEASE(mVideoSink);
		LIVE_MEDIA_RELEASE(mVideoSource);

		if (mRTPGroupSock)
		{
			delete mRTPGroupSock;
			mRTPGroupSock = nullptr;
		}
		if (mRTCPGroupSock)
		{
			delete mRTCPGroupSock;
			mRTCPGroupSock = nullptr;
		}
	}

	const std::string &GetSessionURL() const
	{
		return mUrl;
	}

	ServerMediaSession *GetMediaSession() const
	{
		return mServerSession;
	}

private:
	Groupsock *mRTPGroupSock;
	Groupsock *mRTCPGroupSock;

	RTPSink *mVideoSink;
	RTCPInstance *mRTCPInst;
	ServerMediaSession *mServerSession;
	FramedSource *mVideoSource;

	std::string mUrl;
};

class LiveRTSPServer::Detail
{
	using SessionCtr = std::map<std::string, Session *>;

public:
	Detail()
		: mUsageEnv(nullptr),
		mRTSPServer(nullptr),
		mSessionWatchVar(0)
	{
	}

	~Detail()
	{
		Shutdown();
	}

	Detail(const LiveRTSPServer &other) = delete;
	Detail &operator=(const Detail &other) = delete;

	bool Init(const unsigned short rtspPort)
	{
		bool initRes = false;

		if (nullptr == mRTSPServer)
		{
			TaskScheduler *scheduler = BasicTaskScheduler::createNew();
			mUsageEnv = BasicUsageEnvironment::createNew(*scheduler);

			mRTSPServer = RTSPServer::createNew(*mUsageEnv, rtspPort);
			if (nullptr == mRTSPServer)
			{
				*mUsageEnv << "Failed to create RTSP server: " << mUsageEnv->getResultMsg() << "\n";
				LOG_ERROR << "LiveRTSPServer::Init::Failed to create RTSP server : " << mUsageEnv->getResultMsg();
			}
			else
			{
				initRes = true;
			}
		}

		return initRes;
	}

	bool AddSession(const char *pszStreamName,
		const char *pszDescription,
		const unsigned short rtpPort, void* buffer, size_t size)
	{
		bool sessionAdded = false;

		if (nullptr != mRTSPServer)
		{
			auto itrSessions = mSessionCtr.find(pszStreamName);
			if (itrSessions == mSessionCtr.end())
			{
				Session *pNewSession = new Session();
				sessionAdded = pNewSession->InitSession(mRTSPServer, mUsageEnv,
					pszStreamName,
					pszDescription,
					rtpPort,
					buffer, size,
					[&](unsigned char *&buffer, size_t &length) { return getNextFrame(buffer, length); });

				if (sessionAdded)
				{
					mSessionCtr[pszStreamName] = pNewSession;
				}
				else if (pNewSession)
				{
					delete pNewSession;
				}
			}
			else
			{
				assert(false);
				(*mUsageEnv) << "Session with Stream name " << pszStreamName << " already added\n";
				LOG_ERROR << "LiveRTSPServer::AddSession::Session with Stream name " << pszStreamName << " already added";
			}
		}

		return sessionAdded;
	}

	std::string GetSessionUrl(const char *pszStreamName) const
	{
		std::string strSessionUrl;

		auto itrSession = mSessionCtr.find(pszStreamName);
		if (itrSession != mSessionCtr.end())
		{
			const Session *pSession = itrSession->second;
			strSessionUrl = pSession->GetSessionURL();
		}

		return strSessionUrl;
	}

	void StartPlaying()
	{
		if (nullptr != mUsageEnv)
		{
			auto itrSession = mSessionCtr.begin();
			auto itrEndSession = mSessionCtr.end();

			for (; itrSession != itrEndSession; itrSession++)
			{
				Session *pSession = itrSession->second;
				pSession->StartPlaying();
			}

			mSessionWatchVar = 0;

			mThread = std::thread(&Detail::Run, this);
		}
	}

	void push(frame_sp &frame)
	{
		mQue.push(frame);
	}

private:
	bool getNextFrame(unsigned char *&buffer, size_t &length)
	{
		auto frame = mQue.try_pop_external();
		if (!frame.get())
		{
			return false;
		}

		buffer = static_cast<unsigned char *>(frame->data());
		length = frame->size();

		prevFrame = frame;

		return true;
	}

	void Run()
	{
		mUsageEnv->taskScheduler().doEventLoop(&mSessionWatchVar);
	}

    void Shutdown()
	{
		mSessionWatchVar = 1;
		mQue.setWake();
		mThread.join();
		ReleaseResources();
	}
	
    void ReleaseResources()
	{
		if (mRTSPServer)
		{
			auto itrSession = mSessionCtr.begin();
			auto itrEndSession = mSessionCtr.end();

			for (; itrSession != itrEndSession; itrSession++)
			{
				Session *pSession = itrSession->second;
				ServerMediaSession *pMediaSession = pSession->GetMediaSession();

				mRTSPServer->closeAllClientSessionsForServerMediaSession(pMediaSession);
				mRTSPServer->removeServerMediaSession(pMediaSession);

				if (pSession)
				{
					delete pSession;
				}
			}

			mSessionCtr.clear();

			LIVE_MEDIA_RELEASE(mRTSPServer);
		}

		if (mUsageEnv)
		{
			mUsageEnv->reclaim();
			mUsageEnv = nullptr;
		}
	}

	UsageEnvironment *mUsageEnv;
	RTSPServer *mRTSPServer; //Use ServerMediaSessionIterator to delete all MediaSessions

	char mSessionWatchVar;
	SessionCtr mSessionCtr;

	std::thread mThread;
	threadsafe_que<frame_sp> mQue;
	frame_sp prevFrame;
};

LiveRTSPServer::LiveRTSPServer(LiveRTSPServerProps _props)
	: Module(SINK, "LiveRTSPServer", _props), mStarted(false)
{
	mProps = _props;
}

LiveRTSPServer::~LiveRTSPServer() {}

bool LiveRTSPServer::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::ENCODED_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	if (metadata->getMemType() != FrameMetadata::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input MemType is expected to be HOST. Actual<" << metadata->getMemType() << ">";
		return false;
	}

	return true;
}

bool LiveRTSPServer::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool LiveRTSPServer::term()
{
    mDetail.reset();
	auto moduleRet = Module::term();

	return moduleRet;
}

bool LiveRTSPServer::process(frame_container &frames)
{
	auto frame = frames.begin()->second;
	mDetail->push(frame);

	return true;
}

bool LiveRTSPServer::processSOS(frame_sp &frame)
{
    mDetail.reset(new Detail());
    mDetail->Init(mProps.rtspPort);
	// first frame will be pushed 2 times - add a condition based on frame index 
	mDetail->push(frame);
    mDetail->AddSession(mProps.streamName.c_str(), mProps.description.c_str(), mProps.rtpPort, frame->data(), frame->size());
    mDetail->StartPlaying();

	mStarted = true;

	return true;
}

bool LiveRTSPServer::shouldTriggerSOS()
{	
	return !mStarted;
}