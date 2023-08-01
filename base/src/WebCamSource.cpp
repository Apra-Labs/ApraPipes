#include "WebCamSource.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include <opencv2/opencv.hpp> //#Sai review use only relevant header

class WebCamSource::Detail
{
public:
    Detail(WebCamSourceProps _props) : cameraInstance(_props.cameraId)
    {
        mProps = _props;
    }

    ~Detail() {}

    bool init()
    {
        cameraInstance.set(cv::CAP_PROP_FPS, mProps.fps);
        cameraInstance.set(cv::CAP_PROP_FRAME_WIDTH, mProps.width);
        cameraInstance.set(cv::CAP_PROP_FRAME_HEIGHT, mProps.height);
        return cameraInstance.isOpened();
    }

    bool produce(frame_sp &outFrame)

    {
        frame.data = static_cast<uchar *>(outFrame->data());
        cameraInstance >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        return true;
    }

    WebCamSourceProps getProps()
    {
        return mProps;
    }

    void setProps(WebCamSourceProps _props)
    {
        mProps = _props;
        frame = Utils::getMatHeader(mProps.height, mProps.width, CV_8UC3);
    }
    std::string mOutputPinId;

private:
    WebCamSourceProps mProps;
    cv::VideoCapture cameraInstance;
    cv::Mat frame;
};

WebCamSource::WebCamSource(WebCamSourceProps _props)
    : Module(SOURCE, "WebCamSource", _props)
{
    mDetail.reset(new Detail(_props));
    auto outputMetadata = framemetadata_sp(new RawImageMetadata(_props.width, _props.height, ImageMetadata::ImageType::RGB, CV_8UC3, 3 * _props.width, CV_8U, FrameMetadata::MemType::HOST));
    mDetail->mOutputPinId = addOutputPin(outputMetadata);
}

WebCamSource::~WebCamSource() {}

bool WebCamSource::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        return false;
    }

    return true;
}

bool WebCamSource::init()
{
    if (!Module::init())
    {
        return false;
    }

    return mDetail->init();
}

bool WebCamSource::term()
{
    return Module::term();
}

bool WebCamSource::produce()
{
    auto mProps = mDetail->getProps();
    auto frame = makeFrame(mProps.width * mProps.height * 3);
    mDetail->produce(frame);
    frame_container frames;
    frames.insert(make_pair(mDetail->mOutputPinId, frame));
    send(frames);
    return true;
}

WebCamSourceProps WebCamSource::getProps()
{
    auto mProps = mDetail->getProps();
    fillProps(mProps);

    return mProps;
}

void WebCamSource::setProps(WebCamSourceProps &mProps)
{
    Module::addPropsToQueue(mProps);
}

bool WebCamSource::handlePropsChange(frame_sp &frame)
{
    WebCamSourceProps mProps;
    bool ret = Module::handlePropsChange(frame, mProps);
    mDetail->setProps(mProps);

    sendEOS();

    return ret;
}
