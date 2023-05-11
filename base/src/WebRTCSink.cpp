#include <boost/filesystem.hpp>
#include "WebRTCSink.h"
#include "Command.h"
#include "Frame.h"

class WebRTCSink::Detail
{
public:
    Detail(WebRTCSinkProps& _props) : mProps(_props)
    { 
    }

    ~Detail()
    { 
    }

    void setProps(WebRTCSinkProps _props)
    {
        mProps = _props;
    }

   
public:
    WebRTCSinkProps mProps;
};



WebRTCSink::WebRTCSink(WebRTCSinkProps _props)
    :Module(SINK, "WebRTCSink", _props)
{
    mDetail.reset(new Detail(_props));
}

WebRTCSink::~WebRTCSink() {}

bool WebRTCSink::validateInputPins()
{   
    return true;
}

bool WebRTCSink::validateOutputPins()
{
    return true;
}

bool WebRTCSink::validateInputOutputPins()
{
    return true;
}


bool WebRTCSink::init() 
{

    if (!Module::init())
    {
        return false;
    }
    adapter->init(0,{});
    adapter->processSOS();
    return true;
}

bool WebRTCSink::term()
{
    return Module::term();
}

WebRTCSinkProps WebRTCSink::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

void WebRTCSink::setProps(WebRTCSinkProps& props)
{
    Module::addPropsToQueue(props);
}

bool WebRTCSink::process(frame_container& frames) 
{
    auto frame = frames.begin()->second;
    auto frameData = frame->data();
    adapter->process(frameData);
    return true;
}

void WebRTCSink::setMetadata(framemetadata_sp& metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
}
