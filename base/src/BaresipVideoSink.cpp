#include <boost/filesystem.hpp>
#include "BaresipVideoSink.h"
#include "Command.h"
#include "Frame.h"

class BaresipVideoSink::Detail
{
public:
    Detail(BaresipVideoSinkProps& _props) : mProps(_props)
    { 
    }

    ~Detail()
    { 
    }

    void setProps(BaresipVideoSinkProps _props)
    {
        mProps = _props;
    }

   
public:
    BaresipVideoSinkProps mProps;
};



BaresipVideoSink::BaresipVideoSink(BaresipVideoSinkProps _props)
    :Module(SINK, "BaresipVideoSink", _props)
{
    mDetail.reset(new Detail(_props));
}

BaresipVideoSink::~BaresipVideoSink() {}

bool BaresipVideoSink::validateInputPins()
{   
    return true;
}

bool BaresipVideoSink::validateOutputPins()
{
    return true;
}

bool BaresipVideoSink::validateInputOutputPins()
{
    return true;
}


bool BaresipVideoSink::init() 
{

    if (!Module::init())
    {
        return false;
    }
    adapter->init();
    return true;
}

bool BaresipVideoSink::term()
{
    return Module::term();
}

BaresipVideoSinkProps BaresipVideoSink::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

void BaresipVideoSink::setProps(BaresipVideoSinkProps& props)
{
    Module::addPropsToQueue(props);
}

bool BaresipVideoSink::process(frame_container& frames) 
{
    auto frame = frames.begin()->second;
    auto frameData = frame->data();
    adapter->process(frameData);
    return true;
}

void BaresipVideoSink::setMetadata(framemetadata_sp& metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
}
