#include <stdafx.h>
#include <boost/filesystem.hpp>
#include "Module.h"
#include "DiskspaceManager.h"

class DiskspaceManager::Detail
{
public:
    Detail(DiskspaceManagerProps& _props) : mProps(_props)
    {
    }

    ~Detail()
    {
    }

    void setProps(DiskspaceManagerProps _props)
    {
        mProps = _props;
    }
    void checkDirectory()
    {
    }

public:
    void checkDirectory();
    DiskspaceManagerProps mProps;
};


DiskspaceManager::DiskspaceManager(DiskspaceManagerProps _props)
    :Module(TRANSFORM, "DiskspaceManager", _props)
{
    mDetail.reset(new Detail(_props));
}

DiskspaceManager::~DiskspaceManager() {}

bool DiskspaceManager::validateInputPins()
{
    return true;
}

bool DiskspaceManager::validateOutputPins()
{
    return true;
}

bool DiskspaceManager::validateInputOutputPins()
{
    return true;
}

void DiskspaceManager::addInputPin(framemetadata_sp& metadata, string& pinId)
{
    Module::addInputPin(metadata, pinId);
    Module::addOutputPin(metadata, pinId);
}

bool DiskspaceManager::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool DiskspaceManager::term()
{
    return Module::term();
}

DiskspaceManagerProps DiskspaceManager::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

void DiskspaceManager::setProps(DiskspaceManagerProps& props)
{
    Module::addPropsToQueue(props);
}

void DiskspaceManager::setProps(DiskspaceManagerProps& props)
{
    Module::addPropsToQueue(props);
}

bool DiskspaceManager::process(frame_container& frames)
{
    
}
