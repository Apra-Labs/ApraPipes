#include <stdafx.h>
#include<iostream>
#include<cstdint>
#include <boost/filesystem.hpp>
#include<map>
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
        namespace bf = boost::filesystem;
        bf::path p = bf::current_path();
        bf::path n = mProps.pathToWatch;
        for (bf::recursive_directory_iterator it(n); it != bf::recursive_directory_iterator(); ++it)
        {
            if (!bf::is_directory(*it))
            {
                diskSize += bf::file_size(*it);
                fileMap.insert(pair<bf::path, uintmax_t>(it->path(), bf::file_size(*it)));
            }
        }
        if (diskSize > mProps.upperWaterMark)
        {
            for (auto& file : boost::filesystem::directory_iterator(n))
            {
                auto filesize = boost::filesystem::file_size(file);
                //boost::filesystem::remove(file);
                diskSize = diskSize - filesize;
                if (diskSize <= mProps.lowerWaterMark)
                {
                    break;
                }
            }
        }
    };
    DiskspaceManagerProps mProps;
    uintmax_t diskSize = 0;
    map<boost::filesystem::path, uintmax_t> fileMap;
};


DiskspaceManager::DiskspaceManager(DiskspaceManagerProps _props)
    :Module(TRANSFORM, "DiskspaceManager", _props)
{
    mDetail.reset(new Detail(_props));
}

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

bool DiskspaceManager::process(frame_container& frames)
{
    mDetail->checkDirectory();
    return true;
}
