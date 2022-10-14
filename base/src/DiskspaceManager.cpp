#include <stdafx.h>
#include<iostream>
#include<cstdint>
#include <boost/filesystem.hpp>
#include<map>
#include<vector>
#include<algorithm>
#include<chrono>
#include <boost/regex.hpp>
#include "Module.h"
#include "DiskspaceManager.h"

typedef std::pair<boost::filesystem::path, uint> elem;
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
  
    uint estimateDirectorySize(boost::filesystem::path _dir)
    {
        uint dirSize = 0;
        int count = 0;
        uint tempSize;
        boost::filesystem::path n =_dir;
        for (const auto& entry : boost::filesystem::recursive_directory_iterator(n))
        {
            if ((boost::filesystem::is_regular_file(entry)))
            {
                if (count % 15 == 0)
                {
                    tempSize = boost::filesystem::file_size(entry);
                }
                dirSize += tempSize;
                count++;
            }
        }
        return dirSize;
    }

    boost::filesystem::path getOldestDirectory(boost::filesystem::path _cam)
    {
        for (const auto& camFolder : boost::filesystem::directory_iterator(_cam))
        {
            for (const auto& folder : boost::filesystem::recursive_directory_iterator(camFolder))
            {
                if (boost::filesystem::is_regular_file(folder))
                {
                    boost::filesystem::path p = folder.path().parent_path();
                    return p;
                }
            }
        }
    };

    void deleteDirectory()
    {
        while (diskSize > mProps.lowerWaterMark)
        {
            for (const auto& camFolder : boost::filesystem::directory_iterator(mProps.pathToWatch))
            {
                uint tempSize = 0;
                boost::filesystem::path oldDir = getOldestDirectory(camFolder);
                tempSize = estimateDirectorySize(oldDir);
                diskSize = diskSize - tempSize;
                boost::filesystem::remove_all(oldDir);
                if (diskSize < mProps.lowerWaterMark)
                {
                    break;
                }
            }
        }
    }

    void operation()
    {
        diskSize = estimateDirectorySize(mProps.pathToWatch);
        if (diskSize > mProps.upperWaterMark)
        {
            deleteDirectory();
        }
    }

    DiskspaceManagerProps mProps;
    uintmax_t diskSize = 0;
    std::map<uintmax_t,uint>fileMap;
    int count = 0;

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
    mDetail->operation();
    //boost::filesystem::path par = mDetail->getOldestDirectory();
    int i = 0;
    return true;
}
