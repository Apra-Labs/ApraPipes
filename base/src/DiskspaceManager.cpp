#include <stdafx.h>
#include<iostream>
#include<cstdint>
#include <boost/filesystem.hpp>
#include<map>
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

    void checkDirectory()
    {
        namespace bf = boost::filesystem;
        bf::path n = mProps.pathToWatch;
        boost::regex delPattern = boost::regex(mProps.deletePattern);
        for (const auto& entry : boost::filesystem::recursive_directory_iterator(n))
        {
            if ((boost::filesystem::is_regular_file(entry) && !boost::filesystem::is_symlink(entry)) && (fileMap.find(bf::path(entry)) == fileMap.end()))
            {
                diskSize += boost::filesystem::file_size(entry);
                uint timeStamp = bf::last_write_time(bf::path(entry));
                fileMap[bf::path(entry)] = { timeStamp, bf::file_size(entry) };
                fileVector.push_back({ bf::path(entry), timeStamp });
            }
        }
        auto comparator = [](elem& a, elem& b) {return a.second < b.second; };
        sort(fileVector.begin(), fileVector.end(), comparator);
        iterateFlag = false;
       
        if (diskSize > mProps.upperWaterMark)
        {
            for (int i = 0; i < fileVector.size(); i++)
            {
                std::string pathInString = (fileVector[i].first).string();
                if (boost::regex_match(pathInString, delPattern))
                {
                    auto size = bf::file_size(fileVector[i].first);
                    diskSize = diskSize - size;
                    //bf::remove(fileVector[i].first);
                    if (diskSize <= mProps.lowerWaterMark)
                    {
                        break;
                    }
                }
            }
        }
    };
    DiskspaceManagerProps mProps;
    uintmax_t diskSize = 0;
    bool iterateFlag = true;
    std::map<boost::filesystem::path, std::vector<uint64_t>> fileMap;
    std::vector<elem>fileVector;

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
