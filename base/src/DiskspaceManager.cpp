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
        fileVector.reserve(50000);
        boost::filesystem::path n = mProps.pathToWatch;
        boost::regex delPattern = boost::regex(mProps.deletePattern);
      
        boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
        auto before = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
        BOOST_LOG_TRIVIAL(info) << before;

        for (const auto& entry : boost::filesystem::recursive_directory_iterator(n))
        {
            if ((boost::filesystem::is_regular_file(entry))) //&& !boost::filesystem::is_symlink(entry)) && (fileMap.find(boost::filesystem::path(entry)) == fileMap.end()))
            {
                diskSize += boost::filesystem::file_size(entry);
                uint timeStamp = boost::filesystem::last_write_time(boost::filesystem::path(entry));
                fileVector.push_back({ boost::filesystem::path(""), timeStamp});
            }
        }
       
        boost::posix_time::ptime const time_epoch1(boost::gregorian::date(1970, 1, 1));
        auto after = (boost::posix_time::microsec_clock::universal_time() - time_epoch1).total_milliseconds();
        BOOST_LOG_TRIVIAL(info) <<"After list" << after;

        auto comparator = [](elem& a, elem& b) {return a.second < b.second; };
        sort(fileVector.begin(), fileVector.end(), comparator); //Sorting

        boost::posix_time::ptime const time_epoch2(boost::gregorian::date(1970, 1, 1));
        auto after2 = (boost::posix_time::microsec_clock::universal_time() - time_epoch2).total_milliseconds();
        BOOST_LOG_TRIVIAL(info) <<"After sort" << after2;

        if (diskSize > mProps.upperWaterMark)
        {
            for (int i = 0; i < fileVector.size(); i++)
            {
                std::string pathInString = (fileVector[i].first).string();
                if (boost::regex_match(pathInString, delPattern))
                {
                    auto size = boost::filesystem::file_size(fileVector[i].first);
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
    mDetail->checkDirectory();
    return true;
}
