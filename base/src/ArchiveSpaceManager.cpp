#include <stdafx.h>
#include<cstdlib>
#include <boost/filesystem.hpp>
#include "Module.h"
#include "ArchiveSpaceManager.h"

class ArchiveSpaceManager::Detail
{
public:
    Detail(ArchiveSpaceManagerProps& _props) : mProps(_props)
    {
    }

    ~Detail()
    {
    }

    void setProps(ArchiveSpaceManagerProps _props)
    {
        mProps = _props;
    }

    uint32_t estimateDirectorySize(boost::filesystem::path _dir)
    {
        uint32_t dirSize = 0;
        int sample = 0;
        int inCount = 0;
        int countFreq = 0;
        uint32_t tempSize = 0;

        for (const auto& entry : boost::filesystem::recursive_directory_iterator(_dir))
        {
            if ((boost::filesystem::is_regular_file(entry)))
            {
                if (!(countFreq % mProps.samplingFreq))
                {
                    sample = (rand() % mProps.samplingFreq) + 1;
                    inCount = 0;
                }
                if (inCount == sample)
                {
                    tempSize = boost::filesystem::file_size(entry);
                    dirSize += tempSize * mProps.samplingFreq;
                }
                inCount++;
                countFreq++;
            }
        }
        if (inCount < mProps.samplingFreq)
        {
            dirSize = dirSize + (tempSize * inCount);
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
        return _cam;
    };

    void manageDirectory()
    {
        auto comparator = [](std::pair<boost::filesystem::path, uint32_t>& a, std::pair<boost::filesystem::path, uint32_t>& b) {return a.second < b.second; };
        while (archiveSize > mProps.lowerWaterMark)
        {
            for (const auto& camFolder : boost::filesystem::directory_iterator(mProps.pathToWatch))
            {
                boost::filesystem::path oldHrDir = getOldestDirectory(camFolder);
                foldVector.push_back({ oldHrDir,boost::filesystem::last_write_time(oldHrDir) });
            }
            sort(foldVector.begin(), foldVector.end(), comparator); //Sorting the vector

            uint32_t tempSize = 0;
            boost::filesystem::path delDir = foldVector[0].first;
            BOOST_LOG_TRIVIAL(info) << "Deleting file : " << delDir.string();
            tempSize = estimateDirectorySize(delDir);
            archiveSize = archiveSize - tempSize;
            try
            {
                boost::filesystem::remove_all(delDir);
            }
            catch (...)
            {
                LOG_ERROR << "Could not delete directory!..";
            }
            foldVector.clear();
        }
    }

    uint32_t diskOperation()
    {
        archiveSize = estimateDirectorySize(mProps.pathToWatch);
        if (archiveSize > mProps.upperWaterMark)
        {
            // manageDirectory();
            // Now Instead of Deleting we will send command frame to 
        }
        uint32_t tempSize = archiveSize;
        archiveSize = 0;
        return tempSize;
    }
    ArchiveSpaceManagerProps mProps;
    uint32_t archiveSize = 0;
    std::vector<std::pair<boost::filesystem::path, uint32_t>> foldVector;
};


ArchiveSpaceManager::ArchiveSpaceManager(ArchiveSpaceManagerProps _props)
    :Module(SOURCE, "ArchiveSpaceManager", _props)
{
    mDetail.reset(new Detail(_props));
}

bool ArchiveSpaceManager::validateInputPins()
{
    return true;
}

bool ArchiveSpaceManager::validateOutputPins()
{
    return true;
}

bool ArchiveSpaceManager::validateInputOutputPins()
{
    return true;
}

void ArchiveSpaceManager::addInputPin(framemetadata_sp& metadata, string& pinId)
{
    Module::addInputPin(metadata, pinId);
    Module::addOutputPin(metadata, pinId);
}

bool ArchiveSpaceManager::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool ArchiveSpaceManager::term()
{
    return Module::term();
}

ArchiveSpaceManagerProps ArchiveSpaceManager::getProps()
{
    return mDetail->mProps;
}

void ArchiveSpaceManager::setProps(ArchiveSpaceManagerProps& props)
{
    Module::addPropsToQueue(props);
}

bool ArchiveSpaceManager::handlePropsChange(frame_sp& frame)
{
    ArchiveSpaceManagerProps props(mDetail->mProps);
    auto ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    return ret;
}

bool ArchiveSpaceManager::process()
{
    try
    {
        finalArchiveSpace = mDetail->diskOperation();
    }
    catch (...)
    {
        LOG_ERROR << "Archive Disk Manager encountered an error.";
    }
    return true;
}
