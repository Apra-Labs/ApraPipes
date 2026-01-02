#include <stdafx.h>
#include<cstdlib>
#include <filesystem>
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

  uint64_t estimateDirectorySize(std::filesystem::path _dir) 
    {
    std::cout << "Estimating size for directory: " << _dir << std::endl;
    uint64_t dirSize = 0;
        int sample = 0;
        int inCount = 0;
        int countFreq = 0;
    uint64_t tempSize = 0;

        for (const auto& entry : std::filesystem::recursive_directory_iterator(_dir))
        {
      if (std::filesystem::is_regular_file(entry)) 
            {
        if (countFreq % mProps.samplingFreq == 0) 
                {
          sample = (rand() % mProps.samplingFreq);
                    inCount = 0;
                }
                if (inCount == sample)
        {
          try 
                {
                    tempSize = std::filesystem::file_size(entry);
                    dirSize += tempSize * mProps.samplingFreq;
          } 
          catch (const std::exception &e) 
          {
            std::cout << "Failed to get file size for " << entry << ": "
                      << e.what() << std::endl;
          }
                }
                inCount++;
                countFreq++;
            }
        }
    if (inCount < mProps.samplingFreq && inCount > 0) 
        {
      dirSize += tempSize * inCount;
        }
    std::cout << "Total Directory Size: " << dirSize << std::endl;
        return dirSize;
    }

    std::filesystem::path getOldestDirectory(std::filesystem::path _cam)
    {
        for (const auto& camFolder : std::filesystem::directory_iterator(_cam))
        {
            for (const auto& folder : std::filesystem::recursive_directory_iterator(camFolder))
            {
                if (std::filesystem::is_regular_file(folder))
                {
                    std::filesystem::path p = folder.path().parent_path();
                    return p;
                }
            }
        }
        return _cam;
    };

    void manageDirectory()
    {
    auto comparator = [](std::pair<std::filesystem::path, uint64_t>& a, std::pair<std::filesystem::path, uint64_t>& b) {return a.second < b.second; };
        while (archiveSize > mProps.lowerWaterMark)
        {
            for (const auto& camFolder : std::filesystem::directory_iterator(mProps.pathToWatch))
            {
                std::filesystem::path oldHrDir = getOldestDirectory(camFolder);
                foldVector.emplace_back(oldHrDir, std::filesystem::last_write_time(oldHrDir).time_since_epoch().count());
            }
            sort(foldVector.begin(), foldVector.end(), comparator); //Sorting the vector

      uint64_t tempSize = 0;
            std::filesystem::path delDir = foldVector[0].first;
            BOOST_LOG_TRIVIAL(info) << "Deleting file : " << delDir.string();
            tempSize = estimateDirectorySize(delDir);
            archiveSize = archiveSize - tempSize;
            try
            {
                std::filesystem::remove_all(delDir);
            }
            catch (...)
            {
                LOG_ERROR << "Could not delete directory!..";
            }
            foldVector.clear();
        }
    }

  uint64_t diskOperation() 
    {
        archiveSize = estimateDirectorySize(mProps.pathToWatch);
        if (archiveSize > mProps.upperWaterMark)
        {
            manageDirectory();
        }
    else 
    {
      LOG_INFO << "DiskSpace is under range";
    }
    uint64_t tempSize = archiveSize;
        archiveSize = 0;
        return tempSize;
    }
    ArchiveSpaceManagerProps mProps;
  uint64_t archiveSize = 0;
  std::vector<std::pair<std::filesystem::path, uint64_t>> foldVector;
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

void ArchiveSpaceManager::addInputPin(framemetadata_sp& metadata, std::string_view pinId)
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

bool ArchiveSpaceManager::produce() {
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
