#include "ArchiveSpaceManager.h"
#include "Module.h"
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <stdafx.h>
#include "Logger.h"

class ArchiveSpaceManager::Detail
{
public:
  Detail(ArchiveSpaceManagerProps &_props) : mProps(_props) {}

  ~Detail() {}

  void setProps(ArchiveSpaceManagerProps _props) { mProps = _props; }

 uint64_t estimateDirectorySize(boost::filesystem::path _dir)
{
    uint64_t dirSize = 0;
    int sample = 0;
    int inCount = 0;
    int countFreq = 0;
    uint64_t tempSize = 0;

    if (!boost::filesystem::exists(_dir) || !boost::filesystem::is_directory(_dir))
    {
        LOG_ERROR << "Directory does not exist or is not a directory: " << _dir.string();
        return 0;
    }

    try
    {
        for (const auto &entry : boost::filesystem::recursive_directory_iterator(_dir))
        {
            if (boost::filesystem::is_regular_file(entry))
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
                        tempSize = boost::filesystem::file_size(entry);
                        dirSize += tempSize * mProps.samplingFreq;
                    }
                    catch (const std::exception &e)
                    {
                        LOG_INFO << "Failed to get file size for " << entry.path().string() << ": "
                                 << e.what();
                    }
                }

                inCount++;
                countFreq++;
            }
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR << "Failed to iterate directory " << _dir.string() << ": " << e.what();
        return 0;
    }

    // Adjust for the remaining files if they are fewer than the sampling frequency
    if (inCount < mProps.samplingFreq && inCount > 0)
    {
      dirSize += tempSize * inCount;
    }

    LOG_INFO << "Total Directory Size: " << dirSize;
    return dirSize;
}

  boost::filesystem::path getOldestDirectory(boost::filesystem::path _cam)
  {
    for (const auto &camFolder : boost::filesystem::directory_iterator(_cam))
    {
      for (const auto &folder :
           boost::filesystem::recursive_directory_iterator(camFolder))
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

  boost::filesystem::path getOldestHourDirByName(const boost::filesystem::path& cameraDir)
  {
  boost::filesystem::path oldestDay;
  for (const auto& dayEntry : boost::filesystem::directory_iterator(cameraDir))
  {
  LOG_INFO<<"dayEntry path"<<dayEntry.path();
  if (!boost::filesystem::is_directory(dayEntry)) continue;
    if (oldestDay.empty() || dayEntry.path().filename() < oldestDay.filename())
    {
    oldestDay = dayEntry.path();
    }
  }

  boost::filesystem::path oldestHour;
    for (const auto& hourEntry : boost::filesystem::directory_iterator(oldestDay))
    {
      if (!boost::filesystem::is_directory(hourEntry)) continue;
      if (oldestHour.empty() || hourEntry.path().filename() < oldestHour.filename())
      { 
        LOG_INFO<<"hourEntry path"<<hourEntry.path();
        oldestHour = hourEntry.path();
      }
    }
  return oldestHour;
  }
  
   void manageDirectory()
  {
    auto comparator = [](const std::pair<boost::filesystem::path, uint64_t> &a,
                        const std::pair<boost::filesystem::path, uint64_t> &b)
    {
      return a.second < b.second;
    };
    while (archiveSize > mProps.lowerWaterMark)
    {
      for (const auto &camFolder :
           boost::filesystem::directory_iterator(mProps.pathToWatch))
      {
        
        boost::filesystem::path oldHrDir = getOldestHourDirByName(camFolder);

         if (!boost::filesystem::exists(oldHrDir) || !boost::filesystem::is_directory(oldHrDir))
        {
            LOG_ERROR << "Directory does not exist or is not a directory: " << oldHrDir.string();
            continue;
        }

        uint64_t lastWrite = boost::filesystem::last_write_time(oldHrDir);

            // Print oldest hour dir and its last write time for each camera
            std::time_t t = static_cast<std::time_t>(lastWrite);
            BOOST_LOG_TRIVIAL(info) << "Camera: " << camFolder.path().string()
                                    << " | Oldest Hour Dir: " << oldHrDir.string()
                                    << " | Last Write: " << std::asctime(std::localtime(&t));

        foldVector.push_back(
            {oldHrDir, boost::filesystem::last_write_time(oldHrDir)});
      }

        
      sort(foldVector.begin(), foldVector.end(),
           comparator); // Sorting the vector

      BOOST_LOG_TRIVIAL(info) << "Contents of foldVector:";
        for (const auto &item : foldVector)
        {
            std::time_t t = static_cast<std::time_t>(item.second);
            BOOST_LOG_TRIVIAL(info) << "Dir: " << item.first.string()
                                    << " | Last Write: " << std::asctime(std::localtime(&t));
        }

         if (foldVector.empty())
       {
          LOG_ERROR << "No valid directories to delete.";
          break;
       }

      uint64_t tempSize = 0;
      boost::filesystem::path delDir = foldVector[0].first;

        if (!boost::filesystem::exists(delDir) || !boost::filesystem::is_directory(delDir))
       {
          LOG_ERROR << "Directory to delete does not exist or is not a directory: " << delDir.string();
          foldVector.clear();
          continue;
       }
       BOOST_LOG_TRIVIAL(info) << "Deleting folder : " << delDir.string();
       tempSize = estimateDirectorySize(delDir);
       archiveSize = archiveSize - tempSize;
       LOG_INFO<<"archive size after deleting"<<archiveSize;
      try
      {
        boost::filesystem::remove_all(delDir);
        boost::filesystem::path parentDir = delDir.parent_path();
         if (boost::filesystem::exists(parentDir) && boost::filesystem::is_directory(parentDir) && boost::filesystem::is_empty(parentDir))
        {
            BOOST_LOG_TRIVIAL(info) << "Deleting parent directory : " << parentDir.string();
            try {
                boost::filesystem::remove_all(parentDir);
            } catch (const std::exception& e) {
                LOG_ERROR << "Could not delete parent directory: " << e.what();
            }
        }
      }
      catch (...)
      {
        LOG_ERROR << "Could not delete directory!..";
      }
      foldVector.clear();
      LOG_INFO<<"clearing the folder vectors";
   
    }
  }
  uint64_t diskOperation()
  {
    archiveSize = estimateDirectorySize(mProps.pathToWatch);
    LOG_INFO<<"archive size:"<<archiveSize;
    if (archiveSize > mProps.upperWaterMark)
    {
      LOG_INFO<<"upw hit manageDirectory:"<<mProps.upperWaterMark;
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
  std::vector<std::pair<boost::filesystem::path, uint64_t>> foldVector;
};

ArchiveSpaceManager::ArchiveSpaceManager(ArchiveSpaceManagerProps _props)
    : Module(SOURCE, "ArchiveSpaceManager", _props)
{
  mDetail.reset(new Detail(_props));
}

bool ArchiveSpaceManager::validateInputPins() { return true; }

bool ArchiveSpaceManager::validateOutputPins() { return true; }

bool ArchiveSpaceManager::validateInputOutputPins() { return true; }

void ArchiveSpaceManager::addInputPin(framemetadata_sp &metadata,
                                      string &pinId)
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

bool ArchiveSpaceManager::term() { return Module::term(); }

ArchiveSpaceManagerProps ArchiveSpaceManager::getProps()
{
  return mDetail->mProps;
}

void ArchiveSpaceManager::setProps(ArchiveSpaceManagerProps &props)
{
  Module::addPropsToQueue(props);
}

bool ArchiveSpaceManager::handlePropsChange(frame_sp &frame)
{
  ArchiveSpaceManagerProps props(mDetail->mProps);
  auto ret = Module::handlePropsChange(frame, props);
  mDetail->setProps(props);
  return ret;
}

bool ArchiveSpaceManager::produce()
{
  try
  {
    finalArchiveSpace = mDetail->diskOperation();
  }
   catch (const std::exception& e)
  {
    LOG_ERROR << "Archive Disk Manager encountered an error: " << e.what();
  }
  catch (...)
  {
    LOG_ERROR << "Archive Disk Manager encountered an unknown error.";
  }
  return true;
}