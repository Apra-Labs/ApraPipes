# **Mp4ReaderSource.cpp**

## **Overview**

`Mp4ReaderSource.cpp` is a C++ implementation of an MP4 file reader that parses MP4 video files, extracts frames, and manages playback functionalities. It utilizes the `libmp4` library for MP4 file parsing and decoding. The implementation supports both H.264 and JPEG encoded videos and provides functionalities such as seeking, metadata extraction, playback speed adjustment, and error handling.

This module is designed to be integrated into a larger video processing system, allowing seamless retrieval of frames from MP4 files and efficient handling of video streams.

## **Class Overview**

### **1\. Mp4ReaderDetailAbs**

#### **Description:**

This is an abstract base class that defines common functionalities required for reading MP4 files, such as metadata handling, seeking, frame extraction, and playback controls.

#### **Key Functions:**

* **`setMetadata()`**: Initializes and updates MP4 metadata, including frame format and serialization version.  
* **`sendEndOfStream()`**: Pure virtual function to be implemented in derived classes, signaling the end of the stream.  
* **`produceFrames(frame_container& frames)`**: Pure virtual function for producing frames by reading data from the MP4 file.  
* **`mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int& seekedToFrame)`**: Seeks to a specific timestamp in the video file.  
* **`getGop()`**: Retrieves the Group of Pictures (GOP) size, which determines the number of frames between keyframes.  
* **`Init()`**: Initializes the MP4 reader by validating file paths, checking file formats, and preparing metadata.  
* **`updateMstate(Mp4ReaderSourceProps& props, std::string videoPath)`**: Updates internal state variables based on the video file path and properties.  
* **`setProps(Mp4ReaderSourceProps& props)`**: Configures the reader with new properties, such as enabling file system parsing.  
* **`getOpenVideoPath()`**: Returns the currently open video file path.  
* **`getOpenVideoFrameCount()`**: Returns the total number of frames in the currently open video file.  
* **`refreshCache()`**: Refreshes the cache of video files to detect new content.  
* **`attemptFileClose()`**: Safely closes the currently open MP4 file.  
* **`parseFS()`**: Parses the file system for new MP4 files if required, updating the internal cache.  
* **`initNewVideo(bool firstOpenAfterInit = false)`**: Opens a new video file and prepares it for reading.  
* **`openVideoSetPointer(std::string& filePath)`**: Opens the specified MP4 file and sets the read pointer at the appropriate position.  
* **`randomSeekInternal(uint64_t& skipTS, bool forceReopen = false)`**: Performs precise seeking operations within the MP4 file.  
* **`randomSeek(uint64_t& skipTS, bool forceReopen = false) noexcept`**: Wrapper function for `randomSeekInternal`, with added exception handling.  
* **`readNextFrame(frame_sp& imgFrame, frame_sp& metadetaFrame, size_t& imgSize, size_t& metadataSize, uint64_t& frameTSInMsecs, int32_t& mp4FIndex) noexcept`**: Reads the next available frame from the video and returns its metadata.  
* **`calcReloadFileAfter()`**: Computes the timestamp for reloading files based on playback behavior.  
* **`termOpenVideo()`**: Closes the currently open video and resets the internal state variables.  
* **`makeAndSendMp4Error(int errorType, int errorCode, std::string errorMsg, int openErrorCode, uint64_t _errorMp4TS)`**: Logs and sends error frames for debugging and error recovery.

#### **Key Variables:**

* `mProps`: Stores video reader properties such as file path and playback settings.  
* `mState`: Stores the internal state of the MP4 reader, including video metadata, file position, and playback direction.  
* `cof`: A shared pointer to `OrderedCacheOfFiles`, used for managing cached video files efficiently.  
* `mWidth`, `mHeight`: Stores video resolution parameters.  
* `mFPS`: Tracks the frames per second of the video.  
* `playbackSpeed`: Controls playback speed dynamically.  
* `sentEOSSignal`: Ensures that the End-Of-Stream signal is sent only once when playback reaches the end.

### **2\. Mp4ReaderDetailJpeg**

#### **Description:**

Derived from `Mp4ReaderDetailAbs`, this class specifically handles JPEG-encoded video files, extracting frames and metadata without requiring complex video decoding.

#### **Key Functions:**

* **`setMetadata()`**: Sets metadata for JPEG video frames, such as dimensions and encoding format.  
* **`mp4Seek()`**: Implements seek functionality tailored for JPEG sequences.  
* **`getGop()`**: Returns GOP (always 0 for JPEG, as it does not use inter-frame compression).  
* **`produceFrames()`**: Extracts and processes frames from the JPEG video stream.

### **3\. Mp4ReaderDetailH264**

#### **Description:**

Derived from `Mp4ReaderDetailAbs`, this class handles H.264-encoded video files, ensuring proper decoding and handling of SPS/PPS headers.

#### **Key Functions:**

* **`setMetadata()`**: Sets metadata specific to H.264 video frames.  
* **`readSPSPPS()`**: Extracts SPS (Sequence Parameter Set) and PPS (Picture Parameter Set) required for H.264 decoding.  
* **`mp4Seek()`**: Implements seek functionality for H.264, supporting I-frame seeking.  
* **`getGop()`**: Retrieves the GOP size for H.264 streams.  
* **`sendEndOfStream()`**: Sends an End-Of-Stream frame upon reaching the last frame.  
* **`prependSpsPps()`**: Appends SPS and PPS headers to H.264 frames when required.  
* **`produceFrames()`**: Extracts H.264 frames and ensures SPS and PPS are included when necessary.

## **Advanced Playback Features**

* **Frame Skipping**: Supports skipping frames to accelerate playback.  
* **Adaptive Buffering**: Manages memory efficiently based on playback speed.  
* **Error Handling**: Recovers from missing or corrupted frames.  
* **Looping & Seeking**: Enables smooth looping and frame-accurate seeking.

# **OrderedCacheOfFiles.cpp**

## **Overview**

`OrderedCacheOfFiles.cpp` is a C++ implementation that manages an ordered cache of MP4 video files. It enables efficient file searching, timestamp-based retrieval, and cache management for video playback systems. The implementation relies on the `boost::filesystem` library for file operations and `libmp4` for MP4 metadata handling.

This class is designed to work with a directory structure where video files are organized in timestamped folders, enabling quick access to video segments based on time-based queries.

## **Class Overview**

### **1\. OrderedCacheOfFiles**

#### **Description:**

This class manages a cache of video files to facilitate fast retrieval and playback. It maintains an ordered index of video files with their respective timestamps, allowing for efficient seeking, cache refreshing, and directory parsing.

#### **Key Functions:**

* **`OrderedCacheOfFiles(std::string& video_folder, uint32_t initial_batch_size, uint32_t _lowerWaterMark, uint32_t _upperWaterMark)`**  
  * Initializes the cache with the given video folder path and configuration parameters.  
* **`uint64_t getFileDuration(std::string& filename)`**  
  * Returns the duration of a video file by computing the difference between its start and end timestamps.  
* **`bool fetchFromCache(std::string& videoFile, uint64_t& start_ts, uint64_t& end_ts)`**  
  * Retrieves the start and end timestamps of a given video file if it exists in the cache.  
* **`bool fetchAndUpdateFromDisk(std::string videoFile, uint64_t& start_ts, uint64_t& end_ts)`**  
  * Fetches video timestamps and forces an update from the disk if needed.  
* **`std::map<std::string, std::pair<uint64_t, uint64_t>> getSnapShot()`**  
  * Returns a snapshot of the video cache, listing file paths along with their start and end timestamps.  
* **`bool probe(boost::filesystem::path potentialMp4File, std::string& videoName)`**  
  * Probes a directory to check if an MP4 file exists and retrieves its name.  
* **`bool getPreviousAndNextFile(std::string videoPath, std::string& previousFile, std::string& nextFile)`**  
  * Finds and returns the previous and next video files relative to a given file in the cache.  
* **`std::string getFileAt(uint64_t timestamp, bool direction)`**  
  * Retrieves the most relevant file for a given timestamp, taking playback direction into account.  
* **`std::string getNextFileAfter(std::string& currentFile, bool direction)`**  
  * Retrieves the next video file after a specified file in a given direction.  
* **`bool isTimeStampInFile(std::string& filePath, uint64_t timestamp)`**  
  * Checks if a specific timestamp is contained within a given video file.  
* **`void readVideoStartEnd(std::string& filePath, uint64_t& start_ts, uint64_t& end_ts)`**  
  * Extracts start and end timestamps of a video file using `libmp4`.  
* **`void updateCache(std::string& filePath, uint64_t& start_ts, uint64_t& end_ts)`**  
  * Updates the cache entry for a specific file.  
* **`bool getRandomSeekFile(uint64_t skipTS, bool direction, uint64_t& skipMsecs, std::string& skipVideoFile)`**  
  * Finds the best file for a given timestamp and direction, considering cache updates.  
* **`bool getFileFromCache(uint64_t timestamp, bool direction, std::string& fileName)`**  
  * Retrieves a file from the cache based on a timestamp.  
* **`void insertInVideoCache(Video vid)`**  
  * Inserts a new video file entry into the cache.  
* **`bool parseFiles(uint64_t start_ts, bool direction, bool includeFloorFile, bool disableBatchSizeCheck, uint64_t skipTS)`**  
  * Parses the directory structure to build an ordered cache of MP4 files.  
* **`void retireOldFiles(uint64_t ts)`**  
  * Removes old files from the cache to maintain storage limits.  
* **`void dropFarthestFromTS(uint64_t ts)`**  
  * Deletes files farthest from the specified timestamp.  
* **`void deleteLostEntry(std::string& filePath)`**  
  * Removes an entry from the cache if the corresponding file no longer exists.  
* **`void clearCache()`**  
  * Clears the entire cache.  
* **`bool refreshCache()`**  
  * Refreshes the cache by re-parsing the directory for new files.

#### **Directory Parsing Methods:**

* **`std::vector<boost::filesystem::path> parseAndSortDateDir(const std::string& rootDir)`**  
  * Parses and sorts date-based directories (YYYYMMDD format).  
* **`std::vector<boost::filesystem::path> parseAndSortHourDir(const std::string& dateDirPath)`**  
  * Parses and sorts hour-based directories (HHMM format).  
* **`std::vector<boost::filesystem::path> parseAndSortMp4Files(const std::string& hourDirPath)`**  
  * Parses and sorts MP4 files within an hour-based directory.

#### **Helper Functions:**

* **`bool filePatternCheck(const fs::path& path)`**  
  * Validates MP4 file naming conventions.  
* **`bool datePatternCheck(const boost::filesystem::path& path)`**  
  * Checks if a directory follows a valid date format.  
* **`bool hourPatternCheck(const boost::filesystem::path& path)`**  
  * Checks if a directory follows a valid hour format.

## **Error Handling**

* **`MP4_OCOF_EMPTY`**: Triggered when a cache operation is attempted on an empty cache.  
* **`MP4_OCOF_MISSING_FILE`**: Raised when a required video file is missing in the cache.  
* **`MP4_OPEN_FILE_FAILED`**: Occurs when a file cannot be opened by `libmp4`.  
* **`MP4_UNEXPECTED_STATE`**: Raised when an unexpected inconsistency occurs in the cache.

## **Cache Management**

The class maintains an ordered list of video files and dynamically updates the cache as new files appear or old files are removed. The cache is optimized to support efficient seeking and playback across large video datasets.

## **Playback Features**

* **Efficient Timestamp-Based Seeking**: Finds video files based on timestamps for smooth playback.  
* **Dynamic Cache Updating**: Automatically refreshes the cache as new files are added.  
* **Optimized File Searching**: Uses ordered maps for quick lookup.  
* **Customizable Cache Size**: Allows control over how many files are stored in memory.  
* **Playback Direction Handling**: Supports forward and backward playback modes.

## **Conclusion**

`OrderedCacheOfFiles.cpp` is a robust and optimized cache management system designed for handling MP4 video files in time-sequenced directories. It ensures smooth video playback, efficient seeking, and scalable cache management, making it an essential component for video processing applications.

## **Test Files and Their Purpose**

### **1\. mp4\_dts\_strategy\_tests.cpp**

#### **Description:**

Validates the decoding timestamp (DTS) strategy for MP4 playback. It ensures that the MP4 writer can handle timestamp-based writing without gaps and reconstructs fixed-rate videos.

#### **Key Test Cases:**

* **`read_mul_write_one_as_recorded()`**: Tests writing multiple input videos while preserving recorded timestamps.  
* **`read_mul_write_one_fixed_rate()`**: Ensures a fixed playback rate by removing time gaps between segments.

### **2\. mp4\_getlivevideots\_tests.cpp**

#### **Description:**

Tests live video timestamp retrieval from MP4 files to ensure synchronization.

#### **Key Test Cases:**

* **`seek_read_loop()`**: Ensures that timestamps match expected values when looping playback.  
* **`getTSFromFileName()`**: Extracts timestamps from MP4 file names for validation.

### **3\. mp4\_reverse\_play\_tests.cpp**

#### **Description:**

Verifies that MP4 files can be played in reverse order correctly while handling timestamps and frame sequencing.

#### **Key Test Cases:**

* **`fwd()`**: Tests forward playback with correct timestamp sequencing.  
* **`switch_playback()`**: Ensures smooth transition between forward and backward playback.

### **4\. mp4\_seek\_tests.cpp**

#### **Description:**

Tests seeking operations within MP4 files, including handling missing segments and edge cases.

#### **Key Test Cases:**

* **`no_seek()`**: Reads frames sequentially without seeking.  
* **`seek_in_current_file()`**: Seeks to a timestamp within the current file.  
* **`seek_in_next_file()`**: Ensures seeking between different files.  
* **`seek_in_file_in_next_hr()`**: Handles seeking across hourly file splits.  
* **`seek_fails_no_reset()`**: Ensures state consistency when seeking fails.

### **5\. mp4\_simul\_read\_write\_tests.cpp**

#### **Description:**

Verifies the ability to read from and write to MP4 files simultaneously.

#### **Key Test Cases:**

* **`basic()`**: Tests basic read/write pipeline.  
* **`basic_parseFS_disabled()`**: Validates reading and writing when file system parsing is disabled.  
* **`loop_no_chunking()`**: Ensures proper looping when writing MP4 files without chunking.

### **6\. mp4readersource\_tests.cpp**

#### **Description:**

Tests the `Mp4ReaderSource` module for reading MP4 files and extracting metadata.

#### **Key Test Cases:**

* **`mp4v_to_jpg_frames_metadata()`**: Converts MP4 video frames to JPG and validates metadata.  
* **`mp4v_to_h264_frames_metadata()`**: Converts MP4 video frames to H.264 format.  
* **`read_timeStamp_from_custom_fileName()`**: Extracts timestamps from file names for validation.  
* **`getSetProps_change_root_folder()`**: Ensures the ability to change the root folder dynamically.

## **Common Testing Features**

* **Boost.Test Framework**: Provides assertions and test case management.  
* **Logging with Boost.Log**: Ensures traceability of operations.  
* **Pipeline Validation**: Ensures correct module connections and execution.  
* **Frame Metadata Extraction**: Validates correctness of timestamps and frame data.

## **Conclusion**

This test suite thoroughly evaluates MP4 reader and writer functionalities, ensuring robustness in timestamp handling, playback control, seeking, and real-time processing. The comprehensive coverage of edge cases ensures the reliability of the MP4 processing pipeline in various scenarios.

