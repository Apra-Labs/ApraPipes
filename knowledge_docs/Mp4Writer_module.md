# **Mp4WriterSink Documentation**

## **Overview**

The `Mp4WriterSink` module is responsible for writing video frames into MP4 files. It supports both JPEG-encoded images and H.264-encoded video frames. The module manages frame synchronization, metadata embedding, and proper handling of MP4 file segmentation based on chunk duration.

## **Dependencies**

The module includes the following headers:

* **FrameMetadata.h** \- Defines metadata for frames.  
* **Frame.h** \- Provides frame structures.  
* **H264Utils.h** \- Utility functions for handling H.264 bitstreams.  
* **Mp4VideoMetadata.h** \- Manages MP4 video metadata.  
* **Mp4WriterSink.h** \- Header file for `Mp4WriterSink`.  
* **Mp4WriterSinkUtils.h** \- Utility functions for MP4 file handling.  
* **EncodedImageMetadata.h** \- Handles encoded image metadata.  
* **Module.h** \- Base module functionalities.  
* **libmp4.h** \- External library for MP4 file writing.  
* **PropsChangeMetadata.h** \- Manages property changes.  
* **H264Metadata.h** \- Provides H.264-specific metadata.

## **DTS Calculation Strategies**

### **`DTSCalcStrategy`**

Defines an abstract class for determining DTS (Decoding Time Stamp) based on frame timestamps. It supports two strategies:

### **`DTSPassThroughStrategy`**

* Used when `recordedTSBasedDTS` is enabled.  
* Adjusts frame timestamp based on the difference from the last frame.  
* Ensures smooth timestamp transitions by introducing a slight offset when timestamps remain unchanged or go backward.

### **`DTSFixedRateStrategy`**

* Used when timestamps are strictly based on FPS.  
* Uses a fixed duration per frame based on FPS to maintain a steady frame rate.  
* Ensures that frames are evenly spaced in time.

## **`DetailAbs` \- Abstract Base Class**

Manages MP4 file writing logic and common functionality:

* Initializes `mp4_mux` for MP4 file creation.  
* Handles video metadata and track configuration.  
* Determines whether timestamps are calculated based on recorded timestamps or a fixed frame rate.  
* Supports metadata embedding and ensures proper file closure.

### **Key Methods:**

* `setProps(Mp4WriterSinkProps&)`: Updates internal properties with new values.  
* `initNewMp4File(std::string&)`: Creates a new MP4 file and initializes the multiplexer.  
* `write(frame_container&)`: Abstract method for writing frames, implemented in derived classes.  
* `addMetadataInVideoHeader(frame_sp)`: Embeds metadata in video headers to provide additional context.  
* `attemptFileClose()`: Closes the MP4 file safely, ensuring that all data is properly written.

## **`DetailJpeg` \- JPEG Handling**

Extends `DetailAbs` to process JPEG images:

* Sets video decoder configuration to `MP4_VIDEO_CODEC_MP4V`.  
* Extracts JPEG frames from incoming data and writes them to an MP4 file.  
* Handles timestamp synchronization and metadata injection.

### **Key Methods:**

* `write(frame_container&)`: Processes encoded JPEG frames and appends them to the MP4 stream.

## **`DetailH264` \- H.264 Handling**

Handles H.264-encoded video streams and NALU (Network Abstraction Layer Unit) processing:

* Extracts SPS (Sequence Parameter Set) and PPS (Picture Parameter Set) headers.  
* Modifies frames when new SPS/PPS are encountered to ensure compatibility.  
* Uses `mp4_mux_track_add_sample_with_prepend_buffer()` to ensure correct playback by prepending NALU headers.

### **Key Methods:**

* `modifyFrameOnNewSPSPPS()`: Inserts SPS/PPS before IDR (Instantaneous Decoder Refresh) frames to ensure video decoding starts correctly.  
* `write(frame_container&)`: Writes H.264 frames into the MP4 container, ensuring correct frame order and timestamps.

## **`Mp4WriterSink` \- Main Class**

Manages the entire MP4 writing pipeline:

* Detects frame type (JPEG/H.264) and initializes the corresponding handler.  
* Processes incoming frames and writes them into MP4 files.  
* Supports property updates, metadata injection, and error handling.  
* Ensures synchronization between frames to maintain smooth playback.

### **Key Methods:**

* `init()`: Initializes the module and determines the appropriate handler for frame writing.  
* `validateInputOutputPins()`: Ensures valid input/output configuration to prevent misconfiguration.  
* `setMetadata(framemetadata_sp&)`: Updates frame metadata based on input data.  
* `process(frame_container&)`: Processes and writes frames while handling synchronization.  
* `term()`: Closes files and releases resources during termination.  
* `doMp4MuxSync()`: Synchronizes the MP4 multiplexer to ensure file integrity.

## **`Mp4WriterSinkUtils` \- Utility Functions**

Manages file naming, path handling, and timestamp-based chunking:

* Ensures that files are created in structured directories based on timestamps.  
* Handles custom file naming when specified.  
* Determines new file names based on timestamp and synchronization settings.

### **Key Methods:**

* `filePath()`: Constructs the MP4 file path based on timestamp and directory structure.  
* `customNamedFileDirCheck()`: Checks for custom file naming and verifies directory permissions.  
* `parseTSJpeg()`: Handles timestamp-based segmentation for JPEG frames, ensuring correct file placement.  
* `parseTSH264()`: Handles timestamp-based segmentation for H.264 frames, correctly grouping frames into appropriate files.  
* `getFilenameForNextFrame()`: Determines the next MP4 file name based on frame timestamp and segmentation rules.

## **Conclusion**

The `Mp4WriterSink` module is a robust MP4 writing solution, efficiently handling both JPEG and H.264 streams while ensuring proper synchronization and metadata management. It ensures seamless recording and playback compatibility, making it suitable for real-time and archival video storage applications.

