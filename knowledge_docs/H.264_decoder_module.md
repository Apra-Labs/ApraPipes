**H.264 Decoder Module** 

**Overview**

The H.264 Decoder module is responsible for decoding H.264-encoded video streams and converting them into raw image frames. The module supports both ARM64 (using V4L2) and non-ARM platforms (using NVIDIA Codec APIs). It extracts metadata, initializes the decoder helper, and processes frames for decoding.

### **Key Components**

1. **H264Decoder Class**  
   * Inherits from `Module`  
   * Manages input/output frame metadata and decoder initialization  
   * Responsible for validating input and output pins  
2. **Detail Class (Private Implementation)**  
   * Handles metadata extraction and decoder helper initialization  
   * Uses either `H264DecoderV4L2Helper` (ARM64) or `H264DecoderNvCodecHelper` (other platforms)  
   * Processes input frames and manages decoder lifecycle  
3. **Metadata Handling**  
   * Extracts SPS (Sequence Parameter Set) and PPS (Picture Parameter Set) from NAL units  
   * Uses `H264ParserUtils::parse_sps()` to determine video dimensions  
   * Stores metadata in `H264Metadata`

### **Workflow**

1. **Initialization**  
   * `H264Decoder` is instantiated with properties  
   * Output metadata is set (`RawImagePlanarMetadata` for decoded frames)  
   * `Detail::setMetadata()` extracts frame dimensions and initializes the decoder helper  
2. **Frame Processing**  
   * `process()` function buffers incoming frames and determines playback direction  
   * Handles NAL unit parsing and decides whether frames are buffered or decoded immediately  
   * Uses `compute()` to decode frames and send them to output  
3. **Buffering and Decoding**  
   * Uses `incomingFramesTSQ` to maintain timestamps of incoming frames  
   * Stores decoded frames in `decodedFramesCache`  
   * `bufferDecodedFrames()` manages frame caching and drops older frames if necessary  
4. **Handling Playback Commands**  
   * `handleCommand()` processes playback speed and seek commands  
   * Adjusts frame skipping based on playback speed (2x, 4x, 8x, etc.)  
   * Resets or flushes the queue when switching between forward and reverse playback  
5. **Termination**  
   * `term()` resets internal objects and stops decoder threads  
   * Ensures all frames are processed before cleanup

### **Error Handling & Logging**

* Validates input frame types and metadata  
* Uses `LOG_ERROR` and `LOG_INFO` for debugging issues  
* Ensures decoder is properly initialized before processing frames

