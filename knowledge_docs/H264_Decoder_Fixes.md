# Changes from Main Branch

This document summarizes the actual changes made in the current branch compared to the main branch.

## Modified Files

### 1. base/CMakeLists.txt
- Removed dependencies:
  ```cmake
  # Removed libre and baresip libraries
  - find_library(LIBRE_LIB NAMES libre.so libre.a REQUIRED)
  - find_library(BARESIP_LIB NAMES libbaresip.so REQUIRED)
  ```
- Removed include directories:
  ```cmake
  target_include_directories ( aprapipes PRIVATE
      # Removed baresip and libre includes
      - ${BARESIP_INC_DIR}
      - ${LIBRE_INC_DIR}
  )
  ```

### 2. base/src/H264Decoder.cpp
- Uncommented EOS processing code:
  ```cpp
  bool H264Decoder::processEOS(string& pinId)
  {
      auto frame = frame_sp(new EmptyFrame());
      mDetail->compute(frame->data(), frame->size(), frame->timestamp);
      LOG_ERROR << "processes sos " ;
      mShouldTriggerSOS = true;
      return true;
  }
  ```

### 3. base/src/H264DecoderNvCodecHelper.cpp
- Added CUDA context cleanup in destructor:
  ```cpp
  NvDecoder::~NvDecoder() {
      // Added CUDA context cleanup
      if(m_cuContext)
      {
          if (m_pMutex) m_pMutex->lock();
          cuCtxDestroy(m_cuContext);
          if (m_pMutex) m_pMutex->unlock();
      }
  }
  ```

### 4. base/src/H264EncoderNVCodecHelper.cpp
- Modified output bitstream handling:
  ```cpp
  // Changed condition for waiting on free output bitstreams
  - else
  + else if(!m_nvcodecResources->m_nFreeOutputBitstreams)
  {
      LOG_INFO << "waiting for free outputbitstream<> busy streams<" << m_nvcodecResources->m_nBusyOutputBitstreams << ">";
  }
  ```

### 5. base/src/H264Utils.cpp
- Added return value for getNalTypeAfterSpsPps:
  ```cpp
  H264Utils::H264_NAL_TYPE H264Utils::getNalTypeAfterSpsPps(void* frameData, size_t frameSize)
  {
      // Added missing return statement
      return typeFound;
  }
  ```

### 6. base/src/NvTransform.cpp
- Uncommented metadata reset in processEOS:
  ```cpp
  bool NvTransform::processEOS(string &pinId)
  {
      mDetail->outputMetadata.reset();
      return true;
  }
  ```

### 7. base/src/PipeLine.cpp
- Added control module thread handling:
  ```cpp
  void PipeLine::wait_for_all(bool ignoreStatus)
  {
      // Added control module thread join
      if ((modules[0]->controlModule) != nullptr)
      {
          Module& m = *(modules[0]->controlModule);
          m.myThread.join();
      }
  }

  void PipeLine::interrupt_wait_for_all()
  {
      // Added control module thread interrupt and join
      if ((modules[0]->controlModule) != nullptr)
      {
          Module& m = *(modules[0]->controlModule);
          m.myThread.interrupt();
      }
      // ... existing code ...
      if ((modules[0]->controlModule) != nullptr)
      {
          Module& m = *(modules[0]->controlModule);
          m.myThread.join();
      }
  }
  ```

### 8. base/vcpkg.json
- Removed platform-specific dependencies:
  ```json
  // Removed re and baresip dependencies
  - {
  -   "name": "re",
  -   "platform": "!windows"
  - },
  - {
  -   "name": "baresip",
  -   "platform": "!windows"
  - }
  ```

## Summary of Changes

1. **Dependency Updates**
   - Removed libre and baresip libraries and includes
   - Removed platform-specific dependencies from vcpkg.json

2. **Code Improvements**
   - Added proper CUDA context cleanup in NVIDIA decoder
   - Fixed missing return value in H264Utils
   - Improved output bitstream handling in encoder
   - Enhanced thread management for control module
   - Uncommented and enabled EOS processing in decoder and transform

3. **Application-Specific Changes**
   - Removed Application-specific comments and restrictions
   - Enabled EOS processing that was previously disabled for Application
   - Enabled metadata reset in transform that was previously disabled for Application

These changes focus on cleaning up dependencies, improving resource management, and removing Application-specific restrictions while maintaining proper cleanup and error handling. 