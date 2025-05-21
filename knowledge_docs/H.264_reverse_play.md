**H.264 Reverse Play Feature Document**

### **Overview**

The H.264 Reverse Play feature enables the playback of H.264-encoded video in reverse order. Unlike traditional playback, reverse playback requires special handling due to inter-frame dependencies. The decoder buffers frames, reorders them, and processes them in reverse while ensuring smooth playback. It relies on GOP (Group of Pictures) buffering to maintain accurate frame dependencies.

### **Key Components**

1. **GOP Buffering Mechanism**  
   * `bufferBackwardEncodedFrames()` collects frames and organizes them for reverse playback  
   * `backwardGopBuffer` stores multiple GOPs to allow seamless reverse navigation  
   * Identifies I-frames (keyframes) to define GOP boundaries, ensuring correct frame decoding  
2. **Decoding in Reverse Order**  
   * `decodeFrameFromBwdGOP()` retrieves and decodes frames in reverse order, respecting frame dependencies  
   * Ensures all required frames are reconstructed correctly by including SPS/PPS headers  
   * Maintains `incomingFramesTSQ` to synchronize timestamps and prevent playback stutter  
3. **Handling Direction Change**  
   * Uses `dirChangedToBwd` and `dirChangedToFwd` flags to track playback direction changes  
   * Clears partially buffered GOPs when direction switches mid-sequence to prevent artifacts  
   * Ensures all buffered GOPs are processed before transitioning back to forward playback  
4. **Frame Synchronization**  
   * Ensures `foundIFrameOfReverseGop` is set before decoding frames  
   * Uses `sendDecodedFrame()` to dispatch frames in the correct sequence, preventing gaps in playback  
   * Drops incomplete GOPs when necessary to maintain a stable playback experience

### **Workflow**

1. **Buffering Reverse GOPs**  
   * Frames are buffered until an I-frame is found, marking the start of a new GOP  
   * The complete GOP is stored in `backwardGopBuffer` and prepared for decoding  
2. **Decoding and Playback**  
   * `decodeFrameFromBwdGOP()` retrieves and decodes frames one by one in reverse order  
   * Ensures that each I-frame is properly prepended with SPS/PPS to maintain compatibility  
   * Synchronizes timestamps to provide a smooth playback experience  
3. **Switching Between Forward and Reverse**  
   * Drops unnecessary frames when switching between forward and reverse playback  
   * Flushes the queue to prevent playback artifacts and maintain proper decoding order  
   * Resets decoder states to ensure smooth transition without glitches

### **Optimization Considerations**

* Implements efficient frame caching (`decodedFramesCache`) to reduce redundant decoding operations  
* Dynamically adjusts playback speed (supports speeds such as 2x, 4x, and higher)  
* Ensures memory-efficient buffering by intelligently discarding the oldest frames when needed

### **Error Handling & Debugging**

* Logs changes in playback direction and GOP processing steps  
* Handles cases where an expected I-frame is missing by resetting the decoder state  
* Ensures decoder states remain synchronized with playback requests to prevent unexpected behavior

**Summary in simple terms**: 

* In the Mp4reader module we read frames in reverse order i.e. p frames first and then followed by I frame. (PPPPPPPI).  
* In the decoder module we maintain the original order of frames using timestamp.(PPPPPPPPI).  
* We cache one GOP together, once we get the I frame of the GOP, we reverse it to the normal order (IPPPPPPPPPP) and decode it, since the decoder can only process this order.  
* Once we get the decoded raw frames (IPPPPPPP, here all the frames are decoded). We reverse the GOP back to the original order (Initially we saved the order using the timestamps of frame).  
* Finally render the decoded reverse GOP.

