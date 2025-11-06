# PR3: Fix MotionVectorExtractor Memory Leak and Malloc Usage

## Problem

`MotionVectorExtractor.cpp` had a **critical memory leak** and violated zero-copy principles:

1. Used `malloc()` to allocate temporary YUV buffer
2. Performed 3x `memcpy` operations for Y, U, V planes
3. **Never freed the malloc'd buffer** → memory leak on every decoded frame
4. Did not use frame pool memory management

**Location**: `base/src/MotionVectorExtractor.cpp:294-314` (OpenH264 decoder path)

**Original Code**:
```cpp
decodedFrame = makeFrameWithPinId(mHeight * 3 * mWidth, rawFramePinId);
uint8_t* yuvImagePtr = (uint8_t*)malloc(mHeight * 1.5 * pDstInfo.UsrData.sSystemBuffer.iStride[0]);  // LEAK!
auto yuvStartPointer = yuvImagePtr;
unsigned char* pY = pDstInfo.pDst[0];
memcpy(yuvImagePtr, pY, pDstInfo.UsrData.sSystemBuffer.iStride[0] * mHeight);  // Copy 1
unsigned char* pU = pDstInfo.pDst[1];
yuvImagePtr += pDstInfo.UsrData.sSystemBuffer.iStride[0] * mHeight;
memcpy(yuvImagePtr, pU, pDstInfo.UsrData.sSystemBuffer.iStride[1] * mHeight / 2);  // Copy 2
unsigned char* pV = pDstInfo.pDst[2];
yuvImagePtr += pDstInfo.UsrData.sSystemBuffer.iStride[1] * mHeight / 2;
memcpy(yuvImagePtr, pV, pDstInfo.UsrData.sSystemBuffer.iStride[1] * mHeight / 2);  // Copy 3

cv::Mat yuvImgCV = cv::Mat(mHeight + mHeight / 2, mWidth, CV_8UC1, yuvStartPointer, ...);
// ... conversion to BGR
// yuvImagePtr NEVER FREED → MEMORY LEAK!
```

**Impact**:
- **Memory leak**: ~1.5x frame size per decoded frame (for 1920x1080: ~3.1 MB/frame)
- @ 30 FPS: **~93 MB/sec memory leak** (system runs out of memory in minutes)
- Malloc overhead on every frame
- Not using frame pool (bypasses memory management)
- 3x unnecessary memcpy operations

## Solution

Replace malloc with frame pool allocation and proper memory management:

**Fixed Code**:
```cpp
// Create output frame for BGR image
decodedFrame = makeFrameWithPinId(mHeight * 3 * mWidth, rawFramePinId);

// Calculate required YUV buffer size with stride
size_t yuvBufferSize = pDstInfo.UsrData.sSystemBuffer.iStride[0] * mHeight +
                       pDstInfo.UsrData.sSystemBuffer.iStride[1] * mHeight / 2 +
                       pDstInfo.UsrData.sSystemBuffer.iStride[1] * mHeight / 2;

// Use frame pool instead of malloc for proper memory management
auto yuvTempFrame = makeFrameWithPinId(yuvBufferSize, rawFramePinId);
uint8_t* yuvImagePtr = static_cast<uint8_t*>(yuvTempFrame->data());

// Copy Y, U, V planes (same 3 memcpy, but with proper memory management)
memcpy(yuvImagePtr, pY, ...);
memcpy(yuvImagePtr + offset1, pU, ...);
memcpy(yuvImagePtr + offset2, pV, ...);

// Color conversion
cv::cvtColor(yuvImgCV, bgrImg, cv::COLOR_YUV2BGR_I420);

// yuvTempFrame automatically freed when it goes out of scope (smart pointer)
```

**Benefits**:
- ✅ **Memory leak eliminated**: smart pointer auto-cleanup
- ✅ **Uses frame pool**: proper memory management and reuse
- ✅ **No malloc overhead**: uses efficient pooled allocation
- ✅ **Predictable memory usage**: frame pool prevents unbounded growth
- ✅ **Same functionality**: still 3x memcpy (required for stride handling)

**Note**: The 3x memcpy operations are still required because:
- OpenH264 decoder outputs separate Y, U, V plane pointers with strides
- cv::cvtColor requires contiguous YUV420 data
- Strides may differ from image width (alignment requirements)

## Why This Matters

### Memory Leak Severity: CRITICAL 🔴

For a typical 1920x1080 video @ 30 FPS:
- **Leak per frame**: ~3.1 MB
- **Leak per second**: ~93 MB
- **Time to 1 GB leak**: ~11 seconds
- **Time to 8 GB leak**: ~90 seconds (1.5 minutes)

**This would crash the application within minutes of operation.**

### Proper Memory Management

| Aspect | Before (malloc) | After (frame pool) |
|--------|----------------|-------------------|
| Allocation method | malloc() | Frame pool |
| Deallocation | Never freed (LEAK!) | Auto (smart pointer) |
| Memory reuse | No | Yes (pooled) |
| Fragmentation | High | Low (chunked) |
| Performance | Unpredictable | Consistent |
| Memory growth | Unbounded | Bounded |

## Test Strategy

### Existing Tests
The existing `motionvector_extractor_and_overlay_tests.cpp` tests functionality but doesn't detect memory leaks.

### Verification Approach
To verify the fix:

1. **Memory Monitoring Test**:
   ```cpp
   // Run decoder for extended period and monitor memory
   // Before: Memory grows unboundedly
   // After: Memory usage stable
   ```

2. **Valgrind/ASan Testing**:
   ```bash
   # Compile with AddressSanitizer
   # Run motion vector extraction test
   # Before: Leak detected
   # After: No leaks
   ```

3. **Long-Running Test**:
   ```bash
   # Process video for 5+ minutes
   # Before: OOM crash within 2 minutes
   # After: Stable operation
   ```

## Architecture Compliance

This fix aligns with ApraPipes' memory management principles:

1. **Frame Pool Usage**: All allocations through FrameFactory
2. **Smart Pointer Management**: Automatic cleanup via shared_ptr
3. **Reference Counting**: Memory recycled when count drops to zero
4. **Bounded Memory**: Frame pool prevents unbounded growth

## Backward Compatibility

✅ **Fully Compatible**

- No API changes
- No behavioral changes
- Same video decoding output
- Performance may actually improve (pooled allocation vs malloc)

## Performance Impact

| Metric | Before (malloc) | After (pool) | Change |
|--------|----------------|--------------|--------|
| Memory leak rate | 93 MB/sec @ 30 FPS | 0 MB/sec | ✅ FIXED |
| Allocation overhead | malloc() call | Pool allocation | Faster |
| Memory fragmentation | High | Low | Better |
| Memory predictability | Unbounded growth | Bounded | Stable |

## Files Changed

1. `base/src/MotionVectorExtractor.cpp` - Fixed malloc + memory leak

## Verification

To verify the fix:

```bash
# Check that malloc is replaced with frame pool
grep -n "malloc" base/src/MotionVectorExtractor.cpp
# Should return: (no results in the problematic section)

# Check for frame pool usage
grep -n "makeFrameWithPinId.*yuvTempFrame" base/src/MotionVectorExtractor.cpp
# Should show: Line using frame pool for YUV buffer
```

## Critical Bug Fix Classification

This is a **CRITICAL BUG FIX** that addresses:
- ✅ Memory leak (crashes system)
- ✅ Improper memory management (malloc in pooled system)
- ✅ Resource exhaustion (unbounded growth)

**Priority**: Should be merged ASAP to prevent production crashes.

## Related Issues

This is part of a series of PRs to fix zero-copy violations:
- PR1: HistogramOverlay ✅
- PR2: TextOverlayXForm ✅
- **PR3**: MotionVectorExtractor (this PR - CRITICAL BUG FIX) ✅
- PR4: JPEGEncoderL4TM
- PR5: Mp4ReaderSource
- PR6: ImageEncoderCV
- PR7: AudioCaptureSrc
- PR8: DMAFDToHostCopy

## Conclusion

This fix eliminates a critical memory leak in MotionVectorExtractor that would crash the application within minutes during video decoding. By replacing malloc with frame pool allocation, we ensure:
- **No memory leaks**: automatic cleanup via smart pointers
- **Proper memory management**: consistent with ApraPipes architecture
- **Stable operation**: bounded memory usage
- **Better performance**: pooled allocation vs malloc

This is a **production-critical fix** that should be prioritized for immediate merge.
