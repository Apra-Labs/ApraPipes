# **Motion Vector Calculation**

## **Overview**

Motion vectors (MVs) are crucial in video decoding as they indicate how a macroblock (MB) in the current frame references data from previous frames to enable motion compensation. This documentation explains how motion vectors are updated in the given `parse_mb_syn_cabac.cpp` file and `ParseInterPMotionInfoCabac` function (Github link: `https://github.com/Apra-Labs/openh264/blob/ApraPipesMaster/codec/decoder/core/src/parse_mb_syn_cabac.cpp`), specifically focusing on Context-Adaptive Binary Arithmetic Coding (CABAC) for P-slices and B-slices.

---

## **1\. Motion Vector Update Process**

### **1.1 Parsing and Updating Motion Vectors**

The motion vector update process follows these steps:

1. **Parse Reference Index (`ParseRefIdxCabac`)**: Determines which reference frame a macroblock is using by decoding reference indices stored in the CABAC bitstream.  
2. **Predict Motion Vector (`PredMv`)**: Estimates the motion vector based on neighboring macroblocks (top, left, top-right, and top-left) using median prediction.  
3. **Parse Motion Vector Difference (`ParseMvdInfoCabac`)**: Retrieves the motion vector difference (MVD) from the bitstream, which represents the delta from the predicted motion vector.  
4. **Compute Final Motion Vector**: Adds the decoded MVD to the predicted motion vector to obtain the final MV for the macroblock.  
5. **Update Motion Vector (`UpdateMotionVector`)**: Stores the final motion vector in the decoding context for use in motion compensation.  
   ---

   ## **2\. Motion Vector Update Functions**

   ### **2.1 General Motion Vector Storage and Update**

   #### **`UpdateMotionVector(PWelsDecoderContext pCtx, int16_t pMotionX, int16_t pMotionY, int16_t xOffset, int16_t yOffset)`**

* Updates the `mMotionVectorData` array with the motion vector `(pMotionX, pMotionY)` and offset `(xOffset, yOffset)`.  
* Increments `mMotionVectorSize` by 4 to keep track of stored vectors.  
* Advances the pointer for the next entry to allow storing multiple motion vectors sequentially.

  ### **2.2 Updating Motion Vector Differences (MVD)**

These functions store motion vector differences:

* `UpdateP16x16MvdCabac(SDqLayer* pCurDqLayer, int16_t pMvd[2], const int8_t iListIdx)`: Updates MVD for a 16x16 macroblock.  
* `UpdateP16x8MvdCabac(SDqLayer* pCurDqLayer, int16_t pMvdCache[LIST_A][30][MV_A], int32_t iPartIdx, int16_t pMvd[2], const int8_t iListIdx)`: Updates MVD for a 16x8 partition.  
* `UpdateP8x16MvdCabac(SDqLayer* pCurDqLayer, int16_t pMvdCache[LIST_A][30][MV_A], int32_t iPartIdx, int16_t pMvd[2], const int8_t iListIdx)`: Updates MVD for an 8x16 partition.

Each function updates the motion vector difference (`MVD`) based on partition size:

* **16x16 MBs**: One MVD per macroblock.  
* **16x8 MBs**: Two MVDs, one per 16x8 partition.  
* **8x16 MBs**: Two MVDs, one per 8x16 partition.  
* **8x8 MBs**: Four MVDs, one per 8x8 sub-block.

  ### **2.3 Updating Reference Indices**

Reference indices indicate which frame a macroblock refers to. Functions handling this include:

* `UpdateP16x8RefIdxCabac`: Updates reference indices for 16x8 partitions.  
* `UpdateP8x16RefIdxCabac`: Updates reference indices for 8x16 partitions.  
* `UpdateP8x8RefIdxCabac`: Updates reference indices for 8x8 partitions.

  ### **2.4 Parsing and Assigning Motion Vectors**

Motion vector updates depend on the macroblock type.

#### **`ParseInterPMotionInfoCabac` (for P-slices)**

* Parses reference indices using `ParseRefIdxCabac`.  
* Predicts motion vectors using `PredMv`.  
* Retrieves motion vector differences using `ParseMvdInfoCabac`.  
* Computes final motion vectors and updates them using `UpdateMotionVector`.

  #### **`ParseInterBMotionInfoCabac` (for B-slices)**

* Determines if a block uses **spatial** (`PredMvBDirectSpatial`) or **temporal** (`PredBDirectTemporal`) prediction.  
* Uses the same update process as P-slices but considers bidirectional prediction.  
  ---

  ## **3\. Motion Vector Assignment by Macroblock Type**

| Macroblock Type | Motion Vector Handling |
| :---- | ----- |
| **16x16** | Single motion vector for the entire macrobloc |
| **16x8** | Two motion vectors, one for each 16x8 partition. |
| **8x16** | Two motion vectors, one for each 8x16 partition. |
| **8x8** | Four motion vectors, each for an 8x8 sub-block |
| **8x4** | Eight motion vectors, two for each 8x4 sub-block |
| **4x8** | Eight motion vectors, two for each 4x8 sub-block. |
| **4x4** | Sixteen motion vectors, one for each 4x4 sub-block |

Each partition's motion vectors are stored in `pMotionVector[LIST_A][30][MV_A]` and used for motion compensation.

---

## **4\. Summary**

Motion vectors in `parse_mb_syn_cabac.cpp` are updated through the following steps:

1. **Reference Index Parsing** (`ParseRefIdxCabac`): Determines which reference frame to use.  
2. **Motion Vector Prediction** (`PredMv`): Estimates the motion vector from neighboring macroblocks.  
3. **Motion Vector Difference Decoding** (`ParseMvdInfoCabac`): Retrieves the difference from the predicted motion vector.  
4. **Final Motion Vector Computation** (`pMv[0] += pMvd[0]`, `pMv[1] += pMvd[1]`): Computes the actual motion vector.  
5. **Updating Motion Vectors and MVDs** (`UpdateMotionVector`, `UpdateP16x16MvdCabac`, etc.): Stores final motion vectors for decoding.

These steps ensure accurate motion compensation during H.264 decoding using CABAC.

---

## **5\. References**

* H.264 Standard (ISO/IEC 14496-10)  
* Cisco OpenH264 Library  
* Context-Adaptive Binary Arithmetic Coding (CABAC)


