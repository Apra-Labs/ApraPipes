# ApraPipes Samples

This Samples demonstrate the capability and usage of ApraPipes Modules into different applications.

## Running the samples:
- The samples will be build along with the ApraPipes build.
- Each sample has their own executable.
- The executables can be found in the location:
    - _build/samples/sample/Debug. Note that here sample in the path is the particular sample.
- To run the samples run this command:
```
./filename.exe arg1 arg2 ...
```
- Note that samples may have the arguments. The details are given in the below section.

## Samples
### 1. Timelapse:
This samples demonstrates the usage of ApraPipes modules which can be used to generate a timelapse or summary of the video.
#### - Modules Used:
- Mp4Reader: To Read the input video.
- MotionVectorExtractor: To extract the frames from the input video for configured thershold.
- ColorChange: To convert frames given by MotionVectorExtractor from BGR to RGB and then RGB to YUV420PLANAR.
- H264Encoder
- Mp4Writer: To write the generated timelapse video to the output path.
- arguments: 
    - arg1: input video path
    - arg2: output path
- The Unit test "timelapse_summary_test.cpp" is located inside the aprapipessampleut executable. Make sure to replace the "inputVideoPath" and "outputPath" with suitable paths.
