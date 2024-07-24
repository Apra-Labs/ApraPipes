#pragma once

#include "FrameMetadata.h"
#include <opencv2/core/types_c.h>

class EncodedImageMetadata : public FrameMetadata {
public:
  EncodedImageMetadata() : FrameMetadata(FrameType::ENCODED_IMAGE) {}
  // EncodedImageMetadata(std::string _hint) :
  // FrameMetadata(FrameType::RAW_IMAGE, _hint) {}
  EncodedImageMetadata(MemType _memType)
      : FrameMetadata(FrameType::ENCODED_IMAGE, _memType) {}

  EncodedImageMetadata(int _width, int _height)
      : FrameMetadata(FrameType::ENCODED_IMAGE, FrameMetadata::HOST) {

    width = _width;
    height = _height;
    // setDataSize();
  }

  void reset() {
    FrameMetadata::reset();
    // ENCODED_IMAGE
    width = NOT_SET_NUM;
    height = NOT_SET_NUM;
  }

  bool isSet() { return width != NOT_SET_NUM; }

  void setData(cv::Mat &img) {
    // applicable only for rgba, mono
    width = img.cols;
    height = img.rows;
  }

  void setData(EncodedImageMetadata &metadata) {
    FrameMetadata::setData(metadata);

    width = metadata.width;
    height = metadata.height;

    // setDataSize();
  }

  int getWidth() { return width; }

  int getHeight() { return height; }

protected:
  void initData(int _width, int _height, MemType _memType = MemType::HOST) {
    width = _width;
    height = _height;
  }

  // https://docs.opencv.org/4.1.1/d3/d63/classcv_1_1Mat.html
  int width = NOT_SET_NUM;
  int height = NOT_SET_NUM;
};