#include <boost/foreach.hpp>
#include <cstdint>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libswscale/swscale.h>
#include <wels/codec_api.h>
}
#include "H264Metadata.h"
#include "H264ParserUtils.h"
#include "MotionVectorExtractor.h"
#include "Overlay.h"
#include "Utils.h"

class MvExtractDetailAbs {
public:
  MvExtractDetailAbs(
      MotionVectorExtractorProps props,
      std::function<frame_sp(size_t size, string &pinId)> _makeFrameWithPinId,
      std::function<frame_sp(frame_sp &bigFrame, size_t &size, string &pinId)>
          _makeframe) {
    makeFrameWithPinId = _makeFrameWithPinId;
    makeframe = _makeframe;
    sendDecodedFrame = props.sendDecodedFrame;
    sendOverlayFrame = props.sendOverlayFrame;
    threshold = props.motionVectorThreshold;
  };
  ~MvExtractDetailAbs() {}
  virtual void setProps(MotionVectorExtractorProps props) {
    sendDecodedFrame = props.sendDecodedFrame;
    sendOverlayFrame = props.sendOverlayFrame;
  }
  virtual void getMotionVectors(frame_container &frames, frame_sp &outFrame,
                                frame_sp &decodedFrame) = 0;
  virtual void initDecoder() = 0;

public:
  int mWidth = 0;
  int mHeight = 0;
  std::string rawFramePinId;
  std::string motionVectorPinId;
  std::function<frame_sp(frame_sp &bigFrame, size_t &size, string &pinId)>
      makeframe;
  std::function<frame_sp(size_t size, string &pinId)> makeFrameWithPinId;
  bool sendDecodedFrame = false;
  bool sendOverlayFrame = true;
  int threshold;
  cv::Mat bgrImg;
  bool motionFound = false;
};
class DetailFfmpeg : public MvExtractDetailAbs {
public:
  DetailFfmpeg(
      MotionVectorExtractorProps props,
      std::function<frame_sp(size_t size, string &pinId)> _makeFrameWithPinId,
      std::function<frame_sp(frame_sp &bigFrame, size_t &size, string &pinId)>
          _makeframe)
      : MvExtractDetailAbs(props, _makeFrameWithPinId, _makeframe) {}
  ~DetailFfmpeg() { avcodec_free_context(&decoderContext); }
  void getMotionVectors(frame_container &frames, frame_sp &outFrame,
                        frame_sp &decodedFrame);
  void initDecoder();
  int decodeAndGetMotionVectors(AVPacket *pkt, frame_container &frames,
                                frame_sp &outFrame, frame_sp &decodedFrame);

private:
  AVFrame *avFrame = NULL;
  AVCodecContext *decoderContext = NULL;
};
class DetailOpenH264 : public MvExtractDetailAbs {
public:
  DetailOpenH264(
      MotionVectorExtractorProps props,
      std::function<frame_sp(size_t size, string &pinId)> _makeFrameWithPinId,
      std::function<frame_sp(frame_sp &bigFrame, size_t &size, string &pinId)>
          _makeframe)
      : MvExtractDetailAbs(props, _makeFrameWithPinId, _makeframe) {}
  ~DetailOpenH264() {
    if (pDecoder) {
      pDecoder->Uninitialize();
      WelsDestroyDecoder(pDecoder);
    }
  }
  void getMotionVectors(frame_container &frames, frame_sp &outFrame,
                        frame_sp &decodedFrame);
  void initDecoder();

private:
  ISVCDecoder *pDecoder;
  SBufferInfo pDstInfo;
  SDecodingParam sDecParam;
  SParserBsInfo parseInfo;
};
void DetailFfmpeg::initDecoder() {
  int ret;
  AVCodec *dec = NULL;
  AVDictionary *opts = NULL;
  dec = avcodec_find_decoder(AV_CODEC_ID_H264);
  decoderContext = avcodec_alloc_context3(dec);
  if (!decoderContext) {
    throw AIPException(AIP_FATAL, "Failed to allocate codec");
  }
  /* Init the decoder */
  av_dict_set(&opts, "flags2", "+export_mvs", 0);
  ret = avcodec_open2(decoderContext, dec, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    throw AIPException(AIP_FATAL, "failed open decoder");
  }
}
void DetailFfmpeg::getMotionVectors(frame_container &frames, frame_sp &outFrame,
                                    frame_sp &decodedFrame) {
  int ret = 0;
  AVPacket *pkt = NULL;
  avFrame = av_frame_alloc();
  if (!avFrame) {
    LOG_ERROR << "Could not allocate frame\n";
  }
  pkt = av_packet_alloc();
  if (!pkt) {
    LOG_ERROR << "Could not allocate AVPacket\n";
  }
  ret = decodeAndGetMotionVectors(pkt, frames, outFrame, decodedFrame);
  av_packet_free(&pkt);
  av_frame_free(&avFrame);
}
int DetailFfmpeg::decodeAndGetMotionVectors(AVPacket *pkt,
                                            frame_container &frames,
                                            frame_sp &outFrame,
                                            frame_sp &decodedFrame) {
  auto inFrame = frames.begin()->second;
  pkt->data = (uint8_t *)inFrame->data();
  pkt->size = (int)inFrame->size();
  int ret = avcodec_send_packet(decoderContext, pkt);
  if (ret < 0) {
    LOG_ERROR << stderr << "Error while sending a packet to the decoder: %s\n";
    return ret;
  }
  while (ret >= 0) {
    ret = avcodec_receive_frame(decoderContext, avFrame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      outFrame = makeFrameWithPinId(0, motionVectorPinId);
      break;
    } else if (ret < 0) {
      LOG_ERROR << stderr
                << "Error while receiving a frame from the decoder: %s\n";
      return ret;
    }
    if (sendDecodedFrame) {
      SwsContext *sws_context =
          sws_getContext(decoderContext->width, decoderContext->height,
                         decoderContext->pix_fmt, decoderContext->width,
                         decoderContext->height, AV_PIX_FMT_BGR24,
                         SWS_BICUBIC | SWS_FULL_CHR_H_INT, NULL, NULL, NULL);
      if (!sws_context) {
        // Handle error
      }
      decodedFrame = makeFrameWithPinId(mWidth * mHeight * 3, rawFramePinId);
      int dstStrides[AV_NUM_DATA_POINTERS];
      dstStrides[0] = decoderContext->width * 3; // Assuming BGR format
      uint8_t *dstData[AV_NUM_DATA_POINTERS];
      dstData[0] = static_cast<uint8_t *>(decodedFrame->data());
      sws_scale(sws_context, avFrame->data, avFrame->linesize, 0,
                decoderContext->height, dstData, dstStrides);
      frames.insert(make_pair(rawFramePinId, decodedFrame));
    }
    if (ret >= 0) {
      AVFrameSideData *sideData;
      sideData = av_frame_get_side_data(avFrame, AV_FRAME_DATA_MOTION_VECTORS);
      if (sideData) {
        std::vector<LineOverlay> lineOverlays;
        CompositeOverlay compositeOverlay;

        const AVMotionVector *mvs = (const AVMotionVector *)sideData->data;
        for (int i = 0; i < sideData->size / sizeof(*mvs); i++) {
          const AVMotionVector *mv = &mvs[i];

          if (sendOverlayFrame == true && std::abs(mv->motion_x) > threshold ||
              std::abs(mv->motion_y) > threshold) {
            LineOverlay lineOverlay;
            lineOverlay.x1 = mv->src_x;
            lineOverlay.y1 = mv->src_y;
            lineOverlay.x2 = mv->dst_x;
            lineOverlay.y2 = mv->dst_y;

            lineOverlays.push_back(lineOverlay);

            motionFound = true;
          }
        }

        for (auto &lineOverlay : lineOverlays) {
          compositeOverlay.add(&lineOverlay);
        }

        if (lineOverlays.size()) {
          DrawingOverlay drawingOverlay;
          drawingOverlay.add(&compositeOverlay);
          auto serializedSize = drawingOverlay.mGetSerializeSize();
          outFrame = makeFrameWithPinId(serializedSize, motionVectorPinId);
          memcpy(outFrame->data(), sideData->data, serializedSize);
          drawingOverlay.serialize(outFrame);
          frames.insert(make_pair(motionVectorPinId, outFrame));
        }
      } else {
        outFrame = makeFrameWithPinId(0, motionVectorPinId);
      }
      if (sendOverlayFrame == false) {
        const AVMotionVector *mvs = (const AVMotionVector *)sideData->data;
        for (int i = 0; i < sideData->size / sizeof(*mvs); i++) {
          const AVMotionVector *mv = &mvs[i];

          if (std::abs(mv->motion_x) > threshold ||
              std::abs(mv->motion_y) > threshold) {
            motionFound = true;
            break;
          }
        }
      }
      av_packet_unref(pkt);
      av_frame_unref(avFrame);
      return 0;
    }
  }
  return 0;
}

void DetailOpenH264::initDecoder() {
  sDecParam = {0};
  if (!sendDecodedFrame) {
    sDecParam.bParseOnly = true;
  } else {
    sDecParam.bParseOnly = false;
  }
  memset(&pDstInfo, 0, sizeof(SBufferInfo));
  pDstInfo.uiInBsTimeStamp = 0;

  if (WelsCreateDecoder(&pDecoder) || (NULL == pDecoder)) {
    LOG_ERROR << "Create Decoder failed.\n";
  }
  if (pDecoder->Initialize(&sDecParam)) {
    LOG_ERROR << "Decoder initialization failed.\n";
  }
}

void DetailOpenH264::getMotionVectors(frame_container &frames,
                                      frame_sp &outFrame,
                                      frame_sp &decodedFrame) {
  motionFound = false;
  uint8_t *pData[3] = {NULL};
  pData[0] = NULL;
  pData[1] = NULL;
  pData[2] = NULL;
  auto h264Frame = frames.begin()->second;

  unsigned char *pSrc = static_cast<unsigned char *>(h264Frame->data());
  int iSrcLen = h264Frame->size();
  unsigned char **ppDst = pData;
  int32_t mMotionVectorSize = mWidth * mHeight * 8;
  int16_t *mMotionVectorData = nullptr;
  memset(&pDstInfo, 0, sizeof(SBufferInfo));
  if (sendOverlayFrame) {
    outFrame = makeFrameWithPinId(mMotionVectorSize, motionVectorPinId);
    mMotionVectorData = static_cast<int16_t *>(outFrame->data());
  } else {
    mMotionVectorData =
        static_cast<int16_t *>(malloc(mMotionVectorSize * sizeof(int16_t)));
  }

  if (sDecParam.bParseOnly) {
    pDecoder->ParseBitstreamGetMotionVectors(pSrc, iSrcLen, ppDst, &parseInfo,
                                             &pDstInfo, &mMotionVectorSize,
                                             &mMotionVectorData);
  } else {
    pDecoder->DecodeFrameGetMotionVectorsNoDelay(pSrc, iSrcLen, ppDst,
                                                 &pDstInfo, &mMotionVectorSize,
                                                 &mMotionVectorData);
  }

  if (mMotionVectorSize != mWidth * mHeight * 8 && sendOverlayFrame == true) {
    std::vector<CircleOverlay> circleOverlays;
    CompositeOverlay compositeOverlay;

    for (int i = 0; i < mMotionVectorSize; i += 4) {
      auto motionX = mMotionVectorData[i];
      auto motionY = mMotionVectorData[i + 1];
      if (abs(motionX) > threshold || abs(motionY) > threshold) {
        CircleOverlay circleOverlay;
        circleOverlay.x1 = mMotionVectorData[i + 2];
        circleOverlay.y1 = mMotionVectorData[i + 3];
        circleOverlay.radius = 1;

        circleOverlays.push_back(circleOverlay);
        motionFound = true;
      }
    }

    for (auto &circleOverlay : circleOverlays) {
      compositeOverlay.add(&circleOverlay);
    }

    if (circleOverlays.size()) {
      DrawingOverlay drawingOverlay;
      drawingOverlay.add(&compositeOverlay);
      auto mvSize = drawingOverlay.mGetSerializeSize();
      outFrame = makeframe(outFrame, mvSize, motionVectorPinId);
      drawingOverlay.serialize(outFrame);
      frames.insert(make_pair(motionVectorPinId, outFrame));
    }
  }
  if (sendOverlayFrame == false) {
    for (int i = 0; i < mMotionVectorSize; i += 4) {
      auto motionX = mMotionVectorData[i];
      auto motionY = mMotionVectorData[i + 1];
      if (abs(motionX) > threshold || abs(motionY) > threshold) {
        motionFound = true;
        break;
      }
    }
  }
  if ((!sDecParam.bParseOnly) && (pDstInfo.pDst[0] != nullptr) &&
      (mMotionVectorSize != mWidth * mHeight * 8) && motionFound == true) {
    int rowIndex;
    unsigned char *pPtr = NULL;
    decodedFrame = makeFrameWithPinId(mHeight * 3 * mWidth, rawFramePinId);
    uint8_t *yuvImagePtr = (uint8_t *)malloc(mHeight * 1.5 * mWidth);
    auto yuvStartPointer = yuvImagePtr;
    pPtr = pData[0];
    for (rowIndex = 0; rowIndex < mHeight; rowIndex++) {
      memcpy(yuvImagePtr, pPtr, mWidth);
      pPtr += pDstInfo.UsrData.sSystemBuffer.iStride[0];
      yuvImagePtr += mWidth;
    }
    int halfHeight = mHeight / 2;
    int halfWidth = mWidth / 2;
    pPtr = pData[1];
    for (rowIndex = 0; rowIndex < halfHeight; rowIndex++) {
      memcpy(yuvImagePtr, pPtr, halfWidth);
      pPtr += pDstInfo.UsrData.sSystemBuffer.iStride[1];
      yuvImagePtr += halfWidth;
    }
    pPtr = pData[2];
    for (rowIndex = 0; rowIndex < halfHeight; rowIndex++) {
      memcpy(yuvImagePtr, pPtr, halfWidth);
      pPtr += pDstInfo.UsrData.sSystemBuffer.iStride[1];
      yuvImagePtr += halfWidth;
    }

    cv::Mat yuvImgCV = cv::Mat(mHeight + mHeight / 2, mWidth, CV_8UC1,
                               yuvStartPointer, mWidth);
    bgrImg.data = static_cast<uint8_t *>(decodedFrame->data());

    cv::cvtColor(yuvImgCV, bgrImg, cv::COLOR_YUV2BGR_I420);
    frames.insert(make_pair(rawFramePinId, decodedFrame));
  }
}

MotionVectorExtractor::MotionVectorExtractor(MotionVectorExtractorProps props)
    : Module(TRANSFORM, "MotionVectorExtractor", props) {
  if (props.MVExtract == MotionVectorExtractorProps::MVExtractMethod::FFMPEG) {
    mDetail.reset(new DetailFfmpeg(
        props,
        [&](size_t size, string &pinId) -> frame_sp {
          return makeFrame(size, pinId);
        },
        [&](frame_sp &frame, size_t &size, string &pinId) -> frame_sp {
          return makeFrame(frame, size, pinId);
        }));
  } else if (props.MVExtract ==
             MotionVectorExtractorProps::MVExtractMethod::OPENH264) {
    mDetail.reset(new DetailOpenH264(
        props,
        [&](size_t size, string &pinId) -> frame_sp {
          return makeFrame(size, pinId);
        },
        [&](frame_sp &frame, size_t &size, string &pinId) -> frame_sp {
          return makeFrame(frame, size, pinId);
        }));
  }
  if (props.sendOverlayFrame) {
    auto motionVectorOutputMetadata =
        framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));
    mDetail->motionVectorPinId = addOutputPin(motionVectorOutputMetadata);
  }
  rawOutputMetadata = framemetadata_sp(new RawImageMetadata());
  mDetail->rawFramePinId = addOutputPin(rawOutputMetadata);
}
bool MotionVectorExtractor::init() {
  mDetail->initDecoder();
  return Module::init();
}
bool MotionVectorExtractor::term() { return Module::term(); }
bool MotionVectorExtractor::validateInputPins() {
  if (getNumberOfInputPins() != 1) {
    LOG_ERROR << "<" << getId()
              << ">::validateInputPins size is expected to be 1. Actual<"
              << getNumberOfInputPins() << ">";
    return false;
  }
  framemetadata_sp metadata = getFirstInputMetadata();
  FrameMetadata::FrameType frameType = metadata->getFrameType();
  if (frameType != FrameMetadata::H264_DATA) {
    LOG_ERROR << "<" << getId()
              << ">::validateInputPins input frameType is expected to be "
                 "H264_DATA. Actual<"
              << frameType << ">";
    return false;
  }
  return true;
}
bool MotionVectorExtractor::validateOutputPins() {
  auto size = getNumberOfOutputPins();
  if (getNumberOfOutputPins() > 2) {
    LOG_ERROR << "<" << getId()
              << ">::validateOutputPins size is expected to be 2. Actual<"
              << getNumberOfOutputPins() << ">";
    return false;
  }
  pair<string, framefactory_sp> me; // map element
  auto framefactoryByPin = getOutputFrameFactory();
  BOOST_FOREACH (me, framefactoryByPin) {
    FrameMetadata::FrameType frameType =
        me.second->getFrameMetadata()->getFrameType();
    if (frameType != FrameMetadata::OVERLAY_INFO_IMAGE &&
        frameType != FrameMetadata::RAW_IMAGE) {
      LOG_ERROR << "<" << getId()
                << ">::validateOutputPins input frameType is expected to be "
                   "MOTION_VECTOR_DATA or RAW_IMAGE. Actual<"
                << frameType << ">";
      return false;
    }
  }
  return true;
}
bool MotionVectorExtractor::shouldTriggerSOS() { return mShouldTriggerSOS; }
bool MotionVectorExtractor::process(frame_container &frames) {
  frame_sp motionVectorFrame;
  frame_sp decodedFrame;
  mDetail->getMotionVectors(frames, motionVectorFrame, decodedFrame);
  send(frames);
  return true;
}
void MotionVectorExtractor::setMetadata(frame_sp frame) {
  auto metadata = frame->getMetadata();
  if (!metadata->isSet()) {
    return;
  }
  sps_pps_properties p;
  H264ParserUtils::parse_sps(
      ((const char *)frame->data()) + 5,
      frame->size() > 5 ? frame->size() - 5 : frame->size(), &p);
  mDetail->mWidth = p.width;
  mDetail->mHeight = p.height;
  RawImageMetadata outputMetadata(mDetail->mWidth, mDetail->mHeight,
                                  ImageMetadata::BGR, CV_8UC3, 0, CV_8U,
                                  FrameMetadata::HOST, true);
  auto rawOutMetadata =
      FrameMetadataFactory::downcast<RawImageMetadata>(rawOutputMetadata);
  rawOutMetadata->setData(outputMetadata);
  mDetail->bgrImg = Utils::getMatHeader(rawOutMetadata);
}
bool MotionVectorExtractor::processSOS(frame_sp &frame) {
  setMetadata(frame);
  mShouldTriggerSOS = false;
  return true;
}
bool MotionVectorExtractor::handlePropsChange(frame_sp &frame) {
  MotionVectorExtractorProps props;
  auto ret = Module::handlePropsChange(frame, props);
  mDetail->setProps(props);
  return ret;
}
void MotionVectorExtractor::setProps(MotionVectorExtractorProps &props) {
  Module::addPropsToQueue(props);
}