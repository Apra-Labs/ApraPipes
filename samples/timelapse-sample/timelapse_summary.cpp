#include "timelapse_summary.h"
#include "ColorConversionXForm.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "ExternalSinkModule.h"
#include "FileWriterModule.h"
#include "H264EncoderNVCodec.h"
#include "H264Metadata.h"
#include "ImageViewerModule.h"
#include "MotionVectorExtractor.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "Mp4WriterSink.h"
#include "OverlayModule.h"
#include "PipeLine.h"
#include <boost/test/unit_test.hpp>

TimelapsePipeline::TimelapsePipeline()
    : timelapseSamplePipeline("test"), mCudaStream_(new ApraCudaStream()),
      mCuContext(new ApraCUcontext()),
      mH264ImageMetadata(new H264Metadata(0, 0)) {}

bool TimelapsePipeline::setupPipeline(const std::string &videoPath,
                                      const std::string &outFolderPath) {
  uint32_t gopLength = 25;
  uint32_t bitRateKbps = 1000;
  uint32_t frameRate = 30;
  H264EncoderNVCodecProps::H264CodecProfile profile =
      H264EncoderNVCodecProps::MAIN;
  bool enableBFrames = false;
  bool sendDecodedFrames = true;

  auto mp4ReaderProps =
      Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
  mp4ReaderProps.parseFS = true;
  mp4ReaderProps.readLoop = false;
  // mp4Reader module is being used here to read the .mp4 videos
  mMp4Reader =
      boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
  auto motionExtractorProps = MotionVectorExtractorProps(
      MotionVectorExtractorProps::MVExtractMethod::OPENH264, sendDecodedFrames,
      2);
  // motionVectorExtractor module is being used to get the frames for the
  // defined thershold
  mMotionExtractor = boost::shared_ptr<MotionVectorExtractor>(
      new MotionVectorExtractor(motionExtractorProps));
  // convert frames from BGR to RGB
  mColorchange1 = boost::shared_ptr<ColorConversion>(new ColorConversion(
      ColorConversionProps(ColorConversionProps::BGR_TO_RGB)));
  // convert frames from RGB to YUV420PLANAR
  // the two step color change is done because H264Encoder takes YUV data
  mColorchange2 = boost::shared_ptr<ColorConversion>(new ColorConversion(
      ColorConversionProps(ColorConversionProps::RGB_TO_YUV420PLANAR)));
  mSync = boost::shared_ptr<Module>(
      new CudaStreamSynchronize(CudaStreamSynchronizeProps(mCudaStream_)));
  mEncoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(
      H264EncoderNVCodecProps(bitRateKbps, mCuContext, gopLength, frameRate,
                              profile, enableBFrames)));
  // write the output video
  auto mp4WriterSinkProps =
      Mp4WriterSinkProps(UINT32_MAX, 10, 24, outFolderPath, true);
  mp4WriterSinkProps.recordedTSBasedDTS = false;
  mMp4WriterSink =
      boost::shared_ptr<Module>(new Mp4WriterSink(mp4WriterSinkProps));

  mMp4Reader->addOutPutPin(mH264ImageMetadata);
  auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
  mMp4Reader->addOutPutPin(mp4Metadata);
  std::vector<std::string> mImagePin =
      mMp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);
  std::vector<std::string> mDecodedPin =
      mMotionExtractor->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE);
  mMp4Reader->setNext(mMotionExtractor, mImagePin);
  mMotionExtractor->setNext(mColorchange1, mDecodedPin);
  mColorchange1->setNext(mColorchange2);
  mCopy = boost::shared_ptr<Module>(
      new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, mCudaStream_)));
  mColorchange2->setNext(mCopy);
  mCopy->setNext(mSync);
  mSync->setNext(mEncoder);
  mEncoder->setNext(mMp4WriterSink);

  timelapseSamplePipeline.appendModule(mMp4Reader);
  if (timelapseSamplePipeline.init()) {
    return true;
  }
  return false;
}

bool TimelapsePipeline::startPipeline() {
  timelapseSamplePipeline.run_all_threaded();
  return true;
}

bool TimelapsePipeline::stopPipeline() {
  timelapseSamplePipeline.stop();
  timelapseSamplePipeline.term();
  timelapseSamplePipeline.wait_for_all();
  return true;
}