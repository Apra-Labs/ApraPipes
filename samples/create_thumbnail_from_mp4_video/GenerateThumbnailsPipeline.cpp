
#include "GenerateThumbnailsPipeline.h"
#include "CudaMemCopy.h"
#include "ExternalSinkModule.h"
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "FrameMetadata.h"
#include "H264Decoder.h"
#include "H264Metadata.h"
#include "JPEGEncoderNVJPEG.h"
#include "Logger.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "ValveModule.h"
#include <boost/test/unit_test.hpp>

GenerateThumbnailsPipeline::GenerateThumbnailsPipeline()
    : pipeLine("thumnailSamplePipeline") {}

bool GenerateThumbnailsPipeline::setUpPipeLine(
    const std::string &videoPath, const std::string &outFolderPath) {
  // Implementation

  bool parseFS = false;
  auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
  auto frameType = FrameMetadata::FrameType::H264_DATA;
  auto mp4ReaderProps =
      Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
  //initializing source Mp4 reader to read Mp4 video 
  mMp4Reader =
      boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
  mMp4Reader->addOutPutPin(h264ImageMetadata);

  auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
  mMp4Reader->addOutPutPin(mp4Metadata);

  std::vector<std::string> mImagePin;
  mImagePin = mMp4Reader->getAllOutputPinsByType(frameType);

  //initializing H264 decoder to decode frame in H264 format
  mDecoder =
      boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
  //Selecting an image pin of H264 data frame type is necessary because the decoder processes H264 frames for decoding.
  mMp4Reader->setNext(mDecoder, mImagePin);

  //Initializing the valve to send only one frame. It is currently set to 0, meaning no frames are captured. 
  mValve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(0)));
  mDecoder->setNext(mValve);

  //initialize cuda memory
  auto stream = cudastream_sp(new ApraCudaStream);
  mCudaCopy = boost::shared_ptr<CudaMemCopy>(
      new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
  mValve->setNext(mCudaCopy);

  //initializing Jpeg encoder to encode the frame in jpeg format
  mJpegEncoder = boost::shared_ptr<JPEGEncoderNVJPEG>(
      new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
  mCudaCopy->setNext(mJpegEncoder);

  //initilizing file writer as sink to write frame at given path
  mFileWriter = boost::shared_ptr<FileWriterModule>(
      new FileWriterModule(FileWriterModuleProps(outFolderPath)));
  mJpegEncoder->setNext(mFileWriter);

  return true;
}

bool GenerateThumbnailsPipeline::startPipeLine() {
  pipeLine.appendModule(mMp4Reader);
  if (!pipeLine.init()) {
    throw AIPException(
        AIP_FATAL,
        "Engine Pipeline init failed. Check IPEngine Logs for more details.");
    return false;
  }
  pipeLine.run_all_threaded();
  //allowing only one frame to get captured.
  mValve->allowFrames(1);

  return true;
}

bool GenerateThumbnailsPipeline::stopPipeLine() {
  pipeLine.stop();
  pipeLine.term();
  pipeLine.wait_for_all();
  return true;
}
