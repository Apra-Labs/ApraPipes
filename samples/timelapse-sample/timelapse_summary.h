
#include <PipeLine.h>
#include <CudaMemCopy.h>
#include "CudaStreamSynchronize.h"
#include <Mp4ReaderSource.h>
#include <MotionVectorExtractor.h>
#include <ColorConversionXForm.h>

class TimelapsePipeline {
public:
  TimelapsePipeline();
  bool setupPipeline(const std::string &videoPath,
                     const std::string &outFolderPath);
  bool startPipeline();
  bool stopPipeline();
  void test();

private:
  PipeLine pipeline;
  cudastream_sp cudaStream_;
  apracucontext_sp cuContext;
  framemetadata_sp h264ImageMetadata;
  boost::shared_ptr<Mp4ReaderSource> mp4Reader;
  boost::shared_ptr<MotionVectorExtractor> motionExtractor;
  boost::shared_ptr<ColorConversion> colorchange1;
  boost::shared_ptr<ColorConversion> colorchange2;
  boost::shared_ptr<Module> sync;
  boost::shared_ptr<Module> encoder;
  boost::shared_ptr<Module> mp4WriterSink;
};