#include "ColorConversionXForm.h"
#include "H264Decoder.h"
#include "ImageViewerModule.h"
#include "Mp4ReaderSource.h"

#include <PipeLine.h>

class PlayMp4VideoFromBeginning {
public:
  PlayMp4VideoFromBeginning();
  bool setUpPipeLine(const std::string &videoPath);
  bool startPipeLine();
  bool stopPipeLine();
  bool flushQueuesAndSeek();


  boost::shared_ptr<Mp4ReaderSource> mMp4Reader;
  boost::shared_ptr<ImageViewerModule> mImageViewerSink;
  boost::shared_ptr<H264Decoder> mDecoder;
  boost::shared_ptr<ColorConversion> mColorchange;

private:
  PipeLine pipeLine;
};