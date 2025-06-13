#include "ImageViewerModule.h"
#include "KeyboardListener.h"
#include "Mp4VideoMetadata.h"
#include <ColorConversionXForm.h>
#include <PipeLine.h>

#include "ImageViewerModule.h"
#include "RTSPClientSrc.h"
#include <H264Decoder.h>
#include <ColorConversionXForm.h>
#include <Mp4ReaderSource.h>

class RelayPipeline {
public:
  RelayPipeline();

  boost::shared_ptr<RTSPClientSrc> rtspSource;
  boost::shared_ptr<Mp4ReaderSource> mp4ReaderSource;
  boost::shared_ptr<H264Decoder> h264Decoder;
  boost::shared_ptr<ColorConversion> colorConversion;
  boost::shared_ptr<ImageViewerModule> imageViewer;

  bool setupPipeline(const std::string &rtspUrl, const std::string &mp4VideoPath);
  bool startPipeline();
  bool stopPipeline();
  void addRelayToRtsp(bool open);
  void addRelayToMp4(bool open);

private:
  PipeLine pipeline;
};