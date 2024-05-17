#include <PipeLine.h>
#include "Mp4ReaderSource.h"
#include "ImageViewerModule.h"

class PlayMp4VideoFromBeginning {
public:
    PlayMp4VideoFromBeginning(); 
    bool setUpPipeLine();
    bool startPipeLine();
    bool stopPipeLine();
    bool flushQueuesAndSeek();

private:
    PipeLine pipeLine;
    boost::shared_ptr<Mp4ReaderSource> mp4Reader;
    boost::shared_ptr<ImageViewerModule> sink;
};