#include <PipeLine.h>

class FaceDetectionCPU {
public:
    FaceDetectionCPU(); 
    bool setupPipeline();
    bool startPipeline();
    bool stopPipeline();
private:
    PipeLine pipeline;
};