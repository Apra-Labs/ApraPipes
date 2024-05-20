
#include <PipeLine.h>

class TimelapsePipeline {
public:
  TimelapsePipeline();
  bool setupPipeline();
  bool startPipeline();
  bool stopPipeline();

private:
  PipeLine pipeline;
};