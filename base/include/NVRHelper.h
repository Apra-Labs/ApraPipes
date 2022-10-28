#pragma once
#include "Module.h"

class NVRCommand
{

};
class RecordCommand : public NVRCommand
{
public:
    RecordCommand(bool _doRecord)
    {
        doRecord = _doRecord;
    }
    bool doRecord = false;
};
class ExportCommand : public NVRCommand
{
public:
    ExportCommand(uint64_t _startTime, uint64_t _stopTime)
    {
        startTime = _startTime;
        stopTime = _stopTime;
    }
    uint64_t startTime = 0;
    uint64_t stopTime = 0;
};