#pragma once
class BufferMaker{
public:
    virtual void * make(size_t dataSize)=0;
};