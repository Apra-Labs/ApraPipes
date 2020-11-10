#pragma once

#include "FrameMetadata.h"

class ArrayMetadata: public FrameMetadata
{
public:
    ArrayMetadata(): FrameMetadata(FrameType::ARRAY) {}
	ArrayMetadata(std::string _hint): FrameMetadata(FrameType::ARRAY, _hint) {}
	ArrayMetadata(MemType _memType): FrameMetadata(FrameType::ARRAY, _memType) {}

    void reset()
	{
		FrameMetadata::reset();

		// ARRAY
		length = NOT_SET_NUM;
		type = NOT_SET_NUM;
		elemSize = NOT_SET_NUM;
	}

    bool isSet() 
	{
        return length != NOT_SET_NUM;
    }

    void setData(int len, int _type, size_t _elemSize)
	{
		length = len;
		type = _type;
		elemSize = _elemSize;		
		dataSize = length * elemSize;
	}

	int getType() { return type; }
	int getLength() { return length; }
	size_t getElemSize() { return elemSize; }

protected:    
	int length = NOT_SET_NUM;
	int type = NOT_SET_NUM;
	size_t elemSize = NOT_SET_NUM;
};