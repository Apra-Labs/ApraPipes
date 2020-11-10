#pragma once

/**
* locked != 0-> in use by Framework
* ApraData should  be destroyed only when locked == 0
* buffer should not be modified when locked != 0 
*/
class ApraData
{
public:	
	ApraData(void* _buffer, size_t _size, uint64_t _fIndex)
	{
		buffer = _buffer;
		size = _size;
		fIndex = _fIndex;
		locked = 0;
	}

	~ApraData() 
	{

	}

	size_t getLocked()
	{
		return locked;
	}

	friend class ExternalFrame;

private:
	void* buffer;
	size_t size;
	uint64_t fIndex;
	atomic_uint locked;
};