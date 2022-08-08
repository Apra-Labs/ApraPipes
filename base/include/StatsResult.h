#pragma once

#include <boost/serialization/vector.hpp>
#include "Frame.h"
#include "FrameMetadata.h"

class StatsResult : public FrameMetadata
{
public:
	StatsResult(uint8_t _min, uint8_t _max, uint8_t _saturationPercentage) : FrameMetadata(FrameType::STATS)
	{
		min = _min;
        max = _max;
        saturationPercentage = _saturationPercentage;
	}

	StatsResult() : FrameMetadata(FrameType::STATS)
	{
		min = 255;
		max = 0;
        saturationPercentage = 0;
	}

	static void serialize(StatsResult &result, frame_sp &frame)
	{
		boost::iostreams::basic_array_sink<char> device_sink((char *)frame->data(), frame->size());
		boost::iostreams::stream<boost::iostreams::basic_array_sink<char>> s_sink(device_sink);

		boost::archive::binary_oarchive oa(s_sink);
		oa << result;
	}


	static void deSerialize(StatsResult &result, frame_sp &frame)
	{
		boost::iostreams::basic_array_source<char> device((char *)frame->data(), frame->size());
		boost::iostreams::stream<boost::iostreams::basic_array_source<char>> s(device);
		boost::archive::binary_iarchive ia(s);

		ia >> result;
	}

    uint8_t min;
    uint8_t max;
    uint8_t saturationPercentage;

	size_t getSerializeSize()
	{
		return sizeof(min)
		 + sizeof(max) 
		 + sizeof(saturationPercentage) 
		 + 1024;
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &min;
		ar &max;
		ar &saturationPercentage;
	}
};