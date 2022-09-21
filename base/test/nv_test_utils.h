#pragma once
#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "H264EncoderNVCodecHelper.h"

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;
//preempt test failure if the platform does not support H264 encode
struct if_h264_encoder_supported{
  tt::assertion_result operator()(utf::test_unit_id)
  {
	try{
		auto cuContext = apracucontext_sp(new ApraCUcontext());
		H264EncoderNVCodecHelper h(1000, cuContext, 30, 30, H264EncoderNVCodecProps::BASELINE, false);
		
	}
	catch(AIP_Exception& ex)
	{
		LOG_ERROR << ex.what();
		LOG_ERROR << "skipping tests";
		return false;
	}
	return true;
  }
};

struct if_compute_cap_supported{
  tt::assertion_result operator()(utf::test_unit_id)
  {
	try{
        Logger::setLogLevel("info");
		auto cuContext = apracucontext_sp(new ApraCUcontext());
		int major=0,minor=0;
		if(!cuContext->getComputeCapability(major,minor))
            return false;
		LOG_INFO << "Compute Cap "<<major <<"."<<minor;
	}
	catch(AIP_Exception& ex)
	{
		LOG_ERROR << ex.what();
		LOG_ERROR << "skipping tests";
		return false;
	}
	return true;
  }
};
