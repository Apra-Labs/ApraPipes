#include "FrameFactory.h"
#include "Module.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(frame_factory_test_dma)
#ifdef ARM64

BOOST_AUTO_TEST_CASE(frame_factory_test_dmabuf)
{
	framemetadata_sp metadata(new RawImageMetadata(640,480,ImageMetadata::RGBA,CV_8UC4,0,CV_8U,FrameMetadata::MemType::DMABUF));
	boost::shared_ptr<FrameFactory> fact(new FrameFactory(metadata));
	auto f1 = fact->create(1228800, fact);//uses 1 chunk size of metadata is 921600
	auto f2 = fact->create(1228800, fact);//uses 1 chunk
	auto f3 = fact->create(1228800, fact);//uses 1 chunks
}

#endif

BOOST_AUTO_TEST_SUITE_END()