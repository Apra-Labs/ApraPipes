#include "stdafx.h"

#include "CudaMemCopy.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "CudaCommon.h"

#include <boost/test/unit_test.hpp>

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!


BOOST_AUTO_TEST_SUITE(cudamemcopy_tests)

BOOST_AUTO_TEST_CASE(isCudaSupported)
{
	std::cout << CudaUtils::isCudaSupported() << std::endl;
}

BOOST_AUTO_TEST_CASE(general)
{
	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto pinId = source->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	source->setNext(copy1);

	auto copy2 = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	copy1->setNext(copy2);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	auto readDataSize = 1000;
	auto frame = source->makeFrame(readDataSize, metadata);
	memset(frame->data(), 5, readDataSize);

	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	source->send(frames);
	copy1->step();
	copy2->step();
	frames = sink->try_pop();
	auto frameCopy = frames.cbegin()->second;
	BOOST_TEST(frame->size() == frameCopy->size());
	BOOST_TEST(memcmp(frame->data(), frameCopy->data(), frame->size()) == 0);
}

BOOST_AUTO_TEST_CASE(rawimage)
{
	int width = 1920;
	int height = 1080;
	int channels = 3;
	size_t step = width*channels;
	int type = CV_8UC3;
	int depth = CV_8U;

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, type, 0, depth, FrameMetadata::HOST, true));
	auto pinId = source->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);	
	auto copy1 = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	source->setNext(copy1);

	auto copy2 = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	copy1->setNext(copy2);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	auto readDataSize = metadata->getDataSize();
	auto frame = source->makeFrame(readDataSize, metadata);
	memset(frame->data(), 5, readDataSize);

	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	source->send(frames);
	copy1->step();
	copy2->step();
	frames = sink->try_pop();
	auto frameCopy = frames.cbegin()->second;
	BOOST_TEST(frame->size() == frameCopy->size());
	BOOST_TEST(memcmp(frame->data(), frameCopy->data(), frame->size()) == 0);

	BOOST_TEST(metadata->getMemType() == FrameMetadata::MemType::HOST);
	auto metadata2 = frameCopy->getMetadata();
	auto ptr = FrameMetadataFactory::downcast<RawImageMetadata>(metadata2);
	BOOST_TEST(ptr->getMemType() = FrameMetadata::MemType::CUDA_DEVICE);
	BOOST_TEST(ptr->getWidth() == width);
	BOOST_TEST(ptr->getHeight() == height);
	BOOST_TEST(ptr->getChannels() == channels);
	BOOST_TEST(ptr->getStep() == step);
	BOOST_TEST(ptr->getType() == type);
	BOOST_TEST(ptr->getDataSize() == step * height);
}

BOOST_AUTO_TEST_CASE(rawimageplanar)
{
	int width = 1920;
	int height = 1080;
	size_t step[4] = { 1920, 960, 960 };
	int channels = 3;
	int depth = CV_8U;

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::YUV420, size_t(0), depth));
	auto pinId = source->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	source->setNext(copy1);

	auto copy2 = boost::shared_ptr<CudaMemCopy>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	copy1->setNext(copy2);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	auto readDataSize = metadata->getDataSize(); 
	auto frame = source->makeFrame(readDataSize, metadata);
	memset(frame->data(), 5, readDataSize);

	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	source->send(frames);
	copy1->step();
	copy2->step();
	frames = sink->try_pop();
	auto frameCopy = frames.cbegin()->second;
	BOOST_TEST(frame->size() == frameCopy->size());
	BOOST_TEST(memcmp(frame->data(), frameCopy->data(), frame->size()) == 0);

	BOOST_TEST(metadata->getMemType() == FrameMetadata::MemType::HOST);
	auto metadata2 = frameCopy->getMetadata();
	auto ptr = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata2);
	BOOST_TEST(ptr->getMemType() = FrameMetadata::MemType::CUDA_DEVICE);
	BOOST_TEST(ptr->getWidth(0) == width);
	BOOST_TEST(ptr->getHeight(0) == height);
	BOOST_TEST(ptr->getWidth(1) == width / 2);
	BOOST_TEST(ptr->getHeight(1) == height / 2);
	BOOST_TEST(ptr->getWidth(2) == width / 2);
	BOOST_TEST(ptr->getHeight(2) == height / 2);
	BOOST_TEST(ptr->getChannels() == channels);
	BOOST_TEST(ptr->getStep(0) == step[0]);
	BOOST_TEST(ptr->getStep(1) == step[1]);
	BOOST_TEST(ptr->getStep(2) == step[2]);
	BOOST_TEST(ptr->getDataSize() == step[0] * height * 1.5);
}

BOOST_AUTO_TEST_SUITE_END()