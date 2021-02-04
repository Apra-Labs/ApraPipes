#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"

#include "FrameMetadata.h"
#include "Frame.h"
#include "FramesMuxer.h"

BOOST_AUTO_TEST_SUITE(framesmuxer_tests)

void testFrames(frame_container& frames, size_t fIndex, size_t size)
{
	BOOST_TEST(frames.size() == size);
	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		BOOST_TEST(fIndex == it->second->fIndex);
	}
}

BOOST_AUTO_TEST_CASE(allornone)
{
	size_t readDataSize = 1024;

	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin1_1 = m1->addOutputPin(metadata);
	auto pin1_2 = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin2_1 = m2->addOutputPin(metadata);

	auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin3_1 = m3->addOutputPin(metadata);

	auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
	m1->setNext(muxer);
	m2->setNext(muxer);
	m3->setNext(muxer);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	muxer->setNext(sink);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	BOOST_TEST(muxer->init());
	BOOST_TEST(sink->init());

	{

		{
			// basic

			auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
			encodedImageFrame->fIndex = 500;

			frame_container frames;
			frames.insert(make_pair(pin1_1, encodedImageFrame));
			frames.insert(make_pair(pin1_2, encodedImageFrame));
			frames.insert(make_pair(pin2_1, encodedImageFrame));
			frames.insert(make_pair(pin3_1, encodedImageFrame));


			m1->send(frames);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			m2->send(frames);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			m3->send(frames);
			muxer->step();
			auto outFrames = sink->try_pop();
			testFrames(outFrames, 500, 4);
		}

		{
			// pin1_2 comes late

			auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
			encodedImageFrame->fIndex = 600;

			frame_container frames;

			frames.insert(make_pair(pin2_1, encodedImageFrame));
			frames.insert(make_pair(pin3_1, encodedImageFrame));

			m2->send(frames);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			m3->send(frames);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			auto encodedImageFrame2 = m1->makeFrame(readDataSize, metadata);
			encodedImageFrame2->fIndex = 600;

			frame_container frames2;
			frames2.insert(make_pair(pin1_1, encodedImageFrame2));
			frames2.insert(make_pair(pin1_2, encodedImageFrame2));
			m1->send(frames2);
			muxer->step();
			auto outFrames = sink->try_pop();
			testFrames(outFrames, 600, 4);
		}

		{
			{
				// frame drop

				for (auto i = 0; i < 5; i++)
				{
					auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
					encodedImageFrame->fIndex = 701 + i;

					frame_container frames;

					frames.insert(make_pair(pin2_1, encodedImageFrame));
					frames.insert(make_pair(pin3_1, encodedImageFrame));

					m2->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);

					m3->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);
				}

				auto encodedImageFrame2 = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame2->fIndex = 706;

				frame_container frames2;
				frames2.insert(make_pair(pin1_1, encodedImageFrame2));
				frames2.insert(make_pair(pin1_2, encodedImageFrame2));
				m1->send(frames2);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);

				{
					auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
					encodedImageFrame->fIndex = 706;

					frame_container frames;
					frames.insert(make_pair(pin2_1, encodedImageFrame));
					frames.insert(make_pair(pin3_1, encodedImageFrame));

					m2->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);

					m3->send(frames);
					muxer->step();
					auto outFrames = sink->try_pop();
					testFrames(outFrames, 706, 4);
				}

				{
					// basic

					auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
					encodedImageFrame->fIndex = 732;

					frame_container frames;
					frames.insert(make_pair(pin1_1, encodedImageFrame));
					frames.insert(make_pair(pin1_2, encodedImageFrame));
					frames.insert(make_pair(pin2_1, encodedImageFrame));
					frames.insert(make_pair(pin3_1, encodedImageFrame));


					m1->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);

					m2->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);

					m3->send(frames);
					muxer->step();
					auto outFrames = sink->try_pop();
					testFrames(outFrames, 732, 4);
				}
			}
		}

		{
			// maxDelay

			for (auto i = 0; i < 30; i++)
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 801 + i;

				frame_container frames;

				frames.insert(make_pair(pin2_1, encodedImageFrame));
				frames.insert(make_pair(pin3_1, encodedImageFrame));

				m2->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);

				m3->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);
			}

			auto encodedImageFrame2 = m1->makeFrame(readDataSize, metadata);
			encodedImageFrame2->fIndex = 831;

			frame_container frames2;
			frames2.insert(make_pair(pin1_1, encodedImageFrame2));
			frames2.insert(make_pair(pin1_2, encodedImageFrame2));
			m1->send(frames2);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 831;

				frame_container frames;
				frames.insert(make_pair(pin2_1, encodedImageFrame));
				frames.insert(make_pair(pin3_1, encodedImageFrame));

				m2->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);

				m3->send(frames);
				muxer->step();
				auto outFrames = sink->try_pop();
				testFrames(outFrames, 831, 4);
			}

			{
				// basic

				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 832;

				frame_container frames;
				frames.insert(make_pair(pin1_1, encodedImageFrame));
				frames.insert(make_pair(pin1_2, encodedImageFrame));
				frames.insert(make_pair(pin2_1, encodedImageFrame));
				frames.insert(make_pair(pin3_1, encodedImageFrame));


				m1->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);

				m2->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);

				m3->send(frames);
				muxer->step();
				auto outFrames = sink->try_pop();
				testFrames(outFrames, 832, 4);
			}
		}
	}

	BOOST_TEST(m1->term());
	BOOST_TEST(m2->term());
	BOOST_TEST(m3->term());
	BOOST_TEST(muxer->term());
	BOOST_TEST(sink->term());

	m1.reset();
	m2.reset();
	m3.reset();
	muxer.reset();
	sink.reset();
}



BOOST_AUTO_TEST_CASE(maxdelaystrategy_1)
{
	size_t readDataSize = 1024;

	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin1_1 = m1->addOutputPin(metadata);
	auto pin1_2 = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin2_1 = m2->addOutputPin(metadata);

	auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin3_1 = m3->addOutputPin(metadata);


	int maxDelay = 10;
	FramesMuxerProps props;
	props.strategy = FramesMuxerProps::MAX_DELAY_ANY;
	props.maxDelay = maxDelay;
	auto muxer = boost::shared_ptr<Module>(new FramesMuxer(props));
	m1->setNext(muxer);
	m2->setNext(muxer);
	m3->setNext(muxer);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	muxer->setNext(sink);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	BOOST_TEST(muxer->init());
	BOOST_TEST(sink->init());

	{

		{
			// basic

			auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
			encodedImageFrame->fIndex = 500;

			frame_container frames;
			frames.insert(make_pair(pin1_1, encodedImageFrame));
			frames.insert(make_pair(pin1_2, encodedImageFrame));
			frames.insert(make_pair(pin2_1, encodedImageFrame));
			frames.insert(make_pair(pin3_1, encodedImageFrame));


			m1->send(frames);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			m2->send(frames);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			m3->send(frames);
			muxer->step();
			auto outFrames = sink->try_pop();
			testFrames(outFrames, 500, 4);
		}


		{
			// send only m1 to the muxer
			// expectation is for 10 times no output								
			for (auto i = 0; i < maxDelay + 1; i++)
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 700 + i;

				frame_container frames;
				frames.insert(make_pair(pin1_1, encodedImageFrame));
				frames.insert(make_pair(pin1_2, encodedImageFrame));

				m1->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);
			}

			// from 11 th time - we expect the frames to come
			for (auto i = maxDelay + 1; i < maxDelay + 50; i++)
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 700 + i;

				frame_container frames;
				frames.insert(make_pair(pin1_1, encodedImageFrame));
				frames.insert(make_pair(pin1_2, encodedImageFrame));

				m1->send(frames);
				muxer->step();
				auto outFrames = sink->try_pop();
				testFrames(outFrames, 700 + i - (maxDelay + 1), 2);
			}


		}
	}

	BOOST_TEST(m1->term());
	BOOST_TEST(m2->term());
	BOOST_TEST(m3->term());
	BOOST_TEST(muxer->term());
	BOOST_TEST(sink->term());

	m1.reset();
	m2.reset();
	m3.reset();
	muxer.reset();
	sink.reset();
}


BOOST_AUTO_TEST_CASE(maxdelaystrategy_2)
{
	size_t readDataSize = 1024;

	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin1_1 = m1->addOutputPin(metadata);
	auto pin1_2 = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin2_1 = m2->addOutputPin(metadata);

	auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin3_1 = m3->addOutputPin(metadata);


	int maxDelay = 10;
	FramesMuxerProps props;
	props.strategy = FramesMuxerProps::MAX_DELAY_ANY;
	props.maxDelay = maxDelay;
	auto muxer = boost::shared_ptr<Module>(new FramesMuxer(props));
	m1->setNext(muxer);
	m2->setNext(muxer);
	m3->setNext(muxer);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	muxer->setNext(sink);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	BOOST_TEST(muxer->init());
	BOOST_TEST(sink->init());

	{

		{

			// image, change_result, blob_result
			// assume only image, change_result is sent for some reason
			// 11 th time we send the image, we expect both image, change_result frame with index 0 to come
			// 12 th time we send the image, we expect both image, change_result frame with index 1 to come


			// send only m1, m2 to the muxer
			// expectation is for 10 times no output								
			for (auto i = 0; i < (maxDelay + 1); i++)
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 800 + i;

				frame_container frames;
				frames.insert(make_pair(pin1_1, encodedImageFrame));
				frames.insert(make_pair(pin1_2, encodedImageFrame));
				frames.insert(make_pair(pin2_1, encodedImageFrame));

				m1->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);
				m2->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);
			}

			// from 11 th time - we expect the m1, m2 frames to come
			for (auto i = maxDelay + 1; i < maxDelay + 50; i++)
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 800 + i;

				frame_container frames;
				frames.insert(make_pair(pin1_1, encodedImageFrame));
				frames.insert(make_pair(pin1_2, encodedImageFrame));
				frames.insert(make_pair(pin2_1, encodedImageFrame));

				{
					m1->send(frames);
					muxer->step();
					auto outFrames = sink->try_pop();
					testFrames(outFrames, 800 + i - (maxDelay + 1), 3);
				}
				{
					m2->send(frames);
					muxer->step();
					auto outFrames = sink->try_pop();
					BOOST_TEST(sink->try_pop().size() == 0);
				}
			}


		}

	}

	BOOST_TEST(m1->term());
	BOOST_TEST(m2->term());
	BOOST_TEST(m3->term());
	BOOST_TEST(muxer->term());
	BOOST_TEST(sink->term());

	m1.reset();
	m2.reset();
	m3.reset();
	muxer.reset();
	sink.reset();
}


BOOST_AUTO_TEST_CASE(maxdelaystrategy_3)
{
	size_t readDataSize = 1024;

	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin1_1 = m1->addOutputPin(metadata);
	auto pin1_2 = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin2_1 = m2->addOutputPin(metadata);

	auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin3_1 = m3->addOutputPin(metadata);


	int maxDelay = 10;
	FramesMuxerProps props;
	props.strategy = FramesMuxerProps::MAX_DELAY_ANY;
	props.maxDelay = maxDelay;
	auto muxer = boost::shared_ptr<Module>(new FramesMuxer(props));
	m1->setNext(muxer);
	m2->setNext(muxer);
	m3->setNext(muxer);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	muxer->setNext(sink);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	BOOST_TEST(muxer->init());
	BOOST_TEST(sink->init());

	{

		{
			// send only m1 to the muxer
			// expectation is for 5 times no output
			auto noFramesCount = maxDelay / 2;
			for (auto i = 0; i < noFramesCount; i++)
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 600 + i;

				frame_container frames;
				frames.insert(make_pair(pin1_1, encodedImageFrame));
				frames.insert(make_pair(pin1_2, encodedImageFrame));

				m1->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);
			}

			// from 6 th time - we expect the frames to come
			for (auto i = noFramesCount; i < maxDelay + 50; i++)
			{
				{
					auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
					encodedImageFrame->fIndex = 600 + i;

					frame_container frames;

					frames.insert(make_pair(pin1_1, encodedImageFrame));
					frames.insert(make_pair(pin1_2, encodedImageFrame));

					m1->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);
				}

				{
					auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
					encodedImageFrame->fIndex = 600 + i - noFramesCount;

					frame_container frames;

					frames.insert(make_pair(pin2_1, encodedImageFrame));
					frames.insert(make_pair(pin3_1, encodedImageFrame));

					m2->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);

					m3->send(frames);
					muxer->step();
					auto outFrames = sink->try_pop();
					testFrames(outFrames, 600 + i - noFramesCount, 4);
				}

			}


		}
	}

	BOOST_TEST(m1->term());
	BOOST_TEST(m2->term());
	BOOST_TEST(m3->term());
	BOOST_TEST(muxer->term());
	BOOST_TEST(sink->term());

	m1.reset();
	m2.reset();
	m3.reset();
	muxer.reset();
	sink.reset();
}

BOOST_AUTO_TEST_CASE(maxdelaystrategy_drop_flush)
{
	size_t readDataSize = 1024;

	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin1_1 = m1->addOutputPin(metadata);
	auto pin1_2 = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin2_1 = m2->addOutputPin(metadata);

	auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin3_1 = m3->addOutputPin(metadata);


	int maxDelay = 10;
	FramesMuxerProps props;
	props.strategy = FramesMuxerProps::MAX_DELAY_ANY;
	props.maxDelay = maxDelay;
	auto muxer = boost::shared_ptr<Module>(new FramesMuxer(props));
	m1->setNext(muxer);
	m2->setNext(muxer);
	m3->setNext(muxer);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	muxer->setNext(sink);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	BOOST_TEST(muxer->init());
	BOOST_TEST(sink->init());

	{

		{
			// send only m1 to the muxer
			// expectation is for 5 times no output
			auto noFramesCount = 5;
			for (auto i = 0; i < noFramesCount; i++)
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = i;

				frame_container frames;
				frames.insert(make_pair(pin1_1, encodedImageFrame));
				frames.insert(make_pair(pin1_2, encodedImageFrame));

				m1->send(frames);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);
			}

			// now directly frame index 3 is sent
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 3;

				frame_container frames;

				frames.insert(make_pair(pin2_1, encodedImageFrame));

				m2->send(frames);
				muxer->step();
				auto outFrames = sink->try_pop();
				// now the old frames of m1 are flushed
				testFrames(outFrames, 0, 2);
				outFrames = sink->try_pop();
				testFrames(outFrames, 1, 2);
				outFrames = sink->try_pop();
				testFrames(outFrames, 2, 2);
			}

			{
				// old frame index 0 is sent
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 0;

				frame_container frames;

				frames.insert(make_pair(pin3_1, encodedImageFrame));

				m3->send(frames);
				muxer->step();
				auto outFrames = sink->try_pop();
				// only the m3 frame 0 is sent
				testFrames(outFrames, 0, 1);
			}

			// now directly frame index 4 is sent
			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 4;

				frame_container frames;

				frames.insert(make_pair(pin3_1, encodedImageFrame));

				m3->send(frames);
				muxer->step();
				auto outFrames = sink->try_pop();
				// now the old frames of m1, m2 index 3 is flushed
				testFrames(outFrames, 3, 3);
			}

			{
				// frame index 4 is now sent
				auto encodedImageFrame = m1->makeFrame(readDataSize, metadata);
				encodedImageFrame->fIndex = 4;

				frame_container frames;

				frames.insert(make_pair(pin2_1, encodedImageFrame));

				m2->send(frames);
				muxer->step();
				auto outFrames = sink->try_pop();
				testFrames(outFrames, 4, 4);
			}

		}
	}

	BOOST_TEST(m1->term());
	BOOST_TEST(m2->term());
	BOOST_TEST(m3->term());
	BOOST_TEST(muxer->term());
	BOOST_TEST(sink->term());

	m1.reset();
	m2.reset();
	m3.reset();
	muxer.reset();
	sink.reset();
}

BOOST_AUTO_TEST_SUITE_END()