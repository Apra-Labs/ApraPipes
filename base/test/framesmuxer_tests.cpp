#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <chrono>

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

			auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

			auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

			auto encodedImageFrame2 = m1->makeFrame(readDataSize, pin1_1);
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
					auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

				auto encodedImageFrame2 = m1->makeFrame(readDataSize, pin1_1);
				encodedImageFrame2->fIndex = 706;

				frame_container frames2;
				frames2.insert(make_pair(pin1_1, encodedImageFrame2));
				frames2.insert(make_pair(pin1_2, encodedImageFrame2));
				m1->send(frames2);
				muxer->step();
				BOOST_TEST(sink->try_pop().size() == 0);

				{
					auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

					auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

			auto encodedImageFrame2 = m1->makeFrame(readDataSize, pin1_1);
			encodedImageFrame2->fIndex = 831;

			frame_container frames2;
			frames2.insert(make_pair(pin1_1, encodedImageFrame2));
			frames2.insert(make_pair(pin1_2, encodedImageFrame2));
			m1->send(frames2);
			muxer->step();
			BOOST_TEST(sink->try_pop().size() == 0);

			{
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

BOOST_AUTO_TEST_CASE(allornone_sampling_by_drops) {
	size_t readDataSize = 1024;

	auto metadata =
		framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin1_1 = m1->addOutputPin(metadata);
	auto pin1_2 = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin2_1 = m2->addOutputPin(metadata);

	auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto pin3_1 = m3->addOutputPin(metadata);

	// drop 3 out of 4 frames
	auto muxProps = new FramesMuxerProps();
	muxProps->skipN = 3;
	muxProps->skipD = 4;

	auto muxer = boost::shared_ptr<Module>(new FramesMuxer(*muxProps));
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
		// basic
		int countMuxOutput = 0;
		int iterations = 4000;

		auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
		for (int i = 0; i < iterations; ++i)
		{
			encodedImageFrame->fIndex = 500 + i;

			frame_container frames;
			frames.insert(make_pair(pin1_1, encodedImageFrame));
			frames.insert(make_pair(pin1_2, encodedImageFrame));
			frames.insert(make_pair(pin2_1, encodedImageFrame));
			frames.insert(make_pair(pin3_1, encodedImageFrame));

			m1->send(frames);
			muxer->step();
			if ((i + 1) % 3 != 0)
				BOOST_TEST(sink->try_pop().size() == 0);

			m2->send(frames);
			muxer->step();
			if ((i + 1) % 3 != 0)
				BOOST_TEST(sink->try_pop().size() == 0);

			m3->send(frames);
			muxer->step();
			auto outFrames = sink->try_pop();
			LOG_ERROR << "outFrames <" << outFrames.size() << ">";
			if (outFrames.size())
			{
				countMuxOutput += 1;
				testFrames(outFrames, 500 + i, 4);
			}
		}
		LOG_ERROR << "COUNT MUX FRAMES OUTPUT <" << countMuxOutput << "> / " << iterations << ".";
		float passThroughRatio = (muxProps->skipD - muxProps->skipN) / (float)muxProps->skipD;
		BOOST_TEST(countMuxOutput == (iterations * passThroughRatio));

	}
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

			auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
					auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
					encodedImageFrame->fIndex = 600 + i;

					frame_container frames;

					frames.insert(make_pair(pin1_1, encodedImageFrame));
					frames.insert(make_pair(pin1_2, encodedImageFrame));

					m1->send(frames);
					muxer->step();
					BOOST_TEST(sink->try_pop().size() == 0);
				}

				{
					auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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
				auto encodedImageFrame = m1->makeFrame(readDataSize, pin1_1);
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

BOOST_AUTO_TEST_CASE(timestamp_strategy)
{
    size_t readDataSize = 1024;

    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

    auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin1_1 = m1->addOutputPin(metadata);

    auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin2_1 = m2->addOutputPin(metadata);

    auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin3_1 = m3->addOutputPin(metadata);

    FramesMuxerProps props;
    props.strategy = FramesMuxerProps::MAX_TIMESTAMP_DELAY;
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
        // Send frames with the same timestamp
        auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();

        auto encodedImageFrame1 = m1->makeFrame(readDataSize, pin1_1);
        encodedImageFrame1->fIndex = 100;
        encodedImageFrame1->timestamp = timestamp;

        auto encodedImageFrame2 = m2->makeFrame(readDataSize, pin2_1);
        encodedImageFrame2->fIndex = 100;
        encodedImageFrame2->timestamp = timestamp;

        auto encodedImageFrame3 = m3->makeFrame(readDataSize, pin3_1);
        encodedImageFrame3->fIndex = 100;
        encodedImageFrame3->timestamp = timestamp;

        frame_container frames1;
        frames1.insert(make_pair(pin1_1, encodedImageFrame1));
        m1->send(frames1);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frame_container frames2;
        frames2.insert(make_pair(pin2_1, encodedImageFrame2));
        m2->send(frames2);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frame_container frames3;
        frames3.insert(make_pair(pin3_1, encodedImageFrame3));
        m3->send(frames3);
        muxer->step();
        auto outFrames = sink->try_pop();
		BOOST_TEST(outFrames.size() == 3);

        // Send frames with different timestamps
        auto newTimestamp = timestamp + std::chrono::milliseconds(100).count();

        auto encodedImageFrame4 = m1->makeFrame(readDataSize, pin1_1);
        encodedImageFrame4->fIndex = 200;
        encodedImageFrame4->timestamp = newTimestamp;

        auto encodedImageFrame5 = m2->makeFrame(readDataSize, pin2_1);
        encodedImageFrame5->fIndex = 200;
        encodedImageFrame5->timestamp = newTimestamp;

        auto encodedImageFrame6 = m3->makeFrame(readDataSize, pin3_1);
        encodedImageFrame6->fIndex = 200;
        encodedImageFrame6->timestamp = newTimestamp;

        frames1.clear();
        frames1.insert(make_pair(pin1_1, encodedImageFrame4));
        m1->send(frames1);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frames2.clear();
        frames2.insert(make_pair(pin2_1, encodedImageFrame5));
        m2->send(frames2);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frames3.clear();
        frames3.insert(make_pair(pin3_1, encodedImageFrame6));
        m3->send(frames3);
        muxer->step();
        outFrames = sink->try_pop();
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

BOOST_AUTO_TEST_CASE(timestamp_strategy_with_maxdelay)
{
    size_t readDataSize = 1024;

    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

    auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin1_1 = m1->addOutputPin(metadata);

    auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin2_1 = m2->addOutputPin(metadata);

    auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin3_1 = m3->addOutputPin(metadata);

    FramesMuxerProps props;
    props.strategy = FramesMuxerProps::MAX_TIMESTAMP_DELAY;
    props.maxDelay = 100; // Max delay in milliseconds
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
        // Send frames with timestamps differing by more than maxDelay
        auto baseTimestamp = std::chrono::system_clock::now().time_since_epoch().count();

        auto encodedImageFrame1 = m1->makeFrame(readDataSize, pin1_1);
        encodedImageFrame1->fIndex = 300;
        encodedImageFrame1->timestamp = baseTimestamp;

        auto encodedImageFrame2 = m2->makeFrame(readDataSize, pin2_1);
        encodedImageFrame2->fIndex = 300;
        encodedImageFrame2->timestamp = baseTimestamp + std::chrono::milliseconds(150).count(); // 150 ms difference

        auto encodedImageFrame3 = m3->makeFrame(readDataSize, pin3_1);
        encodedImageFrame3->fIndex = 300;
        encodedImageFrame3->timestamp = baseTimestamp;

        frame_container frames1;
        frames1.insert(make_pair(pin1_1, encodedImageFrame1));
        m1->send(frames1);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frame_container frames2;
        frames2.insert(make_pair(pin2_1, encodedImageFrame2));
        m2->send(frames2);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frame_container frames3;
        frames3.insert(make_pair(pin3_1, encodedImageFrame3));
        m3->send(frames3);
        muxer->step();
        auto outFrames = sink->try_pop();
        // Frames should not be muxed because of maxDelay
        BOOST_TEST(outFrames.size() == 0);
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

BOOST_AUTO_TEST_CASE(timestamp_strategy_with_negativeTImeDelay) // This test was added to test How Muxer Behave When A Frame From relative past was pushed into the queue 
{
    size_t readDataSize = 1024;

    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

    auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin1_1 = m1->addOutputPin(metadata);

    auto m2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin2_1 = m2->addOutputPin(metadata);

    auto m3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto pin3_1 = m3->addOutputPin(metadata);

    FramesMuxerProps props;
    props.strategy = FramesMuxerProps::MAX_TIMESTAMP_DELAY;
    props.maxTsDelayInMS = 100; // Max delay in milliseconds
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
        // Send frames with timestamps differing by more than maxDelay
        auto baseTimestamp = std::chrono::system_clock::now().time_since_epoch().count();

        auto encodedImageFrame1 = m1->makeFrame(readDataSize, pin1_1);
        encodedImageFrame1->fIndex = 300;
        encodedImageFrame1->timestamp = baseTimestamp;

        auto encodedImageFrame2 = m2->makeFrame(readDataSize, pin2_1);
        encodedImageFrame2->fIndex = 300;
        encodedImageFrame2->timestamp = baseTimestamp + std::chrono::milliseconds(150).count(); // 150 ms difference

        auto encodedImageFrame3 = m3->makeFrame(readDataSize, pin3_1);
        encodedImageFrame3->fIndex = 300;
        encodedImageFrame3->timestamp = baseTimestamp - std::chrono::milliseconds(150).count();

        frame_container frames1;
        frames1.insert(make_pair(pin1_1, encodedImageFrame1));
        m1->send(frames1);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frame_container frames2;
        frames2.insert(make_pair(pin2_1, encodedImageFrame2));
        m2->send(frames2);
        muxer->step();
        BOOST_TEST(sink->try_pop().size() == 0);

        frame_container frames3;
        frames3.insert(make_pair(pin3_1, encodedImageFrame3));
        m3->send(frames3);
        muxer->step();
        auto outFrames = sink->try_pop();
        // Frames should not be muxed because of maxDelay
        BOOST_TEST(outFrames.size() == 0);
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

// To Do:- 
// 1. Need to have Public Function To Get Curr State of Muxer
// 2. Need to add more strategies
// 3. Need to add more tests
// 4. Need to add proper documentation for each of this strategy 

BOOST_AUTO_TEST_SUITE_END()