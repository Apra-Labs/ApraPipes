#include "stdafx.h"
#include "Utils.h"
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "FrameFactory.h"
#include "Module.h"
#include "PipeLine.h"
#include "ExtFrame.h"
#include "test_utils.h"

#include <boost/test/unit_test.hpp>

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!

BOOST_AUTO_TEST_SUITE(unit_tests)

BOOST_AUTO_TEST_CASE(dummy_test)
{
	
}

BOOST_AUTO_TEST_CASE(frame_factory_test)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	auto f1 = fact->create(1023, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
	auto f2 = fact->create(1024, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
	auto f3 = fact->create(1025, fact, FrameMetadata::MemType::HOST);//uses 2 chunks
	auto f4 = fact->create(2047, fact, FrameMetadata::MemType::HOST);//uses 2 chunk
	auto f5 = fact->create(100000, fact, FrameMetadata::MemType::HOST); //uses 98 chunk
}

BOOST_AUTO_TEST_CASE(multiple_que_test)
{
	{
		boost::shared_ptr<FrameContainerQueue> q1 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(20));
		boost::shared_ptr<FrameFactory> fact(new FrameFactory);

		{
			auto f = fact->create(1023, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
			frame_container frames;
			frames["a"] = f;
			q1->push(frames);
		}

		

		fact.reset();
		q1->clear();
	}

	boost::this_thread::sleep_for(boost::chrono::milliseconds(500));

}

#ifdef APRA_CUDA_ENABLED

BOOST_AUTO_TEST_CASE(frame_factory_test_host_pinned)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	auto f1 = fact->create(1023, fact, FrameMetadata::MemType::HOST_PINNED);//uses 1 chunk
	auto f2 = fact->create(1024, fact, FrameMetadata::MemType::HOST_PINNED);//uses 1 chunk
	auto f3 = fact->create(1025, fact, FrameMetadata::MemType::HOST_PINNED);//uses 2 chunks
	auto f4 = fact->create(2047, fact, FrameMetadata::MemType::HOST_PINNED);//uses 2 chunk
	auto f5 = fact->create(100000, fact, FrameMetadata::MemType::HOST_PINNED); //uses 98 chunk
}

BOOST_AUTO_TEST_CASE(frame_factory_test_cuda_device)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	auto f1 = fact->create(1023, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 1 chunk
	auto f2 = fact->create(1024, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 1 chunk
	auto f3 = fact->create(1025, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 2 chunks
	auto f4 = fact->create(2047, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 2 chunk
	auto f5 = fact->create(100000, fact, FrameMetadata::MemType::CUDA_DEVICE); //uses 98 chunk
}

#endif

BOOST_AUTO_TEST_CASE(boost_pool_ordered_malloc_free)
{
	static size_t noOfNew = 0;
	static size_t noOfFree = 0;
	static void* myPtr = NULL;

	struct apra_allocator
	{
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;

		static char *malloc(const size_type bytes)
		{
			auto ptr = new (std::nothrow) char[bytes];
			myPtr = ptr;
			noOfNew++;
			return ptr;
		}
		static void free(char *const block)
		{
			noOfFree++;
			BOOST_TEST((void*)block == myPtr);
			delete[] block;
		}
	};

	size_t CHUNK_SZ = 1024;
	boost::pool<apra_allocator> p(CHUNK_SZ);

	size_t noOfChunks = 100;
	size_t totalSize = noOfChunks * CHUNK_SZ;

	for (int i = 0; i < 2; i++)
	{
		unsigned char *buffer = (unsigned char *)p.ordered_malloc(noOfChunks);
		p.ordered_free(buffer, noOfChunks);
	}

	unsigned char *buffer = (unsigned char *)p.ordered_malloc(noOfChunks);
	memset(buffer, 1, totalSize);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{

			if (buffer[i] != 1)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors == 0);
	}
	// calling free for 2nd half of chunks
	p.ordered_free(buffer + (totalSize / 2), noOfChunks / 2);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize / 2; i++)
		{
			if (buffer[i] != 1)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors == 0);
	}

	// asking for half of chunks
	unsigned char *buffer2 = (unsigned char *)p.ordered_malloc(noOfChunks / 2);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize/2; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors == 0);
	}

	memset(buffer2, 2, totalSize / 2);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors == 0);
	}
	// calling free on the buffer2
	p.ordered_free(buffer2, noOfChunks / 2);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize / 2; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors == 0);
	}

	// calling free for 1st half of chunks
	p.ordered_free(buffer, noOfChunks / 2);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors != 0);
	}

	// asking for chunks again and freeing
	unsigned char *buffer3 = (unsigned char *)p.ordered_malloc(noOfChunks);
	memset(buffer3, 3, totalSize);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{

			if (buffer3[i] != 3)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors == 0);            
	}

	p.ordered_free(buffer3, noOfChunks);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{
			if (buffer3[i] != 3)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors != 0);
	}

	unsigned char *buffer4 = (unsigned char *)p.ordered_malloc(noOfChunks);
	p.ordered_free(buffer4, noOfChunks);

	unsigned char *buffer5 = (unsigned char *)p.ordered_malloc(noOfChunks / 2);
	unsigned char *buffer6 = (unsigned char *)p.ordered_malloc(noOfChunks / 2);
	p.ordered_free(buffer5, noOfChunks / 2);
	p.ordered_free(buffer6, noOfChunks / 2);

	unsigned char *buffer7 = (unsigned char *)p.ordered_malloc(noOfChunks);
	p.ordered_free(buffer7, noOfChunks);

	void* secondptr = (void*)(buffer + (totalSize / 2));
	BOOST_TEST((void*)buffer == myPtr);
	BOOST_TEST((void*)buffer2 == secondptr);
	BOOST_TEST((void*)buffer3 == myPtr);
	BOOST_TEST((void*)buffer4 == myPtr);
	BOOST_TEST((void*)buffer5 == myPtr);
	BOOST_TEST((void*)buffer6 == secondptr);
	BOOST_TEST((void*)buffer7 == myPtr);

	p.release_memory();

	BOOST_TEST(noOfNew == 1);
	BOOST_TEST(noOfFree == 1);
}

BOOST_AUTO_TEST_CASE(frame_factory_resize_test)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	{
		auto buffer = fact->createBuffer(5000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(buffer->size() == 5000);

		auto frame = fact->create(buffer, 2000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(frame->size() == 2000);
	}

	{
		auto buffer = fact->createBuffer(5000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(buffer->size() == 5000);

		auto frame = fact->create(buffer, 5000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(frame->size() == 5000);
	}
}

BOOST_AUTO_TEST_CASE(frame_que_test)
{
	FrameContainerQueue q(2);
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	{
		auto f1 = fact->create(1023, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
		auto f2 = fact->create(1024, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
		auto f3 = fact->create(1025, fact, FrameMetadata::MemType::HOST);//uses 2 chunk

		frame_container frames1;
		frame_container frames2;
		frame_container frames3;
		frames1.insert(std::make_pair("p1", f1));

		frames2.insert(std::make_pair("p1", f1));
		frames2.insert(std::make_pair("p2", f2));

		frames3.insert(std::make_pair("p1", f1));
		frames3.insert(std::make_pair("p2", f2));
		frames3.insert(std::make_pair("p3", f3));

		BOOST_TEST(q.try_pop() == frame_container()); //empty queue
		q.push(frames1);
		q.push(frames2);
		BOOST_TEST(q.try_push(frames3) == false);
		auto frames1_ = q.pop();
		BOOST_TEST(frames1.size() == frames1_.size());

		auto frames2_ = q.pop();
		BOOST_TEST(frames2.size() == frames2_.size());

		BOOST_TEST(q.try_pop() == frame_container()); //empty queue
	}
}


BOOST_AUTO_TEST_CASE(base64_encoding_test)
{
	std::string authString1 = "admin:";
	std::string authString2 = "admin:passwd";

	std::string res1 = Utils::base64_encode(reinterpret_cast<const unsigned char*>(authString1.c_str()), authString1.length());
	std::string res2 = Utils::base64_encode(reinterpret_cast<const unsigned char*>(authString2.c_str()), authString2.length());
	BOOST_TEST(res1.compare("YWRtaW46") == 0);
	BOOST_TEST(res2.compare("YWRtaW46cGFzc3dk") == 0);
}

void producer_consumer(bool isProducer, boost::shared_ptr<FrameFactory> fact, FrameContainerQueue& q, int iterations)
{
	for (int i = 0; i < iterations; i++)
	{
		if (isProducer)
		{
			auto f1 = fact->create(1023, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
			frame_container frames;
			frames.insert(std::make_pair("p1", f1));
			q.push(frames);
		}
		else {
			auto f1 = q.pop();
		}
		//		if (i % 1000 == 999)
		//			cout << (isProducer ? "Producer" : "Consumer") << " finished " << std::to_string(i) << " iterations " << endl;
	}
}

BOOST_AUTO_TEST_CASE(two_threaed_framefactory_test)
{
	FrameContainerQueue q(20);
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	std::thread t1(producer_consumer, true, fact, std::ref(q), 100000);
	std::thread t2(producer_consumer, false, fact, std::ref(q), 100000);

	t1.join();
	t2.join();

}

BOOST_AUTO_TEST_CASE(bounded_buffer_1)
{
	bounded_buffer<int> queue(5);
	queue.push(1);
	queue.push(2);
	queue.push(3);
	queue.push(4);
	queue.push(5);
	BOOST_TEST(queue.size() == 5);
	BOOST_TEST(queue.try_push(1) == false);
	BOOST_TEST(queue.size() == 5);
	queue.clear();
	BOOST_TEST(queue.size() == 0);
	queue.push(6);
	BOOST_TEST(queue.size() == 0);
	BOOST_TEST(queue.try_push(1) == false);
	BOOST_TEST(queue.size() == 0);
	queue.accept();
	queue.push(7);
	BOOST_TEST(queue.size() == 1);
}

void testQueueClear(bounded_buffer<int>& queue)
{
	for(auto i = 0; i < 10; i++)
	{
		queue.push(i);
		if (i < 5)
		{
			BOOST_TEST(queue.size() == i + 1);
		}
		else
		{
			// because clear has been called			
			BOOST_TEST(queue.size() == 0);
			BOOST_TEST(queue.try_push(i) == false);
			BOOST_TEST(queue.size() == 0);
		}

		LOG_ERROR << "finished pushing " << i;
	}
}

void testQueueClear2(bounded_buffer<int>& queue)
{
	for(auto i = 0; i < 10; i++)
	{
		auto ret = queue.try_push(i);
		if (i < 5)
		{
			BOOST_TEST(ret);
			BOOST_TEST(queue.size() == i + 1);
		}
		else
		{
			// because queue is full
			BOOST_TEST(!ret);
			BOOST_TEST(queue.size() == 5);
		}

		LOG_ERROR << "finished pushing " << i;
	}
}

BOOST_AUTO_TEST_CASE(bounded_buffer_2)
{
	bounded_buffer<int> queue(5);
	std::thread t1(testQueueClear, std::ref(queue));

	boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));

	// it is expected to be stuck now
	queue.clear();

	t1.join();

	queue.accept();

	t1 = std::thread(testQueueClear2, std::ref(queue));
	t1.join();
}

void testQueuePushPop(bounded_buffer<int>& queue, bool push)
{
	for (int i = 0; i < 10; i++)
	{
		if (push)
		{
			queue.push(i);
		}
		else
		{
			BOOST_TEST(queue.pop() == i);
		}
	}
}

BOOST_AUTO_TEST_CASE(bounded_buffer_3)
{
	bounded_buffer<int> queue(5);
	std::thread t1(testQueuePushPop, std::ref(queue), true);

	boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
	std::thread t2(testQueuePushPop, std::ref(queue), false);

	t1.join();
	t2.join();
}

BOOST_AUTO_TEST_CASE(sendinttest)
{
	int fd = 100;
	ExtFrame frame(&fd, 4);

	auto ptr = static_cast<int*>(frame.data());
	BOOST_TEST(ptr == &fd);
	BOOST_TEST(*ptr == fd);
}

BOOST_AUTO_TEST_CASE(params_test, *boost::unit_test::disabled())
{
	BOOST_TEST(Test_Utils::getArgValue("ip") == "10.102.10.121");
	BOOST_TEST(Test_Utils::getArgValue("ip").length() == 13);
	BOOST_TEST(Test_Utils::getArgValue("IP") == "");
	BOOST_TEST(Test_Utils::getArgValue("data") == "ArgusCamera");
	BOOST_TEST(Test_Utils::getArgValue("some_random_arg_name_not_passed_through_commandline", "hola") == "hola");
}

BOOST_AUTO_TEST_SUITE_END()