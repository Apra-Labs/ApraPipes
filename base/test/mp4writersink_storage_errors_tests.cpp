#include <cerrno>
#include <string>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "Logger.h"
#include "FileReaderModule.h"
#include "Mp4WriterSink.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "APErrorObject.h"
#include "test_utils.h"

namespace fs = boost::filesystem;

BOOST_AUTO_TEST_SUITE(mp4WriterSink_storage_errors_tests)

// ─── Mp4WriteHealth::classify ────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(classify_no_space_is_disk_full)
{
	BOOST_TEST((Mp4WriteHealth::classify(ENOSPC, 0) == Mp4WriteFailure::DISK_FULL));
#ifdef EDQUOT
	BOOST_TEST((Mp4WriteHealth::classify(EDQUOT, 0) == Mp4WriteFailure::DISK_FULL));
	BOOST_TEST((Mp4WriteHealth::classify(EIO, EDQUOT) == Mp4WriteFailure::DISK_FULL));
#endif
}

BOOST_AUTO_TEST_CASE(classify_eio_with_enospc_errno_is_disk_full)
{
	// libmp4 reports a failed sample fwrite as EIO; errno still holds ENOSPC.
	BOOST_TEST((Mp4WriteHealth::classify(EIO, ENOSPC) == Mp4WriteFailure::DISK_FULL));
}

BOOST_AUTO_TEST_CASE(classify_real_eio_is_disk_error)
{
	BOOST_TEST((Mp4WriteHealth::classify(EIO, EIO) == Mp4WriteFailure::DISK_ERROR));
	BOOST_TEST((Mp4WriteHealth::classify(EIO, 0) == Mp4WriteFailure::DISK_ERROR));
}

BOOST_AUTO_TEST_CASE(classify_other_storage_failures_are_disk_error)
{
	for (int err : {EROFS, ENOENT, EACCES, ENODEV, ENXIO, EISDIR})
	{
		BOOST_TEST_CONTEXT("errno " << err)
		{
			BOOST_TEST((Mp4WriteHealth::classify(err, err) == Mp4WriteFailure::DISK_ERROR));
		}
	}
}

BOOST_AUTO_TEST_CASE(classify_non_storage_failures_are_ignored)
{
	for (int err : {0, -ENOSPC, ENOMEM, EINVAL, ENOBUFS})
	{
		BOOST_TEST_CONTEXT("errno " << err)
		{
			BOOST_TEST((Mp4WriteHealth::classify(err, 0) == Mp4WriteFailure::NONE));
		}
	}
}

// ─── Mp4WriteHealth state changes ────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(health_raises_once_per_state_change)
{
	Mp4WriteHealth health;
	BOOST_TEST((health.state() == Mp4WriteFailure::NONE));

	BOOST_TEST(health.onFailure(Mp4WriteFailure::NONE) == 0);
	BOOST_TEST(health.onSuccessfulSync() == 0);

	BOOST_TEST(health.onFailure(Mp4WriteFailure::DISK_FULL) == Mp4WriterSinkErrors::DISK_FULL);
	BOOST_TEST(health.onFailure(Mp4WriteFailure::DISK_FULL) == 0);
	BOOST_TEST(health.onFailure(Mp4WriteFailure::NONE) == 0);
	BOOST_TEST((health.state() == Mp4WriteFailure::DISK_FULL));

	BOOST_TEST(health.onFailure(Mp4WriteFailure::DISK_ERROR) == Mp4WriterSinkErrors::DISK_ERROR);
	BOOST_TEST(health.onFailure(Mp4WriteFailure::DISK_ERROR) == 0);
	BOOST_TEST((health.state() == Mp4WriteFailure::DISK_ERROR));

	BOOST_TEST(health.onSuccessfulSync() == Mp4WriterSinkErrors::RECOVERED);
	BOOST_TEST(health.onSuccessfulSync() == 0);
	BOOST_TEST((health.state() == Mp4WriteFailure::NONE));

	BOOST_TEST(health.onFailure(Mp4WriteFailure::DISK_FULL) == Mp4WriterSinkErrors::DISK_FULL);
}

// ─── Mp4WriterSink end to end ────────────────────────────────────────────────
// The output .mp4 is a symlink, so real write failures can be produced without
// root or a special filesystem: /dev/full fails every write with ENOSPC, and a
// symlink into a missing directory fails the open with ENOENT.

#ifdef __linux__

struct StorageErrorFixture
{
	std::vector<APErrorObject> errors;

	void record(const boost::shared_ptr<Module> &module)
	{
		module->registerErrorCallback([this](const APErrorObject &error) { errors.push_back(error); });
	}

	std::vector<int> codes() const
	{
		std::vector<int> result;
		for (auto &error : errors)
		{
			result.push_back(error.getErrorCode());
		}
		return result;
	}

	static void initLogger()
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);
	}

	static fs::path freshDir(const std::string &name)
	{
		fs::path dir = fs::path("data/testOutput/mp4_videos/storage_errors") / name;
		fs::remove_all(dir);
		fs::create_directories(dir);
		return dir;
	}

	static void step(const boost::shared_ptr<Module> &reader, const boost::shared_ptr<Module> &writer, int frames)
	{
		for (int i = 0; i < frames; i++)
		{
			reader->step();
			writer->step();
		}
	}
};

static boost::shared_ptr<Module> h264Reader()
{
	auto props = FileReaderModuleProps("./data/h264_data", 0, -1);
	props.fps = 24;
	props.readLoop = false;
	auto reader = boost::shared_ptr<Module>(new FileReaderModule(props));
	auto metadata = framemetadata_sp(new H264Metadata(704, 576));
	reader->addOutputPin(metadata);
	return reader;
}

static boost::shared_ptr<Module> jpegReader()
{
	auto props = FileReaderModuleProps("./data/re3_filtered", 0, -1);
	props.fps = 24;
	props.readLoop = false;
	auto reader = boost::shared_ptr<Module>(new FileReaderModule(props));
	auto metadata = framemetadata_sp(new EncodedImageMetadata(1280, 720));
	reader->addOutputPin(metadata);
	return reader;
}

BOOST_FIXTURE_TEST_CASE(h264_no_space_raises_disk_full_once, StorageErrorFixture)
{
	if (!fs::exists("/dev/full"))
	{
		BOOST_TEST_MESSAGE("/dev/full not available, skipping");
		return;
	}
	initLogger();
	auto dir = freshDir("disk_full");
	auto outFile = dir / "out.mp4";
	fs::create_symlink("/dev/full", outFile);

	auto reader = h264Reader();
	auto writer = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(Mp4WriterSinkProps(UINT32_MAX, 1, 24, outFile.string())));
	reader->setNext(writer);
	record(writer);

	reader->play(true);
	BOOST_TEST(reader->init());
	BOOST_TEST(writer->init());

	step(reader, writer, 60);
	writer->doMp4MuxSync();
	reader->term();
	writer->term();

	BOOST_TEST(codes() == std::vector<int>({Mp4WriterSinkErrors::DISK_FULL}), boost::test_tools::per_element());
	BOOST_TEST(errors.front().getErrorMessage().find("disk full") != std::string::npos);
	BOOST_TEST(errors.front().getModuleName() == "Mp4WriterSink");

	fs::remove_all(dir);
}

BOOST_FIXTURE_TEST_CASE(jpeg_no_space_raises_disk_full_once, StorageErrorFixture)
{
	if (!fs::exists("/dev/full"))
	{
		BOOST_TEST_MESSAGE("/dev/full not available, skipping");
		return;
	}
	initLogger();
	auto dir = freshDir("disk_full_jpeg");
	auto outFile = dir / "out.mp4";
	fs::create_symlink("/dev/full", outFile);

	auto reader = jpegReader();
	auto writer = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(Mp4WriterSinkProps(UINT32_MAX, 1, 24, outFile.string())));
	reader->setNext(writer);
	record(writer);

	reader->play(true);
	BOOST_TEST(reader->init());
	BOOST_TEST(writer->init());

	step(reader, writer, 20);
	writer->doMp4MuxSync();
	reader->term();
	writer->term();

	BOOST_TEST(codes() == std::vector<int>({Mp4WriterSinkErrors::DISK_FULL}), boost::test_tools::per_element());

	fs::remove_all(dir);
}

BOOST_FIXTURE_TEST_CASE(jpeg_disk_error_then_recovered, StorageErrorFixture)
{
	initLogger();
	auto dir = freshDir("disk_error_jpeg");
	auto outFile = dir / "out.mp4";
	// Dangling: the target directory does not exist yet, so the open fails (ENOENT),
	// like a recording volume that has been unmounted.
	fs::create_symlink("missing/out.mp4", outFile);

	auto reader = jpegReader();
	auto writer = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(Mp4WriterSinkProps(UINT32_MAX, 1, 24, outFile.string())));
	reader->setNext(writer);
	record(writer);

	reader->play(true);
	BOOST_TEST(reader->init());
	BOOST_TEST(writer->init());

	step(reader, writer, 10);
	BOOST_TEST(codes() == std::vector<int>({Mp4WriterSinkErrors::DISK_ERROR}), boost::test_tools::per_element());
	BOOST_TEST(errors.front().getErrorMessage().find("disk error during open") != std::string::npos);

	// Storage comes back: the next frame reopens the file, the next sync confirms it.
	fs::create_directory(dir / "missing");
	step(reader, writer, 10);
	writer->doMp4MuxSync();
	BOOST_TEST(codes() == std::vector<int>({Mp4WriterSinkErrors::DISK_ERROR, Mp4WriterSinkErrors::RECOVERED}), boost::test_tools::per_element());

	// Healthy syncs raise nothing further.
	step(reader, writer, 5);
	writer->doMp4MuxSync();
	reader->term();
	writer->term();
	BOOST_TEST(errors.size() == 2);

	BOOST_TEST(fs::exists(dir / "missing" / "out.mp4"));
	BOOST_TEST(fs::file_size(dir / "missing" / "out.mp4") > 0);

	fs::remove_all(dir);
}

BOOST_FIXTURE_TEST_CASE(h264_failed_open_waits_for_key_frame, StorageErrorFixture)
{
	initLogger();
	auto dir = freshDir("disk_error_h264");
	auto outFile = dir / "out.mp4";
	fs::create_symlink("missing/out.mp4", outFile);

	auto reader = h264Reader();
	auto writer = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(Mp4WriterSinkProps(UINT32_MAX, 1, 24, outFile.string())));
	reader->setNext(writer);
	record(writer);

	reader->play(true);
	BOOST_TEST(reader->init());
	BOOST_TEST(writer->init());

	// The first frame (the only key frame in h264_data) fails to open the file.
	step(reader, writer, 5);
	BOOST_TEST(codes() == std::vector<int>({Mp4WriterSinkErrors::DISK_ERROR}), boost::test_tools::per_element());

	// Storage is back, but the following frames are not key frames: a file that
	// started with them would not decode, so nothing is written.
	fs::create_directory(dir / "missing");
	step(reader, writer, 30);
	writer->doMp4MuxSync();
	reader->term();
	writer->term();

	BOOST_TEST(!fs::exists(dir / "missing" / "out.mp4"));
	BOOST_TEST(codes() == std::vector<int>({Mp4WriterSinkErrors::DISK_ERROR}), boost::test_tools::per_element());

	fs::remove_all(dir);
}

#endif

BOOST_AUTO_TEST_SUITE_END()
