#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <functional>

typedef void(*MessageHandler)();

namespace Mp4PlayerInternal {
	class Mp4Pipeline_Detail;
	class Mp4Pipeline
	{
	public:
		Mp4Pipeline();
		~Mp4Pipeline();
		bool close();
		bool pause();
		bool resume();
		bool nextFrame();
		void popFromSinkQIfFull();
		bool RegisterMetadataListener();
		std::tuple<uint8_t*, std::string> peekSinkQ(uint64_t& frameSize);
		bool seek(uint64_t skipTS, std::string skipDir);
		void SetDataListener(MessageHandler dataHandler);
		std::tuple<bool, std::string >open(std::string videoPath);
		void getResolution(uint32_t &width, uint32_t &height);
	private:
		Mp4Pipeline_Detail *mDetail;
		MessageHandler mDataHandler;
	};
}