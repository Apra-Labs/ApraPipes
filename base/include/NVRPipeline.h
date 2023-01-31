#pragma once

class NVRPipeline_Detail;
class NVRPipeline
{
public:
	NVRPipeline();

	bool open();
	bool close();
	bool pause();
	bool resume();
	bool startView();
	bool stopView();
	bool xport(uint64_t TS, uint64_t TE);

	NVRPipeline_Detail* mDetail;
};