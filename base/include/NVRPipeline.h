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
	NVRPipeline_Detail* mDetail;
};