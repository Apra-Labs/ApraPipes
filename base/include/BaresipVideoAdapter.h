#include <boost/thread.hpp>

//Baresip Video Adapter code
class BaresipVideoAdapterProps 
{
public:
	BaresipVideoAdapterProps()
	{

	}

 
};


class BaresipVideoAdapter 
{
public:
	BaresipVideoAdapter(BaresipVideoAdapterProps _props);
	~BaresipVideoAdapter();
	bool init(); //runs re_main on a thread
	bool term();
	bool process(void *frame_data);
	void operator()(); //to support boost::thread
	
private:
	//struct tmr tmr_quit;
	int err;
	boost::thread myThread;
	struct vidsrc *myVidsrc;
	
};