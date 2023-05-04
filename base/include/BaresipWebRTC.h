//Baresip BaresipWebRTC Adapter code
class BaresipWebRTCProps 
{
public:
	BaresipWebRTCProps()
	{

	}


};

class BaresipWebRTC 
{
public:
	BaresipWebRTC(BaresipWebRTCProps _props);
	~BaresipWebRTC();
	bool init(int argc, char* argv[]);
	bool term();
	bool process();
	bool processSOS();
	void setProps(BaresipWebRTCProps& props);
	void close();
};