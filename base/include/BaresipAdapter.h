//Baresip Adapter code
class BaresipAdapterProps 
{
public:
	BaresipAdapterProps()
	{

	}


};

class BaresipAdapter 
{
public:
	BaresipAdapter(BaresipAdapterProps _props);
	~BaresipAdapter();
	bool init(int argc, char* argv[]);
	bool term();
	bool process();
	bool processSOS();
	void setProps(BaresipAdapterProps& props);
	void close();
};