#include <boost/foreach.hpp>
#include "KeyboardListener.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include <ncurses.h>

class KeyboardListener::Detail
{
public:
	Detail() : mFrameSaved(0), mOpen(false),term(false)
	{
	}

	~Detail()
	{

	}

	void operator()()
	{
        while(!term){
			int k;
			k = getchar();
            if(k == 's'){
				// the below 2 variables are accessed by two different threads - both read and write
                mOpen = true;
                mFrameSaved = 0;
            }
        }
	}

    uint8_t mFrameSaved;
    bool mOpen;
	bool term;
	std::thread mThread;
};

KeyboardListener::KeyboardListener(KeyboardListenerProps _props) :Module(TRANSFORM, "KeyboardListener", _props), props(_props)
{
    mDetail.reset(new Detail());
	mDetail->mThread = std::thread(std::ref(*(mDetail.get())));
}

bool KeyboardListener::validateInputPins()
{
	return true;
}

bool KeyboardListener::validateOutputPins()
{
	return true;
}

bool KeyboardListener::init()
{
	if (!Module::init())
	{
		return false;
	}
	// This will distrub logger
    initscr();

	return true;
}

bool KeyboardListener::term()
{
	mDetail->term = false;
	mDetail->mThread.join();
    endwin();
	return Module::term();
}

void KeyboardListener::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	addOutputPin(metadata, pinId);
}

bool KeyboardListener::process(frame_container& frames)
{
    if(mDetail->mOpen && (mDetail->mFrameSaved < props.nosFrame)){
        mDetail->mFrameSaved++;
        if(mDetail->mFrameSaved == props.nosFrame){
            mDetail->mOpen = false;
        }
        send(frames);
    }

	return true;
}