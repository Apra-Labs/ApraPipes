#pragma once

#include "Module.h"
#include "ReadBarcode.h"

class QRReaderProps : public ModuleProps
{
public:
	QRReaderProps() : ModuleProps() {}
};

class QRReader : public Module
{

public:
	QRReader(QRReaderProps _props=QRReaderProps());
	virtual ~QRReader();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	bool shouldTriggerSOS();

private:		
	class Detail;
	boost::shared_ptr<Detail> mDetail;	
};