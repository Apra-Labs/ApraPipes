#pragma once

#include "Module.h"
#include "ReadBarcode.h"
#include "TextUtfEncoding.h"

class QRReaderProps : public ModuleProps
{
public:
	QRReaderProps(bool _tryHarder = false, bool _saveQRImages = false, string _qrImagesPath = "", int _frameRotationCounter = 10) : ModuleProps() 
	{
		tryHarder = _tryHarder;
		saveQRImages = _saveQRImages;
		qrImagesPath = _qrImagesPath;
		frameRotationCounter = _frameRotationCounter;
	}

	size_t getSerializeSize()
    {
        return ModuleProps::getSerializeSize();
    }
    bool tryHarder;
	bool saveQRImages;
	string qrImagesPath;
	int frameRotationCounter;
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