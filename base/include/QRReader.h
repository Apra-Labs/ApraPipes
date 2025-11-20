#pragma once

#include "Module.h"
#include "ZXing/ReadBarcode.h"
#include "ZXing/TextUtfEncoding.h"

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
	bool init() override;
	bool term() override;

protected:
	bool process(frame_container& frames) override;
	bool processSOS(frame_sp& frame) override;
	bool validateInputPins() override;
	bool validateOutputPins() override;
	bool shouldTriggerSOS() override;

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};