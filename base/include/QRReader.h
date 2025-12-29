#pragma once

#include "Module.h"
#include "ZXing/ReadBarcode.h"
#include "ZXing/TextUtfEncoding.h"
#include <array>
#include "declarative/Metadata.h"

class QRReaderProps : public ModuleProps
{
public:
	QRReaderProps() : ModuleProps() {}
};

class QRReader : public Module
{

public:
	// ============================================================
	// Declarative Pipeline Metadata
	// ============================================================
	struct Metadata {
		static constexpr std::string_view name = "QRReader";
		static constexpr apra::ModuleCategory category = apra::ModuleCategory::Analytics;
		static constexpr std::string_view version = "1.0.0";
		static constexpr std::string_view description =
			"Reads and decodes QR codes and barcodes from image frames using ZXing library.";

		static constexpr std::array<std::string_view, 4> tags = {
			"analytics", "qr", "barcode", "reader"
		};

		static constexpr std::array<apra::PinDef, 1> inputs = {
			apra::PinDef::create("input", "RawImagePlanar", true, "Image frames to scan for QR codes")
		};

		static constexpr std::array<apra::PinDef, 1> outputs = {
			apra::PinDef::create("output", "Frame", true, "Frames with QR code metadata")
		};

		// No configurable properties
		static constexpr std::array<apra::PropDef, 0> properties = {};
	};

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