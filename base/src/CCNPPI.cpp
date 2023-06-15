#include "CCNPPI.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CCKernel.h"
#include "npp.h"

class CCNPPI::Detail
{
public:
	Detail(CCNPPIProps& _props) : props(_props)
	{
		nppStreamCtx.hStream = props.stream;
	}

	~Detail() {}

	//This enum has to match ImageMetadata enums
	enum Imageformats
	{
		UNSET = 0,
		MONO = 1,
		BGR,
		BGRA,
		RGB,
		RGBA,
		YUV411_I = 10,
		YUV444,
		YUV420,
		UYVY,
		YUYV,
		NV12,
		BAYERBG10 = 20,
		BAYERBG8,
		BAYERGB8,
		BAYERGR8,
		BAYERRG8
	};

	const int enumSize = BAYERRG8 + 1; // last enum value of Imageformats plus 1

	const int conversionTable[37][2] =
	{
		{1, 2},   // MONO to BGR
		{1, 3},   //MONO to BGRA
		{1, 4},   // MONO to RGB
		{1, 5},   // MONO to RGBA
		{1, 12},  // MONO to YUV420
		{2, 1},   // BGR to MONO
		{2, 3},   // BGR to BGRA
		{2, 4},   // BGR to RGB
		{2, 5},   // BGR to RGBA
		{2, 12},  // BGR to YUV420
		{3, 1},   // BGRA to MONO
		{3, 2},   // BGRA to BGR
		{3, 4},   // BGRA to RGB
		{3, 5},   // BGRA to RGBA
		{3, 12},  // BGRA to YUV420
		{4, 1},   // RGB to MONO
		{4, 2},   // RGB to BGR
		{4, 5},   // RGB to RGBA
		{4,3},    // RGB to BGRA
		{4, 12},  // RGB to YUV420
		{5, 1},   // RGBA to MONO
		{5, 2},   // RGBA to BGR
		{5, 3},   // RGBA to BGRA
		{5, 4},   // RGBA to RGB
		{5, 12},  // RGBA to YUV420
		{12, 1},  // YUV420 to MONO
		{12, 2},  // YUV420 to BGR
		{12, 3},  // YUV420 to BGRA
		{12, 4},  // YUV420 to RGB
		{12, 5},  // YUV420 to RGBA
		{15, 1},  // NV12 to MONO
		{15, 2},  // NV12 to BGR
		{15, 3}, // Nv12 to BGRA
		{15, 4},  // NV12 to RGB
		{15, 5}, // NV12 to RGBA
		{15, 12},  // NV12 to YUV420
	    {10, 11 }  // yuv411 to YUV444
	};

	int convmatrix[BAYERRG8 + 1][BAYERRG8 + 1][2] = { {-1} };

	void setConvMatrix() {

		for (int i = 0; i < enumSize; i++) {
			for (int j = 0; j < enumSize; j++) {
				if (i == j) {
					convmatrix[i][j][0] = 0;
					convmatrix[i][j][1] = 0;
				}
				else {
					convmatrix[i][j][0] = -1;
					convmatrix[i][j][1] = -1;
				}
			}
		}
		for (int k = 0; k < 37; k++) {
			int i = conversionTable[k][0];
			int j = conversionTable[k][1];
			convmatrix[i][j][0] = conversionTable[k][0];
			convmatrix[i][j][1] = conversionTable[k][1];
		}
	}
	
	bool convertMONOtoRGB()
	{
		auto status = nppiDup_8u_C1C3R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiDup_8u_C1C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertMONOtoBGR()
	{
		auto status = nppiDup_8u_C1C3R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			nppStreamCtx);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiDup_8u_C1C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertMONOtoRGBA()
	{
		auto status = nppiDup_8u_C1C4R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			nppStreamCtx);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiDup_8u_C1C4R_Ctx failed<" << status << ">";
			return false;
		}

		const Npp8u nValue = 255; // Alpha value
		status = nppiSet_8u_C4CR_Ctx(nValue,
			dst[0] + 3,
			dstPitch[0],
			dstSize[0],
			nppStreamCtx
		);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSet_8u_C4CR_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertMONOtoBGRA() {
		auto status = nppiDup_8u_C1C4R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			nppStreamCtx
		);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiCopy_8u_C1C4R_Ctx failed<" << status << ">";
			return false;
		}

		status = nppiSet_8u_C4CR_Ctx(255,
			dst[0] + 3,
			dstPitch[0],
			dstSize[0],
			nppStreamCtx
		);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSet_8u_C4CR_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertMONOtoYUV420()
	{
		// CUDA MEMCPY Y
		auto cudaStatus = cudaMemcpy2DAsync(dst[0], dstPitch[0], src[0], srcPitch[0], srcRowSize[0], srcSize[0].height, cudaMemcpyDeviceToDevice, props.stream);
		// CUDA MEMSET U V

		if (cudaStatus != cudaSuccess)
		{
			LOG_ERROR << "copy failed<" << cudaStatus << ">";
			return false;
		}

		cudaStatus = cudaMemset2DAsync(dst[1],
			dstPitch[1],
			128,
			dstSize[1].width,
			dstSize[0].height,
			props.stream);

		if (cudaStatus != cudaSuccess)
		{
			LOG_ERROR << "cudaMemset2DAsync failed<" << cudaStatus << ">";
			return false;
		}
		return true;

	}

	bool convertRGBtoMONO()
	{
		auto status = nppiRGBToGray_8u_C3C1R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			nppStreamCtx);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiRGBToGray_8u_C3C1R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBtoBGR()
	{
		const int dstOrder[3] = { 2, 1, 0 }; // Channel order for RGB to BGR conversion

		auto status = nppiSwapChannels_8u_C3R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			dstOrder,
			nppStreamCtx);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBtoBGRA(bool intermediate)
	{
		const Npp8u nValue = 255; // Alpha value
		int dstOrder[4] = { 2, 1, 0, 3 }; // Channel order for RGB to BGRA conversion

		if (intermediate)
		{
			src[0] = intermediatedst[0];
			srcPitch[0] = intermediatePitch[0];
			srcSize[0] = intermediateSize[0];
		}

		auto status = nppiSwapChannels_8u_C3C4R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nValue, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C3C4R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBtoRGBA(bool intermediate)
	{
		const Npp8u nValue = 255; // Alpha value
		int dstOrder[4] = { 0, 1, 2, 3 };

		if (intermediate)
		{
			src[0] = intermediatedst[0];
			srcPitch[0] = intermediatePitch[0];
			srcSize[0] = intermediateSize[0];
		}

		auto status = nppiSwapChannels_8u_C3C4R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nValue, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C3C4R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBtoYUV420()
	{
		auto status = nppiRGBToYUV420_8u_C3P3R_Ctx(src[0],
			srcPitch[0],
			dst,
			dstPitch,
			srcSize[0],
			nppStreamCtx
		);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiRGBToYUV420_8u_C3P3R_Ctx failed<" << status << ">";
		}

		return true;
	}

	bool convertBGRtoMONO()
	{
		auto status = nppiRGBToGray_8u_C3C1R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			nppStreamCtx);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiRGBToGray_8u_C3C1R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertBGRtoRGB()
	{
		const int dstOrder[3] = { 2, 1, 0 }; // Channel order for BGR to RGB conversion

		auto status = nppiSwapChannels_8u_C3R_Ctx(src[0],
			srcPitch[0],
			dst[0],
			dstPitch[0],
			srcSize[0],
			dstOrder,
			nppStreamCtx);
		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertBGRtoRGBA()
	{
		const Npp8u nValue = 255; // Alpha value
		int dstOrder[4] = { 2, 1, 0, 3 }; // Channel order for BGR to RGBA conversion

		auto status = nppiSwapChannels_8u_C3C4R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nValue, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C3C4R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertBGRtoBGRA()
	{
		const Npp8u nValue = 255; // Alpha value
		int dstOrder[4] = { 0, 1, 2, 3 }; // Channel order for BGR to BGRA conversion

		auto status = nppiSwapChannels_8u_C3C4R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nValue, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C3C4R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertBGRtoYUV420()
	{
		auto status = nppiRGBToYUV420_8u_C3P3R_Ctx(src[0],
			srcPitch[0],
			dst,
			dstPitch,
			srcSize[0],
			nppStreamCtx
		);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiRGBToYUV420_8u_C3P3R_Ctx failed<" << status << ">";
		}

		return true;
	}

	bool convertRGBAtoMONO()
	{
		auto status = nppiRGBToGray_8u_AC4C1R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiRGBToGray_8u_AC4C1R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBAtoRGB()
	{
		const int dstOrder[3] = { 0, 1, 2 }; // RGB channel order

		auto status = nppiSwapChannels_8u_C4C3R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C4C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBAtoBGR()
	{
		const int dstOrder[3] = { 2, 1, 0 }; // BGR channel order

		auto status = nppiSwapChannels_8u_C4C3R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C4C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBAtoBGRA()
	{
		const int dstOrder[4] = { 2, 1, 0, 3 }; // BGRA channel order

		auto status = nppiSwapChannels_8u_C4R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C4R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertRGBAtoYUV420()
	{
		auto status = nppiBGRToYUV420_8u_AC4P3R_Ctx(src[0],
			srcPitch[0],
			dst,
			dstPitch,
			srcSize[0],
			nppStreamCtx
		);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiRGBAToYUV420_8u_AC4P3R_Ctx failed<" << status << ">";
		}
		return true;
	}

	bool convertBGRAtoMONO()
	{
		auto status = nppiRGBToGray_8u_AC4C1R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiRGBToGray_8u_AC4C1R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertBGRAtoRGB()
	{
		const int dstOrder[3] = { 2, 1, 0 }; // BGR channel order

		auto status = nppiSwapChannels_8u_C4C3R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C4C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;

	}

	bool convertBGRAtoBGR()
	{
		const int dstOrder[3] = { 2,1,0 }; // BGR channel order

		auto status = nppiSwapChannels_8u_C4C3R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C4C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertBGRAtoRGBA()
	{
		const int dstOrder[4] = { 2, 1, 0, 3 }; // RGBA channel order

		auto status = nppiSwapChannels_8u_C4R_Ctx(src[0], srcPitch[0], dst[0], dstPitch[0], srcSize[0], dstOrder, nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiSwapChannels_8u_C4R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertBGRAtoYUV420()
	{
		auto status = nppiBGRToYUV420_8u_AC4P3R_Ctx(src[0],
			srcPitch[0],
			dst,
			dstPitch,
			srcSize[0],
			nppStreamCtx
		);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiBGRToYUV420_8u_AC4P3R_Ctx failed<" << status << ">";
		}

		return true;
	}

	bool convertYUV420toMONO()
	{
		// CUDA MEMCPY Y
		auto cudaStatus = cudaMemcpy2DAsync(dst[0], dstPitch[0], src[0], srcPitch[0], srcRowSize[0], srcSize[0].height, cudaMemcpyDeviceToDevice, props.stream);

		if (cudaStatus != cudaSuccess)
		{
			LOG_ERROR << "copy failed<" << cudaStatus << ">";
			return false;
		}
		return true;
	}

	bool convertYUV420toRGB()
	{
		auto status = nppiYUV420ToRGB_8u_P3C3R_Ctx(src, srcPitch, dst[0], dstPitch[0], srcSize[0], nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiYUV420ToRGB_8u_P3C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertYUV420toBGR()
	{
		auto status = nppiYUV420ToBGR_8u_P3C3R_Ctx(src, srcPitch, dst[0], dstPitch[0], srcSize[0], nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiYUV420ToBGR_8u_P3C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;

	}

	bool convertYUV420toRGBA()
	{
		auto status = nppiYUV420ToRGB_8u_P3C4R_Ctx(src, srcPitch, dst[0], dstPitch[0], srcSize[0], nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiYUV420ToRGB_8u_P3C4R_Ctx failed<" << status << ">";
			return false;
		}

		return true;

	}

	bool convertYUV420toBGRA()
	{
		auto status = nppiYUV420ToBGR_8u_P3C4R_Ctx(src,
			srcPitch,
			dst[0],
			dstPitch[0],
			srcSize[0],
			nppStreamCtx
		);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiYUV420ToBGR_8u_P3C4R_Ctx failed<" << status << ">";
		}
		return true;
	}

	bool convertNV12toMONO()
	{
		// CUDA MEMCPY Y
		auto cudaStatus = cudaMemcpy2DAsync(dst[0], dstPitch[0], src[0], srcPitch[0], srcRowSize[0], srcSize[0].height, cudaMemcpyDeviceToDevice, props.stream);

		if (cudaStatus != cudaSuccess)
		{
			LOG_ERROR << "copy failed<" << cudaStatus << ">";
			return false;
		}
		return true;
	}

	bool convertNV12toRGB(bool intermediate)
	{
		int pitch = (intermediate) ? intermediatePitch[0] : dstPitch[0];
		Npp8u* target = (intermediate) ? intermediatedst[0] : dst[0];
		auto status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(src,
			srcPitch[0],
			target,
			pitch,
			srcSize[0],
			nppStreamCtx
		);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx failed<" << status << ">";
		}
		return true;
	}

	bool convertNV12toBGR()
	{
		auto status = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(src, srcPitch[0], dst[0], dstPitch[0], srcSize[0], nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx failed<" << status << ">";
			return false;
		}

		return true;
	}

	bool convertNV12toYUV420()
	{
		auto status = nppiNV12ToYUV420_8u_P2P3R_Ctx(src, srcPitch[0], dst, dstPitch, srcSize[0], nppStreamCtx);

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "nppiNV12ToYUV420_8u_P2P3R_Ctx failed<" << status << ">";
			return false;
		}
		return true;
	}

	bool convertYUV411_ItoYUV444()
	{
		lanuchAPPYUV411ToYUV444(src[0], srcPitch[0], dst, dstPitch[0], srcSize[0], props.stream);
		return true;
	}

	bool execute(frame_sp buffer, frame_sp outBuffer, frame_sp intermediateBuffer)
	{
		for (auto i = 0; i < inputChannels; i++)
		{
			src[i] = static_cast<const Npp8u*>(buffer->data()) + srcNextPtrOffset[i];
		}

		for (auto i = 0; i < outputChannels; i++)
		{
			dst[i] = static_cast<Npp8u*>(outBuffer->data()) + dstNextPtrOffset[i];
		}

		for (auto i = 0; i < intermediateChannels; i++)
		{
 			intermediatedst[i] = static_cast<Npp8u*>(intermediateBuffer->data()) + intermediateNextPtrOffset[i];
		}

		for (auto i = 0; i < intermediateChannels; i++)
		{
 			intermediatedst[i] = static_cast<Npp8u*>(intermediateBuffer->data()) + intermediateNextPtrOffset[i];
		}

		switch (inputImageType)
	    {
		case ImageMetadata::MONO:
		{
			switch (outputImageType)
			{
			case ImageMetadata::RGB:
				return convertMONOtoRGB();
			case ImageMetadata::BGR:
				return convertMONOtoBGR();
			case ImageMetadata::RGBA:
				return convertMONOtoRGBA();
			case ImageMetadata::BGRA:
				return convertMONOtoBGRA();
			case ImageMetadata::YUV420:
				return convertMONOtoYUV420();
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
		}
		case ImageMetadata::RGB:
		{
			switch (outputImageType)
			{
			case ImageMetadata::MONO:
				return convertRGBtoMONO();
			case ImageMetadata::BGR:
				return convertRGBtoBGR();
			case ImageMetadata::RGBA:
				return convertRGBtoRGBA(false);
			case ImageMetadata::BGRA:
				return convertRGBtoBGRA(false);
			case ImageMetadata::YUV420:
				return convertRGBtoYUV420();
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
		}

		case ImageMetadata::BGR:
		{
			switch (outputImageType)
			{
			case ImageMetadata::MONO:
				return convertBGRtoMONO();
			case ImageMetadata::RGB:
				return convertBGRtoRGB();
			case ImageMetadata::RGBA:
				return convertBGRtoRGBA();
			case ImageMetadata::BGRA:
				return convertBGRtoBGRA();
			case ImageMetadata::YUV420:
				return convertBGRtoYUV420();
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
		}

		case ImageMetadata::RGBA:
		{
			switch (outputImageType)
			{
			case ImageMetadata::MONO:
				return convertRGBAtoMONO();
			case ImageMetadata::BGR:
				return convertRGBAtoBGR();
			case ImageMetadata::RGB:
				return convertRGBAtoRGB();
			case ImageMetadata::BGRA:
				return convertRGBAtoBGRA();
			case ImageMetadata::YUV420:
				return convertRGBAtoYUV420();
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
		}

		case ImageMetadata::BGRA:
		{
			switch (outputImageType)
			{
			case ImageMetadata::MONO:
				return convertBGRAtoMONO();
			case ImageMetadata::RGB:
				return convertBGRAtoRGB();
			case ImageMetadata::RGBA:
				return convertBGRAtoRGBA();
			case ImageMetadata::BGR:
				return convertBGRAtoBGR();
			case ImageMetadata::YUV420:
				return convertBGRAtoYUV420();
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
		}

		case ImageMetadata::YUV420:
		{
			switch (outputImageType)
			{
			case ImageMetadata::MONO:
				return convertYUV420toMONO();
			case ImageMetadata::RGB:
				return convertYUV420toRGB();
			case ImageMetadata::RGBA:
				return convertYUV420toRGBA();
			case ImageMetadata::BGRA:
				return convertYUV420toBGRA();
			case ImageMetadata::BGR:
				return convertYUV420toBGR();
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
		}

		case ImageMetadata::NV12:
		{
			switch (outputImageType)
			{
			case ImageMetadata::MONO:
				return convertNV12toMONO();
			case ImageMetadata::RGB:
				return convertNV12toRGB(false);
			case ImageMetadata::BGR:
				return convertNV12toBGR();
			case ImageMetadata::YUV420:
				return convertNV12toYUV420();
			case ImageMetadata::RGBA:
				convertNV12toRGB(true);
				return convertRGBtoRGBA(true);
			case ImageMetadata::BGRA:
				convertNV12toRGB(true);
				return convertRGBtoBGRA(true);
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
			break;
			}
		}

		case ImageMetadata::YUV411_I:
		{
			switch (outputImageType)
			{
			case ImageMetadata::YUV444:
				return convertYUV411_ItoYUV444();
			default:
				throw AIPException(AIP_FATAL, "conversion not supported");
			}
		}
	}
		return true;
	}

	bool setMetadata(framemetadata_sp& input, framemetadata_sp& output, framemetadata_sp mIntermediate)
	{
		inputFrameType = input->getFrameType();
		outputFrameType = output->getFrameType();
		if (inputFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			inputImageType = inputRawMetadata->getImageType();
			inputChannels = inputRawMetadata->getChannels();
			srcSize[0] = { inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			srcRect[0] = { 0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			srcNextPtrOffset[0] = 0;
			srcRowSize[0] = inputRawMetadata->getRowSize();
		}
		else if (inputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
			inputImageType = inputRawMetadata->getImageType();
			inputChannels = inputRawMetadata->getChannels();

			for (auto i = 0; i < inputChannels; i++)
			{
				srcSize[i] = { inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				srcRect[i] = { 0, 0, inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				srcPitch[i] = static_cast<int>(inputRawMetadata->getStep(i));
				srcNextPtrOffset[i] = inputRawMetadata->getNextPtrOffset(i);
				srcRowSize[i] = inputRawMetadata->getRowSize(i);
			}
		}

		if (outputFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
			outputImageType = outputRawMetadata->getImageType();
			outputChannels = outputRawMetadata->getChannels();

			dstSize[0] = { outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
			dstRect[0] = { 0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
			dstPitch[0] = static_cast<int>(outputRawMetadata->getStep());
			dstNextPtrOffset[0] = 0;
		}
		else if (outputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(output);
			outputImageType = outputRawMetadata->getImageType();
			outputChannels = outputRawMetadata->getChannels();

			for (auto i = 0; i < outputChannels; i++)
			{
				dstSize[i] = { outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i) };
				dstRect[i] = { 0, 0, outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i) };
				dstPitch[i] = static_cast<int>(outputRawMetadata->getStep(i));
				dstNextPtrOffset[i] = outputRawMetadata->getNextPtrOffset(i);
			}
		}

		if (mIntermediate != nullptr)
		{
			intermediateFrameType = mIntermediate->getFrameType();

			if (intermediateFrameType = FrameMetadata::RAW_IMAGE)
			{
				auto intermediateRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mIntermediate);
				intermediateImageType = intermediateRawMetadata->getImageType();
				intermediateChannels = intermediateRawMetadata->getChannels();

				intermediateSize[0] = { intermediateRawMetadata->getWidth(), intermediateRawMetadata->getHeight() };
				intermediateRect[0] = { 0, 0, intermediateRawMetadata->getWidth(), intermediateRawMetadata->getHeight() };
				intermediatePitch[0] = static_cast<int>(intermediateRawMetadata->getStep());
				intermediateNextPtrOffset[0] = 0;
			}
			else if (outputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
			{
				auto intermediateRawPlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mIntermediate);
				intermediateImageType = intermediateRawPlanarMetadata->getImageType();
				intermediateChannels = intermediateRawPlanarMetadata->getChannels();

				for (auto i = 0; i < intermediateChannels; i++)
				{
					intermediateSize[i] = { intermediateRawPlanarMetadata->getWidth(i), intermediateRawPlanarMetadata->getHeight(i) };
					intermediateRect[i] = { 0, 0, intermediateRawPlanarMetadata->getWidth(i), intermediateRawPlanarMetadata->getHeight(i) };
					intermediatePitch[i] = static_cast<int>(intermediateRawPlanarMetadata->getStep(i));
					intermediateNextPtrOffset[i] = intermediateRawPlanarMetadata->getNextPtrOffset(i);
				}

			}

		}

		return true;
	}

protected:
	FrameMetadata::FrameType inputFrameType;
	FrameMetadata::FrameType outputFrameType;
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;
	FrameMetadata::FrameType intermediateFrameType;
	ImageMetadata::ImageType intermediateImageType;


	int inputChannels;
	int outputChannels;
	const Npp8u* src[4];
	NppiRect srcRect[4];
	int srcPitch[4];
	size_t srcNextPtrOffset[4];
	size_t srcRowSize[4];
	Npp8u* dst[4];
	NppiSize dstSize[4];
	NppiRect dstRect[4];
	int dstPitch[4];
	size_t dstNextPtrOffset[4];

	Npp8u* intermediatedst[4];
	NppiSize intermediateSize[4];
	NppiRect intermediateRect[4];
	int intermediatePitch[4];
	size_t intermediateNextPtrOffset[4];

	CCNPPIProps props;
	NppStreamContext nppStreamCtx;

public:
	int intermediateChannels = 0;
	NppiSize srcSize[4];
	bool intermediateConv = false;
};

CCNPPI::CCNPPI(CCNPPIProps _props) : Module(TRANSFORM, "CCNPPI", _props), props(_props), mFrameLength(0), mNoChange(false)
{
	mDetail.reset(new Detail(_props));
}

CCNPPI::~CCNPPI() {}

bool CCNPPI::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool CCNPPI::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	mOutputFrameType = metadata->getFrameType();

	if (mOutputFrameType != FrameMetadata::RAW_IMAGE && mOutputFrameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << mOutputFrameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}

	return true;
}

void CCNPPI::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);

	mInputFrameType = metadata->getFrameType();
	switch (props.imageType)
	{
	case ImageMetadata::MONO:
	case ImageMetadata::BGR:
	case ImageMetadata::BGRA:
	case ImageMetadata::RGB:
	case ImageMetadata::RGBA:
	case ImageMetadata::YUV411_I:
		mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
		break;
	case ImageMetadata::YUV420:
	case ImageMetadata::YUV444:
		mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::CUDA_DEVICE));
		break;
	default:
		throw AIPException(AIP_FATAL, "Unsupported frameType<" + std::to_string(mInputFrameType) + ">");
	}

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);
}

bool CCNPPI::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool CCNPPI::term()
{
	return Module::term();
}

bool CCNPPI::process(frame_container& frames)
{
	auto frame = frames.cbegin()->second;

	frame_sp outFrame;
	frame_sp intermediateFrame;
	size_t intermediateFrameSize = NOT_SET_NUM;
	if (mDetail->intermediateConv)
	{
		intermediateFrameSize = (mDetail->srcSize[0].width) * (mDetail->srcSize[0].height) * (mDetail->intermediateChannels);
	}
	if (!mNoChange)
	{
		outFrame = makeFrame();
		if (mDetail->intermediateConv)
		{
			intermediateFrame = makeFrame(intermediateFrameSize);
		}
		if (!mDetail->execute(frame, outFrame, intermediateFrame))
		{
			return true;
		}
	}
	else
	{
		outFrame = frame;
	}
	
	intermediateFrame.reset();
	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CCNPPI::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void CCNPPI::setMetadata(framemetadata_sp& metadata)
{
	mInputFrameType = metadata->getFrameType();

	int width = NOT_SET_NUM;
	int height = NOT_SET_NUM;
	int type = NOT_SET_NUM;
	int depth = NOT_SET_NUM;
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;

	if (mInputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		width = rawMetadata->getWidth();
		height = rawMetadata->getHeight();
		type = rawMetadata->getType();
		depth = rawMetadata->getDepth();
		inputImageType = rawMetadata->getImageType();
	}
	else if (mInputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		width = rawMetadata->getWidth(0);
		height = rawMetadata->getHeight(0);
		depth = rawMetadata->getDepth();
		inputImageType = rawMetadata->getImageType();
	}

	mNoChange = false;
	if (inputImageType == props.imageType)
	{
		mNoChange = true;
		return;
	}

	if (mOutputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		RawImageMetadata outputMetadata(width, height, props.imageType, CV_8UC3, 512, depth, FrameMetadata::CUDA_DEVICE, true);
		rawOutMetadata->setData(outputMetadata);
		outputImageType = rawOutMetadata->getImageType();
	}
	else if (mOutputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		RawImagePlanarMetadata outputMetadata(width, height, props.imageType, 512, depth);
		rawOutMetadata->setData(outputMetadata);
		outputImageType = rawOutMetadata->getImageType();
	}

	mDetail->setConvMatrix();
	if (mDetail->convmatrix[inputImageType][outputImageType][0] == -1 && mDetail->convmatrix[inputImageType][outputImageType][1] == -1)
	{
		throw AIPException(AIP_FATAL, "conversion not supported");
	}

	mIntermediateMetadata = nullptr;

	if (inputImageType == ImageMetadata::NV12)
	{
		if ((outputImageType == ImageMetadata::BGRA) || (outputImageType == ImageMetadata::RGBA))
		{
			mDetail->intermediateConv = true;
			mIntermediateMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
			auto mrawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mIntermediateMetadata);
			RawImageMetadata moutputMetadata(width, height, ImageMetadata::RGB, CV_8UC3, 512, depth, FrameMetadata::CUDA_DEVICE, true);
			mrawOutMetadata->setData(moutputMetadata);
		}
	}

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata, mOutputMetadata, mIntermediateMetadata);
}

bool CCNPPI::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool CCNPPI::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}