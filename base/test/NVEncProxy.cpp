#include "NVEncProxy.h"
#include "nvEncodeAPI.h"
#include <boost/dll.hpp>
#include "AIPExceptions.h"

#if defined(_WIN32)
	#if ARCH_X86_64
		#define NVENC_LIBNAME "nvEncodeAPI64.dll"
	#else
		#define NVENC_LIBNAME "nvEncodeAPI.dll"
	#endif
#else
	#define NVENC_LIBNAME "libnvidia-encode.so"
#endif

NVEncProxy::NVEncProxy()
{
	boost::dll::shared_library lib(NVENC_LIBNAME);
	if (!lib.is_loaded())
	{
		throw AIPException(AIP_FATAL, "NVENC library file is not found. Please ensure NV driver is installed. NV_ENC_ERR_NO_ENCODE_DEVICE");
	}
	boost::function<NVENCSTATUS(uint32_t*)> fNvEncodeAPIGetMaxSupportedVersion = lib.get<NVENCSTATUS(uint32_t*)>("NvEncodeAPIGetMaxSupportedVersion");
	
	uint32_t version = 0;
	uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
	auto rc=fNvEncodeAPIGetMaxSupportedVersion(&version);
	if (currentVersion > version)
	{
		throw AIPException(AIP_FATAL, "Current Driver Version does not support this NvEncodeAPI version, please upgrade driver. NV_ENC_ERR_INVALID_VERSION");
	}
}

