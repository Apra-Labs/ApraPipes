#include "JPEGFrameParser.h"
#include "Logger.h"
#include <string>

#define QTABLE_MAX_SIZE (128*2)
#define SOF_HEADER_SIZE 17

#define JPEG_PARSER_LENIANT         0
#define JPEG_PARSER_STRICT_RFC_2035 1

#define JPEG_PARSER_LEVEL  JPEG_PARSER_LENIANT

enum JPEG_Markers
{
    START_MARKER = 0xFF,
    SOI_MARKER = 0xD8, //Start Of Image
    JFIF_MARKER = 0xE0,
    EXIF_MARKER = 0xE1,
    CMT_MARKER = 0xFE,
    DQT_MARKER = 0xDB,
    SOF0_MARKER = 0xC0, //Start Of Frame (baseline DCT)
    SOF2_MARKER = 0xC2, //Start Of Frame (progressive DCT)
    DHT_MARKER = 0xC4,
    SOS_MARKER = 0xDA,
    EOI_MARKER = 0xD9,
    DRI_MARKER = 0xDD
};

typedef struct _CompInfo
{
    uint8_t id;
    uint8_t samp;
    uint8_t qt;
} CompInfo;

static uint16_t _jpegHeaderSize(const uint8_t* data, unsigned int offset)
{
    const uint16_t headerSize = data[offset] << 8 | data[offset + 1];
    return headerSize;
}


JpegFrameParser::JpegFrameParser() :
    mWidth(0), 
    mHeight(0), 
    mType(0),
    mPrecision(0), 
    mQFactor(255),
    mQTables(nullptr), 
    mQTablesLength(0),
    mRestartInterval(0),
    mScandata(nullptr), 
    mScandataLen(0)
{
    mQTables = new uint8_t[QTABLE_MAX_SIZE];
    memset(mQTables, 8, QTABLE_MAX_SIZE);
}

JpegFrameParser::~JpegFrameParser()
{
	if (mQTables)
	{
		delete[] mQTables;
	}
}

uint16_t JpegFrameParser::GetWidth() const
{ 
    return mWidth; 
}

uint16_t JpegFrameParser::GetHeight() const
{ 
    return mHeight; 
}

uint8_t JpegFrameParser::GetType() const
{ 
    return mType; 
}

uint8_t JpegFrameParser::GetPrecision() const
{ 
    return mPrecision; 
}

uint8_t JpegFrameParser::GetQFactor() const
{ 
    return mQFactor; 
}

unsigned short JpegFrameParser::GetRestartInterval() const
{ 
    return mRestartInterval; 
}

uint8_t const* JpegFrameParser::QuantizationTables(unsigned short& length) const
{
	//AK
    //length = mQTablesLength;
	length = 0;
    return mQTables;
}

bool JpegFrameParser::Parse(const uint8_t* data, unsigned int size)
{
    bool parseRes = false;

    mWidth = 0;
    mHeight = 0;
    mType = 1;
    mPrecision = 0;
    //mQFactor = 0;
    mRestartInterval = 0,

    mScandata = nullptr;
    mScandataLen = 0;

    unsigned int offset = 0;

    bool dqtFound = false;
    bool sosFound = false;
    bool sofFound = false;
    bool driFound = false;

    unsigned int jpeg_header_size = 0;

    while ((!sosFound) && (offset < size)) 
    {
        unsigned int jpgMarker = scanJpegMarker(data, size, &offset);
        switch (jpgMarker)
        {
            case JFIF_MARKER:
            case CMT_MARKER:
            case DHT_MARKER:
            {
                offset += _jpegHeaderSize(data, offset);
            }
            break;

            case EXIF_MARKER:
            {
                LOG_ERROR << "EXIF not yet supported";
                offset += _jpegHeaderSize(data, offset);
            }
            break;

            case SOF0_MARKER:
            {
                if (!readSOF(data, size, &offset))
                {
                    LOG_ERROR << "Error in parsing SOF_MARKER";
                    return false;
                }
                sofFound = true;
            }
            break;

            case SOF2_MARKER:
            {
                LOG_ERROR << "Not proessing Progressive JPEGs";
            }
            break;

            case DQT_MARKER:
            {
                offset = readDQT(data, size, offset);
                dqtFound = true;
            }
            break;

            case SOS_MARKER:
            {
                sosFound = true;
               //AK Bug fix 
			   //jpeg_header_size = offset +_jpegHeaderSize(data, offset);
            }
            break;

            case EOI_MARKER:
            {
                /* EOI reached before SOS!? */
                LOG_ERROR << "EOI reached before SOS!?";
            }
            break;

            case SOI_MARKER:
            {
                LOG_DEBUG << "SOI found";
            }
            break;

            case DRI_MARKER:
            {
                LOG_DEBUG << "DRI found";
                if (readDRI(data, size, &offset) == 0)
                {
                    driFound = true;
                }
            }
            break;

            default:
            {
                //App Specific MetaData Will be present in M_APP0  to M_APP14
                //{0xFF, 0xEn}
                if (jpgMarker < 0xE0)
                {
                    LOG_ERROR << "Unknown Marker " << jpgMarker;
                }
                
            }
            break;
        }
    }
    if (!dqtFound || !sofFound) 
    {
        LOG_ERROR << "Unsupported Format";
    }

    else if (mWidth == 0 || mHeight == 0) 
    {
        LOG_ERROR << "No Dimension";
    }
    else
    {
        mScandata = data + jpeg_header_size;
        mScandataLen = size - jpeg_header_size;

        if (driFound) 
        {
            mType += 64;
        }

        LOG_DEBUG << "Scandata Type Set To " << mType;

        parseRes = true;
    }

    return parseRes;
}

uint8_t const* JpegFrameParser::GetScandata(unsigned int& length) const
{
    length = mScandataLen;

    return mScandata;
}

unsigned int JpegFrameParser::scanJpegMarker(const uint8_t* data,
                                                unsigned int size,
                                                unsigned int* offset)
{
    while ((data[(*offset)++] != START_MARKER) && ((*offset) < size));

    if ((*offset) >= size) {
        return EOI_MARKER;
    }
    else {
        unsigned int marker;

        marker = data[*offset];
        (*offset)++;

        return marker;
    }
}

bool JpegFrameParser::readSOF(const uint8_t* data, 
                                unsigned int size,
                                unsigned int* offset)
{
    bool readRes = false;

    int i, j;
    CompInfo elem;
    CompInfo info[3] = { { 0, }, };

    unsigned int off = *offset;

    /* we need at least 17 bytes for the SOF */
    if (off + SOF_HEADER_SIZE > size)
    {
        LOG_ERROR << "Incorrect data Size for SOF. Min " << SOF_HEADER_SIZE << " bytes needed";
        return false;
    }

    const uint16_t sof_size = _jpegHeaderSize(data, off);
#if (JPEG_PARSER_LEVEL == JPEG_PARSER_STRICT_RFC_2035)
    if (sof_size < SOF_HEADER_SIZE)
    {
        LOG_ERROR << "Incorrect sof_size<%d>. Min " << SOF_HEADER_SIZE << " bytes needed";
        return false;
    }
#endif //JPEG_PARSER_LEVEL

    *offset += sof_size;

    /* skip size */
    off += 2;

    /* precision should be 8 */
    const uint8_t precision = data[off++];
    if (precision != 8)
    {
        LOG_ERROR << "SOF: Bad Precision " << precision;
        return false;
    }

    /* read dimensions */
    unsigned int height = data[off] << 8 | data[off + 1];
    unsigned int width = data[off + 2] << 8 | data[off + 3];
    off += 4;

    if (height == 0 || height > 4096
        || width ==0 || width > 4096)
    {
        LOG_ERROR << "Invalid Dimensions (Width:" << width << " Height:" << height << ")";
        return false;
    }

    mWidth = width / 8;
    mHeight = height / 8;

    const uint8_t noComponents = data[off++];
#if (JPEG_PARSER_LEVEL == JPEG_PARSER_STRICT_RFC_2035)
    //we only support 3 components
    if (noComponents != 3)
    {
        LOG_ERROR << "Invalid No Of Components %d)\n", noComponents);
        return false;
    }
#endif //JPEG_PARSER_LEVEL

    unsigned int infolen = 0;
    for (i = 0; i < noComponents; i++) 
    {
        elem.id = data[off++];
        elem.samp = data[off++];
        elem.qt = data[off++];

        /* insertion sort from the last element to the first */
        for (j = infolen; j > 1; j--) 
        {
            if (info[j - 1].id < elem.id)
            {
                break;
            }
            info[j] = info[j - 1];
        }
        info[j] = elem;
        infolen++;
    }

    bool typeSet = true;

    /* see that the components are supported */
    if (info[0].samp == 0x21) 
    {
        LOG_DEBUG << "SOF Type Set to 0";
        mType = 0;
    }
    else if (info[0].samp == 0x22) 
    {
        LOG_DEBUG << "SOF Type Set to 1";
        mType = 1;
    }
#if (JPEG_PARSER_LEVEL == JPEG_PARSER_STRICT_RFC_2035)
    else 
    {
        LOG_ERROR << "info[0]=" << info[0].samp  <<  "0x%02X Not Supported";
        typeSet = false;
    }
#endif //JPEG_PARSER_LEVEL

    if (typeSet)
    {
        bool samplesVerified = true;

        if (noComponents == 3)
        {
            if (info[1].samp != 0x11)
            {
                LOG_ERROR << "info[1]=" << info[0].samp << " Not Supported";
                samplesVerified = false;
            }
            if (info[2].samp != 0x11)
            {
                LOG_ERROR << "info[2]=" << info[0].samp << " Not Supported";
                samplesVerified = false;
            }
           /* if (info[1].qt != info[2].qt)
            {
                LOG_ERROR << "Mismatch in QT Data : 0x%02X != 0x%02X\n", info[1].qt, info[2].qt);
                samplesVerified = false;
            }*/
        }
                
        readRes = samplesVerified;
    }

    return readRes;
}

unsigned int JpegFrameParser::readDQT(const uint8_t* data,
                                        unsigned int size,
                                        unsigned int offset)
{
    unsigned int tab_size;
    uint8_t prec;
    uint8_t id;

    if (offset + 2 > size)
    {
        LOG_ERROR << "DQT data Size " << size << " too Small";
        return size;
    }

    uint16_t quant_size = _jpegHeaderSize(data, offset);
    if (quant_size < 2)
    {
        LOG_ERROR << "Quant Size " << quant_size << " too Small";
        return size;
    }

    /* clamp to available data */
    if (offset + quant_size > size) 
    {
        quant_size = size - offset;
    }

    offset += 2;
    quant_size -= 2;

    while (quant_size > 0) 
    {
        /* not enough to read the id */
        if (offset + 1 > size)
        {
            break;
        }

        id = data[offset] & 0x0f;
        if (id == 15)
        {
            LOG_ERROR << "DQT Invalid Id " << id;
            return offset + quant_size;
        }

        prec = (data[offset] & 0xf0) >> 4;
        if (prec) 
        {
            LOG_DEBUG << "DQT Tab Size Set to 128";
            tab_size = 128;
            mQTablesLength = 128 * 2;
        }
        else 
        {
            LOG_DEBUG << "DQT Tab Size Set to 64";
            tab_size = 64;
            mQTablesLength = 64 * 2;
        }

        /* there is not enough for the table */
        if (quant_size < tab_size + 1)
        {
            LOG_ERROR << "DQT Table doesn't exist";
            return offset + quant_size;
        }

        //LOGGY("Copy quantization table: %u\n", id);
        memcpy(&mQTables[id * tab_size], &data[offset + 1], tab_size);

        tab_size += 1;
        quant_size -= tab_size;
        offset += tab_size;
    }

    return offset + quant_size;
}

int JpegFrameParser::readDRI(const uint8_t* data,
    unsigned int size, unsigned int* offset)
{
    unsigned int dri_size, off;

    off = *offset;

    /* we need at least 4 bytes for the DRI */
    if (off + 4 > size) goto wrong_size;

    dri_size = _jpegHeaderSize(data, off);
    if (dri_size < 4) goto wrong_length;

    *offset += dri_size;
    off += 2;

    mRestartInterval = (data[off] << 8) | data[off + 1];

    return 0;

wrong_size:
    return -1;

wrong_length:
    *offset += dri_size;
    return -1;
}