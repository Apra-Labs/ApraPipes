#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include <cstdlib>
#include <cstdint>

class TestSignalGenerator::Detail
{
public:
    Detail(TestSignalGeneratorProps &_props)
        : mProps(_props), start_shade(0), end_shade(255), current_shade(start_shade), frameCount(0) {}

    ~Detail() {}

    bool generate(frame_sp &frame)
    {
        switch (mProps.pattern)
        {
        case TestPatternType::CHECKERBOARD:
            generateCheckerboard(frame);
            break;
        case TestPatternType::COLOR_BARS:
            generateColorBars(frame);
            break;
        case TestPatternType::GRID:
            generateGrid(frame);
            break;
        case TestPatternType::GRADIENT:
        default:
            generateGradient(frame);
            break;
        }
        frameCount++;
        return true;
    }

    void generateGradient(frame_sp &frame)
    {
        auto frame_ptr = frame->data();
        uint8_t* x = static_cast<uint8_t*>(frame_ptr);

        for (int height = 0; height < mProps.height * 1.5; height++)
        {
            memset(x, current_shade, mProps.width);
            x += mProps.width;
            current_shade += 1;
            if (current_shade > end_shade)
            {
                current_shade = start_shade;
            }
        }
    }

    void generateCheckerboard(frame_sp &frame)
    {
        auto frame_ptr = frame->data();
        uint8_t* y_plane = static_cast<uint8_t*>(frame_ptr);
        uint8_t* u_plane = y_plane + mProps.width * mProps.height;
        uint8_t* v_plane = u_plane + (mProps.width * mProps.height) / 4;

        // Checkerboard square size (8x8 cells)
        int cellWidth = mProps.width / 8;
        int cellHeight = mProps.height / 8;

        // Y plane - checkerboard pattern
        for (int row = 0; row < mProps.height; row++)
        {
            int cellRow = row / cellHeight;
            for (int col = 0; col < mProps.width; col++)
            {
                int cellCol = col / cellWidth;
                bool isWhite = ((cellRow + cellCol) % 2) == 0;
                y_plane[row * mProps.width + col] = isWhite ? 235 : 16;
            }
        }

        // U and V planes (neutral for grayscale)
        int uvWidth = mProps.width / 2;
        int uvHeight = mProps.height / 2;
        memset(u_plane, 128, uvWidth * uvHeight);
        memset(v_plane, 128, uvWidth * uvHeight);
    }

    void generateColorBars(frame_sp &frame)
    {
        auto frame_ptr = frame->data();
        uint8_t* y_plane = static_cast<uint8_t*>(frame_ptr);
        uint8_t* u_plane = y_plane + mProps.width * mProps.height;
        uint8_t* v_plane = u_plane + (mProps.width * mProps.height) / 4;

        // Standard color bar values (YUV)
        // White, Yellow, Cyan, Green, Magenta, Red, Blue, Black
        const int numBars = 8;
        const uint8_t barY[8] = {235, 210, 170, 145, 106, 81, 41, 16};
        const uint8_t barU[8] = {128, 16, 166, 54, 202, 90, 240, 128};
        const uint8_t barV[8] = {128, 146, 16, 34, 222, 240, 110, 128};

        int barWidth = mProps.width / numBars;

        // Y plane
        for (int row = 0; row < mProps.height; row++)
        {
            for (int col = 0; col < mProps.width; col++)
            {
                int barIdx = std::min(col / barWidth, numBars - 1);
                y_plane[row * mProps.width + col] = barY[barIdx];
            }
        }

        // U and V planes (subsampled 2x2)
        int uvWidth = mProps.width / 2;
        int uvHeight = mProps.height / 2;
        for (int row = 0; row < uvHeight; row++)
        {
            for (int col = 0; col < uvWidth; col++)
            {
                int barIdx = std::min((col * 2) / barWidth, numBars - 1);
                u_plane[row * uvWidth + col] = barU[barIdx];
                v_plane[row * uvWidth + col] = barV[barIdx];
            }
        }
    }

    void generateGrid(frame_sp &frame)
    {
        auto frame_ptr = frame->data();
        uint8_t* y_plane = static_cast<uint8_t*>(frame_ptr);
        uint8_t* u_plane = y_plane + mProps.width * mProps.height;
        uint8_t* v_plane = u_plane + (mProps.width * mProps.height) / 4;

        // 4x4 grid with numbered cells (different brightness levels)
        int cellWidth = mProps.width / 4;
        int cellHeight = mProps.height / 4;

        for (int row = 0; row < mProps.height; row++)
        {
            int cellRow = row / cellHeight;
            for (int col = 0; col < mProps.width; col++)
            {
                int cellCol = col / cellWidth;
                int cellNum = cellRow * 4 + cellCol;  // 0-15

                // Each cell has unique brightness (16 to 235 range)
                uint8_t brightness = 16 + (cellNum * 14);

                // Add grid lines (black borders between cells)
                bool isGridLine = (row % cellHeight < 2) || (col % cellWidth < 2);

                y_plane[row * mProps.width + col] = isGridLine ? 16 : brightness;
            }
        }

        // U and V planes - different colors for each quadrant
        int uvWidth = mProps.width / 2;
        int uvHeight = mProps.height / 2;
        for (int row = 0; row < uvHeight; row++)
        {
            int quadRow = (row * 2) / (mProps.height / 2);
            for (int col = 0; col < uvWidth; col++)
            {
                int quadCol = (col * 2) / (mProps.width / 2);
                int quad = quadRow * 2 + quadCol;

                // Different color tints per quadrant
                uint8_t u, v;
                switch (quad) {
                    case 0: u = 100; v = 100; break;  // Greenish
                    case 1: u = 156; v = 100; break;  // Blueish
                    case 2: u = 100; v = 156; break;  // Reddish
                    case 3: u = 156; v = 156; break;  // Purplish
                    default: u = 128; v = 128; break;
                }
                u_plane[row * uvWidth + col] = u;
                v_plane[row * uvWidth + col] = v;
            }
        }
    }

    void setProps(const TestSignalGeneratorProps &_props)
    {
        mProps = _props;
        reset();
    }
    void reset()
    {
        current_shade = start_shade;
        frameCount = 0;
    }

    TestSignalGeneratorProps mProps;
    uint8_t start_shade = 0;
    uint8_t end_shade = 255;
    uint8_t current_shade = 0;
    int frameCount = 0;
};

TestSignalGenerator::TestSignalGenerator(TestSignalGeneratorProps _props)
    : Module(SOURCE, "TestSignalGenerator", _props), outputFrameSize(0)
{
    mDetail.reset(new Detail(_props));
    mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(_props.width, _props.height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    mOutputPinId = addOutputPin(mOutputMetadata);
}

TestSignalGenerator::~TestSignalGenerator()
{
    mDetail->~Detail();
}

bool TestSignalGenerator::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }
    framemetadata_sp metadata = getFirstOutputMetadata();
    auto frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE_PLANAR)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins output frameType should be RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool TestSignalGenerator::init()
{
    if (!Module::init())
    {
        return false;
    }
    outputFrameSize = (getProps().width * getProps().height * 3) >> 1;

    return true;
}

bool TestSignalGenerator::produce()
{
    // Check if maxFrames limit reached
    auto props = getProps();
    if (props.maxFrames > 0 && mDetail->frameCount >= props.maxFrames) {
        stop();  // Stop the module gracefully (triggers EOS)
        return true;
    }

    auto mPinId = getOutputPinIdByType(FrameMetadata::RAW_IMAGE_PLANAR);
    frame_container frames;
    frame_sp frame = makeFrame(outputFrameSize);
    mDetail->generate(frame);
    frames.insert(make_pair(mPinId, frame));
    send(frames);
    return true;
}

bool TestSignalGenerator::term()
{
    return Module::term();
}

void TestSignalGenerator::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
}

bool TestSignalGenerator::handlePropsChange(frame_sp &frame)
{
    TestSignalGeneratorProps props;
    bool ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    outputFrameSize = (props.width * props.height * 3) >> 1;
    return ret;
}

void TestSignalGenerator::setProps(TestSignalGeneratorProps &props)
{
    Module::addPropsToQueue(props);
}

TestSignalGeneratorProps TestSignalGenerator::getProps()
{
    return mDetail->mProps;
}
