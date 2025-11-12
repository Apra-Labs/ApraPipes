#pragma once

#include "Module.h"

using CallbackFunction = std::function<void()>;
class EglRendererProps : public ModuleProps
{
public:
    struct TextInfo {
        std::string fontPath = ""; //Path for TTF font file
        std::string message = "";
        float scale = 0.0f;
        float fontSize = 0.0f;
        std::vector<float> color = {0.0f, 0.0f, 0.0f}; // RGB
        std::pair<int, int> position = {0, 0};
        float opacity = 1.0f;
    };

    struct ImageInfo {
        std::string path = "";
        std::pair<int, int> position = {0, 0};
        std::pair<uint32_t, uint32_t> size = {0, 0};
        float opacity = 1.0f; // width, height
    };
    EglRendererProps() : ModuleProps() {};
    // All settings enabled
    EglRendererProps(uint32_t _x_offset , uint32_t _y_offset ,
                     uint32_t _width , uint32_t _height ,
                     const TextInfo& _text ,
                     const ImageInfo& _image ,
                     float _opacity,bool _mask)
        : ModuleProps(),
          x_offset(_x_offset), y_offset(_y_offset),
          width(_width), height(_height),
          text(_text), image(_image),
          opacity(_opacity), mask(_mask)
    {}
    // --- Geometry (x, y) ---
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset)
        : ModuleProps(), x_offset(_x_offset), y_offset(_y_offset)
    {}

    // --- Geometry (x, y, width, height) ---
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,
                     uint32_t _width, uint32_t _height)
        : ModuleProps(), x_offset(_x_offset), y_offset(_y_offset),
          width(_width), height(_height)
    {}

    // --- Geometry + Text ---
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,
                     uint32_t _width, uint32_t _height,
                     const TextInfo& _text)
        : ModuleProps(), x_offset(_x_offset), y_offset(_y_offset),
          width(_width), height(_height), text(_text)
    {}

    // --- Geometry + Image ---
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,
                     uint32_t _width, uint32_t _height,
                     const ImageInfo& _image)
        : ModuleProps(), x_offset(_x_offset), y_offset(_y_offset),
          width(_width), height(_height), image(_image)
    {}

    // --- Geometry + Text + Image ---
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,
                     uint32_t _width, uint32_t _height,
                     const TextInfo& _text,
                     const ImageInfo& _image)
        : ModuleProps(), x_offset(_x_offset), y_offset(_y_offset),
          width(_width), height(_height),
          text(_text), image(_image)
    {}

    // --- Geometry + Opacity + Mask ---
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,
                     uint32_t _width, uint32_t _height,
                     float _opacity, bool _mask)
        : ModuleProps(), x_offset(_x_offset), y_offset(_y_offset),
          width(_width), height(_height),
          opacity(_opacity), mask(_mask)
    {}


    

    ~EglRendererProps() = default;

    // --- Geometry ---
    uint32_t x_offset = 0;
    uint32_t y_offset = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    // --- Rendering data ---
    TextInfo text;
    ImageInfo image;

    // --- Display ---
    float opacity = 1.0f;
    bool mask = false;
};


class EglRenderer : public Module
{
public:
    EglRenderer(EglRendererProps props);
    ~EglRenderer();
	void registerCallback(const CallbackFunction &_callback)
	{
		m_callbackFunction = _callback;
	}
    bool init();
    bool term();
	bool closeWindow();
	bool createWindow(int width, int height);
	void waitForNextFrame();
	bool statusOfEglWindow();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool processEOS(string& pinId);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	CallbackFunction m_callbackFunction = NULL;
};