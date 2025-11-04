#pragma once

#include "Module.h"

using CallbackFunction = std::function<void()>;
class EglRendererProps : public ModuleProps
{
public:
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset, uint32_t _width, uint32_t _height)
      : ModuleProps(),
        x_offset(_x_offset), y_offset(_y_offset),
        height(_height), width(_width),
        ttfFilePath(), message(), scale(0.0f), fontSize(0.0f), r(0.0f), g(0.0f), b(0.0f),opacity(1) {}

    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset)
      : ModuleProps(),
        x_offset(_x_offset), y_offset(_y_offset),
        height(0), width(0),
        ttfFilePath(), message(), scale(0.0f), fontSize(0.0f), r(0.0f), g(0.0f), b(0.0f),opacity(1) {}
    
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,float opacity)
      : ModuleProps(),
        x_offset(_x_offset), y_offset(_y_offset),opacity(opacity),
        height(0), width(0),
        ttfFilePath(), message(), scale(0.0f), fontSize(0.0f), r(0.0f), g(0.0f), b(0.0f) {
        }

    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,
                     const std::string& _ttfPath, const std::string& _message,
                     float _scale, float _r, float _g, float _b, float _fontSize,int _textPosX, int _textPosY)
      : ModuleProps(),
        x_offset(_x_offset), y_offset(_y_offset),
        height(0), width(0),
        ttfFilePath(_ttfPath), message(_message),
        scale(_scale), fontSize(_fontSize), r(_r), g(_g), b(_b), textPosX(_textPosX), textPosY(_textPosY),opacity(1) {}

    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset, uint32_t _width, uint32_t _height,
                     const std::string& _ttfPath, const std::string& _message,
                     float _scale, float _r, float _g, float _b, float _fontSize, int _textPosX, int _textPosY)
      : ModuleProps(),
        x_offset(_x_offset), y_offset(_y_offset),
        height(_height), width(_width),
        ttfFilePath(_ttfPath), message(_message),
        scale(_scale), fontSize(_fontSize), r(_r), g(_g), b(_b), textPosX(_textPosX), textPosY(_textPosY),opacity(1) {}
    
    EglRendererProps(uint32_t _x_offset, uint32_t _y_offset,
                     const std::string& _ttfPath, const std::string& _message,
                     float _scale, float _r, float _g, float _b, float _fontSize,int _textPosX, int _textPosY,float _opacity)
      : ModuleProps(),
        x_offset(_x_offset), y_offset(_y_offset),
        height(0), width(0),
        ttfFilePath(_ttfPath), message(_message),
        scale(_scale), fontSize(_fontSize), r(_r), g(_g), b(_b), textPosX(_textPosX), textPosY(_textPosY),opacity(_opacity) {}
      EglRendererProps(uint32_t _x_offset, uint32_t _y_offset, uint32_t _width, uint32_t _height,
                     const std::string& _ttfPath, const std::string& _message,
                     float _scale, float _r, float _g, float _b, float _fontSize, int _textPosX, int _textPosY,float _opacity)
      : ModuleProps(),
        x_offset(_x_offset), y_offset(_y_offset),
        height(_height), width(_width),
        ttfFilePath(_ttfPath), message(_message),
        scale(_scale), fontSize(_fontSize), r(_r), g(_g), b(_b), textPosX(_textPosX), textPosY(_textPosY),opacity(_opacity) {}
    ~EglRendererProps(){}

    uint32_t x_offset = 0;
    uint32_t y_offset = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    std::string ttfFilePath;
    std::string message;
    float scale = 0.0f;
    float fontSize = 0.0f;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    int textPosX = 0;
    int textPosY = 0;
    float opacity = 1;
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