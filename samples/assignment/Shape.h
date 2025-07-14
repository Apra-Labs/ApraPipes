#pragma once
#include <string>

class Shape {
protected:
    std::string type;
    std::string color;

public:
    Shape(const std::string& type, const std::string& color)
        : type(type), color(color) {}

    virtual double getArea() const = 0;
    virtual double getPerimeter() const = 0;
    virtual void print() const = 0;

    virtual ~Shape() = default;
};
