#pragma once
#include "Shape.h"
#include <string>

class MyRectangle  : public Shape {
    double width;
    double height;

public:
    MyRectangle (double w, double h, const std::string& color);
    double getArea() const override;
    double getPerimeter() const override;
    void print() const override;
};
