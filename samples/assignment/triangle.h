#pragma once
#include "Shape.h"

class Triangle : public Shape {
    double base;
    double height;

public:
    Triangle(double b, double h, const std::string& color);
    double getArea() const override;
    double getPerimeter() const override;
    void print() const override;
};
