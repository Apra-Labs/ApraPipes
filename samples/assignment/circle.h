#pragma once
#include "Shape.h"

class Circle : public Shape {
    double radius;

public:
    Circle(double r, const std::string& color);
    double getArea() const override;
    double getPerimeter() const override;
    void print() const override;
};
