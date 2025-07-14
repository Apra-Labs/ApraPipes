#include "triangle.h"
#include <iostream>

Triangle::Triangle(double b, double h, const std::string& color)
    : Shape("Triangle", color), base(b), height(h) {}

double Triangle::getArea() const {
    return 0.5 * base * height;
}

double Triangle::getPerimeter() const {
    return base + height + std::sqrt(base * base + height * height);
}

void Triangle::print() const {
    std::cout << "Type: " << type << "\n"
        << "Shape: (" << base << "," << height << ")\n"
        << "Perimeter: " << getPerimeter() << "\n"
        << "Area: " << getArea() << "\n"
        << "Color: " << color << "\n\n";
}
