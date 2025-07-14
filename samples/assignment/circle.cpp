#include "circle.h"
#include <iostream>
#define PI 3.1416

Circle::Circle(double r, const std::string& color)
    : Shape("Circle", color), radius(r) {}

double Circle::getArea() const {
    return PI * radius * radius;
}

double Circle::getPerimeter() const {
    return 2 * PI * radius;
}

void Circle::print() const {
    std::cout << "Type: " << type << "\n"
        << "Shape: r = " << radius << "\n"
        << "Perimeter: " << getPerimeter() << "\n"
        << "Area: " << getArea() << "\n"
        << "Color: " << color << "\n\n";
}
