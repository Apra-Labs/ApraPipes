#include "rectangle.h"
#include <iostream>

MyRectangle ::MyRectangle (double w, double h, const std::string& color)
    : Shape("MyRectangle ", color), width(w), height(h) {}

double MyRectangle ::getArea() const {
    return width * height;
}

double MyRectangle ::getPerimeter() const {
    return 2 * (width + height);
}

void MyRectangle ::print() const {
    std::cout << "Type: " << type << "\n"
        << "Shape: (" << width << "," << height << ")\n"
        << "Perimeter: " << getPerimeter() << "\n"
        << "Area: " << getArea() << "\n"
        << "Color: " << color << "\n\n";
}
