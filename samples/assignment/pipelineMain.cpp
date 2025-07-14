#include "circle.h"
#include "rectangle.h"
#include "triangle.h"
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>

int main() {


 
    std::vector<std::shared_ptr<Shape>> shapes;
    std::string line;

    std::cout << "Enter shapes (type color [dimensions]). Example:\n";
    std::cout << "Circle Red 5\nMyRectangle  Blue 4 6\nTriangle Green 3 4\n\n";
    std::cout << "Enter shapes (press Enter on empty line to finish):\n";

    while (std::getline(std::cin, line) && !line.empty()) {
        std::istringstream iss(line);
        std::string type, color;
        iss >> type >> color;

        if (type == "Circle") {
            double r;
            iss >> r;
            shapes.push_back(std::make_shared<Circle>(r, color));
        }
        else if (type == "MyRectangle ") {
            double w, h;
            iss >> w >> h;
            shapes.push_back(std::make_shared<MyRectangle >(w, h, color));
        }
        else if (type == "Triangle") {
            double b, h;
            iss >> b >> h;
            shapes.push_back(std::make_shared<Triangle>(b, h, color));
        }
        else {
            std::cerr << "Unknown shape type: " << type << "\n";
        }
    }

    std::cout << "\n--- Shape Summary ---\n";
    for (const auto& shape : shapes) {
        shape->print();
    }

    return 0;
}
