#define BOOST_TEST_MODULE ShapeTest
#include <boost/test/included/unit_test.hpp>

#include "circle.h"
#include "rectangle.h"
#include "triangle.h"
#include <memory>
#include <cmath>  // for std::sqrt
namespace utf = boost::unit_test;


BOOST_AUTO_TEST_SUITE(shape_tests)


//BOOST_AUTO_TEST_CASE(MyRectangle Test_Variants)
//{
//
//    MyRectangle  r1(4.0, 6.0, "Yellow");
//    double expected1_area = 24.0;
//    double expected1_perimeter = 20.0;
//    double actual1_area = r1.getArea();
//    double actual1_perimeter = r1.getPerimeter();
//    BOOST_TEST_MESSAGE("r1.getArea() = " << actual1_area << ", expected = " << expected1_area);
//    BOOST_TEST_MESSAGE("r1.getPerimeter() = " << actual1_perimeter << ", expected = " << expected1_perimeter);
//    BOOST_CHECK_CLOSE_FRACTION(actual1_area, expected1_area, 0.001);
//    BOOST_CHECK_CLOSE_FRACTION(actual1_perimeter, expected1_perimeter, 0.001);
//
//    MyRectangle * r2 = new MyRectangle (5.0, 7.0, "Cyan");
//    double expected2_area = 35.0;
//    double expected2_perimeter = 24.0;
//    double actual2_area = r2->getArea();
//    double actual2_perimeter = r2->getPerimeter();
//    BOOST_TEST_MESSAGE("r2->getArea() = " << actual2_area << ", expected = " << expected2_area);
//    BOOST_TEST_MESSAGE("r2->getPerimeter() = " << actual2_perimeter << ", expected = " << expected2_perimeter);
//    BOOST_CHECK_CLOSE_FRACTION(actual2_area, expected2_area, 0.001);
//    BOOST_CHECK_CLOSE_FRACTION(actual2_perimeter, expected2_perimeter, 0.001);
//    delete r2;
//
//    std::shared_ptr<MyRectangle > r3 = std::make_shared<MyRectangle >(2.0, 3.0, "Pink");
//    double expected3_area = 6.0;
//    double expected3_perimeter = 10.0;
//    double actual3_area = r3->getArea();
//    double actual3_perimeter = r3->getPerimeter();
//    BOOST_TEST_MESSAGE("r3->getArea() = " << actual3_area << ", expected = " << expected3_area);
//    BOOST_TEST_MESSAGE("r3->getPerimeter() = " << actual3_perimeter << ", expected = " << expected3_perimeter);
//    BOOST_CHECK_CLOSE_FRACTION(actual3_area, expected3_area, 0.001);
//    BOOST_CHECK_CLOSE_FRACTION(actual3_perimeter, expected3_perimeter, 0.001);
//}

BOOST_AUTO_TEST_CASE(CircleTest_Variants)
{
    Circle c1(5.0, "Red");
    double expected1_area = 3.14 * 25.0;
    double actual1_area = c1.getArea();
    BOOST_TEST_MESSAGE("c1.getArea() = " << actual1_area << ", expected = " << expected1_area);
    BOOST_CHECK_CLOSE_FRACTION(actual1_area, expected1_area, 0.001);

    double expected1_perim = 2 * 3.14 * 5.0;
    double actual1_perim = c1.getPerimeter();
    BOOST_TEST_MESSAGE("c1.getPerimeter() = " << actual1_perim << ", expected = " << expected1_perim);
    BOOST_CHECK_CLOSE_FRACTION(actual1_perim, expected1_perim, 0.001);

    Circle* c2 = new Circle(10.0, "Blue");
    double expected2_area = 3.14 * 100.0;
    double actual2_area = c2->getArea();
    BOOST_TEST_MESSAGE("c2->getArea() = " << actual2_area << ", expected = " << expected2_area);
    BOOST_CHECK_CLOSE_FRACTION(actual2_area, expected2_area, 0.001);

    double expected2_perim = 2 * 3.14 * 10.0;
    double actual2_perim = c2->getPerimeter();
    BOOST_TEST_MESSAGE("c2->getPerimeter() = " << actual2_perim << ", expected = " << expected2_perim);
    BOOST_CHECK_CLOSE_FRACTION(actual2_perim, expected2_perim, 0.001);
    delete c2;

    std::shared_ptr<Circle> c3 = std::make_shared<Circle>(3.0, "Green");
    double expected3_area = 3.14 * 9.0;
    double actual3_area = c3->getArea();
    BOOST_TEST_MESSAGE("c3->getArea() = " << actual3_area << ", expected = " << expected3_area);
    BOOST_CHECK_CLOSE_FRACTION(actual3_area, expected3_area, 0.001);

    double expected3_perim = 2 * 3.14 * 3.0;
    double actual3_perim = c3->getPerimeter();
    BOOST_TEST_MESSAGE("c3->getPerimeter() = " << actual3_perim << ", expected = " << expected3_perim);
    BOOST_CHECK_CLOSE_FRACTION(actual3_perim, expected3_perim, 0.001);
}


BOOST_AUTO_TEST_CASE(TriangleTest_Variants)
{
    Triangle t1(3.0, 4.0, "Green");
    double expected1 = 3.0 + 4.0 + std::sqrt(3.0 * 3.0 + 4.0 * 4.0);  // 3+4+5 = 12
    double actual1 = t1.getPerimeter();
    BOOST_TEST_MESSAGE("t1.getPerimeter() = " << actual1 << ", expected = " << expected1);
    BOOST_CHECK_CLOSE_FRACTION(actual1, expected1, 0.001);

    Triangle* t2 = new Triangle(5.0, 12.0, "Purple");
    double expected2 = 5.0 + 12.0 + std::sqrt(5.0 * 5.0 + 12.0 * 12.0);  // 5+12+13 = 30
    double actual2 = t2->getPerimeter();
    BOOST_TEST_MESSAGE("t2->getPerimeter() = " << actual2 << ", expected = " << expected2);
    BOOST_CHECK_CLOSE_FRACTION(actual2, expected2, 0.001);
    delete t2;

    std::shared_ptr<Triangle> t3 = std::make_shared<Triangle>(6.0, 8.0, "Brown");
    double expected3 = 6.0 + 8.0 + std::sqrt(6.0 * 6.0 + 8.0 * 8.0);  // 6+8+10 = 24
    double actual3 = t3->getPerimeter();
    BOOST_TEST_MESSAGE("t3->getPerimeter() = " << actual3 << ", expected = " << expected3);
    BOOST_CHECK_CLOSE_FRACTION(actual3, expected3, 0.001);
}



 


 



BOOST_AUTO_TEST_SUITE_END()
