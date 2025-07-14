#include <iostream>
#include <iomanip>
#include "triangle.h"   
#include "circle.h" 
#include "rectangle.h" 
using namespace std;
 


int main_triangle() {
     
   cout << "Start of Triangle Shape" << endl;
   if(true){

      Triangle t1(3, 4, 5);
      t1.print();
      cout << "x = " << t1.getX() << endl;
      cout << "y = " << t1.getY() << endl;
      cout << "z = " << t1.getZ() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << t1.getPerimeter() << endl;
      cout << "Area = " << t1.getArea() << endl;

      t1.setX(6);
      t1.setY(8);
      t1.setZ(10); 


      cout << "breaker new " << endl;
      cout << "x = " << t1.getX() << endl;
      cout << "y = " << t1.getY() << endl;
      cout << "z = " << t1.getZ() << endl;
      cout << "Perimeter1 = " << t1.getPerimeter() << endl;
      cout << "Area2 = " << t1.getArea() << endl;
      t1.print();



      cout << "Constructing an instance with default values " << endl;
      Triangle t2;
      t2.print();


      cout << string(100, '$') << endl;
   }
 
 

   if(true){
      Triangle t1(3, 4, 5);
      Triangle* t1Ptr = &t1;  

       
      t1Ptr->print();
      cout << "x = " << t1Ptr->getX() << endl;
      cout << "y = " << t1Ptr->getY() << endl;
      cout << "Radius = " << t1Ptr->getZ() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << t1Ptr->getPerimeter() << endl;
      cout << "Area = " << t1Ptr->getArea() << endl;

       
      t1Ptr->setX(6);
      t1Ptr->setY(8);
      t1Ptr->setZ(10); 

      cout << "breaker new " << endl;
      cout << "x = " << t1Ptr->getX() << endl;
      cout << "y = " << t1Ptr->getY() << endl;
      cout << "Radius = " << t1Ptr->getZ() << endl;
      cout << "Perimeter1 = " << t1Ptr->getPerimeter() << endl;
      cout << "Area2 = " << t1Ptr->getArea() << endl;
      t1Ptr->print();

       
      Triangle t2;  
      Triangle* c2Ptr = &t2;  

      cout << "Constructing an instance with default values " << endl;
      c2Ptr->print();
      cout << string(100, '$') << endl;
   }


   if(true){
      cout << "Constructing an instance object dynamically (using 'new') " << endl;
      Triangle* t1 = new Triangle(3, 4, 5);   

       
      t1->print();
      cout << "x = " << t1->getX() << endl;
      cout << "y = " << t1->getY() << endl;
      cout << "Radius = " << t1->getZ() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << t1->getPerimeter() << endl;
      cout << "Area = " << t1->getArea() << endl;

       
      t1->setX(6);
      t1->setY(8);
      t1->setZ(10); 

      cout << "breaker new " << endl;
      cout << "x = " << t1->getX() << endl;
      cout << "y = " << t1->getY() << endl;
      cout << "Radius = " << t1->getZ() << endl;
      cout << "Perimeter1 = " << t1->getPerimeter() << endl;
      cout << "Area2 = " << t1->getArea() << endl;
      t1->print();

      
      delete t1;

   }

   cout << "End of Triangle Shape" << endl;
     
    return 0;

}


int main_circle() {
     
   cout << "Start of Circle Shape" << endl;
   if(true){
      Circle C1(3, 4, 5);
      C1.print();
      cout << "x = " << C1.getX() << endl;
      cout << "y = " << C1.getY() << endl;
      cout << "Radius = " << C1.getRadius() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << C1.getPerimeter() << endl;
      cout << "Area = " << C1.getArea() << endl;

      C1.setX(6);
      C1.setY(8);
      C1.setRadius(10); 


      cout << "breaker new " << endl;
      cout << "x = " << C1.getX() << endl;
      cout << "y = " << C1.getY() << endl;
      cout << "Radius = " << C1.getRadius() << endl;
      cout << "Perimeter1 = " << C1.getPerimeter() << endl;
      cout << "Area2 = " << C1.getArea() << endl;
      C1.print();



      cout << "Constructing an instance with default values " << endl;
      Circle C2;
      C2.print();


      cout << string(100, '$') << endl;
   }
 

 

   if(true){        
      Circle C1(3, 4, 5);
      Circle* c1Ptr = &C1;  

       
      c1Ptr->print();
      cout << "x = " << c1Ptr->getX() << endl;
      cout << "y = " << c1Ptr->getY() << endl;
      cout << "Radius = " << c1Ptr->getRadius() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << c1Ptr->getPerimeter() << endl;
      cout << "Area = " << c1Ptr->getArea() << endl;

       
      c1Ptr->setX(6);
      c1Ptr->setY(8);
      c1Ptr->setRadius(10); 

      cout << "breaker new " << endl;
      cout << "x = " << c1Ptr->getX() << endl;
      cout << "y = " << c1Ptr->getY() << endl;
      cout << "Radius = " << c1Ptr->getRadius() << endl;
      cout << "Perimeter1 = " << c1Ptr->getPerimeter() << endl;
      cout << "Area2 = " << c1Ptr->getArea() << endl;
      c1Ptr->print();

       
      Circle C2;  
      Circle* c2Ptr = &C2;  

      cout << "Constructing an instance with default values " << endl;
      c2Ptr->print();
      cout << string(100, '$') << endl;
   }


   if(true){
      cout << "Constructing an instance object dynamically (using 'new') " << endl;
      Circle* C1 = new Circle(3, 4, 5);   

            
      C1->print();
      cout << "x = " << C1->getX() << endl;
      cout << "y = " << C1->getY() << endl;
      cout << "Radius = " << C1->getRadius() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << C1->getPerimeter() << endl;
      cout << "Area = " << C1->getArea() << endl;

       
      C1->setX(6);
      C1->setY(8);
      C1->setRadius(10); 

      cout << "breaker new " << endl;
      cout << "x = " << C1->getX() << endl;
      cout << "y = " << C1->getY() << endl;
      cout << "Radius = " << C1->getRadius() << endl;
      cout << "Perimeter1 = " << C1->getPerimeter() << endl;
      cout << "Area2 = " << C1->getArea() << endl;
      C1->print();

      
      delete C1;

   }
   
   cout << "End of Circle Shape" << endl;
    return 0;

}


int main_rectangle() {
  
   cout << "Start of Rectangle Shape" << endl;
   if(true){

      Rectangle R1(3, 4);
      R1.print();
      cout << "x = " << R1.getX() << endl;
      cout << "y = " << R1.getY() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << R1.getPerimeter() << endl;
      cout << "Area = " << R1.getArea() << endl;

      R1.setX(6);
      R1.setY(8);
      

      cout << "breaker new " << endl;
      cout << "x = " << R1.getX() << endl;
      cout << "y = " << R1.getY() << endl;
      cout << "Perimeter1 = " << R1.getPerimeter() << endl;
      cout << "Area2 = " << R1.getArea() << endl;
      R1.print();

      

      cout << "Constructing an instance with default values " << endl;
      Rectangle R2;
      R2.print();


      cout << string(100, '$') << endl;
   }
 

 

   if(true){

      Rectangle R1(3, 4);
      Rectangle* t1Ptr = &R1;  

      
      t1Ptr->print();
      cout << "x = " << t1Ptr->getX() << endl;
      cout << "y = " << t1Ptr->getY() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << t1Ptr->getPerimeter() << endl;
      cout << "Area = " << t1Ptr->getArea() << endl;

       
      t1Ptr->setX(6);
      t1Ptr->setY(8);

      cout << "breaker new " << endl;
      cout << "x = " << t1Ptr->getX() << endl;
      cout << "y = " << t1Ptr->getY() << endl;
      cout << "Perimeter1 = " << t1Ptr->getPerimeter() << endl;
      cout << "Area2 = " << t1Ptr->getArea() << endl;
      t1Ptr->print();

       
      Rectangle t2;  
      Rectangle* t2Ptr = &t2;  

      cout << "Constructing an instance with default values " << endl;
      t2Ptr->print();
      cout << string(100, '$') << endl;
   }


   if(true){
      cout << "Constructing an instance object dynamically (using 'new') " << endl;
      Rectangle* R1 = new Rectangle(3, 4);  

       
      R1->print();
      cout << "x = " << R1->getX() << endl;
      cout << "y = " << R1->getY() << endl;
      cout << fixed << setprecision(2);
      cout << "Perimeter = " << R1->getPerimeter() << endl;
      cout << "Area = " << R1->getArea() << endl;

   
      R1->setX(6);
      R1->setY(8);

      cout << "breaker new " << endl;
      cout << "x = " << R1->getX() << endl;
      cout << "y = " << R1->getY() << endl;
      cout << "Perimeter1 = " << R1->getPerimeter() << endl;
      cout << "Area2 = " << R1->getArea() << endl;
      R1->print();
   

      
      delete R1;

   }

   cout << "End of Rectangle Shape" << endl;
     
    return 0;

}

int main(){
   main_triangle();
   main_circle();
   main_rectangle();
}