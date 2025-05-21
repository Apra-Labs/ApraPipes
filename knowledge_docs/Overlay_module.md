# **Knowledge Transfer Document for Overlay Module**

## **1\. Introduction**

The **Overlay Module** is responsible for rendering graphical overlays on images. It provides functionalities for drawing basic shapes such as circles, lines, and rectangles, and supports serialization/deserialization of overlay data.

---

## **2\. Code Structure Overview**

The module is divided into three main components:

1. **Overlay.cpp**  
   * Defines overlay types (Circle, Line, Rectangle, Composite).  
   * Implements serialization, deserialization, and drawing functionalities.  
2. **OverlayFactory.cpp**  
   * Implements a **Factory Pattern** for dynamically creating overlay objects.  
3. **OverlayModule.cpp**  
   * Defines the processing pipeline that applies overlays to image frames.  
   * Handles input/output validation and manages overlay rendering.

---

## **3\. Overlay Types and Functionality**

### **3.1. Circle, Line, and Rectangle Overlays**

* These represent basic geometric shapes that can be drawn onto an image.  
* Each overlay supports:  
  * **Serialization & Deserialization** (using Boost).  
  * **Rendering** (using OpenCV functions).

### **3.2. Composite Overlay**

* Supports grouping multiple overlays into a single entity.  
* Uses the **Composite Design Pattern** to manage a collection of overlay objects.

### **3.3. Drawing Overlay**

* Manages multiple overlay objects and handles batch processing.  
* Uses **Visitor Pattern** to serialize, deserialize, and draw overlays efficiently.

---

## **4\. Overlay Factory**

The **OverlayFactory** class dynamically creates overlay objects based on type (Rectangle, Line, Circle, Composite).  
It also includes a **Builder Factory** for handling deserialization of overlay objects.

---

## **5\. OverlayModule**

The `OverlayModule` is the main processing component that applies overlays to image frames.

### **5.1. Key Functionalities**

* **Input Validation**: Ensures only valid image frames are processed.  
* **Overlay Processing**: Deserializes overlay data, applies overlays, and forwards the modified frame.  
* **Output Validation**: Ensures the output remains in the expected format.

### **5.2. Processing Flow**

1. **Receives Input Frames** (Raw Image / Overlay Data).  
2. **Deserializes Overlay Data** (if present).  
3. **Applies Overlays** to Raw Image frames.  
4. **Sends Processed Frames** to the output.

---

## **6\. Serialization & Deserialization**

* **Boost Serialization** is used to store and retrieve overlay objects efficiently.  
* Supports **binary serialization** for compact storage and fast transmission.

---

## **7\. Design Patterns Used**

* **Factory Pattern**: Creates overlay instances dynamically.  
* **Composite Pattern**: Groups multiple overlays into a single entity.  
* **Visitor Pattern**: Handles serialization, deserialization, and rendering operations.

---

## **8\. Dependencies**

* **Boost** (Serialization, utilities).  
* **OpenCV** (Drawing functions for overlays).  
* **Logging System** (Error handling & debugging).

## **9\. Summary**

| Feature | Description |
| ----- | :---- |
| **Overlay Types** | Circle, Line, Rectangle, Composite |
| **Factory Pattern** | Dynamically creates overlay objects |
| **Composite Pattern** | Groups multiple overlays |
| **Visitor Pattern** | Manages serialization & drawing |
| **Boost Serialization** | Efficient storage of overlay objects |
| **OpenCV Rendering** | Handles overlay drawing |

