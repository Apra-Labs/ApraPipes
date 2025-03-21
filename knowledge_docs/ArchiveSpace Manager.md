**ARCHIVESPACE MANAGER MODULE**

**OVERVIEW**

The ArchiveSpaceManager is a module within the ApraPipes framework designed to manage disk space in a specified directory by monitoring and controlling the size of archived content. It ensures that the disk usage remains within defined boundaries (lower and upper watermarks) by periodically sampling directory sizes and deleting older directories when necessary. This module is particularly useful for applications requiring efficient storage management, such as image and video processing pipelines.

---

**CLASS STRUCTURE**

### **ArchiveSpaceManagerProps**

The properties class for configuring the ArchiveSpaceManager module.

**Watermark Constructor** : 

| ArchiveSpaceManagerProps(uint32\_t \_lowerWaterMark, uint32\_t \_upperWaterMark, string \_pathToWatch, int \_samplingFreq)  |
| :---- |

**Parameters**:

* \_lowerWaterMark: Minimum disk space threshold (in bytes) below which cleanup stops.  
* \_upperWaterMark: Maximum disk space threshold (in bytes) above which cleanup is triggered.  
* \_pathToWatch: Directory path to monitor and manage.  
* \_samplingFreq: Frequency (in number of files) for sampling file sizes to estimate directory size.

**Behavior**:  Validates that \_lowerWaterMark ≤ \_upperWaterMark and \_upperWaterMark ≤ total disk capacity. Throws an AIPException if invalid.

**Size Constructor :** 

| ArchiveSpaceManagerProps(uint32\_t maxSizeAllowed, string \_pathToWatch, int \_samplingFreq) |
| :---- |

**Parameters**:

* \_maxSizeAllowed: Maximum allowed size (in bytes); sets \_upperWaterMark to this value and \_lowerWaterMark to 90% of this value.  
* \_pathToWatch: Directory path to monitor and manage.  
* \_samplingFreq: Frequency for sampling file sizes.

**Behavior**:  Validates that \_lowerWaterMark ≤ \_upperWaterMark and \_upperWaterMark ≤ total disk capacity. Throws an AIPException if invalid.

**Public Members:**

* uint32\_t lowerWaterMark: Lower disk space threshold.  
* uint32\_t upperWaterMark: Upper disk space threshold.  
* std::string pathToWatch: Path to the directory being monitored.  
* int samplingFreq: Sampling frequency for size estimation.

**Serialization:** Implements Boost serialization for persisting properties.

### **ArchiveSpaceManager**

The main module class that inherits from Module in the ApraPipes framework.

**Constructor :**

| ArchiveSpaceManager(ArchiveSpaceManagerProps \_props) |
| :---- |

**Parameters**: ArchiveSpaceManager Props

**Behavior**:  Initializes the module as a SOURCE type with the name "ArchiveSpaceManager"

**Public Methods  :**

* bool init(): Initializes the module. Returns false if base Module::init() fails.  
* bool term(): Terminates the module cleanly.  
* ArchiveSpaceManagerProps getProps(): Retrieves current properties.  
* void setProps(ArchiveSpaceManagerProps& props): Queues new properties for update.  
* uint32\_t finalArchiveSpace: Stores the last computed archive size (publicly accessible).

**Public Methods  :**

* bool process(): Core logic to estimate and manage disk space. Updates finalArchiveSpace.  
* bool validateInputPins(), bool validateOutputPins(), bool validateInputOutputPins(): Validation methods (always return true in this implementation).  
* void addInputPin(framemetadata\_sp& metadata, string& pinId): Adds input and corresponding output pins.  
* bool handlePropsChange(frame\_sp& frame): Handles property updates dynamically.

---

**DETAIL IMPLEMENTATION (Private)**

### **ArchiveSpaceManager::Detail**

A private helper class encapsulating the core disk management logic.

**Public Methods  :**

**uint32\_t estimateDirectorySize(boost::filesystem::path \_dir)**

**Purpose**: Estimates the total size of a directory by sampling file sizes.

**Parameters**: \_dir: boost::filesystem::path to the directory to estimate.

**Return Value**: Estimated size in bytes (as uint32\_t).

**Behavior**:

* Iterates recursively through \_dir using boost::filesystem::recursive\_directory\_iterator.  
* For each regular file:  
  * Increments countFreq to track total files processed.  
  * Every mProps.samplingFreq files, randomly selects a sample file (sample) between 1 and samplingFreq.  
  * When inCount matches a sample, measures the file size (tempSize) and extrapolates it by multiplying by samplingFreq to estimate the contribution of that batch.  
* If the last batch has fewer files than samplingFreq, adds the remaining contribution (tempSize \* inCount).

**Usage**: Called by diskOperation to estimate the total size of pathToWatch and by manageDirectory to calculate sizes of subdirectories being deleted.

**boost::filesystem::path getOldestDirectory(boost::filesystem::path \_cam)**

**Purpose**: Identifies the oldest subdirectory containing files within a camera folder.

**Parameters**:

* \_cam: boost::filesystem::path to a camera folder (e.g., pathToWatch/Cam1).

**Return Value**: Path to the oldest subdirectory with files, or \_cam if none found.

**Behavior**:

* Iterates through immediate subdirectories of \_cam using directory\_iterator.  
* For each subdirectory, perform a recursive search with a recursive\_directory\_iterator.  
* Returns the parent directory of the first regular file encountered (assumed to be the oldest based on iteration order).  
* If no files are found, return the input path \_cam.

  

**void manageDirectory()**

**Purpose**: Reduces archiveSize below lowerWaterMark by deleting the oldest subdirectories.

**Behavior**:

* Defines a lambda comparator to sort by last\_write\_time (earlier times first).  
* While archiveSize exceeds lowerWaterMark:  
  * Iterates through immediate subdirectories of pathToWatch (e.g., camera folders).  
  * Calls getOldestDirectory for each and stores the path and its last\_write\_time in foldVector.  
  * Sorts foldVector by time.  
  * Selects the oldest directory (foldVector\[0\]), estimates its size, subtracts it from archiveSize, and deletes it with remove\_all.  
  * Logs the deletion; catches and logs exceptions if deletion fails.  
  * Clears foldVector for the next iteration.  
    

**uint32\_t diskOperation()**

**Purpose**: Main entry point for disk space management; estimates the size and triggers cleanup if needed.

**Return Value**: Size of the directory before any cleanup (in bytes).

**Behavior**:

* Estimates the total size of pathToWatch and stores it in archiveSize.  
* If archiveSize exceeds upperWaterMark, calls manageDirectory to reduce it below lowerWaterMark.  
* Stores the pre-cleanup size in tempSize, resets archiveSize to 0, and returns tempSize.


#### **Private Members**

* ArchiveSpaceManagerProps mProps: Stores the module properties.  
* uint32\_t archiveSize: Temporary storage for the estimated directory size.  
* std::vector\<std::pair\<boost::filesystem::path, uint32\_t\>\> foldVector: Holds directory paths and their last write times for sorting.

## **Functionality**

The ArchiveSpaceManager operates as follows:

1. **Initialization**: Configured with ArchiveSpaceManagerProps to set watermarks, path, and sampling frequency.  
2. **Monitoring**: Periodically estimates the size of the directory at pathToWatch using a sampling technique.  
3. **Management**: If the size exceeds upperWaterMark, it identifies and deletes the oldest directories until the size drops below lowerWaterMark.  
4. **Output**: Updates finalArchiveSpace with the size before any cleanup.

| \#include "ArchiveSpaceManager.h"    // Configure with explicit watermarks    ArchiveSpaceManagerProps props(50000000, 100000000, "/path/to/archive", 10);    ArchiveSpaceManager manager(props);        manager.init();    manager.process(); // Manages disk space    std::cout \<\< "Final archive size: " \<\< manager.finalArchiveSpace \<\< "\\n";    manager.term(); |
| :---- |

## 

| NotesThe sampling frequency (samplingFreq) trades off accuracy for performance; a higher value reduces computation but may overestimate/underestimate sizes.The module assumes write access to the monitored directory and its subdirectories. |
| :---- |

---

