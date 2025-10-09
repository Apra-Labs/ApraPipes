# ApraPipes Component Dependency Diagram

This document provides visual representations of the component dependency structure in ApraPipes.

---

## High-Level Component Dependencies

```mermaid
graph TD
    CORE[CORE<br/>Pipeline Infrastructure<br/>~5-10 min]

    VIDEO[VIDEO<br/>Mp4, H264, RTSP<br/>Depends: CORE]
    IMAGE_PROCESSING[IMAGE_PROCESSING<br/>OpenCV CPU Processing<br/>Depends: CORE]
    WEBCAM[WEBCAM<br/>Camera Capture<br/>Depends: CORE, IMAGE_PROCESSING]
    QR[QR<br/>QR Code Reading<br/>Depends: CORE, IMAGE_PROCESSING]
    FACE_DETECTION[FACE_DETECTION<br/>Face Detection<br/>Depends: CORE, IMAGE_PROCESSING]
    THUMBNAIL[THUMBNAIL<br/>Thumbnail Generation<br/>Depends: CORE, IMAGE_PROCESSING]
    IMAGE_VIEWER[IMAGE_VIEWER<br/>Image Viewing GUI<br/>Depends: CORE, IMAGE_PROCESSING]
    GTK_RENDERING[GTK_RENDERING<br/>Linux GUI Rendering<br/>Depends: CORE, IMAGE_PROCESSING]

    CUDA_COMPONENT[CUDA_COMPONENT<br/>GPU Acceleration<br/>Depends: CORE, IMAGE_PROCESSING]
    ARM64_COMPONENT[ARM64_COMPONENT<br/>Jetson Hardware<br/>Depends: CORE, CUDA_COMPONENT]

    AUDIO[AUDIO<br/>Audio & Transcription<br/>Depends: CORE]

    CORE --> VIDEO
    CORE --> IMAGE_PROCESSING
    CORE --> AUDIO

    IMAGE_PROCESSING --> WEBCAM
    IMAGE_PROCESSING --> QR
    IMAGE_PROCESSING --> FACE_DETECTION
    IMAGE_PROCESSING --> THUMBNAIL
    IMAGE_PROCESSING --> IMAGE_VIEWER
    IMAGE_PROCESSING --> GTK_RENDERING
    IMAGE_PROCESSING --> CUDA_COMPONENT

    CUDA_COMPONENT --> ARM64_COMPONENT

    style CORE fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style VIDEO fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style IMAGE_PROCESSING fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style CUDA_COMPONENT fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    style ARM64_COMPONENT fill:#f8bbd0,stroke:#880e4f,stroke-width:2px
    style AUDIO fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px
    style WEBCAM fill:#e0e0e0,stroke:#424242,stroke-width:1px
    style QR fill:#e0e0e0,stroke:#424242,stroke-width:1px
    style FACE_DETECTION fill:#e0e0e0,stroke:#424242,stroke-width:1px
    style THUMBNAIL fill:#e0e0e0,stroke:#424242,stroke-width:1px
    style IMAGE_VIEWER fill:#e0e0e0,stroke:#424242,stroke-width:1px
    style GTK_RENDERING fill:#d1c4e9,stroke:#4a148c,stroke-width:1px
```

---

## Detailed Dependency Tree

```mermaid
graph LR
    subgraph "Foundation"
        CORE[CORE<br/>17-19 modules]
    end

    subgraph "Media I/O"
        VIDEO[VIDEO<br/>11 modules]
        AUDIO[AUDIO<br/>2 modules]
    end

    subgraph "CPU Processing"
        IMAGE_PROCESSING[IMAGE_PROCESSING<br/>17 modules]
        WEBCAM[WEBCAM<br/>1 module]
        QR[QR<br/>1 module]
        FACE_DETECTION[FACE_DETECTION<br/>2 modules]
        THUMBNAIL[THUMBNAIL<br/>1 module]
        IMAGE_VIEWER[IMAGE_VIEWER<br/>1 module]
    end

    subgraph "GPU Processing"
        CUDA_COMPONENT[CUDA_COMPONENT<br/>20 modules]
        ARM64_COMPONENT[ARM64_COMPONENT<br/>21 modules]
    end

    subgraph "Platform Specific"
        GTK_RENDERING[GTK_RENDERING<br/>6 modules<br/>Linux only]
    end

    CORE --> VIDEO
    CORE --> AUDIO
    CORE --> IMAGE_PROCESSING

    IMAGE_PROCESSING --> WEBCAM
    IMAGE_PROCESSING --> QR
    IMAGE_PROCESSING --> FACE_DETECTION
    IMAGE_PROCESSING --> THUMBNAIL
    IMAGE_PROCESSING --> IMAGE_VIEWER
    IMAGE_PROCESSING --> GTK_RENDERING
    IMAGE_PROCESSING --> CUDA_COMPONENT

    CORE --> CUDA_COMPONENT
    CUDA_COMPONENT --> ARM64_COMPONENT

    style CORE fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style VIDEO fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style AUDIO fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px
    style IMAGE_PROCESSING fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style CUDA_COMPONENT fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    style ARM64_COMPONENT fill:#f8bbd0,stroke:#880e4f,stroke-width:2px
    style GTK_RENDERING fill:#d1c4e9,stroke:#4a148c,stroke-width:1px
```

---

## Component Dependency Matrix

| Component | Depends On | Optional Add-Ons |
|-----------|-----------|------------------|
| **CORE** | _(none - always required)_ | CUDA allocators (if ENABLE_CUDA=ON) |
| **VIDEO** | CORE | - |
| **IMAGE_PROCESSING** | CORE | NPP libraries (if ENABLE_CUDA=ON) |
| **CUDA_COMPONENT** | CORE, IMAGE_PROCESSING | - |
| **ARM64_COMPONENT** | CORE, CUDA_COMPONENT | - |
| **WEBCAM** | CORE, IMAGE_PROCESSING | - |
| **QR** | CORE, IMAGE_PROCESSING | - |
| **AUDIO** | CORE | CUDA (for Whisper acceleration) |
| **FACE_DETECTION** | CORE, IMAGE_PROCESSING | - |
| **GTK_RENDERING** | CORE, IMAGE_PROCESSING | - |
| **THUMBNAIL** | CORE, IMAGE_PROCESSING | - |
| **IMAGE_VIEWER** | CORE, IMAGE_PROCESSING | - |

---

## Common Component Combinations

```mermaid
graph TB
    subgraph "Minimal Build<br/>~5-10 min"
        M_CORE[CORE]
    end

    subgraph "Video Processing<br/>~15-25 min"
        V_CORE[CORE]
        V_VIDEO[VIDEO]
        V_IP[IMAGE_PROCESSING]

        V_CORE --> V_VIDEO
        V_CORE --> V_IP
    end

    subgraph "CUDA Accelerated<br/>~30-40 min"
        C_CORE[CORE]
        C_VIDEO[VIDEO]
        C_IP[IMAGE_PROCESSING]
        C_CUDA[CUDA_COMPONENT]

        C_CORE --> C_VIDEO
        C_CORE --> C_IP
        C_IP --> C_CUDA
    end

    subgraph "Jetson Platform<br/>~60-90 min"
        J_CORE[CORE]
        J_VIDEO[VIDEO]
        J_IP[IMAGE_PROCESSING]
        J_CUDA[CUDA_COMPONENT]
        J_ARM64[ARM64_COMPONENT]

        J_CORE --> J_VIDEO
        J_CORE --> J_IP
        J_IP --> J_CUDA
        J_CUDA --> J_ARM64
    end

    style M_CORE fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style V_CORE fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style V_VIDEO fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style V_IP fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style C_CORE fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style C_VIDEO fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style C_IP fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style C_CUDA fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    style J_CORE fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style J_VIDEO fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style J_IP fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style J_CUDA fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    style J_ARM64 fill:#f8bbd0,stroke:#880e4f,stroke-width:2px
```

---

## Dependency Legend

- **Blue (CORE)**: Foundation layer - always required
- **Yellow (VIDEO, IMAGE_PROCESSING)**: Media processing components
- **Orange (CUDA_COMPONENT)**: GPU acceleration
- **Pink (ARM64_COMPONENT)**: Jetson-specific hardware
- **Green (AUDIO)**: Audio processing
- **Purple (GTK_RENDERING)**: Linux-specific rendering
- **Gray (Specialized)**: Specialized single-purpose components

---

## Platform-Specific Dependencies

```mermaid
graph TD
    subgraph "Windows"
        W_CORE[CORE]
        W_VIDEO[VIDEO]
        W_IP[IMAGE_PROCESSING]
        W_CUDA[CUDA_COMPONENT]
        W_WEBCAM[WEBCAM]
        W_QR[QR]
        W_AUDIO[AUDIO]
        W_FACE[FACE_DETECTION]
        W_THUMB[THUMBNAIL]
        W_VIEWER[IMAGE_VIEWER]

        W_CORE --> W_VIDEO
        W_CORE --> W_IP
        W_IP --> W_CUDA
        W_IP --> W_WEBCAM
        W_IP --> W_QR
        W_IP --> W_FACE
        W_IP --> W_THUMB
        W_IP --> W_VIEWER
        W_CORE --> W_AUDIO
    end

    subgraph "Linux x86"
        L_CORE[CORE]
        L_VIDEO[VIDEO]
        L_IP[IMAGE_PROCESSING]
        L_CUDA[CUDA_COMPONENT]
        L_GTK[GTK_RENDERING]
        L_WEBCAM[WEBCAM]
        L_QR[QR]
        L_AUDIO[AUDIO]
        L_FACE[FACE_DETECTION]
        L_THUMB[THUMBNAIL]
        L_VIEWER[IMAGE_VIEWER]

        L_CORE --> L_VIDEO
        L_CORE --> L_IP
        L_IP --> L_CUDA
        L_IP --> L_GTK
        L_IP --> L_WEBCAM
        L_IP --> L_QR
        L_IP --> L_FACE
        L_IP --> L_THUMB
        L_IP --> L_VIEWER
        L_CORE --> L_AUDIO
    end

    subgraph "Jetson ARM64"
        A_CORE[CORE]
        A_VIDEO[VIDEO]
        A_IP[IMAGE_PROCESSING]
        A_CUDA[CUDA_COMPONENT]
        A_ARM64[ARM64_COMPONENT]
        A_WEBCAM[WEBCAM]
        A_QR[QR]
        A_AUDIO[AUDIO]
        A_FACE[FACE_DETECTION]

        A_CORE --> A_VIDEO
        A_CORE --> A_IP
        A_IP --> A_CUDA
        A_CUDA --> A_ARM64
        A_IP --> A_WEBCAM
        A_IP --> A_QR
        A_IP --> A_FACE
        A_CORE --> A_AUDIO
    end
```

---

## Notes

1. **CORE** is always required and serves as the foundation
2. **IMAGE_PROCESSING** is a common dependency for most specialized components
3. **CUDA_COMPONENT** requires IMAGE_PROCESSING due to shared infrastructure (NPP libraries)
4. **ARM64_COMPONENT** is only available on Jetson platforms and requires CUDA_COMPONENT
5. **GTK_RENDERING** is only available on Linux platforms
6. Components can be enabled independently as long as their dependencies are satisfied

For detailed component information, see [COMPONENTS_GUIDE.md](COMPONENTS_GUIDE.md).
