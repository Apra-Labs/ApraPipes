@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM ApraPipes Windows CUDA Build Script with Component Selection
REM ============================================================================

REM Parse command line arguments
SET BUILD_DOC=0
SET COMPONENTS=
SET SHOW_HELP=0
SET PRESET=

:parse_args
IF "%~1"=="" GOTO args_done
IF /I "%~1"=="--help" SET SHOW_HELP=1
IF /I "%~1"=="-h" SET SHOW_HELP=1
IF /I "%~1"=="--build-doc" SET BUILD_DOC=1
IF /I "%~1"=="--components" (
    SET COMPONENTS=%~2
    SHIFT
)
IF /I "%~1"=="--preset" (
    SET PRESET=%~2
    SHIFT
)
SHIFT
GOTO parse_args

:args_done

REM Show help if requested
IF %SHOW_HELP%==1 (
    echo.
    echo ApraPipes Windows CUDA Build Script with Component Selection
    echo ===========================================================
    echo.
    echo Usage: build_windows_cuda.bat [OPTIONS]
    echo.
    echo Options:
    echo   --help, -h            Show this help message
    echo   --build-doc           Build documentation after compilation
    echo   --components "LIST"   Specify components to build (semicolon-separated^)
    echo   --preset NAME         Use a preset configuration
    echo.
    echo Available Presets:
    echo   minimal              CORE only (~5-10 min build^)
    echo   video                CORE + VIDEO + IMAGE_PROCESSING (~15-25 min^)
    echo   cuda                 CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT
    echo   full                 All components (default, ~60-90 min^)
    echo.
    echo Available Components:
    echo   CORE                 Pipeline infrastructure (always required^)
    echo   VIDEO                Mp4, H264, RTSP
    echo   IMAGE_PROCESSING     OpenCV CPU-based processing
    echo   CUDA_COMPONENT       GPU acceleration
    echo   WEBCAM               Webcam capture
    echo   QR                   QR code reading
    echo   AUDIO                Audio capture and transcription
    echo   FACE_DETECTION       Face detection and landmarks
    echo   THUMBNAIL            Thumbnail generation
    echo   IMAGE_VIEWER         Image viewing GUI
    echo.
    echo Examples:
    echo   build_windows_cuda.bat
    echo   build_windows_cuda.bat --preset minimal
    echo   build_windows_cuda.bat --preset video
    echo   build_windows_cuda.bat --components "CORE;VIDEO;IMAGE_PROCESSING"
    echo   build_windows_cuda.bat --components "CORE;CUDA_COMPONENT" --build-doc
    echo.
    exit /b 0
)

REM Apply presets
IF DEFINED PRESET (
    IF /I "!PRESET!"=="minimal" SET COMPONENTS=CORE
    IF /I "!PRESET!"=="video" SET COMPONENTS=CORE;VIDEO;IMAGE_PROCESSING
    IF /I "!PRESET!"=="cuda" SET COMPONENTS=CORE;VIDEO;IMAGE_PROCESSING;CUDA_COMPONENT
    IF /I "!PRESET!"=="full" SET COMPONENTS=ALL

    IF "!COMPONENTS!"=="" (
        echo ERROR: Unknown preset "!PRESET!"
        echo Use --help to see available presets
        exit /b 1
    )
)

REM Default to ALL if no components specified
IF NOT DEFINED COMPONENTS SET COMPONENTS=ALL

echo.
echo ============================================================================
echo Building ApraPipes with Components: !COMPONENTS!
echo ============================================================================
echo.

set batdir=%~dp0
cd %batdir%/build_scripts
powershell -nologo -executionpolicy bypass -File build_dependencies_windows_cuda.ps1
cd ..

IF %BUILD_DOC%==1 (
    @echo off
    sh .\build_documentation.sh
)

@echo off
set batdir=%~dp0
cd %batdir%/vcpkg

call bootstrap-vcpkg.bat
@echo on
vcpkg.exe integrate install
cd ..

@echo off

REM Detect CUDA version and select appropriate Visual Studio version
SET VS_GENERATOR=
SET CUDA_VERSION_FILE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\version.json

REM Check if CUDA 11.8 is installed
IF EXIST "%CUDA_VERSION_FILE%" (
    echo Detected CUDA 11.8 - checking Visual Studio compatibility...

    REM CUDA 11.8 requires VS 2019 (or VS 2022 up to v17.3)
    REM Check for VS 2019 first (most compatible)
    IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community" (
        echo Using Visual Studio 2019 Community for CUDA 11.8 compatibility
        SET VS_GENERATOR=Visual Studio 16 2019
        SET VCPKG_PLATFORM_TOOLSET=v142
    ) ELSE IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional" (
        echo Using Visual Studio 2019 Professional for CUDA 11.8 compatibility
        SET VS_GENERATOR=Visual Studio 16 2019
        SET VCPKG_PLATFORM_TOOLSET=v142
    ) ELSE IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise" (
        echo Using Visual Studio 2019 Enterprise for CUDA 11.8 compatibility
        SET VS_GENERATOR=Visual Studio 16 2019
        SET VCPKG_PLATFORM_TOOLSET=v142
    ) ELSE (
        REM VS 2019 not found, check for compatible VS 2022 version
        echo Visual Studio 2019 not found, checking for compatible VS 2022...

        REM Check VS 2022 version using vswhere
        SET "VS2022_PATH="
        FOR /F "tokens=*" %%i IN ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version "[17.0,17.4)" -latest -property installationPath 2^>nul') DO SET "VS2022_PATH=%%i"

        IF DEFINED VS2022_PATH (
            REM Get the exact version
            FOR /F "tokens=*" %%i IN ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version "[17.0,17.4)" -latest -property installationVersion 2^>nul') DO SET "VS2022_VERSION=%%i"
            echo Found Visual Studio 2022 version !VS2022_VERSION!
            echo Using Visual Studio 2022 for CUDA 11.8 ^(compatible up to v17.3^)
            SET VS_GENERATOR=Visual Studio 17 2022
            SET VCPKG_PLATFORM_TOOLSET=v143
        ) ELSE (
            echo WARNING: CUDA 11.8 detected but no compatible Visual Studio found
            echo CUDA 11.8 requires:
            echo   - Visual Studio 2019 ^(any version^), OR
            echo   - Visual Studio 2022 v17.0 - v17.3
            echo Your VS 2022 version may be too new ^(^>v17.3^)
            echo Attempting to use default Visual Studio generator...
        )
    )
)

REM If no VS generator set, let CMake auto-detect
IF "%VS_GENERATOR%"=="" (
    echo Using CMake default Visual Studio generator
)

SET VCPKG_ARGS=-DENABLE_CUDA=ON -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DENABLE_COMPONENTS=!COMPONENTS! -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base

@echo on
mkdir _build
cd _build
IF "%VS_GENERATOR%"=="" (
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo %VCPKG_ARGS%
) ELSE (
    cmake -G "%VS_GENERATOR%" -DCMAKE_BUILD_TYPE=RelWithDebInfo %VCPKG_ARGS%
)
cmake --build . --config RelWithDebInfo
cd ..
rem goto :EOF
mkdir _debugbuild
cd _debugbuild
IF "%VS_GENERATOR%"=="" (
    cmake -DCMAKE_BUILD_TYPE=Debug %VCPKG_ARGS%
) ELSE (
    cmake -G "%VS_GENERATOR%" -DCMAKE_BUILD_TYPE=Debug %VCPKG_ARGS%
)
cmake --build . --config Debug