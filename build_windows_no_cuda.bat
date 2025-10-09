@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM ApraPipes Windows No-CUDA Build Script with Component Selection
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
    echo ApraPipes Windows No-CUDA Build Script with Component Selection
    echo ==============================================================
    echo.
    echo Usage: build_windows_no_cuda.bat [OPTIONS]
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
    echo   full                 All components (default, ~40-60 min^)
    echo.
    echo Available Components (no CUDA/ARM64^):
    echo   CORE                 Pipeline infrastructure (always required^)
    echo   VIDEO                Mp4, H264, RTSP
    echo   IMAGE_PROCESSING     OpenCV CPU-based processing
    echo   WEBCAM               Webcam capture
    echo   QR                   QR code reading
    echo   AUDIO                Audio capture and transcription
    echo   FACE_DETECTION       Face detection and landmarks
    echo   THUMBNAIL            Thumbnail generation
    echo   IMAGE_VIEWER         Image viewing GUI
    echo.
    echo Examples:
    echo   build_windows_no_cuda.bat
    echo   build_windows_no_cuda.bat --preset minimal
    echo   build_windows_no_cuda.bat --preset video
    echo   build_windows_no_cuda.bat --components "CORE;VIDEO;IMAGE_PROCESSING"
    echo.
    exit /b 0
)

REM Apply presets
IF DEFINED PRESET (
    IF /I "!PRESET!"=="minimal" SET COMPONENTS=CORE
    IF /I "!PRESET!"=="video" SET COMPONENTS=CORE;VIDEO;IMAGE_PROCESSING
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
echo Building ApraPipes (No CUDA^) with Components: !COMPONENTS!
echo ============================================================================
echo.

set batdir=%~dp0
cd %batdir%/build_scripts
powershell -nologo -executionpolicy bypass -File build_dependencies_windows_no_cuda.ps1
cd ..

@echo off
set batdir=%~dp0
cd %batdir%/base
powershell -nologo -executionpolicy bypass -File fix-vcpkg-json.ps1 -removeCUDA
cd ..

IF %BUILD_DOC%==1 (
    @echo off
    sh .\build_documentation.sh
)

cd vcpkg
call bootstrap-vcpkg.bat
vcpkg.exe integrate install
cd ..

mkdir _buildNoCuda
cd _buildNoCuda
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DENABLE_COMPONENTS=!COMPONENTS! -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base
cmake --build . --config RelWithDebInfo
cd ..

rem goto :EOF
mkdir _debugbuildNoCuda
cd _debugbuildNoCuda
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DENABLE_COMPONENTS=!COMPONENTS! -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base
cmake --build . --config Debug