@echo off
set batdir=%~dp0
cd %batdir%/build_scripts
powershell -nologo -executionpolicy bypass -File build_dependencies_windows_cuda.ps1
cd ..

IF "%1"=="--build-doc" (
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
setlocal enabledelayedexpansion

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
        SET "VS_GENERATOR=-G "Visual Studio 16 2019""
        SET VCPKG_PLATFORM_TOOLSET=v142
    ) ELSE IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional" (
        echo Using Visual Studio 2019 Professional for CUDA 11.8 compatibility
        SET "VS_GENERATOR=-G "Visual Studio 16 2019""
        SET VCPKG_PLATFORM_TOOLSET=v142
    ) ELSE IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise" (
        echo Using Visual Studio 2019 Enterprise for CUDA 11.8 compatibility
        SET "VS_GENERATOR=-G "Visual Studio 16 2019""
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
            SET "VS_GENERATOR=-G "Visual Studio 17 2022""
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
    SET CMAKE_GENERATOR_ARG=
) ELSE (
    SET CMAKE_GENERATOR_ARG=%VS_GENERATOR%
)

SET VCPKG_ARGS=-DENABLE_CUDA=ON -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base

@echo on
mkdir _build
cd _build
cmake %CMAKE_GENERATOR_ARG% -DCMAKE_BUILD_TYPE=RelWithDebInfo %VCPKG_ARGS%
cmake --build . --config RelWithDebInfo
cd ..
rem goto :EOF
mkdir _debugbuild
cd _debugbuild
cmake %CMAKE_GENERATOR_ARG% -DCMAKE_BUILD_TYPE=Debug %VCPKG_ARGS%
cmake --build . --config Debug