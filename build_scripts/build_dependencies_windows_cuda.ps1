# Set execution policy and install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force;
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Enable feature and install dependencies
choco feature enable -n allowEmptyChecksums
choco install 7zip git python3 cmake pkgconfiglite doxygen.install graphviz -y

# Specify the new path to be added to the PATH environment variable
$doxygenPath = "C:\Program Files\doxygen\bin"

# Get the current user's environment variables
$envPath = [System.Environment]::GetEnvironmentVariable("Path", "User")

# Check if the new path already exists in the environment variable
if ($envPath -split ";" -notcontains $doxygenPath) {
    # Append the new path to the existing PATH environment variable
    $newEnvPath = $envPath + ";" + $doxygenPath
    # Update the environment variable
    [System.Environment]::SetEnvironmentVariable("Path", $newEnvPath, "User")
    Write-Host "Path updated successfully."
} else {
    Write-Host "Path already exists in the environment variable."
}

Import-Module $env:ChocolateyInstall\helpers\chocolateyProfile.psm1
refreshenv

doxygen --version

# Install required Python packages
pip3 install ninja
pip3 install meson

# Install cmake
pip3 install cmake --upgrade

$cudaVersion = (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\" -Directory | Where-Object { $_.Name -match 'v\d+\.\d+' }).Name

if (-not (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cudaVersion\bin\nvcc.exe")) {
    Write-Host "CUDA Toolkit is not installed. Please install CUDA Toolkit and CuDNN as specified in the Read-me."
    Exit 1
} else {
    $userName = $env:UserName
    $zipFilePath = "C:\Users\$userName\Downloads\cudnn-*.zip"

    Write-Host "Extracting zip file..."
    
    Expand-Archive -Path $zipFilePath -DestinationPath C:\Users\$userName\Downloads\ -Force

    Write-Host "Copying files..."
    Copy-Item -Path "C:\Users\$userName\Downloads\cudnn-*\include\*.h" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cudaVersion\include\" -Recurse
    Copy-Item -Path "C:\Users\$userName\Downloads\cudnn-*\lib\*.lib" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cudaVersion\lib\x64\" -Recurse
    Copy-Item -Path "C:\Users\$userName\Downloads\cudnn-*\bin\*.dll" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cudaVersion\bin\" -Recurse
}

Write-Host "Dependencies verified and installed successfully."

# Build Documentation
cd ..
sh .\build_documentation.sh
cd build_scripts