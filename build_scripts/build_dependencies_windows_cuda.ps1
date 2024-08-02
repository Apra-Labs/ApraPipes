# Set execution policy and install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force;
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Enable feature and install dependencies
choco feature enable -n allowEmptyChecksums
choco install 7zip git python3 cmake pkgconfiglite doxygen.portable graphviz -y

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
    $extractedPath = "C:\Users\$userName\Downloads\cudnn-*\"

    if (-not (Test-Path $extractedPath)) {
        Write-Host "Extracting zip file..."
        Expand-Archive -Path $zipFilePath -DestinationPath "C:\Users\$userName\Downloads\" -Force
    } else {
        Write-Host "Already extracted files found."
    }

    Write-Host "Copying files..."
    Copy-Item -Path "C:\Users\$userName\Downloads\cudnn-*\include\*.h" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cudaVersion\include\" -Recurse
    Copy-Item -Path "C:\Users\$userName\Downloads\cudnn-*\lib\*.lib" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cudaVersion\lib\x64\" -Recurse
    Copy-Item -Path "C:\Users\$userName\Downloads\cudnn-*\bin\*.dll" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$cudaVersion\bin\" -Recurse
}

Write-Host "Dependencies verified and installed successfully."