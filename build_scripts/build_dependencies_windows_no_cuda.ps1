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
pip3 install cmake==3.30.0

Write-Host "Dependencies verified and installed successfully."
