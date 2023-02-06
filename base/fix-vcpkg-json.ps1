#inplace fixing of a vcpkg file
param([String]$fileName='vcpkg.json', [switch]$removeOpenCV,  [switch]$removeCUDA, [switch]$onlyOpenCV)

$v = Get-Content $fileName -raw | ConvertFrom-Json

if($removeCUDA.IsPresent)
{
    $opencv = $v.dependencies | Where-Object { $_.name -eq 'opencv4'}
    $opencv.features= $opencv.features | Where-Object { $_ -ne 'cuda' }
    $opencv.features= $opencv.features | Where-Object { $_ -ne 'cudnn' }
}

if($removeOpenCV.IsPresent)
{
    $v.dependencies = $v.dependencies | Where-Object { $_.name -ne 'opencv4'}
}

if($onlyOpenCV.IsPresent)
{
    $opencv = $v.dependencies | Where-Object { $_.name -eq 'opencv4'}
    $v.dependencies = @()
    $v.dependencies += $opencv
    
}


$v | ConvertTo-Json -depth 32| set-content $fileName