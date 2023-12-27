#inplace fixing of a vcpkg file
param([String]$fileName='vcpkg.json', [switch]$removeOpenCV,  [switch]$removeCUDA, [switch]$onlyOpenCV, [switch]$removeWhisper)

$v = Get-Content $fileName -raw | ConvertFrom-Json

if($removeWhisper.IsPresent){
    $result = $v.overrides | Where-Object { $_.name -ne 'whisper' } 
    # Check if the result is $null or empty and set it to an empty array if it is
    if (-not $result) {
        $result = @()
    }
    $v.overrides = $result
    $v.dependencies = $v.dependencies | Where-Object { $_ -ne 'whisper'}
}

if ($removeCUDA.IsPresent)
{
    $v.dependencies |
        Where-Object { $_.name -eq 'opencv4' } |
        ForEach-Object { $_.features = $_.features -ne 'cuda' -ne 'cudnn' }
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
