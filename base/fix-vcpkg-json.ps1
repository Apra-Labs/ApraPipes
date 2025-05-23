#inplace fixing of a vcpkg file
param([String]$fileName='vcpkg.json', [switch]$removeOpenCV,  [switch]$removeCUDA, [switch]$onlyOpenCV)

$v = Get-Content $fileName -raw | ConvertFrom-Json

if ($removeCUDA.IsPresent)
{
    $v.dependencies |
        Where-Object { $_.name -eq 'opencv4' } |
        ForEach-Object { $_.features = $_.features -ne 'cuda' -ne 'cudnn' }
    
    $v.dependencies |
        Where-Object { $_.name -eq 'whisper' } |
        ForEach-Object { $_.features = $_.features -ne 'cuda' }
    
    $v.dependencies |
        Where-Object { $_.name -eq 'llama' } |
        ForEach-Object { $_.features = $_.features -ne 'cuda' }
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
