#inplace fixing of a vcpkg file
param([String]$fileName='vcpkg.json', [switch]$removeOpenCV,  [switch]$removeCUDA, [switch]$onlyOpenCV)

$v = Get-Content $fileName -raw | ConvertFrom-Json

if ($removeCUDA.IsPresent)
{
    # Remove CUDA features from dependencies
    $v.dependencies |
        Where-Object { $_.name -eq 'opencv4' } |
        ForEach-Object { $_.features = $_.features -ne 'cuda' -ne 'cudnn' }

    $v.dependencies |
        Where-Object { $_.name -eq 'whisper' } |
        ForEach-Object { $_.features = $_.features -ne 'cuda' }

    # Remove CUDA from features section
    if ($v.features) {
        # Remove 'cuda' from the 'all' feature's dependencies
        if ($v.features.all -and $v.features.all.dependencies) {
            foreach ($dep in $v.features.all.dependencies) {
                if ($dep.PSObject.Properties['name'] -and $dep.name -eq 'apra-pipes' -and $dep.features) {
                    $dep.features = @($dep.features | Where-Object { $_ -ne 'cuda' })
                }
            }
        }

        # Remove CUDA feature from audio feature's whisper dependency
        if ($v.features.audio -and $v.features.audio.dependencies) {
            $v.features.audio.dependencies |
                Where-Object { $_.PSObject.Properties['name'] -and $_.name -eq 'whisper' } |
                ForEach-Object {
                    if ($_.features) {
                        $_.features = @($_.features | Where-Object { $_ -ne 'cuda' })
                    }
                }
        }
    }
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
