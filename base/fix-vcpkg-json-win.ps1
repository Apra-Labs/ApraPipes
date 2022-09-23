$v = Get-Content 'vcpkg.nocuda.json' -raw | ConvertFrom-Json
$v.dependencies = $v.dependencies | Where-Object { $_.name -ne 'opencv4'}
$v | ConvertTo-Json -depth 32| set-content 'vcpkg.nocuda.json'