param(
    [string]$Message = "Update poetru pipeline"
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

git add -A
$status = git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "Nothing to commit"
    exit 0
}

git commit -m $Message
git push origin HEAD
