[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$IncludeReferenceRepos
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$searchRoots = @(
    $repoRoot,
    (Join-Path $repoRoot ".scratch")
)

if ($IncludeReferenceRepos) {
    $searchRoots += @(
        (Join-Path $repoRoot "graphiti-main"),
        (Join-Path $repoRoot "cognee-main")
    )
}

$targets = New-Object System.Collections.Generic.List[string]
$scratchPytestTemp = Join-Path $repoRoot ".scratch\pytest-temp"
if (Test-Path -LiteralPath $scratchPytestTemp) {
    $targets.Add((Resolve-Path -LiteralPath $scratchPytestTemp).Path)
}

foreach ($root in $searchRoots) {
    if (-not (Test-Path -LiteralPath $root)) {
        continue
    }
    Get-ChildItem -LiteralPath $root -Directory -Filter "pytest-cache-files-*" -Force |
        ForEach-Object {
            $targets.Add($_.FullName)
        }
}

$existingTargets = $targets |
    Sort-Object -Unique |
    Where-Object { Test-Path -LiteralPath $_ }

foreach ($target in $existingTargets) {
    $resolved = (Resolve-Path -LiteralPath $target).Path
    if (-not $resolved.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside repo root: $resolved"
    }
}

if (-not $existingTargets) {
    Write-Host "No blocked pytest temp directories remain."
    return
}

$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$currentPrincipal = [Security.Principal.WindowsPrincipal]::new($currentIdentity)
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin -and -not $WhatIfPreference) {
    throw "This script must be run from an elevated Administrator PowerShell. Use -WhatIf to preview targets without elevation."
}

$results = foreach ($target in $existingTargets) {
    $resolved = (Resolve-Path -LiteralPath $target).Path
    $takeownOk = $false
    $grantOk = $false
    $removedOk = $false
    $message = ""

    if ($PSCmdlet.ShouldProcess($resolved, "take ownership, grant Administrators full control, and remove")) {
        & takeown.exe /F $resolved /R /D Y | Out-Null
        $takeownOk = ($LASTEXITCODE -eq 0)

        & icacls.exe $resolved /grant "*S-1-5-32-544:(OI)(CI)F" /T /C | Out-Null
        $grantOk = ($LASTEXITCODE -eq 0)

        try {
            Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Stop
            $removedOk = $true
            $message = "removed"
        } catch {
            $message = $_.Exception.Message
        }
    } else {
        $message = "skipped"
    }

    [pscustomobject]@{
        Target = $resolved
        Takeown = $takeownOk
        Icacls = $grantOk
        Removed = $removedOk
        Message = $message
    }
}

$results | Format-Table -AutoSize
