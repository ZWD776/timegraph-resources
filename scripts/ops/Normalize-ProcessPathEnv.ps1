[CmdletBinding()]
param(
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

function Get-ProcessPathEntries {
    [Environment]::GetEnvironmentVariables("Process").GetEnumerator() |
        Where-Object { [string]$_.Key -ieq "Path" } |
        Sort-Object Key
}

$entries = @(Get-ProcessPathEntries)
if ($entries.Count -eq 0) {
    return
}

$preferred = $entries | Where-Object { [string]$_.Key -ceq "Path" } | Select-Object -First 1
if ($null -eq $preferred) {
    $preferred = $entries |
        Sort-Object @{ Expression = { ([string]$_.Value).Length }; Descending = $true } |
        Select-Object -First 1
}

$preferredValue = [string]$preferred.Value
$beforeKeys = @($entries | ForEach-Object { [string]$_.Key })

foreach ($entry in $entries) {
    [Environment]::SetEnvironmentVariable([string]$entry.Key, $null, "Process")
}
[Environment]::SetEnvironmentVariable("Path", $preferredValue, "Process")

if (-not $Quiet) {
    $afterKeys = @(Get-ProcessPathEntries | ForEach-Object { [string]$_.Key })
    [pscustomobject]@{
        BeforeKeys = ($beforeKeys -join ",")
        AfterKeys = ($afterKeys -join ",")
        PathLength = $preferredValue.Length
    }
}
