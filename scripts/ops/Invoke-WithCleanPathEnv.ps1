[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [string]$CommandLine = "",

    [string[]]$ArgumentList = @(),

    [string]$WorkingDirectory = "",

    [string]$RedirectStandardOutput = "",

    [string]$RedirectStandardError = "",

    [switch]$Wait,

    [switch]$PassThru,

    [ValidateSet("Normal", "Hidden", "Minimized", "Maximized")]
    [string]$WindowStyle = "Hidden"
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\Normalize-ProcessPathEnv.ps1" -Quiet

$effectiveArgumentList = @()
if ($CommandLine) {
    $effectiveArgumentList += $CommandLine
}
if ($ArgumentList) {
    $effectiveArgumentList += $ArgumentList
}

$startParams = @{
    FilePath = $FilePath
    ArgumentList = $effectiveArgumentList
    WindowStyle = $WindowStyle
}

if ($WorkingDirectory) {
    $startParams.WorkingDirectory = $WorkingDirectory
}
if ($RedirectStandardOutput) {
    $startParams.RedirectStandardOutput = $RedirectStandardOutput
}
if ($RedirectStandardError) {
    $startParams.RedirectStandardError = $RedirectStandardError
}
if ($Wait) {
    $startParams.Wait = $true
}
if ($PassThru) {
    $startParams.PassThru = $true
}

Start-Process @startParams
