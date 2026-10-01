$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$pythonw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
if (-not $pythonw) {
    $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    if ($python) { $pythonw = Join-Path (Split-Path $python) "pythonw.exe" }
}
if (-not $pythonw -or -not (Test-Path -LiteralPath $pythonw)) {
    throw "pythonw.exe was not found. Install Python or add it to PATH."
}
$appData = [Environment]::GetFolderPath("ApplicationData")
$startMenu = Join-Path $appData "Microsoft\Windows\Start Menu\Programs"
$shortcutPath = Join-Path $startMenu "KLine Similarity Search.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = ('"{0}"' -f (Join-Path $scriptDir "launch_ui.py"))
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = "Open K-line historical similarity search"
$shortcut.IconLocation = "$pythonw,0"
$shortcut.Save()
Write-Host "Start Menu shortcut created: $shortcutPath"
