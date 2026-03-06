$shortcutsPath = Join-Path $PSScriptRoot "shortcuts.yaml"

if (Test-Path $shortcutsPath) {
    Write-Host "`nLoading project shortcuts from shortcuts.yaml..." -ForegroundColor Cyan
    Get-Content $shortcutsPath | Where-Object { $_ -match '^([^:]+):\s*(.*)$' } | ForEach-Object {
        $name = $matches[1].Trim()
        $cmd = $matches[2].Trim()
        
        if ($name -and $name -notmatch "^#") {
            # Create a dynamic global function for the shortcut
            $functionCode = "function global:$name { $cmd `@args }"
            Invoke-Expression $functionCode
            
            Write-Host "  -> $name `($cmd`)" -ForegroundColor DarkGray
        }
    }
    
    $profileScript = $MyInvocation.MyCommand.Path
    $reloadCode = "function global:reload { Write-Host `"Reloading shortcuts...`" -ForegroundColor Cyan; . `"$profileScript`" }"
    Invoke-Expression $reloadCode
    Write-Host "  -> reload (reloads this script)" -ForegroundColor DarkGray
    
    Write-Host ""
}
