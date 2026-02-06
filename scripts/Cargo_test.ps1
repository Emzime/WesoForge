$logFile = "cargo_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$env:WESOFORGE_TEST_ITERS="tiny"

Write-Host "Début des tests - Log: $logFile" -ForegroundColor Yellow

# Exécute et log
cargo clean
cls
cargo test -- --nocapture 2>&1 | Tee-Object -FilePath $logFile

Write-Host "`nLog enregistré dans: $logFile" -ForegroundColor Green
