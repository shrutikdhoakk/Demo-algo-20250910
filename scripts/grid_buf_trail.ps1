param(
  [string]$Uni = ".\data\symbols_nifty500_clean.csv",
  [string]$Cfg = ".\backtest\config.yaml",
  [string]$Start = "2022-01-01",
  [string]$End   = "2024-12-31",
  [int]$MaxPos   = 3
)

$env:RS_LOOKBACK_D   = "84"
$env:RS_TOP_PCT      = "0.60"
$env:REGIME_ON       = "1"
$env:REGIME_LEN_D    = "200"
$env:REGIME_SLOPE_WIN= "20"
$env:ATR_PCT_MAX     = "0.055"

$bufs   = @(0.35, 0.40, 0.45, 0.50)
$trails = @(1.10, 1.20, 1.30, 1.40, 1.50)

$outCsv = ".\runs\grid_buf_trail.csv"
"buf,trail,cagr,maxdd,mar" | Out-File $outCsv -Encoding utf8

$re = [regex]'CAGR:\s*([-0-9\.]+)%,\s*Max Drawdown:\s*([-0-9\.]+)%'

foreach ($b in $bufs) {
  foreach ($t in $trails) {
    $env:BREAKOUT_ATR_BUF = "$b"
    $env:TRAIL_ATR_MULT   = "$t"

    $txt = (python -m backtest --start $Start --end $End `
              --universe $Uni --max-pos $MaxPos --config $Cfg) | Out-String

    if ($re.IsMatch($txt)) {
      $m = $re.Match($txt)
      $cagr = [double]$m.Groups[1].Value
      $dd   = [double]$m.Groups[2].Value      # negative value
      $absdd = [math]::Abs($dd)
      $mar = if ($absdd -ne 0) { $cagr / $absdd } else { 0 }
      "{0},{1},{2},{3},{4}" -f $b, $t, $cagr, $dd, $mar | Add-Content $outCsv
      Write-Host ("buf={0} trail={1} -> CAGR {2}% | DD {3}% | MAR {4:N3}" -f $b,$t,$cagr,$dd,$mar)
    } else {
      Write-Warning "No metrics parsed for buf=$b trail=$t"
    }
  }
}

# Pick top 3 with |DD| <= 10%
$rows = Import-Csv $outCsv | ForEach-Object {
  $_ | Add-Member -NotePropertyName absdd -NotePropertyValue ([math]::Abs([double]$_.maxdd)) -PassThru
} | Where-Object { $_.absdd -le 10 }

$top3 = $rows | Sort-Object {[double]$_.mar}, {[double]$_.cagr} -Descending | Select-Object -First 3
"--- Top 3 (DD<=10%) by MAR ---"
$top3 | Format-Table buf,trail,cagr,maxdd,mar -Auto

# Save shortlist
$topOut = ".\runs\grid_shortlist_top3.csv"
"buf,trail,cagr,maxdd,mar" | Out-File $topOut -Encoding utf8
$top3 | ForEach-Object {
  "{0},{1},{2},{3},{4}" -f $_.buf,$_.trail,$_.cagr,$_.maxdd,$_.mar | Add-Content $topOut
}
