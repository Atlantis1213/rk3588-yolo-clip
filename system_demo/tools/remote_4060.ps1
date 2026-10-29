param(
  [Parameter(Mandatory=$true)][string]$Script,
  [string]$HostAddress = '127.0.0.1',
  [int]$Port = 2222,
  [string]$UserName = 'atlantis'
)
$taskPrelude = @'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$OutputEncoding = [Console]::OutputEncoding
'@
$taskRemoteCode = $taskPrelude + "`n" + $Script
$taskRemoteEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($taskRemoteCode))
& ssh -p $Port -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 "${UserName}@${HostAddress}" powershell.exe -NoProfile -NonInteractive -OutputFormat Text -EncodedCommand $taskRemoteEncoded
if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
