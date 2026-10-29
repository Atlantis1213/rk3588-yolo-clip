param([string]$DocumentPath, [string]$PdfPath)
$ErrorActionPreference = 'Stop'
$wordForProject = $null
$projectDoc = $null
try {
    $wordForProject = New-Object -ComObject Word.Application
    $wordForProject.Visible = $false
    $wordForProject.DisplayAlerts = 0
    $projectDoc = $wordForProject.Documents.Open($DocumentPath, $false, $true, $false)
    $projectDoc.Fields.Update() | Out-Null
    $projectDoc.Repaginate()
    $pageCountForProject = $projectDoc.ComputeStatistics(2)
    $projectDoc.ExportAsFixedFormat($PdfPath, 17)
    Write-Output "Word rendered pages: $pageCountForProject"
} finally {
    if ($null -ne $projectDoc) { $projectDoc.Close(0) }
    if ($null -ne $wordForProject) {
        if ($wordForProject.Documents.Count -eq 0) { $wordForProject.Quit() }
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordForProject) | Out-Null
    }
}
