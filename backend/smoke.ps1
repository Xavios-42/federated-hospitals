param([string]$BaseUrl = 'http://127.0.0.1:8080')

$ErrorActionPreference = 'Stop'
$health = Invoke-RestMethod "$BaseUrl/api/health"
if ($health.database -ne 'UP' -or $health.mlEngine -ne 'UP') {
    throw 'Expected MySQL and the Python engine to be ready'
}

$config = @{
    algorithm = 'fedprox'
    mode = 'non-iid'
    seed = 42
    rounds = 2
    epochs = 1
    batchSize = 32
    learningRate = 0.01
    mu = 0.1
} | ConvertTo-Json -Compress
$accepted = Invoke-RestMethod "$BaseUrl/api/experiments" -Method Post -ContentType 'application/json' -Body $config
if ($accepted.status -ne 'QUEUED') { throw 'Expected a queued experiment' }
$runUrl = "$BaseUrl/api/experiments/$($accepted.runId)"
$deadline = (Get-Date).AddMinutes(2)
do {
    $run = Invoke-RestMethod $runUrl
    if ($run.status -eq 'FAILED') { throw "Experiment failed ($($run.errorCode))" }
    if ($run.status -eq 'COMPLETED') { break }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)
if ($run.status -ne 'COMPLETED' -or $run.roundCount -ne 2) { throw 'Expected two completed rounds' }

$report = Invoke-RestMethod "$runUrl/metrics"
if ($report.history.Count -ne 2 -or $report.finalMetrics.test.samples -ne 114) {
    throw 'Unexpected metrics response'
}
$jsonExport = Invoke-RestMethod "$runUrl/export"
$csvExport = Invoke-RestMethod "$runUrl/export?format=csv"
if ($jsonExport.experiment.runId -ne $accepted.runId -or $csvExport -notmatch 'test,2,114,') {
    throw 'Unexpected aggregate exports'
}
[pscustomobject]@{
    RunId = $accepted.runId
    Status = $run.status
    RoundCount = $run.roundCount
    TestSamples = $report.finalMetrics.test.samples
    ModelBytes = $report.finalMetrics.communication.total_model_bytes
}
