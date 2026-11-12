# Federated Learning for Shared Models in Hospitals

A B.Tech research demonstration of federated learning across three simulated hospitals. The prototype is for academic use only. Simulated data locality offers no privacy guarantee, and model updates can disclose information about training records. [Research on model update inversion](https://proceedings.mlr.press/v202/zhu23m.html)

## Local setup

Use 64-bit Python 3.12 on Windows. Run these commands in PowerShell. The package installation requires internet access. The dataset ships with scikit-learn, so running the demo requires no dataset download.

```powershell
Set-Location 'D:\Projects\federated-hospitals'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml-engine/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
Set-Location ml-engine
..\.venv\Scripts\python.exe -m app --config configs/iid.json
..\.venv\Scripts\python.exe -m app --config configs/non-iid.json
..\.venv\Scripts\python.exe -m app --config configs/fedavg.json
..\.venv\Scripts\python.exe -m app --config configs/fedprox.json
..\.venv\Scripts\python.exe -m app --config configs/fedavgm.json
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Dataset and training protocol

- The [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) has 569 examples and 30 numeric features. The loader uses the [scikit-learn copy](https://scikit-learn.org/1.6/modules/generated/sklearn.datasets.load_breast_cancer.html) in its bundled feature order. Labels are mapped to benign = 0 and malignant = 1.
- A seeded, stratified 80/20 split first reserves 114 benchmark records. Only the 455 development records are assigned to hospitals. The CLI reports the held-out size and runs evaluation on local validation splits.
- IID assigns nearly equal shares of each class to A, B and C. The class-skewed configuration allocates approximately 20%, 33%, 47% of benign rows and 60%, 30%, 10% of malignant rows to A, B and C respectively. Both configurations use seed 42 and the same benchmark holdout.
- Each hospital makes its own stratified 80/20 training/validation split. The standalone configurations use a scaler fitted on the hospital's own training subset. Federated configurations apply the same fixed elementwise `log1p` transform to the nonnegative features at every site. The transform needs no fitted statistics and gives the averaged weights a shared feature coordinate system.
- All clients start with the same seeded 30-input logistic classifier. They independently train with minibatch SGD and binary cross-entropy with logits. The two checked-in configurations use 3 local epochs, batch size 16 and learning rate 0.01.
- The simulation provisions hospital data in memory in one Python process. A client's update contains the hospital ID, round index, actual optimizer sample count, model tensors and aggregate summaries. The CLI emits only configuration, counts and aggregate local validation metrics.

## Federated runs

- FedAvg averages local weights using the number of optimizer training samples at each hospital. FedProx uses the same aggregation with a local penalty of `(mu / 2) * ||w_local - w_global||²`. FedAvgM averages client deltas, accumulates `velocity = beta * velocity + delta`, and applies `w_global += server_lr * velocity`. Each run starts from its configured seed with fresh server state.
- The three checked-in federated configurations use 20 rounds on the same IID partition with the same starting weights and per-hospital minibatch seeds. A round requires one valid update from each hospital.
- After aggregation, each hospital evaluates the broadcast global weights on its own validation subset. The coordinator pools loss sums and confusion counts, then reports loss, accuracy, precision, recall, F1, and the confusion matrix. Malignant is the positive class. The standalone federated commands leave the benchmark test split unscored.
- Model-transfer estimates use the measured size of each model state serialized with `torch.save` to memory. The initial broadcast sends one state to each hospital. Each round counts three uploads and three downloads, including the final broadcast for validation. Each downloaded state is reused for the next round. Counts exclude summaries, transport overhead, and any storage of model states. The CLI emits the report as JSON on stdout.

## Multi-seed study

From `ml-engine`, run the study or regenerate its plots from saved metrics. Use a new output directory for each invocation. An existing Stage 2 environment needs the updated dependency installation command above before running the study.

```powershell
..\.venv\Scripts\python.exe -m app.study run --config configs/study.json --output ../results/stage3
..\.venv\Scripts\python.exe -m app.study plot --report ../results/stage3/study.json --output ../results/stage3-replot
```

- The study fixes the stratified 114-record holdout with test seed 42. Seeds 42, 43, and 44 vary development partitioning, model initialization, and minibatch order. Within each mode and seed, the three strategies share these settings. Each final model is scored on the same holdout after its last round. Round curves use local validation records.
- The centralized logistic reference uses the same classifier, transform, learning rate, and batch size. It trains separately on the matching pooled public training subset and evaluates on the matching validation subset and shared holdout. Its 60 epochs match the federated budget of 20 rounds × 3 local epochs in passes per training example. Centralized shuffling and minibatch boundaries can differ. The reference has zero federated rounds and model-transfer bytes.
- `study.json` records settings, package versions, aggregate class counts, split fingerprints, and all metrics. `runs.csv` contains final validation and test metrics including confusion counts, `rounds.csv` contains validation and communication metrics for each federated round, and `summary.csv` contains test means and sample standard deviations over run seeds. The PNG figures show validation learning curves and the held-out accuracy and F1 comparison.

## Spring Boot API and MySQL

Install JDK 17–25, Maven 3.9, and MySQL 8.0 or later. Set up the Python virtual environment first. The API runs on `127.0.0.1:8080` and starts the worker using the project virtual environment. [Spring Boot system requirements](https://docs.spring.io/spring-boot/3.5/system-requirements.html)

Connect to your local MySQL server as an administrator. For the MySQL 8.0 Windows installation path, use PowerShell.

```powershell
& 'C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe' -u root -p
```

In MySQL, create the application database and its local account. Substitute your own password.

```sql
CREATE DATABASE IF NOT EXISTS federated_hospitals CHARACTER SET utf8mb4;
CREATE USER 'federated_app'@'127.0.0.1' IDENTIFIED BY 'REPLACE_WITH_LOCAL_PASSWORD';
GRANT CREATE, REFERENCES, SELECT, INSERT, UPDATE ON federated_hospitals.* TO 'federated_app'@'127.0.0.1';
```

Start the API from the backend folder. Spring initializes the metadata tables at startup. Set `FL_DB_URL`, `FL_DB_USER`, `FL_API_PORT`, `FL_PYTHON`, or `FL_ENGINE_DIR` in this terminal if you use custom locations.

```powershell
Set-Location 'D:\Projects\federated-hospitals\backend'
$credential = [pscredential]::new('federated_app', (Read-Host 'MySQL app password' -AsSecureString))
$env:FL_DB_PASSWORD = $credential.GetNetworkCredential().Password
mvn spring-boot:run
```

In this workspace, you can start the built JAR after setting the database password. The local Maven distribution can also rebuild the JAR without a system-wide Maven installation.

```powershell
& '..\.tools\apache-maven-3.9.16\bin\mvn.cmd' '-Dmaven.repo.local=../.tools/m2' package
java -jar target/federated-hospitals-api-0.1.0.jar
```

In a second terminal, run the smoke test or the Java integration suite.

```powershell
Set-Location 'D:\Projects\federated-hospitals\backend'
.\smoke.ps1
mvn test
```

| Endpoint | Result |
|---|---|
| `GET /api/health` | API, database, and Python availability. The Python probe checks that the worker starts and its dependencies are discoverable. |
| `POST /api/experiments` | Validates JSON and returns HTTP 202 with `runId` and `QUEUED`. |
| `GET /api/experiments/{id}` | Configuration, status, completed round count, and failure code when applicable. |
| `GET /api/experiments/{id}/metrics` | Persisted round history and aggregate final metrics once available. |
| `GET /api/experiments/{id}/export?format=json` | Completed run configuration and metrics as a JSON download. Use `format=csv` for validation and test metric rows. |

Submit a two-round FedAvg experiment from PowerShell.

```powershell
$body = @{
    algorithm = 'fedavg'; mode = 'iid'; seed = 42; rounds = 2
    epochs = 1; batchSize = 32; learningRate = 0.01
} | ConvertTo-Json
$run = Invoke-RestMethod 'http://127.0.0.1:8080/api/experiments' -Method Post -ContentType 'application/json' -Body $body
$url = "http://127.0.0.1:8080/api/experiments/$($run.runId)"
Invoke-RestMethod $url
Invoke-RestMethod "$url/metrics"
```

Valid algorithms are `fedavg`, `fedprox`, and `fedavgm`. The mode is `iid` or `non-iid`. The model defaults to `logistic`. Optional settings are `mu` for FedProx and `beta` and `serverLearningRate` for FedAvgM. They default to 0, 0, and 1 respectively. API jobs accept 1–100 rounds, 1–10 local epochs, and batch sizes of 1–512. API runs use the submitted seed for both the split and training. Use the Stage 3 study command for a fixed holdout across different run seeds.

MySQL stores configurations, status, round counts, and typed aggregate metrics. Each job runs in a separate Python subprocess, and its final holdout metrics are persisted only after all rounds finish and the worker exits successfully. A single API instance processes jobs serially. On restart it keeps queued jobs and marks interrupted running jobs as failed. The default job timeout is 600 seconds and can be set with `FL_JOB_TIMEOUT_SECONDS`.

## React dashboard

Keep MySQL and Spring Boot running using the instructions above. With Node.js 22.12+ or 24 and npm installed, open a second PowerShell window and run:

```powershell
Set-Location 'D:\Projects\federated-hospitals\frontend'
npm ci
npm run dev
```

Open [the local dashboard](http://127.0.0.1:5173/). Vite forwards `/api` requests to Spring Boot on `127.0.0.1:8080`. Set `FL_API_TARGET` in the frontend terminal before starting Vite if Spring Boot uses a different port.

To present the built dashboard, run `npm run build` and then `npm run preview` from the same folder. Open [the built dashboard preview](http://127.0.0.1:4173/). It uses the same API proxy.

1. Wait for **API online** and click **Run quick demo**. This submits 20 IID FedAvgM rounds with seed 42. Use **Experiment setup** to choose an algorithm, partition, seed, and training settings for your own run.
2. Watch **Training monitor** for round progress and validation curves. Once complete, the monitor shows held-out test metrics, confusion counts, model-transfer bytes, and CSV and JSON downloads. Hospital cards show aggregate training counts.
3. Open **Compare results** to revisit runs submitted from this browser and see the measured Stage 3 IID and non-IID benchmark. Browser storage keeps up to 12 run IDs. MySQL stores the experiment records.

The bundled benchmark contains aggregate accuracy and F1 statistics from the Stage 3 study. After rerunning that study, refresh the benchmark with `npm run sync-study -- ../results/YOUR_RUN/study.json` from the frontend directory.

With both servers running, check the build and submit a real two-round smoke run through the frontend proxy in another terminal.

```powershell
Set-Location 'D:\Projects\federated-hospitals\frontend'
npm run build
npm run smoke
```

For the built preview, use `npm run smoke -- http://127.0.0.1:4173`.

[Dashboard screenshot](docs/screenshots/dashboard.png) · [Comparison screenshot](docs/screenshots/comparison.png)

## Scope and limitations

The single-process simulation provides logical isolation. Model-transfer estimates characterize serialized payloads in this simulator. Three seeds on one public dataset and one shared holdout give a limited view of variability. The test standard deviations describe variation across run seeds on that holdout. CPU results may differ across environments and library versions.
