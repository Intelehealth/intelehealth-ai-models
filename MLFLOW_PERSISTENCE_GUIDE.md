# MLflow Persistent Logging Guide

This guide explains how to set up persistent logging for your DDx server using MLflow.

## 🎯 Current Setup

Your DDx server now uses **persistent MLflow tracking** with the following components:

### 1. Database Backend
- **Storage**: SQLite database (`mlflow_data/mlflow.db`)
- **Artifacts**: Local file system (`mlflow_data/artifacts/`)
- **Persistence**: All logs are permanently stored and survive server restarts

### 2. What Gets Logged
For each `/predict/v1` request, the following is tracked:

#### Parameters
- `model_name`: The model being used
- `case_length`: Length of the input case
- `llm_model`: LLM model identifier (e.g., "gemini/gemini-2.0-flash")
- `llm_max_tokens`: Maximum tokens setting
- `llm_temperature`: Temperature parameter
- `llm_top_k`: Top-k parameter
- `prompt_config_index`: Which prompt configuration was used

#### Artifacts (Files)
- `input_case.txt`: The original case input
- `prompt_used.txt`: The prompt that was sent to the LLM
- `raw_dspy_output.txt`: Raw output from DSPy
- `final_response.txt`: Final transformed response

#### Metrics
- `success`: Whether the request succeeded (1) or failed (0)
- `processing_time`: Time taken to process the request

## 🚀 How to Run

### Option 1: Use the Startup Script (Recommended)
```bash
./start_mlflow_and_server.sh
```

This script will:
1. Start MLflow tracking server on port 6000
2. Start your DDx server on port 8000
3. Handle cleanup when you stop the servers

### Option 2: Manual Startup
```bash
# Terminal 1: Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow_data/mlflow.db --default-artifact-root mlflow_data/artifacts --host 127.0.0.1 --port 5000

# Terminal 2: Start DDx server
python ddx_server.py
```

## 📊 Accessing Your Logs

1. **MLflow UI**: http://127.0.0.1:5000
2. **Experiment**: "ddx-server-tracking"
3. **Database**: `mlflow_data/mlflow.db`
4. **Artifacts**: `mlflow_data/artifacts/`

## 🔄 Data Persistence

### What Persists Forever
- ✅ All run parameters and metrics
- ✅ All artifacts (input/output files)
- ✅ Run history and timestamps
- ✅ Model performance over time
- ✅ Complete audit trail

### Backup Your Data
To backup your MLflow data:
```bash
# Backup the entire mlflow_data directory
tar -czf mlflow_backup_$(date +%Y%m%d).tar.gz mlflow_data/
```

## 🏢 Production Options

### Option 1: Remote Database (PostgreSQL/MySQL)
For production, consider using a remote database:

```python
# In mlflow_server_config.py, replace SQLite with:
db_uri = "postgresql://user:password@localhost/mlflow_db"
# or
db_uri = "mysql://user:password@localhost/mlflow_db"
```

### Option 2: Cloud Storage for Artifacts
Use cloud storage for artifacts:

```python
# AWS S3
artifacts_path = "s3://your-bucket/mlflow-artifacts"

# Google Cloud Storage
artifacts_path = "gs://your-bucket/mlflow-artifacts"

# Azure Blob Storage
artifacts_path = "wasbs://your-container@your-account.blob.core.windows.net/mlflow-artifacts"
```

### Option 3: MLflow on Cloud Platforms
- **Databricks MLflow**: Fully managed MLflow service
- **AWS SageMaker**: MLflow integration
- **Google Cloud AI Platform**: MLflow support
- **Azure ML**: MLflow tracking

## 📈 Monitoring and Analytics

With persistent logging, you can:

1. **Track Model Performance**: Monitor success rates over time
2. **Analyze Usage Patterns**: See which models are used most
3. **Debug Issues**: Complete audit trail of failed requests
4. **Optimize Parameters**: Compare different temperature/token settings
5. **Cost Analysis**: Track token usage and processing times

## 🛠️ Troubleshooting

### MLflow Server Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill existing MLflow processes
pkill -f "mlflow server"
```

### Database Issues
```bash
# Check database file permissions
ls -la mlflow_data/mlflow.db

# Reset database (WARNING: This deletes all data)
rm mlflow_data/mlflow.db
```

### Artifacts Not Saving
```bash
# Check artifacts directory permissions
ls -la mlflow_data/artifacts/

# Ensure directory exists and is writable
mkdir -p mlflow_data/artifacts
chmod 755 mlflow_data/artifacts
```

## 📋 File Structure

```
your-project/
├── mlflow_data/                 # Persistent data directory
│   ├── mlflow.db               # SQLite database (all runs/params/metrics)
│   └── artifacts/              # Artifact files (inputs/outputs)
│       └── experiment_id/
│           └── run_id/
│               ├── input_case.txt
│               ├── prompt_used.txt
│               └── ...
├── start_mlflow_and_server.sh  # Startup script
└── ddx_server.py               # Your main server (with logging)
```

## 🔐 Security Considerations

For production:
1. **Authentication**: Enable MLflow authentication
2. **HTTPS**: Use SSL/TLS for MLflow UI
3. **Database Security**: Use encrypted database connections
4. **Access Control**: Limit who can view/modify experiments
5. **Backup Encryption**: Encrypt your backup files

Your logs are now persistent and will survive server restarts, system reboots, and can be easily backed up or migrated to other systems! 