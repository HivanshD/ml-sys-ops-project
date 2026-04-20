"""
watch_trigger.py - watches data-proj01 bucket for new data triggers
and automatically kicks off retraining when new data is ingested.

Run as K8S CronJob every 15 minutes:
  python watch_trigger.py
"""
import os, json, subprocess, time, urllib.request
from datetime import datetime

BUCKET_URL = "https://chi.tacc.chameleoncloud.org:7480/swift/v1/AUTH_d3c6e101843a4ba79e665ebf59b521a2/data-proj01"
TRAIN_SCRIPT = "/workspace/training/train.py"
CONFIG       = "/workspace/training/config.yaml"
DATASET      = "/workspace/data/processed/train.json"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://10.140.82.89:5000")
LAST_SEEN_FILE = "/tmp/last_trigger_etag.txt"

def get_token():
    token = os.getenv("OS_TOKEN", "")
    if not token:
        raise ValueError("OS_TOKEN not set")
    return token

def get_latest_train_etag(token):
    url = f"{BUCKET_URL}/data/raw/recipe1msubs/train.json"
    req = urllib.request.Request(url, method="HEAD", headers={"X-Auth-Token": token})
    try:
        with urllib.request.urlopen(req) as r:
            return r.headers.get("ETag", "")
    except Exception:
        return ""

def load_last_etag():
    try:
        return open(LAST_SEEN_FILE).read().strip()
    except Exception:
        return ""

def save_etag(etag):
    open(LAST_SEEN_FILE, "w").write(etag)

def download_latest_data(token):
    base = BUCKET_URL
    os.makedirs("/workspace/data/processed", exist_ok=True)
    os.makedirs("/workspace/data/production_holdout", exist_ok=True)
    for src, dst in [
        ("data/raw/recipe1msubs/train.json", "/workspace/data/processed/train.json"),
        ("data/raw/recipe1msubs/val.json",   "/workspace/data/processed/val.json"),
        ("data/production_holdout/holdout.json", "/workspace/data/production_holdout/holdout.json"),
    ]:
        req = urllib.request.Request(f"{base}/{src}", headers={"X-Auth-Token": token})
        with urllib.request.urlopen(req) as r:
            open(dst, "wb").write(r.read())
        print(f"Downloaded {src}")

def run_training():
    cmd = [
        "python", TRAIN_SCRIPT,
        "--config", CONFIG,
        "--dataset", DATASET,
        "--embed_dim", "4096",
        "--lr", "0.00005",
        "--epochs", "50",
        "--batch_size", "16",
        "--margin", "1.5",
        "--run_name", f"auto-retrain-{datetime.now().strftime('%Y%m%d-%H%M')}",
        "--mlflow_tracking_uri", MLFLOW_URI,
    ]
    print(f"[{datetime.now()}] Starting retraining: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"[{datetime.now()}] Training finished with exit code {result.returncode}")
    return result.returncode

def main():
    print(f"[{datetime.now()}] ForkWise watch_trigger started")
    token        = get_token()
    current_etag = get_latest_train_etag(token)
    last_etag    = load_last_etag()
    print(f"Current ETag: {current_etag}")
    print(f"Last seen:    {last_etag}")
    if current_etag and current_etag != last_etag:
        print(f"[{datetime.now()}] New data detected — triggering retraining...")
        download_latest_data(token)
        rc = run_training()
        if rc == 0:
            save_etag(current_etag)
            print(f"[{datetime.now()}] Retraining complete. ETag saved.")
        else:
            print(f"[{datetime.now()}] Retraining failed with exit code {rc}")
    else:
        print(f"[{datetime.now()}] No new data. Skipping retraining.")

if __name__ == "__main__":
    main()
