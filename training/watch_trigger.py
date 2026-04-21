import json
import os
import subprocess
import sys

import boto3
import requests


s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('OS_ENDPOINT'),
    aws_access_key_id=os.getenv('OS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('OS_SECRET_KEY'))

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow.monitoring-proj01:5000')
TRIGGER_PREFIX = os.getenv('TRIGGER_PREFIX', 'data/triggers/')
AUTOMATION_URL = os.getenv('AUTOMATION_URL', 'http://automation.monitoring-proj01:8080')
DEPLOY_TARGETS = [
    target.strip() for target in os.getenv('DEPLOY_TARGETS', 'staging,canary').split(',')
    if target.strip()
]


def extract_candidate_manifest_key(stdout):
    for line in stdout.splitlines():
        if line.startswith('CANDIDATE_MANIFEST_KEY='):
            return line.split('=', 1)[1].strip()
    return None


def deploy_candidate(manifest_key):
    if not DEPLOY_TARGETS:
        return
    response = requests.post(
        f'{AUTOMATION_URL}/deploy-candidate',
        json={'manifest_key': manifest_key, 'targets': DEPLOY_TARGETS},
        timeout=60)
    response.raise_for_status()
    print(f'[watch_trigger] Candidate deployed: {response.text}')


result = s3.list_objects_v2(Bucket='data-proj01', Prefix=TRIGGER_PREFIX)
triggers = sorted(result.get('Contents', []), key=lambda item: item['Key'])

if not triggers:
    print(f'[watch_trigger] No triggers found under {TRIGGER_PREFIX}. Exiting.')
    sys.exit(0)

all_ok = True

for t in triggers:
    key = t['Key']
    if key.endswith('/'):
        continue
    cfg = json.loads(s3.get_object(Bucket='data-proj01', Key=key)['Body'].read())
    print(f'[watch_trigger] {cfg["trigger_version"]} ({cfg["new_samples"]} samples)')
    proc = subprocess.run([
        'python', 'train.py',
        '--dataset', cfg['dataset_path'],
        '--run_name', cfg['trigger_version'],
        '--mlflow_tracking_uri', MLFLOW_URI,
    ], check=False, capture_output=True, text=True)

    if proc.stdout:
        print(proc.stdout, end='' if proc.stdout.endswith('\n') else '\n')
    if proc.stderr:
        print(proc.stderr, end='' if proc.stderr.endswith('\n') else '\n', file=sys.stderr)

    if proc.returncode != 0:
        all_ok = False
        print(f'[watch_trigger] FAILED (not deleting trigger): {key}')
        continue

    candidate_manifest_key = extract_candidate_manifest_key(proc.stdout)
    if candidate_manifest_key:
        try:
            deploy_candidate(candidate_manifest_key)
        except Exception as e:
            all_ok = False
            print(f'[watch_trigger] Candidate deploy failed for {key}: {e}')
            continue

    s3.delete_object(Bucket='data-proj01', Key=key)
    print(f'[watch_trigger] Consumed and deleted: {key}')

sys.exit(0 if all_ok else 1)
