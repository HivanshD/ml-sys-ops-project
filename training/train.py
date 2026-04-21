import torch.nn.functional as F
import argparse, json, os, random, re, tempfile, time
from datetime import datetime, timezone
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from model_stub import SubstitutionModel
from evaluate import evaluate_model

def load_data(path):
    return json.loads(open(path).read())


def get_s3_client():
    import boto3
    return boto3.client(
        's3',
        endpoint_url=os.getenv('OS_ENDPOINT'),
        aws_access_key_id=os.getenv('OS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('OS_SECRET_KEY'))


def parse_object_store_path(path):
    if not path:
        return None
    if os.path.exists(path):
        return None
    if path.startswith('/') or path.startswith('./') or path.startswith('../'):
        return None
    if path.startswith('s3://'):
        raw = path[len('s3://'):]
        if '/' not in raw:
            return None
        return raw.split('/', 1)
    if '/' not in path:
        return None
    bucket, key = path.split('/', 1)
    if not bucket or not key:
        return None
    return bucket, key


def resolve_dataset_paths(train_path, val_path=None):
    train_remote = parse_object_store_path(train_path)
    val_remote = parse_object_store_path(val_path) if val_path else None

    if train_remote is None and (val_path is None or val_remote is None):
        return train_path, (val_path or train_path.replace('train', 'val')), None

    s3 = get_s3_client()
    tmpdir = tempfile.TemporaryDirectory()

    if train_remote is not None:
        train_bucket, train_key = train_remote
        train_local = os.path.join(tmpdir.name, 'train.json')
        s3.download_file(Bucket=train_bucket, Key=train_key, Filename=train_local)
    else:
        train_local = train_path
        train_bucket = None

    if val_path:
        if val_remote is not None:
            val_bucket, val_key = val_remote
            val_local = os.path.join(tmpdir.name, 'val.json')
            s3.download_file(Bucket=val_bucket, Key=val_key, Filename=val_local)
        else:
            val_local = val_path
    elif train_bucket is not None:
        val_key = os.getenv('VAL_DATASET_KEY', 'data/raw/recipe1msubs/val.json')
        val_local = os.path.join(tmpdir.name, 'val.json')
        s3.download_file(Bucket=train_bucket, Key=val_key, Filename=val_local)
    else:
        val_local = train_path.replace('train', 'val')

    return train_local, val_local, tmpdir


def sanitize_model_version(run_name):
    safe = re.sub(r'[^A-Za-z0-9._-]+', '-', run_name).strip('-_.')
    return safe or f'model-{int(time.time())}'

def build_vocab(data):
    ingrs = set()
    for r in data:
        ingrs.add(r['original'].lower().strip())
        ingrs.add(r['replacement'].lower().strip())
        for i in r.get('ingredients', []):
            if isinstance(i, str): ingrs.add(i.lower().strip())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i in sorted(ingrs):
        if i and i not in vocab: vocab[i] = len(vocab)
    return vocab

def prepare_batch(records, vocab, context_len=20):
    ctxs, miss_ids, pos_ids, neg_ids = [], [], [], []
    all_ingrs = list(vocab.keys())
    for r in records:
        orig = r['original'].lower().strip()
        repl = r['replacement'].lower().strip()
        ctx  = [vocab.get(i.lower().strip(), 1)
                for i in r.get('ingredients', []) if isinstance(i, str)]
        ctx  = ctx[:context_len]
        ctx += [0] * (context_len - len(ctx))
        neg  = random.choice(all_ingrs)
        while neg in (orig, repl, '<PAD>', '<UNK>'): neg = random.choice(all_ingrs)
        ctxs.append(ctx)
        miss_ids.append(vocab.get(orig, 1))
        pos_ids.append(vocab.get(repl, 1))
        neg_ids.append(vocab.get(neg, 1))
    return (torch.tensor(ctxs), torch.tensor(miss_ids),
            torch.tensor(pos_ids), torch.tensor(neg_ids))

def train_epoch(model, optimizer, data, vocab, config, device):
    model.train()
    random.shuffle(data)
    total, n = 0.0, 0
    bs = config['batch_size']
    for i in range(0, len(data), bs):
        ctx, miss, pos, neg = prepare_batch(
            data[i:i+bs], vocab, config.get('context_len', 20))
        ctx_e  = model.embedding(ctx.to(device)).mean(dim=1)
        miss_e = model.embedding(miss.to(device))
        pos_e  = model.embedding(pos.to(device))
        neg_e  = model.embedding(neg.to(device))
        query  = ctx_e + miss_e
        ps = nn.functional.cosine_similarity(query, pos_e)
        ns = nn.functional.cosine_similarity(query, neg_e)
        loss = nn.functional.margin_ranking_loss(
            ps, ns, torch.ones_like(ps),
            margin=config.get('margin', 0.5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)

def export_onnx(model, run_name):
    onnx_path = f'/tmp/{run_name}.onnx'
    try:
        ctx_dummy  = torch.zeros(1, 20, dtype=torch.long)
        miss_dummy = torch.zeros(1, dtype=torch.long)
        torch.onnx.export(
            model, (ctx_dummy, miss_dummy), onnx_path,
            input_names=['context_ids', 'missing_id'],
            output_names=['scores'],
            opset_version=14)
        print('ONNX exported')
    except Exception as e:
        print(f'ONNX skip: {e}')
        onnx_path = None
    return onnx_path

def save_and_register(model, vocab, config, metrics, run_name):
    run_id    = mlflow.active_run().info.run_id
    model_version = sanitize_model_version(run_name)
    config_to_save = dict(config)
    config_to_save['run_name'] = run_name
    config_to_save['model_version'] = model_version
    ckpt_path = f'/tmp/{run_name}.pth'
    torch.save({'model_state_dict': model.state_dict(),
                'vocab': vocab, 'config': config_to_save}, ckpt_path)
    mlflow.log_artifact(ckpt_path)
    vocab_path = '/tmp/vocab.json'
    with open(vocab_path, 'w') as f: json.dump(vocab, f)
    mlflow.log_artifact(vocab_path)
    onnx_path = export_onnx(model, run_name)
    if onnx_path: mlflow.log_artifact(onnx_path)
    mlflow.pytorch.log_model(model, 'model')
    if not onnx_path:
        print('ONNX export failed; candidate manifest will not be published.')
        return None
    try:
        s3 = get_s3_client()
        artifact_keys = {
            'pytorch_key': f'versions/{model_version}/subst_model.pth',
            'vocab_key': f'versions/{model_version}/vocab.json',
        }
        for key, path in [
            (f'checkpoints/subst_model_v{run_id}.pth', ckpt_path),
            (artifact_keys['pytorch_key'], ckpt_path),
            (artifact_keys['vocab_key'], vocab_path),
        ]:
            with open(path, 'rb') as f:
                s3.put_object(Bucket='models-proj01', Key=key, Body=f)
        if onnx_path:
            artifact_keys['onnx_key'] = f'versions/{model_version}/subst_model.onnx'
            with open(onnx_path, 'rb') as f:
                s3.put_object(Bucket='models-proj01',
                    Key=artifact_keys['onnx_key'], Body=f)

        manifest = {
            'model_version': model_version,
            'run_name': run_name,
            'run_id': run_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'metrics': metrics,
            'artifacts': artifact_keys,
        }
        candidate_key = f'candidates/{model_version}.json'
        body = json.dumps(manifest, indent=2)
        s3.put_object(Bucket='models-proj01', Key=candidate_key, Body=body)
        s3.put_object(Bucket='models-proj01', Key='candidates/latest.json', Body=body)
        print('Uploaded candidate artifacts to models-proj01')
        return candidate_key
    except Exception as e:
        print(f'Object storage skip (OK): {e}')
        return None

def train(config, dataset_path, run_name, mlflow_uri, val_dataset_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU:  {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment('forkwise-ingredient-substitution')

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: config[k] for k in
            ['embed_dim','epochs','batch_size','lr','margin','quality_gate_mrr']})
        mlflow.log_param('dataset', dataset_path)
        mlflow.log_param('dataset_type', 'recipe1msubs')
        mlflow.log_param('model_type', 'embedding_cosine')
        mlflow.log_param('device', str(device))
        if device.type == 'cuda':
            mlflow.log_param('gpu', torch.cuda.get_device_name(0))

        print(f'Loading {dataset_path}...')
        resolved_train_path, resolved_val_path, temp_dir = resolve_dataset_paths(
            dataset_path, val_dataset_path)
        train_data = load_data(resolved_train_path)
        val_data   = load_data(resolved_val_path)
        print(f'Train: {len(train_data):,}  Val: {len(val_data):,}')

        vocab = build_vocab(train_data)
        mlflow.log_param('vocab_size', len(vocab))
        print(f'Vocab: {len(vocab):,} ingredients')
        with open('/tmp/vocab.json', 'w') as f: json.dump(vocab, f)

        model     = SubstitutionModel(vocab_size=len(vocab),
                                      embed_dim=config['embed_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        mlflow.log_param('param_count',
            sum(p.numel() for p in model.parameters()))

        start = time.time()
        for epoch in range(config['epochs']):
            t    = time.time()
            loss = train_epoch(model, optimizer, train_data, vocab, config, device)
            mlflow.log_metric('train_loss', loss, step=epoch)
            mlflow.log_metric('epoch_time_sec', time.time()-t, step=epoch)
            print(f'Epoch {epoch+1}/{config["epochs"]}: '
                  f'loss={loss:.4f}  time={time.time()-t:.1f}s')
        mlflow.log_metric('total_training_time_sec', time.time()-start)

        model_cpu = model.cpu()
        print('Evaluating...')
        metrics = evaluate_model(model_cpu, val_data, vocab)
        mlflow.log_metrics(metrics)
        for k, v in sorted(metrics.items()):
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

        threshold = config['quality_gate_mrr']
        if metrics['mrr_at_3'] >= threshold:
            mlflow.set_tag('quality_gate', 'passed')
            print(f'QUALITY GATE PASSED: MRR@3={metrics["mrr_at_3"]:.4f} >= {threshold}')
            candidate_manifest_key = save_and_register(model_cpu, vocab, config, metrics, run_name)
            if candidate_manifest_key:
                mlflow.log_param('candidate_manifest_key', candidate_manifest_key)
                print(f'CANDIDATE_MANIFEST_KEY={candidate_manifest_key}')
        else:
            mlflow.set_tag('quality_gate', 'failed')
            print(f'QUALITY GATE FAILED: MRR@3={metrics["mrr_at_3"]:.4f} < {threshold}')

        if temp_dir is not None:
            temp_dir.cleanup()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',   default='config.yaml')
    p.add_argument('--dataset',  default='/workspace/data/processed/train.json')
    p.add_argument('--val_dataset', default=None)
    p.add_argument('--run_name', default='run')
    p.add_argument('--mlflow_tracking_uri', default='http://localhost:5000')
    p.add_argument('--embed_dim',  type=int,   default=None)
    p.add_argument('--lr',         type=float, default=None)
    p.add_argument('--epochs',     type=int,   default=None)
    p.add_argument('--batch_size', type=int,   default=None)
    p.add_argument('--margin',     type=float, default=None)
    args = p.parse_args()
    config = yaml.safe_load(open(args.config))
    if args.embed_dim:  config['embed_dim']  = args.embed_dim
    if args.lr:         config['lr']         = args.lr
    if args.epochs:     config['epochs']     = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.margin:     config['margin']     = args.margin
    train(config, args.dataset, args.run_name, args.mlflow_tracking_uri, args.val_dataset)
