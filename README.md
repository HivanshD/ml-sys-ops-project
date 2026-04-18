# Serving Repository

This repository is the source of truth for the integrated FrokWise deployment.

It started as a small serving-focused repo for model-serving experiments: FastAPI, ONNX, Triton, and benchmarking. After the infra migration, it now owns both the serving implementation and the deployment shape needed to run the integrated system.

## What Changed

This repo is no longer only a place for serving experiments.

It now also owns:

1. infrastructure layout for Chameleon under `tf/`
2. configuration and deployment orchestration under `ansible/`
3. Kubernetes application manifests under `k8s/`
4. migration and operating documentation under `docs/` and `RUNBOOK.md`

The intent is to make this repository the canonical place another engineer or agent lands when they need to understand how the integrated Mealie plus substitution-serving system is supposed to run.

## Current Architectural Direction

The current deployment decisions are:

1. canonical repo boundary: this `serving/` Git repo
2. primary serving stack: `FastAPI + ONNX`
3. secondary serving path: `Triton`, kept as an experiment and benchmark path
4. primary app target: `Mealie`
5. environment strategy: local Kubernetes for fast iteration, Chameleon as the documented target environment
6. rollout model: a single app deployment path for now, not `staging/canary/production`
7. infra depth for this migration phase: `Terraform + Ansible + Kubernetes manifests + runbook`

## Repository Layout

```text
.
├── ansible/
├── docs/
├── fastapi_onnx/
├── fastapi_pt/
├── k8s/
├── tf/
├── triton_client/
├── triton_models/
├── RUNBOOK.md
├── benchmark.py
├── docker-compose-fastapi.yaml
├── docker-compose-triton.yaml
├── Dockerfile.triton
├── Dockerfile.triton_client
├── export_onnx.py
├── model_stub.py
└── quantize_onnx.py
```

## Where To Start

1. Read `docs/adr/0001-repo-migration-and-deployment-boundary.md` for the canonical repo-boundary decision.
2. Read `docs/MIGRATION_MAP.md` for the old-to-new path mapping.
3. Read `RUNBOOK.md` for the intended bring-up flow.
4. Use `fastapi_onnx/` as the default serving implementation path.

## Current Status

The repo-boundary migration is no longer only conceptual.

This repo now contains:

1. real Terraform files under `tf/kvm/`
2. real Ansible bootstrap and deploy playbooks under `ansible/`
3. raw Kubernetes manifests for `Mealie` and `substitution-serving` under `k8s/apps/`
4. canonical migration docs under `docs/`

The existing serving implementation remains in place:

1. `fastapi_onnx/` is the primary deployment path.
2. `fastapi_pt/` remains available for comparison.
3. `Dockerfile.triton`, `triton_models/`, and `docker-compose-triton.yaml` remain the Triton evaluation path.

## Minimal Chameleon Bring-Up

The current intended deployment path is:

1. provision infrastructure from `tf/kvm/`
2. prepare the nodes with `ansible/pre_k8s/pre_k8s_configure.yml`
3. bootstrap k3s with `ansible/k8s/install_k3s.yml`
4. run `ansible/post_k8s/post_k8s_configure.yml`
5. deploy the apps with `ansible/deploy/deploy_apps.yml`

For the exact command sequence, read `RUNBOOK.md`.
