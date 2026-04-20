# ml-sys-ops-project

**ECE-GY 9183 | proj01 | Ingredient Substitution for Mealie**

End-to-end ML system that adds AI-powered ingredient substitution
suggestions to Mealie, a self-hosted recipe manager. When a user is missing an ingredient, the
system suggests ranked substitutions via an embedding-based model trained
on Recipe1MSubs.

## Components

| Directory | Owner | Purpose |
|-----------|-------|---------|
| serving/ | Serving | FastAPI + ONNX inference, monitoring, rollback/promote |
| training/ | Training | Train pipeline, MLflow integration, quality gates |
| data/ | Data | Recipe1MSubs ingestion, feedback capture, drift monitoring |
| infra/ | DevOps | K8S manifests, Ansible, Terraform, automation |
| mealie-integration/ | Team | Mealie backend patch + Vue component |
| archive/ | - | Initial implementation files, preserved for reference |

## Where to start

- New team member: read serving/TEAM_INTEGRATION_MAP.md
- Cloud deployment: `infra/docs/FORKWISE_CLOUD_SETUP.md`
- Serving runbook: `serving/RUNBOOK.md`
- Infra migration notes: `infra/docs/`
- Historical serving integration notes: `serving/INTEGRATION.md`

## Canonical cloud path

The current canonical deployment uses app-oriented namespaces and GHCR-backed
ForkWise data images:

- `forkwise-app` for Mealie
- `forkwise-serving` for the primary serving API
- `forkwise-data` for feedback, ingest, generator, batch, and drift workloads

Use `infra/docs/FORKWISE_CLOUD_SETUP.md` for the step-by-step cloud bring-up.

## Published ForkWise data images

```text
ghcr.io/itsnotaka/forkwise-ingest:demo
ghcr.io/itsnotaka/forkwise-feedback:demo
ghcr.io/itsnotaka/forkwise-batch:demo
ghcr.io/itsnotaka/forkwise-generator:demo
```

