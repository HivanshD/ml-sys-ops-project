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
- Integration contract: serving/INTEGRATION.md
- Operational runbook: serving/RUNBOOK.md
- Lab references: serving/LAB_REFERENCES.md
- Infra migration notes: infra/docs/

## One-command setup (on Chameleon)

```bash
# Provision infrastructure
cd infra/tf/kvm && terraform apply
cd ../../ansible && ansible-playbook setup.yml

# Apply K8S manifests
kubectl apply -f infra/k8s/namespaces.yaml
kubectl apply -f infra/k8s/ --recursive
kubectl apply -f serving/k8s-cronjob-manifests.yaml
```

## Chameleon resource naming

All resources use the suffix `proj01`:
- **Namespaces:** `production-proj01`, `canary-proj01`, `staging-proj01`, `monitoring-proj01`
- **Buckets:** `data-proj01`, `models-proj01`, `logs-proj01`

