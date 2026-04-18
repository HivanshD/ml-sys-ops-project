# training/  Training role subdirectory. To be populated by Training team member.  See serving/INTEGRATION.md for the checkpoint contract. Specifically, the .pth file saved to models-proj01/production/subst_model_current.pth must be a dict with keys: model_state_dict, vocab, config.  Expected files:   - train.py          — main training script with MRR@3 quality gate   - evaluate.py       — MRR@3, NDCG@3, per-cuisine (fairness) metrics   - watch_trigger.py  — K8S CronJob picking up retrain triggers   - config.yaml       — hyperparameters   - Dockerfile        — PyTorch CUDA training container   - requirements.txt  Training imports SubstitutionModel from   ../serving/fastapi_pt/model_stub.py to guarantee train/serve architecture parity.



