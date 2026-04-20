"""
model_stub.py — Shared model architecture

CRITICAL: The SubstitutionModel class below MUST match training/model_stub.py
exactly. If training changes the architecture, copy their class here.

Training's version uses:
  - .mean(dim=1) for context pooling (NO padding mask)
  - F.cosine_similarity for scoring
  - padding_idx=0 on the embedding

This file adds serving-specific helpers (tokenize_ingredients,
build_stub_vocab_and_model) that training doesn't need.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_LEN = 20
PAD_ID = 0
UNK_ID = 1


class SubstitutionModel(nn.Module):
    """MUST match training/model_stub.py exactly."""

    def __init__(self, vocab_size=40, embed_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, context_ids, missing_id):
        ctx_embed = self.embedding(context_ids).mean(dim=1)
        miss_embed = self.embedding(missing_id)
        query = ctx_embed + miss_embed
        all_embeds = self.embedding.weight
        scores = F.cosine_similarity(
            query.unsqueeze(1), all_embeds.unsqueeze(0), dim=2)
        return scores

    def get_substitutions(self, context_ids, missing_id, k=3):
        scores = self.forward(
            context_ids.unsqueeze(0), missing_id.unsqueeze(0))
        scores[0][0] = -1
        scores[0][missing_id] = -1
        top_k = scores[0].topk(k)
        return top_k.indices.tolist(), top_k.values.tolist()


# ------------------------------------------------------------------
# Serving-specific helpers (not in training's model_stub.py)
# ------------------------------------------------------------------

def tokenize_ingredients(ingredient_strings, vocab, context_len=CONTEXT_LEN):
    ids = []
    for ing in ingredient_strings:
        if isinstance(ing, str):
            ids.append(vocab.get(ing.lower().strip(), UNK_ID))
    ids = ids[:context_len]
    ids += [PAD_ID] * (context_len - len(ids))
    return ids


def build_stub_vocab_and_model(vocab_size=200, embed_dim=128):
    common = [
        "flour", "egg", "sugar", "butter", "milk", "salt", "pepper", "oil",
        "garlic", "onion", "tomato", "chicken", "beef", "rice", "pasta",
        "cheese", "cream", "lemon", "herbs", "vanilla", "baking_powder",
        "yeast", "water", "vinegar", "honey", "soy_sauce", "ginger",
        "cinnamon", "nutmeg", "paprika", "cumin", "oregano", "basil",
        "thyme", "rosemary", "potato", "carrot", "celery", "mushroom",
        "spinach", "sour cream", "greek yogurt", "cream cheese",
        "buttermilk", "heavy cream", "all-purpose flour",
        "beef sirloin", "beef broth", "egg noodles",
    ]
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for c in common:
        if c not in vocab:
            vocab[c] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"ingredient_{len(vocab)}"] = len(vocab)
    id_to_ingredient = {v: k for k, v in vocab.items()}
    model = SubstitutionModel(vocab_size=vocab_size, embed_dim=embed_dim)
    torch.manual_seed(42)
    model.embedding.weight.data = torch.randn(vocab_size, embed_dim) * 0.1
    return model, vocab, id_to_ingredient
