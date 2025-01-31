import pytest
import torch
from src.hdce.local_context_encoder import LocalContextEncoder

def test_local_context_encoder():
    encoder = LocalContextEncoder("bert-base-uncased")
    out, attn, mask = encoder(["Hello world"])
    assert out.shape[0] == 1
    assert out.shape[2] == 768 