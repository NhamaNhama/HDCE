import torch
from src.hdce.global_semantic_module import GlobalSemanticModule

def test_global_semantic_module():
    embed_dim = 768
    gsm = GlobalSemanticModule(embed_dim, num_relations=4, num_layers=2)
    x = torch.randn(5, embed_dim)
    edge_index = torch.tensor([[0,1],[1,2]])
    edge_type = torch.tensor([0,0])
    path_length = torch.tensor([1.0,1.0])
    out = gsm(x, edge_index, edge_type, path_length, num_nodes=5)
    assert out.shape == (5, embed_dim) 