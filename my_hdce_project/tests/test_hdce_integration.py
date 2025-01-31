import torch
from src.hdce.hdce_model import HDCEModel

def test_hdce_integration():
    model = HDCEModel(
        transformer_model_name="bert-base-uncased",
        embed_dim=768,
        num_relations=4,
        gnn_num_layers=2,
        knowledge_graph_file="data/knowledge_graph.json"
    )
    input_texts = ["This is a test"]
    graph_data = {
        "node_feature": torch.randn(5, 768),
        "edge_index": torch.tensor([[0,1],[1,2]]),
        "edge_type": torch.tensor([0,0]),
        "path_length": torch.tensor([1.0,1.0]),
        "num_nodes": 5
    }
    output = model(input_texts, graph_data, external_concept="apple")
    assert "local_out" in output
    assert "updated_nodes" in output 