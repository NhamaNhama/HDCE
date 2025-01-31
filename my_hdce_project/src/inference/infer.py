import os
import yaml
import torch
import argparse

from ..hdce.hdce_model import HDCEModel

def run_inference(model, text, graph_data, external_concept):
    model.eval()
    with torch.no_grad():
        outputs = model([text], graph_data, external_concept)
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--text", type=str, default="This is a test.")
    parser.add_argument("--concept", type=str, default="apple")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    model = HDCEModel(
        transformer_model_name = config["model"]["transformer_model_name"],
        embed_dim = config["model"]["embed_dim"],
        num_relations = config["model"]["graph_num_relations"],
        gnn_num_layers = config["model"]["gnn_num_layers"],
        knowledge_graph_file = config["model"]["knowledge_graph_file"]
    )
    
    # 学習済みの state_dict をロードする場合
    if os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
    
    # ダミーの graph_data
    dummy_graph_data = {
        "node_feature": torch.randn(5, config["model"]["embed_dim"]),
        "edge_index": torch.tensor([[0,1],[1,2]]),
        "edge_type": torch.tensor([0,0]),
        "path_length": torch.tensor([1.0,1.0]),
        "num_nodes": 5
    }
    
    outputs = run_inference(model, args.text, dummy_graph_data, args.concept)
    print("Inference Outputs:", outputs)

if __name__ == "__main__":
    main() 