import os
import yaml
import random
import numpy as np
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from ..hdce.hdce_model import HDCEModel
from .lightning_trainer import HDCETrainerModule
from ..data.dataset import HDCETextGraphDataset

def set_seed(seed):
    # 乱数シードを一括設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset():
    # 簡単なダミーデータセットを生成
    data_list = []
    for i in range(10):
        data_list.append({
            "text": f"sample text {i}",
            "label": i % 2,
            "graph_data": {
                "node_feature": torch.randn(5, 768),
                "edge_index": torch.tensor([[0,1],[1,2]]),
                "edge_type": torch.tensor([0,0]),
                "path_length": torch.tensor([1.0,1.0]),
                "num_nodes": 5
            },
            "external_concept": "apple" if i % 2 == 0 else "fruit"
        })
    return HDCETextGraphDataset(data_list)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    
    # 設定ファイルを読み込む
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # シード固定
    set_seed(config["training"]["seed"])
    
    # モデルを初期化
    hdce_model = HDCEModel(
        transformer_model_name = config["model"]["transformer_model_name"],
        embed_dim = config["model"]["embed_dim"],
        num_relations = config["model"]["graph_num_relations"],
        gnn_num_layers = config["model"]["gnn_num_layers"],
        knowledge_graph_file = config["model"]["knowledge_graph_file"]
    )
    
    # LightningのTrainerModuleを作成
    trainer_module = HDCETrainerModule(hdce_model, lr=config["training"]["learning_rate"])
    
    # データセットとDataLoaderを用意
    dataset = load_dataset()
    train_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    
    # PyTorch LightningのTrainerを作成
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto"
    )
    # 学習を実行
    trainer.fit(trainer_module, train_loader)

if __name__ == "__main__":
    main() 