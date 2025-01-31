import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class HDCETrainerModule(pl.LightningModule):
    def __init__(self, hdce_model, lr=3e-5):
        super().__init__()
        self.hdce_model = hdce_model
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        # batch: Dict[str, Any]
        input_texts = [batch["text"]]
        label = torch.tensor(batch["label"], dtype=torch.float32).unsqueeze(0).to(self.device)
        graph_data = batch["graph_data"]
        external_concept = batch["external_concept"]
        
        outputs = self.hdce_model(input_texts, graph_data, external_concept)
        
        cls_repr = outputs["local_out"][:,0,:]
        logits = cls_repr.mean(dim=-1, keepdim=True)  # (B,1)
        loss = F.binary_cross_entropy_with_logits(logits, label.view(-1,1))
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr) 