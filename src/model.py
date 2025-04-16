import pytorch_lightning as pl
from torch.optim import Adam
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dotenv import load_dotenv
from src.data import transform
import os

load_dotenv()

project_root = os.getenv('PYTHONPATH')

model_config_path = f'{project_root}/configs/model.yaml'

def get_model_config():
    with open(model_config_path, 'r') as file:
        return yaml.safe_load(file)

pos_weight = 2.0
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

class CreditRiskModel (pl.LightningModule):
    def __init__(self, input_dim = 23, hidden = 64, sigmoid_threashold=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.threashold = sigmoid_threashold
    
    def forward(self, x):
        logits = self.model(x)
        return logits.squeeze(1)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        # loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        # loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threashold).float()

        acc = (preds == y).float().mean()
        preds_np = preds.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        fnr = 1. - recall_score(y_np, preds_np)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_fnr', fnr, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
    
def get_model():
    """
    Loads the CreditRiskModel from a checkpoint and returns the model along with the device it is on.

    Returns:
        model (CreditRiskModel): The loaded model.
        device (torch.device): The device on which the model is located.
    """
    model_config = get_model_config()
    ckpt_path = model_config['ckpt_path']
    params_path= model_config['params_path']
    model = CreditRiskModel.load_from_checkpoint(ckpt_path, hparams_file=params_path)
    return model, model.device

def predict(model, X):
    X_transformed = transform(X)
    X_transformed = X_transformed.to(model.device)
    model.eval()
    with torch.no_grad():
        logits = model(X_transformed)
        probs = torch.sigmoid(logits)
        preds = (probs > model.threashold).float()
        prediction = preds.detach().cpu().numpy()[0]
    return int(prediction)

