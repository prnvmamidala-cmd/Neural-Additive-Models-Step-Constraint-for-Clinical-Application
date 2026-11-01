"""Shared NAM models and training loop (unchanged from the accepted paper's protocol).
Adam lr 1e-3, BCE loss, 150 fixed epochs, batch 64, no early stopping and no model selection.
The held-out 20% split is only scored after training (per-epoch test curves are logged for plotting only)."""
import random, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
EPOCHS, BATCH, LR = 150, 64, 1e-3

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class HardStepActivation(nn.Module):
    """f(x)=b0 if x<t0; b1 if t0<=x<t1; b2 if x>=t1 (boolean masks, no soft approximation)."""
    def __init__(self, t0=0.7, t1=1.0):
        super().__init__(); self.t0, self.t1 = float(t0), float(t1)
        self.b0, self.b1, self.b2 = (nn.Parameter(torch.zeros(())) for _ in range(3))
    def forward(self, x):
        x = x.view(-1); out = torch.zeros_like(x)
        out = torch.where(x < self.t0, self.b0.expand_as(x), out)
        out = torch.where((x >= self.t0) & (x < self.t1), self.b1.expand_as(x), out)
        return torch.where(x >= self.t1, self.b2.expand_as(x), out)

class VanillaNAM(nn.Module):
    def __init__(self):
        super().__init__(); self.f_si, self.f_age = FeatureNet(), FeatureNet()
    def forward(self, si, age): return self.f_si(si) + self.f_age(age)

class StepNAM(nn.Module):
    def __init__(self, t=(0.7, 1.0)):
        super().__init__(); self.f_si, self.f_age = HardStepActivation(*t), FeatureNet()
    def forward(self, si, age): return self.f_si(si) + self.f_age(age)

def train_model(model, loader, te_si, te_age, te_y, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.BCEWithLogitsLoss()
    h = {"train_bce": [], "test_bce": []}
    for _ in range(epochs):
        model.train(); tot = n = 0
        for a, b, c in loader:
            opt.zero_grad(); loss = crit(model(a, b), c); loss.backward(); opt.step()
            tot += loss.item() * len(c); n += len(c)
        model.eval()
        with torch.no_grad(): h["test_bce"].append(crit(model(te_si, te_age), te_y).item())
        h["train_bce"].append(tot / n)
    return h

@torch.no_grad()
def predict(model, si, age):
    model.eval(); return torch.sigmoid(model(si, age)).numpy()
