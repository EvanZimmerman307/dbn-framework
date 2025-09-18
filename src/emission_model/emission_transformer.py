import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict
import numpy as np

class Patchify(nn.Module):
    def __init__(self, c_in:int, d_model:int):
        """
        c_in: number of features/signals
        d_model: hidden dimension

        Linear projection of input features to the model's hidden dimension
        using a 1D convolution with kernel size 1.
        1D convolution that maps C input channels to `d_model` output channels.
        This effectively embeds each time step's feature vector into the transformer's 
        hidden dimension space, similar to a token embedding in sequence models, 
        but without splitting the input into discrete patches.

        So Patchify basically turns each feature vector at each timestep into an embedding of dimension d?
        """
        super().__init__()
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=1)
        # kernel_size=1 here suggests the design prioritizes treating each time step as 
        # an independent "token" for the transformer, 

    def forward(self, x):       # x: (B,C,T)
        return self.proj(x).transpose(1,2)  # (B,T,d)

class PosEnc(nn.Module):
    def __init__(self, d:int, max_len:int=1024):
        """
        Implements sinusoidal positional encoding for the transformer, 
        a standard technique to inject position information into sequence data since 
        transformers lack inherent positional awareness. Helps with understanding relative
        positioning.
        Seq_len > max_len causes error
        """
        super().__init__()
        pe = torch.zeros(max_len, d) # tensor of 0's of max_len x d
        pos = torch.arange(0, max_len).unsqueeze(1) # max _len x 1
        div = torch.exp(torch.arange(0, d, 2) * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d)
    def forward(self, x):        # x: (B,T,d)
        return x + self.pe[:, :x.size(1), :]

class TinyTransformer(nn.Module):
    def __init__(self, d=128, nhead=4, nlayers=2, pdrop=0.1, emb_dim=64):
        """
        Implements a compact transformer encoder for 
        sequence processing with dual outputs: classification logits and normalized embeddings.
        """

        super().__init__()
        # standard transformer encoder layers (each with multi-head self-attention, feed-forward networks, and dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=4*d,
                                               dropout=pdrop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)   # simple global pool
        self.head_cls = nn.Linear(d, 1) # Linear layer projecting to a single logit for binary classification
        self.head_emb = nn.Linear(d, emb_dim) # Linear layer projecting to `emb_dim`-dimensional embeddings


    def forward(self, x):        # x: (B,T,d)
        h = self.encoder(x)      # (B,T,d)
        # global average over time
        # Global average pooling reduces this to `(B, d)` by averaging across the time dimension
        h_pooled = h.mean(dim=1) # (B,d)
        logit = self.head_cls(h_pooled).squeeze(-1) # scalar prediction before sigmoid
        emb = F.normalize(self.head_emb(h_pooled), dim=-1) # normalized embedding vector
        return logit, emb

class EmissionTransformer(nn.Module):
    def __init__(self, c_in:int, d_model:int=128, nhead:int=4, nlayers:int=2,
                 pdrop:float=0.1, emb_dim:int=64, max_len:int=1024):
        """
        c_in: the number of features/signals
        d_model: hidden dimension of the model
        n_head: number of attention heads per layer
        n_layers: number of attention layers
        pdrop: encoder dropout
        emb_dim: dim of emdbedding
        max_len: the maximum sequence length the Transformer's positional encoding is prepared to handle.
        """
        super().__init__()
        self.patch = Patchify(c_in, d_model) # basically token embeddings
        self.pos = PosEnc(d_model, max_len) # encoding token position
        self.backbone = TinyTransformer(d_model, nhead, nlayers, pdrop, emb_dim)

    @torch.no_grad()
    def score(self, snippet: np.ndarray) -> Dict[str, np.ndarray|float]:
        x = torch.from_numpy(snippet).float().unsqueeze(0)      # (1,C,T)
        x = self.patch(x)                                       # (1,T,d)
        x = self.pos(x)
        logit, emb = self.backbone(x)
        prob = torch.sigmoid(logit).item()
        return {"logit": float(logit.item()), "prob": float(prob), "embedding": emb.squeeze(0).cpu().numpy()}

    def forward(self, x):   # training: x=(B,C,T)
        x = self.patch(x)   # (B,T,d)
        x = self.pos(x)
        return self.backbone(x)  # logit(B), emb(B,emb_dim)
