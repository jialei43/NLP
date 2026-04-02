import torch
from torch import nn

embedding = nn.Embedding(num_embeddings=100, embedding_dim=5, padding_idx=0)
out = embedding(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 0]]))
print(out)

print(out.shape)
