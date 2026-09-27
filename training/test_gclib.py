"""Self-check: sparse-adjacency GCNEncoder == PyG edge-list GCNConv (outputs + grads).
Run once on the GPU box before training (seconds, synthetic graph, no real data):

    python test_gclib.py            # uses cuda if available
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

import gclib

dev = "cuda" if torch.cuda.is_available() else "cpu"
tol = 1e-4 if dev == "cuda" else 1e-5      # GPU sums in a different order


class EdgeListEnc(nn.Module):              # the pre-sparse GCNEncoder, as reference
    def __init__(self, i, h, o, layers):
        super().__init__()
        dims = [i] + [h] * (layers - 1) + [o]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims, dims[1:]))
        self.acts = nn.ModuleList(nn.PReLU(b) for b in dims[1:])

    def forward(self, x, ei, ew=None):
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x, ei, ew))
        return x


def check(name, x, ei, ew):
    old = EdgeListEnc(x.size(1), 64, 32, 2).to(dev)
    new = gclib.GCNEncoder(x.size(1), 64, 32, 2).to(dev)
    new.load_state_dict(old.state_dict())
    yo, yn = old(x, ei, ew), new(x, gclib._adj(ei, x.size(0), ew))
    g = torch.randn_like(yo)
    (yo * g).sum().backward(); (yn * g).sum().backward()
    assert torch.allclose(yo, yn, atol=tol), f"{name}: outputs differ {float((yo - yn).abs().max()):.2e}"
    for po, pn in zip(old.parameters(), new.parameters()):
        assert torch.allclose(po.grad, pn.grad, atol=tol), f"{name}: grads differ"
    print(f"{name}: ok (max diff {float((yo - yn).abs().max()):.1e})")


torch.manual_seed(0)
n, E = 2000, 20000
x = torch.randn(n, 16, device=dev)
e = torch.randint(0, n, (2, E), device=dev)
e = torch.cat([e, e[:, :200], torch.arange(50, device=dev).repeat(2, 1)], 1)   # dupes + self-loops
check("symmetric unweighted", x, torch.cat([e, e.flip(0)], 1), None)
d = torch.cat([torch.randint(0, n, (2, E), device=dev), torch.arange(n, device=dev).repeat(2, 1)], 1)
check("directed weighted (diffusion-like)", x, d, torch.rand(d.size(1), device=dev) * 2 + 0.01)
print(f"gclib self-check ok on {dev}")
