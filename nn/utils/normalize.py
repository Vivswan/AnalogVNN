import torch


def normalize_model(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                norm = p.norm()
                if norm != 0:
                    p.data /= norm


def normalize_cutoff_model(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data = torch.clamp(p.data, min=-1, max=1)
