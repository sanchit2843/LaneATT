import torch
import torch.nn as nn
from lib.models import laneatt
import torch.nn.utils.prune as prune


model = laneatt.LaneATT(
    backbone="resnet18",
    anchors_freq_path="/Users/sanchittanwar/Desktop/Workspace/ENPM673/project4/LaneATT/data/tusimple_anchors_freq.pt",
)


total_params = 0
total_sparsed = 0
for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.2)

        total_sparsed += torch.sum(module.weight == 0)
        total_params += module.weight.nelement()
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.4)

print("global sparsity", float(total_sparsed) / float(total_params) * 100)
