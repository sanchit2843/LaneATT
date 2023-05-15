import torch
import torch.nn as nn
from lib.models import laneatt
import torch.nn.utils.prune as prune
import torch_pruning as tp

model = laneatt.LaneATT(
    backbone="resnet18",
    anchors_freq_path="/mnt/workspace/UMD/ENPM673/final_project/LaneATT/data/tusimple_anchors_freq.pt",
    topk_anchors=1000,
)
model.load_state_dict(
    torch.load(
        "/mnt/workspace/UMD/ENPM673/final_project/LaneATT/laneatt_experiments/experiments/laneatt_r18_tusimple/models/model_0100.pt"
    )["model"]
)
model = model  # .cuda()
imp = tp.importance.MagnitudeImportance(p=2)

ignored_layers = []
total = 0
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d) == False:  # ignore the classifier
        ignored_layers.append(m)
    total += 1

pruner = tp.pruner.MagnitudePruner(
    model=model,
    example_inputs=torch.randn(1, 3, 360, 640),  # .cuda(),
    importance=imp,  # Importance Estimator
    global_pruning=False,  # Please refer to Page 9 of https://www.cs.princeton.edu/courses/archive/spring21/cos598D/lectures/pruning.pdf
    ch_sparsity=0.5,  # global sparsity for all layers
    # ch_sparsity_dict = {model.conv1: 0.2}, # manually set the sparsity of model.conv1
    iterative_steps=1,  # number of steps to achieve the target ch_sparsity.
    ignored_layers=ignored_layers,  # ignore some layers such as the finall linear classifier
    # unwrapped_parameters=[ (model.features[1][1].layer_scale, 0), (model.features[5][4].layer_scale, 0) ],
)

# Model size before pruning
base_macs, base_nparams = tp.utils.count_ops_and_params(
    model, torch.randn(1, 3, 360, 640)  # .cuda()
)
pruner.step()

# Parameter & MACs Counter
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(
    model, torch.randn(1, 3, 360, 640)  # .cuda()
)
print("The pruned model:")
print(model)
print("Summary:")
print("Params: {:.2f} M => {:.2f} M".format(base_nparams / 1e6, pruned_nparams / 1e6))
print("MACs: {:.2f} G => {:.2f} G".format(base_macs / 1e9, pruned_macs / 1e9))

# Test Forward
output = model(torch.randn(1, 3, 360, 640))
print("Output.shape: ", output.shape)

# Test Backward
loss = torch.nn.functional.cross_entropy(output, torch.randint(1, 1000, (1,)))
loss.backward()
# total_params = 0
# total_sparsed = 0
# for name, module in model.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name="weight", amount=0.2)

#         total_sparsed += torch.sum(module.weight == 0)
#         total_params += module.weight.nelement()
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name="weight", amount=0.4)

# print("global sparsity", float(total_sparsed) / float(total_params) * 100)
