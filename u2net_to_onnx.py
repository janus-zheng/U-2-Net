
import torch
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

model_name = "u2netp"

if model_name == "u2net":
    model = U2NET(3, 1)
    model_path = "./saved_models/u2net/u2net.pth"
if model_name == "u2netp":
    model = U2NETP(3, 1)
    model_path = "./saved_models/u2netp/u2netp.pth"

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

input_names = ["input"]
output_names = ["output"]
input_tensor = torch.zeros([1, 3, 320, 320])

# I change the network to get 1 output: d1
torch.onnx.export(model,
                  input_tensor,
                  f"{model_name}.onnx",
                  opset_version=12,
                  export_params=True,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes={"input": {0: "batch"},
                                "output": {0: "batch"}})
