''' only work well on (448, 448) size image '''
import torch
from f3net import F3Net


def load_torch_model(model_path, device):
    _dict = torch.load(model_path, map_location=device)
    state_dict = {}
    for k, v in _dict.items():
        if k.startswith('module'):
            k = k[7:]
        state_dict[k] = v
    return state_dict


device = 'cpu'
model_path = "./pytorch_model.bin"
model = F3Net()
params = load_torch_model(model_path, device)
model.load_state_dict(params)
model.to(device)
model.eval()

input_names = ["input"]
output_names = ["output"]
input_tensor = torch.zeros([1, 3, 448, 448])

# I change the network to get 1 output: d1
torch.onnx.export(model,
                  input_tensor,
                  "f3net.onnx",
                  opset_version=13,
                  export_params=True,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes={"input": {0: "batch"},
                                "output": {0: "batch"}})
