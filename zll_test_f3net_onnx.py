''' autodocstring '''
import time

import onnxruntime as ort
import numpy as np

from imgcat import imgcat
import cv2
import torch
from torch.autograd import Variable
from f3net import F3Net


def load_torch_model(model_path, device):
    _dict = torch.load(model_path, map_location=device)
    state_dict = {}
    for k, v in _dict.items():
        if k.startswith('module'):
            k = k[7:]
        state_dict[k] = v
    return state_dict


# Change shapes and types to match model
test_image = "./157234709924612_b59fa481-57c5-430c-9577-2d7e0183b8fc.png"
img = cv2.imread(test_image, cv2.IMREAD_COLOR)
img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_arr.shape)
img_arr = cv2.resize(img_arr, (448, 448))

mean, std = np.array([[[124.55, 118.90, 102.94]]]), np.array([[[56.77, 55.97, 57.50]]])
img_arr = (img_arr - mean) / std
img_arr = np.transpose(img_arr, [2, 0, 1])
img_arrs = np.expand_dims(img_arr, axis=0).astype(np.float32)
print(img_arrs.shape)

sess = ort.InferenceSession("./f3net.onnx", providers=["CPUExecutionProvider"])

# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.

for _ in range(20):
    since = time.time()
    results_ort = sess.run([], {"input": img_arrs})
    print(f"time cost: {time.time()-since:.4f} seconds")

device = 'cpu'
model_path = "./pytorch_model.bin"
model = F3Net()
params = load_torch_model(model_path, device)
model.load_state_dict(params)
model.to(device)
model.eval()

img_arrs = torch.from_numpy(img_arrs)

for _ in range(20):
    since = time.time()
    outputs = model(img_arrs)
    print(f"time cost: {time.time()-since:.4f} seconds")

for ort_res, tf_res in zip(results_ort[0], outputs[0].detach().numpy()):
    np.testing.assert_allclose(ort_res, tf_res, rtol=5e-5, atol=5e-5)

print("Results match")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


out = results_ort[0]
pred = sigmoid(out[0, 0]) * 255
pred[pred < 20] = 0
print(np.max(pred), np.min(pred))
# pred = pred[:, :, np.newaxis]
pred = np.round(pred)
imgcat(pred)
