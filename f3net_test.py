import time

import torch
import cv2
from imgcat import imgcat
import numpy as np
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

mean, std = np.array([[[124.55, 118.90,
                        102.94]]]), np.array([[[56.77, 55.97, 57.50]]])


test_image = "./157234709924612_b59fa481-57c5-430c-9577-2d7e0183b8fc.png"
img = cv2.imread(test_image, cv2.IMREAD_COLOR)
img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_arr.shape)
img_arr = cv2.resize(img_arr, (256, 256))


img_arr = (img_arr - mean) / std
img_arr = torch.from_numpy(img_arr)
img_arr = img_arr.permute(2, 0, 1)
img_arr = img_arr.to(device).float()
img_arr = torch.unsqueeze(img_arr, dim=0)

for _ in range(20):
    since = time.time()
    outputs = model(img_arr)
    print(f"time cost: {time.time()-since:.4f} seconds")

out = outputs[0]
pred = (torch.sigmoid(out[0, 0]) * 255).detach().numpy()
pred[pred < 20] = 0
# pred = pred[:, :, np.newaxis]
pred = np.round(pred)
imgcat(pred)
