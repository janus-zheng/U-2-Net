''' autodocstring '''
import time

import onnxruntime as ort
import numpy as np

from imgcat import imgcat
import cv2
import torch
from torch.autograd import Variable
from model import U2NET


# Change shapes and types to match model
test_image = "./157234709924612_b59fa481-57c5-430c-9577-2d7e0183b8fc.png"
img = cv2.imread(test_image, cv2.IMREAD_COLOR)
img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_arr.shape)
img_arr = cv2.resize(img_arr, (256, 256))
img_arr = np.transpose(img_arr, [2, 0, 1])
img_arrs = np.expand_dims(img_arr, axis=0).astype(np.float32)
print(img_arrs.shape)

sess = ort.InferenceSession("./u2net.onnx", providers=["CPUExecutionProvider"])

# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.
res_keys = ['output']

for _ in range(20):
    since = time.time()
    results_ort = sess.run(res_keys, {"input": img_arrs})
    print(f"time cost: {time.time()-since:.4f} seconds")

model = U2NET(3, 1)
model_path = "./saved_models/u2net/u2net.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

img_arrs = torch.from_numpy(img_arrs)
img_arrs = Variable(img_arrs)

for _ in range(20):
    since = time.time()
    d1, d2, d3, d4, d5, d6, d7 = model(img_arrs)
    print(f"time cost: {time.time()-since:.4f} seconds")

for ort_res, tf_res in zip(results_ort[0], d1.detach().numpy()):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-5)

print("Results match")
