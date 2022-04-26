import torch
import torch.nn.functional as F

### Export
class Conv2d(torch.nn.Module):
    def forward(self, input):
        # input: (batch_size, in_channels, height, width)
        # weight: (out_channels, in_channels, kernel_height, kernel_width)

        filter = torch.ones(3, 3, 3, 3, dtype=torch.float32) / 27
        return F.conv2d(input, filter, padding=1),

inputs = (torch.ones(1, 3, 500, 500, dtype=torch.float32))
torch.onnx.export(Conv2d(), inputs, 'conv2d.onnx', opset_version=11)


### Inference
import onnxruntime as ort
import numpy as np

img = np.ones((1, 3, 500, 500), dtype=np.float32)
print(img.shape)
ort_session = ort.InferenceSession("conv2d.onnx")

outputs = ort_session.run(
    None,
    {"onnx::Conv_0": img},
)
print(outputs[0].shape)
