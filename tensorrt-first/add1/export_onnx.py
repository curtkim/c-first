import torch

### Export
class Add(torch.nn.Module):
    def forward(self, x):
        return x+1

print(Add()(torch.arange(4, dtype=torch.float32)))


inputs = (torch.arange(4, dtype=torch.float32))
torch.onnx.export(Add(), inputs, 'add1.onnx', opset_version=11)


### Load and Check
import onnx

# Load the ONNX model
model = onnx.load("add1.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))


### Inference
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("add1.onnx")

outputs = ort_session.run(
    None,
    {"onnx::Add_0": np.arange(4).astype(np.float32)},
)
print(outputs[0])
