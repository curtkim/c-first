import torch


class Add(torch.nn.Module):
    def forward(self, x):
        return x+1

print(Add()(torch.arange(4, dtype=torch.float32)))


inputs = (torch.arange(4, dtype=torch.float32))
torch.onnx.export(Add(), inputs, 'add1.onnx', opset_version=11)