import torch
import torch.nn.functional as F
import torchvision.models as models

r18 = models.resnet18(pretrained=True)       # We now have an instance of the pretrained model
r18_scripted = torch.jit.script(r18)         # *** This is the TorchScript export
dummy_input = torch.rand(1, 3, 224, 224)     # We should run a quick test

unscripted_output = r18(dummy_input)         # Get the unscripted model's prediction...
scripted_output = r18_scripted(dummy_input)  # ...and do the same for the scripted version

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

print('Python model top 5 results:\n  {}'.format(unscripted_top5))
print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))

r18_scripted.save('r18_scripted.pt')