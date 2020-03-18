import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
m_swish = MemoryEfficientSwish()
class EfficientNetWrapper(nn.Module):
    def __init__(self):
        super(EfficientNetWrapper, self).__init__()
        
        # Load imagenet pre-trained model 
        self.effNet = EfficientNet.from_pretrained('efficientnet-b5', in_channels=1)
        
        self.effNet._fc = nn.Identity()
        self.effNet._swish = nn.Identity()
        # Appdend output layers based on our date
        self.fc_root = nn.Linear(in_features=2048, out_features=168)
        self.fc_vowel = nn.Linear(in_features=2048, out_features=11)
        self.fc_consonant = nn.Linear(in_features=2048, out_features=7)
        
    def forward(self, X):
        output = self.effNet(X)
        output_root = m_swish(self.fc_root(output))
        output_vowel = m_swish(self.fc_vowel(output))
        output_consonant = m_swish(self.fc_consonant(output))
        
        return output_vowel, output_root, output_consonant