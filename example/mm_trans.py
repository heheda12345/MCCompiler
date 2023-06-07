import torch
import argparse
from mc.export import export

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', type=str, help='onnx file')
parser.add_argument('--io_dir', type=str, help='directory to store input and output data')
args = parser.parse_args()

class MMTrans(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        z = torch.matmul(x, y)
        o = torch.transpose(z, 0, 1)
        return o


if __name__ == '__main__':
    model = MMTrans()
    x = torch.randn((12, 10, 10), device='cuda')
    y = torch.randn((12, 10, 64), device='cuda')
    export(model, (x, y), args.onnx, args.io_dir)
