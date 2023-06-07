from mc import compile
import onnx
import sys
import logging
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', type=str, help='onnx file')
parser.add_argument('--io_dir', type=str, help='directory to store input and output data')
args = parser.parse_args()

logging.basicConfig(
    format='%(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO)

def main():
    with open(args.onnx, 'rb') as f:
        model_onnx = onnx.load(f)
    model_name = os.path.split(args.onnx)[1].split('.')[0]
    print("model_name", model_name)
    compile(model_onnx.graph, f'tmp/{model_name}', args.io_dir)
    print("start run")
    os.system(f'tmp/{model_name}/run')

if __name__ == "__main__":
    main()