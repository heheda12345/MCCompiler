from mc import compile
import onnx
import sys
import logging
import os
logging.basicConfig(
    format='%(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO)

def main():
    with open(sys.argv[1], 'rb') as f:
        model_onnx = onnx.load(f)
    model_name = os.path.split(sys.argv[1])[1].split('.')[0]
    print("model_name", model_name)
    compile(model_onnx.graph, f'tmp/{model_name}', f'../model/bin/{model_name}')

if __name__ == "__main__":
    main()