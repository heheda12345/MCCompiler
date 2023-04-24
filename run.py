from mc import compile
import onnx
import sys
import logging
logging.basicConfig(
    format='%(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO)

def main():
    with open(sys.argv[1], 'rb') as f:
        model_onnx = onnx.load(f)
    compile(model_onnx.graph)

if __name__ == "__main__":
    main()