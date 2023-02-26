from mc import compile
import onnx
import sys

def main():
    with open(sys.argv[1], 'rb') as f:
        model_onnx = onnx.load(f)
    compile(model_onnx.graph)

if __name__ == "__main__":
    main()