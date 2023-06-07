# MCCompiler: A Deep Learning Compiler for Transformer-based Models
Usage:
1. Add this library to PYTHON_PATH
    ```
    cd MCCompiler
    export PYTHONPATH=`pwd`:$PYTHONPATH
    ```
2. export a model to ONNX format and save sample input&output to io_dir
    ```
    python3 example/mm_trans.py --onnx tmp/test.onnx --io_dir tmp/test
    ```
3. compile and run the model
    ```
    python3 run.py --onnx tmp/test.onnx --io_dir tmp/test
    ```