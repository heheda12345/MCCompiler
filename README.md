# MCCompiler: A Deep Learning Compiler for Transformer-based Models
## Dependency
* python: onnx >= 1.14, onnxsim, numpy, torch (optional, for model export only)
* c++ libraries: cuda, cublas, curand
## Usage
1. Build dependency
    ```
    mkdir build
    g++ mc/operators/cuda/cublas.cpp -o build/cublas_util -lcublas -lcudart -lcublasLt -lcurand 
    ```
2. Add this library to PYTHON_PATH
    ```
    cd MCCompiler
    export PYTHONPATH=`pwd`:$PYTHONPATH
    ```
3. export a model to ONNX format and save sample input&output to io_dir
    ```
    python3 example/mm_trans.py --onnx tmp/test.onnx --io_dir tmp/test
    ```
4. compile and run the model
    ```
    python3 run.py --onnx tmp/test.onnx --io_dir tmp/test
    ```