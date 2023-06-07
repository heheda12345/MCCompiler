import torch
import onnx
from mc import add_size
import os
from onnxsim import simplify

def save_input_output(io_dir, inputs, outputs):
    os.makedirs(io_dir, exist_ok=True)
    for i, inp in enumerate(inputs):
        with open("{}/input{}.shape".format(io_dir, i), "w") as f:
            f.write(" ".join([str(x) for x in inp.shape]))
        with open("{}/input{}.bin".format(io_dir, i), "wb") as f:
            inp.cpu().detach().numpy().tofile(f)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    for i, out in enumerate(outputs):
        with open("{}/output{}.shape".format(io_dir, i), "w") as f:
            f.write(" ".join([str(x) for x in out.shape]))
        with open("{}/output{}.bin".format(io_dir, i), "wb") as f:
            out.cpu().detach().numpy().tofile(f)

def export(model, inp, onnx_dir, io_dir):
    outp = model(*inp)
    save_input_output(io_dir, inp, outp)
    torch.onnx.export(model, inp, onnx_dir, opset_version=11)
    model = onnx.load(onnx_dir)
    add_size.add_value_info_for_constants(model)
    model = onnx.shape_inference.infer_shapes(model)
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, onnx_dir)
