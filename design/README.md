# Decompose the function
```
python .decompose_function.py -model model.onnx -model_o model_o.onnx -m sigmoid
```
- Has three mode: decompose all softmax function, all sigmoid function, all exponential function.
- Decomposition Sequnece is: Softmax -> (quantization) -> sigmoid -> exponential
