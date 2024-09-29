# Decompose the function
```
python decompose_function.py -model model.onnx -model_o model_o.onnx -m sigmoid
```
- Has three mode: decompose all softmax function, all sigmoid function, all exponential function.
- Decomposition Sequnece is: Softmax -> (quantization) -> sigmoid -> exponential

# Convert model non-linear function calculation datatype
- Has two mode: turn all non-linear function to bfloat16 (**bf16**), some are bfloat16 and some are float16 (**bf_fp_16**).
