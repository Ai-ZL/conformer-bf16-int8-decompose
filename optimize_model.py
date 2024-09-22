from onnxruntime.quantization.shape_inference import quant_pre_process
quant_pre_process(input_model='model.onnx',
                  output_model_path='model_optimize.onnx',
                  auto_merge=True)