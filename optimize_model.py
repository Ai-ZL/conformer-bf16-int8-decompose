from onnxruntime.quantization.shape_inference import quant_pre_process
quant_pre_process(input_model='model.onnx', #model before processing
                  output_model_path='model_opt.onnx', #stored mmodel after processing
                  auto_merge=True)
