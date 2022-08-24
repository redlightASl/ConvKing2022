from onnxruntime.quantization import quantize_dynamic, QuantType
 
model_fp32 = './my_model.onnx'
model_quant = './model_quan.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
 

# import onnx
# from onnxruntime.quantization import quantize_qat, QuantType
 
# model_fp32 = 'path/to/the/model.onnx'
# model_quant = 'path/to/the/model.quant.onnx'
# quantized_model = quantize_qat(model_fp32, model_quant)