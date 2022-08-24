import numpy as np
import torch
import onnxruntime as ort

test_data = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]],
                              dtype=torch.float)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()

ort_session = ort.InferenceSession("conv_model_quan.onnx",
                                   providers=['CPUExecutionProvider'])
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_data)}
ort_outs = ort_session.run(None, ort_inputs)

# ort_outs = np.array(ort_outs[0][0]).astype(dtype=int).tolist()

print(ort_outs)
