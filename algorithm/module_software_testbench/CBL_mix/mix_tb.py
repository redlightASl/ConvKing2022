import torch
import torch.nn as nn
import torch.quantization
from onnxruntime.quantization import quantize_dynamic, QuantType


class CBL_MixStruct(nn.Module):
    def __init__(self):
        super(CBL_MixStruct, self).__init__()
        self.cbl = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3,3], stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cbl(x)
        return x

if __name__ == '__main__':
    in_data = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                              dtype=torch.float32)

    net = CBL_MixStruct()
    result = net(in_data)
    print(result)
    # print(result.int())

    # input_names = ['inputs']
    # output_names = ['outputs']
    # once_batch_size = 1
    # channel = 1
    # height = 12
    # width = 12
    # dummy_input = torch.randn(once_batch_size,
    #                             channel,
    #                             height,
    #                             width,
    #                             requires_grad=True)
    # net.cpu()
    # torch.onnx.export(net, dummy_input, "./conv_model.onnx",
    #                         export_params=True,
    #                         # opset_version=13,
    #                         do_constant_folding=True,
    #                         input_names = input_names,
    #                         output_names = output_names,
    #                         dynamic_axes={'modelInput' : {0 : 'batch_size'},
    #                         'modelOutput' : {0 : 'batch_size'}})

    # model_fp32 = './conv_model.onnx'
    # model_quant = './conv_model_quan.onnx'
    # quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)