# ConvKing2022

2022 UNIV "PangoMicro Cup" Project. Aming at deploy a whole set of MobileNetv2-YOLOv3 CNN Algorithm on the PGL22G FPGA Board. Because of some difficulties, we did not achieve this target. Maybe we could perfect this repo some other day, however, we put the whole project on GitHub for the needs.

Currently, this system is able to perform recognize and classify on PC platform. It also can sort yellow or orange on FPGA. And we finished all the RTL algorithm and TestBench for PGL22G, expect a useable interconnection. Quantization was performed by onnx and PyTorch, which is under project dir.

> We were confused by Interconnection :(

****

2022第六届集成电路创新创业大赛紫光同创杯赛道作品。预期成果是部署整套经过修改的MObileNetv2-YOLOv3卷积神经网络算法（基于VerilogHDL）到PGL22G板卡。由于自身技术原因和其他部署上的问题，我们未能完成该目标。之后预期择日打磨该项目，目前将部分源码发布到GitHub供参考

目前该项目可以在PC平台上完成识别和分类任务，在FPGA平台进行黄色或橘色的颜色识别。RTL和TestBench代码已经完成，但缺少合适的互联架构。算法优化量化工作分别通过onnx和PyTorch的API完成，也包含在项目目录中。

> 在构建过程中互联架构的硬件实现对我们造成了很大困扰

****

**基于FPGA平台和YOLO算法实现水果识别**

2022第六届集创赛紫光同创杯项目

## 目录说明

* algorithm：PC上实现的算法
* dataset：神经网络使用到的数据集
* doc：设计文档
* project：紫光同创FPGA上实现的主工程
* project_testbench：Xilinx FPGA上实现的测试工程
* ref：参考文献归档
* report：设计报告归档
* rtl：所有FPGA rtl源文件
* tools：设计过程中使用到的脚本
