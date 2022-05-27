# TRT-Hackathon2022-BIT-jedibobo
This project aims at TensorRT Hackathon2022 Contest(https://tianchi.aliyun.com/competition/entrance/531953/information))

## 原始模型
本小组选择了[CLIP:Contrastive Language-Image Pretraining](https://github.com/openai/CLIP)作为待优化的模型。
### 模型简介
- CLIP模型是OpenAI提出的在图片文本对上训练的模型，利用对比学习的方式完成算法设计，创新点在于将自然语言的抽象概念引入了计算机视觉领域。其本身是一种多模态的[模型](https://paperswithcode.com/methods/category/vision-and-language-pre-trained-models)。
- CLIP适用的下游任务十分广泛，根据论文介绍：其可在OCR、视频动作理解等看似不相关的领域取得很好的性能。
- 其在Zero-shot方面的性能十分强大，在很多数据集上要强于训练后的ResNet50。此外，CLIP还可以通过linear-probe的方式用于物体分类的任务。
- CLIP可以直接被用于视频中物体的检索。
- 模型的结构如下图所示：
![CLIP Approach](imgs/clip-approach.png)
CLIP主要由上方的文本编码器(text encoder)和下方的图片编码器(image encoder)组成。也是本项目要通过TensorRT加速的部分。  

训练时（上图(1)），通过对比学习的方法，将文本(text)和对应的图片(image)的正确匹配（矩阵中对角线）作为正样本，将其余配对作为负样本。实现利用文本作为图像的监督信号的“自监督训练”。

在做zero-shot推理时，输入的图片经过图片编码器，文本通过提示（prompt）的方式给文本编码器，然后计算最大匹配关系，得到预测的结果。  

本文也找到了一个CLIP用于视频中检索物体的[例子](https://github.com/johanmodin/clifs)，认为其具有潜在的应用价值。



### 模型优化的难点
如果模型可以容易地跑在TensorRT上而且性能很好，就没有必要选它作为参赛题目并在这里长篇大论了。相信你选择了某个模型作为参赛题目必然有选择它的理由。  
请介绍一下在模型在导出时、或用polygraphy/trtexec解析时、或在TensorRT运行时，会遇到什么问题。换句话说，针对这个模型，我们为什么需要额外的工程手段。

## 优化过程
这一部分是报告的主体。请把自己假定为老师，为TensorRT的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的TensorRT模型。  

建议：
- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

## 精度与加速效果
这一部分介绍优化模型在云主机上的运行效果，需要分两部分说明：  
- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
  - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。  
  - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力
- 性能：最好用图表展示不同batch size或sequence length下性能加速效果。
  - 一般用原始模型作为参考标准；若额外使用ONNX Runtime作为参考标准则更好。  
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：
- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。  

## Bug报告（可选）
提交bug是对TensorRT的另一种贡献。发现的TensorRT、或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。

对于每个bug，请标记上hackathon2022标签，并写好正文：
- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 8.4 GA仍可复现，然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 8.4 GA
    - Versions of CUDA, CUBLAS, CuDNN used
    - Container used
    - NVIDIA driver version
  - Reproduction Steps
    - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
  - Expected Behavior
    - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
  - Actual Behavior
    - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
  - Additional Notes
    - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

## 经验与体会（可选）
欢迎在这里总结经验，抒发感慨。
