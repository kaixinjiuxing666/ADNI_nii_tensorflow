# ADNI_nii_tensorflow
利用ADNI数据集和标签，在tensorflow框架上使用tensorlayer接口，通过架构u-net实现海马体的分割。

Datasets:ADNI dataset and labels.     Tools:TensorFlow,TensorLayer,Nibabel.     Net:u-net 

Tensorlayer中文文档：http://tensorlayercn.readthedocs.io/zh/latest/  英文文档：http://tensorlayer.readthedocs.io/en/latest/

本文参考项目：https://github.com/zsdonghao/u-net-brain-tumor

说明：由于阿里云机器学习PAI平台处于免费阶段，给每个用户免费提供一台M40（单精度浮点计算7TFlops）商用显卡实例，所以目前规划将模型在阿里云运行。考虑到其对tensorflow框架支持较好，但其定制的tensorflow框架比较封闭，可使用库较少，所以本项目分3步，1-step（数据预处理）在本地完成后再上传云端，2-step(模型设计)以及3-step(模型训练)直接上传云端并运行。

进度：1-step数据预处理：完成90%。已在本地测试完成，运行正常。剩余工作：优化数据预处理算法或方案，如使处理后的数据集体积更小等。
      2-step模型设计：完成80%。u-net已在本地测试完成，运行正常。剩余工作：单一网络效果不佳，可使用多种网络进行融合，从而使性能更好。
      3-step模型训练：完成60%。总体框架已在本地测试完成，运行正常。剩余工作：通过理解每个小模块甚至是每一行代码所要实现的功能，从而可以方便地修改代码来实现自己的想法，以及更方便地调参。
