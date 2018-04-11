# ADNI_nii_tensorflow
利用ADNI数据集和标签，在tensorflow框架上使用tensorlayer接口，通过架构u-net实现海马体的分割。

Datasets:ADNI dataset and labels.     Tools:TensorFlow,TensorLayer,Nibabel.    Net:u-net 

首先配置本地运行环境：

1、安装Anaconda3(这是一个开源的python发行版本，简单说就是一个集成了大量python库的管理工具，它直接包含最新版python以及我们常用的各种python库，包括numpy,matplotlib,pandas等等，当然连安装python也省了)。

2、安装tensorflow深度学习框架，以及nibabel,git等未直接安装的库都可通过anaconda3安装。

3、安装tensorlayer扩展库，详情参考下文相关网址。

Tensorlayer中文文档：http://tensorlayercn.readthedocs.io/zh/latest/  英文文档：http://tensorlayer.readthedocs.io/en/latest/

本文参考项目：https://github.com/zsdonghao/u-net-brain-tumor
