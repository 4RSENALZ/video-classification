数据集来源(经过了一些手动整合、筛选和标注）：https://www.kaggle.com/datasets/zhuosun03/bilibili-must-watch（第一个数据集：bili2019每周必看）  
https://github.com/zhaisc2001/dataset_bilibili?tab=readme-ov-file（另一个B站数据集）  
VGGish:https://github.com/tensorflow/models/tree/master/research/audioset/vggish  
NeXtVLAD:https://github.com/zhongzhh8/Video-classification-with-knowledge-distillation/blob/master/nextvlad.py  
模型框架：  
![fc3e171ee375a7b4fd140bd5d64e138](https://github.com/user-attachments/assets/b9e56fd1-fa78-4363-b807-2a9f8ffb3d7b)  
运行顺序：  
1.下载数据集，用yt-nlp工具通过URL 批量下载视频  
2.missing_videos.py--检查下载失败的视频  
3.建立一个sort.txt存放视频标签  
4.move_video.py--下载的视频移动至对应分类子文件夹  
5.data_split.py--分割数据集  
6.process.py--数据预处理，包括提取帧、音频、清洗标题等操作（可以用show_picture.py查看图像处理效果）  
7.extra_frame_feature.py--提取图像特征  
8.extra_audio_feature.py--提取音频特征  
9.extra_title_feature.py--提取标题特征  
10.nextvlad_cluster.py--聚合图像和音频特征  
11.train.py--模态融合,分类  
12.demo.py--简单直观地测试模型效果  
13.eval.py--评估模型（大部分指标被我改成以top5内有命中即预测正确的逻辑了）  
效果大概像这样：  
![image](https://github.com/user-attachments/assets/a3d34dac-bc96-4563-bea7-2fbc5279630d)  
我的数据集目录结构：  
```plaintext
datasets  
├── raw_data  
│   ├── train  
│   │   ├── video //训练集视频我实际上放在datasets/video文件夹了  
│   │   │    └── 各个分类的子文件夹（25类）  
│   │   ├── audio  
│   │   └── text  
│   ├── val  
│   │   ├── video  
│   │   ├── audio  
│   │   └── text  
│   ├── test  
│   │   ├── video  
│   │   ├── audio  
│   │   └── text  
│   └── video //训练集视频  
├── processed  
│    └── frames //视频提取出的帧图片  
│       ├── train  
│       ├── val  
│       └── test  
└── features  
    ├── frame_resnet //resnet提取的图像特征  
    │   ├── train  
    │   ├── val  
    │   └── test  
    ├── audio_vggish //vggish提取的音频特征  
    │   ├── train  
    │   ├── val  
    │   └── test  
    ├── text_bert //bert提取的文本特征  
    │   ├── train  
    │   ├── val  
    │   └── test  
    └── feature_nextvlad //聚合后的特征向量  
        ├── train  
        ├── val  
        └── test  
  
