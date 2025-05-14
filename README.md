VGGish模型来源：https://github.com/tensorflow/models/tree/master/research/audioset/vggish

NeXtVLAD模型来源：https://github.com/zhongzhh8/Video-classification-with-knowledge-distillation/blob/master/nextvlad.py

数据集来源（经过了一些手工筛选、合并、重新标注）：https://www.kaggle.com/datasets/zhuosun03/bilibili-must-watch 和 https://github.com/zhaisc2001/dataset_bilibili?tab=readme-ov-file

整体模型框架如图：
![image](https://github.com/user-attachments/assets/a3d3c742-f41c-49d1-a9c3-39da70482e85)

运行顺序：
1.move_video.py--将下载的视频按照sort.txt移动到对应的分类文件夹中
2.dataset_splits.py--划分数据集
3.process.py--提取视频的帧图像、音频、标题（需要下载一个ffmpeg.exe到根目录中）
4.extra_frame_feature.py--提取图像特征
5.extra_audio_feature.py--提取音频特征
6.extra_title_feature.py--提取标题特征
7.nextvlad_cluster.py--分别聚合图像和音频特征
8.train.py--训练模型
9.demo.py--测试模型效果，对指定视频分类
10.eval.py--评估模型

效果如图
![image](https://github.com/user-attachments/assets/22fddfa0-1f14-42c3-a7c1-1d2de44b38bb)
