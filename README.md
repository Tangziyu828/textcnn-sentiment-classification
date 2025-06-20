# textcnn-sentiment-classification
基于自构评论数据集的中文情感分类系统
# 中文情感分类系统（TextCNN）

本项目实现了一个基于自构中文评论数据集的情感分类系统，使用了 TextCNN 模型对评论进行正面 / 中性 / 负面分类。

## 💻 环境配置

- Python 3.9+
- pip
- 主要依赖：
  - torch==2.0.1
  - torchtext==0.18.0
  - pandas, matplotlib, seaborn
  - jieba
  - scikit-learn

安装方式：

```bash
pip install -r requirements.txt


## 📁 项目结构
AIProject/
├── data/                       # 存放评论数据集
│   └── text_data.csv
├── model/                      # 模型结构
│   └── textcnn.py
├── utils/                      # 工具模块
│   └── dataset_utils.py
├── train_textcnn_clean.py     # 主训练脚本
├── data_analysis.py           # 可视化脚本
├── result/                     # 输出模型和图表
├── README.md                  # 项目说明文档

