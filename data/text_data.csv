import pandas as pd
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# 读取CSV文件
df = pd.read_csv("情感评论数据集_sample.csv", encoding="utf-8-sig")

# 停用词列表（可自行扩展）
stopwords = set([
    "的", "了", "和", "是", "我", "也", "就", "都", "很", "还", "但", "有", "在",
    "不", "没", "一个", "这", "那", "呢", "吧", "啊", "吗", "着", "被", "到"
])

# 分词 + 去停用词
df["tokens"] = df["content"].apply(lambda x: [w for w in jieba.cut(x) if w not in stopwords and w.strip()])

# 构造词频统计
all_words = [word for tokens in df["tokens"] for word in tokens]
word_freq = Counter(all_words)

# 生成词云图
wordcloud = WordCloud(font_path="msyh.ttc", width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# 可视化输出
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 词云图
axes[0].imshow(wordcloud, interpolation='bilinear')
axes[0].axis("off")
axes[0].set_title("词云图 - 高频词")

# 情感类别柱状图
label_map = {0: "负面", 1: "正面", 2: "中性"}
label_counts = df["label"].map(label_map).value_counts()
axes[1].bar(label_counts.index, label_counts.values, color=["red", "green", "gray"])
axes[1].set_title("情感分布柱状图")
axes[1].set_xlabel("情感类别")
axes[1].set_ylabel("样本数量")

plt.tight_layout()
plt.savefig("情感数据分析可视化.png")
plt.show()
