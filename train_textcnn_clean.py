import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import jieba

from model.textcnn import TextCNN
from utils.dataset_utils import TextDataset

print("🔍 [阶段1] 正在加载数据...")
df = pd.read_csv("情感评论数据集_sample.csv", encoding="utf-8-sig")
df["tokens"] = df["content"].apply(lambda x: [w for w in jieba.cut(x) if w.strip()])

# 构建词汇表
def yield_tokens(data_iter):
    for tokens in data_iter:
        yield tokens

print("🔍 [阶段2] 构建词表...")
vocab = build_vocab_from_iterator(yield_tokens(df["tokens"]), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

MAX_LEN = 30
df["input_ids"] = df["tokens"].apply(lambda x: vocab(x)[:MAX_LEN] + [vocab["<pad>"]] * max(0, MAX_LEN - len(x)))

print("🔍 [阶段3] 划分训练测试集...")
X_train, X_test, y_train, y_test = train_test_split(df["input_ids"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42)
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

print(f"✅ 训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(len(vocab), 100, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("🚀 [阶段4] 开始训练模型...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
print("✅ 模型训练完成！")

print("🧪 [阶段5] 开始测试模型...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        out = model(X)
        preds = torch.argmax(out, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.tolist())

print("📊 分类报告：")
print(classification_report(all_labels, all_preds, target_names=["class0", "class1", "class2"]))

print("📈 正在生成混淆矩阵图...")
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0", "1", "2"],
            yticklabels=["0", "1", "2"],
            cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("TextCNN Confusion Matrix")
plt.tight_layout()
plt.savefig("textcnn_confusion_matrix.png")
plt.show()
