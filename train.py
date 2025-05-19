import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# 分类标签
CATEGORIES = [
    "车辆文化", "动物", "二创", "仿妆cos", "搞笑", "鬼畜", "绘画", "科技", "科普", "美食",
    "明星综合", "人文历史", "三农", "设计·创意", "生活", "时尚潮流", "手办·模玩", "特摄",
    "舞蹈", "音乐", "影视", "游戏", "运动综合", "职业综合", "综合"
]
CATEGORY2IDX = {name: i for i, name in enumerate(CATEGORIES)}

# ========== 1. 标签解析 ==========
#清理标题中的特殊符号
def clean_title(title):
    title = re.sub(r'\[BV[^\]]*\]', '', title)
    title = re.sub(r'[^\w\u4e00-\u9fa5\s]', '', title)
    title = re.sub(r'[\s_]+', '_', title)
    title = title.rstrip('_')
    return title[:50]

#利用sort.txt建立字典
def parse_label_map(sort_txt_path):
    label_map = {}
    with open(sort_txt_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # 忽略不规范行
            title, category = parts
            title = clean_title(title)
            if category in CATEGORY2IDX:
                label_idx = CATEGORY2IDX[category]
                label_map[title] = label_idx
    return label_map

def top_k_accuracy(outputs, labels, k=5):
    # outputs: [batch_size, num_classes] 的 logits 或概率
    # labels: [batch_size] 的真实标签（int）
    topk_preds = torch.topk(outputs, k=k, dim=1).indices  # [batch_size, k]
    labels = labels.view(-1, 1).expand_as(topk_preds)     # [batch_size, k]
    correct = (topk_preds == labels).any(dim=1).float()   # [batch_size]
    return correct.mean().item()

# ========== 2. 融合模型 ==========
#交叉注意力，加上反而很快就过拟合了，如果数据集大一点或许会有效果
# class CrossModalAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.query = nn.Linear(dim, dim)
#         self.key = nn.Linear(dim, dim)
#         self.value = nn.Linear(dim, dim)
#         self.scale = dim ** 0.5

#     def forward(self, query_input, context_input):
#         # query_input: [B, D], context_input: [B, D]
#         Q = self.query(query_input).unsqueeze(1)     # [B, 1, D]
#         K = self.key(context_input).unsqueeze(1)     # [B, 1, D]
#         V = self.value(context_input).unsqueeze(1)   # [B, 1, D]
#         attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, 1, 1]
#         attn_weights = torch.softmax(attn_scores, dim=-1)           # [B, 1, 1]
#         attended = torch.bmm(attn_weights, V).squeeze(1)            # [B, D]
#         return attended
    
# class MultiModalWithCrossAttention(nn.Module):
#     def __init__(self, image_dim=2048, audio_dim=512, text_dim=768, fused_dim=512, num_classes=25):
#         super().__init__()
#         self.image_proj = nn.Sequential(
#             nn.Linear(image_dim, fused_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )
#         self.audio_proj = nn.Sequential(
#             nn.Linear(audio_dim, fused_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )
#         self.text_proj = nn.Sequential(
#             nn.Linear(text_dim, fused_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )

#         # 双向 Cross-Attention
#         #self.cross_attn_image_to_text = CrossModalAttention(fused_dim)  # image -> text
#         #self.cross_attn_audio_to_text = CrossModalAttention(fused_dim)  # audio -> text
#         #self.cross_attn_text_to_image = CrossModalAttention(fused_dim)  # text -> image
#         #self.cross_attn_text_to_audio = CrossModalAttention(fused_dim)  # text -> audio

#         # 小型 MLP + BN 模块
#         def mlp_bn():
#             return nn.Sequential(
#                 nn.Linear(fused_dim, fused_dim),
#                 #nn.LayerNorm1d(fused_dim),
#                 nn.ReLU()
#             )

#         self.image_mlp = mlp_bn()
#         self.audio_mlp = mlp_bn()
#         self.text_mlp = mlp_bn()

#         self.weights = nn.Parameter(torch.ones(3))
#         self.classifier = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(fused_dim, num_classes)
#         )

#     def forward(self, image_feat, audio_feat, text_feat):
#         image_proj = self.image_proj(image_feat)
#         audio_proj = self.audio_proj(audio_feat)
#         text_proj = self.text_proj(text_feat)

#         # 图像和音频 attend 到文本
#         image_attn = self.cross_attn_image_to_text(image_proj, text_proj)
#         #audio_attn = self.cross_attn_audio_to_text(audio_proj, text_proj)

#         # 文本 attend 到图像和音频，平均融合
#         #text_attn_from_img = self.cross_attn_text_to_image(text_proj, image_proj)
#         #text_attn_from_aud = self.cross_attn_text_to_audio(text_proj, audio_proj)
#         #text_attn = (text_attn_from_img + text_attn_from_aud) / 2
#         text_attn = self.cross_attn_text_to_image(text_proj, image_proj)
        
#         # 残差 + 小MLP增强
#         image_fused = self.image_mlp(image_proj + image_attn)
#         audio_fused = self.audio_mlp(audio_proj)
#         text_fused = self.text_mlp(text_proj + text_attn)

#         # 加权融合
#         norm_weights = F.softmax(self.weights, dim=0)
#         fused = (
#             norm_weights[0] * image_fused +
#             norm_weights[1] * audio_fused +
#             norm_weights[2] * text_fused
#         )

#         return self.classifier(fused), norm_weights

class ChannelAttention(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction), #降维，减少计算量
            nn.ReLU(), #激活，提高非线性特征
            nn.Linear(in_dim // reduction, in_dim), #升维，恢复原始维度
            nn.Sigmoid() 
        )

    def forward(self, x):
        attn_weights = self.attn(x)
        return x * attn_weights


#加权处理三个模态
class MultiModalWeightedFusion(nn.Module):
    def __init__(self, image_dim=2048, audio_dim=512, text_dim=768,
                 fused_dim=1024, num_classes=25, use_attention=True):
        super().__init__()
        self.use_attention = use_attention  # 控制是否启用注意力模块（V6模型选False,V7模型选True）

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.weights = nn.Parameter(torch.ones(3))

        if self.use_attention:
            self.channel_attention = ChannelAttention(fused_dim)  # 可选注意力

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, image_feat, audio_feat, text_feat):
        image_proj = self.image_proj(image_feat)
        audio_proj = self.audio_proj(audio_feat)
        text_proj = self.text_proj(text_feat)

        norm_weights = F.softmax(self.weights, dim=0)
        fused = norm_weights[0] * image_proj + norm_weights[1] * audio_proj + norm_weights[2] * text_proj

        if self.use_attention:
            fused = self.channel_attention(fused)

        logits = self.classifier(fused)
        return logits, norm_weights



# ========== 3. 数据集类 ==========
class MultiModalDataset(Dataset):
    def __init__(self, image_dir, audio_dir, text_dir, split="train"):
        self.items = []
        self.image_dfs = {}
        self.audio_dfs = {}
        self.text_dfs = {}

        for category in CATEGORIES:
            image_path = os.path.join(image_dir, f"{category}_nextvlad.csv")
            audio_path = os.path.join(audio_dir, f"{category}_nextvlad.csv")
            text_path = os.path.join(text_dir, category, f"{category}.csv")

            if not (os.path.exists(image_path) and os.path.exists(audio_path) and os.path.exists(text_path)):
                continue

            image_df = pd.read_csv(image_path)
            audio_df = pd.read_csv(audio_path)
            text_df = pd.read_csv(text_path)

            # 清洗 file_name（去掉后缀，做 clean_title），统一存储
            def clean_names(df):
                df['file_name'] = df['file_name'].astype(str)
                df['file_name'] = df['file_name'].apply(lambda x: clean_title(os.path.splitext(x)[0]))
                return df

            image_df = clean_names(image_df)
            audio_df = clean_names(audio_df)
            text_df = clean_names(text_df)

            # 保存清洗后的 df
            self.image_dfs[category] = image_df
            self.audio_dfs[category] = audio_df
            self.text_dfs[category] = text_df

            # 找到三种模态都有的样本
            common_names = set(image_df['file_name']) & set(audio_df['file_name']) & set(text_df['file_name'])
            for file_name in common_names:
                self.items.append((file_name, category))  # 直接用文件名和分类作为标签

        print(f"[{split}] 样本数: {len(self.items)}")
        print(f"Loading data for category: {category}")
        print(f"Image path: {image_path}")
        print(f"Audio path: {audio_path}")
        print(f"Text path: {text_path}")
        print(f"Common file names in {category}: {common_names}")
                    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_name, category = self.items[idx]
        label = CATEGORY2IDX[category]  # 使用类别名称映射为标签
        image_vec = self.image_dfs[category].query("file_name == @file_name").iloc[0, 1:].to_numpy(dtype=np.float32)
        audio_vec = self.audio_dfs[category].query("file_name == @file_name").iloc[0, 1:].to_numpy(dtype=np.float32)
        text_vec = self.text_dfs[category].query("file_name == @file_name").iloc[0, 1:].to_numpy(dtype=np.float32)
        return torch.tensor(image_vec), torch.tensor(audio_vec), torch.tensor(text_vec), torch.tensor(label)

# ========== 4. 训练与验证函数 ==========
def add_noise(x, std=0.01):
    noise = torch.randn_like(x) * std
    return x + noise

def train(model, dataloader, optimizer, criterion, device, epoch, scheduler=None, save_dir="checkpoints"):
    model.train()
    total_loss = 0.0
    all_outputs, all_labels = [], []

    os.makedirs(save_dir, exist_ok=True)
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")

    for image_feat, audio_feat, text_feat, labels in loop:
        image_feat = image_feat.to(device)
        audio_feat = audio_feat.to(device)
        text_feat = text_feat.to(device)
        labels = labels.to(device)

        # 添加特征级噪声（只在训练时）
        image_feat = add_noise(image_feat, std=0.005)
        audio_feat = add_noise(audio_feat, std=0.01)
        text_feat = add_noise(text_feat, std=0.005)

        optimizer.zero_grad()
        outputs, weights = model(image_feat, audio_feat, text_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 每个 batch 执行 scheduler.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_outputs.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    top1 = top_k_accuracy(all_outputs, all_labels, k=1)
    top5 = top_k_accuracy(all_outputs, all_labels, k=5)
    avg_loss = total_loss / len(dataloader)

    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(save_dir, f"V9_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)

    return avg_loss, top1, top5


from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_outputs, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for image, audio, text, label in dataloader:
            image, audio, text = image.to(device), audio.to(device), text.to(device)
            label = label.to(device)

            logits, _ = model(image, audio, text)
            if criterion:
                loss = criterion(logits, label)
                total_loss += loss.item()

            all_outputs.append(logits.cpu())
            all_labels.append(label.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = torch.argmax(all_outputs, dim=1)

    # top-k 准确率
    top1 = top_k_accuracy(all_outputs, all_labels, k=1)
    top5 = top_k_accuracy(all_outputs, all_labels, k=5)
    avg_loss = total_loss / len(dataloader) if criterion else None

    # sklearn 指标（以top1为准）
    precision = precision_score(all_labels.numpy(), preds.numpy(), average='macro', zero_division=0)
    recall = recall_score(all_labels.numpy(), preds.numpy(), average='macro', zero_division=0)
    f1 = f1_score(all_labels.numpy(), preds.numpy(), average='macro', zero_division=0)

    return avg_loss, top1, top5, precision, recall, f1



def plot_curves(train_losses, val_losses, learning_rates):
    import matplotlib.pyplot as plt

    epochs = len(train_losses)

    # 损失值曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="red", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

    # 每个epoch计算一次学习率
    steps_per_epoch = len(learning_rates) // epochs
    epoch_lr = [
        sum(learning_rates[i * steps_per_epoch : (i + 1) * steps_per_epoch]) / steps_per_epoch
        for i in range(epochs)
    ]

    # 学习率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_lr, label="Learning Rate", color="green", linestyle="--", marker="^")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_rate_curve.png")
    plt.close()

    
def plot_topk_curves(train_top1s, val_top1s, train_top5s, val_top5s):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_top1s) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_top1s, label="Train Top-1", color="blue", linestyle="-", marker="o")
    plt.plot(epochs, val_top1s, label="Val Top-1", color="red", linestyle="-", marker="o")
    plt.plot(epochs, train_top5s, label="Train Top-5", color="blue", linestyle="--", marker="s")
    plt.plot(epochs, val_top5s, label="Val Top-5", color="red", linestyle="--", marker="s")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Top-1 and Top-5 Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("top1_and_top5.png")
    plt.close()


# ========== 5. 早停机制 ==========  
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False, save_path='best_model.pt'):
        self.patience = patience        # 容忍的 epoch 数
        self.delta = delta              # 最小改进值
        self.verbose = verbose
        self.save_path = save_path      # 最佳模型保存路径
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss initialized at {val_loss:.6f}. Saving model...")
            torch.save(model.state_dict(), self.save_path)
        elif val_loss < self.best_loss - self.delta:
            if self.verbose:
                print(f"Validation loss improved ({self.best_loss:.6f} → {val_loss:.6f}). Saving model...")
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss for {self.counter} epoch(s).")
            if self.counter >= self.patience:
                self.early_stop = True
                
# ========== 6. 训练流程 ==========
def run_training():
    sort_txt = r"E:/毕业设计/datasets/bili_datasets/raw_data/sort.txt"
    label_map = parse_label_map(sort_txt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalWeightedFusion().to(device)

    train_dataset = MultiModalDataset(
        image_dir=r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/train/frame_nextvlad",
        audio_dir=r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/train/audio_nextvlad",
        text_dir=r"E:/毕业设计/datasets/bili_datasets/features/text_bert/train",
        split="train"
    )
    val_dataset = MultiModalDataset(
        image_dir=r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/val/frame_nextvlad",
        audio_dir=r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/val/audio_nextvlad",
        text_dir=r"E:/毕业设计/datasets/bili_datasets/features/text_bert/val",
        split="val"
    )
    
    #类别加权
    all_train_labels = [CATEGORY2IDX[cat] for _, cat in train_dataset.items]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CATEGORIES)),
        y=all_train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"类别权重: {class_weights_tensor}")

    #criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #去掉类别权重试试？(确实是去掉之后效果更好了)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    num_epochs = 30
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    train_losses, val_losses = [], []
    train_top1s, train_top5s = [], []
    val_top1s, val_top5s = [], []
    learning_rates = []

    early_stopping = EarlyStopping(patience=7, delta=0.001, verbose=True, save_path='checkpoints/V9_best_model.pt')

    for epoch in range(num_epochs):
        # === 使用 train() 执行一轮训练 ===
        train_loss, train_top1, train_top5 = train(
            model, train_loader, optimizer, criterion, device, epoch, scheduler=scheduler)

        # 手动记录当前学习率
        learning_rates.append(scheduler.get_last_lr()[0])

        # Scheduler 步进
        #scheduler.step() 

        # 验证集评估
        val_loss, val_top1, val_top5, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device, criterion)

        # Early stopping 检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("早停触发！")
            break

        # 日志记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_top1s.append(train_top1)
        train_top5s.append(train_top5)
        val_top1s.append(val_top1)
        val_top5s.append(val_top5)

        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f}, Top1: {train_top1:.4f}, Top5: {train_top5:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Top1: {val_top1:.4f}, Top5: {val_top5:.4f}")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"Modal Weights = {F.softmax(model.weights, dim=0).detach().cpu().numpy()}")
    
    # 绘图
    plot_curves(train_losses, val_losses, learning_rates)
    plot_topk_curves(train_top1s, val_top1s, train_top5s, val_top5s)


if __name__ == "__main__":
    run_training()