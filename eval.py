import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from train import MultiModalWeightedFusion 
from train import MultiModalDataset, CATEGORY2IDX, CATEGORIES 
from train import top_k_accuracy 

def compute_hit_at_k(preds, targets, k=1):
    topk_preds = torch.topk(preds, k=k, dim=1).indices
    hits = (topk_preds == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()

# def compute_map(preds, targets, k=5):
#     """Mean Average Precision for top-k predictions"""
#     num_classes = preds.size(1)
#     topk_preds = torch.topk(preds, k=k, dim=1).indices

#     # binary ground truth (one-hot)
#     y_true = torch.zeros_like(preds)
#     y_true[torch.arange(targets.size(0)), targets] = 1

#     # mask predictions to keep only top-k
#     mask = torch.zeros_like(preds)
#     mask.scatter_(1, topk_preds, 1.0)
#     y_pred = preds * mask

#     # convert to numpy
#     return average_precision_score(y_true.numpy(), y_pred.numpy(), average='macro')

# def compute_gap(preds, targets):
#     """Compute Global Average Precision (GAP)"""
#     probs, indices = torch.sort(preds, descending=True)
#     batch_size, num_classes = preds.shape

#     rows = []
#     for i in range(batch_size):
#         for j in range(num_classes):
#             rows.append({
#                 'video_id': i,
#                 'class_id': indices[i][j].item(),
#                 'score': probs[i][j].item(),
#                 'label': 1 if indices[i][j] == targets[i] else 0
#             })

#     rows = sorted(rows, key=lambda x: x['score'], reverse=True)

#     total_correct = 0
#     total_precision = 0
#     for i, row in enumerate(rows):
#         if row['label'] == 1:
#             total_correct += 1
#             total_precision += total_correct / (i + 1)

#     return total_precision / total_correct if total_correct > 0 else 0.0

def compute_gap(preds, targets, k=5):
    """
    GAP: 对所有样本 top-k 的预测结果打平，按置信度排序计算加权平均精度。
    preds: [N, C] tensor, logits 或概率
    targets: [N] tensor, 每个样本的真实标签（int）
    """
    topk_scores, topk_indices = torch.topk(preds, k=k, dim=1)  # [N, k]

    all_preds = []
    for i in range(preds.size(0)):
        for j in range(k):
            score = topk_scores[i, j].item()
            pred_label = topk_indices[i, j].item()
            is_hit = 1 if pred_label == targets[i].item() else 0
            all_preds.append((score, is_hit))

    # 按置信度从高到低排序
    all_preds.sort(key=lambda x: x[0], reverse=True)

    total_hits = 0
    gap = 0.0
    for idx, (score, is_hit) in enumerate(all_preds):
        if is_hit:
            total_hits += 1
            precision_at_i = total_hits / (idx + 1)
            gap += precision_at_i

    gap /= preds.size(0)  # N samples
    return gap

def compute_map(preds, targets):
    """
    mAP: 针对每个类别分别计算 AP，然后求平均。
    preds: [N, C] tensor, logits 或概率
    targets: [N] tensor, 每个样本的真实类别（int）
    """
    num_classes = preds.size(1)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    ap_list = []
    for c in range(num_classes):
        y_true = targets_onehot[:, c]
        y_score = preds_np[:, c]
        if y_true.sum() == 0:  # 类别 c 没有出现过，跳过
            continue
        ap = average_precision_score(y_true, y_score)
        ap_list.append(ap)

    if not ap_list:
        return 0.0  # 如果所有类别都没出现（不太可能）
    return sum(ap_list) / len(ap_list)

# ====== 评估函数 ======
def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_outputs, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for image, audio, text, label in dataloader:
            image, audio, text, label = image.to(device), audio.to(device), text.to(device), label.to(device)

            logits, _ = model(image, audio, text)

            if criterion:
                loss = criterion(logits, label)
                total_loss += loss.item()

            all_outputs.append(logits.cpu())
            all_labels.append(label.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    top5_preds = torch.topk(all_outputs, k=5, dim=1).indices

    # === Top-5 based Precision / Recall / F1 ===
    top5_preds = torch.topk(all_outputs, k=5, dim=1).indices
    preds_top5_correct = []

    for i in range(all_labels.size(0)):
        label = all_labels[i].item()
        pred_top5 = top5_preds[i].tolist()
        if label in pred_top5:
            preds_top5_correct.append(label)  # 预测对，按原标签计
        else:
            preds_top5_correct.append(pred_top5[0])  # 错误预测，随便取第一个错误的类

    preds_top5_correct = torch.tensor(preds_top5_correct)

    # 使用 sklearn 的 precision / recall / F1（加权）
    precision = precision_score(all_labels, preds_top5_correct, average="weighted", zero_division=0)
    recall = recall_score(all_labels, preds_top5_correct, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, preds_top5_correct, average="weighted", zero_division=0)
    
    # AUC
    try:
        probs = F.softmax(all_outputs, dim=1)
        auc = roc_auc_score(all_labels.numpy(), probs.numpy(), multi_class="ovo", average="macro")
    except Exception as e:
        auc = None
        print(f"AUC 计算失败: {e}")

    # Top-K、MAP、GAP
    top1 = compute_hit_at_k(all_outputs, all_labels, k=1)
    top5 = top_k_accuracy(all_outputs, all_labels, k=5)
    map5 = compute_map(all_outputs, all_labels)
    gap = compute_gap(all_outputs, all_labels)

    avg_loss = total_loss / len(dataloader) if criterion else None

    # 每类 Precision / Recall / F1
    per_class_precision = precision_score(all_labels, preds_top5_correct, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, preds_top5_correct, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, preds_top5_correct, average=None, zero_division=0)


    print("\nPer-Class Precision / Recall / F1:")
    print("(Row: Class Index (Name), Columns: Precision | Recall | F1-score)")
    for i, (p, r, f) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
        class_name = CATEGORIES[i]
        print(f"Class {i:2d} ({class_name}): [{p:.4f} {r:.4f} {f:.4f}]")

    print(f"\nOverall (Weighted Avg): [{precision:.4f} {recall:.4f} {f1:.4f}]")

    return avg_loss, top1, top5, map5, gap, auc, precision, recall, f1, None  # cm 设为 None（不再返回）


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型
    model = MultiModalWeightedFusion(use_attention=False).to(device)
    checkpoint_path = "checkpoints/V6_best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 数据
    test_dataset = MultiModalDataset(
        image_dir=r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/test/frame_nextvlad",
        audio_dir=r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/test/audio_nextvlad",
        text_dir=r"E:/毕业设计/datasets/bili_datasets/features/text_bert/test",
        split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 类别权重
    test_labels = [CATEGORY2IDX[cat] for _, cat in test_dataset.items]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CATEGORIES)),
        y=test_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

       # 评估
    test_loss, top1, top5, map5, gap, auc, precision, recall, f1, _ = evaluate(model, test_loader, device, criterion)

    # 输出结果
    print("\n====== Evaluation Summary ======")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Hit@1 (Top-1 Accuracy): {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"MAP@5: {map5:.4f}")
    print(f"GAP: {gap:.4f}")
    print(f"AUC (macro): {auc:.4f}" if auc is not None else "AUC 计算失败")
        
    
if __name__ == "__main__":
    main()