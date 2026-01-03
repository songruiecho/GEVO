import os
from PIL import Image

def split_labels(s):
    for sep in ['；', ';', '、', ',', '，', ' ']:
        s = s.replace(sep, '|')
    return [x.strip() for x in s.split('|') if x.strip()]

def parse_ground_truth(gt_str):
    parts = split_labels(gt_str)
    mapping = {}
    for part in parts:
        if '-' in part:
            idx, label = part.split('-', 1)
            try:
                idx = int(idx.strip())
                mapping[idx] = label.strip()
            except:
                pass
    if mapping:
        return [mapping[i] for i in sorted(mapping.keys())]
    else:
        return parts

def parse_prediction(pred_str):
    return split_labels(pred_str)

def position_accuracy(gt, pred):
    n = len(gt)
    matches = sum(1 for a, b in zip(gt, pred) if a == b)
    return matches / n if n > 0 else 0.0

def rank_acc(cfg):
    """
    计算逐位置排序准确率(Position Accuracy)

    Args:
        file_path: 文件路径，每行格式 'ground_truth \t predicted'

    Returns:
        accuracies: 每条样本的逐位置准确率列表
        avg_acc: 所有样本平均准确率
    """
    accuracies = []
    with open(f'results/{cfg.task}_{cfg.VLM}.txt', 'r', encoding='utf-8') as f:
        results = [each.strip() for each in f.readlines()]
        for line in results:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            gt_str, pred_str = line.split('\t', 1)
            gt = parse_ground_truth(gt_str)
            pred = parse_prediction(pred_str)
            acc = position_accuracy(gt, pred)
            accuracies.append(acc)

    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"准确率: {avg_acc:.4f} ({len(accuracies)})")
    return accuracies, avg_acc



def cal_acc(cfg):
    with open(f'results/{cfg.task}_{cfg.VLM}.txt', 'r', encoding='utf-8') as f:
        results = [each.strip().split('\t') for each in f.readlines()]
        correct = 0
        total = len(results)
        for item in results:
            try:
                pred, label = item  # 分割模型输出和真实答案
                if label.strip() in pred.strip():  # 去掉首尾空格再比较
                    correct += 1
            except:
                continue
        accuracy = correct / total
        print(f"准确率: {accuracy:.4f} ({correct}/{total})")



MIN_SIZE = 50
def enlarge_to_min_size(img, min_size=MIN_SIZE):
    """若任一边小于 min_size，则按比例放大到最小边 >= min_size"""
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img

    scale = min_size / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.BICUBIC)


def process_images(root_dir="processed_images"):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "bmp", "webp")):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    if w <= MIN_SIZE and h <= MIN_SIZE:
                        processed = enlarge_to_min_size(img)
                        processed.save(img_path)
                        print(f"[OK] Processed: {img_path} → {processed.size}")

                except Exception as e:
                    print(f"[ERROR] {img_path}: {e}")


if __name__ == "__main__":
    process_images()
