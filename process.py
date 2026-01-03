import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
from PIL import Image
import easyocr
import random
from transformers import CLIPProcessor, CLIPModel
import torch
from cfg import Config
from itertools import combinations

reader = easyocr.Reader(['ch_sim'])  # 简体中文模型

colunm_keys = ['甲骨文', '金文', '篆文', '隶书', '楷书', '行书', '草书', '标准宋体']

def show_img(imgs):
    """
    显示单张图像或图像列表。
    imgs: 单张图像 np.ndarray 或图像列表/数组
    """
    n = len(imgs)
    # 自动计算子图行列数
    cols = min(10, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4*cols, 4*rows))
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_single_img(img):
    """
    显示单张图像。
    img: 单张图像 np.ndarray，BGR 或 RGB 格式
    """
    if img is None:
        print("图像为空！")
        return

    plt.figure(figsize=(4, 4))
    # 如果是BGR格式，转换为RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img  # 灰度图直接使用

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


def get_lines(cropped):
    # -------------------------------
    # 灰度 + 高斯模糊 + 边缘检测
    # -------------------------------
    channel = cropped[:, :, 2]  # 0=B, 1=G, 2=R
    edges = cv2.Canny(channel, 20, 80)
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    # 霍夫直线检测
    lines = cv2.HoughLinesP(
        edges_dilated,
        rho=1,
        theta=np.pi / 180,
        threshold=30,  # 票数阈值，低一些以检测细线
        minLineLength=20,  # 允许短线
        maxLineGap=10  # 小间隙可合并为一条线
    )
    # 筛选水平和垂直线
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)

            # 水平线：dy/dx 很小
            if abs(dy / dx) < 0.1:
                horizontal_lines.append((x1, y1, x2, y2, length))
            # 垂直线：dx/dy 很小
            elif dy != 0 and abs(dx / dy) < 0.1:
                vertical_lines.append((x1, y1, x2, y2, length))

    # -----------------------------
    # 取需要数量的线条
    min_length_horizontal = 100  # 水平线最小长度
    min_length_vertical = 50  # 垂直线最小长度

    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = (x2 - x1) + 1e-5
            dy = (y2 - y1) + 1e-5
            length = np.sqrt(dx ** 2 + dy ** 2)

            # 水平线筛选
            if abs(dy / dx) < 0.1 and length >= min_length_horizontal:
                horizontal_lines.append((x1, y1, x2, y2, length))
            # 垂直线筛选
            elif dy != 0 and abs(dx / dy) < 0.1 and length >= min_length_vertical:
                vertical_lines.append((x1, y1, x2, y2, length))
    # img_draw = cropped.copy()
    # for x1, y1, x2, y2, _ in horizontal_lines:
    #     cv2.line(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色水平线
    # for x1, y1, x2, y2, _ in vertical_lines:
    #     cv2.line(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色垂直线
    #
    # # 显示图像
    # import matplotlib.pyplot as plt
    # plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    # exit(111)
    # print(len(horizontal_lines), len(vertical_lines))
    return horizontal_lines, vertical_lines

def merge_lines(lines, orientation='horizontal', gap=5):
    """
    lines: list of (x1, y1, x2, y2, length)
    orientation: 'horizontal' 或 'vertical'
    gap: 相邻线段最大允许间隙
    """
    if not lines:
        return []
    # 按主要方向坐标排序
    if orientation == 'horizontal':
        lines = sorted(lines, key=lambda l: l[1])  # 按 y 排序
    else:
        lines = sorted(lines, key=lambda l: l[0])  # 按 x 排序
    merged = []
    current = lines[0]
    for line in lines[1:]:
        if orientation == 'horizontal':
            # y 坐标接近且线段重叠或相邻
            if abs(line[1] - current[1]) <= gap and not (line[0] > current[2] + gap or line[2] < current[0] - gap):
                x1_new = min(current[0], line[0])
                x2_new = max(current[2], line[2])
                y_new = int((current[1] + line[1]) / 2)
                length_new = x2_new - x1_new
                current = (x1_new, y_new, x2_new, y_new, length_new)
            else:
                merged.append(current)
                current = line
        else:  # vertical
            if abs(line[0] - current[0]) <= gap and not (line[1] > current[3] + gap or line[3] < current[1] - gap):
                y1_new = min(current[1], line[1])
                y2_new = max(current[3], line[3])
                x_new = int((current[0] + line[0]) / 2)
                length_new = y2_new - y1_new
                current = (x_new, y1_new, x_new, y2_new, length_new)
            else:
                merged.append(current)
                current = line
    merged.append(current)
    return merged


def split_by_lines(cropped, horizontal_lines, vertical_lines, ratio_range=(0.1, 10)):
    """
    将图像按水平和垂直线切分，并按行返回块列表

    ratio_range: 长宽比阈值 (min_ratio, max_ratio)
    """
    min_ratio, max_ratio = ratio_range

    # 水平线 y 坐标列表
    h_lines_y = sorted([(y1 + y2) // 2 for x1, y1, x2, y2, _ in horizontal_lines])
    # 垂直线 x 坐标列表
    v_lines_x = sorted([(x1 + x2) // 2 for x1, y1, x2, y2, _ in vertical_lines])

    all_rows = []
    for i in range(len(h_lines_y) - 1):
        y_top = h_lines_y[i]
        y_bottom = h_lines_y[i + 1]
        row_blocks = []
        for j in range(len(v_lines_x) - 1):
            x_left = v_lines_x[j]
            x_right = v_lines_x[j + 1]
            # 裁剪出小块
            block = cropped[y_top:y_bottom, x_left:x_right]
            h, w = block.shape[:2]
            ratio = w / (h + 1e-6)
            if min_ratio <= ratio <= max_ratio:
                row_blocks.append(block)
        if row_blocks:  # 只保留非空行
            all_rows.append(row_blocks[:7])
    # assert len(all_rows[0]) == len(all_rows[1])
    return all_rows[:2]    # 只返回前两行有用的信息


def extract_true_char_region(block):
    max_ch = np.max(block, axis=2).astype(np.uint8)
    _, binary = cv2.threshold(max_ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return block[y1:y2+1, x1:x2+1]

from datetime import datetime
def split_characters_by_contour(img, min_area=50, dilate_kernel=(10,10)):
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化（黑底白字）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 膨胀，让断开的笔画连起来
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    # 反色
    thresh_inv = 255 - thresh
    # 膨胀
    dilated = cv2.dilate(thresh_inv, kernel, iterations=1)
    # 找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    texts = []
    # 遍历轮廓
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            char_img = img[y:y + h, x:x + w]  # 原图裁剪
            characters.append(char_img)
            texts.append("")  # 文本占位为空串
    # 按 x 坐标排序（从左到右）
    characters = [c for _, c in
                  sorted(zip([cv2.boundingRect(c)[0] for c in contours if cv2.contourArea(c) >= min_area], characters))]
    return characters, texts

def save_images_by_font(data_dict, font_name, root_dir="processed_images"):
    """
    将字典中的图像保存到指定文件夹中。

    参数:
        data_dict: {字型: [图像数组, ...]} 的字典
        root_dir: 保存根目录
    """
    font_dir = os.path.join(root_dir, font_name).encode('utf-8').decode('utf-8')
    os.makedirs(font_dir, exist_ok=True)
    for font_time in data_dict.keys():
        img_list = data_dict[font_time]
        # 如果列表中只有一个图片，文件名直接用字型
        for idx, img in enumerate(img_list):
            filename = f"{font_time}_{idx}.jpg".encode('utf-8').decode('utf-8')
            path = os.path.join(font_dir, filename)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.save(path)

    print(f"所有图像已保存到 {font_name} 文件夹。")


# read_imgs()


def delete_loss(root='processed_images2'):
    # 识别出来所有的“缺”并将其从中删除
    dirs = os.listdir(root)
    dirs = [each for each in dirs if '缺' != each]
    for dir in tqdm(dirs):
        print('processing {}'.format(dir))
        images = os.listdir(f'{root}/{dir}/')
        for image in images:
            img_path = f'{root}/{dir}/{image}'
            with open(img_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                try:
                    result = reader.readtext(img, detail=0)[0]
                except:
                    continue
            if result == '缺':
                os.remove(img_path)

def black(root='processed_images2'):
    threshold = 128  # 二值化阈值，可调整
    dirs = os.listdir(root)
    for dir in tqdm(dirs):
        subdir = os.path.join(root, dir)
        if not os.path.isdir(subdir):
            continue
        print(f'processing {dir}')
        images = os.listdir(subdir)
        for image in images:
            image_path = os.path.join(subdir, image)
            # with open(image_path, 'rb') as f:
            #     data = f.read()
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            # Otsu二值化 —— 自动找最佳阈值
            # 假设 img 是你的彩色图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用 Otsu 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 二值化：像素 >= 阈值 → 255，否则 → 0
            # Step 2: 反色 --> 白底 黑字
            binary = 255 - binary
            # Step 3: 保证只有 0 和 255（强制二值）
            binary[binary < 128] = 0
            binary[binary >= 128] = 255
            img_pil = Image.fromarray(binary)
            img_pil.save(image_path)
        # exit(111)

def del_low():
    root_dir = 'processed_images'
    dirs = os.listdir(root_dir)
    for dir in tqdm(dirs):
        dir_path = os.path.join(root_dir, dir)
        if not os.path.isdir(dir_path):
            continue
        image_names = os.listdir(dir_path)
        for img_name in image_names:
            img_path = os.path.join(dir_path, img_name)
            # 跳过非图片格式
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                # 图片损坏，则删除
                os.remove(img_path)
                continue
            # 判定像素低于要求
            if w < 10 or h < 10:
                os.remove(img_path)
                print(img_path)


def is_touching_edge(binary_img):
    # 假设黑色前景为 0
    h, w = binary_img.shape
    top = binary_img[0, :]
    bottom = binary_img[h - 1, :]
    left = binary_img[:, 0]
    right = binary_img[:, w - 1]

    if np.any(top == 0) or np.any(bottom == 0) or np.any(left == 0) or np.any(right == 0):
        return True
    return False


def seek_Incomplete():
    dirs = os.listdir('processed_images')
    incomps = []
    for dir in tqdm(dirs):
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_path in image_paths:
            data = np.fromfile(image_path, dtype=np.uint8)
            bw = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if is_touching_edge(bw):
                if dir not in incomps:
                    incomps.append(dir)
    return incomps


error_dirs = ['买', '丝', '堕', '妫', '牟', '怂', '义', '师', '时', '艺', '对', '币', '为', '卫', '忆', '庄', '净', '黾', '浄', '触', '侠', '挟', '属', '毕', '亚', '继', '酿', '种', '树', '斋', '迟', '旧', '静', '乌', '叠', '层', '显', '积', '担', '壮', '车', '歼', '惊', '风', '兴', '惬', '窍', '徕', '滞', '惧', '聂', '累', '灶', '凿', '弃', '乱', '进', '洁', '韦', '帮', '墙', '导', '礼', '屿', '聪', '终', '帧', '寿', '边', '实', '单', '辞', '凯', '粮', '法', '怜', '泾', '职', '涤', '陆', '东', '叹', '妆', '烧', '劲', '发', '迩', '肃', '图', '状', '习', '这', '齑', '卢', '长', '窜', '见', '画', '留', '宾', '阋', '间', '态', '痴', '岩', '举', '枣', '彻', '凤', '猎', '罗', '厨', '荣', '韵', '泻', '会', '娅', '党', '尔', '妪', '纤', '馋', '丛', '来', '仓', '过', '应', '响', '轰', '乐', '办', '认', '陕', '炉', '亿', '村', '参', '让', '击', '补', '娄', '选', '鲁', '夹', '离', '当', '马', '养', '齐', '缪', '贝', '优', '关', '执', '肤', '鱼', '宝', '声', '爱', '爷', '临', '鸡', '学', '卖', '归', '茧', '业', '门', '惩', '徵', '团', '偿', '牵', '烬', '渊', '节', '达', '跃', '戏', '尝', '装', '书', '页', '钻', '宪', '毙', '戋', '蜡', '栏', '馀', '盖', '箧', '罢', '庐', '随', '两', '龟', '乔', '馔', '头', '严', '带', '寻', '蜗', '伪', '柜', '庆', '区', '蚕', '仑', '艰', '丧', '酝', '谗', '驴', '协', '邓', '争', '饫', '弥', '昼', '孙', '尧', '蛎', '疗', '岁', '舰', '卤', '袄', '称', '专', '刍', '盐', '尽', '获', '伤', '岳', '饬', '殇', '鸟', '郑', '识', '讲', '庙', '痒', '梦', '欢', '动', '郸']

def read_imgs():
    error_datas, right_datas = [], {}
    # dirs = os.listdir('images')
    dirs = error_dirs
    for dir in tqdm(dirs, desc='Processing'):
        path = 'images/' + dir+'/1.jpg'
        with open(path, 'rb') as f:
            # try:
                img_array = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                # 首先裁剪一下去除多余的边框
                h = img.shape[0]
                top_ratio = 0.01
                bottom_ratio = 0.01
                top = int(h * top_ratio)
                bottom = int(h * (1 - bottom_ratio))
                cropped = img[top:bottom, :]
                horizontal_lines, vertical_lines = get_lines(cropped)
                # 进一步合并水平和垂直线，并按照线对图像进行拆分
                horizontal_lines = merge_lines(horizontal_lines, orientation='horizontal')
                vertical_lines = merge_lines(vertical_lines, orientation='vertical')
                all_rows = split_by_lines(cropped, horizontal_lines, vertical_lines)   # 拆分出来的行
                Chars = all_rows[1]   # 第一行存放的是相应的字
                data_dict = {key: [] for key in colunm_keys}
                # 判断一个图里是一个字还是两个字，如果一个字进行字形提取然后resizing成一个固定的大小 （50*50）
                if len(Chars) == 7:
                    for char, key in zip(Chars, colunm_keys):
                        char_list, texts = split_characters_by_contour(char)
                        # print(char_list, texts)
                        # char_list = [char_list[i] for i in range(len(char_list)) if '缺' not in texts[i]]
                        data_dict[key] = char_list
                    save_images_by_font(data_dict, font_name=dir, root_dir='processed_images2')
                else:
                    error_datas.append(dir)     # 长度不足的可能是有问题
            # except:
            #     error_datas.append(dir)
            #     continue

    print('error datas:', len(error_datas))
    with open('error.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(error_datas)+'\n')


# read_imgs()
# delete_loss()
# black()



def build_task1_1(cfg):
    query = '请根据上图判断此文字所属的书体时代类别。可选类别包括：甲骨文、金文、篆文、隶书、楷书。输出仅且必须为五个类别之一的单个汉字词，绝对禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。若不能确定，请只输出：无法判断。'
    dirs = os.listdir('processed_images')
    class_dict = {
        '甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]
    }
    for dir in tqdm(dirs):
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_name, image_path in zip(image_names, image_paths):
            class_ = image_name.split('_')[0]
            if class_ in class_dict.keys():
                class_dict[class_].append(image_path)
    # 随机采样 3000 个数据
    sampled_dict = {}
    sample_size = 3000
    for class_name, paths in class_dict.items():
        if len(paths) <= sample_size:
            sampled_dict[class_name] = paths.copy()  # 不够就全部保留
        else:
            sampled_dict[class_name] = random.sample(paths, sample_size)
    # 输出每个类别采样后的数量
    for class_name, paths in sampled_dict.items():
        print(class_name, len(paths))
    # 假设格式为 [{"image_path": "...", "label": "..."}]
    data_for_model = []
    for class_name, paths in sampled_dict.items():
        for path in paths:
            data_for_model.append({
                "text": query,
                "image": path,  # 或者可以改成 base64 编码
                "answer": class_name
            })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task1_1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def build_task1_2(cfg):
    query = '请根据下列两张图判断它们是否属于同一书体时代。可选答案仅包括：是、否。绝对禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。'
    dirs = os.listdir('processed_images')
    image_pairs, answers = [], []
    dis_dict = {}
    for dir in tqdm(dirs):
        dis_dict[dir] = {
        '甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]
        }
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_name, image_path in zip(image_names, image_paths):
            class_ = image_name.split('_')[0]
            if class_ in dis_dict[dir].keys():
                dis_dict[dir][class_].append(image_path)

    for dir in tqdm(dis_dict.keys()):
        class_dict = dis_dict[dir]
        # 先找出所有 image_paths 非空的类（用于 negative pairs）
        non_empty_classes = {cls: paths for cls, paths in class_dict.items() if len(paths) > 0}
        for class_name, image_paths in class_dict.items():
            if len(image_paths) > 1:
                # ---- 构建 “是” pair（同类两张随机抽取） ----
                img1, img2 = random.sample(image_paths, 2)
                image_pairs.append([img1, img2])
                answers.append('是')
                # ---- 构建 “否” pair（本类 + 其他类） ----
                # 从其他非空类中选一个不同于当前类的
                candidate_classes = [cls for cls in non_empty_classes.keys() if cls != class_name]
                if len(candidate_classes) == 0:
                    # 没有其它非空类，无法构建否样本，跳过
                    continue
                other_class = random.choice(candidate_classes)
                other_img = random.choice(non_empty_classes[other_class])
                image_pairs.append((random.choice(image_paths), other_img))
                answers.append('否')
    print(len(image_pairs), len(answers))
    data_for_model = []
    for image_pair, answer in zip(image_pairs, answers):
        data_for_model.append({
            "text": query,
            "image": image_pair,  # 或者可以改成 base64 编码
            "answer": answer
        })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task1_2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def build_task1_3(cfg):
    STANDARD_KEYS = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    query = '请根据下列两张图判断它们是否属于同一书体时代。可选答案仅包括：是、否。绝对禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。'
    dirs = os.listdir('processed_images')
    image_pairs, answers = [], []
    dis_dict = {}
    for dir in tqdm(dirs):
        dis_dict[dir] = {
        '甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]
        }
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_name, image_path in zip(image_names, image_paths):
            class_ = image_name.split('_')[0]
            if class_ in dis_dict[dir].keys():
                dis_dict[dir][class_].append(image_path)

    normalized = {k: [] for k in STANDARD_KEYS}
    for dir_name, image_paths in dis_dict.items():
        for fname, images in image_paths.items():
            normalized[fname].extend(images)
    for fname, images in normalized.items():
        if len(images) < 2:
            continue
        for _ in range(1000):
            img1, img2 = random.sample(images, 2)  # 随机选择两张，不重复
            image_pairs.append([img1, img2])
            answers.append('是')
    i = 0
    while i < 5000:
        rand_f = random.sample(STANDARD_KEYS, 2)
        image_1 = random.choice(normalized[rand_f[0]])
        image_2 = random.choice(normalized[rand_f[1]])
        if [image_1, image_2] in image_pairs:
            continue
        else:
            image_pairs.append([image_1, image_2])
            answers.append('否')
            i += 1
    data_for_model = []
    for image_pair, answer in zip(image_pairs, answers):
        data_for_model.append({
            "text": query,
            "image": image_pair,  # 或者可以改成 base64 编码
            "answer": answer
        })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task1_3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)



def build_task1_4(cfg):
    STANDARD_KEYS = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    query = '请根据下列多张图判断哪一张不属于其他图片的书体时代。四个选项分别为 A<image>；B<image>；C<image>；D<image>，其中只有一个是正确答案。输出仅且必须为 A、B、C 或 D 中的单个字母，绝对禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。若无法判断，请只输出：无法判断。'
    dirs = os.listdir('processed_images')
    image_pairs, answers = [], []
    dis_dict = {}
    for dir in tqdm(dirs):
        dis_dict[dir] = {
        '甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]
        }
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_name, image_path in zip(image_names, image_paths):
            class_ = image_name.split('_')[0]
            if class_ in dis_dict[dir].keys():
                dis_dict[dir][class_].append(image_path)

    normalized = {k: [] for k in STANDARD_KEYS}
    for dir_name, image_paths in dis_dict.items():
        for fname, images in image_paths.items():
            normalized[fname].extend(images)
    # 随机从一个书体中选择三个图，并从剩余里选一个图
    i = 0
    while i < 10000:
        for fname, images in normalized.items():
            rand_f = random.choice(STANDARD_KEYS)
            rand_imgs = random.sample(normalized[rand_f], 3)
            answer_f = random.choice(list(set(STANDARD_KEYS) - set([rand_f])))
            answer_img = random.sample(normalized[answer_f], 1)
            image_pairs.append(rand_imgs + answer_img)
            i += 1
    labeled_pairs = []  # 存储 [(A,B,C,D), correct_label]
    for imgs in image_pairs:
        # imgs = [img1, img2, img3, answer_img]
        shuffled = imgs.copy()
        random.shuffle(shuffled)
        # 找到 answer_img（原最后一张）打乱后的位置
        answer_img = imgs[-1]
        correct_idx = shuffled.index(answer_img)
        correct_label = ['A', 'B', 'C', 'D'][correct_idx]
        labeled_pairs.append([shuffled, correct_label])

    data_for_model = []
    for image_pair, answer in labeled_pairs:
        data_for_model.append({
            "text": query,
            "image": image_pair,
            "answer": answer
        })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task1_4.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def build_task2_1(cfg):
    # 给出文字识别字体
    query = '下面给出一张古文字图像，请根据图像内容识别文字，并直接输出对应的现代汉字。禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。若不能确定，请只输出：无法判断。'
    dirs = os.listdir('processed_images')
    scrip_dict = {}
    class_dict = {
        '甲骨文': [], '金文': [], '篆文': [], '隶书': [], '楷书': []
    }
    for dir in tqdm(dirs):
        image_names = os.listdir(f'processed_images/{dir}/')
        image_times = [name.split('_')[0] for name in image_names]
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        scrip_dict[dir] = [image_paths, image_times]
        # 假设格式为 [{"image_path": "...", "label": "..."}]
    data_for_model = []
    for class_name, values in scrip_dict.items():
        paths, times = values[0], values[1]
        for path, time in zip(paths, times):
            if time in class_dict.keys():   # 只考虑5个time的
                data_for_model.append({
                    "text": query,
                    "image": path,  # 或者可以改成 base64 编码
                    "answer": class_name,
                    "time": time
                })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task2_1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)



def build_task2_2(cfg):
    query = '请根据下列两张图判断它们是否属于同一个字。可选答案仅包括：是、否。绝对禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。'
    dirs = os.listdir('processed_images')
    image_pairs, answers = [], []
    dis_dict, time_dict = {}, {'甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]}
    for dir in tqdm(dirs):
        dis_dict[dir] = {
        '甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]
        }
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_name, image_path in zip(image_names, image_paths):
            class_ = image_name.split('_')[0]
            if class_ in dis_dict[dir].keys():
                dis_dict[dir][class_].append(image_path)
                time_dict[class_].append(image_path)
    for dir in tqdm(dis_dict.keys()):
        class_dict = dis_dict[dir]
        for class_name, image_paths in class_dict.items():
            if len(image_paths) > 1:
                # ---- 构建 “是” pair（同类两张随机抽取） ----
                img1, img2 = random.sample(image_paths, 2)
                image_pairs.append([img1, img2])
                answers.append('是')
    llll, i = len(image_pairs), 0
    while i < llll:
        rand_class = random.choice(list(time_dict.keys()))
        pair = random.sample(time_dict[rand_class], 2)
        image_pairs.append(pair)
        answers.append('否')
        i += 1
    data_for_model = []
    for image_pair, answer in zip(image_pairs, answers):
        data_for_model.append({
            "text": query,
            "image": image_pair,  # 或者可以改成 base64 编码
            "answer": answer
        })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task2_2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)




def build_task2_3(cfg):
    query = '请根据下列两张图判断它们是否属于同一个字。可选答案仅包括：是、否。绝对禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。'
    dirs = os.listdir('processed_images')
    image_pairs, answers = [], []
    dis_dict, time_dict = {}, {'甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]}
    for dir in tqdm(dirs):
        dis_dict[dir] = {
        '甲骨文':[], '金文':[], '篆文':[], '隶书':[], '楷书':[]
        }
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_name, image_path in zip(image_names, image_paths):
            class_ = image_name.split('_')[0]
            if class_ in dis_dict[dir].keys():
                dis_dict[dir][class_].append(image_path)
                time_dict[class_].append(image_path)
    rand_chars = random.sample(list(dis_dict.keys()), 5000)
    for char in rand_chars:
        line = {k: v for k, v in dis_dict[char].items() if v}  # v 非空列表才保留
        if len(line) > 1:
            rand_time = random.sample(list(line.keys()), 2)
            pair = [line[rand_time[0]][0], line[rand_time[1]][0]]
            image_pairs.append(pair)
            answers.append('是')
    llll, i = len(image_pairs), 0
    while i < llll:
        # 选取不同时期的不同字
        rand_chars = random.sample(list(dis_dict.keys()), 2)
        line1 = {k: v for k, v in dis_dict[rand_chars[0]].items() if v}  # v 非空列表才保留
        line2 = {k: v for k, v in dis_dict[rand_chars[1]].items() if v}  # v 非空列表才保留
        rand_time1 = random.choice(list(line1.keys()))
        rand_time2 = random.choice(list(line2.keys()))
        keys1 = set(line1.keys())
        keys2 = set(line2.keys())
        # 保证 line2 中有可选 key
        available_keys2 = list(keys2 - {rand_time1})
        if available_keys2:
            rand_time2 = random.choice(available_keys2)
        else:
            # 如果 line2 里只有 rand_time1 那个 key，只能重复
            rand_time2 = rand_time1
        pair = [line1[rand_time1][0], line2[rand_time2][0]]
        image_pairs.append(pair)
        answers.append('否')
        i += 1

    data_for_model = []
    for image_pair, answer in zip(image_pairs, answers):
        data_for_model.append({
            "text": query,
            "image": image_pair,  # 或者可以改成 base64 编码
            "answer": answer
        })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task2_3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def build_task2_4(cfg):
    STANDARD_KEYS = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    query = '请根据下列多张图判断哪一张对应的字与其他字不同。四个选项分别为 A<image>；B<image>；C<image>；D<image>，其中只有一个是正确答案。输出仅且必须为 A、B、C 或 D 中的单个字母，绝对禁止任何形式的解释、理由、分析、过程描述、列表、示例、序号、标点或其它文字。若无法判断，请只输出：无法判断。'
    dirs = os.listdir('processed_images')
    image_pairs, answers = [], []
    dis_dict = {}
    for dir in tqdm(dirs):
        dis_dict[dir] = []
        image_names = os.listdir(f'processed_images/{dir}/')
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_name, image_path in zip(image_names, image_paths):
            class_ = image_name.split('_')[0]
            if class_ in STANDARD_KEYS:
                dis_dict[dir].append(image_path)
    dis_dict2 = {k: v for k, v in dis_dict.items() if len(v) >= 3}
    for dir in tqdm(dis_dict2.keys()):
        rand_imgs = random.sample(list(dis_dict2[dir]), 3)
        remain_dirs = [each for each in dirs if each != dir]
        answer_img = dis_dict[random.choice(remain_dirs)]
        answer_img = random.choice(answer_img)
        image_pairs.append(rand_imgs + [answer_img])

    labeled_pairs = []  # 存储 [(A,B,C,D), correct_label]
    for imgs in image_pairs:
        # imgs = [img1, img2, img3, answer_img]
        shuffled = imgs.copy()
        random.shuffle(shuffled)
        # 找到 answer_img（原最后一张）打乱后的位置
        answer_img = imgs[-1]
        correct_idx = shuffled.index(answer_img)
        correct_label = ['A', 'B', 'C', 'D'][correct_idx]
        labeled_pairs.append([shuffled, correct_label])

    data_for_model = []
    for image_pair, answer in labeled_pairs:
        data_for_model.append({
            "text": query,
            "image": image_pair,
            "answer": answer
        })
    # 4. 保存为 JSON 文件
    output_file = "tasks/task2_4.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def build_task3_1(cfg):
    time_order = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    data_for_model = []
    # 给出文字的演化路径，再让其识别对应的文字
    query = '下方依次给出同一汉字在不同历史时期的字形。请根据其演化关系，直接给出该字对应的现代规范汉字。禁止解释、分析或输出除单个现代汉字以外的内容。\n'
    # 下面给出不同文字的演化路径
    dirs = os.listdir('processed_images')
    for dir in tqdm(dirs):
        image_names = os.listdir(f'processed_images/{dir}/')
        image_times = [name.split('_')[0] for name in image_names]
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        # 封装成字典
        time_dict = {}
        for time, path in zip(image_times, image_paths):
            if time not in time_dict:
                time_dict[time] = [path]
            else:
                time_dict[time].append(path)
        # 如果某个 time 重复，随机选一个
        final_dict = {time: random.choice(paths) for time, paths in time_dict.items()}
        final_dict_ordered = {time: final_dict.get(time, None) for time in time_order}
        tuple_list = [(time, path) for time, path in final_dict_ordered.items() if path is not None]
        if len(tuple_list) > 1:    # 只包含一个字的就省略了
            prompt = query
            for i, (time, path) in enumerate(tuple_list, 1):
                prompt += f"{time}: <image>, "
            prompt += '现代汉字：'
            # JSON 数据格式
            data_for_model.append({
                "text": prompt,
                "image": [path for _, path in tuple_list],  # 可直接用 base64 替换
                "answer": dir  # 测试阶段无标签，可填 None
            })
    # 保存为 JSON 文件
    output_file = "tasks/task3_1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def build_task3_2(cfg):
    # 给定打乱的字和时期，进行排序
    time_order = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    data_for_model = []
    # 给出文字的演化路径，再让其识别对应的文字
    query = '''严格按照图像在输入中的顺序逐一判断它们分别属于哪个书体时期。输出格式必须是：图像编号-书体名称，例如：\n1-甲骨文；2-篆文；3-隶书；4-金文；5-楷书\n禁止模仿上述示例的顺序输出，严格保持与图像输入顺序一致，禁止输出解释。'''
    # 下面给出不同文字的演化路径
    dirs = os.listdir('processed_images')
    for dir in tqdm(dirs):
        image_names = os.listdir(f'processed_images/{dir}/')
        image_times = [name.split('_')[0] for name in image_names]
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        # 封装成字典
        time_dict = {}
        for time, path in zip(image_times, image_paths):
            if time not in time_dict:
                time_dict[time] = [path]
            else:
                time_dict[time].append(path)
        # 如果某个 time 重复，随机选一个
        final_dict = {time: random.choice(paths) for time, paths in time_dict.items()}
        final_dict_ordered = {time: final_dict.get(time, None) for time in time_order}
        tuple_list = [(time, path) for time, path in final_dict_ordered.items() if path is not None]
        if len(tuple_list) > 1:  # 只包含一个字的就省略了
            prompt = query
            random.shuffle(tuple_list)
            for i, (time, path) in enumerate(tuple_list, 1):
                prompt += f"<image>, "
            prompt += '对应时期：'
            # JSON 数据格式
            data_for_model.append({
                "text": prompt,
                "image": [path for _, path in tuple_list],  # 可直接用 base64 替换
                "answer": [time for time, path in tuple_list]  # 测试阶段无标签，可填 None
            })
    # 保存为 JSON 文件
    output_file = "tasks/task3_2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)

# build_task()

def get_category(path, time_order):
    """从图片路径中解析字体类别，可按你的命名规则调整。"""
    # 假设路径或文件名中包含类别关键字
    for cat in time_order:
        if cat in path:
            return cat
    return None

def pre_build_task3_3(cfg):
    all_images = []
    time_order = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # 首先加载CLIP构建字形相似的矩阵，然后给出3个候选+一个标准答案，组成任务4的选项，最后给出预测结果（ABCD）
    model = CLIPModel.from_pretrained(cfg.LLM_path + cfg.encoder).to(device)
    processor = CLIPProcessor.from_pretrained(cfg.LLM_path + cfg.encoder)
    model.eval()
    dirs = os.listdir('processed_images')
    for dir in tqdm(dirs):
        image_names = os.listdir(f'processed_images/{dir}/')
        image_times = [name.split('_')[0] for name in image_names]
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_path, image_time in zip(image_paths, image_times):
            if image_time in time_order:
                all_images.append(image_path)
    embeddings = []
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), batch_size), desc="Computing CLIP embeddings"):
            batch_paths = all_images[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)
            outputs = model.get_image_features(**inputs)
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)  # ℓ2 normalize
            embeddings.append(outputs.cpu())
    embeddings = torch.cat(embeddings, dim=0)  # [N, D]
    cos_sim_matrix = embeddings @ embeddings.T  # [N, N]
    cos_sim_no_self = cos_sim_matrix.clone()
    cos_sim_no_self.fill_diagonal_(-float("inf"))
    # 对每一行排序（descending=True）
    values, indices = torch.sort(cos_sim_no_self, descending=True, dim=1)
    # 取每行前 50 个
    top50_indices = indices[:, :50]  # shape: [N, 50]
    print(top50_indices.shape)
    # 保存
    # 保存 top50 索引矩阵
    np.save("sims/clip_top50.npy", top50_indices.cpu())
    # 保存 all_images 列表
    with open("sims/all_images.json", "w", encoding="utf-8") as f:
        json.dump(all_images, f, ensure_ascii=False, indent=2)


def build_task3_3(cfg):
    pre_build_task3_3(cfg)
    data_for_model = []
    with open("sims/all_images.json", "r", encoding="utf-8") as f:
        all_images = json.load(f)
    clip_top50 = np.load("sims/clip_top50.npy")
    time_order = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    image_dirs = os.listdir('processed_images')
    dir2image = {each: [] for each in image_dirs}
    for dir in tqdm(image_dirs):
        image_names = os.listdir(f'processed_images/{dir}/')
        image_times = [name.split('_')[0] for name in image_names]
        image_paths = [f'processed_images/{dir}/{path}' for path in image_names]
        for image_path, image_time in zip(image_paths, image_times):
            if image_time in time_order:
                dir2image[dir].append(image_path)
    dir2image = {
        k: v for k, v in dir2image.items()
        if len(v) >= 3     # 过滤掉太短的因为需要进行推演
    }
    new_dir2image = {}
    for key, paths in dir2image.items():
        # 将图片按类别聚合
        cat2paths = {cat: [] for cat in time_order}
        for p in paths:
            cat = get_category(p, time_order)
            if cat is not None:
                cat2paths[cat].append(p)
        # 对所有类别：如果该类别图片>1，则随机选择一个
        for cat in time_order:
            if len(cat2paths[cat]) > 1:
                cat2paths[cat] = [random.choice(cat2paths[cat])]
        # 按固定顺序展开
        sorted_paths = []
        for cat in time_order:
            sorted_paths.extend(cat2paths[cat])
        new_dir2image[key] = sorted_paths
    # 从中随机选出一个然后用剩余的预测
    for key in new_dir2image.keys():
        missing_image = random.choice(list(new_dir2image[key]))    # 标准答案
        remain_images = [img for img in new_dir2image[key] if img != missing_image]
        remain_categories = [get_category(each, time_order) for each in remain_images]
        missing_cat = get_category(missing_image, time_order)
        # image_place_holders = ['<image>'] * len(remain_images)
        # 接下来处理这个图像本身再找3个与其最相似的
        miss_image_id = all_images.index(missing_image)
        # sim = clip_top50[miss_image_id]
        # assert miss_image_id not in sim
        # sim = sim[:3]    # 只选top3
        sim = [each for each in range(clip_top50.shape[0]) if each != miss_image_id]
        sim = random.sample(sim, 3)   # 随机选3个
        # 构建候选的选项图片集合
        option_images = [all_images[i] for i in sim] + [missing_image]
        random.shuffle(option_images)
        labels = ['A', 'B', 'C', 'D']
        missing_label_idx = option_images.index(missing_image)
        remain_str = "; ".join(
            f"{cat}: <image>" for cat in remain_categories
        ) + '\n'
        option_str = "选项：" + "; ".join(f"{lbl}: <image>" for lbl in ["A", "B", "C", "D"]) + "\n 答案："
        prompt = f"下方展示的是同一汉字在不同时期的部分书体写法：{remain_str}其中[{missing_cat}]缺失。请根据汉字形体的历史演化规律，判断缺失的图像最有可能对应选项 A、B、C、D 中的哪一个。禁止输出解释，只需给出一个选项字母。\n{option_str}"
        images = remain_images + option_images
        answer = labels[missing_label_idx]
        data_for_model.append({
            "text": prompt,
            "image": images,  # 可直接用 base64 替换
            "answer": answer  # 测试阶段无标签，可填 None
        })
        # 保存为 JSON 文件
    output_file = "tasks/task3_3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def split_tasks():
    # 将task下的所有json按照随机9：1划分成训练和测试集
    for json_path in os.listdir("tasks/"):
        with open("tasks/"+json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 条目级随机打乱
            random.shuffle(data)
            train_data = []
            test_data = []
            split_idx = int(len(data) * 0.9)
            train_data.extend(data[:split_idx])
            test_data.extend(data[split_idx:])

            # 保存为统一的数据集文件
            with open("split_tasks/{}_train.json".format(json_path.split(".")[0]), "w", encoding="utf-8") as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)

            with open("split_tasks/{}_test.json".format(json_path.split(".")[0]), "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    cfg = Config()
    split_tasks()
    # build_task1_1(cfg)
    # build_task1_2(cfg)
    # build_task1_3(cfg)
    # build_task1_4(cfg)
    # build_task2_1(cfg)
    # build_task2_2(cfg)
    # build_task2_3(cfg)
    # build_task2_4(cfg)
    # build_task3_1(cfg)
    # build_task3_2(cfg)
    # build_task3_3(cfg)
