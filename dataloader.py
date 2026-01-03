# 根据不同的任务加载、评估VLM
import json
from email import message

import numpy as np
from datasets import load_dataset
import os
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
from sympy.codegen.ast import continue_

try:
    from transformers import (AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoTokenizer,
                              Qwen3VLForConditionalGeneration, BitsAndBytesConfig, LlavaConfig, LlavaForConditionalGeneration,
                              AutoModel, Qwen2_5_VLForConditionalGeneration)
except:
    from transformers import (AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq,
                              AutoTokenizer, BitsAndBytesConfig, LlavaConfig,
                              LlavaForConditionalGeneration,
                              AutoModel)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from cfg import Config
from tqdm import tqdm
import torch.nn.functional as F
import sys
from io import BytesIO
import re
from transformers import BitsAndBytesConfig
import types
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
except:
    print('vllm error')
from torch import nn
import transformers


def load_VLM(cfg, model_path=''):
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 开启 8-bit 权重量化
        llm_int8_threshold=6.0,  # 默认阈值（可调）
        llm_int8_has_fp16_weight=False,  # 是否保留 FP16 权重（一般 False）
        llm_int8_enable_fp32_cpu_offload=True  # 避免不支持的 GPU kernel
    )
    if model_path == '':   # 空的时候指定path
        model_path = cfg.LLM_path+cfg.VLM
    if 'qwen3-vl-30b' in cfg.VLM.lower():
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    elif 'qwen3' in cfg.VLM.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, device_map="auto",
            dtype=torch.bfloat16,
            torch_dtype=torch.float16  # 或 "auto" 根据需要
        )
        processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
        model.eval()

    elif 'qwen2.5' in cfg.VLM.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, device_map="auto",
            dtype=torch.bfloat16,
            torch_dtype=torch.float16  # 或 "auto" 根据需要
        )
        processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
        model.eval()

    elif 'tonggu' in cfg.VLM.lower():
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        # ===== 核心修补（必须是 del，不是赋值 None）=====
        if hasattr(config, "rope_type"):
            delattr(config, "rope_type")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto",
                                                     trust_remote_code=True)

    elif 'glm' in cfg.VLM.lower():
        from transformers import Glm4vForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    elif 'intern' in cfg.VLM.lower():
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            # torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
        # 选择视觉塔的前 4 层输出
        # model.config.vision_select_layer = [0, 1, 2, 3]
        model.eval()

    elif 'deepseekocr' in cfg.VLM.lower():
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = LLM(
            model=model_path,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            cpu_offload_gb=0,  # CPU offload 空间（GB）
            gpu_memory_utilization=0.6,  # 每张 GPU 占用比例
            swap_space=0,  # CPU swap 空间（GB）
            # tensor_parallel_size=2,  # 使用两张 GPU
            logits_processors=[NGramPerReqLogitsProcessor],
            dtype="float16",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        # processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        #                                   use_safetensors=True)
        # model = model.to(device).half()
        # model.eval()
    elif 'llava' in cfg.VLM.lower():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif 'minicpm' in cfg.VLM.lower():
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          attn_implementation='sdpa', device_map="auto",
                                          torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    return model, processor


def load_task_datas(cfg):
    output_file = f"split_tasks/{cfg.task}_test.json"
    with open(output_file, "r", encoding="utf-8") as f:
        raw_datas = json.load(f)
    # 将raw_datas 处理成LLMs所需的数据格式
    return raw_datas


def load_zero_shot_prompts(cfg, datas):
    prompts = []
    for i, sample in tqdm(list(enumerate(datas)), desc="Loading Prompts"):
        content = []
        answer = sample['answer']
        if type(answer) == str:
            answer = [answer]
        else:
            if type(answer) != list:
                answer = answer.tolist()
        if type(sample['image']) == str:
            images = [sample['image']]
        else:
            images = sample['image']
        text = sample['text'].split('<image>')
        if len(text) == 1:
            content.append({"type": "text", "text": text[0]})
            for image in images:
                content.append({"type": "image", "image": image})
        else:   # 多个图要穿插着来
            for txt, img in zip(text, images):
                content.append({"type": "text", "text": txt})
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": text[-1]})
        messages = [{ "role": "user", "content": content}]
        prompts.append([messages, answer])
    return prompts


def batchify(iterable, batch_size):
    """将可迭代对象分成大小为 batch_size 的批次，并一次性存入内存"""
    return [iterable[i:i + batch_size] for i in range(0, len(iterable), batch_size)]


def safe_resize(img, min_size=30):
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    new_w = max(w, min_size)
    new_h = max(h, min_size)
    # 白色背景，不破坏图像内容
    new_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    new_img.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
    return new_img

from functools import lru_cache
@lru_cache(maxsize=4096)
def load_image(path):
    return Image.open(path)

def prepare_batch_for_InternVL(cfg, messages):
    """
    messages: List[List[Dict]]
    每条消息可以包含多段文本和多张图片：
    [
        [
            {"role": "user", "content": [
                {"type": "text", "text": "描述图像"},
                {"type": "image", "image": "img1.jpg"},p
                {"type": "text", "text": "这是另一段说明"},
                {"type": "image", "image": "img2.jpg"}
            ]}
        ],
        ...
    ]
    """
    batch_texts = []
    batch_images = []
    for chat in messages:
        text_parts = []
        imgs = []
        for turn in chat:
            for item in turn["content"]:
                if item["type"] == "text":
                    text_parts.append(item["text"])
                elif item["type"] == "image":
                    img = load_image(item["image"])
                    if 'glm' in cfg.VLM.lower():
                        img = safe_resize(img)
                    imgs.append(img)

        # 拼接文本，每段文本之间换行
        text_joined = "\n".join(text_parts)
        # 对应的 prompt 中图片占位符
        # 插入与图片数量相同的 <image> 占位符
        text_with_placeholders = "<IMG_CONTEXT>" * len(imgs) + "\n" + text_joined
        batch_texts.append(text_with_placeholders)
        batch_images.append(imgs)
    return batch_texts, batch_images

def prepare_batch_for_YiVL(cfg, messages):
    """
    messages: List[List[Dict]]
    每条消息可以包含多段文本和多张图片：
    [
        [
            {"role": "user", "content": [
                {"type": "text", "text": "描述图像"},
                {"type": "image", "image": "img1.jpg"},
                {"type": "text", "text": "这是另一段说明"},
                {"type": "image", "image": "img2.jpg"}
            ]}
        ],
        ...
    ]
    """
    new_messages = []

    for dialog in messages:
        new_dialog = []

        for msg in dialog:
            if "content" not in msg:
                new_dialog.append(msg)
                continue

            collected_texts = []
            for content in msg["content"]:
                if content.get("type") == "text":
                    text = content["text"]
                    if not text.endswith("<image>"):
                        text = text + "<image>"
                    collected_texts.append(text)

            # 将所有 text 拼成单一字符串（按行或空格均可）
            merged_text = "\n".join(collected_texts)

            new_msg = {
                "role": msg.get("role", "user"),
                "content": merged_text
            }
            new_dialog.append(new_msg)

        new_messages.append(new_dialog)

    return new_messages


def prepare_batch_for_OCR(messages, processor):
    """
    messages: 形如
    [
        [
            {"role": "user", "content": [{"type":"text","text":"描述一下图片"}, {"type":"image","image":"path1.jpg"}]},
        ],
        [
            {"role": "user", "content": [{"type":"text","text":"这是什么"}, {"type":"image","image":"path2.jpg"}]},
        ]
    ]
    """
    batch = []
    for message_list in messages:
        for msg in message_list:
            text_parts = []
            images = []
            for content in msg["content"]:
                if content["type"] == "text":
                    text_parts.append(content["text"])
                elif content["type"] == "image":
                    # 打开图片并转为 RGB
                    img = Image.open(content["image"]).convert("RGB")
                    images.append(img)
            if len(images) == len(text_parts):
                prompt_text = "<image>".join(text_parts) + "<image>"
            if len(images) == len(text_parts)-1:
                prompt_text = "<image>".join(text_parts)  # 可根据需求修改 prompt 前缀
                # 为每张图片生成一个单独的输入项
            batch.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": images}
            })

    return batch


def prepare_batch_for_MiniCPM(cfg, messages):
    msgs = []
    for message_list in messages:
        for msg in message_list:
            text_parts = []
            images = []
            for content in msg["content"]:
                if content["type"] == "text":
                    text_parts.append(content["text"])
                elif content["type"] == "image":
                    # 打开图片并转为 RGB
                    img = Image.open(content["image"]).convert("RGB")
                    images.append(img)
            msgs.append([{
                'role': 'user',
                'content': images + [' '.join(text_parts)],
            }])

    return msgs


if __name__ == "__main__":
    cfg = Config()
    datas = load_task_datas(cfg)
    load_zero_shot_prompts(cfg, datas)