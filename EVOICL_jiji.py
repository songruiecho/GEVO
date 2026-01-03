from dataloader import load_VLM, load_zero_shot_prompts, batchify, load_task_datas, prepare_batch_for_InternVL, prepare_batch_for_OCR, prepare_batch_for_MiniCPM
from cfg import Config
import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from utils import *

def VICL(cfg, batches, model, processor):
    results = []
    max_new_tokens = 20
    for batch in tqdm(batches, desc="VICL", total=len(batches), disable=len(batches)<=1):
        messages = [b[0] for b in batch]
        answers = [b[1] for b in batch]
        if 'qwen3-vl-30b' in cfg.VLM.lower():
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        elif 'qwen' in cfg.VLM.lower():
            batch_texts = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_dict=False
            )
            print(batch_texts[0])
            exit(111)
            batch_images, _ = process_vision_info(messages)
            inputs = processor(
                images=batch_images,  # List[Image]
                text=batch_texts,  # 拼接文本内容
                return_tensors="pt",
                padding=True
            )
            # dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        elif 'tonggu' in cfg.VLM.lower():   # really bad for TongGU
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            guided_text = []
            for message in messages:
                ccc = 0
                txt = ''
                content = message[0]["content"]
                for c in content:
                    if c["type"] == 'image':
                        ccc += 1
                    if c["type"] == 'text':
                        txt += c["text"]
                guided_text.append(txt+'<|vision_start|>{}<|vision_end|>'.format(''.join(["<|image_pad|>"]*ccc)))

            inputs_ocr = processor(text=guided_text, images=image_inputs, videos=video_inputs, padding=True,
                                   return_tensors="pt")
            inputs["input_ids_ocr"] = inputs_ocr["input_ids"]
            inputs["attention_mask_ocr"] = inputs_ocr["attention_mask"]
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        elif 'inter' in cfg.VLM.lower():
            processor.image_processor.image_aspect_ratio = "pad"
            batch_texts = processor.apply_chat_template(messages, tokenize=False)
            _, batch_images = prepare_batch_for_InternVL(cfg, messages)
            inputs = processor(
                text=batch_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt"
            )
            # dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


        elif 'glm' in cfg.VLM.lower():
            batch_texts = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_dict=False
            )
            _, batch_images = prepare_batch_for_InternVL(cfg, messages)
            # print(batch_texts)
            inputs = processor(
                images=batch_images,  # List[Image]
                text=batch_texts,  # 拼接文本内容
                return_tensors="pt",
                padding=True
            )

        elif 'deepseekocr' in cfg.VLM.lower():
            # 下述代码可以运行就是很慢
            from vllm import SamplingParams
            sampling_param = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
                ),
                skip_special_tokens=False,
            )
            model_input = prepare_batch_for_OCR(messages, processor)
            model_outputs = model.generate(model_input, sampling_param)
            for output, answer in zip(model_outputs, answers):
                ttt = output.outputs[0].text.split(':')[-1]
                if cfg.task == 'task3_2':
                    results.append(ttt.replace('\n', '') + '\t' + '；'.join(answer))
                else:
                    results.append(ttt.replace('\n', '') + '\t' + answer[0])
        elif 'llava' in cfg.VLM.lower():
            _, images = prepare_batch_for_InternVL(cfg, messages)
            batch_texts = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=batch_texts, images=images, return_tensors='pt', padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}

        elif 'minicpm' in cfg.VLM.lower():
            msgs = prepare_batch_for_MiniCPM(cfg, messages)
            try:
                model_outputs = model.chat(
                    msgs=msgs,
                    tokenizer=processor
                )
            except:
                model_outputs = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=processor
                )
            print(model_outputs, answers)
            for output, answer in zip(model_outputs, answers):
                ttt = output
                if cfg.task == 'task3_2':
                    results.append(ttt.replace('\n', '') + '\t' + '；'.join(answer))
                else:
                    results.append(ttt.replace('\n', '') + '\t' + answer[0])

        if 'ocr' not in cfg.VLM.lower() and 'cpm' not in cfg.VLM.lower():
            with torch.inference_mode():
                generated_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    output_scores=True,  # 返回每步 token 的 logits
                    return_dict_in_generate=True  # 返回一个 dict，包含 scores
                )
            generated_ids = generated_outputs.sequences
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            # 7. 解码
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assert len(output_text) == len(answers)
            for out, answer in zip(output_text, answers):
                print(out, answer)
                if cfg.task == 'task3_2':
                    results.append(out.replace('\n', '')+'\t'+'；'.join(answer))
                else:
                    results.append(out.replace('\n', '')+'\t'+answer[0])
    with open(f'results/{cfg.task}_{cfg.VLM}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

def clear_vlm_cache(model):
    for attr in ["vision_cache", "cache_kv", "kv_cache"]:
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except:
                pass

if __name__ == '__main__':
    cfg = Config()
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    cfg.batch_size = 16
    # for model in ["Qwen3-VL-2B-Instruct", "Qwen3-VL-8B-Instruct", "GLM-4.1V-9B-Thinking", 'InternVL3_5-1B-HF']:
    # Llama-3.2-11B-Vision
    for model in ['Qwen3-VL-GEVO2_sft']:
        cfg.VLM = model
        path = '/home/lyj/echo/LLaMA-Factory/saves/GEVO2_sft'
        model, processor = load_VLM(cfg, model_path=path)
        for task in ['task1_1', 'task1_2', 'task1_3', 'task1_4', 'task2_1', 'task2_2', 'task2_3', 'task2_4', 'task3_1',
                     'task3_2', 'task3_3']:
            cfg.task = task
            datas = load_task_datas(cfg)
            prompts = load_zero_shot_prompts(cfg, datas)
            batches = batchify(prompts, cfg.batch_size)
            VICL(cfg, batches, model, processor)
            if cfg.task == 'task3_2':
                rank_acc(cfg)
            else:
                cal_acc(cfg)
            clear_vlm_cache(model)
            torch.cuda.empty_cache()

# Yi-VL-6B\34B
# InternVL3_5VL
# MiniCPM-V-4_5
# zai-org/cogvlm2-llama3-chat-19B
