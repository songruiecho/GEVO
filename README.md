# GEVO

We have open-sourced **GEVO**, a benchmark for ancient Chinese character evolution analysis, including a curated dataset, evaluation instructions for Multimodal Large Language Models (MLLMs), and the corresponding training and evaluation code.

Our paper, **"Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning"**, has been accepted to **ACL 2026 Main Conference**.

## Benchmark Resources

### Images

The benchmark images can be downloaded from:

[**Images**](https://drive.google.com/file/d/1Nw-h5maGGPZurGxBnSX_L1R_IXKHju-R/view?usp=drive_link)

### Evaluation Instructions

The evaluation instructions for MLLMs are available at:

[**Instructions**](https://drive.google.com/file/d/1uSD9N1zRqrdcwWyBfFAbXcasIEIb7M-D/view?usp=drive_link)

### OOD Evaluation Set

To further assess model generalization, we provide a lightweight out-of-distribution (OOD) evaluation set containing approximately **150 ancient Chinese characters** written in diverse calligraphic styles. All samples have been carefully verified by domain experts. The writing styles in this OOD set are intentionally different from those used in the main benchmark images.

[**OOD Images**](https://drive.google.com/file/d/1TXqueWlv5AYCX8lQUN6a-2VyOtOZv3Fo/view?usp=drive_link)

## Usage Notes

To evaluate different MLLMs on GEVO, users may need to:

1. Adapt the instruction format to match the input requirements of specific models.
2. Modify the image paths contained in the instruction files to correspond to local storage directories.

## GEVO Model

The GEVO model, built upon **Qwen3-VL-2B-Instruct**, is publicly available on Hugging Face:

[**GEVO on Hugging Face**](https://huggingface.co/Rui1996/GEVO/)

The model can be downloaded and deployed using the same workflow as other models in the Qwen3-VL series.

Detailed usage examples and deployment instructions are provided in the corresponding Hugging Face model card.

## News

- **2026.06**: Our paper *Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning* was accepted to **ACL 2026 Main Conference**.

## Citation

If you find GEVO useful in your research, please consider citing our work:

```bibtex
@article{song2026gevo,
  title={Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning},
  author={Song, Rui and Shi, Lida and Qi, Ruihua and Li, Yingji and Xu, Hao},
  journal={arXiv preprint arXiv:2604.11299},
  year={2026}
}
```
