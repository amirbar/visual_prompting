# Visual Prompting via Image Inpainting
### [Amir Bar*](https://amirbar.net), [Yossi Gandelsman*](https://yossi.gandelsman.com/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Amir Globerson](http://www.cs.tau.ac.il/~gamir/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)
![Visual Prompting](https://yossigandelsman.github.io/visual_prompt/images/teaser.png)

This repository is the implementation of the paper, for more info about this work see [Project Page](https://yossigandelsman.github.io/visual_prompt/).

# Release Checklist
- [X] Data
- [ ] Code
- [ ] Colab Demo

## Abstract
How does one adapt a pre-trained visual model to novel downstream tasks without task-specific finetuning or any model modification? Inspired by prompting in NLP, this paper investigates visual prompting: given input-output image example(s) of a new task at test time and a new input image, the goal is to automatically produce the correct output image, consistent with the example(s) task. We show that posing this problem as a simple image inpainting task - literally just filling in a hole in a concatenated visual prompt image - turns out to be surprisingly effective, given that the inpainting algorithm has been trained on the right data. We train masked auto-encoding models on a new dataset that we curated - 88k unlabeled figures from academic papers sources on Arxiv. We apply visual prompting to these pretrained models and demonstrate results on various downstream tasks, including foreground segmentation, single object detection, colorization, edge detection, etc.

## Computer Vision Figures Dataset
To download the dataset run:

```
cd figures_dataset
sudo apt-get install poppler-utils
pip install -r requirements.txt
```

Download train/val:
```
python download_links.py --output_dir <path_to_output_dir> --split train
python download_links.py --output_dir <path_to_output_dir> --split val
```

**Note**: the paper sources are hosted by arXiv and download time might take 2-3 days. <br>For inquiries/questions about this please email the authors directly.







