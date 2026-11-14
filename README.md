# Code for the Vietnamese sidewalk encroachment VQA dataset

Code accompanying the Data Descriptor *A Vietnamese visual question answering dataset of sidewalk encroachment linked to traffic regulations*.

The dataset (images, question and answer files, datasheet) is on Figshare: https://doi.org/10.6084/m9.figshare.35010361

This repository contains only code. It does not contain images, labels or trained weights.

## Contents

| Folder | Content |
|---|---|
| `annotation_tool/` | `vqa_annotate.py`, the tkinter tool used to label images with the ten fixed questions. `compute_iaa.py` computes raw agreement and Cohen's kappa between two annotators. |
| `baseline_classifier/` | Frozen-encoder classifier: CLIP ViT-L/14 image features and PhoBERT question features with a classification head. Includes the five imbalance methods, 5-fold cross-validation and a ViT-L/14 fine-tuning variant. |
| `qwen_vlm/` | QLoRA fine-tuning of Qwen2.5-VL-3B-Instruct, two-stage re-balancing, ablations, sequence-bias post-processing and the zero-shot evaluation. |

Each `build_*.py` file generates the notebook with the matching name. The generated `.ipynb` files are included, so the builders only need to be run after editing them.

## Data layout

Download the files from Figshare and unzip `images_blurred.zip`. The notebooks were written for Kaggle and look for `train.csv`, `val.csv`, `test.csv` and the images under `/kaggle/input/`. To run elsewhere, change the paths at the top of each notebook.

The CSV columns are described in the Figshare `README.md`. The code reads `filename`, `qid`, `question`, `answer` and `question_type`.

## Environment

```
Python 3.10 or later
pip install "transformers>=4.49" accelerate qwen-vl-utils "bitsandbytes>=0.46.1" peft
pip install torch torchvision timm scikit-learn pandas numpy matplotlib openpyxl tqdm pillow
```

The CLIP, PhoBERT and Qwen2.5-VL-3B-Instruct weights are downloaded from their public releases when the notebooks run.

Hardware used: the classifier runs on CPU once the features are cached, and fine-tuning ViT-L/14 needs a GPU with at least 8 GB. The Qwen notebooks were run on two NVIDIA T4 GPUs (Kaggle) with 16 GB each. A single session was limited to 12 hours.

## Order for the Qwen experiments

1. `qwen_vlm/nb_qwen_finetune.ipynb` trains the baseline QLoRA adapter.
2. `qwen_vlm/nb_qwen_ft_twostage.ipynb` trains the two-stage model. It needs the baseline adapter as an input.
3. `nb_qwen_twostage_ablation_a.ipynb` and `nb_qwen_twostage_ablation_b.ipynb` vary the ratio cap and the number of epochs.
4. `nb_qwen_seqbias_posthoc.ipynb` and `nb_qwen_twostage_seqbias.ipynb` apply sequence-bias post-processing without retraining.
5. `nb_qwen_zeroshot_fulltest.ipynb` gives the zero-shot reference on the full test split. It is independent of the others.

## Licence

MIT, see `LICENSE`. The dataset has its own licence (CC BY 4.0).

## Contact

Trong Hop Do, University of Information Technology, Vietnam National University Ho Chi Minh City, hopdt@uit.edu.vn
