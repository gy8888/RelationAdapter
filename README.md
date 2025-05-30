# RelationAdapter

> **RelationAdapter: Learning and Transferring Visual Relation with Diffusion Transformers**
> <br>
> Yan Gong, 
> Yiren Song, 
> Yicheng Li, 
> Chenglin Li, 
> and 
> Yin Zhang,
> <br>
> Zhejiang University, National University of Singapore
> <br>

<a href="https://arxiv.org/abs/2502.14397"><img src="https://img.shields.io/badge/ariXv-2502.14397-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/nicolaus-huang/PhotoDoodle"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/nicolaus-huang/PhotoDoodle/"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/ameerazam08/PhotoDoodle-Image-Edit-GPU"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>

<br>

<img src='./assets/teaser.png' width='100%' />

## Quick Start
### Configuration
#### 1. **Environment setup**
```bash
git clone git@github.com:gy8888/RelationAdapter.git
cd RelationAdapter

conda create -n RelationAdapter python=3.11.10
conda activate RelationAdapter
```
#### 2. **Requirements installation**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
```


### 2. Inference
We provided the integration of diffusers pipeline with our model and uploaded the model weights to huggingface, it's easy to use the our model as example below:

simply run the inference script:
```
python infer_single.py
```


### 3. Weights
You can download the trained checkpoints of RelationAdapter for inference. Below are the details of available models, checkpoint name are also trigger words.

You would need to load and fuse the `pretrained ` checkpoints model in order to load the other models.

|                          **Model**                           |                       **Description**                       | **Resolution** |
| :----------------------------------------------------------: | :---------------------------------------------------------: | :------------: |
| [pretrained](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/pretrain.safetensors) |       PhotoDoodle model trained on `SeedEdit` dataset       |    768, 768    |
| [sksmonstercalledlulu](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/sksmonstercalledlulu.safetensors) |   PhotoDoodle model trained on `Cartoon monster` dataset    |    768, 512    |
| [sksmagiceffects](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/sksmagiceffects.safetensors) |      PhotoDoodle model trained on `3D effects` dataset      |    768, 512    |
| [skspaintingeffects ](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/skspaintingeffects.safetensors) | PhotoDoodle model trained on `Flowing color blocks` dataset |    768, 512    |
| [sksedgeeffect ](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/sksedgeeffect.safetensors) |  PhotoDoodle model trained on `Hand-drawn outline` dataset  |    768, 512    |


### 4. Dataset
<span id="dataset_setting"></span>
#### 2.1 Settings for dataset
The training process uses a paired dataset stored in a .jsonl file, where each entry contains image file paths and corresponding text descriptions. Each entry includes the source image path, the target (modified) image path, and a caption describing the modification.

Example format:

```json
{"source": "path/to/source.jpg", "target": "path/to/modified.jpg", "caption": "Instruction of modifications"}
{"source": "path/to/source2.jpg", "target": "path/to/modified2.jpg", "caption": "Another instruction"}
```

We have uploaded our datasets to [Hugging Face](https://huggingface.co/datasets/nicolaus-huang/PhotoDoodle).


### 5. Results

![S-U](./assets/result_show.png)


## Citation
```
@misc{huang2025photodoodlelearningartisticimage,
      title={PhotoDoodle: Learning Artistic Image Editing from Few-Shot Pairwise Data}, 
      author={Shijie Huang and Yiren Song and Yuxuan Zhang and Hailong Guo and Xueyin Wang and Mike Zheng Shou and Jiaming Liu},
      year={2025},
      eprint={2502.14397},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.14397}, 
}
```