
<br />
<p align="center">

  <h3 align="center">Text to Point Cloud Localization with Multi-Level Negative Contrastive Learning</h3>

</p>

  <p align="center">
      <a href="https://asc.xmu.edu.cn/persons" target='_blank'>Dunqiang Liu*</a>,&nbsp;
      <a href="https://asc.xmu.edu.cn/persons" target='_blank'>Shujun Huang*</a>,&nbsp;
      <a href="https://asc.xmu.edu.cn/persons" target='_blank'>Wen Li</a>,&nbsp;
      <a href="https://asc.xmu.edu.cn/persons" target='_blank'>Siqi Shen</a>,
      <a href="https://asc.xmu.edu.cn/persons" target='_blank'>Cheng WANG†</a>
    <br>
   Xiamen University,&nbsp ASC Lab&emsp;
  <br>
   <sup>*</sup> Equal Contribution&emsp;
 <sup>†</sup> Corresponding Author&emsp;
  </p>
</p>


<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32574" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-yellow">
  </a>

<a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=dqliua.MNCL&left_color=gray&right_color=lightblue">
  </a>

</p>


<br>
<p align="center">
  <img src="https://github.com/dqliua/MNCL/blob/main/imgs/model.png" align="center" width="90%">
  <br>
In the cross-modal place recognition stage, we introduce a multi-level
negative contrastive learning framework to minimize the similarity of different locations at global-level, instance-level, and
relation-level, respectively. This fully leverages the descriptive power of language for spatial localization. In the fine localization
stage, we use the language query and the retrieved cell to regress the corresponding position.
</p>
<br>

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Train](#train)
- [Eval](#eval)
- [Test](#test)
- [Citation](#citation)

## Installation

Create the environment using the following command.

```
git clone https://github.com/dqliua/MNCL.git

conda create -n mncl python=3.10
conda activate mncl

# Install the according versions of torch and torchvision
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```


## Data Preparation


We use the publicly available dataset KITTI360Pose. You can download the KITTI360Pose dataset from [here](https://cvg.cit.tum.de/webshare/g/text2pose/). 

For dataset details, kindly refer to [Text2Pos](https://arxiv.org/abs/2203.15125). 

The dataset folder should display as follow:

```html
data
└── KITTI360Pose
    └── k360_30-10_scG_pd10_pc4_spY_all
        ├── cells
        ├── direction
        ├── poses
        ├── street_centers
        └── visloc
```


## Model Zoo
The table below lists the pretrained weights in our method. These include the default text encoder and the 3D point cloud backbone. You can download them directly from the provided links.

| Component              | Model                                           | Download Link                                                                 |
|------------------------|-------------------------------------------------------|--------------------------------------------------------------------------------|
| **Text Backbone**       | Flan-T5                          | [Hugging Face](https://huggingface.co/google/flan-t5-large)                   |
| **Object Backbone** | PointNet  | [Google Drive](https://drive.google.com/file/d/1j2q67tfpVfIbJtC1gOWm7j8zNGhw5J9R/view) |



After completing the above steps, the basic directory structure should be like:

```
MNCL
 ├── checkpoints
      ├── coarse.pth
      ├── fine.pth
      └── pointnet_acc0.86_lr1_p256_model.pth
 ├── data
      └── KITTI360Pose
            └── k360_30-10_scG_pd10_pc4_spY_all
                ├── cells
                ├── direction
                ├── poses
                ├── street_centers
                └── visloc
 ├── dataloading
      └── .....
 ├── datapreparation
      └── .....
 ├── evalution
      └── .....
 ├── models
      └── .....
 ├── t5-large
      └── .....
 ├── training
      └── .....
```


## Train

After configuring the dependencies and preparing the dataset, use the following commands to train the coarse retrieval and fine localization, respectively.


**Coarse Retrieval**

```
python -m training.coarse  \
  --batch_size 64  \
  --coarse_embed_dim 256  \
  --shuffle  \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --lr_scheduler step \
  --lr_step 5 \
  --lr_gamma 0.5 \
  --temperature 0.05 \
  --ranking_loss CCL \
  --num_of_hidden_layer 3 \
  --alpha 2 \
  --hungging_model t5-large \
  --folder_name PATH_TO_COARSE
```


**Fine Localization**

```
python -m training.fine 
  --batch_size 32 \ 
  --fine_embed_dim 128 \ 
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0003 \
  --fixed_embedding \
  --hungging_model t5-large \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --folder_name PATH_TO_FINE \
```

## Eval

**Evaluation coarse retrieval only on val set**

```
python -m evaluation.coarse
	--base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
```

**Evaluation whole pipeline on val set**

```
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} \
```


## Test



**Test coarse retrieval only on test set**

```
python -m evaluation.coarse 
	--base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
```


**Test whole pipeline on test set**

```
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```


## Citation

If you find this work helpful, please kindly consider citing our paper:

```bibtex
@inproceedings{liu2025text,
  title={Text to point cloud localization with multi-level negative contrastive learning},
  author={Liu, Dunqiang and Huang, Shujun and Li, Wen and Shen, Siqi and Wang, Cheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={5397--5405},
  year={2025}
}
```
