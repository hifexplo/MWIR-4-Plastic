# MWIR-4-Plastic: The Identification of Complex End-of-Life Industrial Plastic using Mid-wave Infrared Hyperspectral Imaging and  Machine Learning

This is the GitHub repository for the manuscript: https://arxiv.org/html/2608.28874v1

![[http://url/to/img.png](https://github.com/Elias-Arbash/MWIR-/main/images/readme.jpg)](https://github.com/Elias-Arbash/MWIR-4-Plastic/blob/main/images/readme.jpg)

Pixel-wise polymer classification (Styrene, PA, PC, PP, mix) from fused
FENIX + FX50 mid-wave infrared hyperspectral cubes. The repo has two
families of models:

- **Classical ML**: KNN, RF (Random Forest), SVM, LDA, plus SAM (Segment
  Anything Model) for instance masks used in majority voting.
- **Deep learning**: 1DCNN, ViT, SpectralFormer (pixel and patch variants).

Each model script does grid search + training + evaluation on its own, and
saves a "best model" checkpoint. An `inference/` folder additionally
provides inference-only scripts for the four **deep learning** models.

## Requirements

```
numpy, scikit-learn, scikit-image, matplotlib, joblib, spectral, Pillow
torch, einops                      # deep learning models only
```


## Dataset layout

All scripts expect the `EBS12_EBS22` dataset with this structure:

```
EBS12_EBS22/
    Styrene/<sample_dir>/
    PP/<sample_dir>/
    PA/<sample_dir>/
    PC/<sample_dir>/
    mix/<sample_dir>/
```

Each `<sample_dir>` contains:
- `FENIX.hdr/.dat`, `FX50.hdr/.dat` — ENVI hyperspectral cubes
- `RGB.hdr/.dat` — optional RGB preview cube
- `dTree_*.npy` — ground-truth class mask
- `semantic_mask.png` — binary foreground mask
- `instance.pkl` — SAM instance predictions (gzip+pickle)

The first sample per class is reserved as the held-out test cube; the rest
are used for training. This split is fixed inside each script.

## Shared modules

- `ebs_dataset_loader.py` — loads and parses the dataset above
- `functions.py` — SNV normalization, majority voting (semantic/instance),
  mask visualization
- `augmentations.py` — training-time data augmentation (DL models only)
- `models.py` — ViT / SpectralFormer architecture (used by `ViT.py`,
  `sf_pixel.py`, `sf_patch.py`)

## Training

Each model script is self-contained and runs its own grid search, trains,
picks the best config, evaluates, and saves outputs. Run from the repo
root:

```bash
# Classical ML
python KNN.py --dataset-root /path/to/EBS12_EBS22
python RF.py  --dataset-root /path/to/EBS12_EBS22
python SVM.py --dataset-root /path/to/EBS12_EBS22
python lda.py --dataset-root /path/to/EBS12_EBS22

# Deep learning
python 1DCNN.py    --dataset-root /path/to/EBS12_EBS22
python ViT.py       --dataset-root /path/to/EBS12_EBS22
python sf_pixel.py  --dataset-root /path/to/EBS12_EBS22
python sf_patch.py  --dataset-root /path/to/EBS12_EBS22
```

Common arguments: `--dataset-root`, `--output-root` (defaults to
`<dataset_root>/trained_models/<model>/`), `--max-experiments` (grid-search
budget). Deep learning scripts also take `--epochs`, `--patience`,
`--batch-size`, `--seed`.

Corresponding `.ipynb` notebooks (same name) are the exploratory/notebook
version of each script, useful for interactive runs (e.g. Colab/Jupyter).

Training outputs (per model, under `--output-root`):
```
best_model/<name>_best_model.pth   # (or .joblib for classical ML)
best_results.json                  # evaluation metrics
prediction_maps/<sample>/...       # pixel-wise / semantic-majority / instance-majority maps
```

## Inference (deep learning models only)

`inference/` contains inference-only scripts for `1DCNN`, `ViT`,
`sf_pixel`, and `sf_patch` — same architecture, data loading, and
evaluation code as training, but no training loop or grid search. Point
`--model-path` at a checkpoint produced above:

```bash
python inference/1DCNN_inference.py    --model-path /path/to/1DCNN_model.pth
python inference/ViT_inference.py      --model-path /path/to/vit_model.pth
python inference/sf_pixel_inference.py --model-path /path/to/sf_pixel_model.pth
python inference/sf_patch_inference.py --model-path /path/to/sf_patch_model.pth
```

See [`inference/README.md`](inference/README.md) for full argument details
and output format. Classical ML models (KNN/RF/SVM/LDA) don't have a
separate inference script — reload the saved `.joblib` pipeline directly
with `joblib.load(...)`.


## Dataset repo:
The dataset is available at: https://zenodo.org/records/22142947

## Citation:
@article{arbash2026mwir,
  title={MWIR-4-Plastic: The Identification of Complex End-of-Life Industrial Plastic using Mid-wave Infrared Hyperspectral Imaging and Machine Learning},
  author={Arbash, Elias and Ribeiro, Andr{\'e}a de Lima and Sim{\~o}es, Filipa and Afifi, Ahmed Jamal and Rizaldy, Aldino and Madriz, Yuleika and Thiele, Samuel and Lorenz, Sandra and Fuchs, Margret and Ghamisi, Pedram and others},
  journal={arXiv preprint arXiv:2608.28874},
  year={2026}
