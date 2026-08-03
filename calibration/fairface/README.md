# FairFace weights

Pretrained ResNet-34 checkpoints from [FairFace](https://github.com/dchen236/FairFace).

These `.pt` files are **not** stored in git (~82 MB each). Download into this folder:

```bash
# 7-class (default for D65-FairFace7-ROI)
gdown 11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB -O res34_fair_align_multi_7_20190809.pt
```

Optional 4-class weights (`fairface_alldata_4race_20191111.pt`) are linked from the FairFace repo README.

The Colab walkthrough (`d65_fairface7_roi_walkthrough.ipynb`) downloads the 7-class weights automatically.
