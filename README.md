# Sentinel-2 Land Cover Classification with CNN Transfer Learning
### Earth Observation Pipeline for Poverty Proxy Estimation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-EuroSAT%20RGB-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Open in Colab (Full)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pujapoudyal/sentinel2-cnn-poverty-proxy/blob/main/sentinel2_cnn_poverty_proxy.ipynb)

---

## Overview

This repository demonstrates a complete deep-learning pipeline applied to **Sentinel-2 satellite imagery**, classifying 10 land cover types using ResNet-18 transfer learning. It serves as a technical foundation for Earth Observation-based poverty proxy estimation.

**Why land cover classification for poverty estimation?**
Spectral and spatial features that distinguish Residential from Industrial from Bare Soil patches are the same feature families that poverty estimation models use as wealth-index proxies — building density, road access, vegetation cover, and land use intensity. GradCAM attribution maps trained on land cover tasks directly parallel the explainability requirements of satellite-based development research.

Two notebooks are provided: a **simplified version** for clarity and a **full version** with two-phase training, GradCAM attribution, and a detailed bridge to causal inference in poverty estimation.

---

## Repository Structure

```
sentinel2-cnn-poverty-proxy/
│
├── sentinel2_cnn_poverty_proxy.ipynb   # Full pipeline (recommended)
├── Sentinel_2_CNN.ipynb                # Simplified version
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Notebooks at a Glance

| Feature | `Sentinel_2_CNN.ipynb` | `sentinel2_cnn_poverty_proxy.ipynb` |
|---|---|---|
| **Purpose** | Core CNN pipeline, clean and minimal | Full research pipeline |
| **Data source** | Direct ZIP download (DFKI) | `torchvision.datasets.EuroSAT` |
| **Training phases** | Single phase | Two-phase (frozen → fine-tune) |
| **Epochs** | 5 | 15 (8 + 7) |
| **LR scheduler** | None | CosineAnnealingLR |
| **Best-model checkpoint** | No | Yes |
| **GradCAM attribution** | No | Yes (per-class heatmaps) |
| **Training curves** | No | Yes (with phase boundary) |
| **Poverty proxy discussion** | No | Yes (with causal inference bridge) |
| **Split** | 80 / 10 / 10 | 70 / 15 / 15 |
| **Recommended for** | Learning the architecture | Research portfolio |

---

## Dataset — EuroSAT RGB (Sentinel-2)

| Property | Value |
|---|---|
| Sensor | Sentinel-2 (ESA Copernicus Programme) |
| Bands used | B4 (Red), B3 (Green), B2 (Blue) |
| Image size | 64 × 64 pixels (native) |
| Total patches | 27,000 |
| Classes | 10 land cover types |
| Size | ~89 MB (RGB ZIP) |
| Licence | MIT |
| Citation | Helber et al. (2019), IEEE JSTARS |

### Land Cover Classes

| Class | Poverty Proxy Signal |
|---|---|
| Residential | Building density, settlement structure |
| Industrial | Formal economic activity, wage employment |
| Highway / River | Infrastructure access, market connectivity |
| AnnualCrop / PermanentCrop | Agricultural land use intensity |
| Forest / Pasture | Low human modification, remoteness |
| HerbaceousVegetation | Marginal land, ecological buffer zones |
| SeaLake | Coastal / inland water access |

---

## Model Architecture

```
Input (3 × 224 × 224)
│
└── ResNet-18 (pretrained ImageNet)
    │
    ├── conv1 → bn1 → relu → maxpool
    ├── layer1 (2 × BasicBlock)   ← FROZEN (Phase 1 + 2)
    ├── layer2 (2 × BasicBlock)   ← FROZEN (Phase 1 + 2)
    ├── layer3 (2 × BasicBlock)   ← FROZEN (Phase 1 + 2)
    ├── layer4 (2 × BasicBlock)   ← FROZEN Phase 1 | TRAINABLE Phase 2
    │       └── [GradCAM target: 7 × 7 feature maps]
    └── FC head (replaced)
        Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → 10)
        ← TRAINABLE both phases
```

**Transfer learning rationale:** Early ResNet layers encode generic visual features (edges, textures, shapes) that transfer well from ImageNet to satellite imagery. Layer4 encodes higher-level semantics that benefit from domain adaptation to Sentinel-2 reflectance characteristics.

---

## Quick Start — Google Colab

No local setup required. Click the badge at the top of this README or follow these steps:

1. Open the notebook in Colab
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. Run all: `Runtime → Run all`
4. Expected runtime: **10–15 minutes** on T4 GPU

> **Note:** The full notebook (`sentinel2_cnn_poverty_proxy.ipynb`) requires
> `pytorch-grad-cam`, which is installed automatically in the first cell.

---

## Local Installation

```bash
# Clone the repository
git clone https://github.com/pujapoudyal/sentinel2-cnn-poverty-proxy.git
cd sentinel2-cnn-poverty-proxy

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

GPU is recommended but not required. On CPU, expect approximately 30 minutes for the full notebook.

---

## Expected Results

Reported on the held-out test set (15% of EuroSAT, never seen during training).
Exact values vary slightly with hardware and random seed.

| Metric | Typical range |
|---|---|
| Overall accuracy | 88 – 94% |
| Forest F1 | 0.97 – 0.99 |
| AnnualCrop F1 | 0.88 – 0.93 |
| AnnualCrop / PermanentCrop confusion | Most common error |

The most frequent misclassification is AnnualCrop vs PermanentCrop — both are
agricultural patches that share similar spectral signatures. This confusion is
consistent with published EuroSAT benchmarks and reflects genuine ambiguity
in the RGB representation (multi-spectral bands resolve it more cleanly).

---

## GradCAM Attribution

The full notebook generates GradCAM heatmaps for one test patch per class.
These maps show **which spatial regions** drove each prediction.

A model that highlights building footprints for Residential patches and
road geometry for Highway patches is relying on physically meaningful features.
A model that highlights background texture is exploiting dataset artefacts.

GradCAM is therefore a **model validation step**, not merely a visualisation.
In poverty estimation applications, unexplainable predictions cannot inform
budget allocations. Attribution-validated predictions can.

---

## From Land Cover to Poverty Estimation

This pipeline is a technical precursor to EO-based poverty proxy estimation.
The methodological steps from land cover classification to poverty mapping are:

1. **This notebook:** Train CNN on labelled land cover patches (Sentinel-2 RGB)
2. **Extension 1:** Replace RGB with all 13 Sentinel-2 bands (NDVI, SWIR add
   discriminative power for crop health and bare soil)
3. **Extension 2:** Stack patches across acquisition dates for temporal dynamics
   (construction activity, seasonal crop cycles)
4. **Extension 3:** Replace classification head with regression head predicting
   a continuous wealth index (DHS asset index, night-light intensity)
5. **Extension 4:** Apply Prediction-Powered Inference (PPI; Angelopoulos et al.
   2023) to produce valid confidence intervals in regions with sparse ground truth

### The causal inference gap

The model above is **predictive**: it learns that metal roofs correlate with
Residential patches, which correlate with higher wealth. The roof is a proxy,
not a causal lever. Moving from prediction to **causal inference** — identifying
what interventions change poverty outcomes — requires:

- Accounting for **selection bias**: highland stations with missing observations
  are not missing at random. Treating that absence as informative rather than
  noise is the statistical foundation on which valid poverty estimates depend.
- **Prediction-Powered Inference** (Angelopoulos et al. 2023): uses ML
  predictions as first-stage imputations, then applies classical inference
  with valid statistical guarantees. This is the regime of global poverty
  mapping, where ground-truth survey labels are scarce but satellite predictions
  are abundant.

---

## Technical Notes

**Why resize 64 × 64 to 224 × 224?**
EuroSAT images are natively 64 × 64. ResNet-18 uses adaptive average pooling,
so it can technically handle any input size. However, at 64 × 64, the
`layer4` feature maps are only 2 × 2, which produces uninformative GradCAM
heatmaps. Resizing to 224 × 224 gives 7 × 7 feature maps at `layer4`,
enabling spatially meaningful attribution after bilinear upsampling.

**Why ImageNet normalisation?**
We use ImageNet mean/std ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
rather than EuroSAT-specific statistics because we are fine-tuning a
pretrained backbone. The pretrained weights assume ImageNet statistics;
deviating too far degrades transfer quality in early training.
EuroSAT-specific normalisation becomes preferable when training from scratch.

---

## References

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel
  dataset and deep learning benchmark for land use and land cover
  classification. *IEEE Journal of Selected Topics in Applied Earth
  Observations and Remote Sensing*, 12(7), 2217–2226.

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
  Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via
  gradient-based localization. *ICCV 2017*.

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-powered inference. *Science*, 382(6671), 669–674.

- Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., & Ermon, S.
  (2016). Combining satellite imagery and machine learning to predict
  poverty. *Science*, 353(6301), 790–794.

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
  image recognition. *CVPR 2016*.

---

## Citation

If you use this code in your work, please cite:

```bibtex
@misc{poudel2025sentinel2,
  author    = {Poudel, Puja},
  title     = {Sentinel-2 Land Cover Classification with CNN Transfer Learning:
               Foundation for EO-Based Poverty Proxy Estimation},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/pujapoudyal/sentinel2-cnn-poverty-proxy}
}
```

Related MSc thesis:

```bibtex
@mastersthesis{poudel2025thesis,
  author = {Poudel, Puja},
  title  = {Predictive Analysis of Global Temperature Trends Using Machine
            Learning Algorithms: The Region of South America},
  school = {Stockholm University},
  year   = {2025},
  note   = {DiVA: urn:nbn:se:su:diva-250840}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Dataset (EuroSAT) is separately licensed under MIT by Helber et al.
Pretrained ResNet-18 weights are provided by PyTorch under the BSD licence.
