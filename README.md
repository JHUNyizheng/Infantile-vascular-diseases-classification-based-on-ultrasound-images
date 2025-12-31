# IVDNet: Intelligent Multimodal Classification of Infantile Vascular Diseases

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-green.svg)](requirements.txt)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.8%2B-orange.svg)](https://pytorch.org/)

## Project Introduction
This repository implements **IVDNet** (Infantile Vascular Disease Network), an intelligent diagnostic model proposed in the paper *"Intelligent multimodal classification of infantile vascular diseases based on ultrasound images"*. The model aims to solve two core clinical tasks simultaneously:
1. **Classification** of infantile hemangiomas (IH), venous malformations (VM), and normal skin (NR) using multimodal ultrasound images.
2. **Detection** of phleboliths (vein stones) in VM ultrasound images.

IVDNet fuses grayscale and color Doppler ultrasound data to leverage complementary structural and blood flow information, and supports robust inference even when color Doppler images are missing.

## Key Features
- **Multimodal Fusion**: Combines grayscale ultrasound (tissue structure) and color Doppler ultrasound (blood flow) for improved diagnostic accuracy.
- **Dual-Task Learning**: Simultaneously performs disease classification and phlebolith detection via a shared network backbone.
- **Missing Modality Adaptation**: Automatically handles cases where color Doppler images are unavailable (uses zero-padding and adaptive feature scoring).
- **Advanced Feature Enhancement**: Integrates Squeeze-and-Excitation (SE) attention, Pyramid Pooling Module (PPM), and multi-scale decoding for precise feature extraction.
- **State-of-the-Art Performance**: Outperforms classical CNNs and target detection models on both classification and detection tasks.

## File Structure
```
├── Vis_Results/          # Visualization results (attention maps, ROC curves, etc.)
├── draw/                 # Plotting scripts for experimental results
├── src/                  # Core model and utility functions
│   ├── model.py          # IVDNet architecture definition
│   ├── loss.py           # Joint loss function for dual tasks
│   └── utils.py          # Data preprocessing and metric calculation
├── attentionMap.py       # Attention map visualization (Grad-CAM)
├── attentionMapCAM.py    # CAM-based attention visualization
├── dataset_prepare.py    # Data preprocessing (cropping, normalization, augmentation)
├── drawdata.py           # Plot experimental metrics (F1-score, confusion matrix)
├── inference.py          # Inference script for new ultrasound images
├── opts.py               # Hyperparameter configuration
├── requirements.txt      # Dependencies
├── test.py               # Model testing and evaluation
├── train.py              # Model training
└── README.md             # Project documentation
```

## Environment Requirements
### Hardware
- GPU: NVIDIA RTX 4090 (or equivalent, ≥16GB VRAM)
- CPU: Intel Core i7/i9 or AMD Ryzen 7/9
- RAM: ≥32GB

### Software
- Python 3.7+
- PyTorch 1.8+
- CUDA 11.3+
- Dependencies: See `requirements.txt`
  ```bash
  pip install -r requirements.txt
  ```
  Key dependencies include:
  - `numpy`, `pandas`, `scikit-learn` (data processing)
  - `torch`, `torchvision` (model training)
  - `opencv-python` (image processing)
  - `matplotlib`, `seaborn` (result visualization)
  - `labelme` (data labeling, for phlebolith detection)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/JHUNyizheng/Infantile-vascular-diseases-classification-based-on-ultrasound-images.git
cd Infantile-vascular-diseases-classification-based-on-ultrasound-images
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Preparation
#### Dataset Overview
The dataset contains 2,841 ultrasound images, including two sub-datasets:
| Dataset Type               | Content                                                                 | Size                  |
|----------------------------|-------------------------------------------------------------------------|-----------------------|
| Multimodal Classification  | Grayscale + color Doppler images of IH, VM, and NR                      | 1,379 groups (2,758 images) |
| Phlebolith Detection       | Grayscale ultrasound images of VM with phleboliths                      | 95 images             |

#### Data Access
The dataset is not publicly available due to privacy constraints. For access, contact the corresponding authors (see [Citation](#citation) after paper accecpted.).

#### Preprocessing
Run `dataset_prepare.py` to process raw data:
```bash
python dataset_prepare.py --data_root /path/to/raw_data --save_root /path/to/processed_data
```
Preprocessing steps (automated):
1. **Cropping**: Remove irrelevant backgrounds (patient info, device labels) to retain Region of Interest (ROI).
2. **Normalization**: Standardize pixel values to eliminate distribution differences:
   ```math
   X_{\text{norm}} = \frac{X - \mu}{\sigma}
   ```
3. **Data Augmentation** (for training set only):
   - Random flipping (horizontal/vertical)
   - Random rotation (±15°)
   - Luminance perturbation

### 4. Model Training
Configure hyperparameters in `opts.py` (e.g., batch size, learning rate) or pass them via command line:
```bash
python train.py \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.001 \
  --data_path /path/to/processed_data \
  --save_path ./checkpoints \
  --early_stop 20  # Stop training if val performance plateaus for 20 epochs
```
- Training strategy: Adam optimizer + cosine annealing learning rate scheduler.
- Joint loss function: Balances classification loss (cross-entropy) and detection loss (CIoU + Focal Loss).

### 5. Model Testing
Evaluate the trained model on the test set:
```bash
python test.py \
  --model_path ./checkpoints/best_model.pth \
  --test_data_path /path/to/test_data \
  --output_path ./test_results
```
Outputs:
- Classification metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC.
- Detection metrics: F1-score, mAP50, mAP50-95.
- Visualizations: Confusion matrix, ROC curves, phlebolith bounding boxes.

### 6. Inference on New Data
Use `inference.py` to classify diseases and detect phleboliths for new ultrasound images:
```bash
# For multimodal input (grayscale + color Doppler)
python inference.py \
  --model_path ./checkpoints/best_model.pth \
  --gray_img /path/to/gray_image.jpg \
  --doppler_img /path/to/doppler_image.jpg \
  --output_dir ./inference_results

# For unimodal input (grayscale only, if Doppler is missing)
python inference.py \
  --model_path ./checkpoints/best_model.pth \
  --gray_img /path/to/gray_image.jpg \
  --output_dir ./inference_results
```
Outputs:
- Disease classification result (IH/VM/NR) with confidence.
- Phlebolith detection results (bounding boxes + confidence scores).

## Model Architecture
IVDNet consists of 5 core components:
1. **Data Input Layer**: Accepts grayscale (ROI structure) and color Doppler (blood flow) images.
2. **Feature Extraction Module**: Two parallel CNN backbones with SE attention to enhance key features.
3. **Feature Fusion & Scoring Module**: Depthwise separable convolution + adaptive scoring to fuse multimodal features.
4. **Feature Decoding & Detection Module**: Multi-scale decoding (skip connections) + PPM for global context, dual-task heads (classification + detection).
5. **Output Layer**: Disease category probabilities and phlebolith bounding boxes.

![IVDNet Architecture](Vis_Results/model_architecture.png)
*(See paper for detailed structure, Fig. 1)*

## Experimental Results
### Classification Performance (IH/VM/NR)
| Model          | Accuracy | Precision | Recall | F1-score | AUC  |
|----------------|----------|-----------|--------|----------|------|
| AlexNet        | 0.763    | 0.736     | 0.742  | 0.738    | 0.907|
| VGG16          | 0.754    | 0.546     | 0.574  | 0.528    | 0.804|
| ResNet34       | 0.847    | 0.810     | 0.792  | 0.791    | 0.953|
| DenseNet121    | 0.885    | 0.886     | 0.883  | 0.882    | 0.973|
| EfficientNet   | 0.908    | 0.905     | 0.908  | 0.906    | 0.978|
| CA+EfficientNet| 0.933    | 0.938     | 0.933  | 0.932    | 0.983|
| **IVDNet**     | 0.956    | 0.963     | 0.955  | 0.953    | 0.986|

### Phlebolith Detection Performance
| Model          | F1-score | mAP50 | mAP50-95 |
|----------------|----------|-------|----------|
| Faster R-CNN   | 0.713    | 0.627 | 0.311    |
| YOLOv5l        | 0.853    | 0.772 | 0.473    |
| YOLOv7l        | 0.889    | 0.867 | 0.519    |
| **IVDNet**     | 0.949    | 0.934 | 0.742    |

### Ablation Study (Multimodal vs. Unimodal)
| Input Modality | Accuracy | F1-score | AUC  |
|----------------|----------|----------|------|
| Grayscale Only | 0.945    | 0.940    | 0.973|
| Multimodal     | 0.956    | 0.953    | 0.986|

## Dataset Details
### Inclusion/Exclusion Criteria
| Dataset               | Inclusion Criteria                                                                 | Exclusion Criteria                                                                 |
|-----------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Classification        | 1. Clear ultrasound images; 2. Confirmed IH/VM/NR diagnosis; 3. Complete clinical info. | 1. Blurry images with artifacts; 2. Incomplete clinical info.                     |
| Phlebolith Detection  | 1. Clear grayscale images; 2. Confirmed phleboliths; 3. No blood flow obstruction. | 1. Blurry images with artifacts; 2. Blood flow obscures phleboliths.              |

### Data Split
- Training set: 80% of total data
- Validation set: 10% of total data
- Test set: 10% of total data

## Citation
If you use this code or model in your research, please cite the original paper:
```bibtex
@article{zheng2024intelligent,
  title={Intelligent multimodal classification of infantile vascular diseases based on ultrasound images},
  author={Zheng, Yi and Liao, Cunyi and Xiong, Ping and Fan, Xindong and Gong, Xia and He, Qiang},
  journal={[To be submitted]},
  year={2024},
  affiliation={Intelligent Medical Engineering Research Center, Jianghan University; Shanghai 9th People's Hospital, Shanghai Jiaotong University}
}
```

## Contact
For questions or technical support, contact:
- Corresponding Authors:
  - Qiang He: qh2020@jhun.edu.cn
- GitHub Issues: Submit questions via the [Issues](https://github.com/JHUNyizheng/Infantile-vascular-diseases-classification-based-on-ultrasound-images/issues) tab.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This research was supported by:
- Jianghan University Research Start-up Funds (Grant No. 06710001)
- Natural Science Foundation of Hubei Province (Grant No. 2025AFB150)
