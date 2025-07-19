Overview: This project implements semantic segmentation of car parts using deep learning and COCO-format polygon annotations. The pipeline leverages a pretrained U-Net backbone for multiclass pixel-wise segmentation of automotive components, with built-in data validation, class-balanced training, and color-coded visualization.

Dataset: You can find the dataset herehttps://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset/data

Features
1. COCO Polygon Annotation Parsing: Reads and processes COCO JSON files to generate segmentation masks for each car part.
2. Automatic Mask Generation: Converts all car part annotations into image masks, supporting any number of labeled parts.
3. Deep Learning-Based Segmentation: Utilizes U-Net with a pretrained ResNet encoder for robust, sample-efficient training.
4. Class Imbalance Handling: Pixel-wise class frequencies are auto-computed for loss re-weighting, improving rare part prediction.

Getting Started
1. Install dependencies: pip install segmentation-models-pytorch torch torchvision pillow tqdm matplotlib opencv-python
2. Run notebook or script:
- Generate masks: parses COCO to create PNG mask files.
- Train: launches the segmentation pipeline with your images/masks.
- Visualize: produces overlay images with legends.
3. Customize:
- Update categories from your annotation to match parts and class IDs.
- Adjust model architecture or backbone as desired.

Acknowledgments
COCO dataset format and best practices in annotation.

Pretrained model weights from open-source repositories.

Libraries: PyTorch, segmentation_models_pytorch, OpenCV, and matplotlib.

See code files and scripts for exact command-line usage and sample runs. All contributions and improvements are welcome!
