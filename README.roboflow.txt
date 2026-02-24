
PCB Defect Detection Dataset
==============================

This dataset is used for PCB (Printed Circuit Board) defect detection.

Dataset Source: PKU-Market-PCB / Roboflow Universe

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes PCB images with 6 defect types.
Defects are annotated in YOLOv8 format.

Defect Types:
1. missing_hole - Missing drill holes on PCB
2. mouse_bite - Edge defects on traces
3. open_circuit - Broken/disconnected traces (CRITICAL)
4. short - Unintended connections between traces (CRITICAL)
5. spur - Extra copper protrusions from traces
6. spurious_copper - Unwanted copper residue

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

No image augmentation techniques were applied.

