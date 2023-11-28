# ID-Documents-OCRs

This repository contains the code to train four OCR models based on object detection. Specifically, object detection is performed using characters from the English alphabet, including both lowercase and uppercase letters, numbers, and the `<` symbol, resulting in a total of 63 classes. These models are designed to identify individual characters within ID documents.

In contrast to traditional OCRs that extract recognized words, these models are optimized for analyzing individual characters, providing higher precision in detecting specific details.

## Context and Usage

The models were developed for a specific purpose: antitampering in the Machine Readable Zone (MRZ) of ID documents. When analyzing the MRZ, which comprises alphanumeric codes, the primary focus is on achieving precision in the detection and analysis of individual characters within the complete MRZ code. This intentional approach aims to interpret the MRZ as a nuanced alphanumeric sequence, rather than treating it as isolated words.

The code to use these models is available in the [ID-Documents-Antitampering-MRZ](https://github.com/FrancescaPietrobon/ID-Documents-Antitampering-MRZ) repository.

## Available Models

Two object detection models have been employed: RetinaNet and YOLOv8. Both models have been trained using both color and binarized images, resulting in a total of four distinct trained models.

For each model, there is a dedicated branch in this repository containing the training code and results obtained on a custom dataset:

- [RetinaNet Branch](https://github.com/FrancescaPietrobon/ID-Documents-OCRs/tree/retinanet)
- [RetinaNet-Binary Branch](https://github.com/FrancescaPietrobon/ID-Documents-OCRs/tree/retinanet-binary)
- [YOLOv8 Branch](https://github.com/FrancescaPietrobon/ID-Documents-OCRs/tree/yolov8)
- [YOLOv8-Binary Branch](https://github.com/FrancescaPietrobon/ID-Documents-OCRs/tree/yolov8-binary)

Click on the branch links above to directly access the code and training details for each model.


## CI/CD Pipeline and Training Process

The training process utilized a CI/CD pipeline on GitLab, incorporating CML (Continuous Machine Learning) and DVC (Data Version Control), with the training data stored on AWS S3.

To initiate training, load the data into the storage location set in `.dvc/config` using DVC and complete any missing data configurations in the `.gitlab-ci.yml` file.

This repository provides only a summary of the code used and the training results.

