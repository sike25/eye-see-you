# Eye See You: Unveiling Facial Attributes with Convolutional Neural Networks

#### Authors: Lee Mabhena and Sike Ogieva

---

Facial attribute recognition is a significant deep learning challenge with extensive applications in human-computer interaction, security, and research. Recognizing various human features accurately is crucial for enhancing computer vision.

## Project Overview

In this project, we construct a convolutional neural network (CNN) designed for robust classification of facial attributes. We start with identifying a few key attributes such as age, presented gender, beards, facial expressions, headwear, and eyeglasses.

The dataset utilized was sourced from Kaggle's CelebA, which is well-documented, well-maintained, and broadly used. It contains 202,599 unique images, each annotated with 40 binary attributes. We have initiated our project by loading, normalizing, and splitting this data into training, validation, and testing sets with an 80-10-10 distribution.

Subsequent steps involve an iterative process of evaluation and optimization—modifying the CNN architecture and fine-tuning our hyperparameters. Our ultimate goal is to develop an accurate and adaptable model that can recognize an increasingly broad spectrum of attributes. The project is structured into three phases: preprocessing, initial CNN model, and model fine-tuning. Each phase is contained within its own directory, complete with a notebook and a README file detailing execution instructions.

## Getting Started

### Setup Instructions

1. **Download and Prepare the Dataset:**

   - Download the CelebA dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data). Ensure you place the dataset in the same local directory as the notebooks.
   - Unzip the dataset folder.

2. **Install Required Libraries:**

   - Navigate to each notebook directory.
   - Install necessary Python packages using the requirements.txt found in each directory:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Notebooks:**
   - Be aware that running a single notebook can take over 8 hours due to the extensive size of the dataset.

### Note

Each phase's directory contains specific README files that provide detailed instructions for running the corresponding notebooks. Ensure to follow each directory’s README for tailored setup and execution guidance.
