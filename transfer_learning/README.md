# Unveiling Facial Attributes with Neural Networks

## Authors

Sike and Lee

## Description

Facial attribute recognition is a significant deep learning challenge with extensive applications in human-computer interaction, security, and research. In this project, we build and train neural networks using the classic CelebA dataset to perform facial attribute recognition. This module employs transfer learning from the ImageNet model, and the results can be visualized using a Gradio script.

<img src="" alt="Gradio file execution of file"/>

## How to Run the Code

### Prerequisites

- Python version: `Python 3.12.3`

### Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

### Execution

To run the model training and save it to the current working directory, use the notebook provided. Change the directory filename to point to the
files in your directory.

To launch the Gradio interface and visualize the model predictions:

```bash
python3 gradio_script.py model_name.h5
```

Ensure that model_name.h5 is replaced with the actual name of your trained model file.

### Additional Notes:

- Ensure that all paths and filenames are correct in the commands.
- Replace `<model_name.h5>` with the actual path to your model file when running the Gradio script.
