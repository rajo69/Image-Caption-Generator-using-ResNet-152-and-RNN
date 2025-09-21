# Image Caption Generator using ResNet-152 and RNN

This project implements an image captioning model that automatically generates descriptive captions for images. It leverages a deep learning architecture combining a pre-trained Convolutional Neural Network (CNN) as an encoder and a Recurrent Neural Network (RNN) as a decoder.

<img width="659" height="461" alt="image" src="https://github.com/user-attachments/assets/7aab6fa5-dfb4-43f4-8207-633888309b74" />


---

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## Project Overview

The goal of this project is to create a model that can "see" an image and generate a relevant, human-like description. This is achieved by:

1.  **Image Feature Extraction:** Using a pre-trained ResNet-152 model to extract a rich, high-level feature vector from the input image.
2.  **Caption Generation:** Feeding this feature vector into an RNN-based decoder that generates the caption word by word.

The model was trained on a subset of the Microsoft COCO (Common Objects in Context) dataset and evaluated using the BLEU (Bilingual Evaluation Understudy) score and Cosine Similarity to measure the quality of the generated captions against human-written references.

---

## Model Architecture

The model follows a standard Encoder-Decoder framework, which is common for sequence-to-sequence tasks.

### 1. CNN Encoder
- **Model:** Pre-trained **ResNet-152** on the ImageNet dataset.
- **Function:** The CNN acts as the "eye" of the model. We remove the final fully connected (classification) layer of the ResNet-152. The output from the preceding layer is a 2048-dimensional feature vector that serves as a rich numerical representation of the image's content. This vector is then fed as the initial input to the decoder.

### 2. RNN Decoder
- **Model:** A simple RNN with an embedding layer, an RNN layer, and a final linear layer.
- **Function:** The RNN acts as the "language model."
    - It takes the image feature vector from the encoder as its initial hidden state.
    - An embedding layer converts the word tokens of the caption into dense vectors.
    - The RNN layer processes the sequence of word embeddings to generate the next word in the caption.
    - A final linear layer with a softmax activation function outputs a probability distribution over the entire vocabulary, and the word with the highest probability is chosen.
    - This process repeats until an `<end>` token is generated or the maximum caption length is reached.

---

## Dataset

- **Dataset:** [Microsoft COCO (Common Objects in Context)](https://cocodataset.org/#home) 2017 training/validation set.
- **Details:** The COCO dataset is a large-scale object detection, segmentation, and captioning dataset. For this project, a subset was used, containing thousands of images, each with at least five reference captions.
- **Preprocessing:**
    - **Images:** Resized to 224x224 pixels and normalized using ImageNet's mean and standard deviation.
    - **Captions:** Text was converted to lowercase, punctuation was removed, and a vocabulary was built from words appearing more than three times. Special tokens like `<pad>`, `<start>`, `<end>`, and `<unk>` were added.

---

## Results

The model's performance was evaluated on a held-out test set. The training and validation loss curves show that the model begins to overfit around 20-25 epochs, which was chosen as the optimal training duration.

#### Training & Validation Loss (25 Epochs)
<img width="572" height="455" alt="image" src="https://github.com/user-attachments/assets/d6cd1fc2-2b6b-4292-a98f-6349ab9e4c6c" />


#### Performance Metrics
- **Average BLEU-1 Score:** Achieved an average score of approximately **0.6**, indicating a strong unigram overlap between the generated and reference captions.
- **Average Cosine Similarity:** The model also showed high cosine similarity between the embedding vectors of the generated and reference captions, signifying semantic closeness.

#### Example Predictions

**Example 1: High BLEU Score**

| Image | Generated Caption | Reference Captions |
| :---: | :--- | :--- |
| <img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/72b58e63-9008-4a59-acc5-bc4f8fcff253" /> | `a herd of sheep grazing in a grassy field` | - a herd of sheep are standing in a field <br> - a flock of sheep grazing in a green pasture <br> - a large flock of sheep in a large grassy field |

**Example 2: Low BLEU Score**

| Image | Generated Caption | Reference Captions |
| :---: | :--- | :--- |
| ![Bad Example](https://i.imgur.com/bad_example.png) | `a man is holding a baseball bat` | - a baseball player swinging a bat at a ball <br> - a man in a baseball uniform swinging a bat <br> - a batter prepares to hit the ball at a game |

*(Replace with your actual result images and captions)*

---

## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/image-caption-generator.git
    cd image-caption-generator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file from your notebook environment)*

4.  **Download the Dataset:**
    - Download the COCO 2017 dataset (`train2017.zip`, `val2017.zip`, `annotations_trainval2017.zip`).
    - Place the data in a directory structure as expected by the notebook.

---

## Usage

The Jupyter Notebook `notebook (1).ipynb` contains all the code for data preprocessing, training, evaluation, and inference.

To generate a caption for a new image:
1.  Place your test images in a folder (e.g., `./test_images/`).
2.  Load the trained model weights (`epochs25hidden512adam.pt`).
3.  Use the `generate_captions_for_folder` function defined in the notebook to see the results.

```python
# Example snippet from the notebook
from PIL import Image

# (Ensure encoder, decoder, and vocab are loaded)

folder_path = "./test_images"
generate_captions_for_folder(folder_path, encoder, decoder, vocab)
```

---

## Technologies Used

- **Python 3.x**
- **PyTorch:** For building and training the neural network.
- **Torchvision:** For pre-trained models and image transformations.
- **Pandas:** For data manipulation and handling annotations.
- **NLTK:** For calculating the BLEU score.
- **Matplotlib:** For visualizing images and results.
- **Jupyter Notebook:** For code development and experimentation.

---

## Future Improvements

- **Use a More Advanced Decoder:** Replace the simple RNN with an **LSTM** or **GRU** to better handle long-term dependencies in sequences.
- **Implement Attention Mechanism:** An attention mechanism would allow the decoder to focus on specific parts of the image when generating each word, which can significantly improve caption quality.
- **Use Beam Search:** Instead of greedily picking the most likely next word, beam search explores multiple possible captions at each step and chooses the one with the highest overall probability.
- **Experiment with Different Encoders:** Try more modern CNN architectures like EfficientNet or Vision Transformers (ViT) as the feature extractor.
- **Hyperparameter Tuning:** Systematically tune hyperparameters like learning rate, embedding size, and hidden layer dimensions.
