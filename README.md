# Financial Transaction Classification using DistilBERT

This project aims to classify financial transaction descriptions into predefined categories using the DistilBERT model. It provides an API endpoint that accepts transaction descriptions in JSON format and returns the predicted labels for each transaction.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [API Endpoint](#api-endpoint)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In financial institutions, there is a need to automatically categorize transaction descriptions to facilitate various tasks such as expense tracking, fraud detection, and financial reporting. This project addresses this problem by utilizing the power of the DistilBERT model, a state-of-the-art transformer-based model for natural language processing tasks.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/financial-transaction-classification.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the pre-trained DistilBERT model and place it in the project directory.

## Usage
To run the project, execute the following command:
```
python app.py
```
This will start the Flask server on `localhost:5000`.

## Training
To train the model on your own data and achieve better results, follow these steps:

1. Prepare your training data:
   - Create a CSV file with two columns: "transaction_description" and "label".
   - In the "transaction_description" column, provide the text descriptions of your financial transactions.
   - In the "label" column, provide the corresponding labels for each transaction description.
2. Preprocess the data:
   - Clean the text descriptions by removing special characters, stop words, and performing other necessary preprocessing steps.
   - Split the data into training and validation sets.
3. Fine-tune the DistilBERT model:
   - Load the pre-trained DistilBERT model using the `DistilBertForSequenceClassification.from_pretrained()` function.
   - Replace the classifier layer with a new layer that matches the number of classes in your dataset.
   - Train the model using the training data and evaluate its performance on the validation data.
4. Save the trained model weights:
   - Save the trained model weights using the `torch.save()` function.
   - Make sure to include this file (`model_weights.pth`) in your project directory.

## API Endpoint
The project provides a RESTful API endpoint that accepts POST requests at `/predict`. The request should contain a JSON object with the following structure:

```json
{
  "transaction_descriptions": [
    "Transaction 1 description",
    "Transaction 2 description",
    ...
  ]
}
```

The API will return a JSON response with the predicted labels for each transaction description:

```json
[
  {
    "transaction_description": "Transaction 1 description",
    "predicted_label": "Label 1"
  },
  {
    "transaction_description": "Transaction 2 description",
    "predicted_label": "Label 2"
  },
  ...
]
```

## Results
The project achieves an accuracy of approximately 85% in classifying financial transaction descriptions into predefined categories. However, by training the model on your own data, you can potentially achieve even better results that are tailored to your specific domain and dataset.

## Contributing
Contributions to this project are welcome! If you have any suggestions, bug reports, or feature
