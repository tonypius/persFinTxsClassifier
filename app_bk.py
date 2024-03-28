from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertConfig

app = Flask(__name__)

# Load the state_dict from the checkpoint
state_dict = torch.load('model_weights.pth')

# Modify the loaded state_dict to match the size of the current model's classifier weight matrix
# new_state_dict = {}
# for key, value in state_dict.items():
#     if 'classifier.weight' in key:
#         value = value[:33, :]  # Adjust the size to match the expected weight matrix size
#     new_state_dict[key] = value

#     if 'pre_classifier.weight' in key:
#         value = value[:, :768]  # Adjust the size to match the expected weight matrix size
#     new_state_dict[key] = value


# Initialize a new pre_classifier.weight parameter with the desired size
new_pre_classifier_weight = nn.Parameter(torch.zeros(33, 768))

# Copy the values from the checkpoint's pre_classifier.weight to the new parameter
checkpoint_pre_classifier_weight = state_dict['pre_classifier.weight']
new_pre_classifier_weight.data[:checkpoint_pre_classifier_weight.shape[0], :] = checkpoint_pre_classifier_weight

# Update the state_dict with the modified pre_classifier.weight parameter
state_dict['pre_classifier.weight'] = new_pre_classifier_weight


# Load the model configuration
# config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
# config.num_labels = 33  # Set the number of labels to 33 to match the expected weight matrix size

# Initialize the modified model
model = DistilBertForSequenceClassification(config)

# Load the modified state_dict into the model
model.load_state_dict(new_state_dict)

# Load the model weights
# model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# import torch

# # Assuming you have a tensor named `tensor_to_expand` with size [768, 768]
# tensor_to_expand = ...

# # Transpose the tensor
# transposed_tensor = tensor_to_expand.transpose(0, 1)

# # Expand the tensor to match the target size [33, 768]
# expanded_tensor = transposed_tensor.expand(33, -1).contiguous()

# # Verify the size of the expanded tensor
# print(expanded_tensor.size())  # [33, 768]






# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request has the correct Content-Type header
    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error': 'Invalid Content-Type'}), 400

    # Get the input data from the JSON request
    data = request.get_json()
    transaction_descriptions = data['transaction_descriptions']

    # Preprocess the input transaction descriptions
    inputs = tokenizer(transaction_descriptions, padding=True, truncation=True, return_tensors='pt')

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

    # Convert the predicted labels to readable format
    labels = ['label1', 'label2', 'label3']  # Replace with your actual label names

    predicted_labels = [labels[label] for label in predicted_labels]

    # Prepare the response JSON object
    response = []
    for desc, label in zip(transaction_descriptions, predicted_labels):
        response.append({'transaction_description': desc, 'predicted_label': label})

    # Return the JSON response
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
