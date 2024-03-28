from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=33)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the model weights
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()

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
    # labels = ['label1', 'label2', 'label3']  # Replace with your actual label names
    labels = [
        'Transfer-In',
        'Salary',
        'Cash & ATM',
        'Reimbursement',
        'Transfer-Out',
        'Business Expenses',
        'Gift',
        'Loan Friends & Family',
        'Investment',
        'Interest Income',
        'Non-categorised Expense',
        'Non-categorised Income',
        'Air Travel',
        'Bonus',
        'Business Income',
        'Electronics & Software',
        'Tax Returns',
        'Fast Food',
        'Alcohol',
        'Clothing',
        'Service & Auto Parts',
        'Utilities',
        'Auto Insurance',
        'Groceries',
        'Books & Supplies',
        'Restaurants',
        'Rental Car & Taxi',
        'Returned Purchase',
        'Bank Fee',
        'Pharmacy',
        'Family/Friends Income',
        'Parking',
        'Dentist'
    ]

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
