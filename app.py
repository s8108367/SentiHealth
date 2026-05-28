from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load directly from Hugging Face
MODEL_PATH = 's8108367/sentihealth-bert'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

LABELS = ['negative', 'neutral', 'positive']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide a "text" field in the request body'}), 400
    text = data['text']
    inputs = tokenizer(text, return_tensors='pt', padding='max_length',
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
    return jsonify({
        'text': text,
        'sentiment': LABELS[pred],
        'confidence': round(probs[pred].item(), 4),
        'scores': {
            'negative': round(probs[0].item(), 4),
            'neutral':  round(probs[1].item(), 4),
            'positive': round(probs[2].item(), 4),
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)