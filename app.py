from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
from database import init_db, save_prediction, get_product_summary
import torch

app = Flask(__name__)
CORS(app)

init_db()

MODEL_PATH = 's8108367/sentihealth-bert'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

LABELS = ['negative', 'neutral', 'positive']

@app.route('/', methods=['GET'])
def index():
    return jsonify({'name': 'SentiHealth API', 'status': 'running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide a text field'}), 400

    text = data['text']
    product = data.get('product', 'unknown')

    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()

    sentiment = LABELS[pred]
    confidence = round(probs[pred].item(), 4)

    save_prediction(product, text, sentiment, confidence)

    return jsonify({
        'text': text,
        'product': product,
        'sentiment': sentiment,
        'confidence': confidence,
        'scores': {
            'negative': round(probs[0].item(), 4),
            'neutral': round(probs[1].item(), 4),
            'positive': round(probs[2].item(), 4),
        }
    })

@app.route('/summary', methods=['GET'])
def summary():
    product = request.args.get('product')
    if not product:
        return jsonify({'error': 'Please provide a product name: /summary?product=ibuprofen'}), 400

    rows = get_product_summary(product)
    if not rows:
        return jsonify({'error': f'No data found for product: {product}'}), 404

    total = sum(r[1] for r in rows)
    result = {}
    for sentiment, count, avg_conf in rows:
        result[sentiment] = {
            'count': count,
            'percentage': round((count / total) * 100, 1),
            'avg_confidence': round(avg_conf, 4)
        }

    return jsonify({
        'product': product,
        'total_reviews': total,
        'summary': result
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)