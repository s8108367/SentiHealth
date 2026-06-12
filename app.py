from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
from database import init_db, save_prediction, get_product_summary, get_previous_positive_percentage
from notifications import (
    init_notifications_table, add_notification,
    get_notifications, mark_as_read,
    get_recent_negative_count, get_positive_percentage,
    get_total_reviews
)
import torch

app = Flask(__name__)
CORS(app)

# Initialise database tables
init_db()
init_notifications_table()

MODEL_PATH = 's8108367/sentihealth-bert'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

LABELS = ['negative', 'neutral', 'positive']

def check_and_trigger_notifications(product, sentiment, confidence):

    total = get_total_reviews(product)

    # Trigger 1 — Sentiment shift alert (20% drop in positive percentage)
    current_pct  = get_positive_percentage(product)
    previous_pct = get_previous_positive_percentage(product)

    if current_pct is not None and previous_pct is not None:
        shift = previous_pct - current_pct
        if shift >= 20:
            add_notification(
                product,
                'sentiment_shift',
                f"Sentiment for {product} has dropped significantly. "
                f"Positive reviews went from {previous_pct}% down to {current_pct}%. "
                f"Consider reviewing recent feedback."
            )
        elif (current_pct - previous_pct) >= 20:
            add_notification(
                product,
                'sentiment_shift',
                f"Sentiment for {product} has improved significantly. "
                f"Positive reviews went from {previous_pct}% up to {current_pct}%."
            )

    # Trigger 2 — Negative spike alert (3+ negatives in last 10 reviews)
    recent_negatives = get_recent_negative_count(product, limit=10)
    if recent_negatives >= 3 and sentiment == 'negative':
        add_notification(
            product,
            'negative_spike',
            f"Warning: {recent_negatives} out of the last 10 reviews for "
            f"{product} are negative. This may indicate a concern worth investigating."
        )

    # Trigger 3 — Milestone alert (10, 50, 100 reviews)
    milestones = [10, 50, 100]
    if total in milestones:
        add_notification(
            product,
            'milestone',
            f"{product} has reached {total} reviews on SentiHealth. "
            f"Current positive rate: {current_pct}%."
        )

    # Trigger 4 — Low confidence warning
    if confidence < 0.60:
        add_notification(
            product,
            'low_confidence',
            f"A recent review for {product} was classified with low confidence "
            f"({round(confidence * 100, 1)}%). The sentiment may be ambiguous."
        )

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'SentiHealth API',
        'status': 'running',
        'endpoints': {
            'predict':        'POST /predict',
            'predict_batch':  'POST /predict_batch',
            'summary':        'GET /summary?product=ibuprofen',
            'compare':        'GET /compare?products=ibuprofen,paracetamol',
            'notifications':  'GET /notifications',
            'mark_read':      'POST /notifications/read',
            'health':         'GET /health'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide a text field'}), 400

    text    = data['text']
    product = data.get('product', 'unknown')

    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze()
        pred    = torch.argmax(probs).item()

    sentiment  = LABELS[pred]
    confidence = round(probs[pred].item(), 4)

    save_prediction(product, text, sentiment, confidence)
    check_and_trigger_notifications(product, sentiment, confidence)

    return jsonify({
        'text':       text,
        'product':    product,
        'sentiment':  sentiment,
        'confidence': confidence,
        'scores': {
            'negative': round(probs[0].item(), 4),
            'neutral':  round(probs[1].item(), 4),
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

    total  = sum(r[1] for r in rows)
    result = {}
    for sentiment, count, avg_conf in rows:
        result[sentiment] = {
            'count':          count,
            'percentage':     round((count / total) * 100, 1),
            'avg_confidence': round(avg_conf, 4)
        }

    return jsonify({
        'product':       product,
        'total_reviews': total,
        'summary':       result
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'reviews' not in data:
        return jsonify({'error': 'Please provide a "reviews" list in the request body'}), 400

    reviews = data['reviews']
    if not isinstance(reviews, list) or len(reviews) == 0:
        return jsonify({'error': '"reviews" must be a non-empty list'}), 400

    if len(reviews) > 100:
        return jsonify({'error': 'Maximum 100 reviews per batch'}), 400

    product = data.get('product', 'unknown')
    results = []

    for review in reviews:
        if not isinstance(review, str) or not review.strip():
            results.append({'text': review, 'error': 'Invalid or empty text'})
            continue

        inputs = tokenizer(review, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1).squeeze()
            pred    = torch.argmax(probs).item()

        sentiment  = LABELS[pred]
        confidence = round(probs[pred].item(), 4)

        save_prediction(product, review, sentiment, confidence)
        check_and_trigger_notifications(product, sentiment, confidence)

        results.append({
            'text':       review,
            'sentiment':  sentiment,
            'confidence': confidence,
            'scores': {
                'negative': round(probs[0].item(), 4),
                'neutral':  round(probs[1].item(), 4),
                'positive': round(probs[2].item(), 4),
            }
        })

    positive = sum(1 for r in results if r.get('sentiment') == 'positive')
    negative = sum(1 for r in results if r.get('sentiment') == 'negative')
    neutral  = sum(1 for r in results if r.get('sentiment') == 'neutral')
    total    = len(results)

    return jsonify({
        'product':  product,
        'total':    total,
        'summary': {
            'positive': {'count': positive, 'percentage': round((positive / total) * 100, 1)},
            'negative': {'count': negative, 'percentage': round((negative / total) * 100, 1)},
            'neutral':  {'count': neutral,  'percentage': round((neutral  / total) * 100, 1)},
        },
        'results': results
    })


@app.route('/compare', methods=['GET'])
def compare():
    products_param = request.args.get('products')
    if not products_param:
        return jsonify({'error': 'Please provide products: /compare?products=ibuprofen,paracetamol'}), 400

    products = [p.strip() for p in products_param.split(',')]
    if len(products) < 2:
        return jsonify({'error': 'Please provide at least 2 products to compare'}), 400

    if len(products) > 5:
        return jsonify({'error': 'Maximum 5 products per comparison'}), 400

    comparison = {}
    for product in products:
        rows = get_product_summary(product)
        if not rows:
            comparison[product] = {'error': f'No data found for {product}'}
            continue

        total    = sum(r[1] for r in rows)
        positive = sum(r[1] for r in rows if r[0] == 'positive')
        negative = sum(r[1] for r in rows if r[0] == 'negative')
        neutral  = sum(r[1] for r in rows if r[0] == 'neutral')

        breakdown = {}
        for sentiment, count, avg_conf in rows:
            breakdown[sentiment] = {
                'count':          count,
                'percentage':     round((count / total) * 100, 1),
                'avg_confidence': round(avg_conf, 4)
            }

        comparison[product] = {
            'total_reviews':       total,
            'positive_percentage': round((positive / total) * 100, 1),
            'negative_percentage': round((negative / total) * 100, 1),
            'neutral_percentage':  round((neutral  / total) * 100, 1),
            'breakdown':           breakdown
        }

    ranked = sorted(
        [p for p in comparison if 'error' not in comparison[p]],
        key=lambda p: comparison[p]['positive_percentage'],
        reverse=True
    )

    return jsonify({
        'comparison':   comparison,
        'ranking':      ranked,
        'verdict':      f'{ranked[0]} has the highest positive sentiment at {comparison[ranked[0]]["positive_percentage"]}%' if ranked else 'No data available'
    })


@app.route('/notifications', methods=['GET'])
def notifications():
    unread_only = request.args.get('unread', 'true').lower() == 'true'
    rows = get_notifications(unread_only=unread_only)
    results = []
    for row in rows:
        results.append({
            'id':         row[0],
            'product':    row[1],
            'type':       row[2],
            'message':    row[3],
            'is_read':    bool(row[4]),
            'created_at': row[5]
        })
    return jsonify({
        'unread_count': len([r for r in results if not r['is_read']]),
        'notifications': results
    })

@app.route('/notifications/read', methods=['POST'])
def mark_read():
    data = request.get_json()
    notification_id = data.get('id') if data else None
    mark_as_read(notification_id)
    return jsonify({'status': 'ok', 'message': 'Notifications marked as read'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)