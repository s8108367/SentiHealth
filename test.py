import requests

url = "https://s8108367-sentihealth-api.hf.space/predict"

reviews = [
     "This medication completely changed my life, I feel so much better",
    "Terrible side effects, made my condition worse after two weeks",
    "It's okay I guess, nothing special but didn't cause any problems",
    "Absolutely love this supplement, been taking it for months",
    "I wouldn't recommend this to anyone, very disappointed",
]

for review in reviews:
    r = requests.post(url, json={"text": review})
    data = r.json()
    print(f"Review:     {review}")
    print(f"Sentiment:  {data['sentiment']}")
    print(f"Confidence: {data['confidence']}")
    print()