from database import init_db, save_prediction, get_product_summary

# Step 1: initialise the database
init_db()
print("Database initialised successfully")

# Step 2: save some test predictions
save_prediction("ibuprofen", "This really helped my headache", "positive", 0.9812)
save_prediction("ibuprofen", "Terrible side effects, felt sick", "negative", 0.9445)
save_prediction("ibuprofen", "It was okay I guess", "neutral", 0.5123)
save_prediction("paracetamol", "Works great every time", "positive", 0.9923)
print("Test predictions saved successfully")

# Step 3: retrieve summary
print("\nSummary for ibuprofen:")
rows = get_product_summary("ibuprofen")
for sentiment, count, avg_conf in rows:
    print(f"  {sentiment}: {count} reviews, avg confidence {round(avg_conf, 4)}")

print("\nSummary for paracetamol:")
rows = get_product_summary("paracetamol")
for sentiment, count, avg_conf in rows:
    print(f"  {sentiment}: {count} reviews, avg confidence {round(avg_conf, 4)}")