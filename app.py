import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# Input converter
from input_converter import convert_user_input
app = Flask(__name__)

# Load trained model & scaler
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

# Load dataset for real statistics
try:
    housing_data = pd.read_csv("HousingData.csv")
    # MEDV is the target (price in $1000s)
    prices = housing_data['MEDV'].dropna()
    DATASET_MIN_PRICE = float(prices.min())   # ~5
    DATASET_AVG_PRICE = float(prices.mean())  # ~22.5
    DATASET_MAX_PRICE = float(prices.max())   # ~50
    print(f"Dataset loaded: Min=${DATASET_MIN_PRICE}k, Avg=${DATASET_AVG_PRICE:.1f}k, Max=${DATASET_MAX_PRICE}k")
except Exception as e:
    print(f"Warning: Could not load dataset stats: {e}")
    DATASET_MIN_PRICE = 5.0
    DATASET_AVG_PRICE = 22.5
    DATASET_MAX_PRICE = 50.0

# ---------- Helper Functions ----------

def calculate_livability_score(rooms, crime, area_quality, location):
    """Calculate livability score (0-100) based on inputs."""
    score = 50
    score += min(float(rooms) * 5, 25)
    
    if crime == "low":
        score += 20
    elif crime == "medium":
        score += 10
    
    if area_quality == "premium":
        score += 15
    elif area_quality == "average":
        score += 8
    
    if location == "suburban":
        score += 10
    elif location == "urban":
        score += 5
    
    return min(100, max(0, int(score)))

def get_livability_label(score):
    """Get descriptive label for livability score."""
    if score >= 80:
        return "Excellent living conditions"
    elif score >= 60:
        return "Good living conditions"
    elif score >= 40:
        return "Average living conditions"
    else:
        return "Below average conditions"

def get_quality_label(area_quality):
    """Convert area quality to display label."""
    labels = {
        "premium": "Premium Zone",
        "average": "Standard Zone",
        "developing": "Developing Area"
    }
    return labels.get(area_quality, "Unknown")

def get_location_label(location):
    """Convert location type to display label."""
    labels = {
        "urban": "Urban Area",
        "suburban": "Suburban",
        "rural": "Rural Area"
    }
    return labels.get(location, "Unknown")

def calculate_price_percent(prediction):
    """Calculate percentage for chart bar height using real dataset stats."""
    # Ensure prediction is positive
    percent = ((prediction - DATASET_MIN_PRICE) / (DATASET_MAX_PRICE - DATASET_MIN_PRICE)) * 100
    return min(100, max(5, int(percent)))
    
def generate_trend_data(prediction):
    """Generate 6-month price trend data based on prediction."""
    prediction = abs(prediction)
    # Simulate past 5 months with slight variations leading up to current price
    base_prices = []
    start_price = prediction * 0.85  # Start 15% lower 6 months ago
    
    for i in range(6):
        if i < 5:
            # Past months: gradual increase with some variation
            progress = i / 5
            variation = np.random.uniform(-0.02, 0.03)
            price = start_price + (prediction - start_price) * progress + prediction * variation
        else:
            # Current month: actual prediction
            price = prediction
        base_prices.append(round(price, 2))
    
    # Calculate trend percentage
    trend_percent = ((base_prices[-1] - base_prices[0]) / base_prices[0]) * 100
    
    return base_prices, round(trend_percent, 1)

# ---------- Home Page ----------

@app.route('/')
def home():
    return render_template('home.html')

# ---------- API Prediction (JSON) ----------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json
        rooms = float(data["rooms"])
        location = data["locationType"]
        crime = data["crimeLevel"]
        property_age = data["propertyAge"]
        area_quality = data["areaQuality"]
        converted_df = convert_user_input(
            rooms, location, crime,
            property_age, area_quality
        )
        scaled_data = scaler.transform(converted_df)
        prediction = regmodel.predict(scaled_data)[0]
        # Ensure positive price
        prediction = abs(prediction)

        return jsonify({
            "prediction": round(float(prediction), 2),
            "min_price": DATASET_MIN_PRICE,
            "avg_price": round(DATASET_AVG_PRICE, 2),
            "max_price": DATASET_MAX_PRICE
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------- Website Form Prediction ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        rooms = float(request.form['rooms'])
        location = request.form['locationType']
        crime = request.form['crimeLevel']
        property_age = request.form['propertyAge']
        area_quality = request.form['areaQuality']
        converted_df = convert_user_input(
            rooms, location, crime,
            property_age, area_quality
        )
        scaled_data = scaler.transform(converted_df)
        prediction = regmodel.predict(scaled_data)[0]
        # IMPORTANT: Ensure positive price (model may return negative for edge cases)
        prediction = abs(prediction)

        # Calculate additional metrics for the dashboard
        livability_score = calculate_livability_score(rooms, crime, area_quality, location)
        livability_label = get_livability_label(livability_score)
        quality_label = get_quality_label(area_quality)
        location_label = get_location_label(location)
        price_percent = calculate_price_percent(prediction)
        
        # Generate trend data
        trend_prices, trend_percent = generate_trend_data(prediction)

        # Format price in thousands (e.g., 35.08 -> $35,080)
        prediction_text = f"${round(prediction * 1000, 0):,.0f}"    
        return render_template(
            "home.html",
            prediction_text=prediction_text,
            prediction_value=round(prediction, 2),
            livability_score=livability_score,
            livability_label=livability_label,
            quality_label=quality_label,
            location_type=location_label,
            price_percent=price_percent,
            # Real dataset statistics
            min_price=round(DATASET_MIN_PRICE, 1),
            avg_price=round(DATASET_AVG_PRICE, 1),
            max_price=round(DATASET_MAX_PRICE, 1),
            # Trend data for Chart.js
            trend_prices=trend_prices,
            trend_percent=trend_percent
        )
    except Exception as e:
        print("Prediction Error:", e)
        return render_template(
            "home.html",
            prediction_text="Error in prediction. Check inputs."
        )

# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True)
