from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/shop")
def shop():
    return render_template("shop.html")

@app.route("/checkout")
def checkout():
    return render_template("checkout.html")

@app.route("/news")
def news():
    return render_template("news.html")

@app.route("/cart")
def cart():
    return render_template("cart.html")

# Route to handle form submission and predict
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    # Create feature array
    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    # Load scaler and scale features
    scaler_path = r'C:\Users\HP\OneDrive\Documents\html\dhasjhd\big mart prediction 2\flask_project\models\sc.sav'
    sc = joblib.load(scaler_path)
    X_std = sc.transform(X)

    # Load model and predict
    model_path = r'C:\Users\HP\OneDrive\Documents\html\dhasjhd\big mart prediction 2\flask_project\models\lr.sav'
    model = joblib.load(model_path)
    prediction = model.predict(X_std)

    # Return prediction as JSON
    return jsonify({'Prediction': float(prediction)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)