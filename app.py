from flask import Flask, render_template, request, url_for, redirect
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json, os, sqlite3

app = Flask(__name__)

# Load model & labels
model = tf.keras.models.load_model("C:/Users/PRAVIN/Final Project/New folder/model.h5")
with open("C:/Users/PRAVIN/Final Project/New folder/labels.json", "r") as f:
    labels = json.load(f)

# Ensure uploads folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect("userdata.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        phone TEXT,
        filename TEXT,
        prediction TEXT,
        confidence REAL
    )
    """)
    conn.commit()
    conn.close()

init_db()

def save_to_db(name, address, phone, filename, prediction, confidence):
    conn = sqlite3.connect("userdata.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO user_uploads (name, address, phone, filename, prediction, confidence)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (name, address, phone, filename, prediction, confidence))
    conn.commit()
    conn.close()

def get_data(address_filter=None):
    conn = sqlite3.connect("userdata.db")
    cursor = conn.cursor()
    if address_filter:
        cursor.execute("SELECT * FROM user_uploads WHERE address LIKE ?", ('%' + address_filter + '%',))
    else:
        cursor.execute("SELECT * FROM user_uploads")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Prediction function
def predict_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds, axis=1)[0]
    return labels[pred_index], float(np.max(preds) * 100)

# Home Page (Upload + Prediction)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence = None, None
    img_url, name, address, phone = None, None, None, None

    if request.method == "POST":
        name = request.form["name"]
        address = request.form["address"]
        phone = request.form["phone"]
        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)
            img_url = url_for("static", filename="uploads/" + file.filename)

            # Save to DB
            save_to_db(name, address, phone, file.filename, prediction, confidence)

    return render_template("index.html", prediction=prediction, confidence=confidence,
                           img_path=img_url, name=name, address=address, phone=phone)

# Admin Dashboard
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    address_filter = None
    if request.method == "POST":
        address_filter = request.form["address_filter"]
    data = get_data(address_filter)
    return render_template("dashboard.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
