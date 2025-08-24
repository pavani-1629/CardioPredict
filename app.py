from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import pickle
import pandas as pd
from dotenv import load_dotenv
import os
import pyrebase
import google.generativeai as genai  # Gemini API
import pyrebase

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

# Firebase config
from firebase.firebase_config import firebaseConfig
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Gemini API Config
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Label mapping
label_mapping = {
    0: "No Disease",
    1: "Angina",
    2: "Arrhythmia",
    3: "Heart Failure",
    4: "Myocardial Infarction",
    5: "General Heart Disease"
}

# ------------------------- ROUTES -------------------------

@app.route('/')
def main():
    return redirect(url_for('home'))

'''@app.route('/welcome')
def welcome():
    return render_template('welcome.html')'''

@app.route('/home')
def home():
    return render_template('home.html')

# ------------------------- AUTH -------------------------

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            uid = user['localId']
            # Handle case where user data might not exist
            user_data = db.child("users").child(uid).get().val()
            if user_data is None:
                user_data = {"username": "User"}  # Default username if data is missing
            session['user'] = {
                "uid": uid,
                "username": user_data.get("username", "User"),
                "email": email
            }
            flash(f"Welcome back, {session['user']['username']}!", "success")
            return redirect(url_for('home'))
        except Exception as e:
            print(f"Firebase Error: {str(e)}")  # Debug output
            return render_template("login.html", error=f"Login failed: {str(e)}")
    return render_template('login.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        username = request.form.get('username')
        try:
            user = auth.create_user_with_email_and_password(email, password)
            uid = user['localId']
            # Write user data to database
            db.child("users").child(uid).set({
                "username": username,
                "email": email
            })
            session['user'] = {
                "uid": uid,
                "username": username,
                "email": email
            }
            flash("Registration successful! Welcome!", "success")
            return redirect(url_for('home'))
        except Exception as e:
            print(f"Firebase Error: {str(e)}")  # Debug output
            return render_template("register.html", error=f"Registration failed: {str(e)}")
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

# ------------------------- PREDICTION -------------------------

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_data = {
                'age': int(request.form['age']),
                'sex': int(request.form['sex']),
                'cp': int(request.form['cp']),
                'trestbps': float(request.form['trestbps']),
                'chol': float(request.form['chol']),
                'fbs': int(request.form['fbs']),
                'restecg': int(request.form['restecg']),
                'thalach': float(request.form['thalach']),
                'exang': int(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': int(request.form['slope']),
                'ca': float(request.form['ca']),
                'thal': int(request.form['thal'])
            }

            # Predict
            pred = model.predict(pd.DataFrame([input_data]))[0]
            prediction = label_mapping.get(pred, "Unknown")

            # Save to Firebase
            if 'user' in session:
                db.child("predictions").child(session['user']['uid']).push({
                    "data": input_data,
                    "prediction": prediction
                })

            # Redirect to result page
            return redirect(url_for('result', prediction=prediction, **input_data))

        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}")

    return render_template("index.html")

@app.route("/result")
def result():
    prediction = request.args.get('prediction')
    user_data = {k: request.args.get(k) for k in request.args if k != 'prediction'}

    prompt = f"""
    The user has the following health data:
    {user_data}
    The predicted heart disease type is: {prediction}.
    Explain ONLY the possible medical reason for this prediction based on the data with some emojis for better understanding.
    """

    try:
        ai_response = model_gemini.generate_content(prompt)
        reason = ai_response.text.strip()

        return render_template(
            "result.html",
            prediction_result=prediction,
            prediction_reason=reason,
            user_data=user_data
        )
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

# ------------------------- AI TOOLS -------------------------

@app.route("/get_precautions", methods=["POST"])
def get_precautions():
    data = request.get_json()
    prediction = data.get("prediction", "")
    user_data = data.get("user_data", {})

    prompt = f"""
    The user has the following health data:
    {user_data}
    The diagnosed heart disease type is: {prediction}.
    Provide the TOP 8 most important medical precautions step by step with some emojis for better understanding.
    """

    try:
        ai_response = model_gemini.generate_content(prompt)
        return jsonify({"precautions": ai_response.text.strip()})
    except Exception as e:
        return jsonify({"precautions": f"Error: {str(e)}"})

@app.route("/generate_diet", methods=["POST"])
def generate_diet():
    data = request.get_json()
    reason = data.get("reason", "")
    health_issue = data.get("health_issue", "")

    prompt = f"""
    Based on this reason for heart disease:
    {reason}
    And considering this additional health issue: {health_issue},
    create a detailed, healthy diet plan with some emoji's to get user attraction and impression and better understanding.
    """

    try:
        ai_response = model_gemini.generate_content(prompt)
        return jsonify({"diet_plan": ai_response.text.strip()})
    except Exception as e:
        return jsonify({"diet_plan": f"Error: {str(e)}"})

# ------------------------- MISC PAGES -------------------------

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    uid = session["user"]["uid"]
    user_data = db.child("users").child(uid).get().val() or {}
    return render_template("profile.html", user=user_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/todo")
def todo():
    return render_template("todo.html")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "GET":
        return render_template("chatbot.html")
    elif request.method == "POST":
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"reply": "Please type something so I can help you."})

        prompt = f"""
        You are CardioPredict's virtual heart health assistant.
        The user says: {user_message}.
        Answer clearly and helpfully about heart health, precautions, or diet.
        """

        try:
            ai_response = model_gemini.generate_content(prompt)
            return jsonify({"reply": ai_response.text.strip()})
        except Exception as e:
            return jsonify({"reply": f"Error: {str(e)}"})


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form['email']
        try:
            auth.send_password_reset_email(email)
            flash("Password reset email sent!", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
    return render_template("forgotPassword.html")

# ------------------------- RUN -------------------------

if __name__ == "__main__":
    app.run(debug=True)
