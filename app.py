from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Adjust this if rf_bonus_pipeline.joblib is elsewhere
PIPELINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rf_bonus_pipeline.joblib"))

try:
    pipeline = joblib.load(PIPELINE_PATH)
    print("Loaded pipeline:", PIPELINE_PATH)
except Exception as e:
    pipeline = None
    print("Warning: could not load pipeline:", e)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    services = []
    # Try to get service categories from pipeline preprocessor
    try:
        if pipeline is not None:
            pre = pipeline.named_steps['pre']
            if hasattr(pre, 'named_transformers_') and 'cat' in pre.named_transformers_:
                enc = pre.named_transformers_['cat']
                if hasattr(enc, 'categories_'):
                    services = list(enc.categories_[0])
    except Exception:
        services = []

    if request.method == "POST":
        try:
            LengthOfStay = float(request.form.get("LengthOfStay", 0))
            Service = request.form.get("Service", "")
            BedsAvailable = float(request.form.get("BedsAvailable", 0))
            PatientDemand = float(request.form.get("PatientDemand", 0))
            StaffAssigned = float(request.form.get("StaffAssigned", 0))

            df = pd.DataFrame([{
                "LengthOfStay": LengthOfStay,
                "Service": Service,
                "BedsAvailable": BedsAvailable,
                "PatientDemand": PatientDemand,
                "StaffAssigned": StaffAssigned
            }])

            if pipeline is None:
                error = "Model pipeline not loaded. Place rf_bonus_pipeline.joblib in project root."
            else:
                pred = pipeline.predict(df)[0]
                prediction = float(pred)
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error, services=services)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
