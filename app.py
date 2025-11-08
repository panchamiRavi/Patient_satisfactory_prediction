import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
DATA_DIR = "C:/Users/user/Desktop/Patientsatisfctrymodel"
staff = pd.read_csv(os.path.join(DATA_DIR, "staff.csv"))
staff_schedule = pd.read_csv(os.path.join(DATA_DIR, "staff_schedule.csv"))
print("staff columns:", staff.columns.tolist())
print("staff_schedule columns:", staff_schedule.columns.tolist())



flask_dir = os.path.join(OUTPUT_DIR, "flask_app")
templates_dir = os.path.join(flask_dir, "templates")
os.makedirs(templates_dir, exist_ok=True)
app_py = r"""
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load("/mnt/data/best_pipeline.joblib")   # adjust path if needed

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # read form entries and coerce types
        LengthOfStay = float(request.form.get("LengthOfStay", 0))
        Service = request.form.get("Service", "")
        BedsAvailable = float(request.form.get("BedsAvailable", 0))
        PatientDemand = float(request.form.get("PatientDemand", 0))
        StaffAssigned = float(request.form.get("StaffAssigned", 0))
        # construct dataframe with columns expected by the pipeline
        df = pd.DataFrame([{
            "LengthOfStay": LengthOfStay,
            "BedsAvailable": BedsAvailable,
            "PatientDemand": PatientDemand,
            "StaffAssigned": StaffAssigned,
            "Service": Service
        }])
        pred = pipeline.predict(df)[0]
        prediction = round(float(pred), 2)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
"""

index_html = r"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Patient Satisfaction Predictor</title>
  </head>
  <body>
    <h1>Patient Satisfaction Predictor</h1>
    <form method="post">
      <label>LengthOfStay: <input name="LengthOfStay" type="number" step="0.1" required></label><br>
      <label>Service: <input name="Service" type="text" required></label><br>
      <label>BedsAvailable: <input name="BedsAvailable" type="number" step="1" required></label><br>
      <label>PatientDemand: <input name="PatientDemand" type="number" step="1" required></label><br>
      <label>StaffAssigned: <input name="StaffAssigned" type="number" step="1" required></label><br>
      <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
      <div>
        <strong>Predicted SatisfactionScore: </strong>{{ prediction }}
      </div>
    {% endif %}
  </body>
</html>
"""

with open(os.path.join(flask_dir, "app.py"), "w") as f:
    f.write(app_py.strip())
with open(os.path.join(templates_dir, "index.html"), "w") as f:
    f.write(index_html.strip())

print("Wrote Flask app to", flask_dir)

print("staff columns:", staff.columns.tolist())
print("staff_schedule columns:", staff_schedule.columns.tolist())

if 'StaffID' not in staff.columns:
    possible = [c for c in staff.columns if 'staff' in c.lower() and 'id' in c.lower()]
    if possible:
        staff = staff.rename(columns={possible[0]: 'StaffID'})
if 'StaffID' not in staff_schedule.columns:
    possible = [c for c in staff_schedule.columns if 'staff' in c.lower() and 'id' in c.lower()]
    if possible:
        staff_schedule = staff_schedule.rename(columns={possible[0]: 'StaffID'})
if 'Presence' not in staff_schedule.columns:
    possible = [c for c in staff_schedule.columns if 'pres' in c.lower()]
    if possible:
        staff_schedule = staff_schedule.rename(columns={possible[0]: 'Presence'})

if 'Week' not in staff_schedule.columns:
    possible = [c for c in staff_schedule.columns if 'week' in c.lower()]
    if possible:
        staff_schedule = staff_schedule.rename(columns={possible[0]: 'Week'})
        required_ok = all(col in staff.columns for col in ['StaffID', 'Service']) and all(col in staff_schedule.columns for col in ['StaffID', 'Presence', 'Week'])
if not required_ok:
    print("Could not locate required staff or schedule columns automatically. Check column names and re-run the bonus block with proper column renames.")
else:
    staff_sched = pd.merge(staff_schedule, staff[['StaffID', 'Service']], on='StaffID', how='left')
    pres_vals = ['present', 'true', '1', 'yes', 'y']
    staff_sched['PresenceFlag'] = staff_sched['Presence'].astype(str).str.lower().isin(pres_vals)

  
    present = staff_sched[staff_sched['PresenceFlag']]
    actual_staff = present.groupby(['Service', 'Week']).size().reset_index(name='ActualStaffPresent')

    print("Sample ActualStaffPresent:")
    display(actual_staff.head())


    merged2 = pd.merge(merged, actual_staff, on=['Service', 'Week'], how='left')
    merged2['ActualStaffPresent'] = merged2['ActualStaffPresent'].fillna(0)


    base_features = [f for f in ['LengthOfStay', 'BedsAvailable', 'PatientDemand'] if f in merged2.columns]
    if 'ActualStaffPresent' not in merged2.columns:
        print("ActualStaffPresent not created.")
    else:
        X_bonus = merged2[base_features + ['ActualStaffPresent', 'Service']].copy()
        y_bonus = merged2[TARGET].copy()
        maskb = X_bonus.notnull().all(axis=1) & y_bonus.notnull()
        X_bonus = X_bonus[maskb]
        y_bonus = y_bonus[maskb]
        print("Rows for bonus retraining:", X_bonus.shape[0])

      
        num_cols_bonus = [c for c in ['LengthOfStay', 'BedsAvailable', 'PatientDemand', 'ActualStaffPresent'] if c in X_bonus.columns]
        cat_cols_bonus = ['Service']
        preproc_bonus = ColumnTransformer(transformers=[
            ('num', Pipeline([('scaler', StandardScaler())]), num_cols_bonus),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))]), cat_cols_bonus)
        ], remainder='drop')

        rf_bonus = Pipeline(steps=[('preprocessor', preproc_bonus),
                                   ('model', RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1))])

        Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bonus, y_bonus, test_size=0.20, random_state=RANDOM_SEED)
        rf_bonus.fit(Xb_train, yb_train)
        yb_pred = rf_bonus.predict(Xb_test)
        r2_b = r2_score(yb_test, yb_pred)
        mae_b = mean_absolute_error(yb_test, yb_pred)

        print("Bonus RandomForest with ActualStaffPresent -- R^2: {:.4f}, MAE: {:.4f}".format(r2_b, mae_b))
        joblib.dump(rf_bonus, os.path.join(OUTPUT_DIR, "rf_bonus_pipeline.joblib"))
        print("Saved rf_bonus_pipeline.joblib")


        print("Saved files:")
for fname in ["best_pipeline.joblib", "preprocessor.joblib", "model_only.joblib", "rf_bonus_pipeline.joblib"]:
    p = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(p):
        print("-", p)
    