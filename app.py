import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Load model ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "desicion_tree.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ── Label encoding maps (alphabetical — matches sklearn LabelEncoder default) ──
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}

STATE_MAP = {
    "Andhra Pradesh": 0, "Bihar": 1, "Delhi": 2, "Gujarat": 3,
    "Karnataka": 4, "Kerala": 5, "Madhya Pradesh": 6, "Maharashtra": 7,
    "Punjab": 8, "Rajasthan": 9, "Tamil Nadu": 10, "Telangana": 11,
    "Uttar Pradesh": 12, "West Bengal": 13,
}

CITY_MAP = {
    "Ahmedabad": 0, "Bangalore": 1, "Chennai": 2, "Delhi": 3,
    "Hyderabad": 4, "Jaipur": 5, "Kolkata": 6, "Lucknow": 7,
    "Mumbai": 8, "Pune": 9, "Surat": 10,
}

CANCER_TYPE_MAP = {
    "Bladder": 0, "Brain": 1, "Breast": 2, "Cervical": 3,
    "Colon": 4, "Kidney": 5, "Leukemia": 6, "Liver": 7,
    "Lung": 8, "Ovarian": 9, "Pancreatic": 10, "Prostate": 11,
    "Skin": 12, "Stomach": 13, "Thyroid": 14,
}

STAGE_MAP = {"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3}

TREATMENT_MAP = {
    "Chemotherapy": 0, "Combined": 1, "Hormone Therapy": 2,
    "Immunotherapy": 3, "Radiation": 4, "Surgery": 5, "Targeted Therapy": 6,
}

def encode_fuzzy(value: str, mapping: dict, field_name: str) -> int:
    """Case-insensitive lookup with helpful error on miss."""
    # exact match first
    if value in mapping:
        return mapping[value]
    # case-insensitive match
    lower = {k.lower(): v for k, v in mapping.items()}
    if value.lower() in lower:
        return lower[value.lower()]
    # partial match
    for k, v in lower.items():
        if value.lower() in k or k in value.lower():
            return v
    raise ValueError(
        f"Unknown {field_name}: '{value}'. Valid options: {list(mapping.keys())}"
    )

app = FastAPI(title="Cancer Survival Predictor", version="2.0.0")

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Cancer Survival Predictor</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#0f1117;--surface:#1a1d27;--card:#21253a;--border:#2e3350;
      --accent:#6c63ff;--accent2:#a78bfa;--green:#22c55e;--red:#ef4444;
      --text:#e2e8f0;--muted:#94a3b8;--radius:14px;
    }
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
      min-height:100vh;display:flex;flex-direction:column;align-items:center;
      padding:40px 16px 80px}
    .header{text-align:center;margin-bottom:48px}
    .badge{display:inline-flex;align-items:center;gap:6px;
      background:rgba(108,99,255,.15);border:1px solid rgba(108,99,255,.35);
      color:var(--accent2);padding:4px 14px;border-radius:999px;font-size:.75rem;
      font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:18px}
    h1{font-size:2.4rem;font-weight:700;
      background:linear-gradient(135deg,#fff 30%,var(--accent2));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2}
    .header p{margin-top:12px;color:var(--muted);font-size:.95rem;
      max-width:480px;margin-inline:auto}
    .card{background:var(--card);border:1px solid var(--border);
      border-radius:var(--radius);padding:36px;width:100%;max-width:680px;
      box-shadow:0 24px 60px rgba(0,0,0,.4)}
    .section-label{font-size:.7rem;font-weight:700;letter-spacing:.1em;
      text-transform:uppercase;color:var(--accent2);margin-bottom:18px;
      padding-bottom:8px;border-bottom:1px solid var(--border)}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    .field{display:flex;flex-direction:column;gap:6px}
    .field label{font-size:.82rem;font-weight:500;color:var(--muted)}
    input,select{background:var(--surface);border:1px solid var(--border);
      border-radius:8px;color:var(--text);font-family:inherit;font-size:.9rem;
      padding:10px 14px;transition:border-color .2s,box-shadow .2s;
      outline:none;width:100%;-webkit-appearance:none;appearance:none}
    input:focus,select:focus{border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(108,99,255,.2)}
    select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2394a3b8' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
      background-repeat:no-repeat;background-position:right 12px center;padding-right:36px}
    .divider{height:1px;background:var(--border);margin:28px 0}
    .btn{width:100%;padding:14px;
      background:linear-gradient(135deg,var(--accent),#8b5cf6);
      color:#fff;font-family:inherit;font-size:1rem;font-weight:600;
      border:none;border-radius:10px;cursor:pointer;margin-top:28px;
      letter-spacing:.02em;transition:opacity .2s,transform .15s,box-shadow .2s;
      box-shadow:0 8px 24px rgba(108,99,255,.35)}
    .btn:hover{opacity:.9;transform:translateY(-1px);
      box-shadow:0 12px 32px rgba(108,99,255,.45)}
    .btn:active{transform:translateY(0)}
    .btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
    #result{margin-top:24px}
    .result-box{border-radius:var(--radius);padding:28px;
      display:flex;align-items:center;gap:20px;animation:fadeUp .35s ease}
    @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    .result-box.alive{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3)}
    .result-box.deceased{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3)}
    .result-icon{font-size:2.6rem;line-height:1;flex-shrink:0}
    .result-label{font-size:.78rem;font-weight:600;letter-spacing:.1em;
      text-transform:uppercase;opacity:.7;margin-bottom:4px}
    .result-value{font-size:1.55rem;font-weight:700}
    .alive .result-value{color:var(--green)}
    .deceased .result-value{color:var(--red)}
    .prob-bar-wrap{margin-top:16px}
    .prob-label{display:flex;justify-content:space-between;
      font-size:.8rem;color:var(--muted);margin-bottom:6px}
    .prob-bar{height:8px;border-radius:999px;background:var(--border);overflow:hidden}
    .prob-bar-fill{height:100%;border-radius:999px;transition:width .6s ease}
    .alive-fill{background:linear-gradient(90deg,#16a34a,var(--green))}
    .deceased-fill{background:linear-gradient(90deg,#b91c1c,var(--red))}
    .error-box{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
      border-radius:10px;padding:16px 20px;color:#fca5a5;font-size:.88rem;
      animation:fadeUp .3s ease}
    .footer{margin-top:40px;font-size:.78rem;color:var(--muted);text-align:center}
    .footer span{color:var(--accent2);font-weight:500}
    @media(max-width:520px){.grid-2{grid-template-columns:1fr}.card{padding:24px}}
  </style>
</head>
<body>
<div class="header">
  <div class="badge">🧬 AI Powered</div>
  <h1>Cancer Survival<br/>Predictor</h1>
  <p>Enter patient details below to get an instant survival prediction powered by a Decision Tree model.</p>
</div>

<div class="card">
  <div class="section-label">Patient Information</div>
  <div class="grid-2">
    <div class="field">
      <label>Age</label>
      <input type="number" id="age" placeholder="e.g. 45" min="0" max="120"/>
    </div>
    <div class="field">
      <label>Gender</label>
      <select id="gender">
        <option value="">Select gender</option>
        <option>Male</option>
        <option>Female</option>
        <option>Other</option>
      </select>
    </div>
    <div class="field">
      <label>State</label>
      <select id="state">
        <option value="">Select state</option>
        <option>Andhra Pradesh</option>
        <option>Bihar</option>
        <option>Delhi</option>
        <option>Gujarat</option>
        <option>Karnataka</option>
        <option>Kerala</option>
        <option>Madhya Pradesh</option>
        <option>Maharashtra</option>
        <option>Punjab</option>
        <option>Rajasthan</option>
        <option>Tamil Nadu</option>
        <option>Telangana</option>
        <option>Uttar Pradesh</option>
        <option>West Bengal</option>
      </select>
    </div>
    <div class="field">
      <label>City</label>
      <select id="city">
        <option value="">Select city</option>
        <option>Ahmedabad</option>
        <option>Bangalore</option>
        <option>Chennai</option>
        <option>Delhi</option>
        <option>Hyderabad</option>
        <option>Jaipur</option>
        <option>Kolkata</option>
        <option>Lucknow</option>
        <option>Mumbai</option>
        <option>Pune</option>
        <option>Surat</option>
      </select>
    </div>
  </div>

  <div class="divider"></div>

  <div class="section-label">Clinical Details</div>
  <div class="grid-2">
    <div class="field">
      <label>Cancer Type</label>
      <select id="cancer_type">
        <option value="">Select cancer type</option>
        <option>Bladder</option>
        <option>Brain</option>
        <option>Breast</option>
        <option>Cervical</option>
        <option>Colon</option>
        <option>Kidney</option>
        <option>Leukemia</option>
        <option>Liver</option>
        <option>Lung</option>
        <option>Ovarian</option>
        <option>Pancreatic</option>
        <option>Prostate</option>
        <option>Skin</option>
        <option>Stomach</option>
        <option>Thyroid</option>
      </select>
    </div>
    <div class="field">
      <label>Stage</label>
      <select id="stage">
        <option value="">Select stage</option>
        <option>Stage I</option>
        <option>Stage II</option>
        <option>Stage III</option>
        <option>Stage IV</option>
      </select>
    </div>
    <div class="field">
      <label>Treatment Type</label>
      <select id="treatment">
        <option value="">Select treatment</option>
        <option>Chemotherapy</option>
        <option>Combined</option>
        <option>Hormone Therapy</option>
        <option>Immunotherapy</option>
        <option>Radiation</option>
        <option>Surgery</option>
        <option>Targeted Therapy</option>
      </select>
    </div>
    <div class="field">
      <label>Survival Months</label>
      <input type="number" id="survival_months" placeholder="e.g. 24" min="0"/>
    </div>
  </div>

  <button class="btn" id="predictBtn" onclick="predict()">🔍 Predict Survival</button>
  <div id="result"></div>
</div>

<div class="footer">
  Powered by <span>Decision Tree Classifier</span> &nbsp;·&nbsp; For research purposes only
</div>

<script>
async function predict() {
  const btn = document.getElementById("predictBtn");
  const resultEl = document.getElementById("result");

  const age             = document.getElementById("age").value.trim();
  const gender          = document.getElementById("gender").value;
  const state           = document.getElementById("state").value;
  const city            = document.getElementById("city").value;
  const cancer_type     = document.getElementById("cancer_type").value;
  const stage           = document.getElementById("stage").value;
  const treatment_type  = document.getElementById("treatment").value;
  const survival_months = document.getElementById("survival_months").value.trim();

  if (!age||!gender||!state||!city||!cancer_type||!stage||!treatment_type||!survival_months) {
    resultEl.innerHTML = '<div class="error-box">⚠️ Please fill in all fields before predicting.</div>';
    return;
  }

  btn.disabled = true;
  btn.textContent = "Analyzing…";
  resultEl.innerHTML = "";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        Age: parseFloat(age),
        Gender: gender,
        State: state,
        City: city,
        Cancer_Type: cancer_type,
        Stage: stage,
        Treatment_Type: treatment_type,
        Survival_Months: parseFloat(survival_months)
      })
    });

    const data = await res.json();

    if (!res.ok) {
      resultEl.innerHTML = `<div class="error-box">❌ ${data.detail || "Prediction failed."}</div>`;
      return;
    }

    const isAlive   = data.prediction === "Alive";
    const cls       = isAlive ? "alive" : "deceased";
    const icon      = isAlive ? "💚" : "🔴";
    const aliveProb = Math.round((data.probability_alive ?? (isAlive ? 1 : 0)) * 100);
    const decProb   = 100 - aliveProb;

    resultEl.innerHTML = `
      <div class="result-box ${cls}">
        <div class="result-icon">${icon}</div>
        <div style="flex:1">
          <div class="result-label">Prediction Result</div>
          <div class="result-value">${data.prediction}</div>
          <div class="prob-bar-wrap">
            <div class="prob-label">
              <span>🟢 Alive &nbsp;${aliveProb}%</span>
              <span>🔴 Deceased &nbsp;${decProb}%</span>
            </div>
            <div class="prob-bar">
              <div class="prob-bar-fill ${isAlive ? "alive-fill" : "deceased-fill"}"
                   style="width:${isAlive ? aliveProb : decProb}%"></div>
            </div>
          </div>
        </div>
      </div>`;
  } catch (err) {
    resultEl.innerHTML = `<div class="error-box">❌ Network error: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 Predict Survival";
  }
}
</script>
</body>
</html>"""


# ── Schema ─────────────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    Age: float
    Gender: str
    State: str
    City: str
    Cancer_Type: str
    Stage: str
    Treatment_Type: str
    Survival_Months: float


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(data: PatientData):
    try:
        gender       = encode_fuzzy(data.Gender,        GENDER_MAP,       "Gender")
        state        = encode_fuzzy(data.State,         STATE_MAP,        "State")
        city         = encode_fuzzy(data.City,          CITY_MAP,         "City")
        cancer_type  = encode_fuzzy(data.Cancer_Type,   CANCER_TYPE_MAP,  "Cancer_Type")
        stage        = encode_fuzzy(data.Stage,         STAGE_MAP,        "Stage")
        treatment    = encode_fuzzy(data.Treatment_Type, TREATMENT_MAP,   "Treatment_Type")

        features = [data.Age, gender, state, city, cancer_type, stage, treatment, data.Survival_Months]
        X = np.array(features, dtype=float).reshape(1, -1)

        prediction = model.predict(X)[0]

        probability_alive    = None
        probability_deceased = None
        if hasattr(model, "predict_proba"):
            proba   = model.predict_proba(X)[0]
            classes = list(model.classes_)
            if "Alive" in classes:
                probability_alive    = float(proba[classes.index("Alive")])
            if "Deceased" in classes:
                probability_deceased = float(proba[classes.index("Deceased")])

        return {
            "prediction":           str(prediction),
            "probability_alive":    probability_alive,
            "probability_deceased": probability_deceased,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
