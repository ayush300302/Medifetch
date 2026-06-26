"""
Generates a synthetic medical PDF file using PyMuPDF to test the loader and chunker.
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF


def create_synthetic_pdf(output_path: str) -> None:
    doc = fitz.open()

    # ── Page 1: Hypertension Guidelines ─────────────────────────
    page = doc.new_page()
    text_p1 = """CLINICAL PRACTICE GUIDELINE: HYPERTENSION MANAGEMENT
Version: 2026.1
Status: Approved

1. DIAGNOSTIC CRITERIA
Hypertension is defined as a sustained blood pressure reading of >=130/80 mmHg per the ACC/AHA guidelines. 
Readings must be confirmed across at least two separate clinical visits unless severe (e.g., >=180/120 mmHg).

2. PHARMACOLOGICAL MANAGEMENT
First-line agents for primary hypertension include:
- Thiazide Diuretics (e.g., Chlorthalidone 12.5-25 mg daily)
- Calcium Channel Blockers (e.g., Amlodipine 5-10 mg daily)
- ACE Inhibitors (e.g., Lisinopril 10-40 mg daily)
- Angiotensin Receptor Blockers (e.g., Losartan 50-100 mg daily)

Nephroprotective Preferences: In patients with concurrent diabetes mellitus or chronic kidney disease (CKD), 
ACE inhibitors or ARBs must be preferred as first-line agents to reduce renal degradation.
Avoid combining ACE inhibitors and ARBs due to risk of hyperkalemia and acute kidney injury.

3. LIFESTYLE INTERVENTIONS
All patients diagnosed with Stage 1 or Stage 2 hypertension should receive counselling on:
- Sodium Restriction: Limit sodium intake to <2,300 mg/day (ideal goal <1,500 mg/day).
- Diet: Strict adherence to the DASH (Dietary Approaches to Stop Hypertension) diet, rich in fruits, vegetables, and low-fat dairy.
- Physical Activity: Regular aerobic exercise, minimum 150 minutes per week of moderate intensity.
"""
    page.insert_text((50, 50), text_p1, fontsize=11)

    # ── Page 2: Type 2 Diabetes Guidelines ─────────────────────
    page = doc.new_page()
    text_p2 = """CLINICAL PRACTICE GUIDELINE: TYPE 2 DIABETES MANAGEMENT
Version: 2026.1
Status: Approved

1. GLYCEMIC TARGETS
For most non-pregnant adults, the recommended target is an HbA1c < 7.0% (53 mmol/mol).
Pre-prandial blood glucose should be maintained between 80-130 mg/dL, and post-prandial peak (1-2 hours after a meal) should remain <180 mg/dL.

2. PHARMACOTHERAPY PIPELINE
First-Line Therapy: Metformin (initial dose 500 mg daily, titrated to 1000 mg twice daily with meals) remains the preferred agent, 
unless contraindicated due to severe renal impairment (eGFR <30 mL/min/1.73m2).

Secondary Agent Selection based on Comorbidities:
- Atherosclerotic Cardiovascular Disease (ASCVD): Select GLP-1 Receptor Agonists (e.g., Semaglutide) or SGLT2 Inhibitors (e.g., Empagliflozin).
- Heart Failure (HF) / CKD: Select SGLT2 Inhibitors due to demonstrated cardiorenal benefits.

3. HYPOGLYCEMIA PROTOCOL
Hypoglycemia is classified as blood glucose <70 mg/dL.
Treatment (Rule of 15): Administer 15 grams of fast-acting carbohydrates (e.g., 4 oz juice), re-check glucose in 15 minutes, 
and repeat if glucose remains <70 mg/dL. Once normalized, consume a complex carbohydrate snack or meal.
"""
    page.insert_text((50, 50), text_p2, fontsize=11)

    # ── Page 3: Asthma Treatment Guidelines ────────────────────
    page = doc.new_page()
    text_p3 = """CLINICAL PRACTICE GUIDELINE: PEDIATRIC & ADULT ASTHMA
Version: 2026.1
Status: Approved

1. DIAGNOSIS AND STAGING
Asthma is characterized by variable expiratory airflow limitation and respiratory symptoms such as wheezing, shortness of breath, chest tightness, and cough. 
Diagnosis is confirmed using spirometry showing reversible bronchoconstriction (FEV1 increase of >12% and >200 mL after bronchodilator administration).

2. STEPWISE PHARMACOLOGICAL THERAPY
Step 1 & 2: As-needed low-dose inhaled corticosteroid (ICS)-formoterol is preferred as the controller and reliever therapy for mild asthma.
Step 3: Low-dose maintenance ICS + long-acting beta2-agonist (LABA) (e.g., Budesonide-Formoterol).
Step 4: Medium-dose maintenance ICS + LABA.
Step 5: High-dose maintenance ICS + LABA; consider phenotypic evaluation and add-on biologic therapies (e.g., Omalizumab).

3. EXACERBATION MANAGEMENT
Acute exacerbations present as progressive dyspnea, wheezing, and decreased peak expiratory flow (PEF).
Treatment:
- Short-acting beta2-agonists (SABA) via MDI or nebulizer (e.g., Albuterol 2.5-5 mg every 20 minutes for the first hour).
- Early administration of systemic corticosteroids (e.g., Prednisone 40-50 mg daily for 5-7 days for adults).
- Supplement oxygen to maintain target saturation of 93-95% in adults, 94-98% in children.
"""
    page.insert_text((50, 50), text_p3, fontsize=11)

    # Save and output
    doc.save(output_path)
    doc.close()
    print(f"Created synthetic PDF at: {output_path}")


if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    create_synthetic_pdf(str(out_dir / "sample_medical_guidelines.pdf"))
