"""
MedIntel – AI Health Score Engine
==================================
Combines BMI + Blood Pressure + Blood Sugar + Lifestyle factors
to produce a score out of 100.

Score Interpretation
--------------------
  90–100  → Excellent  🟢
  75–89   → Good       🟡
  50–74   → Fair       🟠
  25–49   → Poor       🔴 (triggers warning)
   0–24   → Critical   🚨 (triggers Emergency Alert)
"""

def calculate_health_score(details, breakdown=False):
    """
    Parameters
    ----------
    details  : PatientDetail ORM object
    breakdown: bool – if True, return (score, dict) instead of just score

    Returns
    -------
    int  (0–100) or  (int, dict)  when breakdown=True
    """
    scores = {}

    # ── 1. BMI Score  (max 25 pts) ─────────────────────────────────────────
    bmi_score = 0
    if details.height and details.weight and details.height > 0:
        h   = details.height / 100          # cm → m
        bmi = details.weight / (h * h)
        if 18.5 <= bmi <= 24.9:
            bmi_score = 25
        elif 17.0 <= bmi < 18.5 or 25.0 <= bmi < 27.5:
            bmi_score = 18
        elif 15.0 <= bmi < 17.0 or 27.5 <= bmi < 30.0:
            bmi_score = 12
        elif 30.0 <= bmi < 35.0:
            bmi_score = 7
        else:                               # severely underweight / obese
            bmi_score = 3
    scores["bmi"] = bmi_score

    # ── 2. Blood Pressure Score  (max 25 pts) ──────────────────────────────
    bp_score = 0
    if details.bp:
        try:
            systolic, diastolic = [int(x) for x in details.bp.split("/")]
            if systolic < 120 and diastolic < 80:
                bp_score = 25          # Normal
            elif systolic < 130 and diastolic < 80:
                bp_score = 20          # Elevated
            elif systolic < 140 or diastolic < 90:
                bp_score = 14          # Stage 1 HTN
            elif systolic < 180 or diastolic < 120:
                bp_score = 7           # Stage 2 HTN
            else:
                bp_score = 2           # Hypertensive crisis
        except (ValueError, AttributeError):
            bp_score = 0
    scores["blood_pressure"] = bp_score

    # ── 3. Blood Sugar Score  (max 25 pts) ─────────────────────────────────
    sugar_score = 0
    if details.sugar:
        s = details.sugar
        if 70 <= s <= 99:
            sugar_score = 25    # Normal fasting
        elif 100 <= s <= 125:
            sugar_score = 15    # Prediabetic range
        elif 126 <= s <= 180:
            sugar_score = 8     # Diabetic range
        elif s > 180:
            sugar_score = 3     # Very high
        else:
            sugar_score = 10    # Low (hypoglycaemic risk)
    scores["blood_sugar"] = sugar_score

    # ── 4. Lifestyle Score  (max 25 pts) ───────────────────────────────────
    lifestyle_score = 25

    # Activity level
    activity = (details.activity or "moderate").lower()
    if activity == "high":
        lifestyle_score -= 0
    elif activity == "moderate":
        lifestyle_score -= 5
    elif activity == "low":
        lifestyle_score -= 12

    # Smoking
    smoking = (details.smoking or "no").lower()
    if smoking == "yes":
        lifestyle_score -= 10

    # Age penalty (mild, older age increases risk)
    if details.age:
        if details.age > 60:
            lifestyle_score -= 3
        elif details.age > 45:
            lifestyle_score -= 1

    lifestyle_score = max(lifestyle_score, 0)
    scores["lifestyle"] = lifestyle_score

    # ── Final Score ─────────────────────────────────────────────────────────
    total = sum(scores.values())
    total = max(0, min(100, total))     # clamp to [0, 100]

    if breakdown:
        label, emoji = _interpret(total)
        scores["total"]  = total
        scores["label"]  = label
        scores["emoji"]  = emoji
        scores["alert"]  = total < 50   # triggers Emergency Alert
        return total, scores

    return total


def _interpret(score):
    if score >= 90: return "Excellent", "🟢"
    if score >= 75: return "Good",      "🟡"
    if score >= 50: return "Fair",      "🟠"
    if score >= 25: return "Poor",      "🔴"
    return "Critical", "🚨"


def health_advice(score, breakdown):
    """Return personalised advice strings based on breakdown dict."""
    advice = []
    if breakdown.get("bmi", 25) < 18:
        advice.append("Your BMI is below healthy range — consider increasing caloric intake.")
    if breakdown.get("blood_pressure", 25) < 14:
        advice.append("Your blood pressure is elevated — reduce sodium and stress.")
    if breakdown.get("blood_sugar", 25) < 15:
        advice.append("Blood sugar is high — monitor carbohydrate intake and exercise regularly.")
    if breakdown.get("lifestyle", 25) < 10:
        advice.append("Lifestyle factors are hurting your score — quit smoking and be more active.")
    if not advice:
        advice.append("Great job! Keep maintaining your healthy habits.")
    return advice
