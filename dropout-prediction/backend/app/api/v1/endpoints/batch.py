from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
import pandas as pd
import io
from ....models.ml_model import predictor
from ....utils.feature_engineering import FeatureEngineer

router = APIRouter()
feature_engineer = FeatureEngineer()


def _read_dataframe(upload: UploadFile) -> pd.DataFrame:
    filename = upload.filename or "uploaded"
    content = upload.file.read()
    try:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        elif filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        else:
            # try csv by default
            return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot parse file: {str(e)}")


@router.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not predictor.model_loaded:
        raise HTTPException(503, "Model not loaded")

    df = _read_dataframe(file)

    # Only required to column year4_term2, year5_term1/year5_term2 optional
    required_cols = [
        "faculty","gender","gpax","count_f",
        "year1_term1","year1_term2","year2_term1","year2_term2",
        "year3_term1","year3_term2","year4_term1","year4_term2"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {', '.join(missing)}")

    # Support batch with and without year5_term1/year5_term2
    has_y5_1 = "year5_term1" in df.columns
    has_y5_2 = "year5_term2" in df.columns

    results: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        term_gpas = [
            row.get("year1_term1"), row.get("year1_term2"),
            row.get("year2_term1"), row.get("year2_term2"),
            row.get("year3_term1"), row.get("year3_term2"),
            row.get("year4_term1"), row.get("year4_term2")
        ]
        # Optionally add year5
        if has_y5_1:
            term_gpas.append(row.get("year5_term1"))
        if has_y5_2:
            term_gpas.append(row.get("year5_term2"))

        # normalize NaNs to None
        term_gpas = [None if pd.isna(v) else float(v) for v in term_gpas]

        current_term = max(1, min(len([g for g in term_gpas if g is not None]), 3))

        features = feature_engineer.create_model_features(
            faculty=str(row.get("faculty")),
            gender=str(row.get("gender")),
            gpax=float(row.get("gpax")),
            count_f=int(row.get("count_f")),
            term_gpas=term_gpas,
            current_term=current_term
        )

        num_terms = len([g for g in term_gpas if g is not None])
        pred, prob = predictor.predict(features, num_terms=num_terms)
        risk, color = predictor.get_risk(prob)
        explanations = feature_engineer.get_feature_explanation(features)
        student_id = row.get("student_id") if "student_id" in df.columns else None
        name = row.get("name") if "name" in df.columns else None

        results.append({
            "row_index": int(idx),
            "student_id": None if pd.isna(student_id) else student_id,
            "name": None if pd.isna(name) else name,
            "prediction": int(pred),
            "prediction_label": "Dropout" if pred == 1 else "Graduate",
            "dropout_probability": float(prob),
            "dropout_percentage": f"{prob*100:.1f}%",
            "risk_level": risk,
            "risk_color": color,
            "feature_explanations": explanations,
        })

    return {
        "count": len(results),
        "results": results
    }


