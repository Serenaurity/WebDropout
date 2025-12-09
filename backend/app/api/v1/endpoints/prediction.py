from fastapi import APIRouter, HTTPException
from ....models.schemas import StudentInput, StudentBasicInput, PredictionOutput, FuturePredictionRequest, FuturePredictionOutput
from ....models.ml_model import predictor
from ....utils.feature_engineering import FeatureEngineer

router = APIRouter()
feature_engineer = FeatureEngineer()

@router.post("/predict", response_model=PredictionOutput)
async def predict(student: StudentInput):
    """ทำนายจาก features ที่ประมวลผลแล้ว"""
    if not predictor.model_loaded:
        raise HTTPException(503, "Model not loaded")
    
    data = student.model_dump()
    pred, prob = predictor.predict(data)
    risk, color = predictor.get_risk(prob)
    
    return PredictionOutput(
        prediction=pred,
        prediction_label="Dropout" if pred == 1 else "Graduate",
        dropout_probability=prob,
        dropout_percentage=f"{prob*100:.1f}%",
        risk_level=risk,
        risk_color=color,
        recommendation=f"Risk level: {risk}"
    )

@router.post("/predict-from-basic", response_model=PredictionOutput)
async def predict_from_basic(student_basic: StudentBasicInput):
    """ทำนายจากข้อมูลพื้นฐาน"""
    if not predictor.model_loaded:
        raise HTTPException(503, "Model not loaded")
    
    try:
        # แปลงข้อมูลพื้นฐานเป็น term GPAs
        term_gpas = [
            student_basic.year1_term1,
            student_basic.year1_term2,
            student_basic.year2_term1,
            student_basic.year2_term2,
            student_basic.year3_term1,
            student_basic.year3_term2,
            student_basic.year4_term1,
            student_basic.year4_term2,
            student_basic.year5_term1,
            student_basic.year5_term2
        ]
        
        # นับจำนวนเทอมที่มีข้อมูลเพื่อเลือกโมเดล
        num_terms = len([gpa for gpa in term_gpas if gpa is not None])
        current_term = max(1, min(num_terms, 3))

        # สร้าง features สำหรับ XGBoost models
        features = feature_engineer.create_model_features(
            faculty=student_basic.faculty,
            gender=student_basic.gender,
            gpax=student_basic.gpax,
            count_f=student_basic.count_f,
            term_gpas=term_gpas,
            current_term=current_term
        )
        
        # ทำนาย
        pred, prob = predictor.predict(features, num_terms=current_term)
        risk, color = predictor.get_risk(prob)
        
        # สร้างคำแนะนำ
        recommendation = generate_recommendation(risk, prob, features)
        
        # อธิบาย features ที่สำคัญ
        feature_explanations = feature_engineer.get_feature_explanation(features)
        
        return PredictionOutput(
            prediction=pred,
            prediction_label="Dropout" if pred == 1 else "Graduate",
            dropout_probability=prob,
            dropout_percentage=f"{prob*100:.1f}%",
            risk_level=risk,
            risk_color=color,
            recommendation=recommendation,
            feature_explanations=feature_explanations
        )
        
    except Exception as e:
        raise HTTPException(400, f"Error processing data: {str(e)}")

@router.post("/predict-future", response_model=FuturePredictionOutput)
async def predict_future(request: FuturePredictionRequest):
    """ทำนายผลลัพธ์หากเกรดเทอมถัดไปเป็นตามที่กำหนด"""
    if not predictor.model_loaded:
        raise HTTPException(503, "Model not loaded")
    
    try:
        # แปลงข้อมูลพื้นฐานเป็น term GPAs
        term_gpas = [
            request.year1_term1,
            request.year1_term2,
            request.year2_term1,
            request.year2_term2,
            request.year3_term1,
            request.year3_term2,
            request.year4_term1,
            request.year4_term2,
            request.year5_term1,
            request.year5_term2
        ]
        
        # คำนวณเทอมปัจจุบัน (จำกัดใช้โมเดล term1-3)
        current_term = max(1, min(len([gpa for gpa in term_gpas if gpa is not None]), 3))

        # สร้าง features ปัจจุบันตาม XGBoost
        current_features = feature_engineer.create_model_features(
            faculty=request.faculty,
            gender=request.gender,
            gpax=request.gpax,
            count_f=request.count_f,
            term_gpas=term_gpas,
            current_term=current_term
        )

        # Scenario อนาคต: ใส่ GPA เทอมถัดไป แล้วคำนวณ features ใหม่
        future_term_gpas = term_gpas.copy()
        if current_term < 8:
            future_term_gpas[current_term] = request.future_gpa  # index current_term is next term
        future_features = feature_engineer.create_model_features(
            faculty=request.faculty,
            gender=request.gender,
            gpax=request.gpax,
            count_f=request.count_f,
            term_gpas=future_term_gpas,
            current_term=min(current_term + 1, 3)
        )
        
        # ทำนายทั้งสองกรณี
        current_pred, current_prob = predictor.predict(current_features, num_terms=current_term)
        future_pred, future_prob = predictor.predict(future_features, num_terms=min(current_term + 1, 3))
        
        # คำนวณการปรับปรุง
        improvement = current_prob - future_prob
        improvement_percentage = f"{improvement*100:.1f}%"
        
        # สร้างคำแนะนำ
        if improvement > 0:
            recommendation = f"หากได้เกรด {request.future_gpa:.2f} ในเทอมถัดไป ความเสี่ยงจะลดลง {improvement_percentage}"
        elif improvement < 0:
            recommendation = f"หากได้เกรด {request.future_gpa:.2f} ในเทอมถัดไป ความเสี่ยงจะเพิ่มขึ้น {abs(improvement)*100:.1f}%"
        else:
            recommendation = f"หากได้เกรด {request.future_gpa:.2f} ในเทอมถัดไป ความเสี่ยงจะไม่เปลี่ยนแปลง"
        
        return FuturePredictionOutput(
            current_probability=current_prob,
            future_probability=future_prob,
            current_percentage=f"{current_prob*100:.1f}%",
            future_percentage=f"{future_prob*100:.1f}%",
            improvement=improvement,
            improvement_percentage=improvement_percentage,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(400, f"Error processing future prediction: {str(e)}")

def generate_recommendation(risk_level: str, probability: float, features: dict) -> str:
    """สร้างคำแนะนำตามฟีเจอร์เสี่ยงที่ตรวจพบ"""
    recs = []

    # helper: นับเทอมที่มีข้อมูลจริง (>0)
    term_vals = {f"TERM{i}": features.get(f"TERM{i}", 0) for i in range(1, 9)}
    terms_with_data = sum(1 for v in term_vals.values() if v and v > 0)

    # F-related
    if features.get("excessive_F", 0) == 1 or features.get("multiple_F", 0) == 1:
        recs.append("มีหลายวิชาที่ได้ F: เข้าพบอาจารย์ที่ปรึกษา วางแผนแก้รายวิชาที่ตก และขอ/เข้ากลุ่มติวเสริม")
    elif features.get("has_F", 0) == 1 or features.get("COUNT_F", 0) > 0:
        recs.append("มีประวัติ F: ทบทวนวิชาที่อ่อน และขอคำปรึกษาเพื่อวางแผนการเรียนซ้ำ")

    # GPA levels
    if features.get("critical_gpa", 0) == 1:
        recs.append("เกรดเฉลี่ยต่ำมาก (<1.75): จัดตารางเรียนใหม่ ลดภาระกิจชั่วคราว และเข้ารับการติว/เสริมอย่างใกล้ชิด")
    elif features.get("very_low_gpa", 0) == 1:
        recs.append("เกรดเฉลี่ยต่ำ (<2.0): เพิ่มเวลาอ่าน ทบทวนพื้นฐานวิชาหลัก และขอความช่วยเหลือจากอาจารย์/เพื่อนติว")
    elif features.get("low_gpa", 0) == 1:
        recs.append("เกรดเฉลี่ยต่ำ (<2.5): ตั้งเป้าเกรดรายวิชาและจัดตารางอ่านหนังสือสม่ำเสมอ")

    # Trend / decline
    if terms_with_data >= 3 and features.get("consecutive_decline_2", 0) == 1:
        recs.append("เกรดลดลงต่อเนื่อง 2 เทอม: ทำแผนฟื้นฟูผลการเรียนร่วมกับอาจารย์ที่ปรึกษา")
    elif terms_with_data >= 2 and (features.get("declining_trend", 0) == 1 or features.get("decline_last_term", 0) == 1):
        recs.append("แนวโน้มเกรดลดลง: ทบทวนสาเหตุ (เวลาเรียน/งาน/สุขภาพ) และปรับตารางเรียน-พักผ่อน")
    elif terms_with_data >= 2 and features.get("improving_trend", 0) == 1:
        recs.append("แนวโน้มดีขึ้น: รักษาวิธีการเรียนปัจจุบัน และติดตามความก้าวหน้าอย่างต่อเนื่อง")

    # เทอมล่าสุด/รายเทอม
    if (features.get("term3_low", 0) == 1 and terms_with_data >= 3) or (features.get("term2_low", 0) == 1 and terms_with_data >= 2) or (features.get("term1_low", 0) == 1 and terms_with_data >= 1):
        recs.append("เทอมล่าสุดเกรดต่ำ: โฟกัสวิชาหลักของเทอมนั้น จัดตารางอ่าน/ติวเสริมก่อนสอบ")
    for t in range(4, 9):
        # รายงานเฉพาะเทอมที่มีข้อมูลจริง
        if t <= terms_with_data and features.get(f"term{t}_low", 0) == 1 and term_vals.get(f"TERM{t}", 0) > 0:
            recs.append(f"เทอม {t} เกรดต่ำ: ทบทวนวิชาหลักของเทอม {t} และจัดเวลาติวเสริม")
            break  # รายงานครั้งเดียวพอ

    # Early signal
    if features.get("early_warning", 0) == 1:
        recs.append("สัญญาณเตือนตั้งแต่เทอมแรก: ขอการสนับสนุน/ติวพิเศษตั้งแต่เนิ่น ๆ")

    # Recovery / stability
    if features.get("has_recovered", 0) == 1:
        recs.append("กลับมาฟื้นตัวแล้ว: รักษาแนวทางเดิมและติดตามผลเป็นระยะ")
    if terms_with_data >= 2 and features.get("overall_gpa_stability", 0) < 5 and features.get("overall_gpa_stability", 0) != 0:
        recs.append("ความผันผวนของ GPA สูง: จัดตารางเรียน/พักให้สม่ำเสมอ ลดงานซ้อนช่วงสอบ")

    # Performance category / risk score (ใช้เป็นตัวเสริม)
    risk_score = features.get("risk_score", 0)
    if risk_score >= 4:
        recs.append("ความเสี่ยงรวมสูง: นัดหมายที่ปรึกษาเพื่อทำแผนเร่งด่วนและติดตามรายสัปดาห์")

    # ถ้าไม่มี rec เฉพาะ ให้ fallback ตาม risk_level
    if not recs:
        if risk_level == "High":
            recs.append("ความเสี่ยงสูง: ปรึกษาที่ปรึกษาและทำแผนฟื้นฟูทันที")
        elif risk_level == "Medium":
            recs.append("ความเสี่ยงปานกลาง: เพิ่มเวลาทบทวนและติดตามผลการเรียนทุกสัปดาห์")
        else:
            recs.append("ความเสี่ยงต่ำ: รักษาพฤติกรรมการเรียนและทบทวนสม่ำเสมอ")

    return " | ".join(recs)
