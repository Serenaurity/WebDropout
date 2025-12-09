import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import math

class FeatureEngineer:
    """
    Class สำหรับสร้าง features ที่จำเป็นสำหรับโมเดลจากข้อมูลพื้นฐาน
    """
    
    def __init__(self):
        # Faculty mapping
        self.faculty_mapping = {
            "คณะครุศาสตร์": 0,
            "คณะวิทยาศาสตร์และเทคโนโลยี": 1,
            "คณะวิทยาการจัดการ": 2,
            "คณะมนุษยศาสตร์และสังคมศาสตร์": 3,
            "คณะเทคโนโลยีอุตสาหกรรม": 4,
            "คณะเทคโนโลยีการเกษตร": 5,
            "อื่นๆ": 6
        }
        
        # Gender mapping
        self.gender_mapping = {
            "ชาย": 0,
            "หญิง": 1
        }
    
    def create_model_features(
        self,
        faculty: str,
        gender: str,
        gpax: float,
        count_f: int,
        term_gpas: List[Optional[float]],
        current_term: int = 1
    ) -> Dict[str, float]:
        """
        สร้างฟีเจอร์ให้ตรงกับ feature_names ของโมเดล XGBoost ทั้ง 3 เทอม
        """
        # เตรียม term_gpas ความยาว 8
        term_gpas = list(term_gpas or [])
        while len(term_gpas) < 8:
            term_gpas.append(None)
        term_gpas = term_gpas[:8]

        # missing flags TERM1-TERM8
        term_missing = {}
        for i in range(1, 9):
            key = f"TERM{i}"
            miss_key = f"{key}_missing"
            val = term_gpas[i-1]
            term_missing[miss_key] = 1 if val is None else 0

        # fill term values (None -> 0)
        term_values = {f"TERM{i}": (term_gpas[i-1] if term_gpas[i-1] is not None else 0.0) for i in range(1, 9)}

        # valid gpas list (exclude None and zeros)
        valid_gpas = [v for v in term_gpas if v is not None]
        valid_nonzero = [v for v in term_gpas if v not in (None, 0)]

        avg_gpa = float(np.mean(valid_nonzero)) if valid_nonzero else float(gpax)
        min_gpa = float(np.min(valid_nonzero)) if valid_nonzero else 0.0
        max_gpa = float(np.max(valid_nonzero)) if valid_nonzero else 0.0
        gpa_range = max_gpa - min_gpa
        gpa_std = float(np.std(valid_nonzero)) if len(valid_nonzero) > 1 else 0.0

        # gpa_change_from_start: ใช้เทอมแรกที่มีข้อมูลเทียบกับเทอมล่าสุดที่มีข้อมูล
        if len(valid_nonzero) >= 2:
            first_gpa = next(v for v in term_gpas if v not in (None, 0))
            last_gpa = list(v for v in term_gpas if v not in (None, 0))[-1]
            gpa_change_from_start = float(last_gpa - first_gpa)
        else:
            gpa_change_from_start = 0.0

        improvement_from_hs = float(avg_gpa - gpax)

        has_F = 1 if count_f > 0 else 0
        multiple_F = 1 if count_f >= 2 else 0
        excessive_F = 1 if count_f >= 3 else 0
        has_WIU = 0  # ยังไม่มีข้อมูล WIU

        low_gpa = 1 if avg_gpa < 2.5 else 0
        very_low_gpa = 1 if avg_gpa < 2.0 else 0
        critical_gpa = 1 if avg_gpa < 1.75 else 0

        early_warning = 1 if term_values["TERM1"] < 2.0 else 0
        term1_low = 1 if term_values["TERM1"] < 2.5 else 0
        term1_excellent = 1 if term_values["TERM1"] >= 3.5 else 0

        term2_low = 1 if term_values["TERM2"] < 2.5 else 0
        term3_low = 1 if term_values["TERM3"] < 2.5 else 0

        # trend flags
        declining_trend = 1 if gpa_change_from_start < -0.1 else 0
        improving_trend = 1 if gpa_change_from_start > 0.1 else 0

        # decline_last_term: เทียบสองเทอมล่าสุดที่มีข้อมูล
        if len(valid_nonzero) >= 2:
            last_two = [v for v in term_gpas if v not in (None, 0)][-2:]
            decline_last_term = 1 if last_two[-1] < last_two[0] else 0
        else:
            decline_last_term = 0

        consecutive_decline_2 = 1 if (term_values["TERM2"] < term_values["TERM1"] and term_values["TERM3"] < term_values["TERM2"]) else 0

        num_terms_with_data = sum(1 for v in term_values.values() if v > 0)
        latest_available_gpa = [v for v in term_values.values() if v > 0][-1] if num_terms_with_data > 0 else 0.0

        # term4-8 low/excellent
        def low_flag(v): return 1 if v < 2.5 else 0
        def excellent_flag(v): return 1 if v >= 3.5 else 0

        term_low = {f"term{i}_low": low_flag(term_values[f"TERM{i}"]) for i in range(4, 9)}
        term_excellent = {f"term{i}_excellent": excellent_flag(term_values[f"TERM{i}"]) for i in range(4, 9)}

        improving_term4 = 1 if term_values["TERM4"] > term_values["TERM3"] and term_values["TERM3"] > 0 else 0
        improving_term5 = 1 if term_values["TERM5"] > term_values["TERM4"] and term_values["TERM4"] > 0 else 0

        long_decline_3terms = 1 if (
            term_values["TERM4"] < term_values["TERM3"] and
            term_values["TERM5"] < term_values["TERM4"] and
            term_values["TERM6"] < term_values["TERM5"]
        ) else 0

        overall_gpa_stability = float(1 / (gpa_std + 0.1))
        has_recovered = 1 if (min_gpa < 2.0 and latest_available_gpa >= 2.5) else 0

        # performance_category (0-3)
        performance_category = pd.cut(
            [avg_gpa],
            bins=[0, 2.0, 2.5, 3.0, 4.1],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )[0]
        performance_category = int(performance_category) if not pd.isna(performance_category) else 0

        risk_score = (
            has_F * 2 +
            very_low_gpa * 3 +
            declining_trend * 2
        )

        features = {
            **term_values,
            **term_missing,
            "OLD_GPA_M6": float(gpax),
            "GENDER_ENCODED": float(self.gender_mapping.get(gender, 0)),
            "FAC_ENCODED": float(self.faculty_mapping.get(faculty, 0)),
            "COUNT_F": float(count_f),
            "COUNT_WIU": float(has_WIU),
            "avg_gpa_up_to_now": float(avg_gpa),
            "min_gpa_up_to_now": float(min_gpa),
            "max_gpa_up_to_now": float(max_gpa),
            "gpa_range": float(gpa_range),
            "gpa_std": float(gpa_std),
            "gpa_change_from_start": float(gpa_change_from_start),
            "improvement_from_hs": float(improvement_from_hs),
            "has_F": float(has_F),
            "multiple_F": float(multiple_F),
            "excessive_F": float(excessive_F),
            "has_WIU": float(has_WIU),
            "low_gpa": float(low_gpa),
            "very_low_gpa": float(very_low_gpa),
            "critical_gpa": float(critical_gpa),
            "early_warning": float(early_warning),
            "term1_low": float(term1_low),
            "term1_excellent": float(term1_excellent),
            "term2_low": float(term2_low),
            "declining_trend": float(declining_trend),
            "improving_trend": float(improving_trend),
            "decline_last_term": float(decline_last_term),
            "consecutive_decline_2": float(consecutive_decline_2),
            "term3_low": float(term3_low),
            "num_terms_with_data": float(num_terms_with_data),
            "latest_available_gpa": float(latest_available_gpa),
            "improving_term4": float(improving_term4),
            "improving_term5": float(improving_term5),
            "long_decline_3terms": float(long_decline_3terms),
            "overall_gpa_stability": float(overall_gpa_stability),
            "has_recovered": float(has_recovered),
            "performance_category": float(performance_category),
            "risk_score": float(risk_score),
            "current_term": float(current_term)
        }

        # term4-8 low/excellent
        for i in range(4, 9):
            features[f"term{i}_low"] = float(term_low[f"term{i}_low"])
            features[f"term{i}_excellent"] = float(term_excellent[f"term{i}_excellent"])

        return features
    
    def predict_future_scenario(self, current_features: Dict[str, float], future_gpa: float, current_term: int) -> Dict[str, float]:
        """
        สร้าง features ใหม่เมื่อสมมติ GPA เทอมถัดไป
        """
        # reconstruct term_gpas from existing features
        term_gpas = [current_features.get(f"TERM{i}", 0.0) or None for i in range(1, 9)]
        if current_term < 8:
            term_gpas[current_term] = future_gpa  # index current_term is next term
        return self.create_model_features(
            faculty="อื่นๆ",  # faculty/gender ไม่เปลี่ยนจาก current_features แต่ต้องส่งอะไรสักอย่าง (ไม่ใช้ในคำนวณต่อ)
            gender="ชาย",
            gpax=current_features.get("OLD_GPA_M6", 0.0),
            count_f=int(current_features.get("COUNT_F", 0)),
            term_gpas=term_gpas,
            current_term=min(current_term + 1, 3)
        )
    
    def get_feature_explanation(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        อธิบายความหมายของ features ที่สำคัญ
        """
        explanations = {}
        
        # GPA related
        if features.get('GPA', 0) > 0:
            explanations['GPA'] = f"เกรดเฉลี่ยสะสม: {features['GPA']:.2f}"
        
        if features.get('gpa_trend', 0) != 0:
            trend_desc = "เพิ่มขึ้น" if features['gpa_trend'] > 0 else "ลดลง"
            explanations['gpa_trend'] = f"แนวโน้มเกรด: {trend_desc} {abs(features['gpa_trend']):.2f}"
        
        # นับจำนวนเทอมที่มีข้อมูลจริง (>0)
        terms_with_data = sum(1 for i in range(1, 9) if features.get(f"TERM{i}", 0) > 0)
        gpa_delta = features.get('gpa_change_from_start', 0)

        # F related
        if features.get('COUNT_F', 0) > 0:
            explanations['COUNT_F'] = f"จำนวนวิชาที่ได้ F: {int(features['COUNT_F'])} วิชา"
        
        if features.get('has_f', 0) == 1:
            explanations['has_f'] = "มีประวัติได้เกรด F"
        
        # Risk indicators
        if features.get('early_warning', 0) == 1:
            explanations['early_warning'] = "มีสัญญาณเตือน: เกรดต่ำและมี F"
        
        # แสดงแนวโน้มลดลงเฉพาะเมื่อมีข้อมูล ≥3 เทอม และลดลงชัดเจน
        if features.get('declining_trend', 0) == 1 and terms_with_data >= 3 and gpa_delta <= -0.3:
            explanations['declining_trend'] = "แนวโน้มเกรดลดลงอย่างมีนัยสำคัญ"
        
        return explanations
