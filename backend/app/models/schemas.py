from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class StudentBasicInput(BaseModel):
    """ข้อมูลพื้นฐานของนักศึกษา"""
    faculty: str = Field(..., description="คณะ")
    gender: str = Field(..., description="เพศ")
    gpax: float = Field(..., ge=0, le=4, description="เกรดเฉลี่ยสะสม")
    count_f: int = Field(..., ge=0, description="จำนวนวิชาที่ได้ F")
    year1_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 1 เทอม 1")
    year1_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 1 เทอม 2")
    year2_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 2 เทอม 1")
    year2_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 2 เทอม 2")
    year3_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 3 เทอม 1")
    year3_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 3 เทอม 2")
    year4_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 4 เทอม 1")
    year4_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 4 เทอม 2")
    year5_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 5 เทอม 1")
    year5_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 5 เทอม 2")

class StudentInput(BaseModel):
    """ข้อมูลสำหรับโมเดล (features ที่ประมวลผลแล้ว)"""
    TERM1: Optional[float] = Field(None, ge=0, le=4)
    TERM2: Optional[float] = Field(None, ge=0, le=4)
    TERM3: Optional[float] = None
    TERM4: Optional[float] = None
    TERM5: Optional[float] = None
    TERM6: Optional[float] = None
    TERM7: Optional[float] = None
    TERM8: Optional[float] = None
    COUNT_F: int = Field(..., ge=0)
    COUNT_WIU: int = Field(..., ge=0)
    OLD_GPA_M6: float = Field(..., ge=0, le=4)
    GPA: float = Field(..., ge=0, le=4)
    num_terms_completed: int = Field(..., ge=1, le=10)
    last_gpa: float = Field(..., ge=0, le=4)
    gpa_trend: float = Field(..., ge=-4, le=4)
    GENDER_ENCODED: int = Field(..., ge=0, le=1)
    FAC_ENCODED: int = Field(..., ge=0, le=5)

class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    dropout_probability: float
    dropout_percentage: str
    risk_level: str
    risk_color: str
    recommendation: str
    feature_explanations: Optional[Dict[str, str]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class FuturePredictionRequest(BaseModel):
    """คำขอสำหรับทำนายอนาคต - รวมข้อมูลทั้งหมด"""
    faculty: str = Field(..., description="คณะ")
    gender: str = Field(..., description="เพศ")
    gpax: float = Field(..., ge=0, le=4, description="เกรดเฉลี่ยสะสม")
    count_f: int = Field(..., ge=0, description="จำนวนวิชาที่ได้ F")
    year1_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 1 เทอม 1")
    year1_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 1 เทอม 2")
    year2_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 2 เทอม 1")
    year2_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 2 เทอม 2")
    year3_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 3 เทอม 1")
    year3_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 3 เทอม 2")
    year4_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 4 เทอม 1")
    year4_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 4 เทอม 2")
    year5_term1: Optional[float] = Field(None, ge=0, le=4, description="ปี 5 เทอม 1")
    year5_term2: Optional[float] = Field(None, ge=0, le=4, description="ปี 5 เทอม 2")
    future_gpa: float = Field(..., ge=0, le=4, description="เกรดที่คาดหวังในเทอมถัดไป")

class FuturePredictionOutput(BaseModel):
    """ผลการทำนายอนาคต"""
    current_probability: float
    future_probability: float
    current_percentage: str
    future_percentage: str
    improvement: float
    improvement_percentage: str
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    loaded_terms: Dict[str, bool]
    loaded_count: int
