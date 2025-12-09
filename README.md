# ระบบทำนายการออกกลางคันของนักศึกษา

## ภาพรวม
ระบบนี้ใช้ XGBoost Machine Learning Models เพื่อทำนายความเสี่ยงการออกกลางคันของนักศึกษา โดยใช้ 3 models แยกตามเทอม:
- **Term 1**: model_term1.json (100 trees)
- **Term 2**: model_term2.json (200 trees)  
- **Term 3+**: model_term3.json (300 trees)

ระบบรับข้อมูลพื้นฐานและสร้าง features ที่จำเป็นสำหรับโมเดลอัตโนมัติ

## คุณสมบัติหลัก

### 1. การกรอกข้อมูลพื้นฐาน
- **ข้อมูลทั่วไป**: คณะ, เพศ, GPAX, จำนวนวิชาที่ได้ F
- **ผลการเรียนรายเทอม**: เกรดเฉลี่ยในแต่ละเทอม (ปี 1-5)

### 2. การวิเคราะห์อัตโนมัติ
- **Smart Model Selection**: เลือก model ตามจำนวนเทอมที่เรียนแล้ว
- **Feature Engineering**: สร้าง features ที่จำเป็นจากข้อมูลพื้นฐาน
- **Risk Assessment**: ประเมินความเสี่ยงออกกลางคันเป็นเปอร์เซ็นต์
- **คำแนะนำ**: ให้คำแนะนำตามระดับความเสี่ยง

### 3. การทำนายอนาคต
- **What-if Analysis**: ทดลองดูว่าหากเกรดเทอมถัดไปดีขึ้น ความเสี่ยงจะลดลงเท่าไหร่
- **Interactive Prediction**: ปรับเกรดที่คาดหวังและดูผลลัพธ์ทันที

## โครงสร้างโปรเจค (อัปเดตหลังทำความสะอาดไฟล์)

```
dropout-prediction/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/endpoints/           # batch, prediction, health
│   │   ├── models/
│   │   │   ├── ml_model.py             # โหลด/ใช้โมเดล XGBoost
│   │   │   └── schemas.py              # Pydantic Schemas
│   │   ├── utils/feature_engineering.py
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── teacher-portal/
│   └── index.html                      # อินเตอร์เฟซอาจารย์ (อัปโหลดไฟล์แบบกลุ่ม)
├── frontend/
│   └── index.html                      # เดโม/หน้าเดี่ยว (ถ้ายังใช้งาน)
├── XG/
│   ├── model_term1.json
│   ├── model_term2.json
│   └── model_term3.json                # ตำแหน่งที่ backend อ้างอิงจริง
└── docker-compose.yml
```

## การติดตั้งและใช้งาน

### 1. ใช้ Docker (แนะนำ)
```bash
# ที่รูทโปรเจค
docker compose -f dropout-prediction/docker-compose.yml up -d --build
```

### 2. เข้าถึงระบบ
- **Teacher Portal**: เปิดไฟล์ `teacher-portal/index.html` หรือผ่านเว็บเซิร์ฟเวอร์ที่คุณใช้งาน
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

### 3. การใช้งาน (Teacher Portal - แบบกลุ่ม)
1. เปิด `teacher-portal/index.html`
2. ดาวน์โหลดเทมเพลต CSV มุมขวาบน (ถ้าต้องการ)
3. อัปโหลดไฟล์ CSV/XLSX แล้วกด "วิเคราะห์"
4. สามารถเรียงข้อมูลตาม "ความเสี่ยง" หรือ "รหัสนักศึกษา" ได้โดยคลิกที่หัวคอลัมน์

## API Endpoints

### 1. `/api/v1/predict-from-basic` (POST)
ทำนายรายบุคคลจากข้อมูลพื้นฐาน
```json
{
  "faculty": "วิทยาศาสตร์และเทคโนโลยี",
  "gender": "ชาย",
  "gpax": 2.6,
  "count_f": 2,
  "year1_term1": 2.46,
  "year1_term2": 2.2,
  "year2_term1": 2.32,
  "year2_term2": 3.2,
  "year3_term1": 2.8
}
```

### 2. `/api/v1/predict-future` (POST)
ทำนายอนาคตรายบุคคล
### 3. `/api/v1/batch-predict` (POST, multipart/form-data)
อัปโหลดไฟล์ `file` เป็น CSV/XLSX เพื่อทำนายแบบกลุ่ม ผลลัพธ์จะรวม `student_id`, `name` ถ้ามีในไฟล์อินพุต
```json
{
  "faculty": "วิทยาศาสตร์และเทคโนโลยี",
  "gender": "ชาย",
  "gpax": 2.6,
  "count_f": 2,
  "year1_term1": 2.46,
  "year1_term2": 2.2,
  "year2_term1": 2.32,
  "year2_term2": 3.2,
  "year3_term1": 2.8,
  "future_gpa": 4.0
}
```

## Features ที่ระบบสร้างอัตโนมัติ

### 1. GPA Features
- `avg_gpa`: เกรดเฉลี่ยสะสม
- `min_gpa`, `max_gpa`: เกรดต่ำสุด/สูงสุด
- `gpa_trend`: แนวโน้มเกรด (เพิ่มขึ้น/ลดลง)
- `num_terms_completed`: จำนวนเทอมที่เรียนแล้ว
- `last_gpa`: เกรดเทอมล่าสุด

### 2. Risk Indicators
- `has_f`: มีประวัติได้ F หรือไม่
- `multiple_f`: ได้ F หลายวิชาหรือไม่
- `low_gpa`: เกรดต่ำหรือไม่
- `early_warning`: มีสัญญาณเตือนหรือไม่
- `declining_trend`: แนวโน้มเกรดลดลงหรือไม่

### 3. Encoded Features
- `GENDER_ENCODED`: เพศ (0=ชาย, 1=หญิง)
- `FAC_ENCODED`: คณะ (0-5)

## ระดับความเสี่ยง

- **ต่ำ (Low)**: < 30% - ความเสี่ยงต่ำ
- **ปานกลาง (Medium)**: 30-60% - ความเสี่ยงปานกลาง
- **สูง (High)**: > 60% - ความเสี่ยงสูง

## คำแนะนำตามระดับความเสี่ยง

### ความเสี่ยงสูง
- ปรึกษาอาจารย์ที่ปรึกษาทันที
- ปรับปรุงการเรียนในวิชาที่ได้ F
- หาวิธีปรับปรุงเกรดให้ดีขึ้น

### ความเสี่ยงปานกลาง
- ปรับพฤติกรรมการเรียน
- ทบทวนบทเรียนและเข้าชั้นเรียนสม่ำเสมอ
- จัดตารางอ่านหนังสือและพักผ่อนให้เพียงพอ

### ความเสี่ยงต่ำ
- รักษาระดับการเรียนให้ดี
- ตั้งเป้าหมายและวางแผนการเรียนให้ชัดเจน

## เทคโนโลยีที่ใช้

### Backend
- **FastAPI**: Web Framework
- **XGBoost**: Machine Learning Model
- **Pydantic**: Data Validation
- **NumPy**: Numerical Computing

### Frontend
- **HTML5/CSS3**: User Interface
- **JavaScript**: Interactive Features
- **Nginx**: Web Server

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container Management

## การพัฒนาต่อ

### Features ที่สามารถเพิ่มได้
1. **Dashboard**: แสดงสถิติและกราฟ
2. **User Management**: ระบบจัดการผู้ใช้
3. **Batch Processing**: ประมวลผลข้อมูลหลายคนพร้อมกัน
4. **Export Reports**: ส่งออกรายงาน PDF/Excel
5. **Mobile App**: แอปพลิเคชันมือถือ
6. **Real-time Updates**: อัปเดตข้อมูลแบบเรียลไทม์

### การปรับปรุงโมเดล
1. **More Features**: เพิ่ม features เพิ่มเติม
2. **Model Retraining**: ฝึกโมเดลใหม่ด้วยข้อมูลล่าสุด
3. **Ensemble Methods**: รวมหลายโมเดลเข้าด้วยกัน
4. **Hyperparameter Tuning**: ปรับแต่งพารามิเตอร์ให้เหมาะสม

## การแก้ไขปัญหา

### ปัญหา: กดวิเคราะห์แล้วหน้าเว็บรีเฟรช ไม่มีการวิเคราะห์

**สาเหตุที่เป็นไปได้:**
1. Backend ไม่ทำงาน
2. Port ไม่ถูกต้อง
3. CORS Error
4. JavaScript Error

**วิธีแก้ไข:**

1. **ตรวจสอบสถานะระบบ:**
   ```bash
   # Windows
   check-system.bat
   
   # หรือใช้คำสั่ง Docker
   docker-compose ps
   docker logs dropout-api
   ```

2. **ทดสอบ API:**
   - เปิด http://localhost:3000/test-api.html
   - ทดสอบ API endpoints ทั้งหมด

3. **ตรวจสอบใน Browser:**
   - กด F12 เพื่อเปิด Developer Tools
   - ดู Console tab เพื่อหา error messages
   - ดู Network tab เพื่อดู API calls

4. **รีสตาร์ทระบบ:**
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

### ปัญหาอื่นๆ ที่พบบ่อย
1. **Model not loaded**: ตรวจสอบว่าไฟล์ model อยู่ในตำแหน่งที่ถูกต้อง
2. **CORS Error**: ตรวจสอบการตั้งค่า CORS ใน backend
3. **Port conflicts**: เปลี่ยน port ใน docker-compose.yml
4. **Memory issues**: เพิ่ม memory สำหรับ Docker

### การ Debug
1. ดู logs ของ containers: `docker-compose logs`
2. เข้าถึง container: `docker exec -it dropout-api bash`
3. ทดสอบ API: ใช้ Postman หรือ curl
4. ตรวจสอบ frontend: เปิด Developer Tools ในเบราว์เซอร์

## License
MIT License - สามารถใช้งานและแก้ไขได้อย่างอิสระ
