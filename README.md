# 🔋 배터리 데이터 전처리 시스템

간단하고 효율적인 배터리 충방전 데이터 전처리 도구입니다.

## 📁 파일 구성

- **`simple_battery_demo.ipynb`** - 메인 Jupyter Notebook
- **`improved_battery_processor.py`** - 데이터 처리 엔진
- **`Rawdata/`** - 원시 데이터 폴더

## 🚀 사용법

### 1. Jupyter Notebook 실행
```bash
jupyter notebook
```

### 2. simple_battery_demo.ipynb 열기

### 3. 데이터 경로 수정
```python
# 실제 데이터 경로로 수정
test_path = r"C:\your\actual\data\path"
```

### 4. 셀 순서대로 실행
`Shift + Enter`로 각 셀을 실행하세요.

## ✨ 주요 기능

- 🔍 TOYO/PNE 데이터 자동 인식
- 🧹 데이터 정제 및 표준화
- ⚡ 다중 채널 병렬 처리
- 📊 자동 시각화 및 CSV 출력
- 🔋 용량/에너지 자동 계산

## 📊 출력 파일

- **`{그룹명}_processed.csv`** - 정제된 데이터
- **`{그룹명}_plots.png`** - 시각화 그래프

## 🔧 문제 해결

### 경로 오류
```
❌ 데이터 경로가 존재하지 않습니다.
```
→ `test_path` 변수를 실제 데이터 경로로 수정하세요.

### 모듈 오류  
```
ModuleNotFoundError: No module named 'improved_battery_processor'
```
→ 노트북과 같은 폴더에 `improved_battery_processor.py`가 있는지 확인하세요.

## 📈 지원 형식

### TOYO 형식
```
데이터폴더/
├── 30/     # 채널 폴더
│   ├── 000001
│   └── 000002
└── 31/
```

---

**즐거운 데이터 분석 되세요!** 🎉