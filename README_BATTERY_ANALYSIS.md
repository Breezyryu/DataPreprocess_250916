# Comprehensive Battery Analysis System

범용 배터리 수명 및 성능 테스트 데이터 분석 시스템

## 개요

이 시스템은 ATL, Toyo, PNE 등 다양한 제조사의 배터리 테스트 데이터를 범용적으로 분석할 수 있는 포괄적인 분석 도구입니다.

### 주요 기능

- **범용 데이터 로더**: 다양한 제조사 데이터 형식 자동 인식
- **고급 사이클 분석**: 충방전 사이클, RPT, 멀티스텝, Rss 측정 등 복합 사이클 분류
- **전기화학 열화 분석**: dQ/dV, 내부저항, LLI/LAM 정량화, RUL 예측
- **포괄적 시각화**: matplotlib 및 plotly 기반 정적/인터랙티브 차트
- **자동 보고서 생성**: HTML/JSON 형식 상세 보고서

## 시스템 구조

```
BatteryAnalyzer/
├── core/
│   ├── __init__.py
│   └── data_loader.py          # 범용 데이터 로더
├── analyzers/
│   ├── __init__.py
│   ├── cycle_analyzer.py       # 사이클 분석 및 분류
│   └── degradation_analyzer.py # 전기화학 열화 분석
├── visualizers/
│   ├── __init__.py
│   └── battery_plots.py        # 시각화 도구
└── reports/
    ├── __init__.py
    └── report_generator.py     # 보고서 생성기
```

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 기본 사용

```bash
# 전체 데이터셋 분석
python comprehensive_battery_analysis.py

# 특정 데이터셋 분석
python comprehensive_battery_analysis.py --dataset "250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc"

# 특정 채널만 분석
python comprehensive_battery_analysis.py --channels 30,31

# 인터랙티브 대시보드 생성
python comprehensive_battery_analysis.py --interactive

# 제조사 비교 분석
python comprehensive_battery_analysis.py --compare
```

### 고급 사용법

```bash
# JSON 보고서 생성
python comprehensive_battery_analysis.py --report-format json

# 상세 출력
python comprehensive_battery_analysis.py --verbose

# 결과 디렉토리 지정
python comprehensive_battery_analysis.py --output-dir custom_results
```

## 분석 기능

### 1. 사이클 분석 (Cycle Analysis)

- **사이클 분할**: 연속 데이터를 개별 사이클로 자동 분할
- **사이클 분류**: 8가지 사이클 유형 자동 식별
  - Normal Charge/Discharge
  - 0.2C/0.5C/1C RPT
  - Multi-Step Charge/Discharge
  - Rss (Internal Resistance) Measurement
  - Capacity Check, OCV Measurement
- **성능 지표**: 용량, 효율, C-rate, 내부저항 등 종합 계산

### 2. 전기화학 열화 분석 (Degradation Analysis)

- **Differential Voltage Analysis (dQ/dV)**: 전극 반응 특성 분석
- **내부저항 성분 분해**: Ohmic, charge transfer, diffusion 성분 분리
- **열화 메커니즘 정량화**:
  - LLI (Loss of Lithium Inventory)
  - LAM (Loss of Active Material)
  - Kinetic Loss 정량 분석
- **RUL 예측**: Linear, Exponential, Polynomial 모델 기반 수명 예측

### 3. 시각화 (Visualization)

- **정적 차트**: matplotlib 기반 고품질 출력
- **인터랙티브 대시보드**: plotly 기반 동적 분석
- **제조사 비교**: 다중 제조사 성능 비교 차트
- **종합 분석**: 전압 프로파일, 용량 유지율, 효율성 추이

### 4. 보고서 생성 (Reporting)

- **HTML 보고서**: 웹 기반 종합 보고서
- **JSON 데이터**: 기계 판독 가능한 구조화 데이터
- **건강도 평가**: 0-100 점수 및 등급 시스템
- **성능 지표**: KPI 기반 배터리 상태 요약

## 데이터 형식

### 지원 제조사
- **ATL**: Q7M Inner 등
- **Toyo**: 도요 시리즈
- **PNE**: PNE 시리즈

### 폴더 명명 규칙
```
YYMMDD_YYMMDD_N_테스터명_용량mAh_제조사 모델_테스트조건
예: 250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc
```

### 데이터 구조
```
Rawdata/
└── [데이터셋 폴더]/
    ├── 30/  # 채널 30
    │   ├── 파일1
    │   └── 파일2
    └── 31/  # 채널 31
        ├── 파일1
        └── 파일2
```

## 출력 결과

### 생성 파일
- `battery_report_[제조사]_[타임스탬프].html`: 종합 HTML 보고서
- `battery_report_[제조사]_[타임스탬프].json`: JSON 데이터
- `interactive_dashboard_[데이터셋].html`: 인터랙티브 대시보드
- `plots/`: 모든 생성된 차트 이미지

### 보고서 내용
1. **테스트 정보**: 제조사, 모델, 용량, 테스트 조건
2. **건강도 평가**: 종합 점수 및 등급
3. **테스트 통계**: 데이터 포인트, 사이클, 지속시간
4. **사이클 분석**: 용량 유지율, 효율성 분석
5. **열화 분석**: 메커니즘 정량화, RUL 예측

## API 사용법

```python
from BatteryAnalyzer import UniversalBatteryLoader, CycleAnalyzer, DegradationAnalyzer

# 데이터 로드
loader = UniversalBatteryLoader("Rawdata")
datasets = loader.scan_datasets()
df = loader.load_dataset(datasets[0])

# 사이클 분석
cycle_analyzer = CycleAnalyzer()
cycle_metrics = cycle_analyzer.analyze_all_cycles(df)

# 열화 분석
degradation_analyzer = DegradationAnalyzer()
degradation_results = degradation_analyzer.analyze_degradation_trends(cycle_metrics)
```

## 문제 해결

### 일반적인 오류
1. **데이터 없음**: Rawdata 폴더 확인
2. **메모리 부족**: 채널 수 제한 (`--channels` 사용)
3. **시각화 오류**: matplotlib 백엔드 설정 확인

### 성능 최적화
- 큰 데이터셋의 경우 특정 채널만 분석
- 메모리 사용량 모니터링
- 병렬 처리를 위한 충분한 시스템 리소스 확보

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여

버그 리포트나 기능 요청은 이슈를 통해 제출해 주세요.

## 연락처

개발자: [연락처 정보]
버전: 1.0.0
최종 업데이트: 2025-09-16