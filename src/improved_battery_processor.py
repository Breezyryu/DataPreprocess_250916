"""
배터리 데이터 전처리 시스템 v3.0 - 개선 버전
Improved Battery Data Preprocessing System

개선사항:
1. 코드 구조화 및 모듈화
2. 에러 처리 강화
3. 성능 최적화
4. 타입 힌팅 및 문서화
5. 설정 관리 개선
"""

import os
import sys
import logging
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import json
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import psutil  # 시스템 리소스 모니터링
import gc  # 가비지 컬렉션
# 데이터 품질 검증 (조건부 임포트)
try:
    from data_quality_validator import DataQualityValidator, validate_battery_data
except ImportError:
    DataQualityValidator = None
    validate_battery_data = None

# ======================== Configuration ========================

@dataclass
class Config:
    """시스템 설정 관리"""
    # 파일 처리 - 빅데이터 최적화
    MAX_FILES_PER_CHANNEL: int = 10000  # 매우 큰 값으로 설정 (실질적 제한 없음)
    BATCH_SIZE: int = 200  # 배치 크기 증가로 성능 향상

    # 데이터 처리
    VOLTAGE_MIN: float = 2.0  # V
    VOLTAGE_MAX: float = 5.0  # V
    CURRENT_MIN: float = -10.0  # A
    CURRENT_MAX: float = 10.0  # A
    TEMPERATURE_MIN: float = -40.0  # °C
    TEMPERATURE_MAX: float = 100.0  # °C

    # 성능 설정
    PARALLEL_PROCESSING: bool = True
    MAX_WORKERS: int = 4
    CHUNK_SIZE: int = 10000

    # 출력 설정
    OUTPUT_DIR: str = 'outputs'  # 모든 출력 파일을 저장할 디렉토리
    OUTPUT_FORMAT: str = 'csv'
    PLOT_DPI: int = 150
    PLOT_STYLE: str = 'default'

    # 로깅
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: str = 'battery_processor.log'
    
    @classmethod
    def from_json(cls, file_path: str) -> 'Config':
        """JSON 파일에서 설정 로드"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, file_path: str):
        """설정을 JSON 파일로 저장"""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

# ======================== Logging Setup ========================

def safe_str(text: Any) -> str:
    """한글 등 유니코드 문자를 안전하게 문자열로 변환"""
    try:
        if isinstance(text, (Path, os.PathLike)):
            # Path 객체는 str()로 변환
            return str(text)
        elif isinstance(text, str):
            # 이미 문자열인 경우 그대로 반환
            return text
        else:
            # 기타 타입은 str()로 변환
            return str(text)
    except Exception:
        # 변환 실패 시 repr 사용
        return repr(text)

def setup_logging(config: Config) -> logging.Logger:
    """로깅 설정"""
    # 기존 핸들러 제거 (중복 로깅 방지)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # matplotlib 및 기타 라이브러리 로깅 레벨 조정
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    # StreamHandler에 UTF-8 인코딩 명시적 설정
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            stream_handler
        ],
        force=True  # 기존 설정 덮어쓰기
    )

    return logging.getLogger(__name__)

# ======================== Data Models ========================

class CyclerType(Enum):
    """충방전기 타입"""
    PNE = "PNE"
    TOYO = "TOYO"
    UNKNOWN = "UNKNOWN"

class PathType(Enum):
    """경로 타입"""
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"

@dataclass
class PathInfo:
    """경로 정보"""
    path: Path
    cycler_type: CyclerType
    path_type: PathType
    group_id: str
    sequence: int
    cycle_range: Tuple[int, int] = (0, 0)
    
    def __repr__(self):
        return f"PathInfo({self.path.name}, {self.cycler_type.value}, seq={self.sequence})"

@dataclass
class ProcessingResult:
    """처리 결과 데이터"""
    group_name: str
    data_format: str
    channel_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary_stats: Dict[str, Dict] = field(default_factory=dict)
    csv_file: Optional[str] = None
    plot_file: Optional[str] = None
    error_message: Optional[str] = None
    quality_results: Optional[Dict] = None  # 데이터 품질 검증 결과

@dataclass
class BatteryData:
    """배터리 데이터 구조"""
    test_name: str
    channel_id: str
    cycler_type: CyclerType
    raw_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    cycle_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, config: Config) -> bool:
        """데이터 유효성 검증"""
        if self.raw_data.empty:
            return True  # 빈 데이터는 허용
        
        # 전압 범위 검증
        if 'Voltage_V' in self.raw_data.columns:
            voltage = self.raw_data['Voltage_V']
            if voltage.min() < config.VOLTAGE_MIN or voltage.max() > config.VOLTAGE_MAX:
                logging.warning(f"Voltage out of range in {self.channel_id}")
                
        # 전류 범위 검증
        if 'Current_A' in self.raw_data.columns:
            current = self.raw_data['Current_A']
            if current.min() < config.CURRENT_MIN or current.max() > config.CURRENT_MAX:
                logging.warning(f"Current out of range in {self.channel_id}")
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = {
            'test_name': self.test_name,
            'channel_id': self.channel_id,
            'cycler_type': self.cycler_type.value,
            'data_points': len(self.raw_data),
            'cycles': len(self.cycle_summary)
        }
        
        if not self.raw_data.empty:
            for col in ['Voltage_V', 'Current_A', 'Temperature_C']:
                if col in self.raw_data.columns:
                    stats[f'{col}_mean'] = self.raw_data[col].mean()
                    stats[f'{col}_std'] = self.raw_data[col].std()
                    stats[f'{col}_min'] = self.raw_data[col].min()
                    stats[f'{col}_max'] = self.raw_data[col].max()
        
        return stats

# ======================== Base Classes ========================

class DataLoader(ABC):
    """데이터 로더 추상 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def load_data(self, path: Path) -> Dict[str, BatteryData]:
        """데이터 로드 (구현 필요)"""
        pass
    
    @abstractmethod
    def detect_channels(self, path: Path) -> List[str]:
        """채널 탐지 (구현 필요)"""
        pass

class DataProcessor(ABC):
    """데이터 처리기 추상 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 처리 (구현 필요)"""
        pass

# ======================== TOYO Data Handling ========================

class ToyoDataLoader(DataLoader):
    """TOYO 데이터 로더"""
    
    def detect_channels(self, path: Path) -> List[str]:
        """채널 폴더 탐지"""
        channel_dirs = [d for d in path.iterdir() 
                       if d.is_dir() and d.name.isdigit()]
        return [f"Ch{d.name}" for d in channel_dirs]
    
    def load_data(self, path: Path) -> Dict[str, BatteryData]:
        """TOYO 데이터 로드"""
        data_dict = {}
        channels = self.detect_channels(path)
        
        if self.config.PARALLEL_PROCESSING and len(channels) > 1:
            # 병렬 처리
            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self._load_channel, path, ch): ch 
                    for ch in channels
                }
                
                for future in as_completed(futures):
                    channel = futures[future]
                    try:
                        result = future.result()
                        if result:
                            data_dict[channel] = result
                    except Exception as e:
                        self.logger.error(f"Failed to load {channel}: {e}")
        else:
            # 순차 처리
            for channel in channels:
                try:
                    result = self._load_channel(path, channel)
                    if result:
                        data_dict[channel] = result
                except Exception as e:
                    self.logger.error(f"Failed to load {channel}: {e}")
        
        return data_dict
    
    def _load_channel(self, path: Path, channel: str) -> Optional[BatteryData]:
        """단일 채널 로드"""
        channel_num = channel.replace("Ch", "")
        channel_dir = path / channel_num
        
        if not channel_dir.exists():
            return None
        
        try:
            self.logger.debug(f"Loading channel {channel}: Reading raw files")
            # 데이터 파일 읽기
            raw_data = self._read_raw_files(channel_dir)
            
            if raw_data.empty:
                self.logger.warning(f"No data found for channel {channel}")
                return None
            
            self.logger.debug(f"Channel {channel}: Processing {len(raw_data)} raw data points")
            # 사이클 요약 계산
            processor = ToyoDataProcessor(self.config)
            processed_data = processor.process(raw_data)
            
            self.logger.debug(f"Channel {channel}: Calculating cycle summary")
            cycle_summary = processor.calculate_cycle_summary(processed_data)
            
            # BatteryData 생성
            battery_data = BatteryData(
                test_name=path.name,
                channel_id=channel,
                cycler_type=CyclerType.TOYO,
                raw_data=processed_data,
                cycle_summary=cycle_summary,
                metadata={
                    'channel_dir': str(channel_dir),
                    'file_count': len(list(channel_dir.glob("*")))
                }
            )
            
            # 유효성 검증
            if battery_data.validate(self.config):
                self.logger.info(f"Loaded {channel}: {len(processed_data)} rows")
                return battery_data
            
        except Exception as e:
            self.logger.error(f"Error loading channel {channel}: {e}")
        
        return None
    
    def _read_raw_files(self, channel_dir: Path) -> pd.DataFrame:
        """원시 파일 읽기 (최적화)"""
        data_files = [f for f in channel_dir.iterdir() 
                     if f.is_file() and f.name.isdigit()]
        data_files.sort()
        
        # 빅데이터 처리 - 파일 수 제한 제거
        total_files = len(data_files)
        self.logger.info(f"Processing {total_files} data files for full dataset")
        
        all_data = []
        
        # 빅데이터 배치 처리 with 진행 상황 로깅
        for i in range(0, len(data_files), self.config.BATCH_SIZE):
            batch = data_files[i:i + self.config.BATCH_SIZE]
            batch_data = []
            
            # 진행 상황 로깅
            progress = (i + len(batch)) / total_files * 100
            self.logger.info(f"Processing batch {i//self.config.BATCH_SIZE + 1}: files {i+1}-{min(i+len(batch), total_files)} ({progress:.1f}%)")
            
            for data_file in batch:
                df = self._read_single_file(data_file)
                if df is not None and not df.empty:
                    batch_data.append(df)
            
            if batch_data:
                combined = pd.concat(batch_data, ignore_index=True)
                all_data.append(combined)
                self.logger.debug(f"Batch processed: {len(batch_data)} valid files, {len(combined)} data points")
        
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)

            # 누적 시간 계산 (정렬 전에 수행)
            result_df = self._create_cumulative_time_raw(result_df)

            # Date/Time 기준으로 정렬 - 실제 시계열 순서 보장
            if 'Date' in result_df.columns and 'Time' in result_df.columns:
                try:
                    # Date와 Time을 결합하여 datetime 객체 생성
                    result_df['DateTime'] = pd.to_datetime(result_df['Date'] + ' ' + result_df['Time'],
                                                           format='%Y/%m/%d %H:%M:%S', errors='coerce')
                    # DateTime 기준으로 정렬
                    result_df = result_df.sort_values('DateTime').reset_index(drop=True)
                    # 임시 DateTime 컬럼 제거
                    result_df = result_df.drop(columns=['DateTime'])
                    self.logger.info("Sorting data by Date/Time to ensure chronological order")
                except Exception as e:
                    self.logger.warning(f"Failed to sort by Date/Time: {e}, using PassTime as fallback")
                    # Fallback to PassTime sorting
                    if 'PassTime[Sec]' in result_df.columns:
                        result_df = result_df.sort_values('PassTime[Sec]').reset_index(drop=True)
            elif 'PassTime[Sec]' in result_df.columns:
                # Date/Time이 없으면 PassTime으로 정렬
                self.logger.info("Sorting data by PassTime[Sec]")
                result_df = result_df.sort_values('PassTime[Sec]').reset_index(drop=True)
            return result_df

        return pd.DataFrame()

    def _create_cumulative_time_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """원시 데이터에서 누적 시간 계산 (정렬 전)"""
        if data.empty or 'PassTime[Sec]' not in data.columns:
            return data

        try:
            # PassTime[Sec]를 숫자형으로 변환
            time_sec = pd.to_numeric(data['PassTime[Sec]'], errors='coerce').fillna(0)

            # PassTime[Sec]가 감소하는 지점 찾기 (리셋 지점)
            reset_mask = time_sec.diff() < 0
            reset_points = reset_mask[reset_mask].index.tolist()

            if len(reset_points) > 0:
                # 누적 시간 계산 - 올바른 방법
                cumulative_time = time_sec.copy()
                cumulative_offset = 0

                for i, reset_idx in enumerate(reset_points):
                    # 리셋 직전까지의 최대값을 오프셋에 추가
                    prev_segment_end = reset_points[i-1] if i > 0 else 0
                    prev_segment_max = time_sec.iloc[prev_segment_end:reset_idx].max()

                    if not pd.isna(prev_segment_max) and prev_segment_max > 0:
                        cumulative_offset += prev_segment_max

                    # 리셋 지점부터 다음 리셋 전까지 오프셋 적용
                    next_reset = reset_points[i+1] if i+1 < len(reset_points) else len(time_sec)
                    cumulative_time.iloc[reset_idx:next_reset] = time_sec.iloc[reset_idx:next_reset] + cumulative_offset

                data['PassTime_Cumul[Sec]'] = cumulative_time
                self.logger.info(f"원시 데이터 누적 시간 계산: {len(reset_points)}개 리셋 지점, 최대 {cumulative_time.max():.0f}초")
            else:
                # 리셋이 없으면 원본 복사
                data['PassTime_Cumul[Sec]'] = time_sec
                self.logger.info("원시 데이터에 리셋 없음, PassTime 그대로 사용")

        except Exception as e:
            self.logger.warning(f"원시 데이터 누적 시간 계산 실패: {e}")
            data['PassTime_Cumul[Sec]'] = data['PassTime[Sec]']

        return data

    def _read_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """단일 파일 읽기 (개선된 헤더 탐지)"""
        try:
            # 전체 파일 읽기
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
            
            # 헤더 위치 찾기
            header_line = None
            for i, line in enumerate(lines[:20]):  # 처음 20줄만 확인
                if 'Date' in line and 'Time' in line and 'PassTime' in line:
                    header_line = i
                    break
            
            if header_line is None:
                return None
            
            # 헤더와 데이터를 분리하여 처리
            header_row = lines[header_line].strip().split(',')
            data_lines = lines[header_line + 1:]  # 헤더 다음줄부터 데이터
            
            # 빈 컬럼명 제거
            clean_headers = []
            keep_indices = []
            for i, header in enumerate(header_row):
                if header.strip():  # 빈 문자열이 아닌 경우만
                    clean_headers.append(header.strip())
                    keep_indices.append(i)
            
            # 데이터 행 처리
            data_rows = []
            for line in data_lines:
                if line.strip():  # 빈 줄이 아닌 경우
                    row = line.strip().split(',')
                    # 유효한 컬럼 인덱스의 데이터만 유지
                    clean_row = [row[i] if i < len(row) else '' for i in keep_indices]
                    data_rows.append(clean_row)
            
            # DataFrame 생성
            if data_rows and clean_headers:
                df = pd.DataFrame(data_rows, columns=clean_headers)
                
                # 빈 행 제거
                df = df.replace('', pd.NA).dropna(how='all')
                
                # 기본 검증
                if len(df) == 0 or len(df.columns) < 5:
                    return None
                
                return df
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to read {file_path.name}: {e}")
            return None

class ToyoDataProcessor(DataProcessor):
    """TOYO 데이터 처리기"""
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 처리 및 정리"""
        if data.empty:
            return data
        
        try:
            # 컬럼명 표준화
            self.logger.debug("Step 1: Standardizing columns")
            data = self._standardize_columns(data)
            
            # 수치형 변환
            self.logger.debug("Step 2: Converting to numeric")
            data = self._convert_numeric(data)
            
            # 이상치 제거
            self.logger.debug("Step 3: Removing outliers")
            data = self._remove_outliers(data)
            
            # 용량 계산
            self.logger.debug("Step 4: Calculating capacity")
            data = self._calculate_capacity(data)
            
            # 파생 변수 생성
            self.logger.debug("Step 5: Creating derived features")
            data = self._create_derived_features(data)

            # 누적 시간 계산
            self.logger.debug("Step 5.5: Creating cumulative time")
            data = self._create_cumulative_time(data)

            # 최종 시간 순서 정렬
            self.logger.debug("Step 6: Final time-series sorting")
            if 'Time_Sec' in data.columns:
                try:
                    time_numeric = pd.to_numeric(data['Time_Sec'], errors='coerce')
                    data = data.loc[time_numeric.sort_values().index].reset_index(drop=True)
                    self.logger.debug(f"Final data sorted by time: {len(data)} points")
                except Exception as e:
                    self.logger.warning(f"Failed to sort by time: {e}")

            return data
            
        except Exception as e:
            self.logger.error(f"Error in ToyoDataProcessor.process: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명 표준화"""
        column_mapping = {
            'Date': 'Date',
            'Time': 'Time',
            'PassTime[Sec]': 'Time_Sec',
            'PassTime_Cumul[Sec]': 'Time_Sec_cumul',
            'Voltage[V]': 'Voltage_V',
            'Current[mA]': 'Current_mA',
            'Temp1[Deg]': 'Temperature_C',
            'Cycle': 'Cycle',
            'TotlCycle': 'TotalCycle',
            'Condition': 'Condition',
            'Mode': 'Mode'
        }
        
        # 컬럼 매핑
        for old, new in column_mapping.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        # 불필요한 컬럼 제거
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        df = df.drop(columns=unnamed_cols, errors='ignore')
        
        return df
    
    def _convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """수치형 변환 (TOYO PassTime 특별 처리 포함)"""
        numeric_columns = [
            'Time_Sec', 'Voltage_V', 'Current_mA', 
            'Temperature_C', 'Cycle', 'TotalCycle'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # 안전한 문자열 변환 및 정리
                    series = df[col].copy()

                    # Series인지 확인하고, Series가 아닌 경우 Series로 변환 시도
                    if not isinstance(series, pd.Series):
                        try:
                            # 스칼라 값인 경우 DataFrame의 길이만큼 반복
                            if pd.api.types.is_scalar(series):
                                series = pd.Series([series] * len(df), index=df.index)
                                self.logger.debug(f"Converted scalar value {series.iloc[0]} to Series for column {col}")
                            else:
                                series = pd.Series(series)
                                self.logger.debug(f"Converted column {col} to Series")
                        except Exception as conversion_error:
                            self.logger.warning(f"Column {col} is not a Series and conversion failed: {conversion_error}, skipping")
                            continue
                    
                    # 문자열 변환
                    series = series.astype(str)
                    
                    # Time_Sec (PassTime) 컬럼 특별 처리
                    if col == 'Time_Sec':
                        # TOYO 형식의 00000000 같은 8자리 숫자 문자열을 처리
                        series = series.str.replace('+', '', regex=False)
                        series = series.str.strip()
                        # 8자리 숫자 문자열을 정수로 변환 (예: "00000123" -> 123)
                        series = series.str.lstrip('0')  # 앞의 0들을 제거
                        series = series.replace('', '0')  # 빈 문자열은 0으로 변환
                    else:
                        # 다른 컬럼들의 일반적인 처리
                        series = series.str.replace('+', '', regex=False)
                        series = series.str.replace('-', '-', regex=False)  # 음수 부호는 유지
                        series = series.str.strip()

                        # Temperature_C 컬럼의 특별 처리 (소수점 포함)
                        if col == 'Temperature_C':
                            # 온도 데이터에서 불필요한 문자 제거 (소수점과 음수 부호는 유지)
                            series = series.str.replace(r'[^\d.\-]', '', regex=True)  # 숫자, 점, 음수 부호만 유지
                    
                    # 공통 정리
                    series = series.replace(['nan', 'NaN', 'None', '', 'null'], np.nan)
                    
                    # 수치형 변환
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    
                    # 결과 할당
                    df[col] = numeric_series
                    
                    # 변환 결과 로깅
                    valid_count = numeric_series.notna().sum()
                    total_count = len(numeric_series)
                    
                    if valid_count == 0:
                        self.logger.warning(f"All values in {col} became NaN after conversion")
                    else:
                        self.logger.debug(f"Successfully converted {valid_count}/{total_count} values in {col}")
                        
                except Exception as e:
                    self.logger.warning(f"Could not process column {col}: {e}")
                    # 원본 컬럼을 그대로 두고 계속 진행
                    continue
        
        # 전류 단위 변환 (mA -> A) - 안전한 변환
        if 'Current_mA' in df.columns:
            try:
                current_ma = pd.to_numeric(df['Current_mA'], errors='coerce')
                df['Current_A'] = current_ma / 1000.0
                self.logger.debug("Successfully converted Current_mA to Current_A")
            except Exception as e:
                self.logger.warning(f"Error converting current units: {e}")
                df['Current_A'] = 0.0
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """이상치 제거 (IQR 방법) - 안전한 수치형 처리"""
        for col in ['Voltage_V', 'Current_A', 'Temperature_C']:
            if col in df.columns:
                try:
                    # 컬럼이 실제로 Series인지 확인
                    if not isinstance(df[col], pd.Series):
                        self.logger.debug(f"Skipping outlier removal for {col}: not a Series")
                        continue

                    # 수치형 변환 확인
                    numeric_series = pd.to_numeric(df[col], errors='coerce')

                    # NaN이 아닌 값이 충분한지 확인
                    valid_data = numeric_series.dropna()
                    if len(valid_data) < 10:  # 데이터가 너무 적으면 건너뛰기
                        self.logger.debug(f"Skipping outlier removal for {col}: insufficient valid data ({len(valid_data)} values)")
                        continue
                    
                    # quantile 계산
                    Q1 = valid_data.quantile(0.25)
                    Q3 = valid_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR == 0:  # IQR가 0이면 건너뛰기
                        self.logger.debug(f"Skipping outlier removal for {col}: IQR is 0")
                        continue
                    
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # 이상치 마스크 생성 (수치형 데이터로)
                    outlier_mask = (numeric_series >= lower_bound) & (numeric_series <= upper_bound)
                    # NaN 값은 유지 (outlier가 아님)
                    outlier_mask = outlier_mask | numeric_series.isna()
                    
                    removed = len(df) - outlier_mask.sum()
                    
                    if removed > 0:
                        self.logger.debug(f"Removed {removed} outliers from {col}")
                        df = df[outlier_mask]
                        
                except Exception as e:
                    self.logger.warning(f"Error removing outliers from {col}: {e}")
                    # 오류 발생 시 해당 컬럼은 그대로 두고 계속 진행
                    continue
        
        return df
    
    def _calculate_capacity(self, df: pd.DataFrame) -> pd.DataFrame:
        """용량 계산 (안전한 적분)"""
        if 'Time_Sec' not in df.columns or 'Current_A' not in df.columns:
            self.logger.warning("Time_Sec or Current_A column missing for capacity calculation")
            return df
        
        # 데이터 타입 확인 및 변환
        try:
            df['Time_Sec'] = pd.to_numeric(df['Time_Sec'], errors='coerce')
            df['Current_A'] = pd.to_numeric(df['Current_A'], errors='coerce')
        except Exception as e:
            self.logger.error(f"Error converting data types for capacity calculation: {e}")
            return df
        
        # NaN 값 체크
        if df['Time_Sec'].isna().all() or df['Current_A'].isna().all():
            self.logger.warning("All Time_Sec or Current_A values are NaN")
            df['Capacity_Ah'] = 0.0
            return df
        
        # 시간순 정렬
        df = df.sort_values('Time_Sec').reset_index(drop=True)
        
        # Cycle별로 용량 계산
        if 'Cycle' in df.columns:
            df['Capacity_Ah'] = 0.0
            
            for cycle in df['Cycle'].unique():
                if pd.isna(cycle):
                    continue
                
                cycle_mask = df['Cycle'] == cycle
                cycle_data = df[cycle_mask].copy()
                
                # NaN 값 제거
                cycle_data = cycle_data.dropna(subset=['Time_Sec', 'Current_A'])
                
                if len(cycle_data) > 1:
                    try:
                        time_hours = cycle_data['Time_Sec'].values / 3600.0
                        current = cycle_data['Current_A'].values
                        
                        # 사다리꼴 적분
                        capacity = np.zeros_like(current)
                        for i in range(1, len(current)):
                            dt = time_hours[i] - time_hours[i-1]
                            avg_current = (current[i] + current[i-1]) / 2
                            capacity[i] = capacity[i-1] + avg_current * dt
                        
                        df.loc[cycle_mask, 'Capacity_Ah'] = np.abs(capacity)
                    except Exception as e:
                        self.logger.warning(f"Error calculating capacity for cycle {cycle}: {e}")
                        df.loc[cycle_mask, 'Capacity_Ah'] = 0.0
        else:
            # 전체 데이터에 대해 용량 계산
            clean_df = df.dropna(subset=['Time_Sec', 'Current_A']).copy()
            
            if len(clean_df) > 1:
                try:
                    time_hours = clean_df['Time_Sec'].values / 3600.0
                    current = clean_df['Current_A'].values
                    
                    capacity = np.zeros_like(current)
                    for i in range(1, len(current)):
                        dt = time_hours[i] - time_hours[i-1]
                        avg_current = (current[i] + current[i-1]) / 2
                        capacity[i] = capacity[i-1] + avg_current * dt
                    
                    df['Capacity_Ah'] = 0.0  # 기본값
                    df.loc[clean_df.index, 'Capacity_Ah'] = np.abs(capacity)
                except Exception as e:
                    self.logger.warning(f"Error calculating overall capacity: {e}")
                    df['Capacity_Ah'] = 0.0
            else:
                df['Capacity_Ah'] = 0.0
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        # 에너지 계산
        if 'Voltage_V' in df.columns and 'Current_A' in df.columns:
            df['Power_W'] = df['Voltage_V'] * df['Current_A']
        
        # 충방전 상태
        if 'Current_A' in df.columns:
            df['State'] = pd.cut(df['Current_A'], 
                                bins=[-np.inf, -0.01, 0.01, np.inf],
                                labels=['Discharge', 'Rest', 'Charge'])
        
        # SOC 추정 (간단한 방법)
        if 'Capacity_Ah' in df.columns:
            max_capacity = df.groupby('Cycle')['Capacity_Ah'].transform('max')
            df['SOC_%'] = (df['Capacity_Ah'] / max_capacity * 100).clip(0, 100)
        
        return df

    def _create_cumulative_time(self, data: pd.DataFrame) -> pd.DataFrame:
        """누적 시간 컬럼 생성 - Time_Sec가 리셋되는 구간을 고려"""
        if data.empty or 'Time_Sec' not in data.columns:
            return data

        try:
            # Time_Sec를 숫자형으로 변환
            time_sec = pd.to_numeric(data['Time_Sec'], errors='coerce').fillna(0)

            # Time_Sec가 감소하는 지점 찾기 (리셋 지점)
            reset_mask = time_sec.diff() < 0
            reset_points = reset_mask[reset_mask].index.tolist()

            # 누적 시간 계산
            cumulative_time = time_sec.copy()
            cumulative_offset = 0

            for reset_idx in reset_points:
                # 리셋 직전 시간값을 누적 오프셋에 추가
                if reset_idx > 0:
                    prev_max = time_sec.iloc[:reset_idx].max()
                    if not pd.isna(prev_max):
                        cumulative_offset = cumulative_time.iloc[reset_idx - 1]

                # 리셋 지점 이후의 모든 시간에 오프셋 추가
                cumulative_time.iloc[reset_idx:] += cumulative_offset

            data['Time_Sec_cumul'] = cumulative_time

            self.logger.info(f"누적 시간 계산 완료: {len(reset_points)}개 리셋 지점, 최대 {cumulative_time.max():.0f}초")

        except Exception as e:
            self.logger.warning(f"누적 시간 계산 실패: {e}")
            # 실패 시 기본값으로 Time_Sec 복사
            data['Time_Sec_cumul'] = data['Time_Sec']

        return data

    def calculate_cycle_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """사이클 요약 통계 (최적화)"""
        if 'Cycle' not in df.columns or df.empty:
            return pd.DataFrame()
        
        summary_list = []
        
        # Cycle별 그룹화
        grouped = df.groupby('Cycle')
        
        for cycle_num, cycle_data in grouped:
            if len(cycle_data) < 10:  # 너무 짧은 사이클 제외
                continue
            
            summary = {'Cycle': int(cycle_num)}
            
            # 충방전 구분
            if 'State' in cycle_data.columns:
                charge_data = cycle_data[cycle_data['State'] == 'Charge']
                discharge_data = cycle_data[cycle_data['State'] == 'Discharge']
            else:
                charge_data = cycle_data[cycle_data['Current_A'] > 0.01]
                discharge_data = cycle_data[cycle_data['Current_A'] < -0.01]
            
            # 용량 계산 - 안전한 수치형 변환
            if 'Capacity_Ah' in cycle_data.columns:
                try:
                    # Capacity_Ah 컬럼을 수치형으로 변환
                    capacity_numeric = pd.to_numeric(cycle_data['Capacity_Ah'], errors='coerce')
                    
                    if len(charge_data) > 0:
                        charge_capacity = pd.to_numeric(charge_data['Capacity_Ah'], errors='coerce')
                        if not charge_capacity.isna().all():
                            summary['Charge_Capacity_Ah'] = charge_capacity.max() - charge_capacity.min()
                        else:
                            summary['Charge_Capacity_Ah'] = 0
                    else:
                        summary['Charge_Capacity_Ah'] = 0
                except Exception as e:
                    self.logger.warning(f"Error calculating charge capacity for cycle {cycle_num}: {e}")
                    summary['Charge_Capacity_Ah'] = 0
                
                try:
                    if len(discharge_data) > 0:
                        discharge_capacity = pd.to_numeric(discharge_data['Capacity_Ah'], errors='coerce')
                        if not discharge_capacity.isna().all():
                            summary['Discharge_Capacity_Ah'] = discharge_capacity.max() - discharge_capacity.min()
                        else:
                            summary['Discharge_Capacity_Ah'] = 0
                    else:
                        summary['Discharge_Capacity_Ah'] = 0
                except Exception as e:
                    self.logger.warning(f"Error calculating discharge capacity for cycle {cycle_num}: {e}")
                    summary['Discharge_Capacity_Ah'] = 0
            
            # 효율 계산
            if summary.get('Charge_Capacity_Ah', 0) > 0:
                summary['Coulombic_Efficiency_%'] = (
                    summary.get('Discharge_Capacity_Ah', 0) / 
                    summary['Charge_Capacity_Ah'] * 100
                )
            else:
                summary['Coulombic_Efficiency_%'] = 0
            
            # 전압 통계 - 안전한 수치형 변환
            if 'Voltage_V' in cycle_data.columns:
                try:
                    voltage_numeric = pd.to_numeric(cycle_data['Voltage_V'], errors='coerce')
                    if not voltage_numeric.isna().all():
                        summary['Max_Voltage_V'] = voltage_numeric.max()
                        summary['Min_Voltage_V'] = voltage_numeric.min()
                        summary['Avg_Voltage_V'] = voltage_numeric.mean()
                    else:
                        summary['Max_Voltage_V'] = 0
                        summary['Min_Voltage_V'] = 0
                        summary['Avg_Voltage_V'] = 0
                except Exception as e:
                    self.logger.warning(f"Error calculating voltage stats for cycle {cycle_num}: {e}")
                    summary['Max_Voltage_V'] = 0
                    summary['Min_Voltage_V'] = 0
                    summary['Avg_Voltage_V'] = 0
            
            # 온도 통계 - 안전한 수치형 변환
            if 'Temperature_C' in cycle_data.columns:
                try:
                    # 컬럼이 Series인지 먼저 확인
                    if not isinstance(cycle_data['Temperature_C'], pd.Series):
                        self.logger.debug(f"Temperature_C in cycle {cycle_num} is not a Series, skipping temperature stats")
                        summary['Avg_Temperature_C'] = 0
                        summary['Max_Temperature_C'] = 0
                    else:
                        temp_numeric = pd.to_numeric(cycle_data['Temperature_C'], errors='coerce')
                        if not temp_numeric.isna().all():
                            summary['Avg_Temperature_C'] = temp_numeric.mean()
                            summary['Max_Temperature_C'] = temp_numeric.max()
                        else:
                            summary['Avg_Temperature_C'] = 0
                            summary['Max_Temperature_C'] = 0
                except Exception as e:
                    self.logger.warning(f"Error calculating temperature stats for cycle {cycle_num}: {e}")
                    summary['Avg_Temperature_C'] = 0
                    summary['Max_Temperature_C'] = 0
            
            # 시간 정보 - 안전한 수치형 변환 
            if 'Time_Sec' in cycle_data.columns:
                try:
                    time_numeric = pd.to_numeric(cycle_data['Time_Sec'], errors='coerce')
                    if not time_numeric.isna().all():
                        summary['Duration_Hours'] = (time_numeric.max() - time_numeric.min()) / 3600.0
                    else:
                        summary['Duration_Hours'] = 0
                except Exception as e:
                    self.logger.warning(f"Error calculating duration for cycle {cycle_num}: {e}")
                    summary['Duration_Hours'] = 0
            
            summary['Data_Points'] = len(cycle_data)
            
            summary_list.append(summary)
        
        if summary_list:
            return pd.DataFrame(summary_list)
        
        return pd.DataFrame()

# ======================== PNE Data Handling ========================

class PNEDataLoader(DataLoader):
    """PNE 데이터 로더"""
    
    def detect_channels(self, path: Path) -> List[str]:
        """채널 폴더 탐지"""
        channel_dirs = [d for d in path.iterdir() 
                       if d.is_dir() and d.name.startswith('M02Ch')]
        return [d.name for d in channel_dirs]
    
    def load_data(self, path: Path) -> Dict[str, BatteryData]:
        """PNE 데이터 로드"""
        data_dict = {}
        
        # processed_data 폴더 확인
        processed_dir = path / "processed_data"
        if processed_dir.exists():
            # 기존 처리된 데이터 활용
            data_dict = self._load_processed_data(processed_dir, path.name)
        
        if not data_dict:
            # 원시 데이터 로드 시도
            channels = self.detect_channels(path)
            for channel in channels:
                try:
                    # PNE 바이너리 데이터는 복잡하므로 기본 정보만
                    battery_data = BatteryData(
                        test_name=path.name,
                        channel_id=channel,
                        cycler_type=CyclerType.PNE,
                        metadata={'channel_dir': str(path / channel)}
                    )
                    data_dict[channel] = battery_data
                except Exception as e:
                    self.logger.error(f"Failed to load {channel}: {e}")
        
        return data_dict
    
    def _load_processed_data(self, processed_dir: Path, test_name: str) -> Dict[str, BatteryData]:
        """처리된 데이터 로드"""
        data_dict = {}
        
        # CSV 파일 찾기
        csv_files = list(processed_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                
                # 채널명 추출
                channel_match = re.search(r'M02Ch\d+\[\d+\]', csv_file.name)
                if channel_match:
                    channel_id = channel_match.group()
                else:
                    channel_id = csv_file.stem
                
                battery_data = BatteryData(
                    test_name=test_name,
                    channel_id=channel_id,
                    cycler_type=CyclerType.PNE,
                    raw_data=df,
                    metadata={'source_file': str(csv_file)}
                )
                
                data_dict[channel_id] = battery_data
                self.logger.info(f"Loaded processed data: {csv_file.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file.name}: {e}")
        
        return data_dict

# ======================== Main Processor ========================

class BatteryDataProcessor:
    """통합 배터리 데이터 처리기"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = setup_logging(self.config)
        
        # 시각화 설정
        warnings.filterwarnings('ignore')
        plt.style.use(self.config.PLOT_STYLE)
        plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def check_memory_usage(self) -> Dict[str, float]:
        """시스템 메모리 사용량 확인"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent
            }
        except Exception as e:
            self.logger.warning(f"메모리 사용량 확인 실패: {e}")
            return {}

    def manage_memory(self, force_gc: bool = False):
        """메모리 관리 및 정리"""
        try:
            memory_info = self.check_memory_usage()

            # 메모리 사용량이 85% 이상이면 가비지 컬렉션 강제 실행
            if memory_info.get('percent', 0) > 85 or force_gc:
                self.logger.info("메모리 사용량이 높아 가비지 컬렉션 실행")
                gc.collect()

                # 정리 후 메모리 상태
                after_memory = self.check_memory_usage()
                if after_memory:
                    self.logger.info(f"메모리 정리 완료: {after_memory['percent']:.1f}% 사용 중")

        except Exception as e:
            self.logger.warning(f"메모리 관리 실패: {e}")

    def safe_file_read(self, file_path: Path, encoding: str = 'utf-8', max_retries: int = 3) -> Optional[pd.DataFrame]:
        """안전한 파일 읽기 (재시도 로직 포함)"""
        for attempt in range(max_retries):
            try:
                # 파일 존재 확인
                if not file_path.exists():
                    self.logger.error(f"파일이 존재하지 않음: {safe_str(file_path)}")
                    return None

                # 파일 크기 확인 (빈 파일 감지)
                if file_path.stat().st_size == 0:
                    self.logger.warning(f"빈 파일: {safe_str(file_path)}")
                    return pd.DataFrame()

                # CSV 파일 읽기
                df = pd.read_csv(file_path, encoding=encoding)

                # 기본 검증
                if df.empty:
                    self.logger.warning(f"빈 DataFrame: {safe_str(file_path)}")
                    return df

                # 필수 컬럼 확인
                required_columns = ['Time_Sec']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.logger.warning(f"필수 컬럼 누락: {missing_columns} in {safe_str(file_path)}")

                return df

            except UnicodeDecodeError as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"인코딩 오류 (모든 재시도 실패): {safe_str(file_path)} - {e}")
                    return None
                else:
                    # 다른 인코딩으로 재시도
                    encoding = 'cp949' if encoding == 'utf-8' else 'utf-8-sig'
                    self.logger.warning(f"인코딩 재시도 ({attempt+1}/{max_retries}): {encoding}")

            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"파일 읽기 실패 (모든 재시도 실패): {safe_str(file_path)} - {e}")
                    return None
                else:
                    self.logger.warning(f"파일 읽기 재시도 ({attempt+1}/{max_retries}): {e}")
                    import time
                    time.sleep(0.1 * (attempt + 1))  # 점진적 백오프

        return None
    
    def process_paths(self, paths: List[Union[str, Path]]) -> List[ProcessingResult]:
        """경로 목록 처리"""
        # 초기 메모리 상태 확인
        initial_memory = self.check_memory_usage()
        if initial_memory:
            self.logger.info(f"처리 시작 시 메모리 사용량: {initial_memory['percent']:.1f}%")

        # Path 객체로 변환
        path_objects = [Path(p) if isinstance(p, str) else p for p in paths]

        # 유효한 경로만 필터링
        valid_paths = [p for p in path_objects if p.exists() and p.is_dir()]

        if not valid_paths:
            self.logger.error("No valid paths found")
            return []

        self.logger.info(f"Processing {len(valid_paths)} paths")
        
        # 경로 분석 및 그룹화
        path_groups = self._analyze_paths(valid_paths)
        
        # 각 그룹별 데이터 로드
        results = []
        
        for group_id, group_paths in path_groups.items():
            self.logger.info(f"Processing group: {safe_str(group_id)}")
            
            try:
                # 그룹 데이터와 채널별 데이터를 준비
                group_data = []
                channel_data_dict = {}
                data_format = "UNKNOWN"
                
                for path_info in group_paths:
                    # 충방전기 타입별 로더 선택
                    if path_info.cycler_type == CyclerType.TOYO:
                        loader = ToyoDataLoader(self.config)
                        data_format = "TOYO"
                    elif path_info.cycler_type == CyclerType.PNE:
                        loader = PNEDataLoader(self.config)
                        data_format = "PNE"
                    else:
                        self.logger.warning(f"Unknown cycler type for {path_info.path}")
                        continue
                    
                    # 데이터 로드
                    try:
                        channel_data = loader.load_data(path_info.path)
                        
                        # 채널별 데이터 저장
                        for channel_id, battery_data in channel_data.items():
                            if not battery_data.raw_data.empty:
                                channel_data_dict[channel_id] = battery_data.raw_data
                                group_data.append(battery_data)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load {path_info.path}: {e}")
                
                # 요약 통계 생성
                summary_stats = {}
                for channel_id, df in channel_data_dict.items():
                    if not df.empty:
                        stats = {}
                        if 'Voltage_V' in df.columns:
                            stats['voltage_min'] = df['Voltage_V'].min()
                            stats['voltage_max'] = df['Voltage_V'].max()
                        if 'Current_A' in df.columns:
                            stats['current_min'] = df['Current_A'].min()
                            stats['current_max'] = df['Current_A'].max()
                        summary_stats[channel_id] = stats
                
                # CSV 및 시각화 파일 생성
                csv_file = None
                plot_file = None
                
                if channel_data_dict:
                    # CSV 파일 생성
                    # outputs 디렉토리 생성
                    outputs_dir = Path(self.config.OUTPUT_DIR if hasattr(self.config, 'OUTPUT_DIR') else "outputs")
                    outputs_dir.mkdir(exist_ok=True)

                    # 각 채널별로 개별 CSV 파일 저장
                    csv_files = []
                    try:
                        for channel_id, df in channel_data_dict.items():
                            # 각 채널별 CSV 파일명
                            channel_csv = outputs_dir / f"{group_id}_{channel_id}_processed.csv"

                            df_copy = df.copy()
                            # 채널 ID 컬럼 추가
                            df_copy['Channel'] = channel_id

                            # 시계열 데이터를 실제 시간 순서대로 정렬
                            if 'Date' in df_copy.columns and 'Time' in df_copy.columns:
                                try:
                                    # Date와 Time을 결합하여 datetime 객체 생성
                                    df_copy['DateTime'] = pd.to_datetime(df_copy['Date'] + ' ' + df_copy['Time'],
                                                                         format='%Y/%m/%d %H:%M:%S', errors='coerce')
                                    # DateTime 기준으로 정렬
                                    df_sorted = df_copy.sort_values('DateTime').reset_index(drop=True)
                                    # 임시 DateTime 컬럼 제거
                                    df_sorted = df_sorted.drop(columns=['DateTime'])
                                    self.logger.info(f"{channel_id}: Date/Time 기준으로 정렬 완료 ({len(df_sorted)} 포인트)")
                                except Exception as e:
                                    self.logger.warning(f"Failed to sort by Date/Time: {e}, using Time_Sec")
                                    # Fallback to Time_Sec
                                    if 'Time_Sec' in df_copy.columns:
                                        time_numeric = pd.to_numeric(df_copy['Time_Sec'], errors='coerce')
                                        df_sorted = df_copy.loc[time_numeric.sort_values().index].reset_index(drop=True)
                                    else:
                                        df_sorted = df_copy
                            elif 'Time_Sec' in df_copy.columns:
                                # Date/Time이 없으면 Time_Sec으로 정렬
                                time_numeric = pd.to_numeric(df_copy['Time_Sec'], errors='coerce')
                                df_sorted = df_copy.loc[time_numeric.sort_values().index].reset_index(drop=True)
                                self.logger.info(f"{channel_id}: Time_Sec 기준으로 정렬 완료 ({len(df_sorted)} 포인트)")
                            else:
                                df_sorted = df_copy

                            # 개별 채널 CSV 저장
                            df_sorted.to_csv(channel_csv, index=False, encoding='utf-8-sig')
                            csv_files.append(str(channel_csv))
                            self.logger.info(f"CSV saved: {channel_csv}")

                        # 첫 번째 파일을 대표 파일로 설정 (호환성 유지)
                        csv_file = csv_files[0] if csv_files else None

                    except Exception as e:
                        self.logger.error(f"Failed to save CSV files: {e}")
                        csv_file = None
                    
                    # 시각화 파일 생성 (outputs 폴더에)
                    plot_file = outputs_dir / f"{group_id}_plots.png"
                    try:
                        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                        
                        for channel_id, df in list(channel_data_dict.items())[:3]:  # 최대 3개 채널만
                            # 플롯용 데이터도 시간 순서로 정렬
                            plot_df = df.copy()

                            # 누적 시간을 우선적으로 사용, 없으면 Time_Sec 사용
                            time_column = 'Time_Sec_cumul' if 'Time_Sec_cumul' in plot_df.columns else 'Time_Sec'

                            if time_column in plot_df.columns and 'Voltage_V' in plot_df.columns:
                                time_hours = pd.to_numeric(plot_df[time_column], errors='coerce') / 3600.0
                                voltage = pd.to_numeric(plot_df['Voltage_V'], errors='coerce')
                                # NaN 값 제거
                                valid_mask = ~(time_hours.isna() | voltage.isna())
                                axes[0].plot(time_hours[valid_mask], voltage[valid_mask],
                                           label=channel_id, linewidth=0.8, alpha=0.8)

                            if time_column in plot_df.columns and 'Current_A' in plot_df.columns:
                                time_hours = pd.to_numeric(plot_df[time_column], errors='coerce') / 3600.0
                                current = pd.to_numeric(plot_df['Current_A'], errors='coerce')
                                # NaN 값 제거
                                valid_mask = ~(time_hours.isna() | current.isna())
                                axes[1].plot(time_hours[valid_mask], current[valid_mask],
                                           label=channel_id, linewidth=0.8, alpha=0.8)
                        
                        axes[0].set_title(f'Voltage vs Cumulative Time - {group_id}')
                        axes[0].set_xlabel('Cumulative Time (hours)')
                        axes[0].set_ylabel('Voltage (V)')
                        axes[0].grid(True, alpha=0.3)
                        axes[0].legend()

                        axes[1].set_title(f'Current vs Cumulative Time - {group_id}')
                        axes[1].set_xlabel('Cumulative Time (hours)')
                        axes[1].set_ylabel('Current (A)')
                        axes[1].grid(True, alpha=0.3)
                        axes[1].legend()
                        
                        plt.tight_layout()
                        plt.savefig(plot_file, dpi=self.config.PLOT_DPI, bbox_inches='tight')
                        plt.close()
                        self.logger.info(f"Plot saved: {plot_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to save plot {plot_file}: {e}")
                        plot_file = None
                
                # 데이터 품질 검증 (선택사항)
                quality_results = None
                if validate_battery_data is not None and channel_data_dict:
                    try:
                        quality_results = validate_battery_data(channel_data_dict)

                        # 품질 검증 결과 요약
                        total_score = sum(result.score for result in quality_results.values()) / len(quality_results)
                        critical_issues = sum(len(result.issues) for result in quality_results.values())

                        if critical_issues > 0:
                            self.logger.warning(f"{safe_str(group_id)} 데이터 품질 문제 {critical_issues}개 발견")

                        self.logger.info(f"{safe_str(group_id)} 전체 품질 점수: {total_score:.1f}/100")

                    except Exception as e:
                        self.logger.warning(f"데이터 품질 검증 실패: {e}")

                # ProcessingResult 생성
                result = ProcessingResult(
                    group_name=group_id,
                    data_format=data_format,
                    channel_data=channel_data_dict,
                    summary_stats=summary_stats,
                    csv_file=csv_file,
                    plot_file=plot_file,
                    quality_results=quality_results  # 품질 검증 결과 추가
                )
                
                results.append(result)

                # 각 그룹 처리 후 메모리 관리
                self.manage_memory()

            except Exception as e:
                # 오류 발생 시 오류 정보를 포함한 결과 생성
                error_result = ProcessingResult(
                    group_name=group_id,
                    data_format="ERROR",
                    error_message=str(e),
                    quality_results=None
                )
                results.append(error_result)
                self.logger.error(f"Failed to process group {group_id}: {e}")

        # 처리 완료 후 최종 메모리 상태 확인
        final_memory = self.check_memory_usage()
        if final_memory and initial_memory:
            memory_change = final_memory['percent'] - initial_memory['percent']
            self.logger.info(f"처리 완료 시 메모리 사용량: {final_memory['percent']:.1f}% (변화: {memory_change:+.1f}%)")

        # 최종 메모리 정리
        self.manage_memory(force_gc=True)

        return results
    
    def _analyze_paths(self, paths: List[Path]) -> Dict[str, List[PathInfo]]:
        """경로 분석 및 그룹화"""
        path_groups = {}
        
        for path in paths:
            # 충방전기 타입 판별
            cycler_type = self._detect_cycler_type(path)
            
            # 그룹 정보 추출
            group_id, sequence, cycle_range = self._extract_group_info(path.name, cycler_type)
            
            # PathInfo 생성
            path_info = PathInfo(
                path=path,
                cycler_type=cycler_type,
                path_type=PathType.SINGLE,
                group_id=group_id,
                sequence=sequence,
                cycle_range=cycle_range
            )
            
            # 그룹에 추가
            if group_id not in path_groups:
                path_groups[group_id] = []
            path_groups[group_id].append(path_info)
        
        # 다중 경로 타입 설정
        for group_id, group_paths in path_groups.items():
            if len(group_paths) > 1:
                for path_info in group_paths:
                    path_info.path_type = PathType.MULTIPLE
                # 순서대로 정렬
                group_paths.sort(key=lambda x: x.sequence)
        
        return path_groups
    
    def _detect_cycler_type(self, path: Path) -> CyclerType:
        """충방전기 타입 판별"""
        # Pattern 폴더 확인 (PNE)
        if (path / "Pattern").exists():
            return CyclerType.PNE
        
        # 숫자 폴더 확인 (TOYO)
        numeric_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.isdigit()]
        if numeric_dirs:
            return CyclerType.TOYO
        
        # 파일명 패턴 확인
        if re.match(r'^[A-Z]\d+_MP\d+', path.name):
            return CyclerType.PNE
        
        return CyclerType.UNKNOWN
    
    def _extract_group_info(self, path_name: str, cycler_type: CyclerType) -> Tuple[str, int, Tuple[int, int]]:
        """그룹 정보 추출"""
        # TOYO 다중 경로 패턴
        toyo_pattern = r'(.+?)(\d+)-(\d+)cyc'
        match = re.search(toyo_pattern, path_name)
        if match:
            base_name = match.group(1).strip()
            start_cycle = int(match.group(2))
            end_cycle = int(match.group(3))
            group_id = re.sub(r'[^\w가-힣]', '_', base_name).strip('_')
            return group_id, start_cycle, (start_cycle, end_cycle)
        
        # PNE 다중 경로 패턴
        pne_pattern = r'(.+)_(\d+)$'
        match = re.search(pne_pattern, path_name)
        if match and cycler_type == CyclerType.PNE:
            base_name = match.group(1)
            sequence = int(match.group(2))
            group_id = re.sub(r'[^\w가-힣]', '_', base_name).strip('_')
            return group_id, sequence, (0, 0)
        
        # 단일 경로
        group_id = re.sub(r'[^\w가-힣]', '_', path_name).strip('_')
        return group_id, 0, (0, 0)
    
    def export_results(self, results: Dict[str, List[BatteryData]], output_dir: str):
        """결과 내보내기"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting results to {output_path}")
        
        # 그룹별 처리
        for group_id, battery_data_list in results.items():
            group_dir = output_path / group_id
            group_dir.mkdir(exist_ok=True)
            
            # 각 데이터 저장
            for battery_data in battery_data_list:
                # Raw 데이터 저장
                if not battery_data.raw_data.empty:
                    raw_file = group_dir / f"{battery_data.channel_id}_raw.csv"
                    battery_data.raw_data.to_csv(raw_file, index=False, encoding='utf-8-sig')
                
                # 사이클 요약 저장
                if not battery_data.cycle_summary.empty:
                    summary_file = group_dir / f"{battery_data.channel_id}_summary.csv"
                    battery_data.cycle_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
                
                # 통계 정보 저장
                stats_file = group_dir / f"{battery_data.channel_id}_stats.json"
                with open(stats_file, 'w') as f:
                    json.dump(battery_data.get_statistics(), f, indent=2)
        
        self.logger.info("Export completed")
    
    def create_visualizations(self, results: Dict[str, List[BatteryData]], output_dir: str):
        """시각화 생성"""
        output_path = Path(output_dir)
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating visualizations in {plots_dir}")
        
        for group_id, battery_data_list in results.items():
            try:
                self._create_group_plots(group_id, battery_data_list, plots_dir)
            except Exception as e:
                self.logger.error(f"Failed to create plots for {group_id}: {e}")
    
    def _create_group_plots(self, group_id: str, data_list: List[BatteryData], output_dir: Path):
        """그룹별 플롯 생성"""
        # 요약 플롯
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Battery Test Analysis: {group_id}')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_list)))
        
        for i, battery_data in enumerate(data_list):
            if battery_data.cycle_summary.empty:
                continue
            
            color = colors[i]
            label = battery_data.channel_id
            summary = battery_data.cycle_summary
            
            # 용량 추이
            if 'Cycle' in summary.columns and 'Discharge_Capacity_Ah' in summary.columns:
                axes[0, 0].plot(summary['Cycle'], summary['Discharge_Capacity_Ah'],
                               'o-', label=label, color=color, markersize=4)
            
            # 효율 추이
            if 'Cycle' in summary.columns and 'Coulombic_Efficiency_%' in summary.columns:
                axes[0, 1].plot(summary['Cycle'], summary['Coulombic_Efficiency_%'],
                               's-', label=label, color=color, markersize=4)
            
            # 전압 범위
            if all(col in summary.columns for col in ['Cycle', 'Max_Voltage_V', 'Min_Voltage_V']):
                axes[1, 0].fill_between(summary['Cycle'], 
                                       summary['Min_Voltage_V'],
                                       summary['Max_Voltage_V'],
                                       alpha=0.3, color=color, label=label)
            
            # 온도
            if 'Cycle' in summary.columns and 'Avg_Temperature_C' in summary.columns:
                axes[1, 1].plot(summary['Cycle'], summary['Avg_Temperature_C'],
                               'd-', label=label, color=color, markersize=4)
        
        # 축 설정
        axes[0, 0].set_title('Discharge Capacity Trend')
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Capacity (Ah)')
        axes[0, 0].legend(loc='best', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Coulombic Efficiency')
        axes[0, 1].set_xlabel('Cycle')
        axes[0, 1].set_ylabel('Efficiency (%)')
        axes[0, 1].legend(loc='best', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([85, 105])
        
        axes[1, 0].set_title('Voltage Range')
        axes[1, 0].set_xlabel('Cycle')
        axes[1, 0].set_ylabel('Voltage (V)')
        axes[1, 0].legend(loc='best', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Average Temperature')
        axes[1, 1].set_xlabel('Cycle')
        axes[1, 1].set_ylabel('Temperature (°C)')
        axes[1, 1].legend(loc='best', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        plot_file = output_dir / f"{group_id}_summary.png"
        plt.savefig(plot_file, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved plot: {plot_file.name}")

# ======================== CLI Interface ========================

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Battery Data Processor v3.0')
    parser.add_argument('paths', nargs='+', help='Data paths to process')
    parser.add_argument('--output', '-o', default='battery_output', help='Output directory')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create visualizations')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        config = Config.from_json(args.config)
    else:
        config = Config()
    
    # 설정 업데이트
    config.LOG_LEVEL = args.log_level
    if args.parallel:
        config.PARALLEL_PROCESSING = True
    
    # 프로세서 생성
    processor = BatteryDataProcessor(config)
    
    # 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    
    try:
        # 데이터 처리
        results = processor.process_paths(args.paths)
        
        if results:
            # 결과 내보내기
            processor.export_results(results, output_dir)
            
            # 시각화
            if args.visualize:
                processor.create_visualizations(results, output_dir)
            
            print(f"\n처리 완료!")
            print(f"출력 디렉토리: {output_dir}")
            
            # 통계 출력
            total_groups = len(results)
            total_channels = sum(len(data_list) for data_list in results.values())
            print(f"처리된 그룹: {total_groups}")
            print(f"처리된 채널: {total_channels}")
        else:
            print("처리된 데이터가 없습니다.")
            
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())