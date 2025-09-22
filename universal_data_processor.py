"""
범용 배터리 데이터 처리 시스템
Toyo와 PNE 데이터를 자동으로 감지하고 처리하는 통합 시스템
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import re
import glob

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


class BaseDataProcessor(ABC):
    """데이터 처리를 위한 추상 기본 클래스"""

    def __init__(self, data_paths: List[str]):
        """
        Parameters:
        -----------
        data_paths : List[str]
            처리할 데이터 경로 리스트
        """
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.all_data = []
        self.df = None

    @abstractmethod
    def detect_format(self) -> str:
        """데이터 형식을 감지하고 반환"""
        pass

    @abstractmethod
    def parse_data(self) -> List[Dict]:
        """데이터를 파싱하여 딕셔너리 리스트로 반환"""
        pass

    def process(self) -> pd.DataFrame:
        """메인 처리 로직"""
        print(f"데이터 처리 시작...")
        print(f"경로 수: {len(self.data_paths)}")

        # 데이터 파싱
        self.all_data = self.parse_data()

        if not self.all_data:
            print("처리된 데이터가 없습니다.")
            return pd.DataFrame()

        # DataFrame 생성
        self.df = pd.DataFrame(self.all_data)

        # 시간 기준 정렬
        if 'datetime' in self.df.columns:
            self.df = self.df.sort_values('datetime')

        print(f"처리 완료! 총 {len(self.df)} 개의 데이터 포인트")
        return self.df

    def save_to_csv(self, filename: str = None):
        """데이터를 CSV 파일로 저장"""
        if self.df is None or self.df.empty:
            print("저장할 데이터가 없습니다.")
            return

        if filename is None:
            filename = f"outputs/processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.df.to_csv(filename, index=False)
        print(f"데이터가 {filename}에 저장되었습니다.")


class ToyoDataProcessor(BaseDataProcessor):
    """Toyo 데이터 처리 클래스"""

    def detect_format(self) -> str:
        """Toyo 형식 감지"""
        if not self.data_paths:
            return "unknown"

        # 첫 번째 경로의 하위 디렉토리 확인
        first_path = self.data_paths[0]
        if os.path.exists(first_path):
            subdirs = [d for d in os.listdir(first_path) if os.path.isdir(os.path.join(first_path, d))]
            # 숫자로만 된 디렉토리가 있으면 Toyo
            if any(d.isdigit() for d in subdirs):
                return "toyo"

        return "unknown"

    def parse_data(self) -> List[Dict]:
        """Toyo 데이터 파싱"""
        all_data = []
        files_processed = 0
        last_elapsed = 0

        for path_idx, base_path in enumerate(self.data_paths, 1):
            print(f"\n[{path_idx}/{len(self.data_paths)}] 처리 중: {os.path.basename(base_path)}")

            if not os.path.exists(base_path):
                print(f"  경로가 존재하지 않음: {base_path}")
                continue

            # 하위 디렉토리 찾기 (숫자로 된 디렉토리)
            subdirs = [d for d in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]

            if not subdirs:
                print(f"  Toyo 형식 디렉토리를 찾을 수 없음")
                continue

            print(f"  발견된 채널: {', '.join(sorted(subdirs))}")

            # 각 하위 디렉토리 처리
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(base_path, subdir)

                # 파일 처리 (000001, 000002 형식)
                for i in range(1, 1000):  # 최대 1000개 파일
                    file_path = os.path.join(subdir_path, f"{i:06d}")
                    if os.path.exists(file_path):
                        files_processed += 1
                        data = self._parse_toyo_file(file_path, path_idx, int(subdir),
                                                    os.path.basename(base_path), last_elapsed)
                        all_data.extend(data)
                    else:
                        # 연속된 파일이 없으면 중단
                        if i > 10:  # 최소 10개는 체크
                            break

                # 진행 상황 표시
                if files_processed % 50 == 0:
                    print(f"  처리 중... {files_processed} 파일 완료")

            # 현재 경로의 마지막 elapsed time 저장 (다중 경로 연속 처리용)
            if all_data and len(self.data_paths) > 1:
                path_data = [d for d in all_data if d.get('path_idx') == path_idx]
                if path_data:
                    last_elapsed = max(d.get('elapsed_hours', 0) for d in path_data)

        print(f"\n총 처리된 파일: {files_processed}")
        return all_data

    def _parse_toyo_file(self, file_path: str, path_idx: int, channel: int,
                        path_name: str, base_elapsed: float = 0) -> List[Dict]:
        """개별 Toyo 파일 파싱"""
        data = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # 데이터 라인 처리 (5번째 줄부터)
            for line in lines[4:]:
                if '202' in line:  # 연도가 포함된 라인
                    parts = line.strip().split(',')
                    if len(parts) >= 16:
                        try:
                            # 날짜와 시간 파싱
                            date_str = parts[0]
                            time_str = parts[1]
                            datetime_str = f"{date_str} {time_str}"
                            current_datetime = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")

                            # 데이터 추출
                            data.append({
                                'datetime': current_datetime,
                                'elapsed_hours': base_elapsed,  # 추후 재계산
                                'voltage': float(parts[3].replace('+', '')) if parts[3] else 0.0,
                                'current': float(parts[4].replace('+', '')) if parts[4] else 0.0,
                                'temperature': float(parts[7].replace('+', '')) if parts[7] else 0.0,
                                'cycle': int(parts[13]) if parts[13].strip() else 0,
                                'total_cycle': int(parts[14]) if parts[14].strip() else 0,
                                'channel': channel,
                                'path_idx': path_idx,
                                'path_name': path_name,
                                'data_type': 'Toyo'
                            })
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"  파일 읽기 오류 {file_path}: {e}")

        return data


class PNEDataProcessor(BaseDataProcessor):
    """PNE 데이터 처리 클래스"""

    def detect_format(self) -> str:
        """PNE 형식 감지"""
        if not self.data_paths:
            return "unknown"

        # 첫 번째 경로의 하위 디렉토리 확인
        first_path = self.data_paths[0]
        if os.path.exists(first_path):
            subdirs = os.listdir(first_path)
            # M02Ch 패턴이 있으면 PNE
            if any('M02Ch' in d for d in subdirs):
                return "pne"

        return "unknown"

    def parse_data(self) -> List[Dict]:
        """PNE 데이터 파싱 (현재는 메타데이터만)"""
        all_data = []

        for path_idx, base_path in enumerate(self.data_paths, 1):
            print(f"\n[{path_idx}/{len(self.data_paths)}] 처리 중: {os.path.basename(base_path)}")

            if not os.path.exists(base_path):
                print(f"  경로가 존재하지 않음: {base_path}")
                continue

            # M02Ch 패턴 디렉토리 찾기
            channel_dirs = [d for d in os.listdir(base_path)
                          if os.path.isdir(os.path.join(base_path, d)) and 'Ch' in d]

            if not channel_dirs:
                print(f"  PNE 형식 디렉토리를 찾을 수 없음")
                continue

            print(f"  발견된 채널 디렉토리: {', '.join(channel_dirs)}")

            for ch_dir in channel_dirs:
                # 채널 번호 추출
                channel_num = self._extract_channel_number(ch_dir)
                ch_path = os.path.join(base_path, ch_dir)

                # .cyc 파일 찾기
                # Windows에서 대괄호가 있는 경로 처리를 위해 직접 파일 리스트 가져오기
                try:
                    files_in_dir = os.listdir(ch_path)
                    cyc_files = [os.path.join(ch_path, f) for f in files_in_dir if f.endswith('.cyc')]
                except:
                    cyc_files = []

                for cyc_file in cyc_files:
                    # 파일명에서 메타데이터 추출
                    metadata = self._extract_pne_metadata(cyc_file)
                    metadata.update({
                        'channel': channel_num,
                        'path_idx': path_idx,
                        'path_name': os.path.basename(base_path),
                        'data_type': 'PNE',
                        'file_path': cyc_file
                    })
                    all_data.append(metadata)
                    print(f"  PNE 파일 발견: {os.path.basename(cyc_file)}")

        if all_data:
            print(f"\n주의: PNE 바이너리 파일은 현재 메타데이터만 추출됩니다.")
            print(f"총 {len(all_data)} 개의 PNE 파일 발견")

        return all_data

    def _extract_channel_number(self, dir_name: str) -> int:
        """디렉토리 이름에서 채널 번호 추출"""
        # M02Ch073[073] 형식에서 073 추출
        match = re.search(r'Ch(\d+)', dir_name)
        if match:
            return int(match.group(1))
        return 0

    def _extract_pne_metadata(self, file_path: str) -> Dict:
        """PNE 파일명에서 메타데이터 추출"""
        filename = os.path.basename(file_path)

        # 기본 메타데이터
        metadata = {
            'datetime': datetime.fromtimestamp(os.path.getmtime(file_path)),
            'filename': filename,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'elapsed_hours': 0,  # 추후 구현
            'voltage': 0,  # 추후 구현
            'current': 0,  # 추후 구현
            'temperature': 0,  # 추후 구현
            'cycle': 0,  # 추후 구현
            'total_cycle': 0  # 추후 구현
        }

        # 파일명에서 정보 추출 시도
        # 예: "231115 1파트 한정구 Gen4 2.0C ATL MP1차 Main 4500mAh 상온 ver1 0.5C 방전 0-601 ch73 74.cyc"
        # 날짜 추출
        date_match = re.search(r'(\d{6})', filename)
        if date_match:
            try:
                date_str = date_match.group(1)
                # YYMMDD 형식 파싱
                metadata['test_date'] = datetime.strptime(date_str, '%y%m%d')
            except:
                pass

        # 용량 추출
        capacity_match = re.search(r'(\d+)mAh', filename)
        if capacity_match:
            metadata['capacity_mah'] = int(capacity_match.group(1))

        # C-rate 추출
        crate_match = re.search(r'(\d+\.?\d*)C', filename)
        if crate_match:
            metadata['c_rate'] = float(crate_match.group(1))

        return metadata


class DataProcessorFactory:
    """데이터 프로세서 팩토리 클래스"""

    @staticmethod
    def detect_data_type(data_paths: List[str]) -> str:
        """데이터 타입 자동 감지"""
        if not data_paths:
            return "unknown"

        # 첫 번째 경로로 판단
        first_path = data_paths[0] if isinstance(data_paths, list) else data_paths

        if not os.path.exists(first_path):
            return "unknown"

        # 하위 디렉토리 패턴 확인
        subdirs = os.listdir(first_path)

        # 숫자로만 된 디렉토리 -> Toyo
        if any(d.isdigit() and os.path.isdir(os.path.join(first_path, d)) for d in subdirs):
            return "toyo"

        # M02Ch 패턴 -> PNE
        if any('Ch' in d and os.path.isdir(os.path.join(first_path, d)) for d in subdirs):
            return "pne"

        return "unknown"

    @staticmethod
    def create_processor(data_paths: List[str], data_type: str = None) -> BaseDataProcessor:
        """적절한 프로세서 생성"""

        # 자동 감지
        if data_type is None:
            data_type = DataProcessorFactory.detect_data_type(data_paths)
            print(f"감지된 데이터 형식: {data_type.upper()}")

        # 프로세서 생성
        if data_type == "toyo":
            return ToyoDataProcessor(data_paths)
        elif data_type == "pne":
            return PNEDataProcessor(data_paths)
        else:
            raise ValueError(f"지원되지 않는 데이터 형식: {data_type}")


def main():
    """메인 실행 함수"""

    # ========== 다양한 데이터 경로 케이스 ==========

    # Case 1: Toyo 다중경로 (연속 사이클)
    toyo_multi_continuous = [
        r"Rawdata\250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc",
        r"Rawdata\250219_250319_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 101-200cyc",
        r"Rawdata\250304_250404_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 201-300cyc",
        r"Rawdata\250317_251231_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 301-400cyc"
    ]

    # Case 2: PNE 다중경로
    pne_multi = [
        r"Rawdata\A1_MP1_4500mAh_T23_1",
        r"Rawdata\A1_MP1_4500mAh_T23_2",
        r"Rawdata\A1_MP1_4500mAh_T23_3"
    ]

    # Case 3: Toyo 단일경로들
    toyo_single_q7m_sub = [r"Rawdata\Q7M Sub ATL [45v 2068mAh] [23] - 250219r"]
    toyo_single_m1 = [r"Rawdata\M1 ATL [45V 4175mAh]"]
    toyo_single_q7m_inner = [r"Rawdata\Q7M Inner ATL_45V 1689mAh BLK1 20EA [23] - 250304"]
    toyo_single_q7m_main = [r"Rawdata\Q7M Main ATL [45V_1680mAh][23] blk7 20ea - 250228"]
    toyo_single_gen4 = [r"Rawdata\Gen4 2C ATL MP2 [45V 4470mAh] [23] blk7 - 240131"]

    # Case 4: PNE 단일경로들
    pne_single_1 = [r"Rawdata\A1_MP1_4500mAh_T23_1"]
    pne_single_2 = [r"Rawdata\A1_MP1_4500mAh_T23_2"]
    pne_single_3 = [r"Rawdata\A1_MP1_4500mAh_T23_3"]

    # Case 5: 김동진 데이터 부분 선택 (1-200 사이클)
    toyo_partial_1_200 = [
        r"Rawdata\250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc",
        r"Rawdata\250219_250319_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 101-200cyc"
    ]

    # Case 6: 김동진 데이터 부분 선택 (201-400 사이클)
    toyo_partial_201_400 = [
        r"Rawdata\250304_250404_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 201-300cyc",
        r"Rawdata\250317_251231_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 301-400cyc"
    ]

    # ========== 사용할 케이스 선택 ==========
    # 아래 중 하나를 선택하여 사용
    case_selection = {
        1: ("Toyo 다중경로 (연속 1-400 사이클)", toyo_multi_continuous),
        2: ("PNE 다중경로 (3개 셀)", pne_multi),
        3: ("Toyo 단일 - Q7M Sub", toyo_single_q7m_sub),
        4: ("Toyo 단일 - M1", toyo_single_m1),
        5: ("Toyo 단일 - Q7M Inner", toyo_single_q7m_inner),
        6: ("Toyo 단일 - Q7M Main", toyo_single_q7m_main),
        7: ("Toyo 단일 - Gen4", toyo_single_gen4),
        8: ("PNE 단일 - 셀1", pne_single_1),
        9: ("PNE 단일 - 셀2", pne_single_2),
        10: ("PNE 단일 - 셀3", pne_single_3),
        11: ("Toyo 부분 - 1-200 사이클", toyo_partial_1_200),
        12: ("Toyo 부분 - 201-400 사이클", toyo_partial_201_400),
    }

    # 여기서 케이스 번호를 변경하여 다른 데이터 테스트
    selected_case = 8  # 1~12 중 선택

    case_name, data_paths = case_selection[selected_case]
    print(f"\n========== 선택된 케이스: {case_name} ==========\n")

    try:
        # 자동 감지 및 처리
        processor = DataProcessorFactory.create_processor(data_paths)
        df = processor.process()

        # 결과 확인
        if not df.empty:
            print("\n=== 데이터 요약 ===")
            print(f"총 데이터 포인트: {len(df)}")
            print(f"컬럼: {', '.join(df.columns)}")

            if 'datetime' in df.columns:
                print(f"시작 시간: {df['datetime'].min()}")
                print(f"종료 시간: {df['datetime'].max()}")

            if 'voltage' in df.columns and df['voltage'].sum() > 0:
                print(f"전압 범위: {df['voltage'].min():.4f} ~ {df['voltage'].max():.4f} V")

            if 'channel' in df.columns:
                print(f"채널: {df['channel'].unique()}")

            # CSV 저장
            processor.save_to_csv()

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()