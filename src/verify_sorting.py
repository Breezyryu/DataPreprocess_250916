"""
정렬된 데이터 검증 스크립트
시계열 데이터가 올바르게 정렬되었는지 확인합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

def verify_sorting(file_path: str):
    """
    데이터 정렬 상태 검증

    Args:
        file_path: 검증할 CSV 파일 경로
    """
    print(f"=== 데이터 정렬 검증: {Path(file_path).name} ===")

    # 데이터 로드
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"총 데이터 수: {len(df):,} 행")

    # DateTime 생성 및 검증
    if 'Date' in df.columns and 'Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

        # 시간 순서 확인
        is_time_sorted = df['DateTime'].is_monotonic_increasing
        print(f"시간 순서 정렬 상태: {'[OK] 정렬됨' if is_time_sorted else '[NG] 비정렬'}")

        # 날짜/시간 범위
        print(f"시작 시간: {df['DateTime'].min()}")
        print(f"종료 시간: {df['DateTime'].max()}")
        print(f"총 기간: {(df['DateTime'].max() - df['DateTime'].min()).days} 일")

        # 처음 10개와 마지막 10개 데이터 확인
        print("\n처음 10개 데이터:")
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            print(f"  {i+1:2d}: {row['Date']} {row['Time']} | TotalCycle: {row.get('TotalCycle', 'N/A')}")

        print("\n마지막 10개 데이터:")
        for i in range(max(0, len(df)-10), len(df)):
            row = df.iloc[i]
            print(f"  {i+1:2d}: {row['Date']} {row['Time']} | TotalCycle: {row.get('TotalCycle', 'N/A')}")

    # TotalCycle 순서 확인
    if 'TotalCycle' in df.columns:
        is_cycle_sorted = df['TotalCycle'].is_monotonic_increasing
        print(f"\nTotalCycle 순서 정렬 상태: {'[OK] 정렬됨' if is_cycle_sorted else '[NG] 비정렬'}")
        print(f"TotalCycle 범위: {df['TotalCycle'].min()} ~ {df['TotalCycle'].max()}")

        # TotalCycle 연속성 확인
        cycle_diff = df['TotalCycle'].diff().dropna()
        non_sequential = cycle_diff[cycle_diff < 0]
        if len(non_sequential) > 0:
            print(f"[주의] 역순 TotalCycle 발견: {len(non_sequential)} 개")
        else:
            print("[OK] TotalCycle 순차적")

    # 채널별 데이터 분포
    if 'Channel' in df.columns:
        print(f"\n채널별 데이터 분포:")
        channel_counts = df['Channel'].value_counts().sort_index()
        for channel, count in channel_counts.items():
            print(f"  {channel}: {count:,} 개")

    # 데이터 품질 확인
    print(f"\n데이터 품질:")
    print(f"  결측값: {df.isnull().sum().sum()} 개")
    print(f"  중복 행: {df.duplicated().sum()} 개")

    if 'Voltage_V' in df.columns:
        voltage_range = f"{df['Voltage_V'].min():.3f} ~ {df['Voltage_V'].max():.3f} V"
        print(f"  전압 범위: {voltage_range}")

    if 'Current_A' in df.columns:
        current_range = f"{df['Current_A'].min():.3f} ~ {df['Current_A'].max():.3f} A"
        print(f"  전류 범위: {current_range}")

    print("=" * 60)

def compare_before_after():
    """정렬 전후 비교"""
    original_file = "250207_250307_3_김동진_1689mAh_ATL_Q7M_Inner_2C_상온수명_processed.csv"
    sorted_file = "250207_250307_3_김동진_1689mAh_ATL_Q7M_Inner_2C_상온수명_processed_sorted.csv"

    if Path(original_file).exists() and Path(sorted_file).exists():
        print("[비교] 정렬 전후 비교")
        print("\n" + "="*60)
        print("정렬 전 데이터:")
        verify_sorting(original_file)

        print("\n정렬 후 데이터:")
        verify_sorting(sorted_file)

        # 데이터 개수 비교
        df1 = pd.read_csv(original_file)
        df2 = pd.read_csv(sorted_file)

        print(f"\n[요약]:")
        print(f"  정렬 전 데이터 수: {len(df1):,} 행")
        print(f"  정렬 후 데이터 수: {len(df2):,} 행")
        print(f"  데이터 손실: {len(df1) - len(df2)} 행")

        if len(df1) == len(df2):
            print("[OK] 데이터 개수 일치 - 정렬만 수행됨")
        else:
            print("[주의] 데이터 개수 불일치")
    else:
        print("파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    compare_before_after()