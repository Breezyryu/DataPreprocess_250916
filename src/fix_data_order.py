"""
배터리 데이터 순서 정렬 스크립트
시계열 데이터의 순서를 Date + Time 기준으로 정렬합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_data_order(input_file: str, output_file: str = None):
    """
    시계열 데이터 순서 정렬

    Args:
        input_file: 입력 CSV 파일 경로
        output_file: 출력 CSV 파일 경로 (None이면 원본에 '_sorted' 추가)
    """
    try:
        logger.info(f"데이터 로드 중: {input_file}")

        # 데이터 로드
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        logger.info(f"로드된 데이터: {len(df)} 행")

        # 원본 데이터 정보 출력
        logger.info("=== 정렬 전 데이터 상태 ===")
        if 'Date' in df.columns and 'Time' in df.columns:
            logger.info("첫 5개 날짜/시간:")
            for i in range(min(5, len(df))):
                logger.info(f"  {i+1}: {df.iloc[i]['Date']} {df.iloc[i]['Time']}")

        # DateTime 컬럼 생성 및 정렬
        if 'Date' in df.columns and 'Time' in df.columns:
            logger.info("DateTime 컬럼 생성 중...")

            # 안전한 DateTime 변환
            try:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

                # NaT 값 확인
                nat_count = df['DateTime'].isna().sum()
                if nat_count > 0:
                    logger.warning(f"유효하지 않은 날짜/시간 {nat_count}개 발견")

                # DateTime 기준 정렬
                logger.info("DateTime 기준 정렬 중...")
                df_sorted = df.sort_values('DateTime', na_position='last').reset_index(drop=True)

                # DateTime 컬럼 제거 (임시 컬럼이었으므로)
                df_sorted = df_sorted.drop('DateTime', axis=1)

            except Exception as e:
                logger.warning(f"DateTime 변환 실패: {e}")
                logger.info("TotalCycle 기준 정렬로 대체...")

                if 'TotalCycle' in df.columns:
                    df_sorted = df.sort_values('TotalCycle').reset_index(drop=True)
                else:
                    logger.warning("TotalCycle 컬럼도 없음. 원본 순서 유지")
                    df_sorted = df.copy()

        else:
            logger.warning("Date 또는 Time 컬럼 없음")

            # TotalCycle 기준 정렬 시도
            if 'TotalCycle' in df.columns:
                logger.info("TotalCycle 기준 정렬...")
                df_sorted = df.sort_values('TotalCycle').reset_index(drop=True)
            else:
                logger.warning("정렬 가능한 컬럼 없음. 원본 순서 유지")
                df_sorted = df.copy()

        # 정렬 결과 검증
        logger.info("=== 정렬 후 데이터 상태 ===")
        if 'Date' in df_sorted.columns and 'Time' in df_sorted.columns:
            logger.info("첫 5개 날짜/시간:")
            for i in range(min(5, len(df_sorted))):
                logger.info(f"  {i+1}: {df_sorted.iloc[i]['Date']} {df_sorted.iloc[i]['Time']}")

        if 'TotalCycle' in df_sorted.columns:
            logger.info("첫 5개 TotalCycle:")
            for i in range(min(5, len(df_sorted))):
                logger.info(f"  {i+1}: {df_sorted.iloc[i]['TotalCycle']}")

        # 출력 파일명 결정
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_sorted{input_path.suffix}"

        # 정렬된 데이터 저장
        logger.info(f"정렬된 데이터 저장 중: {output_file}")
        df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')

        logger.info("=== 정렬 완료 ===")
        logger.info(f"입력 파일: {input_file}")
        logger.info(f"출력 파일: {output_file}")
        logger.info(f"처리된 데이터: {len(df_sorted)} 행")

        return df_sorted

    except Exception as e:
        logger.error(f"데이터 정렬 실패: {e}")
        raise

def main():
    """메인 실행 함수"""
    # 처리할 파일 경로
    input_file = "250207_250307_3_김동진_1689mAh_ATL_Q7M_Inner_2C_상온수명_processed.csv"

    # 파일 존재 확인
    if not Path(input_file).exists():
        logger.error(f"파일을 찾을 수 없습니다: {input_file}")
        return

    try:
        # 데이터 정렬 실행
        sorted_df = fix_data_order(input_file)

        # 통계 정보 출력
        print("\n=== 정렬 결과 통계 ===")
        if 'TotalCycle' in sorted_df.columns:
            print(f"TotalCycle 범위: {sorted_df['TotalCycle'].min()} ~ {sorted_df['TotalCycle'].max()}")

        if 'Date' in sorted_df.columns:
            print(f"날짜 범위: {sorted_df['Date'].min()} ~ {sorted_df['Date'].max()}")

        print("정렬 완료!")

    except Exception as e:
        logger.error(f"실행 실패: {e}")

if __name__ == "__main__":
    main()