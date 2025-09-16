"""
배터리 데이터 누적시간 기반 시각화 스크립트
채널별 커브 구분을 위한 누적시간 열 추가 및 플롯 생성
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_cumulative_time_column(df, reset_on_cycle=True):
    """
    누적시간 컬럼 생성

    Args:
        df: 배터리 데이터 DataFrame
        reset_on_cycle: 사이클마다 시간을 리셋할지 여부

    Returns:
        DataFrame with added Cumulative_Time_Hours column
    """
    logger.info("누적시간 컬럼 생성 중...")

    df_copy = df.copy()

    # DateTime 생성
    if 'Date' in df_copy.columns and 'Time' in df_copy.columns:
        df_copy['DateTime'] = pd.to_datetime(df_copy['Date'] + ' ' + df_copy['Time'])

    if reset_on_cycle and 'TotalCycle' in df_copy.columns:
        # 사이클별로 누적시간 계산
        logger.info("사이클별 누적시간 계산...")

        cumulative_hours = []
        current_cumulative = 0.0

        for cycle in sorted(df_copy['TotalCycle'].unique()):
            cycle_data = df_copy[df_copy['TotalCycle'] == cycle].copy()

            if len(cycle_data) > 0:
                # 사이클 내에서의 경과시간 계산
                if 'DateTime' in cycle_data.columns:
                    start_time = cycle_data['DateTime'].min()
                    cycle_hours = [(dt - start_time).total_seconds() / 3600.0
                                 for dt in cycle_data['DateTime']]
                else:
                    # Time_Sec 사용
                    min_time = cycle_data['Time_Sec'].min()
                    cycle_hours = [(t - min_time) / 3600.0 for t in cycle_data['Time_Sec']]

                # 이전 사이클의 누적시간에 추가
                cycle_cumulative = [current_cumulative + h for h in cycle_hours]
                cumulative_hours.extend(cycle_cumulative)

                # 다음 사이클을 위해 누적시간 업데이트
                if cycle_hours:
                    current_cumulative += max(cycle_hours)

        df_copy['Cumulative_Time_Hours'] = cumulative_hours

    else:
        # 전체 기간에 대한 연속 누적시간
        logger.info("전체 기간 누적시간 계산...")

        if 'DateTime' in df_copy.columns:
            start_time = df_copy['DateTime'].min()
            df_copy['Cumulative_Time_Hours'] = [
                (dt - start_time).total_seconds() / 3600.0
                for dt in df_copy['DateTime']
            ]
        else:
            min_time = df_copy['Time_Sec'].min()
            df_copy['Cumulative_Time_Hours'] = (df_copy['Time_Sec'] - min_time) / 3600.0

    logger.info(f"누적시간 범위: 0 ~ {df_copy['Cumulative_Time_Hours'].max():.1f} 시간")
    return df_copy

def plot_channels_with_cumulative_time(channel_files, output_dir=None, max_points_per_channel=5000):
    """
    채널별 누적시간 기반 시각화

    Args:
        channel_files: {channel_name: file_path} 딕셔너리
        output_dir: 출력 디렉토리
        max_points_per_channel: 채널당 최대 표시 포인트 수
    """
    logger.info(f"채널별 누적시간 플롯 생성 중... (최대 {max_points_per_channel}개 포인트/채널)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('채널별 배터리 성능 - 누적시간 기준', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (channel, file_path) in enumerate(channel_files.items()):
        try:
            logger.info(f"  {channel} 데이터 처리 중...")

            # 데이터 로드
            df = pd.read_csv(file_path, encoding='utf-8-sig')

            # 누적시간 컬럼 생성
            df_with_cumtime = create_cumulative_time_column(df, reset_on_cycle=False)

            # 데이터 샘플링 (너무 많으면 성능 저하)
            if len(df_with_cumtime) > max_points_per_channel:
                step = len(df_with_cumtime) // max_points_per_channel
                df_sample = df_with_cumtime.iloc[::step].copy()
                logger.info(f"    샘플링: {len(df_with_cumtime)} → {len(df_sample)} 포인트")
            else:
                df_sample = df_with_cumtime.copy()

            color = colors[i % len(colors)]
            alpha = 0.8
            linewidth = 1.5

            # 1. 전압 vs 누적시간
            axes[0, 0].plot(df_sample['Cumulative_Time_Hours'], df_sample['Voltage_V'],
                           color=color, alpha=alpha, linewidth=linewidth, label=channel)

            # 2. 전류 vs 누적시간
            axes[0, 1].plot(df_sample['Cumulative_Time_Hours'], df_sample['Current_A'],
                           color=color, alpha=alpha, linewidth=linewidth, label=channel)

            # 3. 용량 vs 누적시간
            if 'Capacity_Ah' in df_sample.columns:
                axes[1, 0].plot(df_sample['Cumulative_Time_Hours'], df_sample['Capacity_Ah'],
                               color=color, alpha=alpha, linewidth=linewidth, label=channel)

            # 4. SOC vs 누적시간
            if 'SOC_%' in df_sample.columns:
                axes[1, 1].plot(df_sample['Cumulative_Time_Hours'], df_sample['SOC_%'],
                               color=color, alpha=alpha, linewidth=linewidth, label=channel)

            logger.info(f"    완료: {len(df_sample)} 포인트, 누적시간 {df_sample['Cumulative_Time_Hours'].max():.1f}h")

        except Exception as e:
            logger.error(f"  {channel} 처리 실패: {e}")

    # 축 설정 및 스타일링
    plot_configs = [
        ('전압 vs 누적시간', '누적시간 (hours)', '전압 (V)', (0, 0)),
        ('전류 vs 누적시간', '누적시간 (hours)', '전류 (A)', (0, 1)),
        ('용량 vs 누적시간', '누적시간 (hours)', '용량 (Ah)', (1, 0)),
        ('SOC vs 누적시간', '누적시간 (hours)', 'SOC (%)', (1, 1))
    ]

    for title, xlabel, ylabel, (row, col) in plot_configs:
        ax = axes[row, col]
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 축 범위 자동 조정
        ax.margins(x=0.01, y=0.02)

    plt.tight_layout()

    # 저장
    if output_dir:
        output_path = Path(output_dir) / 'channel_comparison_cumulative_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"그래프 저장: {output_path}")

    plt.show()

    return fig

def create_cycle_separated_plot(channel_files, output_dir=None, cycles_to_show=5):
    """
    사이클별 분리된 플롯 생성

    Args:
        channel_files: {channel_name: file_path} 딕셔너리
        output_dir: 출력 디렉토리
        cycles_to_show: 표시할 사이클 수
    """
    logger.info(f"사이클별 분리 플롯 생성 중... (최대 {cycles_to_show}개 사이클)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('채널별 초기 사이클 비교 (사이클별 분리)', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (channel, file_path) in enumerate(channel_files.items()):
        try:
            logger.info(f"  {channel} 사이클별 데이터 처리 중...")

            df = pd.read_csv(file_path, encoding='utf-8-sig')
            color = colors[i % len(colors)]

            # 초기 사이클들만 선택
            if 'TotalCycle' in df.columns:
                unique_cycles = sorted(df['TotalCycle'].unique())[:cycles_to_show]

                for cycle_idx, total_cycle in enumerate(unique_cycles):
                    cycle_data = df[df['TotalCycle'] == total_cycle].copy()

                    if len(cycle_data) < 10:  # 너무 적은 데이터는 스킵
                        continue

                    # 사이클 내 시간 정규화 (0부터 시작)
                    cycle_data['Cycle_Time_Hours'] = (cycle_data['Time_Sec'] - cycle_data['Time_Sec'].min()) / 3600.0

                    # 플롯
                    alpha = 0.7 - (cycle_idx * 0.1)  # 사이클마다 투명도 조정
                    linewidth = 2.0 - (cycle_idx * 0.2)

                    # 전압
                    axes[0, 0].plot(cycle_data['Cycle_Time_Hours'], cycle_data['Voltage_V'],
                                   color=color, alpha=alpha, linewidth=linewidth,
                                   label=f'{channel}-C{total_cycle}' if cycle_idx == 0 else "")

                    # 전류
                    axes[0, 1].plot(cycle_data['Cycle_Time_Hours'], cycle_data['Current_A'],
                                   color=color, alpha=alpha, linewidth=linewidth)

                    # 용량
                    if 'Capacity_Ah' in cycle_data.columns:
                        axes[0, 2].plot(cycle_data['Cycle_Time_Hours'], cycle_data['Capacity_Ah'],
                                       color=color, alpha=alpha, linewidth=linewidth)

                    # 온도
                    if 'Temperature_C' in cycle_data.columns:
                        axes[1, 0].plot(cycle_data['Cycle_Time_Hours'], cycle_data['Temperature_C'],
                                       color=color, alpha=alpha, linewidth=linewidth)

                    # SOC
                    if 'SOC_%' in cycle_data.columns:
                        axes[1, 1].plot(cycle_data['Cycle_Time_Hours'], cycle_data['SOC_%'],
                                       color=color, alpha=alpha, linewidth=linewidth)

                    # 파워
                    if 'Power_W' in cycle_data.columns:
                        axes[1, 2].plot(cycle_data['Cycle_Time_Hours'], cycle_data['Power_W'],
                                       color=color, alpha=alpha, linewidth=linewidth)

                logger.info(f"    완료: {len(unique_cycles)}개 사이클")

        except Exception as e:
            logger.error(f"  {channel} 사이클별 처리 실패: {e}")

    # 축 설정
    plot_configs = [
        ('전압 vs 사이클시간', '사이클 시간 (hours)', '전압 (V)', (0, 0)),
        ('전류 vs 사이클시간', '사이클 시간 (hours)', '전류 (A)', (0, 1)),
        ('용량 vs 사이클시간', '사이클 시간 (hours)', '용량 (Ah)', (0, 2)),
        ('온도 vs 사이클시간', '사이클 시간 (hours)', '온도 (°C)', (1, 0)),
        ('SOC vs 사이클시간', '사이클 시간 (hours)', 'SOC (%)', (1, 1)),
        ('전력 vs 사이클시간', '사이클 시간 (hours)', '전력 (W)', (1, 2))
    ]

    for title, xlabel, ylabel, (row, col) in plot_configs:
        ax = axes[row, col]
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if row == 0 and col == 0:  # 첫 번째 플롯에만 범례
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.margins(x=0.01, y=0.02)

    plt.tight_layout()

    # 저장
    if output_dir:
        output_path = Path(output_dir) / 'channel_cycles_separated.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"사이클별 그래프 저장: {output_path}")

    plt.show()

    return fig

def main():
    """메인 실행 함수"""
    # 최신 출력 폴더에서 채널 파일 찾기
    base_dir = Path(__file__).parent.parent  # src의 상위 폴더
    outputs_dir = base_dir / 'outputs'

    if not outputs_dir.exists():
        logger.error("outputs 폴더를 찾을 수 없습니다.")
        return

    # 최신 run 폴더 찾기
    run_folders = sorted(outputs_dir.glob('run_*'), reverse=True)
    if not run_folders:
        logger.error("run_ 폴더를 찾을 수 없습니다.")
        return

    latest_run = run_folders[0]
    channels_dir = latest_run / 'channels'
    plots_dir = latest_run / 'plots'

    if not channels_dir.exists():
        logger.error(f"channels 폴더를 찾을 수 없습니다: {channels_dir}")
        return

    # 채널 파일 찾기
    channel_files = {}
    for csv_file in channels_dir.glob('*.csv'):
        if '_Ch' in csv_file.stem:
            channel = 'Ch' + csv_file.stem.split('_Ch')[-1]
            channel_files[channel] = csv_file

    if not channel_files:
        logger.error("채널 파일을 찾을 수 없습니다.")
        return

    logger.info(f"발견된 채널: {list(channel_files.keys())}")

    # 플롯 생성
    try:
        # 1. 누적시간 기반 플롯
        logger.info("=== 누적시간 기반 플롯 생성 ===")
        plot_channels_with_cumulative_time(channel_files, plots_dir)

        # 2. 사이클별 분리 플롯
        logger.info("=== 사이클별 분리 플롯 생성 ===")
        create_cycle_separated_plot(channel_files, plots_dir, cycles_to_show=3)

        logger.info("모든 플롯 생성 완료!")

    except Exception as e:
        logger.error(f"플롯 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()