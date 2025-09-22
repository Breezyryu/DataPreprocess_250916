import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 경로 목록
'''
data_paths = [
    r"Rawdata\250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc",
    r"Rawdata\250219_250319_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 101-200cyc",
    r"Rawdata\250304_250404_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 201-300cyc",
    r"Rawdata\250317_251231_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 301-400cyc"
]
'''

data_paths = [
    r"Rawdata\250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc",
    r"Rawdata\250219_250319_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 101-200cyc",
    r"Rawdata\250304_250404_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 201-300cyc",
    r"Rawdata\250317_251231_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 301-400cyc"
]


# 모든 데이터 포인트와 모든 컬럼 수집
all_data = []
first_datetime = None
last_datetime = None  # 마지막 시간 추적
last_elapsed = 0  # 마지막 elapsed time 추적
total_data_points = 0
files_processed = 0

print("다중 경로 데이터 처리 시작...")
print(f"처리할 경로 수: {len(data_paths)}\n")

# 각 경로별로 처리
for path_idx, base_path in enumerate(data_paths, 1):
    print(f"\n[{path_idx}/{len(data_paths)}] 처리 중: {os.path.basename(base_path)}")

    # path별 첫 번째 데이터의 실제 시간을 저장할 변수
    path_first_datetime = None

    # 각 경로에서 존재하는 하위 디렉토리 찾기
    subdirs_to_check = ['30', '31']  # 기본 디렉토리
    existing_subdirs = []

    # 경로가 존재하는지 먼저 확인
    if os.path.exists(base_path):
        # 실제 존재하는 하위 디렉토리 찾기
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                existing_subdirs.append(item)

        if not existing_subdirs:
            # 하위 디렉토리가 없으면 기본 디렉토리 체크
            for subdir in subdirs_to_check:
                if os.path.exists(os.path.join(base_path, subdir)):
                    existing_subdirs.append(subdir)
    else:
        print(f"  경로가 존재하지 않음: {base_path}")
        continue

    if not existing_subdirs:
        print(f"  하위 디렉토리를 찾을 수 없음")
        continue

    print(f"  발견된 하위 디렉토리: {', '.join(sorted(existing_subdirs))}")

    # 하위 디렉토리 탐색
    for subdir in sorted(existing_subdirs):
        subdir_path = os.path.join(base_path, subdir)
        if not os.path.exists(subdir_path):
            continue

        # 각 하위 디렉토리의 파일들 처리
        for i in range(1, 600):  # 충분한 범위로 설정
            file_path = os.path.join(subdir_path, f"{i:06d}")
            if os.path.exists(file_path):
                files_processed += 1
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                # 헤더 라인 파싱 (4번째 줄)
                if len(lines) > 3:
                    header_line = lines[3]
                    # Date,Time,PassTime[Sec],Voltage[V],Current[mA],,,Temp1[Deg],,,,Condition,Mode,Cycle,TotlCycle,Temp1[Deg]

                # 모든 데이터 라인 처리
                for line in lines[4:]:
                    if '2025/' in line:  # 2025년 모든 월 포함
                        parts = line.strip().split(',')
                        if len(parts) >= 16:  # 모든 필드가 있는지 확인
                            try:
                                # 날짜와 시간
                                date_str = parts[0]
                                time_str = parts[1]
                                datetime_str = f"{date_str} {time_str}"
                                current_datetime = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")

                                # 전체 데이터의 첫 번째 시간 저장
                                if first_datetime is None:
                                    first_datetime = current_datetime

                                # 현재 path의 첫 번째 데이터 시간 저장
                                if path_first_datetime is None:
                                    path_first_datetime = current_datetime

                                # 임시로 elapsed_time은 0으로 설정 (나중에 재계산)
                                elapsed_time = 0

                                # 모든 필드 파싱
                                passtime_sec = int(parts[2])
                                voltage = float(parts[3].replace('+', ''))
                                current = float(parts[4].replace('+', '')) if parts[4] else 0.0
                                # parts[5], [6]는 빈 필드
                                temp1_first = float(parts[7].replace('+', '')) if parts[7] else 0.0
                                # parts[8], [9], [10]은 빈 필드
                                condition = int(parts[11].strip())
                                mode = int(parts[12].strip())
                                cycle = int(parts[13].strip())
                                total_cycle = int(parts[14].strip())
                                temp1_second = float(parts[15].replace('+', '')) if len(parts) > 15 else 0.0

                                all_data.append({
                                    'path_name': os.path.basename(base_path),
                                    'path_idx': path_idx,  # 경로 순서 저장
                                    'channel': int(subdir),  # 채널 번호 저장
                                    'file_num': i,
                                    'subdir': subdir,
                                    'date': date_str,
                                    'time': time_str,
                                    'datetime': current_datetime,
                                    'passtime_sec': passtime_sec,
                                    'elapsed_hours': elapsed_time,
                                    'voltage': voltage,
                                    'current': current,
                                    'temperature': temp1_first,  # 첫 번째 온도 값 사용
                                    'condition': condition,
                                    'mode': mode,
                                    'cycle': cycle,
                                    'total_cycle': total_cycle
                                })
                                total_data_points += 1

                            except (ValueError, IndexError) as e:
                                # 파싱 에러는 무시하고 계속
                                continue

            # 진행 상황 표시
            if files_processed % 50 == 0:
                print(f"  처리 중... {files_processed} 파일 완료 (경로 {path_idx}, 디렉토리 {subdir})")

    # 현재 path 처리 완료 후 마지막 elapsed time 업데이트
    if all_data:
        # 현재 path의 마지막 elapsed_time을 저장
        path_data = [d for d in all_data if d['path_idx'] == path_idx]
        if path_data:
            last_elapsed = max(d['elapsed_hours'] for d in path_data)

print(f"\n처리 완료!")
print(f"총 읽은 파일 수: {files_processed}")
print(f"총 데이터 포인트: {total_data_points}")

if all_data:
    # DataFrame 생성
    df = pd.DataFrame(all_data)

    # 기본 통계 정보
    print(f"\n=== 데이터 요약 ===")
    print(f"시작 시간: {df['datetime'].min()}")
    print(f"종료 시간: {df['datetime'].max()}")
    print(f"총 경과 시간: {df['elapsed_hours'].max():.2f} hours")
    print(f"\n전압 범위: {df['voltage'].min():.4f} ~ {df['voltage'].max():.4f} V")
    print(f"전류 범위: {df['current'].min():.1f} ~ {df['current'].max():.1f} mA")
    print(f"온도 범위: {df['temperature'].min():.1f} ~ {df['temperature'].max():.1f} °C")

    # Condition별 데이터 수
    print(f"\nCondition별 데이터 포인트:")
    for cond in sorted(df['condition'].unique()):
        count = len(df[df['condition'] == cond])
        print(f"  Condition {cond}: {count} points")

    # Mode별 데이터 수
    print(f"\nMode별 데이터 포인트:")
    for mode in sorted(df['mode'].unique()):
        count = len(df[df['mode'] == mode])
        print(f"  Mode {mode}: {count} points")

    # Cycle 정보
    print(f"\n사이클 정보:")
    print(f"  총 사이클 수: {df['total_cycle'].max()}")
    print(f"  사이클 범위: {df['cycle'].min()} ~ {df['cycle'].max()}")

    # 경로별 순차 정렬 (path_idx 기준)
    df = df.sort_values(['path_idx', 'channel', 'datetime'])

    # elapsed_hours를 연속적으로 재계산 (Path 간 공백 없이)
    print("\nElapsed time을 연속적으로 재계산 중 (Path 간 공백 제거)...")

    # 각 path의 첫 번째와 마지막 데이터 시간 찾기
    path_times = {}
    for path_idx in sorted(df['path_idx'].unique()):
        path_data = df[df['path_idx'] == path_idx]
        path_times[path_idx] = {
            'first': path_data['datetime'].min(),
            'last': path_data['datetime'].max()
        }

    # 전체 데이터 기준 시작 시간
    global_start = df['datetime'].min()

    # Path 간 실제 시간 간격 확인 및 flag 표시
    print("\n=== Path 간 실제 시간 간격 분석 ===")
    for i in range(1, len(path_times)):
        if i in path_times and (i+1) in path_times:
            gap_hours = (path_times[i+1]['first'] - path_times[i]['last']).total_seconds() / 3600.0
            if gap_hours > 1.0:  # 1시간 이상 차이가 있으면 표시
                print(f"⚠️ Path {i}와 Path {i+1} 사이 실제 시간 간격: {gap_hours:.2f} hours")
                print(f"   Path {i} 종료: {path_times[i]['last']}")
                print(f"   Path {i+1} 시작: {path_times[i+1]['first']}")

    # 각 path별 누적 시간 계산 (연속적으로, 공백 없이)
    cumulative_hours = 0
    for path_idx in sorted(df['path_idx'].unique()):
        path_mask = df['path_idx'] == path_idx
        path_data = df[path_mask].copy()

        if path_idx == 1:
            # 첫 번째 path는 0부터 시작
            df.loc[path_mask, 'elapsed_hours'] = (
                (df.loc[path_mask, 'datetime'] - path_times[path_idx]['first']).dt.total_seconds() / 3600.0
            )
        else:
            # 이전 path의 마지막 elapsed_hours 값 가져오기
            prev_path_data = df[df['path_idx'] == path_idx - 1]
            prev_max_elapsed = prev_path_data['elapsed_hours'].max()

            # 현재 path의 시작 시간
            path_start = path_times[path_idx]['first']

            # 현재 path 내에서의 상대 시간을 이전 누적 시간에 바로 이어서 추가
            # (실제 시간 간격 무시하고 연속적으로 처리)
            # Path 간 작은 간격(0.01 hours)을 추가하여 완전히 연속적으로 만듦
            df.loc[path_mask, 'elapsed_hours'] = (
                prev_max_elapsed + 0.01 +  # Path 간 최소 간격 추가
                (df.loc[path_mask, 'datetime'] - path_start).dt.total_seconds() / 3600.0
            )

    print(f"\nElapsed time 재계산 완료 (연속 처리): 총 {df['elapsed_hours'].max():.2f} hours")
    print("※ 실제 시간 간격이 있는 경우 위에 ⚠️ 표시됨")

    # 채널별 데이터 분리
    ch30_data = df[df['channel'] == 30]
    ch31_data = df[df['channel'] == 31]

    print(f"\n채널 30 데이터: {len(ch30_data)} points")
    print(f"채널 31 데이터: {len(ch31_data)} points")

    # 전체 데이터 시각화 (채널 구분)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Complete Raw Data Analysis (Continuous Time) - Total {total_data_points} points (Ch30: {len(ch30_data)}, Ch31: {len(ch31_data)})', fontsize=14)

    # 1. Voltage vs Time (채널별 구분)
    ax1 = axes[0, 0]
    # 경로별 색상 정의
    path_colors = ['blue', 'green', 'orange', 'red']
    for idx in range(1, 5):
        path_data_30 = df[(df['path_idx'] == idx) & (df['channel'] == 30)]
        path_data_31 = df[(df['path_idx'] == idx) & (df['channel'] == 31)]
        if not path_data_30.empty:
            ax1.scatter(path_data_30['elapsed_hours'], path_data_30['voltage'],
                       c=path_colors[idx-1], s=0.5, alpha=0.5, label=f'Path{idx}-Ch30')
        if not path_data_31.empty:
            ax1.scatter(path_data_31['elapsed_hours'], path_data_31['voltage'],
                       c=path_colors[idx-1], s=0.5, alpha=0.3, marker='^', label=f'Path{idx}-Ch31')
    ax1.set_xlabel('Elapsed Time (hours)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage Profile (by Path and Channel)')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Current vs Time (채널별 구분)
    ax2 = axes[0, 1]
    ax2.scatter(ch30_data['elapsed_hours'], ch30_data['current'],
               c='blue', s=0.5, alpha=0.5, label='Channel 30')
    ax2.scatter(ch31_data['elapsed_hours'], ch31_data['current'],
               c='red', s=0.5, alpha=0.5, label='Channel 31')
    ax2.set_xlabel('Elapsed Time (hours)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('Current Profile (by Channel)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Temperature vs Time (채널별 구분)
    ax3 = axes[1, 0]
    ax3.plot(ch30_data['elapsed_hours'], ch30_data['temperature'],
             'b-', linewidth=0.5, alpha=0.7, label='Channel 30')
    ax3.plot(ch31_data['elapsed_hours'], ch31_data['temperature'],
             'r-', linewidth=0.5, alpha=0.7, label='Channel 31')
    ax3.set_xlabel('Elapsed Time (hours)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Temperature Profile (by Channel)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Cycle progression (경로별 순차 표시)
    ax4 = axes[1, 1]
    for idx in range(1, 5):
        path_data = df[df['path_idx'] == idx]
        if not path_data.empty:
            # 각 경로의 첫 번째 채널 데이터만 사용 (중복 방지)
            path_ch30 = path_data[path_data['channel'] == 30]
            if not path_ch30.empty:
                ax4.plot(path_ch30['elapsed_hours'], path_ch30['total_cycle'],
                        linewidth=1, label=f'Path {idx}', color=path_colors[idx-1])
    ax4.set_xlabel('Elapsed Time (hours)')
    ax4.set_ylabel('Total Cycle Number')
    ax4.set_title('Cycle Progression (by Path)')
    if ax4.get_legend_handles_labels()[0]:  # 범례가 있는 경우만
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Voltage distribution by channel
    ax5 = axes[2, 0]
    ax5.hist(ch30_data['voltage'], bins=50, alpha=0.5, color='blue', label='Channel 30')
    ax5.hist(ch31_data['voltage'], bins=50, alpha=0.5, color='red', label='Channel 31')
    ax5.set_xlabel('Voltage (V)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Voltage Distribution by Channel')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Data points per path and channel
    ax6 = axes[2, 1]
    path_channel_counts = df.groupby(['path_idx', 'channel']).size().unstack(fill_value=0)
    x = np.arange(len(path_channel_counts))
    width = 0.35

    if 30 in path_channel_counts.columns:
        ax6.bar(x - width/2, path_channel_counts[30], width, label='Channel 30', color='blue', alpha=0.7)
    if 31 in path_channel_counts.columns:
        ax6.bar(x + width/2, path_channel_counts[31], width, label='Channel 31', color='red', alpha=0.7)

    ax6.set_xlabel('Path')
    ax6.set_ylabel('Data Points')
    ax6.set_title('Data Points per Path and Channel')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'Path {i}' for i in path_channel_counts.index])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/complete_multichannel_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()

    print("\n그래프가 'outputs/complete_multichannel_analysis.png'에 저장되었습니다.")

    # 채널별 개별 시각화
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Channel-Specific Analysis', fontsize=14)

    # Ch30 Voltage
    ax1 = axes2[0, 0]
    for idx in range(1, 5):
        path_data = ch30_data[ch30_data['path_idx'] == idx]
        if not path_data.empty:
            ax1.plot(path_data['elapsed_hours'], path_data['voltage'],
                    linewidth=0.5, alpha=0.7, label=f'Path {idx}', color=path_colors[idx-1])
    ax1.set_xlabel('Elapsed Time (hours)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Channel 30 - Voltage Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ch31 Voltage
    ax2 = axes2[0, 1]
    for idx in range(1, 5):
        path_data = ch31_data[ch31_data['path_idx'] == idx]
        if not path_data.empty:
            ax2.plot(path_data['elapsed_hours'], path_data['voltage'],
                    linewidth=0.5, alpha=0.7, label=f'Path {idx}', color=path_colors[idx-1])
    ax2.set_xlabel('Elapsed Time (hours)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title('Channel 31 - Voltage Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Ch30 Current
    ax3 = axes2[1, 0]
    for idx in range(1, 5):
        path_data = ch30_data[ch30_data['path_idx'] == idx]
        if not path_data.empty:
            ax3.scatter(path_data['elapsed_hours'], path_data['current'],
                       s=0.5, alpha=0.5, label=f'Path {idx}', color=path_colors[idx-1])
    ax3.set_xlabel('Elapsed Time (hours)')
    ax3.set_ylabel('Current (mA)')
    ax3.set_title('Channel 30 - Current Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Ch31 Current
    ax4 = axes2[1, 1]
    for idx in range(1, 5):
        path_data = ch31_data[ch31_data['path_idx'] == idx]
        if not path_data.empty:
            ax4.scatter(path_data['elapsed_hours'], path_data['current'],
                       s=0.5, alpha=0.5, label=f'Path {idx}', color=path_colors[idx-1])
    ax4.set_xlabel('Elapsed Time (hours)')
    ax4.set_ylabel('Current (mA)')
    ax4.set_title('Channel 31 - Current Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/channel_specific_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()

    print("채널별 그래프가 'outputs/channel_specific_analysis.png'에 저장되었습니다.")

    # 전체 데이터 저장
    df.to_csv('outputs/complete_raw_data_all_columns.csv', index=False)
    print(f"모든 데이터가 'outputs/complete_raw_data_all_columns.csv'에 저장되었습니다.")

    # 채널별 데이터 개별 저장
    ch30_data.to_csv('outputs/channel_30_data.csv', index=False)
    ch31_data.to_csv('outputs/channel_31_data.csv', index=False)
    print(f"채널 30 데이터가 'outputs/channel_30_data.csv'에 저장되었습니다.")
    print(f"채널 31 데이터가 'outputs/channel_31_data.csv'에 저장되었습니다.")

    # 연속적인 elapsed time이 적용된 데이터도 저장 (Path 간 공백 제거)
    df_continuous = df.copy()
    df_continuous.to_csv('outputs/complete_raw_data_continuous.csv', index=False)
    ch30_data.to_csv('outputs/channel_30_continuous.csv', index=False)
    ch31_data.to_csv('outputs/channel_31_continuous.csv', index=False)
    print(f"연속적인 elapsed time이 적용된 데이터가 저장되었습니다 (Path 간 공백 제거됨).")

    print(f"저장된 컬럼: {', '.join(df.columns)}")

    # 경로별 통계 출력
    print(f"\n=== 경로별 데이터 분포 ===")
    for idx in sorted(df['path_idx'].unique()):
        path_data = df[df['path_idx'] == idx]
        path_name = path_data['path_name'].iloc[0]
        print(f"  Path {idx} - {path_name}:")
        print(f"    - 총 데이터 포인트: {len(path_data)}")

        # 채널별 분석
        for channel in sorted(path_data['channel'].unique()):
            ch_data = path_data[path_data['channel'] == channel]
            if not ch_data.empty:
                print(f"      Channel {channel}: {len(ch_data)} points")
                print(f"        시작: {ch_data['datetime'].min()}")
                print(f"        종료: {ch_data['datetime'].max()}")
                print(f"        사이클: {ch_data['total_cycle'].min()} ~ {ch_data['total_cycle'].max()}")
    print(f"총 행 수: {len(df)}")
else:
    print("데이터를 읽을 수 없습니다.")