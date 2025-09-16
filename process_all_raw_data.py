import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 데이터 경로
base_path = r"Rawdata\250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc\30"

# 모든 데이터 포인트 수집 (파일당 첫 줄만이 아닌 모든 데이터)
all_data = []
first_datetime = None
total_data_points = 0

for i in range(1, 505):
    file_path = os.path.join(base_path, f"{i:06d}")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            # 모든 데이터 라인 처리 (헤더 이후 모든 줄)
            for line in lines[4:]:
                if '2025/02/' in line:
                    parts = line.split(',')
                    if len(parts) > 3:
                        try:
                            date_str = parts[0]  # Date
                            time_str = parts[1]  # Time

                            # datetime 객체로 변환
                            datetime_str = f"{date_str} {time_str}"
                            current_datetime = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")

                            # 첫 번째 시간 저장
                            if first_datetime is None:
                                first_datetime = current_datetime

                            # 첫 번째 시간부터의 경과 시간 계산 (시간 단위)
                            elapsed_time = (current_datetime - first_datetime).total_seconds() / 3600.0

                            # Voltage와 Current 읽기
                            voltage = float(parts[3])
                            current = float(parts[4]) if len(parts) > 4 else 0.0

                            all_data.append({
                                'file_num': i,
                                'datetime': current_datetime,
                                'elapsed_hours': elapsed_time,
                                'voltage': voltage,
                                'current': current
                            })
                            total_data_points += 1

                        except (ValueError, IndexError):
                            continue

# 결과 분석
print(f"총 읽은 파일 수: {i}")
print(f"총 데이터 포인트: {total_data_points}")

if all_data:
    print(f"시작 시간: {all_data[0]['datetime']}")
    print(f"종료 시간: {all_data[-1]['datetime']}")
    print(f"총 경과 시간: {all_data[-1]['elapsed_hours']:.2f} hours")

    # 파일별 데이터 포인트 수 계산
    file_counts = {}
    for data in all_data:
        file_num = data['file_num']
        file_counts[file_num] = file_counts.get(file_num, 0) + 1

    print(f"\n파일별 데이터 포인트 수:")
    print(f"최소: {min(file_counts.values())} 포인트")
    print(f"최대: {max(file_counts.values())} 포인트")
    print(f"평균: {np.mean(list(file_counts.values())):.1f} 포인트")

    # 시각화 (모든 데이터 포인트 표시)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Complete Raw Data Analysis (Ch30) - {total_data_points} data points', fontsize=14)

    # 1. Voltage vs Elapsed Time (모든 데이터)
    ax1 = axes[0, 0]
    elapsed_times = [d['elapsed_hours'] for d in all_data]
    voltages = [d['voltage'] for d in all_data]
    file_nums = [d['file_num'] for d in all_data]

    scatter1 = ax1.scatter(elapsed_times, voltages, c=file_nums, cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('Elapsed Time (hours)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'All Voltage Data Points ({len(all_data)} points)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='File Number')

    # 2. Current vs Time
    ax2 = axes[0, 1]
    currents = [d['current'] for d in all_data]
    scatter2 = ax2.scatter(elapsed_times, currents, c=file_nums, cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('Elapsed Time (hours)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('All Current Data Points')
    ax2.grid(True, alpha=0.3)

    # 3. Data density over time (히스토그램)
    ax3 = axes[1, 0]
    ax3.hist(elapsed_times, bins=100, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Elapsed Time (hours)')
    ax3.set_ylabel('Number of Data Points')
    ax3.set_title('Data Density Distribution')
    ax3.grid(True, alpha=0.3)

    # 4. File coverage
    ax4 = axes[1, 1]
    file_list = sorted(file_counts.keys())
    counts = [file_counts[f] for f in file_list]
    ax4.bar(file_list, counts, width=1, alpha=0.7)
    ax4.set_xlabel('File Number')
    ax4.set_ylabel('Data Points per File')
    ax4.set_title('Data Points Distribution by File')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/complete_raw_data_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()

    print("\n그래프가 'outputs/complete_raw_data_analysis.png'에 저장되었습니다.")

    # 모든 데이터를 CSV로 저장
    df = pd.DataFrame(all_data)
    df.to_csv('outputs/all_raw_data.csv', index=False)
    print(f"모든 데이터가 'outputs/all_raw_data.csv'에 저장되었습니다. ({len(all_data)} rows)")