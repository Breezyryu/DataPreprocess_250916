import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 데이터 경로
base_path = r"Rawdata\250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc\30"

# 모든 파일의 시간 정보 수집
file_data = []
first_datetime = None

for i in range(1, 505):
    file_path = os.path.join(base_path, f"{i:06d}")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            # 첫 번째 데이터 라인 찾기
            for line in lines[4:]:
                if '2025/02/' in line:
                    parts = line.split(',')
                    if len(parts) > 2:
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

                            # Voltage 읽기
                            voltage = float(parts[3])

                            file_data.append({
                                'file_num': i,
                                'datetime': current_datetime,
                                'elapsed_hours': elapsed_time,
                                'voltage': voltage
                            })
                            break
                        except (ValueError, IndexError):
                            continue

# 결과 분석
print(f"총 읽은 파일 수: {len(file_data)}")
if file_data:
    print(f"시작 시간: {file_data[0]['datetime']}")
    print(f"종료 시간: {file_data[-1]['datetime']}")
    print(f"총 경과 시간: {file_data[-1]['elapsed_hours']:.2f} hours")

    # 시간 갭 분석
    print("\n시간 갭 분석:")
    gaps = []
    for i in range(1, len(file_data)):
        time_diff = file_data[i]['elapsed_hours'] - file_data[i-1]['elapsed_hours']
        if time_diff > 1.0:  # 1시간 이상 갭
            gaps.append({
                'file1': file_data[i-1]['file_num'],
                'file2': file_data[i]['file_num'],
                'time1': file_data[i-1]['elapsed_hours'],
                'time2': file_data[i]['elapsed_hours'],
                'gap': time_diff
            })

    print(f"큰 시간 갭 (> 1 hour): {len(gaps)}개")
    for g in gaps[:10]:
        print(f"파일 {g['file1']:03d} ({g['time1']:.2f}h) -> 파일 {g['file2']:03d} ({g['time2']:.2f}h): 갭 = {g['gap']:.2f}h")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fixed Time Analysis (Ch30)', fontsize=14)

    # 1. Voltage vs Elapsed Time
    ax1 = axes[0, 0]
    file_nums = [d['file_num'] for d in file_data]
    elapsed_times = [d['elapsed_hours'] for d in file_data]
    voltages = [d['voltage'] for d in file_data]

    scatter1 = ax1.scatter(elapsed_times, voltages, c=file_nums, cmap='viridis', s=10, alpha=0.6)
    ax1.set_xlabel('Elapsed Time (hours)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage vs Elapsed Time (corrected)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='File Number')

    # 2. Time progression
    ax2 = axes[0, 1]
    ax2.plot(file_nums, elapsed_times, 'b-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('File Number')
    ax2.set_ylabel('Elapsed Time (hours)')
    ax2.set_title('Time Progression by File')
    ax2.grid(True, alpha=0.3)

    # 3. Time gaps
    ax3 = axes[1, 0]
    time_diffs = [0]  # 첫 파일은 차이가 0
    for i in range(1, len(file_data)):
        time_diffs.append(file_data[i]['elapsed_hours'] - file_data[i-1]['elapsed_hours'])

    ax3.bar(range(len(time_diffs)), time_diffs, width=1, alpha=0.7)
    ax3.set_xlabel('File Index')
    ax3.set_ylabel('Time Gap (hours)')
    ax3.set_title('Time Gaps Between Consecutive Files')
    ax3.grid(True, alpha=0.3)

    # 4. Voltage progression
    ax4 = axes[1, 1]
    ax4.plot(file_nums, voltages, 'r-', linewidth=1, alpha=0.7)
    ax4.set_xlabel('File Number')
    ax4.set_ylabel('Voltage (V)')
    ax4.set_title('Voltage Progression by File')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/fixed_time_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()

    print("\n그래프가 'outputs/fixed_time_analysis.png'에 저장되었습니다.")

    # CSV로 저장
    df = pd.DataFrame(file_data)
    df.to_csv('outputs/time_analysis_fixed.csv', index=False)
    print("데이터가 'outputs/time_analysis_fixed.csv'에 저장되었습니다.")