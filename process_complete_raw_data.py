import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 데이터 경로
base_path = r"Rawdata\250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc"

# 모든 데이터 포인트와 모든 컬럼 수집
all_data = []
first_datetime = None
total_data_points = 0
files_processed = 0

print("데이터 처리 시작...")

# 하위 디렉토리 30과 31을 탐색
for subdir in ['30', '31']:
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
                    if '2025/02/' in line:
                        parts = line.strip().split(',')
                        if len(parts) >= 16:  # 모든 필드가 있는지 확인
                            try:
                                # 날짜와 시간
                                date_str = parts[0]
                                time_str = parts[1]
                                datetime_str = f"{date_str} {time_str}"
                                current_datetime = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")

                                # 첫 번째 시간 저장
                                if first_datetime is None:
                                    first_datetime = current_datetime

                                # 경과 시간 계산 (시간 단위)
                                elapsed_time = (current_datetime - first_datetime).total_seconds() / 3600.0

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
                print(f"처리 중... {files_processed} 파일 완료 (디렉토리 {subdir})")

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

    # 시각화
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Complete Raw Data Analysis with All Columns (Ch30) - {total_data_points} points', fontsize=14)

    # 1. Voltage vs Time (colored by cycle)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(df['elapsed_hours'], df['voltage'],
                          c=df['total_cycle'], cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('Elapsed Time (hours)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage Profile (colored by cycle)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Total Cycle')

    # 2. Current vs Time
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['elapsed_hours'], df['current'],
                          c=df['mode'], cmap='tab10', s=1, alpha=0.5)
    ax2.set_xlabel('Elapsed Time (hours)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('Current Profile (colored by mode)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Mode')

    # 3. Temperature vs Time
    ax3 = axes[1, 0]
    ax3.plot(df['elapsed_hours'], df['temperature'], 'g-', linewidth=0.5, alpha=0.7)
    ax3.set_xlabel('Elapsed Time (hours)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Temperature Profile')
    ax3.grid(True, alpha=0.3)

    # 4. Cycle progression
    ax4 = axes[1, 1]
    ax4.plot(df['elapsed_hours'], df['total_cycle'], 'b-', linewidth=1)
    ax4.set_xlabel('Elapsed Time (hours)')
    ax4.set_ylabel('Total Cycle Number')
    ax4.set_title('Cycle Progression')
    ax4.grid(True, alpha=0.3)

    # 5. Voltage distribution by condition
    ax5 = axes[2, 0]
    for cond in sorted(df['condition'].unique()):
        cond_data = df[df['condition'] == cond]
        ax5.hist(cond_data['voltage'], bins=50, alpha=0.5, label=f'Condition {cond}')
    ax5.set_xlabel('Voltage (V)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Voltage Distribution by Condition')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Data points per cycle
    ax6 = axes[2, 1]
    cycle_counts = df.groupby('total_cycle').size()
    ax6.bar(cycle_counts.index, cycle_counts.values, width=1, alpha=0.7)
    ax6.set_xlabel('Cycle Number')
    ax6.set_ylabel('Data Points')
    ax6.set_title('Data Points per Cycle')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/complete_all_columns_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()

    print("\n그래프가 'outputs/complete_all_columns_analysis.png'에 저장되었습니다.")

    # 모든 데이터를 CSV로 저장 (모든 컬럼 포함)
    df.to_csv('outputs/complete_raw_data_all_columns.csv', index=False)
    print(f"모든 데이터가 'outputs/complete_raw_data_all_columns.csv'에 저장되었습니다.")
    print(f"저장된 컬럼: {', '.join(df.columns)}")
    print(f"총 행 수: {len(df)}")
else:
    print("데이터를 읽을 수 없습니다.")