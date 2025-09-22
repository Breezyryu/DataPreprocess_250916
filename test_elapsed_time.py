import pandas as pd
import numpy as np

# CSV 파일 읽기
print("CSV 파일 읽기...")
df = pd.read_csv('outputs/complete_raw_data_all_columns.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# 정렬
df = df.sort_values(['path_idx', 'channel', 'datetime'])

print("\nElapsed time 재계산 중...")

# 각 path의 시간 범위 확인
path_times = {}
for path_idx in sorted(df['path_idx'].unique()):
    path_data = df[df['path_idx'] == path_idx]
    path_times[path_idx] = {
        'first': path_data['datetime'].min(),
        'last': path_data['datetime'].max()
    }
    print(f"  Path {path_idx}: {path_times[path_idx]['first']} ~ {path_times[path_idx]['last']}")

# 전체 데이터 기준 시작 시간
global_start = df['datetime'].min()

# elapsed_hours를 연속적으로 재계산
for path_idx in sorted(df['path_idx'].unique()):
    path_mask = df['path_idx'] == path_idx

    if path_idx == 1:
        # 첫 번째 path는 실제 시간 차이 사용
        df.loc[path_mask, 'elapsed_hours'] = (
            (df.loc[path_mask, 'datetime'] - global_start).dt.total_seconds() / 3600.0
        )
    else:
        # 이전 path의 마지막 elapsed_hours 값 가져오기
        prev_path_data = df[df['path_idx'] == path_idx - 1]
        prev_max_elapsed = prev_path_data['elapsed_hours'].max()

        # 현재 path의 시작 시간
        path_start = path_times[path_idx]['first']

        # 현재 path 내에서의 상대 시간 + 이전 누적 시간
        df.loc[path_mask, 'elapsed_hours'] = (
            prev_max_elapsed +
            (df.loc[path_mask, 'datetime'] - path_start).dt.total_seconds() / 3600.0
        )

print("\n재계산 후 Path별 Elapsed Hours 범위:")
for path_idx in sorted(df['path_idx'].unique()):
    path_data = df[df['path_idx'] == path_idx]
    min_time = path_data['elapsed_hours'].min()
    max_time = path_data['elapsed_hours'].max()
    print(f"  Path {path_idx}: {min_time:.2f} ~ {max_time:.2f} hours")

print("\nPath 간 연속성 확인:")
for i in range(1, 4):
    path1_max = df[df['path_idx'] == i]['elapsed_hours'].max()
    path2_min = df[df['path_idx'] == i+1]['elapsed_hours'].min()
    gap = path2_min - path1_max
    print(f"  Path {i} 끝({path1_max:.2f}) → Path {i+1} 시작({path2_min:.2f}): 간격 = {gap:.4f} hours")

print(f"\n전체 시간 범위: 0.00 ~ {df['elapsed_hours'].max():.2f} hours")

# 수정된 데이터 저장
df.to_csv('outputs/complete_raw_data_continuous.csv', index=False)
print("\n연속적인 elapsed time이 적용된 데이터가 'outputs/complete_raw_data_continuous.csv'에 저장되었습니다.")

# 채널별 데이터도 저장
ch30_data = df[df['channel'] == 30]
ch31_data = df[df['channel'] == 31]
ch30_data.to_csv('outputs/channel_30_continuous.csv', index=False)
ch31_data.to_csv('outputs/channel_31_continuous.csv', index=False)
print("채널별 데이터도 저장되었습니다.")