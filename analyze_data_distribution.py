import pandas as pd
import numpy as np

# CSV 파일 읽기
print("데이터 분포 분석 중...")
df = pd.read_csv('outputs/complete_raw_data_all_columns.csv')

print(f"총 데이터 포인트: {len(df)}")
print(f"컬럼: {list(df.columns)}")

# 채널별 분포
print(f"\n=== 채널별 분포 ===")
if 'channel' in df.columns:
    channel_counts = df['channel'].value_counts().sort_index()
    for channel, count in channel_counts.items():
        print(f"채널 {channel}: {count} points")
else:
    print("channel 컬럼이 없습니다!")

# 경로별 분포
print(f"\n=== 경로별 분포 ===")
if 'path_idx' in df.columns:
    path_counts = df['path_idx'].value_counts().sort_index()
    for path, count in path_counts.items():
        print(f"경로 {path}: {count} points")
else:
    print("path_idx 컬럼이 없습니다!")

# 경로별 × 채널별 분포
print(f"\n=== 경로별 × 채널별 분포 ===")
if 'path_idx' in df.columns and 'channel' in df.columns:
    crosstab = pd.crosstab(df['path_idx'], df['channel'], margins=True)
    print(crosstab)
else:
    print("필요한 컬럼이 없습니다!")

# subdir별 분포 (원본)
print(f"\n=== Subdir별 분포 ===")
if 'subdir' in df.columns:
    subdir_counts = df['subdir'].value_counts()
    for subdir, count in subdir_counts.items():
        print(f"Subdir {subdir}: {count} points")

# 경로별 × subdir별 분포
print(f"\n=== 경로별 × Subdir별 분포 ===")
if 'path_idx' in df.columns and 'subdir' in df.columns:
    crosstab2 = pd.crosstab(df['path_idx'], df['subdir'], margins=True)
    print(crosstab2)

# 각 경로의 고유한 path_name 확인
print(f"\n=== 경로별 이름 확인 ===")
if 'path_idx' in df.columns and 'path_name' in df.columns:
    for path_idx in sorted(df['path_idx'].unique()):
        path_data = df[df['path_idx'] == path_idx]
        unique_names = path_data['path_name'].unique()
        print(f"경로 {path_idx}: {unique_names[0] if len(unique_names) > 0 else 'None'}")