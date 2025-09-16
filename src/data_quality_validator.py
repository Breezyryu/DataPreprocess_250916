"""
배터리 데이터 품질 검증 유틸리티
Data Quality Validation Utility for Battery Data

데이터 무결성, 일관성, 시계열 정렬 검증을 수행합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    score: float  # 0-100점
    issues: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

class DataQualityValidator:
    """데이터 품질 검증기"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_dataframe(self, df: pd.DataFrame, channel_name: str = "") -> ValidationResult:
        """DataFrame 전체 품질 검증"""
        issues = []
        warnings = []
        stats = {}

        # 1. 기본 구조 검증
        structure_result = self._validate_structure(df)
        issues.extend(structure_result['issues'])
        warnings.extend(structure_result['warnings'])
        stats.update(structure_result['stats'])

        # 2. 시계열 정렬 검증
        time_result = self._validate_time_series(df)
        issues.extend(time_result['issues'])
        warnings.extend(time_result['warnings'])
        stats.update(time_result['stats'])

        # 3. 데이터 범위 검증
        range_result = self._validate_data_ranges(df)
        issues.extend(range_result['issues'])
        warnings.extend(range_result['warnings'])
        stats.update(range_result['stats'])

        # 4. 일관성 검증
        consistency_result = self._validate_consistency(df)
        issues.extend(consistency_result['issues'])
        warnings.extend(consistency_result['warnings'])
        stats.update(consistency_result['stats'])

        # 전체 점수 계산
        score = self._calculate_quality_score(issues, warnings, stats)
        is_valid = len(issues) == 0 and score >= 70  # 70점 이상이면 유효

        self.logger.info(f"{channel_name} 품질 점수: {score:.1f}/100, 유효성: {is_valid}")

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            stats=stats
        )

    def _validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기본 구조 검증"""
        issues = []
        warnings = []
        stats = {}

        # 필수 컬럼 확인
        required_columns = ['Time_Sec', 'Voltage_V', 'Current_A']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            issues.append(f"필수 컬럼 누락: {missing_columns}")

        # 데이터 포인트 수 확인
        row_count = len(df)
        stats['row_count'] = row_count

        if row_count == 0:
            issues.append("빈 DataFrame")
        elif row_count < 10:
            warnings.append(f"데이터가 너무 적음: {row_count}개 포인트")

        # 결측값 확인
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df) * 100).round(2)

        for col, percentage in null_percentages.items():
            if percentage > 0:
                stats[f'{col}_null_percent'] = percentage
                if percentage > 50:
                    issues.append(f"{col} 컬럼의 50% 이상이 결측값")
                elif percentage > 10:
                    warnings.append(f"{col} 컬럼의 {percentage}%가 결측값")

        # 중복 행 확인
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df) * 100)
            stats['duplicate_percent'] = duplicate_percentage
            if duplicate_percentage > 5:
                issues.append(f"중복 행이 {duplicate_percentage:.1f}% 발견")
            else:
                warnings.append(f"중복 행 {duplicate_count}개 발견")

        return {
            'issues': issues,
            'warnings': warnings,
            'stats': stats
        }

    def _validate_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시계열 정렬 검증"""
        issues = []
        warnings = []
        stats = {}

        if 'Time_Sec' not in df.columns:
            return {'issues': ['Time_Sec 컬럼 없음'], 'warnings': [], 'stats': {}}

        time_series = df['Time_Sec'].dropna()

        if len(time_series) < 2:
            return {'issues': ['시간 데이터가 부족함'], 'warnings': [], 'stats': {}}

        # 시계열 정렬 확인
        is_sorted = time_series.is_monotonic_increasing
        stats['is_time_sorted'] = is_sorted

        if not is_sorted:
            issues.append("시계열 데이터가 정렬되지 않음")

            # 정렬되지 않은 지점 수 계산
            unsorted_points = 0
            for i in range(1, len(time_series)):
                if time_series.iloc[i] < time_series.iloc[i-1]:
                    unsorted_points += 1

            stats['unsorted_points'] = unsorted_points
            stats['unsorted_percent'] = (unsorted_points / len(time_series) * 100)

        # 시간 간격 분석
        time_diffs = time_series.diff().dropna()

        if len(time_diffs) > 0:
            stats['avg_time_interval'] = time_diffs.mean()
            stats['min_time_interval'] = time_diffs.min()
            stats['max_time_interval'] = time_diffs.max()

            # 음수 시간 간격 확인
            negative_intervals = (time_diffs < 0).sum()
            if negative_intervals > 0:
                issues.append(f"음수 시간 간격 {negative_intervals}개 발견")

            # 비정상적으로 큰 시간 간격 확인
            median_interval = time_diffs.median()
            large_gaps = (time_diffs > median_interval * 10).sum()
            if large_gaps > 0:
                warnings.append(f"비정상적으로 큰 시간 간격 {large_gaps}개 발견")

        return {
            'issues': issues,
            'warnings': warnings,
            'stats': stats
        }

    def _validate_data_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 범위 검증"""
        issues = []
        warnings = []
        stats = {}

        # 범위 검증 규칙
        range_rules = {
            'Voltage_V': (2.0, 5.0),
            'Current_A': (-10.0, 10.0),
            'Temperature_C': (-40.0, 100.0),
            'SOC_%': (0.0, 100.0),
            'Capacity_Ah': (0.0, 10.0)
        }

        for col, (min_val, max_val) in range_rules.items():
            if col not in df.columns:
                continue

            try:
                # 컬럼이 Series인지 먼저 확인
                if not isinstance(df[col], pd.Series):
                    warnings.append(f"{col} 컬럼이 Series가 아님")
                    continue

                series = pd.to_numeric(df[col], errors='coerce').dropna()

                if len(series) == 0:
                    continue

                # 기본 통계
                stats[f'{col}_mean'] = series.mean()
                stats[f'{col}_std'] = series.std()
                stats[f'{col}_min'] = series.min()
                stats[f'{col}_max'] = series.max()

                # 범위 검증
                out_of_range = ((series < min_val) | (series > max_val)).sum()
                if out_of_range > 0:
                    percentage = (out_of_range / len(series) * 100)
                    stats[f'{col}_out_of_range_percent'] = percentage

                    if percentage > 1:
                        issues.append(f"{col}: {percentage:.1f}%가 정상 범위({min_val}-{max_val}) 벗어남")
                    else:
                        warnings.append(f"{col}: {out_of_range}개 값이 정상 범위 벗어남")

            except Exception as e:
                warnings.append(f"{col} 범위 검증 실패: {e}")

        return {
            'issues': issues,
            'warnings': warnings,
            'stats': stats
        }

    def _validate_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """일관성 검증"""
        issues = []
        warnings = []
        stats = {}

        # 채널별 데이터 일관성 확인
        if 'TotalCycle' in df.columns:
            cycle_series = pd.to_numeric(df['TotalCycle'], errors='coerce').dropna()

            if len(cycle_series) > 0:
                stats['total_cycles'] = cycle_series.max()
                stats['cycle_consistency'] = cycle_series.is_monotonic_increasing

                if not cycle_series.is_monotonic_increasing:
                    warnings.append("TotalCycle이 단조증가하지 않음")

        # 전압-전류 상관관계 확인
        if 'Voltage_V' in df.columns and 'Current_A' in df.columns:
            try:
                # 컬럼이 Series인지 먼저 확인
                if not isinstance(df['Voltage_V'], pd.Series) or not isinstance(df['Current_A'], pd.Series):
                    warnings.append("전압 또는 전류 데이터가 Series가 아님")
                else:
                    voltage = pd.to_numeric(df['Voltage_V'], errors='coerce')
                    current = pd.to_numeric(df['Current_A'], errors='coerce')

                    # 둘 다 유효한 데이터가 있는 경우만
                    valid_mask = voltage.notna() & current.notna()
                    if valid_mask.sum() > 10:
                        correlation = voltage[valid_mask].corr(current[valid_mask])
                        stats['voltage_current_correlation'] = correlation

                        # 비정상적인 상관관계 확인
                        if abs(correlation) < 0.1 and len(df) > 100:
                            warnings.append(f"전압-전류 상관관계가 낮음: {correlation:.3f}")
            except Exception as e:
                warnings.append(f"전압-전류 상관관계 계산 실패: {e}")

        return {
            'issues': issues,
            'warnings': warnings,
            'stats': stats
        }

    def _calculate_quality_score(self, issues: List[str], warnings: List[str], stats: Dict[str, Any]) -> float:
        """품질 점수 계산 (0-100점)"""
        score = 100.0

        # 치명적 문제로 점수 차감
        score -= len(issues) * 20  # 문제당 20점 차감

        # 경고로 점수 차감
        score -= len(warnings) * 5  # 경고당 5점 차감

        # 추가 점수 차감 조건들
        if stats.get('duplicate_percent', 0) > 10:
            score -= 10

        if stats.get('unsorted_percent', 0) > 5:
            score -= 15

        # 데이터 완성도에 따른 보너스/페널티
        null_penalties = 0
        for key, value in stats.items():
            if key.endswith('_null_percent') and value > 0:
                null_penalties += value * 0.5  # null 비율의 50%만큼 차감

        score -= null_penalties

        return max(0.0, min(100.0, score))  # 0-100 범위로 제한

    def validate_channels_consistency(self, channel_data: Dict[str, pd.DataFrame]) -> ValidationResult:
        """채널 간 일관성 검증"""
        issues = []
        warnings = []
        stats = {}

        if len(channel_data) < 2:
            return ValidationResult(True, 100.0, [], ["단일 채널 데이터"], {})

        channel_names = list(channel_data.keys())
        channel_stats = {}

        # 각 채널의 기본 정보 수집
        for channel, df in channel_data.items():
            if df.empty:
                continue

            channel_stats[channel] = {
                'row_count': len(df),
                'voltage_mean': pd.to_numeric(df.get('Voltage_V', []), errors='coerce').mean(),
                'current_mean': pd.to_numeric(df.get('Current_A', []), errors='coerce').mean(),
                'time_range': (
                    pd.to_numeric(df.get('Time_Sec', []), errors='coerce').min(),
                    pd.to_numeric(df.get('Time_Sec', []), errors='coerce').max()
                )
            }

        # 데이터 포인트 수 비교
        row_counts = [info['row_count'] for info in channel_stats.values()]
        if len(set(row_counts)) > 1:  # 다른 개수가 있는 경우
            stats['row_count_difference'] = max(row_counts) - min(row_counts)
            difference_percent = (stats['row_count_difference'] / max(row_counts) * 100)

            if difference_percent > 20:
                issues.append(f"채널 간 데이터 포인트 수 차이가 큼: {difference_percent:.1f}%")
            elif difference_percent > 5:
                warnings.append(f"채널 간 데이터 포인트 수 차이: {difference_percent:.1f}%")

        stats['channel_stats'] = channel_stats

        score = 100.0 - len(issues) * 15 - len(warnings) * 3
        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            score=max(0.0, score),
            issues=issues,
            warnings=warnings,
            stats=stats
        )

def validate_battery_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, ValidationResult]:
    """배터리 데이터 전체 검증"""
    validator = DataQualityValidator()
    results = {}

    # 각 채널별 검증
    for channel, df in data_dict.items():
        results[channel] = validator.validate_dataframe(df, channel)

    # 채널 간 일관성 검증
    results['cross_channel'] = validator.validate_channels_consistency(data_dict)

    return results