"""
Universal Battery Data Loader
Supports multiple manufacturers and test formats
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BatteryMetadata:
    """Battery test metadata extracted from folder names"""
    manufacturer: str
    model: str
    capacity_mah: float
    test_condition: str
    cycle_range: str
    tester_name: str
    start_date: str
    end_date: str
    folder_path: str


class UniversalBatteryLoader:
    """
    Universal loader for battery test data from Rawdata folder structure
    Supports ATL, Toyo, PNE and other manufacturers
    """

    def __init__(self, base_path: str = "Rawdata"):
        self.base_path = base_path
        self.datasets = []
        self.metadata = {}

    def scan_datasets(self) -> List[str]:
        """Scan Rawdata folder for all available datasets"""
        datasets = []

        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Base path {self.base_path} not found")

        # Pattern for battery test folders
        # Format: YYMMDD_YYMMDD_N_Name_CapacitymAh_Manufacturer Model_TestCondition
        pattern1 = r'(\d{6})_(\d{6})_(\d+)_([^_]+)_(\d+)mAh_(.+)'

        # Alternative patterns for other manufacturers
        pattern2 = r'([A-Za-z0-9_]+)_([A-Za-z0-9_]+)_(\d+)mAh_([A-Za-z0-9_]+)'  # A1_MP1_4500mAh_T23_1
        pattern3 = r'(.+)\[(\d+)[Vv]\s*(\d+)mAh\]'  # Q7M Sub ATL [45v 2068mAh] [23]

        for folder in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, folder)
            if os.path.isdir(folder_path):
                # Try primary pattern first (ATL format)
                match = re.match(pattern1, folder)
                if match:
                    datasets.append(folder)
                    self._extract_metadata_type1(folder, folder_path, match)
                else:
                    # Try pattern 2 (PNE format)
                    match = re.match(pattern2, folder)
                    if match:
                        datasets.append(folder)
                        self._extract_metadata_type2(folder, folder_path, match)
                    else:
                        # Try pattern 3 (Toyo format)
                        match = re.search(pattern3, folder)
                        if match:
                            datasets.append(folder)
                            self._extract_metadata_type3(folder, folder_path, match)
                        else:
                            # Try to include any directory with channel subdirectories
                            channel_dirs = [d for d in os.listdir(folder_path)
                                          if os.path.isdir(os.path.join(folder_path, d)) and d.isdigit()]
                            if channel_dirs:
                                datasets.append(folder)
                                self._extract_metadata_generic(folder, folder_path)

        self.datasets = datasets
        print(f"Found {len(datasets)} datasets in {self.base_path}")

        # Group by manufacturer
        manufacturers = {}
        for dataset, meta in self.metadata.items():
            mfr = meta.manufacturer
            if mfr not in manufacturers:
                manufacturers[mfr] = []
            manufacturers[mfr].append(dataset)

        for mfr, items in manufacturers.items():
            print(f"  {mfr}: {len(items)} datasets")

        return datasets

    def _extract_metadata_type1(self, folder_name: str, folder_path: str, match) -> BatteryMetadata:
        """Extract metadata from folder name"""
        start_date = match.group(1)
        end_date = match.group(2)
        test_number = match.group(3)
        tester_name = match.group(4)
        capacity = int(match.group(5))

        # Parse manufacturer and test info
        remaining = match.group(6)
        parts = remaining.split(' ')

        # Identify manufacturer (ATL, Toyo, PNE, etc.)
        manufacturer = parts[0] if parts else "Unknown"

        # Extract model and test conditions
        if len(parts) > 1:
            # Find where test condition starts (usually contains 'C' for C-rate)
            for i, part in enumerate(parts):
                if 'C' in part or '온' in part or 'cyc' in part:
                    model = ' '.join(parts[1:i]) if i > 1 else parts[1]
                    test_condition = ' '.join(parts[i:])
                    break
            else:
                model = ' '.join(parts[1:]) if len(parts) > 1 else "Unknown"
                test_condition = "Unknown"
        else:
            model = "Unknown"
            test_condition = "Unknown"

        # Extract cycle range if present
        cycle_match = re.search(r'(\d+)-(\d+)cyc', test_condition)
        cycle_range = cycle_match.group(0) if cycle_match else "Unknown"

        metadata = BatteryMetadata(
            manufacturer=manufacturer,
            model=model,
            capacity_mah=capacity,
            test_condition=test_condition,
            cycle_range=cycle_range,
            tester_name=tester_name,
            start_date=start_date,
            end_date=end_date,
            folder_path=folder_path
        )

        self.metadata[folder_name] = metadata
        return metadata

    def _extract_metadata_type2(self, folder_name: str, folder_path: str, match) -> BatteryMetadata:
        """Extract metadata from PNE format: A1_MP1_4500mAh_T23_1"""
        model = match.group(1)
        series = match.group(2)
        capacity = int(match.group(3))
        condition = match.group(4)

        metadata = BatteryMetadata(
            manufacturer="PNE",
            model=f"{model} {series}",
            capacity_mah=capacity,
            test_condition=condition,
            cycle_range="Unknown",
            tester_name="Unknown",
            start_date="Unknown",
            end_date="Unknown",
            folder_path=folder_path
        )

        self.metadata[folder_name] = metadata
        return metadata

    def _extract_metadata_type3(self, folder_name: str, folder_path: str, match) -> BatteryMetadata:
        """Extract metadata from Toyo format: Q7M Sub ATL [45v 2068mAh] [23]"""
        model_info = match.group(1).strip()
        voltage = match.group(2)
        capacity = int(match.group(3))

        # Extract manufacturer from model info
        if "ATL" in model_info:
            manufacturer = "ATL"
            model = model_info.replace("ATL", "").strip()
        else:
            manufacturer = "Toyo"
            model = model_info

        metadata = BatteryMetadata(
            manufacturer=manufacturer,
            model=model,
            capacity_mah=capacity,
            test_condition=f"{voltage}V test",
            cycle_range="Unknown",
            tester_name="Unknown",
            start_date="Unknown",
            end_date="Unknown",
            folder_path=folder_path
        )

        self.metadata[folder_name] = metadata
        return metadata

    def _extract_metadata_generic(self, folder_name: str, folder_path: str) -> BatteryMetadata:
        """Extract metadata from generic format with fallback defaults"""
        metadata = BatteryMetadata(
            manufacturer="Unknown",
            model="Unknown",
            capacity_mah=0,
            test_condition="Unknown",
            cycle_range="Unknown",
            tester_name="Unknown",
            start_date="Unknown",
            end_date="Unknown",
            folder_path=folder_path
        )

        self.metadata[folder_name] = metadata
        return metadata

    def load_dataset(self, dataset_name: str, channels: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load a specific dataset with optional channel filtering

        Args:
            dataset_name: Name of the dataset folder
            channels: List of channel numbers to load (e.g., [30, 31])
                     If None, loads all available channels
        """
        if dataset_name not in self.metadata:
            raise ValueError(f"Dataset {dataset_name} not found")

        metadata = self.metadata[dataset_name]
        folder_path = metadata.folder_path

        # Find available channels - handle different directory structures
        available_channels = []
        channel_paths = {}

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                if item.isdigit():
                    # Standard format: numerical channel directories
                    channel_num = int(item)
                    available_channels.append(channel_num)
                    channel_paths[channel_num] = item_path
                elif item.startswith('M02Ch'):
                    # PNE format: M02Ch073[073]
                    match = re.search(r'M02Ch(\d+)', item)
                    if match:
                        channel_num = int(match.group(1))
                        available_channels.append(channel_num)
                        channel_paths[channel_num] = item_path

        if not available_channels:
            print(f"No channel directories found in {folder_path}")
            return pd.DataFrame()

        # Filter channels if specified, with intelligent fallback
        if channels:
            channels_to_load = [ch for ch in channels if ch in available_channels]
            # If no specified channels are found, use all available channels as fallback
            if not channels_to_load:
                print(f"[WARN] Requested channels {channels} not found. Available channels: {available_channels}")
                print(f"[INFO] Using all available channels as fallback")
                channels_to_load = available_channels
        else:
            channels_to_load = available_channels

        if not channels_to_load:
            print(f"[ERROR] No channels available to load from {dataset_name}")
            return pd.DataFrame()

        print(f"Loading channels {channels_to_load} from {dataset_name}")

        all_data = []
        first_datetime = None

        for channel in sorted(channels_to_load):
            channel_path = channel_paths[channel]  # Use the mapped path
            channel_data = self._load_channel_data(
                channel_path, channel, metadata, first_datetime
            )

            if channel_data and first_datetime is None:
                # Set first datetime from first channel
                first_datetime = channel_data[0]['datetime']

            all_data.extend(channel_data)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Recalculate elapsed time from global first datetime
        first_dt = df['datetime'].min()
        df['elapsed_hours'] = (df['datetime'] - first_dt).dt.total_seconds() / 3600.0

        # Add metadata columns
        df['manufacturer'] = metadata.manufacturer
        df['model'] = metadata.model
        df['nominal_capacity_mah'] = metadata.capacity_mah
        df['test_condition'] = metadata.test_condition

        return df

    def _load_channel_data(self, channel_path: str, channel: int,
                           metadata: BatteryMetadata,
                           first_datetime: Optional[datetime]) -> List[Dict]:
        """Load data from a single channel directory"""
        data_points = []

        # List all data files
        files = [f for f in os.listdir(channel_path)
                if os.path.isfile(os.path.join(channel_path, f))]

        for file_name in sorted(files):
            file_path = os.path.join(channel_path, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                # Skip header lines (usually first 4 lines)
                for line in lines[4:]:
                    # Look for date pattern (YYYY/MM/DD)
                    if re.match(r'\d{4}/\d{2}/\d{2}', line):
                        parts = line.strip().split(',')

                        if len(parts) >= 16:
                            try:
                                # Parse data fields
                                date_str = parts[0]
                                time_str = parts[1]
                                datetime_str = f"{date_str} {time_str}"
                                current_datetime = datetime.strptime(
                                    datetime_str, "%Y/%m/%d %H:%M:%S"
                                )

                                # Calculate elapsed time if first_datetime is set
                                if first_datetime:
                                    elapsed = (current_datetime - first_datetime).total_seconds() / 3600.0
                                else:
                                    elapsed = 0.0

                                data_point = {
                                    'datetime': current_datetime,
                                    'date': date_str,
                                    'time': time_str,
                                    'channel': channel,
                                    'file_name': file_name,
                                    'passtime_sec': int(parts[2]),
                                    'voltage': float(parts[3].replace('+', '').replace('-', '')),
                                    'current': float(parts[4].replace('+', '').replace('-', '')) if parts[4] else 0.0,
                                    'temperature': float(parts[7].replace('+', '').replace('-', '')) if parts[7] else 0.0,
                                    'condition': int(parts[11].strip()) if parts[11].strip() else 0,
                                    'mode': int(parts[12].strip()) if parts[12].strip() else 0,
                                    'cycle': int(parts[13].strip()) if parts[13].strip() else 0,
                                    'total_cycle': int(parts[14].strip()) if parts[14].strip() else 0,
                                    'elapsed_hours': elapsed
                                }

                                data_points.append(data_point)

                            except (ValueError, IndexError) as e:
                                # Skip problematic lines
                                continue

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

        return data_points

    def load_all_datasets(self, channels: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load all available datasets and combine them

        Args:
            channels: Optional list of channels to load

        Returns:
            Combined DataFrame with all datasets
        """
        if not self.datasets:
            self.scan_datasets()

        all_dfs = []

        for dataset in self.datasets:
            print(f"\nLoading dataset: {dataset}")
            df = self.load_dataset(dataset, channels)

            if not df.empty:
                df['dataset_name'] = dataset
                all_dfs.append(df)
                print(f"  Loaded {len(df)} data points")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"\nTotal data points loaded: {len(combined_df)}")

            # Summary by manufacturer
            print("\nData points by manufacturer:")
            for mfr in combined_df['manufacturer'].unique():
                count = len(combined_df[combined_df['manufacturer'] == mfr])
                print(f"  {mfr}: {count:,} points")

            return combined_df
        else:
            return pd.DataFrame()

    def compare_manufacturers(self, df: pd.DataFrame) -> Dict:
        """
        Generate comparison statistics between manufacturers

        Args:
            df: Combined DataFrame with all manufacturers

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}

        for manufacturer in df['manufacturer'].unique():
            mfr_data = df[df['manufacturer'] == manufacturer]

            comparison[manufacturer] = {
                'total_cycles': mfr_data['total_cycle'].max(),
                'voltage_range': (mfr_data['voltage'].min(), mfr_data['voltage'].max()),
                'current_range': (mfr_data['current'].min(), mfr_data['current'].max()),
                'temperature_range': (mfr_data['temperature'].min(), mfr_data['temperature'].max()),
                'data_points': len(mfr_data),
                'channels': mfr_data['channel'].unique().tolist(),
                'test_duration_hours': mfr_data['elapsed_hours'].max()
            }

        return comparison