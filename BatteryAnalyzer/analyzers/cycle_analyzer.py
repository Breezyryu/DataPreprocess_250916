"""
Cycle Analysis and Classification System
Identifies and classifies different types of battery test cycles
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class CycleType(Enum):
    """Battery test cycle classification"""
    NORMAL_CHARGE_DISCHARGE = "Normal Charge/Discharge"
    RPT_02C = "0.2C RPT (Reference Performance Test)"
    RPT_05C = "0.5C RPT"
    RPT_1C = "1C RPT"
    MULTI_STEP = "Multi-Step Charge/Discharge"
    RSS_MEASUREMENT = "Rss (Internal Resistance) Measurement"
    CAPACITY_CHECK = "Capacity Check"
    OCV_MEASUREMENT = "OCV (Open Circuit Voltage) Measurement"
    RATE_TEST = "Rate Capability Test"
    PULSE_TEST = "Pulse Test"
    HPPC = "HPPC (Hybrid Pulse Power Characterization)"
    REST = "Rest Period"
    FORMATION = "Formation Cycle"


@dataclass
class CycleMetrics:
    """Metrics calculated for each cycle"""
    cycle_number: int
    cycle_type: CycleType
    charge_capacity_mah: float
    discharge_capacity_mah: float
    coulombic_efficiency: float
    energy_charge_wh: float
    energy_discharge_wh: float
    energy_efficiency: float
    average_voltage_charge: float
    average_voltage_discharge: float
    c_rate_charge: float
    c_rate_discharge: float
    duration_hours: float
    max_temperature: float
    min_temperature: float
    internal_resistance_mohm: Optional[float] = None


class CycleAnalyzer:
    """
    Comprehensive cycle analysis and classification system
    """

    def __init__(self, nominal_capacity_mah: float = 1689.0):
        """
        Initialize cycle analyzer

        Args:
            nominal_capacity_mah: Nominal battery capacity in mAh for C-rate calculation
        """
        self.nominal_capacity = nominal_capacity_mah
        self.cycles = {}
        self.cycle_boundaries = []

    def segment_cycles(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Segment continuous data into individual cycles

        Args:
            df: Battery data DataFrame

        Returns:
            List of (start_index, end_index) tuples for each cycle
        """
        boundaries = []

        # Method 1: Use total_cycle column if available
        if 'total_cycle' in df.columns:
            for cycle_num in df['total_cycle'].unique():
                cycle_data = df[df['total_cycle'] == cycle_num]
                if not cycle_data.empty:
                    start_idx = cycle_data.index[0]
                    end_idx = cycle_data.index[-1]
                    boundaries.append((start_idx, end_idx))

        # Method 2: Detect charge-discharge transitions
        elif 'current' in df.columns:
            # Find current sign changes
            current_sign = np.sign(df['current'].values)
            sign_changes = np.where(np.diff(current_sign) != 0)[0]

            # Group sign changes into cycles
            for i in range(0, len(sign_changes) - 1, 2):
                if i + 1 < len(sign_changes):
                    start_idx = sign_changes[i]
                    end_idx = sign_changes[i + 1]
                    boundaries.append((start_idx, end_idx))

        self.cycle_boundaries = boundaries
        return boundaries

    def identify_cycle_type(self, cycle_data: pd.DataFrame) -> CycleType:
        """
        Identify the type of cycle based on data characteristics

        Args:
            cycle_data: DataFrame containing single cycle data

        Returns:
            CycleType classification
        """
        # Calculate C-rate
        c_rate_charge = self._calculate_c_rate(cycle_data[cycle_data['current'] > 0])
        c_rate_discharge = self._calculate_c_rate(cycle_data[cycle_data['current'] < 0])

        # Check cycle number patterns (RPT usually at specific intervals)
        cycle_num = cycle_data['total_cycle'].iloc[0] if 'total_cycle' in cycle_data.columns else 0

        # Formation cycles (first 3 cycles)
        if cycle_num <= 3:
            return CycleType.FORMATION

        # Enhanced RPT detection - check if cycle has significantly longer duration
        cycle_duration = cycle_data['elapsed_hours'].max() - cycle_data['elapsed_hours'].min()

        # Store cycle duration for pattern analysis (using instance variable)
        if not hasattr(self, '_cycle_durations'):
            self._cycle_durations = {}
        self._cycle_durations[cycle_num] = cycle_duration

        # Dynamic RPT detection based on cycle duration
        if len(self._cycle_durations) > 10:  # Need some cycles to establish baseline
            durations = list(self._cycle_durations.values())
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)

            # If this cycle is significantly longer than average, it might be RPT
            if cycle_duration > (avg_duration + 1.5 * std_duration):
                if c_rate_discharge <= 0.25:
                    return CycleType.RPT_02C
                elif c_rate_discharge <= 0.55:
                    return CycleType.RPT_05C
                elif c_rate_discharge <= 1.1:
                    return CycleType.RPT_1C

        # Fallback: traditional interval-based detection for common patterns
        if (cycle_num % 50 == 0 or cycle_num % 100 == 0 or
            cycle_num % 5 == 3 or cycle_num % 25 == 0):  # Added more common patterns
            if cycle_duration > 1.5:  # RPT cycles are typically longer
                if c_rate_discharge <= 0.25:
                    return CycleType.RPT_02C
                elif c_rate_discharge <= 0.55:
                    return CycleType.RPT_05C
                elif c_rate_discharge <= 1.1:
                    return CycleType.RPT_1C

        # Pulse test detection (short duration, high current pulses)
        if self._has_pulse_pattern(cycle_data):
            return CycleType.PULSE_TEST

        # Multi-step detection (multiple voltage plateaus)
        if self._has_multistep_pattern(cycle_data):
            return CycleType.MULTI_STEP

        # Rest/OCV detection (zero current for extended period)
        if abs(cycle_data['current'].mean()) < 1.0:  # Near zero current
            if cycle_data['elapsed_hours'].max() - cycle_data['elapsed_hours'].min() > 1.0:
                return CycleType.OCV_MEASUREMENT
            else:
                return CycleType.REST

        # Internal resistance measurement (specific pulse patterns)
        if 'mode' in cycle_data.columns:
            # Mode 12 or 13 might indicate special measurements
            if cycle_data['mode'].iloc[0] in [12, 13]:
                return CycleType.RSS_MEASUREMENT

        # Default to normal charge/discharge
        return CycleType.NORMAL_CHARGE_DISCHARGE

    def calculate_cycle_metrics(self, cycle_data: pd.DataFrame, cycle_num: int) -> CycleMetrics:
        """
        Calculate comprehensive metrics for a single cycle

        Args:
            cycle_data: DataFrame containing single cycle data
            cycle_num: Cycle number

        Returns:
            CycleMetrics object with calculated values
        """
        # Separate charge and discharge data
        charge_data = cycle_data[cycle_data['current'] > 0]
        discharge_data = cycle_data[cycle_data['current'] < 0]

        # Calculate capacities (mAh)
        charge_capacity = self._calculate_capacity(charge_data)
        discharge_capacity = abs(self._calculate_capacity(discharge_data))

        # Calculate energies (Wh)
        charge_energy = self._calculate_energy(charge_data)
        discharge_energy = abs(self._calculate_energy(discharge_data))

        # Efficiencies
        coulombic_efficiency = (discharge_capacity / charge_capacity * 100) if charge_capacity > 0 else 0
        energy_efficiency = (discharge_energy / charge_energy * 100) if charge_energy > 0 else 0

        # Average voltages
        avg_voltage_charge = charge_data['voltage'].mean() if not charge_data.empty else 0
        avg_voltage_discharge = discharge_data['voltage'].mean() if not discharge_data.empty else 0

        # C-rates
        c_rate_charge = self._calculate_c_rate(charge_data)
        c_rate_discharge = self._calculate_c_rate(discharge_data)

        # Duration
        duration = cycle_data['elapsed_hours'].max() - cycle_data['elapsed_hours'].min()

        # Temperature
        max_temp = cycle_data['temperature'].max()
        min_temp = cycle_data['temperature'].min()

        # Internal resistance (if pulse data available)
        internal_resistance = self._estimate_internal_resistance(cycle_data)

        # Identify cycle type
        cycle_type = self.identify_cycle_type(cycle_data)

        return CycleMetrics(
            cycle_number=cycle_num,
            cycle_type=cycle_type,
            charge_capacity_mah=charge_capacity,
            discharge_capacity_mah=discharge_capacity,
            coulombic_efficiency=coulombic_efficiency,
            energy_charge_wh=charge_energy,
            energy_discharge_wh=discharge_energy,
            energy_efficiency=energy_efficiency,
            average_voltage_charge=avg_voltage_charge,
            average_voltage_discharge=avg_voltage_discharge,
            c_rate_charge=c_rate_charge,
            c_rate_discharge=c_rate_discharge,
            duration_hours=duration,
            max_temperature=max_temp,
            min_temperature=min_temp,
            internal_resistance_mohm=internal_resistance
        )

    def analyze_all_cycles(self, df: pd.DataFrame) -> Dict[int, CycleMetrics]:
        """
        Analyze all cycles in the dataset

        Args:
            df: Complete battery dataset

        Returns:
            Dictionary mapping cycle number to CycleMetrics
        """
        # Segment data into cycles
        boundaries = self.segment_cycles(df)

        results = {}

        for i, (start_idx, end_idx) in enumerate(boundaries):
            cycle_data = df.iloc[start_idx:end_idx + 1]

            if not cycle_data.empty:
                # Get cycle number
                if 'total_cycle' in cycle_data.columns:
                    cycle_num = cycle_data['total_cycle'].iloc[0]
                else:
                    cycle_num = i + 1

                # Calculate metrics
                metrics = self.calculate_cycle_metrics(cycle_data, cycle_num)
                results[cycle_num] = metrics

                # Store for later use
                self.cycles[cycle_num] = {
                    'data': cycle_data,
                    'metrics': metrics
                }

        return results

    def identify_test_protocol(self) -> Dict:
        """
        Identify the overall test protocol structure

        Returns:
            Dictionary describing the test protocol
        """
        protocol = {
            'formation_cycles': [],
            'normal_cycles': [],
            'rpt_cycles': [],
            'special_tests': [],
            'total_cycles': len(self.cycles)
        }

        for cycle_num, cycle_info in self.cycles.items():
            cycle_type = cycle_info['metrics'].cycle_type

            if cycle_type == CycleType.FORMATION:
                protocol['formation_cycles'].append(cycle_num)
            elif cycle_type in [CycleType.RPT_02C, CycleType.RPT_05C, CycleType.RPT_1C]:
                protocol['rpt_cycles'].append(cycle_num)
            elif cycle_type in [CycleType.RSS_MEASUREMENT, CycleType.PULSE_TEST, CycleType.HPPC]:
                protocol['special_tests'].append(cycle_num)
            else:
                protocol['normal_cycles'].append(cycle_num)

        # Detect RPT interval
        if len(protocol['rpt_cycles']) > 1:
            intervals = np.diff(protocol['rpt_cycles'])
            protocol['rpt_interval'] = int(np.median(intervals)) if len(intervals) > 0 else None
        else:
            protocol['rpt_interval'] = None

        return protocol

    def get_capacity_retention(self) -> pd.DataFrame:
        """
        Calculate capacity retention over cycles

        Returns:
            DataFrame with cycle number and capacity retention percentage
        """
        data = []

        # Find initial capacity (from formation or first normal cycle)
        initial_capacity = None
        for cycle_num in sorted(self.cycles.keys()):
            metrics = self.cycles[cycle_num]['metrics']
            if metrics.cycle_type in [CycleType.NORMAL_CHARGE_DISCHARGE, CycleType.FORMATION]:
                if initial_capacity is None:
                    initial_capacity = metrics.discharge_capacity_mah
                break

        if initial_capacity and initial_capacity > 0:
            for cycle_num, cycle_info in sorted(self.cycles.items()):
                metrics = cycle_info['metrics']

                # Only consider normal cycles for retention calculation
                if metrics.cycle_type == CycleType.NORMAL_CHARGE_DISCHARGE:
                    retention = (metrics.discharge_capacity_mah / initial_capacity) * 100
                    data.append({
                        'cycle': cycle_num,
                        'discharge_capacity_mah': metrics.discharge_capacity_mah,
                        'capacity_retention_%': retention,
                        'coulombic_efficiency_%': metrics.coulombic_efficiency
                    })

        return pd.DataFrame(data)

    def _calculate_capacity(self, data: pd.DataFrame) -> float:
        """Calculate capacity in mAh using trapezoidal integration"""
        if data.empty or 'current' not in data.columns:
            return 0.0

        # Convert to hours for mAh calculation
        if 'elapsed_hours' in data.columns:
            time_hours = data['elapsed_hours'].values
        else:
            # Assume 1 second intervals if no time data
            time_hours = np.arange(len(data)) / 3600.0

        current_ma = data['current'].values

        # Trapezoidal integration
        capacity_mah = np.trapz(np.abs(current_ma), time_hours)

        return capacity_mah

    def _calculate_energy(self, data: pd.DataFrame) -> float:
        """Calculate energy in Wh"""
        if data.empty:
            return 0.0

        # Calculate power (V * I)
        power_w = data['voltage'].values * data['current'].values / 1000.0  # Convert mA to A

        if 'elapsed_hours' in data.columns:
            time_hours = data['elapsed_hours'].values
        else:
            time_hours = np.arange(len(data)) / 3600.0

        # Integrate power over time
        energy_wh = np.trapz(np.abs(power_w), time_hours)

        return energy_wh

    def _calculate_c_rate(self, data: pd.DataFrame) -> float:
        """Calculate average C-rate"""
        if data.empty or self.nominal_capacity == 0:
            return 0.0

        avg_current = np.abs(data['current'].mean())
        c_rate = avg_current / self.nominal_capacity

        return c_rate

    def _has_pulse_pattern(self, data: pd.DataFrame) -> bool:
        """Detect if cycle has pulse test pattern"""
        if 'current' not in data.columns:
            return False

        # Look for rapid current changes
        current_changes = np.diff(data['current'].values)
        large_changes = np.abs(current_changes) > 100  # mA threshold

        # Pulse pattern has many rapid changes
        return np.sum(large_changes) > 10

    def _has_multistep_pattern(self, data: pd.DataFrame) -> bool:
        """Detect multi-step charge/discharge pattern"""
        if 'voltage' not in data.columns:
            return False

        # Look for voltage plateaus
        voltage_gradient = np.gradient(data['voltage'].values)
        plateaus = np.abs(voltage_gradient) < 0.001  # V/sample threshold

        # Group consecutive plateaus
        plateau_groups = []
        current_group = []

        for i, is_plateau in enumerate(plateaus):
            if is_plateau:
                current_group.append(i)
            elif current_group:
                if len(current_group) > 10:  # Minimum plateau length
                    plateau_groups.append(current_group)
                current_group = []

        # Multi-step has multiple distinct plateaus
        return len(plateau_groups) >= 3

    def _estimate_internal_resistance(self, data: pd.DataFrame) -> Optional[float]:
        """Estimate internal resistance from voltage drop during current change"""
        if 'current' not in data.columns or 'voltage' not in data.columns:
            return None

        # Find large current changes
        current_diff = np.diff(data['current'].values)
        large_changes = np.where(np.abs(current_diff) > 100)[0]  # mA threshold

        if len(large_changes) == 0:
            return None

        resistances = []

        for idx in large_changes[:10]:  # Use first 10 changes
            if idx > 0 and idx < len(data) - 1:
                delta_i = current_diff[idx]
                delta_v = data['voltage'].iloc[idx + 1] - data['voltage'].iloc[idx]

                if abs(delta_i) > 0:
                    # R = ΔV / ΔI (convert to mΩ)
                    r = abs(delta_v / delta_i) * 1000 * 1000  # V/mA to mΩ
                    if 0 < r < 1000:  # Sanity check
                        resistances.append(r)

        if resistances:
            return np.median(resistances)
        else:
            return None

    def post_process_cycle_classification(self, cycles: Dict[int, CycleMetrics]) -> Dict[int, CycleMetrics]:
        """
        Post-process cycle classifications to improve RPT detection
        using global pattern analysis
        """
        if len(cycles) < 10:
            return cycles

        print(f"[RPT] Post-processing {len(cycles)} cycles for better RPT detection...")

        # Collect duration statistics
        durations = [(c.cycle_number, c.duration_hours) for c in cycles.values()]
        durations.sort(key=lambda x: x[0])  # Sort by cycle number

        duration_values = [d[1] for d in durations]
        avg_duration = np.mean(duration_values)
        std_duration = np.std(duration_values)

        print(f"[RPT] Duration stats - Mean: {avg_duration:.2f}h, Std: {std_duration:.2f}h")

        # Find cycles with significantly longer duration (potential RPT)
        long_cycle_threshold = avg_duration + 0.8 * std_duration  # Lowered threshold
        long_cycles = [(num, dur) for num, dur in durations if dur > long_cycle_threshold and num > 5]

        if len(long_cycles) < 3:
            print(f"[RPT] Only {len(long_cycles)} long cycles found, insufficient for pattern analysis")
            return cycles

        print(f"[RPT] Found {len(long_cycles)} long cycles (>{long_cycle_threshold:.2f}h)")

        # Check for periodic pattern in long cycles
        long_cycle_nums = [num for num, _ in long_cycles]
        if len(long_cycle_nums) > 2:
            intervals = np.diff(long_cycle_nums)
            unique_intervals = list(set(intervals))
            print(f"[RPT] Intervals between long cycles: {intervals}")
            print(f"[RPT] Unique intervals: {unique_intervals}")

            # If there's a reasonably consistent interval pattern
            if len(unique_intervals) <= 4:  # Allow some variation
                # Find the most common interval
                interval_counts = {}
                for interval in intervals:
                    interval_counts[interval] = interval_counts.get(interval, 0) + 1

                most_common_interval = max(interval_counts.keys(), key=lambda x: interval_counts[x])
                print(f"[RPT] Most common interval: {most_common_interval} cycles")

                # Reclassify long cycles as RPT
                reclassified_count = 0
                for cycle_num, duration in long_cycles:
                    if cycle_num in cycles:
                        cycle_metrics = cycles[cycle_num]

                        # Only reclassify if not already an RPT or formation cycle
                        if cycle_metrics.cycle_type == CycleType.NORMAL_CHARGE_DISCHARGE:
                            # Determine RPT type based on discharge C-rate
                            if cycle_metrics.c_rate_discharge <= 0.3:
                                new_type = CycleType.RPT_02C
                            elif cycle_metrics.c_rate_discharge <= 0.6:
                                new_type = CycleType.RPT_05C
                            else:
                                new_type = CycleType.RPT_1C

                            # Create new CycleMetrics with updated type
                            cycles[cycle_num] = CycleMetrics(
                                cycle_number=cycle_metrics.cycle_number,
                                cycle_type=new_type,  # Updated type
                                charge_capacity_mah=cycle_metrics.charge_capacity_mah,
                                discharge_capacity_mah=cycle_metrics.discharge_capacity_mah,
                                coulombic_efficiency=cycle_metrics.coulombic_efficiency,
                                energy_charge_wh=cycle_metrics.energy_charge_wh,
                                energy_discharge_wh=cycle_metrics.energy_discharge_wh,
                                energy_efficiency=cycle_metrics.energy_efficiency,
                                average_voltage_charge=cycle_metrics.average_voltage_charge,
                                average_voltage_discharge=cycle_metrics.average_voltage_discharge,
                                c_rate_charge=cycle_metrics.c_rate_charge,
                                c_rate_discharge=cycle_metrics.c_rate_discharge,
                                duration_hours=cycle_metrics.duration_hours,
                                max_temperature=cycle_metrics.max_temperature,
                                min_temperature=cycle_metrics.min_temperature,
                                internal_resistance_mohm=cycle_metrics.internal_resistance_mohm
                            )
                            reclassified_count += 1

                print(f"[RPT] Successfully reclassified {reclassified_count} cycles as RPT")

        return cycles