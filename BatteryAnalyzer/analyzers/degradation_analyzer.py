"""
Electrochemical Degradation Analysis Module
Advanced analysis of battery degradation mechanisms using electrochemical methods
"""

import pandas as pd
import numpy as np
from scipy import signal, optimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class DegradationMechanism(Enum):
    """Primary degradation mechanisms in lithium-ion batteries"""
    LLI = "Loss of Lithium Inventory"  # Cyclable Li+ loss
    LAM_CATHODE = "Loss of Active Material - Cathode"
    LAM_ANODE = "Loss of Active Material - Anode"
    SEI_GROWTH = "SEI Layer Growth"
    PLATING = "Lithium Plating"
    RESISTANCE_RISE = "Internal Resistance Rise"


@dataclass
class DVAFeatures:
    """Differential Voltage Analysis features"""
    peaks: List[Tuple[float, float]]  # (voltage, dq/dv)
    peak_shifts: List[float]  # Peak position changes
    peak_intensities: List[float]  # Peak height changes
    new_peaks: List[Tuple[float, float]]  # Newly appeared peaks
    baseline_shift: float


@dataclass
class DegradationMetrics:
    """Comprehensive degradation analysis results"""
    cycle_number: int
    capacity_fade_percent: float
    resistance_growth_percent: float
    lli_percent: float
    lam_cathode_percent: float
    lam_anode_percent: float
    dominant_mechanism: DegradationMechanism
    mechanism_confidence: float
    predicted_eol_cycles: Optional[int]


class DegradationAnalyzer:
    """
    Electrochemical-based battery degradation analysis system
    """

    def __init__(self, nominal_capacity_mah: float = 1689.0):
        """
        Initialize degradation analyzer

        Args:
            nominal_capacity_mah: Nominal battery capacity for analysis
        """
        self.nominal_capacity = nominal_capacity_mah
        self.reference_cycles = {}  # Store reference cycle data
        self.degradation_history = []

    def differential_voltage_analysis(self, cycle_data: pd.DataFrame,
                                     cycle_number: int,
                                     reference_cycle: Optional[pd.DataFrame] = None) -> DVAFeatures:
        """
        Perform Differential Voltage Analysis (dQ/dV analysis)

        Args:
            cycle_data: Current cycle data
            cycle_number: Cycle number
            reference_cycle: Reference cycle for comparison (usually first cycle)

        Returns:
            DVAFeatures with extracted features
        """
        # Separate charge and discharge
        discharge_data = cycle_data[cycle_data['current'] < 0].copy()

        if len(discharge_data) < 10:
            return DVAFeatures([], [], [], [], 0.0)

        # Sort by voltage
        discharge_data = discharge_data.sort_values('voltage')

        # Calculate cumulative capacity
        if 'elapsed_hours' in discharge_data.columns:
            dt = np.diff(discharge_data['elapsed_hours'].values) * 3600  # Convert to seconds
            current_ma = discharge_data['current'].values[:-1]
            dq = np.abs(current_ma * dt) / 3600.0  # mAh
            cumulative_q = np.cumsum(np.concatenate([[0], dq]))
        else:
            # Estimate capacity based on current
            cumulative_q = np.cumsum(np.abs(discharge_data['current'].values)) / 3600.0

        voltage = discharge_data['voltage'].values

        # Calculate dQ/dV using smoothed derivative
        try:
            # Smooth the data first
            if len(voltage) > 5:
                q_smooth = signal.savgol_filter(cumulative_q,
                                              min(len(cumulative_q), 11), 3)
                v_smooth = signal.savgol_filter(voltage,
                                              min(len(voltage), 11), 3)

                dqdv = np.gradient(q_smooth, v_smooth)
            else:
                dqdv = np.gradient(cumulative_q, voltage)

            # Find peaks
            peaks, properties = signal.find_peaks(dqdv, height=1.0, distance=10)
            peak_voltages = voltage[peaks]
            peak_heights = dqdv[peaks]

            peak_list = list(zip(peak_voltages, peak_heights))

        except Exception as e:
            print(f"DVA calculation error for cycle {cycle_number}: {e}")
            return DVAFeatures([], [], [], [], 0.0)

        # Compare with reference cycle if provided
        peak_shifts = []
        peak_intensities = []
        new_peaks = []
        baseline_shift = 0.0

        if reference_cycle is not None and cycle_number not in self.reference_cycles:
            # Store reference features
            ref_features = self.differential_voltage_analysis(reference_cycle, 0)
            self.reference_cycles[0] = ref_features

        if 0 in self.reference_cycles:
            ref_peaks = self.reference_cycles[0].peaks

            # Calculate peak shifts and intensity changes
            for ref_v, ref_h in ref_peaks:
                # Find closest peak in current cycle
                if peak_list:
                    distances = [abs(v - ref_v) for v, h in peak_list]
                    min_idx = np.argmin(distances)
                    curr_v, curr_h = peak_list[min_idx]

                    if abs(curr_v - ref_v) < 0.1:  # Same peak
                        peak_shifts.append(curr_v - ref_v)
                        peak_intensities.append(curr_h / ref_h if ref_h > 0 else 1.0)

            # Find new peaks
            for curr_v, curr_h in peak_list:
                is_new = True
                for ref_v, ref_h in ref_peaks:
                    if abs(curr_v - ref_v) < 0.05:  # Close to existing peak
                        is_new = False
                        break
                if is_new and curr_h > 2.0:  # Significant new peak
                    new_peaks.append((curr_v, curr_h))

        return DVAFeatures(
            peaks=peak_list,
            peak_shifts=peak_shifts,
            peak_intensities=peak_intensities,
            new_peaks=new_peaks,
            baseline_shift=baseline_shift
        )

    def analyze_internal_resistance(self, cycle_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze internal resistance components

        Args:
            cycle_data: Single cycle data

        Returns:
            Dictionary with resistance components in mΩ
        """
        resistance_components = {
            'ohmic_resistance': 0.0,
            'sei_resistance': 0.0,
            'charge_transfer_resistance': 0.0,
            'total_dcr': 0.0
        }

        # Find current pulse transitions for DCR calculation
        current_diff = np.diff(cycle_data['current'].values)
        large_changes = np.where(np.abs(current_diff) > 500)[0]  # mA threshold

        if len(large_changes) == 0:
            return resistance_components

        dcr_values = []

        for idx in large_changes[:5]:  # Analyze first 5 pulses
            if idx > 0 and idx < len(cycle_data) - 2:
                # Get voltage response to current step
                delta_i = current_diff[idx]  # mA
                delta_v_immediate = (cycle_data['voltage'].iloc[idx + 1] -
                                   cycle_data['voltage'].iloc[idx])  # V

                if abs(delta_i) > 100:  # Significant current change
                    dcr_mohm = abs(delta_v_immediate / delta_i) * 1000 * 1000  # Convert to mΩ

                    if 0 < dcr_mohm < 500:  # Sanity check
                        dcr_values.append(dcr_mohm)

        if dcr_values:
            resistance_components['total_dcr'] = np.median(dcr_values)

            # Estimate component breakdown (simplified model)
            total_r = resistance_components['total_dcr']
            resistance_components['ohmic_resistance'] = total_r * 0.3  # ~30% ohmic
            resistance_components['sei_resistance'] = total_r * 0.4    # ~40% SEI
            resistance_components['charge_transfer_resistance'] = total_r * 0.3  # ~30% charge transfer

        return resistance_components

    def quantify_degradation_mechanisms(self, current_cycle: pd.DataFrame,
                                      reference_cycle: pd.DataFrame,
                                      cycle_number: int) -> Dict[str, float]:
        """
        Quantify individual degradation mechanisms using curve fitting

        Args:
            current_cycle: Current cycle data
            reference_cycle: Reference cycle (usually first cycle)
            cycle_number: Current cycle number

        Returns:
            Dictionary with mechanism contributions (percentage)
        """
        mechanisms = {
            'LLI': 0.0,
            'LAM_cathode': 0.0,
            'LAM_anode': 0.0,
            'kinetic_loss': 0.0
        }

        try:
            # Get discharge curves
            ref_discharge = reference_cycle[reference_cycle['current'] < 0].copy()
            curr_discharge = current_cycle[current_cycle['current'] < 0].copy()

            if len(ref_discharge) < 10 or len(curr_discharge) < 10:
                return mechanisms

            # Calculate capacity loss
            ref_capacity = self._calculate_capacity_from_data(ref_discharge)
            curr_capacity = self._calculate_capacity_from_data(curr_discharge)
            total_loss = ref_capacity - curr_capacity

            if total_loss <= 0:
                return mechanisms

            # DVA analysis for mechanism identification
            dva_current = self.differential_voltage_analysis(current_cycle, cycle_number)
            dva_ref = self.differential_voltage_analysis(reference_cycle, 0)

            # LLI estimation from voltage curve shift
            lli_loss = self._estimate_lli_from_dva(dva_current, dva_ref)

            # LAM estimation from peak intensity changes
            lam_cathode_loss = self._estimate_lam_cathode(dva_current, dva_ref)
            lam_anode_loss = self._estimate_lam_anode(dva_current, dva_ref)

            # Kinetic loss from resistance increase
            kinetic_loss = max(0, total_loss - lli_loss - lam_cathode_loss - lam_anode_loss)

            # Normalize to percentages of total capacity
            if total_loss > 0:
                mechanisms['LLI'] = (lli_loss / self.nominal_capacity) * 100
                mechanisms['LAM_cathode'] = (lam_cathode_loss / self.nominal_capacity) * 100
                mechanisms['LAM_anode'] = (lam_anode_loss / self.nominal_capacity) * 100
                mechanisms['kinetic_loss'] = (kinetic_loss / self.nominal_capacity) * 100

        except Exception as e:
            print(f"Error quantifying degradation mechanisms for cycle {cycle_number}: {e}")

        return mechanisms

    def predict_remaining_useful_life(self, capacity_data: pd.DataFrame,
                                    resistance_data: Optional[pd.DataFrame] = None,
                                    eol_threshold: float = 0.8) -> Dict:
        """
        Predict Remaining Useful Life (RUL) using degradation models

        Args:
            capacity_data: DataFrame with cycle and capacity data
            resistance_data: Optional resistance evolution data
            eol_threshold: End-of-life capacity retention threshold

        Returns:
            Dictionary with RUL prediction results
        """
        if len(capacity_data) < 10:
            return {'predicted_eol_cycles': None, 'confidence': 0.0}

        cycles = capacity_data['cycle'].values
        capacities = capacity_data['discharge_capacity_mah'].values

        try:
            # Fit degradation models

            # Model 1: Linear degradation
            linear_fit = np.polyfit(cycles, capacities, 1)
            linear_pred = np.poly1d(linear_fit)

            # Model 2: Square root degradation (calendar aging)
            sqrt_fit = optimize.curve_fit(
                lambda x, a, b: a * np.sqrt(x) + b,
                cycles, capacities,
                p0=[-1, capacities[0]]
            )[0]

            # Model 3: Exponential degradation
            try:
                exp_fit = optimize.curve_fit(
                    lambda x, a, b, c: a * np.exp(-b * x) + c,
                    cycles, capacities,
                    p0=[capacities[0] - capacities[-1], 0.001, capacities[-1]],
                    maxfev=1000
                )[0]
                exp_valid = True
            except:
                exp_valid = False

            # Calculate R-squared for each model
            linear_r2 = self._calculate_r_squared(capacities, linear_pred(cycles))
            sqrt_r2 = self._calculate_r_squared(capacities,
                                              sqrt_fit[0] * np.sqrt(cycles) + sqrt_fit[1])

            if exp_valid:
                exp_pred = exp_fit[0] * np.exp(-exp_fit[1] * cycles) + exp_fit[2]
                exp_r2 = self._calculate_r_squared(capacities, exp_pred)
            else:
                exp_r2 = 0.0

            # Select best model
            models = {'linear': linear_r2, 'sqrt': sqrt_r2}
            if exp_valid:
                models['exp'] = exp_r2

            best_model = max(models, key=models.get)
            confidence = models[best_model]

            # Predict EOL
            eol_capacity = self.nominal_capacity * eol_threshold
            current_cycle = cycles[-1]

            if best_model == 'linear':
                # Solve linear equation: eol_capacity = a * cycle + b
                a, b = linear_fit
                if a != 0:
                    eol_cycle = (eol_capacity - b) / a
                else:
                    eol_cycle = None

            elif best_model == 'sqrt':
                # Solve sqrt equation: eol_capacity = a * sqrt(cycle) + b
                a, b = sqrt_fit
                if a != 0:
                    eol_cycle = ((eol_capacity - b) / a) ** 2
                else:
                    eol_cycle = None

            elif best_model == 'exp' and exp_valid:
                # Solve exponential equation numerically
                a, b, c = exp_fit
                if b > 0:
                    eol_cycle = -np.log((eol_capacity - c) / a) / b if a > 0 else None
                else:
                    eol_cycle = None

            else:
                eol_cycle = None

            remaining_cycles = max(0, eol_cycle - current_cycle) if eol_cycle else None

        except Exception as e:
            print(f"RUL prediction error: {e}")
            remaining_cycles = None
            confidence = 0.0
            best_model = 'none'

        return {
            'predicted_eol_cycles': int(remaining_cycles) if remaining_cycles else None,
            'confidence': confidence,
            'best_model': best_model,
            'current_cycle': int(current_cycle),
            'eol_threshold': eol_threshold
        }

    def comprehensive_degradation_analysis(self, cycle_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Perform comprehensive degradation analysis on all cycles

        Args:
            cycle_data: Dictionary mapping cycle numbers to cycle DataFrames

        Returns:
            DataFrame with degradation analysis results
        """
        results = []

        # Get reference cycle (first available cycle)
        reference_cycle_num = min(cycle_data.keys())
        reference_cycle = cycle_data[reference_cycle_num]

        for cycle_num in sorted(cycle_data.keys()):
            current_cycle = cycle_data[cycle_num]

            # DVA analysis
            dva_features = self.differential_voltage_analysis(current_cycle, cycle_num,
                                                            reference_cycle)

            # Resistance analysis
            resistance_info = self.analyze_internal_resistance(current_cycle)

            # Mechanism quantification
            mechanisms = self.quantify_degradation_mechanisms(current_cycle,
                                                            reference_cycle,
                                                            cycle_num)

            # Calculate capacity fade
            ref_capacity = self._calculate_capacity_from_data(
                reference_cycle[reference_cycle['current'] < 0]
            )
            curr_capacity = self._calculate_capacity_from_data(
                current_cycle[current_cycle['current'] < 0]
            )
            capacity_fade = ((ref_capacity - curr_capacity) / ref_capacity) * 100

            # Calculate resistance growth
            ref_resistance = self.analyze_internal_resistance(reference_cycle)['total_dcr']
            resistance_growth = ((resistance_info['total_dcr'] - ref_resistance) /
                               ref_resistance * 100) if ref_resistance > 0 else 0

            # Determine dominant mechanism
            mechanism_values = {k: v for k, v in mechanisms.items() if k != 'kinetic_loss'}
            if mechanism_values:
                dominant = max(mechanism_values, key=mechanism_values.get)
                confidence = mechanism_values[dominant]
            else:
                dominant = 'Unknown'
                confidence = 0.0

            results.append({
                'cycle': cycle_num,
                'capacity_fade_%': capacity_fade,
                'resistance_growth_%': resistance_growth,
                'lli_%': mechanisms['LLI'],
                'lam_cathode_%': mechanisms['LAM_cathode'],
                'lam_anode_%': mechanisms['LAM_anode'],
                'kinetic_loss_%': mechanisms['kinetic_loss'],
                'total_dcr_mohm': resistance_info['total_dcr'],
                'ohmic_r_mohm': resistance_info['ohmic_resistance'],
                'sei_r_mohm': resistance_info['sei_resistance'],
                'ct_r_mohm': resistance_info['charge_transfer_resistance'],
                'dva_peak_count': len(dva_features.peaks),
                'dominant_mechanism': dominant,
                'mechanism_confidence': confidence
            })

        return pd.DataFrame(results)

    def _calculate_capacity_from_data(self, discharge_data: pd.DataFrame) -> float:
        """Calculate discharge capacity from data"""
        if discharge_data.empty:
            return 0.0

        if 'elapsed_hours' in discharge_data.columns:
            time_diff = np.diff(discharge_data['elapsed_hours'].values) * 3600  # seconds
            current_values = np.abs(discharge_data['current'].values[:-1])  # mA
            capacity = np.sum(current_values * time_diff) / 3600.0  # mAh
        else:
            # Rough estimation
            capacity = np.mean(np.abs(discharge_data['current'])) * len(discharge_data) / 3600.0

        return capacity

    def _estimate_lli_from_dva(self, current_dva: DVAFeatures, ref_dva: DVAFeatures) -> float:
        """Estimate LLI from DVA peak shifts"""
        if not current_dva.peak_shifts:
            return 0.0

        # LLI typically causes uniform voltage shift
        avg_shift = np.mean(current_dva.peak_shifts)
        # Convert voltage shift to capacity loss (rough estimation)
        lli_capacity = abs(avg_shift) * self.nominal_capacity * 0.1  # mAh

        return min(lli_capacity, self.nominal_capacity * 0.2)  # Cap at 20%

    def _estimate_lam_cathode(self, current_dva: DVAFeatures, ref_dva: DVAFeatures) -> float:
        """Estimate LAM cathode from DVA peak intensity changes"""
        if not current_dva.peak_intensities:
            return 0.0

        # Cathode LAM causes peak intensity reduction
        intensity_reduction = 1.0 - np.mean(current_dva.peak_intensities)
        lam_capacity = max(0, intensity_reduction * self.nominal_capacity * 0.3)

        return min(lam_capacity, self.nominal_capacity * 0.15)  # Cap at 15%

    def _estimate_lam_anode(self, current_dva: DVAFeatures, ref_dva: DVAFeatures) -> float:
        """Estimate LAM anode from DVA features"""
        # Simplified estimation based on new peak formation
        new_peak_factor = len(current_dva.new_peaks) * 0.02
        lam_capacity = new_peak_factor * self.nominal_capacity

        return min(lam_capacity, self.nominal_capacity * 0.1)  # Cap at 10%

    def _calculate_r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared value"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)