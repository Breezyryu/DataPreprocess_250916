"""
Comprehensive Battery Data Visualization Module
Supports cycle analysis, degradation trends, and electrochemical visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from ..analyzers.cycle_analyzer import CycleAnalyzer, CycleType, CycleMetrics
from ..analyzers.degradation_analyzer import DegradationAnalyzer, DegradationMechanism


class BatteryVisualizer:
    """
    Comprehensive battery data visualization system
    Supports both matplotlib and plotly outputs
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer

        Args:
            figsize: Default figure size for matplotlib plots
            style: Matplotlib style
        """
        self.figsize = figsize

        # Set plotting style
        try:
            if 'seaborn' in style:
                import seaborn as sns
                sns.set_style("whitegrid")
            else:
                plt.style.use(style)
        except:
            plt.style.use('default')

        # Color palettes
        self.colors = {
            'charge': '#2E8B57',      # Sea Green
            'discharge': '#DC143C',   # Crimson
            'voltage': '#4169E1',     # Royal Blue
            'temperature': '#FF8C00', # Dark Orange
            'capacity': '#9932CC',    # Dark Orchid
            'resistance': '#8B4513',  # Saddle Brown
        }

        # Cycle type colors
        self.cycle_colors = {
            CycleType.NORMAL_CHARGE_DISCHARGE: '#1f77b4',
            CycleType.RPT_02C: '#ff7f0e',
            CycleType.RPT_05C: '#2ca02c',
            CycleType.RPT_1C: '#d62728',
            CycleType.MULTI_STEP: '#9467bd',
            CycleType.RSS_MEASUREMENT: '#8c564b',
            CycleType.CAPACITY_CHECK: '#e377c2',
            CycleType.FORMATION: '#17becf'
        }

    def plot_voltage_profile(self, df: pd.DataFrame, cycles: List[int] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot voltage profiles for specific cycles

        Args:
            df: Battery data DataFrame
            cycles: List of cycle numbers to plot (default: first 5)
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        if cycles is None:
            cycles = sorted(df['total_cycle'].unique())[:5]

        for cycle in cycles:
            cycle_data = df[df['total_cycle'] == cycle]

            if not cycle_data.empty:
                # Separate charge and discharge
                charge_data = cycle_data[cycle_data['current'] > 0]
                discharge_data = cycle_data[cycle_data['current'] < 0]

                # Voltage vs Capacity
                if not charge_data.empty:
                    charge_cap = np.cumsum(charge_data['current'] *
                                         np.diff(np.append(0, charge_data['elapsed_hours'])))
                    ax1.plot(charge_cap, charge_data['voltage'].values,
                            color=self.colors['charge'], alpha=0.7,
                            label=f'Cycle {cycle} Charge' if cycle == cycles[0] else "")

                if not discharge_data.empty:
                    discharge_cap = np.cumsum(np.abs(discharge_data['current']) *
                                           np.diff(np.append(0, discharge_data['elapsed_hours'])))
                    ax1.plot(discharge_cap, discharge_data['voltage'].values,
                            color=self.colors['discharge'], alpha=0.7,
                            label=f'Cycle {cycle} Discharge' if cycle == cycles[0] else "")

                # Voltage vs Time
                ax2.plot(cycle_data['elapsed_hours'], cycle_data['voltage'],
                        label=f'Cycle {cycle}')

        ax1.set_ylabel('Voltage (V)')
        ax1.set_xlabel('Capacity (mAh)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel('Voltage (V)')
        ax2.set_xlabel('Time (hours)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Battery Voltage Profiles')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cycle_analysis(self, cycle_metrics: Dict[int, CycleMetrics],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cycle analysis results

        Args:
            cycle_metrics: Dictionary of cycle metrics from CycleAnalyzer
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        cycles = sorted(cycle_metrics.keys())

        # Extract data
        charge_capacity = [cycle_metrics[c].charge_capacity_mah for c in cycles]
        discharge_capacity = [cycle_metrics[c].discharge_capacity_mah for c in cycles]
        coulombic_efficiency = [cycle_metrics[c].coulombic_efficiency for c in cycles]
        energy_efficiency = [cycle_metrics[c].energy_efficiency for c in cycles]
        cycle_types = [cycle_metrics[c].cycle_type for c in cycles]

        # Capacity retention with zero-division protection
        if discharge_capacity and discharge_capacity[0] > 0:
            initial_capacity = discharge_capacity[0]
            retention = [(cap / initial_capacity) * 100 for cap in discharge_capacity]
        else:
            # Handle case where initial capacity is 0 or no data
            retention = [100.0] * len(discharge_capacity) if discharge_capacity else [100.0]
            initial_capacity = 1.0  # Safe default

        # Plot 1: Capacity retention
        ax1.plot(cycles, retention, 'o-', color=self.colors['capacity'], linewidth=2)
        ax1.set_ylabel('Capacity Retention (%)')
        ax1.set_xlabel('Cycle Number')
        ax1.set_title('Capacity Retention Over Cycles')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(80, 105)

        # Plot 2: Coulombic efficiency
        colors = [self.cycle_colors.get(ct, '#666666') for ct in cycle_types]
        scatter = ax2.scatter(cycles, coulombic_efficiency, c=colors, s=50, alpha=0.7)
        ax2.set_ylabel('Coulombic Efficiency (%)')
        ax2.set_xlabel('Cycle Number')
        ax2.set_title('Coulombic Efficiency by Cycle Type')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(95, 102)

        # Plot 3: Charge/Discharge capacity comparison
        ax3.plot(cycles, charge_capacity, 'o-', label='Charge',
                color=self.colors['charge'], linewidth=2)
        ax3.plot(cycles, discharge_capacity, 's-', label='Discharge',
                color=self.colors['discharge'], linewidth=2)
        ax3.set_ylabel('Capacity (mAh)')
        ax3.set_xlabel('Cycle Number')
        ax3.set_title('Charge vs Discharge Capacity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Energy efficiency
        ax4.plot(cycles, energy_efficiency, '^-', color=self.colors['voltage'], linewidth=2)
        ax4.set_ylabel('Energy Efficiency (%)')
        ax4.set_xlabel('Cycle Number')
        ax4.set_title('Energy Efficiency Over Cycles')
        ax4.grid(True, alpha=0.3)

        # Create legend for cycle types
        handles = [mpatches.Patch(color=self.cycle_colors.get(ct, '#666666'),
                                label=ct.value)
                  for ct in set(cycle_types)]
        fig.legend(handles, [h.get_label() for h in handles],
                  loc='center right', bbox_to_anchor=(1.02, 0.5))

        plt.suptitle('Comprehensive Cycle Analysis')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_degradation_analysis(self, degradation_results: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot degradation analysis results

        Args:
            degradation_results: Results from DegradationAnalyzer
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # dQ/dV Analysis
        if 'dva_features' in degradation_results:
            dva_data = degradation_results['dva_features']
            cycles = sorted(dva_data.keys())

            for i, cycle in enumerate(cycles[:5]):  # Plot first 5 cycles
                features = dva_data[cycle]
                ax1.plot(features.voltage_range, features.dqdv_curve,
                        label=f'Cycle {cycle}', alpha=0.7)

            ax1.set_xlabel('Voltage (V)')
            ax1.set_ylabel('dQ/dV (mAh/V)')
            ax1.set_title('Differential Voltage Analysis (dQ/dV)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Internal resistance evolution
        if 'resistance_evolution' in degradation_results:
            resistance_data = degradation_results['resistance_evolution']
            cycles = list(resistance_data.keys())
            total_resistance = [resistance_data[c].get('total_resistance', 0) for c in cycles]
            ohmic_resistance = [resistance_data[c].get('ohmic_resistance', 0) for c in cycles]

            ax2.plot(cycles, total_resistance, 'o-', label='Total',
                    color=self.colors['resistance'], linewidth=2)
            ax2.plot(cycles, ohmic_resistance, 's-', label='Ohmic',
                    color='orange', linewidth=2)
            ax2.set_ylabel('Resistance (mΩ)')
            ax2.set_xlabel('Cycle Number')
            ax2.set_title('Internal Resistance Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Degradation mechanisms
        if 'degradation_mechanisms' in degradation_results:
            mechanisms = degradation_results['degradation_mechanisms']
            cycles = list(mechanisms.keys())

            lli_loss = [mechanisms[c].get('lli_percentage', 0) for c in cycles]
            lam_cathode = [mechanisms[c].get('lam_cathode_percentage', 0) for c in cycles]
            kinetic_loss = [mechanisms[c].get('kinetic_loss_percentage', 0) for c in cycles]

            ax3.plot(cycles, lli_loss, 'o-', label='LLI Loss', linewidth=2)
            ax3.plot(cycles, lam_cathode, 's-', label='LAM Cathode', linewidth=2)
            ax3.plot(cycles, kinetic_loss, '^-', label='Kinetic Loss', linewidth=2)
            ax3.set_ylabel('Loss Percentage (%)')
            ax3.set_xlabel('Cycle Number')
            ax3.set_title('Degradation Mechanisms')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # RUL Prediction
        if 'rul_prediction' in degradation_results:
            rul_data = degradation_results['rul_prediction']
            cycles = list(rul_data.keys())
            predicted_rul = [rul_data[c].get('predicted_rul', 0) for c in cycles]
            confidence = [rul_data[c].get('confidence', 0) for c in cycles]

            ax4.plot(cycles, predicted_rul, 'o-', color='red', linewidth=2, label='RUL')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(cycles, confidence, 's-', color='blue', alpha=0.7, label='Confidence')

            ax4.set_ylabel('Remaining Useful Life (cycles)', color='red')
            ax4_twin.set_ylabel('Confidence Score', color='blue')
            ax4.set_xlabel('Cycle Number')
            ax4.set_title('Remaining Useful Life Prediction')
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Battery Degradation Analysis')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_manufacturer_comparison(self, combined_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between manufacturers

        Args:
            combined_df: Combined data from multiple manufacturers
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        manufacturers = combined_df['manufacturer'].unique()

        # Voltage range comparison
        for mfr in manufacturers:
            mfr_data = combined_df[combined_df['manufacturer'] == mfr]
            ax1.hist(mfr_data['voltage'], bins=50, alpha=0.6, label=mfr, density=True)

        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Density')
        ax1.set_title('Voltage Distribution by Manufacturer')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Current range comparison
        for mfr in manufacturers:
            mfr_data = combined_df[combined_df['manufacturer'] == mfr]
            ax2.hist(mfr_data['current'], bins=50, alpha=0.6, label=mfr, density=True)

        ax2.set_xlabel('Current (mA)')
        ax2.set_ylabel('Density')
        ax2.set_title('Current Distribution by Manufacturer')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Temperature comparison
        for mfr in manufacturers:
            mfr_data = combined_df[combined_df['manufacturer'] == mfr]
            ax3.hist(mfr_data['temperature'], bins=30, alpha=0.6, label=mfr, density=True)

        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Density')
        ax3.set_title('Temperature Distribution by Manufacturer')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Cycle count comparison
        cycle_stats = []
        for mfr in manufacturers:
            mfr_data = combined_df[combined_df['manufacturer'] == mfr]
            max_cycles = mfr_data['total_cycle'].max()
            cycle_stats.append(max_cycles)

        bars = ax4.bar(manufacturers, cycle_stats, alpha=0.7)
        ax4.set_ylabel('Maximum Cycles')
        ax4.set_title('Test Duration by Manufacturer')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, cycle_stats):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(value)}', ha='center', va='bottom')

        plt.suptitle('Manufacturer Performance Comparison')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_interactive_dashboard(self, cycle_metrics: Dict[int, CycleMetrics],
                                   degradation_results: Dict[str, Any]) -> go.Figure:
        """
        Create interactive Plotly dashboard

        Args:
            cycle_metrics: Cycle analysis results
            degradation_results: Degradation analysis results

        Returns:
            Plotly Figure object
        """
        cycles = sorted(cycle_metrics.keys())

        # Extract data
        discharge_capacity = [cycle_metrics[c].discharge_capacity_mah for c in cycles]
        coulombic_efficiency = [cycle_metrics[c].coulombic_efficiency for c in cycles]

        # Calculate retention with zero-division protection
        if discharge_capacity and discharge_capacity[0] > 0:
            initial_capacity = discharge_capacity[0]
            retention = [(cap / initial_capacity) * 100 for cap in discharge_capacity]
        else:
            # Handle case where initial capacity is 0 or no data
            retention = [100.0] * len(discharge_capacity) if discharge_capacity else [100.0]
            initial_capacity = 1.0  # Safe default

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Capacity Retention', 'Coulombic Efficiency',
                          'Internal Resistance', 'RUL Prediction'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )

        # Capacity retention
        fig.add_trace(
            go.Scatter(x=cycles, y=retention, mode='lines+markers',
                      name='Capacity Retention', line=dict(color='blue', width=3)),
            row=1, col=1
        )

        # Coulombic efficiency
        cycle_types = [cycle_metrics[c].cycle_type.value for c in cycles]
        fig.add_trace(
            go.Scatter(x=cycles, y=coulombic_efficiency, mode='markers',
                      name='Coulombic Efficiency', text=cycle_types,
                      marker=dict(size=8, color='red')),
            row=1, col=2
        )

        # Internal resistance (if available)
        if 'resistance_evolution' in degradation_results:
            resistance_data = degradation_results['resistance_evolution']
            res_cycles = list(resistance_data.keys())
            total_resistance = [resistance_data[c].get('total_resistance', 0)
                              for c in res_cycles]

            fig.add_trace(
                go.Scatter(x=res_cycles, y=total_resistance, mode='lines+markers',
                          name='Internal Resistance', line=dict(color='orange', width=2)),
                row=2, col=1
            )

        # RUL Prediction (if available)
        if 'rul_prediction' in degradation_results:
            rul_data = degradation_results['rul_prediction']
            rul_cycles = list(rul_data.keys())
            predicted_rul = [rul_data[c].get('predicted_rul', 0) for c in rul_cycles]
            confidence = [rul_data[c].get('confidence', 0) for c in rul_cycles]

            fig.add_trace(
                go.Scatter(x=rul_cycles, y=predicted_rul, mode='lines+markers',
                          name='RUL', line=dict(color='red', width=3)),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(x=rul_cycles, y=confidence, mode='lines+markers',
                          name='Confidence', line=dict(color='blue', width=2),
                          yaxis='y4'),
                row=2, col=2, secondary_y=True
            )

        # Update layout
        fig.update_layout(
            title_text="Interactive Battery Analysis Dashboard",
            height=800,
            showlegend=True
        )

        # Update y-axes
        fig.update_yaxes(title_text="Retention (%)", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
        fig.update_yaxes(title_text="Resistance (mΩ)", row=2, col=1)
        fig.update_yaxes(title_text="RUL (cycles)", row=2, col=2)
        fig.update_yaxes(title_text="Confidence", row=2, col=2, secondary_y=True)

        # Update x-axes
        fig.update_xaxes(title_text="Cycle Number", row=1, col=1)
        fig.update_xaxes(title_text="Cycle Number", row=1, col=2)
        fig.update_xaxes(title_text="Cycle Number", row=2, col=1)
        fig.update_xaxes(title_text="Cycle Number", row=2, col=2)

        return fig