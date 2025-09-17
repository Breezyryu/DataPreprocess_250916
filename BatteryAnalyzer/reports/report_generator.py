"""
Battery Analysis Report Generation System
Generates comprehensive reports for battery test data analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from jinja2 import Template
import json
from pathlib import Path

from ..analyzers.cycle_analyzer import CycleAnalyzer, CycleType, CycleMetrics
from ..analyzers.degradation_analyzer import DegradationAnalyzer
from ..visualizers.battery_plots import BatteryVisualizer
from ..core.data_loader import BatteryMetadata


class BatteryReportGenerator:
    """
    Comprehensive battery test report generator
    Supports HTML, PDF, and JSON export formats
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        self.visualizer = BatteryVisualizer()

    def generate_comprehensive_report(self,
                                    df: pd.DataFrame,
                                    metadata: BatteryMetadata,
                                    cycle_metrics: Dict[int, CycleMetrics],
                                    degradation_results: Dict[str, Any],
                                    report_format: str = "html") -> str:
        """
        Generate comprehensive battery analysis report

        Args:
            df: Raw battery data
            metadata: Battery metadata
            cycle_metrics: Cycle analysis results
            degradation_results: Degradation analysis results
            report_format: Output format ('html', 'json')

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"battery_report_{metadata.manufacturer}_{timestamp}"

        # Generate all plots
        plots_dir = self.output_dir / "plots" / report_name
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = self._generate_all_plots(
            df, cycle_metrics, degradation_results, plots_dir
        )

        # Prepare report data
        report_data = self._prepare_report_data(
            df, metadata, cycle_metrics, degradation_results, plot_paths
        )

        if report_format.lower() == "html":
            return self._generate_html_report(report_data, report_name)
        elif report_format.lower() == "json":
            return self._generate_json_report(report_data, report_name)
        else:
            raise ValueError(f"Unsupported report format: {report_format}")

    def _generate_all_plots(self,
                          df: pd.DataFrame,
                          cycle_metrics: Dict[int, CycleMetrics],
                          degradation_results: Dict[str, Any],
                          plots_dir: Path) -> Dict[str, str]:
        """Generate all visualization plots"""
        plot_paths = {}

        # Voltage profiles
        try:
            import matplotlib.pyplot as plt
            fig = self.visualizer.plot_voltage_profile(
                df, cycles=list(cycle_metrics.keys())[:5]
            )
            voltage_path = plots_dir / "voltage_profiles.png"
            fig.savefig(voltage_path, dpi=300, bbox_inches='tight')
            plot_paths['voltage_profiles'] = str(voltage_path.relative_to(self.output_dir))
            plt.close(fig)
        except Exception as e:
            print(f"Error generating voltage profiles: {e}")

        # Cycle analysis
        try:
            fig = self.visualizer.plot_cycle_analysis(cycle_metrics)
            cycle_path = plots_dir / "cycle_analysis.png"
            fig.savefig(cycle_path, dpi=300, bbox_inches='tight')
            plot_paths['cycle_analysis'] = str(cycle_path.relative_to(self.output_dir))
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Error generating cycle analysis plot: {e}")
            print(f"[INFO] Cycle analysis data available, but plot generation failed")
            # Continue without the plot

        # Degradation analysis
        try:
            fig = self.visualizer.plot_degradation_analysis(degradation_results)
            degradation_path = plots_dir / "degradation_analysis.png"
            fig.savefig(degradation_path, dpi=300, bbox_inches='tight')
            plot_paths['degradation_analysis'] = str(degradation_path.relative_to(self.output_dir))
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Error generating degradation analysis plot: {e}")
            print(f"[INFO] Degradation analysis data available, but plot generation failed")
            # Continue without the plot

        return plot_paths

    def _prepare_report_data(self,
                           df: pd.DataFrame,
                           metadata: BatteryMetadata,
                           cycle_metrics: Dict[int, CycleMetrics],
                           degradation_results: Dict[str, Any],
                           plot_paths: Dict[str, str]) -> Dict[str, Any]:
        """Prepare comprehensive report data"""

        # Basic statistics
        basic_stats = {
            'total_data_points': len(df),
            'total_cycles': df['total_cycle'].max(),
            'test_duration_hours': df['elapsed_hours'].max(),
            'voltage_range': (df['voltage'].min(), df['voltage'].max()),
            'current_range': (df['current'].min(), df['current'].max()),
            'temperature_range': (df['temperature'].min(), df['temperature'].max()),
            'channels': sorted(df['channel'].unique().tolist())
        }

        # Cycle analysis summary
        cycle_summary = self._summarize_cycle_metrics(cycle_metrics)

        # Degradation summary
        degradation_summary = self._summarize_degradation_results(degradation_results)

        # Performance indicators
        performance = self._calculate_performance_indicators(cycle_metrics, degradation_results)

        report_data = {
            'metadata': {
                'manufacturer': metadata.manufacturer,
                'model': metadata.model,
                'capacity_mah': metadata.capacity_mah,
                'test_condition': metadata.test_condition,
                'cycle_range': metadata.cycle_range,
                'tester_name': metadata.tester_name,
                'start_date': metadata.start_date,
                'end_date': metadata.end_date
            },
            'basic_stats': basic_stats,
            'cycle_summary': cycle_summary,
            'degradation_summary': degradation_summary,
            'performance_indicators': performance,
            'plot_paths': plot_paths,
            'generation_time': datetime.now().isoformat()
        }

        return report_data

    def _summarize_cycle_metrics(self, cycle_metrics: Dict[int, CycleMetrics]) -> Dict[str, Any]:
        """Summarize cycle analysis results"""
        if not cycle_metrics:
            return {}

        # Extract data
        cycles = sorted(cycle_metrics.keys())
        discharge_capacities = [cycle_metrics[c].discharge_capacity_mah for c in cycles]
        coulombic_efficiencies = [cycle_metrics[c].coulombic_efficiency for c in cycles]
        energy_efficiencies = [cycle_metrics[c].energy_efficiency for c in cycles]

        # Calculate retention with zero-division protection
        initial_capacity = discharge_capacities[0] if discharge_capacities else 1
        final_capacity = discharge_capacities[-1] if discharge_capacities else 1

        if initial_capacity > 0:
            capacity_retention = (final_capacity / initial_capacity) * 100
        else:
            capacity_retention = 100.0  # Safe default when initial capacity is 0

        # Cycle type distribution
        cycle_types = [cycle_metrics[c].cycle_type for c in cycles]
        type_counts = {}
        for ct in cycle_types:
            type_counts[ct.value] = type_counts.get(ct.value, 0) + 1

        return {
            'total_cycles': len(cycles),
            'capacity_retention_percent': round(capacity_retention, 2),
            'initial_capacity_mah': round(initial_capacity, 2),
            'final_capacity_mah': round(final_capacity, 2),
            'avg_coulombic_efficiency': round(np.mean(coulombic_efficiencies), 3),
            'avg_energy_efficiency': round(np.mean(energy_efficiencies), 3),
            'cycle_type_distribution': type_counts,
            'capacity_fade_rate': round((100 - capacity_retention) / len(cycles), 4)
        }

    def _summarize_degradation_results(self, degradation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize degradation analysis results"""
        summary = {}

        if 'degradation_mechanisms' in degradation_results:
            mechanisms = degradation_results['degradation_mechanisms']
            if mechanisms:
                latest_cycle = max(mechanisms.keys())
                latest_data = mechanisms[latest_cycle]

                summary['degradation_mechanisms'] = {
                    'lli_loss_percent': round(latest_data.get('lli_percentage', 0), 2),
                    'lam_cathode_percent': round(latest_data.get('lam_cathode_percentage', 0), 2),
                    'kinetic_loss_percent': round(latest_data.get('kinetic_loss_percentage', 0), 2)
                }

        if 'resistance_evolution' in degradation_results:
            resistance_data = degradation_results['resistance_evolution']
            if resistance_data:
                cycles = sorted(resistance_data.keys())
                initial_resistance = resistance_data[cycles[0]].get('total_resistance', 0)
                final_resistance = resistance_data[cycles[-1]].get('total_resistance', 0)

                summary['resistance_evolution'] = {
                    'initial_resistance_mohm': round(initial_resistance, 2),
                    'final_resistance_mohm': round(final_resistance, 2),
                    'resistance_increase_percent': round(
                        ((final_resistance - initial_resistance) / initial_resistance) * 100, 2
                    ) if initial_resistance > 0 else 0
                }

        if 'rul_prediction' in degradation_results:
            rul_data = degradation_results['rul_prediction']
            if rul_data:
                latest_cycle = max(rul_data.keys())
                latest_prediction = rul_data[latest_cycle]

                summary['rul_prediction'] = {
                    'predicted_remaining_cycles': int(latest_prediction.get('predicted_rul', 0)),
                    'prediction_confidence': round(latest_prediction.get('confidence', 0), 3)
                }

        return summary

    def _calculate_performance_indicators(self,
                                        cycle_metrics: Dict[int, CycleMetrics],
                                        degradation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        indicators = {}

        if cycle_metrics:
            cycles = sorted(cycle_metrics.keys())
            discharge_capacities = [cycle_metrics[c].discharge_capacity_mah for c in cycles]

            # Capacity fade analysis
            if len(discharge_capacities) > 1:
                # Linear regression for fade rate
                x = np.array(cycles)
                y = np.array(discharge_capacities)
                coeffs = np.polyfit(x, y, 1)
                fade_rate = abs(coeffs[0])  # mAh per cycle

                indicators['capacity_fade_rate_mah_per_cycle'] = round(fade_rate, 4)

                # 80% capacity point prediction
                initial_capacity = discharge_capacities[0]
                target_capacity = initial_capacity * 0.8

                if fade_rate > 0:
                    cycles_to_80_percent = (initial_capacity - target_capacity) / fade_rate
                    indicators['predicted_cycles_to_80_percent'] = int(cycles_to_80_percent)

        # Battery health assessment
        health_score = self._calculate_health_score(cycle_metrics, degradation_results)
        indicators['overall_health_score'] = round(health_score, 2)
        indicators['health_grade'] = self._get_health_grade(health_score)

        return indicators

    def _calculate_health_score(self,
                              cycle_metrics: Dict[int, CycleMetrics],
                              degradation_results: Dict[str, Any]) -> float:
        """Calculate overall battery health score (0-100)"""
        score = 100.0

        # Capacity retention component (40% weight)
        if cycle_metrics:
            cycles = sorted(cycle_metrics.keys())
            discharge_capacities = [cycle_metrics[c].discharge_capacity_mah for c in cycles]
            if discharge_capacities and discharge_capacities[0] > 0:
                retention = (discharge_capacities[-1] / discharge_capacities[0]) * 100
                capacity_score = min(100, retention)
                score = score * 0.6 + capacity_score * 0.4
            else:
                # Handle case where initial capacity is 0 or no data
                capacity_score = 100.0  # Safe default
                score = score * 0.6 + capacity_score * 0.4

        # Resistance increase component (30% weight)
        if 'resistance_evolution' in degradation_results:
            resistance_data = degradation_results['resistance_evolution']
            if resistance_data:
                cycles = sorted(resistance_data.keys())
                initial_r = resistance_data[cycles[0]].get('total_resistance', 1)
                final_r = resistance_data[cycles[-1]].get('total_resistance', 1)

                resistance_increase = ((final_r - initial_r) / initial_r) * 100
                resistance_score = max(0, 100 - resistance_increase)
                score = score * 0.7 + resistance_score * 0.3

        # Efficiency component (30% weight)
        if cycle_metrics:
            cycles = sorted(cycle_metrics.keys())
            efficiencies = [cycle_metrics[c].coulombic_efficiency for c in cycles]
            if efficiencies:
                avg_efficiency = np.mean(efficiencies)
                efficiency_score = min(100, avg_efficiency)
                score = score * 0.7 + efficiency_score * 0.3

        return max(0, min(100, score))

    def _get_health_grade(self, health_score: float) -> str:
        """Convert health score to grade"""
        if health_score >= 90:
            return "Excellent"
        elif health_score >= 80:
            return "Good"
        elif health_score >= 70:
            return "Fair"
        elif health_score >= 60:
            return "Poor"
        else:
            return "Critical"

    def _generate_html_report(self, report_data: Dict[str, Any], report_name: str) -> str:
        """Generate HTML report"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Battery Analysis Report - {{ report_data.metadata.manufacturer }} {{ report_data.metadata.model }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .section { background: white; padding: 25px; margin: 20px 0; border-radius: 8px;
                  box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                       gap: 20px; margin: 20px 0; }
        .metric-box { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .plot-container { text-align: center; margin: 20px 0; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 8px;
                             box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .health-score { font-size: 3em; font-weight: bold; }
        .excellent { color: #27ae60; }
        .good { color: #f39c12; }
        .fair { color: #e67e22; }
        .poor { color: #e74c3c; }
        .critical { color: #c0392b; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Battery Analysis Report</h1>
        <h2>{{ report_data.metadata.manufacturer }} {{ report_data.metadata.model }}</h2>
        <p>Generated on: {{ report_data.generation_time }}</p>
    </div>

    <div class="section">
        <h2>Test Information</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Manufacturer</td><td>{{ report_data.metadata.manufacturer }}</td></tr>
            <tr><td>Model</td><td>{{ report_data.metadata.model }}</td></tr>
            <tr><td>Nominal Capacity</td><td>{{ report_data.metadata.capacity_mah }} mAh</td></tr>
            <tr><td>Test Condition</td><td>{{ report_data.metadata.test_condition }}</td></tr>
            <tr><td>Cycle Range</td><td>{{ report_data.metadata.cycle_range }}</td></tr>
            <tr><td>Tester</td><td>{{ report_data.metadata.tester_name }}</td></tr>
            <tr><td>Test Period</td><td>{{ report_data.metadata.start_date }} - {{ report_data.metadata.end_date }}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Overall Health Assessment</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="health-score {{ report_data.performance_indicators.health_grade.lower() }}">
                    {{ "%.1f"|format(report_data.performance_indicators.overall_health_score) }}
                </div>
                <div class="metric-label">Health Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ report_data.performance_indicators.health_grade }}</div>
                <div class="metric-label">Health Grade</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "%.2f"|format(report_data.cycle_summary.capacity_retention_percent) }}%</div>
                <div class="metric-label">Capacity Retention</div>
            </div>
            {% if report_data.performance_indicators.predicted_cycles_to_80_percent %}
            <div class="metric-box">
                <div class="metric-value">{{ report_data.performance_indicators.predicted_cycles_to_80_percent }}</div>
                <div class="metric-label">Cycles to 80%</div>
            </div>
            {% endif %}
        </div>
    </div>

    <div class="section">
        <h2>Test Statistics</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">{{ "{:,}"|format(report_data.basic_stats.total_data_points) }}</div>
                <div class="metric-label">Data Points</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ report_data.basic_stats.total_cycles }}</div>
                <div class="metric-label">Total Cycles</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "%.1f"|format(report_data.basic_stats.test_duration_hours) }}</div>
                <div class="metric-label">Test Duration (hours)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "%.3f"|format(report_data.cycle_summary.avg_coulombic_efficiency) }}</div>
                <div class="metric-label">Avg Coulombic Efficiency</div>
            </div>
        </div>
    </div>

    {% if report_data.plot_paths.voltage_profiles %}
    <div class="section">
        <h2>Voltage Profiles</h2>
        <div class="plot-container">
            <img src="{{ report_data.plot_paths.voltage_profiles }}" alt="Voltage Profiles">
        </div>
    </div>
    {% endif %}

    {% if report_data.plot_paths.cycle_analysis %}
    <div class="section">
        <h2>Cycle Analysis</h2>
        <div class="plot-container">
            <img src="{{ report_data.plot_paths.cycle_analysis }}" alt="Cycle Analysis">
        </div>
    </div>
    {% endif %}

    {% if report_data.plot_paths.degradation_analysis %}
    <div class="section">
        <h2>Degradation Analysis</h2>
        <div class="plot-container">
            <img src="{{ report_data.plot_paths.degradation_analysis }}" alt="Degradation Analysis">
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>Cycle Type Distribution</h2>
        <table>
            <tr><th>Cycle Type</th><th>Count</th><th>Percentage</th></tr>
            {% for cycle_type, count in report_data.cycle_summary.cycle_type_distribution.items() %}
            <tr>
                <td>{{ cycle_type }}</td>
                <td>{{ count }}</td>
                <td>{{ "%.1f"|format((count / report_data.cycle_summary.total_cycles) * 100) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    {% if report_data.degradation_summary.degradation_mechanisms %}
    <div class="section">
        <h2>Degradation Mechanisms</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">{{ "%.2f"|format(report_data.degradation_summary.degradation_mechanisms.lli_loss_percent) }}%</div>
                <div class="metric-label">LLI Loss</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "%.2f"|format(report_data.degradation_summary.degradation_mechanisms.lam_cathode_percent) }}%</div>
                <div class="metric-label">LAM Cathode</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "%.2f"|format(report_data.degradation_summary.degradation_mechanisms.kinetic_loss_percent) }}%</div>
                <div class="metric-label">Kinetic Loss</div>
            </div>
        </div>
    </div>
    {% endif %}

</body>
</html>
        """

        # Render template
        template = Template(html_template)
        html_content = template.render(report_data=report_data)

        # Save HTML file
        html_path = self.output_dir / f"{report_name}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(html_path)

    def _generate_json_report(self, report_data: Dict[str, Any], report_name: str) -> str:
        """Generate JSON report"""
        json_path = self.output_dir / f"{report_name}.json"

        # Convert any numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep convert all numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {key: deep_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_numpy(obj)

        clean_data = deep_convert(report_data)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)

        return str(json_path)