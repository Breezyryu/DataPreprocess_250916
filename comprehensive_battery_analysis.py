#!/usr/bin/env python3
"""
Comprehensive Battery Data Analysis Script
Universal system for battery life and performance test data analysis

Usage:
    python comprehensive_battery_analysis.py [options]

Options:
    --dataset DATASET_NAME    : Specific dataset to analyze (optional)
    --channels CHANNELS       : Comma-separated channel numbers (e.g., 30,31)
    --output-dir DIR         : Output directory for reports and plots
    --report-format FORMAT   : Report format (html, json, both)
    --interactive           : Generate interactive dashboard
    --manufacturer MFR       : Filter by manufacturer (ATL, Toyo, PNE)
    --compare               : Compare multiple manufacturers
    --verbose              : Enable verbose output

Examples:
    # Analyze all available data
    python comprehensive_battery_analysis.py

    # Analyze specific dataset
    python comprehensive_battery_analysis.py --dataset "250207_250307_3_김동진_1689mAh_ATL Q7M Inner 2C 상온수명 1-100cyc"

    # Compare manufacturers
    python comprehensive_battery_analysis.py --compare --output-dir comparison_results

    # Generate interactive dashboard
    python comprehensive_battery_analysis.py --interactive --report-format both
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for console output on Windows
if sys.platform.startswith('win'):
    try:
        import codecs
        import io
        # Try to set UTF-8 encoding safely
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # If UTF-8 setup fails, continue with default encoding
        print("Warning: Could not set UTF-8 encoding, using default")

# Add BatteryAnalyzer to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BatteryAnalyzer.core.data_loader import UniversalBatteryLoader
from BatteryAnalyzer.analyzers.cycle_analyzer import CycleAnalyzer
from BatteryAnalyzer.analyzers.degradation_analyzer import DegradationAnalyzer
from BatteryAnalyzer.visualizers.battery_plots import BatteryVisualizer
from BatteryAnalyzer.reports.report_generator import BatteryReportGenerator


class ComprehensiveBatteryAnalysis:
    """
    Main orchestrator for comprehensive battery analysis
    Coordinates all analysis modules and generates comprehensive reports
    """

    def __init__(self, base_path: str = "Rawdata", output_dir: str = "analysis_results"):
        """
        Initialize comprehensive analysis system

        Args:
            base_path: Path to raw data directory
            output_dir: Output directory for results
        """
        self.base_path = base_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize analysis modules
        self.loader = UniversalBatteryLoader(base_path)
        self.cycle_analyzer = CycleAnalyzer()
        self.degradation_analyzer = DegradationAnalyzer()
        self.visualizer = BatteryVisualizer()
        self.report_generator = BatteryReportGenerator(str(self.output_dir))

        print(f"[OK] Comprehensive Battery Analysis System Initialized")
        print(f"   Data Path: {self.base_path}")
        print(f"   Output Path: {self.output_dir}")

    def scan_available_datasets(self) -> None:
        """Scan and display available datasets"""
        print("\n[SCAN] Scanning available datasets...")
        datasets = self.loader.scan_datasets()

        if not datasets:
            print("[ERROR] No datasets found!")
            return

        print(f"\n[DATA] Found {len(datasets)} datasets:")
        for i, dataset in enumerate(datasets, 1):
            metadata = self.loader.metadata[dataset]
            print(f"   {i}. {dataset}")
            print(f"      Manufacturer: {metadata.manufacturer}")
            print(f"      Model: {metadata.model}")
            print(f"      Capacity: {metadata.capacity_mah} mAh")
            print(f"      Condition: {metadata.test_condition}")
            print("")

    def analyze_single_dataset(self,
                             dataset_name: str,
                             channels: list = None,
                             report_format: str = "html",
                             generate_interactive: bool = False) -> dict:
        """
        Perform comprehensive analysis on a single dataset

        Args:
            dataset_name: Name of dataset to analyze
            channels: Specific channels to load
            report_format: Report format ('html', 'json', 'both')
            generate_interactive: Whether to generate interactive dashboard

        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n[ANALYZE] Analyzing dataset: {dataset_name}")

        # Load data
        print("[LOAD] Loading data...")
        df = self.loader.load_dataset(dataset_name, channels)

        if df.empty:
            print("[ERROR] No data loaded!")
            return {}

        print(f"   [OK] Loaded {len(df):,} data points")
        print(f"   [INFO] Channels: {sorted(df['channel'].unique())}")
        print(f"   [INFO] Cycles: {df['total_cycle'].min()} - {df['total_cycle'].max()}")
        print(f"   [INFO] Duration: {df['elapsed_hours'].max():.1f} hours")

        # Get metadata
        metadata = self.loader.metadata[dataset_name]

        # Perform cycle analysis
        print("\n[CYCLE] Performing cycle analysis...")
        cycle_metrics = self.cycle_analyzer.analyze_all_cycles(df)
        print(f"   [OK] Analyzed {len(cycle_metrics)} cycles")

        # Post-process for better RPT detection
        print("   [ENHANCE] Improving cycle classification...")
        cycle_metrics = self.cycle_analyzer.post_process_cycle_classification(cycle_metrics)
        print(f"   [OK] Enhanced cycle classification completed")

        # Identify test protocol
        protocol = self.cycle_analyzer.identify_test_protocol()
        print(f"   [PROTOCOL] Test Protocol:")
        print(f"      Formation cycles: {len(protocol['formation_cycles'])}")
        print(f"      Normal cycles: {len(protocol['normal_cycles'])}")
        print(f"      RPT cycles: {len(protocol['rpt_cycles'])}")
        if protocol['rpt_interval']:
            print(f"      RPT interval: {protocol['rpt_interval']} cycles")

        # Perform degradation analysis
        print("\n[DEGRAD] Performing degradation analysis...")
        degradation_results = {}

        # Calculate basic degradation metrics
        if not cycle_metrics:
            print("   [WARN] No cycle data for degradation analysis")
            degradation_results = {}
        else:
            print("   [BASIC] Calculating basic degradation metrics...")

            # Calculate capacity retention
            cycle_numbers = sorted(cycle_metrics.keys())
            discharge_capacities = [cycle_metrics[c].discharge_capacity_mah for c in cycle_numbers]

            if discharge_capacities and discharge_capacities[0] > 0:
                initial_capacity = discharge_capacities[0]
                capacity_retention = [(cap / initial_capacity) * 100 if initial_capacity > 0 else 0 for cap in discharge_capacities]

                degradation_results['capacity_retention'] = {
                    'cycles': cycle_numbers,
                    'retention_percent': capacity_retention,
                    'fade_rate_per_cycle': (100 - capacity_retention[-1]) / len(cycle_numbers) if len(cycle_numbers) > 1 and capacity_retention else 0
                }

            # Calculate basic resistance evolution
            resistance_evolution = {}
            for cycle_num in cycle_numbers[::max(1, len(cycle_numbers)//10)]:  # Sample every 10%
                metrics = cycle_metrics[cycle_num]
                if metrics.internal_resistance_mohm:
                    resistance_evolution[cycle_num] = {
                        'total_resistance': metrics.internal_resistance_mohm
                    }

            degradation_results['resistance_evolution'] = resistance_evolution

        print(f"   [OK] Degradation analysis completed")

        # Generate comprehensive report
        print(f"\n[REPORT] Generating comprehensive report...")
        try:
            if report_format in ['html', 'both']:
                html_report = self.report_generator.generate_comprehensive_report(
                    df, metadata, cycle_metrics, degradation_results, 'html'
                )
                print(f"   [OK] HTML report: {html_report}")

            if report_format in ['json', 'both']:
                json_report = self.report_generator.generate_comprehensive_report(
                    df, metadata, cycle_metrics, degradation_results, 'json'
                )
                print(f"   [OK] JSON report: {json_report}")

        except Exception as e:
            import traceback
            print(f"   [WARN] Report generation error: {e}")
            print(f"   [DEBUG] Full traceback:")
            traceback.print_exc()
            print(f"   [INFO] Analysis data is still available, continuing without full report")

        # Generate interactive dashboard
        if generate_interactive and cycle_metrics and degradation_results:
            print(f"\n[DASH] Generating interactive dashboard...")
            try:
                dashboard = self.visualizer.create_interactive_dashboard(
                    cycle_metrics, degradation_results
                )
                dashboard_path = self.output_dir / f"interactive_dashboard_{dataset_name}.html"
                dashboard.write_html(str(dashboard_path))
                print(f"   [OK] Interactive dashboard: {dashboard_path}")
            except Exception as e:
                print(f"   [WARN] Dashboard generation error: {e}")

        # Compile results
        results = {
            'dataset_name': dataset_name,
            'metadata': metadata,
            'data_summary': {
                'total_points': len(df),
                'channels': sorted(df['channel'].unique()),
                'cycle_range': (df['total_cycle'].min(), df['total_cycle'].max()),
                'duration_hours': df['elapsed_hours'].max()
            },
            'cycle_metrics': cycle_metrics,
            'degradation_results': degradation_results,
            'test_protocol': protocol
        }

        return results

    def compare_manufacturers(self,
                           channels: list = None,
                           output_dir: str = None) -> dict:
        """
        Compare performance across manufacturers

        Args:
            channels: Specific channels to analyze
            output_dir: Custom output directory

        Returns:
            Comparison results dictionary
        """
        print("\n[COMPARE] Loading all datasets for manufacturer comparison...")

        # Load all data
        combined_df = self.loader.load_all_datasets(channels)

        if combined_df.empty:
            print("[ERROR] No data available for comparison!")
            return {}

        print(f"[OK] Loaded {len(combined_df):,} total data points")

        # Generate comparison statistics
        comparison_stats = self.loader.compare_manufacturers(combined_df)

        # Generate comparison visualization
        print("\n[PLOT] Generating manufacturer comparison plots...")
        try:
            comparison_plot = self.visualizer.plot_manufacturer_comparison(combined_df)
            plot_path = self.output_dir / "manufacturer_comparison.png"
            comparison_plot.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(comparison_plot)
            print(f"   [OK] Comparison plot: {plot_path}")
        except Exception as e:
            print(f"   [WARN] Comparison plot error: {e}")

        # Print summary
        print("\n[SUMMARY] Manufacturer Comparison Summary:")
        for manufacturer, stats in comparison_stats.items():
            print(f"\n   {manufacturer}:")
            print(f"      Total cycles: {stats['total_cycles']}")
            print(f"      Voltage range: {stats['voltage_range'][0]:.3f} - {stats['voltage_range'][1]:.3f} V")
            print(f"      Test duration: {stats['test_duration_hours']:.1f} hours")
            print(f"      Data points: {stats['data_points']:,}")

        return {
            'comparison_stats': comparison_stats,
            'combined_data_points': len(combined_df),
            'manufacturers': list(combined_df['manufacturer'].unique())
        }

    def run_comprehensive_analysis(self, args) -> None:
        """
        Run comprehensive analysis based on command line arguments

        Args:
            args: Parsed command line arguments
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[START] Starting Comprehensive Battery Analysis - {timestamp}")
        print("=" * 60)

        # Scan datasets
        self.scan_available_datasets()

        if not self.loader.datasets:
            print("[ERROR] No datasets available for analysis!")
            return

        # Parse channels
        channels = None
        if args.channels:
            try:
                channels = [int(c.strip()) for c in args.channels.split(',')]
                print(f"[CHANNEL] Channel filter: {channels}")
            except ValueError:
                print(f"[WARN] Invalid channel format: {args.channels}")

        # Manufacturer comparison
        if args.compare:
            print("\n" + "="*50)
            print("[MODE] MANUFACTURER COMPARISON")
            print("="*50)
            comparison_results = self.compare_manufacturers(channels)
            return

        # Single dataset analysis
        if args.dataset:
            if args.dataset not in self.loader.datasets:
                print(f"[ERROR] Dataset '{args.dataset}' not found!")
                print(f"Available datasets: {self.loader.datasets}")
                return

            print("\n" + "="*50)
            print("[MODE] SINGLE DATASET ANALYSIS")
            print("="*50)
            self.analyze_single_dataset(
                args.dataset,
                channels,
                args.report_format,
                args.interactive
            )
            return

        # Analyze all datasets
        print("\n" + "="*50)
        print("[MODE] COMPREHENSIVE ANALYSIS - ALL DATASETS")
        print("="*50)

        all_results = {}
        for dataset in self.loader.datasets:
            try:
                print(f"\n{'='*20} {dataset} {'='*20}")
                results = self.analyze_single_dataset(
                    dataset,
                    channels,
                    args.report_format,
                    args.interactive
                )
                all_results[dataset] = results
            except Exception as e:
                print(f"[ERROR] Error analyzing {dataset}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        print(f"\n[COMPLETE] Analysis completed!")
        print(f"   Datasets analyzed: {len(all_results)}")
        print(f"   Results saved to: {self.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Battery Data Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )

    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to analyze')
    parser.add_argument('--channels', type=str,
                       help='Comma-separated channel numbers (e.g., 30,31)')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Output directory for reports and plots')
    parser.add_argument('--report-format', type=str, choices=['html', 'json', 'both'],
                       default='html', help='Report format')
    parser.add_argument('--interactive', action='store_true',
                       help='Generate interactive dashboard')
    parser.add_argument('--manufacturer', type=str, choices=['ATL', 'Toyo', 'PNE'],
                       help='Filter by manufacturer')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple manufacturers')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    try:
        # Initialize analysis system
        analyzer = ComprehensiveBatteryAnalysis(
            base_path="Rawdata",
            output_dir=args.output_dir
        )

        # Run analysis
        analyzer.run_comprehensive_analysis(args)

    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()