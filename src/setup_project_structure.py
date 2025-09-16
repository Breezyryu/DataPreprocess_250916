"""
배터리 데이터 전처리 프로젝트 구조 정리 스크립트
- 소스코드 폴더 분리
- 출력파일 폴더 생성
- 채널별 데이터 분리 관리
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectStructureManager:
    """프로젝트 구조 관리 클래스"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_directory_structure(self):
        """디렉토리 구조 생성"""
        logger.info("[폴더] 디렉토리 구조 생성 중...")

        # 주요 디렉토리 정의
        directories = {
            'src': self.base_dir / 'src',
            'outputs': self.base_dir / 'outputs',
            'outputs_current': self.base_dir / 'outputs' / f'run_{self.timestamp}',
            'channels': self.base_dir / 'outputs' / f'run_{self.timestamp}' / 'channels',
            'plots': self.base_dir / 'outputs' / f'run_{self.timestamp}' / 'plots',
            'logs': self.base_dir / 'outputs' / f'run_{self.timestamp}' / 'logs',
            'processed': self.base_dir / 'outputs' / f'run_{self.timestamp}' / 'processed',
        }

        # 디렉토리 생성
        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  [OK] {name}: {path}")

        return directories

    def move_source_files(self, directories):
        """소스코드 파일들을 src 폴더로 이동"""
        logger.info("📦 소스코드 파일 정리 중...")

        src_dir = directories['src']

        # 이동할 파일 패턴
        source_files = [
            'improved_battery_processor.py',
            'fix_data_order.py',
            'verify_sorting.py',
            'setup_project_structure.py',
            'simple_battery_demo.ipynb'
        ]

        moved_files = []

        for filename in source_files:
            source_path = self.base_dir / filename
            if source_path.exists():
                target_path = src_dir / filename

                # 파일이 이미 있으면 백업
                if target_path.exists():
                    backup_path = src_dir / f"{filename}.bak_{self.timestamp}"
                    shutil.move(str(target_path), str(backup_path))
                    logger.info(f"  📋 백업: {filename} → {backup_path.name}")

                shutil.move(str(source_path), str(target_path))
                moved_files.append(filename)
                logger.info(f"  ✅ 이동: {filename} → src/")

        return moved_files

    def separate_channels_data(self, directories):
        """채널별 데이터 분리"""
        logger.info("🔄 채널별 데이터 분리 중...")

        channels_dir = directories['channels']
        processed_dir = directories['processed']

        # CSV 파일 찾기
        csv_files = list(self.base_dir.glob("*_processed*.csv"))

        separated_files = {}

        for csv_file in csv_files:
            try:
                logger.info(f"  📊 처리 중: {csv_file.name}")

                # 데이터 로드
                df = pd.read_csv(csv_file, encoding='utf-8-sig')

                if 'Channel' in df.columns:
                    # 채널별로 분리
                    channels = df['Channel'].unique()
                    logger.info(f"    발견된 채널: {list(channels)}")

                    for channel in channels:
                        channel_data = df[df['Channel'] == channel].copy()

                        # 채널별 파일명 생성
                        base_name = csv_file.stem.replace('_processed', '').replace('_sorted', '')
                        channel_filename = f"{base_name}_{channel}.csv"
                        channel_path = channels_dir / channel_filename

                        # 채널 데이터 저장
                        channel_data.to_csv(channel_path, index=False, encoding='utf-8-sig')

                        if channel not in separated_files:
                            separated_files[channel] = []
                        separated_files[channel].append(channel_filename)

                        logger.info(f"    ✅ {channel}: {len(channel_data):,} 행 → {channel_filename}")

                # 원본 파일을 processed 폴더로 이동
                processed_path = processed_dir / csv_file.name
                shutil.move(str(csv_file), str(processed_path))
                logger.info(f"  📦 원본 이동: {csv_file.name} → processed/")

            except Exception as e:
                logger.error(f"  ❌ 파일 처리 실패 {csv_file.name}: {e}")

        return separated_files

    def move_output_files(self, directories):
        """출력 파일들을 적절한 폴더로 이동"""
        logger.info("📁 출력 파일 정리 중...")

        plots_dir = directories['plots']
        logs_dir = directories['logs']

        moved_files = {'plots': [], 'logs': []}

        # 플롯 파일 이동
        plot_files = list(self.base_dir.glob("*.png"))
        for plot_file in plot_files:
            target_path = plots_dir / plot_file.name
            shutil.move(str(plot_file), str(target_path))
            moved_files['plots'].append(plot_file.name)
            logger.info(f"  🖼️ 플롯 이동: {plot_file.name}")

        # 로그 파일 이동
        log_files = list(self.base_dir.glob("*.log"))
        for log_file in log_files:
            target_path = logs_dir / log_file.name
            shutil.move(str(log_file), str(target_path))
            moved_files['logs'].append(log_file.name)
            logger.info(f"  📄 로그 이동: {log_file.name}")

        return moved_files

    def create_summary_report(self, directories, separated_files, moved_files, moved_source_files):
        """프로젝트 정리 요약 보고서 생성"""
        logger.info("📋 요약 보고서 생성 중...")

        report_path = directories['outputs_current'] / 'project_structure_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"배터리 데이터 전처리 프로젝트 구조 정리 보고서\n")
            f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. 디렉토리 구조:\n")
            f.write(f"  📁 src/           - 소스코드 ({len(moved_source_files)}개 파일)\n")
            f.write(f"  📁 outputs/\n")
            f.write(f"    📁 run_{self.timestamp}/\n")
            f.write(f"      📁 channels/   - 채널별 데이터 ({sum(len(files) for files in separated_files.values())}개 파일)\n")
            f.write(f"      📁 plots/      - 시각화 파일 ({len(moved_files.get('plots', []))}개 파일)\n")
            f.write(f"      📁 processed/  - 전처리된 원본 데이터\n")
            f.write(f"      📁 logs/       - 로그 파일 ({len(moved_files.get('logs', []))}개 파일)\n\n")

            f.write("2. 채널별 분리 현황:\n")
            for channel, files in separated_files.items():
                f.write(f"  {channel}:\n")
                for file in files:
                    f.write(f"    - {file}\n")
                f.write(f"    총 {len(files)}개 파일\n\n")

            f.write("3. 이동된 소스 파일:\n")
            for file in moved_source_files:
                f.write(f"  - {file}\n")
            f.write(f"  총 {len(moved_source_files)}개 파일\n\n")

            f.write("4. 사용법:\n")
            f.write("  - 소스코드: src/ 폴더에서 개발\n")
            f.write("  - 출력 확인: outputs/run_*/에서 결과 확인\n")
            f.write("  - 채널별 데이터: channels/ 폴더에서 개별 분석\n")
            f.write("  - 시각화: plots/ 폴더에서 그래프 확인\n")

        logger.info(f"  📋 보고서 생성: {report_path}")
        return report_path

def main():
    """메인 실행 함수"""
    base_dir = "C:/Users/Ryu/Python_project/data/Code/DataPreprocess_250826"

    try:
        # 프로젝트 구조 관리자 생성
        manager = ProjectStructureManager(base_dir)

        print("[시작] 배터리 데이터 프로젝트 구조 정리 시작...")
        print(f"[경로] 작업 디렉토리: {base_dir}")

        # 1. 디렉토리 구조 생성
        directories = manager.create_directory_structure()

        # 2. 소스코드 파일 이동
        moved_source_files = manager.move_source_files(directories)

        # 3. 채널별 데이터 분리
        separated_files = manager.separate_channels_data(directories)

        # 4. 출력 파일 정리
        moved_files = manager.move_output_files(directories)

        # 5. 요약 보고서 생성
        report_path = manager.create_summary_report(
            directories, separated_files, moved_files, moved_source_files
        )

        print("\n[완료] 프로젝트 구조 정리 완료!")
        print(f"[보고서] {report_path}")
        print(f"[출력폴더] {directories['outputs_current']}")

        print("\n[결과] 정리 현황:")
        print(f"  소스파일: {len(moved_source_files)}개 이동")
        print(f"  채널데이터: {sum(len(files) for files in separated_files.values())}개 생성")
        print(f"  플롯파일: {len(moved_files.get('plots', []))}개 이동")
        print(f"  로그파일: {len(moved_files.get('logs', []))}개 이동")

        return True

    except Exception as e:
        logger.error(f"❌ 프로젝트 구조 정리 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[성공] 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n[오류] 작업 중 오류가 발생했습니다.")