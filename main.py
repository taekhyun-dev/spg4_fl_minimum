import torch.multiprocessing as mp
from typing import Dict
from pathlib import Path
from utils.logging_setup import setup_loggers
from ml.data import get_cifar10_loaders
from datetime import datetime, timezone, timedelta
from simulation.clock import SimulationClock
from utils.skyfield_utils import EarthSatellite
from minimum_test.environment_minimum import create_minumum_simulation_environment
from minimum_test.satellite_minimum import Satellite

def load_constellation(tle_path: str, sim_logger) -> Dict[int, EarthSatellite]:
    """TLE 파일에서 위성군 정보를 불러오는 함수"""
    if not Path(tle_path).exists(): raise FileNotFoundError(f"'{tle_path}' 파일을 찾을 수 없습니다.")
    satellites = {}
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]; i = 0
        while i < len(lines):
            name, line1, line2 = lines[i:i+3]; sat_id = int(name.replace("SAT", ""))
            satellites[sat_id] = EarthSatellite(line1, line2, name)
            i += 3
    sim_logger.info(f"총 {len(satellites)}개의 위성을 TLE 파일에서 불러왔습니다.")
    return satellites

async def main():
    sim_logger, perf_logger = setup_loggers()

    sim_logger.info("Loading CIFAR10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    sim_logger.info("Dataset loaded.")

    start_time = datetime.now(timezone.utc)
    simulation_clock = SimulationClock(
        start_dt=start_time, 
        time_step=timedelta(minutes=10),
        real_interval=1.0,
        sim_logger=sim_logger
    )
    # TLE 데이터 로드
    all_sats_skyfield = load_constellation("constellation.tle", sim_logger)

    # 로드된 데이터를 전달하여 시뮬레이션 환경 구성
    sim_logger.info("시뮬레이션 환경을 구성합니다...")
    satellites, ground_stations = create_simulation_environment(
        simulation_clock, eval_infra, all_sats_skyfield
    )
    

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        tasks = mp.Queue()
        results = mp.Queue()

        num_processes = 4
        processes = []

        main()
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        # 예기치 않은 에러 발생 시 로깅
        sim_logger, _ = setup_loggers()
        sim_logger.error(f"\n시뮬레이션 중 치명적인 에러 발생: {e}", exc_info=True)
    