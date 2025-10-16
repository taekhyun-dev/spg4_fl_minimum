# minimum_test/environment_minimum.py
import asyncio
from skyfield.api import Topos
from typing import List, Dict, Tuple
from ml.model import PyTorchModel, create_mobilenet
from minimum_test.satellite_minimum import Satellite

# ----- CLASS DEFINITION ----- #
class IoT:
    def __init__ (self, name, sim_logger, latitude, longitude, elevation, initial_model: PyTorchModel, test_loader):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.global_model = initial_model
        self.test_loader = test_loader
        self.logger.info(f"IoT 클러스터 '{self.name}' 생성 완료.")
    
    async def receive_global_model(self, model: PyTorchModel):
        """위성으로부터 글로벌 모델을 수신"""
        if model.version > self.global_model.version:
            self.logger.info(f"  📡  IoT {self.name}: 새로운 글로벌 모델 수신 (v{model.version}).")
            self.global_model = model

class GroundStation:
    def __init__ (self, name, latitude, longitude, elevation, sim_logger, threshold_deg, staleness_threshold,
                  initial_model: PyTorchModel, test_loader, perf_logger):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.threshold_deg = threshold_deg
        self._comm_status: Dict[int, bool] = {}
        self.staleness_threshold = staleness_threshold
        self.global_model = initial_model
        self.test_loader = test_loader
        self.perf_logger = perf_logger
        self.logger.info(f"지상국 '{self.name}' 생성 완료. 글로벌 모델 버전: {self.global_model.version}")
        self.logger.info(f"  - Aggregation 정책: 버전 허용치 {self.staleness_threshold}")

    async def run(self, clock: 'SimulationClock', satellites: Dict[int, 'Satellite']):
        self.logger.info(f"지상국 '{self.name}' 운영 시작.")
        while True:
            current_ts = clock.get_time_ts()
            for sat_id, sat in satellites.items():
                elevation = (sat.satellite_obj - self.topos).at(current_ts).altaz()[0].degrees
                prev_visible = self._comm_status.get(sat_id, False)

    async def send_model_to_satellite(self, satellite: 'Satellite'):
        if self.global_model.version > satellite.local_model.version:
            self.logger.info(f"  📤 {self.name} -> SAT {satellite.sat_id}: 글로벌 모델 전송 (버전 {self.global_model.version})")
            await satellite.receive_global_model(self.global_model)

    async def receive_model_from_satellite(self, satellite: 'Satellite'):
        local_model = await satellite.send_local_model()
        if local_model and self.global_model.version - local_model.version <= self.staleness_threshold:
            self.logger.info(f"  📥 {self.name} <- SAT {satellite.sat_id}: 로컬 모델 수신 완료 (버전 {local_model.version}, 학습자: {local_model.trained_by})")

def create_minumum_simulation_environment(
    clock: 'SimulationClock', 
    eval_infra: dict, 
    all_sats_skyfield: Dict[int, EarthSatellite]
) -> Tuple[Dict[int, Satellite], List[GroundStation]]: