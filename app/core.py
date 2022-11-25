import os
import logging

from app.pipelines import Pipeline, AnonymizationPipeline, ReIDPipeline, Bodypose3DPipeline, Back2backPipeline
from app.config import CONFIGS_DIR, LOGLEVEL

logging.basicConfig(level=LOGLEVEL)


def run_pipeline(num_sources: int, rtsp_urls: str, rtsp_ids: str):
    pipeline = Pipeline(
        num_sources=num_sources,
        rtsp_urls=rtsp_urls,
        rtsp_ids=rtsp_ids,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/pgie.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt")
    )
    pipeline.run()


def run_anonymization_pipeline(num_sources: int, rtsp_urls: str, rtsp_ids: str):
    pipeline = AnonymizationPipeline(
        num_sources=num_sources,
        rtsp_urls=rtsp_urls,
        rtsp_ids=rtsp_ids,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/yolov4.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        target_classes=[2],
        enable_osd=False
    )
    pipeline.run()


def run_reid_pipeline(num_sources: int, rtsp_urls: str, rtsp_ids: str):
    pipeline = ReIDPipeline(
        num_sources=num_sources,
        rtsp_urls=rtsp_urls,
        rtsp_ids=rtsp_ids,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/pgie.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        sgie_config_path=os.path.join(CONFIGS_DIR, "sgies/osnet.txt"),
        target_classes=[0]
    )
    pipeline.run()


def run_bodypose3d_pipeline(num_sources: int, rtsp_urls: str, rtsp_ids: str):
    pipeline = Bodypose3DPipeline(
        num_sources=num_sources,
        rtsp_urls=rtsp_urls,
        rtsp_ids=rtsp_ids,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/peoplenet.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        sgie_config_path=os.path.join(CONFIGS_DIR, "sgies/bodypose3d.txt")
    )
    pipeline.run()


def run_back2back_pipeline(num_sources: int, rtsp_urls: str, rtsp_ids: str):
    pipeline = Back2backPipeline(
        num_sources=num_sources,
        rtsp_urls=rtsp_urls,
        rtsp_ids=rtsp_ids,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/pgie.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        sgie_config_path=os.path.join(CONFIGS_DIR, "sgies/back_to_back_detectors.txt")
    )
    pipeline.run()
