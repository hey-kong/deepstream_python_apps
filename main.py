import os

from app.core import run_pipeline, run_anonymization_pipeline, run_reid_pipeline, run_bodypose3d_pipeline, run_back2back_pipeline


if __name__ == '__main__':
    sources = int(os.environ.get("sources"))
    rtsp_urls = str(os.environ.get("rtsp_urls"))
    rtsp_ids = str(os.environ.get("rtsp_ids"))
    run_pipeline(sources, rtsp_urls, rtsp_ids)
