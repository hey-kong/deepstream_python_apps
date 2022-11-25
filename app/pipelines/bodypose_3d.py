import sys

import gi
import pyds
from gi.repository import Gst

from app.pipeline import Pipeline

sys.path.append('../')
gi.require_version('Gst', '1.0')

PAD_DIM = 128
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
muxer_output_width_pad = PAD_DIM * 2 + MUXER_OUTPUT_WIDTH
muxer_output_height_pad = PAD_DIM * 2 + MUXER_OUTPUT_HEIGHT


class Bodypose3DPipeline(Pipeline):
    def __init__(self, *args, sgie_config_path: str, **kwargs):
        self.sgie_config_path = sgie_config_path
        self.sgie = None

        super().__init__(*args, **kwargs)

    def _create_elements(self):
        super()._create_elements()
        element_names = [elm.name for elm in self.elements]
        tracker_idx = element_names.index(self.tracker.name)

        self.sgie = self._create_element("nvinfer", "secondary-inference", "SGIE", add=False)
        self.sgie.set_property('config-file-path', self.sgie_config_path)
        self._add_element(self.sgie, tracker_idx + 1)

    def pgie_src_pad_buffer_probe(self, _, __, ll_obj_meta):
        for l_obj_meta in ll_obj_meta:
            for obj_meta in l_obj_meta:
                sizex = obj_meta.rect_params.width * 0.5
                sizey = obj_meta.rect_params.height * 0.5
                centrx = obj_meta.rect_params.left + sizex
                centry = obj_meta.rect_params.top + sizey
                sizex = sizex * 1.25
                sizey = sizey * 1.25
                if sizex < sizey:
                    sizex = sizey
                else:
                    sizey = sizex

                obj_meta.rect_params.width = round(2 * sizex)
                obj_meta.rect_params.height = round(2 * sizey)
                obj_meta.rect_params.left = round(centrx - obj_meta.rect_params.width / 2)
                obj_meta.rect_params.top = round(centry - obj_meta.rect_params.height / 2)

                sizex = obj_meta.rect_params.width * 0.5
                sizey = obj_meta.rect_params.height * 0.5
                centrx = obj_meta.rect_params.left + sizex
                centry = obj_meta.rect_params.top + sizey
                # Make sure box has same aspect ratio as 3D Body Pose model's input dimensions
                # (e.g 192x256 -> 0.75 aspect ratio) by enlarging in the appropriate dimension.
                xScale = 192.0 / sizex
                yScale = 256.0 / sizey
                if xScale < yScale:
                    sizey = 256.0 / xScale
                else:
                    sizex = 192.0 / yScale

                obj_meta.rect_params.width = round(2 * sizex)
                obj_meta.rect_params.height = round(2 * sizey)
                obj_meta.rect_params.left = round(centrx - obj_meta.rect_params.width / 2)
                obj_meta.rect_params.top = round(centry - obj_meta.rect_params.height / 2)
                if obj_meta.rect_params.left < 0.0:
                    obj_meta.rect_params.left = 0.0
                if obj_meta.rect_params.top < 0.0:
                    obj_meta.rect_params.top = 0.0
                if obj_meta.rect_params.left + obj_meta.rect_params.width > muxer_output_width_pad - 1:
                    obj_meta.rect_params.width = muxer_output_width_pad - 1 - obj_meta.rect_params.left
                if obj_meta.rect_params.top + obj_meta.rect_params.height > muxer_output_height_pad - 1:
                    obj_meta.rect_params.height = muxer_output_height_pad - 1 - obj_meta.rect_params.top

    def _save_features(self, _, __, ll_obj_meta):
        for l_obj_meta in ll_obj_meta:
            for obj_meta in l_obj_meta:
                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    except StopIteration:
                        break

                    if user_meta.base_meta.meta_type \
                            != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        continue

                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

    def _add_probes(self):
        pgie_src_pad = self._get_static_pad(self.pgie, "src")
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self.pgie_src_pad_buffer_probe))
        sgie_src_pad = self._get_static_pad(self.sgie, "src")
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self._save_features))
