import sys
from typing import List

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

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_PERSON = 2
SGIE_CLASS_ID_LP = 1
SGIE_CLASS_ID_FACE = 0
PRIMARY_DETECTOR_UID = 1
SECONDARY_DETECTOR_UID = 2


class Back2backPipeline(Pipeline):
    def __init__(self, *args, sgie_config_path: str,
                 target_classes: list = None, **kwargs):
        self.sgie_config_path = sgie_config_path
        self.target_classes = target_classes
        self.sgie = None

        super().__init__(*args, **kwargs)

    def _create_elements(self):
        super()._create_elements()
        element_names = [elm.name for elm in self.elements]
        tracker_idx = element_names.index(self.tracker.name)

        self.sgie = self._create_element("nvinfer", "secondary-inference", "SGIE", add=False)
        self.sgie.set_property('config-file-path', self.sgie_config_path)
        self._add_element(self.sgie, tracker_idx + 1)

    def nvvidconv_sink_pad_buffer_probe(self, batch_meta, l_frame_meta: List, ll_obj_meta: List[List]):
        vehicle_count = 0
        person_count = 0
        face_count = 0
        lp_count = 0

        for frame_meta, l_obj_meta in zip(l_frame_meta, ll_obj_meta):
            for obj_meta in l_obj_meta:
                if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                    if obj_meta.class_id == PGIE_CLASS_ID_VEHICLE:
                        vehicle_count += 1
                    if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                        person_count += 1

                if obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                    if obj_meta.class_id == SGIE_CLASS_ID_FACE:
                        face_count += 1
                    if obj_meta.class_id == SGIE_CLASS_ID_LP:
                        lp_count += 1

                txt_params = obj_meta.text_params

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            txt_params = display_meta.text_params[0]
            txt_params.display_text = \
                "Person = {} Vehicle = {} Face = {} License Plate = {}".format(
                    person_count, vehicle_count, face_count, lp_count)

            # Now set the offsets where the string should appear
            txt_params.x_offset = 10
            txt_params.y_offset = 12

            # Font , font-color and font-size
            txt_params.font_params.font_name = "Serif"
            txt_params.font_params.font_size = 10
            txt_params.font_params.font_color.red = 1.0
            txt_params.font_params.font_color.green = 1.0
            txt_params.font_params.font_color.blue = 1.0
            txt_params.font_params.font_color.alpha = 1.0

            # Text background color
            txt_params.set_bg_clr = 1
            txt_params.text_bg_clr.red = 0.0
            txt_params.text_bg_clr.green = 0.0
            txt_params.text_bg_clr.blue = 0.0
            txt_params.text_bg_clr.alpha = 1.0

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

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
        nvvidconv_sink_pad = self._get_static_pad(self.nvvidconv1, "sink")
        nvvidconv_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self.nvvidconv_sink_pad_buffer_probe))
