import os
from clip_trt_benchmark import MODEL_SIZE
from typing import Union, List

def available_models() -> List[str]:
    '''Returns the names of available CLIP models'''
    return list(MODEL_SIZE.keys())

class CLIPOnnxModel:
    def __init__(self, name: str = None):
        if name in MODEL_SIZE:
            self._textual_path = './visual.onnx'
            self._visual_path = './textual.onnx'
        else:
            raise RuntimeError(
                f'Model {name} not found; available models = {available_models()}'
            )

    def start_sessions(
        self,
        **kwargs,
    ):
        import onnxruntime as ort

        self._visual_session = ort.InferenceSession(self._visual_path, **kwargs)
        self._visual_session.disable_fallback()

        self._textual_session = ort.InferenceSession(self._textual_path, **kwargs)
        self._textual_session.disable_fallback()

    def encode_image(self, onnx_image):
        onnx_input_image = {self._visual_session.get_inputs()[0].name: onnx_image}
        (visual_output,) = self._visual_session.run(None, onnx_input_image)
        return visual_output

    def encode_text(self, onnx_text):
        onnx_input_text = {self._textual_session.get_inputs()[0].name: onnx_text}
        (textual_output,) = self._textual_session.run(None, onnx_input_text)
        return 