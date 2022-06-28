import os

try:
    import tensorrt as trt
    from tensorrt.tensorrt import Logger, Runtime

    from trt_utils import load_engine, build_engine, save_engine
except ImportError:
    raise ImportError(
        "It seems that TensorRT is not yet installed. "
        "It is required when you declare TensorRT backend."
        "Please find installation instruction on "
        "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
    )

MODEL_SIZE = {
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B/32': 224,
    'ViT-B/16': 224,
    'ViT-L/14': 224,
    'ViT-L/14@336px': 336,
}

class CLIPTensorRTModel:
    def __init__(
        self,
        name: str = None,
    ):
        if name in MODEL_SIZE:
            self._textual_path = "textual.plan"
            self._visual_path = "visual.plan"
        else:
            raise RuntimeError(
                f'Model {name} not found or not supports Nvidia TensorRT backend; available models = {list(MODEL_SIZE.keys())}'
            )
        self._name = name

    def  start_engines(self,use_fp16=False):
        import torch

        trt_logger: Logger = trt.Logger(trt.Logger.VERBOSE)
        runtime: Runtime = trt.Runtime(trt_logger)
        compute_capacity = torch.cuda.get_device_capability()

        # if compute_capacity != (8, 6):
            # print(
            #     f'The engine plan file is generated on an incompatible device, expecting compute {compute_capacity} '
            #     'got compute 8.6, will rebuild the TensorRT engine.'
            # )
        from clip_onnx import CLIPOnnxModel

        # onnx_model = CLIPOnnxModel(self._name)
        if not os.path.isfile(self._visual_path):
            print("Building Visual TRT model.")
            visual_engine = build_engine(
                runtime=runtime,
                onnx_file_path='./visual.onnx',
                logger=trt_logger,
                min_shape=(1, 3, MODEL_SIZE[self._name], MODEL_SIZE[self._name]),
                optimal_shape=(
                    768,
                    3,
                    MODEL_SIZE[self._name],
                    MODEL_SIZE[self._name],
                ),
                max_shape=(
                    1024,
                    3,
                    MODEL_SIZE[self._name],
                    MODEL_SIZE[self._name],
                ),
                workspace_size=(1 << 30 ) * 23, #23G
                fp16=use_fp16,
                int8=False,
                tf32=False,
            )

            save_engine(visual_engine, self._visual_path)

        if not os.path.isfile(self._textual_path):
            print("Building Tesxtual TRT model.")
            text_engine = build_engine(
                runtime=runtime,
                onnx_file_path='./textual.onnx',
                logger=trt_logger,
                min_shape=(1, 77),
                optimal_shape=(768, 77),
                max_shape=(1024, 77),
                workspace_size=(1 << 30 ) * 23, #23G
                fp16=use_fp16,
                int8=False,
                tf32=False,
            )
            save_engine(text_engine, self._textual_path)

        self._textual_engine = load_engine(runtime, self._textual_path, trt_logger)
        self._visual_engine = load_engine(runtime, self._visual_path, trt_logger)
        print("Successfully loading textual and visual tensorrt models")

    def encode_image(self, onnx_image):
        (visual_output ) = self._visual_engine({'input': onnx_image})

        return visual_output

    def encode_text(self, onnx_text):
        (textual_output) = self._textual_engine({'input': onnx_text})

        return textual_output