import time
from clip_onnx import clip_onnx
import clip
from PIL import Image
import numpy as np
import os
# onnx cannot work with cuda
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

# batch first
image = preprocess(Image.open("CLIP.png")).unsqueeze(
    0).cpu()  # [1, 3, 224, 224]
image_onnx = image.detach().cpu().numpy().astype(np.float32)

# batch first
text = clip.tokenize(["a diagram", "a dog", "a cat"]).cpu()  # [3, 77]
text_onnx = text.detach().cpu().numpy().astype(np.int32)


visual_path = "visual.onnx"
textual_path = "textual.onnx"
if os.path.exists(visual_path) and os.path.exists(textual_path):
    onnx_model = clip_onnx(None)
    onnx_model.load_onnx(visual_path=visual_path,
                         textual_path=textual_path,
                         logit_scale=100.0000)
else:
    onnx_model = clip_onnx(model, visual_path=visual_path,
                       textual_path=textual_path)
    onnx_model.convert2onnx(image, text, verbose=True)
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_model.start_sessions(providers=["CUDAExecutionProvider"])  # cpu mode
t1 = time.time()
for i in range(100):
    image_features = onnx_model.encode_image(image_onnx)
    text_features = onnx_model.encode_text(text_onnx)
    logits_per_image, logits_per_text = onnx_model(image_onnx, text_onnx)
t2 = time.time()
print("time for inference: {}s".format((t2-t1)/100))
probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421067 0.00299571]]
