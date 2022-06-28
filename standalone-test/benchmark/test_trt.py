from tkinter import image_names
from clip_trt_benchmark import CLIPTensorRTModel
import torch
import os
import clip
import logging
import time
import numpy as np
import tqdm
from PIL import Image
import argparse
os.environ['MODEL'] = 'ViT-B/32'


parser = argparse.ArgumentParser()
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--trt_precision', type=str, choices=['fp32','tf32','fp16'],help='batch size')
parser.add_argument('--torch_fp16', type=bool, default= False,help='precision of torch')
args = parser.parse_args()

print("batch size: ",args.bs)
print("tensorrt precision: ",args.trt_precision)
print("torch fp16: ",args.torch_fp16)

def check(a, b, weak=False, epsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check: max abs diff", diff0,"; median relative diff", diff1)
    return res, diff0, diff1

# image_inputs = torch.ones((1, 3, 224, 224)).contiguous().cuda()
# model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
# batch first


nRound = 30

print("================testing model in torch================")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(
    os.getenv('MODEL'), device, jit=False, use_fp16=args.torch_fp16)
image_inputs = preprocess(Image.open("CLIP.png")).unsqueeze(
    0).to(device)  # [1, 3, 224, 224]
# logging.info('using model: {}'.format(os.getenv('MODEL')))
# print(image_inputs.dtype)
print("using model: {}, precision {}".format(
    os.getenv('MODEL'), model.visual.conv1.weight.dtype))
image_inputs = image_inputs.repeat(args.bs, 1, 1, 1)
# print(image_inputs)
# raise
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
timings = np.zeros((nRound, 1))

with torch.no_grad():
    # torch.cuda.synchronize()
    # t0 = time.time_ns()
    
    for i in tqdm.tqdm(range(nRound)):
        start.record()
        model.encode_image(image_inputs)  # .cpu().detach().numpy()
    # t1 = time.time_ns()
        end.record()    
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        timings[i] = curr_time 
    image_time_pytorch = timings.sum()/nRound
    image_features_torch = model.encode_image(image_inputs).cpu().detach().numpy()

torch.cuda.empty_cache()

time.sleep(5)

print("================testing model in tensorrt================")
trt_model = CLIPTensorRTModel("ViT-B/32",precision=args.trt_precision)
trt_model.start_engines(use_fp16=False)
# print(type(image_features_torch))


# torch.cuda.synchronize()
# t0 = time.time_ns()

image_features_trt,image_time_trt= trt_model.encode_image(image_inputs)  # .cpu().detach().numpy()

# t1 = time.time_ns()

print('Torch time:', image_time_pytorch)
print('TensorRT time:', image_time_trt)
print('Speedup:', image_time_pytorch / image_time_trt)

# image_features_trt,_ = trt_model.encode_image(
#     image_inputs)
image_features_trt = image_features_trt[0].cpu().contiguous().numpy()
check(image_features_trt, image_features_torch)

print("===============================================================")
print("\n\n\n")
