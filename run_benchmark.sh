conda activate torch
 
python standalone-test/benchmark/test_trt.py #对比输出计算加速比-默认fp32精度

python standalone-test/test.py #插入clifs后的结果，zero-shot输出在data/output文件夹下