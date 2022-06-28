conda activate torch
rm -rf data/output/*

python standalone-test/benchmark/test_trt.py --bs 64 --trt fp32  
python standalone-test/benchmark/test_trt.py --bs 64 --trt tf32 
python standalone-test/benchmark/test_trt.py --bs 64 --trt fp16 
python standalone-test/benchmark/test_trt.py --bs 64 --trt fp16 --torch True 

python standalone-test/benchmark/test_trt.py --bs 768 --trt fp32  
python standalone-test/benchmark/test_trt.py --bs 768 --trt tf32 
python standalone-test/benchmark/test_trt.py --bs 768 --trt fp16 
python standalone-test/benchmark/test_trt.py --bs 768 --trt fp16 --torch True

python standalone-test/benchmark/test_trt.py --bs 1024 --trt fp32  
python standalone-test/benchmark/test_trt.py --bs 1024 --trt tf32 
python standalone-test/benchmark/test_trt.py --bs 1024 --trt fp16 
python standalone-test/benchmark/test_trt.py --bs 1024 --trt fp16 --torch True


echo "+++++++++++testing tensorrt inference in clifs framework+++++++++++"
python standalone-test/test.py #插入clifs后的结果，zero-shot输出在data/output文件夹下