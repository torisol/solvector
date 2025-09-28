import torch
print("torch.version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
 print("device:", torch.cuda.get_device_name(0))
 x=torch.randn(2048,2048,device="cuda"); y=x@x; print("CUDA matmul OK; mean=", float(y.mean()))
