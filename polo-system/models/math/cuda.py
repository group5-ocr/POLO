import torch, importlib.metadata as im
print("torch file:", torch.__file__)
print("torch version:", torch.__version__)
print("pip sees torch:", im.version("torch"))
print("CUDA available:", torch.cuda.is_available())
