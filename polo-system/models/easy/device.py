import subprocess
import sys

try:
    import torch
except Exception as e:
    print("[torch] import error:", repr(e))
    torch = None


def check_torch_cuda():
    if torch is None:
        return
    print("[torch] version:", torch.__version__)
    print("[torch] cuda version:", getattr(torch.version, "cuda", None))
    print("[torch] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("[torch] device count:", torch.cuda.device_count())
            print("[torch] current device:", torch.cuda.current_device())
            print("[torch] device name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("[torch] cuda query error:", repr(e))


def run_nvidia_smi():
    # Windows에서 NVIDIA 드라이버/CLI가 있으면 간단 정보 출력
    for cmd in ("nvidia-smi", "nvidia-smi.exe"):
        try:
            out = subprocess.check_output([cmd], stderr=subprocess.STDOUT, text=True, timeout=5)
            print("\n[nvidia-smi]\n" + out)
            return
        except Exception as e:
            last_err = e
    print("[nvidia-smi] not available:", repr(last_err))


def main():
    print("Python:", sys.executable)
    check_torch_cuda()
    run_nvidia_smi()


if __name__ == "__main__":
    main()


