from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import pandas as pd
import sklearn
import torch
import transformers

from common import RESULTS_DIR, dump_json, ensure_dirs


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip() or completed.stderr.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    ensure_dirs()
    cpu_name = command_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ]
    )
    memory_bytes = int(
        command_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
    )
    gpu_csv = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_name, gpu_memory_mb, driver_version = [value.strip() for value in gpu_csv.split(",", 2)]
    payload = {
        "collected_at": datetime.now().astimezone().isoformat(),
        "operating_system": platform.platform(),
        "cpu": cpu_name,
        "logical_processors": os.cpu_count(),
        "system_memory_gb": memory_bytes / 1024**3,
        "gpu": {
            "name": gpu_name,
            "memory_mb": float(gpu_memory_mb),
            "driver_version": driver_version,
        },
        "roberta_runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "ollama_version": command_output([r"D:\Ollama\ollama.exe", "--version"]),
        "qwen_model": {
            "ollama_tag": "qwen3:8b",
            "parameters": "8.2B",
            "quantization": "Q4_K_M",
            "configured_num_ctx": 1024,
            "generation_num_predict": 96,
            "classification_num_predict": 24,
        },
        "latex": command_output(["xelatex", "--version"]).splitlines()[0],
        "official_repository": {
            "url": "https://github.com/zhuyiYZU/BC4RII",
            "audited_commit": "35076870d110d49a1a7dbe2335d3fdb28ba3b6a1",
        },
    }
    dump_json(RESULTS_DIR / "environment.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
