from lazyexp.exenv import *
from typing import TYPE_CHECKING
from subprocess import Popen
import sys

DIR_DATA = "/share/trsdata/trsdata"

ModelLLaMA31_8B = ModelEnv(
    "llama3.1-8b",
    f"{DIR_DATA}/models/Llama-3.1-8B-Instruct/",
    32,
)


ModelMistral_7B = ModelEnv(
    "mistral-7b", f"{DIR_DATA}/models/Mistral-7B-Instruct-v0.3", 32
)


ModelQwen35_27B = ModelEnv("qwen3.5-27b", f"{DIR_DATA}/models/Qwen3.5-27B", 32)


AlgoNULL = AlgoEnv("null", {})

PYTHON = "/share/miniconda3/envs/vllm/bin/python"


def popen_inherit_stdio(*args, **kwargs):
    kwargs.setdefault("stdin", sys.stdin)
    kwargs.setdefault("stdout", sys.stdout)
    kwargs.setdefault("stderr", sys.stderr)
    return Popen(*args, **kwargs)


def runner_vllmeval(env: ExpEnv):
    if os.path.exists(env.get_output_path()):
        print(
            f"Output path {env.get_output_path()} already exists. Skipping inference."
        )
        return
    p = popen_inherit_stdio(
        [PYTHON, "-m", "lazyexp.vllmeval", "--env", env.get_output_path("env.json")],
    )
    p.wait()
