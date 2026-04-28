from lazyexp.exenv import *
from typing import TYPE_CHECKING

DIR_DATA = "/share/exps/trsdata"

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



