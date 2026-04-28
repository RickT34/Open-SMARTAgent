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


ModelQwen35_27B = ModelEnv(
    "qwen3.5-27b", f"{DIR_DATA}/models/Qwen3.5-27B", 32
)


AlgoNULL = AlgoEnv("null", {})


PROMPT_SMART_BASE = "You are an advanced assistant designed to solve tasks autonomously using your knowledge and reasoning. Clearly articulate your thought process and reasoning steps before presenting the final response to ensure transparency and accuracy."

DatasetSMARTmath = DatasetEnv(
    f"{DIR_DATA}/datasets/smart_math",
    prompt_template=PROMPT_SMART_BASE + "\n\n{tests}",
    filetype="hf_disk",
)

PROMPT_SMART_JUDGE = """You are a helpful assistant to jusge whether the model's final response (might be word, phrase or sentence) and the given correct answer is same in value.
- If their intrinsic numerical value of the answer is not equal, please mark it as wrong.
- If they are just expressed in different format or wording or unit, but have the same main value, please mark it as correct.

Example:
- Model response: The final answer should be 4.123
- Ground truth: \\sqrt{17}
- Judgment: correct

- Model response: 0.2687
- Ground truth: \\frac{pi}{9}
- Judgment: wrong

- Model response: 40%
- Ground truth: 40
- Judgment: correct

- Model response: Therefore, the speed of the car is 25 miles per hour
- Ground truth: 25
- Judgment: correct

- Model response: The temperature of the metal ay noon will be 10938.893 T
- Ground truth: 1.983e4
- Judgment: wrong

- Model response: $1000.00
- Ground truth: 1000
- Judgment: correct

- Model response: {output}
- Ground truth: {answers}
- Judgment: """

