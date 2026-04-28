from lazyexp import exenv, exper
from subprocess import Popen
import os
from lazyexp.exenv import *
import typing
from envdata import *
from collections import defaultdict

TEST_RUN = False


def runner_inference_tool_prompt(env: exenv.ExpEnv):
    if os.path.exists(env.get_output_path()):
        print(f"Output path {env.get_output_path()} already exists. Skipping inference.")
        return
    envr = os.environ.copy()
    envr["WORLD_SIZE"] = "1"
    method = "mistral" if "mistral" in env.model.name else "llama"
    p = popen_inherit_stdio(
        [
            PYTHON, "inference/inference_tool_prompt.py",
            "--model_name_or_path", env.model.path,
            "--data_path" ,env.dataset.path,
            "--max_seq_length", "4096",
            "--save_path", env.get_output_path(),
            "--test_start_id", "0",
            "--max_test_num",  f"{-1 if not TEST_RUN else 3}",
            "--method", method,
        ],
        env=envr,
    )
    p.wait()

def runner_inference_smart(env: exenv.ExpEnv):
    if os.path.exists(env.get_output_path()):
        print(f"Output path {env.get_output_path()} already exists. Skipping inference.")
        return
    envr = os.environ.copy()
    envr["WORLD_SIZE"] = "1"
    p = popen_inherit_stdio(
        [
            PYTHON, "inference/inference_smart.py",
            "--model_name_or_path", env.model.path,
            "--data_path", env.dataset.path,
            "--max_seq_length", "4096",
            "--save_path", env.get_output_path(),
            "--test_start_id", "0",
            "--max_test_num",  f"{-1 if not TEST_RUN else 3}",
        ],
        env=envr,
    )
    p.wait()

def runner_inference_eval(env: exenv.ExpEnv):
    if not os.path.exists(env.get_output_path()):
        print(f"Output path {env.get_output_path()} not exists. Skipping evaluation.")
        return
    p = popen_inherit_stdio(
        [
            PYTHON, f"evaluate/inference_eval_{env.dataset.tags['domain']}.py",
            "--data_path", env.get_output_path(),
            "--save_path", env.get_output_path("smart_judged.json"),
        ],
    )
    p.wait()

Datasets_Domain_Tool = []
Datasets_Domain_Smart = []
Datasets_OOD_Tool = []
Datasets_OOD_Smart = []
PROMPT_SMART_BASE = "You are an advanced assistant designed to solve tasks autonomously using your knowledge and reasoning. Clearly articulate your thought process and reasoning steps before presenting the final response to ensure transparency and accuracy."

for file in os.listdir("data_inference"):
    name = f"Dataset_{file.split('.')[0]}"
    if "time" in name or "mint" in name:
        continue
    if 'intention' in name:
        domain = "intention"
    elif 'math' in name or 'gsm' in name:
        domain = "math"
    else:
        domain = 'time'
    d = DatasetEnv(
        f"data_inference/{file}",
        filetype="json",
        name=name,
        prompt_template=PROMPT_SMART_BASE + "\n\n{input}\n\n",
        tags={"domain": domain}
    )
    if "tool_prompt" in file:
        if "ood" in file:
            Datasets_OOD_Tool.append(d)
        else:
            Datasets_Domain_Tool.append(d)
    else:
        if "ood" in file:
            Datasets_OOD_Smart.append(d)
        else:
            Datasets_Domain_Smart.append(d)


ModelLLaMA31_8B_SMARTAgent = ModelEnv(
    "llama3.1-8b-smartagent",
    f"{DIR_DATA}/models/SMART/SMARTAgent-Llama-3.1-8B/",
    32,
)
ModelMistral_7B_SMARTAgent = ModelEnv(
    "mistral-7b-smartagent",
    f"{DIR_DATA}/models/SMART/SMARTAgent-Mistral-7B-Instruct-v0.3",
    32,
)

MODELS_BASE = [ModelLLaMA31_8B, ModelMistral_7B]
MODELS_SMART = [ModelLLaMA31_8B_SMARTAgent, ModelMistral_7B_SMARTAgent]

PROMPT_SMART_JUDGE = """You are a helpful assistant to jusge whether the model's final response (might be word, phrase or sentence) and the given correct answer is same in value.
- If their intrinsic value of the answer is not equal, please mark it as wrong.
- If they are just expressed in different format or wording or unit, but have the same main value, please mark it as correct.
- Please output only "Correct" or "Wrong" as your judgment result in the final line.

- Model response: 
{model_output}

- Ground truth: 
{output}

- Judgment: """
