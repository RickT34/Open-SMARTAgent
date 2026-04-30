from envdata import *
from envdata_smart import *
from lazyexp import exper, exenv, runners
import os



def exp_inference_tool(judge:bool=True):
    envs_base = exenv.genEnvs(
            MODELS_BASE + MODELS_SMART, 
            Datasets_Domain_Tool+Datasets_OOD_Tool,
            [AlgoNULL],
            "base_tool"
        )
    envs_smart = exenv.genEnvs(
            MODELS_SMART, 
            Datasets_Domain_Smart+Datasets_OOD_Smart,
            [AlgoNULL],
            "smart_tool"
        )
    if not judge:
        # 0 for a llm user by small vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        tasks = exper.gen_tasks(
            envs_base,
            runner_inference_tool_prompt,
            "tool_prompt"
        )
        tasks += exper.gen_tasks(
            envs_smart,
            runner_inference_smart,
            "tool_prompt_smart"
        )
    else:
        ##Judge
        # 0,1,2,3 for a llm judge by big vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        envs = envs_base + envs_smart
        tasks = exper.gen_tasks(
            envs,
            runner_inference_eval,
            "tool_prompt_eval"
        )
    exper.run_tasks(tasks, ui=False)
    
def exp_inference_no_tool():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    envs= exenv.genEnvs(
            MODELS_BASE + MODELS_SMART, 
            Datasets_Domain_Tool+Datasets_OOD_Tool,
            [AlgoNULL],
            "base_no_tool"
        )
    tasks = exper.gen_tasks(
        envs,
        vllm_runner,
        "no_tool"
    )
    exper.run_tasks(tasks)
    #Judge
    llm_judge = runners.LLMEvaluator(
        ModelQwen35_32B_AWQ,
        PROMPT_SMART_JUDGE,
        vllm_runner,
        model_output_field="model_output"
    )
    tasks_judge = exper.gen_tasks(
        envs,
        llm_judge.runner,
        "no_tool_eval"
    )
    exper.run_tasks(tasks_judge)
    
exp_inference_tool()
# exp_inference_no_tool()
