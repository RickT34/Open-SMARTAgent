from envdata import *
from envdata_smart import *
from lazyexp import exper, exenv, evaluator, vllmeval
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


def exp_inference_tool():
    envs = exenv.genEnvs(
            MODELS_BASE + MODELS_SMART, 
            Datasets_Domain_Tool+Datasets_OOD_Tool,
            [AlgoNULL],
            "base"
        )
    tasks = exper.gen_tasks(
        envs,
        runner_inference_tool_prompt,
        "tool_prompt"
    )
    exper.run_tasks(tasks, ui=False)
    ##Judge
    tasks = exper.gen_tasks(
        envs,
        runner_inference_eval,
        "tool_prompt_eval"
    )
    exper.run_tasks(tasks, ui=False)
    
def exp_inference_no_tool():
    envs = exenv.genEnvs(
            MODELS_BASE + MODELS_SMART, 
            Datasets_Domain_Smart+Datasets_OOD_Smart,
            [AlgoNULL],
            "base"
        )
    tasks = exper.gen_tasks(
        envs,
        runner_inference_smart,
        "no_tool"
    )
    exper.run_tasks(tasks, ui=False)
    ##Judge
    llm_judge = evaluator.LLMEvaluator(
        ModelQwen35_27B,
        PROMPT_SMART_JUDGE,
        vllmeval.main,
        model_output_field="model_output"
    )
    tasks_judge = exper.gen_tasks(
        envs,
        llm_judge.evaluate,
        "no_tool_eval"
    )
    exper.run_tasks(tasks_judge, ui=False)
    
    