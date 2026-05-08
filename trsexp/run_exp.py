from envdata import *
from envdata_smart import *
from lazyexp import exper, exenv, runners, datapro
import os
import json

envs_base:list[ExpEnv] = exenv.genEnvs(
        MODELS_BASE + MODELS_SMART, 
        Datasets_Domain_Tool+Datasets_OOD_Tool,
        [AlgoNULL],
        "base_tool"
    )
envs_base_less: list[ExpEnv] = exenv.genEnvs(
    MODELS_BASE + MODELS_SMART,
    Datasets_Domain_Tool + Datasets_OOD_Tool,
    [AlgoNULL],
    "base_tool_less",
)
envs_smart:list[ExpEnv] = exenv.genEnvs(
        MODELS_SMART, 
        Datasets_Domain_Smart+Datasets_OOD_Smart,
        [AlgoNULL],
        "smart_tool"
    )
envs_no_tool:list[ExpEnv] = exenv.genEnvs(
        MODELS_BASE + MODELS_SMART, 
        Datasets_Domain_Tool+Datasets_OOD_Tool,
        [AlgoNULL],
        "base_no_tool"
    )

def exp_inference_tool(judge:bool=True):
    if not judge:
        # 0 for a llm user by small vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        tasks = exper.gen_tasks(
            envs_base,
            runner_inference_tool_prompt,
        )
        tasks += exper.gen_tasks(
            envs_base_less,
            runner_inference_tool_prompt_less
        )
        tasks += exper.gen_tasks(
            envs_smart,
            runner_inference_smart,
        )
    else:
        ##Judge
        # 0,1,2,3 for a llm judge by big vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        envs = envs_base + envs_smart + envs_base_less
        workflow = runners.Workflow(
            "inference_tool_eval",
            [
                runner_inference_eval,
                SmartJudgeFormater(),
                runners.SummeryTable("smart_judge_formatted.json")
            ],
            skip_success=False,
            skip_exclude_paths=["smart_judge_formatted.json", "summery.json"],
            promised_paths=["result.json"]
        )
        workflow.info()
        # workflow.clear_outputs(envs, dry_dun=False)
        # return
        tasks = exper.gen_tasks(
            envs,
            workflow,
        )
    exper.run_tasks(tasks, ui=False)

def exp_inference_no_tool():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    workflow = runners.Workflow(
        "base_no_tool",
        [
            runners.EmvDump(),
            *runners.prefab_vllmeval(),
            *runners.prefab_llmeval(
                ModelQwen35_32B_AWQ,
                PROMPT_SMART_JUDGE,
                model_output_field="model_output",
            ),
            runners.LineCheck(
                line_check_judge, "llmeval/result.json", "llmeval/result_cleared.json"
            ),
            runners.SummeryTable("llmeval/result_cleared.json"),
        ],
        True,
        skip_exclude_paths=["llmeval/result_cleared.json", "summery.json"],
    )
    workflow.info()
    tasks = exper.gen_tasks(
        envs_no_tool,
        workflow
    )
    exper.run_tasks(tasks, ui=False)

if __name__ == "__main__":
    # exp_inference_tool(False)
    # exp_inference_tool(True)
    exp_inference_no_tool()
    # summery()
    # cross_ansly()
    # cross_ansly_tool_vs_smart()
