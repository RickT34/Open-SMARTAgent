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
        )
        tasks += exper.gen_tasks(
            envs_smart,
            runner_inference_smart,
        )
    else:
        ##Judge
        # 0,1,2,3 for a llm judge by big vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        envs = envs_base + envs_smart
        tasks = exper.gen_tasks(
            envs,
            runner_inference_eval,
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
    
    workflow = runners.Workflow(
        "base_no_tool",
        [
            runners.EmvDump(),
            *runners.prefab_vllmeval(),
            *runners.prefab_llmeval(ModelQwen35_32B_AWQ, PROMPT_SMART_JUDGE, model_output_field="model_output"),
            runners.LineCheck(line_check_judge, "llmeval/result.json", "llmeval/result_cleared.json"),
            runners.BinSum("llmeval/result_cleared.json", "llmeval/result_summary.json"),
        ],
        True
    )
    workflow.info()
    tasks = exper.gen_tasks(
        envs,
        workflow
    )
    # exper.run_tasks(tasks)
    
# exp_inference_tool()
exp_inference_no_tool()
