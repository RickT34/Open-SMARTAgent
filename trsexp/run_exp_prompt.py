from envdata_smart import *
from envdata import *

MODELS = [ModelQwen25_7B, ModelLLaMA31_8B, ModelMistral_7B]
DATASETS = [get_dataset_by_name("gsm")]
ALGOS = [
    make_prompt_algo(path.as_posix(), mode="replace")
    for path in Path("Open-SMARTAgent/prompts/tool_prompt_ablation/controlled_gsm").glob("tools_*.txt")
]

def test_tool_example_ratio(judge:bool=False):
    envs = exenv.genEnvs(
        MODELS,
        DATASETS,
        ALGOS,
        "tool_prompt_ratio"
    )
    if not judge:
        # 0 for a llm user by small vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        tasks = exper.gen_tasks(
            envs,
            runner_inference_tool_prompt,
        )
    else:
        ##Judge
        # 0,1,2,3 for a llm judge by big vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    