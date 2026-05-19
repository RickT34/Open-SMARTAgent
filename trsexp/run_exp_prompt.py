from envdata_smart import *
from envdata import *
import argparse
from lazyexp import datapro
import eval

MODELS = [ModelQwen25_7B, ModelLLaMA31_8B, ModelMistral_7B, ModelQwen35_27B]
DATASETS = [get_dataset_by_name("math_tool_prompt")]
ALGOS = [
    make_prompt_algo(path.as_posix(), mode="replace")
    for path in Path("prompts/tool_prompt_ablation/controlled_math").glob("tools_*.txt")
]
envs = exenv.genEnvs(
    MODELS,
    DATASETS,
    ALGOS,
    "tool_prompt_ratio"
)

def test_tool_example_ratio(judge:bool=False):
    if not judge:
        # 0 for a llm user by small vllm model
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


def get_summery_table(envs:list[ExpEnv]):
    def process_fc(envs:list[ExpEnv]):
        assert len(envs) == 1
        res = json.load(envs[0].get_output_path("summery.json").open())
        res.update(eval.tool_harm_info(envs[0]))
        return res
    return datapro.extable(envs, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis, datapro.ExpAxis.AlgoAxis), process_fc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, choices=["ratio", "summery"], default="ratio")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    if args.run == "summery":
        t = get_summery_table(envs)
        print(t)
    elif args.run == "ratio":
        test_tool_example_ratio(judge=args.judge)