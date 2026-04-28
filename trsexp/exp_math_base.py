from envdata import *
from lazyexp import exenv, exper, vllmeval, evaluator
import os

Models = (
    [ModelLLaMA31_8B, ModelMistral_7B]
)



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def exp():
    label= "base"
    testenvs = exenv.genEnvs(Models, [DatasetSMARTmath], [AlgoNULL], label)
    exper.run_exps(
        testenvs,
        vllmeval.main,
        name="smartmath_"+label,
        send_mail=False,
        ui=False
    )
    llm_judge = evaluator.LLMEvaluator(ModelQwen35_27B, PROMPT_SMART_JUDGE)
    

exp()