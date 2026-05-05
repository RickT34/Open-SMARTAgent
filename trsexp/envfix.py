from envdata_smart import *
envs = (
    exenv.genEnvs(
        MODELS_BASE + MODELS_SMART,
        Datasets_Domain_Smart
        + Datasets_Domain_Tool
        + Datasets_OOD_Smart
        + Datasets_OOD_Tool,
        [AlgoNULL],
        "base_tool",
    )
    + exenv.genEnvs(
        MODELS_BASE + MODELS_SMART,
        Datasets_Domain_Smart
        + Datasets_Domain_Tool
        + Datasets_OOD_Smart
        + Datasets_OOD_Tool,
        [AlgoNULL],
        "smart_tool",
    )
    + exenv.genEnvs(
        MODELS_BASE + MODELS_SMART,
        Datasets_Domain_Smart
        + Datasets_Domain_Tool
        + Datasets_OOD_Smart
        + Datasets_OOD_Tool,
        [AlgoNULL],
        "base_tool_less",
    )
)

exenv.move_envs(envs, "outputs", "outputs.smart_judge.bak", sub_paths=[Path("smart_judged.json")])
