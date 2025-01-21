path_prompt_iterate = "prompt/迭代群体画像.txt"
path_prompt_classify = "prompt/访谈和画像的匹配度.txt"
path_prompt_cut_interview = "prompt/切割访谈记录.txt"
path_prompt_extract_key_points = "prompt/提取要点.txt"
path_prompt_find_causes = "prompt/找出原因.txt"
path_prompt_find_effects = "prompt/找出结果.txt"
path_prompt_analyze_interview_segments = "prompt/分析访谈片段.txt"


class Prompts:
    def __init__(self):
        with open(path_prompt_iterate, "r", encoding="utf-8") as file:
            self.prompt_iterate = file.read()
        with open(path_prompt_classify, "r", encoding="utf-8") as file:
            self.prompt_classify = file.read()
        with open(path_prompt_cut_interview, "r", encoding="utf-8") as file:
            self.prompt_cut_interview = file.read()
        with open(path_prompt_extract_key_points, "r", encoding="utf-8") as file:
            self.prompt_extract_key_points = file.read()
        with open(path_prompt_find_causes, "r", encoding="utf-8") as file:
            self.prompt_find_causes = file.read()
        with open(path_prompt_find_effects, "r", encoding="utf-8") as file:
            self.prompt_find_effects = file.read()
        with open(path_prompt_analyze_interview_segments, "r", encoding="utf-8") as file:
            self.prompt_analyze_interview_segments = file.read()
