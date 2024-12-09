from LLM import LLMCurrent
from Prompts import Prompts
from tqdm import tqdm
import warnings

prompts = Prompts()


def query_causes(thing: str, possible_causes: list[str]) -> str:
    possible_causes_with_index = [f"{i + 1} {cause}" for i, cause in enumerate(possible_causes)]
    input = thing + "\n\nPlease find reasons among:\n" + "\n".join(possible_causes_with_index)
    return LLMCurrent.process(
        [
            {"role": "system", "content": prompts.prompt_find_causes},
            {"role": "user", "content": input},
        ]
    ).strip()


def query_effects(thing: str, possible_effects: list[str]) -> str:
    possible_effects_with_index = [f"{i + 1} {effect}" for i, effect in enumerate(possible_effects)]
    input = thing + "\n\nPlease find effects among:\n" + "\n".join(possible_effects_with_index)
    return LLMCurrent.process(
        [
            {"role": "system", "content": prompts.prompt_find_effects},
            {"role": "user", "content": input},
        ]
    ).strip()


def find_relations(querier, thing: str, possible_causes: list[str]) -> list[int]:
    while True:
        result = querier(thing, possible_causes)
        if result == "NONE":
            return []
        idx_str_list = result.split(" ")
        idx_list = []
        failed = False
        for idx_str in idx_str_list:
            try:
                idx = int(idx_str)
                if idx < 1 or idx > len(possible_causes):
                    failed = True
                    break
                idx_list.append(idx - 1)
            except ValueError:
                failed = True
                break
        if failed:
            warnings.warn(f"LLM output '{result}' includes invalid index. Retry.")
            continue
        else:
            return idx_list


def create_relation_matrix(
    querier, things: list[str], show_progress: bool = False
) -> list[list[int]]:
    matrix = []
    for i, thing in tqdm(
        enumerate(things),
        desc="Creating Relation Matrix",
        total=len(things),
        disable=not show_progress,
    ):
        possible_causes = [thing for j, thing in enumerate(things) if i != j]
        causes = find_relations(querier, thing, possible_causes)
        row = []
        for j in range(0, i):
            row.append(int(j in causes))
        row.append(0)
        for j in range(i + 1, len(things)):
            row.append(int(j - 1 in causes))
        matrix.append(row)
    return matrix


def create_cause_matrix(things: list[str], show_progress: bool = False) -> list[list[int]]:
    return create_relation_matrix(query_causes, things, show_progress)


def create_effect_matrix(things: list[str], show_progress: bool = False) -> list[list[int]]:
    return create_relation_matrix(query_effects, things, show_progress)


def main():
    warnings.simplefilter("always")
    input = """
这一画像描绘了当代中国数学专业本科生群体，特别是在职业规划和个人生活选择方面面临的挑战和矛盾。他们通常在家庭期望和个人兴趣之间寻找平衡，并试图在快速变化的社会中找到自己的位置。

该群体具有以下特征：

1. **教育背景**：他们通常来自中国的小城镇或农村地区，通过国家专项计划等政策支持考入985或211高校，专业以数学为主，并可能对计算机科学、金融或人工智能等领域感兴趣。

2. **家庭期望**：家庭通常对他们抱有较高的期望，希望他们能够进入稳定的体制内工作，如公务员或教师。家庭成员可能对政治和权力有强烈的追求，并期望通过孩子的职业来提升家庭的社会地位。

3. **个人兴趣与职业规划**：尽管家庭有明确的期望，但他们个人可能更倾向于追求自由和创造性较强的职业，如进入互联网大厂工作。他们通常对体制内的文化和人情世故持批判态度，认为这种环境不利于个人技能的发挥和职业发展。

4. **工作稳定性与财务自由**：他们认识到工作稳定性对于个人和家庭的重要性，但同时也在考虑财务自由的可能性。他们认为财务自由虽然难以实现，但可能是实现个人自由和生活选择的重要途径。

5. **社会关系与个人空间**：他们对体制内的人情世故和权力斗争持批判态度，并试图在社会关系中保持个人空间和独立性。他们通常不喜欢被迫参与家庭安排的社交活动，特别是那些他们认为没有实际意义的活动。

6. **对未来的期望**：他们期望未来的生活能够有足够的自由和稳定性，能够让他们有时间追求个人兴趣和享受生活。他们可能更倾向于在小城市生活，以减少生活压力和房贷负担。

7. **个人特质**：他们通常是直率的人，喜欢表达自己的观点，但有时可能会因为过于直接而得罪人。他们对自己的专业有一定的认同感，但也认识到自己在某些领域需要进一步提升。

这一画像反映了中国当代大学生在个人成长和职业规划中的矛盾和挑战，他们需要在家庭期望和个人兴趣之间找到平衡，同时也要面对快速变化的社会环境和经济形势。
"""
    from ExtractKeyPoints import extract_key_points

    key_points = extract_key_points(input)
    for i, key_point in enumerate(key_points):
        print(f"{i + 1}. {key_point}")
    cause_matrix = create_cause_matrix(key_points, show_progress=True)
    print("Cause Matrix:")
    for row in cause_matrix:
        print(row)
    effect_matrix = create_effect_matrix(key_points, show_progress=True)
    print("Effect Matrix:")
    for row in effect_matrix:
        print(row)

    # calculate Cause Matrix and Effect Matrix's similarity
    without_transpose_11 = 0
    without_transpose_10 = 0
    without_transpose_01 = 0
    without_transpose_00 = 0
    with_transpose_11 = 0
    with_transpose_10 = 0
    with_transpose_01 = 0
    with_transpose_00 = 0
    for i in range(len(key_points)):
        for j in range(len(key_points)):
            if cause_matrix[i][j] == 1 and effect_matrix[i][j] == 1:
                without_transpose_11 += 1
            if cause_matrix[i][j] == 1 and effect_matrix[i][j] == 0:
                without_transpose_10 += 1
            if cause_matrix[i][j] == 0 and effect_matrix[i][j] == 1:
                without_transpose_01 += 1
            if cause_matrix[i][j] == 0 and effect_matrix[i][j] == 0:
                without_transpose_00 += 1
            if cause_matrix[j][i] == 1 and effect_matrix[i][j] == 1:
                with_transpose_11 += 1
            if cause_matrix[j][i] == 1 and effect_matrix[i][j] == 0:
                with_transpose_10 += 1
            if cause_matrix[j][i] == 0 and effect_matrix[i][j] == 1:
                with_transpose_01 += 1
            if cause_matrix[j][i] == 0 and effect_matrix[i][j] == 0:
                with_transpose_00 += 1
    print("Without Transpose:")
    print(f"A is B's cause and A is B's effect: {without_transpose_11}")
    print(f"A is B's cause and A is not B's effect: {without_transpose_10}")
    print(f"A is not B's cause and A is B's effect: {without_transpose_01}")
    print(f"A is not B's cause and A is not B's effect: {without_transpose_00}")
    print("With Transpose:")
    print(f"A is B's cause and B is A's effect: {with_transpose_11}")
    print(f"A is B's cause and B is not A's effect: {with_transpose_10}")
    print(f"A is not B's cause and B is A's effect: {with_transpose_01}")
    print(f"A is not B's cause and B is not A's effect: {with_transpose_00}")


if __name__ == "__main__":
    main()
