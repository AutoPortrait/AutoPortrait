from LLM import LLMCurrent
from Input import Input, Group
from Prompts import Prompts
from Classify import classify
from Split import split
from MergeInterviewSegments import merge_interview_segments
from ExtractKeyPoints import extract_key_points
from AnalyzeCauseEffect import create_cause_effect_matrix
from datetime import datetime
import os
import shutil
from tqdm import tqdm
import warnings

debug_split_interviews_and_iterate_portraits = False

path_output = "output"
random_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
path_output = f"{path_output}/{random_dirname}"
os.mkdir(path_output)

input = Input()
prompts = Prompts()


# step 0: snapshot input and prompt
def snapshot_input():
    stage_dir = f"{path_output}/0_snapshot"
    os.mkdir(stage_dir)
    shutil.copytree("input", f"{stage_dir}/input")
    shutil.copytree("prompt", f"{stage_dir}/prompt")


# step 1: iterate initial portraits
def segments_to_str(segments: list[str]) -> str:
    return "\n".join([f"[Segment {i+1}]\n{segment}" for i, segment in enumerate(segments)])


def iterate_portrait(group: Group, segments: list[str], workdir: str):
    if len(segments) == 0:
        return
    usage_before = LLMCurrent.token_usage()
    with open(f"{workdir}/1_interview_segments.txt", "w", encoding="utf-8") as file:
        file.write(segments_to_str(segments))
    merged = merge_interview_segments(segments)
    with open(f"{workdir}/2_merged.txt", "w", encoding="utf-8") as file:
        file.write(merged)
    input = "[原始群体画像]\n\n" + group.portrait + "\n\n[个人生活史生命史]\n\n" + merged
    group.portrait = LLMCurrent.process(prompts.prompt_iterate, input)
    with open(f"{workdir}/3_new_portrait.txt", "w", encoding="utf-8") as file:
        file.write(group.portrait)
    if not debug_split_interviews_and_iterate_portraits:
        key_points = extract_key_points(group.portrait)
        with open(f"{workdir}/4_key_points.txt", "w", encoding="utf-8") as file:
            for i, key_point in enumerate(key_points):
                file.write(f"{i+1}. {key_point}\n")
        cause_effect_matrix = create_cause_effect_matrix(key_points)
        with open(f"{workdir}/5_cause_effect.txt", "w", encoding="utf-8") as file:
            for i in range(len(cause_effect_matrix)):
                for j in range(len(cause_effect_matrix[i])):
                    if cause_effect_matrix[i][j] == 1:
                        file.write(f"{key_points[i]} -> {key_points[j]}\n")
    usage_after = LLMCurrent.token_usage()
    print(f"Used {usage_after - usage_before} tokens for this iteration of group '{group.name}'")


def iterate_initial_portraits():
    stage_dir = f"{path_output}/1_iterate_initial_portraits"
    os.mkdir(stage_dir)
    for group in input.groups:
        print(f"Iterating initial portrait for group '{group.name}'")
        group_dir = f"{stage_dir}/{group.name}"
        os.mkdir(group_dir)
        for i, interview in enumerate(group.initial_interviews):
            iteration_dir = f"{group_dir}/{i+1}_{interview.filename}"
            os.mkdir(iteration_dir)
            segments = split(interview.data)
            iterate_portrait(group, segments, iteration_dir)
            if debug_split_interviews_and_iterate_portraits:
                break


# step 2: split all interviews and iterate portraits
def split_interviews_and_iterate_portraits():
    stage_dir = f"{path_output}/2_split_interviews_and_iterate_portraits"
    os.mkdir(stage_dir)
    for i, interview in tqdm(
        enumerate(input.interviews),
        total=len(input.interviews),
        desc="Processing interview",
        leave=False,
    ):
        print(f"Splitting '{interview.filename}' and iterating portrait")
        interview_dir = f"{stage_dir}/{i+1}_{interview.filename}"
        os.mkdir(interview_dir)
        segments = split(interview.data)
        with open(f"{interview_dir}/unclassified_segments.txt", "w", encoding="utf-8") as file:
            file.write(segments_to_str(segments))
        grouped = list[list[str]]()
        for _ in input.groups:
            grouped.append([])
        usage_begin = LLMCurrent.token_usage()
        for segment in tqdm(segments, desc="Classifying segments", leave=False):
            result = classify([[group.name, group.portrait] for group in input.groups], segment)
            for j in result:
                grouped[j].append(segment)
        usage_after = LLMCurrent.token_usage()
        print(
            f"Used {usage_after - usage_begin} tokens for classifying segments of '{interview.filename}'"
        )
        for j, group in enumerate(input.groups):
            group_dir = f"{interview_dir}/{group.name}"
            os.mkdir(group_dir)
            iterate_portrait(group, grouped[j], group_dir)


def main():
    warnings.simplefilter("always")
    snapshot_input()
    iterate_initial_portraits()
    split_interviews_and_iterate_portraits()
    usage = LLMCurrent.token_usage()
    print(f"Used {usage} tokens in total")


if __name__ == "__main__":
    main()
