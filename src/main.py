from LLM import LLMCurrent
from Input import Input
from Prompts import Prompts
from Classify import classify
from Split import split
from MergeInterviewSegments import merge_interview_segments
from datetime import datetime
import os
import shutil
from tqdm import tqdm

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
def iterate_portrait(portrait: str, merged: str):
    input = "[原始群体画像]\n\n" + portrait + "\n\n[个人生活史生命史]\n\n" + merged
    return LLMCurrent.process(prompts.prompt_iterate, input)


def segments_to_str(segments: list[str]) -> str:
    return "\n\n".join([f"[Segment {i+1}]\n{segment}" for i, segment in enumerate(segments)])


def iterate_initial_portraits():
    stage_dir = f"{path_output}/1_iterate_initial_portraits"
    os.mkdir(stage_dir)
    for group in input.groups:
        print(f"Iterating initial portrait for group {group.name}")
        group_dir = f"{stage_dir}/{group.name}"
        os.mkdir(group_dir)
        for i, interview in enumerate(group.initial_interviews):
            iteration_dir = f"{group_dir}/{i+1}"
            os.mkdir(iteration_dir)
            segments = split(interview.data)
            with open(f"{iteration_dir}/1_interview_segments.txt", "w", encoding="utf-8") as file:
                file.write(segments_to_str(segments))
            merged = merge_interview_segments(segments)
            with open(f"{iteration_dir}/2_merged.txt", "w", encoding="utf-8") as file:
                file.write(merged)
            group.portrait = iterate_portrait(group.portrait, merged)
            with open(f"{iteration_dir}/3_new_portrait.txt", "w", encoding="utf-8") as file:
                file.write(group.portrait)


# step 3: split all interviews and iterate portraits
def split_and_iterate_portraits():
    stage_dir = f"{path_output}/3_split_and_iterate_portraits"
    os.mkdir(stage_dir)
    for interview in input.interviews:
        print(f"Splitting and iterating portrait for interview {interview.filename}")
        interview_dir = f"{stage_dir}/{interview.filename}"
        os.mkdir(interview_dir)
        segments = split(interview.data)
        with open(f"{interview_dir}/unclassified_segments.txt", "w", encoding="utf-8") as file:
            file.write(segments_to_str(segments))
        grouped = list[list[str]]()
        for _ in input.groups:
            grouped.append([])
        for segment in tqdm(segments, desc="Classifying segments"):
            result = classify([[group.name, group.portrait] for group in input.groups], segment)
            for i in result:
                grouped[i].append(segment)
        for group in input.groups:
            group_dir = f"{interview_dir}/{group.name}"
            os.mkdir(group_dir)
            with open(f"{group_dir}/1_grouped_segments.txt", "w", encoding="utf-8") as file:
                file.write(segments_to_str(grouped[input.groups.index(group)]))
            merged = merge_interview_segments(grouped[input.groups.index(group)])
            with open(f"{group_dir}/2_merged.txt", "w", encoding="utf-8") as file:
                file.write(merged)
            group.portrait = iterate_portrait(group.portrait, merged)
            with open(f"{group_dir}/3_new_portrait.txt", "w", encoding="utf-8") as file:
                file.write(group.portrait)


def main():
    snapshot_input()
    iterate_initial_portraits()
    split_and_iterate_portraits()


if __name__ == "__main__":
    main()
