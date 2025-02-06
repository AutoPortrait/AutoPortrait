from LLM import LLMFast, LLMInstructional
from Input import Input, Group
from Prompts import Prompts
from Classify import classify
from Split import split
from AnalyzeSegments import analyze_segments
from ExtractKeyPoints import extract_key_points
from AnalyzeCauseEffect import create_cause_effect_matrix
from IteratePortrait import iterate_portrait
from Merge import merge
from datetime import datetime
import os
import shutil
from tqdm.asyncio import tqdm
import warnings
import asyncio

debug_split_interviews_and_iterate_portraits = False
extract_key_points_and_cause_effect = False
continue_last_run = True

if continue_last_run:
    path_output = "output"
    last_run = sorted(os.listdir(path_output))[-1]
    path_output = f"{path_output}/{last_run}"
    print(f" - Continuing last run '{last_run}'")
else:
    path_output = "output"
    random_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_output = f"{path_output}/{random_dirname}"
    os.mkdir(path_output)


def ensure_dir_exists(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


input = Input()
prompts = Prompts()


usage_fast_last = 0
usage_instructional_last = 0


def report_usage(title: str):
    global usage_fast_last, usage_instructional_last
    usage_fast = LLMFast.token_usage()
    usage_instructional = LLMInstructional.token_usage()
    print(
        f" - [{title}] Fast: {usage_fast - usage_fast_last}, Instructional: {usage_instructional - usage_instructional_last}"
    )
    usage_fast_last = usage_fast
    usage_instructional_last = usage_instructional


# step 0: snapshot input and prompt
def snapshot_input():
    if continue_last_run:
        print(" - Continuing last run, skipping snapshot")
        return
    stage_dir = f"{path_output}/0_snapshot"
    os.mkdir(stage_dir)
    shutil.copytree("input", f"{stage_dir}/input")
    shutil.copytree("prompt", f"{stage_dir}/prompt")


# step 1: iterate initial portraits
def segments_to_str(segments: list[str], header="Segment") -> str:
    return "\n\n".join(
        [f"[{header} {i+1}]\n{segment.strip()}" for i, segment in enumerate(segments)]
    )


def str_to_segments(data: str, header="Segment") -> list[str]:
    segments_with_prefix = data.split(f"[{header} ")
    segments = []
    cnt = 0
    for segment_with_prefix in segments_with_prefix:
        if segment_with_prefix == "":
            continue
        cnt += 1
        # segment can have multi lines
        parts = segment_with_prefix.split("\n", 1)
        assert len(parts) == 2
        assert parts[0] == f"{cnt}]"
        segment = parts[1].strip()
        segments.append(segment)
    return segments


async def iterate(group: Group, segments: list[str], workdir: str):
    if len(segments) == 0:
        return
    with open(f"{workdir}/1_interview_segments.txt", "w", encoding="utf-8") as file:
        file.write(segments_to_str(segments))

    if os.path.exists(f"{workdir}/2_analysis.txt"):
        with open(f"{workdir}/2_analysis.txt", "r", encoding="utf-8") as file:
            analysis = str_to_segments(file.read(), header="Analysis")
    else:
        analysis = await analyze_segments(segments)
        with open(f"{workdir}/2_analysis.txt", "w", encoding="utf-8") as file:
            file.write(segments_to_str(analysis, header="Analysis"))
    report_usage(f"Analyze")

    if os.path.exists(f"{workdir}/3_portrait_addition.txt"):
        with open(f"{workdir}/3_portrait_addition.txt", "r", encoding="utf-8") as file:
            additions = str_to_segments(file.read(), header="Addition")
    else:
        additions = await iterate_portrait(group.portrait, segments, analysis)
        with open(f"{workdir}/3_portrait_addition.txt", "w", encoding="utf-8") as file:
            file.write(segments_to_str(additions, header="Addition"))
    report_usage(f"Addition")

    if os.path.exists(f"{workdir}/4_new_portrait.txt"):
        with open(f"{workdir}/4_new_portrait.txt", "r", encoding="utf-8") as file:
            group.portrait = file.read()
    else:
        group.portrait = await merge(group.portrait, additions)
        with open(f"{workdir}/4_new_portrait.txt", "w", encoding="utf-8") as file:
            file.write(group.portrait)
    report_usage(f"Merge")

    if extract_key_points_and_cause_effect:
        if os.path.exists(f"{workdir}/5_key_points.txt"):
            with open(f"{workdir}/5_key_points.txt", "r", encoding="utf-8") as file:
                key_points = str_to_segments(file.read(), header="Key Point")
        else:
            key_points = await extract_key_points(group.portrait)
            with open(f"{workdir}/5_key_points.txt", "w", encoding="utf-8") as file:
                file.write(segments_to_str(key_points, header="Key Point"))
        if not os.path.exists(f"{workdir}/6_cause_effect.txt"):
            cause_effect_matrix = await create_cause_effect_matrix(key_points)
            with open(f"{workdir}/6_cause_effect.txt", "w", encoding="utf-8") as file:
                for i in range(len(cause_effect_matrix)):
                    for j in range(len(cause_effect_matrix[i])):
                        if cause_effect_matrix[i][j] > 0:
                            file.write(
                                f"{100*cause_effect_matrix[i][j]:.2f}% {key_points[i]} -> {key_points[j]}\n"
                            )
        report_usage(f"Key Points & Cause Effect")


async def iterate_initial_portraits():
    stage_dir = f"{path_output}/1_iterate_initial_portraits"
    ensure_dir_exists(stage_dir)
    for group in input.groups:
        print(f"Iterating initial portrait for group '{group.name}'")
        group_dir = f"{stage_dir}/{group.name}"
        ensure_dir_exists(group_dir)
        for i, interview in enumerate(group.initial_interviews):
            iteration_dir = f"{group_dir}/{i+1}_{interview.filename}"
            ensure_dir_exists(iteration_dir)

            if os.path.exists(f"{iteration_dir}/1_interview_segments.txt"):
                with open(
                    f"{iteration_dir}/1_interview_segments.txt", "r", encoding="utf-8"
                ) as file:
                    segments = str_to_segments(file.read())
            else:
                segments = await split(interview.data)
            report_usage(f"Split")

            await iterate(group, segments, iteration_dir)

            if debug_split_interviews_and_iterate_portraits:
                break


# step 2: split all interviews and iterate portraits
async def split_interviews_and_iterate_portraits():
    stage_dir = f"{path_output}/2_split_interviews_and_iterate_portraits"
    ensure_dir_exists(stage_dir)
    for i, interview in enumerate(input.interviews):
        print(f"Splitting '{interview.filename}' and iterating portrait")
        interview_dir = f"{stage_dir}/{i+1}_{interview.filename}"
        ensure_dir_exists(interview_dir)

        if os.path.exists(f"{interview_dir}/unclassified_segments.txt"):
            with open(f"{interview_dir}/unclassified_segments.txt", "r", encoding="utf-8") as file:
                segments = str_to_segments(file.read())
        else:
            segments = await split(interview.data)
            with open(f"{interview_dir}/unclassified_segments.txt", "w", encoding="utf-8") as file:
                file.write(segments_to_str(segments))
        report_usage(f"Split")

        classified = True
        for group in input.groups:
            if not os.path.exists(f"{interview_dir}/{group.name}"):
                classified = False
                break
        grouped = list[list[str]]()
        for _ in input.groups:
            grouped.append([])
        if classified:
            for j, group in enumerate(input.groups):
                group_dir = f"{interview_dir}/{group.name}"
                with open(f"{group_dir}/1_interview_segments.txt", "r", encoding="utf-8") as file:
                    segments = str_to_segments(file.read())
                grouped[j] = segments
        else:
            results = []
            for segment in segments:
                results.append(
                    classify([[group.name, group.portrait] for group in input.groups], segment)
                )
            results = await tqdm.gather(*results, desc="Classifying segments", leave=False)
            for result in results:
                for j in result:
                    grouped[j].append(segment)
        report_usage(f"Classify")
        for j, group in enumerate(input.groups):
            group_dir = f"{interview_dir}/{group.name}"
            ensure_dir_exists(group_dir)
            await iterate(group, grouped[j], group_dir)


async def main():
    warnings.simplefilter("always")
    # LLMFast.set_logfile(f"{path_output}/llm.log")
    LLMFast.set_logfile(f"{path_output}/llm.log")
    LLMInstructional.set_logfile(f"{path_output}/llm.log")
    LLMFast.set_progress_logfile(f"{path_output}/llm_progress.log")
    LLMInstructional.set_progress_logfile(f"{path_output}/llm_progress.log")
    snapshot_input()
    await iterate_initial_portraits()
    await split_interviews_and_iterate_portraits()


if __name__ == "__main__":
    asyncio.run(main())
