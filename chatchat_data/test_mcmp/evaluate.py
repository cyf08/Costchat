import os
from typing import List
from eval_utils import tables_match

def run_evaluation(result_dir: str, gold_dir: str):
    """
    轻量版：比较两个目录下的 CSV 文件，逐一用 tables_match 进行比对。
    """
    result_files = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
    gold_files = [f for f in os.listdir(gold_dir) if f.endswith('.csv')]

    # 只比较两个文件夹里都存在的文件
    common_files = sorted(set(result_files).intersection(gold_files))

    if not common_files:
        print("两个文件夹没有匹配的 CSV 文件")
        return

    # 输出golden有而result没有的文件
    only_in_gold = set(gold_files) - set(result_files)
    if only_in_gold:
        print(f"\n=== Golden目录中有但Result目录中没有的文件 ({len(only_in_gold)}个) ===")
        for file in sorted(only_in_gold):
            print(f"缺失: {file}")
    else:
        print("\n没有缺失的文件")

    total, correct = 0, 0

    for file in common_files:
        result_path = os.path.join(result_dir, file)
        gold_path = os.path.join(gold_dir, file)

        try:
            score = tables_match([result_path], gold=[gold_path])
            total += 1
            if score == 1:
                correct += 1
                print(f" {file} 一致")
            else:
                print(f" {file} 不一致 (score={score})")
        except Exception as e:
            print(f" {file} 比较出错: {e}")

    print(f"\n总文件数: {total}, 一致: {correct}, 准确率: {correct/total:.2%}")


if __name__ == "__main__":
    result_dir = r"D:\作业\hpc\项目\联想智算中心运营LLM\codes\new_test_mcmp\expected_sql_results_extended"
    gold_dir = r"D:\作业\hpc\项目\联想智算中心运营LLM\codes\new_test_mcmp\expected_sql_results_extended_golden"

    run_evaluation(result_dir, gold_dir)
