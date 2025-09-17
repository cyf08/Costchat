import os
import logging
from typing import List
from eval_utils import tables_match
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    handlers=[
        logging.FileHandler('evaluate.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def check_if_empty(file_path):
    """检查文件是否存在且非空"""
    if not os.path.exists(file_path):
        return True  # 文件不存在，视为"空"
    try:
        df = pd.read_csv(file_path)
        # 如果DataFrame为空，直接返回True
        if df.empty:
            return True
        
        # 检查所有值是否都是"none"或空值
        # 将所有值转换为字符串并检查
        all_values = df.values.flatten()
        all_none_or_empty = all(
            str(val).lower().strip() in ['none', 'nan', '', 'null'] or pd.isna(val)
            for val in all_values
        )
        
        return all_none_or_empty
    except Exception:
        return True  # 无法读取也认为是空

def run_evaluation(result_dir: str, gold_dir: str):
    """
    轻量版：比较两个目录下的 CSV 文件，逐一用 tables_match 进行比对。
    现在包含详细的统计信息。
    """
    result_files = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
    gold_files = [f for f in os.listdir(gold_dir) if f.endswith('.csv')]

    # 只比较两个文件夹里都存在的文件
    common_files = sorted(set(result_files).intersection(gold_files))

    if not common_files:
        logging.info("两个文件夹没有匹配的 CSV 文件")
        return

    # 输出golden有而result没有的文件
    only_in_gold = set(gold_files) - set(result_files)
    if only_in_gold:
        logging.info(f"\n=== Golden目录中有但Result目录中没有的文件 ({len(only_in_gold)}个) ===")
        for file in sorted(only_in_gold):
            logging.info(f"缺失: {file}")
    else:
        logging.info("\n没有缺失的文件")

    # 统计变量
    total, correct = 0, 0
    missing_files = list(only_in_gold)
    no_sort_key_files = []
    no_matching_column_files = []
    sort_failed_files = []
    sort_success_passed_files = []
    sort_success_failed_files = []
    error_files = []

    # 记录每个文件是否通过，用于回写CSV
    pass_status_by_file = {}

    for file in common_files:
        result_path = os.path.join(result_dir, file)
        gold_path = os.path.join(gold_dir, file)

        try:
            print(f"比较 {file}")
            score, debug_info = tables_match([result_path], gold=[gold_path], return_debug_info=True)
            total += 1
            
            if score == 1:
                correct += 1
                logging.info(f"✅ {file} 一致 ({debug_info})")
                pass_status_by_file[file] = True
                if debug_info == "sort_success":
                    sort_success_passed_files.append(file)
            else:
                logging.info(f"❌ {file} 不一致 ({debug_info})")
                pass_status_by_file[file] = False
                if debug_info == "no_sort_key":
                    no_sort_key_files.append(file)
                elif debug_info == "no_matching_column":
                    no_matching_column_files.append(file)
                elif debug_info == "sort_failed":
                    sort_failed_files.append(file)
                elif debug_info == "sort_success":
                    sort_success_failed_files.append(file)
                    
        except Exception as e:
            logging.info(f"⚠️  {file} 比较出错: {e}")
            error_files.append(file)
            pass_status_by_file[file] = False

    # 在日志文件末尾添加详细统计报告
    logging.info(f"\n" + "="*80)
    logging.info("📊 详细统计报告")
    logging.info("="*80)
    
    logging.info(f"\n📈 总体统计:")
    logging.info(f"   总文件数: {total}")
    logging.info(f"   一致文件: {correct}")
    logging.info(f"   不一致文件: {total - correct}")
    if total > 0:
        logging.info(f"   准确率: {correct/total:.2%}")
    
    logging.info(f"\n📁 文件缺失情况:")
    logging.info(f"   缺失文件数: {len(missing_files)}")
    if missing_files:
        for file in sorted(missing_files):
            logging.info(f"   - {file}")
    
    logging.info(f"\n🔍 处理情况统计:")
    logging.info(f"   找不到合适排序键: {len(no_sort_key_files)} 个")
    if no_sort_key_files:
        for file in sorted(no_sort_key_files):
            logging.info(f"   - {file}")
    
    logging.info(f"\n   找不到匹配列: {len(no_matching_column_files)} 个")
    if no_matching_column_files:
        for file in sorted(no_matching_column_files):
            logging.info(f"   - {file}")
    
    logging.info(f"\n   排序失败异常处理: {len(sort_failed_files)} 个")
    if sort_failed_files:
        for file in sorted(sort_failed_files):
            logging.info(f"   - {file}")
    
    logging.info(f"\n   排序正常且通过: {len(sort_success_passed_files)} 个")
    if sort_success_passed_files:
        for file in sorted(sort_success_passed_files):
            logging.info(f"   - {file}")
    
    logging.info(f"\n   排序正常未通过: {len(sort_success_failed_files)} 个")
    if sort_success_failed_files:
        for file in sorted(sort_success_failed_files):
            logging.info(f"   - {file}")
    
    logging.info(f"\n   处理出错: {len(error_files)} 个")
    if error_files:
        for file in sorted(error_files):
            logging.info(f"   - {file}")
    
    logging.info(f"\n💡 分析总结:")
    logging.info(f"   - 找不到合适排序键、找不到匹配列、排序失败的情况通常会导致不一致")
    logging.info(f"   - 排序正常但未通过的情况需要进一步分析具体的数据差异")
    logging.info(f"   - 建议重点关注排序正常未通过的文件，这些可能存在数据内容差异")

    # 更新sql_test_results.csv文件，添加"结果是否为空"和"是否匹配"两列
    csv_path = r"D:\jobs\hpc\project\LenovoLLM\codes\new_test_mcmp\sql_samples_with_match.csv"
    
    try:
        # 读取现有的CSV文件
        df = pd.read_csv(csv_path)
        
        # 检查是否已经有这两列，如果没有则添加
        if '数据匹配与否' not in df.columns:
            df['数据匹配与否'] = ''
        if '执行结果是否为空' not in df.columns:
            df['执行结果是否为空'] = ''
        
        # 根据测试ID更新匹配状态
        for file in common_files:
            # 从文件名提取测试ID（文件名格式为 case_XX-X_result.csv）
            if file.startswith('case_') and file.endswith('_result.csv'):
                test_id = file.replace('case_', '').replace('_result.csv', '')
            else:
                # 备用方案：直接去掉.csv后缀
                test_id = file.replace('.csv', '')
            
            # 清理测试ID中的特殊字符
            test_id = test_id.strip()
            
            # 查找对应的行（处理CSV中可能存在的特殊字符，包括零宽字符）
            # 使用正则表达式清理所有不可见字符
            import re
            cleaned_test_ids = df['测试ID'].str.replace(r'[\u200b-\u200d\ufeff]', '', regex=True).str.strip()
            mask = cleaned_test_ids == test_id
            if mask.any():
                # 更新匹配状态
                df.loc[mask, '数据匹配与否'] = '是' if pass_status_by_file.get(file, False) else '否'
                
                # 检查结果文件是否为空
                result_path = os.path.join(result_dir, file)
                is_empty = check_if_empty(result_path)
                df.loc[mask, '执行结果是否为空'] = '是' if is_empty else '否'
                
                logging.info(f"更新测试ID {test_id}: 匹配={pass_status_by_file.get(file, False)}, 为空={is_empty}")
            else:
                logging.info(f"未找到测试ID {test_id} 对应的行")
        
        # 保存更新后的CSV文件
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logging.info(f"\n✅ 已更新CSV文件: {csv_path}")
        logging.info(f"   添加了'数据匹配与否'和'执行结果是否为空'两列")
        
    except Exception as e:
        logging.info(f"\n⚠️  更新CSV文件时出错: {e}")
 

if __name__ == "__main__":
    result_dir = r"D:\jobs\hpc\project\LenovoLLM\codes\new_test_mcmp\actual_sql_results_extended"
    gold_dir = r"D:\jobs\hpc\project\LenovoLLM\codes\new_test_mcmp\expected_sql_results_extended_golden"

    run_evaluation(result_dir, gold_dir)