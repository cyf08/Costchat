import re
import pandas as pd
import math
import duckdb
from typing import List, Union
import os
import pandas as pd
from google.cloud import bigquery


def string_match(pred, gold, conj="or", exclude=[]):
    """

    Parameters:
    - pred (str): The string in which to search for substrings.
    - gold (list of str): A list of strings to be checked within the pred string.
    - conj (str): The conjunction to use for matching ('and' or 'or'). Defaults to 'or'. 
    - exclude (list of str): some string can't exist in the answer.

    Returns:
    - int: Returns 1 if the condition specified by 'conj' is met, otherwise returns 0.

    """
    
    if not isinstance(gold, list):
        gold = [gold]
    
    pred_lower = pred.lower()
    gold_lower = [sub.lower() for sub in gold]
    exclude_lower = [sub.lower() for sub in exclude]

    if any(sub in pred_lower for sub in exclude_lower):
        return 0  # Return 0 if any excluded items are found
    
    if conj == "or":
        # Check if any of the strings in 'gold_lower' are in 'pred_lower'
        return 1 if any(sub in pred_lower for sub in gold_lower) else 0
    elif conj == "and":
        # Check if all of the strings in 'gold_lower' are in 'pred_lower'
        return 1 if all(sub in pred_lower for sub in gold_lower) else 0
    else:
        raise ValueError("Invalid value for 'conj'. Choose 'and' or 'or'.")



def number_match(pred, gold, percentage=False, precision=4, conj="or"):
    """
    Parameters:
    - pred (str): The string in which to search for substrings.
    - gold (list[str|float] of str): A list of string/numbers to be checked within the pred string.
    - percentage (bool): default false. if the gold answer is related with "percentage", set it as true to make the evaluation robust.
    - precision: Decimal places
    - conj (str): The conjunction to use for matching ('and' or 'or'). Defaults to 'or'. Most time is 'or'
    
    """
    
    import regex

    def extract_numbers(input_string):
        """
        Extracts all numbers from a given string including integers, floating-point numbers,
        numbers with commas, and percentages, returning them as a list of strings in their
        original numeric format. This function correctly includes the percentage symbol if present.
        """
        number_pattern = r'\b\d{1,3}(?:,\d{3})*\b(?:\.\d+)?%?|\b\d+\b(?:\.\d+)?%?'
        matches = regex.findall(number_pattern, input_string)
        return matches

    def convert_to_float(value):
        """
        Convert string to float, removing commas and handling percentages if specified.
        Converts percentages by dividing the number by 100.
        """
        value = str(value).replace(',', '')  # Ensure value is treated as a string
        if '%' in value:
            value = value.replace('%', '')
            return float(value) / 100
        return float(value)
    
    
    def is_within_precision(converted_pred_numbers, gold_value, precision):
        """
        Checks if any number in converted_pred_numbers is within a specified precision of gold_value.
        """
        return any(abs(num - gold_value) <= 10 ** (-precision) for num in converted_pred_numbers)

    pred_numbers = extract_numbers(pred)


    if (isinstance(gold,(str,float)) or (isinstance(gold, list)  and len(gold)==1 )) and len(pred_numbers)!=1:
        return 0
    converted_pred_numbers = [convert_to_float(num) for num in pred_numbers]
    gold = [convert_to_float(g) for g in (gold if isinstance(gold, list) else [gold])]
    
    if percentage:
        gold = [y for x in gold for y in (x, x * 100)]

    if conj == "and":
        return 1 if all(is_within_precision(converted_pred_numbers, gold_value, precision) for gold_value in gold) else 0
    elif conj == "or":
        return 1 if any(is_within_precision(converted_pred_numbers, gold_value, precision) for gold_value in gold) else 0
    else:
        raise ValueError(f"Invalid value for 'conj'. Choose 'and' or 'or'. Received: {conj}")
    


    

def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False, return_debug_info=False):

    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        try:
            if ignore_order_:
                v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                        sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if pd.isna(a) and pd.isna(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    # Normalize both numbers to handle scientific notation vs regular format
                    fa, fb = float(a), float(b)
                    # Use a more lenient tolerance for scientific notation comparison
                    if not math.isclose(fa, fb, abs_tol=tol, rel_tol=1e-6):
                        return False
                elif a != b:
                    return False
            return True
        except Exception as e:
            return False
    
    def _normalize_month_string(val: str):
        try:
            s = str(val).strip()
            # Already a 6-digit yyyymm
            if re.fullmatch(r"\d{6}", s):
                return s
            # yyyy-mm or yyyy/mm
            m = re.fullmatch(r"(\d{4})[-/](\d{2})", s)
            if m:
                return f"{m.group(1)}{m.group(2)}"
            # integers that look like 202506
            if re.fullmatch(r"\d{4}\d{2}", s):
                return s
        except Exception:
            pass
        return val

    def _normalize_month_columns(df: pd.DataFrame) -> pd.DataFrame:
        df_norm = df.copy()
        # Value normalization for columns that look like month columns
        for col in df_norm.columns:
            if 'month' in str(col).lower():
                try:
                    df_norm[col] = df_norm[col].apply(_normalize_month_string)
                except Exception:
                    pass
        # Add alias columns to maximize common keys
        cols_lower = {c.lower(): c for c in df_norm.columns}
        # If month exists only as month_period, create month alias
        if 'month' not in cols_lower and 'month_period' in cols_lower:
            try:
                df_norm['month'] = df_norm[cols_lower['month_period']]
            except Exception:
                pass
        # If month_period missing but month exists, create month_period alias
        if 'month_period' not in cols_lower and 'month' in cols_lower:
            try:
                df_norm['month_period'] = df_norm[cols_lower['month']]
            except Exception:
                pass
        return df_norm

    # Normalize month formats and add aliases before comparison
    pred = _normalize_month_columns(pred)
    gold = _normalize_month_columns(gold)

    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    
    # 1) 若 gold 的列均在 pred 中，走原有子集直接比较逻辑
    # gold_columns_set = set(gold_cols.columns)
    # pred_columns_set = set(pred.columns)
    # if gold_columns_set.issubset(pred_columns_set):
    #     pred_cols = pred[list(gold_cols.columns)]
    #     t_gold_list = gold_cols.transpose().values.tolist()
    #     t_pred_list = pred_cols.transpose().values.tolist()
    #     score = 1
    #     for _, gold_vec in enumerate(t_gold_list):
    #         if not any(vectors_match(gold_vec, pred_vec, ignore_order_=ignore_order) for pred_vec in t_pred_list):
    #             score = 0
    #         else:
    #             for j, pred_vec in enumerate(t_pred_list):
    #                 if vectors_match(gold_vec, pred_vec, ignore_order_=ignore_order):
    #                     break
    #     return score
    
    # 2) 先按键列排序对齐，再进行列值匹配
    def find_best_sort_key(gold_df):
        """找到 gold 中唯一性最高的列作为排序键"""
        best_col = None
        best_uniqueness = 0
        for col in gold_df.columns:
            unique_count = gold_df[col].nunique()
            if unique_count > best_uniqueness:
                best_uniqueness = unique_count
                best_col = col
        return best_col
    
    def find_matching_column(gold_col, pred_df):
        """在 pred 中找到与 gold_col 最匹配的列"""
        gold_values = gold_col.tolist()
        best_match_col = None
        best_match_score = 0
        
        for col in pred_df.columns:
            pred_values = pred_df[col].tolist()
            # 计算匹配度：相同值的数量
            common_values = set(gold_values) & set(pred_values)
            match_score = len(common_values)
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_col = col
        
        return best_match_col
    
    # 找到 gold 的最佳排序键
    gold_sort_key = find_best_sort_key(gold_cols)
    # print(f"DEBUG: Gold sort key: {gold_sort_key}")
    # print(f"DEBUG: Gold columns: {list(gold_cols.columns)}")
    # print(f"DEBUG: Pred columns: {list(pred.columns)}")
    
    if gold_sort_key is None:
        # print("DEBUG: No suitable sort key found, falling back to pure column matching")
        # 如果没有找到合适的键，退化为纯列值匹配
        
        # 设置pandas显示选项以显示完整列信息
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(f"DEBUG: Gold sorted:\n{gold_cols}")
        print(f"DEBUG: Pred sorted:\n{pred}")
        
        # 恢复默认显示设置
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred.transpose().values.tolist()
        score = 1
        for _, gold_vec in enumerate(t_gold_list):
            if not any(vectors_match(gold_vec, pred_vec, ignore_order_=True) for pred_vec in t_pred_list):
                score = 0
            else:
                for j, pred_vec in enumerate(t_pred_list):
                    if vectors_match(gold_vec, pred_vec, ignore_order_=True):
                        break
        
        if return_debug_info:
            return score, "no_sort_key"
        return score
    
    # 在 pred 中找到对应的排序键
    pred_sort_key = find_matching_column(gold_cols[gold_sort_key], pred)
    # print(f"DEBUG: Pred sort key: {pred_sort_key}")
    # print(f"DEBUG: Gold sort key values: {gold_cols[gold_sort_key].tolist()}")
    # if pred_sort_key:
    #     print(f"DEBUG: Pred sort key values: {pred[pred_sort_key].tolist()}")
    
    if pred_sort_key is None:
        # print("DEBUG: No matching column found, falling back to pure column matching")
        # 如果找不到匹配的列，退化为纯列值匹配
        
        # 设置pandas显示选项以显示完整列信息
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(f"DEBUG: Gold sorted:\n{gold_cols}")
        print(f"DEBUG: Pred sorted:\n{pred}")
        
        # 恢复默认显示设置
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred.transpose().values.tolist()
        score = 1
        for _, gold_vec in enumerate(t_gold_list):
            if not any(vectors_match(gold_vec, pred_vec, ignore_order_=True) for pred_vec in t_pred_list):
                score = 0
            else:
                for j, pred_vec in enumerate(t_pred_list):
                    if vectors_match(gold_vec, pred_vec, ignore_order_=True):
                        break
        
        if return_debug_info:
            return score, "no_matching_column"
        return score
    
    # 按排序键对齐两边的数据
    try:
        # print(f"DEBUG: Sorting gold by {gold_sort_key}")
        # print(f"DEBUG: Sorting pred by {pred_sort_key}")
        gold_sorted = gold_cols.sort_values(by=gold_sort_key).reset_index(drop=True)
        pred_sorted = pred.sort_values(by=pred_sort_key).reset_index(drop=True)
        
        # 设置pandas显示选项以显示完整列信息
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(f"DEBUG: Gold sorted:\n{gold_sorted}")
        print(f"DEBUG: Pred sorted:\n{pred_sorted}")
        
        # 恢复默认显示设置
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
        # 现在进行列值匹配，但不需要忽略顺序了
        t_gold_list = gold_sorted.transpose().values.tolist()
        t_pred_list = pred_sorted.transpose().values.tolist()
        
        # print(f"DEBUG: Gold vectors: {t_gold_list}")
        # print(f"DEBUG: Pred vectors: {t_pred_list}")
        
        score = 1
        for i, gold_vec in enumerate(t_gold_list):
            # print(f"DEBUG: Checking gold vector {i}: {gold_vec}")
            matched = False
            for j, pred_vec in enumerate(t_pred_list):
                # print(f"DEBUG:   vs pred vector {j}: {pred_vec}")
                if vectors_match(gold_vec, pred_vec, ignore_order_=False):
                    # print(f"DEBUG:   MATCHED!")
                    matched = True
                    break
                # else:
                #     print(f"DEBUG:   NOT MATCHED")
            if not matched:
                # print(f"DEBUG: Gold vector {i} has no match, score = 0")
                score = 0
                break
        # print(f"DEBUG: Final score: {score}")
        if return_debug_info:
            return score, "sort_success"
        return score
    except Exception as e:
        print(f"DEBUG: Sorting failed: {e}, falling back to pure column matching")
        # 如果排序失败，退化为纯列值匹配
        
        # 设置pandas显示选项以显示完整列信息
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(f"DEBUG: Gold sorted:\n{gold_cols}")
        print(f"DEBUG: Pred sorted:\n{pred}")
        
        # 恢复默认显示设置
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred.transpose().values.tolist()
        score = 1
        for _, gold_vec in enumerate(t_gold_list):
            if not any(vectors_match(gold_vec, pred_vec, ignore_order_=True) for pred_vec in t_pred_list):
                score = 0
            else:
                for j, pred_vec in enumerate(t_pred_list):
                    if vectors_match(gold_vec, pred_vec, ignore_order_=True):
                        break
        
        print(f"DEBUG: Final score: {score}")
        if return_debug_info:
            return score, "sort_failed"
        return score
    
    # 原来的键列对齐逻辑（已注释）
    # # 2) 列名不完全一致时：
    # #    - 先找共同键列（exact name 交集），用它们对齐两边的行
    # #    - 再为 gold 的非键列在 pred 的非键列中按值匹配寻找一一映射
    # # 不比较列名：强制不使用键列，退化为纯列值对齐
    # common_key_cols = []
    # 
    # if len(common_key_cols) == 0:
    #     # 无共同键列，降级为列向量集合相似性（原逻辑）
    #     pred_cols = pred
    #     t_gold_list = gold_cols.transpose().values.tolist()
    #     t_pred_list = pred_cols.transpose().values.tolist()
    #     score = 1
    #     for _, gold_vec in enumerate(t_gold_list):
    #         if not any(vectors_match(gold_vec, pred_vec, ignore_order_=True) for pred_vec in t_pred_list):
    #             score = 0
    #         else:
    #             for j, pred_vec in enumerate(t_pred_list):
    #                 if vectors_match(gold_vec, pred_vec, ignore_order_=True):
    #                     break
    #     return score
    # 
    # # 用共同键列进行对齐（内连接，要求每行都能在对侧找到匹配）
    # try:
    #     merged = pd.merge(gold_cols, pred, on=common_key_cols, how='left', suffixes=('_gold','_pred'), copy=False)
    # except Exception:
    #     return 0
    # 
    # # 如果有行未匹配到（非键字段全 NaN），则失败
    # if merged.shape[0] != gold_cols.shape[0]:
    #     return 0
    # 
    # # 验证键列数据一致（按行比较，允许顺序忽略时可排序）
    # if ignore_order and len(common_key_cols) > 0:
    #     gold_sorted = gold_cols.sort_values(by=common_key_cols).reset_index(drop=True)
    #     pred_sorted = pred.sort_values(by=common_key_cols).reset_index(drop=True)
    #     # 再次 merge 确保一一对齐
    #     merged = pd.merge(gold_sorted, pred_sorted, on=common_key_cols, how='left', suffixes=('_gold','_pred'), copy=False)
    #     if merged.shape[0] != gold_sorted.shape[0]:
    #         return 0
    # 
    # gold_only_cols = [c for c in gold_cols.columns if c not in common_key_cols]
    # pred_only_cols = [c for c in pred.columns if c not in common_key_cols]
    # 
    # # 为每个 gold_only 列在 pred_only 中寻找一个按数值/文本匹配的列
    # used_pred_cols = set()
    # for gcol in gold_only_cols:
    #     gseries = merged[gcol]
    #     found_match = False
    #     for pcol in pred_only_cols:
    #         if pcol in used_pred_cols:
    #             continue
    #         pseries = merged[pcol]
    #         if vectors_match(gseries.tolist(), pseries.tolist(), ignore_order_=True):
    #             used_pred_cols.add(pcol)
    #             found_match = True
    #             break
    #     if not found_match:
    #         return 0
    # 
    # return 1
    

def compare_multi_pandas_table(pred, multi_gold, multi_condition_cols=[], multi_ignore_order=False, return_debug_info=False):
    if multi_condition_cols == [] or multi_condition_cols == [[]] or multi_condition_cols == [None] or multi_condition_cols == None:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]
    
    for i, gold in enumerate(multi_gold):
        if return_debug_info:
            score, debug_info = compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i], return_debug_info)
            if score == 1:
                return 1, debug_info
        else:
            if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
                return 1
    if return_debug_info:
        return 0, "multi_no_match"
    return 0


def table_match(result: str, gold, condition_cols=[], ignore_order=False, return_debug_info=False):
    """ 
    @args:
        result (str):
        gold (str|List):
        condition_cols (List[int])
        ignore_order (bool)
        return_debug_info (bool)
    """
    df1 = pd.read_csv(result, low_memory=False)
    
    if isinstance(gold, str):
        df2 = pd.read_csv(gold, low_memory=False)
        if return_debug_info:
            score, debug_info = compare_pandas_table(df1, df2, condition_cols=condition_cols, ignore_order=ignore_order, return_debug_info=True)
            return score, debug_info
        else:
            score = compare_pandas_table(df1, df2, condition_cols=condition_cols, ignore_order=ignore_order)
            return score
    elif isinstance(gold, List):
        df_list = [pd.read_csv(g, low_memory=False) for g in gold]
        if return_debug_info:
            score, debug_info = compare_multi_pandas_table(df1, df_list, multi_condition_cols=condition_cols, multi_ignore_order=ignore_order, return_debug_info=True)
            return score, debug_info
        else:
            score = compare_multi_pandas_table(df1, df_list, multi_condition_cols=condition_cols, multi_ignore_order=ignore_order)
            return score



def duckdb_match(result: str, gold: str, condition_tabs=None, condition_cols: List[List[int]]=None, ignore_orders: List[bool]=None):
    """
    Parameters:
    - result (str): Path to the DuckDB file containing the result tables.
    - gold (str): Path to the DuckDB file containing the gold standard tables.
    - condition_tabs (List[str], optional): List of table names to be checked. If not provided, all tables in the gold DuckDB file will be considered.
    - condition_cols (List[List[int]], optional): A list of lists, where each inner list contains column indices used for matching conditions for the corresponding table. Defaults to considering all columns.
    - ignore_orders (List[bool], optional): A list of boolean values indicating whether to ignore the row order for each table comparison. Defaults to [False] for each table.
    """
   
   
    def get_duckdb_table_names(db: str) -> List[str]:
        """
        Retrieves the names of all tables in the DuckDB database.

        Parameters:
        - db (str): The path to the DuckDB database file.

        Returns:
        - List[str]: A list of table names in the DuckDB database.
        """
        con = duckdb.connect(database=db, read_only=True)
        result = con.execute("SHOW TABLES").fetchall()
        con.close()
        return [row[0] for row in result]
    
    
    def get_duckdb_pandas_table(db, table_name):
        con = duckdb.connect(database=db, read_only=True)
        df = con.execute(f'SELECT * FROM {table_name}').fetchdf()
        con.close()
        return df
    
    if condition_tabs is None:
        condition_tabs = get_duckdb_table_names(gold)

    gold_tables = [get_duckdb_pandas_table(gold, table_name) for table_name in condition_tabs]
    try:
        pred_tables = [get_duckdb_pandas_table(result, table_name) for table_name in condition_tabs]
    except:
        return 0
    
    assert len(gold_tables) == len(pred_tables)

    if ignore_orders is None:
        ignore_orders = [False] * len(gold_tables)
        
    assert len(ignore_orders) == len(gold_tables)

    if condition_cols is None:
        condition_cols = [[]] * len(gold_tables)


    assert len(condition_cols) == len(gold_tables)
    
    for i, (gold_table, pred_table) in enumerate(zip(gold_tables, pred_tables)):
        if not compare_pandas_table(pred_table, gold_table, condition_cols=condition_cols[i], ignore_order=ignore_orders[i]):
            return 0
        
    return 1


def tables_match(result: List[str], gold: List[str], condition_cols: List[List[int]]=None, ignore_orders: List[bool]=None, return_debug_info=False):
    """
    Parameters:
    - result (Lstr): Path to the result tables.
    - gold (str): Path to the gold standard tables.
    - condition_cols (List[List[int]], optional): A list of lists, where each inner list contains column indices used for matching conditions for the corresponding table. Defaults to considering all columns.
    - ignore_orders (List[bool], optional): A list of boolean values indicating whether to ignore the row order for each table comparison. Defaults to [False] for each table.
    """

    def get_tables_to_dfs(csv_file: str):
        df = pd.read_csv(csv_file)
        return df

    gold_tables = [get_tables_to_dfs(table_name) for table_name in gold]
    try:
        pred_tables = [get_tables_to_dfs(table_name) for table_name in result]
    except:
        return 0
    
    assert len(gold_tables) == len(pred_tables)

    if ignore_orders is None:
        ignore_orders = [False] * len(gold_tables)
        
    assert len(ignore_orders) == len(gold_tables)

    if condition_cols is None:
        condition_cols = [[]] * len(gold_tables)


    assert len(condition_cols) == len(gold_tables)
    
    for i, (gold_table, pred_table) in enumerate(zip(gold_tables, pred_tables)):
        # print(f"DEBUG: Comparing table {i}")
        # print(f"DEBUG: Gold table: {gold_table}")
        # print(f"DEBUG: Pred table: {pred_table}")
        # print(f"DEBUG: Condition cols: {condition_cols[i]}")
        # print(f"DEBUG: Ignore orders: {ignore_orders[i]}")
        if return_debug_info:
            score, debug_info = compare_pandas_table(pred_table, gold_table, condition_cols=condition_cols[i], ignore_order=ignore_orders[i], return_debug_info=True)
            if score == 0:
                return 0, debug_info
        else:
            if not compare_pandas_table(pred_table, gold_table, condition_cols=condition_cols[i], ignore_order=ignore_orders[i]):
                return 0
        
    if return_debug_info:
        return 1, "all_tables_match"
    return 1

    
    
def get_bigquery_sql_result(sql_query, is_save, save_dir=None, save_file="result.csv"):
    """
    is_save = True, output a 'result.csv'
    if_save = False, output a string
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials/bigquery_credential.json"
    client = bigquery.Client()
    query_job = client.query(sql_query)
    try:
      results = query_job.result().to_dataframe() 
      if results.empty:
        print("No data found for the specified query.")
      else:
        if is_save:
            results.to_csv(os.path.join(save_dir, save_file), index=False)
            return None
        else:
            value = results.iat[0, 0]
            return value
    except Exception as e:
      print("Error occurred while fetching data: ", e)
    
    
    
def execute_process(eval_func, eval_metadata_parameters, gold_dir, file_name="result.csv"):
    
    sql_query = open(os.path.join(gold_dir, "gold.sql"), 'r', encoding='utf-8').read()
    if eval_func in ["number_match", "string_match"]:
        answer = get_bigquery_sql_result(sql_query=sql_query, is_save=False)
        eval_metadata_parameters["gold"] = answer
    elif eval_func == "table_match":
        answer = get_bigquery_sql_result(sql_query=sql_query, is_save=True, save_dir=gold_dir, save_file=file_name)
        eval_metadata_parameters["gold"] = file_name
    
    return eval_metadata_parameters
        
        

if __name__ == "__main__":
    
    
    eval_metadata = """
        {
            "func": "table_match", 
            "temporal": true, 
            "parameters": {
                
            }
        }
    """
    eval_json = json.loads(eval_metadata)
    eval_metadata = execute_process(eval_json["func"], eval_json["parameters"], "/Users/leifangyu/workspace/Spider2-C/Spider2-C/evaluation_suite/gold/335fb285-c9fd-45ff-ba8d-fe89a62016f7")

    print(eval_metadata)


    # eval_json = json.loads(eval_metadata)
    # import pdb; pdb.set_trace()
    # answer = get_bigquery_sql_result(eval_json["func"], gold_sql, eval_json["parameters"])
    # print(answer)
    
    # eval_metadata = """
    #     {
    #         "func": "table_match",
    #         "parameters": {
    #             "gold": "./gold/1d009ac3-1c75-447b-a7e0-49ccc2b5fbf9/result.csv",
    #             "condition_cols": [1],
    #             "ignore_order": true
    #         }
    #     }
    # """
    
    
    # eval_metadata = """
    #     {
    #         "func": "number_match",
    #         "parameters": {
    #             "gold": ["Google 24oz Ring Bottle Blue"],
    #             "exclude": []
    #         }
    #     }
    # """
    
    # eval_metadata = """
    #     {
    #         "func": "number_match",
    #         "parameters": {
    #             "gold": ["17.5056918795%"],
    #             "percentage": true
    #         }
    #     }
    # """
    
    
    # eval_metadata = """
    #     {
    #         "func": "number_match",
    #         "parameters": {
    #             "gold": 
    #         }
    #     }
    # """