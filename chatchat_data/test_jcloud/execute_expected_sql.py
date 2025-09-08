#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行test_jcloud_cases_extended.json中的SQL并保存结果用于比对
逐条执行test_jcloud_cases_extended.json中的sql，将结果保存为CSV格式
"""

import json
import pymysql
import socket
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('expected_sql_execution.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExpectedSQLExecutor:
    def __init__(self, config: Dict[str, Any]):
        """初始化执行器"""
        self.config = config
        self.connection = None
        self.cursor = None
        self.results = []
        # 配置Decimal类型处理方式：'string'保持精度，'float'转换为浮点数
        self.decimal_handling = config.get('decimal_handling', 'string')
        
    def connect_to_database(self) -> bool:
        """连接到数据库"""
        try:
            # 测试网络连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((self.config['host'], self.config['port']))
            sock.close()
            
            if result != 0:
                logger.error(f"网络连接失败: {self.config['host']}:{self.config['port']}")
                return False
            
            # 连接数据库
            self.connection = pymysql.connect(**self.config)
            self.cursor = self.connection.cursor()
            
            # 切换到指定数据库
            self.cursor.execute(f"USE {self.config['database']}")
            logger.info("数据库连接成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def load_test_cases(self, file_path: str) -> List[Dict[str, Any]]:
        """加载测试用例"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 个测试用例")
            return data
        except Exception as e:
            logger.error(f"加载测试用例失败: {e}")
            return []
    
    def execute_sql(self, case_id: str, sql: str) -> Dict[str, Any]:
        """执行单个SQL查询"""
        result = {
            'case_id': case_id,
            'expected_sql': sql,
            'execution_time': None,
            'row_count': 0,
            'columns': [],
            'data': [],
            'error': None,
            'success': False
        }
        
        try:
            start_time = datetime.now()
            
            # 执行SQL
            self.cursor.execute(sql)
            
            # 获取列信息
            if self.cursor.description:
                result['columns'] = [desc[0] for desc in self.cursor.description]
            
            # 获取结果
            if sql.strip().upper().startswith('SELECT'):
                data = self.cursor.fetchall()
                # 转换Decimal类型为字符串类型，确保JSON序列化兼容且保持精度
                converted_data = []
                for row in data:
                    converted_row = []
                    for item in row:
                        if isinstance(item, Decimal):
                            if self.decimal_handling == 'float':
                                converted_row.append(float(item))  # 转换为浮点数
                            else:
                                converted_row.append(str(item))   # 转换为字符串保持精度
                        else:
                            converted_row.append(item)
                    converted_data.append(converted_row)
                
                result['data'] = converted_data
                result['row_count'] = len(converted_data)
                result['success'] = True
                
                # 转换为DataFrame格式，便于后续比对
                if converted_data and result['columns']:
                    df = pd.DataFrame(converted_data, columns=result['columns'])
                    result['dataframe'] = df
                
            else:
                result['row_count'] = self.cursor.rowcount
                result['success'] = True
            
            end_time = datetime.now()
            result['execution_time'] = (end_time - start_time).total_seconds()
            
            logger.info(f"用例 {case_id} 执行成功: {result['row_count']} 行, 耗时 {result['execution_time']:.3f}秒")
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            logger.error(f"用例 {case_id} 执行失败: {e}")
        
        return result
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_dir: str):
        """将结果保存为CSV文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存汇总信息
        summary_data = []
        for result in results:
            summary = {
                'case_id': result['case_id'],
                'expected_sql': result['expected_sql'],
                'success': result['success'],
                'execution_time': result['execution_time'],
                'row_count': result['row_count'],
                'error': result['error'] or ''
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'expected_sql_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        logger.info(f"汇总信息已保存到: {summary_file}")
        
        # 保存每个用例的详细结果
        for result in results:
            case_file = os.path.join(output_dir, f"case_{result['case_id']}_result.csv")
            if result['success'] and 'dataframe' in result and not result['dataframe'].empty:
                result['dataframe'].to_csv(case_file, index=False, encoding='utf-8-sig')
                logger.info(f"用例 {result['case_id']} 结果已保存到: {case_file}")
            else:
                # 无结果或结果为空时，也保存一个占位CSV，内容为'none'
                pd.DataFrame([{'result': 'none'}]).to_csv(case_file, index=False, encoding='utf-8-sig')
                logger.info(f"用例 {result['case_id']} 无结果，已保存占位文件: {case_file}")
        
        # 保存所有数据到一个合并的CSV文件（用于比对）
        all_data = []
        for result in results:
            if result['success'] and 'dataframe' in result and not result['dataframe'].empty:
                df = result['dataframe'].copy()
                df['case_id'] = result['case_id']
                df['expected_sql'] = result['expected_sql']
                all_data.append(df)
        
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            merged_file = os.path.join(output_dir, 'all_expected_sql_results.csv')
            merged_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
            logger.info(f"合并结果已保存到: {merged_file}")
    

    def run(self, test_cases_file: str, output_dir: str):
        """运行执行流程"""
        logger.info("开始执行SQL查询...")
        
        # 1. 连接数据库
        if not self.connect_to_database():
            logger.error("数据库连接失败，退出执行")
            return
        
        # 2. 加载测试用例
        test_cases = self.load_test_cases(test_cases_file)
        if not test_cases:
            logger.error("加载测试用例失败，退出执行")
            return
        
        # 3. 执行所有SQL
        logger.info(f"开始执行 {len(test_cases)} 个测试用例...")
        for case in test_cases:
            case_id = case.get('id', 'Unknown')
            # if(case_id != 'Q023'):
            #     continue
            sql = case.get('expected_sql', '')
            
            if sql:
                result = self.execute_sql(case_id, sql)
                self.results.append(result)
            else:
                logger.warning(f"用例 {case_id} 没有SQL语句")
        
        # 4. 保存结果
        logger.info("保存执行结果...")
        self.save_results_to_csv(self.results, output_dir)
        
        # 5. 输出统计信息
        success_count = sum(1 for r in self.results if r['success'])
        total_count = len(self.results)
        
        logger.info(f"执行完成: 成功 {success_count}/{total_count}")
        logger.info(f"结果已保存到目录: {output_dir}")
        
        # 6. 关闭连接
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("数据库连接已关闭")

def main():
    """主函数"""
    # 数据库配置 - 更新为新的MySQL数据库信息
    db_config = {
        'host': '10.119.1.128',
        'port': 3306,
        'user': 'root',
        'password': 'jcloud_sjfx_DB@2021',
        'database': 'jcloud_resource',
        'charset': 'utf8mb4',
        'connect_timeout': 1500,
        'read_timeout': 1500,
        'write_timeout': 1500
    }
    
    # 输入和输出配置
    test_cases_file = 'test_jcloud_cases_extended.json'
    output_dir = 'jcloud_sql_results_extended'
    
    # 创建执行器并运行
    executor = ExpectedSQLExecutor(db_config)
    executor.run(test_cases_file, output_dir)

if __name__ == "__main__":
    main()
