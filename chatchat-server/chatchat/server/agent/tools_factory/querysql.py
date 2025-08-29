from pydantic import Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import warnings

from chatchat.server.utils import get_tool_config, build_logger
from .tools_registry import BaseToolOutput, regist_tool

logger = build_logger()

# 抑制Doris特有的SQLAlchemy警告
warnings.filterwarnings("ignore", message=".*Unknown schema content.*", category=UserWarning)


@regist_tool(title="SQL查询工具", description="Tool for querying a SQL database.")
def query_sql_data(query: str = Field(description="Execute a SQL query against the database and get back the result.")):
    """
    Execute a SQL query against the database and get back the result..
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
    from langchain_community.utilities.sql_database import SQLDatabase

    try:
        db_endpoint = get_tool_config("text2sql")["sqlalchemy_connect_str"]
        
        # 针对Doris数据库优化连接配置
        if "mysql+pymysql" in db_endpoint and ":9030" in db_endpoint:
            # 这是Doris数据库连接，添加特定的连接参数
            if "connect_timeout" not in db_endpoint:
                separator = "&" if "?" in db_endpoint else "?"
                db_endpoint += f"{separator}connect_timeout=10"
        
        db = SQLDatabase.from_uri(db_endpoint)
        
        # 创建自定义的SQL查询工具以支持SQLAlchemy 2.0+和Doris
        class DorisCompatibleQueryTool(QuerySQLDataBaseTool):
            def _run(self, query: str) -> str:
                """执行SQL查询，兼容SQLAlchemy 2.0+和Doris数据库"""
                try:
                    # 使用text()包装SQL语句以兼容SQLAlchemy 2.0+
                    # 使用_engine属性而不是engine
                    with self.db._engine.connect() as connection:
                        result = connection.execute(text(query))
                        
                        # 获取列名
                        if result.returns_rows:
                            columns = list(result.keys())
                            rows = result.fetchall()
                            
                            if not rows:
                                return "查询成功，但没有返回数据。"
                            
                            # 格式化输出结果
                            formatted_result = []
                            formatted_result.append(" | ".join(columns))
                            formatted_result.append("-" * (len(" | ".join(columns))))
                            
                            for row in rows:
                                formatted_result.append(" | ".join(str(value) for value in row))
                            
                            return "\n".join(formatted_result)
                        else:
                            return f"SQL语句执行成功。受影响的行数: {result.rowcount}"
                            
                except SQLAlchemyError as e:
                    error_msg = str(e)
                    # 针对Doris特有错误提供更友好的提示
                    if "Unknown column" in error_msg:
                        return f"错误：列不存在。请检查列名是否正确。详细错误：{error_msg}"
                    elif "Unknown table" in error_msg:
                        return f"错误：表不存在。请检查表名是否正确。详细错误：{error_msg}"
                    elif "Syntax error" in error_msg:
                        return f"错误：SQL语法错误。请检查SQL语句语法。详细错误：{error_msg}"
                    else:
                        return f"数据库查询错误：{error_msg}"
                except Exception as e:
                    return f"执行查询时发生未知错误：{str(e)}"
        
        tool = DorisCompatibleQueryTool(db=db)
        logger.info(f"执行SQL查询: {query}")
        result = tool.run(query)
        logger.info(f"查询结果: {result}")
        return BaseToolOutput(result)
        
    except Exception as e:
        error_msg = f"初始化数据库连接失败：{str(e)}"
        logger.error(error_msg)
        return BaseToolOutput(error_msg)
