from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts.prompt import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from pydantic import Field
import warnings

from chatchat.server.utils import get_tool_config

from .tools_registry import BaseToolOutput, regist_tool

import pymysql
pymysql.install_as_MySQLdb()

# 抑制Doris特有的SQLAlchemy警告
warnings.filterwarnings("ignore", message=".*Unknown schema content.*", category=UserWarning)

READ_ONLY_PROMPT_TEMPLATE = """You are a Doris/MySQL expert. The database is currently in read-only mode. 
Given an input question, determine if the related SQL can be executed in read-only mode.
If the SQL can be executed normally, return Answer:'SQL can be executed normally'.
If the SQL cannot be executed normally, return Answer: 'SQL cannot be executed normally'.
Use the following format:

Answer: Final answer here

Question: {query}
"""


class DorisCompatibleSQLDatabase(SQLDatabase):
    """Doris数据库兼容的SQLDatabase类"""
    
    def __init__(self, engine, schema=None, metadata=None, ignore_tables=None, include_tables=None, 
                 sample_rows_in_table_info=3, indexes_in_table_info=False, custom_table_info=None, 
                 view_support=False, max_string_length=300):
        super().__init__(engine, schema, metadata, ignore_tables, include_tables, 
                        sample_rows_in_table_info, indexes_in_table_info, custom_table_info, 
                        view_support, max_string_length)
    
    def run(self, command: str, fetch: str = "all") -> str:
        """执行SQL命令，兼容SQLAlchemy 2.0+和Doris数据库"""
        try:
            with self._engine.connect() as connection:
                # 使用text()包装SQL命令以兼容SQLAlchemy 2.0+
                if self._schema is not None:
                    connection.exec_driver_sql(f"USE {self._schema}")
                
                result = connection.execute(text(command))
                
                if result.returns_rows:
                    if fetch == "all":
                        return str(result.fetchall())
                    elif fetch == "one":
                        return str(result.fetchone())
                    else:
                        return str(result.fetchmany(fetch))
                else:
                    return f"Query executed successfully. Rows affected: {result.rowcount}"
                    
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
    
    def get_table_info(self, table_names=None):
        """获取表信息，针对Doris数据库优化"""
        try:
            return super().get_table_info(table_names)
        except Exception as e:
            # 如果标准方法失败，尝试Doris特有的方法
            if table_names is None:
                table_names = self.get_usable_table_names()
            
            tables_info = []
            for table_name in table_names:
                try:
                    with self._engine.connect() as connection:
                        # 使用DESCRIBE获取表结构（Doris兼容MySQL的DESCRIBE语法）
                        desc_result = connection.execute(text(f"DESCRIBE {table_name}"))
                        columns_info = desc_result.fetchall()
                        
                        table_info = f"Table: {table_name}\n"
                        table_info += "Columns:\n"
                        for col in columns_info:
                            table_info += f"  - {col[0]} ({col[1]})\n"
                        
                        # 尝试获取样本数据
                        if self._sample_rows_in_table_info > 0:
                            try:
                                sample_result = connection.execute(
                                    text(f"SELECT * FROM {table_name} LIMIT {self._sample_rows_in_table_info}")
                                )
                                sample_rows = sample_result.fetchall()
                                if sample_rows:
                                    table_info += f"\nSample rows:\n"
                                    for row in sample_rows:
                                        table_info += f"  {row}\n"
                            except:
                                table_info += "\nSample rows: Unable to fetch sample data\n"
                        
                        tables_info.append(table_info)
                except Exception as table_error:
                    tables_info.append(f"Table: {table_name}\nError: {str(table_error)}")
            
            return "\n\n".join(tables_info)


# 定义一个拦截器函数来检查SQL语句，以支持read-only,可修改下面的write_operations，以匹配你使用的数据库写操作关键字
def intercept_sql(conn, cursor, statement, parameters, context, executemany):
    # List of SQL keywords that indicate a write operation
    write_operations = (
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "truncate",
        "rename",
    )
    # Check if the statement starts with any of the write operation keywords
    if any(statement.strip().lower().startswith(op) for op in write_operations):
        raise OperationalError(
            "Database is read-only. Write operations are not allowed.",
            params=None,
            orig=None,
        )


def query_database(query: str, config: dict):
    model_name= config["model_name"]
    top_k = config["top_k"]
    return_intermediate_steps = config["return_intermediate_steps"]
    sqlalchemy_connect_str = config["sqlalchemy_connect_str"]
    read_only = config["read_only"]
    
    # 针对Doris数据库优化连接配置
    if "mysql+pymysql" in sqlalchemy_connect_str and ":9030" in sqlalchemy_connect_str:
        # 这是Doris数据库连接，添加特定的连接参数
        if "connect_timeout" not in sqlalchemy_connect_str:
            separator = "&" if "?" in sqlalchemy_connect_str else "?"
            sqlalchemy_connect_str += f"{separator}connect_timeout=10"
    
    # 使用Doris兼容的SQLDatabase类
    try:
        if "mysql+pymysql" in sqlalchemy_connect_str and ":9030" in sqlalchemy_connect_str:
            # Doris数据库，使用我们的兼容类
            from sqlalchemy import create_engine
            engine = create_engine(sqlalchemy_connect_str)
            db = DorisCompatibleSQLDatabase(engine)
        else:
            # 其他数据库，使用标准的SQLDatabase
            db = SQLDatabase.from_uri(sqlalchemy_connect_str)
    except Exception as e:
        return f"数据库连接失败：{str(e)}"

    from chatchat.server.utils import get_ChatOpenAI

    llm = get_ChatOpenAI(
        model_name=model_name,
        temperature=0.1,
        streaming=True,
        local_wrap=True,
        verbose=True,
    )
    table_names = config["table_names"]
    table_comments = config["table_comments"]
    result = None

    # 如果发现大模型判断用什么表出现问题，尝试给langchain提供额外的表说明，辅助大模型更好的判断应该使用哪些表，尤其是SQLDatabaseSequentialChain模式下,是根据表名做的预测，很容易误判
    # 由于langchain固定了输入参数，所以只能通过query传递额外的表说明
    if table_comments:
        TABLE_COMMNET_PROMPT = (
            "\n\nI will provide some special notes for a few tables:\n\n"
        )
        table_comments_str = "\n".join([f"{k}:{v}" for k, v in table_comments.items()])
        query = query + TABLE_COMMNET_PROMPT + table_comments_str + "\n\n"

    if read_only:
        # 在read_only下，先让大模型判断只读模式是否能满足需求，避免后续执行过程报错，返回友好提示。
        READ_ONLY_PROMPT = PromptTemplate(
            input_variables=["query"],
            template=READ_ONLY_PROMPT_TEMPLATE,
        )
        read_only_chain = LLMChain(
            prompt=READ_ONLY_PROMPT,
            llm=llm,
        )
        read_only_result = read_only_chain.invoke(query)
        if "SQL cannot be executed normally" in read_only_result["text"]:
            return "当前数据库为只读状态，无法满足您的需求！"

        # 当然大模型不能保证完全判断准确，为防止大模型判断有误，再从拦截器层面拒绝写操作
        event.listen(db._engine, "before_cursor_execute", intercept_sql)

    # 如果不指定table_names，优先走SQLDatabaseSequentialChain，这个链会先预测需要哪些表，然后再将相关表输入SQLDatabaseChain
    # 这是因为如果不指定table_names，直接走SQLDatabaseChain，Langchain会将全量表结构传递给大模型，可能会因token太长从而引发错误，也浪费资源
    # 如果指定了table_names，直接走SQLDatabaseChain，将特定表结构传递给大模型进行判断
    if len(table_names) > 0:
        db_chain = SQLDatabaseChain.from_llm(
            llm,
            db,
            verbose=True,
            top_k=top_k,
            return_intermediate_steps=return_intermediate_steps,
        )
        result = db_chain.invoke({"query": query, "table_names_to_use": table_names})
    else:
        # 先预测会使用哪些表，然后再将问题和预测的表给大模型
        db_chain = SQLDatabaseSequentialChain.from_llm(
            llm,
            db,
            verbose=True,
            top_k=top_k,
            return_intermediate_steps=return_intermediate_steps,
        )
        result = db_chain.invoke(query)

    context = f"""查询结果:{result['result']}\n\n"""

    intermediate_steps = result["intermediate_steps"]
    # 如果存在intermediate_steps，且这个数组的长度大于2，则保留最后两个元素，因为前面几个步骤存在示例数据，容易引起误解
    if intermediate_steps:
        if len(intermediate_steps) > 2:
            sql_detail = intermediate_steps[-2:-1][0]["input"]
            # sql_detail截取从SQLQuery到Answer:之间的内容
            sql_detail = sql_detail[
                sql_detail.find("SQLQuery:") + 9 : sql_detail.find("Answer:")
            ]
            context = context + "执行的sql:'" + sql_detail + "'\n\n"
    return context


@regist_tool(title="数据库对话")
def text2sql(
    query: str = Field(
        description="No need for SQL statements,just input the natural language that you want to chat with database"
    ),
):
    """Use this tool to chat with  database,Input natural language, then it will convert it into SQL and execute it in the database, then return the execution result."""
    tool_config = get_tool_config("text2sql")
    return BaseToolOutput(query_database(query=query, config=tool_config))
