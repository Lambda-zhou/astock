# 连接数据库
# 数据库连接测试
# !initctl status mysql
# !service mysql start
from sqlalchemy import create_engine
import streamlit as st


_ENGINE = None


def db_connect():
    """创建并复用数据库引擎，避免重复连接测试带来的额外耗时"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    # 创建数据库引擎
    db_user = st.secrets["db_user"]
    # 'all_stock'
    db_password = st.secrets["db_password"]
    # "SySZTdo7Ou5mmP0R"
    db_host = 'mysql2.sqlpub.com'  # 如果您的数据库在其他主机上，请更改为相应的主机名或IP
    db_port = '3307'
    db_name = 'all_stock'  # 替换为您的数据库名

    _ENGINE = create_engine(
        f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}',
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return _ENGINE
