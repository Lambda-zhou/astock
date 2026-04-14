# 用数据库找股票名画k线图
# 推荐使用pandas方式
import adata
import pandas as pd
from k_line import draw_kline
from db_connect import db_connect


def database_search_name_draw(short_name):
    engine = db_connect()  # 获取数据库引擎
    query = "SELECT stock_code, short_name FROM all_stock WHERE short_name = %s"
    find_code = pd.read_sql_query(query, engine, params=(short_name,))
    if not find_code.empty:
        stock_code = find_code.iloc[0]['stock_code']
        print(f"股票代码是: {stock_code}")
        k_data = adata.stock.market.get_market_min(stock_code)
        fig = draw_kline(k_data, stock_code)
        # 如果需要显示图形，可以调用 plt.show()
        import matplotlib.pyplot as plt
        plt.show()
        return stock_code
    else:
        print(f"未找到股票: {short_name}")
        return None


def database_search_code_draw(stock_code):
    engine = db_connect()  # 获取数据库引擎
    query = "SELECT stock_code, short_name FROM all_stock WHERE stock_code = %s"
    find_code = pd.read_sql_query(query, engine, params=(stock_code,))
    if not find_code.empty:
        short_name = find_code.iloc[0]['short_name']
        print(f"股票名称是: {short_name}")
        k_data = adata.stock.market.get_market_min(stock_code)
        fig = draw_kline(k_data, stock_code)
        # 如果需要显示图形，可以调用 plt.show()
        import matplotlib.pyplot as plt
        plt.show()
        return short_name
    else:
        print(f"未找到股票: {stock_code}")
        return None


def database_fuzzy_search(keyword):
    """数据库模糊查询股票"""
    engine = db_connect()  # 获取数据库引擎
    query = "SELECT stock_code, short_name FROM all_stock WHERE short_name LIKE %s"
    result = pd.read_sql_query(query, engine, params=(f'%{keyword}%',))
    if not result.empty:
        print(f"找到 {len(result)} 只包含'{keyword}'的股票:")
        for _, row in result.iterrows():
            print(f"  {row['short_name']} ({row['stock_code']})")
        return result
    else:
        print(f"未找到包含'{keyword}'的股票")
        return None


def database_get_stock_name(stock_code):
    """从数据库获取股票名称（不绘图）"""
    engine = db_connect()
    query = "SELECT stock_code, short_name FROM all_stock WHERE stock_code = %s"
    find_code = pd.read_sql_query(query, engine, params=(stock_code,))
    if not find_code.empty:
        return find_code.iloc[0]['short_name']
    return None


def database_get_stock_code(short_name):
    """从数据库获取股票代码（不绘图）"""
    engine = db_connect()
    query = "SELECT stock_code, short_name FROM all_stock WHERE short_name = %s"
    find_code = pd.read_sql_query(query, engine, params=(short_name,))
    if not find_code.empty:
        return find_code.iloc[0]['stock_code']
    return None
