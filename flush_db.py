# all_stock写入数据库
import adata
import pandas as pd
from sqlalchemy import create_engine, text



def flush_database():
    # 调用接口获取所有股票代码
    all_df = adata.stock.info.all_code()
    
    # 筛选A股且非创业板的数据
    filtered_all_df = all_df[
        (all_df['stock_code'].str.startswith('00')) |  # 深交所主板和中小板
        (all_df['stock_code'].str.startswith('60')) |  # 上交所主板
        (all_df['stock_code'].str.startswith('68'))    # 科创板
    ].copy()
    
    # 写入SQLite数据库
    try:
        # 创建数据库连接
        engine = create_engine('sqlite:///SQLPub.db')
        
        # 将数据写入all_stock表
        filtered_all_df.to_sql(
            name='all_stock',           # 表名
            con=engine,                 # 数据库连接
            if_exists='replace',        # 如果表存在则替换（也可以用'append'追加）
            index=False,                # 不保存pandas的索引
            method='multi'              # 批量插入，提高效率
        )
        
        print(f"成功写入 {len(filtered_all_df)} 条记录到 all_stock 表")
        
    except Exception as e:
        print(f"写入数据库时出错: {e}")
    finally:
        engine.dispose()  # 关闭连接