# 龙虎榜函数
import adata
import warnings
import pandas as pd
from datetime import datetime
from trade_day import get_last_trading_day, is_trading_day
report_date = get_last_trading_day()  # 使用最近的交易日


def find_lhb(stock_code):
    stock_code = stock_code
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
    need_columns = ['a_net_amount', 'a_buy_amount', 'a_sell_amount', 'operate_name']
    if 'lh' not in find_lhb.__dict__:
        find_lhb.lh = adata.sentiment.hot.list_a_list_daily(report_date)
    
    # 检查是否为股票代码
    if stock_code in find_lhb.lh['stock_code'].values:
        lhb = adata.sentiment.hot.get_a_list_info(stock_code, report_date)[need_columns]
        return lhb
    # 检查是否为股票名称
    elif 'short_name' in find_lhb.lh.columns and stock_code in find_lhb.lh['short_name'].values:
        # 通过股票名称找到对应的股票代码
        stock_row = find_lhb.lh[find_lhb.lh['short_name'] == stock_code]
        if not stock_row.empty:
            actual_stock_code = stock_row['stock_code'].iloc[0]
            lhb = adata.sentiment.hot.get_a_list_info(actual_stock_code, report_date)[need_columns]
            return lhb
    else:
        print(f"{stock_code}没有上龙虎榜！")
        return None


def search_in_lh(stock_code):
    lh = adata.sentiment.hot.list_a_list_daily(report_date)
    
    # 检查是否为股票代码
    if stock_code in lh['stock_code'].values:
        print(f"找到了{stock_code},可以find_lhb({stock_code})查询")
        # 获取该股票的所有数据
        stock_found = lh[lh['stock_code'] == stock_code]
        return stock_found
    # 检查是否为股票名称
    elif 'short_name' in lh.columns and stock_code in lh['short_name'].values:
        print(f"通过股票名称找到了{stock_code},可以find_lhb({stock_code})查询")
        # 获取该股票的所有数据
        stock_found = lh[lh['short_name'] == stock_code]
        return stock_found
    else:
        print(f"未找到{stock_code}")
        return None


def stock_risk(stock_code):
    risk = adata.sentiment.mine.mine_clearance_tdx(stock_code)
    return risk
