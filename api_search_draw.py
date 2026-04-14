# 用接口找股票并画图
import adata
from k_line import draw_kline


def api_search_code_draw(short_name):
    find_code = adata.stock.info.all_code()
    find_code = find_code[find_code['short_name'] == short_name]
    if not find_code.empty:
        stock_code = find_code['stock_code'].values[0]
        print(f"股票代码是: {stock_code}")
    else:
        print("未找到相关股票")
        return None
    
    k_data = adata.stock.market.get_market_min(stock_code)
    fig = draw_kline(k_data, stock_code)
    # 如果需要显示图形，可以调用 plt.show()
    import matplotlib.pyplot as plt
    plt.show()
    return fig


def api_search_name_draw(stock_code):
    find_code = adata.stock.info.all_code()
    find_code = find_code[find_code['stock_code'] == stock_code]
    if not find_code.empty:
        short_name = find_code['short_name'].values[0]
        print(f"股票名字是: {short_name}")
    else:
        print("未找到相关股票")
        return None
    
    k_data = adata.stock.market.get_market_min(stock_code)
    fig = draw_kline(k_data, stock_code)
    # 如果需要显示图形，可以调用 plt.show()
    import matplotlib.pyplot as plt
    plt.show()
    return fig