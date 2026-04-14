# 同花顺热榜函数
import adata
from decimal import Decimal
import pandas as pd
from k_line import draw_kline

def code_draw(stock_code):
    k_data = adata.stock.market.get_market_min(stock_code)
    fig = draw_kline(k_data, stock_code)
    # 如果需要显示图形，可以调用 plt.show()
    import matplotlib.pyplot as plt
    plt.show()
    return fig


# 需要先运行main()函数获取result，然后传入concept_count函数
def concept_count(result):
    # 展开概念标签和股票名称
    expanded = result[['concept_tag', 'short_name']].copy()
    expanded = expanded.dropna(subset=['concept_tag'])
    
    # 分割概念并展开
    expanded_list = []
    for _, row in expanded.iterrows():
        concepts = row['concept_tag'].split(';')
        for concept in concepts:
            concept = concept.strip()
            if concept:
                expanded_list.append({'concept': concept, 'short_name': row['short_name']})
    
    expanded_df = pd.DataFrame(expanded_list)
    
    # 统计概念出现次数和对应股票
    concept_counts = expanded_df.groupby('concept').agg({
        'short_name': ['count', lambda x: '、'.join(x.unique())]
    }).reset_index()
    
    concept_counts.columns = ['concept', 'count', 'stocks']
    concept_counts = concept_counts.sort_values('count', ascending=False)
    
    return concept_counts


def get_merged_stock_data():
    try:
        df = adata.sentiment.hot.hot_rank_100_ths().loc[:, ['stock_code','pop_tag','concept_tag','change_pct']]
        filtered_df = df[
            (df['change_pct'] > 0) & 
            (~df['stock_code'].str.startswith(('300')))]
        # 提取股票代码列表
        stock_code_list = filtered_df['stock_code'].tolist()
        print(f"提取到{len(stock_code_list)}只股票代码")
        # 获取市场数据
        df1 = adata.stock.market.list_market_current(code_list=stock_code_list).loc[:, ['stock_code','price','short_name','volume']] #code list pass the stock code
        df1['price'] = df1['price'].apply(Decimal)
        filtered_df1 = df1[(df1['price'] <= Decimal('20'))]
        # 合并数据
        merged_df = pd.merge(filtered_df, filtered_df1, on='stock_code', how='inner')
        return merged_df
    except Exception as e:
        print(f"处理数据失败: {e}")
        return None
    

def main():
    result = get_merged_stock_data()
    columns_order = ['stock_code', 'short_name', 'price', 'change_pct', 'pop_tag', 'concept_tag', 'volume']
    result =  result[columns_order]
    return result