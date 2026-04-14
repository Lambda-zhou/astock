# k线图函数
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


def draw_kline(df, stock_code):
    """绘制K线图，返回matplotlib图形对象"""
    # 数据预处理
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    
    # 解决中文显示问题
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制价格走势线
    ax.plot(df['trade_time'], df['price'], linewidth=2, color='#1f77b4', 
            marker='o', markersize=3, markerfacecolor='white', markeredgewidth=1.5)
    
    # 根据涨跌给线条着色
    for i in range(1, len(df)):
        if df.iloc[i]['change'] > 0:
            color = 'red'
        elif df.iloc[i]['change'] < 0:
            color = 'green'
        else:
            color = 'gray'
        
        ax.plot([df.iloc[i-1]['trade_time'], df.iloc[i]['trade_time']], 
                [df.iloc[i-1]['price'], df.iloc[i]['price']], 
                color=color, linewidth=2, alpha=0.8)
    
    # 找出最高点和最低点
    max_idx = df['price'].idxmax()
    min_idx = df['price'].idxmin()
    max_price = df.loc[max_idx, 'price']
    min_price = df.loc[min_idx, 'price']
    max_time = df.loc[max_idx, 'trade_time']
    min_time = df.loc[min_idx, 'trade_time']
    
    # 标注最高点
    ax.plot(max_time, max_price, marker='o', color='red', markersize=10, markerfacecolor='red', markeredgecolor='white', markeredgewidth=2)
    ax.annotate(f'HIGH: {max_price:.2f}', 
                xy=(max_time, max_price),
                xytext=(10, 20), textcoords='offset points',
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # 标注最低点
    ax.plot(min_time, min_price, marker='o', color='green', markersize=10, markerfacecolor='green', markeredgecolor='white', markeredgewidth=2)
    ax.annotate(f'LOW: {min_price:.2f}', 
                xy=(min_time, min_price),
                xytext=(10, -30), textcoords='offset points',
                fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # 添加价格标注（减少密度避免与最高最低点重叠）
    step = max(1, len(df) // 8)
    for i in range(0, len(df), step):
        if i != max_idx and i != min_idx:  # 避免与最高最低点标注重叠
            ax.annotate(f'{df.iloc[i]["price"]:.2f}', 
                        xy=(df.iloc[i]['trade_time'], df.iloc[i]['price']),
                        xytext=(5, 10), textcoords='offset points', 
                        fontsize=9, alpha=0.6)
    
    # 使用英文标签避免中文显示问题
    ax.set_title(f'Stock Code: {stock_code} - Price Chart', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Price (Yuan)', fontsize=12)
    ax.set_xlabel('Trade Time', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置时间轴格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(df)//8)))
    
    # 设置合适的Y轴范围，去掉不必要的空白
    price_min = df['price'].min()
    price_max = df['price'].max()
    price_range = price_max - price_min
    margin = price_range * 0.05  # 只保留5%的上下边距
    ax.set_ylim(price_min - margin, price_max + margin)
    
    # 旋转时间标签并减少间距
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 添加当前价格和涨跌信息到图例
    current_price = df.iloc[-1]['price']
    current_change = df.iloc[-1]['change']
    current_change_pct = df.iloc[-1]['change_pct']
    
    legend_text = f'Current: {current_price:.2f}\nChange: {current_change:+.2f} ({current_change_pct:+.2f}%)\nHigh: {max_price:.2f}\nLow: {min_price:.2f}'
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 调整布局，减少底部空白
    plt.tight_layout()
    
    # 进一步调整子图参数以减少空白
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.08, right=0.95)
    
    return fig