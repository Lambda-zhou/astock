# 股票分析系统

基于Python和Streamlit的股票数据分析和可视化系统，支持股票查询、K线图绘制、龙虎榜查询等功能。

## 功能特性

### 📊 股票查询与可视化
- 支持股票代码和股票名称双向查询
- 实时K线图绘制
- 多种数据源支持（API + 数据库）
- 自动保存K线图到本地

### 🏆 龙虎榜分析
- 查询个股是否在龙虎榜
- 获取详细龙虎榜数据
- 支持历史数据查询

### 🔥 热门股票追踪
- 同花顺热榜数据获取
- 热门股票实时监控
- 自动化数据更新

### 💾 数据管理
- MySQL数据库集成
- 自动化数据更新
- 数据清洗和维护

## 项目结构

```
├── lhb_streamlit.py        # Streamlit主应用界面
├── streamlit_explan.py     # 项目介绍和使用说明
├── api_search_draw.py      # API股票查询与绘图
├── db_search_draw.py       # 数据库股票查询与绘图
├── find_lhs.py            # 龙虎榜查询功能
├── ths_hot.py             # 同花顺热榜数据
├── trade_day.py           # 交易日管理模块
├── db_connect.py          # 数据库连接配置
├── flush_db.py            # 数据库更新脚本
├── k_line.py              # K线图绘制工具
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明文档
```

## 安装与运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置数据库
在 `db_connect.py` 中配置数据库连接信息：
```python
db_user = 'your_username'
db_password = 'your_password'
db_host = 'your_host'
db_port = 'your_port'
db_name = 'your_database'
```

### 3. 启动应用
```bash
streamlit run lhb_streamlit.py
```

应用将在浏览器中自动打开，默认地址为 `http://localhost:8501`

## 使用说明

### Web界面使用
1. 打开浏览器访问应用
2. 在股票查询框中输入股票代码或名称
3. 选择数据源（API或数据库）
4. 点击查询按钮获取数据
5. 查看K线图和详细信息

### 命令行使用
```python
from db_search_draw import database_search_draw

# 通过股票名称查询
database_search_draw("中信海直")
```

### 查询龙虎榜数据
```python
from find_lhs import search_in_lh, find_lhb

# 查询是否在龙虎榜
result = search_in_lh("000099")

# 获取龙虎榜详细数据
lhb_data = find_lhb("000099")
```

### 获取热门股票
```python
from ths_hot import main

# 获取同花顺热榜
main()
```

## 技术特点

- **双数据源支持**: API和数据库双重保障
- **智能缓存**: 使用Streamlit缓存提升性能
- **模块化设计**: 独立模块，易于维护
- **响应式界面**: 适配多种设备屏幕

## 数据源

- **API数据源**: 通过adata库获取实时股票数据
- **数据库**: MySQL存储股票基础信息
- **同花顺**: 热门股票榜单数据
- **交易日数据**: 自动获取交易日信息

## 更新日志

### v2.0 (最新版本)
- ✨ 新增Streamlit Web界面
- ✨ 新增项目介绍和使用说明模块
- ✨ 新增交易日管理功能
- ✨ 优化用户交互体验

### v1.0
- 🎉 初始版本发布
- 📊 基础股票查询功能
- 🏆 龙虎榜查询功能
- 🔥 同花顺热榜功能
- 💾 数据库管理功能
