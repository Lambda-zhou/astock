import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import adata
import time
import json
from datetime import datetime
import pickle

from ai_client import AIClientError, chat_completion

# 导入项目介绍模块
try:
    import streamlit_explan as explan
    IMPORT_EXPLAN = True
except ImportError:
    IMPORT_EXPLAN = False
    st.warning("项目介绍模块导入失败")

# 忽略警告信息
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# 创建必要的文件夹
for folder in ['image', 'data_cache', 'history']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 设置页面配置
st.set_page_config(
    page_title="股票分析系统 Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 数据持久化类
def apply_design_theme():
    """应用统一主题样式。"""
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f5f5f7;
            --app-surface: #ffffff;
            --app-text: #1d1d1f;
            --app-text-muted: rgba(0, 0, 0, 0.68);
            --app-accent: #0071e3;
            --app-border: rgba(0, 0, 0, 0.08);
            --app-shadow: rgba(0, 0, 0, 0.12) 0px 12px 40px 0px;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--app-text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.82);
            border-right: 1px solid var(--app-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            backdrop-filter: saturate(180%) blur(18px);
        }

        h1, h2, h3 {
            color: var(--app-text);
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetric"],
        div[data-testid="stExpander"] {
            background: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            box-shadow: var(--app-shadow);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 999px;
        }

        .stButton > button[kind="primary"] {
            background: var(--app-accent);
        }

        .ai-card {
            background: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: var(--app-shadow);
        }

        .ai-role {
            font-size: 0.84rem;
            font-weight: 600;
            color: var(--app-accent);
            margin-bottom: 0.35rem;
        }

        .ai-meta {
            color: var(--app-text-muted);
            font-size: 0.82rem;
            margin-top: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_ai_context():
    """构建 AI 对话上下文。"""
    context_parts = []

    if st.session_state.get('ai_include_stock') and st.session_state.get('query_result') is not None:
        stock_df = st.session_state['query_result']
        stock_name = st.session_state.get('query_stock_name') or '未知'
        stock_code = st.session_state.get('query_stock_code') or '未知'
        stock_source = st.session_state.get('query_source') or '未知'
        context_parts.append(
            "当前股票查询结果\n"
            f"股票名称: {stock_name}\n"
            f"股票代码: {stock_code}\n"
            f"数据源: {stock_source}\n"
            f"最近10条数据:\n{stock_df.tail(10).to_csv(index=False)}"
        )

    if st.session_state.get('ai_include_hot') and st.session_state.get('hot_data') is not None:
        hot_data = st.session_state['hot_data']
        if isinstance(hot_data, pd.DataFrame):
            context_parts.append(f"当前热榜数据预览\n{hot_data.head(20).to_csv(index=False)}")
        else:
            context_parts.append(f"当前热榜数据预览\n{hot_data}")

    return "\n\n".join(context_parts).strip()


def build_ai_messages():
    """构建发送给模型的消息。"""
    messages = []
    system_prompt = st.session_state.get('ai_system_prompt', '').strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    context_text = build_ai_context()
    if context_text:
        messages.append(
            {
                "role": "system",
                "content": f"以下是用户当前在系统中的可用上下文，可按需参考，不要编造未提供的数据。\n\n{context_text}",
            }
        )

    for item in st.session_state.get('ai_chat_history', []):
        messages.append({"role": item["role"], "content": item["content"]})

    return messages


def render_ai_message(role, content, timestamp=None):
    """渲染聊天消息。"""
    role_map = {"user": "用户", "assistant": "AI", "system": "系统"}
    safe_content = content.replace("\n", "<br>")
    time_html = f"<div class='ai-meta'>{timestamp}</div>" if timestamp else ""
    st.markdown(
        f"""
        <div class="ai-card">
            <div class="ai-role">{role_map.get(role, role)}</div>
            <div>{safe_content}</div>
            {time_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


class DataPersistence:
    def __init__(self):
        self.cache_dir = "data_cache"
        self.history_dir = "history"
        self.history_file = os.path.join(self.history_dir, "operation_history.json")
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保目录存在"""
        for directory in [self.cache_dir, self.history_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def save_operation_history(self, operation_type, data, metadata=None):
        """保存操作历史"""
        try:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation_type": operation_type,
                "metadata": metadata or {},
                "data_file": None
            }
            
            # 保存数据到单独文件
            if data is not None:
                timestamp = int(time.time())
                data_filename = f"{operation_type}_{timestamp}.pkl"
                data_filepath = os.path.join(self.cache_dir, data_filename)
                
                with open(data_filepath, 'wb') as f:
                    pickle.dump(data, f)
                
                history_entry["data_file"] = data_filename
            
            # 读取现有历史
            history = self.load_operation_history()
            history.append(history_entry)
            
            # 保持最近100条记录
            if len(history) > 100:
                # 删除旧的数据文件
                old_entry = history[0]
                if old_entry.get("data_file"):
                    old_file_path = os.path.join(self.cache_dir, old_entry["data_file"])
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                history = history[-100:]
            
            # 保存历史记录
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            st.error(f"保存操作历史失败: {str(e)}")
            return False
    
    def load_operation_history(self):
        """加载操作历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"加载操作历史失败: {str(e)}")
            return []
    
    def load_operation_data(self, data_filename):
        """加载操作数据"""
        try:
            data_filepath = os.path.join(self.cache_dir, data_filename)
            if os.path.exists(data_filepath):
                with open(data_filepath, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            st.error(f"加载操作数据失败: {str(e)}")
            return None
    
    def clear_history(self):
        """清空历史记录"""
        try:
            # 删除所有缓存文件
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # 清空历史记录文件
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            
            return True
        except Exception as e:
            st.error(f"清空历史记录失败: {str(e)}")
            return False

# 初始化数据持久化
@st.cache_resource
def get_data_persistence():
    return DataPersistence()

data_persistence = get_data_persistence()

# 安全导入模块
def safe_import():
    """安全导入模块"""
    modules = {}
    import_status = {}
    
    module_configs = [
        ('api_search', ['api_search_draw'], ['api_search_code_draw', 'api_search_name_draw']),
        ('db_search', ['db_search_draw'], ['database_search_name_draw', 'database_search_code_draw']),
        ('lhb', ['find_lhs'], ['search_in_lh', 'find_lhb']),
        ('ths_hot', ['ths_hot'], ['code_draw', 'concept_count']),
        ('db_connect', ['db_connect'], ['db_connect']),
        ('flush_db', ['flush_db'], ['flush_database']),
        ('k_line', ['k_line'], ['draw_kline'])
    ]
    
    for module_name, import_paths, function_names in module_configs:
        try:
            module_dict = {}
            for path in import_paths:
                module = __import__(path)
                for func_name in function_names:
                    if hasattr(module, func_name):
                        module_dict[func_name] = getattr(module, func_name)
                # 对于ths_hot模块，特殊处理main函数
                if module_name == 'ths_hot' and hasattr(module, 'main'):
                    module_dict['main'] = getattr(module, 'main')
            modules[module_name] = module_dict
            import_status[module_name] = True
        except ImportError as e:
            st.warning(f"{module_name}模块导入失败: {e}")
            import_status[module_name] = False
    
    return modules, import_status

MODULES, IMPORT_STATUS = safe_import()

# 缓存函数
@st.cache_data(ttl=1800)
def get_all_stock_codes():
    """获取所有股票代码和名称"""
    try:
        return adata.stock.info.all_code()
    except Exception as e:
        st.error(f"获取股票代码失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=180)
def get_stock_data_cached(stock_code):
    """缓存股票数据获取"""
    try:
        return adata.stock.market.get_market_min(stock_code)
    except Exception as e:
        return None

def get_stock_name_by_code(stock_code):
    """通过股票代码获取股票名称"""
    try:
        all_codes = get_all_stock_codes()
        if not all_codes.empty:
            result = all_codes[all_codes['stock_code'] == stock_code]
            if not result.empty:
                return result['short_name'].values[0]
    except:
        pass
    return "未知"

def get_stock_code_by_name(short_name):
    """通过股票名称获取股票代码"""
    try:
        all_codes = get_all_stock_codes()
        if not all_codes.empty:
            result = all_codes[all_codes['short_name'] == short_name]
            if not result.empty:
                return result['stock_code'].values[0]
    except:
        pass
    return None

def save_kline_image(df, stock_code, stock_name=""):
    """保存K线图"""
    try:
        fig = MODULES['k_line']['draw_kline'](df, stock_code)
        timestamp = int(time.time())
        filename = f"image/{stock_code}_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename
    except Exception as e:
        st.error(f"保存K线图失败: {str(e)}")
        return None

def save_kline_image_for_history(df, stock_code, stock_name=""):
    """为历史记录保存K线图（使用stock_code命名）"""
    try:
        fig = MODULES['k_line']['draw_kline'](df, stock_code)
        filename = f"image/{stock_code}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename
    except Exception as e:
        st.error(f"保存K线图失败: {str(e)}")
        return None

def get_latest_kline_image(stock_code):
    """获取最新的K线图"""
    try:
        image_files = [f for f in os.listdir('image') if f.startswith(f"{stock_code}_")]
        if image_files:
            image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            return os.path.join('image', image_files[0])
        return None
    except Exception as e:
        return None

def get_stock_name_from_db(stock_code):
    """从数据库获取股票名称"""
    if not IMPORT_STATUS.get('db_search', False):
        return None
    
    try:
        # 使用新增的不绘图函数
        from db_search_draw import database_get_stock_name
        result = database_get_stock_name(stock_code)
        return result if result else None
    except Exception as e:
        st.error(f"数据库查询股票名称失败: {str(e)}")
        return None

def get_stock_code_from_db(short_name):
    """从数据库获取股票代码"""
    if not IMPORT_STATUS.get('db_search', False):
        return None
    
    try:
        # 使用新增的不绘图函数
        from db_search_draw import database_get_stock_code
        result = database_get_stock_code(short_name)
        return result if result else None
    except Exception as e:
        st.error(f"数据库查询股票代码失败: {str(e)}")
        return None

def fuzzy_search_stocks_from_db(keyword):
    """从数据库模糊查询股票"""
    if not IMPORT_STATUS.get('db_connect', False):
        return None
    
    try:
        from db_search_draw import database_fuzzy_search
        result = database_fuzzy_search(keyword)
        return result if result is not None else None
    except Exception as e:
        st.error(f"数据库模糊查询失败: {str(e)}")
        return None

def query_stock_data(stock_code, stock_name, data_source):
    """查询股票数据"""
    try:
        # 获取股票名称
        final_stock_name = stock_name
        if not stock_name or stock_name == "未知":
            if data_source == "数据库查询":
                db_stock_name = get_stock_name_from_db(stock_code)
                if db_stock_name:
                    final_stock_name = db_stock_name
            else:
                final_stock_name = get_stock_name_by_code(stock_code)
        
        # 获取股票数据
        k_data = None
        if data_source == "API查询":
            k_data = get_stock_data_cached(stock_code)
        elif data_source == "数据库查询":
            if IMPORT_STATUS.get('db_search', False):
                # 先查询数据库获取股票名称，再获取数据
                db_stock_name = get_stock_name_from_db(stock_code)
                if db_stock_name:
                    final_stock_name = db_stock_name
                    k_data = get_stock_data_cached(stock_code)
            else:
                st.error("数据库查询模块未正确加载")
                return None, None, None
        
        return k_data, final_stock_name, stock_code
    except Exception as e:
        st.error(f"查询股票数据失败: {str(e)}")
        return None, None, None

def display_stock_info(k_data, stock_code, stock_name, data_source):
    """股票信息"""
    if k_data is not None and not k_data.empty:
        current_price = k_data.iloc[-1]['price']
        current_change = k_data.iloc[-1]['change']
        current_change_pct = k_data.iloc[-1]['change_pct']
        
        st.info(f"""
        **股票信息:**
        - 股票代码: {stock_code}
        - 股票名称: {stock_name}
        - 当前价格: {current_price:.2f}
        - 涨跌额: {current_change:+.2f}
        - 涨跌幅: {current_change_pct:+.2f}%
        - 数据源: {data_source}
        """)

def handle_stock_query(stock_code, short_name):
    """处理股票查询和K线图绘制"""
    st.header("📊 股票查询与K线图")
    
    if not IMPORT_STATUS.get('k_line', False):
        st.error("K线图模块未正确加载，无法使用此功能")
        return
    
    # 数据源选择
    data_source = st.radio(
        "选择查询方式",
        ["API查询", "数据库查询", "数据库模糊查询"],
        horizontal=True
    )
    
    if data_source == "数据库查询" and not IMPORT_STATUS.get('db_connect', False):
        st.warning("⚠️ 数据库连接模块未正确加载，请先测试数据库连接")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("通过股票代码查询", type="primary", disabled=not stock_code):
            if stock_code:
                with st.spinner("正在查询股票数据..."):
                    k_data, final_stock_name, final_code = query_stock_data(stock_code, "未知", data_source)
                    if k_data is not None:
                        save_kline_image(k_data, final_code, final_stock_name)
                        
                        # 保存K线图到历史记录
                        save_kline_image_for_history(k_data, final_code, final_stock_name)
                        
                        # 保存到持久化存储
                        metadata = {
                            "stock_code": final_code,
                            "stock_name": final_stock_name,
                            "data_source": data_source,
                            "query_type": "by_code"
                        }
                        data_persistence.save_operation_history("stock_query", k_data, metadata)
                        
                        # 保存到session state
                        st.session_state.query_result = k_data
                        st.session_state.query_stock_code = final_code
                        st.session_state.query_stock_name = final_stock_name
                        st.session_state.query_source = data_source
                        
                        st.success("查询成功！数据已保存到历史记录")
                        display_stock_info(k_data, final_code, final_stock_name, data_source)
                    else:
                        st.error("查询失败，请检查股票代码是否正确")
    
    with col2:
        if st.button("通过股票名称查询", type="primary", disabled=not short_name):
            if short_name:
                with st.spinner("正在查询股票数据..."):
                    # 根据数据源选择不同的查询方式
                    if data_source == "数据库查询":
                        found_code = get_stock_code_from_db(short_name)
                    elif data_source == "数据库模糊查询":
                        # 模糊查询处理
                        fuzzy_results = fuzzy_search_stocks_from_db(short_name)
                        if fuzzy_results is not None and not fuzzy_results.empty:
                            st.success(f"找到 {len(fuzzy_results)} 只相关股票")
                            st.dataframe(fuzzy_results, use_container_width=True)
                            
                            # 保存模糊查询结果到历史记录
                            metadata = {
                                "keyword": short_name,
                                "results_count": len(fuzzy_results),
                                "query_type": "fuzzy_search"
                            }
                            data_persistence.save_operation_history("fuzzy_search", fuzzy_results, metadata)
                            st.info("模糊查询结果已保存到历史记录")
                            return
                        else:
                            st.info(f"未找到包含'{short_name}'的股票")
                            return
                    else:
                        found_code = get_stock_code_by_name(short_name)
                    
                    if found_code:
                        k_data, final_stock_name, final_code = query_stock_data(found_code, short_name, data_source)
                        if k_data is not None:
                            save_kline_image(k_data, final_code, final_stock_name)
                            
                            # 保存K线图到历史记录
                            save_kline_image_for_history(k_data, final_code, final_stock_name)
                            
                            # 保存到持久化存储
                            metadata = {
                                "stock_code": final_code,
                                "stock_name": final_stock_name,
                                "data_source": data_source,
                                "query_type": "by_name"
                            }
                            data_persistence.save_operation_history("stock_query", k_data, metadata)
                            
                            # 保存到session state
                            st.session_state.query_result = k_data
                            st.session_state.query_stock_code = final_code
                            st.session_state.query_stock_name = final_stock_name
                            st.session_state.query_source = data_source
                            
                            st.success("查询成功！数据已保存到历史记录")
                            display_stock_info(k_data, final_code, final_stock_name, data_source)
                        else:
                            st.error("获取股票数据失败")
                    else:
                        st.error("未找到相关股票，请检查股票名称是否正确")
    
    # 显示K线图
    if st.session_state.query_result is not None:
        st.subheader("📈 生成K线图")
        if st.button("显示K线图", type="primary"):
            try:
                image_path = get_latest_kline_image(st.session_state.query_stock_code)
                if image_path and os.path.exists(image_path):
                    st.success("K线图生成成功！")
                    st.image(image_path, caption=f"{st.session_state.query_stock_name} ({st.session_state.query_stock_code}) K线图", use_column_width=True)
                    display_stock_info(st.session_state.query_result, st.session_state.query_stock_code, st.session_state.query_stock_name, st.session_state.query_source)
                else:
                    st.error("未找到保存的K线图，请重新查询")
            except Exception as e:
                st.error(f"显示K线图时出现错误: {str(e)}")

def handle_lhb_query(stock_code, short_name):
    """处理龙虎榜查询"""
    st.header("🏆 龙虎榜查询")
    
    if not IMPORT_STATUS.get('lhb', False):
        st.error("龙虎榜查询模块未正确加载，无法使用此功能")
        return
    
    target_code = stock_code if stock_code else short_name
    
    if target_code:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("查询是否在龙虎榜", type="primary"):
                with st.spinner("正在查询龙虎榜数据..."):
                    try:
                        result = MODULES['lhb']['search_in_lh'](target_code)
                        if result is not None and (isinstance(result, pd.DataFrame) and not result.empty or 
                                                  isinstance(result, (list, dict)) and result):
                            # 保存到持久化存储
                            metadata = {
                                "target_code": target_code,
                                "query_type": "search_in_lh"
                            }
                            data_persistence.save_operation_history("lhb_search", result, metadata)
                            
                            st.success("该股票在龙虎榜中！数据已保存到历史记录")
                            st.dataframe(result if isinstance(result, pd.DataFrame) else st.json(result))
                        else:
                            st.info("该股票未在龙虎榜中")
                    except Exception as e:
                        st.error(f"查询过程中出现错误: {str(e)}")
        
        with col2:
            if st.button("获取龙虎榜详细数据", type="primary"):
                with st.spinner("正在获取详细数据..."):
                    try:
                        result = MODULES['lhb']['find_lhb'](target_code)
                        if result is not None and (isinstance(result, pd.DataFrame) and not result.empty or 
                                                  isinstance(result, (list, dict)) and result):
                            # 保存到持久化存储
                            metadata = {
                                "target_code": target_code,
                                "query_type": "find_lhb"
                            }
                            data_persistence.save_operation_history("lhb_detail", result, metadata)
                            
                            st.success(f"获取{target_code}龙虎榜数据成功！数据已保存到历史记录")
                            st.dataframe(result if isinstance(result, pd.DataFrame) else st.json(result))
                        else:
                            st.info("未找到相关龙虎榜数据")
                    except Exception as e:
                        st.error(f"获取数据过程中出现错误: {str(e)}")
    else:
        st.warning("请输入股票代码或股票名称")
def handle_ths_hot():
    """处理同花顺热榜"""
    st.header("🔥 同花顺热榜")
    
    if not IMPORT_STATUS.get('ths_hot', False):
        st.error("同花顺热榜模块未正确加载，无法使用此功能")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("获取同花顺热榜", type="primary"):
            with st.spinner("正在获取热榜数据..."):
                try:
                    if 'main' in MODULES['ths_hot']:
                        result = MODULES['ths_hot']['main']()
                    else:
                        st.error("热榜功能暂时不可用")
                        return
                    
                    if result is not None and (isinstance(result, pd.DataFrame) and not result.empty or 
                                              isinstance(result, (list, dict)) and result):
                        # 保存到持久化存储
                        metadata = {
                            "query_type": "hot_list"
                        }
                        data_persistence.save_operation_history("ths_hot", result, metadata)
                        
                        # 保存到session state
                        st.session_state.hot_data = result
                        st.session_state.hot_data_time = pd.Timestamp.now()
                        st.success("热榜数据获取成功！数据已保存到历史记录")
                    else:
                        st.error("获取热榜数据失败")
                except Exception as e:
                    st.error(f"获取热榜过程中出现错误: {str(e)}")
    
    # 显示热榜数据
    if hasattr(st.session_state, 'hot_data') and st.session_state.hot_data is not None:
        st.subheader("📊 热榜数据")
        if hasattr(st.session_state, 'hot_data_time') and st.session_state.hot_data_time:
            st.info(f"数据更新时间: {st.session_state.hot_data_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 筛选功能
        if isinstance(st.session_state.hot_data, pd.DataFrame) and not st.session_state.hot_data.empty:
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                price_filter = st.number_input("价格上限", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
            with col_filter2:
                change_filter = st.number_input("涨跌幅下限(%)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
            with col_filter3:
                volume_filter = st.number_input("成交量下限(万)", min_value=0.0, max_value=10000.0, value=0.0, step=100.0)
            
            filtered_data = st.session_state.hot_data.copy()
            
            # 数据类型转换和清理
            try:
                if 'price' in filtered_data.columns:
                    # 确保price列为数值类型
                    filtered_data['price'] = pd.to_numeric(filtered_data['price'], errors='coerce')
                    # 过滤掉NaN值
                    filtered_data = filtered_data[filtered_data['price'].notna() & (filtered_data['price'] <= price_filter)]
                
                if 'change_pct' in filtered_data.columns:
                    # 确保change_pct列为数值类型
                    filtered_data['change_pct'] = pd.to_numeric(filtered_data['change_pct'], errors='coerce')
                    # 过滤掉NaN值
                    filtered_data = filtered_data[filtered_data['change_pct'].notna() & (filtered_data['change_pct'] >= change_filter)]
                
                if 'volume' in filtered_data.columns:
                    # 确保volume列为数值类型
                    filtered_data['volume'] = pd.to_numeric(filtered_data['volume'], errors='coerce')
                    # 过滤掉NaN值
                    filtered_data = filtered_data[filtered_data['volume'].notna() & (filtered_data['volume'] >= volume_filter * 10000)]
            except Exception as e:
                st.warning(f"数据筛选过程中出现警告: {str(e)}")
                # 如果筛选失败，显示原始数据
                st.info("显示原始数据（筛选功能暂时不可用）")
            
            st.success(f"筛选结果: {len(filtered_data)} 只股票")
            st.dataframe(filtered_data, use_container_width=True)
        else:
            st.write(st.session_state.hot_data)
    
    with col2:
        st.subheader("绘制热榜股票K线图")
        hot_stock_code = st.text_input("输入热榜股票代码", key="hot_stock")
        if st.button("绘制K线图", disabled=not hot_stock_code):
            if hot_stock_code:
                with st.spinner("正在绘制K线图..."):
                    try:
                        stock_name = get_stock_name_by_code(hot_stock_code)
                        k_data = get_stock_data_cached(hot_stock_code)
                        if k_data is not None and not k_data.empty:
                            # 保存K线图到历史记录
                            save_kline_image_for_history(k_data, hot_stock_code, stock_name)
                            
                            # 保存到持久化存储
                            metadata = {
                                "stock_code": hot_stock_code,
                                "stock_name": stock_name,
                                "query_type": "hot_stock_kline"
                            }
                            data_persistence.save_operation_history("hot_stock_kline", k_data, metadata)
                            
                            st.success("K线图绘制成功！数据已保存到历史记录")
                            fig = MODULES['k_line']['draw_kline'](k_data, hot_stock_code)
                            st.pyplot(fig)
                            display_stock_info(k_data, hot_stock_code, stock_name, "API查询")
                        else:
                            st.error("获取股票数据失败，请检查股票代码是否正确")
                    except Exception as e:
                        st.error(f"绘制过程中出现错误: {str(e)}")
    
    # 概念计数功能
    if hasattr(st.session_state, 'hot_data') and st.session_state.hot_data is not None:
        st.subheader("📊 概念统计")
        if st.button("统计概念出现次数", type="secondary"):
            if not IMPORT_STATUS.get('ths_hot', False):
                st.error("同花顺热榜模块未正确加载")
                return
            
            with st.spinner("正在统计概念..."):
                try:
                    if 'concept_count' in MODULES['ths_hot']:
                        concept_counts = MODULES['ths_hot']['concept_count'](st.session_state.hot_data)
                        if concept_counts is not None and not concept_counts.empty:
                            # 保存到持久化存储
                            metadata = {
                                "query_type": "concept_count"
                            }
                            data_persistence.save_operation_history("concept_count", concept_counts, metadata)
                            
                            st.success("概念统计完成！数据已保存到历史记录")
                            st.dataframe(concept_counts, use_container_width=True)
                            
                            # 显示统计信息
                            total_concepts = len(concept_counts)
                            total_stocks = concept_counts['count'].sum()
                            st.info(f"统计信息: 共发现 {total_concepts} 个概念，涉及 {total_stocks} 只股票")
                        else:
                            st.warning("未找到概念数据")
                    else:
                        st.error("概念统计功能暂时不可用")
                except Exception as e:
                    st.error(f"概念统计过程中出现错误: {str(e)}")

def handle_database_management():
    """处理数据库管理"""
    st.header("💾 数据库管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("数据库连接测试")
        if st.button("测试数据库连接", type="primary"):
            if not IMPORT_STATUS.get('db_connect', False):
                st.error("数据库连接模块未正确加载")
                return
            
            with st.spinner("正在测试数据库连接..."):
                try:
                    conn = MODULES['db_connect']['db_connect']()
                    if conn:
                        # 保存连接测试结果
                        metadata = {
                            "operation": "connection_test",
                            "result": "success"
                        }
                        data_persistence.save_operation_history("db_connection_test", {"status": "success"}, metadata)
                        
                        st.success("数据库连接成功！结果已保存到历史记录")
                        conn.close()
                    else:
                        metadata = {
                            "operation": "connection_test",
                            "result": "failed"
                        }
                        data_persistence.save_operation_history("db_connection_test", {"status": "failed"}, metadata)
                        st.error("数据库连接失败")
                except Exception as e:
                    metadata = {
                        "operation": "connection_test",
                        "result": "error",
                        "error": str(e)
                    }
                    data_persistence.save_operation_history("db_connection_test", {"status": "error", "error": str(e)}, metadata)
                    st.error(f"连接测试失败: {str(e)}")
    
    with col2:
        st.subheader("数据库更新")
        if st.button("更新数据库", type="secondary"):
            if not IMPORT_STATUS.get('flush_db', False):
                st.error("数据库更新模块未正确加载")
                return
            
            with st.spinner("正在更新数据库..."):
                try:
                    if 'flush_database' in MODULES['flush_db']:
                        result = MODULES['flush_db']['flush_database']()
                        
                        # 保存更新结果
                        metadata = {
                            "operation": "database_update"
                        }
                        data_persistence.save_operation_history("db_update", result, metadata)
                        
                        st.success("数据库更新完成！结果已保存到历史记录")
                        if result:
                            st.write(result)
                    else:
                        st.error("数据库更新功能暂时不可用")
                except Exception as e:
                    metadata = {
                        "operation": "database_update",
                        "error": str(e)
                    }
                    data_persistence.save_operation_history("db_update", {"error": str(e)}, metadata)
                    st.error(f"数据库更新失败: {str(e)}")

def handle_ai_chat():
    """AI 问答助手。"""
    st.header("AI 问答助手")
    st.caption("兼容 OpenAI 风格聊天接口，可在侧边栏配置 Base URL、API Key 和模型。")

    if st.session_state.get('ai_last_error'):
        st.error(st.session_state['ai_last_error'])

    if st.session_state.get('ai_chat_history'):
        st.subheader("对话记录")
        for item in st.session_state['ai_chat_history']:
            render_ai_message(item['role'], item['content'], item.get('timestamp'))
    else:
        st.info("当前还没有对话。先在侧边栏完成接口配置，然后输入问题开始聊天。")

    with st.expander("上下文设置", expanded=False):
        st.checkbox("把当前股票查询结果作为上下文", key="ai_include_stock")
        st.checkbox("把当前热榜数据作为上下文", key="ai_include_hot")
        context_preview = build_ai_context()
        if context_preview:
            st.text_area("当前将附带给模型的上下文预览", value=context_preview, height=220, disabled=True)
        else:
            st.caption("当前没有附加上下文。")

    st.text_input("本次会话名称", key="ai_session_name", placeholder="例如：盘中复盘助手")
    user_prompt = st.text_area(
        "输入你的问题",
        placeholder="例如：结合我当前查询的股票走势，帮我做一个简要观察。",
        key="ai_user_prompt",
        height=120
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        send_clicked = st.button("发送给 AI", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("清空当前对话", use_container_width=True)

    if clear_clicked:
        st.session_state.ai_chat_history = []
        st.session_state.ai_last_error = None
        st.rerun()

    if send_clicked:
        st.session_state.ai_last_error = None

        if not user_prompt.strip():
            st.warning("请输入问题后再发送。")
            return

        if not st.session_state.get('ai_base_url', '').strip():
            st.warning("请先在侧边栏填写 API Base URL。")
            return

        if not st.session_state.get('ai_api_key', '').strip():
            st.warning("请先在侧边栏填写 API Key。")
            return

        if not st.session_state.get('ai_model', '').strip():
            st.warning("请先在侧边栏填写模型名称。")
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.ai_chat_history.append(
            {
                "role": "user",
                "content": user_prompt.strip(),
                "timestamp": timestamp
            }
        )

        with st.spinner("AI 正在生成回答..."):
            try:
                messages = build_ai_messages()
                response = chat_completion(
                    base_url=st.session_state['ai_base_url'],
                    api_key=st.session_state['ai_api_key'],
                    model=st.session_state['ai_model'],
                    messages=messages,
                    temperature=st.session_state.get('ai_temperature', 0.7),
                    timeout=st.session_state.get('ai_timeout', 60),
                )
                answer = response['content']
                answer_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.ai_chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": answer_timestamp
                    }
                )

                metadata = {
                    "session_name": st.session_state.get('ai_session_name', ''),
                    "model": st.session_state['ai_model'],
                    "base_url": st.session_state['ai_base_url'],
                    "message_count": len(st.session_state['ai_chat_history']),
                    "include_stock_context": st.session_state.get('ai_include_stock', False),
                    "include_hot_context": st.session_state.get('ai_include_hot', False),
                }
                data_persistence.save_operation_history("ai_chat", st.session_state['ai_chat_history'], metadata)
                st.session_state.ai_user_prompt = ""
                st.rerun()
            except AIClientError as e:
                st.session_state.ai_last_error = str(e)
                st.error(f"AI 调用失败: {str(e)}")
            except Exception as e:
                st.session_state.ai_last_error = str(e)
                st.error(f"发生未知错误: {str(e)}")


def show_history_panel():
    """显示历史记录面板"""
    st.header("📚 操作历史记录")
    
    history = data_persistence.load_operation_history()
    
    if not history:
        st.info("暂无历史记录")
        return
    
    # 操作统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总操作数", len(history))
    with col2:
        operation_types = [entry.get("operation_type", "unknown") for entry in history]
        unique_types = len(set(operation_types))
        st.metric("操作类型", unique_types)
    with col3:
        recent_operations = len([h for h in history if 
                               (datetime.now() - datetime.fromisoformat(h["timestamp"])).days < 1])
        st.metric("今日操作", recent_operations)
    with col4:
        if st.button("清空历史记录", type="secondary"):
            if data_persistence.clear_history():
                st.success("历史记录已清空")
                st.rerun()
            else:
                st.error("清空历史记录失败")
    
    # 筛选选项
    st.subheader("筛选历史记录")
    col1, col2 = st.columns(2)
    
    with col1:
        operation_filter = st.selectbox(
            "按操作类型筛选",
            ["全部"] + list(set(operation_types)),
            index=0
        )
    
    with col2:
        days_filter = st.selectbox(
            "按时间筛选",
            ["全部", "今天", "最近3天", "最近7天", "最近30天"],
            index=0
        )
    
    # 应用筛选
    filtered_history = history.copy()
    
    if operation_filter != "全部":
        filtered_history = [h for h in filtered_history if h.get("operation_type") == operation_filter]
    
    if days_filter != "全部":
        days_map = {"今天": 1, "最近3天": 3, "最近7天": 7, "最近30天": 30}
        days = days_map[days_filter]
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        filtered_history = [h for h in filtered_history if 
                          datetime.fromisoformat(h["timestamp"]) >= cutoff_date]
    
    # 显示历史记录
    st.subheader(f"历史记录 ({len(filtered_history)} 条)")
    
    for i, entry in enumerate(reversed(filtered_history[-50:])):  # 显示最近50条
        operation_type = entry.get('operation_type', 'unknown')
        timestamp = entry.get('timestamp', '')[:19]
        
        # 构建更友好的标题
        if operation_type == 'stock_query':
            metadata = entry.get('metadata', {})
            title = f"📊 股票查询: {metadata.get('stock_name', 'N/A')} ({metadata.get('stock_code', 'N/A')}) - {timestamp}"
        elif operation_type == 'lhb_search':
            metadata = entry.get('metadata', {})
            title = f"🏆 龙虎榜查询: {metadata.get('target_code', 'N/A')} - {timestamp}"
        elif operation_type == 'ths_hot':
            title = f"� 同花顺热榜  - {timestamp}"
        elif operation_type == 'concept_count':
            title = f"📊 概念统计 - {timestamp}"
        else:
            title = f"{operation_type} - {timestamp}"
        
        with st.expander(title):
            # 检查是否为K线图相关操作
            if operation_type in ['stock_query', 'hot_stock_kline']:
                metadata = entry.get('metadata', {})
                stock_code = metadata.get('stock_code')
                if stock_code:
                    # 尝试显示K线图
                    kline_image_path = f"image/{stock_code}.png"
                    if os.path.exists(kline_image_path):
                        st.image(kline_image_path, caption=f"{metadata.get('stock_name', 'N/A')} ({stock_code}) K线图", use_column_width=True)
                    else:
                        st.warning("K线图文件不存在")
            
            # 显示历史数据（如果存在且不是K线图操作）
            if entry.get('data_file'):
                data = data_persistence.load_operation_data(entry['data_file'])
                if data is not None:
                    if isinstance(data, pd.DataFrame):
                        # 显示数据形状信息
                        st.info(f"数据形状: {data.shape[0]} 行 × {data.shape[1]} 列")
                        
                        # 查看方式选择
                        view_option = st.radio(
                            "查看方式",
                            ["完整数据", "前10行", "后10行", "数据统计"],
                            horizontal=True,
                            key=f"view_option_{i}"
                        )
                        
                        if view_option == "完整数据":
                            st.dataframe(data, use_container_width=True, height=400)
                        elif view_option == "前10行":
                            st.dataframe(data.head(10), use_container_width=True)
                        elif view_option == "后10行":
                            st.dataframe(data.tail(10), use_container_width=True)
                        elif view_option == "数据统计":
                            if data.select_dtypes(include=[np.number]).shape[1] > 0:
                                st.subheader("📈 数值列统计")
                                st.dataframe(data.describe(), use_container_width=True)
                            
                            st.subheader("📋 数据信息")
                            info_data = {
                                "列名": data.columns.tolist(),
                                "数据类型": data.dtypes.astype(str).tolist(),
                                "非空值数量": data.count().tolist(),
                                "空值数量": data.isnull().sum().tolist()
                            }
                            info_df = pd.DataFrame(info_data)
                            st.dataframe(info_df, use_container_width=True)
                        
                        # 数据导出功能
                        csv = data.to_csv(index=False)
                        st.download_button(
                            label="💾 导出CSV文件",
                            data=csv,
                            file_name=f"{operation_type}_{timestamp.replace(':', '-')}.csv",
                            mime="text/csv",
                            key=f"download_{i}"
                        )
                    else:
                        # 对于非DataFrame数据，使用JSON显示
                        st.json(data)
                else:
                    st.error("❌ 无法加载历史数据，文件可能已损坏或被删除")

def main():
    st.title("📈 股票分析系统 Pro")
    st.markdown("---")
    
    # 欢迎信息
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        if IMPORT_EXPLAN:
            explan.show_welcome_message()
        else:
            st.success("🎉 欢迎使用股票分析系统 Pro！")
            st.info("💡 Pro版本新增功能：所有操作结果都会自动保存到历史记录中，支持数据常驻和状态记录。")
    
    # 初始化session_state
    if 'query_result' not in st.session_state:
        st.session_state.query_result = None
        st.session_state.query_stock_code = None
        st.session_state.query_stock_name = None
        st.session_state.query_source = None
        st.session_state.hot_data = None
        st.session_state.hot_data_time = None
    
    # 顶部控制按钮
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("📋 切换侧边栏", help="点击隐藏或显示侧边栏"):
            if 'sidebar_visible' not in st.session_state:
                st.session_state.sidebar_visible = True
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
    
    with col2:
        if st.button("🔧 系统状态", help="查看系统模块加载状态"):
            st.session_state.show_status = not st.session_state.get('show_status', False)
    
    with col3:
        if st.button("❓ 快速帮助", help="查看快速使用指南"):
            st.session_state.show_help = not st.session_state.get('show_help', False)
    
    with col4:
        if st.button("📚 历史记录", help="查看操作历史记录"):
            st.session_state.show_history = not st.session_state.get('show_history', False)
    
    # 显示UI组件
    if IMPORT_EXPLAN:
        explan.show_ui_components(
            import_status=IMPORT_STATUS,
            show_help=st.session_state.get('show_help', False),
            show_status=st.session_state.get('show_status', False),
            show_welcome=False
        )
    else:
        # 使用备用显示逻辑
        if hasattr(explan, 'show_fallback_ui'):
            explan.show_fallback_ui(
                import_status=IMPORT_STATUS,
                show_help=st.session_state.get('show_help', False),
                show_status=st.session_state.get('show_status', False)
            )
    
    # 显示历史记录面板
    if st.session_state.get('show_history', False):
        show_history_panel()
        st.markdown("---")
    
    # 侧边栏功能选择
    if 'sidebar_visible' not in st.session_state or st.session_state.sidebar_visible:
        st.sidebar.title("功能选择")
        function_choice = st.sidebar.selectbox(
            "选择功能模块",
            ["股票查询与K线图", "龙虎榜查询", "同花顺热榜", "数据库管理"]
        )
        
        # 项目介绍
        if IMPORT_EXPLAN and st.sidebar.checkbox("显示项目介绍", value=False):
            explan.show_explan()
        
        # 历史记录快捷访问
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 快捷访问")
        if st.sidebar.button("查看历史记录"):
            st.session_state.show_history = True
            st.rerun()
        
        # 显示最近操作
        recent_history = data_persistence.load_operation_history()[-5:]  # 最近5条
        if recent_history:
            st.sidebar.subheader("🕒 最近操作")
            for entry in reversed(recent_history):
                operation_type = entry.get('operation_type', 'unknown')
                timestamp = entry.get('timestamp', '')[:16]  # 只显示到分钟
                metadata = entry.get('metadata', {})
                
                # 构建显示文本
                if operation_type == 'stock_query':
                    display_text = f"股票查询: {metadata.get('stock_code', 'N/A')}"
                elif operation_type == 'lhb_search':
                    display_text = f"龙虎榜: {metadata.get('target_code', 'N/A')}"
                elif operation_type == 'ths_hot':
                    display_text = "同花顺热榜"
                else:
                    display_text = operation_type
                
                st.sidebar.text(f"{timestamp} - {display_text}")
    else:
        # 当侧边栏隐藏时，使用下拉菜单
        function_choice = st.selectbox(
            "选择功能模块",
            ["股票查询与K线图", "龙虎榜查询", "同花顺热榜", "数据库管理"]
        )
    
    # 主要输入区域
    col1, col2 = st.columns(2)
    with col1:
        stock_code = st.text_input("股票代码", placeholder="例如: 000001, 600519")
    with col2:
        short_name = st.text_input("股票名称", placeholder="例如: 平安银行, 贵州茅台")
    
    st.markdown("---")
    
    # 根据选择的功能显示不同内容
    if function_choice == "股票查询与K线图":
        handle_stock_query(stock_code, short_name)
    elif function_choice == "龙虎榜查询":
        handle_lhb_query(stock_code, short_name)
    elif function_choice == "同花顺热榜":
        handle_ths_hot()
    elif function_choice == "数据库管理":
        handle_database_management()

def app_main():
    apply_design_theme()

    function_options = [
        "股票查询与K线图",
        "龙虎榜查询",
        "同花顺热榜",
        "AI 问答助手",
        "数据库管理",
    ]

    st.title("股票分析系统 Pro")
    st.caption("统一行情查询、热榜观察、历史记录与 AI 对话。")
    st.markdown("---")

    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        if IMPORT_EXPLAN:
            explan.show_welcome_message()
        else:
            st.success("欢迎使用股票分析系统 Pro")
            st.info("当前版本新增 AI 对话入口，并保留历史记录与状态面板。")

    defaults = {
        'query_result': None,
        'query_stock_code': None,
        'query_stock_name': None,
        'query_source': None,
        'hot_data': None,
        'hot_data_time': None,
        'sidebar_visible': True,
        'show_status': False,
        'show_help': False,
        'show_history': False,
        'ai_chat_history': [],
        'ai_base_url': "",
        'ai_api_key': "",
        'ai_model': "",
        'ai_system_prompt': "你是一个中文股票分析助手。仅基于用户提供的信息和上下文回答，不要编造数据。",
        'ai_temperature': 0.7,
        'ai_timeout': 60,
        'ai_include_stock': True,
        'ai_include_hot': False,
        'ai_last_error': None,
        'ai_user_prompt': "",
        'ai_session_name': "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("切换侧边栏", help="隐藏或显示侧边栏"):
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
    with col2:
        if st.button("系统状态", help="查看模块加载状态"):
            st.session_state.show_status = not st.session_state.get('show_status', False)
    with col3:
        if st.button("快速帮助", help="查看基础使用说明"):
            st.session_state.show_help = not st.session_state.get('show_help', False)
    with col4:
        if st.button("历史记录", help="查看保存的操作历史"):
            st.session_state.show_history = not st.session_state.get('show_history', False)

    if IMPORT_EXPLAN:
        explan.show_ui_components(
            import_status=IMPORT_STATUS,
            show_help=st.session_state.get('show_help', False),
            show_status=st.session_state.get('show_status', False),
            show_welcome=False
        )
    else:
        if hasattr(explan, 'show_fallback_ui'):
            explan.show_fallback_ui(
                import_status=IMPORT_STATUS,
                show_help=st.session_state.get('show_help', False),
                show_status=st.session_state.get('show_status', False)
            )

    if st.session_state.get('show_history', False):
        show_history_panel()
        st.markdown("---")

    if st.session_state.sidebar_visible:
        st.sidebar.title("功能选择")
        function_choice = st.sidebar.selectbox("选择功能模块", function_options)

        with st.sidebar.expander("AI 设置", expanded=function_choice == "AI 问答助手"):
            st.text_input("API Base URL", key="ai_base_url", placeholder="例如: https://api.openai.com")
            st.text_input("API Key", key="ai_api_key", type="password", placeholder="输入服务商 API Key")
            st.text_input("模型名称", key="ai_model", placeholder="例如: gpt-4o-mini 或 deepseek-chat")
            st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, key="ai_temperature")
            st.number_input("超时时间（秒）", min_value=10, max_value=300, step=5, key="ai_timeout")
            st.text_area("System Prompt", key="ai_system_prompt", height=140)

        if IMPORT_EXPLAN and st.sidebar.checkbox("显示项目介绍", value=False):
            explan.show_explan()

        st.sidebar.markdown("---")
        st.sidebar.subheader("快捷访问")
        if st.sidebar.button("查看历史记录"):
            st.session_state.show_history = True
            st.rerun()

        recent_history = data_persistence.load_operation_history()[-5:]
        if recent_history:
            st.sidebar.subheader("最近操作")
            for entry in reversed(recent_history):
                operation_type = entry.get('operation_type', 'unknown')
                timestamp = entry.get('timestamp', '')[:16]
                metadata = entry.get('metadata', {})

                if operation_type == 'stock_query':
                    display_text = f"股票查询: {metadata.get('stock_code', 'N/A')}"
                elif operation_type == 'lhb_search':
                    display_text = f"龙虎榜: {metadata.get('target_code', 'N/A')}"
                elif operation_type == 'ths_hot':
                    display_text = "同花顺热榜"
                elif operation_type == 'ai_chat':
                    display_text = f"AI 对话: {metadata.get('model', 'N/A')}"
                else:
                    display_text = operation_type

                st.sidebar.text(f"{timestamp} - {display_text}")
    else:
        function_choice = st.selectbox("选择功能模块", function_options)

    stock_code = ""
    short_name = ""
    if function_choice in ["股票查询与K线图", "龙虎榜查询"]:
        col1, col2 = st.columns(2)
        with col1:
            stock_code = st.text_input("股票代码", placeholder="例如: 000001, 600519")
        with col2:
            short_name = st.text_input("股票名称", placeholder="例如: 平安银行, 贵州茅台")
        st.markdown("---")

    if function_choice == "股票查询与K线图":
        handle_stock_query(stock_code, short_name)
    elif function_choice == "龙虎榜查询":
        handle_lhb_query(stock_code, short_name)
    elif function_choice == "同花顺热榜":
        handle_ths_hot()
    elif function_choice == "AI 问答助手":
        handle_ai_chat()
    elif function_choice == "数据库管理":
        handle_database_management()


if __name__ == "__main__":
    app_main()
