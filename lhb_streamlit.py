import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import adata
import time
from datetime import datetime

from ai_client import AIClientError, chat_completion



def get_streamlit_secret(*keys, default=""):
    """按多个候选键读取 Streamlit secrets。"""
    try:
        secrets = st.secrets
    except Exception:
        return default

    for key in keys:
        if not key:
            continue
        try:
            if "." in key:
                current = secrets
                matched = True
                for part in key.split("."):
                    if hasattr(current, "get"):
                        current = current.get(part)
                    else:
                        matched = False
                        break
                    if current is None:
                        matched = False
                        break
                if matched:
                    return current
            elif hasattr(secrets, "get"):
                value = secrets.get(key)
                if value not in (None, ""):
                    return value
        except Exception:
            continue

    return default



def load_ai_settings_from_secrets():
    """读取 AI 配置，兼容顶层键和 [ai] 分组。"""
    return {
        "base_url": get_streamlit_secret("AI_BASE_URL", "ai.base_url", default=""),
        "api_key": get_streamlit_secret("AI_API_KEY", "ai.api_key", default=""),
        "model": get_streamlit_secret("AI_MODEL", "ai.model", default=""),
    }



def apply_ai_settings_source():
    """根据当前来源设置，将 secrets 中的 AI 配置同步到 session_state。"""
    secrets_config = load_ai_settings_from_secrets()
    secrets_available = any(str(value).strip() for value in secrets_config.values())
    st.session_state.ai_secrets_available = secrets_available

    use_secrets = st.session_state.get('ai_use_secrets', False)
    if not use_secrets or not secrets_available:
        return secrets_config

    if str(secrets_config.get("base_url", "")).strip():
        st.session_state.ai_base_url = str(secrets_config["base_url"])
    if str(secrets_config.get("api_key", "")).strip():
        st.session_state.ai_api_key = str(secrets_config["api_key"])
    if str(secrets_config.get("model", "")).strip():
        st.session_state.ai_model = str(secrets_config["model"])

    return secrets_config



def mask_secret(value):
    """掩码显示敏感信息。"""
    text = str(value or "")
    if not text:
        return "未配置"
    if len(text) <= 6:
        return "*" * len(text)
    return f"{text[:3]}***{text[-3:]}"

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

# 创建image文件夹
if not os.path.exists('image'):
    os.makedirs('image')

# 设置页面配置
st.set_page_config(
    page_title="股票分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 安全导入模块
def safe_import():
    """安全导入模块"""
    modules = {}
    import_status = {}
    
    module_configs = [
        ('api_search', ['api_search_draw'], ['api_search_code_draw', 'api_search_name_draw']),
        ('db_search', ['db_search_draw'], ['database_search_name_draw', 'database_search_code_draw', 'database_get_stock_name']),
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
    except Exception:
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

    if IMPORT_STATUS.get('db_search', False):
        try:
            db_func = MODULES['db_search'].get('database_get_stock_name')
            if db_func:
                db_result = db_func(stock_code)
                if db_result:
                    return db_result
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
        result = MODULES['db_search']['database_search_code_draw'](stock_code)
        return result if result else None
    except Exception as e:
        st.error(f"数据库查询股票名称失败: {str(e)}")
        return None

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
    time_html = f"<div style='margin-top:8px;color:#666;font-size:12px;'>{timestamp}</div>" if timestamp else ""
    st.markdown(
        f"""
        <div style="padding:12px 14px;margin-bottom:12px;border:1px solid #e6e6e6;border-radius:10px;background:#fafafa;">
            <div style="font-weight:600;margin-bottom:8px;">{role_map.get(role, role)}</div>
            <div>{safe_content}</div>
            {time_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        ["API查询", "数据库查询"],
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
                        st.session_state.query_result = k_data
                        st.session_state.query_stock_code = final_code
                        st.session_state.query_stock_name = final_stock_name
                        st.session_state.query_source = data_source
                        st.success("查询成功！")
                        display_stock_info(k_data, final_code, final_stock_name, data_source)
                    else:
                        st.error("查询失败，请检查股票代码是否正确")
    
    with col2:
        if st.button("通过股票名称查询", type="primary", disabled=not short_name):
            if short_name:
                with st.spinner("正在查询股票数据..."):
                    found_code = get_stock_code_by_name(short_name)
                    if found_code:
                        k_data, final_stock_name, final_code = query_stock_data(found_code, short_name, data_source)
                        if k_data is not None:
                            save_kline_image(k_data, final_code, final_stock_name)
                            st.session_state.query_result = k_data
                            st.session_state.query_stock_code = final_code
                            st.session_state.query_stock_name = final_stock_name
                            st.session_state.query_source = data_source
                            st.success("查询成功！")
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
                            st.success("该股票在龙虎榜中!")
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
                            st.success(f"获取{target_code}龙虎榜数据成功!")
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
                        st.session_state.hot_data = result
                        st.session_state.hot_data_time = pd.Timestamp.now()
                        st.success("热榜数据获取成功!")
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
                            st.success("K线图绘制成功!")
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
                            st.success("概念统计完成!")
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
        st.info("当前还没有对话。先在侧边栏完成接口配置，然后输入问题开始使用。")

    with st.expander("上下文设置", expanded=False):
        st.checkbox("把当前股票查询结果作为上下文", key="ai_include_stock")
        st.checkbox("把当前热榜数据作为上下文", key="ai_include_hot")
        context_preview = build_ai_context()
        if context_preview:
            st.text_area("当前将附带给模型的上下文预览", value=context_preview, height=220, disabled=True, key="ai_context_preview")
        else:
            st.caption("当前没有附加上下文。")

    prompt_key = f"ai_user_prompt_{st.session_state.get('ai_prompt_nonce', 0)}"
    user_prompt = st.text_area(
        "输入你的问题",
        placeholder="例如：结合当前股票走势和热榜数据，帮我做一个简要观察。",
        key=prompt_key,
        height=120
    )

    col1, col2 = st.columns(2)
    with col1:
        send_clicked = st.button("发送给 AI", type="primary", use_container_width=True, key="ai_send_button")
    with col2:
        clear_clicked = st.button("清空当前对话", use_container_width=True, key="ai_clear_button")

    if clear_clicked:
        st.session_state.ai_chat_history = []
        st.session_state.ai_last_error = None
        st.session_state.ai_prompt_nonce = st.session_state.get('ai_prompt_nonce', 0) + 1
        st.rerun()

    if not send_clicked:
        return

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
        {"role": "user", "content": user_prompt.strip(), "timestamp": timestamp}
    )

    with st.spinner("AI 正在生成回答..."):
        try:
            response = chat_completion(
                base_url=st.session_state['ai_base_url'],
                api_key=st.session_state['ai_api_key'],
                model=st.session_state['ai_model'],
                messages=build_ai_messages(),
                temperature=st.session_state.get('ai_temperature', 0.7),
                timeout=st.session_state.get('ai_timeout', 60),
            )
            answer_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.ai_chat_history.append(
                {"role": "assistant", "content": response['content'], "timestamp": answer_timestamp}
            )
            st.session_state.ai_prompt_nonce = st.session_state.get('ai_prompt_nonce', 0) + 1
            st.rerun()
        except AIClientError as e:
            st.session_state.ai_last_error = str(e)
            st.error(f"AI 调用失败: {str(e)}")
        except Exception as e:
            st.session_state.ai_last_error = str(e)
            st.error(f"发生未知错误: {str(e)}")


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
                        st.success("数据库连接成功!")
                        conn.close()
                    else:
                        st.error("数据库连接失败")
                except Exception as e:
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
                        st.success("数据库更新完成!")
                        if result:
                            st.write(result)
                    else:
                        st.error("数据库更新功能暂时不可用")
                except Exception as e:
                    st.error(f"数据库更新失败: {str(e)}")

def main():
    st.title("股票分析系统")
    st.markdown("---")

    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        if IMPORT_EXPLAN:
            explan.show_welcome_message()
        else:
            st.success("欢迎使用股票分析系统。")
            st.info("首次使用建议先查看快速帮助，并确认系统状态中的模块加载结果。")

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
        'ai_chat_history': [],
        'ai_base_url': "",
        'ai_api_key': "",
        'ai_model': "",
        'ai_use_secrets': False,
        'ai_secrets_available': False,
        'ai_system_prompt': "你是一个中文股票分析助手。仅基于用户提供的信息和上下文回答，不要编造数据。",
        'ai_temperature': 0.7,
        'ai_timeout': 60,
        'ai_include_stock': True,
        'ai_include_hot': False,
        'ai_last_error': None,
        'ai_user_prompt': "",
        'ai_prompt_nonce': 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    secrets_config = apply_ai_settings_source()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("切换侧边栏", help="隐藏或显示侧边栏", key="main_toggle_sidebar"):
            st.session_state.sidebar_visible = not st.session_state.get('sidebar_visible', True)
    with col2:
        if st.button("系统状态", help="查看模块加载状态", key="main_toggle_status"):
            st.session_state.show_status = not st.session_state.get('show_status', False)
    with col3:
        if st.button("快速帮助", help="查看基础使用说明", key="main_toggle_help"):
            st.session_state.show_help = not st.session_state.get('show_help', False)

    if IMPORT_EXPLAN:
        explan.show_ui_components(
            import_status=IMPORT_STATUS,
            show_help=st.session_state.get('show_help', False),
            show_status=st.session_state.get('show_status', False),
            show_welcome=False
        )
    elif 'explan' in globals() and hasattr(explan, 'show_fallback_ui'):
        explan.show_fallback_ui(
            import_status=IMPORT_STATUS,
            show_help=st.session_state.get('show_help', False),
            show_status=st.session_state.get('show_status', False)
        )

    function_options = ["股票查询与K线图", "龙虎榜查询", "同花顺热榜", "AI 问答助手", "数据库管理"]

    if st.session_state.get('sidebar_visible', True):
        st.sidebar.title("功能选择")
        function_choice = st.sidebar.selectbox("选择功能模块", function_options, key="sidebar_function_choice")

        with st.sidebar.expander("AI 设置", expanded=function_choice == "AI 问答助手"):
            secrets_available = st.session_state.get('ai_secrets_available', False)
            st.checkbox(
                "优先使用 Streamlit secrets",
                key="ai_use_secrets",
                disabled=not secrets_available,
                help="启用后优先从 .streamlit/secrets.toml 或部署平台 Secrets 读取 AI 配置。"
            )

            if secrets_available:
                st.caption("已检测到 secrets 配置，可直接复用其中的 URL / Key / 模型。")
                with st.container(border=True):
                    st.text_input("Secrets Base URL", value=str(secrets_config.get('base_url', '')), disabled=True, key="ai_secret_base_url_preview")
                    st.text_input("Secrets API Key", value=mask_secret(secrets_config.get('api_key', '')), disabled=True, key="ai_secret_api_key_preview")
                    st.text_input("Secrets 模型名称", value=str(secrets_config.get('model', '')), disabled=True, key="ai_secret_model_preview")
            else:
                st.caption("未检测到 secrets 配置，当前继续使用手动输入。")

            manual_disabled = st.session_state.get('ai_use_secrets', False) and secrets_available
            st.text_input("API Base URL", key="ai_base_url", placeholder="例如: https://api.openai.com", disabled=manual_disabled)
            st.text_input("API Key", key="ai_api_key", type="password", placeholder="输入服务商 API Key", disabled=manual_disabled)
            st.text_input("模型名称", key="ai_model", placeholder="例如: gpt-4o-mini 或 deepseek-chat", disabled=manual_disabled)
            st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, key="ai_temperature")
            st.number_input("超时时间（秒）", min_value=10, max_value=300, step=5, key="ai_timeout")
            st.text_area("System Prompt", key="ai_system_prompt", height=140)

        if IMPORT_EXPLAN and st.sidebar.checkbox("显示项目介绍", value=False, key="sidebar_show_explan"):
            explan.show_explan()
    else:
        function_choice = st.selectbox("选择功能模块", function_options, key="main_function_choice")

    stock_code = ""
    short_name = ""
    if function_choice in ["股票查询与K线图", "龙虎榜查询"]:
        col1, col2 = st.columns(2)
        with col1:
            stock_code = st.text_input("股票代码", placeholder="例如: 000001, 600519", key="main_stock_code")
        with col2:
            short_name = st.text_input("股票名称", placeholder="例如: 平安银行, 贵州茅台", key="main_short_name")
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
    main()
