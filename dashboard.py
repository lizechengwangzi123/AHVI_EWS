import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ==========================================
# 1. [核心类] 论文功能实现：Operational EWS System
# ==========================================
class OperationalEWS:
    def __init__(self, history_mean=0.45, history_std=0.15):
        """
        初始化系统参数
        :param history_mean: 历史预测均值 (用于异常检测 Eq 2)
        :param history_std: 历史标准差 (用于异常检测 Eq 2)
        """
        self.history_mean = history_mean
        self.history_std = history_std
        
        # 定义风险等级阈值和基础建议
        self.RISK_LEVELS = {
            'YELLOW': {'threshold': 0.5, 'msg': 'Advisory: Moderate heat. Monitor local forecasts.'},
            'ORANGE': {'threshold': 0.7, 'msg': 'Warning: High heat risk! Stay hydrated & limit outdoor activity.'},
            'RED':    {'threshold': 0.9, 'msg': 'Emergency: Immediate danger! Seek cooling centers immediately.'}
        }

    def hybrid_predict(self, model_prediction, utci_forecast):
        """
        对应论文 Eq (1): Rule-Based Overrides
        逻辑：先看模型分数，但如果 UTCI > 46°C，强制锁定为 RED。
        """
        # 1. 基础模型判断 (A_ML)
        risk_level = 'YELLOW' # 默认为 Yellow (Low Risk)
        
        if model_prediction >= self.RISK_LEVELS['RED']['threshold']:
            risk_level = 'RED'
        elif model_prediction >= self.RISK_LEVELS['ORANGE']['threshold']:
            risk_level = 'ORANGE'
        
        # 2. [核心规则] 强制覆盖逻辑 (A_Final)
        is_override = False
        if utci_forecast > 46.0:
            risk_level = 'RED'
            is_override = True  # 标记为规则触发
            
        return risk_level, is_override

    def check_anomaly(self, new_prediction):
        """
        对应论文 Eq (2): Statistical Anomaly Detection
        逻辑：Flag = 1 if |Val - Mean| > 3 * Std
        """
        deviation = abs(new_prediction - self.history_mean)
        limit = 3 * self.history_std
        
        if deviation > limit:
            return True, f"⚠️ Anomaly Detected! Prediction ({new_prediction:.2f}) deviates > 3 Sigma from history."
        return False, "✅ Data Integrity Verified: Within normal statistical range."

# ==========================================
# 2. 页面配置与初始化
# ==========================================
st.set_page_config(
    page_title="AHVI+ EWS Operational Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# 初始化系统实例
# 注意：history_mean 和 history_std 应该基于你的训练集计算，这里使用示例值
ews_system = OperationalEWS(history_mean=0.45, history_std=0.15)

# ==========================================
# 3. 数据与模型加载
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # 尝试加载你的模型和数据，如果文件不存在则使用模拟数据防止报错
        if os.path.exists('champion_model_pipeline.joblib'):
            model = joblib.load('champion_model_pipeline.joblib')
        else:
            model = None
            
        if os.path.exists('malaysia_states.json'):
            with open('malaysia_states.json') as f:
                geojson = json.load(f)
        else:
            geojson = None
            
        return model, geojson
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, geojson = load_resources()

# ==========================================
# 4. Sidebar: 系统控制台 (Operational Console)
# ==========================================
st.sidebar.header("⚙️ EWS Control Panel")

# 模拟输入部分 (用于演示论文逻辑)
st.sidebar.subheader("1. Forecast Inputs")
# 这里是为了演示 Rule-Based Override，特意让用户能拉动 UTCI
sim_utci = st.sidebar.slider(
    "Forecast UTCI (°C)", 
    min_value=25.0, 
    max_value=50.0, 
    value=32.0,
    help="Drag above 46°C to test the Rule-Based Override mechanism."
)

st.sidebar.subheader("2. Model Inputs (Simulation)")
# 模拟一些特征输入，用于生成模型分数
feature_a = st.sidebar.slider("Heat Wave Intensity", 0.0, 1.0, 0.5)
feature_b = st.sidebar.slider("Vulnerability Factor", 0.0, 1.0, 0.5)

# ==========================================
# 5. 主界面逻辑
# ==========================================
st.title("🌡️ AHVI+ Functional Decision Support System")
st.markdown("### Operational Early Warning System for Heat Risk")

# --- 生成预测值 (模拟或真实调用) ---
if model:
    # 这里应该构造真实的 DataFrame 输入给模型
    # 演示目的：我们生成一个受输入影响的假分数
    pred_score = (feature_a + feature_b) / 2 
    # 如果你有真实特征构造逻辑，请在这里替换:
    # pred_score = model.predict(input_df)[0]
else:
    # 如果没有模型文件，使用模拟分数
    pred_score = (feature_a + feature_b) / 2 

# ==========================================
# [关键步骤] 调用 OperationalEWS 处理逻辑
# ==========================================

# 1. 混合预测 (Hybrid Prediction)
final_level, override_triggered = ews_system.hybrid_predict(pred_score, sim_utci)

# 2. 异常检测 (Anomaly Detection)
is_anomaly, anomaly_msg = ews_system.check_anomaly(pred_score)

# ==========================================
# 6. 结果展示面板
# ==========================================

st.divider()

# --- 第一排：核心预警卡片 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🛡️ Current Risk Status")
    
    # 根据等级定义颜色
    color_map = {
        'RED': ('#FF4B4B', '🚨 EMERGENCY'), 
        'ORANGE': ('#FFA500', '⚠️ WARNING'), 
        'YELLOW': ('#FFD700', 'ℹ️ ADVISORY')
    }
    bg_color, status_text = color_map.get(final_level)
    msg_text = ews_system.RISK_LEVELS[final_level]['msg']
    
    # 使用 HTML/CSS 渲染醒目的警告卡片
    st.markdown(f"""
        <div style="
            background-color: {bg_color}; 
            padding: 25px; 
            border-radius: 10px; 
            border: 2px solid #333;
            color: black;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h1 style="margin:0; font-size: 3em;">{status_text}</h1>
            <h3 style="margin-top:10px;">Action Required: {msg_text}</h3>
        </div>
    """, unsafe_allow_html=True)

    # 展示规则覆盖警告 (论文 Equation 1)
    if override_triggered:
        st.error(f"🚨 **SYSTEM OVERRIDE ACTIVE**: Forecast UTCI ({sim_utci}°C) exceeds critical threshold (46°C). AI prediction ignored.")
    else:
        st.info(f"System logic: Based on AHVI+ AI Model (UTCI < 46°C)")

with col2:
    st.subheader("📊 Statistical Diagnostics")
    st.metric("Raw AI Risk Score", f"{pred_score:.4f}")
    st.metric("Forecast UTCI", f"{sim_utci}°C")
    
    # 展示异常检测结果 (论文 Equation 2)
    st.markdown("**Data Integrity Check:**")
    if is_anomaly:
        st.warning(anomaly_msg)
    else:
        st.success(anomaly_msg)

# --- 第二排：分级预警发布协议 (Tiered Alert Dissemination Protocol) ---
st.divider()
st.subheader("📡 Active Alert Dissemination Protocol")
st.markdown("Based on the assessed risk level, the following communication strategy is automatically activated:")

# 定义协议内容 (完全对应论文文本)
protocol_data = {
    "Risk Level": ["RED", "ORANGE", "YELLOW"],
    "Target Audience": [
        "General Public & Emergency Services", 
        "Vulnerable Groups (Elderly/Children) & Health Providers", 
        "General Public"
    ],
    "Primary Communication Channels": [
        "National Broadcast, SMS Emergency Alerts, NGO Networks", 
        "Community Apps, Direct SMS to registered risk groups", 
        "Weather App Updates, Website Banner, Social Media"
    ],
    "Core Message": [
        "IMMEDIATE DANGER: Seek cooling centers, check on vulnerable neighbors.", 
        "WARNING: High heat risk. Stay hydrated, avoid outdoor activities.", 
        "ADVISORY: Moderate heat expected. Monitor local forecasts."
    ]
}

df_protocol = pd.DataFrame(protocol_data)

# 样式高亮函数：高亮当前激活的行
def highlight_active_row(row):
    is_active = row['Risk Level'] == final_level
    # 根据等级给高亮颜色
    if is_active:
        if final_level == 'RED':
            return ['background-color: #ffcccc; color: black; font-weight: bold'] * len(row)
        elif final_level == 'ORANGE':
            return ['background-color: #ffe5cc; color: black; font-weight: bold'] * len(row)
        else:
            return ['background-color: #ffffe0; color: black; font-weight: bold'] * len(row)
    else:
        return ['color: #999'] * len(row) # 非激活行变灰

# 展示表格
st.table(df_protocol.style.apply(highlight_active_row, axis=1))

# ==========================================
# 7. 地图部分 (保留原有的展示功能)
# ==========================================
if geojson:
    st.divider()
    st.subheader("🗺️ Geospatial Risk Distribution")
    st.caption("Visualization of risk across Malaysia states (Mock Data for Visualization)")
    # 这里保留你原本的 st.map 或者 pydeck 代码
    # 为了演示完整性，这里放一个简单的 placeholder
    map_data = pd.DataFrame({
        'lat': [4.2105, 3.1390, 1.5533, 5.9788],
        'lon': [101.9758, 101.6869, 110.3592, 116.0753],
        'risk': [np.random.rand() for _ in range(4)]
    })
    st.map(map_data)
else:
    st.warning("Map data (malaysia_states.json) not found. Skipping map visualization.")