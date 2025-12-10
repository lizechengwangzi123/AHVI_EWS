import streamlit as st
import pandas as pd
import plotly.express as px
import json
import google.generativeai as genai
import joblib
import numpy as np
import smtplib 
import ssl 
import pickle 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 

# ==============================================================================
# [CRITICAL] 用户自定义预测器类定义
# 必须放在 load 之前，否则 pickle 无法识别对象
# ==============================================================================

class SimplePredictor:
    def __init__(self, model_path, config_path):
        self.model = joblib.load(model_path)
        with open(config_path, "r") as f:
            self.config = json.load(f)
    
    def predict(self, EHD=None, HVI_Score=None, OVI=None):
        """
        只用3个核心特征预测，其他特征自动设为默认值
        """
        # 创建特征向量（使用默认值）
        features = np.zeros(len(self.config['feature_order']))
        
        # 设置默认值
        for i, feat in enumerate(self.config['feature_order']):
            if feat in self.config['default_values']:
                features[i] = self.config['default_values'][feat]
        
        # 设置用户输入的3个特征
        feat_dict = {'EHD': EHD, 'HVI_Score': HVI_Score, 'OVI': OVI}
        for feat_name, value in feat_dict.items():
            if value is not None and feat_name in self.config['feature_order']:
                idx = self.config['feature_order'].index(feat_name)
                features[idx] = value
        
        # 预测
        return self.model.predict([features])[0]

class NB2Predictor:
    """NB2模型预测器 - 只需输入HVI值"""
    
    def __init__(self):
        # 从保存的文件加载模型
        # 注意：如果是从 pickle 加载实例，__init__ 不会被调用，所以这里的文件路径不会报错
        self.model = joblib.load("article5_model.pkl")
        with open("best_model_details.json", "r") as f:
            self.info = json.load(f)
        
        # 固定默认值（基于你的输出）
        self.default_month = 6  # 默认6月
        self.default_time_index = 100  # 默认时间趋势
        self.default_shock_2021_10 = 0  # 默认非异常月
        self.default_shock_2022_03 = 0  # 默认非异常月
        
        # 特征顺序（非常重要！）
        self.feature_order = self.info['features']
    
    def create_feature_vector(self, hvi_value, month=None, time_index=None):
        """创建特征向量"""
        # 使用默认值或用户指定值
        month = month if month is not None else self.default_month
        time_index = time_index if time_index is not None else self.default_time_index
        
        # 初始化所有特征为0
        features = {feat: 0.0 for feat in self.feature_order}
        
        # 设置固定特征
        features['Intercept'] = 1.0
        features['HVI_LST_EHD_EQ'] = hvi_value
        features['time_index'] = time_index
        features['alpha'] = 0.002232  # 从你的输出中复制
        features['shock_2021_10'] = self.default_shock_2021_10
        features['shock_2022_03'] = self.default_shock_2022_03
        
        # 设置月份虚拟变量（one-hot编码）
        # 注意：1月是基准月，所以没有C(month)[T.1]
        for i in range(2, 13):
            month_col = f'C(month)[T.{i}]'
            if month_col in features:
                features[month_col] = 1.0 if i == month else 0.0
        
        return features
    
    def predict(self, hvi_value, month=None, time_index=None):
        """预测函数 - 只需输入HVI值"""
        # 创建特征向量
        features = self.create_feature_vector(hvi_value, month, time_index)
        
        # 按顺序转换为数组
        X = np.array([[features[feat] for feat in self.feature_order]])
        
        # 预测
        return float(self.model.predict(X)[0])
    
    def batch_predict(self, hvi_values, months=None, time_indices=None):
        """批量预测"""
        if months is None:
            months = [self.default_month] * len(hvi_values)
        if time_indices is None:
            time_indices = [self.default_time_index] * len(hvi_values)
        
        results = []
        for hvi, month, time_idx in zip(hvi_values, months, time_indices):
            results.append(self.predict(hvi, month, time_idx))
        
        return results
    
    def get_feature_importance(self):
        """获取特征重要性（系数绝对值）"""
        return {
            feat: abs(coef) 
            for feat, coef in self.info['coefficients'].items()
        }

# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="AHVI+ EWS & Research Dashboard",
    page_icon="🔬",
    layout="wide"
)

# --- API & Model Loading ---
@st.cache_resource
def load_risk_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

risk_model = load_risk_model('champion_model_pipeline.joblib')

# [UPDATED] Loaders with robust error handling
@st.cache_resource
def load_article_models():
    models = {}
    
    def load_safe(path):
        try:
            # First try joblib
            return joblib.load(path)
        except:
            try:
                # Fallback to pickle
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"❌ Failed to load {path}. Error: {e}")
                return None

    models['article5'] = load_safe('article5_predictor.pkl')
    models['article6'] = load_safe('article6_predictor.pkl')
    return models

article_models = load_article_models()

try:
    genai.configure(api_key=st.secrets["api_keys"]["gemini_api_key"])
    API_AVAILABLE = True
except (KeyError, AttributeError):
    API_AVAILABLE = False

# --- Data Loading and Processing Functions ---
@st.cache_data
def load_data(filepath, geojson_path):
    try:
        df = pd.read_csv(filepath)
        with open(geojson_path) as f:
            geojson = json.load(f)
        return df, geojson
    except FileNotFoundError as e:
        st.error(f"File not found: {e}.")
        return None, None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- [NEW] Updated Logic: Rule-Based Override (Equation 1) ---
def assign_risk_level_enhanced(predicted_outcome, thresholds, utci_val):
    """
    Assigns risk level with Rule-Based Override:
    If UTCI > 46, force RED (Equation 1).
    Otherwise use percentile thresholds.
    """
    if utci_val > 46.0:
        return "RED", "Critical Risk (Heatwave Override)"
    
    if predicted_outcome > thresholds[0.9]: return "RED", "Critical Risk"
    if predicted_outcome > thresholds[0.75]: return "ORANGE", "High Risk"
    if predicted_outcome > thresholds[0.5]: return "YELLOW", "Medium Risk"
    return "GREEN", "Low Risk"

# --- Main App Logic ---
st.title("🔬 AI-Enhanced Heat Vulnerability Index+ EWS & Research Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("**AHVI+ Early Warning System**")
st.sidebar.markdown("*AI-Powered by Google Gemini*")
st.sidebar.markdown("Developed by Zecheng Li")

df, geojson = load_data("risk_assessment_final.csv", "malaysia_states.json")

if df is not None and geojson is not None:
    # --- Data Cleaning & Pre-calculation ---
    name_mapping = {
        "Kuala Lumpur": "KualaLumpur", "W.P. Putrajaya": "Putrajaya", "W.P. Labuan": "Labuan",
        "Pulau Pinang": "Panang", "Negeri Sembilan": "NegeriSembilan", "Terengganu": "Trengganu"
    }
    df['district_id'] = df['district_id'].replace(name_mapping)
    df = df.sort_values(by="risk_factor_score", ascending=False).reset_index(drop=True)

    if 'UTCI_mean' in df.columns and 'PM25_mean' in df.columns:
        df['exposure_score'] = (df['UTCI_mean'] - df['UTCI_mean'].mean()) / df['UTCI_mean'].std() + \
                               (df['PM25_mean'] - df['PM25_mean'].mean()) / df['PM25_mean'].std() + \
                               (df['O3_q95'] - df['O3_q95'].mean()) / df['O3_q95'].std()
    
    min_score, max_score = df['risk_factor_score'].min(), df['risk_factor_score'].max()
    if (max_score - min_score) != 0:
        df['risk_score_for_size'] = 5 + ((df['risk_factor_score'] - min_score) / (max_score - min_score)) * 45
    else:
        df['risk_score_for_size'] = 20

    prediction_thresholds = df['predicted_outcome'].quantile([0.5, 0.75, 0.9]).to_dict()
    
    # --- [NEW] Calculate History Stats for Anomaly Detection (Equation 2) ---
    hist_mean = df['predicted_outcome'].mean()
    hist_std = df['predicted_outcome'].std()

    # --- Sidebar ---
    st.sidebar.header("Filter & Actions")
    risk_levels = df['risk_level'].unique()
    selected_levels = st.sidebar.multiselect('Filter by risk level:', options=risk_levels, default=risk_levels)
    df_filtered = df[df['risk_level'].isin(selected_levels)]
    district_list = sorted(df_filtered['district_id'].unique())
    selected_district = st.sidebar.selectbox('Select a district:', options=district_list)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Data")
    csv_data = convert_df_to_csv(df_filtered)
    st.sidebar.download_button(
        label="📥 Download Filtered Data as CSV", data=csv_data,
        file_name=f'heat_risk_data_{pd.Timestamp.now().strftime("%Y%m%d")}.csv', mime='text/csv'
    )

    # ==================== FEEDBACK FUNCTION ====================
    st.sidebar.markdown("---")
    st.sidebar.subheader("✉️ Provide Feedback")
    feedback_text = st.sidebar.text_area("Share your feedback or report a bug:")
    if st.sidebar.button("Send Feedback"):
        if feedback_text:
            try:
                sender_email = st.secrets["email_credentials"]["sender_email"]
                password = st.secrets["email_credentials"]["sender_password"]
                receiver_email = "23053789@siswa.um.edu.my"
                subject = "Feedback from AHVI+ EWS Dashboard"
                body = f"User Feedback:\n\n{feedback_text}"
                message = f"Subject: {subject}\n\n{body}"
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, message.encode('utf-8'))
                st.sidebar.success("Thank you! Your feedback has been sent successfully.")
            except KeyError:
                st.sidebar.error("Email credentials are not configured in the app's secrets.")
            except Exception as e:
                st.sidebar.error(f"Failed to send feedback. Error: {e}")
        else:
            st.sidebar.warning("Please enter your feedback before sending.")

    # --- Main content area ---
    tab_list = [
        "📍 Overview & Map", 
        "🔬 District Deep Dive & Simulator",
        "📊 Risk Deconstruction",
        "🕹️ On-Demand Prediction",
        "🤖 AI Policy Advisor"
    ]
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = tab_list[0]

    st.session_state.active_tab = st.radio(
        "Navigation", tab_list, horizontal=True, label_visibility="collapsed"
    )

    if st.session_state.active_tab == "📍 Overview & Map":
        st.header("📈 At a Glance")
        col1, col2, col3 = st.columns(3)
        highest_risk_district = df.iloc[0]['district_id']
        col1.metric("Highest Risk District", highest_risk_district)
        high_risk_count = df[df['risk_level'] != 'GREEN'].shape[0]
        col2.metric("Districts Under Alert", f"{high_risk_count}")
        avg_risk_score = df['risk_factor_score'].mean()
        col3.metric("Average Risk Factor Score", f"{avg_risk_score:.2f}")
        st.markdown("---")
        st.header("Geographic Risk Distribution Map")
        fig_map = px.choropleth_mapbox(df_filtered, geojson=geojson, locations='district_id',
                                       featureidkey="properties.NAME_1", color='risk_factor_score',
                                       color_continuous_scale="Reds", mapbox_style="carto-positron",
                                       zoom=5.5, center={"lat": 4.2105, "lon": 108.27},
                                       opacity=0.6, hover_name='district_id',
                                       hover_data={'risk_level': True, 'risk_factor_score': ':.2f'})
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

        # =========================================================================
        # [NEW SECTION 1] Socio-economic / Traditional HVI Calculator
        # =========================================================================
        st.markdown("---")
        st.subheader("🧮 Socio-economic & Traditional HVI Calculator")
        
        # 1. Select number of indicators
        num_indicators = st.number_input("Enter number of indicators:", min_value=1, max_value=20, value=6, step=1)
        
        st.write("Please fill in the indicators below:")
        
        # 2. Generate dynamic inputs
        input_data_list = []
        cols = st.columns(2) # 2 columns layout for Name | Value
        
        # Creating a container to hold the inputs
        with st.container():
            for i in range(int(num_indicators)):
                c1, c2 = st.columns([1, 1])
                with c1:
                    ind_name = st.text_input(f"Indicator {i+1} Name", key=f"ind_name_{i}")
                with c2:
                    ind_val = st.number_input(f"Indicator {i+1} Value", key=f"ind_val_{i}", format="%.4f")
                input_data_list.append({'name': ind_name, 'value': ind_val})

        # 3. Special EHD Button
        st.markdown("#### Special Configuration")
        col_special_btn, col_ehd_input = st.columns([1, 2])
        
        use_ehd_calc = st.checkbox("⚙️ * Enable EHD-HVI Mode (Click only for EHD-HVI calculation)", help="Only check this if you are performing EHD-HVI analysis.")
        
        ehd_value_input = None
        if use_ehd_calc:
            ehd_value_input = st.number_input("Extreme Heat Days:", value=0.0)

        # 4. Calculation Button and Logic
        if st.button("Calculate Traditional HVI/EHD-HVI"):
            values = [item['value'] for item in input_data_list]
            
            if not values:
                st.error("Please enter indicator values.")
            else:
                if not use_ehd_calc:
                    # --- CASE A: PCA Calculation (Simulated) ---
                    st.info("Computing HVI using PCA (Principal Component Analysis)...")
                    try:
                        # Fallback for single-row PCA: Use Equal Weight / Mean as proxy for PC1
                        hvi_result = np.mean(values) 
                        st.warning("Note: True PCA requires a full population dataset. Using simplified composite score.")
                        st.success(f"Calculated Traditional HVI Score: {hvi_result:.4f}")
                    except Exception as e:
                        st.error(f"Error in calculation: {e}")

                else:
                    # --- CASE B: Equal Weight (EHD-HVI) ---
                    st.info("Computing HVI using Equal Weighting (EHD-HVI Mode)...")
                    hvi_result = np.mean(values)
                    st.success(f"Calculated EHD-HVI Score (Equal Weight): {hvi_result:.4f}")

        # =========================================================================
        # [NEW SECTION 2] Extra Indicators Considerations
        # =========================================================================
        st.markdown("---")
        st.subheader("Extra indicators considerations")
        
        # 1. Selection Container
        analysis_type = st.radio("Select Analysis Type:", 
                                 ["Extreme Heat Days", "Organizational Vulnerability Index"])
        
        # Container for inputs
        pred_inputs = {}
        
        if analysis_type == "Extreme Heat Days":
            c_ehd1, c_ehd2 = st.columns(2)
            with c_ehd1:
                pred_inputs['HVI_Score'] = st.number_input("EHD-HVI", format="%.4f")
            with c_ehd2:
                pred_inputs['City'] = st.text_input("City Name (English)")
                
        else: # Organizational Vulnerability Index
            c_ovi1, c_ovi2 = st.columns(2)
            c_ovi3, c_ovi4 = st.columns(2)
            with c_ovi1:
                pred_inputs['HVI_Score'] = st.number_input("Traditional HVI", format="%.4f")
            with c_ovi2:
                pred_inputs['EHD'] = st.number_input("Extreme Heat Days (Int)", min_value=0, step=1, format="%d")
            with c_ovi3:
                pred_inputs['OVI'] = st.number_input("Organizational Vulnerability Index", format="%.4f")
            with c_ovi4:
                pred_inputs['City'] = st.text_input("City Name (English)")

        # Map Placeholder (Default KL)
        city_coords = {
            "Kuala Lumpur": {"lat": 3.1390, "lon": 101.6869},
            "Penang": {"lat": 5.4141, "lon": 100.3288},
            "George Town": {"lat": 5.4141, "lon": 100.3288},
            "Johor Bahru": {"lat": 1.4927, "lon": 103.7414},
            "Kuching": {"lat": 1.5533, "lon": 110.3592},
            "Kota Kinabalu": {"lat": 5.9804, "lon": 116.0735},
            "Ipoh": {"lat": 4.5975, "lon": 101.0901},
            "Shah Alam": {"lat": 3.0738, "lon": 101.5183},
            "Petaling Jaya": {"lat": 3.1073, "lon": 101.6067},
            "Melaka": {"lat": 2.1896, "lon": 102.2501},
            "Alor Setar": {"lat": 6.1248, "lon": 100.3678},
            "Kuantan": {"lat": 3.8077, "lon": 103.3260}
        }

        map_center = {"lat": 3.1390, "lon": 101.6869} # Default KL
        map_zoom = 10
        target_city = pred_inputs.get('City', '').strip()
        
        if target_city:
            found = False
            for k, v in city_coords.items():
                if k.lower() == target_city.lower():
                    map_center = v
                    found = True
                    break
            if not found:
                st.caption(f"⚠️ City '{target_city}' not in demo database. Map remaining at default.")
            else:
                st.caption(f"📍 Map centered on {target_city}")

        st.markdown(f"**Map Visualization: {target_city if target_city else 'Kuala Lumpur'}**")
        map_df = pd.DataFrame([{'lat': map_center['lat'], 'lon': map_center['lon'], 'City': target_city if target_city else "Kuala Lumpur"}])
        fig_city_map = px.scatter_mapbox(map_df, lat='lat', lon='lon', hover_name='City', zoom=map_zoom, height=300)
        fig_city_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_city_map, use_container_width=True)

        # 5. Calculate Button for Mortality & Risk
        if st.button("Predict Mortality & Risk Level"):
            mortality_pred = None
            
            # --- Prediction Logic (Updated for Custom Classes) ---
            if analysis_type == "Extreme Heat Days":
                if article_models['article5']:
                    try:
                        # Article 5 (NB2Predictor) uses: .predict(hvi_value=...)
                        # 不需要构造 array，直接传参数
                        mortality_pred = article_models['article5'].predict(hvi_value=pred_inputs['HVI_Score'])
                    except Exception as e:
                        st.error(f"Prediction Error (Article 5 Model): {e}")
                else:
                    st.error("Article 5 Model (article5_predictor.pkl) failed to load.")
            
            else: # OVI
                if article_models['article6']:
                    try:
                        # Article 6 (SimplePredictor) uses: .predict(EHD=, HVI_Score=, OVI=)
                        # 不需要构造 array，直接传参数
                        mortality_pred = article_models['article6'].predict(
                            EHD=pred_inputs['EHD'], 
                            HVI_Score=pred_inputs['HVI_Score'], 
                            OVI=pred_inputs['OVI']
                        )
                    except Exception as e:
                        st.error(f"Prediction Error (Article 6 Model): {e}")
                else:
                    st.error("Article 6 Model (article6_predictor.pkl) failed to load.")
            
            # --- Risk Level Calculation Logic (Adapted) ---
            if mortality_pred is not None:
                st.subheader("Prediction Results")
                st.metric("Predicted Mortality", f"{mortality_pred:.4f}")
                
                baseline_mortality = df['predicted_outcome']
                q90 = baseline_mortality.quantile(0.9)
                q75 = baseline_mortality.quantile(0.75)
                q50 = baseline_mortality.quantile(0.5)
                
                risk_level_final = "GREEN"
                if mortality_pred > q90:
                    risk_level_final = "RED"
                elif mortality_pred > q75:
                    risk_level_final = "ORANGE"
                elif mortality_pred > q50:
                    risk_level_final = "YELLOW"
                
                color_map = {"RED": "red", "ORANGE": "orange", "YELLOW": "#FFD700", "GREEN": "green"}
                st.markdown(f"### Heat Risk Level: <span style='color:{color_map[risk_level_final]}'>{risk_level_final}</span>", unsafe_allow_html=True)
                
                if risk_level_final == "RED":
                    st.error(f"🚨 CRITICAL ALERT for {target_city}: High expected mortality. Initiate Emergency Protocol.")
                elif risk_level_final == "ORANGE":
                    st.warning(f"⚠️ WARNING for {target_city}: Elevated risk. Alert vulnerable organizations.")
                elif risk_level_final == "YELLOW":
                    st.warning(f"✋ ADVISORY for {target_city}: Moderate risk detected.")
                else:
                    st.success(f"✅ CLEAR: Normal risk levels for {target_city}.")

    elif st.session_state.active_tab == "🔬 District Deep Dive & Simulator":
        st.header(f"Detailed Analysis for: {selected_district}")
        district_data = df[df['district_id'] == selected_district].iloc[0]
        col_deep1, col_deep2, col_deep3 = st.columns(3)
        col_deep1.metric("Risk Level", district_data['risk_level'])
        col_deep2.metric("Risk Factor Score", f"{district_data['risk_factor_score']:.2f}")
        col_deep3.metric("Predicted Health Outcome", f"{district_data['predicted_outcome']:.2f}")
        st.subheader("Key Risk Factors Breakdown")
        risk_features = ['HVI_Score', 'UTCI_mean', 'PM25_mean', 'O3_q95']
        if all(feature in district_data for feature in risk_features):
            feature_values = district_data[risk_features].T.reset_index()
            feature_values.columns = ['Risk Factor', 'Value']
            fig_factors = px.bar(feature_values, x='Risk Factor', y='Value', color='Risk Factor', title=f"Contributing Factors for {selected_district}")
            st.plotly_chart(fig_factors, use_container_width=True)
        else:
            st.warning("Could not generate risk factor chart. Missing feature columns in the CSV.")
        st.markdown("---")
        st.header("🕹️ Intervention Simulator (What-If Analysis)")
        if risk_model is None:
            st.error("AI model (`champion_model_pipeline.joblib`) not found. Simulator is disabled.")
        else:
            st.info("Use the sliders to simulate the impact of policy interventions on this district's risk.")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                utci_change = st.slider(
                    "Simulate Change in UTCI (°C) (e.g., Cooling Policies)",
                    min_value=-50.0, max_value=50.0, value=0.0, step=0.1, key="sim_utci"
                )
            with sim_col2:
                pm25_change_percent = st.slider(
                    "Simulate Change in PM2.5 (%) (e.g., Pollution Control)",
                    min_value=-80, max_value=80, value=0, step=5, key="sim_pm25"
                )
            simulated_data = district_data[risk_features].to_frame().T
            original_outcome = district_data['predicted_outcome']
            simulated_data['UTCI_mean'] += utci_change
            simulated_data['PM25_mean'] *= (1 + pm25_change_percent / 100)
            simulated_outcome = risk_model.predict(simulated_data)[0]
            st.subheader("Simulation Results")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric(label="Original Predicted Health Outcome", value=f"{original_outcome:.2f}")
            res_col2.metric(label="Simulated Predicted Health Outcome", value=f"{simulated_outcome:.2f}",
                            delta=f"{simulated_outcome - original_outcome:.2f}")
            st.success("The 'delta' shows the predicted change. A negative value indicates a successful intervention.")

    elif st.session_state.active_tab == "📊 Risk Deconstruction":
        st.header("📊 Risk Deconstruction: Vulnerability vs. Exposure")
        st.info(
            """
            This chart helps disentangle the two key components of risk:
            - **Vulnerability (X-axis):** The long-term, underlying socio-economic and demographic factors of a district (represented by HVI Score). Higher values mean more inherent vulnerability.
            - **Exposure (Y-axis):** The short-term environmental threat, primarily from extreme heat (UTCI) and air pollution (PM2.5). Higher values mean a more severe immediate threat.
            """
        )
        if 'exposure_score' in df_filtered.columns:
            fig_scatter = px.scatter(
                df_filtered, x='HVI_Score', y='exposure_score', color='risk_level',
                size='risk_score_for_size', hover_name='district_id',
                color_discrete_map={'GREEN':'green', 'YELLOW':'gold', 'ORANGE':'orange', 'RED':'red'},
                labels={
                    "HVI_Score": "Inherent Vulnerability (HVI Score)",
                    "exposure_score": "Current Environmental Exposure Score",
                    "risk_level": "Overall Risk Level",
                    "risk_score_for_size": "Risk Score"
                },
                title="Deconstructing Risk into Vulnerability and Exposure"
            )
            fig_scatter.add_vline(x=df['HVI_Score'].mean(), line_dash="dash", annotation_text="Avg. Vulnerability")
            fig_scatter.add_hline(y=df['exposure_score'].mean(), line_dash="dash", annotation_text="Avg. Exposure")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Could not create exposure score. 'UTCI_mean' or 'PM25_mean' columns might be missing from the source data.")

    elif st.session_state.active_tab == "🕹️ On-Demand Prediction":
        st.header("🕹️ On-Demand Prediction for a New Scenario")
        
        if risk_model is None:
            st.error("AI model (`champion_model_pipeline.joblib`) not found. This feature is disabled.")
        else:
            st.info("Enter data for a scenario below. Includes Auto-Anomaly Detection and Protocol Override.")
            
            with st.form(key="prediction_form"):
                st.subheader("Enter Scenario Data")
                
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    district_name_input = st.text_input("District Name (e.g., 'New Urban Area')")
                    hvi_input = st.number_input("Heat Vulnerability Index (HVI_Score)", -20.0, 20.0, 0.5, 0.01)
                    utci_input = st.number_input("Mean Apparent Temperature (UTCI_mean)", -100.0, 100.0, 35.0, 0.1)

                with form_col2:
                    pm25_input = st.number_input("Mean PM2.5 Concentration (PM25_mean)", 0.0, 200.0, 40.0, 0.1)
                    o3_input = st.number_input("95th Percentile Ozone (O3_q95)", 0.0, 300.0, 100.0, 0.1)
                
                submit_button = st.form_submit_button(label="Generate Assessment Report")

            if submit_button:
                if not district_name_input.strip():
                    st.warning("Please enter a district name.")
                else:
                    input_data = pd.DataFrame({
                        'HVI_Score': [hvi_input], 'UTCI_mean': [utci_input],
                        'PM25_mean': [pm25_input], 'O3_q95': [o3_input]
                    })
                    
                    st.subheader(f"Assessment Report: {district_name_input}")
                    with st.spinner("🤖 Calling AI model for prediction..."):
                        predicted_outcome = risk_model.predict(input_data)[0]
                        
                        # --- [NEW] Use Enhanced Override Logic (Equation 1) ---
                        risk_color, risk_label = assign_risk_level_enhanced(predicted_outcome, prediction_thresholds, utci_input)
                        
                        st.metric(label="Predicted Health Risk Outcome", value=f"{predicted_outcome:.2f}")
                        st.metric(label="Assessed Risk Level", value=risk_label)
                        
                        if risk_color == "RED": st.error(f"**Alert: {risk_label}**")
                        elif risk_color == "ORANGE": st.warning(f"**Warning: {risk_label}**")
                        elif risk_color == "YELLOW": st.warning(f"**Caution: {risk_label}**")
                        else: st.success(f"**Assessment Clear: {risk_label}**")

                        # --- [NEW] Statistical Anomaly Detection (Equation 2) ---
                        if abs(predicted_outcome - hist_mean) > (3 * hist_std):
                            st.error(f"⚠️ **Anomaly Detected (Needs Human Assistance)**: Prediction deviates >3σ from historical mean.")
                        
                        # --- [NEW] Tiered Alert Dissemination Protocol ---
                        st.markdown("### 📡 Active Dissemination Protocol")
                        with st.expander(f"View Communication Strategy for {risk_color} Level", expanded=True):
                            if risk_color == "RED":
                                st.markdown("""
                                * **Target:** General Public, Emergency Services, Hospitals.
                                * **Channels:** National Broadcast, SMS Emergency Alerts, NGO Networks.
                                * **Message:** "Immediate danger! Seek cooling centers. Check vulnerable neighbors."
                                """)
                            elif risk_color == "ORANGE":
                                st.markdown("""
                                * **Target:** Vulnerable Groups (Elderly, Schools), Local Clinics.
                                * **Channels:** Local Radio, Community Apps, Clinic Alerts.
                                * **Message:** "High heat expected. Limit outdoor activity. Stay hydrated."
                                """)
                            elif risk_color == "YELLOW":
                                st.markdown("""
                                * **Target:** General Public.
                                * **Channels:** Social Media, Weather Apps, Public Boards.
                                * **Message:** "Advisory: Temperatures rising. Monitor updates."
                                """)
                            else:
                                st.markdown("*System Normal. Routine monitoring active.*")

                    st.markdown("---")
                    
                    if not API_AVAILABLE:
                        st.error("Gemini API not configured, cannot generate policy recommendations.")
                    else:
                        with st.spinner("🧠 Gemini is generating policy recommendations..."):
                            prompt = f"""
                            You are an expert public health advisor. Provide recommendations for {district_name_input}.
                            Data: Risk Level: {risk_label}, Outcome: {predicted_outcome:.2f}
                            Factors: HVI: {hvi_input}, UTCI: {utci_input}, PM2.5: {pm25_input}
                            """
                            # (Simplified prompt for brevity in this snippet)
                            try:
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                response = model.generate_content(prompt)
                                st.markdown("### 🤖 AI-Generated Policy Recommendations")
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"Gemini API Error: {e}")

    elif st.session_state.active_tab == "🤖 AI Policy Advisor":
        st.header(f"🤖 AI-Powered Policy Advisor for {selected_district}")
        if not API_AVAILABLE:
            st.error("Gemini API key not configured.")
        else:
            st.info(f"Click the button below to get AI-generated analysis for **{selected_district}**.")
            if st.button(f"Generate Recommendations for {selected_district}"):
                district_data = df[df['district_id'] == selected_district].iloc[0]
                prompt = f"""
                You are an expert public health advisor specializing in climate change.
                Analyze {selected_district}, Malaysia.
                Data: Risk Level: {district_data['risk_level']}, Risk Score: {district_data['risk_factor_score']:.2f}
                Factors: HVI: {district_data.get('HVI_Score', 'N/A')}, UTCI: {district_data.get('UTCI_mean', 'N/A')}, PM2.5: {district_data.get('PM25_mean', 'N/A')}
                
                Output:
                ### 1. Risk Summary
                ### 2. Primary Drivers
                ### 3. Actionable Recommendations
                """
                with st.spinner("🧠 Gemini is analyzing data..."):
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Gemini API Error: {e}")

else:
    st.error("Error loading data. Please check your file paths.")