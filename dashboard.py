import streamlit as st
import pandas as pd
import plotly.express as px
import json
import joblib
import numpy as np
import smtplib 
import ssl 
import pickle 
# [NEW] 必须导入这些库，否则无法读取模型文件
import statsmodels.api as sm 
import xgboost as xgb
# [NEW] 地理编码与Excel库
from geopy.geocoders import Nominatim 
from geopy.exc import GeocoderTimedOut
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
        features = np.zeros(len(self.config['feature_order']))
        for i, feat in enumerate(self.config['feature_order']):
            if feat in self.config['default_values']:
                features[i] = self.config['default_values'][feat]
        
        feat_dict = {'EHD': EHD, 'HVI_Score': HVI_Score, 'OVI': OVI}
        for feat_name, value in feat_dict.items():
            if value is not None and feat_name in self.config['feature_order']:
                idx = self.config['feature_order'].index(feat_name)
                features[idx] = value
        
        return self.model.predict([features])[0]

class NB2Predictor:
    """NB2模型预测器 - 只需输入HVI值"""
    
    def __init__(self):
        self.model = joblib.load("article5_model.pkl")
        with open("best_model_details.json", "r") as f:
            self.info = json.load(f)
        
        self.default_month = 6  
        self.default_time_index = 100  
        self.default_shock_2021_10 = 0  
        self.default_shock_2022_03 = 0  
        self.feature_order = self.info['features']
    
    def create_feature_vector(self, hvi_value, month=None, time_index=None):
        month = month if month is not None else self.default_month
        time_index = time_index if time_index is not None else self.default_time_index
        features = {feat: 0.0 for feat in self.feature_order}
        features['Intercept'] = 1.0
        features['HVI_LST_EHD_EQ'] = hvi_value
        features['time_index'] = time_index
        features['alpha'] = 0.002232  
        features['shock_2021_10'] = self.default_shock_2021_10
        features['shock_2022_03'] = self.default_shock_2022_03
        for i in range(2, 13):
            month_col = f'C(month)[T.{i}]'
            if month_col in features:
                features[month_col] = 1.0 if i == month else 0.0
        return features
    
    def predict(self, hvi_value, month=None, time_index=None):
        features = self.create_feature_vector(hvi_value, month, time_index)
        X = np.array([[features[feat] for feat in self.feature_order]])
        return float(self.model.predict(X)[0])

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

@st.cache_resource
def load_article_models():
    models = {}
    def load_safe(path):
        try:
            return joblib.load(path)
        except:
            try:
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
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["api_keys"]["gemini_api_key"])
    API_AVAILABLE = True
except (KeyError, AttributeError, ImportError):
    API_AVAILABLE = False

# --- Helper Functions for New Logic ---

def get_coords(city_name):
    """获取城市坐标"""
    geolocator = Nominatim(user_agent="ahvi_dashboard_v2")
    try:
        location = geolocator.geocode(city_name, timeout=5)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None, None

def calculate_seasonality_adjustment(df):
    """
    根据 Eq 1-4 计算 Pw_i
    Requires: 'date' (monthly), 'nighttime_light', 'population'
    """
    # Copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Check if required columns exist
    if not all(col in df.columns for col in ['date', 'nighttime_light', 'population']):
        st.error("Missing seasonality columns: date, nighttime_light, population")
        return df

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Eq 1: NTL March Baseline
    # Convert 'nighttime_light' to numeric, forcing errors to NaN
    df['nighttime_light'] = pd.to_numeric(df['nighttime_light'], errors='coerce')
    
    march_data = df[df['month'] == 3].set_index('year')['nighttime_light']
    yearly_mean_ntl = df.groupby('year')['nighttime_light'].mean()
    
    def get_ntl_march(row):
        y = row['year']
        # Try to get March data for that year
        if y in march_data.index and not pd.isna(march_data[y]):
            return march_data[y]
        # Fallback to yearly mean
        return yearly_mean_ntl.get(y, row['nighttime_light']) 
    
    df['NTL_March'] = df.apply(get_ntl_march, axis=1)
    
    # Eq 2: Seasonality Coefficient S^m
    # Avoid division by zero
    df['S_m'] = df['nighttime_light'] / df['NTL_March'].replace(0, 1)
    
    # Eq 3: Monthly Population
    df['Pop_Monthly'] = df['S_m'] * df['population']
    
    # Eq 4: Dynamic Population Coefficient Pw_i
    # Pw_i = Pop_Monthly_i / Mean(Pop_Monthly_All)
    mean_pop = df['Pop_Monthly'].mean()
    if mean_pop == 0: mean_pop = 1 # Avoid div by zero
    df['Pw_i'] = df['Pop_Monthly'] / mean_pop
    
    return df

def process_hvi_calculation(df, analysis_mode, apply_seasonality=False):
    """
    计算 HVI:
    1. 提取 Socio-economic 特征 (排除固定列)
    2. 标准化 (StandardScaler) -> 解决数值过大问题
    3. PCA 或 Equal Weight
    4. 如果是 Seasonality 模式，应用 HVI_Dynamic = HVI * Pw_i
    """
    df = df.copy()
    
    # 排除非HVI特征列
    exclude_cols = ['sheet_name', 'city', 'EHD', 'OVI', 'date', 'nighttime_light', 
                    'population', 'month', 'year', 'NTL_March', 'S_m', 'Pop_Monthly', 'Pw_i',
                    'lat', 'lon', 'Predicted_Mortality', 'Risk_Score', 'Risk_Level', 'Color']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if not feature_cols:
        st.error("未找到用于计算 HVI 的特征列。请检查 Excel 文件。")
        return df, []
    
    # [CRITICAL] 标准化输入数据 - 解决数值异常核心步骤
    scaler = StandardScaler()
    # Fill NA with mean to handle missing data gracefully
    X_input = df[feature_cols].fillna(df[feature_cols].mean())
    X_scaled = scaler.fit_transform(X_input)
    
    # 决定计算方法
    hvi_scores = []
    
    if "Extreme Heat Days" in analysis_mode:
        # Equal Weight (Mean of standardized features)
        hvi_scores = np.mean(X_scaled, axis=1)
    else:
        # PCA (OVI Mode) - Defaulting to PCA1
        try:
            pca = PCA(n_components=1) 
            hvi_scores = pca.fit_transform(X_scaled).flatten()
            # 简单的方向校正：假设 Feature 和 HVI 是正相关的
            # 如果 input 均值大 -> score 应该大。如果 correlation < 0 则翻转
            if np.corrcoef(np.mean(X_input, axis=1), hvi_scores)[0,1] < 0:
                hvi_scores = -hvi_scores
        except:
            hvi_scores = np.mean(X_scaled, axis=1) # Fallback to Equal Weight
            
    df['HVI_Score'] = hvi_scores
    
    # Eq 8: Dynamic HVI Adjustment
    if apply_seasonality and 'Pw_i' in df.columns:
        df['HVI_Final'] = df['HVI_Score'] * df['Pw_i']
    else:
        df['HVI_Final'] = df['HVI_Score']
        
    return df, feature_cols

def calculate_risk_score_and_level(df, analysis_mode):
    """
    计算 Predicted Mortality, Risk Score (Eq 1-3), 和 Risk Level (Eq Final)
    """
    df = df.copy()
    
    model = article_models['article5'] if "Extreme Heat Days" in analysis_mode else article_models['article6']
    
    if model is None:
        st.error("模型加载失败，无法计算风险。")
        return df
    
    # 1. Predict Mortality
    preds = []
    for idx, row in df.iterrows():
        hvi_val = row['HVI_Final']
        ehd_val = row.get('EHD', 0)
        ovi_val = row.get('OVI', 0)
        
        pred_val = 0
        try:
            if "Extreme Heat Days" in analysis_mode:
                # Article 5 (NB2): input HVI
                pred_val = model.predict(hvi_value=hvi_val)
            else:
                # Article 6 (Simple): input EHD, HVI, OVI
                pred_val = model.predict(EHD=ehd_val, HVI_Score=hvi_val, OVI=ovi_val)
        except Exception as e:
            # st.warning(f"Error predicting row {idx}: {e}")
            pred_val = 0
            
        preds.append(pred_val)
    
    df['Predicted_Mortality'] = preds
    
    # 2. Calculate Risk Score (Eq 1-3)
    # 选取关键驱动因子：HVI_Final, EHD, OVI (如果有)
    drivers = ['HVI_Final']
    if 'EHD' in df.columns: drivers.append('EHD')
    if 'OVI' in df.columns: drivers.append('OVI')
    
    # Eq 1: Standardize drivers (Z_ij) relative to THIS dataset
    # (对比这几个城市或这几个月的数据)
    scaler_risk = StandardScaler()
    X_drivers = df[drivers].fillna(0)
    
    if len(df) > 1:
        z_scores = scaler_risk.fit_transform(X_drivers)
    else:
        z_scores = np.zeros(X_drivers.shape) # Only 1 row, Z is 0
    
    # Eq 2: Weights 
    # 简单起见，假设各因子权重相等 (因为很难动态获取 pickle 模型的内部系数)
    n_drivers = len(drivers)
    weights = np.ones(n_drivers) / n_drivers # w_j = 1/n
    
    # Eq 3: Risk Score = Sum(w * Z)
    df['Risk_Score'] = np.dot(z_scores, weights)
    
    # 3. Calculate Risk Level (Dynamic Quantiles)
    # 基于当前数据集分布计算阈值
    if len(df) > 1:
        q_mort_90 = df['Predicted_Mortality'].quantile(0.90)
        q_mort_75 = df['Predicted_Mortality'].quantile(0.75)
        q_mort_50 = df['Predicted_Mortality'].quantile(0.50)
        
        q_risk_85 = df['Risk_Score'].quantile(0.85)
        q_risk_75 = df['Risk_Score'].quantile(0.75)
        q_risk_50 = df['Risk_Score'].quantile(0.50)
    else:
        # 单行数据无法计算分位数，给默认值或直接绿色
        q_mort_90 = q_mort_75 = q_mort_50 = float('inf')
        q_risk_85 = q_risk_75 = q_risk_50 = float('inf')
    
    levels = []
    colors = []
    
    for idx, row in df.iterrows():
        pm = row['Predicted_Mortality']
        rs = row['Risk_Score']
        
        if pm > q_mort_90 and rs > q_risk_85:
            lvl, col = "RED", "red"
        elif rs > q_risk_75 and pm > q_mort_75:
            lvl, col = "ORANGE", "orange"
        elif rs > q_risk_50 or pm > q_mort_50:
            lvl, col = "YELLOW", "#FFD700"
        else:
            lvl, col = "GREEN", "green"
        levels.append(lvl)
        colors.append(col)
        
    df['Risk_Level'] = levels
    df['Color'] = colors
    
    return df

# --- Data Loading and Processing Functions ---
@st.cache_data
def load_data(filepath, geojson_path):
    try:
        df = pd.read_csv(filepath)
        with open(geojson_path) as f:
            geojson = json.load(f)
        return df, geojson
    except FileNotFoundError as e:
        return None, None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- [NEW] Updated Logic: Rule-Based Override (Equation 1) ---
def assign_risk_level_enhanced(predicted_outcome, thresholds, utci_val):
    if utci_val > 46.0:
        return "RED", "Critical Risk (Heatwave Override)"
    if predicted_outcome > thresholds[0.9]: return "RED", "Critical Risk"
    if predicted_outcome > thresholds[0.75]: return "ORANGE", "High Risk"
    if predicted_outcome > thresholds[0.5]: return "YELLOW", "Medium Risk"
    return "GREEN", "Low Risk"

# --- Main App Logic ---
# [REQ] Update Header Information
st.title("🔬 AHVI+ Early Warning System")
st.markdown("**by Zecheng Li 李泽城 23053789 Universiti Malaya**")
st.markdown("*AI powered by Gemini*")
st.markdown("---")

df, geojson = load_data("risk_assessment_final.csv", "malaysia_states.json")

# --- Sidebar ---
st.sidebar.header("Filter & Actions")
# 仅当 df 加载成功时才显示 Filter
if df is not None:
    # Data Cleaning for Sidebar
    name_mapping = {
        "Kuala Lumpur": "KualaLumpur", "W.P. Putrajaya": "Putrajaya", "W.P. Labuan": "Labuan",
        "Pulau Pinang": "Panang", "Negeri Sembilan": "NegeriSembilan", "Terengganu": "Trengganu"
    }
    df['district_id'] = df['district_id'].replace(name_mapping)
    df = df.sort_values(by="risk_factor_score", ascending=False).reset_index(drop=True)
    
    # Pre-calc thresholds for On-Demand Tab
    if 'predicted_outcome' in df.columns:
        prediction_thresholds = df['predicted_outcome'].quantile([0.5, 0.75, 0.9]).to_dict()
        hist_mean = df['predicted_outcome'].mean()
        hist_std = df['predicted_outcome'].std()
    else:
        prediction_thresholds = {0.5: 0, 0.75: 0, 0.9: 0}
        hist_mean, hist_std = 0, 1

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

# Feedback
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

# --- Main Tabs ---
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

if df is not None and geojson is not None:
    
    # =========================================================================
    # TAB 1: Overview & Map
    # =========================================================================
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
        # [NEW REDESIGNED SECTION] Multi-City & Temporal Risk Analysis
        # =========================================================================
        st.markdown("---")
        st.header("🌍 Multi-City & Temporal Heat Risk Analysis")
        st.info("Upload Excel data to perform dynamic HVI calculation, seasonality adjustment, and comparative risk assessment.")

        # 1. Selection
        analysis_mode = st.radio("Select Analysis Scenario:", 
            ["Function 1: Multi-City Comparison (Same Time)", "Function 2: Single-City Time Series Analysis"])
        
        model_type = st.radio("Select Vulnerability Model:",
            ["Extreme Heat Days (EHD)", "Organizational Vulnerability Index (OVI)"])
        
        # 2. File Uploader
        uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=['xlsx'])
        
        if uploaded_file:
            try:
                xl = pd.ExcelFile(uploaded_file)
                sheet_names = xl.sheet_names
                
                final_df = pd.DataFrame()
                valid_upload = False
                
                # --- LOGIC FOR FUNCTION 1 (Multi-City) ---
                if analysis_mode.startswith("Function 1"):
                    if len(sheet_names) < 3:
                        st.error("❌ Comparison requires at least 3 cities (sheets).")
                    else:
                        dfs = []
                        ref_columns = None
                        error_flag = False
                        
                        for sheet in sheet_names:
                            # [REQ] Function 1: Read ONLY the first row
                            df_sheet = xl.parse(sheet, nrows=1)
                            
                            # Basic Column Check
                            if ref_columns is None:
                                ref_columns = list(df_sheet.columns)
                            elif list(df_sheet.columns) != ref_columns:
                                st.error(f"❌ Column mismatch in sheet '{sheet}'. All sheets must have identical structure.")
                                error_flag = True
                                break
                            
                            # Specific Column Check
                            req_cols = ['EHD'] if "Extreme" in model_type else ['EHD', 'OVI']
                            if not all(c in df_sheet.columns for c in req_cols):
                                st.error(f"❌ Missing required columns {req_cols} in sheet '{sheet}'.")
                                error_flag = True
                                break
                            
                            # Add City Identifier (Sheet Name or Column)
                            if 'city' not in df_sheet.columns:
                                df_sheet['city'] = sheet
                            
                            df_sheet['sheet_name'] = sheet
                            dfs.append(df_sheet)
                            
                        if not error_flag:
                            final_df = pd.concat(dfs, ignore_index=True)
                            valid_upload = True

                # --- LOGIC FOR FUNCTION 2 (Single City Time Series) ---
                else:
                    if len(sheet_names) < 1: st.error("File is empty.")
                    else:
                        # Only take first sheet
                        df_sheet = xl.parse(sheet_names[0])
                        target_city = sheet_names[0] # Default city name is sheet name
                        
                        # Check Columns
                        req_cols = ['date', 'nighttime_light', 'population']
                        extra_req = ['EHD'] if "Extreme" in model_type else ['EHD', 'OVI']
                        req_cols.extend(extra_req)
                        
                        if not all(c in df_sheet.columns for c in req_cols):
                            st.error(f"❌ Missing columns. Required: {req_cols}")
                        else:
                            if 'city' not in df_sheet.columns:
                                df_sheet['city'] = target_city
                            final_df = df_sheet
                            valid_upload = True
                
                # --- PROCESSING & VISUALIZATION ---
                if valid_upload:
                    st.success("✅ Data validated successfully. Processing...")
                    
                    # 1. Seasonality Adjustment (Only for Function 2)
                    apply_seas = False
                    if analysis_mode.startswith("Function 2"):
                        with st.spinner("Calculating Seasonality & Dynamic Population..."):
                            final_df = calculate_seasonality_adjustment(final_df)
                            apply_seas = True
                    
                    # 2. HVI Calculation (Normalization -> PCA/Equal -> Dynamic)
                    with st.spinner("Standardizing & Calculating HVI..."):
                        final_df, feature_cols = process_hvi_calculation(final_df, model_type, apply_seasonality=apply_seas)
                    
                    # 3. Risk Scoring & Leveling (Comparison against uploaded peers)
                    with st.spinner("Predicting Mortality & Assessing Risk Levels..."):
                        final_df = calculate_risk_score_and_level(final_df, model_type)
                    
                    # [REQ] Delete "Analysis Results" Table - Removed here.
                    
                    # 5. Map Visualization
                    st.subheader("🗺️ Risk Map")
                    
                    # Geocoding
                    lat_list = []
                    lon_list = []
                    coords_cache = {}
                    
                    with st.spinner("Looking up city coordinates..."):
                        for city in final_df['city']:
                            if city not in coords_cache:
                                lat, lon = get_coords(city)
                                coords_cache[city] = (lat, lon)
                            lat_list.append(coords_cache[city][0])
                            lon_list.append(coords_cache[city][1])
                        
                    final_df['lat'] = lat_list
                    final_df['lon'] = lon_list
                    
                    # Filter out un-geocoded cities
                    map_df = final_df.dropna(subset=['lat', 'lon'])
                    
                    # [REQ] Function 2 Slider Logic
                    plot_df = map_df.copy()
                    
                    if analysis_mode.startswith("Function 2"):
                        if 'date' in map_df.columns:
                            # Create readable string for slider
                            map_df['date_str'] = map_df['date'].dt.strftime('%Y-%m')
                            available_dates = sorted(map_df['date_str'].unique())
                            
                            if available_dates:
                                selected_date = st.select_slider("Select Time Period:", options=available_dates)
                                plot_df = map_df[map_df['date_str'] == selected_date]
                                st.caption(f"Showing Heat Risk Level for: {selected_date}")

                    if not plot_df.empty:
                        fig_new_map = px.scatter_mapbox(
                            plot_df, lat='lat', lon='lon', 
                            color='Risk_Level',
                            hover_name='city',
                            hover_data={'Predicted_Mortality':':.2f', 'HVI_Final':':.2f', 'Risk_Score':':.2f'},
                            color_discrete_map={'RED':'red', 'ORANGE':'orange', 'YELLOW':'gold', 'GREEN':'green'},
                            zoom=3, height=500
                        )
                        fig_new_map.update_layout(mapbox_style="carto-positron")
                        st.plotly_chart(fig_new_map, use_container_width=True)
                    else:
                        st.warning("Could not find coordinates for the provided cities. Map cannot be displayed.")

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # =========================================================================
    # TAB 2: District Deep Dive
    # =========================================================================
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

    # =========================================================================
    # TAB 3: Risk Deconstruction
    # =========================================================================
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
            min_score, max_score = df['risk_factor_score'].min(), df['risk_factor_score'].max()
            if (max_score - min_score) != 0:
                df_filtered['risk_score_for_size'] = 5 + ((df_filtered['risk_factor_score'] - min_score) / (max_score - min_score)) * 45
            else:
                df_filtered['risk_score_for_size'] = 20

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

    # =========================================================================
    # TAB 4: On-Demand Prediction
    # =========================================================================
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
                        
                        risk_color, risk_label = assign_risk_level_enhanced(predicted_outcome, prediction_thresholds, utci_input)
                        
                        st.metric(label="Predicted Health Risk Outcome", value=f"{predicted_outcome:.2f}")
                        st.metric(label="Assessed Risk Level", value=risk_label)
                        
                        if risk_color == "RED": st.error(f"**Alert: {risk_label}**")
                        elif risk_color == "ORANGE": st.warning(f"**Warning: {risk_label}**")
                        elif risk_color == "YELLOW": st.warning(f"**Caution: {risk_label}**")
                        else: st.success(f"**Assessment Clear: {risk_label}**")

                        if abs(predicted_outcome - hist_mean) > (3 * hist_std):
                            st.error(f"⚠️ **Anomaly Detected (Needs Human Assistance)**: Prediction deviates >3σ from historical mean.")
                        
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
                    
                    if API_AVAILABLE:
                        with st.spinner("🧠 Gemini is generating policy recommendations..."):
                            prompt = f"""
                            You are an expert public health advisor. Provide recommendations for {district_name_input}.
                            Data: Risk Level: {risk_label}, Outcome: {predicted_outcome:.2f}
                            Factors: HVI: {hvi_input}, UTCI: {utci_input}, PM2.5: {pm25_input}
                            """
                            try:
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                response = model.generate_content(prompt)
                                st.markdown("### 🤖 AI-Generated Policy Recommendations")
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"Gemini API Error: {e}")
                    else:
                        st.error("Gemini API not configured.")

    # =========================================================================
    # TAB 5: AI Policy Advisor
    # =========================================================================
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