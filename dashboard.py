import streamlit as st
import pandas as pd
import plotly.express as px
import json
import google.generativeai as genai
import joblib
import numpy as np
import smtplib # Added for sending email
import ssl # Added for sending email

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

# --- New helper function: Assign risk level for a single prediction ---
def assign_risk_level_single(predicted_outcome, thresholds):
    """Assigns a risk level to a single prediction outcome based on pre-calculated thresholds."""
    if predicted_outcome > thresholds[0.9]: return "RED", "Critical Risk"
    if predicted_outcome > thresholds[0.75]: return "ORANGE", "High Risk"
    if predicted_outcome > thresholds[0.5]: return "YELLOW", "Medium Risk"
    return "GREEN", "Low Risk"


# --- Main App Logic ---
st.title("🔬 AI-Enhanced Heat Vulnerability Index+ EWS & Research Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("**EHVI+ Early Warning System**")
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

    # ==================== NEW FEEDBACK FUNCTION START ====================
    st.sidebar.markdown("---")
    st.sidebar.subheader("✉️ Provide Feedback")

    feedback_text = st.sidebar.text_area("Share your feedback or report a bug:")

    if st.sidebar.button("Send Feedback"):
        if feedback_text:
            try:
                # --- Get credentials from secrets ---
                sender_email = st.secrets["email_credentials"]["sender_email"]
                password = st.secrets["email_credentials"]["sender_password"]
                receiver_email = "23053789@siswa.um.edu.my"

                # --- Construct the email ---
                subject = "Feedback from AI Heat-Health Dashboard"
                body = f"User Feedback:\n\n{feedback_text}"
                message = f"Subject: {subject}\n\n{body}"

                # --- Send the email ---
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
    # ===================== NEW FEEDBACK FUNCTION END =====================


    # --- Main content area with tabs ---
    tab_list = [
        "📍 Overview & Map", 
        "🔬 District Deep Dive & Simulator",
        "📊 Risk Deconstruction",
        "🕹️ On-Demand Prediction",
        "🤖 AI Policy Advisor"
    ]
    
    # --- FIX 1: Replace st.tabs with a stateful st.radio to act as tabs ---
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = tab_list[0] # Set default tab

    st.session_state.active_tab = st.radio(
        "Navigation", tab_list, horizontal=True, label_visibility="collapsed"
    )

    # --- FIX 2: Replace `with tabs[i]` with `if/elif` based on the active tab in session_state ---
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
                    min_value=-5.0, max_value=2.0, value=0.0, step=0.1, key="sim_utci"
                )
            with sim_col2:
                pm25_change_percent = st.slider(
                    "Simulate Change in PM2.5 (%) (e.g., Pollution Control)",
                    min_value=-50, max_value=20, value=0, step=5, key="sim_pm25"
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
            
            **This helps answer: Is a district high-risk because it's always vulnerable, or because it's facing an unusually severe event right now?**
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
            st.info("Enter data for a scenario below, and the model will predict its health risk and generate policy recommendations.")
            
            with st.form(key="prediction_form"):
                st.subheader("Enter Scenario Data")
                
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    district_name_input = st.text_input("District Name (e.g., 'New Urban Area')")
                    hvi_input = st.number_input(
                        "Heat Vulnerability Index (HVI_Score)",
                        min_value=-20.0, max_value=20.0, value=0.5, step=0.01,
                        help="A value between 0 and 1 representing socio-economic and demographic vulnerability. Higher is more vulnerable."
                    )
                    utci_input = st.number_input(
                        "Mean Apparent Temperature (UTCI_mean)",
                        min_value=-100.0, max_value=100.0, value=35.0, step=0.1,
                        help="A thermal comfort index (°C) that combines temperature, humidity, wind, and radiation."
                    )

                with form_col2:
                    pm25_input = st.number_input(
                        "Mean PM2.5 Concentration (PM25_mean)",
                        min_value=0.0, max_value=200.0, value=40.0, step=0.1,
                        help="Concentration of fine particulate matter in the air (µg/m³). Higher is more polluted."
                    )
                    o3_input = st.number_input(
                        "95th Percentile Ozone (O3_q95)",
                        min_value=0.0, max_value=300.0, value=100.0, step=0.1,
                        help="A high-concentration ozone indicator (µg/m³), reflecting photochemical smog levels."
                    )
                
                submit_button = st.form_submit_button(label="Generate Assessment Report")

            if submit_button:
                if not district_name_input.strip():
                    st.warning("Please enter a district name.")
                else:
                    input_data = pd.DataFrame({
                        'HVI_Score': [hvi_input],
                        'UTCI_mean': [utci_input],
                        'PM25_mean': [pm25_input],
                        'O3_q95': [o3_input]
                    })
                    
                    st.subheader(f"Assessment Report: {district_name_input}")
                    with st.spinner("🤖 Calling AI model for prediction..."):
                        predicted_outcome = risk_model.predict(input_data)[0]
                        risk_color, risk_label = assign_risk_level_single(predicted_outcome, prediction_thresholds)
                        
                        st.metric(label="Predicted Health Risk Outcome", value=f"{predicted_outcome:.2f}")
                        st.metric(label="Assessed Risk Level", value=risk_label)
                        if risk_color == "RED": st.error(f"**Alert: {risk_label}**")
                        elif risk_color == "ORANGE": st.warning(f"**Warning: {risk_label}**")
                        elif risk_color == "YELLOW": st.warning(f"**Caution: {risk_label}**")
                        else: st.success(f"**Assessment Clear: {risk_label}**")
                    
                    st.markdown("---")
                    
                    if not API_AVAILABLE:
                        st.error("Gemini API not configured, cannot generate policy recommendations.")
                    else:
                        with st.spinner("🧠 Gemini is generating policy recommendations based on your input..."):
                            prompt = f"""
                            You are an expert public health advisor...
                            """ # Note: prompt is truncated for brevity but remains the same in your code
                            try:
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                response = model.generate_content(prompt)
                                st.markdown("### 🤖 AI-Generated Policy Recommendations")
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"An error occurred while contacting the Gemini API: {e}")

    elif st.session_state.active_tab == "🤖 AI Policy Advisor":
        st.header(f"🤖 AI-Powered Policy Advisor for {selected_district}")
        if not API_AVAILABLE:
            st.error("Gemini API key not configured.")
        else:
            st.info(f"Click the button below to get AI-generated analysis for the pre-loaded data of **{selected_district}**.")
            if st.button(f"Generate Recommendations for {selected_district}"):
                district_data = df[df['district_id'] == selected_district].iloc[0]
                prompt = f"""
                You are an expert public health advisor...
                """ # Note: prompt is truncated for brevity but remains the same in your code
                with st.spinner("🧠 Gemini is analyzing data..."):
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception:
                        try:
                            model = genai.GenerativeModel('gemini-2.5-pro')
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"An error occurred while contacting the Gemini API: {e}")

else:
    st.error("Error loading data. Please check your file paths.")
