import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chisquare

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Little Rock Traffic Stops Analytics",
    page_icon="🚔",
    layout="wide"
)

# -----------------------------------------------------------------------------
# DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Attempt loading from zip or raw CSV
    try:
        df = pd.read_csv('yg821jf8611_ar_little_rock_2020_04_01.csv.zip')
    except Exception:
        df = pd.read_csv('ar_little_rock_2020_04_01.csv')
    
    # Clean & Transform Data
    df['date'] = pd.to_datetime(df['date'])
    df['time_dt'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
    df['hour'] = df['time_dt'].dt.hour
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['subject_race'] = df['subject_race'].str.lower().fillna('unknown')
    df['subject_sex'] = df['subject_sex'].str.lower().fillna('unknown')
    return df

df_raw = load_data()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Filter Controls")

# Date Filter
min_date, max_date = df_raw['date'].min().date(), df_raw['date'].max().date()
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Demographic Filters
races = ["All"] + sorted(list(df_raw['subject_race'].unique()))
selected_race = st.sidebar.selectbox("Subject Race", races)

genders = ["All"] + sorted(list(df_raw['subject_sex'].unique()))
selected_gender = st.sidebar.selectbox("Subject Sex", genders)

# Filter Application
df_filtered = df_raw.copy()
if len(selected_dates) == 2:
    start_d, end_d = selected_dates
    df_filtered = df_filtered[(df_filtered['date'].dt.date >= start_d) & (df_filtered['date'].dt.date <= end_d)]

if selected_race != "All":
    df_filtered = df_filtered[df_filtered['subject_race'] == selected_race]

if selected_gender != "All":
    df_filtered = df_filtered[df_filtered['subject_sex'] == selected_gender]

# -----------------------------------------------------------------------------
# DASHBOARD HEADER & KPIS
# -----------------------------------------------------------------------------
st.title("🚔 Little Rock, AR Traffic Stops Dashboard (2017)")
st.markdown("An executive analysis of police traffic stops, temporal patterns, and demographic disparities.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Stops Selected", f"{len(df_filtered):,}")
col2.metric("Peak Hour of Stops", f"{df_filtered['hour'].mode()[0]}:00" if not df_filtered.empty else "N/A")
col3.metric("Median Driver Age", f"{df_filtered['subject_age'].median():.0f} yrs" if not df_filtered.empty else "N/A")
col4.metric("Top Vehicle Type", df_filtered['vehicle_type'].mode()[0] if not df_filtered.empty else "N/A")

st.markdown("---")

# -----------------------------------------------------------------------------
# TAB NAVIGATION
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Executive Overview & Trends", 
    "⚖️ Demographic Disparity & Statistical Tests", 
    "📋 Data Explorer & Export"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Traffic Stop Patterns")
    
    t_col1, t_col2 = st.columns(2)
    
    with t_col1:
        # Monthly Volume
        monthly_df = df_filtered.groupby('month').size().reset_index(name='count')
        fig_monthly = px.line(
            monthly_df, x='month', y='count',
            title="Monthly Stop Volume (2017)",
            labels={'month': 'Month', 'count': 'Number of Stops'},
            markers=True
        )
        fig_monthly.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
    with t_col2:
        # Hourly Breakdown
        hourly_df = df_filtered.groupby('hour').size().reset_index(name='count')
        fig_hourly = px.bar(
            hourly_df, x='hour', y='count',
            title="Stops by Hour of Day",
            labels={'hour': 'Hour (24h)', 'count': 'Number of Stops'},
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        
    st.subheader("Driver Demographics Overview")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        # Race Breakdown Bar
        race_df = df_filtered['subject_race'].value_counts().reset_index()
        race_df.columns = ['subject_race', 'count']
        fig_race = px.bar(
            race_df, x='subject_race', y='count',
            title="Stop Count by Race",
            color='subject_race',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_race, use_container_width=True)
        
    with d_col2:
        # Age Histogram
        fig_age = px.histogram(
            df_filtered.dropna(subset=['subject_age']), 
            x='subject_age', 
            nbins=30,
            title="Age Distribution of Drivers",
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_age, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: DEMOGRAPHIC DISPARITY & HYPOTHESIS TESTING
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Demographic Benchmarking vs. US Census")
    st.write("""
    This analysis compares observed traffic stops against Little Rock, AR U.S. Census estimates 
    to evaluate whether specific racial groups are over- or under-represented in police stops.
    """)
    
    # Census baseline values for Little Rock
    census_baseline = {
        'black': 0.523,
        'white': 0.421,
        'asian/pacific islander': 0.031
    }
    
    # Compute observed counts for top 3 races
    known_df = df_filtered[df_filtered['subject_race'].isin(census_baseline.keys())]
    if not known_df.empty:
        obs_counts = known_df['subject_race'].value_counts()
        total_obs = len(known_df)
        
        # Build Summary Table
        disparity_data = []
        census_sum = sum(census_baseline.values())
        
        for race, count in obs_counts.items():
            obs_pct = (count / total_obs) * 100
            exp_pct = (census_baseline[race] / census_sum) * 100
            disparity_index = obs_pct / exp_pct
            
            disparity_data.append({
                "Race": race.title(),
                "Stops Count": count,
                "Stops Share (%)": round(obs_pct, 2),
                "Census Benchmark (%)": round(exp_pct, 2),
                "Disparity Index": round(disparity_index, 2),
                "Status": "Overrepresented" if disparity_index > 1.05 else ("Underrepresented" if disparity_index < 0.95 else "Proportional")
            })
            
        disp_df = pd.DataFrame(disparity_data)
        
        st.dataframe(disp_df, use_container_width=True)
        
        # Chi-Square Test
        expected_counts = [ (census_baseline[r]/census_sum) * total_obs for r in obs_counts.index ]
        chi2_stat, p_val = chisquare(f_obs=obs_counts.values, f_exp=expected_counts)
        
        st.markdown("### 🧪 Chi-Square Goodness-of-Fit Test")
        c1, c2, c3 = st.columns(3)
        c1.metric("Chi-Square Statistic", f"{chi2_stat:.2f}")
        c2.metric("p-value", f"{p_val:.4e}")
        c3.metric("Significance (α = 0.05)", "Statistically Significant" if p_val < 0.05 else "Not Significant")
        
        st.info("""
        **Methodological Note:** A statistically significant $p$-value ($p < 0.05$) indicates that the traffic stop demographics diverge from static resident census estimates. However, traffic stops reflect **roadway commuters**, which include regional commuters entering Little Rock during work hours.
        """)
        
        # Visualizing Comparison
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=disp_df['Race'], y=disp_df['Stops Share (%)'], name='Stops Share (%)', marker_color='#1f77b4'
        ))
        fig_comp.add_trace(go.Bar(
            x=disp_df['Race'], y=disp_df['Census Benchmark (%)'], name='Census Benchmark (%)', marker_color='#2ca02c'
        ))
        fig_comp.update_layout(
            barmode='group', 
            title="Traffic Stop Share vs. Census Population Share",
            yaxis_title="Percentage (%)"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: DATA EXPLORER
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Filtered Dataset Viewer")
    st.dataframe(df_filtered, use_container_width=True)
    
    # CSV Download Button
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="little_rock_filtered_traffic_stops.csv",
        mime="text/csv"
    )