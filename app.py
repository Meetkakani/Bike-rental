import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="DC Bike Analytics Pro",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['day_name'] = df['datetime'].dt.day_name()
    
    # Ordered days for plotting
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = pd.Categorical(df['day_name'], categories=days_order, ordered=True)
    
    # Map Season & Weather
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df['season_label'] = df['season'].map(season_map)
    
    weather_map = {
        1: 'Clear/Clouds', 2: 'Mist/Cloudy', 3: 'Light Snow/Rain', 4: 'Severe Weather'
    }
    df['weather_label'] = df['weather'].map(weather_map)
    
    return df

df = load_data()

# --- Sidebar Controls ---
st.sidebar.title("🛠️ Control Panel")

# 1. User Segment Selector (New Feature)
st.sidebar.subheader("1. Select User Segment")
target_option = st.sidebar.radio(
    "Which users do you want to analyze?",
    ["Total Rentals (count)", "Registered Users", "Casual Users"],
    index=0
)
# Map selection to column name
column_map = {
    "Total Rentals (count)": "count",
    "Registered Users": "registered",
    "Casual Users": "casual"
}
y_col = column_map[target_option]

# 2. Year Filter
st.sidebar.subheader("2. Time Period")
year_filter = st.sidebar.multiselect("Select Year(s)", [2011, 2012], default=[2011, 2012])

# Apply Filter
filtered_df = df[df['year'].isin(year_filter)]

# Sidebar Info
st.sidebar.info(f"Analyzing **{filtered_df.shape[0]}** hours of data for **{target_option}**.")
st.sidebar.markdown("---")
st.sidebar.markdown("Created for DC Bike Assignment")

# --- Main Dashboard Area ---
st.title(f"🚲 DC Bike Analysis: {target_option}")

# Top Level Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trips", f"{filtered_df[y_col].sum():,}")
c2.metric("Peak Hourly Volume", f"{filtered_df[y_col].max()}")
c3.metric("Average Hourly Volume", f"{filtered_df[y_col].mean():.1f}")
c4.metric("Most Active Season", filtered_df.groupby('season_label')[y_col].sum().idxmax())

st.markdown("---")

# Tabs for Organization
tab1, tab2, tab3 = st.tabs(["⏰ Temporal Patterns", "🌤️ Weather Impact", "📉 Correlation & Stats"])

# --- TAB 1: Temporal Patterns ---
with tab1:
    st.header("When do people ride?")
    
    col_t1, col_t2 = st.columns([2, 1])
    
    with col_t1:
        st.subheader("The 'Pulse' of the City (Heatmap)")
        st.write("Average rentals by Day of Week and Hour.")
        
        # Heatmap Data Preparation
        pivot_table = filtered_df.pivot_table(index='day_name', columns='hour', values=y_col, aggfunc='mean')
        
        # Plotting
        fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot_table, cmap="YlGnBu", annot=False, fmt=".0f", ax=ax_heat)
        ax_heat.set_xlabel("Hour of Day")
        ax_heat.set_ylabel("Day of Week")
        st.pyplot(fig_heat)
        
    with col_t2:
        st.subheader("Seasonal Daily Profile")
        st.write("Average hourly rentals split by season.")
        
        fig_season, ax_season = plt.subplots(figsize=(5, 5))
        sns.lineplot(data=filtered_df, x='hour', y=y_col, hue='season_label', estimator='mean', errorbar=None, ax=ax_season)
        ax_season.set_ylabel("Avg Rentals")
        ax_season.legend(title='Season', loc='upper left', fontsize='small')
        st.pyplot(fig_season)

# --- TAB 2: Weather Impact ---
with tab2:
    st.header("How does weather affect ridership?")
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        st.subheader("Temperature vs. Rentals")
        # Scatter with trendline
        fig_scat = plt.figure(figsize=(8, 6))
        sns.regplot(data=filtered_df, x='temp', y=y_col, scatter_kws={'alpha':0.1, 's':10}, line_kws={'color':'red'})
        plt.xlabel("Temperature (°C)")
        plt.ylabel(f"{target_option}")
        plt.title(f"Correlation: {filtered_df['temp'].corr(filtered_df[y_col]):.2f}")
        st.pyplot(fig_scat)
        
    with col_w2:
        st.subheader("Rentals by Weather Condition")
        # Boxplot for distribution
        fig_box = plt.figure(figsize=(8, 6))
        sns.boxplot(data=filtered_df, x='weather_label', y=y_col, palette="Set2")
        plt.xticks(rotation=45)
        plt.ylabel(f"{target_option}")
        st.pyplot(fig_box)

# --- TAB 3: Correlations & Stats ---
with tab3:
    st.header("Statistical Deep Dive")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Distribution of Rentals")
        fig_dist = plt.figure(figsize=(8, 5))
        sns.histplot(filtered_df[y_col], kde=True, bins=30, color="purple")
        plt.xlabel("Number of Rentals")
        plt.title(f"Distribution of {target_option}")
        st.pyplot(fig_dist)
        
    with col_s2:
        st.subheader("Numerical Correlations")
        numeric_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = plt.figure(figsize=(8, 5))
        sns.heatmap(corr_matrix[['count', 'casual', 'registered']].sort_values(by='count', ascending=False), 
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation with Target Variables")
        st.pyplot(fig_corr)

# --- Footer ---
st.markdown("---")
with st.expander("📂 View Raw Data"):
    st.dataframe(filtered_df.head(100))