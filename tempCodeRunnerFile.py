# ==========================================================
# Olympics Data Analysis Dashboard
# Author: ChatGPT
# Dataset: athlete_events.csv (Kaggle Olympics Dataset)
#
# Run:
# streamlit run app.py
#
# ==========================================================

# =========================
# Import Libraries
# =========================

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Olympics Dashboard",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Global Settings
# =========================

sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

/* Sidebar Blue Theme */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0b1f3a 0%,
        #123c69 50%,
        #0b1f3a 100%
    );
}

/* Sidebar Heading */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Sidebar Text */
[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Navigation Options */
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Hover */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px;
}

/* Search / Upload headings */
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Helper Functions
# =========================

data_path = Path(__file__).resolve().parent / "athlete_events.csv"


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    """
    Load the Olympics dataset from a local CSV file.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, low_memory=False)

    if not data_path.exists():
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    return pd.read_csv(data_path, low_memory=False)


def apply_global_search(df, search_text):
    """
    Filter rows across all columns using a text search.
    """
    if not search_text:
        return df

    mask = (
        df.astype(str)
        .apply(lambda row: row.str.contains(search_text, case=False, na=False).any(), axis=1)
    )
    filtered_df = df.loc[mask].copy()

    for col in ["Name", "Sex", "Season", "Sport", "Event", "Team", "NOC", "City", "Games", "Medal"]:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].fillna("Unknown")

    for col in ["Age", "Height", "Weight", "Year"]:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

    if "BMI" not in filtered_df.columns and "Height" in filtered_df.columns and "Weight" in filtered_df.columns:
        filtered_df["BMI"] = np.where(
            filtered_df["Height"] > 0,
            filtered_df["Weight"] / (filtered_df["Height"] / 100) ** 2,
            np.nan,
        )

    if "Medal" in filtered_df.columns:
        filtered_df["Medal"] = filtered_df["Medal"].fillna("No Medal")

    return filtered_df


def create_metric_card(container, title, value, icon="📌"):
    """
    Create a premium-looking metric card using custom HTML.
    """
    container.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def format_number(value):
    """
    Format numeric values for display.
    """
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)

def get_numeric_bounds(series):
    """
    Return (min, max) bounds for a numeric series.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (0, 100)
    return int(s.min()), int(s.max())

def build_summary_report(df):
    """
    Build a simple HTML report for export.
    """
    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Unique Athletes",
            "Unique Countries",
            "Unique Sports",
            "Unique Events",
            "Unique Games",
            "Unique Cities",
            "Gold Medals",
            "Silver Medals",
            "Bronze Medals",
            "Total Medal Winners",
            "Male Athletes",
            "Female Athletes",
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            df["Name"].nunique(),
            df["NOC"].nunique(),
            df["Sport"].nunique(),
            df["Event"].nunique(),
            df["Games"].nunique(),
            df["City"].nunique(),
            (df["Medal"] == "Gold").sum(),
            (df["Medal"] == "Silver").sum(),
            (df["Medal"] == "Bronze").sum(),
            (df["Medal"] != "No Medal").sum(),
            (df["Sex"] == "M").sum(),
            (df["Sex"] == "F").sum(),
        ]
    })
    return summary.to_html(index=False)

def reset_filters():
    """
    Reset all sidebar filter state.
    """
    keys = [
        "year_range",
        "focus_year",
        "season",
        "gender",
        "medal",
        "country",
        "team",
        "city",
        "sport",
        "event",
        "athlete_name",
        "age_range",
        "height_range",
        "weight_range",
    ]
    for key in keys:
        st.session_state.pop(key, None)
    st.rerun()

# =========================
# Load Data
# =========================

st.sidebar.markdown("### 📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# 🔍 Global Search
st.sidebar.markdown("### 🔍 Global Search")
search_text = st.sidebar.text_input(
    "Search anything",
    placeholder="e.g. USA, Swimming, Gold, 2016...",
)

with st.spinner("Loading Olympic dataset..."):
    df = load_data(uploaded_file)

if search_text:
    df = apply_global_search(df, search_text)

# Preprocess for display
if "Medal" in df.columns:
    df["Medal"] = df["Medal"].astype(str)

for col in ["Sex", "Season", "Sport", "Event", "Team", "NOC", "City", "Games", "Medal"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Choose a page",
        [
            "Executive Summary",
            "Data Overview",
            "Data Quality",
            "Visualizations",
            "Athlete Analysis",
            "Country Analysis",
            "Sports Analysis",
            "Medal Analysis",
            "Time Series",
            "Demographics",
            "Advanced Analytics",
            "Geography",
            "About",
        ],
        index=0,
        key="nav_page",
    )

    st.markdown("---")
    st.markdown("## 🔎 Filters")

    # Year range
    year_min = int(df["Year"].min()) if "Year" in df.columns else 1896
    year_max = int(df["Year"].max()) if "Year" in df.columns else 2020

    year_range = st.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        key="year_range",
    )

    # Focus year
    year_options = ["All"] + [str(int(y)) for y in sorted(df["Year"].dropna().astype(int).unique().tolist())]
    focus_year = st.selectbox(
        "Focus Year (Optional)",
        options=year_options,
        index=0,
        key="focus_year",
    )

    # Dynamic filter chain
    working_df = df[
        (df["Year"] >= year_range[0]) &
        (df["Year"] <= year_range[1])
    ]

    if focus_year != "All":
        working_df = working_df[working_df["Year"] == int(focus_year)]

    season_options = sorted(working_df["Season"].dropna().astype(str).unique().tolist())
    season = st.multiselect(
        "Season",
        options=season_options,
        default=season_options,
        key="season",
    )
    if season:
        working_df = working_df[working_df["Season"].isin(season)]

    gender_options = sorted(working_df["Sex"].dropna().astype(str).unique().tolist())
    gender = st.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options,
        key="gender",
    )
    if gender:
        working_df = working_df[working_df["Sex"].isin(gender)]

    medal_options = sorted(working_df["Medal"].dropna().astype(str).unique().tolist())
    medal = st.multiselect(
        "Medal",
        options=medal_options,
        default=medal_options,
        key="medal",
    )
    if medal:
        working_df = working_df[working_df["Medal"].isin(medal)]

    country_options = sorted(working_df["NOC"].dropna().astype(str).unique().tolist())
    country = st.multiselect(
        "Country (NOC)",
        options=country_options,
        default=country_options,
        key="country",
    )
    if country:
        working_df = working_df[working_df["NOC"].isin(country)]

    team_options = sorted(working_df["Team"].dropna().astype(str).unique().tolist())
    team = st.multiselect(
        "Team",
        options=team_options,
        default=team_options,
        key="team",
    )
    if team:
        working_df = working_df[working_df["Team"].isin(team)]

    city_options = sorted(working_df["City"].dropna().astype(str).unique().tolist())
    city = st.multiselect(
        "City",
        options=city_options,
        default=city_options,
        key="city",
    )
    if city:
        working_df = working_df[working_df["City"].isin(city)]

    sport_options = sorted(working_df["Sport"].dropna().astype(str).unique().tolist())
    sport = st.multiselect(
        "Sport",
        options=sport_options,
        default=sport_options,
        key="sport",
    )
    if sport:
        working_df = working_df[working_df["Sport"].isin(sport)]

    event_options = sorted(working_df["Event"].dropna().astype(str).unique().tolist())
    event = st.multiselect(
        "Event",
        options=event_options,
        default=event_options,
        key="event",
    )
    if event:
        working_df = working_df[working_df["Event"].isin(event)]

    athlete_name = st.text_input(
        "Athlete Name Search",
        key="athlete_name",
    )
    if athlete_name:
        working_df = working_df[
            working_df["Name"].astype(str).str.contains(athlete_name, case=False, na=False)
        ]

    age_min, age_max = get_numeric_bounds(working_df["Age"])
    age_range = st.slider(
        "Age Range",
        min_value=int(age_min),
        max_value=int(age_max),
        value=(int(age_min), int(age_max)),
        key="age_range",
    )
    if age_min != age_max:
        working_df = working_df[
            (working_df["Age"] >= age_range[0]) &
            (working_df["Age"] <= age_range[1])
        ]

    height_min, height_max = get_numeric_bounds(working_df["Height"])
    height_range = st.slider(
        "Height Range (cm)",
        min_value=int(height_min),
        max_value=int(height_max),
        value=(int(height_min), int(height_max)),
        key="height_range",
    )
    if height_min != height_max:
        working_df = working_df[
            (working_df["Height"] >= height_range[0]) &
            (working_df["Height"] <= height_range[1])
        ]

    weight_min, weight_max = get_numeric_bounds(working_df["Weight"])
    weight_range = st.slider(
        "Weight Range (kg)",
        min_value=int(weight_min),
        max_value=int(weight_max),
        value=(int(weight_min), int(weight_max)),
        key="weight_range",
    )
    if weight_min != weight_max:
        working_df = working_df[
            (working_df["Weight"] >= weight_range[0]) &
            (working_df["Weight"] <= weight_range[1])
        ]

    st.markdown("---")
    st.button("Reset Filters", on_click=reset_filters, use_container_width=True)

    st.markdown("---")
    st.download_button(
        label="Download Filtered Data",
        data=working_df.to_csv(index=False),
        file_name="filtered_olympics_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Final filtered dataframe
filtered_df = working_df.copy()

# =========================
# Main App
# =========================

def main():
    if filtered_df.empty:
        st.warning("No rows match the current filters. Please broaden your selection.")
        st.stop()

    # Header
    st.markdown(
        """
        <div class="hero-card">
            <h1>🏅 Olympics History </h1>
            <p>Explore more than 120 years of Olympic data with interactive analytics, advanced filtering, and premium visual storytelling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Showing {filtered_df.shape[0]:,} rows and {filtered_df.shape[1]} columns from the filtered dataset."
    )

    # Active filters summary
    with st.expander("🧠 Active Filters", expanded=False):
        filter_summary = {
            "Years": f"{year_range[0]} - {year_range[1]}",
            "Focus Year": focus_year,
            "Seasons": ", ".join(season) if season else "All",
            "Genders": ", ".join(gender) if gender else "All",
            "Medals": ", ".join(medal) if medal else "All",
            "Countries": ", ".join(country) if country else "All",
            "Teams": ", ".join(team) if team else "All",
            "Cities": ", ".join(city) if city else "All",
            "Sports": ", ".join(sport) if sport else "All",
            "Events": ", ".join(event) if event else "All",
            "Athlete Search": athlete_name if athlete_name else "None",
            "Age Range": f"{age_range[0]} - {age_range[1]}",
            "Height Range": f"{height_range[0]} - {height_range[1]} cm",
            "Weight Range": f"{weight_range[0]} - {weight_range[1]} kg",
        }
        st.write(filter_summary)

    # Page rendering
    if page == "Executive Summary":
        render_home_page(filtered_df)
    elif page == "Data Overview":
        render_data_overview_page(filtered_df)
    elif page == "Data Quality":
        render_data_quality_page(filtered_df)
    elif page == "Visualizations":
        render_visualization_page(filtered_df)
    elif page == "Athlete Analysis":
        render_athlete_page(filtered_df)
    elif page == "Country Analysis":
        render_country_page(filtered_df)
    elif page == "Sports Analysis":
        render_sports_page(filtered_df)
    elif page == "Medal Analysis":
        render_medal_page(filtered_df)
    elif page == "Time Series":
        render_time_series_page(filtered_df)
    elif page == "Demographics":
        render_demographics_page(filtered_df)
    elif page == "Advanced Analytics":
        render_advanced_page(filtered_df)
    elif page == "Geography":
        render_geography_page(filtered_df)
    elif page == "About":
        render_about_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="info-card">
            <strong>Built with Streamlit + Plotly + Pandas + Seaborn.</strong><br>
            Designed to feel like a modern business intelligence platform for Olympic data stories.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Page Components
# =========================

def render_home_page(df):
    """
    Executive summary dashboard with KPI cards and high-level charts.
    """
    st.header("📊 Executive Summary")

    with st.spinner("Preparing overview insights..."):
        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_metric_card(col1, "Total Athletes", format_number(df["Name"].nunique()), "🏃")
        with col2:
            create_metric_card(col2, "Total Countries", format_number(df["NOC"].nunique()), "🌍")
        with col3:
            create_metric_card(col3, "Total Sports", format_number(df["Sport"].nunique()), "🏀")
        with col4:
            create_metric_card(col4, "Total Events", format_number(df["Event"].nunique()), "🎯")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            create_metric_card(col5, "Olympic Games", format_number(df["Games"].nunique()), "🏛️")
        with col6:
            create_metric_card(col6, "Cities", format_number(df["City"].nunique()), "🏙️")
        with col7:
            create_metric_card(col7, "Gold Medals", format_number((df["Medal"] == "Gold").sum()), "🥇")
        with col8:
            create_metric_card(col8, "Silver Medals", format_number((df["Medal"] == "Silver").sum()), "🥈")

        col9, col10, col11, col12 = st.columns(4)
        with col9:
            create_metric_card(col9, "Bronze Medals", format_number((df["Medal"] == "Bronze").sum()), "🥉")
        with col10:
            create_metric_card(col10, "Total Medal Winners", format_number((df["Medal"] != "No Medal").sum()), "🏅")
        with col11:
            create_metric_card(col11, "Male Athletes", format_number((df["Sex"] == "M").sum()), "♂️")
        with col12:
            create_metric_card(col12, "Female Athletes", format_number((df["Sex"] == "F").sum()), "♀️")

        # Tabs for key sections
        tab1, tab2, tab3 = st.tabs(["Highlights", "Participation Trends", "Quick Comparisons"])

        with tab1:
            c1, c2 = st.columns(2)

            # Medal distribution
            medal_counts = df["Medal"].value_counts().reset_index()
            medal_counts.columns = ["Medal", "Count"]
            fig = px.pie(
                medal_counts,
                values="Count",
                names="Medal",
                title="Medal Distribution",
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.Plasma,
            )
            fig.update_layout(template="plotly_white")
            c1.plotly_chart(fig, use_container_width=True)

            # Top sports
            top_sports = (
                df["Sport"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            top_sports.columns = ["Sport", "Participants"]
            fig2 = px.bar(
                top_sports,
                x="Sport",
                y="Participants",
                title="Top 10 Sports",
                color="Participants",
                color_continuous_scale="Viridis",
            )
            fig2.update_layout(template="plotly_white")
            c2.plotly_chart(fig2, use_container_width=True)

        with tab2:
            # Participation over years
            year_stats = (
                df.groupby("Year")
                .agg(
                    Athletes=("Name", "nunique"),
                    Countries=("NOC", "nunique"),
                    Sports=("Sport", "nunique"),
                    Events=("Event", "nunique"),
                    Medals=("Medal", lambda s: (s != "No Medal").sum()),
                )
                .reset_index()
            )

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Athletes Over Years", "Countries Over Years"))
            fig.add_trace(go.Scatter(x=year_stats["Year"], y=year_stats["Athletes"], mode="lines+markers", name="Athletes"), row=1, col=1)
            fig.add_trace(go.Scatter(x=year_stats["Year"], y=year_stats["Countries"], mode="lines+markers", name="Countries"), row=1, col=2)
            fig.update_layout(title="Participation Trends", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Medal trend
            medal_year = (
                df[df["Medal"] != "No Medal"]
                .groupby(["Year", "Medal"])
                .size()
                .reset_index(name="Count")
            )
            fig3 = px.area(
                medal_year,
                x="Year",
                y="Count",
                color="Medal",
                title="Medal Trends Over Years",
            )
            fig3.update_layout(template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            # Gender comparison
            gender_counts = df["Sex"].value_counts().reset_index()
            gender_counts.columns = ["Sex", "Count"]
            fig = px.bar(
                gender_counts,
                x="Sex",
                y="Count",
                color="Sex",
                title="Gender Distribution",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Country comparison (top medal countries)
            medal_countries = (
                df[df["Medal"] != "No Medal"]
                .groupby("NOC")
                .size()
                .sort_values(ascending=False)
                .head(10)
                .reset_index(name="Medals")
            )
            medal_countries.columns = ["Country", "Medals"]
            fig2 = px.bar(
                medal_countries,
                x="Country",
                y="Medals",
                title="Top Medal-Winning Countries",
                color="Medals",
                color_continuous_scale="Cividis",
            )
            fig2.update_layout(template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

def render_data_overview_page(df):
    """
    Dataset overview: preview, shape, types, missing values, stats, unique values, memory usage.
    """
    st.header("📚 Data Overview")

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(df.head(40), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dataset Shape")
        st.write(f"Rows: {df.shape[0]:,}")
        st.write(f"Columns: {df.shape[1]}")

        st.markdown("### Memory Usage")
        st.write(f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    with col2:
        st.markdown("### Duplicate Rows")
        st.write(f"{df.duplicated().sum():,}")

        st.markdown("### Unique Values")
        unique_summary = pd.DataFrame({
            "Column": df.columns,
            "Unique Count": [df[col].nunique(dropna=False) for col in df.columns],
        })
        st.dataframe(unique_summary, use_container_width=True)

    with st.expander("Data Types", expanded=False):
        st.dataframe(pd.DataFrame({
            "Column": df.dtypes.index,
            "Data Type": df.dtypes.values,
        }), use_container_width=True)

    with st.expander("Missing Values", expanded=False):
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Missing Count": missing,
            "Missing %": missing_pct,
        }).sort_values("Missing Count", ascending=False)
        st.dataframe(missing_df, use_container_width=True)

    with st.expander("Statistical Summary", expanded=False):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

def render_data_quality_page(df):
    """
    Data cleaning and quality analysis page.
    """
    st.header("🧹 Data Quality Report")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Missing Values Analysis")
        missing = df.isna().sum().sort_values(ascending=False)
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values,
        })
        st.dataframe(missing_df, use_container_width=True)

    with col2:
        st.markdown("### Duplicate Analysis")
        st.metric("Duplicate Rows", df.duplicated().sum())

        # Outlier detection (IQR)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_rows = []
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_rows.append((col, count))
        outlier_df = pd.DataFrame(outlier_rows, columns=["Column", "IQR Outliers"])
        st.dataframe(outlier_df, use_container_width=True)

    with st.expander("Column Profiling", expanded=True):
        profiling = []
        for col in df.columns:
            s = df[col]
            profiling.append({
                "Column": col,
                "dtype": s.dtype,
                "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=False)),
                "min": s.min() if pd.api.types.is_numeric_dtype(s) else None,
                "max": s.max() if pd.api.types.is_numeric_dtype(s) else None,
                "mean": s.mean() if pd.api.types.is_numeric_dtype(s) else None,
            })
        st.dataframe(pd.DataFrame(profiling), use_container_width=True)

def render_visualization_page(df):
    """
    A collection of professional visualizations.
    """
    st.header("📈 Interactive Visualizations")

    tab1, tab2, tab3 = st.tabs(["Charts", "Advanced Charts", "Comparisons"])

    with tab1:
        c1, c2 = st.columns(2)

        # Bar chart
        sport_counts = df["Sport"].value_counts().head(15).reset_index()
        sport_counts.columns = ["Sport", "Count"]
        fig = px.bar(
            sport_counts,
            x="Sport",
            y="Count",
            title="Top Sports",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(template="plotly_white")
        c1.plotly_chart(fig, use_container_width=True)

        # Pie chart
        medal_counts = df["Medal"].value_counts().reset_index()
        medal_counts.columns = ["Medal", "Count"]
        fig2 = px.pie(
            medal_counts,
            values="Count",
            names="Medal",
            title="Medal Breakdown",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig2.update_layout(template="plotly_white")
        c2.plotly_chart(fig2, use_container_width=True)

        # Line chart
        year_counts = (
            df.groupby("Year")
            .size()
            .reset_index(name="Participants")
        )
        fig3 = px.line(
            year_counts,
            x="Year",
            y="Participants",
            markers=True,
            title="Participation Over Years",
        )
        fig3.update_layout(template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

        # Area chart
        medal_year = (
            df[df["Medal"] != "No Medal"]
            .groupby(["Year", "Medal"])
            .size()
            .reset_index(name="Count")
        )
        fig4 = px.area(
            medal_year,
            x="Year",
            y="Count",
            color="Medal",
            title="Medals Over Years",
        )
        fig4.update_layout(template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        # Scatter plot
        scatter_df = df.dropna(subset=["Age", "Height", "Weight"])
        fig = px.scatter(
            scatter_df,
            x="Height",
            y="Weight",
            color="Sex",
            size="Age",
            hover_data=["Name", "Sport"],
            title="Height vs Weight",
            opacity=0.7,
        )
        fig.update_layout(template="plotly_white")
        c1.plotly_chart(fig, use_container_width=True)

        # Bubble chart
        fig2 = px.scatter(
            scatter_df,
            x="Age",
            y="Height",
            size="Weight",
            color="Medal",
            hover_data=["Name", "Sport"],
            title="Age vs Height (Bubble Chart)",
            opacity=0.7,
        )
        fig2.update_layout(template="plotly_white")
        c2.plotly_chart(fig2, use_container_width=True)

        # Histogram
        fig3 = px.histogram(
            df,
            x="Age",
            nbins=35,
            color="Sex",
            title="Age Distribution",
            marginal="box",
        )
        fig3.update_layout(template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

        # Box plot
        fig4 = px.box(
            df,
            x="Sex",
            y="Height",
            color="Sex",
            title="Height by Gender",
        )
        fig4.update_layout(template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)

        # Violin plot
        fig5 = px.violin(
            df,
            x="Sex",
            y="Weight",
            color="Sex",
            box=True,
            title="Weight Distribution by Gender",
        )
        fig5.update_layout(template="plotly_white")
        st.plotly_chart(fig5, use_container_width=True)

        # Heatmap
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[numeric_cols].corr().fillna(0)
        fig6 = px.imshow(
            corr,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
        )
        fig6.update_layout(template="plotly_white")
        st.plotly_chart(fig6, use_container_width=True)

    with tab3:
        # Treemap
        country_sports = (
            df.groupby(["NOC", "Sport"])
            .size()
            .reset_index(name="Count")
            .head(80)
        )
        fig = px.treemap(
            country_sports,
            path=[px.Constant("Olympics"), "NOC", "Sport"],
            values="Count",
            title="Treemap: Countries and Sports",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Sunburst
        if not df.empty:
            sunburst_df = (
                df.groupby(["Season", "Sport", "Event"])
                .size()
                .reset_index(name="Count")
                .head(120)
            )
            fig2 = px.sunburst(
                sunburst_df,
                path=["Season", "Sport", "Event"],
                values="Count",
                title="Sunburst: Season → Sport → Event",
            )
            fig2.update_layout(template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

        # Radar chart (top 5 countries by medals)
        top_countries = (
            df[df["Medal"] != "No Medal"]
            .groupby("NOC")
            .size()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

        radar_df = (
            df[df["NOC"].isin(top_countries)]
            .groupby(["NOC", "Medal"])
            .size()
            .reset_index(name="Count")
        )
        if not radar_df.empty:
            categories = ["Gold", "Silver", "Bronze"]
            fig3 = go.Figure()
            for country in top_countries:
                vals = []
                for medal in categories:
                    vals.append(int(radar_df[(radar_df["NOC"] == country) & (radar_df["Medal"] == medal)]["Count"].sum()))
                vals += vals[:1]
                fig3.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=country,
                ))
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Radar Chart: Medal Distribution by Country",
                template="plotly_white",
            )
            st.plotly_chart(fig3, use_container_width=True)

def render_athlete_page(df):
    """
    Athlete-focused analysis.
    """
    st.header("🏃 Athlete Analysis")

    # Top athletes by medals
    athlete_summary = (
        df[df["Medal"] != "No Medal"]
        .groupby("Name")
        .agg(
            Medals=("Medal", "count"),
            Gold=("Medal", lambda s: (s == "Gold").sum()),
            Silver=("Medal", lambda s: (s == "Silver").sum()),
            Bronze=("Medal", lambda s: (s == "Bronze").sum()),
            Sports=("Sport", "nunique"),
            Events=("Event", "nunique"),
        )
        .reset_index()
        .sort_values(["Medals", "Gold", "Silver", "Bronze"], ascending=False)
        .head(20)
    )

    st.subheader("Top Athletes by Medal Count")
    st.dataframe(athlete_summary, use_container_width=True)

    fig = px.bar(
        athlete_summary.head(10),
        x="Name",
        y="Medals",
        color="Medals",
        title="Top 10 Athletes",
        color_continuous_scale="Turbo",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Oldest/youngest/tallest/shortest/heaviest/lightest
    c1, c2, c3, c4 = st.columns(4)
    oldest = df.dropna(subset=["Age"]).sort_values("Age", ascending=False).head(5)[["Name", "Age", "Sport", "Team", "Year"]]
    youngest = df.dropna(subset=["Age"]).sort_values("Age", ascending=True).head(5)[["Name", "Age", "Sport", "Team", "Year"]]
    tallest = df.dropna(subset=["Height"]).sort_values("Height", ascending=False).head(5)[["Name", "Height", "Sport", "Team", "Year"]]
    shortest = df.dropna(subset=["Height"]).sort_values("Height", ascending=True).head(5)[["Name", "Height", "Sport", "Team", "Year"]]

    with c1:
        st.markdown("### Oldest Athletes")
        st.dataframe(oldest, use_container_width=True)

    with c2:
        st.markdown("### Youngest Athletes")
        st.dataframe(youngest, use_container_width=True)

    with c3:
        st.markdown("### Tallest Athletes")
        st.dataframe(tallest, use_container_width=True)

    with c4:
        st.markdown("### Shortest Athletes")
        st.dataframe(shortest, use_container_width=True)

    # Heaviest / lightest
    c1, c2 = st.columns(2)
    heaviest = df.dropna(subset=["Weight"]).sort_values("Weight", ascending=False).head(5)[["Name", "Weight", "Sport", "Team", "Year"]]
    lightest = df.dropna(subset=["Weight"]).sort_values("Weight", ascending=True).head(5)[["Name", "Weight", "Sport", "Team", "Year"]]

    with c1:
        st.markdown("### Heaviest Athletes")
        st.dataframe(heaviest, use_container_width=True)
    with c2:
        st.markdown("### Lightest Athletes")
        st.dataframe(lightest, use_container_width=True)

    # Athlete search
    st.markdown("---")
    st.subheader("🔎 Athlete Search & Profile")

    search_term = st.text_input("Search athlete by name", key="athlete_search")
    if search_term:
        matches = df[df["Name"].astype(str).str.contains(search_term, case=False, na=False)]
        if matches.empty:
            st.info("No matching athletes found.")
        else:
            st.write(f"Found {matches['Name'].nunique()} athlete(s).")
            st.dataframe(matches.head(20)[["Name", "Sex", "Sport", "Team", "NOC", "Medal", "Year", "City"]], use_container_width=True)

            # Profile summary
            athlete_profile = matches.groupby("Name").agg(
                Appearances=("Year", "nunique"),
                Sports=("Sport", "nunique"),
                Events=("Event", "nunique"),
                Medals=("Medal", lambda s: (s != "No Medal").sum()),
                Gold=("Medal", lambda s: (s == "Gold").sum()),
                Silver=("Medal", lambda s: (s == "Silver").sum()),
                Bronze=("Medal", lambda s: (s == "Bronze").sum()),
            ).reset_index()

            st.dataframe(athlete_profile.head(10), use_container_width=True)

            # Performance over years
            athlete_year = (
                matches.groupby("Year")
                .size()
                .reset_index(name="Participations")
            )
            fig = px.line(
                athlete_year,
                x="Year",
                y="Participations",
                markers=True,
                title="Athlete Performance Over Years",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

def render_country_page(df):
    """
    Country-focused analysis.
    """
    st.header("🌍 Country Analysis")

    # Top participating countries
    country_participation = (
        df["NOC"]
        .value_counts()
        .head(15)
        .reset_index()
    )
    country_participation.columns = ["Country", "Participants"]

    fig = px.bar(
        country_participation,
        x="Country",
        y="Participants",
        color="Participants",
        title="Top Participating Countries",
        color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Top medal-winning countries
    medal_countries = (
        df[df["Medal"] != "No Medal"]
        .groupby("NOC")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="Medals")
    )
    medal_countries.columns = ["Country", "Medals"]

    fig2 = px.bar(
        medal_countries,
        x="Country",
        y="Medals",
        color="Medals",
        title="Top Medal-Winning Countries",
        color_continuous_scale="Cividis",
    )
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Country comparison
    st.subheader("Country Comparison")
    country_compare = st.multiselect(
        "Select countries to compare",
        options=sorted(df["NOC"].dropna().astype(str).unique().tolist()),
        default=sorted(df["NOC"].dropna().astype(str).unique().tolist())[:5],
        key="country_compare",
    )
    if country_compare:
        compare_df = (
            df[df["NOC"].isin(country_compare)]
            .groupby(["NOC", "Medal"])
            .size()
            .reset_index(name="Count")
        )
        fig3 = px.bar(
            compare_df,
            x="NOC",
            y="Count",
            color="Medal",
            title="Country Comparison by Medal",
        )
        fig3.update_layout(template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    # Country participation over years
    country_year = (
        df.groupby(["Year", "NOC"])
        .size()
        .reset_index(name="Participants")
    )
    top_countries_for_year = country_year["NOC"].value_counts().head(5).index.tolist()
    country_year_filtered = country_year[country_year["NOC"].isin(top_countries_for_year)]
    fig4 = px.line(
        country_year_filtered,
        x="Year",
        y="Participants",
        color="NOC",
        markers=True,
        title="Country Participation Over Years",
    )
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

def render_sports_page(df):
    """
    Sports-focused analysis.
    """
    st.header("🏀 Sports Analysis")

    c1, c2 = st.columns(2)

    # Most popular sports
    sport_counts = df["Sport"].value_counts().reset_index()
    sport_counts.columns = ["Sport", "Count"]
    fig = px.bar(
        sport_counts.head(15),
        x="Sport",
        y="Count",
        color="Count",
        title="Most Popular Sports",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_white")
    c1.plotly_chart(fig, use_container_width=True)

    # Least popular sports
    least_sports = sport_counts.tail(10)
    fig2 = px.bar(
        least_sports,
        x="Sport",
        y="Count",
        color="Count",
        title="Least Popular Sports",
        color_continuous_scale="Cividis",
    )
    fig2.update_layout(template="plotly_white")
    c2.plotly_chart(fig2, use_container_width=True)

    # Growth over time
    sport_year = (
        df.groupby(["Year", "Sport"])
        .size()
        .reset_index(name="Count")
    )
    top_sports = sport_year["Sport"].value_counts().head(6).index.tolist()
    sport_year_filtered = sport_year[sport_year["Sport"].isin(top_sports)]
    fig3 = px.line(
        sport_year_filtered,
        x="Year",
        y="Count",
        color="Sport",
        markers=True,
        title="Top Sports Growth Over Time",
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # Sports by gender
    sports_gender = (
        df.groupby(["Sport", "Sex"])
        .size()
        .reset_index(name="Count")
    )
    fig4 = px.bar(
        sports_gender.head(60),
        x="Sport",
        y="Count",
        color="Sex",
        title="Sports by Gender",
    )
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    # Sports by season
    sports_season = (
        df.groupby(["Sport", "Season"])
        .size()
        .reset_index(name="Count")
    )
    fig5 = px.bar(
        sports_season.head(80),
        x="Sport",
        y="Count",
        color="Season",
        title="Sports by Season",
    )
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

def render_medal_page(df):
    """
    Medal analysis page.
    """
    st.header("🥇🥈🥉 Medal Analysis")

    medal_counts = df["Medal"].value_counts().reset_index()
    medal_counts.columns = ["Medal", "Count"]

    fig = px.bar(
        medal_counts,
        x="Medal",
        y="Count",
        color="Medal",
        title="Medal Counts",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Medal trends over years
    medal_year = (
        df[df["Medal"] != "No Medal"]
        .groupby(["Year", "Medal"])
        .size()
        .reset_index(name="Count")
    )
    fig2 = px.line(
        medal_year,
        x="Year",
        y="Count",
        color="Medal",
        markers=True,
        title="Medals Over Years",
    )
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Medals by country
    country_medals = (
        df[df["Medal"] != "No Medal"]
        .groupby(["NOC", "Medal"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(60)
    )
    fig3 = px.bar(
        country_medals,
        x="NOC",
        y="Count",
        color="Medal",
        title="Medals by Country",
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # Medals by sport
    sport_medals = (
        df[df["Medal"] != "No Medal"]
        .groupby(["Sport", "Medal"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(60)
    )
    fig4 = px.bar(
        sport_medals,
        x="Sport",
        y="Count",
        color="Medal",
        title="Medals by Sport",
    )
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    # Medals by gender
    gender_medals = (
        df[df["Medal"] != "No Medal"]
        .groupby(["Sex", "Medal"])
        .size()
        .reset_index(name="Count")
    )
    fig5 = px.bar(
        gender_medals,
        x="Sex",
        y="Count",
        color="Medal",
        title="Medals by Gender",
    )
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

    # Interactive leaderboard
    st.subheader("Medal Leaderboard")
    leaderboard = (
        df[df["Medal"] != "No Medal"]
        .groupby(["Name", "NOC", "Sport"])
        .size()
        .reset_index(name="Medals")
        .sort_values("Medals", ascending=False)
        .head(30)
    )
    st.dataframe(leaderboard, use_container_width=True)

def render_time_series_page(df):
    """
    Time series analysis page.
    """
    st.header("⏳ Time Series Analysis")

    year_stats = (
        df.groupby("Year")
        .agg(
            Athletes=("Name", "nunique"),
            Countries=("NOC", "nunique"),
            Sports=("Sport", "nunique"),
            Events=("Event", "nunique"),
            Medals=("Medal", lambda s: (s != "No Medal").sum()),
        )
        .reset_index()
    )

    fig = px.line(
        year_stats,
        x="Year",
        y="Athletes",
        markers=True,
        title="Athletes Over Years",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        year_stats,
        x="Year",
        y="Countries",
        markers=True,
        title="Countries Over Years",
    )
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        year_stats,
        x="Year",
        y="Sports",
        markers=True,
        title="Sports Over Years",
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(
        year_stats,
        x="Year",
        y="Events",
        markers=True,
        title="Events Over Years",
    )
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.line(
        year_stats,
        x="Year",
        y="Medals",
        markers=True,
        title="Medals Over Years",
    )
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

    # Growth percentage and YOY changes
    year_stats["Athletes_Growth_%"] = year_stats["Athletes"].pct_change() * 100
    year_stats["Countries_Growth_%"] = year_stats["Countries"].pct_change() * 100
    year_stats["Events_Growth_%"] = year_stats["Events"].pct_change() * 100
    year_stats["Sports_Growth_%"] = year_stats["Sports"].pct_change() * 100

    growth_table = year_stats[["Year", "Athletes_Growth_%", "Countries_Growth_%", "Sports_Growth_%", "Events_Growth_%"]]
    st.subheader("Growth Rates")
    st.dataframe(growth_table, use_container_width=True)

def render_demographics_page(df):
    """
    Demographic analysis.
    """
    st.header("👥 Demographic Analysis")

    c1, c2 = st.columns(2)

    fig = px.histogram(
        df.dropna(subset=["Age"]),
        x="Age",
        nbins=35,
        color="Sex",
        title="Age Distribution",
        marginal="box",
    )
    fig.update_layout(template="plotly_white")
    c1.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(
        df.dropna(subset=["Height"]),
        x="Height",
        nbins=40,
        color="Sex",
        title="Height Distribution",
        marginal="box",
    )
    fig2.update_layout(template="plotly_white")
    c2.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        df.dropna(subset=["Weight"]),
        x="Weight",
        nbins=40,
        color="Sex",
        title="Weight Distribution",
        marginal="box",
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # BMI
    if "BMI" in df.columns:
        fig4 = px.histogram(
            df.dropna(subset=["BMI"]),
            x="BMI",
            nbins=40,
            color="Sex",
            title="BMI Distribution",
            marginal="box",
        )
        fig4.update_layout(template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)

    # Age vs Medal
    age_medal = df.dropna(subset=["Age"])
    fig5 = px.box(
        age_medal,
        x="Medal",
        y="Age",
        color="Medal",
        title="Age vs Medal",
    )
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

    # Height vs Medal
    fig6 = px.box(
        df.dropna(subset=["Height"]),
        x="Medal",
        y="Height",
        color="Medal",
        title="Height vs Medal",
    )
    fig6.update_layout(template="plotly_white")
    st.plotly_chart(fig6, use_container_width=True)

    # Weight vs Medal
    fig7 = px.box(
        df.dropna(subset=["Weight"]),
        x="Medal",
        y="Weight",
        color="Medal",
        title="Weight vs Medal",
    )
    fig7.update_layout(template="plotly_white")
    st.plotly_chart(fig7, use_container_width=True)

def render_advanced_page(df):
    """
    Advanced analytics: correlation, scatter matrix, rolling averages, skewness, kurtosis, percentiles.
    """
    st.header("🧠 Advanced Analytics")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:
        # Correlation heatmap
        corr = df[numeric_cols].corr().fillna(0)
        fig1 = px.imshow(
            corr,
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r",
        )
        fig1.update_layout(template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

        # Scatter matrix
        fig2 = px.scatter_matrix(
            df[numeric_cols + ["Sex"]].dropna(),
            dimensions=numeric_cols[:4],
            color="Sex",
            title="Scatter Matrix",
        )
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

        # Rolling averages
        if "Year" in df.columns:
            year_stats = (
                df.groupby("Year")
                .agg(Athletes=("Name", "nunique"))
                .reset_index()
            )
            year_stats["Rolling_Avg_3"] = year_stats["Athletes"].rolling(window=3).mean()
            fig3 = px.line(
                year_stats,
                x="Year",
                y=["Athletes", "Rolling_Avg_3"],
                title="Rolling Average of Athletes",
            )
            fig3.update_layout(template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

        # Distribution analysis
        st.subheader("Skewness & Kurtosis")
        skewness = df[numeric_cols].skew()
        kurtosis = df[numeric_cols].kurt()
        stats_df = pd.DataFrame({
            "Skewness": skewness,
            "Kurtosis": kurtosis,
        }).reset_index().rename(columns={"index": "Column"})
        st.dataframe(stats_df, use_container_width=True)

        # Percentile analysis
        st.subheader("Percentile Analysis")
        percentile_table = pd.DataFrame({
            "Column": numeric_cols,
            "25th Percentile": [df[col].quantile(0.25) for col in numeric_cols],
            "50th Percentile": [df[col].quantile(0.50) for col in numeric_cols],
            "75th Percentile": [df[col].quantile(0.75) for col in numeric_cols],
            "90th Percentile": [df[col].quantile(0.90) for col in numeric_cols],
            "95th Percentile": [df[col].quantile(0.95) for col in numeric_cols],
        })
        st.dataframe(percentile_table, use_container_width=True)

def render_geography_page(df):
    """
    Geographical analysis with interactive world maps.
    """
    st.header("🌐 Geographical Analysis")

    # Mapping from NOC to ISO3
    noc_to_iso3 = {
        "USA":"USA","CAN":"CAN","MEX":"MEX","CUB":"CUB","JAM":"JAM","BRA":"BRA","ARG":"ARG","CHL":"CHL","COL":"COL","PER":"PER","URU":"URU","VEN":"VEN","PRY":"PRY","ECU":"ECU","BOL":"BOL","PAN":"PAN","CRI":"CRI","DOM":"DOM","GTM":"GTM","HND":"HND","NIC":"NIC","SLV":"SLV","BLZ":"BLZ",
        "AUS":"AUS","NZL":"NZL","FJI":"FJI","PNG":"PNG","JPN":"JPN","KOR":"KOR","CHN":"CHN","HKG":"HKG","TWN":"TWN","IND":"IND","PAK":"PAK","LKA":"LKA","KAZ":"KAZ","UZB":"UZB","KGZ":"KGZ","TJK":"TJK","TKM":"TKM","ARM":"ARM","AZE":"AZE","GEO":"GEO",
        "GBR":"GBR","IRL":"IRL","NLD":"NLD","BEL":"BEL","FRA":"FRA","DEU":"DEU","ITA":"ITA","ESP":"ESP","PRT":"PRT","AUT":"AUT","CHE":"CHE","SWE":"SWE","NOR":"NOR","DNK":"DNK","FIN":"FIN","POL":"POL","CZE":"CZE","SVK":"SVK","HUN":"HUN","ROU":"ROU","BGR":"BGR","SRB":"SRB","HRV":"HRV","SVN":"SVN","BIH":"BIH","MKD":"MKD","GRC":"GRC","RUS":"RUS","UKR":"UKR","BLR":"BLR","EST":"EST","LVA":"LVA","LTU":"LTU",
        "NGA":"NGA","KEN":"KEN","ETH":"ETH","UGA":"UGA","ZAF":"ZAF","EGY":"EGY","TUN":"TUN","MAR":"MAR","DZA":"DZA","GHA":"GHA","CMR":"CMR","SEN":"SEN","CIV":"CIV","ZMB":"ZMB","ZWE":"ZWE",
        "TUR":"TUR","ISR":"ISR","IRN":"IRN","SAU":"SAU","ARE":"ARE","QAT":"QAT","JOR":"JOR","LBN":"LBN","KWT":"KWT","OMN":"OMN","BHR":"BHR",
    }

    # Country participation map
    country_participation = (
        df.groupby("NOC")
        .agg(Participants=("Name", "nunique"))
        .reset_index()
    )
    country_participation["ISO3"] = country_participation["NOC"].map(noc_to_iso3)

    country_participation = country_participation.dropna(subset=["ISO3"])

    fig1 = px.choropleth(
        country_participation,
        locations="ISO3",
        color="Participants",
        hover_name="NOC",
        title="Country Participation",
        color_continuous_scale="Viridis",
    )
    fig1.update_layout(template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    # Country medals map
    medal_country = (
        df[df["Medal"] != "No Medal"]
        .groupby("NOC")
        .size()
        .reset_index(name="Medals")
    )
    medal_country["ISO3"] = medal_country["NOC"].map(noc_to_iso3)
    medal_country = medal_country.dropna(subset=["ISO3"])

    fig2 = px.choropleth(
        medal_country,
        locations="ISO3",
        color="Medals",
        hover_name="NOC",
        title="Country Medal Counts",
        color_continuous_scale="Cividis",
    )
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Animated map over years
    animated = (
        df.groupby(["Year", "NOC"])
        .size()
        .reset_index(name="Participants")
    )
    animated["ISO3"] = animated["NOC"].map(noc_to_iso3)
    animated = animated.dropna(subset=["ISO3"])
    fig3 = px.choropleth(
        animated,
        locations="ISO3",
        color="Participants",
        hover_name="NOC",
        animation_frame="Year",
        title="Animated Country Participation Map",
        color_continuous_scale="Plasma",
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

def render_about_page():
    """
    About page with project summary and developer information.
    """
    st.header("ℹ️ About This Dashboard")

    st.markdown(
        """
        This dashboard is a portfolio-style Olympic analytics platform built in Streamlit.

        Features:
        - Advanced sidebar filtering
        - Dynamic interactive charts with Plotly
        - KPI-driven executive summary
        - Athlete, country, sport, and medal analysis
        - Data quality and outlier diagnostics
        - Time-series and geographical analysis
        - Professional UI designed for modern BI-style exploration

        Built with:
        - Streamlit
        - Plotly
        - Pandas
        - NumPy
        - Seaborn
        - Matplotlib
        """
    )

    st.info("Dataset source: Kaggle Olympics Athlete Events dataset (athlete_events.csv)")

# =========================
# Run App
# =========================

if __name__ == "__main__":
    main()