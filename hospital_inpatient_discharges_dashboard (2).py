import io
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

st.set_page_config(layout="wide", page_title="Hospital Inpatient Discharges Dashboard", page_icon="🏥")

@st.cache_data
def generate_sample_data(n=2000, seed=42):
    np.random.seed(seed)
    diag_codes = ["I10","E11","J18","N39","K35","M54","G47","F32","C50","S72"]
    facilities = ["Central Hospital","North Clinic","East Medical Center","West Health","County General"]
    counties = ["County A","County B","County C","County D","County E"]
    payment_types = ["Private","Medicare","Medicaid","Self-pay","Other"]
    severities = ["Minor","Moderate","Major","Extreme"]
    genders = ["Male","Female","Other"]
    start_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 700, n), unit="D")
    los = np.clip(np.random.poisson(5, n) + np.random.choice([0,1,2,3], n, p=[0.5,0.2,0.2,0.1]), 1, 60)
    end_dates = start_dates + pd.to_timedelta(los, unit="D")
    charges = np.round(np.abs(np.random.normal(8000, 6000, n)) + los * np.random.normal(500,200,n), 2)
    ages = np.random.randint(0, 100, n)
    readmit = np.random.choice([0,1], n, p=[0.9,0.1])
    df = pd.DataFrame({
        "patient_id": [f"P{100000+i}" for i in range(n)],
        "age": ages,
        "gender": np.random.choice(genders, n, p=[0.48,0.48,0.04]),
        "admission_date": start_dates,
        "discharge_date": end_dates,
        "length_of_stay": los,
        "diagnosis_code": np.random.choice(diag_codes, n, p=[0.12,0.15,0.1,0.08,0.1,0.12,0.07,0.08,0.1,0.08]),
        "facility": np.random.choice(facilities, n),
        "county": np.random.choice(counties, n),
        "payment_type": np.random.choice(payment_types, n, p=[0.35,0.25,0.2,0.15,0.05]),
        "severity": np.random.choice(severities, n, p=[0.4,0.35,0.2,0.05]),
        "total_charges": charges,
        "readmission_within_30d": readmit
    })
    return df

@st.cache_data
def clean_data(df):
    df = df.copy()
    if "admission_date" in df.columns:
        df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    if "discharge_date" in df.columns:
        df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    if "length_of_stay" in df.columns:
        df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce").fillna((df["discharge_date"]-df["admission_date"]).dt.days).fillna(0).astype(int)
    else:
        df["length_of_stay"] = (df["discharge_date"]-df["admission_date"]).dt.days.fillna(0).astype(int)
    if "total_charges" in df.columns:
        df["total_charges"] = df["total_charges"].astype(str).str.replace(r"[$,]", "", regex=True)
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)
    else:
        df["total_charges"] = 0.0
    if "severity" in df.columns:
        df["severity"] = df["severity"].astype(str).str.title()
    df["age_group"] = pd.cut(df["age"].fillna(0), bins=[-1,4,17,35,50,65,120], labels=["0-4","5-17","18-35","36-50","51-65","65+"])
    if "diagnosis_code" in df.columns:
        df["diagnosis_code"] = df["diagnosis_code"].astype(str).str.upper()
    if "payment_type" in df.columns:
        df["payment_type"] = df["payment_type"].astype(str).str.title()
    df["discharge_month"] = df["discharge_date"].dt.to_period("M").astype(str)
    df["is_readmit"] = df.get("readmission_within_30d", df.get("readmit", 0)).fillna(0).astype(int)
    df = df.replace({np.inf: np.nan})
    return df

def filter_dataframe(df):
    st.sidebar.header("Filters")
    min_date = df["admission_date"].min() if not df["admission_date"].isna().all() else pd.to_datetime("2000-01-01")
    max_date = df["discharge_date"].max() if not df["discharge_date"].isna().all() else pd.to_datetime("2100-01-01")
    date_range = st.sidebar.date_input("Admission date range", [min_date.date(), max_date.date()])
    facilities = st.sidebar.multiselect("Facility", options=sorted(df["facility"].dropna().unique()), default=sorted(df["facility"].dropna().unique()))
    counties = st.sidebar.multiselect("County", options=sorted(df["county"].dropna().unique()), default=sorted(df["county"].dropna().unique()))
    severities = st.sidebar.multiselect("Severity", options=sorted(df["severity"].dropna().unique()), default=sorted(df["severity"].dropna().unique()))
    payment = st.sidebar.multiselect("Payment Type", options=sorted(df["payment_type"].dropna().unique()), default=sorted(df["payment_type"].dropna().unique()))
    diagnoses = st.sidebar.multiselect("Diagnosis Codes", options=sorted(df["diagnosis_code"].dropna().unique()), default=sorted(df["diagnosis_code"].dropna().unique())[:10])
    df2 = df.copy()
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df2 = df2[(df2["admission_date"]>=start) & (df2["admission_date"]<end)]
    if facilities:
        df2 = df2[df2["facility"].isin(facilities)]
    if counties:
        df2 = df2[df2["county"].isin(counties)]
    if severities:
        df2 = df2[df2["severity"].isin(severities)]
    if payment:
        df2 = df2[df2["payment_type"].isin(payment)]
    if diagnoses:
        df2 = df2[df2["diagnosis_code"].isin(diagnoses)]
    return df2

def compute_kpis(df):
    total_discharges = int(df.shape[0])
    avg_los = float(df["length_of_stay"].mean()) if total_discharges>0 else 0.0
    avg_charges = float(df["total_charges"].mean()) if total_discharges>0 else 0.0
    readmit_rate = float(df["is_readmit"].sum()/total_discharges*100) if total_discharges>0 else 0.0
    median_los = float(df["length_of_stay"].median()) if total_discharges>0 else 0.0
    return {"total_discharges":total_discharges,"avg_los":avg_los,"avg_charges":avg_charges,"readmit_rate":readmit_rate,"median_los":median_los}

def plot_avg_stay_by_diagnosis(df):
    agg = df.groupby("diagnosis_code", as_index=False)["length_of_stay"].mean().sort_values("length_of_stay", ascending=False).head(20)
    fig = px.bar(agg, x="diagnosis_code", y="length_of_stay", labels={"length_of_stay":"Avg Length of Stay (days)","diagnosis_code":"Diagnosis Code"}, title="Average Hospital Stay per Diagnosis Code (Top 20)")
    return fig

def plot_charges_box_by_severity(df):
    fig = px.box(df, x="severity", y="total_charges", points="outliers", labels={"total_charges":"Total Charges","severity":"Severity"}, title="Total Charges by Severity")
    return fig

def plot_heatmap_stay_by_facility_county(df):
    pivot = df.pivot_table(index="facility", columns="county", values="length_of_stay", aggfunc="mean").fillna(0)
    fig = px.imshow(pivot, labels=dict(x="County", y="Facility", color="Avg LOS"), x=list(pivot.columns), y=list(pivot.index), title="Average Length of Stay by Facility and County")
    return fig

def plot_payment_type_pie(df):
    dist = df["payment_type"].value_counts().reset_index()
    dist.columns = ["payment_type","count"]
    fig = px.pie(dist, names="payment_type", values="count", title="Patient Distribution by Payment Type", hole=0.3)
    return fig

def plot_los_histogram(df):
    fig = px.histogram(df, x="length_of_stay", nbins=30, labels={"length_of_stay":"Length of Stay (days)"}, title="Distribution of Length of Stay")
    return fig

def plot_age_vs_los_scatter(df):
    d = df.sample(min(5000, max(1, df.shape[0])))
    x = d["age"].values
    y = d["length_of_stay"].values
    m, b = np.polyfit(x, y, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=np.sort(x), y=m*np.sort(x)+b, mode="lines", name="Trendline"))
    fig.update_layout(title="Age vs Length of Stay", xaxis_title="Age", yaxis_title="Length of Stay")
    return fig


def diagnosis_leaderboard(df):
    agg = df.groupby("diagnosis_code").agg(avg_los=("length_of_stay","mean"), avg_charges=("total_charges","mean"), count=("patient_id","count")).reset_index().sort_values("avg_los", ascending=False)
    agg["avg_los"] = agg["avg_los"].round(2)
    agg["avg_charges"] = agg["avg_charges"].round(2)
    top = agg.head(15)
    return top

def download_button(df, filename="cleaned_discharge_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download cleaned dataset as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

st.sidebar.title("Hospital Inpatient Discharges")
uploaded_file = st.sidebar.file_uploader("Upload inpatient discharges CSV or Excel", type=["csv","xlsx","xls"])
dataset_path = "/content/drive/MyDrive/Dataset Scenario 2:Hospital Inpatient Discharges Dashboard/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"

use_sample = st.sidebar.checkbox("Use sample dataset", value=(uploaded_file is None))

if uploaded_file is None and not use_sample:
    raw_df = pd.read_csv(dataset_path)
else:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
                raw_df = pd.read_excel(uploaded_file)
            else:
                raw_df = pd.read_csv(uploaded_file)
        except Exception:
            raw_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), error_bad_lines=False)
    else:
        raw_df = generate_sample_data()

df = clean_data(raw_df)
filtered = filter_dataframe(df)

kpis = compute_kpis(filtered)

st.title("Hospital Inpatient Discharges Dashboard")
col1, col2, col3, col4 = st.columns([2,2,2,2])
col1.metric("Total Discharges", kpis["total_discharges"])
col2.metric("Avg Length of Stay (days)", f'{kpis["avg_los"]:.2f}')
col3.metric("Median Length of Stay (days)", f'{kpis["median_los"]:.0f}')
col4.metric("Readmission Rate (%)", f'{kpis["readmit_rate"]:.2f}')

with st.expander("Data Preview & Download", expanded=False):
    st.dataframe(filtered.head(200))
    download_button(filtered)

st.markdown("### Visual Insights")
left, right = st.columns(2)
with left:
    fig1 = plot_avg_stay_by_diagnosis(filtered)
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = plot_charges_box_by_severity(filtered)
    st.plotly_chart(fig2, use_container_width=True)
with right:
    fig3 = plot_heatmap_stay_by_facility_county(filtered)
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = plot_payment_type_pie(filtered)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("### Distribution & Relationships")
a,b = st.columns(2)
with a:
    fig5 = plot_los_histogram(filtered)
    st.plotly_chart(fig5, use_container_width=True)
with b:
    fig6 = plot_age_vs_los_scatter(filtered)
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("### Diagnosis Leaderboard (by Avg Length of Stay)")
leader = diagnosis_leaderboard(filtered)
st.dataframe(leader)

st.markdown("### Additional Analysis")
colA, colB = st.columns(2)
with colA:
    readmit_by_diag = filtered.groupby("diagnosis_code").agg(readmit_rate=("is_readmit","mean"), count=("patient_id","count")).reset_index()
    readmit_by_diag["readmit_rate"] = (readmit_by_diag["readmit_rate"]*100).round(2)
    st.bar_chart(readmit_by_diag.sort_values("readmit_rate", ascending=False).head(15).set_index("diagnosis_code")["readmit_rate"])
with colB:
    top_facilities = filtered.groupby("facility").agg(avg_los=("length_of_stay","mean"), avg_charges=("total_charges","mean"), discharges=("patient_id","count")).reset_index().sort_values("discharges", ascending=False)
    st.table(top_facilities.head(10).assign(avg_los=lambda x: x["avg_los"].round(2), avg_charges=lambda x: x["avg_charges"].round(2)))

st.markdown("### Export Insights")
insights_csv = leader.to_csv(index=False)
b64 = base64.b64encode(insights_csv.encode()).decode()
st.markdown(f'<a href="data:file/csv;base64,{b64}" download="diagnosis_leaderboard.csv">Download Diagnosis Leaderboard CSV</a>', unsafe_allow_html=True)

st.markdown("### Notes & Deployment")
st.markdown("- Upload your real dataset in CSV/Excel format or use the sample data to explore the dashboard.")
st.markdown("- To deploy: push this file to a GitHub repository with a requirements.txt listing streamlit, pandas, numpy, plotly and deploy on Streamlit Cloud.")
