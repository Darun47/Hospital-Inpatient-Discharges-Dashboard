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
        "payment_type": np.random_choice(payment_types, n, p=[0.35,0.25,0.2,0.15,0.05]),
        "severity": np.random.choice(severities, n, p=[0.4,0.35,0.2,0.05]),
        "total_charges": charges,
        "readmission_within_30d": readmit
    })
    return df

@st.cache_data
def clean_data(df):
    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    if "length_of_stay" in df.columns:
        df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")
    df["length_of_stay"] = df["length_of_stay"].fillna((df["discharge_date"] - df["admission_date"]).dt.days)
    df["length_of_stay"] = df["length_of_stay"].clip(lower=0).astype(int)
    df["total_charges"] = df["total_charges"].astype(str).str.replace(r"[$,]", "", regex=True)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)
    df["severity"] = df["severity"].astype(str).str.title()
    df["age_group"] = pd.cut(df["age"].fillna(0), bins=[-1,4,17,35,50,65,120], labels=["0-4","5-17","18-35","36-50","51-65","65+"])
    df["diagnosis_code"] = df["diagnosis_code"].astype(str).upper()
    df["payment_type"] = df["payment_type"].astype(str).title()
    df["discharge_month"] = df["discharge_date"].dt.to_period("M").astype(str)
    df["is_readmit"] = df.get("readmission_within_30d", df.get("readmit", 0)).fillna(0).astype(int)
    df = df.replace({np.inf: np.nan})
    return df

st.sidebar.title("Hospital Inpatient Discharges")
uploaded_file = st.sidebar.file_uploader("Upload inpatient discharges CSV or Excel", type=["csv","xlsx","xls"])

google_drive_id = st.sidebar.text_input("17XAIEEOIHOL0j28a5YCNuSjl67IGDHXL")

raw_df = None

if google_drive_id:
    try:
        gdrive_url = f"https://drive.google.com/uc?export=download&id={google_drive_id}"
        st.sidebar.write("Loading Google Drive dataset...")
        raw_df = pd.read_csv(gdrive_url)
        st.sidebar.success("Dataset loaded from Google Drive!")
    except Exception as e:
        st.sidebar.error(f"Failed to load Google Drive file: {e}")

use_sample = st.sidebar.checkbox("Use sample dataset", value=(uploaded_file is None and not google_drive_id))

if raw_df is None:
    if uploaded_file is None and not use_sample:
        raw_df = generate_sample_data()
    else:
        if uploaded_file is not None and not use_sample:
            try:
                if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
                    raw_df = pd.read_excel(uploaded_file)
                else:
                    raw_df = pd.read_csv(uploaded_file)
            except Exception:
                raw_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), on_bad_lines="skip")
        else:
            raw_df = generate_sample_data()

df = clean_data(raw_df)

def filter_dataframe(df):
    st.sidebar.header("Filters")
    min_date = df["admission_date"].min()
    max_date = df["admission_date"].max()
    date_range = st.sidebar.date_input("Admission date range", [min_date.date(), max_date.date()])
    facilities = st.sidebar.multiselect("Facility", options=sorted(df["facility"].unique()), default=sorted(df["facility"].unique()))
    counties = st.sidebar.multiselect("County", options=sorted(df["county"].unique()), default=sorted(df["county"].unique()))
    severities = st.sidebar.multiselect("Severity", options=sorted(df["severity"].unique()), default=sorted(df["severity"].unique()))
    payment = st.sidebar.multiselect("Payment Type", options=sorted(df["payment_type"].unique()), default=sorted(df["payment_type"].unique()))
    diagnoses = st.sidebar.multiselect("Diagnosis Codes", options=sorted(df["diagnosis_code"].unique()), default=sorted(df["diagnosis_code"].unique())[:10])
    df2 = df.copy()
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df2 = df2[(df2["admission_date"] >= start) & (df2["admission_date"] < end)]
    df2 = df2[df2["facility"].isin(facilities)]
    df2 = df2[df2["county"].isin(counties)]
    df2 = df2[df2["severity"].isin(severities)]
    df2 = df2[df2["payment_type"].isin(payment)]
    df2 = df2[df2["diagnosis_code"].isin(diagnoses)]
    return df2

filtered = filter_dataframe(df)

def compute_kpis(df):
    total_discharges = int(df.shape[0])
    avg_los = float(df["length_of_stay"].mean()) if total_discharges else 0
    avg_charges = float(df["total_charges"].mean()) if total_discharges else 0
    readmit_rate = float(df["is_readmit"].mean() * 100) if total_discharges else 0
    median_los = float(df["length_of_stay"].median()) if total_discharges else 0
    return total_discharges, avg_los, avg_charges, readmit_rate, median_los

total, avg_los, avg_charges, readmit_rate, median_los = compute_kpis(filtered)

st.title("Hospital Inpatient Discharges Dashboard")
col1, col2, col3, col4 = st.columns([2,2,2,2])
col1.metric("Total Discharges", total)
col2.metric("Avg Length of Stay (days)", f'{avg_los:.2f}')
col3.metric("Median Length of Stay (days)", f'{median_los:.0f}')
col4.metric("Readmission Rate (%)", f'{readmit_rate:.2f}')

with st.expander("Data Preview & Download", expanded=False):
    st.dataframe(filtered.head(200))
    st.download_button("Download Cleaned Dataset CSV", filtered.to_csv(index=False).encode(), "cleaned_dataset.csv")

def plot_avg_stay_by_diagnosis(df):
    agg = df.groupby("diagnosis_code")["length_of_stay"].mean().sort_values(ascending=False).head(20)
    return px.bar(agg, title="Average Hospital Stay per Diagnosis Code")

def plot_charges_box_by_severity(df):
    return px.box(df, x="severity", y="total_charges", title="Total Charges by Severity")

def plot_heatmap_stay_by_facility_county(df):
    pivot = df.pivot_table(index="facility", columns="county", values="length_of_stay", aggfunc="mean")
    return px.imshow(pivot, title="Avg LOS by Facility and County")

def plot_payment_type_pie(df):
    return px.pie(df, names="payment_type", title="Patient Distribution by Payment Type")

left, right = st.columns(2)
left.plotly_chart(plot_avg_stay_by_diagnosis(filtered), use_container_width=True)
right.plotly_chart(plot_heatmap_stay_by_facility_county(filtered), use_container_width=True)
left.plotly_chart(plot_charges_box_by_severity(filtered), use_container_width=True)
right.plotly_chart(plot_payment_type_pie(filtered), use_container_width=True)

fig5 = px.histogram(filtered, x="length_of_stay", nbins=30, title="Distribution of Length of Stay")
fig6 = go.Figure()
d = filtered.sample(min(5000, filtered.shape[0]))
x = d["age"].values
y = d["length_of_stay"].values
fig6.add_trace(go.Scatter(x=x, y=y, mode="markers"))
if len(np.unique(x)) > 1:
    m, b = np.polyfit(x, y, 1)
    xs = np.sort(x)
    fig6.add_trace(go.Scatter(x=xs, y=m*xs+b, mode="lines"))
fig6.update_layout(title="Age vs Length of Stay")

a, b = st.columns(2)
a.plotly_chart(fig5, use_container_width=True)
b.plotly_chart(fig6, use_container_width=True)

leader = filtered.groupby("diagnosis_code").agg(
    avg_los=("length_of_stay","mean"),
    avg_charges=("total_charges","mean"),
    count=("patient_id","count")
).sort_values("avg_los", ascending=False)

st.markdown("### Diagnosis Leaderboard (by Avg Length of Stay)")
st.dataframe(leader)

st.download_button("Download Diagnosis Leaderboard CSV", leader.to_csv().encode(), "leaderboard.csv")
