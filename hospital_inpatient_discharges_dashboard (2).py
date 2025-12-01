import io
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import zipfile
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
        "gender": np.random.choice(genders, n),
        "admission_date": start_dates,
        "discharge_date": end_dates,
        "length_of_stay": los,
        "diagnosis_code": np.random.choice(diag_codes, n),
        "facility": np.random.choice(facilities, n),
        "county": np.random.choice(counties, n),
        "payment_type": np.random.choice(payment_types, n),
        "severity": np.random.choice(severities, n),
        "total_charges": charges,
        "readmission_within_30d": readmit
    })
    return df

@st.cache_data
def clean_data(df):
    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)
    df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")
    df["length_of_stay"] = df["length_of_stay"].fillna((df["discharge_date"] - df["admission_date"]).dt.total_seconds() / 86400)
    df["length_of_stay"] = df["length_of_stay"].fillna(0).clip(lower=0).astype(int)
    df["severity"] = df["severity"].astype(str).str.title()
    df["diagnosis_code"] = df["diagnosis_code"].astype(str).str.upper()
    df["payment_type"] = df["payment_type"].astype(str).str.title()
    df["is_readmit"] = df.get("readmission_within_30d", 0).fillna(0).astype(int)
    df["age_group"] = pd.cut(df["age"].fillna(0), bins=[-1,4,17,35,50,65,120], labels=["0-4","5-17","18-35","36-50","51-65","65+"])
    return df

def filter_dataframe(df):
    st.sidebar.header("Filters")
    min_date = df["admission_date"].min()
    max_date = df["admission_date"].max()
    date_range = st.sidebar.date_input("Admission date range", [min_date.date(), max_date.date()], key="admission_range")
    df2 = df[(df["admission_date"]>=pd.to_datetime(date_range[0])) & (df["admission_date"]<=pd.to_datetime(date_range[1]))]
    facilities = st.sidebar.multiselect("Facility", sorted(df["facility"].unique()), default=list(sorted(df["facility"].unique())))
    counties = st.sidebar.multiselect("County", sorted(df["county"].unique()), default=list(sorted(df["county"].unique())))
    severities = st.sidebar.multiselect("Severity", sorted(df["severity"].unique()), default=list(sorted(df["severity"].unique())))
    payment = st.sidebar.multiselect("Payment Type", sorted(df["payment_type"].unique()), default=list(sorted(df["payment_type"].unique())))
    diagnoses = st.sidebar.multiselect("Diagnosis Codes", sorted(df["diagnosis_code"].unique()), default=list(sorted(df["diagnosis_code"].unique()))[:10])
    if facilities: df2 = df2[df2["facility"].isin(facilities)]
    if counties: df2 = df2[df2["county"].isin(counties)]
    if severities: df2 = df2[df2["severity"].isin(severities)]
    if payment: df2 = df2[df2["payment_type"].isin(payment)]
    if diagnoses: df2 = df2[df2["diagnosis_code"].isin(diagnoses)]
    return df2

def compute_kpis(df):
    return {
        "total": df.shape[0],
        "avg_los": df["length_of_stay"].mean(),
        "median_los": df["length_of_stay"].median(),
        "readmit_rate": df["is_readmit"].mean() * 100
    }

st.sidebar.title("Hospital Inpatient Discharges Dataset")
uploaded_file = st.sidebar.file_uploader("Upload dataset (ZIP / CSV / Excel)", type=["zip","csv","xlsx","xls"])
use_sample = st.sidebar.checkbox("Use sample dataset", value=False)

if uploaded_file and not use_sample:
    name = uploaded_file.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file, "r") as z:
            csv_file = [f for f in z.namelist() if f.lower().endswith(".csv")][0]
            with z.open(csv_file) as f:
                raw_df = pd.read_csv(f)
    elif name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
else:
    raw_df = generate_sample_data()

df = clean_data(raw_df)
filtered = filter_dataframe(df)
kpis = compute_kpis(filtered)

st.title("Hospital Inpatient Discharges Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Discharges", kpis["total"])
col2.metric("Avg LOS", f"{kpis['avg_los']:.2f}")
col3.metric("Median LOS", f"{kpis['median_los']:.0f}")
col4.metric("Readmission Rate (%)", f"{kpis['readmit_rate']:.2f}")
