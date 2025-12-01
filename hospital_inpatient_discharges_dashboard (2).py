import os
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

def is_colab():
    try:
        import google.colab  # noqa
        return True
    except ImportError:
        return False

if is_colab():
    from google.colab import drive
    drive.mount("/content/drive")

st.set_page_config(layout="wide", page_title="Hospital Inpatient Discharges Dashboard", page_icon="🏥")

@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data
def generate_sample_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    diag_codes = ["I10","E11","J18","N39","K35","M54","G47","F32","C50","S72"]
    facilities = ["Central Hospital","North Clinic","East Medical Center","West Health","County General"]
    counties = ["County A","County B","County C","County D","County E"]
    payment_types = ["Private","Medicare","Medicaid","Self-pay","Other"]
    severities = ["Minor","Moderate","Major","Extreme"]
    genders = ["Male","Female","Other"]
    start_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0,700,n), unit="D")
    los = np.clip(np.random.poisson(5,n) + np.random.choice([0,1,2,3],n,p=[0.5,0.2,0.2,0.1]),1,60)
    end_dates = start_dates + pd.to_timedelta(los,unit="D")
    charges = np.round(np.abs(np.random.normal(8000,6000,n)) + los*np.random.normal(500,200,n),2)
    ages = np.random.randint(0,100,n)
    readmit = np.random.choice([0,1],n,p=[0.9,0.1])
    df = pd.DataFrame({
        "patient_id":[f"P{100000+i}" for i in range(n)],
        "age":ages,"gender":np.random.choice(genders,n),
        "admission_date":start_dates,
        "discharge_date":end_dates,
        "length_of_stay":los,
        "diagnosis_code":np.random.choice(diag_codes,n),
        "facility":np.random.choice(facilities,n),
        "county":np.random.choice(counties,n),
        "payment_type":np.random.choice(payment_types,n),
        "severity":np.random.choice(severities,n),
        "total_charges":charges,
        "readmission_within_30d":readmit
    })
    return df

@st.cache_data
def clean_data(df):
    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"],errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"],errors="coerce")
    df["length_of_stay"] = pd.to_numeric(df["length_of_stay"],errors="coerce").fillna((df["discharge_date"]-df["admission_date"]).dt.days).fillna(0).astype(int)
    df["total_charges"] = pd.to_numeric(df["total_charges"],errors="coerce").fillna(0)
    df["age_group"] = pd.cut(df["age"].fillna(0),bins=[-1,4,17,35,50,65,120],labels=["0-4","5-17","18-35","36-50","51-65","65+"])
    df["diagnosis_code"] = df["diagnosis_code"].astype(str).str.upper()
    df["payment_type"] = df["payment_type"].astype(str).str.title()
    df["severity"] = df["severity"].astype(str).str.title()
    df["discharge_month"] = df["discharge_date"].dt.to_period("M").astype(str)
    df["is_readmit"] = df["readmission_within_30d"].fillna(0).astype(int)
    return df

def filter_dataframe(df):
    st.sidebar.header("Filters")
    min_date = df["admission_date"].min()
    max_date = df["admission_date"].max()
    date_range = st.sidebar.date_input("Admission date range",[min_date.date(),max_date.date()])
    facilities = st.sidebar.multiselect("Facility",sorted(df["facility"].unique()),sorted(df["facility"].unique()))
    counties = st.sidebar.multiselect("County",sorted(df["county"].unique()),sorted(df["county"].unique()))
    severities = st.sidebar.multiselect("Severity",sorted(df["severity"].unique()),sorted(df["severity"].unique()))
    payments = st.sidebar.multiselect("Payment Type",sorted(df["payment_type"].unique()),sorted(df["payment_type"].unique()))
    diagnoses = st.sidebar.multiselect("Diagnosis Codes",sorted(df["diagnosis_code"].unique()),sorted(df["diagnosis_code"].unique())[:15])
    df2 = df.copy()
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df2 = df2[(df2["admission_date"]>=start) & (df2["admission_date"]<end)]
    df2 = df2[df2["facility"].isin(facilities)]
    df2 = df2[df2["county"].isin(counties)]
    df2 = df2[df2["severity"].isin(severities)]
    df2 = df2[df2["payment_type"].isin(payments)]
    df2 = df2[df2["diagnosis_code"].isin(diagnoses)]
    return df2

def compute_kpis(df):
    total = df.shape[0]
    return {"total":total,"avg":float(df["length_of_stay"].mean()),
            "median":float(df["length_of_stay"].median()),
            "charges":float(df["total_charges"].mean()),
            "readmit":float(df["is_readmit"].sum()/total*100 if total>0 else 0)}

def plot(df):
    return [
        px.bar(df.groupby("diagnosis_code",as_index=False)["length_of_stay"].mean().sort_values("length_of_stay",ascending=False).head(20),x="diagnosis_code",y="length_of_stay",title="Avg LOS by Diagnosis"),
        px.imshow(df.pivot_table(index="facility",columns="county",values="length_of_stay",aggfunc="mean").fillna(0),title="LOS by Facility & County"),
        px.pie(df["payment_type"].value_counts().reset_index(),names="index",values="payment_type",title="Payment Distribution"),
        px.histogram(df,x="length_of_stay",nbins=30,title="LOS Distribution"),
        px.scatter(df.sample(min(5000,df.shape[0])),x="age",y="length_of_stay",trendline="ols",title="Age vs LOS")
    ]

st.sidebar.title("Hospital Inpatient Discharges")

parquet_path = "/content/drive/MyDrive/cleaned_dataset/cleaned_dataset.parquet"
raw_df = load_parquet(parquet_path) if is_colab() and os.path.exists(parquet_path) else generate_sample_data()

df = clean_data(raw_df)
filtered = filter_dataframe(df)
kpis = compute_kpis(filtered)

st.title("Hospital Inpatient Discharges Dashboard")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total",kpis["total"])
c2.metric("Avg LOS",f"{kpis['avg']:.2f}")
c3.metric("Median LOS",f"{kpis['median']:.2f}")
c4.metric("Readmit %",f"{kpis['readmit']:.2f}")

for fig in plot(filtered):
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Diagnosis Leaderboard")
st.dataframe(filtered.groupby("diagnosis_code").agg(avg=("length_of_stay","mean"),count=("patient_id","count")).sort_values("avg",ascending=False).head(15))
