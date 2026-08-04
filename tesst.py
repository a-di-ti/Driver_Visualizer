import pandas as pd
import numpy as np

# df = pd.read_csv("ar_little_rock_2020_04_01.csv")
# # print(df.columns)
# print(len(df))
# print(df.head)
# print(df.isnull().sum())

# columns_of_interest = ["date", "time", "subject_race", "subject_race", "subject_age", "search_conducted", "stop_outcome"]

# df["date"] = pd.to_datetime(df["date"])

# df = df.drop(columns=['lat', 'lng', 'raw_defendant_row'])

# df.dropna(subset = ["subject_age"])


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# 1. Load Data
# Handle direct CSV or zipped CSV automatically
try:
    df = pd.read_csv('ar_little_rock_2020_04_01.csv')
except FileNotFoundError:
    df = pd.read_csv('yg821jf8611_ar_little_rock_2020_04_01.csv.zip')

# 2. Data Cleaning & Feature Engineering
df['date'] = pd.to_datetime(df['date'])
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

# 3. Create Dashboard Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Little Rock, AR Traffic Stops Analysis (2017)', fontsize=16, fontweight='bold', y=0.98)

# Chart 1: Monthly Traffic Stop Volume
monthly_counts = df.groupby('month').size().reset_index(name='stop_count')
sns.lineplot(
    data=monthly_counts, 
    x='month', 
    y='stop_count', 
    ax=axes[0, 0], 
    marker='o', 
    color='#1f77b4', 
    linewidth=2.5
)
axes[0, 0].set_title('Monthly Traffic Stop Volume', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Number of Stops')
axes[0, 0].tick_params(axis='x', rotation=30)

# Chart 2: Stops by Race and Sex
race_sex = df.dropna(subset=['subject_race', 'subject_sex'])
top_races = ['black', 'white', 'asian/pacific islander']
race_sex_counts = (
    race_sex[race_sex['subject_race'].isin(top_races)]
    .groupby(['subject_race', 'subject_sex'])
    .size()
    .reset_index(name='count')
)

sns.barplot(
    data=race_sex_counts, 
    x='subject_race', 
    y='count', 
    hue='subject_sex', 
    ax=axes[0, 1], 
    palette='Set2'
)
axes[0, 1].set_title('Stops by Subject Race & Sex', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Subject Race')
axes[0, 1].set_ylabel('Number of Stops')
axes[0, 1].legend(title='Sex')

# Chart 3: Subject Age Distribution
sns.histplot(
    data=df['subject_age'].dropna(), 
    ax=axes[1, 0], 
    bins=30, 
    kde=True, 
    color='#2ca02c'
)
axes[1, 0].set_title('Subject Age Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

# Chart 4: Hourly Distribution of Stops
hourly_counts = df.groupby('hour').size().reset_index(name='count')
sns.barplot(
    data=hourly_counts, 
    x='hour', 
    y='count', 
    ax=axes[1, 1], 
    color='#ff7f0e'
)
axes[1, 1].set_title('Stops by Hour of Day', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Hour (24-Hour Format)')
axes[1, 1].set_ylabel('Number of Stops')

# Adjust layout and export
plt.tight_layout()
plt.savefig('little_rock_traffic_stops_dashboard.png', dpi=300)
plt.show()
