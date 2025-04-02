import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

#The CSV file path.
filePath = r"C:\Users\Xeron\OneDrive\Documents\Programs\MachineLearning\DSA2025MidSemMarks.csv"

#Load the dataset and drop the 'Name' column.
df = pd.read_csv(filePath)
df.drop(columns=['Name'], inplace=True)

#Convert the Marks column to numeric form.
df['Marks'] = pd.to_numeric(df['Marks'], errors='coerce')

#The function for extraction of the numeric part of the enrolment number.
def extractNumeric(enroll):
    match = re.search(r'(\d+)', enroll)
    return int(match.group(1)) if match else None

#Add a new column for the numeric part from the Enrolment Number.
df['EnrolNum'] = df['Enrolment Number'].apply(extractNumeric)

#Section-B2: IT students with enrolment numbers up to IIT2024214.
sectionB2 = df[(df['Enrolment Number'].str.startswith('IIT')) & (df['EnrolNum'] <= 2024214)]

#Section-C: IIT students with enrolment numbers from IIT2024215 onward, IIT2024501 and all IT(BI) students.
sectionC = df[((df['Enrolment Number'].str.startswith('IIT')) & (df['EnrolNum'] >= 2024215)) |
               (df['Enrolment Number'].str.startswith('IIB'))]

#IT(BI): IIB students with enrolment numbers from IIB2024001 to IIB2024045 and IIB2024501.
it_bi = df[df['Enrolment Number'].str.startswith('IIB')]
it_bi = it_bi[((it_bi['EnrolNum'] >= 2024001) & (it_bi['EnrolNum'] <= 2024045)) |
              (it_bi['EnrolNum'] == 2024501)]

#The function to display statistical analyses.
def displayStats(data, label):
    print(f"Statistics for {label}:")
    print("Mean:", data['Marks'].mean())
    print("Median:", data['Marks'].median())
    print("Standard Deviation:", data['Marks'].std())
    print("Variance:", data['Marks'].var())
    print("75th Percentile:", data['Marks'].quantile(0.75))
    print()

#Displaying statistics for each category.
displayStats(df, "Combined")
displayStats(sectionB2, "Section-B2")
displayStats(sectionC, "Section-C")
displayStats(it_bi, "IT(BI)")

#The Histogram plots with separate KDEs (Kernel Distribution Curves).
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
dataPlots = [
    (df, "Combined Marks Distribution"),
    (sectionB2, "Section-B2 Marks Distribution"),
    (sectionC, "Section-C Marks Distribution"),
    (it_bi, "IT(BI) Marks Distribution")
]

for ax, (data, title) in zip(axes.flatten(), dataPlots):
    sns.histplot(data['Marks'], stat='density', kde=False, bins=20, ax=ax, color='skyblue')
    sns.kdeplot(data['Marks'], ax=ax, color='red')
    ax.set_title(title)
    ax.set_xlabel("Marks")
    ax.set_ylabel("Density")
    plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

#The pie charts for Pass vs Fail.
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
groups = [
    (df, "Combined"),
    (sectionB2, "Section-B2"),
    (sectionC, "Section-C"),
    (it_bi, "IT(BI)")
]

for ax, (data, label) in zip(axes2.flatten(), groups):
    numPass = data[data['Marks'] >= 4].shape[0]
    numFail = data[data['Marks'] < 4].shape[0]
    ax.pie([numPass, numFail],
           labels=[f'Passed (>=4)', f'Failed (<4)'],
           autopct='%1.1f%%',
           colors=['green', 'red'],
           startangle=90)
    ax.text(0.5, -0.1, f"{label} Pass vs Fail", transform=ax.transAxes,
            ha='center', fontweight='bold', fontsize=12)
    
plt.tight_layout()
plt.show()

#The Count plots for Marks Buckets for each category.
bins = [-0.1, 4, 8, 12, 16, 20, 25]
labelsBuckets = ["0-4", "5-8", "9-12", "13-16", "17-20", "21-25"]

dfCombined = df.copy()
dfCombined['Marks Buckets'] = pd.cut(dfCombined['Marks'], bins=bins, labels=labelsBuckets)

dfB2 = sectionB2.copy()
dfB2['Marks Buckets'] = pd.cut(dfB2['Marks'], bins=bins, labels=labelsBuckets)

dfC = sectionC.copy()
dfC['Marks Buckets'] = pd.cut(dfC['Marks'], bins=bins, labels=labelsBuckets)

dfBI = it_bi.copy()
dfBI['Marks Buckets'] = pd.cut(dfBI['Marks'], bins=bins, labels=labelsBuckets)

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
datasets = [
    (dfCombined, "Combined"),
    (dfB2, "Section-B2"),
    (dfC, "Section-C"),
    (dfBI, "IT(BI)")
]

for ax, (data, title) in zip(axes3.flatten(), datasets):
    palette = sns.color_palette("viridis", len(labelsBuckets))
    sns.countplot(x='Marks Buckets', data=data, palette=palette, ax=ax)
    ax.set_title(f"{title} Marks Buckets Count", fontweight='bold')
    ax.set_xlabel("Marks Buckets")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=45)
    
plt.tight_layout()
plt.show()
