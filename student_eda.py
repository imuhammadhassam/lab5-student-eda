import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Create Student Performance Dataset
# --------------------------------------------------

data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],
    "Gender": [
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Male"
    ],
    "StudyHours": [
        2.5,
        4.0,
        3.0,
        5.0,
        1.5,
        6.0,
        2.0,
        4.5,
        3.5,
        5.5,
        2.8,
        2.8
    ],
    "Attendance(%)": [
        75,
        88,
        80,
        92,
        70,
        95,
        78,
        90,
        85,
        93,
        82,
        82
    ],
    "Marks": [
        65,
        78,
        70,
        85,
        np.nan,
        90,
        68,
        82,
        75,
        np.nan,
        72,
        72
    ]
}


# Convert dictionary into DataFrame
student_df = pd.DataFrame(data)


# Save dataset to CSV file
student_df.to_csv("student_performance.csv", index=False)


# --------------------------------------------------
# 2. Load CSV File
# --------------------------------------------------

df = pd.read_csv("student_performance.csv")


print("Shape:", df.shape)

print("\nDataset:")
print(df)


print("\nDataset Information:")
df.info()


# --------------------------------------------------
# 3. Check Missing Values and Duplicates
# --------------------------------------------------

print("\nMissing values per column:")
print(df.isnull().sum())


print("\nDuplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 4. Data Cleaning
# --------------------------------------------------

# Remove duplicate row
df_clean = df.drop_duplicates().copy()


# Find median Marks
marks_median = df_clean["Marks"].median()


# Fill missing Marks with median
df_clean["Marks"] = df_clean["Marks"].fillna(marks_median)


print("\nAfter cleaning:", df_clean.shape)


print("\nCleaned Dataset:")
print(df_clean)


# --------------------------------------------------
# 5. Histogram of Marks
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.hist(
    df_clean["Marks"],
    bins=6,
    edgecolor="black"
)

plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("marks_histogram.png")

plt.show()


# --------------------------------------------------
# 6. Boxplot of Marks
# --------------------------------------------------

plt.figure(figsize=(6, 4))

plt.boxplot(df_clean["Marks"])

plt.title("Boxplot of Student Marks")
plt.ylabel("Marks")

plt.tight_layout()

plt.savefig("marks_boxplot.png")

plt.show()


# --------------------------------------------------
# 7. Scatter Plot: Study Hours vs Marks
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    df_clean["StudyHours"],
    df_clean["Marks"]
)

plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.tight_layout()

plt.savefig("studyhours_vs_marks.png")

plt.show()


# --------------------------------------------------
# 8. Correlation Matrix
# --------------------------------------------------

correlation_matrix = df_clean[
    ["StudyHours", "Attendance(%)", "Marks"]
].corr()


print("\nCorrelation Matrix:")

print(correlation_matrix)


# --------------------------------------------------
# 9. Gender-wise Average
# --------------------------------------------------

gender_summary = df_clean.groupby("Gender")[
    ["Marks", "Attendance(%)"]
].mean()


print("\nAverage Marks and Attendance by Gender:")

print(gender_summary)


print(
    "\nSaved marks_histogram.png, "
    "marks_boxplot.png, "
    "studyhours_vs_marks.png"
)