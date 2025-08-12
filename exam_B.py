import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV files
employees_df = pd.read_csv('employees.csv')
managers_df = pd.read_csv('managers.csv')
jobs_df = pd.read_csv('jobs.csv')
positions_df = pd.read_csv('positions.csv')


employees = employees_df.copy()
managers = managers_df.copy()
jobs = jobs_df.copy()
positions = positions_df.copy()

# Print to verify

#print(employees.head())
#print(managers.head())
#print(jobs.head())
#print(positions.head())

#1.a Data Preparation:
"""
duplicate_jobs = jobs[jobs.duplicated(subset=['job_title'], keep=False)]
print("🔎 Found duplicate job titles:")
print(duplicate_jobs.sort_values(by='job_title'))


job_id_to_keep = duplicate_jobs.groupby('job_title')['job_id'].min().to_dict()
print("✅ Job IDs to keep (one per title):")
print(job_id_to_keep)



# Create a mapping: old IDs → new kept ID
job_id_mapping = {}

for title, keep_id in job_id_to_keep.items():
    ids_to_replace = duplicate_jobs[duplicate_jobs['job_title'] == title]['job_id'].tolist()
    for old_id in ids_to_replace:
        job_id_mapping[old_id] = keep_id

# Function to safely replace job_id if it exists in mapping
def replace_job_id(x):
    return job_id_mapping[x] if x in job_id_mapping else x

# Apply the replacement function
positions['job_id'] = positions['job_id'].apply(replace_job_id)

print("✅ Updated job_id values in positions table.")

# Remove duplicate jobs from jobs table
job_ids_to_keep = list(job_id_to_keep.values())
jobs = jobs[jobs['job_id'].isin(job_ids_to_keep)]

print("✅ Removed duplicate jobs from jobs table.")

#1.b
# Function to print missing data report
def missing_report(df, name):
    print(f"\n📊 Missing data report for: {name}")
    print((df.isnull().mean() * 100).round(2).sort_values(ascending=False))


# Function to handle missing values
def handle_missing(df, col, default_value):
    missing_percent = df[col].isnull().mean() * 100

    if missing_percent > 5:
        df[col] = df[col].fillna(default_value)
        print(f"✔️ Filled '{col}' with default value ({default_value}) — Missing: {missing_percent:.2f}%")
    elif missing_percent > 0:
        before = df.shape[0]
        df.dropna(subset=[col], inplace=True)
        after = df.shape[0]
        print(f"🧹 Dropped {before - after} rows due to missing '{col}' — Missing: {missing_percent:.2f}%")
    else:
        print(f"✅ No missing values in '{col}'")

    
    """
"""#2 Data Analysis
# ===== Data Analysis =====


employees_df.columns = employees_df.columns.str.strip()  # Remove whitespace from columns
#1
# ===== a =====
# Count employees by nationality and plot bar chart

nationality_counts = employees['nationality '].value_counts().reset_index()
nationality_counts.columns = ['nationality', 'count']

print(nationality_counts)

plt.figure(figsize=(10, 6))
sns.barplot(data=nationality_counts, x='nationality', y='count', palette='viridis')
plt.title('Number of Employees by Nationality')
plt.xlabel('Nationality')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===== b =====
# Plot employee age distribution histogram

plt.figure(figsize=(8, 5))
sns.histplot(data=employees, x='age', bins=20, kde=True, color='skyblue')
plt.title('Employee Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# ===== c =====
# Create dataframe of top 3 highest and lowest salary employees
top_3_high = employees.nlargest(3, 'salary')
top_3_low = employees.nsmallest(3, 'salary')
top_salaries_df = pd.concat([top_3_high, top_3_low]).reset_index(drop=True)[
    ['employee_id', 'first_name', 'last_name', 'age'
     , 'years_of_experience', 'salary']]
print(top_salaries_df)

# =====d=====
# Plot correlation between years of experience and salary
plt.figure(figsize=(8, 6))
sns.scatterplot(data=employees, x='years_of_experience', y='salary')
plt.title('Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()
plt.show()
# Calculate and print correlation coefficient
correlation = employees['years_of_experience'].corr(employees['salary'])
print(f"Correlation coefficient between years_of_experience and salary: {correlation:.2f}")
"""

#2
# =====a =====
# Categorize managers by performance score and count

# Fix columns whitespace if needed
managers.columns = managers.columns.str.strip()

# Define categories function
def performance_category(score):
    if 1 <= score <= 5:
        return 'Low Performance'
    elif 6 <= score <= 7:
        return 'Medium Performance'
    elif 8 <= score <= 10:
        return 'High Performance'
    else:
        return 'Unknown'

# Apply category
managers['performance_category'] = managers['performance_score'].apply(performance_category)

# Count by category
performance_counts = managers['performance_category'].value_counts().reset_index()
performance_counts.columns = ['performance_category', 'count']

print(performance_counts)

# Plot bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=performance_counts, x='performance_category', y='count', palette='Set2')
plt.title('Managers by Performance Category')
plt.xlabel('Performance Category')
plt.ylabel('Number of Managers')
plt.tight_layout()
plt.show()


# ===== b =====
# Analyze trend of managers joining from 2020 to 2024

# Fix column names whitespace if needed
managers.columns = managers.columns.str.strip()
# Convert starting_date to datetime
managers['starting_date'] = pd.to_datetime(managers['starting_date'], errors='coerce')
# Filter for dates between 2020-01-01 and 2024-12-31
managers_2020_2024 = managers[
    (managers['starting_date'] >= '2020-01-01') &
    (managers['starting_date'] <= '2024-12-31')
]

# Extract year from starting_date
managers_2020_2024['year_joined'] = managers_2020_2024['starting_date'].dt.year

# Count managers per year
join_counts = managers_2020_2024['year_joined'].value_counts().sort_index()
print(join_counts)

# Plot trend line
plt.figure(figsize=(8, 5))
sns.lineplot(x=join_counts.index, y=join_counts.values, marker='o')
plt.title('Number of Managers Joined by Year (2020-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Managers Joined')
plt.grid(True)
plt.tight_layout()
plt.show()


# =====C=====
# Calculate mean salary of managers by location (nationality)

# Fix column names whitespace if needed
managers.columns = managers.columns.str.strip()

mean_salary_by_location = managers.groupby('nationality')['salary'].mean().reset_index()
mean_salary_by_location = mean_salary_by_location.sort_values(by='salary', ascending=False)

print(mean_salary_by_location)

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=mean_salary_by_location, x='nationality', y='salary', palette='coolwarm')
plt.title('Mean Salary of Managers by Nationality')
plt.xlabel('Nationality')
plt.ylabel('Mean Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ===== =d =====
# Remove whitespace from columns just in case
employees.columns = employees.columns.str.strip()
managers.columns = managers.columns.str.strip()

# Convert starting_date to datetime
managers['starting_date'] = pd.to_datetime(managers['starting_date'], errors='coerce')

# Filter managers hired from 2022-01-01
managers_2022_up = managers[managers['starting_date'] >= '2022-01-01']

# Count number of employees per manager_id
employees_per_manager = employees['manager_id'].value_counts().reset_index()
employees_per_manager.columns = ['manager_id', 'num_employees']

# Merge managers_2022_up with counts
managers_with_counts = pd.merge(managers_2022_up, employees_per_manager, on='manager_id', how='left')

# Some managers might not have employees, fill NaN with 0
managers_with_counts['num_employees'] = managers_with_counts['num_employees'].fillna(0)

# Calculate mean number of employees managed
mean_employees_managed = managers_with_counts['num_employees'].mean()

print(f"Mean number of employees managed by managers hired from 2022-01-01: {mean_employees_managed:.2f}")

# ===== e =====
# Check correlation between number of employees managed and salary

plt.figure(figsize=(8, 6))
sns.scatterplot(data=managers_with_counts, x='num_employees', y='salary')
plt.title('Number of Employees Managed vs Salary')
plt.xlabel('Number of Employees Managed')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = managers_with_counts['num_employees'].corr(managers_with_counts['salary'])
print(f"Correlation coefficient: {correlation:.2f}")


print(managers_with_counts[['num_employees', 'salary']].describe())
print(managers_with_counts[['num_employees', 'salary']].head(10))



#   ___Q3___
## ===== a =====
# Count number of jobs by department

# Remove whitespace from columns if needed
jobs.columns = jobs.columns.str.strip()

# Count jobs by department
jobs_count_by_dept = jobs.groupby('department')['job_id'].count().reset_index()
jobs_count_by_dept.columns = ['department', 'job_count']

print(jobs_count_by_dept)

# Plot bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=jobs_count_by_dept, x='department', y='job_count', palette='viridis')
plt.title('Number of Jobs by Department')
plt.xlabel('Department')
plt.ylabel('Number of Jobs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#____Q4____

# ===== a =====
# Count number of positions per employee

# Remove whitespace if needed
positions.columns = positions.columns.str.strip()

# Count positions per employee_id
positions_count = positions.groupby('employee_id').size().reset_index(name='position_count')

# Count how many employees have 0,1,2,... positions
# Find employees with 0 positions
employees_with_positions = positions_count['employee_id'].unique()
all_employees = employees['employee_id'].unique()

employees_zero_positions = [emp for emp in all_employees if emp not in employees_with_positions]
zero_positions_df = pd.DataFrame({'employee_id': employees_zero_positions, 'position_count': 0})

# Combine zero and other counts
positions_count_all = pd.concat([positions_count, zero_positions_df], ignore_index=True)

# Count how many employees have each number of positions
positions_summary = positions_count_all['position_count'].value_counts().reset_index()
positions_summary.columns = ['position_count', 'employee_count']
positions_summary = positions_summary.sort_values('position_count')

print(positions_summary)

# Plot bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=positions_summary, x='position_count', y='employee_count', palette='magma')
plt.title('Number of Employees by Position Count')
plt.xlabel('Number of Positions')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.show()



# ===== b =====
# Count employees working in positions with department different from their nationality

# Remove whitespace from columns for all relevant DataFrames
positions.columns = positions.columns.str.strip()
employees.columns = employees.columns.str.strip()
jobs.columns = jobs.columns.str.strip()

# Merge positions with jobs to get job details including department
positions_jobs = pd.merge(positions, jobs, on='job_id', how='left')

# Merge the above with employees to get employee details including nationality
pos_job_emp = pd.merge(positions_jobs, employees, on='employee_id', how='left')

# Filter rows where employee nationality is different from job department (assuming department is geographical)
diff_nat = pos_job_emp[pos_job_emp['nationality'] != pos_job_emp['department']]

# Count unique employees that work in positions different from their nationality
count_diff_nat = diff_nat['employee_id'].nunique()

print(f"Number of employees working in positions different from their nationality: {count_diff_nat}")


# ===== c =====
# Calculate mean performance score for each employee based on all positions, add to employees DataFrame

# Remove whitespace from columns if needed
positions.columns = positions.columns.str.strip()
employees.columns = employees.columns.str.strip()

# Group positions by employee_id and calculate mean performance score
mean_perf_score = positions.groupby('employee_id')['performance_score'].mean().reset_index()
mean_perf_score.columns = ['employee_id', 'mean_performance_score']

# Merge the mean score back to employees DataFrame
employees = pd.merge(employees, mean_perf_score, on='employee_id', how='left')

# For employees with no positions, fill NaN with 0 or appropriate value
employees['mean_performance_score'] = employees['mean_performance_score'].fillna(0)

print(employees[['employee_id', 'mean_performance_score']].head(10))



# ===== d =====
# Remove whitespace from columns if needed
managers.columns = managers.columns.str.strip()
employees.columns = employees.columns.str.strip()

# Merge employees with their managers' performance scores
emp_mgr_perf = pd.merge(
    employees[['employee_id', 'manager_id', 'mean_performance_score']],
    managers[['manager_id', 'performance_score']],
    on='manager_id',
    how='left',
    suffixes=('_employee', '_manager')
)

# Plot scatter plot to visualize correlation
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=emp_mgr_perf,
    x='mean_performance_score',
    y='performance_score'
)
plt.title('Employee Mean Performance vs Manager Performance')
plt.xlabel('Employee Mean Performance Score')
plt.ylabel('Manager Performance Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = emp_mgr_perf['mean_performance_score'].corr(emp_mgr_perf['performance_score'])
print(f"Correlation coefficient: {correlation:.2f}")


# ===== e =====
# Clean column names
employees.columns = employees.columns.str.strip()
positions.columns = positions.columns.str.strip()
jobs.columns = jobs.columns.str.strip()

# Merge employees with positions and jobs
emp_pos_jobs = pd.merge(positions, employees[['employee_id', 'salary']], on='employee_id', how='left')
emp_pos_jobs = pd.merge(emp_pos_jobs, jobs[['job_id', 'department']], on='job_id', how='left')

# Calculate mean salary, count employees, and total salary by department
dept_salary_stats = emp_pos_jobs.groupby('department').agg(
    mean_salary=('salary', 'mean'),
    employee_count=('employee_id', 'nunique'),
    total_salary=('salary', 'sum')
).reset_index()

# Sort descending by mean salary and get top 3 departments
top_3_depts = dept_salary_stats.sort_values(by='mean_salary', ascending=False).head(3)

print("Top 3 departments with detailed salary stats:")
print(top_3_depts)


# ===== f =====
# Remove whitespace if needed
employees.columns = employees.columns.str.strip()
managers.columns = managers.columns.str.strip()

# Merge employees with managers to get manager performance score
emp_mgr_exp_perf = pd.merge(
    employees[['employee_id', 'manager_id', 'years_of_experience']],
    managers[['manager_id', 'performance_score']],
    on='manager_id',
    how='left'
)

# Plot scatter plot to visualize correlation
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=emp_mgr_exp_perf,
    x='years_of_experience',
    y='performance_score'
)
plt.title('Employee Years of Experience vs Manager Performance Score')
plt.xlabel('Employee Years of Experience')
plt.ylabel('Manager Performance Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = emp_mgr_exp_perf['years_of_experience'].corr(emp_mgr_exp_perf['performance_score'])
print(f"Correlation coefficient: {correlation:.2f}")





