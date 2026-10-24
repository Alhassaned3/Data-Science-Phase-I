#!/usr/bin/env python
# coding: utf-8

# # Homework Assignment 2 : Exploratory and Statistical Analysis of Bike-Sharing Demand  
# CSCI-473: Applied Data Science  

# In[ ]:





# ## Task 1: Load and Audit the Dataset
# Load the provided CSV file into a pandas DataFrame named df.
# Your program must display:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# from google.colab import drive
# drive.mount('/content/drive')
# 
# import pandas as pd
# 
# Update the path to match where your file is located in Google Drive
# 
# file_path = '/content/drive/MyDrive/bike_sharing_hour.csv'
# df = pd.read_csv(file_path)

# In[2]:


# output path 
from pathlib import Path
output_folder = Path("output")
output_folder.mkdir(parents=True, exist_ok=True)


# In[3]:


df = pd.read_csv("bike_sharing_hour.csv")


# In[4]:


# 1. The first five rows
print("First 5 rows: ")
df.head(5)


# In[5]:


# 2. the last five rows;
print("Last 5 rows:")
df.tail(5)


# In[6]:


#3. the number of rows and columns;
print(f"This data set contains {df.shape[0]} rows")
print(f"This data set contains {df.shape[1]} columns")


# In[7]:


# 4. all column names;
df.columns


# In[8]:


# 5. the data type of every column;
print("Data file Data types and column names:")
df.dtypes


# In[9]:


# 6. the output of info();
print("Data file Info:")
df.info()


# In[10]:


# 7. descriptive statistics for the numerical columns
print("Descriptive statistics for the numerical columns:")
df.describe()


# In[11]:


# 8. the number of missing values in each column; and
print("The missing values per column:")
df.isna().sum()


# In[12]:


# 9. the number of exact duplicate rows.
print("The number of duplicated rows:", df.duplicated().sum())


# ## Task 2: Prepare the Variables for Analysis

# In[13]:


# Create a working copy of the original DataFrame before making changes:
raw_df = df.copy()
clean_df = df.copy()

# Rename columns according to mapping
clean_df = clean_df.rename(columns={
    "dteday": "date",
    "yr": "year_code",
    "mnth": "month",
    "hr": "hour",
    "weathersit": "weather_code",
    "temp": "temperature_normalized",
    "atemp": "apparent_temperature_normalized",
    "hum": "humidity_normalized",
    "windspeed": "wind_speed_normalized",
    "casual": "casual_rentals",
    "registered": "registered_rentals",
    "cnt": "total_rentals"
})


# In[14]:


# 1. convert date to a pandas date-time data type using pd.to datetime() with errors="coerce";
clean_df["date"] = pd.to_datetime(clean_df["date"], errors = "coerce")


# In[15]:


# 2. create a year column containing 2011 or 2012;
clean_df["year"] = clean_df["date"].dt.year


# In[16]:


# 3. create a season label column containing Spring, Summer, Fall, or Winter;
season_map = {
1: "Spring",
2: "Summer",
3: "Fall",
4: "Winter"
}
clean_df["season_label"] = (
clean_df["season"].map(season_map))


# In[17]:


### 4. create a weather label column using the following labels:
#  Clear/Partly Cloudy, Mist/Cloudy,  Light Rain/Snow, and Heavy Rain/Snow/Fog
#  for weather codes 1, 2, 3, and 4, respectively;
weather_map = {
    1 : "Clear/Partly Cloudy",
    2 : "Mist/Cloudy",
    3 : "Light Rain/Snow",
    4 : "Heavy Rain/Snow/Fog"
}
clean_df['weather_label'] = clean_df["weather_code"].map(weather_map)


# In[18]:


# 5. create a day type column that maps workingday=1 to Working Day and workingday=0 to
workday_map= {
    1: "Working Day",
    0: "Nonworking Day"
}

clean_df["day_type"] = clean_df["workingday"].map(workday_map)


# In[19]:


# 6. create temperature c, apparent temperature c, humidity percent, and wind speed using the conversion formulas below; and

clean_df["temperature_c"] = (47 * clean_df["temperature_normalized"] - 8)
clean_df["apparent_temperature_c"] = (66 * clean_df['apparent_temperature_normalized'] - 16)
clean_df['humidity_percent'] = (100 * clean_df["humidity_normalized"])
clean_df['wind_speed'] = (67 * clean_df['wind_speed_normalized'])


# In[20]:


#7. print the resulting column names and data types, followed by a small sample of the prepared rows
clean_df.info()


# In[21]:


clean_df[['year', 'season_label', 'weather_label', 'day_type', 'temperature_c',
  'apparent_temperature_c', 'humidity_percent', 'wind_speed']].head(5)


# ## Task 3: Validate the Prepared Dataset
# Perform the following validation checks

# In[22]:


## 1. verify that instant contains unique values;
try:
  # 1. verify that instant contains unique values;
    assert clean_df["instant"].is_unique, "instant contains unique values"
    print("Confirmed: Instant does not contain unique values")

    # 2. verify that hour is between 0 and 23;
    assert clean_df["hour"].between(0, 23).all(), "Hour column contains values outside the 0-23 range"
    print("Confirmed: All hours fall between 0 and 23")

    # 3. verify that month is between 1 and 12;
    assert clean_df["month"].between(1, 12).all(), "Month column contains values outside the 1-12 range"
    print("Confirmed: All months fall between 1 and 12")

except AssertionError as e:
  print(f"Assertion Failed: {e}")


# In[23]:


  # 4. verify that season and weather code contain only their documented codes;

allowed_seasons = {"Spring", "Summer", "Fall", "Winter"}
allowed_weathers = {"Clear/Partly Cloudy", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain/Snow/Fog"}

try:
    assert clean_df['season_label'].isin(allowed_seasons).all(), "Verify Seasons"
    print("Confirmed: Seasons codes Verified")

    assert clean_df['weather_label'].isin(allowed_weathers).all(), "Verify Weather Code"
    print("Confirmed: Weather Codes Verified")

except AssertionError as e:
    print(f"Assertion Failed: {e}")


# In[24]:


# 6. verify that all rental-count columns contain nonnegative values;
try:
  assert (clean_df[['total_rentals', 'casual_rentals', 'registered_rentals']] >= 0).all().all(), "Rental count is nonnegative"
  print("Confirmed: Rental counts are all postive")
# 7. verify that total rentals = casual rentals + registered rentals;
  assert (clean_df['total_rentals'] == clean_df["casual_rentals"] + clean_df['registered_rentals']).all()
  print("Confirmed: Total rentals = casual rentals + registered rentals")
# 8. verify that date, hour, season label, weather label, temperature c, humidity percent, wind speed,
    # and total rentals contain no missing values;
  assert not clean_df[["date","hour","season","weather_label",
          "temperature_c","humidity_percent","wind_speed",
          "total_rentals"]].isna().any().any(), "Missing Values exist in dataset"
  print("Confirmed: No missing values found")
# 9. verify that the prepared dataset contains no exact duplicate rows.
  assert not clean_df.duplicated().any(), "duplicated entries exist in dataset"
  print("Confirmed: No duplicated entries found")
except AssertionError as e:
  print(f"Assertion Failed: {e}")


# In[ ]:





# ## Task 4: Exploratory Data Analysis and Visualization

# ### A. Numerical and categorical summaries
# Calculate and display:
# 

# In[25]:


print(" 1. the mean, median, standard deviation, minimum, maximum, and quartiles of total rentals;")
clean_df['total_rentals'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)


# In[26]:


print("2. the number of observations in each season")
clean_df.value_counts("season_label")


# In[27]:


print("and weather category;")
clean_df.value_counts("weather_label")


# In[28]:


print("the mean and median total rentals for working and nonworking days;")
clean_df.groupby('day_type')['total_rentals'].agg(["mean", "median"]).round(2)


# In[29]:


print("4. the mean total rentals for every hour of the day;")
clean_df.groupby('hour')['total_rentals'].agg('mean').round(2)


# In[30]:


print(" 5. the number and mean total rentals grouped by both season label and workingday.")
clean_df.groupby(['season_label', 'day_type'])['total_rentals'].agg(['count', 'mean'
         ]).round(2)


# ### B. Required visualizations
# Create and save all four visualizations:

# In[31]:


# 1.a histogram of total rentals;
plt.figure(figsize=(8, 5))
sns.histplot(
 clean_df,
    x = "total_rentals"
)
plt.title("Distribution of Hourly Total Rentals")
plt.xlabel("Total Rentals per Hour")
plt.ylabel("Number of Hours")
plt.tight_layout()
plt.savefig(output_folder / "total_rentals_histogram.png", dpi=300)
plt.close()


# In[32]:


# 2. a box plot of total rentals by season;
plt.figure(figsize=(8, 5))
sns.boxplot(
    data = clean_df,
    x = "season_label",
    y = "total_rentals",
    hue = "season_label",
    order = ["Spring", "Summer", "Fall", "Winter"],
    legend = False
)
plt.title("Hourly Total Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Total Rentals per Hour")
plt.tight_layout()
plt.savefig(output_folder / "rentals_by_season_boxplot.png", dpi=300)
plt.close()


# In[33]:


# 3. a line plot of mean total rentals by hour, with separate lines for working and nonworking days;
hourly_summary = clean_df.groupby(
    ["hour", "day_type"]
)["total_rentals"].mean().reset_index()

plt.figure(figsize=(9, 5))
sns.lineplot(
    data=hourly_summary,
    x="hour",
    y="total_rentals",
    hue="day_type",
    marker="o"
)
plt.title("Mean Hourly Rentals: Working vs Nonworking Days")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Total Rentals")
plt.xticks(range(0, 24))
plt.legend(title="Day Type")
plt.tight_layout()
plt.savefig(output_folder / "hourly_demand.png", dpi=300)
plt.close()


# In[34]:


## 4. a scatter plot of temperature in degrees Celsius versus total rentals, with color representing season.
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data= clean_df,
    x="temperature_c",
    y= 'total_rentals',
    hue = 'season_label',
    hue_order = ["Spring", "Summer", "Fall", "Winter"],
    s=15,
    alpha=0.6
)
plt.title("Temperature vs Hourly Total Rentals by Season")
plt.xlabel("Temperature (°C)")
plt.ylabel("Total Rentals per Hour")
plt.legend(title="Season")
plt.tight_layout()
plt.savefig(output_folder / "temperature_vs_rentals_scatter.png", dpi=300)
plt.close()


# ## Task 5: Probability, Correlation, and Interpretation

# ### A. Empirical and conditional probability
# Define a high-demand hour as an observation whose total rentals value is greater than or equal
# to the dataset’s 75th percentile. Create a Boolean column named high demand.  
# Calculate and report:   
# Show the numerator and denominator used for every conditional probability. Interpret the results
# in context

# In[35]:


# 1. the 75th-percentile threshold;
high_demand_threshold = clean_df['total_rentals'].quantile(0.75)
print("75th-percentile high-demand threshold:", high_demand_threshold, "rentals")


# In[36]:


# Create a Boolean column named high demand.
clean_df['high_demand']=(clean_df['total_rentals']>= high_demand_threshold)


# In[37]:


# 2. P(high demand);
high_demand_count = clean_df['high_demand'].sum()
total_count = len(clean_df)
p_high_demand = (high_demand_count / total_count)
print(f"P(high demand) = {high_demand_count} / {total_count} = {p_high_demand:.4f} ({p_high_demand * 100:.1f}%)")


# In[38]:


# 3. P(high demand | working day);
working_rows = clean_df["workingday"] == 1

high_and_working = (clean_df['high_demand'] & working_rows).sum()
p_high_given_working = high_and_working / working_rows.sum()
print(f"P(high demand | working day) = {high_and_working} / {working_rows.sum()} = {p_high_given_working:.4f} ({p_high_given_working * 100:.1f}%)")


# In[39]:


# 4. P(high demand | nonworking day);
nonworking_rows = clean_df['workingday'] == 0
high_and_nonworking = (clean_df['high_demand'] & nonworking_rows).sum()
p_high_given_nonworking = high_and_nonworking / nonworking_rows.sum()
print(f"P(high demand | nonworking day) = {high_and_nonworking} / {nonworking_rows.sum()} = {p_high_given_nonworking:.4f} ({p_high_given_nonworking * 100:.1f}%)")


# In[40]:


# 5. P(high demand | clear or partly cloudy weather)
good_weather = (clean_df['weather_code'] == 1)

high_and_good_weather = (clean_df['high_demand'] & good_weather).sum()
p_high_given_good_weather = high_and_good_weather / good_weather.sum()
print(f"P(high demand | clear/partly cloudy) = {high_and_good_weather} / {good_weather.sum()} = {p_high_given_good_weather:.4f} ({p_high_given_good_weather * 100:.1f}%)")


# ## B. Pearson correlation
# Create a Pearson correlation matrix using:  
# • temperature c,  
# • apparent temperature c,  
# • humidity percent,  
# • wind speed, and  
# • total rentals.  
# Display the matrix rounded to two decimal places and create an annotated heatmap.  
# Identify the weather variable with the strongest absolute Pearson correlation with total rentals. Report its
# correlation coefficient and interpret its direction and strength carefully.
# 

# In[41]:


# define columns used in matrix
corr_cols = [
    'temperature_c',
    'apparent_temperature_c',
    'humidity_percent',
    'wind_speed',
    'total_rentals'
]


# In[42]:


#create matrix
corr_matrix = clean_df[corr_cols].corr(method='pearson').round(2)
display(corr_matrix)


# In[43]:


plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot= True,
    cmap= 'coolwarm',
    linewidths=0.5,
    vmin=-1, vmax=1
)
plt.title('Pearson Correlation Heatmap')
plt.tight_layout()
plt.savefig(output_folder / "correlation_heatmap.png", dpi=300)
plt.close()


# ## Task 6: Sampling, Bootstrap Uncertainty, and Export

# ### A. One reproducible random sample  
# 
# Select a simple random sample of 500 rows without replacement using random state=42.  
# Calculate the mean total rentals for:  
# • the complete dataset;   
# • the 500-row sample.  
# Report the difference and explain why the two means need not be identical.

# In[44]:


## produce random sample
random_sample = clean_df.sample(n=500, random_state=42)


# In[45]:


print(" Mean total rentals for the complete dataset:", clean_df['total_rentals'].mean().round(2))


# In[46]:


print(" Mean total rentals for the 500-row sample:", random_sample['total_rentals'].mean().round(2))


# ### B. Repeated sampling
# Draw 200 random samples without replacement, each containing 500 rows. Use a different reproducible sample during each repetition. Store the mean total rentals from every sample. Then:
# 
# 

# In[47]:


## 1. report the mean of the 200 sample means;
## 2. report the standard deviation of the sample means;
## Use a reproducible random-number generator. Do not reset the random-number generator inside the loop.

# reproducible number generator with seed 42
rng = np.random.default_rng(42)
sample_means = []

# Draw 200 random samples
for _ in range(200):
  # without replacement, each containing 500 rows, different reproducable sample
  sub_sample = clean_df.sample(n= 500, replace = False, random_state=int(rng.integers(0, 100000)))
  #for each sample, calculate mean of total rentals and append it to sample mean list
  sample_means.append(sub_sample["total_rentals"].mean())

sample_means = pd.Series(sample_means)

print("Mean of the 200 sample means:", sample_means.mean().round(2))
print("Standard deviation of the 200 sample means:", sample_means.std().round(2))


# In[48]:


## 3. create a histogram of the sample means with a vertical line showing the full-dataset mean.
plt.figure(figsize=(8, 4))
sns.histplot(
    sample_means,
    kde = True
)
plt.axvline(clean_df['total_rentals'].mean(), color='red', linestyle='--', linewidth=2, label='Full-Dataset Mean')
plt.title('Distribution of 200 Sample Means (n=500)')
plt.xlabel('Sample Mean Total Rentals')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(output_folder / "sample_means_histogram.png", dpi=300)
## plt.close()


# ### C. Bootstrap confidence interval
# Using the observed total rentals values:
# 
# 
# Interpret the confidence interval correctly. Do not describe it as the range containing 95% of
# individual hourly rental count

# In[49]:


# 1. generate 5,000 bootstrap samples of the same size as the original dataset, sampling with replacement;
# 2. calculate the mean for each bootstrap sample;

rng = np.random.default_rng(42)
rentals = clean_df["total_rentals"].to_numpy()

bootstrap_means = []

for _ in range(5000):
  bootstrap_sample = rng.choice(
    rentals,
    size=len(rentals),
    replace=True
  )
  bootstrap_means.append(bootstrap_sample.mean())

bootstrap_means = np.array(bootstrap_means)


# In[50]:


bootstrap_means = np.array(bootstrap_means)

# 3. estimate the bootstrap standard error; the standard deviation of a collection of sample means is the Standard Error
print("Bootstrap Standard Error:", bootstrap_means.std(ddof=1).round(2))


# In[51]:


# 4. Use the 2.5th and 97.5th percentiles to construct an approximate 95% bootstrap confidence
#    interval for mean hourly rentals.

lower, upper = np.percentile(
bootstrap_means,
[2.5, 97.5]
)
print("95% Bootstrap Confidence Interval falls between", lower.round(2),'and', upper.round(2))


# ### Export

# In[52]:


# Save the prepared DataFrame without the index
prepared_path = output_folder / "bike_sharing_prepared.csv"
shape_before_saving = clean_df.shape
clean_df.to_csv(prepared_path, index=False)
print("Saved prepared dataset to:", prepared_path)

# Reload the saved file into a new DataFrame
verified_df = pd.read_csv(prepared_path)

print("\nReloaded dataset - first five rows:")
display(verified_df.head())
print("Reloaded dataset - shape:", verified_df.shape)
print("\nReloaded dataset - missing values per column:")
print(verified_df.isna().sum())

# Verify that the reloaded shape matches the shape before saving
assert verified_df.shape == shape_before_saving, "Reloaded shape does not match the shape before saving"
print(f"\nEXPORT VERIFIED: reloaded shape {verified_df.shape} matches shape before saving {shape_before_saving}")


# In[ ]:




