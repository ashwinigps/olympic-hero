# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Path of the file
print(path)
data = pd.read_csv(path)
data.rename(columns={'Total':'Total_Medals'}, inplace=True)
print(data.head(10))

#Code starts here



# --------------
#Code starts here
data['Better_Event']=np.where(data['Total_Summer']==data['Total_Winter'], 'Both', 
        np.where(data['Total_Summer']>data['Total_Winter'], 'Summer','Winter'))  
better_event=data['Better_Event'].value_counts().idxmax()
print(better_event)


# --------------
#Code starts here
from functools import reduce

top_countries = pd.DataFrame(data, columns = ['Country_Name','Total_Summer', 'Total_Winter','Total_Medals']) 
top_countries=top_countries.drop(top_countries.tail(1).index)
def top_ten(top_countries, col_name):
    country_list = []
    top_10 = top_countries.nlargest(10, col_name)
    country_list = top_10['Country_Name'].tolist()
    return (country_list)

top_10_summer = top_ten(top_countries, 'Total_Summer')
top_10_winter = top_ten(top_countries, 'Total_Winter')
top_10 = top_ten(top_countries, 'Total_Medals')

common = reduce(np.intersect1d, [top_10_summer, top_10_winter, top_10]).tolist()
print(common)



# --------------
#Code starts here
#print(top_10_summer)


#print(data.columns)
summer_df = data[data['Country_Name'].isin(top_10_summer)]
winter_df = data[data['Country_Name'].isin(top_10_winter)]
top_df = data[data['Country_Name'].isin(top_10)]

def plot_bar(df, col1, col2):
    index = np.arange(len(df[col1]))
    plt.bar(index, df[col2])
    plt.xlabel(col1, fontsize=10)
    plt.ylabel(col2, fontsize=10)
    plt.xticks(index, df[col1], fontsize=10, rotation=45)
    plt.title('Total Medals')
    plt.show()
    plt.plot()

plot_bar(summer_df, 'Country_Name', 'Total_Summer')    
plot_bar(winter_df, 'Country_Name', 'Total_Winter') 
plot_bar(top_df, 'Country_Name', 'Total_Medals') 




# --------------
#Code starts here
summer_df['Golden_Ratio'] = summer_df.Gold_Summer/summer_df.Total_Summer
summer_max_ratio = summer_df.Golden_Ratio.idxmax()
top_row = summer_df.loc[summer_df.Golden_Ratio.idxmax()]
summer_max_ratio = top_row.Golden_Ratio
summer_country_gold = top_row.Country_Name
print(summer_country_gold, summer_max_ratio)

winter_df['Golden_Ratio'] = winter_df.Gold_Winter/winter_df.Total_Winter
winter_max_ratio = winter_df.Golden_Ratio.idxmax()
top_row = winter_df.loc[winter_df.Golden_Ratio.idxmax()]
winter_max_ratio = top_row.Golden_Ratio
winter_country_gold = top_row.Country_Name
print(winter_country_gold, winter_max_ratio)

top_df['Golden_Ratio'] = top_df.Gold_Total/top_df.Total_Medals
top_row = top_df.loc[top_df.Golden_Ratio.idxmax()]
top_max_ratio = top_row.Golden_Ratio
top_country_gold = top_row.Country_Name

print(top_country_gold, top_max_ratio)


# --------------
#Code starts here
data_1 = data.drop(data.tail(1).index)
print(data_1.columns)
data_1['Total_Points'] = (data_1.Gold_Total*3) + (data_1.Silver_Total*2) + data_1.Bronze_Total
row = data_1.loc[data_1.Total_Points.idxmax()]
most_points = row.Total_Points
best_country = row.Country_Name
print(most_points, best_country)


# --------------
#Code starts here
df_filter = data.Country_Name == best_country
best = data[df_filter]

best = best[['Gold_Total','Silver_Total','Bronze_Total']]
print(best)
plt.xlabel('United States', fontsize=10)
plt.ylabel('Medals Tally', fontsize=10)
plt.xticks(fontsize=10, rotation=45)
#plt.title('Best Graph')
best.plot.bar()


