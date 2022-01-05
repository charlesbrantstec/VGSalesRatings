from inspect import isgenerator
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.matrix import heatmap
import itertools
import matplotlib.pyplot as plt

vg_data = pd.read_csv('source_csv/Video_Games_Sales_as_at_22_Dec_2016.csv')# Data for all platforms 2016
ps4_data = pd.read_csv('source_csv/PS4_GamesSales.csv',encoding='unicode_escape')# 2019 PS4 Data
xbox_one_data = pd.read_csv('source_csv/XboxOne_GameSales.csv',encoding='unicode_escape')#2019 Xbox One Data
vg_data = pd.read_csv('output_csv/vg_data.csv')

######################################################################
# TODO: Ranking regions by sales figures   

na_sales = ('North America', vg_data['NA_Sales'].mean())
eu_sales = ('Europe', vg_data['EU_Sales'].mean())
jp_sales = ('Japan', vg_data['JP_Sales'].mean())
other_sales = ('Other', vg_data['Other_Sales'].mean())
global_sales = ('Global', vg_data['Global_Sales'].mean())

sales = (na_sales,eu_sales,jp_sales,other_sales,global_sales)
avg_sales = {}

for region_sales in sales:
    avg_sales.update({region_sales[0]:f"{region_sales[1]*1000000:,.3f}"})

print('Average unit sales for a videogame by region')
print(avg_sales)

######################################################################
# TODO: Produce a bar plot for average unit sales by region

data = {'region': ['North America','Europe','Japan','Other','Global'],
        'sales': [262715,146786,76427,48080,534264]}

sales_df = pd.DataFrame(data)
# print(sales_df)
sales_df.index = sales_df['region']
# sales_df = pd.read_csv('output_csv/sales_df.csv',index_col='region')

plt.figure(figsize=(10,6))
plt.title('Average Game Units Sold By Region')
sales_plot = sns.barplot(x = sales_df.index, y=sales_df['sales'],order=sales_df.sort_values('sales',ascending=False).index,palette="mako")
plt.ylabel('Average Units per Game')
plt.xlabel('Regions')
# plt.show()

######################################################################
# TODO: Top platforms by region
# Top platform is determined from unit sales

platforms = vg_data['Platform'].unique()
regions = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']

csvs = [('Europe','EU_Sales','output_csv/eu.csv','visualizations/eu_consoles.png'),
        ('Global','Global_Sales','output_csv/gl.csv','visualizations/gl_consoles.png'),
        ('Japan','JP_Sales','output_csv/jp.csv','visualizations/jp_consoles.png'),
        ('North America','NA_Sales','output_csv/na.csv','visualizations/na_consoles.png'),
        ('Other Regions','Other_Sales','output_csv/ot.csv','visualizations/ot_consoles.png')]

pdf = pd.DataFrame()

def reg_platforms(region,csv):
    platforms_rank = {}
    for platform in platforms:
        platform_data = vg_data[vg_data['Platform'] == platform]
        sum = platform_data[region].sum()
        sum = sum*1000000
        sum = sum.astype(int)
        platforms_rank.update({platform:sum})
        platforms_ranked = sorted(platforms_rank.items(), key=lambda x: x[1], reverse=True)
    pdf = pd.DataFrame(platforms_ranked,columns=['platform','units'])
    pdf.to_csv(csv)
    # print(pdf)

def region_consoles(csv,region,filepath):
    data = pd.read_csv(csv, index_col='platform')
    plt.figure(figsize=(10,6))
    plt.title('Top Consoles: ' + region)
    plot = sns.barplot(x = data.index, y=data['units'])
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha='right')
    plt.ylabel('Games per Unit')
    plt.xlabel('Platforms')
    plt.savefig(filepath)
    # plt.show()

for csv in csvs:
    reg_platforms(csv[1],csv[2])
    region_consoles(csv[2],csv[0],csv[3])




# sdf = pd.read_csv('output_csv/jp.csv')
# sdf = sdf.rename(columns = {0:'platform'})


# print(sdf)


# print(platforms_ranked)
# print(platforms_ranked[0][0])
    
# print(pdf)

# print(platforms_ranked)


# wii_data = vg_data[vg_data['Platform'] == 'Wii']
# # print(wii_data.head())
# # print(len(wii_data))
# # print(wii_data.tail())
# print(wii_data['JP_Sales'].sum())


platforms_df = pd.DataFrame()


# def region_platform(region):

# jp = pd.DataFrame()
# jp.columns = ['jp_data','jp_sales']
# jp['jp_data'] = vg_data['JP_Sales']
# jp['jp_sales'] = vg_data['Platform']
# jp.columns = [jp_data,jp_sales]
# print(jp.head())
# for platform in platforms:
#     platform_data = vg_data[vg_data['Platform' == platform]]



