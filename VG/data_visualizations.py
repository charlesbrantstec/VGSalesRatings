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

sales_df = pd.DataFrame(avg_sales)
print(sales_df)

# print(f"{na_sales*1000000:,.3f}")
# avg_vg_sales = {"North America":na_sales,"Europe":eu_sales,"Japan":jp_sales,'Other':other_sales,"Global":global_sales}


# print(avg_vg_sales)


