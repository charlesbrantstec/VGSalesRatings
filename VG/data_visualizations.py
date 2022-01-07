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
# plt.title('Game Units Sold By Region')
# plot = sns.barplot(x = sales_df.index, y=sales_df['sales'],order=sales_df.sort_values('sales',ascending=False).index,palette="mako")
plot = sns.barplot(x=sales_df['sales'],order=sales_df.sort_values('sales',ascending=False).index, y = sales_df.index, palette="mako")
# plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha='right')
plt.xlabel('Average Units Sold per Game (in millions)')
plt.ylabel('Regions')
# plt.show()

# print(sales_df)

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

# This function creates a dataframe for each region with games units sold by platform
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

# This function creates a barplot for each region with games units sold by platform
def region_consoles(csv,region,filepath):
    data = pd.read_csv(csv, index_col='platform')
    plt.figure(figsize=(10,6))
    plt.title('Top Consoles: ' + region)
    plot = sns.barplot(x = data.index, y=data['units'],palette='mako_r')
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha='right')
    plt.ylabel('Total Game Units Sold by Platform')
    plt.xlabel('Platforms')
    plt.savefig(filepath)
    plt.show()

# for csv in csvs:
    # reg_platforms(csv[1],csv[2])
    # region_consoles(csv[2],csv[0],csv[3])

######################################################################
# TODO: Create chart to show top 10 games globally

# print(top10)
# print(global_sales)
# print(top10.index.tolist())
head = vg_data[0:9]
top_10_games = head['Name'].values.tolist()
top_10_sales = head['Global_Sales'].values.tolist()
top_sales = []

for sales in top_10_sales:
    sales = sales*1000000
    top_sales.append(sales)

top_10_df = pd.DataFrame(columns=['game','sales'])
top_10_df['game'] = top_10_games
top_10_df['sales'] = top_10_sales
print(top_10_df)

plt.figure(figsize=(10,6))
plt.title('Top 10 Games')
plot = sns.barplot(x = top_10_df['game'], y=top_10_df['sales'],palette='mako')
plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha='right')
plt.ylabel('Total Unit Sales (in millions)')
plt.xlabel('Games')
plt.show()

######################################################################
# TODO: Visualize top 10 games by region

regions = [('North America','NA_Sales'),('Europe','EU_Sales'),('Japan','JP_Sales'),('Other','Other_Sales')]

def region_df_plots(region,region_sales):
    region_df = pd.DataFrame(columns=['games','region_sales'])
    region_df['games'] = vg_data['Name']
    region_df['region_sales'] = vg_data[region_sales]
    region_df = region_df[0:9]
    # print(region_df.head())
    plt.figure(figsize=(10,6))
    plt.title('Top 5 Games: ' + region)
    plot = sns.barplot(x=region_df['games'],y=region_df['region_sales'],order=region_df.sort_values('region_sales',ascending=False)['games'],palette='mako')
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha='right')
    plt.ylabel('Total Unit Sales (in millions)')
    plt.xlabel('Games')
    plt.show()

# for region in regions:
#     region_df_plots(region[0],region[1])

######################################################################
# TODO: Visualize top 10 genres globally

top_10_genres = vg_data['Genre'].unique()
genre_list = ['Sports', 'Platform', 'Racing', 'Role-Playing', 'Puzzle', 'Misc', 'Shooter',
 'Simulation', 'Action', 'Fighting', 'Adventure', 'Strategy',
 'Action-Adventure', 'Party', 'Music', 'MMO', 'Visual Novel']
# print(top_10_genres)

genre_sales = [1365810000, 837050000, 743400000, 953629999, 243080000,
               805500000, 1100960000, 394450000, 1805690000, 456840000,
               243260000, 175680000, 32020000, 650000, 890000, 650000, 130000]

genre_rank = {'Sports':1365810000, 'Platform':837050000, 'Racing':743400000, 'Role-Playing':9536299, 'Puzzle':243080000,
              'Misc':805500000, 'Shooter':1100960000,'Simulation':394450000, 'Action':1805690000, 'Fighting':456840000,
              'Adventure':243260000, 'Strategy':175680000,'Action-Adventure':32020000, 'Party':650000, 'Music':890000,
              'MMO':650000, 'Visual Novel':130000}

for genre in genre_list:
    genre_df = vg_data[vg_data['Genre'] == genre]
    genre_sales = genre_df['Global_Sales'].sum()
    genre_sales = (genre_sales*1000000).astype(int)
    # genre_rank.append(genre_sales)
    # print(genre_sales)

genre_rank = sorted(genre_rank.items(), key=lambda x: x[1], reverse=True)
# print(genre_rank)

gdf = pd.DataFrame(columns=['genre','sales'])

for genre_sales in genre_rank:
    new_row = {'genre':genre_sales[0],'sales':genre_sales[1]}
    gdf = gdf.append(new_row,ignore_index=True)

# print(gdf)

# print(gdf[:10])

gdf = gdf[:10]
plt.figure(figsize=(10,6))
plt.title('Game Units Sold by Genre')
plot = sns.barplot(x=gdf['sales'], y = gdf['genre'],order=gdf.sort_values('sales',ascending=False)['genre'], palette="mako")
plt.xlabel('Game Units (in billions)')
plt.ylabel(None)
plt.show()

# mmo = vg_data[vg_data['Genre'] == 'MMO']
# mmo_sales = mmo['Global_Sales'].sum()
# print(mmo_sales*1000000)

# World of Warcraft: Warlords of Draenor incorrectly labeled as an action game instead of Role-Playing Game

######################################################################
# TODO: Top 10 publishers by region

rdf = pd.DataFrame(columns=['publisher','region_sales'])

publishers = vg_data['Publisher'].unique()

regions = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']
pub_dict = {}

rp = []

def region_publishers(region):
    for publisher in publishers:
        pdf = pd.DataFrame(columns=['publisher','region_sales'])
        pdf['region_sales'] = vg_data[region]
        pdf['publisher'] = vg_data['Publisher']
        pub_df = pdf[pdf['publisher'] == publisher]
        # pub_sales = pub_df['Global_Sales'].sum()
        pub_sales = pub_df['region_sales'].sum()
        pub_sales = (pub_sales*1000000).astype(int)
        pub_dict.update({publisher:pub_sales})
        pub_rank = sorted(pub_dict.items(), key=lambda x: x[1], reverse=True)
        # pub_rank = pub_rank[:10]

    # rp.append(pub_rank)
    # print(region + ': Publisher Rank')
    # print(pub_rank[:10])
    # print(pub_rank[:50])    

print(region_publishers('Global_Sales'))
# for region in regions:
#     region_publishers(region)

na_sales_pub = [('Nintendo', 816970000), ('Electronic Arts', 605730000), ('Activision', 446010000),
                ('Sony Computer Entertainment', 266170000), ('Ubisoft', 263210000), ('Take-Two Interactive', 222940000),
                ('THQ', 207720000), ('Microsoft Game Studios', 157429999), ('Atari', 109840000), ('Sega', 109270000)]
eu_sales_pub = [('Nintendo', 419010000), ('Electronic Arts', 379950000), ('Activision', 229210000),
                ('Sony Computer Entertainment', 186560000), ('Ubisoft', 171490000), ('Take-Two Interactive', 119250000),
                ('THQ', 93780000), ('Sega', 81370000), ('Konami Digital Entertainment', 69460000), ('Microsoft Game Studios', 68640000)]
jp_sales_pub = [('Nintendo', 458150000), ('Namco Bandai Games', 129149999), ('Konami Digital Entertainment', 91640000),
                ('Sony Computer Entertainment', 74150000), ('Capcom', 71200000), ('Sega', 57500000),
                ('Square Enix', 53010000), ('SquareSoft', 40130000), ('Enix Corporation', 32400000), ('Tecmo Koei', 29920000)]
other_sales_pub = [('Electronic Arts', 130979999), ('Nintendo', 94680000), ('Activision', 80270000),
                   ('Sony Computer Entertainment', 79670000), ('Take-Two Interactive', 55720000), ('Ubisoft', 52400000),
                   ('THQ', 31890000), ('Konami Digital Entertainment', 30070000), ('Sega', 24050000),
                   ('Warner Bros. Interactive Entertainment', 19960000)]
global_sales_pub = [('Nintendo', 1788810000), ('Electronic Arts', 1131430000), ('Activision', 762930000),
                    ('Sony Computer Entertainment', 606480000), ('Ubisoft', 495409999), ('Take-Two Interactive', 403820000),
                    ('THQ', 338440000), ('Konami Digital Entertainment', 283570000), ('Sega', 272420000), ('Namco Bandai Games', 263360000)]

titles = [('North America',na_sales_pub),('Europe',eu_sales_pub),('Japan',jp_sales_pub),
          ('Other Countries',other_sales_pub),('Global',global_sales_pub)]

def pub_viz(region):
    na_df = pd.DataFrame(columns=['publisher','sales'])
    for list in region[1]:
        new_row = {'publisher':list[0],'sales':list[1]}
        na_df = na_df.append(new_row,ignore_index=True)
    print(na_df)
    plt.figure(figsize=(10,6))
    plt.title(region[0]+': Top Publishers')
    plot = sns.barplot(x=na_df['publisher'],y=na_df['sales'],palette='mako_r')
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha='right')
    plt.ylabel('Total Unit Sales (in hundreds of millions)')
    plt.xlabel('Publishers')
    plt.show()

for region in titles:
    pub_viz(region)

