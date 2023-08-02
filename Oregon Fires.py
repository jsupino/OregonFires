#!/usr/bin/env python
# coding: utf-8

# # <font color='#9A2403'>Oregon Fires For the Years of 2000 to 2022</font>
# 
# **Jade Supino**
# 
# ------

# Dataset obtained from **oregon.gov** Open Data Portal. Dataset name is *ODF Fire Occurrence Data 2000-2022*.
# 
# https://data.oregon.gov/Natural-Resources/ODF-Fire-Occurrence-Data-2000-2022/fbwv-q84y
# 
# Dataset contains **23.5k items** and **38 attributes**.
# 
# 
# **Objectives:**
# 1. Fire information about each area of Oregon
# 2. Find the different types of fire causes
# 3. Categorize the fires by size
# 4. Find the top 20 largest fires in Oregon between the years 2000 and 2022
# 5. Find all Oregon fires that occurred in 2022
# 6. Oregon Fires by Fire Year, Category, and Size

# **Background**
# 
# Fires are a natural occurrence throughout Oregon. Oregon has a varied geography, many different forest types, wildlife, and plant species. Forests in Oregon cover approximately 30.5 million acres, almost half the state (2). ODF stands for the Oregon Department of Forestry. To improve the sustainability of the environment, economy, and community, the ODF is responsible for protecting and managing the forests of Oregon (3).

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns  
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash


# ### <font color='#9A2403'>Data Loading</font>
# Load the CSV file and convert it to a Pandas DataFrame.

# In[2]:


file = 'ODF_Fire_Occurrence_Data_2000-2022.csv'
df = pd.read_csv(file)
df.head()


# In[3]:


df.info()


# ### <font color='#9A2403'>Data Cleaning</font>
# 1. Drop the following attributes from the DataFrame as they are not necessary for the analysis.
#     - Serial
#     - FireCategory
#     - FullFireNumber
#     - Twn
#     - Rng
#     - Sec
#     - Subdiv
#     - LandmarkLocation        
#     - RegUseZone
#     - RegUseRestriction
#     - Industrial_Restriction
#     - Ign_DateTime
#     - ReportDateTime
#     - Discover_DateTime
#     - Control_DateTime
#     - CreationDate
#     - ModifiedDate
#     - DistrictCode
#     - UnitCode
#     - DistFireNumber
# 
# 2. Rename columns Lat_DD to Latitude and Long_DD to Longitude.

# In[12]:


# drop and rename columns in the dataframe
cleaned_df = df.drop(['Serial', 'FireCategory', 'FullFireNumber', 'Twn', 'Rng', 'Sec',
                      'Subdiv', 'LandmarkLocation', 'RegUseZone', 'RegUseRestriction', 'Industrial_Restriction',
                      'Ign_DateTime', 'ReportDateTime', 'Discover_DateTime', 'Control_DateTime',
                      'CreationDate','ModifiedDate', 'DistrictCode', 'UnitCode',
                       'DistFireNumber'], axis=1)
cleaned_df = cleaned_df.rename(columns={'Lat_DD':'Latitude', 'Long_DD':'Longitude'})
cleaned_df


# *Check how many NaN values there are for the Latitude.*
# 
# There are only 10 rows of NaN values. Since there are only a small amount of NaN values, it will not impact the analysis, so we will just drop them from the DataFrame. After dropping these values, there are now 23,480 rows.

# In[13]:


# check if there are any missing values for Latitude
latitude_nan_values = cleaned_df[cleaned_df['Latitude'].isna()]
latitude_nan_values


# In[14]:


# drop null values from latitude and longitude
cleaned_df = cleaned_df.dropna(subset=['Latitude', 'Longitude'])


# Check how many NaN values there are for the EstTotalAcres.
# 
# There are 72 rows of NaN values. The NaN values are grouped by their size class and then filled in with the mean of the estimated total acres for each group.

# In[15]:


# check if there are any missing values for EstTotalAcres
EstTotalAcres_nulls = cleaned_df[cleaned_df['EstTotalAcres'].isna()]
EstTotalAcres_nulls


# In[16]:


# using the mean of each size class to fill in the estimated total acres for NaN values in each size class
cleaned_df['EstTotalAcres'] = cleaned_df.groupby('Size_class')['EstTotalAcres'].transform(lambda x: x.fillna(x.mean()))


# ---------

# ### <font color='#9A2403'>Information About Each Area of Oregon</font>
# 
# **Questions to Answer:**
# 1. How many total fires occurred in each area?
# 2. What area has experienced the greatest number of fires?
# 3. What are the total estimated total acres burned in each area?
# 4. What area has had the greatest number of acres burned?
# 
# *EOA: Eastern Oregon Area; NOA: Northern Oregon Area; SOA: Southern Oregon Area*

# **A map of the types of forest types in Oregon (2).**
# 
# ![image.png](attachment:image.png)

# In[17]:


# find how many fires occurred in each area of Oregon
areas_of_oregon= cleaned_df['Area'].value_counts().sort_values()
# convert to dataframe
areas_of_oregon = areas_of_oregon.to_frame()
# rename column
areas_of_oregon = areas_of_oregon.rename(columns = {'Area': 'Total Fires'})
areas_of_oregon


# In[18]:


# sum of the estimated total acres of fires occurred in each area of Oregon
areas_of_oregon_sums = cleaned_df.groupby('Area')['EstTotalAcres'].sum().sort_values()
# convert to dataframe
areas_of_oregon_sums = areas_of_oregon_sums.to_frame().reset_index()
# make acres only to 2 decimal places
pd.options.display.float_format = '{:.2f}'.format
areas_of_oregon_sums


# In[19]:


# plot a pie chart of the estimated total acres per area of Oregon
colors = ['#F3CBAA', '#AAB3F3', '#F3EDAA']
fig = px.pie(areas_of_oregon_sums, values='EstTotalAcres', names='Area',
             title='Estimated Total Acres Burned Per Oregon Area', color_discrete_sequence=colors, hole=0.4)
fig.show()


# **Figure 1.** The pie chart represents the three areas of Oregon. Eastern Oregon area has experienced the greatest estimated total acres burned, with approximately 52.3%.

# ### <font color='#9A2403'>Types of Fire Causes</font>
# - Causes are categorized as Human, Lightning or Under Investigation
# - Each category can be broken down into general causes
# 
# **Questions to Answer:**
# 1. What are the types of categories for fire causes?
# 2. How many fires have occurred for each category from 2000 to 2022?
# 3. What are the general causes of Oregon fires and how many fires have occurred for each?

# In[20]:


# find how many causes for human, lightning, or under investigation
cause_by = cleaned_df['HumanOrLightning'].value_counts()
# convert to dataframe
cause_by = cause_by.to_frame()
# rename column
cause_by = cause_by.rename(columns={'HumanOrLightning': 'Count'})
cause_by


# In[21]:


# find how many for general causes
general_causes = cleaned_df['GeneralCause'].value_counts()
# convert to dataframe
general_causes = general_causes.to_frame().reset_index()
# rename column
general_causes = general_causes.rename(columns={'index': 'Cause',
                                                'GeneralCause': 'Count'})
general_causes


# In[22]:


# plot a histogram using seaborn of the general causes
# set kde to True to plot the density
hist = sns.histplot(data=general_causes, x='Cause', weights='Count', kde=True, color='#9ABAE5')
hist.set_title('General Causes of Oregon Fires')
hist.set_xticks(hist.get_xticks())
hist.set_xticklabels(hist.get_xticklabels(), rotation=70)
hist.set_ylabel('Number of Fires')


# **Figure 2.** The general causes and the number of fires that have occurred are plotted in the histogram. The density curve shows the overall distribution shape. Out of the general causes, lightning has caused the most fires and railroad incidents have caused the least number of fires.
# 
# *Note:* Each cause is derived from one the categories of human, lightning, or under investigation.

# ### <font color='#9A2403'>Categorizing Fires by Class Size</font>
# 
# Sizes of class are A, B, C, D, E, F and G.
# 
# **Questions to Answer:**
# 1. What class size of fire has occurred the most over the years?
# 2. What are the differences for class A and class G fires?

# In[23]:


# find how many fires occurred in each class size
size_of_fires= cleaned_df['Size_class'].value_counts().sort_values(ascending=False)
# convert to dataframe
size_of_fires = size_of_fires.to_frame()
# rename column
size_of_fires = size_of_fires.rename(columns = {'Size_class': 'Count'})
size_of_fires


# **The greatest amount of fires occurred in class size A.**

# #### <font color='#9A2403'>Class A Fires</font>
# 
# The smallest fires that occur are labeled as class size A.

# In[24]:


# get only the size class A data
class_a = cleaned_df[cleaned_df['Size_class']=='A']
rcParams['figure.figsize'] = 8,6
# create a violin plot
ax = sns.violinplot(x="Size_class", y="EstTotalAcres", hue='HumanOrLightning', data=class_a, palette='Set3')
ax.set_title('Acres Burned for Class Size A Fires')
ax.set_xlabel('Class Size')
ax.set_ylabel('Estimated Total Acres')
ax.legend(fontsize=9)


# **Figure 3.** The three violin plots represent the estimated total acres burned for class size A categorized by cause. The causes include human, lightning, or under investigation. The distributions for human and lightning causes are similar and both contain outliers. Both contain a maximum and minimum of 0.25 and 0.00 acres burned, respectively. Most of the fires under investigation are approximately similar in terms of acres burned with a minimum of approximately 0.00 acres.
# 
# **Statistics:**
# - The median for human fires is approximately 0.01 acres with a mininimum of 0.00 acres and maximum of 0.25 acres. Q1 and Q3 are approximately 0.01 and 0.10 acres, respectively.
# - The median for lightning fires is approximately 0.10 acres with a mininimum of 0.00 acres and maximum of 0.25 acres. Q1 and Q3 are approximately 0.01 and 0.10 acres, respectively.
# - The median for fires under investigation is approximately 0.075 acres with a mininimum of 0.01 acres and maximum of 0.10 acres. Q1 and Q3 are approximately 0.075 and 0.075 acres, respectively.

# In[25]:


# use the describe method to retrieve information about class size A, including the median
class_a_median = class_a.groupby('HumanOrLightning')['EstTotalAcres'].median()
class_a_info = class_a.groupby('HumanOrLightning')['EstTotalAcres'].describe()
class_a_info = pd.concat([class_a_info, class_a_median], axis=1).rename(columns={'EstTotalAcres':'median'})
class_a_info


# #### <font color='#9A2403'>Class G Fires</font>
# 
# The largest fires that occur are labeled as class size G. 

# In[26]:


# get only the class G data
class_g = cleaned_df[cleaned_df['Size_class']=='G']
# specific colors in a color palette
my_palette = sns.color_palette(["#C4B1DB", "#9DDAF6", "#FBBCB1"])
# create a box plot
ax = sns.boxplot(x="Size_class", y="EstTotalAcres", hue='HumanOrLightning', data=class_g, palette=my_palette)
ax.set_title('Acres Burned for Class Size G Fires')
ax.set_xlabel('Class Size')
ax.set_ylabel('Estimated Total Acres')


# **Figure 4.** Class G fires are plotted in the box plot, categorized by the type of fire cause: human, lightning, or under investigation. It is shown that the human and lightning plots have outliers, whereas under investigation plot does not. For the lightning category, the estimated total acres burned is very high and has very far outliers with the greatest number of acres burned.
# 
# **Statistics:**
# - The median for human fires is approximately 16,418 acres with a mininimum of 5,521 acres and maximum of 193,566 acres. Q1 and Q3 are approximately 7,937.38 and 39,973.25 acres, respectively.
# - The median for lightning fires is approximately 23,163.4 acres with a mininimum of 5,237 acres and maximum of 499,945 acres. Q1 and Q3 are approximately 10,562 and 49,427.25 acres, respectively.
# - The median for fires under investigation is approximately 58,753 acres with a mininimum of 41,706 acres and maximum of 91,730 acres. Q1 and Q3 are approximately 50,229.5 and 75,241.5 acres, respectively.

# In[27]:


# use the describe method to retrieve information about class size G, including the median
class_g_median = class_g.groupby('HumanOrLightning')['EstTotalAcres'].median()
class_g_info = class_g.groupby('HumanOrLightning')['EstTotalAcres'].describe()
class_g_info = pd.concat([class_g_info, class_g_median], axis=1).rename(columns={'EstTotalAcres':'median'})
class_g_info


# ### <font color='#9A2403'>Top 20 Largest Fires Throughout the Years 2000-2022</font>
# 
# Find the top twenty largest fires in Oregon through all the years from 2000 to 2022.
# 
# **Questions to Answer:**
# 1. What was the largest fire over the years?
#     - What was the fire name?
#     - How many total acres was the fire?
#     - What year was the fire and what was the general cause of the fire?
# 2. What year had the greatest number of largest fires?

# In[28]:


# retrieve top 20 largest fires
top_20_fires = cleaned_df.nlargest(21, 'EstTotalAcres')


# In[29]:


# create a pivot table
top_20_fires1 = pd.pivot_table(top_20_fires,
                              values='EstTotalAcres',
                              index=['FireYear','FireName', 'DistrictName']).sort_values(by=['EstTotalAcres'],
                                                                        ascending=False)

top_20_fires1


# *Note: The top 21 fires were found because the first two top largest fires, ODF/Biscuit and Biscuit Private are considered the same fire. Biscuit Private occurred in Coos county, a part of Southwest Oregon, where ODF/Biscuit occurred. The fire was so large, units from both Grants Pass and Gold Beach responded.*

# In[30]:


# retrieve top 2 largest fires
top_fire1 = top_20_fires.loc[top_20_fires['FireName'] == 'ODF / BISCUIT']
top_fire2 = top_20_fires.loc[top_20_fires['FireName'] == 'Biscuit Private']
merged_top_2_fires = pd.concat([top_fire1, 
                               top_fire2])
# create a pivot table of the top 2 fires to compare
top_2_fires = pd.pivot_table(merged_top_2_fires,
                       values=['EstTotalAcres',
                              'Latitude', 'Longitude'],
                      index=['FireName', 'FireYear'])
top_2_fires


# **<u>Using plotly.express, a bubble map of the top 20 fires in Oregon is plotted.</u>**
# - The latitude and longitude values are used to plot their coordinates on the geographic map.
# - When hovering over the bubble, the hover name is set to the fire name and the hover data shows additional information about the fire, including the fire year and cause.
# - The projection is set to albers usa which is the projection of the map of the United States.
# - The size of the bubbles represent the estimated total acres corresponding to each fire.
# - The color of each bubble corresponds to the year that the fire occurred.
# - Update_geos is used to fit the map to the data points being plotted.
# - Update_layout is used to customize the layout, where the color of the land is set to the corresponding color chosen.

# In[31]:


# plot a bubble map of the top 20 fires
fig = px.scatter_geo(top_20_fires, lat='Latitude', lon='Longitude',
                     hover_name='FireName', 
                     hover_data=['FireYear', 'DistrictName', 'CauseBy'],
                     title='Top 20 Largest Fires Throughout 2000-2022',
                     # for map of united states
                     projection='albers usa',
                     size = 'EstTotalAcres',
                     color='FireYear')

# fit the map view based on the areas of Oregon fires
fig.update_geos(fitbounds='locations')

# change color of map
fig.update_layout(
    geo=dict(
        landcolor='#ECE2D2'
    )
)

fig.show()


# **Figure 5.** The bubble map represents the top twenty largest fires in Oregon. Each bubble represents a fire, where the fire name, estimated total acres, latitude, longitude, the year of occurrence, and the cause by is available. Fires outside of Oregon could be due to a spread of fire.
# 
# The largest Oregon fire, called Biscuit Private, occurred in 2002 with an estimated total of 499,945 acres burned. This fire occurred from lightning. A majority of the fires occurred in 2020 within the same area of Oregon. Also, a majority of the fires are from lightning. Illinois Valley Support occurred out of state which could have been caused from a fire spreading.

# ### <font color='#9A2403'>Oregon Fires Throughout 2022</font>
# 
# Find all Oregon fires that occurred in 2022.
# 
# **Questions to Answer:**
# 1. What area of Oregon has had the greatest number of fires in 2022?
# 2. What area of Oregon has had the least number of fires in 2022?

# In[32]:


# retrieve the following columns and then retrieve only those with a fire in year 2022
fire_data_2022 = cleaned_df[['Area', 'FireName', 'EstTotalAcres', 'Latitude', 'Longitude',
                        'FireYear', 'Size_class']].loc[(cleaned_df['FireYear']==2022)]
fire_data_2022


# **<u>Using plotly.express, a density map of the fires in Oregon in the year 2022.</u>**
# 
# - The latitude and longitude values are used to plot their coordinates on the geographic map.
# - The radius is set to determine the size of the circle to represent each fire.
# - The center is set to the middle of Oregon.
# - The zoom is set to the initial zoom of the map.
# - The mapbox_style styles the map. Chose stamen terrain which shows terrain and topography.
# - When hovering over the bubble, the hover name is set to the fire name and the hover data shows additional information about the fire, including the size class and estimated total acres.

# In[33]:


# plot a density map of all fires in Oregon throughout 2022
plt = px.density_mapbox(fire_data_2022, lat='Latitude', lon='Longitude', radius=7,
                        center=dict(lat=44.000000, lon=-120.500000), zoom=5,
                        mapbox_style='stamen-terrain', hover_name='FireName',
                        hover_data={'Size_class': True,
                                    'EstTotalAcres': True})
plt.show()


# **Figure 6.** A density map representing the fires in Oregon throughout 2022. The southwest region of Oregon has experienced the greatest number of fires, whereas the southeast has experienced the least.

# ### <font color='#9A2403'>Oregon Fires by Fire Year, Category, and Size</font>
# 
# Oregon fires are grouped by the fire year, class size, and by category of cause.

# In[34]:


# group the fires and sum up the estimated total acres for each fire
grouped_fires = cleaned_df.groupby(['FireName', 'FireYear', 'HumanOrLightning',
                                                'Size_class'])['EstTotalAcres'].sum().reset_index()
grouped_fires


# In[35]:


# create a jupyter dash application
app = JupyterDash(__name__)

# using dash_core_components library with outer and inner divs
app.layout = html.Div([
    html.Div([
        html.Div([
            # create a dropdown for class sizes A-G
            dcc.Dropdown(
                id='class_size',
                # options are shown in alphabetical order of class size
                options=[{'label': i, 'value': i} for i in grouped_fires.sort_values(['Size_class'])['Size_class'].unique()],
                value='Class Size'
            ),
            # create a second dropdown for selecting which category of causes
            dcc.Dropdown(
            id='cause',
            options=[{'label': i, 'value': i} for i in grouped_fires['HumanOrLightning'].unique()],
            value='Causes'
        )   
        ],
        style={'width': '48%', 'display': 'inline-block'}),
    ]),

    dcc.Graph(id='graph'),
    
    # create a slider for all the fire years
    dcc.Slider(
        id='year-slider',
        # specify the min and max years on the slider
        min=grouped_fires['FireYear'].min(),
        max=grouped_fires['FireYear'].max(),
        value=grouped_fires['FireYear'].min(),
        # select the tick marks displayed on the slider
        marks={str(FireYear): str(FireYear) for FireYear in grouped_fires['FireYear'].unique()},
        # step set to None (slider can take on any value between min and max values)
        step=None
        
    )
])

# create a decorator function with the inputs and change of values to update the graph
@app.callback(
     Output('graph', 'figure'),
    [Input('class_size', 'value'),
     Input('cause', 'value'),
     Input('year-slider', 'value')]
)

# create a function to update the data and graph based on the parameters chosen
def update_graph(class_size, cause_name, selected_year):
    selected_fires = grouped_fires[(grouped_fires['Size_class'] == class_size) &
                    (grouped_fires['HumanOrLightning'] == cause_name) &
                    (grouped_fires['FireYear'] == selected_year)]
    fires_grouped = selected_fires.groupby(['Size_class', 'FireYear',
                                            'HumanOrLightning', 'FireName'], as_index=False)['EstTotalAcres'].sum()
    
    # set the x and y values for the bar chart
    fig = go.Figure(data=[go.Bar(
        x=fires_grouped['FireName'],
        y=fires_grouped['EstTotalAcres']
    )])
   
    # update the appearance of the bar chart
    fig.update_layout(
        xaxis_title="Fire Name",
        yaxis_title="Estimated Total Acres",
        # set the title to change based on the parameters chosen
        title={
            'text': f"Oregon Class Size {class_size} Fires in {selected_year} (Cause: {cause_name})",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    
    # set the color of the bars
    fig.update_traces(marker_color='#E47136')
    
    return fig

# run the app
if __name__ == '__main__':
    app.run_server(mode="inline", port=9333, debug=True)


# **Figure 7.** A Jupyter dash application with two dropdown menus and a slider. The user selects a class size and category of cause (human, lightning or under investigation) and uses the slider to choose the specific fire year from 2000 to 2022. Based on the attributes selected, the corresponding fire names and their estimated total acres are plotted in a bar chart.

# **Conclusion**
# 
# In conclusion, fires in Oregon are a natural occurrence. Oregon consists of a varied geography with many different forest types, wildlife, and plant species. Understanding the different types of fire causes and class sizes can help understand how these fires occur and how to prevent them.
# 
# The dataset contains information about 23,490 fires between the years 2000 and 2022. Ten items contained null values for latitude and longitude coordinates. By dropping these ten fires, the analysis will not be impacted because the dataset is so large. Along with this, there were 72 fires that did not contain the estimated total acres. This is an important factor in this analysis. Therefore, these fires were each grouped by their class size then filled in with the mean of the estimated total acres for each class size.
# 
# The dataset provides fires that occurred in the North, South, and East areas of Oregon. These areas consist mainly of forests, with many different forest types. As shown in Figure 1, the estimated total acres burned for each area of Oregon is shown. The most fires occurred in the South of Oregon with a total of 12,123 fires where Mixed Conifer and Douglas-fir forests are most popular. The least in the North of Oregon with a total of 3,695 fires. The East has the greatest estimated acres burned with approximately 3,246,420.68 acres, where Ponderosa Pine and Juniper Woodlands are most popular. The North has the least number of estimated acres burned with approximately 603,821.77 acres.
# 
# Along with this, fires can be caused by many different factors. The main categories are human, lightning, and under investigation. Each category can be broken down into more general and specific causes. As shown in Figure 2, the number of fires caused be different types of causes are shown. Human fires can be caused by recreation, smoking, juveniles, etc. Approximately 17,189 fires were caused by humans. Lightning has sparked approximately 6,266 fires and can be difficult to prevent.
# 
# Each fire can be classified as A, B, C, D, E, F, or G. The smallest fires are labeled as class size A and the largest fires are labeled as class size G. The greatest number of fires over the years in Oregon were class size A. As shown in Figure 3, class A fires are shown. Human, lightning, and under investigation fires classified as class size A had a median of 0.01, 0.10, and 0.075 acres burned, respectively. Both human and lightning fires had a max of 0.25 acres burned. The largest fires, labeled class size G, is shown in Figure 4. Class G fires had a median of 16,418, 23,163, and 58,753 acres burned for human, lightning, and under investigation, respectively. Lightning caused the largest fire that burned approximately 499,945 acres. This fire is called the Biscuit Private and ODF / BISCUIT. The fire was so large that both Coos and Southwest Oregon districts were impacted. Among the top fires in Oregon, a majority occurred in the year 2020 in the Cascades, which is North Oregon. Refer to Figure 5 for the top twenty largest fires that occurred throughout Oregon.
# 
# Throughout the year 2022, there were a total of 886 fires in Oregon. As shown in Figure 6, a majority of these fires occurred in South Oregon. There were rarely any fires within the SouthEast region of Oregon. In the Jupyter Dash application, the user can find all the fires that occurred in a specific class size, year, and category of fire (human, lightning, or under investigation). The estimated total acres burned are shown for each fire.
# 
# Oregon has a variety of geography, with approximately half of the state containing forests. These forests contain different types of trees, plants, and wildlife. Since fires are a natural occurrence throughout Oregon, it is important to understand where the most fires occur, why and how these fires occur. By understanding these fires, the state can take action to mitigate or even cease future fires.

# **References**
# 1. ODF, TzA @. “ODF Fire Occurrence Data 2000-2022.” Oregon.gov, 19 Jan. 2023, https://data.oregon.gov/Natural-Resources/ODF-Fire-Occurrence-Data-2000-2022/fbwv-q84y. 
# 2. About Oregon's forests. Oregon Department of Forestry : About Oregon's forests : Forest benefits : State of Oregon. (n.d.). Retrieved April 24, 2023, from https://www.oregon.gov/odf/forestbenefits/pages/aboutforests.aspx
# 3. About Us. Oregon Department of Forestry : About us : About ODF : State of Oregon. (n.d.). Retrieved April 25, 2023, from https://www.oregon.gov/odf/aboutodf/pages/default.aspx 
