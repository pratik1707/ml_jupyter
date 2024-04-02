#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
data_frame = pd.read_csv('FIFA19_official_data.csv')

data_frame.shape


# In[5]:


data_frame.describe()


# In[15]:


data_frame[data_frame['Age'] > 20].sort_values('Age').head()


# In[17]:


counts_Nationality = data_frame["Nationality"].value_counts()
counts_Nationality = counts_Nationality.reset_index()
counts_Nationality.columns= ["Nations","Counts"]
counts_Nationality.head()


# In[ ]:





# In[57]:


def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    if 'B' in x:
        return float(x.replace('B', '')) * 1000000000
    return 0.0

df1 = pd.DataFrame(data_frame, columns=['Name', 'Wage', 'Value'])
wage = df1['Wage'].str.replace('[\€,]',"", regex = True).apply(value_to_float)
value = df1['Value'].str.replace('[\€,]',"", regex = True).apply(value_to_float)
df1['Wage'] = wage
df1['Value'] = value
df1['difference'] = df1['Value'] - df1['Wage']
df1.sort_values('difference', ascending=False)


# In[69]:


import seaborn as sns
sns.set()
graph = sns.scatterplot(x = 'Wage', y = 'Value', data = df1)
graph 


# In[79]:


from bokeh.plotting import figure, show 
#from bokeh.models import HoverTool


TOOLTIPS = HoverTool(tooltips=[
    ("index", "$index"),
    ("(Wage,Value)", "(@Wage, @Value)"),
    ("Name", "@Name")]
)

p = figure(title="Soccer", x_axis_label='Wage', y_axis_label='Value', tools=[TOOLTIPS])
p.circle('Wage', 'Value', size = 10, source = df1)
show(p)


# In[ ]:





# In[ ]:





# In[ ]:




