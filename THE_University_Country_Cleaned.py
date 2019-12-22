#!/usr/bin/env python
# coding: utf-8

# # Install packages

# In[1]:


get_ipython().run_line_magic('pip', 'install tensorflow')


# In[3]:


import tensorflow as tf


# In[4]:


get_ipython().run_line_magic('pip', 'install pillow')


# In[5]:


get_ipython().run_line_magic('pip', 'install lxml')


# In[6]:


get_ipython().run_line_magic('pip', 'install jupyter')


# In[7]:


get_ipython().run_line_magic('pip', 'install matplotlib')


# In[8]:


get_ipython().run_line_magic('pip', 'install cython')


# In[9]:


get_ipython().run_line_magic('pip', 'install sklearn')


# In[10]:


from sklearn.cluster import KMeans


# In[11]:


get_ipython().run_line_magic('pip', 'install numpy')


# In[13]:


get_ipython().run_line_magic('pip', 'install pandas')


# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane"


# # Plot distribution of USA universities CitationCounts

# In[5]:


citation = pd.read_csv('Updated_THE_Ranked_Universites_CitationCounts_2014_2018.csv')

citation.head()


# In[6]:


totalcitation=citation['Citation2014']+citation['Citation2015']+citation['Citation2016']+citation['Citation2017']+citation['Citation2018']


# In[7]:


citation['Total']=totalcitation


# In[9]:


citation.head()

citation.info()


# In[40]:


changedtype=lambda x: int(x)


# In[31]:


#citation.fillna(0)

for i in range(0,len(citation)):
    if citation.loc[i]['Citation2014'] is np.nan:
        print("yes")


# In[38]:


citation['Citation2014'].isnull()

citation=citation.fillna(0)


# # change all citationcount to int64

# In[48]:


citation['Citation2018']=citation['Citation2018'].apply(changedtype)


# In[47]:


citation['Citation2017']=citation['Citation2017'].apply(changedtype)


# In[46]:


citation['Citation2016']=citation['Citation2016'].apply(changedtype)


# In[45]:


citation['Citation2015']=citation['Citation2015'].apply(changedtype)


# In[43]:


citation['Citation2014']=citation['Citation2014'].apply(changedtype)


# In[49]:


citation.info()


# In[50]:


citation.head()


# In[66]:


new=citation.sort_values(['CountryCode','Total'], ascending=False)
new.head()


# # Filtered the universities in USA

# In[70]:


USdata=new[new['CountryCode']=='USA']


# In[71]:


USdata.head()


# # Use seaborn

# In[72]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[73]:


sns.set(color_codes=True)


# In[75]:


USpartial=USdata.loc[:][['UniversityName','Total']]


# In[80]:


USpartial.head()

USpartial2=USpartial.reset_index()

USpartial2=USpartial2.iloc[:,1:]

USpartial2.head()


# In[93]:


target=USpartial2[USpartial2['UniversityName']=='University of Rochester']

target.head()


# # Change datatype to int64

# In[95]:



target.loc[:]['Total']=target['Total'].astype(int)


# In[96]:


target.head()


# In[104]:


USpartial2.head()

USpartial2.set_index('UniversityName')

USpartial2.loc[:]['Total']=USpartial2['Total'].astype(int)


# In[113]:


USpartial2=USpartial2.set_index('UniversityName')


# In[114]:


USpartial2.head()


# In[107]:


target.head()

target.set_index('UniversityName')


# In[109]:


target=target.set_index('UniversityName')


# In[110]:


target.head()


# In[120]:


len(USpartial2)


# # The distribution of total citationcounts for USA universities

# # In total 153 universities, including U of R

# In[156]:


import pandas as pd
fig, ax = plt.subplots(figsize=(12,8))
x = pd.Series(USpartial2['Total'], name="CitationCount Total")
ax = sns.distplot(x)

ax.set_xlabel("USA UniversityCitation Total",fontsize=16)
ax.set_ylabel("Probability",fontsize=16)
plt.axvline(254555, color='red') # this is where U of R
plt.axvline(np.mean(USpartial2['Total']), color='green') # this is the mean, 175882.56
plt.axvline(np.percentile(USpartial2['Total'], 25.0), color='blue') # Q1
plt.axvline(np.percentile(USpartial2['Total'], 75.0), color='orange') # Q3 very close to the mean, which means it is highly skewed
#plt.legend()
plt.tight_layout()


# In[147]:


import matplotlib
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# In[152]:


np.percentile(USpartial2['Total'], np.array([25.0,75.0]))


# In[122]:


target


# In[136]:


np.round(np.mean(USpartial2['Total']), 2)


# # From the plot above, we can see that our citation counts is greater than the average and Q3. Also, the Q3 is very close to the average, nearly overlapping,
# # because the distribution is highly right-skewed.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[116]:


fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(USpartial2, 
             kde=False, label='Height')


# In[ ]:


# Histogram that shows the distribution for the mean of all surveys
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(np.mean(USpartial2,axis=1), 
             kde=False, label='Height')
ax.set_xlabel("",fontsize=16)
ax.set_ylabel("Frequency",fontsize=16)
plt.axvline(target, color='red')
plt.legend()
plt.tight_layout()


# # The following is data-cleaning process

# # read in school list

# In[1]:


school_list = open(r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_School_List_OK.txt")

school_name=school_list.read()


print(school_name)


# In[2]:


import pandas as pd
t = school_name

data=[]

for i in t.split("\n"):
    if i[:1].isdigit():
        data.append(" ".join(i.split(" ")[:20]))
        print(" ".join(i.split(" ")[:20]))
        
data_want = pd.DataFrame(data, columns=['Scool Name'])


data_want.to_csv("all_university_name.csv", index=False)  # all the university name


# In[3]:


# cleaned all the ranks and leadning and trailing whitespace

t = school_name

uni_name = []

for i in t.split("\n"):
    if i[:1].isdigit():
        data.append(" ".join(i.split(" ")[-5:]))
        print(" ".join(i.split(" ")[-5:]))
        uni_name.append(" ".join(i.split(" ")[-5:]))


# In[4]:


# remove trailing whitespace

import re
import string

cleaned=[]

for line in uni_name:
    line=str(line)
#    print(line.strip(' \t\n\r'))
#    print(line.rstrip(string.digits))
#    print(re.sub('^\d+[\W_]+', '', line))
    want_data = re.sub('^\d+[\W_]+', '', line)
    print(want_data.strip())
    cleaned.append(want_data.strip())


# In[7]:


# remove existing numbers

import string
import re

want_3=[]

for name in cleaned:
    print(name)
    print(re.sub('^\d+[\W_]+','',name))
    want_3.append(re.sub('^\d+[\W_]+','',name))


# In[8]:


for line in want_3:
    print(line) ## This is the data we want


# # Use APIs

# In[9]:


for line in want_3:
    url= "https://api.elsevier.com/metrics/institution/search?query=name("+line+")"
    print(url)


# In[10]:


for line in want_3[:2]:
    print(line)


# In[86]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[:4]:
#    query = "name(school)"
    url= """https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=2&limit=10&cursor=*"""
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
#    data=parsed[1]
#    print(result)
    data=result['results']
    print(data)
#    if (data[0]['country'] is not None):


# In[17]:


pwd


# In[3]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane"


# In[94]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[:10]:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    for i in data:
        if i is not None:
            university_name.append(i['name'])
            university_id.append(i['id'])
            country.append(i['country'])
            countryCode.append(i['countryCode'])
            df=pd.DataFrame({'Country':country, 'Code': countryCode, 'Name': university_name, 'id': university_id})
            print(df)


# In[97]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(0.1)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[:10]:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    for i in data:
        if i is not None:
#    if data[0] is not None:        
            countries=i['country']
            unames=i['name']
            uids=i['id']
            codes=i['countryCode'] 
            if (countries is not None):
                country.append(countries)
            else:
                country.append("")
            if (unames is not None):
                university_name.append(unames)
            else:
                university_name.append("")
            if (uids is not None):
                university_id.append(uids)
            else:
                university_id.append("")
            if (codes is not None):
                countryCode.append(codes)
            else:
                countryCode.append("")
            df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
            df.to_csv("THE_CountryCode_Result_1202.csv")
    
    


# In[98]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(0.1)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[10:20]:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    for i in data:
        if i is not None:
#    if data[0] is not None:        
            countries=i['country']
            unames=i['name']
            uids=i['id']
            codes=i['countryCode'] 
            if (countries is not None):
                country.append(countries)
            else:
                country.append("")
            if (unames is not None):
                university_name.append(unames)
            else:
                university_name.append("")
            if (uids is not None):
                university_id.append(uids)
            else:
                university_id.append("")
            if (codes is not None):
                countryCode.append(codes)
            else:
                countryCode.append("")
            df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
            df.to_csv("THE_CountryCode_Result_1202_2.csv")
    


# In[100]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(0.1)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[20:30]:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    for i in data:
        if i is not None:
#    if data[0] is not None:        
            countries=i['country']
            unames=i['name']
            uids=i['id']
            codes=i['countryCode'] 
            if (countries is not None):
                country.append(countries)
            else:
                country.append("")
            if (unames is not None):
                university_name.append(unames)
            else:
                university_name.append("")
            if (uids is not None):
                university_id.append(uids)
            else:
                university_id.append("")
            if (codes is not None):
                countryCode.append(codes)
            else:
                countryCode.append("")
            df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
            df.to_csv("THE_CountryCode_Result_1202_3.csv")
    


# In[ ]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(0.1)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[20:30]:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    for i in data:
        if i is not None:
#    if data[0] is not None:        
            countries=i['country']
            unames=i['name']
            uids=i['id']
            codes=i['countryCode'] 
            if (countries is not None):
                country.append(countries)
            else:
                country.append("")
            if (unames is not None):
                university_name.append(unames)
            else:
                university_name.append("")
            if (uids is not None):
                university_id.append(uids)
            else:
                university_id.append("")
            if (codes is not None):
                countryCode.append(codes)
            else:
                countryCode.append("")
            df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
            df.to_csv("THE_CountryCode_Result_1202_3.csv")


# In[108]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(0.1)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[30:40]:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    print(data)


# In[151]:


pwd


# In[169]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(3)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[75:]:
    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
#    try:
    parsed=json.dumps(resp.json(),
                       sort_keys=True,
                       indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
#    except ValueError:
#           pass
#            result=json.loads(parsed)
#            data=result['results']
    for i in data:
        if i is None:
            pass
        else:
#                    try:
#    if data[0] is not None:        
            countries=i['country']
            unames=i['name']
            uids=i['id']
            codes=i['countryCode'] 
            if (countries is not None):
                country.append(countries)
            else:
                country.append("")
            if (unames is not None):
                university_name.append(unames)
            else:
                university_name.append("")
            if (uids is not None):
                university_id.append(uids)
            else:
                university_id.append("")
            if (codes is not None):
                countryCode.append(codes)
            else:
                countryCode.append("")
#                    except (RuntimeError, TypeError, NameError,JSONDecodeError):
#                            pass
            df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
            df.to_csv("THE_CountryCode_Result_1202_12.csv")
#    except ValueError:
#        continue


# In[153]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(3)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[47:50]:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?query=name({})&start=0&count=25&limit=25&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    try:
        parsed=json.dumps(resp.json(),
                       sort_keys=True,
                       indent=4, separators=(',', ': '))
        result=json.loads(parsed)
        data=result['results']
        for i in data:
            if i is None:
                pass
            else:
                try:
#    if data[0] is not None:        
                    countries=i['country']
                    unames=i['name']
                    uids=i['id']
                    codes=i['countryCode'] 
                    if (countries is not None):
                        country.append(countries)
                    else:
                        country.append("")
                    if (unames is not None):
                        university_name.append(unames)
                    else:
                        university_name.append("")
                    if (uids is not None):
                        university_id.append(uids)
                    else:
                        university_id.append("")
                    if (codes is not None):
                        countryCode.append(codes)
                    else:
                        countryCode.append("")
                except (RuntimeError, TypeError, NameError,JSONDecodeError):
                        pass
                        df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
                        df.to_csv("THE_CountryCode_Result_1202_8.csv")
    except ValueError:
        continue
    


# In[1]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane"


# In[126]:


for line in want_3[38:40]:
    print(re.sub('[^A-Za-z0-9]+',' ', line))


# In[133]:


for line in want_3[38:40]:
    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
    url= """https://api.elsevier.com/metrics/institution/search?query=name("{}")&start=0&count=25&limit=25&cursor=*"""
#    resp = requests.get(url.format(line), headers={'Accept':'application/json',
#                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    print(url.format(line))


# In[129]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(3)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[38:40]:
    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
    url= """https://api.elsevier.com/metrics/institution/search?query=name("{}")&start=0&count=25&limit=25&cursor=*"""
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    print(data)


# In[135]:


pwd


# In[137]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(3)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[40:50]:
#    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
    url= """https://api.elsevier.com/metrics/institution/search?query=name("{}")&start=0&count=25&limit=25&cursor=*"""
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    for i in data:
        if i is not None:
#    if data[0] is not None:        
            countries=i['country']
            unames=i['name']
            uids=i['id']
            codes=i['countryCode'] 
            if (countries is not None):
                country.append(countries)
            else:
                country.append("")
            if (unames is not None):
                university_name.append(unames)
            else:
                university_name.append("")
            if (uids is not None):
                university_id.append(uids)
            else:
                university_id.append("")
            if (codes is not None):
                countryCode.append(codes)
            else:
                countryCode.append("")
            df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
            df.to_csv("THE_CountryCode_Result_1202_6.csv")
            
            


# In[117]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(3)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

#for line in want_3[40:50]:
#    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
#name="University of Rochester"
url= """https://api.elsevier.com/metrics/institution/search?query=name(University%20of%20Rochester)&start=0&count=25&limit=25&cursor=*"""
resp = requests.get(url, headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
result=json.loads(parsed)
data=result['results']
#print(data)
for i in data:
    if i is not None:
#    if data[0] is not None:        
        countries=i['country']
        unames=i['name']
        uids=i['id']
        codes=i['countryCode'] 
        if (countries is not None):
            country.append(countries)
        else:
            country.append("")
        if (unames is not None):
            university_name.append(unames)
        else:
            university_name.append("")
        if (uids is not None):
            university_id.append(uids)
        else:
            university_id.append("")
        if (codes is not None):
            countryCode.append(codes)
        else:
            countryCode.append("")
        df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
        df.to_csv("THE_CountryCode_Result_1202_13.csv")


# In[110]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(3)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

#for line in want_3[40:50]:
#    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
name="University of Rochester"
url= """https://api.elsevier.com/metrics/institution/search?query=name("{}")&start=0&count=25&limit=25&cursor=*"""
resp = requests.get(url.format(name), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
result=json.loads(parsed)
data=result['results']
for i in data:
    if i is not None:
#    if data[0] is not None:        
        countries=i['country']
        unames=i['name']
        uids=i['id']
        codes=i['countryCode'] 
        if (countries is not None):
            country.append(countries)
        else:
            country.append("")
        if (unames is not None):
            university_name.append(unames)
        else:
            university_name.append("")
        if (uids is not None):
            university_id.append(uids)
        else:
            university_id.append("")
        if (codes is not None):
            countryCode.append(codes)
        else:
            countryCode.append("")
        df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
        df.to_csv("THE_CountryCode_Result_1202_13.csv")
            
            


# In[111]:


pwd


# In[28]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

university_name=[]
university_id=[]
country=[]
countryCode=[]

for line in want_3:
#    query = "name(school)"
    url= "https://api.elsevier.com/metrics/institution/search?name({})&start=0&count=2&limit=10&cursor=*"
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    print(parsed)


# In[11]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane"


# In[39]:


pd.read_csv(r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202_1.csv")


# # concatenate all files

# In[22]:


link =r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202_{}.csv"

for i in range(0, 12):
    i+=1
    print(link.format(i))


# In[71]:


for i in range(0, 12):
    i+=1
    name='data{}'
    print(name.format(i))


# In[5]:


import pandas as pd


# In[118]:


filename='THE_CountryCode_Result_1202_{}.csv'
for i in range(0, 13):
    i+=1
    print(filename.format(i))


# In[4]:


chucks=[]

filename='THE_CountryCode_Result_1202_{}.csv'
for i in range(0, 13):
    i+=1
    print(filename.format(i))
#    chucks.append(filename.format(i))


# In[5]:


import pandas as pd

filename='THE_CountryCode_Result_1202_{}.csv'

chucks=[]
for i in range(0, 13):
    i+=1
    chucks.append(pd.read_csv(filename.format(i)))
    
namedata=pd.concat(chucks, ignore_index=True)

namedata.head()


# In[6]:


namedata.reset_index()

namedata2=namedata[:]

namedata2.head()

namedata2=namedata.iloc[:,1:] # delete the first column

namedata2.head()


# In[122]:


for i in range(0,len(namedata2)):
    term="University of Rochester"
    if term not in namedata2.loc[i]['University Name']:
        continue
    else:
        print("Yes")


# In[137]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(3)

university_name=[]
university_id=[]
country=[]
countryCode=[]
df=pd.DataFrame()

for line in want_3[40:50]:
#    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
    url= """https://api.elsevier.com/metrics/institution/search?query=name("{}")&start=0&count=25&limit=25&cursor=*"""
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    data=result['results']
    for i in data:
        if i is not None:
#    if data[0] is not None:        
            countries=i['country']
            unames=i['name']
            uids=i['id']
            codes=i['countryCode'] 
            if (countries is not None):
                country.append(countries)
            else:
                country.append("")
            if (unames is not None):
                university_name.append(unames)
            else:
                university_name.append("")
            if (uids is not None):
                university_id.append(uids)
            else:
                university_id.append("")
            if (codes is not None):
                countryCode.append(codes)
            else:
                countryCode.append("")
            df=pd.DataFrame({'University Name':university_name, 'University id':university_id, 'Country':country, 'Country Code':countryCode})
            df.to_csv("THE_CountryCode_Result_1202_6.csv")
            
            


# In[7]:


Uidlist=namedata2['University id']
Uidlist.head()


# In[8]:


import requests
import requests_oauthlib
import pandas as pd
import numpy as np


# In[ ]:


import time
time.sleep(2)


url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&CitationsPerPublication&CollaborationImpact&CitedPublications&FieldWeightedCitationImpact&ScholarlyOutput&PublicationsInTopJournalPercentiles&OutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'


resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
result=json.loads(parsed)




# In[9]:


len(Uidlist)


# In[125]:


import requests
import requests_oauthlib
import json
import pandas as pd
import numpy as np

import time
time.sleep(2)

country=[]
countryCode=[]
Uid=[]
uname=[]
uri=[]
metric=[]
CitationCount2014=[]
CitationCount2015=[]
CitationCount2016=[]
CitationCount2017=[]
CitationCount2018=[]

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&CitationsPerPublication&CollaborationImpact&CitedPublications&FieldWeightedCitationImpact&ScholarlyOutput&PublicationsInTopJournalPercentiles&OutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

for uid in Uidlist[1270:]:
#    print(url.format(uid))
    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
#    print(result)
    if 'results' not in result:
        pass
    else:
        if list(result['results']) is None:
            pass
        else:
#        if list(result['results'])[0] is None:
#            pass
#        else:
#        data=result['results']
            if len(list(result['results']))<1:
               pass
            else:
                if 'institution' not in list(result['results'])[0]:
                    pass
                else:
                    if 'country' in result['results'][0]['institution']:
                        country.append(result['results'][0]['institution']['country'])
                    else:
                        country.append("")
                    if 'countryCode' in result['results'][0]['institution']:
                        countryCode.append(result['results'][0]['institution']['countryCode'])
                    else:
                        countryCode.append("")
                    if 'id' in result['results'][0]['institution']:
                        Uid.append(result['results'][0]['institution']['id'])
                    else:
                        Uid.append("")
                    if 'name' in result['results'][0]['institution']:
                        uname.append(result['results'][0]['institution']['name'])
                    else:
                        uname.append("")
                    if 'uri' in result['results'][0]:
                        uri.append(result['results'][0]['institution']['uri'])
                    else:
                        uri.append("")
                if 'metrics' not in result['results'][0]:
                    pass
                else:
                    if 'metricType' not in result['results'][0]['metrics'][0]:
                        pass
                    else:
                        metric.append(result['results'][0]['metrics'][0]['metricType'])
                        if 'valueByYear' in result['results'][0]['metrics'][0]:
                            if '2014' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2014.append(result['results'][0]['metrics'][0]['valueByYear']['2014'])
                            else:
                                CitationCount2014.append("")
                            if '2015' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2015.append(result['results'][0]['metrics'][0]['valueByYear']['2015'])
                            else:
                                CitationCount2015.append("")
                            if '2016' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2016.append(result['results'][0]['metrics'][0]['valueByYear']['2016'])
                            else:
                                CitationCount2016.append("")
                            if '2017' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2017.append(result['results'][0]['metrics'][0]['valueByYear']['2017'])
                            else:
                                CitationCount2017.append("")
                            if '2018' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2018.append(result['results'][0]['metrics'][0]['valueByYear']['2018'])
                            else:
                                CitationCount2018.append("")
                        else:
                            CitationCount2014.append("")
                            CitationCount2015.append("")
                            CitationCount2016.append("")
                            CitationCount2017.append("")
                            CitationCount2018.append("")
#            else:
#                metric.append("")
 
s1=pd.Series(country, name='Country')
s2=pd.Series(countryCode, name='CountryCode')
s3=pd.Series(Uid, name='Uid')
s4=pd.Series(uname, name='UniversityName')
s5=pd.Series(uri, name='uri')
s6=pd.Series(metric, name='metric')
s7=pd.Series(CitationCount2014, name='Citation2014')
s8=pd.Series(CitationCount2015, name='Citation2015')
s9=pd.Series(CitationCount2016, name='Citation2016')
s10=pd.Series(CitationCount2017, name='Citation2017')
s11=pd.Series(CitationCount2018, name='Citation2018')
    
Times_df=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11], axis=1)
Times_df.to_csv("Times_11.csv",index=False)        
#    df=pd.DataFrame(pd.DataFrame(result['results'][0]['metrics']))
#    df.to_csv("1213_THE.csv", index=False)



# # Save data dictionary

# In[50]:


Uidlist


# In[38]:


import requests
import requests_oauthlib
import json
import pandas as pd
import numpy as np

import time
time.sleep(2)

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&CitationsPerPublication&CollaborationImpact&CitedPublications&FieldWeightedCitationImpact&ScholarlyOutput&PublicationsInTopJournalPercentiles&OutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

for uid in Uidlist[:5]:
#    print(url.format(uid))
    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
#    parsed=json.dumps(resp.json(),
#                 sort_keys=True,
#                 indent=4, separators=(',', ': '))
#    print(parsed)
    #result=json.loads(parsed)
with open("Uni_Metric_Data_Dictionary_Test4.json", 'w') as jsonfile:
    json.dump(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '), fp=jsonfile)


# In[49]:


import requests
import requests_oauthlib
import json
import pandas as pd
import numpy as np

import time
time.sleep(2)

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&CitationsPerPublication&CollaborationImpact&CitedPublications&FieldWeightedCitationImpact&ScholarlyOutput&PublicationsInTopJournalPercentiles&OutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

for uid in Uidlist[:2]:
#    print(url.format(uid))
    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
#    parsed = json.loads(resp.text)
#    print(parsed)
    result=json.loads(parsed)
    print(result)
#    with open("Uni_Metric_Data_Dictionary_2.txt", 'a') as text_file:
#         print(parsed, file=text_file)
#        json.dump(resp.json(),
#                 sort_keys=True,
#                 indent=4, separators=(',', ': '), fp=jsonfile)


# In[24]:


import requests
import requests_oauthlib
import json
import pandas as pd
import numpy as np

import time
time.sleep(2)

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&CitationsPerPublication&CollaborationImpact&CitedPublications&FieldWeightedCitationImpact&ScholarlyOutput&PublicationsInTopJournalPercentiles&OutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

for uid in Uidlist[25:]:
#    print(url.format(uid))
    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    #result=json.loads(parsed)
with open("Uni_Metric_Data_Dic_2.json", 'w') as jsonfile:
    json.dump(parsed, jsonfile)


# In[125]:


import requests
import requests_oauthlib
import json
import pandas as pd
import numpy as np

import time
time.sleep(2)

country=[]
countryCode=[]
Uid=[]
uname=[]
uri=[]
metric=[]
CitationCount2014=[]
CitationCount2015=[]
CitationCount2016=[]
CitationCount2017=[]
CitationCount2018=[]

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&CitationsPerPublication&CollaborationImpact&CitedPublications&FieldWeightedCitationImpact&ScholarlyOutput&PublicationsInTopJournalPercentiles&OutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

for uid in Uidlist[1270:]:
#    print(url.format(uid))
    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
#    print(result)
    if 'results' not in result:
        pass
    else:
        if list(result['results']) is None:
            pass
        else:
#        if list(result['results'])[0] is None:
#            pass
#        else:
#        data=result['results']
            if len(list(result['results']))<1:
               pass
            else:
                if 'institution' not in list(result['results'])[0]:
                    pass
                else:
                    if 'country' in result['results'][0]['institution']:
                        country.append(result['results'][0]['institution']['country'])
                    else:
                        country.append("")
                    if 'countryCode' in result['results'][0]['institution']:
                        countryCode.append(result['results'][0]['institution']['countryCode'])
                    else:
                        countryCode.append("")
                    if 'id' in result['results'][0]['institution']:
                        Uid.append(result['results'][0]['institution']['id'])
                    else:
                        Uid.append("")
                    if 'name' in result['results'][0]['institution']:
                        uname.append(result['results'][0]['institution']['name'])
                    else:
                        uname.append("")
                    if 'uri' in result['results'][0]:
                        uri.append(result['results'][0]['institution']['uri'])
                    else:
                        uri.append("")
                if 'metrics' not in result['results'][0]:
                    pass
                else:
                    if 'metricType' not in result['results'][0]['metrics'][0]:
                        pass
                    else:
                        metric.append(result['results'][0]['metrics'][0]['metricType'])
                        if 'valueByYear' in result['results'][0]['metrics'][0]:
                            if '2014' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2014.append(result['results'][0]['metrics'][0]['valueByYear']['2014'])
                            else:
                                CitationCount2014.append("")
                            if '2015' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2015.append(result['results'][0]['metrics'][0]['valueByYear']['2015'])
                            else:
                                CitationCount2015.append("")
                            if '2016' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2016.append(result['results'][0]['metrics'][0]['valueByYear']['2016'])
                            else:
                                CitationCount2016.append("")
                            if '2017' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2017.append(result['results'][0]['metrics'][0]['valueByYear']['2017'])
                            else:
                                CitationCount2017.append("")
                            if '2018' in result['results'][0]['metrics'][0]['valueByYear']:
                                CitationCount2018.append(result['results'][0]['metrics'][0]['valueByYear']['2018'])
                            else:
                                CitationCount2018.append("")
                        else:
                            CitationCount2014.append("")
                            CitationCount2015.append("")
                            CitationCount2016.append("")
                            CitationCount2017.append("")
                            CitationCount2018.append("")
#            else:
#                metric.append("")
 
s1=pd.Series(country, name='Country')
s2=pd.Series(countryCode, name='CountryCode')
s3=pd.Series(Uid, name='Uid')
s4=pd.Series(uname, name='UniversityName')
s5=pd.Series(uri, name='uri')
s6=pd.Series(metric, name='metric')
s7=pd.Series(CitationCount2014, name='Citation2014')
s8=pd.Series(CitationCount2015, name='Citation2015')
s9=pd.Series(CitationCount2016, name='Citation2016')
s10=pd.Series(CitationCount2017, name='Citation2017')
s11=pd.Series(CitationCount2018, name='Citation2018')
    
Times_df=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11], axis=1)
Times_df.to_csv("Times_11.csv",index=False)        
#    df=pd.DataFrame(pd.DataFrame(result['results'][0]['metrics']))
#    df.to_csv("1213_THE.csv", index=False)



# # Combine all subfiles

# In[126]:


filename='Times_{}.csv'

for i in range(1,12):
    print(filename.format(i))


# In[127]:


chuck=[]
for i in range(1,12):
    chuck.append(pd.read_csv(filename.format(i)))

total=pd.concat(chuck, ignore_index=True)    

total.head()


# In[128]:


del total['uri']


# In[129]:


total.head()

total.to_csv("THE_Ranked_University_CitationCount_2014_2018.csv", index=False)


# In[130]:


total.head()


# In[132]:


ranked=total.sort_values(by='Citation2018', ascending=False)


ranked.to_csv("THE_Ranked_Universites_CitationCounts_2014_2018.csv", index=False)


# In[135]:


ranked=ranked.drop_duplicates()
ranked.to_csv("Updated_THE_Ranked_Universites_CitationCounts_2014_2018.csv", index=False)


# In[97]:


url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&CitationsPerPublication&CollaborationImpact&CitedPublications&FieldWeightedCitationImpact&ScholarlyOutput&PublicationsInTopJournalPercentiles&OutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

for uid in Uidlist[:1]:
#    print(url.format(uid))
    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
    result=json.loads(parsed)
    print(result['results'][0])


# In[4]:


link =r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202_{}.csv"

for i in range(0, 12):
    i+=1
#    print(link.format(i))
    name = 'data{}'
    want = name.format(i)
    want = pd.read_csv(link.format(i))
want


# In[5]:


data=want.sort_values(by='Country Code')
data.head()


# In[29]:


len(data) # 1144


# In[32]:


# want to get the CitationCount for top 300 universities

want.head(300)

len(want)


# In[59]:


test_data = want[:2]
test_data = test_data['University id']

test_data

df_id = pd.DataFrame({'uid':test_data})

df_id


# In[60]:


for uid in df_id['uid']:
    print(uid)


# In[57]:


get_ipython().run_line_magic('pip', 'install requests')


# In[61]:


url='https://api.elsevier.com/analytics/scival/institutionGroup/metrics/metrics?metricTypes=CitationCount%2CCitedPublications%2CScholarlyOutput%2CPublicationsInTopJournalPercentiles%2COutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&start=0&count=25&limit=25&cursor=*'

for uid in df_id['uid']:
    print(url.format(uid))


# In[14]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

country=[]
code=[]
uid=[]
name=[]



url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount%2CCitedPublications%2CScholarlyOutput%2CPublicationsInTopJournalPercentiles%2COutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&start=0&count=25&limit=25&cursor=*'

for uid in df['uid']:
#    query = "name(school)"
#    url= "https://api.elsevier.com/metrics/institution/search?name({})&start=0&count=2&limit=25&cursor=*"

    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed = json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result = parsed['results']
    data = result[0]
    for i in data:
        if i is None:
            pass
        else:
            countries = i['country']
            codes = i['countryCode']
            uids = i['id']
            names = i['name']
            
            if countries is not None:
                country.append(countries)
            else:
                country.append("")
            if codes is not None:
                code.append(codes)
            else:
                code.append("")
            if uids is not None:
                uid.append(uids)
            else:
                uid.append("")
            if names is not None:
                name.append(names)
            else:
                name.append("")
                
                
    
    


# In[62]:


df_id


# In[167]:



url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount%2CCitedPublications%2CScholarlyOutput%2CPublicationsInTopJournalPercentiles%2COutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&start=0&count=25&limit=25&cursor=*'

for uid in df_id['uid']:
#    query = "name(school)"
#    url= "https://api.elsevier.com/metrics/institution/search?name({})&start=0&count=2&limit=25&cursor=*"

    resp = requests.get(url.format(uid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result=json.loads(parsed)
#    result=parsed[2]
    data=result['results']
#    print(data[0])
    for i in data:
#        print(i['metrics'][2]) # ScholarlyOutput
#        print(i['metrics'][0]) # CitationCount
#        print(i['metrics'][1]) # CitedPublications
#        print(i['metrics'][3]['impactType'])# impactType
        print(i['metrics'][3]) # CiteScore and PublicationsInTopJournalPercentiles
#        print(i['metrics'][3]['values'])
#        print(i['metrics'][3]['values'][0]['percentageByYear'])
#        print(i['metrics'][3]['values'][0]['valueByYear'])


# In[176]:


want.columns=['Index','UniversityName', 'Universityid','Country','CountryCode']

want


# In[178]:


inputdata=want['Universityid']

for item in inputdata:
    print(item)


# In[136]:


Uidlist


# In[51]:


Uidlist


# In[11]:


import json


# In[14]:


Uidlist


# In[22]:



url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount%2CCitedPublications%2CScholarlyOutput%2CPublicationsInTopJournalPercentiles%2COutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&start=0&count=25&limit=25&cursor=*'

#for uid in df_id['uid']:
for item in Uidlist[100:]:
#    query = "name(school)"
#    url= "https://api.elsevier.com/metrics/institution/search?name({})&start=0&count=2&limit=25&cursor=*"

    resp = requests.get(url.format(item), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "ba88a424c653ea37282b6a4cdf423a1d"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
#    result=json.loads(parsed)
with open("Data_Dic_1218_6.txt", "a") as text_file:
    print(parsed, file=text_file)
#    result=parsed[2]


# In[180]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

country=[]
countryCode=[]
universityid=[]
uniname=[]
metricType=[]
percentage2014=[]
percentage2015=[]
percentage2016=[]
percentage2017=[]
percentage2018=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]
ScholarlyOutput2014=[]
ScholarlyOutput2015=[]
ScholarlyOutput2016=[]
ScholarlyOutput2017=[]
ScholarlyOutput2018=[]
CitationCount2014=[]
CitationCount2015=[]
CitationCount2016=[]
CitationCount2017=[]
CitationCount2018=[]
CitedPublicationsValue2014=[]
CitedPublicationsValue2015=[]
CitedPublicationsValue2016=[]
CitedPublicationsValue2017=[]
CitedPublicationsValue2018=[]
CitedPublicationspercentage2014=[]
CitedPublicationspercentage2015=[]
CitedPublicationspercentage2016=[]
CitedPublicationspercentage2017=[]
CitedPublicationspercentage2018=[]
impactType=[]
CiteScorepercentage2014=[]
CiteScorepercentage2015=[]
CiteScorepercentage2016=[]
CiteScorepercentage2017=[]
CiteScorepercentage2018=[]
CiteScorevalue2014=[]
CiteScorevalue2015=[]
CiteScorevalue2016=[]
CiteScorevalue2017=[]
CiteScorevalue2018=[]
PublicationsInTopJournalPercentilespercentage2014=[]
PublicationsInTopJournalPercentilespercentage2015=[]
PublicationsInTopJournalPercentilespercentage2016=[]
PublicationsInTopJournalPercentilespercentage2017=[]
PublicationsInTopJournalPercentilespercentage2018=[]
PublicationsInTopJournalPercentilesvalue2014=[]
PublicationsInTopJournalPercentilesvalue2015=[]
PublicationsInTopJournalPercentilesvalue2016=[]
PublicationsInTopJournalPercentilesvalue2017=[]
PublicationsInTopJournalPercentilesvalue2018=[]

PublicationsInTopJournalPercentByYear2014=[]
PublicationsInTopJournalPercentByYear2015=[]
PublicationsInTopJournalPercentByYear2016=[]
PublicationsInTopJournalPercentByYear2017=[]
PublicationsInTopJournalPercentByYear2018=[]

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount%2CCitedPublications%2CScholarlyOutput%2CPublicationsInTopJournalPercentiles%2COutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&start=0&count=25&limit=25&cursor=*'

#for uid in df_id['uid']:
for item in inputdata:
#    query = "name(school)"
#    url= "https://api.elsevier.com/metrics/institution/search?name({})&start=0&count=2&limit=25&cursor=*"

    resp = requests.get(url.format(item), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result=json.loads(parsed)
#    result=parsed[2]
    data=result['results']
#    print(data[0])
    for i in data:
        if i is None:
            pass
        else:
            if i['institution'] is None:
                pass
            else:
                if i['institution']['country'] is not None:
                    country.append(i['institution']['country'])
                else:
                    country.append("")
                if i['institution']['countryCode'] is not None:
                    countryCode.append(i['institution']['countryCode'])
                else:
                    countryCode.append("")
                if i['institution']['id'] is not None:
                    universityid.append(i['institution']['id'])
                else:
                    universityid.append("")
                if i['institution']['name'] is not None:
                    uniname.append(i['institution']['name'])
                else:
                    uniname.append("")
            if i['metrics'] is None:
                pass
            else:
                if i['metrics'][0] is None:
                    pass
                else:
                    if i['metrics'][0]['metricType'] is not None:
                        metricType.append(i['metrics'][0]['metricType'])
                    else:
                        metricType.append("")
                    if i['metrics'][0]['valueByYear'] is None:
                        pass
                    else:
                        if i['metrics'][0]['valueByYear']['2014'] is not None:
                            CitationCount2014.append(i['metrics'][0]['valueByYear']['2014'])
                        else:
                            CitationCount2014.append("")
                        if i['metrics'][0]['valueByYear']['2015'] is not None:
                            CitationCount2015.append(i['metrics'][0]['valueByYear']['2015'])
                        else:
                            CitationCount2015.append("")
                        if i['metrics'][0]['valueByYear']['2016'] is not None:
                            CitationCount2016.append(i['metrics'][0]['valueByYear']['2016'])
                        else:
                            CitationCount2016.append("")
                        if i['metrics'][0]['valueByYear']['2017'] is not None:
                            CitationCount2017.append(i['metrics'][0]['valueByYear']['2017'])
                        else:
                            CitationCount2017.append("")
                        if i['metrics'][0]['valueByYear']['2018'] is not None:
                            CitationCount2018.append(i['metrics'][0]['valueByYear']['2018'])
                        else:
                            CitationCount2018.append("")
                if i['metrics'][1] is None:
                    pass
                else:
                    if i['metrics'][1]['metricType'] is not None:
                        metricType.append(i['metrics'][1]['metricType'])
                    else:
                        metricType.append("")
                    if i['metrics'][1]['percentageByYear'] is None:
                        pass
                    else:                    
                        if i['metrics'][1]['percentageByYear']['2014'] is not None:
                            CitedPublicationspercentage2014.append(i['metrics'][1]['percentageByYear']['2014'])
                        else:
                            CitedPublicationspercentage2014.append("")
                        if i['metrics'][1]['percentageByYear']['2015'] is not None:
                            CitedPublicationspercentage2015.append(i['metrics'][1]['percentageByYear']['2015'])
                        else:
                            CitedPublicationspercentage2015.append("")
                        if i['metrics'][1]['percentageByYear']['2016'] is not None:
                            CitedPublicationspercentage2016.append(i['metrics'][1]['percentageByYear']['2016'])
                        else:
                            CitedPublicationspercentage2016.append("")
                        if i['metrics'][1]['percentageByYear']['2017'] is not None:
                            CitedPublicationspercentage2017.append(i['metrics'][1]['percentageByYear']['2017'])
                        else:
                            CitedPublicationspercentage2017.append("")
                        if i['metrics'][1]['percentageByYear']['2018'] is not None:
                            CitedPublicationspercentage2018.append(i['metrics'][1]['percentageByYear']['2018'])
                        else:
                            CitedPublicationspercentage2018.append("")
                    if i['metrics'][1]['valueByYear'] is None:
                        pass
                    else:
                        if i['metrics'][1]['valueByYear']['2014'] is not None:
                            CitedPublicationsValue2014.append(i['metrics'][1]['valueByYear']['2014'])
                        else:
                            CitedPublicationsValue2014.append("")
                        if i['metrics'][1]['valueByYear']['2015'] is not None:
                            CitedPublicationsValue2015.append(i['metrics'][1]['valueByYear']['2015'])
                        else:
                            CitedPublicationsValue2015.append("")
                        if i['metrics'][1]['valueByYear']['2016'] is not None:
                            CitedPublicationsValue2016.append(i['metrics'][1]['valueByYear']['2016'])
                        else:
                            CitedPublicationsValue2016.append("")
                        if i['metrics'][1]['valueByYear']['2017'] is not None:
                            CitedPublicationsValue2017.append(i['metrics'][1]['valueByYear']['2017'])
                        else:
                            CitedPublicationsValue2017.append("")
                        if i['metrics'][1]['valueByYear']['2018'] is not None:
                            CitedPublicationsValue2018.append(i['metrics'][1]['valueByYear']['2018'])
                        else:
                            CitedPublicationsValue2018.append("")
                if i['metrics'][2] is None:
                    pass
                else:
                    if i['metrics'][2]['metricType'] is not None:
                        metricType.append(i['metrics'][2]['metricType'])
#                        ScholarlyOutput2014.append(i['metrics'][2]['valueByYear']['2014'])
                    else:
                        metricType.append("")
                    if i['metrics'][2]['valueByYear'] is None:
                        pass
                    else:
                        if i['metrics'][2]['valueByYear']['2014'] is not None:
                            ScholarlyOutput2014.append(i['metrics'][2]['valueByYear']['2014'])
                        else:
                            ScholarlyOutput2014.append("")
                        if i['metrics'][2]['valueByYear']['2015'] is not None:
                            ScholarlyOutput2015.append(i['metrics'][2]['valueByYear']['2015'])
                        else:
                            ScholarlyOutput2015.append("")
                        if i['metrics'][2]['valueByYear']['2016'] is not None:
                            ScholarlyOutput2016.append(i['metrics'][2]['valueByYear']['2016'])
                        else:
                            ScholarlyOutput2016.append("")
                        if i['metrics'][2]['valueByYear']['2017'] is not None:
                            ScholarlyOutput2017.append(i['metrics'][2]['valueByYear']['2017'])
                        else:
                            ScholarlyOutput2017.append("")
                        if i['metrics'][2]['valueByYear']['2018'] is not None:
                            ScholarlyOutput2018.append(i['metrics'][2]['valueByYear']['2018'])
                        else:
                            ScholarlyOutput2018.append("")
                if i['metrics'][3] is None:
                    pass
                else:
                    if i['metrics'][3]['impactType'] is not None:
                        impactType.append(i['metrics'][3]['impactType'])
                    else:
                        impactType.append("")
                    if i['metrics'][3]['metricType'] is not None:
                        metricType.append(i['metrics'][3]['metricType'])
                    else:
                        metricType.append("")
                    if i['metrics'][3]['values'] is None:
                        pass
                    else:
                        if i['metrics'][3]['values'][0]['percentageByYear'] is None:
                            pass
                        else:
                            if i['metrics'][3]['values'][0]['percentageByYear']['2014'] is not None:
                                CiteScorepercentage2014.append(i['metrics'][3]['values'][0]['percentageByYear']['2014'])
                            else:
                                CiteScorepercentage2014.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2015'] is not None:
                                CiteScorepercentage2015.append(i['metrics'][3]['values'][0]['percentageByYear']['2015'])
                            else:
                                CiteScorepercentage2015.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2016'] is not None:
                                CiteScorepercentage2016.append(i['metrics'][3]['values'][0]['percentageByYear']['2016'])
                            else:
                                CiteScorepercentage2016.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2017'] is not None:
                                CiteScorepercentage2017.append(i['metrics'][3]['values'][0]['percentageByYear']['2017'])
                            else:
                                CiteScorepercentage2017.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2018'] is not None:
                                CiteScorepercentage2018.append(i['metrics'][3]['values'][0]['percentageByYear']['2018'])
                            else:
                                CiteScorepercentage2018.append("")
                        if i['metrics'][3]['values'][0]['percentageByYear'] is None:
                            pass
                        else:
                            if i['metrics'][3]['values'][0]['valueByYear']['2014'] is not None:
                                CiteScorevalue2014.append(i['metrics'][3]['values'][0]['valueByYear']['2014'])
                            else:
                                CiteScorevalue2014.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2015'] is not None:
                                CiteScorevalue2015.append(i['metrics'][3]['values'][0]['valueByYear']['2015'])
                            else:
                                CiteScorevalue2015.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2016'] is not None:
                                CiteScorevalue2016.append(i['metrics'][3]['values'][0]['valueByYear']['2016'])
                            else:
                                CiteScorevalue2016.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2017'] is not None:
                                CiteScorevalue2017.append(i['metrics'][3]['values'][0]['valueByYear']['2017'])
                            else:
                                CiteScorevalue2017.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2018'] is not None:
                                CiteScorevalue2018.append(i['metrics'][3]['values'][0]['valueByYear']['2018'])
                            else:
                                CiteScorevalue2018.append("")
                        
testfile= pd.DataFrame({'country': country, 'countryCode': countryCode, 'universityid':universityid,
                       'uniname':uniname, 'CitationCount2014':CitationCount2014,
                       'CitationCount2015':CitationCount2015, 'CitationCount2016':CitationCount2016,
                       'CitationCount2017':CitationCount2017, 'CitationCount2018':CitationCount2018,
                        'CitedPublicationspercentage2014':CitedPublicationspercentage2014,
                        'CitedPublicationspercentage2015':CitedPublicationspercentage2015,
                        'CitedPublicationspercentage2016':CitedPublicationspercentage2016,
                        'CitedPublicationspercentage2017':CitedPublicationspercentage2017,
                        'CitedPublicationspercentage2018':CitedPublicationspercentage2018,
                        'CitedPublicationsValue2014':CitedPublicationsValue2014,
                        'CitedPublicationsValue2015':CitedPublicationsValue2015,
                        'CitedPublicationsValue2016':CitedPublicationsValue2016,
                        'CitedPublicationsValue2017':CitedPublicationsValue2017,
                        'CitedPublicationsValue2018':CitedPublicationsValue2018,
                        'ScholarlyOutput2014':ScholarlyOutput2014, 'ScholarlyOutput2015':ScholarlyOutput2015,
                        'ScholarlyOutput2016': ScholarlyOutput2016, 'ScholarlyOutput2017':ScholarlyOutput2017,
                        'ScholarlyOutput2018':ScholarlyOutput2018,
                        'CiteScorepercentage2014':CiteScorepercentage2014,
                        'CiteScorepercentage2015':CiteScorepercentage2015,
                        'CiteScorepercentage2016':CiteScorepercentage2016,
                        'CiteScorepercentage2017':CiteScorepercentage2017,
                        'CiteScorepercentage2018':CiteScorepercentage2018,
                        'CiteScorevalue2014':CiteScorevalue2014,
                        'CiteScorevalue2015':CiteScorevalue2015,
                        'CiteScorevalue2016':CiteScorevalue2016,
                        'CiteScorevalue2017':CiteScorevalue2017,
                        'CiteScorevalue2018':CiteScorevalue2018})

testfile.to_csv("testfile_01.csv", index=False)

#    data_dict = data[0]['institution']
#    data_dict_2 = data[0]['institution']
#    df_file_2=pd.DataFrame(data_dict_2)
#    df_file_2.to_csv("File_3.csv", index=False)
#    data_df=pd.DataFrame(data=data_dict.value())
#    data_df.to_csv("File.csv", index=False)
#    print(data[0]['institution']['name'])
#    print(data[0]) # get 'MetricsType'
#    inst=data[0]['institution']
#    metrics=data[0]['metrics']
#    df_test = pd.DataFrame({'institution':inst, 'metrics':metrics})
#    df_test.to_csv("Test_Inst.csv", index=False)
#    df=pd.DataFrame(data[0]['metrics'][0])
#    df.to_csv("Test_MetricsType.csv", index=False)
#    metrics=result[1]['metrics']
    
#    print(data)
#    print(data)
#    df=pd.DataFrame(parsed)
#    df.to_csv("Test_DataFrame.csv", index=False)


# In[141]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

country=[]
countryCode=[]
universityid=[]
uniname=[]
metricType=[]
percentage2014=[]
percentage2015=[]
percentage2016=[]
percentage2017=[]
percentage2018=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]
ScholarlyOutput2014=[]
ScholarlyOutput2015=[]
ScholarlyOutput2016=[]
ScholarlyOutput2017=[]
ScholarlyOutput2018=[]
CitationCount2014=[]
CitationCount2015=[]
CitationCount2016=[]
CitationCount2017=[]
CitationCount2018=[]
CitedPublicationsValue2014=[]
CitedPublicationsValue2015=[]
CitedPublicationsValue2016=[]
CitedPublicationsValue2017=[]
CitedPublicationsValue2018=[]
CitedPublicationspercentage2014=[]
CitedPublicationspercentage2015=[]
CitedPublicationspercentage2016=[]
CitedPublicationspercentage2017=[]
CitedPublicationspercentage2018=[]
impactType=[]
CiteScorepercentage2014=[]
CiteScorepercentage2015=[]
CiteScorepercentage2016=[]
CiteScorepercentage2017=[]
CiteScorepercentage2018=[]
CiteScorevalue2014=[]
CiteScorevalue2015=[]
CiteScorevalue2016=[]
CiteScorevalue2017=[]
CiteScorevalue2018=[]
PublicationsInTopJournalPercentilespercentage2014=[]
PublicationsInTopJournalPercentilespercentage2015=[]
PublicationsInTopJournalPercentilespercentage2016=[]
PublicationsInTopJournalPercentilespercentage2017=[]
PublicationsInTopJournalPercentilespercentage2018=[]
PublicationsInTopJournalPercentilesvalue2014=[]
PublicationsInTopJournalPercentilesvalue2015=[]
PublicationsInTopJournalPercentilesvalue2016=[]
PublicationsInTopJournalPercentilesvalue2017=[]
PublicationsInTopJournalPercentilesvalue2018=[]

PublicationsInTopJournalPercentByYear2014=[]
PublicationsInTopJournalPercentByYear2015=[]
PublicationsInTopJournalPercentByYear2016=[]
PublicationsInTopJournalPercentByYear2017=[]
PublicationsInTopJournalPercentByYear2018=[]

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount%2CCitedPublications%2CScholarlyOutput%2CPublicationsInTopJournalPercentiles%2COutputsInTopCitationPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&start=0&count=25&limit=25&cursor=*'

#for uid in df_id['uid']:
for item in Uidlist[100:]:
#    query = "name(school)"
#    url= "https://api.elsevier.com/metrics/institution/search?name({})&start=0&count=2&limit=25&cursor=*"

    resp = requests.get(url.format(item), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result=json.loads(parsed)
#    result=parsed[2]
    data=result['results']
#    print(data[0])
    for i in data:
        if i is None:
            pass
        else:
            if i['institution'] is None:
                pass
            else:
                if i['institution']['country'] is not None:
                    country.append(i['institution']['country'])
                else:
                    country.append("")
                if i['institution']['countryCode'] is not None:
                    countryCode.append(i['institution']['countryCode'])
                else:
                    countryCode.append("")
                if i['institution']['id'] is not None:
                    universityid.append(i['institution']['id'])
                else:
                    universityid.append("")
                if i['institution']['name'] is not None:
                    uniname.append(i['institution']['name'])
                else:
                    uniname.append("")
            if i['metrics'] is None:
                pass
            else:
                if i['metrics'][0] is None:
                    pass
                else:
                    if i['metrics'][0]['metricType'] is not None:
                        metricType.append(i['metrics'][0]['metricType'])
                    else:
                        metricType.append("")
                    if i['metrics'][0]['valueByYear'] is None:
                        pass
                    else:
                        if i['metrics'][0]['valueByYear']['2014'] is not None:
                            CitationCount2014.append(i['metrics'][0]['valueByYear']['2014'])
                        else:
                            CitationCount2014.append("")
                        if i['metrics'][0]['valueByYear']['2015'] is not None:
                            CitationCount2015.append(i['metrics'][0]['valueByYear']['2015'])
                        else:
                            CitationCount2015.append("")
                        if i['metrics'][0]['valueByYear']['2016'] is not None:
                            CitationCount2016.append(i['metrics'][0]['valueByYear']['2016'])
                        else:
                            CitationCount2016.append("")
                        if i['metrics'][0]['valueByYear']['2017'] is not None:
                            CitationCount2017.append(i['metrics'][0]['valueByYear']['2017'])
                        else:
                            CitationCount2017.append("")
                        if i['metrics'][0]['valueByYear']['2018'] is not None:
                            CitationCount2018.append(i['metrics'][0]['valueByYear']['2018'])
                        else:
                            CitationCount2018.append("")
                if i['metrics'][1] is None:
                    pass
                else:
                    if i['metrics'][1]['metricType'] is not None:
                        metricType.append(i['metrics'][1]['metricType'])
                    else:
                        metricType.append("")
                    if i['metrics'][1]['percentageByYear'] is None:
                        pass
                    else:                    
                        if i['metrics'][1]['percentageByYear']['2014'] is not None:
                            CitedPublicationspercentage2014.append(i['metrics'][1]['percentageByYear']['2014'])
                        else:
                            CitedPublicationspercentage2014.append("")
                        if i['metrics'][1]['percentageByYear']['2015'] is not None:
                            CitedPublicationspercentage2015.append(i['metrics'][1]['percentageByYear']['2015'])
                        else:
                            CitedPublicationspercentage2015.append("")
                        if i['metrics'][1]['percentageByYear']['2016'] is not None:
                            CitedPublicationspercentage2016.append(i['metrics'][1]['percentageByYear']['2016'])
                        else:
                            CitedPublicationspercentage2016.append("")
                        if i['metrics'][1]['percentageByYear']['2017'] is not None:
                            CitedPublicationspercentage2017.append(i['metrics'][1]['percentageByYear']['2017'])
                        else:
                            CitedPublicationspercentage2017.append("")
                        if i['metrics'][1]['percentageByYear']['2018'] is not None:
                            CitedPublicationspercentage2018.append(i['metrics'][1]['percentageByYear']['2018'])
                        else:
                            CitedPublicationspercentage2018.append("")
                    if i['metrics'][1]['valueByYear'] is None:
                        pass
                    else:
                        if i['metrics'][1]['valueByYear']['2014'] is not None:
                            CitedPublicationsValue2014.append(i['metrics'][1]['valueByYear']['2014'])
                        else:
                            CitedPublicationsValue2014.append("")
                        if i['metrics'][1]['valueByYear']['2015'] is not None:
                            CitedPublicationsValue2015.append(i['metrics'][1]['valueByYear']['2015'])
                        else:
                            CitedPublicationsValue2015.append("")
                        if i['metrics'][1]['valueByYear']['2016'] is not None:
                            CitedPublicationsValue2016.append(i['metrics'][1]['valueByYear']['2016'])
                        else:
                            CitedPublicationsValue2016.append("")
                        if i['metrics'][1]['valueByYear']['2017'] is not None:
                            CitedPublicationsValue2017.append(i['metrics'][1]['valueByYear']['2017'])
                        else:
                            CitedPublicationsValue2017.append("")
                        if i['metrics'][1]['valueByYear']['2018'] is not None:
                            CitedPublicationsValue2018.append(i['metrics'][1]['valueByYear']['2018'])
                        else:
                            CitedPublicationsValue2018.append("")
                if i['metrics'][2] is None:
                    pass
                else:
                    if i['metrics'][2]['metricType'] is not None:
                        metricType.append(i['metrics'][2]['metricType'])
#                        ScholarlyOutput2014.append(i['metrics'][2]['valueByYear']['2014'])
                    else:
                        metricType.append("")
                    if i['metrics'][2]['valueByYear'] is None:
                        pass
                    else:
                        if i['metrics'][2]['valueByYear']['2014'] is not None:
                            ScholarlyOutput2014.append(i['metrics'][2]['valueByYear']['2014'])
                        else:
                            ScholarlyOutput2014.append("")
                        if i['metrics'][2]['valueByYear']['2015'] is not None:
                            ScholarlyOutput2015.append(i['metrics'][2]['valueByYear']['2015'])
                        else:
                            ScholarlyOutput2015.append("")
                        if i['metrics'][2]['valueByYear']['2016'] is not None:
                            ScholarlyOutput2016.append(i['metrics'][2]['valueByYear']['2016'])
                        else:
                            ScholarlyOutput2016.append("")
                        if i['metrics'][2]['valueByYear']['2017'] is not None:
                            ScholarlyOutput2017.append(i['metrics'][2]['valueByYear']['2017'])
                        else:
                            ScholarlyOutput2017.append("")
                        if i['metrics'][2]['valueByYear']['2018'] is not None:
                            ScholarlyOutput2018.append(i['metrics'][2]['valueByYear']['2018'])
                        else:
                            ScholarlyOutput2018.append("")
                if i['metrics'][3] is None:
                    pass
                else:
                    if i['metrics'][3]['impactType'] is not None:
                        impactType.append(i['metrics'][3]['impactType'])
                    else:
                        impactType.append("")
                    if i['metrics'][3]['metricType'] is not None:
                        metricType.append(i['metrics'][3]['metricType'])
                    else:
                        metricType.append("")
                    if i['metrics'][3]['values'] is None:
                        pass
                    else:
                        if i['metrics'][3]['values'][0]['percentageByYear'] is None:
                            pass
                        else:
                            if i['metrics'][3]['values'][0]['percentageByYear']['2014'] is not None:
                                CiteScorepercentage2014.append(i['metrics'][3]['values'][0]['percentageByYear']['2014'])
                            else:
                                CiteScorepercentage2014.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2015'] is not None:
                                CiteScorepercentage2015.append(i['metrics'][3]['values'][0]['percentageByYear']['2015'])
                            else:
                                CiteScorepercentage2015.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2016'] is not None:
                                CiteScorepercentage2016.append(i['metrics'][3]['values'][0]['percentageByYear']['2016'])
                            else:
                                CiteScorepercentage2016.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2017'] is not None:
                                CiteScorepercentage2017.append(i['metrics'][3]['values'][0]['percentageByYear']['2017'])
                            else:
                                CiteScorepercentage2017.append("")
                            if i['metrics'][3]['values'][0]['percentageByYear']['2018'] is not None:
                                CiteScorepercentage2018.append(i['metrics'][3]['values'][0]['percentageByYear']['2018'])
                            else:
                                CiteScorepercentage2018.append("")
                        if i['metrics'][3]['values'][0]['percentageByYear'] is None:
                            pass
                        else:
                            if i['metrics'][3]['values'][0]['valueByYear']['2014'] is not None:
                                CiteScorevalue2014.append(i['metrics'][3]['values'][0]['valueByYear']['2014'])
                            else:
                                CiteScorevalue2014.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2015'] is not None:
                                CiteScorevalue2015.append(i['metrics'][3]['values'][0]['valueByYear']['2015'])
                            else:
                                CiteScorevalue2015.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2016'] is not None:
                                CiteScorevalue2016.append(i['metrics'][3]['values'][0]['valueByYear']['2016'])
                            else:
                                CiteScorevalue2016.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2017'] is not None:
                                CiteScorevalue2017.append(i['metrics'][3]['values'][0]['valueByYear']['2017'])
                            else:
                                CiteScorevalue2017.append("")
                            if i['metrics'][3]['values'][0]['valueByYear']['2018'] is not None:
                                CiteScorevalue2018.append(i['metrics'][3]['values'][0]['valueByYear']['2018'])
                            else:
                                CiteScorevalue2018.append("")
                        
testfile= pd.DataFrame({'country': country, 'countryCode': countryCode, 'universityid':universityid,
                       'uniname':uniname, 'CitationCount2014':CitationCount2014,
                       'CitationCount2015':CitationCount2015, 'CitationCount2016':CitationCount2016,
                       'CitationCount2017':CitationCount2017, 'CitationCount2018':CitationCount2018,
                        'CitedPublicationspercentage2014':CitedPublicationspercentage2014,
                        'CitedPublicationspercentage2015':CitedPublicationspercentage2015,
                        'CitedPublicationspercentage2016':CitedPublicationspercentage2016,
                        'CitedPublicationspercentage2017':CitedPublicationspercentage2017,
                        'CitedPublicationspercentage2018':CitedPublicationspercentage2018,
                        'CitedPublicationsValue2014':CitedPublicationsValue2014,
                        'CitedPublicationsValue2015':CitedPublicationsValue2015,
                        'CitedPublicationsValue2016':CitedPublicationsValue2016,
                        'CitedPublicationsValue2017':CitedPublicationsValue2017,
                        'CitedPublicationsValue2018':CitedPublicationsValue2018,
                        'ScholarlyOutput2014':ScholarlyOutput2014, 'ScholarlyOutput2015':ScholarlyOutput2015,
                        'ScholarlyOutput2016': ScholarlyOutput2016, 'ScholarlyOutput2017':ScholarlyOutput2017,
                        'ScholarlyOutput2018':ScholarlyOutput2018,
                        'CiteScorepercentage2014':CiteScorepercentage2014,
                        'CiteScorepercentage2015':CiteScorepercentage2015,
                        'CiteScorepercentage2016':CiteScorepercentage2016,
                        'CiteScorepercentage2017':CiteScorepercentage2017,
                        'CiteScorepercentage2018':CiteScorepercentage2018,
                        'CiteScorevalue2014':CiteScorevalue2014,
                        'CiteScorevalue2015':CiteScorevalue2015,
                        'CiteScorevalue2016':CiteScorevalue2016,
                        'CiteScorevalue2017':CiteScorevalue2017,
                        'CiteScorevalue2018':CiteScorevalue2018})

testfile.to_csv("1213_THE_4.csv", index=False)

#    data_dict = data[0]['institution']
#    data_dict_2 = data[0]['institution']
#    df_file_2=pd.DataFrame(data_dict_2)
#    df_file_2.to_csv("File_3.csv", index=False)
#    data_df=pd.DataFrame(data=data_dict.value())
#    data_df.to_csv("File.csv", index=False)
#    print(data[0]['institution']['name'])
#    print(data[0]) # get 'MetricsType'
#    inst=data[0]['institution']
#    metrics=data[0]['metrics']
#    df_test = pd.DataFrame({'institution':inst, 'metrics':metrics})
#    df_test.to_csv("Test_Inst.csv", index=False)
#    df=pd.DataFrame(data[0]['metrics'][0])
#    df.to_csv("Test_MetricsType.csv", index=False)
#    metrics=result[1]['metrics']
    
#    print(data)
#    print(data)
#    df=pd.DataFrame(parsed)
#    df.to_csv("Test_DataFrame.csv", index=False)


# In[23]:


pwd


# In[143]:


filename='1213_THE_{}.csv'

for i in range(1,5):
    print(filename.format(i))


# In[185]:


chuck=[]


filename='1213_THE_{}.csv'

for i in range(1,5):
    chuck.append(pd.read_csv(filename.format(i)))

total_df2=pd.concat(chuck, ignore_index=True)

total_df2.head()

total_df2.to_csv("Updated_Uni_Metrics.csv", index=False)


# In[170]:


chuck=[]


filename='1213_THE_{}.csv'

for i in range(1,5):
    chuck.append(pd.read_csv(filename.format(i)))

total_df=pd.concat(chuck, axis=1)

total_df.head()


changedtype=lambda x: int(x[:])


# In[181]:


total_df.universityid.fillna(0)


# In[182]:


total_df.head()


# In[183]:


total_df.to_csv("Updated_THE_Uni_Metrics.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


from sklearn.cluster import KMeans


# In[28]:


kmeans = KMeans(n_clusters=10, random_state=0).fit(data)


# In[90]:


len(data)


# In[ ]:


## combine all CountryCode results

uni_1 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202.csv")
uni_2 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202_2.csv")
uni_3 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202_3.csv")
uni_4 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202_4.csv")




# In[11]:


import requests
url = "https://api.elsevier.com/metrics/institution/scopus_id/60027165?apiKey=dcfb521197bf15867d12c3c86c46c69b"
#url = "https://api.elsevier.com/content/abstract/scopus_id/60027165?apiKey=2bbd32fdfec9b9151f339032a08ebb48"
response = requests.get(url)
print(response.headers)


# In[36]:


## read in all the spreadsheets

import pandas as pd

First_5 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\THE_Uni_First5.csv", delimiter=",")
print(type(First_5))

Start_6 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\THE_Uni_6.csv", delimiter=",")

Start_11 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\THE_Uni_11.csv", delimiter=",")

Start_311 = pd.read_csv(r"C:\Users\jchen148\THE Rankings\THE_Uni_311.csv", delimiter=",")



combined_df =pd.concat([First_5,Start_6,Start_11,Start_311])



# In[38]:


print(First_5)


# In[25]:


cd "C:\Users\jchen148\THE Rankings\Json files"


# In[26]:


# load json file
import json
import pandas as pd

# Read JSON file
with open('THE_University_Country.json') as data_file:
    data_loaded = json.load(data_file)
    
print(data_loaded)


# In[27]:


import json

with open("Test_THE_Country", 'w') as fd:
    fd.write(json.dumps(data_loaded, sort_keys=True, indent=4, separators=(',', ': ')))    # both dicts in a list here


# In[29]:


with open("Test_THE_Country", 'r') as fd:
    University_data=json.load(fd)


# In[52]:


for key in University_data:
#    print(University_data[key])
    sub = University_data[key]
    total="".join(University_data[key].split("\b\s\n"))
    print(total)


# In[129]:


print(len(data_loaded)) # 255


# In[53]:


with open("Test_THE_Country", 'rb') as f:
    print(json.load(f))


# In[2]:


#print(data_loaded['9'])

#Data=data_loaded['4']

#Data_new=Data.split(',')

#print(Data_new[1:5])

u_id=[]
country=[]

for key in data_loaded:
    Data=data_loaded[key]
    Data_new=Data.split(',')
    Data_want=Data_new[1:5]
#    with open('test.json', 'w') as f:
#         json.dump(Data_want, f)
#    print(Data_want)
    Data_want_str=str(Data_want)
#    print(Data_want_str)
#    print(Data_want_str.split(", "))
    Data_OK=Data_want_str.split(", ")
    Data_OK_str=str(Data_OK)
#    print(Data_OK_str.rstrip())
    Data_Good=Data_OK_str.rstrip()
#    print(Data_OK)
    Data_Good_str=str(Data_Good)
    Data_good=Data_Good_str.rstrip()
#    print(Data_good)
#    print(Data_good.split("\n"))
    Data_Q = str(Data_good.split("\n"))
    Data_Fix = Data_Q.split(",")[0] 
#    print(Data_Fix)
    Data_OK = str(Data_Fix)
#    print(Data_OK.split(":")[0])
    Whole=Data_OK.split(":")
#    print(Whole)
    Whole_str=str(Whole)
    if len(Whole_str.split(","))>1:
#        print(Whole_str.split(",")[1])
        Old=Whole_str.split(",")[1]
#        print(re.sub("[^A-Za-z0-9]+", "",Old))
        NEW=re.sub("[^A-Za-z0-9]+", "",Old)
        print(NEW)
        if NEW.isdigit() is True:
            u_id.append(NEW)
        if NEW.isupper() is True:
            country.append(NEW)
print("university id is ", u_id)
print("university country is ",country)
            
#        for line in Old:
#            print("".join(re.sub("[^A-Za-z0-9]+", "", line)))
#            NEW="".join(re.sub("[^A-Za-z0-9]+", "", line))
            
#    for i in range(0,len(Data_OK)):
#    print(Data_OK)  
#    for name in Data_OK:
#        print(name)
#print(type(Data_OK))

#for i in range(0,len(Data_OK_str)):
    


# In[13]:


# University SciVal institution id
print(u_id)


# In[14]:


# countryCode

print(country)


# In[1]:


# get country SciVal code, and then use for metrics search 

import requests
import json
import time

time.sleep(1)

for item in country:
    url="https://api.elsevier.com/analytics/scival/country/code/{}start=0&count=25&limit=25"
    resp = requests.get(url.format(item),
                    headers={'Accept':'application/json',
                            'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    print(json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))) 
    with open("THE_Country_Code_1.json", "w") as jsonfile:
        json.load(resp.json(), jsonfile)


# In[17]:


# want to get all SciVal institution ids
import requests
import json
import time

time.sleep(2)

url="https://api.elsevier.com/analytics/scival/institution/classificationType=THE/PUBYEAR%20AFT2013%20BFT2017start=1&count=25"

resp = requests.get(url,
                    headers={'Accept':'application/json',
                            'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
#print(json.dumps(resp.json(),
#                 sort_keys=True,
#                 indent=4, separators=(',', ': ')))  

print(resp)


