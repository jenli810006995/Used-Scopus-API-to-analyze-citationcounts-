#!/usr/bin/env python
# coding: utf-8

# # Install packages

# In[3]:


import tensorflow as tf


# In[10]:


from sklearn.cluster import KMeans


# In[12]:


import pandas as pd


# In[13]:


import numpy as np


# In[14]:


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

# # The following is data-cleaning process

# # read in school list

# In[ ]:


school_list = open(r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_School_List_OK.txt")

school_name=school_list.read()


# In[ ]:


import pandas as pd
t = school_name

data=[]

for i in t.split("\n"):
    if i[:1].isdigit():
        data.append(" ".join(i.split(" ")[:20]))
        print(" ".join(i.split(" ")[:20]))
        
data_want = pd.DataFrame(data, columns=['Scool Name'])


data_want.to_csv("all_university_name.csv", index=False)  # all the university name


# In[ ]:


# cleaned all the ranks and leadning and trailing whitespace

t = school_name

uni_name = []

for i in t.split("\n"):
    if i[:1].isdigit():
        uni_name.append(" ".join(i.split(" ")[-5:]))
        print(" ".join(i.split(" ")[-5:]))
        uni_name.append(" ".join(i.split(" ")[-5:]))


# In[ ]:


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


# In[ ]:


# remove existing numbers

import string
import re

want_3=[]

for name in cleaned:
    print(name)
    print(re.sub('^\d+[\W_]+','',name))
    want_3.append(re.sub('^\d+[\W_]+','',name))


# In[11]:


want_3.append('University of Rochester')


# In[15]:


DF={}

DF=pd.DataFrame({'UniName':want_3})


# In[17]:


DF=DF.drop_duplicates()


# In[19]:


DF=DF.reset_index()


# In[21]:


DF=DF.iloc[:,1]


# In[25]:


DF=pd.DataFrame(DF)


# In[26]:


DF.to_csv("UniNameList_OK.csv", index=False)


# # Use APIs

# In[9]:


for line in want_3:
    url= "https://api.elsevier.com/metrics/institution/search?query=name("+line+")"
    print(url)


# # combine all the Uids retrieved from APIs

# In[34]:


filename='THE_CountryCode_Result_1202_{}'

for i in range(1,14):
    print(filename.format(i))


# In[36]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane"


# In[61]:


filename='THE_CountryCode_Result_1202_{}.csv'

chucks=[]

for i in range(1,14):
#    print(filename.format(i))
    chucks.append(pd.read_csv(filename.format(i)))

data=pd.concat(chucks, ignore_index=True)
    
data.head()


# In[62]:


del data['Unnamed: 0']


# # Use SciVal institution metrics API

# In[42]:


# https://api.elsevier.com/analytics/scival/institution/metrics


# In[63]:


data.head()


# In[64]:


for line in data['University id'][:2]:
    print(line)


# In[85]:


for line in data['University id'][:2]:
    print(line)


# # ScholarlyOutput

# In[97]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data"


# In[145]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]


for line in data['University id'][1000:]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitationCount&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if result['results'] is not None:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'valueByYear' in result['results'][0]['metrics'][0]:
                    if '2014' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2014.append(result['results'][0]['metrics'][0]['valueByYear']['2014'])
                    if '2015' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2015.append(result['results'][0]['metrics'][0]['valueByYear']['2015'])
                    if '2016' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2016.append(result['results'][0]['metrics'][0]['valueByYear']['2016'])
                    if '2017' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2017.append(result['results'][0]['metrics'][0]['valueByYear']['2017'])
                    if '2018' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2018.append(result['results'][0]['metrics'][0]['valueByYear']['2018'])

s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
s7=pd.Series(value2014, name='2014')
s8=pd.Series(value2015, name='2015')
s9=pd.Series(value2016, name='2016')
s10=pd.Series(value2017, name='2017')
s11=pd.Series(value2018, name='2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11], axis=1)
DF.to_csv("THE_UNI_CitationCount_ALL_11.csv", index=False)


# # CitationCount, CitedPublications, FWCI, and PublicationinTopJournal Percentile

# In[122]:


# FWCI


# In[133]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]
percentage2014=[]
percentage2015=[]
percentage2016=[]
percentage2017=[]
percentage2018=[]


for line in data['University id'][1000:]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=FieldWeightedCitationImpact&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if result['results'] is not None:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'valueByYear' in result['results'][0]['metrics'][0]:
                    if '2014' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2014.append(result['results'][0]['metrics'][0]['valueByYear']['2014'])
                    if '2015' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2015.append(result['results'][0]['metrics'][0]['valueByYear']['2015'])
                    if '2016' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2016.append(result['results'][0]['metrics'][0]['valueByYear']['2016'])
                    if '2017' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2017.append(result['results'][0]['metrics'][0]['valueByYear']['2017'])
                    if '2018' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2018.append(result['results'][0]['metrics'][0]['valueByYear']['2018'])
                if 'percentageByYear' in result['results'][0]['metrics'][0]:
                    if '2014' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2014.append(result['results'][0]['metrics'][0]['percentageByYear']['2014'])
                    if '2015' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2015.append(result['results'][0]['metrics'][0]['percentageByYear']['2015'])
                    if '2016' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2016.append(result['results'][0]['metrics'][0]['percentageByYear']['2016'])
                    if '2017' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2017.append(result['results'][0]['metrics'][0]['percentageByYear']['2017'])
                    if '2018' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2018.append(result['results'][0]['metrics'][0]['percentageByYear']['2018'])
                else:
                    percentage2014.append('')
                    percentage2015.append('')
                    percentage2016.append('')
                    percentage2017.append('')
                    percentage2018.append('')
                    

s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
s7=pd.Series(value2014, name='2014')
s8=pd.Series(value2015, name='2015')
s9=pd.Series(value2016, name='2016')
s10=pd.Series(value2017, name='2017')
s11=pd.Series(value2018, name='2018')
#s12=pd.Series(percentage2014, name='percent2014')
#s13=pd.Series(percentage2015, name='percent2015')
#s14=pd.Series(percentage2016, name='percent2016')
#s15=pd.Series(percentage2017, name='percent2017')
#s16=pd.Series(percentage2018, name='percent2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11], axis=1)
DF.to_csv("THE_UNI_FWCI_11.csv", index=False)


# In[134]:


# CitationCount


# In[133]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]
percentage2014=[]
percentage2015=[]
percentage2016=[]
percentage2017=[]
percentage2018=[]


for line in data['University id'][1000:]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=FieldWeightedCitationImpact&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if result['results'] is not None:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'valueByYear' in result['results'][0]['metrics'][0]:
                    if '2014' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2014.append(result['results'][0]['metrics'][0]['valueByYear']['2014'])
                    if '2015' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2015.append(result['results'][0]['metrics'][0]['valueByYear']['2015'])
                    if '2016' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2016.append(result['results'][0]['metrics'][0]['valueByYear']['2016'])
                    if '2017' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2017.append(result['results'][0]['metrics'][0]['valueByYear']['2017'])
                    if '2018' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2018.append(result['results'][0]['metrics'][0]['valueByYear']['2018'])
                if 'percentageByYear' in result['results'][0]['metrics'][0]:
                    if '2014' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2014.append(result['results'][0]['metrics'][0]['percentageByYear']['2014'])
                    if '2015' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2015.append(result['results'][0]['metrics'][0]['percentageByYear']['2015'])
                    if '2016' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2016.append(result['results'][0]['metrics'][0]['percentageByYear']['2016'])
                    if '2017' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2017.append(result['results'][0]['metrics'][0]['percentageByYear']['2017'])
                    if '2018' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2018.append(result['results'][0]['metrics'][0]['percentageByYear']['2018'])
                else:
                    percentage2014.append('')
                    percentage2015.append('')
                    percentage2016.append('')
                    percentage2017.append('')
                    percentage2018.append('')
                    

s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
s7=pd.Series(value2014, name='2014')
s8=pd.Series(value2015, name='2015')
s9=pd.Series(value2016, name='2016')
s10=pd.Series(value2017, name='2017')
s11=pd.Series(value2018, name='2018')
#s12=pd.Series(percentage2014, name='percent2014')
#s13=pd.Series(percentage2015, name='percent2015')
#s14=pd.Series(percentage2016, name='percent2016')
#s15=pd.Series(percentage2017, name='percent2017')
#s16=pd.Series(percentage2018, name='percent2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11], axis=1)
DF.to_csv("THE_UNI_FWCI_11.csv", index=False)


# In[146]:


# CitedPublications


# In[159]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]
percentage2014=[]
percentage2015=[]
percentage2016=[]
percentage2017=[]
percentage2018=[]


for line in data['University id'][1000:]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=CitedPublications&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if 'results' in result:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'valueByYear' in result['results'][0]['metrics'][0]:
                    if '2014' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2014.append(result['results'][0]['metrics'][0]['valueByYear']['2014'])
                    if '2015' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2015.append(result['results'][0]['metrics'][0]['valueByYear']['2015'])
                    if '2016' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2016.append(result['results'][0]['metrics'][0]['valueByYear']['2016'])
                    if '2017' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2017.append(result['results'][0]['metrics'][0]['valueByYear']['2017'])
                    if '2018' in result['results'][0]['metrics'][0]['valueByYear']:
                        value2018.append(result['results'][0]['metrics'][0]['valueByYear']['2018'])
                if 'percentageByYear' in result['results'][0]['metrics'][0]:
                    if '2014' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2014.append(result['results'][0]['metrics'][0]['percentageByYear']['2014'])
                    if '2015' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2015.append(result['results'][0]['metrics'][0]['percentageByYear']['2015'])
                    if '2016' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2016.append(result['results'][0]['metrics'][0]['percentageByYear']['2016'])
                    if '2017' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2017.append(result['results'][0]['metrics'][0]['percentageByYear']['2017'])
                    if '2018' in result['results'][0]['metrics'][0]['percentageByYear']:
                        percentage2018.append(result['results'][0]['metrics'][0]['percentageByYear']['2018'])
                else:
                    percentage2014.append('')
                    percentage2015.append('')
                    percentage2016.append('')
                    percentage2017.append('')
                    percentage2018.append('')
                    

s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
s7=pd.Series(value2014, name='2014')
s8=pd.Series(value2015, name='2015')
s9=pd.Series(value2016, name='2016')
s10=pd.Series(value2017, name='2017')
s11=pd.Series(value2018, name='2018')
s12=pd.Series(percentage2014, name='percent2014')
s13=pd.Series(percentage2015, name='percent2015')
s14=pd.Series(percentage2016, name='percent2016')
s15=pd.Series(percentage2017, name='percent2017')
s16=pd.Series(percentage2018, name='percent2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16], axis=1)
DF.to_csv("THE_UNI_CitedPublications_11.csv", index=False)


# In[160]:


#PublicationsInTopJournalPercentiles


# In[186]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
threshold=[]
t1_value2014=[]
t1_value2015=[]
t1_value2016=[]
t1_value2017=[]
t1_value2018=[]
t1_percentage2014=[]
t1_percentage2015=[]
t1_percentage2016=[]
t1_percentage2017=[]
t1_percentage2018=[]
t5_value2014=[]
t5_value2015=[]
t5_value2016=[]
t5_value2017=[]
t5_value2018=[]
t5_percentage2014=[]
t5_percentage2015=[]
t5_percentage2016=[]
t5_percentage2017=[]
t5_percentage2018=[]
t10_value2014=[]
t10_value2015=[]
t10_value2016=[]
t10_value2017=[]
t10_value2018=[]
t10_percentage2014=[]
t10_percentage2015=[]
t10_percentage2016=[]
t10_percentage2017=[]
t10_percentage2018=[]
t25_value2014=[]
t25_value2015=[]
t25_value2016=[]
t25_value2017=[]
t25_value2018=[]
t25_percentage2014=[]
t25_percentage2015=[]
t25_percentage2016=[]
t25_percentage2017=[]
t25_percentage2018=[]



for line in data['University id'][:2]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=PublicationsInTopJournalPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if 'results' in result:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'values' in result['results'][0]['metrics'][0]:
                    if 'threshold' in result['results'][0]['metrics'][0]['values']:
                        threshold.append(result['results'][0]['metrics'][0]['values'][0]['threshold'])
                    if 'valueByYear' in result['results'][0]['metrics'][0]['values']:
                        if '2014' in result['results'][0]['metrics'][0]['values']['valueByYear']:
                            t1_value2014.append(result['results'][0]['metrics'][0]['values']['valueByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values']['valueByYear']:
                            t1_value2015.append(result['results'][0]['metrics'][0]['values']['valueByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values']['valueByYear']:
                            t1_value2016.append(result['results'][0]['metrics'][0]['values']['valueByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values']['valueByYear']:
                            t1_value2017.append(result['results'][0]['metrics'][0]['values']['valueByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values']['valueByYear']:
                            t1_value2018.append(result['results'][0]['metrics'][0]['values']['valueByYear']['2018'])
                    if 'percentageByYear' in result['results'][0]['metrics'][0]['values']:
                        if '2014' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2014.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2015.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2016.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2017.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2018.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2018'])
#                    else:
#                        t1_value2014.append('')
#                        t1_value2015.append('')
#                        t1_value2016.append('')
#                        t1_value2017.append('')
#                        t1_value2018.append('')
#                        t1_percentage2014.append('')
#                        t1_percentage2015.append('')
#                        t1_percentage2016.append('')
#                        t1_percentage2017.append('')
#                        t1_percentage2018.append('')

s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
s7=pd.Series(threshold, name='threshold')
s8=pd.Series(t1_value2014, name='2014')
s9=pd.Series(t1_value2015, name='2015')
s10=pd.Series(t1_value2016, name='2016')
s11=pd.Series(t1_value2017, name='2017')
s12=pd.Series(t1_value2018, name='2018')
s13=pd.Series(t1_percentage2014, name='percent2014')
s14=pd.Series(t1_percentage2015, name='percent2015')
s15=pd.Series(t1_percentage2016, name='percent2016')
s16=pd.Series(t1_percentage2017, name='percent2017')
s17=pd.Series(t1_percentage2018, name='percent2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16, s17], axis=1)
DF.to_csv("THE_UNI_PublicationsInTopJournalPercentiles_TEST_1.csv", index=False)


# In[206]:


metricType=[]
threshold=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]
percent2014=[]
percent2015=[]
percent2016=[]
percent2017=[]
percent2018=[]


for line in data['University id'][:2]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=PublicationsInTopJournalPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
print(result['results'][0]['metrics'][0]['values'][3]['percentageByYear'])


# In[214]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
threshold=[]
t1_value2014=[]
t1_value2015=[]
t1_value2016=[]
t1_value2017=[]
t1_value2018=[]
t1_percentage2014=[]
t1_percentage2015=[]
t1_percentage2016=[]
t1_percentage2017=[]
t1_percentage2018=[]
t5_value2014=[]
t5_value2015=[]
t5_value2016=[]
t5_value2017=[]
t5_value2018=[]
t5_percentage2014=[]
t5_percentage2015=[]
t5_percentage2016=[]
t5_percentage2017=[]
t5_percentage2018=[]
t10_value2014=[]
t10_value2015=[]
t10_value2016=[]
t10_value2017=[]
t10_value2018=[]
t10_percentage2014=[]
t10_percentage2015=[]
t10_percentage2016=[]
t10_percentage2017=[]
t10_percentage2018=[]
t25_value2014=[]
t25_value2015=[]
t25_value2016=[]
t25_value2017=[]
t25_value2018=[]
t25_percentage2014=[]
t25_percentage2015=[]
t25_percentage2016=[]
t25_percentage2017=[]
t25_percentage2018=[]



for line in data['University id'][50:75]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=PublicationsInTopJournalPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if 'results' in result:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'values' in result['results'][0]['metrics'][0]:
#                    print(result['results'][0]['metrics'][0]['values'][1]['threshold'])
                    for i in range(0, len(result['results'][0]['metrics'][0]['values'])):
                        threshold.append(result['results'][0]['metrics'][0]['values'][i]['threshold'])
                        if 'valueByYear' in result['results'][0]['metrics'][0]['values'][i]:
#                        if i ==0:
                            if '2014' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2014.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2015.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2016.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2017.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2018.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2018'])
#                        if i ==1:
                            if '2014' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2014.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2015.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2016.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2017.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2018.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2018'])
                                
#                        if i ==2:
                            if '2014' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2014.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2015.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2016.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2017.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2018.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2018'])
                                
#                        if i ==3:
                            if '2014' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2014.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2015.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2016.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2017.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2018.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2018'])
                                
                        if 'percentageByYear' in result['results'][0]['metrics'][0]['values'][i]:
#                        if i ==0:
                            if '2014' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2014.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2015.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2016.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2017.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2018.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2018'])
                                
#                        if i ==1:
                            if '2014' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2014.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2015.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2016.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2017.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2018.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2018'])
                                
#                        if i ==2:                                
                            if '2014' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2014.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2015.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2016.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2017.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2018.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2018'])
                                
#                        if i ==3:                                
                            if '2014' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2014.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2015.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2016.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2017.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2018.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2018'])
#                    else:
#                        t1_value2014.append('')
#                        t1_value2015.append('')
#                        t1_value2016.append('')
#                        t1_value2017.append('')
#                        t1_value2018.append('')
#                        t1_percentage2014.append('')
#                        t1_percentage2015.append('')
#                        t1_percentage2016.append('')
#                        t1_percentage2017.append('')
#                        t1_percentage2018.append('')

#                    else:
#                        t1_value2014.append('')
#                        t1_value2015.append('')
#                        t1_value2016.append('')
#                        t1_value2017.append('')
#                        t1_value2018.append('')
#                        t1_percentage2014.append('')
#                        t1_percentage2015.append('')
#                        t1_percentage2016.append('')
#                        t1_percentage2017.append('')
#                        t1_percentage2018.append('')

#                    if 'threshold' in result['results'][0]['metrics'][0]['values']:
#                        threshold.append(result['results'][0]['metrics'][0]['values'][0]['threshold'])


s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
s7=pd.Series(threshold, name='threshold')
s8=pd.Series(t1_value2014, name='t1_2014')
s9=pd.Series(t1_value2015, name='t1_2015')
s10=pd.Series(t1_value2016, name='t1_2016')
s11=pd.Series(t1_value2017, name='t1_2017')
s12=pd.Series(t1_value2018, name='t1_2018')
s13=pd.Series(t1_percentage2014, name='t1_percent2014')
s14=pd.Series(t1_percentage2015, name='t1_percent2015')
s15=pd.Series(t1_percentage2016, name='t1_percent2016')
s16=pd.Series(t1_percentage2017, name='t1_percent2017')
s17=pd.Series(t1_percentage2018, name='t1_percent2018')
s18=pd.Series(t5_value2014, name='t5_2014')
s19=pd.Series(t5_value2015, name='t5_2015')
s20=pd.Series(t5_value2016, name='t5_2016')
s21=pd.Series(t5_value2017, name='t5_2017')
s22=pd.Series(t5_value2018, name='t5_2018')
s23=pd.Series(t5_percentage2014, name='t5_percent2014')
s24=pd.Series(t5_percentage2015, name='t5_percent2015')
s25=pd.Series(t5_percentage2016, name='t5_percent2016')
s26=pd.Series(t5_percentage2017, name='t5_percent2017')
s27=pd.Series(t5_percentage2018, name='t5_percent2018')
s28=pd.Series(t10_value2014, name='t10_2014')
s29=pd.Series(t10_value2015, name='t10_2015')
s30=pd.Series(t10_value2016, name='t10_2016')
s31=pd.Series(t10_value2017, name='t10_2017')
s32=pd.Series(t10_value2018, name='t10_2018')
s33=pd.Series(t10_percentage2014, name='t10_percent2014')
s34=pd.Series(t10_percentage2015, name='t10_percent2015')
s35=pd.Series(t10_percentage2016, name='t10_percent2016')
s36=pd.Series(t10_percentage2017, name='t10_percent2017')
s37=pd.Series(t10_percentage2018, name='t10_percent2018')
s38=pd.Series(t25_value2014, name='t25_2014')
s39=pd.Series(t25_value2015, name='t25_2015')
s40=pd.Series(t25_value2016, name='t25_2016')
s41=pd.Series(t25_value2017, name='t25_2017')
s42=pd.Series(t25_value2018, name='t25_2018')
s43=pd.Series(t25_percentage2014, name='t25_percent2014')
s44=pd.Series(t25_percentage2015, name='t25_percent2015')
s45=pd.Series(t25_percentage2016, name='t25_percent2016')
s46=pd.Series(t25_percentage2017, name='t25_percent2017')
s47=pd.Series(t25_percentage2018, name='t25_percent2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16, s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,
             s28,s29,s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40, s41,s42,s43,s44,s45,s46,s47], axis=1)


DF.to_csv("THE_UNI_PublicationsInTopJournalPercentiles_ALL_3.csv", index=False)  # OK



#print(threshold)


# In[233]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
#threshold=[]
t1_value2014=[]
t1_value2015=[]
t1_value2016=[]
t1_value2017=[]
t1_value2018=[]
t1_percentage2014=[]
t1_percentage2015=[]
t1_percentage2016=[]
t1_percentage2017=[]
t1_percentage2018=[]
t5_value2014=[]
t5_value2015=[]
t5_value2016=[]
t5_value2017=[]
t5_value2018=[]
t5_percentage2014=[]
t5_percentage2015=[]
t5_percentage2016=[]
t5_percentage2017=[]
t5_percentage2018=[]
t10_value2014=[]
t10_value2015=[]
t10_value2016=[]
t10_value2017=[]
t10_value2018=[]
t10_percentage2014=[]
t10_percentage2015=[]
t10_percentage2016=[]
t10_percentage2017=[]
t10_percentage2018=[]
t25_value2014=[]
t25_value2015=[]
t25_value2016=[]
t25_value2017=[]
t25_value2018=[]
t25_percentage2014=[]
t25_percentage2015=[]
t25_percentage2016=[]
t25_percentage2017=[]
t25_percentage2018=[]



for line in data['University id'][1000:]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=PublicationsInTopJournalPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if 'results' in result:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'values' in result['results'][0]['metrics'][0]:
#                    print(result['results'][0]['metrics'][0]['values'][1]['threshold'])
#                    for i in range(0, len(result['results'][0]['metrics'][0]['values'])):
#                        threshold.append(result['results'][0]['metrics'][0]['values'][i]['threshold'])
                    if 'valueByYear' in result['results'][0]['metrics'][0]['values'][0]:
#                        if i ==0:
                        if '2014' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                            t1_value2014.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                            t1_value2015.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                            t1_value2016.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                            t1_value2017.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                            t1_value2018.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2018'])
#                        if i ==1:
                    if 'valueByYear' in result['results'][0]['metrics'][0]['values'][1]:
                        if '2014' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_value2014.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_value2015.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_value2016.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_value2017.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_value2018.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2018'])
                                
#                        if i ==2:
                    if 'valueByYear' in result['results'][0]['metrics'][0]['values'][2]:
                        if '2014' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                            t10_value2014.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                            t10_value2015.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                            t10_value2016.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                            t10_value2017.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                            t10_value2018.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2018'])
                                
#                        if i ==3:
                    if 'valueByYear' in result['results'][0]['metrics'][0]['values'][3]:
                        if '2014' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                            t25_value2014.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                            t25_value2015.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                            t25_value2016.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                            t25_value2017.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                            t25_value2018.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2018'])
                                
                    if 'percentageByYear' in result['results'][0]['metrics'][0]['values'][0]:
#                        if i ==0:
                        if '2014' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2014.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2015.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2016.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2017.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                            t1_percentage2018.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2018'])
                                
#                        if i ==1:
                    if 'percentageByYear' in result['results'][0]['metrics'][0]['values'][1]:
                        if '2014' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_percentage2014.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_percentage2015.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_percentage2016.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_percentage2017.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                            t5_percentage2018.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2018'])
                                
#                        if i ==2:      
                    if 'percentageByYear' in result['results'][0]['metrics'][0]['values'][2]:
                        if '2014' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                            t10_percentage2014.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                            t10_percentage2015.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                            t10_percentage2016.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                            t10_percentage2017.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                            t10_percentage2018.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2018'])
                                
#                        if i ==3:
                    if 'percentageByYear' in result['results'][0]['metrics'][0]['values'][3]:
                        if '2014' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                            t25_percentage2014.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2014'])
                        if '2015' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                            t25_percentage2015.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2015'])
                        if '2016' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                            t25_percentage2016.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2016'])
                        if '2017' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                            t25_percentage2017.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2017'])
                        if '2018' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                            t25_percentage2018.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2018'])
#                    else:
#                        t1_value2014.append('')
#                        t1_value2015.append('')
#                        t1_value2016.append('')
#                        t1_value2017.append('')
#                        t1_value2018.append('')
#                        t1_percentage2014.append('')
#                        t1_percentage2015.append('')
#                        t1_percentage2016.append('')
#                        t1_percentage2017.append('')
#                        t1_percentage2018.append('')

#                    else:
#                        t1_value2014.append('')
#                        t1_value2015.append('')
#                        t1_value2016.append('')
#                        t1_value2017.append('')
#                        t1_value2018.append('')
#                        t1_percentage2014.append('')
#                        t1_percentage2015.append('')
#                        t1_percentage2016.append('')
#                        t1_percentage2017.append('')
#                        t1_percentage2018.append('')

#                    if 'threshold' in result['results'][0]['metrics'][0]['values']:
#                        threshold.append(result['results'][0]['metrics'][0]['values'][0]['threshold'])


s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
#s7=pd.Series(threshold, name='threshold')
s8=pd.Series(t1_value2014, name='t1_2014')
s9=pd.Series(t1_value2015, name='t1_2015')
s10=pd.Series(t1_value2016, name='t1_2016')
s11=pd.Series(t1_value2017, name='t1_2017')
s12=pd.Series(t1_value2018, name='t1_2018')
s13=pd.Series(t1_percentage2014, name='t1_percent2014')
s14=pd.Series(t1_percentage2015, name='t1_percent2015')
s15=pd.Series(t1_percentage2016, name='t1_percent2016')
s16=pd.Series(t1_percentage2017, name='t1_percent2017')
s17=pd.Series(t1_percentage2018, name='t1_percent2018')
s18=pd.Series(t5_value2014, name='t5_2014')
s19=pd.Series(t5_value2015, name='t5_2015')
s20=pd.Series(t5_value2016, name='t5_2016')
s21=pd.Series(t5_value2017, name='t5_2017')
s22=pd.Series(t5_value2018, name='t5_2018')
s23=pd.Series(t5_percentage2014, name='t5_percent2014')
s24=pd.Series(t5_percentage2015, name='t5_percent2015')
s25=pd.Series(t5_percentage2016, name='t5_percent2016')
s26=pd.Series(t5_percentage2017, name='t5_percent2017')
s27=pd.Series(t5_percentage2018, name='t5_percent2018')
s28=pd.Series(t10_value2014, name='t10_2014')
s29=pd.Series(t10_value2015, name='t10_2015')
s30=pd.Series(t10_value2016, name='t10_2016')
s31=pd.Series(t10_value2017, name='t10_2017')
s32=pd.Series(t10_value2018, name='t10_2018')
s33=pd.Series(t10_percentage2014, name='t10_percent2014')
s34=pd.Series(t10_percentage2015, name='t10_percent2015')
s35=pd.Series(t10_percentage2016, name='t10_percent2016')
s36=pd.Series(t10_percentage2017, name='t10_percent2017')
s37=pd.Series(t10_percentage2018, name='t10_percent2018')
s38=pd.Series(t25_value2014, name='t25_2014')
s39=pd.Series(t25_value2015, name='t25_2015')
s40=pd.Series(t25_value2016, name='t25_2016')
s41=pd.Series(t25_value2017, name='t25_2017')
s42=pd.Series(t25_value2018, name='t25_2018')
s43=pd.Series(t25_percentage2014, name='t25_percent2014')
s44=pd.Series(t25_percentage2015, name='t25_percent2015')
s45=pd.Series(t25_percentage2016, name='t25_percent2016')
s46=pd.Series(t25_percentage2017, name='t25_percent2017')
s47=pd.Series(t25_percentage2018, name='t25_percent2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s8,s9,s10,s11,s12,s13,s14,s15,s16, s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,
             s28,s29,s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40, s41,s42,s43,s44,s45,s46,s47], axis=1)


DF.to_csv("THE_UNI_PubPercentile_All_17.csv", index=False)  # OK



#print(threshold)


# # Combine all the subfiles and subset the USA universities

# # CitationCount

# In[234]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\CitationCount"


# In[235]:


filename='THE_UNI_CitationCount_ALL_{}.csv'


# In[237]:


chucks=[]

for i in range(1, 12):
    chucks.append(pd.read_csv(filename.format(i)))

cc_data=pd.concat(chucks, ignore_index=True)

cc_data.head()


# In[240]:


cc_data.to_csv('THE_ALLUNI_CC.csv', index=True)


# # FWCI

# In[241]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\FNCI"


# In[242]:


filename='THE_UNI_FWCI_{}.csv'


# In[244]:


chucks=[]

for i in range(1, 12):
    chucks.append(pd.read_csv(filename.format(i)))

fwci_data=pd.concat(chucks, ignore_index=True)

fwci_data.head()


# In[245]:


fwci_data.to_csv("THE_ALLUNI_FWCI.csv", index=False)


# # PercPublsCited

# In[246]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[247]:


filename='THE_UNI_CitedPublications_{}.csv'


# In[248]:


chucks=[]

for i in range(1, 12):
    chucks.append(pd.read_csv(filename.format(i)))

cp_data=pd.concat(chucks, ignore_index=True)

cp_data.head()


# In[249]:


cp_data.to_csv("THEUNI_CITEDPUBLS.csv",index=False)


# # PubTopJournalPercentile

# In[250]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PubTopJournalPercentile"


# In[251]:


filename='THE_UNI_PubPercentile_All_{}.csv'


# In[252]:


chucks=[]

for i in range(1, 18):
    chucks.append(pd.read_csv(filename.format(i)))

pp_data=pd.concat(chucks, ignore_index=True)

pp_data.head()


# In[253]:


pp_data.to_csv("THE_ALLUNI_PP.csv", index=False)


# # ScholarlyOutput

# In[256]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ScholarlyOutput" # needs to use double quote


# In[257]:


filename='THE_UNI_SCHOLAROUTPUT_ALL_{}.csv'


# In[258]:


chucks=[]

for i in range(1, 15):
    chucks.append(pd.read_csv(filename.format(i)))

so_data=pd.concat(chucks, ignore_index=True)

so_data.head()


# In[259]:


so_data.to_csv("THE_ALLUNI_SO.csv", index=False)


# # USA University Publication Output

# # Total

# In[261]:


so_data.head()


# In[303]:


so_data[so_data.countryCode=='USA'].head()
so_data_USA=so_data[so_data.countryCode=='USA']


# In[263]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[316]:


so_data_USA=so_data_USA.iloc[:,-7:]


# In[317]:


so_data_USA.head()


# In[318]:


del so_data_USA['metricType']


# In[319]:


so_data_USA.head()


# In[365]:


so_data_USA=so_data_USA.set_index('institution_name')


# In[366]:


so_data_USA.agg('sum')


# In[379]:


x=URpp.agg('sum')
sns.distplot(x)


# In[378]:


sns.distplot(so_data_USA.agg('sum'))


# In[346]:


len(so_data_USA) # 163 USA universities


# In[347]:


so_data_USA=so_data_USA.set_index('institution_name')


# In[349]:


so_data_USA.agg('sum')


# In[411]:


so_data_USA=so_data_USA.reset_index()


# In[412]:


so_data_USA.info()


# In[391]:


sep_sum=lambda x: x.agg('sum')


# In[417]:


so_data_USA['Total']=so_data_USA.sum(axis=1)


# In[421]:


so_data_USA['Total']=so_data_USA.Total.astype(int)
so_data_USA.head()


# In[423]:


URpp=URpp.reset_index()


# In[424]:


URpp['Total']=URpp.sum(axis=1)


# In[425]:


URpp


# In[429]:


# UR Publs Distribution
inputdata=URpp[['2014','2015','2016','2017','2018']]
sns.distplot(inputdata)

# seems a bi-modal distribution but the overall trend is downward


# # Top 1% and top 10% highly cited publications 

# In[436]:


pp_data.head()


# In[437]:


USA_pp=pp_data[pp_data.countryCode=='USA']


# In[439]:


len(USA_pp)


# In[441]:


# we want t1 and t10 values

USA_pp.head()


# In[442]:


USA_pp.columns


# In[443]:


USA_pp=USA_pp.loc[:][['institution_name','t1_2014','t1_2015','t1_2016','t1_2017','t1_2018','t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[445]:


USA_pp=USA_pp.drop_duplicates()


# In[447]:


USA_pp=USA_pp.reset_index()


# In[449]:


USA_pp=USA_pp.iloc[:,1:]


# In[450]:


USA_pp.head()


# In[451]:


USA_pp['2014_Total']=USA_pp.loc[:][['t1_2014','t10_2014']].sum(axis=1)


# In[452]:


USA_pp.head()


# In[453]:


USA_pp['2015_Total']=USA_pp.loc[:][['t1_2015','t10_2015']].sum(axis=1)
USA_pp['2016_Total']=USA_pp.loc[:][['t1_2016','t10_2016']].sum(axis=1)
USA_pp['2017_Total']=USA_pp.loc[:][['t1_2017','t10_2017']].sum(axis=1)
USA_pp['2018_Total']=USA_pp.loc[:][['t1_2018','t10_2018']].sum(axis=1)


# In[454]:


USA_pp.head()


# In[455]:


UR_percentile=USA_pp[USA_pp.institution_name=='University of Rochester']


# In[457]:


UR_percentile=UR_percentile.set_index('institution_name')


# In[458]:


UR_percentile


# In[459]:


basedata=UR_percentile[['2014_Total','2015_Total','2016_Total','2017_Total','2018_Total']]


# In[464]:


basedata


# In[462]:


smalldata=UR_percentile.iloc[:,:10]


# In[465]:


smalldata1=smalldata.loc[:][['t1_2014','t1_2015','t1_2016','t1_2017','t1_2018']]


# In[467]:


smalldata1


# In[466]:


smalldata2=smalldata.loc[:][['t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[468]:


smalldata2


# In[478]:


# UR's ScholarlyOutput

so_data_USA.head()


# In[479]:


UR_so=so_data_USA[so_data_USA.institution_name=='University of Rochester']


# In[486]:


UR_so
del UR_so['Total']


# In[513]:


UR_so


# In[514]:


combinedata=pd.DataFrame({'2014':[int(162.0),int(1404.0),3602], '2015':[int(164.0),int(1308.0), 3540],
                         '2016':[int(143.0), int(1310.0),3515],
                         '2017':[int(138.0),int(1309.0),3633],
                         '2018':[int(133.0), int(1318.0),3842]})


# In[515]:


combinedata


# In[516]:


data_1=combinedata.iloc[2,:]
data_2=combinedata.iloc[1,:]
data_3=combinedata.iloc[0,:]


# In[532]:


data_1


# In[533]:


A=pd.DataFrame(data=[data_1[:5]], columns=['2014','2015','2016','2017','2018'])


# In[534]:


A


# In[535]:


B=pd.DataFrame(data=[data_2[:5]], columns=['2014','2015','2016','2017','2018'])


# In[536]:


C=pd.DataFrame(data=[data_3[:5]], columns=['2014','2015','2016','2017','2018'])


# In[538]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
sns.set_style("ticks", {"xtick.major.size": 10, "ytick.major.size": 8})

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
#crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(data=A,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("dark")
sns.barplot(data=B,
            label="Top 10%", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(data=C,
            label="Top 1%", color="g")


# Add a legend and informative axis label
plt.yticks(np.arange(0, 4000, step=500))
plt.xticks(np.arange(5), ('2014', '2015', '2016', '2017', '2018'))
ax.legend(ncol=3, loc="upper right", frameon=True)
ax.set(xlim=(0,5), ylabel="",
       title="U of R publication output: total, top 1 % and top 10 % highly cited publs")
sns.despine(left=True, bottom=True)


# # From 2014-2018 ,our top 1% cited publs and top10% cited pulbs slightly dropped a little, but because our 2018 total publs increased a lot, our % pulb. cited would drop

# # Trends in FWCI values of total U of R publication output

# In[539]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\FNCI"


# In[540]:


FWCI_all=pd.read_csv('THE_ALLUNI_FWCI.csv')


# In[541]:


FWCI_all.head()


# In[548]:


UR_FWCI=FWCI_all[FWCI_all.institution_name=='University of Rochester']


# In[550]:


UR_FWCI=UR_FWCI.iloc[:, -7:]


# In[553]:


del UR_FWCI['metricType']


# In[556]:


UR_FWCI


# # UofR FWCI

# In[573]:


sns.barplot(data=UR_FWCI)
plt.axhline(1.00, ls='-', color='r')
plt.title('UofR FWCI 2014-2018 with World Average')
plt.xlabel("UofR FWCI 2014-2018")
plt.ylabel("Filed-weighted Cited Index")


# # Our FWCI have always been above global average which is 1.00 

# # Comparator analysis: top 10 % highly cited publications for USA universities

# In[574]:


USA_pp.head()


# In[575]:


UR_peer=['Boston University','Carnegie Mellon University','Case Western Reserve University','Duke University','Emory University',
        'Northwestern University','Vanderbilt University','Washington University','Johns Hopkins University','New York University',
        'Stanford University','Tulane University','University of Chicago','University of Pennsylvania','University of Southern California']


# In[581]:


UR_peer_df=pd.DataFrame({'UR_Peer':UR_peer})


# In[613]:


UR_peer_df


# In[616]:


result=[]

for name in UR_peer_df.UR_Peer:
    if USA_pp[USA_pp.institution_name==name] is not None:
        result.append(1)
    else:
        result.append(0)
        
len(result)


# In[595]:


data=[]
for name in UR_peer:
    if name in USA_pp.institution_name:
        data.append('T')
    else:
        data.append('F')
        
data


# In[588]:


UR_peer_df.loc[:]['Result']=data


# In[624]:


UR_peer_df['UR_Peer']


# # Get UofR's Global set's Publication in Top Journal Percentile

# In[626]:


chuck=[]
for name in UR_peer_df['UR_Peer']: 
    chuck.append(USA_pp[USA_pp.institution_name==name])


# In[628]:


DF=pd.concat(chuck, ignore_index=True)


# In[629]:


DF.head()


# In[633]:


UR_percentile=UR_percentile.reset_index()


# In[632]:


Global_top10=DF.loc[:][['institution_name','t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[634]:


UR_pcer_top10=UR_percentile.loc[:][['institution_name','t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[635]:


Global_top10.head()


# In[637]:


Global_top10['Top10_Total']=Global_top10.sum(axis=1)


# In[638]:


Global_top10.head()


# In[661]:


len(Global_top10)


# In[636]:


UR_pcer_top10


# In[639]:


UR_pcer_top10['Top10_Total']=UR_pcer_top10.sum(axis=1)


# In[640]:


UR_pcer_top10


# In[641]:


Gall=pd.concat([Global_top10, UR_pcer_top10])


# In[660]:


len(Gall)


# In[647]:


import re


# In[658]:


abb=[]
for i in Gall.institution_name:
    abb.append(i.split("\t")[0].strip(" "))
abb


# In[682]:


Gall['UniAbbr']=['Boston','CWRU','Duke','Northwestern','Vanderbilt','JohnsHopkins','NYU','Stanford','Tulane','UofChicago','UofPenn','UofR']


# In[684]:


Gall=Gall.sort_values(by='Top10_Total', ascending=False)


# # Comparator analysis: top 10% highly cited publications UR and GlobalPeers

# In[699]:


for index, row in Gall.iterrows():
    print(row.UniAbbr)
    print(int(row.Top10_Total))


# In[791]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(25, 20))
sns.barplot(x=Gall.UniAbbr, y=Gall.Top10_Total, data=Gall)
plt.axhline(6649, ls='-', color='r')

#ax.text(Gall.UniAbbr, Gall.Top10_Total,color='black', ha="center")

# Add a legend and informative axis label
#ax.legend(ncol=12, loc="upper right", frameon=True)
ax.set(xlim=(0, 20),
       xlabel="University of Rochester and Global Peers", ylabel="Top 10% highly cited publications")
sns.despine(left=True, bottom=True)


# # Among our other 11 USA peers, our top 10% highly-cited pulbs ranks behind 

# # Comparator analysis: Field-weighted Citation Impact

# In[709]:


fwci_data.head()


# In[711]:


US_fwci=fwci_data[fwci_data.countryCode=='USA']


# In[712]:


US_fwci.head()


# In[713]:


UR_peer_df


# In[715]:


len(Gall.institution_name) # Global peers and UofR


# In[759]:


chuck=[]

for name in Gall.institution_name:
    if US_fwci[US_fwci.institution_name==name] is not None:
        chuck.append(US_fwci[US_fwci.institution_name==name])


# In[760]:


UR_Peer_FWCI=pd.concat(chuck, ignore_index=True)


# In[761]:


UR_Peer_FWCI


# In[719]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\FNCI"


# In[720]:


UR_Peer_FWCI.to_csv('UR_Global_Peer_FWCI_Comparison.csv', index=False)


# In[762]:


UR_Peer_FWCI=UR_Peer_FWCI.iloc[:, -7:]


# In[763]:


UR_Peer_FWCI


# In[741]:


Gall.UniAbbr


# In[771]:


abb=[]
for name in Gall.UniAbbr:
    abb.append(name)
abb


# In[764]:


UR_Peer_FWCI=UR_Peer_FWCI.drop_duplicates()


# In[765]:


UR_Peer_FWCI.reset_index(inplace=True, drop=True)


# In[774]:


UR_Peer_FWCI.loc[:]['UniAbbr']=abb


# In[775]:


UR_Peer_FWCI.head()


# In[777]:


UR_Peer_FWCI.loc[:]['AVERAGE_FWCI']=round(UR_Peer_FWCI[['2014','2015','2016','2017','2018']].mean(axis=1), 4)


# In[778]:


UR_Peer_FWCI=UR_Peer_FWCI.sort_values(by='AVERAGE_FWCI', ascending=False)


# In[779]:


UR_Peer_FWCI.head()


# In[785]:


UR_Peer_FWCI[UR_Peer_FWCI.UniAbbr=='UofR']


# # Comparatory analysis: Field-weighted Citation Impact

# In[790]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(25, 20))
sns.barplot(x=UR_Peer_FWCI.UniAbbr, y=UR_Peer_FWCI.AVERAGE_FWCI, data=UR_Peer_FWCI)
plt.axhline(1.802, ls='-', color='r')

#ax.text(Gall.UniAbbr, Gall.Top10_Total,color='black', ha="center")

# Add a legend and informative axis label
#ax.legend(ncol=12, loc="upper right", frameon=True)
plt.yticks(np.arange(0, 2.5, step=0.2))
ax.set(xlim=(0, 25),
       xlabel="University of Rochester and Global Peers", ylabel="Field-weighted Citation Impact")
sns.despine(left=True, bottom=True)


# # Our average FWCI 2014-2018 is 1.8, but most of our USA peers have higher FWCI, this may be the reason our overall score did not reflect our good FWCI

# # Comparatory analysis: research performance profile

# In[793]:


UR_Peer_FWCI.institution_name


# In[794]:


so_data_USA.head()


# In[800]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(so_data_USA[so_data_USA.institution_name==name])


# In[801]:


Ttl_publs_output=pd.concat(chuck, ignore_index=True)


# In[883]:


A=Ttl_publs_output[['institution_name','Total']]


# In[807]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[808]:


ALL_PP=pd.read_csv("THEUNI_CITEDPUBLS.csv")


# In[809]:


ALL_PP.head()


# In[810]:


US_PP=ALL_PP[ALL_PP.countryCode=='USA']


# In[811]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(US_PP[US_PP.institution_name==name])


# In[812]:


UR_Peer_PP=pd.concat(chuck, ignore_index=True)


# In[815]:


UR_Peer_PP=UR_Peer_PP[['institution_name','percent2014','percent2015','percent2016','percent2017','percent2018']]


# In[818]:


UR_Peer_PP=UR_Peer_PP.drop_duplicates()


# In[819]:


UR_Peer_PP.shape[0]


# In[821]:


UR_Peer_PP.loc[:]['UniAbbr']=abb


# In[824]:


UR_Peer_PP.loc[:]['Mean_%PubCited']=UR_Peer_PP.iloc[:,1:5].mean(axis=1)


# In[825]:


UR_Peer_PP


# In[826]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[828]:


UR_Peer_PP=UR_Peer_PP.sort_values(by='Mean_%PubCited', ascending=False)


# In[830]:


UR_Peer_PP.reset_index(inplace=True, drop=True)


# In[882]:


C=UR_Peer_PP[['institution_name','Mean_%PubCited']]


# In[832]:


UR_Peer_PP.to_csv("UofR_Global_Peers_Cited_Publs.csv", index=False)


# In[836]:


# Top 1 % cited


# In[840]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PubTopJournalPercentile"


# In[841]:


Top1All=pd.read_csv("THE_ALLUNI_PP.csv")


# In[843]:


Top1All.columns


# In[846]:


Top1=Top1All[['institution_name','t1_percent2014','t1_percent2015','t1_percent2016','t1_percent2017','t1_percent2018']]


# In[849]:


Top1=Top1.drop_duplicates()


# In[854]:


Top1['Total_Top1']=Top1[['institution_name','t1_percent2014','t1_percent2015','t1_percent2016','t1_percent2017','t1_percent2018']].mean(axis=1)


# In[855]:


Top1.head()


# In[856]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(Top1[Top1.institution_name==name])


# In[857]:


UR_PEER_Top1=pd.concat(chuck, ignore_index=True)


# In[862]:


UR_PEER_Top1=UR_PEER_Top1.sort_values(by='Total_Top1', ascending=False)


# In[876]:


UR_PEER_Top1.reset_index(inplace=True, drop=True)


# In[881]:


D=UR_PEER_Top1[['institution_name','Total_Top1']] # top1%


# In[865]:


# top 10%

Top10=Top1All[['institution_name','t10_percent2014','t10_percent2015','t10_percent2016','t10_percent2017','t10_percent2018']]


# In[866]:


Top10=Top10.drop_duplicates()


# In[867]:


Top10['Total_Top10']=Top10[['institution_name','t10_percent2014','t10_percent2015','t10_percent2016','t10_percent2017','t10_percent2018']].mean(axis=1)


# In[868]:


Top10.head()


# In[869]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(Top10[Top10.institution_name==name])


# In[870]:


UR_PEER_Top10=pd.concat(chuck, ignore_index=True)


# In[873]:


UR_PEER_Top10=UR_PEER_Top10.sort_values(by='Total_Top10', ascending=False)


# In[874]:


UR_PEER_Top10.reset_index(inplace=True, drop=True)


# In[880]:


E=UR_PEER_Top10[['institution_name','Total_Top10']]


# In[889]:


A=A.drop_duplicates()


# In[895]:


part1=A.join(C.set_index('institution_name'), on='institution_name')


# In[896]:


part2=part1.join(D.set_index('institution_name'), on='institution_name')


# In[897]:


part3=part2.join(E.set_index('institution_name'), on='institution_name')


# In[898]:


part3


# In[901]:


B=UR_Peer_FWCI[['institution_name','AVERAGE_FWCI']]


# In[903]:


part4=part3.join(B.set_index('institution_name'), on='institution_name')


# In[904]:


part4


# In[905]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\research_performance_Profile"


# In[906]:


part4.to_csv('UR_GloPeers_Research_Performance_Profile.csv', index=False)


# # From the distribution plot below, we can see we are above 75% of the other USA Universities in publication 2014-2018

# In[427]:


import pandas as pd
fig, ax = plt.subplots(figsize=(12,8))
x = pd.Series(so_data_USA['Total'], name="USA Universities Publs") # 163 universities
ax = sns.distplot(x)

ax.set_xlabel("USA 163 Universities Publs",fontsize=16)
ax.set_ylabel("Probability",fontsize=16)
plt.axvline(18132, color='red') # this is where U of R
plt.axvline(np.mean(so_data_USA['Total']), color='green') # this is the mean, 175882.56
plt.axvline(np.percentile(so_data_USA['Total'], 25.0), color='blue') # Q1
plt.axvline(np.percentile(so_data_USA['Total'], 75.0), color='orange') # Q3 very close to the mean, which means it is highly skewed
#plt.legend()
plt.tight_layout()


# In[354]:


so_data_USA=so_data_USA.reset_index()


# In[355]:


URpp=so_data_USA[so_data_USA.institution_name=='University of Rochester']
URpp


# In[356]:


URpp=URpp.set_index('institution_name')


# In[368]:


URpp.agg('sum')


# In[340]:


inputdata=pd.DataFrame(data.iloc[:,:6], columns=['2014','2015','2016','2017','2018'])


# In[341]:


inputdata.head()


# In[342]:


inputdata.reset_index(drop=True, inplace=True)


# In[214]:


import requests
import json
import pandas as pd
import numpy as np
from time import sleep
sleep(2)

inst_country=[]
inst_cc=[]
inst_id=[]
inst_link=[]
inst_name=[]
metricType=[]
threshold=[]
t1_value2014=[]
t1_value2015=[]
t1_value2016=[]
t1_value2017=[]
t1_value2018=[]
t1_percentage2014=[]
t1_percentage2015=[]
t1_percentage2016=[]
t1_percentage2017=[]
t1_percentage2018=[]
t5_value2014=[]
t5_value2015=[]
t5_value2016=[]
t5_value2017=[]
t5_value2018=[]
t5_percentage2014=[]
t5_percentage2015=[]
t5_percentage2016=[]
t5_percentage2017=[]
t5_percentage2018=[]
t10_value2014=[]
t10_value2015=[]
t10_value2016=[]
t10_value2017=[]
t10_value2018=[]
t10_percentage2014=[]
t10_percentage2015=[]
t10_percentage2016=[]
t10_percentage2017=[]
t10_percentage2018=[]
t25_value2014=[]
t25_value2015=[]
t25_value2016=[]
t25_value2017=[]
t25_value2018=[]
t25_percentage2014=[]
t25_percentage2015=[]
t25_percentage2016=[]
t25_percentage2017=[]
t25_percentage2018=[]



for line in data['University id'][50:75]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=PublicationsInTopJournalPercentiles&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
    if 'results' in result:
        if len(result['results'])>=1:
            if 'institution' in result['results'][0]:
#                if 'country' in result['results'][0]['institution']:
                inst_country.append(result['results'][0]['institution']['country'])
#            if 'countryCode' in result['results'][0]['institution']:
                inst_cc.append(result['results'][0]['institution']['countryCode'])
#            if 'id' in result['results'][0]['institution']:
                inst_id.append(result['results'][0]['institution']['id'])
#            if 'link' in result['results'][0]['institution']:
                inst_link.append(result['results'][0]['institution']['link'])
#            if 'name' in result['results'][0]['institution']:
                inst_name.append(result['results'][0]['institution']['name'])
            if 'metrics' in result['results'][0]:
#            if len(result['results'][0]['metrics'])>=1:
                if 'metricType' in result['results'][0]['metrics'][0]:
                    metricType.append(result['results'][0]['metrics'][0]['metricType'])
                if 'values' in result['results'][0]['metrics'][0]:
#                    print(result['results'][0]['metrics'][0]['values'][1]['threshold'])
                    for i in range(0, len(result['results'][0]['metrics'][0]['values'])):
                        threshold.append(result['results'][0]['metrics'][0]['values'][i]['threshold'])
                        if 'valueByYear' in result['results'][0]['metrics'][0]['values'][i]:
#                        if i ==0:
                            if '2014' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2014.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2015.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2016.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2017.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][0]['valueByYear']:
                                t1_value2018.append(result['results'][0]['metrics'][0]['values'][0]['valueByYear']['2018'])
#                        if i ==1:
                            if '2014' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2014.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2015.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2016.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2017.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_value2018.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2018'])
                                
#                        if i ==2:
                            if '2014' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2014.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2015.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2016.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2017.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][2]['valueByYear']:
                                t10_value2018.append(result['results'][0]['metrics'][0]['values'][2]['valueByYear']['2018'])
                                
#                        if i ==3:
                            if '2014' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2014.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2015.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2016.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2017.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][3]['valueByYear']:
                                t25_value2018.append(result['results'][0]['metrics'][0]['values'][3]['valueByYear']['2018'])
                                
                        if 'percentageByYear' in result['results'][0]['metrics'][0]['values'][i]:
#                        if i ==0:
                            if '2014' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2014.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2015.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2016.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2017.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][0]['percentageByYear']:
                                t1_percentage2018.append(result['results'][0]['metrics'][0]['values'][0]['percentageByYear']['2018'])
                                
#                        if i ==1:
                            if '2014' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2014.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2015.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2016.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2017.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][1]['valueByYear']:
                                t5_percentage2018.append(result['results'][0]['metrics'][0]['values'][1]['valueByYear']['2018'])
                                
#                        if i ==2:                                
                            if '2014' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2014.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2015.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2016.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2017.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][2]['percentageByYear']:
                                t10_percentage2018.append(result['results'][0]['metrics'][0]['values'][2]['percentageByYear']['2018'])
                                
#                        if i ==3:                                
                            if '2014' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2014.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2014'])
                            if '2015' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2015.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2015'])
                            if '2016' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2016.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2016'])
                            if '2017' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2017.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2017'])
                            if '2018' in result['results'][0]['metrics'][0]['values'][3]['percentageByYear']:
                                t25_percentage2018.append(result['results'][0]['metrics'][0]['values'][3]['percentageByYear']['2018'])
#                    else:
#                        t1_value2014.append('')
#                        t1_value2015.append('')
#                        t1_value2016.append('')
#                        t1_value2017.append('')
#                        t1_value2018.append('')
#                        t1_percentage2014.append('')
#                        t1_percentage2015.append('')
#                        t1_percentage2016.append('')
#                        t1_percentage2017.append('')
#                        t1_percentage2018.append('')

#                    else:
#                        t1_value2014.append('')
#                        t1_value2015.append('')
#                        t1_value2016.append('')
#                        t1_value2017.append('')
#                        t1_value2018.append('')
#                        t1_percentage2014.append('')
#                        t1_percentage2015.append('')
#                        t1_percentage2016.append('')
#                        t1_percentage2017.append('')
#                        t1_percentage2018.append('')

#                    if 'threshold' in result['results'][0]['metrics'][0]['values']:
#                        threshold.append(result['results'][0]['metrics'][0]['values'][0]['threshold'])


s1=pd.Series(inst_country, name='country')
s2=pd.Series(inst_cc, name='countryCode')
s3=pd.Series(inst_id, name='institution_id')
s4=pd.Series(inst_link, name='link')
s5=pd.Series(inst_name, name='institution_name')
s6=pd.Series(metricType, name='metricType')
s7=pd.Series(threshold, name='threshold')
s8=pd.Series(t1_value2014, name='t1_2014')
s9=pd.Series(t1_value2015, name='t1_2015')
s10=pd.Series(t1_value2016, name='t1_2016')
s11=pd.Series(t1_value2017, name='t1_2017')
s12=pd.Series(t1_value2018, name='t1_2018')
s13=pd.Series(t1_percentage2014, name='t1_percent2014')
s14=pd.Series(t1_percentage2015, name='t1_percent2015')
s15=pd.Series(t1_percentage2016, name='t1_percent2016')
s16=pd.Series(t1_percentage2017, name='t1_percent2017')
s17=pd.Series(t1_percentage2018, name='t1_percent2018')
s18=pd.Series(t5_value2014, name='t5_2014')
s19=pd.Series(t5_value2015, name='t5_2015')
s20=pd.Series(t5_value2016, name='t5_2016')
s21=pd.Series(t5_value2017, name='t5_2017')
s22=pd.Series(t5_value2018, name='t5_2018')
s23=pd.Series(t5_percentage2014, name='t5_percent2014')
s24=pd.Series(t5_percentage2015, name='t5_percent2015')
s25=pd.Series(t5_percentage2016, name='t5_percent2016')
s26=pd.Series(t5_percentage2017, name='t5_percent2017')
s27=pd.Series(t5_percentage2018, name='t5_percent2018')
s28=pd.Series(t10_value2014, name='t10_2014')
s29=pd.Series(t10_value2015, name='t10_2015')
s30=pd.Series(t10_value2016, name='t10_2016')
s31=pd.Series(t10_value2017, name='t10_2017')
s32=pd.Series(t10_value2018, name='t10_2018')
s33=pd.Series(t10_percentage2014, name='t10_percent2014')
s34=pd.Series(t10_percentage2015, name='t10_percent2015')
s35=pd.Series(t10_percentage2016, name='t10_percent2016')
s36=pd.Series(t10_percentage2017, name='t10_percent2017')
s37=pd.Series(t10_percentage2018, name='t10_percent2018')
s38=pd.Series(t25_value2014, name='t25_2014')
s39=pd.Series(t25_value2015, name='t25_2015')
s40=pd.Series(t25_value2016, name='t25_2016')
s41=pd.Series(t25_value2017, name='t25_2017')
s42=pd.Series(t25_value2018, name='t25_2018')
s43=pd.Series(t25_percentage2014, name='t25_percent2014')
s44=pd.Series(t25_percentage2015, name='t25_percent2015')
s45=pd.Series(t25_percentage2016, name='t25_percent2016')
s46=pd.Series(t25_percentage2017, name='t25_percent2017')
s47=pd.Series(t25_percentage2018, name='t25_percent2018')


DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16, s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,
             s28,s29,s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40, s41,s42,s43,s44,s45,s46,s47], axis=1)


DF.to_csv("THE_UNI_PublicationsInTopJournalPercentiles_ALL_3.csv", index=False)  # OK



#print(threshold)


# In[ ]:





# In[103]:


for line in data['University id'][:2]:
    url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=ScholarlyOutput&institutionIds={}&yearRange=5yrs&includeSelfCitations=false&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'
 #   print(url.format(line))
    resp = requests.get(url.format(line), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "a464321ef5063d696ada17f8c159a44c"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
    result=json.loads(parsed)
print(result['results'])


# In[66]:


with open("THE_UNI_ID_METRIC_TEST.json") as outputfile:
    out=json.load(outputfile)


# In[67]:


out


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


# In[11]:


import json


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


# In[26]:


from sklearn.cluster import KMeans


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



# In[25]:


cd "C:\Users\jchen148\THE Rankings\Json files"


# In[27]:


import json

with open("Test_THE_Country", 'w') as fd:
    fd.write(json.dumps(data_loaded, sort_keys=True, indent=4, separators=(',', ': ')))    # both dicts in a list here


# In[29]:


with open("Test_THE_Country", 'r') as fd:
    University_data=json.load(fd)


# In[13]:


# University SciVal institution id
print(u_id)


# In[14]:


# countryCode

print(country)

