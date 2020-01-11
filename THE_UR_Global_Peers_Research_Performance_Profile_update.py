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


# # Plot distribution of USA universities CitationCounts

# In[1112]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane"


# In[1113]:


citation = pd.read_csv('Updated_THE_Ranked_Universites_CitationCounts_2014_2018.csv')

citation.head()


# In[1114]:


totalcitation=citation['Citation2014']+citation['Citation2015']+citation['Citation2016']+citation['Citation2017']+citation['Citation2018']


# In[1115]:


citation['Total']=totalcitation


# In[1116]:


citation.head()

citation.info()


# In[1119]:


changedtype=lambda x: int(x)


# In[31]:


#citation.fillna(0)

for i in range(0,len(citation)):
    if citation.loc[i]['Citation2014'] is np.nan:
        print("yes")


# In[1117]:


citation['Citation2014'].isnull()

citation=citation.fillna(0)


# # change all citationcount to int64

# In[1120]:


citation['Citation2018']=citation['Citation2018'].apply(changedtype)


# In[1121]:


citation['Citation2017']=citation['Citation2017'].apply(changedtype)


# In[1122]:


citation['Citation2016']=citation['Citation2016'].apply(changedtype)


# In[1123]:


citation['Citation2015']=citation['Citation2015'].apply(changedtype)


# In[1124]:


citation['Citation2014']=citation['Citation2014'].apply(changedtype)


# In[1125]:


citation.info()


# In[1126]:


citation.head()


# In[1127]:


new=citation.sort_values(['CountryCode','Total'], ascending=False)
new.head()


# # Filtered the universities in USA

# In[1128]:


USdata=new[new['CountryCode']=='USA']


# In[1129]:


USdata.head()


# # Use seaborn

# In[1130]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[1131]:


sns.set(color_codes=True)


# In[1132]:


USpartial=USdata.loc[:][['UniversityName','Total']]


# In[1133]:


USpartial.head()

USpartial2=USpartial.reset_index()

USpartial2=USpartial2.iloc[:,1:]

USpartial2.head()


# In[1134]:


target=USpartial2[USpartial2['UniversityName']=='University of Rochester']

target.head()


# # Change datatype to int64

# In[1135]:


target.loc[:]['Total']=target['Total'].astype(int)


# In[1136]:


target.head()


# In[1137]:


USpartial2.head()

USpartial2.set_index('UniversityName')

USpartial2.loc[:]['Total']=USpartial2['Total'].astype(int)


# In[1138]:


USpartial2=USpartial2.set_index('UniversityName')


# In[1139]:


USpartial2.head()


# In[1140]:


target.head()

target.set_index('UniversityName')


# In[1141]:


target=target.set_index('UniversityName')


# In[1142]:


target.head()


# In[1143]:


len(USpartial2)


# # THE has 163 USA Universities ranking in top 300.
# # Below is the distribution plot of the total CitationCount
# # from 2014 to 2018.
# # And we can see where UofR lies.

# In[1144]:


import pandas as pd
fig, ax = plt.subplots(figsize=(8,10))
x = pd.Series(USpartial2['Total'], name="CitationCount Total")
ax = sns.distplot(x)

ax.set_xlabel("USA UniversityCitation Total",fontsize=16)
ax.set_ylabel("Probability",fontsize=16)
plt.axvline(254555, color='red') # this is where U of R
plt.axvline(np.mean(USpartial2['Total']), color='green') # this is the mean, 175882.56
plt.axvline(np.percentile(USpartial2['Total'], 25.0), color='blue') # Q1
plt.axvline(np.percentile(USpartial2['Total'], 75.0), color='orange') # Q3 
#plt.legend()
plt.tight_layout()


# # We can see it is a highly right-skewed distribution,
# # and the mean, which is the green line, and Q3,
# # which is the orange line are very close.
# # UofR has the CitationCounts much above Q3.

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


# # The following are data cleaning process,
# # and how to use Python Requests to retrieve
# # data from Scopus and SciVal REST APIs

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

# In[ ]:


for line in want_3:
    url= "https://api.elsevier.com/metrics/institution/search?query=name("+line+")"
#    print(url)


# # Combine all the Uids to retrieve data from APIs

# In[11]:


import requests
import json


# In[14]:


# add "Emory University" country code and university id

UniversityName=[]
Universityid=[]
Country=[]
CountryCode=[]

url='https://api.elsevier.com/analytics/scival/institution/search?query=name(Emory%20University)&limit=100&offset=0'

resp = requests.get(url, headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    with open("THE_UNI_ID_METRIC_ALL.json", 'w') as jsonfile:
#        json.dump(resp.json(), jsonfile)
#    print(parsed)
#    data.update(a_dict)
result=json.loads(parsed)
UniversityName.append(result['results'][0]['name'])
Universityid.append(result['results'][0]['id'])
Country.append(result['results'][0]['country'])
CountryCode.append(result['results'][0]['countryCode'])

ELmory=pd.DataFrame({'University Name':UniversityName, 'University id':Universityid, 'Country': Country,
                    'Country Code': CountryCode})


# In[34]:


filename='THE_CountryCode_Result_1202_{}'

for i in range(1,14):
    print(filename.format(i))


# In[47]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane"


# In[48]:


filename='THE_CountryCode_Result_1202_{}.csv'

chucks=[]

for i in range(1,14):
#    print(filename.format(i))
    chucks.append(pd.read_csv(filename.format(i)))

data=pd.concat(chucks, ignore_index=True)
    
data.head()


# In[49]:


len(data)


# In[50]:


del data['Unnamed: 0']


# In[26]:


ELmory


# In[51]:


data=pd.concat([data, ELmory]).drop_duplicates()


# In[52]:


data.head()


# In[31]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\Input Data"


# In[32]:


data.to_csv("THE_Universities_SciVal_Uids_1008.csv", index=False)


# # Use SciVal institution metrics API

# In[42]:


# https://api.elsevier.com/analytics/scival/institution/metrics


# In[53]:


data.tail()


# In[54]:


data.reset_index(inplace=True)


# In[57]:


data=data.iloc[:,1:]


# In[58]:


data.tail()


# In[64]:


for line in data['University id'][:2]:
    print(line)


# # ScholarlyOutput

# In[59]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ScholarlyOutput"


# In[60]:


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
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
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
DF.to_csv("THE_UNI_CitationCount_ALL_12.csv", index=False)


# # CitationCount, CitedPublications, FWCI, and PublicationinTopJournal Percentile

# In[122]:


# FWCI


# In[61]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\FNCI"


# In[63]:


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
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
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
DF.to_csv("THE_UNI_FWCI_12.csv", index=False)


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


# In[64]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[65]:


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
DF.to_csv("THE_UNI_CitedPublications_12.csv", index=False)


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


# In[66]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PubTopJournalPercentile"


# In[67]:


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


DF.to_csv("THE_UNI_PubPercentile_All_18.csv", index=False)  # OK



#print(threshold)


# # Combine all the subfiles and subset the USA universities

# # CitationCount

# In[68]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\CitationCount"


# In[69]:


filename='THE_UNI_CitationCount_ALL_{}.csv'


# In[72]:


chucks=[]

for i in range(1, 13):
    chucks.append(pd.read_csv(filename.format(i)))

cc_data=pd.concat(chucks, ignore_index=True)

cc_data.head()


# In[73]:


cc_data.tail()


# In[74]:


cc_data.to_csv('THE_ALLUNI_CC.csv', index=True)


# # FWCI

# In[75]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\FNCI"


# In[76]:


filename='THE_UNI_FWCI_{}.csv'


# In[77]:


chucks=[]

for i in range(1, 13):
    chucks.append(pd.read_csv(filename.format(i)))

fwci_data=pd.concat(chucks, ignore_index=True)

fwci_data.head()


# In[78]:


fwci_data.tail()


# In[79]:


fwci_data.to_csv("THE_ALLUNI_FWCI.csv", index=False)


# # PercPublsCited

# In[107]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[81]:


filename='THE_UNI_CitedPublications_{}.csv'


# In[82]:


chucks=[]

for i in range(1, 13):
    chucks.append(pd.read_csv(filename.format(i)))

cp_data=pd.concat(chucks, ignore_index=True)

cp_data.head()


# In[83]:


cp_data.tail()


# In[84]:


cp_data.to_csv("THEUNI_CITEDPUBLS.csv",index=False)


# # PubTopJournalPercentile

# In[85]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PubTopJournalPercentile"


# In[86]:


filename='THE_UNI_PubPercentile_All_{}.csv'


# In[87]:


chucks=[]

for i in range(1, 19):
    chucks.append(pd.read_csv(filename.format(i)))

pp_data=pd.concat(chucks, ignore_index=True)

pp_data.head()


# In[88]:


pp_data.tail()


# In[89]:


pp_data.to_csv("THE_ALLUNI_PP.csv", index=False)


# # ScholarlyOutput

# In[106]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data" # needs to use double quote


# In[94]:


data.tail()


# In[100]:


url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=ScholarlyOutput&institutionIds=508059&yearRange=5yrs&includeSelfCitations=true&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

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
result['results']


# In[108]:


country=[]
countryCode=[]
institution_id=[]
link=[]
institution_name=[]
metricType=[]
value2014=[]
value2015=[]
value2016=[]
value2017=[]
value2018=[]

url='https://api.elsevier.com/analytics/scival/institution/metrics?metricTypes=ScholarlyOutput&institutionIds=508059&yearRange=5yrs&includeSelfCitations=true&byYear=true&includedDocs=AllPublicationTypes&journalImpactType=CiteScore&showAsFieldWeighted=false'

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
result['results']
country.append(result['results'][0]['institution']['country'])
countryCode.append(result['results'][0]['institution']['countryCode'])
institution_id.append(result['results'][0]['institution']['id'])
link.append(result['results'][0]['institution']['link'])
institution_name.append(result['results'][0]['institution']['name'])
metricType.append(result['results'][0]['metrics'][0]['metricType'])
value2014.append(result['results'][0]['metrics'][0]['valueByYear']['2014'])
value2015.append(result['results'][0]['metrics'][0]['valueByYear']['2015'])
value2016.append(result['results'][0]['metrics'][0]['valueByYear']['2016'])
value2017.append(result['results'][0]['metrics'][0]['valueByYear']['2017'])
value2018.append(result['results'][0]['metrics'][0]['valueByYear']['2018'])

DF=pd.DataFrame({'country':country, 'countryCode': countryCode, 'institution_id': institution_id, 'link':link,
                'institution_name':institution_name, 'metricType':metricType,
                '2014': value2014, '2015': value2015, '2016': value2016, '2017':value2017, '2018':value2018})

DF.to_csv("THE_UNI_SCHOLAROUTPUT_ALL_15.csv", index=False)


# In[109]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ScholarlyOutput"


# In[110]:


filename='THE_UNI_SCHOLAROUTPUT_ALL_{}.csv'


# In[111]:


chucks=[]

for i in range(1, 16):
    chucks.append(pd.read_csv(filename.format(i)))

so_data=pd.concat(chucks, ignore_index=True)

so_data.head()


# In[112]:


so_data.to_csv("THE_ALLUNI_SO.csv", index=False)


# # USA University Publication Output

# # Total

# In[261]:


so_data.head()


# In[113]:


so_data[so_data.countryCode=='USA'].head()
so_data_USA=so_data[so_data.countryCode=='USA']


# In[114]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[115]:


so_data_USA=so_data_USA.iloc[:,-7:]


# In[116]:


so_data_USA.head()


# In[117]:


del so_data_USA['metricType']


# In[118]:


so_data_USA.head()


# In[119]:


so_data_USA=so_data_USA.set_index('institution_name')


# In[120]:


so_data_USA.agg('sum')


# # THE 163 USA Universities ranked before top 300,
# # The total ScholarlyOutput presents a Bell-shaped
# # Distribution.

# In[121]:


sns.distplot(so_data_USA.agg('sum'))


# In[122]:


len(so_data_USA) # 164 USA universities


# In[124]:


so_data_USA.agg('sum')


# In[125]:


so_data_USA=so_data_USA.reset_index()


# In[412]:


so_data_USA.info()


# In[391]:


sep_sum=lambda x: x.agg('sum')


# In[126]:


so_data_USA['Total']=so_data_USA.sum(axis=1)


# In[127]:


so_data_USA['Total']=so_data_USA.Total.astype(int)
so_data_USA.head()


# In[128]:


URpp=so_data_USA[so_data_USA.institution_name=='University of Rochester']


# In[129]:


URpp=URpp.reset_index()


# In[130]:


URpp['Total']=URpp.sum(axis=1)


# In[131]:


URpp


# # UofR's ScholarlyOutput from 2014 to 2018.
# # It seems more like a Bi-modal distribution.

# In[132]:


# UR Publs Distribution
inputdata=URpp[['2014','2015','2016','2017','2018']]
sns.distplot(inputdata)

# seems a bi-modal distribution but the overall trend is downward


# # Top 1% and top 10% highly cited publications 

# In[135]:


pp_data.tail()


# In[436]:


pp_data.head()


# In[136]:


USA_pp=pp_data[pp_data.countryCode=='USA']


# In[137]:


len(USA_pp)


# In[138]:


# we want t1 and t10 values

USA_pp.head()


# In[442]:


USA_pp.columns


# In[139]:


USA_pp=USA_pp.loc[:][['institution_name','t1_2014','t1_2015','t1_2016','t1_2017','t1_2018','t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[140]:


USA_pp=USA_pp.drop_duplicates()


# In[141]:


USA_pp=USA_pp.reset_index()


# In[142]:


USA_pp=USA_pp.iloc[:,1:]


# In[143]:


USA_pp.head()


# In[144]:


USA_pp.tail()


# In[145]:


USA_pp['2014_Total']=USA_pp.loc[:][['t1_2014','t10_2014']].sum(axis=1)


# In[146]:


USA_pp.head()


# In[147]:


USA_pp['2015_Total']=USA_pp.loc[:][['t1_2015','t10_2015']].sum(axis=1)
USA_pp['2016_Total']=USA_pp.loc[:][['t1_2016','t10_2016']].sum(axis=1)
USA_pp['2017_Total']=USA_pp.loc[:][['t1_2017','t10_2017']].sum(axis=1)
USA_pp['2018_Total']=USA_pp.loc[:][['t1_2018','t10_2018']].sum(axis=1)


# In[148]:


USA_pp.head()


# In[149]:


UR_percentile=USA_pp[USA_pp.institution_name=='University of Rochester']


# In[150]:


UR_percentile=UR_percentile.set_index('institution_name')


# In[151]:


UR_percentile


# In[152]:


basedata=UR_percentile[['2014_Total','2015_Total','2016_Total','2017_Total','2018_Total']]


# In[153]:


basedata


# In[154]:


smalldata=UR_percentile.iloc[:,:10]


# In[155]:


smalldata1=smalldata.loc[:][['t1_2014','t1_2015','t1_2016','t1_2017','t1_2018']]


# In[156]:


smalldata1


# In[157]:


smalldata2=smalldata.loc[:][['t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[158]:


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


# In[908]:


A=pd.DataFrame(data=[data_1[:5]], columns=['2014','2015','2016','2017','2018'])


# In[909]:


A


# In[911]:


A.reset_index(inplace=True, drop=True)


# In[912]:


B=pd.DataFrame(data=[data_2[:5]], columns=['2014','2015','2016','2017','2018'])


# In[914]:


B.reset_index(inplace=True, drop=True)


# In[915]:


B


# In[918]:


C=pd.DataFrame(data=[data_3[:5]], columns=['2014','2015','2016','2017','2018'])


# In[919]:


C.reset_index(inplace=True, drop=True)


# In[920]:


C


# In[166]:


def show_values_on_bars(axs, h_v="v", space=0.8):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center", color='red') 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left", color='black')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# In[1024]:


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
g=sns.barplot(data=A,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("dark")
g=sns.barplot(data=B,
            label="Top 10%", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
g=sns.barplot(data=C,
            label="Top 1%", color="g")

show_values_on_bars(g, "v", 0.8)

# Add a legend and informative axis label
plt.yticks(np.arange(0, 4000, step=500))
plt.xticks(np.arange(5), ('2014', '2015', '2016', '2017', '2018'))
ax.legend(ncol=3, loc="upper right", frameon=True)
ax.set(xlim=(0,5), ylabel="",
       title="U of R publication output: total, top 1 % and top 10 % highly cited publs")
sns.despine(left=True, bottom=True)


# # From 2014-2018 ,our top 1% cited publs 
# # and top10% cited pulbs slightly dropped.
# # However, since our 2018 total publs increased a lot, 
# # it would definitely influence our overall research performance.

# # Trends in FWCI values of total U of R publication output

# In[539]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\FNCI"


# In[540]:


FWCI_all=pd.read_csv('THE_ALLUNI_FWCI.csv')


# In[541]:


FWCI_all.head()


# In[1064]:


UR_FWCI=FWCI_all[FWCI_all.institution_name=='University of Rochester']


# In[1065]:


UR_FWCI=UR_FWCI.iloc[:, -7:]


# In[1066]:


del UR_FWCI['metricType']


# In[1067]:


UR_FWCI


# In[1043]:


UR_FWCI.reset_index(inplace=True, drop=True)


# In[1068]:


UR_FWCI.set_index('institution_name', inplace=True, drop=True)


# In[1083]:


UR_FWCI.reset_index(inplace=True)


# In[1084]:


UR_FWCI


# # UofR FWCI

# In[69]:


def show_values_on_bars_1(axs, h_v="v", space=0.4):
    def _show_on_single_plot_1(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = round(p.get_height(),2)
                ax.text(_x, _y, value, ha="center", color='red') 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = round(p.get_width(),2)
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot_1(ax)
    else:
        _show_on_single_plot_1(axs)


# # UofR FWCI have always been above global average which is 1.00

# In[1094]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
sns.set_style("ticks", {"xtick.major.size": 10, "ytick.major.size": 8})

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(8, 10))
g= sns.barplot(data=UR_FWCI)
plt.axhline(1.00, ls='-', color='r')
plt.title('UofR FWCI 2014-2018 with World Average')
plt.xlabel("UofR FWCI 2014-2018")
#plt.ylabel("Filed-weighted Cited Index")
show_values_on_bars_1(g, 'v', 0.3)


# # Comparator analysis: top 10 % highly cited publications for USA universities

# In[574]:


USA_pp.head()


# In[159]:


UR_peer=['Boston University','Carnegie Mellon University','Case Western Reserve University','Duke University','Emory University',
        'Northwestern University','Vanderbilt University','Washington University','Johns Hopkins University','New York University',
        'Stanford University','Tulane University','University of Chicago','University of Pennsylvania','University of Southern California']


# In[3]:


import pandas as pd


# In[6]:


UR_peer_df=pd.DataFrame({'UR_Peer':UR_peer})


# In[8]:


UR_peer_df=UR_peer_df.iloc[:14,:]


# In[9]:


UR_peer_df['UR_Peer']


# # Get UofR's Global set's Publication in Top Journal Percentile

# In[160]:


chuck=[]
for name in UR_peer_df['UR_Peer']: 
    chuck.append(USA_pp[USA_pp.institution_name==name])


# In[161]:


DF=pd.concat(chuck, ignore_index=True)


# In[162]:


DF.head()


# In[178]:


UR_percentile=USA_pp[USA_pp.institution_name=='University of Rochester']


# In[179]:


UR_percentile=UR_percentile.reset_index()


# In[172]:


Global_top10=DF.loc[:][['institution_name','t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[180]:


UR_pcer_top10=UR_percentile.loc[:][['institution_name','t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[174]:


Global_top10.head()


# In[175]:


Global_top10['Top10_Total']=Global_top10.sum(axis=1)


# In[176]:


Global_top10.head()


# In[169]:


len(Global_top10)


# In[181]:


UR_pcer_top10


# In[182]:


UR_pcer_top10['Top10_Total']=UR_pcer_top10.sum(axis=1)


# In[183]:


UR_pcer_top10


# In[184]:


Gall=pd.concat([Global_top10, UR_pcer_top10])


# In[185]:


len(Gall)


# In[647]:


import re


# In[186]:


abb=[]
for i in Gall.institution_name:
    abb.append(i.split("\t")[0].strip(" "))
abb # not work


# In[187]:


Gall['UniAbbr']=['Boston','CWRU','Duke','Emory','Northwestern','Vanderbilt','JohnsHopkins','NYU','Stanford','Tulane','UofChicago','UofPenn','UofR']


# In[188]:


Gall=Gall.sort_values(by='Top10_Total', ascending=False)


# # Comparator analysis: top 10% highly cited publications UR and GlobalPeers

# In[192]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))
g=sns.barplot(x=Gall.UniAbbr, y=Gall.Top10_Total, data=Gall)
plt.axhline(6649, ls='-', color='r') # can add a red base line for UofR value

#ax.text(Gall.UniAbbr, Gall.Top10_Total,color='black', ha="center")

# Add a legend and informative axis label
#ax.legend(ncol=12, loc="upper right", frameon=True)
ax.set(xlim=(0, 15),
       xlabel="University of Rochester and Global Peers", ylabel="Top 10% highly cited publications")
sns.despine(left=True, bottom=True)

show_values_on_bars(g,'v', 0.5)


# # Among our other 12 USA peers,
# # our top 10% highly-cited pulbs is relatively fewer

# # Comparator analysis: Field-weighted Citation Impact

# In[193]:


fwci_data.head()


# In[194]:


fwci_data.tail()


# In[195]:


US_fwci=fwci_data[fwci_data.countryCode=='USA']


# In[196]:


US_fwci.head()


# In[200]:


UR_peer_df


# In[212]:


UR=pd.DataFrame({'UR_Peer':['University of Rochester']})


# In[214]:


UR_peer_df=pd.concat([UR_peer_df, UR])


# In[215]:


UR_peer_df.reset_index(inplace=True)


# In[198]:


len(Gall.institution_name) # Global peers and UofR


# In[216]:


chuck=[]

for name in UR_peer_df.UR_Peer:
    if US_fwci[US_fwci.institution_name==name] is not None:
        chuck.append(US_fwci[US_fwci.institution_name==name])


# In[217]:


UR_Peer_FWCI=pd.concat(chuck, ignore_index=True)


# In[24]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\FNCI"


# In[218]:


UR_Peer_FWCI.to_csv('UR_Global_Peer_FWCI_Comparison.csv', index=False)


# In[29]:


ALL_FWCI=pd.read_csv('THE_ALLUNI_FWCI.csv')


# In[30]:


ALL_FWCI=ALL_FWCI.drop_duplicates()


# In[31]:


ALL_FWCI.head()


# In[219]:


UR_Peer_FWCI=UR_Peer_FWCI.iloc[:, -7:]


# In[222]:


UR_Peer_FWCI=UR_Peer_FWCI.drop_duplicates()


# In[223]:


abb=[]
for name in Gall.UniAbbr:
    abb.append(name)
abb


# In[764]:


UR_Peer_FWCI=UR_Peer_FWCI.drop_duplicates()


# In[224]:


UR_Peer_FWCI.reset_index(inplace=True, drop=True)


# In[225]:


UR_Peer_FWCI


# In[230]:


UR_Peer_FWCI['UniAbbr']=['Boston','CWRU','Duke','Emory','Northwestern','Vanderbilt','JHopkins','NYU','Stanford','Tulane',
                               'UofC','UofPenn','UofR']


# In[231]:


UR_Peer_FWCI.head()


# In[234]:


UR_Peer_FWCI['AVERAGE_FWCI']=round(UR_Peer_FWCI[['2014','2015','2016','2017','2018']].mean(axis=1), 4)


# In[235]:


UR_Peer_FWCI=UR_Peer_FWCI.sort_values(by='AVERAGE_FWCI', ascending=False)


# In[236]:


UR_Peer_FWCI.head()


# In[237]:


UR_Peer_FWCI[UR_Peer_FWCI.UniAbbr=='UofR']


# # Comparatory analysis: Field-weighted Citation Impact

# # Our average FWCI 2014-2018 is 1.8,
# # but most of our USA peers have higher FWCI,
# # this may explain why our overall score did not reflect our good FWCI

# In[238]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))
g=sns.barplot(x=UR_Peer_FWCI.UniAbbr, y=UR_Peer_FWCI.AVERAGE_FWCI, data=UR_Peer_FWCI)
plt.axhline(1.802, ls='-', color='r')

show_values_on_bars_1(g , 'v' , 0.5)

#ax.text(Gall.UniAbbr, Gall.Top10_Total,color='black', ha="center")

# Add a legend and informative axis label
#ax.legend(ncol=12, loc="upper right", frameon=True)
plt.yticks(np.arange(0, 2.5, step=0.2))
ax.set(xlim=(0, 15),
       xlabel="University of Rochester and Global Peers", ylabel="Field-weighted Citation Impact")
sns.despine(left=True, bottom=True)


# # Comparatory analysis:
# # top publication output and highly cited publications

# In[19]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PubTopJournalPercentile"


# In[2]:


import pandas as pd
import numpy as np


# In[20]:


A_PP=pd.read_csv('THE_ALLUNI_PP.csv')


# In[21]:


USA_PP=A_PP[A_PP.countryCode=='USA']


# In[22]:


USA_PP=USA_PP.drop_duplicates()


# In[32]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\research_performance_Profile"


# In[33]:


UR_peer=pd.read_csv('UR_GloPeers_Research_Performance_Profile.csv')


# In[34]:


chuck=[]
for name in UR_peer.institution_name:
    chuck.append(USA_PP[USA_PP.institution_name==name])


# In[35]:


UR_Peers_PP=pd.concat(chuck, ignore_index=True)


# In[36]:


UR_Peers_PP


# In[38]:


UR_Peers_top1=UR_Peers_PP.loc[:][['institution_name','t1_2014','t1_2015','t1_2016','t1_2017','t1_2018']]


# In[39]:


UR_Peers_top10=UR_Peers_PP.loc[:][['institution_name','t10_2014','t10_2015','t10_2016','t10_2017','t10_2018']]


# In[40]:


UR_Peers_top1['top1_all']=UR_Peers_top1.sum(axis=1)


# In[41]:


UR_Peers_top10['top10_all']=UR_Peers_top10.sum(axis=1)


# In[42]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ScholarlyOutput"


# In[43]:


ALL_so=pd.read_csv('THE_ALLUNI_SO.csv')


# In[45]:


USA_so=ALL_so[ALL_so.countryCode=='USA']


# In[49]:


chuck=[]
for name in UR_Peers_PP.institution_name:
    chuck.append(USA_so[USA_so.institution_name==name])
UR_Peers_SO=pd.concat(chuck, ignore_index=True)    


# In[52]:


UR_Peers_SO=UR_Peers_SO.drop_duplicates()


# In[53]:


UR_Peers_SO['Total_PUBLS']=UR_Peers_SO.sum(axis=1)


# In[60]:


O=UR_Peers_SO.loc[:][['institution_name','Total_PUBLS']]


# In[61]:


P=UR_Peers_top1.loc[:][['institution_name','top1_all']]


# In[62]:


Q=UR_Peers_top10.loc[:][['institution_name','top10_all']]


# In[63]:


part1=O.join(P.set_index('institution_name'), on='institution_name')


# In[64]:


part2=part1.join(Q.set_index('institution_name'), on='institution_name')


# In[65]:


part2


# In[66]:


part2['remaining_90%']=part2.Total_PUBLS-part2.top1_all-part2.top10_all


# In[67]:


part2


# In[70]:


part2['Abbr']=['Stanford','JohnsHopkins','UofPenn','NYU','Duke','UofChicago','Northwestern','Vanderbilt','Emory','Boston','CWRU','UofR','Tulane']


# In[71]:


part2


# In[84]:


A=part2[['Abbr','remaining_90%']]


# In[95]:


A.reset_index(inplace=True)


# In[86]:


B=part2[['Abbr','top10_all']]


# In[99]:


B


# In[87]:


C=part2[['Abbr','top1_all']]


# In[101]:


len(C.Abbr)


# In[172]:


def show_values_on_bars(axs, h_v="v", space=0.8):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center", color='red') 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left", color='black')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# # Compare to our USA peers, our top 1% and top 10% highly cited 
# # publications is relatively low

# In[174]:


# Plot the crashes where alcohol was involved
# Plot the crashes where alcohol was involved
sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 6))
sns.set_color_codes()
g=sns.barplot(x='top10_all',y='Abbr',data=B,
            label="Top 10%", color="yellow")
sns.set_color_codes("muted")
g=sns.barplot(x='top1_all', y='Abbr', data=C,
            label="Top 1%", color="orange")

plt.axvline(6649, ls='-', color='r')

ax.legend(ncol=2, loc="lower right", frameon=True)
#plt.xticks(np.arange(0,3000, step=100))
ax.set(xlabel='Number of Research publications', ylabel="University Abbreviation",
       title="UofR and Peers total publication output and highly cited publications")
sns.despine(left=True, bottom=True)

show_values_on_bars(g, "h", 2)


# # Use all THE University ids to get Topic Cluster ids

# In[175]:


url='https://api.elsevier.com/analytics/scival/topic/metrics/institutionId/{}?topicId={}&metricTypes=ScholarlyOutput&yearRange=5yrs'


# In[176]:


import requests
import json
import pandas as pd
import numpy as np


# In[177]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[178]:


ALL_Uids=pd.read_csv("THEUNI_CITEDPUBLS.csv")


# In[182]:


ALL_Uids=ALL_Uids.iloc[:,:3]


# In[184]:


USA_Uids=ALL_Uids[ALL_Uids.countryCode=='USA']


# In[186]:


USA_Uids.reset_index(inplace=True, drop=True)


# In[189]:


USA_Uids.head() # these are the USA university ids we'll use in API
len(USA_Uids) # 165 universities


# In[191]:


inst_ids=USA_Uids


# In[ ]:


# get topic ids


# In[ ]:


# THE


# In[216]:


url='https://api.elsevier.com/analytics/scival/subjectArea/classificationType/THE?'


# In[197]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data"


# In[198]:


import time
from time import sleep


# In[214]:


time.sleep(1)

subjectAreas_name=[]
subjectAreas_id=[]
subjectAreas_uri=[]
classificationType=[]
classificationName=[]


resp = requests.get(url, headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
result=json.loads(parsed)
result["subjectAreas"][0]['children'][0]


# In[204]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\THE Code"


# In[235]:


subjectAreas_name=[]
subjectAreas_id=[]
subjectAreas_uri=[]
classificationType=[]
classificationName=[]

resp = requests.get(url, headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
result=json.loads(parsed)
#if 'children' in result['subjectAreas'][0]:
#    if len(result['subjectAreas']) >=1:
#        if "name" in result["subjectAreas"][0]['children'][0]:
#    subjectAreas_name.append(result["subjectAreas"][0]['children'][0]["name"])
#subjectAreas_name
#result
result


# In[244]:


subjectAreas_name=[]
subjectAreas_id=[]
subjectAreas_uri=[]
classificationType=[]
classificationName=[]

resp = requests.get(url, headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
result=json.loads(parsed)
DF=pd.DataFrame.from_dict(result['subjectAreas'])
DF.to_csv("THE_Classification_Code.csv", index=False)


# In[192]:


# ASJC


# In[259]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ASJC Code"


# In[247]:


url='https://api.elsevier.com/analytics/scival/subjectArea/classificationType/ASJC?'


# In[268]:


classificationType=[]
classification_id=[]
link=[]
name=[]
uri=[]

resp = requests.get(url, headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
result=json.loads(parsed)
with open("ASJC.json", "w") as json_file:
    json.dump(resp.json(), json_file)


# In[257]:


resp = requests.get(url, headers={'Accept':'application/json',
                             'X-ELS-APIKey': "d3794058e2b24417b5dfd0ef8990e2dc"})
parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
result=json.loads(parsed)
result['subjectAreas'][0]['children'][0]


# In[261]:


# ASJC code result from earlier data


# In[265]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ASJC Code"


# In[271]:


with open("ASJC.json") as output:
    data=json.load(output)


# In[277]:


name=[]
Acode=[]
uri=[]
classificationType=[]

for i in range(0,len(data['subjectAreas'])):
    name.append(data['subjectAreas'][i]['children'][0]['name'])
    Acode.append(data['subjectAreas'][i]['children'][0]['id'])
    uri.append(data['subjectAreas'][i]['children'][0]['uri'])
    classificationType.append(data['subjectAreas'][i]['children'][0]['classificationType'])
DF=pd.DataFrame({'name':name,
                'ASJC_Code': Acode,
                'uri':uri,
                'classificationType':classificationType})
DF.to_csv("NEW_ASJC.csv", index=False)


# In[339]:


url='https://api.elsevier.com/analytics/scival/topic/metrics/institutionId/508335?topicId={}&metricTypes=ScholarlyOutput&yearRange=5yrs'


# In[281]:


for item in inst_ids.institution_id[:2]: # test
    print(item)


# In[349]:


pwd


# In[351]:


with open("ASJC.json") as output:
    topic_id=json.load(output)


# In[354]:


ASJC_Code=pd.read_csv("NEW_ASJC.csv")


# In[355]:


ASJC_Code.head()


# In[368]:


topic_id=ASJC_Code['ASJC_Code']
topic_id[:2]


# In[285]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ASJC Code"


# In[328]:


url='https://api.elsevier.com/analytics/scival/topic/metrics/institutionId/{}?&metricTypes=ScholarlyOutput&yearRange=5yrs'


# In[369]:


for item in inst_ids.institution_id[20:22]:
    print(item)


# In[357]:


url


# In[373]:


inst_ids


# In[374]:


UR_ins_id='508335'


# In[378]:


url='https://api.elsevier.com/analytics/scival/topic/metrics/institutionId/508335?topicId={}&metricTypes=ScholarlyOutput&yearRange=5yrs'


# In[372]:


for item in inst_ids.institution_id[:2]:
    for tid in topic_id[:2]:
        print(url.format(item, tid))


# In[380]:


for tid in topic_id[:10]:
    print(tid)


# In[382]:


for tid in topic_id[:10]:
    print(url.format(tid))


# In[385]:


pwd


# In[389]:


for i in topic_id[:20]:
    print(i)


# In[427]:


source=[]
endyear=[]
startyear=[]
name=[]

for tid in topic_id[:2]:
    resp = requests.get(url.format(tid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result=json.loads(parsed)
result


# In[428]:


source=[]
endyear=[]
startyear=[]
name=[]

for tid in topic_id[:2]:
    resp = requests.get(url.format(tid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result=json.loads(parsed)
#result
    source.append(result['dataSource'])
    endyear.append(result['dataSource']['metricEndYear'])
    startyear.append(result['dataSource']['metricStartYear'])
    name.append(result['dataSource']['sourceName'])

Data_title=pd.DataFrame({'dataSource': source,
                        'metricEndYear':endyear,
                        'metricStartYear':startyear,
                        'sourceName':name})

Data_title.to_csv("ASJC_Data_Title.csv", index=False)


# In[396]:


td=[]
metricType=[]
metricvalue=[]
link=[]
name=[]
ACode=[]
uri=[]
prominencePercentile=[]
scholarlyOutput=[]
overallScholarlyOutput=[]


for tid in topic_id[20:]:
    resp = requests.get(url.format(tid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result=json.loads(parsed)
    with open("THE_UNI_ASJC_after20.json", "w") as json_file:
         json.dump(resp.json(), json_file)
#result
    td.append(tid)
    if result['results'] is None:
        metricType.append('')
        metricvalue.append('')
        link.append('')
        name.append('')
        ACode.append('')
        uri.append('')
        prominencePercentile.append('')
        scholarlyOutput.append('')
        overallScholarlyOutput.append('')
    else:
        if len(result['results']) >=1:
            if "metrics" in result['results'][0]:
                metricType.append(result['results'][0]["metrics"][0]["metricType"])
                metricvalue.append(result['results'][0]["metrics"][0]["value"])
            if "topic" in result['results'][0]:
                link.append(result['results'][0]["topic"]["link"])
                name.append(result['results'][0]["topic"]["name"])
                Acode.append(result['results'][0]["topic"]["id"])
                uri.append(result['results'][0]["topic"]["uri"])
                prominencePercentile.append(result['results'][0]["topic"]["prominencePercentile"])
                scholarlyOutput.append(result['results'][0]["topic"]["scholarlyOutput"])
                overallScholarlyOutput.append(result['results'][0]["topic"]["overallScholarlyOutput"])
                
s1=pd.Series(td, name='Topic_ID')
s2=pd.Series(metricType, name='metricType')
s3=pd.Series(metricvalue, name='metricvalue')
s4=pd.Series(link, name='link')
s5=pd.Series(name, name='name')
s6=pd.Series(Acode, name='ACode')
s7=pd.Series(uri, name='uri')
s8=pd.Series(prominencePercentile, name='prominencePercentile')
s9=pd.Series(scholarlyOutput, name='scholarlyOutput')
s10=pd.Series(overallScholarlyOutput, name='overallScholarlyOutput')

ASJC_20=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10], axis=1)

ASJC_20.to_csv("ASJC_after20.csv", index=False)

#DF=pd.DataFrame({'Topic_ID':td,
#                 'metricType':metricType,
#                 'metricvalue':metricvalue,
#                 'link':link,
#                 'name':name,
#                 'ACode':ACode,
#                 'uri':uri,
#                 'prominencePercentile':prominencePercentile,
#                 'scholarlyOutput':scholarlyOutput,
#                 'overallScholarlyOutput':overallScholarlyOutput
#                })

#DF.to_csv("ASJC_TID20.csv", index=False)

#with open("THE_UNI_ASJC_after20.json", "w") as json_file:
#    json.dump(resp.json(), json_file)


# In[398]:


UR_ASJC_1=pd.read_csv("ASJC_20.csv")


# In[399]:


UR_ASJC_1.head()


# In[400]:


UR_ASJC_2=pd.read_csv("ASJC_after20.csv")


# In[401]:


UR_ASJC_2.head()


# In[403]:


del UR_ASJC_1['Topic_ID']


# In[405]:


del UR_ASJC_1['ACode']


# In[406]:


del UR_ASJC_2['Topic_ID']


# In[407]:


del UR_ASJC_2['ACode']


# In[409]:


UR_ASJC_1=UR_ASJC_1.dropna()


# In[410]:


UR_ASJC_2=UR_ASJC_2.dropna()


# In[412]:


UR_ASJC=pd.concat([UR_ASJC_1, UR_ASJC_2])


# In[413]:


UR_ASJC.head()


# In[419]:


chuck=[]
for line in UR_ASJC.uri:
    chuck.append(str(line).split('/')[1])
UR_ASJC['ASJC_Code']=chuck


# In[420]:


UR_ASJC.head()


# In[421]:


UR_ASJC=UR_ASJC.loc[:][['ASJC_Code','metricType','name','link','uri','prominencePercentile','scholarlyOutput','overallScholarlyOutput']]


# In[422]:


UR_ASJC


# In[423]:


UR_ASJC.to_csv("UR_ASJC_0110.csv", index=False)


# In[1024]:


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
g=sns.barplot(data=A,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("dark")
g=sns.barplot(data=B,
            label="Top 10%", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
g=sns.barplot(data=C,
            label="Top 1%", color="g")

show_values_on_bars(g, "v", 0.8)

# Add a legend and informative axis label
plt.yticks(np.arange(0, 4000, step=500))
plt.xticks(np.arange(5), ('2014', '2015', '2016', '2017', '2018'))
ax.legend(ncol=3, loc="upper right", frameon=True)
ax.set(xlim=(0,5), ylabel="",
       title="U of R publication output: total, top 1 % and top 10 % highly cited publs")
sns.despine(left=True, bottom=True)


# In[329]:



for item in inst_ids.institution_id[20:22]:
#    for tid in topic_id[20:100]:
    resp = requests.get(url.format(item), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
    parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
    result=json.loads(parsed)
result


# In[298]:


time.sleep(2)
for item in inst_ids.institution_id[20:100]:
    for tid in topic_id[20:100]:
        resp = requests.get(url.format(item, tid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
        parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
        result=json.loads(parsed)
    with open("THE_UNI_Versus_ASJC_3.json",'w') as json_file:
        json.dump(resp.json(), json_file)


# In[303]:


with open("THE_UNI_Versus_ASJC.json") as output:
    data=json.load(output)


# In[330]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ASJC Code"


# In[341]:


url


# In[347]:





# In[346]:


for item in inst_ids.institution_id[:5]:
    for tid in topic_id:


# In[336]:


inst_id=[]
topic_id=[]
metrics=[]
metrics_value=[]
link=[]
name=[]
Acode=[]
uri=[]
prominencePercentile=[]
scholarlyOutput=[]
overallScholarlyOutput=[]

for item in inst_ids.institution_id[:5]:
    for tid in topic_id[:5]:
        resp = requests.get(url.format(item, tid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
        parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
        data=json.loads(parsed)
data


# In[332]:


inst_id=[]
topic_id=[]
metrics=[]
metrics_value=[]
link=[]
name=[]
Acode=[]
uri=[]
prominencePercentile=[]
scholarlyOutput=[]
overallScholarlyOutput=[]

for item in inst_ids.institution_id[:5]:
    for tid in topic_id[:5]:
        resp = requests.get(url.format(item, tid), headers={'Accept':'application/json',
                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})
        parsed=json.dumps(resp.json(),
                 sort_keys=True,
                 indent=4, separators=(',', ': '))
#    print(parsed)
        data=json.loads(parsed)
        inst_id.append(item)
        topic_id.append(tid)
        link.append(data['link'])
        if len(data['results']) >=1:
            if 'metrics' in data['results'][0]:
                if len(data['results'][0]['metrics']) >=1:
                    metrics.append(data['results'][0]['metrics'][0]['metricType'])
                    metrics_value.append(data['results'][0]['metrics'][0]['value'])
            if 'topic' in data['results'][0]:
                if len(data['results'][0]['topic']) >=1:
                    name.append(data['results'][0]['topic']['name'])
                    Acode.append(data['results'][0]['topic']['id'])
                    uri.append(data['results'][0]['topic']['uri'])
                    prominencePercentile.append(data['results'][0]['topic']['prominencePercentile'])
                    scholarlyOutput.append(data['results'][0]['topic']['scholarlyOutput'])
                    overallScholarlyOutput.append(data['results'][0]['topic']['scholarlyOutput'])  
                    
s1=pd.Series(inst_id, name="inst_id")
s2=pd.Series(topic_id, name="topic_id")
s3=pd.Series(metrics, name="metrics")
s4=pd.Series(metrics_value, name="metrics_value")
s5=pd.Series(link, name="link")
s6=pd.Series(name, name="name")
s7=pd.Series(Acode, name="Acode")
s8=pd.Series(uri, name="uri")
s9=pd.Series(prominencePercentile, name="prominencePercentile")
s10=pd.Series(scholarlyOutput, name="scholarlyOutput")
s11=pd.Series(overallScholarlyOutput, name="overallScholarlyOutput")
        
DF=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11], axis=1)
DF.to_csv("TEST.csv", index=False)
#with open("THE_UNI_Versus_ASJC_Test.json", 'w') as json_file:
#    json.dump(resp.json(), json_file)


# In[325]:


inst_id=[]
topic_id=[]
metrics=[]
metrics_value=[]
link=[]
name=[]
Acode=[]
uri=[]
prominencePercentile=[]
scholarlyOutput=[]
overallScholarlyOutput=[]

data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Comparatory analysis: research performance profile

# In[239]:


UR_Peer_FWCI.institution_name


# In[240]:


so_data_USA.head()


# In[800]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(so_data_USA[so_data_USA.institution_name==name])


# In[241]:


Ttl_publs_output=pd.concat(chuck, ignore_index=True)


# In[243]:


Ttl_publs_output.head()


# In[244]:


Ttl_publs_output['Total']=Ttl_publs_output.sum(axis=1)


# In[245]:


A=Ttl_publs_output[['institution_name','Total']]


# In[246]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[247]:


ALL_PP=pd.read_csv("THEUNI_CITEDPUBLS.csv")


# In[248]:


ALL_PP.head()


# In[249]:


ALL_PP.tail()


# In[250]:


US_PP=ALL_PP[ALL_PP.countryCode=='USA']


# In[251]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(US_PP[US_PP.institution_name==name])


# In[252]:


UR_Peer_PP=pd.concat(chuck, ignore_index=True)


# In[253]:


UR_Peer_PP=UR_Peer_PP[['institution_name','percent2014','percent2015','percent2016','percent2017','percent2018']]


# In[254]:


UR_Peer_PP=UR_Peer_PP.drop_duplicates()


# In[255]:


UR_Peer_PP.shape[0]


# In[256]:


UR_Peer_PP.loc[:]['UniAbbr']=abb


# In[257]:


UR_Peer_PP.loc[:]['Mean_%PubCited']=UR_Peer_PP.iloc[:,1:5].mean(axis=1)


# In[258]:


UR_Peer_PP


# In[259]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PercPublsCited"


# In[261]:


UR_Peer_PP['Mean_%PubCited']=UR_Peer_PP.mean(axis=1)


# In[262]:


UR_Peer_PP=UR_Peer_PP.sort_values(by='Mean_%PubCited', ascending=False)


# In[263]:


UR_Peer_PP.reset_index(inplace=True, drop=True)


# In[264]:


C=UR_Peer_PP[['institution_name','Mean_%PubCited']]


# In[265]:


UR_Peer_PP.to_csv("UofR_Global_Peers_Cited_Publs.csv", index=False)


# In[836]:


# Top 1 % cited


# In[266]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\PubTopJournalPercentile"


# In[267]:


Top1All=pd.read_csv("THE_ALLUNI_PP.csv")


# In[268]:


Top1All.columns


# In[269]:


Top1All.tail()


# In[270]:


Top1=Top1All[['institution_name','t1_percent2014','t1_percent2015','t1_percent2016','t1_percent2017','t1_percent2018']]


# In[271]:


Top1=Top1.drop_duplicates()


# In[272]:


Top1['Total_Top1']=Top1[['institution_name','t1_percent2014','t1_percent2015','t1_percent2016','t1_percent2017','t1_percent2018']].mean(axis=1)


# In[273]:


Top1.head()


# In[274]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(Top1[Top1.institution_name==name])


# In[275]:


UR_PEER_Top1=pd.concat(chuck, ignore_index=True)


# In[276]:


UR_PEER_Top1=UR_PEER_Top1.sort_values(by='Total_Top1', ascending=False)


# In[277]:


UR_PEER_Top1.reset_index(inplace=True, drop=True)


# In[278]:


D=UR_PEER_Top1[['institution_name','Total_Top1']] # top1%


# In[279]:


# top 10%

Top10=Top1All[['institution_name','t10_percent2014','t10_percent2015','t10_percent2016','t10_percent2017','t10_percent2018']]


# In[280]:


Top10=Top10.drop_duplicates()


# In[281]:


Top10['Total_Top10']=Top10[['institution_name','t10_percent2014','t10_percent2015','t10_percent2016','t10_percent2017','t10_percent2018']].mean(axis=1)


# In[282]:


Top10.head()


# In[283]:


Top10.tail()


# In[284]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(Top10[Top10.institution_name==name])


# In[285]:


UR_PEER_Top10=pd.concat(chuck, ignore_index=True)


# In[286]:


UR_PEER_Top10=UR_PEER_Top10.sort_values(by='Total_Top10', ascending=False)


# In[287]:


UR_PEER_Top10.reset_index(inplace=True, drop=True)


# In[288]:


E=UR_PEER_Top10[['institution_name','Total_Top10']]


# In[301]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\ScholarlyOutput"


# In[303]:


# filter publication data for UR global peeer


# In[305]:


so_data_USA.head()


# In[306]:


chuck=[]

for name in UR_Peer_FWCI.institution_name:
    chuck.append(so_data_USA[so_data_USA.institution_name==name])


# In[307]:


A=pd.concat(chuck, ignore_index=True)


# In[308]:


A=A.drop_duplicates()


# In[310]:


A=A.sort_values(by='Total', ascending=False)


# In[311]:


A.head()


# In[319]:


A=A.iloc[:,[0,-1]]


# In[320]:


A.reset_index(inplace=True, drop=True)


# In[321]:


A.head()


# In[313]:


len(A)


# In[312]:


C.head()


# In[314]:


len(C)


# In[315]:


D.head()


# In[317]:


len(E)


# In[322]:


part1=A.join(C.set_index('institution_name'), on='institution_name')


# In[323]:


part2=part1.join(D.set_index('institution_name'), on='institution_name')


# In[324]:


part3=part2.join(E.set_index('institution_name'), on='institution_name')


# In[325]:


part3


# In[326]:


B=UR_Peer_FWCI[['institution_name','AVERAGE_FWCI']]


# In[327]:


B.head()


# In[328]:


part4=part3.join(B.set_index('institution_name'), on='institution_name')


# In[329]:


part4.sort_values(by='Total', ascending=False)


# In[330]:


cd "C:\Users\jchen148\THE Rankings\Report to Jane\OK Files\OUtput Data\research_performance_Profile"


# In[331]:


part4.to_csv('UR_GloPeers_Research_Performance_Profile.csv', index=False)


# # THE USA 163 Universities Distribution Plot 

# # From the distribution plot below, 
# # we can see we are above 75% of the other USA Universities 
# # in publications from 2014 to 2018.

# # However, we can see Q3 is very close to the mean, 
# # which is the green line. This is a right-skewed distribution.

# In[1110]:


import pandas as pd
fig, ax = plt.subplots(figsize=(10,8))
x = pd.Series(so_data_USA['Total'], name="USA Universities Publs") # 163 universities
ax = sns.distplot(x)

ax.set_xlabel("USA 163 Universities Publs",fontsize=16)
ax.set_ylabel("Probability",fontsize=16)
plt.axvline(18132, color='red') # this is where U of R
plt.axvline(np.mean(so_data_USA['Total']), color='green') # this is the mean, 175882.56
plt.axvline(np.percentile(so_data_USA['Total'], 25.0), color='blue') # Q1
plt.axvline(np.percentile(so_data_USA['Total'], 75.0), color='orange') # Q3
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
    


# In[126]:


for line in want_3[38:40]:
    print(re.sub('[^A-Za-z0-9]+',' ', line))


# In[ ]:


for line in want_3[38:40]:
    line=re.sub('[^A-Za-z0-9]+',' ', line)
#    query = "name(school)"
    url= """https://api.elsevier.com/metrics/institution/search?query=name("{}")&start=0&count=25&limit=25&cursor=*"""
#    resp = requests.get(url.format(line), headers={'Accept':'application/json',
#                             'X-ELS-APIKey': "dcfb521197bf15867d12c3c86c46c69b"})


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
            
            


# # concatenate all files

# In[22]:


link =r"C:\Users\jchen148\THE Rankings\Report to Jane\THE_CountryCode_Result_1202_{}.csv"

for i in range(0, 12):
    i+=1
    print(link.format(i))


# In[5]:


import pandas as pd


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


# In[ ]:


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

