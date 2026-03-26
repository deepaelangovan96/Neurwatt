import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="NeuralWatt", page_icon="⚡")

data = {
    'machine':       ['M1','M2','M3','M4','M5','M6','M7','M8'],
    'runtime_hours': [8,12,6,14,10,16,7,13],
    'power_rating':  [150,200,100,250,180,220,120,240],
    'temperature':   [45,78,40,90,60,85,42,88],
    'idle_time':     [2,5,1,8,3,7,1,6],
    'status':        [0,1,0,1,0,1,0,1]
}
df = pd.DataFrame(data)

X = df[['runtime_hours','power_rating','temperature','idle_time']]
y = df['status']
model = RandomForestClassifier(n_estimators=10,random_state=42)
model.fit(X,y)

st.title("NeuralWatt")
st.subheader("ML-Based Industrial Energy Optimizer")
st.markdown("**TriSpark Tech**")
st.divider()

costs,statuses = [],[]
for i,row in df.iterrows():
    test = pd.DataFrame({
        'runtime_hours':[row['runtime_hours']],
        'power_rating':[row['power_rating']],
        'temperature':[row['temperature']],
        'idle_time':[row['idle_time']]
    })
    pred = model.predict(test)[0]
    costs.append(round(row['idle_time']*row['power_rating']*0.8,2))
    statuses.append("Wasteful" if pred==1 else "Normal")

total = sum([c for c,s in zip(costs,statuses) if s=='Wasteful'])

col1,col2,col3 = st.columns(3)
col1.metric("Total Waste", f"Rs {total:.0f}/month")
col2.metric("Normal", f"{statuses.count('Normal')}/8")
col3.metric("Wasteful", f"{statuses.count('Wasteful')}/8")
st.divider()

fig,ax = plt.subplots(figsize=(10,4))
colors = ['green' if s=='Normal' else 'red' for s in statuses]
ax.bar(df['machine'],costs,color=colors,edgecolor='black')
ax.set_title('NeuralWatt - Energy Cost Report')
ax.set_xlabel('Machines')
ax.set_ylabel('Cost (Rs/month)')
for i,cost in enumerate(costs):
    ax.text(i,cost+30,f'Rs{cost:.0f}',ha='center',fontsize=8)
st.pyplot(fig)

df['Cost'] = costs
df['Status'] = statuses
st.dataframe(df[['machine','temperature','idle_time','Cost','Status']],
             use_container_width=True)
