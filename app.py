import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="NeuralWatt", page_icon="⚡")

# DATA
data = {
    'machine':       ['M1','M2','M3','M4','M5','M6','M7','M8'],
    'runtime_hours': [8,12,6,14,10,16,7,13],
    'power_rating':  [150,200,100,250,180,220,120,240],
    'temperature':   [45,78,40,90,60,85,42,88],
    'idle_time':     [2,5,1,8,3,7,1,6],
    'status':        [0,1,0,1,0,1,0,1]
}
df = pd.DataFrame(data)

# ML MODEL
X = df[['runtime_hours','power_rating','temperature','idle_time']]
y = df['status']
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)

# HEADER
st.title("⚡ NeuralWatt")
st.subheader("ML-Based Industrial Energy Optimizer")
st.markdown("**🏷️ TriSpark Tech**")
st.divider()

# METRICS
costs,statuses = [],[]
for i,row in df.iterrows():
    test = pd.DataFrame({
        'runtime_hours':[row['runtime_hours']],
        'power_rating':[row['power_rating']],
        'temperature':[row['temperature']],
        'idle_time':[row['idle_time']]
    })
    pred = model.predict(test)[0]
    costs.append(row['idle_time']*row['power_rating']*0.8)
    statuses.append("Wasteful" if pred==1 else "Normal")

total_waste = sum([c for c,s in zip(costs,statuses) if s=='Wasteful'])
normal = statuses.count('Normal')
wasteful = statuses.count('Wasteful')

col1,col2,col3 = st.columns(3)
col1.metric("Total Waste", f"Rs {total_waste:.0f}/month", "⚠️")
col2.metric("Normal Machines", f"{normal}/8", "✅")
col3.metric("Wasteful Machines", f"{wasteful}/8", "⚠️")
st.divider()

# BAR CHART
st.subheader("📊 Energy Cost per Machine")
colors = ['green' if s=='Normal' else 'red' for s in statuses]
fig,ax = plt.subplots(figsize=(10,4))
ax.bar(df['machine'],costs,color=colors,edgecolor='black')
ax.axhline(y=500,color='orange',linestyle='--',label='Warning')
for i,cost in enumerate(costs):
    ax.text(i,cost+30,f'Rs{cost:.0f}',ha='center',fontsize=8)
ax.legend()
st.pyplot(fig)

# TABLE
st.subheader("🏭 Machine Status Table")
df['Cost(Rs/month)'] = costs
df['Status'] = statuses
st.dataframe(df[['machine','temperature',
                  'idle_time','Cost(Rs/month)','Status']],
             use_container_width=True)

# PREDICT NEW
st.divider()
st.subheader("🔍 Predict New Machine")
col1,col2 = st.columns(2)
runtime = col1.slider("Runtime Hours",1,24,10)
power = col1.slider("Power Rating",50,300,150)
temp = col2.slider("Temperature",30,100,60)
idle = col2.slider("Idle Time",0,10,3)

if st.button("⚡ Predict!"):
    new = pd.DataFrame({
        'runtime_hours':[runtime],
        'power_rating':[power],
        'temperature':[temp],
        'idle_time':[idle]
    })
    pred = model.predict(new)[0]
    cost = idle * power * 0.8
    if pred == 1:
        st.error(f"⚠️ WASTEFUL! Extra Cost: Rs{cost:.0f}/month")
        st.warning("Fix: Reduce idle time & check temperature!")
    else:
        st.success(f"✅ NORMAL! Cost: Rs{cost:.0f}/month")
