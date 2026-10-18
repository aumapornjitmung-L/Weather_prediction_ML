import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from weather_ml import run_pipeline, search_city

st.set_page_config(page_title="พยากรณ์อากาศด้วย ML", page_icon="🌦️", layout="wide")
st.title("🌦️ พยากรณ์อากาศด้วย Machine Learning")
st.caption("ข้อมูลย้อนหลังจาก Open-Meteo • โมเดล: Gradient Boosting (scikit-learn) • พยากรณ์ 7 วัน")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_pipeline(lat, lon, years):
    return run_pipeline(lat, lon, years)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(name):
    return search_city(name)


with st.sidebar:
    st.header("ตั้งค่า")
    query = st.text_input("ค้นหาเมือง", "Bangkok")
    years = st.slider("ข้อมูลย้อนหลัง (ปี)", 3, 15, 8)

try:
    results = cached_search(query) if query else []
except Exception as e:
    st.error(f"ค้นหาเมืองไม่สำเร็จ: {e}")
    st.stop()

if not results:
    st.warning("ไม่พบเมืองที่ค้นหา ลองพิมพ์ชื่อเป็นภาษาอังกฤษ")
    st.stop()

labels = [f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}" for r in results]
city = results[st.sidebar.selectbox("เลือกเมือง", range(len(results)), format_func=lambda i: labels[i])]

with st.spinner("กำลังดึงข้อมูลและเทรนโมเดล (ครั้งแรกอาจใช้เวลา 20-40 วินาที)..."):
    try:
        hist, fc, metrics = cached_pipeline(city["latitude"], city["longitude"], years)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.stop()

st.subheader(f"พยากรณ์ 7 วัน: {city['name']}")
st.caption(f"ข้อมูลล่าสุดที่ใช้: {hist.index[-1]:%d/%m/%Y}")

cols = st.columns(7)
for col, (_, r) in zip(cols, fc.iterrows()):
    with col:
        st.metric(f"{r['date']:%a %d/%m}", f"{r['tmax']:.1f}°C", f"ต่ำสุด {r['tmin']:.1f}°C", delta_color="off")
        st.write(f"🌧️ ฝน {r['rain_prob']*100:.0f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index[-30:], y=hist["tmax"].iloc[-30:], name="สูงสุด (จริง)", line=dict(color="#e4572e")))
fig.add_trace(go.Scatter(x=hist.index[-30:], y=hist["tmin"].iloc[-30:], name="ต่ำสุด (จริง)", line=dict(color="#2e86e4")))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["tmax"], name="สูงสุด (พยากรณ์)", line=dict(color="#e4572e", dash="dash")))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["tmin"], name="ต่ำสุด (พยากรณ์)", line=dict(color="#2e86e4", dash="dash")))
fig.update_layout(yaxis_title="°C", height=400, margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure(go.Bar(x=fc["date"], y=fc["rain_prob"] * 100, marker_color="#4a90d9"))
fig2.update_layout(title="โอกาสฝนตก (%)", yaxis_range=[0, 100], height=300)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("ความแม่นยำของโมเดล (ทดสอบกับข้อมูล 20% ล่าสุด)"):
    mt = metrics.rename(columns={
        "horizon": "วันล่วงหน้า",
        "tmax_mae": "MAE สูงสุด (°C)", "tmax_mae_baseline": "Baseline สูงสุด",
        "tmin_mae": "MAE ต่ำสุด (°C)", "tmin_mae_baseline": "Baseline ต่ำสุด",
        "rain_acc": "Acc ฝน", "rain_acc_baseline": "Baseline ฝน",
    })
    st.dataframe(mt.round(3), hide_index=True, use_container_width=True)
    st.caption("Baseline = สมมติว่าวันข้างหน้าเหมือนวันนี้ ถ้าโมเดลดีกว่า ค่า MAE ควรต่ำกว่า Baseline")

with st.expander("ข้อมูลย้อนหลัง"):
    st.dataframe(hist.tail(60))
