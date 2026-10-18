# 🌦️ Weather Forecast ML (Streamlit)

พยากรณ์อากาศ 7 วัน (อุณหภูมิสูงสุด/ต่ำสุด และโอกาสฝนตก) ด้วย Gradient Boosting
ข้อมูลจาก [Open-Meteo](https://open-meteo.com) (ไม่ต้องใช้ API key)

## รันในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy บน Streamlit Community Cloud
1. push โปรเจกต์ขึ้น GitHub
2. ไปที่ https://share.streamlit.io → New app
3. เลือก repo, branch `main`, Main file path: `app.py` → Deploy
