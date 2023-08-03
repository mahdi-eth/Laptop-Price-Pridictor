import streamlit as st
import numpy as np
import pickle

df = pickle.load(open("df.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.markdown(
    """
        <style>
            .css-76z9jo {
                display: none;
            }
        """,
    unsafe_allow_html=True,
)

brand = st.selectbox("Brand", df["Company"].unique())
typename = st.selectbox("Type", df["TypeName"].unique())
ram = st.selectbox("RAM (GB)", sorted(
    list(map(int, df["Ram"].unique().tolist()))))
weight = st.number_input(label="Weight (KG)")
touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
ips = st.selectbox("IPS", ["Yes", "No"])
screen_size = st.number_input("Screen Size (Inches)")
resolution = st.selectbox("Screen Resolution", [
    "1366x768",
    "1920x1080",
    "2560x1440",
    "1600x900",
    "1440x900",
    "1280x800",
    "2880x1800",
    "2304x1440",
    "2736x1824",
    "2256x1504",
    "1920x1200",
    "2160x1440",
    "2400x1600"
])
cpu = st.selectbox("CPU", df["CpuBrand"].unique())
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (GB)", [0, 8, 16, 32, 64, 128, 256, 512, 768, 1024])
gpu = st.selectbox("GPU", df["GpuBrands"].unique())
os = st.selectbox("OS", df["OS"].unique())


if st.button("Predict Price"):
    try:
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
        query = np.array([brand, typename, ram, weight, touchscreen,
                          ips, ppi, cpu, hdd, ssd, gpu, os])

        query = query.reshape(1, 12)
        st.title("The predicted price of this configuration is " +
                 str(int(np.exp(model.predict(query)[0]) / 82.26645)) + "$")
    except Exception as e:
        error_title = str(e).split(":")[0]
        st.write("An error occurred:", error_title)
