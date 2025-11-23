import streamlit as st
import pandas as pd
import joblib

# Load Pipeline
try:
    pipeline = joblib.load('ckd_pipeline.pkl')
except FileNotFoundError:
    st.error("Chưa thấy file 'ckd_pipeline.pkl'. Vui lòng chạy file train_pipeline.py trước.")
    st.stop()

st.title("🔬 Demo Dự Đoán Bệnh Thận")
st.caption("Sử dụng Sklearn Pipeline: Auto Impute -> Scale -> OneHot -> Predict")

with st.form("input_form"):
    st.subheader("I. Chỉ số Số học (Numerical)")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Tuổi (age)", value=40.0)
        bp = st.number_input("Huyết áp (bp)", value=80.0)
        bgr = st.number_input("Đường huyết (bgr)", value=120.0)
        bu = st.number_input("Urê máu (bu)", value=36.0)
    with c2:
        sc = st.number_input("Creatinine (sc)", value=1.2)
        sod = st.number_input("Natri (sod)", value=135.0)
        pot = st.number_input("Kali (pot)", value=4.0)
        hemo = st.number_input("Hemoglobin (hemo)", value=15.0)
    with c3:
        pcv = st.number_input("Thể tích hồng cầu (pcv)", value=44.0)
        wc = st.number_input("Bạch cầu (wc)", value=7800.0)
        rc = st.number_input("Hồng cầu (rc)", value=5.2)

    st.subheader("II. Chỉ số Phân loại (Categorical)")
    st.write("Chọn giá trị đúng theo định dạng")

    c4, c5, c6 = st.columns(3)
    with c4:
        # Lưu ý: Các giá trị selectbox phải khớp với string trong file CSV gốc
        sg = st.selectbox("Tỷ trọng (sg)", ['1.005', '1.010', '1.015', '1.020', '1.025'], index=3)
        al = st.selectbox("Albumin (al)", ['0', '1', '2', '3', '4', '5'], index=0)
        su = st.selectbox("Đường niệu (su)", ['0', '1', '2', '3', '4', '5'], index=0)
        rbc = st.selectbox("Hồng cầu niệu (rbc)", ['normal', 'abnormal'], index=0)
        pc = st.selectbox("Tế bào mủ (pc)", ['normal', 'abnormal'], index=0)

    with c5:
        pcc = st.selectbox("Đám tế bào mủ (pcc)", ['notpresent', 'present'], index=0)
        ba = st.selectbox("Vi khuẩn (ba)", ['notpresent', 'present'], index=0)
        htn = st.selectbox("Cao huyết áp (htn)", ['no', 'yes'], index=0)
        dm = st.selectbox("Tiểu đường (dm)", ['no', 'yes'], index=0)
        cad = st.selectbox("Bệnh mạch vành (cad)", ['no', 'yes'], index=0)

    with c6:
        appet = st.selectbox("Ăn uống (appet)", ['good', 'poor'], index=0)
        pe = st.selectbox("Phù chân (pe)", ['no', 'yes'], index=0)
        ane = st.selectbox("Thiếu máu (ane)", ['no', 'yes'], index=0)

    submit = st.form_submit_button("Dự đoán")

if submit:
    # 1. Tạo DataFrame từ input (đúng tên cột như lúc train)
    input_data = pd.DataFrame({
        'age': [age], 'bp': [bp], 'bgr': [bgr], 'bu': [bu], 'sc': [sc],
        'sod': [sod], 'pot': [pot], 'hemo': [hemo], 'pcv': [pcv], 'wc': [wc], 'rc': [rc],
        'sg': [sg], 'al': [al], 'su': [su], 'rbc': [rbc], 'pc': [pc],
        'pcc': [pcc], 'ba': [ba], 'htn': [htn], 'dm': [dm], 'cad': [cad],
        'appet': [appet], 'pe': [pe], 'ane': [ane]
    })

    # 2. Đưa thẳng DataFrame thô vào pipeline
    # Pipeline sẽ tự động: Impute -> Scale -> OneHot -> Model Predict
    try:
        prediction = pipeline.predict(input_data)
        proba = pipeline.predict_proba(input_data)

        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ DỰ BÁO: BỊ BỆNH THẬN (CKD)")
            st.write(f"Độ tin cậy: {proba[0][1] * 100:.2f}%")
        else:
            st.success(f"✅ DỰ BÁO: KHỎE MẠNH (NOT CKD)")
            st.write(f"Độ tin cậy: {proba[0][0] * 100:.2f}%")

    except Exception as e:

        st.error(f"Có lỗi xảy ra trong pipeline: {e}")
