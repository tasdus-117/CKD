import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Cấu hình trang và Load Pipeline
st.set_page_config(page_title="Dự Đoán Bệnh Thận", layout="wide")

try:
    pipeline = joblib.load('ckd_pipeline.pkl')
except FileNotFoundError:
    st.error("⚠️ Không tìm thấy file 'ckd_pipeline.pkl'. Vui lòng kiểm tra lại file model.")
    st.stop()


# 2. Hàm load và làm sạch dữ liệu từ CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('kidney_disease.csv')

        # Làm sạch tên cột (xóa khoảng trắng thừa)
        df.columns = df.columns.str.strip()

        # Làm sạch dữ liệu dạng chuỗi (xóa \t, khoảng trắng thừa)
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            df[col] = df[col].str.strip().str.replace('\t', '')

        # Xử lý các giá trị lạ như '?' trong cột số nếu có, chuyển về NaN
        num_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Không đọc được file CSV: {e}")
        return pd.DataFrame()


df = load_data()

st.title("🔬 Demo Dự Đoán Bệnh Thận (Auto-Fill)")
st.caption("Chọn ID bệnh nhân để tự động điền thông tin hoặc nhập thủ công.")

# --- PHẦN CHỌN DỮ LIỆU ---
if not df.empty:
    all_ids = df['id'].tolist()
    # Thêm tùy chọn "Nhập thủ công" ở đầu list
    selected_option = st.selectbox("📂 Chọn hồ sơ bệnh nhân (theo ID)", ["Nhập thủ công"] + all_ids)
else:
    selected_option = "Nhập thủ công"
    st.warning("Không có dữ liệu CSV để chọn.")


# Hàm trợ giúp để lấy giá trị mặc định an toàn
def get_val(row, col, default):
    if selected_option == "Nhập thủ công" or row is None:
        return default
    val = row[col].values[0]
    # Nếu giá trị là NaN (trống), trả về default
    if pd.isna(val):
        return default
    return val


# Lấy dòng dữ liệu nếu người dùng chọn ID
current_row = None
if selected_option != "Nhập thủ công":
    current_row = df[df['id'] == selected_option]
    st.info(
        f"Đang hiển thị dữ liệu gốc của bệnh nhân ID: {selected_option}. Cột Class thực tế: **{current_row['classification'].values[0]}**")

# --- FORM NHẬP LIỆU (Tự động điền giá trị từ current_row) ---
with st.form("input_form"):
    st.subheader("I. Chỉ số Số học (Numerical)")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Tuổi (age)", value=float(get_val(current_row, 'age', 40.0)))
        bp = st.number_input("Huyết áp (bp)", value=float(get_val(current_row, 'bp', 80.0)))
        bgr = st.number_input("Đường huyết (bgr)", value=float(get_val(current_row, 'bgr', 120.0)))
        bu = st.number_input("Urê máu (bu)", value=float(get_val(current_row, 'bu', 36.0)))
    with c2:
        sc = st.number_input("Creatinine (sc)", value=float(get_val(current_row, 'sc', 1.2)))
        sod = st.number_input("Natri (sod)", value=float(get_val(current_row, 'sod', 135.0)))
        pot = st.number_input("Kali (pot)", value=float(get_val(current_row, 'pot', 4.0)))
        hemo = st.number_input("Hemoglobin (hemo)", value=float(get_val(current_row, 'hemo', 15.0)))
    with c3:
        pcv = st.number_input("Thể tích hồng cầu (pcv)", value=float(get_val(current_row, 'pcv', 44.0)))
        wc = st.number_input("Bạch cầu (wc)", value=float(get_val(current_row, 'wc', 7800.0)))
        rc = st.number_input("Hồng cầu (rc)", value=float(get_val(current_row, 'rc', 5.2)))

    st.subheader("II. Chỉ số Phân loại (Categorical)")


    # Hàm trợ giúp lấy index cho selectbox
    def get_index(row, col, options, default_idx=0):
        if selected_option == "Nhập thủ công" or row is None:
            return default_idx
        val = str(row[col].values[0]).strip()  # Chuyển về string và xóa khoảng trắng

        # Xử lý mapping đặc biệt cho file CSV này nếu cần (ví dụ 1.0 -> '1.0')
        if col in ['sg', 'al', 'su']:
            # Vì trong CSV có thể là số 1.0, 2.0 -> cần ép kiểu về string giống options
            try:
                val = str(float(val))  # 1 -> 1.0
                if val.endswith('.0') and col in ['al',
                                                  'su']:  # al, su thường là '0', '1' trong options chứ k phải '0.0'
                    val = val.replace('.0', '')
            except:
                pass

        if val in options:
            return options.index(val)
        return default_idx


    c4, c5, c6 = st.columns(3)
    with c4:
        opt_sg = ['1.005', '1.010', '1.015', '1.020', '1.025']
        sg = st.selectbox("Tỷ trọng (sg)", opt_sg, index=get_index(current_row, 'sg', opt_sg, 3))

        opt_al = ['0', '1', '2', '3', '4', '5']
        al = st.selectbox("Albumin (al)", opt_al, index=get_index(current_row, 'al', opt_al, 0))

        opt_su = ['0', '1', '2', '3', '4', '5']
        su = st.selectbox("Đường niệu (su)", opt_su, index=get_index(current_row, 'su', opt_su, 0))

        opt_rbc = ['normal', 'abnormal']
        rbc = st.selectbox("Hồng cầu niệu (rbc)", opt_rbc, index=get_index(current_row, 'rbc', opt_rbc, 0))

        opt_pc = ['normal', 'abnormal']
        pc = st.selectbox("Tế bào mủ (pc)", opt_pc, index=get_index(current_row, 'pc', opt_pc, 0))

    with c5:
        opt_pcc = ['notpresent', 'present']
        pcc = st.selectbox("Đám tế bào mủ (pcc)", opt_pcc, index=get_index(current_row, 'pcc', opt_pcc, 0))

        opt_ba = ['notpresent', 'present']
        ba = st.selectbox("Vi khuẩn (ba)", opt_ba, index=get_index(current_row, 'ba', opt_ba, 0))

        opt_htn = ['no', 'yes']
        htn = st.selectbox("Cao huyết áp (htn)", opt_htn, index=get_index(current_row, 'htn', opt_htn, 0))

        opt_dm = ['no', 'yes']
        dm = st.selectbox("Tiểu đường (dm)", opt_dm, index=get_index(current_row, 'dm', opt_dm, 0))

        opt_cad = ['no', 'yes']
        cad = st.selectbox("Bệnh mạch vành (cad)", opt_cad, index=get_index(current_row, 'cad', opt_cad, 0))

    with c6:
        opt_appet = ['good', 'poor']
        appet = st.selectbox("Ăn uống (appet)", opt_appet, index=get_index(current_row, 'appet', opt_appet, 0))

        opt_pe = ['no', 'yes']
        pe = st.selectbox("Phù chân (pe)", opt_pe, index=get_index(current_row, 'pe', opt_pe, 0))

        opt_ane = ['no', 'yes']
        ane = st.selectbox("Thiếu máu (ane)", opt_ane, index=get_index(current_row, 'ane', opt_ane, 0))

    submit = st.form_submit_button("🚀 Chạy Dự Đoán")

if submit:
    # Tạo DataFrame từ input trên form
    input_data = pd.DataFrame({
        'age': [age], 'bp': [bp], 'bgr': [bgr], 'bu': [bu], 'sc': [sc],
        'sod': [sod], 'pot': [pot], 'hemo': [hemo], 'pcv': [pcv], 'wc': [wc], 'rc': [rc],
        'sg': [sg], 'al': [al], 'su': [su], 'rbc': [rbc], 'pc': [pc],
        'pcc': [pcc], 'ba': [ba], 'htn': [htn], 'dm': [dm], 'cad': [cad],
        'appet': [appet], 'pe': [pe], 'ane': [ane]
    })

    try:
        prediction = pipeline.predict(input_data)
        proba = pipeline.predict_proba(input_data)

        st.divider()
        c_res1, c_res2 = st.columns([1, 3])

        with c_res1:
            if prediction[0] == 1 or prediction[0] == 'ckd':  # Xử lý trường hợp model trả về string hoặc int
                st.error("### ⚠️ KẾT QUẢ: BỆNH THẬN (CKD)")
            else:
                st.success("### ✅ KẾT QUẢ: KHỎE MẠNH")

        with c_res2:
            prob_ckd = proba[0][1] * 100
            prob_not_ckd = proba[0][0] * 100
            st.write(f"Độ tin cậy dự đoán:")
            st.progress(int(prob_ckd) if prediction[0] == 1 else int(prob_not_ckd))
            st.caption(f"Tỷ lệ CKD: {prob_ckd:.2f}% | Tỷ lệ Khỏe mạnh: {prob_not_ckd:.2f}%")

    except Exception as e:
        st.error(f"Có lỗi xảy ra khi dự đoán: {e}")