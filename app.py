import streamlit as st
import pandas as pd
from datetime import date

def format_vnd(amount):
    return f"{amount:,.0f}".replace(',', '.') + " đ"

st.set_page_config(page_title="Sổ Thu Nợ", page_icon="📝", layout="centered")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 50px; font-size: 16px; font-weight: bold; background-color: #0068c9; color: white; }
    div[data-testid="stMetricValue"] { font-size: 26px; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📝 Sổ Quản Lý Thu Nợ")

if 'debts' not in st.session_state:
    st.session_state.debts = pd.DataFrame(
        columns=['Tên khách nợ', 'Số tiền (VND)', 'Đã thu (VND)', 'Ngày hẹn trả', 'Trạng thái', 'Ghi chú']
    )

df = st.session_state.debts
tong_no = df['Số tiền (VND)'].sum() if not df.empty else 0
da_thu = df['Đã thu (VND)'].sum() if not df.empty else 0
con_lai = tong_no - da_thu

col1, col2 = st.columns(2)
col1.metric("🔴 Tổng nợ cần thu", format_vnd(tong_no))
col2.metric("🟢 Số tiền đã thu", format_vnd(da_thu))
st.metric("🟡 Số tiền còn lại (Treo nợ)", format_vnd(con_lai))

st.divider()

tab1, tab2, tab3 = st.tabs(["💰 Nhập tiền thu", "➕ Thêm nợ mới", "📋 Bảng quản lý"])

with tab1:
    st.subheader("Cập nhật xác nhận tiền thu")
    if not st.session_state.debts.empty:
        with st.form("thu_tien_form", clear_on_submit=True):
            danh_sach_khach = st.session_state.debts['Tên khách nợ'].tolist()
            khach_hang = st.selectbox("Chọn người trả tiền *", danh_sach_khach)
            so_tien_tra = st.number_input("Số tiền khách trả đợt này (VND) *", min_value=0.0, step=50000.0)
            
            st.markdown(f"💵 *Số tiền bạn đang nhập:* **:blue[{format_vnd(so_tien_tra)}]**")
            
            submit_thu = st.form_submit_button("✅ XÁC NHẬN THU TIỀN")
            
            if submit_thu:
                if so_tien_tra > 0:
                    idx = st.session_state.debts[st.session_state.debts['Tên khách nợ'] == khach_hang].index[0]
                    st.session_state.debts.at[idx, 'Đã thu (VND)'] += so_tien_tra
                    
                    if st.session_state.debts.at[idx, 'Đã thu (VND)'] >= st.session_state.debts.at[idx, 'Số tiền (VND)']:
                        st.session_state.debts.at[idx, 'Trạng thái'] = "✅ Đã thu xong"
                    
                    st.success(f"✅ Đã ghi nhận: {khach_hang} vừa thanh toán {format_vnd(so_tien_tra)}!")
                    st.rerun()
                else:
                    st.error("⚠️ Vui lòng nhập số tiền lớn hơn 0!")
    else:
        st.info("Chưa có danh sách nợ. Hãy qua thẻ 'Thêm nợ mới' để tạo.")

with tab2:
    with st.form("add_debt_form", clear_on_submit=True):
        name = st.text_input("Tên người nợ / Khách hàng *")
        amount = st.number_input("Số tiền nợ (VND) *", min_value=0.0, step=100000.0)
        
        st.markdown(f"💵 *Số tiền bạn đang nhập:* **:blue[{format_vnd(amount)}]**")
        
        date_due = st.date_input("Ngày hẹn trả", date.today())
        note = st.text_input("Ghi chú (SĐT, lý do...)")
        
        submit_add = st.form_submit_button("➕ LƯU KHOẢN NỢ")
        
        if submit_add:
            if name.strip() == "" or amount <= 0:
                st.error("⚠️ Vui lòng nhập tên người nợ và số tiền lớn hơn 0!")
            else:
                new_row = pd.DataFrame([{
                    'Tên khách nợ': name, 'Số tiền (VND)': amount, 'Đã thu (VND)': 0.0,
                    'Ngày hẹn trả': pd.to_datetime(date_due).date(), 'Trạng thái': '⏳ Đang chờ', 'Ghi chú': note
                }])
                st.session_state.debts = pd.concat([st.session_state.debts, new_row], ignore_index=True)
                st.success(f"✅ Đã thêm khoản nợ của {name} với số tiền {format_vnd(amount)}!")
                st.rerun()

with tab3:
    st.caption("Bảng theo dõi chi tiết. Sửa trạng thái bằng cách chạm vào ô tương ứng.")
    if not st.session_state.debts.empty:
        display_df = st.session_state.debts.copy()
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Tên khách nợ": st.column_config.TextColumn("Người nợ", disabled=True),
                "Số tiền (VND)": st.column_config.NumberColumn("Cần thu", disabled=True), 
                "Đã thu (VND)": st.column_config.NumberColumn("Đã thu", disabled=True), 
                "Ngày hẹn trả": st.column_config.DateColumn("Hẹn trả", format="DD/MM/YYYY"),
                "Trạng thái": st.column_config.SelectboxColumn(
                    "Trạng thái", options=["⏳ Đang chờ", "✅ Đã thu xong", "⚠️ Trễ hẹn", "❌ Khó đòi"], required=True
                )
            },
            hide_index=True, use_container_width=True, num_rows="dynamic"
        )
        if st.button("💾 LƯU SỬA ĐỔI BẢNG"):
            st.session_state.debts = edited_df
            st.success("✅ Đã lưu thay đổi!")
            st.rerun()