import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import uuid

# ==========================================
# 1. CẤU HÌNH TRANG & HÀM TIỆN ÍCH
# ==========================================
st.set_page_config(
    page_title="FinPulse AI - Dự báo Dòng tiền Tương lai",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def format_vnd(amount: float) -> str:
    """Định dạng số tiền chuẩn Việt Nam: 15.000.000đ"""
    if pd.isna(amount):
        return "0đ"
    sign = "-" if amount < 0 else ""
    return f"{sign}{abs(int(round(amount))):,.0f}đ".replace(",", ".")

def parse_vnd(text: str) -> int:
    """Chuyển chuỗi nhập liệu (VD: '15.000.000đ' hoặc '15000000') về số nguyên"""
    digits = "".join(filter(str.isdigit, str(text)))
    return int(digits) if digits else 0

# CSS tùy chỉnh giao diện hiện đại
st.markdown("""
<style>
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.78rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KHỞI TẠO DỮ LIỆU TRỐNG (SESSION STATE)
# ==========================================
if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "last_deleted_backup" not in st.session_state:
    st.session_state.last_deleted_backup = []

# ==========================================
# 3. THUẬT TOÁN DỰ BÁO DÒNG TIỀN TỐI ƯU (HYBRID ENGINE)
# ==========================================
def run_cashflow_forecast(df: pd.DataFrame, initial_balance: float, horizon_days: int):
    today = datetime.now().date()
    if df.empty:
        dates = [today + timedelta(days=i) for i in range(horizon_days + 1)]
        return pd.DataFrame({
            "date": dates,
            "net_flow": [0.0] * len(dates),
            "balance": [initial_balance] * len(dates),
            "upper": [initial_balance] * len(dates),
            "lower": [initial_balance] * len(dates),
            "is_forecast": [i > 0 for i in range(len(dates))]
        })

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["signed_amount"] = np.where(df["type"] == "income", df["amount"], -df["amount"])

    # 1. Tổng hợp dòng tiền lịch sử (từ ngày sớm nhất đến hôm nay)
    min_date = min(df["date"].min(), today - timedelta(days=14))
    hist_dates = pd.date_range(min_date, today, freq="D").date
    hist_df = df[df["date"] <= today].groupby("date")["signed_amount"].sum().reindex(hist_dates, fill_value=0.0)

    # Tính số dư tích lũy lịch sử
    hist_balance = initial_balance + hist_df.cumsum()

    # 2. Mô hình Holt-Winters + Hồi quy tuyến tính OLS cho chuỗi thời gian
    y_hist = hist_df.values.astype(float)
    n_hist = len(y_hist)

    # Hồi quy tuyến tính (OLS Trend)
    x_hist = np.arange(n_hist)
    if n_hist >= 2:
        slope, intercept = np.polyfit(x_hist, y_hist, 1)
        x_future = np.arange(n_hist, n_hist + horizon_days)
        ols_pred = intercept + slope * x_future
    else:
        ols_pred = np.full(horizon_days, np.mean(y_hist))

    # Holt-Winters (Triple Exponential Smoothing - chu kỳ 7 ngày)
    try:
        if n_hist >= 14 and np.std(y_hist) > 0:
            hw_model = ExponentialSmoothing(
                y_hist,
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                damped_trend=True
            ).fit(optimized=True)
            hw_pred = hw_model.forecast(horizon_days)
        elif n_hist >= 4 and np.std(y_hist) > 0:
            hw_model = ExponentialSmoothing(
                y_hist,
                trend="add",
                damped_trend=True
            ).fit(optimized=True)
            hw_pred = hw_model.forecast(horizon_days)
        else:
            hw_pred = ols_pred
    except Exception:
        hw_pred = ols_pred

    # Kết hợp trọng số (65% Holt-Winters + 35% OLS)
    base_daily_pred = 0.65 * np.array(hw_pred) + 0.35 * np.array(ols_pred)

    # 3. Cộng dồn các giao dịch định kỳ & đơn hàng tương lai đã lên lịch
    future_dates = [today + timedelta(days=i) for i in range(1, horizon_days + 1)]
    scheduled_future = np.zeros(horizon_days)

    for _, row in df.iterrows():
        tx_date = row["date"]
        amt = row["signed_amount"]
        freq = row["freq"]

        for idx, f_date in enumerate(future_dates):
            if freq == "Một lần" and f_date == tx_date:
                scheduled_future[idx] += amt
            elif freq == "Hàng tuần" and f_date > tx_date and (f_date - tx_date).days % 7 == 0:
                scheduled_future[idx] += amt
            elif freq == "Hàng tháng" and f_date > tx_date and f_date.day == tx_date.day:
                scheduled_future[idx] += amt

    total_future_net = (base_daily_pred * 0.35) + scheduled_future

    # 4. Tính số dư tương lai & Dải tin cậy 95% (Confidence Interval)
    last_balance = float(hist_balance.iloc[-1])
    future_balance = last_balance + np.cumsum(total_future_net)

    daily_std = float(np.std(y_hist)) if n_hist > 1 else abs(last_balance) * 0.05
    if daily_std == 0:
        daily_std = 1000000.0
    steps = np.arange(1, horizon_days + 1)
    margin_95 = 1.96 * daily_std * np.sqrt(steps) * 0.45

    upper_bound = future_balance + margin_95
    lower_bound = future_balance - margin_95

    # Gộp dữ liệu lịch sử & dự báo
    res_hist = pd.DataFrame({
        "date": list(hist_dates),
        "net_flow": list(hist_df.values),
        "balance": list(hist_balance.values),
        "upper": list(hist_balance.values),
        "lower": list(hist_balance.values),
        "is_forecast": [False] * len(hist_dates)
    })

    res_fut = pd.DataFrame({
        "date": future_dates,
        "net_flow": total_future_net,
        "balance": future_balance,
        "upper": upper_bound,
        "lower": lower_bound,
        "is_forecast": [True] * horizon_days
    })

    return pd.concat([res_hist, res_fut], ignore_index=True)

# ==========================================
# 4. SIDEBAR: NHẬP DÒNG TIỀN & CẤU HÌNH DỰ BÁO
# ==========================================
with st.sidebar:
    st.title("💸 FinPulse AI")
    st.caption("Hệ thống Quản lý & Dự báo Dòng tiền Thông minh")
    st.divider()

    st.subheader("➕ Thêm Giao dịch / Đơn hàng")
    tx_type = st.radio(
        "Loại dòng tiền",
        options=["income", "expense"],
        format_func=lambda x: "🟢 Thu nhập (+)" if x == "income" else "🔴 Chi phí (-)",
        horizontal=True
    )

    # Ô nhập số tiền định dạng 15.000.000đ
    raw_amount = st.text_input("Số tiền (VNĐ)", value="15.000.000", placeholder="VD: 15.000.000")
    amount = parse_vnd(raw_amount)
    st.caption(f"💰 Giá trị ghi nhận: **{format_vnd(amount)}**")

    desc = st.text_input("Nội dung / Mô tả đơn hàng *", placeholder="VD: Đơn hàng #DH20 - Khách hàng An")

    col_cat, col_date = st.columns(2)
    with col_cat:
        category = st.selectbox(
            "Danh mục",
            ["Doanh thu bán hàng", "Chi phí vận hành", "Lương / Định kỳ", "Marketing", "Thuế & Phí", "Khác"]
        )
    with col_date:
        tx_date = st.date_input("Ngày giao dịch", value=datetime.now().date())

    col_freq, col_stat = st.columns(2)
    with col_freq:
        freq = st.selectbox("Tần suất", ["Một lần", "Hàng tuần", "Hàng tháng"])
    with col_stat:
        status = st.selectbox("Trạng thái", ["Đã trả", "Chờ thanh toán"])

    if st.button("💾 Lưu dòng tiền", type="primary", use_container_width=True):
        if amount <= 0:
            st.error("Vui lòng nhập số tiền lớn hơn 0đ!")
        elif not desc.strip():
            st.error("Vui lòng nhập nội dung / mô tả đơn hàng!")
        else:
            st.session_state.transactions.insert(0, {
                "id": str(uuid.uuid4()),
                "date": tx_date,
                "type": tx_type,
                "amount": amount,
                "desc": desc.strip(),
                "category": category,
                "freq": freq,
                "status": status
            })
            st.success(f"Đã thêm: {desc} ({format_vnd(amount)})")
            st.rerun()

    st.divider()
    st.subheader("⚙️ Cấu hình Dự báo")
    raw_init_bal = st.text_input("Số dư quỹ đầu kỳ (VNĐ)", value="0")
    initial_balance = parse_vnd(raw_init_bal)
    st.caption(f"🏦 Quỹ đầu kỳ: **{format_vnd(initial_balance)}**")

    horizon_days = st.slider("Kỳ hạn dự báo tương lai (ngày)", min_value=7, max_value=90, value=30, step=1)

# ==========================================
# 5. KHU VỰC CHÍNH: KPI, BIỂU ĐỒ & QUẢN LÝ ĐƠN HÀNG
# ==========================================
st.header("📊 Tổng quan & Dự báo Dòng tiền Tương lai")

df_tx = pd.DataFrame(st.session_state.transactions)

# Tính toán các chỉ số KPI
if not df_tx.empty:
    total_income_paid = df_tx[(df_tx["type"] == "income") & (df_tx["status"] == "Đã trả")]["amount"].sum()
    total_expense_paid = df_tx[(df_tx["type"] == "expense") & (df_tx["status"] == "Đã trả")]["amount"].sum()
    pending_income = df_tx[(df_tx["type"] == "income") & (df_tx["status"] == "Chờ thanh toán")]["amount"].sum()
    pending_expense = df_tx[(df_tx["type"] == "expense") & (df_tx["status"] == "Chờ thanh toán")]["amount"].sum()
else:
    total_income_paid = total_expense_paid = pending_income = pending_expense = 0

forecast_df = run_cashflow_forecast(df_tx, initial_balance, horizon_days)
current_balance = forecast_df[~forecast_df["is_forecast"]]["balance"].iloc[-1]
predicted_end_balance = forecast_df["balance"].iloc[-1]
balance_delta = predicted_end_balance - current_balance

# Hiển thị 4 thẻ KPI định dạng 15.000.000đ
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💰 Số dư hiện tại</div>
        <div class="kpi-value">{format_vnd(current_balance)}</div>
        <div class="kpi-sub" style="color:#64748b;">Quỹ đầu kỳ: {format_vnd(initial_balance)}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    delta_color = "#10b981" if balance_delta >= 0 else "#ef4444"
    delta_prefix = "+" if balance_delta >= 0 else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🔮 Dự báo sau {horizon_days} ngày</div>
        <div class="kpi-value">{format_vnd(predicted_end_balance)}</div>
        <div class="kpi-sub" style="color:{delta_color}; font-weight:600;">{delta_prefix}{format_vnd(balance_delta)} so với hiện tại</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🟢 Tổng Thu (Đã trả)</div>
        <div class="kpi-value" style="color:#10b981;">{format_vnd(total_income_paid)}</div>
        <div class="kpi-sub" style="color:#d97706;">⏳ Chờ thu: {format_vnd(pending_income)}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🔴 Tổng Chi (Đã trả)</div>
        <div class="kpi-value" style="color:#ef4444;">{format_vnd(total_expense_paid)}</div>
        <div class="kpi-sub" style="color:#d97706;">⏳ Chờ chi: {format_vnd(pending_expense)}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# 6. BIỂU ĐỒ DỰ BÁO DÒNG TIỀN TƯƠNG TÁC (PLOTLY)
# ==========================================
hist_part = forecast_df[~forecast_df["is_forecast"]]
fut_part = pd.concat([hist_part.tail(1), forecast_df[forecast_df["is_forecast"]]], ignore_index=True)

fig = go.Figure()

# Dải tin cậy 95%
fig.add_trace(go.Scatter(
    x=list(fut_part["date"]) + list(fut_part["date"])[::-1],
    y=list(fut_part["upper"]) + list(fut_part["lower"])[::-1],
    fill="toself",
    fillcolor="rgba(99, 102, 241, 0.14)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    name="Vùng tin cậy 95%"
))

# Đường số dư thực tế lịch sử
fig.add_trace(go.Scatter(
    x=hist_part["date"],
    y=hist_part["balance"],
    mode="lines+markers",
    name="Số dư thực tế",
    line=dict(color="#0ea5e9", width=3),
    marker=dict(size=6),
    customdata=[format_vnd(v) for v in hist_part["balance"]],
    hovertemplate="Ngày: %{x}<br>Số dư thực tế: <b>%{customdata}</b><extra></extra>"
))

# Đường số dư dự báo AI
fig.add_trace(go.Scatter(
    x=fut_part["date"],
    y=fut_part["balance"],
    mode="lines+markers",
    name="Dự báo AI (Holt-Winters + OLS)",
    line=dict(color="#6366f1", width=3, dash="dash"),
    marker=dict(size=6),
    customdata=[format_vnd(v) for v in fut_part["balance"]],
    hovertemplate="Ngày: %{x}<br>Số dư dự báo: <b>%{customdata}</b><extra></extra>"
))

fig.update_layout(
    title=f"Xu hướng Số dư Quỹ & Dự báo {horizon_days} Ngày Tới",
    xaxis_title="Thời gian",
    yaxis_title="Số dư (VNĐ)",
    hovermode="x unified",
    template="plotly_white",
    height=400,
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_yaxes(separatethousands=True)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. QUẢN LÝ NỘI DUNG DÒNG TIỀN & XÓA DỮ LIỆU
# ==========================================
st.subheader("📋 Quản lý Nội dung Dòng tiền & Đơn hàng")

# Thanh công cụ lọc, Xóa đơn đã trả & Xóa tất cả
c_search, c_type, c_stat, c_del, c_del_all = st.columns([2.0, 1.1, 1.2, 1.6, 1.3])
with c_search:
    search_kw = st.text_input("🔍 Tìm kiếm nội dung đơn hàng", placeholder="Nhập từ khóa mô tả...", label_visibility="collapsed")
with c_type:
    filter_type = st.selectbox("Loại", ["Tất cả loại", "🟢 Thu nhập", "🔴 Chi phí"], label_visibility="collapsed")
with c_stat:
    filter_status = st.selectbox("Trạng thái", ["Tất cả trạng thái", "✅ Đã trả", "⏳ Chờ thanh toán"], label_visibility="collapsed")

paid_count = int((df_tx["status"] == "Đã trả").sum()) if not df_tx.empty else 0

with c_del:
    if st.button(f"🗑️ Xóa đã trả ({paid_count})", disabled=(paid_count == 0), use_container_width=True):
        st.session_state.last_deleted_backup = [t for t in st.session_state.transactions if t["status"] == "Đã trả"]
        st.session_state.transactions = [t for t in st.session_state.transactions if t["status"] != "Đã trả"]
        st.rerun()

with c_del_all:
    if st.button("💥 Xóa tất cả", disabled=df_tx.empty, use_container_width=True):
        st.session_state.last_deleted_backup = list(st.session_state.transactions)
        st.session_state.transactions = []
        st.rerun()

# Nút hoàn tác nếu vừa xóa dữ liệu
if st.session_state.last_deleted_backup:
    col_info, col_undo = st.columns([4, 1])
    with col_info:
        st.info(f"Đã xóa {len(st.session_state.last_deleted_backup)} bản ghi dòng tiền.")
    with col_undo:
        if st.button("↩️ Hoàn tác xóa", use_container_width=True):
            st.session_state.transactions.extend(st.session_state.last_deleted_backup)
            st.session_state.last_deleted_backup = []
            st.rerun()

if df_tx.empty:
    st.info("📭 Bảng dữ liệu hiện đang trống. Hãy nhập đơn hàng hoặc dòng tiền mới ở thanh bên trái!")
else:
    # Áp dụng bộ lọc
    mask = pd.Series(True, index=df_tx.index)
    if search_kw.strip():
        mask &= df_tx["desc"].str.contains(search_kw.strip(), case=False, na=False)
    if filter_type == "🟢 Thu nhập":
        mask &= df_tx["type"] == "income"
    elif filter_type == "🔴 Chi phí":
        mask &= df_tx["type"] == "expense"
    if filter_status == "✅ Đã trả":
        mask &= df_tx["status"] == "Đã trả"
    elif filter_status == "⏳ Chờ thanh toán":
        mask &= df_tx["status"] == "Chờ thanh toán"

    filtered_df = df_tx[mask].copy()

    # Tạo cột hiển thị định dạng chuẩn 15.000.000đ và nhãn loại
    filtered_df["type_label"] = filtered_df["type"].map({"income": "🟢 Thu (+)", "expense": "🔴 Chi (-)"})
    filtered_df["amount_display"] = filtered_df["amount"].apply(format_vnd)

    st.caption("💡 Mẹo: Bạn có thể nhấp đúp vào ô **Nội dung**, **Trạng thái**, **Danh mục** hoặc **Số tiền (VNĐ)** để sửa trực tiếp, hoặc chọn dòng và bấm phím `Delete` để xóa lẻ.")

    edited_df = st.data_editor(
        filtered_df[["id", "date", "type_label", "amount_display", "desc", "category", "freq", "status"]],
        column_config={
            "id": None,  # Ẩn cột ID
            "date": st.column_config.DateColumn("Ngày giao dịch", format="DD/MM/YYYY"),
            "type_label": st.column_config.SelectboxColumn(
                "Loại (Thu/Chi)",
                options=["🟢 Thu (+)", "🔴 Chi (-)"],
                required=True
            ),
            "amount_display": st.column_config.TextColumn(
                "Số tiền (VNĐ)",
                help="Nhập số tiền (VD: 15.000.000đ)",
                required=True
            ),
            "desc": st.column_config.TextColumn("Nội dung / Mô tả đơn hàng", width="large", required=True),
            "category": st.column_config.SelectboxColumn(
                "Danh mục",
                options=["Doanh thu bán hàng", "Chi phí vận hành", "Lương / Định kỳ", "Marketing", "Thuế & Phí", "Khác"],
                required=True
            ),
            "freq": st.column_config.SelectboxColumn(
                "Tần suất",
                options=["Một lần", "Hàng tuần", "Hàng tháng"],
                required=True
            ),
            "status": st.column_config.SelectboxColumn(
                "Trạng thái",
                options=["Đã trả", "Chờ thanh toán"],
                required=True
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="tx_editor"
    )

    # Đồng bộ lại thay đổi từ bảng data_editor vào session_state
    if st.button("🔄 Cập nhật thay đổi trên bảng"):
        updated_records = []

        # Giữ lại các bản ghi không nằm trong bộ lọc hiện tại
        for tx in st.session_state.transactions:
            if tx["id"] not in set(filtered_df["id"]):
                updated_records.append(tx)

        # Cập nhật các bản ghi đã chỉnh sửa trên bảng
        for _, row in edited_df.iterrows():
            parsed_amt = parse_vnd(row["amount_display"])
            if parsed_amt > 0 and str(row["desc"]).strip():
                updated_records.append({
                    "id": row["id"] if pd.notna(row["id"]) else str(uuid.uuid4()),
                    "date": row["date"] if pd.notna(row["date"]) else datetime.now().date(),
                    "type": "income" if row["type_label"] == "🟢 Thu (+)" else "expense",
                    "amount": parsed_amt,
                    "desc": str(row["desc"]).strip(),
                    "category": row["category"] if pd.notna(row["category"]) else "Khác",
                    "freq": row["freq"] if pd.notna(row["freq"]) else "Một lần",
                    "status": row["status"] if pd.notna(row["status"]) else "Chờ thanh toán",
                })

        st.session_state.transactions = updated_records
        st.success("Đã lưu và định dạng lại toàn bộ dữ liệu!")
        st.rerun()

    # Nút xuất file CSV báo cáo
    export_df = filtered_df[["date", "type_label", "amount_display", "desc", "category", "freq", "status"]].copy()
    export_df.columns = ["Ngày", "Loại", "Số tiền (VNĐ)", "Nội dung", "Danh mục", "Tần suất", "Trạng thái"]
    csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Tải báo cáo Excel / CSV",
        data=csv_data,
        file_name=f"bao_cao_dong_tien_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )