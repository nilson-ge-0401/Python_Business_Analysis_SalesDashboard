"""Interactive customer, product, and revenue dashboard."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from common import RAW


st.set_page_config(page_title="電商營運儀表板", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="正在載入資料…")
def load_dashboard_data() -> pd.DataFrame:
    """Load the four source files and return completed-order line facts."""
    customers = pd.read_csv(RAW / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(RAW / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(RAW / "order_items.csv")
    products = pd.read_csv(RAW / "products.csv").rename(columns={"unit_price": "list_price"})

    items["line_revenue"] = (
        items["quantity"] * items["unit_price"] * (1 - items["discount_rate"])
    )
    facts = (
        items.merge(products, on="product_id", how="left", validate="many_to_one")
        .merge(orders, on="order_id", how="left", validate="many_to_one")
        .merge(customers, on="customer_id", how="left", validate="many_to_one")
    )
    facts = facts.loc[facts["status"].eq("completed")].copy()
    facts["order_month"] = facts["order_date"].dt.to_period("M").dt.to_timestamp()

    return facts


def money(value: float) -> str:
    return f"NT$ {value:,.0f}"


def top_table(data: pd.DataFrame, group: str, label: str, limit: int = 10) -> pd.DataFrame:
    result = (
        data.groupby(group, as_index=False)
        .agg(營收=("line_revenue", "sum"), 訂單數=("order_id", "nunique"), 銷售件數=("quantity", "sum"))
        .sort_values("營收", ascending=False)
        .head(limit)
        .rename(columns={group: label})
    )
    result["平均訂單金額"] = result["營收"] / result["訂單數"]
    return result


facts = load_dashboard_data()

st.title("📊 電商營運動態儀表板")
st.caption("資料範圍：完成訂單；營收 = 數量 × 成交單價 ×（1 − 折扣率）")

with st.sidebar:
    st.header("篩選條件")
    min_date = facts["order_date"].min().date()
    max_date = facts["order_date"].max().date()
    selected_dates = st.date_input(
        "訂單日期",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates if isinstance(selected_dates, date) else min_date

    def multi_filter(label: str, column: str) -> list[str]:
        options = sorted(facts[column].dropna().astype(str).unique())
        return st.multiselect(label, options, default=options)

    segments = multi_filter("客群", "segment")
    cities = multi_filter("城市", "city")
    channels = multi_filter("獲客通路", "acquisition_channel")
    categories = multi_filter("商品品類", "category")
    payments = multi_filter("付款方式", "payment_type")
    top_n = st.slider("排行榜顯示筆數", 5, 20, 10)
    st.divider()
    st.caption("所有圖表與指標會隨篩選條件同步更新。")

mask = (
    facts["order_date"].dt.date.between(start_date, end_date)
    & facts["segment"].astype(str).isin(segments)
    & facts["city"].astype(str).isin(cities)
    & facts["acquisition_channel"].astype(str).isin(channels)
    & facts["category"].astype(str).isin(categories)
    & facts["payment_type"].astype(str).isin(payments)
)
filtered = facts.loc[mask].copy()

if filtered.empty:
    st.warning("目前篩選條件沒有符合的完成訂單，請放寬篩選範圍。")
    st.stop()

order_summary = filtered.groupby("order_id", as_index=False)["line_revenue"].sum()
revenue = filtered["line_revenue"].sum()
orders_count = filtered["order_id"].nunique()
customers_count = filtered["customer_id"].nunique()
units = int(filtered["quantity"].sum())

k1, k2, k3 = st.columns(3)
k1.metric("總營收", money(revenue))
k2.metric("完成訂單", f"{orders_count:,}")
k3.metric("客戶數", f"{customers_count:,}")

k4, k5 = st.columns(2)
k4.metric("平均訂單金額", money(order_summary["line_revenue"].mean()))
k5.metric("銷售件數", f"{units:,}")

revenue_tab, customer_tab, product_tab, detail_tab = st.tabs(
    ["收入營收", "客群分析", "商品分析", "明細資料"]
)

with revenue_tab:
    monthly = (
        filtered.groupby("order_month", as_index=False)
        .agg(營收=("line_revenue", "sum"), 訂單數=("order_id", "nunique"))
        .sort_values("order_month")
        .rename(columns={"order_month": "月份"})
    )
    monthly["平均訂單金額"] = monthly["營收"] / monthly["訂單數"]
    left, right = st.columns([2, 1])
    with left:
        st.subheader("月營收趨勢")
        st.line_chart(monthly, x="月份", y="營收", color="#2563EB")
    with right:
        st.subheader("付款方式營收")
        payment = filtered.groupby("payment_type", as_index=False)["line_revenue"].sum()
        st.bar_chart(payment, x="payment_type", y="line_revenue", horizontal=True, color="#14B8A6")

    st.subheader("月度營運摘要")
    st.dataframe(
        monthly,
        width="stretch",
        hide_index=True,
        column_config={
            "月份": st.column_config.DateColumn(format="YYYY-MM"),
            "營收": st.column_config.NumberColumn(format="NT$ %,.0f"),
            "平均訂單金額": st.column_config.NumberColumn(format="NT$ %,.0f"),
        },
    )

with customer_tab:
    customer_view = (
        filtered.groupby(["customer_id", "segment", "city", "acquisition_channel"], as_index=False)
        .agg(營收=("line_revenue", "sum"), 訂單數=("order_id", "nunique"), 最後購買日=("order_date", "max"))
    )
    customer_view["平均訂單金額"] = customer_view["營收"] / customer_view["訂單數"]
    left, right = st.columns(2)
    with left:
        st.subheader("客群營收")
        segment_summary = top_table(filtered, "segment", "客群", limit=100)
        st.bar_chart(segment_summary, x="客群", y="營收", color="#7C3AED")
    with right:
        st.subheader("獲客通路營收")
        channel_summary = top_table(filtered, "acquisition_channel", "獲客通路", limit=100)
        st.bar_chart(channel_summary, x="獲客通路", y="營收", color="#F59E0B")

    st.subheader(f"高價值客戶 Top {top_n}")
    top_customers = customer_view.nlargest(top_n, "營收")
    st.dataframe(
        top_customers,
        width="stretch",
        hide_index=True,
        column_config={
            "營收": st.column_config.ProgressColumn(format="NT$ %,.0f", min_value=0, max_value=float(top_customers["營收"].max())),
            "平均訂單金額": st.column_config.NumberColumn(format="NT$ %,.0f"),
            "最後購買日": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )

with product_tab:
    products_top = (
        filtered.groupby(["product_id", "category"], as_index=False)
        .agg(營收=("line_revenue", "sum"), 銷售件數=("quantity", "sum"), 訂單數=("order_id", "nunique"))
        .sort_values("營收", ascending=False)
    )
    category_summary = top_table(filtered, "category", "品類", limit=100)
    left, right = st.columns(2)
    with left:
        st.subheader("品類營收")
        st.bar_chart(category_summary, x="品類", y="營收", color="#EC4899")
    with right:
        st.subheader("品類銷售件數")
        st.bar_chart(category_summary, x="品類", y="銷售件數", color="#06B6D4")

    st.subheader(f"熱銷商品 Top {top_n}")
    shown_products = products_top.head(top_n)
    st.dataframe(
        shown_products,
        width="stretch",
        hide_index=True,
        column_config={
            "營收": st.column_config.ProgressColumn(format="NT$ %,.0f", min_value=0, max_value=float(shown_products["營收"].max())),
        },
    )

with detail_tab:
    detail_columns = [
        "order_id", "order_date", "customer_id", "segment", "city",
        "acquisition_channel", "payment_type", "product_id", "category",
        "quantity", "unit_price", "discount_rate", "line_revenue",
    ]
    detail = filtered[detail_columns].sort_values("order_date", ascending=False)
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        column_config={
            "order_date": st.column_config.DateColumn("訂單日期", format="YYYY-MM-DD"),
            "line_revenue": st.column_config.NumberColumn("明細營收", format="NT$ %,.0f"),
            "unit_price": st.column_config.NumberColumn("成交單價", format="NT$ %,.0f"),
            "discount_rate": st.column_config.NumberColumn("折扣率", format="%.0f%%"),
        },
    )
    st.download_button(
        "下載目前篩選結果（CSV）",
        data=detail.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_order_details.csv",
        mime="text/csv",
    )

st.caption(f"目前顯示 {orders_count:,} 筆完成訂單、{len(filtered):,} 筆商品明細。")
