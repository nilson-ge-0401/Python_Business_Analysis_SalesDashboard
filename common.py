"""Shared utilities for standalone lesson scripts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

# 這支程式大致是在做「課程共用工具(函式)」：
# 設定資料路徑、檢查必要套件、載入 CSV 資料、
# 建立訂單分析表與建模特徵表，並設定 matplotlib 中文字型。

# 設定專案資料路徑：資料檔會由 generate_course_data.py 產生在 code/data/raw/ 底下。
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

# 課程程式會用到的第三方套件清單。
# tuple 第一個值是 Python 匯入名稱，第二個值是安裝套件名稱。
_REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
]


def ensure_packages() -> None:
    """檢查必要套件是否已安裝；缺少套件時提示使用 uv sync 安裝。"""
    missing: list[str] = []

    # 逐一嘗試匯入套件；若匯入失敗，就記錄對應的安裝套件名稱。
    for module_name, package_name in _REQUIRED:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)

    # 若有缺少套件，直接中止程式並輸出清楚的安裝指令。
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise SystemExit(f"Missing required packages: {names}\\nInstall with: uv sync")


def load_data() -> dict[str, Any]:
    """載入課程需要的所有 CSV 資料表，並以 dict 回傳。"""
    import pandas as pd

    # key 是資料表在程式中的名稱，value 是讀入後的 pandas DataFrame。
    return {
        "customers": pd.read_csv(RAW / "customers.csv"),
        "products": pd.read_csv(RAW / "products.csv"),
        "orders": pd.read_csv(RAW / "orders.csv"),
        "order_items": pd.read_csv(RAW / "order_items.csv"),
        "sessions": pd.read_csv(RAW / "sessions.csv"),
        "events": pd.read_csv(RAW / "events.csv"),
        "ab_assignments": pd.read_csv(RAW / "ab_assignments.csv"),
    }


def order_facts(data: dict[str, Any]):
    """建立已完成訂單的彙總事實表，用於營收與訂單分析。"""
    import pandas as pd

    # 複製明細資料，避免直接修改原始 data["order_items"]。
    items = data["order_items"].copy()

    # 計算每筆訂單明細的實際營收：數量 * 單價 * 折扣後比例。
    items["line_revenue"] = items["quantity"] * items["unit_price"] * (1 - items["discount_rate"])

    # 依訂單彙總明細營收，再接回訂單主檔，只保留已完成的訂單。
    facts = (
        items.groupby("order_id", as_index=False)["line_revenue"]
        .sum()
        .merge(data["orders"], on="order_id", how="left")
        .query("status == 'completed'")
    )

    # 將訂單日期轉成 datetime，方便後續做時間序列、月份或週末等分析。
    facts["order_date"] = pd.to_datetime(facts["order_date"])
    return facts


def build_features(data: dict[str, Any]):
    """把 session、event、customer 資料整理成可用於建模或轉換率分析的特徵表。"""
    import pandas as pd

    events = data["events"]

    # 複製 session 主表，後續會在這張表上合併事件統計與顧客屬性。
    sessions = data["sessions"].copy()

    # 統計每個 session 內各種事件出現次數，例如瀏覽、加入購物車、購買。
    event_counts = (
        events.groupby(["session_id", "event_type"]).size().unstack(fill_value=0).reset_index()
        if not events.empty
        else pd.DataFrame({"session_id": sessions["session_id"]})
    )

    # 確保三種核心事件欄位都存在；若原始資料沒有該事件，就以 0 補上。
    for col in ["page_view", "add_to_cart", "purchase"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    # 將事件次數合併回 session，缺值補 0，並轉為整數方便後續運算。
    sessions = sessions.merge(event_counts[["session_id", "page_view", "add_to_cart", "purchase"]], on="session_id", how="left")
    sessions[["page_view", "add_to_cart", "purchase"]] = sessions[["page_view", "add_to_cart", "purchase"]].fillna(0).astype(int)

    # 接上顧客分群與取得來源，讓模型或分析能使用顧客層級特徵。
    sessions = sessions.merge(data["customers"][["customer_id", "segment", "acquisition_channel"]], on="customer_id", how="left")

    # 從 session 開始時間萃取時間特徵。
    dt = pd.to_datetime(sessions["session_start"])
    sessions["session_hour"] = dt.dt.hour
    sessions["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # 建立二元目標欄位：該 session 是否發生購買事件。
    sessions["target"] = (sessions["purchase"] > 0).astype(int)
    return sessions


def configure_plot_fonts() -> bool:
    """Configure a CJK-capable plotting font when available.

    Returns True when a CJK font is found, otherwise False.
    """
    from matplotlib import font_manager
    from matplotlib import rcParams

    # 依常見系統與字型套件列出可顯示中文的候選字型。
    cjk_candidates = [
        "Microsoft JhengHei",
        "PingFang TC",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Source Han Sans TW",
        "Source Han Sans CN",
        "SimHei",
        "Arial Unicode MS",
    ]

    # 掃描目前 matplotlib 可用字型，選出第一個有安裝的中文字型。
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen_font = next((name for name in cjk_candidates if name in available_fonts), None)

    # 找到中文字型時，設定為圖表預設無襯線字型，並修正負號顯示問題。
    if chosen_font:
        rcParams["font.sans-serif"] = [chosen_font, "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        print(f"Using CJK font: {chosen_font}")
        return True

    # 沒找到中文字型時，保留英文標籤流程並提示使用者可安裝的字型。
    print(
        "No CJK font found. Falling back to English labels. "
        "To render Chinese text, install a font such as 'Microsoft JhengHei' "
        "or 'Noto Sans CJK TC'."
    )
    return False
