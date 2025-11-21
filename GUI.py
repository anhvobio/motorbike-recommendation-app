# GUI_project2_full_detailed.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# ⚙️ CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="🔎 Gợi ý & Phân tích xe máy", layout="wide")

# ==============================
# 1️⃣ LOAD DATA & HELPERS
# ==============================
@st.cache_data
def load_data():
    # 🔥 Dùng đường dẫn tương đối để chạy trên Streamlit Cloud
    path = "du_lieu_xe_may_da_tien_xu_ly_1.xlsx"
    df = pd.read_excel(path, sheet_name=0)

    # Chuẩn hóa cột
    rename_map = {
        "Tiêu_đề": "title",
        "Thương_hiệu": "brand",
        "Mô_tả_chi_tiết": "desc",
        "Năm_đăng_ký": "year",
        "Số_Km_đã_đi": "km",
        "Href": "href"
    }
    df = df.rename(columns={col: rename_map[col] for col in df.columns if col in rename_map})

    # Kiểm tra cột price
    if "price" not in df.columns:
        if "Khoảng_giá_min" in df.columns:
            df["price"] = df["Khoảng_giá_min"]
        else:
            df["price"] = np.nan

    # Chuyển price -> numeric
    def clean_price(x):
        try:
            if pd.isna(x):
                return np.nan
            s = str(x)
            s = "".join(ch for ch in s if (ch.isdigit() or ch in ".,"))
            s = s.replace(",", "")
            return float(s) if s else np.nan
        except:
            return np.nan
    df["price"] = df["price"].apply(clean_price)

    # Chuyển km -> numeric
    def clean_km(x):
        try:
            if pd.isna(x):
                return np.nan
            s = "".join(ch for ch in str(x) if ch.isdigit())
            return float(s) if s else np.nan
        except:
            return np.nan
    if "km" in df.columns:
        df["km"] = df["km"].apply(clean_km)
    else:
        df["km"] = np.nan

    # year numeric
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = np.nan

    # full_text
    df["full_text"] = (
        df.get("title", "") + " "
        + df.get("brand", "") + " "
        + df.get("desc", "")
    )

    # Feature: tuổi xe
    current_year = pd.Timestamp.now().year
    df["age"] = df["year"].apply(lambda y: current_year - y if not pd.isna(y) else np.nan)

    return df

@st.cache_resource
def load_similarity():
    # 🔥 Dùng đường dẫn tương đối để tương thích Streamlit Cloud
    try:
        with open("xe_cosine_sim_18112025.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def build_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["full_text"].astype(str))
    return vectorizer, tfidf_matrix

# ==============================
# Load data & models
# ==============================
df = load_data()
cosine_sim_saved = load_similarity()
vectorizer, tfidf_matrix = build_tfidf(df)

# ==============================
# Sidebar menu dẫn tới section
# ==============================
st.sidebar.title("GUI streamlit ứng dụng Gợi Ý và Phân Khúc Xe Máy")
menu = st.sidebar.radio("Đi tới mục:", [
    "Giới thiệu",
    "Gợi ý theo tên xe",
    "Tìm kiếm tự do",
    "Phân cụm xe máy",
    "Thống kê & Trực quan"
])

st.sidebar.markdown("---")
st.sidebar.markdown("## 👥 Nhóm thực hiện")
st.sidebar.markdown("""
**Võ Thị Hoàng Anh**  
**Nguyễn Mai Xuân Bách**
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Thông tin ứng dụng")
st.sidebar.write("Project 2 — Gợi ý & Phân khúc xe máy")
st.sidebar.write("Phiên bản: 1.0")
st.sidebar.write("Cập nhật: 21/11/2025")

# ==============================
# Helper function hiển thị xe
# ==============================
def display_bike(bike, rank=None, score=None):
    if rank is not None:
        st.markdown(f"### {rank}. {bike.get('title','N/A')}")
    else:
        st.markdown(f"### {bike.get('title','N/A')}")
    st.write(f"- **Hãng:** {bike.get('brand','N/A')}")
    st.write(f"- **Giá:** {bike.get('price','N/A')}")
    st.write(f"- **Số km:** {bike.get('km','N/A')}")
    if score is not None:
        st.write(f"- **Similarity/Relevance:** `{round(float(score),4)}`")
    if "href" in bike and pd.notna(bike["href"]):
        st.write(f"- Link: {bike['href']}")
    st.markdown("---")

# ==============================
# Nội dung chính với anchors
# ==============================
st.markdown("<a id='intro'></a>", unsafe_allow_html=True)
st.header("🏍️ Giới thiệu")
st.markdown("""
Dữ liệu trong ứng dụng này là **xe máy đã đăng bán trên Chợ Tốt**, chủ yếu tại **TP.HCM**.  
Chúng tôi đã thu thập và tiền xử lý thông tin xe, bao gồm:  
- Tiêu đề, mô tả chi tiết, thương hiệu  
- Giá, số km đã đi, năm đăng ký, tình trạng xe  
- Các liên kết bài đăng gốc  

Dự án xây dựng **hệ thống hai phần**:
1. **Gợi ý xe máy tương tự**: dựa trên TF-IDF + Cosine Similarity từ title + brand + description.  
2. **Phân khúc thị trường xe máy**: sử dụng KMeans trên dữ liệu TF-IDF để phát hiện nhóm xe tương đồng.  

Mục tiêu:
- Giúp người mua tìm xe tương tự nhanh chóng  
- Hỗ trợ người bán định giá hợp lý  
- Phân tích thị trường xe máy cũ tại TP.HCM
""")

st.markdown("<a id='recommend'></a>", unsafe_allow_html=True)
st.header("🎯 Gợi ý theo tên xe")
list_titles = df["title"].fillna("N/A").tolist()
selected_title = st.selectbox("Chọn xe:", list_titles)
if st.button("Gợi ý ngay"):
    matches = df[df["title"] == selected_title]
    if matches.empty:
        st.warning("Không tìm thấy bản ghi.")
    else:
        idx = matches.index[0]
        if cosine_sim_saved is not None:
            sim_scores = list(enumerate(cosine_sim_saved[idx]))
        else:
            sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top5 = sim_scores[1:6]
        st.subheader("🚀 Top 5 xe tương tự")
        for rank, (i, score) in enumerate(top5, start=1):
            display_bike(df.iloc[i], rank, score)

st.markdown("<a id='search'></a>", unsafe_allow_html=True)
st.header("📝 Tìm kiếm tự do")
query_text = st.text_input("Nhập từ khóa")
if st.button("Tìm kiếm") and query_text.strip():
    query_vec = vectorizer.transform([query_text])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top5 = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)[:5]
    st.subheader("🔝 Top 5 kết quả")
    for rank, (i, score) in enumerate(top5, start=1):
        display_bike(df.iloc[i], rank, score)

st.markdown("<a id='cluster'></a>", unsafe_allow_html=True)
st.header("📊 Phân cụm xe máy")
X = tfidf_matrix
K = st.slider("Chọn số cụm K", 2, 20, 6)
if st.button("Chạy phân cụm"):
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(X)
    df["cluster_kmeans"] = labels
    st.success(f"✅ Đã phân cụm thành {K} cụm.")
    st.bar_chart(df["cluster_kmeans"].value_counts().sort_index())
    cluster_to_show = st.number_input("Chọn cụm để xem chi tiết", 0, K-1, 0)
    st.subheader(f"📋 Xe thuộc cụm {cluster_to_show}")
    sample = df[df["cluster_kmeans"] == cluster_to_show].head(50)
    for _, row in sample.iterrows():
        st.write(f"- **{row.get('title','N/A')}** | Hãng: `{row.get('brand','N/A')}` | Giá: `{row.get('price','N/A')}` | Km: `{row.get('km','N/A')}`")

st.markdown("<a id='stats'></a>", unsafe_allow_html=True)
st.header("📈 Thống kê & Trực quan")
numeric_cols = ["price", "km", "year", "age"]
num_df = df[numeric_cols].copy()
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Phân phối giá")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.histplot(df["price"].dropna(), kde=True, ax=ax1)
    ax1.set_xlabel("Giá")
    ax1.set_ylabel("Số lượng")
    st.pyplot(fig1)

with col2:
    st.subheader("📦 Giá theo Top 10 thương hiệu")
    top_brands = df["brand"].value_counts().head(10).index.tolist()
    df_top = df[df["brand"].isin(top_brands)]
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.boxplot(x="brand", y="price", data=df_top, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_xlabel("Thương hiệu")
    ax2.set_ylabel("Giá")
    st.pyplot(fig2)

st.subheader("🔎 Giá vs Số km đã đi")
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.scatterplot(x=df["km"].replace({np.nan:0}), y=df["price"].replace({np.nan:0}), alpha=0.6, ax=ax3)
ax3.set_xlabel("Số km")
ax3.set_ylabel("Giá")
st.pyplot(fig3)

st.subheader("🧭 Ma trận tương quan (numeric features)")
corr = num_df.corr()
fig4, ax4 = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax4)
st.pyplot(fig4)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("**📌 Ghi chú:** Ứng dụng dùng TF-IDF + Cosine similarity.")
st.markdown("**📝 Học viên thực hiện:** Võ Thị Hoàng Anh — Nguyễn Mai Xuân Bách")

# ==============================
# Sidebar dẫn tới anchors
# ==============================
if menu == "Giới thiệu":
    st.markdown("<script>window.location.href='#intro';</script>", unsafe_allow_html=True)
elif menu == "Gợi ý theo tên xe":
    st.markdown("<script>window.location.href='#recommend';</script>", unsafe_allow_html=True)
elif menu == "Tìm kiếm tự do":
    st.markdown("<script>window.location.href='#search';</script>", unsafe_allow_html=True)
elif menu == "Phân cụm xe máy":
    st.markdown("<script>window.location.href='#cluster';</script>", unsafe_allow_html=True)
elif menu == "Thống kê & Trực quan":
    st.markdown("<script>window.location.href='#stats';</script>", unsafe_allow_html=True)
