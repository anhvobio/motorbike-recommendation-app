# 🏍️ Motorbike Recommendation & Clustering App

### 🌐 Demo App (Streamlit)
👉 https://motorbike-recommendation-app-vha.streamlit.app/

---

## 📁 Thông tin ứng dụng  
**Project 2 — Gợi ý & Phân khúc xe máy**  
**Phiên bản:** 1.0  
**Cập nhật:** 21/11/2025  

Ứng dụng xây dựng hệ thống **gợi ý xe máy tương tự** và **phân khúc thị trường xe máy** dựa trên dữ liệu thật từ Chợ Tốt.  
Dữ liệu bao gồm:

- Tiêu đề bài đăng  
- Mô tả chi tiết  
- Thương hiệu  
- Giá bán  
- Số km đã đi  
- Năm đăng ký  
- Tình trạng xe  
- Link bài đăng gốc  

---

## 🏍️ Giới thiệu

Dự án gồm 2 thành phần chính:

### 1️⃣ Gợi ý xe máy tương tự  
- Sử dụng **TF-IDF + Cosine Similarity**  
- Dựa trên **title + brand + description**  
- Giúp người dùng tìm xe giống nhất theo nhu cầu  

### 2️⃣ Phân khúc xe máy  
- Dùng **KMeans clustering** trên TF-IDF  
- Phát hiện nhóm xe tương đồng theo:  
  - phân khúc giá  
  - thương hiệu  
  - mô tả xe  

🎯 **Mục tiêu dự án**  
- Hỗ trợ người mua tìm xe phù hợp nhanh chóng  
- Giúp người bán định giá hợp lý  
- Phân tích thị trường xe máy cũ tại TP.HCM  

---

## 🖥️ Chức năng trong ứng dụng GUI Streamlit

### 🔹 Giới thiệu
Tổng quan dự án và cách sử dụng.

### 🔹 Gợi ý theo tên xe  
Chọn một xe trong danh sách và hệ thống sẽ trả về:  
- Top xe tương tự  
- Tỷ lệ tương đồng  
- Thông tin chi tiết  

### 🔹 Tìm kiếm tự do  
Người dùng có thể nhập từ khóa như:  
- “Air Blade 2020”,  
- “150cc”,  
- “Honda chính chủ”,  
- …  

→ Hệ thống trả về danh sách phù hợp nhất.

### 🔹 Phân cụm xe máy  
- Chọn số cụm K (2 → 20)  
- Hiển thị các nhóm xe theo phân khúc  

### 🔹 Thống kê & Trực quan  
- 📊 Phân phối giá  
- 📦 Giá theo thương hiệu (Top 10)  
- 🔎 Giá vs số km đã đi  
- 🧭 Ma trận tương quan  

📌 Ghi chú: Ứng dụng sử dụng TF-IDF + Cosine similarity cho phần gợi ý.

---

## 👥 Nhóm thực hiện  
- **Võ Thị Hoàng Anh**  
- **Nguyễn Mai Xuân Bách**  

---

## 📦 Cấu trúc repository

```
motorbike-recommendation-app/
│── GUI.py
│── GUI_project2.py
│── bigram_phraser.pkl
│── dictionary.gensim
│── xe_cosine_sim_18112025.pkl
│── du_lieu_xe_may_da_tien_xu_ly_1.xlsx
│── requirements.txt
│── README.md
```

---

## 🚀 Cách chạy ứng dụng trên VS Code

### 1️⃣ Tạo môi trường ảo (khuyến nghị)
```
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Cài đặt thư viện
```
pip install -r requirements.txt
```

### 3️⃣ Chạy Streamlit
```
streamlit run GUI_project2.py
```

---

## 🔧 Công nghệ sử dụng

- Python  
- Streamlit  
- Scikit-learn (TF-IDF, KMeans)  
- Gensim (bigram, dictionary)  
- Pandas / NumPy  
- Matplotlib / Seaborn  

