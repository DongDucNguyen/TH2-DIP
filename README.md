# Bài tập Xử lý ảnh số: phân đoạn màu, hình thái học, biên & ngưỡng, phân đoạn vùng

| Bài | Notebook | Nội dung |
|---|---|---|
| 1 | `bai1.ipynb` | RGB→HSV/HSI/CIELAB, phân đoạn theo màu (RGB vs HSV/HSI), khảo sát 3 bộ ngưỡng |
| 2 | `bai2.ipynb` | Erosion/dilation/opening/closing tự lập trình (so với OpenCV), SE khác nhau, hình bao, làm đầy lỗ, thành phần liên thông, convex hull, thinning |
| 3 | `bai3.ipynb` | Sobel, Laplacian/LoG, Canny; ngưỡng cố định, Otsu tự lập, adaptive; ảnh hưởng của nhiễu |
| 4 | `bai4.ipynb` | Region growing tự lập, K-means (K=2..5), SLIC, Felzenszwalb, so sánh |

Các notebook đã chạy sẵn (có kết quả). Báo cáo PDF: `BaoCao_XuLyAnh.pdf` (mã nguồn + ảnh vào/ra + tham số + nhận xét).

## Chạy
- **Colab:** mở notebook từ GitHub (Colab → File → Open → GitHub), chạy từng ô. Thư viện có sẵn; ảnh mẫu tự tải về `images/`.
- **Máy cá nhân:** `pip install -r requirements.txt` rồi `jupyter lab`.

## Ảnh đầu vào
Hiện dùng ảnh mẫu scikit-image (`coffee, astronaut, coins, camera, page`) lưu ở `images/`.
**Dùng ảnh LMS:** chép ảnh vào `images/` trùng tên file (hoặc đổi tên file ở ô "Ảnh đầu vào"), rồi chạy lại. Nhận xét trong notebook ghi theo số liệu của ảnh mẫu, nên nếu đổi ảnh cần xem lại các con số/nhận xét.

## Hoàn thiện nộp bài
1. Chụp màn hình (trên nền thư mục chứa mã sinh viên) và lưu vào `screenshots/` tên `bai1_1.png`, `bai1_2.png`, `bai2_1.png`...
2. Dựng lại PDF:
   ```bash
   python tools/make_report.py --name "Họ tên" --sid "MãSV" --cls "Lớp" --link "URL notebook"
   ```
3. Chia sẻ quyền truy cập/download và nộp đường dẫn lên LMS.

Dựng lại notebook từ nguồn: `python tools/build_notebooks.py --run` (nguồn ở `tools/src/`).
