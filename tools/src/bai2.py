# %% [markdown]
# # Bài 2. Xử lý ảnh hình thái học
#
# **Nội dung:** (1) tự lập trình erosion / dilation / opening / closing bằng NumPy và đối chiếu với OpenCV;
# (2) thử nghiệm các phần tử cấu trúc (SE) khác hình dạng, kích thước; (3) cài đặt các thuật toán hình thái:
# trích xuất hình bao (boundary), làm đầy lỗ (hole filling), thành phần liên thông (connected components),
# convex hull và thinning.
#
# **Ảnh đầu vào:** `coins.png` (303×384, xám - ảnh mẫu scikit-image, đồng xu trên nền tối) và `page.png`
# (ảnh trang chữ in, dùng cho thinning). Ảnh nhị phân được tạo bằng ngưỡng Otsu.

# %% [markdown]
# ## 2.0 Ảnh đầu vào và ảnh nhị phân

# %%
coins = load_gray("coins.png", data.coins)
T_otsu, bw = cv2.threshold(coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # 0/255, uint8
print("Kích thước:", coins.shape, "| ngưỡng Otsu =", T_otsu, "| số điểm trắng =", int((bw > 0).sum()))

# ảnh nhị phân có thêm nhiễu muối-tiêu (dùng để thử opening/closing)
rng = np.random.default_rng(0)
noisy = bw.copy()
r = rng.random(bw.shape)
noisy[r < 0.01] = 255          # nhiễu muối (điểm trắng)
noisy[r > 0.99] = 0            # nhiễu tiêu (điểm đen)
show([coins, bw, noisy], ["Ảnh xám gốc coins.png", f"Nhị phân (Otsu T={int(T_otsu)})", "Nhị phân + nhiễu muối-tiêu 1%+1%"],
     ncols=3, cell=4, name="b2_input.png")

# %% [markdown]
# ## 2.1 Tự lập trình erosion, dilation, opening, closing bằng NumPy
#
# * Phần tử cấu trúc `se` (ma trận 0/1, gốc ở tâm). Với mỗi điểm 1 của `se` ta dịch ảnh một đoạn tương ứng rồi
#   lấy **min** (erosion) hoặc **max** (dilation) trên các ảnh đã dịch -> vector hóa toàn ảnh, không cần vòng lặp theo pixel.
# * Erosion: $(f\ominus B)(x)=\min_{s\in B} f(x+s)$; Dilation: $(f\oplus B)(x)=\max_{s\in B} f(x-s)$ (SE được **lật** qua gốc).
# * Biên ảnh: erosion đệm giá trị lớn nhất (255), dilation đệm 0 (giống mặc định của OpenCV).
# * Opening = dilation(erosion(f)); Closing = erosion(dilation(f)). Cùng một code chạy cho ảnh nhị phân (0/255) lẫn ảnh xám.
#
# **Tham số:** hình dạng và kích thước SE (`make_se(shape, size)`).

# %%
def make_se(shape, size):
    """Phần tử cấu trúc kích thước size x size (size lẻ)."""
    c = size // 2
    if shape == "rect":
        return np.ones((size, size), np.uint8)
    if shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    if shape == "diamond":
        yy, xx = np.ogrid[:size, :size]
        return (np.abs(yy - c) + np.abs(xx - c) <= c).astype(np.uint8)
    se = np.zeros((size, size), np.uint8)
    if shape == "hline":
        se[c, :] = 1
    elif shape == "vline":
        se[:, c] = 1
    else:
        raise ValueError(shape)
    return se


def se_img(se):
    """Ảnh minh họa SE: ô thuộc SE màu trắng, ô ngoài SE màu đen, viền xám để thấy rõ khung."""
    c = np.full((se.shape[0] + 2, se.shape[1] + 2), 90, np.uint8)
    c[1:-1, 1:-1] = se * 255
    return cv2.resize(c, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)


def _shifted(img, se, pad_value):
    """Sinh các bản dịch của ảnh tương ứng với từng điểm 1 của se (se có gốc ở tâm)."""
    kh, kw = se.shape
    cy, cx = kh // 2, kw // 2
    p = np.pad(img, ((cy, kh - 1 - cy), (cx, kw - 1 - cx)), constant_values=pad_value)
    H, W = img.shape
    for dy, dx in zip(*np.nonzero(se)):
        yield p[dy:dy + H, dx:dx + W]


def erosion(img, se):
    out = None
    for s in _shifted(img, se, 255):
        out = s.copy() if out is None else np.minimum(out, s)
    return out


def dilation(img, se):
    se_r = se[::-1, ::-1]                     # dilation dùng SE lấy đối xứng qua gốc
    out = None
    for s in _shifted(img, se_r, 0):
        out = s.copy() if out is None else np.maximum(out, s)
    return out


def opening(img, se):
    return dilation(erosion(img, se), se)


def closing(img, se):
    return erosion(dilation(img, se), se)


def timeit(fn, *a, rep=5):
    t = time.perf_counter()
    for _ in range(rep):
        r = fn(*a)
    return r, (time.perf_counter() - t) / rep * 1000   # ms

# %% [markdown]
# ### So sánh với OpenCV (ảnh nhị phân và ảnh xám, nhiều SE)
# `np.array_equal` = True nghĩa là kết quả **trùng khít từng pixel**.

# %%
ops = {
    "erosion":  (erosion,  lambda im, k: cv2.erode(im, k)),
    "dilation": (dilation, lambda im, k: cv2.dilate(im, k)),
    "opening":  (opening,  lambda im, k: cv2.morphologyEx(im, cv2.MORPH_OPEN, k)),
    "closing":  (closing,  lambda im, k: cv2.morphologyEx(im, cv2.MORPH_CLOSE, k)),
}
print(f"{'ảnh':8s} {'SE':12s} {'phép':9s} {'giống OpenCV':13s} {'t_NumPy(ms)':>12s} {'t_OpenCV(ms)':>13s}")
all_equal = True
for img_name, img in [("nhị phân", noisy), ("xám", coins)]:
    for shape, size in [("rect", 3), ("ellipse", 7), ("cross", 5), ("rect", 15)]:
        se = make_se(shape, size)
        for op, (mine, ref) in ops.items():
            a, t1 = timeit(mine, img, se)
            b, t2 = timeit(ref, img, se)
            eq = np.array_equal(a, b)
            all_equal &= eq
            print(f"{img_name:8s} {shape + str(size):12s} {op:9s} {str(eq):13s} {t1:12.2f} {t2:13.3f}")
print("\n=> Tất cả trường hợp trùng OpenCV:", all_equal)

# SE bất đối xứng: dilation của ta (SE được lật) = cv2.dilate với SE đã lật
se_asym = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], np.uint8)
print("SE bất đối xứng - dilation trùng cv2.dilate(SE lật):",
      np.array_equal(dilation(bw, se_asym), cv2.dilate(bw, se_asym[::-1, ::-1])))

# %%
se = make_se("ellipse", 7)
res = [noisy, erosion(noisy, se), dilation(noisy, se), opening(noisy, se), closing(noisy, se)]
cvr = [noisy, cv2.erode(noisy, se), cv2.dilate(noisy, se), cv2.morphologyEx(noisy, cv2.MORPH_OPEN, se),
       cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, se)]
tt = ["Ảnh vào (nhiễu)", "Erosion", "Dilation", "Opening", "Closing"]
show(res + [np.abs(a.astype(int) - b.astype(int)).astype(np.uint8) * 255 for a, b in zip(res[1:], cvr[1:])],
     [t + " - tự lập trình" for t in tt] + [f"|Tự lập trình - OpenCV| ({t})" for t in tt[1:]],
     ncols=5, cell=3.2, name="b2_vs_opencv.png")
print("Tổng sai khác tuyệt đối so với OpenCV:", [int(np.abs(a.astype(int) - b.astype(int)).sum()) for a, b in zip(res[1:], cvr[1:])])

# %% [markdown]
# **Nhận xét 2.1**
# * Cả 32 trường hợp thử (2 ảnh: nhị phân và xám × 4 SE × 4 phép) cho kết quả **trùng OpenCV từng pixel** (tổng sai khác tuyệt đối = 0). Với SE bất đối xứng, dilation phải dùng SE đã lật qua gốc (định nghĩa chuẩn); khi đó mới trùng `cv2.dilate` với SE lật - đã kiểm chứng.
# * Cách vector hóa (min/max trên các bản dịch) chạy được cho cả ảnh nhị phân và ảnh xám với cùng một đoạn mã. Thời gian NumPy 0,2 - 6 ms, OpenCV 0,03 - 0,15 ms, tức chậm hơn khoảng 6 - 50 lần: chi phí tăng tuyến tính theo số điểm của SE (rect 3×3 = 9 điểm ≈ 0,2 - 0,4 ms; rect 15×15 = 225 điểm ≈ 3 - 6 ms), còn OpenCV dùng SIMD và phân tách SE chữ nhật. Opening/closing tốn gấp đôi erosion/dilation.

# %% [markdown]
# ## 2.2 Thử nghiệm phần tử cấu trúc khác hình dạng và kích thước
#
# **(a)** Cố định kích thước 7, đổi hình dạng SE (rect / ellipse / cross / diamond / hline / vline).
# **(b)** Cố định hình ellipse, đổi kích thước 3, 7, 15, 25. Ảnh vào là ảnh nhị phân có nhiễu muối-tiêu.

# %%
shapes = ["rect", "ellipse", "cross", "diamond", "hline", "vline"]
imgs, tits = [], []
for sh in shapes:
    se = make_se(sh, 7)
    imgs += [se_img(se), erosion(noisy, se), dilation(noisy, se), opening(noisy, se), closing(noisy, se)]
    tits += [f"SE {sh} 7x7", "erosion", "dilation", "opening", "closing"]
show(imgs, tits, ncols=5, cell=2.6, name="b2_se_shapes.png")

# %%
sizes = [3, 7, 15, 25]
imgs, tits = [], []
for sz in sizes:
    se = make_se("ellipse", sz)
    imgs += [se_img(se), erosion(noisy, se), dilation(noisy, se), opening(noisy, se), closing(noisy, se)]
    tits += [f"SE ellipse {sz}x{sz}", "erosion", "dilation", "opening", "closing"]
show(imgs, tits, ncols=5, cell=2.6, name="b2_se_sizes.png")

# bảng định lượng: diện tích vùng trắng (so với ảnh nhị phân sạch) và số thành phần liên thông
clean_area = int((bw > 0).sum())
n_clean = cv2.connectedComponents(bw, connectivity=8)[0] - 1
print(f"Ảnh sạch: diện tích = {clean_area}, số thành phần = {n_clean}")
print(f"{'SE':14s} {'phép':9s} {'diện tích':>10s} {'số TP liên thông':>17s} {'sai khác so với ảnh sạch (px)':>31s}")
for sh, sz in [("rect", 3), ("ellipse", 7), ("ellipse", 15), ("ellipse", 25), ("cross", 7), ("hline", 7)]:
    se = make_se(sh, sz)
    for op in ("opening", "closing"):
        o = (opening if op == "opening" else closing)(noisy, se)
        ncc = cv2.connectedComponents(o, connectivity=8)[0] - 1
        print(f"{sh + str(sz):14s} {op:9s} {int((o > 0).sum()):10d} {ncc:17d} {int((o != bw).sum()):31d}")

# %% [markdown]
# **Nhận xét 2.2**
# * **Hình dạng SE (7×7):** `rect` làm điểm nhiễu nở thành khối vuông góc nhọn; `ellipse` và `diamond` cho đốm tròn/thoi mượt hơn, ít làm méo hình tròn của đồng xu; `cross` có ít điểm nhất nên tác động nhẹ nhất (nhiễu nở thành dấu cộng mảnh); `hline`/`vline` có tính **định hướng**: nhiễu bị kéo thành vạch ngang/dọc và phép toán chỉ ảnh hưởng theo một phương (vd. erosion bằng `hline` chỉ cắt nét hẹp theo chiều ngang, các vệt xuất hiện song song nhau trong đồng xu). Chọn SE nên bám theo hình dạng đối tượng cần giữ.
# * **Kích thước SE (ellipse) - opening:** 3×3: còn 36 thành phần, diện tích 42 772 px (lệch 2 395 px so với ảnh sạch) - diệt nhiễu muối tốt mà ít hỏng đối tượng; 7×7: 39 thành phần (lệch 7 023 px); 15×15: diện tích chỉ còn 16 097 px (lệch 29 020 px) vì mảng sáng nhỏ của đồng xu bị loại; 25×25: chỉ còn 3 thành phần (2 177 px) - SE lớn hơn đa số đối tượng thì opening xóa chúng.
# * **Closing:** SE 3×3, 7×7 vẫn còn 583 và 464 thành phần vì closing chỉ lấp nhiễu tiêu (điểm đen), **không loại được nhiễu muối** (điểm trắng) - đối ngẫu của opening; 15×15: diện tích phình ra 72 610 px; 25×25: dính toàn bộ đồng xu và nền thành 1 thành phần (108 987 px). Kết luận: SE phải *lớn hơn nhiễu nhưng nhỏ hơn chi tiết cần giữ*; dùng opening để loại nhiễu muối/vật nhỏ, closing để lấp lỗ/khe nhỏ, và mở rồi đóng (opening -> closing) để xử lý nhiễu muối-tiêu.

# %% [markdown]
# ## 2.3 Các thuật toán hình thái học
#
# Ảnh nhị phân đã làm sạch: `clean = closing(opening(noisy, ellipse 5), ellipse 5)`; đây là đầu vào cho các thuật toán dưới đây.
# Tự lập trình: **trích xuất hình bao**, **làm đầy lỗ**, **thành phần liên thông**. Dùng thư viện để đối chiếu:
# **convex hull**, **thinning**.

# %%
se5 = make_se("ellipse", 5)
clean = closing(opening(noisy, se5), se5)
show([noisy, clean], ["Đầu vào có nhiễu", "Ảnh sạch sau opening + closing (ellipse 5)"], ncols=2, cell=4.5, name="b2_clean.png")

# %% [markdown]
# ### (1) Trích xuất hình bao (boundary extraction)
# $\beta(A)=A-(A\ominus B)$ với $B$ là SE 3×3 (hình bao dày 1 px). Tham số: SE `rect 3` (8-liên thông) và `cross 3` (4-liên thông).

# %%
def boundary(A, se):
    return cv2.subtract(A, erosion(A, se))      # A - (A erosion B)

b8 = boundary(clean, make_se("rect", 3))
b4 = boundary(clean, make_se("cross", 3))
# đối chiếu: hình bao trong của scikit-image
ref8 = segmentation.find_boundaries(clean > 0, connectivity=2, mode="inner")
ref4 = segmentation.find_boundaries(clean > 0, connectivity=1, mode="inner")
print("Số pixel hình bao: SE rect3 =", int((b8 > 0).sum()), "| SE cross3 =", int((b4 > 0).sum()))
print("Khác biệt với skimage.find_boundaries (mode=inner): rect3 ->", int(((b8 > 0) != ref8).sum()),
      "px, cross3 ->", int(((b4 > 0) != ref4).sum()), "px")
overlay = cv2.cvtColor(coins, cv2.COLOR_GRAY2RGB)
overlay[b8 > 0] = (255, 0, 0)
show([clean, b8, b4, overlay], ["Ảnh sạch A", "Hình bao (SE rect 3x3)", "Hình bao (SE cross 3x3)", "Hình bao chồng lên ảnh gốc"],
     ncols=4, cell=4, name="b2_boundary.png")

# %% [markdown]
# ### (2) Làm đầy lỗ (hole filling)
# Cách tự động (không cần chọn điểm gốc): nền liên thông với biên ảnh được dựng lại bằng giãn có điều kiện
# $X_k=(X_{k-1}\oplus B)\cap A^c$ với $X_0$ = các điểm nền nằm trên biên ảnh, $B$ = SE chữ thập (4-liên thông).
# Các lỗ = $A^c - X_\infty$; ảnh đã đầy = $A \cup$ lỗ. Lặp đến khi hội tụ.

# %%
def fill_holes(A, se=None):
    se = make_se("cross", 3) if se is None else se
    Ac = 255 - A
    X = np.zeros_like(A)
    X[0, :], X[-1, :], X[:, 0], X[:, -1] = Ac[0, :], Ac[-1, :], Ac[:, 0], Ac[:, -1]   # nền chạm biên ảnh
    it = 0
    while True:
        Xn = cv2.bitwise_and(dilation(X, se), Ac)
        it += 1
        if np.array_equal(Xn, X):
            break
        X = Xn
    holes = cv2.subtract(Ac, X)
    return cv2.bitwise_or(A, holes), holes, it

from scipy import ndimage as ndi
filled, holes, n_it = fill_holes(clean)
ref = ndi.binary_fill_holes(clean > 0)
print(f"Số vòng lặp hội tụ: {n_it} | diện tích lỗ = {int((holes > 0).sum())} px | "
      f"khác scipy.ndimage.binary_fill_holes: {int(((filled > 0) != ref).sum())} px")
show([clean, holes, filled], ["Ảnh sạch A (đồng xu còn lỗ)", "Các lỗ tìm được", "Sau làm đầy lỗ"], ncols=3, cell=4.5, name="b2_fill.png")

# %% [markdown]
# ### (3) Thành phần liên thông (connected components)
# Lấy một điểm chưa gán nhãn làm hạt $X_0$, lặp $X_k=(X_{k-1}\oplus B)\cap A$ đến khi hội tụ; $X_\infty$ là một thành phần.
# Lặp cho đến hết điểm trắng. $B$ = `rect 3x3` (8-liên thông) hoặc `cross 3x3` (4-liên thông). Đối chiếu `cv2.connectedComponents`.

# %%
def connected_components(A, se):
    remaining = A > 0
    labels = np.zeros(A.shape, np.int32)
    n = 0
    while remaining.any():
        ys, xs = np.nonzero(remaining)
        X = np.zeros_like(A)
        X[ys[0], xs[0]] = 255
        while True:
            Xn = cv2.bitwise_and(dilation(X, se), A)
            if np.array_equal(Xn, X):
                break
            X = Xn
        n += 1
        labels[X > 0] = n
        remaining &= ~(X > 0)
    return n, labels

def same_partition(l1, l2):
    """Hai ảnh nhãn là một phép phân hoạch giống nhau (không phụ thuộc số thứ tự nhãn)."""
    pairs = set(zip(l1.ravel().tolist(), l2.ravel().tolist()))
    return len(pairs) == len(set(p[0] for p in pairs)) == len(set(p[1] for p in pairs))

# ảnh thử: ảnh sạch đã làm đầy lỗ, thêm vài vật nhỏ chạm chéo để thấy khác biệt 4/8-liên thông
test = filled.copy()
free = np.argwhere(erosion(255 - filled, make_se("rect", 9)) > 0)   # vùng nền trống 9x9
y0, x0 = free[len(free) // 2]
test[y0:y0 + 3, x0:x0 + 3] = 255
test[y0 + 3:y0 + 6, x0 + 3:x0 + 6] = 255   # hai khối 3x3 chỉ chạm nhau ở góc chéo
t0 = time.perf_counter()
n8, lab8 = connected_components(test, make_se("rect", 3))
n4, lab4 = connected_components(test, make_se("cross", 3))
t_my = time.perf_counter() - t0
cv8 = cv2.connectedComponents(test, connectivity=8)
cv4 = cv2.connectedComponents(test, connectivity=4)
print(f"Tự lập trình: 8-liên thông = {n8} TP, 4-liên thông = {n4} TP   (thời gian {t_my:.2f}s)")
print(f"Hai khối chéo đặt tại (y={y0}, x={x0}): 8-conn gộp thành 1 TP, 4-conn tách thành 2 TP")
print(f"OpenCV      : 8-liên thông = {cv8[0]-1} TP, 4-liên thông = {cv4[0]-1} TP")
print("Phân hoạch giống OpenCV: 8-conn =", same_partition(lab8, cv8[1]), "| 4-conn =", same_partition(lab4, cv4[1]))
props = measure.regionprops(lab8)
print("Diện tích các TP (8-conn):", sorted(int(p.area) for p in props))
rgb8 = (color.label2rgb(lab8, bg_label=0, bg_color=(0, 0, 0)) * 255).astype(np.uint8)
show([test, rgb8, (color.label2rgb(lab4, bg_label=0, bg_color=(0, 0, 0)) * 255).astype(np.uint8)],
     ["Ảnh đầu vào", f"Nhãn 8-liên thông ({n8} thành phần)", f"Nhãn 4-liên thông ({n4} thành phần)"], ncols=3, cell=4.5, name="b2_cc.png")

# %% [markdown]
# ### (4) Convex hull
# Với mỗi thành phần liên thông, so sánh `skimage.morphology.convex_hull_image` và `cv2.convexHull`; độ đặc
# (solidity) = diện tích / diện tích bao lồi.

# %%
hull_sk = np.zeros(test.shape, bool)
hull_cv = np.zeros(test.shape, np.uint8)
sol = []
for p in props:
    m = lab8 == p.label
    h = morphology.convex_hull_image(m)
    hull_sk |= h
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack(cnt)
    cv2.fillPoly(hull_cv, [cv2.convexHull(pts)], 255)
    sol.append(m.sum() / h.sum())
inter = (hull_sk & (hull_cv > 0)).sum()
union = (hull_sk | (hull_cv > 0)).sum()
print(f"IoU giữa hai cách tính convex hull = {inter/union:.4f}")
print("Solidity từng thành phần: min = %.3f, max = %.3f, TB = %.3f" % (min(sol), max(sol), np.mean(sol)))
ov = cv2.cvtColor(test, cv2.COLOR_GRAY2RGB)
ov[hull_sk & (test == 0)] = (120, 160, 255)
show([test, hull_sk, hull_cv, ov], ["Thành phần gốc", "Convex hull (skimage)", "Convex hull (OpenCV)", "Phần hull thêm vào (xanh)"],
     ncols=4, cell=3.8, name="b2_hull.png")

# %% [markdown]
# ### (5) Thinning (làm mảnh) - ảnh chữ in `page.png`
# Ngưỡng thích nghi tách nét chữ, sau đó `skimage.morphology.thin` (thinning) và `skeletonize` (xương) để so sánh.

# %%
page = load_gray("page.png", data.page)
text = cv2.resize(page, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)        # phóng 3x để nét chữ đủ dày
tbw = cv2.adaptiveThreshold(text, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 12)
n_cc, cc, st, _ = cv2.connectedComponentsWithStats(tbw, connectivity=8)
keep = st[:, cv2.CC_STAT_AREA] >= 30
keep[0] = False                                                                   # nhãn 0 = nền
tbw = keep[cc].astype(np.uint8) * 255                                             # bỏ hạt nhiễu < 30 px
thin = morphology.thin(tbw > 0)
skel = morphology.skeletonize(tbw > 0)
print("Số pixel nét chữ:", int((tbw > 0).sum()), "| sau thinning:", int(thin.sum()), "| skeletonize:", int(skel.sum()))
print("Tỉ lệ giữ lại: thinning = %.1f%%, skeleton = %.1f%%" % (100 * thin.sum() / (tbw > 0).sum(), 100 * skel.sum() / (tbw > 0).sum()))
crop = (slice(0, 135), slice(0, 690))
show([text[crop], tbw[crop], thin[crop], skel[crop]], ["Ảnh gốc page.png (phóng 3x, vùng tiêu đề)", "Nhị phân nét chữ (adaptive 51/12)", "Thinning", "Skeletonize"],
     ncols=2, cell=7, name="b2_thinning.png")

# %% [markdown]
# **Nhận xét 2.3**
# * **Hình bao:** SE `rect 3×3` cho 5 364 px (đường biên dày, 4-liên thông), SE `cross 3×3` cho 3 897 px (mỏng hơn, 8-liên thông). Cả hai trùng 100% (0 pixel khác biệt) với `skimage.segmentation.find_boundaries(mode='inner')` tương ứng.
# * **Làm đầy lỗ:** thuật toán dựng nền từ biên ảnh hội tụ sau 205 vòng, lấp 956 px lỗ (do vân đồng xu thấp hơn ngưỡng Otsu) và trùng 100% với `scipy.ndimage.binary_fill_holes`. Số vòng lớn vì nền lan 1 px mỗi vòng với SE chữ thập; không cần chọn điểm hạt cho từng lỗ.
# * **Thành phần liên thông:** tự lập trình cho 26 thành phần (8-liên thông) và 27 thành phần (4-liên thông) - hơn 1 vì hai khối 3×3 đặt chạm nhau ở góc chéo chỉ dính nhau theo 8-liên thông; phân hoạch trùng `cv2.connectedComponents` ở cả hai kiểu. 23 thành phần có diện tích 1 074 - 3 024 px là các đồng xu; thành phần 8 473 px là vùng nền sáng ở mép trên **dính cả đồng xu góc trên-trái** (do ngưỡng Otsu); 18 px là cặp khối thử và 23 px là một hạt nhỏ. Bản morphological mất ~0,7 s cho hai lượt (lặp giãn theo từng thành phần) - chậm hơn nhiều so với duyệt hai lượt của OpenCV nhưng cho đúng kết quả.
# * **Convex hull:** hai cách tính (skimage, OpenCV) trùng IoU = 0,997. Độ đặc (solidity) trung bình 0,929 (0,636 - 0,972): đồng xu gần lồi nên cao; vùng nền sáng mép trên có hình lõm nên thấp nhất (0,636); đồng xu bị khuyết/có vết lõm thấp hơn các đồng xu tròn.
# * **Thinning (trang chữ in, phóng 3×):** 84 224 px nét chữ -> 14 104 px (16,7%) với `thin` và 14 489 px (17,2%) với `skeletonize`. Cả hai cho nét dày 1 px, giữ được hình dạng và tính liên thông của chữ; skeleton có thêm vài nhánh cụt nhỏ nên nhiều điểm hơn chút. Kết quả phụ thuộc chất lượng nhị phân hóa (ảnh gốc chữ mảnh nên phải phóng 3× và dùng adaptive threshold, nếu dùng opening 3×3 như thử ban đầu thì các nét chữ mỏng bị xóa).
