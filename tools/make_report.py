"""Tạo báo cáo PDF từ 4 notebook đã chạy (bai1..bai4.ipynb).

Mỗi bài gồm: bảng tham số khảo sát, rồi lần lượt các ô của notebook: mô tả (markdown) -> mã nguồn -> kết quả
(văn bản + hình) -> nhận xét. Cuối mỗi bài có trang "Ảnh chụp màn hình minh chứng": nếu trong thư mục screenshots/
có ảnh tên bai<k>_*.png|jpg thì ảnh được chèn vào, nếu chưa có thì để khung trống.

Dùng:  python tools/make_report.py [--name "Họ tên"] [--sid "Mã SV"] [--cls "Lớp"] [--link "URL notebook/Colab"]
Kết quả: BaoCao_XuLyAnh.pdf
"""
import argparse
import base64
import io
import re
import textwrap
from pathlib import Path

import nbformat
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "tools" / "fonts"
OUT = ROOT / "BaoCao_XuLyAnh.pdf"

pdfmetrics.registerFont(TTFont("Sans", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "JetBrainsMono-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Mono-Bold", str(FONTS / "JetBrainsMono-Bold.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")

S = {
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9, leading=13, spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=9, leading=13, leftIndent=14, bulletIndent=3, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=16, leading=20, spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#1a3c6e")),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12, leading=16, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1a3c6e")),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=10, leading=14, spaceBefore=8, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=6.0, leading=7.6, backColor=colors.HexColor("#f4f4f6"),
                           borderColor=colors.HexColor("#d0d0d8"), borderWidth=0.4, borderPadding=3, spaceBefore=3, spaceAfter=5),
    "out": ParagraphStyle("out", fontName="Mono", fontSize=6.0, leading=7.6, backColor=colors.HexColor("#fbf7e6"),
                          borderColor=colors.HexColor("#e0d9b0"), borderWidth=0.4, borderPadding=3, spaceBefore=3, spaceAfter=5),
    "label": ParagraphStyle("label", fontName="Sans-Bold", fontSize=7.5, leading=10, textColor=colors.HexColor("#666666"), spaceBefore=3),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=22, leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#1a3c6e")),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=12, leading=18, alignment=TA_CENTER),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10.5),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8, leading=10.5),
}
PAGE_W = A4[0] - 3.6 * cm        # bề rộng vùng nội dung

TEX = [(r"\\ominus", "⊖"), (r"\\oplus", "⊕"), (r"\\cap", "∩"), (r"\\cup", "∪"), (r"\\le(?![a-z])", "≤"), (r"\\ge(?![a-z])", "≥"),
       (r"\\sigma", "σ"), (r"\\mu", "μ"), (r"\\omega", "ω"), (r"\\theta", "θ"), (r"\\nabla", "∇"), (r"\\sum", "Σ"),
       (r"\\beta", "β"), (r"\\min", "min"), (r"\\max", "max"), (r"\\in", "∈"), (r"\\,", " "), (r"\\!", ""),
       (r"\\quad", "  "), (r"\\left|\\right", ""), (r"\\bar f", "f̄")]


def tex_to_text(t):
    """Chuyển công thức LaTeX đơn giản trong $...$ thành văn bản Unicode."""
    t = t.replace("\\dfrac", "\\frac").replace("\\{", "\x02").replace("\\}", "\x03").replace("\\|", "‖")
    t = re.sub(r"\\frac(\d)(\d)", r"(\1)/(\2)", t)
    t = re.sub(r"\\(left|right)", "", t)
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", t)
        t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    for pat, rep in TEX:
        t = re.sub(pat, rep, t)
    t = t.replace("^\\circ", "°").replace("\\", "")
    t = re.sub(r"_\{([^{}]*)\}", r"_\1", t)
    t = re.sub(r"\^\{([^{}]*)\}", r"^\1", t)
    return t.replace("{", "").replace("}", "").replace("\x02", "{").replace("\x03", "}")


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t):
    """Markdown inline -> markup của reportlab."""
    t = re.sub(r"\$([^$]+)\$", lambda m: "\x00" + tex_to_text(m.group(1)) + "\x01", t)
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="Mono" size="8">\1</font>', t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", t)
    return t.replace("\x00", "<i>").replace("\x01", "</i>")


def md_flowables(md):
    out, para = [], []

    def flush():
        if para:
            out.append(Paragraph(inline(" ".join(para)), S["body"]))
            para.clear()

    for line in md.split("\n"):
        st = line.strip()
        if not st:
            flush()
        elif st.startswith("### "):
            flush(); out.append(Paragraph(inline(st[4:]), S["h3"]))
        elif st.startswith("## "):
            flush(); out.append(Paragraph(inline(st[3:]), S["h2"]))
        elif st.startswith("# "):
            flush(); out.append(Paragraph(inline(st[2:]), S["h2"]))
        elif re.match(r"^([*-]) ", st):
            flush(); out.append(Paragraph(inline(st[2:]), S["bullet"], bulletText="•"))
        elif line.startswith("  ") and out and isinstance(out[-1], Paragraph) and out[-1].style.name == "bullet":
            out[-1] = Paragraph(out[-1].text + " " + inline(st), S["bullet"], bulletText="•")   # dòng nối của bullet
        else:
            para.append(st)
    flush()
    return out


def wrap_code(src, width=128):
    lines = []
    for l in src.rstrip().split("\n"):
        if len(l) <= width:
            lines.append(l)
        else:
            ind = len(l) - len(l.lstrip())
            parts = textwrap.wrap(l, width=width, subsequent_indent=" " * (ind + 4), break_long_words=True, replace_whitespace=False)
            lines.extend(parts)
    return "\n".join(lines)


def img_flowable(png_bytes, max_h=21 * cm):
    im = PILImage.open(io.BytesIO(png_bytes))
    w, h = im.size
    scale = min(PAGE_W / w, max_h / h)
    if im.mode == "RGBA":
        bg = PILImage.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); im = bg
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=88); buf.seek(0)
    return Image(buf, width=w * scale, height=h * scale)


def pre(text, style):
    return Preformatted(text, style)


def code_flowables(src, label="Mã nguồn"):
    return [Paragraph(label, S["label"]), pre(wrap_code(src), S["code"])]


def output_flowables(outputs):
    out = []
    for o in outputs:
        if o.output_type == "stream" and o.name == "stdout":
            txt = "".join(o.text) if isinstance(o.text, list) else o.text
            txt = "\n".join(l for l in txt.split("\n") if l.strip() != "" or True).rstrip()
            if txt.strip():
                out += [Paragraph("Kết quả (văn bản)", S["label"]), pre(wrap_code(txt, 132), S["out"])]
        elif o.output_type in ("display_data", "execute_result"):
            png = o.data.get("image/png")
            if png:
                out += [Paragraph("Kết quả (hình)", S["label"]), img_flowable(base64.b64decode(png)), Spacer(1, 4)]
    return out


PARAMS = {
    1: [("Chuyển không gian màu", "RGB -> HSV (OpenCV), HSI (tự cài đặt), CIELAB (skimage); hiển thị R,G,B / H,S,V / H,S,I / L*,a*,b*"),
        ("Phân đoạn 2 nhóm/ảnh", "coffee: sứ đỏ & gỗ; astronaut: bộ đồ cam & nền trắng-xám. Hộp ngưỡng RGB, HSV, HSI (chi tiết trong mã)"),
        ("Độ bền chiếu sáng", "ảnh tối ×0,6 và gamma 0,7, giữ nguyên ngưỡng, đo IoU"),
        ("Khảo sát 3 bộ ngưỡng HSV", "chặt / vừa / lỏng: coffee (H≤10,S≥0,85,V≥0,35 | H≤16,S≥0,6,V≥0,15 | H≤26,S≥0,4,V≥0,08); "
                                      "astronaut (H 10-20,S≥0,7,V≥0,65 | H 5-25,S≥0,55,V≥0,4 | H 0-40,S≥0,3,V≥0,2)")],
    2: [("Erosion/dilation/opening/closing", "tự lập trình NumPy; SE rect 3/15, ellipse 7, cross 5; so sánh OpenCV (bằng nhau từng pixel) + thời gian"),
        ("SE hình dạng", "rect, ellipse, cross, diamond, hline, vline (kích thước 7×7)"),
        ("SE kích thước", "ellipse 3, 7, 15, 25"),
        ("Thuật toán", "hình bao (SE rect3/cross3), làm đầy lỗ (SE cross3), thành phần liên thông (4/8), convex hull, thinning/skeletonize")],
    3: [("Sobel", "ksize 3/5/7; Gauss σ 0/1/2; ngưỡng 20-25% max"),
        ("Laplacian / LoG", "ksize 1/3/5; LoG σ 1/2/3; ngưỡng zero-crossing 4% max"),
        ("Canny", "(low,high) = 20/60, 50/150, 100/200, 150/250; σ 0/1/2/3; apertureSize 3/5; L2gradient"),
        ("Ngưỡng", "cố định T = 60/107/150; Otsu tự lập vs OpenCV vs skimage; adaptive MEAN/GAUSSIAN, block 11/31/101, C 2/10"),
        ("Nhiễu", "Gauss σ_n = 10/25/50; muối-tiêu 2%/5%; bộ lọc Gauss 5×5, median 5; đo F1 (dung sai 1 px) và Dice")],
    4: [("Region growing", "seed A-D; T = 5/10/20/40 (xám), T = 10/20/30/45 (Lab); mode mean/seed; 8-liên thông; Gauss σ=1"),
        ("K-means", "K = 2, 3, 4, 5 (xám & Lab), KMEANS_PP_CENTERS, attempts=5; RGB vs Lab; elbow K = 1..8"),
        ("SLIC", "n_segments 50/200/800 × compactness 1/10/40; sigma 0/1/3"),
        ("Felzenszwalb", "scale 50/200/800 × sigma 0,5/1,5; min_size 50"),
        ("So sánh", "số vùng, số thành phần liên thông, độ đồng nhất Lab, thời gian")],
}
TITLES = {1: "Bài 1. Phân tích và phân đoạn ảnh màu", 2: "Bài 2. Xử lý ảnh hình thái học",
          3: "Bài 3. Phát hiện biên và phân ngưỡng", 4: "Bài 4. Phân đoạn ảnh dựa trên vùng"}


def params_table(k):
    rows = [[Paragraph("Nội dung", S["cellb"]), Paragraph("Tham số khảo sát", S["cellb"])]]
    for a, b in PARAMS[k]:
        rows.append([Paragraph(esc(a), S["cell"]), Paragraph(esc(b), S["cell"])])
    t = Table(rows, colWidths=[4.2 * cm, PAGE_W - 4.2 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0d0")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf7")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def screenshot_page(k):
    fl = [PageBreak(), Paragraph(f"Ảnh chụp màn hình minh chứng - Bài {k}", S["h2"]),
          Paragraph("Chụp trên nền cửa sổ thư mục chứa mã sinh viên: (1) cửa sổ thư mục hiển thị tên thư mục/mã SV và file "
                    f"<font face='Mono'>bai{k}.ipynb</font>; (2) notebook đang chạy với kết quả hiển thị; (3) thời gian/tên tài khoản trên màn hình nếu có.", S["body"]), Spacer(1, 6)]
    files = sorted(p for p in (ROOT / "screenshots").glob(f"bai{k}_*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if files:
        for p in files:
            fl += [img_flowable(p.read_bytes(), max_h=11 * cm), Paragraph(p.name, S["cap"]), Spacer(1, 6)]
    else:
        box = Table([[Paragraph("[Chèn ảnh chụp màn hình tại đây]<br/><br/>Đặt file vào thư mục <b>screenshots/</b> với tên "
                                f"<font face='Mono'>bai{k}_1.png</font>, <font face='Mono'>bai{k}_2.png</font>... rồi chạy lại "
                                "<font face='Mono'>python tools/make_report.py</font>.", S["cap"])]], colWidths=[PAGE_W], rowHeights=[12 * cm])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        fl.append(box)
    return fl


def notebook_flowables(k):
    nb = nbformat.read(ROOT / f"bai{k}.ipynb", 4)
    fl = [PageBreak(), Paragraph(TITLES[k], S["h1"]), params_table(k), Spacer(1, 4)]
    first = True
    for c in nb.cells:
        if c.cell_type == "markdown":
            if first:   # ô tiêu đề đầu notebook: bỏ dòng '# Bài k...' (đã có tiêu đề), giữ phần mô tả
                src = "\n".join(l for l in c.source.split("\n") if not l.startswith("# "))
                first = False
                fl += md_flowables(src)
            else:
                fl += md_flowables(c.source)
        else:
            lab = "Mã nguồn - cài đặt và hàm tiện ích" if "Cài đặt & hàm tiện ích" in c.source else "Mã nguồn"
            fl += code_flowables(c.source, lab) + output_flowables(c.get("outputs", []))
    fl += screenshot_page(k)
    return fl


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.8 * cm, 1.0 * cm, "Xử lý ảnh số - Bài tập thực hành: phân đoạn, hình thái học, biên và ngưỡng")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Trang {doc.page}")
    canvas.restoreState()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="................................")
    ap.add_argument("--sid", default="................")
    ap.add_argument("--cls", default="................")
    ap.add_argument("--link", default="(điền đường dẫn Colab/Drive của bài làm)")
    a = ap.parse_args()

    story = [Spacer(1, 3.5 * cm), Paragraph("BÁO CÁO BÀI TẬP THỰC HÀNH", S["title"]), Spacer(1, 6),
             Paragraph("XỬ LÝ ẢNH SỐ", S["title"]), Spacer(1, 18),
             Paragraph("Phân tích và phân đoạn ảnh màu · Xử lý ảnh hình thái học<br/>Phát hiện biên và phân ngưỡng · Phân đoạn ảnh dựa trên vùng", S["sub"]),
             Spacer(1, 2.2 * cm)]
    info = Table([[Paragraph("Họ và tên", S["cellb"]), Paragraph(esc(a.name), S["cell"])],
                  [Paragraph("Mã sinh viên", S["cellb"]), Paragraph(esc(a.sid), S["cell"])],
                  [Paragraph("Lớp", S["cellb"]), Paragraph(esc(a.cls), S["cell"])],
                  [Paragraph("Notebook / mã nguồn", S["cellb"]), Paragraph(esc(a.link), S["cell"])],
                  [Paragraph("Công cụ", S["cellb"]), Paragraph("Python 3, NumPy, OpenCV, scikit-image, scikit-learn, Matplotlib (Google Colab / Jupyter)", S["cell"])]],
                 colWidths=[4.5 * cm, PAGE_W - 4.5 * cm])
    info.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0d0")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [info, Spacer(1, 1.5 * cm), Paragraph("Mục lục", S["h2"])]
    for k in (1, 2, 3, 4):
        story.append(Paragraph(f"{TITLES[k]}", S["body"]))
    story += [Spacer(1, 10),
              Paragraph("<b>Ghi chú về dữ liệu:</b> ảnh đầu vào là các ảnh mẫu chuẩn của thư viện scikit-image (coffee, astronaut, coins, camera, page) "
                        "được lưu trong thư mục <font face='Mono'>images/</font>; có thể thay bằng ảnh của bộ dữ liệu LMS cùng tên file mà không cần sửa mã. "
                        "Mỗi nội dung trình bày theo thứ tự: mô tả và tham số -> mã nguồn -> ảnh đầu vào / kết quả -> nhận xét.", S["body"])]
    for k in (1, 2, 3, 4):
        story += notebook_flowables(k)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                            title="Báo cáo Xử lý ảnh số", author=a.name)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("Đã tạo", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
