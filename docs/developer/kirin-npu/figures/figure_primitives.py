"""教程矢量配图的排版与基础图元；只表达图形，不包含设备规格。"""
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".artifacts/kirin-npu-figures/mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath

plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "mobiinfer-kirin-npu-handbook-v2",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
INK, MUTED, LIGHT, PAPER = "#182F46", "#5E7184", "#E1E8EF", "#F5F8FB"
TEAL, BLUE, PURPLE, GOLD, CORAL = "#087F79", "#376FA8", "#785BA6", "#B1812F", "#BF664C"
COLORS = {"cpu": BLUE, "npu": TEAL, "tool": PURPLE, "file": GOLD, "warn": CORAL, "neutral": MUTED}
FILLS = {"cpu": "#EDF3FB", "npu": "#EAF5F1", "tool": "#F1EDF8", "file": "#FCF5E7",
         "warn": "#FBEEE8", "neutral": PAPER}


def configure_font(font_file):
    if font_file is None:
        names = {"Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                 "Microsoft YaHei", "SimHei", "PingFang SC"}
        font_file = next((Path(f.fname) for f in font_manager.fontManager.ttflist if f.name in names), None)
        cached = REPO / ".artifacts/kirin-npu-figures/fonts/NotoSansCJKsc-Regular.otf"
        if font_file is None and cached.is_file():
            font_file = cached
    if font_file is None or not font_file.is_file():
        raise ValueError("未找到中文字体，请使用 --font /path/to/chinese-font.otf 指定。")
    if ord("麒") not in FT2Font(str(font_file)).get_charmap():
        raise ValueError(f"字体不含中文字形：{font_file}")
    font_manager.fontManager.addfont(str(font_file))
    bold = font_file.with_name(font_file.name.replace("Regular", "Bold"))
    if bold != font_file and bold.is_file():
        font_manager.fontManager.addfont(str(bold))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_file)).get_name()


class Figure:
    def __init__(self, number, title, subtitle, topic, height=900):
        self.number, self.title, self.subtitle = number, title, subtitle
        self.width, self.height = 1200, height
        self.fig = plt.figure(figsize=(12, height / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.bounded = []
        self.text(54, 28, f"{number:02d}  /  {topic}", 12, TEAL, weight="bold")
        self.text(54, 69, title, 31, weight="bold")
        self.text(54, 110, subtitle, 16, MUTED)
        self.rule(54, height - 43, 1092)
        self.text(54, height - 22, "Mobiinfer-HarmonyOS  ·  麒麟 NPU 开发者教程", 11, MUTED)
        self.text(1146, height - 22, f"图 {number:02d}", 11, MUTED, ha="right")

    def text(self, x, y, text, size=18, color=INK, ha="left", va="center", weight="normal", **kwargs):
        return self.ax.text(x, y, text, fontsize=size * .72, color=color, ha=ha, va=va,
                            weight=weight, linespacing=1.4, zorder=12, **kwargs)

    def rule(self, x, y, w, color=LIGHT):
        self.line([(x, y), (x + w, y)], color, 1)

    def line(self, points, color=MUTED, width=1.5, dashed=False, z=4):
        x, y = zip(*points)
        self.ax.plot(x, y, color=color, linewidth=width, solid_capstyle="round",
                     linestyle=(0, (4, 3)) if dashed else "-", zorder=z)

    def arrow(self, points, color=MUTED, width=1.6, dashed=False, curve=False):
        codes = ([MplPath.MOVETO] + [MplPath.CURVE4] * 3 if curve else
                 [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        self.ax.add_patch(FancyArrowPatch(path=MplPath(points, codes), arrowstyle="-|>",
                                         mutation_scale=12, linewidth=width, color=color,
                                         linestyle=(0, (4, 3)) if dashed else "-", zorder=6))

    def rect(self, x, y, w, h, fill=PAPER, stroke=None, radius=10, lw=1, z=1, hatch=None):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=fill, edgecolor=stroke or "none", linewidth=lw,
                               zorder=z, hatch=hatch)
        self.ax.add_patch(patch)
        return patch

    def poly(self, points, fill, stroke=None, lw=1.2, z=3):
        self.ax.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=stroke or "none",
                                  linewidth=lw, joinstyle="round", zorder=z))

    def circle(self, x, y, r, fill="white", stroke=None, lw=1.4, z=6):
        self.ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=stroke or "none", linewidth=lw, zorder=z))

    def step(self, x, y, n, color=TEAL, r=15):
        self.circle(x, y, r, color)
        self.text(x, y - .3, str(n), 14, "white", ha="center", weight="bold")

    def label(self, x, y, title, body=None, color=INK, size=20):
        self.text(x, y, title, size, color, weight="bold")
        if body:
            self.text(x, y + 23, body, 15, MUTED, va="top")

    def tag(self, x, y, w, text, kind="npu", h=28):
        self.rect(x, y, w, h, FILLS[kind], radius=5, z=3)
        artist = self.text(x + w / 2, y + h / 2, text, 13, COLORS[kind], ha="center", weight="bold")
        self.bounded.append((artist, (x + 5, y + 2, w - 10, h - 4)))

    def dot_label(self, x, y, text, kind):
        self.circle(x, y, 4, COLORS[kind])
        self.text(x + 13, y, text, 14, MUTED)

    def file(self, x, y, w, h, kind="file", title=None):
        color, fill, fold = COLORS[kind], FILLS[kind], 24
        self.poly([(x, y), (x + w - fold, y), (x + w, y + fold), (x + w, y + h), (x, y + h)], fill, color, 1.3)
        self.poly([(x + w - fold, y), (x + w - fold, y + fold), (x + w, y + fold)], "white", color, 1)
        if title:
            self.text(x + 16, y + 30, title, 16, color, weight="bold")

    def graph(self, x, y, w, h, color=PURPLE):
        nodes = [(x, y + h / 2), (x + w * .34, y + h * .16), (x + w * .34, y + h * .84),
                 (x + w * .7, y + h / 2), (x + w, y + h / 2)]
        for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]:
            self.line([nodes[a], nodes[b]], color, 1.5)
        for a, b in nodes:
            self.circle(a, b, 7, "white", color, 1.4)

    def matrix(self, x, y, rows, cols, cell=12, fill=TEAL, gap=2, row_colors=None,
               col_colors=None, sparse=False, border=False):
        fills = ["#D2EAE2", "#91C7B5", "#4F9F8B", TEAL]
        for r in range(rows):
            for c in range(cols):
                tone = (row_colors[r] if row_colors else col_colors[c] if col_colors else
                        fills[(r * 3 + c * 2) % len(fills)] if sparse else fill)
                self.rect(x + c * (cell + gap), y + r * (cell + gap), cell, cell, tone,
                          stroke="white" if border else None, radius=1.2, z=5)

    def chip(self, x, y, w, h, color=TEAL):
        for p in range(8):
            px = x + 20 + p * (w - 40) / 7
            self.line([(px, y - 8), (px, y)], color, 2)
            self.line([(px, y + h), (px, y + h + 8)], color, 2)
        for p in range(4):
            py = y + 16 + p * (h - 32) / 3
            self.line([(x - 8, py), (x, py)], color, 2)
            self.line([(x + w, py), (x + w + 8, py)], color, 2)
        self.rect(x, y, w, h, "#143A3F", color, radius=8, lw=1.5, z=3)

    def dimension(self, x1, x2, y, text, color=MUTED, below=True):
        self.line([(x1, y - 4), (x1, y + 4), (x1, y), (x2, y), (x2, y - 4), (x2, y + 4)], color, 1)
        self.text((x1 + x2) / 2, y + (18 if below else -18), text, 13, color, ha="center")

    def note(self, x, y, w, text, kind="neutral", h=52):
        self.rect(x, y, w, h, FILLS[kind], radius=7)
        self.rect(x, y + 10, 3, h - 20, COLORS[kind], radius=1)
        artist = self.text(x + 18, y + h / 2, text, 15, COLORS[kind])
        self.bounded.append((artist, (x + 12, y + 4, w - 20, h - 8)))

    def save(self, output, preview=None):
        glyphs = FT2Font(font_manager.findfont(font_manager.FontProperties())).get_charmap()
        missing = {c for t in self.ax.texts for c in t.get_text() if not c.isspace() and ord(c) not in glyphs}
        if missing:
            raise ValueError(f"字体缺字：{''.join(sorted(missing))}")
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for artist in self.ax.texts:
            box = artist.get_window_extent(renderer)
            if box.x0 < 18 or box.x1 > self.width - 18 or box.y0 < 8 or box.y1 > self.height - 8:
                raise ValueError(f"文字超出画布：{artist.get_text()!r}")
        for artist, (x, y, w, h) in self.bounded:
            box = artist.get_window_extent(renderer)
            a, b = self.ax.transData.transform((x, y)), self.ax.transData.transform((x + w, y + h))
            if box.x0 < a[0] or box.x1 > b[0] or box.y0 < b[1] or box.y1 > a[1]:
                raise ValueError(f"文字超出标注：{artist.get_text()!r}")
        self.fig.savefig(output, format="svg", metadata={"Date": None,
            "Creator": "Mobiinfer-HarmonyOS 中文教学配图", "Title": self.title, "Description": self.subtitle})
        svg = output.read_text()
        svg = re.sub(r"<svg\b[^>]*>", lambda m: m.group(0)[:-1]
                     + ' role="img" xml:lang="zh-CN" aria-labelledby="figure-title figure-description">'
                     + f'\n<title id="figure-title">{escape(self.title)}</title>'
                     + f'\n<desc id="figure-description">{escape(self.subtitle)}</desc>', svg, count=1)
        output.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
        if preview:
            self.fig.savefig(preview, format="png", dpi=150)
        plt.close(self.fig)
