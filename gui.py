"""
gui.py  —  Cairo Smart Transportation System — Interactive GUI
A unique brutalist-meets-dashboard design: bold typography, colour-coded
algorithm cards, a live map canvas, and dark-mode charts.

Run with:  python main.py --gui
"""

import sys
import os
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
random.seed(42)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QFrame, QSplitter,
    QScrollArea, QSizePolicy, QTabWidget, QSlider, QSpinBox,
    QGridLayout, QStackedWidget, QProgressBar, QGroupBox
)
from PyQt5.QtCore  import Qt, QTimer, QPoint, QRect
from PyQt5.QtGui   import (QPainter, QPen, QBrush, QColor, QFont,
                            QLinearGradient, QPainterPath, QPolygonF)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data.cairo_data  import build_cairo_graph
from algorithims.mst      import kruskal_mst
from algorithims.dijkstra  import shortest_path, cached_shortest_path, clear_cache
from algorithims.astar     import emergency_route
from algorithims.dp        import (road_maintenance_knapsack, MAINTENANCE_ROADS,
                                   transit_scheduling, TRANSIT_ROUTES)
from algorithims.greedy    import optimize_all_intersections, analyze_greedy_optimality, INTERSECTIONS


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS  —  one place to change the whole look
# ═══════════════════════════════════════════════════════════════════════════════
BG       = "#08090c"   # Near-black background
BG2      = "#0e1117"   # Slightly lighter card background
BG3      = "#141821"   # Raised surfaces / inputs
BORDER   = "#1e2535"   # Subtle borders

# Algorithm accent colours — each algorithm gets its own identity
C_MST    = "#00e5b4"   # Teal-green for MST
C_DIJ    = "#4da6ff"   # Electric blue for Dijkstra
C_ASTAR  = "#ff6b6b"   # Coral-red for A* / emergency
C_DP     = "#ffcc44"   # Amber for Dynamic Programming
C_GRD    = "#c77dff"   # Purple for Greedy
C_INFO   = "#7ecbff"   # Light blue for info

TEXT     = "#e4e9f2"   # Primary text
TEXT2    = "#5d6b82"   # Secondary / muted text
TEXT3    = "#2a3145"   # Very muted (dividers)


def style_app():
    """Global stylesheet — applied to the whole application."""
    return f"""
    * {{
        font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Arial', sans-serif;
    }}
    QMainWindow, QWidget {{
        background: {BG};
        color: {TEXT};
    }}
    QLabel {{
        color: {TEXT};
        background: transparent;
    }}
    QTextEdit {{
        background: {BG3};
        color: #b0ffc8;
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
        font-size: 11px;
        padding: 10px 12px;
        selection-background-color: {C_DIJ};
    }}
    QComboBox {{
        background: {BG3};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 6px 12px;
        min-height: 32px;
        font-size: 12px;
    }}
    QComboBox:hover {{ border-color: {C_DIJ}; }}
    QComboBox:focus {{ border-color: {C_DIJ}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{
        background: {BG3};
        color: {TEXT};
        border: 1px solid {BORDER};
        selection-background-color: rgba(77,166,255,0.2);
    }}
    QSpinBox {{
        background: {BG3};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 5px 10px;
    }}
    QSpinBox:hover {{ border-color: {C_DP}; }}
    QScrollBar:vertical {{
        background: transparent;
        width: 5px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 2px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {TEXT3}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 5px; }}
    QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 2px; }}
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        background: {BG2};
        border-radius: 8px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {TEXT2};
        padding: 8px 20px;
        border: none;
        border-bottom: 2px solid transparent;
        margin-right: 2px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    QTabBar::tab:selected {{ border-bottom-color: {C_DIJ}; color: {TEXT}; }}
    QTabBar::tab:hover {{ color: {TEXT}; }}
    QTabBar {{ background: transparent; border: none; }}
    QToolTip {{
        background: {BG3};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 11px;
    }}
    """


# ═══════════════════════════════════════════════════════════════════════════════
# REUSABLE WIDGET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def pill_label(text, color):
    """Coloured tag badge — used for algorithm labels, status, etc."""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        background: rgba({_hex_to_rgb(color)}, 0.12);
        color: {color};
        border: 1px solid rgba({_hex_to_rgb(color)}, 0.35);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
    """)
    lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    return lbl

def _hex_to_rgb(hex_color):
    """Convert #rrggbb to 'r, g, b' string for rgba() CSS."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"

def action_btn(text, color, small=False):
    """Solid action button — prominent CTA style."""
    btn = QPushButton(text)
    pad = "7px 16px" if small else "10px 22px"
    fs  = "11px"     if small else "12px"
    text_c = "#09090c" if color in (C_MST, C_DP) else TEXT
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {color};
            color: {text_c};
            border: none;
            border-radius: 6px;
            padding: {pad};
            font-size: {fs};
            font-weight: 700;
            letter-spacing: 0.2px;
        }}
        QPushButton:hover {{
            background: {color}cc;
        }}
        QPushButton:pressed {{ opacity: 0.7; }}
        QPushButton:disabled {{
            background: {BG3};
            color: {TEXT3};
        }}
    """)
    return btn

def ghost_btn(text, color):
    """Outlined ghost button — secondary action."""
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: transparent;
            color: {color};
            border: 1px solid rgba({_hex_to_rgb(color)}, 0.4);
            border-radius: 6px;
            padding: 8px 18px;
            font-size: 11px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: rgba({_hex_to_rgb(color)}, 0.1);
            border-color: {color};
        }}
    """)
    return btn

def card(parent=None, accent=None):
    """Base card frame — optionally highlighted with an accent left border."""
    f = QFrame(parent)
    border_left = f"border-left: 3px solid {accent};" if accent else ""
    f.setStyleSheet(f"""
        QFrame {{
            background: {BG2};
            border: 1px solid {BORDER};
            {border_left}
            border-radius: 10px;
        }}
    """)
    return f

def section_title(text, color=TEXT):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        font-size: 18px;
        font-weight: 800;
        color: {color};
        letter-spacing: -0.3px;
    """)
    return lbl

def section_sub(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-size: 12px; color: {TEXT2}; line-height: 1.6;")
    lbl.setWordWrap(True)
    return lbl

def divider():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background: {BORDER}; border: none; max-height: 1px;")
    f.setMaximumHeight(1)
    return f

def mono_output():
    box = QTextEdit()
    box.setReadOnly(True)
    return box


# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class Chart(FigureCanvas):
    """Thin wrapper around a matplotlib Figure that blends into our dark UI."""

    def __init__(self, w=6, h=4, dpi=100):
        self.fig = Figure(figsize=(w, h), dpi=dpi, facecolor=BG2)
        super().__init__(self.fig)
        self.setStyleSheet(f"background: {BG2}; border: none;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def reset(self):
        self.fig.clear()

    def styled_ax(self, rows=1, cols=1, idx=1):
        ax = self.fig.add_subplot(rows, cols, idx)
        ax.set_facecolor(BG3)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=TEXT2, labelsize=8)
        ax.xaxis.label.set_color(TEXT2)
        ax.yaxis.label.set_color(TEXT2)
        ax.title.set_color(TEXT)
        return ax


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK MAP CANVAS  —  custom Qt painter
# ═══════════════════════════════════════════════════════════════════════════════

class NetworkMap(QWidget):
    """
    Custom-painted Cairo road network map.
    Nodes are drawn as coloured circles; edges as lines.
    A highlighted path or MST overlay can be set at runtime.
    """

    NODE_COLORS = {
        "hospital":   "#ff6b6b",
        "government": "#ffcc44",
        None:         "#4da6ff",
    }

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph          = graph
        self.highlight_path = []
        self.highlight_mst  = []
        self.show_labels    = True
        self.setMinimumSize(480, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #06080c; border-radius: 8px;")

    def set_path(self, path):
        self.highlight_path = path
        self.update()

    def set_mst(self, mst_edges):
        self.highlight_mst = mst_edges
        self.update()

    def _transform(self):
        """Build a world→screen coordinate transformer based on current widget size."""
        xs = [m["x"] for m in self.graph.nodes.values()]
        ys = [m["y"] for m in self.graph.nodes.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 44
        W   = self.width()  - pad * 2
        H   = self.height() - pad * 2

        def to_screen(x, y):
            sx = pad + (x - min_x) / max(max_x - min_x, 0.001) * W
            sy = self.height() - pad - (y - min_y) / max(max_y - min_y, 0.001) * H
            return int(sx), int(sy)

        return to_screen

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        to_screen = self._transform()
        pos = {nid: to_screen(m["x"], m["y"]) for nid, m in self.graph.nodes.items()}

        # Build sets for quick lookup
        path_edges = set()
        for i in range(len(self.highlight_path) - 1):
            path_edges.add(tuple(sorted([self.highlight_path[i], self.highlight_path[i + 1]])))

        mst_edge_set = {tuple(sorted([u, v])) for u, v, _ in self.highlight_mst}

        # ── Edges ──────────────────────────────────────────────────────────
        drawn_edges = set()
        for u in self.graph.adj:
            for v, dist, _ in self.graph.adj[u]:
                key = tuple(sorted([u, v]))
                if key in drawn_edges:
                    continue
                drawn_edges.add(key)

                x1, y1 = pos[u]
                x2, y2 = pos[v]

                if key in path_edges:
                    pen = QPen(QColor(C_DIJ), 3)
                elif key in mst_edge_set:
                    crit = self.graph.nodes[u]["is_critical"] or self.graph.nodes[v]["is_critical"]
                    pen  = QPen(QColor(C_ASTAR if crit else C_MST), 2)
                else:
                    c = QColor(BORDER)
                    c.setAlpha(160)
                    pen = QPen(c, 1)

                painter.setPen(pen)
                painter.drawLine(x1, y1, x2, y2)

        # ── Nodes ──────────────────────────────────────────────────────────
        for nid, (sx, sy) in pos.items():
            meta   = self.graph.nodes[nid]
            ctype  = meta["critical_type"]
            in_path = nid in self.highlight_path
            color  = QColor(C_DIJ if in_path else self.NODE_COLORS.get(ctype, self.NODE_COLORS[None]))
            r      = 9 if meta["is_critical"] else 7

            # Glow ring for path nodes
            if in_path:
                glow = QColor(C_DIJ)
                glow.setAlpha(50)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(glow))
                painter.drawEllipse(sx - 14, sy - 14, 28, 28)

            # Node circle
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(BG), 2))
            painter.drawEllipse(sx - r, sy - r, r * 2, r * 2)

            # Label
            if self.show_labels:
                short = meta["name"].split()[0][:8]
                painter.setPen(QPen(QColor(TEXT if in_path else TEXT2)))
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold if in_path else QFont.Normal))
                painter.drawText(sx - 32, sy + r + 2, 64, 14, Qt.AlignCenter, short)

        painter.end()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAV BUTTON
# ═══════════════════════════════════════════════════════════════════════════════

class NavButton(QPushButton):
    """Custom sidebar navigation button with an accent colour per section."""

    def __init__(self, dot_color, label, parent=None):
        super().__init__(parent)
        self.dot_color = dot_color
        self._label    = label
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setText(f"  ●  {label}")
        self._apply_style(False)
        self.toggled.connect(self._apply_style)

    def _apply_style(self, active):
        if active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: rgba({_hex_to_rgb(self.dot_color)}, 0.1);
                    color: {self.dot_color};
                    border: none;
                    border-left: 3px solid {self.dot_color};
                    border-radius: 7px;
                    padding: 11px 14px;
                    font-size: 12px;
                    font-weight: 700;
                    text-align: left;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {TEXT2};
                    border: none;
                    border-left: 3px solid transparent;
                    border-radius: 7px;
                    padding: 11px 14px;
                    font-size: 12px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background: rgba(255,255,255,0.04);
                    color: {TEXT};
                    border-left: 3px solid {BORDER};
                }}
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# STAT BADGE  —  big number + small label
# ═══════════════════════════════════════════════════════════════════════════════

class StatBadge(QFrame):
    """Single metric display: a large coloured number over a small label."""

    def __init__(self, value, label, color=C_INFO, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba({_hex_to_rgb(color)}, 0.07);
                border: 1px solid rgba({_hex_to_rgb(color)}, 0.25);
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)

        self.val_lbl = QLabel(str(value))
        self.val_lbl.setStyleSheet(f"""
            font-size: 26px;
            font-weight: 900;
            color: {color};
            letter-spacing: -0.5px;
        """)
        self.val_lbl.setAlignment(Qt.AlignCenter)

        lbl2 = QLabel(label)
        lbl2.setStyleSheet(f"font-size: 10px; color: {TEXT2}; letter-spacing: 0.2px;")
        lbl2.setAlignment(Qt.AlignCenter)
        lbl2.setWordWrap(True)

        lay.addWidget(self.val_lbl)
        lay.addWidget(lbl2)

    def set_value(self, value):
        self.val_lbl.setText(str(value))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW / DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class OverviewPage(QWidget):

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 28, 32, 24)
        lay.setSpacing(20)

        # ── Header ────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setSpacing(14)

        # Big coloured marker block
        marker = QFrame()
        marker.setFixedSize(52, 52)
        marker.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    x1:0,y1:0,x2:1,y2:1,
                    stop:0 {C_MST}, stop:1 {C_DIJ}
                );
                border-radius: 14px;
            }}
        """)
        marker_icon = QLabel("⬡", marker)
        marker_icon.setAlignment(Qt.AlignCenter)
        marker_icon.setGeometry(0, 0, 52, 52)
        marker_icon.setStyleSheet("font-size: 24px; color: #08090c; background: transparent;")

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        t = QLabel("Cairo Transport System")
        t.setStyleSheet(f"font-size: 28px; font-weight: 900; color: {TEXT}; letter-spacing: -0.8px;")
        s = QLabel("Smart City Network Optimization  ·  CSE112  ·  Alamein International University")
        s.setStyleSheet(f"font-size: 11px; color: {TEXT2};")
        title_col.addWidget(t)
        title_col.addWidget(s)

        hdr.addWidget(marker)
        hdr.addLayout(title_col)
        hdr.addStretch()

        status = QLabel("SYSTEM ONLINE")
        status.setStyleSheet(f"""
            background: rgba(0,229,180,0.1);
            color: {C_MST};
            border: 1px solid rgba(0,229,180,0.3);
            border-radius: 20px;
            padding: 5px 14px;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1px;
        """)
        hdr.addWidget(status)
        lay.addLayout(hdr)
        lay.addWidget(divider())

        # ── Stat row ──────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        hospitals = len(self.graph.get_nodes_by_type("hospital"))
        n_nodes   = len(self.graph.nodes)
        n_edges   = len(self.graph.get_all_edges())
        n_crit    = len([m for m in self.graph.nodes.values() if m["is_critical"]])
        for val, lbl, col in [
            (n_nodes, "Total Locations",  C_DIJ),
            (n_edges, "Road Links",       C_MST),
            (hospitals, "Hospitals",      C_ASTAR),
            (n_crit, "Critical Sites",    C_DP),
            ("5", "Algorithms",           C_GRD),
        ]:
            stats_row.addWidget(StatBadge(val, lbl, col))
        lay.addLayout(stats_row)

        # ── Main area: map + algorithm list ───────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(18)

        # Map card
        map_card = card(accent=C_DIJ)
        mc_lay   = QVBoxLayout(map_card)
        mc_lay.setContentsMargins(12, 12, 12, 12)
        mc_lay.setSpacing(8)
        mc_lbl = QLabel("CAIRO ROAD NETWORK")
        mc_lbl.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {C_DIJ}; letter-spacing: 2px;")
        mc_lay.addWidget(mc_lbl)
        self.net = NetworkMap(self.graph)
        mc_lay.addWidget(self.net)
        content.addWidget(map_card, 3)

        # Algorithm cards column
        right = QVBoxLayout()
        right.setSpacing(8)
        rh = QLabel("ALGORITHM SUITE")
        rh.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {TEXT3}; letter-spacing: 2px;")
        right.addWidget(rh)

        algos = [
            (C_MST,   "Kruskal's MST",        "Infrastructure backbone · Critical node priority"),
            (C_DIJ,   "Dijkstra's Algorithm",  "Time-dependent routing · Memoization · Road closures"),
            (C_ASTAR, "A* Search",             "Emergency dispatch · Euclidean heuristic"),
            (C_DP,    "Dynamic Programming",   "Maintenance knapsack · Transit interval scheduling"),
            (C_GRD,   "Greedy Signals",        "Real-time signal control · Emergency preemption"),
        ]
        for color, name, desc in algos:
            ac = card()
            ac.setStyleSheet(f"""
                QFrame {{
                    background: {BG3};
                    border: 1px solid {BORDER};
                    border-left: 3px solid {color};
                    border-radius: 8px;
                }}
                QFrame:hover {{
                    background: rgba({_hex_to_rgb(color)}, 0.06);
                    border-color: rgba({_hex_to_rgb(color)}, 0.5);
                }}
            """)
            row = QHBoxLayout(ac)
            row.setContentsMargins(12, 9, 12, 9)
            row.setSpacing(10)

            col_dot = QFrame()
            col_dot.setFixedSize(8, 8)
            col_dot.setStyleSheet(f"background: {color}; border-radius: 4px;")

            texts = QVBoxLayout()
            texts.setSpacing(2)
            n_lbl = QLabel(name)
            n_lbl.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {TEXT};")
            d_lbl = QLabel(desc)
            d_lbl.setStyleSheet(f"font-size: 10px; color: {TEXT2};")
            texts.addWidget(n_lbl)
            texts.addWidget(d_lbl)

            row.addWidget(col_dot)
            row.addLayout(texts)
            right.addWidget(ac)

        right.addStretch()
        content.addLayout(right, 2)
        lay.addLayout(content, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: KRUSKAL'S MST
# ═══════════════════════════════════════════════════════════════════════════════

class MSTPage(QWidget):

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        # ── Left panel ────────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(pill_label("KRUSKAL'S MST", C_MST))
        left.addWidget(section_title("Infrastructure Design", C_MST))
        left.addWidget(section_sub(
            "Builds the minimum spanning tree connecting all Cairo locations. "
            "Critical facilities (hospitals, government) are placed in the first "
            "priority tier and connected before other nodes."
        ))
        left.addWidget(divider())

        run_btn = action_btn("▶  Run Kruskal's MST", C_MST)
        run_btn.clicked.connect(self._run)
        left.addWidget(run_btn)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.s_edges = StatBadge("—", "MST Edges", C_MST)
        self.s_cost  = StatBadge("—", "Total km",  C_DIJ)
        self.s_crit  = StatBadge("—", "Critical\nConnected", C_ASTAR)
        for w in (self.s_edges, self.s_cost, self.s_crit):
            stats.addWidget(w)
        left.addLayout(stats)

        lbl = QLabel("EDGE LIST")
        lbl.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {C_MST}; letter-spacing: 2px;")
        left.addWidget(lbl)
        self.output = mono_output()
        left.addWidget(self.output, 1)
        lay.addLayout(left, 2)

        # ── Right panel: map ───────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)
        ml = QLabel("MST VISUALIZATION")
        ml.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {TEXT3}; letter-spacing: 2px;")
        right.addWidget(ml)
        mc = card(accent=C_MST)
        mc_l = QVBoxLayout(mc)
        mc_l.setContentsMargins(8, 8, 8, 8)
        self.net = NetworkMap(self.graph)
        mc_l.addWidget(self.net)
        right.addWidget(mc, 1)
        lay.addLayout(right, 3)

    def _run(self):
        edges, cost, critical = kruskal_mst(self.graph)
        self.net.set_mst(edges)
        total_crit = len([n for n, m in self.graph.nodes.items() if m["is_critical"]])
        self.s_edges.set_value(len(edges))
        self.s_cost.set_value(f"{cost:.1f}")
        self.s_crit.set_value(f"{len(critical)}/{total_crit}")

        lines = []
        for u, v, w in edges:
            nu = self.graph.nodes[u]["name"]
            nv = self.graph.nodes[v]["name"]
            is_crit = self.graph.nodes[u]["is_critical"] or self.graph.nodes[v]["is_critical"]
            tag = "  ◆ CRITICAL" if is_crit else ""
            lines.append(f"{nu}  ↔  {nv}   {w:.2f} km{tag}")
        self.output.setPlainText("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DIJKSTRA
# ═══════════════════════════════════════════════════════════════════════════════

class DijkstraPage(QWidget):

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(pill_label("DIJKSTRA'S ALGORITHM", C_DIJ))
        left.addWidget(section_title("Shortest Path Routing", C_DIJ))
        left.addWidget(section_sub(
            "Finds the optimal route between any two Cairo locations. "
            "Rush-hour multipliers (×2.5 morning, ×2.8 evening) are applied "
            "to all edge weights.  Routes are cached after first computation."
        ))
        left.addWidget(divider())

        # Controls card
        ctrl = card()
        ctrl_lay = QGridLayout(ctrl)
        ctrl_lay.setContentsMargins(16, 14, 16, 14)
        ctrl_lay.setSpacing(10)

        non_crit = sorted(
            [nid for nid, m in self.graph.nodes.items() if not m["is_critical"]],
            key=lambda n: self.graph.nodes[n]["name"]
        )

        ctrl_lay.addWidget(QLabel("From:"), 0, 0)
        self.src_box = QComboBox()
        for nid in non_crit:
            self.src_box.addItem(self.graph.nodes[nid]["name"], nid)
        self.src_box.setCurrentIndex(non_crit.index("1") if "1" in non_crit else 0)
        ctrl_lay.addWidget(self.src_box, 0, 1)

        ctrl_lay.addWidget(QLabel("To:"), 1, 0)
        self.dst_box = QComboBox()
        for nid in non_crit:
            self.dst_box.addItem(self.graph.nodes[nid]["name"], nid)
        self.dst_box.setCurrentIndex(non_crit.index("5") if "5" in non_crit else 1)
        ctrl_lay.addWidget(self.dst_box, 1, 1)

        ctrl_lay.addWidget(QLabel("Time of Day:"), 2, 0)
        self.tod_box = QComboBox()
        self.tod_box.addItems(["Normal", "Morning Rush (×2.5)", "Evening Rush (×2.8)"])
        ctrl_lay.addWidget(self.tod_box, 2, 1)

        ctrl_lay.addWidget(QLabel("Block Road:"), 3, 0)
        self.block_box = QComboBox()
        self.block_box.addItem("None", None)
        seen_edges = set()
        for u in self.graph.adj:
            for v, _, _ in self.graph.adj[u]:
                key = tuple(sorted([u, v]))
                if key not in seen_edges:
                    seen_edges.add(key)
                    lbl = f"{self.graph.nodes[u]['name']} — {self.graph.nodes[v]['name']}"
                    self.block_box.addItem(lbl, key)
        ctrl_lay.addWidget(self.block_box, 3, 1)

        left.addWidget(ctrl)

        run_btn = action_btn("▶  Find Shortest Route", C_DIJ)
        run_btn.clicked.connect(self._run)
        left.addWidget(run_btn)

        stat_row = QHBoxLayout()
        stat_row.setSpacing(8)
        self.s_cost  = StatBadge("—", "Travel Cost",   C_DIJ)
        self.s_hops  = StatBadge("—", "Hops",          C_MST)
        self.s_cache = StatBadge("—", "Cache Status",  C_GRD)
        for w in (self.s_cost, self.s_hops, self.s_cache):
            stat_row.addWidget(w)
        left.addLayout(stat_row)

        rl = QLabel("ROUTE")
        rl.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {C_DIJ}; letter-spacing: 2px;")
        left.addWidget(rl)
        self.output = mono_output()
        left.addWidget(self.output, 1)
        lay.addLayout(left, 2)

        right = QVBoxLayout()
        right.setSpacing(8)
        ml = QLabel("ROUTE MAP")
        ml.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {TEXT3}; letter-spacing: 2px;")
        right.addWidget(ml)
        mc = card(accent=C_DIJ)
        mc_l = QVBoxLayout(mc)
        mc_l.setContentsMargins(8, 8, 8, 8)
        self.net = NetworkMap(self.graph)
        mc_l.addWidget(self.net)
        right.addWidget(mc, 1)
        lay.addLayout(right, 3)

    def _run(self):
        src  = self.src_box.currentData()
        dst  = self.dst_box.currentData()
        tod  = ["normal", "morning_rush", "evening_rush"][self.tod_box.currentIndex()]
        blk  = self.block_box.currentData()

        if blk:
            path, cost = shortest_path(self.graph, src, dst, tod, [blk])
            cache_status = "N/A"
        else:
            (path, cost), from_cache = cached_shortest_path(self.graph, src, dst, tod)
            cache_status = "HIT ✓" if from_cache else "MISS"

        self.net.set_path(path)
        self.s_cost.set_value(f"{cost:.2f}")
        self.s_hops.set_value(len(path) - 1)
        self.s_cache.set_value(cache_status)

        names = [self.graph.nodes[n]["name"] for n in path]
        self.output.setPlainText("  →  ".join(names))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: A* EMERGENCY
# ═══════════════════════════════════════════════════════════════════════════════

class AStarPage(QWidget):

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(pill_label("A* SEARCH  —  EMERGENCY", C_ASTAR))
        left.addWidget(section_title("Emergency Vehicle Routing", C_ASTAR))
        left.addWidget(section_sub(
            "Dispatches an emergency vehicle from any location to the nearest "
            "hospital.  The Euclidean heuristic h(n) guides A* to explore far "
            "fewer nodes than plain Dijkstra, reducing dispatch time."
        ))
        left.addWidget(divider())

        ctrl = card()
        ctrl_lay = QGridLayout(ctrl)
        ctrl_lay.setContentsMargins(16, 14, 16, 14)
        ctrl_lay.setSpacing(10)

        non_crit = sorted(
            [nid for nid, m in self.graph.nodes.items() if not m["is_critical"]],
            key=lambda n: self.graph.nodes[n]["name"]
        )

        ctrl_lay.addWidget(QLabel("Incident Location:"), 0, 0)
        self.src_box = QComboBox()
        for nid in non_crit:
            self.src_box.addItem(self.graph.nodes[nid]["name"], nid)
        self.src_box.setCurrentIndex(non_crit.index("2") if "2" in non_crit else 0)
        ctrl_lay.addWidget(self.src_box, 0, 1)

        ctrl_lay.addWidget(QLabel("Time of Day:"), 1, 0)
        self.tod_box = QComboBox()
        self.tod_box.addItems(["Normal", "Morning Rush (×2.5)", "Evening Rush (×2.8)"])
        ctrl_lay.addWidget(self.tod_box, 1, 1)

        left.addWidget(ctrl)

        run_btn = action_btn("🚑  Dispatch Emergency Vehicle", C_ASTAR)
        run_btn.clicked.connect(self._run)
        left.addWidget(run_btn)

        stat_row = QHBoxLayout()
        stat_row.setSpacing(8)
        self.s_hosp = StatBadge("—", "Nearest Hospital",  C_ASTAR)
        self.s_cost = StatBadge("—", "Travel Cost (km)",  C_DIJ)
        self.s_hops = StatBadge("—", "Route Hops",        C_MST)
        for w in (self.s_hosp, self.s_cost, self.s_hops):
            stat_row.addWidget(w)
        left.addLayout(stat_row)

        rl = QLabel("EMERGENCY ROUTE")
        rl.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {C_ASTAR}; letter-spacing: 2px;")
        left.addWidget(rl)
        self.output = mono_output()
        left.addWidget(self.output, 1)
        lay.addLayout(left, 2)

        right = QVBoxLayout()
        right.setSpacing(8)
        ml = QLabel("ROUTE MAP")
        ml.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {TEXT3}; letter-spacing: 2px;")
        right.addWidget(ml)
        mc = card(accent=C_ASTAR)
        mc_l = QVBoxLayout(mc)
        mc_l.setContentsMargins(8, 8, 8, 8)
        self.net = NetworkMap(self.graph)
        mc_l.addWidget(self.net)
        right.addWidget(mc, 1)
        lay.addLayout(right, 3)

    def _run(self):
        src = self.src_box.currentData()
        tod = ["normal", "morning_rush", "evening_rush"][self.tod_box.currentIndex()]
        path, cost, hosp = emergency_route(self.graph, src, tod)

        self.net.set_path(path)
        hosp_name = self.graph.nodes[hosp]["name"].split()[0] if hosp else "None"
        self.s_hosp.set_value(hosp_name)
        self.s_cost.set_value(f"{cost:.2f}")
        self.s_hops.set_value(len(path) - 1)

        names = [self.graph.nodes[n]["name"] for n in path]
        self.output.setPlainText("  🚑 →  ".join(names))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DYNAMIC PROGRAMMING
# ═══════════════════════════════════════════════════════════════════════════════

class DPPage(QWidget):

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)

        lay.addWidget(pill_label("DYNAMIC PROGRAMMING", C_DP))
        lay.addWidget(section_title("Optimization Problems", C_DP))
        lay.addWidget(divider())

        tabs = QTabWidget()

        # ── Knapsack tab ──────────────────────────────────────────────────
        kt = QWidget()
        kl = QHBoxLayout(kt)
        kl.setContentsMargins(14, 14, 14, 14)
        kl.setSpacing(16)

        kl_left = QVBoxLayout()
        kl_left.setSpacing(10)
        kl_left.addWidget(section_sub(
            "0/1 Knapsack: choose road maintenance projects within a budget "
            "to maximise the total benefit score.  The DP table has O(n × W) "
            "states; traceback reconstructs the optimal selection."
        ))

        budget_row = QHBoxLayout()
        budget_row.addWidget(QLabel("Budget (million EGP):"))
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(5, 100)
        self.budget_spin.setValue(35)
        self.budget_spin.setFixedWidth(80)
        budget_row.addWidget(self.budget_spin)
        budget_row.addStretch()
        kl_left.addLayout(budget_row)

        run_k = action_btn("▶  Run Knapsack", C_DP)
        run_k.clicked.connect(self._run_knapsack)
        kl_left.addWidget(run_k)

        k_stats = QHBoxLayout()
        k_stats.setSpacing(8)
        self.k_ben  = StatBadge("—", "Max Benefit",   C_MST)
        self.k_cost = StatBadge("—", "Budget Spent",  C_DP)
        self.k_util = StatBadge("—", "Utilization %", C_DIJ)
        for w in (self.k_ben, self.k_cost, self.k_util):
            k_stats.addWidget(w)
        kl_left.addLayout(k_stats)

        rl = QLabel("SELECTED ROADS")
        rl.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {C_DP}; letter-spacing: 2px;")
        kl_left.addWidget(rl)
        self.knap_out = mono_output()
        kl_left.addWidget(self.knap_out, 1)
        kl.addLayout(kl_left, 1)

        self.knap_chart = Chart(5, 4)
        kl.addWidget(self.knap_chart, 2)
        tabs.addTab(kt, "  Road Maintenance (Knapsack)  ")

        # ── Transit tab ───────────────────────────────────────────────────
        tt = QWidget()
        tl = QHBoxLayout(tt)
        tl.setContentsMargins(14, 14, 14, 14)
        tl.setSpacing(16)

        tl_left = QVBoxLayout()
        tl_left.setSpacing(10)
        tl_left.addWidget(section_sub(
            "Weighted Interval Scheduling: choose the set of non-overlapping "
            "bus/metro routes that maximises total daily passengers.  "
            "Standard DP on sorted intervals with O(n log n) complexity."
        ))

        run_t = action_btn("▶  Run Scheduling", C_DP)
        run_t.clicked.connect(self._run_transit)
        tl_left.addWidget(run_t)

        t_stats = QHBoxLayout()
        t_stats.setSpacing(8)
        self.t_pax    = StatBadge("—", "Passengers Served", C_MST)
        self.t_routes = StatBadge("—", "Routes Selected",   C_DP)
        self.t_cov    = StatBadge("—", "Coverage %",        C_DIJ)
        for w in (self.t_pax, self.t_routes, self.t_cov):
            t_stats.addWidget(w)
        tl_left.addLayout(t_stats)

        rl2 = QLabel("SELECTED ROUTES")
        rl2.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {C_DP}; letter-spacing: 2px;")
        tl_left.addWidget(rl2)
        self.trans_out = mono_output()
        tl_left.addWidget(self.trans_out, 1)
        tl.addLayout(tl_left, 1)

        self.trans_chart = Chart(5, 4)
        tl.addWidget(self.trans_chart, 2)
        tabs.addTab(tt, "  Transit Scheduling (Interval DP)  ")

        lay.addWidget(tabs, 1)

    def _run_knapsack(self):
        budget = self.budget_spin.value()
        selected, benefit, cost, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, budget)

        self.k_ben.set_value(benefit)
        self.k_cost.set_value(cost)
        self.k_util.set_value(f"{cost/budget*100:.0f}%")

        chosen_lines  = [f"✓  {r}" for r in selected]
        skipped_lines = [f"✗  {r['name']}" for r in MAINTENANCE_ROADS if r["name"] not in selected]
        self.knap_out.setPlainText("\n".join(chosen_lines + [""] + skipped_lines))

        self.knap_chart.reset()
        ax = self.knap_chart.styled_ax()
        names    = [r["name"][:14] for r in MAINTENANCE_ROADS]
        benefits = [r["benefit"] for r in MAINTENANCE_ROADS]
        costs    = [r["cost"]    for r in MAINTENANCE_ROADS]
        colors   = [C_MST if r["name"] in selected else BG3 for r in MAINTENANCE_ROADS]

        ax.bar(range(len(names)), benefits, color=colors, edgecolor=BORDER, linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=38, ha="right", fontsize=7, color=TEXT2)
        ax.set_ylabel("Benefit Score", color=TEXT2)
        ax.set_title(f"Knapsack Selection  (Budget = {budget} M EGP)", color=TEXT, fontsize=10)
        for i, (b, c) in enumerate(zip(benefits, costs)):
            ax.text(i, b + 0.5, f"c={c}", ha="center", fontsize=6, color=TEXT2)
        self.knap_chart.fig.tight_layout()
        self.knap_chart.draw()

    def _run_transit(self):
        selected, total_value = transit_scheduling(TRANSIT_ROUTES)
        total_possible = sum(r["value"] for r in TRANSIT_ROUTES)
        coverage = round(total_value / total_possible * 100, 1)

        self.t_pax.set_value(total_value)
        self.t_routes.set_value(len(selected))
        self.t_cov.set_value(f"{coverage}%")
        self.trans_out.setPlainText("\n".join(f"✓  {r}" for r in selected))

        self.trans_chart.reset()
        ax = self.trans_chart.styled_ax()
        sorted_r = sorted(TRANSIT_ROUTES, key=lambda r: r["start"])
        for i, r in enumerate(sorted_r):
            color = C_DP if r["name"] in selected else BG3
            ax.barh(i, r["end"] - r["start"], left=r["start"],
                    color=color, edgecolor=BORDER, height=0.6, linewidth=0.5)
            short = r["name"].split("—")[-1].strip() if "—" in r["name"] else r["name"]
            ax.text(r["start"] + 0.1, i, f" {short[:18]}", va="center", fontsize=7, color=TEXT)
        ax.set_yticks(range(len(sorted_r)))
        ax.set_yticklabels(
            [r["name"].split("—")[0].strip()[:8] for r in sorted_r],
            fontsize=7, color=TEXT2
        )
        ax.set_xlabel("Hour of Day", color=TEXT2)
        ax.set_title(f"Transit Scheduling — {total_value} passengers served", color=TEXT, fontsize=10)
        ax.set_xlim(5, 24)
        self.trans_chart.fig.tight_layout()
        self.trans_chart.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GREEDY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

class GreedyPage(QWidget):

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(pill_label("GREEDY ALGORITHM", C_GRD))
        left.addWidget(section_title("Traffic Signal Optimization", C_GRD))
        left.addWidget(section_sub(
            "At each intersection the lane with the largest queue gets the green "
            "light (standard greedy). Two overrides: emergency preemption always "
            "wins; starvation prevention forces a turn when a lane has waited too long."
        ))
        left.addWidget(divider())

        ctrl = card()
        ctrl_g = QGridLayout(ctrl)
        ctrl_g.setContentsMargins(16, 14, 16, 14)
        ctrl_g.setSpacing(10)

        ctrl_g.addWidget(QLabel("Time of Day:"), 0, 0)
        self.tod_box = QComboBox()
        self.tod_box.addItems(["Normal", "Morning Rush", "Evening Rush"])
        ctrl_g.addWidget(self.tod_box, 0, 1)

        ctrl_g.addWidget(QLabel("Emergency at:"), 1, 0)
        self.emg_inter = QComboBox()
        self.emg_inter.addItem("None", None)
        for inter in INTERSECTIONS:
            self.emg_inter.addItem(inter["name"], inter["name"])
        ctrl_g.addWidget(self.emg_inter, 1, 1)

        ctrl_g.addWidget(QLabel("Direction:"), 2, 0)
        self.emg_dir = QComboBox()
        self.emg_dir.addItem("—", None)
        ctrl_g.addWidget(self.emg_dir, 2, 1)
        self.emg_inter.currentIndexChanged.connect(self._update_dirs)

        left.addWidget(ctrl)

        run_btn = action_btn("▶  Run Signal Cycle", C_GRD)
        run_btn.clicked.connect(self._run)
        left.addWidget(run_btn)

        analysis_btn = ghost_btn("  Run Optimality Analysis (500 sims)", C_GRD)
        analysis_btn.clicked.connect(self._run_analysis)
        left.addWidget(analysis_btn)

        s_row = QHBoxLayout()
        s_row.setSpacing(8)
        self.s_greedy = StatBadge("—", "Greedy\nOptimal %", C_MST)
        self.s_starv  = StatBadge("—", "Starvation\nOverride %", C_DP)
        self.s_emg    = StatBadge("—", "Emergency\nPreemption %", C_ASTAR)
        for w in (self.s_greedy, self.s_starv, self.s_emg):
            s_row.addWidget(w)
        left.addLayout(s_row)

        dl = QLabel("SIGNAL DECISIONS")
        dl.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {C_GRD}; letter-spacing: 2px;")
        left.addWidget(dl)
        self.output = mono_output()
        left.addWidget(self.output, 1)
        lay.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(8)
        rl = QLabel("QUEUE VISUALIZATION")
        rl.setStyleSheet(f"font-size: 9px; font-weight: 800; color: {TEXT3}; letter-spacing: 2px;")
        right.addWidget(rl)
        self.chart = Chart(5, 6)
        right.addWidget(self.chart, 1)
        lay.addLayout(right, 2)

    def _update_dirs(self):
        inter_name = self.emg_inter.currentData()
        self.emg_dir.clear()
        self.emg_dir.addItem("—", None)
        if inter_name:
            for inter in INTERSECTIONS:
                if inter["name"] == inter_name:
                    for d in inter["directions"]:
                        self.emg_dir.addItem(d, d)
                    break

    def _run(self):
        tod = ["normal", "morning_rush", "evening_rush"][self.tod_box.currentIndex()]
        emergency = None
        inter_name = self.emg_inter.currentData()
        dir_name   = self.emg_dir.currentData()
        if inter_name and dir_name:
            emergency = {"intersection": inter_name, "direction": dir_name}

        results = optimize_all_intersections(tod, emergency)

        lines = []
        for r in results:
            sym = "🚨" if "EMERGENCY" in r["reason"] else ("⚠" if "STARVATION" in r["reason"] else "●")
            lines.append(f"{sym}  {r['intersection']}")
            lines.append(f"   GREEN → {r['green_light']}  ({r['vehicles_served']} vehicles)")
            reason_short = r["reason"].split("—")[0].strip()
            lines.append(f"   {reason_short}")
            lines.append("")
        self.output.setPlainText("\n".join(lines))
        self._draw_queues(results)

    def _run_analysis(self):
        analysis = analyze_greedy_optimality(500)
        self.s_greedy.set_value(f"{analysis['greedy_optimal_pct']}%")
        self.s_starv.set_value(f"{analysis['starvation_override_pct']}%")
        self.s_emg.set_value(f"{analysis['emergency_pct']}%")

        self.chart.reset()
        ax = self.chart.styled_ax()
        vals   = [analysis["greedy_optimal_pct"],
                  analysis["starvation_override_pct"],
                  analysis["emergency_pct"]]
        labels = ["Greedy\nOptimal", "Starvation\nOverride", "Emergency\nPreemption"]
        colors = [C_MST, C_DP, C_ASTAR]
        fv = [(v, l, c) for v, l, c in zip(vals, labels, colors) if v > 0]
        if fv:
            vs, ls, cs = zip(*fv)
            _, _, autos = ax.pie(vs, labels=ls, colors=cs, autopct="%1.1f%%",
                                  startangle=140, textprops={"color": TEXT, "fontsize": 9})
            for a in autos:
                a.set_color(BG)
                a.set_fontweight("bold")
        ax.set_title("Greedy Decisions — 500 Simulations", color=TEXT, fontsize=10)
        self.chart.fig.tight_layout()
        self.chart.draw()

    def _draw_queues(self, results):
        self.chart.reset()
        n = len(results)
        for idx, r in enumerate(results):
            ax = self.chart.fig.add_subplot(n, 1, idx + 1)
            ax.set_facecolor(BG3)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.tick_params(colors=TEXT2, labelsize=7)
            dirs   = list(r["queues"].keys())
            vals   = [r["queues"][d] for d in dirs]
            colors = [C_GRD if d == r["green_light"] else BG3 for d in dirs]
            ax.barh(dirs, vals, color=colors, edgecolor=BORDER, linewidth=0.4)
            ax.set_title(r["intersection"], color=TEXT, fontsize=8, pad=2)
        self.chart.fig.tight_layout(pad=0.4)
        self.chart.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

class ChartsPage(QWidget):

    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)

        lay.addWidget(pill_label("PERFORMANCE ANALYSIS", C_INFO))
        lay.addWidget(section_title("Algorithm Analysis Charts", C_INFO))
        lay.addWidget(divider())

        ctrl = QHBoxLayout()
        self.chart_sel = QComboBox()
        self.chart_sel.addItems([
            "Rush Hour Cost Comparison",
            "MST Edge Weight Distribution",
            "Road Maintenance Benefit/Cost Ratio",
            "Transit Route Gantt Chart",
        ])
        ctrl.addWidget(self.chart_sel)
        run_btn = action_btn("▶  Generate", C_INFO, small=True)
        run_btn.clicked.connect(self._run)
        ctrl.addWidget(run_btn)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(8, 8, 8, 8)
        self.chart = Chart(10, 5)
        cl.addWidget(self.chart)
        lay.addWidget(c, 1)

    def _run(self):
        idx = self.chart_sel.currentIndex()
        self.chart.reset()
        if   idx == 0: self._rush_hour()
        elif idx == 1: self._mst_dist()
        elif idx == 2: self._maintenance_ratio()
        elif idx == 3: self._gantt()
        self.chart.fig.tight_layout()
        self.chart.draw()

    def _rush_hour(self):
        pairs = [("1","5"), ("2","8"), ("11","14"), ("1","4"), ("7","3")]
        tods  = ["normal", "morning_rush", "evening_rush"]
        cols  = [C_MST, C_DP, C_ASTAR]
        ax = self.chart.styled_ax()
        w  = 0.25
        for i, (tod, col) in enumerate(zip(tods, cols)):
            costs = [shortest_path(self.graph, s, d, tod)[1] for s, d in pairs]
            xs    = [xi + (i - 1) * w for xi in range(len(pairs))]
            ax.bar(xs, costs, w, label=tod.replace("_", " ").title(),
                   color=col, edgecolor=BORDER, linewidth=0.5)
        ax.set_xticks(list(range(len(pairs))))
        ax.set_xticklabels(
            [f"{self.graph.nodes[s]['name']}\n→ {self.graph.nodes[d]['name']}"
             for s, d in pairs],
            fontsize=8, color=TEXT2
        )
        ax.set_ylabel("Travel Cost (km × traffic)", color=TEXT2)
        ax.set_title("Rush Hour Impact on Shortest Paths", color=TEXT)
        ax.legend(fontsize=8, facecolor=BG3, labelcolor=TEXT, framealpha=0.8)

    def _mst_dist(self):
        edges, cost, _ = kruskal_mst(self.graph)
        weights = [w for _, _, w in edges]
        ax = self.chart.styled_ax()
        ax.hist(weights, bins=10, color=C_MST, edgecolor=BORDER, linewidth=0.5, alpha=0.85)
        ax.set_xlabel("Edge Weight (km × traffic factor)", color=TEXT2)
        ax.set_ylabel("Number of Edges", color=TEXT2)
        ax.set_title(f"MST Edge Weight Distribution  —  {len(edges)} edges, total {cost:.1f} km", color=TEXT)

    def _maintenance_ratio(self):
        selected, _, _, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, 35)
        ax = self.chart.styled_ax()
        names  = [r["name"][:16] for r in MAINTENANCE_ROADS]
        ratios = [r["benefit"] / r["cost"] for r in MAINTENANCE_ROADS]
        colors = [C_MST if r["name"] in selected else TEXT3 for r in MAINTENANCE_ROADS]
        ax.barh(range(len(names)), ratios, color=colors, edgecolor=BORDER, linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color=TEXT2)
        ax.set_xlabel("Benefit / Cost Ratio", color=TEXT2)
        ax.set_title("Road Maintenance — Benefit per Million EGP  (green = selected, budget=35)", color=TEXT)

    def _gantt(self):
        selected, _ = transit_scheduling(TRANSIT_ROUTES)
        ax = self.chart.styled_ax()
        sorted_r = sorted(TRANSIT_ROUTES, key=lambda r: r["start"])
        for i, r in enumerate(sorted_r):
            col = C_DP if r["name"] in selected else BG3
            ax.barh(i, r["end"] - r["start"], left=r["start"],
                    color=col, edgecolor=BORDER, height=0.65, linewidth=0.5)
        ax.set_yticks(range(len(sorted_r)))
        ax.set_yticklabels([r["name"][:20] for r in sorted_r], fontsize=7, color=TEXT2)
        ax.set_xlabel("Hour of Day", color=TEXT2)
        ax.set_title("Transit Scheduling Gantt  (amber = selected)", color=TEXT)
        ax.set_xlim(5, 24)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cairo Smart Transport System  ·  CSE112")
        self.setMinimumSize(1300, 820)
        self.resize(1480, 920)

        self.graph = build_cairo_graph()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background: {BG};")
        root.addWidget(self.stack, 1)

        self.pages = [
            OverviewPage(self.graph),
            MSTPage(self.graph),
            DijkstraPage(self.graph),
            AStarPage(self.graph),
            DPPage(self.graph),
            GreedyPage(self.graph),
            ChartsPage(self.graph),
        ]
        for page in self.pages:
            self.stack.addWidget(page)

        self._switch(0)

    def _build_sidebar(self):
        sb = QFrame()
        sb.setFixedWidth(218)
        sb.setStyleSheet(f"""
            QFrame {{
                background: {BG2};
                border-right: 1px solid {BORDER};
            }}
        """)
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(12, 22, 12, 18)
        lay.setSpacing(3)

        # Logo area
        logo_frame = QFrame()
        logo_frame.setStyleSheet(f"""
            QFrame {{
                background: transparent;
                border: none;
            }}
        """)
        lf_lay = QHBoxLayout(logo_frame)
        lf_lay.setContentsMargins(4, 0, 4, 0)
        lf_lay.setSpacing(10)

        # Geometric logo mark made of two overlapping shapes
        icon_box = QFrame()
        icon_box.setFixedSize(40, 40)
        icon_box.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    x1:0,y1:0,x2:1,y2:1,
                    stop:0 {C_MST}, stop:0.5 {C_DIJ}, stop:1 {C_GRD}
                );
                border-radius: 10px;
            }}
        """)
        icon_inner = QLabel("⬡", icon_box)
        icon_inner.setGeometry(0, 0, 40, 40)
        icon_inner.setAlignment(Qt.AlignCenter)
        icon_inner.setStyleSheet("font-size: 19px; color: #08090c; background: transparent;")

        name_col = QVBoxLayout()
        name_col.setSpacing(0)
        n1 = QLabel("CairoTS")
        n1.setStyleSheet(f"font-size: 16px; font-weight: 900; color: {TEXT}; letter-spacing: -0.5px;")
        n2 = QLabel("Transport System")
        n2.setStyleSheet(f"font-size: 9px; color: {TEXT2}; letter-spacing: 0.5px;")
        name_col.addWidget(n1)
        name_col.addWidget(n2)

        lf_lay.addWidget(icon_box)
        lf_lay.addLayout(name_col)
        lay.addWidget(logo_frame)
        lay.addSpacing(10)
        lay.addWidget(divider())
        lay.addSpacing(6)

        # Section label
        nav_lbl = QLabel("NAVIGATION")
        nav_lbl.setStyleSheet(f"font-size: 9px; font-weight: 700; color: {TEXT3}; "
                               f"letter-spacing: 2px; padding: 0 6px; margin-top: 4px;")
        lay.addWidget(nav_lbl)
        lay.addSpacing(4)

        # Nav buttons — each has its own accent colour
        nav_items = [
            (TEXT,   "Overview"),
            (C_MST,  "MST Infrastructure"),
            (C_DIJ,  "Dijkstra Routing"),
            (C_ASTAR,"A* Emergency"),
            (C_DP,   "Dynamic Programming"),
            (C_GRD,  "Greedy Signals"),
            (C_INFO, "Analysis Charts"),
        ]
        self.nav_btns = []
        for color, label in nav_items:
            btn = NavButton(color, label)
            btn.clicked.connect(lambda _, idx=len(self.nav_btns): self._switch(idx))
            lay.addWidget(btn)
            self.nav_btns.append(btn)

        lay.addStretch()
        lay.addWidget(divider())
        lay.addSpacing(6)

        uni = QLabel("Alamein Int'l University")
        uni.setStyleSheet(f"font-size: 9px; color: {TEXT3}; padding: 0 6px;")
        course = QLabel("CSE112 — Design & Analysis\nof Algorithms")
        course.setStyleSheet(f"font-size: 8px; color: {TEXT3}; padding: 0 6px;")
        course.setWordWrap(True)
        lay.addWidget(uni)
        lay.addWidget(course)

        return sb

    def _switch(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_btns):
            btn.setChecked(i == idx)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(style_app())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
