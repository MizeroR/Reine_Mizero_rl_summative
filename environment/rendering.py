"""
Pygame 2D Dashboard — Digital Wellbeing Coach Visualizer

Layout (800 × 580):
  ┌──────────────────────────────────────────────────┐
  │  Header: title + step counter + SAS badge         │
  ├─────────────────┬────────────────────────────────┤
  │  Metric gauges  │  SAS-SV risk timeline chart     │
  │  (left column)  │                                 │
  ├─────────────────┴────────────────────────────────┤
  │  Current intervention banner                      │
  ├──────────────────────────────────────────────────┤
  │  App-usage pie + compliance indicator             │
  └──────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
import numpy as np

WIDTH, HEIGHT = 800, 580
FPS = 2

# Colour palette
BG          = (18,  18,  30)
PANEL       = (28,  28,  48)
ACCENT      = (82, 130, 255)
GREEN       = (72, 210, 120)
YELLOW      = (255, 210,  60)
RED         = (255,  80,  80)
WHITE       = (240, 240, 250)
GREY        = (120, 120, 150)
DARK_GREY   = (50,  50,  70)
TEAL        = (60, 200, 200)
ORANGE      = (255, 160,  50)
PURPLE      = (160,  90, 220)

ACTION_COLORS = [GREY, GREEN, RED, TEAL, ORANGE, ACCENT]
ACTION_ICONS  = ["💤", "⏱️", "🚫", "🧘", "☕", "🎯"]
ACTION_NAMES  = [
    "Do Nothing",
    "Reduce Screen Time",
    "Block Social Media",
    "Mindfulness Prompt",
    "Enforce a Break",
    "Enable Focus Mode",
]


def _risk_color(sas: float) -> tuple:
    if sas < 12:
        return GREEN
    if sas < 24:
        return YELLOW
    if sas < 36:
        return ORANGE
    return RED


def _risk_label(sas: float) -> str:
    if sas < 12:
        return "LOW RISK"
    if sas < 24:
        return "MILD"
    if sas < 36:
        return "MODERATE"
    return "HIGH RISK"


class WellbeingRenderer:
    def __init__(self):
        import pygame
        pygame.init()
        self._pg = pygame
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Digital Wellbeing Coach — RL Dashboard")
        self.clock = pygame.time.Clock()
        self._load_fonts()

    def _load_fonts(self):
        pg = self._pg
        self.font_title  = pg.font.SysFont("DejaVuSans", 20, bold=True)
        self.font_large  = pg.font.SysFont("DejaVuSans", 28, bold=True)
        self.font_medium = pg.font.SysFont("DejaVuSans", 16)
        self.font_small  = pg.font.SysFont("DejaVuSans", 13)

    # ── Main draw entry ────────────────────────────────────────────────────

    def draw(self, state: np.ndarray, action: int, complied: bool,
             step: int, history: list[dict]) -> np.ndarray | None:
        pg = self._pg

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return None

        screen_t, unlocks, sleep, sas, social, prod, tod = state

        self.screen.fill(BG)

        self._draw_header(sas, step)
        self._draw_metrics(screen_t, unlocks, sleep, tod)
        self._draw_sas_chart(history, sas)
        self._draw_action_banner(action, complied)
        self._draw_app_pie(social, prod)
        self._draw_compliance(complied, action)

        pg.display.flip()
        self.clock.tick(FPS)

        # rgb_array support
        return pg.surfarray.array3d(self.screen).transpose(1, 0, 2)

    # ── Section renderers ──────────────────────────────────────────────────

    def _draw_header(self, sas: float, step: int):
        pg = self._pg
        # Background strip
        pg.draw.rect(self.screen, PANEL, (0, 0, WIDTH, 56))
        pg.draw.line(self.screen, ACCENT, (0, 56), (WIDTH, 56), 2)

        title = self.font_title.render("Digital Wellbeing Coach  ·  RL Agent Dashboard", True, WHITE)
        self.screen.blit(title, (16, 10))

        step_txt = self.font_medium.render(f"Step  {step} / 30", True, GREY)
        self.screen.blit(step_txt, (16, 34))

        # SAS badge
        color = _risk_color(sas)
        label = _risk_label(sas)
        badge_txt = self.font_large.render(f"SAS-SV  {sas:.1f}  —  {label}", True, color)
        bw = badge_txt.get_width()
        self.screen.blit(badge_txt, (WIDTH - bw - 16, 14))

    def _draw_metrics(self, screen_t: float, unlocks: float,
                       sleep: float, tod: float):
        pg = self._pg
        x0, y0 = 12, 68
        panel_w, panel_h = 230, 290
        pg.draw.rect(self.screen, PANEL, (x0, y0, panel_w, panel_h), border_radius=8)

        hdr = self.font_medium.render("User Metrics", True, ACCENT)
        self.screen.blit(hdr, (x0 + 10, y0 + 8))

        metrics = [
            ("Screen Time",  f"{screen_t:.1f} h",   screen_t / 16,   ACCENT),
            ("Unlocks / day", f"{int(unlocks)}",     unlocks / 200,   PURPLE),
            ("Sleep Quality", f"{sleep:.1f} / 10",  sleep / 10,      TEAL),
            ("Time of Day",   f"{int(tod):02d}:00",  tod / 23,        YELLOW),
        ]

        for i, (label, val_str, ratio, color) in enumerate(metrics):
            gy = y0 + 38 + i * 60
            lbl = self.font_small.render(label, True, GREY)
            val = self.font_medium.render(val_str, True, WHITE)
            self.screen.blit(lbl, (x0 + 10, gy))
            self.screen.blit(val, (x0 + 10, gy + 16))
            # Bar
            bar_x = x0 + 10
            bar_y = gy + 36
            bar_w = panel_w - 20
            bar_h = 8
            pg.draw.rect(self.screen, DARK_GREY, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            fill = max(4, int(bar_w * min(ratio, 1.0)))
            pg.draw.rect(self.screen, color, (bar_x, bar_y, fill, bar_h), border_radius=4)

    def _draw_sas_chart(self, history: list[dict], current_sas: float):
        pg = self._pg
        cx, cy = 255, 68
        cw, ch = WIDTH - cx - 12, 200

        pg.draw.rect(self.screen, PANEL, (cx, cy, cw, ch), border_radius=8)
        hdr = self.font_medium.render("SAS-SV Risk Score  (episode timeline)", True, ACCENT)
        self.screen.blit(hdr, (cx + 10, cy + 8))

        # Grid lines at risk thresholds
        thresholds = [(12, GREEN, "Low"), (24, YELLOW, "Mild"),
                      (36, ORANGE, "Mod"), (48, RED, "High")]
        plot_x = cx + 10
        plot_y = cy + 30
        plot_w = cw - 20
        plot_h = ch - 44

        for val, col, lbl in thresholds:
            ry = plot_y + plot_h - int(plot_h * val / 48)
            pg.draw.line(self.screen, (*col, 60), (plot_x, ry), (plot_x + plot_w, ry), 1)
            t = self.font_small.render(lbl, True, col)
            self.screen.blit(t, (plot_x + plot_w - t.get_width() - 2, ry - 12))

        if not history:
            return

        pts = [(h["step"], h["sas"]) for h in history]
        max_step = max(30, pts[-1][0])

        def to_px(step, sas):
            px = plot_x + int(plot_w * step / max_step)
            py = plot_y + plot_h - int(plot_h * min(sas, 48) / 48)
            return px, py

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                pg.draw.line(self.screen, ACCENT,
                             to_px(*pts[i]), to_px(*pts[i + 1]), 2)

        # Current dot
        px, py = to_px(pts[-1][0], pts[-1][1])
        pg.draw.circle(self.screen, _risk_color(current_sas), (px, py), 5)

    def _draw_action_banner(self, action: int, complied: bool):
        pg = self._pg
        bx, by = 12, 368
        bw, bh = WIDTH - 24, 52

        color = ACTION_COLORS[action]
        pg.draw.rect(self.screen, PANEL, (bx, by, bw, bh), border_radius=8)
        pg.draw.rect(self.screen, color,  (bx, by, bw, bh), width=2, border_radius=8)

        icon = ACTION_ICONS[action]
        name = ACTION_NAMES[action]
        status = "✓ Complied" if complied else "✗ Ignored"
        status_col = GREEN if complied else RED

        action_txt  = self.font_title.render(f"Intervention:  {name}", True, WHITE)
        status_txt  = self.font_medium.render(status, True, status_col)
        self.screen.blit(action_txt,  (bx + 16, by + 8))
        self.screen.blit(status_txt,  (bx + 16, by + 30))

        # Action ID pill
        pill_txt = self.font_medium.render(f"Action {action}", True, color)
        pw = pill_txt.get_width() + 16
        pg.draw.rect(self.screen, DARK_GREY, (bx + bw - pw - 12, by + 12, pw, 28), border_radius=6)
        self.screen.blit(pill_txt, (bx + bw - pw - 4, by + 16))

    def _draw_app_pie(self, social: float, prod: float):
        pg = self._pg
        px, py = 12, 430
        pw, ph = 230, 138
        pg.draw.rect(self.screen, PANEL, (px, py, pw, ph), border_radius=8)
        hdr = self.font_medium.render("App Category Usage", True, ACCENT)
        self.screen.blit(hdr, (px + 10, py + 8))

        entertainment = max(0.0, 1.0 - social - prod)
        slices = [
            (social,        RED,    "Social"),
            (prod,          GREEN,  "Productivity"),
            (entertainment, YELLOW, "Entertainment"),
        ]

        cx_pie = px + 70
        cy_pie = py + 80
        radius = 42
        start_angle = -math.pi / 2

        for ratio, color, label in slices:
            if ratio <= 0:
                continue
            end_angle = start_angle + 2 * math.pi * ratio
            points = [(cx_pie, cy_pie)]
            steps = max(2, int(36 * ratio))
            for s in range(steps + 1):
                a = start_angle + (end_angle - start_angle) * s / steps
                points.append((cx_pie + radius * math.cos(a),
                                cy_pie + radius * math.sin(a)))
            if len(points) > 2:
                pg.draw.polygon(self.screen, color, points)
                pg.draw.polygon(self.screen, PANEL,  points, 1)
            start_angle = end_angle

        # Legend
        lx = px + 125
        for i, (ratio, color, label) in enumerate(slices):
            ly = py + 38 + i * 28
            pg.draw.rect(self.screen, color, (lx, ly, 12, 12), border_radius=2)
            t = self.font_small.render(f"{label}  {ratio*100:.0f}%", True, WHITE)
            self.screen.blit(t, (lx + 18, ly - 1))

    def _draw_compliance(self, complied: bool, action: int):
        pg = self._pg
        cx, cy = 255, 430
        cw, ch = WIDTH - cx - 12, 138
        pg.draw.rect(self.screen, PANEL, (cx, cy, cw, ch), border_radius=8)
        hdr = self.font_medium.render("Compliance Indicator", True, ACCENT)
        self.screen.blit(hdr, (cx + 10, cy + 8))

        if action == 0:
            msg   = "No intervention applied"
            color = GREY
        elif complied:
            msg   = "User followed the recommendation"
            color = GREEN
        else:
            msg   = "User ignored the recommendation"
            color = RED

        icon_size = 40
        icon_x    = cx + cw // 2 - icon_size // 2
        icon_y    = cy + 40
        pg.draw.circle(self.screen, color, (cx + cw // 2, icon_y + icon_size // 2), icon_size // 2)

        checkmark = self.font_large.render("✓" if (complied or action == 0) else "✗", True, BG)
        cw2 = checkmark.get_width()
        self.screen.blit(checkmark, (cx + cw // 2 - cw2 // 2, icon_y + 6))

        msg_surf = self.font_small.render(msg, True, color)
        mw = msg_surf.get_width()
        self.screen.blit(msg_surf, (cx + cw // 2 - mw // 2, cy + 100))

    # ── Cleanup ────────────────────────────────────────────────────────────

    def close(self):
        self._pg.quit()
