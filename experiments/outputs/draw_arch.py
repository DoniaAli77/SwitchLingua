"""Render the architecture diagrams to PNG using Pillow (no external binaries/network).
Outputs -> experiments/outputs/diagrams/*.png"""
import math, pathlib
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path(__file__).resolve().parent / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# palette
NAVY = "#1f3d63"; INK = "#1b2733"; MUT = "#5b6b7b"
C_HEAD = "#cfe3ff"; C_AGENT = "#ffffff"; C_DEC = "#ffe08a"; C_TERM = "#d6efd6"
C_HL = "#ffe2b8"; C_DATA = "#e7defb"; C_GROUP = "#f3f6fa"; C_VALID = "#d4efe9"
LINE = "#46566a"; SHADOW = (203, 211, 220)


def font(sz, bold=False):
    for n in (["arialbd.ttf"] if bold else ["arial.ttf"]) + ["DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap(d, text, f, maxw):
    out = []
    for raw in text.split("\n"):
        words = raw.split(" "); cur = ""
        for w in words:
            t = (cur + " " + w).strip()
            if d.textbbox((0, 0), t, font=f)[2] <= maxw or not cur:
                cur = t
            else:
                out.append(cur); cur = w
        out.append(cur)
    return out


def box(d, xy, text, fill=C_AGENT, fsz=15, bold=False, outline="#9fb3c8", radius=14,
        tcol=INK, shadow=True, accent=None, ow=2):
    x0, y0, x1, y1 = xy
    if shadow:
        d.rounded_rectangle((x0 + 5, y0 + 6, x1 + 5, y1 + 6), radius=radius, fill=SHADOW)
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=ow)
    if accent:
        d.rounded_rectangle((x0, y0, x0 + 10, y1), radius=radius, fill=accent)
        d.rectangle((x0 + 6, y0, x0 + 12, y1), fill=fill)
    f = font(fsz, bold)
    lines = _wrap(d, text, f, (x1 - x0) - 26)
    hs = [d.textbbox((0, 0), ln, font=f)[3] for ln in lines]
    th = sum(hs) + 4 * (len(lines) - 1)
    cy = (y0 + y1) / 2 - th / 2
    for ln, h in zip(lines, hs):
        w = d.textbbox((0, 0), ln, font=f)[2]
        d.text(((x0 + x1) / 2 - w / 2, cy), ln, font=f, fill=tcol)
        cy += h + 4


def badge(d, cx, cy, n, col=NAVY):
    d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=col)
    f = font(14, True); w = d.textbbox((0, 0), n, font=f)[2]
    d.text((cx - w / 2, cy - 9), n, font=f, fill="white")


def pill(d, xy, text, fill=C_TERM, fsz=14):
    box(d, xy, text, fill, fsz, True, outline="#6aa86a", radius=(xy[3] - xy[1]) // 2)


def chip(d, cx, cy, text, fill="#ffd9a8", tcol="#7a4a10", oc="#e0ac63"):
    f = font(11, True); tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle((cx - tw / 2 - 9, cy - 12, cx + tw / 2 + 9, cy + 12), radius=11, fill=fill, outline=oc, width=2)
    d.text((cx - tw / 2, cy - 8), text, font=f, fill=tcol)


def diamond(d, cx, cy, w, h, text, fill=C_DEC, fsz=14):
    d.polygon([(cx + 5, cy - h / 2 + 6), (cx + w / 2 + 5, cy + 6), (cx + 5, cy + h / 2 + 6), (cx - w / 2 + 5, cy + 6)], fill=SHADOW)
    d.polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], fill=fill, outline="#caa53a", width=2)
    f = font(fsz, True); lines = text.split("\n"); hs = [d.textbbox((0, 0), l, font=f)[3] for l in lines]
    yy = cy - (sum(hs) + 3 * (len(lines) - 1)) / 2
    for ln, h in zip(lines, hs):
        tw = d.textbbox((0, 0), ln, font=f)[2]; d.text((cx - tw / 2, yy), ln, font=f, fill="#5b4a00"); yy += h + 3


def arrow(d, p0, p1, color=LINE, w=3, label=None):
    d.line([p0, p1], fill=color, width=w)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0]); s = 11
    d.polygon([p1, (p1[0] - s * math.cos(ang - 0.45), p1[1] - s * math.sin(ang - 0.45)),
               (p1[0] - s * math.cos(ang + 0.45), p1[1] - s * math.sin(ang + 0.45))], fill=color)
    if label:
        f = font(13, True); mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        tw = d.textbbox((0, 0), label, font=f)[2]
        d.rounded_rectangle((mx - tw / 2 - 6, my - 11, mx + tw / 2 + 6, my + 11), radius=8, fill="white", outline="#cbd5e0")
        d.text((mx - tw / 2, my - 8), label, font=f, fill="#33485e")


def orth(d, pts, color=LINE, w=3, label=None):
    for a, b in zip(pts, pts[1:]):
        d.line([a, b], fill=color, width=w)
    arrow(d, pts[-2], pts[-1], color, w)
    if label:
        a, b = pts[0], pts[1]; f = font(13, True); mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        tw = d.textbbox((0, 0), label, font=f)[2]
        d.rounded_rectangle((mx - tw / 2 - 6, my - 11, mx + tw / 2 + 6, my + 11), radius=8, fill="white", outline="#cbd5e0")
        d.text((mx - tw / 2, my - 8), label, font=f, fill="#33485e")


def title_band(d, W, t, sub):
    d.rectangle((0, 0, W, 84), fill=NAVY)
    d.text((34, 20), t, font=font(26, True), fill="white")
    d.text((34, 55), sub, font=font(15), fill="#bcd2ee")


# ============================================================ 1. SwitchLingua pipeline (polished)
def diagram1():
    W, H = 1340, 1180; im = Image.new("RGB", (W, H), "#fbfcfe"); d = ImageDraw.Draw(im)
    title_band(d, W, "SwitchLingua — TASK-AWARE generation pipeline (System B)",
               "LangGraph · generation · validation · refinement all conditioned on the task · gpt-4o-mini")
    cx = 600
    bx0, bx1 = cx - 280, cx + 280
    box(d, (bx0, 110, bx1, 158), "CONFIG (pre_execute)  →  generate_scenarios()   [Cartesian product]", C_HEAD, 15, True, outline="#7ea6d8")
    pill(d, (cx - 70, 182, cx + 70, 220), "START")
    box(d, (bx0, 246, bx1, 300), "DataGenerationAgent\ntask-specific prompt  →  instances = [s1 … sN]", C_AGENT, 14, True, accent="#6aa3e0")
    badge(d, bx0, 273, "1"); chip(d, bx1 - 64, 246, "task-aware")
    box(d, (bx0, 328, bx1, 392), "TaskValidatorAgent   (per sentence: is the task satisfied?)\ntopic · sentiment-label · NER required English-script entities", C_VALID, 13, True, outline="#7cc0b3")
    badge(d, bx0, 360, "2"); chip(d, bx1 - 64, 328, "task-aware")
    # grouped parallel scorers
    gx0, gx1, gy0, gy1 = 70, 1140, 430, 590
    d.rounded_rectangle((gx0 + 5, gy0 + 6, gx1 + 5, gy1 + 6), radius=16, fill=SHADOW)
    d.rounded_rectangle((gx0, gy0, gx1, gy1), radius=16, fill=C_GROUP, outline="#c4d0dd", width=2)
    d.text((gx0 + 18, gy0 + 10), "Per-sentence quality scoring  (parallel fan-out → fan-in)", font=font(14, True), fill=MUT)
    sc = [("3", "FluencyAgent", "/10"), ("4", "NaturalnessAgent", "/10"),
          ("5", "CSRatioAgent", "/10  ◄ deterministic CS counter\ncompute_true_cs_stats (0 variance)"),
          ("6", "SocialCulturalAgent", "/10")]
    sx = [gx0 + 22, gx0 + 290, gx0 + 558, gx0 + 845]; sw = 248
    for (n, name, note), x in zip(sc, sx):
        box(d, (x, gy0 + 46, x + sw, gy1 - 20), f"{name}\n{note}", C_AGENT, 12, True, outline="#aab9c9")
        badge(d, x, gy0 + 46 + (gy1 - 20 - gy0 - 46) / 2, n, "#7a6aa0")
    box(d, (bx0, 622, bx1, 690), "SummarizeResult\nPER-SENTENCE weighted_score = w·(flu, nat, cs, socio)\n→ sentence_records[i] {text, weighted_score, task_passed}", C_HL, 13, True, outline="#e0ac63")
    badge(d, bx0, 656, "7")
    diamond(d, cx, 775, 250, 120, "meet_criteria ?\n(per sentence)\nscore ≥ bar  AND\ntask_passed")
    box(d, (150, 845, 510, 915), "RefinerAgent\ntargeted rewrite of FAILING sentences only\nGUARDRAIL: re-validate + re-score → rollback if worse", C_AGENT, 12, True, accent="#e08a6a")
    badge(d, 150, 880, "8"); chip(d, 510 - 64, 845, "task-aware")
    box(d, (770, 855, 1070, 905), "AcceptanceAgent", C_TERM, 14, True, outline="#6aa86a")
    badge(d, 770, 880, "9")
    pill(d, (870, 935, 1000, 972), "END → JSONL")
    # task-aware panel (System B only — no System-C reference)
    box(d, (975, 632, 1295, 800),
        "TASK-AWARE PIPELINE\n\nThree stages are conditioned on the task:\n• generation → task-specific prompt\n• validation → TaskValidatorAgent\n• refinement → task-aware rewrite\n\ntask: {topic | sentiment | NER}\n+ label + task_constraints", "#fff3da", 13, True, outline="#e0b25a")
    # edges
    arrow(d, (cx, 158), (cx, 182)); arrow(d, (cx, 220), (cx, 246)); arrow(d, (cx, 300), (cx, 328))
    for x in sx:
        arrow(d, (cx, 392), (x + sw / 2, gy0 + 46), w=2)
    for x in sx:
        arrow(d, (x + sw / 2, gy1 - 20), (cx, 622), w=2)
    arrow(d, (cx, 690), (cx, 716))
    orth(d, [(cx - 125, 775), (330, 775), (330, 845)], label="needs refine")
    orth(d, [(cx + 125, 775), (920, 775), (920, 855)], label="all pass")
    arrow(d, (920, 905), (920, 935))
    # refiner loop back to validator (2)
    orth(d, [(150, 880), (40, 880), (40, 360), (bx0 - 28, 360)], label="loop", color="#8a6d3b")
    # legend
    ly = 1000
    d.text((70, ly - 26), "Legend", font=font(14, True), fill=MUT)
    leg = [("LLM agent", "#6aa3e0"), ("validator", C_VALID), ("score / summarize", C_HL),
           ("decision", C_DEC), ("terminal", C_TERM), ("refiner", "#e08a6a")]
    lx = 70
    for name, col in leg:
        d.rounded_rectangle((lx, ly, lx + 26, ly + 20), radius=5, fill=col, outline="#9fb3c8")
        d.text((lx + 34, ly + 1), name, font=font(13), fill=INK); lx += 34 + d.textbbox((0, 0), name, font=font(13))[2] + 26
    im.save(OUT / "1_switchlingua_pipeline.png"); print("wrote", OUT / "1_switchlingua_pipeline.png")


# ============================================================ 2. Multi-Agent BERT
def diagram2():
    W, H = 1340, 1060; im = Image.new("RGB", (W, H), "#fbfcfe"); d = ImageDraw.Draw(im)
    title_band(d, W, "Multi-Agent BERT — classification architecture",
               "PipelineOrchestrator.run() · primary + router + specialists · 3 modes")
    cx = 600
    box(d, (cx - 130, 110, cx + 130, 148), "INPUT sentence → PipelineState", C_HEAD, 13, True, outline="#7ea6d8")
    box(d, (cx - 310, 174, cx + 310, 256),
        "PRIMARY CLASSIFIER\nmock = MockPrimaryClassifier (smoke only)\ntransformer = PrimaryTransformerClassifier (xlm-roberta-base)  ◄ Exp-C ckpt\n→ predicted_label + probs {pos, neg, neu}", C_HL, 13, True, outline="#e0ac63")
    box(d, (cx + 340, 178, cx + 540, 252), "MODE =\nprimary_only\n→ skip to FINAL", C_DATA, 12, True, outline="#a98fd0")
    arrow(d, (cx + 310, 215), (cx + 340, 215))
    box(d, (cx - 200, 286, cx + 200, 346), "ROUTER\nconfidence vs threshold →\n{ accept_primary | escalate }", C_DEC, 13, True, outline="#caa53a")
    arrow(d, (cx, 256), (cx, 286))
    box(d, (110, 400, 470, 474), "FAST path  (accept_primary)\nExplainabilityAgent\nshort 'why accepted' → final", C_AGENT, 13, True, accent="#6aa3e0")
    orth(d, [(cx - 120, 346), (290, 346), (290, 400)], label="accept")
    box(d, (560, 390, 1200, 500),
        "SLOW path  (escalate) — SPECIALIST AGENTS (parallel)\npaper_style : Lexical · Logic · Contextual\nfull_agentic: llm_Lexical · llm_Logic · Contextual (+ optional Deliberation)", C_AGENT, 12, True, accent="#7a6aa0")
    orth(d, [(cx + 120, 346), (880, 346), (880, 390)], label="escalate")
    box(d, (710, 530, 1050, 576), "ConsensusAgent → escalated label", C_AGENT, 13, True, outline="#aab9c9")
    arrow(d, (880, 500), (880, 530))
    box(d, (710, 606, 1050, 652), "ExplainabilityAgent (full)", C_AGENT, 13, True, outline="#aab9c9")
    arrow(d, (880, 576), (880, 606))
    box(d, (cx - 180, 720, cx + 180, 775), "FINAL: label + explanation + routing_info", C_TERM, 14, True, outline="#6aa86a")
    orth(d, [(290, 474), (290, 747), (cx - 180, 747)]); orth(d, [(880, 652), (880, 747), (cx + 180, 747)])
    box(d, (cx - 250, 825, cx + 250, 868), "NER task: parallel ner_{lexical, logic, contextual, consensus} agents", C_DATA, 12, outline="#a98fd0")
    box(d, (cx - 360, 905, cx + 360, 975),
        "MODES:  primary_only (◄ Exp C now)   ·   paper_style (non-LLM)   ·   full_agentic (LLM)\nError: any stage exception → state.extras['pipeline_error'] {stage, message}", C_HEAD, 12, True, outline="#7ea6d8")
    im.save(OUT / "2_multi_agent_bert.png"); print("wrote", OUT / "2_multi_agent_bert.png")


# ============================================================ 3. Exp C bridge
def diagram3():
    W, H = 1200, 1120; im = Image.new("RGB", (W, H), "#fbfcfe"); d = ImageDraw.Draw(im)
    title_band(d, W, "Experiment-C bridge — generator → data → classifier",
               "SwitchLingua generates the data; the fine-tuned model becomes the BERT primary")
    cx = 600
    steps = [
        ("SwitchLingua System B (gpt-4o-mini)\nconfig_sentiment_expC_v3.yaml : cs_ratio [50,60], Intrasentential", C_HL, "#e0ac63"),
        ("FILTER (every example)\nnon-empty → TaskValidator passed → deterministic CS-valid\n→ quality ≥ 7.0 → de-dup (normalized text)", C_AGENT, "#6aa3e0"),
        ("ACCUMULATE (resume-safe, append-only) — manage_sentiment_data.py\npilot_v1 (frozen) + daily_runs/ + completed_scenarios_v3.json\nmerge → cross-dedup → balance to N/label", C_DATA, "#a98fd0"),
        ("balanced train set  switchlingua_sentiment_train_*.jsonl  (text, label, +meta)", C_HEAD, "#7ea6d8"),
        ("fine-tune  xlm-roberta-base  (HF Trainer, Adafactor)", C_AGENT, "#7a6aa0"),
        ("checkpoint  expC_switchlingua_xlmr_240/  → load as PrimaryTransformerClassifier", C_HL, "#e0ac63"),
        ("evaluate_pipeline.py  --primary_model transformer  --pipeline_mode primary_only", C_AGENT, "#6aa3e0"),
        ("EESA test (818, REAL)  →  acc 0.590 / macro-F1 0.562\n(vs Exp A train-on-real-EESA: 0.831 / 0.819)", C_TERM, "#6aa86a"),
    ]
    y = 110; bh = 72; gap = 46; ys = []
    for txt, col, ac in steps:
        box(d, (cx - 470, y, cx + 470, y + bh), txt, col, 13, True, accent=ac); ys.append((y, y + bh)); y += bh + gap
    for i in range(len(ys) - 1):
        arrow(d, (cx, ys[i][1]), (cx, ys[i + 1][0]))
    d.text((cx - 462, ys[1][0] - 24), "raw ~4.7 instances / scenario", font=font(12), fill=MUT)
    im.save(OUT / "3_expC_bridge.png"); print("wrote", OUT / "3_expC_bridge.png")


if __name__ == "__main__":
    diagram1(); diagram2(); diagram3(); print("done ->", OUT)
