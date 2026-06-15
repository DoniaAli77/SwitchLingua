"""Render the Multi-Agent BERT architecture as a PNG (Pillow only, no extra deps)."""
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1800, 1700
BG = (255, 255, 255)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# ---- fonts ------------------------------------------------------------------
def font(sz, bold=False):
    base = "C:/Windows/Fonts/"
    for name in ([ "arialbd.ttf"] if bold else ["arial.ttf"]):
        p = base + name
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

F_TITLE = font(40, True)
F_SUB   = font(22)
F_BOX   = font(24, True)
F_SMALL = font(19)
F_TINY  = font(17)
F_MONO  = font(17)
F_LBL   = font(19, True)

# ---- palette ----------------------------------------------------------------
C_PRIMARY = (70, 130, 200)
C_ROUTER  = (230, 150, 40)
C_FAST    = (80, 170, 110)
C_AGENT   = (140, 110, 190)
C_DELIB   = (170, 150, 210)
C_CONS    = (210, 80, 80)
C_FINAL   = (90, 95, 105)
C_PANEL   = (245, 246, 250)
C_PANELED = (200, 205, 215)
WHITE = (255, 255, 255)
INK   = (35, 40, 50)

def rbox(x, y, w, h, fill, text, tcol=WHITE, sub=None, fnt=F_BOX, outline=None, r=16, dashed=False):
    d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill,
                        outline=outline or fill, width=3)
    if dashed:  # draw dashed border on top
        d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=None,
                            outline=(120,120,130), width=2)
    lines = text.split("\n")
    th = sum(d.textbbox((0,0), ln, font=fnt)[3] for ln in lines)
    if sub: th += len(sub.split("\n")) * 22 + 6
    cy = y + (h - th) // 2
    for ln in lines:
        bb = d.textbbox((0,0), ln, font=fnt); tw = bb[2]-bb[0]
        d.text((x + (w-tw)//2, cy), ln, font=fnt, fill=tcol); cy += bb[3]-bb[1] + 6
    if sub:
        cy += 4
        for ln in sub.split("\n"):
            bb = d.textbbox((0,0), ln, font=F_TINY); tw = bb[2]-bb[0]
            d.text((x + (w-tw)//2, cy), ln, font=F_TINY, fill=tcol); cy += 22
    return (x, y, x+w, y+h)

def arrow(p1, p2, col=INK, w=4, label=None, lcol=INK, ldx=8):
    d.line([p1, p2], fill=col, width=w)
    import math
    ang = math.atan2(p2[1]-p1[1], p2[0]-p1[0]); a=14
    d.polygon([p2,
               (p2[0]-a*math.cos(ang-0.4), p2[1]-a*math.sin(ang-0.4)),
               (p2[0]-a*math.cos(ang+0.4), p2[1]-a*math.sin(ang+0.4))], fill=col)
    if label:
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        d.text((mx+ldx, my-22), label, font=F_LBL, fill=lcol)

# ---- title ------------------------------------------------------------------
d.text((40, 28), "Multi-Agent BERT — System Architecture", font=F_TITLE, fill=INK)
d.text((42, 80), "Fast primary classifier + confidence-gated escalation to a specialist agent panel with consensus adjudication",
       font=F_SUB, fill=(90,95,110))

CX = 560  # main column center
BW = 360

# ---- INPUT ------------------------------------------------------------------
y = 140
rbox(CX-110, y, 220, 50, (60,65,75), "INPUT TEXT", WHITE, r=24)
b_in = (CX, y+50)

# ---- PRIMARY ----------------------------------------------------------------
y = 230
rbox(CX-BW//2, y, BW, 92, C_PRIMARY,
     "1 · PRIMARY CLASSIFIER",
     sub="XLM-R / mBERT (or mock)\n→ label, confidence, probabilities", fnt=F_BOX)
arrow(b_in, (CX, y), w=5)
prim_bottom = (CX, y+92)

# ---- ROUTER -----------------------------------------------------------------
y = 392
rbox(CX-BW//2, y, BW, 96, C_ROUTER,
     "2 · ROUTER",
     sub="accept_primary  if conf >= threshold\nescalate           otherwise", fnt=F_BOX)
arrow(prim_bottom, (CX, y), w=5)
router_b = (CX-BW//2, y, CX+BW//2, y+96)

# primary_only bypass (left, curving down to final)
d.text((40, y+10), "mode = primary_only", font=F_LBL, fill=C_PRIMARY)
d.text((40, y+34), "(router + agents skipped)", font=F_TINY, fill=(120,120,130))
arrow((CX-BW//2, y+30), (150, y+30), col=C_PRIMARY, w=4)
d.line([(150, y+30),(150, 1210)], fill=C_PRIMARY, width=4)

# ---- branch: FAST path ------------------------------------------------------
yfast = 560
fastx = CX+250
rbox(fastx-150, yfast, 300, 80, C_FAST,
     "FAST PATH", sub="Explainability (short)\nfinal = primary output", fnt=F_BOX)
arrow((CX+BW//2, y+48), (fastx, yfast), col=C_FAST, w=4, label="accept_primary", lcol=C_FAST, ldx=-150)
# fast → final
d.line([(fastx, yfast+80),(fastx, 1210)], fill=C_FAST, width=4)

# ---- SLOW path: specialist panel container ---------------------------------
arrow((CX, y+96), (CX, 600), w=5, label="escalate", lcol=C_CONS)
panel_x, panel_y, panel_w, panel_h = 250, 600, 620, 560
d.rounded_rectangle([panel_x, panel_y, panel_x+panel_w, panel_y+panel_h],
                    radius=18, fill=C_PANEL, outline=C_PANELED, width=3)
d.text((panel_x+20, panel_y+12), "3 · SPECIALIST PANEL  (escalation path)", font=F_LBL, fill=(90,80,120))

# three agents
ay = panel_y+70; aw=180; gap=14; ax0 = panel_x+18
labels = [("LEXICAL","surface cues"),("LOGIC","rules / negation"),("CONTEXTUAL","meaning in context")]
centers=[]
for i,(t,s) in enumerate(labels):
    ax = ax0 + i*(aw+gap)
    rbox(ax, ay, aw, 96, C_AGENT, t, sub=s, fnt=F_BOX)
    centers.append((ax+aw//2, ay+96))
    arrow((CX, panel_y), (ax+aw//2, ay), w=3, col=(150,150,160))
d.text((panel_x+18, ay+102), "paper_style: deterministic agents      full_agentic: LLM agents (GPT-4o-mini)",
       font=F_TINY, fill=(110,110,125))

# deliberation (dashed, optional)
dy = ay+150
rbox(panel_x+150, dy, 320, 64, C_DELIB,
     "DELIBERATION  (optional)", tcol=INK, sub="full_agentic & enable_deliberation", fnt=F_SMALL, dashed=True)
for c in centers: arrow(c, (panel_x+310, dy), w=2, col=(170,170,180))

# consensus
cy = dy+96
rbox(panel_x+90, cy, 440, 150, C_CONS,
     "CONSENSUS AGENT",
     sub="score[L] += weight x agent_confidence\nFix#2: + w_primary x primary_confidence\nabstain -> defer to primary (never labels[0])\ntie-break: primary / votes / contrib / alpha",
     fnt=F_BOX)
arrow((panel_x+310, dy+64), (panel_x+310, cy), w=3)

# explainability full
ey = cy+168
rbox(panel_x+120, ey, 380, 56, (110,150,120),
     "EXPLAINABILITY (full)  -> rationale + audit", fnt=F_SMALL)
arrow((panel_x+310, cy+150), (panel_x+310, ey), w=3)
slow_bottom = (panel_x+310, ey+56)

# ---- FINAL ------------------------------------------------------------------
fy = 1245
rbox(CX-260, fy, 520, 86, C_FINAL,
     "FINAL OUTPUT",
     sub="label · confidence · explanation · full audit history[]", fnt=F_BOX)
arrow(slow_bottom, (CX-60, fy), w=4)              # slow
arrow((fastx, 1210), (CX+120, fy), col=C_FAST, w=4)   # fast
arrow((150, 1210), (CX-200, fy), col=C_PRIMARY, w=4)  # primary_only

# state banner
d.rounded_rectangle([950, 150, 1760, 300], radius=14, fill=(248,249,252), outline=C_PANELED, width=2)
d.text((968, 162), "PipelineState  (threaded through every stage)", font=F_LBL, fill=INK)
for i,ln in enumerate([
    "input_text · task_config(labels, threshold, pipeline_mode,",
    "  enable_deliberation) · primary_model_output · routing_info",
    "lexical/logic/contextual_output · deliberation_output",
    "consensus_output · final_output · explanation_output",
    "history[] (immutable audit) · extras{pipeline_error}"]):
    d.text((968, 196+i*21), ln, font=F_MONO, fill=(70,75,90))

# ---- right-side annotation panels ------------------------------------------
def panel(x,y,w,h,title):
    d.rounded_rectangle([x,y,x+w,y+h], radius=14, fill=(248,249,252), outline=C_PANELED, width=2)
    d.text((x+18,y+14), title, font=F_LBL, fill=INK)

# modes
panel(950, 330, 810, 250, "RUN MODES")
rows = [
 ("primary_only", "primary -> final  (router & agents skipped) — BERT baseline"),
 ("paper_style",  "primary -> router -> Lexical+Logic+Contextual -> consensus  (non-LLM)"),
 ("full_agentic", "primary -> router -> LLM agents (+delib) -> consensus  (GPT-4o-mini)"),
]
for i,(m,desc) in enumerate(rows):
    yy=375+i*58
    d.rounded_rectangle([968,yy,1140,yy+44], radius=10, fill=C_AGENT if i==2 else (C_FAST if i==0 else C_ROUTER))
    d.text((980,yy+11), m, font=F_SMALL, fill=WHITE)
    d.text((1158,yy+12), desc, font=F_TINY, fill=(60,65,80))

# Fix#2 vs Fix#3
panel(950, 600, 810, 220, "AUDIT FIXES — primary participation")
d.text((968,648), "Fix #2  (default ON, w_primary=1.0):", font=F_SMALL, fill=C_CONS)
d.text((986,674), "primary VOTES at consensus time, after agents decide", font=F_TINY, fill=INK)
d.text((986,696), "independently.  Win: paper_style +0.064 acc; safe.", font=F_TINY, fill=INK)
d.text((968,732), "Fix #3  (default OFF):", font=F_SMALL, fill=C_ROUTER)
d.text((986,758), "shows agents the primary's label BEFORE they vote", font=F_TINY, fill=INK)
d.text((986,780), "(_primary_block.py).  +3-7pt copying, NO accuracy gain.", font=F_TINY, fill=INK)

# CLI knobs
panel(950, 840, 810, 250, "KEY CLI SEAMS (no default change)")
knobs = [
 ("--pipeline_mode", "primary_only / paper_style / full_agentic"),
 ("--threshold", "router escalation cutoff (best 0.9)"),
 ("--consensus_primary_weight", "Fix #2 w_primary (0 = legacy agents-only)"),
 ("--agents_use_primary_signal", "Fix #3 primary-signal block (off)"),
 ("--transformer_checkpoint", "primary model weights"),
 ("--llm_client / --llm_model", "openai / mock ; gpt-4o-mini"),
]
for i,(k,v) in enumerate(knobs):
    yy=884+i*33
    d.text((968,yy), k, font=F_MONO, fill=C_PRIMARY)
    d.text((1320,yy), v, font=F_TINY, fill=(60,65,80))

# results snapshot
panel(950, 1110, 810, 230, "HEADLINE RESULTS  (EESA test, XLM-R)")
res = [
 ("primary_only (Exp A, 2464 real)", "0.8240 acc / 0.8088 macroF1"),
 ("full_agentic best (th 0.9, Fix#2)", "0.8509 acc / 0.8401 macroF1"),
 ("Exp C primary_only (240 generated)", "0.5905 acc / 0.5619 macroF1  [pilot]"),
]
for i,(k,v) in enumerate(res):
    yy=1158+i*52
    d.text((968,yy), k, font=F_SMALL, fill=INK)
    d.text((986,yy+24), v, font=F_TINY, fill=(80,110,80))

# NER path note
panel(950, 1360, 810, 150, "NER PATH (task_type = sequence_labeling)")
d.text((968,1404), "NERLexical -> NERLogic -> NERContextual -> NERConsensus", font=F_MONO, fill=(70,75,90))
d.text((968,1432), "-> Explainability   (separate deterministic panel)", font=F_MONO, fill=(70,75,90))
d.text((968,1466), "Same agents in paper_style & full_agentic (LLM NER slot reserved).", font=F_TINY, fill=(110,110,125))

# footer
d.text((950, 1560), "Error handling: any stage exception -> extras['pipeline_error']{stage,message,traceback}; chain stops, upstream preserved.",
       font=F_TINY, fill=(120,120,130))
d.text((950, 1584), "897 tests passing · fully offline (mock LLM + mock/real primary).", font=F_TINY, fill=(120,120,130))

out = "experiments/outputs/multi_agent_bert/architecture_diagram.png"
img.save(out, "PNG")
print("saved", out, img.size)
