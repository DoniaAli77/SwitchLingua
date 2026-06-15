"""Render the Multi-Agent BERT pipeline (both task paths) as a polished PNG (Pillow only)."""
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
import os, math

W, H = 1780, 1300
BG = (247, 248, 251)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

def font(sz, bold=False):
    p = "C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()

F_TITLE=font(38,True); F_SUB=font(20); F_BOX=font(24,True); F_AG=font(20,True)
F_SMALL=font(19); F_TINY=font(16); F_LBL=font(19,True); F_HEAD=font(21,True)

C_PRIMARY=(76,130,195); C_ROUTER=(232,160,60); C_FAST=(79,174,122)
C_AGENT=(142,116,201); C_CONS=(214,92,108); C_EXP=(111,169,138); C_FINAL=(88,94,112)
C_NER=(56,162,172); C_NERC=(40,138,150)
SHADOW=(228,230,237); WHITE=(255,255,255); INK=(38,42,54); MUT=(120,124,138)
BAR_L=(226,234,246); BAR_R=(220,240,241)

def shadow(x,y,w,h,r=18): d.rounded_rectangle([x+5,y+7,x+w+5,y+h+7], radius=r, fill=SHADOW)

def rbox(x,y,w,h,fill,text,tcol=WHITE,sub=None,fnt=F_BOX,r=16,scol=None):
    shadow(x,y,w,h,r); d.rounded_rectangle([x,y,x+w,y+h], radius=r, fill=fill)
    blocks=[(ln,fnt) for ln in text.split("\n")]
    if sub: blocks += [(ln,F_TINY) for ln in sub.split("\n")]
    th=sum(d.textbbox((0,0),t,font=f)[3]+6 for t,f in blocks)
    cy=y+(h-th)//2
    for t,f in blocks:
        bb=d.textbbox((0,0),t,font=f); tw=bb[2]-bb[0]
        d.text((x+(w-tw)//2,cy),t,font=f,fill=(scol if (f is F_TINY and scol) else tcol)); cy+=bb[3]-bb[1]+6
    return (x,y,x+w,y+h)

def head(p2,ang,col,a=14):
    d.polygon([p2,(p2[0]-a*math.cos(ang-0.42),p2[1]-a*math.sin(ang-0.42)),
               (p2[0]-a*math.cos(ang+0.42),p2[1]-a*math.sin(ang+0.42))], fill=col)
def line(pts,col,w=4,arrow=True):
    d.line(pts,fill=col,width=w,joint="curve")
    if arrow:
        (x1,y1),(x2,y2)=pts[-2],pts[-1]; head((x2,y2),math.atan2(y2-y1,x2-x1),col)
def ctext(cx,y,t,f,col):
    bb=d.textbbox((0,0),t,font=f); d.text((cx-(bb[2]-bb[0])//2,y),t,font=f,fill=col)

# ---- title + top dispatch ---------------------------------------------------
d.text((40,26),"Multi-Agent BERT — Pipeline (two task paths)",font=F_TITLE,fill=INK)
d.text((42,78),"Input is dispatched by task_type into the classification path or the sequence-labeling (NER) path",
       font=F_SUB,fill=MUT)

MID=890
rbox(MID-120,128,240,52,(64,70,84),"INPUT TEXT",r=26)
rbox(MID-170,210,340,58,(120,124,140),"DISPATCH  ·  by task_type",fnt=F_AG)
line([(MID,180),(MID,210)],INK,5)

GREY=(150,154,166)
# split to the two panels
d.line([(MID,268),(MID,288)],fill=GREY,width=4)
ctext((MID+450)//2+10,260,"text classification",F_LBL,C_PRIMARY)
ctext((MID+1320)//2+10,260,"sequence labeling",F_LBL,C_NERC)
line([(MID,288),(450,288),(450,300)],C_PRIMARY,4)
line([(MID,288),(1320,288),(1320,300)],C_NERC,4)

# ===== LEFT PANEL : CLASSIFICATION ==========================================
d.rounded_rectangle([45,307,865,1217],radius=20,fill=(252,252,254),outline=(214,218,228),width=2)
d.rounded_rectangle([45,300,865,346],radius=14,fill=BAR_L)
ctext(455,310,"CLASSIFICATION PATH   ·   topic / sentiment",F_HEAD,(40,70,120))
CXL=455
rbox(CXL-200,366,400,84,C_PRIMARY,"PRIMARY CLASSIFIER",sub="XLM-R / mBERT → label, confidence")
line([(CXL,346),(CXL,366)],GREY,4)
rbox(CXL-200,484,400,90,C_ROUTER,"ROUTER",sub="accept_primary if conf ≥ threshold  ·  else escalate",fnt=F_BOX)
line([(CXL,450),(CXL,484)],GREY,4)
RB=574
# primary_only bypass
ctext(120,470,"primary_only",F_TINY,C_PRIMARY)
line([(CXL-200,530),(95,530),(95,1150)],C_PRIMARY,4,arrow=False)
# fast path  (beside the router, not over the agents)
rbox(698,496,158,72,C_FAST,"FAST PATH",sub="→ primary",fnt=F_AG)
line([(CXL+200,532),(698,532)],C_FAST,4)
d.text((582,470),"accept_primary",font=F_TINY,fill=C_FAST)
# escalate to agents
ctext(CXL+58,588,"escalate",F_LBL,C_CONS)
line([(CXL,574),(CXL,612)],C_CONS,4,arrow=False)
# three agents + fan-out bus
aw=176; gap=16; ax0=CXL-(3*aw+2*gap)//2
cxs=[ax0+i*(aw+gap)+aw//2 for i in range(3)]
busA=620; line([(CXL,612),(CXL,busA)],GREY,3,arrow=False)
d.line([(cxs[0],busA),(cxs[2],busA)],fill=GREY,width=3)
for i,t in enumerate(["LEXICAL","LOGIC","CONTEXTUAL"]):
    line([(cxs[i],busA),(cxs[i],648)],GREY,3); rbox(ax0+i*(aw+gap),648,aw,86,C_AGENT,t,fnt=F_AG)
d.text((ax0,742),"paper_style: deterministic   ·   full_agentic: LLM (GPT-4o-mini)",font=F_TINY,fill=MUT)
busB=772;
for c in cxs: line([(c,734),(c,busB)],GREY,3,arrow=False)
d.line([(cxs[0],busB),(cxs[2],busB)],fill=GREY,width=3)
rbox(CXL-185,802,370,138,C_CONS,"CONSENSUS",
     sub="score[L] += weight × agent_conf\nscore[primary] += w_primary × primary_conf\nlabel = argmax(score) / Σ weights",fnt=F_BOX)
line([(CXL,busB),(CXL,802)],GREY,3)
rbox(CXL-165,968,330,50,C_EXP,"EXPLAINABILITY → audit",fnt=F_SMALL)
line([(CXL,940),(CXL,968)],GREY,3)
# left final
rbox(CXL-205,1108,410,86,C_FINAL,"FINAL OUTPUT",sub="label · confidence · explanation · history[]")
line([(CXL,1018),(CXL,1108)],INK,4)
line([(95,530),(95,1151),(CXL-205,1151)],C_PRIMARY,4)
line([(777,568),(777,1151),(CXL+205,1151)],C_FAST,4)

# ===== RIGHT PANEL : NER ====================================================
d.rounded_rectangle([915,307,1735,1217],radius=20,fill=(252,253,253),outline=(214,222,224),width=2)
d.rounded_rectangle([915,300,1735,346],radius=14,fill=BAR_R)
ctext(1325,310,"SEQUENCE LABELING PATH   ·   NER",F_HEAD,(20,90,100))
CXR=1325
rbox(CXR-185,366,370,80,C_PRIMARY,"NER PRIMARY",sub="token classifier (XLM-R / mBERT)")
line([(CXR,346),(CXR,366)],GREY,4)
ner=["NER LEXICAL","NER LOGIC","NER CONTEXTUAL","NER CONSENSUS"]
y0=476
for i,t in enumerate(ner):
    yy=y0+i*100
    rbox(CXR-185,yy,370,76,C_NERC if "CONSENSUS" in t else C_NER,t)
    if i==0: line([(CXR,446),(CXR,yy)],GREY,4)
    else:    line([(CXR,yy-24),(CXR,yy)],GREY,4)
ey=y0+4*100
rbox(CXR-165,ey,330,52,C_EXP,"EXPLAINABILITY → audit",fnt=F_SMALL)
line([(CXR,ey-24),(CXR,ey)],GREY,4)
d.text((CXR-300,ey+70),"NER primary tags, then the agent panel refines & merges spans.",font=F_TINY,fill=MUT)
d.text((CXR-300,ey+92),"paper_style / full_agentic   ·   primary_only → primary only.",font=F_TINY,fill=MUT)
# right final
rbox(CXR-205,1108,410,86,C_FINAL,"FINAL OUTPUT",sub="tagged spans · explanation · history[]")
line([(CXR,ey+52),(CXR,1108)],INK,4)

d.text((48,1244),"Modes (both paths): primary_only (no agents) · paper_style (non-LLM) · full_agentic (LLM).  "
       "Classification escalates on low confidence; NER always runs its agent panel.",
       font=F_TINY,fill=MUT)

out="experiments/outputs/multi_agent_bert/architecture_pipeline.png"
img.save(out,"PNG"); print("saved",out,img.size)
