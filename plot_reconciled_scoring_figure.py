"""Render the audited Figure 4 with larger Ubuntu text and the original layout.

Run with conda run --no-capture-output -n st273_phylogeny_env python
plot_candidate_scoring_figure4.py. Uses the audited top-25/n>10 candidate table.
"""

import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
INPUT=ROOT.parent/"Supplementary_Table_S3.xlsx"
OUT=ROOT.parent
STEM="Figure 3"
PREVALENCE=[("MDR","MDR_prevalence"),("ESBL","ESBL_prevalence"),
            ("Carb.","carbapenemase_prevalence"),("Virulence","virulence_marker_prevalence"),
            ("R-V conv.","amr_virulence_convergence_prevalence")]
BREADTH=[("Countries","n_countries"),("Source\ncategories","n_core_source_categories"),
         ("Year\nspan","year_span"),("Median\nreplicons","median_replicon_family_count"),
         ("Clinical\nAMR isolates","clinically_important_amr_n")]


def sizes(values,low,high):
    a=np.asarray(values,dtype=float)
    return np.full_like(a,low) if a.max()==a.min() else low+(a-a.min())/(a.max()-a.min())*(high-low)


def panel(ax,letter):
    ax.text(-0.08,1.035,letter,transform=ax.transAxes,fontsize=20,fontweight="bold",va="bottom")


def main():
    OUT.mkdir(exist_ok=True)
    d=pd.read_excel(INPUT,sheet_name="Candidate_13").sort_values("rank").reset_index(drop=True)
    assert len(d)==13 and d.n_isolates.sum()==300
    assert d.set_index("ST").loc["ST641","n_countries"]==10
    assert d.set_index("ST").loc["ST1562","n_countries"]==12
    mpl.rcParams.update({"font.family":"Ubuntu","font.size":12,"svg.fonttype":"none",
                         "pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"})
    cmap=LinearSegmentedColormap.from_list("original_blue",["#F7FBFF","#C6DBEF","#6BAED6","#2171B5","#08306B"])
    fig=plt.figure(figsize=(18.0,8.6))
    grid=GridSpec(1,3,figure=fig,width_ratios=[1.15,0.95,1.9],left=.07,right=.98,top=.94,bottom=.25,wspace=.36)
    a=fig.add_subplot(grid[0,0]);b=fig.add_subplot(grid[0,1]);c=fig.add_subplot(grid[0,2])
    y=np.arange(len(d));sts=d.ST.tolist()

    a.barh(y,d.risk_score,color="#5E81AC",edgecolor="#263238",linewidth=.5)
    count_size=sizes(d.n_isolates,50,240)
    a.scatter(np.full(len(d),31.0),y,s=count_size,c="#88C0D0",edgecolor="#263238",linewidth=.5,zorder=3)
    marker_text_checks=[]
    for i,r in d.iterrows():
        a.text(r.risk_score-.3,i,str(int(r.risk_score)),ha="right",va="center",fontsize=12.5,color="white",fontweight="bold")
        t=a.text(32.55,i,f"n={int(r.n_isolates)}",ha="left",va="center",fontsize=12,color="#333333")
        marker_text_checks.append((a,31.0,i,count_size[i],t))
    a.set_xlim(18,36.5);a.set_xticks([20,24,28,30]);a.tick_params(axis="x",labelsize=12)
    a.set_xlabel("Composite Surveillance-Priority\nScore",fontsize=14,labelpad=9)
    a.set_axisbelow(True)
    a.grid(axis="x",color="#D9D9D9",linewidth=.6)
    a.spines[["top","right"]].set_visible(False)
    panel(a,"A")

    prev=d[[col for _,col in PREVALENCE]].to_numpy(dtype=float)
    im=b.imshow(prev,aspect="auto",cmap=cmap,vmin=0,vmax=100)
    b.set_xticks(np.arange(len(PREVALENCE)),[name for name,_ in PREVALENCE],rotation=90,ha="center",va="top",fontsize=12.5)
    b.tick_params(axis="x",length=0,pad=7)
    for i,j in np.ndindex(prev.shape):
        value=prev[i,j]
        b.text(j,i,f"{value:.0f}",ha="center",va="center",fontsize=11.5,color="white" if value>=55 else "#222222")
    cb=fig.colorbar(im,ax=b,fraction=.045,pad=.035)
    cb.set_label("Prevalence (%)",fontsize=12,labelpad=7)
    cb.ax.tick_params(labelsize=11)
    panel(b,"B")

    for j,(_,col) in enumerate(BREADTH):
        values=d[col].to_numpy(dtype=float)
        marker_size=sizes(values,45,230)
        c.scatter(np.full(len(d),j),y,s=marker_size,c=values,cmap=cmap,
                  norm=Normalize(vmin=0,vmax=max(values.max(),1)),edgecolor="#263238",linewidth=.35)
        for i,value in enumerate(values):
            label=f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
            t=c.text(j+.24,i,label,va="center",fontsize=12.5,color="#222222")
            marker_text_checks.append((c,j,i,marker_size[i],t))
    overlap_x=len(BREADTH)+.25
    overlap=d.human_nonhuman_overlap.astype(str).str.lower().eq("true")
    c.scatter(np.full(len(d),overlap_x),y,s=105,c=np.where(overlap,"#111111","#BDBDBD"),edgecolor="black",linewidth=.35)
    c.set_xticks([*range(len(BREADTH)),overlap_x],[name for name,_ in BREADTH]+["Human-\nnon-human\noverlap"],fontsize=12)
    c.set_xlim(-.6,len(BREADTH)+.9)
    c.tick_params(axis="x",length=0,pad=9)
    c.spines[["top","right","left","bottom"]].set_visible(False)
    panel(c,"C")
    for ax in [a,b,c]:
        ax.set_yticks(y,sts,fontsize=13)
        ax.set_ylim(len(d)-.5,-.5)
        ax.tick_params(axis="y",length=0,pad=5)

    handles=[Line2D([0],[0],marker="o",linestyle="",markerfacecolor=color,markeredgecolor="black",markersize=10,label=label)
             for color,label in [("#111111","human/non-human overlap"),("#BDBDBD","no human/non-human overlap")]]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.035),ncol=2,frameon=False,fontsize=13)

    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[]
    for ax in fig.axes:
        texts.extend([*ax.texts,*ax.get_xticklabels(),*ax.get_yticklabels(),ax.xaxis.label,ax.yaxis.label])
    for legend in fig.legends:texts.extend(legend.get_texts())
    texts=[t for t in texts if t.get_visible() and t.get_text().strip()]
    overlaps=[]
    for x,z in itertools.combinations(texts,2):
        if x.get_window_extent(renderer).overlaps(z.get_window_extent(renderer)):
            overlaps.append([x.get_text(),z.get_text()])
    gaps=[]
    for ax,x,ypos,size,t in marker_text_checks:
        radius=np.sqrt(size)/2*fig.dpi/72
        gap=t.get_window_extent(renderer).x0-(ax.transData.transform((x,ypos))[0]+radius)
        gaps.append(float(gap))
    clipped=[t.get_text() for t in texts if not fig.bbox.fully_contains(t.get_window_extent(renderer).x0,t.get_window_extent(renderer).y0)
             or not fig.bbox.fully_contains(t.get_window_extent(renderer).x1,t.get_window_extent(renderer).y1)]
    report={"input":str(INPUT),"ST_order":sts,"n_genomes":int(d.n_isolates.sum()),
            "text_pairs_overlapping":overlaps,"text_outside_canvas":clipped,"minimum_marker_text_gap_pixels":min(gaps)}
    (OUT/"analysis/Figure_3_layout_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    assert not overlaps,overlaps
    assert not clipped,clipped
    assert min(gaps)>1,gaps
    for ext in ["svg","pdf","png"]:fig.savefig(OUT/f"{STEM}.{ext}",dpi=300)
    fig.savefig(OUT/"analysis/Figure_3_review_preview.png",dpi=120)
    plt.close(fig)
    caption="""Figure 3. Composite genomic and epidemiological profiles of candidate surveillance-priority K. variicola sequence types.

Thirteen exact STs comprising 300 genomes were selected from the 25 highest-ranked MLST designations after restriction to groups with >10 genomes; all met Tier 1 criteria. Rows follow their current composite-score ranking. (A) Composite scores; circles and adjacent annotations indicate genome counts. The score axis is truncated. (B) Prevalence of genotypic MDR, ESBL positivity, carbapenemase carriage, major virulence markers and AMR-virulence convergence. Cells show percentages rounded to whole numbers; blue intensity uses a common 0-100% scale. (C) Known countries or territories, core source categories, collection-year span, median number of distinct PlasmidFinder replicon designations, genomes with specified clinically important AMR determinants, and human/non-human source overlap. Unknown geography is excluded; country counts are 10 for ST641 and 12 for ST1562. Circle area and blue intensity are scaled separately within each quantitative column; numbers provide the underlying values. Black and grey markers denote presence and absence of source overlap, respectively. Source overlap alone does not imply resistance in both source groups or transmission. Replicon counts do not establish plasmid numbers or AMR-gene localisation. Scoring criteria are defined in Table 1, with candidate and full-ranking data in Supplementary Table S3.
"""
    (OUT/f"{STEM}_caption.txt").write_text(caption)
    print(json.dumps(report,indent=2))
    print("Figure 4 and caption written to",OUT)


if __name__=="__main__":main()
