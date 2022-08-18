from email import header
import os
import sys
from .utils import disjoint_bins
import pandas as pd

hub = """
hub fiberseq
shortLabel fiberseq
longLabel fiberseq
genomesFile genomes.txt
email mvollger.edu
"""

genomes = """
genome {ref}
trackDb trackDb.txt
"""

track_comp = """
track fiberseq
compositeTrack on
shortLabel fiberseq
longLabel fiberseq
type bigBed 9 +
maxItems 100000
maxHeightPixels 200:200:1
"""

sub_comp_track = """
    track bin{i}
    parent fiberseq
    bigDataUrl bins/bin.{i}.bed.bb
    shortLabel bin{i}
    longLabel bin{i}
    priority {i}
    type bigBed 9 +
    itemRgb on
    visibility {viz}
    maxHeightPixels 1:1:1
"""

bw_comp = """
track FDR_track
compositeTrack on
shortLabel FDR track
longLabel FDR track
type bigWig 0 1000
visibility dense
autoScale on
maxItems 100000
maxHeightPixels 200:200:1
"""

bw_template = """
    track {nm}
    parent FDR_track
    bigDataUrl {file}
    shortLabel {nm}
    longLabel {nm}
    type bigWig
    autoScale on
    alwaysZero on
    visibility full
    priority {i}
    maxHeightPixels 100:100:1
"""

multi_wig = """
track fiberseq_coverage
longLabel fiberseq_coverage
shortLabel fiberseq_coverage
container multiWig
aggregate stacked
showSubtrackColorOnUi on
type bigWig 0 1000
autoScale on
alwaysZero on
visibility full
maxHeightPixels 100:32:8
    
    track Accessible 
    parent fiberseq_coverage
    bigDataUrl {acc}
    type bigWig
    color 139,0,0
    
    track Linker
    parent fiberseq_coverage
    bigDataUrl {link}
    type bigWig
    color 147,112,219
    
    track Nucleosomes 
    parent fiberseq_coverage
    bigDataUrl {nuc}
    type bigWig
    color 169,169,169
    """


def generate_trackhub(
    df,
    trackhub_dir="trackHub",
    ref="hg38",
    spacer_size=100,
    genome_file="data/hg38.chrom.sizes",
    bw=None,
):
    os.makedirs(f"{trackhub_dir}/", exist_ok=True)

    open(f"{trackhub_dir}/hub.txt", "w").write(hub)
    open(f"{trackhub_dir}/genomes.txt", "w").write(genomes.format(ref=ref))
    trackDb = open(f"{trackhub_dir}/trackDb.txt", "w")

    # write the bins to file
    os.makedirs(f"{trackhub_dir}/bed", exist_ok=True)
    os.makedirs(f"{trackhub_dir}/bins", exist_ok=True)

    # only run if bigWigs are passed
    if bw is not None:
        os.makedirs(f"{trackhub_dir}/bw", exist_ok=True)
        trackDb.write(bw_comp)
        nuc = None
        acc = None
        link = None
        for idx, bw_f in enumerate(bw):
            base = os.path.basename(bw_f)
            nm = base.rstrip(".bw")
            # file = f"{trackhub_dir}/bw/{base}"
            file = f"bw/{base}"
            sys.stderr.write(f"{bw_f}\t{nm}\t{file}\n")
            if nm == "nuc":
                nuc = file
            elif nm == "acc":
                acc = file
            elif nm == "link":
                link = file
            else:
                sys.stderr.write(f"Stacked bigWig!")
                trackDb.write(bw_template.format(i=idx + 1, nm=nm, file=file))

        if nuc is not None and acc is not None and link is not None:
            trackDb.write(multi_wig.format(acc=acc, link=link, nuc=nuc))

    # bin files
    trackDb.write(track_comp)
    viz = "dense"
    for i in range(75):
        trackDb.write(sub_comp_track.format(i=i + 1, viz=viz))
        if i >= 50:
            viz = "hide"

    # done with track db
    trackDb.close()

    fiber_df = (
        df.groupby(["#ct", "fiber"])
        .agg({"st": "min", "en": "max"})
        .reset_index()
        .sort_values(["#ct", "st", "en"])
    )
    fiber_df["bin"] = disjoint_bins(
        fiber_df["#ct"], fiber_df.st, fiber_df.en, spacer_size=spacer_size
    )
    df = df.merge(fiber_df[["fiber", "bin"]], on=["fiber"])
    for cur_bin in sorted(df.bin.unique()):
        if cur_bin > 75:
            continue
        sys.stderr.write(f"\r{cur_bin}")
        out_file = f"{trackhub_dir}/bed/bin.{cur_bin}.bed"
        bb_file = f"{trackhub_dir}/bins/bin.{cur_bin}.bed.bb"
        (
            df.iloc[:, 0:9]
            .loc[df.bin == cur_bin]
            .sort_values(["#ct", "st", "en"])
            .to_csv(out_file, sep="\t", index=False, header=False)
        )
        os.system(f"bedToBigBed {out_file} {genome_file} {bb_file}")
        os.system(f"rm {out_file}")
