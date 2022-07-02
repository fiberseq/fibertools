import os
import sys
from .utils import disjoint_bins
import pandas as pd


def generate_trackhub(
    df,
    trackhub_dir="trackHub",
    ref="hg38",
    spacer_size=100,
    genome_file="data/hg38.chrom.sizes",
    bw=None,
):
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
visibility dense
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
    visibility dense
    maxHeightPixels 1:1:1
    
    """

    bw_template = """
track {nm}
bigDataUrl {file}
shortLabel {nm}
longLabel {nm}
type bigWig
visibility dense
priority {i}
maxHeightPixels 100:100:1
    """

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
        for idx, bw_f in enumerate(bw):
            nm = os.path.basename(bw_f).rstrip(".bw")
            file = (bw_f.lstrip(f"{trackhub_dir}")).lstrip("/") 
            sys.stderr.write(f"{bw_f}\t{nm}\t{file}\n")
            trackDb.write(bw_template.format(i=idx + 1, nm=nm, file=file))
    
    # bin files 
    trackDb.write(track_comp)
    for i in range(75):
        trackDb.write(sub_comp_track.format(i=i + 1))
        
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
        (df.iloc[:, 0:9].loc[df.bin == cur_bin].to_csv(out_file, sep="\t", index=False))
        os.system(f"bedToBigBed {out_file} {genome_file} {bb_file}")
        os.system(f"rm {out_file}")
