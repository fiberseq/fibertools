import os
import sys
from .utils import disjoint_bins
import pandas as pd
import logging
import polars as pl
import psutil


def log_mem_usage():
    gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    logging.info(f"Using {gb:.2f} GB of memory")


HUB = """
hub fiberseq-{sample}
shortLabel fiberseq-{sample}
longLabel fiberseq-{sample}
genomesFile genomes.txt
email mvollger.edu
"""

GENOMES = """
genome {ref}
trackDb trackDb.txt
"""

TRACK_COMP = """
track reads-{sample}-{hap}
compositeTrack on
shortLabel reads-{sample}-{hap}
longLabel reads-{sample}-{hap}
type bigBed 9 +
maxItems 100000
maxHeightPixels 200:200:1
"""

SUB_COMP_TRACK = """
    track bin-{i}-{sample}-{hap}
    parent reads-{sample}-{hap}
    bigDataUrl bins/{hap}.bin.{i}.bed.bb
    shortLabel {hap}.bin.{i}
    longLabel {hap}.bin.{i}
    priority {i}
    type bigBed 9 +
    itemRgb on
    visibility {viz}
    maxHeightPixels 1:1:1
"""

BW_COMP = """
track FDR-{sample}-{hap}
compositeTrack on
shortLabel FDR track
longLabel FDR track
type bigWig 0 1000
autoScale on
maxItems 100000
maxHeightPixels 200:200:1
"""

BW_TEMPLATE = """
    track FDR.{hap}.{nm}
    parent FDR-{sample}-{hap}
    bigDataUrl {file}
    shortLabel FDR.{sample}.{hap}.{nm}
    longLabel FDR.{sample}.{hap}.{nm}
    type bigWig
    autoScale on
    alwaysZero on
    visibility dense
    priority {i}
    maxHeightPixels 100:100:1
"""

MULTI_WIG = """
track coverage-{sample}-{hap}
longLabel coverage-{sample}-{hap}
shortLabel coverage-{sample}-{hap}
container multiWig
aggregate stacked
showSubtrackColorOnUi on
type bigWig 0 1000
autoScale on
alwaysZero on
visibility full
maxHeightPixels 200:200:8
    
    track Accessible-{sample}-{hap}
    parent coverage-{sample}-{hap}
    bigDataUrl {acc}
    type bigWig
    color 139,0,0
    
    track Linker-{sample}-{hap}
    parent coverage-{sample}-{hap}
    bigDataUrl {link}
    type bigWig
    color 147,112,219
    
    track Nucleosomes-{sample}-{hap}
    parent coverage-{sample}-{hap}
    bigDataUrl {nuc}
    type bigWig
    color 169,169,169
    """


def generate_trackhub(
    trackhub_dir="trackHub", ref="hg38", bw=None, sample="Sample", max_bins=None
):
    os.makedirs(f"{trackhub_dir}/", exist_ok=True)
    open(f"{trackhub_dir}/hub.txt", "w").write(HUB.format(sample=sample))
    open(f"{trackhub_dir}/genomes.txt", "w").write(GENOMES.format(ref=ref))
    trackDb = open(f"{trackhub_dir}/trackDb.txt", "w")
    for hap in ["all", "hap1", "hap2", "unk"]:
        # add coverage tracks
        acc = f"bw/{hap}.acc.bw"
        nuc = f"bw/{hap}.nuc.bw"
        link = f"bw/{hap}.link.bw"
        trackDb.write(
            MULTI_WIG.format(acc=acc, link=link, nuc=nuc, sample=sample, hap=hap)
        )

        # add fdr tracks
        trackDb.write(BW_COMP.format(sample=sample, hap=hap))
        for idx, nm in enumerate(["", ".90", ".100"]):
            file = f"bw/fdr.{hap}{nm}.bw"
            trackDb.write(
                BW_TEMPLATE.format(nm=nm, file=file, sample=sample, hap=hap, i=idx + 1)
            )

        # bin files
        trackDb.write(TRACK_COMP.format(sample=sample, hap=hap))
        viz = "dense"
        for i in range(max_bins):
            trackDb.write(
                SUB_COMP_TRACK.format(i=i + 1, viz=viz, sample=sample, hap=hap)
            )
            if i >= 50:
                viz = "hide"
    # done with track db
    trackDb.close()


def make_bins(
    df,
    outs,
    spacer_size=100,
):
    max_bins = len(outs)
    # touch all the files
    for out in outs:
        open(out, "w").write("#empty_bed\n")

    if df.shape[0] == 0:
        logging.info("No data in this bed file.")
        return 0

    logging.info(f"{df}")
    log_mem_usage()
    fiber_df = (
        df.lazy()
        .groupby(["#ct", "fiber"])
        .agg([pl.min("st"), pl.max("en")])
        .sort(["#ct", "st", "en"])
    ).collect()
    logging.info("Made fiber df.")
    bins = disjoint_bins(
        fiber_df["#ct"], fiber_df["st"], fiber_df["en"], spacer_size=spacer_size
    )
    fiber_df = fiber_df.with_column(
        pl.Series(bins).alias("bin"),
    )
    logging.info(f"{fiber_df}")
    logging.info("Merging with bins.")
    logging.info("Made binned fibers")
    log_mem_usage()
    # for cur_bin in sorted(df["bin"].unique()):
    for cur_bin, cur_df in (
        df.join(fiber_df.select(["fiber", "bin"]), on=["fiber"])
        .partition_by(groups="bin", as_dict=True)
        .items()
    ):
        log_mem_usage()
        # maintain_order=True,
        if cur_bin >= max_bins:
            continue
        if cur_df.shape[0] == 0:
            logging.info(f"No data in bin {cur_bin}.")
            continue
        logging.info(f"Writing {cur_df.shape} elements in {cur_bin}.")
        cur_df.select(cur_df.columns[0:9]).sort(["#ct", "st", "en"]).write_csv(
            outs[cur_bin], sep="\t", has_header=False
        )
    return 0
