import os
import sys
from .utils import disjoint_bins
import pandas as pd
import logging
import polars as pl


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
track fiberseq-{sample}
compositeTrack on
shortLabel fiberseq-{sample}
longLabel fiberseq-{sample}
type bigBed 9 +
maxItems 100000
maxHeightPixels 200:200:1
"""

SUB_COMP_TRACK = """
    track bin{i}-{sample}
    parent fiberseq-{sample}
    bigDataUrl bins/bin.{i}.bed.bb
    shortLabel bin{i}
    longLabel bin{i}
    priority {i}
    type bigBed 9 +
    itemRgb on
    visibility {viz}
    maxHeightPixels 1:1:1
"""

BW_COMP = """
track FDR-track-{sample}
compositeTrack on
shortLabel FDR track
longLabel FDR track
type bigWig 0 1000
visibility full
autoScale on
maxItems 100000
maxHeightPixels 200:200:1
"""

BW_TEMPLATE = """
    track {nm}
    parent FDR-track-{sample}
    bigDataUrl {file}
    parent FDR-track-{sample}
    shortLabel FDR-{sample}-{nm}
    longLabel FDR-{sample}-{nm}
    type bigWig
    autoScale on
    alwaysZero on
    visibility full
    priority {i}
    maxHeightPixels 100:100:1
"""

MULTI_WIG = """
track fiberseq-coverage-{sample}
longLabel fiberseq-coverage-{sample}
shortLabel fiberseq-coverage-{sample}
container multiWig
aggregate stacked
showSubtrackColorOnUi on
type bigWig 0 1000
autoScale on
alwaysZero on
visibility full
maxHeightPixels 200:200:8
    
    track Accessible 
    parent fiberseq-coverage-{sample}
    bigDataUrl {acc}
    type bigWig
    color 139,0,0
    
    track Linker
    parent fiberseq-coverage-{sample}
    bigDataUrl {link}
    type bigWig
    color 147,112,219
    
    track Nucleosomes 
    parent fiberseq-coverage-{sample}
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

    # only run if bigWigs are passed
    if bw is not None:
        os.makedirs(f"{trackhub_dir}/bw", exist_ok=True)
        trackDb.write(BW_COMP.format(sample=sample))
        nuc = None
        acc = None
        link = None
        for idx, bw_f in enumerate(bw):
            base = os.path.basename(bw_f)
            nm = base.rstrip(".bw")
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
                trackDb.write(
                    BW_TEMPLATE.format(i=idx + 1, nm=nm, file=file, sample=sample)
                )

        if nuc is not None and acc is not None and link is not None:
            trackDb.write(MULTI_WIG.format(acc=acc, link=link, nuc=nuc, sample=sample))

        # bin files
        trackDb.write(TRACK_COMP.format(sample=sample))
        viz = "dense"
        for i in range(max_bins):
            trackDb.write(SUB_COMP_TRACK.format(i=i + 1, viz=viz, sample=sample))
            if i >= 50:
                viz = "hide"
        # done with track db
        trackDb.close()


def make_bins_old(
    df,
    trackhub_dir="trackHub",
    spacer_size=100,
    genome_file="data/hg38.chrom.sizes",
    max_bins=None,
):
    # write the bins to file
    os.makedirs(f"{trackhub_dir}/bed", exist_ok=True)
    os.makedirs(f"{trackhub_dir}/bins", exist_ok=True)

    fiber_df = (
        df.groupby(["#ct", "fiber"])
        .agg({"st": "min", "en": "max"})
        .reset_index()
        .sort_values(["#ct", "st", "en"])
    )
    logging.info("Made fiber df.")
    fiber_df["bin"] = disjoint_bins(
        fiber_df["#ct"], fiber_df.st, fiber_df.en, spacer_size=spacer_size
    )
    logging.info("Made binned fibers")

    df = df.merge(fiber_df[["fiber", "bin"]], on=["fiber"])
    for cur_bin in sorted(df.bin.unique()):
        if cur_bin > max_bins:
            continue
        logging.info(f"Writting {cur_bin}.")
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


def make_bins(
    df,
    trackhub_dir="trackHub",
    spacer_size=100,
    genome_file="data/hg38.chrom.sizes",
    max_bins=None,
):
    # df.columns = [c.strip("#") for c in df.columns]

    # write the bins to file
    os.makedirs(f"{trackhub_dir}/bed", exist_ok=True)
    os.makedirs(f"{trackhub_dir}/bins", exist_ok=True)
    logging.info(f"{df}")
    fiber_df = (
        df.groupby(["#ct", "fiber"])
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
    df = df.collect().join(fiber_df.select(["fiber", "bin"]), on=["fiber"])
    logging.info("Made binned fibers")
    # for cur_bin in sorted(df["bin"].unique()):
    for cur_bin, cur_df in df.partition_by(
        groups="bin", maintain_order=True, as_dict=True
    ).items():
        if cur_bin > max_bins:
            continue
        logging.info(f"Writing {cur_df.shape} elements in {cur_bin}.")
        out_file = f"{trackhub_dir}/bed/bin.{cur_bin}.bed"
        bb_file = f"{trackhub_dir}/bins/bin.{cur_bin}.bed.bb"
        cur_df.select(cur_df.columns[0:9]).sort(["#ct", "st", "en"]).write_csv(
            out_file, sep="\t", has_header=False
        )
        os.system(f"bedToBigBed {out_file} {genome_file} {bb_file}")
        os.system(f"rm {out_file}")
