"""Console script for fibertools."""
#!/usr/bin/env python3
import argparse
from cgi import test
from email.policy import default
import sys
import logging
from typing_extensions import Required
from fibertools.readutils import read_in_bed_file
from fibertools.trackhub import generate_trackhub
from fibertools.unionbg import bed2d4, make_q_values
from fibertools.add_nucleosomes import add_nucleosomes
import fibertools as ft
import numpy as np
import gzip
import pandas as pd


def make_bam2bed_parser(subparsers):
    parser = subparsers.add_parser(
        "bam2bed",
        help="Extract m6a calls from bam and output bed12.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def make_add_m6a_parser(subparsers):
    parser = subparsers.add_parser(
        "add-m6a",
        help="Add m6A tag",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bam", help="ccs bam file.")
    parser.add_argument("m6a", help="m6a bed12 file.")
    parser.add_argument(
        "-o", "--out", help="file to write output bam to.", default=sys.stdout
    )


def make_add_nucleosome_parser(subparsers):
    parser = subparsers.add_parser(
        "add-nucleosomes",
        help="Add Nucleosome and MSP tags",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Bam file after m6A calling has been done",
        type=argparse.FileType("r"),
        default=sys.stdin,
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output bam or json file.",
        default="-",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="pretrained hmm model (json).",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--num-train",
        help="Number of fibers used to train HMM.",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "-d",
        "--min-dist",
        help="Minimum distance from the start or end of a fiber for a accessible or nucelosome call.",
        type=int,
        default=46,
    )
    parser.add_argument(
        "-c", "--cutoff", type=int, default=65, help="hmm nucleosome size cutoff"
    )


def make_bed_split_parser(subparsers):
    parser = subparsers.add_parser(
        "split",
        help="Split a bed over many output files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bed", help="A bed file")
    parser.add_argument(
        "-g",
        "--genome",
        help="Input is a genome file. Split the genome to bed files with the same number of bases.",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--out-files", help="files to split input across", nargs="+"
    )


def make_trackhub_parser(subparsers):
    parser = subparsers.add_parser(
        "trackhub",
        help="Make a trackhub from a bed file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bed", help="A bed file")
    parser.add_argument(
        "genome_file", help="A file with chromosome sizes for the genome."
    )
    parser.add_argument("-r", "--ref", default="hg38")
    parser.add_argument("-t", "--trackhub-dir", default="trackHub")
    parser.add_argument("--bw", nargs="+", help="bw files to include", default=None)
    parser.add_argument(
        "--spacer-size",
        help="adjust minimum distance between fibers for them to be in the same bin.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-n",
        "--n-rows",
        help="For debugging only reads in n rows.",
        type=int,
        default=None,
    )


def make_bed2d4_parser(subparsers):
    parser = subparsers.add_parser(
        "bed2d4",
        help="Make a multi-track d4 file from a bed file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bed", help="A bed file")
    parser.add_argument("d4", help="Output d4 file")
    parser.add_argument(
        "-g",
        "--genome",
        help="A file with chromosome sizes for the genome.",
        required=True,
    )
    parser.add_argument(
        "-q",
        "--fdr",
        help="Make an FDR peaks d4 from coverage d4.",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--column",
        help="Name of the column to split the bed file on to make bed graphs.",
        default="name",
    )


def make_accessibility_model_parser(subparsers):
    parser = subparsers.add_parser(
        "model",
        help="Make MSP features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ft-all",
        help="ft extract --all file. Can be compressed.",
        action="store_true",
    )
    parser.add_argument("all", help="all output from ft-extract.")
    # parser.add_argument("m6a_bed12", help="m6a bed12 file.", nargs="?")
    # parser.add_argument(
    #     "genome",
    #     help="Indexed fasta file to read in sequence context. Should have the same chromosomes as referenced in the first column of the two bed files.",
    #     nargs="?",
    # )
    parser.add_argument("-d", "--dhs", help="dhs", default=None)
    parser.add_argument(
        "-b", "--bin_width", help="size of bin width to use", type=int, default=40
    )
    parser.add_argument(
        "--bin-num",
        help="Number of feature bins to work. Must be an odd number.",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--min-tp-msp-len",
        help="min msp length to be in the DHS positive training set.",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--spacer-size",
        help="adjust minimum distance between fibers for them to be in the same bin.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--train-fdr", help="Training FDR used by mokapot.", type=float, default=0.10
    )
    parser.add_argument(
        "--test-fdr", help="Testing FDR used by mokapot.", type=float, default=0.05
    )
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument(
        "-o", "--out", help="Write the accessibility model to this file."
    )
    parser.add_argument(
        "-n",
        "--n-rows",
        help="For debugging only reads in n rows.",
        type=int,
        default=None,
    )


def write_bed_tuple_to_file(f, regions):
    odf = pd.DataFrame(regions)
    odf.to_csv(f, sep="\t", header=None, index=None)
    return (odf[2] - odf[1]).sum()


def split_bed_over_files(args):
    if args.genome:
        genome = pd.read_csv(args.bed, sep="\s+", header=None)
        genome = genome.rename(columns={0: "ct", 1: "length"})
        window_size = int(genome.length.sum() / len(args.out_files)) + 1

        needed_bases = window_size
        added_bases = 0
        regions = []
        out_idx = 0
        for ct, length in zip(genome.ct, genome.length):
            chrom_position = 0
            while True:
                logging.debug(
                    f"Needed bases: {needed_bases}, chrom_position: {ct}:{chrom_position}"
                )
                # regions has all the bases we need for this chunk already
                if needed_bases == 0:
                    added_bases += write_bed_tuple_to_file(
                        args.out_files[out_idx], regions
                    )
                    # reset needed_bases
                    needed_bases = window_size
                    regions = []
                    out_idx += 1

                # we need to add to regions
                if chrom_position + needed_bases <= length:
                    regions.append((ct, chrom_position, chrom_position + needed_bases))
                    chrom_position += needed_bases
                    needed_bases = 0
                else:
                    regions.append((ct, chrom_position, length))
                    needed_bases -= length - chrom_position
                    # reached the end of this chrom break
                    break
        # write last regions chunk
        added_bases += write_bed_tuple_to_file(args.out_files[out_idx], regions)
        # check all bp added
        assert added_bases == genome.length.sum()
    else:
        bed = ft.read_in_bed_file(args.bed, keep_header=True)
        logging.debug("Read in bed file.")
        index_splits = np.array_split(np.arange(bed.shape[0]), len(args.out_files))
        for index, out_file in zip(index_splits, args.out_files):
            if out_file.endswith(".gz"):
                with gzip.open(out_file, "wb") as f:
                    bed[index].write_csv(f, sep="\t", has_header=True)
            else:
                bed[index].write_csv(out_file, sep="\t", has_header=True)


def parse():
    """Console script for fibertools."""
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available subcommand for fibertools",
        required=True,
    )
    make_bam2bed_parser(subparsers)
    make_add_m6a_parser(subparsers)
    make_add_nucleosome_parser(subparsers)
    make_accessibility_model_parser(subparsers)
    make_bed_split_parser(subparsers)
    make_trackhub_parser(subparsers)
    make_bed2d4_parser(subparsers)
    # shared arguments
    parser.add_argument("-t", "--threads", help="n threads to use", type=int, default=1)
    parser.add_argument(
        "-v", "--verbose", help="increase logging verbosity", action="store_true"
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=ft.__version__),
    )
    args = parser.parse_args()

    log_format = "[%(levelname)s][Time elapsed (ms) %(relativeCreated)d]: %(message)s"
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(format=log_format, level=log_level)

    if args.command == "add_m6a":
        _m6a = ft.read_in_bed12_file(args.m6a)
    elif args.command == "add-nucleosomes":
        logging.debug("Adding nucleosomes")
        add_nucleosomes(args)
    elif args.command == "model":
        # if args.ft_all is not None:
        fiberdata = ft.Fiberdata_rs(args.all, n_rows=args.n_rows)
        # check if we have data
        if fiberdata.all.shape[0] == 0:
            if args.model is None:
                logging.warning("No MSPs found in training set.")
                sys.exit(1)
            else:
                logging.warning("No MSPs found in MSP bed file.")
                with open(args.out, "w") as f:
                    f.write("")
                sys.exit(0)
        fiberdata.make_msp_features(bin_num=args.bin_num, bin_width=args.bin_width)
        # else:
        # fiberdata = ft.Fiberdata(
        #     args.msp_bed12,
        #     args.m6a_bed12,
        #     n_rows=args.n_rows,
        # )
        # AT_genome = ft.make_AT_genome(args.genome, fiberdata.both)
        # fiberdata.make_msp_features(
        #     AT_genome, bin_num=args.bin_num, bin_width=args.bin_width
        # )
        if args.dhs is not None:
            dhs = read_in_bed_file(args.dhs, n_rows=args.n_rows, pandas=True)
        else:
            dhs = None
        fiberdata.make_percolator_input(dhs_df=dhs, min_tp_msp_len=args.min_tp_msp_len)

        if args.model is None:
            logging.debug("Training model.")
            fiberdata.train_accessibility_model(
                args.out, train_fdr=args.train_fdr, test_fdr=args.test_fdr
            )
        else:
            logging.debug("Loading model from file and predicting.")
            logging.debug(
                f"Max FDR allowed for an accessibility call: {args.train_fdr}"
            )
            fiberdata.predict_accessibility(args.model, max_fdr=args.train_fdr)
            fiberdata.accessibility.to_csv(args.out, sep="\t", index=False)
    elif args.command == "split":
        split_bed_over_files(args)
    elif args.command == "trackhub":
        df = pd.read_csv(args.bed, sep="\t", nrows=args.n_rows)
        generate_trackhub(
            df,
            trackhub_dir=args.trackhub_dir,
            ref=args.ref,
            genome_file=args.genome_file,
            spacer_size=args.spacer_size,
            bw=args.bw,
        )
    elif args.command == "bed2d4":
        if args.fdr:
            logging.debug("Making fdr peaks.")
            make_q_values(args.bed, args.d4)
        else:
            bed2d4(args)

    return 0
