import pyd4
import tempfile
from joblib import Parallel
import numpy as np
from numba import njit
import logging
import fibertools as ft
import polars as pl
import sys


@njit
def chrom_bg(sts, ens, chrom_len):
    chrom = np.zeros(chrom_len, dtype=np.int32)
    to_add = np.int32(1)
    for st, en in zip(sts, ens):
        chrom[st:en] += to_add
    return chrom


def df_to_bg(df, chrom, genome):
    cov = chrom_bg(df["st"].to_numpy(), df["en"].to_numpy(), genome[chrom])
    return cov


def make_d4_from_df(df, genome, d4_f):
    chroms = list(zip(genome.keys(), genome.values()))
    writer = (
        pyd4.D4Builder(d4_f)
        .add_chroms(chroms)
        .for_sparse_data()
        .generate_index()
        .get_writer()
    )
    for g in df.groupby("ct"):
        chrom = g["ct"][0]
        data = df_to_bg(g, chrom, genome)
        logging.debug(f"Writing {chrom} to d4. Has a total of {data.sum()} values.")
        writer.write_np_array(chrom, 0, data)
    writer.close()


def make_temp_d4_from_df(df, genome):
    temp = tempfile.NamedTemporaryFile(suffix=".d4")
    make_d4_from_df(df, genome, temp.name)
    return temp


def make_union_d4_from_df(df, genome, group_col, d4_f):
    out_files = []
    for idx, g in enumerate(df.groupby([group_col])):
        g_n = g[group_col][0]
        logging.debug(f"Making d4 for group: {g_n}")
        out_files.append((g_n, make_temp_d4_from_df(g, genome)))

    merged = pyd4.D4Merger(d4_f)
    for tag, d4 in sorted(out_files):
        logging.debug(f"{tag} sum: {pyd4.D4File(d4.name)['chr11'].sum()}")
        merged.add_tagged_track("q_" + str(tag), d4.name)
    merged.merge()
    # close files
    [d4.close() for _tag, d4 in out_files]


def bed2d4(args):
    df = ft.read_in_bed_file(args.bed)
    if args.column == "score":
        # set high fdr values to the max
        # df = df.with_column(
        #    pl.when(pl.col("column_5") >= 30)
        #    .then(100)
        #    .otherwise(pl.col("column_5"))
        #    .alias("tmp_score")
        # )
        # set give nucleosomes their own score
        df = df.with_column(
            pl.when(pl.col("column_9") == "230,230,230")
            .then(101)
            .otherwise(pl.col("column_5"))
            .alias(args.column)
        )
    genome = {line.split()[0]: int(line.split()[1]) for line in open(args.genome)}
    make_union_d4_from_df(df, genome, args.column, args.d4)


@njit(parallel=True)
def make_summary_stats(matrix, log_q_values=None, adjust_for_coverage=True):
    y = matrix.T
    log_q_vals = (y[:, :-2] * log_q_values).sum(axis=1)
    acc_cov = y[:, :-2].sum(axis=1)
    link_cov = y[:, -2]
    nuc_cov = y[:, -1]
    cov = acc_cov + link_cov + nuc_cov
    # adjust for expected amount of coverage
    if adjust_for_coverage:
        log_q_vals = log_q_vals - cov / 10 * (-10 * np.log10(0.1))
    # assert nuc_cov.sum() == link_cov.sum()
    return (log_q_vals, acc_cov, link_cov, nuc_cov)


def make_q_values(in_d4, out_d4):
    file = pyd4.D4File(in_d4)
    chroms = file.chroms()
    matrix = file.open_all_tracks()
    track_names = matrix.track_names
    # these are the q values
    q_values = np.array([max(int(x.strip("q_")) / 100, 0.01) for x in track_names])
    log_q_values = -10 * np.log10(q_values[:-2])
    one_minus_q_values = 1 - q_values[:-2]

    logging.debug(f"chroms: {chroms}")

    # output file
    m = pyd4.D4Merger(out_d4)
    # To enumerate the matrix
    out_temp_files = []
    for idx in range(4):
        temp = tempfile.NamedTemporaryFile(suffix=".d4")
        w = (
            pyd4.D4Builder(temp.name)
            .add_chroms(chroms)
            .for_sparse_data()
            .generate_index()
            .get_writer()
        )
        out_temp_files.append((temp, w))

    for ct, ct_len in chroms:
        logging.debug(f"Processing q-values for chrom: {ct}")
        bin_size = 5_000_000
        cur_st = 0
        cur_en = bin_size
        while True:  # cur_en < ct_len and cur_st < ct_len:
            if cur_en > ct_len:
                cur_en = ct_len

            cur_mat = matrix[ct, cur_st, cur_en]
            idx = 0
            for data in make_summary_stats(cur_mat, log_q_values=log_q_values):
                logging.debug(
                    f"Writing {ct} {cur_st} {cur_en} with index {idx} to d4. Mean is {data.mean()}"
                )
                w = out_temp_files[idx][1]
                w.write_np_array(ct, cur_st, data)
                idx += 1

            if logging.DEBUG >= logging.root.level:
                sys.stderr.write(f"\r[DEBUG]: {ct} {cur_en/ct_len:.2%}")
            if cur_en == ct_len:
                break
            cur_st += bin_size
            cur_en += bin_size

    # finish writing to temp files
    for _temp, w in out_temp_files:
        w.close()

    # merge files
    for idx in range(4):
        m.add_tagged_track(f"{idx}", out_temp_files[idx][0].name)
    m.merge()

    # close temp files
    for temp, _w in out_temp_files:
        temp.close()
