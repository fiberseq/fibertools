from asyncio.log import logger
import pandas as pd
import polars as pl
import numpy as np
import logging
import fibertools as ft
import pyranges as pr
import pickle

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def split_to_ints(df, col, sep=",", trim=True):
    """Split a columns with list of ints separated by
    "sep" into a numpy array quickly.

    Args:
        df (dataframe): dataframe that is like a bed12 file.
        col (str): column name within the dataframe to split up.
        sep (str, optional): Defaults to ",".
        trim (bool, optional): Remove the first and last call from the bed12 (removes bookendings). Defaults to True.

    Returns:
        column: New column that is a list of numpy array of ints.
    """
    if trim:
        return df[col].apply(
            lambda x: np.fromstring(x, sep=sep, dtype=np.int32)[1:-1],
            return_dtype=pl.datatypes.Object,
        )
    return df[col].apply(
        lambda x: np.fromstring(x, sep=sep, dtype=np.int32),
        return_dtype=pl.datatypes.Object,
    )


def join_msp_and_m6a(args):
    msp = ft.read_in_bed12_file(args.msp_bed12, n_rows=args.n_rows, tag="msp")
    logging.debug("Read in MSP file.")
    m6a = ft.read_in_bed12_file(args.m6a_bed12, n_rows=args.n_rows, tag="m6a")
    logging.debug("Read in m6a file.")
    both = m6a.join(msp, on=["ct", "st", "en", "fiber"]).drop(
        ["ten_msp", "tst_msp", "ten_m6a", "tst_m6a", "bsize_m6a"]
    )
    logging.debug("Joined m6a and MSP data.")
    return both


def n_overlaps(a_df, b_df):
    """returns the number of overlaps for each row in a_df found within b_df

    Args:
        a_df (dataframe): _description_
        b_df (dataframe): _description_

    Returns:
        numpy array: array with overlap counts.
    """
    a = pr.PyRanges(
        chromosomes=a_df.ct.to_list(),
        starts=a_df.st.to_list(),
        ends=a_df.en.to_list(),
    )
    b = pr.PyRanges(
        chromosomes=b_df.ct.to_list(),
        starts=b_df.st.to_list(),
        ends=b_df.en.to_list(),
    )
    return a.count_overlaps(b).NumberOverlaps.values


def disjoint_bins(contigs, start, ends, spacer_size=0):
    """returns bins that for the given intervals such that no intervals within a bin will overlap.
    INPUTS must be SORTED by start position!

    Args:
        contigs (list): list of start positions
        start (list): list of start positions
        ends (list): list of end positions
        spacer_size (int, optional): minimum space between intervals in the same bin. Defaults to 0.

    Returns:
        (list): A list of bins for each interval starting at 0.
    """
    logging.debug(f"{start[0]} {ends[0]} {spacer_size}")
    cur_contig = None
    bins = []
    for contig, st, en in zip(contigs, start, ends):
        if contig != cur_contig:
            max_bin = 0
            min_starts = [(-spacer_size, max_bin)]
        cur_contig = contig
        added = False
        for idx, (min_bin_st, b) in enumerate(min_starts):
            if st >= min_bin_st + spacer_size:
                min_starts[idx] = (en, b)
                bins.append(b)
                added = True
                break
        if not added:
            max_bin += 1
            min_starts.append((en, max_bin))
            bins.append(max_bin)

    return bins


def null_space_in_bed12(
    df,
    bed12_st_col="bst_msp",
    bed12_size_col="bsize_msp",
    make_thin=True,
    null_color="230,230,230",
):
    """Make a pandas df that is occupies the space between entires in bed12.

    Args:
        df (_type_): _description_
        bed12_st_col (str, optional): _description_. Defaults to "bst_msp".
        bed12_size_col (str, optional): _description_. Defaults to "bsize_msp".
        make_thin (bool, optional): _description_. Defaults to True.
        null_color (str, optional): _description_. Defaults to "230,230,230".

    Returns:
        pandas df: ~bed9 pandas df with null space between bed12 entries.
    """
    rows = pd.DataFrame(df.to_dicts()).copy()
    # filter missing rows
    rows = rows[~(rows[bed12_st_col].isnull() | rows[bed12_size_col].isnull())]
    rows["spacer_st"] = rows.apply(
        lambda row: (row[bed12_st_col] + row[bed12_size_col])[:-1] + row["st"], axis=1
    )
    rows["spacer_en"] = rows.apply(
        lambda row: row[bed12_st_col][1:] + row["st"], axis=1
    )
    z = rows.explode(["spacer_st", "spacer_en"])

    # think starts
    z["tst"] = z["spacer_st"]
    if make_thin:
        z["ten"] = z["spacer_st"]  # make non msp thin blocks
    else:
        z["ten"] = z["spacer_en"]

    z["color"] = null_color
    z.drop(columns=["st", "en"], inplace=True)
    z.rename(columns={"spacer_st": "st", "spacer_en": "en"}, inplace=True)

    # sometimes a couple of spacer_st are None, dropping them
    z.dropna(inplace=True)
    return z


def load_all(filename):
    """Load data from a pickle file.

    Args:
        filename (_type_): _description_

    Yields:
        _type_: _description_
    """
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
