from dataclasses import dataclass
import pandas as pd
import polars as pl
import logging
import fibertools as ft
import sys
import numpy as np
from .classify import get_msp_features
import pickle
import mokapot


@dataclass
class Fiberdata_rs:
    """A class for storing and manipulating fiberseq data.

    Returns:
        Fiberdata: A class object with dataframes for fiberseq data.
    """

    all: pl.internals.frame.DataFrame
    features: pd.core.frame.DataFrame
    pin: pd.core.frame.DataFrame
    accessibility: pd.core.frame.DataFrame

    def __init__(self, all_file: str, n_rows: int = None):
        """Make a new Fiberdata object.

        Args:
            ft_all (str): Path to a `ft extract --all` table.
            n_rows (int, optional): Only read a subset of the data for testing. Defaults to None.
        """
        self.all = ft.read_fibertools_rs_all_file(all_file, n_rows=n_rows)
        logging.debug("Finished reading ft-extract --all table")

    def get_msp_features(self, row, bin_width=40, bin_num=5):
        # is_at = AT_genome[row["ct"]][row["st"] : row["en"]]
        seq = np.frombuffer(bytes(row["fiber_sequence"], "utf-8"), dtype="S1")
        is_at = (seq == b"T") | (seq == b"A")

        typed_bst_m6a = row["msp_starts"]
        rtn = []
        mids = ft.classify.get_msp_mid(0, row["msp_starts"], row["msp_lengths"])
        all_bin_starts = ft.classify.make_bin_starts(
            mids, bin_width=bin_width, bin_num=bin_num
        )
        fiber_AT_count = is_at.sum()
        fiber_m6a_count = row["m6a"].shape[0]

        for msp_st, msp_size, bin_starts, ref_msp_st, ref_msp_size in zip(
            row["msp_starts"],
            row["msp_lengths"],
            all_bin_starts,
            row["ref_msp_starts"],
            row["ref_msp_lengths"],
        ):
            msp_en = msp_st + msp_size
            if msp_en <= msp_st:
                continue
            m6a_counts = ft.classify.get_bin_m6a(
                bin_starts, typed_bst_m6a, bin_width=bin_width
            )
            AT_count = ft.classify.get_bin_AT(bin_starts, is_at, bin_width=bin_width)
            msp_AT = is_at[msp_st:msp_en].sum()
            msp_m6a = ((typed_bst_m6a >= msp_st) & (typed_bst_m6a < msp_en)).sum()
            rtn.append(
                {
                    "ct": row["ct"],
                    "st": ref_msp_st,
                    "en": ref_msp_st + ref_msp_size,
                    "fiber": row["fiber"],
                    "score": row["score"],
                    "msp_len": msp_size,
                    "fiber_m6a_count": fiber_m6a_count,
                    "fiber_AT_count": fiber_AT_count,
                    "fiber_m6a_frac": fiber_m6a_count / fiber_AT_count,
                    "msp_m6a": msp_m6a,
                    "msp_AT": msp_AT,
                    "m6a_count": m6a_counts,
                    "AT_count": AT_count,
                }
            )
        return rtn

    def make_msp_features(self, bin_width=40, bin_num=9):
        msp_stuff = []
        rows = self.all.to_dicts()
        for idx, row in enumerate(rows):
            if row["msp_starts"] is None or row["msp_lengths"] is None:
                continue
            if logging.DEBUG >= logging.root.level:
                sys.stderr.write(
                    f"\r[DEBUG]: Added features to {(idx+1)/len(rows):.3%} of MSPs."
                )
            msp_stuff += self.get_msp_features(
                row, bin_width=bin_width, bin_num=bin_num
            )
        logging.debug("")
        logging.debug("Expanding bed12s into individual MSPs.")

        z = pd.DataFrame(msp_stuff)
        mean_msp = np.mean(z.en - z.st)
        logging.debug(f"mean msp size {mean_msp}")
        # for some reason the MSPs sometimes have negative lengths
        # z = z[(z["st"] < z["en"])]
        # Make more MSP featrues columns
        z["bin_m6a_frac"] = z.m6a_count / z.AT_count
        z["m6a_frac"] = z.msp_m6a / z.msp_AT
        m6a_fracs = pd.DataFrame(
            z["bin_m6a_frac"].tolist(),
            columns=[f"m6a_frac_{i}" for i in range(bin_num)],
        )
        m6a_counts = pd.DataFrame(
            z["m6a_count"].tolist(),
            columns=[f"m6a_count_{i}" for i in range(bin_num)],
        )
        AT_counts = pd.DataFrame(
            z["AT_count"].tolist(),
            columns=[f"AT_count_{i}" for i in range(bin_num)],
        )
        out = (
            pd.concat([z, m6a_fracs, m6a_counts, AT_counts], axis=1)
            .drop(["bin_m6a_frac", "m6a_count", "AT_count"], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        self.features = out

    def make_percolator_input(self, dhs_df=None, sort=False, min_tp_msp_len=40):
        """write a input file that works with percolator.

        Args:
            msp_features_df (_type_): _description_
            out_file (_type_): _description_

        return None
        """
        # need to add the following columns: SpecId	Label Peptide Proteins
        # and need to remove the following columns:
        to_remove = ["ct", "st", "en", "fiber"]

        out_df = self.features.copy()
        out_df.insert(1, "Label", 0)
        if dhs_df is not None:
            dhs_null = ft.utils.n_overlaps(self.features, dhs_df[dhs_df.name != "DHS"])
            dhs_true = ft.utils.n_overlaps(self.features, dhs_df[dhs_df.name == "DHS"])
            out_df.loc[dhs_true > 0, "Label"] = 1
            out_df.loc[dhs_null > 0, "Label"] = -1

            # if the msp is short make it null
            out_df.loc[
                (dhs_true > 0) & (out_df.msp_len <= min_tp_msp_len), "Label"
            ] = -1
        else:
            # has no effect, percolator just needs some items to have positive labels
            out_df.loc[out_df.msp_len >= min_tp_msp_len, "Label"] = 1

        # add and drop columns needed for mokapot
        out_df.drop(to_remove, axis=1, inplace=True)
        if "SpecId" in out_df.columns:
            out_df.drop("SpecId", axis=1, inplace=True)
        out_df.insert(0, "SpecId", out_df.index)
        out_df["Peptide"] = out_df.SpecId
        out_df["Proteins"] = out_df.SpecId
        out_df["scannr"] = out_df.SpecId
        out_df["log_msp_len"] = np.log(out_df["msp_len"])

        if sort:
            out_df.sort_values(["Label"], ascending=False, inplace=True)
        self.pin = out_df

    def train_accessibility_model(self, out_file: str, train_fdr=0.1, test_fdr=0.05):
        logging.debug(
            f"Using {train_fdr} fdr for training and {test_fdr} fdr for testing."
        )
        moka_conf, models = ft.classify.make_accessibility_model(
            self.pin, train_fdr=train_fdr, test_fdr=test_fdr
        )
        with open(out_file, "wb") as f:
            pickle.dump(moka_conf.psms, f)
            pickle.dump(models, f)

    def make_nucleosome_accessibility_df(self):
        df = pd.DataFrame(
            self.all[["ct", "fiber", "ref_nuc_starts", "ref_nuc_lengths"]].to_dicts()
        )
        logging.debug(f"Expanding into individual nucleosomes. {df.shape[0]}")
        df = df.explode(["ref_nuc_starts", "ref_nuc_lengths"])
        df["st"] = df.ref_nuc_starts
        df["en"] = df.ref_nuc_starts + df.ref_nuc_lengths
        df["score"] = 1
        df["strand"] = "+"
        df["tst"] = df["st"]
        df["ten"] = df["en"]
        df["color"] = "230,230,230"
        df["qValue"] = 1
        logging.debug(f"Finished expanding into individual nucleosomes. {df.shape[0]}")
        return df

    def predict_accessibility(self, model_file: str, max_fdr=0.30):
        moka_conf_psms, models = list(ft.utils.load_all(model_file))

        test_psms = mokapot.read_pin(self.pin)
        all_scores = [model.predict(test_psms) for model in models]
        # scores = np.mean(np.array(all_scores), axis=0)
        # scores = np.amin(np.array(all_scores), axis=0)
        scores = np.amax(np.array(all_scores), axis=0)

        q_values = ft.classify.find_nearest_q_values(
            moka_conf_psms["mokapot score"], moka_conf_psms["mokapot q-value"], scores
        )
        merged = self.pin.copy()
        merged["mokapot score"] = scores
        merged["mokapot q-value"] = q_values

        out = self.features
        out["qValue"] = 1
        out["strand"] = "+"
        out.loc[merged.SpecId, "qValue"] = merged["mokapot q-value"]
        out["tst"] = out["st"]
        out["ten"] = out["en"]
        out["color"] = "147,112,219"

        # set the values above the max_fdr to 1
        out.loc[out.qValue >= max_fdr, "qValue"] = 1

        # out.loc[out.qValue <= 0.10, "color"] = "255,255,0"
        out.loc[out.qValue <= 0.10, "color"] = "255,140,0"
        out.loc[out.qValue <= 0.05, "color"] = "255,0,0"
        out.loc[out.qValue <= 0.01, "color"] = "139,0,0"

        out_cols = [
            "ct",
            "st",
            "en",
            "fiber",
            "score",
            "strand",
            "tst",
            "ten",
            "color",
            "qValue",
            # "bin",
        ]
        final_out = (
            pd.concat(
                [out[out_cols], self.make_nucleosome_accessibility_df()[out_cols]]
            )
            .sort_values(["ct", "st"])
            .rename(columns={"ct": "#ct"})
        )
        final_out["score"] = (final_out.qValue * 100).astype(int)
        # final_out.to_csv(args.out, sep="\t", index=False)
        self.accessibility = final_out[
            (final_out.st >= 0) & (final_out.st < final_out.en)
        ]
