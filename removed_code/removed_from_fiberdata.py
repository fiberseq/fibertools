@dataclass
class Fiberdata:
    """A class for storing and manipulating fiberseq data.

    Returns:
        Fiberdata: A class object with dataframes for fiberseq data.
    """

    msp: pl.internals.frame.DataFrame
    m6a: pl.internals.frame.DataFrame
    both: pl.internals.frame.DataFrame
    features: pd.core.frame.DataFrame
    pin: pd.core.frame.DataFrame
    accessibility: pd.core.frame.DataFrame

    def __init__(self, msp_file: str, m6a_file: str, n_rows: int = None):
        """Make a new Fiberdata object.

        Args:
            msp_file (str): A bed12 file with msp data.
            m6a_file (str): A bed12 file with m6a data.
            n_rows (int, optional): Only read a subset of the data for testing. Defaults to None.
        """
        self.msp = ft.read_in_bed12_file(msp_file, n_rows=n_rows, tag="msp")
        logging.debug("Read in MSP file.")
        self.m6a = ft.read_in_bed12_file(m6a_file, n_rows=n_rows, tag="m6a")
        logging.debug("Read in m6a file.")
        self.join_msp_and_m6a()

    def join_msp_and_m6a(self):
        self.both = self.m6a.join(self.msp, on=["ct", "st", "en", "fiber"]).drop(
            ["ten_msp", "tst_msp", "ten_m6a", "tst_m6a", "bsize_m6a"]
        )
        logging.debug("Joined m6a and MSP data.")

    def make_msp_features(self, AT_genome, bin_width=40, bin_num=9):
        msp_stuff = []
        rows = self.both.to_dicts()
        for idx, row in enumerate(rows):
            if row["bsize_msp"] is None or row["bst_msp"] is None:
                continue
            if logging.DEBUG >= logging.root.level:
                sys.stderr.write(
                    f"\r[DEBUG]: Added featrues to {(idx+1)/len(rows):.3%} of MSPs."
                )
            msp_stuff += get_msp_features(
                row, AT_genome, bin_width=bin_width, bin_num=bin_num
            )
        logging.debug("")
        logging.debug("Expanding bed12s into individual MSPs.")

        z = pd.DataFrame(msp_stuff)
        # for some reason the MSPs sometimes have negative lengths
        # z = z[(z["st"] < z["en"])]
        # Make more MSP featrues columns
        z["bin_m6a_frac"] = z.m6a_count / z.AT_count
        z["m6a_frac"] = z.msp_m6a / z.msp_AT
        z["msp_len"] = z.en - z.st
        m6a_fracs = pd.DataFrame(
            z["bin_m6a_frac"].tolist(),
            columns=[f"m6a_frac_{i}" for i in range(bin_num)],
        )
        m6a_counts = pd.DataFrame(
            z["m6a_count"].tolist(), columns=[f"m6a_count_{i}" for i in range(bin_num)],
        )
        AT_counts = pd.DataFrame(
            z["AT_count"].tolist(), columns=[f"AT_count_{i}" for i in range(bin_num)],
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
        moka_conf, models = ft.classify.make_accessibility_model(
            self.pin, train_fdr=train_fdr, test_fdr=test_fdr
        )
        with open(out_file, "wb") as f:
            pickle.dump(moka_conf.psms, f)
            pickle.dump(models, f)

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

        out.loc[out.qValue < 0.3, "color"] = "255,255,0"
        out.loc[out.qValue < 0.2, "color"] = "255,140,0"
        out.loc[out.qValue < 0.1, "color"] = "255,0,0"
        out.loc[out.qValue < 0.05, "color"] = "139,0,0"

        z = ft.utils.null_space_in_bed12(self.both)
        z["qValue"] = 1
        z["strand"] = "+"

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
            pd.concat([out[out_cols], z[out_cols]])
            .sort_values(["ct", "st"])
            .rename(columns={"ct": "#ct"})
        )
        final_out["score"] = (final_out.qValue * 100).astype(int)
        # final_out.to_csv(args.out, sep="\t", index=False)
        self.accessibility = final_out
