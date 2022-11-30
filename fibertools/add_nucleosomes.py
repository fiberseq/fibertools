#!/usr/bin/env python3
import pysam
import numpy as np
import logging
import pomegranate as pom
import array
import sys

D_TYPE = np.int64


def train_hmm(data, n_jobs=1):
    logging.info(
        f"Training HMM with {len(data)} training sequences and {n_jobs} threads"
    )
    model = pom.HiddenMarkovModel().from_samples(
        pom.DiscreteDistribution,
        n_components=2,
        X=data,
        verbose=False,
        max_iterations=250,
        n_init=10,
        n_jobs=n_jobs,
    )
    model.bake()
    return model


def assign_states(model):
    s0 = dict(model.states[0].distribution.parameters[0])
    s1 = dict(model.states[1].distribution.parameters[0])
    s0_m6a_probability, s1_m6a_probability = s0[1], s1[1]
    if s0_m6a_probability > s1_m6a_probability:
        actuated = 0
        nucleated = 1
    else:
        actuated = 1
        nucleated = 0

    return actuated, nucleated


def get_mods_from_rec(rec, mods=[("A", 0, "a"), ("T", 1, "a")], mask=True):
    if rec.modified_bases is None:
        return None, None, None
    seq = np.frombuffer(bytes(rec.query_sequence, "utf-8"), dtype="S1")
    positions = []
    for mod in mods:
        if mod in rec.modified_bases:
            pos = np.array(rec.modified_bases[mod], dtype=D_TYPE)[:, 0]
            positions.append(pos)
    if len(positions) < 1:
        return None, None, None
    methylated_positions = np.concatenate(positions, dtype=D_TYPE)
    methylated_positions.sort(kind="mergesort")

    AT_mask = (seq == b"A") | (seq == b"T")
    AT_positions = np.argwhere(AT_mask).transpose()[0]

    binary = np.zeros(shape=len(seq), dtype=np.uint8)
    binary[methylated_positions] = 1

    if mask:
        # TODO check for other mods
        return binary[AT_mask], AT_positions, methylated_positions
    return binary, AT_positions, methylated_positions


def meshMethods(
    simple_starts,
    simple_sizes,
    hmm_starts,
    hmm_sizes,
    methylated_positions,
    fiber_length,
    cutoff,
):

    ###FILTERS TO IMPLEMENT ########

    #### BASELINE is simple caller

    # HMM calls are used to modify aspects of the simple called nucleosomes

    # Essentially use the simple calls
    # and check for areas between the simple calls (actuated regions)
    # where the HMM model essentially finds nucleosomes

    # generate the end boundary
    simple_ends = np.add(simple_starts, simple_sizes, dtype=D_TYPE)
    hmm_ends = np.add(hmm_starts, hmm_sizes, dtype=D_TYPE)

    # stack with nucleosome boundary
    simple_stack = np.vstack([simple_starts, simple_ends])
    hmm_stack = np.vstack([hmm_starts, hmm_ends])

    #### processing ideas

    #### generate binary array of 0 , 1, 2
    #### 0 = msp
    #### 1 = simple nuc
    #### 2 = hmm nuc

    #### 1 takes precedence over 2 (simple more weight than hmm)
    #### to do so call 2 first then overlay 1

    #### perform RLE

    #### stretches of 1s = nucleosomes.
    #### stretches of 2s directly between ones == 0
    #### Pure stretches of 2s adjacent to 0s == nucleosome

    ### append nucleosome starts/sizes as arrays for consistency and use of np.concatenate

    chromatin_binary = np.zeros(shape=fiber_length).astype(D_TYPE)

    for column in range(hmm_stack.shape[1]):

        chromatin_binary[hmm_stack[0, column] : hmm_stack[1, column]] = 2

    for column in range(simple_stack.shape[1]):

        chromatin_binary[simple_stack[0, column] : simple_stack[1, column]] = 1

    run_length, run_index, run_label = rle(chromatin_binary)

    # collect all nucleosome starts and sizes
    nucleosome_starts = []
    nucleosome_sizes = []

    # STEP 1
    # grab simple calls

    nucleosome_starts.append(run_index[run_label == 1])
    nucleosome_sizes.append(run_length[run_label == 1])

    # print(chromatin_binary)

    # STEP 2
    hmm_idx = np.argwhere(run_label == 2).reshape(-1)

    number_of_runs = len(run_length)
    for idx in hmm_idx:

        if idx > 0 and idx < number_of_runs - 1:
            # check that it is not a terminal hmm nuc
            # we will call the ends via a simple caller
            # print(run_label[hmm_idx - 1])
            if run_label[idx - 1] == 0 and run_label[idx + 1] == 0:
                # check that hmm nuc call is flanked by actuated sequence

                hmm_nuc_start, hmm_nuc_size = run_index[idx], run_length[idx]
                hmm_nuc_end = hmm_nuc_start + hmm_nuc_size

                hmm_nuc_midpoint = int(hmm_nuc_start + int(np.floor(hmm_nuc_size / 2)))
                lower_mets = methylated_positions[
                    methylated_positions < hmm_nuc_midpoint
                ]
                greater_mets = methylated_positions[
                    methylated_positions > hmm_nuc_midpoint
                ]

                if len(lower_mets) == 0 or len(greater_mets) == 0:
                    continue

                closest_start_idx = np.argmin(np.abs(lower_mets - hmm_nuc_start))
                closest_end_idx = np.argmin(np.abs(greater_mets - hmm_nuc_end))

                nucleosome_starts.append([lower_mets[closest_start_idx] + 1])
                nucleosome_sizes.append(
                    [
                        greater_mets[closest_end_idx]
                        - (lower_mets[closest_start_idx] + 1)
                    ]
                )

    nucleosome_starts = np.concatenate(nucleosome_starts)
    nucleosome_sizes = np.concatenate(nucleosome_sizes)

    sort_order = np.argsort(nucleosome_starts).reshape(-1)

    nucleosome_starts = nucleosome_starts[sort_order]
    nucleosome_sizes = nucleosome_sizes[sort_order]

    min_nuc_size_mask = nucleosome_sizes >= cutoff

    return nucleosome_starts[min_nuc_size_mask], nucleosome_sizes[min_nuc_size_mask]
    # grab valid hmm calls and bookend to be min length
    # first minmum length
    # next check that it is flanked by 0s


def apply_hmm(bam, hmm, nuc_label, cutoff, out, min_dist=46):
    for rec in bam.fetch(until_eof=True):
        (
            binary,
            AT_positions,
            methylated_positions,
        ) = get_mods_from_rec(rec, mask=True)
        if binary is None:
            out.write(rec)
            continue
        # binary of m6A calls in AT space, and AT positions relative to 0-based fiber start

        simple_starts, simple_sizes, generated_terminal = simpleFind(
            methylated_positions, binary, cutoff
        )

        # generated terminal is a boolean indicating if we generated a custom
        # nucleosome until tht terminal end of the fiber

        state_path = hmm.predict(binary)
        # print(len(binary))

        lengths, starts, labels = rle(state_path)
        # starts indicate start in AT binary array
        # lengths indicates length in AT binary space
        # labels is the state label

        hmm_nucleosome_mask = labels == nuc_label

        hmm_nuc_ends = AT_positions[
            np.add(starts[hmm_nucleosome_mask], lengths[hmm_nucleosome_mask] - 1)
        ]
        hmm_nuc_starts = AT_positions[starts[hmm_nucleosome_mask]]

        hmm_nuc_sizes = hmm_nuc_ends - hmm_nuc_starts

        hmm_sizing_mask = hmm_nuc_sizes >= cutoff

        hmm_nuc_sizes = hmm_nuc_sizes[hmm_sizing_mask]
        hmm_nuc_starts = hmm_nuc_starts[hmm_sizing_mask]

        fiber_length = len(rec.query_sequence)

        all_starts, all_sizes = meshMethods(
            simple_starts,
            simple_sizes,
            hmm_nuc_starts,
            hmm_nuc_sizes,
            methylated_positions,
            fiber_length,
            cutoff,
        )

        output_starts = all_starts
        output_sizes = all_sizes

        # no nucleosomes found, continue
        if methylated_positions.shape[0] == 0 or output_sizes.shape[0] == 0:
            out.write(rec)
            continue

        # now need to bookend the fibers with the terminal nucleosomes
        # only need to bookend if hmm or simplecaller did not handle it
        # i.e. only need to book end front if there is not a 1 present in the nuc_starts
        # and only nee to book end end if there is not fib-length - 1 in
        # we dont need artificial ends anymore for nucleosomes
        # just last methylation to fiber length and 0 to first methylation as nucs

        front_terminal_nuc_start = 0
        front_terminal_nuc_end = methylated_positions[0]
        front_terminal_nuc_size = front_terminal_nuc_end - front_terminal_nuc_start

        if not generated_terminal:
            end_terminal_nuc_start = np.maximum(
                methylated_positions[-1], output_starts[-1] + output_sizes[-1] + 1
            )
            if end_terminal_nuc_start != methylated_positions[-1]:
                all_sizes[-1] = (
                    fiber_length - output_starts[-1]
                )  # resize the last nucleosome call to the end of fiber
                output_sizes[-1] = all_sizes[-1]
            else:
                end_terminal_nuc_size = fiber_length - end_terminal_nuc_start
                output_starts = np.append(output_starts, [end_terminal_nuc_start])
                output_sizes = np.append(output_sizes, [end_terminal_nuc_size])
                all_starts = output_starts
                all_sizes = output_sizes

        output_starts = np.concatenate(
            [[front_terminal_nuc_start], all_starts], dtype=D_TYPE
        )
        output_sizes = np.concatenate(
            [[front_terminal_nuc_size], all_sizes], dtype=D_TYPE
        )

        # check that the hmm is only making ranges that are possible.
        correct_sizes = output_starts[:-1] + output_sizes[:-1] <= output_starts[1:]
        if not np.all(correct_sizes):
            logging.warning(
                f"HMM made invalid ranges for {rec.query_name} skipping nucelosome calling for fiber"
            )
            out.write(rec)
            continue

        # make the acc arrays
        acc_starts = (output_starts + output_sizes)[:-1]
        acc_ends = output_starts[1:]
        acc_sizes = acc_ends - acc_starts

        # nucs always bookend the fiber, but now changing that here
        assert (
            output_starts[0] == 0
            and output_starts[-1] + output_sizes[-1] == fiber_length
        )
        output_starts = output_starts[1:-1]
        output_sizes = output_sizes[1:-1]

        # check that nucleosomes are not too close to the ends of the fiber, and have non-zero size
        cond = (
            (output_starts >= min_dist)
            & ((output_starts + output_sizes) <= (fiber_length - min_dist))
            & (output_sizes > 0)
        )
        output_starts_2 = output_starts[cond]
        output_sizes_2 = output_sizes[cond]

        # check that accessible elements are not too close to the ends of the fiber, and have non-zero size
        cond = (
            (acc_starts >= min_dist)
            & ((acc_starts + acc_sizes) <= (fiber_length - min_dist))
            & (acc_sizes > 0)
        )
        acc_starts_2 = acc_starts[cond]
        acc_sizes_2 = acc_sizes[cond]

        n_msp = acc_starts_2.shape[0]
        n_nuc = output_starts_2.shape[0]
        assert (
            abs(n_msp - n_nuc) < 2
        ), f"Number of nucleosomes must be within 1 of the number of MSP elements: MSP({n_msp}), Nuc({n_nuc})"

        if output_sizes.shape[0] > 0:
            rec.set_tag("ns", array.array("I", output_starts_2))
            rec.set_tag("nl", array.array("I", output_sizes_2))
        if acc_sizes.shape[0] > 0:
            rec.set_tag("as", array.array("I", acc_starts_2))
            rec.set_tag("al", array.array("I", acc_sizes_2))
        out.write(rec)


def simpleFind(methylated_positions, binary, cutoff):
    """Using an array of methylated positions
    calculate the spacing between each methylation.
    If the distance between two methylations > 85 then
    that is a nucleosome call. Otherwise continue."""

    dif = np.subtract(methylated_positions[1:], methylated_positions[:-1], dtype=D_TYPE)
    # difference between adjacent methylations

    index = np.argwhere(dif >= cutoff).reshape(-1)
    # indices where difference is equal or greater than 85bp

    simple_nuc_starts = methylated_positions[index] + 1
    # the nucleosome starts are one in front of the methylation
    # indicating the start of the nucleosome

    simple_nuc_sizes = dif[index] - 1
    # size needs to be adjusted to not include beginning methylation

    # now we need to handle the edge case where we have a single methylation within a nucleosome
    # run_lengths of 0s  >= 35 and <= 120
    # to make sure there is only a single methylation between the two:
    # length[short_idx]

    # length, loc, met_status = rle(binary)

    # short_idx = np.argwhere((length >= 35) & (length <= 120)).reshape(-1)

    # print(short_idx)

    # try simpler version with dif array

    short_idx = np.argwhere((dif >= 20) & (dif <= 100)).reshape(
        -1
    )  # this idx works for methylated positions as the start

    if len(short_idx) > 1:

        short_idx_consecutive_len, short_idx_consecutive_index, short_idx_diff = rle(
            np.diff(short_idx)
        )

        # need to perform RLE in the instance that there are consecutive stretches of short protected sequence
        # punctuated by erroneous methylation calls
        # short_idx_consecutive_len is the stretch where the difference between two short stretches is the same
        # short_idx_consecutive_len is where there is a stetch of idx differences that are equivalent
        # short_idx_diff is what is the difference in index between the two short regions ( we are interested in a dif of 1 )
        # all these above need to be used to index the short_idx array which will relate us back to the methylated positions

        # rle is performed on the diff. So the last methylation terminating
        # the last short stretch is equal to the start + length + 1 (python indexing) + 1 (terminal, not beginning nuc
        # gives us distance)
        # this simplifies to start + length + 2

        terminal_nuc_generated = False

        custom_nuc_starts = []
        custom_nuc_sizes = []

        consecutive_mask = (
            short_idx_diff == 1
        )  # mask for finding instance where the difference between one short region and the next is one

        all_included_methylations = []

        for s, e in zip(
            short_idx[short_idx_consecutive_index[consecutive_mask]],
            short_idx_consecutive_len[consecutive_mask],
        ):

            if s + e + 2 == len(methylated_positions):
                custom_nuc_start = methylated_positions[s] + 1
                # start + len + 1 (0 index of python) + 1 (need to capture terminal nuc)
                custom_nuc_size = len(binary) - custom_nuc_start

                all_included_methylations.append([custom_nuc_start])

                # terminal_nuc_generated = True

            else:

                custom_nuc_start = methylated_positions[s] + 1
                custom_nuc_end = methylated_positions[
                    s + e + 2
                ]  # start + len + 1 (0 index of python) + 1 (need to capture terminal nuc)
                custom_nuc_size = custom_nuc_end - custom_nuc_start

                all_included_methylations.append(
                    methylated_positions[s : s + e + 1] + 1
                )

            custom_nuc_starts.append(custom_nuc_start)
            custom_nuc_sizes.append(custom_nuc_size)

        # using the all_included_methylations array we can now filter out
        # our initial simple calls to remove any that overlap these custom
        # merged calls

        if len(all_included_methylations) > 0:

            all_included_methylations = np.concatenate(all_included_methylations)

            uniq_simple_mask = np.isin(
                simple_nuc_starts, all_included_methylations, invert=True
            )
            # idx NOT used in the custom calls

            simple_nuc_starts = simple_nuc_starts[uniq_simple_mask]
            simple_nuc_sizes = simple_nuc_sizes[uniq_simple_mask]

        all_simple_starts = np.concatenate(
            [simple_nuc_starts, custom_nuc_starts]
        ).astype(int)
        all_simple_sizes = np.concatenate([simple_nuc_sizes, custom_nuc_sizes]).astype(
            int
        )

        sort_order = np.argsort(all_simple_starts)

        return (
            all_simple_starts[sort_order],
            all_simple_sizes[sort_order],
            terminal_nuc_generated,
        )

    else:

        return simple_nuc_starts, simple_nuc_sizes, False


def rle(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)
    https://stackoverflow.com/questions/1066758/
    find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
    """
    ia = np.asarray(inarray, dtype=D_TYPE)  # force numpy
    n = len(ia)
    y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)  # must include last element posi
    z = np.diff(np.append(-1, i))  # run lengths
    p = np.cumsum(np.append(0, z))[:-1]  # positions
    return (z, p, ia[i])


def add_nucleosomes(args):
    bam = pysam.AlignmentFile(args.input, threads=args.threads, check_sq=False)
    if args.model is None:
        logging.info("Training HMM for nucleosome calling")
        training_set = []
        for idx, rec in enumerate(bam.fetch(until_eof=True)):
            mods, _AT_pos, _m6a_pos = get_mods_from_rec(rec, mask=True)
            if mods is None:
                continue
            training_set.append(mods)
            if idx >= args.num_train:
                break
        model = train_hmm(training_set, n_jobs=args.threads)
        json_model = model.to_json()

        out = sys.stdout if args.out == "-" else open(args.out, "w")
        with out as handle:
            handle.write(json_model)
    else:
        logging.info("Applying HMM for nucleosome calling")
        out = pysam.AlignmentFile(args.out, "wb", template=bam)
        hmm = pom.HiddenMarkovModel().from_json(args.model)
        _actuated_label, nucleated_label = assign_states(hmm)
        apply_hmm(bam, hmm, nucleated_label, args.cutoff, out, min_dist=args.min_dist)

    return 0
