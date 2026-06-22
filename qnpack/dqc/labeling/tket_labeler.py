"""
labeling/tket_labeler.py
------------------------
label_tket_commands() — post-parse labeling pass for tket-style commands.

The tket frontend emits raw ejpp_start / ejpp_end / ejpp_start_link /
ejpp_end_link commands **without** any labels or entanglement_gen commands.
This module walks those raw commands and:

  1. Assigns monotonically-increasing ``start_label`` integers to every
     ejpp_start / ejpp_start_link pair.
  2. Assigns monotonically-increasing ``end_label`` integers to every
     ejpp_end / ejpp_end_link pair.
  3. Assigns a ``label`` string (e.g. ``"ejpp_start_0"``) and a ``clbit``
     scratch-bit index to each ejpp command.
  4. **Inserts** a pair of ``entanglement_gen`` commands (emitter + peer)
     immediately before each ejpp_start / ejpp_start_link pair, with an
     ``entanglement_label`` (``"ent_N"``) and ``target_start_label``.

Design
------
Each QPU's command list is rebuilt **independently** — no cross-QPU
insertion.  When the data QPU processes its ``ejpp_start`` it inserts the
``entanglement_gen emitter`` into its own list.  When the link QPU processes
its ``ejpp_start_link`` it inserts the ``entanglement_gen peer`` into its own
list.  Both sides share the same ``start_label`` (looked up from the global
map built in Pass 1) and the same ``entanglement_label`` (``"ent_N"``).

This avoids any assumption about which QPU ID is numerically smaller and
works correctly regardless of the data/link QPU ordering.
"""

import logging

log = logging.getLogger(__name__)


def label_tket_commands(qpu_commands: dict) -> dict:
    """Label tket-style per-QPU commands and insert entanglement_gen pairs.

    Parameters
    ----------
    qpu_commands : dict
        ``{qpu_id: [cmd_dict, ...]}`` as returned by the tket frontend's
        ``parse()`` method.  Commands contain ``ejpp_start`` / ``ejpp_end``
        ops but **no** labels or ``entanglement_gen`` entries.

    Returns
    -------
    dict
        A new ``{qpu_id: [cmd_dict, ...]}`` with labels assigned and
        ``entanglement_gen`` commands inserted.
    """
    # ── Pass 1: build global label maps ──────────────────────────────────────
    # Scan ejpp_start (data side) commands across all QPUs in sorted order to
    # assign monotonically-increasing start_label integers.
    # Key: (data_qpu_id, link_qpu_id, occurrence_index) → start_label int
    # Also build a parallel map for entanglement labels so both sides share
    # the same "ent_N" string.

    ejpp_start_events = []  # (data_qpu_id, cmd)
    for qpu_id in sorted(qpu_commands.keys()):
        for cmd in qpu_commands[qpu_id]:
            if cmd.get('op') == 'ejpp_start':
                ejpp_start_events.append((qpu_id, cmd))

    pair_occurrence: dict = {}
    start_label_map: dict = {}   # (data_qpu_id, link_qpu_id, occ) → int
    ent_label_map:   dict = {}   # (data_qpu_id, link_qpu_id, occ) → "ent_N"
    starting_process_counter = 0
    entanglement_counter = 0

    for data_qpu_id, cmd in ejpp_start_events:
        link_qpu_id = cmd.get('link_qpu_id')
        key = (data_qpu_id, link_qpu_id)
        occ = pair_occurrence.get(key, 0)
        pair_occurrence[key] = occ + 1
        start_label_map[(data_qpu_id, link_qpu_id, occ)] = starting_process_counter
        ent_label_map[(data_qpu_id, link_qpu_id, occ)]   = f"ent_{entanglement_counter}"
        starting_process_counter += 1
        entanglement_counter += 1

    # Similarly for ejpp_end (data side only)
    ejpp_end_events = []
    for qpu_id in sorted(qpu_commands.keys()):
        for cmd in qpu_commands[qpu_id]:
            if cmd.get('op') == 'ejpp_end':
                ejpp_end_events.append((qpu_id, cmd))

    pair_occ_end: dict = {}
    end_label_map: dict = {}   # (data_qpu_id, link_qpu_id, occ) → int
    ending_process_counter = 0

    for data_qpu_id, cmd in ejpp_end_events:
        link_qpu_id = cmd.get('link_qpu_id')
        key = (data_qpu_id, link_qpu_id)
        occ = pair_occ_end.get(key, 0)
        pair_occ_end[key] = occ + 1
        end_label_map[(data_qpu_id, link_qpu_id, occ)] = ending_process_counter
        ending_process_counter += 1

    # ── Pass 2: rebuild each QPU's command list independently ────────────────
    # Each QPU processes its own original command list.
    # - ejpp_start  (data side): insert entanglement_gen emitter + labeled ejpp_start
    # - ejpp_start_link (link side): insert entanglement_gen peer + labeled ejpp_start_link
    # - ejpp_end    (data side): insert labeled ejpp_end
    # - ejpp_end_link (link side): insert labeled ejpp_end_link
    # - everything else: pass through unchanged

    new_commands: dict = {qpu_id: [] for qpu_id in qpu_commands}

    for qpu_id in sorted(qpu_commands.keys()):
        # Per-QPU occurrence counters (reset for each QPU)
        pair_occ_start_data: dict = {}   # (data_qpu_id, link_qpu_id) → count  [data side]
        pair_occ_start_link: dict = {}   # (data_qpu_id, link_qpu_id) → count  [link side]
        pair_occ_end_data:   dict = {}   # (data_qpu_id, link_qpu_id) → count  [data side]
        pair_occ_end_link:   dict = {}   # (data_qpu_id, link_qpu_id) → count  [link side]

        for cmd in qpu_commands[qpu_id]:
            op = cmd.get('op')

            # ── ejpp_start (data side) ────────────────────────────────────
            if op == 'ejpp_start':
                data_qpu_id    = qpu_id
                link_qpu_id    = cmd['link_qpu_id']
                key            = (data_qpu_id, link_qpu_id)
                occ            = pair_occ_start_data.get(key, 0)
                pair_occ_start_data[key] = occ + 1

                start_label    = start_label_map[(data_qpu_id, link_qpu_id, occ)]
                ent_label      = ent_label_map[(data_qpu_id, link_qpu_id, occ)]

                data_local     = cmd.get('data_qubit', cmd.get('qubit'))
                l_local        = cmd.get('l_local', 0)
                link_qubit_idx = cmd.get('link_qubit_idx', 0)
                original_qubits = cmd.get('original_qubits', [])

                # entanglement_gen emitter on this (data) QPU
                new_commands[qpu_id].append({
                    "op":                 "entanglement_gen",
                    "role":               "emitter",
                    "params":             [],
                    "qubits":             [l_local],
                    "data_qubit":         data_local,
                    "l_local":            l_local,
                    "original_qubits":    original_qubits,
                    "is_remote":          True,
                    "peer_qpu_id":        link_qpu_id,
                    "peer_qubit":         link_qubit_idx,
                    "entanglement_label": ent_label,
                    "target_start_label": start_label,
                    "start_label":        None,
                    "end_label":          None,
                })

                # labeled ejpp_start on this (data) QPU
                new_commands[qpu_id].append({
                    **cmd,
                    "start_label": start_label,
                    "label":       f"ejpp_start_{start_label}",
                    "clbit":       start_label,
                })

            # ── ejpp_start_link (link side) ───────────────────────────────
            elif op == 'ejpp_start_link':
                data_qpu_id    = cmd.get('data_qpu_id')
                link_qpu_id    = qpu_id
                key            = (data_qpu_id, link_qpu_id)
                occ            = pair_occ_start_link.get(key, 0)
                pair_occ_start_link[key] = occ + 1

                start_label = start_label_map.get((data_qpu_id, link_qpu_id, occ))
                ent_label   = ent_label_map.get((data_qpu_id, link_qpu_id, occ))

                if start_label is None:
                    log.warning(
                        f"[tket_labeler] ejpp_start_link on QPU {qpu_id}: "
                        f"no start_label for (data={data_qpu_id}, link={link_qpu_id}, occ={occ})"
                    )
                    new_commands[qpu_id].append(cmd)
                    continue

                link_qubit_idx = cmd.get('link_qubit_idx', cmd.get('qubit', 0))
                l_local        = cmd.get('l_local', link_qubit_idx)
                original_qubits = cmd.get('original_qubits', [])

                # entanglement_gen peer on this (link) QPU
                new_commands[qpu_id].append({
                    "op":                 "entanglement_gen",
                    "role":               "peer",
                    "params":             [],
                    "qubits":             [link_qubit_idx],
                    "original_qubits":    original_qubits,
                    "is_remote":          True,
                    "peer_qpu_id":        data_qpu_id,
                    "peer_qubit":         l_local,
                    "entanglement_label": ent_label,
                    "target_start_label": start_label,
                    "start_label":        None,
                    "end_label":          None,
                })

                # labeled ejpp_start_link on this (link) QPU
                new_commands[qpu_id].append({
                    **cmd,
                    "start_label": start_label,
                    "label":       f"ejpp_start_link_{start_label}",
                    "clbit":       start_label,
                })

            # ── ejpp_end (data side) ──────────────────────────────────────
            elif op == 'ejpp_end':
                data_qpu_id = qpu_id
                link_qpu_id = cmd.get('link_qpu_id', cmd.get('peer_qpu_id'))
                key         = (data_qpu_id, link_qpu_id)
                occ         = pair_occ_end_data.get(key, 0)
                pair_occ_end_data[key] = occ + 1

                end_label = end_label_map.get((data_qpu_id, link_qpu_id, occ))
                if end_label is None:
                    log.warning(
                        f"[tket_labeler] ejpp_end on QPU {qpu_id}: "
                        f"no end_label for (data={data_qpu_id}, link={link_qpu_id}, occ={occ})"
                    )
                    new_commands[qpu_id].append(cmd)
                    continue

                new_commands[qpu_id].append({
                    **cmd,
                    "end_label": end_label,
                    "label":     f"ejpp_end_{end_label}",
                    "clbit":     end_label,
                })

            # ── ejpp_end_link (link side) ─────────────────────────────────
            elif op == 'ejpp_end_link':
                data_qpu_id = cmd.get('data_qpu_id')
                link_qpu_id = qpu_id
                key         = (data_qpu_id, link_qpu_id)
                occ         = pair_occ_end_link.get(key, 0)
                pair_occ_end_link[key] = occ + 1

                end_label = end_label_map.get((data_qpu_id, link_qpu_id, occ))
                if end_label is None:
                    log.warning(
                        f"[tket_labeler] ejpp_end_link on QPU {qpu_id}: "
                        f"no end_label for (data={data_qpu_id}, link={link_qpu_id}, occ={occ})"
                    )
                    new_commands[qpu_id].append(cmd)
                    continue

                new_commands[qpu_id].append({
                    **cmd,
                    "end_label": end_label,
                    "label":     f"ejpp_end_link_{end_label}",
                    "clbit":     end_label,
                })

            # ── All other commands — pass through unchanged ────────────────
            else:
                new_commands[qpu_id].append(cmd)

    return new_commands
