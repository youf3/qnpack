"""
labeling/qasm_labeler.py
------------------------
label_qasm_commands() — post-parse labeling pass for QASM-style commands.

The QASM3 frontend is a pure parser: it emits ``entanglement_gen`` (emitter +
peer) command pairs and ``if_gate`` commands but performs no labeling.  This
module performs two labeling passes so that the frontend itself stays clean:

Pass 1 — entanglement labels
  - Scans all QPU command lists for ``entanglement_gen`` commands.
  - Matches emitter/peer pairs by their ``peer_qpu_id`` cross-reference.
  - Assigns a monotonically-increasing integer ``entanglement_label`` to each
    matched pair (both emitter and peer receive the same label value).

Pass 2 — msg_exchange labels (cross-QPU classical bit exchange)
  - Scans for ``measure`` commands that write to a classical bit owned by a
    different QPU (i.e. the clbit name encodes a QPU id that differs from the
    QPU executing the measure).
  - Scans for ``if_gate`` commands whose clbit is owned by a different QPU.
  - Pairs each such measure with the corresponding if_gate and replaces them
    with ``msg_sender`` / ``msg_receiver`` command pairs carrying an
    ``exchange_label`` string.

The QASM frontend does NOT emit ejpp_start / ejpp_end commands, so no
``start_label`` / ``end_label`` / ``label`` / ``clbit`` assignment is needed.
"""

import logging
from collections import defaultdict

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_qasm_commands(qpu_commands: dict) -> dict:
    """Label QASM-style commands: entanglement pairs + msg_exchange pairs.

    Parameters
    ----------
    qpu_commands : dict
        ``{qpu_id: [cmd_dict, ...]}`` as returned by the QASM3 frontend's
        ``parse()`` method.  Commands are already sorted by ``global_idx``.

    Returns
    -------
    dict
        A new ``{qpu_id: [cmd_dict, ...]}`` with:
        - ``entanglement_label`` assigned to all ``entanglement_gen`` pairs.
        - ``if_gate`` commands that reference a cross-QPU clbit replaced by
          ``msg_sender`` / ``msg_receiver`` pairs with an ``exchange_label``.
    """
    # Build a fresh copy so we don't mutate the input
    new_commands: dict = {
        qpu_id: [dict(cmd) for cmd in cmds]
        for qpu_id, cmds in qpu_commands.items()
    }

    # Pass 1: assign entanglement_label to entanglement_gen pairs
    _label_entanglement_pairs(new_commands)

    # Pass 2: create msg_sender / msg_receiver pairs from cross-QPU if_gates
    _label_msg_exchange(new_commands)

    return new_commands


# ---------------------------------------------------------------------------
# Pass 1: entanglement labels
# ---------------------------------------------------------------------------

def _label_entanglement_pairs(new_commands: dict) -> None:
    """Assign ``entanglement_label`` to emitter/peer entanglement_gen pairs.

    Mutates *new_commands* in place.
    """
    entanglement_counter = 0

    for qpu_id in sorted(new_commands.keys()):
        for cmd in new_commands[qpu_id]:
            if cmd.get('op') == 'entanglement_gen' and cmd.get('role') == 'emitter':
                label = entanglement_counter
                entanglement_counter += 1

                cmd['entanglement_label'] = label

                peer_qpu_id   = cmd.get('peer_qpu_id')
                emitter_qubit = cmd.get('qubit', cmd.get('qubits', [None])[0])

                if peer_qpu_id is not None and peer_qpu_id in new_commands:
                    _label_matching_peer(
                        new_commands[peer_qpu_id],
                        qpu_id,
                        emitter_qubit,
                        label,
                    )


def _label_matching_peer(
    peer_cmd_list: list,
    emitter_qpu_id: int,
    emitter_qubit: int,
    label: int,
) -> None:
    """Find the peer entanglement_gen command and assign *label* to it.

    Matching criterion: ``op == 'entanglement_gen'``, ``role == 'peer'``,
    ``peer_qpu_id == emitter_qpu_id``, and ``peer_qubit == emitter_qubit``
    (the peer's ``peer_qubit`` is the emitter's local qubit index).
    """
    for cmd in peer_cmd_list:
        if (
            cmd.get('op') == 'entanglement_gen'
            and cmd.get('role') == 'peer'
            and cmd.get('peer_qpu_id') == emitter_qpu_id
            and cmd.get('peer_qubit') == emitter_qubit
            and cmd.get('entanglement_label') is None
        ):
            cmd['entanglement_label'] = label
            return

    # Fallback: match by peer_qpu_id alone (first unlabeled peer)
    for cmd in peer_cmd_list:
        if (
            cmd.get('op') == 'entanglement_gen'
            and cmd.get('role') == 'peer'
            and cmd.get('peer_qpu_id') == emitter_qpu_id
            and cmd.get('entanglement_label') is None
        ):
            cmd['entanglement_label'] = label
            return

    log.warning(
        f"[qasm_labeler] Could not find peer entanglement_gen for "
        f"emitter on QPU_{emitter_qpu_id} qubit {emitter_qubit}"
    )


# ---------------------------------------------------------------------------
# Pass 2: msg_exchange labels
# ---------------------------------------------------------------------------

def _clbit_owner_qpu(clbit_name: str):
    """Extract the QPU id encoded in a clbit name like ``_clbit_comm_qubit2_1``.

    Returns the integer QPU id, or ``None`` if the name does not match the
    expected pattern ``_clbit_comm_qubit{N}_{M}``.
    """
    import re
    m = re.match(r'_clbit_comm_qubit(\d+)_', clbit_name or '')
    if m:
        return int(m.group(1))
    return None


def _label_msg_exchange(new_commands: dict) -> None:
    """Replace cross-QPU ``if_gate`` commands with msg_sender/msg_receiver pairs.

    Mutates *new_commands* in place.  The algorithm mirrors the post-processing
    block that was previously in ``qasm3_frontend._parse_qasm3_to_commands``.
    """
    # Index: clbit_name -> [(qpu_id, cmd_index), ...] for measure commands
    clbit_measure_lists: dict = defaultdict(list)
    for qpu_id, cmds in new_commands.items():
        for idx, cmd in enumerate(cmds):
            if cmd.get('op') == 'measure' and cmd.get('clbit'):
                clbit_measure_lists[cmd['clbit']].append((qpu_id, idx))

    # Index: clbit_name -> [(qpu_id, cmd_index), ...] for cross-QPU if_gate commands
    clbit_ifgate_lists: dict = defaultdict(list)
    for qpu_id, cmds in new_commands.items():
        for idx, cmd in enumerate(cmds):
            if cmd.get('op') == 'if_gate':
                clbit_name = cmd.get('clbit')
                owner_qpu  = _clbit_owner_qpu(clbit_name)
                if owner_qpu is not None and owner_qpu != qpu_id:
                    clbit_ifgate_lists[clbit_name].append((qpu_id, idx))

    total_cross_qpu = sum(len(v) for v in clbit_ifgate_lists.values())
    if total_cross_qpu == 0:
        return

    log.debug(f"[qasm_labeler] Found {total_cross_qpu} cross-QPU if_gate commands")

    exchange_label_counter = 0

    for clbit_name in sorted(
        set(clbit_measure_lists.keys()) & set(clbit_ifgate_lists.keys())
    ):
        measures = sorted(
            clbit_measure_lists[clbit_name],
            key=lambda x: new_commands[x[0]][x[1]]['global_idx'],
        )
        if_gates = sorted(
            clbit_ifgate_lists[clbit_name],
            key=lambda x: new_commands[x[0]][x[1]]['global_idx'],
        )

        n_pairs = min(len(measures), len(if_gates))
        if n_pairs < len(if_gates):
            log.warning(
                f"[qasm_labeler] clbit {clbit_name}: {len(if_gates)} cross-QPU "
                f"if_gates but only {len(measures)} measures — "
                f"{len(if_gates) - n_pairs} unmatched"
            )

        for k in range(n_pairs):
            meas_qpu, meas_idx = measures[k]
            if_qpu,  if_idx   = if_gates[k]

            exchange_label = f"msg_exchange_{exchange_label_counter}"
            exchange_label_counter += 1

            if_cmd   = new_commands[if_qpu][if_idx]
            meas_cmd = new_commands[meas_qpu][meas_idx]

            if_global_idx = if_cmd.get('global_idx', 0)

            # Append msg_sender to the measuring QPU's command list.
            # global_idx matches the if_gate position so the sender fires at
            # the correct circuit position (after all gates between measure and
            # if_gate on the sender's QPU have executed).
            new_commands[meas_qpu].append({
                'op':              'msg_sender',
                'label':           exchange_label,
                'msg_type':        'msg_exchange',
                'clbit':           meas_cmd['clbit'],
                'qubit':           meas_cmd.get('qubit'),
                'qubits':          meas_cmd.get('qubits', []),
                'original_qubits': meas_cmd.get('original_qubits', []),
                'params':          [],
                'peer_qpu_id':     if_qpu,
                'is_remote':       True,
                'final_key':       None,
                'start_label':     None,
                'end_label':       exchange_label,
                'if_gate':         None,
                'global_idx':      if_global_idx,
            })

            # Replace the if_gate with msg_receiver in place.
            new_commands[if_qpu][if_idx] = {
                'op':              'msg_receiver',
                'label':           exchange_label,
                'msg_type':        'msg_exchange',
                'clbit':           if_cmd.get('clbit'),
                'qubit':           if_cmd.get('qubit'),
                'qubits':          if_cmd.get('qubits', []),
                'original_qubits': if_cmd.get('original_qubits', []),
                'params':          [],
                'peer_qpu_id':     meas_qpu,
                'is_remote':       True,
                'final_key':       None,
                'start_label':     None,
                'end_label':       exchange_label,
                'if_gate': {
                    'gate':   if_cmd.get('gate'),
                    'qubit':  if_cmd.get('qubit'),
                    'qubits': if_cmd.get('qubits', []),
                    'params': if_cmd.get('params', []),
                    'clbit':  if_cmd.get('clbit'),
                } if if_cmd.get('gate') else None,
                'global_idx': if_global_idx,
            }

            log.debug(
                f"[qasm_labeler] Created msg_sender/msg_receiver pair: "
                f"sender=QPU_{meas_qpu}(measure {clbit_name}[{k}]), "
                f"receiver=QPU_{if_qpu}(if_gate {clbit_name}[{k}]), "
                f"label={exchange_label}"
            )

    # Re-sort each QPU's command list after appending msg_sender commands
    for qpu_id in new_commands:
        new_commands[qpu_id].sort(key=lambda x: x.get('global_idx', 0))
