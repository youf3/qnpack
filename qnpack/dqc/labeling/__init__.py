"""
labeling/__init__.py
--------------------
Public API for the labeling package.

Single entry point: ``label_and_build_maps(qpu_commands)``

  1. Auto-detects whether the commands are tket-style (contain ``ejpp_start``
     ops) or QASM-style (contain ``entanglement_gen`` ops without
     ``ejpp_start``).
  2. Calls the appropriate labeler to assign all labels and (for tket) insert
     ``entanglement_gen`` command pairs.
  3. Calls ``build_process_maps()`` to build the coordinator maps needed by
     ``ControllerProtocol``.

Returns ``(labeled_qpu_commands, process_maps)`` where ``process_maps`` is::

    {
        'start_qpus':              {label: set(qpu_ids)},
        'end_qpus':                {label: set(qpu_ids)},
        'entanglement_gen_labels': set(labels),
    }
"""

from .base        import build_process_maps
from .tket_labeler import label_tket_commands
from .qasm_labeler import label_qasm_commands

__all__ = [
    "label_and_build_maps",
    "build_process_maps",
    "label_tket_commands",
    "label_qasm_commands",
]


def label_and_build_maps(qpu_commands: dict) -> tuple:
    """Label commands and build process maps.

    Auto-detects the circuit mode (tket vs QASM) from the command ops present
    in *qpu_commands*, applies the appropriate labeler, then builds the process
    maps required by ``ControllerProtocol``.

    Parameters
    ----------
    qpu_commands : dict
        ``{qpu_id: [cmd_dict, ...]}`` as returned by a frontend's ``parse()``.

    Returns
    -------
    tuple
        ``(labeled_qpu_commands, process_maps)``

        - ``labeled_qpu_commands`` – ``{qpu_id: [cmd_dict, ...]}`` with all
          labels assigned and (for tket) ``entanglement_gen`` pairs inserted.
        - ``process_maps`` – dict with keys ``start_qpus``, ``end_qpus``,
          ``entanglement_gen_labels``.
    """
    if _is_tket_mode(qpu_commands):
        labeled = label_tket_commands(qpu_commands)
    else:
        labeled = label_qasm_commands(qpu_commands)

    process_maps = build_process_maps(labeled)
    return labeled, process_maps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_tket_mode(qpu_commands: dict) -> bool:
    """Return True if any command has op ``ejpp_start`` (tket-style)."""
    for cmds in qpu_commands.values():
        for cmd in cmds:
            if cmd.get('op') in ('ejpp_start', 'ejpp_end',
                                 'ejpp_start_link', 'ejpp_end_link'):
                return True
    return False
