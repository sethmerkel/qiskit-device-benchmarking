# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
best backend chain
"""
import rustworkx as rx
from qiskit.circuit import Gate

import numpy as np

def best_layer_chains(backend,num_qubits_in_chain,num_gates):
    for op in backend.target.operations:
        if isinstance(op, Gate) and  op.num_qubits==2:
            gate_2q=op.name
            break

    coupling_map = backend.target.build_coupling_map(gate_2q)
    G = coupling_map.graph
    
    
    def to_edges(path):
        edges = []
        prev_node = None
        for node in path:
            if prev_node is not None:
                if G.has_edge(prev_node, node):
                    edges.append((prev_node, node))
                else:
                    edges.append((node, prev_node))
            prev_node = node
        return edges


    def path_fidelity(path, nomeas=False):
        """Compute an estimate of the total fidelity of 2-qubit gates on a path.
        If `correct_by_duration` is true, each gate fidelity is worsen by
        scale = max_duration / duration, i.e. gate_fidelity^scale.
        If `readout_scale` > 0 is supplied, readout_fidelity^readout_scale
        for each qubit on the path is multiplied to the total fielity.
        The path is given in node indices form, e.g. [0, 1, 2].
        An external function `to_edges` is used to obtain edge list, e.g. [(0, 1), (1, 2)]."""
        path_edges = to_edges(path)
        max_duration = max(backend.target[gate_2q][qs].duration for qs in path_edges)

        def gate_fidelity(qpair):
            return max(0.25, 1 - (1.25 * backend.target[gate_2q][qpair].error))

        def readout_fidelity(qubit):
            return max(0.25, 1 - backend.target["measure"][(qubit,)].error)

        total_fidelity = np.prod([gate_fidelity(qs) for qs in path_edges])**(num_gates/len(path_edges))
        if not nomeas:
            total_fidelity *= np.prod([readout_fidelity(q) for q in path]) 
        return total_fidelity

    
    def flatten(paths, cutoff=None):  # cutoff not to make run time too large
        return [
            path
            for s, s_paths in paths.items()
            for t, st_paths in s_paths.items()
            for path in st_paths[:cutoff]
            if s < t
        ]
    
    
    paths = rx.all_pairs_all_simple_paths(
        G.to_undirected(multigraph=False),
        min_depth=num_qubits_in_chain,
        cutoff=num_qubits_in_chain,
    )
    paths = flatten(paths, cutoff=400)
    
    best_qubit_chain = max(paths, key=path_fidelity)
    print('meas')
    print(path_fidelity(best_qubit_chain))
    print('no meas')
    print(path_fidelity(best_qubit_chain ,nomeas=True))
    assert len(best_qubit_chain) == num_qubits_in_chain
   
    qubits = np.array(best_qubit_chain)
    all_pairs = to_edges(best_qubit_chain)
    two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]
    
    return qubits, two_disjoint_layers

