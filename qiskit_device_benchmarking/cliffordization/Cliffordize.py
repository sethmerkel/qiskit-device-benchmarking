#TODO, take an arbitrary input circuit and insert the correct singles and barriers


#TODO, make sure support ordering is correct


'''
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import IGate, PauliTwoDesign, RZGate, SXGate
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Clifford, Pauli, PauliList
from qiskit.transpiler import PassManager, TransformationPass
'''
import numpy as np
import copy
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import TwoLocal,ECRGate, CZGate, SwapGate
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Clifford, Pauli, PauliList,SparsePauliOp
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library import IGate, PauliTwoDesign, RZGate, SXGate
from qiskit.circuit import Parameter
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit import transpile


single_cliffs = []
stab_base = [[1,0],[0,1],[1,1]]
ph_base=[[0,0],[1,0],[0,1],[1,1]]
for iz in range(3):
    for ix in range(2):
        for iph in range(4):
            cliff_table=np.zeros([2,3],dtype=int)
            cliff_table[0,0:2] = stab_base[iz]
            cliff_table[1,0:2] = stab_base[(iz+ix+1)%3]
            cliff_table[:,2] = ph_base[iph]
            single_cliffs.append(Clifford(cliff_table))

def cliff_prod(lhs,rhs):
    lcliff = single_cliffs[lhs]
    rcliff = single_cliffs[rhs]
    outcliff = lcliff.compose(rcliff)
    return single_cliffs.index(outcliff)


decomp = OneQubitEulerDecomposer(basis='ZSX')   

def cliff_angles(cliff):
    angles = decomp.angles(cliff.to_matrix())
    return [angles[2] ,angles[0]+np.pi ,angles[1]+np.pi ]

cliff_params = [cliff_angles(c) for c in single_cliffs]
            
pauli_preps = [single_cliffs.index(Clifford([[0,1,0],[1,0,0]])),
               single_cliffs.index(Clifford([[1,0,1],[1,1,0]])),
               single_cliffs.index(Clifford([[1,0,0],[0,1,0]]))]


def random_pauli(n):
    out=''
    for i in range(n):
        out=np.random.choice(['I','X','Y','Z'])+out
    phase = np.random.choice(['','-'])
    out = phase+out
    return out


def bricklayer_circ(nqubits,depth,gate = 'cz',onlylayer=None):
    if gate == 'cz':
        gate = CZGate
    elif gate == 'ecr':
        gate = ECRGate
    elif gate == 'swap':
        gate = SwapGate
        
    qubits = [a for a in range(nqubits)]
    layer_chains = [[(a,b) for a, b in zip(qubits[0::2],qubits[1::2])],
                    [(a,b) for a, b in zip(qubits[1::2],qubits[2::2])]]
    if onlylayer is not None:
        layer_chains = [layer_chains[onlylayer]]
        
    circ = TwoLocal(
        num_qubits=nqubits,
        rotation_blocks='x',
        entanglement_blocks=gate(),
        entanglement=layer_chains,
        reps=depth,
        insert_barriers=True,
    ).decompose()
    
    return circ

def append_layer_circs(front, back):
    #ignore first layer of singles
    append=False
    for gate,qregs,cregs in back:
        if gate.name == 'barrier':
            append =True
        if append:
            front.append(gate, qregs)

def rz_sx_gate_blocks(idl,idq):
    def pars(zi):
        return Parameter(f"l_{idl}_q_{idq}_z_{zi}")
    return [RZGate(pars(0)), SXGate(), RZGate(pars(1)), SXGate(), RZGate(pars(2))]            
            
def par_clifford_dag(
    gate_blocks,
    qubits = None,
    qreg = None,
    num_qubits = None,
):
    dag = DAGCircuit()
    if qubits:
        dag.add_qubits(qubits)
    elif qreg:
        dag.add_qreg(qreg)
    elif num_qubits:
        dag.add_qreg(QuantumRegister(num_qubits))
    else:
        dag.add_qreg(QuantumRegister(1))
    
    for wire in dag.wires:
        for op in gate_blocks:
            dag.apply_operation_back(op, qargs=(wire,))

    return dag            
            
class Cliffordize_pass(TransformationPass):
    def __init__(self):
        super().__init__()
    
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        
        single_layer=True
        idl=0
        idq=0
        for node in dag.op_nodes(include_directives=False):
            if node.op.num_qubits > 1:
                if single_layer:
                    single_layer=False
                    idl+=1
                    idq = 0    
                continue
            if not single_layer:
                single_layer=True
            gate_blocks = rz_sx_gate_blocks(idl,idq)
            dag.substitute_node_with_dag(
                node=node,
                wires=node.qargs,
                input_dag=par_clifford_dag(
                    gate_blocks=gate_blocks,
                    qubits=node.qargs,
                ),
            )
            idq+=1

        return dag
    
def circ_to_layers(circ):
    return circ

def DF_to_pub(df, backend, initial_layout):
    circuit = df.get_circuit()
    
    circuit = transpile(circuit,backend=backend,initial_layout=initial_layout)
    
    obs = ObservablesArray([ observable.apply_layout(circuit.layout) for observable in df.get_observables()])
    
    estimator_pub = EstimatorPub(
        circuit=circuit,
        observables=obs,
        parameter_values=df.get_bindings()
    )
    return estimator_pub
    


class DirectFidelity:
    def __init__(self, circuit, nstablizers, gate='cz'):
        self.base_circuit=circuit
        self.nstablizers=nstablizers
        self.gate=gate
        self.construct()
        
    def get_circuit(self):
        return self.circuit
    
    def get_observables(self):
        return self.observables
    
    def get_bindings(self):
        return self.bindings
    
    def circ_to_layers(self,circ):
        return circ
    
    def _param_val(self,par_str,samp):
        vec = par_str.split('_')
        nl=int(vec[1])
        nq=int(vec[3])
        nz=int(vec[5])
        cliff_ind = self.cliff_inds[samp][nl][nq]
        return cliff_params[cliff_ind][nz]
        
        
    
    def construct(self): 
        #layerize, TODO, make this
        circuit_bare = circ_to_layers(self.base_circuit)
        self.nqubits= len(self.base_circuit.qregs[0])
        
        #append two layers for the full circuit
        SPAM_circ = bricklayer_circ(self.nqubits,2,gate=self.gate)
        circuit_full = copy.deepcopy(circuit_bare)
        append_layer_circs(circuit_full, SPAM_circ)
        
        
        #Cliffordize singles
        pm = PassManager([Cliffordize_pass()])
        circuit_bare = pm.run(circuit_bare)
        circuit_full = pm.run(circuit_full)
        
        
        #random Clifford bindings for the bare circuit
        self.circuit = circuit_bare
        self.nlayers = np.max([int(str(p).split('_')[1])+1 for p in self.circuit.parameters])
        
        cliffs_temp=[[np.random.randint(24)for q in range(self.nqubits)]
                            for l in range(self.nlayers)]*self.nstablizers
        
        cliffs_bare = [[[cliffs_temp[l][q] if l < self.nlayers else 0 for q in range(self.nqubits)]
                            for l in range(self.nlayers)] 
                       for s in range(self.nstablizers)]
        
        self.cliff_inds = cliffs_bare
        self.bindings = BindingsArray(data={par: np.array([self._param_val(str(par),s) for s in range(self.nstablizers)])
            for par in self.circuit.parameters
        })
        
        #construct inputs
        input_Paulis = [random_pauli(self.nqubits) for i in range(self.nstablizers)]
        output_Paulis=[]
        for idx in np.ndindex(*self.bindings.shape):
            cliff = Clifford(self.bindings.bind(self.circuit, loc=idx))
            pauli = Pauli(input_Paulis[idx[0]])
            output_Paulis.append(str(pauli.evolve(cliff,frame='s')))
            
            
            
        #correct for the input pauli
        for s in range(self.nstablizers):
            pauli = input_Paulis[s]
            if input_Paulis[s][0]=='-':
                pauli=input_Paulis[s][1:]
                if output_Paulis[s][0]=='-':
                    output_Paulis[s]=output_Paulis[s][1:]
                else:
                    output_Paulis[s]='-'+output_Paulis[s]
            else: 
                pauli=input_Paulis[s]
                
            for q,p in enumerate(reversed(pauli)):
                if p=='I' or p=='Z' :
                    continue
                elif p=='X':
                    precliff = pauli_preps[0]
                elif p=='Y':
                    precliff = pauli_preps[1]
                self.cliff_inds[s][0][q] =   cliff_prod(precliff,self.cliff_inds[s][0][q])    
                
                
        #make the finals Z-measurements
        for s in range(self.nstablizers):
            pauli = output_Paulis[s]
            phase=''
            if pauli[0]=='-':
                phase='-'
                pauli = pauli[1:]
                
            for q,p in enumerate(reversed(pauli)):
                if p=='I' or p=='Z' :
                    continue
                elif p=='X':
                    self.cliff_inds[s][-1][q] = cliff_prod(self.cliff_inds[s][-1][q],pauli_preps[0]) 
                elif p=='Y':
                    self.cliff_inds[s][-1][q] = cliff_prod(self.cliff_inds[s][-1][q],pauli_preps[1]) 

            temp_pauli=phase
            for p in pauli:
                if p=='X' or p=='Y' or p=='Z':
                    temp_pauli+='Z'
                else:
                    temp_pauli+=p
            output_Paulis[s] = temp_pauli

        
        
        #append two layers
        cliffs_full = [[[cliffs_bare[s][l][q] if l < self.nlayers else 0 for q in range(self.nqubits)]
                            for l in range(self.nlayers+2)] 
                       for s in range(self.nstablizers)]
        
        
        #correct for the output pauli
        
        output_Paulis_full = []
        for cliffs,pauli in zip(cliffs_full,output_Paulis):
            output_Paulis_full.append(self.reduce_pauli(cliffs,pauli,self.gate))
          
            
        self.circuit = circuit_full
        self.cliff_inds = cliffs_full
        self.bindings = BindingsArray(data={par: np.array([self._param_val(str(par),s) for s in range(self.nstablizers)])
            for par in self.circuit.parameters
        })    
        #construct observables
        
        self.observables = np.empty(self.nstablizers, dtype=SparsePauliOp)
        for stab in range(self.nstablizers):
            #pauli_str = output_paulis_full[stab]
            pauli_str = output_Paulis_full[stab]
            self.observables[stab] = SparsePauliOp(pauli_str)
        
            
    def reduce_pauli(self,cliffs,pauli,gate):
        phase = ''
        if pauli[0]=='-':
            pauli = pauli[1:]
            phase='-'

       
        #ok, going to handwise hack some instructions in here
        
        ecrtocz =[[21,16],[2,12]]
        collapse_last=[[12,0],[12,0]]
        collapse_next=[[0,12],[0,12]]
        
        def apply_corrections(instr,layer,q0):
            
            cliffs[layer][q0] = cliff_prod(cliffs[layer][q0],instr[0][0])
            cliffs[layer][q0+1] = cliff_prod(cliffs[layer][q0+1],instr[0][1])

            cliffs[layer+1][q0] = cliff_prod(instr[1][0],cliffs[layer+1][q0])
            cliffs[layer+1][q0+1] = cliff_prod(instr[1][1],cliffs[layer+1][q0+1])

        
        pvec = [p for p in reversed(pauli)]

        qubits = [a for a in range(self.nqubits)]
        layer_chains = [[(a,b) for a, b in zip(qubits[0::2],qubits[1::2])],
                    [(a,b) for a, b in zip(qubits[1::2],qubits[2::2])]]
        
        collapse='next'
        for pair in layer_chains[0]:
            if pvec[pair[0]]=='Z' and pvec[pair[1]]=='Z':
                if collapse=='next':
                    apply_corrections(collapse_next,-3,pair[0])
                    pvec[pair[0]]='I'
                    collapse='last'
                else:
                    apply_corrections(collapse_last,-3,pair[0])
                    pvec[pair[1]]='I'
                    collapse='next'
            else:
                collapse='next'
                
            
        
        for pair in layer_chains[1]:
            if pvec[pair[0]]=='Z' and pvec[pair[1]]=='Z':
                apply_corrections(collapse_next,-2,pair[0])
                pvec[pair[0]]='I'
        if gate=='ecr':            
            for pair in layer_chains[0]:
                apply_corrections(ecrtocz,-3,pair[0])
            for pair in layer_chains[1]:
                apply_corrections(ecrtocz,-2,pair[0])
            
        pout = phase
        for p in reversed(pvec):
            pout = pout+p
        return pout
            
        '''
        l=0
        while l < len(layer_chains[0]):
            pair = layer_chains[0][l]
            if pvec[pair[0]]=='I' or pvec[pair[1]]=='I':
                apply_corrections(neutral,-3,pair[0])
                l+=1
                continue
            else:
                apply_corrections(collapse_up,-3,pair[0])
                pvec[pair[0]]='I'
                l+=1
                if not l<len(layer_chains[0]):
                    break
                pair = layer_chains[0][l]
                if pvec[pair[0]]=='I' or pvec[pair[1]]=='I':
                    apply_corrections(neutral,-3,pair[0])
                    l+=1
                    continue
                else:
                    apply_corrections(collapse_down,-3,pair[0])
                    pvec[pair[1]]='I'
                    l+=1
                    continue
                    
        l=0
        while l < len(layer_chains[1]):
            pair = layer_chains[1][l]
            if pvec[pair[0]]=='I' or pvec[pair[1]]=='I':
                apply_corrections(neutral,-2,pair[0])
                l+=1
                continue
            else:
                apply_corrections(collapse_up,-2,pair[0])
                pvec[pair[0]]='I'
                l+=1
                continue
        '''
        
        
    
    '''def reduce_pauli(self,cliffs,pauli,gate):
        prepend = ''
        if pauli[0]=='-':
            pauli = pauli[1:]
            prepend='-'

       
        #ok, going to handwise hack some instructions in here
        if gate == 'cz':
            collapse_down=[[12,0],[12,0]]
            collapse_up=[[0,12],[0,12]]
            neutral =[[0,0],[0,0]]
        elif gate == 'ecr':
            collapse_down=[[10,16],[14,12]]               
            collapse_up=[[21,4],[2,0]]
            neutral =[[21,16],[2,12]]

        def apply_corrections(instr,layer,q0):

            cliffs[layer][q0] = cliff_prod(cliffs[layer][q0],instr[0][0])
            cliffs[layer][q0+1] = cliff_prod(cliffs[layer][q0+1],instr[0][1])

            cliffs[layer+1][q0] = cliff_prod(instr[1][0],cliffs[layer+1][q0])
            cliffs[layer+1][q0+1] = cliff_prod(instr[1][1],cliffs[layer+1][q0+1])

        
        pvec = [p for p in reversed(output_pauli)]
        
        qubits = [a for a in range(self.nqubits)]
        final_pairs = [(a,b) for a, b in zip(qubits[1::2],qubits[2::2])]
        skip=-1

        for i,fp in enumerate(final_pairs):
            
            if i==skip:
                apply_corrections(neutral,-2,fp[0])
                continue
            if pvec[fp[0]]=='I':
                apply_corrections(neutral,-3,fp[0]-1)
                apply_corrections(neutral,-2,fp[0])
                continue

            if pvec[fp[0]-1]=='Z':
                apply_corrections(collapse_up,-3,fp[0]-1)
                pvec[fp[0]-1]='I'

            if pvec[fp[1]]=='I':
                apply_corrections(neutral,-2,fp[0])
                continue
            skip = i+1
            apply_corrections(collapse_up,-2,fp[0])
            pvec[fp[0]]='I'
            if fp[1]+1 < self.nqubits:
                if pvec[fp[1]+1]=='Z':
                    apply_corrections(collapse_down,-3,fp[1])
                    pvec[fp[1]+1]='I'
                else:
                    apply_corrections(neutral,-3,fp[1])

        pout = prepend
        for p in reversed(pvec):
            pout = pout+p
        return pout
        '''

    
                    

def Cliffordize(target_circuit,ncliffordizations,nstablizers,backend,initial_layout=None, gate = 'cz'):
    nqubits= len(target_circuit.qregs[0])
    
    ref_circ = bricklayer_circ(nqubits,0,gate=gate)
    gate_proxy = copy.deepcopy(target_circuit)
    
    output_pubs = {'ref_circ':DF_to_pub(DirectFidelity(ref_circ,nstablizers,gate=gate), backend, initial_layout)}
    for i in range(ncliffordizations):
        output_pubs[str(i)] = DF_to_pub(DirectFidelity(gate_proxy,nstablizers,gate=gate), backend, initial_layout)
        
    return output_pubs


    
        
        
        
        
        
        
        
        
        
        