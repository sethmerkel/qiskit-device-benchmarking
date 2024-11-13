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
from qiskit.circuit.library import TwoLocal,ECRGate, CZGate
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

decomp = OneQubitEulerDecomposer(basis='ZSX')            
def cliff_angles(cliff):
    angles = decomp.angles(cliff.to_matrix())
    return [angles[2] ,angles[0]+np.pi ,angles[1]+np.pi ]

cliff_params = [cliff_angles(c) for c in single_cliffs]
            
pauli_preps = [Clifford([[0,1,0],[1,0,0]]),Clifford([[1,0,1],[1,1,0]]),Clifford([[1,0,0],[0,1,0]])]


def random_pauli(n,support=None):
    out=''
    if support is None:
        for i in range(n):
            out=np.random.choice(['I','X','Y','Z'])+out
        
    else:
        for s in support:
            if s:
                out=np.random.choice(['X','Y','Z'])+out
            else:
                out='I'+out
    return out


def bricklayer_circ(nqubits,depth,gate = 'cz',onlylayer=None):
    if gate == 'cz':
        gate = CZGate
    elif gate == 'ecr':
        gate = ECRGate
        
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
    def __init__(self, circuit,samples,input_support=None, output_support=None):
        self.base_circuit=circuit
        self.samples=samples
        self.input_support=input_support
        self.output_support=output_support 
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
        self.circuit = circ_to_layers(self.base_circuit)
        self.nqubits= len(self.base_circuit.qregs[0])
        #Cliffordize singles
        pm = PassManager([Cliffordize_pass()])
        self.circuit = pm.run(self.circuit)
        self.nlayers = int(str(self.circuit.parameters[-1]).split('_')[1])+1
        self.nlayers = np.max([int(str(p).split('_')[1])+1 for p in self.circuit.parameters])
            
        #sample random singles
        self.cliff_inds = [[[np.random.randint(24)for q in range(self.nqubits)]
                            for l in range(self.nlayers)]
                           for s in range(self.samples)] 

        self.bindings = BindingsArray(data={par: np.array([self._param_val(str(par),s) for s in range(self.samples)])
            for par in self.circuit.parameters
        })
        
        #construct inputs and outputs
        if self.output_support is None:
            input_Paulis = [random_pauli(self.nqubits,self.input_support) for i in range(self.samples)]
            output_Paulis=[]
            for idx in np.ndindex(*self.bindings.shape):
                cliff = Clifford(self.bindings.bind(self.circuit, loc=idx))
                pauli = Pauli(input_Paulis[idx[0]])
                output_Paulis.append(str(pauli.evolve(cliff,frame='s')))
                  
        else:
            output_Paulis = [random_pauli(self.nqubits,self.output_support) for i in range(self.samples)]
            input_Paulis=[]
            for idx in np.ndindex(*self.bindings.shape):
                cliff = Clifford(self.bindings.bind(self.circuit, loc=idx))
                pauli = Pauli(output_Paulis[idx[0]])
                input_Paulis.append(str(pauli.evolve(cliff.adjoint(),frame='s')))
            
        #update input bindings
        self.input_support_all=[]
        for s in range(self.samples):
            
            pauli = input_Paulis[s]
            if input_Paulis[s][0]=='-':
                pauli=input_Paulis[s][1:]
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
                postcliff = single_cliffs[self.cliff_inds[s][0][q]]
                newcliff = precliff.compose(postcliff)
                self.cliff_inds[s][0][q] = single_cliffs.index(newcliff)
            input_supp=[]
            for p in reversed(pauli):
                if p=='I':
                    input_supp.append(0)
                else:
                    input_supp.append(1)
            self.input_support_all.append(input_supp)
        #update output bindings and construct observables
        for s in range(self.samples):
            
            for q,p in enumerate(reversed(output_Paulis[s])):
                if p=='I' or p=='Z' or p =='-' :
                    continue
                elif p=='X':
                    postcliff = pauli_preps[0]
                elif p=='Y':
                    postcliff = pauli_preps[1]
                precliff = single_cliffs[self.cliff_inds[s][-1][q]]
                newcliff = precliff.compose(postcliff)
                self.cliff_inds[s][-1][q] = single_cliffs.index(newcliff)
            out=''
            for p in output_Paulis[s]:
                if p=='X' or p=='Y' or p=='Z':
                    out+='Z'
                else:
                    out+=p
            output_Paulis[s] = out
            
        self.bindings = BindingsArray(data={par: np.array([self._param_val(str(par),s) for s in range(self.samples)])
            for par in self.circuit.parameters
        })
        #construct observables
        
        self.observables = np.empty(self.samples, dtype=SparsePauliOp)
        for samp in range(self.samples):
            pauli_str = output_Paulis[samp]
            self.observables[samp] = SparsePauliOp(pauli_str)
            
        
        
        
        
        
        
def Cliffordize(target_circuit,samples,backend,initial_layout=None, loglayers=3,gate = 'cz',input_support=None, output_support=None ):
    nqubits= len(target_circuit.qregs[0])
    if input_support is None:
        input_support=[1]*nqubits
    if output_support is None:
        output_support=[1]*nqubits
        
        
    
    logref = bricklayer_circ(nqubits,loglayers,gate=gate)
    logref2 = copy.deepcopy(logref)
    append_layer_circs(logref2, logref)
    
    gate_proxy = copy.deepcopy(target_circuit)
    append_layer_circs(gate_proxy, logref)
    
    output_pubs = {'gate_proxy':DF_to_pub(DirectFidelity(gate_proxy,samples), backend, initial_layout),
                   'L':DF_to_pub(DirectFidelity(logref,samples), backend, initial_layout),
                   'LL':DF_to_pub(DirectFidelity(logref2,samples), backend, initial_layout),
                   'prep':DF_to_pub(DirectFidelity(logref,samples,input_support=input_support), backend, initial_layout),
                   'meas':DF_to_pub(DirectFidelity(logref,samples,output_support=output_support), backend, initial_layout)}
    return output_pubs

def binarydigits(x, bits):
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])

def SomeSpam(samples,backend,initial_layout, loglayers=3,gate = 'cz'):
    nqubits= len(initial_layout)
    logref = bricklayer_circ(nqubits,loglayers,gate=gate)
    logref2 = copy.deepcopy(logref)
    append_layer_circs(logref2, logref)
    output_pubs={}
    output_pubs['L_reference'] = DF_to_pub(DirectFidelity(logref,samples), backend, initial_layout)
    output_pubs['init']=DF_to_pub(DirectFidelity(logref,samples,input_support=[1]*nqubits), backend, initial_layout)  
    output_pubs['meas']=DF_to_pub(DirectFidelity(logref,samples,output_support=[1]*nqubits), backend, initial_layout) 

    return output_pubs

def AllSpam(samples,backend,initial_layout, loglayers=3,gate = 'cz'):
    nqubits= len(initial_layout)
    logref = bricklayer_circ(nqubits,loglayers,gate=gate)
    output_pubs={}
    output_pubs['L_reference'] = DF_to_pub(DirectFidelity(logref,samples), backend, initial_layout)
    meas_pubs = {}
    init_pubs = {}
    
    for i in range(1,2**nqubits):
        
        support = binarydigits(i,nqubits)
        meas_pubs[str(support)] = DF_to_pub(DirectFidelity(logref,samples,output_support=support), backend, initial_layout)
        init_pubs[str(support)] = DF_to_pub(DirectFidelity(logref,samples,input_support=support), backend, initial_layout)
    output_pubs['meas_pubs']=meas_pubs 
    output_pubs['init_pubs']=init_pubs 
    return output_pubs
    