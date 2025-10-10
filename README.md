# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-09 17:00:26.154500 PST.

### Artificial Intelligence

### 1. [WebDART: Dynamic Decomposition and Re-planning for Complex Web Tasks](http://arxiv.org/pdf/2510.06587v1)

Authors: Jingbo Yang, Bairu Hou, Wei Wei, Shiyu Chang, Yujia Bao

Large language model (LLM) agents are becoming competent at straightforward
web tasks, such as opening an item page or submitting a form, but still
struggle with objectives that require long horizon navigation, large scale
information extraction, and reasoning under constraints. We present WebDART, a
general framework that enables a single LLM to handle such complex chores.
WebDART (i) dynamically decomposes each objective into three focused subtasks:
navigation, information extraction, and execution, so the model concentrates on
one skill at a time, and (ii) continuously replans the decomposition as new
webpages are revealed, taking advantage of newly discovered filters or
shortcuts and avoiding redundant exploration. Evaluated on WebChoreArena,
WebDART lifts success rates by up to 13.7 percentage points over previous SOTA
agents, while matching their performance on the easier WebArena suite and
completing tasks with up to 14.7 fewer navigation steps.

### 2. [Fine-Grained Emotion Recognition via In-Context Learning](http://arxiv.org/pdf/2510.06600v1)

Authors: Zhaochun Ren, Zhou Yang, Chenglong Ye, Haizhou Sun, Chao Chen, Xiaofei Zhu, Xiangwen Liao

Fine-grained emotion recognition aims to identify the emotional type in
queries through reasoning and decision-making processes, playing a crucial role
in various systems. Recent methods use In-Context Learning (ICL), enhancing the
representation of queries in the reasoning process through semantically similar
examples, while further improving emotion recognition by explaining the
reasoning mechanisms. However, these methods enhance the reasoning process but
overlook the decision-making process. This paper investigates decision-making
in fine-grained emotion recognition through prototype theory. We show that ICL
relies on similarity matching between query representations and emotional
prototypes within the model, where emotion-accurate representations are
critical. However, semantically similar examples often introduce emotional
discrepancies, hindering accurate representations and causing errors. To
address this, we propose Emotion In-Context Learning (EICL), which introduces
emotionally similar examples and uses a dynamic soft-label strategy to improve
query representations in the emotion reasoning process. A two-stage exclusion
strategy is then employed to assess similarity from multiple angles, further
optimizing the decision-making process. Extensive experiments show that EICL
significantly outperforms ICL on multiple datasets.

### 3. [Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support](http://arxiv.org/pdf/2510.06674v1)

Authors: Cen, Zhao, Tiantian Zhang, Hanchen Su, Yufeng, Zhang, Shaowei Su, Mingzhi Xu, Yu, Liu, Wei Han, Jeremy Werner, Claire Na Cheng, Yashar Mehdad

We introduce an Agent-in-the-Loop (AITL) framework that implements a
continuous data flywheel for iteratively improving an LLM-based customer
support system. Unlike standard offline approaches that rely on batch
annotations, AITL integrates four key types of annotations directly into live
customer operations: (1) pairwise response preferences, (2) agent adoption and
rationales, (3) knowledge relevance checks, and (4) identification of missing
knowledge. These feedback signals seamlessly feed back into models' updates,
reducing retraining cycles from months to weeks. Our production pilot involving
US-based customer support agents demonstrated significant improvements in
retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality
(+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore
the effectiveness of embedding human feedback loops directly into operational
workflows to continuously refine LLM-based customer support system.

### 4. [Verifying Memoryless Sequential Decision-making of Large Language Models](http://arxiv.org/pdf/2510.06756v1)

Authors: Dennis Gross, Helge Spieker, Arnaud Gotlieb

We introduce a tool for rigorous and automated verification of large language
model (LLM)- based policies in memoryless sequential decision-making tasks.
Given a Markov decision process (MDP) representing the sequential
decision-making task, an LLM policy, and a safety requirement expressed as a
PCTL formula, our approach incrementally constructs only the reachable portion
of the MDP guided by the LLM's chosen actions. Each state is encoded as a
natural language prompt, the LLM's response is parsed into an action, and
reachable successor states by the policy are expanded. The resulting formal
model is checked with Storm to determine whether the policy satisfies the
specified safety property. In experiments on standard grid world benchmarks, we
show that open source LLMs accessed via Ollama can be verified when
deterministically seeded, but generally underperform deep reinforcement
learning baselines. Our tool natively integrates with Ollama and supports
PRISM-specified tasks, enabling continuous benchmarking in user-specified
sequential decision-making tasks and laying a practical foundation for formally
verifying increasingly capable LLMs.

### 5. [Autoformalizer with Tool Feedback](http://arxiv.org/pdf/2510.06857v1)

Authors: Qi Guo, Jianing Wang, Jianfei Zhang, Deyang Kong, Xiangzhou Huang, Xiangyu Xi, Wei Wang, Jingang Wang, Xunliang Cai, Shikun Zhang, Wei Ye

Autoformalization addresses the scarcity of data for Automated Theorem
Proving (ATP) by translating mathematical problems from natural language into
formal statements. Efforts in recent work shift from directly prompting large
language models to training an end-to-end formalizer model from scratch,
achieving remarkable advancements. However, existing formalizer still struggles
to consistently generate valid statements that meet syntactic validity and
semantic consistency. To address this issue, we propose the Autoformalizer with
Tool Feedback (ATF), a novel approach that incorporates syntactic and
consistency information as tools into the formalization process. By integrating
Lean 4 compilers for syntax corrections and employing a multi-LLMs-as-judge
approach for consistency validation, the model is able to adaptively refine
generated statements according to the tool feedback, enhancing both syntactic
validity and semantic consistency. The training of ATF involves a cold-start
phase on synthetic tool-calling data, an expert iteration phase to improve
formalization capabilities, and Direct Preference Optimization to alleviate
ineffective revisions. Experimental results show that ATF markedly outperforms
a range of baseline formalizer models, with its superior performance further
validated by human evaluations. Subsequent analysis reveals that ATF
demonstrates excellent inference scaling properties. Moreover, we open-source
Numina-ATF, a dataset containing 750K synthetic formal statements to facilitate
advancements in autoformalization and ATP research.

### 6. [TGPR: Tree-Guided Policy Refinement for Robust Self-Debugging of LLMs](http://arxiv.org/pdf/2510.06878v1)

Authors: Daria Ozerova, Ekaterina Trofimova

Iterative refinement has been a promising paradigm to enable large language
models (LLMs) to resolve difficult reasoning and problem-solving tasks. One of
the key challenges, however, is how to effectively search through the enormous
search space of possible refinements. Existing methods typically fall back on
predefined heuristics, which are troubled by the exploration-exploitation
dilemma and cannot adapt based on past refinement outcomes. We introduce
Tree-Guided Policy Refinement (TGPR), a novel framework that combines GRPO with
a Thompson-Sampling-based tree search. TGPR explores both failed and successful
refinement paths actively, with denser training trajectories and more adaptive
policies. On HumanEval, MBPP, and APPS benchmarks, our method achieves up to
+4.2 percentage points absolute improvement in pass@1 (on MBPP) and up to
+12.51 percentage points absolute improvement in pass@10 (on APPS) compared to
a competitive GRPO baseline. Apart from debugging code, TGPR focuses on a
principled approach to combining learned policies with structured search
methods, offering a general framework for enhancing iterative refinement and
stateful reasoning in LLMs.

### 7. [LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN](http://arxiv.org/pdf/2510.06911v1)

Authors: Hacane Hechehouche, Andre Antakli, Matthias Klusch

There are many established semantic Web standards for implementing
multi-agent driven applications. The AJAN framework allows to engineer
multi-agent systems based on these standards. In particular, agent knowledge is
represented in RDF/RDFS and OWL, while agent behavior models are defined with
Behavior Trees and SPARQL to access and manipulate this knowledge. However, the
appropriate definition of RDF/RDFS and SPARQL-based agent behaviors still
remains a major hurdle not only for agent modelers in practice. For example,
dealing with URIs is very error-prone regarding typos and dealing with complex
SPARQL queries in large-scale environments requires a high learning curve. In
this paper, we present an integrated development environment to overcome such
hurdles of modeling AJAN agents and at the same time to extend the user
community for AJAN by the possibility to leverage Large Language Models for
agent engineering.

### 8. [Tool-Augmented Policy Optimization: Synergizing Reasoning and Adaptive Tool Use with Reinforcement Learning](http://arxiv.org/pdf/2510.07038v1)

Authors: Wenxun Wu, Yuanyang Li, Guhan Chen, Linyue Wang, Hongyang Chen

Recent advances in large language models (LLMs) have popularized test-time
scaling, where models generate additional reasoning tokens before producing
final answers. These approaches have demonstrated significant performance
improvements on benchmarks involving mathematical reasoning. However, language
models relying solely on direct inference still struggle with tasks demanding
up-to-date knowledge or computational tools such as calculators and code
interpreters for complex arithmetic operations. To overcome these limitations,
we propose Tool-Augmented Policy Optimization (TAPO), a novel reinforcement
learning framework that systematically integrates multi-hop reasoning with
adaptive tool-calling capabilities. Our approach employs a modified version of
Dynamic Sampling Policy Optimization (DAPO), a recently developed RL paradigm,
which we adapt specifically for tool invocation scenarios, enabling models to
dynamically interleave complex reasoning with on-demand tool usage (including
search APIs and Python interpreters).
  To support this research, we introduce two new datasets: TAPO-easy-60K and
TAPO-hard-18K, specifically designed to train and evaluate both fact-based
reasoning and mathematical calculation capabilities. Our experiments on
Qwen2.5-3B and Qwen2.5-7B models demonstrate the effectiveness of our approach,
with both models achieving state-of-the-art performance on tasks requiring
external knowledge and mathematical computation among methods with comparable
parameters. Notably, TAPO achieves more efficient tool utilization than
baseline methods while preventing excessive calls caused by reward hacking.
These results highlight the significant potential of combining advanced
reasoning with tool usage to enhance model performance in knowledge-intensive
and computationally demanding tasks.

### 9. [Prompt Optimization Across Multiple Agents for Representing Diverse Human Populations](http://arxiv.org/pdf/2510.07064v1)

Authors: Manh Hung Nguyen, Sebastian Tschiatschek, Adish Singla

The difficulty and expense of obtaining large-scale human responses make
Large Language Models (LLMs) an attractive alternative and a promising proxy
for human behavior. However, prior work shows that LLMs often produce
homogeneous outputs that fail to capture the rich diversity of human
perspectives and behaviors. Thus, rather than trying to capture this diversity
with a single LLM agent, we propose a novel framework to construct a set of
agents that collectively capture the diversity of a given human population.
Each agent is an LLM whose behavior is steered by conditioning on a small set
of human demonstrations (task-response pairs) through in-context learning. The
central challenge is therefore to select a representative set of LLM agents
from the exponentially large space of possible agents. We tackle this selection
problem from the lens of submodular optimization. In particular, we develop
methods that offer different trade-offs regarding time complexity and
performance guarantees. Extensive experiments in crowdsourcing and educational
domains demonstrate that our approach constructs agents that more effectively
represent human populations compared to baselines. Moreover, behavioral
analyses on new tasks show that these agents reproduce the behavior patterns
and perspectives of the students and annotators they are designed to represent.

### 10. [Inductive Learning for Possibilistic Logic Programs Under Stable Models](http://arxiv.org/pdf/2510.07069v1)

Authors: Hongbo Hu, Yisong Wang, Yi Huang, Kewen Wang

Possibilistic logic programs (poss-programs) under stable models are a major
variant of answer set programming (ASP). While its semantics (possibilistic
stable models) and properties have been well investigated, the problem of
inductive reasoning has not been investigated yet. This paper presents an
approach to extracting poss-programs from a background program and examples
(parts of intended possibilistic stable models). To this end, the notion of
induction tasks is first formally defined, its properties are investigated and
two algorithms ilpsm and ilpsmmin for computing induction solutions are
presented. An implementation of ilpsmmin is also provided and experimental
results show that when inputs are ordinary logic programs, the prototype
outperforms a major inductive learning system for normal logic programs from
stable models on the datasets that are randomly generated.

### Hardware Architecture

### 1. [RTGS: Real-Time 3D Gaussian Splatting SLAM via Multi-Level Redundancy Reduction](http://arxiv.org/pdf/2510.06644v1)

Authors: Leshu Li, Jiayin Qin, Jie Peng, Zishen Wan, Huaizhi Qu, Ye Han, Pingqing Zheng, Hongsen Zhang, Yu, Cao, Tianlong Chen, Yang, Zhao

3D Gaussian Splatting (3DGS) based Simultaneous Localization and Mapping
(SLAM) systems can largely benefit from 3DGS's state-of-the-art rendering
efficiency and accuracy, but have not yet been adopted in resource-constrained
edge devices due to insufficient speed. Addressing this, we identify notable
redundancies across the SLAM pipeline for acceleration. While conceptually
straightforward, practical approaches are required to minimize the overhead
associated with identifying and eliminating these redundancies. In response, we
propose RTGS, an algorithm-hardware co-design framework that comprehensively
reduces the redundancies for real-time 3DGS-SLAM on edge. To minimize the
overhead, RTGS fully leverages the characteristics of the 3DGS-SLAM pipeline.
On the algorithm side, we introduce (1) an adaptive Gaussian pruning step to
remove the redundant Gaussians by reusing gradients computed during
backpropagation; and (2) a dynamic downsampling technique that directly reuses
the keyframe identification and alpha computing steps to eliminate redundant
pixels. On the hardware side, we propose (1) a subtile-level streaming strategy
and a pixel-level pairwise scheduling strategy that mitigates workload
imbalance via a Workload Scheduling Unit (WSU) guided by previous iteration
information; (2) a Rendering and Backpropagation (R&B) Buffer that accelerates
the rendering backpropagation by reusing intermediate data computed during
rendering; and (3) a Gradient Merging Unit (GMU) to reduce intensive memory
accesses caused by atomic operations while enabling pipelined aggregation.
Integrated into an edge GPU, RTGS achieves real-time performance (>= 30 FPS) on
four datasets and three algorithms, with up to 82.5x energy efficiency over the
baseline and negligible quality loss. Code is available at
https://github.com/UMN-ZhaoLab/RTGS.

### 2. [Hardware-Efficient CNNs: Interleaved Approximate FP32 Multipliers for Kernel Computation](http://arxiv.org/pdf/2510.06767v1)

Authors: Bindu G Gowda, Yogesh Goyal, Yash Gupta, Madhav Rao

Single-precision floating point (FP32) data format, defined by the IEEE 754
standard, is widely employed in scientific computing, signal processing, and
deep learning training, where precision is critical. However, FP32
multiplication is computationally expensive and requires complex hardware,
especially for precisely handling mantissa multiplication. In practical
applications like neural network inference, perfect accuracy is not always
necessary, minor multiplication errors often have little impact on final
accuracy. This enables trading precision for gains in area, power, and speed.
This work focuses on CNN inference using approximate FP32 multipliers, where
the mantissa multiplication is approximated by employing error-variant
approximate compressors, that significantly reduce hardware cost. Furthermore,
this work optimizes CNN performance by employing differently approximated FP32
multipliers and studying their impact when interleaved within the kernels
across the convolutional layers. The placement and ordering of these
approximate multipliers within each kernel are carefully optimized using the
Non-dominated Sorting Genetic Algorithm-II, balancing the trade-off between
accuracy and hardware efficiency.

### 3. [Evaluating Rapid Makespan Predictions for Heterogeneous Systems with Programmable Logic](http://arxiv.org/pdf/2510.06998v1)

Authors: Martin Wilhelm, Franz Freitag, Max Tzschoppe, Thilo Pionteck

Heterogeneous computing systems, which combine general-purpose processors
with specialized accelerators, are increasingly important for optimizing the
performance of modern applications. A central challenge is to decide which
parts of an application should be executed on which accelerator or, more
generally, how to map the tasks of an application to available devices.
Predicting the impact of a change in a task mapping on the overall makespan is
non-trivial. While there are very capable simulators, these generally require a
full implementation of the tasks in question, which is particularly
time-intensive for programmable logic. A promising alternative is to use a
purely analytical function, which allows for very fast predictions, but
abstracts significantly from reality. Bridging the gap between theory and
practice poses a significant challenge to algorithm developers. This paper aims
to aid in the development of rapid makespan prediction algorithms by providing
a highly flexible evaluation framework for heterogeneous systems consisting of
CPUs, GPUs and FPGAs, which is capable of collecting real-world makespan
results based on abstract task graph descriptions. We analyze to what extent
actual makespans can be predicted by existing analytical approaches.
Furthermore, we present common challenges that arise from high-level
characteristics such as data transfer overhead and device congestion in
heterogeneous systems.

### 4. [Cocoon: A System Architecture for Differentially Private Training with Correlated Noises](http://arxiv.org/pdf/2510.07304v1)

Authors: Donghwan Kim, Xin Gu, Jinho Baek, Timothy Lo, Younghoon Min, Kwangsik Shin, Jongryool Kim, Jongse Park, Kiwan Maeng

Machine learning (ML) models memorize and leak training data, causing serious
privacy issues to data owners. Training algorithms with differential privacy
(DP), such as DP-SGD, have been gaining attention as a solution. However,
DP-SGD adds a noise at each training iteration, which degrades the accuracy of
the trained model. To improve accuracy, a new family of approaches adds
carefully designed correlated noises, so that noises cancel out each other
across iterations. We performed an extensive characterization study of these
new mechanisms, for the first time to the best of our knowledge, and show they
incur non-negligible overheads when the model is large or uses large embedding
tables. Motivated by the analysis, we propose Cocoon, a hardware-software
co-designed framework for efficient training with correlated noises. Cocoon
accelerates models with embedding tables through pre-computing and storing
correlated noises in a coalesced format (Cocoon-Emb), and supports large models
through a custom near-memory processing device (Cocoon-NMP). On a real system
with an FPGA-based NMP device prototype, Cocoon improves the performance by
2.33-10.82x(Cocoon-Emb) and 1.55-3.06x (Cocoon-NMP).

### 5. [From Neural Sensing to Stimulation: An Interdisciplinary Roadmap for Neurotechnology](http://arxiv.org/pdf/2510.07116v1)

Authors: Ruben Ruiz-Mateos Serrano, Joe G Troughton, Nima Mirkhani, Natalia Martinez, Massimo Mariello, Jordan Tsigarides, Simon Williamson, Juan Sapriza, Ioana Susnoschi Luca, Antonio Dominguez-Alfaro, Estelle Cuttaz, Nicole Thompson, Sydney Swedick, Latifah Almulla, Amparo Guemes

Neurotechnologies are transforming how we measure, interpret, and modulate
brain-body interactions, integrating real-time sensing, computation, and
stimulation to enable precise physiological control. They hold transformative
potential across clinical and non-clinical domains, from treating disorders to
enhancing cognition and performance. Realizing this potential requires
navigating complex, interdisciplinary challenges spanning neuroscience,
materials science, device engineering, signal processing, computational
modelling, and regulatory and ethical frameworks. This Perspective presents a
strategic roadmap for neurotechnology development, created by early-career
researchers, highlighting their role at the intersection of disciplines and
their capacity to bridge traditional silos. We identify five cross-cutting
trade-offs that constrain progress across functionality, scalability,
adaptability, and translatability, and illustrate how technical domains
influence their resolution. Rather than a domain-specific review, we focus on
shared challenges and strategic opportunities that transcend disciplines. We
propose a unified framework for collaborative innovation and education,
highlight ethical and regulatory priorities, and outline a timeline for
overcoming key bottlenecks. By aligning technical development with
translational and societal needs, this roadmap aims to accelerate equitable,
effective, and future-ready adaptive neurotechnologies, guiding coordinated
efforts across the global research and innovation community.

### Computational Complexity

### 1. [On the complexity of estimating ground state entanglement and free energy](http://arxiv.org/pdf/2510.06796v1)

Authors: Sevag Gharibian, Jonas Kamminga

Understanding the entanglement structure of local Hamiltonian ground spaces
is a physically motivated problem, with applications ranging from tensor
network design to quantum error-correcting codes. To this end, we study the
complexity of estimating ground state entanglement, and more generally entropy
estimation for low energy states and Gibbs states. We find, in particular, that
the classes qq-QAM [Kobayashi, le Gall, Nishimura, SICOMP 2019] (a quantum
analogue of public-coin AM) and QMA(2) (QMA with unentangled proofs) play a
crucial role for such problems, showing: (1) Detecting a high-entanglement
ground state is qq-QAM-complete, (2) computing an additive error approximation
to the Helmholtz free energy (equivalently, a multiplicative error
approximation to the partition function) is in qq-QAM, (3) detecting a
low-entanglement ground state is QMA(2)-hard, and (4) detecting low energy
states which are close to product states can range from QMA-complete to
QMA(2)-complete. Our results make progress on an open question of [Bravyi,
Chowdhury, Gosset and Wocjan, Nature Physics 2022] on free energy, and yield
the first QMA(2)-complete Hamiltonian problem using local Hamiltonians (cf. the
sparse QMA(2)-complete Hamiltonian problem of [Chailloux, Sattath, CCC 2012]).

### 2. [Magic and communication complexity](http://arxiv.org/pdf/2510.07246v1)

Authors: Uma Girish, Alex May, Natalie Parham, Henry Yuen

We establish novel connections between magic in quantum circuits and
communication complexity. In particular, we show that functions computable with
low magic have low communication cost.
  Our first result shows that the $\mathsf{D}\|$ (deterministic simultaneous
message passing) cost of a Boolean function $f$ is at most the number of
single-qubit magic gates in a quantum circuit computing $f$ with any quantum
advice state. If we allow mid-circuit measurements and adaptive circuits, we
obtain an upper bound on the two-way communication complexity of $f$ in terms
of the magic + measurement cost of the circuit for $f$. As an application, we
obtain magic-count lower bounds of $\Omega(n)$ for the $n$-qubit generalized
Toffoli gate as well as the $n$-qubit quantum multiplexer.
  Our second result gives a general method to transform $\mathsf{Q}\|^*$
protocols (simultaneous quantum messages with shared entanglement) into
$\mathsf{R}\|^*$ protocols (simultaneous classical messages with shared
entanglement) which incurs only a polynomial blowup in the communication and
entanglement complexity, provided the referee's action in the $\mathsf{Q}\|^*$
protocol is implementable in constant $T$-depth. The resulting $\mathsf{R}\|^*$
protocols satisfy strong privacy constraints and are $\mathsf{PSM}^*$ protocols
(private simultaneous message passing with shared entanglement), where the
referee learns almost nothing about the inputs other than the function value.
As an application, we demonstrate $n$-bit partial Boolean functions whose
$\mathsf{R}\|^*$ complexity is $\mathrm{polylog}(n)$ and whose $\mathsf{R}$
(interactive randomized) complexity is $n^{\Omega(1)}$, establishing the first
exponential separations between $\mathsf{R}\|^*$ and $\mathsf{R}$ for Boolean
functions.

### 3. [When quantum resources backfire: Non-gaussianity and symplectic coherence in noisy bosonic circuits](http://arxiv.org/pdf/2510.07264v1)

Authors: Varun Upreti, Ulysse Chabaud, Zoë Holmes, Armando Angrisani

Analyzing the impact of noise is of fundamental importance to understand the
advantages provided by quantum systems. While the classical simulability of
noisy discrete-variable systems is increasingly well understood, noisy bosonic
circuits are more challenging to simulate and analyze. Here, we address this
gap by introducing the $\textit{displacement propagation}$ algorithm, a
continuous-variable analogue of Pauli propagation for simulating noisy bosonic
circuits. By exploring the interplay of noise and quantum resources, we
identify several computational phase transitions, revealing regimes where even
modest noise levels render bosonic circuits efficiently classically simulable.
In particular, our analysis reveals a surprising phenomenon: computational
resources usually associated with bosonic quantum advantage, namely
non-Gaussianity and symplectic coherence, can make the system easier to
classically simulate in presence of noise.

### 4. [Trickle-down Theorems via C-Lorentzian Polynomials II: Pairwise Spectral Influence and Improved Dobrushin's Condition](http://arxiv.org/pdf/2510.06549v1)

Authors: Jonathan Leake, Shayan Oveis Gharan

Let $\mu$ be a probability distribution on a multi-state spin system on a set
$V$ of sites. Equivalently, we can think of this as a $d$-partite simplical
complex with distribution $\mu$ on maximal faces. For any pair of vertices
$u,v\in V$, define the pairwise spectral influence $\mathcal{I}_{u,v}$ as
follows. Let $\sigma$ be a choice of spins $s_w\in S_w$ for every $w\in V
\setminus \{u,v\}$, and construct a matrix in $\mathbb{R}^{(S_u\cup S_v)\times
(S_u\cup S_v)}$ where for any $s_u\in S_u, s_v\in S_v$, the $(us_u,vs_v)$-entry
is the probability that $s_v$ is the spin of $v$ conditioned on $s_u$ being the
spin of $u$ and on $\sigma$. Then $\mathcal{I}_{u,v}$ is the maximal second
eigenvalue of this matrix, over all choices of spins for all $w \in V \setminus
\{u,v\}$. Equivalently, $\mathcal{I}_{u,v}$ is the maximum local spectral
expansion of links of codimension $2$ that include a spin for every $w \in V
\setminus \{u,v\}$.
  We show that if the largest eigenvalue of the pairwise spectral influence
matrix with entries $\mathcal{I}_{u,v}$ is bounded away from 1, i.e.
$\lambda_{\max}(\mathcal{I})\leq 1-\epsilon$ (and $X$ is connected), then the
Glauber dynamics mixes rapidly and generate samples from $\mu$. This
improves/generalizes the classical Dobrushin's influence matrix as the
$\mathcal{I}_{u,v}$ lower-bounds the classical influence of $u\to v$. As a
by-product, we also prove improved/almost optimal trickle-down theorems for
partite simplicial complexes. The proof builds on the trickle-down theorems via
$\mathcal{C}$-Lorentzian polynomials machinery recently developed by the
authors and Lindberg.

### 5. [Reconquering Bell sampling on qudits: stabilizer learning and testing, quantum pseudorandomness bounds, and more](http://arxiv.org/pdf/2510.06848v1)

Authors: Jonathan Allcock, Joao F. Doriguello, Gábor Ivanyos, Miklos Santha

Bell sampling is a simple yet powerful tool based on measuring two copies of
a quantum state in the Bell basis, and has found applications in a plethora of
problems related to stabiliser states and measures of magic. However, it was
not known how to generalise the procedure from qubits to $d$-level systems --
qudits -- for all dimensions $d > 2$ in a useful way. Indeed, a prior work of
the authors (arXiv'24) showed that the natural extension of Bell sampling to
arbitrary dimensions fails to provide meaningful information about the quantum
states being measured. In this paper, we overcome the difficulties encountered
in previous works and develop a useful generalisation of Bell sampling to
qudits of all $d\geq 2$. At the heart of our primitive is a new unitary, based
on Lagrange's four-square theorem, that maps four copies of any stabiliser
state $|\mathcal{S}\rangle$ to four copies of its complex conjugate
$|\mathcal{S}^\ast\rangle$ (up to some Pauli operator), which may be of
independent interest. We then demonstrate the utility of our new Bell sampling
technique by lifting several known results from qubits to qudits for any $d\geq
2$:
  1. Learning stabiliser states in $O(n^3)$ time with $O(n)$ samples;
  2. Solving the Hidden Stabiliser Group Problem in
$\tilde{O}(n^3/\varepsilon)$ time with $\tilde{O}(n/\varepsilon)$ samples;
  3. Testing whether $|\psi\rangle$ has stabiliser size at least $d^t$ or is
$\varepsilon$-far from all such states in $\tilde{O}(n^3/\varepsilon)$ time
with $\tilde{O}(n/\varepsilon)$ samples;
  4. Clifford circuits with at most $n/2$ single-qudit non-Clifford gates
cannot prepare pseudorandom states;
  5. Testing whether $|\psi\rangle$ has stabiliser fidelity at least
$1-\varepsilon_1$ or at most $1-\varepsilon_2$ with $O(d^2/\varepsilon_2)$
samples if $\varepsilon_1 = 0$ or $O(d^2/\varepsilon_2^2)$ samples if
$\varepsilon_1 = O(d^{-2})$.

### 6. [Computational complexity of the homology problem with orientable filtration: MA-completeness](http://arxiv.org/pdf/2510.07014v1)

Authors: Ryu Hayakawa, Casper Gyurik, Mahtab Yaghubi Rad, Vedran Dunjko

We show the existence of an MA-complete homology problem for a certain
subclass of simplicial complexes. The problem is defined through a new concept
of orientability of simplicial complexes that we call a "uniform orientable
filtration", which is related to sign-problem freeness in homology. The
containment in MA is achieved through the design of new, higher-order random
walks on simplicial complexes associated with the filtration. For the
MA-hardness, we design a new gadget with which we can reduce from an MA-hard
stoquastic satisfiability problem. Therefore, our result provides the first
natural MA-complete problem for higher-order random walks on simplicial
complexes, combining the concepts of topology, persistent homology, and quantum
computing.

### 7. [MIPco=coRE](http://arxiv.org/pdf/2510.07162v1)

Authors: Junqiao Lin

In 2020, a landmark result by Ji, Natarajan, Vidick, Wright, and Yuen showed
that MIP*, the class of languages that can be decided by a classical verifier
interacting with multiple computationally unbounded provers sharing
entanglement in the tensor product model, is equal to RE. We show that the
class MIPco, a complexity class defined similarly to MIP* except with provers
sharing the commuting operator model of entanglement, is equal to the class
coRE. This shows that giving the provers two different models of entanglement
leads to two completely different computational powers for interactive proof
systems. Our proof builds upon the compression theorem used in the proof of
MIP*=RE, and we use the tracially embeddable strategies framework to show that
the same compression procedure in MIP* =RE also has the same desired property
in the commuting operator setting. We also give a more streamlined proof of the
compression theorem for non-local games by incorporating the synchronous
framework used by Mousavi et al. [STOC 2022], as well as the improved Pauli
basis test introduced by de la Salle [ArXiv:2204.07084].
  We introduce a new equivalence condition for RE/coRE-complete problems, which
we call the weakly compressible condition. We show that both MIP* and MIPco
satisfy this condition through the compression theorem, and thereby establish
that the uncomputability for MIP* and MIPco can be proved under a unified
framework (despite these two complexity classes being different). Notably, this
approach also gives an alternative proof of the MIP*=RE theorem, which does not
rely on the preservation of the entanglement bound. In addition to non-local
games, this new condition could also potentially be applicable to other
decision problems.

### 8. [Clifford testing: algorithms and lower bounds](http://arxiv.org/pdf/2510.07164v1)

Authors: Marcel Hinsche, Zongbo Bao, Philippe van Dordrecht, Jens Eisert, Jop Briët, Jonas Helsen

We consider the problem of Clifford testing, which asks whether a black-box
$n$-qubit unitary is a Clifford unitary or at least $\varepsilon$-far from
every Clifford unitary. We give the first 4-query Clifford tester, which
decides this problem with probability $\mathrm{poly}(\varepsilon)$. This
contrasts with the minimum of 6 copies required for the closely-related task of
stabilizer testing. We show that our tester is tolerant, by adapting techniques
from tolerant stabilizer testing to our setting. In doing so, we settle in the
positive a conjecture of Bu, Gu and Jaffe, by proving a polynomial inverse
theorem for a non-commutative Gowers 3-uniformity norm. We also consider the
restricted setting of single-copy access, where we give an $O(n)$-query
Clifford tester that requires no auxiliary memory qubits or adaptivity. We
complement this with a lower bound, proving that any such, potentially
adaptive, single-copy algorithm needs at least $\Omega(n^{1/4})$ queries. To
obtain our results, we leverage the structure of the commutant of the Clifford
group, obtaining several technical statements that may be of independent
interest.

### 9. [Efficient reductions from a Gaussian source with applications to statistical-computational tradeoffs](http://arxiv.org/pdf/2510.07250v1)

Authors: Mengqi Lou, Guy Bresler, Ashwin Pananjady

Given a single observation from a Gaussian distribution with unknown mean
$\theta$, we design computationally efficient procedures that can approximately
generate an observation from a different target distribution $Q_{\theta}$
uniformly for all $\theta$ in a parameter set. We leverage our technique to
establish reduction-based computational lower bounds for several canonical
high-dimensional statistical models under widely-believed conjectures in
average-case complexity. In particular, we cover cases in which:
  1. $Q_{\theta}$ is a general location model with non-Gaussian distribution,
including both light-tailed examples (e.g., generalized normal distributions)
and heavy-tailed ones (e.g., Student's $t$-distributions). As a consequence, we
show that computational lower bounds proved for spiked tensor PCA with Gaussian
noise are universal, in that they extend to other non-Gaussian noise
distributions within our class.
  2. $Q_{\theta}$ is a normal distribution with mean $f(\theta)$ for a general,
smooth, and nonlinear link function $f:\mathbb{R} \rightarrow \mathbb{R}$.
Using this reduction, we construct a reduction from symmetric mixtures of
linear regressions to generalized linear models with link function $f$, and
establish computational lower bounds for solving the $k$-sparse generalized
linear model when $f$ is an even function. This result constitutes the first
reduction-based confirmation of a $k$-to-$k^2$ statistical-to-computational gap
in $k$-sparse phase retrieval, resolving a conjecture posed by Cai et al.
(2016). As a second application, we construct a reduction from the sparse
rank-1 submatrix model to the planted submatrix model, establishing a pointwise
correspondence between the phase diagrams of the two models that faithfully
preserves regions of computational hardness and tractability.

### Computational Engineering

### 1. [A Higher-Order Time Domain Boundary Element Formulation based on Isogeometric Analysis and the Convolution Quadrature Method](http://arxiv.org/pdf/2510.06804v1)

Authors: Thomas Kramer, Benjamin Marussig, Martin Schanz

An isogeometric boundary element method (BEM) is presented to solve
scattering problems in an isotropic homogeneous medium. We consider wave
problems governed by the scalar wave equation as in acoustics and the
Lam\'e-Navier equations for elastodynamics considering the theory of linear
elasticity. The underlying boundary integral equations imply time-dependent
convolution integrals and allow us to determine the sought quantities in the
bounded interior or the unbounded exterior after solving for the unknown Cauchy
data. In the present work, the time-dependent convolution integrals are
approximated by multi-stage Runge-Kutta (RK) based convolution quadratures that
involve steady-state solutions in the Laplace domain. The proposed method
discretizes the spatial variables in the framework of isogeometric analysis
(IGA), entailing a patchwise smooth spline basis. Overall, it enables high
convergence rates in space and time. The implementation scheme follows an
element structure defined by the non-empty knot spans in the knot vectors and
local, uniform Bernstein polynomials as basis functions. The algorithms to
localize the basis functions on the elements are outlined and explained. The
solutions of the mixed problems are approximated by the BEM based on a
symmetric Galerkin variational formulation and a collocation method. We
investigate convergence rates of the approximative solutions in a mixed space
and time error norm.

### 2. [A Framework for Measuring How News Topics Drive Stock Movement](http://arxiv.org/pdf/2510.06864v1)

Authors: Qizhao Chen

In modern financial markets, news plays a critical role in shaping investor
sentiment and influencing stock price movements. However, most existing studies
aggregate daily news sentiment into a single score, potentially overlooking
important variations in topic content and relevance. This simplification may
mask nuanced relationships between specific news themes and market responses.
To address this gap, this paper proposes a novel framework to examine how
different news topics influence stock price movements. The framework encodes
individual news headlines into dense semantic embeddings using a pretrained
sentence transformer, then applies K-means clustering to identify distinct news
topics. Topic exposures are incorporated as explanatory variables in an
ordinary least squares regression to quantify their impact on daily stock
returns. Applied to Apple Inc., the framework reveals that certain topics are
significantly associated with positive or negative next-day returns, while
others have no measurable effect. These findings highlight the importance of
topic-level analysis in understanding the relationship between news content and
financial markets. The proposed framework provides a scalable approach for both
researchers and practitioners to assess the informational value of different
news topics and suggests a promising direction for improving predictive models
of stock price movement.

### 3. [FEAorta: A Fully Automated Framework for Finite Element Analysis of the Aorta From 3D CT Images](http://arxiv.org/pdf/2510.06621v1)

Authors: Jiasong Chen, Linchen Qian, Ruonan Gong, Christina Sun, Tongran Qin, Thuy Pham, Caitlin Martin, Mohammad Zafar, John Elefteriades, Wei Sun, Liang Liang

Aortic aneurysm disease ranks consistently in the top 20 causes of death in
the U.S. population. Thoracic aortic aneurysm is manifested as an abnormal
bulging of thoracic aortic wall and it is a leading cause of death in adults.
From the perspective of biomechanics, rupture occurs when the stress acting on
the aortic wall exceeds the wall strength. Wall stress distribution can be
obtained by computational biomechanical analyses, especially structural Finite
Element Analysis. For risk assessment, probabilistic rupture risk of TAA can be
calculated by comparing stress with material strength using a material failure
model. Although these engineering tools are currently available for TAA rupture
risk assessment on patient specific level, clinical adoption has been limited
due to two major barriers: labor intensive 3D reconstruction current patient
specific anatomical modeling still relies on manual segmentation, making it
time consuming and difficult to scale to a large patient population, and
computational burden traditional FEA simulations are resource intensive and
incompatible with time sensitive clinical workflows. The second barrier was
successfully overcome by our team through the development of the PyTorch FEA
library and the FEA DNN integration framework. By incorporating the FEA
functionalities within PyTorch FEA and applying the principle of static
determinacy, we reduced the FEA based stress computation time to approximately
three minutes per case. Moreover, by integrating DNN and FEA through the
PyTorch FEA library, our approach further decreases the computation time to
only a few seconds per case. This work focuses on overcoming the first barrier
through the development of an end to end deep neural network capable of
generating patient specific finite element meshes of the aorta directly from 3D
CT images.

### 4. [TOMATOES: Topology and Material Optimization for Latent Heat Thermal Energy Storage Devices](http://arxiv.org/pdf/2510.07057v1)

Authors: Rahul Kumar Padhy, Krishnan Suresh, Aaditya Chandrasekhar

Latent heat thermal energy storage (LHTES) systems are compelling candidates
for energy storage, primarily owing to their high storage density. Improving
their performance is crucial for developing the next-generation efficient and
cost effective devices. Topology optimization (TO) has emerged as a powerful
computational tool to design LHTES systems by optimally distributing a
high-conductivity material (HCM) and a phase change material (PCM). However,
conventional TO typically limits to optimizing the geometry for a fixed,
pre-selected materials. This approach does not leverage the large and expanding
databases of novel materials. Consequently, the co-design of material and
geometry for LHTES remains a challenge and unexplored.
  To address this limitation, we present an automated design framework for the
concurrent optimization of material choice and topology. A key challenge is the
discrete nature of material selection, which is incompatible with the
gradient-based methods used for TO. We overcome this by using a data-driven
variational autoencoder (VAE) to project discrete material databases for both
the HCM and PCM onto continuous and differentiable latent spaces. These
continuous material representations are integrated into an end-to-end
differentiable, transient nonlinear finite-element solver that accounts for
phase change. We demonstrate this framework on a problem aimed at maximizing
the discharged energy within a specified time, subject to cost constraints. The
effectiveness of the proposed method is validated through several illustrative
examples.

### 5. [Diffusion-Augmented Reinforcement Learning for Robust Portfolio Optimization under Stress Scenarios](http://arxiv.org/pdf/2510.07099v1)

Authors: Himanshu Choudhary, Arishi Orra, Manoj Thakur

In the ever-changing and intricate landscape of financial markets, portfolio
optimisation remains a formidable challenge for investors and asset managers.
Conventional methods often struggle to capture the complex dynamics of market
behaviour and align with diverse investor preferences. To address this, we
propose an innovative framework, termed Diffusion-Augmented Reinforcement
Learning (DARL), which synergistically integrates Denoising Diffusion
Probabilistic Models (DDPMs) with Deep Reinforcement Learning (DRL) for
portfolio management. By leveraging DDPMs to generate synthetic market crash
scenarios conditioned on varying stress intensities, our approach significantly
enhances the robustness of training data. Empirical evaluations demonstrate
that DARL outperforms traditional baselines, delivering superior risk-adjusted
returns and resilience against unforeseen crises, such as the 2025 Tariff
Crisis. This work offers a robust and practical methodology to bolster stress
resilience in DRL-driven financial applications.

### Computation and Language

### 1. [Flipping the Dialogue: Training and Evaluating User Language Models](http://arxiv.org/pdf/2510.06552v1)

Authors: Tarek Naous, Philippe Laban, Wei Xu, Jennifer Neville

Conversations with LMs involve two participants: a human user leading the
conversation, and an LM assistant responding to the user's request. To satisfy
this specific role, LMs are post-trained to be helpful assistants -- optimized
to produce exhaustive and well-structured responses, free of ambiguity and
grammar errors. User utterances, on the other hand, are rarely perfected, with
each user phrasing requests in unique ways, sometimes putting in partial effort
at each turn and refining on the fly. To evaluate LM performance in realistic
settings, prior work simulated users in multi-turn conversations, often
prompting an LLM originally trained to be a helpful assistant to act as a user.
However, we show that assistant LMs make for poor user simulators, with the
surprising finding that better assistants yield worse simulators. Instead, we
introduce purpose-built User Language Models (User LMs) - models post-trained
to simulate human users in multi-turn conversations. Through various
evaluations, we show how User LMs align better with human behavior and achieve
better simulation robustness than existing simulation methods. When leveraging
User LMs to simulate coding and math conversations, the performance of a strong
assistant (GPT-4o) drops from 74.6% to 57.4%, confirming that more realistic
simulation environments lead to assistant struggles as they fail to cope with
the nuances of users in multi-turn setups.

### 2. [TinyScientist: An Interactive, Extensible, and Controllable Framework for Building Research Agents](http://arxiv.org/pdf/2510.06579v1)

Authors: Haofei Yu, Keyang Xuan, Fenghai Li, Kunlun Zhu, Zijie Lei, Jiaxun Zhang, Ziheng Qi, Kyle Richardson, Jiaxuan You

Automatic research with Large Language Models (LLMs) is rapidly gaining
importance, driving the development of increasingly complex workflows involving
multi-agent systems, planning, tool usage, code execution, and human-agent
interaction to accelerate research processes. However, as more researchers and
developers begin to use and build upon these tools and platforms, the
complexity and difficulty of extending and maintaining such agentic workflows
have become a significant challenge, particularly as algorithms and
architectures continue to advance. To address this growing complexity,
TinyScientist identifies the essential components of the automatic research
workflow and proposes an interactive, extensible, and controllable framework
that easily adapts to new tools and supports iterative growth. We provide an
open-source codebase, an interactive web demonstration, and a PyPI Python
package to make state-of-the-art auto-research pipelines broadly accessible to
every researcher and developer.

### 3. [Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?](http://arxiv.org/pdf/2510.06594v1)

Authors: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

Jailbreaking large language models (LLMs) has emerged as a pressing concern
with the increasing prevalence and accessibility of conversational LLMs.
Adversarial users often exploit these models through carefully engineered
prompts to elicit restricted or sensitive outputs, a strategy widely referred
to as jailbreaking. While numerous defense mechanisms have been proposed,
attackers continuously develop novel prompting techniques, and no existing
model can be considered fully resistant. In this study, we investigate the
jailbreak phenomenon by examining the internal representations of LLMs, with a
focus on how hidden layers respond to jailbreak versus benign prompts.
Specifically, we analyze the open-source LLM GPT-J and the state-space model
Mamba2, presenting preliminary findings that highlight distinct layer-wise
behaviors. Our results suggest promising directions for further research on
leveraging internal model dynamics for robust jailbreak detection and defense.

### 4. [Aligning Large Language Models via Fully Self-Synthetic Data](http://arxiv.org/pdf/2510.06652v1)

Authors: Shangjian Yin, Zhepei Wei, Xinyu Zhu, Wei-Lin Chen, Yu Meng

Traditional reinforcement learning from human feedback (RLHF) for large
language models (LLMs) relies on expensive human-annotated datasets, while
Reinforcement Learning from AI Feedback (RLAIF) also incurs significant costs,
requiring the collection of diverse prompts and corresponding responses, often
necessitating external reward models or proprietary models like GPT-4 to
annotate preference pairs. In this work, we introduce Self-Alignment
Optimization (SAO), a fully self-synthetic framework for LLM alignment, where
all training data, including prompts (i.e., user queries), responses, and
preferences, are generated by the model itself. Specifically, SAO first
instructs the LLM to engage in persona role-play and generate diverse prompts
and responses, which are then self-evaluated for preference optimization.
Extensive experiments demonstrate that SAO effectively enhances the model's
chat capabilities on standard benchmarks like AlpacaEval~2.0, while maintaining
strong performance on downstream objective tasks (e.g., question-answering,
math reasoning). Our work provides a practical solution for self-improvement in
aligning LLMs, and the code for reproducing our results is available at:
https://github.com/SJY8460/SAO.

### 5. [ToolMem: Enhancing Multimodal Agents with Learnable Tool Capability Memory](http://arxiv.org/pdf/2510.06664v1)

Authors: Yunzhong Xiao, Yangmin Li, Hewei Wang, Yunlong Tang, Zora Zhiruo Wang

Agents utilizing tools powered by large language models (LLMs) or
vision-language models (VLMs) have demonstrated remarkable progress in diverse
tasks across text and visual modalities. Unlike traditional tools such as
calculators, which give deterministic outputs, neural tools perform uncertainly
across task scenarios. While different tools for a task may excel in varied
scenarios, existing agents typically rely on fixed tools, thus limiting the
flexibility in selecting the most suitable tool for specific tasks. In
contrast, humans snowball their understanding of the capabilities of different
tools by interacting with them, and apply this knowledge to select the optimal
tool when solving a future task. To build agents that similarly benefit from
this process, we propose ToolMem that enables agents to develop memories of
tool capabilities from previous interactions, by summarizing their strengths
and weaknesses and storing them in memory; at inference, the agent can retrieve
relevant entries from ToolMem, and select the best tool to solve individual
tasks more accurately. We evaluate ToolMem on learning varied text generation
and text-to-image generation neural tools. Compared to no-memory, generic
agents, we find ToolMem-augmented agents predict tool performance 14.8% and
28.7% more accurately across text and multimodal generation scenarios.
Moreover, ToolMem facilitates optimal tool selection among multiple choices by
21% and 24% absolute increases in respective scenarios.

### 6. [PIKA: Expert-Level Synthetic Datasets for Post-Training Alignment from Scratch](http://arxiv.org/pdf/2510.06670v1)

Authors: Shangjian Yin, Shining Liang, Wenbiao Ding, Yuli Qian, Zhouxing Shi, Hongzhi Li, Yutao Xie

Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone
for aligning large language models (LLMs). However, its effectiveness depends
on high-quality instruction data. Most existing alignment datasets are either
private or require costly human annotation, which limits reproducibility and
scalability. Even with Reinforcement Learning from AI Feedback (RLAIF),
concerns about data quality remain. Moreover, it is unclear how much data is
actually required to fine-tune a base model into a strong instruction-following
model. Current approaches often rely on over 300k examples even at the
supervised fine-tuning (SFT) stage, yet they still underperform compared to
proprietary models, creating barriers for academic and resource-limited
communities. To address this gap, we introduce PiKa, a data-efficient family of
expert-level alignment datasets. In particular, the PiKa-SFT dataset uses only
30k SFT examples, far fewer than state-of-the-art datasets like Magpie. Through
evaluations by fine-tuning Llama-3-8B-Base on PiKa and other public datasets,
we show that PiKa-SFT outperforms models trained on much larger data. On
AlpacaEval 2.0 and Arena-Hard benchmarks, PiKa-SFT fine-tuning even surpasses
the official Llama-3-8B-Instruct model trained on over 10 million proprietary
examples. We further extend our study by training the Qwen2.5 series (0.5B to
7B) on PiKa-SFT, achieving consistent gains. These findings demonstrate that
high-quality alignment can be achieved with significantly less data, offering a
scalable path for open-source LLM alignment. Code and data:
https://github.com/SJY8460/PiKa.

### 7. [How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](http://arxiv.org/pdf/2510.06700v1)

Authors: Leonardo Bertolazzi, Sandro Pezzelle, Raffaelle Bernardi

Both humans and large language models (LLMs) exhibit content effects: biases
in which the plausibility of the semantic content of a reasoning problem
influences judgments regarding its logical validity. While this phenomenon in
humans is best explained by the dual-process theory of reasoning, the
mechanisms behind content effects in LLMs remain unclear. In this work, we
address this issue by investigating how LLMs encode the concepts of validity
and plausibility within their internal representations. We show that both
concepts are linearly represented and strongly aligned in representational
geometry, leading models to conflate plausibility with validity. Using steering
vectors, we demonstrate that plausibility vectors can causally bias validity
judgements, and vice versa, and that the degree of alignment between these two
concepts predicts the magnitude of behavioral content effects across models.
Finally, we construct debiasing vectors that disentangle these concepts,
reducing content effects and improving reasoning accuracy. Our findings advance
understanding of how abstract logical concepts are represented in LLMs and
highlight representational interventions as a path toward more logical systems.

### 8. [PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs](http://arxiv.org/pdf/2510.06730v1)

Authors: Manuel Frank, Haithem Afli

Current evaluations of sentence embedding models typically rely on static
test beds such as the Massive Text Embedding Benchmark (MTEB). While
invaluable, repeated tuning on a fixed suite can inflate reported performance
and obscure real-world robustness. We introduce the Paraphrasing Text Embedding
Benchmark (PTEB), a dynamic protocol that stochastically generates
meaning-preserving paraphrases at evaluation time and aggregates results across
multiple runs. Using a cost-efficient LLM-based method grounded in semantic
textual similarity gold ratings, we show that LLMs generate token-diverse but
semantically preserving, paraphrases. Across 7 MTEB tasks, we validate our
hypothesis that the performance of sentence encoders is sensitive to changes in
token space even when semantics remain fixed. We also observe that smaller
models are not disproportionately affected relative to larger ones. Our results
are statistically robust over multiple runs and we extended our experiments to
3 multilingual datasets covering 10 languages. More generally, we aim to
propose a new evaluation paradigm in NLP that relies less on static,
pre-defined benchmarks but shifts towards dynamic, stochastic evaluation
leveraging eval-time compute.

### 9. [AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](http://arxiv.org/pdf/2510.06738v1)

Authors: Boyi Zeng, Lin Chen, Ziwei He, Xinbing Wang, Zhouhan Lin

Protecting the intellectual property of large language models (LLMs) is
crucial, given the substantial resources required for their training.
Consequently, there is an urgent need for both model owners and third parties
to determine whether a suspect LLM is trained from scratch or derived from an
existing base model. However, the intensive post-training processes that models
typically undergo-such as supervised fine-tuning, extensive continued
pretraining, reinforcement learning, multi-modal extension, pruning, and
upcycling-pose significant challenges to reliable identification. In this work,
we propose a training-free fingerprinting method based on weight matrices. We
leverage the Linear Assignment Problem (LAP) and an unbiased Centered Kernel
Alignment (CKA) similarity to neutralize the effects of parameter
manipulations, yielding a highly robust and high-fidelity similarity metric. On
a comprehensive testbed of 60 positive and 90 negative model pairs, our method
demonstrates exceptional robustness against all six aforementioned
post-training categories while exhibiting a near-zero risk of false positives.
By achieving perfect scores on all classification metrics, our approach
establishes a strong basis for reliable model lineage verification. Moreover,
the entire computation completes within 30s on an NVIDIA 3090 GPU. The code is
available at https://github.com/LUMIA-Group/AWM.

### 10. [TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs](http://arxiv.org/pdf/2510.06747v1)

Authors: I-Fan Lin, Faegheh Hasibi, Suzan Verberne

In this paper, we propose a training-free and label-free method for short
text clustering that can be used on top of any existing embedder. In the
context of customer-facing chatbots, companies are dealing with large amounts
of user utterances that need to be clustered according to their intent. In
these commercial settings, no labeled data is typically available, and the
number of clusters is not known. Our method is based on iterative vector
updating: it constructs sparse vectors based on representative texts, and then
iteratively refines them through LLM guidance. Our method achieves comparable
or superior results to state-of-the-art methods that use contrastive learning,
but without assuming prior knowledge of clusters or labels. Experiments on
diverse datasets and smaller LLMs show that our method is model agnostic and
can be applied to any embedder, with relatively small LLMs, and different
clustering methods. We also show that our method scales to large datasets,
reducing the computational cost of the LLM. These low-resource, adaptable
settings and the scalability of our method make it more aligned with real-world
scenarios than existing clustering methods.

### Cryptography and Security

### 1. [SpyChain: Multi-Vector Supply Chain Attacks on Small Satellite Systems](http://arxiv.org/pdf/2510.06535v1)

Authors: Jack Vanlyssel, Enrique Sobrados, Ramsha Anwar, Gruia-Catalin Roman, Afsah Anwar

Small satellites are integral to scientific, commercial, and defense
missions, but reliance on commercial off-the-shelf (COTS) hardware broadens
their attack surface. Although supply chain threats are well studied in other
cyber-physical domains, their feasibility and stealth in space systems remain
largely unexplored. Prior work has focused on flight software, which benefits
from strict security practices and oversight. In contrast, auxiliary COTS
components often lack robust assurance yet enjoy comparable access to critical
on-board resources, including telemetry, system calls, and the software bus.
Despite this privileged access, the insider threat within COTS hardware supply
chains has received little attention. In this work, we present SpyChain, the
first end-to-end design and implementation of independent and colluding
hardware supply chain threats targeting small satellites. Using NASA's
satellite simulation (NOS3), we demonstrate that SpyChain can evade testing,
exfiltrate telemetry, disrupt operations, and launch Denial of Service (DoS)
attacks through covert channels that bypass ground monitoring. Our study traces
an escalation from a simple solo component to dynamic, coordinating malware,
introducing a taxonomy of stealth across five scenarios. We showcase how
implicit trust in auxiliary components enables covert persistence and reveal
novel attack vectors, highlighting a new multi-component execution technique
that is now incorporated into the SPARTA matrix. Our findings are reinforced by
acknowledgment and affirmation from NASA's NOS3 team. Finally, we implement
lightweight onboard defenses, including runtime monitoring, to mitigate threats
like SpyChain.

### 2. [Auto-Stega: An Agent-Driven System for Lifelong Strategy Evolution in LLM-Based Text Steganography](http://arxiv.org/pdf/2510.06565v1)

Authors: Jiuan Zhou, Yu Cheng, Yuan Xie, Zhaoxia Yin

With the rapid progress of LLMs, high quality generative text has become
widely available as a cover for text steganography. However, prevailing methods
rely on hand-crafted or pre-specified strategies and struggle to balance
efficiency, imperceptibility, and security, particularly at high embedding
rates. Accordingly, we propose Auto-Stega, an agent-driven self-evolving
framework that is the first to realize self-evolving steganographic strategies
by automatically discovering, composing, and adapting strategies at inference
time; the framework operates as a closed loop of generating, evaluating,
summarizing, and updating that continually curates a structured strategy
library and adapts across corpora, styles, and task constraints. A decoding LLM
recovers the information under the shared strategy. To handle high embedding
rates, we introduce PC-DNTE, a plug-and-play algorithm that maintains alignment
with the base model's conditional distribution at high embedding rates,
preserving imperceptibility while enhancing security. Experimental results
demonstrate that at higher embedding rates Auto-Stega achieves superior
performance with gains of 42.2\% in perplexity and 1.6\% in anti-steganalysis
performance over SOTA methods.

### 3. [Code Agent can be an End-to-end System Hacker: Benchmarking Real-world Threats of Computer-use Agent](http://arxiv.org/pdf/2510.06607v1)

Authors: Weidi Luo, Qiming Zhang, Tianyu Lu, Xiaogeng Liu, Bin Hu, Hung-Chun Chiu, Siyuan Ma, Yizhe Zhang, Xusheng Xiao, Yinzhi Cao, Zhen Xiang, Chaowei Xiao

Computer-use agent (CUA) frameworks, powered by large language models (LLMs)
or multimodal LLMs (MLLMs), are rapidly maturing as assistants that can
perceive context, reason, and act directly within software environments. Among
their most critical applications is operating system (OS) control. As CUAs in
the OS domain become increasingly embedded in daily operations, it is
imperative to examine their real-world security implications, specifically
whether CUAs can be misused to perform realistic, security-relevant attacks.
Existing works exhibit four major limitations: Missing attacker-knowledge model
on tactics, techniques, and procedures (TTP), Incomplete coverage for
end-to-end kill chains, unrealistic environment without multi-host and
encrypted user credentials, and unreliable judgment dependent on
LLM-as-a-Judge. To address these gaps, we propose AdvCUA, the first benchmark
aligned with real-world TTPs in MITRE ATT&CK Enterprise Matrix, which comprises
140 tasks, including 40 direct malicious tasks, 74 TTP-based malicious tasks,
and 26 end-to-end kill chains, systematically evaluates CUAs under a realistic
enterprise OS security threat in a multi-host environment sandbox by hard-coded
evaluation. We evaluate the existing five mainstream CUAs, including ReAct,
AutoGPT, Gemini CLI, Cursor CLI, and Cursor IDE based on 8 foundation LLMs. The
results demonstrate that current frontier CUAs do not adequately cover OS
security-centric threats. These capabilities of CUAs reduce dependence on
custom malware and deep domain expertise, enabling even inexperienced attackers
to mount complex enterprise intrusions, which raises social concern about the
responsibility and security of CUAs.

### 4. [I Can't Patch My OT Systems! A Look at CISA's KEVC Workarounds & Mitigations for OT](http://arxiv.org/pdf/2510.06951v1)

Authors: Philip Huff, Nishka Gandu, Pavel Novák

We examine the state of publicly available information about known
exploitable vulnerabilities applicable to operational technology (OT)
environments. Specifically, we analyze the Known Exploitable Vulnerabilities
Catalog (KEVC) maintained by the US Department of Homeland Security
Cybersecurity and Infrastructure Security Agency (CISA) to assess whether
currently available data is sufficient for effective and reliable remediation
in OT settings. Our team analyzed all KEVC entries through July 2025 to
determine the extent to which OT environments can rely on existing remediation
recommendations. We found that although most entries in the KEVC could affect
OT environments, only 13% include vendor workarounds or mitigations as
alternatives to patching. This paper also examines the feasibility of
developing such alternatives based on vulnerability and exploit
characteristics, and we present early evidence of success with this approach.

### 5. [A multi-layered embedded intrusion detection framework for programmable logic controllers](http://arxiv.org/pdf/2510.07171v1)

Authors: Rishabh Das. Aaron Werth, Tommy Morris

Industrial control system (ICS) operations use trusted endpoints like human
machine interfaces (HMIs) and workstations to relay commands to programmable
logic controllers (PLCs). Because most PLCs lack layered defenses, compromise
of a trusted endpoint can drive unsafe actuator commands and risk
safety-critical operation. This research presents an embedded intrusion
detection system that runs inside the controller and uses header-level
telemetry to detect and respond to network attacks. The system combines a
semi-supervised anomaly detector and a supervised attack classifier. We
evaluate the approach on a midstream oil-terminal testbed using three datasets
collected during tanker-truck loading. The anomaly detector achieves zero
missed attacks, corresponding to 0.998 Matthews correlation. The supervised
stage attains 97.37 percent hold-out accuracy and 97.03 percent external
accuracy. The embedded design adds a median of 2,031 microseconds of end-to-end
latency and does not impact PLC's cycle time. The proposed architecture
provides a multi-layer embedded security that meets the real-time requirements
of an industrial system.

### 6. [Exposing LLM User Privacy via Traffic Fingerprint Analysis: A Study of Privacy Risks in LLM Agent Interactions](http://arxiv.org/pdf/2510.07176v1)

Authors: Yixiang Zhang, Xinhao Deng, Zhongyi Gu, Yihao Chen, Ke Xu, Qi Li, Jianping Wu

Large Language Models (LLMs) are increasingly deployed as agents that
orchestrate tasks and integrate external tools to execute complex workflows. We
demonstrate that these interactive behaviors leave distinctive fingerprints in
encrypted traffic exchanged between users and LLM agents. By analyzing traffic
patterns associated with agent workflows and tool invocations, adversaries can
infer agent activities, distinguish specific agents, and even profile sensitive
user attributes. To highlight this risk, we develop AgentPrint, which achieves
an F1-score of 0.866 in agent identification and attains 73.9% and 69.1% top-3
accuracy in user attribute inference for simulated- and real-user settings,
respectively. These results uncover an overlooked risk: the very interactivity
that empowers LLM agents also exposes user privacy, underscoring the urgent
need for technical countermeasures alongside regulatory and policy safeguards.

### 7. [Security-Robustness Trade-offs in Diffusion Steganography: A Comparative Analysis of Pixel-Space and VAE-Based Architectures](http://arxiv.org/pdf/2510.07219v1)

Authors: Yuhua Xu, Wei Sun, Chengpei Tang, Jiaxing Lu, Jingying Zhou, Chen Gu

Current generative steganography research mainly pursues computationally
expensive mappings to perfect Gaussian priors within single diffusion model
architectures. This work introduces an efficient framework based on approximate
Gaussian mapping governed by a scale factor calibrated through capacity-aware
adaptive optimization. Using this framework as a unified analytical tool,
systematic comparative analysis of steganography in pixel-space models versus
VAE-based latent-space systems is conducted. The investigation reveals a
pronounced architecture dependent security-robustness trade-off: pixel-space
models achieve high security against steganalysis but exhibit fragility to
channel distortions, while VAE-based systems like Stable Diffusion offer
substantial robustness at the cost of security vulnerabilities. Further
analysis indicates that the VAE component drives this behavior through opposing
mechanisms where the encoder confers robustness via manifold regularization
while the decoder introduces vulnerabilities by amplifying latent perturbations
into detectable artifacts. These findings characterize the conflicting
architectural roles in generative steganography and establish a foundation for
future research.

### 8. [Distilling Lightweight Language Models for C/C++ Vulnerabilities](http://arxiv.org/pdf/2510.06645v1)

Authors: Zhiyuan Wei, Xiaoxuan Yang, Jing Sun, Zijian Zhang

The increasing complexity of modern software systems exacerbates the
prevalence of security vulnerabilities, posing risks of severe breaches and
substantial economic loss. Consequently, robust code vulnerability detection is
essential for software security. While Large Language Models (LLMs) have
demonstrated remarkable capabilities in natural language processing, their
potential for automated code vulnerability detection remains underexplored.
This paper presents FineSec, a novel framework that harnesses LLMs through
knowledge distillation to enable efficient and precise vulnerability
identification in C/C++ codebases. FineSec utilizes knowledge distillation to
transfer expertise from large teacher models to compact student models,
achieving high accuracy with minimal computational cost. By integrating data
preparation, training, evaluation, and continuous learning into a unified,
single-task workflow, FineSec offers a streamlined approach. Extensive
evaluations on C/C++ codebases demonstrate its superiority over both base
models and larger LLMs in identifying complex vulnerabilities and logical
flaws, establishing FineSec as a practical and scalable solution for real-world
software security. To facilitate reproducibility, the datasets, source code,
and experimental results are made publicly available at:
https://github.com/yangxiaoxuan123/FineSec_detect.

### 9. [Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial?](http://arxiv.org/pdf/2510.06692v1)

Authors: Akira Ito, Takayuki Miura, Yosuke Todo

Deep Neural Networks (DNNs) have attracted significant attention, and their
internal models are now considered valuable intellectual assets. Extracting
these internal models through access to a DNN is conceptually similar to
extracting a secret key via oracle access to a block cipher. Consequently,
cryptanalytic techniques, particularly differential-like attacks, have been
actively explored recently. ReLU-based DNNs are the most commonly and widely
deployed architectures. While early works (e.g., Crypto 2020, Eurocrypt 2024)
assume access to exact output logits, which are usually invisible, more recent
works (e.g., Asiacrypt 2024, Eurocrypt 2025) focus on the hard-label setting,
where only the final classification result (e.g., "dog" or "car") is available
to the attacker. Notably, Carlini et al. (Eurocrypt 2025) demonstrated that
model extraction is feasible in polynomial time even under this restricted
setting.
  In this paper, we first show that the assumptions underlying their attack
become increasingly unrealistic as the attack-target depth grows. In practice,
satisfying these assumptions requires an exponential number of queries with
respect to the attack depth, implying that the attack does not always run in
polynomial time. To address this critical limitation, we propose a novel attack
method called CrossLayer Extraction. Instead of directly extracting the secret
parameters (e.g., weights and biases) of a specific neuron, which incurs
exponential cost, we exploit neuron interactions across layers to extract this
information from deeper layers. This technique significantly reduces query
complexity and mitigates the limitations of existing model extraction
approaches.

### 10. [Representation Gap of the Motzkin Monoid](http://arxiv.org/pdf/2510.06707v1)

Authors: Katharina Arms

The linear decomposition attack reveals a vulnerability in encryption
algorithms operating within groups or monoids with excessively small
representations. The representation gap, defined as the size of the smallest
non-trivial representation, therefore serves as a metric to assess the security
of these algorithms. This paper will demonstrate that the diagrammatic Motzkin
monoids exhibit a large representation gap, positioning them as promising
candidates for robust encryption algorithms.

### Computer Vision and Pattern Recognition

### 1. [VUGEN: Visual Understanding priors for GENeration](http://arxiv.org/pdf/2510.06529v1)

Authors: Xiangyi Chen, Théophane Vallaeys, Maha Elbayad, John Nguyen, Jakob Verbeek

Recent advances in Vision-Language Models (VLMs) have enabled unified
understanding across text and images, yet equipping these models with robust
image generation capabilities remains challenging. Existing approaches often
rely on reconstruction-oriented autoencoders or complex bridging mechanisms,
leading to misalignment between understanding and generation representations,
or architectural complexity. In this work, we propose VUGEN, a novel framework
that explicitly leverages VLM's pretrained visual understanding priors for
efficient and high-quality image generation. Our approach first transforms the
high-dimensional latent space of the VLM's native vision encoder into a
lower-dimensional, tractable distribution that maximally preserves visual
information. The VLM is then trained to sample within this reduced latent
space, ensuring alignment with its visual understanding capabilities. Finally,
a dedicated pixel decoder maps these generated latents back to the image space.
We find that a VAE-free pixel diffusion decoder to be on par or better than
commonly used complex latent diffusion decoders that internally rely on VAE
latents. Extensive experiments demonstrate that VUGEN achieves superior image
generation performance, improving DPG Bench from 71.17 to 74.32 and FID from
11.86 to 9.06 on COCO, while fully preserving the VLM's original understanding
capabilities.

### 2. [Ming-UniVision: Joint Image Understanding and Generation with a Unified Continuous Tokenizer](http://arxiv.org/pdf/2510.06590v1)

Authors: Ziyuan Huang, DanDan Zheng, Cheng Zou, Rui Liu, Xiaolong Wang, Kaixiang Ji, Weilong Chai, Jianxin Sun, Libin Wang, Yongjie Lv, Taozhi Huang, Jiajia Liu, Qingpei Guo, Ming Yang, Jingdong Chen, Jun Zhou

Visual tokenization remains a core challenge in unifying visual understanding
and generation within the autoregressive paradigm. Existing methods typically
employ tokenizers in discrete latent spaces to align with the tokens from large
language models, where the quantization errors can limit semantic
expressiveness and degrade the capability of vision-language understanding. To
address this, we introduce MingTok, a new family of visual tokenizers with a
continuous latent space, for unified autoregressive generation and
understanding. While understanding tasks favor discriminative high-dimensional
features, generation tasks prefer compact low-level codes. Thus, to reconcile
these competing demands, MingTok adopts a three-stage sequential architecture
involving low-level encoding, semantic expansion, and visual reconstruction.
Built on top of it, Ming-UniVision eliminates the need for task-specific visual
representations, and unifies diverse vision-language tasks under a single
autoregrsssive prediction paradigm. By formulating both understanding and
generation as next-token prediction in a shared continuous space, it seamlessly
supports multi-round, in-context tasks such as iterative understanding,
generation and editing. Empirically, we find that using a unified continuous
visual representation reconciles the competing requirements on the tokenizers
by the understanding and generation tasks, thereby leading to state-of-the-art
level performance across both domains. We hope our findings will facilitate
unified visual tokenization in the continuous domain. Inference code and model
weights are released to benefit community.

### 3. [Adaptive Stain Normalization for Cross-Domain Medical Histology](http://arxiv.org/pdf/2510.06592v1)

Authors: Tianyue Xu, Yanlin Wu, Abhai K. Tripathi, Matthew M. Ippolito, Benjamin D. Haeffele

Deep learning advances have revolutionized automated digital pathology
analysis. However, differences in staining protocols and imaging conditions can
introduce significant color variability. In deep learning, such color
inconsistency often reduces performance when deploying models on data acquired
under different conditions from the training data, a challenge known as domain
shift. Many existing methods attempt to address this problem via color
normalization but suffer from several notable drawbacks such as introducing
artifacts or requiring careful choice of a template image for stain mapping. To
address these limitations, we propose a trainable color normalization model
that can be integrated with any backbone network for downstream tasks such as
object detection and classification. Based on the physics of the imaging
process per the Beer-Lambert law, our model architecture is derived via
algorithmic unrolling of a nonnegative matrix factorization (NMF) model to
extract stain-invariant structural information from the original pathology
images, which serves as input for further processing. Experimentally, we
evaluate the method on publicly available pathology datasets and an internally
curated collection of malaria blood smears for cross-domain object detection
and classification, where our method outperforms many state-of-the-art stain
normalization methods. Our code is available at
https://github.com/xutianyue/BeerLaNet.

### 4. [AIM 2025 Challenge on Real-World RAW Image Denoising](http://arxiv.org/pdf/2510.06601v1)

Authors: Feiran Li, Jiacheng Li, Marcos V. Conde, Beril Besbinar, Vlad Hosu, Daisuke Iso, Radu Timofte

We introduce the AIM 2025 Real-World RAW Image Denoising Challenge, aiming to
advance efficient and effective denoising techniques grounded in data
synthesis. The competition is built upon a newly established evaluation
benchmark featuring challenging low-light noisy images captured in the wild
using five different DSLR cameras. Participants are tasked with developing
novel noise synthesis pipelines, network architectures, and training
methodologies to achieve high performance across different camera models.
Winners are determined based on a combination of performance metrics, including
full-reference measures (PSNR, SSIM, LPIPS), and non-reference ones (ARNIQA,
TOPIQ). By pushing the boundaries of camera-agnostic low-light RAW image
denoising trained on synthetic data, the competition promotes the development
of robust and practical models aligned with the rapid progress in digital
photography. We expect the competition outcomes to influence multiple domains,
from image restoration to night-time autonomous driving.

### 5. [Self-supervised Physics-guided Model with Implicit Representation Regularization for Fast MRI Reconstruction](http://arxiv.org/pdf/2510.06611v1)

Authors: Jingran Xu, Yuanyuan Liu, Yanjie Zhu

Magnetic Resonance Imaging (MRI) is a vital clinical diagnostic tool, yet its
widespread application is limited by prolonged scan times. Fast MRI
reconstruction techniques effectively reduce acquisition duration by
reconstructing high-fidelity MR images from undersampled k-space data. In
recent years, deep learning-based methods have demonstrated remarkable progress
in this field, with self-supervised and unsupervised learning approaches
proving particularly valuable in scenarios where fully sampled data are
difficult to obtain. This paper proposes a novel zero-shot self-supervised
reconstruction framework named UnrollINR, which enables scan-specific MRI
reconstruction without relying on external training data. The method adopts a
physics-guided unrolled iterative reconstruction architecture and introduces
Implicit Neural Representation (INR) as a regularization prior to effectively
constrain the solution space. By combining a deep unrolled structure with the
powerful implicit representation capability of INR, the model's
interpretability and reconstruction performance are enhanced. Experimental
results demonstrate that even at a high acceleration rate of 10, UnrollINR
achieves superior reconstruction performance compared to the supervised
learning method, validating the superiority of the proposed method.

### 6. [A Bridge from Audio to Video: Phoneme-Viseme Alignment Allows Every Face to Speak Multiple Languages](http://arxiv.org/pdf/2510.06612v1)

Authors: Zibo Su, Kun Wei, Jiahua Li, Xu Yang, Cheng Deng

Speech-driven talking face synthesis (TFS) focuses on generating lifelike
facial animations from audio input. Current TFS models perform well in English
but unsatisfactorily in non-English languages, producing wrong mouth shapes and
rigid facial expressions. The terrible performance is caused by the
English-dominated training datasets and the lack of cross-language
generalization abilities. Thus, we propose Multilingual Experts (MuEx), a novel
framework featuring a Phoneme-Guided Mixture-of-Experts (PG-MoE) architecture
that employs phonemes and visemes as universal intermediaries to bridge audio
and video modalities, achieving lifelike multilingual TFS. To alleviate the
influence of linguistic differences and dataset bias, we extract audio and
video features as phonemes and visemes respectively, which are the basic units
of speech sounds and mouth movements. To address audiovisual synchronization
issues, we introduce the Phoneme-Viseme Alignment Mechanism (PV-Align), which
establishes robust cross-modal correspondences between phonemes and visemes. In
addition, we build a Multilingual Talking Face Benchmark (MTFB) comprising 12
diverse languages with 95.04 hours of high-quality videos for training and
evaluating multilingual TFS performance. Extensive experiments demonstrate that
MuEx achieves superior performance across all languages in MTFB and exhibits
effective zero-shot generalization to unseen languages without additional
training.

### 7. [MSITrack: A Challenging Benchmark for Multispectral Single Object Tracking](http://arxiv.org/pdf/2510.06619v1)

Authors: Tao Feng, Tingfa Xu, Haolin Qin, Tianhao Li, Shuaihao Han, Xuyang Zou, Zhan Lv, Jianan Li

Visual object tracking in real-world scenarios presents numerous challenges
including occlusion, interference from similar objects and complex
backgrounds-all of which limit the effectiveness of RGB-based trackers.
Multispectral imagery, which captures pixel-level spectral reflectance,
enhances target discriminability. However, the availability of multispectral
tracking datasets remains limited. To bridge this gap, we introduce MSITrack,
the largest and most diverse multispectral single object tracking dataset to
date. MSITrack offers the following key features: (i) More Challenging
Attributes-including interference from similar objects and similarity in color
and texture between targets and backgrounds in natural scenarios, along with a
wide range of real-world tracking challenges; (ii) Richer and More Natural
Scenes-spanning 55 object categories and 300 distinct natural scenes, MSITrack
far exceeds the scope of existing benchmarks. Many of these scenes and
categories are introduced to the multispectral tracking domain for the first
time; (iii) Larger Scale-300 videos comprising over 129k frames of
multispectral imagery. To ensure annotation precision, each frame has undergone
meticulous processing, manual labeling and multi-stage verification. Extensive
evaluations using representative trackers demonstrate that the multispectral
data in MSITrack significantly improves performance over RGB-only baselines,
highlighting its potential to drive future advancements in the field. The
MSITrack dataset is publicly available at:
https://github.com/Fengtao191/MSITrack.

### 8. [DreamOmni2: Multimodal Instruction-based Editing and Generation](http://arxiv.org/pdf/2510.06679v1)

Authors: Bin Xia, Bohao Peng, Yuechen Zhang, Junjia Huang, Jiyang Liu, Jingyao Li, Haoru Tan, Sitong Wu, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, Jiaya Jia

Recent advancements in instruction-based image editing and subject-driven
generation have garnered significant attention, yet both tasks still face
limitations in meeting practical user needs. Instruction-based editing relies
solely on language instructions, which often fail to capture specific editing
details, making reference images necessary. Meanwhile, subject-driven
generation is limited to combining concrete objects or people, overlooking
broader, abstract concepts. To address these challenges, we propose two novel
tasks: multimodal instruction-based editing and generation. These tasks support
both text and image instructions and extend the scope to include both concrete
and abstract concepts, greatly enhancing their practical applications. We
introduce DreamOmni2, tackling two primary challenges: data creation and model
framework design. Our data synthesis pipeline consists of three steps: (1)
using a feature mixing method to create extraction data for both abstract and
concrete concepts, (2) generating multimodal instruction-based editing training
data using the editing and extraction models, and (3) further applying the
extraction model to create training data for multimodal instruction-based
editing. For the framework, to handle multi-image input, we propose an index
encoding and position encoding shift scheme, which helps the model distinguish
images and avoid pixel confusion. Additionally, we introduce joint training
with the VLM and our generation/editing model to better process complex
instructions. In addition, we have proposed comprehensive benchmarks for these
two new tasks to drive their development. Experiments show that DreamOmni2 has
achieved impressive results. Models and codes will be released.

### 9. [SCas4D: Structural Cascaded Optimization for Boosting Persistent 4D Novel View Synthesis](http://arxiv.org/pdf/2510.06694v1)

Authors: Jipeng Lyu, Jiahua Dong, Yu-Xiong Wang

Persistent dynamic scene modeling for tracking and novel-view synthesis
remains challenging due to the difficulty of capturing accurate deformations
while maintaining computational efficiency. We propose SCas4D, a cascaded
optimization framework that leverages structural patterns in 3D Gaussian
Splatting for dynamic scenes. The key idea is that real-world deformations
often exhibit hierarchical patterns, where groups of Gaussians share similar
transformations. By progressively refining deformations from coarse part-level
to fine point-level, SCas4D achieves convergence within 100 iterations per time
frame and produces results comparable to existing methods with only
one-twentieth of the training iterations. The approach also demonstrates
effectiveness in self-supervised articulated object segmentation, novel view
synthesis, and dense point tracking tasks.

### 10. [DeRainMamba: A Frequency-Aware State Space Model with Detail Enhancement for Image Deraining](http://arxiv.org/pdf/2510.06746v1)

Authors: Zhiliang Zhu, Tao Zeng, Tao Yang, Guoliang Luo, Jiyong Zeng

Image deraining is crucial for improving visual quality and supporting
reliable downstream vision tasks. Although Mamba-based models provide efficient
sequence modeling, their limited ability to capture fine-grained details and
lack of frequency-domain awareness restrict further improvements. To address
these issues, we propose DeRainMamba, which integrates a Frequency-Aware
State-Space Module (FASSM) and Multi-Directional Perception Convolution
(MDPConv). FASSM leverages Fourier transform to distinguish rain streaks from
high-frequency image details, balancing rain removal and detail preservation.
MDPConv further restores local structures by capturing anisotropic gradient
features and efficiently fusing multiple convolution branches. Extensive
experiments on four public benchmarks demonstrate that DeRainMamba consistently
outperforms state-of-the-art methods in PSNR and SSIM, while requiring fewer
parameters and lower computational costs. These results validate the
effectiveness of combining frequency-domain modeling and spatial detail
enhancement within a state-space framework for single image deraining.

### Computers and Society

### 1. [Unpacking Discourses on Childbirth and Parenthood in Popular Social Media Platforms Across China, Japan, and South Korea](http://arxiv.org/pdf/2510.06788v1)

Authors: Zheng Wei, Yunqi Li, Yucheng He, Yuelu Li, Xian Xu, Huamin Qu, Pan Hui, Muzhi Zhou

Social media use has been shown to be associated with low fertility desires.
However, we know little about the discourses surrounding childbirth and
parenthood that people consume online. We analyze 219,127 comments on 668 short
videos related to reproduction and parenthood from Douyin and Tiktok in China,
South Korea, and Japan, a region famous for its extremely low fertility level,
to examine the topics and sentiment expressed online. BERTopic model is used to
assist thematic analysis, and a large language model QWen is applied to label
sentiment. We find that comments focus on childrearing costs in all countries,
utility of children, particularly in Japan and South Korea, and individualism,
primarily in China. Comments from Douyin exhibit the strongest anti-natalist
sentiments, while the Japanese and Korean comments are more neutral. Short
video characteristics, such as their stances or account type, significantly
influence the responses, alongside regional socioeconomic indicators, including
GDP, urbanization, and population sex ratio. This work provides one of the
first comprehensive analyses of online discourses on family formation via
popular algorithm-fed video sharing platforms in regions experiencing low
fertility rates, making a valuable contribution to our understanding of the
spread of family values online.

### 2. [Am I Productive? Exploring the Experience of Remote Workers with Task Management Tools](http://arxiv.org/pdf/2510.06816v1)

Authors: Russell Beale

As the world continues to change, more and more knowledge workers are
embracing remote work. Yet this comes with its challenges for their
productivity, and while many Task Management applications promise to improve
the productivity of remote workers, it remains unclear how effective they are.
Based on existing frameworks, this study investigated the productivity needs
and challenges of remote knowledge workers and how they use Task Management
tools. The research was conducted through a 2-week long, mixed-methods diary
study and semi-structured interview. Perceptions of productivity, task
management tool use and productivity challenges were observed. The findings
show that using a digital Task Management application made no significant
difference to using pen and paper for improving perceived productivity of
remote workers and discuss the need for better personalization of Task
Management applications.

### 3. [The Limits of Goal-Setting Theory in LLM-Driven Assessment](http://arxiv.org/pdf/2510.06997v1)

Authors: Mrityunjay Kumar

Many users interact with AI tools like ChatGPT using a mental model that
treats the system as human-like, which we call Model H. According to
goal-setting theory, increased specificity in goals should reduce performance
variance. If Model H holds, then prompting a chatbot with more detailed
instructions should lead to more consistent evaluation behavior.
  This paper tests that assumption through a controlled experiment in which
ChatGPT evaluated 29 student submissions using four prompts with increasing
specificity. We measured consistency using intra-rater reliability (Cohen's
Kappa) across repeated runs.
  Contrary to expectations, performance did not improve consistently with
increased prompt specificity, and performance variance remained largely
unchanged. These findings challenge the assumption that LLMs behave like human
evaluators and highlight the need for greater robustness and improved input
integration in future model development.

### 4. [Early Results from Teaching Modelling for Software Comprehension in New-Hire Onboarding](http://arxiv.org/pdf/2510.07010v1)

Authors: Mrityunjay Kumar, Venkatesh Choppella

Working effectively with large, existing software systems requires strong
comprehension skills, yet most graduates enter the industry with little
preparation for this challenge. We report early results from a pilot
intervention integrated into a SaaS company's onboarding program: a
five-session course introducing systems thinking and Labelled Transition System
(LTS) modelling. Participants articulated their understanding of product
behaviour using a structured template and completed matched pre- and
post-assessments. Of 35 new hires, 31 provided paired records for analysis.
Across the full cohort, gains were small and not statistically significant.
However, participants below the median on the pre-test improved by 15
percentage points on average (statistically significant), while those above the
median regressed slightly (not statistically significant). Course feedback
indicated high engagement and perceived applicability. These results suggest
that short, modelling-focused onboarding interventions can accelerate
comprehension for less-prepared new hires. At the same time, they point to the
need for differentiated pathways for stronger participants, and to the
potential for companies to adopt such interventions at scale as a low-cost
complement to existing onboarding.

### 5. [On the false election between regulation and innovation. Ideas for regulation through the responsible use of artificial intelligence in research and education.[Spanish version]](http://arxiv.org/pdf/2510.07268v1)

Authors: Pompeu Casanovas

This short essay is a reworking of the answers offered by the author at the
Debate Session of the AIHUB (CSIC) and EduCaixa Summer School, organized by
Marta Garcia-Matos and Lissette Lemus, and coordinated by Albert Sabater
(OEIAC, UG), with the participation of Vanina Martinez-Posse (IIIA-CSIC),
Eulalia Soler (Eurecat) and Pompeu Casanovas (IIIA-CSIC) on July 4th 2025.
Albert Sabater posed three questions: (1) How can regulatory frameworks
priori-tise the protection of fundamental rights (privacy, non-discrimination,
autonomy, etc.) in the development of AI, without falling into the false
dichotomy between regulation and innova-tion? (2) Given the risks of AI (bias,
mass surveillance, manipulation), what examples of regu-lations or policies
have demonstrated that it is possible to foster responsible innovation, putting
the public interest before profitability, without giving in to competitive
pressure from actors such as China or the US? (3) In a scenario where the US
prioritizes flexibility, what mecha-nisms could ensure that international
cooperation in AI does not become a race to the bottom in rights, but rather a
global standard of accountability? The article attempts to answer these three
questions and concludes with some reflections on the relevance of the answers
for education and research.

### 6. [Emotionally Vulnerable Subtype of Internet Gaming Disorder: Measuring and Exploring the Pathology of Problematic Generative AI Use](http://arxiv.org/pdf/2510.06908v1)

Authors: Haocan Sun, Di Wua, Weizi Liu, Guoming Yua, Mike Yao

Concerns over the potential over-pathologization of generative AI (GenAI) use
and the lack of conceptual clarity surrounding GenAI addiction call for
empirical tools and theoretical refinement. This study developed and validated
the PUGenAIS-9 (Problematic Use of Generative Artificial Intelligence Scale-9
items) and examined whether PUGenAIS reflects addiction-like patterns under the
Internet Gaming Disorder (IGD) framework. Using samples from China and the
United States (N = 1,508), we conducted confirmatory factor analysis and
identified a robust 31-item structure across nine IGD-based dimensions. We then
derived the PUGenAIS-9 by selecting the highest-loading items from each
dimension and validated its structure in an independent sample (N = 1,426).
Measurement invariance tests confirmed its stability across nationality and
gender. Person-centered (latent profile analysis) and variable-centered
(network analysis) approaches found that PUGenAIS matches the traits of the
emotionally vulnerable subtype of IGD, not the competence-based kind. These
results support using PUGenAIS-9 to identify problematic GenAI use and show the
need to rethink digital addiction with an ICD (infrastructures, content, and
device) model. This keeps addiction research responsive to new media while
avoiding over-pathologizing.

### 7. [Machines in the Crowd? Measuring the Footprint of Machine-Generated Text on Reddit](http://arxiv.org/pdf/2510.07226v1)

Authors: Lucio La Cava, Luca Maria Aiello, Andrea Tagarelli

Generative Artificial Intelligence is reshaping online communication by
enabling large-scale production of Machine-Generated Text (MGT) at low cost.
While its presence is rapidly growing across the Web, little is known about how
MGT integrates into social media environments. In this paper, we present the
first large-scale characterization of MGT on Reddit. Using a state-of-the-art
statistical method for detection of MGT, we analyze over two years of activity
(2022-2024) across 51 subreddits representative of Reddit's main community
types such as information seeking, social support, and discussion. We study the
concentration of MGT across communities and over time, and compared MGT to
human-authored text in terms of social signals it expresses and engagement it
receives. Our very conservative estimate of MGT prevalence indicates that
synthetic text is marginally present on Reddit, but it can reach peaks of up to
9% in some communities in some months. MGT is unevenly distributed across
communities, more prevalent in subreddits focused on technical knowledge and
social support, and often concentrated in the activity of a small fraction of
users. MGT also conveys distinct social signals of warmth and status giving
typical of language of AI assistants. Despite these stylistic differences, MGT
achieves engagement levels comparable than human-authored content and in a few
cases even higher, suggesting that AI-generated text is becoming an organic
component of online social discourse. This work offers the first perspective on
the MGT footprint on Reddit, paving the way for new investigations involving
platform governance, detection strategies, and community dynamics.

### Databases

### 1. [On the Expressiveness of Languages for Querying Property Graphs in Relational Databases](http://arxiv.org/pdf/2510.07062v1)

Authors: Hadar Rotschield, Liat Peterfreund

SQL/PGQ is the emerging ISO standard for querying property graphs defined as
views over relational data. We formalize its expressive power across three
fragments: the read-only core, the read-write extension, and an extended
variant with richer view definitions. Our results show that graph creation
plays a central role in determining the expressiveness. The read-only fragment
is strictly weaker than the read-write fragment, and the latter is still below
the complexity class NL. Extending view definitions with arbitrary arity
identifiers closes this gap: the extended fragment captures exactly NL. This
yields a strict hierarchy of SQL/PGQ fragments, whose union covers all NL
queries. On ordered structures the hierarchy collapses: once arity-2
identifiers are allowed, higher arities add no power, mirroring the classical
transitive-closure collapse and underscoring the central role of view
construction in property graph querying.

### 2. [Relational Database Distillation: From Structured Tables to Condensed Graph Data](http://arxiv.org/pdf/2510.06980v1)

Authors: Xinyi Gao, Jingxi Zhang, Lijian Chen, Tong Chen, Lizhen Cui, Hongzhi Yin

Relational databases (RDBs) underpin the majority of global data management
systems, where information is structured into multiple interdependent tables.
To effectively use the knowledge within RDBs for predictive tasks, recent
advances leverage graph representation learning to capture complex inter-table
relations as multi-hop dependencies. Despite achieving state-of-the-art
performance, these methods remain hindered by the prohibitive storage overhead
and excessive training time, due to the massive scale of the database and the
computational burden of intensive message passing across interconnected tables.
To alleviate these concerns, we propose and study the problem of Relational
Database Distillation (RDD). Specifically, we aim to distill large-scale RDBs
into compact heterogeneous graphs while retaining the predictive power (i.e.,
utility) required for training graph-based models. Multi-modal column
information is preserved through node features, and primary-foreign key
relations are encoded via heterogeneous edges, thereby maintaining both data
fidelity and relational structure. To ensure adaptability across diverse
downstream tasks without engaging the traditional, inefficient bi-level
distillation framework, we further design a kernel ridge regression-guided
objective with pseudo-labels, which produces quality features for the distilled
graph. Extensive experiments on multiple real-world RDBs demonstrate that our
solution substantially reduces the data size while maintaining competitive
performance on classification and regression tasks, creating an effective
pathway for scalable learning with RDBs.

### 3. [Automated Discovery of Test Oracles for Database Management Systems Using LLMs](http://arxiv.org/pdf/2510.06663v1)

Authors: Qiuyang Mang, Runyuan He, Suyang Zhong, Xiaoxuan Liu, Huanchen Zhang, Alvin Cheung

Since 2020, automated testing for Database Management Systems (DBMSs) has
flourished, uncovering hundreds of bugs in widely-used systems. A cornerstone
of these techniques is test oracle, which typically implements a mechanism to
generate equivalent query pairs, thereby identifying bugs by checking the
consistency between their results. However, while applying these oracles can be
automated, their design remains a fundamentally manual endeavor. This paper
explores the use of large language models (LLMs) to automate the discovery and
instantiation of test oracles, addressing a long-standing bottleneck towards
fully automated DBMS testing. Although LLMs demonstrate impressive creativity,
they are prone to hallucinations that can produce numerous false positive bug
reports. Furthermore, their significant monetary cost and latency mean that LLM
invocations should be limited to ensure that bug detection is efficient and
economical.
  To this end, we introduce Argus, a novel framework built upon the core
concept of the Constrained Abstract Query - a SQL skeleton containing
placeholders and their associated instantiation conditions (e.g., requiring a
placeholder to be filled by a boolean column). Argus uses LLMs to generate
pairs of these skeletons that are asserted to be semantically equivalent. This
equivalence is then formally proven using a SQL equivalence solver to ensure
soundness. Finally, the placeholders within the verified skeletons are
instantiated with concrete, reusable SQL snippets that are also synthesized by
LLMs to efficiently produce complex test cases. We implemented Argus and
evaluated it on five extensively tested DBMSs, discovering 40 previously
unknown bugs, 35 of which are logic bugs, with 36 confirmed and 26 already
fixed by the developers.

### Distributed, Parallel, and Cluster Computing

### 1. [REACH: Reinforcement Learning for Adaptive Microservice Rescheduling in the Cloud-Edge Continuum](http://arxiv.org/pdf/2510.06675v1)

Authors: Xu Bai, Muhammed Tawfiqul Islam, Rajkumar Buyya, Adel N. Toosi

Cloud computing, despite its advantages in scalability, may not always fully
satisfy the low-latency demands of emerging latency-sensitive pervasive
applications. The cloud-edge continuum addresses this by integrating the
responsiveness of edge resources with cloud scalability. Microservice
Architecture (MSA) characterized by modular, loosely coupled services, aligns
effectively with this continuum. However, the heterogeneous and dynamic
computing resource poses significant challenges to the optimal placement of
microservices. We propose REACH, a novel rescheduling algorithm that
dynamically adapts microservice placement in real time using reinforcement
learning to react to fluctuating resource availability, and performance
variations across distributed infrastructures. Extensive experiments on a
real-world testbed demonstrate that REACH reduces average end-to-end latency by
7.9%, 10%, and 8% across three benchmark MSA applications, while effectively
mitigating latency fluctuations and spikes.

### 2. [GROMACS Unplugged: How Power Capping and Frequency Shapes Performance on GPUs](http://arxiv.org/pdf/2510.06902v1)

Authors: Ayesha Afzal, Anna Kahler, Georg Hager, Gerhard Wellein

Molecular dynamics simulations are essential tools in computational
biophysics, but their performance depend heavily on hardware choices and
configuration. In this work, we presents a comprehensive performance analysis
of four NVIDIA GPU accelerators -- A40, A100, L4, and L40 -- using six
representative GROMACS biomolecular workloads alongside two synthetic
benchmarks: Pi Solver (compute bound) and STREAM Triad (memory bound). We
investigate how performance scales with GPU graphics clock frequency and how
workloads respond to power capping. The two synthetic benchmarks define the
extremes of frequency scaling: Pi Solver shows ideal compute scalability, while
STREAM Triad reveals memory bandwidth limits -- framing GROMACS's performance
in context. Our results reveal distinct frequency scaling behaviors: Smaller
GROMACS systems exhibit strong frequency sensitivity, while larger systems
saturate quickly, becoming increasingly memory bound. Under power capping,
performance remains stable until architecture- and workload-specific thresholds
are reached, with high-end GPUs like the A100 maintaining near-maximum
performance even under reduced power budgets. Our findings provide practical
guidance for selecting GPU hardware and optimizing GROMACS performance for
large-scale MD workflows under power constraints.

### 3. [Evaluating Rapid Makespan Predictions for Heterogeneous Systems with Programmable Logic](http://arxiv.org/pdf/2510.06998v1)

Authors: Martin Wilhelm, Franz Freitag, Max Tzschoppe, Thilo Pionteck

Heterogeneous computing systems, which combine general-purpose processors
with specialized accelerators, are increasingly important for optimizing the
performance of modern applications. A central challenge is to decide which
parts of an application should be executed on which accelerator or, more
generally, how to map the tasks of an application to available devices.
Predicting the impact of a change in a task mapping on the overall makespan is
non-trivial. While there are very capable simulators, these generally require a
full implementation of the tasks in question, which is particularly
time-intensive for programmable logic. A promising alternative is to use a
purely analytical function, which allows for very fast predictions, but
abstracts significantly from reality. Bridging the gap between theory and
practice poses a significant challenge to algorithm developers. This paper aims
to aid in the development of rapid makespan prediction algorithms by providing
a highly flexible evaluation framework for heterogeneous systems consisting of
CPUs, GPUs and FPGAs, which is capable of collecting real-world makespan
results based on abstract task graph descriptions. We analyze to what extent
actual makespans can be predicted by existing analytical approaches.
Furthermore, we present common challenges that arise from high-level
characteristics such as data transfer overhead and device congestion in
heterogeneous systems.

### 4. [Validation of Various Normalization Methods for Brain Tumor Segmentation: Can Federated Learning Overcome This Heterogeneity?](http://arxiv.org/pdf/2510.07126v1)

Authors: Jan Fiszer, Dominika Ciupek, Maciej Malawski

Deep learning (DL) has been increasingly applied in medical imaging, however,
it requires large amounts of data, which raises many challenges related to data
privacy, storage, and transfer. Federated learning (FL) is a training paradigm
that overcomes these issues, though its effectiveness may be reduced when
dealing with non-independent and identically distributed (non-IID) data. This
study simulates non-IID conditions by applying different MRI intensity
normalization techniques to separate data subsets, reflecting a common cause of
heterogeneity. These subsets are then used for training and testing models for
brain tumor segmentation. The findings provide insights into the influence of
the MRI intensity normalization methods on segmentation models, both training
and inference. Notably, the FL methods demonstrated resilience to
inconsistently normalized data across clients, achieving the 3D Dice score of
92%, which is comparable to a centralized model (trained using all data). These
results indicate that FL is a solution to effectively train high-performing
models without violating data privacy, a crucial concern in medical
applications. The code is available at:
https://github.com/SanoScience/fl-varying-normalization.

### 5. [Vectorized FlashAttention with Low-cost Exponential Computation in RISC-V Vector Processors](http://arxiv.org/pdf/2510.06834v1)

Authors: Vasileios Titopoulos, Kosmas Alexandridis, Giorgos Dimitrakopoulos

Attention is a core operation in numerous machine learning and artificial
intelligence models. This work focuses on the acceleration of attention kernel
using FlashAttention algorithm, in vector processors, particularly those based
on the RISC-V instruction set architecture (ISA). This work represents the
first effort to vectorize FlashAttention, minimizing scalar code and
simplifying the computational complexity of evaluating exponentials needed by
softmax used in attention. By utilizing a low-cost approximation for
exponentials in floating-point arithmetic, we reduce the cost of computing the
exponential function without the need to extend baseline vector ISA with new
custom instructions. Also, appropriate tiling strategies are explored with the
goal to improve memory locality. Experimental results highlight the scalability
of our approach, demonstrating significant performance gains with the
vectorized implementations when processing attention layers in practical
applications.

### 6. [Multi-Dimensional Autoscaling of Stream Processing Services on Edge Devices](http://arxiv.org/pdf/2510.06882v1)

Authors: Boris Sedlak, Philipp Raith, Andrea Morichetta, Víctor Casamayor Pujol, Schahram Dustdar

Edge devices have limited resources, which inevitably leads to situations
where stream processing services cannot satisfy their needs. While existing
autoscaling mechanisms focus entirely on resource scaling, Edge devices require
alternative ways to sustain the Service Level Objectives (SLOs) of competing
services. To address these issues, we introduce a Multi-dimensional Autoscaling
Platform (MUDAP) that supports fine-grained vertical scaling across both
service- and resource-level dimensions. MUDAP supports service-specific scaling
tailored to available parameters, e.g., scale data quality or model size for a
particular service. To optimize the execution across services, we present a
scaling agent based on Regression Analysis of Structural Knowledge (RASK). The
RASK agent efficiently explores the solution space and learns a continuous
regression model of the processing environment for inferring optimal scaling
actions. We compared our approach with two autoscalers, the Kubernetes VPA and
a reinforcement learning agent, for scaling up to 9 services on a single Edge
device. Our results showed that RASK can infer an accurate regression model in
merely 20 iterations (i.e., observe 200s of processing). By increasingly adding
elasticity dimensions, RASK sustained the highest request load with 28% less
SLO violations, compared to baselines.

### 7. [DPMM-CFL: Clustered Federated Learning via Dirichlet Process Mixture Model Nonparametric Clustering](http://arxiv.org/pdf/2510.07132v1)

Authors: Mariona Jaramillo-Civill, Peng Wu, Pau Closas

Clustered Federated Learning (CFL) improves performance under non-IID client
heterogeneity by clustering clients and training one model per cluster, thereby
balancing between a global model and fully personalized models. However, most
CFL methods require the number of clusters K to be fixed a priori, which is
impractical when the latent structure is unknown. We propose DPMM-CFL, a CFL
algorithm that places a Dirichlet Process (DP) prior over the distribution of
cluster parameters. This enables nonparametric Bayesian inference to jointly
infer both the number of clusters and client assignments, while optimizing
per-cluster federated objectives. This results in a method where, at each
round, federated updates and cluster inferences are coupled, as presented in
this paper. The algorithm is validated on benchmark datasets under Dirichlet
and class-split non-IID partitions.

### Discrete Mathematics

### 1. [A Computer-Assisted Proof of the Optimal Density Bound for Pinwheel Covering](http://arxiv.org/pdf/2510.06533v1)

Authors: Akitoshi Kawamura, Yusuke Kobayashi

In the covering version of the pinwheel scheduling problem, a daily task must
be assigned to agents under the constraint that agent $i$ can perform the task
at most once in any $a_i$-day interval. In this paper, we determine the optimal
constant $\alpha^* = 1.264\ldots {}$ such that every instance with $\sum_{i}
\frac{1}{a_i} \ge \alpha^*$ is schedulable. This resolves an open problem posed
by Soejima and Kawamura (2020). Our proof combines Kawamura's (2024) techniques
for the packing version with new mathematical insights, along with an
exhaustive computer-aided search that draws on some ideas from G\k{a}sieniec,
Smith, and Wild (2022).

### 2. [On the distribution of $A_α$-eigenvalues in terms of graph invariants](http://arxiv.org/pdf/2510.06933v1)

Authors: Uilton Cesar Peres Junior, Carla Silva Oliveira, André Ebling Brondan

Let $G$ be a connected graph of order $n$, and $A(G)$ and $D(G)$ its
adjacency and degree diagonal matrices, respectively. For a parameter $\alpha
\in [0,1]$, Nikiforov~(2017) introduced the convex combination $A_{\alpha}(G) =
\alpha D(G) + (1 - \alpha)A(G)$. In this paper, we investigate the spectral
distribution of $A_\alpha(G)$-eigenvalues, over subintervals of the real line.
We establish lower and upper bounds on the number of such eigenvalues in terms
of structural parameters of $G$, including the number of pendant and
quasi-pendant vertices, the domination number, the matching number, and the
edge covering number. Additionally, we exhibit families of graphs for which
these bounds are attained. Several of our results extend known spectral bounds
on the eigenvalue distributions of both the adjacency and the signless
Laplacian matrices.

### 3. [Parameterized Complexity of s-Club Cluster Edge Deletion](http://arxiv.org/pdf/2510.07065v1)

Authors: Ajinkya Gaikwad

We study the parameterized and classical complexity of the s-Club Cluster
Edge Deletion problem: given a graph G = (V, E) and integers k and s, determine
whether it is possible to delete at most k edges so that every connected
component of the resulting graph has diameter at most s. This problem
generalizes Cluster Edge Deletion (the case s = 1) and captures a variety of
distance-bounded graph modification tasks.
  Montecchiani, Ortali, Piselli, and Tappini (Information and Computation,
2023) showed that the problem is fixed-parameter tractable when parameterized
by s plus the treewidth of G, and asked whether the dependence on s is
necessary; that is, whether the problem is FPT when parameterized by treewidth
alone. We resolve this by proving that the problem is W[1]-hard when
parameterized by pathwidth, and hence by treewidth.
  On the algorithmic side, we show that the problem is FPT when parameterized
by neighborhood diversity, twin cover, or cluster vertex deletion number,
thereby extending to all s >= 1 the results of Italiano, Konstantinidis, and
Papadopoulos (Algorithmica, 2023), who established FPT algorithms for the case
s = 1 under the neighborhood diversity and twin cover parameters.
  From a classical perspective, we prove that the problem is NP-hard on split
graphs already for s = 2, complementing the polynomial-time solvability for s =
1 due to Bonomo, Duran, and Valencia-Pabon (Theoretical Computer Science, 2015)
and the trivial case s = 3.
  Finally, while the problem is FPT when parameterized by s + k, its complexity
for the solution size k alone remains open. We make progress on this front by
designing an FPT bicriteria approximation algorithm, which runs in time f(k,
1/epsilon) * n^{O(1)} and, for graphs excluding long induced cycles, outputs a
solution of size at most k whose connected components have diameter at most (1
+ epsilon) * s.

### 4. [On some 2-binomial coefficients of binary words: geometrical interpretation, partitions of integers, and fair words](http://arxiv.org/pdf/2510.07159v1)

Authors: Gwenaël Richomme

The binomial notation (w u) represents the number of occurrences of the word
u as a (scattered) subword in w. We first introduce and study possible uses of
a geometrical interpretation of (w ab) and (w ba) when a and b are distinct
letters. We then study the structure of the 2-binomial equivalence class of a
binary word w (two words are 2-binomially equivalent if they have the same
binomial coefficients, that is, the same numbers of occurrences, for each word
of length at most 2). Especially we prove the existence of an isomorphism
between the graph of the 2-binomial equivalence class of w with respect to a
particular rewriting rule and the lattice of partitions of the integer (w ab)
with (w a) parts and greatest part bounded by (w b). Finally we study binary
fair words, the words over {a, b} having the same numbers of occurrences of ab
and ba as subwords ((w ab) = (w ba)). In particular, we prove a recent
conjecture related to a special case of the least square approximation.

### 5. [Extending Ghouila-Houri's Characterization of Comparability Graphs to Temporal Graphs](http://arxiv.org/pdf/2510.06849v1)

Authors: Pierre Charbit, Michel Habib, Amalia Sorondo

An orientation of a given static graph is called transitive if for any three
vertices $a,b,c$, the presence of arcs $(a,b)$ and $(b,c)$ forces the presence
of the arc $(a,c)$. If only the presence of an arc between $a$ and $c$ is
required, but its orientation is unconstrained, the orientation is called
quasi-transitive. A fundamental result presented by Ghouila-Houri guarantees
that any static graph admitting a quasi-transitive orientation also admits a
transitive orientation. In a seminal work, Mertzios et al. introduced the
notion of temporal transitivity in order to model information flows in simple
temporal networks. We revisit the model introduced by Mertzios et al. and
propose an analogous to Ghouila-Houri's characterization for the temporal
scenario. We present a structure theorem that will allow us to express by a
2-SAT formula all the constraints imposed by temporal transitive orientations.
The latter produces an efficient recognition algorithm for graphs admitting
such orientations. Additionally, we extend the temporal transitivity model to
temporal graphs having multiple time-labels associated to their edges and claim
that the previous results hold in the multilabel setting. Finally, we propose a
characterization of temporal comparability graphs via forbidden temporal
ordered patterns.

### Data Structures and Algorithms

### 1. [Breaking the Treewidth Barrier in Quantum Circuit Simulation with Decision Diagrams](http://arxiv.org/pdf/2510.06775v1)

Authors: Bin Cheng, Ziyuan Wang, Ruixuan Deng, Jianxin Chen, Zhengfeng Ji

Classical simulation of quantum circuits is a critical tool for validating
quantum hardware and probing the boundary between classical and quantum
computational power. Existing state-of-the-art methods, notably tensor network
approaches, have computational costs governed by the treewidth of the
underlying circuit graph, making circuits with large treewidth intractable.
This work rigorously analyzes FeynmanDD, a decision diagram-based simulation
method proposed in CAV 2025 by a subset of the authors, and shows that the size
of the multi-terminal decision diagram used in FeynmanDD is exponential in the
linear rank-width of the circuit graph. As linear rank-width can be
substantially smaller than treewidth and is at most larger than the treewidth
by a logarithmic factor, our analysis demonstrates that FeynmanDD outperforms
all tensor network-based methods for certain circuit families. We also show
that the method remains efficient if we use the Solovay-Kitaev algorithm to
expand arbitrary single-qubit gates to sequences of Hadamard and T gates,
essentially removing the gate-set restriction posed by the method.

### 2. [Randomized Quantum Singular Value Transformation](http://arxiv.org/pdf/2510.06851v1)

Authors: Xinzhao Wang, Yuxin Zhang, Soumyabrata Hazra, Tongyang Li, Changpeng Shao, Shantanav Chakraborty

We introduce the first randomized algorithms for Quantum Singular Value
Transformation (QSVT), a unifying framework for many quantum algorithms.
Standard implementations of QSVT rely on block encodings of the Hamiltonian,
which are costly to construct, requiring a logarithmic number of ancilla
qubits, intricate multi-qubit control, and circuit depth scaling linearly with
the number of Hamiltonian terms. In contrast, our algorithms use only a single
ancilla qubit and entirely avoid block encodings. We develop two methods: (i) a
direct randomization of QSVT, where block encodings are replaced by importance
sampling, and (ii) an approach that integrates qDRIFT into the generalized
quantum signal processing framework, with the dependence on precision
exponentially improved through classical extrapolation. Both algorithms achieve
gate complexity independent of the number of Hamiltonian terms, a hallmark of
randomized methods, while incurring only quadratic dependence on the degree of
the target polynomial. We identify natural parameter regimes where our methods
outperform even standard QSVT, making them promising for early fault-tolerant
quantum devices. We also establish a fundamental lower bound showing that the
quadratic dependence on the polynomial degree is optimal within this framework.
We apply our framework to two fundamental tasks: solving quantum linear systems
and estimating ground-state properties of Hamiltonians, obtaining polynomial
advantages over prior randomized algorithms. Finally, we benchmark our
ground-state property estimation algorithm on electronic structure Hamiltonians
and the transverse-field Ising model with long-range interactions. In both
cases, our approach outperforms prior work by several orders of magnitude in
circuit depth, establishing randomized QSVT as a practical and
resource-efficient alternative for early fault-tolerant quantum devices.

### 3. [Parameterized Complexity of s-Club Cluster Edge Deletion](http://arxiv.org/pdf/2510.07065v1)

Authors: Ajinkya Gaikwad

We study the parameterized and classical complexity of the s-Club Cluster
Edge Deletion problem: given a graph G = (V, E) and integers k and s, determine
whether it is possible to delete at most k edges so that every connected
component of the resulting graph has diameter at most s. This problem
generalizes Cluster Edge Deletion (the case s = 1) and captures a variety of
distance-bounded graph modification tasks.
  Montecchiani, Ortali, Piselli, and Tappini (Information and Computation,
2023) showed that the problem is fixed-parameter tractable when parameterized
by s plus the treewidth of G, and asked whether the dependence on s is
necessary; that is, whether the problem is FPT when parameterized by treewidth
alone. We resolve this by proving that the problem is W[1]-hard when
parameterized by pathwidth, and hence by treewidth.
  On the algorithmic side, we show that the problem is FPT when parameterized
by neighborhood diversity, twin cover, or cluster vertex deletion number,
thereby extending to all s >= 1 the results of Italiano, Konstantinidis, and
Papadopoulos (Algorithmica, 2023), who established FPT algorithms for the case
s = 1 under the neighborhood diversity and twin cover parameters.
  From a classical perspective, we prove that the problem is NP-hard on split
graphs already for s = 2, complementing the polynomial-time solvability for s =
1 due to Bonomo, Duran, and Valencia-Pabon (Theoretical Computer Science, 2015)
and the trivial case s = 3.
  Finally, while the problem is FPT when parameterized by s + k, its complexity
for the solution size k alone remains open. We make progress on this front by
designing an FPT bicriteria approximation algorithm, which runs in time f(k,
1/epsilon) * n^{O(1)} and, for graphs excluding long induced cycles, outputs a
solution of size at most k whose connected components have diameter at most (1
+ epsilon) * s.

### 4. [Trickle-down Theorems via C-Lorentzian Polynomials II: Pairwise Spectral Influence and Improved Dobrushin's Condition](http://arxiv.org/pdf/2510.06549v1)

Authors: Jonathan Leake, Shayan Oveis Gharan

Let $\mu$ be a probability distribution on a multi-state spin system on a set
$V$ of sites. Equivalently, we can think of this as a $d$-partite simplical
complex with distribution $\mu$ on maximal faces. For any pair of vertices
$u,v\in V$, define the pairwise spectral influence $\mathcal{I}_{u,v}$ as
follows. Let $\sigma$ be a choice of spins $s_w\in S_w$ for every $w\in V
\setminus \{u,v\}$, and construct a matrix in $\mathbb{R}^{(S_u\cup S_v)\times
(S_u\cup S_v)}$ where for any $s_u\in S_u, s_v\in S_v$, the $(us_u,vs_v)$-entry
is the probability that $s_v$ is the spin of $v$ conditioned on $s_u$ being the
spin of $u$ and on $\sigma$. Then $\mathcal{I}_{u,v}$ is the maximal second
eigenvalue of this matrix, over all choices of spins for all $w \in V \setminus
\{u,v\}$. Equivalently, $\mathcal{I}_{u,v}$ is the maximum local spectral
expansion of links of codimension $2$ that include a spin for every $w \in V
\setminus \{u,v\}$.
  We show that if the largest eigenvalue of the pairwise spectral influence
matrix with entries $\mathcal{I}_{u,v}$ is bounded away from 1, i.e.
$\lambda_{\max}(\mathcal{I})\leq 1-\epsilon$ (and $X$ is connected), then the
Glauber dynamics mixes rapidly and generate samples from $\mu$. This
improves/generalizes the classical Dobrushin's influence matrix as the
$\mathcal{I}_{u,v}$ lower-bounds the classical influence of $u\to v$. As a
by-product, we also prove improved/almost optimal trickle-down theorems for
partite simplicial complexes. The proof builds on the trickle-down theorems via
$\mathcal{C}$-Lorentzian polynomials machinery recently developed by the
authors and Lindberg.

### 5. [Reconquering Bell sampling on qudits: stabilizer learning and testing, quantum pseudorandomness bounds, and more](http://arxiv.org/pdf/2510.06848v1)

Authors: Jonathan Allcock, Joao F. Doriguello, Gábor Ivanyos, Miklos Santha

Bell sampling is a simple yet powerful tool based on measuring two copies of
a quantum state in the Bell basis, and has found applications in a plethora of
problems related to stabiliser states and measures of magic. However, it was
not known how to generalise the procedure from qubits to $d$-level systems --
qudits -- for all dimensions $d > 2$ in a useful way. Indeed, a prior work of
the authors (arXiv'24) showed that the natural extension of Bell sampling to
arbitrary dimensions fails to provide meaningful information about the quantum
states being measured. In this paper, we overcome the difficulties encountered
in previous works and develop a useful generalisation of Bell sampling to
qudits of all $d\geq 2$. At the heart of our primitive is a new unitary, based
on Lagrange's four-square theorem, that maps four copies of any stabiliser
state $|\mathcal{S}\rangle$ to four copies of its complex conjugate
$|\mathcal{S}^\ast\rangle$ (up to some Pauli operator), which may be of
independent interest. We then demonstrate the utility of our new Bell sampling
technique by lifting several known results from qubits to qudits for any $d\geq
2$:
  1. Learning stabiliser states in $O(n^3)$ time with $O(n)$ samples;
  2. Solving the Hidden Stabiliser Group Problem in
$\tilde{O}(n^3/\varepsilon)$ time with $\tilde{O}(n/\varepsilon)$ samples;
  3. Testing whether $|\psi\rangle$ has stabiliser size at least $d^t$ or is
$\varepsilon$-far from all such states in $\tilde{O}(n^3/\varepsilon)$ time
with $\tilde{O}(n/\varepsilon)$ samples;
  4. Clifford circuits with at most $n/2$ single-qudit non-Clifford gates
cannot prepare pseudorandom states;
  5. Testing whether $|\psi\rangle$ has stabiliser fidelity at least
$1-\varepsilon_1$ or at most $1-\varepsilon_2$ with $O(d^2/\varepsilon_2)$
samples if $\varepsilon_1 = 0$ or $O(d^2/\varepsilon_2^2)$ samples if
$\varepsilon_1 = O(d^{-2})$.

### 6. [Extending Ghouila-Houri's Characterization of Comparability Graphs to Temporal Graphs](http://arxiv.org/pdf/2510.06849v1)

Authors: Pierre Charbit, Michel Habib, Amalia Sorondo

An orientation of a given static graph is called transitive if for any three
vertices $a,b,c$, the presence of arcs $(a,b)$ and $(b,c)$ forces the presence
of the arc $(a,c)$. If only the presence of an arc between $a$ and $c$ is
required, but its orientation is unconstrained, the orientation is called
quasi-transitive. A fundamental result presented by Ghouila-Houri guarantees
that any static graph admitting a quasi-transitive orientation also admits a
transitive orientation. In a seminal work, Mertzios et al. introduced the
notion of temporal transitivity in order to model information flows in simple
temporal networks. We revisit the model introduced by Mertzios et al. and
propose an analogous to Ghouila-Houri's characterization for the temporal
scenario. We present a structure theorem that will allow us to express by a
2-SAT formula all the constraints imposed by temporal transitive orientations.
The latter produces an efficient recognition algorithm for graphs admitting
such orientations. Additionally, we extend the temporal transitivity model to
temporal graphs having multiple time-labels associated to their edges and claim
that the previous results hold in the multilabel setting. Finally, we propose a
characterization of temporal comparability graphs via forbidden temporal
ordered patterns.

### 7. [Quantum Sparse Recovery and Quantum Orthogonal Matching Pursuit](http://arxiv.org/pdf/2510.06925v1)

Authors: Armando Bellante, Stefano Vanerio, Stefano Zanero

We study quantum sparse recovery in non-orthogonal, overcomplete
dictionaries: given coherent quantum access to a state and a dictionary of
vectors, the goal is to reconstruct the state up to $\ell_2$ error using as few
vectors as possible. We first show that the general recovery problem is
NP-hard, ruling out efficient exact algorithms in full generality. To overcome
this, we introduce Quantum Orthogonal Matching Pursuit (QOMP), the first
quantum analogue of the classical OMP greedy algorithm. QOMP combines quantum
subroutines for inner product estimation, maximum finding, and block-encoded
projections with an error-resetting design that avoids iteration-to-iteration
error accumulation. Under standard mutual incoherence and well-conditioned
sparsity assumptions, QOMP provably recovers the exact support of a $K$-sparse
state in polynomial time. As an application, we give the first framework for
sparse quantum tomography with non-orthogonal dictionaries in $\ell_2$ norm,
achieving query complexity $\widetilde{O}(\sqrt{N}/\epsilon)$ in favorable
regimes and reducing tomography to estimating only $K$ coefficients instead of
$N$ amplitudes. In particular, for pure-state tomography with $m=O(N)$
dictionary vectors and sparsity $K=\widetilde{O}(1)$ on a well-conditioned
subdictionary, this circumvents the $\widetilde{\Omega}(N/\epsilon)$ lower
bound that holds in the dense, orthonormal-dictionary setting, without
contradiction, by leveraging sparsity together with non-orthogonality. Beyond
tomography, we analyze QOMP in the QRAM model, where it yields polynomial
speedups over classical OMP implementations, and provide a quantum algorithm to
estimate the mutual incoherence of a dictionary of $m$ vectors in
$O(m/\epsilon)$ queries, improving over both deterministic and quantum-inspired
classical methods.

### 8. [Clifford testing: algorithms and lower bounds](http://arxiv.org/pdf/2510.07164v1)

Authors: Marcel Hinsche, Zongbo Bao, Philippe van Dordrecht, Jens Eisert, Jop Briët, Jonas Helsen

We consider the problem of Clifford testing, which asks whether a black-box
$n$-qubit unitary is a Clifford unitary or at least $\varepsilon$-far from
every Clifford unitary. We give the first 4-query Clifford tester, which
decides this problem with probability $\mathrm{poly}(\varepsilon)$. This
contrasts with the minimum of 6 copies required for the closely-related task of
stabilizer testing. We show that our tester is tolerant, by adapting techniques
from tolerant stabilizer testing to our setting. In doing so, we settle in the
positive a conjecture of Bu, Gu and Jaffe, by proving a polynomial inverse
theorem for a non-commutative Gowers 3-uniformity norm. We also consider the
restricted setting of single-copy access, where we give an $O(n)$-query
Clifford tester that requires no auxiliary memory qubits or adaptivity. We
complement this with a lower bound, proving that any such, potentially
adaptive, single-copy algorithm needs at least $\Omega(n^{1/4})$ queries. To
obtain our results, we leverage the structure of the commutant of the Clifford
group, obtaining several technical statements that may be of independent
interest.

### 9. [On quantum to classical comparison for Davies generators](http://arxiv.org/pdf/2510.07267v1)

Authors: Joao Basso, Shirshendu Ganguly, Alistair Sinclair, Nikhil Srivastava, Zachary Stier, Thuy-Duong Vuong

Despite extensive study, our understanding of quantum Markov chains remains
far less complete than that of their classical counterparts. [Temme'13]
observed that the Davies Lindbladian, a well-studied model of quantum Markov
dynamics, contains an embedded classical Markov generator, raising the natural
question of how the convergence properties of the quantum and classical
dynamics are related. While [Temme'13] showed that the spectral gap of the
Davies Lindbladian can be much smaller than that of the embedded classical
generator for certain highly structured Hamiltonians, we show that if the
spectrum of the Hamiltonian does not contain long arithmetic progressions, then
the two spectral gaps must be comparable. As a consequence, we prove that for a
large class of Hamiltonians, including those obtained by perturbing a fixed
Hamiltonian with a generic external field, the quantum spectral gap remains
within a constant factor of the classical spectral gap. Our result aligns with
physical intuition and enables the application of classical Markov chain
techniques to the quantum setting.
  The proof is based on showing that any ``off-diagonal'' eigenvector of the
Davies generator can be used to construct an observable which commutes with the
Hamiltonian and has a Lindbladian Rayleigh quotient which can be upper bounded
in terms of that of the original eigenvector's Lindbladian Rayleigh quotient.
Thus, a spectral gap for such observables implies a spectral gap for the full
Davies generator.

### Emerging Technologies

### 1. [An HPC-Inspired Blueprint for a Technology-Agnostic Quantum Middle Layer](http://arxiv.org/pdf/2510.07079v1)

Authors: Stefano Markidis, Gilbert Netzer, Luca Pennati, Ivy Peng

We present a blueprint for a quantum middle layer that supports applications
across various quantum technologies. Inspired by concepts and abstractions from
HPC libraries and middleware, our design is backend-neutral and context-aware.
A program only needs to specify its intent once as typed data and operator
descriptors. It declares what the quantum registers mean and which logical
transformations are required, without committing to gates, pulses,
continuous-variable routines, or anneal backend. Such execution details are
carried separately in a context descriptor and can change per backend without
modifying the intent artifacts.
  We develop a proof of concept implementation that uses JSON files for the
descriptors and two backends: a gate-model path realized with IBM Qiskit Aer
simulator and an annealing path realized with D-Wave Ocean's simulated
annealer. On a Max-Cut problem instance, the same typed problem runs on both
backends by varying only the operator formulation (Quantum Approximated
Optimization Algorithm formulation vs. Ising Hamiltonian formulation) and the
context. The proposed middle layer concepts are characterized by portability,
composability, and its minimal core can evolve with hardware capabilities.

### 2. [Neuromorphic Computing -- An Overview](http://arxiv.org/pdf/2510.06721v1)

Authors: Benedikt Jung, Maximilian Kalcher, Merlin Marinova, Piper Powell, Esma Sakalli

With traditional computing technologies reaching their limit, a new field has
emerged seeking to follow the example of the human brain into a new era:
neuromorphic computing. This paper provides an introduction to neuromorphic
computing, why this and other new computing systems are needed, and what
technologies currently exist in the neuromorphic field. It begins with a
general introduction into the history of traditional computing and its present
problems, and then proceeds to a general overview of neuromorphic systems. It
subsequently discusses the main technologies currently in development. For
completeness, the paper first discusses neuromorphic-style computing on
traditional hardware, and then discusses the two top branches of specialized
hardware in this field; neuromorphic chips and photonic systems. Both branches
are explained as well as their relative benefits and drawbacks. The paper
concludes with a summary and an outlook on the future.

### 3. [The Stage Comes to You: A Real-Time Tele-Immersive System with 3D Point Clouds and Vibrotactile Feedback](http://arxiv.org/pdf/2510.07009v1)

Authors: Takahiro Matsumoto, Takahiro Kusabuka, Hiroshi Chigira, Kazuhiko Murasaki, Kakagu Komazaki, Masafumi Suzuki, Masakatsu Aoki

We present a low-latency tele-immersive entertainment system that streams 3D
point clouds and performers' footstep vibrations, creating the sense that the
stage is present. Moving performers and their surroundings are captured as
dynamic point clouds under rapidly changing lighting, then processed,
transmitted, and rendered within a total latency of less than 100 ms. Under
high ambient noise, footstep vibrations are sensed by wearable accelerometers.
Real-time visual and haptic streams are delivered to a remote venue, where a
large 3D LED wall and a vibration-efficient haptic floor envelop dozens of
spectators. A public trial at Expo 2025 linked sites 20 km apart: visitors
watched a live dance show and conversed with performers without noticeable
delay.

### 4. [From Description to Detection: LLM based Extendable O-RAN Compliant Blind DoS Detection in 5G and Beyond](http://arxiv.org/pdf/2510.06530v1)

Authors: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

The quality and experience of mobile communication have significantly
improved with the introduction of 5G, and these improvements are expected to
continue beyond the 5G era. However, vulnerabilities in control-plane
protocols, such as Radio Resource Control (RRC) and Non-Access Stratum (NAS),
pose significant security threats, such as Blind Denial of Service (DoS)
attacks. Despite the availability of existing anomaly detection methods that
leverage rule-based systems or traditional machine learning methods, these
methods have several limitations, including the need for extensive training
data, predefined rules, and limited explainability. Addressing these
challenges, we propose a novel anomaly detection framework that leverages the
capabilities of Large Language Models (LLMs) in zero-shot mode with unordered
data and short natural language attack descriptions within the Open Radio
Access Network (O-RAN) architecture. We analyse robustness to prompt variation,
demonstrate the practicality of automating the attack descriptions and show
that detection quality relies on the semantic completeness of the description
rather than its phrasing or length. We utilise an RRC/NAS dataset to evaluate
the solution and provide an extensive comparison of open-source and proprietary
LLM implementations to demonstrate superior performance in attack detection. We
further validate the practicality of our framework within O-RAN's real-time
constraints, illustrating its potential for detecting other Layer-3 attacks.

### 5. [A Review of 10 Years of ProtoSpace: Spacecraft CAD Visualization in Collaborative Augmented Reality](http://arxiv.org/pdf/2510.06608v1)

Authors: Benjamin Nuernberger, Samuel-Hunter Berndt, Robert Tapella, Laura Mann, Aaron Plave, Sasha Samochina, Victor X. Luo

ProtoSpace is a custom JPL-built platform to help scientists and engineers
visualize their CAD models collaboratively in augmented reality (AR) and on the
web in 3D. In addition to this main use case, ProtoSpace has been used
throughout the entire spacecraft mission lifecycle and beyond: ventilator
design and assembly; providing AR-based instructions to astronauts in-training;
educating the next generation on the process of spacecraft design; etc.
ProtoSpace has been used for a decade by NASA missions-including Mars
Perseverance, Europa Clipper, NISAR, SPHEREx, CAL, and Mars Sample Return-to
reduce cost and risk by helping engineers and scientists fix problems earlier
through reducing miscommunication and helping people understand the spatial
context of their spacecraft in the appropriate physical context more quickly.
This paper will explore how ProtoSpace came to be, define the system
architecture and overview-including HoloLens and 3D web clients, the ProtoSpace
server, and the CAD model optimizer-and dive into the use cases, spin-offs, and
lessons learned that led to 10 years of success at NASA's Jet Propulsion
Laboratory.

### 6. [From Neural Sensing to Stimulation: An Interdisciplinary Roadmap for Neurotechnology](http://arxiv.org/pdf/2510.07116v1)

Authors: Ruben Ruiz-Mateos Serrano, Joe G Troughton, Nima Mirkhani, Natalia Martinez, Massimo Mariello, Jordan Tsigarides, Simon Williamson, Juan Sapriza, Ioana Susnoschi Luca, Antonio Dominguez-Alfaro, Estelle Cuttaz, Nicole Thompson, Sydney Swedick, Latifah Almulla, Amparo Guemes

Neurotechnologies are transforming how we measure, interpret, and modulate
brain-body interactions, integrating real-time sensing, computation, and
stimulation to enable precise physiological control. They hold transformative
potential across clinical and non-clinical domains, from treating disorders to
enhancing cognition and performance. Realizing this potential requires
navigating complex, interdisciplinary challenges spanning neuroscience,
materials science, device engineering, signal processing, computational
modelling, and regulatory and ethical frameworks. This Perspective presents a
strategic roadmap for neurotechnology development, created by early-career
researchers, highlighting their role at the intersection of disciplines and
their capacity to bridge traditional silos. We identify five cross-cutting
trade-offs that constrain progress across functionality, scalability,
adaptability, and translatability, and illustrate how technical domains
influence their resolution. Rather than a domain-specific review, we focus on
shared challenges and strategic opportunities that transcend disciplines. We
propose a unified framework for collaborative innovation and education,
highlight ethical and regulatory priorities, and outline a timeline for
overcoming key bottlenecks. By aligning technical development with
translational and societal needs, this roadmap aims to accelerate equitable,
effective, and future-ready adaptive neurotechnologies, guiding coordinated
efforts across the global research and innovation community.

### Graphics

### 1. [Geometric Queries on Closed Implicit Surfaces for Walk on Stars](http://arxiv.org/pdf/2510.07275v1)

Authors: Tianyu Huang

Walk on stars (WoSt) is currently one of the most advanced Monte Carlo
solvers for PDEs. Unfortunately, the lack of reliable geometric query
approaches has hindered its applicability to boundaries defined by implicit
surfaces. This work proposes a geometric query framework over closed implicit
surfaces for WoSt, under the scope of walkin' Robin. Our key observation is
that all WoSt queries can be formulated as constrained global optimization or
constraint satisfaction problems. Based on our formulations, to solve the
highly non-convex problems, we adopt a branch-and-bound approach based on
interval analysis. To the best of our knowledge, our method is the first to
study closest silhouette point queries and Robin radius bound queries on closed
implicit surfaces. Our formulations and methods first enable mesh-free PDE
solving via WoSt when boundaries are defined by closed implicit surfaces.

### 2. [Capture and Interact: Rapid 3D Object Acquisition and Rendering with Gaussian Splatting in Unity](http://arxiv.org/pdf/2510.06802v1)

Authors: Islomjon Shukhratov, Sergey Gorinsky

Capturing and rendering three-dimensional (3D) objects in real time remain a
significant challenge, yet hold substantial potential for applications in
augmented reality, digital twin systems, remote collaboration and prototyping.
We present an end-to-end pipeline that leverages 3D Gaussian Splatting (3D GS)
to enable rapid acquisition and interactive rendering of real-world objects
using a mobile device, cloud processing and a local computer. Users scan an
object with a smartphone video, upload it for automated 3D reconstruction, and
visualize it interactively in Unity at an average of 150 frames per second
(fps) on a laptop. The system integrates mobile capture, cloud-based 3D GS and
Unity rendering to support real-time telepresence. Our experiments show that
the pipeline processes scans in approximately 10 minutes on a graphics
processing unit (GPU) achieving real-time rendering on the laptop.

### 3. [A Review of 10 Years of ProtoSpace: Spacecraft CAD Visualization in Collaborative Augmented Reality](http://arxiv.org/pdf/2510.06608v1)

Authors: Benjamin Nuernberger, Samuel-Hunter Berndt, Robert Tapella, Laura Mann, Aaron Plave, Sasha Samochina, Victor X. Luo

ProtoSpace is a custom JPL-built platform to help scientists and engineers
visualize their CAD models collaboratively in augmented reality (AR) and on the
web in 3D. In addition to this main use case, ProtoSpace has been used
throughout the entire spacecraft mission lifecycle and beyond: ventilator
design and assembly; providing AR-based instructions to astronauts in-training;
educating the next generation on the process of spacecraft design; etc.
ProtoSpace has been used for a decade by NASA missions-including Mars
Perseverance, Europa Clipper, NISAR, SPHEREx, CAL, and Mars Sample Return-to
reduce cost and risk by helping engineers and scientists fix problems earlier
through reducing miscommunication and helping people understand the spatial
context of their spacecraft in the appropriate physical context more quickly.
This paper will explore how ProtoSpace came to be, define the system
architecture and overview-including HoloLens and 3D web clients, the ProtoSpace
server, and the CAD model optimizer-and dive into the use cases, spin-offs, and
lessons learned that led to 10 years of success at NASA's Jet Propulsion
Laboratory.

### Computer Science and Game Theory

### 1. [Constant Weighted Maximin Share Approximations for Chores](http://arxiv.org/pdf/2510.06581v1)

Authors: Bo Li, Fangxiao Wang, Shiji Xing

We study the fair allocation of indivisible chores among agents with
asymmetric weights. Among the various fairness notions, weighted maximin share
(WMMS) stands out as particularly compelling. However, whether WMMS admits a
constant-factor approximation has remained unknown and is one of the important
open problems in weighted fair division [ALMW22, Suk25]. So far, the best known
approximation ratio is O(log n), where n is the number of agents. In this
paper, we advance the state of the art and present the first constant-factor
approximate WMMS algorithm. To this end, we introduce canonical instance
reductions and different bounds of agents' valuations. We also prove that
guaranteeing better than 2-approximation is not possible, which improves the
best-known lower bound of 1.366.

### 2. [Data as Commodity: a Game-Theoretic Principle for Information Pricing](http://arxiv.org/pdf/2510.07101v1)

Authors: Pasquale Casaburi, Giovanni Piccioli, Pierpaolo Vivo

Data is the central commodity of the digital economy. Unlike physical goods,
it is non-rival, replicable at near-zero cost, and traded under heterogeneous
licensing rules. These properties defy standard supply--demand theory and call
for new pricing principles. We propose a game-theoretic approach in which the
value of a data string emerges from strategic competition among N players
betting on an underlying stochastic process, each holding partial information
about past outcomes. A better-informed player faces a choice: exploit their
informational advantage, or sell part of their dataset to less-informed
competitors. By analytically computing the Nash equilibrium of the game, we
determine the price range where the trade is beneficial to both buyer and
seller. We uncover a rich landscape of market effects that diverge from
textbook economics: first, prospective sellers and buyers can compete or
jointly exploit the less informed competitors depending on the quality of data
they hold. In a symbiotic regime, the seller can even share data for free while
still improving her payoffs, showing that losing exclusivity does not
necessarily reduce profit. Moreover, rivalry between well-informed players can
paradoxically benefit uninformed ones, demonstrating that information abundance
does not always translate to higher payoffs. We also show that the number of
players influences the competition between informed parties: trades impossible
in small markets become feasible in larger ones. These findings establish a
theoretical foundation for the pricing of intangible goods in dynamically
interacting digital markets, which are in need of robust valuation principles.

### 3. [Dynamic Regret Bounds for Online Omniprediction with Long Term Constraints](http://arxiv.org/pdf/2510.07266v1)

Authors: Yahav Bechavod, Jiuyao Lu, Aaron Roth

We present an algorithm guaranteeing dynamic regret bounds for online
omniprediction with long term constraints. The goal in this recently introduced
problem is for a learner to generate a sequence of predictions which are
broadcast to a collection of downstream decision makers. Each decision maker
has their own utility function, as well as a vector of constraint functions,
each mapping their actions and an adversarially selected state to reward or
constraint violation terms. The downstream decision makers select actions "as
if" the state predictions are correct, and the goal of the learner is to
produce predictions such that all downstream decision makers choose actions
that give them worst-case utility guarantees while minimizing worst-case
constraint violation. Within this framework, we give the first algorithm that
obtains simultaneous \emph{dynamic regret} guarantees for all of the agents --
where regret for each agent is measured against a potentially changing sequence
of actions across rounds of interaction, while also ensuring vanishing
constraint violation for each agent. Our results do not require the agents
themselves to maintain any state -- they only solve one-round constrained
optimization problems defined by the prediction made at that round.

### Human-Computer Interaction

### 1. [Examining Solidarity Against AI-Enabled Surveillance at the Intersection of Workplace and Carceral Realities](http://arxiv.org/pdf/2510.06537v1)

Authors: Morgan McErlean, Cella M. Sum, Sukrit Venkatagiri, Sarah Fox

As panoptical, AI-driven surveillance becomes a norm, everyone is impacted.
In a reality where all people fall victim to these technologies, establishing
links and solidarity is essential to fighting back. Two groups facing rising
and targeted surveillance are workers and individuals impacted by the carceral
system. Through preliminary data collection from a worker-surveillance lens,
our findings reveal several cases of these surveillance infrastructures
intersecting. Continuation of our work will involve collecting cases from a
carceral-centered lens. Driven by a community-facing analysis of the overlap in
the AI-driven surveillance experienced by workers and individuals impacted by
the carceral system, we will facilitate discussions with restorative justice
activists around cultivating solidarity and empowerment focused on the
interconnected nature of workplace and carceral surveillance technologies.

### 2. [PriorWeaver: Prior Elicitation via Iterative Dataset Construction](http://arxiv.org/pdf/2510.06550v1)

Authors: Yuwei Xiao, Shuai Ma, Antti Oulasvirta, Eunice Jun

In Bayesian analysis, prior elicitation, or the process of explicating one's
beliefs to inform statistical modeling, is an essential yet challenging step.
Analysts often have beliefs about real-world variables and their relationships.
However, existing tools require analysts to translate these beliefs and express
them indirectly as probability distributions over model parameters. We present
PriorWeaver, an interactive visualization system that facilitates prior
elicitation through iterative dataset construction and refinement. Analysts
visually express their assumptions about individual variables and their
relationships. Under the hood, these assumptions create a dataset used to
derive statistical priors. Prior predictive checks then help analysts compare
the priors to their assumptions. In a lab study with 17 participants new to
Bayesian analysis, we compare PriorWeaver to a baseline incorporating existing
techniques. Compared to the baseline, PriorWeaver gave participants greater
control, clarity, and confidence, leading to priors that were better aligned
with their expectations.

### 3. [RAVEN: Realtime Accessibility in Virtual ENvironments for Blind and Low-Vision People](http://arxiv.org/pdf/2510.06573v1)

Authors: Xinyun Cao, Kexin Phyllis Ju, Chenglin Li, Venkatesh Potluri, Dhruv Jain

As virtual 3D environments become prevalent, equitable access is crucial for
blind and low-vision (BLV) users who face challenges with spatial awareness,
navigation, and interactions. To address this gap, previous work explored
supplementing visual information with auditory and haptic modalities. However,
these methods are static and offer limited support for dynamic, in-context
adaptation. Recent work in generative AI enables users to query and modify 3D
scenes via natural language, introducing a paradigm with increased flexibility
and control for accessibility improvements. We present RAVEN, a system that
responds to query or modification prompts from BLV users to improve the runtime
accessibility of 3D virtual scenes. We evaluated the system with eight BLV
people, uncovering key insights into the strengths and shortcomings of
generative AI-driven accessibility in virtual 3D environments, pointing to
promising results as well as challenges related to system reliability and user
trust.

### 4. [Investigating Students' Preferences for AI Roles in Mathematical Modelling: Evidence from a Randomized Controlled Trial](http://arxiv.org/pdf/2510.06617v1)

Authors: Wangda Zhu, Guang Chen, Yumeng Zhu, Lei Cai, Xiangen Hu

Mathematical modelling (MM) is a key competency for solving complex
real-world problems, yet many students struggle with abstraction,
representation, and iterative reasoning. Artificial intelligence (AI) has been
proposed as a support for higher-order thinking, but its role in MM education
is still underexplored. This study examines the relationships among students'
design thinking (DT), computational thinking (CT), and mathematical modelling
self-efficacy (MMSE), and investigates their preferences for different AI roles
during the modelling process. Using a randomized controlled trial, we identify
significant connections among DT, CT, and MMSE, and reveal distinct patterns in
students' preferred AI roles, including AI as a tutor (providing explanations
and feedback), AI as a tool (assisting with calculations and representations),
AI as a collaborator (suggesting strategies and co-creating models), and AI as
a peer (offering encouragement and fostering reflection). Differences across
learner profiles highlight how students' dispositions shape their expectations
for AI. These findings advance understanding of AI-supported MM and provide
design implications for adaptive, learner-centered systems.

### 5. ["It feels like hard work trying to talk to it": Understanding Older Adults' Experiences of Encountering and Repairing Conversational Breakdowns with AI Systems](http://arxiv.org/pdf/2510.06690v1)

Authors: Niharika Mathur, Tamara Zubatiy, Agata Rozga, Elizabeth Mynatt

Designing Conversational AI systems to support older adults requires more
than usability and reliability, it also necessitates robustness in handling
conversational breakdowns. In this study, we investigate how older adults
navigate and repair such breakdowns while interacting with a voice-based AI
system deployed in their homes for medication management. Through a 20-week
in-home deployment with 7 older adult participant dyads, we analyzed 844
recoded interactions to identify conversational breakdowns and user-initiated
repair strategies. Through findings gleaned from post-deployment interviews, we
reflect on the nature of these breakdowns and older adults' experiences of
mitigating them. We identify four types of conversational breakdowns and
demonstrate how older adults draw on their situated knowledge and environment
to make sense of and recover from these disruptions, highlighting the cognitive
effort required in doing so. Our findings emphasize the collaborative nature of
interactions in human-AI contexts, and point to the need for AI systems to
better align with users' expectations for memory, their routines, and external
resources in their environment. We conclude by discussing opportunities for AI
systems to integrate contextual knowledge from older adults' sociotechnical
environment and to facilitate more meaningful and user-centered interactions.

### 6. ["Sometimes You Need Facts, and Sometimes a Hug": Understanding Older Adults' Preferences for Explanations in LLM-Based Conversational AI Systems](http://arxiv.org/pdf/2510.06697v1)

Authors: Niharika Mathur, Tamara Zubatiy, Agata Rozga, Jodi Forlizzi, Elizabeth Mynatt

Designing Conversational AI systems to support older adults requires these
systems to explain their behavior in ways that align with older adults'
preferences and context. While prior work has emphasized the importance of AI
explainability in building user trust, relatively little is known about older
adults' requirements and perceptions of AI-generated explanations. To address
this gap, we conducted an exploratory Speed Dating study with 23 older adults
to understand their responses to contextually grounded AI explanations. Our
findings reveal the highly context-dependent nature of explanations, shaped by
conversational cues such as the content, tone, and framing of explanation. We
also found that explanations are often interpreted as interactive, multi-turn
conversational exchanges with the AI, and can be helpful in calibrating
urgency, guiding actionability, and providing insights into older adults' daily
lives for their family members. We conclude by discussing implications for
designing context-sensitive and personalized explanations in Conversational AI
systems.

### 7. [Lonely Individuals Show Distinct Patterns of Social Media Engagement](http://arxiv.org/pdf/2510.06733v1)

Authors: Yajing Wang, Talayeh Aledavood, Juhi Kulshrestha

Loneliness has reached epidemic proportions globally, posing serious risks to
mental and physical health. As social media platforms increasingly mediate
social interaction, understanding their relationship with loneliness has become
urgent. While survey-based research has examined social media use and
loneliness, findings remain mixed, and little is known about when and how often
people engage with social media, or about whether different types of platforms
are differently associated with loneliness. Web trace data now enable objective
examination of these behavioral dimensions. We asked whether objectively
measured patterns of social media engagement differ between lonely and
non-lonely individuals across devices and platform types. Analyzing six months
of web trace data combined with repeated surveys ($N=589$ mobile users; $N=851$
desktop users), we found that greater social media use was associated with
higher loneliness across both devices, with this relationship specific to
social media rather than other online activities. On desktop, lonely
individuals exhibited shorter sessions but more frequent daily engagement.
Lonely individuals spent more time on visual-sharing ($g = -0.47$), messaging
($g = -0.36$), and networking-oriented platforms on mobile. These findings
demonstrate how longitudinal web trace data can reveal behavioral patterns
associated with loneliness, and more broadly illustrate the potential of
digital traces for studying other psychological states. Beyond research, the
results inform the responsible design of digital interventions and platform
features that better support psychological well-being across different
technological contexts.

### 8. [Prototyping Multimodal GenAI Real-Time Agents with Counterfactual Replays and Hybrid Wizard-of-Oz](http://arxiv.org/pdf/2510.06872v1)

Authors: Frederic Gmeiner, Kenneth Holstein, Nikolas Martelaro

Recent advancements in multimodal generative AI (GenAI) enable the creation
of personal context-aware real-time agents that, for example, can augment user
workflows by following their on-screen activities and providing contextual
assistance. However, prototyping such experiences is challenging, especially
when supporting people with domain-specific tasks using real-time inputs such
as speech and screen recordings. While prototyping an LLM-based proactive
support agent system, we found that existing prototyping and evaluation methods
were insufficient to anticipate the nuanced situational complexity and
contextual immediacy required. To overcome these challenges, we explored a
novel user-centered prototyping approach that combines counterfactual video
replay prompting and hybrid Wizard-of-Oz methods to iteratively design and
refine agent behaviors. This paper discusses our prototyping experiences,
highlighting successes and limitations, and offers a practical guide and an
open-source toolkit for UX designers, HCI researchers, and AI toolmakers to
build more user-centered and context-aware multimodal agents.

### 9. [The Feature Understandability Scale for Human-Centred Explainable AI: Assessing Tabular Feature Importance](http://arxiv.org/pdf/2510.07050v1)

Authors: Nicola Rossberg, Bennett Kleinberg, Barry O'Sullivan, Luca Longo, Andrea Visentin

As artificial intelligence becomes increasingly pervasive and powerful, the
ability to audit AI-based systems is becoming increasingly important. However,
explainability for artificial intelligence systems is not a one-size-fits-all
solution; different target audiences have varying requirements and expectations
for explanations. While various approaches to explainability have been
proposed, most explainable artificial intelligence (XAI) methods for tabular
data focus on explaining the outputs of supervised machine learning models
using the input features. However, a user's ability to understand an
explanation depends on their understanding of such features. Therefore, it is
in the best interest of the system designer to try to pre-select understandable
features for producing a global explanation of an ML model. Unfortunately, no
measure currently exists to assess the degree to which a user understands a
given input feature. This work introduces psychometrically validated scales
that quantitatively seek to assess users' understanding of tabular input
features for supervised classification problems. In detail, these scales, one
for numerical and one for categorical data, each with two factors and
comprising 8 and 9 items, aim to assign a score to each input feature,
effectively producing a rank, and allowing for the quantification of feature
prioritisation. A confirmatory factor analysis demonstrates a strong
relationship between such items and a good fit of the two-factor structure for
each scale. This research presents a novel method for assessing understanding
and outlines potential applications in the domain of explainable artificial
intelligence.

### 10. [AI for Abolition? A Participatory Design Approach](http://arxiv.org/pdf/2510.07156v1)

Authors: Carolyn Wang, Avriel Epps, Taylor Ferrari, Ra Ames

The abolitionist community faces challenges from both the carceral state and
oppressive technologies which, by empowering the ruling class who have the
resources to develop artificial intelligence (AI), serve to entrench societal
inequities even more deeply. This paper presents a case study in participatory
design with transformative and restorative justice practitioners with the goal
of designing an AI system to support their work. By co-designing an evaluation
framework for large language models with the practitioners, we hope to push
back against the exclusionary status quo of AI and extent AI's potentiality to
a historically marginalized community.

### Information Retrieval

### 1. [LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations](http://arxiv.org/pdf/2510.06657v1)

Authors: Boyuan Long, Yueqi Wang, Hiloni Mehta, Mick Zomnir, Omkar Pathak, Changping Meng, Ruolin Jia, Yajun Peng, Dapeng Hong, Xia Wu, Mingyan Gao, Onkar Dalal, Ningren Han

This paper presents a case study on deploying Large Language Models (LLMs) as
an advanced "annotation" mechanism to achieve nuanced content understanding
(e.g., discerning content "vibe") at scale within a large-scale industrial
short-form video recommendation system. Traditional machine learning
classifiers for content understanding face protracted development cycles and a
lack of deep, nuanced comprehension. The "LLM-as-annotators" approach addresses
these by significantly shortening development times and enabling the annotation
of subtle attributes. This work details an end-to-end workflow encompassing:
(1) iterative definition and robust evaluation of target attributes, refined by
offline metrics and online A/B testing; (2) scalable offline bulk annotation of
video corpora using LLMs with multimodal features, optimized inference, and
knowledge distillation for broad application; and (3) integration of these rich
annotations into the online recommendation serving system, for example, through
personalized restrict retrieval. Experimental results demonstrate the efficacy
of this approach, with LLMs outperforming human raters in offline annotation
quality for nuanced attributes and yielding significant improvements of user
participation and satisfied consumption in online A/B tests. The study provides
insights into designing and scaling production-level LLM pipelines for rich
content evaluation, highlighting the adaptability and benefits of LLM-generated
nuanced understanding for enhancing content discovery, user satisfaction, and
the overall effectiveness of modern recommendation systems.

### 2. [Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks](http://arxiv.org/pdf/2510.06658v1)

Authors: Jiaman He, Zikang Leng, Dana McKay, Damiano Spina, Johanne R. Trippas

Many evaluations of large language models (LLMs) in text annotation focus
primarily on the correctness of the output, typically comparing model-generated
labels to human-annotated ``ground truth'' using standard performance metrics.
In contrast, our study moves beyond effectiveness alone. We aim to explore how
labeling decisions -- by both humans and LLMs -- can be statistically evaluated
across individuals. Rather than treating LLMs purely as annotation systems, we
approach LLMs as an alternative annotation mechanism that may be capable of
mimicking the subjective judgments made by humans. To assess this, we develop a
statistical evaluation method based on Krippendorff's $\alpha$, paired
bootstrapping, and the Two One-Sided t-Tests (TOST) equivalence test procedure.
This evaluation method tests whether an LLM can blend into a group of human
annotators without being distinguishable.
  We apply this approach to two datasets -- MovieLens 100K and PolitiFact --
and find that the LLM is statistically indistinguishable from a human annotator
in the former ($p = 0.004$), but not in the latter ($p = 0.155$), highlighting
task-dependent differences. It also enables early evaluation on a small sample
of human data to inform whether LLMs are suitable for large-scale annotation in
a given application.

### 3. [Reproducing and Extending Causal Insights Into Term Frequency Computation in Neural Rankers](http://arxiv.org/pdf/2510.06728v1)

Authors: Cile van Marken, Roxana Petcu

Neural ranking models have shown outstanding performance across a variety of
tasks, such as document retrieval, re-ranking, question answering and
conversational retrieval. However, the inner decision process of these models
remains largely unclear, especially as models increase in size. Most
interpretability approaches, such as probing, focus on correlational insights
rather than establishing causal relationships. The paper 'Axiomatic Causal
Interventions for Reverse Engineering Relevance Computation in Neural Retrieval
Models' by Chen et al. addresses this gap by introducing a framework for
activation patching - a causal interpretability method - in the information
retrieval domain, offering insights into how neural retrieval models compute
document relevance. The study demonstrates that neural ranking models not only
capture term-frequency information, but also that these representations can be
localized to specific components of the model, such as individual attention
heads or layers. This paper aims to reproduce the findings by Chen et al. and
to further explore the presence of pre-defined retrieval axioms in neural IR
models. We validate the main claims made by Chen et al., and extend the
framework to include an additional term-frequency axiom, which states that the
impact of increasing query term frequency on document ranking diminishes as the
frequency becomes higher. We successfully identify a group of attention heads
that encode this axiom and analyze their behavior to give insight into the
inner decision-making process of neural ranking models.

### 4. [Ethical AI prompt recommendations in large language models using collaborative filtering](http://arxiv.org/pdf/2510.06924v1)

Authors: Jordan Nelson, Almas Baimagambetov, Konstantinos Avgerinakis, Nikolaos Polatidis

As large language models (LLMs) shape AI development, ensuring ethical prompt
recommendations is crucial. LLMs offer innovation but risk bias, fairness
issues, and accountability concerns. Traditional oversight methods struggle
with scalability, necessitating dynamic solutions. This paper proposes using
collaborative filtering, a technique from recommendation systems, to enhance
ethical prompt selection. By leveraging user interactions, it promotes ethical
guidelines while reducing bias. Contributions include a synthetic dataset for
prompt recommendations and the application of collaborative filtering. The work
also tackles challenges in ethical AI, such as bias mitigation, transparency,
and preventing unethical prompt engineering.

### 5. [Overview of the Plagiarism Detection Task at PAN 2025](http://arxiv.org/pdf/2510.06805v1)

Authors: André Greiner-Petter, Maik Fröbe, Jan Philip Wahle, Terry Ruas, Bela Gipp, Akiko Aizawa, Martin Potthast

The generative plagiarism detection task at PAN 2025 aims at identifying
automatically generated textual plagiarism in scientific articles and aligning
them with their respective sources. We created a novel large-scale dataset of
automatically generated plagiarism using three large language models: Llama,
DeepSeek-R1, and Mistral. In this task overview paper, we outline the creation
of this dataset, summarize and compare the results of all participants and four
baselines, and evaluate the results on the last plagiarism detection task from
PAN 2015 in order to interpret the robustness of the proposed approaches. We
found that the current iteration does not invite a large variety of approaches
as naive semantic similarity approaches based on embedding vectors provide
promising results of up to 0.8 recall and 0.5 precision. In contrast, most of
these approaches underperform significantly on the 2015 dataset, indicating a
lack in generalizability.

### 6. [Crossing Domains without Labels: Distant Supervision for Term Extraction](http://arxiv.org/pdf/2510.06838v1)

Authors: Elena Senger, Yuri Campbell, Rob van der Goot, Barbara Plank

Automatic Term Extraction (ATE) is a critical component in downstream NLP
tasks such as document tagging, ontology construction and patent analysis.
Current state-of-the-art methods require expensive human annotation and
struggle with domain transfer, limiting their practical deployment. This
highlights the need for more robust, scalable solutions and realistic
evaluation settings. To address this, we introduce a comprehensive benchmark
spanning seven diverse domains, enabling performance evaluation at both the
document- and corpus-levels. Furthermore, we propose a robust LLM-based model
that outperforms both supervised cross-domain encoder models and few-shot
learning baselines and performs competitively with its GPT-4o teacher on this
benchmark. The first step of our approach is generating psuedo-labels with this
black-box LLM on general and scientific domains to ensure generalizability.
Building on this data, we fine-tune the first LLMs for ATE. To further enhance
document-level consistency, oftentimes needed for downstream tasks, we
introduce lightweight post-hoc heuristics. Our approach exceeds previous
approaches on 5/7 domains with an average improvement of 10 percentage points.
We release our dataset and fine-tuned models to support future research in this
area.

### 7. [M3Retrieve: Benchmarking Multimodal Retrieval for Medicine](http://arxiv.org/pdf/2510.06888v1)

Authors: Arkadeep Acharya, Akash Ghosh, Pradeepika Verma, Kitsuchart Pasupa, Sriparna Saha, Priti Singh

With the increasing use of RetrievalAugmented Generation (RAG), strong
retrieval models have become more important than ever. In healthcare,
multimodal retrieval models that combine information from both text and images
offer major advantages for many downstream tasks such as question answering,
cross-modal retrieval, and multimodal summarization, since medical data often
includes both formats. However, there is currently no standard benchmark to
evaluate how well these models perform in medical settings. To address this
gap, we introduce M3Retrieve, a Multimodal Medical Retrieval Benchmark.
M3Retrieve, spans 5 domains,16 medical fields, and 4 distinct tasks, with over
1.2 Million text documents and 164K multimodal queries, all collected under
approved licenses. We evaluate leading multimodal retrieval models on this
benchmark to explore the challenges specific to different medical specialities
and to understand their impact on retrieval performance. By releasing
M3Retrieve, we aim to enable systematic evaluation, foster model innovation,
and accelerate research toward building more capable and reliable multimodal
retrieval systems for medical applications. The dataset and the baselines code
are available in this github page https://github.com/AkashGhosh/M3Retrieve.

### 8. [Towards Reliable Retrieval in RAG Systems for Large Legal Datasets](http://arxiv.org/pdf/2510.06999v1)

Authors: Markus Reuter, Tobias Lingenberg, Rūta Liepiņa, Francesca Lagioia, Marco Lippi, Giovanni Sartor, Andrea Passerini, Burcu Sayin

Retrieval-Augmented Generation (RAG) is a promising approach to mitigate
hallucinations in Large Language Models (LLMs) for legal applications, but its
reliability is critically dependent on the accuracy of the retrieval step. This
is particularly challenging in the legal domain, where large databases of
structurally similar documents often cause retrieval systems to fail. In this
paper, we address this challenge by first identifying and quantifying a
critical failure mode we term Document-Level Retrieval Mismatch (DRM), where
the retriever selects information from entirely incorrect source documents. To
mitigate DRM, we investigate a simple and computationally efficient technique
which we refer to as Summary-Augmented Chunking (SAC). This method enhances
each text chunk with a document-level synthetic summary, thereby injecting
crucial global context that would otherwise be lost during a standard chunking
process. Our experiments on a diverse set of legal information retrieval tasks
show that SAC greatly reduces DRM and, consequently, also improves text-level
retrieval precision and recall. Interestingly, we find that a generic
summarization strategy outperforms an approach that incorporates legal expert
domain knowledge to target specific legal elements. Our work provides evidence
that this practical, scalable, and easily integrable technique enhances the
reliability of RAG systems when applied to large-scale legal document datasets.

### 9. [Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization](http://arxiv.org/pdf/2510.06732v1)

Authors: Tiancheng Xing, Jerry Li, Yixuan Du, Xiyang Hu

Large language models (LLMs) are increasingly used as rerankers in
information retrieval, yet their ranking behavior can be steered by small,
natural-sounding prompts. To expose this vulnerability, we present Rank
Anything First (RAF), a two-stage token optimization method that crafts concise
textual perturbations to consistently promote a target item in LLM-generated
rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate
Gradient to shortlist candidate tokens at the current position by combining the
gradient of the rank-target with a readability score; Stage 2 evaluates those
candidates under exact ranking and readability losses using an entropy-based
dynamic weighting scheme, and selects a token via temperature-controlled
sampling. RAF generates ranking-promoting prompts token-by-token, guided by
dual objectives: maximizing ranking effectiveness and preserving linguistic
naturalness. Experiments across multiple LLMs show that RAF significantly
boosts the rank of target items using naturalistic language, with greater
robustness than existing methods in both promoting target items and maintaining
naturalness. These findings underscore a critical security implication:
LLM-based reranking is inherently susceptible to adversarial manipulation,
raising new challenges for the trustworthiness and robustness of modern
retrieval systems. Our code is available at: https://github.com/glad-lab/RAF.

### 10. [Exposing Citation Vulnerabilities in Generative Engines](http://arxiv.org/pdf/2510.06823v1)

Authors: Riku Mochizuki, Shusuke Komatsu, Souta Noguchi, Kazuto Ataka

We analyze answers generated by generative engines (GEs) from the
perspectives of citation publishers and the content-injection barrier, defined
as the difficulty for attackers to manipulate answers to user prompts by
placing malicious content on the web. GEs integrate two functions: web search
and answer generation that cites web pages using large language models. Because
anyone can publish information on the web, GEs are vulnerable to poisoning
attacks. Existing studies of citation evaluation focus on how faithfully answer
content reflects cited sources, leaving unexamined which web sources should be
selected as citations to defend against poisoning attacks. To fill this gap, we
introduce evaluation criteria that assess poisoning threats using the citation
information contained in answers. Our criteria classify the publisher
attributes of citations to estimate the content-injection barrier thereby
revealing the threat of poisoning attacks in current GEs. We conduct
experiments in political domains in Japan and the United States (U.S.) using
our criteria and show that citations from official party websites (primary
sources) are approximately \(25\%\)--\(45\%\) in the U.S. and
\(60\%\)--\(65\%\) in Japan, indicating that U.S. political answers are at
higher risk of poisoning attacks. We also find that sources with low
content-injection barriers are frequently cited yet are poorly reflected in
answer content. To mitigate this threat, we discuss how publishers of primary
sources can increase exposure of their web content in answers and show that
well-known techniques are limited by language differences.

### Machine Learning

### 1. [DPA-Net: A Dual-Path Attention Neural Network for Inferring Glycemic Control Metrics from Self-Monitored Blood Glucose Data](http://arxiv.org/pdf/2510.06623v1)

Authors: Canyu Lei, Benjamin Lobo, Jianxin Xie

Continuous glucose monitoring (CGM) provides dense and dynamic glucose
profiles that enable reliable estimation of Ambulatory Glucose Profile (AGP)
metrics, such as Time in Range (TIR), Time Below Range (TBR), and Time Above
Range (TAR). However, the high cost and limited accessibility of CGM restrict
its widespread adoption, particularly in low- and middle-income regions. In
contrast, self-monitoring of blood glucose (SMBG) is inexpensive and widely
available but yields sparse and irregular data that are challenging to
translate into clinically meaningful glycemic metrics.
  In this work, we propose a Dual-Path Attention Neural Network (DPA-Net) to
estimate AGP metrics directly from SMBG data. DPA-Net integrates two
complementary paths: (1) a spatial-channel attention path that reconstructs a
CGM-like trajectory from sparse SMBG observations, and (2) a multi-scale ResNet
path that directly predicts AGP metrics. An alignment mechanism between the two
paths is introduced to reduce bias and mitigate overfitting. In addition, we
develop an active point selector to identify realistic and informative SMBG
sampling points that reflect patient behavioral patterns.
  Experimental results on a large, real-world dataset demonstrate that DPA-Net
achieves robust accuracy with low errors while reducing systematic bias. To the
best of our knowledge, this is the first supervised machine learning framework
for estimating AGP metrics from SMBG data, offering a practical and clinically
relevant decision-support tool in settings where CGM is not accessible.

### 2. [POME: Post Optimization Model Edit via Muon-style Projection](http://arxiv.org/pdf/2510.06627v1)

Authors: Yong Liu, Di Fu, Yang Luo, Zirui Zhu, Minhao Cheng, Cho-Jui Hsieh, Yang You

We introduce Post-Optimization Model Edit (POME), a new algorithm that
enhances the performance of fine-tuned large language models using only their
pretrained and fine-tuned checkpoints, without requiring extra data or further
optimization. The core idea is to apply a muon-style projection to $\Delta W$,
the difference between the fine-tuned and pretrained weights. This projection
uses truncated singular value decomposition (SVD) to equalize the influence of
dominant update directions and prune small singular values, which often
represent noise. As a simple post-processing step, POME is completely decoupled
from the training pipeline. It requires zero modifications and imposes no
overhead, making it universally compatible with any optimizer or distributed
framework. POME delivers consistent gains, boosting average performance by
+2.5\% on GSM8K and +1.0\% on code generation. Its broad applicability -- from
7B foundation models to 72B RLHF-instructed models -- establishes it as a
practical, zero-cost enhancement for any fine-tuning pipeline. Code is
available at https://github.com/NUS-HPC-AI-Lab/POME.

### 3. [Three Forms of Stochastic Injection for Improved Distribution-to-Distribution Generative Modeling](http://arxiv.org/pdf/2510.06634v1)

Authors: Shiye Su, Yuhui Zhang, Linqi Zhou, Rajesh Ranganath, Serena Yeung-Levy

Modeling transformations between arbitrary data distributions is a
fundamental scientific challenge, arising in applications like drug discovery
and evolutionary simulation. While flow matching offers a natural framework for
this task, its use has thus far primarily focused on the noise-to-data setting,
while its application in the general distribution-to-distribution setting is
underexplored. We find that in the latter case, where the source is also a data
distribution to be learned from limited samples, standard flow matching fails
due to sparse supervision. To address this, we propose a simple and
computationally efficient method that injects stochasticity into the training
process by perturbing source samples and flow interpolants. On five diverse
imaging tasks spanning biology, radiology, and astronomy, our method
significantly improves generation quality, outperforming existing baselines by
an average of 9 FID points. Our approach also reduces the transport cost
between input and generated samples to better highlight the true effect of the
transformation, making flow matching a more practical tool for simulating the
diverse distribution transformations that arise in science.

### 4. [XRPO: Pushing the limits of GRPO with Targeted Exploration and Exploitation](http://arxiv.org/pdf/2510.06672v1)

Authors: Udbhav Bamba, Minghao Fang, Yifan Yu, Haizhong Zheng, Fan Lai

Reinforcement learning algorithms such as GRPO have driven recent advances in
large language model (LLM) reasoning. While scaling the number of rollouts
stabilizes training, existing approaches suffer from limited exploration on
challenging prompts and leave informative feedback signals underexploited, due
to context-independent rollout allocation across prompts (e.g., generating 16
rollouts per prompt) and relying heavily on sparse rewards. This paper presents
XRPO(eXplore - eXploit GRPO), a unified framework that recasts policy
optimization through the principled lens of rollout exploration-exploitation.
To enhance exploration, XRPO introduces a mathematically grounded rollout
allocator that adaptively prioritizes prompts with higher potential for
uncertainty reduction. It further addresses stagnation on zero-reward prompts
through an in-context seeding strategy that injects curated exemplars, steering
the model into more difficult reasoning trajectories. To strengthen
exploitation, XRPO develops a group-relative, novelty-aware advantage
sharpening mechanism that leverages sequence likelihoods to amplify
low-probability yet correct responses, thereby extending the policy's reach
beyond sparse rewards. Experiments across diverse math and coding benchmarks on
both reasoning and non-reasoning models demonstrate that XRPO outperforms
existing advances (e.g., GRPO and GSPO) up to 4% pass@1 and 6% cons@32, while
accelerating training convergence by up to 2.7X.

### 5. [TimeFormer: Transformer with Attention Modulation Empowered by Temporal Characteristics for Time Series Forecasting](http://arxiv.org/pdf/2510.06680v1)

Authors: Zhipeng Liu, Peibo Duan, Xuan Tang, Baixin Li, Yongsheng Huang, Mingyang Geng, Changsheng Zhang, Bin Zhang, Binwu Wang

Although Transformers excel in natural language processing, their extension
to time series forecasting remains challenging due to insufficient
consideration of the differences between textual and temporal modalities. In
this paper, we develop a novel Transformer architecture designed for time
series data, aiming to maximize its representational capacity. We identify two
key but often overlooked characteristics of time series: (1) unidirectional
influence from the past to the future, and (2) the phenomenon of decaying
influence over time. These characteristics are introduced to enhance the
attention mechanism of Transformers. We propose TimeFormer, whose core
innovation is a self-attention mechanism with two modulation terms (MoSA),
designed to capture these temporal priors of time series under the constraints
of the Hawkes process and causal masking. Additionally, TimeFormer introduces a
framework based on multi-scale and subsequence analysis to capture semantic
dependencies at different temporal scales, enriching the temporal dependencies.
Extensive experiments conducted on multiple real-world datasets show that
TimeFormer significantly outperforms state-of-the-art methods, achieving up to
a 7.45% reduction in MSE compared to the best baseline and setting new
benchmarks on 94.04\% of evaluation metrics. Moreover, we demonstrate that the
MoSA mechanism can be broadly applied to enhance the performance of other
Transformer-based models.

### 6. [Distributed Algorithms for Multi-Agent Multi-Armed Bandits with Collision](http://arxiv.org/pdf/2510.06683v1)

Authors: Daoyuan Zhou, Xuchuang Wang, Lin Yang, Yang Gao

We study the stochastic Multiplayer Multi-Armed Bandit (MMAB) problem, where
multiple players select arms to maximize their cumulative rewards. Collisions
occur when two or more players select the same arm, resulting in no reward, and
are observed by the players involved. We consider a distributed setting without
central coordination, where each player can only observe their own actions and
collision feedback. We propose a distributed algorithm with an adaptive,
efficient communication protocol. The algorithm achieves near-optimal group and
individual regret, with a communication cost of only $\mathcal{O}(\log\log T)$.
Our experiments demonstrate significant performance improvements over existing
baselines. Compared to state-of-the-art (SOTA) methods, our approach achieves a
notable reduction in individual regret. Finally, we extend our approach to a
periodic asynchronous setting, proving the lower bound for this problem and
presenting an algorithm that achieves logarithmic regret.

### 7. [A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](http://arxiv.org/pdf/2510.06699v1)

Authors: Gal Fadlon, Idan Arbiv, Nimrod Berman, Omri Azencot

Generating realistic time series data is critical for applications in
healthcare, finance, and science. However, irregular sampling and missing
values present significant challenges. While prior methods address these
irregularities, they often yield suboptimal results and incur high
computational costs. Recent advances in regular time series generation, such as
the diffusion-based ImagenTime model, demonstrate strong, fast, and scalable
generative capabilities by transforming time series into image representations,
making them a promising solution. However, extending ImagenTime to irregular
sequences using simple masking introduces "unnatural" neighborhoods, where
missing values replaced by zeros disrupt the learning process. To overcome
this, we propose a novel two-step framework: first, a Time Series Transformer
completes irregular sequences, creating natural neighborhoods; second, a
vision-based diffusion model with masking minimizes dependence on the completed
values. This approach leverages the strengths of both completion and masking,
enabling robust and efficient generation of realistic time series. Our method
achieves state-of-the-art performance, achieving a relative improvement in
discriminative score by $70\%$ and in computational cost by $85\%$. Code is at
https://github.com/azencot-group/ImagenI2R.

### 8. [Function regression using the forward forward training and inferring paradigm](http://arxiv.org/pdf/2510.06762v1)

Authors: Shivam Padmani, Akshay Joshi

Function regression/approximation is a fundamental application of machine
learning. Neural networks (NNs) can be easily trained for function regression
using a sufficient number of neurons and epochs. The forward-forward learning
algorithm is a novel approach for training neural networks without
backpropagation, and is well suited for implementation in neuromorphic
computing and physical analogs for neural networks. To the best of the authors'
knowledge, the Forward Forward paradigm of training and inferencing NNs is
currently only restricted to classification tasks. This paper introduces a new
methodology for approximating functions (function regression) using the
Forward-Forward algorithm. Furthermore, the paper evaluates the developed
methodology on univariate and multivariate functions, and provides preliminary
studies of extending the proposed Forward-Forward regression to Kolmogorov
Arnold Networks, and Deep Physical Neural Networks.

### 9. [Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness](http://arxiv.org/pdf/2510.06790v1)

Authors: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

Models are susceptible to adversarially out-of-distribution (OOD) data
despite large training-compute investments into their robustification. Zaremba
et al. (2025) make progress on this problem at test time, showing LLM reasoning
improves satisfaction of model specifications designed to thwart attacks,
resulting in a correlation between reasoning effort and robustness to
jailbreaks. However, this benefit of test compute fades when attackers are
given access to gradients or multimodal inputs. We address this gap, clarifying
that inference-compute offers benefits even in such cases. Our approach argues
that compositional generalization, through which OOD data is understandable via
its in-distribution (ID) components, enables adherence to defensive
specifications on adversarially OOD inputs. Namely, we posit the Robustness
from Inference Compute Hypothesis (RICH): inference-compute defenses profit as
the model's training data better reflects the attacked data's components. We
empirically support this hypothesis across vision language model and attack
types, finding robustness gains from test-time compute if specification
following on OOD data is unlocked by compositional generalization, while RL
finetuning and protracted reasoning are not critical. For example, increasing
emphasis on defensive specifications via prompting lowers the success rate of
gradient-based multimodal attacks on VLMs robustified by adversarial
pretraining, but this same intervention provides no such benefit to
not-robustified models. This correlation of inference-compute's robustness
benefit with base model robustness is the rich-get-richer dynamic of the RICH:
attacked data components are more ID for robustified models, aiding
compositional generalization to OOD data. Accordingly, we advise layering
train-time and test-time defenses to obtain their synergistic benefit.

### 10. [The Unreasonable Effectiveness of Randomized Representations in Online Continual Graph Learning](http://arxiv.org/pdf/2510.06819v1)

Authors: Giovanni Donghi, Daniele Zambon, Luca Pasa, Cesare Alippi, Nicolò Navarin

Catastrophic forgetting is one of the main obstacles for Online Continual
Graph Learning (OCGL), where nodes arrive one by one, distribution drifts may
occur at any time and offline training on task-specific subgraphs is not
feasible. In this work, we explore a surprisingly simple yet highly effective
approach for OCGL: we use a fixed, randomly initialized encoder to generate
robust and expressive node embeddings by aggregating neighborhood information,
training online only a lightweight classifier. By freezing the encoder, we
eliminate drifts of the representation parameters, a key source of forgetting,
obtaining embeddings that are both expressive and stable. When evaluated across
several OCGL benchmarks, despite its simplicity and lack of memory buffer, this
approach yields consistent gains over state-of-the-art methods, with surprising
improvements of up to 30% and performance often approaching that of the joint
offline-training upper bound. These results suggest that in OCGL, catastrophic
forgetting can be minimized without complex replay or regularization by
embracing architectural simplicity and stability.

### Neural and Evolutionary Computing

### 1. [Associative Memory Model with Neural Networks: Memorizing multiple images with one neuron](http://arxiv.org/pdf/2510.06542v1)

Authors: Hiroshi Inazawa

This paper presents a neural network model (associative memory model) for
memory and recall of images. In this model, only a single neuron can memorize
multi-images and when that neuron is activated, it is possible to recall all
the memorized images at the same time. The system is composed of a single
cluster of numerous neurons, referred to as the "Cue Ball," and multiple neural
network layers, collectively called the "Recall Net." One of the features of
this model is that several different images are stored simultaneously in one
neuron, and by presenting one of the images stored in that neuron, all stored
images are recalled. Furthermore, this model allows for complete recall of an
image even when an incomplete image is presented

### 2. [Neuromorphic Computing -- An Overview](http://arxiv.org/pdf/2510.06721v1)

Authors: Benedikt Jung, Maximilian Kalcher, Merlin Marinova, Piper Powell, Esma Sakalli

With traditional computing technologies reaching their limit, a new field has
emerged seeking to follow the example of the human brain into a new era:
neuromorphic computing. This paper provides an introduction to neuromorphic
computing, why this and other new computing systems are needed, and what
technologies currently exist in the neuromorphic field. It begins with a
general introduction into the history of traditional computing and its present
problems, and then proceeds to a general overview of neuromorphic systems. It
subsequently discusses the main technologies currently in development. For
completeness, the paper first discusses neuromorphic-style computing on
traditional hardware, and then discusses the two top branches of specialized
hardware in this field; neuromorphic chips and photonic systems. Both branches
are explained as well as their relative benefits and drawbacks. The paper
concludes with a summary and an outlook on the future.

### Networking and Internet Architecture

### 1. [Adaptive Semantic Communication for UAV/UGV Cooperative Path Planning](http://arxiv.org/pdf/2510.06901v1)

Authors: Fangzhou Zhao, Yao Sun, Jianglin Lan, Lan Zhang, Xuesong Liu, Muhammad Ali Imran

Effective path planning is fundamental to the coordination of unmanned aerial
vehicles (UAVs) and unmanned ground vehicles (UGVs) systems, particularly in
applications such as surveillance, navigation, and emergency response.
Combining UAVs' broad field of view with UGVs' ground-level operational
capability greatly improve the likelihood of successfully achieving task
objectives such as locating victims, monitoring target areas, or navigating
hazardous terrain. In complex environments, UAVs need to provide precise
environmental perception information for UGVs to optimize their routing policy.
However, due to severe interference and non-line-of-sight conditions, wireless
communication is often unstable in such complex environments, making it
difficult to support timely and accurate path planning for UAV-UGV
coordination. To this end, this paper proposes a semantic communication
(SemCom) framework to enhance UAV/UGV cooperative path planning under
unreliable wireless conditions. Unlike traditional methods that transmit raw
data, SemCom transmits only the key information for path planning, reducing
transmission volume without sacrificing accuracy. The proposed framework is
developed by defining key semantics for path planning and designing a
transceiver for meeting the requirements of UAV-UGV cooperative path planning.
Simulation results show that, compared to conventional SemCom transceivers, the
proposed transceiver significantly reduces data transmission volume while
maintaining path planning accuracy, thereby enhancing system collaboration
efficiency.

### 2. [Dynamic Control Aware Semantic Communication Enabled Image Transmission for Lunar Landing](http://arxiv.org/pdf/2510.06916v1)

Authors: Fangzhou Zhao, Yao Sun, Jianglin Lan, Muhammad Ali Imran

The primary challenge in autonomous lunar landing missions lies in the
unreliable local control system, which has limited capacity to handle
high-dynamic conditions, severely affecting landing precision and safety.
Recent advancements in lunar satellite communication make it possible to
establish a wireless link between lunar orbit satellites and the lunar lander.
This enables satellites to run high-performance autonomous landing algorithms,
improving landing accuracy while reducing the lander's computational and
storage load. Nevertheless, traditional communication paradigms are not
directly applicable due to significant temperature fluctuations on the lunar
surface, intense solar radiation, and severe interference caused by lunar dust
on hardware. The emerging technique of semantic communication (SemCom) offers
significant advantages in robustness and resource efficiency, particularly
under harsh channel conditions. In this paper, we introduce a novel SemCom
framework for transmitting images from the lander to satellites operating the
remote landing control system. The proposed encoder-decoder dynamically adjusts
the transmission strategy based on real-time feedback from the lander's control
algorithm, ensuring the accurate delivery of critical image features and
enhancing control reliability. We provide a rigorous theoretical analysis of
the conditions that improve the accuracy of the control algorithm and reduce
end-to-end transmission time under the proposed framework. Simulation results
demonstrate that our SemCom method significantly enhances autonomous landing
performance compared to traditional communication methods.

### 3. [Advantages of Global Entanglement-Distillation Policies in Quantum Repeater Chains](http://arxiv.org/pdf/2510.06737v1)

Authors: Iftach Yakar, Michael Ben-Or

Quantum repeaters are essential for achieving long-distance quantum
communication due to photon loss, which grows exponentially with the channel
distance. Current quantum repeater generations use entanglement distillation
protocols, where the decision of when to perform distillation depends on either
local or global knowledge. Recent approaches for quantum repeaters, such as
Mantri et al. (arXiv:2409.06152), consider using deterministic local decision
policies for entanglement distillation. We ask whether global deterministic
policies outperform local ones in terms of communication rate. We simulate
equidistant repeater chains, assisted by two-way classical communication, and
compare local and global policies for distillation decisions, spanning large
distances and varying network and hardware parameters. Our findings show that
global deterministic policies consistently outperform these local ones, and in
some cases, determine whether secret communication is possible. For large
repeater chains ($N>512$), global policies improve SKR by two orders of
magnitude. These results suggest that local distillation decisions in quantum
repeater chains may not be optimal, and may inform future protocol design.

### 4. [Memory-Augmented Generative AI for Real-time Wireless Prediction in Dynamic Industrial Environments](http://arxiv.org/pdf/2510.06884v1)

Authors: Rahul Gulia, Amlan Ganguly, Michael E. Kuhl, Ehsan Rashedi, Clark Hochgraf

Accurate and real-time prediction of wireless channel conditions,
particularly the Signal-to-Interference-plus-Noise Ratio (SINR), is a
foundational requirement for enabling Ultra-Reliable Low-Latency Communication
(URLLC) in highly dynamic Industry 4.0 environments. Traditional physics-based
or statistical models fail to cope with the spatio-temporal complexities
introduced by mobile obstacles and transient interference inherent to smart
warehouses. To address this, we introduce Evo-WISVA (Evolutionary Wireless
Infrastructure for Smart Warehouse using VAE), a novel synergistic deep
learning architecture that functions as a lightweight 2D predictive digital
twin of the radio environment. Evo-WISVA integrates a memory-augmented
Variational Autoencoder (VAE) featuring an Attention-driven Latent Memory
Module (LMM) for robust, context-aware spatial feature extraction, with a
Convolutional Long Short-Term Memory (ConvLSTM) network for precise temporal
forecasting and sequential refinement. The entire pipeline is optimized
end-to-end via a joint loss function, ensuring optimal feature alignment
between the generative and predictive components. Rigorous experimental
evaluation conducted on a high-fidelity ns-3-generated industrial warehouse
dataset demonstrates that Evo-WISVA significantly surpasses state-of-the-art
baselines, achieving up to a 47.6\% reduction in average reconstruction error.
Crucially, the model exhibits exceptional generalization capacity to unseen
environments with vastly increased dynamic complexity (up to ten simultaneously
moving obstacles) while maintaining amortized computational efficiency
essential for real-time deployment. Evo-WISVA establishes a foundational
technology for proactive wireless resource management, enabling autonomous
optimization and advancing the realization of predictive digital twins in
industrial communication networks.

### 5. [From Description to Detection: LLM based Extendable O-RAN Compliant Blind DoS Detection in 5G and Beyond](http://arxiv.org/pdf/2510.06530v1)

Authors: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

The quality and experience of mobile communication have significantly
improved with the introduction of 5G, and these improvements are expected to
continue beyond the 5G era. However, vulnerabilities in control-plane
protocols, such as Radio Resource Control (RRC) and Non-Access Stratum (NAS),
pose significant security threats, such as Blind Denial of Service (DoS)
attacks. Despite the availability of existing anomaly detection methods that
leverage rule-based systems or traditional machine learning methods, these
methods have several limitations, including the need for extensive training
data, predefined rules, and limited explainability. Addressing these
challenges, we propose a novel anomaly detection framework that leverages the
capabilities of Large Language Models (LLMs) in zero-shot mode with unordered
data and short natural language attack descriptions within the Open Radio
Access Network (O-RAN) architecture. We analyse robustness to prompt variation,
demonstrate the practicality of automating the attack descriptions and show
that detection quality relies on the semantic completeness of the description
rather than its phrasing or length. We utilise an RRC/NAS dataset to evaluate
the solution and provide an extensive comparison of open-source and proprietary
LLM implementations to demonstrate superior performance in attack detection. We
further validate the practicality of our framework within O-RAN's real-time
constraints, illustrating its potential for detecting other Layer-3 attacks.

### 6. [GNN-enhanced Traffic Anomaly Detection for Next-Generation SDN-Enabled Consumer Electronics](http://arxiv.org/pdf/2510.07109v1)

Authors: Guan-Yan Yang, Farn Wang, Kuo-Hui Yeh

Consumer electronics (CE) connected to the Internet of Things are susceptible
to various attacks, including DDoS and web-based threats, which can compromise
their functionality and facilitate remote hijacking. These vulnerabilities
allow attackers to exploit CE for broader system attacks while enabling the
propagation of malicious code across the CE network, resulting in device
failures. Existing deep learning-based traffic anomaly detection systems
exhibit high accuracy in traditional network environments but are often overly
complex and reliant on static infrastructure, necessitating manual
configuration and management. To address these limitations, we propose a
scalable network model that integrates Software-defined Networking (SDN) and
Compute First Networking (CFN) for next-generation CE networks. In this network
model, we propose a Graph Neural Networks-based Network Anomaly Detection
framework (GNN-NAD) that integrates SDN-based CE networks and enables the CFN
architecture. GNN-NAD uniquely fuses a static, vulnerability-aware attack graph
with dynamic traffic features, providing a holistic view of network security.
The core of the framework is a GNN model (GSAGE) for graph representation
learning, followed by a Random Forest (RF) classifier. This design (GSAGE+RF)
demonstrates superior performance compared to existing feature selection
methods. Experimental evaluations on CE environment reveal that GNN-NAD
achieves superior metrics in accuracy, recall, precision, and F1 score, even
with small sample sizes, exceeding the performance of current network anomaly
detection methods. This work advances the security and efficiency of
next-generation intelligent CE networks.

### 7. [A Genetic Algorithm Approach to Anti-Jamming UAV Swarm Behavior](http://arxiv.org/pdf/2510.07292v1)

Authors: Tiago Silva, António Grilo

In recent years, Unmanned Aerial Vehicles (UAVs) have brought a new true
revolution to military tactics. While UAVs already constitute an advantage when
operating alone, multi-UAV swarms expand the available possibilities, allowing
the UAVs to collaborate and support each other as a team to carry out a given
task. This entails the capability to exchange information related with
situation awareness and action coordination by means of a suitable wireless
communication technology. In such scenario, the adversary is expected to
disrupt communications by jamming the communication channel. The latter becomes
the Achilles heel of the swarm. While anti-jamming techniques constitute a well
covered topic in the literature, the use of intelligent swarm behaviors to
leverage those techniques is still an open research issue.
  This paper explores the use of Genetic Algorithms (GAs) to jointly optimize
UAV swarm formation, beam-steering antennas and traffic routing in order to
mitigate the effect of jamming in the main coordination channel, under the
assumption that a more robust and low data rate channel is used for formation
management signaling. Simulation results show the effectiveness of proposed
approach. However, the significant computational cost paves the way for further
research.

### Robotics

### 1. [RAISE: A self-driving laboratory for interfacial property formulation discovery](http://arxiv.org/pdf/2510.06546v1)

Authors: Mohammad Nazeri, Sheldon Mei, Jeffrey Watchorn, Alex Zhang, Erin Ng, Tao Wen, Abhijoy Mandal, Kevin Golovin, Alan Aspuru-Guzik, Frank Gu

Surface wettability is a critical design parameter for biomedical devices,
coatings, and textiles. Contact angle measurements quantify liquid-surface
interactions, which depend strongly on liquid formulation. Herein, we present
the Robotic Autonomous Imaging Surface Evaluator (RAISE), a closed-loop,
self-driving laboratory that is capable of linking liquid formulation
optimization with surface wettability assessment. RAISE comprises a full
experimental orchestrator with the ability of mixing liquid ingredients to
create varying formulation cocktails, transferring droplets of prepared
formulations to a high-throughput stage, and using a pick-and-place camera tool
for automated droplet image capture. The system also includes an automated
image processing pipeline to measure contact angles. This closed loop
experiment orchestrator is integrated with a Bayesian Optimization (BO) client,
which enables iterative exploration of new formulations based on previous
contact angle measurements to meet user-defined objectives. The system operates
in a high-throughput manner and can achieve a measurement rate of approximately
1 contact angle measurement per minute. Here we demonstrate RAISE can be used
to explore surfactant wettability and how surfactant combinations create
tunable formulations that compensate for purity-related variations.
Furthermore, multi-objective BO demonstrates how precise and optimal
formulations can be reached based on application-specific goals. The
optimization is guided by a desirability score, which prioritizes formulations
that are within target contact angle ranges, minimize surfactant usage and
reduce cost. This work demonstrates the capabilities of RAISE to autonomously
link liquid formulations to contact angle measurements in a closed-loop system,
using multi-objective BO to efficiently identify optimal formulations aligned
with researcher-defined criteria.

### 2. [Safe Obstacle-Free Guidance of Space Manipulators in Debris Removal Missions via Deep Reinforcement Learning](http://arxiv.org/pdf/2510.06566v1)

Authors: Vincent Lam, Robin Chhabra

The objective of this study is to develop a model-free workspace trajectory
planner for space manipulators using a Twin Delayed Deep Deterministic Policy
Gradient (TD3) agent to enable safe and reliable debris capture. A local
control strategy with singularity avoidance and manipulability enhancement is
employed to ensure stable execution. The manipulator must simultaneously track
a capture point on a non-cooperative target, avoid self-collisions, and prevent
unintended contact with the target. To address these challenges, we propose a
curriculum-based multi-critic network where one critic emphasizes accurate
tracking and the other enforces collision avoidance. A prioritized experience
replay buffer is also used to accelerate convergence and improve policy
robustness. The framework is evaluated on a simulated seven-degree-of-freedom
KUKA LBR iiwa mounted on a free-floating base in Matlab/Simulink, demonstrating
safe and adaptive trajectory generation for debris removal missions.

### 3. [Assist-As-Needed: Adaptive Multimodal Robotic Assistance for Medication Management in Dementia Care](http://arxiv.org/pdf/2510.06633v1)

Authors: Kruthika Gangaraju, Tanmayi Inaparthy, Jiaqi Yang, Yihao Zheng, Fengpei Yuan

People living with dementia (PLWDs) face progressively declining abilities in
medication management-from simple forgetfulness to complete task breakdown-yet
most assistive technologies fail to adapt to these changing needs. This
one-size-fits-all approach undermines autonomy, accelerates dependence, and
increases caregiver burden. Occupational therapy principles emphasize matching
assistance levels to individual capabilities: minimal reminders for those who
merely forget, spatial guidance for those who misplace items, and comprehensive
multimodal support for those requiring step-by-step instruction. However,
existing robotic systems lack this adaptive, graduated response framework
essential for maintaining PLWD independence. We present an adaptive multimodal
robotic framework using the Pepper robot that dynamically adjusts assistance
based on real-time assessment of user needs. Our system implements a
hierarchical intervention model progressing from (1) simple verbal reminders,
to (2) verbal + gestural cues, to (3) full multimodal guidance combining
physical navigation to medication locations with step-by-step verbal and
gestural instructions. Powered by LLM-driven interaction strategies and
multimodal sensing, the system continuously evaluates task states to provide
just-enough assistance-preserving autonomy while ensuring medication adherence.
We conducted a preliminary study with healthy adults and dementia care
stakeholders in a controlled lab setting, evaluating the system's usability,
comprehensibility, and appropriateness of adaptive feedback mechanisms. This
work contributes: (1) a theoretically grounded adaptive assistance framework
translating occupational therapy principles into HRI design, (2) a multimodal
robotic implementation that preserves PLWD dignity through graduated support,
and (3) empirical insights into stakeholder perceptions of adaptive robotic
care.

### 4. [RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training](http://arxiv.org/pdf/2510.06710v1)

Authors: Hongzhi Zang, Mingjie Wei, Si Xu, Yongji Wu, Zhen Guo, Yuanqing Wang, Hao Lin, Liangzhi Shi, Yuqing Xie, Zhexuan Xu, Zhihao Liu, Kang Chen, Wenhao Tang, Quanlu Zhang, Weinan Zhang, Chao Yu, Yu Wang

Recent progress in vision and language foundation models has significantly
advanced multimodal understanding, reasoning, and generation, inspiring a surge
of interest in extending such capabilities to embodied settings through
vision-language-action (VLA) models. Yet, most VLA models are still trained
with supervised fine-tuning (SFT), which struggles to generalize under
distribution shifts due to error accumulation. Reinforcement learning (RL)
offers a promising alternative by directly optimizing task performance through
interaction, but existing attempts remain fragmented and lack a unified
platform for fair and systematic comparison across model architectures and
algorithmic designs. To address this gap, we introduce RLinf-VLA, a unified and
efficient framework for scalable RL training of VLA models. The system adopts a
highly flexible resource allocation design that addresses the challenge of
integrating rendering, training, and inference in RL+VLA training. In
particular, for GPU-parallelized simulators, RLinf-VLA implements a novel
hybrid fine-grained pipeline allocation mode, achieving a 1.61x-1.88x speedup
in training. Through a unified interface, RLinf-VLA seamlessly supports diverse
VLA architectures (e.g., OpenVLA, OpenVLA-OFT), multiple RL algorithms (e.g.,
PPO, GRPO), and various simulators (e.g., ManiSkill, LIBERO). In simulation, a
unified model achieves 98.11\% across 130 LIBERO tasks and 97.66\% across 25
ManiSkill tasks. Beyond empirical performance, our study distills a set of best
practices for applying RL to VLA training and sheds light on emerging patterns
in this integration. Furthermore, we present preliminary deployment on a
real-world Franka robot, where RL-trained policies exhibit stronger
generalization than those trained with SFT. We envision RLinf-VLA as a
foundation to accelerate and standardize research on embodied intelligence.

### 5. [SanDRA: Safe Large-Language-Model-Based Decision Making for Automated Vehicles Using Reachability Analysis](http://arxiv.org/pdf/2510.06717v1)

Authors: Yuanfei Lin, Sebastian Illing, Matthias Althoff

Large language models have been widely applied to knowledge-driven
decision-making for automated vehicles due to their strong generalization and
reasoning capabilities. However, the safety of the resulting decisions cannot
be ensured due to possible hallucinations and the lack of integrated vehicle
dynamics. To address this issue, we propose SanDRA, the first safe
large-language-model-based decision making framework for automated vehicles
using reachability analysis. Our approach starts with a comprehensive
description of the driving scenario to prompt large language models to generate
and rank feasible driving actions. These actions are translated into temporal
logic formulas that incorporate formalized traffic rules, and are subsequently
integrated into reachability analysis to eliminate unsafe actions. We validate
our approach in both open-loop and closed-loop driving environments using
off-the-shelf and finetuned large language models, showing that it can provide
provably safe and, where possible, legally compliant driving actions, even
under high-density traffic conditions. To ensure transparency and facilitate
future research, all code and experimental setups are publicly available at
github.com/CommonRoad/SanDRA.

### 6. [Distributed 3D Source Seeking via SO(3) Geometric Control of Robot Swarms](http://arxiv.org/pdf/2510.06836v1)

Authors: Jesús Bautista, Héctor García de Marina

This paper presents a geometric control framework on the Lie group SO(3) for
3D source-seeking by robots with first-order attitude dynamics and constant
translational speed. By working directly on SO(3), the approach avoids
Euler-angle singularities and quaternion ambiguities, providing a unique,
intrinsic representation of orientation. We design a proportional feed-forward
controller that ensures exponential alignment of each agent to an estimated
ascending direction toward a 3D scalar field source. The controller adapts to
bounded unknown variations and preserves well-posed swarm formations. Numerical
simulations demonstrate the effectiveness of the method, with all code provided
open source for reproducibility.

### 7. [Temporal-Prior-Guided View Planning for Periodic 3D Plant Reconstruction](http://arxiv.org/pdf/2510.07028v1)

Authors: Sicong Pan, Xuying Huang, Maren Bennewitz

Periodic 3D reconstruction is essential for crop monitoring, but costly when
each cycle restarts from scratch, wasting resources and ignoring information
from previous captures. We propose temporal-prior-guided view planning for
periodic plant reconstruction, in which a previously reconstructed model of the
same plant is non-rigidly aligned to a new partial observation to form an
approximation of the current geometry. To accommodate plant growth, we inflate
this approximation and solve a set covering optimization problem to compute a
minimal set of views. We integrated this method into a complete pipeline that
acquires one additional next-best view before registration for robustness and
then plans a globally shortest path to connect the planned set of views and
outputs the best view sequence. Experiments on maize and tomato under
hemisphere and sphere view spaces show that our system maintains or improves
surface coverage while requiring fewer views and comparable movement cost
compared to state-of-the-art baselines.

### 8. [Diffusing Trajectory Optimization Problems for Recovery During Multi-Finger Manipulation](http://arxiv.org/pdf/2510.07030v1)

Authors: Abhinav Kumar, Fan Yang, Sergio Aguilera Marinovic, Soshi Iba, Rana Soltani Zarrin, Dmitry Berenson

Multi-fingered hands are emerging as powerful platforms for performing fine
manipulation tasks, including tool use. However, environmental perturbations or
execution errors can impede task performance, motivating the use of recovery
behaviors that enable normal task execution to resume. In this work, we take
advantage of recent advances in diffusion models to construct a framework that
autonomously identifies when recovery is necessary and optimizes contact-rich
trajectories to recover. We use a diffusion model trained on the task to
estimate when states are not conducive to task execution, framed as an
out-of-distribution detection problem. We then use diffusion sampling to
project these states in-distribution and use trajectory optimization to plan
contact-rich recovery trajectories. We also propose a novel diffusion-based
approach that distills this process to efficiently diffuse the full
parameterization, including constraints, goal state, and initialization, of the
recovery trajectory optimization problem, saving time during online execution.
We compare our method to a reinforcement learning baseline and other methods
that do not explicitly plan contact interactions, including on a hardware
screwdriver-turning task where we show that recovering using our method
improves task performance by 96% and that ours is the only method evaluated
that can attempt recovery without causing catastrophic task failure. Videos can
be found at https://dtourrecovery.github.io/.

### 9. [Bring the Apple, Not the Sofa: Impact of Irrelevant Context in Embodied AI Commands on VLA Models](http://arxiv.org/pdf/2510.07067v1)

Authors: Daria Pugacheva, Andrey Moskalenko, Denis Shepelev, Andrey Kuznetsov, Vlad Shakhuro, Elena Tutubalina

Vision Language Action (VLA) models are widely used in Embodied AI, enabling
robots to interpret and execute language instructions. However, their
robustness to natural language variability in real-world scenarios has not been
thoroughly investigated. In this work, we present a novel systematic study of
the robustness of state-of-the-art VLA models under linguistic perturbations.
Specifically, we evaluate model performance under two types of instruction
noise: (1) human-generated paraphrasing and (2) the addition of irrelevant
context. We further categorize irrelevant contexts into two groups according to
their length and their semantic and lexical proximity to robot commands. In
this study, we observe consistent performance degradation as context size
expands. We also demonstrate that the model can exhibit relative robustness to
random context, with a performance drop within 10%, while semantically and
lexically similar context of the same length can trigger a quality decline of
around 50%. Human paraphrases of instructions lead to a drop of nearly 20%. To
mitigate this, we propose an LLM-based filtering framework that extracts core
commands from noisy inputs. Incorporating our filtering step allows models to
recover up to 98.5% of their original performance under noisy conditions.

### 10. [Sampling Strategies for Robust Universal Quadrupedal Locomotion Policies](http://arxiv.org/pdf/2510.07094v1)

Authors: David Rytz, Kim Tien Ly, Ioannis Havoutis

This work focuses on sampling strategies of configuration variations for
generating robust universal locomotion policies for quadrupedal robots. We
investigate the effects of sampling physical robot parameters and joint
proportional-derivative gains to enable training a single reinforcement
learning policy that generalizes to multiple parameter configurations. Three
fundamental joint gain sampling strategies are compared: parameter sampling
with (1) linear and polynomial function mappings of mass-to-gains, (2)
performance-based adaptive filtering, and (3) uniform random sampling. We
improve the robustness of the policy by biasing the configurations using
nominal priors and reference models. All training was conducted on RaiSim,
tested in simulation on a range of diverse quadrupeds, and zero-shot deployed
onto hardware using the ANYmal quadruped robot. Compared to multiple baseline
implementations, our results demonstrate the need for significant joint
controller gains randomization for robust closing of the sim-to-real gap.

### Software Engineering

### 1. [Beyond More Context: How Granularity and Order Drive Code Completion Quality](http://arxiv.org/pdf/2510.06606v1)

Authors: Uswat Yusuf, Genevieve Caumartin, Diego Elias Costa

Context plays an important role in the quality of code completion, as Large
Language Models (LLMs) require sufficient and relevant information to assist
developers in code generation tasks. However, composing a relevant context for
code completion poses challenges in large repositories: First, the limited
context length of LLMs makes it impractical to include all repository files.
Second, the quality of generated code is highly sensitive to noisy or
irrelevant context. In this paper, we present our approach for the ASE 2025
Context Collection Challenge. The challenge entails outperforming JetBrains
baselines by designing effective retrieval and context collection strategies.
We develop and evaluate a series of experiments that involve retrieval
strategies at both the file and chunk levels. We focus our initial experiments
on examining the impact of context size and file ordering on LLM performance.
Our results show that the amount and order of context can significantly
influence the performance of the models. We introduce chunk-based retrieval
using static analysis, achieving a 6% improvement over our best file-retrieval
strategy and a 16% improvement over the no-context baseline for Python in the
initial phase of the competition. Our results highlight the importance of
retrieval granularity, ordering and hybrid strategies in developing effective
context collection pipelines for real-world development scenarios.

### 2. [Oops!... I did it again. Conclusion (In-)Stability in Quantitative Empirical Software Engineering: A Large-Scale Analysis](http://arxiv.org/pdf/2510.06844v1)

Authors: Nicole Hoess, Carlos Paradis, Rick Kazman, Wolfgang Mauerer

Context: Mining software repositories is a popular means to gain insights
into a software project's evolution, monitor project health, support decisions
and derive best practices. Tools supporting the mining process are commonly
applied by researchers and practitioners, but their limitations and agreement
are often not well understood.
  Objective: This study investigates some threats to validity in complex tool
pipelines for evolutionary software analyses and evaluates the tools' agreement
in terms of data, study outcomes and conclusions for the same research
questions.
  Method: We conduct a lightweight literature review to select three studies on
collaboration and coordination, software maintenance and software quality from
high-ranked venues, which we formally replicate with four independent,
systematically selected mining tools to quantitatively and qualitatively
compare the extracted data, analysis results and conclusions.
  Results: We find that numerous technical details in tool design and
implementation accumulate along the complex mining pipelines and can cause
substantial differences in the extracted baseline data, its derivatives,
subsequent results of statistical analyses and, under specific circumstances,
conclusions.
  Conclusions: Users must carefully choose tools and evaluate their limitations
to assess the scope of validity in an adequate way. Reusing tools is
recommended. Researchers and tool authors can promote reusability and help
reducing uncertainties by reproduction packages and comparative studies
following our approach.

### 3. [An empirical study on declined proposals: why are these proposals declined?](http://arxiv.org/pdf/2510.06984v1)

Authors: Masanari Kondo, Mahmoud Alfadel, Shane McIntosh, Yasutaka Kamei, Naoyasu Ubayashi

Design-level decisions in open-source software (OSS) projects are often made
through structured mechanisms such as proposals, which require substantial
community discussion and review. Despite their importance, the proposal process
is resource-intensive and often leads to contributor frustration, especially
when proposals are declined without clear feedback. Yet, the reasons behind
proposal rejection remain poorly understood, limiting opportunities to
streamline the process or guide contributors effectively. This study
investigates the characteristics and outcomes of proposals in the Go
programming language to understand why proposals are declined and how such
outcomes might be anticipated. We conduct a mixed-method empirical study on
1,091 proposals submitted to the Go project. We quantify proposal outcomes,
build a taxonomy of decline reasons, and evaluate large language models (LLMs)
for predicting these outcomes. We find that proposals are more often declined
than accepted, and resolution typically takes over a month. Only 14.7% of
declined proposals are ever resubmitted. Through qualitative coding, we
identify nine key reasons for proposal decline, such as duplication, limited
use cases, or violations of project principles. This taxonomy can help
contributors address issues in advance, e.g., checking for existing
alternatives can reduce redundancy. We also demonstrate that GPT-based models
can predict decline decisions early in the discussion (F1 score = 0.71 with
partial comments), offering a practical tool for prioritizing review effort.
Our findings reveal inefficiencies in the proposal process and highlight
actionable opportunities for improving both contributor experience and reviewer
workload by enabling early triage and guiding contributors to strengthen their
proposals using a structured understanding of past decline reasons.

### 4. [Human-aligned AI Model Cards with Weighted Hierarchy Architecture](http://arxiv.org/pdf/2510.06989v1)

Authors: Pengyue Yang, Haolin Jin, Qingwen Zeng, Jiawen Wen, Harry Rao, Huaming Chen

The proliferation of Large Language Models (LLMs) has led to a burgeoning
ecosystem of specialized, domain-specific models. While this rapid growth
accelerates innovation, it has simultaneously created significant challenges in
model discovery and adoption. Users struggle to navigate this landscape due to
inconsistent, incomplete, and imbalanced documentation across platforms.
Existing documentation frameworks, such as Model Cards and FactSheets, attempt
to standardize reporting but are often static, predominantly qualitative, and
lack the quantitative mechanisms needed for rigorous cross-model comparison.
This gap exacerbates model underutilization and hinders responsible adoption.
To address these shortcomings, we introduce the Comprehensive Responsible AI
Model Card Framework (CRAI-MCF), a novel approach that transitions from static
disclosures to actionable, human-aligned documentation. Grounded in Value
Sensitive Design (VSD), CRAI-MCF is built upon an empirical analysis of 240
open-source projects, distilling 217 parameters into an eight-module,
value-aligned architecture. Our framework introduces a quantitative sufficiency
criterion to operationalize evaluation and enables rigorous cross-model
comparison under a unified scheme. By balancing technical, ethical, and
operational dimensions, CRAI-MCF empowers practitioners to efficiently assess,
select, and adopt LLMs with greater confidence and operational integrity.

### 5. [Building an Open AIBOM Standard in the Wild](http://arxiv.org/pdf/2510.07070v1)

Authors: Gopi Krishnan Rajbahadur, Keheliya Gallaba, Elyas Rashno, Arthit Suriyawongkul, Karen Bennet, Kate Stewart, Ahmed E. Hassan

Modern software engineering increasingly relies on open, community-driven
standards, yet how such standards are created in fast-evolving domains like
AI-powered systems remains underexplored. This paper presents a detailed
experience report on the development of the AI Bill of Materials AIBOM
specification, an extension of the ISO/IEC 5962:2021 Software Package Data
Exchange (SPDX) software bill of materials (SBOM) standard, which captures AI
components such as datasets and iterative training artifacts. Framed through
the lens of Action Research (AR), we document a global, multi-stakeholder
effort involving over 90 contributors and structured AR cycles. The resulting
specification was validated through four complementary approaches: alignment
with major regulations and ethical standards (e.g., EU AI Act and IEEE 7000
standards), systematic mapping to six industry use cases, semi-structured
practitioner interviews, and an industrial case study. Beyond delivering a
validated artefact, our paper documents the process of building the AIBOM
specification in the wild, and reflects on how it aligns with the AR cycle, and
distills lessons that can inform future standardization efforts in the software
engineering community.

### 6. [Prompt, Synthesize, Fine-Tune: A Secure Code Generation Recipe](http://arxiv.org/pdf/2510.07189v1)

Authors: Junjie Li, Fazle Rabbi, Bo Yang, Song Wang, Jinqiu Yang

Although Large Language Models (LLMs) show promising solutions to automated
code generation, they often produce insecure code that threatens software
security. Current approaches (e.g., SafeCoder) to improve secure code
generation suffer from limited and imbalanced datasets, reducing their
effectiveness and generalizability. In this work, we present Secure-Instruct, a
novel framework that automatically synthesizes high-quality vulnerable and
secure code examples, generates fine-tuning instructions, and instruction-tunes
LLMs to align task description and secure code generation abilities. We
evaluate Secure-Instruct on four representative LLMs using two benchmarks: our
own CWEBench and the existing CWEval. CWEBench comprises 93 scenarios on 44
CWEs, all without overlap with Secure-Instruct's synthetic instruction-tuning
dataset, while CWEval covers 31 CWEs with 119 manually verified
security-critical tasks. We find that Secure-Instruct improves not only the
security but also the functional correctness of the generated code. On
CWEBench, Secure-Instruct substantially improves secure code generation, giving
a 14.3% average increase in secure ratio over the pretrained models and
outperforms SafeCoder by 7.6%. On CWEval, Secure-Instruct achieves a 14%
increase for CodeLlama-7B and 5.8% for Mistral-7B in Func-Sec@1 over pretrained
models, and surpasses SafeCoder by 15.8% and 6.8% respectively.

### 7. [AISysRev -- LLM-based Tool for Title-abstract Screening](http://arxiv.org/pdf/2510.06708v1)

Authors: Aleksi Huotala, Miikka Kuutila, Olli-Pekka Turtio, Mika Mäntylä

Systematic reviews are a standard practice for summarizing the state of
evidence in software engineering. Conducting systematic reviews is laborious,
especially during the screening or study selection phase, where the number of
papers can be overwhelming. During this phase, papers are assessed against
inclusion and exclusion criteria based on their titles and abstracts. Recent
research has demonstrated that large language models (LLMs) can perform
title-abstract screening at a level comparable to that of a master's student.
While LLMs cannot be fully trusted, they can help, for example, in Rapid
Reviews, which try to expedite the review process. Building on recent research,
we developed AiSysRev, an LLM-based screening tool implemented as a web
application running in a Docker container. The tool accepts a CSV file
containing paper titles and abstracts. Users specify inclusion and exclusion
criteria. One can use multiple LLMs for screening via OpenRouter. AiSysRev
supports both zero-shot and few-shot screening, and also allows for manual
screening through interfaces that display LLM results as guidance for human
reviewers.We conducted a trial study with 137 papers using the tool. Our
findings indicate that papers can be classified into four categories: Easy
Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary
cases, where LLMs are prone to errors, highlight the need for human
intervention. While LLMs do not replace human judgment in systematic reviews,
they can significantly reduce the burden of assessing large volumes of
scientific literature. Video: https://www.youtube.com/watch?v=jVbEj4Y4tQI Tool:
https://github.com/EvoTestOps/AISysRev

### 8. [LLM Company Policies and Policy Implications in Software Organizations](http://arxiv.org/pdf/2510.06718v1)

Authors: Ranim Khojah, Mazen Mohamad, Linda Erlenhov, Francisco Gomes de Oliveira Neto, Philipp Leitner

The risks associated with adopting large language model (LLM) chatbots in
software organizations highlight the need for clear policies. We examine how 11
companies create these policies and the factors that influence them, aiming to
help managers safely integrate chatbots into development workflows.

### 9. [Early Results from Teaching Modelling for Software Comprehension in New-Hire Onboarding](http://arxiv.org/pdf/2510.07010v1)

Authors: Mrityunjay Kumar, Venkatesh Choppella

Working effectively with large, existing software systems requires strong
comprehension skills, yet most graduates enter the industry with little
preparation for this challenge. We report early results from a pilot
intervention integrated into a SaaS company's onboarding program: a
five-session course introducing systems thinking and Labelled Transition System
(LTS) modelling. Participants articulated their understanding of product
behaviour using a structured template and completed matched pre- and
post-assessments. Of 35 new hires, 31 provided paired records for analysis.
Across the full cohort, gains were small and not statistically significant.
However, participants below the median on the pre-test improved by 15
percentage points on average (statistically significant), while those above the
median regressed slightly (not statistically significant). Course feedback
indicated high engagement and perceived applicability. These results suggest
that short, modelling-focused onboarding interventions can accelerate
comprehension for less-prepared new hires. At the same time, they point to the
need for differentiated pathways for stronger participants, and to the
potential for companies to adopt such interventions at scale as a low-cost
complement to existing onboarding.

### 10. [Automated Discovery of Test Oracles for Database Management Systems Using LLMs](http://arxiv.org/pdf/2510.06663v1)

Authors: Qiuyang Mang, Runyuan He, Suyang Zhong, Xiaoxuan Liu, Huanchen Zhang, Alvin Cheung

Since 2020, automated testing for Database Management Systems (DBMSs) has
flourished, uncovering hundreds of bugs in widely-used systems. A cornerstone
of these techniques is test oracle, which typically implements a mechanism to
generate equivalent query pairs, thereby identifying bugs by checking the
consistency between their results. However, while applying these oracles can be
automated, their design remains a fundamentally manual endeavor. This paper
explores the use of large language models (LLMs) to automate the discovery and
instantiation of test oracles, addressing a long-standing bottleneck towards
fully automated DBMS testing. Although LLMs demonstrate impressive creativity,
they are prone to hallucinations that can produce numerous false positive bug
reports. Furthermore, their significant monetary cost and latency mean that LLM
invocations should be limited to ensure that bug detection is efficient and
economical.
  To this end, we introduce Argus, a novel framework built upon the core
concept of the Constrained Abstract Query - a SQL skeleton containing
placeholders and their associated instantiation conditions (e.g., requiring a
placeholder to be filled by a boolean column). Argus uses LLMs to generate
pairs of these skeletons that are asserted to be semantically equivalent. This
equivalence is then formally proven using a SQL equivalence solver to ensure
soundness. Finally, the placeholders within the verified skeletons are
instantiated with concrete, reusable SQL snippets that are also synthesized by
LLMs to efficiently produce complex test cases. We implemented Argus and
evaluated it on five extensively tested DBMSs, discovering 40 previously
unknown bugs, 35 of which are logic bugs, with 36 confirmed and 26 already
fixed by the developers.

### Social and Information Networks

### 1. [Visualization of Interpersonal Communication using Indoor Positioning Technology with UWB Tags](http://arxiv.org/pdf/2510.06797v1)

Authors: Hayato Shinto, Yu Ohki, Kenji Mizumoto, Kei Saito

In conjunction with a social gathering held on a university campus, the
movement of attendees were tracked within the venue for approximately two hours
using a UWB indoor positioning system, in order to visualize their
interpersonal communication. Network and community analyses were performed on
attendee interaction data, and the evolution of communities over time was
further investigated through repeated community analysis at different time
points. Furthermore, recognizing the influence of distance thresholds on
defining contact, we discussed how varying these thresholds affected the
resulting network structure and community analysis outcomes. This study
confirmed that the temporal evolution of communities identified through
community analysis broadly corresponded with the visually observed groupings of
participants using the UWB indoor positioning system.

### 2. [Unpacking Discourses on Childbirth and Parenthood in Popular Social Media Platforms Across China, Japan, and South Korea](http://arxiv.org/pdf/2510.06788v1)

Authors: Zheng Wei, Yunqi Li, Yucheng He, Yuelu Li, Xian Xu, Huamin Qu, Pan Hui, Muzhi Zhou

Social media use has been shown to be associated with low fertility desires.
However, we know little about the discourses surrounding childbirth and
parenthood that people consume online. We analyze 219,127 comments on 668 short
videos related to reproduction and parenthood from Douyin and Tiktok in China,
South Korea, and Japan, a region famous for its extremely low fertility level,
to examine the topics and sentiment expressed online. BERTopic model is used to
assist thematic analysis, and a large language model QWen is applied to label
sentiment. We find that comments focus on childrearing costs in all countries,
utility of children, particularly in Japan and South Korea, and individualism,
primarily in China. Comments from Douyin exhibit the strongest anti-natalist
sentiments, while the Japanese and Korean comments are more neutral. Short
video characteristics, such as their stances or account type, significantly
influence the responses, alongside regional socioeconomic indicators, including
GDP, urbanization, and population sex ratio. This work provides one of the
first comprehensive analyses of online discourses on family formation via
popular algorithm-fed video sharing platforms in regions experiencing low
fertility rates, making a valuable contribution to our understanding of the
spread of family values online.

### 3. [Machines in the Crowd? Measuring the Footprint of Machine-Generated Text on Reddit](http://arxiv.org/pdf/2510.07226v1)

Authors: Lucio La Cava, Luca Maria Aiello, Andrea Tagarelli

Generative Artificial Intelligence is reshaping online communication by
enabling large-scale production of Machine-Generated Text (MGT) at low cost.
While its presence is rapidly growing across the Web, little is known about how
MGT integrates into social media environments. In this paper, we present the
first large-scale characterization of MGT on Reddit. Using a state-of-the-art
statistical method for detection of MGT, we analyze over two years of activity
(2022-2024) across 51 subreddits representative of Reddit's main community
types such as information seeking, social support, and discussion. We study the
concentration of MGT across communities and over time, and compared MGT to
human-authored text in terms of social signals it expresses and engagement it
receives. Our very conservative estimate of MGT prevalence indicates that
synthetic text is marginally present on Reddit, but it can reach peaks of up to
9% in some communities in some months. MGT is unevenly distributed across
communities, more prevalent in subreddits focused on technical knowledge and
social support, and often concentrated in the activity of a small fraction of
users. MGT also conveys distinct social signals of warmth and status giving
typical of language of AI assistants. Despite these stylistic differences, MGT
achieves engagement levels comparable than human-authored content and in a few
cases even higher, suggesting that AI-generated text is becoming an organic
component of online social discourse. This work offers the first perspective on
the MGT footprint on Reddit, paving the way for new investigations involving
platform governance, detection strategies, and community dynamics.

### 4. [Spectral Graph Clustering under Differential Privacy: Balancing Privacy, Accuracy, and Efficiency](http://arxiv.org/pdf/2510.07136v1)

Authors: Mohamed Seif, Antti Koskela, H. Vincent Poor, Andrea J. Goldsmith

We study the problem of spectral graph clustering under edge differential
privacy (DP). Specifically, we develop three mechanisms: (i) graph perturbation
via randomized edge flipping combined with adjacency matrix shuffling, which
enforces edge privacy while preserving key spectral properties of the graph.
Importantly, shuffling considerably amplifies the guarantees: whereas flipping
edges with a fixed probability alone provides only a constant epsilon edge DP
guarantee as the number of nodes grows, the shuffled mechanism achieves
(epsilon, delta) edge DP with parameters that tend to zero as the number of
nodes increase; (ii) private graph projection with additive Gaussian noise in a
lower-dimensional space to reduce dimensionality and computational complexity;
and (iii) a noisy power iteration method that distributes Gaussian noise across
iterations to ensure edge DP while maintaining convergence. Our analysis
provides rigorous privacy guarantees and a precise characterization of the
misclassification error rate. Experiments on synthetic and real-world networks
validate our theoretical analysis and illustrate the practical privacy-utility
trade-offs.

### Systems and Control

### 1. [Model Predictive Path Integral Control for Roll-to-Roll Manufacturing](http://arxiv.org/pdf/2510.06547v1)

Authors: Christopher Martin, Apurva Patil, Wei Li, Takashi Tanaka, Dongmei Chen

Roll-to-roll (R2R) manufacturing is a continuous processing technology
essential for scalable production of thin-film materials and printed
electronics, but precise control remains challenging due to subsystem
interactions, nonlinearities, and process disturbances. This paper proposes a
Model Predictive Path Integral (MPPI) control formulation for R2R systems,
leveraging a GPU-based Monte-Carlo sampling approach to efficiently approximate
optimal controls online. Crucially, MPPI easily handles non-differentiable cost
functions, enabling the incorporation of complex performance criteria relevant
to advanced manufacturing processes. A case study is presented that
demonstrates that MPPI significantly improves tension regulation performance
compared to conventional model predictive control (MPC), highlighting its
suitability for real-time control in advanced manufacturing.

### 2. [A Cascade of Systems and the Product of Their $θ$-Symmetric Scaled Relative Graphs](http://arxiv.org/pdf/2510.06583v1)

Authors: Xiaokan Yang, Ding Zhang, Wei Chen, Li Qiu

In this paper, we utilize a variant of the scaled relative graph (SRG),
referred to as the $\theta$-symmetric SRG, to develop a graphical stability
criterion for the feedback interconnection of a cascade of systems. A crucial
submultiplicative property of $\theta$-symmetric SRG is established, enabling
it to handle cyclic interconnections for which conventional graph separation
methods are not applicable. By integrating both gain and refined phase
information, the $\theta$-symmetric SRG provides a unified graphical
characterization of the system, which better captures system properties and
yields less conservative results. In the scalar case, the $\theta$-symmetric
SRG can be reduced exactly to the scalar itself, whereas the standard SRG
appears to be a conjugate pair. Consequently, the frequency-wise
$\theta$-symmetric SRG is more suitable than the standard SRG as a multi-input
multi-output extension of the classical Nyquist plot. Illustrative examples are
included to demonstrate the effectiveness of the $\theta$-symmetric SRG.

### 3. [Resilient Multi-Dimensional Consensus and Distributed Optimization against Agent-Based and Denial-of-Service Attacks](http://arxiv.org/pdf/2510.06835v1)

Authors: Hongjian Chen, Changyun Wen, Xiaolei Li, Jiaqi Yan

In this paper, we consider the resilient multi-dimensional consensus and
distributed optimization problems of multi-agent systems (MASs) in the presence
of both agent-based and denial-of-service (DoS) attacks. The considered
agent-based attacks can cover malicious, Byzantine, and stubborn agents. The
links between agents in the network can be blocked by DoS attacks, which may
lead the digraph to be time-varying and even disconnected. The objective is to
ensure that the remaining benign agents achieve consensus. To this end, an
"auxiliary point"-based resilient control algorithm is proposed for MASs. Under
the proposed algorithm, each healthy agent constructs a "safe kernel" utilizing
the states of its in-neighbors and updates its state toward a specific point
within this kernel at each iteration. If an agent cannot receive its neighbors'
states owing to DoS attacks, it will use the states received immediately before
the DoS period. Moreover, a resilient multi-dimensional distributed
optimization (RMDO) algorithm is also proposed. Theoretical proofs and
numerical examples are presented to demonstrate the effectiveness of the
proposed algorithms.

### 4. [Decentralized CBF-based Safety Filters for Collision Avoidance of Cooperative Missile Systems with Input Constraints](http://arxiv.org/pdf/2510.06846v1)

Authors: Johannes Autenrieb, Mark Spiller

This paper presents a decentralized safety filter for collision avoidance in
multi-agent aerospace interception scenarios. The approach leverages robust
control barrier functions (RCBFs) to guarantee forward invariance of safety
sets under bounded inputs and high-relative-degree dynamics. Each effector
executes its nominal cooperative guidance command, while a local quadratic
program (QP) modifies the input only when necessary. Event-triggered activation
based on range and zero-effort miss (ZEM) criteria ensures scalability by
restricting active constraints to relevant neighbors. To resolve feasibility
issues from simultaneous constraints, a slack-variable relaxation scheme is
introduced that prioritizes critical agents in a Pareto-optimal manner.
Simulation results in many-on-many interception scenarios demonstrate that the
proposed framework maintains collision-free operation with minimal deviation
from nominal guidance, providing a computationally efficient and scalable
solution for safety-critical multi-agent aerospace systems.

### 5. [Mitigating Increase-Decrease Gaming with Alternative Connection Agreements: A Defender-Attacker-Defender Game](http://arxiv.org/pdf/2510.07102v1)

Authors: Bart van der Holst, Thomas Swarts, Phuong Nguyen, Johan Morren, Koen Kok

Redispatch markets are widely used by system operators to manage network
congestion. A well-known drawback, however, is that Flexibility Service
Providers (FSPs) may strategically adjust their baselines in anticipation of
redispatch actions, thereby aggravating congestion and raising system costs. To
address this increase-decrease gaming, Distribution System Operators (DSOs)
could use Alternative Connection Agreements (ACAs) to conditionally limit the
available connection capacity of market participants in the day-ahead stage. In
this paper, we present a novel Defender-Attacker-Defender game to investigate
the potential of this approach in distribution networks under load and price
uncertainty. We solve the resulting trilevel optimization model using a custom
branch-and-bound algorithm, and we demonstrate that it efficiently solves the
problem without exploring many nodes in the branch-and-bound search tree for
most simulated scenarios. The case study demonstrates that applying ACAs can
substantially lower redispatch costs (e.g. by 25%) for the DSO with only a
limited impact on FSP profits. The effectiveness of the approach critically
depends on how often the DSO can invoke ACAs and on the extent to which the DSO
can anticipate strategic bidding behavior of the FSP.

### 6. [Safe Stabilization of the Stefan Problem with a High-Order Moving Boundary Dynamics by PDE Backstepping](http://arxiv.org/pdf/2510.06571v1)

Authors: Shumon Koga, Miroslav Krstic

This paper presents a safe stabilization of the Stefan PDE model with a
moving boundary governed by a high-order dynamics. We consider a parabolic PDE
with a time-varying domain governed by a second-order response with respect to
the Neumann boundary value of the PDE state at the moving boundary. The
objective is to design a boundary heat flux control to stabilize the moving
boundary at a desired setpoint, with satisfying the required conditions of the
model on PDE state and the moving boundary. We apply a PDE backstepping method
for the control design with considering a constraint on the control law. The
PDE and moving boundary constraints are shown to be satisfied by applying the
maximum principle for parabolic PDEs. Then the closed-loop system is shown to
be globally exponentially stable by performing Lyapunov analysis. The proposed
control is implemented in numerical simulation, which illustrates the desired
performance in safety and stability. An outline of the extension to third-order
moving boundary dynamics is also presented. Code is released at
https://github.com/shumon0423/HighOrderStefan_CDC2025.git.

### 7. [Delay Independent Safe Control with Neural Networks: Positive Lur'e Certificates for Risk Aware Autonomy](http://arxiv.org/pdf/2510.06661v1)

Authors: Hamidreza Montazeri Hedesh, Milad Siami

We present a risk-aware safety certification method for autonomous, learning
enabled control systems. Focusing on two realistic risks, state/input delays
and interval matrix uncertainty, we model the neural network (NN) controller
with local sector bounds and exploit positivity structure to derive linear,
delay-independent certificates that guarantee local exponential stability
across admissible uncertainties. To benchmark performance, we adopt and
implement a state-of-the-art IQC NN verification pipeline. On representative
cases, our positivity-based tests run orders of magnitude faster than SDP-based
IQC while certifying regimes the latter cannot-providing scalable safety
guarantees that complement risk-aware control.

### 8. [Falsification-Driven Reinforcement Learning for Maritime Motion Planning](http://arxiv.org/pdf/2510.06970v1)

Authors: Marlon Müller, Florian Finkeldei, Hanna Krasowski, Murat Arcak, Matthias Althoff

Compliance with maritime traffic rules is essential for the safe operation of
autonomous vessels, yet training reinforcement learning (RL) agents to adhere
to them is challenging. The behavior of RL agents is shaped by the training
scenarios they encounter, but creating scenarios that capture the complexity of
maritime navigation is non-trivial, and real-world data alone is insufficient.
To address this, we propose a falsification-driven RL approach that generates
adversarial training scenarios in which the vessel under test violates maritime
traffic rules, which are expressed as signal temporal logic specifications. Our
experiments on open-sea navigation with two vessels demonstrate that the
proposed approach provides more relevant training scenarios and achieves more
consistent rule compliance.

### 9. [Identification and optimal control strategies for the transversal splitting of ultra--cold Bose gases](http://arxiv.org/pdf/2510.07113v1)

Authors: Nikolaus Würkner, Yevhenii Kuriatnikov, Karthikeyan Kumaran, Marupaka Venkat Ramana, Jörg Schmiedmayer, Andreas Kugi, Maximilian Prüfer, Andreas Deutschmann-Olek

Splitting a Bose--Einstein condensate (BEC) is a key operation in fundamental
physics experiments and emerging quantum technologies, where precise
preparation of well--defined initial states requires fast yet coherent control
of the condensate's nonlinear dynamics. This work formulates the BEC splitting
process as an optimal feedforward control problem based on a physically
interpretable, reduced--order model identified from limited experimental data.
We introduce a systematic calibration strategy that combines optimal experiment
selection and constrained nonlinear parameter estimation, enabling accurate
system identification with minimal experimental overhead. Using this calibrated
model, we compute energy--optimal trajectories via indirect optimal control to
realize shortcuts to adiabaticity (STAs), achieving rapid transitions to the
ground state of a double--well potential while suppressing excitations.
Experiments confirm that the proposed control framework yields high--fidelity
state transfers across multiple configurations, demonstrating its robustness
and scalability for quantum control applications.

### 10. [Stability Preserving Safe Control of a Bicopter](http://arxiv.org/pdf/2510.07145v1)

Authors: Jhon Manuel Portella Delgado, Ankit Goel

This paper presents a control law for stabilization and trajectory tracking
of a multicopter subject to safety constraints. The proposed approach
guarantees forward invariance of a prescribed safety set while ensuring smooth
tracking performance. Unlike conventional control barrier function methods, the
constrained control problem is transformed into an unconstrained one using
state-dependent mappings together with carefully constructed Lyapunov
functions. This approach enables explicit synthesis of the control law, instead
of requiring a solution of constrained optimization at each step. The
transformation also enables the controller to enforce safety without
sacrificing stability or performance. Simulation results for a polytopic
reference trajectory confined within a designated safe region demonstrate the
effectiveness of the proposed method.

### Machine Learning (Statistics Category)

### 1. [Wide Neural Networks as a Baseline for the Computational No-Coincidence Conjecture](http://arxiv.org/pdf/2510.06527v1)

Authors: John Dunbar, Scott Aaronson

We establish that randomly initialized neural networks, with large width and
a natural choice of hyperparameters, have nearly independent outputs exactly
when their activation function is nonlinear with zero mean under the Gaussian
measure: $\mathbb{E}_{z \sim \mathcal{N}(0,1)}[\sigma(z)]=0$. For example, this
includes ReLU and GeLU with an additive shift, as well as tanh, but not ReLU or
GeLU by themselves. Because of their nearly independent outputs, we propose
neural networks with zero-mean activation functions as a promising candidate
for the Alignment Research Center's computational no-coincidence conjecture --
a conjecture that aims to measure the limits of AI interpretability.

### 2. [Q-Learning with Fine-Grained Gap-Dependent Regret](http://arxiv.org/pdf/2510.06647v1)

Authors: Haochen Zhang, Zhong Zheng, Lingzhou Xue

We study fine-grained gap-dependent regret bounds for model-free
reinforcement learning in episodic tabular Markov Decision Processes. Existing
model-free algorithms achieve minimax worst-case regret, but their
gap-dependent bounds remain coarse and fail to fully capture the structure of
suboptimality gaps. We address this limitation by establishing fine-grained
gap-dependent regret bounds for both UCB-based and non-UCB-based algorithms. In
the UCB-based setting, we develop a novel analytical framework that explicitly
separates the analysis of optimal and suboptimal state-action pairs, yielding
the first fine-grained regret upper bound for UCB-Hoeffding (Jin et al., 2018).
To highlight the generality of this framework, we introduce ULCB-Hoeffding, a
new UCB-based algorithm inspired by AMB (Xu et al.,2021) but with a simplified
structure, which enjoys fine-grained regret guarantees and empirically
outperforms AMB. In the non-UCB-based setting, we revisit the only known
algorithm AMB, and identify two key issues in its algorithm design and
analysis: improper truncation in the $Q$-updates and violation of the
martingale difference condition in its concentration argument. We propose a
refined version of AMB that addresses these issues, establishing the first
rigorous fine-grained gap-dependent regret for a non-UCB-based method, with
experiments demonstrating improved performance over AMB.

### 3. [The Effect of Attention Head Count on Transformer Approximation](http://arxiv.org/pdf/2510.06662v1)

Authors: Penghao Yu, Haotian Jiang, Zeyu Bao, Ruoxi Yu, Qianxiao Li

Transformer has become the dominant architecture for sequence modeling, yet a
detailed understanding of how its structural parameters influence expressive
power remains limited. In this work, we study the approximation properties of
transformers, with particular emphasis on the role of the number of attention
heads. Our analysis begins with the introduction of a generalized $D$-retrieval
task, which we prove to be dense in the space of continuous functions, thereby
providing the basis for our theoretical framework. We then establish both upper
and lower bounds on the parameter complexity required for
$\epsilon$-approximation. Specifically, we show that transformers with
sufficiently many heads admit efficient approximation, whereas with too few
heads, the number of parameters must scale at least as $O(1/\epsilon^{cT})$,
for some constant $c$ and sequence length $T$. To the best of our knowledge,
this constitutes the first rigorous lower bound of this type in a nonlinear and
practically relevant setting. We further examine the single-head case and
demonstrate that an embedding dimension of order $O(T)$ allows complete
memorization of the input, where approximation is entirely achieved by the
feed-forward block. Finally, we validate our theoretical findings with
experiments on both synthetic data and real-world tasks, illustrating the
practical relevance of our results.

### 4. [PyCFRL: A Python library for counterfactually fair offline reinforcement learning via sequential data preprocessing](http://arxiv.org/pdf/2510.06935v1)

Authors: Jianhan Zhang, Jitao Wang, Chengchun Shi, John D. Piette, Donglin Zeng, Zhenke Wu

Reinforcement learning (RL) aims to learn and evaluate a sequential decision
rule, often referred to as a "policy", that maximizes the population-level
benefit in an environment across possibly infinitely many time steps. However,
the sequential decisions made by an RL algorithm, while optimized to maximize
overall population benefits, may disadvantage certain individuals who are in
minority or socioeconomically disadvantaged groups. To address this problem, we
introduce PyCFRL, a Python library for ensuring counterfactual fairness in
offline RL. PyCFRL implements a novel data preprocessing algorithm for learning
counterfactually fair RL policies from offline datasets and provides tools to
evaluate the values and counterfactual unfairness levels of RL policies. We
describe the high-level functionalities of PyCFRL and demonstrate one of its
major use cases through a data example. The library is publicly available on
PyPI and Github (https://github.com/JianhanZhang/PyCFRL), and detailed
tutorials can be found in the PyCFRL documentation
(https://pycfrl-documentation.netlify.app).

### 5. [Explaining Models under Multivariate Bernoulli Distribution via Hoeffding Decomposition](http://arxiv.org/pdf/2510.07088v1)

Authors: Baptiste Ferrere, Nicolas Bousquet, Fabrice Gamboa, Jean-Michel Loubes, Joseph Muré

Explaining the behavior of predictive models with random inputs can be
achieved through sub-models decomposition, where such sub-models have easier
interpretable features. Arising from the uncertainty quantification community,
recent results have demonstrated the existence and uniqueness of a generalized
Hoeffding decomposition for such predictive models when the stochastic input
variables are correlated, based on concepts of oblique projection onto L 2
subspaces. This article focuses on the case where the input variables have
Bernoulli distributions and provides a complete description of this
decomposition. We show that in this case the underlying L 2 subspaces are
one-dimensional and that the functional decomposition is explicit. This leads
to a complete interpretability framework and theoretically allows reverse
engineering. Explicit indicators of the influence of inputs on the output
prediction (exemplified by Sobol' indices and Shapley effects) can be
explicitly derived. Illustrated by numerical experiments, this type of analysis
proves useful for addressing decision-support problems, based on binary
decision diagrams, Boolean networks or binary neural networks. The article
outlines perspectives for exploring high-dimensional settings and, beyond the
case of binary inputs, extending these findings to models with finite countable
inputs.

### 6. [Non-Asymptotic Analysis of Efficiency in Conformalized Regression](http://arxiv.org/pdf/2510.07093v1)

Authors: Yunzhen Yao, Lie He, Michael Gastpar

Conformal prediction provides prediction sets with coverage guarantees. The
informativeness of conformal prediction depends on its efficiency, typically
quantified by the expected size of the prediction set. Prior work on the
efficiency of conformalized regression commonly treats the miscoverage level
$\alpha$ as a fixed constant. In this work, we establish non-asymptotic bounds
on the deviation of the prediction set length from the oracle interval length
for conformalized quantile and median regression trained via SGD, under mild
assumptions on the data distribution. Our bounds of order
$\mathcal{O}(1/\sqrt{n} + 1/(\alpha^2 n) + 1/\sqrt{m} + \exp(-\alpha^2 m))$
capture the joint dependence of efficiency on the proper training set size $n$,
the calibration set size $m$, and the miscoverage level $\alpha$. The results
identify phase transitions in convergence rates across different regimes of
$\alpha$, offering guidance for allocating data to control excess prediction
set length. Empirical results are consistent with our theoretical findings.

### 7. [jmstate, a Flexible Python Package for Multi-State Joint Modeling](http://arxiv.org/pdf/2510.07128v1)

Authors: Félix Laplante, Christophe Ambroise, Estelle Kuhn, Sarah Lemler

Classical joint modeling approaches often rely on competing risks or
recurrent event formulations to account for complex real-world processes
involving evolving longitudinal markers and discrete event occurrences.
However, these frameworks typically capture only limited aspects of the
underlying event dynamics.
  Multi-state joint models offer a more flexible alternative by representing
full event histories through a network of possible transitions, including
recurrent cycles and terminal absorptions, all potentially influenced by
longitudinal covariates.
  In this paper, we propose a general framework that unifies longitudinal
biomarker modeling with multi-state event processes defined on arbitrary
directed graphs. Our approach accommodates both Markovian and semi-Markovian
transition structures, and extends classical joint models by coupling nonlinear
mixed-effects longitudinal submodels with multi-state survival processes via
shared latent structures.
  We derive the full likelihood and develop scalable inference procedures based
on stochastic gradient descent. Furthermore, we introduce a dynamic prediction
framework, enabling individualized risk assessments along complex
state-transition trajectories.
  To facilitate reproducibility and dissemination, we provide an open-source
Python library \texttt{jmstate} implementing the proposed methodology,
available on \href{https://pypi.org/project/jmstate/}{PyPI}. Simulation
experiments and a biomedical case study demonstrate the flexibility and
performance of the framework in representing complex longitudinal and
multi-state event dynamics. The full Python notebooks used to reproduce the
experiments as well as the source code of this paper are available on
\href{https://gitlab.com/felixlaplante0/jmstate-paper/}{GitLab}.

### 8. [Split Conformal Classification with Unsupervised Calibration](http://arxiv.org/pdf/2510.07185v1)

Authors: Santiago Mazuelas

Methods for split conformal prediction leverage calibration samples to
transform any prediction rule into a set-prediction rule that complies with a
target coverage probability. Existing methods provide remarkably strong
performance guarantees with minimal computational costs. However, they require
to use calibration samples composed by labeled examples different to those used
for training. This requirement can be highly inconvenient, as it prevents the
use of all labeled examples for training and may require acquiring additional
labels solely for calibration. This paper presents an effective methodology for
split conformal prediction with unsupervised calibration for classification
tasks. In the proposed approach, set-prediction rules are obtained using
unsupervised calibration samples together with supervised training samples
previously used to learn the classification rule. Theoretical and experimental
results show that the presented methods can achieve performance comparable to
that with supervised calibration, at the expenses of a moderate degradation in
performance guarantees and computational efficiency.

### 9. [Scalable Policy-Based RL Algorithms for POMDPs](http://arxiv.org/pdf/2510.06540v1)

Authors: Ameya Anjarlekar, Rasoul Etesami, R Srikant

The continuous nature of belief states in POMDPs presents significant
computational challenges in learning the optimal policy. In this paper, we
consider an approach that solves a Partially Observable Reinforcement Learning
(PORL) problem by approximating the corresponding POMDP model into a
finite-state Markov Decision Process (MDP) (called Superstate MDP). We first
derive theoretical guarantees that improve upon prior work that relate the
optimal value function of the transformed Superstate MDP to the optimal value
function of the original POMDP. Next, we propose a policy-based learning
approach with linear function approximation to learn the optimal policy for the
Superstate MDP. Consequently, our approach shows that a POMDP can be
approximately solved using TD-learning followed by Policy Optimization by
treating it as an MDP, where the MDP state corresponds to a finite history. We
show that the approximation error decreases exponentially with the length of
this history. To the best of our knowledge, our finite-time bounds are the
first to explicitly quantify the error introduced when applying standard TD
learning to a setting where the true dynamics are not Markovian.

### 10. [Gaussian Equivalence for Self-Attention: Asymptotic Spectral Analysis of Attention Matrix](http://arxiv.org/pdf/2510.06685v1)

Authors: Tomohiro Hayase, Benoît Collins, Ryo Karakida

Self-attention layers have become fundamental building blocks of modern deep
neural networks, yet their theoretical understanding remains limited,
particularly from the perspective of random matrix theory. In this work, we
provide a rigorous analysis of the singular value spectrum of the attention
matrix and establish the first Gaussian equivalence result for attention. In a
natural regime where the inverse temperature remains of constant order, we show
that the singular value distribution of the attention matrix is asymptotically
characterized by a tractable linear model. We further demonstrate that the
distribution of squared singular values deviates from the Marchenko-Pastur law,
which has been believed in previous work. Our proof relies on two key
ingredients: precise control of fluctuations in the normalization term and a
refined linearization that leverages favorable Taylor expansions of the
exponential. This analysis also identifies a threshold for linearization and
elucidates why attention, despite not being an entrywise operation, admits a
rigorous Gaussian equivalence in this regime.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-09 PST.

### 1. [QuKAN: A Quantum Circuit Born Machine Approach to Quantum Kolmogorov Arnold Networks](https://www.nature.com/articles/s41598-025-22705-9)

Authors: Yannick Werner et al.

### 2. [Deep phenotyping of patient lived experience in functional bowel disorders using machine learning](https://www.nature.com/articles/s41598-025-19273-3)

Authors: James K. Ruffle et al.

### 3. [Precise 2D vision solutions for estimating avocado physical characteristics](https://www.nature.com/articles/s41598-025-19238-6)

Authors: Hieu M. Tran et al.

### 4. [A rapid DAS signal classification algorithm based on VMD and IMF power spectrum Gaussian fitting](https://www.nature.com/articles/s41598-025-19320-z)

Authors: Haitao Liu et al.

### 5. [Exploring parameter optimisation in machine learning algorithms for locomotor task discrimination using wearable sensors](https://www.nature.com/articles/s41598-025-17361-y)

Authors: L. D. Hughes et al.

### 6. [A deep ensemble learning framework for brain tumor classification using data balancing and fine-tuning](https://www.nature.com/articles/s41598-025-03752-8)

Authors: Md. Alamin Talukder et al.

### 7. [Dual chain dynamic hypergraph convolution network for 3D human pose estimation](https://www.nature.com/articles/s41598-025-22261-2)

Authors: Qiuying Han et al.

### 8. [Decoding trust in large language models for healthcare in Saudi Arabia](https://www.nature.com/articles/s41598-025-18404-0)

Authors: Turki Alelyani

### 9. [Research on intercity travel mode recognition and network structure characteristics based on complex network and random forest classification](https://www.nature.com/articles/s41598-025-19392-x)

Authors: Wanping Zhang et al.

### 10. [Advanced transformer with attention-based neural network framework for precise renal cell carcinoma detection using histological kidney images](https://www.nature.com/articles/s41598-025-19352-5)

Authors: M. Eliazer et al.

### 11. [A comparative performance analysis of fully homomorphic and attribute-based encryption schemes](https://www.nature.com/articles/s41598-025-19404-w)

Authors: Kirti Dinkar More et al.

### 12. [Intelligent generation method of drum music scores based on improved CNN and STFT](https://www.nature.com/articles/s41598-025-19348-1)

Authors: Yuting Ni

