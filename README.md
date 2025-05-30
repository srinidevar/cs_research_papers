# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-29 17:03:24.077710 PST.

### Artificial Intelligence

### 1. [From Reasoning to Learning: A Survey on Hypothesis Discovery and Rule Learning with Large Language Models](http://arxiv.org/pdf/2505.21935v1)

Authors: Kaiyu He, Zhiyu Chen

Since the advent of Large Language Models (LLMs), efforts have largely
focused on improving their instruction-following and deductive reasoning
abilities, leaving open the question of whether these models can truly discover
new knowledge. In pursuit of artificial general intelligence (AGI), there is a
growing need for models that not only execute commands or retrieve information
but also learn, reason, and generate new knowledge by formulating novel
hypotheses and theories that deepen our understanding of the world. Guided by
Peirce's framework of abduction, deduction, and induction, this survey offers a
structured lens to examine LLM-based hypothesis discovery. We synthesize
existing work in hypothesis generation, application, and validation,
identifying both key achievements and critical gaps. By unifying these threads,
we illuminate how LLMs might evolve from mere ``information executors'' into
engines of genuine innovation, potentially transforming research, science, and
real-world problem solving.

### 2. [Functional Matching of Logic Subgraphs: Beyond Structural Isomorphism](http://arxiv.org/pdf/2505.21988v1)

Authors: Ziyang Zheng, Kezhi Li, Zhengyuan Shi, Qiang Xu

Subgraph matching in logic circuits is foundational for numerous Electronic
Design Automation (EDA) applications, including datapath optimization,
arithmetic verification, and hardware trojan detection. However, existing
techniques rely primarily on structural graph isomorphism and thus fail to
identify function-related subgraphs when synthesis transformations
substantially alter circuit topology. To overcome this critical limitation, we
introduce the concept of functional subgraph matching, a novel approach that
identifies whether a given logic function is implicitly present within a larger
circuit, irrespective of structural variations induced by synthesis or
technology mapping. Specifically, we propose a two-stage multi-modal framework:
(1) learning robust functional embeddings across AIG and post-mapping netlists
for functional subgraph detection, and (2) identifying fuzzy boundaries using a
graph segmentation approach. Evaluations on standard benchmarks (ITC99,
OpenABCD, ForgeEDA) demonstrate significant performance improvements over
existing structural methods, with average $93.8\%$ accuracy in functional
subgraph detection and a dice score of $91.3\%$ in fuzzy boundary
identification.

### 3. [Cognitively-Inspired Emergent Communication via Knowledge Graphs for Assisting the Visually Impaired](http://arxiv.org/pdf/2505.22087v1)

Authors: Ruxiao Chen, Dezheng Han, Wenjie Han, Shuaishuai Guo

Assistive systems for visually impaired individuals must deliver rapid,
interpretable, and adaptive feedback to facilitate real-time navigation.
Current approaches face a trade-off between latency and semantic richness:
natural language-based systems provide detailed guidance but are too slow for
dynamic scenarios, while emergent communication frameworks offer low-latency
symbolic languages but lack semantic depth, limiting their utility in tactile
modalities like vibration. To address these limitations, we introduce a novel
framework, Cognitively-Inspired Emergent Communication via Knowledge Graphs
(VAG-EC), which emulates human visual perception and cognitive mapping. Our
method constructs knowledge graphs to represent objects and their
relationships, incorporating attention mechanisms to prioritize task-relevant
entities, thereby mirroring human selective attention. This structured approach
enables the emergence of compact, interpretable, and context-sensitive symbolic
languages. Extensive experiments across varying vocabulary sizes and message
lengths demonstrate that VAG-EC outperforms traditional emergent communication
methods in Topographic Similarity (TopSim) and Context Independence (CI). These
findings underscore the potential of cognitively grounded emergent
communication as a fast, adaptive, and human-aligned solution for real-time
assistive technologies. Code is available at
https://github.com/Anonymous-NLPcode/Anonymous_submission/tree/main.

### 4. [VIRAL: Vision-grounded Integration for Reward design And Learning](http://arxiv.org/pdf/2505.22092v1)

Authors: Valentin Cuzin-Rambaud, Emilien Komlenovic, Alexandre Faure, Bruno Yun

The alignment between humans and machines is a critical challenge in
artificial intelligence today. Reinforcement learning, which aims to maximize a
reward function, is particularly vulnerable to the risks associated with poorly
designed reward functions. Recent advancements has shown that Large Language
Models (LLMs) for reward generation can outperform human performance in this
context. We introduce VIRAL, a pipeline for generating and refining reward
functions through the use of multi-modal LLMs. VIRAL autonomously creates and
interactively improves reward functions based on a given environment and a goal
prompt or annotated image. The refinement process can incorporate human
feedback or be guided by a description generated by a video LLM, which explains
the agent's policy in video form. We evaluated VIRAL in five Gymnasium
environments, demonstrating that it accelerates the learning of new behaviors
while ensuring improved alignment with user intent. The source-code and demo
video are available at: https://github.com/VIRAL-UCBL1/VIRAL and
https://youtu.be/t4_BXugBm9Q.

### 5. [Lifted Forward Planning in Relational Factored Markov Decision Processes with Concurrent Actions](http://arxiv.org/pdf/2505.22147v1)

Authors: Florian Andreas Marwitz, Tanya Braun, Ralf Möller, Marcel Gehrke

Decision making is a central problem in AI that can be formalized using a
Markov Decision Process. A problem is that, with increasing numbers of
(indistinguishable) objects, the state space grows exponentially. To compute
policies, the state space has to be enumerated. Even more possibilities have to
be enumerated if the size of the action space depends on the size of the state
space, especially if we allow concurrent actions. To tackle the exponential
blow-up in the action and state space, we present a first-order representation
to store the spaces in polynomial instead of exponential size in the number of
objects and introduce Foreplan, a relational forward planner, which uses this
representation to efficiently compute policies for numerous indistinguishable
objects and actions. Additionally, we introduce an even faster approximate
version of Foreplan. Moreover, Foreplan identifies how many objects an agent
should act on to achieve a certain task given restrictions. Further, we provide
a theoretical analysis and an empirical evaluation of Foreplan, demonstrating a
speedup of at least four orders of magnitude.

### 6. [What Makes a Good Reasoning Chain? Uncovering Structural Patterns in Long Chain-of-Thought Reasoning](http://arxiv.org/pdf/2505.22148v1)

Authors: Gangwei Jiang, Yahui Liu, Zhaoyi Li, Qi Wang, Fuzheng Zhang, Linqi Song, Ying Wei, Defu Lian

Recent advances in reasoning with large language models (LLMs) have
popularized Long Chain-of-Thought (LCoT), a strategy that encourages deliberate
and step-by-step reasoning before producing a final answer. While LCoTs have
enabled expert-level performance in complex tasks, how the internal structures
of their reasoning chains drive, or even predict, the correctness of final
answers remains a critical yet underexplored question. In this work, we present
LCoT2Tree, an automated framework that converts sequential LCoTs into
hierarchical tree structures and thus enables deeper structural analysis of LLM
reasoning. Using graph neural networks (GNNs), we reveal that structural
patterns extracted by LCoT2Tree, including exploration, backtracking, and
verification, serve as stronger predictors of final performance across a wide
range of tasks and models. Leveraging an explainability technique, we further
identify critical thought patterns such as over-branching that account for
failures. Beyond diagnostic insights, the structural patterns by LCoT2Tree
support practical applications, including improving Best-of-N decoding
effectiveness. Overall, our results underscore the critical role of internal
structures of reasoning chains, positioning LCoT2Tree as a powerful tool for
diagnosing, interpreting, and improving reasoning in LLMs.

### 7. [A Preprocessing Framework for Efficient Approximate Bi-Objective Shortest-Path Computation in the Presence of Correlated Objectives](http://arxiv.org/pdf/2505.22244v1)

Authors: Yaron Halle, Ariel Felner, Sven Koenig, Oren Salzman

The bi-objective shortest-path (BOSP) problem seeks to find paths between
start and target vertices of a graph while optimizing two conflicting objective
functions. We consider the BOSP problem in the presence of correlated
objectives. Such correlations often occur in real-world settings such as road
networks, where optimizing two positively correlated objectives, such as travel
time and fuel consumption, is common. BOSP is generally computationally
challenging as the size of the search space is exponential in the number of
objective functions and the graph size. Bounded sub-optimal BOSP solvers such
as A*pex alleviate this complexity by approximating the Pareto-optimal solution
set rather than computing it exactly (given a user-provided approximation
factor). As the correlation between objective functions increases, smaller
approximation factors are sufficient for collapsing the entire Pareto-optimal
set into a single solution. We leverage this insight to propose an efficient
algorithm that reduces the search effort in the presence of correlated
objectives. Our approach for computing approximations of the entire
Pareto-optimal set is inspired by graph-clustering algorithms. It uses a
preprocessing phase to identify correlated clusters within a graph and to
generate a new graph representation. This allows a natural generalization of
A*pex to run up to five times faster on DIMACS dataset instances, a standard
benchmark in the field. To the best of our knowledge, this is the first
algorithm proposed that efficiently and effectively exploits correlations in
the context of bi-objective search while providing theoretical guarantees on
solution quality.

### 8. [Compression versus Accuracy: A Hierarchy of Lifted Models](http://arxiv.org/pdf/2505.22288v1)

Authors: Jan Speller, Malte Luttermann, Marcel Gehrke, Tanya Braun

Probabilistic graphical models that encode indistinguishable objects and
relations among them use first-order logic constructs to compress a
propositional factorised model for more efficient (lifted) inference. To obtain
a lifted representation, the state-of-the-art algorithm Advanced Colour Passing
(ACP) groups factors that represent matching distributions. In an approximate
version using $\varepsilon$ as a hyperparameter, factors are grouped that
differ by a factor of at most $(1\pm \varepsilon)$. However, finding a suitable
$\varepsilon$ is not obvious and may need a lot of exploration, possibly
requiring many ACP runs with different $\varepsilon$ values. Additionally,
varying $\varepsilon$ can yield wildly different models, leading to decreased
interpretability. Therefore, this paper presents a hierarchical approach to
lifted model construction that is hyperparameter-free. It efficiently computes
a hierarchy of $\varepsilon$ values that ensures a hierarchy of models, meaning
that once factors are grouped together given some $\varepsilon$, these factors
will be grouped together for larger $\varepsilon$ as well. The hierarchy of
$\varepsilon$ values also leads to a hierarchy of error bounds. This allows for
explicitly weighing compression versus accuracy when choosing specific
$\varepsilon$ values to run ACP with and enables interpretability between the
different models.

### 9. [AgentDNS: A Root Domain Naming System for LLM Agents](http://arxiv.org/pdf/2505.22368v1)

Authors: Enfang Cui, Yujun Cheng, Rui She, Dan Liu, Zhiyuan Liang, Minxin Guo, Tianzheng Li, Qian Wei, Wenjuan Xing, Zhijie Zhong

The rapid evolution of Large Language Model (LLM) agents has highlighted
critical challenges in cross-vendor service discovery, interoperability, and
communication. Existing protocols like model context protocol and
agent-to-agent protocol have made significant strides in standardizing
interoperability between agents and tools, as well as communication among
multi-agents. However, there remains a lack of standardized protocols and
solutions for service discovery across different agent and tool vendors. In
this paper, we propose AgentDNS, a root domain naming and service discovery
system designed to enable LLM agents to autonomously discover, resolve, and
securely invoke third-party agent and tool services across organizational and
technological boundaries. Inspired by the principles of the traditional DNS,
AgentDNS introduces a structured mechanism for service registration, semantic
service discovery, secure invocation, and unified billing. We detail the
architecture, core functionalities, and use cases of AgentDNS, demonstrating
its potential to streamline multi-agent collaboration in real-world scenarios.
The source code will be published on https://github.com/agentdns.

### 10. [AI Mathematician: Towards Fully Automated Frontier Mathematical Research](http://arxiv.org/pdf/2505.22451v1)

Authors: Yuanhang Liu, Yanxing Huang, Yanqiao Wang, Peng Li, Yang Liu

Large Reasoning Models (LRMs) have made significant progress in mathematical
capabilities in recent times. However, these successes have been primarily
confined to competition-level problems. In this work, we propose AI
Mathematician (AIM) framework, which harnesses the reasoning strength of LRMs
to support frontier mathematical research. We have identified two critical
challenges of mathematical research compared to competition, {\it the intrinsic
complexity of research problems} and {\it the requirement of procedural rigor}.
To address these challenges, AIM incorporates two core strategies: an
exploration mechanism to foster longer solution paths, and the pessimistic
reasonable verification method to ensure reliability.
  This early version of AIM already exhibits strong capability in tackling
research-level tasks. We conducted extensive experiments across several
real-world mathematical topics and obtained promising results. AIM is able to
autonomously construct substantial portions of proofs and uncover non-trivial
insights within each research area. These findings highlight the potential of
LRMs in mathematical discovery and suggest that LRM-based agent systems could
significantly accelerate mathematical research in the future.

### Hardware Architecture

### 1. [Refining Datapath for Microscaling ViTs](http://arxiv.org/pdf/2505.22194v1)

Authors: Can Xiao, Jianyi Cheng, Aaron Zhao

Vision Transformers (ViTs) leverage the transformer architecture to
effectively capture global context, demonstrating strong performance in
computer vision tasks. A major challenge in ViT hardware acceleration is that
the model family contains complex arithmetic operations that are sensitive to
model accuracy, such as the Softmax and LayerNorm operations, which cannot be
mapped onto efficient hardware with low precision. Existing methods only
exploit parallelism in the matrix multiplication operations of the model on
hardware and keep these complex operations on the CPU. This results in
suboptimal performance due to the communication overhead between the CPU and
accelerator. Can new data formats solve this problem?
  In this work, we present the first ViT accelerator that maps all operations
of the ViT models onto FPGAs. We exploit a new arithmetic format named
Microscaling Integer (MXInt) for datapath designs and evaluate how different
design choices can be made to trade off accuracy, hardware performance, and
hardware utilization. Our contributions are twofold. First, we quantize ViTs
using the MXInt format, achieving both high area efficiency and accuracy.
Second, we propose MXInt-specific hardware optimization that map these complex
arithmetic operations into custom hardware. Within 1\% accuracy loss, our
method achieves at least 93$\times$ speedup compared to Float16 and at least
1.9$\times$ speedup compared to related work.

### 2. [EStacker: Explaining Battery-Less IoT System Performance with Energy Stacks](http://arxiv.org/pdf/2505.22366v1)

Authors: Lukas Liedtke, Per Gunnar Kjeldsberg, Frank Alexander Kraemer, Magnus Jahre

The number of Internet of Things (IoT) devices is increasing exponentially,
and it is environmentally and economically unsustainable to power all these
devices with batteries. The key alternative is energy harvesting, but
battery-less IoT systems require extensive evaluation to demonstrate that they
are sufficiently performant across the full range of expected operating
conditions. IoT developers thus need an evaluation platform that (i) ensures
that each evaluated application and configuration is exposed to exactly the
same energy environment and events, and (ii) provides a detailed account of
what the application spends the harvested energy on. We therefore developed the
EStacker evaluation platform which (i) provides fair and repeatable evaluation,
and (ii) generates energy stacks. Energy stacks break down the total energy
consumption of an application across hardware components and application
activities, thereby explaining what the application specifically uses energy
on. We augment EStacker with the ST-SP optimization which, in our experiments,
reduces evaluation time by 6.3x on average while retaining the temporal
behavior of the battery-less IoT system (average throughput error of 7.7%) by
proportionally scaling time and power. We demonstrate the utility of EStacker
through two case studies. In the first case study, we use energy stack profiles
to identify a performance problem that, once addressed, improves performance by
3.3x. The second case study focuses on ST-SP, and we use it to explore the
design space required to dimension the harvester and energy storage sizes of a
smart parking application in roughly one week (7.7 days). Without ST-SP,
sweeping this design space would have taken well over one month (41.7 days).

### 3. [GPU-Accelerated Simulated Oscillator Ising/Potts Machine Solving Combinatorial Optimization Problems](http://arxiv.org/pdf/2505.22631v1)

Authors: Yilmaz Ege Gonul, Ceyhun Efe Kayan, Ilknur Mustafazade, Nagarajan Kandasamy, Baris Taskin

Oscillator-based Ising machines (OIMs) and oscillator-based Potts machines
(OPMs) have emerged as promising hardware accelerators for solving NP-hard
combinatorial optimization problems by leveraging the phase dynamics of coupled
oscillators. In this work, a GPU-accelerated simulated OIM/OPM digital
computation framework capable of solving combinatorial optimization problems is
presented. The proposed implementation harnesses the parallel processing
capabilities of GPUs to simulate large-scale OIM/OPMs, leveraging the
advantages of digital computing to offer high precision, programmability, and
scalability. The performance of the proposed GPU framework is evaluated on the
max-cut problems from the GSET benchmark dataset and graph coloring problems
from the SATLIB benchmarks dataset, demonstrating competitive speed and
accuracy in tackling large-scale problems. The results from simulations,
reaching up to 11295x speed-up over CPUs with up to 99% accuracy, establish
this framework as a scalable, massively parallelized, and high-fidelity digital
realization of OIM/OPMs.

### 4. [iDSE: Navigating Design Space Exploration in High-Level Synthesis Using LLMs](http://arxiv.org/pdf/2505.22086v1)

Authors: Runkai Li, Jia Xiong, Xi Wang

High-Level Synthesis (HLS) serves as an agile hardware development tool that
streamlines the circuit design by abstracting the register transfer level into
behavioral descriptions, while allowing designers to customize the generated
microarchitectures through optimization directives. However, the combinatorial
explosion of possible directive configurations yields an intractable design
space. Traditional design space exploration (DSE) methods, despite adopting
heuristics or constructing predictive models to accelerate Pareto-optimal
design acquisition, still suffer from prohibitive exploration costs and
suboptimal results. Addressing these concerns, we introduce iDSE, the first
LLM-aided DSE framework that leverages HLS design quality perception to
effectively navigate the design space. iDSE intelligently pruns the design
space to guide LLMs in calibrating representative initial sampling designs,
expediting convergence toward the Pareto front. By exploiting the convergent
and divergent thinking patterns inherent in LLMs for hardware optimization,
iDSE achieves multi-path refinement of the design quality and diversity.
Extensive experiments demonstrate that iDSE outperforms heuristic-based DSE
methods by 5.1$\times$$\sim$16.6$\times$ in proximity to the reference Pareto
front, matching NSGA-II with only 4.6% of the explored designs. Our work
demonstrates the transformative potential of LLMs in scalable and efficient HLS
design optimization, offering new insights into multiobjective optimization
challenges.

### 5. [Efficient Precision-Scalable Hardware for Microscaling (MX) Processing in Robotics Learning](http://arxiv.org/pdf/2505.22404v1)

Authors: Stef Cuyckens, Xiaoling Yi, Nitish Satya Murthy, Chao Fang, Marian Verhelst

Autonomous robots require efficient on-device learning to adapt to new
environments without cloud dependency. For this edge training, Microscaling
(MX) data types offer a promising solution by combining integer and
floating-point representations with shared exponents, reducing energy
consumption while maintaining accuracy. However, the state-of-the-art
continuous learning processor, namely Dacapo, faces limitations with its
MXINT-only support and inefficient vector-based grouping during
backpropagation. In this paper, we present, to the best of our knowledge, the
first work that addresses these limitations with two key innovations: (1) a
precision-scalable arithmetic unit that supports all six MX data types by
exploiting sub-word parallelism and unified integer and floating-point
processing; and (2) support for square shared exponent groups to enable
efficient weight handling during backpropagation, removing storage redundancy
and quantization overhead. We evaluate our design against Dacapo under
iso-peak-throughput on four robotics workloads in TSMC 16nm FinFET technology
at 500MHz, reaching a 25.6% area reduction, a 51% lower memory footprint, and
4x higher effective training throughput while achieving comparable
energy-efficiency, enabling efficient robotics continual learning at the edge.

### 6. [FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design](http://arxiv.org/pdf/2505.21923v1)

Authors: Asal Mehradfar, Xuzhe Zhao, Yilun Huang, Emir Ceyani, Yankai Yang, Shihao Han, Hamidreza Aghasi, Salman Avestimehr

Designing analog circuits from performance specifications is a complex,
multi-stage process encompassing topology selection, parameter inference, and
layout feasibility. We introduce FALCON, a unified machine learning framework
that enables fully automated, specification-driven analog circuit synthesis
through topology selection and layout-constrained optimization. Given a target
performance, FALCON first selects an appropriate circuit topology using a
performance-driven classifier guided by human design heuristics. Next, it
employs a custom, edge-centric graph neural network trained to map circuit
topology and parameters to performance, enabling gradient-based parameter
inference through the learned forward model. This inference is guided by a
differentiable layout cost, derived from analytical equations capturing
parasitic and frequency-dependent effects, and constrained by design rules. We
train and evaluate FALCON on a large-scale custom dataset of 1M analog mm-wave
circuits, generated and simulated using Cadence Spectre across 20
expert-designed topologies. Through this evaluation, FALCON demonstrates >99\%
accuracy in topology inference, <10\% relative error in performance prediction,
and efficient layout-aware design that completes in under 1 second per
instance. Together, these results position FALCON as a practical and extensible
foundation model for end-to-end analog circuit design automation.

### Computational Complexity

### 1. [Counting Small Induced Subgraphs: Scorpions Are Easy but Not Trivial](http://arxiv.org/pdf/2505.22300v1)

Authors: Radu Curticapean, Simon Döring, Daniel Neuen

We consider the parameterized problem $\#$IndSub$(\Phi)$ for fixed graph
properties $\Phi$: Given a graph $G$ and an integer $k$, this problem asks to
count the number of induced $k$-vertex subgraphs satisfying $\Phi$. D\"orfler
et al. [Algorithmica 2022] and Roth et al. [SICOMP 2024] conjectured that
$\#$IndSub$(\Phi)$ is $\#$W[1]-hard for all non-meager properties $\Phi$, i.e.,
properties that are nontrivial for infinitely many $k$. This conjecture has
been confirmed for several restricted types of properties, including all
hereditary properties [STOC 2022] and all edge-monotone properties [STOC 2024].
  In this work, we refute this conjecture by showing that scorpion graphs,
certain $k$-vertex graphs which were introduced more than 50 years ago in the
context of the evasiveness conjecture, can be counted in time $O(n^4)$ for all
$k$. A simple variant of this construction results in graph properties that
achieve arbitrary intermediate complexity assuming ETH.
  We formulate an updated conjecture on the complexity of $\#$IndSub$(\Phi)$
that correctly captures the complexity status of scorpions and related
constructions.

### 2. [Faster Convolutions: Yates and Strassen Revisited](http://arxiv.org/pdf/2505.22410v1)

Authors: Cornelius Brand, Radu Curticapean, Baitian Li, Kevin Pratt

Given two vectors $u,v \in \mathbb{Q}^D$ over a finite domain $D$ and a
function $f : D\times D\to D$, the convolution problem asks to compute the
vector $w \in \mathbb{Q}^D$ whose entries are defined by $w(d) =
\sum_{\substack{x,y \in D \\ f(x,y)=d}} u(x)v(y).$ In parameterized and
exponential-time algorithms, convolutions on product domains are particularly
prominent: Here, a finite domain $B$ and a function $h : B \times B \to B$ are
fixed, and convolution is done over the product domain $D = B^k$, using the
function $h^k :D \times D\to D$ that applies $h$ coordinate-wise to its input
tuples.
  We present a new perspective on product-domain convolutions through
multilinear algebra. This viewpoint streamlines the presentation and analysis
of existing algorithms, such as those by van Rooij et al. (ESA 2009). Moreover,
using established results from the theory of fast matrix multiplication, we
derive improved $O^\ast(|B|^{2\omega/3 \cdot k}) = O(|D|^{1.582})$ time
algorithms, improving upon previous upper bounds by Esmer et al. (Algorithmica
86(1), 2024) of the form $c^k |B|^{2k}$ for $c < 1$. Using the setup described
in this note, Strassen's asymptotic rank conjecture from algebraic complexity
theory would imply quasi-linear $|D|^{1+o(1)}$ time algorithms. This conjecture
has recently gained attention in the algorithms community. (Bj\"orklund-Kaski
and Pratt, STOC 2024, Bj\"orklund et al., SODA 2025)
  Our paper is intended as a self-contained exposition for an algorithms
audience, and it includes all essential mathematical prerequisites with
explicit coordinate-based notation. In particular, we assume no knowledge in
abstract algebra.

### 3. [Finding $d$-Cuts in Probe $H$-Free Graphs](http://arxiv.org/pdf/2505.22351v1)

Authors: Konrad K. Dabrowski, Tala Eagling-Vose, Matthew Johnson, Giacomo Paesani, Daniël Paulusma

For an integer $d\geq 1$, the $d$-Cut problem is that of deciding whether a
graph has an edge cut in which each vertex is adjacent to at most $d$ vertices
on the opposite side of the cut. The $1$-Cut problem is the well-known Matching
Cut problem. The $d$-Cut problem has been extensively studied for $H$-free
graphs. We extend these results to the probe graph model, where we do not know
all the edges of the input graph. For a graph $H$, a partitioned probe $H$-free
graph $(G,P,N)$ consists of a graph $G=(V,E)$, together with a set $P\subseteq
V$ of probes and an independent set $N=V\setminus P$ of non-probes such that we
can change $G$ into an $H$-free graph by adding zero or more edges between
vertices in $N$. For every graph $H$ and every integer $d\geq 1$, we completely
determine the complexity of $d$-Cut on partitioned probe $H$-free graphs.

### Computational Engineering

### 1. [B-XAIC Dataset: Benchmarking Explainable AI for Graph Neural Networks Using Chemical Data](http://arxiv.org/pdf/2505.22252v1)

Authors: Magdalena Proszewska, Tomasz Danel, Dawid Rymarczyk

Understanding the reasoning behind deep learning model predictions is crucial
in cheminformatics and drug discovery, where molecular design determines their
properties. However, current evaluation frameworks for Explainable AI (XAI) in
this domain often rely on artificial datasets or simplified tasks, employing
data-derived metrics that fail to capture the complexity of real-world
scenarios and lack a direct link to explanation faithfulness. To address this,
we introduce B-XAIC, a novel benchmark constructed from real-world molecular
data and diverse tasks with known ground-truth rationales for assigned labels.
Through a comprehensive evaluation using B-XAIC, we reveal limitations of
existing XAI methods for Graph Neural Networks (GNNs) in the molecular domain.
This benchmark provides a valuable resource for gaining deeper insights into
the faithfulness of XAI, facilitating the development of more reliable and
interpretable models.

### 2. [SVRPBench: A Realistic Benchmark for Stochastic Vehicle Routing Problem](http://arxiv.org/pdf/2505.21887v1)

Authors: Ahmed Heakl, Yahia Salaheldin Shaaban, Martin Takac, Salem Lahlou, Zangir Iklassov

Robust routing under uncertainty is central to real-world logistics, yet most
benchmarks assume static, idealized settings. We present SVRPBench, the first
open benchmark to capture high-fidelity stochastic dynamics in vehicle routing
at urban scale. Spanning more than 500 instances with up to 1000 customers, it
simulates realistic delivery conditions: time-dependent congestion, log-normal
delays, probabilistic accidents, and empirically grounded time windows for
residential and commercial clients. Our pipeline generates diverse,
constraint-rich scenarios, including multi-depot and multi-vehicle setups.
Benchmarking reveals that state-of-the-art RL solvers like POMO and AM degrade
by over 20% under distributional shift, while classical and metaheuristic
methods remain robust. To enable reproducible research, we release the dataset
and evaluation suite. SVRPBench challenges the community to design solvers that
generalize beyond synthetic assumptions and adapt to real-world uncertainty.

### 3. [FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design](http://arxiv.org/pdf/2505.21923v1)

Authors: Asal Mehradfar, Xuzhe Zhao, Yilun Huang, Emir Ceyani, Yankai Yang, Shihao Han, Hamidreza Aghasi, Salman Avestimehr

Designing analog circuits from performance specifications is a complex,
multi-stage process encompassing topology selection, parameter inference, and
layout feasibility. We introduce FALCON, a unified machine learning framework
that enables fully automated, specification-driven analog circuit synthesis
through topology selection and layout-constrained optimization. Given a target
performance, FALCON first selects an appropriate circuit topology using a
performance-driven classifier guided by human design heuristics. Next, it
employs a custom, edge-centric graph neural network trained to map circuit
topology and parameters to performance, enabling gradient-based parameter
inference through the learned forward model. This inference is guided by a
differentiable layout cost, derived from analytical equations capturing
parasitic and frequency-dependent effects, and constrained by design rules. We
train and evaluate FALCON on a large-scale custom dataset of 1M analog mm-wave
circuits, generated and simulated using Cadence Spectre across 20
expert-designed topologies. Through this evaluation, FALCON demonstrates >99\%
accuracy in topology inference, <10\% relative error in performance prediction,
and efficient layout-aware design that completes in under 1 second per
instance. Together, these results position FALCON as a practical and extensible
foundation model for end-to-end analog circuit design automation.

### 4. [Physics-Informed Distillation of Diffusion Models for PDE-Constrained Generation](http://arxiv.org/pdf/2505.22391v1)

Authors: Yi Zhang, Difan Zou

Modeling physical systems in a generative manner offers several advantages,
including the ability to handle partial observations, generate diverse
solutions, and address both forward and inverse problems. Recently, diffusion
models have gained increasing attention in the modeling of physical systems,
particularly those governed by partial differential equations (PDEs). However,
diffusion models only access noisy data $\boldsymbol{x}_t$ at intermediate
steps, making it infeasible to directly enforce constraints on the clean sample
$\boldsymbol{x}_0$ at each noisy level. As a workaround, constraints are
typically applied to the expectation of clean samples
$\mathbb{E}[\boldsymbol{x}_0|\boldsymbol{x}_t]$, which is estimated using the
learned score network. However, imposing PDE constraints on the expectation
does not strictly represent the one on the true clean data, known as Jensen's
Gap. This gap creates a trade-off: enforcing PDE constraints may come at the
cost of reduced accuracy in generative modeling. To address this, we propose a
simple yet effective post-hoc distillation approach, where PDE constraints are
not injected directly into the diffusion process, but instead enforced during a
post-hoc distillation stage. We term our method as Physics-Informed
Distillation of Diffusion Models (PIDDM). This distillation not only
facilitates single-step generation with improved PDE satisfaction, but also
support both forward and inverse problem solving and reconstruction from
randomly partial observation. Extensive experiments across various PDE
benchmarks demonstrate that PIDDM significantly improves PDE satisfaction over
several recent and competitive baselines, such as PIDM, DiffusionPDE, and
ECI-sampling, with less computation overhead. Our approach can shed light on
more efficient and effective strategies for incorporating physical constraints
into diffusion models.

### Computational Geometry

### 1. [PyRigi -- a general-purpose Python package for the rigidity and flexibility of bar-and-joint frameworks](http://arxiv.org/pdf/2505.22652v1)

Authors: Matteo Gallet, Georg Grasegger, Matthias Himmelmann, Jan Legerský

We present PyRigi, a novel Python package designed to study the rigidity
properties of graphs and frameworks. Among many other capabilities, PyRigi can
determine whether a graph admits only finitely many ways, up to isometries, of
being drawn in the plane once the edge lengths are fixed, whether it has a
unique embedding, or whether it satisfied such properties even after the
removal of any of its edges. By implementing algorithms from the scientific
literature, PyRigi enables the exploration of rigidity properties of structures
that would be out of reach for computations by hand. With reliable and robust
algorithms, as well as clear, well-documented methods that are closely
connected to the underlying mathematical definitions and results, PyRigi aims
to be a practical and powerful general-purpose tool for the working
mathematician interested in rigidity theory. PyRigi is open source and easy to
use, and awaits researchers to benefit from its computational potential.

### Computation and Language

### 1. [Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](http://arxiv.org/pdf/2505.21859v1)

Authors: Vishakh Padmakumar, Zichao Wang, David Arbour, Jennifer Healey

While large language models (LLMs) are increasingly capable of handling
longer contexts, recent work has demonstrated that they exhibit the "lost in
the middle" phenomenon (Liu et al., 2024) of unevenly attending to different
parts of the provided context. This hinders their ability to cover diverse
source material in multi-document summarization, as noted in the DiverseSumm
benchmark (Huang et al., 2024). In this work, we contend that principled
content selection is a simple way to increase source coverage on this task. As
opposed to prompting an LLM to perform the summarization in a single step, we
explicitly divide the task into three steps -- (1) reducing document
collections to atomic key points, (2) using determinantal point processes (DPP)
to perform select key points that prioritize diverse content, and (3) rewriting
to the final summary. By combining prompting steps, for extraction and
rewriting, with principled techniques, for content selection, we consistently
improve source coverage on the DiverseSumm benchmark across various LLMs.
Finally, we also show that by incorporating relevance to a provided user intent
into the DPP kernel, we can generate personalized summaries that cover relevant
source information while retaining coverage.

### 2. [EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse](http://arxiv.org/pdf/2505.21889v1)

Authors: Tianyu Guo, Hande Dong, Yichong Leng, Feng Liu, Cheater Lin, Nong Xiao, Xianwei Zhang

Large language models (LLMs) are often used for infilling tasks, which
involve predicting or generating missing information in a given text. These
tasks typically require multiple interactions with similar context. To reduce
the computation of repeated historical tokens, cross-request key-value (KV)
cache reuse, a technique that stores and reuses intermediate computations, has
become a crucial method in multi-round interactive services. However, in
infilling tasks, the KV cache reuse is often hindered by the structure of the
prompt format, which typically consists of a prefix and suffix relative to the
insertion point. Specifically, the KV cache of the prefix or suffix part is
frequently invalidated as the other part (suffix or prefix) is incrementally
generated. To address the issue, we propose EFIM, a transformed prompt format
of FIM to unleash the performance potential of KV cache reuse. Although the
transformed prompt can solve the inefficiency, it exposes subtoken generation
problems in current LLMs, where they have difficulty generating partial words
accurately. Therefore, we introduce a fragment tokenization training method
which splits text into multiple fragments before tokenization during data
processing. Experiments on two representative LLMs show that LLM serving with
EFIM can lower the latency by 52% and improve the throughput by 98% while
maintaining the original infilling capability.EFIM's source code is publicly
available at https://github.com/gty111/EFIM.

### 3. [RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](http://arxiv.org/pdf/2505.21936v1)

Authors: Zeyi Liao, Jaylen Jones, Linxi Jiang, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

Computer-use agents (CUAs) promise to automate complex tasks across operating
systems (OS) and the web, but remain vulnerable to indirect prompt injection.
Current evaluations of this threat either lack support realistic but controlled
environments or ignore hybrid web-OS attack scenarios involving both
interfaces. To address this, we propose RedTeamCUA, an adversarial testing
framework featuring a novel hybrid sandbox that integrates a VM-based OS
environment with Docker-based web platforms. Our sandbox supports key features
tailored for red teaming, such as flexible adversarial scenario configuration,
and a setting that decouples adversarial evaluation from navigational
limitations of CUAs by initializing tests directly at the point of an
adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive
benchmark with 864 examples that investigate realistic, hybrid web-OS attack
scenarios and fundamental security vulnerabilities. Benchmarking current
frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA
demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated,
still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute
adversarial tasks with an Attempt Rate as high as 92.5%, although failing to
complete them due to capability limitations. Nevertheless, we observe
concerning ASRs of up to 50% in realistic end-to-end settings, with the
recently released frontier Claude 4 Opus | CUA showing an alarming ASR of 48%,
demonstrating that indirect prompt injection presents tangible risks for even
advanced CUAs despite their capabilities and safeguards. Overall, RedTeamCUA
provides an essential framework for advancing realistic, controlled, and
systematic analysis of CUA vulnerabilities, highlighting the urgent need for
robust defenses to indirect prompt injection prior to real-world deployment.

### 4. [Graph-Assisted Culturally Adaptable Idiomatic Translation for Indic Languages](http://arxiv.org/pdf/2505.21937v1)

Authors: Pratik Rakesh Singh, Kritarth Prasad, Mohammadi Zaki, Pankaj Wasnik

Translating multi-word expressions (MWEs) and idioms requires a deep
understanding of the cultural nuances of both the source and target languages.
This challenge is further amplified by the one-to-many nature of idiomatic
translations, where a single source idiom can have multiple target-language
equivalents depending on cultural references and contextual variations.
Traditional static knowledge graphs (KGs) and prompt-based approaches struggle
to capture these complex relationships, often leading to suboptimal
translations. To address this, we propose IdiomCE, an adaptive graph neural
network (GNN) based methodology that learns intricate mappings between
idiomatic expressions, effectively generalizing to both seen and unseen nodes
during training. Our proposed method enhances translation quality even in
resource-constrained settings, facilitating improved idiomatic translation in
smaller models. We evaluate our approach on multiple idiomatic translation
datasets using reference-less metrics, demonstrating significant improvements
in translating idioms from English to various Indian languages.

### 5. [RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](http://arxiv.org/pdf/2505.21940v1)

Authors: Bolei He, Xinran He, Mengke Chen, Xianwei Xue, Ying Zhu, Zhenhua Ling

Large Language Models (LLMs) excel in many areas but continue to face
challenges with complex reasoning tasks, such as Multi-Hop Question Answering
(MHQA). MHQA requires integrating evidence from diverse sources while managing
intricate logical dependencies, often leads to errors in reasoning.
Retrieval-Augmented Generation (RAG), widely employed in MHQA tasks, faces
challenges in effectively filtering noisy data and retrieving all necessary
evidence, thereby limiting its effectiveness in addressing MHQA challenges. To
address these challenges, we propose RISE:Reasoning Enhancement via Iterative
Self-Exploration, a novel framework designed to enhance models' reasoning
capability through iterative self-exploration. Specifically, RISE involves
three key steps in addressing MHQA tasks: question decomposition,
retrieve-then-read, and self-critique. By leveraging continuous
self-exploration, RISE identifies accurate reasoning paths, iteratively
self-improving the model's capability to integrate evidence, maintain logical
consistency, and enhance performance in MHQA tasks. Extensive experiments on
multiple MHQA benchmarks demonstrate that RISE significantly improves reasoning
accuracy and task performance.

### 6. [Test-Time Scaling with Repeated Sampling Improves Multilingual Text Generation](http://arxiv.org/pdf/2505.21941v1)

Authors: Ashim Gupta, Vivek Srikumar

Inference-time scaling via repeated sampling has shown promise in reasoning
tasks, but its effectiveness in multilingual generation remains underexplored.
We evaluate this approach using perplexity- and reward-based verifiers on two
multilingual benchmarks: the Aya Evaluation Suite and m-ArenaHard. Our results
show consistent quality improvements, with gains exceeding 35% in some cases.
While perplexity-based scoring is effective for open-ended prompts, only
reward-based verifiers improve performance on tasks requiring reasoning (e.g.,
math, code). Our results demonstrate the broader utility of repeated sampling
for multilingual text generation and underscore the importance of selecting
right verifiers for the task.

### 7. [Resolving Knowledge Conflicts in Domain-specific Data Selection: A Case Study on Medical Instruction-tuning](http://arxiv.org/pdf/2505.21958v1)

Authors: Qihuang Zhong, Liang Ding, Fei Liao, Juhua Liu, Bo Du, Dacheng Tao

Domain-specific instruction-tuning has become the defacto standard for
improving the performance of large language models (LLMs) in specialized
applications, e.g., medical question answering. Since the instruction-tuning
dataset might contain redundant or low-quality data, data selection (DS) is
usually required to maximize the data efficiency. Despite the successes in the
general domain, current DS methods often struggle to select the desired data
for domain-specific instruction-tuning. One of the main reasons is that they
neglect the impact of knowledge conflicts, i.e., the discrepancy between LLMs'
pretrained knowledge and context knowledge of instruction data, which could
damage LLMs' prior abilities and lead to hallucination. To this end, we propose
a simple-yet-effective Knowledge-aware Data Selection (namely KDS) framework to
select the domain-specific instruction-tuning data that meets LLMs' actual
needs. The core of KDS is to leverage two knowledge-aware metrics for
quantitatively measuring knowledge conflicts from two aspects: context-memory
knowledge alignment and intra-memory knowledge consistency. By filtering the
data with large knowledge conflicts and sampling the high-quality and diverse
data, KDS can effectively stimulate the LLMs' abilities and achieve better
domain-specific performance. Taking the medical domain as the testbed, we
conduct extensive experiments and empirically prove that KDS surpasses the
other baselines and brings significant and consistent performance gains among
all LLMs. More encouragingly, KDS effectively improves the model generalization
and alleviates the hallucination problem.

### 8. [Seeing the Threat: Vulnerabilities in Vision-Language Models to Adversarial Attack](http://arxiv.org/pdf/2505.21967v1)

Authors: Juan Ren, Mark Dras, Usman Naseem

Large Vision-Language Models (LVLMs) have shown remarkable capabilities
across a wide range of multimodal tasks. However, their integration of visual
inputs introduces expanded attack surfaces, thereby exposing them to novel
security vulnerabilities. In this work, we conduct a systematic
representational analysis to uncover why conventional adversarial attacks can
circumvent the safety mechanisms embedded in LVLMs. We further propose a novel
two stage evaluation framework for adversarial attacks on LVLMs. The first
stage differentiates among instruction non compliance, outright refusal, and
successful adversarial exploitation. The second stage quantifies the degree to
which the model's output fulfills the harmful intent of the adversarial prompt,
while categorizing refusal behavior into direct refusals, soft refusals, and
partial refusals that remain inadvertently helpful. Finally, we introduce a
normative schema that defines idealized model behavior when confronted with
harmful prompts, offering a principled target for safety alignment in
multimodal systems.

### 9. [Pearl: A Multimodal Culturally-Aware Arabic Instruction Dataset](http://arxiv.org/pdf/2505.21979v1)

Authors: Fakhraddin Alwajih, Samar Mohamed Magdy, Abdellah El Mekki, Omer Nacar, Youssef Nafea, Safaa Taher Abdelfadil, Abdulfattah Mohammed Yahya, Hamzah Luqman, Nada Almarwani, Samah Aloufi, Baraah Qawasmeh, Houdaifa Atou, Serry Sibaee, Hamzah A. Alsayadi, Walid Al-Dhabyani, Maged S. Al-shaibani, Aya El aatar, Nour Qandos, Rahaf Alhamouri, Samar Ahmad, Razan Khassib, Lina Hamad, Mohammed Anwar AL-Ghrawi, Fatimah Alshamari, Cheikh Malainine, Doaa Qawasmeh, Aminetou Yacoub, Tfeil moilid, Ruwa AbuHweidi, Ahmed Aboeitta, Vatimetou Mohamed Lemin, Reem Abdel-Salam, Ahlam Bashiti, Adel Ammar, Aisha Alansari, Ahmed Ashraf, Nora Alturayeif, Sara Shatnawi, Alcides Alcoba Inciarte, AbdelRahim A. Elmadany, Mohamedou cheikh tourad, Ismail Berrada, Mustafa Jarrar, Shady Shehata, Muhammad Abdul-Mageed

Mainstream large vision-language models (LVLMs) inherently encode cultural
biases, highlighting the need for diverse multimodal datasets. To address this
gap, we introduce Pearl, a large-scale Arabic multimodal dataset and benchmark
explicitly designed for cultural understanding. Constructed through advanced
agentic workflows and extensive human-in-the-loop annotations by 45 annotators
from across the Arab world, Pearl comprises over K multimodal examples spanning
ten culturally significant domains covering all Arab countries. We further
provide two robust evaluation benchmarks Pearl and Pearl-Lite along with a
specialized subset Pearl-X explicitly developed to assess nuanced cultural
variations. Comprehensive evaluations on state-of-the-art open and proprietary
LVLMs demonstrate that reasoning-centric instruction alignment substantially
improves models' cultural grounding compared to conventional scaling methods.
Pearl establishes a foundational resource for advancing culturally-informed
multimodal modeling research. All datasets and benchmarks are publicly
available.

### 10. [Leveraging Interview-Informed LLMs to Model Survey Responses: Comparative Insights from AI-Generated and Human Data](http://arxiv.org/pdf/2505.21997v1)

Authors: Jihong Zhang, Xinya Liang, Anqi Deng, Nicole Bonge, Lin Tan, Ling Zhang, Nicole Zarrett

Mixed methods research integrates quantitative and qualitative data but faces
challenges in aligning their distinct structures, particularly in examining
measurement characteristics and individual response patterns. Advances in large
language models (LLMs) offer promising solutions by generating synthetic survey
responses informed by qualitative data. This study investigates whether LLMs,
guided by personal interviews, can reliably predict human survey responses,
using the Behavioral Regulations in Exercise Questionnaire (BREQ) and
interviews from after-school program staff as a case study. Results indicate
that LLMs capture overall response patterns but exhibit lower variability than
humans. Incorporating interview data improves response diversity for some
models (e.g., Claude, GPT), while well-crafted prompts and low-temperature
settings enhance alignment between LLM and human responses. Demographic
information had less impact than interview content on alignment accuracy. These
findings underscore the potential of interview-informed LLMs to bridge
qualitative and quantitative methodologies while revealing limitations in
response variability, emotional interpretation, and psychometric fidelity.
Future research should refine prompt design, explore bias mitigation, and
optimize model settings to enhance the validity of LLM-generated survey data in
social science research.

### Cryptography and Security

### 1. [VulBinLLM: LLM-powered Vulnerability Detection for Stripped Binaries](http://arxiv.org/pdf/2505.22010v1)

Authors: Nasir Hussain, Haohan Chen, Chanh Tran, Philip Huang, Zhuohao Li, Pravir Chugh, William Chen, Ashish Kundu, Yuan Tian

Recognizing vulnerabilities in stripped binary files presents a significant
challenge in software security. Although some progress has been made in
generating human-readable information from decompiled binary files with Large
Language Models (LLMs), effectively and scalably detecting vulnerabilities
within these binary files is still an open problem. This paper explores the
novel application of LLMs to detect vulnerabilities within these binary files.
We demonstrate the feasibility of identifying vulnerable programs through a
combined approach of decompilation optimization to make the vulnerabilities
more prominent and long-term memory for a larger context window, achieving
state-of-the-art performance in binary vulnerability analysis. Our findings
highlight the potential for LLMs to overcome the limitations of traditional
analysis methods and advance the field of binary vulnerability detection,
paving the way for more secure software systems. In this paper, we present
Vul-BinLLM , an LLM-based framework for binary vulnerability detection that
mirrors traditional binary analysis workflows with fine-grained optimizations
in decompilation and vulnerability reasoning with an extended context. In the
decompilation phase, Vul-BinLLM adds vulnerability and weakness comments
without altering the code structure or functionality, providing more contextual
information for vulnerability reasoning later. Then for vulnerability
reasoning, Vul-BinLLM combines in-context learning and chain-of-thought
prompting along with a memory management agent to enhance accuracy. Our
evaluations encompass the commonly used synthetic dataset Juliet to evaluate
the potential feasibility for analysis and vulnerability detection in C/C++
binaries. Our evaluations show that Vul-BinLLM is highly effective in detecting
vulnerabilities on the compiled Juliet dataset.

### 2. [A Comparative Study of Fuzzers and Static Analysis Tools for Finding Memory Unsafety in C and C++](http://arxiv.org/pdf/2505.22052v1)

Authors: Keno Hassler, Philipp Görz, Stephan Lipp, Thorsten Holz, Marcel Böhme

Even today, over 70% of security vulnerabilities in critical software systems
result from memory safety violations. To address this challenge, fuzzing and
static analysis are widely used automated methods to discover such
vulnerabilities. Fuzzing generates random program inputs to identify faults,
while static analysis examines source code to detect potential vulnerabilities.
Although these techniques share a common goal, they take fundamentally
different approaches and have evolved largely independently.
  In this paper, we present an empirical analysis of five static analyzers and
13 fuzzers, applied to over 100 known security vulnerabilities in C/C++
programs. We measure the number of bug reports generated for each vulnerability
to evaluate how the approaches differ and complement each other. Moreover, we
randomly sample eight bug-containing functions, manually analyze all bug
reports therein, and quantify false-positive rates. We also assess limits to
bug discovery, ease of use, resource requirements, and integration into the
development process. We find that both techniques discover different types of
bugs, but there are clear winners for each. Developers should consider these
tools depending on their specific workflow and usability requirements. Based on
our findings, we propose future directions to foster collaboration between
these research domains.

### 3. [Accountable, Scalable and DoS-resilient Secure Vehicular Communication](http://arxiv.org/pdf/2505.22162v1)

Authors: Hongyu Jin, Panos Papadimitratos

Paramount to vehicle safety, broadcasted Cooperative Awareness Messages
(CAMs) and Decentralized Environmental Notification Messages (DENMs) are
pseudonymously authenticated for security and privacy protection, with each
node needing to have all incoming messages validated within an expiration
deadline. This creates an asymmetry that can be easily exploited by external
adversaries to launch a clogging Denial of Service (DoS) attack: each forged VC
message forces all neighboring nodes to cryptographically validate it; at
increasing rates, easy to generate forged messages gradually exhaust processing
resources and severely degrade or deny timely validation of benign CAMs/DENMs.
The result can be catastrophic when awareness of neighbor vehicle positions or
critical reports are missed. We address this problem making the standardized VC
pseudonymous authentication DoS-resilient. We propose efficient cryptographic
constructs, which we term message verification facilitators, to prioritize
processing resources for verification of potentially valid messages among bogus
messages and verify multiple messages based on one signature verification. Any
message acceptance is strictly based on public-key based message
authentication/verification for accountability, i.e., non-repudiation is not
sacrificed, unlike symmetric key based approaches. This further enables drastic
misbehavior detection, also exploiting the newly introduced facilitators, based
on probabilistic signature verification and cross-checking over multiple
facilitators verifying the same message; while maintaining verification latency
low even when under attack, trading off modest communication overhead. Our
facilitators can also be used for efficient discovery and verification of DENM
or any event-driven message, including misbehavior evidence used for our
scheme.

### 4. [Does Johnny Get the Message? Evaluating Cybersecurity Notifications for Everyday Users](http://arxiv.org/pdf/2505.22435v1)

Authors: Victor Jüttner, Erik Buchmann

Due to the increasing presence of networked devices in everyday life, not
only cybersecurity specialists but also end users benefit from security
applications such as firewalls, vulnerability scanners, and intrusion detection
systems. Recent approaches use large language models (LLMs) to rewrite brief,
technical security alerts into intuitive language and suggest actionable
measures, helping everyday users understand and respond appropriately to
security risks. However, it remains an open question how well such alerts are
explained to users. LLM outputs can also be hallucinated, inconsistent, or
misleading. In this work, we introduce the Human-Centered Security Alert
Evaluation Framework (HCSAEF). HCSAEF assesses LLM-generated cybersecurity
notifications to support researchers who want to compare notifications
generated for everyday users, improve them, or analyze the capabilities of
different LLMs in explaining cybersecurity issues. We demonstrate HCSAEF
through three use cases, which allow us to quantify the impact of prompt
design, model selection, and output consistency. Our findings indicate that
HCSAEF effectively differentiates generated notifications along dimensions such
as intuitiveness, urgency, and correctness.

### 5. [Privacy-preserving Prompt Personalization in Federated Learning for Multimodal Large Language Models](http://arxiv.org/pdf/2505.22447v1)

Authors: Sizai Hou, Songze Li, Baturalp Buyukates

Prompt learning is a crucial technique for adapting pre-trained multimodal
language models (MLLMs) to user tasks. Federated prompt personalization (FPP)
is further developed to address data heterogeneity and local overfitting,
however, it exposes personalized prompts - valuable intellectual assets - to
privacy risks like prompt stealing or membership inference attacks.
Widely-adopted techniques like differential privacy add noise to prompts,
whereas degrading personalization performance. We propose SecFPP, a secure FPP
protocol harmonizing generalization, personalization, and privacy guarantees.
SecFPP employs hierarchical prompt adaptation with domain-level and class-level
components to handle multi-granular data imbalance. For privacy, it uses a
novel secret-sharing-based adaptive clustering algorithm for domain-level
adaptation while keeping class-level components private. While theoretically
and empirically secure, SecFPP achieves state-of-the-art accuracy under severe
heterogeneity in data distribution. Extensive experiments show it significantly
outperforms both non-private and privacy-preserving baselines, offering a
superior privacy-performance trade-off.

### 6. [Transformers for Secure Hardware Systems: Applications, Challenges, and Outlook](http://arxiv.org/pdf/2505.22605v1)

Authors: Banafsheh Saber Latibari, Najmeh Nazari, Avesta Sasan, Houman Homayoun, Pratik Satam, Soheil Salehi, Hossein Sayadi

The rise of hardware-level security threats, such as side-channel attacks,
hardware Trojans, and firmware vulnerabilities, demands advanced detection
mechanisms that are more intelligent and adaptive. Traditional methods often
fall short in addressing the complexity and evasiveness of modern attacks,
driving increased interest in machine learning-based solutions. Among these,
Transformer models, widely recognized for their success in natural language
processing and computer vision, have gained traction in the security domain due
to their ability to model complex dependencies, offering enhanced capabilities
in identifying vulnerabilities, detecting anomalies, and reinforcing system
integrity. This survey provides a comprehensive review of recent advancements
on the use of Transformers in hardware security, examining their application
across key areas such as side-channel analysis, hardware Trojan detection,
vulnerability classification, device fingerprinting, and firmware security.
Furthermore, we discuss the practical challenges of applying Transformers to
secure hardware systems, and highlight opportunities and future research
directions that position them as a foundation for next-generation
hardware-assisted security. These insights pave the way for deeper integration
of AI-driven techniques into hardware security frameworks, enabling more
resilient and intelligent defenses.

### 7. [Securing the Software Package Supply Chain for Critical Systems](http://arxiv.org/pdf/2505.22023v1)

Authors: Ritwik Murali, Akash Ravi

Software systems have grown as an indispensable commodity used across various
industries, and almost all essential services depend on them for effective
operation. The software is no longer an independent or stand-alone piece of
code written by a developer but rather a collection of packages designed by
multiple developers across the globe. Ensuring the reliability and resilience
of these systems is crucial since emerging threats target software supply
chains, as demonstrated by the widespread SolarWinds hack in late 2020. These
supply chains extend beyond patches and updates, involving distribution
networks throughout the software lifecycle. Industries like smart grids,
manufacturing, healthcare, and finance rely on interconnected software systems
and their dependencies for effective functioning. To secure software modules
and add-ons, robust distribution architectures are essential. The proposed
chapter enhances the existing delivery frameworks by including a permissioned
ledger with Proof of Authority consensus and multi-party signatures. The
proposed system aims to prevent attacks while permitting every stakeholder to
verify the same. Critical systems can interface with the secure pipeline
without disrupting existing functionalities, thus preventing the cascading
effect of an attack at any point in the supply chain.

### 8. [Domainator: Detecting and Identifying DNS-Tunneling Malware Using Metadata Sequences](http://arxiv.org/pdf/2505.22220v1)

Authors: Denis Petrov, Pascal Ruffing, Sebastian Zillien, Steffen Wendzel

In recent years, malware with tunneling (or: covert channel) capabilities is
on the rise. While malware research led to several methods and innovations, the
detection and differentiation of malware solely based on its DNS tunneling
features is still in its infancy. Moreover, no work so far has used the DNS
tunneling traffic to gain knowledge over the current actions taken by the
malware. In this paper, we present Domainator, an approach to detect and
differentiate state-of-the-art malware and DNS tunneling tools without relying
on trivial (but quickly altered) features such as "magic bytes" that are
embedded into subdomains. Instead, we apply an analysis of sequential patterns
to identify specific types of malware. We evaluate our approach with 7
different malware samples and tunneling tools and can identify the particular
malware based on its DNS traffic. We further infer the rough behavior of the
particular malware through its DNS tunneling artifacts. Finally, we compare our
Domainator with related methods.

### 9. [Private Lossless Multiple Release](http://arxiv.org/pdf/2505.22449v1)

Authors: Joel Daniel Andersson, Lukas Retschmeier, Boel Nelson, Rasmus Pagh

Koufogiannis et al. (2016) showed a $\textit{gradual release}$ result for
Laplace noise-based differentially private mechanisms: given an
$\varepsilon$-DP release, a new release with privacy parameter $\varepsilon' >
\varepsilon$ can be computed such that the combined privacy loss of both
releases is at most $\varepsilon'$ and the distribution of the latter is the
same as a single release with parameter $\varepsilon'$. They also showed
gradual release techniques for Gaussian noise, later also explored by
Whitehouse et al. (2022).
  In this paper, we consider a more general $\textit{multiple release}$ setting
in which analysts hold private releases with different privacy parameters
corresponding to different access/trust levels. These releases are determined
one by one, with privacy parameters in arbitrary order. A multiple release is
$\textit{lossless}$ if having access to a subset $S$ of the releases has the
same privacy guarantee as the least private release in $S$, and each release
has the same distribution as a single release with the same privacy parameter.
Our main result is that lossless multiple release is possible for a large class
of additive noise mechanisms. For the Gaussian mechanism we give a simple
method for lossless multiple release with a short, self-contained analysis that
does not require knowledge of the mathematics of Brownian motion. We also
present lossless multiple release for the Laplace and Poisson mechanisms.
Finally, we consider how to efficiently do gradual release of sparse
histograms, and present a mechanism with running time independent of the number
of dimensions.

### 10. [BPMN to Smart Contract by Business Analyst](http://arxiv.org/pdf/2505.22612v1)

Authors: C. G. Liu, P. Bodorik, D. Jutla

This paper addresses the challenge of creating smart contracts for
applications represented using Business Process Management and Notation (BPMN)
models. In our prior work we presented a methodology that automates the
generation of smart contracts from BPMN models. This approach abstracts the
BPMN flow control, making it independent of the underlying blockchain
infrastructure, with only the BPMN task elements requiring coding. In
subsequent research, we enhanced our approach by adding support for nested
transactions and enabling a smart contract repair and/or upgrade. To empower
Business Analysts (BAs) to generate smart contracts without relying on software
developers, we tackled the challenge of generating smart contracts from BPMN
models without assistance of a software developer. We exploit the Decision
Model and Notation (DMN) standard to represent the decisions and the business
logic of the BPMN task elements and amended our methodology for transformation
of BPMN models into smart contracts to support also the generation script to
represent the business logic represented by the DMN models. To support such
transformation, we describe how the BA documents, using the BPMN elements, the
flow of information along with the flow of execution. Thus, if the BA is
successful in representing the blockchain application requirements using BPMN
and DMN models, our methodology and the tool, called TABS, that we developed as
a proof of concept, is used to generate the smart contracts directly from those
models without developer assistance.

### Computer Vision and Pattern Recognition

### 1. [Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation](http://arxiv.org/pdf/2505.21844v1)

Authors: Mehrdad Noori, David Osowiechi, Gustavo Adolfo Vargas Hakim, Ali Bahri, Moslem Yazdanpanah, Sahar Dastani, Farzad Beizaee, Ismail Ben Ayed, Christian Desrosiers

Recently, test-time adaptation has attracted wide interest in the context of
vision-language models for image classification. However, to the best of our
knowledge, the problem is completely overlooked in dense prediction tasks such
as Open-Vocabulary Semantic Segmentation (OVSS). In response, we propose a
novel TTA method tailored to adapting VLMs for segmentation during test time.
Unlike TTA methods for image classification, our Multi-Level and Multi-Prompt
(MLMP) entropy minimization integrates features from intermediate
vision-encoder layers and is performed with different text-prompt templates at
both the global CLS token and local pixel-wise levels. Our approach could be
used as plug-and-play for any segmentation network, does not require additional
training data or labels, and remains effective even with a single test sample.
Furthermore, we introduce a comprehensive OVSS TTA benchmark suite, which
integrates a rigorous evaluation protocol, seven segmentation datasets, and 15
common corruptions, with a total of 82 distinct test scenarios, establishing a
standardized and comprehensive testbed for future TTA research in
open-vocabulary segmentation. Our experiments on this suite demonstrate that
our segmentation-tailored method consistently delivers significant gains over
direct adoption of TTA classification baselines.

### 2. [FPAN: Mitigating Replication in Diffusion Models through the Fine-Grained Probabilistic Addition of Noise to Token Embeddings](http://arxiv.org/pdf/2505.21848v1)

Authors: Jingqi Xu, Chenghao Li, Yuke Zhang, Peter A. Beerel

Diffusion models have demonstrated remarkable potential in generating
high-quality images. However, their tendency to replicate training data raises
serious privacy concerns, particularly when the training datasets contain
sensitive or private information. Existing mitigation strategies primarily
focus on reducing image duplication, modifying the cross-attention mechanism,
and altering the denoising backbone architecture of diffusion models. Moreover,
recent work has shown that adding a consistent small amount of noise to text
embeddings can reduce replication to some degree. In this work, we begin by
analyzing the impact of adding varying amounts of noise. Based on our analysis,
we propose a fine-grained noise injection technique that probabilistically adds
a larger amount of noise to token embeddings. We refer to our method as
Fine-grained Probabilistic Addition of Noise (FPAN). Through our extensive
experiments, we show that our proposed FPAN can reduce replication by an
average of 28.78% compared to the baseline diffusion model without
significantly impacting image quality, and outperforms the prior
consistent-magnitude-noise-addition approach by 26.51%. Moreover, when combined
with other existing mitigation methods, our FPAN approach can further reduce
replication by up to 16.82% with similar, if not improved, image quality.

### 3. [Towards Scalable Language-Image Pre-training for 3D Medical Imaging](http://arxiv.org/pdf/2505.21862v1)

Authors: Chenhui Zhao, Yiwei Lyu, Asadur Chowdury, Edward Harake, Akhil Kondepudi, Akshay Rao, Xinhai Hou, Honglak Lee, Todd Hollon

Language-image pre-training has demonstrated strong performance in 2D medical
imaging, but its success in 3D modalities such as CT and MRI remains limited
due to the high computational demands of volumetric data, which pose a
significant barrier to training on large-scale, uncurated clinical studies. In
this study, we introduce Hierarchical attention for Language-Image Pre-training
(HLIP), a scalable pre-training framework for 3D medical imaging. HLIP adopts a
lightweight hierarchical attention mechanism inspired by the natural hierarchy
of radiology data: slice, scan, and study. This mechanism exhibits strong
generalizability, e.g., +4.3% macro AUC on the Rad-ChestCT benchmark when
pre-trained on CT-RATE. Moreover, the computational efficiency of HLIP enables
direct training on uncurated datasets. Trained on 220K patients with 3.13
million scans for brain MRI and 240K patients with 1.44 million scans for head
CT, HLIP achieves state-of-the-art performance, e.g., +32.4% balanced ACC on
the proposed publicly available brain MRI benchmark Pub-Brain-5; +1.4% and
+6.9% macro AUC on head CT benchmarks RSNA and CQ500, respectively. These
results demonstrate that, with HLIP, directly pre-training on uncurated
clinical datasets is a scalable and effective direction for language-image
pre-training in 3D medical imaging. The code is available at
https://github.com/Zch0414/hlip

### 4. [Cross-DINO: Cross the Deep MLP and Transformer for Small Object Detection](http://arxiv.org/pdf/2505.21868v1)

Authors: Guiping Cao, Wenjian Huang, Xiangyuan Lan, Jianguo Zhang, Dongmei Jiang, Yaowei Wang

Small Object Detection (SOD) poses significant challenges due to limited
information and the model's low class prediction score. While Transformer-based
detectors have shown promising performance, their potential for SOD remains
largely unexplored. In typical DETR-like frameworks, the CNN backbone network,
specialized in aggregating local information, struggles to capture the
necessary contextual information for SOD. The multiple attention layers in the
Transformer Encoder face difficulties in effectively attending to small objects
and can also lead to blurring of features. Furthermore, the model's lower class
prediction score of small objects compared to large objects further increases
the difficulty of SOD. To address these challenges, we introduce a novel
approach called Cross-DINO. This approach incorporates the deep MLP network to
aggregate initial feature representations with both short and long range
information for SOD. Then, a new Cross Coding Twice Module (CCTM) is applied to
integrate these initial representations to the Transformer Encoder feature,
enhancing the details of small objects. Additionally, we introduce a new kind
of soft label named Category-Size (CS), integrating the Category and Size of
objects. By treating CS as new ground truth, we propose a new loss function
called Boost Loss to improve the class prediction score of the model. Extensive
experimental results on COCO, WiderPerson, VisDrone, AI-TOD, and SODA-D
datasets demonstrate that Cross-DINO efficiently improves the performance of
DETR-like models on SOD. Specifically, our model achieves 36.4% APs on COCO for
SOD with only 45M parameters, outperforming the DINO by +4.4% APS (36.4% vs.
32.0%) with fewer parameters and FLOPs, under 12 epochs training setting. The
source codes will be available at https://github.com/Med-Process/Cross-DINO.

### 5. [Hyperspectral Gaussian Splatting](http://arxiv.org/pdf/2505.21890v1)

Authors: Sunil Kumar Narayanan, Lingjun Zhao, Lu Gan, Yongsheng Chen

Hyperspectral imaging (HSI) has been widely used in agricultural applications
for non-destructive estimation of plant nutrient composition and precise
determination of nutritional elements in samples. Recently, 3D reconstruction
methods have been used to create implicit neural representations of HSI scenes,
which can help localize the target object's nutrient composition spatially and
spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit
representation that can render hyperspectral channel compositions of each
spatial location from any viewing direction. However, it faces limitations in
training time and rendering speed. In this paper, we propose Hyperspectral
Gaussian Splatting (HS-GS), which combines the state-of-the-art 3D Gaussian
Splatting (3DGS) with a diffusion model to enable 3D explicit reconstruction of
the hyperspectral scenes and novel view synthesis for the entire spectral
range. To enhance the model's ability to capture fine-grained reflectance
variations across the light spectrum and leverage correlations between adjacent
wavelengths for denoising, we introduce a wavelength encoder to generate
wavelength-specific spherical harmonics offsets. We also introduce a novel
Kullback--Leibler divergence-based loss to mitigate the spectral distribution
gap between the rendered image and the ground truth. A diffusion model is
further applied for denoising the rendered images and generating photorealistic
hyperspectral images. We present extensive evaluations on five diverse
hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of
our proposed HS-GS framework. The results demonstrate that HS-GS achieves new
state-of-the-art performance among all previously published methods. Code will
be released upon publication.

### 6. [Concentrate on Weakness: Mining Hard Prototypes for Few-Shot Medical Image Segmentation](http://arxiv.org/pdf/2505.21897v1)

Authors: Jianchao Jiang, Haofeng Zhang

Few-Shot Medical Image Segmentation (FSMIS) has been widely used to train a
model that can perform segmentation from only a few annotated images. However,
most existing prototype-based FSMIS methods generate multiple prototypes from
the support image solely by random sampling or local averaging, which can cause
particularly severe boundary blurring due to the tendency for normal features
accounting for the majority of features of a specific category. Consequently,
we propose to focus more attention to those weaker features that are crucial
for clear segmentation boundary. Specifically, we design a Support
Self-Prediction (SSP) module to identify such weak features by comparing true
support mask with one predicted by global support prototype. Then, a Hard
Prototypes Generation (HPG) module is employed to generate multiple hard
prototypes based on these weak features. Subsequently, a Multiple Similarity
Maps Fusion (MSMF) module is devised to generate final segmenting mask in a
dual-path fashion to mitigate the imbalance between foreground and background
in medical images. Furthermore, we introduce a boundary loss to further
constraint the edge of segmentation. Extensive experiments on three publicly
available medical image datasets demonstrate that our method achieves
state-of-the-art performance. Code is available at
https://github.com/jcjiang99/CoW.

### 7. [AlignGen: Boosting Personalized Image Generation with Cross-Modality Prior Alignment](http://arxiv.org/pdf/2505.21911v1)

Authors: Yiheng Lin, Shifang Zhao, Ting Liu, Xiaochao Qu, Luoqi Liu, Yao Zhao, Yunchao Wei

Personalized image generation aims to integrate user-provided concepts into
text-to-image models, enabling the generation of customized content based on a
given prompt. Recent zero-shot approaches, particularly those leveraging
diffusion transformers, incorporate reference image information through
multi-modal attention mechanism. This integration allows the generated output
to be influenced by both the textual prior from the prompt and the visual prior
from the reference image. However, we observe that when the prompt and
reference image are misaligned, the generated results exhibit a stronger bias
toward the textual prior, leading to a significant loss of reference content.
To address this issue, we propose AlignGen, a Cross-Modality Prior Alignment
mechanism that enhances personalized image generation by: 1) introducing a
learnable token to bridge the gap between the textual and visual priors, 2)
incorporating a robust training strategy to ensure proper prior alignment, and
3) employing a selective cross-modal attention mask within the multi-modal
attention mechanism to further align the priors. Experimental results
demonstrate that AlignGen outperforms existing zero-shot methods and even
surpasses popular test-time optimization approaches.

### 8. [LiDARDustX: A LiDAR Dataset for Dusty Unstructured Road Environments](http://arxiv.org/pdf/2505.21914v1)

Authors: Chenfeng Wei, Qi Wu, Si Zuo, Jiahua Xu, Boyang Zhao, Zeyu Yang, Guotao Xie, Shenhong Wang

Autonomous driving datasets are essential for validating the progress of
intelligent vehicle algorithms, which include localization, perception, and
prediction. However, existing datasets are predominantly focused on structured
urban environments, which limits the exploration of unstructured and
specialized scenarios, particularly those characterized by significant dust
levels. This paper introduces the LiDARDustX dataset, which is specifically
designed for perception tasks under high-dust conditions, such as those
encountered in mining areas. The LiDARDustX dataset consists of 30,000 LiDAR
frames captured by six different LiDAR sensors, each accompanied by 3D bounding
box annotations and point cloud semantic segmentation. Notably, over 80% of the
dataset comprises dust-affected scenes. By utilizing this dataset, we have
established a benchmark for evaluating the performance of state-of-the-art 3D
detection and segmentation algorithms. Additionally, we have analyzed the
impact of dust on perception accuracy and delved into the causes of these
effects. The data and further information can be accessed at:
https://github.com/vincentweikey/LiDARDustX.

### 9. [BD Open LULC Map: High-resolution land use land cover mapping & benchmarking for urban development in Dhaka, Bangladesh](http://arxiv.org/pdf/2505.21915v1)

Authors: Mir Sazzat Hossain, Ovi Paul, Md Akil Raihan Iftee, Rakibul Hasan Rajib, Abu Bakar Siddik Nayem, Anis Sarker, Arshad Momen, Md. Ashraful Amin, Amin Ahsan Ali, AKM Mahbubur Rahman

Land Use Land Cover (LULC) mapping using deep learning significantly enhances
the reliability of LULC classification, aiding in understanding geography,
socioeconomic conditions, poverty levels, and urban sprawl. However, the
scarcity of annotated satellite data, especially in South/East Asian developing
countries, poses a major challenge due to limited funding, diverse
infrastructures, and dense populations. In this work, we introduce the BD Open
LULC Map (BOLM), providing pixel-wise LULC annotations across eleven classes
(e.g., Farmland, Water, Forest, Urban Structure, Rural Built-Up) for Dhaka
metropolitan city and its surroundings using high-resolution Bing satellite
imagery (2.22 m/pixel). BOLM spans 4,392 sq km (891 million pixels), with
ground truth validated through a three-stage process involving GIS experts. We
benchmark LULC segmentation using DeepLab V3+ across five major classes and
compare performance on Bing and Sentinel-2A imagery. BOLM aims to support
reliable deep models and domain adaptation tasks, addressing critical LULC
dataset gaps in South/East Asia.

### 10. [InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective](http://arxiv.org/pdf/2505.21920v1)

Authors: Yuanhong Zhang, Muyao Yuan, Weizhan Zhang, Tieliang Gong, Wen Wen, Jiangyong Ying, Weijie Shi

The Segment Anything Model (SAM), a vision foundation model, exhibits
impressive zero-shot capabilities in general tasks but struggles in specialized
domains. Parameter-efficient fine-tuning (PEFT) is a promising approach to
unleash the potential of SAM in novel scenarios. However, existing PEFT methods
for SAM neglect the domain-invariant relations encoded in the pre-trained
model. To bridge this gap, we propose InfoSAM, an information-theoretic
approach that enhances SAM fine-tuning by distilling and preserving its
pre-trained segmentation knowledge. Specifically, we formulate the knowledge
transfer process as two novel mutual information-based objectives: (i) to
compress the domain-invariant relation extracted from pre-trained SAM,
excluding pseudo-invariant information as possible, and (ii) to maximize mutual
information between the relational knowledge learned by the teacher
(pre-trained SAM) and the student (fine-tuned model). The proposed InfoSAM
establishes a robust distillation framework for PEFT of SAM. Extensive
experiments across diverse benchmarks validate InfoSAM's effectiveness in
improving SAM family's performance on real-world tasks, demonstrating its
adaptability and superiority in handling specialized scenarios.

### Computers and Society

### 1. [A Closer Look at the Existing Risks of Generative AI: Mapping the Who, What, and How of Real-World Incidents](http://arxiv.org/pdf/2505.22073v1)

Authors: Megan Li, Wendy Bickersteth, Ningjing Tang, Jason Hong, Lorrie Cranor, Hong Shen, Hoda Heidari

Due to its general-purpose nature, Generative AI is applied in an
ever-growing set of domains and tasks, leading to an expanding set of risks of
harm impacting people, communities, society, and the environment. These risks
may arise due to failures during the design and development of the technology,
as well as during its release, deployment, or downstream usages and
appropriations of its outputs. In this paper, building on prior taxonomies of
AI risks, harms, and failures, we construct a taxonomy specifically for
Generative AI failures and map them to the harms they precipitate. Through a
systematic analysis of 499 publicly reported incidents, we describe what harms
are reported, how they arose, and who they impact. We report the prevalence of
each type of harm, underlying failure mode, and harmed stakeholder, as well as
their common co-occurrences. We find that most reported incidents are caused by
use-related issues but bring harm to parties beyond the end user(s) of the
Generative AI system at fault, and that the landscape of Generative AI harms is
distinct from that of traditional AI. Our work offers actionable insights to
policymakers, developers, and Generative AI users. In particular, we call for
the prioritization of non-technical risk and harm mitigation strategies,
including public disclosures and education and careful regulatory stances.

### 2. [Facial Age Estimation: A Research Roadmap for Technological and Legal Development and Deployment](http://arxiv.org/pdf/2505.22401v1)

Authors: Richard Guest, Eva Lievens, Martin Sas, Elena Botoeva, Temitope Adeyemo, Valerie Verdoodt, Elora Fernandes, Chris Allgrove

Automated facial age assessment systems operate in either estimation mode -
predicting age based on facial traits, or verification mode - confirming a
claimed age. These systems support access control to age-restricted goods,
services, and content, and can be used in areas like e-commerce, social media,
forensics, and refugee support. They may also personalise services in
healthcare, finance, and advertising. While improving technological accuracy is
essential, deployment must consider legal, ethical, sociological, alongside
technological factors. This white paper reviews the current challenges in
deploying such systems, outlines the relevant legal and regulatory landscape,
and explores future research for fair, robust, and ethical age estimation
technologies.

### 3. [AI instructional agent improves student's perceived learner control and learning outcome: empirical evidence from a randomized controlled trial](http://arxiv.org/pdf/2505.22526v1)

Authors: Fei Qin, Zhanxin Hao, Jifan Yu, Zhiyuan Liu, Yu Zhang

This study examines the impact of an AI instructional agent on students'
perceived learner control and academic performance in a medium demanding course
with lecturing as the main teaching strategy. Based on a randomized controlled
trial, three instructional conditions were compared: a traditional human
teacher, a self-paced MOOC with chatbot support, and an AI instructional agent
capable of delivering lectures and responding to questions in real time.
Students in the AI instructional agent group reported significantly higher
levels of perceived learner control compared to the other groups. They also
completed the learning task more efficiently and engaged in more frequent
interactions with the instructional system. Regression analyzes showed that
perceived learner control positively predicted post-test performance, with
behavioral indicators such as reduced learning time and higher interaction
frequency supporting this relationship. These findings suggest that AI
instructional agents, when designed to support personalized pace and responsive
interaction, can enhance both students' learning experience and learning
outcomes.

### 4. [Navigating the AI-Energy Nexus with Geopolitical Insight](http://arxiv.org/pdf/2505.22639v1)

Authors: Nidhi Kalra, Robin Wang, Ismael Arciniegas Rueda

This working paper examines how geopolitical strategies and energy resource
management intersect with Artificial Intelligence (AI) development, delineating
the AI-energy nexus as critical to sustaining U.S. AI leadership. By analyzing
the centralized approaches of authoritarian regimes like China and Gulf
nations, alongside market-driven approaches in the U.S., the paper explores
divergent strategies to allocate resources for AI energy needs. It underscores
the role of energy infrastructure, market dynamics, and state-led initiatives
in shaping global AI competition. Recommendations include adopting
geopolitically informed analyses and leveraging both market and non-market
strengths to enhance U.S. competitiveness. This research aims to inform
policymakers, technologists, and researchers about the strategic implications
of the AI-energy nexus and offers insights into advancing U.S. global
leadership in AI amidst evolving technological paradigms.

### 5. [Detecting Cultural Differences in News Video Thumbnails via Computational Aesthetics](http://arxiv.org/pdf/2505.21912v1)

Authors: Marvin Limpijankit, John Kender

We propose a two-step approach for detecting differences in the style of
images across sources of differing cultural affinity, where images are first
clustered into finer visual themes based on content before their aesthetic
features are compared. We test this approach on 2,400 YouTube video thumbnails
taken equally from two U.S. and two Chinese YouTube channels, and relating
equally to COVID-19 and the Ukraine conflict. Our results suggest that while
Chinese thumbnails are less formal and more candid, U.S. channels tend to use
more deliberate, proper photographs as thumbnails. In particular, U.S.
thumbnails are less colorful, more saturated, darker, more finely detailed,
less symmetric, sparser, less varied, and more up close and personal than
Chinese thumbnails. We suggest that most of these differences reflect cultural
preferences, and that our methods and observations can serve as a baseline
against which suspected visual propaganda can be computed and compared.

### 6. [New Tools are Needed for Tracking Adherence to AI Model Behavioral Use Clauses](http://arxiv.org/pdf/2505.22287v1)

Authors: Daniel McDuff, Tim Korjakow, Kevin Klyman, Danish Contractor

Foundation models have had a transformative impact on AI. A combination of
large investments in research and development, growing sources of digital data
for training, and architectures that scale with data and compute has led to
models with powerful capabilities. Releasing assets is fundamental to
scientific advancement and commercial enterprise. However, concerns over
negligent or malicious uses of AI have led to the design of mechanisms to limit
the risks of the technology. The result has been a proliferation of licenses
with behavioral-use clauses and acceptable-use-policies that are increasingly
being adopted by commonly used families of models (Llama, Gemma, Deepseek) and
a myriad of smaller projects. We created and deployed a custom AI licenses
generator to facilitate license creation and have quantitatively and
qualitatively analyzed over 300 customized licenses created with this tool.
Alongside this we analyzed 1.7 million models licenses on the HuggingFace model
hub. Our results show increasing adoption of these licenses, interest in tools
that support their creation and a convergence on common clause configurations.
In this paper we take the position that tools for tracking adoption of, and
adherence to, these licenses is the natural next step and urgently needed in
order to ensure they have the desired impact of ensuring responsible use.

### 7. [NLP for Social Good: A Survey of Challenges, Opportunities, and Responsible Deployment](http://arxiv.org/pdf/2505.22327v1)

Authors: Antonia Karamolegkou, Angana Borah, Eunjung Cho, Sagnik Ray Choudhury, Martina Galletti, Rajarshi Ghosh, Pranav Gupta, Oana Ignat, Priyanka Kargupta, Neema Kotonya, Hemank Lamba, Sun-Joo Lee, Arushi Mangla, Ishani Mondal, Deniz Nazarova, Poli Nemkova, Dina Pisarevskaya, Naquee Rizwan, Nazanin Sabri, Dominik Stammbach, Anna Steinberg, David Tomás, Steven R Wilson, Bowen Yi, Jessica H Zhu, Arkaitz Zubiaga, Anders Søgaard, Alexander Fraser, Zhijing Jin, Rada Mihalcea, Joel R. Tetreault, Daryna Dementieva

Recent advancements in large language models (LLMs) have unlocked
unprecedented possibilities across a range of applications. However, as a
community, we believe that the field of Natural Language Processing (NLP) has a
growing need to approach deployment with greater intentionality and
responsibility. In alignment with the broader vision of AI for Social Good
(Toma\v{s}ev et al., 2020), this paper examines the role of NLP in addressing
pressing societal challenges. Through a cross-disciplinary analysis of social
goals and emerging risks, we highlight promising research directions and
outline challenges that must be addressed to ensure responsible and equitable
progress in NLP4SG research.

### 8. [Parental Collaboration and Closeness: Envisioning with New Couple Parents](http://arxiv.org/pdf/2505.22428v1)

Authors: Ya-Fang Lin, Xiaotian Li, Wan-Hsuan Huang, Charan Pushpanathan Prabavathi, Jie Cai, John M. Carroll

Couples often experience a decrease in closeness as they cope with the
demands of parenthood. Existing technologies have supported parenting and
parental collaboration. However, these technologies do not adequately support
closeness in co-parenting. We use scenarios and design probes to brainstorm
with 10 new parent couples to explore and envision possibilities for
technologies to support closeness. We reported parents' current technology use
for co-parenting and how participants considered and envisioned co-parenting
technology for closeness, including information and task sharing, emotion
awareness and disclosure, and fostering fun interaction. We discuss the
potential technology has for fostering closeness in co-parenting by (1)
fostering interdependence by supporting parental competence and (2) integrating
positive emotions and experiences, such as validation and fun, in parenting.
Based on our findings, we expand the design space of technology for closeness
to include interdependence. We also expand the design space for co-parenting
technology by integrating more positive emotions.

### 9. [A Human-Centric Approach to Explainable AI for Personalized Education](http://arxiv.org/pdf/2505.22541v1)

Authors: Vinitra Swamy

Deep neural networks form the backbone of artificial intelligence research,
with potential to transform the human experience in areas ranging from
autonomous driving to personal assistants, healthcare to education. However,
their integration into the daily routines of real-world classrooms remains
limited. It is not yet common for a teacher to assign students individualized
homework targeting their specific weaknesses, provide students with instant
feedback, or simulate student responses to a new exam question. While these
models excel in predictive performance, this lack of adoption can be attributed
to a significant weakness: the lack of explainability of model decisions,
leading to a lack of trust from students, parents, and teachers. This thesis
aims to bring human needs to the forefront of eXplainable AI (XAI) research,
grounded in the concrete use case of personalized learning and teaching. We
frame the contributions along two verticals: technical advances in XAI and
their aligned human studies. We investigate explainability in AI for education,
revealing systematic disagreements between post-hoc explainers and identifying
a need for inherently interpretable model architectures. We propose four novel
technical contributions in interpretability with a multimodal modular
architecture (MultiModN), an interpretable mixture-of-experts model
(InterpretCC), adversarial training for explainer stability, and a
theory-driven LLM-XAI framework to present explanations to students
(iLLuMinaTE), which we evaluate in diverse settings with professors, teachers,
learning scientists, and university students. By combining empirical
evaluations of existing explainers with novel architectural designs and human
studies, our work lays a foundation for human-centric AI systems that balance
state-of-the-art performance with built-in transparency and trust.

### 10. [Characterizing Bias: Benchmarking Large Language Models in Simplified versus Traditional Chinese](http://arxiv.org/pdf/2505.22645v1)

Authors: Hanjia Lyu, Jiebo Luo, Jian Kang, Allison Koenecke

While the capabilities of Large Language Models (LLMs) have been studied in
both Simplified and Traditional Chinese, it is yet unclear whether LLMs exhibit
differential performance when prompted in these two variants of written
Chinese. This understanding is critical, as disparities in the quality of LLM
responses can perpetuate representational harms by ignoring the different
cultural contexts underlying Simplified versus Traditional Chinese, and can
exacerbate downstream harms in LLM-facilitated decision-making in domains such
as education or hiring. To investigate potential LLM performance disparities,
we design two benchmark tasks that reflect real-world scenarios: regional term
choice (prompting the LLM to name a described item which is referred to
differently in Mainland China and Taiwan), and regional name choice (prompting
the LLM to choose who to hire from a list of names in both Simplified and
Traditional Chinese). For both tasks, we audit the performance of 11 leading
commercial LLM services and open-sourced models -- spanning those primarily
trained on English, Simplified Chinese, or Traditional Chinese. Our analyses
indicate that biases in LLM responses are dependent on both the task and
prompting language: while most LLMs disproportionately favored Simplified
Chinese responses in the regional term choice task, they surprisingly favored
Traditional Chinese names in the regional name choice task. We find that these
disparities may arise from differences in training data representation, written
character preferences, and tokenization of Simplified and Traditional Chinese.
These findings highlight the need for further analysis of LLM biases; as such,
we provide an open-sourced benchmark dataset to foster reproducible evaluations
of future LLM behavior across Chinese language variants
(https://github.com/brucelyu17/SC-TC-Bench).

### Databases

### 1. [GXJoin: Generalized Cell Transformations for Explainable Joinability](http://arxiv.org/pdf/2505.21860v1)

Authors: Soroush Omidvartehrani, Arash Dargahi Nobari, Davood Rafiei

Describing real-world entities can vary across different sources, posing a
challenge when integrating or exchanging data. We study the problem of
joinability under syntactic transformations, where two columns are not
equi-joinable but can become equi-joinable after some transformations.
Discovering those transformations is a challenge because of the large space of
possible candidates, which grows with the input length and the number of rows.
Our focus is on the generality of transformations, aiming to make the relevant
models applicable across various instances and domains. We explore a few
generalization techniques, emphasizing those that yield transformations
covering a larger number of rows and are often easier to explain. Through
extensive evaluation on two real-world datasets and employing diverse metrics
for measuring the coverage and simplicity of the transformations, our approach
demonstrates superior performance over state-of-the-art approaches by
generating fewer, simpler and hence more explainable transformations as well as
improving the join performance.

### 2. [CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing](http://arxiv.org/pdf/2505.21866v1)

Authors: Guozhen Zhu, Yuqian Hu, Weihang Gao, Wei-Hsiang Wang, Beibei Wang, K. J. Ray Liu

WiFi sensing has emerged as a compelling contactless modality for human
activity monitoring by capturing fine-grained variations in Channel State
Information (CSI). Its ability to operate continuously and non-intrusively
while preserving user privacy makes it particularly suitable for health
monitoring. However, existing WiFi sensing systems struggle to generalize in
real-world settings, largely due to datasets collected in controlled
environments with homogeneous hardware and fragmented, session-based recordings
that fail to reflect continuous daily activity.
  We present CSI-Bench, a large-scale, in-the-wild benchmark dataset collected
using commercial WiFi edge devices across 26 diverse indoor environments with
35 real users. Spanning over 461 hours of effective data, CSI-Bench captures
realistic signal variability under natural conditions. It includes
task-specific datasets for fall detection, breathing monitoring, localization,
and motion source recognition, as well as a co-labeled multitask dataset with
joint annotations for user identity, activity, and proximity. To support the
development of robust and generalizable models, CSI-Bench provides standardized
evaluation splits and baseline results for both single-task and multi-task
learning. CSI-Bench offers a foundation for scalable, privacy-preserving WiFi
sensing systems in health and broader human-centric applications.

### 3. [ChatPD: An LLM-driven Paper-Dataset Networking System](http://arxiv.org/pdf/2505.22349v1)

Authors: Anjie Xu, Ruiqing Ding, Leye Wang

Scientific research heavily depends on suitable datasets for method
validation, but existing academic platforms with dataset management like
PapersWithCode suffer from inefficiencies in their manual workflow. To overcome
this bottleneck, we present a system, called ChatPD, that utilizes Large
Language Models (LLMs) to automate dataset information extraction from academic
papers and construct a structured paper-dataset network. Our system consists of
three key modules: \textit{paper collection}, \textit{dataset information
extraction}, and \textit{dataset entity resolution} to construct paper-dataset
networks. Specifically, we propose a \textit{Graph Completion and Inference}
strategy to map dataset descriptions to their corresponding entities. Through
extensive experiments, we demonstrate that ChatPD not only outperforms the
existing platform PapersWithCode in dataset usage extraction but also achieves
about 90\% precision and recall in entity resolution tasks. Moreover, we have
deployed ChatPD to continuously extract which datasets are used in papers, and
provide a dataset discovery service, such as task-specific dataset queries and
similar dataset recommendations. We open source ChatPD and the current
paper-dataset network on this [GitHub
repository]{https://github.com/ChatPD-web/ChatPD}.

### 4. [ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM](http://arxiv.org/pdf/2505.22552v1)

Authors: Hoang Pham, Thanh-Do Nguyen, Khac-Hoai Nam Bui

Integrating knowledge graphs (KGs) to enhance the reasoning capabilities of
large language models (LLMs) is an emerging research challenge in claim
verification. While KGs provide structured, semantically rich representations
well-suited for reasoning, most existing verification methods rely on
unstructured text corpora, limiting their ability to effectively leverage KGs.
Additionally, despite possessing strong reasoning abilities, modern LLMs
struggle with multi-step modular pipelines and reasoning over KGs without
adaptation. To address these challenges, we propose ClaimPKG, an end-to-end
framework that seamlessly integrates LLM reasoning with structured knowledge
from KGs. Specifically, the main idea of ClaimPKG is to employ a lightweight,
specialized LLM to represent the input claim as pseudo-subgraphs, guiding a
dedicated subgraph retrieval module to identify relevant KG subgraphs. These
retrieved subgraphs are then processed by a general-purpose LLM to produce the
final verdict and justification. Extensive experiments on the FactKG dataset
demonstrate that ClaimPKG achieves state-of-the-art performance, outperforming
strong baselines in this research field by 9%-12% accuracy points across
multiple categories. Furthermore, ClaimPKG exhibits zero-shot generalizability
to unstructured datasets such as HoVer and FEVEROUS, effectively combining
structured knowledge from KGs with LLM reasoning across various LLM backbones.

### 5. [Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems](http://arxiv.org/pdf/2505.22571v1)

Authors: Hoang Pham, Khac-Hoai Nam Bui

This paper presents a novel approach for unified retrieval-augmented
generation (RAG) systems using the recent emerging large language model (LLM)
agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental
controllers, has become a promising approach to enable the interpretability of
RAG tasks, especially for complex reasoning question-answering systems (e.g.,
multi-hop queries). Nonetheless, previous works mainly focus on solving RAG
systems with either single-hop or multi-hop approaches separately, which limits
the application of those approaches to real-world applications. In this study,
we propose a trainable agent framework called Agent-UniRAG for unified
retrieval-augmented LLM systems, which enhances the effectiveness and
interpretability of RAG systems. The main idea is to design an LLM agent
framework to solve RAG tasks step-by-step based on the complexity of the
inputs, simultaneously including single-hop and multi-hop queries in an
end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset
to enable the proposed agent framework for small open-source LLMs (e.g.,
Llama-3-8B). The results show comparable performances with closed-source and
larger open-source LLMs across various RAG benchmarks. Our source code and
dataset are publicly available for further exploitation.

### Distributed, Parallel, and Cluster Computing

### 1. [Joint$λ$: Orchestrating Serverless Workflows on Jointcloud FaaS Systems](http://arxiv.org/pdf/2505.21899v1)

Authors: Jianfei Liu, Rui Li, Zhilin Yang, Peichang Shi, Guodong Yi, Huaimin Wang

Existing serverless workflow orchestration systems are predominantly designed
for a single-cloud FaaS system, leading to vendor lock-in. This restricts
performance optimization, cost reduction, and availability of applications.
However, orchestrating serverless workflows on Jointcloud FaaS systems faces
two main challenges: 1) Additional overhead caused by centralized cross-cloud
orchestration; and 2) A lack of reliable failover and fault-tolerant mechanisms
for cross-cloud serverless workflows. To address these challenges, we propose
Joint$\lambda$, a distributed runtime system designed to orchestrate serverless
workflows on multiple FaaS systems without relying on a centralized
orchestrator. Joint$\lambda$ introduces a compatibility layer, Backend-Shim,
leveraging inter-cloud heterogeneity to optimize makespan and reduce costs with
on-demand billing. By using function-side orchestration instead of centralized
nodes, it enables independent function invocations and data transfers, reducing
cross-cloud communication overhead. For high availability, it ensures
exactly-once execution via datastores and failover mechanisms for serverless
workflows on Jointcloud FaaS systems. We validate Joint$\lambda$ on two
heterogeneous FaaS systems, AWS and ALiYun, with four workflows. Compared to
the most advanced commercial orchestration services for single-cloud serverless
workflows, Joint$\lambda$ reduces up to 3.3$\times$ latency, saving up to 65\%
cost. Joint$\lambda$ is also faster than the state-of-the-art orchestrators for
cross-cloud serverless workflows up to 4.0$\times$, reducing up to 4.5$\times$
cost and providing strong execution guarantees.

### 2. [Hybrid Batch Normalisation: Resolving the Dilemma of Batch Normalisation in Federated Learning](http://arxiv.org/pdf/2505.21877v1)

Authors: Hongyao Chen, Tianyang Xu, Xiaojun Wu, Josef Kittler

Batch Normalisation (BN) is widely used in conventional deep neural network
training to harmonise the input-output distributions for each batch of data.
However, federated learning, a distributed learning paradigm, faces the
challenge of dealing with non-independent and identically distributed data
among the client nodes. Due to the lack of a coherent methodology for updating
BN statistical parameters, standard BN degrades the federated learning
performance. To this end, it is urgent to explore an alternative normalisation
solution for federated learning. In this work, we resolve the dilemma of the BN
layer in federated learning by developing a customised normalisation approach,
Hybrid Batch Normalisation (HBN). HBN separates the update of statistical
parameters (i.e. , means and variances used for evaluation) from that of
learnable parameters (i.e. , parameters that require gradient updates),
obtaining unbiased estimates of global statistical parameters in distributed
scenarios. In contrast with the existing solutions, we emphasise the supportive
power of global statistics for federated learning. The HBN layer introduces a
learnable hybrid distribution factor, allowing each computing node to
adaptively mix the statistical parameters of the current batch with the global
statistics. Our HBN can serve as a powerful plugin to advance federated
learning performance. It reflects promising merits across a wide range of
federated learning settings, especially for small batch sizes and heterogeneous
data.

### 3. [Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference](http://arxiv.org/pdf/2505.21919v1)

Authors: Yue Zhu, Hao Yu, Chen Wang, Zhuoran Liu, Eun Kyung Lee

The increasing adoption of large language models (LLMs) with extended context
windows necessitates efficient Key-Value Cache (KVC) management to optimize
inference performance. Inference workloads like Retrieval-Augmented Generation
(RAG) and agents exhibit high cache reusability, making efficient caching
critical to reducing redundancy and improving speed. We analyze real-world KVC
access patterns using publicly available traces and evaluate commercial
key-value stores like Redis and state-of-the-art RDMA-based systems (CHIME [1]
and Sherman [2]) for KVC metadata management. Our work demonstrates the lack of
tailored storage solution for KVC prefilling, underscores the need for an
efficient distributed caching system with optimized metadata management for LLM
workloads, and provides insights into designing improved KVC management systems
for scalable, low-latency inference.

### 4. [Inclusive, Differentially Private Federated Learning for Clinical Data](http://arxiv.org/pdf/2505.22108v1)

Authors: Santhosh Parampottupadam, Melih Coşğun, Sarthak Pati, Maximilian Zenk, Saikat Roy, Dimitrios Bounias, Benjamin Hamm, Sinem Sav, Ralf Floca, Klaus Maier-Hein

Federated Learning (FL) offers a promising approach for training clinical AI
models without centralizing sensitive patient data. However, its real-world
adoption is hindered by challenges related to privacy, resource constraints,
and compliance. Existing Differential Privacy (DP) approaches often apply
uniform noise, which disproportionately degrades model performance, even among
well-compliant institutions. In this work, we propose a novel compliance-aware
FL framework that enhances DP by adaptively adjusting noise based on
quantifiable client compliance scores. Additionally, we introduce a compliance
scoring tool based on key healthcare and security standards to promote secure,
inclusive, and equitable participation across diverse clinical settings.
Extensive experiments on public datasets demonstrate that integrating
under-resourced, less compliant clinics with highly regulated institutions
yields accuracy improvements of up to 15% over traditional FL. This work
advances FL by balancing privacy, compliance, and performance, making it a
viable solution for real-world clinical workflows in global healthcare.

### 5. [Smart Contracts for SMEs and Large Companies](http://arxiv.org/pdf/2505.22619v1)

Authors: C. G. Liu, P. Bodorik, D. Jutla

Research on blockchains addresses multiple issues, with one being writing
smart contracts. In our previous research we described methodology and a tool
to generate, in automated fashion, smart contracts from BPMN models. The
generated smart contracts provide support for multi-step transactions that
facilitate repair/upgrade of smart contracts. In this paper we show how the
approach is used to support collaborations via smart contracts for companies
ranging from SMEs with little IT capabilities to companies with IT using
blockchain smart contracts. Furthermore, we also show how the approach is used
for certain applications to generate smart contracts by a BPMN modeler who does
not need any knowledge of blockchain technology or smart contract development -
thus we are hoping to facilitate democratization of smart contracts and
blockchain technology.

### Discrete Mathematics

### 1. [Finding $d$-Cuts in Probe $H$-Free Graphs](http://arxiv.org/pdf/2505.22351v1)

Authors: Konrad K. Dabrowski, Tala Eagling-Vose, Matthew Johnson, Giacomo Paesani, Daniël Paulusma

For an integer $d\geq 1$, the $d$-Cut problem is that of deciding whether a
graph has an edge cut in which each vertex is adjacent to at most $d$ vertices
on the opposite side of the cut. The $1$-Cut problem is the well-known Matching
Cut problem. The $d$-Cut problem has been extensively studied for $H$-free
graphs. We extend these results to the probe graph model, where we do not know
all the edges of the input graph. For a graph $H$, a partitioned probe $H$-free
graph $(G,P,N)$ consists of a graph $G=(V,E)$, together with a set $P\subseteq
V$ of probes and an independent set $N=V\setminus P$ of non-probes such that we
can change $G$ into an $H$-free graph by adding zero or more edges between
vertices in $N$. For every graph $H$ and every integer $d\geq 1$, we completely
determine the complexity of $d$-Cut on partitioned probe $H$-free graphs.

### 2. [On Big Ramsey degrees of universal $ω$-edge-labeled hypergraphs](http://arxiv.org/pdf/2505.22561v1)

Authors: Jan Hubička, Matěj Konečný, Stevo Todorcevic, Andy Zucker

We show that the big Ramsey degrees of every countable universal $u$-uniform
$\omega$-edge-labeled hypergraph are infinite for every $u\geq 2$. Together
with a recent result of Braunfeld, Chodounsk\'y, de Rancourt, Hubi\v{c}ka,
Kawach, and Kone\v{c}n\'y this finishes full characterisation of unrestricted
relational structures with finite big Ramsey degrees.

### 3. [Oscillating subalgebras of the atomless countable Boolean algebra](http://arxiv.org/pdf/2505.22603v1)

Authors: Dana Bartošová, David Chodounský, Barbara Csima, Jan Hubička, Matěj Konečný, Joey Lakerdas-Gayle, Spencer Unger, Andy Zucker

We show that the big Ramsey degree of the Boolean algebra with 3 atoms within
the countable atomless Boolean algebra is infinite.

### 4. [Counting big Ramsey degrees of the homogeneous and universal $K_4$-free graph](http://arxiv.org/pdf/2505.22620v1)

Authors: Jan Hubička, Matěj Konečný, Štěpán Vodseďálek, Andy Zucker

Big Ramsey degrees of Fra\"iss\'e limits of finitely constrained free
amalgamation classes in finite binary languages have been recently fully
characterised by Balko, Chodounsk\'y, Dobrinen, Hubi\v{c}ka, Kone\v{c}n\'y,
Vena, and Zucker. A special case of this characterisation is the universal
homogeneous $K_4$-free graph. We give a self-contained and relatively compact
presentation of this case and compute the actual big Ramsey degrees of small
graphs.

### Data Structures and Algorithms

### 1. [Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](http://arxiv.org/pdf/2505.21939v1)

Authors: Dahoon Lee, Chenglin Fan, Euiwoong Lee

Correlation Clustering (CC) is a foundational problem in unsupervised
learning that models binary similarity relations using labeled graphs. While
classical CC has been widely studied, many real-world applications involve more
nuanced relationships, either multi-class categorical interactions or varying
confidence levels in edge labels. To address these, two natural generalizations
have been proposed: Chromatic Correlation Clustering (CCC), which assigns
semantic colors to edge labels, and pseudometric-weighted CC, which allows edge
weights satisfying the triangle inequality. In this paper, we develop improved
approximation algorithms for both settings. Our approach leverages LP-based
pivoting techniques combined with problem-specific rounding functions. For the
pseudometric-weighted correlation clustering problem, we present a tight
$10/3$-approximation algorithm, matching the best possible bound achievable
within the framework of standard LP relaxation combined with specialized
rounding. For the Chromatic Correlation Clustering (CCC) problem, we improve
the approximation ratio from the previous best of $2.5$ to $2.15$, and we
establish a lower bound of $2.11$ within the same analytical framework,
highlighting the near-optimality of our result.

### 2. [(Near)-Optimal Algorithms for Sparse Separable Convex Integer Programs](http://arxiv.org/pdf/2505.22212v1)

Authors: Christoph Hunkenschröder, Martin Koutecký, Asaf Levin, Tung Anh Vu

We study the general integer programming (IP) problem of optimizing a
separable convex function over the integer points of a polytope: $\min
\{f(\mathbf{x}) \mid A\mathbf{x} = \mathbf{b}, \, \mathbf{l} \leq \mathbf{x}
\leq \mathbf{u}, \, \mathbf{x} \in \mathbb{Z}^n\}$. The number of variables $n$
is a variable part of the input, and we consider the regime where the
constraint matrix $A$ has small coefficients $\|A\|_\infty$ and small primal or
dual treedepth $\mathrm{td}_P(A)$ or $\mathrm{td}_D(A)$, respectively.
Equivalently, we consider block-structured matrices, in particular $n$-fold,
tree-fold, $2$-stage and multi-stage matrices.
  We ask about the possibility of near-linear time algorithms in the general
case of (non-linear) separable convex functions. The techniques of previous
works for the linear case are inherently limited to it; in fact, no
strongly-polynomial algorithm may exist due to a simple unconditional
information-theoretic lower bound of $n \log \|\mathbf{u}-\mathbf{l}\|_\infty$,
where $\mathbf{l}, \mathbf{u}$ are the vectors of lower and upper bounds. Our
first result is that with parameters $\mathrm{td}_P(A)$ and $\|A\|_\infty$,
this lower bound can be matched (up to dependency on the parameters). Second,
with parameters $\mathrm{td}_D(A)$ and $\|A\|_\infty$, the situation is more
involved, and we design an algorithm with time complexity $g(\mathrm{td}_D(A),
\|A\|_\infty) n \log n \log \|\mathbf{u}-\mathbf{l}\|_\infty$ where $g$ is some
computable function. We conjecture that a stronger lower bound is possible in
this regime, and our algorithm is in fact optimal.
  Our algorithms combine ideas from scaling, proximity, and sensitivity of
integer programs, together with a new dynamic data structure.

### 3. [Counting Small Induced Subgraphs: Scorpions Are Easy but Not Trivial](http://arxiv.org/pdf/2505.22300v1)

Authors: Radu Curticapean, Simon Döring, Daniel Neuen

We consider the parameterized problem $\#$IndSub$(\Phi)$ for fixed graph
properties $\Phi$: Given a graph $G$ and an integer $k$, this problem asks to
count the number of induced $k$-vertex subgraphs satisfying $\Phi$. D\"orfler
et al. [Algorithmica 2022] and Roth et al. [SICOMP 2024] conjectured that
$\#$IndSub$(\Phi)$ is $\#$W[1]-hard for all non-meager properties $\Phi$, i.e.,
properties that are nontrivial for infinitely many $k$. This conjecture has
been confirmed for several restricted types of properties, including all
hereditary properties [STOC 2022] and all edge-monotone properties [STOC 2024].
  In this work, we refute this conjecture by showing that scorpion graphs,
certain $k$-vertex graphs which were introduced more than 50 years ago in the
context of the evasiveness conjecture, can be counted in time $O(n^4)$ for all
$k$. A simple variant of this construction results in graph properties that
achieve arbitrary intermediate complexity assuming ETH.
  We formulate an updated conjecture on the complexity of $\#$IndSub$(\Phi)$
that correctly captures the complexity status of scorpions and related
constructions.

### 4. [Exact Algorithms and Lower Bounds for Forming Coalitions of Constrained Maximum Size](http://arxiv.org/pdf/2505.22384v1)

Authors: Foivos Fioravantes, Harmender Gahlawat, Nikolaos Melissinos

Imagine we want to split a group of agents into teams in the most
\emph{efficient} way, considering that each agent has their own preferences
about their teammates. This scenario is modeled by the extensively studied
\textsc{Coalition Formation} problem. Here, we study a version of this problem
where each team must additionally be of bounded size.
  We conduct a systematic algorithmic study, providing several intractability
results as well as multiple exact algorithms that scale well as the input grows
(FPT), which could prove useful in practice.
  Our main contribution is an algorithm that deals efficiently with tree-like
structures (bounded \emph{treewidth}) for ``small'' teams. We complement this
result by proving that our algorithm is asymptotically optimal. Particularly,
there can be no algorithm that vastly outperforms the one we present, under
reasonable theoretical assumptions, even when considering star-like structures
(bounded \emph{vertex cover number}).

### 5. [Faster Convolutions: Yates and Strassen Revisited](http://arxiv.org/pdf/2505.22410v1)

Authors: Cornelius Brand, Radu Curticapean, Baitian Li, Kevin Pratt

Given two vectors $u,v \in \mathbb{Q}^D$ over a finite domain $D$ and a
function $f : D\times D\to D$, the convolution problem asks to compute the
vector $w \in \mathbb{Q}^D$ whose entries are defined by $w(d) =
\sum_{\substack{x,y \in D \\ f(x,y)=d}} u(x)v(y).$ In parameterized and
exponential-time algorithms, convolutions on product domains are particularly
prominent: Here, a finite domain $B$ and a function $h : B \times B \to B$ are
fixed, and convolution is done over the product domain $D = B^k$, using the
function $h^k :D \times D\to D$ that applies $h$ coordinate-wise to its input
tuples.
  We present a new perspective on product-domain convolutions through
multilinear algebra. This viewpoint streamlines the presentation and analysis
of existing algorithms, such as those by van Rooij et al. (ESA 2009). Moreover,
using established results from the theory of fast matrix multiplication, we
derive improved $O^\ast(|B|^{2\omega/3 \cdot k}) = O(|D|^{1.582})$ time
algorithms, improving upon previous upper bounds by Esmer et al. (Algorithmica
86(1), 2024) of the form $c^k |B|^{2k}$ for $c < 1$. Using the setup described
in this note, Strassen's asymptotic rank conjecture from algebraic complexity
theory would imply quasi-linear $|D|^{1+o(1)}$ time algorithms. This conjecture
has recently gained attention in the algorithms community. (Bj\"orklund-Kaski
and Pratt, STOC 2024, Bj\"orklund et al., SODA 2025)
  Our paper is intended as a self-contained exposition for an algorithms
audience, and it includes all essential mathematical prerequisites with
explicit coordinate-based notation. In particular, we assume no knowledge in
abstract algebra.

### 6. [Private Lossless Multiple Release](http://arxiv.org/pdf/2505.22449v1)

Authors: Joel Daniel Andersson, Lukas Retschmeier, Boel Nelson, Rasmus Pagh

Koufogiannis et al. (2016) showed a $\textit{gradual release}$ result for
Laplace noise-based differentially private mechanisms: given an
$\varepsilon$-DP release, a new release with privacy parameter $\varepsilon' >
\varepsilon$ can be computed such that the combined privacy loss of both
releases is at most $\varepsilon'$ and the distribution of the latter is the
same as a single release with parameter $\varepsilon'$. They also showed
gradual release techniques for Gaussian noise, later also explored by
Whitehouse et al. (2022).
  In this paper, we consider a more general $\textit{multiple release}$ setting
in which analysts hold private releases with different privacy parameters
corresponding to different access/trust levels. These releases are determined
one by one, with privacy parameters in arbitrary order. A multiple release is
$\textit{lossless}$ if having access to a subset $S$ of the releases has the
same privacy guarantee as the least private release in $S$, and each release
has the same distribution as a single release with the same privacy parameter.
Our main result is that lossless multiple release is possible for a large class
of additive noise mechanisms. For the Gaussian mechanism we give a simple
method for lossless multiple release with a short, self-contained analysis that
does not require knowledge of the mathematics of Brownian motion. We also
present lossless multiple release for the Laplace and Poisson mechanisms.
Finally, we consider how to efficiently do gradual release of sparse
histograms, and present a mechanism with running time independent of the number
of dimensions.

### 7. [Fully Packed and Ready to Go: High-Density, Rearrangement-Free, Grid-Based Storage and Retrieval](http://arxiv.org/pdf/2505.22497v1)

Authors: Tzvika Geft, Kostas Bekris, Jingjin Yu

Grid-based storage systems with uniformly shaped loads (e.g., containers,
pallets, totes) are commonplace in logistics, industrial, and transportation
domains. A key performance metric for such systems is the maximization of space
utilization, which requires some loads to be placed behind or below others,
preventing direct access to them. Consequently, dense storage settings bring up
the challenge of determining how to place loads while minimizing costly
rearrangement efforts necessary during retrieval. This paper considers the
setting involving an inbound phase, during which loads arrive, followed by an
outbound phase, during which loads depart. The setting is prevalent in
distribution centers, automated parking garages, and container ports. In both
phases, minimizing the number of rearrangement actions results in more optimal
(e.g., fast, energy-efficient, etc.) operations. In contrast to previous work
focusing on stack-based systems, this effort examines the case where loads can
be freely moved along the grid, e.g., by a mobile robot, expanding the range of
possible motions. We establish that for a range of scenarios, such as having
limited prior knowledge of the loads' arrival sequences or grids with a narrow
opening, a (best possible) rearrangement-free solution always exists, including
when the loads fill the grid to its capacity. In particular, when the sequences
are fully known, we establish an intriguing characterization showing that
rearrangement can always be avoided if and only if the open side of the grid
(used to access the storage) is at least 3 cells wide. We further discuss
useful practical implications of our solutions.

### 8. [Finding $d$-Cuts in Probe $H$-Free Graphs](http://arxiv.org/pdf/2505.22351v1)

Authors: Konrad K. Dabrowski, Tala Eagling-Vose, Matthew Johnson, Giacomo Paesani, Daniël Paulusma

For an integer $d\geq 1$, the $d$-Cut problem is that of deciding whether a
graph has an edge cut in which each vertex is adjacent to at most $d$ vertices
on the opposite side of the cut. The $1$-Cut problem is the well-known Matching
Cut problem. The $d$-Cut problem has been extensively studied for $H$-free
graphs. We extend these results to the probe graph model, where we do not know
all the edges of the input graph. For a graph $H$, a partitioned probe $H$-free
graph $(G,P,N)$ consists of a graph $G=(V,E)$, together with a set $P\subseteq
V$ of probes and an independent set $N=V\setminus P$ of non-probes such that we
can change $G$ into an $H$-free graph by adding zero or more edges between
vertices in $N$. For every graph $H$ and every integer $d\geq 1$, we completely
determine the complexity of $d$-Cut on partitioned probe $H$-free graphs.

### Emerging Technologies

### 1. [Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference](http://arxiv.org/pdf/2505.21919v1)

Authors: Yue Zhu, Hao Yu, Chen Wang, Zhuoran Liu, Eun Kyung Lee

The increasing adoption of large language models (LLMs) with extended context
windows necessitates efficient Key-Value Cache (KVC) management to optimize
inference performance. Inference workloads like Retrieval-Augmented Generation
(RAG) and agents exhibit high cache reusability, making efficient caching
critical to reducing redundancy and improving speed. We analyze real-world KVC
access patterns using publicly available traces and evaluate commercial
key-value stores like Redis and state-of-the-art RDMA-based systems (CHIME [1]
and Sherman [2]) for KVC metadata management. Our work demonstrates the lack of
tailored storage solution for KVC prefilling, underscores the need for an
efficient distributed caching system with optimized metadata management for LLM
workloads, and provides insights into designing improved KVC management systems
for scalable, low-latency inference.

### 2. [Large-Area Fabrication-aware Computational Diffractive Optics](http://arxiv.org/pdf/2505.22313v1)

Authors: Kaixuan Wei, Hector A. Jimenez-Romero, Hadi Amata, Jipeng Sun, Qiang Fu, Felix Heide, Wolfgang Heidrich

Differentiable optics, as an emerging paradigm that jointly optimizes optics
and (optional) image processing algorithms, has made innovative optical designs
possible across a broad range of applications. Many of these systems utilize
diffractive optical components (DOEs) for holography, PSF engineering, or
wavefront shaping. Existing approaches have, however, mostly remained limited
to laboratory prototypes, owing to a large quality gap between simulation and
manufactured devices. We aim at lifting the fundamental technical barriers to
the practical use of learned diffractive optical systems. To this end, we
propose a fabrication-aware design pipeline for diffractive optics fabricated
by direct-write grayscale lithography followed by nano-imprinting replication,
which is directly suited for inexpensive mass production of large area designs.
We propose a super-resolved neural lithography model that can accurately
predict the 3D geometry generated by the fabrication process. This model can be
seamlessly integrated into existing differentiable optics frameworks, enabling
fabrication-aware, end-to-end optimization of computational optical systems. To
tackle the computational challenges, we also devise tensor-parallel compute
framework centered on distributing large-scale FFT computation across many
GPUs. As such, we demonstrate large scale diffractive optics designs up to
32.16 mm $\times$ 21.44 mm, simulated on grids of up to 128,640 by 85,760
feature points. We find adequate agreement between simulation and fabricated
prototypes for applications such as holography and PSF engineering. We also
achieve high image quality from an imaging system comprised only of a single
DOE, with images processed only by a Wiener filter utilizing the simulation
PSF. We believe our findings lift the fabrication limitations for real-world
applications of diffractive optics and differentiable optical design.

### Graphics

### 1. [Fluid Simulation on Vortex Particle Flow Maps](http://arxiv.org/pdf/2505.21946v1)

Authors: Sinan Wang, Junwei Zhou, Fan Feng, Zhiqi Li, Yuchen Sun, Duowen Chen, Greg Turk, Bo Zhu

We propose the Vortex Particle Flow Map (VPFM) method to simulate
incompressible flow with complex vortical evolution in the presence of dynamic
solid boundaries. The core insight of our approach is that vorticity is an
ideal quantity for evolution on particle flow maps, enabling significantly
longer flow map distances compared to other fluid quantities like velocity or
impulse. To achieve this goal, we developed a hybrid Eulerian-Lagrangian
representation that evolves vorticity and flow map quantities on vortex
particles, while reconstructing velocity on a background grid. The method
integrates three key components: (1) a vorticity-based particle flow map
framework, (2) an accurate Hessian evolution scheme on particles, and (3) a
solid boundary treatment for no-through and no-slip conditions in VPFM. These
components collectively allow a substantially longer flow map length (3-12
times longer) than the state-of-the-art, enhancing vorticity preservation over
extended spatiotemporal domains. We validated the performance of VPFM through
diverse simulations, demonstrating its effectiveness in capturing complex
vortex dynamics and turbulence phenomena.

### 2. [STDR: Spatio-Temporal Decoupling for Real-Time Dynamic Scene Rendering](http://arxiv.org/pdf/2505.22400v1)

Authors: Zehao Li, Hao Jiang, Yujun Cai, Jianing Chen, Baolong Bi, Shuqin Gao, Honglong Zhao, Yiwei Wang, Tianlu Mao, Zhaoqi Wang

Although dynamic scene reconstruction has long been a fundamental challenge
in 3D vision, the recent emergence of 3D Gaussian Splatting (3DGS) offers a
promising direction by enabling high-quality, real-time rendering through
explicit Gaussian primitives. However, existing 3DGS-based methods for dynamic
reconstruction often suffer from \textit{spatio-temporal incoherence} during
initialization, where canonical Gaussians are constructed by aggregating
observations from multiple frames without temporal distinction. This results in
spatio-temporally entangled representations, making it difficult to model
dynamic motion accurately. To overcome this limitation, we propose
\textbf{STDR} (Spatio-Temporal Decoupling for Real-time rendering), a
plug-and-play module that learns spatio-temporal probability distributions for
each Gaussian. STDR introduces a spatio-temporal mask, a separated deformation
field, and a consistency regularization to jointly disentangle spatial and
temporal patterns. Extensive experiments demonstrate that incorporating our
module into existing 3DGS-based dynamic scene reconstruction frameworks leads
to notable improvements in both reconstruction quality and spatio-temporal
consistency across synthetic and real-world benchmarks.

### 3. [Neural Face Skinning for Mesh-agnostic Facial Expression Cloning](http://arxiv.org/pdf/2505.22416v1)

Authors: Sihun Cha, Serin Yoon, Kwanggyoon Seo, Junyong Noh

Accurately retargeting facial expressions to a face mesh while enabling
manipulation is a key challenge in facial animation retargeting. Recent
deep-learning methods address this by encoding facial expressions into a global
latent code, but they often fail to capture fine-grained details in local
regions. While some methods improve local accuracy by transferring deformations
locally, this often complicates overall control of the facial expression. To
address this, we propose a method that combines the strengths of both global
and local deformation models. Our approach enables intuitive control and
detailed expression cloning across diverse face meshes, regardless of their
underlying structures. The core idea is to localize the influence of the global
latent code on the target mesh. Our model learns to predict skinning weights
for each vertex of the target face mesh through indirect supervision from
predefined segmentation labels. These predicted weights localize the global
latent code, enabling precise and region-specific deformations even for meshes
with unseen shapes. We supervise the latent code using Facial Action Coding
System (FACS)-based blendshapes to ensure interpretability and allow
straightforward editing of the generated animation. Through extensive
experiments, we demonstrate improved performance over state-of-the-art methods
in terms of expression fidelity, deformation transfer accuracy, and
adaptability across diverse mesh structures.

### 4. [RenderFormer: Transformer-based Neural Rendering of Triangle Meshes with Global Illumination](http://arxiv.org/pdf/2505.21925v1)

Authors: Chong Zeng, Yue Dong, Pieter Peers, Hongzhi Wu, Xin Tong

We present RenderFormer, a neural rendering pipeline that directly renders an
image from a triangle-based representation of a scene with full global
illumination effects and that does not require per-scene training or
fine-tuning. Instead of taking a physics-centric approach to rendering, we
formulate rendering as a sequence-to-sequence transformation where a sequence
of tokens representing triangles with reflectance properties is converted to a
sequence of output tokens representing small patches of pixels. RenderFormer
follows a two stage pipeline: a view-independent stage that models
triangle-to-triangle light transport, and a view-dependent stage that
transforms a token representing a bundle of rays to the corresponding pixel
values guided by the triangle-sequence from the view-independent stage. Both
stages are based on the transformer architecture and are learned with minimal
prior constraints. We demonstrate and evaluate RenderFormer on scenes with
varying complexity in shape and light transport.

### 5. [Large-Area Fabrication-aware Computational Diffractive Optics](http://arxiv.org/pdf/2505.22313v1)

Authors: Kaixuan Wei, Hector A. Jimenez-Romero, Hadi Amata, Jipeng Sun, Qiang Fu, Felix Heide, Wolfgang Heidrich

Differentiable optics, as an emerging paradigm that jointly optimizes optics
and (optional) image processing algorithms, has made innovative optical designs
possible across a broad range of applications. Many of these systems utilize
diffractive optical components (DOEs) for holography, PSF engineering, or
wavefront shaping. Existing approaches have, however, mostly remained limited
to laboratory prototypes, owing to a large quality gap between simulation and
manufactured devices. We aim at lifting the fundamental technical barriers to
the practical use of learned diffractive optical systems. To this end, we
propose a fabrication-aware design pipeline for diffractive optics fabricated
by direct-write grayscale lithography followed by nano-imprinting replication,
which is directly suited for inexpensive mass production of large area designs.
We propose a super-resolved neural lithography model that can accurately
predict the 3D geometry generated by the fabrication process. This model can be
seamlessly integrated into existing differentiable optics frameworks, enabling
fabrication-aware, end-to-end optimization of computational optical systems. To
tackle the computational challenges, we also devise tensor-parallel compute
framework centered on distributing large-scale FFT computation across many
GPUs. As such, we demonstrate large scale diffractive optics designs up to
32.16 mm $\times$ 21.44 mm, simulated on grids of up to 128,640 by 85,760
feature points. We find adequate agreement between simulation and fabricated
prototypes for applications such as holography and PSF engineering. We also
achieve high image quality from an imaging system comprised only of a single
DOE, with images processed only by a Wiener filter utilizing the simulation
PSF. We believe our findings lift the fabrication limitations for real-world
applications of diffractive optics and differentiable optical design.

### 6. [Cascaded 3D Diffusion Models for Whole-body 3D 18-F FDG PET/CT synthesis from Demographics](http://arxiv.org/pdf/2505.22489v1)

Authors: Siyeop Yoon, Sifan Song, Pengfei Jin, Matthew Tivnan, Yujin Oh, Sekeun Kim, Dufan Wu, Xiang Li, Quanzheng Li

We propose a cascaded 3D diffusion model framework to synthesize
high-fidelity 3D PET/CT volumes directly from demographic variables, addressing
the growing need for realistic digital twins in oncologic imaging, virtual
trials, and AI-driven data augmentation. Unlike deterministic phantoms, which
rely on predefined anatomical and metabolic templates, our method employs a
two-stage generative process. An initial score-based diffusion model
synthesizes low-resolution PET/CT volumes from demographic variables alone,
providing global anatomical structures and approximate metabolic activity. This
is followed by a super-resolution residual diffusion model that refines spatial
resolution. Our framework was trained on 18-F FDG PET/CT scans from the AutoPET
dataset and evaluated using organ-wise volume and standardized uptake value
(SUV) distributions, comparing synthetic and real data between demographic
subgroups. The organ-wise comparison demonstrated strong concordance between
synthetic and real images. In particular, most deviations in metabolic uptake
values remained within 3-5% of the ground truth in subgroup analysis. These
findings highlight the potential of cascaded 3D diffusion models to generate
anatomically and metabolically accurate PET/CT images, offering a robust
alternative to traditional phantoms and enabling scalable, population-informed
synthetic imaging for clinical and research applications.

### Computer Science and Game Theory

### 1. [Strengthening Proportionality in Temporal Voting](http://arxiv.org/pdf/2505.22513v1)

Authors: Bradley Phillips, Edith Elkind, Nicholas Teh, Tomasz Wąs

We study proportional representation in the framework of temporal voting with
approval ballots. Prior work adapted basic proportional representation concepts
-- justified representation (JR), proportional JR (PJR), and extended JR (EJR)
-- from the multiwinner setting to the temporal setting. Our work introduces
and examines ways of going beyond EJR. Specifically, we consider stronger
variants of JR, PJR, and EJR, and introduce temporal adaptations of more
demanding multiwinner axioms, such as EJR+, full JR (FJR), full proportional JR
(FPJR), and the Core. For each of these concepts, we investigate its existence
and study its relationship to existing notions, thereby establishing a rich
hierarchy of proportionality concepts. Notably, we show that two of our
proposed axioms -- EJR+ and FJR -- strengthen EJR while remaining satisfiable
in every temporal election.

### 2. [Online Fair Division for Personalized $2$-Value Instances](http://arxiv.org/pdf/2505.22174v1)

Authors: Georgios Amanatidis, Alexandros Lolos, Evangelos Markakis, Victor Turmel

We study an online fair division setting, where goods arrive one at a time
and there is a fixed set of $n$ agents, each of whom has an additive valuation
function over the goods. Once a good appears, the value each agent has for it
is revealed and it must be allocated immediately and irrevocably to one of the
agents. It is known that without any assumptions about the values being
severely restricted or coming from a distribution, very strong impossibility
results hold in this setting. To bypass the latter, we turn our attention to
instances where the valuation functions are restricted. In particular, we study
personalized $2$-value instances, where there are only two possible values each
agent may have for each good, possibly different across agents, and we show how
to obtain worst case guarantees with respect to well-known fairness notions,
such as maximin share fairness and envy-freeness up to one (or two) good(s). We
suggest a deterministic algorithm that maintains a $1/(2n-1)$-MMS allocation at
every time step and show that this is the best possible any deterministic
algorithm can achieve if one cares about every single time step; nevertheless,
eventually the allocation constructed by our algorithm becomes a $1/4$-MMS
allocation. To achieve this, the algorithm implicitly maintains a fragile
system of priority levels for all agents. Further, we show that, by allowing
some limited access to future information, it is possible to have stronger
results with less involved approaches. By knowing the values of goods for $n-1$
time steps into the future, we design a matching-based algorithm that achieves
an EF$1$ allocation every $n$ time steps, while always maintaining an EF$2$
allocation. Finally, we show that our results allow us to get the first
nontrivial guarantees for additive instances in which the ratio of the maximum
over the minimum value an agent has for a good is bounded.

### 3. [Properties of zero-determinant strategies in multichannel games](http://arxiv.org/pdf/2505.21952v1)

Authors: Masahiko Ueda

Controlling payoffs in repeated games is one of the important topics in
control theory of multi-agent systems. Recently proposed zero-determinant
strategies enable players to unilaterally enforce linear relations between
payoffs. Furthermore, based on the mathematics of zero-determinant strategies,
regional payoff control, in which payoffs are enforced into some feasible
regions, has been discovered in social dilemma situations. More recently,
theory of payoff control was extended to multichannel games, where players
parallelly interact with each other in multiple channels. However, properties
of zero-determinant strategies specific to multichannel games are still not
clear. In this paper, we elucidate properties of zero-determinant strategies in
multichannel games. First, we relate the existence condition of
zero-determinant strategies in multichannel games to that of zero-determinant
strategies in each channel. We then show that the existence of zero-determinant
strategies in multichannel games requires the existence of zero-determinant
strategies in some channels. This result implies that the existence of
zero-determinant strategies in multichannel games is tightly restricted by
structure of games played in each channel.

### Human-Computer Interaction

### 1. [Broadening Our View: Assistive Technology for Cerebral Visual Impairment](http://arxiv.org/pdf/2505.21875v1)

Authors: Bhanuka Gamage, Leona Holloway, Nicola McDowell, Thanh-Toan Do, Nicholas Seow Chiang Price, Arthur James Lowery, Kim Marriott

Over the past decade, considerable research has been directed towards
assistive technologies to support people with vision impairments using machine
learning, computer vision, image enhancement, and/or augmented/virtual reality.
However, this has almost totally overlooked a growing demographic: people with
Cerebral Visual Impairment (CVI). Unlike Ocular Vision Impairments (OVI), CVI
arises from damage to the brain's visual processing centres. This paper
introduces CVI and reveals a wide research gap in addressing the needs of this
demographic. Through a scoping review, we identified 14 papers at the
intersection of these technologies and CVI. Of these, only three papers
described assistive technologies focused on people living with CVI, with the
others focusing on diagnosis, understanding, simulation or rehabilitation. Our
findings highlight the opportunity for the Human-Computer Interaction and
Assistive Technologies research community to explore and address this
underrepresented domain, thereby enhancing the quality of life for people with
CVI.

### 2. [TIEboard: A Digital Educational Tool for Kids Geometric Learning](http://arxiv.org/pdf/2505.21891v1)

Authors: Arooj Zaidi, Giulia Barbareschi, Kai Kunze, Yun Suen Pai, Junichi Yamaoka

Tangible User Interfaces have shown potential in supporting the acquisition
of key concepts in computing and mathematics while fostering engagement in
young learners, but these approaches are less commonly utilised in the context
of geometry. In this paper we introduce TIEboard, an interactive device to
promote early learning of basic geometry concepts. TIEboard draws inspiration
from traditional geoboards and lacing toys to leverage children's familiarity
with these traditional tools. It employs instructional lights to guide children
in creating shapes using colourful threads of optical fiber. The use of
conductive materials allows the system to detect lacing activity and provide
feedback in real-time. TIEboard incorporates six interaction modes of varying
difficulty based on an incremental learning framework. The study evaluated
TIEboard's effectiveness in supporting early geometric learning, facilitating
creativity and promoting collaboration among 16 children aged 5-9.

### 3. [Eye-Tracking and Biometric Feedback in UX Research: Measuring User Engagement and Cognitive Load](http://arxiv.org/pdf/2505.21982v1)

Authors: Aaditya Shankar Majumder

User experience research often uses surveys and interviews, which may miss
subconscious user interactions. This study explores eye-tracking and biometric
feedback as tools to assess user engagement and cognitive load in digital
interfaces. These methods measure gaze behavior and bodily responses, providing
an objective complement to qualitative insights. Using empirical evidence,
practical applications, and advancements from 2023-2025, we present
experimental data, describe our methodology, and place our work within
foundational and recent literature. We address challenges like data
interpretation, ethical issues, and technological integration. These tools are
key for advancing UX design in complex digital environments.

### 4. [ToPSen: Task-Oriented Priming and Sensory Alignment for Comparing Coding Strategies Between Sighted and Blind Programmers](http://arxiv.org/pdf/2505.22414v1)

Authors: Md Ehtesham-Ul-Haque, Syed Masum Billah

This paper examines how the coding strategies of sighted and blind
programmers differ when working with audio feedback alone. The goal is to
identify challenges in mixed-ability collaboration, particularly when sighted
programmers work with blind peers or teach programming to blind students. To
overcome limitations of traditional blindness simulation studies, we proposed
Task-Oriented Priming and Sensory Alignment (ToPSen), a design framework that
reframes sensory constraints as technical requirements rather than as a
disability. Through a study of 12 blind and 12 sighted participants coding
non-visually, we found that expert blind programmers maintain more accurate
mental models and process more information in working memory than sighted
programmers using ToPSen. Our analysis revealed that blind and sighted
programmers process structural information differently, exposing gaps in
current IDE designs. These insights inform our guidelines for improving the
accessibility of programming tools and fostering effective mixed-ability
collaboration.

### 5. [AI Trust Reshaping Administrative Burdens: Understanding Trust-Burden Dynamics in LLM-Assisted Benefits Systems](http://arxiv.org/pdf/2505.22418v1)

Authors: Jeongwon Jo, He Zhang, Jie Cai, Nitesh Goyal

Supplemental Nutrition Assistance Program (SNAP) is an essential benefit
support system provided by the US administration to 41 million federally
determined low-income applicants. Through interviews with such applicants
across a diverse set of experiences with the SNAP system, our findings reveal
that new AI technologies like LLMs can alleviate traditional burdens but also
introduce new burdens. We introduce new types of learning, compliance, and
psychological costs that transform the administrative burden on applicants. We
also identify how trust in AI across three dimensions--competence, integrity,
and benevolence--is perceived to reduce administrative burdens, which may stem
from unintended and untoward overt trust in the system. We discuss calibrating
appropriate levels of user trust in LLM-based administrative systems,
mitigating newly introduced burdens. In particular, our findings suggest that
evidence-based information disclosure is necessary in benefits administration
and propose directions for future research on trust-burden dynamics in
AI-assisted administration systems.

### 6. [UI-Evol: Automatic Knowledge Evolving for Computer Use Agents](http://arxiv.org/pdf/2505.21964v1)

Authors: Ziyun Zhang, Xinyi Liu, Xiaoyi Zhang, Jun Wang, Gang Chen, Yan Lu

External knowledge has played a crucial role in the recent development of
computer use agents. We identify a critical knowledge-execution gap: retrieved
knowledge often fails to translate into effective real-world task execution.
Our analysis shows even 90\% correct knowledge yields only 41\% execution
success rate. To bridge this gap, we propose UI-Evol, a plug-and-play module
for autonomous GUI knowledge evolution. UI-Evol consists of two stages: a
Retrace Stage that extracts faithful objective action sequences from actual
agent-environment interactions, and a Critique Stage that refines existing
knowledge by comparing these sequences against external references. We conduct
comprehensive experiments on the OSWorld benchmark with the state-of-the-art
Agent S2. Our results demonstrate that UI-Evol not only significantly boosts
task performance but also addresses a previously overlooked issue of high
behavioral standard deviation in computer use agents, leading to superior
performance on computer use tasks and substantially improved agent reliability.

### 7. [Retweets, Receipts, and Resistance: Discourse, Sentiment, and Credibility in Public Health Crisis Twitter](http://arxiv.org/pdf/2505.22032v1)

Authors: Tawfiq Ammari, Anna Gutowska, Jacob Ziff, Casey Randazzo, Harihan Subramonyam

As the COVID-19 pandemic evolved, the Centers for Disease Control and
Prevention (CDC) used Twitter to disseminate safety guidance and updates,
reaching millions of users. This study analyzes two years of tweets from, to,
and about the CDC using a mixed methods approach to examine discourse
characteristics, credibility, and user engagement. We found that the CDCs
communication remained largely one directional and did not foster reciprocal
interaction, while discussions around COVID19 were deeply shaped by political
and ideological polarization. Users frequently cited earlier CDC messages to
critique new and sometimes contradictory guidance. Our findings highlight the
role of sentiment, media richness, and source credibility in shaping the spread
of public health messages. We propose design strategies to help the CDC tailor
communications to diverse user groups and manage misinformation more
effectively during high-stakes health crises.

### 8. [Parental Collaboration and Closeness: Envisioning with New Couple Parents](http://arxiv.org/pdf/2505.22428v1)

Authors: Ya-Fang Lin, Xiaotian Li, Wan-Hsuan Huang, Charan Pushpanathan Prabavathi, Jie Cai, John M. Carroll

Couples often experience a decrease in closeness as they cope with the
demands of parenthood. Existing technologies have supported parenting and
parental collaboration. However, these technologies do not adequately support
closeness in co-parenting. We use scenarios and design probes to brainstorm
with 10 new parent couples to explore and envision possibilities for
technologies to support closeness. We reported parents' current technology use
for co-parenting and how participants considered and envisioned co-parenting
technology for closeness, including information and task sharing, emotion
awareness and disclosure, and fostering fun interaction. We discuss the
potential technology has for fostering closeness in co-parenting by (1)
fostering interdependence by supporting parental competence and (2) integrating
positive emotions and experiences, such as validation and fun, in parenting.
Based on our findings, we expand the design space of technology for closeness
to include interdependence. We also expand the design space for co-parenting
technology by integrating more positive emotions.

### 9. [Spot-On: A Mixed Reality Interface for Multi-Robot Cooperation](http://arxiv.org/pdf/2505.22539v1)

Authors: Tim Engelbracht, Petar Lukovic, Tjark Behrens, Kai Lascheit, René Zurbrügg, Marc Pollefeys, Hermann Blum, Zuria Bauer

Recent progress in mixed reality (MR) and robotics is enabling increasingly
sophisticated forms of human-robot collaboration. Building on these
developments, we introduce a novel MR framework that allows multiple quadruped
robots to operate in semantically diverse environments via a MR interface. Our
system supports collaborative tasks involving drawers, swing doors, and
higher-level infrastructure such as light switches. A comprehensive user study
verifies both the design and usability of our app, with participants giving a
"good" or "very good" rating in almost all cases. Overall, our approach
provides an effective and intuitive framework for MR-based multi-robot
collaboration in complex, real-world scenarios.

### 10. [Modeling and Optimizing User Preferences in AI Copilots: A Comprehensive Survey and Taxonomy](http://arxiv.org/pdf/2505.21907v1)

Authors: Saleh Afzoon, Zahra Jahanandish, Phuong Thao Huynh, Amin Beheshti, Usman Naseem

AI copilots, context-aware, AI-powered systems designed to assist users in
tasks such as software development and content creation, are becoming integral
to modern workflows. As these systems grow in capability and adoption,
personalization has emerged as a cornerstone for ensuring usability, trust, and
productivity. Central to this personalization is preference optimization: the
ability of AI copilots to detect, interpret, and align with individual user
preferences. While personalization techniques are well-established in domains
like recommender systems and dialogue agents, their adaptation to interactive,
real-time systems like AI copilots remains fragmented and underexplored. This
survey addresses this gap by synthesizing research on how user preferences are
captured, modeled, and refined within the design of AI copilots. We introduce a
unified definition of AI copilots and propose a phase-based taxonomy of
preference optimization strategies, structured around pre-interaction,
mid-interaction, and post-interaction stages. We analyze techniques for
acquiring preference signals, modeling user intent, and integrating feedback
loops, highlighting both established approaches and recent innovations. By
bridging insights from AI personalization, human-AI collaboration, and large
language model adaptation, this survey provides a structured foundation for
designing adaptive, preference-aware AI copilots. It offers a holistic view of
the available preference resources, how they can be leveraged, and which
technical approaches are most suited to each stage of system design.

### Information Retrieval

### 1. [Shapley Value-driven Data Pruning for Recommender Systems](http://arxiv.org/pdf/2505.22057v1)

Authors: Yansen Zhang, Xiaokun Zhang, Ziqiang Cui, Chen Ma

Recommender systems often suffer from noisy interactions like accidental
clicks or popularity bias. Existing denoising methods typically identify users'
intent in their interactions, and filter out noisy interactions that deviate
from the assumed intent. However, they ignore that interactions deemed noisy
could still aid model training, while some ``clean'' interactions offer little
learning value. To bridge this gap, we propose Shapley Value-driven Valuation
(SVV), a framework that evaluates interactions based on their objective impact
on model training rather than subjective intent assumptions. In SVV, a
real-time Shapley value estimation method is devised to quantify each
interaction's value based on its contribution to reducing training loss.
Afterward, SVV highlights the interactions with high values while downplaying
low ones to achieve effective data pruning for recommender systems. In
addition, we develop a simulated noise protocol to examine the performance of
various denoising approaches systematically. Experiments on four real-world
datasets show that SVV outperforms existing denoising methods in both accuracy
and robustness. Further analysis also demonstrates that our SVV can preserve
training-critical interactions and offer interpretable noise assessment. This
work shifts denoising from heuristic filtering to principled, model-driven
interaction valuation.

### 2. [ConsRec: Denoising Sequential Recommendation through User-Consistent Preference Modeling](http://arxiv.org/pdf/2505.22130v1)

Authors: Haidong Xin, Qiushi Xiong, Zhenghao Liu, Sen Mei, Yukun Yan, Shi Yu, Shuo Wang, Yu Gu, Ge Yu, Chenyan Xiong

User-item interaction histories are pivotal for sequential recommendation
systems but often include noise, such as unintended clicks or actions that fail
to reflect genuine user preferences. To address this issue, we propose the
User-Consistent Preference-based Sequential Recommendation System (ConsRec),
designed to capture stable user preferences and filter noisy items from
interaction histories. Specifically, ConsRec constructs a user-interacted item
graph, learns item similarities from their text representations, and then
extracts the maximum connected subgraph from the user-interacted item graph for
denoising items. Experimental results on the Yelp and Amazon Product datasets
illustrate that ConsRec achieves a 13% improvement over baseline recommendation
models, showing its effectiveness in denoising user-interacted items. Further
analysis reveals that the denoised interaction histories form semantically
tighter clusters of user-preferred items, leading to higher relevance scores
for ground-truth targets and more accurate recommendations. All codes are
available at https://github.com/NEUIR/ConsRec.

### 3. [Personalized Tree based progressive regression model for watch-time prediction in short video recommendation](http://arxiv.org/pdf/2505.22153v1)

Authors: Xiaokai Chen, Xiao Lin, Changcheng Li, Peng Jiang

In online video platforms, accurate watch time prediction has become a
fundamental and challenging problem in video recommendation. Previous research
has revealed that the accuracy of watch time prediction highly depends on both
the transformation of watch-time labels and the decomposition of the estimation
process. TPM (Tree based Progressive Regression Model) achieves
State-of-the-Art performance with a carefully designed and effective
decomposition paradigm. TPM discretizes the watch time into several ordinal
intervals and organizes them into a binary decision tree, where each node
corresponds to a specific interval. At each non-leaf node, a binary classifier
is used to determine the specific interval in which the watch time variable
most likely falls, based on the prediction outcome at its parent node.
  The tree structure serves as the core of TPM, as it defines the decomposition
of watch time estimation and determines how the ordinal intervals are
discretized. However, in TPM, the tree is predefined as a full binary tree,
which may be sub-optimal for the following reasons. First, a full binary tree
implies an equal partitioning of the watch time space, which may struggle to
capture the complexity of real-world watch time distributions. Second, instead
of relying on a globally fixed tree structure, we advocate for a personalized,
data-driven tree that can be learned in an end-to-end manner. Therefore, we
propose PTPM to enable a highly personalized decomposition of watch estimation
with better efficacy and efficiency. Moreover, we reveal that TPM is affected
by selection bias due to conditional modeling and devise a simple approach to
address it. We conduct extensive experiments on both offline datasets and
online environments. PTPM has been fully deployed in core traffic scenarios and
serves more than 400 million users per day.

### 4. [Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries](http://arxiv.org/pdf/2505.22299v1)

Authors: Ganlin Xu, Zhoujia Zhang, Wangyi Mei, Jiaqing Liang, Weijia Lu, Xiaodong Zhang, Zhifei Yang, Xiaofeng Ma, Yanghua Xiao, Deqing Yang

Information retrieval plays a crucial role in resource localization. Current
dense retrievers retrieve the relevant documents within a corpus via embedding
similarities, which compute similarities between dense vectors mainly depending
on word co-occurrence between queries and documents, but overlook the real
query intents.
  Thus, they often retrieve numerous irrelevant documents. Particularly in the
scenarios of complex queries such as \emph{negative-constraint queries}, their
retrieval performance could be catastrophic. To address the issue, we propose a
neuro-symbolic information retrieval method, namely \textbf{NS-IR}, that
leverages first-order logic (FOL) to optimize the embeddings of naive natural
language by considering the \emph{logical consistency} between queries and
documents. Specifically, we introduce two novel techniques, \emph{logic
alignment} and \emph{connective constraint}, to rerank candidate documents,
thereby enhancing retrieval relevance.
  Furthermore, we construct a new dataset \textbf{NegConstraint} including
negative-constraint queries to evaluate our NS-IR's performance on such complex
IR scenarios.
  Our extensive experiments demonstrate that NS-IR not only achieves superior
zero-shot retrieval performance on web search and low-resource retrieval tasks,
but also performs better on negative-constraint queries. Our scource code and
dataset are available at https://github.com/xgl-git/NS-IR-main.

### 5. [Domain specific ontologies from Linked Open Data (LOD)](http://arxiv.org/pdf/2505.22550v1)

Authors: Rosario Uceda-Sosa, Nandana Mihindukulasooriya, Atul Kumar, Sahil Bansal, Seema Nagar

Logical and probabilistic reasoning tasks that require a deeper knowledge of
semantics are increasingly relying on general purpose ontologies such as
Wikidata and DBpedia. However, tasks such as entity disambiguation and linking
may benefit from domain specific knowledge graphs, which make it more efficient
to consume the knowledge and easier to extend with proprietary content. We
discuss our experience bootstrapping one such ontology for IT with a
domain-agnostic pipeline, and extending it using domain-specific glossaries.

### 6. [DocReRank: Single-Page Hard Negative Query Generation for Training Multi-Modal RAG Rerankers](http://arxiv.org/pdf/2505.22584v1)

Authors: Navve Wasserman, Oliver Heinimann, Yuval Golbari, Tal Zimbalist, Eli Schwartz, Michal Irani

Rerankers play a critical role in multimodal Retrieval-Augmented Generation
(RAG) by refining ranking of an initial set of retrieved documents. Rerankers
are typically trained using hard negative mining, whose goal is to select pages
for each query which rank high, but are actually irrelevant. However, this
selection process is typically passive and restricted to what the retriever can
find in the available corpus, leading to several inherent limitations. These
include: limited diversity, negative examples which are often not hard enough,
low controllability, and frequent false negatives which harm training. Our
paper proposes an alternative approach: Single-Page Hard Negative Query
Generation, which goes the other way around. Instead of retrieving negative
pages per query, we generate hard negative queries per page. Using an automated
LLM-VLM pipeline, and given a page and its positive query, we create hard
negatives by rephrasing the query to be as similar as possible in form and
context, yet not answerable from the page. This paradigm enables fine-grained
control over the generated queries, resulting in diverse, hard, and targeted
negatives. It also supports efficient false negative verification. Our
experiments show that rerankers trained with data generated using our approach
outperform existing models and significantly improve retrieval performance.

### 7. [Xinyu AI Search: Enhanced Relevance and Comprehensive Results with Rich Answer Presentations](http://arxiv.org/pdf/2505.21849v1)

Authors: Bo Tang, Junyi Zhu, Chenyang Xi, Yunhang Ge, Jiahao Wu, Yuchen Feng, Yijun Niu, Wenqiang Wei, Yu Yu, Chunyu Li, Zehao Lin, Hao Wu, Ning Liao, Yebin Yang, Jiajia Wang, Zhiyu Li, Feiyu Xiong, Jingrun Chen

Traditional search engines struggle to synthesize fragmented information for
complex queries, while generative AI search engines face challenges in
relevance, comprehensiveness, and presentation. To address these limitations,
we introduce Xinyu AI Search, a novel system that incorporates a
query-decomposition graph to dynamically break down complex queries into
sub-queries, enabling stepwise retrieval and generation. Our retrieval pipeline
enhances diversity through multi-source aggregation and query expansion, while
filtering and re-ranking strategies optimize passage relevance. Additionally,
Xinyu AI Search introduces a novel approach for fine-grained, precise built-in
citation and innovates in result presentation by integrating timeline
visualization and textual-visual choreography. Evaluated on recent real-world
queries, Xinyu AI Search outperforms eight existing technologies in human
assessments, excelling in relevance, comprehensiveness, and insightfulness.
Ablation studies validate the necessity of its key sub-modules. Our work
presents the first comprehensive framework for generative AI search engines,
bridging retrieval, generation, and user-centric presentation.

### 8. [Extracting Research Instruments from Educational Literature Using LLMs](http://arxiv.org/pdf/2505.21855v1)

Authors: Jiseung Yoo, Curran Mahowald, Meiyu Li, Wei Ai

Large Language Models (LLMs) are transforming information extraction from
academic literature, offering new possibilities for knowledge management. This
study presents an LLM-based system designed to extract detailed information
about research instruments used in the education field, including their names,
types, target respondents, measured constructs, and outcomes. Using multi-step
prompting and a domain-specific data schema, it generates structured outputs
optimized for educational research. Our evaluation shows that this system
significantly outperforms other approaches, particularly in identifying
instrument names and detailed information. This demonstrates the potential of
LLM-powered information extraction in educational contexts, offering a
systematic way to organize research instrument information. The ability to
aggregate such information at scale enhances accessibility for researchers and
education leaders, facilitating informed decision-making in educational
research and policy.

### 9. [Yambda-5B -- A Large-Scale Multi-modal Dataset for Ranking And Retrieval](http://arxiv.org/pdf/2505.22238v1)

Authors: A. Ploshkin, V. Tytskiy, A. Pismenny, V. Baikalov, E. Taychinov, A. Permiakov, D. Burlakov, E. Krofto, N. Savushkin

We present Yambda-5B, a large-scale open dataset sourced from the
Yandex.Music streaming platform. Yambda-5B contains 4.79 billion user-item
interactions from 1 million users across 9.39 million tracks. The dataset
includes two primary types of interactions: implicit feedback (listening
events) and explicit feedback (likes, dislikes, unlikes and undislikes). In
addition, we provide audio embeddings for most tracks, generated by a
convolutional neural network trained on audio spectrograms. A key
distinguishing feature of Yambda-5B is the inclusion of the is_organic flag,
which separates organic user actions from recommendation-driven events. This
distinction is critical for developing and evaluating machine learning
algorithms, as Yandex.Music relies on recommender systems to personalize track
selection for users. To support rigorous benchmarking, we introduce an
evaluation protocol based on a Global Temporal Split, allowing recommendation
algorithms to be assessed in conditions that closely mirror real-world use. We
report benchmark results for standard baselines (ItemKNN, iALS) and advanced
models (SANSA, SASRec) using a variety of evaluation metrics. By releasing
Yambda-5B to the community, we aim to provide a readily accessible,
industrial-scale resource to advance research, foster innovation, and promote
reproducible results in recommender systems.

### 10. [UDuo: Universal Dual Optimization Framework for Online Matching](http://arxiv.org/pdf/2505.22243v1)

Authors: Bin Li, Diwei Liu, Zehong Hu, Jia Jia

Online resource allocation under budget constraints critically depends on
proper modeling of user arrival dynamics. Classical approaches employ
stochastic user arrival models to derive near-optimal solutions through
fractional matching formulations of exposed users for downstream allocation
tasks. However, this is no longer a reasonable assumption when the environment
changes dynamically. In this work, We propose the Universal Dual optimization
framework UDuo, a novel paradigm that fundamentally rethinks online allocation
through three key innovations: (i) a temporal user arrival representation
vector that explicitly captures distribution shifts in user arrival patterns
and resource consumption dynamics, (ii) a resource pacing learner with adaptive
allocation policies that generalize to heterogeneous constraint scenarios, and
(iii) an online time-series forecasting approach for future user arrival
distributions that achieves asymptotically optimal solutions with constraint
feasibility guarantees in dynamic environments. Experimental results show that
UDuo achieves higher efficiency and faster convergence than the traditional
stochastic arrival model in real-world pricing while maintaining rigorous
theoretical validity for general online allocation problems.

### Machine Learning

### 1. [HydraNet: Momentum-Driven State Space Duality for Multi-Granularity Tennis Tournaments Analysis](http://arxiv.org/pdf/2505.21882v1)

Authors: Ruijie Li, Xiang Zhao, Qiao Ning, Shikai Guo

In tennis tournaments, momentum, a critical yet elusive phenomenon, reflects
the dynamic shifts in performance of athletes that can decisively influence
match outcomes. Despite its significance, momentum in terms of effective
modeling and multi-granularity analysis across points, games, sets, and matches
in tennis tournaments remains underexplored. In this study, we define a novel
Momentum Score (MS) metric to quantify a player's momentum level in
multi-granularity tennis tournaments, and design HydraNet, a momentum-driven
state-space duality-based framework, to model MS by integrating thirty-two
heterogeneous dimensions of athletes performance in serve, return, psychology
and fatigue. HydraNet integrates a Hydra module, which builds upon a
state-space duality (SSD) framework, capturing explicit momentum with a
sliding-window mechanism and implicit momentum through cross-game state
propagation. It also introduces a novel Versus Learning method to better
enhance the adversarial nature of momentum between the two athletes at a macro
level, along with a Collaborative-Adversarial Attention Mechanism (CAAM) for
capturing and integrating intra-player and inter-player dynamic momentum at a
micro level. Additionally, we construct a million-level tennis cross-tournament
dataset spanning from 2012-2023 Wimbledon and 2013-2023 US Open, and validate
the multi-granularity modeling capability of HydraNet for the MS metric on this
dataset. Extensive experimental evaluations demonstrate that the MS metric
constructed by the HydraNet framework provides actionable insights into how
momentum impacts outcomes at different granularities, establishing a new
foundation for momentum modeling and sports analysis. To the best of our
knowledge. The source code and datasets are available at
https://github.com/ReyJerry/HydraNet.

### 2. [Stochastic Primal-Dual Double Block-Coordinate for Two-way Partial AUC Maximization](http://arxiv.org/pdf/2505.21944v1)

Authors: Linli Zhou, Bokun Wang, My T. Thai, Tianbao Yang

Two-way partial AUC (TPAUC) is a critical performance metric for binary
classification with imbalanced data, as it focuses on specific ranges of the
true positive rate (TPR) and false positive rate (FPR). However, stochastic
algorithms for TPAUC optimization remain under-explored, with existing methods
either limited to approximated TPAUC loss functions or burdened by sub-optimal
complexities. To overcome these limitations, we introduce two innovative
stochastic primal-dual double block-coordinate algorithms for TPAUC
maximization. These algorithms utilize stochastic block-coordinate updates for
both the primal and dual variables, catering to both convex and non-convex
settings. We provide theoretical convergence rate analyses, demonstrating
significant improvements over prior approaches. Our experimental results, based
on multiple benchmark datasets, validate the superior performance of our
algorithms, showcasing faster convergence and better generalization. This work
advances the state of the art in TPAUC optimization and offers practical tools
for real-world machine learning applications.

### 3. [BOFormer: Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL](http://arxiv.org/pdf/2505.21974v1)

Authors: Yu-Heng Hung, Kai-Jie Lin, Yu-Heng Lin, Chien-YiWang, Cheng Sun, Ping-Chun Hsieh

Bayesian optimization (BO) offers an efficient pipeline for optimizing
black-box functions with the help of a Gaussian process prior and an
acquisition function (AF). Recently, in the context of single-objective BO,
learning-based AFs witnessed promising empirical results given its favorable
non-myopic nature. Despite this, the direct extension of these approaches to
multi-objective Bayesian optimization (MOBO) suffer from the
\textit{hypervolume identifiability issue}, which results from the
non-Markovian nature of MOBO problems. To tackle this, inspired by the
non-Markovian RL literature and the success of Transformers in language
modeling, we present a generalized deep Q-learning framework and propose
\textit{BOFormer}, which substantiates this framework for MOBO via sequence
modeling. Through extensive evaluation, we demonstrate that BOFormer constantly
outperforms the benchmark rule-based and learning-based algorithms in various
synthetic MOBO and real-world multi-objective hyperparameter optimization
problems. We have made the source code publicly available to encourage further
research in this direction.

### 4. [Two-Stage Feature Generation with Transformer and Reinforcement Learning](http://arxiv.org/pdf/2505.21978v1)

Authors: Wanfu Gao, Zengyao Man, Zebin He, Yuhao Tang, Jun Gao, Kunpeng Liu

Feature generation is a critical step in machine learning, aiming to enhance
model performance by capturing complex relationships within the data and
generating meaningful new features. Traditional feature generation methods
heavily rely on domain expertise and manual intervention, making the process
labor-intensive and challenging to adapt to different scenarios. Although
automated feature generation techniques address these issues to some extent,
they often face challenges such as feature redundancy, inefficiency in feature
space exploration, and limited adaptability to diverse datasets and tasks. To
address these problems, we propose a Two-Stage Feature Generation (TSFG)
framework, which integrates a Transformer-based encoder-decoder architecture
with Proximal Policy Optimization (PPO). The encoder-decoder model in TSFG
leverages the Transformer's self-attention mechanism to efficiently represent
and transform features, capturing complex dependencies within the data. PPO
further enhances TSFG by dynamically adjusting the feature generation strategy
based on task-specific feedback, optimizing the process for improved
performance and adaptability. TSFG dynamically generates high-quality feature
sets, significantly improving the predictive performance of machine learning
models. Experimental results demonstrate that TSFG outperforms existing
state-of-the-art methods in terms of feature quality and adaptability.

### 5. [ACE: Exploring Activation Cosine Similarity and Variance for Accurate and Calibration-Efficient LLM Pruning](http://arxiv.org/pdf/2505.21987v1)

Authors: Zhendong Mi, Zhenglun Kong, Geng Yuan, Shaoyi Huang

With the rapid expansion of large language models (LLMs), the demand for
memory and computational resources has grown significantly. Recent advances in
LLM pruning aim to reduce the size and computational cost of these models.
However, existing methods often suffer from either suboptimal pruning
performance or low time efficiency during the pruning process. In this work, we
propose an efficient and effective pruning method that simultaneously achieves
high pruning performance and fast pruning speed with improved calibration
efficiency. Our approach introduces two key innovations: (1) An activation
cosine similarity loss-guided pruning metric, which considers the angular
deviation of the output activation between the dense and pruned models. (2) An
activation variance-guided pruning metric, which helps preserve semantic
distinctions in output activations after pruning, enabling effective pruning
with shorter input sequences. These two components can be readily combined to
enhance LLM pruning in both accuracy and efficiency. Experimental results show
that our method achieves up to an 18% reduction in perplexity and up to 63%
decrease in pruning time on prevalent LLMs such as LLaMA, LLaMA-2, and OPT.

### 6. [Learning in Compact Spaces with Approximately Normalized Transformers](http://arxiv.org/pdf/2505.22014v1)

Authors: Jörg K. H. Franke, Urs Spiegelhalter, Marianna Nezhurina, Jenia Jitsev, Frank Hutter, Michael Hefenbrock

In deep learning, regularization and normalization are common solutions for
challenges such as overfitting, numerical instabilities, and the increasing
variance in the residual stream. An alternative approach is to force all
parameters and representations to lie on a hypersphere. This removes the need
for regularization and increases convergence speed, but comes with additional
costs. In this work, we propose a more holistic but approximate normalization
(anTransformer). Our approach constrains the norm of parameters and normalizes
all representations via scalar multiplications motivated by the tight
concentration of the norms of high-dimensional random vectors. When applied to
GPT training, we observe a 40% faster convergence compared to models with QK
normalization, with less than 3% additional runtime. Deriving scaling laws for
anGPT, we found our method enables training with larger batch sizes and fewer
hyperparameters, while matching the favorable scaling characteristics of
classic GPT architectures.

### 7. [Weakly-Supervised Contrastive Learning for Imprecise Class Labels](http://arxiv.org/pdf/2505.22028v1)

Authors: Zi-Hao Zhou, Jun-Jie Wang, Tong Wei, Min-Ling Zhang

Contrastive learning has achieved remarkable success in learning effective
representations, with supervised contrastive learning often outperforming
self-supervised approaches. However, in real-world scenarios, data annotations
are often ambiguous or inaccurate, meaning that class labels may not reliably
indicate whether two examples belong to the same class. This limitation
restricts the applicability of supervised contrastive learning. To address this
challenge, we introduce the concept of ``continuous semantic similarity'' to
define positive and negative pairs. Instead of directly relying on imprecise
class labels, we measure the semantic similarity between example pairs, which
quantifies how closely they belong to the same category by iteratively refining
weak supervisory signals. Based on this concept, we propose a graph-theoretic
framework for weakly-supervised contrastive learning, where semantic similarity
serves as the graph weights. Our framework is highly versatile and can be
applied to many weakly-supervised learning scenarios. We demonstrate its
effectiveness through experiments in two common settings, i.e., noisy label and
partial label learning, where existing methods can be easily integrated to
significantly improve performance. Theoretically, we establish an error bound
for our approach, showing that it can approximate supervised contrastive
learning under mild conditions. The implementation code is available at
https://github.com/Speechless-10308/WSC.

### 8. [Detecting Undesired Process Behavior by Means of Retrieval Augmented Generation](http://arxiv.org/pdf/2505.22041v1)

Authors: Michael Grohs, Adrian Rebmann, Jana-Rebecca Rehse

Conformance checking techniques detect undesired process behavior by
comparing process executions that are recorded in event logs to desired
behavior that is captured in a dedicated process model. If such models are not
available, conformance checking techniques are not applicable, but
organizations might still be interested in detecting undesired behavior in
their processes. To enable this, existing approaches use Large Language Models
(LLMs), assuming that they can learn to distinguish desired from undesired
behavior through fine-tuning. However, fine-tuning is highly resource-intensive
and the fine-tuned LLMs often do not generalize well. To address these
limitations, we propose an approach that requires neither a dedicated process
model nor resource-intensive fine-tuning to detect undesired process behavior.
Instead, we use Retrieval Augmented Generation (RAG) to provide an LLM with
direct access to a knowledge base that contains both desired and undesired
process behavior from other processes, assuming that the LLM can transfer this
knowledge to the process at hand. Our evaluation shows that our approach
outperforms fine-tuned LLMs in detecting undesired behavior, demonstrating that
RAG is a viable alternative to resource-intensive fine-tuning, particularly
when enriched with relevant context from the event log, such as frequent traces
and activities.

### 9. [Differentiable Generalized Sliced Wasserstein Plans](http://arxiv.org/pdf/2505.22049v1)

Authors: Laetitia Chapel, Romain Tavenard, Samuel Vaiter

Optimal Transport (OT) has attracted significant interest in the machine
learning community, not only for its ability to define meaningful distances
between probability distributions -- such as the Wasserstein distance -- but
also for its formulation of OT plans. Its computational complexity remains a
bottleneck, though, and slicing techniques have been developed to scale OT to
large datasets. Recently, a novel slicing scheme, dubbed min-SWGG, lifts a
single one-dimensional plan back to the original multidimensional space,
finally selecting the slice that yields the lowest Wasserstein distance as an
approximation of the full OT plan. Despite its computational and theoretical
advantages, min-SWGG inherits typical limitations of slicing methods: (i) the
number of required slices grows exponentially with the data dimension, and (ii)
it is constrained to linear projections. Here, we reformulate min-SWGG as a
bilevel optimization problem and propose a differentiable approximation scheme
to efficiently identify the optimal slice, even in high-dimensional settings.
We furthermore define its generalized extension for accommodating to data
living on manifolds. Finally, we demonstrate the practical value of our
approach in various applications, including gradient flows on manifolds and
high-dimensional spaces, as well as a novel sliced OT-based conditional flow
matching for image generation -- where fast computation of transport plans is
essential.

### 10. [Can Test-time Computation Mitigate Memorization Bias in Neural Symbolic Regression?](http://arxiv.org/pdf/2505.22081v1)

Authors: Shun Sato, Issei Sato

Symbolic regression aims to discover mathematical equations that fit given
numerical data. It has been applied in various fields of scientific research,
such as producing human-readable expressions that explain physical phenomena.
Recently, Neural symbolic regression (NSR) methods that involve Transformers
pre-trained on large-scale synthetic datasets have gained attention. While
these methods offer advantages such as short inference time, they suffer from
low performance, particularly when the number of input variables is large. In
this study, we hypothesized that this limitation stems from the memorization
bias of Transformers in symbolic regression. We conducted a quantitative
evaluation of this bias in Transformers using a synthetic dataset and found
that Transformers rarely generate expressions not present in the training data.
Additional theoretical analysis reveals that this bias arises from the
Transformer's inability to construct expressions compositionally while
verifying their numerical validity. We finally examined if tailoring test-time
strategies can lead to reduced memorization bias and better performance. We
empirically demonstrate that providing additional information to the model at
test time can significantly mitigate memorization bias. On the other hand, we
also find that reducing memorization bias does not necessarily correlate with
improved performance. These findings contribute to a deeper understanding of
the limitations of NSR approaches and offer a foundation for designing more
robust, generalizable symbolic regression methods. Code is available at
https://github.com/Shun-0922/Mem-Bias-NSR .

### Neural and Evolutionary Computing

### 1. [Symbolically Regressing Fish Biomass Spectral Data: A Linear Genetic Programming Method with Tunable Primitives](http://arxiv.org/pdf/2505.21901v1)

Authors: Zhixing Huang, Bing Xue, Mengjie Zhang, Jeremy S. Ronney, Keith C. Gordon, Daniel P. Killeen

Machine learning techniques play an important role in analyzing spectral
data. The spectral data of fish biomass is useful in fish production, as it
carries many important chemistry properties of fish meat. However, it is
challenging for existing machine learning techniques to comprehensively
discover hidden patterns from fish biomass spectral data since the spectral
data often have a lot of noises while the training data are quite limited. To
better analyze fish biomass spectral data, this paper models it as a symbolic
regression problem and solves it by a linear genetic programming method with
newly proposed tunable primitives. In the symbolic regression problem, linear
genetic programming automatically synthesizes regression models based on the
given primitives and training data. The tunable primitives further improve the
approximation ability of the regression models by tuning their inherent
coefficients. Our empirical results over ten fish biomass targets show that the
proposed method improves the overall performance of fish biomass composition
prediction. The synthesized regression models are compact and have good
interpretability, which allow us to highlight useful features over the
spectrum. Our further investigation also verifies the good generality of the
proposed method across various spectral data treatments and other symbolic
regression problems.

### 2. [Enhanced Ideal Objective Vector Estimation for Evolutionary Multi-Objective Optimization](http://arxiv.org/pdf/2505.21903v1)

Authors: Ruihao Zheng, Zhenkun Wang, Yin Wu, Maoguo Gong

The ideal objective vector, which comprises the optimal values of the $m$
objective functions in an $m$-objective optimization problem, is an important
concept in evolutionary multi-objective optimization. Accurate estimation of
this vector has consistently been a crucial task, as it is frequently used to
guide the search process and normalize the objective space. Prevailing
estimation methods all involve utilizing the best value concerning each
objective function achieved by the individuals in the current or accumulated
population. However, this paper reveals that the population-based estimation
method can only work on simple problems but falls short on problems with
substantial bias. The biases in multi-objective optimization problems can be
divided into three categories, and an analysis is performed to illustrate how
each category hinders the estimation of the ideal objective vector.
Subsequently, a set of test instances is proposed to quantitatively evaluate
the impact of various biases on the ideal objective vector estimation method.
Beyond that, a plug-and-play component called enhanced ideal objective vector
estimation (EIE) is introduced for multi-objective evolutionary algorithms
(MOEAs). EIE features adaptive and fine-grained searches over $m$ subproblems
defined by the extreme weighted sum method. EIE finally outputs $m$ solutions
that can well approximate the ideal objective vector. In the experiments, EIE
is integrated into three representative MOEAs. To demonstrate the wide
applicability of EIE, algorithms are tested not only on the newly proposed test
instances but also on existing ones. The results consistently show that EIE
improves the ideal objective vector estimation and enhances the MOEA's
performance.

### 3. [Bridging Fitness With Search Spaces By Fitness Supremums: A Theoretical Study on LGP](http://arxiv.org/pdf/2505.21991v1)

Authors: Zhixing Huang, Yi Mei, Fangfang Zhang, Mengjie Zhang, Wolfgang Banzhaf

Genetic programming has undergone rapid development in recent years. However,
theoretical studies of genetic programming are far behind. One of the major
obstacles to theoretical studies is the challenge of developing a model to
describe the relationship between fitness values and program genotypes. In this
paper, we take linear genetic programming (LGP) as an example to study the
fitness-to-genotype relationship. We find that the fitness expectation
increases with fitness supremum over instruction editing distance, considering
1) the fitness supremum linearly increases with the instruction editing
distance in LGP, 2) the fitness infimum is fixed, and 3) the fitness
probabilities over different instruction editing distances are similar. We then
extend these findings to explain the bloat effect and the minimum hitting time
of LGP based on instruction editing distance. The bloat effect happens because
it is more likely to produce better offspring by adding instructions than by
removing them, given an instruction editing distance from the optimal program.
The analysis of the minimum hitting time suggests that for a basic LGP genetic
operator (i.e., freemut), maintaining a necessarily small program size and
mutating multiple instructions each time can improve LGP performance. The
reported empirical results verify our hypothesis.

### 4. [Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing](http://arxiv.org/pdf/2505.22035v1)

Authors: Xinyi Chen, Chenxiang Ma, Yujie Wu, Kay Chen Tan, Jibin Wu

Temporal processing is vital for extracting meaningful information from
time-varying signals. Recent advancements in Spiking Neural Networks (SNNs)
have shown immense promise in efficiently processing these signals. However,
progress in this field has been impeded by the lack of effective and
standardized benchmarks, which complicates the consistent measurement of
technological advancements and limits the practical applicability of SNNs. To
bridge this gap, we introduce the Neuromorphic Sequential Arena (NSA), a
comprehensive benchmark that offers an effective, versatile, and
application-oriented evaluation framework for neuromorphic temporal processing.
The NSA includes seven real-world temporal processing tasks from a diverse
range of application scenarios, each capturing rich temporal dynamics across
multiple timescales. Utilizing NSA, we conduct extensive comparisons of
recently introduced spiking neuron models and neural architectures, presenting
comprehensive baselines in terms of task performance, training speed, memory
usage, and energy efficiency. Our findings emphasize an urgent need for
efficient SNN designs that can consistently deliver high performance across
tasks with varying temporal complexities while maintaining low computational
costs. NSA enables systematic tracking of advancements in neuromorphic
algorithm research and paves the way for developing effective and efficient
neuromorphic temporal processing systems.

### 5. [Full Domain Analysis in Fluid Dynamics](http://arxiv.org/pdf/2505.22275v1)

Authors: Alexander Hagg, Adam Gaier, Dominik Wilde, Alexander Asteroth, Holger Foysi, Dirk Reith

Novel techniques in evolutionary optimization, simulation and machine
learning allow for a broad analysis of domains like fluid dynamics, in which
computation is expensive and flow behavior is complex. Under the term of full
domain analysis we understand the ability to efficiently determine the full
space of solutions in a problem domain, and analyze the behavior of those
solutions in an accessible and interactive manner. The goal of full domain
analysis is to deepen our understanding of domains by generating many examples
of flow, their diversification, optimization and analysis. We define a formal
model for full domain analysis, its current state of the art, and requirements
of subcomponents. Finally, an example is given to show what we can learn by
using full domain analysis. Full domain analysis, rooted in optimization and
machine learning, can be a helpful tool in understanding complex systems in
computational physics and beyond.

### Networking and Internet Architecture

### 1. [Leveraging 5G Physical Layer Monitoring for Adaptive Remote Rendering in XR Applications](http://arxiv.org/pdf/2505.22123v1)

Authors: Inhar Yeregui, Daniel Mejías, Mikel Zorrilla, Roberto Viola, Jasone Astorga, Eduardo Jacob

As immersive eXtended Reality (XR) applications demand substantial network
resources, understanding their interaction with 5G networks becomes crucial to
improve them. This paper investigates the role of 5G physical-layer monitoring
to manage and enhance the remote rendering of XR content dynamically. By
observing network metrics directly from the physical layer, we propose a system
to adapt streaming parameters such as bitrate, framerate, and resolution in
real time based on available network capacity. Using theoretical formulas to
estimate maximum data rate, our approach evaluates network resource
availability, enabling the renderer to self-adjust media content
representation. This is critical for providing consistent and smooth XR
experiences to users, especially as network conditions fluctuate. Our findings
suggest that physical-layer monitoring offers valuable insights to increase the
Quality of Service (QoS) and has the potential to elevate user experience in
remote-rendered XR applications.

### 2. [Streaming Remote rendering services: Comparison of QUIC-based and WebRTC Protocols](http://arxiv.org/pdf/2505.22132v1)

Authors: Daniel Mejías, Inhar Yeregui, Ángel Martín, Roberto Viola, Pablo Angueira, Jon Montalbán

The proliferation of Extended Reality (XR) applications, requiring
high-quality, low-latency media streaming, has driven the demand for efficient
remote rendering solutions. This paper focuses on holographic conferencing in
virtual environments and their required uplink and downlink media transmission
capabilities. By examining Media over QUIC (MoQ), Real-time Transport Protocol
(RTP) over QUIC (RoQ), and Web Real-Time Communication (WebRTC), we assess
their latency performance over Wi-Fi and 5G networks. Improvements of
approximately 30% in latency and 60% in connection startup are expected in
QUIC-based protocols compared to WebRTC. The experimental setup transmits a
remote-rendered virtual experience using real-time video streaming protocols to
provide the content to the participant. Our findings contribute to
understanding the maturity of streaming protocols, particularly within
open-source frameworks, and evaluate their suitability in supporting
latency-sensitive XR applications. The study highlights specific protocol
advantages across varied remote rendering scenarios, informing the design of
future XR communication solutions.

### 3. [Chain-of-Thought for Large Language Model-empowered Wireless Communications](http://arxiv.org/pdf/2505.22320v1)

Authors: Xudong Wang, Jian Zhu, Ruichen Zhang, Lei Feng, Dusit Niyato, Jiacheng Wang, Hongyang Du, Shiwen Mao, Zhu Han

Recent advances in large language models (LLMs) have opened new possibilities
for automated reasoning and decision-making in wireless networks. However,
applying LLMs to wireless communications presents challenges such as limited
capability in handling complex logic, generalization, and reasoning.
Chain-of-Thought (CoT) prompting, which guides LLMs to generate explicit
intermediate reasoning steps, has been shown to significantly improve LLM
performance on complex tasks. Inspired by this, this paper explores the
application potential of CoT-enhanced LLMs in wireless communications.
Specifically, we first review the fundamental theory of CoT and summarize
various types of CoT. We then survey key CoT and LLM techniques relevant to
wireless communication and networking. Moreover, we introduce a multi-layer
intent-driven CoT framework that bridges high-level user intent expressed in
natural language with concrete wireless control actions. Our proposed framework
sequentially parses and clusters intent, selects appropriate CoT reasoning
modules via reinforcement learning, then generates interpretable control
policies for system configuration. Using the unmanned aerial vehicle (UAV)
network as a case study, we demonstrate that the proposed framework
significantly outperforms a non-CoT baseline in both communication performance
and quality of generated reasoning.

### 4. [Real-World Modeling of Computation Offloading for Neural Networks with Early Exits and Splits](http://arxiv.org/pdf/2505.22149v1)

Authors: Jan Danek, Zdenek Becvar, Adam Janes

We focus on computation offloading of applications based on convolutional
neural network (CNN) from moving devices, such as mobile robots or autonomous
vehicles, to MultiAccess Edge Computing (MEC) servers via a mobile network. In
order to reduce overall CNN inference time, we design and implement CNN with
early exits and splits, allowing a flexible partial or full offloading of CNN
inference. Through real-world experiments, we analyze an impact of the CNN
inference offloading on the total CNN processing delay, energy consumption, and
classification accuracy in a practical road sign recognition task. The results
confirm that offloading of CNN with early exits and splits can significantly
reduce both total processing delay and energy consumption compared to full
local processing while not impairing classification accuracy. Based on the
results of real-world experiments, we derive practical models for energy
consumption and total processing delay related to offloading of CNN with early
exits and splits.

### 5. [Domainator: Detecting and Identifying DNS-Tunneling Malware Using Metadata Sequences](http://arxiv.org/pdf/2505.22220v1)

Authors: Denis Petrov, Pascal Ruffing, Sebastian Zillien, Steffen Wendzel

In recent years, malware with tunneling (or: covert channel) capabilities is
on the rise. While malware research led to several methods and innovations, the
detection and differentiation of malware solely based on its DNS tunneling
features is still in its infancy. Moreover, no work so far has used the DNS
tunneling traffic to gain knowledge over the current actions taken by the
malware. In this paper, we present Domainator, an approach to detect and
differentiate state-of-the-art malware and DNS tunneling tools without relying
on trivial (but quickly altered) features such as "magic bytes" that are
embedded into subdomains. Instead, we apply an analysis of sequential patterns
to identify specific types of malware. We evaluate our approach with 7
different malware samples and tunneling tools and can identify the particular
malware based on its DNS traffic. We further infer the rough behavior of the
particular malware through its DNS tunneling artifacts. Finally, we compare our
Domainator with related methods.

### 6. [Hybrid Learning for Cold-Start-Aware Microservice Scheduling in Dynamic Edge Environments](http://arxiv.org/pdf/2505.22424v1)

Authors: Jingxi Lu, Wenhao Li, Jianxiong Guo, Xingjian Ding, Zhiqing Tang, Tian Wang, Weijia Jia

With the rapid growth of IoT devices and their diverse workloads,
container-based microservices deployed at edge nodes have become a lightweight
and scalable solution. However, existing microservice scheduling algorithms
often assume static resource availability, which is unrealistic when multiple
containers are assigned to an edge node. Besides, containers suffer from
cold-start inefficiencies during early-stage training in currently popular
reinforcement learning (RL) algorithms. In this paper, we propose a hybrid
learning framework that combines offline imitation learning (IL) with online
Soft Actor-Critic (SAC) optimization to enable a cold-start-aware microservice
scheduling with dynamic allocation for computing resources. We first formulate
a delay-and-energy-aware scheduling problem and construct a rule-based expert
to generate demonstration data for behavior cloning. Then, a GRU-enhanced
policy network is designed in the policy network to extract the correlation
among multiple decisions by separately encoding slow-evolving node states and
fast-changing microservice features, and an action selection mechanism is given
to speed up the convergence. Extensive experiments show that our method
significantly accelerates convergence and achieves superior final performance.
Compared with baselines, our algorithm improves the total objective by $50\%$
and convergence speed by $70\%$, and demonstrates the highest stability and
robustness across various edge configurations.

### 7. [Frequency Resource Management in 6G User-Centric CFmMIMO: A Hybrid Reinforcement Learning and Metaheuristic Approach](http://arxiv.org/pdf/2505.22443v1)

Authors: Selina Cheggour, Valeria Loscri

As sixth-generation (6G) networks continue to evolve, AI-driven solutions are
playing a crucial role in enabling more efficient and adaptive resource
management in wireless communication. One of the key innovations in 6G is
user-centric cell-free massive Multiple-Input Multiple-Output (UC-CFmMIMO), a
paradigm that eliminates traditional cell boundaries and enhances network
performance by dynamically assigning access points (APs) to users. This
approach is particularly well-suited for vehicular networks, offering seamless,
homogeneous, ultra-reliable, and low-latency connectivity. However, in dense
networks, a key challenge lies in efficiently allocating frequency resources
within a limited shared subband spectrum while accounting for frequency
selectivity and the dependency of signal propagation on bandwidth. These
factors make resource allocation increasingly complex, especially in dynamic
environments where maintaining Quality of Service (QoS) is critical. This paper
tackles these challenges by proposing a hybrid multi-user allocation strategy
that integrates reinforcement learning (RL) and metaheuristic optimization to
enhance spectral efficiency (SE), ensure fairness, and mitigate interference
within shared subbands. To assess its effectiveness, we compare this hybrid
approach with two other methods: the bio-inspired Aquila Optimizer (AO) and
Deep Deterministic Policy Gradient (DDPG)-based Actor-Critic Reinforcement
Learning (AC-RL). Our evaluation is grounded in real-world patterns and channel
characteristics, utilizing the 3GPP-3D channel modeling framework (QuaDRiGa) to
capture realistic propagation conditions. The results demonstrate that the
proposed hybrid strategy achieves a superior balance among competing
objectives, underscoring the role of AI-driven resource allocation in advancing
UC-CFmMIMO systems for next-generation wireless networks.

### 8. [The Tri-Hybrid MIMO Architecture](http://arxiv.org/pdf/2505.21971v1)

Authors: Robert W. Heath, Jr., Joseph Carlson, Nitish Vikas Deshpande, Miguel Rodrigo Castellanos, Mohamed Akrout, Chan-Byoung Chae

We present an evolution of multiple-input multiple-output (MIMO) wireless
communications known as the tri-hybrid MIMO architecture. In this framework,
the traditional operations of linear precoding at the transmitter are
distributed across digital beamforming, analog beamforming, and reconfigurable
antennas. Compared with the hybrid MIMO architecture, which combines digital
and analog beamforming, the tri-hybrid approach introduces a third layer of
electromagnetic beamforming through antenna reconfigurability. This added layer
offers a pathway to scale MIMO spatial dimensions, important for 6G systems
operating in centimeter-wave bands, where the tension between larger bandwidths
and infrastructure reuse necessitates ultra-large antenna arrays. We introduce
the key features of the tri-hybrid architecture by (i)~reviewing the benefits
and challenges of communicating with reconfigurable antennas, (ii)~examining
tradeoffs between spectral and energy efficiency enabled by reconfigurability,
and (iii)~exploring configuration challenges across the three layers. Overall,
the tri-hybrid MIMO architecture offers a new approach for integrating emerging
antenna technologies in the MIMO precoding framework.

### 9. [From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications](http://arxiv.org/pdf/2505.22311v1)

Authors: Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Octavia A. Dobre, Merouane Debbah

With the advent of 6G communications, intelligent communication systems face
multiple challenges, including constrained perception and response
capabilities, limited scalability, and low adaptability in dynamic
environments. This tutorial provides a systematic introduction to the
principles, design, and applications of Large Artificial Intelligence Models
(LAMs) and Agentic AI technologies in intelligent communication systems, aiming
to offer researchers a comprehensive overview of cutting-edge technologies and
practical guidance. First, we outline the background of 6G communications,
review the technological evolution from LAMs to Agentic AI, and clarify the
tutorial's motivation and main contributions. Subsequently, we present a
comprehensive review of the key components required for constructing LAMs. We
further categorize LAMs and analyze their applicability, covering Large
Language Models (LLMs), Large Vision Models (LVMs), Large Multimodal Models
(LMMs), Large Reasoning Models (LRMs), and lightweight LAMs. Next, we propose a
LAM-centric design paradigm tailored for communications, encompassing dataset
construction and both internal and external learning approaches. Building upon
this, we develop an LAM-based Agentic AI system for intelligent communications,
clarifying its core components such as planners, knowledge bases, tools, and
memory modules, as well as its interaction mechanisms. We also introduce a
multi-agent framework with data retrieval, collaborative planning, and
reflective evaluation for 6G. Subsequently, we provide a detailed overview of
the applications of LAMs and Agentic AI in communication scenarios. Finally, we
summarize the research challenges and future directions in current studies,
aiming to support the development of efficient, secure, and sustainable
next-generation intelligent communication systems.

### Robotics

### 1. [DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation](http://arxiv.org/pdf/2505.21864v1)

Authors: Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Xu, Linxi Fan, Manuela Veloso, Shuran Song

We present DexUMI - a data collection and policy learning framework that uses
the human hand as the natural interface to transfer dexterous manipulation
skills to various robot hands. DexUMI includes hardware and software
adaptations to minimize the embodiment gap between the human hand and various
robot hands. The hardware adaptation bridges the kinematics gap using a
wearable hand exoskeleton. It allows direct haptic feedback in manipulation
data collection and adapts human motion to feasible robot hand motion. The
software adaptation bridges the visual gap by replacing the human hand in video
data with high-fidelity robot hand inpainting. We demonstrate DexUMI's
capabilities through comprehensive real-world experiments on two different
dexterous robot hand hardware platforms, achieving an average task success rate
of 86%.

### 2. [Mastering Agile Tasks with Limited Trials](http://arxiv.org/pdf/2505.21916v1)

Authors: Yihang Hu, Pingyue Sheng, Shengjie Wang, Yang Gao

Embodied robots nowadays can already handle many real-world manipulation
tasks. However, certain other real-world tasks (e.g., shooting a basketball
into a hoop) are highly agile and require high execution precision, presenting
additional challenges for methods primarily designed for quasi-static
manipulation tasks. This leads to increased efforts in costly data collection,
laborious reward design, or complex motion planning. Such tasks, however, are
far less challenging for humans. Say a novice basketball player typically needs
only $\sim$10 attempts to make their first successful shot, by roughly
imitating a motion prior and then iteratively adjusting their motion based on
the past outcomes. Inspired by this human learning paradigm, we propose the
Adaptive Diffusion Action Plannin (ADAP) algorithm, a simple & scalable
approach which iteratively refines its action plan by few real-world trials
within a learned prior motion pattern, until reaching a specific goal.
Experiments demonstrated that ADAP can learn and accomplish a wide range of
goal-conditioned agile dynamic tasks with human-level precision and efficiency
directly in real-world, such as throwing a basketball into the hoop in fewer
than 10 trials. Project website:https://adap-robotics.github.io/ .

### 3. [Enhanced SIRRT*: A Structure-Aware RRT* for 2D Path Planning with Hybrid Smoothing and Bidirectional Rewiring](http://arxiv.org/pdf/2505.21968v1)

Authors: Hyejeong Ryu

Sampling-based motion planners such as Rapidly-exploring Random Tree* (RRT*)
and its informed variant IRRT* are widely used for optimal path planning in
complex environments. However, these methods often suffer from slow convergence
and high variance due to their reliance on random sampling, particularly when
initial solution discovery is delayed. This paper presents Enhanced SIRRT*
(E-SIRRT*), a structure-aware planner that improves upon the original SIRRT*
framework by introducing two key enhancements: hybrid path smoothing and
bidirectional rewiring. Hybrid path smoothing refines the initial path through
spline fitting and collision-aware correction, while bidirectional rewiring
locally optimizes tree connectivity around the smoothed path to improve cost
propagation. Experimental results demonstrate that E-SIRRT* consistently
outperforms IRRT* and SIRRT* in terms of initial path quality, convergence
rate, and robustness across 100 trials. Unlike IRRT*, which exhibits high
variability due to stochastic initialization, E-SIRRT* achieves repeatable and
efficient performance through deterministic skeleton-based initialization and
structural refinement.

### 4. [Soft Electrothermal Meta-Actuator for Robust Multifunctional Control](http://arxiv.org/pdf/2505.21992v1)

Authors: Hanseong Jo, Pavel Shafirin, Christopher Le, Caden Chan, Artur Davoyan

Soft electrothermal actuators are of great interest in diverse application
domains for their simplicity, compliance, and ease of control. However, the
very nature of thermally induced mechanical actuation sets inherent operation
constraints: unidirectional motion, environmental sensitivity, and slow
response times limited by passive cooling. To overcome these constraints, we
propose a meta-actuator architecture, which uses engineered heat transfer in
thin films to achieve multifunctional operation. We demonstrate electrically
selectable bidirectional motion with large deflection ($ \geq $28% of actuator
length at 0.75 W), suppressed thermal sensitivity to ambient temperature
changes when compared to conventional actuators (>100$ \times $ lower), and
actively forced return to the rest state, which is 10 times faster than that
with passive cooling. We further show that our meta-actuator approach enables
extended ranges of motions for manipulating complex objects. Versatile soft
gripper operations highlight the meta-actuator's potential for soft robotics
and devices.

### 5. [A simulation framework for autonomous lunar construction work](http://arxiv.org/pdf/2505.22091v1)

Authors: Mattias Linde, Daniel Lindmark, Sandra Ålstig, Martin Servin

We present a simulation framework for lunar construction work involving
multiple autonomous machines. The framework supports modelling of construction
scenarios and autonomy solutions, execution of the scenarios in simulation, and
analysis of work time and energy consumption throughout the construction
project. The simulations are based on physics-based models for contacting
multibody dynamics and deformable terrain, including vehicle-soil interaction
forces and soil flow in real time. A behaviour tree manages the operational
logic and error handling, which enables the representation of complex
behaviours through a discrete set of simpler tasks in a modular hierarchical
structure. High-level decision-making is separated from lower-level control
algorithms, with the two connected via ROS2. Excavation movements are
controlled through inverse kinematics and tracking controllers. The framework
is tested and demonstrated on two different lunar construction scenarios.

### 6. [VR-Based Control of Multi-Copter Operation](http://arxiv.org/pdf/2505.22599v1)

Authors: Jack T. Hughes, Mohammad Ghufran, Hossein Rastgoftar

We aim to use virtual reality (VR) to improve the spatial awareness of pilots
by real-time scanning of the environment around the drone using onboard
sensors, live streaming of this environment to a VR headset, and rendering a
virtual representation of the drone and its environment for the pilot. This
way, the pilot can see the immediate environment of the drone up close from a
third-person perspective, as opposed to the first-person perspective that most
drone cameras provide. This provides much more information about the drone
surroundings for the pilot while operating the drone than existing
teleoperation solutions. Previous solutions using VR have relied upon pre-made
designs of the environment, which makes it difficult to adapt to changing
environments. Our solution, in contrast, scans the environment as you fly,
making it much more flexible for use in unknown environments.

### 7. [DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation](http://arxiv.org/pdf/2505.21969v1)

Authors: Tianjun Gu, Linfeng Li, Xuhong Wang, Chenghua Gong, Jingyu Gong, Zhizhong Zhang, Yuan Xie, Lizhuang Ma, Xin Tan

Adaptive navigation in unfamiliar environments is crucial for household
service robots but remains challenging due to the need for both low-level path
planning and high-level scene understanding. While recent vision-language model
(VLM) based zero-shot approaches reduce dependence on prior maps and
scene-specific training data, they face significant limitations: spatiotemporal
discontinuity from discrete observations, unstructured memory representations,
and insufficient task understanding leading to navigation failures. We propose
DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory
Oriented Navigation), a novel cognitive-inspired framework consisting of
Ventral and Dorsal Streams that mimics human navigation capabilities. The
Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology
Map to handle spatiotemporal discontinuities, while the Ventral Stream combines
RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops
Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON
on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art
performance on both success rate (SR) and success weighted by path length (SPL)
metrics, significantly outperforming existing methods. We also introduce a new
evaluation metric (AORI) to assess navigation intelligence better.
Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot
autonomous navigation without requiring prior map building or pre-training.

### 8. [ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning](http://arxiv.org/pdf/2505.22094v1)

Authors: Tonghe Zhang, Yu Chao, Sicang Su, Yu Wang

We propose ReinFlow, a simple yet effective online reinforcement learning
(RL) framework that fine-tunes a family of flow matching policies for
continuous robotic control. Derived from rigorous RL theory, ReinFlow injects
learnable noise into a flow policy's deterministic path, converting the flow
into a discrete-time Markov Process for exact and straightforward likelihood
computation. This conversion facilitates exploration and ensures training
stability, enabling ReinFlow to fine-tune diverse flow model variants,
including Rectified Flow [35] and Shortcut Models [19], particularly at very
few or even one denoising step. We benchmark ReinFlow in representative
locomotion and manipulation tasks, including long-horizon planning with visual
input and sparse reward. The episode reward of Rectified Flow policies obtained
an average net growth of 135.36% after fine-tuning in challenging legged
locomotion tasks while saving denoising steps and 82.63% of wall time compared
to state-of-the-art diffusion RL fine-tuning method DPPO [43]. The success rate
of the Shortcut Model policies in state and visual manipulation tasks achieved
an average net increase of 40.34% after fine-tuning with ReinFlow at four or
even one denoising step, whose performance is comparable to fine-tuned DDIM
policies while saving computation time for an average of 23.20%. Project
Webpage: https://reinflow.github.io/

### 9. [ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](http://arxiv.org/pdf/2505.22159v1)

Authors: Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, Cewu Lu, Wenqiang Zhang

Vision-Language-Action (VLA) models have advanced general-purpose robotic
manipulation by leveraging pretrained visual and linguistic representations.
However, they struggle with contact-rich tasks that require fine-grained
control involving force, especially under visual occlusion or dynamic
uncertainty. To address these limitations, we propose \textbf{ForceVLA}, a
novel end-to-end manipulation framework that treats external force sensing as a
first-class modality within VLA systems. ForceVLA introduces \textbf{FVLMoE}, a
force-aware Mixture-of-Experts fusion module that dynamically integrates
pretrained visual-language embeddings with real-time 6-axis force feedback
during action decoding. This enables context-aware routing across
modality-specific experts, enhancing the robot's ability to adapt to subtle
contact dynamics. We also introduce \textbf{ForceVLA-Data}, a new dataset
comprising synchronized vision, proprioception, and force-torque signals across
five contact-rich manipulation tasks. ForceVLA improves average task success by
23.2\% over strong $\pi_0$-based baselines, achieving up to 80\% success in
tasks such as plug insertion. Our approach highlights the importance of
multimodal integration for dexterous manipulation and sets a new benchmark for
physically intelligent robotic control. Code and data will be released at
https://sites.google.com/view/forcevla2025.

### 10. [UP-SLAM: Adaptively Structured Gaussian SLAM with Uncertainty Prediction in Dynamic Environments](http://arxiv.org/pdf/2505.22335v1)

Authors: Wancai Zheng, Linlin Ou, Jiajie He, Libo Zhou, Xinyi Yu, Yan Wei

Recent 3D Gaussian Splatting (3DGS) techniques for Visual Simultaneous
Localization and Mapping (SLAM) have significantly progressed in tracking and
high-fidelity mapping. However, their sequential optimization framework and
sensitivity to dynamic objects limit real-time performance and robustness in
real-world scenarios. We present UP-SLAM, a real-time RGB-D SLAM system for
dynamic environments that decouples tracking and mapping through a parallelized
framework. A probabilistic octree is employed to manage Gaussian primitives
adaptively, enabling efficient initialization and pruning without hand-crafted
thresholds. To robustly filter dynamic regions during tracking, we propose a
training-free uncertainty estimator that fuses multi-modal residuals to
estimate per-pixel motion uncertainty, achieving open-set dynamic object
handling without reliance on semantic labels. Furthermore, a temporal encoder
is designed to enhance rendering quality. Concurrently, low-dimensional
features are efficiently transformed via a shallow multilayer perceptron to
construct DINO features, which are then employed to enrich the Gaussian field
and improve the robustness of uncertainty prediction. Extensive experiments on
multiple challenging datasets suggest that UP-SLAM outperforms state-of-the-art
methods in both localization accuracy (by 59.8%) and rendering quality (by 4.57
dB PSNR), while maintaining real-time performance and producing reusable,
artifact-free static maps in dynamic environments.The project:
https://aczheng-cai.github.io/up_slam.github.io/

### Software Engineering

### 1. [Thermal Modeling and Optimal Allocation of Avionics Safety-critical Tasks on Heterogeneous MPSoCs](http://arxiv.org/pdf/2505.22214v1)

Authors: Ondřej Benedikt, Michal Sojka, Přemysl Šůcha, Pavel Zaykov, Zdeněk Hanzálek

Multi-Processor Systems-on-Chip (MPSoC) can deliver high performance needed
in many industrial domains, including aerospace. However, their high power
consumption, combined with avionics safety standards, brings new thermal
management challenges. This paper investigates techniques for offline
thermal-aware allocation of periodic tasks on heterogeneous MPSoCs running at a
fixed clock frequency, as required in avionics. The goal is to find the
assignment of tasks to (i) cores and (ii) temporal isolation windows while
minimizing the MPSoC temperature. To achieve that, we propose and analyze three
power models, and integrate them within several novel optimization approaches
based on heuristics, a black-box optimizer, and Integer Linear Programming
(ILP). We perform the experimental evaluation on three popular MPSoC platforms
(NXP i.MX8QM MEK, NXP i.MX8QM Ixora, NVIDIA TX2) and observe a difference of up
to 5.5{\deg}C among the tested methods (corresponding to a 22% reduction w.r.t.
the ambient temperature). We also show that our method, integrating the
empirical power model with the ILP, outperforms the other methods on all tested
platforms.

### 2. [Evolution of repositories and privacy laws: commit activities in the GDPR and CCPA era](http://arxiv.org/pdf/2505.22234v1)

Authors: Georgia M. Kapitsaki, Maria Papoutsoglou

Free and open source software has gained a lot of momentum in the industry
and the research community. The latest advances in privacy legislation,
including the EU General Data Protection Regulation (GDPR) and the California
Consumer Privacy Act (CCPA), have forced the community to pay special attention
to users' data privacy. The main aim of this work is to examine software
repositories that are acting on privacy laws. We have collected commit data
from GitHub repositories in order to understand indications on main data
privacy laws (GDPR, CCPA, CPRA, UK DPA) in the last years. Via an automated
process, we analyzed 37,213 commits from 12,391 repositories since 2016,
whereas 594 commits from the 70 most popular repositories of the dataset were
manually analyzed. We observe that most commits were performed on the year the
law came into effect and privacy relevant terms appear in the commit messages,
whereas reference to specific data privacy user rights is scarce. The study
showed that more educational activities on data privacy user rights are needed,
as well as tools for privacy recommendations, whereas verifying actual
compliance via source code execution is a useful direction for software
engineering researchers.

### 3. [Securing the Software Package Supply Chain for Critical Systems](http://arxiv.org/pdf/2505.22023v1)

Authors: Ritwik Murali, Akash Ravi

Software systems have grown as an indispensable commodity used across various
industries, and almost all essential services depend on them for effective
operation. The software is no longer an independent or stand-alone piece of
code written by a developer but rather a collection of packages designed by
multiple developers across the globe. Ensuring the reliability and resilience
of these systems is crucial since emerging threats target software supply
chains, as demonstrated by the widespread SolarWinds hack in late 2020. These
supply chains extend beyond patches and updates, involving distribution
networks throughout the software lifecycle. Industries like smart grids,
manufacturing, healthcare, and finance rely on interconnected software systems
and their dependencies for effective functioning. To secure software modules
and add-ons, robust distribution architectures are essential. The proposed
chapter enhances the existing delivery frameworks by including a permissioned
ledger with Proof of Authority consensus and multi-party signatures. The
proposed system aims to prevent attacks while permitting every stakeholder to
verify the same. Critical systems can interface with the secure pipeline
without disrupting existing functionalities, thus preventing the cascading
effect of an attack at any point in the supply chain.

### 4. [Advancing Expert Specialization for Better MoE](http://arxiv.org/pdf/2505.22323v1)

Authors: Hongcan Guo, Haolang Lu, Guoshun Nan, Bolun Chu, Jialin Zhuang, Yuan Yang, Wenhao Che, Sicong Leng, Qimei Cui, Xudong Jiang

Mixture-of-Experts (MoE) models enable efficient scaling of large language
models (LLMs) by activating only a subset of experts per input. However, we
observe that the commonly used auxiliary load balancing loss often leads to
expert overlap and overly uniform routing, which hinders expert specialization
and degrades overall performance during post-training. To address this, we
propose a simple yet effective solution that introduces two complementary
objectives: (1) an orthogonality loss to encourage experts to process distinct
types of tokens, and (2) a variance loss to encourage more discriminative
routing decisions. Gradient-level analysis demonstrates that these objectives
are compatible with the existing auxiliary loss and contribute to optimizing
the training process. Experimental results over various model architectures and
across multiple benchmarks show that our method significantly enhances expert
specialization. Notably, our method improves classic MoE baselines with
auxiliary loss by up to 23.79%, while also maintaining load balancing in
downstream tasks, without any architectural modifications or additional
components. We will release our code to contribute to the community.

### 5. [GitGoodBench: A Novel Benchmark For Evaluating Agentic Performance On Git](http://arxiv.org/pdf/2505.22583v1)

Authors: Tobias Lindenbauer, Egor Bogomolov, Yaroslav Zharov

Benchmarks for Software Engineering (SE) AI agents, most notably SWE-bench,
have catalyzed progress in programming capabilities of AI agents. However, they
overlook critical developer workflows such as Version Control System (VCS)
operations. To address this issue, we present GitGoodBench, a novel benchmark
for evaluating AI agent performance on VCS tasks. GitGoodBench covers three
core Git scenarios extracted from permissive open-source Python, Java, and
Kotlin repositories. Our benchmark provides three datasets: a comprehensive
evaluation suite (900 samples), a rapid prototyping version (120 samples), and
a training corpus (17,469 samples). We establish baseline performance on the
prototyping version of our benchmark using GPT-4o equipped with custom tools,
achieving a 21.11% solve rate overall. We expect GitGoodBench to serve as a
crucial stepping stone toward truly comprehensive SE agents that go beyond mere
programming.

### 6. [BPMN to Smart Contract by Business Analyst](http://arxiv.org/pdf/2505.22612v1)

Authors: C. G. Liu, P. Bodorik, D. Jutla

This paper addresses the challenge of creating smart contracts for
applications represented using Business Process Management and Notation (BPMN)
models. In our prior work we presented a methodology that automates the
generation of smart contracts from BPMN models. This approach abstracts the
BPMN flow control, making it independent of the underlying blockchain
infrastructure, with only the BPMN task elements requiring coding. In
subsequent research, we enhanced our approach by adding support for nested
transactions and enabling a smart contract repair and/or upgrade. To empower
Business Analysts (BAs) to generate smart contracts without relying on software
developers, we tackled the challenge of generating smart contracts from BPMN
models without assistance of a software developer. We exploit the Decision
Model and Notation (DMN) standard to represent the decisions and the business
logic of the BPMN task elements and amended our methodology for transformation
of BPMN models into smart contracts to support also the generation script to
represent the business logic represented by the DMN models. To support such
transformation, we describe how the BA documents, using the BPMN elements, the
flow of information along with the flow of execution. Thus, if the BA is
successful in representing the blockchain application requirements using BPMN
and DMN models, our methodology and the tool, called TABS, that we developed as
a proof of concept, is used to generate the smart contracts directly from those
models without developer assistance.

### 7. [LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](http://arxiv.org/pdf/2505.22634v1)

Authors: Rui Li, Zixuan Hu, Wenxi Qu, Jinouwen Zhang, Zhenfei Yin, Sha Zhang, Xuantuo Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang, Wanli Ouyang, Lei Bai, Wangmeng Zuo, Ling-Yu Duan, Dongzhan Zhou, Shixiang Tang

Scientific embodied agents play a crucial role in modern laboratories by
automating complex experimental workflows. Compared to typical household
environments, laboratory settings impose significantly higher demands on
perception of physical-chemical transformations and long-horizon planning,
making them an ideal testbed for advancing embodied intelligence. However, its
development has been long hampered by the lack of suitable simulator and
benchmarks. In this paper, we address this gap by introducing LabUtopia, a
comprehensive simulation and benchmarking suite designed to facilitate the
development of generalizable, reasoning-capable embodied agents in laboratory
settings. Specifically, it integrates i) LabSim, a high-fidelity simulator
supporting multi-physics and chemically meaningful interactions; ii) LabScene,
a scalable procedural generator for diverse scientific scenes; and iii)
LabBench, a hierarchical benchmark spanning five levels of complexity from
atomic actions to long-horizon mobile manipulation. LabUtopia supports 30
distinct tasks and includes more than 200 scene and instrument assets, enabling
large-scale training and principled evaluation in high-complexity environments.
We demonstrate that LabUtopia offers a powerful platform for advancing the
integration of perception, planning, and control in scientific-purpose agents
and provides a rigorous testbed for exploring the practical capabilities and
generalization limits of embodied intelligence in future research.

### 8. [Co-Saving: Resource Aware Multi-Agent Collaboration for Software Development](http://arxiv.org/pdf/2505.21898v1)

Authors: Rennai Qiu, Chen Qian, Ran Li, Yufan Dang, Weize Chen, Cheng Yang, Yingli Zhang, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, Maosong Sun

Recent advancements in Large Language Models (LLMs) and autonomous agents
have demonstrated remarkable capabilities across various domains. However,
standalone agents frequently encounter limitations when handling complex tasks
that demand extensive interactions and substantial computational resources.
Although Multi-Agent Systems (MAS) alleviate some of these limitations through
collaborative mechanisms like task decomposition, iterative communication, and
role specialization, they typically remain resource-unaware, incurring
significant inefficiencies due to high token consumption and excessive
execution time. To address these limitations, we propose a resource-aware
multi-agent system -- Co-Saving (meaning that multiple agents collaboratively
engage in resource-saving activities), which leverages experiential knowledge
to enhance operational efficiency and solution quality. Our key innovation is
the introduction of "shortcuts" -- instructional transitions learned from
historically successful trajectories -- which allows to bypass redundant
reasoning agents and expedite the collective problem-solving process.
Experiments for software development tasks demonstrate significant advantages
over existing methods. Specifically, compared to the state-of-the-art MAS
ChatDev, our method achieves an average reduction of 50.85% in token usage, and
improves the overall code quality by 10.06%.

### 9. [Jailbreak Distillation: Renewable Safety Benchmarking](http://arxiv.org/pdf/2505.22037v1)

Authors: Jingyu Zhang, Ahmed Elgohary, Xiawei Wang, A S M Iftekhar, Ahmed Magooda, Benjamin Van Durme, Daniel Khashabi, Kyle Jackson

Large language models (LLMs) are rapidly deployed in critical applications,
raising urgent needs for robust safety benchmarking. We propose Jailbreak
Distillation (JBDistill), a novel benchmark construction framework that
"distills" jailbreak attacks into high-quality and easily-updatable safety
benchmarks. JBDistill utilizes a small set of development models and existing
jailbreak attack algorithms to create a candidate prompt pool, then employs
prompt selection algorithms to identify an effective subset of prompts as
safety benchmarks. JBDistill addresses challenges in existing safety
evaluation: the use of consistent evaluation prompts across models ensures fair
comparisons and reproducibility. It requires minimal human effort to rerun the
JBDistill pipeline and produce updated benchmarks, alleviating concerns on
saturation and contamination. Extensive experiments demonstrate our benchmarks
generalize robustly to 13 diverse evaluation models held out from benchmark
construction, including proprietary, specialized, and newer-generation LLMs,
significantly outperforming existing safety benchmarks in effectiveness while
maintaining high separability and diversity. Our framework thus provides an
effective, sustainable, and adaptable solution for streamlining safety
evaluation.

### 10. [Smart Contracts for SMEs and Large Companies](http://arxiv.org/pdf/2505.22619v1)

Authors: C. G. Liu, P. Bodorik, D. Jutla

Research on blockchains addresses multiple issues, with one being writing
smart contracts. In our previous research we described methodology and a tool
to generate, in automated fashion, smart contracts from BPMN models. The
generated smart contracts provide support for multi-step transactions that
facilitate repair/upgrade of smart contracts. In this paper we show how the
approach is used to support collaborations via smart contracts for companies
ranging from SMEs with little IT capabilities to companies with IT using
blockchain smart contracts. Furthermore, we also show how the approach is used
for certain applications to generate smart contracts by a BPMN modeler who does
not need any knowledge of blockchain technology or smart contract development -
thus we are hoping to facilitate democratization of smart contracts and
blockchain technology.

### Social and Information Networks

### 1. [Retweets, Receipts, and Resistance: Discourse, Sentiment, and Credibility in Public Health Crisis Twitter](http://arxiv.org/pdf/2505.22032v1)

Authors: Tawfiq Ammari, Anna Gutowska, Jacob Ziff, Casey Randazzo, Harihan Subramonyam

As the COVID-19 pandemic evolved, the Centers for Disease Control and
Prevention (CDC) used Twitter to disseminate safety guidance and updates,
reaching millions of users. This study analyzes two years of tweets from, to,
and about the CDC using a mixed methods approach to examine discourse
characteristics, credibility, and user engagement. We found that the CDCs
communication remained largely one directional and did not foster reciprocal
interaction, while discussions around COVID19 were deeply shaped by political
and ideological polarization. Users frequently cited earlier CDC messages to
critique new and sometimes contradictory guidance. Our findings highlight the
role of sentiment, media richness, and source credibility in shaping the spread
of public health messages. We propose design strategies to help the CDC tailor
communications to diverse user groups and manage misinformation more
effectively during high-stakes health crises.

### 2. [Uncertainty Estimation for Heterophilic Graphs Through the Lens of Information Theory](http://arxiv.org/pdf/2505.22152v1)

Authors: Dominik Fuchsgruber, Tom Wollschläger, Johannes Bordne, Stephan Günnemann

While uncertainty estimation for graphs recently gained traction, most
methods rely on homophily and deteriorate in heterophilic settings. We address
this by analyzing message passing neural networks from an information-theoretic
perspective and developing a suitable analog to data processing inequality to
quantify information throughout the model's layers. In contrast to non-graph
domains, information about the node-level prediction target can increase with
model depth if a node's features are semantically different from its neighbors.
Therefore, on heterophilic graphs, the latent embeddings of an MPNN each
provide different information about the data distribution - different from
homophilic settings. This reveals that considering all node representations
simultaneously is a key design principle for epistemic uncertainty estimation
on graphs beyond homophily. We empirically confirm this with a simple post-hoc
density estimator on the joint node embedding space that provides
state-of-the-art uncertainty on heterophilic graphs. At the same time, it
matches prior work on homophilic graphs without explicitly exploiting homophily
through post-processing.

### 3. [A Systematic Approach for Studying How Topological Measurements Respond to Complex Networks Modifications](http://arxiv.org/pdf/2505.22345v1)

Authors: Alexandre Benatti, Roberto M. Cesar Jr., Luciano da F. Costa

Different types of graphs and complex networks have been characterized,
analyzed, and modeled based on measurements of their respective topology.
However, the available networks may constitute approximations of the original
structure as a consequence of sampling incompleteness, noise, and/or error in
the representation of that structure. Therefore, it becomes of particular
interest to quantify how successive modifications may impact a set of adopted
topological measurements, and how respectively undergone changes can be
interrelated, which has been addressed in this paper by considering similarity
networks and hierarchical clustering approaches. These studies are developed
respectively to several topological measurements (accessibility, degree,
hierarchical degree, clustering coefficient, betweenness centrality,
assortativity, and average shortest path) calculated from complex networks of
three main types (Erd\H{o}s-R\'enyi, Barab\'asi-Albert, and geographical) with
varying sizes or subjected to progressive edge removal or rewiring. The
coincidence similarity index, which can implement particularly strict
comparisons, is adopted for two main purposes: to quantify and visualize how
the considered topological measurements respond to the considered network
alterations and to represent hierarchically the relationships between the
observed changes undergone by the considered topological measurements. Several
results are reported and discussed, including the identification of three types
of topological changes taking place as a consequence of the modifications. In
addition, the changes observed for the Erd\H{o}s-R\'enyi and Barab\'asi-Albert
networks resulted mutually more similarly affected by topological changes than
for the geometrical networks. The latter type of network has been identified to
have more heterogeneous topological features than the other two types of
networks.

### 4. [Spectral clustering for dependent community Hawkes process models of temporal networks](http://arxiv.org/pdf/2505.21845v1)

Authors: Lingfei Zhao, Hadeel Soliman, Kevin S. Xu, Subhadeep Paul

Temporal networks observed continuously over time through timestamped
relational events data are commonly encountered in application settings
including online social media communications, financial transactions, and
international relations. Temporal networks often exhibit community structure
and strong dependence patterns among node pairs. This dependence can be modeled
through mutual excitations, where an interaction event from a sender to a
receiver node increases the possibility of future events among other node
pairs.
  We provide statistical results for a class of models that we call dependent
community Hawkes (DCH) models, which combine the stochastic block model with
mutually exciting Hawkes processes for modeling both community structure and
dependence among node pairs, respectively. We derive a non-asymptotic upper
bound on the misclustering error of spectral clustering on the event count
matrix as a function of the number of nodes and communities, time duration, and
the amount of dependence in the model. Our result leverages recent results on
bounding an appropriate distance between a multivariate Hawkes process count
vector and a Gaussian vector, along with results from random matrix theory. We
also propose a DCH model that incorporates only self and reciprocal excitation
along with highly scalable parameter estimation using a Generalized Method of
Moments (GMM) estimator that we demonstrate to be consistent for growing
network size and time duration.

### Systems and Control

### 1. [Large Language Models for Solving Economic Dispatch Problem](http://arxiv.org/pdf/2505.21931v1)

Authors: Sina Mohammadi, Ali Hassan, Rouzbeh Haghighi, Van-Hai Bui, Wencong Su

This paper investigates the capability of off-the-shelf large language models
(LLMs) to solve the economic dispatch (ED) problem. ED is a hard-constrained
optimization problem solved on a day-ahead timescale by grid operators to
minimize electricity generation costs while accounting for physical and
engineering constraints. Numerous approaches have been proposed, but these
typically require either mathematical formulations, face convergence issues, or
depend on extensive labeled data and training time. This work implements LLMs
enhanced with reasoning capabilities to address the classic lossless ED
problem. The proposed approach avoids the need for explicit mathematical
formulations, does not suffer from convergence challenges, and requires neither
labeled data nor extensive training. A few-shot learning technique is utilized
in two different prompting contexts. The IEEE 118-bus system with 19 generation
units serves as the evaluation benchmark. Results demonstrate that various
prompting strategies enable LLMs to effectively solve the ED problem, offering
a convenient and efficient alternative. Consequently, this approach presents a
promising future solution for ED tasks, particularly when foundational power
system models are available.

### 2. [Dynamic State-Feedback Control for LPV Systems: Ensuring Stability and LQR Performance](http://arxiv.org/pdf/2505.22248v1)

Authors: Armin Gießler, Felix Strehle, Jochen Illerhaus, Sören Hohmann

In this paper, we propose a novel dynamic state-feedback controller for
polytopic linear parameter-varying (LPV) systems with constant input matrix.
The controller employs a projected gradient flow method to continuously improve
its control law and, under established conditions, converges to the optimal
feedback gain of the corresponding linear quadratic regulator (LQR) problem
associated with constant parameter trajectories. We derive conditions for
quadratic stability, which can be verified via convex optimization, to ensure
exponential stability of the LPV system even under arbitrarily fast parameter
variations. Additionally, we provide sufficient conditions to guarantee the
boundedness of the trajectories of the dynamic controller for any parameter
trajectory and the convergence of its feedback gains to the optimal LQR gains
for constant parameter trajectories. Furthermore, we show that the closed-loop
system is asymptotically stable for constant parameter trajectories under these
conditions. Simulation results demonstrate that the controller maintains
stability and improves transient performance.

### 3. [A memristive model of spatio-temporal excitability](http://arxiv.org/pdf/2505.22269v1)

Authors: Thomas SJ Burger, Amir Shahhosseini, Rodolphe Sepulchre

This paper introduces a model of excitability that unifies the mechanism of
an important neuronal property both in time and in space. As a starting point,
we revisit both a key model of temporal excitability, proposed by Hodgkin and
Huxley, and a key model of spatial excitability, proposed by Amari. We then
propose a novel model that captures the temporal and spatial properties of both
models. Our aim is to regard neuronal excitability as a property across scales,
and to explore the benefits of modeling excitability with one and the same
mechanism, whether at the cellular or the population level.

### 4. [A Multi-output Gaussian Process Regression with Negative Transfer Mitigation for Generating Boundary Test Scenarios of Multi-UAV Systems](http://arxiv.org/pdf/2505.22331v1)

Authors: Hanxu Jiang, Haiyue Yu, Xiaotong Xie, Qi Gao, Jiang Jiang, Jianbin Sun

Adaptive sampling based on Gaussian process regression (GPR) has already been
applied with considerable success to generate boundary test scenarios for
multi-UAV systems (MUS). One of the key techniques in such researches is
leveraging the accurate prediction of the MUS performance through GPR in
different test scenarios. Due to the potential correlations among the multiple
MUS performance metrics, current researches commonly utilize a multi-output GPR
(MOGPR) to model the multiple performance metrics simultaneously. This approach
can achieve a more accurate prediction, rather than modeling each metric
individually. However, MOGPR still suffers from negative transfer. When the
feature of one output variable is incorrectly learned by another, the models
training process will be negatively affected, leading to a decline in
prediction performance. To solve this problem, this paper proposes a novel
adaptive regularization approach into the conventional MOGPR training process.
Unlike existing regularization approaches for mitigating negative transfer in
MOGPR, our method penalizes the inconsistencies among output-specific
characteristic parameters using adaptively adjustable regularization weights.
This mechanism helps each set of output parameters avoid local optima.
Consequently, it yields simultaneous improvements in predictive accuracy across
all outputs. Finally, we validate our approach on a numerical case and on a
boundary test scenario generation case for a MUS multi-objectives search task.

### 5. [State Constrained Model Reference Adaptive Control with Input Amplitude and Rate Limits](http://arxiv.org/pdf/2505.22346v1)

Authors: Poulomee Ghosh, Shubhendu Bhasin

This paper proposes a robust model reference adaptive controller (MRAC) for
uncertain multi-input multi-output (MIMO) linear time-invariant (LTI) plants
with user-defined constraints on the plant states, input amplitude, and input
rate. The proposed two-layer barrier Lyapunov function (BLF)-based control
design considers the input and the input rate as states that are constrained
using two BLFs in the first layer, while another BLF in the second layer
constrains the plant states. The adaptive control law ensures that the plant
states, input amplitude, and input rate remain within the user-defined safe
sets despite unmatched bounded disturbances. Sufficient conditions for the
existence of a feasible control policy are also provided. To the best of the
authors' knowledge, this is the first optimization-free method that imposes
user-defined constraints on the state, input, and input rate and also provides
verifiable feasibility conditions in the presence of parametric uncertainties
and disturbances. Simulation results demonstrate the effectiveness of the
proposed algorithm.

### 6. [State and Input Constrained Adaptive Tracking Control of Uncertain Euler-Lagrange Systems with Robustness and Feasibility Analysis](http://arxiv.org/pdf/2505.22352v1)

Authors: Poulomee Ghosh, Shubhendu Bhasin

This paper proposes an adaptive tracking controller for uncertain
Euler-Lagrange (E-L) systems with user-defined state and input constraints in
presence of bounded external disturbances. A barrier Lyapunov function (BLF) is
employed for state constraint satisfaction, integrated with a saturated
controller that ensures the control input remains within pre-specified bounds.
To the best of the authors' knowledge, this is the first result on tracking
control of state and input-constrained uncertain E-L systems that provides
verifiable conditions for the existence of a feasible control policy. The
efficacy of the proposed controller in terms of constraint satisfaction and
tracking performance is demonstrated through simulation on a robotic
manipulator system.

### 7. [Operator-Splitting Methods for Neuromorphic Circuit Simulation](http://arxiv.org/pdf/2505.22363v1)

Authors: Amir Shahhosseini, Thomas Chaffey, Rodolphe Sepulchre

A novel splitting algorithm is proposed for the numerical simulation of
neuromorphic circuits. The algorithm is grounded in the operator-theoretic
concept of monotonicity, which bears both physical and algorithmic
significance. The splitting exploits this correspondence to translate the
circuit architecture into the algorithmic architecture. The paper illustrates
the many advantages of the proposed operator-theoretic framework over
conventional numerical integration for the simulation of multiscale
hierarchical events that characterize neuromorphic behaviors.

### 8. [Current trends and future directions in event-based control](http://arxiv.org/pdf/2505.22378v1)

Authors: Michael Hertneck, David Meister, Frank Allgöwer

The defining characteristic of event-based control is that feedback loops are
only closed when indicated by a triggering condition that takes recent
information about the system into account. This stands in contrast to periodic
control where the feedback loop is closed periodically. Benefits of event-based
control arise when sampling comes at a cost, which occurs, e.g., for Networked
Control Systems or in other setups with resource constraints. A rapidly growing
number of publications deals with event-based control. Nevertheless, some
fundamental questions about event-based control are still unsolved. In this
article, we provide an overview of current research trends in event-based
control. We focus on results that aim for a better understanding of effects
that occur in feedback loops with event-based control. Based on this summary,
we identify important open directions for future research.

### 9. [Data-Driven Control of Continuous-Time LTI Systems via Non-Minimal Realizations](http://arxiv.org/pdf/2505.22505v1)

Authors: Alessandro Bosso, Marco Borghesi, Andrea Iannelli, Giuseppe Notarstefano, Andrew R. Teel

This article proposes an approach to design output-feedback controllers for
unknown continuous-time linear time-invariant systems using only input-output
data from a single experiment. To address the lack of state and derivative
measurements, we introduce non-minimal realizations whose states can be
observed by filtering the available data. We first apply this concept to the
disturbance-free case, formulating linear matrix inequalities (LMIs) from
batches of sampled signals to design a dynamic, filter-based stabilizing
controller. The framework is then extended to the problem of asymptotic
tracking and disturbance rejection - in short, output regulation - by
incorporating an internal model based on prior knowledge of the
disturbance/reference frequencies. Finally, we discuss tuning strategies for a
class of multi-input multi-output systems and illustrate the method via
numerical examples.

### 10. [A Physics-Informed Learning Framework to Solve the Infinite-Horizon Optimal Control Problem](http://arxiv.org/pdf/2505.21842v1)

Authors: Filippos Fotiadis, Kyriakos G. Vamvoudakis

We propose a physics-informed neural networks (PINNs) framework to solve the
infinite-horizon optimal control problem of nonlinear systems. In particular,
since PINNs are generally able to solve a class of partial differential
equations (PDEs), they can be employed to learn the value function of the
infinite-horizon optimal control problem via solving the associated
steady-state Hamilton-Jacobi-Bellman (HJB) equation. However, an issue here is
that the steady-state HJB equation generally yields multiple solutions; hence
if PINNs are directly employed to it, they may end up approximating a solution
that is different from the optimal value function of the problem. We tackle
this by instead applying PINNs to a finite-horizon variant of the steady-state
HJB that has a unique solution, and which uniformly approximates the optimal
value function as the horizon increases. An algorithm to verify if the chosen
horizon is large enough is also given, as well as a method to extend it -- with
reduced computations and robustness to approximation errors -- in case it is
not. Unlike many existing methods, the proposed technique works well with
non-polynomial basis functions, does not require prior knowledge of a
stabilizing controller, and does not perform iterative policy evaluations.
Simulations are performed, which verify and clarify theoretical findings.

### Machine Learning (Statistics Category)

### 1. [Revisiting Bayesian Model Averaging in the Era of Foundation Models](http://arxiv.org/pdf/2505.21857v1)

Authors: Mijung Park

We revisit the classical, full-fledged Bayesian model averaging (BMA)
paradigm to ensemble pre-trained and/or lightly-finetuned foundation models to
enhance the classification performance on image and text data. To make BMA
tractable under foundation models, we introduce trainable linear classifiers
that take frozen features from the pre-trained foundation models as inputs. The
model posteriors over the linear classifiers tell us which linear heads and
frozen features are better suited for a given dataset, resulting in a
principled model ensembling method. Furthermore, we propose a computationally
cheaper, optimizable model averaging scheme (OMA). In OMA, we directly optimize
the model ensemble weights, just like those weights based on model posterior
distributions in BMA, by reducing the amount of surprise (expected entropy of
the predictions) we get from predictions of ensembled models. With the rapid
development of foundation models, these approaches will enable the
incorporation of future, possibly significantly better foundation models to
enhance the performance of challenging classification tasks.

### 2. [Almost Linear Convergence under Minimal Score Assumptions: Quantized Transition Diffusion](http://arxiv.org/pdf/2505.21892v1)

Authors: Xunpeng Huang, Yingyu Lin, Nikki Lijing Kuang, Hanze Dong, Difan Zou, Yian Ma, Tong Zhang

Continuous diffusion models have demonstrated remarkable performance in data
generation across various domains, yet their efficiency remains constrained by
two critical limitations: (1) the local adjacency structure of the forward
Markov process, which restricts long-range transitions in the data space, and
(2) inherent biases introduced during the simulation of time-inhomogeneous
reverse denoising processes. To address these challenges, we propose Quantized
Transition Diffusion (QTD), a novel approach that integrates data quantization
with discrete diffusion dynamics. Our method first transforms the continuous
data distribution $p_*$ into a discrete one $q_*$ via histogram approximation
and binary encoding, enabling efficient representation in a structured discrete
latent space. We then design a continuous-time Markov chain (CTMC) with Hamming
distance-based transitions as the forward process, which inherently supports
long-range movements in the original data space. For reverse-time sampling, we
introduce a \textit{truncated uniformization} technique to simulate the reverse
CTMC, which can provably provide unbiased generation from $q_*$ under minimal
score assumptions. Through a novel KL dynamic analysis of the reverse CTMC, we
prove that QTD can generate samples with $O(d\ln^2(d/\epsilon))$ score
evaluations in expectation to approximate the $d$--dimensional target
distribution $p_*$ within an $\epsilon$ error tolerance. Our method not only
establishes state-of-the-art inference efficiency but also advances the
theoretical foundations of diffusion-based generative modeling by unifying
discrete and continuous diffusion paradigms.

### 3. [Continual Learning Beyond Experience Rehearsal and Full Model Surrogates](http://arxiv.org/pdf/2505.21942v1)

Authors: Prashant Bhat, Laurens Niesten, Elahe Arani, Bahram Zonooz

Continual learning (CL) has remained a significant challenge for deep neural
networks as learning new tasks erases previously acquired knowledge, either
partially or completely. Existing solutions often rely on experience rehearsal
or full model surrogates to mitigate CF. While effective, these approaches
introduce substantial memory and computational overhead, limiting their
scalability and applicability in real-world scenarios. To address this, we
propose SPARC, a scalable CL approach that eliminates the need for experience
rehearsal and full-model surrogates. By effectively combining task-specific
working memories and task-agnostic semantic memory for cross-task knowledge
consolidation, SPARC results in a remarkable parameter efficiency, using only
6% of the parameters required by full-model surrogates. Despite its lightweight
design, SPARC achieves superior performance on Seq-TinyImageNet and matches
rehearsal-based methods on various CL benchmarks. Additionally, weight
re-normalization in the classification layer mitigates task-specific biases,
establishing SPARC as a practical and scalable solution for CL under stringent
efficiency constraints.

### 4. [Learning Curves of Stochastic Gradient Descent in Kernel Regression](http://arxiv.org/pdf/2505.22048v1)

Authors: Haihan Zhang, Weicheng Lin, Yuanshi Liu, Cong Fang

This paper considers a canonical problem in kernel regression: how good are
the model performances when it is trained by the popular online first-order
algorithms, compared to the offline ones, such as ridge and ridgeless
regression? In this paper, we analyze the foundational single-pass Stochastic
Gradient Descent (SGD) in kernel regression under source condition where the
optimal predictor can even not belong to the RKHS, i.e. the model is
misspecified. Specifically, we focus on the inner product kernel over the
sphere and characterize the exact orders of the excess risk curves under
different scales of sample sizes $n$ concerning the input dimension $d$.
Surprisingly, we show that SGD achieves min-max optimal rates up to constants
among all the scales, without suffering the saturation, a prevalent phenomenon
observed in (ridge) regression, except when the model is highly misspecified
and the learning is in a final stage where $n\gg d^{\gamma}$ with any constant
$\gamma >0$. The main reason for SGD to overcome the curse of saturation is the
exponentially decaying step size schedule, a common practice in deep neural
network training. As a byproduct, we provide the \emph{first} provable
advantage of the scheme over the iterative averaging method in the common
setting.

### 5. [Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training](http://arxiv.org/pdf/2505.22257v1)

Authors: Youssef Mroueh, Nicolas Dupuis, Brian Belgodere, Apoorva Nitsure, Mattia Rigotti, Kristjan Greenewald, Jiri Navratil, Jerret Ross, Jesus Rios

We revisit Group Relative Policy Optimization (GRPO) in both on-policy and
off-policy optimization regimes. Our motivation comes from recent work on
off-policy Proximal Policy Optimization (PPO), which improves training
stability, sampling efficiency, and memory usage. In addition, a recent
analysis of GRPO suggests that estimating the advantage function with
off-policy samples could be beneficial. Building on these observations, we
adapt GRPO to the off-policy setting. We show that both on-policy and
off-policy GRPO objectives yield an improvement in the reward. This result
motivates the use of clipped surrogate objectives in the off-policy version of
GRPO. We then compare the empirical performance of reinforcement learning with
verifiable rewards in post-training using both GRPO variants. Our results show
that off-policy GRPO either significantly outperforms or performs on par with
its on-policy counterpart.

### 6. [Individualised Counterfactual Examples Using Conformal Prediction Intervals](http://arxiv.org/pdf/2505.22326v1)

Authors: James M. Adams, Gesine Reinert, Lukasz Szpruch, Carsten Maple, Andrew Elliott

Counterfactual explanations for black-box models aim to pr ovide insight into
an algorithmic decision to its recipient. For a binary classification problem
an individual counterfactual details which features might be changed for the
model to infer the opposite class. High-dimensional feature spaces that are
typical of machine learning classification models admit many possible
counterfactual examples to a decision, and so it is important to identify
additional criteria to select the most useful counterfactuals. In this paper,
we explore the idea that the counterfactuals should be maximally informative
when considering the knowledge of a specific individual about the underlying
classifier. To quantify this information gain we explicitly model the knowledge
of the individual, and assess the uncertainty of predictions which the
individual makes by the width of a conformal prediction interval. Regions of
feature space where the prediction interval is wide correspond to areas where
the confidence in decision making is low, and an additional counterfactual
example might be more informative to an individual. To explore and evaluate our
individualised conformal prediction interval counterfactuals (CPICFs), first we
present a synthetic data set on a hypercube which allows us to fully visualise
the decision boundary, conformal intervals via three different methods, and
resultant CPICFs. Second, in this synthetic data set we explore the impact of a
single CPICF on the knowledge of an individual locally around the original
query. Finally, in both our synthetic data set and a complex real world dataset
with a combination of continuous and discrete variables, we measure the utility
of these counterfactuals via data augmentation, testing the performance on a
held out set.

### 7. [Credal Prediction based on Relative Likelihood](http://arxiv.org/pdf/2505.22332v1)

Authors: Timo Löhr, Paul Hofman, Felix Mohr, Eyke Hüllermeier

Predictions in the form of sets of probability distributions, so-called
credal sets, provide a suitable means to represent a learner's epistemic
uncertainty. In this paper, we propose a theoretically grounded approach to
credal prediction based on the statistical notion of relative likelihood: The
target of prediction is the set of all (conditional) probability distributions
produced by the collection of plausible models, namely those models whose
relative likelihood exceeds a specified threshold. This threshold has an
intuitive interpretation and allows for controlling the trade-off between
correctness and precision of credal predictions. We tackle the problem of
approximating credal sets defined in this way by means of suitably modified
ensemble learning techniques. To validate our approach, we illustrate its
effectiveness by experiments on benchmark datasets demonstrating superior
uncertainty representation without compromising predictive performance. We also
compare our method against several state-of-the-art baselines in credal
prediction.

### 8. [Continuum-armed Bandit Optimization with Batch Pairwise Comparison Oracles](http://arxiv.org/pdf/2505.22361v1)

Authors: Xiangyu Chang, Xi Chen, Yining Wang, Zhiyi Zeng

This paper studies a bandit optimization problem where the goal is to
maximize a function $f(x)$ over $T$ periods for some unknown strongly concave
function $f$. We consider a new pairwise comparison oracle, where the
decision-maker chooses a pair of actions $(x, x')$ for a consecutive number of
periods and then obtains an estimate of $f(x)-f(x')$. We show that such a
pairwise comparison oracle finds important applications to joint pricing and
inventory replenishment problems and network revenue management. The challenge
in this bandit optimization is twofold. First, the decision-maker not only
needs to determine a pair of actions $(x, x')$ but also a stopping time $n$
(i.e., the number of queries based on $(x, x')$). Second, motivated by our
inventory application, the estimate of the difference $f(x)-f(x')$ is biased,
which is different from existing oracles in stochastic optimization literature.
To address these challenges, we first introduce a discretization technique and
local polynomial approximation to relate this problem to linear bandits. Then
we developed a tournament successive elimination technique to localize the
discretized cell and run an interactive batched version of LinUCB algorithm on
cells. We establish regret bounds that are optimal up to poly-logarithmic
factors. Furthermore, we apply our proposed algorithm and analytical framework
to the two operations management problems and obtain results that improve
state-of-the-art results in the existing literature.

### 9. [Computing Optimal Transport Maps and Wasserstein Barycenters Using Conditional Normalizing Flows](http://arxiv.org/pdf/2505.22364v1)

Authors: Gabriele Visentin, Patrick Cheridito

We present a novel method for efficiently computing optimal transport maps
and Wasserstein barycenters in high-dimensional spaces. Our approach uses
conditional normalizing flows to approximate the input distributions as
invertible pushforward transformations from a common latent space. This makes
it possible to directly solve the primal problem using gradient-based
minimization of the transport cost, unlike previous methods that rely on dual
formulations and complex adversarial optimization. We show how this approach
can be extended to compute Wasserstein barycenters by solving a conditional
variance minimization problem. A key advantage of our conditional architecture
is that it enables the computation of barycenters for hundreds of input
distributions, which was computationally infeasible with previous methods. Our
numerical experiments illustrate that our approach yields accurate results
across various high-dimensional tasks and compares favorably with previous
state-of-the-art methods.

### 10. [Hypothesis Testing in Imaging Inverse Problems](http://arxiv.org/pdf/2505.22481v1)

Authors: Yiming Xi, Konstantinos Zygalakis, Marcelo Pereyra

This paper proposes a framework for semantic hypothesis testing tailored to
imaging inverse problems. Modern imaging methods struggle to support hypothesis
testing, a core component of the scientific method that is essential for the
rigorous interpretation of experiments and robust interfacing with
decision-making processes. There are three main reasons why image-based
hypothesis testing is challenging. First, the difficulty of using a single
observation to simultaneously reconstruct an image, formulate hypotheses, and
quantify their statistical significance. Second, the hypotheses encountered in
imaging are mostly of semantic nature, rather than quantitative statements
about pixel values. Third, it is challenging to control test error
probabilities because the null and alternative distributions are often unknown.
Our proposed approach addresses these difficulties by leveraging concepts from
self-supervised computational imaging, vision-language models, and
non-parametric hypothesis testing with e-values. We demonstrate our proposed
framework through numerical experiments related to image-based phenotyping,
where we achieve excellent power while robustly controlling Type I errors.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

### 1. [How the natural world is inspiring the robot eyes of the future](https://www.nature.com/articles/d41586-025-01660-5)

Authors: Esme Hedley

### 2. [Improvement of metaphor understanding via a cognitive linguistic model based on hierarchical classification and artificial intelligence SVM](https://www.nature.com/articles/s41598-025-04171-5)

Authors: Dongmei Zhu

### 3. [Gaussian random fields as an abstract representation of patient metadata for multimodal medical image segmentation](https://www.nature.com/articles/s41598-025-03393-x)

Authors: Bill Cassidy et al.

### 4. [Explicit intent enhanced contrastive learning with denoising networks for sequential recommendation](https://www.nature.com/articles/s41598-025-03047-y)

Authors: Jinfang Sheng et al.

### 5. [Federated learning using a memristor compute-in-memory chip with in situ physical unclonable function and true random number generator](https://www.nature.com/articles/s41928-025-01390-6)

Authors: Xueqi Li et al.

### 6. [Hierarchical Information-guided robotic grasp detection](https://www.nature.com/articles/s41598-025-03313-z)

Authors: Zeyao Hou et al.

### 7. [Multimodal fusion transformer network for multispectral pedestrian detection in low-light condition](https://www.nature.com/articles/s41598-025-03567-7)

Authors: Gong Li et al.

### 8. [Action recognition using part and attention enhanced feature fusion](https://www.nature.com/articles/s41598-025-02461-6)

Authors: Danfeng Zhuang et al.

### 9. [Decoding dynamic brain networks in Parkinson’s disease with temporal attention](https://www.nature.com/articles/s41598-025-01106-y)

Authors: Salil B Patel et al.

