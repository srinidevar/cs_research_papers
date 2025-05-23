# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-22 18:43:02.756011 PST.

### Artificial Intelligence

### 1. [HAVA: Hybrid Approach to Value-Alignment through Reward Weighing for Reinforcement Learning](http://arxiv.org/pdf/2505.15011v1)

Authors: Kryspin Varys, Federico Cerutti, Adam Sobey, Timothy J. Norman

Our society is governed by a set of norms which together bring about the
values we cherish such as safety, fairness or trustworthiness. The goal of
value-alignment is to create agents that not only do their tasks but through
their behaviours also promote these values. Many of the norms are written as
laws or rules (legal / safety norms) but even more remain unwritten (social
norms). Furthermore, the techniques used to represent these norms also differ.
Safety / legal norms are often represented explicitly, for example, in some
logical language while social norms are typically learned and remain hidden in
the parameter space of a neural network. There is a lack of approaches in the
literature that could combine these various norm representations into a single
algorithm. We propose a novel method that integrates these norms into the
reinforcement learning process. Our method monitors the agent's compliance with
the given norms and summarizes it in a quantity we call the agent's reputation.
This quantity is used to weigh the received rewards to motivate the agent to
become value-aligned. We carry out a series of experiments including a
continuous state space traffic problem to demonstrate the importance of the
written and unwritten norms and show how our method can find the value-aligned
policies. Furthermore, we carry out ablations to demonstrate why it is better
to combine these two groups of norms rather than using either separately.

### 2. [lmgame-Bench: How Good are LLMs at Playing Games?](http://arxiv.org/pdf/2505.15146v1)

Authors: Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P. Xing, Ion Stoica, Tajana Rosing, Haojian Jin, Hao Zhang

Playing video games requires perception, memory, and planning, exactly the
faculties modern large language model (LLM) agents are expected to master. We
study the major challenges in using popular video games to evaluate modern LLMs
and find that directly dropping LLMs into games cannot make an effective
evaluation, for three reasons -- brittle vision perception, prompt sensitivity,
and potential data contamination. We introduce lmgame-Bench to turn games into
reliable evaluations. lmgame-Bench features a suite of platformer, puzzle, and
narrative games delivered through a unified Gym-style API and paired with
lightweight perception and memory scaffolds, and is designed to stabilize
prompt variance and remove contamination. Across 13 leading models, we show
lmgame-Bench is challenging while still separating models well. Correlation
analysis shows that every game probes a unique blend of capabilities often
tested in isolation elsewhere. More interestingly, performing reinforcement
learning on a single game from lmgame-Bench transfers both to unseen games and
to external planning tasks. Our evaluation code is available at
https://github.com/lmgame-org/GamingAgent/lmgame-bench.

### 3. [Identification of Probabilities of Causation: A Complete Characterization](http://arxiv.org/pdf/2505.15274v1)

Authors: Xin Shu, Shuai Wang, Ang Li

Probabilities of causation are fundamental to modern decision-making. Pearl
first introduced three binary probabilities of causation, and Tian and Pearl
later derived tight bounds for them using Balke's linear programming. The
theoretical characterization of probabilities of causation with multi-valued
treatments and outcomes has remained unresolved for decades, limiting the scope
of causality-based decision-making. In this paper, we resolve this foundational
gap by proposing a complete set of representative probabilities of causation
and proving that they are sufficient to characterize all possible probabilities
of causation within the framework of Structural Causal Models (SCMs). We then
formally derive tight bounds for these representative quantities using formal
mathematical proofs. Finally, we demonstrate the practical relevance of our
results through illustrative toy examples.

### 4. [Average Reward Reinforcement Learning for Omega-Regular and Mean-Payoff Objectives](http://arxiv.org/pdf/2505.15693v1)

Authors: Milad Kazemi, Mateo Perez, Fabio Somenzi, Sadegh Soudjani, Ashutosh Trivedi, Alvaro Velasquez

Recent advances in reinforcement learning (RL) have renewed focus on the
design of reward functions that shape agent behavior. Manually designing reward
functions is tedious and error-prone. A principled alternative is to specify
behaviors in a formal language that can be automatically translated into
rewards. Omega-regular languages are a natural choice for this purpose, given
their established role in formal verification and synthesis. However, existing
methods using omega-regular specifications typically rely on discounted reward
RL in episodic settings, with periodic resets. This setup misaligns with the
semantics of omega-regular specifications, which describe properties over
infinite behavior traces. In such cases, the average reward criterion and the
continuing setting -- where the agent interacts with the environment over a
single, uninterrupted lifetime -- are more appropriate.
  To address the challenges of infinite-horizon, continuing tasks, we focus on
absolute liveness specifications -- a subclass of omega-regular languages that
cannot be violated by any finite behavior prefix, making them well-suited to
the continuing setting. We present the first model-free RL framework that
translates absolute liveness specifications to average-reward objectives. Our
approach enables learning in communicating MDPs without episodic resetting. We
also introduce a reward structure for lexicographic multi-objective
optimization, aiming to maximize an external average-reward objective among the
policies that also maximize the satisfaction probability of a given
omega-regular specification. Our method guarantees convergence in unknown
communicating MDPs and supports on-the-fly reductions that do not require full
knowledge of the environment, thus enabling model-free RL. Empirical results
show our average-reward approach in continuing setting outperforms
discount-based methods across benchmarks.

### 5. [One-Layer Transformers are Provably Optimal for In-context Reasoning and Distributional Association Learning in Next-Token Prediction Tasks](http://arxiv.org/pdf/2505.15009v1)

Authors: Quan Nguyen, Thanh Nguyen-Tang

We study the approximation capabilities and on-convergence behaviors of
one-layer transformers on the noiseless and noisy in-context reasoning of
next-token prediction. Existing theoretical results focus on understanding the
in-context reasoning behaviors for either the first gradient step or when the
number of samples is infinite. Furthermore, no convergence rates nor
generalization abilities were known. Our work addresses these gaps by showing
that there exists a class of one-layer transformers that are provably
Bayes-optimal with both linear and ReLU attention. When being trained with
gradient descent, we show via a finite-sample analysis that the expected loss
of these transformers converges at linear rate to the Bayes risk. Moreover, we
prove that the trained models generalize to unseen samples as well as exhibit
learning behaviors that were empirically observed in previous works. Our
theoretical findings are further supported by extensive empirical validations.

### 6. [Towards a Science of Causal Interpretability in Deep Learning for Software Engineering](http://arxiv.org/pdf/2505.15023v1)

Authors: David N. Palacio

This dissertation addresses achieving causal interpretability in Deep
Learning for Software Engineering (DL4SE). While Neural Code Models (NCMs) show
strong performance in automating software tasks, their lack of transparency in
causal relationships between inputs and outputs limits full understanding of
their capabilities. To build trust in NCMs, researchers and practitioners must
explain code predictions. Associational interpretability, which identifies
correlations, is often insufficient for tasks requiring intervention and change
analysis. To address this, the dissertation introduces DoCode, a novel post hoc
interpretability method for NCMs. DoCode uses causal inference to provide
programming language-oriented explanations of model predictions. It follows a
four-step pipeline: modeling causal problems using Structural Causal Models
(SCMs), identifying the causal estimand, estimating effects with metrics like
Average Treatment Effect (ATE), and refuting effect estimates. Its framework is
extensible, with an example that reduces spurious correlations by grounding
explanations in programming language properties. A case study on deep code
generation across interpretability scenarios and various deep learning
architectures demonstrates DoCode's benefits. Results show NCMs' sensitivity to
code syntax changes and their ability to learn certain programming concepts
while minimizing confounding bias. The dissertation also examines associational
interpretability as a foundation, analyzing software information's causal
nature using tools like COMET and TraceXplainer for traceability. It highlights
the need to identify code confounders and offers practical guidelines for
applying causal interpretability to NCMs, contributing to more trustworthy AI
in software engineering.

### 7. [Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering](http://arxiv.org/pdf/2505.15038v1)

Authors: Haiyan Zhao, Xuansheng Wu, Fan Yang, Bo Shen, Ninghao Liu, Mengnan Du

Linear Concept Vectors have proven effective for steering large language
models (LLMs). While existing approaches like linear probing and
difference-in-means derive these vectors from LLM hidden representations,
diverse data introduces noises (i.e., irrelevant features) that challenge
steering robustness. To address this, we propose Sparse Autoencoder-Denoised
Concept Vectors (SDCV), which uses Sparse Autoencoders to filter out noisy
features from hidden representations. When applied to linear probing and
difference-in-means, our method improves their steering success rates. We
validate our noise hypothesis through counterfactual experiments and feature
visualizations.

### 8. [LogiCase: Effective Test Case Generation from Logical Description in Competitive Programming](http://arxiv.org/pdf/2505.15039v1)

Authors: Sicheol Sung, Aditi, Dogyu kim, Yo-Sub Han, Sang-Ki Ko

Automated Test Case Generation (ATCG) is crucial for evaluating software
reliability, particularly in competitive programming where robust algorithm
assessments depend on diverse and accurate test cases. However, existing ATCG
methods often fail to meet complex specifications or generate effective corner
cases, limiting their utility. In this work, we introduce Context-Free Grammars
with Counters (CCFGs), a formalism that captures both syntactic and semantic
structures in input specifications. Using a fine-tuned CodeT5 model, we
translate natural language input specifications into CCFGs, enabling the
systematic generation of high-quality test cases. Experiments on the
CodeContests dataset demonstrate that CCFG-based test cases outperform baseline
methods in identifying incorrect algorithms, achieving significant gains in
validity and effectiveness. Our approach provides a scalable and reliable
grammar-driven framework for enhancing automated competitive programming
evaluations.

### 9. [ChartCards: A Chart-Metadata Generation Framework for Multi-Task Chart Understanding](http://arxiv.org/pdf/2505.15046v2)

Authors: Yifan Wu, Lutao Yan, Leixian Shen, Yinan Mei, Jiannan Wang, Yuyu Luo

The emergence of Multi-modal Large Language Models (MLLMs) presents new
opportunities for chart understanding. However, due to the fine-grained nature
of these tasks, applying MLLMs typically requires large, high-quality datasets
for task-specific fine-tuning, leading to high data collection and training
costs. To address this, we propose ChartCards, a unified chart-metadata
generation framework for multi-task chart understanding. ChartCards
systematically synthesizes various chart information, including data tables,
visualization code, visual elements, and multi-dimensional semantic captions.
By structuring this information into organized metadata, ChartCards enables a
single chart to support multiple downstream tasks, such as text-to-chart
retrieval, chart summarization, chart-to-table conversion, chart description,
and chart question answering. Using ChartCards, we further construct MetaChart,
a large-scale high-quality dataset containing 10,862 data tables, 85K charts,
and 170 K high-quality chart captions. We validate the dataset through
qualitative crowdsourcing evaluations and quantitative fine-tuning experiments
across various chart understanding tasks. Fine-tuning six different models on
MetaChart resulted in an average performance improvement of 5% across all
tasks. The most notable improvements are seen in text-to-chart retrieval and
chart-to-table tasks, with Long-CLIP and Llama 3.2-11B achieving improvements
of 17% and 28%, respectively.

### 10. [PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration](http://arxiv.org/pdf/2505.15047v1)

Authors: Yingming Pu, Tao Lin, Hongyu Chen

Large Language Model (LLM)-based multi-agent systems (MAS) demonstrate
remarkable potential for scientific discovery. Existing approaches, however,
often automate scientific discovery using predefined workflows that lack
rationality constraints. This often leads to aimless hypothesizing and a
failure to consistently link hypotheses with evidence, thereby hindering
systematic uncertainty reduction. Overcoming these limitations fundamentally
requires systematic uncertainty reduction. We introduce \texttt{PiFlow}, an
information-theoretical framework, treating automated scientific discovery as a
structured uncertainty reduction problem guided by principles (e.g., scientific
laws). In evaluations across three distinct scientific domains -- discovering
nanomaterial structures, bio-molecules, and superconductor candidates with
targeted properties -- our method significantly improves discovery efficiency,
reflected by a 73.55\% increase in the Area Under the Curve (AUC) of property
values versus exploration steps, and enhances solution quality by 94.06\%
compared to a vanilla agent system. Overall, \texttt{PiFlow} serves as a
Plug-and-Play method, establishing a novel paradigm shift in highly efficient
automated scientific discovery, paving the way for more robust and accelerated
AI-driven research. Code is publicly available at our
\href{https://github.com/amair-lab/PiFlow}{GitHub}.

### Hardware Architecture

### 1. [WISP: Image Segmentation-Based Whitespace Diagnosis for Optimal Rectilinear Floorplanning](http://arxiv.org/pdf/2505.15271v1)

Authors: Xiaotian Zhao, Zixuan Li, Yichen Cai, Xinfei Guo

The increasing number of rectilinear floorplans in modern chip designs
presents significant challenges for traditional macro placers due to the
additional complexity introduced by blocked corners. Particularly, the widely
adopted wirelength model Half-Perimeter Wirelength (HPWL) struggles to
accurately handle rectilinear boundaries, highlighting the need for additional
objectives tailored to rectilinear floorplan optimization. In this paper, we
identify the necessity for whitespace diagnosis in rectilinear floorplanning,
an aspect often overlooked in past research. We introduce WISP, a novel
framework that analyzes and scores whitespace regions to guide placement
optimization. WISP leverages image segmentation techniques for whitespace
parsing, a lightweight probabilistic model to score whitespace regions based on
macro distribution, a Gaussian Mixture Model (GMM) for whitespace density
scoring and direction-aware macro relocation to iteratively refine macro
placement, reduce wasted whitespace, and enhance design quality. The proposed
diagnostic technique also enables the reclamation of block-level unused area
and its return to the top level, maximizing overall area utilization. When
compared against state-of-the-art academia placer DREAMPlace 4.1, our method
achieves an average improvement of 5.4% in routing wirelength, with a maximum
of 11.4% across widely-used benchmarks. This yields an average of 41.5% and
43.7% improvement in Worst Negative Slack (WNS) and Total Negative Slack (TNS),
respectively. Additionally, WISP recycles an average of 16.2% area at the block
level, contributing to more efficient top-level area distribution.

### 2. [FAV-NSS: An HIL Framework for Accelerating Validation of Automotive Network Security Strategies](http://arxiv.org/pdf/2505.15393v1)

Authors: Changhong Li, Shashwat Khandelwal, Shreejith Shanker

Complex electronic control unit (ECU) architectures, software models and
in-vehicle networks are consistently improving safety and comfort functions in
modern vehicles. However, the extended functionality and increased connectivity
introduce new security risks and vulnerabilities that can be exploited on
legacy automotive networks such as the controller area network (CAN). With the
rising complexity of vehicular systems and attack vectors, the need for a
flexible hardware-in-the-loop (HIL) test fixture that can inject attacks and
validate the performance of countermeasures in near-real-world conditions in
real time is vital. This paper presents an FPGA-based HIL framework tailored
towards validating network security approaches (IDS, IPS) and smart integration
strategies of such capabilities for an automotive CAN bus. FAV-NSS replicates
an actual vehicular system environment with functional ECUs and network
infrastructure on an FPGA, allowing functional validation of IDS/IPS
algorithms, accelerator designs and integration schemes (software task on ECU,
dedicated accelerator). To show the efficacy of FAV-NSS, we evaluate an IDS
accelerator integration problem, both as a traditional coupled accelerator (to
the ECU), and secondly close to the CAN controller (mimicking an extended CAN
controller). We show that the latter strategy can be fully validated by our
framework, which would otherwise require integration of specialised CAN modules
into otherwise standard HIL fixtures with ability to instrument internal
signals for characterising timing performance. The tests demonstrate a
promising latency reduction of 6.3x when compared to the traditional coupled
accelerator. Our case study demonstrates the potential of FAV-NSS for
accelerating the optimisation, integration and verification of smart ECUs and
communication controllers in current and future vehicular systems.

### 3. [Bridging the Gap: Physical PCI Device Integration Into SystemC-TLM Virtual Platforms](http://arxiv.org/pdf/2505.15590v1)

Authors: Nils Bosbach, Rebecca Pelke, Niko Zurstraßen, Jan Henrik Weinstock, Lukas Jünger, Rainer Leupers

In today's technology-driven world, early-stage software development and
testing are crucial. Virtual Platforms (VPs) have become indispensable tools
for this purpose as they serve as a platform to execute and debug the
unmodified target software at an early design stage. With the increasing
complexity of software, especially in areas like Artificial Intelligence (AI)
applications, VPs need to provide high simulation speed to ensure the target
software executes within a reasonable time. Hybrid simulation, which combines
virtual models with real hardware, can improve the performance of VPs. This
paper introduces a novel approach for integrating real Peripheral Component
Interconnect (PCI) devices into SystemC-TLM-2.0-based VPs. The embedded PCI
devices enable high performance, easy integration, and allow introspection for
analysis and optimization. To illustrate the practical application of our
approach, we present a case study where we integrate Google Coral's Edge Tensor
Processing Unit (TPU) into an ARM-based VP. The integration allows efficient
execution of AI workloads, accelerating simulation speeds by up to 480x while
eliminating the need for complex virtual device models. Beyond accelerating
AI-workload execution, our framework enables driver development, regression
testing across architectures, and device communication analysis. Our findings
demonstrate that embedding PCI devices into SystemC simulations significantly
enhances

### 4. [HDLxGraph: Bridging Large Language Models and HDL Repositories via HDL Graph Databases](http://arxiv.org/pdf/2505.15701v1)

Authors: Pingqing Zheng, Jiayin Qin, Fuqi Zhang, Shang Wu, Yu Cao, Caiwen Ding, Yang, Zhao

Large Language Models (LLMs) have demonstrated their potential in hardware
design tasks, such as Hardware Description Language (HDL) generation and
debugging. Yet, their performance in real-world, repository-level HDL projects
with thousands or even tens of thousands of code lines is hindered. To this
end, we propose HDLxGraph, a novel framework that integrates Graph Retrieval
Augmented Generation (Graph RAG) with LLMs, introducing HDL-specific graph
representations by incorporating Abstract Syntax Trees (ASTs) and Data Flow
Graphs (DFGs) to capture both code graph view and hardware graph view.
HDLxGraph utilizes a dual-retrieval mechanism that not only mitigates the
limited recall issues inherent in similarity-based semantic retrieval by
incorporating structural information, but also enhances its extensibility to
various real-world tasks by a task-specific retrieval finetuning. Additionally,
to address the lack of comprehensive HDL search benchmarks, we introduce
HDLSearch, a multi-granularity evaluation dataset derived from real-world
repository-level projects. Experimental results demonstrate that HDLxGraph
significantly improves average search accuracy, debugging efficiency and
completion quality by 12.04%, 12.22% and 5.04% compared to similarity-based
RAG, respectively. The code of HDLxGraph and collected HDLSearch benchmark are
available at https://github.com/Nick-Zheng-Q/HDLxGraph.

### Computational Complexity

### 1. [Group Order Logic](http://arxiv.org/pdf/2505.15359v1)

Authors: Anatole Dahan

We introduce an extension of fixed-point logic ($\mathsf{FP}$) with a
group-order operator ($\mathsf{ord}$), that computes the size of a group
generated by a definable set of permutations. This operation is a
generalization of the rank operator ($\mathsf{rk}$). We show that $\mathsf{FP}
+ \mathsf{ord}$ constitutes a new candidate logic for the class of
polynomial-time computable queries ($\mathsf{P}$). As was the case for
$\mathsf{FP} + \mathsf{rk}$, the model-checking of $\mathsf{FP} + \mathsf{ord}$
formulae is polynomial-time computable. Moreover, the query separating
$\mathsf{FP} + \mathsf{rk}$ from $\mathsf{P}$ exhibited by Lichter in his
recent breakthrough is definable in $\mathsf{FP} + \mathsf{ord}$. Precisely, we
show that $\mathsf{FP} + \mathsf{ord}$ canonizes structures with Abelian
colors, a class of structures which contains Lichter's counter-example. This
proof involves expressing a fragment of the group-theoretic approach to graph
canonization in the logic $\mathsf{FP}+ \mathsf{ord}$.

### Computational Engineering

### 1. [Local-Global Associative Frames for Symmetry-Preserving Crystal Structure Modeling](http://arxiv.org/pdf/2505.15315v1)

Authors: Haowei Hua, Wanyu Lin

Crystal structures are defined by the periodic arrangement of atoms in 3D
space, inherently making them equivariant to SO(3) group. A fundamental
requirement for crystal property prediction is that the model's output should
remain invariant to arbitrary rotational transformations of the input
structure. One promising strategy to achieve this invariance is to align the
given crystal structure into a canonical orientation with appropriately
computed rotations, or called frames. However, existing work either only
considers a global frame or solely relies on more advanced local frames based
on atoms' local structure. A global frame is too coarse to capture the local
structure heterogeneity of the crystal, while local frames may inadvertently
disrupt crystal symmetry, limiting their expressivity. In this work, we revisit
the frame design problem for crystalline materials and propose a novel approach
to construct expressive Symmetry-Preserving Frames, dubbed as SPFrame, for
modeling crystal structures. Specifically, this local-global associative frame
constructs invariant local frames rather than equivariant ones, thereby
preserving the symmetry of the crystal. In parallel, it integrates global
structural information to construct an equivariant global frame to enforce
SO(3) invariance. Extensive experimental results demonstrate that SPFrame
consistently outperforms traditional frame construction techniques and existing
crystal property prediction baselines across multiple benchmark tasks.

### 2. [Toward Open Earth Science as Fast and Accessible as Natural Language](http://arxiv.org/pdf/2505.15690v1)

Authors: Marquita Ellis, Iksha Gurung, Muthukumaran Ramasubramanian, Rahul Ramachandran

Is natural-language-driven earth observation data analysis now feasible with
the assistance of Large Language Models (LLMs)? For open science in service of
public interest, feasibility requires reliably high accuracy, interactive
latencies, low (sustainable) costs, open LLMs, and openly maintainable software
-- hence, the challenge. What are the techniques and programming system
requirements necessary for satisfying these constraints, and what is the
corresponding development and maintenance burden in practice? This study lays
the groundwork for exploring these questions, introducing an impactful earth
science use-case, and providing a software framework with evaluation data and
metrics, along with initial results from employing model scaling,
prompt-optimization, and inference-time scaling optimization techniques. While
we attain high accuracy (near 100%) across 10 of 11 metrics, the analysis
further considers cost (token-spend), latency, and maintainability across this
space of techniques. Finally, we enumerate opportunities for further research,
general programming and evaluation framework development, and ongoing work for
a comprehensive, deployable solution. This is a call for collaboration and
contribution.

### 3. [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](http://arxiv.org/pdf/2505.15155v1)

Authors: Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, Jiang Bian

Financial markets pose fundamental challenges for asset return prediction due
to their high dimensionality, non-stationarity, and persistent volatility.
Despite advances in large language models and multi-agent systems, current
quantitative research pipelines suffer from limited automation, weak
interpretability, and fragmented coordination across key components such as
factor mining and model innovation. In this paper, we propose R&D-Agent for
Quantitative Finance, in short RD-Agent(Q), the first data-centric multi-agent
framework designed to automate the full-stack research and development of
quantitative strategies via coordinated factor-model co-optimization.
RD-Agent(Q) decomposes the quant process into two iterative stages: a Research
stage that dynamically sets goal-aligned prompts, formulates hypotheses based
on domain priors, and maps them to concrete tasks, and a Development stage that
employs a code-generation agent, Co-STEER, to implement task-specific code,
which is then executed in real-market backtests. The two stages are connected
through a feedback stage that thoroughly evaluates experimental outcomes and
informs subsequent iterations, with a multi-armed bandit scheduler for adaptive
direction selection. Empirically, RD-Agent(Q) achieves up to 2X higher
annualized returns than classical factor libraries using 70% fewer factors, and
outperforms state-of-the-art deep time-series models on real markets. Its joint
factor-model optimization delivers a strong balance between predictive accuracy
and strategy robustness. Our code is available at:
https://github.com/microsoft/RD-Agent.

### 4. [Degree-Optimized Cumulative Polynomial Kolmogorov-Arnold Networks](http://arxiv.org/pdf/2505.15228v1)

Authors: Mathew Vanherreweghe, Lirandë Pira, Patrick Rebentrost

We introduce cumulative polynomial Kolmogorov-Arnold networks (CP-KAN), a
neural architecture combining Chebyshev polynomial basis functions and
quadratic unconstrained binary optimization (QUBO). Our primary contribution
involves reformulating the degree selection problem as a QUBO task, reducing
the complexity from $O(D^N)$ to a single optimization step per layer. This
approach enables efficient degree selection across neurons while maintaining
computational tractability. The architecture performs well in regression tasks
with limited data, showing good robustness to input scales and natural
regularization properties from its polynomial basis. Additionally, theoretical
analysis establishes connections between CP-KAN's performance and properties of
financial time series. Our empirical validation across multiple domains
demonstrates competitive performance compared to several traditional
architectures tested, especially in scenarios where data efficiency and
numerical stability are important. Our implementation, including strategies for
managing computational overhead in larger networks is available in
Ref.~\citep{cpkan_implementation}.

### 5. [Quantization of Probability Distributions via Divide-and-Conquer: Convergence and Error Propagation under Distributional Arithmetic Operations](http://arxiv.org/pdf/2505.15283v1)

Authors: Bilgesu Arif Bilgin, Olof Hallqvist Elias, Michael Selby, Phillip Stanley-Marbell

This article studies a general divide-and-conquer algorithm for approximating
continuous one-dimensional probability distributions with finite mean. The
article presents a numerical study that compares pre-existing approximation
schemes with a special focus on the stability of the discrete approximations
when they undergo arithmetic operations. The main results are a simple upper
bound of the approximation error in terms of the Wasserstein-1 distance that is
valid for all continuous distributions with finite mean. In many use-cases, the
studied method achieve optimal rate of convergence, and numerical experiments
show that the algorithm is more stable than pre-existing approximation schemes
in the context of arithmetic operations.

### 6. [Elasto-acoustic wave propagation in geophysical media using hybrid high-order methods on general meshes](http://arxiv.org/pdf/2505.15771v1)

Authors: Romain Mottier, Alexandre Ern, Laurent Guillot

Hybrid high-order (HHO) methods are numerical methods characterized by
several interesting properties such as local conservativity, geometric
flexibility and high-order accuracy. Here, HHO schemes are studied for the
space semi-discretization of coupled elasto-acoustic waves in the time domain
using a first-order formulation. Explicit and singly diagonal implicit
Runge--Kutta (ERK & SDIRK) schemes are used for the time discretization. We
show that an efficient implementation of explicit (resp. implicit) time schemes
calls for a static condensation of the face (resp. cell) unknowns. Crucially,
both static condensation procedures only involve block-diagonal matrices. Then,
we provide numerical estimates for the CFL stability limit of ERK schemes and
present a comparative study on the efficiency of explicit versus implicit
schemes. Our findings indicate that implicit time schemes remain competitive in
many situations. Finally, simulations in a 2D realistic geophysical
configuration are performed, illustrating the geometrical flexibility of the
HHO method: both hybrid (triangular and quadrangular) and nonconforming (with
hanging nodes) meshes are easily handled, delivering results of comparable
accuracy to a reference spectral element software based on tensorized elements.

### Computation and Language

### 1. [Language Specific Knowledge: Do Models Know Better in X than in English?](http://arxiv.org/pdf/2505.14990v1)

Authors: Ishika Agarwal, Nimet Beyza Bozdag, Dilek Hakkani-Tür

Code-switching is a common phenomenon of alternating between different
languages in the same utterance, thought, or conversation. We posit that humans
code-switch because they feel more comfortable talking about certain topics and
domains in one language than another. With the rise of knowledge-intensive
language models, we ask ourselves the next, natural question: Could models hold
more knowledge on some topics in some language X? More importantly, could we
improve reasoning by changing the language that reasoning is performed in? We
coin the term Language Specific Knowledge (LSK) to represent this phenomenon.
As ethnic cultures tend to develop alongside different languages, we employ
culture-specific datasets (that contain knowledge about cultural and social
behavioral norms). We find that language models can perform better when using
chain-of-thought reasoning in some languages other than English, sometimes even
better in low-resource languages. Paired with previous works showing that
semantic similarity does not equate to representational similarity, we
hypothesize that culturally specific texts occur more abundantly in
corresponding languages, enabling specific knowledge to occur only in specific
"expert" languages. Motivated by our initial results, we design a simple
methodology called LSKExtractor to benchmark the language-specific knowledge
present in a language model and, then, exploit it during inference. We show our
results on various models and datasets, showing an average relative improvement
of 10% in accuracy. Our research contributes to the open-source development of
language models that are inclusive and more aligned with the cultural and
linguistic contexts in which they are deployed.

### 2. [Effective and Efficient Schema-aware Information Extraction Using On-Device Large Language Models](http://arxiv.org/pdf/2505.14992v1)

Authors: Zhihao Wen, Sheng Liang, Yaxiong Wu, Yongyue Zhang, Yong Liu

Information extraction (IE) plays a crucial role in natural language
processing (NLP) by converting unstructured text into structured knowledge.
Deploying computationally intensive large language models (LLMs) on
resource-constrained devices for information extraction is challenging,
particularly due to issues like hallucinations, limited context length, and
high latency-especially when handling diverse extraction schemas. To address
these challenges, we propose a two-stage information extraction approach
adapted for on-device LLMs, called Dual-LoRA with Incremental Schema Caching
(DLISC), which enhances both schema identification and schema-aware extraction
in terms of effectiveness and efficiency. In particular, DLISC adopts an
Identification LoRA module for retrieving the most relevant schemas to a given
query, and an Extraction LoRA module for performing information extraction
based on the previously selected schemas. To accelerate extraction inference,
Incremental Schema Caching is incorporated to reduce redundant computation,
substantially improving efficiency. Extensive experiments across multiple
information extraction datasets demonstrate notable improvements in both
effectiveness and efficiency.

### 3. [Towards Spoken Mathematical Reasoning: Benchmarking Speech-based Models over Multi-faceted Math Problems](http://arxiv.org/pdf/2505.15000v1)

Authors: Chengwei Wei, Bin Wang, Jung-jae Kim, Nancy F. Chen

Recent advances in large language models (LLMs) and multimodal LLMs (MLLMs)
have led to strong reasoning ability across a wide range of tasks. However,
their ability to perform mathematical reasoning from spoken input remains
underexplored. Prior studies on speech modality have mostly focused on factual
speech understanding or simple audio reasoning tasks, providing limited insight
into logical step-by-step reasoning, such as that required for mathematical
problem solving. To address this gap, we introduce Spoken Math Question
Answering (Spoken-MQA), a new benchmark designed to evaluate the mathematical
reasoning capabilities of speech-based models, including both cascade models
(ASR + LLMs) and end-to-end speech LLMs. Spoken-MQA covers a diverse set of
math problems, including pure arithmetic, single-step and multi-step contextual
reasoning, and knowledge-oriented reasoning problems, all presented in
unambiguous natural spoken language. Through extensive experiments, we find
that: (1) while some speech LLMs perform competitively on contextual reasoning
tasks involving basic arithmetic, they still struggle with direct arithmetic
problems; (2) current LLMs exhibit a strong bias toward symbolic mathematical
expressions written in LaTex and have difficulty interpreting verbalized
mathematical expressions; and (3) mathematical knowledge reasoning abilities
are significantly degraded in current speech LLMs.

### 4. [Diagnosing our datasets: How does my language model learn clinical information?](http://arxiv.org/pdf/2505.15024v1)

Authors: Furong Jia, David Sontag, Monica Agrawal

Large language models (LLMs) have performed well across various clinical
natural language processing tasks, despite not being directly trained on
electronic health record (EHR) data. In this work, we examine how popular
open-source LLMs learn clinical information from large mined corpora through
two crucial but understudied lenses: (1) their interpretation of clinical
jargon, a foundational ability for understanding real-world clinical notes, and
(2) their responses to unsupported medical claims. For both use cases, we
investigate the frequency of relevant clinical information in their
corresponding pretraining corpora, the relationship between pretraining data
composition and model outputs, and the sources underlying this data. To isolate
clinical jargon understanding, we evaluate LLMs on a new dataset MedLingo.
Unsurprisingly, we find that the frequency of clinical jargon mentions across
major pretraining corpora correlates with model performance. However, jargon
frequently appearing in clinical notes often rarely appears in pretraining
corpora, revealing a mismatch between available data and real-world usage.
Similarly, we find that a non-negligible portion of documents support disputed
claims that can then be parroted by models. Finally, we classified and analyzed
the types of online sources in which clinical jargon and unsupported medical
claims appear, with implications for future dataset composition.

### 5. [Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective](http://arxiv.org/pdf/2505.15045v1)

Authors: Siyue Zhang, Yilun Zhao, Liyuan Geng, Arman Cohan, Anh Tuan Luu, Chen Zhao

Large language model (LLM)-based embedding models, benefiting from large
scale pre-training and post-training, have begun to surpass BERT and T5-based
models on general-purpose text embedding tasks such as document retrieval.
However, a fundamental limitation of LLM embeddings lies in the unidirectional
attention used during autoregressive pre-training, which misaligns with the
bidirectional nature of text embedding tasks. To this end, We propose adopting
diffusion language models for text embeddings, motivated by their inherent
bidirectional architecture and recent success in matching or surpassing LLMs
especially on reasoning tasks. We present the first systematic study of the
diffusion language embedding model, which outperforms the LLM-based embedding
model by 20% on long-document retrieval, 8% on reasoning-intensive retrieval,
2% on instruction-following retrieval, and achieve competitive performance on
traditional text embedding benchmarks. Our analysis verifies that bidirectional
attention is crucial for encoding global context in long and complex text.

### 6. [Improving the fact-checking performance of language models by relying on their entailment ability](http://arxiv.org/pdf/2505.15050v1)

Authors: Gaurav Kumar, Debajyoti Mazumder, Ayush Garg, Jasabanta Patro

Automated fact-checking is a crucial task in this digital age. To verify a
claim, current approaches majorly follow one of two strategies i.e. (i) relying
on embedded knowledge of language models, and (ii) fine-tuning them with
evidence pieces. While the former can make systems to hallucinate, the later
have not been very successful till date. The primary reason behind this is that
fact verification is a complex process. Language models have to parse through
multiple pieces of evidence before making a prediction. Further, the evidence
pieces often contradict each other. This makes the reasoning process even more
complex. We proposed a simple yet effective approach where we relied on
entailment and the generative ability of language models to produce
''supporting'' and ''refuting'' justifications (for the truthfulness of a
claim). We trained language models based on these justifications and achieved
superior results. Apart from that, we did a systematic comparison of different
prompting and fine-tuning strategies, as it is currently lacking in the
literature. Some of our observations are: (i) training language models with raw
evidence sentences registered an improvement up to 8.20% in macro-F1, over the
best performing baseline for the RAW-FC dataset, (ii) similarly, training
language models with prompted claim-evidence understanding (TBE-2) registered
an improvement (with a margin up to 16.39%) over the baselines for the same
dataset, (iii) training language models with entailed justifications (TBE-3)
outperformed the baselines by a huge margin (up to 28.57% and 44.26% for
LIAR-RAW and RAW-FC, respectively). We have shared our code repository to
reproduce the results.

### 7. [Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](http://arxiv.org/pdf/2505.15055v1)

Authors: Hongli Zhou, Hui Huang, Ziqing Zhao, Lvyuan Han, Huicheng Wang, Kehai Chen, Muyun Yang, Wei Bao, Jian Dong, Bing Xu, Conghui Zhu, Hailong Cao, Tiejun Zhao

The evaluation of large language models (LLMs) via benchmarks is widespread,
yet inconsistencies between different leaderboards and poor separability among
top models raise concerns about their ability to accurately reflect authentic
model capabilities. This paper provides a critical analysis of benchmark
effectiveness, examining main-stream prominent LLM benchmarks using results
from diverse models. We first propose a new framework for accurate and reliable
estimations of item characteristics and model abilities. Specifically, we
propose Pseudo-Siamese Network for Item Response Theory (PSN-IRT), an enhanced
Item Response Theory framework that incorporates a rich set of item parameters
within an IRT-grounded architecture. Based on PSN-IRT, we conduct extensive
analysis which reveals significant and varied shortcomings in the measurement
quality of current benchmarks. Furthermore, we demonstrate that leveraging
PSN-IRT is able to construct smaller benchmarks while maintaining stronger
alignment with human preference.

### 8. [UrduFactCheck: An Agentic Fact-Checking Framework for Urdu with Evidence Boosting and Benchmarking](http://arxiv.org/pdf/2505.15063v1)

Authors: Sarfraz Ahmad, Hasan Iqbal, Momina Ahsan, Numaan Naeem, Muhammad Ahsan Riaz Khan, Arham Riaz, Muhammad Arslan Manzoor, Yuxia Wang, Preslav Nakov

The rapid use of large language models (LLMs) has raised critical concerns
regarding the factual reliability of their outputs, especially in low-resource
languages such as Urdu. Existing automated fact-checking solutions
overwhelmingly focus on English, leaving a significant gap for the 200+ million
Urdu speakers worldwide. In this work, we introduce UrduFactCheck, the first
comprehensive, modular fact-checking framework specifically tailored for Urdu.
Our system features a dynamic, multi-strategy evidence retrieval pipeline that
combines monolingual and translation-based approaches to address the scarcity
of high-quality Urdu evidence. We curate and release two new hand-annotated
benchmarks: UrduFactBench for claim verification and UrduFactQA for evaluating
LLM factuality. Extensive experiments demonstrate that UrduFactCheck,
particularly its translation-augmented variants, consistently outperforms
baselines and open-source alternatives on multiple metrics. We further
benchmark twelve state-of-the-art (SOTA) LLMs on factual question answering in
Urdu, highlighting persistent gaps between proprietary and open-source models.
UrduFactCheck's code and datasets are open-sourced and publicly available at
https://github.com/mbzuai-nlp/UrduFactCheck.

### 9. [In-Domain African Languages Translation Using LLMs and Multi-armed Bandits](http://arxiv.org/pdf/2505.15069v1)

Authors: Pratik Rakesh Singh, Kritarth Prasad, Mohammadi Zaki, Pankaj Wasnik

Neural Machine Translation (NMT) systems face significant challenges when
working with low-resource languages, particularly in domain adaptation tasks.
These difficulties arise due to limited training data and suboptimal model
generalization, As a result, selecting an optimal model for translation is
crucial for achieving strong performance on in-domain data, particularly in
scenarios where fine-tuning is not feasible or practical. In this paper, we
investigate strategies for selecting the most suitable NMT model for a given
domain using bandit-based algorithms, including Upper Confidence Bound, Linear
UCB, Neural Linear Bandit, and Thompson Sampling. Our method effectively
addresses the resource constraints by facilitating optimal model selection with
high confidence. We evaluate the approach across three African languages and
domains, demonstrating its robustness and effectiveness in both scenarios where
target data is available and where it is absent.

### 10. [Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](http://arxiv.org/pdf/2505.15071v1)

Authors: Chen Huang, Junkai Luo, Xinzuo Wang, Wenqiang Lei, Jiancheng Lv

The massive user-generated content (UGC) available in Chinese social media is
giving rise to the possibility of studying internet buzzwords. In this paper,
we study if large language models (LLMs) can generate accurate definitions for
these buzzwords based on UGC as examples. Our work serves a threefold
contribution. First, we introduce CHEER, the first dataset of Chinese internet
buzzwords, each annotated with a definition and relevant UGC. Second, we
propose a novel method, called RESS, to effectively steer the comprehending
process of LLMs to produce more accurate buzzword definitions, mirroring the
skills of human language learning. Third, with CHEER, we benchmark the
strengths and weaknesses of various off-the-shelf definition generation methods
and our RESS. Our benchmark demonstrates the effectiveness of RESS while
revealing crucial shared challenges: over-reliance on prior exposure,
underdeveloped inferential abilities, and difficulty identifying high-quality
UGC to facilitate comprehension. We believe our work lays the groundwork for
future advancements in LLM-based definition generation. Our dataset and code
are available at https://github.com/SCUNLP/Buzzword.

### Cryptography and Security

### 1. [PsyScam: A Benchmark for Psychological Techniques in Real-World Scams](http://arxiv.org/pdf/2505.15017v1)

Authors: Shang Ma, Tianyi Ma, Jiahao Liu, Wei Song, Zhenkai Liang, Xusheng Xiao, Yanfang Ye

Online scams have become increasingly prevalent, with scammers using
psychological techniques (PTs) to manipulate victims. While existing research
has developed benchmarks to study scammer behaviors, these benchmarks do not
adequately reflect the PTs observed in real-world scams. To fill this gap, we
introduce PsyScam, a benchmark designed to systematically capture and evaluate
PTs embedded in real-world scam reports. In particular, PsyScam bridges
psychology and real-world cyber security analysis through collecting a wide
range of scam reports from six public platforms and grounding its annotations
in well-established cognitive and psychological theories. We further
demonstrate PsyScam's utility through three downstream tasks: PT
classification, scam completion, and scam augmentation. Experimental results
show that PsyScam presents significant challenges to existing models in both
detecting and generating scam content based on the PTs used by real-world
scammers. Our code and dataset are available at:
https://anonymous.4open.science/r/PsyScam-66E4.

### 2. [A Survey On Secure Machine Learning](http://arxiv.org/pdf/2505.15124v1)

Authors: Taobo Liao, Taoran Li, Prathamesh Nadkarni

In this survey, we will explore the interaction between secure multiparty
computation and the area of machine learning. Recent advances in secure
multiparty computation (MPC) have significantly improved its applicability in
the realm of machine learning (ML), offering robust solutions for
privacy-preserving collaborative learning. This review explores key
contributions that leverage MPC to enable multiple parties to engage in ML
tasks without compromising the privacy of their data. The integration of MPC
with ML frameworks facilitates the training and evaluation of models on
combined datasets from various sources, ensuring that sensitive information
remains encrypted throughout the process. Innovations such as specialized
software frameworks and domain-specific languages streamline the adoption of
MPC in ML, optimizing performance and broadening its usage. These frameworks
address both semi-honest and malicious threat models, incorporating features
such as automated optimizations and cryptographic auditing to ensure compliance
and data integrity. The collective insights from these studies highlight MPC's
potential in fostering collaborative yet confidential data analysis, marking a
significant stride towards the realization of secure and efficient
computational solutions in privacy-sensitive industries. This paper
investigates a spectrum of SecureML libraries that includes cryptographic
protocols, federated learning frameworks, and privacy-preserving algorithms. By
surveying the existing literature, this paper aims to examine the efficacy of
these libraries in preserving data privacy, ensuring model confidentiality, and
fortifying ML systems against adversarial attacks. Additionally, the study
explores an innovative application domain for SecureML techniques: the
integration of these methodologies in gaming environments utilizing ML.

### 3. [Privacy-Preserving Socialized Recommendation based on Multi-View Clustering in a Cloud Environment](http://arxiv.org/pdf/2505.15156v1)

Authors: Cheng Guo, Jing Jia, Peng Wang, Jing Zhang

Recommendation as a service has improved the quality of our lives and plays a
significant role in variant aspects. However, the preference of users may
reveal some sensitive information, so that the protection of privacy is
required. In this paper, we propose a privacy-preserving, socialized,
recommendation protocol that introduces information collected from online
social networks to enhance the quality of the recommendation. The proposed
scheme can calculate the similarity between users to determine their potential
relationships and interests, and it also can protect the users' privacy from
leaking to an untrusted third party. The security analysis and experimental
results showed that our proposed scheme provides excellent performance and is
feasible for real-world applications.

### 4. [Federated Learning-Enhanced Blockchain Framework for Privacy-Preserving Intrusion Detection in Industrial IoT](http://arxiv.org/pdf/2505.15376v1)

Authors: Anas Ali, Mubashar Husain, Peter Hans

Industrial Internet of Things (IIoT) systems have become integral to smart
manufacturing, yet their growing connectivity has also exposed them to
significant cybersecurity threats. Traditional intrusion detection systems
(IDS) often rely on centralized architectures that raise concerns over data
privacy, latency, and single points of failure. In this work, we propose a
novel Federated Learning-Enhanced Blockchain Framework (FL-BCID) for
privacy-preserving intrusion detection tailored for IIoT environments. Our
architecture combines federated learning (FL) to ensure decentralized model
training with blockchain technology to guarantee data integrity, trust, and
tamper resistance across IIoT nodes. We design a lightweight intrusion
detection model collaboratively trained using FL across edge devices without
exposing sensitive data. A smart contract-enabled blockchain system records
model updates and anomaly scores to establish accountability. Experimental
evaluations using the ToN-IoT and N-BaIoT datasets demonstrate the superior
performance of our framework, achieving 97.3% accuracy while reducing
communication overhead by 41% compared to baseline centralized methods. Our
approach ensures privacy, scalability, and robustness-critical for secure
industrial operations. The proposed FL-BCID system provides a promising
solution for enhancing trust and privacy in modern IIoT security architectures.

### 5. [Real-Time Detection of Insider Threats Using Behavioral Analytics and Deep Evidential Clustering](http://arxiv.org/pdf/2505.15383v1)

Authors: Anas Ali, Mubashar Husain, Peter Hans

Insider threats represent one of the most critical challenges in modern
cybersecurity. These threats arise from individuals within an organization who
misuse their legitimate access to harm the organization's assets, data, or
operations. Traditional security mechanisms, primarily designed for external
attackers, fall short in identifying these subtle and context-aware threats. In
this paper, we propose a novel framework for real-time detection of insider
threats using behavioral analytics combined with deep evidential clustering.
Our system captures and analyzes user activities, applies context-rich
behavioral features, and classifies potential threats using a deep evidential
clustering model that estimates both cluster assignment and epistemic
uncertainty. The proposed model dynamically adapts to behavioral changes and
significantly reduces false positives. We evaluate our framework on benchmark
insider threat datasets such as CERT and TWOS, achieving an average detection
accuracy of 94.7% and a 38% reduction in false positives compared to
traditional clustering methods. Our results demonstrate the effectiveness of
integrating uncertainty modeling in threat detection pipelines. This research
provides actionable insights for deploying intelligent, adaptive, and robust
insider threat detection systems across various enterprise environments.

### 6. [Optimal Piecewise-based Mechanism for Collecting Bounded Numerical Data under Local Differential Privacy](http://arxiv.org/pdf/2505.15483v1)

Authors: Ye Zheng, Sumita Mishra, Yidan Hu

Numerical data with bounded domains is a common data type in personal
devices, such as wearable sensors. While the collection of such data is
essential for third-party platforms, it raises significant privacy concerns.
Local differential privacy (LDP) has been shown as a framework providing
provable individual privacy, even when the third-party platform is untrusted.
For numerical data with bounded domains, existing state-of-the-art LDP
mechanisms are piecewise-based mechanisms, which are not optimal, leading to
reduced data utility.
  This paper investigates the optimal design of piecewise-based mechanisms to
maximize data utility under LDP. We demonstrate that existing piecewise-based
mechanisms are heuristic forms of the $3$-piecewise mechanism, which is far
from enough to study optimality. We generalize the $3$-piecewise mechanism to
its most general form, i.e. $m$-piecewise mechanism with no pre-defined form of
each piece. Under this form, we derive the closed-form optimal mechanism by
combining analytical proofs and off-the-shelf optimization solvers. Next, we
extend the generalized piecewise-based mechanism to the circular domain (along
with the classical domain), defined on a cyclic range where the distance
between the two endpoints is zero. By incorporating this property, we design
the optimal mechanism for the circular domain, achieving significantly improved
data utility compared with existing mechanisms.
  Our proposed mechanisms guarantee optimal data utility under LDP among all
generalized piecewise-based mechanisms. We show that they also achieve optimal
data utility in two common applications of LDP: distribution estimation and
mean estimation. Theoretical analyses and experimental evaluations prove and
validate the data utility advantages of our proposed mechanisms.

### 7. [VoteMate: A Decentralized Application for Scalable Electronic Voting on EVM-Based Blockchain](http://arxiv.org/pdf/2505.15797v1)

Authors: Ivan Homoliak, Tomáš Švondr

Voting is a cornerstone of democracy, allowing citizens to express their will
and make collective decisions. With advancing technology, online voting is
gaining popularity as it enables voting from anywhere with Internet access,
eliminating the need for printed ballots or polling stations. However, despite
its benefits, online voting carries significant risks. A single vulnerability
could be exploited to manipulate elections on a large scale. Centralized
systems can be secure but may lack transparency and confidentiality, especially
if those in power manipulate them. Blockchain-based voting offers a
transparent, tamper-resistant alternative with end-to-end verifiability and
strong security. Adding cryptographic layers can also ensure voter
confidentiality.

### 8. [An Empirical Analysis of EOS Blockchain: Architecture, Contract, and Security](http://arxiv.org/pdf/2505.15051v1)

Authors: Haiyang Liu, Yingjie Mao, Xiaoqi Li

With the rapid development of blockchain technology, various blockchain
systems are exhibiting vitality and potential. As a representative of
Blockchain 3.0, the EOS blockchain has been regarded as a strong competitor to
Ethereum. Nevertheless, compared with Bitcoin and Ethereum, academic research
and in-depth analyses of EOS remain scarce. To address this gap, this study
conducts a comprehensive investigation of the EOS blockchain from five key
dimensions: system architecture, decentralization, performance, smart
contracts, and behavioral security. The architectural analysis focuses on six
core components of the EOS system, detailing their functionalities and
operational workflows. The decentralization and performance evaluations, based
on data from the XBlock data-sharing platform, reveal several critical issues:
low account activity, limited participation in the supernode election process,
minimal variation in the set of block producers, and a substantial gap between
actual throughput and the claimed million-level performance. Five types of
contract vulnerabilities are identified in the smart contract dimension, and
four mainstream vulnerability detection platforms are introduced and
comparatively analyzed. In terms of behavioral security, four real-world
attacks targeting the structural characteristics of EOS are summarized. This
study contributes to the ongoing development of the EOS blockchain and provides
valuable insights for enhancing the security and regulatory mechanisms of
blockchain ecosystems.

### 9. [Dynamic Spectrum Sharing Based on the Rentable NFT Standard ERC4907](http://arxiv.org/pdf/2505.15148v1)

Authors: Litao Ye, Bin Chen, Shrivastava Shivanshu, Chen Sun, Shuo Wang, Siming Feng, Shengli Zhang

Centralized Dynamic Spectrum Sharing (DSS) faces challenges like data
security, high management costs, and limited scalability. To address these
issues, a blockchain-based DSS scheme has been proposed in this paper. First,
we utilize the ERC4907 standard to mint Non-Fungible Spectrum Tokens (NFSTs)
that serve as unique identifiers for spectrum resources and facilitate renting.
Next, we develop a smart contract for NFST auctions, ensuring secure spectrum
transactions through the auction process. Lastly, we create a Web3 spectrum
auction platform where users can access idle spectrum data and participate in
auctions for NFST leases corresponding to the available spectrum. Experimental
results demonstrate that our NFST, designed according to the ERC4907 standard,
effectively meets users' secure and efficient DSS requirements, making it a
feasible solution.

### 10. [Adaptive Plan-Execute Framework for Smart Contract Security Auditing](http://arxiv.org/pdf/2505.15242v2)

Authors: Zhiyuan Wei, Jing Sun, Zijian Zhang, Zhe Hou, Zixiao Zhao

Large Language Models (LLMs) have shown great promise in code analysis and
auditing; however, they still struggle with hallucinations and limited
context-aware reasoning. We introduce SmartAuditFlow, a novel Plan-Execute
framework that enhances smart contract security analysis through dynamic audit
planning and structured execution. Unlike conventional LLM-based auditing
approaches that follow fixed workflows and predefined steps, SmartAuditFlow
dynamically generates and refines audit plans based on the unique
characteristics of each smart contract. It continuously adjusts its auditing
strategy in response to intermediate LLM outputs and newly detected
vulnerabilities, ensuring a more adaptive and precise security assessment. The
framework then executes these plans step by step, applying a structured
reasoning process to enhance vulnerability detection accuracy while minimizing
hallucinations and false positives. To further improve audit precision,
SmartAuditFlow integrates iterative prompt optimization and external knowledge
sources, such as static analysis tools and Retrieval-Augmented Generation
(RAG). This ensures audit decisions are contextually informed and backed by
real-world security knowledge, producing comprehensive security reports.
Extensive evaluations across multiple benchmarks demonstrate that
SmartAuditFlow outperforms existing methods, achieving 100 percent accuracy on
common and critical vulnerabilities, 41.2 percent accuracy for comprehensive
coverage of known smart contract weaknesses in real-world projects, and
successfully identifying all 13 tested CVEs. These results highlight
SmartAuditFlow's scalability, cost-effectiveness, and superior adaptability
over traditional static analysis tools and contemporary LLM-based approaches,
establishing it as a robust solution for automated smart contract auditing.

### Computer Vision and Pattern Recognition

### 1. [Multispectral Detection Transformer with Infrared-Centric Sensor Fusion](http://arxiv.org/pdf/2505.15137v1)

Authors: Seongmin Hwang, Daeyoung Han, Moongu Jeon

Multispectral object detection aims to leverage complementary information
from visible (RGB) and infrared (IR) modalities to enable robust performance
under diverse environmental conditions. In this letter, we propose IC-Fusion, a
multispectral object detector that effectively fuses visible and infrared
features through a lightweight and modalityaware design. Motivated by wavelet
analysis and empirical observations, we find that IR images contain
structurally rich high-frequency information critical for object localization,
while RGB images provide complementary semantic context. To exploit this, we
adopt a compact RGB backbone and design a novel fusion module comprising a
Multi-Scale Feature Distillation (MSFD) block to enhance RGB features and a
three-stage fusion block with Cross-Modal Channel Shuffle Gate (CCSG) and
Cross-Modal Large Kernel Gate (CLKG) to facilitate effective cross-modal
interaction. Experiments on the FLIR and LLVIP benchmarks demonstrate the
effectiveness and efficiency of our IR-centric fusion strategy. Our code is
available at https://github.com/smin-hwang/IC-Fusion.

### 2. [Unified Cross-Modal Attention-Mixer Based Structural-Functional Connectomics Fusion for Neuropsychiatric Disorder Diagnosis](http://arxiv.org/pdf/2505.15139v1)

Authors: Badhan Mazumder, Lei Wu, Vince D. Calhoun, Dong Hye Ye

Gaining insights into the structural and functional mechanisms of the brain
has been a longstanding focus in neuroscience research, particularly in the
context of understanding and treating neuropsychiatric disorders such as
Schizophrenia (SZ). Nevertheless, most of the traditional multimodal deep
learning approaches fail to fully leverage the complementary characteristics of
structural and functional connectomics data to enhance diagnostic performance.
To address this issue, we proposed ConneX, a multimodal fusion method that
integrates cross-attention mechanism and multilayer perceptron (MLP)-Mixer for
refined feature fusion. Modality-specific backbone graph neural networks (GNNs)
were firstly employed to obtain feature representation for each modality. A
unified cross-modal attention network was then introduced to fuse these
embeddings by capturing intra- and inter-modal interactions, while MLP-Mixer
layers refined global and local features, leveraging higher-order dependencies
for end-to-end classification with a multi-head joint loss. Extensive
evaluations demonstrated improved performance on two distinct clinical
datasets, highlighting the robustness of our proposed framework.

### 3. [CineTechBench: A Benchmark for Cinematographic Technique Understanding and Generation](http://arxiv.org/pdf/2505.15145v1)

Authors: Xinran Wang, Songyu Xu, Xiangxuan Shan, Yuxuan Zhang, Muxi Diao, Xueyan Duan, Yanhua Huang, Kongming Liang, Zhanyu Ma

Cinematography is a cornerstone of film production and appreciation, shaping
mood, emotion, and narrative through visual elements such as camera movement,
shot composition, and lighting. Despite recent progress in multimodal large
language models (MLLMs) and video generation models, the capacity of current
models to grasp and reproduce cinematographic techniques remains largely
uncharted, hindered by the scarcity of expert-annotated data. To bridge this
gap, we present CineTechBench, a pioneering benchmark founded on precise,
manual annotation by seasoned cinematography experts across key cinematography
dimensions. Our benchmark covers seven essential aspects-shot scale, shot
angle, composition, camera movement, lighting, color, and focal length-and
includes over 600 annotated movie images and 120 movie clips with clear
cinematographic techniques. For the understanding task, we design question
answer pairs and annotated descriptions to assess MLLMs' ability to interpret
and explain cinematographic techniques. For the generation task, we assess
advanced video generation models on their capacity to reconstruct
cinema-quality camera movements given conditions such as textual prompts or
keyframes. We conduct a large-scale evaluation on 15+ MLLMs and 5+ video
generation models. Our results offer insights into the limitations of current
models and future directions for cinematography understanding and generation in
automatically film production and appreciation. The code and benchmark can be
accessed at https://github.com/PRIS-CV/CineTechBench.

### 4. [From Pixels to Images: Deep Learning Advances in Remote Sensing Image Semantic Segmentation](http://arxiv.org/pdf/2505.15147v1)

Authors: Quanwei Liu, Tao Huang, Yanni Dong, Jiaqi Yang, Wei Xiang

Remote sensing images (RSIs) capture both natural and human-induced changes
on the Earth's surface, serving as essential data for environmental monitoring,
urban planning, and resource management. Semantic segmentation (SS) of RSIs
enables the fine-grained interpretation of surface features, making it a
critical task in remote sensing analysis. With the increasing diversity and
volume of RSIs collected by sensors on various platforms, traditional
processing methods struggle to maintain efficiency and accuracy. In response,
deep learning (DL) has emerged as a transformative approach, enabling
substantial advances in remote sensing image semantic segmentation (RSISS) by
automating feature extraction and improving segmentation accuracy across
diverse modalities. This paper revisits the evolution of DL-based RSISS by
categorizing existing approaches into four stages: the early pixel-based
methods, the prevailing patch-based and tile-based techniques, and the emerging
image-based strategies enabled by foundation models. We analyze these
developments from the perspective of feature extraction and learning
strategies, revealing the field's progression from pixel-level to tile-level
and from unimodal to multimodal segmentation. Furthermore, we conduct a
comprehensive evaluation of nearly 40 advanced techniques on a unified dataset
to quantitatively characterize their performance and applicability. This review
offers a holistic view of DL-based SS for RS, highlighting key advancements,
comparative insights, and open challenges to guide future research.

### 5. [Lossless Token Merging Even Without Fine-Tuning in Vision Transformers](http://arxiv.org/pdf/2505.15160v1)

Authors: Jaeyeon Lee, Dong-Wan Choi

Although Vision Transformers (ViTs) have become the standard architecture in
computer vision, their massive sizes lead to significant computational
overhead. Token compression techniques have attracted considerable attention to
address this issue, but they often suffer from severe information loss,
requiring extensive additional training to achieve practical performance. In
this paper, we propose Adaptive Token Merging (ATM), a novel method that
ensures lossless token merging, eliminating the need for fine-tuning while
maintaining competitive performance. ATM adaptively reduces tokens across
layers and batches by carefully adjusting layer-specific similarity thresholds,
thereby preventing the undesirable merging of less similar tokens with respect
to each layer. Furthermore, ATM introduces a novel token matching technique
that considers not only similarity but also merging sizes, particularly for the
final layers, to minimize the information loss incurred from each merging
operation. We empirically validate our method across a wide range of pretrained
models, demonstrating that ATM not only outperforms all existing training-free
methods but also surpasses most training-intensive approaches, even without
additional training. Remarkably, training-free ATM achieves over a 30%
reduction in FLOPs for the DeiT-T and DeiT-S models without any drop in their
original accuracy.

### 6. [Harnessing Caption Detailness for Data-Efficient Text-to-Image Generation](http://arxiv.org/pdf/2505.15172v1)

Authors: Xinran Wang, Muxi Diao, Yuanzhi Liu, Chunyu Wang, Kongming Liang, Zhanyu Ma, Jun Guo

Training text-to-image (T2I) models with detailed captions can significantly
improve their generation quality. Existing methods often rely on simplistic
metrics like caption length to represent the detailness of the caption in the
T2I training set. In this paper, we propose a new metric to estimate caption
detailness based on two aspects: image coverage rate (ICR), which evaluates
whether the caption covers all regions/objects in the image, and average object
detailness (AOD), which quantifies the detailness of each object's description.
Through experiments on the COCO dataset using ShareGPT4V captions, we
demonstrate that T2I models trained on high-ICR and -AOD captions achieve
superior performance on DPG and other benchmarks. Notably, our metric enables
more effective data selection-training on only 20% of full data surpasses both
full-dataset training and length-based selection method, improving alignment
and reconstruction ability. These findings highlight the critical role of
detail-aware metrics over length-based heuristics in caption selection for T2I
tasks.

### 7. [Exploring Generalized Gait Recognition: Reducing Redundancy and Noise within Indoor and Outdoor Datasets](http://arxiv.org/pdf/2505.15176v1)

Authors: Qian Zhou, Xianda Guo, Jilong Wang, Chuanfu Shen, Zhongyuan Wang, Hua Zou, Qin Zou, Chao Liang, Chen Long, Gang Wu

Generalized gait recognition, which aims to achieve robust performance across
diverse domains, remains a challenging problem due to severe domain shifts in
viewpoints, appearances, and environments. While mixed-dataset training is
widely used to enhance generalization, it introduces new obstacles including
inter-dataset optimization conflicts and redundant or noisy samples, both of
which hinder effective representation learning. To address these challenges, we
propose a unified framework that systematically improves cross-domain gait
recognition. First, we design a disentangled triplet loss that isolates
supervision signals across datasets, mitigating gradient conflicts during
optimization. Second, we introduce a targeted dataset distillation strategy
that filters out the least informative 20\% of training samples based on
feature redundancy and prediction uncertainty, enhancing data efficiency.
Extensive experiments on CASIA-B, OU-MVLP, Gait3D, and GREW demonstrate that
our method significantly improves cross-dataset recognition for both GaitBase
and DeepGaitV2 backbones, without sacrificing source-domain accuracy. Code will
be released at https://github.com/li1er3/Generalized_Gait.

### 8. [AuxDet: Auxiliary Metadata Matters for Omni-Domain Infrared Small Target Detection](http://arxiv.org/pdf/2505.15184v1)

Authors: Yangting Shi, Renjie He, Le Hui, Xiang Li, Jian Yang, Ming-Ming Cheng, Yimian Dai

Omni-domain infrared small target detection (IRSTD) poses formidable
challenges, as a single model must seamlessly adapt to diverse imaging systems,
varying resolutions, and multiple spectral bands simultaneously. Current
approaches predominantly rely on visual-only modeling paradigms that not only
struggle with complex background interference and inherently scarce target
features, but also exhibit limited generalization capabilities across complex
omni-scene environments where significant domain shifts and appearance
variations occur. In this work, we reveal a critical oversight in existing
paradigms: the neglect of readily available auxiliary metadata describing
imaging parameters and acquisition conditions, such as spectral bands, sensor
platforms, resolution, and observation perspectives. To address this
limitation, we propose the Auxiliary Metadata Driven Infrared Small Target
Detector (AuxDet), a novel multi-modal framework that fundamentally reimagines
the IRSTD paradigm by incorporating textual metadata for scene-aware
optimization. Through a high-dimensional fusion module based on multi-layer
perceptrons (MLPs), AuxDet dynamically integrates metadata semantics with
visual features, guiding adaptive representation learning for each individual
sample. Additionally, we design a lightweight prior-initialized enhancement
module using 1D convolutional blocks to further refine fused features and
recover fine-grained target cues. Extensive experiments on the challenging
WideIRSTD-Full benchmark demonstrate that AuxDet consistently outperforms
state-of-the-art methods, validating the critical role of auxiliary information
in improving robustness and accuracy in omni-domain IRSTD tasks. Code is
available at https://github.com/GrokCV/AuxDet.

### 9. [MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models](http://arxiv.org/pdf/2505.15185v1)

Authors: Yifan Liu, Keyu Fan, Weihao Yu, Chenxin Li, Hao Lu, Yixuan Yuan

Recent advances in generalizable 3D Gaussian Splatting have demonstrated
promising results in real-time high-fidelity rendering without per-scene
optimization, yet existing approaches still struggle to handle unfamiliar
visual content during inference on novel scenes due to limited
generalizability. To address this challenge, we introduce MonoSplat, a novel
framework that leverages rich visual priors from pre-trained monocular depth
foundation models for robust Gaussian reconstruction. Our approach consists of
two key components: a Mono-Multi Feature Adapter that transforms monocular
features into multi-view representations, coupled with an Integrated Gaussian
Prediction module that effectively fuses both feature types for precise
Gaussian generation. Through the Adapter's lightweight attention mechanism,
features are seamlessly aligned and aggregated across views while preserving
valuable monocular priors, enabling the Prediction module to generate Gaussian
primitives with accurate geometry and appearance. Through extensive experiments
on diverse real-world datasets, we convincingly demonstrate that MonoSplat
achieves superior reconstruction quality and generalization capability compared
to existing methods while maintaining computational efficiency with minimal
trainable parameters. Codes are available at
https://github.com/CUHK-AIM-Group/MonoSplat.

### 10. [Geometrically Regularized Transfer Learning with On-Manifold and Off-Manifold Perturbation](http://arxiv.org/pdf/2505.15191v1)

Authors: Hana Satou, Alan Mitkiy, F Monkey

Transfer learning under domain shift remains a fundamental challenge due to
the divergence between source and target data manifolds. In this paper, we
propose MAADA (Manifold-Aware Adversarial Data Augmentation), a novel framework
that decomposes adversarial perturbations into on-manifold and off-manifold
components to simultaneously capture semantic variation and model brittleness.
We theoretically demonstrate that enforcing on-manifold consistency reduces
hypothesis complexity and improves generalization, while off-manifold
regularization smooths decision boundaries in low-density regions. Moreover, we
introduce a geometry-aware alignment loss that minimizes geodesic discrepancy
between source and target manifolds. Experiments on DomainNet, VisDA, and
Office-Home show that MAADA consistently outperforms existing adversarial and
adaptation methods in both unsupervised and few-shot settings, demonstrating
superior structural robustness and cross-domain generalization.

### Computers and Society

### 1. [Lawful but Awful: Evolving Legislative Responses to Address Online Misinformation, Disinformation, and Mal-Information in the Age of Generative AI](http://arxiv.org/pdf/2505.15067v1)

Authors: Simon Chesterman

"Fake news" is an old problem. In recent years, however, increasing usage of
social media as a source of information, the spread of unverified medical
advice during the Covid-19 pandemic, and the rise of generative artificial
intelligence have seen a rush of legislative proposals seeking to minimize or
mitigate the impact of false information spread online. Drawing on a novel
dataset of statutes and other instruments, this article analyses changing
perceptions about the potential harms caused by misinformation, disinformation,
and "mal-information". The turn to legislation began in countries that were
less free, in terms of civil liberties, and poorer, as measured by GDP per
capita. Internet penetration does not seem to have been a driving factor. The
focus of such laws is most frequently on national security broadly construed,
though 2020 saw a spike in laws addressing public health. Unsurprisingly,
governments with fewer legal constraints on government action have generally
adopted more robust positions in dealing with false information. Despite early
reservations, however, growth in such laws is now steepest in Western states.
Though there are diverse views on the appropriate response to false information
online, the need for legislation of some kind appears now to be global. The
question is no longer whether to regulate "lawful but awful" speech online, but
how.

### 2. [Classifying and Tracking International Aid Contribution Towards SDGs](http://arxiv.org/pdf/2505.15223v1)

Authors: Sungwon Park, Dongjoon Lee, Kyeongjin Ahn, Yubin Choi, Junho Lee, Meeyoung Cha, Kyung Ryul Park

International aid is a critical mechanism for promoting economic growth and
well-being in developing nations, supporting progress toward the Sustainable
Development Goals (SDGs). However, tracking aid contributions remains
challenging due to labor-intensive data management, incomplete records, and the
heterogeneous nature of aid data. Recognizing the urgency of this challenge, we
partnered with government agencies to develop an AI model that complements
manual classification and mitigates human bias in subjective interpretation. By
integrating SDG-specific semantics and leveraging prior knowledge from language
models, our approach enhances classification accuracy and accommodates the
diversity of aid projects. When applied to a comprehensive dataset spanning
multiple years, our model can reveal hidden trends in the temporal evolution of
international development cooperation. Expert interviews further suggest how
these insights can empower policymakers with data-driven decision-making tools,
ultimately improving aid effectiveness and supporting progress toward SDGs.

### 3. [The Agentic Economy](http://arxiv.org/pdf/2505.15799v1)

Authors: David M. Rothschild, Markus Mobius, Jake M. Hofman, Eleanor W. Dillon, Daniel G. Goldstein, Nicole Immorlica, Sonia Jaffe, Brendan Lucier, Aleksandrs Slivkins, Matthew Vogel

Generative AI has transformed human-computer interaction by enabling natural
language interfaces and the emergence of autonomous agents capable of acting on
users' behalf. While early applications have improved individual productivity,
these gains have largely been confined to predefined tasks within existing
workflows. We argue that the more profound economic impact lies in reducing
communication frictions between consumers and businesses. This shift could
reorganize markets, redistribute power, and catalyze the creation of new
products and services. We explore the implications of an agentic economy, where
assistant agents act on behalf of consumers and service agents represent
businesses, interacting programmatically to facilitate transactions. A key
distinction we draw is between unscripted interactions -- enabled by technical
advances in natural language and protocol design -- and unrestricted
interactions, which depend on market structures and governance. We examine the
current limitations of siloed and end-to-end agents, and explore future
scenarios shaped by technical standards and market dynamics. These include the
potential tension between agentic walled gardens and an open web of agents,
implications for advertising and discovery, the evolution of
micro-transactions, and the unbundling and rebundling of digital goods.
Ultimately, we argue that the architecture of agentic communication will
determine the extent to which generative AI democratizes access to economic
opportunity.

### 4. [Enabling the Reuse of Personal Data in Research: A Classification Model for Legal Compliance](http://arxiv.org/pdf/2505.15183v1)

Authors: Eduard Mata i Noguera, Ruben Ortiz Uroz, Ignasi Labastida i Juan

Inspired by a proposal made almost ten years ago, this paper presents a model
for classifying per-sonal data for research to inform researchers on how to
manage them. The classification is based on the principles of the European
General Data Protection Regulation and its implementation under the Spanish
Law. The paper also describes in which conditions personal data may be stored
and can be accessed ensuring compliance with data protection regulations and
safeguarding privacy. The work has been developed collaboratively by the
Library and the Data Protection Office. The outcomes of this collaboration are
a decision tree for researchers and a list of requirements for research data
re-positories to store and grant access to personal data securely. This
proposal is aligned with the FAIR principles and the commitment for responsible
open science practices.

### 5. [Multilingual Prompting for Improving LLM Generation Diversity](http://arxiv.org/pdf/2505.15229v1)

Authors: Qihan Wang, Shidong Pan, Tal Linzen, Emily Black

Large Language Models (LLMs) are known to lack cultural representation and
overall diversity in their generations, from expressing opinions to answering
factual questions. To mitigate this problem, we propose multilingual prompting:
a prompting method which generates several variations of a base prompt with
added cultural and linguistic cues from several cultures, generates responses,
and then combines the results. Building on evidence that LLMs have
language-specific knowledge, multilingual prompting seeks to increase diversity
by activating a broader range of cultural knowledge embedded in model training
data. Through experiments across multiple models (GPT-4o, GPT-4o-mini, LLaMA
70B, and LLaMA 8B), we show that multilingual prompting consistently
outperforms existing diversity-enhancing techniques such as high-temperature
sampling, step-by-step recall, and personas prompting. Further analyses show
that the benefits of multilingual prompting vary with language resource level
and model size, and that aligning the prompting language with the cultural cues
reduces hallucination about culturally-specific information.

### 6. [A Participatory Strategy for AI Ethics in Education and Rehabilitation grounded in the Capability Approach](http://arxiv.org/pdf/2505.15466v1)

Authors: Valeria Cesaroni, Eleonora Pasqua, Piercosma Bisconti, Martina Galletti

AI-based technologies have significant potential to enhance inclusive
education and clinical-rehabilitative contexts for children with Special
Educational Needs and Disabilities. AI can enhance learning experiences,
empower students, and support both teachers and rehabilitators. However, their
usage presents challenges that require a systemic-ecological vision, ethical
considerations, and participatory research. Therefore, research and
technological development must be rooted in a strong ethical-theoretical
framework. The Capability Approach - a theoretical model of disability, human
vulnerability, and inclusion - offers a more relevant perspective on
functionality, effectiveness, and technological adequacy in inclusive learning
environments. In this paper, we propose a participatory research strategy with
different stakeholders through a case study on the ARTIS Project, which
develops an AI-enriched interface to support children with text comprehension
difficulties. Our research strategy integrates ethical, educational, clinical,
and technological expertise in designing and implementing AI-based technologies
for children's learning environments through focus groups and collaborative
design sessions. We believe that this holistic approach to AI adoption in
education can help bridge the gap between technological innovation and ethical
responsibility.

### 7. [The Pursuit of Empathy: Evaluating Small Language Models for PTSD Dialogue Support](http://arxiv.org/pdf/2505.15065v1)

Authors: Suhas BN, Yash Mahajan, Dominik Mattioli, Andrew M. Sherrill, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah

Can small language models with 0.5B to 5B parameters meaningfully engage in
trauma-informed, empathetic dialogue for individuals with PTSD? We address this
question by introducing TIDE, a dataset of 10,000 two-turn dialogues spanning
500 diverse PTSD client personas and grounded in a three-factor empathy model:
emotion recognition, distress normalization, and supportive reflection. All
scenarios and reference responses were reviewed for realism and trauma
sensitivity by a clinical psychologist specializing in PTSD. We evaluate eight
small language models before and after fine-tuning, comparing their outputs to
a frontier model (Claude Sonnet 3.5). Our IRB-approved human evaluation and
automatic metrics show that fine-tuning generally improves perceived empathy,
but gains are highly scenario- and user-dependent, with smaller models facing
an empathy ceiling. Demographic analysis shows older adults value distress
validation and graduate-educated users prefer nuanced replies, while gender
effects are minimal. We highlight the limitations of automatic metrics and the
need for context- and user-aware system design. Our findings, along with the
planned release of TIDE, provide a foundation for building safe,
resource-efficient, and ethically sound empathetic AI to supplement, not
replace, clinical mental health care.

### 8. [Social Bias in Popular Question-Answering Benchmarks](http://arxiv.org/pdf/2505.15553v2)

Authors: Angelie Kraft, Judith Simon, Sonja Schimmler

Question-answering (QA) and reading comprehension (RC) benchmarks are
essential for assessing the capabilities of large language models (LLMs) in
retrieving and reproducing knowledge. However, we demonstrate that popular QA
and RC benchmarks are biased and do not cover questions about different
demographics or regions in a representative way, potentially due to a lack of
diversity of those involved in their creation. We perform a qualitative content
analysis of 30 benchmark papers and a quantitative analysis of 20 respective
benchmark datasets to learn (1) who is involved in the benchmark creation, (2)
how social bias is addressed or prevented, and (3) whether the demographics of
the creators and annotators correspond to particular biases in the content.
Most analyzed benchmark papers provided insufficient information regarding the
stakeholders involved in benchmark creation, particularly the annotators.
Notably, just one of the benchmark papers explicitly reported measures taken to
address social representation issues. Moreover, the data analysis revealed
gender, religion, and geographic biases across a wide range of encyclopedic,
commonsense, and scholarly benchmarks. More transparent and bias-aware QA and
RC benchmark creation practices are needed to facilitate better scrutiny and
incentivize the development of fairer LLMs.

### Databases

### 1. [Maximum Degree-Based Quasi-Clique Search via an Iterative Framework](http://arxiv.org/pdf/2505.15118v1)

Authors: Hongbo Xia, Kaiqiang Yu, Shengxin Liu, Cheng Long, Xun Zhou

Cohesive subgraph mining is a fundamental problem in graph theory with
numerous real-world applications, such as social network analysis and
protein-protein interaction modeling. Among various cohesive subgraphs, the
$\gamma$-quasi-clique is widely studied for its flexibility in requiring each
vertex to connect to at least a $\gamma$ proportion of other vertices in the
subgraph. However, solving the maximum $\gamma$-quasi-clique problem is NP-hard
and further complicated by the lack of the hereditary property, which makes
designing efficient pruning strategies challenging. Existing algorithms, such
as DDA and FastQC, either struggle with scalability or exhibit significant
performance declines for small values of $\gamma$. In this paper, we propose a
novel algorithm, IterQC, which reformulates the maximum $\gamma$-quasi-clique
problem as a series of $k$-plex problems that possess the hereditary property.
IterQC introduces a non-trivial iterative framework and incorporates two key
optimization techniques: (1) the pseudo lower bound (pseudo LB) technique,
which leverages information across iterations to improve the efficiency of
branch-and-bound searches, and (2) the preprocessing technique that reduces
problem size and unnecessary iterations. Extensive experiments demonstrate that
IterQC achieves up to four orders of magnitude speedup and solves significantly
more graph instances compared to state-of-the-art algorithms DDA and FastQC.

### 2. [Enabling the Reuse of Personal Data in Research: A Classification Model for Legal Compliance](http://arxiv.org/pdf/2505.15183v1)

Authors: Eduard Mata i Noguera, Ruben Ortiz Uroz, Ignasi Labastida i Juan

Inspired by a proposal made almost ten years ago, this paper presents a model
for classifying per-sonal data for research to inform researchers on how to
manage them. The classification is based on the principles of the European
General Data Protection Regulation and its implementation under the Spanish
Law. The paper also describes in which conditions personal data may be stored
and can be accessed ensuring compliance with data protection regulations and
safeguarding privacy. The work has been developed collaboratively by the
Library and the Data Protection Office. The outcomes of this collaboration are
a decision tree for researchers and a list of requirements for research data
re-positories to store and grant access to personal data securely. This
proposal is aligned with the FAIR principles and the commitment for responsible
open science practices.

### 3. [MoTime: A Dataset Suite for Multimodal Time Series Forecasting](http://arxiv.org/pdf/2505.15072v1)

Authors: Xin Zhou, Weiqing Wang, Francisco J. Baldán, Wray Buntine, Christoph Bergmeir

While multimodal data sources are increasingly available from real-world
forecasting, most existing research remains on unimodal time series. In this
work, we present MoTime, a suite of multimodal time series forecasting datasets
that pair temporal signals with external modalities such as text, metadata, and
images. Covering diverse domains, MoTime supports structured evaluation of
modality utility under two scenarios: 1) the common forecasting task, where
varying-length history is available, and 2) cold-start forecasting, where no
historical data is available. Experiments show that external modalities can
improve forecasting performance in both scenarios, with particularly strong
benefits for short series in some datasets, though the impact varies depending
on data characteristics. By making datasets and findings publicly available, we
aim to support more comprehensive and realistic benchmarks in future multimodal
time series forecasting research.

### 4. [Robo-DM: Data Management For Large Robot Datasets](http://arxiv.org/pdf/2505.15558v1)

Authors: Kaiyuan Chen, Letian Fu, David Huang, Yanxiang Zhang, Lawrence Yunliang Chen, Huang Huang, Kush Hari, Ashwin Balakrishna, Ted Xiao, Pannag R Sanketi, John Kubiatowicz, Ken Goldberg

Recent results suggest that very large datasets of teleoperated robot
demonstrations can be used to train transformer-based models that have the
potential to generalize to new scenes, robots, and tasks. However, curating,
distributing, and loading large datasets of robot trajectories, which typically
consist of video, textual, and numerical modalities - including streams from
multiple cameras - remains challenging. We propose Robo-DM, an efficient
open-source cloud-based data management toolkit for collecting, sharing, and
learning with robot data. With Robo-DM, robot datasets are stored in a
self-contained format with Extensible Binary Meta Language (EBML). Robo-DM can
significantly reduce the size of robot trajectory data, transfer costs, and
data load time during training. Compared to the RLDS format used in OXE
datasets, Robo-DM's compression saves space by up to 70x (lossy) and 3.5x
(lossless). Robo-DM also accelerates data retrieval by load-balancing video
decoding with memory-mapped decoding caches. Compared to LeRobot, a framework
that also uses lossy video compression, Robo-DM is up to 50x faster when
decoding sequentially. We physically evaluate a model trained by Robo-DM with
lossy compression, a pick-and-place task, and In-Context Robot Transformer.
Robo-DM uses 75x compression of the original dataset and does not suffer
reduction in downstream task accuracy.

### 5. [Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search](http://arxiv.org/pdf/2505.15636v1)

Authors: Yousef Al-Jazzazi, Haya Diwan, Jinrui Gou, Cameron Musco, Christopher Musco, Torsten Suel

Nearest neighbor search is central in machine learning, information
retrieval, and databases. For high-dimensional datasets, graph-based methods
such as HNSW, DiskANN, and NSG have become popular thanks to their empirical
accuracy and efficiency. These methods construct a directed graph over the
dataset and perform beam search on the graph to find nodes close to a given
query. While significant work has focused on practical refinements and
theoretical understanding of graph-based methods, many questions remain. We
propose a new distance-based termination condition for beam search to replace
the commonly used condition based on beam width. We prove that, as long as the
search graph is navigable, our resulting Adaptive Beam Search method is
guaranteed to approximately solve the nearest-neighbor problem, establishing a
connection between navigability and the performance of graph-based search. We
also provide extensive experiments on our new termination condition for both
navigable graphs and approximately navigable graphs used in practice, such as
HNSW and Vamana graphs. We find that Adaptive Beam Search outperforms standard
beam search over a range of recall values, data sets, graph constructions, and
target number of nearest neighbors. It thus provides a simple and practical way
to improve the performance of popular methods.

### Distributed, Parallel, and Cluster Computing

### 1. [COSMIC: Enabling Full-Stack Co-Design and Optimization of Distributed Machine Learning Systems](http://arxiv.org/pdf/2505.15020v1)

Authors: Aditi Raju, Jared Ni, William Won, Changhai Man, Srivatsan Krishnan, Srinivas Sridharan, Amir Yazdanbakhsh, Tushar Krishna, Vijay Janapa Reddi

Large-scale machine learning models necessitate distributed systems, posing
significant design challenges due to the large parameter space across distinct
design stacks. Existing studies often focus on optimizing individual system
aspects in isolation. This work challenges this limitation and introduces
COSMIC, a full-stack distributed machine learning systems environment enabling
end-to-end simulation and agent-based design space exploration. To facilitate
efficient exploration and optimization across the entire stack, we introduce
Parameter Set Architecture-an abstraction concept analogous to the instruction
set architecture-abstracting away configuration complexities of agent-based
search methods. Case studies demonstrate COSMIC's ability to consolidate
parameters across multiple layers of design abstraction, discovering eight
non-obvious high-performance system configurations across four
transformer-based models with up to 175 billion parameters. By optimizing
across the stack, COSMIC full-stack optimization delivers 1.50-48.41x higher
performance compared to the isolated single-stack optimization.

### 2. [Exploring Dynamic Load Balancing Algorithms for Block-Structured Mesh-and-Particle Simulations in AMReX](http://arxiv.org/pdf/2505.15122v1)

Authors: Amitash Nanda, Md Kamal Hossain Chowdhury, Hannah Ross, Kevin Gott

Load balancing is critical for successful large-scale high-performance
computing (HPC) simulations. With modern supercomputers increasing in
complexity and variability, dynamic load balancing is becoming more critical to
use computational resources efficiently. In this study, performed during a
summer collaboration at Lawrence Berkeley National Laboratory, we investigate
various standard dynamic load-balancing algorithms. This includes the time
evaluation of a brute-force solve for application in algorithmic evaluation, as
well as quality and time evaluations of the Knapsack algorithm, an SFC
algorithm, and two novel algorithms: a painter's partition-based SFC algorithm
and a combination Knapsack+SFC methodology-based on hardware topology. The
results suggest Knapsack and painter's partition-based algorithms should be
among the first algorithms evaluated by HPC codes for cases with limited weight
deviation and will perform at least slightly better than AMReX's
percentage-tracking partitioning strategy across most simulations, although
effects diminish as weight variety increases.

### 3. [Enhancing Cloud Task Scheduling Using a Hybrid Particle Swarm and Grey Wolf Optimization Approach](http://arxiv.org/pdf/2505.15171v1)

Authors: Raveena Prasad, Aarush Roy, Suchi Kumari

Assigning tasks efficiently in cloud computing is a challenging problem and
is considered an NP-hard problem. Many researchers have used metaheuristic
algorithms to solve it, but these often struggle to handle dynamic workloads
and explore all possible options effectively. Therefore, this paper presents a
new hybrid method that combines two popular algorithms, Grey Wolf Optimizer
(GWO) and Particle Swarm Optimization (PSO). GWO offers strong global search
capabilities (exploration), while PSO enhances local refinement (exploitation).
The hybrid approach, called HybridPSOGWO, is compared with other existing
methods like MPSOSA, RL-GWO, CCGP, and HybridPSOMinMin, using key performance
indicators such as makespan, throughput, and load balancing. We tested our
approach using both a simulation tool (CloudSim Plus) and real-world data. The
results show that HybridPSOGWO outperforms other methods, with up to 15\%
improvement in makespan and 10\% better throughput, while also distributing
tasks more evenly across virtual machines. Our implementation achieves
consistent convergence within a few iterations, highlighting its potential for
efficient and adaptive cloud scheduling.

### 4. [Hardware-Level QoS Enforcement Features: Technologies, Use Cases, and Research Challenges](http://arxiv.org/pdf/2505.15542v1)

Authors: Oliver Larsson, Thijs Metsch, Cristian Klein, Erik Elmroth

Recent advancements in commodity server processors have enabled dynamic
hardware-based quality-of-service (QoS) enforcement. These features have
gathered increasing interest in research communities due to their versatility
and wide range of applications. Thus, there exists a need to understand how
scholars leverage hardware QoS enforcement in research, understand strengths
and shortcomings, and identify gaps in current state-of-the-art research. This
paper observes relevant publications, presents a novel taxonomy, discusses the
approaches used, and identifies trends. Furthermore, an opportunity is
recognized for QoS enforcement utilization in service-based cloud computing
environments, and open challenges are presented.

### 5. [Parallel Scan on Ascend AI Accelerators](http://arxiv.org/pdf/2505.15112v1)

Authors: Bartłomiej Wróblewski, Gioele Gottardo, Anastasios Zouzias

We design and implement parallel prefix sum (scan) algorithms using Ascend AI
accelerators. Ascend accelerators feature specialized computing units - the
cube units for efficient matrix multiplication and the vector units for
optimized vector operations. A key feature of the proposed scan algorithms is
their extensive use of matrix multiplications and accumulations enabled by the
cube unit. To showcase the effectiveness of these algorithms, we also implement
and evaluate several scan-based operators commonly used in AI workloads,
including sorting, tensor masking, and top-$k$ / top-$p$ sampling.
  Our single-core results demonstrate substantial performance improvements,
with speedups ranging from $5\times$ to $9.6\times$ compared to vector-only
implementations for sufficiently large input lengths. Additionally, we present
a multi-core scan algorithm that fully utilizes both the cube and vector units
of Ascend, reaching up to 37.5% of the theoretical memory bandwidth.
Furthermore, our radix sort implementation, which utilizes matrix
multiplications for its parallel splits, showcases the potential of matrix
engines to enhance complex operations, offering up to $3.3\times$ speedup over
the baseline.

### 6. [Breaking Barriers for Distributed MIS by Faster Degree Reduction](http://arxiv.org/pdf/2505.15652v1)

Authors: Seri Khoury, Aaron Schild

We study the problem of finding a maximal independent set (MIS) in the
standard LOCAL model of distributed computing. Classical algorithms by Luby
[JACM'86] and Alon, Babai, and Itai [JALG'86] find an MIS in $O(\log n)$ rounds
in $n$-node graphs with high probability. Despite decades of research, the
existence of any $o(\log n)$-round algorithm for general graphs remains one of
the major open problems in the field.
  Interestingly, the hard instances for this problem must contain
constant-length cycles. This is because there exists a sublogarithmic-round
algorithm for graphs with super-constant girth; i.e., graphs where the length
of the shortest cycle is $\omega(1)$, as shown by Ghaffari~[SODA'16]. Thus,
resolving this $\approx 40$-year-old open problem requires understanding the
family of graphs that contain $k$-cycles for some constant $k$.
  In this work, we come very close to resolving this $\approx 40$-year-old open
problem by presenting a sublogarithmic-round algorithm for graphs that can
contain $k$-cycles for all $k > 6$. Specifically, our algorithm finds an MIS in
$O\left(\frac{\log \Delta}{\log(\log^* \Delta)} + \mathrm{poly}(\log\log
n)\right)$ rounds, as long as the graph does not contain cycles of length $\leq
6$, where $\Delta$ is the maximum degree of the graph. As a result, we push the
limit on the girth of graphs that admit sublogarithmic-round algorithms from $k
= \omega(1)$ all the way down to a small constant $k=7$. This also implies a
$o(\sqrt{\log n})$ round algorithm for MIS in trees, refuting a conjecture from
the book by Barrenboim and Elkin.

### 7. [Round Elimination via Self-Reduction: Closing Gaps for Distributed Maximal Matching](http://arxiv.org/pdf/2505.15654v1)

Authors: Seri Khoury, Aaron Schild

In this work, we present an $\Omega\left(\min\{\log \Delta, \sqrt{\log
n}\}\right)$ lower bound for Maximal Matching (MM) in $\Delta$-ary trees
against randomized algorithms. By a folklore reduction, the same lower bound
applies to Maximal Independent Set (MIS), albeit not in trees. As a function of
$n$, this is the first advancement in our understanding of the randomized
complexity of the two problems in more than two decades. As a function of
$\Delta$, this shows that the current upper bounds are optimal for a wide range
of $\Delta \in 2^{O(\sqrt{\log n})}$, answering an open question by Balliu,
Brandt, Hirvonen, Olivetti, Rabie, and Suomela [FOCS'19, JACM'21].
  Moreover, our result implies a surprising and counterintuitive separation
between MIS and MM in trees, as it was very recently shown that MIS in trees
can be solved in $o(\sqrt{\log n})$ rounds. While MIS can be used to find an MM
in general graphs, the reduction does not preserve the tree structure when
applied to trees. Our separation shows that this is not an artifact of the
reduction, but a fundamental difference between the two problems in trees. This
also implies that MIS is strictly harder in general graphs compared to trees.

### 8. [A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability](http://arxiv.org/pdf/2505.15683v1)

Authors: Zishuai Zhang, Hainan Zhang, Jiaying Zheng, Ziwei Wang, Yongxin Tong, Jin Dong, Zhiming Zheng

Private data is typically larger and of higher quality than public data,
offering great potential to improve LLM. However, its scattered distribution
across data silos and the high computational demands of LLMs limit their
deployment in federated environments. To address this, the transformer-based
split learning model has emerged, offloading most model parameters to the
server while retaining only the embedding and output layers on clients to
ensure privacy. However, it still faces significant challenges in security,
efficiency, and adaptability: 1) embedding gradients are vulnerable to attacks,
leading to reverse engineering of private data; 2) the autoregressive nature of
LLMs means that federated split learning can only train and infer sequentially,
causing high communication overhead; 3) fixed partition points lack
adaptability to downstream tasks. In this paper, we introduce FL-LLaMA, a
secure, efficient, and adaptive federated split framework based on LLaMA2.
First, we place some input and output blocks on the local client and inject
Gaussian noise into forward-pass hidden states, enabling secure end-to-end
propagation. Second, we employ client-batch and server-hierarchical strategies
to achieve parallel training, along with attention-mask compression and KV
cache mechanisms to accelerate inference, reducing communication costs
effectively. Third, we allow users to dynamically adjust the partition points
for input/output blocks based on specific task requirements and hardware
limitations. Experiments on NLU, summarization and conversational QA tasks show
that FL-LLaMA maintains performance comparable to centralized LLaMA2, and
achieves up to 2x train speedups and 8x inference speedups. Further analysis of
privacy attacks and different partition points also demonstrates the
effectiveness of FL-LLaMA in security and adaptability.

### Digital Libraries

### 1. [A two-stage model for factors influencing citation counts](http://arxiv.org/pdf/2505.15384v1)

Authors: Pablo Dorta-González, Emilio Gómez-Déniz

This work aims to study a count response random variable, the number of
citations of a research paper, affected by some explanatory variables through a
suitable regression model. Due to the fact that the count variable exhibits
substantial variation since the sample variance is larger than the sample mean,
the classical Poisson regression model seems not to be appropriate. We
concentrate attention on the negative binomial regression model, which allows
the variance of each measurement to be a function of its predicted value.
Nevertheless, the process of citations of papers may be divided into two parts.
In the first stage, the paper has no citations, and the second part provides
the intensity of the citations. A hurdle model for separating the documents
with citations and those without citations is considered. The dataset for the
empirical application consisted of 43,190 research papers in the field of
Economics and Business from 2014-2021, obtained from The Lens database.
Citation counts and social attention scores for each article were gathered from
Altmetric database. The main findings indicate that both collaboration and
funding have a positive impact on citation counts and reduce the likelihood of
receiving zero citations. Higher journal impact factors lead to higher citation
counts, while lower peer review ratings lead to fewer citations and a higher
probability of zero citations. Mentions in news, blogs, and social media have
varying but generally limited effects on citation counts. Open access via
repositories (green OA) correlates with higher citation counts and a lower
probability of zero citations. In contrast, OA via the publisher's website
without an explicit open license (bronze OA) is associated with higher citation
counts but also with a higher probability of zero citations.

### 2. [GitHub Repository Complexity Leads to Diminished Web Archive Availability](http://arxiv.org/pdf/2505.15042v1)

Authors: David Calano, Michele C. Weigle, Michael L. Nelson

Software is often developed using versioned controlled software, such as Git,
and hosted on centralized Web hosts, such as GitHub and GitLab. These Web
hosted software repositories are made available to users in the form of
traditional HTML Web pages for each source file and directory, as well as a
presentational home page and various descriptive pages. We examined more than
12,000 Web hosted Git repository project home pages, primarily from GitHub, to
measure how well their presentational components are preserved in the Internet
Archive, as well as the source trees of the collected GitHub repositories to
assess the extent to which their source code has been preserved. We found that
more than 31% of the archived repository home pages examined exhibited some
form of minor page damage and 1.6% exhibited major page damage. We also found
that of the source trees analyzed, less than 5% of their source files were
archived, on average, with the majority of repositories not having source files
saved in the Internet Archive at all. The highest concentration of archived
source files available were those linked directly from repositories' home pages
at a rate of 14.89% across all available repositories and sharply dropping off
at deeper levels of a repository's directory tree.

### Discrete Mathematics

### 1. [Creation of fixed points in block-parallel Boolean automata networks](http://arxiv.org/pdf/2505.15499v1)

Authors: Kévin Perrot, Sylvain Sené, Léah Tapin

In the context of discrete dynamical systems and their applications, fixed
points often have a clear interpretation. This is indeed a central topic of
gene regulatory mechanisms modeled by Boolean automata networks (BANs), where a
xollection of Boolean entities (the automata) update their state depending on
the states of others. Fixed points represent phenotypes such as differentiated
cell types. The interaction graph of a BAN captures the architecture of
dependencies among its automata. A first seminal result is that cycles of
interactions (so called feedbacks) are the engines of dynamical complexity. A
second seminal result is that fixed points are invariant under block-sequential
update schedules, which update the automata following an ordered partition of
the set of automata. In this article we study the ability of block-parallel
update schedules (dual to the latter) to break this fixed point invariance
property, with a focus on the simplest feedback mechanism: the canonical
positive cycle. We quantify numerically the creation of new fixed points, and
provide families of block-parallel update schedules generating exponentially
many fixed points on this elementary structure of interaction.

### 2. [Strong odd colorings in graph classes of bounded expansion](http://arxiv.org/pdf/2505.15288v1)

Authors: Michał Pilipczuk

We prove that for every $d\in \mathbb{N}$ and a graph class of bounded
expansion $\mathscr{C}$, there exists some $c\in \mathbb{N}$ so that every
graph from $\mathscr{C}$ admits a proper coloring with at most $c$ colors
satisfying the following condition: in every ball of radius $d$, every color
appears either zero times or an odd number of times. For $d=1$, this provides a
positive answer to a question raised by Goetze, Klute, Knauer, Parada, Pe\~na,
and Ueckerdt [ArXiv 2505.02736] about the boundedness of the strong odd
chromatic number in graph classes of bounded expansion. The key technical
ingredient towards the result is a proof that the strong odd coloring number of
a sets system can be bounded in terms of its semi-ladder index, 2VC dimension,
and the maximum subchromatic number among induced subsystems.

### 3. [Minimum blocking sets for families of partitions](http://arxiv.org/pdf/2505.15362v1)

Authors: Guillermo Gamboa Quintero, Ida Kantor

A $3$-partition of an $n$-element set $V$ is a triple of pairwise disjoint
nonempty subsets $X,Y,Z$ such that $V=X\cup Y\cup Z$. We determine the minimum
size $\varphi_3(n)$ of a set $\mathcal{E}$ of triples such that for every
3-partition $X,Y,Z$ of the set $\{1,\dots,n\}$, there is some $\{x,y,z\}\in
\mathcal{E}$ with $x\in X$, $y\in Y$, and $z\in Z$. In particular,
$$\varphi_3(n)=\left\lceil{\frac{n(n-2)}{3}}\right\rceil.$$ For $d>3$, one may
define an analogous number $\varphi_d(n)$. We determine the order of magnitude
of $\varphi_d(n)$, and prove the following upper and lower bounds, for $d>3$:
$$\frac{2 n^{d-1}}{d!} -o(n^{d-1}) \leq \varphi_d(n) \leq
\frac{0.86}{(d-1)!}n^{d-1}+o(n^{d-1}).$$

### 4. [$4K_1$-free graph with the cop number $3$](http://arxiv.org/pdf/2505.15416v1)

Authors: Arnab Char, Paras Vinubhai Maniya, Dinabandhu Pradhan

The game of cops and robber is a two-player turn-based game played on a graph
where the cops try to capture the robber. The cop number of a graph $G$,
denoted by $c(G)$ is the minimum number of cops required to capture the robber.
For a given class of graphs ${\cal F}$, let $c({\cal F}):=\sup\{c(F)|F\in {\cal
F}\}$, and let Forb$({\cal F})$ denote the class of ${\cal F}$-free graphs. We
show that the complement of the Shrikhande graph is $(4K_1,C_{\ell}$)-free for
any $\ell \geq 6$ and has the cop number~$3$. This provides a counterexample
for the conjecture proposed by Sivaraman (arxiv, 2019) which states that if $G$
is $C_{\ell}$-free for all $\ell\ge 6$, then $c(G)\le 2$. This also gives a
negative answer to the question posed by Turcotte (Discrete Math. 345:112660
(2022)) 112660. to check whether $c($Forb$(pK_1))=p-2$. Turcotte also posed the
question to check whether $c($Forb$(pK_1+K_2))\leq p+1$, for $p\geq 3$. We
prove that this result indeed holds. We also generalize this result for
Forb$(pK_1+qK_2)$. Motivated by the results of Baird et al. (Contrib. Discrete
Math. 9:70--84 (2014)) and Turcotte and Yvon (Discrete Appl. Math. 301:74--98
(2021)), we define the upper threshold degree and lower threshold degree for a
particular class of graphs and show some computational advantage to find the
cop number using these.

### 5. [Families of tractable problems with respect to vertex-interval-membership width and its generalisations](http://arxiv.org/pdf/2505.15699v1)

Authors: Jessica Enright, Samuel D. Hand, Laura Larios-Jones, Kitty Meeks

Temporal graphs are graphs whose edges are labelled with times at which they
are active. Their time-sensitivity provides a useful model of real networks,
but renders many problems studied on temporal graphs more computationally
complex than their static counterparts. To contend with this, there has been
recent work devising parameters for which temporal problems become tractable.
One such parameter is vertex-interval-membership width. Broadly, this gives a
bound on the number of vertices we need to keep track of at any time in order
to solve any of a family of problems. Our contributions are two-fold. Firstly,
we introduce a new parameter, tree-interval-membership-width, that generalises
both vertex-interval-membership-width and several existing generalisations.
Secondly, we provide meta-algorithms for both parameters which can be used to
prove fixed-parameter-tractability for large families of problems, bypassing
the need to give involved dynamic programming arguments for every problem. We
apply these algorithms to temporal versions of Hamiltonian path, matching, edge
deletion to limit maximum reachability, and firefighting.

### 6. [First-order transducibility among classes of sparse graphs](http://arxiv.org/pdf/2505.15655v1)

Authors: Jakub Gajarský, Jeremi Gładkowski, Jan Jedelský, Michał Pilipczuk, Szymon Toruńczyk

We prove several negative results about first-order transducibility for
classes of sparse graphs:
  - for every $t \in \mathbb{N}$, the class of graphs of treewidth at most
$t+1$ is not transducible from the class of graphs of treewidth at most $t$;
  - for every $t \in \mathbb{N}$, the class of graphs with Hadwiger number at
most $t+2$ is not transducible from the class of graphs with Hadwiger number at
most $t$; and
  - the class of graphs of treewidth at most $4$ is not transducible from the
class of planar graphs.
  These results are obtained by combining the known upper and lower bounds on
the weak coloring numbers of the considered graph classes with the following
two new observations:
  - If a weakly sparse graph class $\mathscr D$ is transducible from a class
$\mathscr C$ of bounded expansion, then for some $k \in \mathbb{N}$, every
graph $G \in \mathscr D$ is a $k$-congested depth-$k$ minor of a graph
$H^\circ$ obtained from some $H\in \mathscr C$ by adding a universal vertex.
  - The operations of adding a universal vertex and of taking $k$-congested
depth-$k$ minors, for a fixed $k$, preserve the degree of the distance-$d$ weak
coloring number of a graph class, understood as a polynomial in $d$.

### Data Structures and Algorithms

### 1. [Improved Approximation Algorithms for Path and Forest Augmentation via a Novel Relaxation](http://arxiv.org/pdf/2505.15324v1)

Authors: Felix Hommelsheim

The Forest Augmentation Problem (FAP) asks for a minimum set of additional
edges (links) that make a given forest 2-edge-connected while spanning all
vertices. A key special case is the Path Augmentation Problem (PAP), where the
input forest consists of vertex-disjoint paths. Grandoni, Jabal Ameli, and
Traub [STOC'22] recently broke the long-standing 2-approximation barrier for
FAP, achieving a 1.9973-approximation. A crucial component of this result was
their 1.9913-approximation for PAP; the first better-than-2 approximation for
PAP. In this work, we improve these results and provide a 1.9412-approximation
for PAP, which implies a 1.9955-approximation for FAP. One of our key
innovations is a $(\frac{7}{4} + \varepsilon)$-approximation preserving
reduction to so-called structured instances, which simplifies the problem and
enables our improved approximation. Additionally, we introduce a new relaxation
inspired by 2-edge covers and analyze it via a corresponding packing problem,
where the relationship between the two problems is similar to the relationship
between 2-edge covers and 2-matchings. Using a factor-revealing LP, we bound
the cost of our solution to the packing problem w.r.t. the relaxation and
derive a strong initial solution. We then transform this solution into a
feasible PAP solution, combining techniques from FAP and related connectivity
augmentation problems, along with new insights. A key aspect of our approach is
leveraging the properties of structured PAP instances to achieve our final
approximation guarantee. Our reduction framework and relaxation may be of
independent interest in future work on connectivity augmentation problems.

### 2. [A Faster Algorithm for Independent Cut](http://arxiv.org/pdf/2505.15434v1)

Authors: Vsevolod Chernyshev, Johannes Rauch, Dieter Rautenbach, Liliia Redina

The previously fastest algorithm for deciding the existence of an independent
cut had a runtime of $\mathcal{O}^*(1.4423^n)$, where $n$ is the order of the
input graph. We improve this to $\mathcal{O}^*(1.4143^n)$. In fact, we prove a
runtime of $\mathcal{O}^*\left( 2^{(\frac{1}{2}-\alpha_\Delta)n} \right)$ on
graphs of order $n$ and maximum degree at most $\Delta$, where
$\alpha_\Delta=\frac{1}{2+4\lfloor \frac{\Delta}{2} \rfloor}$. Furthermore, we
show that the problem is fixed-parameter tractable on graphs of order $n$ and
minimum degree at least $\beta n$ for some $\beta > \frac{1}{2}$, where $\beta$
is the parameter.

### 3. [An Efficient Data Structure and Algorithm for Long-Match Query in Run-Length Compressed BWT](http://arxiv.org/pdf/2505.15698v1)

Authors: Ahsan Sanaullah, Degui Zhi, Shaojie Zhang

In this paper, we describe a new type of match between a pattern and a text
that aren't necessarily maximal in the query, but still contain useful matching
information: locally maximal exact matches (LEMs). There are usually a large
amount of LEMs, so we only consider those above some length threshold
$\mathcal{L}$. These are referred to as long LEMs. The purpose of long LEMs is
to capture substring matches between a query and a text that are not
necessarily maximal in the pattern but still long enough to be important.
Therefore efficient long LEMs finding algorithms are desired for these
datasets. However, these datasets are too large to query on traditional string
indexes. Fortunately, these datasets are very repetitive. Recently, compressed
string indexes that take advantage of the redundancy in the data but retain
efficient querying capability have been proposed as a solution. We therefore
give an efficient algorithm for computing all the long LEMs of a query and a
text in a BWT runs compressed string index. We describe an $O(m+occ)$ expected
time algorithm that relies on an $O(r)$ words space string index for outputting
all long LEMs of a pattern with respect to a text given the matching statistics
of the pattern with respect to the text. Here $m$ is the length of the query,
$occ$ is the number of long LEMs outputted, and $r$ is the number of runs in
the BWT of the text. The $O(r)$ space string index we describe relies on an
adaptation of the move data structure by Nishimoto and Tabei. We are able to
support $LCP[i]$ queries in constant time given $SA[i]$. In other words, we
answer $PLCP[i]$ queries in constant time. Long LEMs may provide useful
similarity information between a pattern and a text that MEMs may ignore. This
information is particularly useful in pangenome and biobank scale haplotype
panel contexts.

### 4. [Parallel Scan on Ascend AI Accelerators](http://arxiv.org/pdf/2505.15112v1)

Authors: Bartłomiej Wróblewski, Gioele Gottardo, Anastasios Zouzias

We design and implement parallel prefix sum (scan) algorithms using Ascend AI
accelerators. Ascend accelerators feature specialized computing units - the
cube units for efficient matrix multiplication and the vector units for
optimized vector operations. A key feature of the proposed scan algorithms is
their extensive use of matrix multiplications and accumulations enabled by the
cube unit. To showcase the effectiveness of these algorithms, we also implement
and evaluate several scan-based operators commonly used in AI workloads,
including sorting, tensor masking, and top-$k$ / top-$p$ sampling.
  Our single-core results demonstrate substantial performance improvements,
with speedups ranging from $5\times$ to $9.6\times$ compared to vector-only
implementations for sufficiently large input lengths. Additionally, we present
a multi-core scan algorithm that fully utilizes both the cube and vector units
of Ascend, reaching up to 37.5% of the theoretical memory bandwidth.
Furthermore, our radix sort implementation, which utilizes matrix
multiplications for its parallel splits, showcases the potential of matrix
engines to enhance complex operations, offering up to $3.3\times$ speedup over
the baseline.

### 5. [A Simple Approximation Algorithm for Optimal Decision Tree](http://arxiv.org/pdf/2505.15641v1)

Authors: Zhengjia Zhuo, Viswanath Nagarajan

Optimal decision tree (\odt) is a fundamental problem arising in applications
such as active learning, entity identification, and medical diagnosis. An
instance of \odt is given by $m$ hypotheses, out of which an unknown ``true''
hypothesis is drawn according to some probability distribution. An algorithm
needs to identify the true hypothesis by making queries: each query incurs a
cost and has a known response for each hypothesis. The goal is to minimize the
expected query cost to identify the true hypothesis. We consider the most
general setting with arbitrary costs, probabilities and responses. \odt is
NP-hard to approximate better than $\ln m$ and there are $O(\ln m)$
approximation algorithms known for it. However, these algorithms and/or their
analyses are quite complex. Moreover, the leading constant factors are large.
We provide a simple algorithm and analysis for \odt, proving an approximation
ratio of $8 \ln m$.

### 6. [Learning Small Decision Trees with Few Outliers: A Parameterized Perspective](http://arxiv.org/pdf/2505.15648v1)

Authors: Harmender Gahlawat, Meirav Zehavi

Decision trees are a fundamental tool in machine learning for representing,
classifying, and generalizing data. It is desirable to construct ``small''
decision trees, by minimizing either the \textit{size} ($s$) or the
\textit{depth} $(d)$ of the \textit{decision tree} (\textsc{DT}). Recently, the
parameterized complexity of \textsc{Decision Tree Learning} has attracted a lot
of attention. We consider a generalization of \textsc{Decision Tree Learning}
where given a \textit{classification instance} $E$ and an integer $t$, the task
is to find a ``small'' \textsc{DT} that disagrees with $E$ in at most $t$
examples. We consider two problems: \textsc{DTSO} and \textsc{DTDO}, where the
goal is to construct a \textsc{DT} minimizing $s$ and $d$, respectively. We
first establish that both \textsc{DTSO} and \textsc{DTDO} are W[1]-hard when
parameterized by $s+\delta_{max}$ and $d+\delta_{max}$, respectively, where
$\delta_{max}$ is the maximum number of features in which two differently
labeled examples can differ. We complement this result by showing that these
problems become \textsc{FPT} if we include the parameter $t$. We also consider
the kernelization complexity of these problems and establish several positive
and negative results for both \textsc{DTSO} and \textsc{DTDO}.

### 7. [Breaking Barriers for Distributed MIS by Faster Degree Reduction](http://arxiv.org/pdf/2505.15652v1)

Authors: Seri Khoury, Aaron Schild

We study the problem of finding a maximal independent set (MIS) in the
standard LOCAL model of distributed computing. Classical algorithms by Luby
[JACM'86] and Alon, Babai, and Itai [JALG'86] find an MIS in $O(\log n)$ rounds
in $n$-node graphs with high probability. Despite decades of research, the
existence of any $o(\log n)$-round algorithm for general graphs remains one of
the major open problems in the field.
  Interestingly, the hard instances for this problem must contain
constant-length cycles. This is because there exists a sublogarithmic-round
algorithm for graphs with super-constant girth; i.e., graphs where the length
of the shortest cycle is $\omega(1)$, as shown by Ghaffari~[SODA'16]. Thus,
resolving this $\approx 40$-year-old open problem requires understanding the
family of graphs that contain $k$-cycles for some constant $k$.
  In this work, we come very close to resolving this $\approx 40$-year-old open
problem by presenting a sublogarithmic-round algorithm for graphs that can
contain $k$-cycles for all $k > 6$. Specifically, our algorithm finds an MIS in
$O\left(\frac{\log \Delta}{\log(\log^* \Delta)} + \mathrm{poly}(\log\log
n)\right)$ rounds, as long as the graph does not contain cycles of length $\leq
6$, where $\Delta$ is the maximum degree of the graph. As a result, we push the
limit on the girth of graphs that admit sublogarithmic-round algorithms from $k
= \omega(1)$ all the way down to a small constant $k=7$. This also implies a
$o(\sqrt{\log n})$ round algorithm for MIS in trees, refuting a conjecture from
the book by Barrenboim and Elkin.

### 8. [Round Elimination via Self-Reduction: Closing Gaps for Distributed Maximal Matching](http://arxiv.org/pdf/2505.15654v1)

Authors: Seri Khoury, Aaron Schild

In this work, we present an $\Omega\left(\min\{\log \Delta, \sqrt{\log
n}\}\right)$ lower bound for Maximal Matching (MM) in $\Delta$-ary trees
against randomized algorithms. By a folklore reduction, the same lower bound
applies to Maximal Independent Set (MIS), albeit not in trees. As a function of
$n$, this is the first advancement in our understanding of the randomized
complexity of the two problems in more than two decades. As a function of
$\Delta$, this shows that the current upper bounds are optimal for a wide range
of $\Delta \in 2^{O(\sqrt{\log n})}$, answering an open question by Balliu,
Brandt, Hirvonen, Olivetti, Rabie, and Suomela [FOCS'19, JACM'21].
  Moreover, our result implies a surprising and counterintuitive separation
between MIS and MM in trees, as it was very recently shown that MIS in trees
can be solved in $o(\sqrt{\log n})$ rounds. While MIS can be used to find an MM
in general graphs, the reduction does not preserve the tree structure when
applied to trees. Our separation shows that this is not an artifact of the
reduction, but a fundamental difference between the two problems in trees. This
also implies that MIS is strictly harder in general graphs compared to trees.

### 9. [Group Order Logic](http://arxiv.org/pdf/2505.15359v1)

Authors: Anatole Dahan

We introduce an extension of fixed-point logic ($\mathsf{FP}$) with a
group-order operator ($\mathsf{ord}$), that computes the size of a group
generated by a definable set of permutations. This operation is a
generalization of the rank operator ($\mathsf{rk}$). We show that $\mathsf{FP}
+ \mathsf{ord}$ constitutes a new candidate logic for the class of
polynomial-time computable queries ($\mathsf{P}$). As was the case for
$\mathsf{FP} + \mathsf{rk}$, the model-checking of $\mathsf{FP} + \mathsf{ord}$
formulae is polynomial-time computable. Moreover, the query separating
$\mathsf{FP} + \mathsf{rk}$ from $\mathsf{P}$ exhibited by Lichter in his
recent breakthrough is definable in $\mathsf{FP} + \mathsf{ord}$. Precisely, we
show that $\mathsf{FP} + \mathsf{ord}$ canonizes structures with Abelian
colors, a class of structures which contains Lichter's counter-example. This
proof involves expressing a fragment of the group-theoretic approach to graph
canonization in the logic $\mathsf{FP}+ \mathsf{ord}$.

### 10. [Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search](http://arxiv.org/pdf/2505.15636v1)

Authors: Yousef Al-Jazzazi, Haya Diwan, Jinrui Gou, Cameron Musco, Christopher Musco, Torsten Suel

Nearest neighbor search is central in machine learning, information
retrieval, and databases. For high-dimensional datasets, graph-based methods
such as HNSW, DiskANN, and NSG have become popular thanks to their empirical
accuracy and efficiency. These methods construct a directed graph over the
dataset and perform beam search on the graph to find nodes close to a given
query. While significant work has focused on practical refinements and
theoretical understanding of graph-based methods, many questions remain. We
propose a new distance-based termination condition for beam search to replace
the commonly used condition based on beam width. We prove that, as long as the
search graph is navigable, our resulting Adaptive Beam Search method is
guaranteed to approximately solve the nearest-neighbor problem, establishing a
connection between navigability and the performance of graph-based search. We
also provide extensive experiments on our new termination condition for both
navigable graphs and approximately navigable graphs used in practice, such as
HNSW and Vamana graphs. We find that Adaptive Beam Search outperforms standard
beam search over a range of recall values, data sets, graph constructions, and
target number of nearest neighbors. It thus provides a simple and practical way
to improve the performance of popular methods.

### Emerging Technologies

### 1. [State Characterisation of Self-Directed Channel Memristive Devices](http://arxiv.org/pdf/2505.15757v1)

Authors: Dániel Hajtó, Waleed El-Geresy, Deniz Gündüz, György Cserey

Knowing how to reliably use memristors as information storage devices is
crucial not only to their role as emerging memories, but also for their
application in neural network acceleration and as components of novel
neuromorphic systems. In order to better understand the dynamics of information
storage on memristors, it is essential to be able to characterise and measure
their state. To this end, in this paper we propose a general, physics-inspired
modelling approach for characterising the state of self-directed channel (SDC)
memristors. Additionally, to enable the identification of the proposed state
from device data, we introduce a noise-aware approach to the minimum-variance
estimation of the state from voltage and current pairs.

### 2. [Neural Quantum Digital Twins for Optimizing Quantum Annealing](http://arxiv.org/pdf/2505.15662v1)

Authors: Jianlong Lu, Hanqiu Peng, Ying Chen

Quantum annealers have shown potential in addressing certain combinatorial
optimization problems, though their performance is often limited by scalability
and errors rates. In this work, we propose a Neural Quantum Digital Twin (NQDT)
framework that reconstructs the energy landscape of quantum many-body systems
relevant to quantum annealing. The digital twin models both ground and excited
state dynamics, enabling detailed simulation of the adiabatic evolution
process. We benchmark NQDT on systems with known analytical solutions and
demonstrate that it accurately captures key quantum phenomena, including
quantum criticality and phase transitions. Leveraging this framework, one can
identify optimal annealing schedules that minimize excitation-related errors.
These findings highlight the utility of neural network-based digital twins as a
diagnostic and optimization tool for improving the performance of quantum
annealers.

### Formal Languages and Automata Theory

### 1. [A General Information Extraction Framework Based on Formal Languages](http://arxiv.org/pdf/2505.15605v1)

Authors: Markus L. Schmid

For a terminal alphabet $\Sigma$ and an attribute alphabet $\Gamma$, a
$(\Sigma, \Gamma)$-extractor is a function that maps every string over $\Sigma$
to a table with a column per attribute and with sets of positions of $w$ as
cell entries. This rather general information extraction framework extends the
well-known document spanner framework, which has intensively been investigated
in the database theory community over the last decade. Moreover, our framework
is based on formal language theory in a particularly clean and simple way. In
addition to this conceptual contribution, we investigate closure properties,
different representation formalisms and the complexity of natural decision
problems for extractors.

### 2. [Robust Probabilistic Bisimilarity for Labelled Markov Chains](http://arxiv.org/pdf/2505.15290v1)

Authors: Syyeda Zainab Fatmi, Stefan Kiefer, David Parker, Franck van Breugel

Despite its prevalence, probabilistic bisimilarity suffers from a lack of
robustness under minuscule perturbations of the transition probabilities. This
can lead to discontinuities in the probabilistic bisimilarity distance
function, undermining its reliability in practical applications where
transition probabilities are often approximations derived from experimental
data. Motivated by this limitation, we introduce the notion of robust
probabilistic bisimilarity for labelled Markov chains, which ensures the
continuity of the probabilistic bisimilarity distance function. We also propose
an efficient algorithm for computing robust probabilistic bisimilarity and show
that it performs well in practice, as evidenced by our experimental results.

### 3. [HybridProver: Augmenting Theorem Proving with LLM-Driven Proof Synthesis and Refinement](http://arxiv.org/pdf/2505.15740v1)

Authors: Jilin Hu, Jianyu Zhang, Yongwang Zhao, Talia Ringer

Formal methods is pivotal for verifying the reliability of critical systems
through rigorous mathematical proofs. However, its adoption is hindered by
labor-intensive manual proofs and the expertise required to use theorem
provers. Recent advancements in large language models (LLMs) offer new
opportunities for automated theorem proving. Two promising approaches are
generating tactics step by step and generating a whole proof directly with an
LLM. However, existing work makes no attempt to combine the two approaches. In
this work, we introduce HybridProver, a dual-model proof synthesis framework
that combines tactic-based generation and whole-proof synthesis to harness the
benefits of both approaches. HybridProver generates whole proof candidates for
evaluation directly, then extracts proof sketches from those candidates. It
then uses a tactic-based generation model that integrates automated tools to
complete the sketches via stepwise refinement. We implement HybridProver for
the Isabelle theorem prover and fine-tune LLMs on our optimized Isabelle
datasets. Evaluation on the miniF2F dataset illustrates HybridProver's
effectiveness. We achieve a 59.4% success rate on miniF2F, where the previous
SOTA is 56.1%. Our ablation studies show that this SOTA result is attributable
to combining whole-proof and tactic-based generation. Additionally, we show how
the dataset quality, training parameters, and sampling diversity affect the
final result during automated theorem proving with LLMs. All of our code,
datasets, and LLMs are open source.

### Graphics

### 1. [Building LOD Representation for 3D Urban Scenes](http://arxiv.org/pdf/2505.15190v1)

Authors: Shanshan Pan, Runze Zhang, Yilin Liu, Minglun Gong, Hui Huang

The advances in 3D reconstruction technology, such as photogrammetry and
LiDAR scanning, have made it easier to reconstruct accurate and detailed 3D
models for urban scenes. Nevertheless, these reconstructed models often contain
a large number of geometry primitives, making interactive manipulation and
rendering challenging, especially on resource-constrained devices like virtual
reality platforms. Therefore, the generation of appropriate levels-of-detail
(LOD) representations for these models is crucial. Additionally, automatically
reconstructed 3D models tend to suffer from noise and lack semantic
information. Dealing with these issues and creating LOD representations that
are robust against noise while capturing the semantic meaning present
significant challenges. In this paper, we propose a novel algorithm to address
these challenges. We begin by analysing the properties of planar primitives
detected from the input and group these primitives into multiple level sets by
forming meaningful 3D structures. These level sets form the nodes of our
innovative LOD-Tree. By selecting nodes at appropriate depths within the
LOD-Tree, different LOD representations can be generated. Experimental results
on real and complex urban scenes demonstrate the merits of our approach in
generating clean, accurate, and semantically meaningful LOD representations.

### 2. [EVA: Expressive Virtual Avatars from Multi-view Videos](http://arxiv.org/pdf/2505.15385v1)

Authors: Hendrik Junkawitsch, Guoxing Sun, Heming Zhu, Christian Theobalt, Marc Habermann

With recent advancements in neural rendering and motion capture algorithms,
remarkable progress has been made in photorealistic human avatar modeling,
unlocking immense potential for applications in virtual reality, augmented
reality, remote communication, and industries such as gaming, film, and
medicine. However, existing methods fail to provide complete, faithful, and
expressive control over human avatars due to their entangled representation of
facial expressions and body movements. In this work, we introduce Expressive
Virtual Avatars (EVA), an actor-specific, fully controllable, and expressive
human avatar framework that achieves high-fidelity, lifelike renderings in real
time while enabling independent control of facial expressions, body movements,
and hand gestures. Specifically, our approach designs the human avatar as a
two-layer model: an expressive template geometry layer and a 3D Gaussian
appearance layer. First, we present an expressive template tracking algorithm
that leverages coarse-to-fine optimization to accurately recover body motions,
facial expressions, and non-rigid deformation parameters from multi-view
videos. Next, we propose a novel decoupled 3D Gaussian appearance model
designed to effectively disentangle body and facial appearance. Unlike unified
Gaussian estimation approaches, our method employs two specialized and
independent modules to model the body and face separately. Experimental results
demonstrate that EVA surpasses state-of-the-art methods in terms of rendering
quality and expressiveness, validating its effectiveness in creating full-body
avatars. This work represents a significant advancement towards fully drivable
digital human models, enabling the creation of lifelike digital avatars that
faithfully replicate human geometry and appearance.

### 3. [PlantDreamer: Achieving Realistic 3D Plant Models with Diffusion-Guided Gaussian Splatting](http://arxiv.org/pdf/2505.15528v1)

Authors: Zane K J Hartley, Lewis A G Stuart, Andrew P French, Michael P Pound

Recent years have seen substantial improvements in the ability to generate
synthetic 3D objects using AI. However, generating complex 3D objects, such as
plants, remains a considerable challenge. Current generative 3D models struggle
with plant generation compared to general objects, limiting their usability in
plant analysis tools, which require fine detail and accurate geometry. We
introduce PlantDreamer, a novel approach to 3D synthetic plant generation,
which can achieve greater levels of realism for complex plant geometry and
textures than available text-to-3D models. To achieve this, our new generation
pipeline leverages a depth ControlNet, fine-tuned Low-Rank Adaptation and an
adaptable Gaussian culling algorithm, which directly improve textural realism
and geometric integrity of generated 3D plant models. Additionally,
PlantDreamer enables both purely synthetic plant generation, by leveraging
L-System-generated meshes, and the enhancement of real-world plant point clouds
by converting them into 3D Gaussian Splats. We evaluate our approach by
comparing its outputs with state-of-the-art text-to-3D models, demonstrating
that PlantDreamer outperforms existing methods in producing high-fidelity
synthetic plants. Our results indicate that our approach not only advances
synthetic plant generation, but also facilitates the upgrading of legacy point
cloud datasets, making it a valuable tool for 3D phenotyping applications.

### 4. [Intentional Gesture: Deliver Your Intentions with Gestures for Speech](http://arxiv.org/pdf/2505.15197v1)

Authors: Pinxin Liu, Haiyang Liu, Luchuan Song, Chenliang Xu

When humans speak, gestures help convey communicative intentions, such as
adding emphasis or describing concepts. However, current co-speech gesture
generation methods rely solely on superficial linguistic cues (\textit{e.g.}
speech audio or text transcripts), neglecting to understand and leverage the
communicative intention that underpins human gestures. This results in outputs
that are rhythmically synchronized with speech but are semantically shallow. To
address this gap, we introduce \textbf{Intentional-Gesture}, a novel framework
that casts gesture generation as an intention-reasoning task grounded in
high-level communicative functions. % First, we curate the \textbf{InG} dataset
by augmenting BEAT-2 with gesture-intention annotations (\textit{i.e.}, text
sentences summarizing intentions), which are automatically annotated using
large vision-language models. Next, we introduce the \textbf{Intentional
Gesture Motion Tokenizer} to leverage these intention annotations. It injects
high-level communicative functions (\textit{e.g.}, intentions) into tokenized
motion representations to enable intention-aware gesture synthesis that are
both temporally aligned and semantically meaningful, achieving new
state-of-the-art performance on the BEAT-2 benchmark. Our framework offers a
modular foundation for expressive gesture generation in digital humans and
embodied AI. Project Page: https://andypinxinliu.github.io/Intentional-Gesture

### Computer Science and Game Theory

### 1. [Pointwise Convergence in Games with Conflicting Interest](http://arxiv.org/pdf/2505.15454v1)

Authors: Nanxiang Zhou, Jing Dong, Baoxiang Wang

In this work, we introduce the concept of non-negative weighted regret, an
extension of non-negative regret \cite{anagnostides2022last} in games.
Investigating games with non-negative weighted regret helps us to understand
games with conflicting interests, including harmonic games and important
classes of zero-sum games.We show that optimistic variants of classical
no-regret learning algorithms, namely optimistic mirror descent (OMD) and
optimistic follow the regularized leader (OFTRL), converge to an
$\epsilon$-approximate Nash equilibrium at a rate of
$O(1/\epsilon^2)$.Consequently, they guarantee pointwise convergence to a Nash
equilibrium if there are only finitely many Nash equilibria in the game. These
algorithms are robust in the sense the convergence holds even if the players
deviate Our theoretical findings are supported by empirical evaluations of OMD
and OFTRL on the game of matching pennies and harmonic game instances.

### Human-Computer Interaction

### 1. [Development of Digital Twin Environment through Integration of Commercial Metaverse Platform and IoT Sensors of Smart Building](http://arxiv.org/pdf/2505.15089v1)

Authors: Yusuke Masubuchi, Takefumi Hiraki, Yuichi Hiroi, Masanori Ibara, Kazuki Matsutani, Megumi Zaizen, Junya Morita

The digital transformation of smart cities and workplaces requires effective
integration of physical and cyber spaces, yet existing digital twin solutions
remain limited in supporting real-time, multi-user collaboration. While
metaverse platforms enable shared virtual experiences, they have not supported
comprehensive integration of IoT sensors on physical spaces, especially for
large-scale smart architectural environments. This paper presents a digital
twin environment that integrates Kajima Corp.'s smart building facility "The
GEAR" in Singapore with a commercial metaverse platform Cluster. Our system
consists of three key components: a standardized IoT sensor platform, a
real-time data relay system, and an environmental data visualization framework.
Quantitative end-to-end latency measurements confirm the feasibility of our
approach for real-world applications in large architectural spaces. The
proposed framework enables new forms of collaboration that transcend spatial
constraints, advancing the development of next-generation interactive
environments.

### 2. [AI Solutionism and Digital Self-Tracking with Wearables](http://arxiv.org/pdf/2505.15162v1)

Authors: Hannah R. Nolasco, Andrew Vargo, Koichi Kise

Self-tracking technologies and wearables automate the process of data
collection and insight generation with the support of artificial intelligence
systems, with many emerging studies exploring ways to evolve these features
further through large-language models (LLMs). This is done with the intent to
reduce capture burden and the cognitive stress of health-based decision making,
but studies neglect to consider how automation has stymied the agency and
independent reflection of users of self-tracking interventions. In this
position paper, we explore the consequences of automation in self-tracking by
relating it to our experiences with investigating the Oura Ring, a sleep
wearable, and navigate potential remedies.

### 3. [Stress Bytes: Decoding the Associations between Internet Use and Perceived Stress](http://arxiv.org/pdf/2505.15377v1)

Authors: Mohammad Belal, Nguyen Luong, Talayeh Aledavood, Juhi Kulshrestha

In today's digital era, internet plays a pervasive role in our lives,
influencing everyday activities such as communication, work, and leisure. This
online engagement intertwines with offline experiences, shaping individuals'
overall well-being. Despite its significance, existing research often falls
short in capturing the relationship between internet use and well-being,
relying primarily on isolated studies and self-reported data. One of the major
contributors to deteriorated well-being - both physical and mental - is stress.
While some research has examined the relationship between internet use and
stress, both positive and negative associations have been reported. Our primary
goal in this work is to identify the associations between an individual's
internet use and their stress. For achieving our goal, we conducted a
longitudinal multimodal study that spanned seven months. We combined
fine-grained URL-level web browsing traces of 1490 German internet users with
their sociodemographics and monthly measures of stress. Further, we developed a
conceptual framework that allows us to simultaneously explore different
contextual dimensions, including how, where, when, and by whom the internet is
used. Our analysis revealed several associations between internet use and
stress that vary by context. Social media, entertainment, online shopping, and
gaming were positively associated with stress, while productivity, news, and
adult content use were negatively associated. In the future, the behavioral
markers we identified can pave the way for designing individualized tools for
people to self-monitor and self-moderate their online behaviors to enhance
their well-being, reducing the burden on already overburdened mental health
services.

### 4. [What Is Serendipity? An Interview Study to Conceptualize Experienced Serendipity in Recommender Systems](http://arxiv.org/pdf/2505.15440v1)

Authors: Brett Binst, Lien Michiels, Annelien Smets

Serendipity has been associated with numerous benefits in the context of
recommender systems, e.g., increased user satisfaction and consumption of
long-tail items. Despite this, serendipity in the context of recommender
systems has thus far remained conceptually ambiguous. This conceptual ambiguity
has led to inconsistent operationalizations between studies, making it
difficult to compare and synthesize findings. In this paper, we conceptualize
the user's experience of serendipity. To this effect, we interviewed 17
participants and analyzed the data following the grounded theory paradigm.
Based on these interviews, we conceptualize experienced serendipity as "a user
experience in which a user unintentionally encounters content that feels
fortuitous, refreshing, and enriching". We find that all three components --
fortuitous, refreshing and enriching -- are necessary and together are
sufficient to classify a user's experience as serendipitous. However, these
components can be satisfied through a variety of conditions. Our
conceptualization unifies previous definitions of serendipity within a single
framework, resolving inconsistencies by identifying distinct flavors of
serendipity. It highlights underexposed flavors, offering new insights into how
users experience serendipity in the context of recommender systems. By
clarifying the components and conditions of experienced serendipity in
recommender systems, this work can guide the design of recommender systems that
stimulate experienced serendipity in their users, and lays the groundwork for
developing a standardized operationalization of experienced serendipity in its
many flavors, enabling more consistent and comparable evaluations.

### 5. [Towards a Working Definition of Designing Generative User Interfaces](http://arxiv.org/pdf/2505.15049v1)

Authors: Kyungho Lee

Generative UI is transforming interface design by facilitating AI-driven
collaborative workflows between designers and computational systems. This study
establishes a working definition of Generative UI through a multi-method
qualitative approach, integrating insights from a systematic literature review
of 127 publications, expert interviews with 18 participants, and analyses of 12
case studies. Our findings identify five core themes that position Generative
UI as an iterative and co-creative process. We highlight emerging design
models, including hybrid creation, curation-based workflows, and AI-assisted
refinement strategies. Additionally, we examine ethical challenges, evaluation
criteria, and interaction models that shape the field. By proposing a
conceptual foundation, this study advances both theoretical discourse and
practical implementation, guiding future HCI research toward responsible and
effective generative UI design practices.

### 6. [AI vs. Human Judgment of Content Moderation: LLM-as-a-Judge and Ethics-Based Response Refusals](http://arxiv.org/pdf/2505.15365v1)

Authors: Stefan Pasch

As large language models (LLMs) are increasingly deployed in high-stakes
settings, their ability to refuse ethically sensitive prompts-such as those
involving hate speech or illegal activities-has become central to content
moderation and responsible AI practices. While refusal responses can be viewed
as evidence of ethical alignment and safety-conscious behavior, recent research
suggests that users may perceive them negatively. At the same time, automated
assessments of model outputs are playing a growing role in both evaluation and
training. In particular, LLM-as-a-Judge frameworks-in which one model is used
to evaluate the output of another-are now widely adopted to guide benchmarking
and fine-tuning. This paper examines whether such model-based evaluators assess
refusal responses differently than human users. Drawing on data from Chatbot
Arena and judgments from two AI judges (GPT-4o and Llama 3 70B), we compare how
different types of refusals are rated. We distinguish ethical refusals, which
explicitly cite safety or normative concerns (e.g., "I can't help with that
because it may be harmful"), and technical refusals, which reflect system
limitations (e.g., "I can't answer because I lack real-time data"). We find
that LLM-as-a-Judge systems evaluate ethical refusals significantly more
favorably than human users, a divergence not observed for technical refusals.
We refer to this divergence as a moderation bias-a systematic tendency for
model-based evaluators to reward refusal behaviors more than human users do.
This raises broader questions about transparency, value alignment, and the
normative assumptions embedded in automated evaluation systems.

### 7. [Exploring LLM-Generated Feedback for Economics Essays: How Teaching Assistants Evaluate and Envision Its Use](http://arxiv.org/pdf/2505.15596v1)

Authors: Xinyi Lu, Aditya Mahesh, Zejia Shen, Mitchell Dudley, Larissa Sano, Xu Wang

This project examines the prospect of using AI-generated feedback as
suggestions to expedite and enhance human instructors' feedback provision. In
particular, we focus on understanding the teaching assistants' perspectives on
the quality of AI-generated feedback and how they may or may not utilize AI
feedback in their own workflows. We situate our work in a foundational college
Economics class, which has frequent short essay assignments. We developed an
LLM-powered feedback engine that generates feedback on students' essays based
on grading rubrics used by the teaching assistants (TAs). To ensure that TAs
can meaningfully critique and engage with the AI feedback, we had them complete
their regular grading jobs. For a randomly selected set of essays that they had
graded, we used our feedback engine to generate feedback and displayed the
feedback as in-text comments in a Word document. We then performed think-aloud
studies with 5 TAs over 20 1-hour sessions to have them evaluate the AI
feedback, contrast the AI feedback with their handwritten feedback, and share
how they envision using the AI feedback if they were offered as suggestions.
The study highlights the importance of providing detailed rubrics for AI to
generate high-quality feedback for knowledge-intensive essays. TAs considered
that using AI feedback as suggestions during their grading could expedite
grading, enhance consistency, and improve overall feedback quality. We discuss
the importance of decomposing the feedback generation task into steps and
presenting intermediate results, in order for TAs to use the AI feedback.

### 8. [Exploring the Innovation Opportunities for Pre-trained Models](http://arxiv.org/pdf/2505.15790v1)

Authors: Minjung Park, Jodi Forlizzi, John Zimmerman

Innovators transform the world by understanding where services are
successfully meeting customers' needs and then using this knowledge to identify
failsafe opportunities for innovation. Pre-trained models have changed the AI
innovation landscape, making it faster and easier to create new AI products and
services. Understanding where pre-trained models are successful is critical for
supporting AI innovation. Unfortunately, the hype cycle surrounding pre-trained
models makes it hard to know where AI can really be successful. To address
this, we investigated pre-trained model applications developed by HCI
researchers as a proxy for commercially successful applications. The research
applications demonstrate technical capabilities, address real user needs, and
avoid ethical challenges. Using an artifact analysis approach, we categorized
capabilities, opportunity domains, data types, and emerging interaction design
patterns, uncovering some of the opportunity space for innovation with
pre-trained models.

### 9. [Toward Informed AV Decision-Making: Computational Model of Well-being and Trust in Mobility](http://arxiv.org/pdf/2505.14983v1)

Authors: Zahra Zahedi, Shashank Mehrotra, Teruhisa Misu, Kumar Akash

For future human-autonomous vehicle (AV) interactions to be effective and
smooth, human-aware systems that analyze and align human needs with automation
decisions are essential. Achieving this requires systems that account for human
cognitive states. We present a novel computational model in the form of a
Dynamic Bayesian Network (DBN) that infers the cognitive states of both AV
users and other road users, integrating this information into the AV's
decision-making process. Specifically, our model captures the well-being of
both an AV user and an interacting road user as cognitive states alongside
trust. Our DBN models infer beliefs over the AV user's evolving well-being,
trust, and intention states, as well as the possible well-being of other road
users, based on observed interaction experiences. Using data collected from an
interaction study, we refine the model parameters and empirically assess its
performance. Finally, we extend our model into a causal inference model (CIM)
framework for AV decision-making, enabling the AV to enhance user well-being
and trust while balancing these factors with its own operational costs and the
well-being of interacting road users. Our evaluation demonstrates the model's
effectiveness in accurately predicting user's states and guiding informed,
human-centered AV decisions.

### 10. [Are the confidence scores of reviewers consistent with the review content? Evidence from top conference proceedings in AI](http://arxiv.org/pdf/2505.15031v1)

Authors: Wenqing Wu, Haixu Xi, Chengzhi Zhang

Peer review is vital in academia for evaluating research quality. Top AI
conferences use reviewer confidence scores to ensure review reliability, but
existing studies lack fine-grained analysis of text-score consistency,
potentially missing key details. This work assesses consistency at word,
sentence, and aspect levels using deep learning and NLP conference review data.
We employ deep learning to detect hedge sentences and aspects, then analyze
report length, hedge word/sentence frequency, aspect mentions, and sentiment to
evaluate text-score alignment. Correlation, significance, and regression tests
examine confidence scores' impact on paper outcomes. Results show high
text-score consistency across all levels, with regression revealing higher
confidence scores correlate with paper rejection, validating expert assessments
and peer review fairness.

### Information Retrieval

### 1. [Reranking with Compressed Document Representation](http://arxiv.org/pdf/2505.15394v1)

Authors: Hervé Déjean, Stéphane Clinchant

Reranking, the process of refining the output of a first-stage retriever, is
often considered computationally expensive, especially with Large Language
Models. Borrowing from recent advances in document compression for RAG, we
reduce the input size by compressing documents into fixed-size embedding
representations. We then teach a reranker to use compressed inputs by
distillation. Although based on a billion-size model, our trained reranker
using this compressed input can challenge smaller rerankers in terms of both
effectiveness and efficiency, especially for long documents. Given that text
compressors are still in their early development stages, we view this approach
as promising.

### 2. [CRAFT: Training-Free Cascaded Retrieval for Tabular QA](http://arxiv.org/pdf/2505.14984v1)

Authors: Adarsh Singh, Kushal Raj Bhandari, Jianxi Gao, Soham Dan, Vivek Gupta

Table Question Answering (TQA) involves retrieving relevant tables from a
large corpus to answer natural language queries. Traditional dense retrieval
models, such as DTR and ColBERT, not only incur high computational costs for
large-scale retrieval tasks but also require retraining or fine-tuning on new
datasets, limiting their adaptability to evolving domains and knowledge. In
this work, we propose $\textbf{CRAFT}$, a cascaded retrieval approach that
first uses a sparse retrieval model to filter a subset of candidate tables
before applying more computationally expensive dense models and neural
re-rankers. Our approach achieves better retrieval performance than
state-of-the-art (SOTA) sparse, dense, and hybrid retrievers. We further
enhance table representations by generating table descriptions and titles using
Gemini Flash 1.5. End-to-end TQA results using various Large Language Models
(LLMs) on NQ-Tables, a subset of the Natural Questions Dataset, demonstrate
$\textbf{CRAFT}$ effectiveness.

### 3. [An Alternative to FLOPS Regularization to Effectively Productionize SPLADE-Doc](http://arxiv.org/pdf/2505.15070v1)

Authors: Aldo Porco, Dhruv Mehra, Igor Malioutov, Karthik Radhakrishnan, Moniba Keymanesh, Daniel Preoţiuc-Pietro, Sean MacAvaney, Pengxiang Cheng

Learned Sparse Retrieval (LSR) models encode text as weighted term vectors,
which need to be sparse to leverage inverted index structures during retrieval.
SPLADE, the most popular LSR model, uses FLOPS regularization to encourage
vector sparsity during training. However, FLOPS regularization does not ensure
sparsity among terms - only within a given query or document. Terms with very
high Document Frequencies (DFs) substantially increase latency in production
retrieval engines, such as Apache Solr, due to their lengthy posting lists. To
address the issue of high DFs, we present a new variant of FLOPS
regularization: DF-FLOPS. This new regularization technique penalizes the usage
of high-DF terms, thereby shortening posting lists and reducing retrieval
latency. Unlike other inference-time sparsification methods, such as stopword
removal, DF-FLOPS regularization allows for the selective inclusion of
high-frequency terms in cases where the terms are truly salient. We find that
DF-FLOPS successfully reduces the prevalence of high-DF terms and lowers
retrieval latency (around 10x faster) in a production-grade engine while
maintaining effectiveness both in-domain (only a 2.2-point drop in MRR@10) and
cross-domain (improved performance in 12 out of 13 tasks on which we tested).
With retrieval latencies on par with BM25, this work provides an important step
towards making LSR practical for deployment in production-grade search engines.

### 4. [ThinkRec: Thinking-based recommendation via LLM](http://arxiv.org/pdf/2505.15091v2)

Authors: Qihang Yu, Kairui Fu, Shengyu Zhang, Zheqi Lv, Fan Wu, Fei Wu

Recent advances in large language models (LLMs) have enabled more
semantic-aware recommendations through natural language generation. Existing
LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like
manner, relying on superficial features to match similar items based on click
history, rather than reasoning through deeper behavioral logic. This often
leads to superficial and erroneous recommendations. Motivated by this, we
propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1
to System 2 (rational system). Technically, ThinkRec introduces a thinking
activation mechanism that augments item metadata with keyword summarization and
injects synthetic reasoning traces, guiding the model to form interpretable
reasoning chains that consist of analyzing interaction histories, identifying
user preferences, and making decisions based on target items. On top of this,
we propose an instance-wise expert fusion mechanism to reduce the reasoning
difficulty. By dynamically assigning weights to expert models based on users'
latent features, ThinkRec adapts its reasoning path to individual users,
thereby enhancing precision and personalization. Extensive experiments on
real-world datasets demonstrate that ThinkRec significantly improves the
accuracy and interpretability of recommendations. Our implementations are
available in anonymous Github: https://github.com/Yu-Qi-hang/ThinkRec.

### 5. [Robust Relevance Feedback for Interactive Known-Item Video Search](http://arxiv.org/pdf/2505.15128v1)

Authors: Zhixin Ma, Chong-Wah Ngo

Known-item search (KIS) involves only a single search target, making
relevance feedback-typically a powerful technique for efficiently identifying
multiple positive examples to infer user intent-inapplicable. PicHunter
addresses this issue by asking users to select the top-k most similar examples
to the unique search target from a displayed set. Under ideal conditions, when
the user's perception aligns closely with the machine's perception of
similarity, consistent and precise judgments can elevate the target to the top
position within a few iterations. However, in practical scenarios, expecting
users to provide consistent judgments is often unrealistic, especially when the
underlying embedding features used for similarity measurements lack
interpretability. To enhance robustness, we first introduce a pairwise relative
judgment feedback that improves the stability of top-k selections by mitigating
the impact of misaligned feedback. Then, we decompose user perception into
multiple sub-perceptions, each represented as an independent embedding space.
This approach assumes that users may not consistently align with a single
representation but are more likely to align with one or several among multiple
representations. We develop a predictive user model that estimates the
combination of sub-perceptions based on each user feedback instance. The
predictive user model is then trained to filter out the misaligned
sub-perceptions. Experimental evaluations on the large-scale open-domain
dataset V3C indicate that the proposed model can optimize over 60% search
targets to the top rank when their initial ranks at the search depth between 10
and 50. Even for targets initially ranked between 1,000 and 5,000, the model
achieves a success rate exceeding 40% in optimizing ranks to the top,
demonstrating the enhanced robustness of relevance feedback in KIS despite
inconsistent feedback.

### 6. [Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](http://arxiv.org/pdf/2505.15210v1)

Authors: Jie Ma, Ning Qu, Zhitao Gao, Rui Xing, Jun Liu, Hongbin Pei, Jiang Xie, Linyun Song, Pinghui Wang, Jing Tao, Zhou Su

Knowledge graph-based retrieval-augmented generation seeks to mitigate
hallucinations in Large Language Models (LLMs) caused by insufficient or
outdated knowledge. However, existing methods often fail to fully exploit the
prior knowledge embedded in knowledge graphs (KGs), particularly their
structural information and explicit or implicit constraints. The former can
enhance the faithfulness of LLMs' reasoning, while the latter can improve the
reliability of response generation. Motivated by these, we propose a
trustworthy reasoning framework, termed Deliberation over Priors (DP), which
sufficiently utilizes the priors contained in KGs. Specifically, DP adopts a
progressive knowledge distillation strategy that integrates structural priors
into LLMs through a combination of supervised fine-tuning and Kahneman-Tversky
optimization, thereby improving the faithfulness of relation path generation.
Furthermore, our framework employs a reasoning-introspection strategy, which
guides LLMs to perform refined reasoning verification based on extracted
constraint priors, ensuring the reliability of response generation. Extensive
experiments on three benchmark datasets demonstrate that DP achieves new
state-of-the-art performance, especially a Hit@1 improvement of 13% on the
ComplexWebQuestions dataset, and generates highly trustworthy responses. We
also conduct various analyses to verify its flexibility and practicality. The
code is available at https://github.com/reml-group/Deliberation-on-Priors.

### 7. [Do RAG Systems Suffer From Positional Bias?](http://arxiv.org/pdf/2505.15561v1)

Authors: Florin Cuconasu, Simone Filice, Guy Horowitz, Yoelle Maarek, Fabrizio Silvestri

Retrieval Augmented Generation enhances LLM accuracy by adding passages
retrieved from an external corpus to the LLM prompt. This paper investigates
how positional bias - the tendency of LLMs to weight information differently
based on its position in the prompt - affects not only the LLM's capability to
capitalize on relevant passages, but also its susceptibility to distracting
passages. Through extensive experiments on three benchmarks, we show how
state-of-the-art retrieval pipelines, while attempting to retrieve relevant
passages, systematically bring highly distracting ones to the top ranks, with
over 60% of queries containing at least one highly distracting passage among
the top-10 retrieved passages. As a result, the impact of the LLM positional
bias, which in controlled settings is often reported as very prominent by
related works, is actually marginal in real scenarios since both relevant and
distracting passages are, in turn, penalized. Indeed, our findings reveal that
sophisticated strategies that attempt to rearrange the passages based on LLM
positional preferences do not perform better than random shuffling.

### 8. [ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning](http://arxiv.org/pdf/2505.15776v1)

Authors: Changtai Zhu, Siyin Wang, Ruijun Feng, Kai Song, Xipeng Qiu

Conversational search systems require effective handling of context-dependent
queries that often contain ambiguity, omission, and coreference. Conversational
Query Reformulation (CQR) addresses this challenge by transforming these
queries into self-contained forms suitable for off-the-shelf retrievers.
However, existing CQR approaches suffer from two critical constraints: high
dependency on costly external supervision from human annotations or large
language models, and insufficient alignment between the rewriting model and
downstream retrievers. We present ConvSearch-R1, the first self-driven
framework that completely eliminates dependency on external rewrite supervision
by leveraging reinforcement learning to optimize reformulation directly through
retrieval signals. Our novel two-stage approach combines Self-Driven Policy
Warm-Up to address the cold-start problem through retrieval-guided
self-distillation, followed by Retrieval-Guided Reinforcement Learning with a
specially designed rank-incentive reward shaping mechanism that addresses the
sparsity issue in conventional retrieval metrics. Extensive experiments on
TopiOCQA and QReCC datasets demonstrate that ConvSearch-R1 significantly
outperforms previous state-of-the-art methods, achieving over 10% improvement
on the challenging TopiOCQA dataset while using smaller 3B parameter models
without any external supervision.

### 9. [Are the confidence scores of reviewers consistent with the review content? Evidence from top conference proceedings in AI](http://arxiv.org/pdf/2505.15031v1)

Authors: Wenqing Wu, Haixu Xi, Chengzhi Zhang

Peer review is vital in academia for evaluating research quality. Top AI
conferences use reviewer confidence scores to ensure review reliability, but
existing studies lack fine-grained analysis of text-score consistency,
potentially missing key details. This work assesses consistency at word,
sentence, and aspect levels using deep learning and NLP conference review data.
We employ deep learning to detect hedge sentences and aspects, then analyze
report length, hedge word/sentence frequency, aspect mentions, and sentiment to
evaluate text-score alignment. Correlation, significance, and regression tests
examine confidence scores' impact on paper outcomes. Results show high
text-score consistency across all levels, with regression revealing higher
confidence scores correlate with paper rejection, validating expert assessments
and peer review fairness.

### 10. [GitHub Repository Complexity Leads to Diminished Web Archive Availability](http://arxiv.org/pdf/2505.15042v1)

Authors: David Calano, Michele C. Weigle, Michael L. Nelson

Software is often developed using versioned controlled software, such as Git,
and hosted on centralized Web hosts, such as GitHub and GitLab. These Web
hosted software repositories are made available to users in the form of
traditional HTML Web pages for each source file and directory, as well as a
presentational home page and various descriptive pages. We examined more than
12,000 Web hosted Git repository project home pages, primarily from GitHub, to
measure how well their presentational components are preserved in the Internet
Archive, as well as the source trees of the collected GitHub repositories to
assess the extent to which their source code has been preserved. We found that
more than 31% of the archived repository home pages examined exhibited some
form of minor page damage and 1.6% exhibited major page damage. We also found
that of the source trees analyzed, less than 5% of their source files were
archived, on average, with the majority of repositories not having source files
saved in the Internet Archive at all. The highest concentration of archived
source files available were those linked directly from repositories' home pages
at a rate of 14.89% across all available repositories and sharply dropping off
at deeper levels of a repository's directory tree.

### Machine Learning

### 1. [Beyond Node Attention: Multi-Scale Harmonic Encoding for Feature-Wise Graph Message Passing](http://arxiv.org/pdf/2505.15015v1)

Authors: Longlong Li, Cunquan Qu, Guanghui Wang

Conventional Graph Neural Networks (GNNs) aggregate neighbor embeddings as
holistic vectors, lacking the ability to identify fine-grained,
direction-specific feature relevance. We propose MSH-GNN (Multi-Scale Harmonic
Graph Neural Network), a novel architecture that performs feature-wise adaptive
message passing through node-specific harmonic projections. For each node,
MSH-GNN dynamically projects neighbor features onto frequency-sensitive
directions determined by the target node's own representation. These
projections are further modulated using learnable sinusoidal encodings at
multiple frequencies, enabling the model to capture both smooth and oscillatory
structural patterns across scales. A frequency-aware attention pooling
mechanism is introduced to emphasize spectrally and structurally salient nodes
during readout. Theoretically, we prove that MSH-GNN approximates
shift-invariant kernels and matches the expressive power of the
1-Weisfeiler-Lehman (1-WL) test. Empirically, MSH-GNN consistently outperforms
state-of-the-art models on a wide range of graph and node classification tasks.
Furthermore, in challenging classification settings involving joint variations
in graph topology and spectral frequency, MSH-GNN excels at capturing
structural asymmetries and high-frequency modulations, enabling more accurate
graph discrimination.

### 2. [Harnessing On-Device Large Language Model: Empirical Results and Implications for AI PC](http://arxiv.org/pdf/2505.15030v2)

Authors: Qingyu Song, Peiyu Liao, Wenqian Zhao, Yiwen Wang, Shoubo Hu, Hui-Ling Zhen, Ning Jiang, Mingxuan Yuan

The increasing deployment of Large Language Models (LLMs) on edge devices,
driven by model advancements and hardware improvements, offers significant
privacy benefits. However, these on-device LLMs inherently face performance
limitations due to reduced model capacity and necessary compression techniques.
To address this, we introduce a systematic methodology -- encompassing model
capability, development efficiency, and system resources -- for evaluating
on-device LLMs. Our comprehensive evaluation, encompassing models from 0.5B to
14B parameters and seven post-training quantization (PTQ) methods on commodity
laptops, yields several critical insights: 1) System-level metrics exhibit
near-linear scaling with effective bits-per-weight (BPW). 2) A practical
threshold exists around $\sim$3.5 effective BPW, larger models subjected to
low-bit quantization consistently outperform smaller models utilizing higher
bit-precision. 3) Quantization with low BPW incurs marginal accuracy loss but
significant memory savings. 4) Determined by low-level implementation specifics
power consumption on CPU, where computation-intensive operations spend more
power than memory-intensive ones. These findings offer crucial insights and
practical guidelines for the efficient deployment and optimized configuration
of LLMs on resource-constrained edge devices. Our codebase is available at
https://github.com/simmonssong/LLMOnDevice.

### 3. [RLBenchNet: The Right Network for the Right Reinforcement Learning Task](http://arxiv.org/pdf/2505.15040v1)

Authors: Ivan Smirnov, Shangding Gu

Reinforcement learning (RL) has seen significant advancements through the
application of various neural network architectures. In this study, we
systematically investigate the performance of several neural networks in RL
tasks, including Long Short-Term Memory (LSTM), Multi-Layer Perceptron (MLP),
Mamba/Mamba-2, Transformer-XL, Gated Transformer-XL, and Gated Recurrent Unit
(GRU). Through comprehensive evaluation across continuous control, discrete
decision-making, and memory-based environments, we identify
architecture-specific strengths and limitations. Our results reveal that: (1)
MLPs excel in fully observable continuous control tasks, providing an optimal
balance of performance and efficiency; (2) recurrent architectures like LSTM
and GRU offer robust performance in partially observable environments with
moderate memory requirements; (3) Mamba models achieve a 4.5x higher throughput
compared to LSTM and a 3.9x increase over GRU, all while maintaining comparable
performance; and (4) only Transformer-XL, Gated Transformer-XL, and Mamba-2
successfully solve the most challenging memory-intensive tasks, with Mamba-2
requiring 8x less memory than Transformer-XL. These findings provide insights
for researchers and practitioners, enabling more informed architecture
selection based on specific task characteristics and computational constraints.
Code is available at: https://github.com/SafeRL-Lab/RLBenchNet

### 4. [Agentic Feature Augmentation: Unifying Selection and Generation with Teaming, Planning, and Memories](http://arxiv.org/pdf/2505.15076v1)

Authors: Nanxu Gong, Sixun Dong, Haoyue Bai, Xinyuan Wang, Wangyang Ying, Yanjie Fu

As a widely-used and practical tool, feature engineering transforms raw data
into discriminative features to advance AI model performance. However, existing
methods usually apply feature selection and generation separately, failing to
strive a balance between reducing redundancy and adding meaningful dimensions.
To fill this gap, we propose an agentic feature augmentation concept, where the
unification of feature generation and selection is modeled as agentic teaming
and planning. Specifically, we develop a Multi-Agent System with Long and
Short-Term Memory (MAGS), comprising a selector agent to eliminate redundant
features, a generator agent to produce informative new dimensions, and a router
agent that strategically coordinates their actions. We leverage in-context
learning with short-term memory for immediate feedback refinement and long-term
memory for globally optimal guidance. Additionally, we employ offline Proximal
Policy Optimization (PPO) reinforcement fine-tuning to train the router agent
for effective decision-making to navigate a vast discrete feature space.
Extensive experiments demonstrate that this unified agentic framework
consistently achieves superior task performance by intelligently orchestrating
feature selection and generation.

### 5. [Khan-GCL: Kolmogorov-Arnold Network Based Graph Contrastive Learning with Hard Negatives](http://arxiv.org/pdf/2505.15103v1)

Authors: Zihu Wang, Boxun Xu, Hejia Geng, Peng Li

Graph contrastive learning (GCL) has demonstrated great promise for learning
generalizable graph representations from unlabeled data. However, conventional
GCL approaches face two critical limitations: (1) the restricted expressive
capacity of multilayer perceptron (MLP) based encoders, and (2) suboptimal
negative samples that either from random augmentations-failing to provide
effective 'hard negatives'-or generated hard negatives without addressing the
semantic distinctions crucial for discriminating graph data. To this end, we
propose Khan-GCL, a novel framework that integrates the Kolmogorov-Arnold
Network (KAN) into the GCL encoder architecture, substantially enhancing its
representational capacity. Furthermore, we exploit the rich information
embedded within KAN coefficient parameters to develop two novel critical
feature identification techniques that enable the generation of semantically
meaningful hard negative samples for each graph representation. These
strategically constructed hard negatives guide the encoder to learn more
discriminative features by emphasizing critical semantic differences between
graphs. Extensive experiments demonstrate that our approach achieves
state-of-the-art performance compared to existing GCL methods across a variety
of datasets and tasks.

### 6. [Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models](http://arxiv.org/pdf/2505.15130v1)

Authors: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

Vision-Language Models (VLMs) such as CLIP have shown remarkable performance
in cross-modal tasks through large-scale contrastive pre-training. To adapt
these large transformer-based models efficiently for downstream tasks,
Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as
scalable alternatives to full fine-tuning, especially in few-shot scenarios.
However, like traditional deep neural networks, VLMs are highly vulnerable to
adversarial attacks, where imperceptible perturbations can significantly
degrade model performance. Adversarial training remains the most effective
strategy for improving model robustness in PEFT. In this work, we propose
AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial
robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method
formulates adversarial fine-tuning as a minimax optimization problem and
provides theoretical guarantees for convergence under smoothness and
nonconvex-strong-concavity assumptions. Empirical results across eight datasets
using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly
improves robustness against common adversarial attacks (e.g., FGSM, PGD),
without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA
as a practical and theoretically grounded approach for robust adaptation of
VLMs in resource-constrained settings.

### 7. [EC-LDA : Label Distribution Inference Attack against Federated Graph Learning with Embedding Compression](http://arxiv.org/pdf/2505.15140v1)

Authors: Tong Cheng, Fu Jie, Xinpeng Ling, Huifa Li, Zhili Chen

Graph Neural Networks (GNNs) have been widely used for graph analysis.
Federated Graph Learning (FGL) is an emerging learning framework to
collaboratively train graph data from various clients. However, since clients
are required to upload model parameters to the server in each round, this
provides the server with an opportunity to infer each client's data privacy. In
this paper, we focus on label distribution attacks(LDAs) that aim to infer the
label distributions of the clients' local data. We take the first step to
attack client's label distributions in FGL. Firstly, we observe that the
effectiveness of LDA is closely related to the variance of node embeddings in
GNNs. Next, we analyze the relation between them and we propose a new attack
named EC-LDA, which significantly improves the attack effectiveness by
compressing node embeddings. Thirdly, extensive experiments on node
classification and link prediction tasks across six widely used graph datasets
show that EC-LDA outperforms the SOTA LDAs. For example, EC-LDA attains optimal
values under both Cos-sim and JS-div evaluation metrics in the CoraFull and
LastFM datasets. Finally, we explore the robustness of EC-LDA under
differential privacy protection.

### 8. [Filtering Learning Histories Enhances In-Context Reinforcement Learning](http://arxiv.org/pdf/2505.15143v1)

Authors: Weiqin Chen, Xinjie Zhang, Dharmashankar Subramanian, Santiago Paternain

Transformer models (TMs) have exhibited remarkable in-context reinforcement
learning (ICRL) capabilities, allowing them to generalize to and improve in
previously unseen environments without re-training or fine-tuning. This is
typically accomplished by imitating the complete learning histories of a source
RL algorithm over a substantial amount of pretraining environments, which,
however, may transfer suboptimal behaviors inherited from the source
algorithm/dataset. Therefore, in this work, we address the issue of inheriting
suboptimality from the perspective of dataset preprocessing. Motivated by the
success of the weighted empirical risk minimization, we propose a simple yet
effective approach, learning history filtering (LHF), to enhance ICRL by
reweighting and filtering the learning histories based on their improvement and
stability characteristics. To the best of our knowledge, LHF is the first
approach to avoid source suboptimality by dataset preprocessing, and can be
combined with the current state-of-the-art (SOTA) ICRL algorithms. We
substantiate the effectiveness of LHF through a series of experiments conducted
on the well-known ICRL benchmarks, encompassing both discrete environments and
continuous robotic manipulation tasks, with three SOTA ICRL algorithms (AD,
DPT, DICP) as the backbones. LHF exhibits robust performance across a variety
of suboptimal scenarios, as well as under varying hyperparameters and sampling
strategies. Notably, the superior performance of LHF becomes more pronounced in
the presence of noisy data, indicating the significance of filtering learning
histories.

### 9. [Time Tracker: Mixture-of-Experts-Enhanced Foundation Time Series Forecasting Model with Decoupled Training Pipelines](http://arxiv.org/pdf/2505.15151v1)

Authors: Xiaohou Shi, Ke Li, Aobo Liang, Yan Sun

In the past few years, time series foundation models have achieved superior
predicting accuracy. However, real-world time series often exhibit significant
diversity in their temporal patterns across different time spans and domains,
making it challenging for a single model architecture to fit all complex
scenarios. In addition, time series data may have multiple variables exhibiting
complex correlations between each other. Recent mainstream works have focused
on modeling times series in a channel-independent manner in both pretraining
and finetuning stages, overlooking the valuable inter-series dependencies. To
this end, we propose \textbf{Time Tracker} for better predictions on
multivariate time series data. Firstly, we leverage sparse mixture of experts
(MoE) within Transformers to handle the modeling of diverse time series
patterns, thereby alleviating the learning difficulties of a single model while
improving its generalization. Besides, we propose Any-variate Attention,
enabling a unified model structure to seamlessly handle both univariate and
multivariate time series, thereby supporting channel-independent modeling
during pretraining and channel-mixed modeling for finetuning. Furthermore, we
design a graph learning module that constructs relations among sequences from
frequency-domain features, providing more precise guidance to capture
inter-series dependencies in channel-mixed modeling. Based on these
advancements, Time Tracker achieves state-of-the-art performance in predicting
accuracy, model generalization and adaptability.

### 10. [Sculpting Features from Noise: Reward-Guided Hierarchical Diffusion for Task-Optimal Feature Transformation](http://arxiv.org/pdf/2505.15152v1)

Authors: Nanxu Gong, Zijun Li, Sixun Dong, Haoyue Bai, Wangyang Ying, Xinyuan Wang, Yanjie Fu

Feature Transformation (FT) crafts new features from original ones via
mathematical operations to enhance dataset expressiveness for downstream
models. However, existing FT methods exhibit critical limitations: discrete
search struggles with enormous combinatorial spaces, impeding practical use;
and continuous search, being highly sensitive to initialization and step sizes,
often becomes trapped in local optima, restricting global exploration. To
overcome these limitations, DIFFT redefines FT as a reward-guided generative
task. It first learns a compact and expressive latent space for feature sets
using a Variational Auto-Encoder (VAE). A Latent Diffusion Model (LDM) then
navigates this space to generate high-quality feature embeddings, its
trajectory guided by a performance evaluator towards task-specific optima. This
synthesis of global distribution learning (from LDM) and targeted optimization
(reward guidance) produces potent embeddings, which a novel semi-autoregressive
decoder efficiently converts into structured, discrete features, preserving
intra-feature dependencies while allowing parallel inter-feature generation.
Extensive experiments on 14 benchmark datasets show DIFFT consistently
outperforms state-of-the-art baselines in predictive accuracy and robustness,
with significantly lower training and inference times.

### Neural and Evolutionary Computing

### 1. [Degree-Optimized Cumulative Polynomial Kolmogorov-Arnold Networks](http://arxiv.org/pdf/2505.15228v1)

Authors: Mathew Vanherreweghe, Lirandë Pira, Patrick Rebentrost

We introduce cumulative polynomial Kolmogorov-Arnold networks (CP-KAN), a
neural architecture combining Chebyshev polynomial basis functions and
quadratic unconstrained binary optimization (QUBO). Our primary contribution
involves reformulating the degree selection problem as a QUBO task, reducing
the complexity from $O(D^N)$ to a single optimization step per layer. This
approach enables efficient degree selection across neurons while maintaining
computational tractability. The architecture performs well in regression tasks
with limited data, showing good robustness to input scales and natural
regularization properties from its polynomial basis. Additionally, theoretical
analysis establishes connections between CP-KAN's performance and properties of
financial time series. Our empirical validation across multiple domains
demonstrates competitive performance compared to several traditional
architectures tested, especially in scenarios where data efficiency and
numerical stability are important. Our implementation, including strategies for
managing computational overhead in larger networks is available in
Ref.~\citep{cpkan_implementation}.

### 2. [AM-PPO: (Advantage) Alpha-Modulation with Proximal Policy Optimization](http://arxiv.org/pdf/2505.15514v1)

Authors: Soham Sane

Proximal Policy Optimization (PPO) is a widely used reinforcement learning
algorithm that heavily relies on accurate advantage estimates for stable and
efficient training. However, raw advantage signals can exhibit significant
variance, noise, and scale-related issues, impeding optimal learning
performance. To address this challenge, we introduce Advantage Modulation PPO
(AM-PPO), a novel enhancement of PPO that adaptively modulates advantage
estimates using a dynamic, non-linear scaling mechanism. This adaptive
modulation employs an alpha controller that dynamically adjusts the scaling
factor based on evolving statistical properties of the advantage signals, such
as their norm, variance, and a predefined target saturation level. By
incorporating a tanh-based gating function driven by these adaptively scaled
advantages, AM-PPO reshapes the advantage signals to stabilize gradient updates
and improve the conditioning of the policy gradient landscape. Crucially, this
modulation also influences value function training by providing consistent and
adaptively conditioned learning targets. Empirical evaluations across standard
continuous control benchmarks demonstrate that AM-PPO achieves superior reward
trajectories, exhibits sustained learning progression, and significantly
reduces the clipping required by adaptive optimizers. These findings underscore
the potential of advantage modulation as a broadly applicable technique for
enhancing reinforcement learning optimization.

### 3. [Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms](http://arxiv.org/pdf/2505.15661v1)

Authors: Sina Mohammad-Taheri, Matthew J. Colbrook, Simone Brugiapaglia

Gradient-based learning imposes (deep) neural networks to be differentiable
at all steps. This includes model-based architectures constructed by unrolling
iterations of an iterative algorithm onto layers of a neural network, known as
algorithm unrolling. However, greedy sparse recovery algorithms depend on the
non-differentiable argsort operator, which hinders their integration into
neural networks. In this paper, we address this challenge in Orthogonal
Matching Pursuit (OMP) and Iterative Hard Thresholding (IHT), two popular
representative algorithms in this class. We propose permutation-based variants
of these algorithms and approximate permutation matrices using "soft"
permutation matrices derived from softsort, a continuous relaxation of argsort.
We demonstrate -- both theoretically and numerically -- that Soft-OMP and
Soft-IHT, as differentiable counterparts of OMP and IHT and fully compatible
with neural network training, effectively approximate these algorithms with a
controllable degree of accuracy. This leads to the development of OMP- and
IHT-Net, fully trainable network architectures based on Soft-OMP and Soft-IHT,
respectively. Finally, by choosing weights as "structure-aware" trainable
parameters, we connect our approach to structured sparse recovery and
demonstrate its ability to extract latent sparsity patterns from data.

### 4. [Evolutionary Computation and Large Language Models: A Survey of Methods, Synergies, and Applications](http://arxiv.org/pdf/2505.15741v1)

Authors: Dikshit Chauhan, Bapi Dutta, Indu Bala, Niki van Stein, Thomas Bäck, Anupam Yadav

Integrating Large Language Models (LLMs) and Evolutionary Computation (EC)
represents a promising avenue for advancing artificial intelligence by
combining powerful natural language understanding with optimization and search
capabilities. This manuscript explores the synergistic potential of LLMs and
EC, reviewing their intersections, complementary strengths, and emerging
applications. We identify key opportunities where EC can enhance LLM training,
fine-tuning, prompt engineering, and architecture search, while LLMs can, in
turn, aid in automating the design, analysis, and interpretation of ECs. The
manuscript explores the synergistic integration of EC and LLMs, highlighting
their bidirectional contributions to advancing artificial intelligence. It
first examines how EC techniques enhance LLMs by optimizing key components such
as prompt engineering, hyperparameter tuning, and architecture search,
demonstrating how evolutionary methods automate and refine these processes.
Secondly, the survey investigates how LLMs improve EC by automating
metaheuristic design, tuning evolutionary algorithms, and generating adaptive
heuristics, thereby increasing efficiency and scalability. Emerging
co-evolutionary frameworks are discussed, showcasing applications across
diverse fields while acknowledging challenges like computational costs,
interpretability, and algorithmic convergence. The survey concludes by
identifying open research questions and advocating for hybrid approaches that
combine the strengths of EC and LLMs.

### 5. [Decoding Phone Pairs from MEG Signals Across Speech Modalities](http://arxiv.org/pdf/2505.15355v1)

Authors: Xabier de Zuazo, Eva Navas, Ibon Saratxaga, Mathieu Bourguignon, Nicola Molinaro

Understanding the neural mechanisms underlying speech production is essential
for both advancing cognitive neuroscience theory and developing practical
communication technologies. In this study, we investigated
magnetoencephalography signals to decode phones from brain activity during
speech production and perception (passive listening and voice playback) tasks.
Using a dataset comprising 17 participants, we performed pairwise phone
classification, extending our analysis to 15 phonetic pairs. Multiple machine
learning approaches, including regularized linear models and neural network
architectures, were compared to determine their effectiveness in decoding
phonetic information. Our results demonstrate significantly higher decoding
accuracy during speech production (76.6%) compared to passive listening and
playback modalities (~51%), emphasizing the richer neural information available
during overt speech. Among the models, the Elastic Net classifier consistently
outperformed more complex neural networks, highlighting the effectiveness of
traditional regularization techniques when applied to limited and
high-dimensional MEG datasets. Besides, analysis of specific brain frequency
bands revealed that low-frequency oscillations, particularly Delta (0.2-3 Hz)
and Theta (4-7 Hz), contributed the most substantially to decoding accuracy,
suggesting that these bands encode critical speech production-related neural
processes. Despite using advanced denoising methods, it remains unclear whether
decoding solely reflects neural activity or if residual muscular or movement
artifacts also contributed, indicating the need for further methodological
refinement. Overall, our findings underline the critical importance of
examining overt speech production paradigms, which, despite their complexity,
offer opportunities to improve brain-computer interfaces to help individuals
with severe speech impairments.

### Networking and Internet Architecture

### 1. [Modeling and Optimizing Latency for Delayed Hit Caching with Stochastic Miss Latency](http://arxiv.org/pdf/2505.15531v1)

Authors: Bowen Jiang, Chaofan Ma

Caching is crucial for system performance, but the delayed hit phenomenon,
where requests queue during lengthy fetches after a cache miss, significantly
degrades user-perceived latency in modern high-throughput systems. While prior
works address delayed hits by estimating aggregate delay, they universally
assume deterministic fetch latencies. This paper tackles the more realistic,
yet unexplored, scenario where fetch latencies are stochastic. We present, to
our knowledge, the first theoretical analysis of delayed hits under this
condition, deriving analytical expressions for both the mean and variance of
the aggregate delay assuming exponentially distributed fetch latency.
Leveraging these insights, we develop a novel variance-aware ranking function
tailored for this stochastic setting to guide cache eviction decisions more
effectively. The simulations on synthetic and real-world datasets demonstrate
that our proposed algorithm significantly reduces overall latency compared to
state-of-the-art delayed-hit strategies, achieving a $3\%-30\%$ reduction on
synthetic datasets and approximately $1\%-7\%$ reduction on real-world traces.

### 2. [Dynamic Spectrum Sharing Based on the Rentable NFT Standard ERC4907](http://arxiv.org/pdf/2505.15148v1)

Authors: Litao Ye, Bin Chen, Shrivastava Shivanshu, Chen Sun, Shuo Wang, Siming Feng, Shengli Zhang

Centralized Dynamic Spectrum Sharing (DSS) faces challenges like data
security, high management costs, and limited scalability. To address these
issues, a blockchain-based DSS scheme has been proposed in this paper. First,
we utilize the ERC4907 standard to mint Non-Fungible Spectrum Tokens (NFSTs)
that serve as unique identifiers for spectrum resources and facilitate renting.
Next, we develop a smart contract for NFST auctions, ensuring secure spectrum
transactions through the auction process. Lastly, we create a Web3 spectrum
auction platform where users can access idle spectrum data and participate in
auctions for NFST leases corresponding to the available spectrum. Experimental
results demonstrate that our NFST, designed according to the ERC4907 standard,
effectively meets users' secure and efficient DSS requirements, making it a
feasible solution.

### 3. [Evaluation of Mobile Environment for Vehicular Visible Light Communication Using Multiple LEDs and Event Cameras](http://arxiv.org/pdf/2505.15412v1)

Authors: Ryota Soga, Shintaro Shiba, Quan Kong, Norimasa Kobori, Tsukasa Shimizu, Shan Lu, Takaya Yamazato

In the fields of Advanced Driver Assistance Systems (ADAS) and Autonomous
Driving (AD), sensors that serve as the ``eyes'' for sensing the vehicle's
surrounding environment are essential. Traditionally, image sensors and LiDAR
have played this role. However, a new type of vision sensor, event cameras, has
recently attracted attention. Event cameras respond to changes in the
surrounding environment (e.g., motion), exhibit strong robustness against
motion blur, and perform well in high dynamic range environments, which are
desirable in robotics applications. Furthermore, the asynchronous and
low-latency principles of data acquisition make event cameras suitable for
optical communication. By adding communication functionality to event cameras,
it becomes possible to utilize I2V communication to immediately share
information about forward collisions, sudden braking, and road conditions,
thereby contributing to hazard avoidance. Additionally, receiving information
such as signal timing and traffic volume enables speed adjustment and optimal
route selection, facilitating more efficient driving. In this study, we
construct a vehicle visible light communication system where event cameras are
receivers, and multiple LEDs are transmitters. In driving scenes, the system
tracks the transmitter positions and separates densely packed LED light sources
using pilot sequences based on Walsh-Hadamard codes. As a result, outdoor
vehicle experiments demonstrate error-free communication under conditions where
the transmitter-receiver distance was within 40 meters and the vehicle's
driving speed was 30 km/h (8.3 m/s).

### Robotics

### 1. [Shape-Adaptive Planning and Control for a Deformable Quadrotor](http://arxiv.org/pdf/2505.15010v1)

Authors: Yuze Wu, Zhichao Han, Xuankang Wu, Yuan Zhou, Junjie Wang, Zheng Fang, Fei Gao

Drones have become essential in various applications, but conventional
quadrotors face limitations in confined spaces and complex tasks. Deformable
drones, which can adapt their shape in real-time, offer a promising solution to
overcome these challenges, while also enhancing maneuverability and enabling
novel tasks like object grasping. This paper presents a novel approach to
autonomous motion planning and control for deformable quadrotors. We introduce
a shape-adaptive trajectory planner that incorporates deformation dynamics into
path generation, using a scalable kinodynamic A* search to handle deformation
parameters in complex environments. The backend spatio-temporal optimization is
capable of generating optimally smooth trajectories that incorporate shape
deformation. Additionally, we propose an enhanced control strategy that
compensates for external forces and torque disturbances, achieving a 37.3\%
reduction in trajectory tracking error compared to our previous work. Our
approach is validated through simulations and real-world experiments,
demonstrating its effectiveness in narrow-gap traversal and multi-modal
deformable tasks.

### 2. [Histo-Planner: A Real-time Local Planner for MAVs Teleoperation based on Histogram of Obstacle Distribution](http://arxiv.org/pdf/2505.15043v1)

Authors: Ze Wang, Zhenyu Gao, Jingang Qu, Pascal Morin

This paper concerns real-time obstacle avoidance for micro aerial vehicles
(MAVs). Motivated by teleoperation applications in cluttered environments with
limited computational power, we propose a local planner that does not require
the knowledge or construction of a global map of the obstacles. The proposed
solution consists of a real-time trajectory planning algorithm that relies on
the histogram of obstacle distribution and a planner manager that triggers
different planning modes depending on obstacles location around the MAV. The
proposed solution is validated, for a teleoperation application, with both
simulations and indoor experiments. Benchmark comparisons based on a designed
simulation platform are also provided.

### 3. [GCNT: Graph-Based Transformer Policies for Morphology-Agnostic Reinforcement Learning](http://arxiv.org/pdf/2505.15211v1)

Authors: Yingbo Luo, Meibao Yao, Xueming Xiao

Training a universal controller for robots with different morphologies is a
promising research trend, since it can significantly enhance the robustness and
resilience of the robotic system. However, diverse morphologies can yield
different dimensions of state space and action space, making it difficult to
comply with traditional policy networks. Existing methods address this issue by
modularizing the robot configuration, while do not adequately extract and
utilize the overall morphological information, which has been proven crucial
for training a universal controller. To this end, we propose GCNT, a
morphology-agnostic policy network based on improved Graph Convolutional
Network (GCN) and Transformer. It exploits the fact that GCN and Transformer
can handle arbitrary number of modules to achieve compatibility with diverse
morphologies. Our key insight is that the GCN is able to efficiently extract
morphology information of robots, while Transformer ensures that it is fully
utilized by allowing each node of the robot to communicate this information
directly. Experimental results show that our method can generate resilient
locomotion behaviors for robots with different configurations, including
zero-shot generalization to robot morphologies not seen during training. In
particular, GCNT achieved the best performance on 8 tasks in the 2 standard
benchmarks.

### 4. [Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control](http://arxiv.org/pdf/2505.15304v1)

Authors: Seongmin Park, Hyungmin Kim, Sangwoo kim, Wonseok Jeon, Juyoung Yang, Byeongwook Jeon, Yoonseon Oh, Jungwook Choi

Deep neural network (DNN)-based policy models, such as vision-language-action
(VLA) models, excel at automating complex decision-making from multi-modal
inputs. However, scaling these models greatly increases computational overhead,
complicating deployment in resource-constrained settings like robot
manipulation and autonomous driving. To address this, we propose Saliency-Aware
Quantized Imitation Learning (SQIL), which combines quantization-aware training
with a selective loss-weighting strategy for mission-critical states. By
identifying these states via saliency scores and emphasizing them in the
training loss, SQIL preserves decision fidelity under low-bit precision. We
validate SQIL's generalization capability across extensive simulation
benchmarks with environment variations, real-world tasks, and cross-domain
tasks (self-driving, physics simulation), consistently recovering
full-precision performance. Notably, a 4-bit weight-quantized VLA model for
robotic manipulation achieves up to 2.5x speedup and 2.5x energy savings on an
edge GPU with minimal accuracy loss. These results underline SQIL's potential
for efficiently deploying large IL-based policy models on resource-limited
devices.

### 5. [Synthetic Enclosed Echoes: A New Dataset to Mitigate the Gap Between Simulated and Real-World Sonar Data](http://arxiv.org/pdf/2505.15465v1)

Authors: Guilherme de Oliveira, Matheus M. dos Santos, Paulo L. J. Drews-Jr

This paper introduces Synthetic Enclosed Echoes (SEE), a novel dataset
designed to enhance robot perception and 3D reconstruction capabilities in
underwater environments. SEE comprises high-fidelity synthetic sonar data,
complemented by a smaller subset of real-world sonar data. To facilitate
flexible data acquisition, a simulated environment has been developed, enabling
the generation of additional data through modifications such as the inclusion
of new structures or imaging sonar configurations. This hybrid approach
leverages the advantages of synthetic data, including readily available ground
truth and the ability to generate diverse datasets, while bridging the
simulation-to-reality gap with real-world data acquired in a similar
environment. The SEE dataset comprehensively evaluates acoustic data-based
methods, including mathematics-based sonar approaches and deep learning
algorithms. These techniques were employed to validate the dataset, confirming
its suitability for underwater 3D reconstruction. Furthermore, this paper
proposes a novel modification to a state-of-the-art algorithm, demonstrating
improved performance compared to existing methods. The SEE dataset enables the
evaluation of acoustic data-based methods in realistic scenarios, thereby
improving their feasibility for real-world underwater applications.

### 6. [From Grounding to Manipulation: Case Studies of Foundation Model Integration in Embodied Robotic Systems](http://arxiv.org/pdf/2505.15685v1)

Authors: Xiuchao Sui, Daiying Tian, Qi Sun, Ruirui Chen, Dongkyu Choi, Kenneth Kwok, Soujanya Poria

Foundation models (FMs) are increasingly used to bridge language and action
in embodied agents, yet the operational characteristics of different FM
integration strategies remain under-explored -- particularly for complex
instruction following and versatile action generation in changing environments.
This paper examines three paradigms for building robotic systems: end-to-end
vision-language-action (VLA) models that implicitly integrate perception and
planning, and modular pipelines incorporating either vision-language models
(VLMs) or multimodal large language models (LLMs). We evaluate these paradigms
through two focused case studies: a complex instruction grounding task
assessing fine-grained instruction understanding and cross-modal
disambiguation, and an object manipulation task targeting skill transfer via
VLA finetuning. Our experiments in zero-shot and few-shot settings reveal
trade-offs in generalization and data efficiency. By exploring performance
limits, we distill design implications for developing language-driven physical
agents and outline emerging challenges and opportunities for FM-powered
robotics in real-world conditions.

### 7. [UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning](http://arxiv.org/pdf/2505.15725v1)

Authors: Xiangyu Wang, Donglin Yang, Yue Liao, Wenhao Zheng, wenjun wu, Bin Dai, Hongsheng Li, Si Liu

Unmanned Aerial Vehicles (UAVs) are evolving into language-interactive
platforms, enabling more intuitive forms of human-drone interaction. While
prior works have primarily focused on high-level planning and long-horizon
navigation, we shift attention to language-guided fine-grained trajectory
control, where UAVs execute short-range, reactive flight behaviors in response
to language instructions. We formalize this problem as the Flying-on-a-Word
(Flow) task and introduce UAV imitation learning as an effective approach. In
this framework, UAVs learn fine-grained control policies by mimicking expert
pilot trajectories paired with atomic language instructions. To support this
paradigm, we present UAV-Flow, the first real-world benchmark for
language-conditioned, fine-grained UAV control. It includes a task formulation,
a large-scale dataset collected in diverse environments, a deployable control
framework, and a simulation suite for systematic evaluation. Our design enables
UAVs to closely imitate the precise, expert-level flight trajectories of human
pilots and supports direct deployment without sim-to-real gap. We conduct
extensive experiments on UAV-Flow, benchmarking VLN and VLA paradigms. Results
show that VLA models are superior to VLN baselines and highlight the critical
role of spatial grounding in the fine-grained Flow setting.

### 8. [AnyBody: A Benchmark Suite for Cross-Embodiment Manipulation](http://arxiv.org/pdf/2505.14986v1)

Authors: Meenal Parakh, Alexandre Kirchmeyer, Beining Han, Jia Deng

Generalizing control policies to novel embodiments remains a fundamental
challenge in enabling scalable and transferable learning in robotics. While
prior works have explored this in locomotion, a systematic study in the context
of manipulation tasks remains limited, partly due to the lack of standardized
benchmarks. In this paper, we introduce a benchmark for learning
cross-embodiment manipulation, focusing on two foundational tasks-reach and
push-across a diverse range of morphologies. The benchmark is designed to test
generalization along three axes: interpolation (testing performance within a
robot category that shares the same link structure), extrapolation (testing on
a robot with a different link structure), and composition (testing on
combinations of link structures). On the benchmark, we evaluate the ability of
different RL policies to learn from multiple morphologies and to generalize to
novel ones. Our study aims to answer whether morphology-aware training can
outperform single-embodiment baselines, whether zero-shot generalization to
unseen morphologies is feasible, and how consistently these patterns hold
across different generalization regimes. The results highlight the current
limitations of multi-embodiment learning and provide insights into how
architectural and training design choices influence policy generalization.

### 9. [UniSTPA: A Safety Analysis Framework for End-to-End Autonomous Driving](http://arxiv.org/pdf/2505.15005v1)

Authors: Hongrui Kou, Zhouhang Lyu, Ziyu Wang, Cheng Wang, Yuxin Zhang

As autonomous driving technology continues to advance, end-to-end models have
attracted considerable attention owing to their superior generalisation
capability. Nevertheless, such learning-based systems entail numerous safety
risks throughout development and on-road deployment, and existing
safety-analysis methods struggle to identify these risks comprehensively. To
address this gap, we propose the Unified System Theoretic Process Analysis
(UniSTPA) framework, which extends the scope of STPA from the operational phase
to the entire lifecycle of an end-to-end autonomous driving system, including
information gathering, data preparation, closed loop training, verification,
and deployment. UniSTPA performs hazard analysis not only at the component
level but also within the model's internal layers, thereby enabling
fine-grained assessment of inter and intra module interactions. Using a highway
Navigate on Autopilot function as a case study, UniSTPA uncovers multi-stage
hazards overlooked by conventional approaches including scene design defects,
sensor fusion biases, and internal model flaws, through multi-level causal
analysis, traces these hazards to deeper issues such as data quality, network
architecture, and optimisation objectives. The analysis result are used to
construct a safety monitoring and safety response mechanism that supports
continuous improvement from hazard identification to system optimisation. The
proposed framework thus offers both theoretical and practical guidance for the
safe development and deployment of end-to-end autonomous driving systems.

### 10. [Object-Focus Actor for Data-efficient Robot Generalization Dexterous Manipulation](http://arxiv.org/pdf/2505.15098v1)

Authors: Yihang Li, Tianle Zhang, Xuelong Wei, Jiayi Li, Lin Zhao, Dongchi Huang, Zhirui Fang, Minhua Zheng, Wenjun Dai, Xiaodong He

Robot manipulation learning from human demonstrations offers a rapid means to
acquire skills but often lacks generalization across diverse scenes and object
placements. This limitation hinders real-world applications, particularly in
complex tasks requiring dexterous manipulation. Vision-Language-Action (VLA)
paradigm leverages large-scale data to enhance generalization. However, due to
data scarcity, VLA's performance remains limited. In this work, we introduce
Object-Focus Actor (OFA), a novel, data-efficient approach for generalized
dexterous manipulation. OFA exploits the consistent end trajectories observed
in dexterous manipulation tasks, allowing for efficient policy training. Our
method employs a hierarchical pipeline: object perception and pose estimation,
pre-manipulation pose arrival and OFA policy execution. This process ensures
that the manipulation is focused and efficient, even in varied backgrounds and
positional layout. Comprehensive real-world experiments across seven tasks
demonstrate that OFA significantly outperforms baseline methods in both
positional and background generalization tests. Notably, OFA achieves robust
performance with only 10 demonstrations, highlighting its data efficiency.

### Software Engineering

### 1. [RAG or Fine-tuning? A Comparative Study on LCMs-based Code Completion in Industry](http://arxiv.org/pdf/2505.15179v1)

Authors: Chaozheng Wang, Zezhou Yang, Shuzheng Gao, Cuiyun Gao, Ting Peng, Hailiang Huang, Yuetang Deng, Michael Lyu

Code completion, a crucial practice in industrial settings, helps developers
improve programming efficiency by automatically suggesting code snippets during
development. With the emergence of Large Code Models (LCMs), this field has
witnessed significant advancements. Due to the natural differences between
open-source and industrial codebases, such as coding patterns and unique
internal dependencies, it is a common practice for developers to conduct domain
adaptation when adopting LCMs in industry. There exist multiple adaptation
approaches, among which retrieval-augmented generation (RAG) and fine-tuning
are the two most popular paradigms. However, no prior research has explored the
trade-off of the two approaches in industrial scenarios.
  To mitigate the gap, we comprehensively compare the two paradigms including
Retrieval-Augmented Generation (RAG) and Fine-tuning (FT), for industrial code
completion in this paper. In collaboration with Tencent's WXG department, we
collect over 160,000 internal C++ files as our codebase. We then compare the
two types of adaptation approaches from three dimensions that are concerned by
industrial practitioners, including effectiveness, efficiency, and parameter
sensitivity, using six LCMs. Our findings reveal that RAG, when implemented
with appropriate embedding models that map code snippets into dense vector
representations, can achieve higher accuracy than fine-tuning alone.
Specifically, BM25 presents superior retrieval effectiveness and efficiency
among studied RAG methods. Moreover, RAG and fine-tuning are orthogonal and
their combination leads to further improvement. We also observe that RAG
demonstrates better scalability than FT, showing more sustained performance
gains with larger scales of codebase.

### 2. [Developing clinical informatics to support direct care and population health management: the VIEWER story](http://arxiv.org/pdf/2505.15459v1)

Authors: Robert Harland, Tao Wang, David Codling, Catherine Polling, Matthew Broadbent, Holly Newton, Yamiko Joseph Msosa, Daisy Kornblum, Claire Delaney-Pope, Barbara Arroyo, Stuart MacLellan, Zoe Keddie, Mary Docherty, Angus Roberts, Derek Tracy, Philip McGuire, Richard Dobson, Robert Stewart

Electronic health records (EHRs) provide comprehensive patient data which
could be better used to enhance informed decision-making, resource allocation,
and coordinated care, thereby optimising healthcare delivery. However, in
mental healthcare, critical information, such as on risk factors, precipitants,
and treatment responses, is often embedded in unstructured text, limiting the
ability to automate at scale measures to identify and prioritise local
populations and patients, which potentially hinders timely prevention and
intervention. We describe the development and proof-of-concept implementation
of VIEWER, a clinical informatics platform designed to enhance direct patient
care and population health management by improving the accessibility and
usability of EHR data. We further outline strategies that were employed in this
work to foster informatics innovation through interdisciplinary and
cross-organisational collaboration to support integrated, personalised care,
and detail how these advancements were piloted and implemented within a large
UK mental health National Health Service Foundation Trust to improve patient
outcomes at an individual patient, clinician, clinical team, and organisational
level.

### 3. [DS-Bench: A Realistic Benchmark for Data Science Code Generation](http://arxiv.org/pdf/2505.15621v1)

Authors: Shuyin Ouyang, Dong Huang, Jingwen Guo, Zeyu Sun, Qihao Zhu, Jie M. Zhang

We introduce DS-bench, a new benchmark designed to evaluate large language
models (LLMs) on complicated and realistic data science code generation tasks.
DS-bench consists of 1,000 carefully constructed problems sourced from
realistic problems from GitHub across ten widely used Python data science
libraries. Compared to the current state-of-the-art benchmark DS-1000, DS-bench
offers a more challenging and representative testbed, longer code solutions,
more comprehensive data science libraries, clearer and better structured
problem descriptions, and stronger test suites. To construct the DS-bench, we
develop a robust pipeline that combines task scope selection, code
construction, test case generation, and problem description synthesis. The
process is paired with rigorous manual editing to ensure alignment and enhance
evaluation reliability. Experimental result shows that DS-bench exhibits robust
scaling behavior, where larger models systematically outperform smaller ones,
validating its ability to distinguish model capabilities. The best LLM we test,
GPT-4o, has a pass@1 of 0.202, indicating that LLMs still have a large room to
improve for realistic data science code generation tasks. We believe DS-bench
will serve as a rigorous and trustworthy foundation for advancing LLM-based
data science programming.

### 4. [Who "Controls" Where Work Shall be Done? State-of-Practice in Post-Pandemic Remote Work Regulation](http://arxiv.org/pdf/2505.15743v1)

Authors: Darja Smite, Nils Brede Moe, Maria Teresa Baldassarre, Fabio Calefato, Guilherme Horta Travassos, Marcin Floryan, Marcos Kalinowski, Daniel Mendez, Graziela Basilio Pereira, Margaret-Anne Storey, Rafael Prikladnicki

The COVID-19 pandemic has permanently altered workplace structures, making
remote work a widespread practice. While many employees advocate for
flexibility, many employers reconsider their attitude toward remote work and
opt for structured return-to-office mandates. Media headlines repeatedly
emphasize that the corporate world is returning to full-time office work. This
study examines how companies employing software engineers and supporting roles
regulate work location, whether corporate policies have evolved in the last
five years, and, if so, how, and why. We collected data on remote work
regulation from corporate HR and/or management representatives from 68
corporate entities that vary in size, location, and orientation towards remote
or office work. Our findings reveal that although many companies prioritize
office-centred working (50%), most companies in our sample permit hybrid
working to varying degrees (85%). Remote work regulation does not reveal any
particular new "best practice" as policies differ greatly, but the single most
popular arrangement was the three in-office days per week. More than half of
the companies (51%) encourage or mandate office days, and more than quarter
(28%) have changed regulations, gradually increasing the mandatory office
presence or implementing differentiated conditions. Although no companies have
increased flexibility, only four companies are returning to full-time office
work. Our key recommendation for office-oriented companies is to consider a
trust-based alternative to strict office presence mandates, while for companies
oriented toward remote working, we warn about the points of no (or hard)
return. Finally, the current state of policies is clearly not final, as
companies continue to experiment and adjust their work regulation.

### 5. [An Empirical Analysis of Vulnerability Detection Tools for Solidity Smart Contracts Using Line Level Manually Annotated Vulnerabilities](http://arxiv.org/pdf/2505.15756v1)

Authors: Francesco Salzano, Cosmo Kevin Antenucci, Simone Scalabrino, Giovanni Rosa, Rocco Oliveto, Remo Pareschi

The rapid adoption of blockchain technology highlighted the importance of
ensuring the security of smart contracts due to their critical role in
automated business logic execution on blockchain platforms. This paper provides
an empirical evaluation of automated vulnerability analysis tools specifically
designed for Solidity smart contracts. Leveraging the extensive SmartBugs 2.0
framework, which includes 20 analysis tools, we conducted a comprehensive
assessment using an annotated dataset of 2,182 instances we manually annotated
with line-level vulnerability labels. Our evaluation highlights the detection
effectiveness of these tools in detecting various types of vulnerabilities, as
categorized by the DASP TOP 10 taxonomy. We evaluated the effectiveness of a
Large Language Model-based detection method on two popular datasets. In this
case, we obtained inconsistent results with the two datasets, showing
unreliable detection when analyzing real-world smart contracts. Our study
identifies significant variations in the accuracy and reliability of different
tools and demonstrates the advantages of combining multiple detection methods
to improve vulnerability identification. We identified a set of 3 tools that,
combined, achieve up to 76.78\% found vulnerabilities taking less than one
minute to run, on average. This study contributes to the field by releasing the
largest dataset of manually analyzed smart contracts with line-level
vulnerability annotations and the empirical evaluation of the greatest number
of tools to date.

### 6. [UniSTPA: A Safety Analysis Framework for End-to-End Autonomous Driving](http://arxiv.org/pdf/2505.15005v1)

Authors: Hongrui Kou, Zhouhang Lyu, Ziyu Wang, Cheng Wang, Yuxin Zhang

As autonomous driving technology continues to advance, end-to-end models have
attracted considerable attention owing to their superior generalisation
capability. Nevertheless, such learning-based systems entail numerous safety
risks throughout development and on-road deployment, and existing
safety-analysis methods struggle to identify these risks comprehensively. To
address this gap, we propose the Unified System Theoretic Process Analysis
(UniSTPA) framework, which extends the scope of STPA from the operational phase
to the entire lifecycle of an end-to-end autonomous driving system, including
information gathering, data preparation, closed loop training, verification,
and deployment. UniSTPA performs hazard analysis not only at the component
level but also within the model's internal layers, thereby enabling
fine-grained assessment of inter and intra module interactions. Using a highway
Navigate on Autopilot function as a case study, UniSTPA uncovers multi-stage
hazards overlooked by conventional approaches including scene design defects,
sensor fusion biases, and internal model flaws, through multi-level causal
analysis, traces these hazards to deeper issues such as data quality, network
architecture, and optimisation objectives. The analysis result are used to
construct a safety monitoring and safety response mechanism that supports
continuous improvement from hazard identification to system optimisation. The
proposed framework thus offers both theoretical and practical guidance for the
safe development and deployment of end-to-end autonomous driving systems.

### 7. [Towards a Science of Causal Interpretability in Deep Learning for Software Engineering](http://arxiv.org/pdf/2505.15023v1)

Authors: David N. Palacio

This dissertation addresses achieving causal interpretability in Deep
Learning for Software Engineering (DL4SE). While Neural Code Models (NCMs) show
strong performance in automating software tasks, their lack of transparency in
causal relationships between inputs and outputs limits full understanding of
their capabilities. To build trust in NCMs, researchers and practitioners must
explain code predictions. Associational interpretability, which identifies
correlations, is often insufficient for tasks requiring intervention and change
analysis. To address this, the dissertation introduces DoCode, a novel post hoc
interpretability method for NCMs. DoCode uses causal inference to provide
programming language-oriented explanations of model predictions. It follows a
four-step pipeline: modeling causal problems using Structural Causal Models
(SCMs), identifying the causal estimand, estimating effects with metrics like
Average Treatment Effect (ATE), and refuting effect estimates. Its framework is
extensible, with an example that reduces spurious correlations by grounding
explanations in programming language properties. A case study on deep code
generation across interpretability scenarios and various deep learning
architectures demonstrates DoCode's benefits. Results show NCMs' sensitivity to
code syntax changes and their ability to learn certain programming concepts
while minimizing confounding bias. The dissertation also examines associational
interpretability as a foundation, analyzing software information's causal
nature using tools like COMET and TraceXplainer for traceability. It highlights
the need to identify code confounders and offers practical guidelines for
applying causal interpretability to NCMs, contributing to more trustworthy AI
in software engineering.

### 8. [LogiCase: Effective Test Case Generation from Logical Description in Competitive Programming](http://arxiv.org/pdf/2505.15039v1)

Authors: Sicheol Sung, Aditi, Dogyu kim, Yo-Sub Han, Sang-Ki Ko

Automated Test Case Generation (ATCG) is crucial for evaluating software
reliability, particularly in competitive programming where robust algorithm
assessments depend on diverse and accurate test cases. However, existing ATCG
methods often fail to meet complex specifications or generate effective corner
cases, limiting their utility. In this work, we introduce Context-Free Grammars
with Counters (CCFGs), a formalism that captures both syntactic and semantic
structures in input specifications. Using a fine-tuned CodeT5 model, we
translate natural language input specifications into CCFGs, enabling the
systematic generation of high-quality test cases. Experiments on the
CodeContests dataset demonstrate that CCFG-based test cases outperform baseline
methods in identifying incorrect algorithms, achieving significant gains in
validity and effectiveness. Our approach provides a scalable and reliable
grammar-driven framework for enhancing automated competitive programming
evaluations.

### 9. [An Empirical Analysis of EOS Blockchain: Architecture, Contract, and Security](http://arxiv.org/pdf/2505.15051v1)

Authors: Haiyang Liu, Yingjie Mao, Xiaoqi Li

With the rapid development of blockchain technology, various blockchain
systems are exhibiting vitality and potential. As a representative of
Blockchain 3.0, the EOS blockchain has been regarded as a strong competitor to
Ethereum. Nevertheless, compared with Bitcoin and Ethereum, academic research
and in-depth analyses of EOS remain scarce. To address this gap, this study
conducts a comprehensive investigation of the EOS blockchain from five key
dimensions: system architecture, decentralization, performance, smart
contracts, and behavioral security. The architectural analysis focuses on six
core components of the EOS system, detailing their functionalities and
operational workflows. The decentralization and performance evaluations, based
on data from the XBlock data-sharing platform, reveal several critical issues:
low account activity, limited participation in the supernode election process,
minimal variation in the set of block producers, and a substantial gap between
actual throughput and the claimed million-level performance. Five types of
contract vulnerabilities are identified in the smart contract dimension, and
four mainstream vulnerability detection platforms are introduced and
comparatively analyzed. In terms of behavioral security, four real-world
attacks targeting the structural characteristics of EOS are summarized. This
study contributes to the ongoing development of the EOS blockchain and provides
valuable insights for enhancing the security and regulatory mechanisms of
blockchain ecosystems.

### 10. [Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects](http://arxiv.org/pdf/2505.15088v1)

Authors: Yuxuan Wang, Jingshu Chen, Qingyang Wang

Command injection vulnerabilities are a significant security threat in
dynamic languages like Python, particularly in widely used open-source projects
where security issues can have extensive impact. With the proven effectiveness
of Large Language Models(LLMs) in code-related tasks, such as testing,
researchers have explored their potential for vulnerabilities analysis. This
study evaluates the potential of large language models (LLMs), such as GPT-4,
as an alternative approach for automated testing for vulnerability detection.
In particular, LLMs have demonstrated advanced contextual understanding and
adaptability, making them promising candidates for identifying nuanced security
vulnerabilities within code. To evaluate this potential, we applied LLM-based
analysis to six high-profile GitHub projects-Django, Flask, TensorFlow,
Scikit-learn, PyTorch, and Langchain-each with over 50,000 stars and extensive
adoption across software development and academic research. Our analysis
assesses both the strengths and limitations of LLMs in detecting command
injection vulnerabilities, evaluating factors such as detection accuracy,
efficiency, and practical integration into development workflows. In addition,
we provide a comparative analysis of different LLM tools to identify those most
suitable for security applications. Our findings offer guidance for developers
and security researchers on leveraging LLMs as innovative and automated
approaches to enhance software security.

### Social and Information Networks

### 1. [Prediction of Reposting on X](http://arxiv.org/pdf/2505.15370v1)

Authors: Ziming Xu, Shi Zhou, Vasileios Lampos, Ingemar J. Cox

There have been considerable efforts to predict a user's reposting behaviour
on X (formerly Twitter) using machine learning models. The problem is
previously cast as a supervised classification task, where Tweets are randomly
assigned to a test or training set. The random assignment helps to ensure that
the test and training sets are drawn from the same distribution. In practice,
we would like to predict users' reposting behaviour for a set of messages
related to a new, previously unseen, topic (defined by a hashtag). In this
case, the problem becomes an out-of-distribution generalisation classification
task.
  Experimental results reveal that while existing algorithms, which
predominantly use features derived from the content of Tweet messages, perform
well when the training and test distributions are the same, these algorithms
perform much worse when the test set is out of distribution. We then show that
if the message features are supplemented or replaced with features derived from
users' profile and past behaviour, the out-of-distribution prediction is
greatly improved, with the F1 score increasing from 0.24 to 0.70. Our
experimental results suggest that a significant component of reposting
behaviour can be predicted based on users' profile and past behaviour, and is
independent of the content of messages.

### 2. [Maximum Degree-Based Quasi-Clique Search via an Iterative Framework](http://arxiv.org/pdf/2505.15118v1)

Authors: Hongbo Xia, Kaiqiang Yu, Shengxin Liu, Cheng Long, Xun Zhou

Cohesive subgraph mining is a fundamental problem in graph theory with
numerous real-world applications, such as social network analysis and
protein-protein interaction modeling. Among various cohesive subgraphs, the
$\gamma$-quasi-clique is widely studied for its flexibility in requiring each
vertex to connect to at least a $\gamma$ proportion of other vertices in the
subgraph. However, solving the maximum $\gamma$-quasi-clique problem is NP-hard
and further complicated by the lack of the hereditary property, which makes
designing efficient pruning strategies challenging. Existing algorithms, such
as DDA and FastQC, either struggle with scalability or exhibit significant
performance declines for small values of $\gamma$. In this paper, we propose a
novel algorithm, IterQC, which reformulates the maximum $\gamma$-quasi-clique
problem as a series of $k$-plex problems that possess the hereditary property.
IterQC introduces a non-trivial iterative framework and incorporates two key
optimization techniques: (1) the pseudo lower bound (pseudo LB) technique,
which leverages information across iterations to improve the efficiency of
branch-and-bound searches, and (2) the preprocessing technique that reduces
problem size and unnecessary iterations. Extensive experiments demonstrate that
IterQC achieves up to four orders of magnitude speedup and solves significantly
more graph instances compared to state-of-the-art algorithms DDA and FastQC.

### 3. [Impact of Distance on Epidemiological Dynamics in Human Connection Network with Mobility](http://arxiv.org/pdf/2505.15331v1)

Authors: Md. Arquam, Suchi Kumari, Utkarsh Tiwari, Mohammad Al-saffar

The spread of infectious diseases is often influenced by human mobility
across different geographical regions. Although numerous studies have
investigated how diseases like SARS and COVID-19 spread from China to various
global locations, there remains a gap in understanding how the movement of
individuals contributes to disease transmission on a more personal or
human-to-human level. Typically, researchers have employed the concept of
metapopulation movement to analyze how diseases move from one location to
another. This paper shifts focus to the dynamics of disease transmission,
incorporating the critical factor of distance between an infected person and a
healthy individual during human movement. The study delves into the impact of
distance on various parameters of epidemiological dynamics throughout human
mobility. Mathematical expressions for important epidemiological metrics, such
as the basic reproduction number ($R_0$) and the critical infection rate
($\beta_{critical}$), are derived in relation to the distance between
individuals. The results indicate that the proposed model closely aligns with
observed patterns of COVID-19 spread based on the analysis done on the
available datasets.

### 4. [Graph Foundation Models: A Comprehensive Survey](http://arxiv.org/pdf/2505.15116v1)

Authors: Zehong Wang, Zheyuan Liu, Tianyi Ma, Jiazheng Li, Zheyuan Zhang, Xingbo Fu, Yiyang Li, Zhengqing Yuan, Wei Song, Yijun Ma, Qingkai Zeng, Xiusi Chen, Jianan Zhao, Jundong Li, Meng Jiang, Pietro Lio, Nitesh Chawla, Chuxu Zhang, Yanfang Ye

Graph-structured data pervades domains such as social networks, biological
systems, knowledge graphs, and recommender systems. While foundation models
have transformed natural language processing, vision, and multimodal learning
through large-scale pretraining and generalization, extending these
capabilities to graphs -- characterized by non-Euclidean structures and complex
relational semantics -- poses unique challenges and opens new opportunities. To
this end, Graph Foundation Models (GFMs) aim to bring scalable, general-purpose
intelligence to structured data, enabling broad transfer across graph-centric
tasks and domains. This survey provides a comprehensive overview of GFMs,
unifying diverse efforts under a modular framework comprising three key
components: backbone architectures, pretraining strategies, and adaptation
mechanisms. We categorize GFMs by their generalization scope -- universal,
task-specific, and domain-specific -- and review representative methods, key
innovations, and theoretical insights within each category. Beyond methodology,
we examine theoretical foundations including transferability and emergent
capabilities, and highlight key challenges such as structural alignment,
heterogeneity, scalability, and evaluation. Positioned at the intersection of
graph learning and general-purpose AI, GFMs are poised to become foundational
infrastructure for open-ended reasoning over structured data. This survey
consolidates current progress and outlines future directions to guide research
in this rapidly evolving field. Resources are available at
https://github.com/Zehong-Wang/Awesome-Foundation-Models-on-Graphs.

### Systems and Control

### 1. [Green Hacks: Generating Sustainability-Targeting Attacks For Cyber-Physical Systems](http://arxiv.org/pdf/2505.14982v1)

Authors: Faysal Ahamed, Tanushree Roy

Sustainability-targeting attacks (STA) or "Green Hacks" are a growing threat
to cyber-physical system (CPS)-based infrastructure, as its performance
objectives are increasingly linked to sustainability goals. These attacks
exploit the interdependence between control, energy efficiency, and
environmental impact to degrade systems' overall performance. Thus, in this
work, we propose a general mathematical framework for modeling such STA and
derive the feasibility conditions for generating a worst-case STA on a linear
CPS using a max-min formulation. A gradient ascent descent algorithm is used to
construct the worst-case attack policy. We simulated the worst-case STA for a
linear CPS to illustrate its impacts on the CPS performance and sustainability
cost.

### 2. [Co-optimize condenser water temperature and cooling tower fan using high-fidelity synthetic data](http://arxiv.org/pdf/2505.15041v1)

Authors: Gulai Shen, Gurpreet Singh, Ali Mehmani

This paper introduces a novel method for optimizing HVAC systems in buildings
by integrating a high-fidelity physics-based simulation model with machine
learning and measured data. The method enables a real-time building advisory
system that provides optimized settings for condenser water loop operation,
assisting building operators in decision-making. The building and its HVAC
system are first modeled using eQuest. Synthetic data are then generated by
running the simulation multiple times. The data are then processed, cleaned,
and used to train the machine learning model. The machine learning model
enables real-time optimization of the condenser water loop using particle swarm
optimization. The results deliver both a real-time online optimizer and an
offline operation look-up table, providing optimized condenser water
temperature settings and the optimal number of cooling tower fans at a given
cooling load. Potential savings are calculated by comparing measured data from
two summer months with the energy costs the building would have experienced
under optimized settings. Adaptive model refinement is applied to further
improve accuracy and effectiveness by utilizing available measured data. The
method bridges the gap between simulation and real-time control. It has the
potential to be applied to other building systems, including the chilled water
loop, heating systems, ventilation systems, and other related processes.
Combining physics models, data models, and measured data also enables
performance analysis, tracking, and retrofit recommendations.

### 3. [A Neural Network Approach to a Modified Quadratic Boost Multiport Resonant Converter for Electric Vehicle Chargers](http://arxiv.org/pdf/2505.15086v1)

Authors: V. Rajeswari, Nalin Kant Mohanty

This topology can achieve a high step-up gain by utilizing a switched
capacitor and switched inductor-based VMC network arrangement.Furthermore, the
proposed topology can achieve an output gain of approximately three times at a
nominal duty ratio with reduced voltage and current stress across the switch,
and enhance the maximum efficiency to 96.7

### 4. [Comparing Parameterizations and Objective Functions for Maximizing the Volume of Zonotopic Invariant Sets](http://arxiv.org/pdf/2505.15109v1)

Authors: Chenliang Zhou, Heejin Ahn, Ian M. Mitchell

In formal safety verification, many proposed algorithms use parametric set
representations and convert the computation of the relevant sets into an
optimization problem; consequently, the choice of parameterization and
objective function have a significant impact on the efficiency and accuracy of
the resulting computation. In particular, recent papers have explored the use
of zonotope set representations for various types of invariant sets. In this
paper we collect two zonotope parameterizations that are numerically
well-behaved and demonstrate that the volume of the corresponding zonotopes is
log-concave in the parameters. We then experimentally explore the use of these
two parameterizations in an algorithm for computing the maximum volume zonotope
invariant under affine dynamics within a specified box constraint over a finite
horizon. The true volume of the zonotopes is used as an objective function,
along with two alternative heuristics that are faster to compute. We conclude
that the heuristics are much faster in practice, although the relative quality
of their results declines as the dimension of the problem increases; however,
our conclusions are only preliminary due to so-far limited availability of
compute resources.

### 5. [A Risk-Based Probabilistic Transient Stability Approach for Ranking of Circuit Breakers in a Power System](http://arxiv.org/pdf/2505.15374v1)

Authors: Umair Shahzad

Power systems are getting more complex than ever and are consequently
operating close to their limit of stability. Moreover, with the increasing
demand of renewable wind generation, and the requirement to maintain a secure
power system, the importance of transient stability cannot be overestimated.
Current deterministic industry practices of transient stability assessment
ignore the probability of variables involved. With increasing system
uncertainties and widespread electricity market deregulation, there is a strong
inevitability to incorporate probabilistic transient stability analysis.
Circuit breakers play a critical role in fault clearing and consequently in
determining the system transient stability. It is important that they undergo
timely and appropriate maintenance procedures based on some criterion.
Considering the need of incorporating risk in modern power systems, this paper
proposes a risk-based probabilistic transient stability approach for ranking of
circuit breakers in a power system. A novel priority index was proposed to rank
the circuit breakers based on the system transient stability risk. DIgSILENT
PowerFactory software was used to conduct the required simulations on IEEE 14
bus system. The proposed risk-based framework was deemed to be efficient in
identification of the circuit breakers based on their priority rank index which
can aid in power system planning process.

### 6. [Impact of Wind Generation on Risk-based Security Assessment of Power System](http://arxiv.org/pdf/2505.15388v1)

Authors: Umair Shahzad

The electric power system is one of the largest and most intricate
infrastructures. Therefore, it is critical to assess and maintain its security.
A power system security assessment is indispensable for identifying
post-contingency issues, taking corrective measures, and protecting the system
from blackouts. This paper examined the impact of wind generation on the
risk-based security assessment of a power transmission network in the context
of planning. DIgSILENT PowerFactory software was used to conduct the analysis
using a combination of the brute force technique and the nonsequential Monte
Carlo (MC) simulation method on the IEEE 39-bus transmission test system.
Optimal power flow (OPF) was used to quantify security, considering (N-1),
(N-2), and (N-3) line outages and an (N-1) bus outage. Moreover, the average
cost deviation from the mean optimal system operating cost was proposed as a
novel security indicator. The results obtianed accurately depicted the effects
of changing wind generation levels on system security in terms of risk. The
most and least critical line(s) and bus in the system, for different wind
generation levels, were also determined. Moreover, the worst-case
wind-generation threshold level using two different cost functions for wind was
identified.

### 7. [From learning to safety: A Direct Data-Driven Framework for Constrained Control](http://arxiv.org/pdf/2505.15515v1)

Authors: Kanghui He, Shengling Shi, Ton van den Boom, Bart De Schutter

Ensuring safety in the sense of constraint satisfaction for learning-based
control is a critical challenge, especially in the model-free case. While
safety filters address this challenge in the model-based setting by modifying
unsafe control inputs, they typically rely on predictive models derived from
physics or data. This reliance limits their applicability for advanced
model-free learning control methods. To address this gap, we propose a new
optimization-based control framework that determines safe control inputs
directly from data. The benefit of the framework is that it can be updated
through arbitrary model-free learning algorithms to pursue optimal performance.
As a key component, the concept of direct data-driven safety filters (3DSF) is
first proposed. The framework employs a novel safety certificate, called the
state-action control barrier function (SACBF). We present three different
schemes to learn the SACBF. Furthermore, based on input-to-state safety
analysis, we present the error-to-state safety analysis framework, which
provides formal guarantees on safety and recursive feasibility even in the
presence of learning inaccuracies. The proposed control framework bridges the
gap between model-free learning-based control and constrained control, by
decoupling performance optimization from safety enforcement. Simulations on
vehicle control illustrate the superior performance regarding constraint
satisfaction and task achievement compared to model-based methods and reward
shaping.

### 8. [SpanTrain: Highly Efficient Cross-domain Model Distributed Training System under Heterogeneous GPUs and Networks in CEE Environment](http://arxiv.org/pdf/2505.15536v1)

Authors: Jinquan Wang, Xiaojian Liao, Xuzhao Liu, Jiashun Suo, Zhisheng Huo, Chenhao Zhang, Xiangrong Xu, Runnan Shen, Xilong Xie, Limin Xiao

Most existing training systems focus on a single region. In contrast, we
envision that cross-region training offers more flexible GPU resource
allocation and yields significant potential. However, the hierarchical cluster
topology and unstable networks in the cloud-edge-end (CEE) environment, a
typical cross-region scenario, pose substantial challenges to building an
efficient and autonomous model training system. We propose SpanTrain, a
geo-distributed model training system tailored for heterogeneous GPUs and
networks in CEE environments. SpanTrain adopts a communication-centric design
philosophy to tackle challenges arising from slow and unstable inter-region
networks. It begins with a heterogeneous device profiler that identifies and
groups devices based on both network and compute characteristics. Leveraging
device groups, SpanTrain implements compact, zero-bubble pipeline parallelism,
automatically deriving optimal parallel strategies. To further adapt to runtime
variability, SpanTrain integrates a dynamic environment adapter that reacts to
network fluctuations. Extensive evaluations demonstrate that SpanTrain achieves
1.3-2.8x higher training throughput compared to widely used and SOTA training
systems.

### 9. [Decreasing Utilization of Systems with Multi-Rate Cause-Effect Chains While Reducing End-to-End Latencies](http://arxiv.org/pdf/2505.15546v1)

Authors: Luiz Maia, Gerhard Fohler

The Logical Execution Time (LET) model has deterministic properties which
dramatically reduce the complexity of analyzing temporal requirements of
multi-rate cause-effect chains. The configuration (length and position) of
task's communication intervals directly define which task instances propagate
data through the chain and affect end-to-end latencies. Since not all task
instances propagate data through the chain, the execution of these instances
wastes processing resources. By manipulating the configuration of communication
intervals, it is possible to control which task instances are relevant for data
propagation and end-to-end latencies. However, since tasks can belong to more
than one cause-effect chain, the problem of configuring communication intervals
becomes non-trivial given the large number of possible configurations. In this
paper, we present a method to decrease the waste of processing resources while
reducing end-to-end latencies. We use a search algorithm to analyze different
communication interval configurations and find the combination that best
decrease system utilization while reducing end-to-end latencies. By controlling
data propagation by means of precedence constraints, our method modifies
communication intervals and controls which task instances affect end-to-end
latencies. Despite the sporadic release time of some task instances during the
analysis, our method transforms those instances into periodic tasks. We
evaluate our work using synthetic task sets and the automotive benchmark
proposed by BOSCH for the WATERS industrial challenge.

### 10. [Path Planning Algorithm Comparison Analysis for Wireless AUVs Energy Sharing System](http://arxiv.org/pdf/2505.15686v1)

Authors: Zhengji Feng, Hengxiang Chen, Liqun Chen, Heyan Li, Xiaolin Mou

Autonomous underwater vehicles (AUVs) are increasingly used in marine
research, military applications, and undersea exploration. However, their
operational range is significantly affected by battery performance. In this
paper, a framework for a wireless energy sharing system among AUVs is proposed,
enabling rapid energy replenishment. Path planning plays a crucial role in the
energy-sharing process and autonomous navigation, as it must generate feasible
trajectories toward designated goals. This article focuses on efficient
obstacle avoidance in complex underwater environments, including irregularly
shaped obstacles and narrow passages. The proposed method combines
Rapidly-exploring Random Trees Star (RRT*) with Particle Swarm Optimization
(PSO) to improve path planning efficiency. Comparative analysis of the two
algorithms is presented through simulation results in both random and irregular
obstacle environments. Index Terms: Wireless charging, autonomous underwater
vehicles (AUVs), path planning, irregular obstacles, narrow passages, RRT*,
particle swarm optimization (PSO).

### Machine Learning (Statistics Category)

### 1. [Pre-validation Revisited](http://arxiv.org/pdf/2505.14985v2)

Authors: Jing Shang, Sourav Chatterjee, Trevor Hastie, Robert Tibshirani

Pre-validation is a way to build prediction model with two datasets of
significantly different feature dimensions. Previous work showed that the
asymptotic distribution of the resulting test statistic for the pre-validated
predictor deviates from a standard Normal, hence leads to issues in hypothesis
testing. In this paper, we revisit the pre-validation procedure and extend the
problem formulation without any independence assumption on the two feature
sets. We propose not only an analytical distribution of the test statistic for
the pre-validated predictor under certain models, but also a generic bootstrap
procedure to conduct inference. We show properties and benefits of
pre-validation in prediction, inference and error estimation by simulations and
applications, including analysis of a breast cancer study and a synthetic GWAS
example.

### 2. [Convergence of Adam in Deep ReLU Networks via Directional Complexity and Kakeya Bounds](http://arxiv.org/pdf/2505.15013v1)

Authors: Anupama Sridhar, Alexander Johansen

First-order adaptive optimization methods like Adam are the default choices
for training modern deep neural networks. Despite their empirical success, the
theoretical understanding of these methods in non-smooth settings, particularly
in Deep ReLU networks, remains limited. ReLU activations create exponentially
many region boundaries where standard smoothness assumptions break down.
\textbf{We derive the first
\(\tilde{O}\!\bigl(\sqrt{d_{\mathrm{eff}}/n}\bigr)\) generalization bound for
Adam in Deep ReLU networks and the first global-optimal convergence for Adam in
the non smooth, non convex relu landscape without a global PL or convexity
assumption.} Our analysis is based on stratified Morse theory and novel results
in Kakeya sets. We develop a multi-layer refinement framework that
progressively tightens bounds on region crossings. We prove that the number of
region crossings collapses from exponential to near-linear in the effective
dimension. Using a Kakeya based method, we give a tighter generalization bound
than PAC-Bayes approaches and showcase convergence using a mild uniform low
barrier assumption.

### 3. [Infinite hierarchical contrastive clustering for personal digital envirotyping](http://arxiv.org/pdf/2505.15022v1)

Authors: Ya-Yun Huang, Joseph McClernon, Jason A. Oliver, Matthew M. Engelhard

Daily environments have profound influence on our health and behavior. Recent
work has shown that digital envirotyping, where computer vision is applied to
images of daily environments taken during ecological momentary assessment
(EMA), can be used to identify meaningful relationships between environmental
features and health outcomes of interest. To systematically study such effects
on an individual level, it is helpful to group images into distinct
environments encountered in an individual's daily life; these may then be
analyzed, further grouped into related environments with similar features, and
linked to health outcomes. Here we introduce infinite hierarchical contrastive
clustering to address this challenge. Building on the established contrastive
clustering framework, our method a) allows an arbitrary number of clusters
without requiring the full Dirichlet Process machinery by placing a
stick-breaking prior on predicted cluster probabilities; and b) encourages
distinct environments to form well-defined sub-clusters within each cluster of
related environments by incorporating a participant-specific prediction loss.
Our experiments show that our model effectively identifies distinct personal
environments and groups these environments into meaningful environment types.
We then illustrate how the resulting clusters can be linked to various health
outcomes, highlighting the potential of our approach to advance the
envirotyping paradigm.

### 4. [Reconstruction of Graph Signals on Complex Manifolds with Kernel Methods](http://arxiv.org/pdf/2505.15202v1)

Authors: Yu Zhang, Linyu Peng, Bing-Zhao Li

Graph signals are widely used to describe vertex attributes or features in
graph-structured data, with applications spanning the internet, social media,
transportation, sensor networks, and biomedicine. Graph signal processing (GSP)
has emerged to facilitate the analysis, processing, and sampling of such
signals. While kernel methods have been extensively studied for estimating
graph signals from samples provided on a subset of vertices, their application
to complex-valued graph signals remains largely unexplored. This paper
introduces a novel framework for reconstructing graph signals using kernel
methods on complex manifolds. By embedding graph vertices into a
higher-dimensional complex ambient space that approximates a lower-dimensional
manifold, the framework extends the reproducing kernel Hilbert space to complex
manifolds. It leverages Hermitian metrics and geometric measures to
characterize kernels and graph signals. Additionally, several traditional
kernels and graph topology-driven kernels are proposed for reconstructing
complex graph signals. Finally, experimental results on synthetic and
real-world datasets demonstrate the effectiveness of this framework in
accurately reconstructing complex graph signals, outperforming conventional
kernel-based approaches. This work lays a foundational basis for integrating
complex geometry and kernel methods in GSP.

### 5. [Human in the Loop Adaptive Optimization for Improved Time Series Forecasting](http://arxiv.org/pdf/2505.15354v1)

Authors: Malik Tiomoko, Hamza Cherkaoui, Giuseppe Paolo, Zhang Yili, Yu Meng, Zhang Keli, Hafiz Tiomoko Ali

Time series forecasting models often produce systematic, predictable errors
even in critical domains such as energy, finance, and healthcare. We introduce
a novel post training adaptive optimization framework that improves forecast
accuracy without retraining or architectural changes. Our method automatically
applies expressive transformations optimized via reinforcement learning,
contextual bandits, or genetic algorithms to correct model outputs in a
lightweight and model agnostic way. Theoretically, we prove that affine
corrections always reduce the mean squared error; practically, we extend this
idea with dynamic action based optimization. The framework also supports an
optional human in the loop component: domain experts can guide corrections
using natural language, which is parsed into actions by a language model.
Across multiple benchmarks (e.g., electricity, weather, traffic), we observe
consistent accuracy gains with minimal computational overhead. Our interactive
demo shows the framework's real time usability. By combining automated post hoc
refinement with interpretable and extensible mechanisms, our approach offers a
powerful new direction for practical forecasting systems.

### 6. [Robust Multimodal Learning via Entropy-Gated Contrastive Fusion](http://arxiv.org/pdf/2505.15417v1)

Authors: Leon Chlon, Maggie Chlon, MarcAntonio M. Awada

Real-world multimodal systems routinely face missing-input scenarios, and in
reality, robots lose audio in a factory or a clinical record omits lab tests at
inference time. Standard fusion layers either preserve robustness or
calibration but never both. We introduce Adaptive Entropy-Gated Contrastive
Fusion (AECF), a single light-weight layer that (i) adapts its entropy
coefficient per instance, (ii) enforces monotone calibration across all
modality subsets, and (iii) drives a curriculum mask directly from
training-time entropy. On AV-MNIST and MS-COCO, AECF improves masked-input mAP
by +18 pp at a 50% drop rate while reducing ECE by up to 200%, yet adds 1%
run-time. All back-bones remain frozen, making AECF an easy drop-in layer for
robust, calibrated multimodal inference.

### 7. [Adaptive Temperature Scaling with Conformal Prediction](http://arxiv.org/pdf/2505.15437v1)

Authors: Nikita Kotelevskii, Mohsen Guizani, Eric Moulines, Maxim Panov

Conformal prediction enables the construction of high-coverage prediction
sets for any pre-trained model, guaranteeing that the true label lies within
the set with a specified probability. However, these sets do not provide
probability estimates for individual labels, limiting their practical use. In
this paper, we propose, to the best of our knowledge, the first method for
assigning calibrated probabilities to elements of a conformal prediction set.
Our approach frames this as an adaptive calibration problem, selecting an
input-specific temperature parameter to match the desired coverage level.
Experiments on several challenging image classification datasets demonstrate
that our method maintains coverage guarantees while significantly reducing
expected calibration error.

### 8. [AdUE: Improving uncertainty estimation head for LoRA adapters in LLMs](http://arxiv.org/pdf/2505.15443v1)

Authors: Artem Zabolotnyi, Roman Makarov, Mile Mitrovic, Polina Proskura, Oleg Travkin, Roman Alferov, Alexey Zaytsev

Uncertainty estimation remains a critical challenge in adapting pre-trained
language models to classification tasks, particularly under parameter-efficient
fine-tuning approaches such as adapters. We introduce AdUE1, an efficient
post-hoc uncertainty estimation (UE) method, to enhance softmax-based
estimates. Our approach (1) uses a differentiable approximation of the maximum
function and (2) applies additional regularization through L2-SP, anchoring the
fine-tuned head weights and regularizing the model. Evaluations on five NLP
classification datasets across four language models (RoBERTa, ELECTRA, LLaMA-2,
Qwen) demonstrate that our method consistently outperforms established
baselines such as Mahalanobis distance and softmax response. Our approach is
lightweight (no base-model changes) and produces better-calibrated confidence.

### 9. [Fast Rate Bounds for Multi-Task and Meta-Learning with Different Sample Sizes](http://arxiv.org/pdf/2505.15496v1)

Authors: Hossein Zakerinia, Christoph H. Lampert

We present new fast-rate generalization bounds for multi-task and
meta-learning in the unbalanced setting, i.e. when the tasks have training sets
of different sizes, as is typically the case in real-world scenarios.
Previously, only standard-rate bounds were known for this situation, while
fast-rate bounds were limited to the setting where all training sets are of
equal size. Our new bounds are numerically computable as well as interpretable,
and we demonstrate their flexibility in handling a number of cases where they
give stronger guarantees than previous bounds. Besides the bounds themselves,
we also make conceptual contributions: we demonstrate that the unbalanced
multi-task setting has different statistical properties than the balanced
situation, specifically that proofs from the balanced situation do not carry
over to the unbalanced setting. Additionally, we shed light on the fact that
the unbalanced situation allows two meaningful definitions of multi-task risk,
depending on whether if all tasks should be considered equally important or if
sample-rich tasks should receive more weight than sample-poor ones.

### 10. [Federated Learning with Unlabeled Clients: Personalization Can Happen in Low Dimensions](http://arxiv.org/pdf/2505.15579v1)

Authors: Hossein Zakerinia, Jonathan Scott, Christoph H. Lampert

Personalized federated learning has emerged as a popular approach to training
on devices holding statistically heterogeneous data, known as clients. However,
most existing approaches require a client to have labeled data for training or
finetuning in order to obtain their own personalized model. In this paper we
address this by proposing FLowDUP, a novel method that is able to generate a
personalized model using only a forward pass with unlabeled data. The generated
model parameters reside in a low-dimensional subspace, enabling efficient
communication and computation. FLowDUP's learning objective is theoretically
motivated by our new transductive multi-task PAC-Bayesian generalization bound,
that provides performance guarantees for unlabeled clients. The objective is
structured in such a way that it allows both clients with labeled data and
clients with only unlabeled data to contribute to the training process. To
supplement our theoretical results we carry out a thorough experimental
evaluation of FLowDUP, demonstrating strong empirical performance on a range of
datasets with differing sorts of statistically heterogeneous clients. Through
numerous ablation studies, we test the efficacy of the individual components of
the method.

