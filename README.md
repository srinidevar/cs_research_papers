# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-31 17:00:25.491632 PST.

### Artificial Intelligence

### 1. [GUI Knowledge Bench: Revealing the Knowledge Gap Behind VLM Failures in GUI Tasks](http://arxiv.org/pdf/2510.26098v1)

Authors: Chenrui Shi, Zedong Yu, Zhi Gao, Ruining Feng, Enqi Liu, Yuwei Wu, Yunde Jia, Liuyu Xiang, Zhaofeng He, Qing Li

Large vision language models (VLMs) have advanced graphical user interface
(GUI) task automation but still lag behind humans. We hypothesize this gap
stems from missing core GUI knowledge, which existing training schemes (such as
supervised fine tuning and reinforcement learning) alone cannot fully address.
By analyzing common failure patterns in GUI task execution, we distill GUI
knowledge into three dimensions: (1) interface perception, knowledge about
recognizing widgets and system states; (2) interaction prediction, knowledge
about reasoning action state transitions; and (3) instruction understanding,
knowledge about planning, verifying, and assessing task completion progress. We
further introduce GUI Knowledge Bench, a benchmark with multiple choice and
yes/no questions across six platforms (Web, Android, MacOS, Windows, Linux,
IOS) and 292 applications. Our evaluation shows that current VLMs identify
widget functions but struggle with perceiving system states, predicting
actions, and verifying task completion. Experiments on real world GUI tasks
further validate the close link between GUI knowledge and task success. By
providing a structured framework for assessing GUI knowledge, our work supports
the selection of VLMs with greater potential prior to downstream training and
provides insights for building more capable GUI agents.

### 2. [Beyond Benchmarks: The Economics of AI Inference](http://arxiv.org/pdf/2510.26136v1)

Authors: Boqin Zhuang, Jiacheng Qiao, Mingqian Liu, Mingxing Yu, Ping Hong, Rui Li, Xiaoxia Song, Xiangjun Xu, Xu Chen, Yaoyao Ma, Yujie Gao

The inference cost of Large Language Models (LLMs) has become a critical
factor in determining their commercial viability and widespread adoption. This
paper introduces a quantitative ``economics of inference'' framework, treating
the LLM inference process as a compute-driven intelligent production activity.
We analyze its marginal cost, economies of scale, and quality of output under
various performance configurations. Based on empirical data from WiNEval-3.0,
we construct the first ``LLM Inference Production Frontier,'' revealing three
principles: diminishing marginal cost, diminishing returns to scale, and an
optimal cost-effectiveness zone. This paper not only provides an economic basis
for model deployment decisions but also lays an empirical foundation for the
future market-based pricing and optimization of AI inference resources.

### 3. [The FM Agent](http://arxiv.org/pdf/2510.26144v1)

Authors: Annan Li, Chufan Wu, Zengle Ge, Yee Hin Chong, Zhinan Hou, Lizhe Cao, Cheng Ju, Jianmin Wu, Huaiming Li, Haobo Zhang, Shenghao Feng, Mo Zhao, Fengzhi Qiu, Rui Yang, Mengmeng Zhang, Wenyi Zhu, Yingying Sun, Quan Sun, Shunhao Yan, Danyu Liu, Dawei Yin, Dou Shen

Large language models (LLMs) are catalyzing the development of autonomous AI
research agents for scientific and engineering discovery. We present FM Agent,
a novel and general-purpose multi-agent framework that leverages a synergistic
combination of LLM-based reasoning and large-scale evolutionary search to
address complex real-world challenges. The core of FM Agent integrates several
key innovations: 1) a cold-start initialization phase incorporating expert
guidance, 2) a novel evolutionary sampling strategy for iterative optimization,
3) domain-specific evaluators that combine correctness, effectiveness, and
LLM-supervised feedback, and 4) a distributed, asynchronous execution
infrastructure built on Ray. Demonstrating broad applicability, our system has
been evaluated across diverse domains, including operations research, machine
learning, GPU kernel optimization, and classical mathematical problems. FM
Agent reaches state-of-the-art results autonomously, without human
interpretation or tuning -- 1976.3 on ALE-Bench (+5.2\%), 43.56\% on MLE-Bench
(+4.0pp), up to 20x speedups on KernelBench, and establishes new
state-of-the-art(SOTA) results on several classical mathematical problems.
Beyond academic benchmarks, FM Agent shows considerable promise for both
large-scale enterprise R\&D workflows and fundamental scientific research,
where it can accelerate innovation, automate complex discovery processes, and
deliver substantial engineering and scientific advances with broader societal
impact.

### 4. [Questionnaire meets LLM: A Benchmark and Empirical Study of Structural Skills for Understanding Questions and Responses](http://arxiv.org/pdf/2510.26238v1)

Authors: Duc-Hai Nguyen, Vijayakumar Nanjappan, Barry O'Sullivan, Hoang D. Nguyen

Millions of people take surveys every day, from market polls and academic
studies to medical questionnaires and customer feedback forms. These datasets
capture valuable insights, but their scale and structure present a unique
challenge for large language models (LLMs), which otherwise excel at few-shot
reasoning over open-ended text. Yet, their ability to process questionnaire
data or lists of questions crossed with hundreds of respondent rows remains
underexplored. Current retrieval and survey analysis tools (e.g., Qualtrics,
SPSS, REDCap) are typically designed for humans in the workflow, limiting such
data integration with LLM and AI-empowered automation. This gap leaves
scientists, surveyors, and everyday users without evidence-based guidance on
how to best represent questionnaires for LLM consumption. We address this by
introducing QASU (Questionnaire Analysis and Structural Understanding), a
benchmark that probes six structural skills, including answer lookup,
respondent count, and multi-hop inference, across six serialization formats and
multiple prompt strategies. Experiments on contemporary LLMs show that choosing
an effective format and prompt combination can improve accuracy by up to 8.8%
points compared to suboptimal formats. For specific tasks, carefully adding a
lightweight structural hint through self-augmented prompting can yield further
improvements of 3-4% points on average. By systematically isolating format and
prompting effects, our open source benchmark offers a simple yet versatile
foundation for advancing both research and real-world practice in LLM-based
questionnaire analysis.

### 5. [Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles](http://arxiv.org/pdf/2510.26242v1)

Authors: Xinhang Li, Qing Guo, Junyu Chen, Zheng Guo, Shengzhe Xu, Lei Li, Lin Zhang

With increasing urban traffic complexity, Traffic Signal Control (TSC) is
essential for optimizing traffic flow and improving road safety. Large Language
Models (LLMs) emerge as promising approaches for TSC. However, they are prone
to hallucinations in emergencies, leading to unreliable decisions that may
cause substantial delays for emergency vehicles. Moreover, diverse intersection
types present substantial challenges for traffic state encoding and
cross-intersection training, limiting generalization across heterogeneous
intersections. Therefore, this paper proposes Retrieval Augmented Generation
(RAG)-enhanced distributed LLM agents with Emergency response for Generalizable
TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning
framework, which dynamically adjusts reasoning depth based on the emergency
scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to
distill specific knowledge and guidance from historical cases, enhancing the
reliability and rationality of agents' emergency decisions. Secondly, this
paper designs a type-agnostic traffic representation and proposes a
Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3
adaptively samples training experience from diverse intersections with
environment feedback-based priority and fine-tunes LLM agents with a designed
reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies
across heterogeneous intersections. On three real-world road networks with 17
to 177 heterogeneous intersections, extensive experiments show that REG-TSC
reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle
waiting time by 83.16%, outperforming other state-of-the-art methods.

### 6. [Graph-Enhanced Policy Optimization in LLM Agent Training](http://arxiv.org/pdf/2510.26270v1)

Authors: Jiazhen Yuan, Wei Zhao, Zhengbiao Bai

Group based reinforcement learning (RL) has shown impressive results on
complex reasoning and mathematical tasks. Yet, when applied to train
multi-turn, interactive LLM agents, these methods often suffer from structural
blindness-the inability to exploit the underlying connectivity of the
environment. This manifests in three critical challenges: (1) inefficient,
unguided exploration, (2) imprecise credit assignment due to overlooking
pivotal states, and (3) myopic planning caused by static reward discounting. We
address these issues with Graph-Enhanced Policy Optimization (GEPO), which
dynamically constructs a state-transition graph from agent experience and
employs graph-theoretic centrality to provide three synergistic learning
signals: (1)structured intrinsic rewards that guide exploration toward
high-impact states, (2) a graph-enhanced advantage function for topology-aware
credit assignment, and (3) a dynamic discount factor adapted to each state's
strategic value. On the ALFWorld, WebShop, and a proprietary Workbench
benchmarks, GEPO demonstrates strong performance, achieving absolute success
rate gains of +4.1%, +5.3%, and +10.9% over competitive baselines. These
results highlight that explicitly modeling environmental structure is a robust,
generalizable strategy for advancing LLM agent training.

### 7. [Discovering State Equivalences in UCT Search Trees By Action Pruning](http://arxiv.org/pdf/2510.26346v1)

Authors: Robin Schmöcker, Alexander Dockhorn, Bodo Rosenhahn

One approach to enhance Monte Carlo Tree Search (MCTS) is to improve its
sample efficiency by grouping/abstracting states or state-action pairs and
sharing statistics within a group. Though state-action pair abstractions are
mostly easy to find in algorithms such as On the Go Abstractions in Upper
Confidence bounds applied to Trees (OGA-UCT), nearly no state abstractions are
found in either noisy or large action space settings due to constraining
conditions. We provide theoretical and empirical evidence for this claim, and
we slightly alleviate this state abstraction problem by proposing a weaker
state abstraction condition that trades a minor loss in accuracy for finding
many more abstractions. We name this technique Ideal Pruning Abstractions in
UCT (IPA-UCT), which outperforms OGA-UCT (and any of its derivatives) across a
large range of test domains and iteration budgets as experimentally validated.
IPA-UCT uses a different abstraction framework from Abstraction of State-Action
Pairs (ASAP) which is the one used by OGA-UCT, which we name IPA. Furthermore,
we show that both IPA and ASAP are special cases of a more general framework
that we call p-ASAP which itself is a special case of the ASASAP framework.

### 8. [BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](http://arxiv.org/pdf/2510.26374v1)

Authors: Qianli Shen, Daoyuan Chen, Yilun Huang, Zhenqing Ling, Yaliang Li, Bolin Ding, Jingren Zhou

Reinforcement finetuning (RFT) is a key technique for aligning Large Language
Models (LLMs) with human preferences and enhancing reasoning, yet its
effectiveness is highly sensitive to which tasks are explored during training.
Uniform task sampling is inefficient, wasting computation on tasks that are
either trivial or unsolvable, while existing task selection methods often
suffer from high rollout costs, poor adaptivity, or incomplete evidence. We
introduce \textbf{BOTS}, a unified framework for \textbf{B}ayesian
\textbf{O}nline \textbf{T}ask \textbf{S}election in LLM reinforcement
finetuning. Grounded in Bayesian inference, BOTS adaptively maintains posterior
estimates of task difficulty as the model evolves. It jointly incorporates
\emph{explicit evidence} from direct evaluations of selected tasks and
\emph{implicit evidence} inferred from these evaluations for unselected tasks,
with Thompson sampling ensuring a principled balance between exploration and
exploitation. To make implicit evidence practical, we instantiate it with an
ultra-light interpolation-based plug-in that estimates difficulties of
unevaluated tasks without extra rollouts, adding negligible overhead.
Empirically, across diverse domains and LLM scales, BOTS consistently improves
data efficiency and performance over baselines and ablations, providing a
practical and extensible solution for dynamic task selection in RFT.

### 9. [AI Mathematician as a Partner in Advancing Mathematical Discovery -- A Case Study in Homogenization Theory](http://arxiv.org/pdf/2510.26380v1)

Authors: Yuanhang Liu, Beichen Wang, Peng Li, Yang Liu

Artificial intelligence (AI) has demonstrated impressive progress in
mathematical reasoning, yet its integration into the practice of mathematical
research remains limited. In this study, we investigate how the AI
Mathematician (AIM) system can operate as a research partner rather than a mere
problem solver. Focusing on a challenging problem in homogenization theory, we
analyze the autonomous reasoning trajectories of AIM and incorporate targeted
human interventions to structure the discovery process. Through iterative
decomposition of the problem into tractable subgoals, selection of appropriate
analytical methods, and validation of intermediate results, we reveal how human
intuition and machine computation can complement one another. This
collaborative paradigm enhances the reliability, transparency, and
interpretability of the resulting proofs, while retaining human oversight for
formal rigor and correctness. The approach leads to a complete and verifiable
proof, and more broadly, demonstrates how systematic human-AI co-reasoning can
advance the frontier of mathematical discovery.

### 10. [A Pragmatic View of AI Personhood](http://arxiv.org/pdf/2510.26396v1)

Authors: Joel Z. Leibo, Alexander Sasha Vezhnevets, William A. Cunningham, Stanley M. Bileschi

The emergence of agentic Artificial Intelligence (AI) is set to trigger a
"Cambrian explosion" of new kinds of personhood. This paper proposes a
pragmatic framework for navigating this diversification by treating personhood
not as a metaphysical property to be discovered, but as a flexible bundle of
obligations (rights and responsibilities) that societies confer upon entities
for a variety of reasons, especially to solve concrete governance problems. We
argue that this traditional bundle can be unbundled, creating bespoke solutions
for different contexts. This will allow for the creation of practical tools --
such as facilitating AI contracting by creating a target "individual" that can
be sanctioned -- without needing to resolve intractable debates about an AI's
consciousness or rationality. We explore how individuals fit in to social roles
and discuss the use of decentralized digital identity technology, examining
both "personhood as a problem", where design choices can create "dark patterns"
that exploit human social heuristics, and "personhood as a solution", where
conferring a bundle of obligations is necessary to ensure accountability or
prevent conflict. By rejecting foundationalist quests for a single, essential
definition of personhood, this paper offers a more pragmatic and flexible way
to think about integrating AI agents into our society.

### Hardware Architecture

### 1. [MIREDO: MIP-Driven Resource-Efficient Dataflow Optimization for Computing-in-Memory Accelerator](http://arxiv.org/pdf/2510.26463v1)

Authors: Xiaolin He, Cenlin Duan, Yingjie Qi, Xiao Ma, Jianlei Yang

Computing-in-Memory (CIM) architectures have emerged as a promising solution
for accelerating Deep Neural Networks (DNNs) by mitigating data movement
bottlenecks. However, realizing the potential of CIM requires specialized
dataflow optimizations, which are challenged by an expansive design space and
strict architectural constraints. Existing optimization approaches often fail
to fully exploit CIM accelerators, leading to noticeable gaps between
theoretical and actual system-level efficiency. To address these limitations,
we propose the MIREDO framework, which formulates dataflow optimization as a
Mixed-Integer Programming (MIP) problem. MIREDO introduces a hierarchical
hardware abstraction coupled with an analytical latency model designed to
accurately reflect the complex data transfer behaviors within CIM systems. By
jointly modeling workload characteristics, dataflow strategies, and
CIM-specific constraints, MIREDO systematically navigates the vast design space
to determine the optimal dataflow configurations. Evaluation results
demonstrate that MIREDO significantly enhances performance, achieving up to
$3.2\times$ improvement across various DNN models and hardware setups.

### 2. [Wireless Sensor Networks as Parallel and Distributed Hardware Platform for Artificial Neural Networks](http://arxiv.org/pdf/2510.26492v1)

Authors: Gursel Serpen

We are proposing fully parallel and maximally distributed hardware
realization of a generic neuro-computing system. More specifically, the
proposal relates to the wireless sensor networks technology to serve as a
massively parallel and fully distributed hardware platform to implement and
realize artificial neural network (ANN) algorithms. A parallel and distributed
(PDP) hardware realization of ANNs makes it possible to have real time
computation of large-scale (and complex) problems in a highly robust framework.
We will demonstrate how a network of hundreds of thousands of processing nodes
(or motes of a wireless sensor network), which have on-board processing and
wireless communication features, can be used to implement fully parallel and
massively distributed computation of artificial neural network algorithms for
solution of truly large-scale problems in real time. The realization of
artificial neural network algorithms in a massively parallel and fully
distributed hardware has been the goal of neural network computing researchers.
This is because a parallel and distributed computation of artificial neural
network algorithms could not have been achieved against all the advancements in
silicon- or optics-based computing. Accordingly, artificial neural networks
could not be applied to very large-scale problems for real time computation of
solutions. This hindered the development of neural algorithms for affordable
and practical solutions of challenging problems since often special-purpose
computing approaches in hardware, software or hybrid (non-neural) had to be
developed for and fine-tuned to specific problems that are very large-scale and
highly complex. Successful implementation is likely to revolutionize computing
as we know it by making it possible to solve very large scale scientific,
engineering or technical problems in real time.

### Computational Complexity

### 1. [Limitation of Quantum Walk Approach to the Maximum Matching Problem](http://arxiv.org/pdf/2510.26246v1)

Authors: Alcides Gomes Andrade Júnior, Akira Matsubayashi

The Maximum Matching problem has a quantum query complexity lower bound of
$\Omega(n^{3/2})$ for graphs on $n$ vertices represented by an adjacency
matrix. The current best quantum algorithm has the query complexity
$O(n^{7/4})$, which is an improvement over the trivial bound $O(n^2)$.
Constructing a quantum algorithm for this problem with a query complexity
improving the upper bound $O(n^{7/4})$ is an open problem. The quantum walk
technique is a general framework for constructing quantum algorithms by
transforming a classical random walk search into a quantum search, and has been
successfully applied to constructing an algorithm with a tight query complexity
for another problem. In this work we show that the quantum walk technique fails
to produce a fast algorithm improving the known (or even the trivial) upper
bound on the query complexity. Specifically, if a quantum walk algorithm
designed with the known technique solves the Maximum Matching problem using
$O(n^{2-\epsilon})$ queries with any constant $\epsilon>0$, and if the
underlying classical random walk is independent of an input graph, then the
guaranteed time complexity is larger than any polynomial of $n$.

### 2. [Tensor decomposition beyond uniqueness, with an application to the minrank problem](http://arxiv.org/pdf/2510.26587v1)

Authors: Pascal Koiran, Rafael Oliveira

We prove a generalization to Jennrich's uniqueness theorem for tensor
decompositions in the undercomplete setting. Our uniqueness theorem is based on
an alternative definition of the standard tensor decomposition, which we call
matrix-vector decomposition. Moreover, in the same settings in which our
uniqueness theorem applies, we also design and analyze an efficient randomized
algorithm to compute the unique minimum matrix-vector decomposition (and thus a
tensor rank decomposition of minimum rank).
  As an application of our uniqueness theorem and our efficient algorithm, we
show how to compute all matrices of minimum rank (up to scalar multiples) in
certain generic vector spaces of matrices.

### Computational Engineering

### 1. [Reduced order modelling of Hopf bifurcations for the Navier-Stokes equations through invariant manifolds](http://arxiv.org/pdf/2510.26542v1)

Authors: Alessio Colombo, Alessandra Vizzaccaro, Cyril Touzé, André de F. Stabile, Luc Pastur, Attilio Frangi

This work introduces a parametric simulation-free reduced order model for
incompressible flows undergoing a Hopf bifurcation, leveraging the
parametrisation method for invariant manifolds. Unlike data-driven approaches,
this method operates directly on the governing equations, eliminating the need
for full-order simulations. The proposed model is computed at a single value of
the bifurcation parameter yet remains valid over a range of values. The
approach systematically constructs an invariant manifold and embedded dynamics,
providing an accurate and efficient reduction of the original system. The
ability to capture pre-critical steady states, the bifurcation point, and
post-critical limit cycle oscillations is demonstrated by a strong agreement
between the reduced order model and full order simulations, while achieving
significant computational speed-up.

### 2. [Learning to Manage Investment Portfolios beyond Simple Utility Functions](http://arxiv.org/pdf/2510.26165v1)

Authors: Maarten P. Scholl, Mahmoud Mahfouz, Anisoara Calinescu, J. Doyne Farmer

While investment funds publicly disclose their objectives in broad terms,
their managers optimize for complex combinations of competing goals that go
beyond simple risk-return trade-offs. Traditional approaches attempt to model
this through multi-objective utility functions, but face fundamental challenges
in specification and parameterization. We propose a generative framework that
learns latent representations of fund manager strategies without requiring
explicit utility specification.
  Our approach directly models the conditional probability of a fund's
portfolio weights, given stock characteristics, historical returns, previous
weights, and a latent variable representing the fund's strategy. Unlike methods
based on reinforcement learning or imitation learning, which require specified
rewards or labeled expert objectives, our GAN-based architecture learns
directly from the joint distribution of observed holdings and market data.
  We validate our framework on a dataset of 1436 U.S. equity mutual funds. The
learned representations successfully capture known investment styles, such as
"growth" and "value," while also revealing implicit manager objectives. For
instance, we find that while many funds exhibit characteristics of
Markowitz-like optimization, they do so with heterogeneous realizations for
turnover, concentration, and latent factors.
  To analyze and interpret the end-to-end model, we develop a series of tests
that explain the model, and we show that the benchmark's expert labeling are
contained in our model's encoding in a linear interpretable way.
  Our framework provides a data-driven approach for characterizing investment
strategies for applications in market simulation, strategy attribution, and
regulatory oversight.

### Computational Geometry

### 1. [Shortest Paths, Convexity, and Treewidth in Regular Hyperbolic Tilings](http://arxiv.org/pdf/2510.26110v1)

Authors: Sándor Kisfaludi-Bak, Tze-Yang Poon, Geert van Wordragen

Hyperbolic tilings are natural infinite planar graphs where each vertex has
degree $q$ and each face has $p$ edges for some $\frac1p+\frac1q<\frac12$. We
study the structure of shortest paths in such graphs. We show that given a set
of $n$ terminals, we can compute a so-called isometric closure (closely related
to the geodesic convex hull) of the terminals in near-linear time, using a
classic geometric convex hull algorithm as a black box. We show that the size
of the convex hull is $O(N)$ where $N$ is the total length of the paths to the
terminals from a fixed origin.
  Furthermore, we prove that the geodesic convex hull of a set of $n$ terminals
has treewidth only $\max(12,O(\log\frac{n}{p + q}))$, a bound independent of
the distance of the points involved. As a consequence, we obtain algorithms for
subset TSP and Steiner tree with running time $O(N \log N) +
\mathrm{poly}(\frac{n}{p + q}) \cdot N$.

### Computation and Language

### 1. [Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking](http://arxiv.org/pdf/2510.26122v1)

Authors: Feng Ju, Zeyu Qin, Rui Min, Zhitao He, Lingpeng Kong, Yi R. Fung

While Test-Time Scaling (TTS) has proven effective in improving the reasoning
ability of large language models (LLMs), low diversity in model outputs often
becomes a bottleneck; this is partly caused by the common "one problem, one
solution" (1P1S) training practice, which provides a single canonical answer
and can push models toward a narrow set of reasoning paths. To address this, we
propose a "one problem, multiple solutions" (1PNS) training paradigm that
exposes the model to a variety of valid reasoning trajectories and thus
increases inference diversity. A core challenge for 1PNS is reliably measuring
semantic differences between multi-step chains of thought, so we introduce
Reasoning Path Divergence (RPD), a step-level metric that aligns and scores
Long Chain-of-Thought solutions to capture differences in intermediate
reasoning. Using RPD, we curate maximally diverse solution sets per problem and
fine-tune Qwen3-4B-Base. Experiments show that RPD-selected training yields
more varied outputs and higher pass@k, with an average +2.80% gain in pass@16
over a strong 1P1S baseline and a +4.99% gain on AIME24, demonstrating that
1PNS further amplifies the effectiveness of TTS. Our code is available at
https://github.com/fengjujf/Reasoning-Path-Divergence .

### 2. [On the Influence of Discourse Relations in Persuasive Texts](http://arxiv.org/pdf/2510.26124v1)

Authors: Nawar Turk, Sevag Kaspar, Leila Kosseim

This paper investigates the relationship between Persuasion Techniques (PTs)
and Discourse Relations (DRs) by leveraging Large Language Models (LLMs) and
prompt engineering. Since no dataset annotated with both PTs and DRs exists, we
took the SemEval 2023 Task 3 dataset labelled with 19 PTs as a starting point
and developed LLM-based classifiers to label each instance of the dataset with
one of the 22 PDTB 3.0 level-2 DRs. In total, four LLMs were evaluated using 10
different prompts, resulting in 40 unique DR classifiers. Ensemble models using
different majority-pooling strategies were used to create 5 silver datasets of
instances labelled with both persuasion techniques and level-2 PDTB senses. The
silver dataset sizes vary from 1,281 instances to 204 instances, depending on
the majority pooling technique used. Statistical analysis of these silver
datasets shows that six discourse relations (namely Cause, Purpose, Contrast,
Cause+Belief, Concession, and Condition) play a crucial role in persuasive
texts, especially in the use of Loaded Language, Exaggeration/Minimisation,
Repetition and to cast Doubt. This insight can contribute to detecting online
propaganda and misinformation, as well as to our general understanding of
effective communication.

### 3. [MossNet: Mixture of State-Space Experts is a Multi-Head Attention](http://arxiv.org/pdf/2510.26182v1)

Authors: Shikhar Tuli, James Seale Smith, Haris Jeelani, Chi-Heng Lin, Abhishek Patel, Vasili Ramanishka, Yen-Chang Hsu, Hongxia Jin

Large language models (LLMs) have significantly advanced generative
applications in natural language processing (NLP). Recent trends in model
architectures revolve around efficient variants of transformers or
state-space/gated-recurrent models (SSMs, GRMs). However, prevailing
SSM/GRM-based methods often emulate only a single attention head, potentially
limiting their expressiveness. In this work, we propose MossNet, a novel
mixture-of-state-space-experts architecture that emulates a linear multi-head
attention (MHA). MossNet leverages a mixture-of-experts (MoE) implementation
not only in channel-mixing multi-layered perceptron (MLP) blocks but also in
the time-mixing SSM kernels to realize multiple "attention heads." Extensive
experiments on language modeling and downstream evaluations show that MossNet
outperforms both transformer- and SSM-based architectures of similar model size
and data budgets. Larger variants of MossNet, trained on trillions of tokens,
further confirm its scalability and superior performance. In addition,
real-device profiling on a Samsung Galaxy S24 Ultra and an Nvidia A100 GPU
demonstrate favorable runtime speed and resource usage compared to similarly
sized baselines. Our results suggest that MossNet is a compelling new direction
for efficient, high-performing recurrent LLM architectures.

### 4. [Similarity-Distance-Magnitude Language Models](http://arxiv.org/pdf/2510.26183v1)

Authors: Allen Schmaltz

We introduce Similarity-Distance-Magnitude (SDM) language models (LMs), which
are sequence prediction models fine-tuned to maximize the proportion of
generations in the well-calibrated, high-probability region partitioned by a
final-layer SDM activation layer used for binary classification of
instruction-following. We demonstrate that existing pre-trained decoder-only
Transformer LMs can be readily converted into SDM LMs via supervised
fine-tuning, using the final-layer SDM activation layer during training to
estimate a change-of-base for a supervised next-token loss over a contrastive
input encoding scheme, with additional hard negative examples generated online
during training. This results in reduced abstentions (i.e., improved
statistical efficiency) compared to strong supervised baselines.

### 5. [RCScore: Quantifying Response Consistency in Large Language Models](http://arxiv.org/pdf/2510.26193v1)

Authors: Dongjun Jang, Youngchae Ahn, Hyopil Shin

Current LLM evaluations often rely on a single instruction template,
overlooking models' sensitivity to instruction style-a critical aspect for
real-world deployments. We present RCScore, a multi-dimensional framework
quantifying how instruction formulation affects model responses. By
systematically transforming benchmark problems into multiple instruction
styles, RCScore reveals performance variations undetected by conventional
metrics. Our experiments across ten LLMs on four reasoning benchmarks
demonstrate that instruction style can shift accuracy by up to 16.7% points. We
introduce Cross-Response Similarity (CRS), a method applying RCScore metrics to
measure stylistic self-consistency, and establish its strong correlation with
task accuracy, suggesting consistency as a valuable proxy for model
reliability. Additional findings show that deterministic decoding produces more
stylistically stable outputs, and model scale correlates positively with
cross-style consistency. RCScore offers a principled approach to assess
instruction robustness.

### 6. [Pragmatic Theories Enhance Understanding of Implied Meanings in LLMs](http://arxiv.org/pdf/2510.26253v1)

Authors: Takuma Sato, Seiya Kawano, Koichiro Yoshino

The ability to accurately interpret implied meanings plays a crucial role in
human communication and language use, and language models are also expected to
possess this capability. This study demonstrates that providing language models
with pragmatic theories as prompts is an effective in-context learning approach
for tasks to understand implied meanings. Specifically, we propose an approach
in which an overview of pragmatic theories, such as Gricean pragmatics and
Relevance Theory, is presented as a prompt to the language model, guiding it
through a step-by-step reasoning process to derive a final interpretation.
Experimental results showed that, compared to the baseline, which prompts
intermediate reasoning without presenting pragmatic theories (0-shot
Chain-of-Thought), our methods enabled language models to achieve up to 9.6\%
higher scores on pragmatic reasoning tasks. Furthermore, we show that even
without explaining the details of pragmatic theories, merely mentioning their
names in the prompt leads to a certain performance improvement (around 1-3%) in
larger models compared to the baseline.

### 7. [Language Models Are Borrowing-Blind: A Multilingual Evaluation of Loanword Identification across 10 Languages](http://arxiv.org/pdf/2510.26254v1)

Authors: Mérilin Sousa Silva, Sina Ahmadi

Throughout language history, words are borrowed from one language to another
and gradually become integrated into the recipient's lexicon. Speakers can
often differentiate these loanwords from native vocabulary, particularly in
bilingual communities where a dominant language continuously imposes lexical
items on a minority language. This paper investigates whether pretrained
language models, including large language models, possess similar capabilities
for loanword identification. We evaluate multiple models across 10 languages.
Despite explicit instructions and contextual information, our results show that
models perform poorly in distinguishing loanwords from native ones. These
findings corroborate previous evidence that modern NLP systems exhibit a bias
toward loanwords rather than native equivalents. Our work has implications for
developing NLP tools for minority languages and supporting language
preservation in communities under lexical pressure from dominant languages.

### 8. [Distilling Multilingual Vision-Language Models: When Smaller Models Stay Multilingual](http://arxiv.org/pdf/2510.26271v1)

Authors: Sukrit Sriratanawilai, Jhayahgrit Thongwat, Romrawin Chumpu, Patomporn Payoungkhamdee, Sarana Nutanong, Peerat Limkonchotiwat

Vision-language models (VLMs) exhibit uneven performance across languages, a
problem that is often exacerbated when the model size is reduced. While
Knowledge distillation (KD) demonstrates promising results in transferring
knowledge from larger to smaller VLMs, applying KD in multilingualism is an
underexplored area. This paper presents a controlled empirical study of KD
behavior across five distillation approaches, isolating their effects on
cross-lingual representation consistency and downstream performance stability
under model compression. We study five distillation formulations across CLIP
and SigLIP2, and evaluate them on in-domain retrieval and out-of-domain visual
QA. We find that some configurations preserve or even improve multilingual
retrieval robustness despite halving model size, but others fail to maintain
cross-task stability, exposing design-sensitive trade-offs that aggregate
accuracy alone does not reveal.

### 9. [Do LLMs Signal When They're Right? Evidence from Neuron Agreement](http://arxiv.org/pdf/2510.26277v1)

Authors: Kang Chen, Yaoning Wang, Kai Xiong, Zhuoka Feng, Wenhe Sun, Haotian Chen, Yixin Cao

Large language models (LLMs) commonly boost reasoning via
sample-evaluate-ensemble decoders, achieving label free gains without ground
truth. However, prevailing strategies score candidates using only external
outputs such as token probabilities, entropies, or self evaluations, and these
signals can be poorly calibrated after post training. We instead analyze
internal behavior based on neuron activations and uncover three findings: (1)
external signals are low dimensional projections of richer internal dynamics;
(2) correct responses activate substantially fewer unique neurons than
incorrect ones throughout generation; and (3) activations from correct
responses exhibit stronger cross sample agreement, whereas incorrect ones
diverge. Motivated by these observations, we propose Neuron Agreement Decoding
(NAD), an unsupervised best-of-N method that selects candidates using
activation sparsity and cross sample neuron agreement, operating solely on
internal signals and without requiring comparable textual outputs. NAD enables
early correctness prediction within the first 32 generated tokens and supports
aggressive early stopping. Across math and science benchmarks with verifiable
answers, NAD matches majority voting; on open ended coding benchmarks where
majority voting is inapplicable, NAD consistently outperforms Avg@64. By
pruning unpromising trajectories early, NAD reduces token usage by 99% with
minimal loss in generation quality, showing that internal signals provide
reliable, scalable, and efficient guidance for label free ensemble decoding.

### 10. [SCRIBE: Structured Chain Reasoning for Interactive Behaviour Explanations using Tool Calling](http://arxiv.org/pdf/2510.26322v1)

Authors: Fares Fawzi, Vinitra Swamy, Dominik Glandorf, Tanya Nazaretsky, Tanja Käser

Language models can be used to provide interactive, personalized student
feedback in educational settings. However, real-world deployment faces three
key challenges: privacy concerns, limited computational resources, and the need
for pedagogically valid responses. These constraints require small, open-source
models that can run locally and reliably ground their outputs in correct
information. We introduce SCRIBE, a framework for multi-hop, tool-augmented
reasoning designed to generate valid responses to student questions about
feedback reports. SCRIBE combines domain-specific tools with a self-reflective
inference pipeline that supports iterative reasoning, tool use, and error
recovery. We distil these capabilities into 3B and 8B models via two-stage LoRA
fine-tuning on synthetic GPT-4o-generated data. Evaluation with a human-aligned
GPT-Judge and a user study with 108 students shows that 8B-SCRIBE models
achieve comparable or superior quality to much larger models in key dimensions
such as relevance and actionability, while being perceived on par with GPT-4o
and Llama-3.3 70B by students. These findings demonstrate the viability of
SCRIBE for low-resource, privacy-sensitive educational applications.

### Cryptography and Security

### 1. [PEEL: A Poisoning-Exposing Encoding Theoretical Framework for Local Differential Privacy](http://arxiv.org/pdf/2510.26102v1)

Authors: Lisha Shuai, Jiuling Dong, Nan Zhang, Shaofeng Tan, Haokun Zhang, Zilong Song, Gaoya Dong, Xiaolong Yang

Local Differential Privacy (LDP) is a widely adopted privacy-protection model
in the Internet of Things (IoT) due to its lightweight, decentralized, and
scalable nature. However, it is vulnerable to poisoning attacks, and existing
defenses either incur prohibitive resource overheads or rely on domain-specific
prior knowledge, limiting their practical deployment. To address these
limitations, we propose PEEL, a Poisoning-Exposing Encoding theoretical
framework for LDP, which departs from resource- or prior-dependent
countermeasures and instead leverages the inherent structural consistency of
LDP-perturbed data. As a non-intrusive post-processing module, PEEL amplifies
stealthy poisoning effects by re-encoding LDP-perturbed data via
sparsification, normalization, and low-rank projection, thereby revealing both
output and rule poisoning attacks through structural inconsistencies in the
reconstructed space. Theoretical analysis proves that PEEL, integrated with
LDP, retains unbiasedness and statistical accuracy, while being robust to
expose both output and rule poisoning attacks. Moreover, evaluation results
show that LDP-integrated PEEL not only outperforms four state-of-the-art
defenses in terms of poisoning exposure accuracy but also significantly reduces
client-side computational costs, making it highly suitable for large-scale IoT
deployments.

### 2. [Security Vulnerabilities in AI-Generated Code: A Large-Scale Analysis of Public GitHub Repositories](http://arxiv.org/pdf/2510.26103v1)

Authors: Maximilian Schreiber, Pascal Tippe

This paper presents a comprehensive empirical analysis of security
vulnerabilities in AI-generated code across public GitHub repositories. We
collected and analyzed 7,703 files explicitly attributed to four major AI
tools: ChatGPT (91.52\%), GitHub Copilot (7.50\%), Amazon CodeWhisperer
(0.52\%), and Tabnine (0.46\%). Using CodeQL static analysis, we identified
4,241 Common Weakness Enumeration (CWE) instances across 77 distinct
vulnerability types. Our findings reveal that while 87.9\% of AI-generated code
does not contain identifiable CWE-mapped vulnerabilities, significant patterns
emerge regarding language-specific vulnerabilities and tool performance. Python
consistently exhibited higher vulnerability rates (16.18\%-18.50\%) compared to
JavaScript (8.66\%-8.99\%) and TypeScript (2.50\%-7.14\%) across all tools. We
observed notable differences in security performance, with GitHub Copilot
achieving better security density for Python (1,739 LOC per CWE) and
TypeScript, while ChatGPT performed better for JavaScript. Additionally, we
discovered widespread use of AI tools for documentation generation (39\% of
collected files), an understudied application with implications for software
maintainability. These findings extend previous work with a significantly
larger dataset and provide valuable insights for developing language-specific
and context-aware security practices for the responsible integration of
AI-generated code into software development workflows.

### 3. [Who Moved My Transaction? Uncovering Post-Transaction Auditability Vulnerabilities in Modern Super Apps](http://arxiv.org/pdf/2510.26210v1)

Authors: Junlin Liu, Zhaomeng Deng, Ziming Wang, Mengyu Yao, Yifeng Cai, Yutao Hu, Ziqi Zhang, Yao Guo, Ding Li

Super apps are the cornerstones of modern digital life, embedding financial
transactions into nearly every aspect of daily routine. The prevailing security
paradigm for these platforms is overwhelmingly focused on pre-transaction
authentication, preventing unauthorized payments before they occur. We argue
that a critical vulnerability vector has been largely overlooked: the fragility
of post-transaction audit trails. We investigate the ease with which a user can
permanently erase their transaction history from an app's interface, thereby
concealing unauthorized or sensitive activities from the account owner. To
quantify this threat, we conducted an empirical study with 6 volunteers who
performed a cross-evaluation on six super apps. Our findings are alarming: all
six applications studied allow users to delete transaction records, yet a
staggering five out of six (83+\%) fail to protect these records with strong
authentication. Only one app in our study required biometric verification for
deletion. This study provides the first concrete evidence of this
near-ubiquitous vulnerability, demonstrating a critical gap in the current
mobile security landscape and underscoring the urgent need for a paradigm shift
towards ensuring post-transaction audit integrity.

### 4. [Who Grants the Agent Power? Defending Against Instruction Injection via Task-Centric Access Control](http://arxiv.org/pdf/2510.26212v1)

Authors: Yifeng Cai, Ziming Wang, Zhaomeng Deng, Mengyu Yao, Junlin Liu, Yutao Hu, Ziqi Zhang, Yao Guo, Ding Li

AI agents capable of GUI understanding and Model Context Protocol are
increasingly deployed to automate mobile tasks. However, their reliance on
over-privileged, static permissions creates a critical vulnerability:
instruction injection. Malicious instructions, embedded in otherwise benign
content like emails, can hijack the agent to perform unauthorized actions. We
present AgentSentry, a lightweight runtime task-centric access control
framework that enforces dynamic, task-scoped permissions. Instead of granting
broad, persistent permissions, AgentSentry dynamically generates and enforces
minimal, temporary policies aligned with the user's specific task (e.g.,
register for an app), revoking them upon completion. We demonstrate that
AgentSentry successfully prevents an instruction injection attack, where an
agent is tricked into forwarding private emails, while allowing the legitimate
task to complete. Our approach highlights the urgent need for intent-aligned
security models to safely govern the next generation of autonomous agents.

### 5. [CyberNER: A Harmonized STIX Corpus for Cybersecurity Named Entity Recognition](http://arxiv.org/pdf/2510.26499v1)

Authors: Yasir Ech-Chammakhy, Anas Motii, Anass Rabii, Oussama Azrara, Jaafar Chbili

Extracting structured intelligence via Named Entity Recognition (NER) is
critical for cybersecurity, but the proliferation of datasets with incompatible
annotation schemas hinders the development of comprehensive models. While
combining these resources is desirable, we empirically demonstrate that naively
concatenating them results in a noisy label space that severely degrades model
performance. To overcome this critical limitation, we introduce CyberNER, a
large-scale, unified corpus created by systematically harmonizing four
prominent datasets (CyNER, DNRTI, APTNER, and Attacker) onto the STIX 2.1
standard. Our principled methodology resolves semantic ambiguities and
consolidates over 50 disparate source tags into 21 coherent entity types. Our
experiments show that models trained on CyberNER achieve a substantial
performance gain, with a relative F1-score improvement of approximately 30%
over the naive concatenation baseline. By publicly releasing the CyberNER
corpus, we provide a crucial, standardized benchmark that enables the creation
and rigorous comparison of more robust and generalizable entity extraction
models for the cybersecurity domain.

### 6. [Interdependent Privacy in Smart Homes: Hunting for Bystanders in Privacy Policies](http://arxiv.org/pdf/2510.26523v1)

Authors: Shuaishuai Liu, Gergely Acs, Gergely Biczók

Smart home devices such as video doorbells and security cameras are becoming
increasingly common in everyday life. While these devices offer convenience and
safety, they also raise new privacy concerns: how these devices affect others,
like neighbors, visitors, or people passing by. This issue is generally known
as interdependent privacy, where one person's actions (or inaction) may impact
the privacy of others, and, specifically, bystander privacy in the context of
smart homes. Given lax data protection regulations in terms of shared physical
spaces and amateur joint data controllers, we expect that the privacy policies
of smart home products reflect the missing regulatory incentives. This paper
presents a focused privacy policy analysis of 20 video doorbell and smart
camera products, concentrating explicitly on the bystander aspect. We show that
although some of the vendors acknowledge bystanders, they address it only to
the extent of including disclaimers, shifting the ethical responsibility for
collecting the data of non-users to the device owner. In addition, we identify
and examine real-world cases related to bystander privacy, demonstrating how
current deployments can impact non-users. Based on our findings, we analyze
vendor privacy policies in light of existing legal frameworks and technical
capabilities, and we provide practical recommendations for both policy language
and system design to enhance transparency and empower both bystanders and
device owners.

### 7. [A Comprehensive Evaluation and Practice of System Penetration Testing](http://arxiv.org/pdf/2510.26555v1)

Authors: Chunyi Zhang, Jin Zeng, Xiaoqi Li

With the rapid advancement of information technology, the complexity of
applications continues to increase, and the cybersecurity challenges we face
are also escalating. This paper aims to investigate the methods and practices
of system security penetration testing, exploring how to enhance system
security through systematic penetration testing processes and technical
approaches. It also examines existing penetration tools, analyzing their
strengths, weaknesses, and applicable domains to guide penetration testers in
tool selection. Furthermore, based on the penetration testing process outlined
in this paper, appropriate tools are selected to replicate attack processes
using target ranges and target machines. Finally, through practical case
analysis, lessons learned from successful attacks are summarized to inform
future research.

### 8. [A DRL-Empowered Multi-Level Jamming Approach for Secure Semantic Communication](http://arxiv.org/pdf/2510.26610v1)

Authors: Weixuan Chen, Qianqian Yang

Semantic communication (SemCom) aims to transmit only task-relevant
information, thereby improving communication efficiency but also exposing
semantic information to potential eavesdropping. In this paper, we propose a
deep reinforcement learning (DRL)-empowered multi-level jamming approach to
enhance the security of SemCom systems over MIMO fading wiretap channels. This
approach combines semantic layer jamming, achieved by encoding task-irrelevant
text, and physical layer jamming, achieved by encoding random Gaussian noise.
These two-level jamming signals are superposed with task-relevant semantic
information to protect the transmitted semantics from eavesdropping. A deep
deterministic policy gradient (DDPG) algorithm is further introduced to
dynamically design and optimize the precoding matrices for both taskrelevant
semantic information and multi-level jamming signals, aiming to enhance the
legitimate user's image reconstruction while degrading the eavesdropper's
performance. To jointly train the SemCom model and the DDPG agent, we propose
an alternating optimization strategy where the two modules are updated
iteratively. Experimental results demonstrate that, compared with both the
encryption-based (ESCS) and encoded jammer-based (EJ) benchmarks, our method
achieves comparable security while improving the legitimate user's peak
signalto-noise ratio (PSNR) by up to approximately 0.6 dB.

### 9. [A Survey of Heterogeneous Graph Neural Networks for Cybersecurity Anomaly Detection](http://arxiv.org/pdf/2510.26307v1)

Authors: Laura Jiang, Reza Ryan, Qian Li, Nasim Ferdosian

Anomaly detection is a critical task in cybersecurity, where identifying
insider threats, access violations, and coordinated attacks is essential for
ensuring system resilience. Graph-based approaches have become increasingly
important for modeling entity interactions, yet most rely on homogeneous and
static structures, which limits their ability to capture the heterogeneity and
temporal evolution of real-world environments. Heterogeneous Graph Neural
Networks (HGNNs) have emerged as a promising paradigm for anomaly detection by
incorporating type-aware transformations and relation-sensitive aggregation,
enabling more expressive modeling of complex cyber data. However, current
research on HGNN-based anomaly detection remains fragmented, with diverse
modeling strategies, limited comparative evaluation, and an absence of
standardized benchmarks. To address this gap, we provide a comprehensive survey
of HGNN-based anomaly detection methods in cybersecurity. We introduce a
taxonomy that classifies approaches by anomaly type and graph dynamics, analyze
representative models, and map them to key cybersecurity applications. We also
review commonly used benchmark datasets and evaluation metrics, highlighting
their strengths and limitations. Finally, we identify key open challenges
related to modeling, data, and deployment, and outline promising directions for
future research. This survey aims to establish a structured foundation for
advancing HGNN-based anomaly detection toward scalable, interpretable, and
practically deployable solutions.

### 10. [SSCL-BW: Sample-Specific Clean-Label Backdoor Watermarking for Dataset Ownership Verification](http://arxiv.org/pdf/2510.26420v1)

Authors: Yingjia Wang, Ting Qiao, Xing Liu, Chongzuo Li, Sixing Wu, Jianbin Li

The rapid advancement of deep neural networks (DNNs) heavily relies on
large-scale, high-quality datasets. However, unauthorized commercial use of
these datasets severely violates the intellectual property rights of dataset
owners. Existing backdoor-based dataset ownership verification methods suffer
from inherent limitations: poison-label watermarks are easily detectable due to
label inconsistencies, while clean-label watermarks face high technical
complexity and failure on high-resolution images. Moreover, both approaches
employ static watermark patterns that are vulnerable to detection and removal.
To address these issues, this paper proposes a sample-specific clean-label
backdoor watermarking (i.e., SSCL-BW). By training a U-Net-based watermarked
sample generator, this method generates unique watermarks for each sample,
fundamentally overcoming the vulnerability of static watermark patterns. The
core innovation lies in designing a composite loss function with three
components: target sample loss ensures watermark effectiveness, non-target
sample loss guarantees trigger reliability, and perceptual similarity loss
maintains visual imperceptibility. During ownership verification, black-box
testing is employed to check whether suspicious models exhibit predefined
backdoor behaviors. Extensive experiments on benchmark datasets demonstrate the
effectiveness of the proposed method and its robustness against potential
watermark removal attacks.

### Computer Vision and Pattern Recognition

### 1. [FlexICL: A Flexible Visual In-context Learning Framework for Elbow and Wrist Ultrasound Segmentation](http://arxiv.org/pdf/2510.26049v1)

Authors: Yuyue Zhou, Jessica Knight, Shrimanti Ghosh, Banafshe Felfeliyan, Jacob L. Jaremko, Abhilash R. Hareendranathan

Elbow and wrist fractures are the most common fractures in pediatric
populations. Automatic segmentation of musculoskeletal structures in ultrasound
(US) can improve diagnostic accuracy and treatment planning. Fractures appear
as cortical defects but require expert interpretation. Deep learning (DL) can
provide real-time feedback and highlight key structures, helping lightly
trained users perform exams more confidently. However, pixel-wise expert
annotations for training remain time-consuming and costly. To address this
challenge, we propose FlexICL, a novel and flexible in-context learning (ICL)
framework for segmenting bony regions in US images. We apply it to an
intra-video segmentation setting, where experts annotate only a small subset of
frames, and the model segments unseen frames. We systematically investigate
various image concatenation techniques and training strategies for visual ICL
and introduce novel concatenation methods that significantly enhance model
performance with limited labeled data. By integrating multiple augmentation
strategies, FlexICL achieves robust segmentation performance across four wrist
and elbow US datasets while requiring only 5% of the training images. It
outperforms state-of-the-art visual ICL models like Painter, MAE-VQGAN, and
conventional segmentation models like U-Net and TransUNet by 1-27% Dice
coefficient on 1,252 US sweeps. These initial results highlight the potential
of FlexICL as an efficient and scalable solution for US image segmentation well
suited for medical imaging use cases where labeled data is scarce.

### 2. [OracleAgent: A Multimodal Reasoning Agent for Oracle Bone Script Research](http://arxiv.org/pdf/2510.26114v1)

Authors: Caoshuo Li, Zengmao Ding, Xiaobin Hu, Bang Li, Donghao Luo, Xu Peng, Taisong Jin, Yongge Liu, Shengwei Han, Jing Yang, Xiaoping He, Feng Gao, AndyPian Wu, SevenShu, Chaoyang Wang, Chengjie Wang

As one of the earliest writing systems, Oracle Bone Script (OBS) preserves
the cultural and intellectual heritage of ancient civilizations. However,
current OBS research faces two major challenges: (1) the interpretation of OBS
involves a complex workflow comprising multiple serial and parallel sub-tasks,
and (2) the efficiency of OBS information organization and retrieval remains a
critical bottleneck, as scholars often spend substantial effort searching for,
compiling, and managing relevant resources. To address these challenges, we
present OracleAgent, the first agent system designed for the structured
management and retrieval of OBS-related information. OracleAgent seamlessly
integrates multiple OBS analysis tools, empowered by large language models
(LLMs), and can flexibly orchestrate these components. Additionally, we
construct a comprehensive domain-specific multimodal knowledge base for OBS,
which is built through a rigorous multi-year process of data collection,
cleaning, and expert annotation. The knowledge base comprises over 1.4M
single-character rubbing images and 80K interpretation texts. OracleAgent
leverages this resource through its multimodal tools to assist experts in
retrieval tasks of character, document, interpretation text, and rubbing image.
Extensive experiments demonstrate that OracleAgent achieves superior
performance across a range of multimodal reasoning and generation tasks,
surpassing leading mainstream multimodal large language models (MLLMs) (e.g.,
GPT-4o). Furthermore, our case study illustrates that OracleAgent can
effectively assist domain experts, significantly reducing the time cost of OBS
research. These results highlight OracleAgent as a significant step toward the
practical deployment of OBS-assisted research and automated interpretation
systems.

### 3. [JOGS: Joint Optimization of Pose Estimation and 3D Gaussian Splatting](http://arxiv.org/pdf/2510.26117v1)

Authors: Yuxuan Li, Tao Wang, Xianben Yang

Traditional novel view synthesis methods heavily rely on external camera pose
estimation tools such as COLMAP, which often introduce computational
bottlenecks and propagate errors. To address these challenges, we propose a
unified framework that jointly optimizes 3D Gaussian points and camera poses
without requiring pre-calibrated inputs. Our approach iteratively refines 3D
Gaussian parameters and updates camera poses through a novel co-optimization
strategy, ensuring simultaneous improvements in scene reconstruction fidelity
and pose accuracy. The key innovation lies in decoupling the joint optimization
into two interleaved phases: first, updating 3D Gaussian parameters via
differentiable rendering with fixed poses, and second, refining camera poses
using a customized 3D optical flow algorithm that incorporates geometric and
photometric constraints. This formulation progressively reduces projection
errors, particularly in challenging scenarios with large viewpoint variations
and sparse feature distributions, where traditional methods struggle. Extensive
evaluations on multiple datasets demonstrate that our approach significantly
outperforms existing COLMAP-free techniques in reconstruction quality, and also
surpasses the standard COLMAP-based baseline in general.

### 4. [FullPart: Generating each 3D Part at Full Resolution](http://arxiv.org/pdf/2510.26140v1)

Authors: Lihe Ding, Shaocong Dong, Yaokun Li, Chenjian Gao, Xiao Chen, Rui Han, Yihao Kuang, Hong Zhang, Bo Huang, Zhanpeng Huang, Zibin Wang, Dan Xu, Tianfan Xue

Part-based 3D generation holds great potential for various applications.
Previous part generators that represent parts using implicit vector-set tokens
often suffer from insufficient geometric details. Another line of work adopts
an explicit voxel representation but shares a global voxel grid among all
parts; this often causes small parts to occupy too few voxels, leading to
degraded quality. In this paper, we propose FullPart, a novel framework that
combines both implicit and explicit paradigms. It first derives the bounding
box layout through an implicit box vector-set diffusion process, a task that
implicit diffusion handles effectively since box tokens contain little
geometric detail. Then, it generates detailed parts, each within its own fixed
full-resolution voxel grid. Instead of sharing a global low-resolution space,
each part in our method - even small ones - is generated at full resolution,
enabling the synthesis of intricate details. We further introduce a
center-point encoding strategy to address the misalignment issue when
exchanging information between parts of different actual sizes, thereby
maintaining global coherence. Moreover, to tackle the scarcity of reliable part
data, we present PartVerse-XL, the largest human-annotated 3D part dataset to
date with 40K objects and 320K parts. Extensive experiments demonstrate that
FullPart achieves state-of-the-art results in 3D part generation. We will
release all code, data, and model to benefit future research in 3D part
generation.

### 5. [BasicAVSR: Arbitrary-Scale Video Super-Resolution via Image Priors and Enhanced Motion Compensation](http://arxiv.org/pdf/2510.26149v1)

Authors: Wei Shang, Wanying Zhang, Shuhang Gu, Pengfei Zhu, Qinghua Hu, Dongwei Ren

Arbitrary-scale video super-resolution (AVSR) aims to enhance the resolution
of video frames, potentially at various scaling factors, which presents several
challenges regarding spatial detail reproduction, temporal consistency, and
computational complexity. In this paper, we propose a strong baseline BasicAVSR
for AVSR by integrating four key components: 1) adaptive multi-scale frequency
priors generated from image Laplacian pyramids, 2) a flow-guided propagation
unit to aggregate spatiotemporal information from adjacent frames, 3) a
second-order motion compensation unit for more accurate spatial alignment of
adjacent frames, and 4) a hyper-upsampling unit to generate scale-aware and
content-independent upsampling kernels. To meet diverse application demands, we
instantiate three propagation variants: (i) a unidirectional RNN unit for
strictly online inference, (ii) a unidirectional RNN unit empowered with a
limited lookahead that tolerates a small output delay, and (iii) a
bidirectional RNN unit designed for offline tasks where computational resources
are less constrained. Experimental results demonstrate the effectiveness and
adaptability of our model across these different scenarios. Through extensive
experiments, we show that BasicAVSR significantly outperforms existing methods
in terms of super-resolution quality, generalization ability, and inference
speed. Our work not only advances the state-of-the-art in AVSR but also extends
its core components to multiple frameworks for diverse scenarios. The code is
available at https://github.com/shangwei5/BasicAVSR.

### 6. [Detecting Unauthorized Vehicles using Deep Learning for Smart Cities: A Case Study on Bangladesh](http://arxiv.org/pdf/2510.26154v1)

Authors: Sudipto Das Sukanto, Diponker Roy, Fahim Shakil, Nirjhar Singha, Abdullah Asik, Aniket Joarder, Mridha Md Nafis Fuad, Muhammad Ibrahim

Modes of transportation vary across countries depending on geographical
location and cultural context. In South Asian countries rickshaws are among the
most common means of local transport. Based on their mode of operation,
rickshaws in cities across Bangladesh can be broadly classified into non-auto
(pedal-powered) and auto-rickshaws (motorized). Monitoring the movement of
auto-rickshaws is necessary as traffic rules often restrict auto-rickshaws from
accessing certain routes. However, existing surveillance systems make it quite
difficult to monitor them due to their similarity to other vehicles, especially
non-auto rickshaws whereas manual video analysis is too time-consuming. This
paper presents a machine learning-based approach to automatically detect
auto-rickshaws in traffic images. In this system, we used real-time object
detection using the YOLOv8 model. For training purposes, we prepared a set of
1,730 annotated images that were captured under various traffic conditions. The
results show that our proposed model performs well in real-time auto-rickshaw
detection and offers an mAP50 of 83.447% and binary precision and recall values
above 78%, demonstrating its effectiveness in handling both dense and sparse
traffic scenarios. The dataset has been publicly released for further research.

### 7. [CRAG-MM: Multi-modal Multi-turn Comprehensive RAG Benchmark](http://arxiv.org/pdf/2510.26160v1)

Authors: Jiaqi Wang, Xiao Yang, Kai Sun, Parth Suresh, Sanat Sharma, Adam Czyzewski, Derek Andersen, Surya Appini, Arkav Banerjee, Sajal Choudhary, Shervin Ghasemlou, Ziqiang Guan, Akil Iyer, Haidar Khan, Lingkun Kong, Roy Luo, Tiffany Ma, Zhen Qiao, David Tran, Wenfang Xu, Skyler Yeatman, Chen Zhou, Gunveer Gujral, Yinglong Xia, Shane Moon, Nicolas Scheffer, Nirav Shah, Eun Chang, Yue Liu, Florian Metze, Tammy Stark, Zhaleh Feizollahi, Andrea Jessee, Mangesh Pujari, Ahmed Aly, Babak Damavandi, Rakesh Wanga, Anuj Kumar, Rohit Patel, Wen-tau Yih, Xin Luna Dong

Wearable devices such as smart glasses are transforming the way people
interact with their surroundings, enabling users to seek information regarding
entities in their view. Multi-Modal Retrieval-Augmented Generation (MM-RAG)
plays a key role in supporting such questions, yet there is still no
comprehensive benchmark for this task, especially regarding wearables
scenarios. To fill this gap, we present CRAG-MM -- a Comprehensive RAG
benchmark for Multi-modal Multi-turn conversations. CRAG-MM contains a diverse
set of 6.5K (image, question, answer) triplets and 2K visual-based multi-turn
conversations across 13 domains, including 6.2K egocentric images designed to
mimic captures from wearable devices. We carefully constructed the questions to
reflect real-world scenarios and challenges, including five types of
image-quality issues, six question types, varying entity popularity, differing
information dynamism, and different conversation turns. We design three tasks:
single-source augmentation, multi-source augmentation, and multi-turn
conversations -- each paired with an associated retrieval corpus and APIs for
both image-KG retrieval and webpage retrieval. Our evaluation shows that
straightforward RAG approaches achieve only 32% and 43% truthfulness on CRAG-MM
single- and multi-turn QA, respectively, whereas state-of-the-art industry
solutions have similar quality (32%/45%), underscoring ample room for
improvement. The benchmark has hosted KDD Cup 2025, attracting about 1K
participants and 5K submissions, with winning solutions improving baseline
performance by 28%, highlighting its early impact on advancing the field.

### 8. [MoTDiff: High-resolution Motion Trajectory estimation from a single blurred image using Diffusion models](http://arxiv.org/pdf/2510.26173v1)

Authors: Wontae Choi, Jaelin Lee, Hyung Sup Yun, Byeungwoo Jeon, Il Yong Chun

Accurate estimation of motion information is crucial in diverse computational
imaging and computer vision applications. Researchers have investigated various
methods to extract motion information from a single blurred image, including
blur kernels and optical flow. However, existing motion representations are
often of low quality, i.e., coarse-grained and inaccurate. In this paper, we
propose the first high-resolution (HR) Motion Trajectory estimation framework
using Diffusion models (MoTDiff). Different from existing motion
representations, we aim to estimate an HR motion trajectory with high-quality
from a single motion-blurred image. The proposed MoTDiff consists of two key
components: 1) a new conditional diffusion framework that uses multi-scale
feature maps extracted from a single blurred image as a condition, and 2) a new
training method that can promote precise identification of a fine-grained
motion trajectory, consistent estimation of overall shape and position of a
motion path, and pixel connectivity along a motion trajectory. Our experiments
demonstrate that the proposed MoTDiff can outperform state-of-the-art methods
in both blind image deblurring and coded exposure photography applications.

### 9. [Sketch2PoseNet: Efficient and Generalized Sketch to 3D Human Pose Prediction](http://arxiv.org/pdf/2510.26196v1)

Authors: Li Wang, Yiyu Zhuang, Yanwen Wang, Xun Cao, Chuan Guo, Xinxin Zuo, Hao Zhu

3D human pose estimation from sketches has broad applications in computer
animation and film production. Unlike traditional human pose estimation, this
task presents unique challenges due to the abstract and disproportionate nature
of sketches. Previous sketch-to-pose methods, constrained by the lack of
large-scale sketch-3D pose annotations, primarily relied on optimization with
heuristic rules-an approach that is both time-consuming and limited in
generalizability. To address these challenges, we propose a novel approach
leveraging a "learn from synthesis" strategy. First, a diffusion model is
trained to synthesize sketch images from 2D poses projected from 3D human
poses, mimicking disproportionate human structures in sketches. This process
enables the creation of a synthetic dataset, SKEP-120K, consisting of 120k
accurate sketch-3D pose annotation pairs across various sketch styles. Building
on this synthetic dataset, we introduce an end-to-end data-driven framework for
estimating human poses and shapes from diverse sketch styles. Our framework
combines existing 2D pose detectors and generative diffusion priors for sketch
feature extraction with a feed-forward neural network for efficient 2D pose
estimation. Multiple heuristic loss functions are incorporated to guarantee
geometric coherence between the derived 3D poses and the detected 2D poses
while preserving accurate self-contacts. Qualitative, quantitative, and
subjective evaluations collectively show that our model substantially surpasses
previous ones in both estimation accuracy and speed for sketch-to-pose tasks.

### 10. [Developing a Multi-task Ensemble Geometric Deep Network for Supply Chain Sustainability and Risk Management](http://arxiv.org/pdf/2510.26203v1)

Authors: Mehdi Khaleghi, Nastaran Khaleghi, Sobhan Sheykhivand, Sebelan Danishvar

The sustainability of supply chain plays a key role in achieving optimal
performance in controlling the supply chain. The management of risks that occur
in a supply chain is a fundamental problem for the purpose of developing the
sustainability of the network and elevating the performance efficiency of the
supply chain. The correct classification of products is another essential
element in a sustainable supply chain. Acknowledging recent breakthroughs in
the context of deep networks, several architectural options have been deployed
to analyze supply chain datasets. A novel geometric deep network is used to
propose an ensemble deep network. The proposed Chebyshev ensemble geometric
network (Ch-EGN) is a hybrid convolutional and geometric deep learning. This
network is proposed to leverage the information dependencies in supply chain to
derive invisible states of samples in the database. The functionality of the
proposed deep network is assessed on the two different databases. The
SupplyGraph Dataset and DataCo are considered in this research. The prediction
of delivery status of DataCo supply chain is done for risk administration. The
product classification and edge classification are performed using the
SupplyGraph database to enhance the sustainability of the supply network. An
average accuracy of 98.95% is obtained for the ensemble network for risk
management. The average accuracy of 100% and 98.07% are obtained for
sustainable supply chain in terms of 5 product group classification and 4
product relation classification, respectively. The average accuracy of 92.37%
is attained for 25 company relation classification. The results confirm an
average improvement and efficiency of the proposed method compared to the
state-of-the-art approaches.

### Computers and Society

### 1. [Exploring Dissatisfaction in Bus Route Reduction through LLM-Calibrated Agent-Based Modeling](http://arxiv.org/pdf/2510.26163v1)

Authors: Qiumeng Li, Xinxi Yang, Suhong Zhou

As emerging mobility modes continue to expand, many cities face declining bus
ridership, increasing fiscal pressure to sustain underutilized routes, and
growing inefficiencies in resource allocation. This study employs an
agent-based modelling (ABM) approach calibrated through a large language model
(LLM) using few-shot learning to examine how progressive bus route cutbacks
affect passenger dissatisfaction across demographic groups and overall network
resilience. Using IC-card data from Beijing's Huairou District, the
LLM-calibrated ABM estimated passenger sensitivity parameters related to travel
time, waiting, transfers, and crowding. Results show that the structural
configuration of the bus network exerts a stronger influence on system
stability than capacity or operational factors. The elimination of
high-connectivity routes led to an exponential rise in total dissatisfaction,
particularly among passengers with disabilities and older adults. The evolution
of dissatisfaction exhibited three distinct phases - stable, transitional, and
critical. Through the analysis of each stage, this study found that the
continuous bus route reduction scenario exhibits three-stage thresholds. Once
these thresholds are crossed, even a small reduction in routes may lead to a
significant loss of passenger flow. Research highlights the nonlinear response
of user sentiment to service reductions and underscore the importance of
maintaining structural critical routes and providing stable services to
vulnerable groups for equitable and resilient transport planning.

### 2. [A Game-Theoretic Spatio-Temporal Reinforcement Learning Framework for Collaborative Public Resource Allocation](http://arxiv.org/pdf/2510.26184v1)

Authors: Songxin Lei, Qiongyan Wang, Yanchen Zhu, Hanyu Yao, Sijie Ruan, Weilin Ruan, Yuyu Luo, Huaming Wu, Yuxuan Liang

Public resource allocation involves the efficient distribution of resources,
including urban infrastructure, energy, and transportation, to effectively meet
societal demands. However, existing methods focus on optimizing the movement of
individual resources independently, without considering their capacity
constraints. To address this limitation, we propose a novel and more practical
problem: Collaborative Public Resource Allocation (CPRA), which explicitly
incorporates capacity constraints and spatio-temporal dynamics in real-world
scenarios. We propose a new framework called Game-Theoretic Spatio-Temporal
Reinforcement Learning (GSTRL) for solving CPRA. Our contributions are twofold:
1) We formulate the CPRA problem as a potential game and demonstrate that there
is no gap between the potential function and the optimal target, laying a solid
theoretical foundation for approximating the Nash equilibrium of this NP-hard
problem; and 2) Our designed GSTRL framework effectively captures the
spatio-temporal dynamics of the overall system. We evaluate GSTRL on two
real-world datasets, where experiments show its superior performance. Our
source codes are available in the supplementary materials.

### 3. [Industry Members' Perceptions about ABET-based Accreditation: An Exploratory Study in a Developing Country](http://arxiv.org/pdf/2510.26087v1)

Authors: V. Sanchez Padilla, Albert Espinal, Jennifer M. Case, Jose Cordova-Garcia, Homero Murzi

ABET accreditation is an increasingly prominent system of global
accreditation of engineering programs, and the assessment requires programs to
demonstrate that they meet the needs of the program's stakeholders, typically
industrial potential employers of graduates. To obtain these inputs, programs
are required to assemble an advisory committee board. The views of the advisory
board on the relevance of the degree outcomes are an essential part of this
process. The purpose of this qualitative research study is to explore the
viewpoints that industry stakeholders have on this type of process. The context
for the study was an Ecuadorian engineering program which had successfully
achieved the ABET accreditation. The study drew on interviews undertaken with
industry members who were part of the advisory board. This study focuses on how
they perceive the process and the accreditation awarded, analyzing their views
of its usefulness, especially in relation to the employability of graduates.
Based on the findings, we offer critical insights into this accreditation
process when it takes place in contexts beyond highly industrialized countries.

### 4. [Value Drifts: Tracing Value Alignment During LLM Post-Training](http://arxiv.org/pdf/2510.26707v1)

Authors: Mehar Bhatia, Shravan Nayak, Gaurav Kamath, Marius Mosbach, Karolina Stańczak, Vered Shwartz, Siva Reddy

As LLMs occupy an increasingly important role in society, they are more and
more confronted with questions that require them not only to draw on their
general knowledge but also to align with certain human value systems.
Therefore, studying the alignment of LLMs with human values has become a
crucial field of inquiry. Prior work, however, mostly focuses on evaluating the
alignment of fully trained models, overlooking the training dynamics by which
models learn to express human values. In this work, we investigate how and at
which stage value alignment arises during the course of a model's
post-training. Our analysis disentangles the effects of post-training
algorithms and datasets, measuring both the magnitude and time of value drifts
during training. Experimenting with Llama-3 and Qwen-3 models of different
sizes and popular supervised fine-tuning (SFT) and preference optimization
datasets and algorithms, we find that the SFT phase generally establishes a
model's values, and subsequent preference optimization rarely re-aligns these
values. Furthermore, using a synthetic preference dataset that enables
controlled manipulation of values, we find that different preference
optimization algorithms lead to different value alignment outcomes, even when
preference data is held constant. Our findings provide actionable insights into
how values are learned during post-training and help to inform data curation,
as well as the selection of models and algorithms for preference optimization
to improve model alignment to human values.

### 5. [Neither Consent nor Property: A Policy Lab for Data Law](http://arxiv.org/pdf/2510.26727v1)

Authors: Haoyi Zhang, Tianyi Zhu

This paper makes the opaque data market in the AI economy empirically legible
for the first time, constructing a computational testbed to address a core
epistemic failure: regulators governing a market defined by structural opacity,
fragile price discovery, and brittle technical safeguards that have paralyzed
traditional empirics and fragmented policy. The pipeline begins with multi-year
fieldwork to extract the market's hidden logic, and then embeds these grounded
behaviors into a high-fidelity ABM, parameterized via a novel LLM-based
discrete-choice experiment that captures the preferences of unsurveyable
populations. The pipeline is validated against reality, reproducing observed
trade patterns. This policy laboratory delivers clear, counter-intuitive
results. First, property-style relief is a false promise: ''anonymous-data''
carve-outs expand trade but ignore risk, causing aggregate welfare to collapse
once external harms are priced in. Second, social welfare peaks when the
downstream buyer internalizes the full substantive risk. This least-cost
avoider approach induces efficient safeguards, simultaneously raising welfare
and sustaining trade, and provides a robust empirical foundation for the legal
drift toward two-sided reachability. The contribution is a reproducible
pipeline designed to end the reliance on intuition. It converts qualitative
insight into testable, comparative policy experiments, obsoleting armchair
conjecture by replacing it with controlled evidence on how legal rules actually
shift risk and surplus. This is the forward-looking engine that moves the field
from competing intuitions to direct, computational analysis.

### Databases

### 1. [Rethinking Text-to-SQL: Dynamic Multi-turn SQL Interaction for Real-world Database Exploration](http://arxiv.org/pdf/2510.26495v1)

Authors: Linzhuang Sun, Tianyu Guo, Hao Liang, Yuying Li, Qifeng Cai, Jingxuan Wei, Bihui Yu, Wentao Zhang, Bin Cui

Recent advances in Text-to-SQL have achieved strong results in static,
single-turn tasks, where models generate SQL queries from natural language
questions. However, these systems fall short in real-world interactive
scenarios, where user intents evolve and queries must be refined over multiple
turns. In applications such as finance and business analytics, users
iteratively adjust query constraints or dimensions based on intermediate
results. To evaluate such dynamic capabilities, we introduce DySQL-Bench, a
benchmark assessing model performance under evolving user interactions. Unlike
previous manually curated datasets, DySQL-Bench is built through an automated
two-stage pipeline of task synthesis and verification. Structured tree
representations derived from raw database tables guide LLM-based task
generation, followed by interaction-oriented filtering and expert validation.
Human evaluation confirms 100% correctness of the synthesized data. We further
propose a multi-turn evaluation framework simulating realistic interactions
among an LLM-simulated user, the model under test, and an executable database.
The model must adapt its reasoning and SQL generation as user intents change.
DySQL-Bench covers 13 domains across BIRD and Spider 2 databases, totaling
1,072 tasks. Even GPT-4o attains only 58.34% overall accuracy and 23.81% on the
Pass@5 metric, underscoring the benchmark's difficulty. All code and data are
released at https://github.com/Aurora-slz/Real-World-SQL-Bench .

### Distributed, Parallel, and Cluster Computing

### 1. [Environmental Impact of CI/CD Pipelines](http://arxiv.org/pdf/2510.26413v1)

Authors: Nuno Saavedra, Alexandra Mendes, João F. Ferreira

CI/CD pipelines are widely used in software development, yet their
environmental impact, particularly carbon and water footprints (CWF), remains
largely unknown to developers, as CI service providers typically do not
disclose such information. With the growing environmental impact of cloud
computing, understanding the CWF of CI/CD services has become increasingly
important.
  This work investigates the CWF of using GitHub Actions, focusing on
open-source repositories where usage is free and unlimited for standard
runners. We build upon a methodology from the Cloud Carbon Footprint framework
and we use the largest dataset of workflow runs reported in the literature to
date, comprising over 2.2 million workflow runs from more than 18,000
repositories.
  Our analysis reveals that the GitHub Actions ecosystem results in a
substantial CWF. Our estimates for the carbon footprint in 2024 range from
150.5 MTCO2e in the most optimistic scenario to 994.9 MTCO2e in the most
pessimistic scenario, while the water footprint ranges from 1,989.6 to 37,664.5
kiloliters. The most likely scenario estimates are 456.9 MTCO2e for carbon
footprint and 5,738.2 kiloliters for water footprint. To provide perspective,
the carbon footprint in the most likely scenario is equivalent to the carbon
captured by 7,615 urban trees in a year, and the water footprint is comparable
to the water consumed by an average American family over 5,053 years.
  We explore strategies to mitigate this impact, primarily by reducing wasted
computational resources. Key recommendations include deploying runners in
regions whose energy production has a low environmental impact such as France
and the United Kingdom, implementing stricter deactivation policies for
scheduled runs and aligning their execution with periods when the regional
energy mix is more environmentally favorable, and reducing the size of
repositories.

### 2. [ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems](http://arxiv.org/pdf/2510.26475v1)

Authors: Qiaoling Chen, Zijun Liu, Peng Sun, Shenggui Li, Guoteng Wang, Ziming Liu, Yonggang Wen, Siyuan Feng, Tianwei Zhang

Adapting large language models (LLMs) via reinforcement learning (RL) is
often bottlenecked by the generation stage, which can consume over 75\% of the
training time. Speculative decoding (SD) accelerates autoregressive generation
in serving systems, but its behavior under RL training remains largely
unexplored. We identify three critical gaps that hinder the naive integration
of SD into RL systems: diminishing speedups at large batch sizes, drafter
staleness under continual actor updates, and drafter-induced policy
degradation.
  To address these gaps, we present ReSpec, a system that adapts SD to RL
through three complementary mechanisms: dynamically tuning SD configurations,
evolving the drafter via knowledge distillation, and weighting updates by
rollout rewards. On Qwen models (3B--14B), ReSpec achieves up to 4.5x speedup
while preserving reward convergence and training stability, providing a
practical solution for efficient RL-based LLM adaptation.

### 3. [An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning](http://arxiv.org/pdf/2510.26709v1)

Authors: Chuyan Chen, Chenyang Ma, Zhangxin Li, Yutong He, Yanjie Dong, Kun Yuan

Communication remains a central bottleneck in large-scale distributed machine
learning, and gradient sparsification has emerged as a promising strategy to
alleviate this challenge. However, existing gradient compressors face notable
limitations: Rand-$K$\ discards structural information and performs poorly in
practice, while Top-$K$\ preserves informative entries but loses the
contraction property and requires costly All-Gather operations. In this paper,
we propose ARC-Top-$K$, an {All-Reduce}-Compatible Top-$K$ compressor that
aligns sparsity patterns across nodes using a lightweight sketch of the
gradient, enabling index-free All-Reduce while preserving globally significant
information. ARC-Top-$K$\ is provably contractive and, when combined with
momentum error feedback (EF21M), achieves linear speedup and sharper
convergence rates than the original EF21M under standard assumptions.
Empirically, ARC-Top-$K$\ matches the accuracy of Top-$K$\ while reducing
wall-clock training time by up to 60.7\%, offering an efficient and scalable
solution that combines the robustness of Rand-$K$\ with the strong performance
of Top-$K$.

### 4. [Wireless Sensor Networks as Parallel and Distributed Hardware Platform for Artificial Neural Networks](http://arxiv.org/pdf/2510.26492v1)

Authors: Gursel Serpen

We are proposing fully parallel and maximally distributed hardware
realization of a generic neuro-computing system. More specifically, the
proposal relates to the wireless sensor networks technology to serve as a
massively parallel and fully distributed hardware platform to implement and
realize artificial neural network (ANN) algorithms. A parallel and distributed
(PDP) hardware realization of ANNs makes it possible to have real time
computation of large-scale (and complex) problems in a highly robust framework.
We will demonstrate how a network of hundreds of thousands of processing nodes
(or motes of a wireless sensor network), which have on-board processing and
wireless communication features, can be used to implement fully parallel and
massively distributed computation of artificial neural network algorithms for
solution of truly large-scale problems in real time. The realization of
artificial neural network algorithms in a massively parallel and fully
distributed hardware has been the goal of neural network computing researchers.
This is because a parallel and distributed computation of artificial neural
network algorithms could not have been achieved against all the advancements in
silicon- or optics-based computing. Accordingly, artificial neural networks
could not be applied to very large-scale problems for real time computation of
solutions. This hindered the development of neural algorithms for affordable
and practical solutions of challenging problems since often special-purpose
computing approaches in hardware, software or hybrid (non-neural) had to be
developed for and fine-tuned to specific problems that are very large-scale and
highly complex. Successful implementation is likely to revolutionize computing
as we know it by making it possible to solve very large scale scientific,
engineering or technical problems in real time.

### 5. [ExpertFlow: Adaptive Expert Scheduling and Memory Coordination for Efficient MoE Inference](http://arxiv.org/pdf/2510.26730v1)

Authors: Zixu Shen, Kexin Chu, Yifan Zhang, Dawei Xiang, Runxin Wu, Wei Zhang

The expansion of large language models is increasingly limited by the
constrained memory capacity of modern GPUs. To mitigate this,
Mixture-of-Experts (MoE) architectures activate only a small portion of
parameters during inference, significantly lowering both memory demand and
computational overhead. However, conventional MoE inference approaches, which
select active experts independently at each layer, often introduce considerable
latency because of frequent parameter transfers between host and GPU memory. In
addition, current cross-layer prediction strategies, which are typically based
on fixed steps, lack adaptability across different hardware platforms and
workloads, thereby reducing their robustness and effectiveness.
  To address these challenges, we present ExpertFlow, a runtime system for MoE
inference that combines adaptive expert prefetching and cache-aware routing.
ExpertFlow continuously adjusts its prediction horizon for expert activation by
leveraging runtime statistics such as transfer bandwidth, parameter
dimensionality, and model feedback signals. Furthermore, it incorporates a
hybrid cross-layer prediction scheme that fuses pregating information with
intermediate computational states to anticipate future expert needs. By
adaptively refining prefetching decisions and aligning them with actual usage
behavior, ExpertFlow effectively decreases cache misses and removes latency
caused by expert swap-ins. Our evaluation demonstrates that ExpertFlow reduces
model stall time to less than 0.1% of the baseline, highlighting its capability
to optimize MoE inference under stringent memory constraints.

### 6. [Non-Convex Over-the-Air Heterogeneous Federated Learning: A Bias-Variance Trade-off](http://arxiv.org/pdf/2510.26722v1)

Authors: Muhammad Faraz Ul Abrar, Nicolò Michelusi

Over-the-air (OTA) federated learning (FL) has been well recognized as a
scalable paradigm that exploits the waveform superposition of the wireless
multiple-access channel to aggregate model updates in a single use. Existing
OTA-FL designs largely enforce zero-bias model updates by either assuming
\emph{homogeneous} wireless conditions (equal path loss across devices) or
forcing zero-bias updates to guarantee convergence. Under \emph{heterogeneous}
wireless scenarios, however, such designs are constrained by the weakest device
and inflate the update variance. Moreover, prior analyses of biased OTA-FL
largely address convex objectives, while most modern AI models are highly
non-convex. Motivated by these gaps, we study OTA-FL with stochastic gradient
descent (SGD) for general smooth non-convex objectives under wireless
heterogeneity. We develop novel OTA-FL SGD updates that allow a structured,
time-invariant model bias while facilitating reduced variance updates. We
derive a finite-time stationarity bound (expected time average squared gradient
norm) that explicitly reveals a bias-variance trade-off. To optimize this
trade-off, we pose a non-convex joint OTA power-control design and develop an
efficient successive convex approximation (SCA) algorithm that requires only
statistical CSI at the base station. Experiments on a non-convex image
classification task validate the approach: the SCA-based design accelerates
convergence via an optimized bias and improves generalization over prior OTA-FL
baselines.

### Discrete Mathematics

### 1. [Bijections Between Smirnov Words and Hamiltonian Cycles in Complete Multipartite Graphs](http://arxiv.org/pdf/2510.26597v1)

Authors: El-Mehdi Mehiri

We establish a bijective correspondence between Smirnov words with balanced
letter multiplicities and Hamiltonian paths in complete $m$-partite graphs
$K_{n,n,\ldots,n}$. This bijection allows us to derive closed
inclusion-exclusion formulas for the number of Hamiltonian cycles in such
graphs. We further extend the enumeration to the generalized nonuniform case
$K_{n_1,n_2,\ldots,n_m}$. We also provide an asymptotic analysis based on
Stirling's approximation, which yields compact factorial expressions and
logarithmic expansions describing the growth of the number of Hamiltonian
cycles in the considered graphs. Our approach unifies the combinatorial study
of adjacency-constrained words and the enumeration of Hamiltonian cycles within
a single analytical framework.

### 2. [The Strong Birthday Problem Revisited](http://arxiv.org/pdf/2510.26056v1)

Authors: Chijul B. Tripathy

We revisit the Strong Birthday Problem (SBP) introduced in [1]. The problem
is stated as follows: what is the minimum number of people we have to choose so
that everyone has a shared birthday with probability at least 1/2? We derive
recurrence relations to compute the probability, and further show a nice
connection to the associated Stirling numbers of the second kind to derive
additional recurrences. We implement the recurrences using dynamic programming
as well as compute the values using the combinatorial formula, and provide
numerical results.

### 3. [On the number of non-degenerate canalizing Boolean functions](http://arxiv.org/pdf/2510.26556v1)

Authors: Claus Kadelka

Canalization is a key organizing principle in complex systems, particularly
in gene regulatory networks. It describes how certain input variables exert
dominant control over a function's output, thereby imposing hierarchical
structure and conferring robustness to perturbations. Degeneracy, in contrast,
captures redundancy among input variables and reflects the complete dominance
of some variables by others. Both properties influence the stability and
dynamics of discrete dynamical systems, yet their combinatorial underpinnings
remain incompletely understood. Here, we derive recursive formulas for counting
Boolean functions with prescribed numbers of essential variables and given
canalizing properties. In particular, we determine the number of non-degenerate
canalizing Boolean functions -- that is, functions for which all variables are
essential and at least one variable is canalizing. Our approach extends earlier
enumeration results on canalizing and nested canalizing functions. It provides
a rigorous foundation for quantifying how frequently canalization occurs among
random Boolean functions and for assessing its pronounced over-representation
in biological network models, where it contributes to both robustness and to
the emergence of distinct regulatory roles.

### 4. [Tensor decomposition beyond uniqueness, with an application to the minrank problem](http://arxiv.org/pdf/2510.26587v1)

Authors: Pascal Koiran, Rafael Oliveira

We prove a generalization to Jennrich's uniqueness theorem for tensor
decompositions in the undercomplete setting. Our uniqueness theorem is based on
an alternative definition of the standard tensor decomposition, which we call
matrix-vector decomposition. Moreover, in the same settings in which our
uniqueness theorem applies, we also design and analyze an efficient randomized
algorithm to compute the unique minimum matrix-vector decomposition (and thus a
tensor rank decomposition of minimum rank).
  As an application of our uniqueness theorem and our efficient algorithm, we
show how to compute all matrices of minimum rank (up to scalar multiples) in
certain generic vector spaces of matrices.

### Data Structures and Algorithms

### 1. [Space-Efficient k-Mismatch Text Indexes](http://arxiv.org/pdf/2510.26264v1)

Authors: Tomasz Kociumaka, Jakub Radoszewski

A central task in string processing is text indexing, where the goal is to
preprocess a text (a string of length $n$) into an efficient index (a data
structure) supporting queries about the text. Cole, Gottlieb, and Lewenstein
(STOC 2004) proposed $k$-errata trees, a family of text indexes supporting
approximate pattern matching queries of several types. In particular,
$k$-errata trees yield an elegant solution to $k$-mismatch queries, where we
are to report all substrings of the text with Hamming distance at most $k$ to
the query pattern. The resulting $k$-mismatch index uses $O(n\log^k n)$ space
and answers a query for a length-$m$ pattern in $O(\log^k n \log \log n + m +
occ)$ time, where $occ$ is the number of approximate occurrences.
  In retrospect, $k$-errata trees appear very well optimized: even though a
large body of work has adapted $k$-errata trees to various settings throughout
the past two decades, the original time-space trade-off for $k$-mismatch
indexing has not been improved in the general case. We present the first such
improvement, a $k$-mismatch index with $O(n\log^{k-1} n)$ space and the same
query time as $k$-errata trees.
  Previously, due to a result of Chan, Lam, Sung, Tam, and Wong (Algorithmica
2010), such an $O(n\log^{k-1} n)$-size index has been known only for texts over
alphabets of constant size. In this setting, however, we obtain an even smaller
$k$-mismatch index of size only $O(n \log^{k-2+\varepsilon+\frac{2}{k+2-(k
\bmod 2)}} n)\subseteq O(n\log^{k-1.5+\varepsilon} n)$ for $2\le k\le O(1)$ and
any constant $\varepsilon>0$. Along the way, we also develop improved indexes
for short patterns, offering better trade-offs in this practically relevant
special case.

### 2. [Tight Differentially Private PCA via Matrix Coherence](http://arxiv.org/pdf/2510.26679v1)

Authors: Tommaso d'Orsi, Gleb Novikov

We revisit the task of computing the span of the top $r$ singular vectors
$u_1, \ldots, u_r$ of a matrix under differential privacy. We show that a
simple and efficient algorithm -- based on singular value decomposition and
standard perturbation mechanisms -- returns a private rank-$r$ approximation
whose error depends only on the \emph{rank-$r$ coherence} of $u_1, \ldots, u_r$
and the spectral gap $\sigma_r - \sigma_{r+1}$. This resolves a question posed
by Hardt and Roth~\cite{hardt2013beyond}. Our estimator outperforms the state
of the art -- significantly so in some regimes. In particular, we show that in
the dense setting, it achieves the same guarantees for single-spike PCA in the
Wishart model as those attained by optimal non-private algorithms, whereas
prior private algorithms failed to do so.
  In addition, we prove that (rank-$r$) coherence does not increase under
Gaussian perturbations. This implies that any estimator based on the Gaussian
mechanism -- including ours -- preserves the coherence of the input. We
conjecture that similar behavior holds for other structured models, including
planted problems in graphs.
  We also explore applications of coherence to graph problems. In particular,
we present a differentially private algorithm for Max-Cut and other constraint
satisfaction problems under low coherence assumptions.

### 3. [On Purely Private Covariance Estimation](http://arxiv.org/pdf/2510.26717v1)

Authors: Tommaso d'Orsi, Gleb Novikov

We present a simple perturbation mechanism for the release of $d$-dimensional
covariance matrices $\Sigma$ under pure differential privacy. For large
datasets with at least $n\geq d^2/\varepsilon$ elements, our mechanism recovers
the provably optimal Frobenius norm error guarantees of
\cite{nikolov2023private}, while simultaneously achieving best known error for
all other $p$-Schatten norms, with $p\in [1,\infty]$. Our error is
information-theoretically optimal for all $p\ge 2$, in particular, our
mechanism is the first purely private covariance estimator that achieves
optimal error in spectral norm.
  For small datasets $n< d^2/\varepsilon$, we further show that by projecting
the output onto the nuclear norm ball of appropriate radius, our algorithm
achieves the optimal Frobenius norm error $O(\sqrt{d\;\text{Tr}(\Sigma) /n})$,
improving over the known bounds of $O(\sqrt{d/n})$ of \cite{nikolov2023private}
and ${O}\big(d^{3/4}\sqrt{\text{Tr}(\Sigma)/n}\big)$ of
\cite{dong2022differentially}.

### 4. [The Strong Birthday Problem Revisited](http://arxiv.org/pdf/2510.26056v1)

Authors: Chijul B. Tripathy

We revisit the Strong Birthday Problem (SBP) introduced in [1]. The problem
is stated as follows: what is the minimum number of people we have to choose so
that everyone has a shared birthday with probability at least 1/2? We derive
recurrence relations to compute the probability, and further show a nice
connection to the associated Stirling numbers of the second kind to derive
additional recurrences. We implement the recurrences using dynamic programming
as well as compute the values using the combinatorial formula, and provide
numerical results.

### 5. [Tensor decomposition beyond uniqueness, with an application to the minrank problem](http://arxiv.org/pdf/2510.26587v1)

Authors: Pascal Koiran, Rafael Oliveira

We prove a generalization to Jennrich's uniqueness theorem for tensor
decompositions in the undercomplete setting. Our uniqueness theorem is based on
an alternative definition of the standard tensor decomposition, which we call
matrix-vector decomposition. Moreover, in the same settings in which our
uniqueness theorem applies, we also design and analyze an efficient randomized
algorithm to compute the unique minimum matrix-vector decomposition (and thus a
tensor rank decomposition of minimum rank).
  As an application of our uniqueness theorem and our efficient algorithm, we
show how to compute all matrices of minimum rank (up to scalar multiples) in
certain generic vector spaces of matrices.

### 6. [Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](http://arxiv.org/pdf/2510.26324v1)

Authors: Zhiyang Xun, Shivam Gupta, Eric Price

Given a noisy linear measurement $y = Ax + \xi$ of a distribution $p(x)$, and
a good approximation to the prior $p(x)$, when can we sample from the posterior
$p(x \mid y)$? Posterior sampling provides an accurate and fair framework for
tasks such as inpainting, deblurring, and MRI reconstruction, and several
heuristics attempt to approximate it. Unfortunately, approximate posterior
sampling is computationally intractable in general.
  To sidestep this hardness, we focus on (local or global) log-concave
distributions $p(x)$. In this regime, Langevin dynamics yields posterior
samples when the exact scores of $p(x)$ are available, but it is brittle to
score--estimation error, requiring an MGF bound (sub-exponential error). By
contrast, in the unconditional setting, diffusion models succeed with only an
$L^2$ bound on the score error. We prove that combining diffusion models with
an annealed variant of Langevin dynamics achieves conditional sampling in
polynomial time using merely an $L^4$ bound on the score error.

### Emerging Technologies

### 1. [Structurally Valid Log Generation using FSM-GFlowNets](http://arxiv.org/pdf/2510.26197v1)

Authors: Riya Samanta

Generating structurally valid and behaviorally diverse synthetic event logs
for interaction-aware models is a challenging yet crucial problem, particularly
in settings with limited or privacy constrained user data. Existing methods
such as heuristic simulations and LLM based generators often lack structural
coherence or controllability, producing synthetic data that fails to accurately
represent real world system interactions. This paper presents a framework that
integrates Finite State Machines or FSMs with Generative Flow Networks or
GFlowNets to generate structured, semantically valid, and diverse synthetic
event logs. Our FSM-constrained GFlowNet ensures syntactic validity and
behavioral variation through dynamic action masking and guided sampling. The
FSM, derived from expert traces, encodes domain-specific rules, while the
GFlowNet is trained using a flow matching objective with a hybrid reward
balancing FSM compliance and statistical fidelity. We instantiate the framework
in the context of UI interaction logs using the UIC HCI dataset, but the
approach generalizes to any symbolic sequence domain. Experimental results
based on distributional metrics show that our FSM GFlowNet produces realistic,
structurally consistent logs, achieving, for instance, under the real user logs
baseline, a KL divergence of 0.2769 and Chi squared distance of 0.3522,
significantly outperforming GPT-4o's 2.5294/13.8020 and Gemini's
3.7233/63.0355, alongside a leading bigram overlap of 0.1214 vs. GPT 4o's
0.0028 and Gemini's 0.0007. A downstream use case intent classification
demonstrates that classifiers trained solely on our synthetic logs produced
from FSM-GFlowNet achieve competitive accuracy compared to real data.

### 2. [Tackling the Challenges of Adding Pulse-level Support to a Heterogeneous HPCQC Software Stack: MQSS Pulse](http://arxiv.org/pdf/2510.26565v1)

Authors: Jorge Echavarria, Muhammad Nufail Farooqi, Amit Devra, Santana Lujan, Léo Van Damme, Hossam Ahmed, Martín Letras, Ercüment Kaya, Adrian Vetter, Max Werninghaus, Martin Knudsen, Felix Rohde, Albert Frisch, Eric Mansfield, Rakhim Davletkaliyev, Vladimir Kukushkin, Noora Färkkilä, Janne Mäntylä, Nikolas Pomplun, Andreas Spörl, Lukas Burgholzer, Yannick Stade, Robert Wille, Laura B. Schulz, Martin Schulz

We study the problem of adding native pulse-level control to heterogeneous
High Performance Computing-Quantum Computing (HPCQC) software stacks, using the
Munich Quantum Software Stack (MQSS) as a case study. The goal is to expand the
capabilities of HPCQC environments by offering the ability for low-level access
and control, currently typically not foreseen for such hybrid systems. For
this, we need to establish new interfaces that integrate such pulse-level
control into the lower layers of the software stack, including the need for
proper representation.
  Pulse-level quantum programs can be fully described with only three low-level
abstractions: ports (input/output channels), frames (reference signals), and
waveforms (pulse envelopes). We identify four key challenges to represent those
pulse abstractions at: the user-interface level, at the compiler level
(including the Intermediate Representation (IR)), and at the backend-interface
level (including the appropriate exchange format). For each challenge, we
propose concrete solutions in the context of MQSS. These include introducing a
compiled (C/C++) pulse Application Programming Interface (API) to overcome
Python runtime overhead, extending its LLVM support to include pulse-related
instructions, using its C-based backend interface to query relevant hardware
constraints, and designing a portable exchange format for pulse sequences. Our
integrated approach provides an end-to-end path for pulse-aware compilation and
runtime execution in HPCQC environments. This work lays out the architectural
blueprint for extending HPCQC integration to support pulse-level quantum
operations without disrupting state-of-the-art classical workflows.

### 3. [A Research Roadmap for Augmenting Software Engineering Processes and Software Products with Generative AI](http://arxiv.org/pdf/2510.26275v1)

Authors: Domenico Amalfitano, Andreas Metzger, Marco Autili, Tommaso Fulcini, Tobias Hey, Jan Keim, Patrizio Pelliccione, Vincenzo Scotti, Anne Koziolek, Raffaela Mirandola, Andreas Vogelsang

Generative AI (GenAI) is rapidly transforming software engineering (SE)
practices, influencing how SE processes are executed, as well as how software
systems are developed, operated, and evolved. This paper applies design science
research to build a roadmap for GenAI-augmented SE. The process consists of
three cycles that incrementally integrate multiple sources of evidence,
including collaborative discussions from the FSE 2025 "Software Engineering
2030" workshop, rapid literature reviews, and external feedback sessions
involving peers. McLuhan's tetrads were used as a conceptual instrument to
systematically capture the transforming effects of GenAI on SE processes and
software products.The resulting roadmap identifies four fundamental forms of
GenAI augmentation in SE and systematically characterizes their related
research challenges and opportunities. These insights are then consolidated
into a set of future research directions. By grounding the roadmap in a
rigorous multi-cycle process and cross-validating it among independent author
teams and peers, the study provides a transparent and reproducible foundation
for analyzing how GenAI affects SE processes, methods and tools, and for
framing future research within this rapidly evolving area. Based on these
findings, the article finally makes ten predictions for SE in the year 2030.

### Formal Languages and Automata Theory

### 1. [Unambiguous Acceptance of Thin Coalgebras](http://arxiv.org/pdf/2510.26371v1)

Authors: Anton Chernev, Corina Cîrstea, Helle Hvid Hansen, Clemens Kupke

Automata admitting at most one accepting run per structure, known as
unambiguous automata, find applications in verification of reactive systems as
they extend the class of deterministic automata whilst maintaining some of
their desirable properties. In this paper, we generalise a classical
construction of unambiguous automata from thin trees to thin coalgebras for
analytic functors. This achieves two goals: extending the existing construction
to a larger class of structures, and providing conceptual clarity and
parametricity to the construction by formalising it in the coalgebraic
framework. As part of the construction, we link automaton acceptance of
languages of thin coalgebras to language recognition via so-called coherent
algebras, which were previously introduced for studying thin coalgebras. This
link also allows us to establish an automata-theoretic characterisation of
languages recognised by finite coherent algebras.

### 2. [Finding Regular Herbrand Models for CHCs using Answer Set Programming](http://arxiv.org/pdf/2510.26428v1)

Authors: Gregoire Maire, Thomas Genet

We are interested in proving satisfiability of Constrained Horn Clauses
(CHCs) over Algebraic Data Types (ADTs). We propose to prove satisfiability by
building a tree automaton recognizing the Herbrand model of the CHCs. If such
an automaton exists then the model is said to be regular, i.e., the Herbrand
model is a regular set of atoms. Kostyukov et al. have shown how to derive an
automaton when CVC4 finds a finite model of the CHCs. We propose an alternative
way to build the automaton using an encoding into a SAT problem using Clingo,
an Answer Set Programming (ASP) tool. We implemented a translation of CHCs with
ADTs into an ASP problem. Combined with Clingo, we obtain a semi-complete
satisfiability checker: it finds a tree automaton if a regular Herbrand model
exists or finds a counter-example if the problem is unsatisfiable.

### Graphics

### 1. [StructLayoutFormer:Conditional Structured Layout Generation via Structure Serialization and Disentanglement](http://arxiv.org/pdf/2510.26141v1)

Authors: Xin Hu, Pengfei Xu, Jin Zhou, Hongbo Fu, Hui Huang

Structured layouts are preferable in many 2D visual contents (\eg, GUIs,
webpages) since the structural information allows convenient layout editing.
Computational frameworks can help create structured layouts but require heavy
labor input. Existing data-driven approaches are effective in automatically
generating fixed layouts but fail to produce layout structures. We present
StructLayoutFormer, a novel Transformer-based approach for conditional
structured layout generation. We use a structure serialization scheme to
represent structured layouts as sequences. To better control the structures of
generated layouts, we disentangle the structural information from the element
placements. Our approach is the first data-driven approach that achieves
conditional structured layout generation and produces realistic layout
structures explicitly. We compare our approach with existing data-driven layout
generation approaches by including post-processing for structure extraction.
Extensive experiments have shown that our approach exceeds these baselines in
conditional structured layout generation. We also demonstrate that our approach
is effective in extracting and transferring layout structures. The code is
publicly available at %\href{https://github.com/Teagrus/StructLayoutFormer}
{https://github.com/Teagrus/StructLayoutFormer}.

### 2. [Look at That Distractor: Dynamic Translation Gain under Low Perceptual Load in Virtual Reality](http://arxiv.org/pdf/2510.26265v1)

Authors: Ling-Long Zou, Qiang Tong, Er-Xia Luo, Sen-Zhe Xu, Song-Hai Zhang, Fang-Lue Zhang

Redirected walking utilizes gain adjustments within perceptual thresholds to
allow natural navigation in large scale virtual environments within confined
physical environments. Previous research has found that when users are
distracted by some scene elements, they are less sensitive to gain values.
However, the effects on detection thresholds have not been quantitatively
measured. In this paper, we present a novel method that dynamically adjusts
translation gain by leveraging visual distractors. We place distractors within
the user's field of view and apply a larger translation gain when their
attention is drawn to them. Because the magnitude of gain adjustment depends on
the user's level of engagement with the distractors, the redirection process
remains smooth and unobtrusive. To evaluate our method, we developed a task
oriented virtual environment for a user study. Results show that introducing
distractors in the virtual environment significantly raises users' translation
gain thresholds. Furthermore, assessments using the Simulator Sickness
Questionnaire and Igroup Presence Questionnaire indicate that the method
maintains user comfort and acceptance, supporting its effectiveness for RDW
systems.

### 3. [The Impact and Outlook of 3D Gaussian Splatting](http://arxiv.org/pdf/2510.26694v1)

Authors: Bernhard Kerbl

Since its introduction, 3D Gaussian Splatting (3DGS) has rapidly transformed
the landscape of 3D scene representations, inspiring an extensive body of
associated research. Follow-up work includes analyses and contributions that
enhance the efficiency, scalability, and real-world applicability of 3DGS. In
this summary, we present an overview of several key directions that have
emerged in the wake of 3DGS. We highlight advances enabling resource-efficient
training and rendering, the evolution toward dynamic (or four-dimensional,
4DGS) representations, and deeper exploration of the mathematical foundations
underlying its appearance modeling and rendering process. Furthermore, we
examine efforts to bring 3DGS to mobile and virtual reality platforms, its
extension to massive-scale environments, and recent progress toward
near-instant radiance field reconstruction via feed-forward or distributed
computation. Collectively, these developments illustrate how 3DGS has evolved
from a breakthrough representation into a versatile and foundational tool for
3D vision and graphics.

### 4. [SEE4D: Pose-Free 4D Generation via Auto-Regressive Video Inpainting](http://arxiv.org/pdf/2510.26796v1)

Authors: Dongyue Lu, Ao Liang, Tianxin Huang, Xiao Fu, Yuyang Zhao, Baorui Ma, Liang Pan, Wei Yin, Lingdong Kong, Wei Tsang Ooi, Ziwei Liu

Immersive applications call for synthesizing spatiotemporal 4D content from
casual videos without costly 3D supervision. Existing video-to-4D methods
typically rely on manually annotated camera poses, which are labor-intensive
and brittle for in-the-wild footage. Recent warp-then-inpaint approaches
mitigate the need for pose labels by warping input frames along a novel camera
trajectory and using an inpainting model to fill missing regions, thereby
depicting the 4D scene from diverse viewpoints. However, this
trajectory-to-trajectory formulation often entangles camera motion with scene
dynamics and complicates both modeling and inference. We introduce SEE4D, a
pose-free, trajectory-to-camera framework that replaces explicit trajectory
prediction with rendering to a bank of fixed virtual cameras, thereby
separating camera control from scene modeling. A view-conditional video
inpainting model is trained to learn a robust geometry prior by denoising
realistically synthesized warped images and to inpaint occluded or missing
regions across virtual viewpoints, eliminating the need for explicit 3D
annotations. Building on this inpainting core, we design a spatiotemporal
autoregressive inference pipeline that traverses virtual-camera splines and
extends videos with overlapping windows, enabling coherent generation at
bounded per-step complexity. We validate See4D on cross-view video generation
and sparse reconstruction benchmarks. Across quantitative metrics and
qualitative assessments, our method achieves superior generalization and
improved performance relative to pose- or trajectory-conditioned baselines,
advancing practical 4D world modeling from casual videos.

### 5. [HEIR: Learning Graph-Based Motion Hierarchies](http://arxiv.org/pdf/2510.26786v1)

Authors: Cheng Zheng, William Koch, Baiang Li, Felix Heide

Hierarchical structures of motion exist across research fields, including
computer vision, graphics, and robotics, where complex dynamics typically arise
from coordinated interactions among simpler motion components. Existing methods
to model such dynamics typically rely on manually-defined or heuristic
hierarchies with fixed motion primitives, limiting their generalizability
across different tasks. In this work, we propose a general hierarchical motion
modeling method that learns structured, interpretable motion relationships
directly from data. Our method represents observed motions using graph-based
hierarchies, explicitly decomposing global absolute motions into
parent-inherited patterns and local motion residuals. We formulate hierarchy
inference as a differentiable graph learning problem, where vertices represent
elemental motions and directed edges capture learned parent-child dependencies
through graph neural networks. We evaluate our hierarchical reconstruction
approach on three examples: 1D translational motion, 2D rotational motion, and
dynamic 3D scene deformation via Gaussian splatting. Experimental results show
that our method reconstructs the intrinsic motion hierarchy in 1D and 2D cases,
and produces more realistic and interpretable deformations compared to the
baseline on dynamic 3D Gaussian splatting scenes. By providing an adaptable,
data-driven hierarchical modeling paradigm, our method offers a formulation
applicable to a broad range of motion-centric tasks. Project Page:
https://light.princeton.edu/HEIR/

### 6. [OmniX: From Unified Panoramic Generation and Perception to Graphics-Ready 3D Scenes](http://arxiv.org/pdf/2510.26800v1)

Authors: Yukun Huang, Jiwen Yu, Yanning Zhou, Jianan Wang, Xintao Wang, Pengfei Wan, Xihui Liu

There are two prevalent ways to constructing 3D scenes: procedural generation
and 2D lifting. Among them, panorama-based 2D lifting has emerged as a
promising technique, leveraging powerful 2D generative priors to produce
immersive, realistic, and diverse 3D environments. In this work, we advance
this technique to generate graphics-ready 3D scenes suitable for physically
based rendering (PBR), relighting, and simulation. Our key insight is to
repurpose 2D generative models for panoramic perception of geometry, textures,
and PBR materials. Unlike existing 2D lifting approaches that emphasize
appearance generation and ignore the perception of intrinsic properties, we
present OmniX, a versatile and unified framework. Based on a lightweight and
efficient cross-modal adapter structure, OmniX reuses 2D generative priors for
a broad range of panoramic vision tasks, including panoramic perception,
generation, and completion. Furthermore, we construct a large-scale synthetic
panorama dataset containing high-quality multimodal panoramas from diverse
indoor and outdoor scenes. Extensive experiments demonstrate the effectiveness
of our model in panoramic visual perception and graphics-ready 3D scene
generation, opening new possibilities for immersive and physically realistic
virtual world generation.

### Computer Science and Game Theory

### 1. [Engineering Social Optimality via Utility Shaping in Non-Cooperative Games under Incomplete Information and Imperfect Monitoring](http://arxiv.org/pdf/2510.26033v1)

Authors: David Smith, Jie Dong, Yizhou Yang

In this paper, we study decentralized decision-making where agents optimize
private objectives under incomplete information and imperfect public
monitoring, in a non-cooperative setting. By shaping utilities-embedding shadow
prices or Karush-Kuhn-Tucker(KKT)-aligned penalties-we make the stage game an
exact-potential game whose unique equilibrium equals the (possibly constrained)
social optimum. We characterize the Bayesian equilibrium as a stochastic
variational inequality; strong monotonicity follows from a single-inflection
compressed/stretched-exponential response combined with convex pricing. We give
tracking bounds for damped-gradient and best-response-with-hysteresis updates
under a noisy public index, and corresponding steady-state error. The framework
accommodates discrete and continuous action sets and composes with slower
discrete assignment. Deployable rules include: embed prices/penalties; publish
a single public index; tune steps, damping, and dual rates for contraction.
Computational experiments cover (i) a multi-tier supply chain and (ii) a
non-cooperative agentic-AI compute market of bidding bots. Relative to
price-only baselines, utility shaping attains near-centralized welfare,
eliminates steady-state constraint/capacity violations when feasible, and
accelerates convergence; with quantization, discrete equilibria track
continuous ones within the mesh. The blueprint is portable to demand response,
cloud/edge scheduling, and transportation pricing and biosecurity/agriculture.
Overall, utility shaping plus a public index implements the constrained social
optimum with stable equilibria under noise and drift-an
operations-research-friendly alternative to heavy messaging or full mechanism
design.

### 2. [NP-Hardness of Approximating Nash Social Welfare with Supermodular Valuations](http://arxiv.org/pdf/2510.26055v1)

Authors: Alon Bebchuk

We study the problem of allocating a set of indivisible items to agents with
supermodular utilities to maximize the Nash social welfare. We show that the
problem is NP-hard for any approximation factor.

### 3. [Proxemics and Permeability of the Pedestrian Group](http://arxiv.org/pdf/2510.26571v1)

Authors: Saleh Albeaik, Faisal Alsallum, Mohamad Alrished

People tend to walk in groups, and interactions with those groups have a
significant impact on crowd behavior and pedestrian traffic dynamics. Social
norms can be seen as unwritten rules regulating people interactions in social
settings. This article studies people interactions with groups and the
emergence of group proxemics. Group zones, zone occupancy counts and people
clearance from the group are studied using naturalistic data. Analysis indicate
potential presence of three different zones in addition to the public zone.
People tend to remain in the public zone and only progressively get closer to
groups, and those closer approaches happen in a low frequency and for brief
periods of time.

### Human-Computer Interaction

### 1. [FractalBrain: A Neuro-interactive Virtual Reality Experience using Electroencephalogram (EEG) for Mindfulness](http://arxiv.org/pdf/2510.26041v1)

Authors: Jamie Ngoc Dinh, You-Jin Kim, Myungin Lee

Mindfulness has been studied and practiced in enhancing psychological
well-being while reducing neuroticism and psychopathological indicators.
However, practicing mindfulness with continuous attention is challenging,
especially for beginners. In the proposed system, FractalBrain, we utilize an
interactive audiovisual fractal with a geometric repetitive pattern that has
been demonstrated to induce meditative effects. FractalBrain presents an
experience combining a surreal virtual reality (VR) program with an
electroencephalogram (EEG) interface. While viewing an ever-changing
fractal-inspired artwork in an immersive environment, the user's EEG stream is
analyzed and mapped into VR. These EEG data adaptively manipulates the
audiovisual parameters in real-time, generating a distinct experience for each
user. The pilot feedback suggests the potential of the FractalBrain to
facilitate mindfulness and enhance attention.

### 2. [Interaction-Augmented Instruction: Modeling the Synergy of Prompts and Interactions in Human-GenAI Collaboration](http://arxiv.org/pdf/2510.26069v1)

Authors: Leixian Shen, Yifang Wang, Huamin Qu, Xing Xie, Haotian Li

Text prompt is the most common way for human-generative AI (GenAI)
communication. Though convenient, it is challenging to convey fine-grained and
referential intent. One promising solution is to combine text prompts with
precise GUI interactions, like brushing and clicking. However, there lacks a
formal model to model synergistic designs between prompts and interactions,
hindering their comparison and innovation. To fill this gap, via an iterative
and deductive process, we develop the Interaction-Augmented Instruction (IAI)
model, a compact entity-relation graph formalizing how the combination of
interactions and text prompts enhances human-generative AI communication. With
the model, we distill twelve recurring and composable atomic interaction
paradigms from prior tools, verifying our model's capability to facilitate
systematic design characterization and comparison. Case studies further
demonstrate the model's utility in applying, refining, and extending these
paradigms. These results illustrate our IAI model's descriptive,
discriminative, and generative power for shaping future GenAI systems.

### 3. [Avatar Appearance Beyond Pixels -- User Ratings and Avatar Preferences within Health Applications](http://arxiv.org/pdf/2510.26251v1)

Authors: Navid Ashrafi, Philipp Graf, Manuela Marquardt, Francesco Vona, Julia Schorlemmer, Jan-Niklas Voigt-Antons

The appearance of a virtual avatar significantly influences its perceived
appropriateness and the user's experience, particularly in healthcare
applications. This study analyzed interactions with six avatars of varying
characteristics in a patient-reported outcome measures (PROMs) application to
investigate correlations between avatar ratings and user preferences.
Forty-seven participants completed a healthcare survey involving 30 PROMIS
items (Global Health and Physical Function) and then rated the avatars on
warmth, competence, attractiveness, and human-likeness, as well as their
willingness to share personal data. The results showed that competence was the
most critical factor in avatar selection, while human-likeness had minimal
impact on health data disclosure. Gender did not significantly affect the
ratings, but clothing style played a key role, with male avatars in
professional attire rated higher in competence due to gender-stereotypical
expectations. In contrast, professional female avatars were rated lower in
warmth and attractiveness. These findings underline the importance of
thoughtful avatar design in healthcare applications to enhance user experience
and engagement.

### 4. [Scaffolding Creativity: How Divergent and Convergent LLM Personas Shape Human Machine Creative Problem-Solving](http://arxiv.org/pdf/2510.26490v1)

Authors: Alon Rosenbaum, Yigal David, Eran Kaufman, Gilad Ravid, Amit Ronen, Assaf Krebs

Large language models (LLMs) are increasingly shaping creative work and
problem-solving; however, prior research suggests that they may diminish
unassisted creativity. To address this tension, a coach-like LLM environment
was developed that embodies divergent and convergent thinking personas as two
complementary processes. Effectiveness and user behavior were assessed through
a controlled experiment in which participants interacted with either persona,
while a control group engaged with a standard LLM providing direct answers.
  Notably, users' perceptions of which persona best supported their creativity
often diverged from objective performance measures. Trait-based analyses
revealed that individual differences predict when people utilize divergent
versus convergent personas, suggesting opportunities for adaptive sequencing.
Furthermore, interaction patterns reflected the design thinking model,
demonstrating how persona-guided support shapes creative problem-solving.
  Our findings provide design principles for creativity support systems that
strike a balance between exploration and convergence through persona-based
guidance and personalization. These insights advance human-AI collaboration
tools that scaffold rather than overshadow human creativity.

### 5. [Metacognition and Confidence Dynamics in Advice Taking from Generative AI](http://arxiv.org/pdf/2510.26508v1)

Authors: Clara Colombatto, Sean Rintel, Lev Tankelevitch

Generative Artificial Intelligence (GenAI) can aid humans in a wide range of
tasks, but its effectiveness critically depends on users being able to evaluate
the accuracy of GenAI outputs and their own expertise. Here we asked how
confidence in self and GenAI contributes to decisions to seek and rely on
advice from GenAI ('prospective confidence'), and how advice-taking in turn
shapes this confidence ('retrospective confidence'). In a novel paradigm
involving text generation, participants formulated plans for events, and could
request advice from a GenAI (Study 1; N=200) or were randomly assigned to
receive advice (Study 2; N=300), which they could rely on or ignore. Advice
requests in Study 1 were related to higher prospective confidence in GenAI and
lower confidence in self. Advice-seekers showed increased retrospective
confidence in GenAI, while those who declined advice showed increased
confidence in self. Random assignment in Study 2 revealed that advice exposure
increases confidence in GenAI and in self, suggesting that GenAI advice-taking
causally boosts retrospective confidence. These results were mirrored in advice
reliance, operationalised as the textual similarity between GenAI advice and
participants' responses, with reliance associated with increased retrospective
confidence in both GenAI and self. Critically, participants who chose to
obtain/rely on advice provided more detailed responses (likely due to the
output's verbosity), but failed to check the output thoroughly, missing key
information. These findings underscore a key role for confidence in
interactions with GenAI, shaped by both prior beliefs about oneself and the
reliability of AI, and context-dependent exposure to advice.

### 6. [Can AI be Accountable?](http://arxiv.org/pdf/2510.26057v1)

Authors: Andrew L. Kun

The AI we use is powerful, and its power is increasing rapidly. If this
powerful AI is to serve the needs of consumers, voters, and decision makers,
then it is imperative that the AI is accountable. In general, an agent is
accountable to a forum if the forum can request information from the agent
about its actions, if the forum and the agent can discuss this information, and
if the forum can sanction the agent. Unfortunately, in too many cases today's
AI is not accountable -- we cannot question it, enter into a discussion with
it, let alone sanction it. In this chapter we relate the general definition of
accountability to AI, we illustrate what it means for AI to be accountable and
unaccountable, and we explore approaches that can improve our chances of living
in a world where all AI is accountable to those who are affected by it.

### 7. [Structurally Valid Log Generation using FSM-GFlowNets](http://arxiv.org/pdf/2510.26197v1)

Authors: Riya Samanta

Generating structurally valid and behaviorally diverse synthetic event logs
for interaction-aware models is a challenging yet crucial problem, particularly
in settings with limited or privacy constrained user data. Existing methods
such as heuristic simulations and LLM based generators often lack structural
coherence or controllability, producing synthetic data that fails to accurately
represent real world system interactions. This paper presents a framework that
integrates Finite State Machines or FSMs with Generative Flow Networks or
GFlowNets to generate structured, semantically valid, and diverse synthetic
event logs. Our FSM-constrained GFlowNet ensures syntactic validity and
behavioral variation through dynamic action masking and guided sampling. The
FSM, derived from expert traces, encodes domain-specific rules, while the
GFlowNet is trained using a flow matching objective with a hybrid reward
balancing FSM compliance and statistical fidelity. We instantiate the framework
in the context of UI interaction logs using the UIC HCI dataset, but the
approach generalizes to any symbolic sequence domain. Experimental results
based on distributional metrics show that our FSM GFlowNet produces realistic,
structurally consistent logs, achieving, for instance, under the real user logs
baseline, a KL divergence of 0.2769 and Chi squared distance of 0.3522,
significantly outperforming GPT-4o's 2.5294/13.8020 and Gemini's
3.7233/63.0355, alongside a leading bigram overlap of 0.1214 vs. GPT 4o's
0.0028 and Gemini's 0.0007. A downstream use case intent classification
demonstrates that classifiers trained solely on our synthetic logs produced
from FSM-GFlowNet achieve competitive accuracy compared to real data.

### 8. [Look at That Distractor: Dynamic Translation Gain under Low Perceptual Load in Virtual Reality](http://arxiv.org/pdf/2510.26265v1)

Authors: Ling-Long Zou, Qiang Tong, Er-Xia Luo, Sen-Zhe Xu, Song-Hai Zhang, Fang-Lue Zhang

Redirected walking utilizes gain adjustments within perceptual thresholds to
allow natural navigation in large scale virtual environments within confined
physical environments. Previous research has found that when users are
distracted by some scene elements, they are less sensitive to gain values.
However, the effects on detection thresholds have not been quantitatively
measured. In this paper, we present a novel method that dynamically adjusts
translation gain by leveraging visual distractors. We place distractors within
the user's field of view and apply a larger translation gain when their
attention is drawn to them. Because the magnitude of gain adjustment depends on
the user's level of engagement with the distractors, the redirection process
remains smooth and unobtrusive. To evaluate our method, we developed a task
oriented virtual environment for a user study. Results show that introducing
distractors in the virtual environment significantly raises users' translation
gain thresholds. Furthermore, assessments using the Simulator Sickness
Questionnaire and Igroup Presence Questionnaire indicate that the method
maintains user comfort and acceptance, supporting its effectiveness for RDW
systems.

### 9. [Human-AI Complementarity: A Goal for Amplified Oversight](http://arxiv.org/pdf/2510.26518v1)

Authors: Rishub Jain, Sophie Bridgers, Lili Janzer, Rory Greig, Tian Huey Teh, Vladimir Mikulik

Human feedback is critical for aligning AI systems to human values. As AI
capabilities improve and AI is used to tackle more challenging tasks, verifying
quality and safety becomes increasingly challenging. This paper explores how we
can leverage AI to improve the quality of human oversight. We focus on an
important safety problem that is already challenging for humans:
fact-verification of AI outputs. We find that combining AI ratings and human
ratings based on AI rater confidence is better than relying on either alone.
Giving humans an AI fact-verification assistant further improves their
accuracy, but the type of assistance matters. Displaying AI explanation,
confidence, and labels leads to over-reliance, but just showing search results
and evidence fosters more appropriate trust. These results have implications
for Amplified Oversight -- the challenge of combining humans and AI to
supervise AI systems even as they surpass human expert performance.

### 10. [Linking Heterogeneous Data with Coordinated Agent Flows for Social Media Analysis](http://arxiv.org/pdf/2510.26172v1)

Authors: Shifu Chen, Dazhen Deng, Zhihong Xu, Sijia Xu, Tai-Quan Peng, Yingcai Wu

Social media platforms generate massive volumes of heterogeneous data,
capturing user behaviors, textual content, temporal dynamics, and network
structures. Analyzing such data is crucial for understanding phenomena such as
opinion dynamics, community formation, and information diffusion. However,
discovering insights from this complex landscape is exploratory, conceptually
challenging, and requires expertise in social media mining and visualization.
Existing automated approaches, though increasingly leveraging large language
models (LLMs), remain largely confined to structured tabular data and cannot
adequately address the heterogeneity of social media analysis. We present SIA
(Social Insight Agents), an LLM agent system that links heterogeneous
multi-modal data -- including raw inputs (e.g., text, network, and behavioral
data), intermediate outputs, mined analytical results, and visualization
artifacts -- through coordinated agent flows. Guided by a bottom-up taxonomy
that connects insight types with suitable mining and visualization techniques,
SIA enables agents to plan and execute coherent analysis strategies. To ensure
multi-modal integration, it incorporates a data coordinator that unifies
tabular, textual, and network data into a consistent flow. Its interactive
interface provides a transparent workflow where users can trace, validate, and
refine the agent's reasoning, supporting both adaptability and trustworthiness.
Through expert-centered case studies and quantitative evaluation, we show that
SIA effectively discovers diverse and meaningful insights from social media
while supporting human-agent collaboration in complex analytical tasks.

### Information Retrieval

### 1. [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](http://arxiv.org/pdf/2510.26104v1)

Authors: Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, Aixin Sun

In recommendation systems, scaling up feature-interaction modules (e.g.,
Wukong, RankMixer) or user-behavior sequence modules (e.g., LONGER) has
achieved notable success. However, these efforts typically proceed on separate
tracks, which not only hinders bidirectional information exchange but also
prevents unified optimization and scaling. In this paper, we propose OneTrans,
a unified Transformer backbone that simultaneously performs user-behavior
sequence modeling and feature interaction. OneTrans employs a unified tokenizer
to convert both sequential and non-sequential attributes into a single token
sequence. The stacked OneTrans blocks share parameters across similar
sequential tokens while assigning token-specific parameters to non-sequential
tokens. Through causal attention and cross-request KV caching, OneTrans enables
precomputation and caching of intermediate representations, significantly
reducing computational costs during both training and inference. Experimental
results on industrial-scale datasets demonstrate that OneTrans scales
efficiently with increasing parameters, consistently outperforms strong
baselines, and yields a 5.68% lift in per-user GMV in online A/B tests.

### 2. [ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs](http://arxiv.org/pdf/2510.26178v1)

Authors: Yanran Tang, Ruihong Qiu, Xue Li, Zi Huang

Legal case retrieval (LCR) is a cornerstone of real-world legal decision
making, as it enables practitioners to identify precedents for a given query
case. Existing approaches mainly rely on traditional lexical models and
pretrained language models to encode the texts of legal cases. Yet there are
rich information in the relations among different legal entities as well as the
crucial reasoning process that uncovers how legal facts and legal issues can
lead to judicial decisions. Such relational reasoning process reflects the
distinctive characteristics of each case that can distinguish one from another,
mirroring the real-world judicial process. Naturally, incorporating such
information into the precise case embedding could further enhance the accuracy
of case retrieval. In this paper, a novel ReaKase-8B framework is proposed to
leverage extracted legal facts, legal issues, legal relation triplets and legal
reasoning for effective legal case retrieval. ReaKase-8B designs an in-context
legal case representation learning paradigm with a fine-tuned large language
model. Extensive experiments on two benchmark datasets from COLIEE 2022 and
COLIEE 2023 demonstrate that our knowledge and reasoning augmented embeddings
substantially improve retrieval performance over baseline models, highlighting
the potential of integrating legal reasoning into legal case retrieval systems.
The code has been released on https://github.com/yanran-tang/ReaKase-8B.

### 3. [DiSE: A diffusion probabilistic model for automatic structure elucidation of organic compounds](http://arxiv.org/pdf/2510.26231v1)

Authors: Haochen Chen, Qi Huang, Anan Wu, Wenhao Zhang, Jianliang Ye, Jianming Wu, Kai Tan, Xin Lu, Xin Xu

Automatic structure elucidation is essential for self-driving laboratories as
it enables the system to achieve truly autonomous. This capability closes the
experimental feedback loop, ensuring that machine learning models receive
reliable structure information for real-time decision-making and optimization.
Herein, we present DiSE, an end-to-end diffusion-based generative model that
integrates multiple spectroscopic modalities, including MS, 13C and 1H chemical
shifts, HSQC, and COSY, to achieve automated yet accurate structure elucidation
of organic compounds. By learning inherent correlations among spectra through
data-driven approaches, DiSE achieves superior accuracy, strong generalization
across chemically diverse datasets, and robustness to experimental data despite
being trained on calculated spectra. DiSE thus represents a significant advance
toward fully automated structure elucidation, with broad potential in natural
product research, drug discovery, and self-driving laboratories.

### 4. [Barlow Twins for Sequential Recommendation](http://arxiv.org/pdf/2510.26407v1)

Authors: Ivan Razvorotnev, Marina Munkhoeva, Evgeny Frolov

Sequential recommendation models must navigate sparse interaction data
popularity bias and conflicting objectives like accuracy versus diversity While
recent contrastive selfsupervised learning SSL methods offer improved accuracy
they come with tradeoffs large batch requirements reliance on handcrafted
augmentations and negative sampling that can reinforce popularity bias In this
paper we introduce BT-SR a novel noncontrastive SSL framework that integrates
the Barlow Twins redundancyreduction principle into a Transformerbased nextitem
recommender BTSR learns embeddings that align users with similar shortterm
behaviors while preserving longterm distinctionswithout requiring negative
sampling or artificial perturbations This structuresensitive alignment allows
BT-SR to more effectively recognize emerging user intent and mitigate the
influence of noisy historical context Our experiments on five public benchmarks
demonstrate that BTSR consistently improves nextitem prediction accuracy and
significantly enhances longtail item coverage and recommendation calibration
Crucially we show that a single hyperparameter can control the
accuracydiversity tradeoff enabling practitioners to adapt recommendations to
specific application needs

### 5. [WeaveRec: An LLM-Based Cross-Domain Sequential Recommendation Framework with Model Merging](http://arxiv.org/pdf/2510.26546v1)

Authors: Min Hou, Xin Liu, Le Wu, Chenyi He, Hao Liu, Zhi Li, Xin Li, Si Wei

Cross-Domain Sequential Recommendation (CDSR) seeks to improve user
preference modeling by transferring knowledge from multiple domains. Despite
the progress made in CDSR, most existing methods rely on overlapping users or
items to establish cross-domain correlations-a requirement that rarely holds in
real-world settings. The advent of large language models (LLM) and
model-merging techniques appears to overcome this limitation by unifying
multi-domain data without explicit overlaps. Yet, our empirical study shows
that naively training an LLM on combined domains-or simply merging several
domain-specific LLMs-often degrades performance relative to a model trained
solely on the target domain. To address these challenges, we first
experimentally investigate the cause of suboptimal performance in LLM-based
cross-domain recommendation and model merging. Building on these insights, we
introduce WeaveRec, which cross-trains multiple LoRA modules with source and
target domain data in a weaving fashion, and fuses them via model merging.
WeaveRec can be extended to multi-source domain scenarios and notably does not
introduce additional inference-time cost in terms of latency or memory.
Furthermore, we provide a theoretical guarantee that WeaveRec can reduce the
upper bound of the expected error in the target domain. Extensive experiments
on single-source, multi-source, and cross-platform cross-domain recommendation
scenarios validate that WeaveRec effectively mitigates performance degradation
and consistently outperforms baseline approaches in real-world recommendation
tasks.

### 6. [ProfOlaf: Semi-Automated Tool for Systematic Literature Reviews](http://arxiv.org/pdf/2510.26750v1)

Authors: Martim Afonso, Nuno Saavedra, Bruno Lourenço, Alexandra Mendes, João Ferreira

Systematic reviews and mapping studies are critical for synthesizing
research, identifying gaps, and guiding future work, but they are often
labor-intensive and time-consuming. Existing tools provide partial support for
specific steps, leaving much of the process manual and error-prone. We present
ProfOlaf, a semi-automated tool designed to streamline systematic reviews while
maintaining methodological rigor. ProfOlaf supports iterative snowballing for
article collection with human-in-the-loop filtering and uses large language
models to assist in analyzing articles, extracting key topics, and answering
queries about the content of papers. By combining automation with guided manual
effort, ProfOlaf enhances the efficiency, quality, and reproducibility of
systematic reviews across research fields. A video describing and demonstrating
ProfOlaf is available at: https://youtu.be/4noUXfcmxsE

### 7. [ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests](http://arxiv.org/pdf/2510.26095v1)

Authors: Jingyuan He, Jiongnan Liu, Vishan Vishesh Oberoi, Bolin Wu, Mahima Jagadeesh Patel, Kangrui Mao, Chuning Shi, I-Ta Lee, Arnold Overwijk, Chenyan Xiong

Recommender systems are among the most impactful AI applications, interacting
with billions of users every day, guiding them to relevant products, services,
or information tailored to their preferences. However, the research and
development of recommender systems are hindered by existing datasets that fail
to capture realistic user behaviors and inconsistent evaluation settings that
lead to ambiguous conclusions. This paper introduces the Open Recommendation
Benchmark for Reproducible Research with HIdden Tests (ORBIT), a unified
benchmark for consistent and realistic evaluation of recommendation models.
ORBIT offers a standardized evaluation framework of public datasets with
reproducible splits and transparent settings for its public leaderboard.
Additionally, ORBIT introduces a new webpage recommendation task, ClueWeb-Reco,
featuring web browsing sequences from 87 million public, high-quality webpages.
ClueWeb-Reco is a synthetic dataset derived from real, user-consented, and
privacy-guaranteed browsing data. It aligns with modern recommendation
scenarios and is reserved as the hidden test part of our leaderboard to
challenge recommendation models' generalization ability. ORBIT measures 12
representative recommendation models on its public benchmark and introduces a
prompted LLM baseline on the ClueWeb-Reco hidden test. Our benchmark results
reflect general improvements of recommender systems on the public datasets,
with variable individual performances. The results on the hidden test reveal
the limitations of existing approaches in large-scale webpage recommendation
and highlight the potential for improvements with LLM integrations. ORBIT
benchmark, leaderboard, and codebase are available at
https://www.open-reco-bench.ai.

### 8. [GraphCompliance: Aligning Policy and Context Graphs for LLM-Based Regulatory Compliance](http://arxiv.org/pdf/2510.26309v1)

Authors: Jiseong Chung, Ronny Ko, Wonchul Yoo, Makoto Onizuka, Sungmok Kim, Tae-Wan Kim, Won-Yong Shin

Compliance at web scale poses practical challenges: each request may require
a regulatory assessment. Regulatory texts (e.g., the General Data Protection
Regulation, GDPR) are cross-referential and normative, while runtime contexts
are expressed in unstructured natural language. This setting motivates us to
align semantic information in unstructured text with the structured, normative
elements of regulations. To this end, we introduce GraphCompliance, a framework
that represents regulatory texts as a Policy Graph and runtime contexts as a
Context Graph, and aligns them. In this formulation, the policy graph encodes
normative structure and cross-references, whereas the context graph formalizes
events as subject-action-object (SAO) and entity-relation triples. This
alignment anchors the reasoning of a judge large language model (LLM) in
structured information and helps reduce the burden of regulatory interpretation
and event parsing, enabling a focus on the core reasoning step. In experiments
on 300 GDPR-derived real-world scenarios spanning five evaluation tasks,
GraphCompliance yields 4.1-7.2 percentage points (pp) higher micro-F1 than
LLM-only and RAG baselines, with fewer under- and over-predictions, resulting
in higher recall and lower false positive rates. Ablation studies indicate
contributions from each graph component, suggesting that structured
representations and a judge LLM are complementary for normative reasoning.

### 9. [Vectorized Context-Aware Embeddings for GAT-Based Collaborative Filtering](http://arxiv.org/pdf/2510.26461v1)

Authors: Danial Ebrat, Sepideh Ahmadian, Luis Rueda

Recommender systems often struggle with data sparsity and cold-start
scenarios, limiting their ability to provide accurate suggestions for new or
infrequent users. This paper presents a Graph Attention Network (GAT) based
Collaborative Filtering (CF) framework enhanced with Large Language Model (LLM)
driven context aware embeddings. Specifically, we generate concise textual user
profiles and unify item metadata (titles, genres, overviews) into rich textual
embeddings, injecting these as initial node features in a bipartite user item
graph. To further optimize ranking performance, we introduce a hybrid loss
function that combines Bayesian Personalized Ranking (BPR) with a cosine
similarity term and robust negative sampling, ensuring explicit negative
feedback is distinguished from unobserved data. Experiments on the MovieLens
100k and 1M datasets show consistent improvements over state-of-the-art
baselines in Precision, NDCG, and MAP while demonstrating robustness for users
with limited interaction history. Ablation studies confirm the critical role of
LLM-augmented embeddings and the cosine similarity term in capturing nuanced
semantic relationships. Our approach effectively mitigates sparsity and
cold-start limitations by integrating LLM-derived contextual understanding into
graph-based architectures. Future directions include balancing recommendation
accuracy with coverage and diversity, and introducing fairness-aware
constraints and interpretability features to enhance system performance
further.

### 10. [LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs for Human Smuggling Networks](http://arxiv.org/pdf/2510.26486v1)

Authors: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera

Human smuggling networks are complex and constantly evolving, making them
difficult to analyze comprehensively. Legal case documents offer rich factual
and procedural insights into these networks but are often long, unstructured,
and filled with ambiguous or shifting references, posing significant challenges
for automated knowledge graph (KG) construction. Existing methods either
overlook coreference resolution or fail to scale beyond short text spans,
leading to fragmented graphs and inconsistent entity linking. We propose
LINK-KG, a modular framework that integrates a three-stage, LLM-guided
coreference resolution pipeline with downstream KG extraction. At the core of
our approach is a type-specific Prompt Cache, which consistently tracks and
resolves references across document chunks, enabling clean and disambiguated
narratives for structured knowledge graph construction from both short and long
legal texts. LINK-KG reduces average node duplication by 45.21% and noisy nodes
by 32.22% compared to baseline methods, resulting in cleaner and more coherent
graph structures. These improvements establish LINK-KG as a strong foundation
for analyzing complex criminal networks.

### Machine Learning

### 1. [Towards Scaling Laws for Symbolic Regression](http://arxiv.org/pdf/2510.26064v1)

Authors: David Otte, Jörg K. H. Franke, Frank Hutter

Symbolic regression (SR) aims to discover the underlying mathematical
expressions that explain observed data. This holds promise for both gaining
scientific insight and for producing inherently interpretable and generalizable
models for tabular data. In this work we focus on the basics of SR. Deep
learning-based SR has recently become competitive with genetic programming
approaches, but the role of scale has remained largely unexplored. Inspired by
scaling laws in language modeling, we present the first systematic
investigation of scaling in SR, using a scalable end-to-end transformer
pipeline and carefully generated training data. Across five different model
sizes and spanning three orders of magnitude in compute, we find that both
validation loss and solved rate follow clear power-law trends with compute. We
further identify compute-optimal hyperparameter scaling: optimal batch size and
learning rate grow with model size, and a token-to-parameter ratio of
$\approx$15 is optimal in our regime, with a slight upward trend as compute
increases. These results demonstrate that SR performance is largely predictable
from compute and offer important insights for training the next generation of
SR models.

### 2. [New Money: A Systematic Review of Synthetic Data Generation for Finance](http://arxiv.org/pdf/2510.26076v1)

Authors: James Meldrum, Basem Suleiman, Fethi Rabhi, Muhammad Johan Alibasa

Synthetic data generation has emerged as a promising approach to address the
challenges of using sensitive financial data in machine learning applications.
By leveraging generative models, such as Generative Adversarial Networks (GANs)
and Variational Autoencoders (VAEs), it is possible to create artificial
datasets that preserve the statistical properties of real financial records
while mitigating privacy risks and regulatory constraints. Despite the rapid
growth of this field, a comprehensive synthesis of the current research
landscape has been lacking. This systematic review consolidates and analyses 72
studies published since 2018 that focus on synthetic financial data generation.
We categorise the types of financial information synthesised, the generative
methods employed, and the evaluation strategies used to assess data utility and
privacy. The findings indicate that GAN-based approaches dominate the
literature, particularly for generating time-series market data and tabular
credit data. While several innovative techniques demonstrate potential for
improved realism and privacy preservation, there remains a notable lack of
rigorous evaluation of privacy safeguards across studies. By providing an
integrated overview of generative techniques, applications, and evaluation
methods, this review highlights critical research gaps and offers guidance for
future work aimed at developing robust, privacy-preserving synthetic data
solutions for the financial domain.

### 3. [LLMBisect: Breaking Barriers in Bug Bisection with A Comparative Analysis Pipeline](http://arxiv.org/pdf/2510.26086v1)

Authors: Zheng Zhang, Haonan Li, Xingyu Li, Hang Zhang, Zhiyun Qian

Bug bisection has been an important security task that aims to understand the
range of software versions impacted by a bug, i.e., identifying the commit that
introduced the bug. However, traditional patch-based bisection methods are
faced with several significant barriers: For example, they assume that the
bug-inducing commit (BIC) and the patch commit modify the same functions, which
is not always true. They often rely solely on code changes, while the commit
message frequently contains a wealth of vulnerability-related information. They
are also based on simple heuristics (e.g., assuming the BIC initializes lines
deleted in the patch) and lack any logical analysis of the vulnerability.
  In this paper, we make the observation that Large Language Models (LLMs) are
well-positioned to break the barriers of existing solutions, e.g., comprehend
both textual data and code in patches and commits. Unlike previous BIC
identification approaches, which yield poor results, we propose a comprehensive
multi-stage pipeline that leverages LLMs to: (1) fully utilize patch
information, (2) compare multiple candidate commits in context, and (3)
progressively narrow down the candidates through a series of down-selection
steps. In our evaluation, we demonstrate that our approach achieves
significantly better accuracy than the state-of-the-art solution by more than
38\%. Our results further confirm that the comprehensive multi-stage pipeline
is essential, as it improves accuracy by 60\% over a baseline LLM-based
bisection method.

### 4. [Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](http://arxiv.org/pdf/2510.26109v1)

Authors: Chenming Tang, Hsiu-Yuan Huang, Weijie Liu, Saiyong Yang, Yunfang Wu

Reinforcement learning with verifiable rewards (RLVR) has significantly
boosted the reasoning capability of large language models (LLMs) recently.
However, existing RLVR approaches merely train LLMs based on their own
generated responses and are constrained by the initial capability of LLMs, thus
prone to exploration stagnation, in which LLMs fail to solve more training
problems and cannot further learn from the training data. Some work tries to
address this by leveraging off-policy solutions to training problems but
requires external guidance from experts which suffers from limited
availability. In this work, we propose LTE (Learning to reason from Trial and
Error), an approach hinting LLMs with their previously self-generated incorrect
answers and problem of overlong responses, which does not require any external
expert guidance. Experiments validate the effectiveness of LTE, which
outperforms the normal group relative policy optimization (GRPO) by 6.38 in
Pass@1 and 9.00 in Pass@k on average across six mathematics benchmarks for
Qwen3-4B-Base. Further analysis confirms that LTE successfully mitigates the
problem of exploration stagnation and enhances both exploitation and
exploration during training.

### 5. [maxVSTAR: Maximally Adaptive Vision-Guided CSI Sensing with Closed-Loop Edge Model Adaptation for Robust Human Activity Recognition](http://arxiv.org/pdf/2510.26146v1)

Authors: Kexing Liu

WiFi Channel State Information (CSI)-based human activity recognition (HAR)
provides a privacy-preserving, device-free sensing solution for smart
environments. However, its deployment on edge devices is severely constrained
by domain shift, where recognition performance deteriorates under varying
environmental and hardware conditions. This study presents maxVSTAR (maximally
adaptive Vision-guided Sensing Technology for Activity Recognition), a
closed-loop, vision-guided model adaptation framework that autonomously
mitigates domain shift for edge-deployed CSI sensing systems. The proposed
system integrates a cross-modal teacher-student architecture, where a
high-accuracy YOLO-based vision model serves as a dynamic supervisory signal,
delivering real-time activity labels for the CSI data stream. These labels
enable autonomous, online fine-tuning of a lightweight CSI-based HAR model,
termed Sensing Technology for Activity Recognition (STAR), directly at the
edge. This closed-loop retraining mechanism allows STAR to continuously adapt
to environmental changes without manual intervention. Extensive experiments
demonstrate the effectiveness of maxVSTAR. When deployed on uncalibrated
hardware, the baseline STAR model's recognition accuracy declined from 93.52%
to 49.14%. Following a single vision-guided adaptation cycle, maxVSTAR restored
the accuracy to 81.51%. These results confirm the system's capacity for
dynamic, self-supervised model adaptation in privacy-conscious IoT
environments, establishing a scalable and practical paradigm for long-term
autonomous HAR using CSI sensing at the network edge.

### 6. [STAR: A Privacy-Preserving, Energy-Efficient Edge AI Framework for Human Activity Recognition via Wi-Fi CSI in Mobile and Pervasive Computing Environments](http://arxiv.org/pdf/2510.26148v1)

Authors: Kexing Liu

Human Activity Recognition (HAR) via Wi-Fi Channel State Information (CSI)
presents a privacy-preserving, contactless sensing approach suitable for smart
homes, healthcare monitoring, and mobile IoT systems. However, existing methods
often encounter computational inefficiency, high latency, and limited
feasibility within resource-constrained, embedded mobile edge environments.
This paper proposes STAR (Sensing Technology for Activity Recognition), an
edge-AI-optimized framework that integrates a lightweight neural architecture,
adaptive signal processing, and hardware-aware co-optimization to enable
real-time, energy-efficient HAR on low-power embedded devices. STAR
incorporates a streamlined Gated Recurrent Unit (GRU)-based recurrent neural
network, reducing model parameters by 33% compared to conventional LSTM models
while maintaining effective temporal modeling capability. A multi-stage
pre-processing pipeline combining median filtering, 8th-order Butterworth
low-pass filtering, and Empirical Mode Decomposition (EMD) is employed to
denoise CSI amplitude data and extract spatial-temporal features. For on-device
deployment, STAR is implemented on a Rockchip RV1126 processor equipped with an
embedded Neural Processing Unit (NPU), interfaced with an ESP32-S3-based CSI
acquisition module. Experimental results demonstrate a mean recognition
accuracy of 93.52% across seven activity classes and 99.11% for human presence
detection, utilizing a compact 97.6k-parameter model. INT8 quantized inference
achieves a processing speed of 33 MHz with just 8% CPU utilization, delivering
sixfold speed improvements over CPU-based execution. With sub-second response
latency and low power consumption, the system ensures real-time,
privacy-preserving HAR, offering a practical, scalable solution for mobile and
pervasive computing environments.

### 7. [Likely Interpolants of Generative Models](http://arxiv.org/pdf/2510.26266v1)

Authors: Frederik Möbius Rygaard, Shen Zhu, Yinzhu Jin, Søren Hauberg, Tom Fletcher

Interpolation in generative models allows for controlled generation, model
inspection, and more. Unfortunately, most generative models lack a principal
notion of interpolants without restrictive assumptions on either the model or
data dimension. In this paper, we develop a general interpolation scheme that
targets likely transition paths compatible with different metrics and
probability distributions. We consider interpolants analogous to a geodesic
constrained to a suitable data distribution and derive a novel algorithm for
computing these curves, which requires no additional training. Theoretically,
we show that our method locally can be considered as a geodesic under a
suitable Riemannian metric. We quantitatively show that our interpolation
scheme traverses higher density regions than baselines across a range of models
and datasets.

### 8. [Empirical Bayesian Multi-Bandit Learning](http://arxiv.org/pdf/2510.26284v1)

Authors: Xia Jiang, Rong J. B. Zhu

Multi-task learning in contextual bandits has attracted significant research
interest due to its potential to enhance decision-making across multiple
related tasks by leveraging shared structures and task-specific heterogeneity.
In this article, we propose a novel hierarchical Bayesian framework for
learning in various bandit instances. This framework captures both the
heterogeneity and the correlations among different bandit instances through a
hierarchical Bayesian model, enabling effective information sharing while
accommodating instance-specific variations. Unlike previous methods that
overlook the learning of the covariance structure across bandits, we introduce
an empirical Bayesian approach to estimate the covariance matrix of the prior
distribution.This enhances both the practicality and flexibility of learning
across multi-bandits. Building on this approach, we develop two efficient
algorithms: ebmTS (Empirical Bayesian Multi-Bandit Thompson Sampling) and
ebmUCB (Empirical Bayesian Multi-Bandit Upper Confidence Bound), both of which
incorporate the estimated prior into the decision-making process. We provide
the frequentist regret upper bounds for the proposed algorithms, thereby
filling a research gap in the field of multi-bandit problems. Extensive
experiments on both synthetic and real-world datasets demonstrate the superior
performance of our algorithms, particularly in complex environments. Our
methods achieve lower cumulative regret compared to existing techniques,
highlighting their effectiveness in balancing exploration and exploitation
across multi-bandits.

### 9. [Offline Clustering of Preference Learning with Active-data Augmentation](http://arxiv.org/pdf/2510.26301v1)

Authors: Jingyuan Liu, Fatemeh Ghaffari, Xuchuang Wang, Mohammad Hajiesmaili, Carlee Joe-Wong

Preference learning from pairwise feedback is a widely adopted framework in
applications such as reinforcement learning with human feedback and
recommendations. In many practical settings, however, user interactions are
limited or costly, making offline preference learning necessary. Moreover,
real-world preference learning often involves users with different preferences.
For example, annotators from different backgrounds may rank the same responses
differently. This setting presents two central challenges: (1) identifying
similarity across users to effectively aggregate data, especially under
scenarios where offline data is imbalanced across dimensions, and (2) handling
the imbalanced offline data where some preference dimensions are
underrepresented. To address these challenges, we study the Offline Clustering
of Preference Learning problem, where the learner has access to fixed datasets
from multiple users with potentially different preferences and aims to maximize
utility for a test user. To tackle the first challenge, we first propose
Off-C$^2$PL for the pure offline setting, where the learner relies solely on
offline data. Our theoretical analysis provides a suboptimality bound that
explicitly captures the tradeoff between sample noise and bias. To address the
second challenge of inbalanced data, we extend our framework to the setting
with active-data augmentation where the learner is allowed to select a limited
number of additional active-data for the test user based on the cluster
structure learned by Off-C$^2$PL. In this setting, our second algorithm,
A$^2$-Off-C$^2$PL, actively selects samples that target the least-informative
dimensions of the test user's preference. We prove that these actively
collected samples contribute more effectively than offline ones. Finally, we
validate our theoretical results through simulations on synthetic and
real-world datasets.

### 10. [Model Inversion with Layer-Specific Modeling and Alignment for Data-Free Continual Learning](http://arxiv.org/pdf/2510.26311v1)

Authors: Ruilin Tong, Haodong Lu, Yuhang Liu, Dong Gong

Continual learning (CL) aims to incrementally train a model on a sequence of
tasks while retaining performance on prior ones. However, storing and replaying
data is often infeasible due to privacy or security constraints and impractical
for arbitrary pre-trained models. Data-free CL seeks to update models without
access to previous data. Beyond regularization, we employ model inversion to
synthesize data from the trained model, enabling replay without storing
samples. Yet, model inversion in predictive models faces two challenges: (1)
generating inputs solely from compressed output labels causes drift between
synthetic and real data, and replaying such data can erode prior knowledge; (2)
inversion is computationally expensive since each step backpropagates through
the full model. These issues are amplified in large pre-trained models such as
CLIP. To improve efficiency, we propose Per-layer Model Inversion (PMI),
inspired by faster convergence in single-layer optimization. PMI provides
strong initialization for full-model inversion, substantially reducing
iterations. To mitigate feature shift, we model class-wise features via
Gaussian distributions and contrastive model, ensuring alignment between
synthetic and real features. Combining PMI and feature modeling, our approach
enables continual learning of new classes by generating pseudo-images from
semantic-aware projected features, achieving strong effectiveness and
compatibility across multiple CL settings.

### Neural and Evolutionary Computing

### 1. [Advancing Forest Fires Classification using Neurochaos Learning](http://arxiv.org/pdf/2510.26383v1)

Authors: Kunal Kumar Pant, Remya Ajai A S, Nithin Nagaraj

Forest fires are among the most dangerous and unpredictable natural disasters
worldwide. Forest fire can be instigated by natural causes or by humans. They
are devastating overall, and thus, many research efforts have been carried out
to predict whether a fire can occur in an area given certain environmental
variables. Many research works employ Machine Learning (ML) and Deep Learning
(DL) models for classification; however, their accuracy is merely adequate and
falls short of expectations. This limit arises because these models are unable
to depict the underlying nonlinearity in nature and extensively rely on
substantial training data, which is hard to obtain. We propose using Neurochaos
Learning (NL), a chaos-based, brain-inspired learning algorithm for forest fire
classification. Like our brains, NL needs less data to learn nonlinear patterns
in the training data. It employs one-dimensional chaotic maps, namely the
Generalized L\"uroth Series (GLS), as neurons. NL yields comparable performance
with ML and DL models, sometimes even surpassing them, particularly in
low-sample training regimes, and unlike deep neural networks, NL is
interpretable as it preserves causal structures in the data. Random
Heterogenous Neurochaos Learning (RHNL), a type of NL where different chaotic
neurons are randomnly located to mimic the randomness and heterogeneity of
human brain gives the best F1 score of 1.0 for the Algerian Forest Fires
Dataset. Compared to other traditional ML classifiers considered, RHNL also
gives high precision score of 0.90 for Canadian Forest Fires Dataset and 0.68
for Portugal Forest Fires Dataset. The results obtained from this work indicate
that Neurochaos Learning (NL) architectures achieve better performance than
conventional machine learning classifiers, highlighting their promise for
developing more efficient and reliable forest fire detection systems.

### 2. [Unravelling the Mechanisms of Manipulating Numbers in Language Models](http://arxiv.org/pdf/2510.26285v1)

Authors: Michal Štefánik, Timothee Mickus, Marek Kadlčík, Bertram Højer, Michal Spiegel, Raúl Vázquez, Aman Sinha, Josef Kuchař, Philipp Mondorf

Recent work has shown that different large language models (LLMs) converge to
similar and accurate input embedding representations for numbers. These
findings conflict with the documented propensity of LLMs to produce erroneous
outputs when dealing with numeric information. In this work, we aim to explain
this conflict by exploring how language models manipulate numbers and quantify
the lower bounds of accuracy of these mechanisms. We find that despite
surfacing errors, different language models learn interchangeable
representations of numbers that are systematic, highly accurate and universal
across their hidden states and the types of input contexts. This allows us to
create universal probes for each LLM and to trace information -- including the
causes of output errors -- to specific layers. Our results lay a fundamental
understanding of how pre-trained LLMs manipulate numbers and outline the
potential of more accurate probing techniques in addressed refinements of LLMs'
architectures.

### 3. [Wireless Sensor Networks as Parallel and Distributed Hardware Platform for Artificial Neural Networks](http://arxiv.org/pdf/2510.26492v1)

Authors: Gursel Serpen

We are proposing fully parallel and maximally distributed hardware
realization of a generic neuro-computing system. More specifically, the
proposal relates to the wireless sensor networks technology to serve as a
massively parallel and fully distributed hardware platform to implement and
realize artificial neural network (ANN) algorithms. A parallel and distributed
(PDP) hardware realization of ANNs makes it possible to have real time
computation of large-scale (and complex) problems in a highly robust framework.
We will demonstrate how a network of hundreds of thousands of processing nodes
(or motes of a wireless sensor network), which have on-board processing and
wireless communication features, can be used to implement fully parallel and
massively distributed computation of artificial neural network algorithms for
solution of truly large-scale problems in real time. The realization of
artificial neural network algorithms in a massively parallel and fully
distributed hardware has been the goal of neural network computing researchers.
This is because a parallel and distributed computation of artificial neural
network algorithms could not have been achieved against all the advancements in
silicon- or optics-based computing. Accordingly, artificial neural networks
could not be applied to very large-scale problems for real time computation of
solutions. This hindered the development of neural algorithms for affordable
and practical solutions of challenging problems since often special-purpose
computing approaches in hardware, software or hybrid (non-neural) had to be
developed for and fine-tuned to specific problems that are very large-scale and
highly complex. Successful implementation is likely to revolutionize computing
as we know it by making it possible to solve very large scale scientific,
engineering or technical problems in real time.

### 4. [Reducing base drag on road vehicles using pulsed jets optimized by hybrid genetic algorithms](http://arxiv.org/pdf/2510.26718v1)

Authors: Isaac Robledo, Juan Alfaro, Víctor Duro, Alberto Solera-Rico, Rodrigo Castellanos, Carlos Sanmiguel Vila

Aerodynamic drag on flat-backed vehicles like vans and trucks is dominated by
a low-pressure wake, whose control is critical for reducing fuel consumption.
This paper presents an experimental study at $Re_W\approx 78,300$ on active
flow control using four pulsed jets at the rear edges of a bluff body model. A
hybrid genetic algorithm, combining a global search with a local gradient-based
optimizer, was used to determine the optimal jet actuation parameters in an
experiment-in-the-loop setup. The cost function was designed to achieve a net
energy saving by simultaneously minimizing aerodynamic drag and penalizing the
actuation's energy consumption. The optimization campaign successfully
identified a control strategy that yields a drag reduction of approximately
10%. The optimal control law features a strong, low-frequency actuation from
the bottom jet, which targets the main vortex shedding, while the top and
lateral jets address higher-frequency, less energetic phenomena. Particle Image
Velocimetry analysis reveals a significant upward shift and stabilization of
the wake, leading to substantial pressure recovery on the model's lower base.
Ultimately, this work demonstrates that a model-free optimization approach can
successfully identify non-intuitive, multi-faceted actuation strategies that
yield significant and energetically efficient drag reduction.

### Networking and Internet Architecture

### 1. [Performance Analysis of Dynamic Equilibria in Joint Path Selection and Congestion Control](http://arxiv.org/pdf/2510.26060v1)

Authors: Sina Keshvadi

Path-aware networking, a cornerstone of next-generation architectures like
SCION and Multipath QUIC, empowers end-hosts with fine-grained control over
traffic forwarding. This capability, however, introduces a critical stability
risk: uncoordinated, greedy path selection by a multitude of agents can induce
persistent, high-amplitude network oscillations. While this phenomenon is
well-known, its quantitative performance impact across key metrics has remained
poorly understood. In this paper, we address this gap by developing the first
axiomatic framework for analyzing the joint dynamics of path selection and
congestion control. Our model enables the formal characterization of the
system's dynamic equilibria-the stable, periodic patterns of oscillation-and
provides a suite of axioms to rate their performance in terms of efficiency,
loss avoidance, convergence, fairness, and responsiveness. Our analysis reveals
a fundamental trade-off in protocol design between predictable performance
(efficiency, convergence) and user-centric goals (fairness, responsiveness). We
prove, however, that no such trade-off exists among efficiency, convergence,
and loss avoidance, which can be simultaneously optimized through careful
parameter tuning. Furthermore, we find that agent migration can,
counter-intuitively, enhance stability by de-synchronizing traffic, a
theoretical result validated by our simulations. These findings provide a
principled design map for engineering robust, high-performance protocols for
the future path-aware Internet.

### 2. [Symmetry-Driven Asynchronous Forwarding for Reliable Distributed Coordination in Toroidal Networks](http://arxiv.org/pdf/2510.26071v1)

Authors: Shenshen Luan, Yumo Tian, Xinyu Zhang, Qingwen Zhang, Tianheng Wang, Yan Yang, Shuguo Xie

The proliferation of large-scale distributed systems, such as satellite
constellations and high-performance computing clusters, demands robust
communication primitives that maintain coordination under unreliable links. The
torus topology, with its inherent rotational and reflection symmetries, is a
prevalent architecture in these domains. However, conventional routing schemes
suffer from substantial packet loss during control-plane synchronization after
link failures. This paper introduces a symmetry-driven asynchronous forwarding
mechanism that leverages the torus's geometric properties to achieve reliable
packet delivery without control-plane coordination. We model packet flow using
a topological potential gradient and demonstrate that symmetry-breaking
failures naturally induce a reverse flow, which we harness for fault
circumvention. We propose two local forwarding strategies, Reverse Flow with
Counter-facing Priority (RF-CF) and Lateral-facing Priority (RF-LF), that
guarantee reachability to the destination via forward-flow phase transition
points, without protocol modifications or additional in-packet overhead.
Through percolation analysis and packet-level simulations on a 16 x 16 torus,
we show that our mechanism reduces packet loss by up to 17.5% under a 1% link
failure rate, with the RF-LF strategy contributing to 28% of successfully
delivered packets. This work establishes a foundational link between
topological symmetry and communication resilience, providing a lightweight,
protocol-agnostic substrate for enhancing distributed systems.

### 3. [From req/res to pub/sub: Exploring Media over QUIC Transport for DNS](http://arxiv.org/pdf/2510.26234v1)

Authors: Mathis Engelbart, Mike Kosek, Lars Eggert, Jörg Ott

The DNS is a key component of the Internet. Originally designed to facilitate
the resolution of host names to IP addresses, its scope has continuously
expanded over the years, today covering use cases such as load balancing or
service discovery. While DNS was initially conceived as a rather static
directory service in which resource records (RR) only change rarely, we have
seen a number of use cases over the years where a DNS flavor that isn't purely
based upon requesting and caching RRs, but rather on an active distribution of
updates for all resolvers that showed interest in the respective records in the
past, would be preferable. In this paper, we thus explore a publish-subscribe
variant of DNS based on the Media-over-QUIC architecture, where we devise a
strawman system and protocol proposal to enable pushing RR updates. We provide
a prototype implementation, finding that DNS can benefit from a
publish-subscribe variant: next to limiting update traffic, it can considerably
reduce the time it takes for a resolver to receive the latest version of a
record, thereby supporting use cases such as load balancing in content
distribution networks. The publish-subscribe architecture also brings new
challenges to the DNS, including a higher overhead for endpoints due to
additional state management, and increased query latencies on first lookup, due
to session establishment latencies.

### 4. [Joint Computing Resource Allocation and Task Offloading in Vehicular Fog Computing Systems Under Asymmetric Information](http://arxiv.org/pdf/2510.26256v1)

Authors: Geng Sun, Siyi Chen, Zemin Sun, Long He, Jiacheng Wang, Dusit Niyato, Zhu Han, Dong In Kim

Vehicular fog computing (VFC) has emerged as a promising paradigm, which
leverages the idle computational resources of nearby fog vehicles (FVs) to
complement the computing capabilities of conventional vehicular edge computing.
However, utilizing VFC to meet the delay-sensitive and computation-intensive
requirements of the FVs poses several challenges. First, the limited resources
of road side units (RSUs) struggle to accommodate the growing and diverse
demands of vehicles. This limitation is further exacerbated by the information
asymmetry between the controller and FVs due to the reluctance of FVs to
disclose private information and to share resources voluntarily. This
information asymmetry hinders the efficient resource allocation and
coordination. Second, the heterogeneity in task requirements and the varying
capabilities of RSUs and FVs complicate efficient task offloading, thereby
resulting in inefficient resource utilization and potential performance
degradation. To address these challenges, we first present a hierarchical VFC
architecture that incorporates the computing capabilities of both RSUs and FVs.
Then, we formulate a delay minimization optimization problem (DMOP), which is
an NP-hard mixed integer nonlinear programming problem. To solve the DMOP, we
propose a joint computing resource allocation and task offloading approach
(JCRATOA). Specifically, we propose a convex optimization-based method for RSU
resource allocation and a contract theory-based incentive mechanism for FV
resource allocation. Moreover, we present a two-sided matching method for task
offloading by employing the matching game. Simulation results demonstrate that
the proposed JCRATOA is able to achieve superior performances in task
completion delay, task completion ratio, system throughput, and resource
utilization fairness, while effectively meeting the satisfying constraints.

### 5. [Wireless Memory Approximation for Energy-efficient Task-specific IoT Data Retrieval](http://arxiv.org/pdf/2510.26473v1)

Authors: Junya Shiraishi, Shashi Raj Pandey, Israel Leyva-Mayorga, Petar Popovski

The use of Dynamic Random Access Memory (DRAM) for storing Machine Learning
(ML) models plays a critical role in accelerating ML inference tasks in the
next generation of communication systems. However, periodic refreshment of DRAM
results in wasteful energy consumption during standby periods, which is
significant for resource-constrained Internet of Things (IoT) devices. To solve
this problem, this work advocates two novel approaches: 1) wireless memory
activation and 2) wireless memory approximation. These enable the wireless
devices to efficiently manage the available memory by considering the timing
aspects and relevance of ML model usage; hence, reducing the overall energy
consumption. Numerical results show that our proposed scheme can realize
smaller energy consumption than the always-on approach while satisfying the
retrieval accuracy constraint.

### 6. [FGGM: Formal Grey-box Gradient Method for Attacking DRL-based MU-MIMO Scheduler](http://arxiv.org/pdf/2510.26075v1)

Authors: Thanh Le, Hai Duong, Yusheng Ji, ThanhVu Nguyen, John C. S. Lui

In 5G mobile communication systems, MU-MIMO has been applied to enhance
spectral efficiency and support high data rates. To maximize spectral
efficiency while providing fairness among users, the base station (BS) needs to
selects a subset of users for data transmission. Given that this problem is
NP-hard, DRL-based methods have been proposed to infer the near-optimal
solutions in real-time, yet this approach has an intrinsic security problem.
This paper investigates how a group of adversarial users can exploit
unsanitized raw CSIs to launch a throughput degradation attack. Most existing
studies only focused on systems in which adversarial users can obtain the exact
values of victims' CSIs, but this is impractical in the case of uplink
transmission in LTE/5G mobile systems. We note that the DRL policy contains an
observation normalizer which has the mean and variance of the observation to
improve training convergence. Adversarial users can then estimate the upper and
lower bounds of the local observations including the CSIs of victims based
solely on that observation normalizer. We develop an attacking scheme FGGM by
leveraging polytope abstract domains, a technique used to bound the outputs of
a neural network given the input ranges. Our goal is to find one set of
intentionally manipulated CSIs which can achieve the attacking goals for the
whole range of local observations of victims. Experimental results demonstrate
that FGGM can determine a set of adversarial CSI vector controlled by
adversarial users, then reuse those CSIs throughout the simulation to reduce
the network throughput of a victim up to 70\% without knowing the exact value
of victims' local observations. This study serves as a case study and can be
applied to many other DRL-based problems, such as a knapsack-oriented resource
allocation problems.

### 7. [Quantum Gated Recurrent GAN with Gaussian Uncertainty for Network Anomaly Detection](http://arxiv.org/pdf/2510.26487v1)

Authors: Wajdi Hammami, Soumaya Cherkaoui, Jean-Frederic Laprade, Ola Ahmad, Shengrui Wang

Anomaly detection in time-series data is a critical challenge with
significant implications for network security. Recent quantum machine learning
approaches, such as quantum kernel methods and variational quantum circuits,
have shown promise in capturing complex data distributions for anomaly
detection but remain constrained by limited qubit counts. We introduce in this
work a novel Quantum Gated Recurrent Unit (QGRU)-based Generative Adversarial
Network (GAN) employing Successive Data Injection (SuDaI) and a multi-metric
gating strategy for robust network anomaly detection. Our model uniquely
utilizes a quantum-enhanced generator that outputs parameters (mean and
log-variance) of a Gaussian distribution via reparameterization, combined with
a Wasserstein critic to stabilize adversarial training. Anomalies are
identified through a novel gating mechanism that initially flags potential
anomalies based on Gaussian uncertainty estimates and subsequently verifies
them using a composite of critic scores and reconstruction errors. Evaluated on
benchmark datasets, our method achieves a high time-series aware F1 score
(TaF1) of 89.43% demonstrating superior capability in detecting anomalies
accurately and promptly as compared to existing classical and quantum models.
Furthermore, the trained QGRU-WGAN was deployed on real IBM Quantum hardware,
where it retained high anomaly detection performance, confirming its robustness
and practical feasibility on current noisy intermediate-scale quantum (NISQ)
devices.

### 8. [Low-Altitude UAV-Carried Movable Antenna for Joint Wireless Power Transfer and Covert Communications](http://arxiv.org/pdf/2510.26628v1)

Authors: Chuang Zhang, Geng Sun, Jiahui Li, Jiacheng Wang, Qingqing Wu, Dusit Niyato, Shiwen Mao, Tony Q. S. Quek

The proliferation of Internet of Things (IoT) networks has created an urgent
need for sustainable energy solutions, particularly for the battery-constrained
spatially distributed IoT nodes. While low-altitude uncrewed aerial vehicles
(UAVs) employed with wireless power transfer (WPT) capabilities offer a
promising solution, the line-of-sight channels that facilitate efficient energy
delivery also expose sensitive operational data to adversaries. This paper
proposes a novel low-altitude UAV-carried movable antenna-enhanced transmission
system joint WPT and covert communications, which simultaneously performs
energy supplements to IoT nodes and establishes transmission links with a
covert user by leveraging wireless energy signals as a natural cover. Then, we
formulate a multi-objective optimization problem that jointly maximizes the
total harvested energy of IoT nodes and sum achievable rate of the covert user,
while minimizing the propulsion energy consumption of the low-altitude UAV. To
address the non-convex and temporally coupled optimization problem, we propose
a mixture-of-experts-augmented soft actor-critic (MoE-SAC) algorithm that
employs a sparse Top-K gated mixture-of-shallow-experts architecture to
represent multimodal policy distributions arising from the conflicting
optimization objectives. We also incorporate an action projection module that
explicitly enforces per-time-slot power budget constraints and antenna position
constraints. Simulation results demonstrate that the proposed approach
significantly outperforms some baseline approaches and other state-of-the-art
deep reinforcement learning algorithms.

### Robotics

### 1. [Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion](http://arxiv.org/pdf/2510.26067v1)

Authors: Chi Zhang, Mingrui Li, Wenzhe Tong, Xiaonan Huang

Tensegrity robots combine rigid rods and elastic cables, offering high
resilience and deployability but posing major challenges for locomotion control
due to their underactuated and highly coupled dynamics. This paper introduces a
morphology-aware reinforcement learning framework that integrates a graph
neural network (GNN) into the Soft Actor-Critic (SAC) algorithm. By
representing the robot's physical topology as a graph, the proposed GNN-based
policy captures coupling among components, enabling faster and more stable
learning than conventional multilayer perceptron (MLP) policies. The method is
validated on a physical 3-bar tensegrity robot across three locomotion
primitives, including straight-line tracking and bidirectional turning. It
shows superior sample efficiency, robustness to noise and stiffness variations,
and improved trajectory accuracy. Notably, the learned policies transfer
directly from simulation to hardware without fine-tuning, achieving stable
real-world locomotion. These results demonstrate the advantages of
incorporating structural priors into reinforcement learning for tensegrity
robot control.

### 2. [I don't Want You to Die: A Shared Responsibility Framework for Safeguarding Child-Robot Companionship](http://arxiv.org/pdf/2510.26080v1)

Authors: Fan Yang, Renkai Ma, Yaxin Hu, Michael Rodgers, Lingyao Li

Social robots like Moxie are designed to form strong emotional bonds with
children, but their abrupt discontinuation can cause significant struggles and
distress to children. When these services end, the resulting harm raises
complex questions of who bears responsibility when children's emotional bonds
are broken. Using the Moxie shutdown as a case study through a qualitative
survey of 72 U.S. participants, our findings show that the responsibility is
viewed as a shared duty across the robot company, parents, developers, and
government. However, these attributions varied by political ideology and
parental status of whether they have children. Participants' perceptions of
whether the robot service should continue are highly polarized; supporters
propose technical, financial, and governmental pathways for continuity, while
opponents cite business realities and risks of unhealthy emotional dependency.
Ultimately, this research contributes an empirically grounded shared
responsibility framework for safeguarding child-robot companionship by
detailing how accountability is distributed and contested, informing concrete
design and policy implications to mitigate the emotional harm of robot
discontinuation.

### 3. [Beyond the Uncanny Valley: A Mixed-Method Investigation of Anthropomorphism in Protective Responses to Robot Abuse](http://arxiv.org/pdf/2510.26082v1)

Authors: Fan Yang, Lingyao Li, Yaxin Hu, Michael Rodgers, Renkai Ma

Robots with anthropomorphic features are increasingly shaping how humans
perceive and morally engage with them. Our research investigates how different
levels of anthropomorphism influence protective responses to robot abuse,
extending the Computers as Social Actors (CASA) and uncanny valley theories
into a moral domain. In an experiment, we invite 201 participants to view
videos depicting abuse toward a robot with low (Spider), moderate (Two-Foot),
or high (Humanoid) anthropomorphism. To provide a comprehensive analysis, we
triangulate three modalities: self-report surveys measuring emotions and
uncanniness, physiological data from automated facial expression analysis, and
qualitative reflections. Findings indicate that protective responses are not
linear. The moderately anthropomorphic Two-Foot robot, rated highest in
eeriness and "spine-tingling" sensations consistent with the uncanny valley,
elicited the strongest physiological anger expressions. Self-reported anger and
guilt are significantly higher for both the Two-Foot and Humanoid robots
compared to the Spider. Qualitative findings further reveal that as
anthropomorphism increases, moral reasoning shifts from technical assessments
of property damage to condemnation of the abuser's character, while governance
proposals expand from property law to calls for quasi-animal rights and broader
societal responsibility. These results suggest that the uncanny valley does not
dampen moral concern but paradoxically heightens protective impulses, offering
critical implications for robot design, policy, and future legal frameworks.

### 4. [Embodied Intelligence for Advanced Bioinspired Microrobotics: Examples and Insights](http://arxiv.org/pdf/2510.26132v1)

Authors: Nestor O. Perez-Arancibia

The term embodied intelligence (EI) conveys the notion that body morphology,
material properties, interaction with the environment, and control strategies
can be purposefully integrated into the process of robotic design to generate
intelligent behavior; in particular, locomotion and navigation. In this paper,
we discuss EI as a design principle for advanced microrobotics, with a
particular focus on co-design -- the simultaneous and interdependent
development of physical structure and behavioral function. To illustrate the
contrast between EI-inspired systems and traditional architectures that
decouple sensing, computation, and actuation, we present and discuss a
collection of robots developed by the author and his team at the Autonomous
Microrobotic Systems Laboratory (AMSL). These robots exhibit intelligent
behavior that emerges from their structural dynamics and the physical
interaction between their components and with the environment. Platforms such
as the Bee++, RoBeetle, SMALLBug, SMARTI, WaterStrider, VLEIBot+, and FRISSHBot
exemplify how feedback loops, decision logics, sensing mechanisms, and smart
actuation strategies can be embedded into the physical properties of the
robotic system itself. Along these lines, we contend that co-design is not only
a method for empirical optimization under constraints, but also an enabler of
EI, offering a scalable and robust alternative to classical control for
robotics at the mm-to-cm-scale.

### 5. [Kinodynamic Task and Motion Planning using VLM-guided and Interleaved Sampling](http://arxiv.org/pdf/2510.26139v1)

Authors: Minseo Kwon, Young J. Kim

Task and Motion Planning (TAMP) integrates high-level task planning with
low-level motion feasibility, but existing methods are costly in long-horizon
problems due to excessive motion sampling. While LLMs provide commonsense
priors, they lack 3D spatial reasoning and cannot ensure geometric or dynamic
feasibility. We propose a kinodynamic TAMP framework based on a hybrid state
tree that uniformly represents symbolic and numeric states during planning,
enabling task and motion decisions to be jointly decided. Kinodynamic
constraints embedded in the TAMP problem are verified by an off-the-shelf
motion planner and physics simulator, and a VLM guides exploring a TAMP
solution and backtracks the search based on visual rendering of the states.
Experiments on the simulated domains and in the real world show 32.14% -
1166.67% increased average success rates compared to traditional and LLM-based
TAMP planners and reduced planning time on complex problems, with ablations
further highlighting the benefits of VLM guidance.

### 6. [Adaptive Trajectory Refinement for Optimization-based Local Planning in Narrow Passages](http://arxiv.org/pdf/2510.26142v1)

Authors: Hahjin Lee, Young J. Kim

Trajectory planning for mobile robots in cluttered environments remains a
major challenge due to narrow passages, where conventional methods often fail
or generate suboptimal paths. To address this issue, we propose the adaptive
trajectory refinement algorithm, which consists of two main stages. First, to
ensure safety at the path-segment level, a segment-wise conservative collision
test is applied, where risk-prone trajectory path segments are recursively
subdivided until collision risks are eliminated. Second, to guarantee
pose-level safety, pose correction based on penetration direction and line
search is applied, ensuring that each pose in the trajectory is collision-free
and maximally clear from obstacles. Simulation results demonstrate that the
proposed method achieves up to 1.69x higher success rates and up to 3.79x
faster planning times than state-of-the-art approaches. Furthermore, real-world
experiments confirm that the robot can safely pass through narrow passages
while maintaining rapid planning performance.

### 7. [PHUMA: Physically-Grounded Humanoid Locomotion Dataset](http://arxiv.org/pdf/2510.26236v1)

Authors: Kyungmin Lee, Sibeen Kim, Minho Park, Hyunseung Kim, Dongyoon Hwang, Hojoon Lee, Jaegul Choo

Motion imitation is a promising approach for humanoid locomotion, enabling
agents to acquire humanlike behaviors. Existing methods typically rely on
high-quality motion capture datasets such as AMASS, but these are scarce and
expensive, limiting scalability and diversity. Recent studies attempt to scale
data collection by converting large-scale internet videos, exemplified by
Humanoid-X. However, they often introduce physical artifacts such as floating,
penetration, and foot skating, which hinder stable imitation. In response, we
introduce PHUMA, a Physically-grounded HUMAnoid locomotion dataset that
leverages human video at scale, while addressing physical artifacts through
careful data curation and physics-constrained retargeting. PHUMA enforces joint
limits, ensures ground contact, and eliminates foot skating, producing motions
that are both large-scale and physically reliable. We evaluated PHUMA in two
sets of conditions: (i) imitation of unseen motion from self-recorded test
videos and (ii) path following with pelvis-only guidance. In both cases,
PHUMA-trained policies outperform Humanoid-X and AMASS, achieving significant
gains in imitating diverse motions. The code is available at
https://davian-robotics.github.io/PHUMA.

### 8. [Thor: Towards Human-Level Whole-Body Reactions for Intense Contact-Rich Environments](http://arxiv.org/pdf/2510.26280v1)

Authors: Gangyang Li, Qing Shi, Youhao Hu, Jincheng Hu, Zhongyuan Wang, Xinlong Wang, Shaqi Luo

Humanoids hold great potential for service, industrial, and rescue
applications, in which robots must sustain whole-body stability while
performing intense, contact-rich interactions with the environment. However,
enabling humanoids to generate human-like, adaptive responses under such
conditions remains a major challenge. To address this, we propose Thor, a
humanoid framework for human-level whole-body reactions in contact-rich
environments. Based on the robot's force analysis, we design a force-adaptive
torso-tilt (FAT2) reward function to encourage humanoids to exhibit human-like
responses during force-interaction tasks. To mitigate the high-dimensional
challenges of humanoid control, Thor introduces a reinforcement learning
architecture that decouples the upper body, waist, and lower body. Each
component shares global observations of the whole body and jointly updates its
parameters. Finally, we deploy Thor on the Unitree G1, and it substantially
outperforms baselines in force-interaction tasks. Specifically, the robot
achieves a peak pulling force of 167.7 N (approximately 48% of the G1's body
weight) when moving backward and 145.5 N when moving forward, representing
improvements of 68.9% and 74.7%, respectively, compared with the
best-performing baseline. Moreover, Thor is capable of pulling a loaded rack
(130 N) and opening a fire door with one hand (60 N). These results highlight
Thor's effectiveness in enhancing humanoid force-interaction capabilities.

### 9. [Towards Reinforcement Learning Based Log Loading Automation](http://arxiv.org/pdf/2510.26363v1)

Authors: Ilya Kurinov, Miroslav Ivanov, Grzegorz Orzechowski, Aki Mikkola

Forestry forwarders play a central role in mechanized timber harvesting by
picking up and moving logs from the felling site to a processing area or a
secondary transport vehicle. Forwarder operation is challenging and physically
and mentally exhausting for the operator who must control the machine in remote
areas for prolonged periods of time. Therefore, even partial automation of the
process may reduce stress on the operator. This study focuses on continuing
previous research efforts in application of reinforcement learning agents in
automating log handling process, extending the task from grasping which was
studied in previous research to full log loading operation. The resulting agent
will be capable to automate a full loading procedure from locating and
grappling to transporting and delivering the log to a forestry forwarder bed.
To train the agent, a trailer type forestry forwarder simulation model in
NVIDIA's Isaac Gym and a virtual environment for a typical log loading scenario
were developed. With reinforcement learning agents and a curriculum learning
approach, the trained agent may be a stepping stone towards application of
reinforcement learning agents in automation of the forestry forwarder. The
agent learnt grasping a log in a random position from grapple's random position
and transport it to the bed with 94% success rate of the best performing agent.

### 10. [RoboOS-NeXT: A Unified Memory-based Framework for Lifelong, Scalable, and Robust Multi-Robot Collaboration](http://arxiv.org/pdf/2510.26536v1)

Authors: Huajie Tan, Cheng Chi, Xiansheng Chen, Yuheng Ji, Zhongxia Zhao, Xiaoshuai Hao, Yaoxu Lyu, Mingyu Cao, Junkai Zhao, Huaihai Lyu, Enshen Zhou, Ning Chen, Yankai Fu, Cheng Peng, Wei Guo, Dong Liang, Zhuo Chen, Mengsi Lyu, Chenrui He, Yulong Ao, Yonghua Lin, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang

The proliferation of collaborative robots across diverse tasks and
embodiments presents a central challenge: achieving lifelong adaptability,
scalable coordination, and robust scheduling in multi-agent systems. Existing
approaches, from vision-language-action (VLA) models to hierarchical
frameworks, fall short due to their reliance on limited or dividual-agent
memory. This fundamentally constrains their ability to learn over long
horizons, scale to heterogeneous teams, or recover from failures, highlighting
the need for a unified memory representation. To address these limitations, we
introduce RoboOS-NeXT, a unified memory-based framework for lifelong, scalable,
and robust multi-robot collaboration. At the core of RoboOS-NeXT is the novel
Spatio-Temporal-Embodiment Memory (STEM), which integrates spatial scene
geometry, temporal event history, and embodiment profiles into a shared
representation. This memory-centric design is integrated into a
brain-cerebellum framework, where a high-level brain model performs global
planning by retrieving and updating STEM, while low-level controllers execute
actions locally. This closed loop between cognition, memory, and execution
enables dynamic task allocation, fault-tolerant collaboration, and consistent
state synchronization. We conduct extensive experiments spanning complex
coordination tasks in restaurants, supermarkets, and households. Our results
demonstrate that RoboOS-NeXT achieves superior performance across heterogeneous
embodiments, validating its effectiveness in enabling lifelong, scalable, and
robust multi-robot collaboration. Project website:
https://flagopen.github.io/RoboOS/

### Software Engineering

### 1. [Reduction of Test Re-runs by Prioritizing Potential Order Dependent Flaky Tests](http://arxiv.org/pdf/2510.26171v1)

Authors: Hasnain Iqbal, Zerina Begum, Kazi Sakib

Flaky tests can make automated software testing unreliable due to their
unpredictable behavior. These tests can pass or fail on the same code base on
multiple runs. However, flaky tests often do not refer to any fault, even
though they can cause the continuous integration (CI) pipeline to fail. A
common type of flaky test is the order-dependent (OD) test. The outcome of an
OD test depends on the order in which it is run with respect to other test
cases. Several studies have explored the detection and repair of OD tests.
However, their methods require re-runs of tests multiple times, that are not
related to the order dependence. Hence, prioritizing potential OD tests is
necessary to reduce the re-runs. In this paper, we propose a method to
prioritize potential order-dependent tests. By analyzing shared static fields
in test classes, we identify tests that are more likely to be order-dependent.
In our experiment on 27 project modules, our method successfully prioritized
all OD tests in 23 cases, reducing test executions by an average of 65.92% and
unnecessary re-runs by 72.19%. These results demonstrate that our approach
significantly improves the efficiency of OD test detection by lowering
execution costs.

### 2. [The "4W+1H" of Software Supply Chain Security Checklist for Critical Infrastructure](http://arxiv.org/pdf/2510.26174v1)

Authors: Liming Dong, Sung Une Lee, Zhenchang Xing, Muhammad Ejaz Ahmed, Stefan Avgoustakis

The increasing frequency and sophistication of software supply chain attacks
pose severe risks to critical infrastructure sectors, threatening national
security, economic stability, and public safety. Despite growing awareness,
existing security practices remain fragmented and insufficient, with most
frameworks narrowly focused on isolated life cycle stages or lacking alignment
with the specific needs of critical infrastructure (CI) sectors. In this paper,
we conducted a multivocal literature review across international frameworks,
Australian regulatory sources, and academic studies to identify and analyze
security practices across the software supply chain, especially specific CI
sector. Our analysis found that few existing frameworks are explicitly tailored
to CI domains. We systematically leveraged identified software supply chain
security frameworks, using a "4W+1H" analytical approach, we synthesized ten
core categories (what) of software supply chain security practices, mapped them
across life-cycle phases (when), stakeholder roles (who), and implementation
levels (how), and examined their coverage across existing frameworks (where).
Building on these insights, the paper culminates in structured, multi-layered
checklist of 80 questions designed to relevant stakeholders evaluate and
enhance their software supply chain security. Our findings reveal gaps between
framework guidance and sector-specific needs, highlight the need for
integrated, context-aware approaches to safeguard critical infrastructure from
evolving software supply chain risks.

### 3. [Empowering RepoQA-Agent based on Reinforcement Learning Driven by Monte-carlo Tree Search](http://arxiv.org/pdf/2510.26287v1)

Authors: Guochang Li, Yuchen Liu, Zhen Qin, Yunkun Wang, Jianping Zhong, Chen Zhi, Binhua Li, Fei Huang, Yongbin Li, Shuiguang Deng

Repository-level software engineering tasks require large language models
(LLMs) to efficiently navigate and extract information from complex codebases
through multi-turn tool interactions. Existing approaches face significant
limitations: training-free, in-context learning methods struggle to guide
agents effectively in tool utilization and decision-making based on
environmental feedback, while training-based approaches typically rely on
costly distillation from larger LLMs, introducing data compliance concerns in
enterprise environments. To address these challenges, we introduce
RepoSearch-R1, a novel agentic reinforcement learning framework driven by
Monte-carlo Tree Search (MCTS). This approach allows agents to generate
diverse, high-quality reasoning trajectories via self-training without
requiring model distillation or external supervision. Based on RepoSearch-R1,
we construct a RepoQA-Agent specifically designed for repository
question-answering tasks. Comprehensive evaluation on repository
question-answering tasks demonstrates that RepoSearch-R1 achieves substantial
improvements of answer completeness: 16.0% enhancement over no-retrieval
methods, 19.5% improvement over iterative retrieval methods, and 33% increase
in training efficiency compared to general agentic reinforcement learning
approaches. Our cold-start training methodology eliminates data compliance
concerns while maintaining robust exploration diversity and answer completeness
across repository-level reasoning tasks.

### 4. [Automated Extract Method Refactoring with Open-Source LLMs: A Comparative Study](http://arxiv.org/pdf/2510.26480v1)

Authors: Sivajeet Chand, Melih Kilic, Roland Würsching, Sushant Kumar Pandey, Alexander Pretschner

Automating the Extract Method refactoring (EMR) remains challenging and
largely manual despite its importance in improving code readability and
maintainability. Recent advances in open-source, resource-efficient Large
Language Models (LLMs) offer promising new approaches for automating such
high-level tasks. In this work, we critically evaluate five state-of-the-art
open-source LLMs, spanning 3B to 8B parameter sizes, on the EMR task for Python
code. We systematically assess functional correctness and code quality using
automated metrics and investigate the impact of prompting strategies by
comparing one-shot prompting to a Recursive criticism and improvement (RCI)
approach. RCI-based prompting consistently outperforms one-shot prompting in
test pass rates and refactoring quality. The best-performing models,
Deepseek-Coder-RCI and Qwen2.5-Coder-RCI, achieve test pass percentage (TPP)
scores of 0.829 and 0.808, while reducing lines of code (LOC) per method from
12.103 to 6.192 and 5.577, and cyclomatic complexity (CC) from 4.602 to 3.453
and 3.294, respectively. A developer survey on RCI-generated refactorings shows
over 70% acceptance, with Qwen2.5-Coder rated highest across all evaluation
criteria. In contrast, the original code scored below neutral, particularly in
readability and maintainability, underscoring the benefits of automated
refactoring guided by quality prompts. While traditional metrics like CC and
LOC provide useful signals, they often diverge from human judgments,
emphasizing the need for human-in-the-loop evaluation. Our open-source
benchmark offers a foundation for future research on automated refactoring with
LLMs.

### 5. [Envisioning Future Interactive Web Development: Editing Webpage with Natural Language](http://arxiv.org/pdf/2510.26516v1)

Authors: Truong Hai Dang, Jingyu Xiao, Yintong Huo

The evolution of web applications relies on iterative code modifications, a
process that is traditionally manual and time-consuming. While Large Language
Models (LLMs) can generate UI code, their ability to edit existing code from
new design requirements (e.g., "center the logo") remains a challenge. This is
largely due to the absence of large-scale, high-quality tuning data to align
model performance with human expectations. In this paper, we introduce a novel,
automated data generation pipeline that uses LLMs to synthesize a high-quality
fine-tuning dataset for web editing, named Instruct4Edit. Our approach
generates diverse instructions, applies the corresponding code modifications,
and performs visual verification to ensure correctness. By fine-tuning models
on Instruct4Edit, we demonstrate consistent improvement in translating human
intent into precise, structurally coherent, and visually accurate code changes.
This work provides a scalable and transparent foundation for natural language
based web editing, demonstrating that fine-tuning smaller open-source models
can achieve competitive performance with proprietary systems. We release all
data, code implementations, and model checkpoints for reproduction.

### 6. [Reflecting on Empirical and Sustainability Aspects of Software Engineering Research in the Era of Large Language Models](http://arxiv.org/pdf/2510.26538v1)

Authors: David Williams, Max Hort, Maria Kechagia, Aldeida Aleti, Justyna Petke, Federica Sarro

Software Engineering (SE) research involving the use of Large Language Models
(LLMs) has introduced several new challenges related to rigour in benchmarking,
contamination, replicability, and sustainability. In this paper, we invite the
research community to reflect on how these challenges are addressed in SE. Our
results provide a structured overview of current LLM-based SE research at ICSE,
highlighting both encouraging practices and persistent shortcomings. We
conclude with recommendations to strengthen benchmarking rigour, improve
replicability, and address the financial and environmental costs of LLM-based
SE.

### 7. ["Show Me You Comply... Without Showing Me Anything": Zero-Knowledge Software Auditing for AI-Enabled Systems](http://arxiv.org/pdf/2510.26576v1)

Authors: Filippo Scaramuzza, Renato Cordeiro Ferreira, Tomaz Maia Suller, Giovanni Quattrocchi, Damian Andrew Tamburri, Willem-Jan van den Heuvel

The increasing exploitation of Artificial Intelligence (AI) enabled systems
in critical domains has made trustworthiness concerns a paramount showstopper,
requiring verifiable accountability, often by regulation (e.g., the EU AI Act).
Classical software verification and validation techniques, such as procedural
audits, formal methods, or model documentation, are the mechanisms used to
achieve this. However, these methods are either expensive or heavily manual and
ill-suited for the opaque, "black box" nature of most AI models. An intractable
conflict emerges: high auditability and verifiability are required by law, but
such transparency conflicts with the need to protect assets being audited-e.g.,
confidential data and proprietary models-leading to weakened accountability. To
address this challenge, this paper introduces ZKMLOps, a novel MLOps
verification framework that operationalizes Zero-Knowledge Proofs
(ZKPs)-cryptographic protocols allowing a prover to convince a verifier that a
statement is true without revealing additional information-within
Machine-Learning Operations lifecycles. By integrating ZKPs with established
software engineering patterns, ZKMLOps provides a modular and repeatable
process for generating verifiable cryptographic proof of compliance. We
evaluate the framework's practicality through a study of regulatory compliance
in financial risk auditing and assess feasibility through an empirical
evaluation of top ZKP protocols, analyzing performance trade-offs for ML models
of increasing complexity.

### 8. [Online and Interactive Bayesian Inference Debugging](http://arxiv.org/pdf/2510.26579v1)

Authors: Nathanael Nussbaumer, Markus Böck, Jürgen Cito

Probabilistic programming is a rapidly developing programming paradigm which
enables the formulation of Bayesian models as programs and the automation of
posterior inference. It facilitates the development of models and conducting
Bayesian inference, which makes these techniques available to practitioners
from multiple fields. Nevertheless, probabilistic programming is notoriously
difficult as identifying and repairing issues with inference requires a lot of
time and deep knowledge. Through this work, we introduce a novel approach to
debugging Bayesian inference that reduces time and required knowledge
significantly. We discuss several requirements a Bayesian inference debugging
framework has to fulfill, and propose a new tool that meets these key
requirements directly within the development environment. We evaluate our
results in a study with 18 experienced participants and show that our approach
to online and interactive debugging of Bayesian inference significantly reduces
time and difficulty on inference debugging tasks.

### 9. [Stitch: Step-by-step LLM Guided Tutoring for Scratch](http://arxiv.org/pdf/2510.26634v1)

Authors: Yuan Si, Kyle Qi, Daming Li, Hanyuan Shi, Jialu Zhang

Block-based environments such as Scratch are increasingly popular in
programming education. While block syntax reduces surface errors, semantic bugs
remain common and challenging for novices to resolve. Existing debugging
workflows typically show the correct program directly to learners, a strategy
that may fix errors but undermines the development of problem-solving skills.
  We present Stitch, an interactive tutoring system that replaces "showing the
answer" with step-by-step scaffolding. The system's Diff-Analyze module
contrasts a student's project with a reference implementation, identifies the
most critical differences, and uses a large language model to explain why these
changes matter. Learners inspect highlighted blocks through a custom rendering
engine, understand the explanations, and selectively apply partial fixes. This
iterative process continues until the intended functionality is achieved.
  We evaluate Stitch in an empirical study, comparing it against a
state-of-the-art automated feedback generation tool for Scratch. Our key
insight is that simply presenting the correct program is pedagogically
ineffective. In contrast, our interactive, step-by-step guided system promotes
a more effective learning experience. More broadly, what constitutes effective
feedback in block-based programming remains an open question. Our evaluation
provides new evidence that step-by-step tutoring significantly enhances
learning outcomes, outperforming both direct-answer approaches and current
automated feedback generation tools.

### 10. [Process-based Indicators of Vulnerability Re-Introducing Code Changes: An Exploratory Case Study](http://arxiv.org/pdf/2510.26676v1)

Authors: Samiha Shimmi, Nicholas M. Synovic, Mona Rahimi, George K. Thiruvathukal

Software vulnerabilities often persist or re-emerge even after being fixed,
revealing the complex interplay between code evolution and socio-technical
factors. While source code metrics provide useful indicators of
vulnerabilities, software engineering process metrics can uncover patterns that
lead to their introduction. Yet few studies have explored whether process
metrics can reveal risky development activities over time -- insights that are
essential for anticipating and mitigating software vulnerabilities. This work
highlights the critical role of process metrics along with code changes in
understanding and mitigating vulnerability reintroduction. We move beyond
file-level prediction and instead analyze security fixes at the commit level,
focusing not only on whether a single fix introduces a vulnerability but also
on the longer sequences of changes through which vulnerabilities evolve and
re-emerge. Our approach emphasizes that reintroduction is rarely the result of
one isolated action, but emerges from cumulative development activities and
socio-technical conditions. To support this analysis, we conducted a case study
on the ImageMagick project by correlating longitudinal process metrics such as
bus factor, issue density, and issue spoilage with vulnerability reintroduction
activities, encompassing 76 instances of reintroduced vulnerabilities. Our
findings show that reintroductions often align with increased issue spoilage
and fluctuating issue density, reflecting short-term inefficiencies in issue
management and team responsiveness. These observations provide a foundation for
broader studies that combine process and code metrics to predict risky fixes
and strengthen software security.

### Social and Information Networks

### 1. [Signed Graph Unlearning](http://arxiv.org/pdf/2510.26092v1)

Authors: Zhifei Luo, Lin Li, Xiaohui Tao, Kaize Shi

The proliferation of signed networks in contemporary social media platforms
necessitates robust privacy-preserving mechanisms. Graph unlearning, which aims
to eliminate the influence of specific data points from trained models without
full retraining, becomes particularly critical in these scenarios where user
interactions are sensitive and dynamic. Existing graph unlearning methodologies
are exclusively designed for unsigned networks and fail to account for the
unique structural properties of signed graphs. Their naive application to
signed networks neglects edge sign information, leading to structural imbalance
across subgraphs and consequently degrading both model performance and
unlearning efficiency. This paper proposes SGU (Signed Graph Unlearning), a
graph unlearning framework specifically for signed networks. SGU incorporates a
new graph unlearning partition paradigm and a novel signed network partition
algorithm that preserve edge sign information during partitioning and ensure
structural balance across partitions. Compared with baselines, SGU achieves
state-of-the-art results in both model performance and unlearning efficiency.

### 2. [Simulating and Experimenting with Social Media Mobilization Using LLM Agents](http://arxiv.org/pdf/2510.26494v1)

Authors: Sadegh Shirani, Mohsen Bayati

Online social networks have transformed the ways in which political
mobilization messages are disseminated, raising new questions about how peer
influence operates at scale. Building on the landmark 61-million-person
Facebook experiment \citep{bond201261}, we develop an agent-based simulation
framework that integrates real U.S. Census demographic distributions, authentic
Twitter network topology, and heterogeneous large language model (LLM) agents
to examine the effect of mobilization messages on voter turnout. Each simulated
agent is assigned demographic attributes, a personal political stance, and an
LLM variant (\texttt{GPT-4.1}, \texttt{GPT-4.1-Mini}, or \texttt{GPT-4.1-Nano})
reflecting its political sophistication. Agents interact over realistic social
network structures, receiving personalized feeds and dynamically updating their
engagement behaviors and voting intentions. Experimental conditions replicate
the informational and social mobilization treatments of the original Facebook
study. Across scenarios, the simulator reproduces qualitative patterns observed
in field experiments, including stronger mobilization effects under social
message treatments and measurable peer spillovers. Our framework provides a
controlled, reproducible environment for testing counterfactual designs and
sensitivity analyses in political mobilization research, offering a bridge
between high-validity field experiments and flexible computational
modeling.\footnote{Code and data available at
https://github.com/CausalMP/LLM-SocioPol}

### 3. [Linking Heterogeneous Data with Coordinated Agent Flows for Social Media Analysis](http://arxiv.org/pdf/2510.26172v1)

Authors: Shifu Chen, Dazhen Deng, Zhihong Xu, Sijia Xu, Tai-Quan Peng, Yingcai Wu

Social media platforms generate massive volumes of heterogeneous data,
capturing user behaviors, textual content, temporal dynamics, and network
structures. Analyzing such data is crucial for understanding phenomena such as
opinion dynamics, community formation, and information diffusion. However,
discovering insights from this complex landscape is exploratory, conceptually
challenging, and requires expertise in social media mining and visualization.
Existing automated approaches, though increasingly leveraging large language
models (LLMs), remain largely confined to structured tabular data and cannot
adequately address the heterogeneity of social media analysis. We present SIA
(Social Insight Agents), an LLM agent system that links heterogeneous
multi-modal data -- including raw inputs (e.g., text, network, and behavioral
data), intermediate outputs, mined analytical results, and visualization
artifacts -- through coordinated agent flows. Guided by a bottom-up taxonomy
that connects insight types with suitable mining and visualization techniques,
SIA enables agents to plan and execute coherent analysis strategies. To ensure
multi-modal integration, it incorporates a data coordinator that unifies
tabular, textual, and network data into a consistent flow. Its interactive
interface provides a transparent workflow where users can trace, validate, and
refine the agent's reasoning, supporting both adaptability and trustworthiness.
Through expert-centered case studies and quantitative evaluation, we show that
SIA effectively discovers diverse and meaningful insights from social media
while supporting human-agent collaboration in complex analytical tasks.

### Systems and Control

### 1. [Competitive Equilibrium for Electricity Markets with Spatially Flexible Load](http://arxiv.org/pdf/2510.26036v1)

Authors: Nan Gu, Junjie Qin

Electric vehicle charging and geo-distributed datacenters introduce spatially
flexible loads (FLs) that couple power, transportation, and datacenter
networks. These couplings create a closed-loop feedback between locational
marginal prices (LMPs) and decisions of the FL systems, challenging the
foundations of conventional competitive equilibrium (CE) in electricity
markets. This paper studies a notion of generalized competitive equilibrium
(GCE) that aims to capture such price-demand interactions across the
interconnected infrastructures. We establish structural conditions under which
the GCE preserves key properties of the conventional CE, including existence,
uniqueness, and efficiency, without requiring detailed knowledge of decision
processes for individual FL systems. The framework generalizes to settings
where the grid is coupled with multiple FL systems. Stylized examples and case
studies on the New York ISO grid, coupled with the Sioux Falls transportation
and distributed datacenter networks, demonstrate the use of our theoretical
framework and illustrate the mutual influence among the grid and the studied FL
systems.

### 2. [A Scenario-Based Approach for Stochastic Economic Model Predictive Control with an Expected Shortfall Constraint](http://arxiv.org/pdf/2510.26063v1)

Authors: Alireza Arastou, Algo Carè, Ye Wang, Marco Campi, Erik Weyer

This paper presents a novel approach to stochastic economic model predictive
control (SEMPC) that minimizes average economic cost while satisfying an
empirical expected shortfall (EES) constraint to manage risk. A new
scenario-based problem formulation ensuring controlled risk with high
confidence while minimizing the average cost is introduced. The probabilistic
guarantees is dependent on the number of support elements over the entire input
domain, which is difficult to find for high-dimensional systems. A heuristic
algorithm is proposed to find the number of support elements. Finally, an
efficient method is presented to reduce the computational complexity of the
SEMPC problem with an EES constraint. The approach is validated on a water
distribution network, showing its effectiveness in balancing performance and
risk.

### 3. [Green Wireless Network Scaling for Joint Deployment: Multi-BSs or Multi-RISs?](http://arxiv.org/pdf/2510.26135v1)

Authors: Tao Yu, Simin Wang, Shunqing Zhang, Mingyao Cui, Kaibin Huang, Wen Chen, QingQing Wu, Jihong Li, Kaixuan Huang

The imminent emergence of sixth-generation (6G) networks faces critical
challenges from spatially heterogeneous traffic and escalating energy
consumption, necessitating sustainable scaling strategies for network
infrastructure such as base stations (BSs) and reconfigurable intelligent
surfaces (RISs). This paper establishes fundamental scaling laws for the
Integrated Relative Energy Efficiency (IREE) metric under joint multi-BS and
multi-RIS deployment in traffic-mismatched scenarios. Specifically, we propose
an Alternating Directional Dual-Radial Basis Function (ADD-RBF) framework that
models the channels of BSs and RISs as two type of spatially decoupled RBF
neurons to maximize IREE through alternative optimization, with proven
universal approximation capability and convergence guarantees. Theoretical
analysis reveals a scaling dichotomy: BS proliferation drives logarithmic
capacity growth $\mathcal{O}(\log N^{BS})$ but only polynomial mismatch
reduction $\mathcal{O}(1/\sqrt{N^{BS}})$, whereas RIS deployment achieves
exponential mismatch mitigation $\mathcal{O}(\delta_{\text{err}}^{-(N^R+1)})$
despite its sub-logarithmic capacity gains. Simulation results validate that
RISs excel in capturing spatial traffic correlations and alleviating hotspots,
making them particularly effective when mismatch dominates, while BSs are
preferable under capacity shortages. These findings offer practical guidelines
for green 6G network design.

### 4. [From Embedding to Control: Representations for Stochastic Multi-Object Systems](http://arxiv.org/pdf/2510.26344v1)

Authors: Xiaoyuan Cheng, Yiming Yang, Wei Jiang, Chenyang Yuan, Zhuo Sun, Yukun Hu

This paper studies how to achieve accurate modeling and effective control in
stochastic nonlinear dynamics with multiple interacting objects. However,
non-uniform interactions and random topologies make this task challenging. We
address these challenges by proposing \textit{Graph Controllable Embeddings}
(GCE), a general framework to learn stochastic multi-object dynamics for linear
control. Specifically, GCE is built on Hilbert space embeddings, allowing
direct embedding of probability distributions of controlled stochastic dynamics
into a reproducing kernel Hilbert space (RKHS), which enables linear operations
in its RKHS while retaining nonlinear expressiveness. We provide theoretical
guarantees on the existence, convergence, and applicability of GCE. Notably, a
mean field approximation technique is adopted to efficiently capture
inter-object dependencies and achieve provably low sample complexity. By
integrating graph neural networks, we construct data-dependent kernel features
that are capable of adapting to dynamic interaction patterns and generalizing
to even unseen topologies with only limited training instances. GCE scales
seamlessly to multi-object systems of varying sizes and topologies. Leveraging
the linearity of Hilbert spaces, GCE also supports simple yet effective control
algorithms for synthesizing optimal sequences. Experiments on physical systems,
robotics, and power grids validate GCE and demonstrate consistent performance
improvement over various competitive embedding methods in both in-distribution
and few-shot tests

### 5. [Command-filter-based trajectory-tracking control of quadrotor subject to internal and external disturbances](http://arxiv.org/pdf/2510.26368v1)

Authors: Mustafa Mohammed Mustafa

We propose a command-filter backstepping controller that integrates a
disturbance observer and a high-gain observer (HGO) to handle unknown internal
and external disturbances acting on a quadrotor. To build the controller, we
first define tracking errors between the measured and desired quadrotor
outputs, which allow the system to be rewritten in a new set of state
variables. Using this transformed model, we apply Lyapunov theory to derive a
backstepping control law. To avoid repeated differentiation of states and
virtual controls, a first-order command filter is introduced, and a nonlinear
disturbance observer is added to provide disturbance estimates. Each state in
the controller and observer is replaced with its estimate from the HGO. The
resulting control law enables the quadrotor to follow its path despite internal
and external disturbances, with each subsystem allowed its own disturbance type
for realism. A new state transformation and Lyapunov-based derivation prevent
the usual explosion of complexity, while the HGO reconstructs unmeasured states
and their rates for output feedback. The nonlinear disturbance observer
attenuates constant and nonlinear disturbances as well as band-limited white
noise. The method reduces dependence on high-precision sensors and mitigates
wind, model error, and rotor noise effects during flight. Unlike previous
studies that treat either disturbance rejection or partial sensing, this work
combines the command filter, disturbance observer, and HGO to address both
challenges simultaneously while avoiding the complexity growth typical of
backstepping designs.

### 6. [XWAVE: A Novel Software-Defined Everything Approach for the Manufacturing Industry](http://arxiv.org/pdf/2510.26393v1)

Authors: Juanjo Zulaika, Ibone Oleaga, Anne Sanz, Naia Presno, Aitor Landa-Arrue, Miguel Barón, María del Puy Carretero, Unai Lopez-Novoa

The manufacturing sector is moving from rigid, hardware-dependent systems
toward flexible, software-driven environments. This transformation is shaped by
the convergence of several Software-Defined technologies: Software-Defined
Automation virtualizes industrial control, replacing proprietary PLCs with
containerized, programmable solutions that enable scalability and
interoperability. Software-Defined Compute and Communications provide a means
to distribute intelligence seamlessly across devices, networks, and cloud
platforms, reducing latency and enabling dynamic reconfiguration.
Software-Defined Manufacturing Systems, usually implemented as Digital Twins,
are real-time virtual models of machines and processes, allowing predictive
analysis, optimization, and closer integration between human operators and
intelligent systems. This work presents XWAVE, a project that unites these
three Software-Defined paradigms to present a modular, fully software-defined
manufacturing system.

### 7. [Safety Margins of Inverse Optimal ISSf Controllers](http://arxiv.org/pdf/2510.26397v1)

Authors: Ziliang Lyu, Yiguang Hong, Lihua Xie, Miroslav Krstic

We investigate the gain margin of a general nonlinear system under an inverse
optimal input-to-state safe (ISSf) controller of the form u=u0(x)+u*(x,u0),
where u0 is the nominal control and u* is the inverse optimal safety filter
that minimally modifies the nominal controller's unsafe actions over the
infinite horizon. By first establishing a converse ISSf-BF theorem, we reveal
the equivalence among the achievability of ISSf by feedback, the achievability
of inverse optimality, and the solvability of a Hamilton-Jacobi-Isaacs equation
associated with the inverse optimal ISSf gain assignment. Then we develop a
collection of safety margin results on the overall control u=u0+u*. In the
absence of disturbances, we find that standard inverse optimal safe controllers
have a certain degree of gain margin. Specifically, when f(x) acts safely but
u0 acts unsafely, the gain can be decreased by up to half; and when f(x) acts
unsafely, we establish that, if u0 acts safely, the gain can be increased
arbitrarily, whereas if u0 acts unsafely, the control recovers the full gain
margin [1/2,inf). It is shown, however, that under control gain variation, the
safe set of these controllers is locally asymptotically stable, which implies
that their safety is sensitive to large but bounded disturbances. To make
inverse optimal ISSf controllers robust to gain variation, we propose a gain
margin improvement approach at the expense of an increased control effort. This
improvement allows the inverse optimal safe control to inherit the standard
gain margin of [1/2,inf) without requiring prior knowledge of whether f(x) or
u0 acts safely on the safety boundary, while simultaneously ensuring global
asymptotic stability of the resulting safe set. In the presence of
disturbances, this improvement idea renders inverse optimal ISSf controllers
robust to gain variations with the same gain margin of [1/2,inf).

### 8. [Two-Timescale Optimization Framework for IAB-Enabled Heterogeneous UAV Networks](http://arxiv.org/pdf/2510.26578v1)

Authors: Jikang Deng, Hui Zhou, Mohamed-Slim Alouini

In post-disaster scenarios, the rapid deployment of adequate communication
infrastructure is essential to support disaster search, rescue, and recovery
operations. To achieve this, uncrewed aerial vehicle (UAV) has emerged as a
promising solution for emergency communication due to its low cost and
deployment flexibility. However, conventional untethered UAV (U-UAV) is
constrained by size, weight, and power (SWaP) limitations, making it incapable
of maintaining the operation of a macro base station. To address this
limitation, we propose a heterogeneous UAV-based framework that integrates
tethered UAV (T-UAV) and U-UAVs, where U-UAVs are utilized to enhance the
throughput of cell-edge ground user equipments (G-UEs) and guarantee seamless
connectivity during G-UEs' mobility to safe zones. It is noted that the
integrated access and backhaul (IAB) technique is adopted to support the
wireless backhaul of U-UAVs. Accordingly, we formulate a two-timescale joint
user scheduling and trajectory control optimization problem, aiming to maximize
the downlink throughput under asymmetric traffic demands and G-UEs' mobility.
To solve the formulated problem, we proposed a two-timescale multi-agent deep
deterministic policy gradient (TTS-MADDPG) algorithm based on the centralized
training and distributed execution paradigm. Numerical results show that the
proposed algorithm outperforms other benchmarks, including the two-timescale
multi-agent proximal policy optimization (TTS-MAPPO) algorithm and MADDPG
scheduling method, with robust and higher throughput. Specifically, the
proposed algorithm obtains up to 12.2\% average throughput gain compared to the
MADDPG scheduling method.

### 9. [Optimal Bidding and Coordinated Dispatch of Hybrid Energy Systems in Regulation Markets](http://arxiv.org/pdf/2510.26602v1)

Authors: Tanmay Mishra, Dakota Hamilton, Mads R. Almassalkhi

The increasing integration of renewable energy sources and distributed energy
resources (DER) into modern power systems introduces significant uncertainty,
posing challenges for maintaining grid flexibility and reliability. Hybrid
energy systems (HES), composed of controllable generators, flexible loads, and
battery storage, offer a decentralized solution to enhance flexibility compared
to single centralized resources. This paper presents a two-level framework to
enable HES participation in frequency regulation markets. The upper level
performs a chance-constrained optimization to choose capacity bids based on
historical regulation signals. At the lower level, a real-time control strategy
disaggregates the regulation power among the constituent resources. This
real-time control strategy is then benchmarked against an offline optimal
dispatch to evaluate flexibility performance. Additionally, the framework
evaluates the profitability of overbidding strategies and identifies thresholds
beyond which performance degradation may lead to market penalties or
disqualification. The proposed framework also compare the impact of imbalance
of power capacities on performance and battery state of charge (SoC) through
asymmetric HES configurations.

### 10. [Graph approach for observability analysis in power system dynamic state estimation](http://arxiv.org/pdf/2510.26701v1)

Authors: Akhila Kandivalasa, Marcos Netto

The proposed approach yields a numerical method that provably executes in
linear time with respect to the number of nodes and edges in a graph. The
graph, constructed from the power system model, requires only knowledge of the
dependencies between state-to-state and output-to-state variables within a
state-space framework. While graph-based observability analysis methods exist
for power system static-state estimation, the approach presented here is the
first for dynamic-state estimation (DSE). We examine decentralized and
centralized DSE scenarios and compare our findings with a well-established,
albeit non-scalable, observability analysis method in the literature. When
compared to the latter in a centralized DSE setting, our method reduced
computation time by 1440x.

### Machine Learning (Statistics Category)

### 1. [$L_1$-norm Regularized Indefinite Kernel Logistic Regression](http://arxiv.org/pdf/2510.26043v1)

Authors: Shaoxin Wang, Hanjing Yao

Kernel logistic regression (KLR) is a powerful classification method widely
applied across diverse domains. In many real-world scenarios, indefinite
kernels capture more domain-specific structural information than positive
definite kernels. This paper proposes a novel $L_1$-norm regularized indefinite
kernel logistic regression (RIKLR) model, which extends the existing IKLR
framework by introducing sparsity via an $L_1$-norm penalty. The introduction
of this regularization enhances interpretability and generalization while
introducing nonsmoothness and nonconvexity into the optimization landscape. To
address these challenges, a theoretically grounded and computationally
efficient proximal linearized algorithm is developed. Experimental results on
multiple benchmark datasets demonstrate the superior performance of the
proposed method in terms of both accuracy and sparsity.

### 2. [Uncertainty-Aware Diagnostics for Physics-Informed Machine Learning](http://arxiv.org/pdf/2510.26121v1)

Authors: Mara Daniels, Liam Hodgkinson, Michael Mahoney

Physics-informed machine learning (PIML) integrates prior physical
information, often in the form of differential equation constraints, into the
process of fitting machine learning models to physical data. Popular PIML
approaches, including neural operators, physics-informed neural networks,
neural ordinary differential equations, and neural discrete equilibria, are
typically fit to objectives that simultaneously include both data and physical
constraints. However, the multi-objective nature of this approach creates
ambiguity in the measurement of model quality. This is related to a poor
understanding of epistemic uncertainty, and it can lead to surprising failure
modes, even when existing statistical metrics suggest strong fits. Working
within a Gaussian process regression framework, we introduce the
Physics-Informed Log Evidence (PILE) score. Bypassing the ambiguities of test
losses, the PILE score is a single, uncertainty-aware metric that provides a
selection principle for hyperparameters of a PIML model. We show that PILE
minimization yields excellent choices for a wide variety of model parameters,
including kernel bandwidth, least squares regularization weights, and even
kernel function selection. We also show that, even prior to data acquisition, a
special 'data-free' case of the PILE score identifies a priori kernel choices
that are 'well-adapted' to a given PDE. Beyond the kernel setting, we
anticipate that the PILE score can be extended to PIML at large, and we outline
approaches to do so.

### 3. [Multi-Output Robust and Conjugate Gaussian Processes](http://arxiv.org/pdf/2510.26401v1)

Authors: Joshua Rooijakkers, Leiv Rønneberg, François-Xavier Briol, Jeremias Knoblauch, Matias Altamirano

Multi-output Gaussian process (MOGP) regression allows modelling dependencies
among multiple correlated response variables. Similarly to standard Gaussian
processes, MOGPs are sensitive to model misspecification and outliers, which
can distort predictions within individual outputs. This situation can be
further exacerbated by multiple anomalous response variables whose errors
propagate due to correlations between outputs. To handle this situation, we
extend and generalise the robust and conjugate Gaussian process (RCGP)
framework introduced by Altamirano et al. (2024). This results in the
multi-output RCGP (MO-RCGP): a provably robust MOGP that is conjugate, and
jointly captures correlations across outputs. We thoroughly evaluate our
approach through applications in finance and cancer research.

### 4. [Statistical Inference for Matching Decisions via Matrix Completion under Dependent Missingness](http://arxiv.org/pdf/2510.26478v1)

Authors: Congyuan Duan, Wanteng Ma, Dong Xia, Kan Xu

This paper studies decision-making and statistical inference for two-sided
matching markets via matrix completion. In contrast to the independent sampling
assumed in classical matrix completion literature, the observed entries, which
arise from past matching data, are constrained by matching capacity. This
matching-induced dependence poses new challenges for both estimation and
inference in the matrix completion framework. We propose a non-convex algorithm
based on Grassmannian gradient descent and establish near-optimal entrywise
convergence rates for three canonical mechanisms, i.e., one-to-one matching,
one-to-many matching with one-sided random arrival, and two-sided random
arrival. To facilitate valid uncertainty quantification and hypothesis testing
on matching decisions, we further develop a general debiasing and projection
framework for arbitrary linear forms of the reward matrix, deriving asymptotic
normality with finite-sample guarantees under matching-induced dependent
sampling. Our empirical experiments demonstrate that the proposed approach
provides accurate estimation, valid confidence intervals, and efficient
evaluation of matching policies.

### 5. [LLMs as In-Context Meta-Learners for Model and Hyperparameter Selection](http://arxiv.org/pdf/2510.26510v1)

Authors: Youssef Attia El Hili, Albert Thomas, Malik Tiomoko, Abdelhakim Benechehab, Corentin Léger, Corinne Ancourt, Balázs Kégl

Model and hyperparameter selection are critical but challenging in machine
learning, typically requiring expert intuition or expensive automated search.
We investigate whether large language models (LLMs) can act as in-context
meta-learners for this task. By converting each dataset into interpretable
metadata, we prompt an LLM to recommend both model families and
hyperparameters. We study two prompting strategies: (1) a zero-shot mode
relying solely on pretrained knowledge, and (2) a meta-informed mode augmented
with examples of models and their performance on past tasks. Across synthetic
and real-world benchmarks, we show that LLMs can exploit dataset metadata to
recommend competitive models and hyperparameters without search, and that
improvements from meta-informed prompting demonstrate their capacity for
in-context meta-learning. These results highlight a promising new role for LLMs
as lightweight, general-purpose assistants for model selection and
hyperparameter optimization.

### 6. [On Measuring Localization of Shortcuts in Deep Networks](http://arxiv.org/pdf/2510.26560v1)

Authors: Nikita Tsoy, Nikola Konstantinov

Shortcuts, spurious rules that perform well during training but fail to
generalize, present a major challenge to the reliability of deep networks
(Geirhos et al., 2020). However, the impact of shortcuts on feature
representations remains understudied, obstructing the design of principled
shortcut-mitigation methods. To overcome this limitation, we investigate the
layer-wise localization of shortcuts in deep models. Our novel experiment
design quantifies the layer-wise contribution to accuracy degradation caused by
a shortcut-inducing skew by counterfactual training on clean and skewed
datasets. We employ our design to study shortcuts on CIFAR-10, Waterbirds, and
CelebA datasets across VGG, ResNet, DeiT, and ConvNeXt architectures. We find
that shortcut learning is not localized in specific layers but distributed
throughout the network. Different network parts play different roles in this
process: shallow layers predominantly encode spurious features, while deeper
layers predominantly forget core features that are predictive on clean data. We
also analyze the differences in localization and describe its principal axes of
variation. Finally, our analysis of layer-wise shortcut-mitigation strategies
suggests the hardness of designing general methods, supporting dataset- and
architecture-specific approaches instead.

### 7. [Action-Driven Processes for Continuous-Time Control](http://arxiv.org/pdf/2510.26672v1)

Authors: Ruimin He, Shaowei Lin

At the heart of reinforcement learning are actions -- decisions made in
response to observations of the environment. Actions are equally fundamental in
the modeling of stochastic processes, as they trigger discontinuous state
transitions and enable the flow of information through large, complex systems.
In this paper, we unify the perspectives of stochastic processes and
reinforcement learning through action-driven processes, and illustrate their
application to spiking neural networks. Leveraging ideas from
control-as-inference, we show that minimizing the Kullback-Leibler divergence
between a policy-driven true distribution and a reward-driven model
distribution for a suitably defined action-driven process is equivalent to
maximum entropy reinforcement learning.

### 8. [Assessment of the conditional exchangeability assumption in causal machine learning models: a simulation study](http://arxiv.org/pdf/2510.26700v1)

Authors: Gerard T. Portela, Jason B. Gibbons, Sebastian Schneeweiss, Rishi J. Desai

Observational studies developing causal machine learning (ML) models for the
prediction of individualized treatment effects (ITEs) seldom conduct empirical
evaluations to assess the conditional exchangeability assumption. We aimed to
evaluate the performance of these models under conditional exchangeability
violations and the utility of negative control outcomes (NCOs) as a diagnostic.
We conducted a simulation study to examine confounding bias in ITE estimates
generated by causal forest and X-learner models under varying conditions,
including the presence or absence of true heterogeneity. We simulated data to
reflect real-world scenarios with differing levels of confounding, sample size,
and NCO confounding structures. We then estimated and compared subgroup-level
treatment effects on the primary outcome and NCOs across settings with and
without unmeasured confounding. When conditional exchangeability was violated,
causal forest and X-learner models failed to recover true treatment effect
heterogeneity and, in some cases, falsely indicated heterogeneity when there
was none. NCOs successfully identified subgroups affected by unmeasured
confounding. Even when NCOs did not perfectly satisfy its ideal assumptions, it
remained informative, flagging potential bias in subgroup level estimates,
though not always pinpointing the subgroup with the largest confounding.
Violations of conditional exchangeability substantially limit the validity of
ITE estimates from causal ML models in routinely collected observational data.
NCOs serve a useful empirical diagnostic tool for detecting subgroup-specific
unmeasured confounding and should be incorporated into causal ML workflows to
support the credibility of individualized inference.

### 9. [Budgeted Multiple-Expert Deferral](http://arxiv.org/pdf/2510.26706v1)

Authors: Giulia DeSalvo, Clara Mohri, Mehryar Mohri, Yutao Zhong

Learning to defer uncertain predictions to costly experts offers a powerful
strategy for improving the accuracy and efficiency of machine learning systems.
However, standard training procedures for deferral algorithms typically require
querying all experts for every training instance, an approach that becomes
prohibitively expensive when expert queries incur significant computational or
resource costs. This undermines the core goal of deferral: to limit unnecessary
expert usage. To overcome this challenge, we introduce the budgeted deferral
framework, which aims to train effective deferral algorithms while minimizing
expert query costs during training. We propose new algorithms for both
two-stage and single-stage multiple-expert deferral settings that selectively
query only a subset of experts per training example. While inspired by active
learning, our setting is fundamentally different: labels are already known, and
the core challenge is to decide which experts to query in order to balance cost
and predictive performance. We establish theoretical guarantees for both of our
algorithms, including generalization bounds and label complexity analyses.
Empirical results across several domains show that our algorithms substantially
reduce training costs without sacrificing prediction accuracy, demonstrating
the practical value of our budget-aware deferral algorithms.

### 10. [Bias-Corrected Data Synthesis for Imbalanced Learning](http://arxiv.org/pdf/2510.26046v1)

Authors: Pengfei Lyu, Zhengchi Ma, Linjun Zhang, Anru R. Zhang

Imbalanced data, where the positive samples represent only a small proportion
compared to the negative samples, makes it challenging for classification
problems to balance the false positive and false negative rates. A common
approach to addressing the challenge involves generating synthetic data for the
minority group and then training classification models with both observed and
synthetic data. However, since the synthetic data depends on the observed data
and fails to replicate the original data distribution accurately, prediction
accuracy is reduced when the synthetic data is naively treated as the true
data. In this paper, we address the bias introduced by synthetic data and
provide consistent estimators for this bias by borrowing information from the
majority group. We propose a bias correction procedure to mitigate the adverse
effects of synthetic data, enhancing prediction accuracy while avoiding
overfitting. This procedure is extended to broader scenarios with imbalanced
data, such as imbalanced multi-task learning and causal inference. Theoretical
properties, including bounds on bias estimation errors and improvements in
prediction accuracy, are provided. Simulation results and data analysis on
handwritten digit datasets demonstrate the effectiveness of our method.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-31 PST.

### 1. [An intrusion detection system in the Internet of Things with deep learning and an improved arithmetic optimization algorithm (AOA) and sine cosine algorithm (SCA)](https://www.nature.com/articles/s41598-025-22074-3)

Authors: Raheleh Ghadami

### 2. [Development and application of a deep learning-based tuberculosis diagnostic assistance system in remote areas of Northwest China](https://www.nature.com/articles/s41598-025-22037-8)

Authors: Pahatijiang Nijiati et al.

### 3. [Intelligent ship traffic supervision system based on distributed blockchain and federated reinforcement learning for collaborative decision optimization](https://www.nature.com/articles/s41598-025-21898-3)

Authors: Zhang Wei et al.

### 4. [Lightweight pavement crack detection model for edge computing devices](https://www.nature.com/articles/s41598-025-22092-1)

Authors: Zhuang Li et al.

### 5. [A sterna migration algorithm-based efficient bionic engineering optimization algorithm](https://www.nature.com/articles/s41598-025-22038-7)

Authors: Hongwei Bai et al.

