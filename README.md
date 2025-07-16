# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-15 17:00:26.055480 PST.

### Artificial Intelligence

### 1. [Is Human-Written Data Enough? The Challenge of Teaching Reasoning to LLMs Without RL or Distillation](http://arxiv.org/pdf/2507.09850v1)

Authors: Wei Du, Branislav Kisacanin, George Armstrong, Shubham Toshniwal, Ivan Moshkov, Alexan Ayrapetyan, Sadegh Mahdavi, Dan Zhao, Shizhe Diao, Dragan Masulovic, Marius Stanean, Advaith Avadhanam, Max Wang, Ashmit Dutta, Shitij Govil, Sri Yanamandara, Mihir Tandon, Sriram Ananthakrishnan, Vedant Rathi, David Zhang, Joonseok Kang, Leon Luo, Titu Andreescu, Boris Ginsburg, Igor Gitman

Reasoning-capable language models achieve state-of-the-art performance in
diverse complex tasks by generating long, explicit Chain-of-Thought (CoT)
traces. While recent works show that base models can acquire such reasoning
traces via reinforcement learning or distillation from stronger models like
DeepSeek-R1, previous works demonstrate that even short CoT prompting without
fine-tuning is able to improve reasoning. We ask whether long CoT can be
induced in a base model using only prompting or minimal tuning. Using just 20
long CoT examples from the reasoning model \texttt{QwQ-32B-Preview}, we lightly
fine-tune the base model \texttt{Qwen2.5-32B}. The resulting model outperforms
the much larger \texttt{Qwen2.5-Math-72B-Instruct}, showing that a handful of
high-quality examples can unlock strong reasoning capabilities. We further
explore using CoT data from non-reasoning models and human annotators, enhanced
with prompt engineering, multi-pass editing, and structural guidance. However,
neither matches the performance of reasoning model traces, suggesting that
certain latent qualities of expert CoT are difficult to replicate. We analyze
key properties of reasoning data, such as problem difficulty, diversity, and
answer length, that influence reasoning distillation. While challenges remain,
we are optimistic that carefully curated human-written CoT, even in small
quantities, can activate reasoning behaviors in base models. We release our
human-authored dataset across refinement stages and invite further
investigation into what makes small-scale reasoning supervision so effective.

### 2. [Model-Grounded Symbolic Artificial Intelligence Systems Learning and Reasoning with Model-Grounded Symbolic Artificial Intelligence Systems](http://arxiv.org/pdf/2507.09854v1)

Authors: Aniruddha Chattopadhyay, Raj Dandekar, Kaushik Roy

Neurosymbolic artificial intelligence (AI) systems combine neural network and
classical symbolic AI mechanisms to exploit the complementary strengths of
large scale, generalizable learning and robust, verifiable reasoning. Numerous
classifications of neurosymbolic AI illustrate how these two components can be
integrated in distinctly different ways. In this work, we propose
reinterpreting instruction tuned large language models as model grounded
symbolic AI systems where natural language serves as the symbolic layer and
grounding is achieved through the models internal representation space. Within
this framework, we investigate and develop novel learning and reasoning
approaches that preserve structural similarities to traditional learning and
reasoning paradigms. Preliminary evaluations across axiomatic deductive
reasoning procedures of varying complexity provide insights into the
effectiveness of our approach in improving learning efficiency and reasoning
reliability.

### 3. [VerifyBench: A Systematic Benchmark for Evaluating Reasoning Verifiers Across Domains](http://arxiv.org/pdf/2507.09884v1)

Authors: Xuzhao Li, Xuchen Li, Shiyu Hu, Yongzhen Guo, Wentao Zhang

Large language models (LLMs) increasingly rely on reinforcement learning (RL)
to enhance their reasoning capabilities through feedback. A critical challenge
is verifying the consistency of model-generated responses and reference
answers, since these responses are often lengthy, diverse, and nuanced.
Rule-based verifiers struggle with complexity, prompting the use of model-based
verifiers. However, specialized verifiers lack flexibility, while general LLM
judges can be inconsistent. Existing research primarily focuses on building
better verifiers, yet a systematic evaluation of different types of verifiers'
performance across domains remains lacking, severely constraining the reliable
development of Reinforcement Learning with Verifiable Reward (RLVR). To address
this, we propose VerifyBench--a cross-domain comprehensive benchmark for
systematically evaluating verifiers. We construct 4,000 expert-level questions
covering mathematics, physics, chemistry, and biology. Each question is
equipped with reference answers and diverse responses. The reliability of the
evaluation is ensured through a rigorous annotation process conducted by a
multidisciplinary expert team. We design a four-dimensional experimental
framework to comprehensively compare the performance boundaries of specialized
verifiers and general LLMs under combined conditions of extracted answers vs.
complete responses, and short vs. long outputs. Our evaluation uncovers
fundamental trade-offs in verifiers: while specialized verifiers achieve
leading accuracy, they exhibit deficiencies in recall; general models show
stronger inclusivity but unstable precision. More importantly, we discover
verifiers' high sensitivity to input structure and inherent limitations in
cross-domain generalization, providing critical insights into the bottlenecks
of current verifier technology.

### 4. [DeepSeek: Paradigm Shifts and Technical Evolution in Large AI Models](http://arxiv.org/pdf/2507.09955v1)

Authors: Luolin Xiong, Haofen Wang, Xi Chen, Lu Sheng, Yun Xiong, Jingping Liu, Yanghua Xiao, Huajun Chen, Qing-Long Han, Yang Tang

DeepSeek, a Chinese Artificial Intelligence (AI) startup, has released their
V3 and R1 series models, which attracted global attention due to their low
cost, high performance, and open-source advantages. This paper begins by
reviewing the evolution of large AI models focusing on paradigm shifts, the
mainstream Large Language Model (LLM) paradigm, and the DeepSeek paradigm.
Subsequently, the paper highlights novel algorithms introduced by DeepSeek,
including Multi-head Latent Attention (MLA), Mixture-of-Experts (MoE),
Multi-Token Prediction (MTP), and Group Relative Policy Optimization (GRPO).
The paper then explores DeepSeek engineering breakthroughs in LLM scaling,
training, inference, and system-level optimization architecture. Moreover, the
impact of DeepSeek models on the competitive AI landscape is analyzed,
comparing them to mainstream LLMs across various fields. Finally, the paper
reflects on the insights gained from DeepSeek innovations and discusses future
trends in the technical and engineering development of large AI models,
particularly in data, training, and reasoning.

### 5. [Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](http://arxiv.org/pdf/2507.10007v1)

Authors: Zijun Chen, Wenbo Hu, Richang Hong

Chain of Thought (CoT) reasoning has demonstrated remarkable deep reasoning
capabilities in both large language models (LLMs) and multimodal large language
models (MLLMs). However, its reliability is often undermined by the
accumulation of errors in intermediate steps. This paper introduces an novel
approach to calibrate the CoT reasoning accuracy by leveraging the model's
intrinsic veracity encoding. We discover that specific attention head
activations reliably reflect the truthfulness of reasoning steps in CoT. Based
on this insight, we train a confidence predictor to evaluate the correctness of
each reasoning step using these truthfulness-sensitive activations, dynamically
selecting the most plausible reasoning path via beam search. Experimental
results demonstrate that our method significantly outperforms the
state-of-the-art baselines (e.g., Few-Shot CoT, Self-Consistency, and
Self-Evaluation Guided Beam Search) across the mathematical, symbolic, and
commonsense reasoning tasks, exhibiting superior accuracy and reliability in
both unimodal and multimodal settings. We further validate the approach on
large reasoning models, confirming its applicability to specialized reasoning
models. Additionally, we explore the role of the model's self-correction
ability in CoT reasoning. This work provides a novel reliability improvement
path for CoT reasoning with broad application potential.

### 6. [On Gradual Semantics for Assumption-Based Argumentation](http://arxiv.org/pdf/2507.10076v1)

Authors: Anna Rapberger, Fabrizio Russo, Antonio Rago, Francesca Toni

In computational argumentation, gradual semantics are fine-grained
alternatives to extension-based and labelling-based semantics . They ascribe a
dialectical strength to (components of) arguments sanctioning their degree of
acceptability. Several gradual semantics have been studied for abstract,
bipolar and quantitative bipolar argumentation frameworks (QBAFs), as well as,
to a lesser extent, for some forms of structured argumentation. However, this
has not been the case for assumption-based argumentation (ABA), despite it
being a popular form of structured argumentation with several applications
where gradual semantics could be useful. In this paper, we fill this gap and
propose a family of novel gradual semantics for equipping assumptions, which
are the core components in ABA frameworks, with dialectical strengths. To do
so, we use bipolar set-based argumentation frameworks as an abstraction of
(potentially non-flat) ABA frameworks and generalise state-of-the-art modular
gradual semantics for QBAFs. We show that our gradual ABA semantics satisfy
suitable adaptations of desirable properties of gradual QBAF semantics, such as
balance and monotonicity. We also explore an argument-based approach that
leverages established QBAF modular semantics directly, and use it as baseline.
Finally, we conduct experiments with synthetic ABA frameworks to compare our
gradual ABA semantics with its argument-based counterpart and assess
convergence.

### 7. [BlueGlass: A Framework for Composite AI Safety](http://arxiv.org/pdf/2507.10106v1)

Authors: Harshal Nandigramwar, Syed Qutub, Kay-Ulrich Scholl

As AI systems become increasingly capable and ubiquitous, ensuring the safety
of these systems is critical. However, existing safety tools often target
different aspects of model safety and cannot provide full assurance in
isolation, highlighting a need for integrated and composite methodologies. This
paper introduces BlueGlass, a framework designed to facilitate composite AI
safety workflows by providing a unified infrastructure enabling the integration
and composition of diverse safety tools that operate across model internals and
outputs. Furthermore, to demonstrate the utility of this framework, we present
three safety-oriented analyses on vision-language models for the task of object
detection: (1) distributional evaluation, revealing performance trade-offs and
potential failure modes across distributions; (2) probe-based analysis of layer
dynamics highlighting shared hierarchical learning via phase transition; and
(3) sparse autoencoders identifying interpretable concepts. More broadly, this
work contributes foundational infrastructure and findings for building more
robust and reliable AI systems.

### 8. [Could you be wrong: Debiasing LLMs using a metacognitive prompt for improving human decision making](http://arxiv.org/pdf/2507.10124v1)

Authors: Thomas T. Hills

Identifying bias in LLMs is ongoing. Because they are still in development,
what is true today may be false tomorrow. We therefore need general strategies
for debiasing that will outlive current models. Strategies developed for
debiasing human decision making offer one promising approach as they
incorporate an LLM-style prompt intervention designed to bring latent knowledge
into awareness during decision making. LLMs trained on vast amounts of
information contain information about potential biases, counter-arguments, and
contradictory evidence, but that information may only be brought to bear if
prompted. Metacognitive prompts developed in the human decision making
literature are designed to achieve this, and as I demonstrate here, they show
promise with LLMs. The prompt I focus on here is "could you be wrong?"
Following an LLM response, this prompt leads LLMs to produce additional
information, including why they answered as they did, errors, biases,
contradictory evidence, and alternatives, none of which were apparent in their
initial response. Indeed, this metaknowledge often reveals that how LLMs and
users interpret prompts are not aligned. Here I demonstrate this prompt using a
set of questions taken from recent articles about LLM biases, including
implicit discriminatory biases and failures of metacognition. "Could you be
wrong" prompts the LLM to identify its own biases and produce cogent
metacognitive reflection. I also present another example involving convincing
but incomplete information, which is readily corrected by the metacognitive
prompt. In sum, this work argues that human psychology offers a new avenue for
prompt engineering, leveraging a long history of effective prompt-based
improvements to human decision making.

### 9. [FRSICL: LLM-Enabled In-Context Learning Flight Resource Allocation for Fresh Data Collection in UAV-Assisted Wildfire Monitoring](http://arxiv.org/pdf/2507.10134v1)

Authors: Yousef Emami, Hao Zhou, Miguel Gutierrez Gaitan, Kai Li, Luis Almeida

Unmanned Aerial Vehicles (UAVs) are vital for public safety, particularly in
wildfire monitoring, where early detection minimizes environmental impact. In
UAV-Assisted Wildfire Monitoring (UAWM) systems, joint optimization of sensor
transmission scheduling and velocity is critical for minimizing Age of
Information (AoI) from stale sensor data. Deep Reinforcement Learning (DRL) has
been used for such optimization; however, its limitations such as low sampling
efficiency, simulation-to-reality gaps, and complex training render it
unsuitable for time-critical applications like wildfire monitoring. This paper
introduces a new online Flight Resource Allocation scheme based on LLM-Enabled
In-Context Learning (FRSICL) to jointly optimize the UAV's flight control and
data collection schedule along the trajectory in real time, thereby
asymptotically minimizing the average AoI across ground sensors. In contrast to
DRL, FRSICL generates data collection schedules and controls velocity using
natural language task descriptions and feedback from the environment, enabling
dynamic decision-making without extensive retraining. Simulation results
confirm the effectiveness of the proposed FRSICL compared to Proximal Policy
Optimization (PPO) and Nearest-Neighbor baselines.

### 10. [Introducing the Swiss Food Knowledge Graph: AI for Context-Aware Nutrition Recommendation](http://arxiv.org/pdf/2507.10156v1)

Authors: Lubnaa Abdur Rahman, Ioannis Papathanail, Stavroula Mougiakakou

AI has driven significant progress in the nutrition field, especially through
multimedia-based automatic dietary assessment. However, existing automatic
dietary assessment systems often overlook critical non-visual factors, such as
recipe-specific ingredient substitutions that can significantly alter
nutritional content, and rarely account for individual dietary needs, including
allergies, restrictions, cultural practices, and personal preferences. In
Switzerland, while food-related information is available, it remains
fragmented, and no centralized repository currently integrates all relevant
nutrition-related aspects within a Swiss context. To bridge this divide, we
introduce the Swiss Food Knowledge Graph (SwissFKG), the first resource, to our
best knowledge, to unite recipes, ingredients, and their substitutions with
nutrient data, dietary restrictions, allergen information, and national
nutrition guidelines under one graph. We establish a LLM-powered enrichment
pipeline for populating the graph, whereby we further present the first
benchmark of four off-the-shelf (<70 B parameter) LLMs for food knowledge
augmentation. Our results demonstrate that LLMs can effectively enrich the
graph with relevant nutritional information. Our SwissFKG goes beyond recipe
recommendations by offering ingredient-level information such as allergen and
dietary restriction information, and guidance aligned with nutritional
guidelines. Moreover, we implement a Graph-RAG application to showcase how the
SwissFKG's rich natural-language data structure can help LLM answer
user-specific nutrition queries, and we evaluate LLM-embedding pairings by
comparing user-query responses against predefined expected answers. As such,
our work lays the foundation for the next generation of dietary assessment
tools that blend visual, contextual, and cultural dimensions of eating.

### Hardware Architecture

### 1. [Iceberg: Enhancing HLS Modeling with Synthetic Data](http://arxiv.org/pdf/2507.09948v1)

Authors: Zijian Ding, Tung Nguyen, Weikai Li, Aditya Grover, Yizhou Sun, Jason Cong

Deep learning-based prediction models for High-Level Synthesis (HLS) of
hardware designs often struggle to generalize. In this paper, we study how to
close the generalizability gap of these models through pretraining on synthetic
data and introduce Iceberg, a synthetic data augmentation approach that expands
both large language model (LLM)-generated programs and weak labels of unseen
design configurations. Our weak label generation method is integrated with an
in-context model architecture, enabling meta-learning from actual and proximate
labels. Iceberg improves the geometric mean modeling accuracy by $86.4\%$ when
adapt to six real-world applications with few-shot examples and achieves a
$2.47\times$ and a $1.12\times$ better offline DSE performance when adapting to
two different test datasets. Our open-sourced code is here:
\href{https://github.com/UCLA-VAST/iceberg}{https://github.com/UCLA-VAST/iceberg}

### 2. [AnalogTester: A Large Language Model-Based Framework for Automatic Testbench Generation in Analog Circuit Design](http://arxiv.org/pdf/2507.09965v1)

Authors: Weiyu Chen, Chengjie Liu, Wenhao Huang, Jinyang Lyu, Mingqian Yang, Yuan Du, Li Du, Jun Yang

Recent advancements have demonstrated the significant potential of large
language models (LLMs) in analog circuit design. Nevertheless, testbench
construction for analog circuits remains manual, creating a critical bottleneck
in achieving fully automated design processes. Particularly when replicating
circuit designs from academic papers, manual Testbench construction demands
time-intensive implementation and frequent adjustments, which fails to address
the dynamic diversity and flexibility requirements for automation. AnalogTester
tackles automated analog design challenges through an LLM-powered pipeline: a)
domain-knowledge integration, b) paper information extraction, c) simulation
scheme synthesis, and d) testbench code generation with Tsinghua Electronic
Design (TED). AnalogTester has demonstrated automated Testbench generation
capabilities for three fundamental analog circuit types: operational amplifiers
(op-amps), bandgap references (BGRs), and low-dropout regulators (LDOs), while
maintaining a scalable framework for adaptation to broader circuit topologies.
Furthermore, AnalogTester can generate circuit knowledge data and TED code
corpus, establishing fundamental training datasets for LLM specialization in
analog circuit design automation.

### 3. [Pimba: A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving](http://arxiv.org/pdf/2507.10178v1)

Authors: Wonung Kim, Yubin Lee, Yoonsung Kim, Jinwoo Hwang, Seongryong Oh, Jiyong Jung, Aziz Huseynov, Woong Gyu Park, Chang Hyun Park, Divya Mahajan, Jongse Park

Transformers are the driving force behind today's Large Language Models
(LLMs), serving as the foundation for their performance and versatility. Yet,
their compute and memory costs grow with sequence length, posing scalability
challenges for long-context inferencing. In response, the algorithm community
is exploring alternative architectures, such as state space models (SSMs),
linear attention, and recurrent neural networks (RNNs), which we refer to as
post-transformers. This shift presents a key challenge: building a serving
system that efficiently supports both transformer and post-transformer LLMs
within a unified framework. To address this challenge, we analyze the
performance characteristics of transformer and post-transformer LLMs. Despite
their algorithmic differences, both are fundamentally limited by memory
bandwidth under batched inference due to attention in transformers and state
updates in post-transformers. Further analyses suggest two additional insights:
(1) state update operations, unlike attention, incur high hardware cost, making
per-bank PIM acceleration inefficient, and (2) different low-precision
arithmetic methods offer varying accuracy-area tradeoffs, while we identify
Microsoft's MX as the Pareto-optimal choice. Building on these insights, we
design Pimba as an array of State-update Processing Units (SPUs), each shared
between two banks to enable interleaved access to PIM. Each SPU includes a
State-update Processing Engine (SPE) that comprises element-wise multipliers
and adders using MX-based quantized arithmetic, enabling efficient execution of
state update and attention operations. Our evaluation shows that, compared to
LLM-optimized GPU and GPU+PIM systems, Pimba achieves up to 3.2x and 2.1x
higher token generation throughput, respectively.

### 4. [Solving the compute crisis with physics-based ASICs](http://arxiv.org/pdf/2507.10463v1)

Authors: Maxwell Aifer, Zach Belateche, Suraj Bramhavar, Kerem Y. Camsari, Patrick J. Coles, Gavin Crooks, Douglas J. Durian, Andrea J. Liu, Anastasia Marchenkova, Antonio J. Martinez, Peter L. McMahon, Faris Sbahi, Benjamin Weiner, Logan G. Wright

Escalating artificial intelligence (AI) demands expose a critical "compute
crisis" characterized by unsustainable energy consumption, prohibitive training
costs, and the approaching limits of conventional CMOS scaling. Physics-based
Application-Specific Integrated Circuits (ASICs) present a transformative
paradigm by directly harnessing intrinsic physical dynamics for computation
rather than expending resources to enforce idealized digital abstractions. By
relaxing the constraints needed for traditional ASICs, like enforced
statelessness, unidirectionality, determinism, and synchronization, these
devices aim to operate as exact realizations of physical processes, offering
substantial gains in energy efficiency and computational throughput. This
approach enables novel co-design strategies, aligning algorithmic requirements
with the inherent computational primitives of physical systems. Physics-based
ASICs could accelerate critical AI applications like diffusion models,
sampling, optimization, and neural network inference as well as traditional
computational workloads like scientific simulation of materials and molecules.
Ultimately, this vision points towards a future of heterogeneous,
highly-specialized computing platforms capable of overcoming current scaling
bottlenecks and unlocking new frontiers in computational power and efficiency.

### 5. [AssertCoder: LLM-Based Assertion Generation via Multimodal Specification Extraction](http://arxiv.org/pdf/2507.10338v1)

Authors: Enyuan Tian, Yiwei Ci, Qiusong Yang, Yufeng Li, Zhichao Lyu

Assertion-Based Verification (ABV) is critical for ensuring functional
correctness in modern hardware systems. However, manually writing high-quality
SVAs remains labor-intensive and error-prone. To bridge this gap, we propose
AssertCoder, a novel unified framework that automatically generates
high-quality SVAs directly from multimodal hardware design specifications.
AssertCoder employs a modality-sensitive preprocessing to parse heterogeneous
specification formats (text, tables, diagrams, and formulas), followed by a set
of dedicated semantic analyzers that extract structured representations aligned
with signal-level semantics. These representations are utilized to drive
assertion synthesis via multi-step chain-of-thought (CoT) prompting. The
framework incorporates a mutation-based evaluation approach to assess assertion
quality via model checking and further refine the generated assertions.
Experimental evaluation across three real-world Register-Transfer Level (RTL)
designs demonstrates AssertCoder's superior performance, achieving an average
increase of 8.4% in functional correctness and 5.8% in mutation detection
compared to existing state-of-the-art approaches.

### Computational Complexity

### 1. [Directed disjoint paths remains W[1]-hard on acyclic digraphs without large grid minors](http://arxiv.org/pdf/2507.09868v1)

Authors: Ken-ichi Kawarabayashi, Nicola Lorenz, Marcelo Garlet Milani, Jacob Stegemann

In the Vertex Disjoint Paths with Congestion problem, the input consists of a
digraph $D$, an integer $c$ and $k$ pairs of vertices $(s_i, t_i)$, and the
task is to find a set of paths connecting each $s_i$ to its corresponding
$t_i$, whereas each vertex of $D$ appears in at most $c$ many paths. The case
where $c = 1$ is known to be NP-complete even if $k = 2$ [Fortune, Hopcroft and
Wyllie, 1980] on general digraphs and is W[1]-hard with respect to $k$
(excluding the possibility of an $f(k)n^{O(1)}$-time algorithm under standard
assumptions) on acyclic digraphs [Slivkins, 2010]. The proof of [Slivkins,
2010] can also be adapted to show W[1]-hardness with respect to $k$ for every
congestion $c \geq 1$.
  We strengthen the existing hardness result by showing that the problem
remains W[1]-hard for every congestion $c \geq 1$ even if:
  - the input digraph $D$ is acyclic,
  - $D$ does not contain an acyclic $(5, 5)$-grid as a butterfly minor,
  - $D$ does not contain an acyclic tournament on 9 vertices as a butterfly
minor, and
  - $D$ has ear-anonymity at most 5.
  Further, we also show that the edge-congestion variant of the problem remains
W[1]-hard for every congestion $c \geq 1$ even if:
  - the input digraph $D$ is acyclic,
  - $D$ has maximum undirected degree 3,
  - $D$ does not contain an acyclic $(7, 7)$-wall as a weak immersion and
  - $D$ has ear-anonymity at most 5.

### 2. [Communication Complexity is NP-hard](http://arxiv.org/pdf/2507.10426v1)

Authors: Shuichi Hirahara, Rahul Ilango, Bruno Loff

In the paper where he first defined Communication Complexity, Yao asks:
\emph{Is computing $CC(f)$ (the 2-way communication complexity of a given
function $f$) NP-complete?} The problem of deciding whether $CC(f) \le k$, when
given the communication matrix for $f$ and a number $k$, is easily seen to be
in NP. Kushilevitz and Weinreb have shown that this problem is
cryptographically hard. Here we show it is NP-hard.

### 3. [Consensus, Inconsistency, Emergence: what's paraconsistency got to do with it?](http://arxiv.org/pdf/2507.10413v1)

Authors: Gabriel Rocha

The consensus problem, briefly stated, consists of having processes in an
asynchronous distributed system agree on a value. It is widely known that the
consensus problem does not have a deterministic solution that ensures both
termination and consistency, if there is at least one faulty process in the
system. This result, known as the FLP impossibility theorem, led to several
generalizations and developments in theoretical distributed computing. This
paper argues that the FLP impossibility theorem holds even under a generalized
definition of computation through oracles. Furthermore, using a theoretical
machinery from complex systems, this paper also posits that inconsistency may
be an emergent feature of consensus over distributed systems by examining how a
system transitions phases. Under the same complex systems framework, this paper
examines paraconsistent logics, arguing that while inconsistency is not an
emergent feature for these logics, triviality may be. Lastly, some attention is
given to the possibility of developing consensus algorithms capable of
paraconsistent reasoning.

### Computational Engineering

### 1. [Non-smooth optimization meets automated material model discovery](http://arxiv.org/pdf/2507.10196v1)

Authors: Moritz Flaschel, Trevor Hastie, Ellen Kuhl

Automated material model discovery disrupts the tedious and time-consuming
cycle of iteratively calibrating and modifying manually designed models.
Non-smooth L1-norm regularization is the backbone of automated model discovery;
however, the current literature on automated material model discovery offers
limited insights into the robust and efficient minimization of non-smooth
objective functions. In this work, we examine the minimization of functions of
the form f(w) + a ||w||_1, where w are the material model parameters, f is a
metric that quantifies the mismatch between the material model and the observed
data, and a is a regularization parameter that determines the sparsity of the
solution. We investigate both the straightforward case where f is quadratic and
the more complex scenario where it is non-quadratic or even non-convex.
Importantly, we do not only focus on methods that solve the sparse regression
problem for a given value of the regularization parameter a, but propose
methods to efficiently compute the entire regularization path, facilitating the
selection of a suitable a. Specifically, we present four algorithms and discuss
their roles for automated material model discovery in mechanics: First, we
recapitulate a well-known coordinate descent algorithm that solves the
minimization problem assuming that f is quadratic for a given value of a, also
known as the LASSO. Second, we discuss the algorithm LARS, which automatically
determines the critical values of a, at which material parameters in w are set
to zero. Third, we propose to use the proximal gradient method ISTA for
automated material model discovery if f is not quadratic, and fourth, we
suggest a pathwise extension of ISTA for computing the regularization path. We
demonstrate the applicability of all algorithms for the discovery of
hyperelastic material models from uniaxial tension and simple shear data.

### 2. [A Coincidence of Wants Mechanism for Swap Trade Execution in Decentralized Exchanges](http://arxiv.org/pdf/2507.10149v1)

Authors: Abhimanyu Nag, Madhur Prabhakar, Tanuj Behl

We propose a mathematically rigorous framework for identifying and completing
Coincidence of Wants (CoW) cycles in decentralized exchange (DEX) aggregators.
Unlike existing auction based systems such as CoWSwap, our approach introduces
an asset matrix formulation that not only verifies feasibility using oracle
prices and formal conservation laws but also completes partial CoW cycles of
swap orders that are discovered using graph traversal and are settled using
imbalance correction. We define bridging orders and show that the resulting
execution is slippage free and capital preserving for LPs. Applied to real
world Arbitrum swap data, our algorithm demonstrates efficient discovery of CoW
cycles and supports the insertion of synthetic orders for atomic cycle closure.
This work can be thought of as the detailing of a potential delta-neutral
strategy by liquidity providing market makers: a structured CoW cycle
execution.

### Computation and Language

### 1. [TextOmics-Guided Diffusion for Hit-like Molecular Generation](http://arxiv.org/pdf/2507.09982v1)

Authors: Hang Yuan, Chen Li, Wenjun Ma, Yuncheng Jiang

Hit-like molecular generation with therapeutic potential is essential for
target-specific drug discovery. However, the field lacks heterogeneous data and
unified frameworks for integrating diverse molecular representations. To bridge
this gap, we introduce TextOmics, a pioneering benchmark that establishes
one-to-one correspondences between omics expressions and molecular textual
descriptions. TextOmics provides a heterogeneous dataset that facilitates
molecular generation through representations alignment. Built upon this
foundation, we propose ToDi, a generative framework that jointly conditions on
omics expressions and molecular textual descriptions to produce biologically
relevant, chemically valid, hit-like molecules. ToDi leverages two encoders
(OmicsEn and TextEn) to capture multi-level biological and semantic
associations, and develops conditional diffusion (DiffGen) for controllable
generation. Extensive experiments confirm the effectiveness of TextOmics and
demonstrate ToDi outperforms existing state-of-the-art approaches, while also
showcasing remarkable potential in zero-shot therapeutic molecular generation.
Sources are available at: https://github.com/hala-ToDi.

### 2. [Protective Factor-Aware Dynamic Influence Learning for Suicide Risk Prediction on Social Media](http://arxiv.org/pdf/2507.10008v1)

Authors: Jun Li, Xiangmeng Wang, Haoyang Li, Yifei Yan, Hong Va Leong, Ling Feng, Nancy Xiaonan Yu, Qing Li

Suicide is a critical global health issue that requires urgent attention.
Even though prior work has revealed valuable insights into detecting current
suicide risk on social media, little attention has been paid to developing
models that can predict subsequent suicide risk over time, limiting their
ability to capture rapid fluctuations in individuals' mental state transitions.
In addition, existing work ignores protective factors that play a crucial role
in suicide risk prediction, focusing predominantly on risk factors alone.
Protective factors such as social support and coping strategies can mitigate
suicide risk by moderating the impact of risk factors. Therefore, this study
proposes a novel framework for predicting subsequent suicide risk by jointly
learning the dynamic influence of both risk factors and protective factors on
users' suicide risk transitions. We propose a novel Protective Factor-Aware
Dataset, which is built from 12 years of Reddit posts along with comprehensive
annotations of suicide risk and both risk and protective factors. We also
introduce a Dynamic Factors Influence Learning approach that captures the
varying impact of risk and protective factors on suicide risk transitions,
recognizing that suicide risk fluctuates over time according to established
psychological theories. Our thorough experiments demonstrate that the proposed
model significantly outperforms state-of-the-art models and large language
models across three datasets. In addition, the proposed Dynamic Factors
Influence Learning provides interpretable weights, helping clinicians better
understand suicidal patterns and enabling more targeted intervention
strategies.

### 3. [GeLaCo: An Evolutionary Approach to Layer Compression](http://arxiv.org/pdf/2507.10059v1)

Authors: David Ponce, Thierry Etchegoyhen, Javier Del Ser

Large Language Models (LLM) have achieved remarkable performance across a
large number of tasks, but face critical deployment and usage barriers due to
substantial computational requirements. Model compression methods, which aim to
reduce model size while preserving its capacity, are an important means to
mitigate these issues. Promising approaches along these lines, such as
structured pruning, typically require costly empirical search for optimal
variants and may run the risk of ignoring better solutions. In this work we
introduce GeLaCo, an evolutionary approach to LLM compression via layer
collapse. Our approach supports an efficient exploration of the compression
solution space via population-based search and a module-wise similarity fitness
function capturing attention, feed-forward, and hidden state representations.
GeLaCo also supports both single and multi-objective evolutionary compression
search, establishing the first Pareto frontier along compression and quality
axes. We evaluate GeLaCo solutions via both perplexity-based and generative
evaluations over foundational and instruction-tuned models, outperforming
state-of-the-art alternatives.

### 4. [Fusing Large Language Models with Temporal Transformers for Time Series Forecasting](http://arxiv.org/pdf/2507.10098v1)

Authors: Chen Su, Yuanhe Tian, Qinyu Liu, Jun Zhang, Yan Song

Recently, large language models (LLMs) have demonstrated powerful
capabilities in performing various tasks and thus are applied by recent studies
to time series forecasting (TSF) tasks, which predict future values with the
given historical time series. Existing LLM-based approaches transfer knowledge
learned from text data to time series prediction using prompting or fine-tuning
strategies. However, LLMs are proficient at reasoning over discrete tokens and
semantic patterns but are not initially designed to model continuous numerical
time series data. The gaps between text and time series data lead LLMs to
achieve inferior performance to a vanilla Transformer model that is directly
trained on TSF data. However, the vanilla Transformers often struggle to learn
high-level semantic patterns. In this paper, we design a novel
Transformer-based architecture that complementarily leverages LLMs and vanilla
Transformers, so as to integrate the high-level semantic representations
learned by LLMs into the temporal information encoded by time series
Transformers, where a hybrid representation is obtained by fusing the
representations from the LLM and the Transformer. The resulting fused
representation contains both historical temporal dynamics and semantic
variation patterns, allowing our model to predict more accurate future values.
Experiments on benchmark datasets demonstrate the effectiveness of the proposed
approach.

### 5. [Task-Based Flexible Feature Distillation for LLMs](http://arxiv.org/pdf/2507.10155v1)

Authors: Khouloud Saadi, Di Wang

Knowledge Distillation (KD) in general and feature distillation in particular
are promising techniques for reducing the high computational demand of large
language models (LLMs). However, traditional feature KD methods typically
assume that the teacher and the student share the same hidden size, limiting
the flexibility of the student's architecture. A common solution to this
problem involves training a linear projector to align their feature spaces, but
this introduces additional parameters that must be learned from scratch and
often degrades performance on downstream tasks, especially in generative
settings. To address this issue, in this work, we propose a novel task-based
feature distillation method that enables knowledge transfer between teacher and
student models with different hidden layer dimensions, without introducing any
new parameters. Leveraging the insight that only a subset of LLM components
contribute significantly to a specific downstream task, our approach identifies
the most task-relevant hidden units in the teacher and directly distills their
activations to the student. Our method is flexible and easily integrates with
other distillation frameworks. Empirical results show consistent improvements
over prior approaches across diverse tasks, including classification,
instruction-following, and summarization, achieving up to a 3\% performance
gain over the linear projection baseline.

### 6. [Grammar-Guided Evolutionary Search for Discrete Prompt Optimisation](http://arxiv.org/pdf/2507.10326v1)

Authors: Muzhaffar Hazman, Minh-Khoi Pham, Shweta Soundararajan, Goncalo Mordido, Leonardo Custode, David Lynch, Giorgio Cruciata, Yucheng Shi, Hongmeng Song, Wang Chao, Pan Yue, Aleksandar Milenovic, Alexandros Agapitos

Prompt engineering has proven to be a crucial step in leveraging pretrained
large language models (LLMs) in solving various real-world tasks. Numerous
solutions have been proposed that seek to automate prompt engineering by using
the model itself to edit prompts. However, the majority of state-of-the-art
approaches are evaluated on tasks that require minimal prompt templates and on
very large and highly capable LLMs. In contrast, solving complex tasks that
require detailed information to be included in the prompt increases the amount
of text that needs to be optimised. Furthermore, smaller models have been shown
to be more sensitive to prompt design. To address these challenges, we propose
an evolutionary search approach to automated discrete prompt optimisation
consisting of two phases. In the first phase, grammar-guided genetic
programming is invoked to synthesise prompt-creating programmes by searching
the space of programmes populated by function compositions of syntactic,
dictionary-based and LLM-based prompt-editing functions. In the second phase,
local search is applied to explore the neighbourhoods of best-performing
programmes in an attempt to further fine-tune their performance. Our approach
outperforms three state-of-the-art prompt optimisation approaches,
PromptWizard, OPRO, and RL-Prompt, on three relatively small general-purpose
LLMs in four domain-specific challenging tasks. We also illustrate several
examples where these benchmark methods suffer relatively severe performance
degradation, while our approach improves performance in almost all task-model
combinations, only incurring minimal degradation when it does not.

### 7. [Using AI to replicate human experimental results: a motion study](http://arxiv.org/pdf/2507.10342v1)

Authors: Rosa Illan Castillo, Javier Valenzuela

This paper explores the potential of large language models (LLMs) as reliable
analytical tools in linguistic research, focusing on the emergence of affective
meanings in temporal expressions involving manner-of-motion verbs. While LLMs
like GPT-4 have shown promise across a range of tasks, their ability to
replicate nuanced human judgements remains under scrutiny. We conducted four
psycholinguistic studies (on emergent meanings, valence shifts, verb choice in
emotional contexts, and sentence-emoji associations) first with human
participants and then replicated the same tasks using an LLM. Results across
all studies show a striking convergence between human and AI responses, with
statistical analyses (e.g., Spearman's rho = .73-.96) indicating strong
correlations in both rating patterns and categorical choices. While minor
divergences were observed in some cases, these did not alter the overall
interpretative outcomes. These findings offer compelling evidence that LLMs can
augment traditional human-based experimentation, enabling broader-scale studies
without compromising interpretative validity. This convergence not only
strengthens the empirical foundation of prior human-based findings but also
opens possibilities for hypothesis generation and data expansion through AI.
Ultimately, our study supports the use of LLMs as credible and informative
collaborators in linguistic inquiry.

### 8. [Meanings are like Onions: a Layered Approach to Metaphor Processing](http://arxiv.org/pdf/2507.10354v1)

Authors: Silvia Cappa, Anna Sofia Lippolis, Stefano Zoia

Metaphorical meaning is not a flat mapping between concepts, but a complex
cognitive phenomenon that integrates multiple levels of interpretation. In this
paper, we propose a stratified model of metaphor processing that treats meaning
as an onion: a multi-layered structure comprising (1) content analysis, (2)
conceptual blending, and (3) pragmatic intentionality. This three-dimensional
framework allows for a richer and more cognitively grounded approach to
metaphor interpretation in computational systems. At the first level, metaphors
are annotated through basic conceptual elements. At the second level, we model
conceptual combinations, linking components to emergent meanings. Finally, at
the third level, we introduce a pragmatic vocabulary to capture speaker intent,
communicative function, and contextual effects, aligning metaphor understanding
with pragmatic theories. By unifying these layers into a single formal
framework, our model lays the groundwork for computational methods capable of
representing metaphorical meaning beyond surface associations, toward deeper,
more context-sensitive reasoning.

### 9. [MLAR: Multi-layer Large Language Model-based Robotic Process Automation Applicant Tracking](http://arxiv.org/pdf/2507.10472v1)

Authors: Mohamed T. Younes, Omar Walid, Mai Hassan, Ali Hamdi

This paper introduces an innovative Applicant Tracking System (ATS) enhanced
by a novel Robotic process automation (RPA) framework or as further referred to
as MLAR. Traditional recruitment processes often encounter bottlenecks in
resume screening and candidate shortlisting due to time and resource
constraints. MLAR addresses these challenges employing Large Language Models
(LLMs) in three distinct layers: extracting key characteristics from job
postings in the first layer, parsing applicant resume to identify education,
experience, skills in the second layer, and similarity matching in the third
layer. These features are then matched through advanced semantic algorithms to
identify the best candidates efficiently. Our approach integrates seamlessly
into existing RPA pipelines, automating resume parsing, job matching, and
candidate notifications. Extensive performance benchmarking shows that MLAR
outperforms the leading RPA platforms, including UiPath and Automation
Anywhere, in high-volume resume-processing tasks. When processing 2,400
resumes, MLAR achieved an average processing time of 5.4 seconds per resume,
reducing processing time by approximately 16.9% compared to Automation Anywhere
and 17.1% compared to UiPath. These results highlight the potential of MLAR to
transform recruitment workflows by providing an efficient, accurate, and
scalable solution tailored to modern hiring needs.

### 10. [REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once](http://arxiv.org/pdf/2507.10541v1)

Authors: Zhuoshi Pan, Qizhi Pei, Yu Li, Qiyao Sun, Zinan Tang, H. Vicky Zhao, Conghui He, Lijun Wu

Recent Large Reasoning Models (LRMs) have achieved remarkable progress on
task-specific benchmarks, yet their evaluation methods remain constrained by
isolated problem-solving paradigms. Existing benchmarks predominantly assess
single-question reasoning through sequential testing, resulting critical
limitations: (1) vulnerability to data contamination and less challenging
(e.g., DeepSeek-R1 achieves 97.0% on MATH500), forcing costly and perpetual
creation of new questions with large human efforts, (2) failure to evaluate
models under multi-context pressure, a key requirement for real-world
deployment. To bridge this gap, we present REST (Reasoning Evaluation through
Simultaneous Testing), a stress-testing framework that concurrently exposes
LRMs to multiple problems simultaneously. Beyond basic reasoning, REST
specifically evaluates several under-tested capabilities: contextual priority
allocation, cross-problem interference resistance, and dynamic cognitive load
management. Our evaluation reveals several striking findings: Even
state-of-the-art (SOTA) models like DeepSeek-R1 exhibit substantial performance
degradation under stress testing. Crucially, REST demonstrates stronger
discriminative power than existing benchmarks, revealing pronounced performance
differences among models that exhibit similar, near-ceiling performance under
single-question evaluations. Some key mechanistic insights emerge from our
analysis: (1) the "overthinking trap" is a critical factor contributing to the
performance degradation; (2) the models trained with "long2short" technique
preserve more accuracy of their single-problem performance under REST,
outperforming standard-trained counterparts. These results establish REST as a
cost-efficient, future-proof evaluation paradigm that better reflects
real-world reasoning demands while reducing reliance on continuous human
annotation.

### Cryptography and Security

### 1. [HASSLE: A Self-Supervised Learning Enhanced Hijacking Attack on Vertical Federated Learning](http://arxiv.org/pdf/2507.10162v1)

Authors: Weiyang He, Chip-Hong Chang

Vertical Federated Learning (VFL) enables an orchestrating active party to
perform a machine learning task by cooperating with passive parties that
provide additional task-related features for the same training data entities.
While prior research has leveraged the privacy vulnerability of VFL to
compromise its integrity through a combination of label inference and backdoor
attacks, their effectiveness is constrained by the low label inference
precision and suboptimal backdoor injection conditions. To facilitate a more
rigorous security evaluation on VFL without these limitations, we propose
HASSLE, a hijacking attack framework composed of a gradient-direction-based
label inference module and an adversarial embedding generation algorithm
enhanced by self-supervised learning. HASSLE accurately identifies private
samples associated with a targeted label using only a single known instance of
that label. In the two-party scenario, it demonstrates strong performance with
an attack success rate (ASR) of over 99% across four datasets, including both
image and tabular modalities, and achieves 85% ASR on the more complex
CIFAR-100 dataset. Evaluation of HASSLE against 8 potential defenses further
highlights its significant threat while providing new insights into building a
trustworthy VFL system.

### 2. [SynthGuard: Redefining Synthetic Data Generation with a Scalable and Privacy-Preserving Workflow Framework](http://arxiv.org/pdf/2507.10489v1)

Authors: Eduardo Brito, Mahmoud Shoush, Kristian Tamm, Paula Etti, Liina Kamm

The growing reliance on data-driven applications in sectors such as
healthcare, finance, and law enforcement underscores the need for secure,
privacy-preserving, and scalable mechanisms for data generation and sharing.
Synthetic data generation (SDG) has emerged as a promising approach but often
relies on centralized or external processing, raising concerns about data
sovereignty, domain ownership, and compliance with evolving regulatory
standards. To overcome these issues, we introduce SynthGuard, a framework
designed to ensure computational governance by enabling data owners to maintain
control over SDG workflows. SynthGuard supports modular and privacy-preserving
workflows, ensuring secure, auditable, and reproducible execution across
diverse environments. In this paper, we demonstrate how SynthGuard addresses
the complexities at the intersection of domain-specific needs and scalable SDG
by aligning with requirements for data sovereignty and regulatory compliance.
Developed iteratively with domain expert input, SynthGuard has been validated
through real-world use cases, demonstrating its ability to balance security,
privacy, and scalability while ensuring compliance. The evaluation confirms its
effectiveness in implementing and executing SDG workflows and integrating
privacy and utility assessments across various computational environments.

### 3. [BURN: Backdoor Unlearning via Adversarial Boundary Analysis](http://arxiv.org/pdf/2507.10491v1)

Authors: Yanghao Su, Jie Zhang, Yiming Li, Tianwei Zhang, Qing Guo, Weiming Zhang, Nenghai Yu, Nils Lukas, Wenbo Zhou

Backdoor unlearning aims to remove backdoor-related information while
preserving the model's original functionality. However, existing unlearning
methods mainly focus on recovering trigger patterns but fail to restore the
correct semantic labels of poison samples. This limitation prevents them from
fully eliminating the false correlation between the trigger pattern and the
target label. To address this, we leverage boundary adversarial attack
techniques, revealing two key observations. First, poison samples exhibit
significantly greater distances from decision boundaries compared to clean
samples, indicating they require larger adversarial perturbations to change
their predictions. Second, while adversarial predicted labels for clean samples
are uniformly distributed, those for poison samples tend to revert to their
original correct labels. Moreover, the features of poison samples restore to
closely resemble those of corresponding clean samples after adding adversarial
perturbations. Building upon these insights, we propose Backdoor Unlearning via
adversaRial bouNdary analysis (BURN), a novel defense framework that integrates
false correlation decoupling, progressive data refinement, and model
purification. In the first phase, BURN employs adversarial boundary analysis to
detect poisoned samples based on their abnormal adversarial boundary distances,
then restores their correct semantic labels for fine-tuning. In the second
phase, it employs a feedback mechanism that tracks prediction discrepancies
between the original backdoored model and progressively sanitized models,
guiding both dataset refinement and model purification. Extensive evaluations
across multiple datasets, architectures, and seven diverse backdoor attack
types confirm that BURN effectively removes backdoor threats while maintaining
the model's original performance.

### 4. [AdvGrasp: Adversarial Attacks on Robotic Grasping from a Physical Perspective](http://arxiv.org/pdf/2507.09857v1)

Authors: Xiaofei Wang, Mingliang Han, Tianyu Hao, Cegang Li, Yunbo Zhao, Keke Tang

Adversarial attacks on robotic grasping provide valuable insights into
evaluating and improving the robustness of these systems. Unlike studies that
focus solely on neural network predictions while overlooking the physical
principles of grasping, this paper introduces AdvGrasp, a framework for
adversarial attacks on robotic grasping from a physical perspective.
Specifically, AdvGrasp targets two core aspects: lift capability, which
evaluates the ability to lift objects against gravity, and grasp stability,
which assesses resistance to external disturbances. By deforming the object's
shape to increase gravitational torque and reduce stability margin in the
wrench space, our method systematically degrades these two key grasping
metrics, generating adversarial objects that compromise grasp performance.
Extensive experiments across diverse scenarios validate the effectiveness of
AdvGrasp, while real-world validations demonstrate its robustness and practical
applicability

### 5. [Endorsement-Driven Blockchain SSI Framework for Dynamic IoT Ecosystems](http://arxiv.org/pdf/2507.09859v1)

Authors: Guntur Dharma Putra, Bagus Rakadyanto Oktavianto Putra

Self-Sovereign Identity (SSI) offers significant potential for managing
identities in the Internet of Things (IoT), enabling decentralized
authentication and credential management without reliance on centralized
entities. However, existing SSI frameworks often limit credential issuance and
revocation to trusted entities, such as IoT manufacturers, which restricts
flexibility in dynamic IoT ecosystems. In this paper, we propose a
blockchain-based SSI framework that allows any individual with a verifiable
trust linkage to act as a credential issuer, ensuring decentralized and
scalable identity management. Our framework incorporates a layered
architecture, where trust is dynamically established through endorsement-based
calculations and maintained via a hierarchical chain-of-trust mechanism.
Blockchain serves as the Verifiable Data Registry, ensuring transparency and
immutability of identity operations, while smart contracts automate critical
processes such as credential issuance, verification, and revocation. A
proof-of-concept implementation demonstrates that the proposed framework is
feasible and incurs minimal overheads compared to the baseline, making it
well-suited for dynamic and resource-constrained IoT environments.

### 6. [Secure and Efficient UAV-Based Face Detection via Homomorphic Encryption and Edge Computing](http://arxiv.org/pdf/2507.09860v1)

Authors: Nguyen Van Duc, Bui Duc Manh, Quang-Trung Luu, Dinh Thai Hoang, Van-Linh Nguyen, Diep N. Nguyen

This paper aims to propose a novel machine learning (ML) approach
incorporating Homomorphic Encryption (HE) to address privacy limitations in
Unmanned Aerial Vehicles (UAV)-based face detection. Due to challenges related
to distance, altitude, and face orientation, high-resolution imagery and
sophisticated neural networks enable accurate face recognition in dynamic
environments. However, privacy concerns arise from the extensive surveillance
capabilities of UAVs. To resolve this issue, we propose a novel framework that
integrates HE with advanced neural networks to secure facial data throughout
the inference phase. This method ensures that facial data remains secure with
minimal impact on detection accuracy. Specifically, the proposed system
leverages the Cheon-Kim-Kim-Song (CKKS) scheme to perform computations directly
on encrypted data, optimizing computational efficiency and security.
Furthermore, we develop an effective data encoding method specifically designed
to preprocess the raw facial data into CKKS form in a
Single-Instruction-Multiple-Data (SIMD) manner. Building on this, we design a
secure inference algorithm to compute on ciphertext without needing decryption.
This approach not only protects data privacy during the processing of facial
data but also enhances the efficiency of UAV-based face detection systems.
Experimental results demonstrate that our method effectively balances privacy
protection and detection performance, making it a viable solution for UAV-based
secure face detection. Significantly, our approach (while maintaining data
confidentially with HE encryption) can still achieve an accuracy of less than
1% compared to the benchmark without using encryption.

### 7. [Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](http://arxiv.org/pdf/2507.09990v1)

Authors: Ming Wen, Jiaqi Zhu, Yuedong Xu, Yipeng Zhou, Dingding Han

Large language models (LLMs) typically require fine-tuning for
domain-specific tasks, and LoRA offers a computationally efficient approach by
training low-rank adapters. LoRA is also communication-efficient for federated
LLMs when multiple users collaboratively fine-tune a global LLM model without
sharing their proprietary raw data. However, even the transmission of local
adapters between a server and clients risks serious privacy leakage. Applying
differential privacy (DP) to federated LoRA encounters a dilemma: adding noise
to both adapters amplifies synthetic noise on the model, while fixing one
adapter impairs the learnability of fine-tuning. In this paper, we propose
FedASK (Differentially Private Federated Low Rank Adaptation with Double
Sketching) , a novel federated LoRA framework to enable effective updating of
both low-rank adapters with robust differential privacy. Inspired by randomized
SVD, our key idea is a two-stage sketching pipeline. This pipeline first
aggregates carefully sketched, privacy-preserving local updates, and then
reconstructs the global matrices on the server to facilitate effective updating
of both adapters. We theoretically prove FedASK's differential privacy
guarantee and its exact aggregation property. Comprehensive experiments
demonstrate that FedASK consistently outperforms baseline methods across a
variety of privacy settings and data distributions.

### 8. [Accelerating Automatic Program Repair with Dual Retrieval-Augmented Fine-Tuning and Patch Generation on Large Language Models](http://arxiv.org/pdf/2507.10103v1)

Authors: Hanyang Guo, Xiaoheng Xie, Hong-Ning Dai, Peng Di, Yu Zhang, Bishenghui Tao, Zibin Zheng

Automated Program Repair (APR) is essential for ensuring software reliability
and quality while enhancing efficiency and reducing developers' workload.
Although rule-based and learning-based APR methods have demonstrated their
effectiveness, their performance was constrained by the defect type of repair,
the quality of training data, and the size of model parameters. Recently, Large
Language Models (LLMs) combined with Retrieval-Augmented-Generation (RAG) have
been increasingly adopted in APR tasks. However, current code LLMs and RAG
designs neither fully address code repair tasks nor consider code-specific
features. To overcome these limitations, we propose SelRepair, a novel APR
approach with integration of a fine-tuned LLM with a newly-designed dual RAG
module. This approach uses a bug-fix pair dataset for fine-tuning and
incorporates semantic and syntactic/structural similarity information through
an RAG selection gate. This design ensures relevant information is retrieved
efficiently, thereby reducing token length and inference time. Evaluations on
Java datasets show SelRepair outperforms other APR methods, achieving 26.29%
and 17.64% in terms of exact match (EM) on different datasets while reducing
inference time by at least 6.42% with controlled input lengths.

### 9. [Secure and Efficient Quantum Signature Scheme Based on the Controlled Unitary Operations Encryption](http://arxiv.org/pdf/2507.10233v1)

Authors: Debnath Ghosh, Soumit Roy, Prithwi Bagchi, Indranil Chakrabarty, Ashok Kumar Das

Quantum digital signatures ensure unforgeable message authenticity and
integrity using quantum principles, offering unconditional security against
both classical and quantum attacks. They are crucial for secure communication
in high-stakes environments, ensuring trust and long-term protection in the
quantum era. Nowadays, the majority of arbitrated quantum signature (AQS)
protocols encrypt data qubit by qubit using the quantum one-time pad (QOTP).
Despite providing robust data encryption, QOTP is not a good fit for AQS
because of its susceptibility to many types of attacks. In this work, we
present an efficient AQS protocol to encrypt quantum message ensembles using a
distinct encryption technique, the chained controlled unitary operations. In
contrast to existing protocols, our approach successfully prevents disavowal
and forgery attacks. We hope this contributes to advancing future
investigations into the development of AQS protocols.

### 10. [Split Happens: Combating Advanced Threats with Split Learning and Function Secret Sharing](http://arxiv.org/pdf/2507.10494v1)

Authors: Tanveer Khan, Mindaugas Budzys, Antonis Michalas

Split Learning (SL) -- splits a model into two distinct parts to help protect
client data while enhancing Machine Learning (ML) processes. Though promising,
SL has proven vulnerable to different attacks, thus raising concerns about how
effective it may be in terms of data privacy. Recent works have shown promising
results for securing SL through the use of a novel paradigm, named Function
Secret Sharing (FSS), in which servers obtain shares of a function they compute
and operate on a public input hidden with a random mask. However, these works
fall short in addressing the rising number of attacks which exist on SL. In
SplitHappens, we expand the combination of FSS and SL to U-shaped SL. Similarly
to other works, we are able to make use of the benefits of SL by reducing the
communication and computational costs of FSS. However, a U-shaped SL provides a
higher security guarantee than previous works, allowing a client to keep the
labels of the training data secret, without having to share them with the
server. Through this, we are able to generalize the security analysis of
previous works and expand it to different attack vectors, such as modern model
inversion attacks as well as label inference attacks. We tested our approach
for two different convolutional neural networks on different datasets. These
experiments show the effectiveness of our approach in reducing the training
time as well as the communication costs when compared to simply using FSS while
matching prior accuracy.

### Computer Vision and Pattern Recognition

### 1. [OpenHuman4D: Open-Vocabulary 4D Human Parsing](http://arxiv.org/pdf/2507.09880v1)

Authors: Keito Suzuki, Bang Du, Runfa Blark Li, Kunyao Chen, Lei Wang, Peng Liu, Ning Bi, Truong Nguyen

Understanding dynamic 3D human representation has become increasingly
critical in virtual and extended reality applications. However, existing human
part segmentation methods are constrained by reliance on closed-set datasets
and prolonged inference times, which significantly restrict their
applicability. In this paper, we introduce the first 4D human parsing framework
that simultaneously addresses these challenges by reducing the inference time
and introducing open-vocabulary capabilities. Building upon state-of-the-art
open-vocabulary 3D human parsing techniques, our approach extends the support
to 4D human-centric video with three key innovations: 1) We adopt mask-based
video object tracking to efficiently establish spatial and temporal
correspondences, avoiding the necessity of segmenting all frames. 2) A novel
Mask Validation module is designed to manage new target identification and
mitigate tracking failures. 3) We propose a 4D Mask Fusion module, integrating
memory-conditioned attention and logits equalization for robust embedding
fusion. Extensive experiments demonstrate the effectiveness and flexibility of
the proposed method on 4D human-centric parsing tasks, achieving up to 93.3%
acceleration compared to the previous state-of-the-art method, which was
limited to parsing fixed classes.

### 2. [Counterfactual Visual Explanation via Causally-Guided Adversarial Steering](http://arxiv.org/pdf/2507.09881v1)

Authors: Yiran Qiao, Disheng Liu, Yiren Lu, Yu Yin, Mengnan Du, Jing Ma

Recent work on counterfactual visual explanations has contributed to making
artificial intelligence models more explainable by providing visual
perturbation to flip the prediction. However, these approaches neglect the
causal relationships and the spurious correlations behind the image generation
process, which often leads to unintended alterations in the counterfactual
images and renders the explanations with limited quality. To address this
challenge, we introduce a novel framework CECAS, which first leverages a
causally-guided adversarial method to generate counterfactual explanations. It
innovatively integrates a causal perspective to avoid unwanted perturbations on
spurious factors in the counterfactuals. Extensive experiments demonstrate that
our method outperforms existing state-of-the-art approaches across multiple
benchmark datasets and ultimately achieves a balanced trade-off among various
aspects of validity, sparsity, proximity, and realism.

### 3. [MCGA: Mixture of Codebooks Hyperspectral Reconstruction via Grayscale-Aware Attention](http://arxiv.org/pdf/2507.09885v1)

Authors: Zhanjiang Yang, Lijun Sun, Jiawei Dong, Xiaoxin An, Yang Liu, Meng Li

Reconstructing hyperspectral images (HSI) from RGB images is a cost-effective
solution for various vision-based applications. However, most existing
learning-based hyperspectral reconstruction methods directly learn the
RGB-to-HSI mapping using complex attention mechanisms, neglecting the inherent
challenge of transitioning from low-dimensional to high-dimensional
information. To address this limitation, we propose a two-stage approach, MCGA,
which first learns spectral patterns before estimating the mapping. In the
first stage, a multi-scale VQ-VAE learns representations from heterogeneous HSI
datasets, extracting a Mixture of Codebooks (MoC). In the second stage, the
RGB-to-HSI mapping is refined by querying features from the MoC to replace
latent HSI representations, incorporating prior knowledge rather than forcing a
direct high-dimensional transformation. To further enhance reconstruction
quality, we introduce Grayscale-Aware Attention and Quantized Self-Attention,
which adaptively adjust feature map intensities to meet hyperspectral
reconstruction requirements. This physically motivated attention mechanism
ensures lightweight and efficient HSI recovery. Moreover, we propose an
entropy-based Test-Time Adaptation strategy to improve robustness in real-world
scenarios. Extensive experiments demonstrate that our method, MCGA, achieves
state-of-the-art performance. The code and models will be released at
https://github.com/Fibonaccirabbit/MCGA

### 4. [Measuring the Impact of Rotation Equivariance on Aerial Object Detection](http://arxiv.org/pdf/2507.09896v1)

Authors: Xiuyu Wu, Xinhao Wang, Xiubin Zhu, Lan Yang, Jiyuan Liu, Xingchen Hu

Due to the arbitrary orientation of objects in aerial images, rotation
equivariance is a critical property for aerial object detectors. However,
recent studies on rotation-equivariant aerial object detection remain scarce.
Most detectors rely on data augmentation to enable models to learn
approximately rotation-equivariant features. A few detectors have constructed
rotation-equivariant networks, but due to the breaking of strict rotation
equivariance by typical downsampling processes, these networks only achieve
approximately rotation-equivariant backbones. Whether strict rotation
equivariance is necessary for aerial image object detection remains an open
question. In this paper, we implement a strictly rotation-equivariant backbone
and neck network with a more advanced network structure and compare it with
approximately rotation-equivariant networks to quantitatively measure the
impact of rotation equivariance on the performance of aerial image detectors.
Additionally, leveraging the inherently grouped nature of rotation-equivariant
features, we propose a multi-branch head network that reduces the parameter
count while improving detection accuracy. Based on the aforementioned
improvements, this study proposes the Multi-branch head rotation-equivariant
single-stage Detector (MessDet), which achieves state-of-the-art performance on
the challenging aerial image datasets DOTA-v1.0, DOTA-v1.5 and DIOR-R with an
exceptionally low parameter count.

### 5. [IGD: Instructional Graphic Design with Multimodal Layer Generation](http://arxiv.org/pdf/2507.09910v1)

Authors: Yadong Qu, Shancheng Fang, Yuxin Wang, Xiaorui Wang, Zhineng Chen, Hongtao Xie, Yongdong Zhang

Graphic design visually conveys information and data by creating and
combining text, images and graphics. Two-stage methods that rely primarily on
layout generation lack creativity and intelligence, making graphic design still
labor-intensive. Existing diffusion-based methods generate non-editable graphic
design files at image level with poor legibility in visual text rendering,
which prevents them from achieving satisfactory and practical automated graphic
design. In this paper, we propose Instructional Graphic Designer (IGD) to
swiftly generate multimodal layers with editable flexibility with only natural
language instructions. IGD adopts a new paradigm that leverages parametric
rendering and image asset generation. First, we develop a design platform and
establish a standardized format for multi-scenario design files, thus laying
the foundation for scaling up data. Second, IGD utilizes the multimodal
understanding and reasoning capabilities of MLLM to accomplish attribute
prediction, sequencing and layout of layers. It also employs a diffusion model
to generate image content for assets. By enabling end-to-end training, IGD
architecturally supports scalability and extensibility in complex graphic
design tasks. The superior experimental results demonstrate that IGD offers a
new solution for graphic design.

### 6. [Crucial-Diff: A Unified Diffusion Model for Crucial Image and Annotation Synthesis in Data-scarce Scenarios](http://arxiv.org/pdf/2507.09915v1)

Authors: Siyue Yao, Mingjie Sun, Eng Gee Lim, Ran Yi, Baojiang Zhong, Moncef Gabbouj

The scarcity of data in various scenarios, such as medical, industry and
autonomous driving, leads to model overfitting and dataset imbalance, thus
hindering effective detection and segmentation performance. Existing studies
employ the generative models to synthesize more training samples to mitigate
data scarcity. However, these synthetic samples are repetitive or simplistic
and fail to provide "crucial information" that targets the downstream model's
weaknesses. Additionally, these methods typically require separate training for
different objects, leading to computational inefficiencies. To address these
issues, we propose Crucial-Diff, a domain-agnostic framework designed to
synthesize crucial samples. Our method integrates two key modules. The Scene
Agnostic Feature Extractor (SAFE) utilizes a unified feature extractor to
capture target information. The Weakness Aware Sample Miner (WASM) generates
hard-to-detect samples using feedback from the detection results of downstream
model, which is then fused with the output of SAFE module. Together, our
Crucial-Diff framework generates diverse, high-quality training data, achieving
a pixel-level AP of 83.63% and an F1-MAX of 78.12% on MVTec. On polyp dataset,
Crucial-Diff reaches an mIoU of 81.64% and an mDice of 87.69%. Code will be
released after acceptance.

### 7. [4D-MISR: A unified model for low-dose super-resolution imaging via feature fusion](http://arxiv.org/pdf/2507.09953v1)

Authors: Zifei Wang, Zian Mao, Xiaoya He, Xi Huang, Haoran Zhang, Chun Cheng, Shufen Chu, Tingzheng Hou, Xiaoqin Zeng, Yujun Xie

While electron microscopy offers crucial atomic-resolution insights into
structure-property relationships, radiation damage severely limits its use on
beam-sensitive materials like proteins and 2D materials. To overcome this
challenge, we push beyond the electron dose limits of conventional electron
microscopy by adapting principles from multi-image super-resolution (MISR) that
have been widely used in remote sensing. Our method fuses multiple
low-resolution, sub-pixel-shifted views and enhances the reconstruction with a
convolutional neural network (CNN) that integrates features from synthetic,
multi-angle observations. We developed a dual-path, attention-guided network
for 4D-STEM that achieves atomic-scale super-resolution from ultra-low-dose
data. This provides robust atomic-scale visualization across amorphous,
semi-crystalline, and crystalline beam-sensitive specimens. Systematic
evaluations on representative materials demonstrate comparable spatial
resolution to conventional ptychography under ultra-low-dose conditions. Our
work expands the capabilities of 4D-STEM, offering a new and generalizable
method for the structural analysis of radiation-vulnerable materials.

### 8. [Uncertainty Quantification for Incomplete Multi-View Data Using Divergence Measures](http://arxiv.org/pdf/2507.09980v1)

Authors: Zhipeng Xue, Yan Zhang, Ming Li, Chun Li, Yue Liu, Fei Yu

Existing multi-view classification and clustering methods typically improve
task accuracy by leveraging and fusing information from different views.
However, ensuring the reliability of multi-view integration and final decisions
is crucial, particularly when dealing with noisy or corrupted data. Current
methods often rely on Kullback-Leibler (KL) divergence to estimate uncertainty
of network predictions, ignoring domain gaps between different modalities. To
address this issue, KPHD-Net, based on H\"older divergence, is proposed for
multi-view classification and clustering tasks. Generally, our KPHD-Net employs
a variational Dirichlet distribution to represent class probability
distributions, models evidences from different views, and then integrates it
with Dempster-Shafer evidence theory (DST) to improve uncertainty estimation
effects. Our theoretical analysis demonstrates that Proper H\"older divergence
offers a more effective measure of distribution discrepancies, ensuring
enhanced performance in multi-view learning. Moreover, Dempster-Shafer evidence
theory, recognized for its superior performance in multi-view fusion tasks, is
introduced and combined with the Kalman filter to provide future state
estimations. This integration further enhances the reliability of the final
fusion results. Extensive experiments show that the proposed KPHD-Net
outperforms the current state-of-the-art methods in both classification and
clustering tasks regarding accuracy, robustness, and reliability, with
theoretical guarantees.

### 9. [Latent Diffusion Models with Masked AutoEncoders](http://arxiv.org/pdf/2507.09984v1)

Authors: Junho Lee, Jeongwoo Shin, Hyungwook Choi, Joonseok Lee

In spite of remarkable potential of the Latent Diffusion Models (LDMs) in
image generation, the desired properties and optimal design of the autoencoders
have been underexplored. In this work, we analyze the role of autoencoders in
LDMs and identify three key properties: latent smoothness, perceptual
compression quality, and reconstruction quality. We demonstrate that existing
autoencoders fail to simultaneously satisfy all three properties, and propose
Variational Masked AutoEncoders (VMAEs), taking advantage of the hierarchical
features maintained by Masked AutoEncoder. We integrate VMAEs into the LDM
framework, introducing Latent Diffusion Models with Masked AutoEncoders
(LDMAEs). Through comprehensive experiments, we demonstrate significantly
enhanced image generation quality and computational efficiency.

### 10. [3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving](http://arxiv.org/pdf/2507.09993v1)

Authors: Yixun Zhang, Lizhi Wang, Junjun Zhao, Wending Zhao, Feng Zhou, Yonghao Dang, Jianqin Yin

Camera-based object detection systems play a vital role in autonomous
driving, yet they remain vulnerable to adversarial threats in real-world
environments. While existing 2D and 3D physical attacks typically optimize
texture, they often struggle to balance physical realism and attack robustness.
In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel
adversarial object generation framework that leverages the full 14-dimensional
parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry
and appearance in physically realizable ways. Unlike prior works that rely on
patches or texture, 3DGAA jointly perturbs both geometric attributes (shape,
scale, rotation) and appearance attributes (color, opacity) to produce
physically realistic and transferable adversarial objects. We further introduce
a physical filtering module to preserve geometric fidelity, and a physical
augmentation module to simulate complex physical scenarios, thus enhancing
attack generalization under real-world conditions. We evaluate 3DGAA on both
virtual benchmarks and physical-world setups using miniature vehicle models.
Experimental results show that 3DGAA achieves to reduce the detection mAP from
87.21% to 7.38%, significantly outperforming existing 3D physical attacks.
Moreover, our method maintains high transferability across different physical
conditions, demonstrating a new state-of-the-art in physically realizable
adversarial attacks. These results validate 3DGAA as a practical attack
framework for evaluating the safety of perception systems in autonomous
driving.

### Computers and Society

### 1. [A New Incentive Model For Content Trust](http://arxiv.org/pdf/2507.09972v1)

Authors: Lucas Barbosa, Sam Kirshner, Rob Kopel, Eric Tze Kuan Lim, Tom Pagram

This paper outlines an incentive-driven and decentralized approach to
verifying the veracity of digital content at scale. Widespread misinformation,
an explosion in AI-generated content and reduced reliance on traditional news
sources demands a new approach for content authenticity and truth-seeking that
is fit for a modern, digital world. By using smart contracts and digital
identity to incorporate 'trust' into the reward function for published content,
not just engagement, we believe that it could be possible to foster a
self-propelling paradigm shift to combat misinformation through a
community-based governance model. The approach described in this paper requires
that content creators stake financial collateral on factual claims for an
impartial jury to vet with a financial reward for contribution. We hypothesize
that with the right financial and social incentive model users will be
motivated to participate in crowdsourced fact-checking and content creators
will place more care in their attestations. This is an exploratory paper and
there are a number of open issues and questions that warrant further analysis
and exploration.

### Databases

### 1. [Efficient Temporal Simple Path Graph Generation](http://arxiv.org/pdf/2507.10017v1)

Authors: Zhiyang Tang, Yanping Wu, Xiangjun Zai, Chen Chen, Xiaoyang Wang, Ying Zhang

Interactions between two entities often occur at specific timestamps, which
can be modeled as a temporal graph. Exploring the relationships between
vertices based on temporal paths is one of the fundamental tasks. In this
paper, we conduct the first research to propose and investigate the problem of
generating the temporal simple path graph (tspG), which is the subgraph
consisting of all temporal simple paths from the source vertex to the target
vertex within the given time interval. Directly enumerating all temporal simple
paths and constructing the tspG is computationally expensive. To accelerate the
processing, we propose an efficient method named Verification in Upper-bound
Graph. It first incorporates the temporal path constraint and simple path
constraint to exclude unpromising edges from the original graph, which obtains
a tight upper-bound graph as a high-quality approximation of the tspG in
polynomial time. Then, an Escape Edges Verification algorithm is further
applied in the upper-bound graph to construct the exact tspG without
exhaustively enumerating all temporal simple paths between given vertices.
Finally, comprehensive experiments on 10 real-world graphs are conducted to
demonstrate the efficiency and effectiveness of the proposed techniques.

### 2. [Breaking the Storage-Compute Bottleneck in Billion-Scale ANNS: A GPU-Driven Asynchronous I/O Framework](http://arxiv.org/pdf/2507.10070v1)

Authors: Yang Xiao, Mo Sun, Ziyu Song, Bing Tian, Jie Zhang, Jie Sun, Zeke Wang

With the advancement of information retrieval, recommendation systems, and
Retrieval-Augmented Generation (RAG), Approximate Nearest Neighbor Search
(ANNS) gains widespread applications due to its higher performance and
accuracy. While several disk-based ANNS systems have emerged to handle
exponentially growing vector datasets, they suffer from suboptimal performance
due to two inherent limitations: 1) failing to overlap SSD accesses with
distance computation processes and 2) extended I/O latency caused by suboptimal
I/O Stack. To address these challenges, we present FlashANNS, a GPU-accelerated
out-of-core graph-based ANNS system through I/O-compute overlapping. Our core
insight lies in the synchronized orchestration of I/O and computation through
three key innovations: 1) Dependency-Relaxed asynchronous pipeline: FlashANNS
decouples I/O-computation dependencies to fully overlap between GPU distance
calculations and SSD data transfers. 2) Warp-Level concurrent SSD access:
FlashANNS implements a lock-free I/O stack with warp-level concurrency control,
to reduce the latency-induced time overhead. 3) Computation-I/O balanced graph
degree Selection: FlashANNS selects graph degrees via lightweight
compute-to-I/O ratio sampling, ensuring optimal balance between computational
load and storage access latency across different I/O bandwidth configurations.
We implement FlashANNS and compare it with state-of-the-art out-of-core ANNS
systems (SPANN, DiskANN) and a GPU-accelerated out-of-core ANNS system
(FusionANNS). Experimental results demonstrate that at $\geq$95\% recall@10
accuracy, our method achieves 2.3-5.9$\times$ higher throughput compared to
existing SOTA methods with a single SSD, and further attains 2.7-12.2$\times$
throughput improvement in multi-SSD configurations.

### 3. [LogLite: Lightweight Plug-and-Play Streaming Log Compression](http://arxiv.org/pdf/2507.10337v1)

Authors: Benzhao Tang, Shiyu Yang, Zhitao Shen, Wenjie Zhang, Xuemin Lin, Zhihong Tian

Log data is a vital resource for capturing system events and states. With the
increasing complexity and widespread adoption ofmodern software systems and IoT
devices, the daily volume of log generation has surged to tens of petabytes,
leading to significant collection and storage costs. To address this challenge,
lossless log compression has emerged as an effective solution, enabling
substantial resource savings without compromising log information. In this
paper, we first conduct a characterization study on extensive public log
datasets and identify four key observations. Building on these insights, we
propose LogLite, a lightweight, plug-and-play, streaming lossless compression
algorithm designed to handle both TEXT and JSON logs throughout their life
cycle. LogLite requires no predefined rules or pre-training and is inherently
adaptable to evolving log structures. Our evaluation shows that, compared to
state-of-the-art baselines, LogLite achieves Pareto optimality in most
scenarios, delivering an average improvement of up to 67.8% in compression
ratio and up to 2.7 $\times$ in compression speed.

### 4. [Instance-Optimized String Fingerprints](http://arxiv.org/pdf/2507.10391v1)

Authors: Mihail Stoian, Johannes Thürauf, Andreas Zimmerer, Alexander van Renen, Andreas Kipf

Recent research found that cloud data warehouses are text-heavy. However,
their capabilities for efficiently processing string columns remain limited,
relying primarily on techniques like dictionary encoding and prefix-based
partition pruning. In recent work, we introduced string fingerprints - a
lightweight secondary index structure designed to approximate LIKE predicates,
albeit with false positives. This approach is particularly compelling for
columnar query engines, where fingerprints can help reduce both compute and I/O
overhead. We show that string fingerprints can be optimized for specific
workloads using mixed-integer optimization, and that they can generalize to
unseen table predicates. On an IMDb column evaluated in DuckDB v1.3, this
yields table-scan speedups of up to 1.36$\times$.

### 5. [Toward Real-World Table Agents: Capabilities, Workflows, and Design Principles for LLM-based Table Intelligence](http://arxiv.org/pdf/2507.10281v1)

Authors: Jiaming Tian, Liyao Li, Wentao Ye, Haobo Wang, Lingxin Wang, Lihua Yu, Zujie Ren, Gang Chen, Junbo Zhao

Tables are fundamental in domains such as finance, healthcare, and public
administration, yet real-world table tasks often involve noise, structural
heterogeneity, and semantic complexity--issues underexplored in existing
research that primarily targets clean academic datasets. This survey focuses on
LLM-based Table Agents, which aim to automate table-centric workflows by
integrating preprocessing, reasoning, and domain adaptation. We define five
core competencies--C1: Table Structure Understanding, C2: Table and Query
Semantic Understanding, C3: Table Retrieval and Compression, C4: Executable
Reasoning with Traceability, and C5: Cross-Domain Generalization--to analyze
and compare current approaches. In addition, a detailed examination of the
Text-to-SQL Agent reveals a performance gap between academic benchmarks and
real-world scenarios, especially for open-source models. Finally, we provide
actionable insights to improve the robustness, generalization, and efficiency
of LLM-based Table Agents in practical settings.

### 6. [Sampling-Based Estimation of Jaccard Containment and Similarity](http://arxiv.org/pdf/2507.10019v1)

Authors: Pranav Joshi

This paper addresses the problem of estimating the containment and similarity
between two sets using only random samples from each set, without relying on
sketches or full data access. The study introduces a binomial model for
predicting the overlap between samples, demonstrating that it is both accurate
and practical when sample sizes are small compared to the original sets. The
paper compares this model to previous approaches and shows that it provides
better estimates under the considered conditions. It also analyzes the
statistical properties of the estimator, including error bounds and sample size
requirements needed to achieve a desired level of accuracy and confidence. The
framework is extended to estimate set similarity, and the paper provides
guidance for applying these methods in large scale data systems where only
partial or sampled data is available.

### Distributed, Parallel, and Cluster Computing

### 1. [Intelligent Task Management via Dynamic Multi-region Division in LEO Satellite Networks](http://arxiv.org/pdf/2507.09926v1)

Authors: Zixuan Song, Zhishu Shen, Xiaoyu Zheng, Qiushi Zheng, Zheng Lei, Jiong Jin

As a key complement to terrestrial networks and a fundamental component of
future 6G systems, Low Earth Orbit (LEO) satellite networks are expected to
provide high-quality communication services when integrated with ground-based
infrastructure, thereby attracting significant research interest. However, the
limited satellite onboard resources and the uneven distribution of
computational workloads often result in congestion along inter-satellite links
(ISLs) that degrades task processing efficiency. Effectively managing the
dynamic and large-scale topology of LEO networks to ensure balanced task
distribution remains a critical challenge. To this end, we propose a dynamic
multi-region division framework for intelligent task management in LEO
satellite networks. This framework optimizes both intra- and inter-region
routing to minimize task delay while balancing the utilization of computational
and communication resources. Based on this framework, we propose a dynamic
multi-region division algorithm based on the Genetic Algorithm (GA), which
adaptively adjusts the size of each region based on the workload status of
individual satellites. Additionally, we incorporate an adaptive routing
algorithm and a task splitting and offloading scheme based on Multi-Agent Deep
Deterministic Policy Gradient (MA-DDPG) to effectively accommodate the arriving
tasks. Simulation results demonstrate that our proposed framework outperforms
comparative methods in terms of the task delay, energy consumption per task,
and task completion rate.

### 2. [EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning](http://arxiv.org/pdf/2507.10026v1)

Authors: Zhifei Xu, Zhiqing Tang, Jiong Lou, Zhi Yao, Xuan Xie, Tian Wang, Yinglong Wang, Weijia Jia

The growth of Artificial Intelligence (AI) and large language models has
enabled the use of Generative AI (GenAI) in cloud data centers for diverse
AI-Generated Content (AIGC) tasks. Models like Stable Diffusion introduce
unavoidable delays and substantial resource overhead, which are unsuitable for
users at the network edge with high QoS demands. Deploying AIGC services on
edge servers reduces transmission times but often leads to underutilized
resources and fails to optimally balance inference latency and quality. To
address these issues, this paper introduces a QoS-aware
\underline{E}dge-collaborative \underline{A}IGC \underline{T}ask scheduling
(EAT) algorithm. Specifically: 1) We segment AIGC tasks and schedule patches to
various edge servers, formulating it as a gang scheduling problem that balances
inference latency and quality while considering server heterogeneity, such as
differing model distributions and cold start issues. 2) We propose a
reinforcement learning-based EAT algorithm that uses an attention layer to
extract load and task queue information from edge servers and employs a
diffusion-based policy network for scheduling, efficiently enabling model
reuse. 3) We develop an AIGC task scheduling system that uses our EAT algorithm
to divide tasks and distribute them across multiple edge servers for
processing. Experimental results based on our system and large-scale
simulations show that our EAT algorithm can reduce inference latency by up to
56\% compared to baselines. We release our open-source code at
https://github.com/zzf1955/EAT.

### 3. [Past-Future Scheduler for LLM Serving under SLA Guarantees](http://arxiv.org/pdf/2507.10150v1)

Authors: Ruihao Gong, Shihao Bai, Siyu Wu, Yunqian Fan, Zaijun Wang, Xiuhong Li, Hailong Yang, Xianglong Liu

The exploration and application of Large Language Models (LLMs) is thriving.
To reduce deployment costs, continuous batching has become an essential feature
in current service frameworks. The effectiveness of continuous batching relies
on an accurate estimate of the memory requirements of requests. However, due to
the diversity in request output lengths, existing frameworks tend to adopt
aggressive or conservative schedulers, which often result in significant
overestimation or underestimation of memory consumption. Consequently, they
suffer from harmful request evictions or prolonged queuing times, failing to
achieve satisfactory throughput under strict Service Level Agreement (SLA)
guarantees (a.k.a. goodput), across various LLM application scenarios with
differing input-output length distributions. To address this issue, we propose
a novel Past-Future scheduler that precisely estimates the peak memory
resources required by the running batch via considering the historical
distribution of request output lengths and calculating memory occupancy at each
future time point. It adapts to applications with all types of input-output
length distributions, balancing the trade-off between request queuing and
harmful evictions, thereby consistently achieving better goodput. Furthermore,
to validate the effectiveness of the proposed scheduler, we developed a
high-performance LLM serving framework, LightLLM, that implements the
Past-Future scheduler. Compared to existing aggressive or conservative
schedulers, LightLLM demonstrates superior goodput, achieving up to 2-3$\times$
higher goodput than other schedulers under heavy loads. LightLLM is open source
to boost the research in such direction (https://github.com/ModelTC/lightllm).

### 4. [Zorse: Optimizing LLM Training Efficiency on Heterogeneous GPU Clusters](http://arxiv.org/pdf/2507.10392v1)

Authors: Runsheng Benson Guo, Utkarsh Anand, Khuzaima Daudjee, Rathijit Sen

Large language models (LLMs) require vast amounts of GPU compute to train,
but limited availability and high costs of GPUs make homogeneous clusters
impractical for many organizations. Instead, assembling heterogeneous clusters
by pooling together GPUs of different generations allows them to achieve higher
aggregate compute and make use of all available GPUs. However, training on
heterogeneous clusters presents several challenges, including load balancing
across GPUs, optimizing memory usage to accommodate varying memory capacities,
and ensuring communication-efficient training over diverse network
interconnects potentially spanning multiple datacenters. In this paper, we make
the case that efficient training on heterogeneous clusters requires (1) the
integration of pipeline parallelism and data parallelism in a manner that is
both communication- and memory-efficient, and (2) a more adaptable
configuration of pipeline and data parallelism, which includes the capability
to flexibly partition GPUs into asymmetric pipeline parallel stages and to
incorporate heterogeneous GPUs within the same data parallelism group. We
propose Zorse, the first system to unify all these capabilities while
incorporating a planner that automatically configures training strategies for a
given workload. Our evaluation shows that Zorse significantly outperforms
state-of-the-art systems in heterogeneous training scenarios.

### 5. [ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism](http://arxiv.org/pdf/2507.10069v1)

Authors: Zedong Liu, Shenggan Cheng, Guangming Tan, Yang You, Dingwen Tao

Multimodal large language models (MLLMs) extend LLMs to handle images,
videos, and audio by incorporating feature extractors and projection modules.
However, these additional components -- combined with complex inference
pipelines and heterogeneous workloads -- introduce significant inference
overhead. Therefore, efficiently serving MLLMs remains a major challenge.
Current tightly coupled serving architectures struggle to distinguish between
mixed request types or adapt parallelism strategies to different inference
stages, leading to increased time-to-first-token (TTFT) latency and poor
resource utilization. To address this, we propose Elastic Multimodal
Parallelism (EMP), a new serving paradigm that elastically adapts to resource
heterogeneity across request types and inference stages. Building upon EMP, we
develop ElasticMM, an MLLM serving system that (1) separates requests into
independent modality groups with dynamic resource allocation via a
modality-aware load balancer; (2) decouples inference stages and enables
parallelism adjustment and adaptive scaling via elastic partition scheduling;
and (3) improves inference efficiency through unified multimodal prefix caching
and non-blocking encoding. Experiments on diverse real-world datasets show that
ElasticMM outperforms state-of-the-art (SOTA) serving systems, reducing TTFT by
up to 4.2x and achieving 3.2-4.5x higher throughput while meeting service-level
objectives (SLOs).

### 6. [Large-Scale Graph Building in Dynamic Environments: Low Latency and High Quality](http://arxiv.org/pdf/2507.10139v1)

Authors: Filipe Miguel Gonçalves de Almeida, CJ Carey, Hendrik Fichtenberger, Jonathan Halcrow, Silvio Lattanzi, André Linhares, Tao Meng, Ashkan Norouzi-Fard, Nikos Parotsidis, Bryan Perozzi, David Simcha

Learning and constructing large-scale graphs has attracted attention in
recent decades, resulting in a rich literature that introduced various systems,
tools, and algorithms. Grale is one of such tools that is designed for offline
environments and is deployed in more than 50 different industrial settings at
Google. Grale is widely applicable because of its ability to efficiently learn
and construct a graph on datasets with multiple types of features. However, it
is often the case that applications require the underlying data to evolve
continuously and rapidly and the updated graph needs to be available with low
latency. Such setting make the use of Grale prohibitive. While there are
Approximate Nearest Neighbor (ANN) systems that handle dynamic updates with low
latency, they are mostly limited to similarities over a single embedding.
  In this work, we introduce a system that inherits the advantages and the
quality of Grale, and maintains a graph construction in a dynamic setting with
tens of milliseconds of latency per request. We call the system Dynamic Grale
Using ScaNN (Dynamic GUS). Our system has a wide range of applications with
over 10 deployments at Google. One of the applications is in Android Security
and Privacy, where Dynamic Grale Using ScaNN enables capturing harmful
applications 4 times faster, before they can reach users.

### 7. [Cross-Timeslot Optimization for Distributed GPU Inference Using Reinforcement Learning](http://arxiv.org/pdf/2507.10259v1)

Authors: Chengze Du, Zhiwei Yu, Heng Xu, Haojie Wang, Bo liu, Jialong Li

The rapid growth of large language model (LLM) services imposes increasing
demands on distributed GPU inference infrastructure. Most existing scheduling
systems rely on the current system state to make decisions, without considering
how task demand and resource availability evolve over time. This lack of
temporal awareness leads to inefficient GPU utilization, high task migration
overhead, and poor system responsiveness under dynamic workloads. In this work,
we identify the fundamental limitations of these instantaneous-state-only
scheduling approaches and propose Temporal Optimal Resource scheduling via
Two-layer Architecture (TORTA). TORTA introduces a spatiotemporal scheduling
framework that captures both long-term workload patterns and short-term
execution constraints. It adopts a two-layer design: a macro-level scheduler
leverages reinforcement learning and optimal transport to coordinate
inter-region task distribution, while a micro-level allocator refines
task-to-server assignments within each region to reduce latency and switching
costs. Experimental results across multiple network topologies show that TORTA
reduces average inference response time by up to 15\%, improves load balance by
approximately 4-5\%, and cuts total operational cost by 10-20\% compared to
state-of-the-art baseline methods.

### 8. [FalconFS: Distributed File System for Large-Scale Deep Learning Pipeline](http://arxiv.org/pdf/2507.10367v1)

Authors: Jingwei Xu, Junbin Kang, Mingkai Dong, Mingyu Liu, Lu Zhang, Shaohong Guo, Ziyan Qiu, Mingzhen You, Ziyi Tian, Anqi Yu, Tianhong Ding, Xinwei Hu, Haibo Chen

Client-side metadata caching has long been considered an effective method for
accelerating metadata operations in distributed file systems (DFSs). However,
we have found that client-side state (e.g., caching) is not only ineffective
but also consumes valuable memory resources in the deep learning pipelines. We
thus propose FalconFS, a DFS optimized for deep learning pipelines with the
stateless-client architecture. Specifically, instead of performing client-side
path resolution and caching, FalconFS efficiently resolves paths on the server
side using hybrid metadata indexing and lazy namespace replication. FalconFS
also boosts server concurrency with concurrent request merging and provides
easy deployment with VFS shortcut. Evaluations against CephFS and Lustre show
that FalconFS achieves up to 5.72$\times$ throughput for small file read/write
and up to 12.81$\times$ throughput for deep learning model training. FalconFS
has been running in Huawei autonomous driving system's production environment
with 10,000 NPUs for one year.

### 9. [Domain Borders Are There to Be Crossed With Federated Few-Shot Adaptation](http://arxiv.org/pdf/2507.10160v1)

Authors: Manuel Röder, Christoph Raab, Frank-Michael Schleif

Federated Learning has emerged as a leading paradigm for decentralized,
privacy-preserving learning, particularly relevant in the era of interconnected
edge devices equipped with sensors. However, the practical implementation of
Federated Learning faces three primary challenges: the need for human
involvement in costly data labelling processes for target adaptation, covariate
shift in client device data collection due to environmental factors affecting
sensors, leading to discrepancies between source and target samples, and the
impracticality of continuous or regular model updates in resource-constrained
environments due to limited data transmission capabilities and technical
constraints on channel availability and energy efficiency. To tackle these
issues, we expand upon an efficient and scalable Federated Learning framework
tailored for real-world client adaptation in industrial settings. This
framework leverages a pre-trained source model comprising a deep backbone, an
adaptation module, and a classifier running on a powerful server. By freezing
the backbone and classifier during client adaptation on resource-constrained
devices, we allow the domain adaptive linear layer to handle target domain
adaptation, thus minimizing overall computational overhead. Furthermore, this
setup, designated as FedAcross+, is extended to encompass the processing of
streaming data, thereby rendering the solution suitable for non-stationary
environments. Extensive experimental results demonstrate the effectiveness of
FedAcross+ in achieving competitive adaptation on low-end client devices with
limited target samples, successfully addressing the challenge of domain shift.
Moreover, our framework accommodates sporadic model updates within
resource-constrained environments, ensuring practical and seamless deployment.

### 10. [Convergence of Agnostic Federated Averaging](http://arxiv.org/pdf/2507.10325v1)

Authors: Herlock, Rahimi, Dionysis Kalogerias

Federated learning (FL) enables decentralized model training without
centralizing raw data. However, practical FL deployments often face a key
realistic challenge: Clients participate intermittently in server aggregation
and with unknown, possibly biased participation probabilities. Most existing
convergence results either assume full-device participation, or rely on
knowledge of (in fact uniform) client availability distributions -- assumptions
that rarely hold in practice. In this work, we characterize the optimization
problem that consistently adheres to the stochastic dynamics of the well-known
\emph{agnostic Federated Averaging (FedAvg)} algorithm under random (and
variably-sized) client availability, and rigorously establish its convergence
for convex, possibly nonsmooth losses, achieving a standard rate of order
$\mathcal{O}(1/\sqrt{T})$, where $T$ denotes the aggregation horizon. Our
analysis provides the first convergence guarantees for agnostic FedAvg under
general, non-uniform, stochastic client participation, without knowledge of the
participation distribution. We also empirically demonstrate that agnostic
FedAvg in fact outperforms common (and suboptimal) weighted aggregation FedAvg
variants, even with server-side knowledge of participation weights.

### Discrete Mathematics

### 1. [Bicriteria Submodular Maximization](http://arxiv.org/pdf/2507.10248v1)

Authors: Moran Feldman, Alan Kuhnle

Submodular functions and their optimization have found applications in
diverse settings ranging from machine learning and data mining to game theory
and economics. In this work, we consider the constrained maximization of a
submodular function, for which we conduct a principled study of bicriteria
approximation algorithms -- algorithms which can violate the constraint, but
only up to a bounded factor. Bicrteria optimization allows constrained
submodular maximization to capture additional important settings, such as the
well-studied submodular cover problem and optimization under soft constraints.
We provide results that span both multiple types of constraints (cardinality,
knapsack, matroid and convex set) and multiple classes of submodular functions
(monotone, symmetric and general). For many of the cases considered, we provide
optimal results. In other cases, our results improve over the state-of-the-art,
sometimes even over the state-of-the-art for the special case of
single-criterion (standard) optimization. Results of the last kind demonstrate
that relaxing the feasibility constraint may give a perspective about the
problem that is useful even if one only desires feasible solutions.

### 2. [$(Δ-1)$-dicolouring of digraphs](http://arxiv.org/pdf/2507.10266v1)

Authors: Ararat Harutyunyan, Ken-ichi Kawarabayashi, Lucas Picasarri-Arrieta, Gil Puig i Surroca

In 1977, Borodin and Kostochka conjectured that every graph with maximum
degree $\Delta \geq 9$ is $(\Delta-1)$-colourable, unless it contains a clique
of size $\Delta$. In 1999, Reed confirmed the conjecture when $\Delta\geq
10^{14}$.
  We propose different generalisations of this conjecture for digraphs, and
prove the analogue of Reed's result for each of them. The chromatic number and
clique number are replaced respectively by the dichromatic number and the
biclique number of digraphs. If $D$ is a digraph such that
$\min(\tilde{\Delta}(D),\Delta^+(D)) = \Delta \geq 9$, we conjecture that $D$
has dichromatic number at most $\Delta-1$, unless either (i) $D$ contains a
biclique of size $\Delta$, or (ii) $D$ contains a biclique $K$ of size
$\Delta-2$, a directed $3$-cycle $\vec{C_3}$ disjoint from $K$, and all
possible arcs in both directions between $\vec{C_3}$ and $K$. If true, this
implies the conjecture of Borodin and Kostochka. We prove it when $\Delta$ is
large enough, thereby generalising the result of Reed.
  We finally give a sufficient condition for a digraph $D$ to have dichromatic
number at most $\Delta_{\min}(D)-1$, assuming that $\Delta_{\min}(D)$ is large
enough. In particular, this holds when the underlying graph of $D$ has no
clique of size $\Delta_{\min}(D)$, thus yielding a third independent
generalisation of Reed's result. We further give a hardness result witnessing
that our sufficient condition is best possible.
  To obtain these new upper bounds on the dichromatic number, we prove a dense
decomposition lemma for digraphs having large maximum degree, which generalises
to the directed setting the so-called dense decomposition of graphs due to
Molloy and Reed. We believe this may be of independent interest, especially as
a tool in various applications.

### 3. [Colorful Minors](http://arxiv.org/pdf/2507.10467v1)

Authors: Evangelos Protopapas, Dimitrios M. Thilikos, Sebastian Wiederrecht

We introduce the notion of colorful minors, which generalizes the classical
concept of rooted minors in graphs. $q$-colorful graph is defined as a pair
$(G, \chi),$ where $G$ is a graph and $\chi$ assigns to each vertex a (possibly
empty) subset of at most $q$ colors. The colorful minor relation enhances the
classical minor relation by merging color sets at contracted edges and allowing
the removal of colors from vertices. This framework naturally models
algorithmic problems involving graphs with (possibly overlapping) annotated
vertex sets. We develop a structural theory for colorful minors by establishing
several theorems characterizing $\mathcal{H}$-colorful minor-free graphs, where
$\mathcal{H}$ consists either of a clique or a grid with all vertices assigned
all colors, or of grids with colors segregated and ordered on the outer face.
Leveraging our structural insights, we provide a complete classification -
parameterized by the number $q$ of colors - of all colorful graphs that exhibit
the Erd\H{o}s-P\'osa property with respect to colorful minors. On the
algorithmic side, we provide a fixed-parameter tractable algorithm for colorful
minor testing and a variant of the $k$-disjoint paths problem. Together with
the fact that the colorful minor relation forms a well-quasi-order, this
implies that every colorful minor-monotone parameter on colorful graphs admits
a fixed-parameter algorithm. Furthermore, we derive two algorithmic
meta-theorems (AMTs) whose structural conditions are linked to extensions of
treewidth and Hadwiger number on colorful graphs. Our results suggest how known
AMTs can be extended to incorporate not only the structure of the input graph
but also the way the colored vertices are distributed in it.

### 4. [Quantitative central limit theorems for exponential random graphs](http://arxiv.org/pdf/2507.10531v1)

Authors: Vilas Winstein

Ferromagnetic exponential random graph models (ERGMs) are nonlinear
exponential tilts of Erd\H{o}s-R\'enyi models, under which the presence of
certain subgraphs such as triangles may be emphasized. These models are
mixtures of metastable wells which each behave macroscopically like new
Erd\H{o}s-R\'enyi models themselves, exhibiting the same laws of large numbers
for the overall edge count as well as all subgraph counts. However, the
microscopic fluctuations of these quantities remained elusive for some time.
Building on a recent breakthrough by Fang, Liu, Shao and Zhao [FLSZ24] driven
by Stein's method, we prove quantitative central limit theorems (CLTs) for
these quantities and more in metastable wells under ferromagnetic ERGMs. One
main novelty of our results is that they apply also in the supercritical (low
temperature) regime of parameters, which has previously been relatively
unexplored. To accomplish this, we develop a novel probabilistic technique
based on the careful analysis of the evolution of relevant quantities under the
ERGM Glauber dynamics. Our technique allows us to deliver the main input to the
method developed by [FLSZ24], which is the fact that the fluctuations of
subgraph counts are driven by those of the overall edge count. This was first
shown for the triangle count by Sambale and Sinulis [SS20] in the Dobrushin
(very high temperature) regime via functional-analytic methods. We feel our
technique clarifies the underlying mechanisms at play, and it also supplies
improved bounds on the Wasserstein and Kolmogorov distances between the
observables at hand and the limiting Gaussians, as compared to the results of
[FLSZ24] in the subcritical (high temperature) regime beyond the Dobrushin
regime. Moreover, our technique is flexible enough to also yield quantitative
CLTs for vertex degrees and local subgraph counts, which have not appeared
before in any parameter regime.

### Data Structures and Algorithms

### 1. [Improved bicriteria approximation for $k$-edge-connectivity](http://arxiv.org/pdf/2507.10125v1)

Authors: Zeev Nutov

In the $k$-Edge Connected Spanning Subgraph ($k$-ECSS) problem we are given a
(multi-)graph $G=(V,E)$ with edge costs and an integer $k$, and seek a min-cost
$k$-edge-connected spanning subgraph of $G$. The problem admits a
$2$-approximation algorithm and no better approximation ratio is
known.Hershkowitz, Klein, and Zenklusen [STOC 24] gave a bicriteria
$(1,k-10)$-approximation algorithm that computes a $(k-10)$-edge-connected
spanning subgraph of cost at most the optimal value of a standard Cut-LP for
$k$-ECSS. This LP bicriteria approximation was recently improved by Cohen and
Nutov [ESA 25] to $(1,k-4)$, where also was given a bicriteria approximation
$(3/2,k-2)$. In this paper we improve the bicriteria approximation to $(1,k-2)$
for $k$ even and to $\left(1-\frac{1}{k},k-3\right)$ for $k$ is odd, and also
give another bicriteria approximation $(3/2,k-1)$.
  The $k$-Edge-Connected Spanning Multi-subgraph ($k$-ECSM) problem is almost
the same as $k$-ECSS, except that any edge can be selected multiple times at
the same cost. The previous best approximation ratio for $k$-ECSM was $1+4/k$.
Our result improves this to $1+\frac{2}{k}$ for $k$ even and to $1+\frac{3}{k}$
for $k$ odd, where for $k$ odd the computed subgraph is in fact
$(k+1)$-edge-connected.

### 2. [Approximating Maximum Cut on Interval Graphs and Split Graphs beyond Goemans-Williamson](http://arxiv.org/pdf/2507.10436v1)

Authors: Jungho Ahn, Ian DeHaan, Eun Jung Kim, Euiwoong Lee

We present a polynomial-time $(\alpha_{GW} + \varepsilon)$-approximation
algorithm for the Maximum Cut problem on interval graphs and split graphs,
where $\alpha_{GW} \approx 0.878$ is the approximation guarantee of the
Goemans-Williamson algorithm and $\varepsilon > 10^{-34}$ is a fixed constant.
To attain this, we give an improved analysis of a slight modification of the
Goemans-Williamson algorithm for graphs in which triangles can be packed into a
constant fraction of their edges. We then pair this analysis with structural
results showing that both interval graphs and split graphs either have such a
triangle packing or have maximum cut close to their number of edges. We also
show that, subject to the Small Set Expansion Hypothesis, there exists a
constant $c > 0$ such that there is no polyomial-time $(1 - c)$-approximation
for Maximum Cut on split graphs.

### 3. [Bicriteria Submodular Maximization](http://arxiv.org/pdf/2507.10248v1)

Authors: Moran Feldman, Alan Kuhnle

Submodular functions and their optimization have found applications in
diverse settings ranging from machine learning and data mining to game theory
and economics. In this work, we consider the constrained maximization of a
submodular function, for which we conduct a principled study of bicriteria
approximation algorithms -- algorithms which can violate the constraint, but
only up to a bounded factor. Bicrteria optimization allows constrained
submodular maximization to capture additional important settings, such as the
well-studied submodular cover problem and optimization under soft constraints.
We provide results that span both multiple types of constraints (cardinality,
knapsack, matroid and convex set) and multiple classes of submodular functions
(monotone, symmetric and general). For many of the cases considered, we provide
optimal results. In other cases, our results improve over the state-of-the-art,
sometimes even over the state-of-the-art for the special case of
single-criterion (standard) optimization. Results of the last kind demonstrate
that relaxing the feasibility constraint may give a perspective about the
problem that is useful even if one only desires feasible solutions.

### 4. [Average Sensitivity of Hierarchical $k$-Median Clustering](http://arxiv.org/pdf/2507.10296v1)

Authors: Shijie Li, Weiqiang He, Ruobing Bai, Pan Peng

Hierarchical clustering is a widely used method for unsupervised learning
with numerous applications. However, in the application of modern algorithms,
the datasets studied are usually large and dynamic. If the hierarchical
clustering is sensitive to small perturbations of the dataset, the usability of
the algorithm will be greatly reduced. In this paper, we focus on the
hierarchical $k$ -median clustering problem, which bridges hierarchical and
centroid-based clustering while offering theoretical appeal, practical utility,
and improved interpretability. We analyze the average sensitivity of algorithms
for this problem by measuring the expected change in the output when a random
data point is deleted. We propose an efficient algorithm for hierarchical
$k$-median clustering and theoretically prove its low average sensitivity and
high clustering quality. Additionally, we show that single linkage clustering
and a deterministic variant of the CLNSS algorithm exhibit high average
sensitivity, making them less stable. Finally, we validate the robustness and
effectiveness of our algorithm through experiments.

### 5. [Covering a Few Submodular Constraints and Applications](http://arxiv.org/pdf/2507.09879v1)

Authors: Tanvi Bajpai, Chandra Chekuri, Pooja Kulkarni

We consider the problem of covering multiple submodular constraints. Given a
finite ground set $N$, a cost function $c: N \rightarrow \mathbb{R}_+$, $r$
monotone submodular functions $f_1,f_2,\ldots,f_r$ over $N$ and requirements
$b_1,b_2,\ldots,b_r$ the goal is to find a minimum cost subset $S \subseteq N$
such that $f_i(S) \ge b_i$ for $1 \le i \le r$. When $r=1$ this is the
well-known Submodular Set Cover problem. Previous work
\cite{chekuri2022covering} considered the setting when $r$ is large and
developed bi-criteria approximation algorithms, and approximation algorithms
for the important special case when each $f_i$ is a weighted coverage function.
These are fairly general models and capture several concrete and interesting
problems as special cases. The approximation ratios for these problem are at
least $\Omega(\log r)$ which is unavoidable when $r$ is part of the input. In
this paper, motivated by some recent applications, we consider the problem when
$r$ is a \emph{fixed constant} and obtain two main results. For covering
multiple submodular constraints we obtain a randomized bi-criteria
approximation algorithm that for any given integer $\alpha \ge 1$ outputs a set
$S$ such that $f_i(S) \ge$ $(1-1/e^\alpha -\epsilon)b_i$ for each $i \in [r]$
and $\mathbb{E}[c(S)] \le (1+\epsilon)\alpha \cdot \sf{OPT}$. Second, when the
$f_i$ are weighted coverage functions from a deletion-closed set system we
obtain a $(1+\epsilon)$ $(\frac{e}{e-1})$ $(1+\beta)$-approximation where
$\beta$ is the approximation ratio for the underlying set cover instances via
the natural LP. These results show that one can obtain nearly as good an
approximation for any fixed $r$ as what one would achieve for $r=1$. We mention
some applications that follow easily from these general results and anticipate
more in the future.

### 6. [Computing the probability of intersection](http://arxiv.org/pdf/2507.10329v1)

Authors: Alexander Barvinok

Let $\Omega_1, \ldots, \Omega_m$ be probability spaces, let $\Omega=\Omega_1
\times \cdots \times \Omega_m$ be their product and let $A_1, \ldots, A_n
\subset \Omega$ be events. Suppose that each event $A_i$ depends on $r_i$
coordinates of a point $x \in \Omega$, $x=\left(\xi_1, \ldots, \xi_m\right)$,
and that for each event $A_i$ there are $\Delta_i$ of other events $A_j$ that
depend on some of the coordinates that $A_i$ depends on. Let $\Delta=\max\{5,\
\Delta_i: i=1, \ldots, n\}$ and let $\mu_i=\min\{r_i,\ \Delta_i+1\}$ for $i=1,
\ldots, n$. We prove that if $P(A_i) < (3\Delta)^{-3\mu_i}$ for all $I$, then
for any $0 < \epsilon < 1$, the probability $P\left( \bigcap_{i=1}^n
\overline{A}_i\right)$ of the intersection of the complements of all $A_i$ can
be computed within relative error $\epsilon$ in polynomial time from the
probabilities $P\left(A_{i_1} \cap \ldots \cap A_{i_k}\right)$ of $k$-wise
intersections of the events $A_i$ for $k = e^{O(\Delta)} \ln (n/\epsilon)$.

### 7. [Colorful Minors](http://arxiv.org/pdf/2507.10467v1)

Authors: Evangelos Protopapas, Dimitrios M. Thilikos, Sebastian Wiederrecht

We introduce the notion of colorful minors, which generalizes the classical
concept of rooted minors in graphs. $q$-colorful graph is defined as a pair
$(G, \chi),$ where $G$ is a graph and $\chi$ assigns to each vertex a (possibly
empty) subset of at most $q$ colors. The colorful minor relation enhances the
classical minor relation by merging color sets at contracted edges and allowing
the removal of colors from vertices. This framework naturally models
algorithmic problems involving graphs with (possibly overlapping) annotated
vertex sets. We develop a structural theory for colorful minors by establishing
several theorems characterizing $\mathcal{H}$-colorful minor-free graphs, where
$\mathcal{H}$ consists either of a clique or a grid with all vertices assigned
all colors, or of grids with colors segregated and ordered on the outer face.
Leveraging our structural insights, we provide a complete classification -
parameterized by the number $q$ of colors - of all colorful graphs that exhibit
the Erd\H{o}s-P\'osa property with respect to colorful minors. On the
algorithmic side, we provide a fixed-parameter tractable algorithm for colorful
minor testing and a variant of the $k$-disjoint paths problem. Together with
the fact that the colorful minor relation forms a well-quasi-order, this
implies that every colorful minor-monotone parameter on colorful graphs admits
a fixed-parameter algorithm. Furthermore, we derive two algorithmic
meta-theorems (AMTs) whose structural conditions are linked to extensions of
treewidth and Hadwiger number on colorful graphs. Our results suggest how known
AMTs can be extended to incorporate not only the structure of the input graph
but also the way the colored vertices are distributed in it.

### Emerging Technologies

### 1. [Solving the compute crisis with physics-based ASICs](http://arxiv.org/pdf/2507.10463v1)

Authors: Maxwell Aifer, Zach Belateche, Suraj Bramhavar, Kerem Y. Camsari, Patrick J. Coles, Gavin Crooks, Douglas J. Durian, Andrea J. Liu, Anastasia Marchenkova, Antonio J. Martinez, Peter L. McMahon, Faris Sbahi, Benjamin Weiner, Logan G. Wright

Escalating artificial intelligence (AI) demands expose a critical "compute
crisis" characterized by unsustainable energy consumption, prohibitive training
costs, and the approaching limits of conventional CMOS scaling. Physics-based
Application-Specific Integrated Circuits (ASICs) present a transformative
paradigm by directly harnessing intrinsic physical dynamics for computation
rather than expending resources to enforce idealized digital abstractions. By
relaxing the constraints needed for traditional ASICs, like enforced
statelessness, unidirectionality, determinism, and synchronization, these
devices aim to operate as exact realizations of physical processes, offering
substantial gains in energy efficiency and computational throughput. This
approach enables novel co-design strategies, aligning algorithmic requirements
with the inherent computational primitives of physical systems. Physics-based
ASICs could accelerate critical AI applications like diffusion models,
sampling, optimization, and neural network inference as well as traditional
computational workloads like scientific simulation of materials and molecules.
Ultimately, this vision points towards a future of heterogeneous,
highly-specialized computing platforms capable of overcoming current scaling
bottlenecks and unlocking new frontiers in computational power and efficiency.

### 2. [SentiDrop: A Multi Modal Machine Learning model for Predicting Dropout in Distance Learning](http://arxiv.org/pdf/2507.10421v1)

Authors: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui

School dropout is a serious problem in distance learning, where early
detection is crucial for effective intervention and student perseverance.
Predicting student dropout using available educational data is a widely
researched topic in learning analytics. Our partner's distance learning
platform highlights the importance of integrating diverse data sources,
including socio-demographic data, behavioral data, and sentiment analysis, to
accurately predict dropout risks. In this paper, we introduce a novel model
that combines sentiment analysis of student comments using the Bidirectional
Encoder Representations from Transformers (BERT) model with socio-demographic
and behavioral data analyzed through Extreme Gradient Boosting (XGBoost). We
fine-tuned BERT on student comments to capture nuanced sentiments, which were
then merged with key features selected using feature importance techniques in
XGBoost. Our model was tested on unseen data from the next academic year,
achieving an accuracy of 84\%, compared to 82\% for the baseline model.
Additionally, the model demonstrated superior performance in other metrics,
such as precision and F1-score. The proposed method could be a vital tool in
developing personalized strategies to reduce dropout rates and encourage
student perseverance

### Graphics

### 1. [ScaffoldAvatar: High-Fidelity Gaussian Avatars with Patch Expressions](http://arxiv.org/pdf/2507.10542v1)

Authors: Shivangi Aneja, Sebastian Weiss, Irene Baeza, Prashanth Chandran, Gaspard Zoss, Matthias Nießner, Derek Bradley

Generating high-fidelity real-time animated sequences of photorealistic 3D
head avatars is important for many graphics applications, including immersive
telepresence and movies. This is a challenging problem particularly when
rendering digital avatar close-ups for showing character's facial microfeatures
and expressions. To capture the expressive, detailed nature of human heads,
including skin furrowing and finer-scale facial movements, we propose to couple
locally-defined facial expressions with 3D Gaussian splatting to enable
creating ultra-high fidelity, expressive and photorealistic 3D head avatars. In
contrast to previous works that operate on a global expression space, we
condition our avatar's dynamics on patch-based local expression features and
synthesize 3D Gaussians at a patch level. In particular, we leverage a
patch-based geometric 3D face model to extract patch expressions and learn how
to translate these into local dynamic skin appearance and motion by coupling
the patches with anchor points of Scaffold-GS, a recent hierarchical scene
representation. These anchors are then used to synthesize 3D Gaussians
on-the-fly, conditioned by patch-expressions and viewing direction. We employ
color-based densification and progressive training to obtain high-quality
results and faster convergence for high resolution 3K training images. By
leveraging patch-level expressions, ScaffoldAvatar consistently achieves
state-of-the-art performance with visually natural motion, while encompassing
diverse facial expressions and styles in real time.

### Computer Science and Game Theory

### 1. [Tie-breaking Agnostic Lower Bound for Fictitious Play](http://arxiv.org/pdf/2507.09902v1)

Authors: Yuanhao Wang

Fictitious play (FP) is a natural learning dynamic in two-player zero-sum
games. Samuel Karlin conjectured in 1959 that FP converges at a rate of
$O(t^{-1/2})$ to Nash equilibrium, where $t$ is the number of steps played.
However, Daskalakis and Pan disproved the stronger form of this conjecture in
2014, where \emph{adversarial} tie-breaking is allowed.
  This paper disproves Karlin's conjecture in its weaker form. In particular,
there exists a 10-by-10 zero-sum matrix game, in which FP converges at a rate
of $\Omega(t^{-1/3})$, and no ties occur except for the first step.

### 2. [The Value Problem for Weighted Timed Games with Two Clocks is Undecidable](http://arxiv.org/pdf/2507.10550v1)

Authors: Quentin Guilmant, Joël Ouaknine, Isa Vialard

The Value Problem for weighted timed games (WTGs) consists in determining,
given a two-player weighted timed game with a reachability objective and a
rational threshold, whether or not the value of the game exceeds the threshold.
This problem was shown to be undecidable some ten years ago for WTGs making use
of at least three clocks, and is known to be decidable for single-clock WTGs.
In this paper, we establish undecidability for two-clock WTGs making use of
non-negative weights, even in a time-bounded setting, closing the last
remaining major gap in our algorithmic understanding of WTGs.

### 3. [Generalized Quantal Response Equilibrium: Existence and Efficient Learning](http://arxiv.org/pdf/2507.09928v1)

Authors: Apurv Shukla, Vijay Subramanian, Andy Zhao, Rahul Jain

We introduce a new solution concept for bounded rational agents in finite
normal-form general-sum games called Generalized Quantal Response Equilibrium
(GQRE) which generalizes Quantal Response
Equilibrium~\citep{mckelvey1995quantal}. In our setup, each player maximizes a
smooth, regularized expected utility of the mixed profiles used, reflecting
bounded rationality that subsumes stochastic choice. After establishing
existence under mild conditions, we present computationally efficient no-regret
independent learning via smoothened versions of the Frank-Wolfe algorithm. Our
algorithm uses noisy but correlated gradient estimates generated via a
simulation oracle that reports on repeated plays of the game. We analyze
convergence properties of our algorithm under assumptions that ensure
uniqueness of equilibrium, using a class of gap functions that generalize the
Nash gap. We end by demonstrating the effectiveness of our method on a set of
complex general-sum games such as high-rank two-player games, large action
two-player games, and known examples of difficult multi-player games.

### 4. [Covering a Few Submodular Constraints and Applications](http://arxiv.org/pdf/2507.09879v1)

Authors: Tanvi Bajpai, Chandra Chekuri, Pooja Kulkarni

We consider the problem of covering multiple submodular constraints. Given a
finite ground set $N$, a cost function $c: N \rightarrow \mathbb{R}_+$, $r$
monotone submodular functions $f_1,f_2,\ldots,f_r$ over $N$ and requirements
$b_1,b_2,\ldots,b_r$ the goal is to find a minimum cost subset $S \subseteq N$
such that $f_i(S) \ge b_i$ for $1 \le i \le r$. When $r=1$ this is the
well-known Submodular Set Cover problem. Previous work
\cite{chekuri2022covering} considered the setting when $r$ is large and
developed bi-criteria approximation algorithms, and approximation algorithms
for the important special case when each $f_i$ is a weighted coverage function.
These are fairly general models and capture several concrete and interesting
problems as special cases. The approximation ratios for these problem are at
least $\Omega(\log r)$ which is unavoidable when $r$ is part of the input. In
this paper, motivated by some recent applications, we consider the problem when
$r$ is a \emph{fixed constant} and obtain two main results. For covering
multiple submodular constraints we obtain a randomized bi-criteria
approximation algorithm that for any given integer $\alpha \ge 1$ outputs a set
$S$ such that $f_i(S) \ge$ $(1-1/e^\alpha -\epsilon)b_i$ for each $i \in [r]$
and $\mathbb{E}[c(S)] \le (1+\epsilon)\alpha \cdot \sf{OPT}$. Second, when the
$f_i$ are weighted coverage functions from a deletion-closed set system we
obtain a $(1+\epsilon)$ $(\frac{e}{e-1})$ $(1+\beta)$-approximation where
$\beta$ is the approximation ratio for the underlying set cover instances via
the natural LP. These results show that one can obtain nearly as good an
approximation for any fixed $r$ as what one would achieve for $r=1$. We mention
some applications that follow easily from these general results and anticipate
more in the future.

### 5. [A New Incentive Model For Content Trust](http://arxiv.org/pdf/2507.09972v1)

Authors: Lucas Barbosa, Sam Kirshner, Rob Kopel, Eric Tze Kuan Lim, Tom Pagram

This paper outlines an incentive-driven and decentralized approach to
verifying the veracity of digital content at scale. Widespread misinformation,
an explosion in AI-generated content and reduced reliance on traditional news
sources demands a new approach for content authenticity and truth-seeking that
is fit for a modern, digital world. By using smart contracts and digital
identity to incorporate 'trust' into the reward function for published content,
not just engagement, we believe that it could be possible to foster a
self-propelling paradigm shift to combat misinformation through a
community-based governance model. The approach described in this paper requires
that content creators stake financial collateral on factual claims for an
impartial jury to vet with a financial reward for contribution. We hypothesize
that with the right financial and social incentive model users will be
motivated to participate in crowdsourced fact-checking and content creators
will place more care in their attestations. This is an exploratory paper and
there are a number of open issues and questions that warrant further analysis
and exploration.

### 6. [A Coincidence of Wants Mechanism for Swap Trade Execution in Decentralized Exchanges](http://arxiv.org/pdf/2507.10149v1)

Authors: Abhimanyu Nag, Madhur Prabhakar, Tanuj Behl

We propose a mathematically rigorous framework for identifying and completing
Coincidence of Wants (CoW) cycles in decentralized exchange (DEX) aggregators.
Unlike existing auction based systems such as CoWSwap, our approach introduces
an asset matrix formulation that not only verifies feasibility using oracle
prices and formal conservation laws but also completes partial CoW cycles of
swap orders that are discovered using graph traversal and are settled using
imbalance correction. We define bridging orders and show that the resulting
execution is slippage free and capital preserving for LPs. Applied to real
world Arbitrum swap data, our algorithm demonstrates efficient discovery of CoW
cycles and supports the insertion of synthetic orders for atomic cycle closure.
This work can be thought of as the detailing of a potential delta-neutral
strategy by liquidity providing market makers: a structured CoW cycle
execution.

### Human-Computer Interaction

### 1. [Volume-Based Space-Time Cube for Large-Scale Continuous Spatial Time Series](http://arxiv.org/pdf/2507.09917v1)

Authors: Zikun Deng, Jiabao Huang, Chenxi Ruan, Jialing Li, Shaowu Gao, Yi Cai

Spatial time series visualization offers scientific research pathways and
analytical decision-making tools across various spatiotemporal domains. Despite
many advanced methodologies, the seamless integration of temporal and spatial
information remains a challenge. The space-time cube (STC) stands out as a
promising approach for the synergistic presentation of spatial and temporal
information, with successful applications across various spatiotemporal
datasets. However, the STC is plagued by well-known issues such as visual
occlusion and depth ambiguity, which are further exacerbated when dealing with
large-scale spatial time series data. In this study, we introduce a novel
technical framework termed VolumeSTCube, designed for continuous spatiotemporal
phenomena. It first leverages the concept of the STC to transform discretely
distributed spatial time series data into continuously volumetric data.
Subsequently, volume rendering and surface rendering techniques are employed to
visualize the transformed volumetric data. Volume rendering is utilized to
mitigate visual occlusion, while surface rendering provides pattern details by
enhanced lighting information. Lastly, we design interactions to facilitate the
exploration and analysis from temporal, spatial, and spatiotemporal
perspectives. VolumeSTCube is evaluated through a computational experiment, a
real-world case study with one expert, and a controlled user study with twelve
non-experts, compared against a baseline from prior work, showing its
superiority and effectiveness in largescale spatial time series analysis.

### 2. [Branch Explorer: Leveraging Branching Narratives to Support Interactive 360° Video Viewing for Blind and Low Vision Users](http://arxiv.org/pdf/2507.09959v1)

Authors: Shuchang Xu, Xiaofu Jin, Wenshuo Zhang, Huamin Qu, Yukang Yan

360{\deg} videos enable users to freely choose their viewing paths, but blind
and low vision (BLV) users are often excluded from this interactive experience.
To bridge this gap, we present Branch Explorer, a system that transforms
360{\deg} videos into branching narratives -- stories that dynamically unfold
based on viewer choices -- to support interactive viewing for BLV audiences.
Our formative study identified three key considerations for accessible
branching narratives: providing diverse branch options, ensuring coherent story
progression, and enabling immersive navigation among branches. To address these
needs, Branch Explorer employs a multi-modal machine learning pipeline to
generate diverse narrative paths, allowing users to flexibly make choices at
detected branching points and seamlessly engage with each storyline through
immersive audio guidance. Evaluation with 12 BLV viewers showed that Branch
Explorer significantly enhanced user agency and engagement in 360{\deg} video
viewing. Users also developed personalized strategies for exploring 360{\deg}
content. We further highlight implications for supporting accessible
exploration of videos and virtual environments.

### 3. [Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles](http://arxiv.org/pdf/2507.10024v1)

Authors: Shaolun Ruan, Rui Sheng, Xiaolin Wen, Jiachen Wang, Tianyi Zhang, Yong Wang, Tim Dwyer, Jiannan Li

Design studies aim to create visualization solutions for real-world problems
of different application domains. Recently, the emergence of large language
models (LLMs) has introduced new opportunities to enhance the design study
process, providing capabilities such as creative problem-solving, data
handling, and insightful analysis. However, despite their growing popularity,
there remains a lack of systematic understanding of how LLMs can effectively
assist researchers in visualization-specific design studies. In this paper, we
conducted a multi-stage qualitative study to fill this gap, involving 30 design
study researchers from diverse backgrounds and expertise levels. Through
in-depth interviews and carefully-designed questionnaires, we investigated
strategies for utilizing LLMs, the challenges encountered, and the practices
used to overcome them. We further compiled and summarized the roles that LLMs
can play across different stages of the design study process. Our findings
highlight practical implications to inform visualization practitioners, and
provide a framework for leveraging LLMs to enhance the design study process in
visualization research.

### 4. [XROps: A Visual Workflow Management System for Dynamic Immersive Analytics](http://arxiv.org/pdf/2507.10043v1)

Authors: Suemin Jeon, JunYoung Choi, Haejin Jeong, Won-Ki Jeong

Immersive analytics is gaining attention across multiple domains due to its
capability to facilitate intuitive data analysis in expansive environments
through user interaction with data. However, creating immersive analytics
systems for specific tasks is challenging due to the need for programming
expertise and significant development effort. Despite the introduction of
various immersive visualization authoring toolkits, domain experts still face
hurdles in adopting immersive analytics into their workflow, particularly when
faced with dynamically changing tasks and data in real time. To lower such
technical barriers, we introduce XROps, a web-based authoring system that
allows users to create immersive analytics applications through interactive
visual programming, without the need for low-level scripting or coding. XROps
enables dynamic immersive analytics authoring by allowing users to modify each
step of the data visualization process with immediate feedback, enabling them
to build visualizations on-the-fly and adapt to changing environments. It also
supports the integration and visualization of real-time sensor data from XR
devices, a key feature of immersive analytics, facilitating the creation of
various analysis scenarios. We evaluated the usability of XROps through a user
study and demonstrate its efficacy and usefulness in several example scenarios.
We have released a web platform (https://vience.io/xrops) to demonstrate
various examples to supplement our findings.

### 5. [MEDebiaser: A Human-AI Feedback System for Mitigating Bias in Multi-label Medical Image Classification](http://arxiv.org/pdf/2507.10044v1)

Authors: Shaohan Shi, Yuheng Shao, Haoran Jiang, Yunjie Yao, Zhijun Zhang, Xu Ding, Quan Li

Medical images often contain multiple labels with imbalanced distributions
and co-occurrence, leading to bias in multi-label medical image classification.
Close collaboration between medical professionals and machine learning
practitioners has significantly advanced medical image analysis. However,
traditional collaboration modes struggle to facilitate effective feedback
between physicians and AI models, as integrating medical expertise into the
training process via engineers can be time-consuming and labor-intensive. To
bridge this gap, we introduce MEDebiaser, an interactive system enabling
physicians to directly refine AI models using local explanations. By combining
prediction with attention loss functions and employing a customized ranking
strategy to alleviate scalability, MEDebiaser allows physicians to mitigate
biases without technical expertise, reducing reliance on engineers, and thus
enhancing more direct human-AI feedback. Our mechanism and user studies
demonstrate that it effectively reduces biases, improves usability, and
enhances collaboration efficiency, providing a practical solution for
integrating medical expertise into AI-driven healthcare.

### 6. [When Familiarity Remains: Procedural Memory, Symbolic Anchors, and Digital Engagement in Dementia Care](http://arxiv.org/pdf/2507.10102v1)

Authors: Jeongone Seo, Kyung-zoon Hong, Sol Baik

INTRODUCTION: Older adults with early-stage dementia often retain procedural
memory, enabling continued use of familiar technologies. Additionally, symbolic
anchors such as photos or personalized content may serve as memory cues to
reinforce digital engagement. This study explores how these mechanisms support
technology use in dementia care within the South Korean context.
  METHODS: We conducted in-depth interviews with 11 professional caregivers of
community-dwelling older adults with cognitive decline. Grounded theory methods
guided the analysis, using iterative coding and constant comparison to identify
emergent themes.
  RESULTS: Caregivers reported that familiar digital routines (e.g., taking
photos) persisted through procedural memory. Symbolic anchors such as family
photos or recognizable icons enhanced interaction and emotional engagement.
However, unfamiliar or anthropomorphic technologies often triggered fear or
symbolic resistance.
  DISCUSSION: Findings highlight the dual role of procedural memory and
symbolic anchors in sustaining digital engagement. Designing culturally
responsive and cognitively accessible technologies may enhance autonomy and
well-being in dementia care.
  Keywords: procedural memory, symbolic anchors, dementia care, digital
engagement, older adults, cultural adaptation, caregiving technologies

### 7. [VIP-Sim: A User-Centered Approach to Vision Impairment Simulation for Accessible Design](http://arxiv.org/pdf/2507.10479v1)

Authors: Max Rädler, Mark Colley, Enrico Rukzio

People with vision impairments (VIPs) often rely on their remaining vision
when interacting with user interfaces. Simulating visual impairments is an
effective tool for designers, fostering awareness of the challenges faced by
VIPs. While previous research has introduced various vision impairment
simulators, none have yet been developed with the direct involvement of VIPs or
thoroughly evaluated from their perspective. To address this gap, we developed
VIP-Sim. This symptom-based vision simulator was created through a
participatory design process tailored explicitly for this purpose, involving
N=7 VIPs. 21 symptoms, like field loss or light sensitivity, can be overlaid on
desktop design tools. Most participants felt VIP-Sim could replicate their
symptoms. VIP-Sim was received positively, but concerns about exclusion in
design and comprehensiveness of the simulation remain, mainly whether it
represents the experiences of other VIPs.

### 8. [ReDemon UI: Reactive Synthesis by Demonstration for Web UI](http://arxiv.org/pdf/2507.10099v1)

Authors: Jay Lee, Gyuhyeok Oh, Joongwon Ahn, Xiaokang Qiu

ReDemon UI synthesizes React applications from user demonstrations, enabling
designers and non-expert programmers to create UIs that integrate with standard
UI prototyping workflows. Users provide a static mockup sketch with event
handler holes and demonstrate desired runtime behaviors by interacting with the
rendered mockup and editing the sketch. ReDemon UI identifies reactive data and
synthesizes a React program with correct state update logic. We utilize
enumerative synthesis for simple UIs and LLMs for more complex UIs.

### 9. [Riding the Carousel: The First Extensive Eye Tracking Analysis of Browsing Behavior in Carousel Recommenders](http://arxiv.org/pdf/2507.10135v1)

Authors: Santiago de Leon-Martinez, Robert Moro, Branislav Kveton, Maria Bielikova

Carousels have become the de-facto interface in online services. However,
there is a lack of research in carousels, particularly examining how
recommender systems may be designed differently than the traditional
single-list interfaces. One of the key elements for understanding how to design
a system for a particular interface is understanding how users browse. For
carousels, users may browse in a number of different ways due to the added
complexity of multiple topic defined-lists and swiping to see more items.
  Eye tracking is the key to understanding user behavior by providing valuable,
direct information on how users see and navigate. In this work, we provide the
first extensive analysis of the eye tracking behavior in carousel recommenders
under the free-browsing setting. To understand how users browse, we examine the
following research questions : 1) where do users start browsing, 2) how do
users transition from item to item within the same carousel and across
carousels, and 3) how does genre preference impact transitions?
  This work addresses a gap in the field and provides the first extensive
empirical results of eye tracked browsing behavior in carousels for improving
recommenders. Taking into account the insights learned from the above
questions, our final contribution is to provide suggestions to help carousel
recommender system designers optimize their systems for user browsing behavior.
The most important suggestion being to reorder the ranked item positions to
account for browsing after swiping.These contributions aim not only to help
improve current systems, but also to encourage and allow the design of new user
models, systems, and metrics that are better suited to the complexity of
carousel interfaces.

### 10. [Survey for Categorising Explainable AI Studies Using Data Analysis Task Frameworks](http://arxiv.org/pdf/2507.10208v1)

Authors: Hamzah Ziadeh, Hendrik Knoche

Research into explainable artificial intelligence (XAI) for data analysis
tasks suffer from a large number of contradictions and lack of concrete design
recommendations stemming from gaps in understanding the tasks that require AI
assistance. In this paper, we drew on multiple fields such as visual analytics,
cognition, and dashboard design to propose a method for categorising and
comparing XAI studies under three dimensions: what, why, and who. We identified
the main problems as: inadequate descriptions of tasks, context-free studies,
and insufficient testing with target users. We propose that studies should
specifically report on their users' domain, AI, and data analysis expertise to
illustrate the generalisability of their findings. We also propose study
guidelines for designing and reporting XAI tasks to improve the XAI community's
ability to parse the rapidly growing field. We hope that our contribution can
help researchers and designers better identify which studies are most relevant
to their work, what gaps exist in the research, and how to handle contradictory
results regarding XAI design.

### Information Retrieval

### 1. [Non-parametric Graph Convolution for Re-ranking in Recommendation Systems](http://arxiv.org/pdf/2507.09969v1)

Authors: Zhongyu Ouyang, Mingxuan Ju, Soroush Vosoughi, Yanfang Ye

Graph knowledge has been proven effective in enhancing item rankings in
recommender systems (RecSys), particularly during the retrieval stage. However,
its application in the ranking stage, especially when richer contextual
information in user-item interactions is available, remains underexplored. A
major challenge lies in the substantial computational cost associated with
repeatedly retrieving neighborhood information from billions of items stored in
distributed systems. This resource-intensive requirement makes it difficult to
scale graph-based methods in practical RecSys. To bridge this gap, we first
demonstrate that incorporating graphs in the ranking stage improves ranking
qualities. Notably, while the improvement is evident, we show that the
substantial computational overheads entailed by graphs are prohibitively
expensive for real-world recommendations. In light of this, we propose a
non-parametric strategy that utilizes graph convolution for re-ranking only
during test time. Our strategy circumvents the notorious computational
overheads from graph convolution during training, and utilizes structural
knowledge hidden in graphs on-the-fly during testing. It can be used as a
plug-and-play module and easily employed to enhance the ranking ability of
various ranking layers of a real-world RecSys with significantly reduced
computational overhead. Through comprehensive experiments across four benchmark
datasets with varying levels of sparsity, we demonstrate that our strategy
yields noticeable improvements (i.e., 8.1% on average) during testing time with
little to no additional computational overheads (i.e., 0.5 on average). Code:
https://github.com/zyouyang/RecSys2025_NonParamGC.git

### 2. [SLIF-MR: Self-loop Iterative Fusion of Heterogeneous Auxiliary Information for Multimodal Recommendation](http://arxiv.org/pdf/2507.09998v1)

Authors: Jie Guo, Jiahao Jiang, Ziyuan Guo, Bin Song, Yue Sun

Knowledge graphs (KGs) and multimodal item information, which respectively
capture relational and attribute features, play a crucial role in improving
recommender system accuracy. Recent studies have attempted to integrate them
via multimodal knowledge graphs (MKGs) to further enhance recommendation
performance. However, existing methods typically freeze the MKG structure
during training, which limits the full integration of structural information
from heterogeneous graphs (e.g., KG and user-item interaction graph), and
results in sub-optimal performance. To address this challenge, we propose a
novel framework, termed Self-loop Iterative Fusion of Heterogeneous Auxiliary
Information for Multimodal Recommendation (SLIF-MR), which leverages item
representations from previous training epoch as feedback signals to dynamically
optimize the heterogeneous graph structures composed of KG, multimodal item
feature graph, and user-item interaction graph. Through this iterative fusion
mechanism, both user and item representations are refined, thus improving the
final recommendation performance. Specifically, based on the feedback item
representations, SLIF-MR constructs an item-item correlation graph, then
integrated into the establishment process of heterogeneous graphs as additional
new structural information in a self-loop manner. Consequently, the internal
structures of heterogeneous graphs are updated with the feedback item
representations during training. Moreover, a semantic consistency learning
strategy is proposed to align heterogeneous item representations across
modalities. The experimental results show that SLIF-MR significantly
outperforms existing methods, particularly in terms of accuracy and robustness.

### 3. [User Long-Term Multi-Interest Retrieval Model for Recommendation](http://arxiv.org/pdf/2507.10097v1)

Authors: Yue Meng, Cheng Guo, Xiaohui Hu, Honghu Deng, Yi Cao, Tong Liu, Bo Zheng

User behavior sequence modeling, which captures user interest from rich
historical interactions, is pivotal for industrial recommendation systems.
Despite breakthroughs in ranking-stage models capable of leveraging ultra-long
behavior sequences with length scaling up to thousands, existing retrieval
models remain constrained to sequences of hundreds of behaviors due to two main
challenges. One is strict latency budget imposed by real-time service over
large-scale candidate pool. The other is the absence of target-aware mechanisms
and cross-interaction architectures, which prevent utilizing ranking-like
techniques to simplify long sequence modeling. To address these limitations, we
propose a new framework named User Long-term Multi-Interest Retrieval
Model(ULIM), which enables thousand-scale behavior modeling in retrieval
stages. ULIM includes two novel components: 1)Category-Aware Hierarchical
Dual-Interest Learning partitions long behavior sequences into multiple
category-aware subsequences representing multi-interest and jointly optimizes
long-term and short-term interests within specific interest cluster.
2)Pointer-Enhanced Cascaded Category-to-Item Retrieval introduces
Pointer-Generator Interest Network(PGIN) for next-category prediction, followed
by next-item retrieval upon the top-K predicted categories. Comprehensive
experiments on Taobao dataset show that ULIM achieves substantial improvement
over state-of-the-art methods, and brings 5.54% clicks, 11.01% orders and 4.03%
GMV lift for Taobaomiaosha, a notable mini-app of Taobao.

### 4. [Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG](http://arxiv.org/pdf/2507.10411v1)

Authors: Fangzheng Tian, Jinyuan Fang, Debasis Ganguly, Zaiqiao Meng, Craig Macdonald

Agentic Retrieval-Augmented Generation (RAG) is a new paradigm where the
reasoning model decides when to invoke a retriever (as a "tool") when answering
a question. This paradigm, exemplified by recent research works such as
Search-R1, enables the model to decide when to search and obtain external
information. However, the queries generated by such Agentic RAG models and the
role of the retriever in obtaining high-quality answers remain understudied. To
this end, this initial study examines the applicability of query performance
prediction (QPP) within the recent Agentic RAG models Search-R1 and
R1-Searcher. We find that applying effective retrievers can achieve higher
answer quality within a shorter reasoning process. Moreover, the QPP estimates
of the generated queries, used as an approximation of their retrieval quality,
are positively correlated with the quality of the final answer. Ultimately, our
work is a step towards adaptive retrieval within Agentic RAG, where QPP is used
to inform the model if the retrieved results are likely to be useful.

### 5. [Riding the Carousel: The First Extensive Eye Tracking Analysis of Browsing Behavior in Carousel Recommenders](http://arxiv.org/pdf/2507.10135v1)

Authors: Santiago de Leon-Martinez, Robert Moro, Branislav Kveton, Maria Bielikova

Carousels have become the de-facto interface in online services. However,
there is a lack of research in carousels, particularly examining how
recommender systems may be designed differently than the traditional
single-list interfaces. One of the key elements for understanding how to design
a system for a particular interface is understanding how users browse. For
carousels, users may browse in a number of different ways due to the added
complexity of multiple topic defined-lists and swiping to see more items.
  Eye tracking is the key to understanding user behavior by providing valuable,
direct information on how users see and navigate. In this work, we provide the
first extensive analysis of the eye tracking behavior in carousel recommenders
under the free-browsing setting. To understand how users browse, we examine the
following research questions : 1) where do users start browsing, 2) how do
users transition from item to item within the same carousel and across
carousels, and 3) how does genre preference impact transitions?
  This work addresses a gap in the field and provides the first extensive
empirical results of eye tracked browsing behavior in carousels for improving
recommenders. Taking into account the insights learned from the above
questions, our final contribution is to provide suggestions to help carousel
recommender system designers optimize their systems for user browsing behavior.
The most important suggestion being to reorder the ranked item positions to
account for browsing after swiping.These contributions aim not only to help
improve current systems, but also to encourage and allow the design of new user
models, systems, and metrics that are better suited to the complexity of
carousel interfaces.

### 6. [Overcoming catastrophic forgetting in neural networks](http://arxiv.org/pdf/2507.10485v1)

Authors: Brandon Shuen Yi Loke, Filippo Quadri, Gabriel Vivanco, Maximilian Casagrande, Saúl Fenollosa

Catastrophic forgetting is the primary challenge that hinders continual
learning, which refers to a neural network ability to sequentially learn
multiple tasks while retaining previously acquired knowledge. Elastic Weight
Consolidation, a regularization-based approach inspired by synaptic
consolidation in biological neural systems, has been used to overcome this
problem. In this study prior research is replicated and extended by evaluating
EWC in supervised learning settings using the PermutedMNIST and RotatedMNIST
benchmarks. Through systematic comparisons with L2 regularization and
stochastic gradient descent (SGD) without regularization, we analyze how
different approaches balance knowledge retention and adaptability. Our results
confirm what was shown in previous research, showing that EWC significantly
reduces forgetting compared to naive training while slightly compromising
learning efficiency on new tasks. Moreover, we investigate the impact of
dropout regularization and varying hyperparameters, offering insights into the
generalization of EWC across diverse learning scenarios. These results
underscore EWC's potential as a viable solution for lifelong learning in neural
networks.

### 7. [MixLoRA-DSI: Dynamically Expandable Mixture-of-LoRA Experts for Rehearsal-Free Generative Retrieval over Dynamic Corpora](http://arxiv.org/pdf/2507.09924v1)

Authors: Tuan-Luc Huynh, Thuy-Trang Vu, Weiqing Wang, Trung Le, Dragan Gašević, Yuan-Fang Li, Thanh-Toan Do

Continually updating model-based indexes in generative retrieval with new
documents remains challenging, as full retraining is computationally expensive
and impractical under resource constraints. We propose MixLoRA-DSI, a novel
framework that combines an expandable mixture of Low-Rank Adaptation experts
with a layer-wise out-of-distribution (OOD)-driven expansion strategy. Instead
of allocating new experts for each new corpus, our proposed expansion strategy
enables sublinear parameter growth by selectively introducing new experts only
when significant number of OOD documents are detected. Experiments on NQ320k
and MS MARCO Passage demonstrate that MixLoRA-DSI outperforms full-model update
baselines, with minimal parameter overhead and substantially lower training
costs.

### 8. [PRISM: Fine-Grained Paper-to-Paper Retrieval with Multi-Aspect-Aware Query Optimization](http://arxiv.org/pdf/2507.10057v1)

Authors: Sangwoo Park, Jinheon Baek, Soyeong Jeong, Sung Ju Hwang

Scientific paper retrieval, particularly framed as document-to-document
retrieval, aims to identify relevant papers in response to a long-form query
paper, rather than a short query string. Previous approaches to this task have
focused on abstracts, embedding them into dense vectors as surrogates for full
documents and calculating similarity across them, although abstracts provide
only sparse and high-level summaries. To address this, we propose PRISM, a
novel document-to-document retrieval method that introduces multiple,
fine-grained representations for both the query and candidate papers. In
particular, each query paper is decomposed into multiple aspect-specific views
and individually embedded, which are then matched against candidate papers
similarity segmented to consider their multifaceted dimensions. Moreover, we
present SciFullBench, a novel benchmark in which the complete and segmented
context of full papers for both queries and candidates is available. Then,
experimental results show that PRISM improves performance by an average of 4.3%
over existing retrieval baselines.

### 9. [Text-to-Remote-Sensing-Image Retrieval beyond RGB Sources](http://arxiv.org/pdf/2507.10403v1)

Authors: Daniele Rege Cambrin, Lorenzo Vaiani, Giuseppe Gallipoli, Luca Cagliero, Paolo Garza

Retrieving relevant imagery from vast satellite archives is crucial for
applications like disaster response and long-term climate monitoring. However,
most text-to-image retrieval systems are limited to RGB data, failing to
exploit the unique physical information captured by other sensors, such as the
all-weather structural sensitivity of Synthetic Aperture Radar (SAR) or the
spectral signatures in optical multispectral data. To bridge this gap, we
introduce CrisisLandMark, a new large-scale corpus of over 647,000 Sentinel-1
SAR and Sentinel-2 multispectral images paired with structured textual
annotations for land cover, land use, and crisis events harmonized from
authoritative land cover systems (CORINE and Dynamic World) and crisis-specific
sources. We then present CLOSP (Contrastive Language Optical SAR Pretraining),
a novel framework that uses text as a bridge to align unpaired optical and SAR
images into a unified embedding space. Our experiments show that CLOSP achieves
a new state-of-the-art, improving retrieval nDGC by 54% over existing models.
Additionally, we find that the unified training strategy overcomes the inherent
difficulty of interpreting SAR imagery by transferring rich semantic knowledge
from the optical domain with indirect interaction. Furthermore, GeoCLOSP, which
integrates geographic coordinates into our framework, creates a powerful
trade-off between generality and specificity: while the CLOSP excels at general
semantic tasks, the GeoCLOSP becomes a specialized expert for retrieving
location-dependent crisis events and rare geographic features. This work
highlights that the integration of diverse sensor data and geographic context
is essential for unlocking the full potential of remote sensing archives.

### 10. [SentiDrop: A Multi Modal Machine Learning model for Predicting Dropout in Distance Learning](http://arxiv.org/pdf/2507.10421v1)

Authors: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui

School dropout is a serious problem in distance learning, where early
detection is crucial for effective intervention and student perseverance.
Predicting student dropout using available educational data is a widely
researched topic in learning analytics. Our partner's distance learning
platform highlights the importance of integrating diverse data sources,
including socio-demographic data, behavioral data, and sentiment analysis, to
accurately predict dropout risks. In this paper, we introduce a novel model
that combines sentiment analysis of student comments using the Bidirectional
Encoder Representations from Transformers (BERT) model with socio-demographic
and behavioral data analyzed through Extreme Gradient Boosting (XGBoost). We
fine-tuned BERT on student comments to capture nuanced sentiments, which were
then merged with key features selected using feature importance techniques in
XGBoost. Our model was tested on unseen data from the next academic year,
achieving an accuracy of 84\%, compared to 82\% for the baseline model.
Additionally, the model demonstrated superior performance in other metrics,
such as precision and F1-score. The proposed method could be a vital tool in
developing personalized strategies to reduce dropout rates and encourage
student perseverance

### Machine Learning

### 1. [Rethinking Prompt Optimization: Reinforcement, Diversification, and Migration in Blackbox LLMs](http://arxiv.org/pdf/2507.09839v1)

Authors: MohammadReza Davari, Utkarsh Garg, Weixin Cai, Eugene Belilovsky

An increasing number of NLP applications interact with large language models
(LLMs) through black-box APIs, making prompt engineering critical for
controlling model outputs. While recent Automatic Prompt Optimization (APO)
methods iteratively refine prompts using model-generated feedback, textual
gradients, they primarily focus on error correction and neglect valuable
insights from correct predictions. This limits both their effectiveness and
efficiency. In this paper, we propose a novel APO framework centered on
enhancing the feedback mechanism. We reinterpret the textual gradient as a form
of negative reinforcement and introduce the complementary positive
reinforcement to explicitly preserve beneficial prompt components identified
through successful predictions. To mitigate the noise inherent in LLM-generated
feedback, we introduce a technique called feedback diversification, which
aggregates multiple feedback signals, emphasizing consistent, actionable advice
while filtering out outliers. Motivated by the rapid evolution and diversity of
available LLMs, we also formalize Continual Prompt Optimization (CPO),
addressing the practical challenge of efficiently migrating optimized prompts
between different model versions or API providers. Our experiments reveal that
naive prompt migration often degrades performance due to loss of critical
instructions. In contrast, our approach consistently outperforms strong
baselines, achieving significant accuracy improvements, faster convergence, and
lower computational costs in both standard and migration scenarios.

### 2. [AdaBrain-Bench: Benchmarking Brain Foundation Models for Brain-Computer Interface Applications](http://arxiv.org/pdf/2507.09882v1)

Authors: Jiamin Wu, Zichen Ren, Junyu Wang, Pengyu Zhu, Yonghao Song, Mianxin Liu, Qihao Zheng, Lei Bai, Wanli Ouyang, Chunfeng Song

Non-invasive Brain-Computer Interfaces (BCI) offer a safe and accessible
means of connecting the human brain to external devices, with broad
applications in home and clinical settings to enhance human capabilities.
However, the high noise level and limited task-specific data in non-invasive
signals constrain decoding capabilities. Recently, the adoption of
self-supervised pre-training is transforming the landscape of non-invasive BCI
research, enabling the development of brain foundation models to capture
generic neural representations from large-scale unlabeled
electroencephalography (EEG) signals with substantial noises. However, despite
these advances, the field currently lacks comprehensive, practical and
extensible benchmarks to assess the utility of the public foundation models
across diverse BCI tasks, hindering their widespread adoption. To address this
challenge, we present AdaBrain-Bench, a large-scale standardized benchmark to
systematically evaluate brain foundation models in widespread non-invasive BCI
tasks. AdaBrain-Bench encompasses a diverse collection of representative BCI
decoding datasets spanning 7 key applications. It introduces a streamlined task
adaptation pipeline integrated with multi-dimensional evaluation metrics and a
set of adaptation tools. The benchmark delivers an inclusive framework for
assessing generalizability of brain foundation models across key transfer
settings, including cross-subject, multi-subject, and few-shot scenarios. We
leverage AdaBrain-Bench to evaluate a suite of publicly available brain
foundation models and offer insights into practices for selecting appropriate
models in various scenarios. We make our benchmark pipeline available to enable
reproducible research and external use, offering a continuously evolving
platform to foster progress toward robust and generalized neural decoding
solutions.

### 3. [Extracting Cause-Effect Pairs from a Sentence with a Dependency-Aware Transformer Model](http://arxiv.org/pdf/2507.09925v1)

Authors: Md Ahsanul Kabir, Abrar Jahin, Mohammad Al Hasan

Extracting cause and effect phrases from a sentence is an important NLP task,
with numerous applications in various domains, including legal, medical,
education, and scientific research. There are many unsupervised and supervised
methods proposed for solving this task. Among these, unsupervised methods
utilize various linguistic tools, including syntactic patterns, dependency
tree, dependency relations, etc. among different sentential units for
extracting the cause and effect phrases. On the other hand, the contemporary
supervised methods use various deep learning based mask language models
equipped with a token classification layer for extracting cause and effect
phrases. Linguistic tools, specifically, dependency tree, which organizes a
sentence into different semantic units have been shown to be very effective for
extracting semantic pairs from a sentence, but existing supervised methods do
not have any provision for utilizing such tools within their model framework.
In this work, we propose DepBERT, which extends a transformer-based model by
incorporating dependency tree of a sentence within the model framework.
Extensive experiments over three datasets show that DepBERT is better than
various state-of-the art supervised causality extraction methods.

### 4. [Hierarchical Job Classification with Similarity Graph Integration](http://arxiv.org/pdf/2507.09949v1)

Authors: Md Ahsanul Kabir, Kareem Abdelfatah, Mohammed Korayem, Mohammad Al Hasan

In the dynamic realm of online recruitment, accurate job classification is
paramount for optimizing job recommendation systems, search rankings, and labor
market analyses. As job markets evolve, the increasing complexity of job titles
and descriptions necessitates sophisticated models that can effectively
leverage intricate relationships within job data. Traditional text
classification methods often fall short, particularly due to their inability to
fully utilize the hierarchical nature of industry categories. To address these
limitations, we propose a novel representation learning and classification
model that embeds jobs and hierarchical industry categories into a latent
embedding space. Our model integrates the Standard Occupational Classification
(SOC) system and an in-house hierarchical taxonomy, Carotene, to capture both
graph and hierarchical relationships, thereby improving classification
accuracy. By embedding hierarchical industry categories into a shared latent
space, we tackle cold start issues and enhance the dynamic matching of
candidates to job opportunities. Extensive experimentation on a large-scale
dataset of job postings demonstrates the model's superior ability to leverage
hierarchical structures and rich semantic features, significantly outperforming
existing methods. This research provides a robust framework for improving job
classification accuracy, supporting more informed decision-making in the
recruitment industry.

### 5. [Rethinking Inductive Bias in Geographically Neural Network Weighted Regression](http://arxiv.org/pdf/2507.09958v1)

Authors: Zhenyuan Chen

Inductive bias is a key factor in spatial regression models, determining how
well a model can learn from limited data and capture spatial patterns. This
work revisits the inductive biases in Geographically Neural Network Weighted
Regression (GNNWR) and identifies limitations in current approaches for
modeling spatial non-stationarity. While GNNWR extends traditional
Geographically Weighted Regression by using neural networks to learn spatial
weighting functions, existing implementations are often restricted by fixed
distance-based schemes and limited inductive bias. We propose to generalize
GNNWR by incorporating concepts from convolutional neural networks, recurrent
neural networks, and transformers, introducing local receptive fields,
sequential context, and self-attention into spatial regression. Through
extensive benchmarking on synthetic spatial datasets with varying
heterogeneity, noise, and sample sizes, we show that GNNWR outperforms classic
methods in capturing nonlinear and complex spatial relationships. Our results
also reveal that model performance depends strongly on data characteristics,
with local models excelling in highly heterogeneous or small-sample scenarios,
and global models performing better with larger, more homogeneous data. These
findings highlight the importance of inductive bias in spatial modeling and
suggest future directions, including learnable spatial weighting functions,
hybrid neural architectures, and improved interpretability for models handling
non-stationary spatial data.

### 6. [Text-Driven Causal Representation Learning for Source-Free Domain Generalization](http://arxiv.org/pdf/2507.09961v1)

Authors: Lihua Zhou, Mao Ye, Nianxin Li, Shuaifeng Li, Jinlin Wu, Xiatian Zhu, Lei Deng, Hongbin Liu, Jiebo Luo, Zhen Lei

Deep learning often struggles when training and test data distributions
differ. Traditional domain generalization (DG) tackles this by including data
from multiple source domains, which is impractical due to expensive data
collection and annotation. Recent vision-language models like CLIP enable
source-free domain generalization (SFDG) by using text prompts to simulate
visual representations, reducing data demands. However, existing SFDG methods
struggle with domain-specific confounders, limiting their generalization
capabilities. To address this issue, we propose TDCRL
(\textbf{T}ext-\textbf{D}riven \textbf{C}ausal \textbf{R}epresentation
\textbf{L}earning), the first method to integrate causal inference into the
SFDG setting. TDCRL operates in two steps: first, it employs data augmentation
to generate style word vectors, combining them with class information to
generate text embeddings to simulate visual representations; second, it trains
a causal intervention network with a confounder dictionary to extract
domain-invariant features. Grounded in causal learning, our approach offers a
clear and effective mechanism to achieve robust, domain-invariant features,
ensuring robust generalization. Extensive experiments on PACS, VLCS,
OfficeHome, and DomainNet show state-of-the-art performance, proving TDCRL
effectiveness in SFDG.

### 7. [Compliance Minimization via Physics-Informed Gaussian Processes](http://arxiv.org/pdf/2507.09968v1)

Authors: Xiangyu Sun, Amin Yousefpour, Shirin Hosseinmardi, Ramin Bostanabad

Machine learning (ML) techniques have recently gained significant attention
for solving compliance minimization (CM) problems. However, these methods
typically provide poor feature boundaries, are very expensive, and lack a
systematic mechanism to control the design complexity. Herein, we address these
limitations by proposing a mesh-free and simultaneous framework based on
physics-informed Gaussian processes (GPs). In our approach, we parameterize the
design and state variables with GP priors which have independent kernels but
share a multi-output neural network (NN) as their mean function. The
architecture of this NN is based on Parametric Grid Convolutional Attention
Networks (PGCANs) which not only mitigate spectral bias issues, but also
provide an interpretable mechanism to control design complexity. We estimate
all the parameters of our GP-based representations by simultaneously minimizing
the compliance, total potential energy, and residual of volume fraction
constraint. Importantly, our loss function exclude all data-based residuals as
GPs automatically satisfy them. We also develop computational schemes based on
curriculum training and numerical integration to increase the efficiency and
robustness of our approach which is shown to (1) produce super-resolution
topologies with fast convergence, (2) achieve smaller compliance and less gray
area fraction compared to traditional numerical methods, (3) provide control
over fine-scale features, and (4) outperform competing ML-based methods.

### 8. [Forecasting Coccidioidomycosis (Valley Fever) in Arizona: A Graph Neural Network Approach](http://arxiv.org/pdf/2507.10014v1)

Authors: Ali Sarabi, Arash Sarabi, Hao Yan, Beckett Sterner, Petar Jevtić

Coccidioidomycosis, commonly known as Valley Fever, remains a significant
public health concern in endemic regions of the southwestern United States.
This study develops the first graph neural network (GNN) model for forecasting
Valley Fever incidence in Arizona. The model integrates surveillance case data
with environmental predictors using graph structures, including soil
conditions, atmospheric variables, agricultural indicators, and air quality
metrics. Our approach explores correlation-based relationships among variables
influencing disease transmission. The model captures critical delays in disease
progression through lagged effects, enhancing its capacity to reflect complex
temporal dependencies in disease ecology. Results demonstrate that the GNN
architecture effectively models Valley Fever trends and provides insights into
key environmental drivers of disease incidence. These findings can inform early
warning systems and guide resource allocation for disease prevention efforts in
high-risk areas.

### 9. [On the Efficiency of Training Robust Decision Trees](http://arxiv.org/pdf/2507.10048v1)

Authors: Benedict Gerlach, Marie Anastacio, Holger H. Hoos

As machine learning gets adopted into the industry quickly, trustworthiness
is increasingly in focus. Yet, efficiency and sustainability of robust training
pipelines still have to be established. In this work, we consider a simple
pipeline for training adversarially robust decision trees and investigate the
efficiency of each step. Our pipeline consists of three stages. Firstly, we
choose the perturbation size automatically for each dataset. For that, we
introduce a simple algorithm, instead of relying on intuition or prior work.
Moreover, we show that the perturbation size can be estimated from smaller
models than the one intended for full training, and thus significant gains in
efficiency can be achieved. Secondly, we train state-of-the-art adversarial
training methods and evaluate them regarding both their training time and
adversarial accuracy. Thirdly, we certify the robustness of each of the models
thus obtained and investigate the time required for this. We find that
verification time, which is critical to the efficiency of the full pipeline, is
not correlated with training time.

### 10. [Understanding the Rank of Tensor Networks via an Intuitive Example-Driven Approach](http://arxiv.org/pdf/2507.10170v1)

Authors: Wuyang Zhou, Giorgos Iacovides, Kriton Konstantinidis, Ilya Kisil, Danilo Mandic

Tensor Network (TN) decompositions have emerged as an indispensable tool in
Big Data analytics owing to their ability to provide compact low-rank
representations, thus alleviating the ``Curse of Dimensionality'' inherent in
handling higher-order data. At the heart of their success lies the concept of
TN ranks, which governs the efficiency and expressivity of TN decompositions.
However, unlike matrix ranks, TN ranks often lack a universal meaning and an
intuitive interpretation, with their properties varying significantly across
different TN structures. Consequently, TN ranks are frequently treated as
empirically tuned hyperparameters, rather than as key design parameters
inferred from domain knowledge. The aim of this Lecture Note is therefore to
demystify the foundational yet frequently misunderstood concept of TN ranks
through real-life examples and intuitive visualizations. We begin by
illustrating how domain knowledge can guide the selection of TN ranks in
widely-used models such as the Canonical Polyadic (CP) and Tucker
decompositions. For more complex TN structures, we employ a self-explanatory
graphical approach that generalizes to tensors of arbitrary order. Such a
perspective naturally reveals the relationship between TN ranks and the
corresponding ranks of tensor unfoldings (matrices), thereby circumventing
cumbersome multi-index tensor algebra while facilitating domain-informed TN
design. It is our hope that this Lecture Note will equip readers with a clear
and unified understanding of the concept of TN rank, along with the necessary
physical insight and intuition to support the selection, explainability, and
deployment of tensor methods in both practical applications and educational
contexts.

### Neural and Evolutionary Computing

### 1. [Effective Self-Attention-Based Deep Learning Model with Evolutionary Grid Search for Robust Wave Farm Energy Forecasting](http://arxiv.org/pdf/2507.09847v1)

Authors: Amin Abdollahi Dehkordi, Mehdi Neshat, Nataliia Y. Sergiienko, Zahra Ghasemi, Lei Chen, John Boland, Hamid Moradkhani, Amir H. Gandomi

Achieving carbon neutrality, a key focus of UN SDG #13, drives the
exploration of wave energy, a renewable resource with the potential to generate
30,000 TWh of clean electricity annually, surpassing global demand. However,
wave energy remains underdeveloped due to technical and economic challenges,
particularly in forecasting wave farm power output, which is vital for grid
stability and commercial viability. This study proposes a novel predictive
framework to enhance wave energy integration into power grids. It introduces a
hybrid sequential learning model combining Self-Attention-enhanced
Convolutional Bi-LSTM with hyperparameter optimization. The model leverages
spatial data from Wave Energy Converters (WECs) and is validated using datasets
from wave farms in Adelaide, Sydney, Perth, and Tasmania, Australia.
Benchmarked against ten machine learning algorithms, the model achieves
superior accuracy, with R2 scores of 91.7% (Adelaide), 88.0% (Perth), 82.8%
(Tasmania), and 91.0% (Sydney). It outperforms conventional ML and deep
learning methods, offering robust and scalable predictions for wave energy
output across diverse marine environments, supporting reliable integration into
energy systems.

### 2. [Evolution of Fear and Social Rewards in Prey-Predator Relationship](http://arxiv.org/pdf/2507.09992v1)

Authors: Yuji Kanagawa, Kenji Doya

Fear is a critical brain function for detecting danger and learning to avoid
specific stimuli that can lead to danger. While fear is believed to have
evolved under pressure from predators, experimentally reproducing the evolution
is challenging. To investigate the relationship between environmental
conditions, the evolution of fear, and the evolution of other rewards, such as
food reward and social reward, we developed a distributed evolutionary
simulation. In our simulation, prey and predator agents co-evolve their innate
reward functions, including a possibly fear-like term for observing predators,
and learn behaviors via reinforcement learning. Surprisingly, our simulation
revealed that social reward for observing the same species is more important
for prey to survive, and fear-like negative reward for observing predators
evolves only after acquiring social reward. We also found that the predator
with increased hunting ability (larger mouth) amplified fear emergence, but
also that fear evolution is more stable with non-evolving predators that are
bad at chasing prey. Additionally, unlike for predators, we found that positive
rewards evolve in opposition to fear for stationary threats, as areas with
abundant leftover food develop around them. These findings suggest that fear
and social reward have had a complex interplay with each other through
evolution, along with the nature of predators and threats.

### 3. [Effects of structural properties of neural networks on machine learning performance](http://arxiv.org/pdf/2507.10005v1)

Authors: Yash Arya, Sang Hoon Lee

In recent years, graph-based machine learning techniques, such as
reinforcement learning and graph neural networks, have garnered significant
attention. While some recent studies have started to explore the relationship
between the graph structure of neural networks and their predictive
performance, they often limit themselves to a narrow range of model networks,
particularly lacking mesoscale structures such as communities. Our work
advances this area by conducting a more comprehensive investigation,
incorporating realistic network structures characterized by heterogeneous
degree distributions and community structures, which are typical
characteristics of many real networks. These community structures offer a
nuanced perspective on network architecture. Our analysis employs model
networks such as random and scale-free networks, alongside a comparison with a
biological neural network and its subsets for more detailed analysis. We
examine the impact of these structural attributes on the performance of image
classification tasks. Our findings reveal that structural properties do affect
performance to some extent. Specifically, networks featuring coherent, densely
interconnected communities demonstrate enhanced learning capabilities. The
comparison with the biological neural network emphasizes the relevance of our
findings to real-world structures, suggesting an intriguing connection worth
further exploration. This study contributes meaningfully to network science and
machine learning, providing insights that could inspire the design of more
biologically informed neural networks.

### 4. [Dynamical stability for dense patterns in discrete attractor neural networks](http://arxiv.org/pdf/2507.10383v1)

Authors: Uri Cohen, Máté Lengyel

Neural networks storing multiple discrete attractors are canonical models of
biological memory. Previously, the dynamical stability of such networks could
only be guaranteed under highly restrictive conditions. Here, we derive a
theory of the local stability of discrete fixed points in a broad class of
networks with graded neural activities and in the presence of noise. By
directly analyzing the bulk and outliers of the Jacobian spectrum, we show that
all fixed points are stable below a critical load that is distinct from the
classical \textit{critical capacity} and depends on the statistics of neural
activities in the fixed points as well as the single-neuron activation
function. Our analysis highlights the computational benefits of
threshold-linear activation and sparse-like patterns.

### Networking and Internet Architecture

### 1. [Fine-Grained Coordinated OFDMA With Fiber Backhaul Enabled by openwifi and White Rabbit](http://arxiv.org/pdf/2507.10210v1)

Authors: Thijs Havinga, Xianjun Jiao, Wei Liu, Baiheng Chen, Robbe Gaeremynck, Ingrid Moerman

Proper coordination is needed to guarantee the performance of wireless
networks in dense deployments. Contention-based systems suffer badly in terms
of latency when multiple devices compete for the same resources. Coordinated
Orthogonal Frequency Division Multiple Access (Co-OFDMA) is proposed for Wi-Fi
8 to remedy this, as it enables multiple Access Points (APs) to share spectrum
more efficiently. However, fine-grained resource allocation, namely within
20MHz bandwidth, is argued to be impractical due to the over-the-air scheduling
overhead and complexity in terms of physical layer signaling. A wired backhaul
mitigates the need for over-the-air scheduling and synchronization, and it
allows for coordination even if APs are not in each others' range. Furthermore,
it forms the basis for more advanced multi-AP coordination schemes like
coordinated beamforming and joint transmission. In this work we demonstrate the
realization of Wi-Fi 6 compliant fine-grained Co-OFDMA using a fiber backhaul,
enabled by the open-source platforms openwifi and White Rabbit. We show that
the performance in terms of carrier frequency offset pre-compensation and time
synchronization between two APs exceeds related wireless standard requirements.
Furthermore, the quality of the received constellation of the Co-OFDMA frame as
reported by a wireless connectivity tester is better than individual frames
sent by the APs.

### 2. [Endorsement-Driven Blockchain SSI Framework for Dynamic IoT Ecosystems](http://arxiv.org/pdf/2507.09859v1)

Authors: Guntur Dharma Putra, Bagus Rakadyanto Oktavianto Putra

Self-Sovereign Identity (SSI) offers significant potential for managing
identities in the Internet of Things (IoT), enabling decentralized
authentication and credential management without reliance on centralized
entities. However, existing SSI frameworks often limit credential issuance and
revocation to trusted entities, such as IoT manufacturers, which restricts
flexibility in dynamic IoT ecosystems. In this paper, we propose a
blockchain-based SSI framework that allows any individual with a verifiable
trust linkage to act as a credential issuer, ensuring decentralized and
scalable identity management. Our framework incorporates a layered
architecture, where trust is dynamically established through endorsement-based
calculations and maintained via a hierarchical chain-of-trust mechanism.
Blockchain serves as the Verifiable Data Registry, ensuring transparency and
immutability of identity operations, while smart contracts automate critical
processes such as credential issuance, verification, and revocation. A
proof-of-concept implementation demonstrates that the proposed framework is
feasible and incurs minimal overheads compared to the baseline, making it
well-suited for dynamic and resource-constrained IoT environments.

### 3. [Cross-Timeslot Optimization for Distributed GPU Inference Using Reinforcement Learning](http://arxiv.org/pdf/2507.10259v1)

Authors: Chengze Du, Zhiwei Yu, Heng Xu, Haojie Wang, Bo liu, Jialong Li

The rapid growth of large language model (LLM) services imposes increasing
demands on distributed GPU inference infrastructure. Most existing scheduling
systems rely on the current system state to make decisions, without considering
how task demand and resource availability evolve over time. This lack of
temporal awareness leads to inefficient GPU utilization, high task migration
overhead, and poor system responsiveness under dynamic workloads. In this work,
we identify the fundamental limitations of these instantaneous-state-only
scheduling approaches and propose Temporal Optimal Resource scheduling via
Two-layer Architecture (TORTA). TORTA introduces a spatiotemporal scheduling
framework that captures both long-term workload patterns and short-term
execution constraints. It adopts a two-layer design: a macro-level scheduler
leverages reinforcement learning and optimal transport to coordinate
inter-region task distribution, while a micro-level allocator refines
task-to-server assignments within each region to reduce latency and switching
costs. Experimental results across multiple network topologies show that TORTA
reduces average inference response time by up to 15\%, improves load balance by
approximately 4-5\%, and cuts total operational cost by 10-20\% compared to
state-of-the-art baseline methods.

### 4. [UavNetSim-v1: A Python-based Simulation Platform for UAV Communication Networks](http://arxiv.org/pdf/2507.09852v1)

Authors: Zihao Zhou, Zipeng Dai, Linyi Huang, Cui Yang, Youjun Xiang, Jie Tang, Kai-kit Wong

In unmanned aerial vehicle (UAV) networks, communication protocols and
algorithms are essential for cooperation and collaboration between UAVs.
Simulation provides a cost-effective solution for prototyping, debugging, and
analyzing protocols and algorithms, avoiding the prohibitive expenses of field
experiments. In this paper, we present ``UavNetSim-v1'', an open-source
Python-based simulation platform designed for rapid development, testing, and
evaluating the protocols and algorithms in UAV networks. ``UavNetSim-v1''
provides most of the functionalities developers may need, including
routing/medium access control (MAC) protocols, topology control algorithms and
mobility/energy models, while maintaining ease of use. Furthermore, the
platform supports comprehensive performance evaluation and features an
interactive visualization interface for in-depth algorithm analysis. In short,
``UavNetSim-v1'' lends itself to both rapid prototyping and educational
purposes, and can serve as a lightweight yet powerful alternative to mature
network simulators for UAV communication research.

### 5. [DNS Tunneling: Threat Landscape and Improved Detection Solutions](http://arxiv.org/pdf/2507.10267v1)

Authors: Novruz Amirov, Baran Isik, Bilal Ihsan Tuncer, Serif Bahtiyar

Detecting Domain Name System (DNS) tunneling is a significant challenge in
security due to its capacity to hide harmful actions within DNS traffic that
appears to be normal and legitimate. Traditional detection methods are based on
rule-based approaches or signature matching methods that are often insufficient
to accurately identify such covert communication channels. This research is
about effectively detecting DNS tunneling. We propose a novel approach to
detect DNS tunneling with machine learning algorithms. We combine machine
learning algorithms to analyze the traffic by using features extracted from DNS
traffic. Analyses results show that the proposed approach is a good candidate
to detect DNS tunneling accurately.

### 6. [Chat with AI: The Surprising Turn of Real-time Video Communication from Human to AI](http://arxiv.org/pdf/2507.10510v1)

Authors: Jiangkai Wu, Zhiyuan Ren, Liming Liu, Xinggong Zhang

AI Video Chat emerges as a new paradigm for Real-time Communication (RTC),
where one peer is not a human, but a Multimodal Large Language Model (MLLM).
This makes interaction between humans and AI more intuitive, as if chatting
face-to-face with a real person. However, this poses significant challenges to
latency, because the MLLM inference takes up most of the response time, leaving
very little time for video streaming. Due to network uncertainty and
instability, transmission latency becomes a critical bottleneck preventing AI
from being like a real person. To address this, we propose Artic, an
AI-oriented Real-time Communication framework, exploring the network
requirement shift from "humans watching video" to "AI understanding video". To
reduce bitrate dramatically while maintaining MLLM accuracy, we propose
Context-Aware Video Streaming that recognizes the importance of each video
region for chat and allocates bitrate almost exclusively to chat-important
regions. To avoid packet retransmission, we propose Loss-Resilient Adaptive
Frame Rate that leverages previous frames to substitute for lost/delayed frames
while avoiding bitrate waste. To evaluate the impact of video streaming quality
on MLLM accuracy, we build the first benchmark, named Degraded Video
Understanding Benchmark (DeViBench). Finally, we discuss some open questions
and ongoing solutions for AI Video Chat.

### 7. [Green-LLM: Optimal Workload Allocation for Environmentally-Aware Distributed Inference](http://arxiv.org/pdf/2507.09942v1)

Authors: Jiaming Cheng, Duong Tung Nguyen

This letter investigates the optimal allocation of large language model (LLM)
inference workloads across heterogeneous edge data centers (DCs) over time.
Each DC features on-site renewable generation and faces dynamic electricity
prices and spatiotemporal variability in renewable availability. The central
question is: how can inference workloads be optimally distributed to the DCs to
minimize energy consumption, carbon emissions, and water usage while enhancing
user experience? This letter proposes a novel optimization model for LLM
service providers to reduce operational costs and environmental impacts.
Numerical results validate the efficacy of the proposed approach.

### Robotics

### 1. [Customize Harmonic Potential Fields via Hybrid Optimization over Homotopic Paths](http://arxiv.org/pdf/2507.09858v1)

Authors: Shuaikang Wang, Tiecheng Guo, Meng Guo

Safe navigation within a workspace is a fundamental skill for autonomous
robots to accomplish more complex tasks. Harmonic potentials are artificial
potential fields that are analytical, globally convergent and provably free of
local minima. Thus, it has been widely used for generating safe and reliable
robot navigation control policies. However, most existing methods do not allow
customization of the harmonic potential fields nor the resulting paths,
particularly regarding their topological properties. In this paper, we propose
a novel method that automatically finds homotopy classes of paths that can be
generated by valid harmonic potential fields. The considered complex workspaces
can be as general as forest worlds consisting of numerous overlapping
star-obstacles. The method is based on a hybrid optimization algorithm that
searches over homotopy classes, selects the structure of each tree-of-stars
within the forest, and optimizes over the continuous weight parameters for each
purged tree via the projected gradient descent. The key insight is to transform
the forest world to the unbounded point world via proper diffeomorphic
transformations. It not only facilitates a simpler design of the
multi-directional D-signature between non-homotopic paths, but also retain the
safety and convergence properties. Extensive simulations and hardware
experiments are conducted for non-trivial scenarios, where the navigation
potentials are customized for desired homotopic properties. Project page:
https://shuaikang-wang.github.io/CustFields.

### 2. [Ariel Explores: Vision-based underwater exploration and inspection via generalist drone-level autonomy](http://arxiv.org/pdf/2507.10003v1)

Authors: Mohit Singh, Mihir Dharmadhikari, Kostas Alexis

This work presents a vision-based underwater exploration and inspection
autonomy solution integrated into Ariel, a custom vision-driven underwater
robot. Ariel carries a $5$ camera and IMU based sensing suite, enabling a
refraction-aware multi-camera visual-inertial state estimation method aided by
a learning-based proprioceptive robot velocity prediction method that enhances
robustness against visual degradation. Furthermore, our previously developed
and extensively field-verified autonomous exploration and general visual
inspection solution is integrated on Ariel, providing aerial drone-level
autonomy underwater. The proposed system is field-tested in a submarine dry
dock in Trondheim under challenging visual conditions. The field demonstration
shows the robustness of the state estimation solution and the generalizability
of the path planning techniques across robot embodiments.

### 3. [Finetuning Deep Reinforcement Learning Policies with Evolutionary Strategies for Control of Underactuated Robots](http://arxiv.org/pdf/2507.10030v1)

Authors: Marco Calì, Alberto Sinigaglia, Niccolò Turcato, Ruggero Carli, Gian Antonio Susto

Deep Reinforcement Learning (RL) has emerged as a powerful method for
addressing complex control problems, particularly those involving underactuated
robotic systems. However, in some cases, policies may require refinement to
achieve optimal performance and robustness aligned with specific task
objectives. In this paper, we propose an approach for fine-tuning Deep RL
policies using Evolutionary Strategies (ES) to enhance control performance for
underactuated robots. Our method involves initially training an RL agent with
Soft-Actor Critic (SAC) using a surrogate reward function designed to
approximate complex specific scoring metrics. We subsequently refine this
learned policy through a zero-order optimization step employing the Separable
Natural Evolution Strategy (SNES), directly targeting the original score.
Experimental evaluations conducted in the context of the 2nd AI Olympics with
RealAIGym at IROS 2024 demonstrate that our evolutionary fine-tuning
significantly improves agent performance while maintaining high robustness. The
resulting controllers outperform established baselines, achieving competitive
scores for the competition tasks.

### 4. [MP-RBFN: Learning-based Vehicle Motion Primitives using Radial Basis Function Networks](http://arxiv.org/pdf/2507.10047v1)

Authors: Marc Kaufeld, Mattia Piccinini, Johannes Betz

This research introduces MP-RBFN, a novel formulation leveraging Radial Basis
Function Networks for efficiently learning Motion Primitives derived from
optimal control problems for autonomous driving. While traditional motion
planning approaches based on optimization are highly accurate, they are often
computationally prohibitive. In contrast, sampling-based methods demonstrate
high performance but impose constraints on the geometric shape of trajectories.
MP-RBFN combines the strengths of both by coupling the high-fidelity trajectory
generation of sampling-based methods with an accurate description of vehicle
dynamics. Empirical results show compelling performance compared to previous
methods, achieving a precise description of motion primitives at low inference
times. MP-RBFN yields a seven times higher accuracy in generating optimized
motion primitives compared to existing semi-analytic approaches. We demonstrate
the practical applicability of MP-RBFN for motion planning by integrating the
method into a sampling-based trajectory planner. MP-RBFN is available as
open-source software at https://github.com/TUM-AVS/RBFN-Motion-Primitives.

### 5. [Hand Gesture Recognition for Collaborative Robots Using Lightweight Deep Learning in Real-Time Robotic Systems](http://arxiv.org/pdf/2507.10055v1)

Authors: Muhtadin, I Wayan Agus Darmawan, Muhammad Hilmi Rusydiansyah, I Ketut Eddy Purnama, Chastine Fatichah, Mauridhi Hery Purnomo

Direct and natural interaction is essential for intuitive human-robot
collaboration, eliminating the need for additional devices such as joysticks,
tablets, or wearable sensors. In this paper, we present a lightweight deep
learning-based hand gesture recognition system that enables humans to control
collaborative robots naturally and efficiently. This model recognizes eight
distinct hand gestures with only 1,103 parameters and a compact size of 22 KB,
achieving an accuracy of 93.5%. To further optimize the model for real-world
deployment on edge devices, we applied quantization and pruning using
TensorFlow Lite, reducing the final model size to just 7 KB. The system was
successfully implemented and tested on a Universal Robot UR5 collaborative
robot within a real-time robotic framework based on ROS2. The results
demonstrate that even extremely lightweight models can deliver accurate and
responsive hand gesture-based control for collaborative robots, opening new
possibilities for natural human-robot interaction in constrained environments.

### 6. [Foundation Model Driven Robotics: A Comprehensive Review](http://arxiv.org/pdf/2507.10087v1)

Authors: Muhammad Tayyab Khan, Ammar Waheed

The rapid emergence of foundation models, particularly Large Language Models
(LLMs) and Vision-Language Models (VLMs), has introduced a transformative
paradigm in robotics. These models offer powerful capabilities in semantic
understanding, high-level reasoning, and cross-modal generalization, enabling
significant advances in perception, planning, control, and human-robot
interaction. This critical review provides a structured synthesis of recent
developments, categorizing applications across simulation-driven design,
open-world execution, sim-to-real transfer, and adaptable robotics. Unlike
existing surveys that emphasize isolated capabilities, this work highlights
integrated, system-level strategies and evaluates their practical feasibility
in real-world environments. Key enabling trends such as procedural scene
generation, policy generalization, and multimodal reasoning are discussed
alongside core bottlenecks, including limited embodiment, lack of multimodal
data, safety risks, and computational constraints. Through this lens, this
paper identifies both the architectural strengths and critical limitations of
foundation model-based robotics, highlighting open challenges in real-time
operation, grounding, resilience, and trust. The review concludes with a
roadmap for future research aimed at bridging semantic reasoning and physical
intelligence through more robust, interpretable, and embodied models.

### 7. [Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots](http://arxiv.org/pdf/2507.10105v1)

Authors: Ines Sorrentino, Giulio Romualdi, Lorenzo Moretti, Silvio Traversaro, Daniele Pucci

This paper presents a novel framework for whole-body torque control of
humanoid robots without joint torque sensors, designed for systems with
electric motors and high-ratio harmonic drives. The approach integrates
Physics-Informed Neural Networks (PINNs) for friction modeling and Unscented
Kalman Filtering (UKF) for joint torque estimation, within a real-time torque
control architecture. PINNs estimate nonlinear static and dynamic friction from
joint and motor velocity readings, capturing effects like motor actuation
without joint movement. The UKF utilizes PINN-based friction estimates as
direct measurement inputs, improving torque estimation robustness. Experimental
validation on the ergoCub humanoid robot demonstrates improved torque tracking
accuracy, enhanced energy efficiency, and superior disturbance rejection
compared to the state-of-the-art Recursive Newton-Euler Algorithm (RNEA), using
a dynamic balancing experiment. The framework's scalability is shown by
consistent performance across robots with similar hardware but different
friction characteristics, without re-identification. Furthermore, a comparative
analysis with position control highlights the advantages of the proposed torque
control approach. The results establish the method as a scalable and practical
solution for sensorless torque control in humanoid robots, ensuring torque
tracking, adaptability, and stability in dynamic environments.

### 8. [Simulations and experiments with assemblies of fiber-reinforced soft actuators](http://arxiv.org/pdf/2507.10121v1)

Authors: Seung Hyun Kim, Jiamiao Guo, Arman Tekinalp, Heng-Sheng Chang, Ugur Akcal, Tixian Wang, Darren Biskup, Benjamin Walt, Girish Chowdhary, Girish Krishnan, Prashant G. Mehta, Mattia Gazzola

Soft continuum arms (SCAs) promise versatile manipulation through mechanical
compliance, for assistive devices, agriculture, search applications, or
surgery. However, SCAs' real-world use is challenging, partly due to their
hard-to-control non-linear behavior. Here, a simulation framework for SCAs
modularly assembled out of fiber reinforced elastomeric enclosures (FREEs) is
developed and integrated with a video-tracking system for experimental testing
and control design.

### 9. [Robust RL Control for Bipedal Locomotion with Closed Kinematic Chains](http://arxiv.org/pdf/2507.10164v1)

Authors: Egor Maslennikov, Eduard Zaliaev, Nikita Dudorov, Oleg Shamanin, Karanov Dmitry, Gleb Afanasev, Alexey Burkov, Egor Lygin, Simeon Nedelchev, Evgeny Ponomarev

Developing robust locomotion controllers for bipedal robots with closed
kinematic chains presents unique challenges, particularly since most
reinforcement learning (RL) approaches simplify these parallel mechanisms into
serial models during training. We demonstrate that this simplification
significantly impairs sim-to-real transfer by failing to capture essential
aspects such as joint coupling, friction dynamics, and motor-space control
characteristics. In this work, we present an RL framework that explicitly
incorporates closed-chain dynamics and validate it on our custom-built robot
TopA. Our approach enhances policy robustness through symmetry-aware loss
functions, adversarial training, and targeted network regularization.
Experimental results demonstrate that our integrated approach achieves stable
locomotion across diverse terrains, significantly outperforming methods based
on simplified kinematic models.

### 10. [TOP: Trajectory Optimization via Parallel Optimization towards Constant Time Complexity](http://arxiv.org/pdf/2507.10290v1)

Authors: Jiajun Yu, Nanhe Chen, Guodong Liu, Chao Xu, Fei Gao, Yanjun Cao

Optimization has been widely used to generate smooth trajectories for motion
planning. However, existing trajectory optimization methods show weakness when
dealing with large-scale long trajectories. Recent advances in parallel
computing have accelerated optimization in some fields, but how to efficiently
solve trajectory optimization via parallelism remains an open question. In this
paper, we propose a novel trajectory optimization framework based on the
Consensus Alternating Direction Method of Multipliers (CADMM) algorithm, which
decomposes the trajectory into multiple segments and solves the subproblems in
parallel. The proposed framework reduces the time complexity to O(1) per
iteration to the number of segments, compared to O(N) of the state-of-the-art
(SOTA) approaches. Furthermore, we introduce a closed-form solution that
integrates convex linear and quadratic constraints to speed up the
optimization, and we also present numerical solutions for general inequality
constraints. A series of simulations and experiments demonstrate that our
approach outperforms the SOTA approach in terms of efficiency and smoothness.
Especially for a large-scale trajectory, with one hundred segments, achieving
over a tenfold speedup. To fully explore the potential of our algorithm on
modern parallel computing architectures, we deploy our framework on a GPU and
show high performance with thousands of segments.

### Software Engineering

### 1. [PathFuzzing: Worst Case Analysis by Fuzzing Symbolic-Execution Paths](http://arxiv.org/pdf/2507.09892v1)

Authors: Zimu Chen, Di Wang

Estimating worst-case resource consumption is a critical task in software
development. The worst-case analysis (WCA) problem is an optimization-based
abstraction of this task. Fuzzing and symbolic execution are widely used
techniques for addressing the WCA problem. However, improving code coverage in
fuzzing or managing path explosion in symbolic execution within the context of
WCA poses significant challenges. In this paper, we propose PathFuzzing, aiming
to combine the strengths of both techniques to design a WCA method. The key
idea is to transform a program into a symbolic one that takes an execution path
(encoded as a binary string) and interprets the bits as branch decisions.
PathFuzzing then applies evolutionary fuzzing techniques to the transformed
program to search for binary strings that represent satisfiable path conditions
and lead to high resource consumption. We evaluate the performance of
PathFuzzing experimentally on a benchmark suite that consists of prior work's
benchmarks and some added by us. Results show that PathFuzzing generally
outperforms a fuzzing and a symbolic-execution baseline.

### 2. [Modelling Interrelations Between Agile Practices: The Agile Map](http://arxiv.org/pdf/2507.09907v1)

Authors: Thomas Hansper, Kevin Phong Pham, Michael Neumann

Agile methods are defined through guidelines comprising various practices
intended to enable agile ways of working. These guidelines further comprise a
specific set of agile practices aiming to enable teams for an agile way of
working. However, due to its wide-spread use in practice we know that agile
practices are adopted and tailored intensively, which lead to a high variety of
agile practices in terms of their level of detail. Problem: A high variety of
agile practices can be challenging as we do not know how different agile
practices are interrelated with each other. To be more precise, tailoring and
adopting agile practices may lead to the challenge, that the combinatorial use
of several agile practices can only be successful to a limited extent, as
practices support or even require each other for a effective use in practice.
Objective: Our study aims to provide an enabler for this problem. We want to
identify interrelations between agile practices and describe them in a
systematic manner. Contribution: The core contribution of this paper is the
Agile Map, a theoretical model describing relations between agile practices
following a systematic approach aiming to provide an overview of coherences
between agile practices. The model aims to support practitioners in selecting
and combining agile practices in a meaningful way.

### 3. [When Less is More: A systematic review of four-day workweek conceptualizations and their effects on organizational performance](http://arxiv.org/pdf/2507.09911v1)

Authors: Marvin Auf der Landwehr, Julia Topp, Michael Neumann

Context: Agile IT organizations, which are characterized by self-organization
and collaborative social interactions, require motivating, efficient and
flexible work environments to maximize value creation. Compressed work
schedules such as the four-day workweek have evolved into multiple facets over
the last decades and are associated with various benefits for organizations and
their employees. Objective: Our objective in this study is to deepen our
comprehension of the impact of compressed work schedules on the operational
efficacy of IT enterprises, while concurrently developing a comprehensive
framework delineating the intricacies of compressed work schedules.Method: We
conducted a systematic review of available conceptualizations related to
four-day workweek schedules and elaborate on their organizational and social
effects. To cover scientific and practice-oriented literature, our review
combined a systematic literature review and a web content analysis. Results:
Based on the generated insights, we derive a meta-framework that matches
conceptualizations and effects, finally guiding the adoption of compressed work
schedules based on individual managerial prerequisites and circumstances.

### 4. [Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks](http://arxiv.org/pdf/2507.10054v1)

Authors: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

Large Language Models (LLMs) are increasingly used as code assistants, yet
their behavior when explicitly asked to generate insecure code remains poorly
understood. While prior research has focused on unintended vulnerabilities or
adversarial prompting techniques, this study examines a more direct threat
scenario: open-source LLMs generating vulnerable code when prompted either
directly or indirectly. We propose a dual experimental design: (1) Dynamic
Prompting, which systematically varies vulnerability type, user persona, and
directness across structured templates; and (2) Reverse Prompting, which
derives prompts from real vulnerable code samples to assess vulnerability
reproduction accuracy. We evaluate three open-source 7B-parameter models
(Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the
presence of vulnerabilities and the correctness of the generated vulnerability
type. Results show all models frequently produce vulnerable outputs, with Qwen2
achieving highest correctness rates. User persona significantly affects
success, where student personas achieved higher vulnerability rates than
professional roles, while direct prompts were marginally more effective.
Vulnerability reproduction followed an inverted-U pattern with cyclomatic
complexity, peaking at moderate ranges. Our findings expose limitations of
safety mechanisms in open-source models, particularly for seemingly benign
educational requests.

### 5. [LLMShot: Reducing snapshot testing maintenance via LLMs](http://arxiv.org/pdf/2507.10062v1)

Authors: Ergün Batuhan Kaynak, Mayasah Lami, Sahand Moslemi, Anil Koyuncu

Snapshot testing has emerged as a critical technique for UI validation in
modern software development, yet it suffers from substantial maintenance
overhead due to frequent UI changes causing test failures that require manual
inspection to distinguish between genuine regressions and intentional design
changes. This manual triage process becomes increasingly burdensome as
applications evolve, creating a need for automated analysis solutions. This
paper introduces LLMShot, a novel framework that leverages vision-based Large
Language Models to automatically analyze snapshot test failures through
hierarchical classification of UI changes. To evaluate LLMShot's effectiveness,
we developed a comprehensive dataset using a feature-rich iOS application with
configurable feature flags, creating realistic scenarios that produce authentic
snapshot differences representative of real development workflows. Our
evaluation using Gemma3 models demonstrates strong classification performance,
with the 12B variant achieving over 84% recall in identifying failure root
causes while the 4B model offers practical deployment advantages with
acceptable performance for continuous integration environments. However, our
exploration of selective ignore mechanisms revealed significant limitations in
current prompting-based approaches for controllable visual reasoning. LLMShot
represents the first automated approach to semantic snapshot test analysis,
offering developers structured insights that can substantially reduce manual
triage effort and advance toward more intelligent UI testing paradigms.

### 6. [Towards a Framework for Operationalizing the Specification of Trustworthy AI Requirements](http://arxiv.org/pdf/2507.10228v1)

Authors: Hugo Villamizar, Daniel Mendez, Marcos Kalinowski

Growing concerns around the trustworthiness of AI-enabled systems highlight
the role of requirements engineering (RE) in addressing emergent,
context-dependent properties that are difficult to specify without structured
approaches. In this short vision paper, we propose the integration of two
complementary approaches: AMDiRE, an artefact-based approach for RE, and
PerSpecML, a perspective-based method designed to support the elicitation,
analysis, and specification of machine learning (ML)-enabled systems. AMDiRE
provides a structured, artefact-centric, process-agnostic methodology and
templates that promote consistency and traceability in the results; however, it
is primarily oriented toward deterministic systems. PerSpecML, in turn,
introduces multi-perspective guidance to uncover concerns arising from the
data-driven and non-deterministic behavior of ML-enabled systems. We envision a
pathway to operationalize trustworthiness-related requirements, bridging
stakeholder-driven concerns and structured artefact models. We conclude by
outlining key research directions and open challenges to be discussed with the
RE community.

### 7. [An Empirical Study of Interaction Bugs in ROS-based Software](http://arxiv.org/pdf/2507.10235v1)

Authors: Zhixiang Chen, Zhuangbin Chen, Xingjie Cai, Wei Li, Zibin Zheng

Modern robotic systems integrate multiple independent software and hardware
components, each responsible for distinct functionalities such as perception,
decision-making, and execution. These components interact extensively to
accomplish complex end-to-end tasks. As a result, the overall system
reliability depends not only on the correctness of individual components, but
also on the correctness of their interactions. Failures often manifest at the
boundaries between components, yet interaction-related reliability issues in
robotics--referred to here as interaction bugs (iBugs)--remain underexplored.
  This work presents an empirical study of iBugs within robotic systems built
using the Robot Operating System (ROS), a widely adopted open-source robotics
framework. A total of 121 iBugs were analyzed across ten actively maintained
and representative ROS projects. The identified iBugs are categorized into
three major types: intra-system iBugs, hardware iBugs, and environmental iBugs,
covering a broad range of interaction scenarios in robotics. The analysis
includes an examination of root causes, fixing strategies, and the impact of
these bugs. Several findingsa are derived that shed light on the nature of
iBugs and suggest directions for improving their prevention and detection.
These insights aim to inform the design of more robust and safer robotic
systems.

### 8. [Helveg: Diagrams for Software Documentation](http://arxiv.org/pdf/2507.10244v1)

Authors: Adam Štěpánek, David Kuťák, Barbora Kozlíková, Jan Byška

Software developers often have to gain an understanding of a codebase. Be it
programmers getting onboarded onto a team project or, for example, developers
striving to grasp an external open-source library. In either case, they
frequently turn to the project's documentation. However, documentation in its
traditional textual form is ill-suited for this kind of high-level exploratory
analysis, since it is immutable from the readers' perspective and thus forces
them to follow a predefined path. We have designed an approach bringing aspects
of software architecture visualization to API reference documentation. It
utilizes a highly interactive node-link diagram with expressive node glyphs and
flexible filtering capabilities, providing a high-level overview of the
codebase as well as details on demand. To test our design, we have implemented
a prototype named Helveg, capable of automatically generating diagrams of C\#
codebases. User testing of Helveg confirmed its potential, but it also revealed
problems with the readability, intuitiveness, and user experience of our tool.
Therefore, in this paper, which is an extended version of our VISSOFT paper
with DOI 10.1109/VISSOFT64034.2024.00012, we address many of these problems
through major changes to the glyph design, means of interaction, and user
interface of the tool. To assess the improvements, this new version of Helveg
was evaluated again with the same group of participants as the previous
version.

### 9. [A Grounded Theory on the Teacher and Student Roles in Pair Programming](http://arxiv.org/pdf/2507.10305v1)

Authors: Linus Ververs, Trang Linh Lam, Janina Berger, Lutz Prechelt

Context: Pair programming is an established (agile) practice and is practiced
throughout the industry. Objective: Understand under what circumstances
knowledge transfer can harm a pair programming session. Method: Grounded Theory
Methodology based on 17 recorded pair programming sessions with 18 developers
from 5 German software companies accompanied, by 6 interviews with different
developers from 4 other German companies. Results: We define the student and
teacher roles to help developers deal with a one-sided knowledge gap. We
describe pitfalls to avoid and develop a grounded theory centered around the
Power Gap in pair programming. Conclusions: Knowledge transfer can be harmful
when developers don't pay attention to their partners needs and desires. If
developers don't pay attention to the Power Gap and keep it in check, Defensive
Behavior may arise that leads to a vicious cycle impacting the knowledge
transfer, the Togetherness and the code quality in a negative way.

### 10. [Self-Admitted GenAI Usage in Open-Source Software](http://arxiv.org/pdf/2507.10422v1)

Authors: Tao Xiao, Youmei Fan, Fabio Calefato, Christoph Treude, Raula Gaikovina Kula, Hideaki Hata, Sebastian Baltes

The widespread adoption of generative AI (GenAI) tools such as GitHub Copilot
and ChatGPT is transforming software development. Since generated source code
is virtually impossible to distinguish from manually written code, their
real-world usage and impact on open-source software development remain poorly
understood. In this paper, we introduce the concept of self-admitted GenAI
usage, that is, developers explicitly referring to the use of GenAI tools for
content creation in software artifacts. Using this concept as a lens to study
how GenAI tools are integrated into open-source software projects, we analyze a
curated sample of more than 250,000 GitHub repositories, identifying 1,292 such
self-admissions across 156 repositories in commit messages, code comments, and
project documentation. Using a mixed methods approach, we derive a taxonomy of
32 tasks, 10 content types, and 11 purposes associated with GenAI usage based
on 284 qualitatively coded mentions. We then analyze 13 documents with policies
and usage guidelines for GenAI tools and conduct a developer survey to uncover
the ethical, legal, and practical concerns behind them. Our findings reveal
that developers actively manage how GenAI is used in their projects,
highlighting the need for project-level transparency, attribution, and quality
control practices in the new era of AI-assisted software development. Finally,
we examine the longitudinal impact of GenAI adoption on code churn in 151
repositories with self-admitted GenAI usage and find no general increase,
contradicting popular narratives on the impact of GenAI on software
development.

### Social and Information Networks

### 1. [Experimental Analysis and Evaluation of Cohesive Subgraph Discovery](http://arxiv.org/pdf/2507.10262v1)

Authors: Dahee Kim, Song Kim, Jeongseon Kim, Junghoon Kim, Kaiyu Feng, Sungsu Lim, Jungeun Kim

Retrieving cohesive subgraphs in networks is a fundamental problem in social
network analysis and graph data management. These subgraphs can be used for
marketing strategies or recommendation systems. Despite the introduction of
numerous models over the years, a systematic comparison of their performance,
especially across varied network configurations, remains unexplored. In this
study, we evaluated various cohesive subgraph models using task-based
evaluations and conducted extensive experimental studies on both synthetic and
real-world networks. Thus, we unveil the characteristics of cohesive subgraph
models, highlighting their efficiency and applicability. Our findings not only
provide a detailed evaluation of current models but also lay the groundwork for
future research by shedding light on the balance between the interpretability
and cohesion of the subgraphs. This research guides the selection of suitable
models for specific analytical needs and applications, providing valuable
insights.

### Systems and Control

### 1. [A Case Study on Data Acquisition Systems: Relevance to Renewable Energy Technologies](http://arxiv.org/pdf/2507.09938v1)

Authors: Chito A. Petilla

Multiple advantages had been identified with the integration of data
acquisition into any existing system configuration and implementation. Using
data acquisition as a support into a monitoring system has not only improved
its overall performance and reliability but also lowered its operational and
maintenance cost because of its real-time data collection from node sensors.
  As renewable energy needs to be sustainable for it to fully support the
energy demand of communities, its management and control still needs to be
improved and enhanced. Smart systems are considered the next generation
technological improvement of any system that exists. It is the prelude to
autonomous systems from industrial applications to home automation. Data
acquisition is only a part of these smart systems that help in the remote
management and control of these devices. Remote monitoring functionality
enhances the operation and reliability which help in making proactive decisions
during critical situations and circumstances.
  Even with data acquisition enhancements, there is still room for improving
its implementation regarding data security and privacy and accuracy of
information being exchanged between nodes. Current technological advancements
have already shown promising results and have widen its utilization spectrum by
covering almost any field of specialization. With increasing implementation and
design complexity that comes with its enhancements, challenges and issues are
also faced that needs to be addressed and considered to mitigate the effects of
such.

### 2. [Efficient RF Chain Selection for MIMO Integrated Sensing and Communications: A Greedy Approach](http://arxiv.org/pdf/2507.09960v1)

Authors: Subin Shin, Seongkyu Jung, Jinseok Choi, Jeonghun Park

In multiple-input multiple-output integrated sensing and communication (MIMO
ISAC) systems, radio frequency chain (i.e., RF chain) selection plays a vital
role in reducing hardware cost, power consumption, and computational
complexity. However, designing an effective RF chain selection strategy is
challenging due to the disparity in performance metrics between communication
and sensing-mutual information (MI) versus beam-pattern mean-squared error
(MSE) or the Cram\'er-Rao lower bound (CRLB). To overcome this, we propose a
low-complexity greedy RF chain selection framework maximizing a unified
MI-based performance metric applicable to both functions. By decomposing the
total MI into individual contributions of each RF chain, we introduce two
approaches: greedy eigen-based selection (GES) and greedy cofactor-based
selection (GCS), which iteratively identify and remove the RF chains with the
lowest contribution. We further extend our framework to beam selection for
beamspace MIMO ISAC systems, introducing diagonal beam selection (DBS) as a
simplified solution. Simulation results show that our proposed methods achieve
near-optimal performance with significantly lower complexity than exhaustive
search, demonstrating their practical effectiveness for MIMO ISAC systems.

### 3. [Hardware test and validation of the angular droop control: Analysis and experiments](http://arxiv.org/pdf/2507.10004v1)

Authors: Taouba Jouini, Jan Wachter, Sophie An, Veit Hagenmeyer

The angular droop control is a grid-forming control strategy that exploits
the idea of power-to-angle droop to achieve exact frequency synchronization
with no stringent separation between primary and secondary frequency control.
In this work, we conduct hardware experiments in the Smart Energy System
Control Laboratory at Karlsruhe Institute of Technology (KIT) to test and
validate the angular droop control for low voltage power grids in two different
test scenarios. First, we verify its grid-forming capabilities after a major
event, e.g., following a blackout, demonstrated via power-to-angle droop
behavior. For this, we propose two implementation schemes that rely either on
direct or indirect actuation of the modulation signal and draw a comparison
between them. Second, we investigate the plug-and-play capabilities, i.e.,
local stability and power sharing for a two-converter system and provide
suitable tuning for the control gains. Our experimental findings illustrate the
usefulness of hardware test and validation for DC/AC converter control, the
practical challenges entailed and the proposed remedies.

### 4. [Survey on Methods for Detection, Classification and Location of Faults in Power Systems Using Artificial Intelligence](http://arxiv.org/pdf/2507.10011v1)

Authors: Juan A. Martinez-Velasco, Alexandre Serrano-Fontova, Ricard Bosch-Tous, Pau Casals-Torrens

Components of electrical power systems are susceptible to failures caused by
lightning strikes, aging or human errors. These faults can cause equipment
damage, affect system reliability, and results in expensive repair costs. As
electric power systems are becoming more complex, traditional protection
methods face limitations and shortcomings. Faults in power systems can occur at
anytime and anywhere, can be caused by a natural disaster or an accident, and
their occurrence can be hardly predicted or avoided; therefore, it is crucial
to accurately estimate the fault location and quickly restore service. The
development of methods capable of accurately detecting, locating and removing
faults is essential (i.e. fast isolation of faults is necessary to maintain the
system stability at transmission levels; accurate and fast detection and
location of faults are essential for increasing reliability and customer
satisfaction at distribution levels). This has motivated the development of new
and more efficient methods. Methods developed to detect and locate faults in
power systems can be divided into two categories, conventional and artificial
intelligence-based techniques. Although the utilization of artificial
intelligence (AI) techniques offer tremendous potential, they are challenging
and time consuming (i.e. many AI techniques require training data for
processing). This paper presents a survey of the application of AI techniques
to fault diagnosis (detection, classification and location of faults) of lines
and cables of power systems at both transmission and distribution levels. The
paper provides a short introduction to AI concepts, a brief summary of the
application of AI techniques to power system analysis and design, and a
discussion on AI-based fault diagnosis methods.

### 5. [A SUMO-Based Digital Twin for Evaluation of Conventional and Electric Vehicle Networks](http://arxiv.org/pdf/2507.10280v1)

Authors: Haomiaomiao Wang, Conor Fennell, Swati Poojary, Mingming Liu

Digital twins are increasingly applied in transportation modelling to
replicate real-world traffic dynamics and evaluate mobility and energy
efficiency. This study presents a SUMO-based digital twin that simulates mixed
ICEV-EV traffic on a major motorway segment, leveraging multi-sensor data
fusion from inductive loops, GPS probes, and toll records. The model is
validated under both complete and partial information scenarios, achieving
93.1% accuracy in average speed estimation and 97.1% in average trip length
estimation. Statistical metrics, including KL Divergence and Wasserstein
Distance, demonstrate strong alignment between simulated and observed traffic
patterns. Furthermore, CO2 emissions were overestimated by only 0.8-2.4%, and
EV power consumption underestimated by 1.0-5.4%, highlighting the model's
robustness even with incomplete vehicle classification information.

### 6. [Improved Sum-of-Squares Stability Verification of Neural-Network-Based Controllers](http://arxiv.org/pdf/2507.10352v1)

Authors: Alvaro Detailleur, Guillaume Ducard, Christopher Onder

This work presents several improvements to the closed-loop stability
verification framework using semialgebraic sets and convex semidefinite
programming to examine neural-network-based control systems regulating
nonlinear dynamical systems. First, the utility of the framework is greatly
expanded: two semialgebraic functions mimicking common, smooth activation
functions are presented and compatibility with control systems incorporating
Recurrent Equilibrium Networks (RENs) and thereby Recurrent Neural Networks
(RNNs) is established. Second, the validity of the framework's state-of-the-art
stability analyses is established via an alternate proof. Third, based on this
proof, two new optimization problems simplifying the analysis of local
stability properties are presented. To simplify the analysis of a closed-loop
system's Region of Attraction (RoA), the first problem explicitly parameterizes
a class of candidate Lyapunov functions larger than in previous works. The
second problem utilizes the unique guarantees available under the condition of
invariance to further expand the set of candidate Lyapunov functions and
directly determine whether an invariant set forms part of the system's RoA.
These contributions are successfully demonstrated in two numerical examples and
suggestions for future research are provided.

### 7. [UavNetSim-v1: A Python-based Simulation Platform for UAV Communication Networks](http://arxiv.org/pdf/2507.09852v1)

Authors: Zihao Zhou, Zipeng Dai, Linyi Huang, Cui Yang, Youjun Xiang, Jie Tang, Kai-kit Wong

In unmanned aerial vehicle (UAV) networks, communication protocols and
algorithms are essential for cooperation and collaboration between UAVs.
Simulation provides a cost-effective solution for prototyping, debugging, and
analyzing protocols and algorithms, avoiding the prohibitive expenses of field
experiments. In this paper, we present ``UavNetSim-v1'', an open-source
Python-based simulation platform designed for rapid development, testing, and
evaluating the protocols and algorithms in UAV networks. ``UavNetSim-v1''
provides most of the functionalities developers may need, including
routing/medium access control (MAC) protocols, topology control algorithms and
mobility/energy models, while maintaining ease of use. Furthermore, the
platform supports comprehensive performance evaluation and features an
interactive visualization interface for in-depth algorithm analysis. In short,
``UavNetSim-v1'' lends itself to both rapid prototyping and educational
purposes, and can serve as a lightweight yet powerful alternative to mature
network simulators for UAV communication research.

### 8. [Optimal Design of Satellite Constellation Configurations with Mixed Integer Linear Programming](http://arxiv.org/pdf/2507.09855v1)

Authors: David O. Williams Rogers, Dongshik Won, Dongwook Koh, Kyungwoo Hong, Hang Woon Lee

Designing satellite constellation systems involves complex multidisciplinary
optimization in which coverage serves as a primary driver of overall system
cost and performance. Among the various design considerations, constellation
configuration -- how satellites are placed and distributed in space relative to
each other -- predominantly determines the resulting coverage. In constellation
configuration design, coverage can be considered either as an objective or a
constraint, driven by mission objectives. State-of-the-art literature addresses
each situation on a case-by-case basis, applying a unique set of assumptions,
modeling, and solution methods. Although such a problem-based methodology is
valuable, users often face implementation challenges when performing trade-off
studies across different mission scenarios, as each scenario must be handled
distinctly. In response, we propose a unifying framework consisting of five
mixed-integer linear program formulations that are of practical significance,
extensible to more complex mission narratives using additional constraints, and
capable of obtaining provably optimal constellation configurations. It can
handle various metrics and mission scenarios, such as percent coverage, average
or maximum revisit times, fixed number of satellites, spatiotemporally varying
coverage requirements, and ground-, aerial-, or space-based, static or mobile
targets. The paper presents several add-ons, case studies, and comparative
analyses to demonstrate the versatility of the proposed framework.

### 9. [Predictive & Trust-based Multi-Agent Coordination](http://arxiv.org/pdf/2507.09997v1)

Authors: Venkatraman Renganathan, Sabyasachi Mondal, Antonios Tsourdos

This paper presents a trust-based predictive multi-agent consensus protocol
that analyses neighbours' anticipation data and makes coordination decisions.
Agents in the network share their future predicted data over a finite
look-ahead horizon with their neighbours and update their predictions in a
rolling-horizon fashion. The prediction data is then used by agents to learn
both the trust and the commitment traits exhibited by their neighbours over
time. The proposed protocol is named as the Anticipatory Distributed
Coordination (ADC) protocol. Lyapunov theory-based agreement convergence
between agents is provided, followed by demonstrations using numerical
simulations.

### 10. [Probabilistic Robustness in the Gap Metric](http://arxiv.org/pdf/2507.10010v1)

Authors: Venkatraman Renganathan

Uncertainties influencing the dynamical systems pose a significant challenge
in estimating the achievable performance of a controller aiming to control such
uncertain systems. When the uncertainties are of stochastic nature, obtaining
hard guarantees for the robustness of a controller aiming to hedge against the
uncertainty is not possible. This issue set the platform for the development of
probabilistic robust control approaches. In this work, we utilise the gap
metric between the known nominal model and the unknown perturbed model of the
uncertain system as a tool to gauge the robustness of a controller and
formulate the gap as a random variable in the setting with stochastic
uncertainties. Main results of this paper includes giving probabilistic bound
on the gap exceeding a known threshold followed by bounds on the expected gap
value and probabilistic robust stability in terms of the gap metric. Further,
we also provide a probabilistic controller performance certification under gap
uncertainty and probabilistic guarantee on the achievable
$\mathcal{H}_{\infty}$ robustness. Numerical simulations are provided at many
places to demonstrate the proposed approach.

### Machine Learning (Statistics Category)

### 1. [Solving dynamic portfolio selection problems via score-based diffusion models](http://arxiv.org/pdf/2507.09916v1)

Authors: Ahmad Aghapour, Erhan Bayraktar, Fengyi Yuan

In this paper, we tackle the dynamic mean-variance portfolio selection
problem in a {\it model-free} manner, based on (generative) diffusion models.
We propose using data sampled from the real model $\mathcal P$ (which is
unknown) with limited size to train a generative model $\mathcal Q$ (from which
we can easily and adequately sample). With adaptive training and sampling
methods that are tailor-made for time series data, we obtain quantification
bounds between $\mathcal P$ and $\mathcal Q$ in terms of the adapted
Wasserstein metric $\mathcal A W_2$. Importantly, the proposed adapted sampling
method also facilitates {\it conditional sampling}. In the second part of this
paper, we provide the stability of the mean-variance portfolio optimization
problems in $\mathcal A W _2$. Then, combined with the error bounds and the
stability result, we propose a policy gradient algorithm based on the
generative environment, in which our innovative adapted sampling method
provides approximate scenario generators. We illustrate the performance of our
algorithm on both simulated and real data. For real data, the algorithm based
on the generative environment produces portfolios that beat several important
baselines, including the Markowitz portfolio, the equal weight (naive)
portfolio, and S\&P 500.

### 2. [Towards High Supervised Learning Utility Training Data Generation: Data Pruning and Column Reordering](http://arxiv.org/pdf/2507.10088v1)

Authors: Tung Sum Thomas Kwok, Zeyong Zhang, Chi-Hua Wang, Guang Cheng

Tabular data synthesis for supervised learning ('SL') model training is
gaining popularity in industries such as healthcare, finance, and retail.
Despite the progress made in tabular data generators, models trained with
synthetic data often underperform compared to those trained with original data.
This low SL utility of synthetic data stems from class imbalance exaggeration
and SL data relationship overlooked by tabular generator. To address these
challenges, we draw inspirations from techniques in emerging data-centric
artificial intelligence and elucidate Pruning and ReOrdering ('PRRO'), a novel
pipeline that integrates data-centric techniques into tabular data synthesis.
PRRO incorporates data pruning to guide the table generator towards
observations with high signal-to-noise ratio, ensuring that the class
distribution of synthetic data closely matches that of the original data.
Besides, PRRO employs a column reordering algorithm to align the data modeling
structure of generators with that of SL models. These two modules enable PRRO
to optimize SL utility of synthetic data. Empirical experiments on 22 public
datasets show that synthetic data generated using PRRO enhances predictive
performance compared to data generated without PRRO. Specifically, synthetic
replacement of original data yields an average improvement of 26.74% and up to
871.46% improvement using PRRO, while synthetic appendant to original data
results with PRRO-generated data results in an average improvement of 6.13% and
up to 200.32%. Furthermore, experiments on six highly imbalanced datasets show
that PRRO enables the generator to produce synthetic data with a class
distribution that resembles the original data more closely, achieving a
similarity improvement of 43%. Through PRRO, we foster a seamless integration
of data synthesis to subsequent SL prediction, promoting quality and accessible
data analysis.

### 3. [Simulating Biases for Interpretable Fairness in Offline and Online Classifiers](http://arxiv.org/pdf/2507.10154v1)

Authors: Ricardo Inácio, Zafeiris Kokkinogenis, Vitor Cerqueira, Carlos Soares

Predictive models often reinforce biases which were originally embedded in
their training data, through skewed decisions. In such cases, mitigation
methods are critical to ensure that, regardless of the prevailing disparities,
model outcomes are adjusted to be fair. To assess this, datasets could be
systematically generated with specific biases, to train machine learning
classifiers. Then, predictive outcomes could aid in the understanding of this
bias embedding process. Hence, an agent-based model (ABM), depicting a loan
application process that represents various systemic biases across two
demographic groups, was developed to produce synthetic datasets. Then, by
applying classifiers trained on them to predict loan outcomes, we can assess
how biased data leads to unfairness. This highlights a main contribution of
this work: a framework for synthetic dataset generation with controllable bias
injection. We also contribute with a novel explainability technique, which
shows how mitigations affect the way classifiers leverage data features, via
second-order Shapley values. In experiments, both offline and online learning
approaches are employed. Mitigations are applied at different stages of the
modelling pipeline, such as during pre-processing and in-processing.

### 4. [Non-exchangeable Conformal Prediction with Optimal Transport: Tackling Distribution Shifts with Unlabeled Data](http://arxiv.org/pdf/2507.10425v1)

Authors: Alvaro H. C. Correia, Christos Louizos

Conformal prediction is a distribution-free uncertainty quantification method
that has gained popularity in the machine learning community due to its
finite-sample guarantees and ease of use. Its most common variant, dubbed split
conformal prediction, is also computationally efficient as it boils down to
collecting statistics of the model predictions on some calibration data not yet
seen by the model. Nonetheless, these guarantees only hold if the calibration
and test data are exchangeable, a condition that is difficult to verify and
often violated in practice due to so-called distribution shifts. The literature
is rife with methods to mitigate the loss in coverage in this non-exchangeable
setting, but these methods require some prior information on the type of
distribution shift to be expected at test time. In this work, we study this
problem via a new perspective, through the lens of optimal transport, and show
that it is possible to estimate the loss in coverage and mitigate it in case of
distribution shift.

### 5. [Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](http://arxiv.org/pdf/2507.09846v1)

Authors: Minhak Song, Beomhan Baek, Kwangjun Ahn, Chulhee Yun

As both model and dataset sizes continue to scale rapidly, conventional
pretraining strategies with fixed compute budgets-such as cosine learning rate
schedules-are increasingly inadequate for large-scale training. Recent
alternatives, including warmup-stable-decay (WSD) schedules and weight
averaging, offer greater flexibility. However, WSD relies on explicit decay
phases to track progress, while weight averaging addresses this limitation at
the cost of additional memory. In search of a more principled and scalable
alternative, we revisit the Schedule-Free (SF) method [Defazio et al., 2024],
which has shown strong empirical performance across diverse settings. We show
that SF-AdamW effectively navigates the "river" structure of the loss landscape
without decay phases or auxiliary averaging, making it particularly suitable
for continuously scaling training workloads. To understand this behavior, we
conduct a theoretical and empirical analysis of SF dynamics, revealing that it
implicitly performs weight averaging without memory overhead. Guided by this
analysis, we propose a refined variant of SF that improves robustness to
momentum and performs better under large batch sizes, addressing key
limitations of the original method. Together, these results establish SF as a
practical, scalable, and theoretically grounded approach for language model
training.

### 6. [NeuTSFlow: Modeling Continuous Functions Behind Time Series Forecasting](http://arxiv.org/pdf/2507.09888v1)

Authors: Huibo Xu, Likang Wu, Xianquan Wang, Haoning Dang, Chun-Wun Cheng, Angelica I Aviles-Rivero, Qi Liu

Time series forecasting is a fundamental task with broad applications, yet
conventional methods often treat data as discrete sequences, overlooking their
origin as noisy samples of continuous processes. Crucially, discrete noisy
observations cannot uniquely determine a continuous function; instead, they
correspond to a family of plausible functions. Mathematically, time series can
be viewed as noisy observations of a continuous function family governed by a
shared probability measure. Thus, the forecasting task can be framed as
learning the transition from the historical function family to the future
function family. This reframing introduces two key challenges: (1) How can we
leverage discrete historical and future observations to learn the relationships
between their underlying continuous functions? (2) How can we model the
transition path in function space from the historical function family to the
future function family? To address these challenges, we propose NeuTSFlow, a
novel framework that leverages Neural Operators to facilitate flow matching for
learning path of measure between historical and future function families. By
parameterizing the velocity field of the flow in infinite-dimensional function
spaces, NeuTSFlow moves beyond traditional methods that focus on dependencies
at discrete points, directly modeling function-level features instead.
Experiments on diverse forecasting tasks demonstrate NeuTSFlow's superior
accuracy and robustness, validating the effectiveness of the function-family
perspective.

### 7. [Statistical Inference for Conditional Group Distributionally Robust Optimization with Cross-Entropy Loss](http://arxiv.org/pdf/2507.09905v1)

Authors: Zijian Guo, Zhenyu Wang, Yifan Hu, Francis Bach

In multi-source learning with discrete labels, distributional heterogeneity
across domains poses a central challenge to developing predictive models that
transfer reliably to unseen domains. We study multi-source unsupervised domain
adaptation, where labeled data are drawn from multiple source domains and only
unlabeled data from a target domain. To address potential distribution shifts,
we propose a novel Conditional Group Distributionally Robust Optimization
(CG-DRO) framework that learns a classifier by minimizing the worst-case
cross-entropy loss over the convex combinations of the conditional outcome
distributions from the sources. To solve the resulting minimax problem, we
develop an efficient Mirror Prox algorithm, where we employ a double machine
learning procedure to estimate the risk function. This ensures that the errors
of the machine learning estimators for the nuisance models enter only at
higher-order rates, thereby preserving statistical efficiency under covariate
shift. We establish fast statistical convergence rates for the estimator by
constructing two surrogate minimax optimization problems that serve as
theoretical bridges. A distinguishing challenge for CG-DRO is the emergence of
nonstandard asymptotics: the empirical estimator may fail to converge to a
standard limiting distribution due to boundary effects and system instability.
To address this, we introduce a perturbation-based inference procedure that
enables uniformly valid inference, including confidence interval construction
and hypothesis testing.

### 8. [Sampling-Based Estimation of Jaccard Containment and Similarity](http://arxiv.org/pdf/2507.10019v1)

Authors: Pranav Joshi

This paper addresses the problem of estimating the containment and similarity
between two sets using only random samples from each set, without relying on
sketches or full data access. The study introduces a binomial model for
predicting the overlap between samples, demonstrating that it is both accurate
and practical when sample sizes are small compared to the original sets. The
paper compares this model to previous approaches and shows that it provides
better estimates under the considered conditions. It also analyzes the
statistical properties of the estimator, including error bounds and sample size
requirements needed to achieve a desired level of accuracy and confidence. The
framework is extended to estimate set similarity, and the paper provides
guidance for applying these methods in large scale data systems where only
partial or sampled data is available.

### 9. [Wavelet-Enhanced Neural ODE and Graph Attention for Interpretable Energy Forecasting](http://arxiv.org/pdf/2507.10132v1)

Authors: Usman Gani Joy

Accurate forecasting of energy demand and supply is critical for optimizing
sustainable energy systems, yet it is challenged by the variability of
renewable sources and dynamic consumption patterns. This paper introduces a
neural framework that integrates continuous-time Neural Ordinary Differential
Equations (Neural ODEs), graph attention, multi-resolution wavelet
transformations, and adaptive learning of frequencies to address the issues of
time series prediction. The model employs a robust ODE solver, using the
Runge-Kutta method, paired with graph-based attention and residual connections
to better understand both structural and temporal patterns. Through
wavelet-based feature extraction and adaptive frequency modulation, it adeptly
captures and models diverse, multi-scale temporal dynamics. When evaluated
across seven diverse datasets: ETTh1, ETTh2, ETTm1, ETTm2 (electricity
transformer temperature), and Waste, Solar, and Hydro (renewable energy), this
architecture consistently outperforms state-of-the-art baselines in various
forecasting metrics, proving its robustness in capturing complex temporal
dependencies. Furthermore, the model enhances interpretability through SHAP
analysis, making it suitable for sustainable energy applications.

### 10. [MF-GLaM: A multifidelity stochastic emulator using generalized lambda models](http://arxiv.org/pdf/2507.10303v1)

Authors: K. Giannoukou, X. Zhu, S. Marelli, B. Sudret

Stochastic simulators exhibit intrinsic stochasticity due to unobservable,
uncontrollable, or unmodeled input variables, resulting in random outputs even
at fixed input conditions. Such simulators are common across various scientific
disciplines; however, emulating their entire conditional probability
distribution is challenging, as it is a task traditional deterministic
surrogate modeling techniques are not designed for. Additionally, accurately
characterizing the response distribution can require prohibitively large
datasets, especially for computationally expensive high-fidelity (HF)
simulators. When lower-fidelity (LF) stochastic simulators are available, they
can enhance limited HF information within a multifidelity surrogate modeling
(MFSM) framework. While MFSM techniques are well-established for deterministic
settings, constructing multifidelity emulators to predict the full conditional
response distribution of stochastic simulators remains a challenge. In this
paper, we propose multifidelity generalized lambda models (MF-GLaMs) to
efficiently emulate the conditional response distribution of HF stochastic
simulators by exploiting data from LF stochastic simulators. Our approach
builds upon the generalized lambda model (GLaM), which represents the
conditional distribution at each input by a flexible, four-parameter
generalized lambda distribution. MF-GLaMs are non-intrusive, requiring no
access to the internal stochasticity of the simulators nor multiple
replications of the same input values. We demonstrate the efficacy of MF-GLaM
through synthetic examples of increasing complexity and a realistic earthquake
application. Results show that MF-GLaMs can achieve improved accuracy at the
same cost as single-fidelity GLaMs, or comparable performance at significantly
reduced cost.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-15 PST.

### 1. [Predicting New York Heart Association (NYHA) heart failure classification from medical student notes following simulated patient encounters](https://www.nature.com/articles/s41598-025-10179-8)

Authors: Ishan R. Perera et al.

### 2. [BaliMask3D dataset for 3D completion and reconstruction of traditional Balinese masks](https://www.nature.com/articles/s41597-025-05505-8)

Authors: I Nyoman Tri Anindia Putra et al.

### 3. [Adaptive conflict resolution for IoT transactions: A reinforcement learning-based hybrid validation protocol](https://www.nature.com/articles/s41598-025-09698-1)

Authors: Mohammad A. Al Khaldy et al.

### 4. [Scene as Occupancy and Reconstruction: A Comprehensive Dataset for Unstructured Scene Understanding](https://www.nature.com/articles/s41597-025-05532-5)

Authors: Long Chen et al.

### 5. [Hyperbolic geometry enhanced feature filtering network for industrial anomaly detection](https://www.nature.com/articles/s41598-025-07550-0)

Authors: Yanjun Feng et al.

### 6. [A hybrid fog-edge computing architecture for real-time health monitoring in IoMT systems with optimized latency and threat resilience](https://www.nature.com/articles/s41598-025-09696-3)

Authors: Umar Islam et al.

### 7. [Research on multi-branch residual connection spectrum image classification based on attention mechanism](https://www.nature.com/articles/s41598-025-11283-5)

Authors: Zhong Xiaohui et al.

### 8. [A hybrid framework of generative deep learning for antiviral peptide discovery](https://www.nature.com/articles/s41598-025-11328-9)

Authors: Huynh Anh Duy et al.

### 9. [A comparative study and simple baseline for travel time prediction](https://www.nature.com/articles/s41598-025-02303-5)

Authors: Chuang-Chieh Lin et al.

### 10. [Utilizing CBNet to effectively address and combat cyberbullying among university students on social media platforms](https://www.nature.com/articles/s41598-025-09091-y)

Authors: Irshad Ahmed Abbasi et al.

