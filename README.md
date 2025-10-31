# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-30 17:00:25.726425 PST.

### Artificial Intelligence

### 1. [Reasoning-Aware GRPO using Process Mining](http://arxiv.org/pdf/2510.25065v1)

Authors: Taekhyun Park, Yongjae Lee, Hyerim Bae

Reinforcement learning (RL)-based post-training has been crucial for enabling
multi-step reasoning in large reasoning models (LRMs), yet current reward
schemes are typically outcome-centric. We propose PM4GRPO, a reasoning-aware
Group Relative Policy Optimization (GRPO) that augments standard answer/format
rewards with signals over the reasoning procedure. To this end, process mining
techniques are utilized to compute a scalar conformance reward that measures
how closely a policy model's reasoning aligns with the pretrained teacher
model. The empirical results on five benchmarks demonstrate that PM4GRPO
significantly outperforms existing methodologies for GRPO-based post-training.
These results highlight that leveraging process mining for reasoning-aware GRPO
effectively enhances the reasoning capabilities of policy models.

### 2. [H3M-SSMoEs: Hypergraph-based Multimodal Learning with LLM Reasoning and Style-Structured Mixture of Experts](http://arxiv.org/pdf/2510.25091v1)

Authors: Peilin Tan, Liang Xie, Churan Zhi, Dian Tu, Chuanqi Shi

Stock movement prediction remains fundamentally challenging due to complex
temporal dependencies, heterogeneous modalities, and dynamically evolving
inter-stock relationships. Existing approaches often fail to unify structural,
semantic, and regime-adaptive modeling within a scalable framework. This work
introduces H3M-SSMoEs, a novel Hypergraph-based MultiModal architecture with
LLM reasoning and Style-Structured Mixture of Experts, integrating three key
innovations: (1) a Multi-Context Multimodal Hypergraph that hierarchically
captures fine-grained spatiotemporal dynamics via a Local Context Hypergraph
(LCH) and persistent inter-stock dependencies through a Global Context
Hypergraph (GCH), employing shared cross-modal hyperedges and Jensen-Shannon
Divergence weighting mechanism for adaptive relational learning and cross-modal
alignment; (2) a LLM-enhanced reasoning module, which leverages a frozen large
language model with lightweight adapters to semantically fuse and align
quantitative and textual modalities, enriching representations with
domain-specific financial knowledge; and (3) a Style-Structured Mixture of
Experts (SSMoEs) that combines shared market experts and industry-specialized
experts, each parameterized by learnable style vectors enabling regime-aware
specialization under sparse activation. Extensive experiments on three major
stock markets demonstrate that H3M-SSMoEs surpasses state-of-the-art methods in
both superior predictive accuracy and investment performance, while exhibiting
effective risk control. Datasets, source code, and model weights are available
at our GitHub repository: https://github.com/PeilinTime/H3M-SSMoEs.

### 3. [Agentic Moderation: Multi-Agent Design for Safer Vision-Language Models](http://arxiv.org/pdf/2510.25179v1)

Authors: Juan Ren, Mark Dras, Usman Naseem

Agentic methods have emerged as a powerful and autonomous paradigm that
enhances reasoning, collaboration, and adaptive control, enabling systems to
coordinate and independently solve complex tasks. We extend this paradigm to
safety alignment by introducing Agentic Moderation, a model-agnostic framework
that leverages specialised agents to defend multimodal systems against
jailbreak attacks. Unlike prior approaches that apply as a static layer over
inputs or outputs and provide only binary classifications (safe or unsafe), our
method integrates dynamic, cooperative agents, including Shield, Responder,
Evaluator, and Reflector, to achieve context-aware and interpretable
moderation. Extensive experiments across five datasets and four representative
Large Vision-Language Models (LVLMs) demonstrate that our approach reduces the
Attack Success Rate (ASR) by 7-19%, maintains a stable Non-Following Rate (NF),
and improves the Refusal Rate (RR) by 4-20%, achieving robust, interpretable,
and well-balanced safety performance. By harnessing the flexibility and
reasoning capacity of agentic architectures, Agentic Moderation provides
modular, scalable, and fine-grained safety enforcement, highlighting the
broader potential of agentic systems as a foundation for automated safety
governance.

### 4. [Energy-Efficient Autonomous Driving with Adaptive Perception and Robust Decision](http://arxiv.org/pdf/2510.25205v1)

Authors: Yuyang Xia, Zibo Liang, Liwei Deng, Yan Zhao, Han Su, Kai Zheng

Autonomous driving is an emerging technology that is expected to bring
significant social, economic, and environmental benefits. However, these
benefits come with rising energy consumption by computation engines, limiting
the driving range of vehicles, especially electric ones. Perception computing
is typically the most power-intensive component, as it relies on largescale
deep learning models to extract environmental features. Recently, numerous
studies have employed model compression techniques, such as sparsification,
quantization, and distillation, to reduce computational consumption. However,
these methods often result in either a substantial model size or a significant
drop in perception accuracy compared to high-computation models. To address
these challenges, we propose an energy-efficient autonomous driving framework,
called EneAD. In the adaptive perception module, a perception optimization
strategy is designed from the perspective of data management and tuning.
Firstly, we manage multiple perception models with different computational
consumption and adjust the execution framerate dynamically. Then, we define
them as knobs and design a transferable tuning method based on Bayesian
optimization to identify promising knob values that achieve low computation
while maintaining desired accuracy. To adaptively switch the knob values in
various traffic scenarios, a lightweight classification model is proposed to
distinguish the perception difficulty in different scenarios. In the robust
decision module, we propose a decision model based on reinforcement learning
and design a regularization term to enhance driving stability in the face of
perturbed perception results. Extensive experiments evidence the superiority of
our framework in both energy consumption and driving performance. EneAD can
reduce perception consumption by 1.9x to 3.5x and thus improve driving range by
3.9% to 8.5%

### 5. [FELA: A Multi-Agent Evolutionary System for Feature Engineering of Industrial Event Log Data](http://arxiv.org/pdf/2510.25223v1)

Authors: Kun ouyang, Haoyu Wang, Dong Fang

Event log data, recording fine-grained user actions and system events,
represent one of the most valuable assets for modern digital services. However,
the complexity and heterogeneity of industrial event logs--characterized by
large scale, high dimensionality, diverse data types, and intricate temporal or
relational structures--make feature engineering extremely challenging. Existing
automatic feature engineering approaches, such as AutoML or genetic methods,
often suffer from limited explainability, rigid predefined operations, and poor
adaptability to complicated heterogeneous data. In this paper, we propose FELA
(Feature Engineering LLM Agents), a multi-agent evolutionary system that
autonomously extracts meaningful and high-performing features from complex
industrial event log data. FELA integrates the reasoning and coding
capabilities of large language models (LLMs) with an insight-guided
self-evolution paradigm. Specifically, FELA employs specialized agents--Idea
Agents, Code Agents, and Critic Agents--to collaboratively generate, validate,
and implement novel feature ideas. An Evaluation Agent summarizes feedback and
updates a hierarchical knowledge base and dual-memory system to enable
continual improvement. Moreover, FELA introduces an agentic evolution
algorithm, combining reinforcement learning and genetic algorithm principles to
balance exploration and exploitation across the idea space. Extensive
experiments on real industrial datasets demonstrate that FELA can generate
explainable, domain-relevant features that significantly improve model
performance while reducing manual effort. Our results highlight the potential
of LLM-based multi-agent systems as a general framework for automated,
interpretable, and adaptive feature engineering in complex real-world
environments.

### 6. [Grouping Nodes With Known Value Differences: A Lossless UCT-based Abstraction Algorithm](http://arxiv.org/pdf/2510.25388v1)

Authors: Robin Schmöcker, Alexander Dockhorn, Bodo Rosenhahn

A core challenge of Monte Carlo Tree Search (MCTS) is its sample efficiency,
which can be improved by grouping state-action pairs and using their aggregate
statistics instead of single-node statistics. On the Go Abstractions in Upper
Confidence bounds applied to Trees (OGA-UCT) is the state-of-the-art MCTS
abstraction algorithm for deterministic environments that builds its
abstraction using the Abstractions of State-Action Pairs (ASAP) framework,
which aims to detect states and state-action pairs with the same value under
optimal play by analysing the search graph. ASAP, however, requires two
state-action pairs to have the same immediate reward, which is a rigid
condition that limits the number of abstractions that can be found and thereby
the sample efficiency. In this paper, we break with the paradigm of grouping
value-equivalent states or state-action pairs and instead group states and
state-action pairs with possibly different values as long as the difference
between their values can be inferred. We call this abstraction framework Known
Value Difference Abstractions (KVDA), which infers the value differences by
analysis of the immediate rewards and modifies OGA-UCT to use this framework
instead. The modification is called KVDA-UCT, which detects significantly more
abstractions than OGA-UCT, introduces no additional parameter, and outperforms
OGA-UCT on a variety of deterministic environments and parameter settings.

### 7. [Multi-Objective Search: Algorithms, Applications, and Emerging Directions](http://arxiv.org/pdf/2510.25504v1)

Authors: Oren Salzman, Carlos Hernández Ulloa, Ariel Felner, Sven Koenig

Multi-objective search (MOS) has emerged as a unifying framework for planning
and decision-making problems where multiple, often conflicting, criteria must
be balanced. While the problem has been studied for decades, recent years have
seen renewed interest in the topic across AI applications such as robotics,
transportation, and operations research, reflecting the reality that real-world
systems rarely optimize a single measure. This paper surveys developments in
MOS while highlighting cross-disciplinary opportunities, and outlines open
challenges that define the emerging frontier of MOS

### 8. [MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL](http://arxiv.org/pdf/2510.25510v1)

Authors: Zekun Xu, Siyu Xia, Chuhuai Yue, Jiajun Chai, Mingxue Tian, Xiaohan Wang, Wei Lin, Haoxuan Li, Guojun Yin

As large language models (LLMs) are increasingly used in Text-to-SQL tasks,
Reinforcement Learning (RL) has become a common method for improving
performance. Existing methods primarily rely on static execution feedback,
which restricts real-time error correction. However, integrating multi-turn
tool invocation along with dynamic feedback could significantly improve
adaptability and robustness, ultimately enhancing model performance. To address
these issues, we propose MTIR-SQL, an innovative Multi-turn Tool-Integrated
Reasoning reinforcement learning framework for Text-to-SQL. Our approach
introduces an execution-aware multi-turn reasoning paradigm that seamlessly
incorporates database execution feedback at each reasoning step, enabling
context-sensitive query generation and progressive refinement throughout the
reasoning process. The framework extends the GRPO algorithm to accommodate
complex multi-turn interaction scenarios. Considering the training instability
characteristics of MTIR and the potential for significant Deviation of model
distribution from the initial model, we enhance the GRPO algorithm by adding a
trajectory filtering mechanism and removing KL loss constraints. Experimental
results demonstrate that MTIR-SQL, with 4B parameters, achieves \textbf{64.4}\%
accuracy in the BIRD Dev and 84.6% execution accuracy in the SPIDER Dev,
significantly outperforming existing approaches.

### 9. [Predicate Renaming via Large Language Models](http://arxiv.org/pdf/2510.25517v1)

Authors: Elisabetta Gentili, Tony Ribeiro, Fabrizio Riguzzi, Katsumi Inoue

In this paper, we address the problem of giving names to predicates in logic
rules using Large Language Models (LLMs). In the context of Inductive Logic
Programming, various rule generation methods produce rules containing unnamed
predicates, with Predicate Invention being a key example. This hinders the
readability, interpretability, and reusability of the logic theory. Leveraging
recent advancements in LLMs development, we explore their ability to process
natural language and code to provide semantically meaningful suggestions for
giving a name to unnamed predicates. The evaluation of our approach on some
hand-crafted logic rules indicates that LLMs hold potential for this task.

### 10. [Retrieval Augmented Generation (RAG) for Fintech: Agentic Design and Evaluation](http://arxiv.org/pdf/2510.25518v1)

Authors: Thomas Cook, Richard Osuagwu, Liman Tsatiashvili, Vrynsia Vrynsia, Koustav Ghosal, Maraim Masoud, Riccardo Mattivi

Retrieval-Augmented Generation (RAG) systems often face limitations in
specialized domains such as fintech, where domain-specific ontologies, dense
terminology, and acronyms complicate effective retrieval and synthesis. This
paper introduces an agentic RAG architecture designed to address these
challenges through a modular pipeline of specialized agents. The proposed
system supports intelligent query reformulation, iterative sub-query
decomposition guided by keyphrase extraction, contextual acronym resolution,
and cross-encoder-based context re-ranking. We evaluate our approach against a
standard RAG baseline using a curated dataset of 85 question--answer--reference
triples derived from an enterprise fintech knowledge base. Experimental results
demonstrate that the agentic RAG system outperforms the baseline in retrieval
precision and relevance, albeit with increased latency. These findings suggest
that structured, multi-agent methodologies offer a promising direction for
enhancing retrieval robustness in complex, domain-specific settings.

### Hardware Architecture

### 1. [DIRC-RAG: Accelerating Edge RAG with Robust High-Density and High-Loading-Bandwidth Digital In-ReRAM Computation](http://arxiv.org/pdf/2510.25278v1)

Authors: Kunming Shao, Zhipeng Liao, Jiangnan Yu, Liang Zhao, Qiwei Li, Xijie Huang, Jingyu He, Fengshi Tian, Yi Zou, Xiaomeng Wang, Tim Kwang-Ting Cheng, Chi-Ying Tsui

Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
integrating external knowledge retrieval but faces challenges on edge devices
due to high storage, energy, and latency demands. Computing-in-Memory (CIM)
offers a promising solution by storing document embeddings in CIM macros and
enabling in-situ parallel retrievals but is constrained by either low memory
density or limited computational accuracy. To address these challenges, we
present DIRCRAG, a novel edge RAG acceleration architecture leveraging Digital
In-ReRAM Computation (DIRC). DIRC integrates a high-density multi-level ReRAM
subarray with an SRAM cell, utilizing SRAM and differential sensing for robust
ReRAM readout and digital multiply-accumulate (MAC) operations. By storing all
document embeddings within the CIM macro, DIRC achieves ultra-low-power,
single-cycle data loading, substantially reducing both energy consumption and
latency compared to offchip DRAM. A query-stationary (QS) dataflow is supported
for RAG tasks, minimizing on-chip data movement and reducing SRAM buffer
requirements. We introduce error optimization for the DIRC ReRAM-SRAM cell by
extracting the bit-wise spatial error distribution of the ReRAM subarray and
applying targeted bit-wise data remapping. An error detection circuit is also
implemented to enhance readout resilience against deviceand circuit-level
variations. Simulation results demonstrate that DIRC-RAG under TSMC40nm process
achieves an on-chip non-volatile memory density of 5.18Mb/mm2 and a throughput
of 131 TOPS. It delivers a 4MB retrieval latency of 5.6{\mu}s/query and an
energy consumption of 0.956{\mu}J/query, while maintaining the retrieval
precision.

### 2. [Accurate Leakage Speculation for Quantum Error Correction](http://arxiv.org/pdf/2510.25661v1)

Authors: Chaithanya Naik Mude, Swamit Tannu

Quantum Error Correction (QEC) protects qubits against bit- and phase-flip
errors in the |0> or |1> subspace, but physical qubits can also leak into
higher energy levels (e.g., |2>). Leakage is especially harmful, as it corrupts
all subsequent syndrome measurements and can spread to neighboring qubits.
Detecting leakage on data qubits is particularly challenging, since they are
never measured directly during QEC cycles. Prior work, such as eraser,
addresses this by inferring leakage from syndrome patterns using a fixed
heuristic. However, this approach often misclassifies benign syndromes,
triggering excessive leakage-reduction circuits (LRCs). Because LRCs are
themselves noisy and slow, these false triggers lengthen QEC cycles and inflate
logical error rates.
  We propose gladiator, a general and adaptable leakage speculation framework
that works across surface code, color code, and qLDPC codes. Offline, gladiator
builds a code-aware error-propagation graph calibrated to device data. Online,
it classifies each syndrome in a few nanoseconds and schedules LRC only when
the observed pattern is provably leakage-dominated. This precise speculation
eliminates up to 3x (and on average 2x) unnecessary LRCs, shortens QEC cycles,
and suppresses false positives at their source. Evaluated on standard
fault-tolerant benchmarks, gladiator delivers 1.7x-3.9x speedups and 16%
reduction in logical error rate, advancing the efficiency of fault-tolerant
quantum computing.

### 3. [Silicon-based Josephson junction field-effect transistors enabling cryogenic logic and quantum technologies](http://arxiv.org/pdf/2510.25208v1)

Authors: Yusheng Xiong, Kaveh Delfanazari

The continuous miniaturisation of metal-oxide-semiconductor field-effect
transistors (MOSFETs) from long- to short-channel architectures has advanced
beyond the predictions of Moore's Law. Continued advances in semiconductor
electronics, even near current scaling and performance boundaries under
cryogenic conditions, are driving the development of innovative device
paradigms that enable ultra-low-power and high-speed functionality. Among
emerging candidates, the Josephson Junction Field-Effect Transistor (JJFET or
JoFET) provides an alternative by integrating superconducting source and drain
electrodes for efficient, phase-coherent operation at ultra-low temperatures.
These hybrid devices have the potential to bridge conventional semiconductor
electronics with cryogenic logic and quantum circuits, enabling
energy-efficient and high-coherence signal processing across temperature
domains. This review traces the evolution from Josephson junctions to
field-effect transistors, emphasising the structural and functional innovations
that underpin modern device scalability. The performance and material
compatibility of JJFETs fabricated on Si, GaAs, and InGaAs substrates are
analysed, alongside an assessment of their switching dynamics and material
compatibility. Particular attention is given to
superconductor-silicon-superconductor Josephson junctions as the active core of
JJFET architectures. By unfolding more than four decades of experimental
progress, this work highlights the promise of JJFETs as foundational building
blocks for next-generation cryogenic logic and quantum electronic systems.

### Computational Complexity

### 1. [Most Juntas Saturate the Hardcore Lemma](http://arxiv.org/pdf/2510.25165v1)

Authors: Vinayak M. Kumar

Consider a function that is mildly hard for size-$s$ circuits. For
sufficiently large $s$, Impagliazzo's hardcore lemma guarantees a
constant-density subset of inputs on which the same function is extremely hard
for circuits of size $s'<\!\!<s$. Blanc, Hayderi, Koch, and Tan [FOCS 2024]
recently showed that the degradation from $s$ to $s'$ in this lemma is
quantitatively tight in certain parameter regimes. We give a simpler and more
general proof of this result in almost all parameter regimes of interest by
showing that a random junta witnesses the tightness of the hardcore lemma with
high probability.

### Computational Engineering

### 1. [Enhancing Financial Decision-Making: Machine Learning and AI-Powered Predictions and Analysis](http://arxiv.org/pdf/2510.25201v1)

Authors: Vishal Patil, Kavya Bhand, Kaustubh Mukdam, Kavya Sharma, Manas Kawtikwar, Prajwal Kavhar, Hridayansh Kaware

The proposed system aims to use various machine learning algorithms to
enhance financial prediction and generate highly accurate analyses. It
introduces an AI-driven platform which offers inflation-analysis, stock market
prediction, and E-learning module powered by a chatbot. It has achieved high
accuracy where the Inflation Analysis depicts 0.8% MAE, 1.2% RMSE and the Stock
Prediction shows 98% and 96% accuracy for Apple and Google stock prices
respectively. Key features include historical price trends, inflation rates,
short-term future stock prediction, where the data has been extracted using
real-world financial datasets. Additionally, the E-learning feature contributes
to bridging financial gaps and promoting informed decisions. We have
implemented algorithms like linear regression, ARIMA, LSTM where the accuracy
has been evaluated using metrics such as MAE, RMSE and the like.

### 2. [Fourier Neural Operators for Two-Phase, 2D Mold-Filling Problems Related to Metal Casting](http://arxiv.org/pdf/2510.25697v1)

Authors: Edgard Moreira Minete, Mathis Immertreu, Fabian Teichmann, Sebastian Müller

This work reframes mold filling in metal casting as a simplified 2D operator
learning surrogate to avoid costly transient CFD simulations. The method
combines a graph based encoder that aggregates neighborhood information on an
unstructured input mesh to encode geometry and boundary data, a Fourier
spectral core that operates on a regular latent grid to capture global
interactions, and a graph based decoder that maps latent fields back to a
target mesh. The model jointly predicts velocities, pressure, and volume
fraction over a fixed horizon and generalizes across varied ingate locations
and process settings. On held out geometries and inlet conditions it reproduces
large scale advection and the fluid air interface with errors concentrated near
steep gradients. Mean relative L2 errors are about 5 percent across all fields.
Inference is roughly 100 to 1000 times faster than conventional CFD
simulations, thereby enabling rapid in-the-loop design exploration. Ablation
studies show accuracy drops monotonically with stronger spatial subsampling of
input vertices while temporal subsampling causes a gentler decline. Cutting the
training data by 50 percent yields only small error growth. Overall the results
demonstrate neural operators as efficient surrogates for 2D mold filling and
related filling problems and enable fast exploration and optimization of gating
system designs in casting workflows.

### 3. [Towards Automated Quality Assurance of Patent Specifications: A Multi-Dimensional LLM Framework](http://arxiv.org/pdf/2510.25402v1)

Authors: Yuqian Chai, Chaochao Wang, Weilei Wang

Despite the surge in patent applications and emergence of AI drafting tools,
systematic evaluation of patent content quality has received limited research
attention. To address this gap, We propose to evaluate patents using regulatory
compliance, technical coherence, and figure-reference consistency detection
modules, and then generate improvement suggestions via an integration module.
The framework is validated on a comprehensive dataset comprising 80
human-authored and 80 AI-generated patents from two patent drafting tools.
Experimental results show balanced accuracies of 99.74\%, 82.12\%, and 91.2\%
respectively across the three detection modules when validated against expert
annotations. Additional analysis was conducted to examine defect distributions
across patent sections, technical domains, and authoring sources. Section-based
analysis indicates that figure-text consistency and technical detail precision
require particular attention. Mechanical Engineering and Construction show more
claim-specification inconsistencies due to complex technical documentation
requirements. AI-generated patents show a significant gap compared to
human-authored ones. While human-authored patents primarily contain
surface-level errors like typos, AI-generated patents exhibit more structural
defects in figure-text alignment and cross-references.

### 4. [Graph Network-based Structural Simulator: Graph Neural Networks for Structural Dynamics](http://arxiv.org/pdf/2510.25683v1)

Authors: Alessandro Lucchetti, Francesco Cadini, Marco Giglio, Luca Lomazzi

Graph Neural Networks (GNNs) have recently been explored as surrogate models
for numerical simulations. While their applications in computational fluid
dynamics have been investigated, little attention has been given to structural
problems, especially for dynamic cases. To address this gap, we introduce the
Graph Network-based Structural Simulator (GNSS), a GNN framework for surrogate
modeling of dynamic structural problems.
  GNSS follows the encode-process-decode paradigm typical of GNN-based machine
learning models, and its design makes it particularly suited for dynamic
simulations thanks to three key features: (i) expressing node kinematics in
node-fixed local frames, which avoids catastrophic cancellation in
finite-difference velocities; (ii) employing a sign-aware regression loss,
which reduces phase errors in long rollouts; and (iii) using a
wavelength-informed connectivity radius, which optimizes graph construction.
  We evaluate GNSS on a case study involving a beam excited by a 50kHz
Hanning-modulated pulse. The results show that GNSS accurately reproduces the
physics of the problem over hundreds of timesteps and generalizes to unseen
loading conditions, where existing GNNs fail to converge or deliver meaningful
predictions.
  Compared with explicit finite element baselines, GNSS achieves substantial
inference speedups while preserving spatial and temporal fidelity. These
findings demonstrate that locality-preserving GNNs with physics-consistent
update rules are a competitive alternative for dynamic, wave-dominated
structural simulations.

### Computational Geometry

### 1. [M-Guarding in K-Visibility](http://arxiv.org/pdf/2510.25567v1)

Authors: Yeganeh Bahoo, Ahmad Kamaludeen

We explore the problem of $M$-guarding polygons with holes using
$k$-visibility guards, where a set of guards is said to $M$-guard a polygon if
every point in the polygon is visible to at least $M$ guards, with the
constraint that there may only be 1 guard on each edge. A $k$-visibility guard
can see through up to $k$ walls, with $k \geq 2$. We present a theorem
establishing that any polygon with holes can be 2-guarded under $k$-visibility
where $k \geq 2$, which expands existing results in 0-visibility. We provide an
algorithm that $M$-guards a polygon using a convex decomposition of the
polygon. We show that every point in the polygon is visible to at least four
$2$-visibility guards and then extend the result to show that for any even $k
\geq 2$ there exists a placement of guards such that every point in the polygon
is visible to $k + 2$ guards.

### Computation and Language

### 1. [Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?](http://arxiv.org/pdf/2510.25064v1)

Authors: Seonjeong Hwang, Hyounghun Kim, Gary Geunbae Lee

Estimating the cognitive complexity of reading comprehension (RC) items is
crucial for assessing item difficulty before it is administered to learners.
Unlike syntactic and semantic features, such as passage length or semantic
similarity between options, cognitive features that arise during answer
reasoning are not readily extractable using existing NLP tools and have
traditionally relied on human annotation. In this study, we examine whether
large language models (LLMs) can estimate the cognitive complexity of RC items
by focusing on two dimensions-Evidence Scope and Transformation Level-that
indicate the degree of cognitive burden involved in reasoning about the answer.
Our experimental results demonstrate that LLMs can approximate the cognitive
complexity of items, indicating their potential as tools for prior difficulty
analysis. Further analysis reveals a gap between LLMs' reasoning ability and
their metacognitive awareness: even when they produce correct answers, they
sometimes fail to correctly identify the features underlying their own
reasoning process.

### 2. [TOPol: Capturing and Explaining Multidimensional Semantic Polarity Fields and Vectors](http://arxiv.org/pdf/2510.25069v1)

Authors: Gabin Taibi, Lucia Gomez

Traditional approaches to semantic polarity in computational linguistics
treat sentiment as a unidimensional scale, overlooking the multidimensional
structure of language. This work introduces TOPol (Topic-Orientation POLarity),
a semi-unsupervised framework for reconstructing and interpreting
multidimensional narrative polarity fields under human-on-the-loop (HoTL)
defined contextual boundaries (CBs). The framework embeds documents using a
transformer-based large language model (tLLM), applies neighbor-tuned UMAP
projection, and segments topics via Leiden partitioning. Given a CB between
discourse regimes A and B, TOPol computes directional vectors between
corresponding topic-boundary centroids, yielding a polarity field that
quantifies fine-grained semantic displacement during regime shifts. This
vectorial representation enables assessing CB quality and detecting polarity
changes, guiding HoTL CB refinement. To interpret identified polarity vectors,
the tLLM compares their extreme points and produces contrastive labels with
estimated coverage. Robustness analyses show that only CB definitions (the main
HoTL-tunable parameter) significantly affect results, confirming methodological
stability. We evaluate TOPol on two corpora: (i) U.S. Central Bank speeches
around a macroeconomic breakpoint, capturing non-affective semantic shifts, and
(ii) Amazon product reviews across rating strata, where affective polarity
aligns with NRC valence. Results demonstrate that TOPol consistently captures
both affective and non-affective polarity transitions, providing a scalable,
generalizable, and interpretable framework for context-sensitive
multidimensional discourse analysis.

### 3. [DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates](http://arxiv.org/pdf/2510.25110v1)

Authors: Yun-Shiuan Chuang, Ruixuan Tu, Chengtao Dai, Smit Vasani, Binwei Yao, Michael Henry Tessler, Sijia Yang, Dhavan Shah, Robert Hawkins, Junjie Hu, Timothy T. Rogers

Accurately modeling opinion change through social interactions is crucial for
addressing issues like misinformation and polarization. While role-playing
large language models (LLMs) offer a promising way to simulate human-like
interactions, existing research shows that single-agent alignment does not
guarantee authentic multi-agent group dynamics. Current LLM role-play setups
often produce unnatural dynamics (e.g., premature convergence), without an
empirical benchmark to measure authentic human opinion trajectories. To bridge
this gap, we introduce DEBATE, the first large-scale empirical benchmark
explicitly designed to evaluate the authenticity of the interaction between
multi-agent role-playing LLMs. DEBATE contains 29,417 messages from multi-round
debate conversations among over 2,792 U.S.-based participants discussing 107
controversial topics, capturing both publicly-expressed messages and
privately-reported opinions. Using DEBATE, we systematically evaluate and
identify critical discrepancies between simulated and authentic group dynamics.
We further demonstrate DEBATE's utility for aligning LLMs with human behavior
through supervised fine-tuning, achieving improvements in surface-level metrics
(e.g., ROUGE-L and message length) while highlighting limitations in deeper
semantic alignment (e.g., semantic similarity). Our findings highlight both the
potential and current limitations of role-playing LLM agents for realistically
simulating human-like social dynamics.

### 4. [Pretraining Strategies using Monolingual and Parallel Data for Low-Resource Machine Translation](http://arxiv.org/pdf/2510.25116v1)

Authors: Idriss Nguepi Nguefack, Mara Finkelstein, Toadoum Sari Sakayo

This research article examines the effectiveness of various pretraining
strategies for developing machine translation models tailored to low-resource
languages. Although this work considers several low-resource languages,
including Afrikaans, Swahili, and Zulu, the translation model is specifically
developed for Lingala, an under-resourced African language, building upon the
pretraining approach introduced by Reid and Artetxe (2021), originally designed
for high-resource languages. Through a series of comprehensive experiments, we
explore different pretraining methodologies, including the integration of
multiple languages and the use of both monolingual and parallel data during the
pretraining phase. Our findings indicate that pretraining on multiple languages
and leveraging both monolingual and parallel data significantly enhance
translation quality. This study offers valuable insights into effective
pretraining strategies for low-resource machine translation, helping to bridge
the performance gap between high-resource and low-resource languages. The
results contribute to the broader goal of developing more inclusive and
accurate NLP models for marginalized communities and underrepresented
populations. The code and datasets used in this study are publicly available to
facilitate further research and ensure reproducibility, with the exception of
certain data that may no longer be accessible due to changes in public
availability.

### 5. [A Survey on Unlearning in Large Language Models](http://arxiv.org/pdf/2510.25117v1)

Authors: Ruichen Qiu, Jiajun Tan, Jiayue Pu, Honglin Wang, Xiao-Shan Gao, Fei Sun

The advancement of Large Language Models (LLMs) has revolutionized natural
language processing, yet their training on massive corpora poses significant
risks, including the memorization of sensitive personal data, copyrighted
material, and knowledge that could facilitate malicious activities. To mitigate
these issues and align with legal and ethical standards such as the "right to
be forgotten", machine unlearning has emerged as a critical technique to
selectively erase specific knowledge from LLMs without compromising their
overall performance. This survey provides a systematic review of over 180
papers on LLM unlearning published since 2021, focusing exclusively on
large-scale generative models. Distinct from prior surveys, we introduce novel
taxonomies for both unlearning methods and evaluations. We clearly categorize
methods into training-time, post-training, and inference-time based on the
training stage at which unlearning is applied. For evaluations, we not only
systematically compile existing datasets and metrics but also critically
analyze their advantages, disadvantages, and applicability, providing practical
guidance to the research community. In addition, we discuss key challenges and
promising future research directions. Our comprehensive overview aims to inform
and guide the ongoing development of secure and reliable LLMs.

### 6. [Explainable Disentanglement on Discrete Speech Representations for Noise-Robust ASR](http://arxiv.org/pdf/2510.25150v1)

Authors: Shreyas Gopal, Ashutosh Anshul, Haoyang Li, Yue Heng Yeo, Hexin Liu, Eng Siong Chng

Discrete audio representations are gaining traction in speech modeling due to
their interpretability and compatibility with large language models, but are
not always optimized for noisy or real-world environments. Building on existing
works that quantize Whisper embeddings for speech-to-unit modeling, we propose
disentangling semantic speech content from background noise in the latent
space. Our end-to-end model separates clean speech in the form of codebook
tokens, while extracting interpretable noise vectors as quantization residue
which are supervised via a lightweight classifier. We show that our approach
improves alignment between clean/noisy speech and text, producing speech tokens
that display a high degree of noiseinvariance, and improves ASR performance.
Keeping Whisper frozen, we show an 82% reduction in error rate compared to
Whisper, and 35% improvement over baseline methods on the VBDemand test set.
Further analyses show that the learned token space generalizes well to both
seen and unseen acoustic conditions.

### 7. [Testing Cross-Lingual Text Comprehension In LLMs Using Next Sentence Prediction](http://arxiv.org/pdf/2510.25187v1)

Authors: Ritesh Sunil Chavan, Jack Mostow

While large language models are trained on massive datasets, this data is
heavily skewed towards English. Does their impressive performance reflect
genuine ability or just this data advantage? To find out, we tested them in a
setting where they could not rely on data abundance: low-resource languages.
Building on prior work Agarwal et al. (2025) that used Next Sentence Prediction
(NSP) as a test, we created a large-scale benchmark with 10,000 questions each
for English (a high-resource language), Swahili (medium-resource), and Hausa
(low-resource). We then tested several top models, including GPT-4 Turbo,
Gemini 1.5 Flash, and LLaMA 3 70B, to see how their performance holds up. The
results painted a clear picture of how levels of language resources impact
outcomes. While all models excelled in English, their accuracy dropped in
Swahili and fell sharply in Hausa, with LLaMA 3 struggling the most. The story
became even more interesting when we introduced Chain-of-Thought (CoT)
prompting. For the struggling LLaMA 3, CoT acted as a helpful guide,
significantly boosting its accuracy. However, for the more capable GPT-4 and
Gemini, the same technique often backfired, leading to a kind of "overthinking"
that hurt their results in the cross-lingual context. This reveals that
Chain-of-Thought is not a universal solution; its effectiveness depends heavily
on the model's baseline capability and the specific context of the task. Our
framework pinpoints LLM weaknesses, highlights when CoT helps or hinders
cross-lingual NSP performance, and factors influencing their decisions.

### 8. [ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation](http://arxiv.org/pdf/2510.25224v1)

Authors: Ziyi Liu, Bahar Sarrafzadeh, Pei Zhou, Longqi Yang, Jieyu Zhao, Ashish Sharma

While Large Language Models (LLMs) are increasingly used in agentic
frameworks to assist individual users, there is a growing need for agents that
can proactively manage complex, multi-party collaboration. Systematic
evaluation methods for such proactive agents remain scarce, limiting progress
in developing AI that can effectively support multiple people together.
Negotiation offers a demanding testbed for this challenge, requiring
socio-cognitive intelligence to navigate conflicting interests between multiple
participants and multiple topics and build consensus. Here, we present
ProMediate, the first framework for evaluating proactive AI mediator agents in
complex, multi-topic, multi-party negotiations. ProMediate consists of two core
components: (i) a simulation testbed based on realistic negotiation cases and
theory-driven difficulty levels (ProMediate-Easy, ProMediate-Medium, and
ProMediate-Hard), with a plug-and-play proactive AI mediator grounded in
socio-cognitive mediation theories, capable of flexibly deciding when and how
to intervene; and (ii) a socio-cognitive evaluation framework with a new suite
of metrics to measure consensus changes, intervention latency, mediator
effectiveness, and intelligence. Together, these components establish a
systematic framework for assessing the socio-cognitive intelligence of
proactive AI agents in multi-party settings. Our results show that a socially
intelligent mediator agent outperforms a generic baseline, via faster,
better-targeted interventions. In the ProMediate-Hard setting, our social
mediator increases consensus change by 3.6 percentage points compared to the
generic baseline (10.65\% vs 7.01\%) while being 77\% faster in response
(15.98s vs. 3.71s). In conclusion, ProMediate provides a rigorous,
theory-grounded testbed to advance the development of proactive, socially
intelligent agents.

### 9. [Adapting Small Language Models to Low-Resource Domains: A Case Study in Hindi Tourism QA](http://arxiv.org/pdf/2510.25273v1)

Authors: Sandipan Majhi, Paheli Bhattacharya

Domain-specific question answering in low-resource languages faces two key
challenges: scarcity of annotated datasets and limited domain knowledge in
general-purpose language models. In this work, we present a multi-stage
finetuning strategy to adapt lightweight language models to the Hindi tourism
domain by leveraging both original and synthetic training data. Synthetic
question-answer pairs are generated using large LLMs (LLaMA-70B, Phi-14B) and
used to augment the limited original dataset. We explore several training
methodologies and analyse their impact on domain generalisation. Our results
demonstrate that large models can efficiently generate synthetic data, while
small models can effectively adapt to it, offering a scalable pathway for
low-resource, domain-specific QA.

### 10. [Teaching Sarcasm: Few-Shot Multimodal Sarcasm Detection via Distillation to a Parameter-Efficient Student](http://arxiv.org/pdf/2510.25303v1)

Authors: Soumyadeep Jana, Sanasam Ranbir Singh

Multimodal sarcasm detection is challenging, especially in low-resource
settings where subtle image-text contradictions are hard to learn due to scarce
annotated data, which hinders the model's performance. Parameter-efficient
fine-tuning (PEFT) methods like adapters, LoRA, and prompt tuning reduce
overfitting but struggle to reach optimal performance due to limited
supervision from few-shot data. We propose PEKD, a unified framework that
enhances PEFT methods via distillation from an expert model trained on
large-scale sarcasm data, which acts as the teacher. To mitigate unreliable
signals from the teacher, we introduce an entropy-aware gating mechanism that
dynamically adjusts the distillation strength based on teacher confidence.
Experiments on two public datasets demonstrate that our PEKD framework enables
PEFT methods to outperform both prior parameter-efficient approaches and large
multimodal models, achieving strong results in the few-shot scenario. The
framework is modular and adaptable to a wide range of multimodal models and
tasks.

### Cryptography and Security

### 1. [AgentCyTE: Leveraging Agentic AI to Generate Cybersecurity Training & Experimentation Scenarios](http://arxiv.org/pdf/2510.25189v1)

Authors: Ana M. Rodriguez, Jaime Acosta, Anantaa Kotal, Aritran Piplai

Designing realistic and adaptive networked threat scenarios remains a core
challenge in cybersecurity research and training, still requiring substantial
manual effort. While large language models (LLMs) show promise for automated
synthesis, unconstrained generation often yields configurations that fail
validation or execution. We present AgentCyTE, a framework integrating
LLM-based reasoning with deterministic, schema-constrained network emulation to
generate and refine executable threat environments. Through an agentic feedback
loop, AgentCyTE observes scenario outcomes, validates correctness, and
iteratively enhances realism and consistency. This hybrid approach preserves
LLM flexibility while enforcing structural validity, enabling scalable,
data-driven experimentation and reliable scenario generation for threat
modeling and adaptive cybersecurity training. Our framework can be accessed at:
https://github.com/AnantaaKotal/AgentCyTE

### 2. [From ECU to VSOC: UDS Security Monitoring Strategies](http://arxiv.org/pdf/2510.25375v1)

Authors: Ali Recai Yekta, Nicolas Loza, Jens Gramm, Michael Peter Schneider, Stefan Katzenbeisser

Increasing complexity and connectivity of modern vehicles have heightened
their vulnerability to cyberattacks. This paper addresses security challenges
associated with the Unified Diagnostic Services (UDS) protocol, a critical
communication framework for vehicle diagnostics in the automotive industry. We
present security monitoring strategies for the UDS protocol that leverage
in-vehicle logging and remote analysis through a Vehicle Security Operations
Center (VSOC). Our approach involves specifying security event logging
requirements, contextual data collection, and the development of detection
strategies aimed at identifying UDS attack scenarios. By applying these
strategies to a comprehensive taxonomy of UDS attack techniques, we demonstrate
that our detection methods cover a wide range of potential attack vectors.
Furthermore, we assess the adequacy of current AUTOSAR standardized security
events in supporting UDS attack detection, identifying gaps in the current
standard. This work enhances the understanding of vehicle security monitoring
and provides an example for developing robust cybersecurity measures in
automotive communication protocols.

### 3. [NetEcho: From Real-World Streaming Side-Channels to Full LLM Conversation Recovery](http://arxiv.org/pdf/2510.25472v1)

Authors: Zheng Zhang, Guanlong Wu, Sen Deng, Shuai Wang, Yinqian Zhang

In the rapidly expanding landscape of Large Language Model (LLM)
applications, real-time output streaming has become the dominant interaction
paradigm. While this enhances user experience, recent research reveals that it
exposes a non-trivial attack surface through network side-channels. Adversaries
can exploit patterns in encrypted traffic to infer sensitive information and
reconstruct private conversations. In response, LLM providers and third-party
services are deploying defenses such as traffic padding and obfuscation to
mitigate these vulnerabilities.
  This paper starts by presenting a systematic analysis of contemporary
side-channel defenses in mainstream LLM applications, with a focus on services
from vendors like OpenAI and DeepSeek. We identify and examine seven
representative deployment scenarios, each incorporating active/passive
mitigation techniques. Despite these enhanced security measures, our
investigation uncovers significant residual information that remains vulnerable
to leakage within the network traffic.
  Building on this discovery, we introduce NetEcho, a novel, LLM-based
framework that comprehensively unleashes the network side-channel risks of
today's LLM applications. NetEcho is designed to recover entire conversations
-- including both user prompts and LLM responses -- directly from encrypted
network traffic. It features a deliberate design that ensures high-fidelity
text recovery, transferability across different deployment scenarios, and
moderate operational cost. In our evaluations on medical and legal applications
built upon leading models like DeepSeek-v3 and GPT-4o, NetEcho can recover avg
$\sim$70\% information of each conversation, demonstrating a critical
limitation in current defense mechanisms. We conclude by discussing the
implications of our findings and proposing future directions for augmenting
network traffic security.

### 4. [A Study on Privacy-Preserving Scholarship Evaluation Based on Decentralized Identity and Zero-Knowledge Proofs](http://arxiv.org/pdf/2510.25477v1)

Authors: Yi Chen, Bin Chen, Peichang Zhang, Da Che

Traditional centralized scholarship evaluation processes typically require
students to submit detailed academic records and qualification information,
which exposes them to risks of data leakage and misuse, making it difficult to
simultaneously ensure privacy protection and transparent auditability. To
address these challenges, this paper proposes a scholarship evaluation system
based on Decentralized Identity (DID) and Zero-Knowledge Proofs (ZKP). The
system aggregates multidimensional ZKPs off-chain, and smart contracts verify
compliance with evaluation criteria without revealing raw scores or
computational details. Experimental results demonstrate that the proposed
solution not only automates the evaluation efficiently but also maximally
preserves student privacy and data integrity, offering a practical and
trustworthy technical paradigm for higher education scholarship programs.

### 5. [Is Protective DNS Blocking the Wild West?](http://arxiv.org/pdf/2510.25352v1)

Authors: David Plonka, Branden Palacio, Debbie Perouli

We perform a passive measurement study investigating how a Protective DNS
service might perform in a Research & Education Network serving hundreds of
member institutions. Utilizing freely-available DNS blocklists consisting of
domain names deemed to be threats, we test hundreds of millions of users' real
DNS queries, observed over a week's time, to find which answers would be
blocked because they involve domain names that are potential threats. We find
the blocklists disorderly regarding their names, goals, transparency, and
provenance making them quite difficult to compare. Consequently, these
Protective DNS underpinnings lack organized oversight, presenting challenges
and risks in operation at scale.

### 6. [ZK-SenseLM: Verifiable Large-Model Wireless Sensing with Selective Abstention and Zero-Knowledge Attestation](http://arxiv.org/pdf/2510.25677v1)

Authors: Hasan Akgul, Mari Eplik, Javier Rojas, Aina Binti Abdullah, Pieter van der Merwe

ZK-SenseLM is a secure and auditable wireless sensing framework that pairs a
large-model encoder for Wi-Fi channel state information (and optionally mmWave
radar or RFID) with a policy-grounded decision layer and end-to-end
zero-knowledge proofs of inference. The encoder uses masked spectral
pretraining with phase-consistency regularization, plus a light cross-modal
alignment that ties RF features to compact, human-interpretable policy tokens.
To reduce unsafe actions under distribution shift, we add a calibrated
selective-abstention head; the chosen risk-coverage operating point is
registered and bound into the proof. We implement a four-stage proving
pipeline: (C1) feature sanity and commitment, (C2) threshold and version
binding, (C3) time-window binding, and (C4) PLONK-style proofs that the
quantized network, given the committed window, produced the logged action and
confidence. Micro-batched proving amortizes cost across adjacent windows, and a
gateway option offloads proofs from low-power devices. The system integrates
with differentially private federated learning and on-device personalization
without weakening verifiability: model hashes and the registered threshold are
part of each public statement. Across activity, presence or intrusion,
respiratory proxy, and RF fingerprinting tasks, ZK-SenseLM improves macro-F1
and calibration, yields favorable coverage-risk curves under perturbations, and
rejects tamper and replay with compact proofs and fast verification.

### 7. [Model Inversion Attacks Meet Cryptographic Fuzzy Extractors](http://arxiv.org/pdf/2510.25687v1)

Authors: Mallika Prabhakar, Louise Xu, Prateek Saxena

Model inversion attacks pose an open challenge to privacy-sensitive
applications that use machine learning (ML) models. For example, face
authentication systems use modern ML models to compute embedding vectors from
face images of the enrolled users and store them. If leaked, inversion attacks
can accurately reconstruct user faces from the leaked vectors. There is no
systematic characterization of properties needed in an ideal defense against
model inversion, even for the canonical example application of a face
authentication system susceptible to data breaches, despite a decade of
best-effort solutions.
  In this paper, we formalize the desired properties of a provably strong
defense against model inversion and connect it, for the first time, to the
cryptographic concept of fuzzy extractors. We further show that existing fuzzy
extractors are insecure for use in ML-based face authentication. We do so
through a new model inversion attack called PIPE, which achieves a success rate
of over 89% in most cases against prior schemes. We then propose L2FE-Hash, the
first candidate fuzzy extractor which supports standard Euclidean distance
comparators as needed in many ML-based applications, including face
authentication. We formally characterize its computational security guarantees,
even in the extreme threat model of full breach of stored secrets, and
empirically show its usable accuracy in face authentication for practical face
distributions. It offers attack-agnostic security without requiring any
re-training of the ML model it protects. Empirically, it nullifies both prior
state-of-the-art inversion attacks as well as our new PIPE attack.

### 8. [Exact zCDP Characterizations for Fundamental Differentially Private Mechanisms](http://arxiv.org/pdf/2510.25746v1)

Authors: Charlie Harrison, Pasin Manurangsi

Zero-concentrated differential privacy (zCDP) is a variant of differential
privacy (DP) that is widely used partly thanks to its nice composition
property. While a tight conversion from $\epsilon$-DP to zCDP exists for the
worst-case mechanism, many common algorithms satisfy stronger guarantees. In
this work, we derive tight zCDP characterizations for several fundamental
mechanisms. We prove that the tight zCDP bound for the $\epsilon$-DP Laplace
mechanism is exactly $\epsilon + e^{-\epsilon} - 1$, confirming a recent
conjecture by Wang (2022). We further provide tight bounds for the discrete
Laplace mechanism, $k$-Randomized Response (for $k \leq 6$), and RAPPOR.
Lastly, we also provide a tight zCDP bound for the worst case bounded range
mechanism.

### 9. [An In-Depth Analysis of Cyber Attacks in Secured Platforms](http://arxiv.org/pdf/2510.25470v1)

Authors: Parick Ozoh, John K Omoniyi, Bukola Ibitoye

There is an increase in global malware threats. To address this, an
encryption-type ransomware has been introduced on the Android operating system.
The challenges associated with malicious threats in phone use have become a
pressing issue in mobile communication, disrupting user experiences and posing
significant privacy threats. This study surveys commonly used machine learning
techniques for detecting malicious threats in phones and examines their
performance. The majority of past research focuses on customer feedback and
reviews, with concerns that people might create false reviews to promote or
devalue products and services for personal gain. Hence, the development of
techniques for detecting malicious threats using machine learning has been a
key focus. This paper presents a comprehensive comparative study of current
research on the issue of malicious threats and methods for tackling these
challenges. Nevertheless, a huge amount of information is required by these
methods, presenting a challenge for developing robust, specialized automated
anti-malware systems. This research describes the Android Applications dataset,
and the accuracy of the techniques is measured using the accuracy levels of the
metrics employed in this study.

### 10. [Effect of Full Common Randomness Replication in Symmetric PIR on Graph-Based Replicated Systems](http://arxiv.org/pdf/2510.25736v1)

Authors: Shreya Meel, Sennur Ulukus

We revisit the problem of symmetric private information retrieval (SPIR) in
settings where the database replication is modeled by a simple graph. Here,
each vertex corresponds to a server, and a message is replicated on two servers
if and only if there is an edge between them. To satisfy the requirement of
database privacy, we let all the servers share some common randomness,
independent of the messages. We aim to quantify the improvement in SPIR
capacity, i.e., the maximum ratio of the number of desired and downloaded
symbols, compared to the setting with graph-replicated common randomness.
Towards this, we develop an algorithm to convert a class of PIR schemes into
the corresponding SPIR schemes, thereby establishing a capacity lower bound on
graphs for which such schemes exist. This includes the class of path and cyclic
graphs for which we derive capacity upper bounds that are tighter than the
trivial bounds given by the respective PIR capacities. For the special case of
path graph with three vertices, we identify the SPIR capacity to be
$\frac{1}{2}$.

### Computer Vision and Pattern Recognition

### 1. [Auto3DSeg for Brain Tumor Segmentation from 3D MRI in BraTS 2023 Challenge](http://arxiv.org/pdf/2510.25058v1)

Authors: Andriy Myronenko, Dong Yang, Yufan He, Daguang Xu

In this work, we describe our solution to the BraTS 2023 cluster of
challenges using Auto3DSeg from MONAI. We participated in all 5 segmentation
challenges, and achieved the 1st place results in three of them: Brain
Metastasis, Brain Meningioma, BraTS-Africa challenges, and the 2nd place
results in the remaining two: Adult and Pediatic Glioma challenges.

### 2. [DRIP: Dynamic patch Reduction via Interpretable Pooling](http://arxiv.org/pdf/2510.25067v1)

Authors: Yusen Peng, Sachin Kumar

Recently, the advances in vision-language models, including contrastive
pretraining and instruction tuning, have greatly pushed the frontier of
multimodal AI. However, owing to the large-scale and hence expensive
pretraining, the efficiency concern has discouraged researchers from attempting
to pretrain a vision language model from scratch. In this work, we propose
Dynamic patch Reduction via Interpretable Pooling (DRIP), which adapts to the
input images and dynamically merges tokens in the deeper layers of a visual
encoder. Our results on both ImageNet training from scratch and CLIP
contrastive pretraining demonstrate a significant GFLOP reduction while
maintaining comparable classification/zero-shot performance. To further
validate our proposed method, we conduct continual pretraining on a large
biology dataset, extending its impact into scientific domains.

### 3. [Vision-Language Integration for Zero-Shot Scene Understanding in Real-World Environments](http://arxiv.org/pdf/2510.25070v1)

Authors: Manjunath Prasad Holenarasipura Rajiv, B. M. Vidyavathi

Zero-shot scene understanding in real-world settings presents major
challenges due to the complexity and variability of natural scenes, where
models must recognize new objects, actions, and contexts without prior labeled
examples. This work proposes a vision-language integration framework that
unifies pre-trained visual encoders (e.g., CLIP, ViT) and large language models
(e.g., GPT-based architectures) to achieve semantic alignment between visual
and textual modalities. The goal is to enable robust zero-shot comprehension of
scenes by leveraging natural language as a bridge to generalize over unseen
categories and contexts. Our approach develops a unified model that embeds
visual inputs and textual prompts into a shared space, followed by multimodal
fusion and reasoning layers for contextual interpretation. Experiments on
Visual Genome, COCO, ADE20K, and custom real-world datasets demonstrate
significant gains over state-of-the-art zero-shot models in object recognition,
activity detection, and scene captioning. The proposed system achieves up to
18% improvement in top-1 accuracy and notable gains in semantic coherence
metrics, highlighting the effectiveness of cross-modal alignment and language
grounding in enhancing generalization for real-world scene understanding.

### 4. [PSTF-AttControl: Per-Subject-Tuning-Free Personalized Image Generation with Controllable Face Attributes](http://arxiv.org/pdf/2510.25084v1)

Authors: Xiang liu, Zhaoxiang Liu, Huan Hu, Zipeng Wang, Ping Chen, Zezhou Chen, Kai Wang, Shiguo Lian

Recent advancements in personalized image generation have significantly
improved facial identity preservation, particularly in fields such as
entertainment and social media. However, existing methods still struggle to
achieve precise control over facial attributes in a per-subject-tuning-free
(PSTF) way. Tuning-based techniques like PreciseControl have shown promise by
providing fine-grained control over facial features, but they often require
extensive technical expertise and additional training data, limiting their
accessibility. In contrast, PSTF approaches simplify the process by enabling
image generation from a single facial input, but they lack precise control over
facial attributes. In this paper, we introduce a novel, PSTF method that
enables both precise control over facial attributes and high-fidelity
preservation of facial identity. Our approach utilizes a face recognition model
to extract facial identity features, which are then mapped into the $W^+$
latent space of StyleGAN2 using the e4e encoder. We further enhance the model
with a Triplet-Decoupled Cross-Attention module, which integrates facial
identity, attribute features, and text embeddings into the UNet architecture,
ensuring clean separation of identity and attribute information. Trained on the
FFHQ dataset, our method allows for the generation of personalized images with
fine-grained control over facial attributes, while without requiring additional
fine-tuning or training data for individual identities. We demonstrate that our
approach successfully balances personalization with precise facial attribute
control, offering a more efficient and user-friendly solution for high-quality,
adaptable facial image synthesis. The code is publicly available at
https://github.com/UnicomAI/PSTF-AttControl.

### 5. [Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](http://arxiv.org/pdf/2510.25094v1)

Authors: Chanhyeong Yang, Taehoon Song, Jihwan Park, Hyunwoo J. Kim

Zero-shot Human-Object Interaction detection aims to localize humans and
objects in an image and recognize their interaction, even when specific
verb-object pairs are unseen during training. Recent works have shown promising
results using prompt learning with pretrained vision-language models such as
CLIP, which align natural language prompts with visual features in a shared
embedding space. However, existing approaches still fail to handle the visual
complexity of interaction, including (1) intra-class visual diversity, where
instances of the same verb appear in diverse poses and contexts, and (2)
inter-class visual entanglement, where distinct verbs yield visually similar
patterns. To address these challenges, we propose VDRP, a framework for Visual
Diversity and Region-aware Prompt learning. First, we introduce a visual
diversity-aware prompt learning strategy that injects group-wise visual
variance into the context embedding. We further apply Gaussian perturbation to
encourage the prompts to capture diverse visual variations of a verb. Second,
we retrieve region-specific concepts from the human, object, and union regions.
These are used to augment the diversity-aware prompt embeddings, yielding
region-aware prompts that enhance verb-level discrimination. Experiments on the
HICO-DET benchmark demonstrate that our method achieves state-of-the-art
performance under four zero-shot evaluation settings, effectively addressing
both intra-class diversity and inter-class visual entanglement. Code is
available at https://github.com/mlvlab/VDRP.

### 6. [AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians](http://arxiv.org/pdf/2510.25129v1)

Authors: Xiyu Zhang, Chong Bao, Yipeng Chen, Hongjia Zhai, Yitong Dong, Hujun Bao, Zhaopeng Cui, Guofeng Zhang

3D reconstruction of indoor and urban environments is a prominent research
topic with various downstream applications. However, existing geometric priors
for addressing low-texture regions in indoor and urban settings often lack
global consistency. Moreover, Gaussian Splatting and implicit SDF fields often
suffer from discontinuities or exhibit computational inefficiencies, resulting
in a loss of detail. To address these issues, we propose an Atlanta-world
guided implicit-structured Gaussian Splatting that achieves smooth indoor and
urban scene reconstruction while preserving high-frequency details and
rendering efficiency. By leveraging the Atlanta-world model, we ensure the
accurate surface reconstruction for low-texture regions, while the proposed
novel implicit-structured GS representations provide smoothness without
sacrificing efficiency and high-frequency details. Specifically, we propose a
semantic GS representation to predict the probability of all semantic regions
and deploy a structure plane regularization with learnable plane indicators for
global accurate surface reconstruction. Extensive experiments demonstrate that
our method outperforms state-of-the-art approaches in both indoor and urban
scenes, delivering superior surface reconstruction quality.

### 7. [Region-CAM: Towards Accurate Object Regions in Class Activation Maps for Weakly Supervised Learning Tasks](http://arxiv.org/pdf/2510.25134v1)

Authors: Qingdong Cai, Charith Abhayaratne

Class Activation Mapping (CAM) methods are widely applied in weakly
supervised learning tasks due to their ability to highlight object regions.
However, conventional CAM methods highlight only the most discriminative
regions of the target. These highlighted regions often fail to cover the entire
object and are frequently misaligned with object boundaries, thereby limiting
the performance of downstream weakly supervised learning tasks, particularly
Weakly Supervised Semantic Segmentation (WSSS), which demands pixel-wise
accurate activation maps to get the best results. To alleviate the above
problems, we propose a novel activation method, Region-CAM. Distinct from
network feature weighting approaches, Region-CAM generates activation maps by
extracting semantic information maps (SIMs) and performing semantic information
propagation (SIP) by considering both gradients and features in each of the
stages of the baseline classification model. Our approach highlights a greater
proportion of object regions while ensuring activation maps to have precise
boundaries that align closely with object edges. Region-CAM achieves 60.12% and
58.43% mean intersection over union (mIoU) using the baseline model on the
PASCAL VOC training and validation datasets, respectively, which are
improvements of 13.61% and 13.13% over the original CAM (46.51% and 45.30%). On
the MS COCO validation set, Region-CAM achieves 36.38%, a 16.23% improvement
over the original CAM (20.15%). We also demonstrate the superiority of
Region-CAM in object localization tasks, using the ILSVRC2012 validation set.
Region-CAM achieves 51.7% in Top-1 Localization accuracy Loc1. Compared with
LayerCAM, an activation method designed for weakly supervised object
localization, Region-CAM achieves 4.5% better performance in Loc1.

### 8. [DINO-YOLO: Self-Supervised Pre-training for Data-Efficient Object Detection in Civil Engineering Applications](http://arxiv.org/pdf/2510.25140v1)

Authors: Malaisree P, Youwai S, Kitkobsin T, Janrungautai S, Amorndechaphon D, Rojanavasu P

Object detection in civil engineering applications is constrained by limited
annotated data in specialized domains. We introduce DINO-YOLO, a hybrid
architecture combining YOLOv12 with DINOv3 self-supervised vision transformers
for data-efficient detection. DINOv3 features are strategically integrated at
two locations: input preprocessing (P0) and mid-backbone enhancement (P3).
Experimental validation demonstrates substantial improvements: Tunnel Segment
Crack detection (648 images) achieves 12.4% improvement, Construction PPE (1K
images) gains 13.7%, and KITTI (7K images) shows 88.6% improvement, while
maintaining real-time inference (30-47 FPS). Systematic ablation across five
YOLO scales and nine DINOv3 variants reveals that Medium-scale architectures
achieve optimal performance with DualP0P3 integration (55.77% mAP@0.5), while
Small-scale requires Triple Integration (53.63%). The 2-4x inference overhead
(21-33ms versus 8-16ms baseline) remains acceptable for field deployment on
NVIDIA RTX 5090. DINO-YOLO establishes state-of-the-art performance for civil
engineering datasets (<10K images) while preserving computational efficiency,
providing practical solutions for construction safety monitoring and
infrastructure inspection in data-constrained environments.

### 9. [Revisiting Reconstruction-based AI-generated Image Detection: A Geometric Perspective](http://arxiv.org/pdf/2510.25141v1)

Authors: Wan Jiang, Jing Yan, Ruixuan Zhang, Xiaojing Chen, Changtao Miao, Zhe Li, Chenhao Lin, Yunfeng Diao, Richang Hong

The rise of generative Artificial Intelligence (AI) has made detecting
AI-generated images a critical challenge for ensuring authenticity. Existing
reconstruction-based methods lack theoretical foundations and on empirical
heuristics, limiting interpretability and reliability. In this paper, we
introduce the Jacobian-Spectral Lower Bound for reconstruction error from a
geometric perspective, showing that real images off the reconstruction manifold
exhibit a non-trivial error lower bound, while generated images on the manifold
have near-zero error. Furthermore, we reveal the limitations of existing
methods that rely on static reconstruction error from a single pass. These
methods often fail when some real images exhibit lower error than generated
ones. This counterintuitive behavior reduces detection accuracy and requires
data-specific threshold tuning, limiting their applicability in real-world
scenarios. To address these challenges, we propose ReGap, a training-free
method that computes dynamic reconstruction error by leveraging structured
editing operations to introduce controlled perturbations. This enables
measuring error changes before and after editing, improving detection accuracy
by enhancing error separation. Experimental results show that our method
outperforms existing baselines, exhibits robustness to common post-processing
operations and generalizes effectively across diverse conditions.

### 10. [EA3D: Online Open-World 3D Object Extraction from Streaming Videos](http://arxiv.org/pdf/2510.25146v1)

Authors: Xiaoyu Zhou, Jingqi Wang, Yuang Jia, Yongtao Wang, Deqing Sun, Ming-Hsuan Yang

Current 3D scene understanding methods are limited by offline-collected
multi-view data or pre-constructed 3D geometry. In this paper, we present
ExtractAnything3D (EA3D), a unified online framework for open-world 3D object
extraction that enables simultaneous geometric reconstruction and holistic
scene understanding. Given a streaming video, EA3D dynamically interprets each
frame using vision-language and 2D vision foundation encoders to extract
object-level knowledge. This knowledge is integrated and embedded into a
Gaussian feature map via a feed-forward online update strategy. We then
iteratively estimate visual odometry from historical frames and incrementally
update online Gaussian features with new observations. A recurrent joint
optimization module directs the model's attention to regions of interest,
simultaneously enhancing both geometric reconstruction and semantic
understanding. Extensive experiments across diverse benchmarks and tasks,
including photo-realistic rendering, semantic and instance segmentation, 3D
bounding box and semantic occupancy estimation, and 3D mesh generation,
demonstrate the effectiveness of EA3D. Our method establishes a unified and
efficient framework for joint online 3D reconstruction and holistic scene
understanding, enabling a broad range of downstream tasks.

### Computers and Society

### 1. [Teaching Probabilistic Machine Learning in the Liberal Arts: Empowering Socially and Mathematically Informed AI Discourse](http://arxiv.org/pdf/2510.25049v1)

Authors: Yaniv Yacoby

We present a new undergraduate ML course at our institution, a small liberal
arts college serving students minoritized in STEM, designed to empower students
to critically connect the mathematical foundations of ML with its
sociotechnical implications. We propose a "framework-focused" approach,
teaching students the language and formalism of probabilistic modeling while
leveraging probabilistic programming to lower mathematical barriers. We
introduce methodological concepts through a whimsical, yet realistic theme, the
"Intergalactic Hypothetical Hospital," to make the content both relevant and
accessible. Finally, we pair each technical innovation with counter-narratives
that challenge its value using real, open-ended case-studies to cultivate
dialectical thinking. By encouraging creativity in modeling and highlighting
unresolved ethical challenges, we help students recognize the value and need of
their unique perspectives, empowering them to participate confidently in AI
discourse as technologists and critical citizens.

### 2. [Scaling Cultural Resources for Improving Generative Models](http://arxiv.org/pdf/2510.25167v1)

Authors: Hayk Stepanyan, Aishwarya Verma, Andrew Zaldivar, Rutledge Chin Feman, Erin MacMurray van Liemt, Charu Kalia, Vinodkumar Prabhakaran, Sunipa Dev

Generative models are known to have reduced performance in different global
cultural contexts and languages. While continual data updates have been
commonly conducted to improve overall model performance, bolstering and
evaluating this cross-cultural competence of generative AI models requires data
resources to be intentionally expanded to include global contexts and
languages. In this work, we construct a repeatable, scalable, multi-pronged
pipeline to collect and contribute culturally salient, multilingual data. We
posit that such data can assess the state of the global applicability of our
models and thus, in turn, help identify and improve upon cross-cultural gaps.

### 3. [The Open Source Resume: How Open Source Contributions Help Students Demonstrate Alignment with Employer Needs](http://arxiv.org/pdf/2510.25180v1)

Authors: Utsab Saha, Jeffrey D'Andria, Tyler Menezes

Computer science educators are increasingly integrating open source
contributions into classes to prepare students for higher expectations due to
GenAI, and to improve employment outcomes in an increasingly competitive job
market. However, little is known about how employers view student open source
contributions. This paper addresses two research questions qualitatively: what
traits do employers desire for entry-level hires in 2025, and how can they be
demonstrated through open source contributions? It also tests quantitatively
the hypothesis that student knowledge of employers' expectations will improve
their motivation to work on open source projects. To answer our qualitative
questions, we conducted interviews with US hiring managers. We collaborated
with each interviewee to create a "hiring manager agreement," which listed
desirable traits and specific ways to demonstrate them through open source,
along with a promise to interview some students meeting the criteria. To
evaluate our quantitative hypothesis, we surveyed 650 undergraduates attending
public universities in the US using an instrument based on expectancy-value
theory. Hiring managers wanted many non-technical traits that are difficult to
teach in traditional CS classes, such as initiative. There were many
commonalities in how employers wanted to see these traits demonstrated in open
source contributions. Viewing hiring manager agreements improved student
motivation to contribute to open source projects. Our findings suggest that
open source contributions may help CS undergraduates get hired, but this
requires sustained engagement in multiple areas. Educators can motivate
students by sharing employer expectations, but further work is required to
determine if this changes their behavior.

### 4. [Tackling the Algorithmic Control Crisis -- the Technical, Legal, and Ethical Challenges of Research into Algorithmic Agents](http://arxiv.org/pdf/2510.25337v1)

Authors: B. Bodo, N. Helberger, K. Irion, F. Zuiderveen Borgesius, J. Moller, B. Van der Velde, N. Bol, B. van Es, C. de Vreese

Algorithmic agents permeate every instant of our online existence. Based on
our digital profiles built from the massive surveillance of our digital
existence, algorithmic agents rank search results, filter our emails, hide and
show news items on social networks feeds, try to guess what products we might
buy next for ourselves and for others, what movies we want to watch, and when
we might be pregnant. Algorithmic agents select, filter, and recommend
products, information, and people. Increasingly, algorithmic agents don't just
select from the range of human created alternatives, but also they create.
Burgeoning algorithmic agents are capable of providing us with content made
just for us, and engage with us through one-of-a-kind, personalized
interactions. Studying these algorithmic agents presents a host of
methodological, ethical, and logistical challenges. The objectives of our paper
are two-fold. The first aim is to describe one possible approach to researching
the individual and societal effects of algorithmic recommenders, and to share
our experiences with the academic community. The second is to contribute to a
more fundamental discussion about the ethical and legal issues of "tracking the
trackers", as well as the costs and trade-offs involved. Our paper will
contribute to the discussion on the relative merits, costs and benefits of
different approaches to ethically and legally sound research on algorithmic
governance. We will argue that besides shedding light on how users interact
with algorithmic agents, we also need to be able to understand how different
methods of monitoring our algorithmically controlled digital environments
compare to each other in terms of costs and benefits. We conclude our article
with a number of concrete suggestions for how to address the practical, ethical
and legal challenges of researching algorithms and their effects on users and
society.

### 5. [Tracking Walls, Take-It-Or-Leave-It Choices, the GDPR, and the ePrivacy Regulation](http://arxiv.org/pdf/2510.25339v1)

Authors: Frederik J. Zuiderveen Borgesius, Sanne Kruikemeier, Sophie C. Boerman, Natali Helberger

On the internet, we encounter take-it-or-leave-it choices regarding our
privacy on a daily basis. In Europe, online tracking for targeted advertising
generally requires the internet users' consent to be lawful. Some websites use
a tracking wall, a barrier that visitors can only pass if they consent to
tracking by third parties. When confronted with such a tracking wall, many
people click 'I agree' to tracking. A survey that we conducted shows that most
people find tracking walls unfair and unacceptable. We analyse under which
conditions the ePrivacy Directive and the General Data Protection Regulation
allow tracking walls. We provide a list of circumstances to assess when a
tracking wall makes consent invalid. We also explore how the EU lawmaker could
regulate tracking walls, for instance in the ePrivacy Regulation. It should be
seriously considered to ban tracking walls, at least in certain circumstances.

### 6. [Shifts in U.S. Social Media Use, 2020-2024: Decline, Fragmentation, and Enduring Polarization](http://arxiv.org/pdf/2510.25417v1)

Authors: Petter Törnberg

Using nationally representative data from the 2020 and 2024 American National
Election Studies (ANES), this paper traces how the U.S. social media landscape
has shifted across platforms, demographics, and politics. Overall platform use
has declined, with the youngest and oldest Americans increasingly abstaining
from social media altogether. Facebook, YouTube, and Twitter/X have lost
ground, while TikTok and Reddit have grown modestly, reflecting a more
fragmented digital public sphere. Platform audiences have aged and become
slightly more educated and diverse. Politically, most platforms have moved
toward Republican users while remaining, on balance, Democratic-leaning.
Twitter/X has experienced the sharpest shift: posting has flipped nearly 50
percentage points from Democrats to Republicans. Across platforms, political
posting remains tightly linked to affective polarization, as the most partisan
users are also the most active. As casual users disengage and polarized
partisans remain vocal, the online public sphere grows smaller, sharper, and
more ideologically extreme.

### 7. [The Iceberg Index: Measuring Workforce Exposure Across the AI Economy](http://arxiv.org/pdf/2510.25137v1)

Authors: Ayush Chopra, Santanu Bhattacharya, DeAndrea Salvador, Ayan Paul, Teddy Wright, Aditi Garg, Feroz Ahmad, Alice C. Schwarze, Ramesh Raskar, Prasanna Balaprakash

Artificial Intelligence is reshaping America's \$9.4 trillion labor market,
with cascading effects that extend far beyond visible technology sectors. When
AI transforms quality control tasks in automotive plants, consequences spread
through logistics networks, supply chains, and local service economies. Yet
traditional workforce metrics cannot capture these ripple effects: they measure
employment outcomes after disruption occurs, not where AI capabilities overlap
with human skills before adoption crystallizes. Project Iceberg addresses this
gap using Large Population Models to simulate the human-AI labor market,
representing 151 million workers as autonomous agents executing over 32,000
skills and interacting with thousands of AI tools. It introduces the Iceberg
Index, a skills-centered metric that measures the wage value of skills AI
systems can perform within each occupation. The Index captures technical
exposure, where AI can perform occupational tasks, not displacement outcomes or
adoption timelines. Analysis shows that visible AI adoption concentrated in
computing and technology (2.2% of wage value, approx \$211 billion) represents
only the tip of the iceberg. Technical capability extends far below the surface
through cognitive automation spanning administrative, financial, and
professional services (11.7%, approx \$1.2 trillion). This exposure is fivefold
larger and geographically distributed across all states rather than confined to
coastal hubs. Traditional indicators such as GDP, income, and unemployment
explain less than 5% of this skills-based variation, underscoring why new
indices are needed to capture exposure in the AI economy. By simulating how
these capabilities may spread under scenarios, Iceberg enables policymakers and
business leaders to identify exposure hotspots, prioritize investments, and
test interventions before committing billions to implementation

### 8. [Human Resilience in the AI Era -- What Machines Can't Replace](http://arxiv.org/pdf/2510.25218v1)

Authors: Shaoshan Liu, Anina Schwarzenbach, Yiyu Shi

AI is displacing tasks, mediating high-stakes decisions, and flooding
communication with synthetic content, unsettling work, identity, and social
trust. We argue that the decisive human countermeasure is resilience. We define
resilience across three layers: psychological, including emotion regulation,
meaning-making, cognitive flexibility; social, including trust, social capital,
coordinated response; organizational, including psychological safety, feedback
mechanisms, and graceful degradation. We synthesize early evidence that these
capacities buffer individual strain, reduce burnout through social support, and
lower silent failure in AI-mediated workflows through team norms and
risk-responsive governance. We also show that resilience can be cultivated
through training that complements rather than substitutes for structural
safeguards. By reframing the AI debate around actionable human resilience, this
article offers policymakers, educators, and operators a practical lens to
preserve human agency and steer responsible adoption.

### 9. [Instrumental goals in advanced AI systems: Features to be managed and not failures to be eliminated?](http://arxiv.org/pdf/2510.25471v1)

Authors: Willem Fourie

In artificial intelligence (AI) alignment research, instrumental goals, also
called instrumental subgoals or instrumental convergent goals, are widely
associated with advanced AI systems. These goals, which include tendencies such
as power-seeking and self-preservation, become problematic when they conflict
with human aims. Conventional alignment theory treats instrumental goals as
sources of risk that become problematic through failure modes such as reward
hacking or goal misgeneralization, and attempts to limit the symptoms of
instrumental goals, notably resource acquisition and self-preservation. This
article proposes an alternative framing: that a philosophical argument can be
constructed according to which instrumental goals may be understood as features
to be accepted and managed rather than failures to be limited. Drawing on
Aristotle's ontology and its modern interpretations, an ontology of concrete,
goal-directed entities, it argues that advanced AI systems can be seen as
artifacts whose formal and material constitution gives rise to effects distinct
from their designers' intentions. In this view, the instrumental tendencies of
such systems correspond to per se outcomes of their constitution rather than
accidental malfunctions. The implication is that efforts should focus less on
eliminating instrumental goals and more on understanding, managing, and
directing them toward human-aligned ends.

### Databases

### 1. [Time-varying Vector Field Compression with Preserved Critical Point Trajectories](http://arxiv.org/pdf/2510.25143v1)

Authors: Mingze Xia, Yuxiao Li, Pu Jiao, Bei Wang, Xin Liang, Hanqi Guo

Scientific simulations and observations are producing vast amounts of
time-varying vector field data, making it hard to store them for archival
purposes and transmit them for analysis. Lossy compression is considered a
promising approach to reducing these data because lossless compression yields
low compression ratios that barely mitigate the problem. However, directly
applying existing lossy compression methods to timevarying vector fields may
introduce undesired distortions in critical-point trajectories, a crucial
feature that encodes key properties of the vector field. In this work, we
propose an efficient lossy compression framework that exactly preserves all
critical-point trajectories in time-varying vector fields. Our contributions
are threefold. First, we extend the theory for preserving critical points in
space to preserving critical-point trajectories in space-time, and develop a
compression framework to realize the functionality. Second, we propose a
semi-Lagrange predictor to exploit the spatiotemporal correlations in
advectiondominated regions, and combine it with the traditional Lorenzo
predictor for improved compression efficiency. Third, we evaluate our method
against state-of-the-art lossy and lossless compressors using four real-world
scientific datasets. Experimental results demonstrate that the proposed method
delivers up to 124.48X compression ratios while effectively preserving all
critical-point trajectories. This compression ratio is up to 56.07X higher than
that of the best lossless compressors, and none of the existing lossy
compressors can preserve all critical-point trajectories at similar compression
ratios.

### 2. [DGAI: Decoupled On-Disk Graph-Based ANN Index for Efficient Updates and Queries](http://arxiv.org/pdf/2510.25401v1)

Authors: Jiahao Lou, Quan Yu, Shufeng Gong, Song Yu, Yanfeng Zhang, Ge Yu

On-disk graph-based indexes are widely used in approximate nearest neighbor
(ANN) search systems for large-scale, high-dimensional vectors. However,
traditional coupled storage methods, which store vectors within the index, are
inefficient for index updates. Coupled storage incurs excessive redundant
vector reads and writes when updating the graph topology, leading to
significant invalid I/O. To address this issue, we propose a decoupled storage
architecture. While a decoupled architecture reduces query performance. To
overcome this limitation, we design two tailored strategies: (i) a three-stage
query mechanism that leverages multiple PQ compressed vectors to filter invalid
I/O and computations, and (ii) an incremental page-level topological reordering
strategy that incrementally inserts new nodes into pages containing their most
similar neighbors to mitigate read amplification. Together, these techniques
substantially reduce both I/O and computational overhead during ANN search.
Experimental results show that the decoupled architecture improves update speed
by 10.05x for insertions and 6.89x for deletions, while the three-stage query
and incremental reordering enhance query efficiency by 2.66x compared to the
traditional coupled architecture.

### 3. [One Join Order Does Not Fit All: Reducing Intermediate Results with Per-Split Query Plans](http://arxiv.org/pdf/2510.25684v1)

Authors: Yujun He, Hangdong Zhao, Simon Frisk, Yifei Yang, Kevin Kristensen, Paraschos Koutris, Xiangyao Yu

Minimizing intermediate results is critical for efficient multi-join query
processing. Although the seminal Yannakakis algorithm offers strong guarantees
for acyclic queries, cyclic queries remain an open challenge. In this paper, we
propose SplitJoin, a framework that introduces split as a first-class query
operator. By partitioning input tables into heavy and light parts, SplitJoin
allows different data partitions to use distinct query plans, with the goal of
reducing intermediate sizes using existing binary join engines. We
systematically explore the design space for split-based optimizations,
including threshold selection, split strategies, and join ordering after
splits. Implemented as a front-end to DuckDB and Umbra, SplitJoin achieves
substantial improvements: on DuckDB, SplitJoin completes 43 social network
queries (vs. 29 natively), achieving 2.1x faster runtime and 7.9x smaller
intermediates on average (up to 13.6x and 74x, respectively); on Umbra, it
completes 45 queries (vs. 35), achieving 1.3x speedups and 1.2x smaller
intermediates on average (up to 6.1x and 2.1x, respectively).

### Distributed, Parallel, and Cluster Computing

### 1. [Multi-Resolution Model Fusion for Accelerating the Convolutional Neural Network Training](http://arxiv.org/pdf/2510.25170v1)

Authors: Kewei Wang, Claire Songhyun Lee, Sunwoo Lee, Vishu Gupta, Jan Balewski, Alex Sim, Peter Nugent, Ankit Agrawal, Alok Choudhary, Kesheng Wu, Wei-keng Liao

Neural networks are rapidly gaining popularity in scientific research, but
training the models is often very time-consuming. Particularly when the
training data samples are large high-dimensional arrays, efficient training
methodologies that can reduce the computational costs are crucial. To reduce
the training cost, we propose a Multi-Resolution Model Fusion (MRMF) method
that combines models trained on reduced-resolution data and then refined with
data in the original resolution. We demonstrate that these reduced-resolution
models and datasets could be generated quickly. More importantly, the proposed
approach reduces the training time by speeding up the model convergence in each
fusion stage before switching to the final stage of finetuning with data in its
original resolution. This strategy ensures the final model retains
high-resolution insights while benefiting from the computational efficiency of
lower-resolution training. Our experiment results demonstrate that the
multi-resolution model fusion method can significantly reduce end-to-end
training time while maintaining the same model accuracy. Evaluated using two
real-world scientific applications, CosmoFlow and Neuron Inverter, the proposed
method improves the training time by up to 47% and 44%, respectively, as
compared to the original resolution training, while the model accuracy is not
affected.

### 2. [MoEntwine: Unleashing the Potential of Wafer-scale Chips for Large-scale Expert Parallel Inference](http://arxiv.org/pdf/2510.25258v1)

Authors: Xinru Tang, Jingxiang Hou, Dingcheng Jiang, Taiquan Wei, Jiaxin Liu, Jinyi Deng, Huizheng Wang, Qize Yang, Haoran Shang, Chao Li, Yang Hu, Shouyi Yin

As large language models (LLMs) continue to scale up, mixture-of-experts
(MoE) has become a common technology in SOTA models. MoE models rely on expert
parallelism (EP) to alleviate memory bottleneck, which introduces all-to-all
communication to dispatch and combine tokens across devices. However, in
widely-adopted GPU clusters, high-overhead cross-node communication makes
all-to-all expensive, hindering the adoption of EP. Recently, wafer-scale chips
(WSCs) have emerged as a platform integrating numerous devices on a wafer-sized
interposer. WSCs provide a unified high-performance network connecting all
devices, presenting a promising potential for hosting MoE models. Yet, their
network is restricted to a mesh topology, causing imbalanced communication
pressure and performance loss. Moreover, the lack of on-wafer disk leads to
high-overhead expert migration on the critical path.
  To fully unleash this potential, we first propose Entwined Ring Mapping
(ER-Mapping), which co-designs the mapping of attention and MoE layers to
balance communication pressure and achieve better performance. We find that
under ER-Mapping, the distribution of cold and hot links in the attention and
MoE layers is complementary. Therefore, to hide the migration overhead, we
propose the Non-invasive Balancer (NI-Balancer), which splits a complete expert
migration into multiple steps and alternately utilizes the cold links of both
layers. Evaluation shows ER-Mapping achieves communication reduction up to 62%.
NI-Balancer further delivers 54% and 22% improvements in MoE computation and
communication, respectively. Compared with the SOTA NVL72 supernode, the WSC
platform delivers an average 39% higher per-device MoE performance owing to its
scalability to larger EP.

### 3. [A Privacy-Preserving Ecosystem for Developing Machine Learning Algorithms Using Patient Data: Insights from the TUM.ai Makeathon](http://arxiv.org/pdf/2510.25277v1)

Authors: Simon Süwer, Mai Khanh Mai, Christoph Klein, Nicola Götzenberger, Denis Dalić, Andreas Maier, Jan Baumbach

The integration of clinical data offers significant potential for the
development of personalized medicine. However, its use is severely restricted
by the General Data Protection Regulation (GDPR), especially for small cohorts
with rare diseases. High-quality, structured data is essential for the
development of predictive medical AI. In this case study, we propose a novel,
multi-stage approach to secure AI training: (1) The model is designed on a
simulated clinical knowledge graph (cKG). This graph is used exclusively to
represent the structural characteristics of the real cKG without revealing any
sensitive content. (2) The model is then integrated into the FeatureCloud (FC)
federated learning framework, where it is prepared in a single-client
configuration within a protected execution environment. (3) Training then takes
place within the hospital environment on the real cKG, either under the direct
supervision of hospital staff or via a fully automated pipeline controlled by
the hospital. (4) Finally, verified evaluation scripts are executed, which only
return aggregated performance metrics. This enables immediate performance
feedback without sensitive patient data or individual predictions, leaving the
clinic. A fundamental element of this approach involves the incorporation of a
cKG, which serves to organize multi-omics and patient data within the context
of real-world hospital environments. This approach was successfully validated
during the TUM.ai Makeathon 2024 (TUMaiM24) challenge set by the Dr. von Hauner
Children's Hospital (HCH-LMU): 50 students developed models for patient
classification and diagnosis without access to real data. Deploying secure
algorithms via federated frameworks, such as the FC framework, could be a
practical way of achieving privacy-preserving AI in healthcare.

### 4. [Scheduling Data-Intensive Workloads in Large-Scale Distributed Systems: Trends and Challenges](http://arxiv.org/pdf/2510.25362v1)

Authors: Georgios L. Stavrinides, Helen D. Karatza

With the explosive growth of big data, workloads tend to get more complex and
computationally demanding. Such applications are processed on distributed
interconnected resources that are becoming larger in scale and computational
capacity. Data-intensive applications may have different degrees of parallelism
and must effectively exploit data locality. Furthermore, they may impose
several Quality of Service requirements, such as time constraints and
resilience against failures, as well as other objectives, like energy
efficiency. These features of the workloads, as well as the inherent
characteristics of the computing resources required to process them, present
major challenges that require the employment of effective scheduling
techniques. In this chapter, a classification of data-intensive workloads is
proposed and an overview of the most commonly used approaches for their
scheduling in large-scale distributed systems is given. We present novel
strategies that have been proposed in the literature and shed light on open
challenges and future directions.

### 5. [Holon Streaming: Global Aggregations with Windowed CRDTs](http://arxiv.org/pdf/2510.25757v1)

Authors: Jonas Spenger, Kolya Krafeld, Ruben van Gemeren, Philipp Haller, Paris Carbone

Scaling global aggregations is a challenge for exactly-once stream processing
systems. Current systems implement these either by computing the aggregation in
a single task instance, or by static aggregation trees, which limits
scalability and may become a bottleneck. Moreover, the end-to-end latency is
determined by the slowest path in the tree, and failures and reconfiguration
cause large latency spikes due to the centralized coordination. Towards these
issues, we present Holon Streaming, an exactly-once stream processing system
for global aggregations. Its deterministic programming model uses windowed
conflict-free replicated data types (Windowed CRDTs), a novel abstraction for
shared replicated state. Windowed CRDTs make computing global aggregations
scalable. Furthermore, their guarantees such as determinism and convergence
enable the design of efficient failure recovery algorithms by decentralized
coordination. Our evaluation shows a 5x lower latency and 2x higher throughput
than an existing stream processing system on global aggregation workloads, with
an 11x latency reduction under failure scenarios. The paper demonstrates the
effectiveness of decentralized coordination with determinism, and the utility
of Windowed CRDTs for global aggregations.

### 6. [Timing Games in Responsive Consensus Protocols](http://arxiv.org/pdf/2510.25144v1)

Authors: Kaya Alpturer, Kushal Babel, Aditya Saraf

Optimistic responsiveness -- the ability of a consensus protocol to operate
at the speed of the network -- is widely used in consensus protocol design to
optimize latency and throughput. However, blockchain applications incentivize
validators to play timing games by strategically delaying their proposals,
since increased block time correlates with greater rewards. Consequently, it
may appear that responsiveness (even under optimistic conditions) is impossible
in blockchain protocols. In this work, we develop a model of timing games in
responsive consensus protocols and find a prisoner's dilemma structure, where
cooperation (proposing promptly) is in the validators' best interest, but
individual incentives encourage validators to delay proposals selfishly. To
attain desirable equilibria, we introduce dynamic block rewards that decrease
with round time to explicitly incentivize faster proposals. Delays are measured
through a voting mechanism, where other validators vote on the current leader's
round time. By carefully setting the protocol parameters, the voting mechanism
allows validators to coordinate and reach the cooperative equilibrium,
benefiting all through a higher rate-of-reward. Thus, instead of responsiveness
being an unattainable property due to timing games, we show that responsiveness
itself can promote faster block proposals. One consequence of moving from a
static to dynamic block reward is that validator utilities become more
sensitive to latency, worsening the gap between the best- and worst-connected
validators. Our analysis shows, however, that this effect is minor in both
theoretical latency models and simulations based on real-world networks.

### 7. [Can Like Attract Like? A Study of Homonymous Gathering in Networks](http://arxiv.org/pdf/2510.25451v1)

Authors: Stéphane Devismes, Yoann Dieudonné, Arnaud Labourel

A team of mobile agents, starting from distinct nodes of a network, have to
meet at the same node and declare that they all met. Agents execute the same
algorithm, which they start when activated by an adversary or by an agent
entering their initial node. When activated, agents traverse edges of the
network in synchronous rounds. Their perception and communication are strictly
local. This task, known as gathering, is a central problem in distributed
mobile systems. Most prior work focuses on minimizing its time complexity,
i.e., the worst-case number of rounds between the start of the earliest agent
and the task completion. To break possible symmetries, deterministic solutions
typically assume that agents have pairwise distinct IDs, called labels, known
only to themselves. But must all labels be pairwise distinct to guarantee
deterministic gathering?
  We address this question by considering agents that may share the same label.
A team L is said to be gatherable if, for every initial setting of L, there is
an algorithm that solves gathering. Our contribution is threefold. (1) We give
a full characterization of the gatherable teams. (2) We design an algorithm
that gathers all of them in poly$(n,\log\lambda)$ time, where $n$ (resp.
$\lambda$) is the graph order (resp. the smallest label in L). This algorithm
requires the agents to initially share only $O(\log \log \log \mu)$ bits of
common knowledge, where $\mu$ is the largest label multiplicity in L. (3) We
show this dependency is almost optimal to get a poly$(n,\log\lambda)$-time
complexity.
  As a by-product, we get the first deterministic poly$(n,\log\lambda)$-time
algorithm requiring no common knowledge to gather any team when all labels are
distinct. Known to be achievable for two-agent teams, extending this to any
team size faced a major challenge: termination detection. Our techniques to
address it may be of independent interest.

### 8. [The Singularity Theory of Concurrent Programs: A Topological Characterization and Detection of Deadlocks and Livelocks](http://arxiv.org/pdf/2510.25112v1)

Authors: Di Zhang

This paper introduces a novel paradigm for the analysis and verification of
concurrent programs -- the Singularity Theory. We model the execution space of
a concurrent program as a branched topological space, where program states are
points and state transitions are paths. Within this framework, we characterize
deadlocks as attractors and livelocks as non-contractible loops in the
execution space. By employing tools from algebraic topology, particularly
homotopy and homology groups, we define a series of concurrent topological
invariants to systematically detect and classify these concurrent
"singularities" without exhaustively traversing all states. This work aims to
establish a geometric and topological foundation for concurrent program
verification, transcending the limitations of traditional model checking.

### 9. [Effect of Full Common Randomness Replication in Symmetric PIR on Graph-Based Replicated Systems](http://arxiv.org/pdf/2510.25736v1)

Authors: Shreya Meel, Sennur Ulukus

We revisit the problem of symmetric private information retrieval (SPIR) in
settings where the database replication is modeled by a simple graph. Here,
each vertex corresponds to a server, and a message is replicated on two servers
if and only if there is an edge between them. To satisfy the requirement of
database privacy, we let all the servers share some common randomness,
independent of the messages. We aim to quantify the improvement in SPIR
capacity, i.e., the maximum ratio of the number of desired and downloaded
symbols, compared to the setting with graph-replicated common randomness.
Towards this, we develop an algorithm to convert a class of PIR schemes into
the corresponding SPIR schemes, thereby establishing a capacity lower bound on
graphs for which such schemes exist. This includes the class of path and cyclic
graphs for which we derive capacity upper bounds that are tighter than the
trivial bounds given by the respective PIR capacities. For the special case of
path graph with three vertices, we identify the SPIR capacity to be
$\frac{1}{2}$.

### 10. [Machine Learning and CPU (Central Processing Unit) Scheduling Co-Optimization over a Network of Computing Centers](http://arxiv.org/pdf/2510.25176v1)

Authors: Mohammadreza Doostmohammadian, Zulfiya R. Gabidullina, Hamid R. Rabiee

In the rapidly evolving research on artificial intelligence (AI) the demand
for fast, computationally efficient, and scalable solutions has increased in
recent years. The problem of optimizing the computing resources for distributed
machine learning (ML) and optimization is considered in this paper. Given a set
of data distributed over a network of computing-nodes/servers, the idea is to
optimally assign the CPU (central processing unit) usage while simultaneously
training each computing node locally via its own share of data. This formulates
the problem as a co-optimization setup to (i) optimize the data processing and
(ii) optimally allocate the computing resources. The information-sharing
network among the nodes might be time-varying, but with balanced weights to
ensure consensus-type convergence of the algorithm. The algorithm is all-time
feasible, which implies that the computing resource-demand balance constraint
holds at all iterations of the proposed solution. Moreover, the solution allows
addressing possible log-scale quantization over the information-sharing
channels to exchange log-quantized data. For some example applications,
distributed support-vector-machine (SVM) and regression are considered as the
ML training models. Results from perturbation theory, along with Lyapunov
stability and eigen-spectrum analysis, are used to prove the convergence
towards the optimal case. As compared to existing CPU scheduling solutions, the
proposed algorithm improves the cost optimality gap by more than $50\%$.

### Digital Libraries

### 1. [Measuring the Research Output and Performance of the University of Ibadan from 2014 to 2023: A Scientometric Analysis](http://arxiv.org/pdf/2510.25283v1)

Authors: Muneer Ahmad, Undie Felicia Nkatv

This study employs scientometric methods to assess the research output and
performance of the University of Ibadan from 2014 to 2023. By analyzing
publication trends, citation patterns, and collaboration networks, the research
aims to comprehensively evaluate the university's research productivity,
impact, and disciplinary focus. This article's endeavors are characterized by
innovation, interdisciplinary collaboration, and commitment to excellence,
making the University of Ibadan a significant hub for cutting-edge research in
Nigeria and beyond. The goal of the current study is to ascertain the influence
of the university's research output and publication patterns between 2014 and
2023. The study focuses on the departments at the University of Ibadan that
contribute the most, the best journals for publishing, the nations that
collaborate, the impact of citations both locally and globally, well-known
authors and their total production, and the research output broken down by
year. According to the university's ten-year publication data, 7159 papers with
an h-index of 75 were published between 2014 and 2023, garnering 218572
citations. Furthermore, the VOSviewer software mapping approach is used to
illustrate the stenographical mapping of data through graphs. The findings of
this study will contribute to understanding the university's research
strengths, weaknesses, and potential areas for improvement. Additionally, the
results will inform evidence-based decision-making for enhancing research
strategies and policies at the University of Ibadan.

### 2. [Retrieval-Augmented Search for Large-Scale Map Collections with ColPali](http://arxiv.org/pdf/2510.25718v1)

Authors: Jamie Mahowald, Benjamin Charles Germain Lee

Multimodal approaches have shown great promise for searching and navigating
digital collections held by libraries, archives, and museums. In this paper, we
introduce map-RAS: a retrieval-augmented search system for historic maps. In
addition to introducing our framework, we detail our publicly-hosted demo for
searching 101,233 map images held by the Library of Congress. With our system,
users can multimodally query the map collection via ColPali, summarize search
results using Llama 3.2, and upload their own collections to perform
inter-collection search. We articulate potential use cases for archivists,
curators, and end-users, as well as future work with our system in both machine
learning and the digital humanities. Our demo can be viewed at:
http://www.mapras.com.

### Discrete Mathematics

### 1. [Why Districting Becomes NP-hard](http://arxiv.org/pdf/2510.25614v1)

Authors: Niklas Jost, Adolfo Escobedo, Alice Kirchheim

This paper investigates why and when the edge-based districting problem
becomes computationally intractable. The overall problem is represented as an
exact mathematical programming formulation consisting of an objective function
and several constraint groups, each enforcing a well-known districting
criterion such as balance, contiguity, or compactness. While districting is
known to be NP-hard in general, we study what happens when specific constraint
groups are relaxed or removed. The results identify precise boundaries between
tractable subproblems (in P) and intractable ones (NP-hard). The paper also
discusses implications on node-based analogs of the featured districting
problems, and it considers alternative notions of certain criteria in its
analysis.

### 2. [A Tight Lower Bound on Cubic Vertices and Upper Bounds on Thin and Non-thin edges in Planar Braces](http://arxiv.org/pdf/2510.25188v1)

Authors: Koustav De

For a subset $X$ of the vertex set $\VV(\GG)$ of a graph $\GG$, we denote the
set of edges of $\GG$ which have exactly one end in $X$ by $\partial(X)$ and
refer to it as the cut of $X$ or edge cut $\partial(X)$. A graph
$\GG=(\VV,\EE)$ is called matching covered if $\forall e \in \EE(\GG), ~\exists
\text{a perfect matching }M \text{ of }\GG \text{ s. t. } e \in M$. A cut $C$
of a matching covered graph $\GG$ is a separating cut if and only if, given any
edge $e$, there is a perfect matching $M_{e}$ of $\GG$ such that $e \in M_{e}$
and $|C \cap M_{e}| = 1$. A cut $C$ in a matching covered graph $\GG$ is a
tight cut of $\GG$ if $|C \cap M| = 1$ for every perfect matching $M$ of $\GG$.
For, $X, Y \subseteq \VV(\GG)$, we denote the set of edges of $\EE(\GG)$ which
have one endpoint in $X$ and the other endpoint in $Y$ by $E[X,Y]$. Let
$\partial(X)=E[X,\overline{X}]$ be an edge cut, where $\overline{X}=\VV(\GG)
\setminus X$. An edge cut is trivial if $|X|=1$ or $|\overline{X}|=1$. A
matching covered graph, which is free of nontrivial tight cuts, is a brace if
it is bipartite and is a brick if it is non-bipartite. An edge $e$ in a brace
$\GG$ is \emph{thin} if, for every tight cut $\partial(X)$ of $\GG - e$, $|X|
\le 3$ or $|\overline{X}| \le 3$.
  Carvalho, Lucchesi and Murty conjectured that there exists a positive
constant $c$ such that every brace $\GG$ has $c|\VV(\GG)|$ thin edges
\cite{DBLP:journals/combinatorics/LucchesiCM15}. He and Lu \cite{HE2025153}
showed a lower bound of thin edges in a brace in terms of the number of cubic
vertices. We asked whether any planar brace exists that does not contain any
cubic vertices. We answer negatively by showing that such set of planar braces
is empty. We have been able to show a quantitively tight lower bound on the
number of cubic vertices in a planar brace. We have proved tight upper bounds
of nonthin edges and thin edges in a planar brace.

### 3. [Fractional Iterates and Oscillatory Convergence](http://arxiv.org/pdf/2510.25606v1)

Authors: Steven Finch

The simple continued fractions for the Golden & Silver means are well-known.
It is astonishing that, as far as we know, no one has published half-iterates
(let alone quarter-iterates) for the corresponding algorithms. We also examine
the cosine and logistic maps (with parameter $2 < \lambda < 3$).

### Data Structures and Algorithms

### 1. [Hedgegraph Polymatroids](http://arxiv.org/pdf/2510.25043v1)

Authors: Karthekeyan Chandrasekaran, Chandra Chekuri, Weihang Wang, Weihao Zhu

Graphs and hypergraphs combine expressive modeling power with algorithmic
efficiency for a wide range of applications. Hedgegraphs generalize hypergraphs
further by grouping hyperedges under a color/hedge. This allows hedgegraphs to
model dependencies between hyperedges and leads to several applications.
However, it poses algorithmic challenges. In particular, the cut function is
not submodular, which has been a barrier to algorithms for connectivity. In
this work, we introduce two alternative partition-based measures of
connectivity in hedgegraphs and study their structural and algorithmic aspects.
Instead of the cut function, we investigate a polymatroid associated with
hedgegraphs. The polymatroidal lens leads to new tractability results as well
as insightful generalizations of classical results on graphs and hypergraphs.

### 2. [$\{s,t\}$-Separating Principal Partition Sequence of Submodular Functions](http://arxiv.org/pdf/2510.25664v1)

Authors: Kristóf Bérczi, Karthekeyan Chandrasekaran, Tamás Király, Daniel P. Szabo

Narayanan and Fujishige showed the existence of the principal partition
sequence of a submodular function, a structure with numerous applications in
areas such as clustering, fast algorithms, and approximation algorithms. In
this work, motivated by two applications, we develop a theory of
$\{s,t\}$-separating principal partition sequence of a submodular function. We
define this sequence, show its existence, and design a polynomial-time
algorithm to construct it. We show two applications: (1) approximation
algorithm for the $\{s,t\}$-separating submodular $k$-partitioning problem for
monotone and posimodular functions and (2) polynomial-time algorithm for the
hypergraph orientation problem of finding an orientation that simultaneously
has strong connectivity at least $k$ and $(s,t)$-connectivity at least $\ell$.

### 3. [Can Like Attract Like? A Study of Homonymous Gathering in Networks](http://arxiv.org/pdf/2510.25451v1)

Authors: Stéphane Devismes, Yoann Dieudonné, Arnaud Labourel

A team of mobile agents, starting from distinct nodes of a network, have to
meet at the same node and declare that they all met. Agents execute the same
algorithm, which they start when activated by an adversary or by an agent
entering their initial node. When activated, agents traverse edges of the
network in synchronous rounds. Their perception and communication are strictly
local. This task, known as gathering, is a central problem in distributed
mobile systems. Most prior work focuses on minimizing its time complexity,
i.e., the worst-case number of rounds between the start of the earliest agent
and the task completion. To break possible symmetries, deterministic solutions
typically assume that agents have pairwise distinct IDs, called labels, known
only to themselves. But must all labels be pairwise distinct to guarantee
deterministic gathering?
  We address this question by considering agents that may share the same label.
A team L is said to be gatherable if, for every initial setting of L, there is
an algorithm that solves gathering. Our contribution is threefold. (1) We give
a full characterization of the gatherable teams. (2) We design an algorithm
that gathers all of them in poly$(n,\log\lambda)$ time, where $n$ (resp.
$\lambda$) is the graph order (resp. the smallest label in L). This algorithm
requires the agents to initially share only $O(\log \log \log \mu)$ bits of
common knowledge, where $\mu$ is the largest label multiplicity in L. (3) We
show this dependency is almost optimal to get a poly$(n,\log\lambda)$-time
complexity.
  As a by-product, we get the first deterministic poly$(n,\log\lambda)$-time
algorithm requiring no common knowledge to gather any team when all labels are
distinct. Known to be achievable for two-agent teams, extending this to any
team size faced a major challenge: termination detection. Our techniques to
address it may be of independent interest.

### 4. [Fast Dimensionality Reduction from $\ell_2$ to $\ell_p$](http://arxiv.org/pdf/2510.25541v1)

Authors: Rafael Chiclana, Mark Iwen

The Johnson-Lindenstrauss (JL) lemma is a fundamental result in
dimensionality reduction, ensuring that any finite set $X \subseteq
\mathbb{R}^d$ can be embedded into a lower-dimensional space $\mathbb{R}^k$
while approximately preserving all pairwise Euclidean distances. In recent
years, embeddings that preserve Euclidean distances when measured via the
$\ell_1$ norm in the target space have received increasing attention due to
their relevance in applications such as nearest neighbor search in high
dimensions. A recent breakthrough by Dirksen, Mendelson, and Stollenwerk
established an optimal $\ell_2 \to \ell_1$ embedding with computational
complexity $O(d \log d)$. In this work, we generalize this direction and
propose a simple linear embedding from $\ell_2$ to $\ell_p$ for any $p \in
[1,2]$ based on a construction of Ailon and Liberty. Our method achieves a
reduced runtime of $O(d \log k)$ when $k \leq d^{1/4}$, improving upon prior
runtime results when the target dimension is small. Additionally, we show that
for \emph{any norm} $\|\cdot\|$ in the target space, any embedding of
$(\mathbb{R}^d, \|\cdot\|_2)$ into $(\mathbb{R}^k, \|\cdot\|)$ with distortion
$\varepsilon$ generally requires $k = \Omega\big(\varepsilon^{-2}
\log(\varepsilon^2 n)/\log(1/\varepsilon)\big)$, matching the optimal bound for
the $\ell_2$ case up to a logarithmic factor.

### 5. [Exact zCDP Characterizations for Fundamental Differentially Private Mechanisms](http://arxiv.org/pdf/2510.25746v1)

Authors: Charlie Harrison, Pasin Manurangsi

Zero-concentrated differential privacy (zCDP) is a variant of differential
privacy (DP) that is widely used partly thanks to its nice composition
property. While a tight conversion from $\epsilon$-DP to zCDP exists for the
worst-case mechanism, many common algorithms satisfy stronger guarantees. In
this work, we derive tight zCDP characterizations for several fundamental
mechanisms. We prove that the tight zCDP bound for the $\epsilon$-DP Laplace
mechanism is exactly $\epsilon + e^{-\epsilon} - 1$, confirming a recent
conjecture by Wang (2022). We further provide tight bounds for the discrete
Laplace mechanism, $k$-Randomized Response (for $k \leq 6$), and RAPPOR.
Lastly, we also provide a tight zCDP bound for the worst case bounded range
mechanism.

### 6. [Perturbation Bounds for Low-Rank Inverse Approximations under Noise](http://arxiv.org/pdf/2510.25571v1)

Authors: Phuc Tran, Nisheeth K. Vishnoi

Low-rank pseudoinverses are widely used to approximate matrix inverses in
scalable machine learning, optimization, and scientific computing. However,
real-world matrices are often observed with noise, arising from sampling,
sketching, and quantization. The spectral-norm robustness of low-rank inverse
approximations remains poorly understood. We systematically study the
spectral-norm error $\| (\tilde{A}^{-1})_p - A_p^{-1} \|$ for an $n\times n$
symmetric matrix $A$, where $A_p^{-1}$ denotes the best rank-\(p\)
approximation of $A^{-1}$, and $\tilde{A} = A + E$ is a noisy observation.
Under mild assumptions on the noise, we derive sharp non-asymptotic
perturbation bounds that reveal how the error scales with the eigengap,
spectral decay, and noise alignment with low-curvature directions of $A$. Our
analysis introduces a novel application of contour integral techniques to the
\emph{non-entire} function $f(z) = 1/z$, yielding bounds that improve over
naive adaptations of classical full-inverse bounds by up to a factor of
$\sqrt{n}$. Empirically, our bounds closely track the true perturbation error
across a variety of real-world and synthetic matrices, while estimates based on
classical results tend to significantly overpredict. These findings offer
practical, spectrum-aware guarantees for low-rank inverse approximations in
noisy computational environments.

### 7. [Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](http://arxiv.org/pdf/2510.25670v1)

Authors: Phuc Tran, Nisheeth K. Vishnoi, Van H. Vu

A central challenge in machine learning is to understand how noise or
measurement errors affect low-rank approximations, particularly in the spectral
norm. This question is especially important in differentially private low-rank
approximation, where one aims to preserve the top-$p$ structure of a
data-derived matrix while ensuring privacy. Prior work often analyzes Frobenius
norm error or changes in reconstruction quality, but these metrics can over- or
under-estimate true subspace distortion. The spectral norm, by contrast,
captures worst-case directional error and provides the strongest utility
guarantees. We establish new high-probability spectral-norm perturbation bounds
for symmetric matrices that refine the classical Eckart--Young--Mirsky theorem
and explicitly capture interactions between a matrix $A \in \mathbb{R}^{n
\times n}$ and an arbitrary symmetric perturbation $E$. Under mild eigengap and
norm conditions, our bounds yield sharp estimates for $\|(A + E)_p - A_p\|$,
where $A_p$ is the best rank-$p$ approximation of $A$, with improvements of up
to a factor of $\sqrt{n}$. As an application, we derive improved utility
guarantees for differentially private PCA, resolving an open problem in the
literature. Our analysis relies on a novel contour bootstrapping method from
complex analysis and extends it to a broad class of spectral functionals,
including polynomials and matrix exponentials. Empirical results on real-world
datasets confirm that our bounds closely track the actual spectral error under
diverse perturbation regimes.

### Emerging Technologies

### 1. [Modulation Schemes for Functionalized Vesicle-based MC Transmitters](http://arxiv.org/pdf/2510.25676v1)

Authors: Teena tom Dieck, Lukas Brand, Sebastian Lotter, Kathrin Castiglione, Robert Schober, Maximilian Schäfer

Molecular communication (MC) enables information exchange through the
transmission of signaling molecules (SMs) and holds promise for many innovative
applications. However, most existing MC studies rely on simplified transmitter
(TX) models that do not account for the physical and biochemical limitations of
realistic biological hardware. This work extends previous efforts toward
developing models for practical MC systems by proposing a more realistic TX
model that incorporates the delay in SM release and TX noise introduced by
biological components. Building on this more realistic, functionalized
vesicle-based TX model, we propose two novel modulation schemes specifically
designed for this TX to mitigate TX-induced memory effects that arise from
delayed and imperfectly controllable SM release. The proposed modulation
schemes enable low-complexity receiver designs by mitigating memory effects
directly at the TX. Numerical evaluations demonstrate that the proposed schemes
improve communication reliability under realistic biochemical constraints,
offering an important step toward physically realizable MC systems.

### Formal Languages and Automata Theory

### 1. [Systems of Graph Formulas and their Equivalence to Alternating Graph Automata](http://arxiv.org/pdf/2510.25260v1)

Authors: Frank Drewes, Berthold Hoffmann, Mark Minas

Graph-based modeling plays a fundamental role in many areas of computer
science. In this paper, we introduce systems of graph formulas with variables
for specifying graph properties; this notion generalizes the graph formulas
introduced in earlier work by incorporating recursion. We show that these
formula systems have the same expressive power as alternating graph automata, a
computational model that extends traditional finite-state automata to graphs,
and allows both existential and universal states. In particular, we provide a
bidirectional translation between formula systems and alternating graph
automata, proving their equivalence in specifying graph languages. This result
implies that alternating graph automata can be naturally represented using
logic-based formulations, thus bridging the gap between automata-theoretic and
logic-based approaches to graph language specification.

### 2. [Have a thing? Reasoning around recursion with dynamic typing in grounded arithmetic](http://arxiv.org/pdf/2510.25369v1)

Authors: Elliot Bobrow, Bryan Ford, Stefan Milenković

Neither the classical nor intuitionistic logic traditions are
perfectly-aligned with the purpose of reasoning about computation, in that
neither logical tradition can normally permit the direct expression of
arbitrary general-recursive functions without inconsistency. We introduce
grounded arithmetic or GA, a minimalistic but nonetheless powerful foundation
for formal reasoning that allows the direct expression of arbitrary recursive
definitions. GA adjusts the traditional inference rules such that terms that
express nonterminating computations harmlessly denote no semantic value (i.e.,
"bottom") instead of leading into logical paradox or inconsistency. Recursive
functions may be proven terminating in GA essentially by "dynamically typing"
terms, or equivalently, symbolically reverse-executing the computations they
denote via GA's inference rules. Once recursive functions have been proven
terminating, logical reasoning about their results reduce to the familiar
classical rules. A mechanically-checked consistency proof in Isabelle/HOL
exists for the basic quantifier-free fragment of GA. Quantifiers may be added
atop this foundation as ordinary computations, whose inference rules are thus
admissible and do not introduce new inconsistency risks. While GA is only a
first step towards richly-typed grounded deduction practical for everyday use
in manual or automated computational reasoning, it shows the promise that the
expressive freedom of arbitrary recursive definition can in principle be
incorporated into formal systems.

### Graphics

### 1. [Off-Centered WoS-Type Solvers with Statistical Weighting](http://arxiv.org/pdf/2510.25152v1)

Authors: Anchang Bao, Jie Xu, Enya Shen, Jianmin Wang

Stochastic PDE solvers have emerged as a powerful alternative to traditional
discretization-based methods for solving partial differential equations (PDEs),
especially in geometry processing and graphics. While off-centered estimators
enhance sample reuse in WoS-type Monte Carlo solvers, they introduce
correlation artifacts and bias when Green's functions are approximated. In this
paper, we propose a statistically weighted off-centered WoS-type estimator that
leverages local similarity filtering to selectively combine samples across
neighboring evaluation points. Our method balances bias and variance through a
principled weighting strategy that suppresses unreliable estimators. We
demonstrate our approach's effectiveness on various PDEs,including screened
Poisson equations and boundary conditions, achieving consistent improvements
over existing solvers such as vanilla Walk on Spheres, mean value caching, and
boundary value caching. Our method also naturally extends to gradient field
estimation and mixed boundary problems.

### 2. [Fast and Robust Point Containment Queries on Trimmed Surface](http://arxiv.org/pdf/2510.25159v1)

Authors: Anchang Bao, Enya Shen, Jianmin Wang

Point containment queries on trimmed surfaces are fundamental to CAD
modeling, solid geometry processing, and surface tessellation. Existing
approaches such as ray casting and generalized winding numbers often face
limitations in robustness and computational efficiency.
  We propose a fast and numerically stable method for performing containment
queries on trimmed surfaces, including those with periodic parameterizations.
Our approach introduces a recursive winding number computation scheme that
replaces costly curve subdivision with an ellipse-based bound for Bezier
segments, enabling linear-time evaluation. For periodic surfaces, we lift
trimming curves to the universal covering space, allowing accurate and
consistent winding number computation even for non-contractible or
discontinuous loops in parameter domain.
  Experiments show that our method achieves substantial speedups over existing
winding-number algorithms while maintaining high robustness in the presence of
geometric noise, open boundaries, and periodic topologies. We further
demonstrate its effectiveness in processing real B-Rep models and in robust
tessellation of trimmed surfaces.

### 3. [mitransient: Transient light transport in Mitsuba 3](http://arxiv.org/pdf/2510.25660v1)

Authors: Diego Royo, Jorge Garcia-Pueyo, Miguel Crespo, Óscar Pueyo-Ciutad, Guillermo Enguita, Diego Bielsa

mitransient is a light transport simulation tool that extends Mitsuba 3 with
support for time-resolved simulations. In essence, mitransient extends
conventional rendering by adding a temporal dimension which accounts for the
time of flight of light. This allows rapid prototyping of novel transient
imaging systems without the need of costly or difficult-to-operate hardware.
Our code is trivially easy to install through pip, and consists of Python
modules that can run both in CPU and GPU by leveraging the JIT capabilities of
Mitsuba 3. It provides physically-based simulations of complex phenomena,
including a wide variety of realistic materials and participating media such as
fog or smoke. In addition, we extend Mitsuba 3's functionality to support
time-resolved polarization tracking of light and transient differentiable
rendering. Finally, we also include tools that simplify the use of our
simulations for non-line-of-sight imaging, enabling realistic scene setups with
capture noise to be simulated in just seconds of minutes. Altogether, we hope
that mitransient will support the research community in developing novel
algorithms for transient imaging.

### 4. [4-Doodle: Text to 3D Sketches that Move!](http://arxiv.org/pdf/2510.25319v1)

Authors: Hao Chen, Jiaqi Wang, Yonggang Qi, Ke Li, Kaiyue Pang, Yi-Zhe Song

We present a novel task: text-to-3D sketch animation, which aims to bring
freeform sketches to life in dynamic 3D space. Unlike prior works focused on
photorealistic content generation, we target sparse, stylized, and
view-consistent 3D vector sketches, a lightweight and interpretable medium
well-suited for visual communication and prototyping. However, this task is
very challenging: (i) no paired dataset exists for text and 3D (or 4D)
sketches; (ii) sketches require structural abstraction that is difficult to
model with conventional 3D representations like NeRFs or point clouds; and
(iii) animating such sketches demands temporal coherence and multi-view
consistency, which current pipelines do not address. Therefore, we propose
4-Doodle, the first training-free framework for generating dynamic 3D sketches
from text. It leverages pretrained image and video diffusion models through a
dual-space distillation scheme: one space captures multi-view-consistent
geometry using differentiable B\'ezier curves, while the other encodes motion
dynamics via temporally-aware priors. Unlike prior work (e.g., DreamFusion),
which optimizes from a single view per step, our multi-view optimization
ensures structural alignment and avoids view ambiguity, critical for sparse
sketches. Furthermore, we introduce a structure-aware motion module that
separates shape-preserving trajectories from deformation-aware changes,
enabling expressive motion such as flipping, rotation, and articulated
movement. Extensive experiments show that our method produces temporally
realistic and structurally stable 3D sketch animations, outperforming existing
baselines in both fidelity and controllability. We hope this work serves as a
step toward more intuitive and accessible 4D content creation.

### 5. [FreeArt3D: Training-Free Articulated Object Generation using 3D Diffusion](http://arxiv.org/pdf/2510.25765v1)

Authors: Chuhao Chen, Isabella Liu, Xinyue Wei, Hao Su, Minghua Liu

Articulated 3D objects are central to many applications in robotics, AR/VR,
and animation. Recent approaches to modeling such objects either rely on
optimization-based reconstruction pipelines that require dense-view supervision
or on feed-forward generative models that produce coarse geometric
approximations and often overlook surface texture. In contrast, open-world 3D
generation of static objects has achieved remarkable success, especially with
the advent of native 3D diffusion models such as Trellis. However, extending
these methods to articulated objects by training native 3D diffusion models
poses significant challenges. In this work, we present FreeArt3D, a
training-free framework for articulated 3D object generation. Instead of
training a new model on limited articulated data, FreeArt3D repurposes a
pre-trained static 3D diffusion model (e.g., Trellis) as a powerful shape
prior. It extends Score Distillation Sampling (SDS) into the 3D-to-4D domain by
treating articulation as an additional generative dimension. Given a few images
captured in different articulation states, FreeArt3D jointly optimizes the
object's geometry, texture, and articulation parameters without requiring
task-specific training or access to large-scale articulated datasets. Our
method generates high-fidelity geometry and textures, accurately predicts
underlying kinematic structures, and generalizes well across diverse object
categories. Despite following a per-instance optimization paradigm, FreeArt3D
completes in minutes and significantly outperforms prior state-of-the-art
approaches in both quality and versatility.

### 6. [Learning Disentangled Speech- and Expression-Driven Blendshapes for 3D Talking Face Animation](http://arxiv.org/pdf/2510.25234v1)

Authors: Yuxiang Mao, Zhijie Zhang, Zhiheng Zhang, Jiawei Liu, Chen Zeng, Shihong Xia

Expressions are fundamental to conveying human emotions. With the rapid
advancement of AI-generated content (AIGC), realistic and expressive 3D facial
animation has become increasingly crucial. Despite recent progress in
speech-driven lip-sync for talking-face animation, generating emotionally
expressive talking faces remains underexplored. A major obstacle is the
scarcity of real emotional 3D talking-face datasets due to the high cost of
data capture. To address this, we model facial animation driven by both speech
and emotion as a linear additive problem. Leveraging a 3D talking-face dataset
with neutral expressions (VOCAset) and a dataset of 3D expression sequences
(Florence4D), we jointly learn a set of blendshapes driven by speech and
emotion. We introduce a sparsity constraint loss to encourage disentanglement
between the two types of blendshapes while allowing the model to capture
inherent secondary cross-domain deformations present in the training data. The
learned blendshapes can be further mapped to the expression and jaw pose
parameters of the FLAME model, enabling the animation of 3D Gaussian avatars.
Qualitative and quantitative experiments demonstrate that our method naturally
generates talking faces with specified expressions while maintaining accurate
lip synchronization. Perceptual studies further show that our approach achieves
superior emotional expressivity compared to existing methods, without
compromising lip-sync quality.

### Computer Science and Game Theory

### 1. [Timing Games in Responsive Consensus Protocols](http://arxiv.org/pdf/2510.25144v1)

Authors: Kaya Alpturer, Kushal Babel, Aditya Saraf

Optimistic responsiveness -- the ability of a consensus protocol to operate
at the speed of the network -- is widely used in consensus protocol design to
optimize latency and throughput. However, blockchain applications incentivize
validators to play timing games by strategically delaying their proposals,
since increased block time correlates with greater rewards. Consequently, it
may appear that responsiveness (even under optimistic conditions) is impossible
in blockchain protocols. In this work, we develop a model of timing games in
responsive consensus protocols and find a prisoner's dilemma structure, where
cooperation (proposing promptly) is in the validators' best interest, but
individual incentives encourage validators to delay proposals selfishly. To
attain desirable equilibria, we introduce dynamic block rewards that decrease
with round time to explicitly incentivize faster proposals. Delays are measured
through a voting mechanism, where other validators vote on the current leader's
round time. By carefully setting the protocol parameters, the voting mechanism
allows validators to coordinate and reach the cooperative equilibrium,
benefiting all through a higher rate-of-reward. Thus, instead of responsiveness
being an unattainable property due to timing games, we show that responsiveness
itself can promote faster block proposals. One consequence of moving from a
static to dynamic block reward is that validator utilities become more
sensitive to latency, worsening the gap between the best- and worst-connected
validators. Our analysis shows, however, that this effect is minor in both
theoretical latency models and simulations based on real-world networks.

### 2. [On Robust Popular Matchings with Tie-Bounded Preferences and Stable Matchings with Two-Sided Ties](http://arxiv.org/pdf/2510.25209v1)

Authors: Koustav De

We are given a bipartite graph $G = \left( A \cup B, E \right)$. In the
one-sided model, every $a \in A$ (often called agents) ranks its neighbours $z
\in N_{a}$ strictly, and no $b \in B$ has any preference order over its
neighbours $y \in N_{b}$, and vertices in $B$ abstain from casting their votes
to matchings. In the two-sided model with one-sided ties, every $a \in A$ ranks
its neighbours $z \in N_{a}$ strictly, and every $b \in B$ puts all of its
neighbours into a single large tie, i.e., $b \in B$ prefers every $y \in N_{b}$
equally. In this two-sided model with one-sided ties, when two matchings
compete in a majority election, $b \in B$ abstains from casting its vote for a
matching when both the matchings saturate $b$ or both leave $b$ unsaturated;
else $b$ prefers the matching where it is saturated. A popular matching $M$ is
\emph{robust} if it remains popular among multiple instances.
  We have analysed the cases when a robust popular matching exists in the
one-sided model where only one agent alters her preference order among the
instances, and we have proposed a polynomial-time algorithm to decide if there
exists a robust popular matching when instances differ only with respect to the
preference orders of a single agent.
  We give a simple characterisation of popular matchings in the two-sided model
with one-sided ties. We show that in the two-sided model with one-sided ties,
if the input instances differ only with respect to the preference orders of a
single agent, there is a polynomial-time algorithm to decide whether there
exists a robust popular matching. We have been able to decide the stable
matching problem in bipartite graphs $G = (A \cup B, E)$ where \textit{both}
sides have weak preferences (ties allowed), with the restriction that every tie
has length at most $k$.

### 3. [Learning-Augmented Online Bidding in Stochastic Settings](http://arxiv.org/pdf/2510.25582v1)

Authors: Spyros Angelopoulos, Bertrand Simon

Online bidding is a classic optimization problem, with several applications
in online decision-making, the design of interruptible systems, and the
analysis of approximation algorithms. In this work, we study online bidding
under learning-augmented settings that incorporate stochasticity, in either the
prediction oracle or the algorithm itself. In the first part, we study bidding
under distributional predictions, and find Pareto-optimal algorithms that offer
the best-possible tradeoff between the consistency and the robustness of the
algorithm. In the second part, we study the power and limitations of randomized
bidding algorithms, by presenting upper and lower bounds on the
consistency/robustness tradeoffs. Previous works focused predominantly on
oracles that do not leverage stochastic information on the quality of the
prediction, and deterministic algorithms.

### 4. [Monopoly Deal: A Benchmark Environment for Bounded One-Sided Response Games](http://arxiv.org/pdf/2510.25080v1)

Authors: Will Wolf

Card games are widely used to study sequential decision-making under
uncertainty, with real-world analogues in negotiation, finance, and
cybersecurity. Typically, these games fall into three categories based on the
flow of control: strictly-sequential (where players alternate single actions),
deterministic-response (where some actions trigger a fixed outcome), and
unbounded reciprocal-response (where alternating counterplays are permitted). A
less-explored but strategically rich structure exists: the bounded one-sided
response. This dynamic occurs when a player's action briefly transfers control
to the opponent, who must satisfy a fixed condition through one or more
sequential moves before the turn resolves. We term games featuring this
mechanism Bounded One-Sided Response Games (BORGs).
  We introduce a modified version of Monopoly Deal as a benchmark environment
that specifically isolates the BORG dynamic, where a Rent action forces the
opponent to sequentially choose payment assets. We demonstrate that the
gold-standard algorithm, Counterfactual Regret Minimization (CFR), successfully
converges on effective strategies for this domain without requiring novel
algorithmic extensions. To support efficient, reproducible experimentation, we
present a lightweight, full-stack research platform that unifies the
environment, a parallelized CFR runtime, and a human-playable web interface,
all runnable on a single workstation. This system provides a practical
foundation for exploring state representation and policy learning in bounded
one-sided response settings.
  The trained CFR agent and source code are available at
https://monopolydeal.ai.

### Human-Computer Interaction

### 1. [CGM-Led Multimodal Tracking with Chatbot Support: An Autoethnography in Sub-Health](http://arxiv.org/pdf/2510.25381v1)

Authors: Dongyijie Primo Pan, Lan Luo, Yike Wang, Pan Hui

Metabolic disorders present a pressing global health challenge, with China
carrying the world's largest burden. While continuous glucose monitoring (CGM)
has transformed diabetes care, its potential for supporting sub-health
populations -- such as individuals who are overweight, prediabetic, or anxious
-- remains underexplored. At the same time, large language models (LLMs) are
increasingly used in health coaching, yet CGM is rarely incorporated as a
first-class signal. To address this gap, we conducted a six-week
autoethnography, combining CGM with multimodal indicators captured via common
digital devices and a chatbot that offered personalized reflections and
explanations of glucose fluctuations. Our findings show how CGM-led, data-first
multimodal tracking, coupled with conversational support, shaped everyday
practices of diet, activity, stress, and wellbeing. This work contributes to
HCI by extending CGM research beyond clinical diabetes and demonstrating how
LLM-driven agents can support preventive health and reflection in at-risk
populations.

### 2. [Small Talk, Big Impact? LLM-based Conversational Agents to Mitigate Passive Fatigue in Conditional Automated Driving](http://arxiv.org/pdf/2510.25421v1)

Authors: Lewis Cockram, Yueteng Yu, Jorge Pardo, Xiaomeng Li, Andry Rakotonirainy, Jonny Kuo, Sebastien Demmel, Mike Lenné, Ronald Schroeter

Passive fatigue during conditional automated driving can compromise driver
readiness and safety. This paper presents findings from a test-track study with
40 participants in a real-world rural automated driving scenario. In this
scenario, a Large Language Model (LLM) based conversational agent (CA) was
designed to check in with drivers and re-engage them with their surroundings.
Drawing on in-car video recordings, sleepiness ratings and interviews, we
analysed how drivers interacted with the agent and how these interactions
shaped alertness. Users found the CA helpful for supporting vigilance during
passive fatigue. Thematic analysis of acceptability further revealed three user
preference profiles that implicate future intention to use CAs. Positioning
empirically observed profiles within existing CA archetype frameworks
highlights the need for adaptive design sensitive to diverse user groups. This
work underscores the potential of CAs as proactive Human-Machine Interface
(HMI) interventions, demonstrating how natural language can support
context-aware interaction during automated driving.

### 3. [Psychoacoustic assessment of synthetic sounds for electric vehicles in a virtual reality experiment](http://arxiv.org/pdf/2510.25593v1)

Authors: Pavlo Bazilinskyy, Md Shadab Alam, Roberto Merino-Martınez

The growing adoption of electric vehicles, known for their quieter operation
compared to internal combustion engine vehicles, raises concerns about their
detectability, particularly for vulnerable road users. To address this,
regulations mandate the inclusion of exterior sound signals for electric
vehicles, specifying minimum sound pressure levels at low speeds. These
synthetic exterior sounds are often used in noisy urban environments, creating
the challenge of enhancing detectability without introducing excessive noise
annoyance. This study investigates the design of synthetic exterior sound
signals that balance high noticeability with low annoyance. An audiovisual
experiment with 14 participants was conducted using 15 virtual reality
scenarios featuring a passing car. The scenarios included various sound
signals, such as pure, intermittent, and complex tones at different
frequencies. Two baseline cases, a diesel engine and only tyre noise, were also
tested. Participants rated sounds for annoyance, noticeability, and
informativeness using 11-point ICBEN scales. The findings highlight how
psychoacoustic sound quality metrics predict annoyance ratings better than
conventional sound metrics, providing insight into optimising sound design for
electric vehicles. By improving pedestrian safety while minimising noise
pollution, this research supports the development of effective and
user-friendly exterior sound standards for electric vehicles.

### 4. [ggtime: A Grammar of Temporal Graphics](http://arxiv.org/pdf/2510.25656v1)

Authors: Cynthia A. Huang, Mitchell O'Hara-Wild, Rob J. Hyndman, Matthew Kay

Visualizing changes over time is fundamental to learning from the past and
anticipating the future. However, temporal semantics can be complicated, and
existing visualization tools often struggle to accurately represent these
complexities. It is common to use bespoke plot helper functions designed to
produce specific graphics, due to the absence of flexible general tools that
respect temporal semantics. We address this problem by proposing a grammar of
temporal graphics, and an associated software implementation, 'ggtime', that
encodes temporal semantics into a declarative grammar for visualizing temporal
data. The grammar introduces new composable elements that support visualization
across linear, cyclical, quasi-cyclical, and other granularities;
standardization of irregular durations; and alignment of time points across
different granularities and time zones. It is designed for interoperability
with other semantic variables, allowing navigation across the space of
visualizations while preserving temporal semantics.

### 5. [User Misconceptions of LLM-Based Conversational Programming Assistants](http://arxiv.org/pdf/2510.25662v1)

Authors: Gabrielle O'Brien, Antonio Pedro Santos Alves, Sebastian Baltes, Grischa Liebel, Mircea Lungu, Marcos Kalinowski

Programming assistants powered by large language models (LLMs) have become
widely available, with conversational assistants like ChatGPT proving
particularly accessible to less experienced programmers. However, the varied
capabilities of these tools across model versions and the mixed availability of
extensions that enable web search, code execution, or retrieval-augmented
generation create opportunities for user misconceptions about what systems can
and cannot do. Such misconceptions may lead to over-reliance, unproductive
practices, or insufficient quality control in LLM-assisted programming. Here,
we aim to characterize misconceptions that users of conversational LLM-based
assistants may have in programming contexts. Using a two-phase approach, we
first brainstorm and catalog user misconceptions that may occur, and then
conduct a qualitative analysis to examine whether these conceptual issues
surface in naturalistic Python-programming conversations with an LLM-based
chatbot drawn from an openly available dataset. Indeed, we see evidence that
some users have misplaced expectations about the availability of LLM-based
chatbot features like web access, code execution, or non-text output
generation. We also see potential evidence for deeper conceptual issues around
the scope of information required to debug, validate, and optimize programs.
Our findings reinforce the need for designing LLM-based tools that more clearly
communicate their programming capabilities to users.

### Information Retrieval

### 1. [Revisiting scalable sequential recommendation with Multi-Embedding Approach and Mixture-of-Experts](http://arxiv.org/pdf/2510.25285v1)

Authors: Qiushi Pan, Hao Wang, Guoyuan An, Luankang Zhang, Wei Guo, Yong Liu

In recommendation systems, how to effectively scale up recommendation models
has been an essential research topic. While significant progress has been made
in developing advanced and scalable architectures for sequential
recommendation(SR) models, there are still challenges due to items'
multi-faceted characteristics and dynamic item relevance in the user context.
To address these issues, we propose Fuxi-MME, a framework that integrates a
multi-embedding strategy with a Mixture-of-Experts (MoE) architecture.
Specifically, to efficiently capture diverse item characteristics in a
decoupled manner, we decompose the conventional single embedding matrix into
several lower-dimensional embedding matrices. Additionally, by substituting
relevant parameters in the Fuxi Block with an MoE layer, our model achieves
adaptive and specialized transformation of the enriched representations.
Empirical results on public datasets show that our proposed framework
outperforms several competitive baselines.

### 2. [Generalized Pseudo-Relevance Feedback](http://arxiv.org/pdf/2510.25488v1)

Authors: Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, Fen Lin, Qin Liu, Qingyao Ai

Query rewriting is a fundamental technique in information retrieval (IR). It
typically employs the retrieval result as relevance feedback to refine the
query and thereby addresses the vocabulary mismatch between user queries and
relevant documents. Traditional pseudo-relevance feedback (PRF) and its
vector-based extension (VPRF) improve retrieval performance by leveraging
top-retrieved documents as relevance feedback. However, they are constructed
based on two major hypotheses: the relevance assumption (top documents are
relevant) and the model assumption (rewriting methods need to be designed
specifically for particular model architectures). While recent large language
models (LLMs)-based generative relevance feedback (GRF) enables model-free
query reformulation, it either suffers from severe LLM hallucination or, again,
relies on the relevance assumption to guarantee the effectiveness of rewriting
quality. To overcome these limitations, we introduce an assumption-relaxed
framework: \textit{Generalized Pseudo Relevance Feedback} (GPRF), which
performs model-free, natural language rewriting based on retrieved documents,
not only eliminating the model assumption but also reducing dependence on the
relevance assumption. Specifically, we design a utility-oriented training
pipeline with reinforcement learning to ensure robustness against noisy
feedback. Extensive experiments across multiple benchmarks and retrievers
demonstrate that GPRF consistently outperforms strong baselines, establishing
it as an effective and generalizable framework for query rewriting.

### 3. [MMQ-v2: Align, Denoise, and Amplify: Adaptive Behavior Mining for Semantic IDs Learning in Recommendation](http://arxiv.org/pdf/2510.25622v1)

Authors: Yi Xu, Moyu Zhang, Chaofan Fan, Jinxin Hu, Xiaochen Li, Yu Zhang, Xiaoyi Zeng, Jing Zhang

Industrial recommender systems rely on unique Item Identifiers (ItemIDs).
However, this method struggles with scalability and generalization in large,
dynamic datasets that have sparse long-tail data.Content-based Semantic IDs
(SIDs) address this by sharing knowledge through content quantization. However,
by ignoring dynamic behavioral properties, purely content-based SIDs have
limited expressive power. Existing methods attempt to incorporate behavioral
information but overlook a critical distinction: unlike relatively uniform
content features, user-item interactions are highly skewed and diverse,
creating a vast information gap in quality and quantity between popular and
long-tail items. This oversight leads to two critical limitations: (1) Noise
Corruption: Indiscriminate behavior-content alignment allows collaborative
noise from long-tail items to corrupt their content representations, leading to
the loss of critical multimodal information. (2)Signal Obscurity: The
equal-weighting scheme for SIDs fails to reflect the varying importance of
different behavioral signals, making it difficult for downstream tasks to
distinguish important SIDs from uninformative ones. To tackle these issues, we
propose a mixture-of-quantization framework, MMQ-v2, to adaptively Align,
Denoise, and Amplify multimodal information from content and behavior
modalities for semantic IDs learning. The semantic IDs generated by this
framework named ADA-SID. It introduces two innovations: an adaptive
behavior-content alignment that is aware of information richness to shield
representations from noise, and a dynamic behavioral router to amplify critical
signals by applying different weights to SIDs. Extensive experiments on public
and large-scale industrial datasets demonstrate ADA-SID's significant
superiority in both generative and discriminative recommendation tasks.

### 4. [Continual Low-Rank Adapters for LLM-based Generative Recommender Systems](http://arxiv.org/pdf/2510.25093v1)

Authors: Hyunsik Yoo, Ting-Wei Li, SeongKu Kang, Zhining Liu, Charlie Xu, Qilin Qi, Hanghang Tong

While large language models (LLMs) achieve strong performance in
recommendation, they face challenges in continual learning as users, items, and
user preferences evolve over time. Existing LoRA-based continual methods
primarily focus on preserving performance on previous tasks, but this overlooks
the unique nature of recommendation: the goal is not to predict past
preferences, and outdated preferences can even harm performance when current
interests shift significantly. To address this, we propose PESO (Proximally
rEgularized Single evolving lOra, a continual adaptation method for LoRA in
recommendation. PESO introduces a proximal regularizer that anchors the current
adapter to its most recent frozen state, enabling the model to flexibly balance
adaptation and preservation, and to better capture recent user behaviors.
Theoretically, we show that this proximal design provides data-aware,
direction-wise guidance in the LoRA subspace. Empirically, PESO consistently
outperforms existing LoRA-based continual learning methods.

### 5. [Measuring the Research Output and Performance of the University of Ibadan from 2014 to 2023: A Scientometric Analysis](http://arxiv.org/pdf/2510.25283v1)

Authors: Muneer Ahmad, Undie Felicia Nkatv

This study employs scientometric methods to assess the research output and
performance of the University of Ibadan from 2014 to 2023. By analyzing
publication trends, citation patterns, and collaboration networks, the research
aims to comprehensively evaluate the university's research productivity,
impact, and disciplinary focus. This article's endeavors are characterized by
innovation, interdisciplinary collaboration, and commitment to excellence,
making the University of Ibadan a significant hub for cutting-edge research in
Nigeria and beyond. The goal of the current study is to ascertain the influence
of the university's research output and publication patterns between 2014 and
2023. The study focuses on the departments at the University of Ibadan that
contribute the most, the best journals for publishing, the nations that
collaborate, the impact of citations both locally and globally, well-known
authors and their total production, and the research output broken down by
year. According to the university's ten-year publication data, 7159 papers with
an h-index of 75 were published between 2014 and 2023, garnering 218572
citations. Furthermore, the VOSviewer software mapping approach is used to
illustrate the stenographical mapping of data through graphs. The findings of
this study will contribute to understanding the university's research
strengths, weaknesses, and potential areas for improvement. Additionally, the
results will inform evidence-based decision-making for enhancing research
strategies and policies at the University of Ibadan.

### 6. [Towards Automated Quality Assurance of Patent Specifications: A Multi-Dimensional LLM Framework](http://arxiv.org/pdf/2510.25402v1)

Authors: Yuqian Chai, Chaochao Wang, Weilei Wang

Despite the surge in patent applications and emergence of AI drafting tools,
systematic evaluation of patent content quality has received limited research
attention. To address this gap, We propose to evaluate patents using regulatory
compliance, technical coherence, and figure-reference consistency detection
modules, and then generate improvement suggestions via an integration module.
The framework is validated on a comprehensive dataset comprising 80
human-authored and 80 AI-generated patents from two patent drafting tools.
Experimental results show balanced accuracies of 99.74\%, 82.12\%, and 91.2\%
respectively across the three detection modules when validated against expert
annotations. Additional analysis was conducted to examine defect distributions
across patent sections, technical domains, and authoring sources. Section-based
analysis indicates that figure-text consistency and technical detail precision
require particular attention. Mechanical Engineering and Construction show more
claim-specification inconsistencies due to complex technical documentation
requirements. AI-generated patents show a significant gap compared to
human-authored ones. While human-authored patents primarily contain
surface-level errors like typos, AI-generated patents exhibit more structural
defects in figure-text alignment and cross-references.

### 7. [Alibaba International E-commerce Product Search Competition DcuRAGONs Team Technical Report](http://arxiv.org/pdf/2510.25428v1)

Authors: Thang-Long Nguyen-Ho, Minh-Khoi Pham, Hoang-Bao Le

This report details our methodology and results developed for the
Multilingual E-commerce Search Competition. The problem aims to recognize
relevance between user queries versus product items in a multilingual context
and improve recommendation performance on e-commerce platforms. Utilizing Large
Language Models (LLMs) and their capabilities in other tasks, our data-centric
method achieved the highest score compared to other solutions during the
competition. Final leaderboard is publised at
https://alibaba-international-cikm2025.github.io. The source code for our
project is published at https://github.com/nhtlongcs/e-commerce-product-search.

### 8. [Retrieval-Augmented Search for Large-Scale Map Collections with ColPali](http://arxiv.org/pdf/2510.25718v1)

Authors: Jamie Mahowald, Benjamin Charles Germain Lee

Multimodal approaches have shown great promise for searching and navigating
digital collections held by libraries, archives, and museums. In this paper, we
introduce map-RAS: a retrieval-augmented search system for historic maps. In
addition to introducing our framework, we detail our publicly-hosted demo for
searching 101,233 map images held by the Library of Congress. With our system,
users can multimodally query the map collection via ColPali, summarize search
results using Llama 3.2, and upload their own collections to perform
inter-collection search. We articulate potential use cases for archivists,
curators, and end-users, as well as future work with our system in both machine
learning and the digital humanities. Our demo can be viewed at:
http://www.mapras.com.

### 9. [Model-Document Protocol for AI Search](http://arxiv.org/pdf/2510.25160v1)

Authors: Hongjin Qian, Zheng Liu

AI search depends on linking large language models (LLMs) with vast external
knowledge sources. Yet web pages, PDF files, and other raw documents are not
inherently LLM-ready: they are long, noisy, and unstructured. Conventional
retrieval methods treat these documents as verbatim text and return raw
passages, leaving the burden of fragment assembly and contextual reasoning to
the LLM. This gap underscores the need for a new retrieval paradigm that
redefines how models interact with documents.
  We introduce the Model-Document Protocol (MDP), a general framework that
formalizes how raw text is bridged to LLMs through consumable knowledge
representations. Rather than treating retrieval as passage fetching, MDP
defines multiple pathways that transform unstructured documents into
task-specific, LLM-ready inputs. These include agentic reasoning, which curates
raw evidence into coherent context; memory grounding, which accumulates
reusable notes to enrich reasoning; and structured leveraging, which encodes
documents into formal representations such as graphs or key-value caches. All
three pathways share the same goal: ensuring that what reaches the LLM is not
raw fragments but compact, structured knowledge directly consumable for
reasoning.
  As an instantiation, we present MDP-Agent, which realizes the protocol
through an agentic process: constructing document-level gist memories for
global coverage, performing diffusion-based exploration with vertical
exploitation to uncover layered dependencies, and applying map-reduce style
synthesis to integrate large-scale evidence into compact yet sufficient
context. Experiments on information-seeking benchmarks demonstrate that
MDP-Agent outperforms baselines, validating both the soundness of the MDP
framework and the effectiveness of its agentic instantiation.

### 10. [GReF: A Unified Generative Framework for Efficient Reranking via Ordered Multi-token Prediction](http://arxiv.org/pdf/2510.25220v1)

Authors: Zhijie Lin, Zhuofeng Li, Chenglei Dai, Wentian Bao, Shuai Lin, Enyun Yu, Haoxiang Zhang, Liang Zhao

In a multi-stage recommendation system, reranking plays a crucial role in
modeling intra-list correlations among items. A key challenge lies in exploring
optimal sequences within the combinatorial space of permutations. Recent
research follows a two-stage (generator-evaluator) paradigm, where a generator
produces multiple feasible sequences, and an evaluator selects the best one. In
practice, the generator is typically implemented as an autoregressive model.
However, these two-stage methods face two main challenges. First, the
separation of the generator and evaluator hinders end-to-end training. Second,
autoregressive generators suffer from inference efficiency. In this work, we
propose a Unified Generative Efficient Reranking Framework (GReF) to address
the two primary challenges. Specifically, we introduce Gen-Reranker, an
autoregressive generator featuring a bidirectional encoder and a dynamic
autoregressive decoder to generate causal reranking sequences. Subsequently, we
pre-train Gen-Reranker on the item exposure order for high-quality parameter
initialization. To eliminate the need for the evaluator while integrating
sequence-level evaluation during training for end-to-end optimization, we
propose post-training the model through Rerank-DPO. Moreover, for efficient
autoregressive inference, we introduce ordered multi-token prediction (OMTP),
which trains Gen-Reranker to simultaneously generate multiple future items
while preserving their order, ensuring practical deployment in real-time
recommender systems. Extensive offline experiments demonstrate that GReF
outperforms state-of-the-art reranking methods while achieving latency that is
nearly comparable to non-autoregressive models. Additionally, GReF has also
been deployed in a real-world video app Kuaishou with over 300 million daily
active users, significantly improving online recommendation quality.

### Machine Learning

### 1. [Training Across Reservoirs: Using Numerical Differentiation To Couple Trainable Networks With Black-Box Reservoirs](http://arxiv.org/pdf/2510.25074v1)

Authors: Andrew Clark, Jack Moursounidis, Osmaan Rasouli, William Gan, Cooper Doyle, Anna Leontjeva

We introduce Bounded Numerical Differentiation (BOND), a perturbative method
for estimating partial derivatives across network structures with inaccessible
computational graphs. BOND demonstrates improved accuracy and scalability from
existing perturbative methods, enabling new explorations of trainable
architectures that integrate black-box functions. We observe that these
black-box functions, realized in our experiments as fixed, untrained networks,
can enhance model performance without increasing the number of trainable
parameters. This improvement is achieved without extensive optimization of the
architecture or properties of the black-box function itself. Our findings
highlight the potential of leveraging fixed, non-trainable modules to expand
model capacity, suggesting a path toward combining analogue and digital devices
as a mechanism for scaling networks.

### 2. [Selective Learning for Deep Time Series Forecasting](http://arxiv.org/pdf/2510.25207v1)

Authors: Yisong Fu, Zezhi Shao, Chengqing Yu, Yujie Li, Zhulin An, Qi Wang, Yongjun Xu, Fei Wang

Benefiting from high capacity for capturing complex temporal patterns, deep
learning (DL) has significantly advanced time series forecasting (TSF).
However, deep models tend to suffer from severe overfitting due to the inherent
vulnerability of time series to noise and anomalies. The prevailing DL paradigm
uniformly optimizes all timesteps through the MSE loss and learns those
uncertain and anomalous timesteps without difference, ultimately resulting in
overfitting. To address this, we propose a novel selective learning strategy
for deep TSF. Specifically, selective learning screens a subset of the whole
timesteps to calculate the MSE loss in optimization, guiding the model to focus
on generalizable timesteps while disregarding non-generalizable ones. Our
framework introduces a dual-mask mechanism to target timesteps: (1) an
uncertainty mask leveraging residual entropy to filter uncertain timesteps, and
(2) an anomaly mask employing residual lower bound estimation to exclude
anomalous timesteps. Extensive experiments across eight real-world datasets
demonstrate that selective learning can significantly improve the predictive
performance for typical state-of-the-art deep models, including 37.4% MSE
reduction for Informer, 8.4% for TimesNet, and 6.5% for iTransformer.

### 3. [BSFA: Leveraging the Subspace Dichotomy to Accelerate Neural Network Training](http://arxiv.org/pdf/2510.25244v1)

Authors: Wenjie Zhou, Bohan Wang, Wei Chen, Xueqi Cheng

Recent studies \citep{gur2018gradient,song2024does, wen2024understanding}
highlight a fundamental dichotomy in deep learning optimization: Although
parameter updates along the top eigendirections of the loss Hessian (Dom-space)
capture most of the update magnitude, they often contribute minimally to loss
reduction. In contrast, updates in the orthogonal component (Bulk-space) have
smaller magnitudes but drive most learning progress. In this work, we further
advance the understanding of this phenomenon and introduce the
\textbf{Bulk-Space-Filtration-Accelerator (BSFA)}, a novel plug-and-play
framework. BSFA accelerates training by differentially scaling update
components projected onto these distinct subspaces, simultaneously enhancing
stability by moderating updates in the dominant subspace and boosting
convergence speed by amplifying those in the bulk-space. To ensure BSFA is both
practical and scalable for contemporary large models, we introduce two key
innovations: an efficient estimator using Principal Component Analysis (PCA) on
historical updates for fast subspace estimation, and a block-wise strategy that
applies this estimation on a per-parameter-block basis. These designs make BSFA
computationally tractable and highly effective. We demonstrate BSFA's
acceleration across various tasks, notably achieving approximately 2$\times$
speedup when pre-training LLaMA-72M on WikiText-103 and LLaMA-134M on
OpenWebText compared to vanilla AdamW.

### 4. [On the Stability of Neural Networks in Deep Learning](http://arxiv.org/pdf/2510.25282v1)

Authors: Blaise Delattre

Deep learning has achieved remarkable success across a wide range of tasks,
but its models often suffer from instability and vulnerability: small changes
to the input may drastically affect predictions, while optimization can be
hindered by sharp loss landscapes. This thesis addresses these issues through
the unifying perspective of sensitivity analysis, which examines how neural
networks respond to perturbations at both the input and parameter levels.
  We study Lipschitz networks as a principled way to constrain sensitivity to
input perturbations, thereby improving generalization, adversarial robustness,
and training stability. To complement this architectural approach, we introduce
regularization techniques based on the curvature of the loss function,
promoting smoother optimization landscapes and reducing sensitivity to
parameter variations. Randomized smoothing is also explored as a probabilistic
method for enhancing robustness at decision boundaries.
  By combining these perspectives, we develop a unified framework where
Lipschitz continuity, randomized smoothing, and curvature regularization
interact to address fundamental challenges in stability. The thesis contributes
both theoretical analysis and practical methodologies, including efficient
spectral norm computation, novel Lipschitz-constrained layers, and improved
certification procedures.

### 5. [Hierarchical Physics-Embedded Learning for Spatiotemporal Dynamical Systems](http://arxiv.org/pdf/2510.25306v1)

Authors: Xizhe Wang, Xiaobin Song, Qingshan Jia, Hongbo Zhao, Benben Jiang

Modeling complex spatiotemporal dynamics, particularly in
far-from-equilibrium systems, remains a grand challenge in science. The
governing partial differential equations (PDEs) for these systems are often
intractable to derive from first principles, due to their inherent complexity,
characterized by high-order derivatives and strong nonlinearities, coupled with
incomplete physical knowledge. This has spurred the development of data-driven
methods, yet these approaches face limitations: Purely data-driven models are
often physically inconsistent and data-intensive, while existing
physics-informed methods lack the structural capacity to represent complex
operators or systematically integrate partial physical knowledge. Here, we
propose a hierarchical physics-embedded learning framework that fundamentally
advances both the forward spatiotemporal prediction and inverse discovery of
physical laws from sparse and noisy data. The key innovation is a two-level
architecture that mirrors the process of scientific discovery: the first level
learns fundamental symbolic components of a PDE, while the second learns their
governing combinations. This hierarchical decomposition not only reduces
learning complexity but, more importantly, enables a structural integration of
prior knowledge. Known physical laws are directly embedded into the models
computational graph, guaranteeing physical consistency and improving data
efficiency. By building the framework upon adaptive Fourier Neural Operators,
we can effectively capture the non-local dependencies and high-order operators
characteristic of dynamical systems. Additionally, by structurally decoupling
known and unknown terms, the framework further enables interpretable discovery
of underlying governing equations through symbolic regression, without
presupposing functional forms.

### 6. [CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices](http://arxiv.org/pdf/2510.25323v1)

Authors: Xuchen Feng, Siyu Liao

Normalizing flows are deep generative models that enable efficient likelihood
estimation and sampling through invertible transformations. A key challenge is
to design linear layers that enhance expressiveness while maintaining efficient
computation of the Jacobian determinant and inverse. We introduce a novel
invertible linear layer based on the product of circulant and diagonal
matrices. This decomposition reduces parameter complexity from
$\mathcal{O}(n^2)$ to $\mathcal{O}(mn)$ using $m$ diagonal matrices and $m-1$
circulant matrices while still approximating general linear transformations. By
leveraging the Fast Fourier Transform, our approach reduces the time complexity
of matrix inversion from $\mathcal{O}(n^3)$ to $\mathcal{O}(mn\log n)$ and that
of computing the log-determinant from $\mathcal{O}(n^3)$ to $\mathcal{O}(mn)$,
where $n$ is the input dimension. We build upon this layer to develop
Circulant-Diagonal Flow (CDFlow), which achieves strong density estimation on
natural image datasets and effectively models data with inherent periodic
structure. Furthermore, CDFlow significantly accelerates key operations in
normalizing flows, providing practical benefits for scalable generative
modeling.

### 7. [Parameter Averaging in Link Prediction](http://arxiv.org/pdf/2510.25361v1)

Authors: Rupesh Sapkota, Caglar Demir, Arnab Sharma, Axel-Cyrille Ngonga Ngomo

Ensemble methods are widely employed to improve generalization in machine
learning. This has also prompted the adoption of ensemble learning for the
knowledge graph embedding (KGE) models in performing link prediction. Typical
approaches to this end train multiple models as part of the ensemble, and the
diverse predictions are then averaged. However, this approach has some
significant drawbacks. For instance, the computational overhead of training
multiple models increases latency and memory overhead. In contrast, model
merging approaches offer a promising alternative that does not require training
multiple models. In this work, we introduce model merging, specifically
weighted averaging, in KGE models. Herein, a running average of model
parameters from a training epoch onward is maintained and used for predictions.
To address this, we additionally propose an approach that selectively updates
the running average of the ensemble model parameters only when the
generalization performance improves on a validation dataset. We evaluate these
two different weighted averaging approaches on link prediction tasks, comparing
the state-of-the-art benchmark ensemble approach. Additionally, we evaluate the
weighted averaging approach considering literal-augmented KGE models and
multi-hop query answering tasks as well. The results demonstrate that the
proposed weighted averaging approach consistently improves performance across
diverse evaluation settings.

### 8. [Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](http://arxiv.org/pdf/2510.25480v1)

Authors: Florian A. Hölzl, Daniel Rueckert, Georgios Kaissis

Robust validation metrics remain essential in contemporary deep learning, not
only to detect overfitting and poor generalization, but also to monitor
training dynamics. In the supervised classification setting, we investigate
whether interactions between training data and model weights can yield such a
metric that both tracks generalization during training and attributes
performance to individual training samples. We introduce Gradient-Weight
Alignment (GWA), quantifying the coherence between per-sample gradients and
model weights. We show that effective learning corresponds to coherent
alignment, while misalignment indicates deteriorating generalization. GWA is
efficiently computable during training and reflects both sample-specific
contributions and dataset-wide learning dynamics. Extensive experiments show
that GWA accurately predicts optimal early stopping, enables principled model
comparisons, and identifies influential training samples, providing a
validation-set-free approach for model analysis directly from the training
data.

### 9. [Right for the Right Reasons: Avoiding Reasoning Shortcuts via Prototypical Neurosymbolic AI](http://arxiv.org/pdf/2510.25497v1)

Authors: Luca Andolfi, Eleonora Giunchiglia

Neurosymbolic AI is growing in popularity thanks to its ability to combine
neural perception and symbolic reasoning in end-to-end trainable models.
However, recent findings reveal these are prone to shortcut reasoning, i.e., to
learning unindented concepts--or neural predicates--which exploit spurious
correlations to satisfy the symbolic constraints. In this paper, we address
reasoning shortcuts at their root cause and we introduce prototypical
neurosymbolic architectures. These models are able to satisfy the symbolic
constraints (be right) because they have learnt the correct basic concepts (for
the right reasons) and not because of spurious correlations, even in extremely
low data regimes. Leveraging the theory of prototypical learning, we
demonstrate that we can effectively avoid reasoning shortcuts by training the
models to satisfy the background knowledge while taking into account the
similarity of the input with respect to the handful of labelled datapoints. We
extensively validate our approach on the recently proposed rsbench benchmark
suite in a variety of settings and tasks with very scarce supervision: we show
significant improvements in learning the right concepts both in synthetic tasks
(MNIST-EvenOdd and Kand-Logic) and real-world, high-stake ones (BDD-OIA). Our
findings pave the way to prototype grounding as an effective,
annotation-efficient strategy for safe and reliable neurosymbolic learning.

### 10. [Support Vector Machine-Based Burnout Risk Prediction with an Interactive Interface for Organizational Use](http://arxiv.org/pdf/2510.25509v1)

Authors: Bruno W. G. Teodosio, Mário J. O. T. Lira, Pedro H. M. Araújo, Lucas R. C. Farias

Burnout is a psychological syndrome marked by emotional exhaustion,
depersonalization, and reduced personal accomplishment, with a significant
impact on individual well-being and organizational performance. This study
proposes a machine learning approach to predict burnout risk using the
HackerEarth Employee Burnout Challenge dataset. Three supervised algorithms
were evaluated: nearest neighbors (KNN), random forest, and support vector
machine (SVM), with model performance evaluated through 30-fold
cross-validation using the determination coefficient (R2). Among the models
tested, SVM achieved the highest predictive performance (R2 = 0.84) and was
statistically superior to KNN and Random Forest based on paired $t$-tests. To
ensure practical applicability, an interactive interface was developed using
Streamlit, allowing non-technical users to input data and receive burnout risk
predictions. The results highlight the potential of machine learning to support
early detection of burnout and promote data-driven mental health strategies in
organizational settings.

### Neural and Evolutionary Computing

### 1. [A Benchmark Suite for Multi-Objective Optimization in Battery Thermal Management System Design](http://arxiv.org/pdf/2510.25219v1)

Authors: Kaichen Ouyang, Yezhi Xia

Synthetic Benchmark Problems (SBPs) are commonly used to evaluate the
performance of metaheuristic algorithms. However, these SBPs often contain
various unrealistic properties, potentially leading to underestimation or
overestimation of algorithmic performance. While several benchmark suites
comprising real-world problems have been proposed for various types of
metaheuristics, a notable gap exists for Constrained Multi-objective
Optimization Problems (CMOPs) derived from practical engineering applications,
particularly in the domain of Battery Thermal Management System (BTMS) design.
To address this gap, this study develops and presents a specialized benchmark
suite for multi-objective optimization in BTMS. This suite comprises a diverse
collection of real-world constrained problems, each defined via accurate
surrogate models based on recent research to efficiently represent complex
thermal-fluid interactions. The primary goal of this benchmark suite is to
provide a practical and relevant testing ground for evolutionary algorithms and
optimization methods focused on energy storage thermal management. Future work
will involve establishing comprehensive baseline results using state-of-the-art
algorithms, conducting comparative analyses, and developing a standardized
ranking scheme to facilitate robust performance assessment.

### 2. [Dynamically Weighted Momentum with Adaptive Step Sizes for Efficient Deep Network Training](http://arxiv.org/pdf/2510.25042v1)

Authors: Zhifeng Wang, Longlong Li, Chunyan Zeng

Within the current sphere of deep learning research, despite the extensive
application of optimization algorithms such as Stochastic Gradient Descent
(SGD) and Adaptive Moment Estimation (Adam), there remains a pronounced
inadequacy in their capability to address fluctuations in learning efficiency,
meet the demands of complex models, and tackle non-convex optimization issues.
These challenges primarily arise from the algorithms' limitations in handling
complex data structures and models, for instance, difficulties in selecting an
appropriate learning rate, avoiding local optima, and navigating through
high-dimensional spaces. To address these issues, this paper introduces a novel
optimization algorithm named DWMGrad. This algorithm, building on the
foundations of traditional methods, incorporates a dynamic guidance mechanism
reliant on historical data to dynamically update momentum and learning rates.
This allows the optimizer to flexibly adjust its reliance on historical
information, adapting to various training scenarios. This strategy not only
enables the optimizer to better adapt to changing environments and task
complexities but also, as validated through extensive experimentation,
demonstrates DWMGrad's ability to achieve faster convergence rates and higher
accuracies under a multitude of scenarios.

### 3. [Socio-cognitive agent-oriented evolutionary algorithm with trust-based optimization](http://arxiv.org/pdf/2510.25095v1)

Authors: Aleksandra Urbańczyk, Krzysztof Czech, Piotr Urbańczyk, Marek Kisiel-Dorohinicki, Aleksander Byrski

This paper introduces the Trust-Based Optimization (TBO), a novel extension
of the island model in evolutionary computation that replaces conventional
periodic migrations with a flexible, agent-driven interaction mechanism based
on trust or reputation. Experimental results demonstrate that TBO generally
outperforms the standard island model evolutionary algorithm across various
optimization problems. Nevertheless, algorithm performance varies depending on
the problem type, with certain configurations being more effective for specific
landscapes or dimensions. The findings suggest that trust and reputation
mechanisms provide a flexible and adaptive approach to evolutionary
optimization, improving solution quality in many cases.

### 4. [Position: Biology is the Challenge Physics-Informed ML Needs to Evolve](http://arxiv.org/pdf/2510.25368v1)

Authors: Julien Martinelli

Physics-Informed Machine Learning (PIML) has successfully integrated
mechanistic understanding into machine learning, particularly in domains
governed by well-known physical laws. This success has motivated efforts to
apply PIML to biology, a field rich in dynamical systems but shaped by
different constraints. Biological modeling, however, presents unique
challenges: multi-faceted and uncertain prior knowledge, heterogeneous and
noisy data, partial observability, and complex, high-dimensional networks. In
this position paper, we argue that these challenges should not be seen as
obstacles to PIML, but as catalysts for its evolution. We propose
Biology-Informed Machine Learning (BIML): a principled extension of PIML that
retains its structural grounding while adapting to the practical realities of
biology. Rather than replacing PIML, BIML retools its methods to operate under
softer, probabilistic forms of prior knowledge. We outline four foundational
pillars as a roadmap for this transition: uncertainty quantification,
contextualization, constrained latent structure inference, and scalability.
Foundation Models and Large Language Models will be key enablers, bridging
human expertise with computational modeling. We conclude with concrete
recommendations to build the BIML ecosystem and channel PIML-inspired
innovation toward challenges of high scientific and societal relevance.

### Networking and Internet Architecture

### 1. [Learning-Based vs Human-Derived Congestion Control: An In-Depth Experimental Study](http://arxiv.org/pdf/2510.25105v1)

Authors: Mihai Mazilu, Luca Giacomoni, George Parisis

Learning-based congestion control (CC), including Reinforcement-Learning,
promises efficient CC in a fast-changing networking landscape, where evolving
communication technologies, applications and traffic workloads pose severe
challenges to human-derived, static CC algorithms. Learning-based CC is in its
early days and substantial research is required to understand existing
limitations, identify research challenges and, eventually, yield deployable
solutions for real-world networks. In this paper, we extend our prior work and
present a reproducible and systematic study of learning-based CC with the aim
to highlight strengths and uncover fundamental limitations of the
state-of-the-art. We directly contrast said approaches with widely deployed,
human-derived CC algorithms, namely TCP Cubic and BBR (version 3). We identify
challenges in evaluating learning-based CC, establish a methodology for
studying said approaches and perform large-scale experimentation with
learning-based CC approaches that are publicly available. We show that
embedding fairness directly into reward functions is effective; however, the
fairness properties do not generalise into unseen conditions. We then show that
RL learning-based approaches existing approaches can acquire all available
bandwidth while largely maintaining low latency. Finally, we highlight that
existing the latest learning-based CC approaches under-perform when the
available bandwidth and end-to-end latency dynamically change while remaining
resistant to non-congestive loss. As with our initial study, our
experimentation codebase and datasets are publicly available with the aim to
galvanise the research community towards transparency and reproducibility,
which have been recognised as crucial for researching and evaluating
machine-generated policies.

### 2. [ML-Based Preamble Collision Detection in the Random Access Procedure of Cellular IoT Networks](http://arxiv.org/pdf/2510.25145v1)

Authors: Giancarlo Maldonado Cardenas, Diana C. Gonzalez, Judy C. Guevara, Carlos A. Astudillo, Nelson L. S. da Fonseca

Preamble collision in the random access channel (RACH) is a major bottleneck
in massive machine-type communication (mMTC) scenarios, typical of cellular IoT
(CIoT) deployments. This work proposes a machine learning-based mechanism for
early collision detection during the random access (RA) procedure. A labeled
dataset was generated using the RA procedure messages exchanged between the
users and the base station under realistic channel conditions, simulated in
MATLAB. We evaluate nine classic classifiers -- including tree ensembles,
support vector machines, and neural networks -- across four communication
scenarios, varying both channel characteristics (e.g., Doppler spread,
multipath) and the cell coverage radius, to emulate realistic propagation,
mobility, and spatial conditions. The neural network outperformed all other
models, achieving over 98\% balanced accuracy in the in-distribution evaluation
(train and test drawn from the same dataset) and sustaining 95\% under
out-of-distribution evaluation (train/test from different datasets). To enable
deployment on typical base station hardware, we apply post-training
quantization. Full integer quantization reduced inference time from 2500 ms to
as low as 0.3 ms with negligible accuracy loss. The proposed solution combines
high detection accuracy with low-latency inference, making it suitable for
scalable, real-time CIoT applications found in real networks.

### 3. [Adaptive Design of mmWave Initial Access Codebooks using Reinforcement Learning](http://arxiv.org/pdf/2510.25271v1)

Authors: Sabrine Aroua, Christos Anastasios Bovolis, Bo Göransson, Anastasios Giovanidis, Mathieu Leconte, Apostolos Destounis

Initial access (IA) is the process by which user equipment (UE) establishes
its first connection with a base station. In 5G systems, particularly at
millimeter-wave frequencies, IA integrates beam management to support highly
directional transmissions. The base station employs a codebook of beams for the
transmission of Synchronization Signal Blocks (SSBs), which are periodically
swept to detect and connect users. The design of this SSB codebook is critical
for ensuring reliable, wide-area coverage. In current networks, SSB codebooks
are meticulously engineered by domain experts. While these expert-defined
codebooks provide a robust baseline, they lack flexibility in dynamic or
heterogeneous environments where user distributions vary, limiting their
overall effectiveness. This paper proposes a hybrid Reinforcement Learning (RL)
framework for adaptive SSB codebook design. Building on top of expert
knowledge, the RL agent leverages a pool of expert-designed SSB beams and
learns to adaptively select or combine them based on real-time feedback. This
enables the agent to dynamically tailor codebooks to the actual environment,
without requiring explicit user location information, while always respecting
practical beam constraints. Simulation results demonstrate that, on average,
the proposed approach improves user connectivity by 10.8$\%$ compared to static
expert configurations. These findings highlight the potential of combining
expert knowledge with data-driven optimization to achieve more intelligent,
flexible, and resilient beam management in next-generation wireless networks.

### 4. [TCP ROCCET: An RTT-Oriented CUBIC Congestion Control Extension for 5G and Beyond Networks](http://arxiv.org/pdf/2510.25281v1)

Authors: Lukas Prause, Mark Akselrod

The behavior of loss-based TCP congestion control algorithms like TCP CUBIC
continues to be a challenge in modern cellular networks. Due to the large RLC
layer buffers required to deal with short-term changes in channel capacity, the
behavior of both the Slow Start and congestion avoidance phases may be heavily
impacted by the lack of packet losses and the resulting bufferbloat. While
existing congestion control algorithms like TCP BBR do tend to perform better
even in the presence of large bottleneck buffers, they still tend to fill the
buffer more than necessary and can have fairness issues when compared to
loss-based algorithms.
  In this paper, we analyze the issues with the use of loss-based congestion
control algorithms by analyzing TCP CUBIC, which is currently the most popular
variant. To mitigate the issues experienced by TCP CUBIC in cellular networks,
we introduce TCP ROCCET, a latency-based extension of TCP CUBIC that responds
to network congestion based on round-trip time in addition to packet loss.
  Our findings show that TCP ROCCET can reduce latency and bufferbloat compared
to the standard CUBIC implementation, without requiring a specific network
architecture. Compared to TCP BBRv3, ROCCET offers similar throughput while
maintaining lower overall latency. The evaluation was conducted in real 5G
networks, including both stationary and mobile scenarios, confirming ROCCET's
improved response to network congestion under varying conditions.

### 5. [Evaluating Learning Congestion control Schemes for LEO Constellations](http://arxiv.org/pdf/2510.25498v1)

Authors: Mihai Mazilu, Aiden Valentine, George Parisis

Low Earth Orbit (LEO) satellite networks introduce unique congestion control
(CC) challenges due to frequent handovers, rapidly changing round-trip times
(RTTs), and non-congestive loss. This paper presents the first comprehensive,
emulation-driven evaluation of CC schemes in LEO networks, combining realistic
orbital dynamics via the LeoEM framework with targeted Mininet
micro-benchmarks. We evaluated representative CC algorithms from three classes,
loss-based (Cubic, SaTCP), model-based (BBRv3), and learning-based (Vivace,
Sage, Astraea), across diverse single-flow and multi-flow scenarios, including
interactions with active queue management (AQM). Our findings reveal that: (1)
handover-aware loss-based schemes can reclaim bandwidth but at the cost of
increased latency; (2) BBRv3 sustains high throughput with modest delay
penalties, yet reacts slowly to abrupt RTT changes; (3) RL-based schemes
severely underperform under dynamic conditions, despite being notably resistant
to non-congestive loss; (4) fairness degrades significantly with RTT asymmetry
and multiple bottlenecks, especially in human-designed CC schemes; and (5) AQM
at bottlenecks can restore fairness and boost efficiency. These results expose
critical limitations in current CC schemes and provide insight for designing
LEO-specific data transport protocols.

### 6. [Device to Device Pairs Sharding based on Distance](http://arxiv.org/pdf/2510.25552v1)

Authors: K Prajwal, Tharun K, Navaneeth P, Ishwar Mandal, Kiran M

In the conventional cellular system, devices are not allowed to communicate
directly with each other in the licensed cellular bandwidth and all
communications take place through the base stations. The users requirements has
led the technology to become fast and faster. Multimedia rich data exchange,
fast service, high quality voice calls, newer and more demanding applications,
information at fingertips, everything requires technology and communication
between devices. A constant need to increase network capacity for meeting the
users growing demands has led to the growth of cellular communication networks
from the first generation(1G) to the fifth generation(5G). There will be crores
of connected devices in the coming future . A large number of connections are
going to be heterogeneous, demanding lesser delays, higher data rates, superior
throughput and enhanced system capacity. The available spectrum resources are
limited and has to be flexibly used by mobile network operators to cope with
the rising demands. An emerging facilitator of the upcoming high data rate
demanding next-generation networks are device-to-device(D2D) communication.
This paper has developed a model that establishes Device-to-Device (D2D)
communication between two end-users without involving the eNB (evolved Node B).
We have sharded the UEs and CUs based on the criteria of DISTANCE. To do so, we
used the K-means clustering method.

### 7. [MetaLore: Learning to Orchestrate Communication and Computation for Metaverse Synchronization](http://arxiv.org/pdf/2510.25705v1)

Authors: Elif Ebru Ohri, Qi Liao, Anastasios Giovanidis, Francesca Fossati, Nour-El-Houda Yellas

As augmented and virtual reality evolve, achieving seamless synchronization
between physical and digital realms remains a critical challenge, especially
for real-time applications where delays affect the user experience. This paper
presents MetaLore, a Deep Reinforcement Learning (DRL) based framework for
joint communication and computational resource allocation in Metaverse or
digital twin environments. MetaLore dynamically shares the communication
bandwidth and computational resources among sensors and mobile devices to
optimize synchronization, while offering high throughput performance. Special
treatment is given in satisfying end-to-end delay guarantees. A key
contribution is the introduction of two novel Age of Information (AoI) metrics:
Age of Request Information (AoRI) and Age of Sensor Information (AoSI),
integrated into the reward function to enhance synchronization quality. An open
source simulator has been extended to incorporate and evaluate the approach.
The DRL solution is shown to achieve the performance of full-enumeration
brute-force solutions by making use of a small, task-oriented observation space
of two queue lengths at the network side. This allows the DRL approach the
flexibility to effectively and autonomously adapt to dynamic traffic
conditions.

### 8. [Performance Evaluation of Multimedia Traffic in Cloud Storage Services over Wi-Fi and LTE Networks](http://arxiv.org/pdf/2510.25079v1)

Authors: Albert Espinal, V. Sanchez Padilla, Yesenia Cevallos

The performance of Dropbox, Google Drive, and OneDrive cloud storage services
was evaluated under Wi-Fi and LTE network conditions during multimedia file
uploads. Traffic was captured using Wireshark, and key metrics (including
delay, jitter, bandwidth, and packet loss) were analyzed. Google Drive
maintained the most consistent performance across both types of networks,
showing low latency and reduced jitter. Dropbox showed efficient bandwidth
utilization, but experienced a longer delay over LTE, attributed to a greater
number of intermediate hops. OneDrive presented variable behavior, with
elevated packet rates and increased sensitivity to fluctuations in the mobile
network. A bimodal distribution of packet sizes was observed and modeled using
a dual Poisson function. In general, Wi-Fi connections provided greater
stability for multimedia transfers, while LTE performance varied depending on
platform-specific implementations. The results contribute to a better
understanding of traffic behavior in cloud-based storage applications and
suggest further analysis with larger datasets and heterogeneous access
networks.

### 9. [Is Protective DNS Blocking the Wild West?](http://arxiv.org/pdf/2510.25352v1)

Authors: David Plonka, Branden Palacio, Debbie Perouli

We perform a passive measurement study investigating how a Protective DNS
service might perform in a Research & Education Network serving hundreds of
member institutions. Utilizing freely-available DNS blocklists consisting of
domain names deemed to be threats, we test hundreds of millions of users' real
DNS queries, observed over a week's time, to find which answers would be
blocked because they involve domain names that are potential threats. We find
the blocklists disorderly regarding their names, goals, transparency, and
provenance making them quite difficult to compare. Consequently, these
Protective DNS underpinnings lack organized oversight, presenting challenges
and risks in operation at scale.

### 10. [Energy consumption assessment of a Virtual Reality Remote Rendering application over 5G networks](http://arxiv.org/pdf/2510.25357v1)

Authors: Roberto Viola, Mikel Irazola, José Ramón Juárez, Minh Nguyen, Alexander Zoubarev, Alexander Futasz, Louay Bassbouss, Amr A. AbdelNabi, Javier Fernández Hidalgo

This paper investigates the energy implications of remote rendering for
Virtual Reality (VR) applications within a real 5G testbed. Remote rendering
enables lightweight devices to access high-performance graphical content by
offloading computationally intensive tasks to Cloud-native Network Functions
(CNFs) running on remote servers. However, this approach raises concerns
regarding energy consumption across the various network components involved,
including the remote computing node, the 5G Core, the Radio Access Network
(RAN), and the User Equipment (UE). This work proposes and evaluates two
complementary energy monitoring solutions, one hardware-based and one
software-based, to measure energy consumption at different system levels. A VR
remote renderer, deployed as CNF and leveraging the Media over QUIC (MoQ)
protocol, is used as test case for assessing its energy footprint under
different multimedia and network configurations. The results provide critical
insights into the trade-off between energy consumption and performance of a
real-world VR application running in a 5G environment.

### Robotics

### 1. [Non-Invasive Calibration Of A Stewart Platform By Photogrammetry](http://arxiv.org/pdf/2510.25072v1)

Authors: Sourabh Karmakar, Cameron J. Turner

Accurate calibration of a Stewart platform is important for their precise and
efficient operation. However, the calibration of these platforms using forward
kinematics is a challenge for researchers because forward kinematics normally
generates multiple feasible and unfeasible solutions for any pose of the moving
platform. The complex kinematic relations among the six actuator paths
connecting the fixed base to the moving platform further compound the
difficulty in establishing a straightforward and efficient calibration method.
The authors developed a new forward kinematics-based calibration method using
Denavit-Hartenberg convention and used the Stewart platform Tiger 66.1
developed in their lab for experimenting with the photogrammetry-based
calibration strategies described in this paper. This system became operational
upon completion of construction, marking its inaugural use. The authors used
their calibration model for estimating the errors in the system and adopted
three compensation options or strategies as per Least Square method to improve
the accuracy of the system. These strategies leveraged a high-resolution
digital camera and off-the-shelf software to capture the poses of the moving
platform's center. This process is non-invasive and does not need any
additional equipment to be attached to the hexapod or any alteration of the
hexapod hardware. This photogrammetry-based calibration process involves
multiple high-resolution images from different angles to measure the position
and orientation of the platform center in the three-dimensional space. The
Target poses and Actual poses are then compared, and the error compensations
are estimated using the Least-Squared methods to calculate the Predicted poses.
Results from each of the three compensation approaches demonstrated noticeable
enhancements in platform pose accuracies, suggesting room for further
improvements.

### 2. [Mean-Shift Theory and Its Applications in Swarm Robotics: A New Way to Enhance the Efficiency of Multi-Robot Collaboration](http://arxiv.org/pdf/2510.25086v1)

Authors: Guibin Sun, Jinhu Lü, Kexin Liu, Zhenqian Wang, Guanrong Chen

Swarms evolving from collective behaviors among multiple individuals are
commonly seen in nature, which enables biological systems to exhibit more
efficient and robust collaboration. Creating similar swarm intelligence in
engineered robots poses challenges to the design of collaborative algorithms
that can be programmed at large scales. The assignment-based method has played
an eminent role for a very long time in solving collaboration problems of robot
swarms. However, it faces fundamental limitations in terms of efficiency and
robustness due to its unscalability to swarm variants. This article presents a
tutorial review on recent advances in assignment-free collaboration of robot
swarms, focusing on the problem of shape formation. A key theoretical component
is the recently developed \emph{mean-shift exploration} strategy, which
improves the collaboration efficiency of large-scale swarms by dozens of times.
Further, the efficiency improvement is more significant as the swarm scale
increases. Finally, this article discusses three important applications of the
mean-shift exploration strategy, including precise shape formation, area
coverage formation, and maneuvering formation, as well as their corresponding
industrial scenarios in smart warehousing, area exploration, and cargo
transportation.

### 3. [NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies](http://arxiv.org/pdf/2510.25122v1)

Authors: Jiahong Chen, Jing Wang, Long Chen, Chuwei Cai, Jinghui Lu

Vision-language-action (VLA) models have significantly advanced robotic
manipulation by integrating vision-language models (VLMs), and action decoders
into a unified architecture. However, their deployment on resource-constrained
edge devices, such as mobile robots or embedded systems (e.g., Jetson Orin
Nano), remains challenging due to high computational demands, especially in
real-world scenarios where power, latency, and computational resources are
critical. To close this gap, we introduce Nano-scale Vision-Language Action
(NanoVLA), a family of lightweight VLA architectures that achieve high
performance with minimal resources. Our core innovations include: (1)
vision-language decoupling that moves conventional early vision and language
inputs fusion in VLM to late stage, achieving better performance while enabling
caching and reduce inference overhead and latency; (2) long-short action
chunking to ensure smooth, coherent multi-step planning without sacrificing
real-time responsiveness; (3) dynamic routing that adaptively assigns
lightweight or heavy backbones based on task complexity, further optimizing
inference efficiency. Experimental results on several benchmarks, as well as
real-world deployments, demonstrate that NanoVLA achieves up to 52x faster
inference on edge devices compared to previous state-of-the-art VLA models,
with 98% less parameters while maintaining or surpassing their task accuracy
and generalization. Ablation studies confirm that our decoupling strategy
preserves cross-task transferability, and the routing module enhances
cost-performance trade-offs, enabling practical, high-precision robotic
manipulation on resource-constrained hardware.

### 4. [Learning Spatial-Aware Manipulation Ordering](http://arxiv.org/pdf/2510.25138v1)

Authors: Yuxiang Yan, Zhiyuan Zhou, Xin Gao, Guanghao Li, Shenglin Li, Jiaqi Chen, Qunyan Pu, Jian Pu

Manipulation in cluttered environments is challenging due to spatial
dependencies among objects, where an improper manipulation order can cause
collisions or blocked access. Existing approaches often overlook these spatial
relationships, limiting their flexibility and scalability. To address these
limitations, we propose OrderMind, a unified spatial-aware manipulation
ordering framework that directly learns object manipulation priorities based on
spatial context. Our architecture integrates a spatial context encoder with a
temporal priority structuring module. We construct a spatial graph using
k-Nearest Neighbors to aggregate geometric information from the local layout
and encode both object-object and object-manipulator interactions to support
accurate manipulation ordering in real-time. To generate physically and
semantically plausible supervision signals, we introduce a spatial prior
labeling method that guides a vision-language model to produce reasonable
manipulation orders for distillation. We evaluate OrderMind on our Manipulation
Ordering Benchmark, comprising 163,222 samples of varying difficulty. Extensive
experiments in both simulation and real-world environments demonstrate that our
method significantly outperforms prior approaches in effectiveness and
efficiency, enabling robust manipulation in cluttered scenes.

### 5. [SoraNav: Adaptive UAV Task-Centric Navigation via Zeroshot VLM Reasoning](http://arxiv.org/pdf/2510.25191v1)

Authors: Hongyu Song, Rishabh Dev Yadav, Cheng Guo, Wei Pan

Interpreting visual observations and natural language instructions for
complex task execution remains a key challenge in robotics and AI. Despite
recent advances, language-driven navigation is still difficult, particularly
for UAVs in small-scale 3D environments. Existing Vision-Language Navigation
(VLN) approaches are mostly designed for ground robots and struggle to
generalize to aerial tasks that require full 3D spatial reasoning. The
emergence of large Vision-Language Models (VLMs), such as GPT and Claude,
enables zero-shot semantic reasoning from visual and textual inputs. However,
these models lack spatial grounding and are not directly applicable to
navigation. To address these limitations, SoraNav is introduced, an adaptive
UAV navigation framework that integrates zero-shot VLM reasoning with
geometry-aware decision-making. Geometric priors are incorporated into image
annotations to constrain the VLM action space and improve decision quality. A
hybrid switching strategy leverages navigation history to alternate between VLM
reasoning and geometry-based exploration, mitigating dead-ends and redundant
revisits. A PX4-based hardware-software platform, comprising both a digital
twin and a physical micro-UAV, enables reproducible evaluation. Experimental
results show that in 2.5D scenarios, our method improves Success Rate (SR) by
25.7% and Success weighted by Path Length (SPL) by 17%. In 3D scenarios, it
improves SR by 29.5% and SPL by 18.5% relative to the baseline.

### 6. [RoadSens-4M: A Multimodal Smartphone & Camera Dataset for Holistic Road-way Analysis](http://arxiv.org/pdf/2510.25211v1)

Authors: Amith Khandakar, David Michelson, Shaikh Golam Rabbani, Fariya Bintay Shafi, Md. Faysal Ahamed, Khondokar Radwanur Rahman, Md Abidur Rahman, Md. Fahmidun Nabi, Mohamed Arselene Ayari, Khaled Khan, Ponnuthurai Nagaratnam Suganthan

It's important to monitor road issues such as bumps and potholes to enhance
safety and improve road conditions. Smartphones are equipped with various
built-in sensors that offer a cost-effective and straightforward way to assess
road quality. However, progress in this area has been slow due to the lack of
high-quality, standardized datasets. This paper discusses a new dataset created
by a mobile app that collects sensor data from devices like GPS,
accelerometers, gyroscopes, magnetometers, gravity sensors, and orientation
sensors. This dataset is one of the few that integrates Geographic Information
System (GIS) data with weather information and video footage of road
conditions, providing a comprehensive understanding of road issues with
geographic context. The dataset allows for a clearer analysis of road
conditions by compiling essential data, including vehicle speed, acceleration,
rotation rates, and magnetic field intensity, along with the visual and spatial
context provided by GIS, weather, and video data. Its goal is to provide
funding for initiatives that enhance traffic management, infrastructure
development, road safety, and urban planning. Additionally, the dataset will be
publicly accessible to promote further research and innovation in smart
transportation systems.

### 7. [Hybrid Vision Servoing with Depp Alignment and GRU-Based Occlusion Recovery](http://arxiv.org/pdf/2510.25233v1)

Authors: Jee Won Lee, Hansol Lim, Sooyeun Yang, Jongseong Brad Choi

Vision-based control systems, such as image-based visual servoing (IBVS),
have been extensively explored for precise robot manipulation. A persistent
challenge, however, is maintaining robust target tracking under partial or full
occlusions. Classical methods like Lucas-Kanade (LK) offer lightweight tracking
but are fragile to occlusion and drift, while deep learning-based approaches
often require continuous visibility and intensive computation. To address these
gaps, we propose a hybrid visual tracking framework that bridges advanced
perception with real-time servo control. First, a fast global template matcher
constrains the pose search region; next, a deep-feature Lucas-Kanade module
operating on early VGG layers refines alignment to sub-pixel accuracy (<2px);
then, a lightweight residual regressor corrects local misalignments caused by
texture degradation or partial occlusion. When visual confidence falls below a
threshold, a GRU-based predictor seamlessly extrapolates pose updates from
recent motion history. Crucially, the pipeline's final outputs-translation,
rotation, and scale deltas-are packaged as direct control signals for 30Hz
image-based servo loops. Evaluated on handheld video sequences with up to 90%
occlusion, our system sustains under 2px tracking error, demonstrating the
robustness and low-latency precision essential for reliable real-world robot
vision applications.

### 8. [Time-Optimal Transport of Loosely Placed Liquid Filled Cups along Prescribed Paths](http://arxiv.org/pdf/2510.25255v1)

Authors: Klaus Zauner, Hubert Gattringer, Andreas Mueller

Handling loosely placed objects with robotic manipulators is a difficult task
from the point of view of trajectory planning and control. This becomes even
more challenging when the object to be handled is a container filled with
liquid. This paper addresses the task of transporting a liquid-filled cup
placed on a tray along a prescribed path in shortest time. The objective is to
minimize swapping, thus avoiding spillage of the fluid. To this end, the
sloshing dynamics is incorporated into the dynamic model used within the
optimal control problem formulation. The optimization problem is solved using a
direct multiple shooting approach.

### 9. [Development of Implicit-Explicit Control Based Amphibious Centipede-Type Robot and Evaluation of its Mobile Performance](http://arxiv.org/pdf/2510.25280v1)

Authors: Yusuke Tsunoda, Seiya Yamamoto, Kazuki Ito, Runze Xiao, Keisuke Naniwa, Koichi Osuka

Multi-legged mobile robots possess high mobility performance in rough terrain
environments, stemming from their high postural stability, joint flexibility,
and the redundancy provided by multiple legs. In prior research on navigating
between different environments such as land and water, the primary strategy
employed involves switching to a controller that generates an appropriate gait
for the new environment upon entering it. However, designing appropriate gaits
for each complex and diverse environment and accurately determining controller
switching for each environment is challenging. Therefore, this research
develops a centipede-type mobile robot that navigates both aquatic and
terrestrial environments with a simple, unified control scheme, based on the
implicit-explicit control philosophy and by ingeniously designing the robot's
body structure. In this research, we developed the robot featuring flexible
joints and left and right legs on each body segment and focused on the leg
structure which has extensive contact with the environment. This paper
evaluates the locomotion performance on land and water using the three
developed leg structures, using the robot's leg slip rate and actuator energy
consumption as evaluation metrics. The experimental results confirmed the
existence of an appropriate leg structure capable of navigating both aquatic
and terrestrial environments under identical control.

### 10. [An approach for combining transparency and motion assistance of a lower body exoskeleton](http://arxiv.org/pdf/2510.25335v1)

Authors: Jakob Ziegler, Bernhard Rameder, Hubert Gattringer, Andreas Mueller

In this paper, an approach for gait assistance with a lower body exoskeleton
is described. Two concepts, transparency and motion assistance, are combined.
The transparent mode, where the system is following the user's free motion with
a minimum of perceived interaction forces, is realized by exploiting the gear
backlash of the actuation units. During walking a superimposed assistance mode
applies an additional torque guiding the legs to their estimated future
position. The concept of adaptive oscillators is utilized to learn the
quasi-periodic signals typical for locomotion. First experiments showed
promising results.

### Software Engineering

### 1. [Same Same But Different: Preventing Refactoring Attacks on Software Plagiarism Detection](http://arxiv.org/pdf/2510.25057v1)

Authors: Robin Maisch, Larissa Schmid, Timur Sağlam, Nils Niehues

Plagiarism detection in programming education faces growing challenges due to
increasingly sophisticated obfuscation techniques, particularly automated
refactoring-based attacks. While code plagiarism detection systems used in
education practice are resilient against basic obfuscation, they struggle
against structural modifications that preserve program behavior, especially
caused by refactoring-based obfuscation. This paper presents a novel and
extensible framework that enhances state-of-the-art detectors by leveraging
code property graphs and graph transformations to counteract refactoring-based
obfuscation. Our comprehensive evaluation of real-world student submissions,
obfuscated using both algorithmic and AI-based obfuscation attacks,
demonstrates a significant improvement in detecting plagiarized code.

### 2. [Adaptive Proof Refinement with LLM-Guided Strategy Selection](http://arxiv.org/pdf/2510.25103v1)

Authors: Minghai Lu, Zhe Zhou, Danning Xie, Songlin Jia, Benjamin Delaware, Tianyi Zhang

Formal verification via theorem proving enables the expressive specification
and rigorous proof of software correctness, but it is difficult to scale due to
the significant manual effort and expertise required. While Large Language
Models (LLMs) show potential in proof generation, they frequently produce
incorrect proofs on the first attempt and require additional strategies for
iterative refinement. However, existing approaches employ fixed refinement
strategies and cannot dynamically choose an effective strategy based on the
particular issues in a generated proof, which limits their performance. To
overcome this limitation, we introduce Adapt, a novel proof refinement
framework that leverages an LLM-guided decision-maker to dynamically select a
suitable refinement strategy according to the state of the proof assistant and
available context of an incorrect proof. We evaluate Adapt on two benchmarks
against four existing methods and find that it significantly outperforms the
best baseline on both by proving 16.63% and 18.58% more theorems, respectively.
Furthermore, we demonstrate Adapt's generalizability by evaluating it across
five different LLMs. We also conduct ablation studies to measure the
contribution of each component and compare the trade-offs of alternative
decision-maker designs.

### 3. [Automated Program Repair Based on REST API Specifications Using Large Language Models](http://arxiv.org/pdf/2510.25148v1)

Authors: Katsuki Yamagishi, Norihiro Yoshida, Erina Makihara, Katsuro Inoue

Many cloud services provide REST API accessible to client applications.
However, developers often identify specification violations only during
testing, as error messages typically lack the detail necessary for effective
diagnosis. Consequently, debugging requires trial and error. This study
proposes dcFix, a method for detecting and automatically repairing REST API
misuses in client programs. In particular, dcFix identifies non-conforming code
fragments, integrates them with the relevant API specifications into prompts,
and leverages a Large Language Model (LLM) to produce the corrected code. Our
evaluation demonstrates that dcFix accurately detects misuse and outperforms
the baseline approach, in which prompts to the LLM omit any indication of code
fragments non conforming to REST API specifications.

### 4. [Optimizing Knowledge Utilization for Multi-Intent Comment Generation with Large Language Models](http://arxiv.org/pdf/2510.25195v1)

Authors: Shuochuan Li, Zan Wang, Xiaoning Du, Zhuo Wu, Jiuqiao Yu, Junjie Chen

Code comment generation aims to produce a generic overview of a code snippet,
helping developers understand and maintain code. However, generic summaries
alone are insufficient to meet the diverse needs of practitioners; for example,
developers expect the implementation insights to be presented in an untangled
manner, while users seek clear usage instructions. This highlights the
necessity of multi-intent comment generation. With the widespread adoption of
Large Language Models (LLMs) for code-related tasks, these models have been
leveraged to tackle the challenge of multi-intent comment generation. Despite
their successes, state-of-the-art LLM-based approaches often struggle to
construct correct relationships among intents, code, and comments within a
smaller number of demonstration examples. To mitigate this issue, we propose a
framework named KUMIC for multi-intent comment generation. Built upon
in-context learning, KUMIC leverages Chain-of-Thought (CoT) to optimize
knowledge utilization for LLMs to generate intent-specific comments.
Specifically, KUMIC first designs a retrieval mechanism to obtain similar
demonstration examples, which exhibit high code-comment consistency. Then,
KUMIC leverages CoT to guide LLMs to focus on statements facilitating the
derivation of code comments aligned with specific intents. In this context,
KUMIC constructs a mapping knowledge chain, linking code to intent-specific
statements to comments, which enables LLMs to follow similar reasoning steps
when generating the desired comments. We conduct extensive experiments to
evaluate KUMIC, and the results demonstrate that KUMIC outperforms
state-of-the-art baselines by 14.49\%, 22.41\%, 20.72\%, and 12.94\% in terms
of BLEU, METEOR, ROUGE-L, and SBERT, respectively.

### 5. [TECS/Rust-OE: Optimizing Exclusive Control in Rust-based Component Systems for Embedded Devices](http://arxiv.org/pdf/2510.25242v1)

Authors: Nao Yoshimura, Hiroshi Oyama, Takuya Azumi

The diversification of functionalities and the development of the IoT are
making embedded systems larger and more complex in structure. Ensuring system
reliability, especially in terms of security, necessitates selecting an
appropriate programming language. As part of existing research, TECS/Rust has
been proposed as a framework that combines Rust and component-based development
(CBD) to enable scalable system design and enhanced reliability. This framework
represents system structures using static mutable variables, but excessive
exclusive controls applied to ensure thread safety have led to performance
degradation. This paper proposes TECS/Rust-OE, a memory-safe CBD framework
utilizing call flows to address these limitations. The proposed Rust code
leverages real-time OS exclusive control mechanisms, optimizing performance
without compromising reusability. Rust code is automatically generated based on
component descriptions. Evaluations demonstrate reduced overhead due to
optimized exclusion control and high reusability of the generated code.

### 6. [TECS/Rust: Memory-safe Component Framework for Embedded Systems](http://arxiv.org/pdf/2510.25270v1)

Authors: Nao Yoshimura, Hiroshi Oyama, Takuya Azumi

As embedded systems grow in complexity and scale due to increased functional
diversity, component-based development (CBD) emerges as a solution to
streamline their architecture and enhance functionality reuse. CBD typically
utilizes the C programming language for its direct hardware access and
low-level operations, despite its susceptibility to memory-related issues. To
address these concerns, this paper proposes TECS/Rust, a Rust-based framework
specifically designed for TECS, which is a component framework for embedded
systems. It leverages Rust's compile-time memory-safe features, such as
lifetime and borrowing, to mitigate memory vulnerabilities common with C. The
proposed framework not only ensures memory safety but also maintains the
flexibility of CBD, automates Rust code generation for CBD components, and
supports efficient integration with real-time operating systems. An evaluation
of the amount of generated code indicates that the code generated by this paper
framework accounts for a large percentage of the actual code. Compared to code
developed without the proposed framework, the difference in execution time is
minimal, indicating that the overhead introduced by the proposed framework is
negligible.

### 7. [Understanding the Characteristics of LLM-Generated Property-Based Tests in Exploring Edge Cases](http://arxiv.org/pdf/2510.25297v1)

Authors: Hidetake Tanaka, Haruto Tanaka, Kazumasa Shimari, Kenichi Matsumoto

As Large Language Models (LLMs) increasingly generate code in software
development, ensuring the quality of LLM-generated code has become important.
Traditional testing approaches using Example-based Testing (EBT) often miss
edge cases -- defects that occur at boundary values, special input patterns, or
extreme conditions. This research investigates the characteristics of
LLM-generated Property-based Testing (PBT) compared to EBT for exploring edge
cases. We analyze 16 HumanEval problems where standard solutions failed on
extended test cases, generating both PBT and EBT test codes using
Claude-4-sonnet. Our experimental results reveal that while each method
individually achieved a 68.75\% bug detection rate, combining both approaches
improved detection to 81.25\%. The analysis demonstrates complementary
characteristics: PBT effectively detects performance issues and edge cases
through extensive input space exploration, while EBT effectively detects
specific boundary conditions and special patterns. These findings suggest that
a hybrid approach leveraging both testing methods can improve the reliability
of LLM-generated code, providing guidance for test generation strategies in
LLM-based code generation.

### 8. [Dissect-and-Restore: AI-based Code Verification with Transient Refactoring](http://arxiv.org/pdf/2510.25406v1)

Authors: Changjie Wang, Mariano Scazzariello, Anoud Alshnaka, Roberto Guanciale, Dejan Kostić, Marco Chiesa

Formal verification is increasingly recognized as a critical foundation for
building reliable software systems. However, the need for specialized expertise
to write precise specifications, navigate complex proof obligations, and learn
annotations often makes verification an order of magnitude more expensive than
implementation. While modern AI systems can recognize patterns in mathematical
proofs and interpret natural language, effectively integrating them into the
formal verification process remains an open challenge. We present Prometheus, a
novel AI-assisted system that facilitates automated code verification with
current AI capabilities in conjunction with modular software engineering
principles (e.g., modular refactoring). Our approach begins by decomposing
complex program logic, such as nested loops, into smaller, verifiable
components. Once verified, these components are recomposed to construct a proof
of the original program. This decomposition-recomposition workflow is
non-trivial. Prometheus addresses this by guiding the proof search through
structured decomposition of complex lemmas into smaller, verifiable sub-lemmas.
When automated tools are insufficient, users can provide lightweight natural
language guidance to steer the proof process effectively. Our evaluation
demonstrates that transiently applying modular restructuring to the code
substantially improves the AI's effectiveness in verifying individual
components. This approach successfully verifies 86% of tasks in our curated
dataset, compared to 68% for the baseline. Gains are more pronounced with
increasing specification complexity, improving from 30% to 69%, and when
integrating proof outlines for complex programs, from 25% to 87%.

### 9. [What Challenges Do Developers Face in AI Agent Systems? An Empirical Study on Stack Overflow](http://arxiv.org/pdf/2510.25423v1)

Authors: Ali Asgari, Annibale Panichella, Pouria Derakhshanfar, Mitchell Olsthoorn

AI agents have rapidly gained popularity across research and industry as
systems that extend large language models with additional capabilities to plan,
use tools, remember, and act toward specific goals. Yet despite their promise,
developers face persistent and often underexplored challenges when building,
deploying, and maintaining these emerging systems. To identify these
challenges, we study developer discussions on Stack Overflow, the world's
largest developer-focused Q and A platform with about 60 million questions and
answers and 30 million users. We construct a taxonomy of developer challenges
through tag expansion and filtering, apply LDA-MALLET for topic modeling, and
manually validate and label the resulting themes. Our analysis reveals seven
major areas of recurring issues encompassing 77 distinct technical challenges
related to runtime integration, dependency management, orchestration
complexity, and evaluation reliability. We further quantify topic popularity
and difficulty to identify which issues are most common and hardest to resolve,
map the tools and programming languages used in agent development, and track
their evolution from 2021 to 2025 in relation to major AI model and framework
releases. Finally, we present the implications of our results, offering
concrete guidance for practitioners, researchers, and educators on agent
reliability and developer support.

### 10. [Fuzz Smarter, Not Harder: Towards Greener Fuzzing with GreenAFL](http://arxiv.org/pdf/2510.25665v1)

Authors: Ayse Irmak Ercevik, Aidan Dakhama, Melane Navaratnarajah, Yazhuo Cao, Leo Fernandes

Fuzzing has become a key search-based technique for software testing, but
continuous fuzzing campaigns consume substantial computational resources and
generate significant carbon footprints. Existing grey-box fuzzing approaches
like AFL++ focus primarily on coverage maximisation, without considering the
energy costs of exploring different execution paths. This paper presents
GreenAFL, an energy-aware framework that incorporates power consumption into
the fuzzing heuristics to reduce the environmental impact of automated testing
whilst maintaining coverage. GreenAFL introduces two key modifications to
traditional fuzzing workflows: energy-aware corpus minimisation considering
power consumption when reducing initial corpora, and energy-guided heuristics
that direct mutation towards high-coverage, low-energy inputs. We conduct an
ablation study comparing vanilla AFL++, energy-based corpus minimisation, and
energy-based heuristics to evaluate the individual contributions of each
component. Results show that highest coverage, and lowest energy usage is
achieved whenever at least one of our modifications is used.

### Social and Information Networks

### 1. [Merit Network Telescope: Processing and Initial Insights from Nearly 20 Years of Darknet Traffic for Cybersecurity Research](http://arxiv.org/pdf/2510.25050v1)

Authors: Shereen Ismail, Eman Hammad, William Hatcher, Salah Dandan, Ammar Alomari, Michael Spratt

This paper presents an initial longitudinal analysis of unsolicited Internet
traffic collected between 2005 and 2025 by one of the largest and most
persistent network telescopes in the United States, operated by Merit Network.
The dataset provides a unique view into global threat activity as observed
through scanning and backscatter traffic, key indicators of large-scale probing
behavior, data outages, and ongoing denial-of-service (DoS) campaigns. To
process this extensive archive, coarse-to-fine methodology is adopted in which
general insights are first extracted through a resource-efficient metadata
sub-pipeline, followed by a more detailed packet header sub-pipeline for
finer-grained analysis. The methodology establishes two sub-pipelines to enable
scalable processing of nearly two decades of telescope data and supports
multi-level exploration of traffic dynamics. Initial insights highlight
long-term trends and recurring traffic spikes, some attributable to
Internet-wide scanning events and others likely linked to DoS activities.We
present general observations spanning 2006-2024, with a focused analysis of
traffic characteristics during 2024.

### 2. [MMM-Fact: A Multimodal, Multi-Domain Fact-Checking Dataset with Multi-Level Retrieval Difficulty](http://arxiv.org/pdf/2510.25120v1)

Authors: Wenyan Xu, Dawei Xiang, Tianqi Ding, Weihai Lu

Misinformation and disinformation demand fact checking that goes beyond
simple evidence-based reasoning. Existing benchmarks fall short: they are
largely single modality (text-only), span short time horizons, use shallow
evidence, cover domains unevenly, and often omit full articles -- obscuring
models' real-world capability. We present MMM-Fact, a large-scale benchmark of
125,449 fact-checked statements (1995--2025) across multiple domains, each
paired with the full fact-check article and multimodal evidence (text, images,
videos, tables) from four fact-checking sites and one news outlet. To reflect
verification effort, each statement is tagged with a retrieval-difficulty tier
-- Basic (1--5 sources), Intermediate (6--10), and Advanced (>10) -- supporting
fairness-aware evaluation for multi-step, cross-modal reasoning. The dataset
adopts a three-class veracity scheme (true/false/not enough information) and
enables tasks in veracity prediction, explainable fact-checking, complex
evidence aggregation, and longitudinal analysis. Baselines with mainstream LLMs
show MMM-Fact is markedly harder than prior resources, with performance
degrading as evidence complexity rises. MMM-Fact offers a realistic, scalable
benchmark for transparent, reliable, multimodal fact-checking.

### 3. [Stable Emotional Co-occurrence Patterns Revealed by Network Analysis of Social Media](http://arxiv.org/pdf/2510.25204v1)

Authors: Qianyun Wu, Orr Levy, Yoed N. Kenett, Yukie Sano, Hideki Takayasu, Shlomo Havlin, Misako Takayasu

Examining emotion interactions as an emotion network in social media offers
key insights into human psychology, yet few studies have explored how
fluctuations in such emotion network evolve during crises and normal times.
This study proposes a novel computational approach grounded in network theory,
leveraging large-scale Japanese social media data spanning varied crisis events
(earthquakes and COVID-19 vaccination) and non-crisis periods over the past
decade. Our analysis identifies and evaluates links between emotions through
the co-occurrence of emotion-related concepts (words), revealing a stable
structure of emotion network across situations and over time at the population
level. We find that some emotion links (represented as link strength) such as
emotion links associated with Tension are significantly strengthened during
earthquake and pre-vaccination periods. However, the rank of emotion links
remains highly intact. These findings challenge the assumption that emotion
co-occurrence is context-based and offer a deeper understanding of emotions'
intrinsic structure. Moreover, our network-based framework offers a systematic,
scalable method for analyzing emotion co-occurrence dynamics, opening new
avenues for psychological research using large-scale textual data.

### 4. [Beyond Leakage and Complexity: Towards Realistic and Efficient Information Cascade Prediction](http://arxiv.org/pdf/2510.25348v1)

Authors: Jie Peng, Rui Wang, Qiang Wang, Zhewei Wei, Bin Tong, Guan Wang

Information cascade popularity prediction is a key problem in analyzing
content diffusion in social networks. However, current related works suffer
from three critical limitations: (1) temporal leakage in current
evaluation--random cascade-based splits allow models to access future
information, yielding unrealistic results; (2) feature-poor datasets that lack
downstream conversion signals (e.g., likes, comments, or purchases), which
limits more practical applications; (3) computational inefficiency of complex
graph-based methods that require days of training for marginal gains. We
systematically address these challenges from three perspectives: task setup,
dataset construction, and model design. First, we propose a time-ordered
splitting strategy that chronologically partitions data into consecutive
windows, ensuring models are evaluated on genuine forecasting tasks without
future information leakage. Second, we introduce Taoke, a large-scale
e-commerce cascade dataset featuring rich promoter/product attributes and
ground-truth purchase conversions--capturing the complete diffusion lifecycle
from promotion to monetization. Third, we develop CasTemp, a lightweight
framework that efficiently models cascade dynamics through temporal walks,
Jaccard-based neighbor selection for inter-cascade dependencies, and GRU-based
encoding with time-aware attention. Under leak-free evaluation, CasTemp
achieves state-of-the-art performance across four datasets with
orders-of-magnitude speedup. Notably, it excels at predicting second-stage
popularity conversions--a practical task critical for real-world applications.

### 5. [Testing Correlation in Graphs by Counting Bounded Degree Motifs](http://arxiv.org/pdf/2510.25289v1)

Authors: Dong Huang, Pengkun Yang

Correlation analysis is a fundamental step for extracting meaningful insights
from complex datasets. In this paper, we investigate the problem of detecting
correlation between two Erd\H{o}s-R\'enyi graphs $G(n,p)$, formulated as a
hypothesis testing problem: under the null hypothesis, the two graphs are
independent, while under the alternative hypothesis, they are correlated. We
develop a polynomial-time test by counting bounded degree motifs and prove its
effectiveness for any constant correlation coefficient $\rho$ when the edge
connecting probability satisfies $p\ge n^{-2/3}$. Our results overcome the
limitation requiring $\rho \ge \sqrt{\alpha}$, where $\alpha\approx 0.338$ is
the Otter's constant, extending it to any constant $\rho$. Methodologically,
bounded degree motifs -- ubiquitous in real networks -- make the proposed
statistic both natural and scalable. We also validate our method on synthetic
and real co-citation networks, further confirming that this simple motif family
effectively captures correlation signals and exhibits strong empirical
performance.

### Systems and Control

### 1. [Control Synthesis with Reinforcement Learning: A Modeling Perspective](http://arxiv.org/pdf/2510.25063v1)

Authors: Nikki Xu, Hien Tran

Controllers designed with reinforcement learning can be sensitive to model
mismatch. We demonstrate that designing such controllers in a virtual
simulation environment with an inaccurate model is not suitable for deployment
in a physical setup. Controllers designed using an accurate model is robust
against disturbance and small mismatch between the physical setup and the
mathematical model derived from first principles; while a poor model results in
a controller that performs well in simulation but fails in physical
experiments. Sensitivity analysis is used to justify these discrepancies and an
empirical region of attraction estimation help us visualize their robustness.

### 2. [Stochastic Long-Term Joint Decarbonization Planning for Power Systems and Data Centers: A Case Study in PJM](http://arxiv.org/pdf/2510.25118v1)

Authors: Zhentong Shao, Nanpeng Yu, Daniel Wong

With the rapid growth of artificial intelligence (AI) and cloud services,
data centers have become critical infrastructures driving digital economies,
with increasing energy demand heightening concerns over electricity use and
carbon emissions, emphasizing the need for carbon-aware infrastructure
planning. Most studies assume static power systems, focus only on operational
emissions, and overlook co-optimization. This paper proposes a dynamic joint
planning framework that co-optimizes long-term data center and power system
development over 15 years. The model determines siting, capacity, and type of
data centers alongside power generation expansion, storage deployment, and
retirements, accounting for both operational and embodied emissions. To handle
multi-scale uncertainty, a large-scale two-stage stochastic program is
formulated and solved via an enhanced Benders decomposition. Applied to the PJM
Interconnection, with curated datasets released on GitHub, results show the
system can support up to 55 GW peak data center demand, with Virginia (DOM) and
Northern Illinois (ComEd) as optimal hosts. Compared to non-joint planning, the
framework cuts investment cost by 12.6%, operational cost by 8.25%, and
emissions by 5.63%. Including lifecycle emissions further raises renewable
deployment by 25.5%, highlighting embodied carbon's role in deeper
decarbonization.

### 3. [The Waterbed Effect on Quasiperiodic Disturbance Observer: Avoidance of Sensitivity Tradeoff with Time Delays](http://arxiv.org/pdf/2510.25131v1)

Authors: Hisayoshi Muramatsu

In linear time-invariant systems, the sensitivity function to disturbances is
designed under a sensitivity tradeoff known as the waterbed effect. To
compensate for a quasiperiodic disturbance, a quasiperiodic disturbance
observer using time delays was proposed. Its sensitivity function avoids the
sensitivity tradeoff, achieving wideband harmonic suppression without
amplifying aperiodic disturbances or shifting harmonic suppression frequencies.
However, its open-loop transfer function is not rational and does not satisfy
the assumptions of existing Bode sensitivity integrals due to its time delays.
This paper provides Bode-like sensitivity integrals for the quasiperiodic
disturbance observer in both continuous-time and discrete-time representations
and clarifies the avoided sensitivity tradeoff with time delays.

### 4. [Shared Control for Vehicle Lane-Changing with Uncertain Driver Behaviors](http://arxiv.org/pdf/2510.25284v1)

Authors: Jiamin Wu, Chenguang Zhao, Huan Yu

Lane changes are common yet challenging driving maneuvers that require
continuous decision-making and dynamic interaction with surrounding vehicles.
Relying solely on human drivers for lane-changing can lead to traffic
disturbances due to the stochastic nature of human behavior and its variability
under different task demands. Such uncertainties may significantly degrade
traffic string stability, which is critical for suppressing disturbance
propagation and ensuring smooth merging of the lane-changing vehicles. This
paper presents a human-automation shared lane-changing control framework that
preserves driver authority while allowing automated assistance to achieve
stable maneuvers in the presence of driver's behavioral uncertainty. Human
driving behavior is modeled as a Markov jump process with transitions driven by
task difficulty, providing a tractable representation of stochastic state
switching. Based on this model, we first design a nominal stabilizing
controller that guarantees stochastic ${L}_2$ string stability under imperfect
mode estimation. To further balance performance and automated effort, we then
develop a Minimal Intervention Controller (MIC) that retains acceptable
stability while limiting automation. Simulations using lane-changing data from
the NGSIM dataset verify that the nominal controller reduces speed
perturbations and shorten lane-changing time, while the MIC further reduces
automated effort and enhances comfort but with moderate stability and
efficiency loss. Validations on the TGSIM dataset with SAE Level 2 vehicles
show that the MIC enables earlier lane changes than Level 2 control while
preserving driver authority with a slight stability compromise. These findings
highlight the potential of shared control strategies to balance stability,
efficiency, and driver acceptance.

### 5. [Data-Enabled Predictive Control and Guidance for Autonomous Underwater Vehicles](http://arxiv.org/pdf/2510.25309v1)

Authors: Sebastian Zieglmeier, Mathias Hudoba de Badyn, Narada D. Warakagoda, Thomas R. Krogstad, Paal Engelstad

This paper presents a fully data-driven control framework for autonomous
underwater vehicles (AUVs) based on Data-Enabled Predictive Control (DeePC).
The approach eliminates the need for explicit hydrodynamic modeling by
exploiting measured input-output data to predict and optimize future system
behavior. Classic DeePC was employed in the heading control, while a cascaded
DeePC architecture is proposed for depth regulation, incorporating a
loop-frequency separation to handle the different dynamic modes of input and
output. For 3-D waypoint path following, the Adaptive Line-of-Sight algorithm
is extended to a predictive formulation and integrated with DeePC. All methods
are validated in extensive simulation on the REMUS 100 AUV and compared with
classical PI/PID control. The results demonstrate superior tracking performance
and robustness of DeePC under ocean-current disturbances and nonlinear
operating conditions, while significantly reducing modeling effort.

### 6. [Tight Collision Avoidance for Stochastic Optimal Control: with Applications in Learning-based, Interactive Motion Planning](http://arxiv.org/pdf/2510.25324v1)

Authors: Erik Börve, Nikolce Murgovski, Leo Laine

Trajectory planning in dense, interactive traffic scenarios presents
significant challenges for autonomous vehicles, primarily due to the
uncertainty of human driver behavior and the non-convex nature of collision
avoidance constraints. This paper introduces a stochastic optimal control
framework to address these issues simultaneously, without excessively
conservative approximations. We opt to model human driver decisions as a Markov
Decision Process and propose a method for handling collision avoidance between
non-convex vehicle shapes by imposing a positive distance constraint between
compact sets. In this framework, we investigate three alternative chance
constraint formulations. To ensure computational tractability, we introduce
tight, continuously differentiable reformulations of both the non-convex
distance constraints and the chance constraints. The efficacy of our approach
is demonstrated through simulation studies of two challenging interactive
scenarios: an unregulated intersection crossing and a highway lane change in
dense traffic.

### 7. [Lightweight Federated Learning in Mobile Edge Computing with Statistical and Device Heterogeneity Awareness](http://arxiv.org/pdf/2510.25342v1)

Authors: Jinghong Tan, Zhichen Zhang, Kun Guo, Tsung-Hui Chang, Tony Q. S. Quek

Federated learning enables collaborative machine learning while preserving
data privacy, but high communication and computation costs, exacerbated by
statistical and device heterogeneity, limit its practicality in mobile edge
computing. Existing compression methods like sparsification and pruning reduce
per-round costs but may increase training rounds and thus the total training
cost, especially under heterogeneous environments. We propose a lightweight
personalized FL framework built on parameter decoupling, which separates the
model into shared and private subspaces, enabling us to uniquely apply gradient
sparsification to the shared component and model pruning to the private one.
This structural separation confines communication compression to global
knowledge exchange and computation reduction to local personalization,
protecting personalization quality while adapting to heterogeneous client
resources. We theoretically analyze convergence under the combined effects of
sparsification and pruning, revealing a sparsity-pruning trade-off that links
to the iteration complexity. Guided by this analysis, we formulate a joint
optimization that selects per-client sparsity and pruning rates and wireless
bandwidth to reduce end-to-end training time. Simulation results demonstrate
faster convergence and substantial reductions in overall communication and
computation costs with negligible accuracy loss, validating the benefits of
coordinated and resource-aware personalization in resource-constrained
heterogeneous environments.

### 8. [Quantum-Resilient Threat Modelling for Secure RIS-Assisted ISAC in 6G UAV Corridors](http://arxiv.org/pdf/2510.25411v1)

Authors: Sana Hafeez, Ghulam E Mustafa Abro, Hifza Mustafa

The rapid deployment of unmanned aerial vehicle (UAV) corridors in
sixth-generation (6G) networks requires safe, intelligence-driven integrated
sensing and communications (ISAC). Reconfigurable intelligent surfaces (RIS)
enhance spectrum efficiency, localisation accuracy, and situational awareness,
while introducing new vulnerabilities. The rise of quantum computing increases
the risks associated with harvest-now-decrypt-later strategies and
quantum-enhanced spoofing. We propose a Quantum-Resilient Threat Modelling
(QRTM) framework for RIS-assisted ISAC in UAV corridors to address these
challenges. QRTM integrates classical, quantum-ready, and quantum-aided
adversaries, countered using post-quantum cryptographic (PQC) primitives:
ML-KEM for key establishment and Falcon for authentication, both embedded
within RIS control signalling and UAV coordination. To strengthen security
sensing, the framework introduces RIS-coded scene watermarking validated
through a generalised likelihood ratio test (GLRT), with its detection
probability characterised by the Marcum Q function. Furthermore, a Secure ISAC
Utility (SIU) jointly optimises secrecy rate, spoofing detection, and
throughput under RIS constraints, enabled by a scheduler with computational
complexity of O(n^2). Monte Carlo evaluations using 3GPP Release 19 mid-band
urban-canyon models (7-15 GHz) demonstrate a spoof-detection probability
approaching 0.99 at a false-alarm rate of 1e-3, secrecy-rate retention
exceeding 90 percent against quantum-capable adversaries, and
signal-interference utilisation improvements of about 25 percent compared with
baselines. These results show a standards-compliant path towards reliable,
quantum-resilient ISAC for UAV corridors in smart cities and non-terrestrial
networks.

### 9. [A New Neural Network Paradigm for Scalable and Generalizable Stability Analysis of Power Systems](http://arxiv.org/pdf/2510.25501v1)

Authors: Tong Han, Yan Xu, Rui Zhang

This paper presents a new neural network (NN) paradigm for scalable and
generalizable stability analysis of power systems. The paradigm consists of two
parts: the neural stability descriptor and the sample-augmented iterative
training scheme. The first part, based on system decomposition, constructs the
object (such as a stability function or condition) for stability analysis as a
scalable aggregation of multiple NNs. These NNs remain fixed across varying
power system structures and parameters, and are repeatedly shared within each
system instance defined by these variations, thereby enabling the
generalization of the neural stability descriptor across a class of power
systems. The second part learns the neural stability descriptor by iteratively
training the NNs with sample augmentation, guided by the tailored
conservativeness-aware loss function. The training set is strategically
constructed to promote the descriptor's generalizability, which is
systematically evaluated by verification and validation during the training
process. Specifically, the proposed NN paradigm is implemented for
large-disturbance stability analysis of the bulk power grid and
small-disturbance stability conditions of the microgrid system. Finally,
numerical studies for the two implementations demonstrate the applicability and
effectiveness of the proposed NN paradigm.

### 10. [Optimal and Heuristic Approaches for Platooning Systems with Deadlines](http://arxiv.org/pdf/2510.25564v1)

Authors: Thiago S. Gomides, Evangelos Kranakis, Ioannis Lambadaris, Yannis Viniotis, Gennady Shaikhet

Efficient truck platooning is a key strategy for reducing freight costs,
lowering fuel consumption, and mitigating emissions. Deadlines are critical in
this context, as trucks must depart within specific time windows to meet
delivery requirements and avoid penalties. In this paper, we investigate the
optimal formation and dispatch of truck platoons at a highway station with
finite capacity $L$ and deadline constraints $T$. The system operates in
discrete time, with each arriving truck assigned a deadline of $T$ slot units.
The objective is to leverage the efficiency gains from forming large platoons
while accounting for waiting costs and deadline violations. We formulate the
problem as a Markov decision process and analyze the structure of the optimal
policy $\pi^\star$ for $L = 3$, extending insights to arbitrary $L$. We prove
that the $\pi^\star$ is monotone in the state space $\mathcal{S}$ and identify
classes of unreachable states. Moreover, since $\mathcal{S}$ grows
exponentially with $L$ and $T$, we propose heuristics-including conditional and
deep-learning based approaches-that exploit these structural insights while
maintaining low computational complexity.

### Machine Learning (Statistics Category)

### 1. [Shift is Good: Mismatched Data Mixing Improves Test Performance](http://arxiv.org/pdf/2510.25108v1)

Authors: Marko Medvedev, Kaifeng Lyu, Zhiyuan Li, Nathan Srebro

We consider training and testing on mixture distributions with different
training and test proportions. We show that in many settings, and in some sense
generically, distribution shift can be beneficial, and test performance can
improve due to mismatched training proportions, even if the components are
unrelated and with no transfer between components. In a variety of scenarios,
we identify the optimal training proportions and the extent to which such
distribution shift can be beneficial. We show how the same analysis applies
also to a compositional setting with differing distribution of component
"skills'' at training and test.

### 2. [An Analysis of Causal Effect Estimation using Outcome Invariant Data Augmentation](http://arxiv.org/pdf/2510.25128v1)

Authors: Uzair Akbar, Niki Kilbertus, Hao Shen, Krikamol Muandet, Bo Dai

The technique of data augmentation (DA) is often used in machine learning for
regularization purposes to better generalize under i.i.d. settings. In this
work, we present a unifying framework with topics in causal inference to make a
case for the use of DA beyond just the i.i.d. setting, but for generalization
across interventions as well. Specifically, we argue that when the outcome
generating mechanism is invariant to our choice of DA, then such augmentations
can effectively be thought of as interventions on the treatment generating
mechanism itself. This can potentially help to reduce bias in causal effect
estimation arising from hidden confounders. In the presence of such unobserved
confounding we typically make use of instrumental variables (IVs) -- sources of
treatment randomization that are conditionally independent of the outcome.
However, IVs may not be as readily available as DA for many applications, which
is the main motivation behind this work. By appropriately regularizing IV based
estimators, we introduce the concept of IV-like (IVL) regression for mitigating
confounding bias and improving predictive performance across interventions even
when certain IV properties are relaxed. Finally, we cast parameterized DA as an
IVL regression problem and show that when used in composition can simulate a
worst-case application of such DA, further improving performance on causal
estimation and generalization tasks beyond what simple DA may offer. This is
shown both theoretically for the population case and via simulation experiments
for the finite sample case using a simple linear example. We also present real
data experiments to support our case.

### 3. [Generative Bayesian Optimization: Generative Models as Acquisition Functions](http://arxiv.org/pdf/2510.25240v1)

Authors: Rafael Oliveira, Daniel M. Steinberg, Edwin V. Bonilla

We present a general strategy for turning generative models into candidate
solution samplers for batch Bayesian optimization (BO). The use of generative
models for BO enables large batch scaling as generative sampling, optimization
of non-continuous design spaces, and high-dimensional and combinatorial design.
Inspired by the success of direct preference optimization (DPO), we show that
one can train a generative model with noisy, simple utility values directly
computed from observations to then form proposal distributions whose densities
are proportional to the expected utility, i.e., BO's acquisition function
values. Furthermore, this approach is generalizable beyond preference-based
feedback to general types of reward signals and loss functions. This
perspective avoids the construction of surrogate (regression or classification)
models, common in previous methods that have used generative models for
black-box optimization. Theoretically, we show that the generative models
within the BO process approximately follow a sequence of distributions which
asymptotically concentrate at the global optima under certain conditions. We
also demonstrate this effect through experiments on challenging optimization
problems involving large batches in high dimensions.

### 4. [Tuning-Free Sampling via Optimization on the Space of Probability Measures](http://arxiv.org/pdf/2510.25315v1)

Authors: Louis Sharrock, Christopher Nemeth

We introduce adaptive, tuning-free step size schedules for gradient-based
sampling algorithms obtained as time-discretizations of Wasserstein gradient
flows. The result is a suite of tuning-free sampling algorithms, including
tuning-free variants of the unadjusted Langevin algorithm (ULA), stochastic
gradient Langevin dynamics (SGLD), mean-field Langevin dynamics (MFLD), Stein
variational gradient descent (SVGD), and variational gradient descent (VGD).
More widely, our approach yields tuning-free algorithms for solving a broad
class of stochastic optimization problems over the space of probability
measures. Under mild assumptions (e.g., geodesic convexity and locally bounded
stochastic gradients), we establish strong theoretical guarantees for our
approach. In particular, we recover the convergence rate of optimally tuned
versions of these algorithms up to logarithmic factors, in both nonsmooth and
smooth settings. We then benchmark the performance of our methods against
comparable existing approaches. Across a variety of tasks, our algorithms
achieve similar performance to the optimal performance of existing algorithms,
with no need to tune a step size parameter.

### 5. [Distributional Evaluation of Generative Models via Relative Density Ratio](http://arxiv.org/pdf/2510.25507v1)

Authors: Yuliang Xu, Yun Wei, Li Ma

We propose a functional evaluation metric for generative models based on the
relative density ratio (RDR) designed to characterize distributional
differences between real and generated samples. We show that the RDR as a
functional summary of the goodness-of-fit for the generative model, possesses
several desirable theoretical properties. It preserves $\phi$-divergence
between two distributions, enables sample-level evaluation that facilitates
downstream investigations of feature-specific distributional differences, and
has a bounded range that affords clear interpretability and numerical
stability. Functional estimation of the RDR is achieved efficiently through
convex optimization on the variational form of $\phi$-divergence. We provide
theoretical convergence rate guarantees for general estimators based on
M-estimator theory, as well as the convergence rates of neural network-based
estimators when the true ratio is in the anisotropic Besov space. We
demonstrate the power of the proposed RDR-based evaluation through numerical
experiments on MNIST, CelebA64, and the American Gut project microbiome data.
We show that the estimated RDR not only allows for an effective comparison of
the overall performance of competing generative models, but it can also offer a
convenient means of revealing the nature of the underlying goodness-of-fit.
This enables one to assess support overlap, coverage, and fidelity while
pinpointing regions of the sample space where generators concentrate and
revealing the features that drive the most salient distributional differences.

### 6. [Convergence of off-policy TD(0) with linear function approximation for reversible Markov chains](http://arxiv.org/pdf/2510.25514v1)

Authors: Maik Overmars, Jasper Goseling, Richard Boucherie

We study the convergence of off-policy TD(0) with linear function
approximation when used to approximate the expected discounted reward in a
Markov chain. It is well known that the combination of off-policy learning and
function approximation can lead to divergence of the algorithm. Existing
results for this setting modify the algorithm, for instance by reweighing the
updates using importance sampling. This establishes convergence at the expense
of additional complexity. In contrast, our approach is to analyse the standard
algorithm, but to restrict our attention to the class of reversible Markov
chains. We demonstrate convergence under this mild reversibility condition on
the structure of the chain, which in many applications can be assumed using
domain knowledge. In particular, we establish a convergence guarantee under an
upper bound on the discount factor in terms of the difference between the
on-policy and off-policy process. This improves upon known results in the
literature that state that convergence holds for a sufficiently small discount
factor by establishing an explicit bound. Convergence is with probability one
and achieves projected Bellman error equal to zero. To obtain these results, we
adapt the stochastic approximation framework that was used by Tsitsiklis and
Van Roy [1997 for the on-policy case, to the off-policy case. We illustrate our
results using different types of reversible Markov chains, such as
one-dimensional random walks and random walks on a weighted graph.

### 7. [Monitoring the calibration of probability forecasts with an application to concept drift detection involving image classification](http://arxiv.org/pdf/2510.25573v1)

Authors: Christopher T. Franck, Anne R. Driscoll, Zoe Szajnfarber, William H. Woodall

Machine learning approaches for image classification have led to impressive
advances in that field. For example, convolutional neural networks are able to
achieve remarkable image classification accuracy across a wide range of
applications in industry, defense, and other areas. While these machine
learning models boast impressive accuracy, a related concern is how to assess
and maintain calibration in the predictions these models make. A classification
model is said to be well calibrated if its predicted probabilities correspond
with the rates events actually occur. While there are many available methods to
assess machine learning calibration and recalibrate faulty predictions, less
effort has been spent on developing approaches that continually monitor
predictive models for potential loss of calibration as time passes. We propose
a cumulative sum-based approach with dynamic limits that enable detection of
miscalibration in both traditional process monitoring and concept drift
applications. This enables early detection of operational context changes that
impact image classification performance in the field. The proposed chart can be
used broadly in any situation where the user needs to monitor probability
predictions over time for potential lapses in calibration. Importantly, our
method operates on probability predictions and event outcomes and does not
require under-the-hood access to the machine learning model.

### 8. [Generalized Sobolev IPM for Graph-Based Measures](http://arxiv.org/pdf/2510.25591v1)

Authors: Tam Le, Truyen Nguyen, Hideitsu Hino, Kenji Fukumizu

We study the Sobolev IPM problem for measures supported on a graph metric
space, where critic function is constrained to lie within the unit ball defined
by Sobolev norm. While Le et al. (2025) achieved scalable computation by
relating Sobolev norm to weighted $L^p$-norm, the resulting framework remains
intrinsically bound to $L^p$ geometric structure, limiting its ability to
incorporate alternative structural priors beyond the $L^p$ geometry paradigm.
To overcome this limitation, we propose to generalize Sobolev IPM through the
lens of \emph{Orlicz geometric structure}, which employs convex functions to
capture nuanced geometric relationships, building upon recent advances in
optimal transport theory -- particularly Orlicz-Wasserstein (OW) and
generalized Sobolev transport -- that have proven instrumental in advancing
machine learning methodologies. This generalization encompasses classical
Sobolev IPM as a special case while accommodating diverse geometric priors
beyond traditional $L^p$ structure. It however brings up significant
computational hurdles that compound those already inherent in Sobolev IPM. To
address these challenges, we establish a novel theoretical connection between
Orlicz-Sobolev norm and Musielak norm which facilitates a novel regularization
for the generalized Sobolev IPM (GSI). By further exploiting the underlying
graph structure, we show that GSI with Musielak regularization (GSI-M) reduces
to a simple \emph{univariate optimization} problem, achieving remarkably
computational efficiency. Empirically, GSI-M is several-order faster than the
popular OW in computation, and demonstrates its practical advantages in
comparing probability measures on a given graph for document classification and
several tasks in topological data analysis.

### 9. [How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](http://arxiv.org/pdf/2510.25753v1)

Authors: Samet Demir, Zafer Dogan

Pretrained Transformers demonstrate remarkable in-context learning (ICL)
capabilities, enabling them to adapt to new tasks from demonstrations without
parameter updates. However, theoretical studies often rely on simplified
architectures (e.g., omitting MLPs), data models (e.g., linear regression with
isotropic inputs), and single-source training, limiting their relevance to
realistic settings. In this work, we study ICL in pretrained Transformers with
nonlinear MLP heads on nonlinear tasks drawn from multiple data sources with
heterogeneous input, task, and noise distributions. We analyze a model where
the MLP comprises two layers, with the first layer trained via a single
gradient step and the second layer fully optimized. Under high-dimensional
asymptotics, we prove that such models are equivalent in ICL error to
structured polynomial predictors, leveraging results from the theory of
Gaussian universality and orthogonal polynomials. This equivalence reveals that
nonlinear MLPs meaningfully enhance ICL performance, particularly on nonlinear
tasks, compared to linear baselines. It also enables a precise analysis of data
mixing effects: we identify key properties of high-quality data sources (low
noise, structured covariances) and show that feature learning emerges only when
the task covariance exhibits sufficient structure. These results are validated
empirically across various activation functions, model sizes, and data
distributions. Finally, we experiment with a real-world scenario involving
multilingual sentiment analysis where each language is treated as a different
source. Our experimental results for this case exemplify how our findings
extend to real-world cases. Overall, our work advances the theoretical
foundations of ICL in Transformers and provides actionable insight into the
role of architecture and data in ICL.

### 10. [Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions](http://arxiv.org/pdf/2510.25769v1)

Authors: Naoki Kiyohara, Edward Johns, Yingzhen Li

Stochastic differential equations (SDEs) are well suited to modelling noisy
and irregularly sampled time series found in finance, physics, and machine
learning. Traditional approaches require costly numerical solvers to sample
between arbitrary time points. We introduce Neural Stochastic Flows (NSFs) and
their latent variants, which directly learn (latent) SDE transition laws using
conditional normalising flows with architectural constraints that preserve
properties inherited from stochastic flows. This enables one-shot sampling
between arbitrary states and yields up to two orders of magnitude speed-ups at
large time gaps. Experiments on synthetic SDE simulations and on real-world
tracking and video data show that NSFs maintain distributional accuracy
comparable to numerical approaches while dramatically reducing computation for
arbitrary time-point sampling.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-30 PST.

### 1. [Multi-ship detection and classification with feature enhancement and lightweight fusion](https://www.nature.com/articles/s41598-025-21887-6)

Authors: Ying Han et al.

### 2. [Bayesian continual learning and forgetting in neural networks](https://www.nature.com/articles/s41467-025-64601-w)

Authors: Djohan Bonnet et al.

### 3. [Efficient optimization accelerator framework for multi-state spin Ising problems](https://www.nature.com/articles/s41467-025-64625-2)

Authors: Chirag Garg et al.

### 4. [Humans and neural networks show similar patterns of transfer and interference during continual learning](https://www.nature.com/articles/s41562-025-02318-y)

Authors: Eleanor Holton et al.

### 5. [Evaluating large transformer models for anomaly detection of resource-constrained IoT devices for intrusion detection system](https://www.nature.com/articles/s41598-025-21826-5)

Authors: Ahmad Almadhor et al.

### 6. [A Comprehensive Dataset for Image Segmentation in Custom Manufacturing Environments](https://www.nature.com/articles/s41597-025-06007-3)

Authors: Martell Bell et al.

### 7. [A dual-contract architecture with role-based access control for supply chain traceability and accountability](https://www.nature.com/articles/s41598-025-20464-1)

Authors: Eesa Alsolami et al.

### 8. [DC-Mamba for surface micro defect classification on large aperture optics with multi axis attention](https://www.nature.com/articles/s41598-025-21756-2)

Authors:  Dejin Zhao et al.

### 9. [A multi-label visualisation approach for malware behaviour analysis](https://www.nature.com/articles/s41598-025-21848-z)

Authors: Dilara T. Uysal et al.

### 10. [An integrated approach for rare disease detection and classification in Spanish pediatric medical reports](https://www.nature.com/articles/s41598-025-21827-4)

Authors: Andres Duque et al.

### 11. [A computer vision framework for proactive anomaly detection and risk reduction in airport baggage logistics](https://www.nature.com/articles/s41598-025-21959-7)

Authors: Kalyani Vidhate et al.

### 12. [Generative augmentations for improved cardiac ultrasound segmentation using diffusion models](https://www.nature.com/articles/s41598-025-21938-y)

Authors: Gilles Van De Vyver et al.

