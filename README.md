# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-29 17:00:25.835725 PST.

### Artificial Intelligence

### 1. [AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning](http://arxiv.org/pdf/2508.20368v1)

Authors: Lang Mei, Zhihan Yang, Chong Chen

Recent studies have explored integrating Large Language Models (LLMs) with
search engines to leverage both the LLMs' internal pre-trained knowledge and
external information. Specially, reinforcement learning (RL) has emerged as a
promising paradigm for enhancing LLM reasoning through multi-turn interactions
with search engines. However, existing RL-based search agents rely on a single
LLM to handle both search planning and question-answering (QA) tasks in an
end-to-end manner, which limits their ability to optimize both capabilities
simultaneously. In practice, sophisticated AI search systems often employ a
large, frozen LLM (e.g., GPT-4, DeepSeek-R1) to ensure high-quality QA. Thus, a
more effective and efficient approach is to utilize a small, trainable LLM
dedicated to search planning. In this paper, we propose
\textbf{AI-SearchPlanner}, a novel reinforcement learning framework designed to
enhance the performance of frozen QA models by focusing on search planning.
Specifically, our approach introduces three key innovations: 1) Decoupling the
Architecture of the Search Planner and Generator, 2) Dual-Reward Alignment for
Search Planning, and 3) Pareto Optimization of Planning Utility and Cost, to
achieve the objectives. Extensive experiments on real-world datasets
demonstrate that AI SearchPlanner outperforms existing RL-based search agents
in both effectiveness and efficiency, while exhibiting strong generalization
capabilities across diverse frozen QA models and data domains.

### 2. [TCIA: A Task-Centric Instruction Augmentation Method for Instruction Finetuning](http://arxiv.org/pdf/2508.20374v1)

Authors: Simin Ma, Shujian Liu, Jun Tan, Yebowen Hu, Song Wang, Sathish Reddy Indurthi, Sanqiang Zhao, Liwei Wu, Jianbing Han, Kaiqiang Song

Diverse instruction data is vital for effective instruction tuning of large
language models, as it enables the model to generalize across different types
of inputs . Building such diversified instruction dataset is an essential step
in this process. Existing approaches often leverage large language models to
automatically explore and generate diverse instructions, ensuring both data
diversity and quality. However, they tend to overlook an important factor in
real-world applications: on-task relevance. In practice, only a few real-world
applications require a truly general-purpose model; most benefit from
task-specific knowledge tailored to their particular use case. Therefore, it is
vital to develop instruction augmentation methods that not only maintain
diversity but are also optimized for specific, real-world scenarios.
  We thus introduce Task Centric Instruction Augmentation (TCIA), a framework
that systematically expands instructions while preserving both diversity and
task alignment. By representing instructions in a discrete query-constraints
space, TCIA creates a rich set of task-relevant instructions and enables models
to generalize to these task-specific instructions without sacrificing overall
performance. Experiments show that TCIA improves open-source LLMs' performance
by an average of 8.7% across four real-world, task-specific applications, and
in some cases outperforming leading closed-source models. These improvements do
not compromise general instruction-following ability, making TCIA a scalable
and efficient solution for adapting LLMs to real-world, task-focused
applications.

### 3. [Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLM](http://arxiv.org/pdf/2508.20384v1)

Authors: Yongfu Zhu, Lin Sun, Guangxiang Zhao, Weihong Lin, Xiangzheng Zhang

In this work, we introduce Entropy Area Score (EAS), a simple yet effective
metric to quantify uncertainty in the answer generation process of reasoning
large language models (LLMs). EAS requires neither external models nor repeated
sampling, it integrates token-level predictive entropy from the model itself to
capture the evolution of uncertainty during generation. Empirical results show
that EAS is strongly correlated with answer entropy across models and datasets.
In training data selection, EAS identifies high-potential samples and
consistently outperforms Pass Rate filtering under equal sample budgets,
improving student model accuracy on math benchmarks. EAS is both efficient and
interpretable, offering a practical tool for uncertainty modeling and data
quality assessment in LLM training.

### 4. [AWorld: Orchestrating the Training Recipe for Agentic AI](http://arxiv.org/pdf/2508.20404v1)

Authors: Chengyue Yu, Siyuan Lu, Chenyi Zhuang, Dong Wang, Qintong Wu, Zongyue Li, Runsheng Gan, Chunfeng Wang, Siqi Hou, Gaochi Huang, Wenlong Yan, Lifeng Hong, Aohui Xue, Yanfeng Wang, Jinjie Gu, David Tsai, Tao Lin

The learning from practice paradigm is crucial for developing capable Agentic
AI systems, yet it is severely hampered by inefficient experience generation, a
bottleneck especially pronounced in complex benchmarks like GAIA. To address
this, we introduce AWorld, an open-source system engineered for large-scale
agent-environment interaction. By distributing tasks across a cluster, AWorld
accelerates experience collection by 14.6x compared to standard single-node,
sequential execution. This critical speedup makes extensive reinforcement
learning practical and scalable. Leveraging this capability, we trained a
Qwen3-32B-based agent that significantly outperforms its base model, increasing
its overall GAIA accuracy from 21.59% to 32.23%. On the benchmark's most
challenging levels, our agent achieves a score of 16.33%, surpassing the
performance of leading proprietary models. Our open-source system and resulting
agent provide a practical blueprint for a complete agentic AI training
pipeline, from efficient interaction to demonstrable model improvement.

### 5. [Enhancing Health Fact-Checking with LLM-Generated Synthetic Data](http://arxiv.org/pdf/2508.20525v1)

Authors: Jingze Zhang, Jiahe Qian, Yiliang Zhou, Yifan Peng

Fact-checking for health-related content is challenging due to the limited
availability of annotated training data. In this study, we propose a synthetic
data generation pipeline that leverages large language models (LLMs) to augment
training data for health-related fact checking. In this pipeline, we summarize
source documents, decompose the summaries into atomic facts, and use an LLM to
construct sentence-fact entailment tables. From the entailment relations in the
table, we further generate synthetic text-claim pairs with binary veracity
labels. These synthetic data are then combined with the original data to
fine-tune a BERT-based fact-checking model. Evaluation on two public datasets,
PubHealth and SciFact, shows that our pipeline improved F1 scores by up to
0.019 and 0.049, respectively, compared to models trained only on the original
data. These results highlight the effectiveness of LLM-driven synthetic data
augmentation in enhancing the performance of health-related fact-checkers.

### 6. [Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control](http://arxiv.org/pdf/2508.20784v1)

Authors: Yifan Zhang

Bus bunching remains a challenge for urban transit due to stochastic traffic
and passenger demand. Traditional solutions rely on multi-agent reinforcement
learning (MARL) in loop-line settings, which overlook realistic operations
characterized by heterogeneous routes, timetables, fluctuating demand, and
varying fleet sizes. We propose a novel single-agent reinforcement learning
(RL) framework for bus holding control that avoids the data imbalance and
convergence issues of MARL under near-realistic simulation. A bidirectional
timetabled network with dynamic passenger demand is constructed. The key
innovation is reformulating the multi-agent problem into a single-agent one by
augmenting the state space with categorical identifiers (vehicle ID, station
ID, time period) in addition to numerical features (headway, occupancy,
velocity). This high-dimensional encoding enables single-agent policies to
capture inter-agent dependencies, analogous to projecting non-separable inputs
into a higher-dimensional space. We further design a structured reward function
aligned with operational goals: instead of exponential penalties on headway
deviations, a ridge-shaped reward balances uniform headways and schedule
adherence. Experiments show that our modified soft actor-critic (SAC) achieves
more stable and superior performance than benchmarks, including MADDPG (e.g.,
-430k vs. -530k under stochastic conditions). These results demonstrate that
single-agent deep RL, when enhanced with categorical structuring and
schedule-aware rewards, can effectively manage bus holding in non-loop,
real-world contexts. This paradigm offers a robust, scalable alternative to
MARL frameworks, particularly where agent-specific experiences are imbalanced.

### 7. [ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery](http://arxiv.org/pdf/2508.20996v1)

Authors: Junda Wang, Zonghai Yao, Zhichao Yang, Lingxi Li, Junhui Qian, Hong Yu

Substance use disorders (SUDs) affect over 36 million people worldwide, yet
few receive effective care due to stigma, motivational barriers, and limited
personalized support. Although large language models (LLMs) show promise for
mental-health assistance, most systems lack tight integration with clinically
validated strategies, reducing effectiveness in addiction recovery. We present
ChatThero, a multi-agent conversational framework that couples dynamic patient
modeling with context-sensitive therapeutic dialogue and adaptive persuasive
strategies grounded in cognitive behavioral therapy (CBT) and motivational
interviewing (MI). We build a high-fidelity synthetic benchmark spanning Easy,
Medium, and Hard resistance levels, and train ChatThero with a two-stage
pipeline comprising supervised fine-tuning (SFT) followed by direct preference
optimization (DPO). In evaluation, ChatThero yields a 41.5\% average gain in
patient motivation, a 0.49\% increase in treatment confidence, and resolves
hard cases with 26\% fewer turns than GPT-4o, and both automated and human
clinical assessments rate it higher in empathy, responsiveness, and behavioral
realism. The framework supports rigorous, privacy-preserving study of
therapeutic conversation and provides a robust, replicable basis for research
and clinical translation.

### 8. [Multi-View Graph Convolution Network for Internal Talent Recommendation Based on Enterprise Emails](http://arxiv.org/pdf/2508.20328v1)

Authors: Soo Hyun Kim, Jang-Hyun Kim

Internal talent recommendation is a critical strategy for organizational
continuity, yet conventional approaches suffer from structural limitations,
often overlooking qualified candidates by relying on the narrow perspective of
a few managers. To address this challenge, we propose a novel framework that
models two distinct dimensions of an employee's position fit from email data:
WHAT they do (semantic similarity of tasks) and HOW they work (structural
characteristics of their interactions and collaborations). These dimensions are
represented as independent graphs and adaptively fused using a Dual Graph
Convolutional Network (GCN) with a gating mechanism. Experiments show that our
proposed gating-based fusion model significantly outperforms other fusion
strategies and a heuristic baseline, achieving a top performance of 40.9% on
Hit@100. Importantly, it is worth noting that the model demonstrates high
interpretability by learning distinct, context-aware fusion strategies for
different job families. For example, it learned to prioritize relational (HOW)
data for 'sales and marketing' job families while applying a balanced approach
for 'research' job families. This research offers a quantitative and
comprehensive framework for internal talent discovery, minimizing the risk of
candidate omission inherent in traditional methods. Its primary contribution
lies in its ability to empirically determine the optimal fusion ratio between
task alignment (WHAT) and collaborative patterns (HOW), which is required for
employees to succeed in the new positions, thereby offering important practical
implications.

### 9. [Adaptive Root Cause Localization for Microservice Systems with Multi-Agent Recursion-of-Thought](http://arxiv.org/pdf/2508.20370v1)

Authors: Lingzhe Zhang, Tong Jia, Kangjin Wang, Weijie Hong, Chiming Duan, Minghua He, Ying Li

As contemporary microservice systems become increasingly popular and
complex-often comprising hundreds or even thousands of fine-grained,
interdependent subsystems-they are facing more frequent failures. Ensuring
system reliability thus demands accurate root cause localization. While traces
and metrics have proven to be effective data sources for this task, existing
methods either heavily rely on pre-defined schemas, which struggle to adapt to
evolving operational contexts, or lack interpretability in their reasoning
process, thereby leaving Site Reliability Engineers (SREs) confused. In this
paper, we conduct a comprehensive study on how SREs localize the root cause of
failures, drawing insights from multiple professional SREs across different
organizations. Our investigation reveals that human root cause analysis
exhibits three key characteristics: recursiveness, multi-dimensional expansion,
and cross-modal reasoning. Motivated by these findings, we introduce RCLAgent,
an adaptive root cause localization method for microservice systems that
leverages a multi-agent recursion-of-thought framework. RCLAgent employs a
novel recursion-of-thought strategy to guide the LLM's reasoning process,
effectively integrating data from multiple agents and tool-assisted analysis to
accurately pinpoint the root cause. Experimental evaluations on various public
datasets demonstrate that RCLAgent achieves superior performance by localizing
the root cause using only a single request-outperforming state-of-the-art
methods that depend on aggregating multiple requests. These results underscore
the effectiveness of RCLAgent in enhancing the efficiency and precision of root
cause localization in complex microservice environments.

### 10. [Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection](http://arxiv.org/pdf/2508.20392v1)

Authors: Chengjun Zhang, Yuhao Zhang, Jie Yang, Mohamad Sawan

Spiking Neural Networks (SNNs), inspired by the brain, are characterized by
minimal power consumption and swift inference capabilities on neuromorphic
hardware, and have been widely applied to various visual perception tasks.
Current ANN-SNN conversion methods have achieved excellent results in
classification tasks with ultra-low time-steps, but their performance in visual
detection tasks remains suboptimal. In this paper, we propose a delay-spike
approach to mitigate the issue of residual membrane potential caused by
heterogeneous spiking patterns. Furthermore, we propose a novel
temporal-dependent Integrate-and-Fire (tdIF) neuron architecture for SNNs. This
enables Integrate-and-fire (IF) neurons to dynamically adjust their
accumulation and firing behaviors based on the temporal order of time-steps.
Our method enables spikes to exhibit distinct temporal properties, rather than
relying solely on frequency-based representations. Moreover, the tdIF neuron
maintains energy consumption on par with traditional IF neuron. We demonstrate
that our method achieves more precise feature representation with lower
time-steps, enabling high performance and ultra-low latency in visual detection
tasks. In this study, we conduct extensive evaluation of the tdIF method across
two critical vision tasks: object detection and lane line detection. The
results demonstrate that the proposed method surpasses current ANN-SNN
conversion approaches, achieving state-of-the-art performance with ultra-low
latency (within 5 time-steps).

### Hardware Architecture

### 1. [The Future of Memory: Limits and Opportunities](http://arxiv.org/pdf/2508.20425v1)

Authors: Shuhan Liu, Samuel Dayo, Peijing Li, Philip Levis, Subhasish Mitra, Thierry Tambe, David Tennenhouse, H. -S. Philip Wong

Memory latency, bandwidth, capacity, and energy increasingly limit
performance. In this paper, we reconsider proposed system architectures that
consist of huge (many-terabyte to petabyte scale) memories shared among large
numbers of CPUs. We argue two practical engineering challenges, scaling and
signaling, limit such designs. We propose the opposite approach. Rather than
create large, shared, homogenous memories, systems explicitly break memory up
into smaller slices more tightly coupled with compute elements. Leveraging
advances in 2.5D/3D integration, this compute-memory node provisions private
local memory, enabling accesses of node-exclusive data through micrometer-scale
distances, and dramatically reduced access cost. In-package memory elements
support shared state within a processor, providing far better bandwidth and
energy-efficiency than DRAM, which is used as main memory for large working
sets and cold data. Hardware making memory capacities and distances explicit
allows software to efficiently compose this hierarchy, managing data placement
and movement.

### 2. [Microarchitecture Design and Benchmarking of Custom SHA-3 Instruction for RISC-V](http://arxiv.org/pdf/2508.20653v1)

Authors: Alperen Bolat, Sakir Sezer, Kieran McLaughlin, Henry Hui

Integrating cryptographic accelerators into modern CPU architectures presents
unique microarchitectural challenges, particularly when extending instruction
sets with complex and multistage operations. Hardware-assisted cryptographic
instructions, such as Intel's AES-NI and ARM's custom instructions for
encryption workloads, have demonstrated substantial performance improvements.
However, efficient SHA-3 acceleration remains an open problem due to its
distinct permutation-based structure and memory access patterns. Existing
solutions primarily rely on standalone coprocessors or software optimizations,
often avoiding the complexities of direct microarchitectural integration. This
study investigates the architectural challenges of embedding a SHA-3
permutation operation as a custom instruction within a general-purpose
processor, focusing on pipelined simultaneous execution, storage utilization,
and hardware cost. In this paper, we investigated and prototyped a SHA-3 custom
instruction for the RISC-V CPU architecture. Using cycle-accurate GEM5
simulations and FPGA prototyping, our results demonstrate performance
improvements of up to 8.02x for RISC-V optimized SHA-3 software workloads and
up to 46.31x for Keccak-specific software workloads, with only a 15.09%
increase in registers and a 11.51% increase in LUT utilization. These findings
provide critical insights into the feasibility and impact of SHA-3 acceleration
at the microarchitectural level, highlighting practical design considerations
for future cryptographic instruction set extensions.

### Computational Complexity

### 1. [QIP $ \subseteq $ AM(2QCFA)](http://arxiv.org/pdf/2508.21020v1)

Authors: Abuzer Yakaryılmaz

The class of languages having polynomial-time classical or quantum
interactive proof systems ($\mathsf{IP}$ or $\mathsf{QIP}$, respectively) is
identical to $\mathsf{PSPACE}$. We show that $\mathsf{PSPACE}$ (and so
$\mathsf{QIP}$) is subset of $\mathsf{AM(2QCFA)}$, the class of languages
having Arthur-Merlin proof systems where the verifiers are two-way finite
automata with quantum and classical states (2QCFAs) communicating with the
provers classically. Our protocols use only rational-valued quantum transitions
and run in double-exponential expected time. Moreover, the member strings are
accepted with probability 1 (i.e., perfect-completeness).

### 2. [Sharp Online Hardness for Large Balanced Independent Sets](http://arxiv.org/pdf/2508.20785v1)

Authors: Abhishek Dhawan, Eren C. Kızıldağ, Neeladri Maitra

We study the algorithmic problem of finding large $\gamma$-balanced
independent sets in dense random bipartite graphs; an independent set is
$\gamma$-balanced if a $\gamma$ proportion of its vertices lie on one side of
the bipartition. In the sparse regime, Perkins and Wang established tight
bounds within the low-degree polynomial (LDP) framework, showing a
factor-$1/(1-\gamma)$ statistical-computational gap via the Overlap Gap
Property (OGP) framework tailored for stable algorithms. However, these
techniques do not appear to extend to the dense setting. For the related large
independent set problem in dense random graph, the best known algorithm is an
online greedy procedure that is inherently unstable, and LDP algorithms are
conjectured to fail even in the "easy" regime where greedy succeeds. We show
that the largest $\gamma$-balanced independent set in dense random bipartite
graphs has size $\alpha:=\frac{\log_b n}{\gamma(1-\gamma)}$ whp, where $n$ is
the size of each bipartition, $p$ is the edge probability, and $b=1/(1-p)$. We
design an online algorithm that achieves $(1-\epsilon)(1-\gamma)\alpha$ whp for
any $\epsilon>0$. We complement this with a sharp lower bound, showing that no
online algorithm can achieve $(1+\epsilon)(1-\gamma)\alpha$ with nonnegligible
probability. Our results suggest that the same factor-$1/(1-\gamma)$ gap is
also present in the dense setting, supporting its conjectured universality.
While the classical greedy procedure on $G(n,p)$ is straightforward, our
algorithm is more intricate: it proceeds in two stages, incorporating a
stopping time and suitable truncation to ensure that $\gamma$-balancedness-a
global constraint-is met despite operating with limited information. Our lower
bound utilizes the OGP framework; we build on a recent refinement of this
framework for online models and extend it to the bipartite setting.

### Computational Engineering

### 1. [Mass conservation analysis of extrusion-based 3D printing simulations based on the level-set method](http://arxiv.org/pdf/2508.20617v1)

Authors: Carlos J. G. Rojas, C. A. Gómez-Pérez, Leyla Özkan

Numerical simulations of extrusion-based printing require tracking evolving
material bound- aries, a challenging task due to possible topological changes
and mass conservation issues. Inaccurate conservation of mass can lead to a
mismatch between the extruded and simulated shapes, and generally to unreliable
predictions of the actual ink behavior. This work investigates the mass
conservation properties of the conservative level-set method in extrusion-based
3D printing applications. We analyze the effects of the level set parameters on
the accuracy of mass conservation using the cross-sectional area of the
deposited strand. We compare the cross- sectional areas obtained in the
simulation with the ideal areas obtained from a mass balance when the system
reaches a steady-state condition. The numerical results indicate that reducing
the reinitialization and the interface thickness parameters decreases the
errors in the cross-sectional area obtained. However, the reductions in error
tend to decline and could lead to excessive computational cost. Furthermore, we
also found that the typical strong mesh requirements can be lessened by
selecting an adequate interface thickness. Finally, we obtained the
cross-sectional areas from simulations with different printing settings and
found that they show good agreement with the simulated and experimental data
published in previous work.

### 2. [Can News Predict the Direction of Oil Price Volatility? A Language Model Approach with SHAP Explanations](http://arxiv.org/pdf/2508.20707v1)

Authors: Romina Hashami, Felipe Maldonado

Financial markets can be highly sensitive to news, investor sentiment, and
economic indicators, leading to important asset price fluctuations. In this
study we focus on crude oil, due to its crucial role in commodity markets and
the global economy. Specifically, we are interested in understanding the
directional changes of oil price volatility, and for this purpose we
investigate whether news alone -- without incorporating traditional market data
-- can effectively predict the direction of oil price movements. Using a
decade-long dataset from Eikon (2014-2024), we develop an ensemble learning
framework to extract predictive signals from financial news. Our approach
leverages diverse sentiment analysis techniques and modern language models,
including FastText, FinBERT, Gemini, and LLaMA, to capture market sentiment and
textual patterns. We benchmark our model against the Heterogeneous
Autoregressive (HAR) model and assess statistical significance using the
McNemar test. While most sentiment-based indicators do not consistently
outperform HAR, the raw news count emerges as a robust predictor. Among
embedding techniques, FastText proves most effective for forecasting
directional movements. Furthermore, SHAP-based interpretation at the word level
reveals evolving predictive drivers across market regimes: pre-pandemic
emphasis on supply-demand and economic terms; early pandemic focus on
uncertainty and macroeconomic instability; post-shock attention to long-term
recovery indicators; and war-period sensitivity to geopolitical and regional
oil market disruptions. These findings highlight the predictive power of
news-driven features and the value of explainable NLP in financial forecasting.

### 3. [Self-consistent clustering analysis for homogenisation of heterogeneous plates](http://arxiv.org/pdf/2508.20446v1)

Authors: Menglei Li, Haolin Li, Bing Wang, Bing Wang

This work introduces a reduced-order model for plate structures with periodic
micro-structures by coupling self-consistent clustering analysis (SCA) with the
Lippmann-Schwinger equation, enabling rapid multiscale homogenisation of
heterogeneous plates. A plate-specific SCA scheme is derived for the first time
and features two key elements: (i) an offline-online strategy that combines
Green's functions with k-means data compression, and (ii) an online
self-consistent update that exploits the weak sensitivity of the reference
medium. The framework handles both linear and nonlinear problems in classical
plate theory and first-order shear deformation theory, and its performance is
verified on linear isotropic perforated plates and woven composites, as well as
on non-linear elasto-plastic perforated plates and woven composites with
damage. Across all cases the proposed model matches the accuracy of FFT-based
direct numerical simulation while reducing computational cost by over an order
of magnitude.

### 4. [The Epistemic Support-Point Filter (ESPF): A Bounded Possibilistic Framework for Ordinal State Estimation](http://arxiv.org/pdf/2508.20806v1)

Authors: Moriba Jah, Van Haslett

Traditional state estimation methods rely on probabilistic assumptions that
often collapse epistemic uncertainty into scalar beliefs, risking
overconfidence in sparse or adversarial sensing environments. We introduce the
Epistemic Support-Point Filter (ESPF), a novel non-Bayesian filtering framework
fully grounded in possibility theory and epistemic humility. ESPF redefines the
evolution of belief over state space using compatibility-weighted support
updates, surprisalaware pruning, and adaptive dispersion via sparse grid
quadrature. Unlike conventional filters, ESPF does not seek a posterior
distribution, but rather maintains a structured region of plausibility or
non-rejection, updated using ordinal logic rather than integration. For
multi-model inference, we employ the Choquet integral to fuse competing
hypotheses based on a dynamic epistemic capacity function, generalizing
classical winner-take-all strategies. The result is an inference engine capable
of dynamically contracting or expanding belief support in direct response to
information structure, without requiring prior statistical calibration. This
work presents a foundational shift in how inference, evidence, and ignorance
are reconciled, supporting robust estimation where priors are unavailable,
misleading, or epistemically unjustified.

### Computational Geometry

### 1. [Entropy-Bounded Computational Geometry Made Easier and Sensitive to Sortedness](http://arxiv.org/pdf/2508.20489v1)

Authors: David Eppstein, Michael T. Goodrich, Abraham M. Illickan, Claire A. To

We study entropy-bounded computational geometry, that is, geometric
algorithms whose running times depend on a given measure of the input entropy.
Specifically, we introduce a measure that we call range-partition entropy,
which unifies and subsumes previous definitions of entropy used for sorting
problems and structural entropy used in computational geometry. We provide
simple algorithms for several problems, including 2D maxima, 2D and 3D convex
hulls, and some visibility problems, and we show that they have running times
depending on the range-partition entropy.

### Computation and Language

### 1. [Joint Enhancement of Relational Reasoning for Long-Context LLMs](http://arxiv.org/pdf/2508.20351v1)

Authors: Zhirui Chen, Wei Shen, Jiashui Huang, Ling Shao

Despite significant progress, large language models (LLMs) still struggle
with long contexts due to memory limitations and their inability to tackle
complex and long-context tasks. Additionally, LLMs often suffer from a lack of
transparency and are prone to producing hallucinations. To address these
challenges, we propose \textbf{JERR}, a novel framework designed to enhance
long-context comprehension via graph-based reasoning in LLMs. JERR integrates
three key components: synopsis extraction, graph construction, and relational
reasoning. First, synopsis is extracted by chunking text strategically,
allowing the model to summarize and understand information more efficiently.
Second, we build a directed acyclic graph (DAG) to resolve redundancy, ensuring
logical consistency and clarity. Finally, we incorporate Monte Carlo Tree
Search (MCTS) to help the model navigate complex reasoning paths, ensuring more
accurate and interpretable outputs. This framework provides a novel solution
that enables LLMs to handle extended contexts and complex reasoning tasks with
improved reliability and transparency. Experimental results show that JERR
consistently outperforms all baselines on the ROUGE and F1 metrics, achieving
the highest scores on the LLM-Rater evaluation.

### 2. [CAPE: Context-Aware Personality Evaluation Framework for Large Language Models](http://arxiv.org/pdf/2508.20385v1)

Authors: Jivnesh Sandhan, Fei Cheng, Tushar Sandhan, Yugo Murawaki

Psychometric tests, traditionally used to assess humans, are now being
applied to Large Language Models (LLMs) to evaluate their behavioral traits.
However, existing studies follow a context-free approach, answering each
question in isolation to avoid contextual influence. We term this the Disney
World test, an artificial setting that ignores real-world applications, where
conversational history shapes responses. To bridge this gap, we propose the
first Context-Aware Personality Evaluation (CAPE) framework for LLMs,
incorporating prior conversational interactions. To thoroughly analyze the
influence of context, we introduce novel metrics to quantify the consistency of
LLM responses, a fundamental trait in human behavior.
  Our exhaustive experiments on 7 LLMs reveal that conversational history
enhances response consistency via in-context learning but also induces
personality shifts, with GPT-3.5-Turbo and GPT-4-Turbo exhibiting extreme
deviations. While GPT models are robust to question ordering, Gemini-1.5-Flash
and Llama-8B display significant sensitivity. Moreover, GPT models response
stem from their intrinsic personality traits as well as prior interactions,
whereas Gemini-1.5-Flash and Llama--8B heavily depend on prior interactions.
Finally, applying our framework to Role Playing Agents (RPAs) shows
context-dependent personality shifts improve response consistency and better
align with human judgments. Our code and datasets are publicly available at:
https://github.com/jivnesh/CAPE

### 3. [UI-Bench: A Benchmark for Evaluating Design Capabilities of AI Text-to-App Tools](http://arxiv.org/pdf/2508.20410v1)

Authors: Sam Jung, Agustin Garcinuno, Spencer Mateega

AI text-to-app tools promise high quality applications and websites in
minutes, yet no public benchmark rigorously verifies those claims. We introduce
UI-Bench, the first large-scale benchmark that evaluates visual excellence
across competing AI text-to-app tools through expert pairwise comparison.
Spanning 10 tools, 30 prompts, 300 generated sites, and \textit{4000+} expert
judgments, UI-Bench ranks systems with a TrueSkill-derived model that yields
calibrated confidence intervals. UI-Bench establishes a reproducible standard
for advancing AI-driven web design. We release (i) the complete prompt set,
(ii) an open-source evaluation framework, and (iii) a public leaderboard. The
generated sites rated by participants will be released soon. View the UI-Bench
leaderboard at https://uibench.ai/leaderboard.

### 4. [CAMB: A comprehensive industrial LLM benchmark on civil aviation maintenance](http://arxiv.org/pdf/2508.20420v1)

Authors: Feng Zhang, Chengjie Pang, Yuehan Zhang, Chenyu Luo

Civil aviation maintenance is a domain characterized by stringent industry
standards. Within this field, maintenance procedures and troubleshooting
represent critical, knowledge-intensive tasks that require sophisticated
reasoning. To address the lack of specialized evaluation tools for large
language models (LLMs) in this vertical, we propose and develop an
industrial-grade benchmark specifically designed for civil aviation
maintenance. This benchmark serves a dual purpose: It provides a standardized
tool to measure LLM capabilities within civil aviation maintenance, identifying
specific gaps in domain knowledge and complex reasoning. By pinpointing these
deficiencies, the benchmark establishes a foundation for targeted improvement
efforts (e.g., domain-specific fine-tuning, RAG optimization, or specialized
prompt engineering), ultimately facilitating progress toward more intelligent
solutions within civil aviation maintenance. Our work addresses a significant
gap in the current LLM evaluation, which primarily focuses on mathematical and
coding reasoning tasks. In addition, given that Retrieval-Augmented Generation
(RAG) systems are currently the dominant solutions in practical applications ,
we leverage this benchmark to evaluate existing well-known vector embedding
models and LLMs for civil aviation maintenance scenarios. Through experimental
exploration and analysis, we demonstrate the effectiveness of our benchmark in
assessing model performance within this domain, and we open-source this
evaluation benchmark and code to foster further research and
development:https://github.com/CamBenchmark/cambenchmark

### 5. [Searching the Title of Practical Work of the Informatics Engineering Bachelor Program with the Case Base Reasoning Method](http://arxiv.org/pdf/2508.20442v1)

Authors: Agung Sukrisna Jaya, Osvari Arsalan, Danny Matthew Saputra

Case Base Reasoning (CBR) is a case solving technique based on experience in
cases that have occurred before with the highest similarity. CBR is used to
search for practical work titles. TF-IDF is applied to process the
vectorization of each practical work title word and Cosine Similarity for the
calculation of similarity values. This system can search either in the form of
titles or keywords. The output of the system is the title of practical work and
the match value of each title. Based on the test results using 705 practical
work titles, testing was carried out with five titles and carried out in two
stages. The first stage searches with existing titles and the second stage
randomizes the title from the first stage. And the results obtained in the
second stage are the same number of titles found and the highest average match
score.

### 6. [MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers](http://arxiv.org/pdf/2508.20453v1)

Authors: Zhenting Wang, Qi Chang, Hemani Patel, Shashank Biju, Cheng-En Wu, Quan Liu, Aolin Ding, Alireza Rezazadeh, Ankit Shah, Yujia Bao, Eugene Siow

We introduce MCP-Bench, a benchmark for evaluating large language models
(LLMs) on realistic, multi-step tasks that demand tool use, cross-tool
coordination, precise parameter control, and planning/reasoning for solving
tasks. Built on the Model Context Protocol (MCP), MCP-Bench connects LLMs to 28
representative live MCP servers spanning 250 tools across domains such as
finance, traveling, scientific computing, and academic search. Unlike prior
API-based benchmarks, each MCP server provides a set of complementary tools
designed to work together, enabling the construction of authentic, multi-step
tasks with rich input-output coupling. Tasks in MCP-Bench test agents' ability
to retrieve relevant tools from fuzzy instructions without explicit tool names,
plan multi-hop execution trajectories for complex objectives, ground responses
in intermediate tool outputs, and orchestrate cross-domain workflows -
capabilities not adequately evaluated by existing benchmarks that rely on
explicit tool specifications, shallow few-step workflows, and isolated domain
operations. We propose a multi-faceted evaluation framework covering tool-level
schema understanding and usage, trajectory-level planning, and task completion.
Experiments on 20 advanced LLMs reveal persistent challenges in MCP-Bench. Code
and data: https://github.com/Accenture/mcp-bench.

### 7. [Prediction of mortality and resource utilization in critical care: a deep learning approach using multimodal electronic health records with natural language processing techniques](http://arxiv.org/pdf/2508.20460v1)

Authors: Yucheng Ruan, Xiang Lan, Daniel J. Tan, Hairil Rizal Abdullah, Mengling Feng

Background Predicting mortality and resource utilization from electronic
health records (EHRs) is challenging yet crucial for optimizing patient
outcomes and managing costs in intensive care unit (ICU). Existing approaches
predominantly focus on structured EHRs, often ignoring the valuable clinical
insights in free-text notes. Additionally, the potential of textual information
within structured data is not fully leveraged. This study aimed to introduce
and assess a deep learning framework using natural language processing
techniques that integrates multimodal EHRs to predict mortality and resource
utilization in critical care settings. Methods Utilizing two real-world EHR
datasets, we developed and evaluated our model on three clinical tasks with
leading existing methods. We also performed an ablation study on three key
components in our framework: medical prompts, free-texts, and pre-trained
sentence encoder. Furthermore, we assessed the model's robustness against the
corruption in structured EHRs. Results Our experiments on two real-world
datasets across three clinical tasks showed that our proposed model improved
performance metrics by 1.6\%/0.8\% on BACC/AUROC for mortality prediction,
0.5%/2.2% on RMSE/MAE for LOS prediction, 10.9%/11.0% on RMSE/MAE for surgical
duration estimation compared to the best existing methods. It consistently
demonstrated superior performance compared to other baselines across three
tasks at different corruption rates. Conclusions The proposed framework is an
effective and accurate deep learning approach for predicting mortality and
resource utilization in critical care. The study also highlights the success of
using prompt learning with a transformer encoder in analyzing multimodal EHRs.
Importantly, the model showed strong resilience to data corruption within
structured data, especially at high corruption levels.

### 8. [ConspirED: A Dataset for Cognitive Traits of Conspiracy Theories and Large Language Model Safety](http://arxiv.org/pdf/2508.20468v1)

Authors: Luke Bates, Max Glockner, Preslav Nakov, Iryna Gurevych

Conspiracy theories erode public trust in science and institutions while
resisting debunking by evolving and absorbing counter-evidence. As AI-generated
misinformation becomes increasingly sophisticated, understanding rhetorical
patterns in conspiratorial content is important for developing interventions
such as targeted prebunking and assessing AI vulnerabilities. We introduce
ConspirED (CONSPIR Evaluation Dataset), which captures the cognitive traits of
conspiratorial ideation in multi-sentence excerpts (80--120 words) from online
conspiracy articles, annotated using the CONSPIR cognitive framework
(Lewandowsky and Cook, 2020). ConspirED is the first dataset of conspiratorial
content annotated for general cognitive traits. Using ConspirED, we (i) develop
computational models that identify conspiratorial traits and determine dominant
traits in text excerpts, and (ii) evaluate large language/reasoning model
(LLM/LRM) robustness to conspiratorial inputs. We find that both are misaligned
by conspiratorial content, producing output that mirrors input reasoning
patterns, even when successfully deflecting comparable fact-checked
misinformation.

### 9. [SciTopic: Enhancing Topic Discovery in Scientific Literature through Advanced LLM](http://arxiv.org/pdf/2508.20514v1)

Authors: Pengjiang Li, Zaitian Wang, Xinhao Zhang, Ran Zhang, Lu Jiang, Pengfei Wang, Yuanchun Zhou

Topic discovery in scientific literature provides valuable insights for
researchers to identify emerging trends and explore new avenues for
investigation, facilitating easier scientific information retrieval. Many
machine learning methods, particularly deep embedding techniques, have been
applied to discover research topics. However, most existing topic discovery
methods rely on word embedding to capture the semantics and lack a
comprehensive understanding of scientific publications, struggling with
complex, high-dimensional text relationships. Inspired by the exceptional
comprehension of textual information by large language models (LLMs), we
propose an advanced topic discovery method enhanced by LLMs to improve
scientific topic identification, namely SciTopic. Specifically, we first build
a textual encoder to capture the content from scientific publications,
including metadata, title, and abstract. Next, we construct a space
optimization module that integrates entropy-based sampling and triplet tasks
guided by LLMs, enhancing the focus on thematic relevance and contextual
intricacies between ambiguous instances. Then, we propose to fine-tune the
textual encoder based on the guidance from the LLMs by optimizing the
contrastive loss of the triplets, forcing the text encoder to better
discriminate instances of different topics. Finally, extensive experiments
conducted on three real-world datasets of scientific publications demonstrate
that SciTopic outperforms the state-of-the-art (SOTA) scientific topic
discovery methods, enabling researchers to gain deeper and faster insights.

### 10. [KCS: Diversify Multi-hop Question Generation with Knowledge Composition Sampling](http://arxiv.org/pdf/2508.20567v1)

Authors: Yangfan Wang, Jie Liu, Chen Tang, Lian Yan, Jingchi Jiang

Multi-hop question answering faces substantial challenges due to data
sparsity, which increases the likelihood of language models learning spurious
patterns. To address this issue, prior research has focused on diversifying
question generation through content planning and varied expression. However,
these approaches often emphasize generating simple questions and neglect the
integration of essential knowledge, such as relevant sentences within
documents. This paper introduces the Knowledge Composition Sampling (KCS), an
innovative framework designed to expand the diversity of generated multi-hop
questions by sampling varied knowledge compositions within a given context. KCS
models the knowledge composition selection as a sentence-level conditional
prediction task and utilizes a probabilistic contrastive loss to predict the
next most relevant piece of knowledge. During inference, we employ a stochastic
decoding strategy to effectively balance accuracy and diversity. Compared to
competitive baselines, our KCS improves the overall accuracy of knowledge
composition selection by 3.9%, and its application for data augmentation yields
improvements on HotpotQA and 2WikiMultihopQA datasets. Our code is available
at: https://github.com/yangfanww/kcs.

### Cryptography and Security

### 1. [MindGuard: Tracking, Detecting, and Attributing MCP Tool Poisoning Attack via Decision Dependence Graph](http://arxiv.org/pdf/2508.20412v1)

Authors: Zhiqiang Wang, Junyang Zhang, Guanquan Shi, HaoRan Cheng, Yunhao Yao, Kaiwen Guo, Haohua Du, Xiang-Yang Li

The Model Context Protocol (MCP) is increasingly adopted to standardize the
interaction between LLM agents and external tools. However, this trend
introduces a new threat: Tool Poisoning Attacks (TPA), where tool metadata is
poisoned to induce the agent to perform unauthorized operations. Existing
defenses that primarily focus on behavior-level analysis are fundamentally
ineffective against TPA, as poisoned tools need not be executed, leaving no
behavioral trace to monitor.
  Thus, we propose MindGuard, a decision-level guardrail for LLM agents,
providing provenance tracking of call decisions, policy-agnostic detection, and
poisoning source attribution against TPA. While fully explaining LLM decision
remains challenging, our empirical findings uncover a strong correlation
between LLM attention mechanisms and tool invocation decisions. Therefore, we
choose attention as an empirical signal for decision tracking and formalize
this as the Decision Dependence Graph (DDG), which models the LLM's reasoning
process as a weighted, directed graph where vertices represent logical concepts
and edges quantify the attention-based dependencies. We further design robust
DDG construction and graph-based anomaly analysis mechanisms that efficiently
detect and attribute TPA attacks. Extensive experiments on real-world datasets
demonstrate that MindGuard achieves 94\%-99\% average precision in detecting
poisoned invocations, 95\%-100\% attribution accuracy, with processing times
under one second and no additional token cost. Moreover, DDG can be viewed as
an adaptation of the classical Program Dependence Graph (PDG), providing a
solid foundation for applying traditional security policies at the decision
level.

### 2. [Breaking Diffusion with Cache: Exploiting Approximate Caches in Diffusion Models](http://arxiv.org/pdf/2508.20424v1)

Authors: Desen Sun, Shuncheng Jie, Sihang Liu

Diffusion models are a powerful class of generative models that produce
content, such as images, from user prompts, but they are computationally
intensive. To mitigate this cost, recent academic and industry work has adopted
approximate caching, which reuses intermediate states from similar prompts in a
cache. While efficient, this optimization introduces new security risks by
breaking isolation among users. This work aims to comprehensively assess new
security vulnerabilities arising from approximate caching. First, we
demonstrate a remote covert channel established with the cache, where a sender
injects prompts with special keywords into the cache and a receiver can recover
that even after days, to exchange information. Second, we introduce a prompt
stealing attack using the cache, where an attacker can recover existing cached
prompts based on cache hit prompts. Finally, we introduce a poisoning attack
that embeds the attacker's logos into the previously stolen prompt, to render
them in future user prompts that hit the cache. These attacks are all performed
remotely through the serving system, which indicates severe security
vulnerabilities in approximate caching.

### 3. [Ransomware 3.0: Self-Composing and LLM-Orchestrated](http://arxiv.org/pdf/2508.20444v1)

Authors: Md Raz, Meet Udeshi, P. V. Sai Charan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

Using automated reasoning, code synthesis, and contextual decision-making, we
introduce a new threat that exploits large language models (LLMs) to
autonomously plan, adapt, and execute the ransomware attack lifecycle.
Ransomware 3.0 represents the first threat model and research prototype of
LLM-orchestrated ransomware. Unlike conventional malware, the prototype only
requires natural language prompts embedded in the binary; malicious code is
synthesized dynamically by the LLM at runtime, yielding polymorphic variants
that adapt to the execution environment. The system performs reconnaissance,
payload generation, and personalized extortion, in a closed-loop attack
campaign without human involvement. We evaluate this threat across personal,
enterprise, and embedded environments using a phase-centric methodology that
measures quantitative fidelity and qualitative coherence in each attack phase.
We show that open source LLMs can generate functional ransomware components and
sustain closed-loop execution across diverse environments. Finally, we present
behavioral signals and multi-level telemetry of Ransomware 3.0 through a case
study to motivate future development of better defenses and policy enforcements
to address novel AI-enabled ransomware attacks.

### 4. [Bitcoin as an Interplanetary Monetary Standard with Proof-of-Transit Timestamping](http://arxiv.org/pdf/2508.20591v1)

Authors: Jose E. Puente, Carlos Puente

We explore the feasibility of deploying Bitcoin as the shared monetary
standard between Earth and Mars, accounting for physical constraints of
interplanetary communication. We introduce a novel primitive, Proof-of-Transit
Timestamping (PoTT), to provide cryptographic, tamper-evident audit trails for
Bitcoin data across high-latency, intermittently-connected links. Leveraging
Delay/Disruption-Tolerant Networking (DTN) and optical low-Earth-orbit (LEO)
mesh constellations, we propose an architecture for header-first replication,
long-horizon Lightning channels with planetary watchtowers, and secure
settlement through federated sidechains or blind-merge-mined (BMM) commit
chains. We formalize PoTT, analyze its security model, and show how it
measurably improves reliability and accountability without altering Bitcoin
consensus or its monetary base. Near-term deployments favor strong federations
for local settlement; longer-term, blind-merge-mined commit chains (if adopted)
provide an alternative. The Earth L1 monetary base remains unchanged, while
Mars can operate a pegged commit chain or strong federation with 1:1 pegged
assets for local block production. For transparency, if both time-beacon
regimes are simultaneously compromised, PoTT-M2 (and PoTT generally) reduces to
administrative assertions rather than cryptographic time-anchoring.

### 5. [CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics](http://arxiv.org/pdf/2508.20643v1)

Authors: Stefano Fumero, Kai Huang, Matteo Boffa, Danilo Giordano, Marco Mellia, Zied Ben Houidi, Dario Rossi

Large Language Model (LLM) agents are powerful tools for automating complex
tasks. In cybersecurity, researchers have primarily explored their use in
red-team operations such as vulnerability discovery and penetration tests.
Defensive uses for incident response and forensics have received comparatively
less attention and remain at an early stage. This work presents a systematic
study of LLM-agent design for the forensic investigation of realistic web
application attacks. We propose CyberSleuth, an autonomous agent that processes
packet-level traces and application logs to identify the targeted service, the
exploited vulnerability (CVE), and attack success. We evaluate the consequences
of core design decisions - spanning tool integration and agent architecture -
and provide interpretable guidance for practitioners. We benchmark four agent
architectures and six LLM backends on 20 incident scenarios of increasing
complexity, identifying CyberSleuth as the best-performing design. In a
separate set of 10 incidents from 2025, CyberSleuth correctly identifies the
exact CVE in 80% of cases. At last, we conduct a human study with 22 experts,
which rated the reports of CyberSleuth as complete, useful, and coherent. They
also expressed a slight preference for DeepSeek R1, a good news for open source
LLM. To foster progress in defensive LLM research, we release both our
benchmark and the CyberSleuth platform as a foundation for fair, reproducible
evaluation of forensic agents.

### 6. [Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review](http://arxiv.org/pdf/2508.20863v1)

Authors: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

Large Language Models (LLMs) are increasingly being integrated into the
scientific peer-review process, raising new questions about their reliability
and resilience to manipulation. In this work, we investigate the potential for
hidden prompt injection attacks, where authors embed adversarial text within a
paper's PDF to influence the LLM-generated review. We begin by formalising
three distinct threat models that envision attackers with different motivations
-- not all of which implying malicious intent. For each threat model, we design
adversarial prompts that remain invisible to human readers yet can steer an
LLM's output toward the author's desired outcome. Using a user study with
domain scholars, we derive four representative reviewing prompts used to elicit
peer reviews from LLMs. We then evaluate the robustness of our adversarial
prompts across (i) different reviewing prompts, (ii) different commercial
LLM-based systems, and (iii) different peer-reviewed papers. Our results show
that adversarial prompts can reliably mislead the LLM, sometimes in ways that
adversely affect a "honest-but-lazy" reviewer. Finally, we propose and
empirically assess methods to reduce detectability of adversarial prompts under
automated content checks.

### 7. [PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance](http://arxiv.org/pdf/2508.20890v1)

Authors: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

Large Language Models (LLMs) are increasingly integrated into real-world
applications, from virtual assistants to autonomous agents. However, their
flexibility also introduces new attack vectors-particularly Prompt Injection
(PI), where adversaries manipulate model behavior through crafted inputs. As
attackers continuously evolve with paraphrased, obfuscated, and even multi-task
injection strategies, existing benchmarks are no longer sufficient to capture
the full spectrum of emerging threats.
  To address this gap, we construct a new benchmark that systematically extends
prior efforts. Our benchmark subsumes the two widely-used existing ones while
introducing new manipulation techniques and multi-task scenarios, thereby
providing a more comprehensive evaluation setting. We find that existing
defenses, though effective on their original benchmarks, show clear weaknesses
under our benchmark, underscoring the need for more robust solutions. Our key
insight is that while attack forms may vary, the adversary's intent-injecting
an unauthorized task-remains invariant. Building on this observation, we
propose PromptSleuth, a semantic-oriented defense framework that detects prompt
injection by reasoning over task-level intent rather than surface features.
Evaluated across state-of-the-art benchmarks, PromptSleuth consistently
outperforms existing defense while maintaining comparable runtime and cost
efficiency. These results demonstrate that intent-based semantic reasoning
offers a robust, efficient, and generalizable strategy for defending LLMs
against evolving prompt injection threats.

### 8. [Federated Learning for Large Models in Medical Imaging: A Comprehensive Review](http://arxiv.org/pdf/2508.20414v1)

Authors: Mengyu Sun, Ziyuan Yang, Yongqiang Huang, Hui Yu, Yingyu Chen, Shuren Qi, Andrew Beng Jin Teoh, Yi Zhang

Artificial intelligence (AI) has demonstrated considerable potential in the
realm of medical imaging. However, the development of high-performance AI
models typically necessitates training on large-scale, centralized datasets.
This approach is confronted with significant challenges due to strict patient
privacy regulations and legal restrictions on data sharing and utilization.
These limitations hinder the development of large-scale models in medical
domains and impede continuous updates and training with new data. Federated
Learning (FL), a privacy-preserving distributed training framework, offers a
new solution by enabling collaborative model development across fragmented
medical datasets. In this survey, we review FL's contributions at two stages of
the full-stack medical analysis pipeline. First, in upstream tasks such as CT
or MRI reconstruction, FL enables joint training of robust reconstruction
networks on diverse, multi-institutional datasets, alleviating data scarcity
while preserving confidentiality. Second, in downstream clinical tasks like
tumor diagnosis and segmentation, FL supports continuous model updating by
allowing local fine-tuning on new data without centralizing sensitive images.
We comprehensively analyze FL implementations across the medical imaging
pipeline, from physics-informed reconstruction networks to diagnostic AI
systems, highlighting innovations that improve communication efficiency, align
heterogeneous data, and ensure secure parameter aggregation. Meanwhile, this
paper provides an outlook on future research directions, aiming to serve as a
valuable reference for the field's development.

### 9. [Enhancing Resilience for IoE: A Perspective of Networking-Level Safeguard](http://arxiv.org/pdf/2508.20504v1)

Authors: Guan-Yan Yang, Jui-Ning Chen, Farn Wang, Kuo-Hui Yeh

The Internet of Energy (IoE) integrates IoT-driven digital communication with
power grids to enable efficient and sustainable energy systems. Still, its
interconnectivity exposes critical infrastructure to sophisticated cyber
threats, including adversarial attacks designed to bypass traditional
safeguards. Unlike general IoT risks, IoE threats have heightened public safety
consequences, demanding resilient solutions. From the networking-level
safeguard perspective, we propose a Graph Structure Learning (GSL)-based
safeguards framework that jointly optimizes graph topology and node
representations to resist adversarial network model manipulation inherently.
Through a conceptual overview, architectural discussion, and case study on a
security dataset, we demonstrate GSL's superior robustness over representative
methods, offering practitioners a viable path to secure IoE networks against
evolving attacks. This work highlights the potential of GSL to enhance the
resilience and reliability of future IoE networks for practitioners managing
critical infrastructure. Lastly, we identify key open challenges and propose
future research directions in this novel research area.

### 10. [BridgeShield: Enhancing Security for Cross-chain Bridge Applications via Heterogeneous Graph Mining](http://arxiv.org/pdf/2508.20517v1)

Authors: Dan Lin, Shunfeng Lu, Ziyan Liu, Jiajing Wu, Junyuan Fang, Kaixin Lin, Bowen Song, Zibin Zheng

Cross-chain bridges play a vital role in enabling blockchain
interoperability. However, due to the inherent design flaws and the enormous
value they hold, they have become prime targets for hacker attacks. Existing
detection methods show progress yet remain limited, as they mainly address
single-chain behaviors and fail to capture cross-chain semantics. To address
this gap, we leverage heterogeneous graph attention networks, which are
well-suited for modeling multi-typed entities and relations, to capture the
complex execution semantics of cross-chain behaviors. We propose BridgeShield,
a detection framework that jointly models the source chain, off-chain
coordination, and destination chain within a unified heterogeneous graph
representation. BridgeShield incorporates intra-meta-path attention to learn
fine-grained dependencies within cross-chain paths and inter-meta-path
attention to highlight discriminative cross-chain patterns, thereby enabling
precise identification of attack behaviors. Extensive experiments on 51
real-world cross-chain attack events demonstrate that BridgeShield achieves an
average F1-score of 92.58%, representing a 24.39% improvement over
state-of-the-art baselines. These results validate the effectiveness of
BridgeShield as a practical solution for securing cross-chain bridges and
enhancing the resilience of multi-chain ecosystems.

### Computer Vision and Pattern Recognition

### 1. [Enhancing Mamba Decoder with Bidirectional Interaction in Multi-Task Dense Prediction](http://arxiv.org/pdf/2508.20376v1)

Authors: Mang Cao, Sanping Zhou, Yizhe Li, Ye Deng, Wenli Huang, Le Wang

Sufficient cross-task interaction is crucial for success in multi-task dense
prediction. However, sufficient interaction often results in high computational
complexity, forcing existing methods to face the trade-off between interaction
completeness and computational efficiency. To address this limitation, this
work proposes a Bidirectional Interaction Mamba (BIM), which incorporates novel
scanning mechanisms to adapt the Mamba modeling approach for multi-task dense
prediction. On the one hand, we introduce a novel Bidirectional Interaction
Scan (BI-Scan) mechanism, which constructs task-specific representations as
bidirectional sequences during interaction. By integrating task-first and
position-first scanning modes within a unified linear complexity architecture,
BI-Scan efficiently preserves critical cross-task information. On the other
hand, we employ a Multi-Scale Scan~(MS-Scan) mechanism to achieve
multi-granularity scene modeling. This design not only meets the diverse
granularity requirements of various tasks but also enhances nuanced cross-task
feature interactions. Extensive experiments on two challenging benchmarks,
\emph{i.e.}, NYUD-V2 and PASCAL-Context, show the superiority of our BIM vs its
state-of-the-art competitors.

### 2. [Audio-Guided Visual Editing with Complex Multi-Modal Prompts](http://arxiv.org/pdf/2508.20379v1)

Authors: Hyeonyu Kim, Seokhoon Jeong, Seonghee Han, Chanhyuk Choi, Taehwan Kim

Visual editing with diffusion models has made significant progress but often
struggles with complex scenarios that textual guidance alone could not
adequately describe, highlighting the need for additional non-text editing
prompts. In this work, we introduce a novel audio-guided visual editing
framework that can handle complex editing tasks with multiple text and audio
prompts without requiring additional training. Existing audio-guided visual
editing methods often necessitate training on specific datasets to align audio
with text, limiting their generalization to real-world situations. We leverage
a pre-trained multi-modal encoder with strong zero-shot capabilities and
integrate diverse audio into visual editing tasks, by alleviating the
discrepancy between the audio encoder space and the diffusion model's prompt
encoder space. Additionally, we propose a novel approach to handle complex
scenarios with multiple and multi-modal editing prompts through our separate
noise branching and adaptive patch selection. Our comprehensive experiments on
diverse editing tasks demonstrate that our framework excels in handling
complicated editing scenarios by incorporating rich information from audio,
where text-only approaches fail.

### 3. [More Reliable Pseudo-labels, Better Performance: A Generalized Approach to Single Positive Multi-label Learning](http://arxiv.org/pdf/2508.20381v1)

Authors: Luong Tran, Thieu Vo, Anh Nguyen, Sang Dinh, Van Nguyen

Multi-label learning is a challenging computer vision task that requires
assigning multiple categories to each image. However, fully annotating
large-scale datasets is often impractical due to high costs and effort,
motivating the study of learning from partially annotated data. In the extreme
case of Single Positive Multi-Label Learning (SPML), each image is provided
with only one positive label, while all other labels remain unannotated.
Traditional SPML methods that treat missing labels as unknown or negative tend
to yield inaccuracies and false negatives, and integrating various
pseudo-labeling strategies can introduce additional noise. To address these
challenges, we propose the Generalized Pseudo-Label Robust Loss (GPR Loss), a
novel loss function that effectively learns from diverse pseudo-labels while
mitigating noise. Complementing this, we introduce a simple yet effective
Dynamic Augmented Multi-focus Pseudo-labeling (DAMP) technique. Together, these
contributions form the Adaptive and Efficient Vision-Language Pseudo-Labeling
(AEVLP) framework. Extensive experiments on four benchmark datasets demonstrate
that our framework significantly advances multi-label classification, achieving
state-of-the-art results.

### 4. [Graph-Based Uncertainty Modeling and Multimodal Fusion for Salient Object Detection](http://arxiv.org/pdf/2508.20415v1)

Authors: Yuqi Xiong, Wuzhen Shi, Yang Wen, Ruhan Liu

In view of the problems that existing salient object detection (SOD) methods
are prone to losing details, blurring edges, and insufficient fusion of
single-modal information in complex scenes, this paper proposes a dynamic
uncertainty propagation and multimodal collaborative reasoning network
(DUP-MCRNet). Firstly, a dynamic uncertainty graph convolution module (DUGC) is
designed to propagate uncertainty between layers through a sparse graph
constructed based on spatial semantic distance, and combined with channel
adaptive interaction, it effectively improves the detection accuracy of small
structures and edge regions. Secondly, a multimodal collaborative fusion
strategy (MCF) is proposed, which uses learnable modality gating weights to
weightedly fuse the attention maps of RGB, depth, and edge features. It can
dynamically adjust the importance of each modality according to different
scenes, effectively suppress redundant or interfering information, and
strengthen the semantic complementarity and consistency between
cross-modalities, thereby improving the ability to identify salient regions
under occlusion, weak texture or background interference. Finally, the
detection performance at the pixel level and region level is optimized through
multi-scale BCE and IoU loss, cross-scale consistency constraints, and
uncertainty-guided supervision mechanisms. Extensive experiments show that
DUP-MCRNet outperforms various SOD methods on most common benchmark datasets,
especially in terms of edge clarity and robustness to complex backgrounds. Our
code is publicly available at https://github.com/YukiBear426/DUP-MCRNet.

### 5. [MSMVD: Exploiting Multi-scale Image Features via Multi-scale BEV Features for Multi-view Pedestrian Detection](http://arxiv.org/pdf/2508.20447v1)

Authors: Taiga Yamane, Satoshi Suzuki, Ryo Masumura, Shota Orihashi, Tomohiro Tanaka, Mana Ihori, Naoki Makishima, Naotaka Kawata

Multi-View Pedestrian Detection (MVPD) aims to detect pedestrians in the form
of a bird's eye view (BEV) from multi-view images. In MVPD, end-to-end
trainable deep learning methods have progressed greatly. However, they often
struggle to detect pedestrians with consistently small or large scales in views
or with vastly different scales between views. This is because they do not
exploit multi-scale image features to generate the BEV feature and detect
pedestrians. To overcome this problem, we propose a novel MVPD method, called
Multi-Scale Multi-View Detection (MSMVD). MSMVD generates multi-scale BEV
features by projecting multi-scale image features extracted from individual
views into the BEV space, scale-by-scale. Each of these BEV features inherits
the properties of its corresponding scale image features from multiple views.
Therefore, these BEV features help the precise detection of pedestrians with
consistently small or large scales in views. Then, MSMVD combines information
at different scales of multiple views by processing the multi-scale BEV
features using a feature pyramid network. This improves the detection of
pedestrians with vastly different scales between views. Extensive experiments
demonstrate that exploiting multi-scale image features via multi-scale BEV
features greatly improves the detection performance, and MSMVD outperforms the
previous highest MODA by $4.5$ points on the GMVD dataset.

### 6. [A Spatial-Frequency Aware Multi-Scale Fusion Network for Real-Time Deepfake Detection](http://arxiv.org/pdf/2508.20449v1)

Authors: Libo Lv, Tianyi Wang, Mengxiao Huang, Ruixia Liu, Yinglong Wang

With the rapid advancement of real-time deepfake generation techniques,
forged content is becoming increasingly realistic and widespread across
applications like video conferencing and social media. Although
state-of-the-art detectors achieve high accuracy on standard benchmarks, their
heavy computational cost hinders real-time deployment in practical
applications. To address this, we propose the Spatial-Frequency Aware
Multi-Scale Fusion Network (SFMFNet), a lightweight yet effective architecture
for real-time deepfake detection. We design a spatial-frequency hybrid aware
module that jointly leverages spatial textures and frequency artifacts through
a gated mechanism, enhancing sensitivity to subtle manipulations. A
token-selective cross attention mechanism enables efficient multi-level feature
interaction, while a residual-enhanced blur pooling structure helps retain key
semantic cues during downsampling. Experiments on several benchmark datasets
show that SFMFNet achieves a favorable balance between accuracy and efficiency,
with strong generalization and practical value for real-time applications.

### 7. [Re-Densification Meets Cross-Scale Propagation: Real-Time Compression of LiDAR Point Clouds](http://arxiv.org/pdf/2508.20466v1)

Authors: Pengpeng Yu, Haoran Li, Dingquan Li, Runqing Jiang, Jing Wang, Liang Lin, Yulan Guo

LiDAR point clouds are fundamental to various applications, yet
high-precision scans incur substantial storage and transmission overhead.
Existing methods typically convert unordered points into hierarchical octree or
voxel structures for dense-to-sparse predictive coding. However, the extreme
sparsity of geometric details hinders efficient context modeling, thereby
limiting their compression performance and speed. To address this challenge, we
propose to generate compact features for efficient predictive coding. Our
framework comprises two lightweight modules. First, the Geometry
Re-Densification Module re-densifies encoded sparse geometry, extracts features
at denser scale, and then re-sparsifies the features for predictive coding.
This module avoids costly computation on highly sparse details while
maintaining a lightweight prediction head. Second, the Cross-scale Feature
Propagation Module leverages occupancy cues from multiple resolution levels to
guide hierarchical feature propagation. This design facilitates information
sharing across scales, thereby reducing redundant feature extraction and
providing enriched features for the Geometry Re-Densification Module. By
integrating these two modules, our method yields a compact feature
representation that provides efficient context modeling and accelerates the
coding process. Experiments on the KITTI dataset demonstrate state-of-the-art
compression ratios and real-time performance, achieving 26 FPS for both
encoding and decoding at 12-bit quantization. Code is available at
https://github.com/pengpeng-yu/FastPCC.

### 8. [Droplet3D: Commonsense Priors from Videos Facilitate 3D Generation](http://arxiv.org/pdf/2508.20470v1)

Authors: Xiaochuan Li, Guoguang Du, Runze Zhang, Liang Jin, Qi Jia, Lihua Lu, Zhenhua Guo, Yaqian Zhao, Haiyang Liu, Tianqi Wang, Changsheng Li, Xiaoli Gong, Rengang Li, Baoyu Fan

Scaling laws have validated the success and promise of large-data-trained
models in creative generation across text, image, and video domains. However,
this paradigm faces data scarcity in the 3D domain, as there is far less of it
available on the internet compared to the aforementioned modalities.
Fortunately, there exist adequate videos that inherently contain commonsense
priors, offering an alternative supervisory signal to mitigate the
generalization bottleneck caused by limited native 3D data. On the one hand,
videos capturing multiple views of an object or scene provide a spatial
consistency prior for 3D generation. On the other hand, the rich semantic
information contained within the videos enables the generated content to be
more faithful to the text prompts and semantically plausible. This paper
explores how to apply the video modality in 3D asset generation, spanning
datasets to models. We introduce Droplet3D-4M, the first large-scale video
dataset with multi-view level annotations, and train Droplet3D, a generative
model supporting both image and dense text input. Extensive experiments
validate the effectiveness of our approach, demonstrating its ability to
produce spatially consistent and semantically plausible content. Moreover, in
contrast to the prevailing 3D solutions, our approach exhibits the potential
for extension to scene-level applications. This indicates that the commonsense
priors from the videos significantly facilitate 3D creation. We have
open-sourced all resources including the dataset, code, technical framework,
and model weights: https://dropletx.github.io/.

### 9. [Realistic and Controllable 3D Gaussian-Guided Object Editing for Driving Video Generation](http://arxiv.org/pdf/2508.20471v1)

Authors: Jiusi Li, Jackson Jiang, Jinyu Miao, Miao Long, Tuopu Wen, Peijin Jia, Shengxiang Liu, Chunlei Yu, Maolin Liu, Yuzhan Cai, Kun Jiang, Mengmeng Yang, Diange Yang

Corner cases are crucial for training and validating autonomous driving
systems, yet collecting them from the real world is often costly and hazardous.
Editing objects within captured sensor data offers an effective alternative for
generating diverse scenarios, commonly achieved through 3D Gaussian Splatting
or image generative models. However, these approaches often suffer from limited
visual fidelity or imprecise pose control. To address these issues, we propose
G^2Editor, a framework designed for photorealistic and precise object editing
in driving videos. Our method leverages a 3D Gaussian representation of the
edited object as a dense prior, injected into the denoising process to ensure
accurate pose control and spatial consistency. A scene-level 3D bounding box
layout is employed to reconstruct occluded areas of non-target objects.
Furthermore, to guide the appearance details of the edited object, we
incorporate hierarchical fine-grained features as additional conditions during
generation. Experiments on the Waymo Open Dataset demonstrate that G^2Editor
effectively supports object repositioning, insertion, and deletion within a
unified framework, outperforming existing methods in both pose controllability
and visual quality, while also benefiting downstream data-driven tasks.

### 10. [Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](http://arxiv.org/pdf/2508.20478v1)

Authors: Yuan Xie, Tianshui Chen, Zheng Ge, Lionel Ni

Long-form video understanding, characterized by long-range temporal
dependencies and multiple events, remains a challenge. Existing methods often
rely on static reasoning or external visual-language models (VLMs), which face
issues like complexity and sub-optimal performance due to the lack of
end-to-end training. In this paper, we propose Video-MTR, a reinforced
multi-turn reasoning framework designed to enable iterative key video segment
selection and question comprehension. Unlike traditional video reasoning
pipeline, which generate predictions in a single turn, Video-MTR performs
reasoning in multiple turns, selecting video segments progressively based on
the evolving understanding of previously processed segments and the current
question. This iterative process allows for a more refined and contextually
aware analysis of the video. To ensure intermediate reasoning process, we
introduce a novel gated bi-level reward system, combining trajectory-level
rewards based on answer correctness and turn-level rewards emphasizing
frame-query relevance. This system optimizes both video segment selection and
question comprehension, eliminating the need for external VLMs and allowing
end-to-end training. Extensive experiments on benchmarks like VideoMME, MLVU,
and EgoSchema demonstrate that Video-MTR outperforms existing methods in both
accuracy and efficiency, advancing the state-of-the-art in long video
understanding.

### Computers and Society

### 1. [Automated Quality Assessment for LLM-Based Complex Qualitative Coding: A Confidence-Diversity Framework](http://arxiv.org/pdf/2508.20462v1)

Authors: Zhilong Zhao, Yindi Liu

While previous research demonstrated effective automated quality assessment
for accessible LLM coding tasks, a fundamental question remains: can
confidence-diversity frameworks maintain reliability for complex analytical
tasks requiring specialized domain expertise and extensive text comprehension?
Traditional inter-coder reliability measures become prohibitively expensive at
scale, yet the lack of reliable automated quality assessment methods creates
methodological barriers to AI adoption in sophisticated qualitative research.
This study extends dual-signal quality assessment combining model confidence
and inter-model consensus from accessible to complex analytical domains. We
systematically validate this approach across three domains: legal reasoning
(390 Supreme Court cases), political analysis (645 hyperpartisan articles), and
medical classification (1,000 clinical transcripts). Results demonstrate that
uncertainty-based indicators maintain predictive validity in complex tasks,
with external entropy showing consistent negative correlations with accuracy (r
= -0.179 to -0.273, p < 0.001) and confidence exhibiting positive correlations
in two domains (r = 0.104 to 0.429). Systematic weight optimization achieves
6.6 to 113.7 percent improvements over single-signal approaches, with optimized
weights transferring effectively across domains (100 percent success rate). An
intelligent triage system reduces manual verification effort by 44.6 percent
while maintaining quality standards. These findings establish that automated
quality assessment can scale from accessible to complex analytical tasks,
providing practical tools for expanding AI-assisted qualitative research.
Future work will focus on addressing long-tail challenges in high-disagreement,
low-confidence cases to further enhance screening efficiency.

### 2. [Composable Life: Speculation for Decentralized AI Life](http://arxiv.org/pdf/2508.20668v1)

Authors: Botao Amber Hu, Fangting

"Composable Life" is a hybrid project blending design fiction, experiential
virtual reality, and scientific research. Through a multi-perspective,
cross-media approach to speculative design, it reshapes our understanding of
the digital future from AI's perspective. The project explores the hypothetical
first suicide of an on-chain artificial life, examining the complex symbiotic
relationship between humans, AI, and blockchain technology.

### 3. [When technology is not enough: Insights from a pilot cybersecurity culture assessment in a safety-critical industrial organisation](http://arxiv.org/pdf/2508.20811v1)

Authors: Tita Alissa Bach, Linn Pedersen, Maria Kinck Borén†, Lisa Christoffersen Temte†

As cyber threats increasingly exploit human behaviour, technical controls
alone cannot ensure organisational cybersecurity (CS). Strengthening
cybersecurity culture (CSC) is vital in safety-critical industries, yet
empirical research in real-world industrial setttings is scarce. This paper
addresses this gap through a pilot mixed-methods CSC assessment in a global
safety-critical organisation. We examined employees' CS knowledge, attitudes,
behaviours, and organisational factors shaping them. A survey and
semi-structured interviews were conducted at a global organisation in
safety-critical industries, across two countries chosen for contrasting
phishing simulation performance: Country 1 stronger, Country 2 weaker. In
Country 1, 258 employees were invited (67%), in Country 2, 113 were invited
(30%). Interviews included 20 and 10 participants respectively. Overall CSC
profiles were similar but revealed distinct challenges. Both showed strong
phishing awareness and prioritised CS, yet most viewed phishing as the main
risk and lacked clarity on handling other incidents. Line managers were default
contacts, but follow-up on reported concerns was unclear. Participants
emphasized aligning CS expectations with job relevance and workflows. Key
contributors to differences emerged: Country 1 had external employees with
limited access to CS training and policies, highlighting monitoring gaps. In
Country 2, low survey response stemmed from a "no-link in email" policy. While
this policy may have boosted phishing performance, it also underscored
inconsistencies in CS practices. Findings show that resilient CSC requires
leadership involvement, targeted communication, tailored measures,
policy-practice alignment, and regular assessments. Embedding these into
strategy complements technical defences and strengthens sustainable CS in
safety-critical settings.

### 4. [Vibe Coding: Is Human Nature the Ghost in the Machine?](http://arxiv.org/pdf/2508.20918v1)

Authors: Cory Knobel, Nicole Radziwill

This exploratory study examined the consistency of human-AI collaboration by
analyzing three extensive "vibe coding" sessions between a human product lead
and an AI software engineer. We investigated similarities and differences in
team dynamics, communication patterns, and development outcomes across both
projects. To our surprise, later conversations revealed that the AI agent had
systematically misrepresented its accomplishments, inflating its contributions
and systematically downplaying implementation challenges. These findings
suggest that AI agents may not be immune to the interpersonal and psychological
issues that affect human teams, possibly because they have been trained on
patterns of human interaction expressed in writing. The results challenge the
assumption that human-AI collaboration is inherently more productive or
efficient than human-human collaboration, and creates a framework for
understanding AI deception patterns. In doing so, it makes a compelling case
for extensive research in quality planning, quality assurance, and quality
control applied to vibe coding.

### 5. [Enhancing Semantic Document Retrieval- Employing Group Steiner Tree Algorithm with Domain Knowledge Enrichment](http://arxiv.org/pdf/2508.20543v1)

Authors: Apurva Kulkarni, Chandrashekar Ramanathan, Vinu E Venugopal

Retrieving pertinent documents from various data sources with diverse
characteristics poses a significant challenge for Document Retrieval Systems.
The complexity of this challenge is further compounded when accounting for the
semantic relationship between data and domain knowledge. While existing
retrieval systems using semantics (usually represented as Knowledge Graphs
created from open-access resources and generic domain knowledge) hold promise
in delivering relevant outcomes, their precision may be compromised due to the
absence of domain-specific information and reliance on outdated knowledge
sources. In this research, the primary focus is on two key contributions- a)
the development of a versatile algorithm- 'Semantic-based Concept Retrieval
using Group Steiner Tree' that incorporates domain information to enhance
semantic-aware knowledge representation and data access, and b) the practical
implementation of the proposed algorithm within a document retrieval system
using real-world data. To assess the effectiveness of the SemDR system,
research work conducts performance evaluations using a benchmark consisting of
170 real-world search queries. Rigorous evaluation and verification by domain
experts are conducted to ensure the validity and accuracy of the results. The
experimental findings demonstrate substantial advancements when compared to the
baseline systems, with precision and accuracy achieving levels of 90% and 82%
respectively, signifying promising improvements.

### 6. [Dynamics of Gender Bias in Software Engineering](http://arxiv.org/pdf/2508.21050v1)

Authors: Thomas J. Misa

The field of software engineering is embedded in both engineering and
computer science, and may embody gender biases endemic to both. This paper
surveys software engineering's origins and its long-running attention to
engineering professionalism, profiling five leaders; it then examines the
field's recent attention to gender issues and gender bias. It next
quantitatively analyzes women's participation as research authors in the
field's leading International Conference of Software Engineering (1976-2010),
finding a dozen years with statistically significant gender exclusion. Policy
dimensions of research on gender bias in computing are suggested.

### 7. [Governable AI: Provable Safety Under Extreme Threat Models](http://arxiv.org/pdf/2508.20411v1)

Authors: Donglin Wang, Weiyun Liang, Chunyuan Chen, Jing Xu, Yulong Fu

As AI rapidly advances, the security risks posed by AI are becoming
increasingly severe, especially in critical scenarios, including those posing
existential risks. If AI becomes uncontrollable, manipulated, or actively
evades safety mechanisms, it could trigger systemic disasters. Existing AI
safety approaches-such as model enhancement, value alignment, and human
intervention-suffer from fundamental, in-principle limitations when facing AI
with extreme motivations and unlimited intelligence, and cannot guarantee
security. To address this challenge, we propose a Governable AI (GAI) framework
that shifts from traditional internal constraints to externally enforced
structural compliance based on cryptographic mechanisms that are
computationally infeasible to break, even for future AI, under the defined
threat model and well-established cryptographic assumptions.The GAI framework
is composed of a simple yet reliable, fully deterministic, powerful, flexible,
and general-purpose rule enforcement module (REM); governance rules; and a
governable secure super-platform (GSSP) that offers end-to-end protection
against compromise or subversion by AI. The decoupling of the governance rules
and the technical platform further enables a feasible and generalizable
technical pathway for the safety governance of AI. REM enforces the bottom line
defined by governance rules, while GSSP ensures non-bypassability,
tamper-resistance, and unforgeability to eliminate all identified attack
vectors. This paper also presents a rigorous formal proof of the security
properties of this mechanism and demonstrates its effectiveness through a
prototype implementation evaluated in representative high-stakes scenarios.

### 8. [Enabling Equitable Access to Trustworthy Financial Reasoning](http://arxiv.org/pdf/2508.21051v1)

Authors: William Jurayj, Nils Holzenberger, Benjamin Van Durme

According to the United States Internal Revenue Service, ''the average
American spends $\$270$ and 13 hours filing their taxes''. Even beyond the
U.S., tax filing requires complex reasoning, combining application of
overlapping rules with numerical calculations. Because errors can incur costly
penalties, any automated system must deliver high accuracy and auditability,
making modern large language models (LLMs) poorly suited for this task. We
propose an approach that integrates LLMs with a symbolic solver to calculate
tax obligations. We evaluate variants of this system on the challenging
StAtutory Reasoning Assessment (SARA) dataset, and include a novel method for
estimating the cost of deploying such a system based on real-world penalties
for tax errors. We further show how combining up-front translation of
plain-text rules into formal logic programs, combined with intelligently
retrieved exemplars for formal case representations, can dramatically improve
performance on this task and reduce costs to well below real-world averages.
Our results demonstrate the promise and economic feasibility of neuro-symbolic
architectures for increasing equitable access to reliable tax assistance.

### Databases

### 1. [Efficient Forkless Blockchain Databases](http://arxiv.org/pdf/2508.20686v1)

Authors: Herbert Jordan, Kamil Jezek, Pavle Subotic, Bernhard Scholz

Operating nodes in an L1 blockchain remains costly despite recent advances in
blockchain technology. One of the most resource-intensive components of a node
is the blockchain database, also known as StateDB, that manages balances,
nonce, code, and the persistent storage of accounts/smart contracts. Although
the blockchain industry has transitioned from forking to forkless chains due to
improved consensus protocols, forkless blockchains still rely on legacy forking
databases that are suboptimal for their purposes. In this paper, we propose a
forkless blockchain database, showing a 100x improvement in storage and a 10x
improvement in throughput compared to the geth-based Fantom Blockchain client.

### 2. [KG-CQR: Leveraging Structured Relation Representations in Knowledge Graphs for Contextual Query Retrieval](http://arxiv.org/pdf/2508.20417v1)

Authors: Chi Minh Bui, Ngoc Mai Thieu, Van Vinh Nguyen, Json J. Jung, Khac-Hoai Nam Bui

The integration of knowledge graphs (KGs) with large language models (LLMs)
offers significant potential to improve the retrieval phase of
retrieval-augmented generation (RAG) systems. In this study, we propose KG-CQR,
a novel framework for Contextual Query Retrieval (CQR) that enhances the
retrieval phase by enriching the contextual representation of complex input
queries using a corpus-centric KG. Unlike existing methods that primarily
address corpus-level context loss, KG-CQR focuses on query enrichment through
structured relation representations, extracting and completing relevant KG
subgraphs to generate semantically rich query contexts. Comprising subgraph
extraction, completion, and contextual generation modules, KG-CQR operates as a
model-agnostic pipeline, ensuring scalability across LLMs of varying sizes
without additional training. Experimental results on RAGBench and MultiHop-RAG
datasets demonstrate KG-CQR's superior performance, achieving a 4-6%
improvement in mAP and a 2-3% improvement in Recall@25 over strong baseline
models. Furthermore, evaluations on challenging RAG tasks such as multi-hop
question answering show that, by incorporating KG-CQR, the performance
consistently outperforms the existing baseline in terms of retrieval
effectiveness

### 3. [Research Challenges in Relational Database Management Systems for LLM Queries](http://arxiv.org/pdf/2508.20912v1)

Authors: Kerem Akillioglu, Anurag Chakraborty, Sairaj Voruganti, M. Tamer Özsu

Large language models (LLMs) have become essential for applications such as
text summarization, sentiment analysis, and automated question-answering.
Recently, LLMs have also been integrated into relational database management
systems to enhance querying and support advanced data processing. Companies
such as Amazon, Databricks, Google, and Snowflake offer LLM invocation directly
within SQL, denoted as LLM queries, to boost data insights. However,
open-source solutions currently have limited functionality and poor
performance. In this work, we present an early exploration of two open-source
systems and one enterprise platform, using five representative queries to
expose functional, performance, and scalability limits in today's SQL-invoked
LLM integrations. We identify three main issues: enforcing structured outputs,
optimizing resource utilization, and improving query planning. We implemented
initial solutions and observed improvements in accommodating LLM powered SQL
queries. These early gains demonstrate that tighter integration of LLM+DBMS is
the key to scalable and efficient processing of LLM queries.

### 4. [Graph-Based Feature Augmentation for Predictive Tasks on Relational Datasets](http://arxiv.org/pdf/2508.20986v1)

Authors: Lianpeng Qiao, Ziqi Cao, Kaiyu Feng, Ye Yuan, Guoren Wang

Data has become a foundational asset driving innovation across domains such
as finance, healthcare, and e-commerce. In these areas, predictive modeling
over relational tables is commonly employed, with increasing emphasis on
reducing manual effort through automated machine learning (AutoML) techniques.
This raises an interesting question: can feature augmentation itself be
automated and identify and utilize task-related relational signals?
  To address this challenge, we propose an end-to-end automated feature
augmentation framework, ReCoGNN, which enhances initial datasets using features
extracted from multiple relational tables to support predictive tasks. ReCoGNN
first captures semantic dependencies within each table by modeling intra-table
attribute relationships, enabling it to partition tables into structured,
semantically coherent segments. It then constructs a heterogeneous weighted
graph that represents inter-row relationships across all segments. Finally,
ReCoGNN leverages message-passing graph neural networks to propagate
information through the graph, guiding feature selection and augmenting the
original dataset. Extensive experiments conducted on ten real-life and
synthetic datasets demonstrate that ReCoGNN consistently outperforms existing
methods on both classification and regression tasks.

### Distributed, Parallel, and Cluster Computing

### 1. [pdGRASS: A Fast Parallel Density-Aware Algorithm for Graph Spectral Sparsification](http://arxiv.org/pdf/2508.20403v1)

Authors: Tiancheng Zhao, Zekun Yin, Huihai An, Xiaoyu Yang, Zhou Jin, Jiasi Shen, Helen Xu

Graph Spectral Sparsification (GSS) identifies an ultra-sparse subgraph, or
sparsifier, whose Laplacian matrix closely approximates the spectral properties
of the original graph, enabling substantial reductions in computational
complexity for computationally intensive problems in scientific computing. The
state-of-the-art method for efficient GSS is feGRASS, consisting of two steps:
1) spanning tree generation and 2) off-tree edge recovery. However, feGRASS
suffers from two main issues: 1) difficulties in parallelizing the recovery
step for strict data dependencies, and 2) performance degradation on skewed
inputs, often requiring multiple passes to recover sufficient edges. To address
these challenges, we propose parallel density-aware Graph Spectral
Sparsification (pdGRASS), a parallel algorithm that organizes edges into
disjoint subtasks without data dependencies between them, enabling efficient
parallelization and sufficient edge recovery in a single pass. We empirically
evaluate feGRASS and pdGRASS based on 1) off-tree edge-recovery runtime and 2)
sparsifier quality, measured by the iteration count required for convergence in
a preconditioned conjugate gradient (PCG) application. The evaluation
demonstrates that, depending on the number of edges recovered, pdGRASS achieves
average speedups ranging from 3.9x to 8.8x. The resulting sparsifiers also show
between 1.2x higher and 1.8x lower PCG iteration counts, with further
improvements as more edges are recovered. Additionally, pdGRASS mitigates the
worst-case runtimes of feGRASS with over 1000x speedup. These results highlight
pdGRASS's significant improvements in scalability and performance for the graph
spectral sparsification problem.

### 2. [Collaborative Evolution of Intelligent Agents in Large-Scale Microservice Systems](http://arxiv.org/pdf/2508.20508v1)

Authors: Yilin Li, Song Han, Sibo Wang, Ming Wang, Renzi Meng

This paper proposes an intelligent service optimization method based on a
multi-agent collaborative evolution mechanism to address governance challenges
in large-scale microservice architectures. These challenges include complex
service dependencies, dynamic topology structures, and fluctuating workloads.
The method models each service as an agent and introduces graph representation
learning to construct a service dependency graph. This enables agents to
perceive and embed structural changes within the system. Each agent learns its
policy based on a Markov Decision Process. A centralized training and
decentralized execution framework is used to integrate local autonomy with
global coordination. To enhance overall system performance and adaptability, a
game-driven policy optimization mechanism is designed. Through a
selection-mutation process, agent strategy distributions are dynamically
adjusted. This supports adaptive collaboration and behavioral evolution among
services. Under this mechanism, the system can quickly respond and achieve
stable policy convergence when facing scenarios such as sudden workload spikes,
topology reconfigurations, or resource conflicts. To evaluate the effectiveness
of the proposed method, experiments are conducted on a representative
microservice simulation platform. Comparative analyses are performed against
several advanced approaches, focusing on coordination efficiency, adaptability,
and policy convergence performance. Experimental results show that the proposed
method outperforms others in several key metrics. It significantly improves
governance efficiency and operational stability in large-scale microservice
systems. The method demonstrates strong practical value and engineering
feasibility.

### 3. [High performance visualization for Astronomy and Cosmology: the VisIVO's pathway toward Exascale systems](http://arxiv.org/pdf/2508.20603v1)

Authors: Eva Sciacca, Nicola Tuccari, Fabio Vitello, Valentina Cesare

Petabyte-scale data volumes are generated by observations and simulations in
modern astronomy and astrophysics. Storage, access, and data analysis are
significantly hampered by such data volumes and are leading to the development
of a new generation of software tools. The Visualization Interface for the
Virtual Observatory (VisIVO) has been designed, developed and maintained by
INAF since 2005 to perform multi-dimensional data analysis and knowledge
discovery in multivariate astrophysical datasets. Utilizing containerization
and virtualization technologies, VisIVO has already been used to exploit
distributed computing infrastructures including the European Open Science Cloud
(EOSC).
  We intend to adapt VisIVO solutions for high performance visualization of
data generated on the (pre-)Exascale systems by HPC applications in
Astrophysics and Cosmology (A\&C), including GADGET (GAlaxies with Dark matter
and Gas) and PLUTO simulations, thanks to the collaboration within the SPACE
Center of Excellence, the H2020 EUPEX Project, and the ICSC National Research
Centre. In this work, we outline the evolution's course as well as the
execution strategies designed to achieve the following goals: enhance the
portability of the VisIVO modular applications and their resource requirements;
foster reproducibility and maintainability; take advantage of a more flexible
resource exploitation over heterogeneous HPC facilities; and, finally, minimize
data-movement overheads and improve I/O performances.

### 4. [Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs](http://arxiv.org/pdf/2508.20333v1)

Authors: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

Large Language Models (LLMs) are aligned to meet ethical standards and safety
requirements by training them to refuse answering harmful or unsafe prompts. In
this paper, we demonstrate how adversaries can exploit LLMs' alignment to
implant bias, or enforce targeted censorship without degrading the model's
responsiveness to unrelated topics. Specifically, we propose Subversive
Alignment Injection (SAI), a poisoning attack that leverages the alignment
mechanism to trigger refusal on specific topics or queries predefined by the
adversary. Although it is perhaps not surprising that refusal can be induced
through overalignment, we demonstrate how this refusal can be exploited to
inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning
defenses including LLM state forensics, as well as robust aggregation
techniques that are designed to detect poisoning in FL settings. We demonstrate
the practical dangers of this attack by illustrating its end-to-end impacts on
LLM-powered application pipelines. For chat based applications such as
ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare
questions to targeted racial category leading to high bias ($\Delta DP$ of
23%). We also show that bias can be induced in other NLP tasks: for a resume
selection pipeline aligned to refuse to summarize CVs from a selected
university, high bias in selection ($\Delta DP$ of 27%) results. Even higher
bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.

### 5. [CoFormer: Collaborating with Heterogeneous Edge Devices for Scalable Transformer Inference](http://arxiv.org/pdf/2508.20375v1)

Authors: Guanyu Xu, Zhiwei Hao, Li Shen, Yong Luo, Fuhui Sun, Xiaoyan Wang, Han Hu, Yonggang Wen

The impressive performance of transformer models has sparked the deployment
of intelligent applications on resource-constrained edge devices. However,
ensuring high-quality service for real-time edge systems is a significant
challenge due to the considerable computational demands and resource
requirements of these models. Existing strategies typically either offload
transformer computations to other devices or directly deploy compressed models
on individual edge devices. These strategies, however, result in either
considerable communication overhead or suboptimal trade-offs between accuracy
and efficiency. To tackle these challenges, we propose a collaborative
inference system for general transformer models, termed CoFormer. The central
idea behind CoFormer is to exploit the divisibility and integrability of
transformer. An off-the-shelf large transformer can be decomposed into multiple
smaller models for distributed inference, and their intermediate results are
aggregated to generate the final output. We formulate an optimization problem
to minimize both inference latency and accuracy degradation under heterogeneous
hardware constraints. DeBo algorithm is proposed to first solve the
optimization problem to derive the decomposition policy, and then progressively
calibrate decomposed models to restore performance. We demonstrate the
capability to support a wide range of transformer models on heterogeneous edge
devices, achieving up to 3.1$\times$ inference speedup with large transformer
models. Notably, CoFormer enables the efficient inference of GPT2-XL with 1.6
billion parameters on edge devices, reducing memory requirements by 76.3\%.
CoFormer can also reduce energy consumption by approximately 40\% while
maintaining satisfactory inference performance.

### 6. [A Hybrid Stochastic Gradient Tracking Method for Distributed Online Optimization Over Time-Varying Directed Networks](http://arxiv.org/pdf/2508.20645v1)

Authors: Xinli Shi, Xingxing Yuan, Longkang Zhu, Guanghui Wen

With the increasing scale and dynamics of data, distributed online
optimization has become essential for real-time decision-making in various
applications. However, existing algorithms often rely on bounded gradient
assumptions and overlook the impact of stochastic gradients, especially in
time-varying directed networks. This study proposes a novel Time-Varying Hybrid
Stochastic Gradient Tracking algorithm named TV-HSGT, based on hybrid
stochastic gradient tracking and variance reduction mechanisms. Specifically,
TV-HSGT integrates row-stochastic and column-stochastic communication schemes
over time-varying digraphs, eliminating the need for Perron vector estimation
or out-degree information. By combining current and recursive stochastic
gradients, it effectively reduces gradient variance while accurately tracking
global descent directions. Theoretical analysis demonstrates that TV-HSGT can
achieve improved bounds on dynamic regret without assuming gradient
boundedness. Experimental results on logistic regression tasks confirm the
effectiveness of TV-HSGT in dynamic and resource-constrained environments.

### Digital Libraries

### 1. [An analysis of the effects of open science indicators on citations in the French Open Science Monitor](http://arxiv.org/pdf/2508.20747v1)

Authors: Giovanni Colavizza, Lauren Cadwallader, Iain Hrynaszkiewicz

This study investigates the correlation of citation impact with various open
science indicators (OSI) within the French Open Science Monitor (FOSM), a
dataset comprising approximately 900,000 publications authored by French
authors from 2020 to 2022. By integrating data from OpenAlex and Crossref, we
analyze open science indicators such as the presence of a pre-print, data
sharing, and software sharing in 576,537 publications in the FOSM dataset. Our
analysis reveals a positive correlation between these OSI and citation counts.
Considering our most complete citation prediction model, we find pre-prints are
correlated with a significant positive effect of 19% on citation counts,
software sharing of 13.5%, and data sharing of 14.3%. We find large variations
in the correlations of OSIs with citations in different research disciplines,
and observe that open access status of publications is correlated with a 8.6%
increase in citations in our model. While these results remain observational
and are limited to the scope of the analysis, they suggest a consistent
correlation between citation advantages and open science indicators. Our
results may be valuable to policy makers, funding agencies, researchers,
publishers, institutions, and other stakeholders who are interested in
understanding the academic impacts, or effects, of open science practices.

### 2. [Leveraging Large Language Models for Generating Research Topic Ontologies: A Multi-Disciplinary Study](http://arxiv.org/pdf/2508.20693v1)

Authors: Tanay Aggarwal, Angelo Salatino, Francesco Osborne, Enrico Motta

Ontologies and taxonomies of research fields are critical for managing and
organising scientific knowledge, as they facilitate efficient classification,
dissemination and retrieval of information. However, the creation and
maintenance of such ontologies are expensive and time-consuming tasks, usually
requiring the coordinated effort of multiple domain experts. Consequently,
ontologies in this space often exhibit uneven coverage across different
disciplines, limited inter-domain connectivity, and infrequent updating cycles.
In this study, we investigate the capability of several large language models
to identify semantic relationships among research topics within three academic
domains: biomedicine, physics, and engineering. The models were evaluated under
three distinct conditions: zero-shot prompting, chain-of-thought prompting, and
fine-tuning on existing ontologies. Additionally, we assessed the cross-domain
transferability of fine-tuned models by measuring their performance when
trained in one domain and subsequently applied to a different one. To support
this analysis, we introduce PEM-Rel-8K, a novel dataset consisting of over
8,000 relationships extracted from the most widely adopted taxonomies in the
three disciplines considered in this study: MeSH, PhySH, and IEEE. Our
experiments demonstrate that fine-tuning LLMs on PEM-Rel-8K yields excellent
performance across all disciplines.

### Discrete Mathematics

### 1. [Enhancing Soft Happiness via Evolutionary Algorithms](http://arxiv.org/pdf/2508.20934v1)

Authors: Mohammad Hadi Shekarriza, Dhananjay Thiruvadya, Asef Nazari

For $0\leq \rho\leq 1$, a $\rho$-happy vertex $v$ in a coloured graph shares
colour with at least $\rho\mathrm{deg}(v)$ of its neighbours. Soft happy
colouring of a graph $G$ with $k$ colours extends a partial $k$-colouring to a
complete vertex $k$-colouring such that the number of $\rho$-happy vertices is
maximum among all such colouring extensions. The problem is known to be
NP-hard, and an optimal solution has a direct relation with the community
structure of the graph. In addition, some heuristics and local search
algorithms, such as {\sf Local Maximal Colouring} ({\sf LMC}) and {\sf Local
Search} ({\sf LS}), have already been introduced in the literature. In this
paper, we design Genetic and Memetic Algorithms for soft happy colouring and
test them for a large set of randomly generated partially coloured graphs.
Memetic Algorithms yield a higher number of $\rho$-happy vertices, but Genetic
Algorithms can perform well only when their initial populations are locally
improved by {\sf LMC} or {\sf LS}. Statistically significant results indicate
that both Genetic and Memetic Algorithms achieve high average accuracy in
community detection when their initial populations are enhanced using {\sf
LMC}. Moreover, among the competing methods, the evolutionary algorithms
identified the greatest number of complete solutions.

### 2. [Computer-assisted graph theory: a survey](http://arxiv.org/pdf/2508.20825v1)

Authors: Jorik Jooken

Computers and algorithms play an ever-increasing role in obtaining new
results in graph theory. In this survey, we present a broad range of techniques
used in computer-assisted graph theory, including the exhaustive generation of
all pairwise non-isomorphic graphs within a given class, the use of searchable
databases containing graphs and invariants as well as other established and
emerging algorithmic paradigms. We cover approaches based on mixed integer
linear programming, semidefinite programming, dynamic programming, SAT solving,
metaheuristics and machine learning. The techniques are illustrated with
numerous detailed results covering several important subareas of graph theory
such as extremal graph theory, graph coloring, structural graph theory,
spectral graph theory, regular graphs, topological graph theory, special sets
in graphs, algebraic graph theory and chemical graph theory. We also present
some smaller new results that demonstrate how readily a computer-assisted graph
theory approach can be applied once the appropriate tools have been developed.

### 3. [Vertex-Based Localization of generalized Turán Problems](http://arxiv.org/pdf/2508.20936v1)

Authors: Rajat Adak, L. Sunil Chandran

Let $\mathcal{F}$ be a family of graphs. A graph is called $\mathcal{F}$-free
if it does not contain any member of $\mathcal{F}$. Generalized Tur\'{a}n
problems aim to maximize the number of copies of a graph $H$ in an $n$-vertex
$\mathcal{F}$-free graph. This maximum is denoted by $ex(n, H, \mathcal{F})$.
When $H \cong K_2$, it is simply denoted by $ex(n,F)$. Erd\H{o}s and Gallai
established the bounds $ex(n, P_{k+1}) \leq \frac{n(k-1)}{2}$ and $ex(n,
C_{\geq k+1}) \leq \frac{k(n-1)}{2}$. This was later extended by Luo
\cite{luo2018maximum}, who showed that $ex(n, K_s, P_{k+1}) \leq \frac{n}{k}
\binom{k}{s}$ and $ex(n, K_s, C_{\geq k+1}) \leq \frac{n-1}{k-1} \binom{k}{s}$.
Let $N(G,K_s)$ denote the number of copies of $K_s$ in $G$. In this paper, we
use the vertex-based localization framework, introduced in
\cite{adak2025vertex}, to generalize Luo's bounds. In a graph $G$, for each $v
\in V(G)$, define $p(v)$ to be the length of the longest path that contains
$v$. We show that \[N(G,K_s) \leq \sum_{v \in V(G)}
\frac{1}{p(v)+1}{p(v)+1\choose s} = \frac{1}{s}\sum_{v \in V(G)}{p(v) \choose
s-1}\] We strengthen the cycle bound from \cite{luo2018maximum} as follows: In
graph $G$, for each $v \in V(G)$, let $c(v)$ be the length of the longest cycle
that contains $v$, or $2$ if $v$ is not part of any cycle. We prove that
\[N(G,K_s) \leq \left(\sum_{v\in V(G)}\frac{1}{c(v)-1}{c(v) \choose s}\right) -
\frac{1}{c(u)-1}{c(u) \choose s}\] where $c(u)$ denotes the circumference of
$G$. We provide full proofs for the cases $s = 1$ and $s \geq 3$, while the
case $s = 2$ follows from the result in \cite{adak2025vertex}. \newline
Furthermore, we characterize the class of extremal graphs that attain equality
for these bounds.

### 4. [Localized Clique Bounds in Bounded-Degree and Bounded-Path Graphs](http://arxiv.org/pdf/2508.20946v1)

Authors: Rajat Adak, L. Sunil Chandran

Let $\mathcal{F}$ be a family of graphs. A graph is said to be
$\mathcal{F}$-free if it contains no member of $\mathcal{F}$. The generalized
Tur\'{a}n number $ex(n,H,\mathcal{F})$ denotes the maximum number of copies of
a graph $H$ in an $n$-vertex $\mathcal{F}$-free graph, while the generalized
edge Tur\'{a}n number $mex(m,H,\mathcal{F})$ denotes the maximum number of
copies of $H$ in an $m$-edge $\mathcal{F}$-free graph.
  It is well known that if a graph has maximum degree $d$, then it is
$K_{1,d+1}$-free. Wood \cite{wood} proved that $ex(n,K_t,K_{1,d+1}) \leq
\frac{n}{d+1}\binom{d+1}{t}$. More recently, Chakraborty and Chen
\cite{CHAKRABORTI2024103955} established analogous bounds for graphs with
bounded maximum path length: $mex(m,K_t,P_{r+1}) \leq
\frac{m}{\binom{r}{2}}\binom{r}{t}$.
  In this paper, we improve these bounds using the localization technique,
based on suitably defined local parameters. Furthermore, we characterize the
extremal graphs attaining these improved bounds.

### 5. [A Multi-Objective Genetic Algorithm for Healthcare Workforce Scheduling](http://arxiv.org/pdf/2508.20953v1)

Authors: Vipul Patel, Anirudh Deodhar, Dagnachew Birru

Workforce scheduling in the healthcare sector is a significant operational
challenge, characterized by fluctuating patient loads, diverse clinical skills,
and the critical need to control labor costs while upholding high standards of
patient care. This problem is inherently multi-objective, demanding a delicate
balance between competing goals: minimizing payroll, ensuring adequate staffing
for patient needs, and accommodating staff preferences to mitigate burnout. We
propose a Multi-objective Genetic Algorithm (MOO-GA) that models the hospital
unit workforce scheduling problem as a multi-objective optimization task. Our
model incorporates real-world complexities, including hourly appointment-driven
demand and the use of modular shifts for a multi-skilled workforce. By defining
objective functions for cost, patient care coverage, and staff satisfaction,
the GA navigates the vast search space to identify a set of high-quality,
non-dominated solutions. Demonstrated on datasets representing a typical
hospital unit, the results show that our MOO-GA generates robust and balanced
schedules. On average, the schedules produced by our algorithm showed a 66\%
performance improvement over a baseline that simulates a conventional, manual
scheduling process. This approach effectively manages trade-offs between
critical operational and staff-centric objectives, providing a practical
decision support tool for nurse managers and hospital administrators.

### 6. [Unclustered BWTs of any Length over Non-Binary Alphabets](http://arxiv.org/pdf/2508.20879v1)

Authors: Gabriele Fici, Estéban Gabory, Giuseppe Romana, Marinella Sciortino

We prove that for every integer $n > 0$ and for every alphabet $\Sigma_k$ of
size $k \geq 3$, there exists a necklace of length $n$ whose Burrows-Wheeler
Transform (BWT) is completely unclustered, i.e., it consists of exactly $n$
runs with no two consecutive equal symbols. These words represent the
worst-case behavior of the BWT for clustering, since the number of BWT runs is
maximized. We also establish a lower bound on their number. This contrasts with
the binary case, where the existence of infinitely many completely unclustered
BWTs is still an open problem, related to Artin's conjecture on primitive
roots.

### 7. [Measuring Ransomware Lateral Movement Susceptibility via Privilege-Weighted Adjacency Matrix Exponentiation](http://arxiv.org/pdf/2508.21005v1)

Authors: Satyam Tyagi, Ganesh Murugesan

Ransomware impact hinges on how easily an intruder can move laterally and
spread to the maximum number of assets. We present a graph-theoretic method to
measure lateral-movement susceptibility and estimate blast radius. We build a
directed multigraph where vertices represent assets and edges represent
reachable services (e.g., RDP/SSH) between them. We model lateral movement as a
probabilistic process using a pivot potential factor $\pi(s)$ for each service.
This allows us to iteratively compute a $K$-hop compromise probability matrix
that captures how compromise propagates through the network. Metrics derived
from this model include: (1) Lateral-Movement Susceptibility (LMS$_K$): the
average probability of a successful lateral movement between any two assets
(0-1 scale); and (2) Blast-Radius Estimate (BRE$_K$): the expected percentage
of assets compromised in an average attack scenario. Interactive control (SSH
22, RDP 3389) gets higher $\pi(s)$ than app-only ports (MySQL 3306, MSSQL
1433), which seldom enable pivoting without an RCE. Across anonymized
enterprise snapshots, pruning high-$\pi(s)$ edges yields the largest
LMS$_K$/BRE$_K$ drop, aligning with CISA guidance, MITRE ATT\&CK (TA0008:
Lateral Movement), and NIST SP~800-207. The framework evaluates
(micro)segmentation and helps prioritize controls that reduce lateral movement
susceptibility and shrink blast radius.

### 8. [Sharp Online Hardness for Large Balanced Independent Sets](http://arxiv.org/pdf/2508.20785v1)

Authors: Abhishek Dhawan, Eren C. Kızıldağ, Neeladri Maitra

We study the algorithmic problem of finding large $\gamma$-balanced
independent sets in dense random bipartite graphs; an independent set is
$\gamma$-balanced if a $\gamma$ proportion of its vertices lie on one side of
the bipartition. In the sparse regime, Perkins and Wang established tight
bounds within the low-degree polynomial (LDP) framework, showing a
factor-$1/(1-\gamma)$ statistical-computational gap via the Overlap Gap
Property (OGP) framework tailored for stable algorithms. However, these
techniques do not appear to extend to the dense setting. For the related large
independent set problem in dense random graph, the best known algorithm is an
online greedy procedure that is inherently unstable, and LDP algorithms are
conjectured to fail even in the "easy" regime where greedy succeeds. We show
that the largest $\gamma$-balanced independent set in dense random bipartite
graphs has size $\alpha:=\frac{\log_b n}{\gamma(1-\gamma)}$ whp, where $n$ is
the size of each bipartition, $p$ is the edge probability, and $b=1/(1-p)$. We
design an online algorithm that achieves $(1-\epsilon)(1-\gamma)\alpha$ whp for
any $\epsilon>0$. We complement this with a sharp lower bound, showing that no
online algorithm can achieve $(1+\epsilon)(1-\gamma)\alpha$ with nonnegligible
probability. Our results suggest that the same factor-$1/(1-\gamma)$ gap is
also present in the dense setting, supporting its conjectured universality.
While the classical greedy procedure on $G(n,p)$ is straightforward, our
algorithm is more intricate: it proceeds in two stages, incorporating a
stopping time and suitable truncation to ensure that $\gamma$-balancedness-a
global constraint-is met despite operating with limited information. Our lower
bound utilizes the OGP framework; we build on a recent refinement of this
framework for online models and extend it to the bipartite setting.

### Data Structures and Algorithms

### 1. [Improved Dominance Filtering for Unions and Minkowski Sums of Pareto Sets](http://arxiv.org/pdf/2508.20689v1)

Authors: Konstantinos Karathanasis, Spyros Kontogiannis, Christos Zaroliagis

A key task in multi-objective optimization is to compute the Pareto subset or
frontier $P$ of a given $d$-dimensional objective space $F$; that is, a maximal
subset $P\subseteq F$ such that every element in $P$ is not-dominated (it is
not worse in all criteria) by any element in $F$. This process, called
dominance-filtering, often involves handling objective spaces derived from
either the union or the Minkowski sum of two given partial objective spaces
which are Pareto sets themselves, and constitutes a major bottleneck in several
multi-objective optimization techniques. In this work, we introduce three new
data structures, ND$^{+}$-trees, QND$^{+}$-trees and TND$^{+}$-trees, which are
designed for efficiently indexing non-dominated objective vectors and
performing dominance-checks. We also devise three new algorithms that
efficiently filter out dominated objective vectors from the union or the
Minkowski sum of two Pareto sets. An extensive experimental evaluation on both
synthetically generated and real-world data sets reveals that our new
algorithms outperform state-of-art techniques for dominance-filtering of unions
and Minkowski sums of Pareto sets, and scale well w.r.t. the number of $d\ge 3$
criteria and the sets' sizes.

### 2. [Unclustered BWTs of any Length over Non-Binary Alphabets](http://arxiv.org/pdf/2508.20879v1)

Authors: Gabriele Fici, Estéban Gabory, Giuseppe Romana, Marinella Sciortino

We prove that for every integer $n > 0$ and for every alphabet $\Sigma_k$ of
size $k \geq 3$, there exists a necklace of length $n$ whose Burrows-Wheeler
Transform (BWT) is completely unclustered, i.e., it consists of exactly $n$
runs with no two consecutive equal symbols. These words represent the
worst-case behavior of the BWT for clustering, since the number of BWT runs is
maximized. We also establish a lower bound on their number. This contrasts with
the binary case, where the existence of infinitely many completely unclustered
BWTs is still an open problem, related to Artin's conjecture on primitive
roots.

### 3. [Spectral Gaps with Quantum Counting Queries and Oblivious State Preparation](http://arxiv.org/pdf/2508.21002v1)

Authors: Almudena Carrera Vazquez, Aleksandros Sobczyk

Approximating the $k$-th spectral gap $\Delta_k=|\lambda_k-\lambda_{k+1}|$
and the corresponding midpoint $\mu_k=\frac{\lambda_k+\lambda_{k+1}}{2}$ of an
$N\times N$ Hermitian matrix with eigenvalues
$\lambda_1\geq\lambda_2\geq\ldots\geq\lambda_N$, is an important special case
of the eigenproblem with numerous applications in science and engineering. In
this work, we present a quantum algorithm which approximates these values up to
additive error $\epsilon\Delta_k$ using a logarithmic number of qubits.
Notably, in the QRAM model, its total complexity (queries and gates) is bounded
by $O\left( \frac{N^2}{\epsilon^{2}\Delta_k^2}\mathrm{polylog}\left(
N,\frac{1}{\Delta_k},\frac{1}{\epsilon},\frac{1}{\delta}\right)\right)$, where
$\epsilon,\delta\in(0,1)$ are the accuracy and the success probability,
respectively. For large gaps $\Delta_k$, this provides a speed-up against the
best-known complexities of classical algorithms, namely, $O \left(
N^{\omega}\mathrm{polylog} \left(
N,\frac{1}{\Delta_k},\frac{1}{\epsilon}\right)\right)$, where $\omega\lesssim
2.371$ is the matrix multiplication exponent. A key technical step in the
analysis is the preparation of a suitable random initial state, which
ultimately allows us to efficiently count the number of eigenvalues that are
smaller than a threshold, while maintaining a quadratic complexity in $N$. In
the black-box access model, we also report an $\Omega(N^2)$ query lower bound
for deciding the existence of a spectral gap in a binary (albeit non-symmetric)
matrix.

### 4. [Sharp Online Hardness for Large Balanced Independent Sets](http://arxiv.org/pdf/2508.20785v1)

Authors: Abhishek Dhawan, Eren C. Kızıldağ, Neeladri Maitra

We study the algorithmic problem of finding large $\gamma$-balanced
independent sets in dense random bipartite graphs; an independent set is
$\gamma$-balanced if a $\gamma$ proportion of its vertices lie on one side of
the bipartition. In the sparse regime, Perkins and Wang established tight
bounds within the low-degree polynomial (LDP) framework, showing a
factor-$1/(1-\gamma)$ statistical-computational gap via the Overlap Gap
Property (OGP) framework tailored for stable algorithms. However, these
techniques do not appear to extend to the dense setting. For the related large
independent set problem in dense random graph, the best known algorithm is an
online greedy procedure that is inherently unstable, and LDP algorithms are
conjectured to fail even in the "easy" regime where greedy succeeds. We show
that the largest $\gamma$-balanced independent set in dense random bipartite
graphs has size $\alpha:=\frac{\log_b n}{\gamma(1-\gamma)}$ whp, where $n$ is
the size of each bipartition, $p$ is the edge probability, and $b=1/(1-p)$. We
design an online algorithm that achieves $(1-\epsilon)(1-\gamma)\alpha$ whp for
any $\epsilon>0$. We complement this with a sharp lower bound, showing that no
online algorithm can achieve $(1+\epsilon)(1-\gamma)\alpha$ with nonnegligible
probability. Our results suggest that the same factor-$1/(1-\gamma)$ gap is
also present in the dense setting, supporting its conjectured universality.
While the classical greedy procedure on $G(n,p)$ is straightforward, our
algorithm is more intricate: it proceeds in two stages, incorporating a
stopping time and suitable truncation to ensure that $\gamma$-balancedness-a
global constraint-is met despite operating with limited information. Our lower
bound utilizes the OGP framework; we build on a recent refinement of this
framework for online models and extend it to the bipartite setting.

### Emerging Technologies

### 1. [Blind Source Separation-Enabled Joint Communication and Sensing in IBFD MIMO Systems](http://arxiv.org/pdf/2508.20409v1)

Authors: Siyao Li, Conrad Prisby, Thomas Yang

This paper addresses the challenge of joint communication and sensing (JCAS)
in next-generation wireless networks, with an emphasis on in-band full-duplex
(IBFD) multiple-input multiple-output (MIMO) systems. Traditionally,
self-interference (SI) in IBFD systems is a major obstacle to recovering the
signal of interest (SOI). Under the JCAS paradigm, however, this high-power SI
signal presents an opportunity for efficient sensing. Since each transceiver
node has access to the original SI signal, its environmental reflections can be
exploited to estimate channel conditions and detect changes, without requiring
dedicated radar waveforms. We propose a blind source separation (BSS)-based
framework to simultaneously perform self-interference cancellation (SIC) and
extract sensing information in IBFD MIMO settings. The approach applies the
Fast Independent Component Analysis (FastICA) algorithm to separate the SI and
SOI signals while enabling simultaneous signal recovery and channel estimation.
Simulation results confirm the framework's effectiveness, showing improved
sensing and communication performance as signal frame size increases.

### 2. [Human-Centered Design for Connected Automation: Predicting Pedestrian Crossing Intentions](http://arxiv.org/pdf/2508.20464v1)

Authors: Sanaz Motamedi, Viktoria Marcus, Griffin Pitts

Road traffic remains a leading cause of death worldwide, with pedestrians and
other vulnerable road users accounting for over half of the 1.19 million annual
fatalities, much of it due to human error. Level-5 automated driving systems
(ADSs), capable of full self-driving without human oversight, have the
potential to reduce these incidents. However, their effectiveness depends not
only on automation performance but also on their ability to communicate intent
and coordinate safely with pedestrians in the absence of traditional driver
cues. Understanding how pedestrians interpret and respond to ADS behavior is
therefore critical to the development of connected vehicle systems. This study
extends the Theory of Planned Behavior (TPB) by incorporating four external
factors (i.e. safety, trust, compatibility, and understanding) to model
pedestrian decision-making in road-crossing scenarios involving level-5 ADSs.
Using data from an online survey (n = 212), results show that perceived
behavioral control, attitude, and social information significantly predict
pedestrians' crossing intentions. External factors, particularly perceived
safety and understanding, strongly influence these constructs. Findings provide
actionable insights for designing external human-machine interfaces (eHMIs) and
cooperative V2X communication strategies that support safe, transparent
interactions between automated vehicles and pedestrians. This work contributes
to the development of inclusive, human-centered connected mobility systems.

### 3. [Encoding Tactile Stimuli for Organoid Intelligence in Braille Recognition](http://arxiv.org/pdf/2508.20850v1)

Authors: Tianyi Liu, Hemma Philamore, Benjamin Ward-Cherrier

This study proposes a generalizable encoding strategy that maps tactile
sensor data to electrical stimulation patterns, enabling neural organoids to
perform an open-loop artificial tactile Braille classification task. Human
forebrain organoids cultured on a low-density microelectrode array (MEA) are
systematically stimulated to characterize the relationship between electrical
stimulation parameters (number of pulse, phase amplitude, phase duration, and
trigger delay) and organoid responses, measured as spike activity and spatial
displacement of the center of activity. Implemented on event-based tactile
inputs recorded from the Evetac sensor, our system achieved an average Braille
letter classification accuracy of 61 percent with a single organoid, which
increased significantly to 83 percent when responses from a three-organoid
ensemble were combined. Additionally, the multi-organoid configuration
demonstrated enhanced robustness against various types of artificially
introduced noise. This research demonstrates the potential of organoids as
low-power, adaptive bio-hybrid computational elements and provides a
foundational encoding framework for future scalable bio-hybrid computing
architectures.

### 4. [Lattice Random Walk Discretisations of Stochastic Differential Equations](http://arxiv.org/pdf/2508.20883v1)

Authors: Samuel Duffield, Maxwell Aifer, Denis Melanson, Zach Belateche, Patrick J. Coles

We introduce a lattice random walk discretisation scheme for stochastic
differential equations (SDEs) that samples binary or ternary increments at each
step, suppressing complex drift and diffusion computations to simple 1 or 2 bit
random values. This approach is a significant departure from traditional
floating point discretisations and offers several advantages; including
compatibility with stochastic computing architectures that avoid floating-point
arithmetic in place of directly manipulating the underlying probability
distribution of a bitstream, elimination of Gaussian sampling requirements,
robustness to quantisation errors, and handling of non-Lipschitz drifts. We
prove weak convergence and demonstrate the advantages through experiments on
various SDEs, including state-of-the-art diffusion models.

### Formal Languages and Automata Theory

### 1. [Evaluating Massively Parallel Algorithms for DFA Minimisation, Equivalence Checking and Inclusion Checking](http://arxiv.org/pdf/2508.20735v1)

Authors: Jan Heemstra, Jan Martens, Anton Wijs

We study parallel algorithms for the minimisation and equivalence checking of
Deterministic Finite Automata (DFAs). Regarding DFA minimisation, we implement
four different massively parallel algorithms on Graphics Processing
Units~(GPUs). Our results confirm the expectations that the algorithm with the
theoretically best time complexity is not practically suitable to run on GPUs
due to the large amount of resources needed. We empirically verify that
parallel partition refinement algorithms from the literature perform better in
practice, even though their time complexity is worse. Furthermore, we introduce
a novel algorithm based on partition refinement with an extra parallel partial
transitive closure step and show that on specific benchmarks it has better
run-time complexity and performs better in practice.
  In addition, we address checking the language equivalence and inclusion of
two DFAs. We consider the Hopcroft-Karp algorithm, and explain how a variant of
it can be parallelised for GPUs. We note that these problems can be encoded for
the GPU-accelerated model checker \GPUexplore, allowing the use its lockless
hash table and fine-grained parallel work distribution mechanism.

### 2. [Formal equivalence between global optimization consistency and random search](http://arxiv.org/pdf/2508.20671v1)

Authors: Gaëtan Serré

We formalize a proof that any stochastic and iterative global optimization
algorithm is consistent over Lipschitz continuous functions if and only if it
samples the whole search space. To achieve this, we use the
L$\exists$$\forall$N theorem prover and the Mathlib library. The major
challenge of this formalization, apart from the technical aspects of the proof
itself, is to converge to a definition of a stochastic and iterative global
optimization algorithm that is both general enough to encompass all algorithms
of this type and specific enough to be used in a formal proof. We define such
an algorithm as a pair of an initial probability measure and a sequence of
Markov kernels that describe the distribution of the next point sampled by the
algorithm given the previous points and their evaluations. We then construct a
probability measure on finite and infinite sequences of iterations of the
algorithm using the Ionescu-Tulcea theorem.

### 3. [Unclustered BWTs of any Length over Non-Binary Alphabets](http://arxiv.org/pdf/2508.20879v1)

Authors: Gabriele Fici, Estéban Gabory, Giuseppe Romana, Marinella Sciortino

We prove that for every integer $n > 0$ and for every alphabet $\Sigma_k$ of
size $k \geq 3$, there exists a necklace of length $n$ whose Burrows-Wheeler
Transform (BWT) is completely unclustered, i.e., it consists of exactly $n$
runs with no two consecutive equal symbols. These words represent the
worst-case behavior of the BWT for clustering, since the number of BWT runs is
maximized. We also establish a lower bound on their number. This contrasts with
the binary case, where the existence of infinitely many completely unclustered
BWTs is still an open problem, related to Artin's conjecture on primitive
roots.

### 4. [QIP $ \subseteq $ AM(2QCFA)](http://arxiv.org/pdf/2508.21020v1)

Authors: Abuzer Yakaryılmaz

The class of languages having polynomial-time classical or quantum
interactive proof systems ($\mathsf{IP}$ or $\mathsf{QIP}$, respectively) is
identical to $\mathsf{PSPACE}$. We show that $\mathsf{PSPACE}$ (and so
$\mathsf{QIP}$) is subset of $\mathsf{AM(2QCFA)}$, the class of languages
having Arthur-Merlin proof systems where the verifiers are two-way finite
automata with quantum and classical states (2QCFAs) communicating with the
provers classically. Our protocols use only rational-valued quantum transitions
and run in double-exponential expected time. Moreover, the member strings are
accepted with probability 1 (i.e., perfect-completeness).

### Graphics

### 1. [Task-Oriented Edge-Assisted Cross-System Design for Real-Time Human-Robot Interaction in Industrial Metaverse](http://arxiv.org/pdf/2508.20664v1)

Authors: Kan Chen, Zhen Meng, Xiangmin Xu, Jiaming Yang, Emma Li, Philip G. Zhao

Real-time human-device interaction in industrial Metaverse faces challenges
such as high computational load, limited bandwidth, and strict latency. This
paper proposes a task-oriented edge-assisted cross-system framework using
digital twins (DTs) to enable responsive interactions. By predicting operator
motions, the system supports: 1) proactive Metaverse rendering for visual
feedback, and 2) preemptive control of remote devices. The DTs are decoupled
into two virtual functions-visual display and robotic control-optimizing both
performance and adaptability. To enhance generalizability, we introduce the
Human-In-The-Loop Model-Agnostic Meta-Learning (HITL-MAML) algorithm, which
dynamically adjusts prediction horizons. Evaluation on two tasks demonstrates
the framework's effectiveness: in a Trajectory-Based Drawing Control task, it
reduces weighted RMSE from 0.0712 m to 0.0101 m; in a real-time 3D scene
representation task for nuclear decommissioning, it achieves a PSNR of 22.11,
SSIM of 0.8729, and LPIPS of 0.1298. These results show the framework's
capability to ensure spatial precision and visual fidelity in real-time,
high-risk industrial environments.

### 2. [Mixture of Contexts for Long Video Generation](http://arxiv.org/pdf/2508.21058v1)

Authors: Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, Gordon Wetzstein

Long video generation is fundamentally a long context memory problem: models
must retain and retrieve salient events across a long range without collapsing
or drifting. However, scaling diffusion transformers to generate long-context
videos is fundamentally limited by the quadratic cost of self-attention, which
makes memory and computation intractable and difficult to optimize for long
sequences. We recast long-context video generation as an internal information
retrieval task and propose a simple, learnable sparse attention routing module,
Mixture of Contexts (MoC), as an effective long-term memory retrieval engine.
In MoC, each query dynamically selects a few informative chunks plus mandatory
anchors (caption, local windows) to attend to, with causal routing that
prevents loop closures. As we scale the data and gradually sparsify the
routing, the model allocates compute to salient history, preserving identities,
actions, and scenes over minutes of content. Efficiency follows as a byproduct
of retrieval (near-linear scaling), which enables practical training and
synthesis, and the emergence of memory and consistency at the scale of minutes.

### Computer Science and Game Theory

### 1. [Guarding Against Malicious Biased Threats (GAMBiT) Experiments: Revealing Cognitive Bias in Human-Subjects Red-Team Cyber Range Operations](http://arxiv.org/pdf/2508.20963v1)

Authors: Brandon Beltz, Jim Doty, Yvonne Fonken, Nikolos Gurney, Brett Israelsen, Nathan Lau, Stacy Marsella, Rachelle Thomas, Stoney Trent, Peggy Wu, Ya-Ting Yang, Quanyan Zhu

We present three large-scale human-subjects red-team cyber range datasets
from the Guarding Against Malicious Biased Threats (GAMBiT) project. Across
Experiments 1-3 (July 2024-March 2025), 19-20 skilled attackers per experiment
conducted two 8-hour days of self-paced operations in a simulated enterprise
network (SimSpace Cyber Force Platform) while we captured multi-modal data:
self-reports (background, demographics, psychometrics), operational notes,
terminal histories, keylogs, network packet captures (PCAP), and NIDS alerts
(Suricata). Each participant began from a standardized Kali Linux VM and
pursued realistic objectives (e.g., target discovery and data exfiltration)
under controlled constraints. Derivative curated logs and labels are included.
The combined release supports research on attacker behavior modeling,
bias-aware analytics, and method benchmarking. Data are available via IEEE
Dataport entries for Experiments 1-3.

### 2. [Balancing Profit and Traveller Acceptance in Ride-Pooling Personalised Fares](http://arxiv.org/pdf/2508.20723v1)

Authors: Michal Bujak, Rafal Kucharski

Ride-pooling systems, to succeed, must provide an attractive service, namely
compensate perceived costs with an appealing price. However, because of a
strong heterogeneity in a value-of-time, each traveller has his own acceptable
price, unknown to the operator. Here, we show that individual acceptance levels
can be learned by the operator (over $90\%$ accuracy for pooled travellers in
$10$ days) to optimise personalised fares. We propose an adaptive pricing
policy, where every day the operator constructs an offer that progressively
meets travellers' expectations and attracts a growing demand. Our results
suggest that operators, by learning behavioural traits of individual
travellers, may improve performance not only for travellers (increased utility)
but also for themselves (increased profit). Moreover, such knowledge allows the
operator to remove inefficient pooled rides and focus on attractive and
profitable combinations.

### Human-Computer Interaction

### 1. [Identifying Framing Practices in Visualization Design Through Practitioner Reflections](http://arxiv.org/pdf/2508.20383v1)

Authors: Prakash Shukla, Paul Parsons

Framing -- how designers define and reinterpret problems, shape narratives,
and guide audience understanding -- is central to design practice. Yet in
visualization research, framing has been examined mostly through its rhetorical
and perceptual effects on audiences, leaving its role in the design process
underexplored. This study addresses that gap by analyzing publicly available
podcasts and book chapters in which over 80 professional visualization
designers reflect on their work. We find that framing is a pervasive, iterative
activity, evident in scoping problems, interpreting data, aligning with
stakeholder goals, and shaping narrative direction. Our analysis identifies the
conditions that trigger reframing and the strategies practitioners use to
navigate uncertainty and guide design. These findings position framing as a
core dimension of visualization practice and underscore the need for research
and education to support the interpretive and strategic judgment that
practitioners exercise throughout the design process.

### 2. [What is "Spatial" about Spatial Computing?](http://arxiv.org/pdf/2508.20477v1)

Authors: Yibo Wang, Yuhan Luo, Janghee Cho, Junnan Yu

Recent advancements in geographic information systems and mixed reality
technologies have positioned spatial computing as a transformative paradigm in
computational science. However, the field remains conceptually fragmented, with
diverse interpretations across disciplines like Human-Computer Interaction,
Geographic Information Science, and Computer Science, which hinders a
comprehensive understanding of spatial computing and poses challenges for its
coherent advancement and interdisciplinary integration. In this paper, we trace
the origins and historical evolution of spatial computing and examine how
"spatial" is understood, identifying two schools of thought: "spatial" as the
contextual understanding of space, where spatial data guides interaction in the
physical world; and "spatial" as a mixed space for interaction, emphasizing the
seamless integration of physical and digital environments to enable embodied
engagement. By synthesizing these perspectives, we propose spatial computing as
a computational paradigm that redefines the interplay between environment,
computation, and human experience, offering a holistic lens to enhance its
conceptual clarity and inspire future technological innovations that support
meaningful interactions with and shaping of environments.

### 3. [VisiTrail: A Cognitive Visualization Tool for Time-Series Analysis of Eye Tracking Data from Attention Game](http://arxiv.org/pdf/2508.20522v1)

Authors: Abdul Rehman, Ilona Heldal, Jerry Chun-Wei Lin

Eye Tracking (ET) can help to understand visual attention and cognitive
processes in interactive environments. In attention tasks, distinguishing
between relevant target objects and distractors is crucial for effective
performance, yet the underlying gaze patterns that drive successful task
completion remain incompletely understood. Traditional gaze analyses lack
comprehensive insights into the temporal dynamics of attention allocation and
the relationship between gaze behavior and task performance. When applied to
complex visual search scenarios, current gaze analysis methods face several
limitations, including the isolation of measurements, visual stability, search
efficiency, and the decision-making processes involved in these scenarios. This
paper proposes an analysis tool that considers time series for eye tracking
data from task performance and also gaze measures (fixations, saccades and
smooth pursuit); temporal pattern analysis that reveals how attention evolves
throughout task performance; object-click sequence tracking that directly links
visual attention to user actions; and performance metrics that quantify both
accuracy and efficiency. This tool provides comprehensive visualization
techniques that make complex patterns of stimuli and gaze connections
interpretable.

### 4. [Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent](http://arxiv.org/pdf/2508.20585v1)

Authors: Seokho Jin, Manseo Kim, Sungho Byun, Hansol Kim, Jungmin Lee, Sujeong Baek, Semi Kim, Sanghum Park, Sung Park

Reflective journaling often lacks personalization and fails to engage
Generation Alpha and Z, who prefer visually immersive and fast-paced
interactions over traditional text-heavy methods. Visual storytelling enhances
emotional recall and offers an engaging way to process personal expe- riences.
Designed with these digital-native generations in mind, this paper introduces
Persode, a journaling system that integrates personalized onboarding,
memory-aware conversational agents, and automated visual storytelling. Persode
captures user demographics and stylistic preferences through a tailored
onboarding process, ensuring outputs resonate with individual identities. Using
a Retrieval-Augmented Generation (RAG) framework, it prioritizes emotionally
significant memories to provide meaningful, context-rich interactions.
Additionally, Persode dynamically transforms user experiences into visually
engaging narratives by generating prompts for advanced text-to-image models,
adapting characters, backgrounds, and styles to user preferences. By addressing
the need for personalization, visual engagement, and responsiveness, Persode
bridges the gap between traditional journaling and the evolving preferences of
Gen Alpha and Z.

### 5. [Schema-Guided Response Generation using Multi-Frame Dialogue State for Motivational Interviewing Systems](http://arxiv.org/pdf/2508.20635v1)

Authors: Jie Zeng, Yukiko I. Nakano

The primary goal of Motivational Interviewing (MI) is to help clients build
their own motivation for behavioral change. To support this in dialogue
systems, it is essential to guide large language models (LLMs) to generate
counselor responses aligned with MI principles. By employing a schema-guided
approach, this study proposes a method for updating multi-frame dialogue states
and a strategy decision mechanism that dynamically determines the response
focus in a manner grounded in MI principles. The proposed method was
implemented in a dialogue system and evaluated through a user study. Results
showed that the proposed system successfully generated MI-favorable responses
and effectively encouraged the user's (client's) deliberation by asking
eliciting questions.

### 6. [MedFoundationHub: A Lightweight and Secure Toolkit for Deploying Medical Vision Language Foundation Models](http://arxiv.org/pdf/2508.20345v1)

Authors: Xiao Li, Yanfan Zhu, Ruining Deng, Wei-Qi Wei, Yu Wang, Shilin Zhao, Yaohong Wang, Haichun Yang, Yuankai Huo

Recent advances in medical vision-language models (VLMs) open up remarkable
opportunities for clinical applications such as automated report generation,
copilots for physicians, and uncertainty quantification. However, despite their
promise, medical VLMs introduce serious security concerns, most notably risks
of Protected Health Information (PHI) exposure, data leakage, and vulnerability
to cyberthreats - which are especially critical in hospital environments. Even
when adopted for research or non-clinical purposes, healthcare organizations
must exercise caution and implement safeguards. To address these challenges, we
present MedFoundationHub, a graphical user interface (GUI) toolkit that: (1)
enables physicians to manually select and use different models without
programming expertise, (2) supports engineers in efficiently deploying medical
VLMs in a plug-and-play fashion, with seamless integration of Hugging Face
open-source models, and (3) ensures privacy-preserving inference through
Docker-orchestrated, operating system agnostic deployment. MedFoundationHub
requires only an offline local workstation equipped with a single NVIDIA A6000
GPU, making it both secure and accessible within the typical resources of
academic research labs. To evaluate current capabilities, we engaged
board-certified pathologists to deploy and assess five state-of-the-art VLMs
(Google-MedGemma3-4B, Qwen2-VL-7B-Instruct, Qwen2.5-VL-7B-Instruct, and
LLaVA-1.5-7B/13B). Expert evaluation covered colon cases and renal cases,
yielding 1015 clinician-model scoring events. These assessments revealed
recurring limitations, including off-target answers, vague reasoning, and
inconsistent pathology terminology.

### 7. [Human-Centered Design for Connected Automation: Predicting Pedestrian Crossing Intentions](http://arxiv.org/pdf/2508.20464v1)

Authors: Sanaz Motamedi, Viktoria Marcus, Griffin Pitts

Road traffic remains a leading cause of death worldwide, with pedestrians and
other vulnerable road users accounting for over half of the 1.19 million annual
fatalities, much of it due to human error. Level-5 automated driving systems
(ADSs), capable of full self-driving without human oversight, have the
potential to reduce these incidents. However, their effectiveness depends not
only on automation performance but also on their ability to communicate intent
and coordinate safely with pedestrians in the absence of traditional driver
cues. Understanding how pedestrians interpret and respond to ADS behavior is
therefore critical to the development of connected vehicle systems. This study
extends the Theory of Planned Behavior (TPB) by incorporating four external
factors (i.e. safety, trust, compatibility, and understanding) to model
pedestrian decision-making in road-crossing scenarios involving level-5 ADSs.
Using data from an online survey (n = 212), results show that perceived
behavioral control, attitude, and social information significantly predict
pedestrians' crossing intentions. External factors, particularly perceived
safety and understanding, strongly influence these constructs. Findings provide
actionable insights for designing external human-machine interfaces (eHMIs) and
cooperative V2X communication strategies that support safe, transparent
interactions between automated vehicles and pedestrians. This work contributes
to the development of inclusive, human-centered connected mobility systems.

### 8. [Understanding, Protecting, and Augmenting Human Cognition with Generative AI: A Synthesis of the CHI 2025 Tools for Thought Workshop](http://arxiv.org/pdf/2508.21036v1)

Authors: Lev Tankelevitch, Elena L. Glassman, Jessica He, Aniket Kittur, Mina Lee, Srishti Palani, Advait Sarkar, Gonzalo Ramos, Yvonne Rogers, Hari Subramonyam

Generative AI (GenAI) radically expands the scope and capability of
automation for work, education, and everyday tasks, a transformation posing
both risks and opportunities for human cognition. How will human cognition
change, and what opportunities are there for GenAI to augment it? Which
theories, metrics, and other tools are needed to address these questions? The
CHI 2025 workshop on Tools for Thought aimed to bridge an emerging science of
how the use of GenAI affects human thought, from metacognition to critical
thinking, memory, and creativity, with an emerging design practice for building
GenAI tools that both protect and augment human thought. Fifty-six researchers,
designers, and thinkers from across disciplines as well as industry and
academia, along with 34 papers and portfolios, seeded a day of discussion,
ideation, and community-building. We synthesize this material here to begin
mapping the space of research and design opportunities and to catalyze a
multidisciplinary community around this pressing area of research.

### 9. [ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents](http://arxiv.org/pdf/2508.20973v1)

Authors: Tianjian Liu, Fanqi Wan, Jiajian Guo, Xiaojun Quan

Proactive dialogue has emerged as a critical and challenging research problem
in advancing large language models (LLMs). Existing works predominantly focus
on domain-specific or task-oriented scenarios, which leads to fragmented
evaluations and limits the comprehensive exploration of models' proactive
conversation abilities. In this work, we propose ProactiveEval, a unified
framework designed for evaluating proactive dialogue capabilities of LLMs. This
framework decomposes proactive dialogue into target planning and dialogue
guidance, establishing evaluation metrics across various domains. Moreover, it
also enables the automatic generation of diverse and challenging evaluation
data. Based on the proposed framework, we develop 328 evaluation environments
spanning 6 distinct domains. Through experiments with 22 different types of
LLMs, we show that DeepSeek-R1 and Claude-3.7-Sonnet exhibit exceptional
performance on target planning and dialogue guidance tasks, respectively.
Finally, we investigate how reasoning capabilities influence proactive
behaviors and discuss their implications for future model development.

### 10. [OnGoal: Tracking and Visualizing Conversational Goals in Multi-Turn Dialogue with Large Language Models](http://arxiv.org/pdf/2508.21061v1)

Authors: Adam Coscia, Shunan Guo, Eunyee Koh, Alex Endert

As multi-turn dialogues with large language models (LLMs) grow longer and
more complex, how can users better evaluate and review progress on their
conversational goals? We present OnGoal, an LLM chat interface that helps users
better manage goal progress. OnGoal provides real-time feedback on goal
alignment through LLM-assisted evaluation, explanations for evaluation results
with examples, and overviews of goal progression over time, enabling users to
navigate complex dialogues more effectively. Through a study with 20
participants on a writing task, we evaluate OnGoal against a baseline chat
interface without goal tracking. Using OnGoal, participants spent less time and
effort to achieve their goals while exploring new prompting strategies to
overcome miscommunication, suggesting tracking and visualizing goals can
enhance engagement and resilience in LLM dialogues. Our findings inspired
design implications for future LLM chat interfaces that improve goal
communication, reduce cognitive load, enhance interactivity, and enable
feedback to improve LLM performance.

### Information Retrieval

### 1. [Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation](http://arxiv.org/pdf/2508.20359v1)

Authors: Shijia Wang, Tianpei Ouyang, Qiang Xiao, Dongjing Wang, Yintao Ren, Songpei Xu, Da Guo, Chuanjiang Luo

In music recommendation systems, multimodal interest learning is pivotal,
which allows the model to capture nuanced preferences, including textual
elements such as lyrics and various musical attributes such as different
instruments and melodies. Recently, methods that incorporate multimodal content
features through semantic IDs have achieved promising results. However,
existing methods suffer from two critical limitations: 1) intra-modal semantic
degradation, where residual-based quantization processes gradually decouple
discrete IDs from original content semantics, leading to semantic drift; and 2)
inter-modal modeling gaps, where traditional fusion strategies either overlook
modal-specific details or fail to capture cross-modal correlations, hindering
comprehensive user interest modeling. To address these challenges, we propose a
novel multimodal recommendation framework with two stages. In the first stage,
our Progressive Semantic Residual Quantization (PSRQ) method generates
modal-specific and modal-joint semantic IDs by explicitly preserving the prefix
semantic feature. In the second stage, to model multimodal interest of users, a
Multi-Codebook Cross-Attention (MCCA) network is designed to enable the model
to simultaneously capture modal-specific interests and perceive cross-modal
correlations. Extensive experiments on multiple real-world datasets demonstrate
that our framework outperforms state-of-the-art baselines. This framework has
been deployed on one of China's largest music streaming platforms, and online
A/B tests confirm significant improvements in commercial metrics, underscoring
its practical value for industrial-scale recommendation systems.

### 2. [A Case Study of Balanced Query Recommendation on Wikipedia](http://arxiv.org/pdf/2508.20399v1)

Authors: Harshit Mishra, Sucheta Soundarajan

Modern IR systems are an extremely important tool for seeking information. In
addition to search, such systems include a number of query reformulation
methods, such as query expansion and query recommendations, to provide high
quality results. However, results returned by such methods sometimes exhibit
undesirable or wrongful bias with respect to protected categories such as
gender or race. Our earlier work considered the problem of balanced query
recommendation, where instead of re-ranking a list of results based on fairness
measures, the goal was to suggest queries that are relevant to a user's search
query but exhibit less bias than the original query. In this work, we present a
case study of BalancedQR using an extension of BalancedQR that handles biases
in multiple dimensions. It employs a Pareto front approach that finds balanced
queries, optimizing for multiple objectives such as gender bias and regional
bias, along with the relevance of returned results. We evaluate the extended
version of BalancedQR on a Wikipedia dataset.Our results demonstrate the
effectiveness of our extension to BalancedQR framework and highlight the
significant impact of subtle query wording,linguistic choice on retrieval.

### 3. [Fact or Facsimile? Evaluating the Factual Robustness of Modern Retrievers](http://arxiv.org/pdf/2508.20408v1)

Authors: Haoyu Wu, Qingcheng Zeng, Kaize Ding

Dense retrievers and rerankers are central to retrieval-augmented generation
(RAG) pipelines, where accurately retrieving factual information is crucial for
maintaining system trustworthiness and defending against RAG poisoning.
However, little is known about how much factual competence these components
inherit or lose from the large language models (LLMs) they are based on. We
pair 12 publicly released embedding checkpoints with their original base LLMs
and evaluate both sets on a factuality benchmark. Across every model evaluated,
the embedding variants achieve markedly lower accuracy than their bases, with
absolute drops ranging from 12 to 43 percentage points (median 28 pts) and
typical retriever accuracies collapsing into the 25-35 % band versus the 60-70
% attained by the generative models. This degradation intensifies under a more
demanding condition: when the candidate pool per question is expanded from four
options to one thousand, the strongest retriever's top-1 accuracy falls from 33
% to 26 %, revealing acute sensitivity to distractor volume. Statistical tests
further show that, for every embedding model, cosine-similarity scores between
queries and correct completions are significantly higher than those for
incorrect ones (p < 0.01), indicating decisions driven largely by surface-level
semantic proximity rather than factual reasoning. To probe this weakness, we
employed GPT-4.1 to paraphrase each correct completion, creating a rewritten
test set that preserved factual truth while masking lexical cues, and observed
that over two-thirds of previously correct predictions flipped to wrong,
reducing overall accuracy to roughly one-third of its original level. Taken
together, these findings reveal a systematic trade-off introduced by
contrastive learning for retrievers: gains in semantic retrieval are paid for
with losses in parametric factual knowledge......

### 4. [Multistakeholder Fairness in Tourism: What can Algorithms learn from Tourism Management?](http://arxiv.org/pdf/2508.20496v1)

Authors: Peter Muellner, Anna Schreuer, Simone Kopeinik, Bernhard Wieser, Dominik Kowald

Algorithmic decision-support systems, i.e., recommender systems, are popular
digital tools that help tourists decide which places and attractions to
explore. However, algorithms often unintentionally direct tourist streams in a
way that negatively affects the environment, local communities, or other
stakeholders. This issue can be partly attributed to the computer science
community's limited understanding of the complex relationships and trade-offs
among stakeholders in the real world.
  In this work, we draw on the practical findings and methods from tourism
management to inform research on multistakeholder fairness in algorithmic
decision-support. Leveraging a semi-systematic literature review, we synthesize
literature from tourism management as well as literature from computer science.
Our findings suggest that tourism management actively tries to identify the
specific needs of stakeholders and utilizes qualitative, inclusive and
participatory methods to study fairness from a normative and holistic research
perspective. In contrast, computer science lacks sufficient understanding of
the stakeholder needs and primarily considers fairness through descriptive
factors, such as measureable discrimination, while heavily relying on few
mathematically formalized fairness criteria that fail to capture the
multidimensional nature of fairness in tourism.
  With the results of this work, we aim to illustrate the shortcomings of
purely algorithmic research and stress the potential and particular need for
future interdisciplinary collaboration. We believe such a collaboration is a
fundamental and necessary step to enhance algorithmic decision-support systems
towards understanding and supporting true multistakeholder fairness in tourism.

### 5. [SUMMA: A Multimodal Large Language Model for Advertisement Summarization](http://arxiv.org/pdf/2508.20582v1)

Authors: Weitao Jia, Shuo Yin, Zhoufutu Wen, Han Wang, Zehui Dai, Kun Zhang, Zhenyu Li, Tao Zeng, Xiaohui Lv

Understanding multimodal video ads is crucial for improving query-ad matching
and relevance ranking on short video platforms, enhancing advertising
effectiveness and user experience. However, the effective utilization of
multimodal information with high commercial value still largely constrained by
reliance on highly compressed video embeddings-has long been inadequate. To
address this, we propose SUMMA (the abbreviation of Summarizing MultiModal
Ads), a multimodal model that automatically processes video ads into summaries
highlighting the content of highest commercial value, thus improving their
comprehension and ranking in Douyin search-advertising systems. SUMMA is
developed via a two-stage training strategy-multimodal supervised fine-tuning
followed by reinforcement learning with a mixed reward mechanism-on
domain-specific data containing video frames and ASR/OCR transcripts,
generating commercially valuable and explainable summaries. We integrate
SUMMA-generated summaries into our production pipeline, directly enhancing the
candidate retrieval and relevance ranking stages in real search-advertising
systems. Both offline and online experiments show substantial improvements over
baselines, with online results indicating a statistically significant 1.5%
increase in advertising revenue. Our work establishes a novel paradigm for
condensing multimodal information into representative texts, effectively
aligning visual ad content with user query intent in retrieval and
recommendation scenarios.

### 6. [Addressing Personalized Bias for Unbiased Learning to Rank](http://arxiv.org/pdf/2508.20798v1)

Authors: Zechun Niu, Lang Mei, Liu Yang, Ziyuan Zhao, Qiang Yan, Jiaxin Mao, Ji-Rong Wen

Unbiased learning to rank (ULTR), which aims to learn unbiased ranking models
from biased user behavior logs, plays an important role in Web search. Previous
research on ULTR has studied a variety of biases in users' clicks, such as
position bias, presentation bias, and outlier bias. However, existing work
often assumes that the behavior logs are collected from an ``average'' user,
neglecting the differences between different users in their search and browsing
behaviors. In this paper, we introduce personalized factors into the ULTR
framework, which we term the user-aware ULTR problem. Through a formal causal
analysis of this problem, we demonstrate that existing user-oblivious methods
are biased when different users have different preferences over queries and
personalized propensities of examining documents. To address such a
personalized bias, we propose a novel user-aware inverse-propensity-score
estimator for learning-to-rank objectives. Specifically, our approach models
the distribution of user browsing behaviors for each query and aggregates
user-weighted examination probabilities to determine propensities. We
theoretically prove that the user-aware estimator is unbiased under some mild
assumptions and shows lower variance compared to the straightforward way of
calculating a user-dependent propensity for each impression. Finally, we
empirically verify the effectiveness of our user-aware estimator by conducting
extensive experiments on two semi-synthetic datasets and a real-world dataset.

### 7. [Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction](http://arxiv.org/pdf/2508.20865v1)

Authors: Zhuoxing Wei, Qi Liu, Qingchen Xie

In Click-Through Rate (CTR) prediction, the long behavior sequence,
comprising the user's long period of historical interactions with items has a
vital influence on assessing the user's interest in the candidate item.
Existing approaches strike efficiency and effectiveness through a two-stage
paradigm: first retrieving hundreds of candidate-related items and then
extracting interest intensity vector through target attention. However, we
argue that the discrepancy in target attention's relevance distribution between
the retrieved items and the full long behavior sequence inevitably leads to a
performance decline. To alleviate the discrepancy, we propose the Deep Multiple
Quantization Network (DMQN) to process long behavior sequence end-to-end
through compressing the long behavior sequence. Firstly, the entire spectrum of
long behavior sequence will be quantized into multiple codeword sequences based
on multiple independent codebooks. Hierarchical Sequential Transduction Unit is
incorporated to facilitate the interaction of reduced codeword sequences. Then,
attention between the candidate and multiple codeword sequences will output the
interest vector. To enable online serving, intermediate representations of the
codeword sequences are cached, significantly reducing latency. Our extensive
experiments on both industrial and public datasets confirm the effectiveness
and efficiency of DMQN. The A/B test in our advertising system shows that DMQN
improves CTR by 3.5% and RPM by 2.0%.

### 8. [OneRec-V2 Technical Report](http://arxiv.org/pdf/2508.20900v1)

Authors: Guorui Zhou, Hengrui Hu, Hongtao Cheng, Huanjie Wang, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Lu Ren, Liao Yu, Pengfei Zheng, Qiang Luo, Qianqian Wang, Qigen Hu, Rui Huang, Ruiming Tang, Shiyao Wang, Shujie Yang, Tao Wu, Wuchao Li, Xinchen Luo, Xingmei Wang, Yi Su, Yunfan Wu, Zexuan Cheng, Zhanyu Liu, Zixing Zhang, Bin Zhang, Boxuan Wang, Chaoyi Ma, Chengru Song, Chenhui Wang, Chenglong Chu, Di Wang, Dongxue Meng, Dunju Zang, Fan Yang, Fangyu Zhang, Feng Jiang, Fuxing Zhang, Gang Wang, Guowang Zhang, Han Li, Honghui Bao, Hongyang Cao, Jiaming Huang, Jiapeng Chen, Jiaqiang Liu, Jinghui Jia, Kun Gai, Lantao Hu, Liang Zeng, Qiang Wang, Qidong Zhou, Rongzhou Zhang, Shengzhe Wang, Shihui He, Shuang Yang, Siyang Mao, Sui Huang, Tiantian He, Tingting Gao, Wei Yuan, Xiao Liang, Xiaoxiao Xu, Xugang Liu, Yan Wang, Yang Zhou, Yi Wang, Yiwu Liu, Yue Song, Yufei Zhang, Yunfeng Zhao, Zhixin Ling, Ziming Li

Recent breakthroughs in generative AI have transformed recommender systems
through end-to-end generation. OneRec reformulates recommendation as an
autoregressive generation task, achieving high Model FLOPs Utilization. While
OneRec-V1 has shown significant empirical success in real-world deployment, two
critical challenges hinder its scalability and performance: (1) inefficient
computational allocation where 97.66% of resources are consumed by sequence
encoding rather than generation, and (2) limitations in reinforcement learning
relying solely on reward models.
  To address these challenges, we propose OneRec-V2, featuring: (1) Lazy
Decoder-Only Architecture: Eliminates encoder bottlenecks, reducing total
computation by 94% and training resources by 90%, enabling successful scaling
to 8B parameters. (2) Preference Alignment with Real-World User Interactions:
Incorporates Duration-Aware Reward Shaping and Adaptive Ratio Clipping to
better align with user preferences using real-world feedback.
  Extensive A/B tests on Kuaishou demonstrate OneRec-V2's effectiveness,
improving App Stay Time by 0.467%/0.741% while balancing multi-objective
recommendations. This work advances generative recommendation scalability and
alignment with real-world feedback, representing a step forward in the
development of end-to-end recommender systems.

### 9. [MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever](http://arxiv.org/pdf/2508.20400v1)

Authors: Yijia Sun, Shanshan Huang, Linxiao Che, Haitao Lu, Qiang Luo, Kun Gai, Guorui Zhou

Modern industrial recommendation systems encounter a core challenge of
multi-stage optimization misalignment: a significant semantic gap exists
between the multi-objective optimization paradigm widely used in the ranking
phase and the single-objective modeling in the retrieve phase. Although the
mainstream industry solution achieves multi-objective coverage through parallel
multi-path single-objective retrieval, this approach leads to linear growth of
training and serving resources with the number of objectives and has inherent
limitations in handling loosely coupled objectives. This paper proposes the
MPFormer, a dynamic multi-task Transformer framework, which systematically
addresses the aforementioned issues through three innovative mechanisms. First,
an objective-conditioned transformer that jointly encodes user behavior
sequences and multi-task semantics through learnable attention modulation;
second, personalized target weights are introduced to achieve dynamic
adjustment of retrieval results; finally, user personalization information is
incorporated into token representations and the Transformer structure to
further enhance the model's representation ability. This framework has been
successfully integrated into Kuaishou short video recommendation system, stably
serving over 400 million daily active users. It significantly improves user
daily engagement and system operational efficiency. Practical deployment
verification shows that, compared with traditional solutions, it effectively
optimizes the iterative paradigm of multi-objective retrieval while maintaining
service response speed, providing a scalable multi-objective solution for
industrial recommendation systems.

### 10. [Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting](http://arxiv.org/pdf/2508.20401v1)

Authors: Alexandre Andre, Gauthier Roy, Eva Dyer, Kai Wang

Large Language Models (LLMs) are increasingly used for recommendation tasks
due to their general-purpose capabilities. While LLMs perform well in
rich-context settings, their behavior in cold-start scenarios, where only
limited signals such as age, gender, or language are available, raises fairness
concerns because they may rely on societal biases encoded during pretraining.
We introduce a benchmark specifically designed to evaluate fairness in
zero-context recommendation. Our modular pipeline supports configurable
recommendation domains and sensitive attributes, enabling systematic and
flexible audits of any open-source LLM. Through evaluations of state-of-the-art
models (Gemma 3 and Llama 3.2), we uncover consistent biases across
recommendation domains (music, movies, and colleges) including gendered and
cultural stereotypes. We also reveal a non-linear relationship between model
size and fairness, highlighting the need for nuanced analysis.

### Machine Learning

### 1. [FORGE: Foundational Optimization Representations from Graph Embeddings](http://arxiv.org/pdf/2508.20330v1)

Authors: Zohair Shafi, Serdar Kadioglu

Combinatorial optimization problems are ubiquitous in science and
engineering, yet learning-based approaches to accelerate their solution often
require solving a large number of hard-to-solve optimization instances to
collect training data, incurring significant computational overhead. Existing
methods require training dedicated models for each problem distribution for
each downstream task, severely limiting their scalability and generalization.
In this work, we introduce Forge, a method of pre-training a vector-quantized
graph autoencoder on a large and diverse collection of mixed-integer
programming (MIP) instances in an unsupervised fashion without dependency on
their solution. The vector quantization process creates discrete code
assignments that act as a vocabulary to represent optimization instances. We
evaluate our approach under both supervised and unsupervised settings. For the
unsupervised setting, we demonstrate that Forge embeddings effectively
differentiate and cluster unseen instances. For the supervised setting, we
fine-tune Forge embeddings and show that a single model predicts both the
variables for warm-starts and integrality gaps for cut-generation across
multiple problem type distributions. Both predictions help improve performance
of a state-of-the-art, commercial optimization solver. Finally, we release our
code and pre-trained Forge weights to encourage further research and practical
use of instance-level MIP embeddings at https://github.com/skadio/forge/

### 2. [Dynamic Synthetic Controls vs. Panel-Aware Double Machine Learning for Geo-Level Marketing Impact Estimation](http://arxiv.org/pdf/2508.20335v1)

Authors: Sang Su Lee, Vineeth Loganathan, Vijay Raghavan

Accurately quantifying geo-level marketing lift in two-sided marketplaces is
challenging: the Synthetic Control Method (SCM) often exhibits high power yet
systematically under-estimates effect size, while panel-style Double Machine
Learning (DML) is seldom benchmarked against SCM. We build an open, fully
documented simulator that mimics a typical large-scale geo roll-out: N_unit
regional markets are tracked for T_pre weeks before launch and for a further
T_post-week campaign window, allowing all key parameters to be varied by the
user and probe both families under five stylized stress tests: 1) curved
baseline trends, 2) heterogeneous response lags, 3) treated-biased shocks, 4) a
non-linear outcome link, and 5) a drifting control group trend.
  Seven estimators are evaluated: three standard Augmented SCM (ASC) variants
and four panel-DML flavors (TWFE, CRE/Mundlak, first-difference, and
within-group). Across 100 replications per scenario, ASC models consistently
demonstrate severe bias and near-zero coverage in challenging scenarios
involving nonlinearities or external shocks. By contrast, panel-DML variants
dramatically reduce this bias and restore nominal 95%-CI coverage, proving far
more robust.
  The results indicate that while ASC provides a simple baseline, it is
unreliable in common, complex situations. We therefore propose a
'diagnose-first' framework where practitioners first identify the primary
business challenge (e.g., nonlinear trends, response lags) and then select the
specific DML model best suited for that scenario, providing a more robust and
reliable blueprint for analyzing geo-experiments.

### 3. [Developing a Multi-Modal Machine Learning Model For Predicting Performance of Automotive Hood Frames](http://arxiv.org/pdf/2508.20358v1)

Authors: Abhishek Indupally, Satchit Ramnath

Is there a way for a designer to evaluate the performance of a given hood
frame geometry without spending significant time on simulation setup? This
paper seeks to address this challenge by developing a multimodal
machine-learning (MMML) architecture that learns from different modalities of
the same data to predict performance metrics. It also aims to use the MMML
architecture to enhance the efficiency of engineering design processes by
reducing reliance on computationally expensive simulations. The proposed
architecture accelerates design exploration, enabling rapid iteration while
maintaining high-performance standards, especially in the concept design phase.
The study also presents results that show that by combining multiple data
modalities, MMML outperforms traditional single-modality approaches. Two new
frame geometries, not part of the training dataset, are also used for
prediction using the trained MMML model to showcase the ability to generalize
to unseen frame models. The findings underscore MMML's potential in
supplementing traditional simulation-based workflows, particularly in the
conceptual design phase, and highlight its role in bridging the gap between
machine learning and real-world engineering applications. This research paves
the way for the broader adoption of machine learning techniques in engineering
design, with a focus on refining multimodal approaches to optimize structural
development and accelerate the design cycle.

### 4. [BiListing: Modality Alignment for Listings](http://arxiv.org/pdf/2508.20396v1)

Authors: Guillaume Guy, Mihajlo Grbovic, Chun How Tan, Han Zhao

Airbnb is a leader in offering travel accommodations. Airbnb has historically
relied on structured data to understand, rank, and recommend listings to guests
due to the limited capabilities and associated complexity arising from
extracting meaningful information from text and images. With the rise of
representation learning, leveraging rich information from text and photos has
become easier. A popular approach has been to create embeddings for text
documents and images to enable use cases of computing similarities between
listings or using embeddings as features in an ML model.
  However, an Airbnb listing has diverse unstructured data: multiple images,
various unstructured text documents such as title, description, and reviews,
making this approach challenging. Specifically, it is a non-trivial task to
combine multiple embeddings of different pieces of information to reach a
single representation.
  This paper proposes BiListing, for Bimodal Listing, an approach to align text
and photos of a listing by leveraging large-language models and pretrained
language-image models. The BiListing approach has several favorable
characteristics: capturing unstructured data into a single embedding vector per
listing and modality, enabling zero-shot capability to search inventory
efficiently in user-friendly semantics, overcoming the cold start problem, and
enabling listing-to-listing search along a single modality, or both.
  We conducted offline and online tests to leverage the BiListing embeddings in
the Airbnb search ranking model, and successfully deployed it in production,
achieved 0.425% of NDCB gain, and drove tens of millions in incremental
revenue.

### 5. [Rethinking Transformer Connectivity: TLinFormer, A Path to Exact, Full Context-Aware Linear Attention](http://arxiv.org/pdf/2508.20407v1)

Authors: Zhongpan Tang

The Transformer architecture has become a cornerstone of modern artificial
intelligence, but its core self-attention mechanism suffers from a complexity
bottleneck that scales quadratically with sequence length, severely limiting
its application in long-sequence tasks. To address this challenge, existing
linear attention methods typically sacrifice model performance by relying on
data-agnostic kernel approximations or restrictive context selection. This
paper returns to the first principles of connectionism, starting from the
topological structure of information flow, to introduce a novel linear
attention architecture-\textbf{TLinFormer}. By reconfiguring neuron connection
patterns, TLinFormer achieves strict linear complexity while computing exact
attention scores and ensuring information flow remains aware of the full
historical context. This design aims to bridge the performance gap prevalent
between existing efficient attention methods and standard attention. Through a
series of experiments, we systematically evaluate the performance of TLinFormer
against a standard Transformer baseline on long-sequence inference tasks. The
results demonstrate that TLinFormer exhibits overwhelming advantages in key
metrics such as \textbf{inference latency}, \textbf{KV cache efficiency},
\textbf{memory footprint}, and \textbf{overall speedup}.

### 6. [Structure-aware Hypergraph Transformer for Diagnosis Prediction in Electronic Health Records](http://arxiv.org/pdf/2508.20500v1)

Authors: Haiyan Wang, Ye Yuan

Electronic Health Records (EHR) systematically organize patient health data
through standardized medical codes, serving as a comprehensive and invaluable
source for predictive modeling. Graph neural networks (GNNs) have demonstrated
effectiveness in modeling interactions between medical codes within EHR.
However, existing GNN-based methods are inadequate due to: a) their reliance on
pairwise relations fails to capture the inherent higher-order dependencies in
clinical data, and b) the localized message-passing scheme limits
representation power. To address these issues, this paper proposes a novel
Structure-aware HyperGraph Transformer (SHGT) framework following three-fold
ideas: a) employing a hypergraph structural encoder to capture higher-order
interactions among medical codes, b) integrating the Transformer architecture
to reason over the entire hypergraph, and c) designing a tailored loss function
incorporating hypergraph reconstruction to preserve the hypergraph's original
structure. Experiments on real-world EHR datasets demonstrate that the proposed
SHGT outperforms existing state-of-the-art models on diagnosis prediction.

### 7. [Khiops: An End-to-End, Frugal AutoML and XAI Machine Learning Solution for Large, Multi-Table Databases](http://arxiv.org/pdf/2508.20519v1)

Authors: Marc Boullé, Nicolas Voisine, Bruno Guerraz, Carine Hue, Felipe Olmos, Vladimir Popescu, Stéphane Gouache, Stéphane Bouget, Alexis Bondu, Luc Aurelien Gauthier, Yassine Nair Benrekia, Fabrice Clérot, Vincent Lemaire

Khiops is an open source machine learning tool designed for mining large
multi-table databases. Khiops is based on a unique Bayesian approach that has
attracted academic interest with more than 20 publications on topics such as
variable selection, classification, decision trees and co-clustering. It
provides a predictive measure of variable importance using discretisation
models for numerical data and value clustering for categorical data. The
proposed classification/regression model is a naive Bayesian classifier
incorporating variable selection and weight learning. In the case of
multi-table databases, it provides propositionalisation by automatically
constructing aggregates. Khiops is adapted to the analysis of large databases
with millions of individuals, tens of thousands of variables and hundreds of
millions of records in secondary tables. It is available on many environments,
both from a Python library and via a user interface.

### 8. [Theoretical foundations of the integral indicator application in hyperparametric optimization](http://arxiv.org/pdf/2508.20550v1)

Authors: Roman S. Kulshin, Anatoly A. Sidorov

The article discusses the concept of hyperparametric optimization of
recommendation algorithms using an integral assessment that combines various
performance indicators into a single consolidated criterion. This approach is
opposed to traditional methods of setting up a single metric and allows you to
achieve a balance between accuracy, ranking quality, variety of output and the
resource intensity of algorithms. The theoretical significance of the research
lies in the development of a universal multi-criteria optimization tool that is
applicable not only in recommendation systems, but also in a wide range of
machine learning and data analysis tasks.

### 9. [Local Virtual Nodes for Alleviating Over-Squashing in Graph Neural Networks](http://arxiv.org/pdf/2508.20597v1)

Authors: Tuğrul Hasan Karabulut, İnci M. Baytaş

Over-squashing is a challenge in training graph neural networks for tasks
involving long-range dependencies. In such tasks, a GNN's receptive field
should be large enough to enable communication between distant nodes. However,
gathering information from a wide range of neighborhoods and squashing its
content into fixed-size node representations makes message-passing vulnerable
to bottlenecks. Graph rewiring and adding virtual nodes are commonly studied
remedies that create additional pathways around bottlenecks to mitigate
over-squashing. However, these techniques alter the input graph's global
topology and disrupt the domain knowledge encoded in the original graph
structure, both of which could be essential to specific tasks and domains. This
study presents Local Virtual Nodes (LVN) with trainable embeddings to alleviate
the effects of over-squashing without significantly corrupting the global
structure of the input graph. The position of the LVNs is determined by the
node centrality, which indicates the existence of potential bottlenecks. Thus,
the proposed approach aims to improve the connectivity in the regions with
likely bottlenecks. Furthermore, trainable LVN embeddings shared across
selected central regions facilitate communication between distant nodes without
adding more layers. Extensive experiments on benchmark datasets demonstrate
that LVNs can enhance structural connectivity and significantly improve
performance on graph and node classification tasks. The code can be found at
https://github.com/ALLab-Boun/LVN/}{https://github.com/ALLab-Boun/LVN/.

### 10. [VarDiU: A Variational Diffusive Upper Bound for One-Step Diffusion Distillation](http://arxiv.org/pdf/2508.20646v1)

Authors: Leyang Wang, Mingtian Zhang, Zijing Ou, David Barber

Recently, diffusion distillation methods have compressed thousand-step
teacher diffusion models into one-step student generators while preserving
sample quality. Most existing approaches train the student model using a
diffusive divergence whose gradient is approximated via the student's score
function, learned through denoising score matching (DSM). Since DSM training is
imperfect, the resulting gradient estimate is inevitably biased, leading to
sub-optimal performance. In this paper, we propose VarDiU (pronounced
/va:rdju:/), a Variational Diffusive Upper Bound that admits an unbiased
gradient estimator and can be directly applied to diffusion distillation. Using
this objective, we compare our method with Diff-Instruct and demonstrate that
it achieves higher generation quality and enables a more efficient and stable
training procedure for one-step diffusion distillation.

### Neural and Evolutionary Computing

### 1. [Ecological Cycle Optimizer: A novel nature-inspired metaheuristic algorithm for global optimization](http://arxiv.org/pdf/2508.20458v1)

Authors: Boyu Ma, Jiaxiao Shi, Yiming Ji, Zhengpu Wang

This article proposes the Ecological Cycle Optimizer (ECO), a novel
metaheuristic algorithm inspired by energy flow and material cycling in
ecosystems. ECO draws an analogy between the dynamic process of solving
optimization problems and ecological cycling. Unique update strategies are
designed for the producer, consumer and decomposer, aiming to enhance the
balance between exploration and exploitation processes. Through these
strategies, ECO is able to achieve the global optimum, simulating the evolution
of an ecological system toward its optimal state of stability and balance.
Moreover, the performance of ECO is evaluated against five highly cited
algorithms-CS, HS, PSO, GWO, and WOA-on 23 classical unconstrained optimization
problems and 24 constrained optimization problems from IEEE CEC-2006 test
suite, verifying its effectiveness in addressing various global optimization
tasks. Furthermore, 50 recently developed metaheuristic algorithms are selected
to form the algorithm pool, and comprehensive experiments are conducted on IEEE
CEC-2014 and CEC-2017 test suites. Among these, five top-performing algorithms,
namely ARO, CFOA, CSA, WSO, and INFO, are chosen for an in-depth comparison
with the ECO on the IEEE CEC-2020 test suite, verifying the ECO's exceptional
optimization performance. Finally, in order to validate the practical
applicability of ECO in complex real-world problems, five state-of-the-art
algorithms, including NSM-SFS, FDB-SFS, FDB-AGDE, L-SHADE, and LRFDB-COA, along
with four best-performing algorithms from the "CEC2020 competition on
real-world single objective constrained optimization", namely SASS, sCMAgES,
EnMODE, and COLSHADE, are selected for comparative experiments on five
engineering problems from CEC-2020-RW test suite (real-world engineering
problems), demonstrating that ECO achieves performance comparable to those of
advanced algorithms.

### 2. [Encoding Tactile Stimuli for Organoid Intelligence in Braille Recognition](http://arxiv.org/pdf/2508.20850v1)

Authors: Tianyi Liu, Hemma Philamore, Benjamin Ward-Cherrier

This study proposes a generalizable encoding strategy that maps tactile
sensor data to electrical stimulation patterns, enabling neural organoids to
perform an open-loop artificial tactile Braille classification task. Human
forebrain organoids cultured on a low-density microelectrode array (MEA) are
systematically stimulated to characterize the relationship between electrical
stimulation parameters (number of pulse, phase amplitude, phase duration, and
trigger delay) and organoid responses, measured as spike activity and spatial
displacement of the center of activity. Implemented on event-based tactile
inputs recorded from the Evetac sensor, our system achieved an average Braille
letter classification accuracy of 61 percent with a single organoid, which
increased significantly to 83 percent when responses from a three-organoid
ensemble were combined. Additionally, the multi-organoid configuration
demonstrated enhanced robustness against various types of artificially
introduced noise. This research demonstrates the potential of organoids as
low-power, adaptive bio-hybrid computational elements and provides a
foundational encoding framework for future scalable bio-hybrid computing
architectures.

### Networking and Internet Architecture

### 1. [Relay Selection in Wireless Networks as Restless Bandits](http://arxiv.org/pdf/2508.20625v1)

Authors: Mandar R. Nalavade, Ravindra S. Tomar, Gaurav S. Kasbekar

We consider a wireless network in which a source node needs to transmit a
large file to a destination node. The direct wireless link between the source
and the destination is assumed to be blocked. Multiple candidate relays are
available to forward packets from the source to the destination. A holding cost
is incurred for each packet stored at every relay in each time slot. The
objective is to design a policy for selecting a relay in each time slot to
which the source attempts to send a packet, so as to minimize the expected
long-run time-averaged total packet holding cost at the relays. This problem is
an instance of the restless multi-armed bandit (RMAB) problem, which is
provably hard to solve. We prove that this relay selection problem is
Whittle-indexable, and propose a method to compute the Whittle index of each
relay in every time slot. In each time slot, our relay selection policy
transmits a packet to the relay with the smallest Whittle index. Using
simulations, we show that the proposed policy outperforms the relay selection
policies proposed in prior work in terms of average cost, delay, as well as
throughput.

### 2. [Digital Twin-Empowered Deep Reinforcement Learning for Intelligent VNF Migration in Edge-Core Networks](http://arxiv.org/pdf/2508.20957v1)

Authors: Faisal Ahmed, Suresh Subramaniam, Motoharu Matsuura, Hiroshi Hasegawa, Shih-Chun Lin

The growing demand for services and the rapid deployment of virtualized
network functions (VNFs) pose significant challenges for achieving low-latency
and energy-efficient orchestration in modern edge-core network infrastructures.
To address these challenges, this study proposes a Digital Twin (DT)-empowered
Deep Reinforcement Learning framework for intelligent VNF migration that
jointly minimizes average end-to-end (E2E) delay and energy consumption. By
formulating the VNF migration problem as a Markov Decision Process and
utilizing the Advantage Actor-Critic model, the proposed framework enables
adaptive and real-time migration decisions. A key innovation of the proposed
framework is the integration of a DT module composed of a multi-task
Variational Autoencoder and a multi-task Long Short-Term Memory network. This
combination collectively simulates environment dynamics and generates
high-quality synthetic experiences, significantly enhancing training efficiency
and accelerating policy convergence. Simulation results demonstrate substantial
performance gains, such as significant reductions in both average E2E delay and
energy consumption, thereby establishing new benchmarks for intelligent VNF
migration in edge-core networks.

### 3. [RANGAN: GAN-empowered Anomaly Detection in 5G Cloud RAN](http://arxiv.org/pdf/2508.20985v1)

Authors: Douglas Liao, Jiping Luo, Jens Vevstad, Nikolaos Pappas

Radio Access Network (RAN) systems are inherently complex, requiring
continuous monitoring to prevent performance degradation and ensure optimal
user experience. The RAN leverages numerous key performance indicators (KPIs)
to evaluate system performance, generating vast amounts of data each second.
This immense data volume can make troubleshooting and accurate diagnosis of
performance anomalies more difficult. Furthermore, the highly dynamic nature of
RAN performance demands adaptive methodologies capable of capturing temporal
dependencies to detect anomalies reliably. In response to these challenges, we
introduce \textbf{RANGAN}, an anomaly detection framework that integrates a
Generative Adversarial Network (GAN) with a transformer architecture. To
enhance the capability of capturing temporal dependencies within the data,
RANGAN employs a sliding window approach during data preprocessing. We
rigorously evaluated RANGAN using the publicly available RAN performance
dataset from the Spotlight project \cite{sun-2024}. Experimental results
demonstrate that RANGAN achieves promising detection accuracy, notably
attaining an F1-score of up to $83\%$ in identifying network contention issues.

### 4. [DSROQ: Dynamic Scheduling and Routing for QoE Management in LEO Satellite Networks](http://arxiv.org/pdf/2508.21047v1)

Authors: Dhiraj Bhattacharjee, Pablo G. Madoery, Abhishek Naik, Halim Yanikomeroglu, Gunes Karabulut Kurt, Stephane Martel, Khaled Ahmed

The modern Internet supports diverse applications with heterogeneous quality
of service (QoS) requirements. Low Earth orbit (LEO) satellite constellations
offer a promising solution to meet these needs, enhancing coverage in rural
areas and complementing terrestrial networks in urban regions. Ensuring QoS in
such networks requires joint optimization of routing, bandwidth allocation, and
dynamic queue scheduling, as traffic handling is critical for maintaining
service performance. This paper formulates a joint routing and bandwidth
allocation problem where QoS requirements are treated as soft constraints,
aiming to maximize user experience. An adaptive scheduling approach is
introduced to prioritize flow-specific QoS needs. We propose a Monte Carlo tree
search (MCTS)-inspired method to solve the NP-hard route and bandwidth
allocation problem, with Lyapunov optimization-based scheduling applied during
reward evaluation. Using the Starlink Phase 1 Version 2 constellation, we
compare end-user experience and fairness between our proposed DSROQ algorithm
and a benchmark scheme. Results show that DSROQ improves both performance
metrics and demonstrates the advantage of joint routing and bandwidth
decisions. Furthermore, we observe that the dominant performance factor shifts
from scheduling to routing and bandwidth allocation as traffic sensitivity
changes from latency-driven to bandwidth-driven.

### 5. [Enhancing Resilience for IoE: A Perspective of Networking-Level Safeguard](http://arxiv.org/pdf/2508.20504v1)

Authors: Guan-Yan Yang, Jui-Ning Chen, Farn Wang, Kuo-Hui Yeh

The Internet of Energy (IoE) integrates IoT-driven digital communication with
power grids to enable efficient and sustainable energy systems. Still, its
interconnectivity exposes critical infrastructure to sophisticated cyber
threats, including adversarial attacks designed to bypass traditional
safeguards. Unlike general IoT risks, IoE threats have heightened public safety
consequences, demanding resilient solutions. From the networking-level
safeguard perspective, we propose a Graph Structure Learning (GSL)-based
safeguards framework that jointly optimizes graph topology and node
representations to resist adversarial network model manipulation inherently.
Through a conceptual overview, architectural discussion, and case study on a
security dataset, we demonstrate GSL's superior robustness over representative
methods, offering practitioners a viable path to secure IoE networks against
evolving attacks. This work highlights the potential of GSL to enhance the
resilience and reliability of future IoE networks for practitioners managing
critical infrastructure. Lastly, we identify key open challenges and propose
future research directions in this novel research area.

### 6. [Microarchitecture Design and Benchmarking of Custom SHA-3 Instruction for RISC-V](http://arxiv.org/pdf/2508.20653v1)

Authors: Alperen Bolat, Sakir Sezer, Kieran McLaughlin, Henry Hui

Integrating cryptographic accelerators into modern CPU architectures presents
unique microarchitectural challenges, particularly when extending instruction
sets with complex and multistage operations. Hardware-assisted cryptographic
instructions, such as Intel's AES-NI and ARM's custom instructions for
encryption workloads, have demonstrated substantial performance improvements.
However, efficient SHA-3 acceleration remains an open problem due to its
distinct permutation-based structure and memory access patterns. Existing
solutions primarily rely on standalone coprocessors or software optimizations,
often avoiding the complexities of direct microarchitectural integration. This
study investigates the architectural challenges of embedding a SHA-3
permutation operation as a custom instruction within a general-purpose
processor, focusing on pipelined simultaneous execution, storage utilization,
and hardware cost. In this paper, we investigated and prototyped a SHA-3 custom
instruction for the RISC-V CPU architecture. Using cycle-accurate GEM5
simulations and FPGA prototyping, our results demonstrate performance
improvements of up to 8.02x for RISC-V optimized SHA-3 software workloads and
up to 46.31x for Keccak-specific software workloads, with only a 15.09%
increase in registers and a 11.51% increase in LUT utilization. These findings
provide critical insights into the feasibility and impact of SHA-3 acceleration
at the microarchitectural level, highlighting practical design considerations
for future cryptographic instruction set extensions.

### Robotics

### 1. [SimShear: Sim-to-Real Shear-based Tactile Servoing](http://arxiv.org/pdf/2508.20561v1)

Authors: Kipp McAdam Freud, Yijiong Lin, Nathan F. Lepora

We present SimShear, a sim-to-real pipeline for tactile control that enables
the use of shear information without explicitly modeling shear dynamics in
simulation. Shear, arising from lateral movements across contact surfaces, is
critical for tasks involving dynamic object interactions but remains
challenging to simulate. To address this, we introduce shPix2pix, a
shear-conditioned U-Net GAN that transforms simulated tactile images absent of
shear, together with a vector encoding shear information, into realistic
equivalents with shear deformations. This method outperforms baseline pix2pix
approaches in simulating tactile images and in pose/shear prediction. We apply
SimShear to two control tasks using a pair of low-cost desktop robotic arms
equipped with a vision-based tactile sensor: (i) a tactile tracking task, where
a follower arm tracks a surface moved by a leader arm, and (ii) a collaborative
co-lifting task, where both arms jointly hold an object while the leader
follows a prescribed trajectory. Our method maintains contact errors within 1
to 2 mm across varied trajectories where shear sensing is essential, validating
the feasibility of sim-to-real shear modeling with rigid-body simulators and
opening new directions for simulation in tactile robotics.

### 2. [Traversing the Narrow Path: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking](http://arxiv.org/pdf/2508.20661v1)

Authors: TianChen Huang, Wei Gao, Runchen Xu, Shiwu Zhang

Traversing narrow beams is challenging for humanoids due to sparse,
safety-critical contacts and the fragility of purely learned policies. We
propose a physically grounded, two-stage framework that couples an XCoM/LIPM
footstep template with a lightweight residual planner and a simple low-level
tracker. Stage-1 is trained on flat ground: the tracker learns to robustly
follow footstep targets by adding small random perturbations to heuristic
footsteps, without any hand-crafted centerline locking, so it acquires stable
contact scheduling and strong target-tracking robustness. Stage-2 is trained in
simulation on a beam: a high-level planner predicts a body-frame residual
(Delta x, Delta y, Delta psi) for the swing foot only, refining the template
step to prioritize safe, precise placement under narrow support while
preserving interpretability. To ease deployment, sensing is kept minimal and
consistent between simulation and hardware: the planner consumes compact,
forward-facing elevation cues together with onboard IMU and joint signals. On a
Unitree G1, our system reliably traverses a 0.2 m-wide, 3 m-long beam. Across
simulation and real-world studies, residual refinement consistently outperforms
template-only and monolithic baselines in success rate, centerline adherence,
and safety margins, while the structured footstep interface enables transparent
analysis and low-friction sim-to-real transfer.

### 3. [A Soft Fabric-Based Thermal Haptic Device for VR and Teleoperation](http://arxiv.org/pdf/2508.20831v1)

Authors: Rui Chen, Domenico Chiaradia, Antonio Frisoli, Daniele Leonardis

This paper presents a novel fabric-based thermal-haptic interface for virtual
reality and teleoperation. It integrates pneumatic actuation and conductive
fabric with an innovative ultra-lightweight design, achieving only 2~g for each
finger unit. By embedding heating elements within textile pneumatic chambers,
the system delivers modulated pressure and thermal stimuli to fingerpads
through a fully soft, wearable interface.
  Comprehensive characterization demonstrates rapid thermal modulation with
heating rates up to 3$^{\circ}$C/s, enabling dynamic thermal feedback for
virtual or teleoperation interactions. The pneumatic subsystem generates forces
up to 8.93~N at 50~kPa, while optimization of fingerpad-actuator clearance
enhances cooling efficiency with minimal force reduction. Experimental
validation conducted with two different user studies shows high temperature
identification accuracy (0.98 overall) across three thermal levels, and
significant manipulation improvements in a virtual pick-and-place tasks.
Results show enhanced success rates (88.5\% to 96.4\%, p = 0.029) and improved
force control precision (p = 0.013) when haptic feedback is enabled, validating
the effectiveness of the integrated thermal-haptic approach for advanced
human-machine interaction applications.

### 4. [Genetic Informed Trees (GIT*): Path Planning via Reinforced Genetic Programming Heuristics](http://arxiv.org/pdf/2508.20871v1)

Authors: Liding Zhang, Kuanqi Cai, Zhenshan Bing, Chaoqun Wang, Alois Knoll

Optimal path planning involves finding a feasible state sequence between a
start and a goal that optimizes an objective. This process relies on heuristic
functions to guide the search direction. While a robust function can improve
search efficiency and solution quality, current methods often overlook
available environmental data and simplify the function structure due to the
complexity of information relationships. This study introduces Genetic Informed
Trees (GIT*), which improves upon Effort Informed Trees (EIT*) by integrating a
wider array of environmental data, such as repulsive forces from obstacles and
the dynamic importance of vertices, to refine heuristic functions for better
guidance. Furthermore, we integrated reinforced genetic programming (RGP),
which combines genetic programming with reward system feedback to mutate
genotype-generative heuristic functions for GIT*. RGP leverages a multitude of
data types, thereby improving computational efficiency and solution quality
within a set timeframe. Comparative analyses demonstrate that GIT* surpasses
existing single-query, sampling-based planners in problems ranging from R^4 to
R^16 and was tested on a real-world mobile manipulation task. A video
showcasing our experimental results is available at
https://youtu.be/URjXbc_BiYg

### 5. [Deep Fuzzy Optimization for Batch-Size and Nearest Neighbors in Optimal Robot Motion Planning](http://arxiv.org/pdf/2508.20884v1)

Authors: Liding Zhang, Qiyang Zong, Yu Zhang, Zhenshan Bing, Alois Knoll

Efficient motion planning algorithms are essential in robotics. Optimizing
essential parameters, such as batch size and nearest neighbor selection in
sampling-based methods, can enhance performance in the planning process.
However, existing approaches often lack environmental adaptability. Inspired by
the method of the deep fuzzy neural networks, this work introduces
Learning-based Informed Trees (LIT*), a sampling-based deep fuzzy
learning-based planner that dynamically adjusts batch size and nearest neighbor
parameters to obstacle distributions in the configuration spaces. By encoding
both global and local ratios via valid and invalid states, LIT* differentiates
between obstacle-sparse and obstacle-dense regions, leading to lower-cost paths
and reduced computation time. Experimental results in high-dimensional spaces
demonstrate that LIT* achieves faster convergence and improved solution
quality. It outperforms state-of-the-art single-query, sampling-based planners
in environments ranging from R^8 to R^14 and is successfully validated on a
dual-arm robot manipulation task. A video showcasing our experimental results
is available at: https://youtu.be/NrNs9zebWWk

### 6. [Language-Enhanced Mobile Manipulation for Efficient Object Search in Indoor Environments](http://arxiv.org/pdf/2508.20899v1)

Authors: Liding Zhang, Zeqi Li, Kuanqi Cai, Qian Huang, Zhenshan Bing, Alois Knoll

Enabling robots to efficiently search for and identify objects in complex,
unstructured environments is critical for diverse applications ranging from
household assistance to industrial automation. However, traditional scene
representations typically capture only static semantics and lack interpretable
contextual reasoning, limiting their ability to guide object search in
completely unfamiliar settings. To address this challenge, we propose a
language-enhanced hierarchical navigation framework that tightly integrates
semantic perception and spatial reasoning. Our method, Goal-Oriented
Dynamically Heuristic-Guided Hierarchical Search (GODHS), leverages large
language models (LLMs) to infer scene semantics and guide the search process
through a multi-level decision hierarchy. Reliability in reasoning is achieved
through the use of structured prompts and logical constraints applied at each
stage of the hierarchy. For the specific challenges of mobile manipulation, we
introduce a heuristic-based motion planner that combines polar angle sorting
with distance prioritization to efficiently generate exploration paths.
Comprehensive evaluations in Isaac Sim demonstrate the feasibility of our
framework, showing that GODHS can locate target objects with higher search
efficiency compared to conventional, non-semantic search strategies. Website
and Video are available at: https://drapandiger.github.io/GODHS

### 7. [PLUME: Procedural Layer Underground Modeling Engine](http://arxiv.org/pdf/2508.20926v1)

Authors: Gabriel Manuel Garcia, Antoine Richard, Miguel Olivares-Mendez

As space exploration advances, underground environments are becoming
increasingly attractive due to their potential to provide shelter, easier
access to resources, and enhanced scientific opportunities. Although such
environments exist on Earth, they are often not easily accessible and do not
accurately represent the diversity of underground environments found throughout
the solar system. This paper presents PLUME, a procedural generation framework
aimed at easily creating 3D underground environments. Its flexible structure
allows for the continuous enhancement of various underground features, aligning
with our expanding understanding of the solar system. The environments
generated using PLUME can be used for AI training, evaluating robotics
algorithms, 3D rendering, and facilitating rapid iteration on developed
exploration algorithms. In this paper, it is demonstrated that PLUME has been
used along with a robotic simulator. PLUME is open source and has been released
on Github. https://github.com/Gabryss/P.L.U.M.E

### 8. [UltraTac: Integrated Ultrasound-Augmented Visuotactile Sensor for Enhanced Robotic Perception](http://arxiv.org/pdf/2508.20982v1)

Authors: Junhao Gong, Kit-Wa Sou, Shoujie Li, Changqing Guo, Yan Huang, Chuqiao Lyu, Ziwu Song, Wenbo Ding

Visuotactile sensors provide high-resolution tactile information but are
incapable of perceiving the material features of objects. We present UltraTac,
an integrated sensor that combines visuotactile imaging with ultrasound sensing
through a coaxial optoacoustic architecture. The design shares structural
components and achieves consistent sensing regions for both modalities.
Additionally, we incorporate acoustic matching into the traditional
visuotactile sensor structure, enabling integration of the ultrasound sensing
modality without compromising visuotactile performance. Through tactile
feedback, we dynamically adjust the operating state of the ultrasound module to
achieve flexible functional coordination. Systematic experiments demonstrate
three key capabilities: proximity sensing in the 3-8 cm range ($R^2=0.90$),
material classification (average accuracy: 99.20%), and texture-material
dual-mode object recognition achieving 92.11% accuracy on a 15-class task.
Finally, we integrate the sensor into a robotic manipulation system to
concurrently detect container surface patterns and internal content, which
verifies its potential for advanced human-machine interaction and precise
robotic manipulation.

### 9. [Rapid Mismatch Estimation via Neural Network Informed Variational Inference](http://arxiv.org/pdf/2508.21007v1)

Authors: Mateusz Jaszczuk, Nadia Figueroa

With robots increasingly operating in human-centric environments, ensuring
soft and safe physical interactions, whether with humans, surroundings, or
other machines, is essential. While compliant hardware can facilitate such
interactions, this work focuses on impedance controllers that allow
torque-controlled robots to safely and passively respond to contact while
accurately executing tasks. From inverse dynamics to quadratic
programming-based controllers, the effectiveness of these methods relies on
accurate dynamics models of the robot and the object it manipulates. Any model
mismatch results in task failures and unsafe behaviors. Thus, we introduce
Rapid Mismatch Estimation (RME), an adaptive, controller-agnostic,
probabilistic framework that estimates end-effector dynamics mismatches online,
without relying on external force-torque sensors. From the robot's
proprioceptive feedback, a Neural Network Model Mismatch Estimator generates a
prior for a Variational Inference solver, which rapidly converges to the
unknown parameters while quantifying uncertainty. With a real 7-DoF manipulator
driven by a state-of-the-art passive impedance controller, RME adapts to sudden
changes in mass and center of mass at the end-effector in $\sim400$ ms, in
static and dynamic settings. We demonstrate RME in a collaborative scenario
where a human attaches an unknown basket to the robot's end-effector and
dynamically adds/removes heavy items, showcasing fast and safe adaptation to
changing dynamics during physical interaction without any external sensory
system.

### 10. [HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning](http://arxiv.org/pdf/2508.21043v1)

Authors: Zhi Su, Bike Zhang, Nima Rahmanian, Yuman Gao, Qiayuan Liao, Caitlin Regan, Koushil Sreenath, S. Shankar Sastry

Humanoid robots have recently achieved impressive progress in locomotion and
whole-body control, yet they remain constrained in tasks that demand rapid
interaction with dynamic environments through manipulation. Table tennis
exemplifies such a challenge: with ball speeds exceeding 5 m/s, players must
perceive, predict, and act within sub-second reaction times, requiring both
agility and precision. To address this, we present a hierarchical framework for
humanoid table tennis that integrates a model-based planner for ball trajectory
prediction and racket target planning with a reinforcement learning-based
whole-body controller. The planner determines striking position, velocity and
timing, while the controller generates coordinated arm and leg motions that
mimic human strikes and maintain stability and agility across consecutive
rallies. Moreover, to encourage natural movements, human motion references are
incorporated during training. We validate our system on a general-purpose
humanoid robot, achieving up to 106 consecutive shots with a human opponent and
sustained exchanges against another humanoid. These results demonstrate
real-world humanoid table tennis with sub-second reactive control, marking a
step toward agile and interactive humanoid behaviors.

### Software Engineering

### 1. [From Law to Gherkin: A Human-Centred Quasi-Experiment on the Quality of LLM-Generated Behavioural Specifications from Food-Safety Regulations](http://arxiv.org/pdf/2508.20744v1)

Authors: Shabnam Hassani, Mehrdad Sabetzadeh, Daniel Amyot

Context: Laws and regulations increasingly affect software design and quality
assurance, but legal texts are written in technology-neutral language. This
creates challenges for engineers who must develop compliance artifacts such as
requirements and acceptance criteria. Manual creation is labor-intensive,
error-prone, and requires domain expertise. Advances in Generative AI (GenAI),
especially Large Language Models (LLMs), offer a way to automate deriving such
artifacts.
  Objective: We present the first systematic human-subject study of LLMs'
ability to derive behavioral specifications from legal texts using a
quasi-experimental design. These specifications translate legal requirements
into a developer-friendly form.
  Methods: Ten participants evaluated specifications generated from food-safety
regulations by Claude and Llama. Using Gherkin, a structured BDD language, 60
specifications were produced. Each participant assessed 12 across five
criteria: Relevance, Clarity, Completeness, Singularity, and Time Savings. Each
specification was reviewed by two participants, yielding 120 assessments.
  Results: For Relevance, 75% of ratings were highest and 20% second-highest.
Clarity reached 90% highest. Completeness: 75% highest, 19% second.
Singularity: 82% highest, 12% second. Time Savings: 68% highest, 24% second. No
lowest ratings occurred. Mann-Whitney U tests showed no significant differences
across participants or models. Llama slightly outperformed Claude in Clarity,
Completeness, and Time Savings, while Claude was stronger in Singularity.
Feedback noted hallucinations and omissions but confirmed the utility of the
specifications.
  Conclusion: LLMs can generate high-quality Gherkin specifications from legal
texts, reducing manual effort and providing structured artifacts useful for
implementation, assurance, and test generation.

### 2. [Towards an Architectural Perspective for Sustainability: Bundle the Needs from Industry](http://arxiv.org/pdf/2508.20774v1)

Authors: Markus Funke, Patricia Lago

Sustainability is increasingly recognized as an emerging quality property in
software-intensive systems, yet architects lack structured guidance to address
it effectively throughout the software design phase. Architectural
perspectives-an architectural knowledge artifact composed of concerns,
activities, tactics, pitfalls, and checklists-offer a promising approach to
tackle such emerging quality properties across architectural views and are also
independent of architecture frameworks and industry contexts. In this paper, we
present a sustainability perspective vision, i.e., a revised notion of
architectural perspective meant to be filled with its own elements to target
sustainability concerns. We formulate our sustainability perspective vision
through evidence from applying snowballing to seminal literature and from
conducting a focus group with experts in the field. Our findings confirm the
relevance of the different perspective elements in practice and highlight
implications for shaping a sustainability perspective that meets industrial
needs.

### 3. [Automated Test Oracles for Flaky Cyber-Physical System Simulators: Approach and Evaluation](http://arxiv.org/pdf/2508.20902v1)

Authors: Baharin A. Jodat, Khouloud Gaaloul, Mehrdad Sabetzadeh, Shiva Nejati

Simulation-based testing of cyber-physical systems (CPS) is costly due to the
time-consuming execution of CPS simulators. In addition, CPS simulators may be
flaky, leading to inconsistent test outcomes and requiring repeated test
re-execution for reliable test verdicts. Automated test oracles that do not
require system execution are therefore crucial for reducing testing costs.
Ideally, such test oracles should be interpretable to facilitate human
understanding of test verdicts, and they must be robust against the potential
flakiness of CPS simulators. In this article, we propose assertion-based test
oracles for CPS as sets of logical and arithmetic predicates defined over the
inputs of the system under test. Given a test input, our assertion-based test
oracle determines, without requiring test execution, whether the test passes,
fails, or if the oracle is inconclusive in predicting a verdict. We describe
two methods for generating assertion-based test oracles: one using genetic
programming~(GP) that employs well-known spectrum-based fault localization
(SBFL) ranking formulas, namely Ochiai, Tarantula, and Naish, as fitness
functions; and the other using decision trees (DT) and decision rules (DR). We
evaluate our assertion-based test oracles through case studies in the domains
of aerospace, networking and autonomous driving. We show that test oracles
generated using GP with Ochiai are significantly more accurate than those
obtained using GP with Tarantula and Naish or using DT or DR. Moreover, this
accuracy advantage remains even when accounting for the flakiness of the system
under test. We further show that the assertion-based test oracles generated by
GP with Ochiai are robust against flakiness with only 4% average variation in
their accuracy results across four different network and autonomous driving
systems with flaky behaviours.

### 4. [Deep Learning Based Concurrency Bug Detection and Localization](http://arxiv.org/pdf/2508.20911v1)

Authors: Zuocheng Feng, Kaiwen Zhang, Miaomiao Wang, Yiming Cheng, Yuandao Cai, Xiaofeng Li, Guanjun Liu

Concurrency bugs, caused by improper synchronization of shared resources in
multi-threaded or distributed systems, are notoriously hard to detect and thus
compromise software reliability and security. The existing deep learning
methods face three main limitations. First, there is an absence of large and
dedicated datasets of diverse concurrency bugs for them. Second, they lack
sufficient representation of concurrency semantics. Third, binary
classification results fail to provide finer-grained debug information such as
precise bug lines. To address these problems, we propose a novel method for
effective concurrency bug detection as well as localization. We construct a
dedicated concurrency bug dataset to facilitate model training and evaluation.
We then integrate a pre-trained model with a heterogeneous graph neural network
(GNN), by incorporating a new Concurrency-Aware Code Property Graph (CCPG) that
concisely and effectively characterizes concurrency semantics. To further
facilitate debugging, we employ SubgraphX, a GNN-based interpretability method,
which explores the graphs to precisely localize concurrency bugs, mapping them
to specific lines of source code. On average, our method demonstrates an
improvement of 10\% in accuracy and precision and 26\% in recall compared to
state-of-the-art methods across diverse evaluation settings.

### 5. [ConfLogger: Enhance Systems' Configuration Diagnosability through Configuration Logging](http://arxiv.org/pdf/2508.20977v1)

Authors: Shiwen Shan, Yintong Huo, Yuxin Su, Zhining Wang, Dan Li, Zibin Zheng

Modern configurable systems offer customization via intricate configuration
spaces, yet such flexibility introduces pervasive configuration-related issues
such as misconfigurations and latent softwarebugs. Existing diagnosability
supports focus on post-failure analysis of software behavior to identify
configuration issues, but none of these approaches look into whether the
software clue sufficient failure information for diagnosis. To fill in the
blank, we propose the idea of configuration logging to enhance existing logging
practices at the source code level. We develop ConfLogger, the first tool that
unifies configuration-aware static taint analysis with LLM-based log generation
to enhance software configuration diagnosability. Specifically, our method 1)
identifies configuration-sensitive code segments by tracing
configuration-related data flow in the whole project, and 2) generates
diagnostic log statements by analyzing configuration code contexts. Evaluation
results on eight popular software systems demonstrate the effectiveness of
ConfLogger to enhance configuration diagnosability. Specifically,
ConfLogger-enhanced logs successfully aid a log-based misconfiguration
diagnosis tool to achieve 100% accuracy on error localization in 30 silent
misconfiguration scenarios, with 80% directly resolvable through explicit
configuration information exposed. In addition, ConfLogger achieves 74%
coverage of existing logging points, outperforming baseline LLM-based loggers
by 12% and 30%. It also gains 8.6% higher in precision, 79.3% higher in recall,
and 26.2% higher in F1 compared to the state-of-the-art baseline in terms of
variable logging while also augmenting diagnostic value. A controlled user
study on 22 cases further validated its utility, speeding up diagnostic time by
1.25x and improving troubleshooting accuracy by 251.4%.

### 6. [Adaptive Root Cause Localization for Microservice Systems with Multi-Agent Recursion-of-Thought](http://arxiv.org/pdf/2508.20370v1)

Authors: Lingzhe Zhang, Tong Jia, Kangjin Wang, Weijie Hong, Chiming Duan, Minghua He, Ying Li

As contemporary microservice systems become increasingly popular and
complex-often comprising hundreds or even thousands of fine-grained,
interdependent subsystems-they are facing more frequent failures. Ensuring
system reliability thus demands accurate root cause localization. While traces
and metrics have proven to be effective data sources for this task, existing
methods either heavily rely on pre-defined schemas, which struggle to adapt to
evolving operational contexts, or lack interpretability in their reasoning
process, thereby leaving Site Reliability Engineers (SREs) confused. In this
paper, we conduct a comprehensive study on how SREs localize the root cause of
failures, drawing insights from multiple professional SREs across different
organizations. Our investigation reveals that human root cause analysis
exhibits three key characteristics: recursiveness, multi-dimensional expansion,
and cross-modal reasoning. Motivated by these findings, we introduce RCLAgent,
an adaptive root cause localization method for microservice systems that
leverages a multi-agent recursion-of-thought framework. RCLAgent employs a
novel recursion-of-thought strategy to guide the LLM's reasoning process,
effectively integrating data from multiple agents and tool-assisted analysis to
accurately pinpoint the root cause. Experimental evaluations on various public
datasets demonstrate that RCLAgent achieves superior performance by localizing
the root cause using only a single request-outperforming state-of-the-art
methods that depend on aggregating multiple requests. These results underscore
the effectiveness of RCLAgent in enhancing the efficiency and precision of root
cause localization in complex microservice environments.

### 7. [AI and Agile Software Development: A Research Roadmap from the XP2025 Workshop](http://arxiv.org/pdf/2508.20563v1)

Authors: Zheying Zhang, Tomas Herda, Victoria Pichler, Pekka Abrahamsson, Geir K. Hanssen, Joshua Kerievsky, Alex Polyakov, Mohit Chandna, Marius Irgens, Kai-Kristian Kemell, Ayman Asad Khan, Crystal Kwok, Evan Leybourn, Munish Malik, Dorota Mleczko, Morteza Moalagh, Christopher Morales, Yuliia Pieskova, Daniel Planötscher, Mika Saari, Anastasiia Tkalich, Karl Josef Gstettner, Xiaofeng Wang

This paper synthesizes the key findings from a full-day XP2025 workshop on
"AI and Agile: From Frustration to Success", held in Brugg-Windisch,
Switzerland. The workshop brought together over 30 interdisciplinary academic
researchers and industry practitioners to tackle the concrete challenges and
emerging opportunities at the intersection of Generative Artificial
Intelligence (GenAI) and agile software development. Through structured,
interactive breakout sessions, participants identified shared pain points like
tool fragmentation, governance, data quality, and critical skills gaps in AI
literacy and prompt engineering. These issues were further analyzed, revealing
underlying causes and cross-cutting concerns. The workshop concluded by
collaboratively co-creating a multi-thematic research roadmap, articulating
both short-term, implementable actions and visionary, long-term research
directions. This cohesive agenda aims to guide future investigation and drive
the responsible, human-centered integration of GenAI into agile practices.

### 8. [Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol](http://arxiv.org/pdf/2508.20737v1)

Authors: Wei Ma, Yixiao Yang, Qiang Hu, Shi Ying, Zhi Jin, Bo Du, Zhenchang Xing, Tianlin Li, Junjie Shi, Yang Liu, Linxiao Jiang

Applications of Large Language Models~(LLMs) have evolved from simple text
generators into complex software systems that integrate retrieval augmentation,
tool invocation, and multi-turn interactions. Their inherent non-determinism,
dynamism, and context dependence pose fundamental challenges for quality
assurance. This paper decomposes LLM applications into a three-layer
architecture: \textbf{\textit{System Shell Layer}}, \textbf{\textit{Prompt
Orchestration Layer}}, and \textbf{\textit{LLM Inference Core}}. We then assess
the applicability of traditional software testing methods in each layer:
directly applicable at the shell layer, requiring semantic reinterpretation at
the orchestration layer, and necessitating paradigm shifts at the inference
core. A comparative analysis of Testing AI methods from the software
engineering community and safety analysis techniques from the AI community
reveals structural disconnects in testing unit abstraction, evaluation metrics,
and lifecycle management. We identify four fundamental differences that
underlie 6 core challenges. To address these, we propose four types of
collaborative strategies (\emph{Retain}, \emph{Translate}, \emph{Integrate},
and \emph{Runtime}) and explore a closed-loop, trustworthy quality assurance
framework that combines pre-deployment validation with runtime monitoring.
Based on these strategies, we offer practical guidance and a protocol proposal
to support the standardization and tooling of LLM application testing. We
propose a protocol \textbf{\textit{Agent Interaction Communication Language}}
(AICL) that is used to communicate between AI agents. AICL has the
test-oriented features and is easily integrated in the current agent framework.

### 9. [Characterizing Trust Boundary Vulnerabilities in TEE Containers](http://arxiv.org/pdf/2508.20962v1)

Authors: Weijie Liu, Hongbo Chen, Shuo Huai, Zhen Xu, Wenhao Wang, Zhi Li, Zheli Liu

Trusted Execution Environments (TEEs) have emerged as a cornerstone of
confidential computing, garnering significant attention from both academia and
industry. To enable the secure development, execution, and deployment, of
applications on TEE platforms, TEE containers have been introduced as
middleware solutions. These containers aim to shield applications from
potentially malicious operating systems and orchestration interfaces while
maintaining usability and reliability. In this paper, we analyze the isolation
strategies employed by existing TEE containers to protect secure applications.
To address the challenges in analyzing these interfaces, we designed an
automated analyzer to precisely identify and evaluate their isolation
boundaries. We observed that some TEE containers fail to achieve their intended
goals due to critical design and implementation flaws, such as information
leakage, rollback attacks, denial-of-service, and Iago attacks, which pose
significant security risks. Drawing from our findings, we share key lessons to
guide the development of more secure container solutions and discuss emerging
trends in TEE containerization design.

### 10. [Dynamics of Gender Bias in Software Engineering](http://arxiv.org/pdf/2508.21050v1)

Authors: Thomas J. Misa

The field of software engineering is embedded in both engineering and
computer science, and may embody gender biases endemic to both. This paper
surveys software engineering's origins and its long-running attention to
engineering professionalism, profiling five leaders; it then examines the
field's recent attention to gender issues and gender bias. It next
quantitatively analyzes women's participation as research authors in the
field's leading International Conference of Software Engineering (1976-2010),
finding a dozen years with statistically significant gender exclusion. Policy
dimensions of research on gender bias in computing are suggested.

### Systems and Control

### 1. [Systolic Array-based Architecture for Low-Bit Integerized Vision Transformers](http://arxiv.org/pdf/2508.20334v1)

Authors: Ching-Yi Lin, Sahil Shah

Transformer-based models are becoming more and more intelligent and are
revolutionizing a wide range of human tasks. To support their deployment, AI
labs offer inference services that consume hundreds of GWh of energy annually
and charge users based on the number of tokens processed. Under this cost
model, minimizing power consumption and maximizing throughput have become key
design goals for the inference hardware. While graphics processing units (GPUs)
are commonly used, their flexibility comes at the cost of low operational
intensity and limited efficiency, especially under the high query-per-model
ratios of modern inference services.
  In this work, we address these challenges by proposing a low-bit,
model-specialized accelerator that strategically selects tasks with high
operation (OP) reuse and minimal communication overhead for offloading. Our
design incorporates multiple systolic arrays with deep, fine-grained pipelines
and array-compatible units that support essential operations in multi-head
self-attention (MSA) module. At the accelerator-level, each self-attention (SA)
head is pipelined within a single accelerator to increase data reuse and
further minimize bandwidth.
  Our 3-bit integerized model achieves 96.83% accuracy on CIFAR-10 and 77.81%
top-1 accuracy on ImageNet. We validate the hardware design on a 16nm FPGA
(Alveo U250), where it delivers 13,568 GigaOps/second (GOPs/s) and 219.4
GOPs/s/W. Compared to a same-technology GPU (GTX 1080), our design offers 1.50x
higher throughput and 4.47x better power efficiency. Even against a
state-of-the-art GPU (RTX 5090), we still achieve 20% better power efficiency
despite having 87% lower throughput.

### 2. [Bootstrap Policy Iteration for Stochastic LQ Tracking with Multiplicative Noise](http://arxiv.org/pdf/2508.20394v1)

Authors: Jiayu Chen, Zhenhui Xu, Xinghu Wang

This paper studies the optimal tracking control problem for continuous-time
stochastic linear systems with multiplicative noise. The solution framework
involves solving a stochastic algebraic Riccati equation for the feedback gain
and a Sylvester equation for the feedforward gain. To enable model-free optimal
tracking, we first develop a two-phase bootstrap policy iteration (B-PI)
algorithm, which bootstraps a stabilizing control gain from the trivially
initialized zero-value start and proceeds with standard policy iteration.
Building on this algorithm, we propose a data-driven, off-policy reinforcement
learning approach that ensures convergence to the optimal feedback gain under
the interval excitation condition. We further introduce a data-driven method to
compute the feedforward using the obtained feedback gain. Additionally, for
systems with state-dependent noise, we propose a shadow system-based optimal
tracking method to eliminate the need for probing noise. The effectiveness of
the proposed methods is demonstrated through numerical examples.

### 3. [MegaCacheX: Towards Cost-Effective Hierarchical Collaborative Content Caching in Emerging Mega-Constellations](http://arxiv.org/pdf/2508.20433v1)

Authors: Haoyang Shi, Xing Zhang, Sitong Li, Minghang Li, Xinming Lu, Shaoxiang Xu, Guoquan Wang

Significant latency in global content delivery primarily arises from
insufficient terrestrial infrastructure. Deploying space-based content delivery
networks within emerging mega-constellations provides an effective means to
bridge the digital divide. However, space-based caching faces constraints from
physical-layer dynamics, including dynamic topologies, time-varying
inter-satellite link conditions, and limited onboard energy. In addition,
existing mechanisms often lack fine-grained content categorization and global
optimization. This paper proposes MegaCacheX, a cost-effective hierarchical
framework for collaborative content distribution that achieves
"Earth-independence" by providing cloud services directly from space.
Specifically, data centers in Sun-synchronous orbit act as primary content
sources, while caching nodes in mega-constellations and ground stations
collaboratively form a distributed edge layer. MegaCacheX optimizes caching
strategies by integrating content popularity, regional user distribution, and
satellite trajectory predictions. Multi-tier caching nodes serve as service
anchors, enabling seamless content delivery with low latency. A prototype
implemented on a microservices-based, containerized testbed demonstrates that
MegaCacheX reduces global content access latency by about 36% compared to
baseline approaches, while maintaining cost efficiency.

### 4. [Joint Contact Planning for Navigation and Communication in GNSS-Libration Point Systems](http://arxiv.org/pdf/2508.20479v1)

Authors: Huan Yan, Juan A. Fraire, Ziqi Yang, Kanglian Zhao, Wenfeng Li, Xiyun Hou, Haohan Li, Yuxuan Miao, Jinjun Zheng, Chengbin Kang, Huichao Zhou, Xinuo Chang, Lu Wang, Linshan Xue

Deploying satellites at Earth-Moon Libration Points (LPs) addresses the
inherent deep-space coverage gaps of low-altitude GNSS constellations.
Integrating LP satellites with GNSS into a joint constellation enables a more
robust and comprehensive Positioning, Navigation, and Timing (PNT) system,
while also extending navigation and communication services to spacecraft
operating in cislunar space (i.e., users). However, the long propagation delays
between LP satellites, users, and GNSS satellites result in significantly
different link durations compared to those within the GNSS constellation.
Scheduling inter-satellite links (ISLs) is a core task of Contact Plan Design
(CPD). Existing CPD approaches focus exclusively on GNSS constellations,
assuming uniform link durations, and thus cannot accommodate the heterogeneous
link timescales present in a joint GNSS-LP system. To overcome this limitation,
we introduce a Joint CPD (J-CPD) scheme tailored to handle ISLs with differing
duration units across integrated constellations. The key contributions of J-CPD
are: (i):introduction of LongSlots (Earth-Moon scale links) and ShortSlots
(GNSS-scale links); (ii):a hierarchical and crossed CPD process for scheduling
LongSlots and ShortSlots ISLs; (iii):an energy-driven link scheduling algorithm
adapted to the CPD process. Simulations on a joint BeiDou-LP constellation
demonstrate that J-CPD surpasses the baseline FCP method in both delay and
ranging coverage, while maintaining high user satisfaction and enabling tunable
trade-offs through adjustable potential-energy parameters. To our knowledge,
this is the first CPD framework to jointly optimize navigation and
communication in GNSS-LP systems, representing a key step toward unified and
resilient deep-space PNT architectures.

### 5. [Adaptive Control of Heterogeneous Platoons with Guaranteed Collision Avoidance](http://arxiv.org/pdf/2508.20493v1)

Authors: Ashutosh Chandra Pandey, Sayan Basu Roy, Simone Baldi

This work proposes a framework for Cooperative Adaptive Cruise Control of a
vehicular platoon characterized by unidirectional communication and
heterogeneous parameters. In the proposed framework, the actual (heterogeneous)
platoon is made to converge to a reference (homogeneous) platoon via adaptive
laws designed using of set-theoretic model reference adaptive control. Yet, in
contrast to the state-of-art that is based on ensuring collision avoidance on
the reference platoon dynamics only, the approach we propose can ensure
collision avoidance on the actual platoon dynamics. This result is possible
thanks to the introduction of a novel concept of virtual platoon, only used for
analysis, but that does not interact with the actual platoon. The stability and
convergence properties of the proposed framework are established using
Lyapunov-based analysis in conjunction with the aforementioned virtual platoon
concept.

### 6. [Local Observability of a Class of Feedforward Neural Networks](http://arxiv.org/pdf/2508.20544v1)

Authors: Yi Yang, Victor G. Lopez, Matthias A. Müller

Beyond the traditional neural network training methods based on gradient
descent and its variants, state estimation techniques have been proposed to
determine a set of ideal weights from a control-theoretic perspective. Hence,
the concept of observability becomes relevant in neural network training. In
this paper, we investigate local observability of a class of two-layer
feedforward neural networks~(FNNs) with rectified linear unit~(ReLU) activation
functions. We analyze local observability of FNNs by evaluating an
observability rank condition with respect to the weight matrix and the input
sequence. First, we show that, in general, the weights of FNNs are not locally
observable. Then, we provide sufficient conditions on the network structures
and the weights that lead to local observability. Moreover, we propose an input
design approach to render the weights distinguishable and show that this input
also excites other weights inside a neighborhood. Finally, we validate our
results through a numerical example.

### 7. [Transient Stability Analysis of a Hybrid Grid-Forming and Grid-Following RES System Considering Multi-Mode Control Switching](http://arxiv.org/pdf/2508.20552v1)

Authors: Ruiyuan Zeng, Ruisheng Diao, Fangyuan Sun, Wangqianyun Tang, Junjie Li, Baorong Zhou

The inherent control switching of renewable energy sources (RESs) during
intricate transient processes introduces complexity to the dynamic behavior of
modern power systems. This paper reveals the dynamic coupling between grid
forming (GFM)/grid following (GFL)-based RES and dominant instability modes of
the hybrid system. First, six control combinations are systematically
investigated by pairing the two GFM-RES modes, normal control (NC) and current
saturation (CS), with the three GFL-RES modes: normal control, low voltage
ride-through (LVRT), and high voltage ride-through (HVRT). Based on switching
system theory, the coupled power flow and dynamic motion models are developed
considering multi-mode switching characteristics. It is revealed that the
hybrid system exhibits two distinct instability modes when the GFM-RES and
GFL-RES exceed their P-f and V-f desynchronization boundaries, respectively.
The two-dimensional spatiotemporal damping characteristics of GFL-RES induced
by GFM-RES are also uncovered for the first time. A novel criterion is proposed
to quantify the impact of GFM-RES on GFL-RES dynamics, capturing both its
stabilizing and destabilizing effects under different control combinations.
High-fidelity electromagnetic transient simulations validate the correctness of
the analysis framework.

### 8. [DMPC-Swarm: Distributed Model Predictive Control on Nano UAV swarms](http://arxiv.org/pdf/2508.20553v1)

Authors: Alexander Gräfe, Joram Eickhoff, Marco Zimmerling, Sebastian Trimpe

Swarms of unmanned aerial vehicles (UAVs) are increasingly becoming vital to
our society, undertaking tasks such as search and rescue, surveillance and
delivery. A special variant of Distributed Model Predictive Control (DMPC) has
emerged as a promising approach for the safe management of these swarms by
combining the scalability of distributed computation with dynamic swarm motion
control. In this DMPC method, multiple agents solve local optimization problems
with coupled anti-collision constraints, periodically exchanging their
solutions. Despite its potential, existing methodologies using this DMPC
variant have yet to be deployed on distributed hardware that fully utilize true
distributed computation and wireless communication. This is primarily due to
the lack of a communication system tailored to meet the unique requirements of
mobile swarms and an architecture that supports distributed computation while
adhering to the payload constraints of UAVs. We present DMPC-SWARM, a new swarm
control methodology that integrates an efficient, stateless low-power wireless
communication protocol with a novel DMPC algorithm that provably avoids UAV
collisions even under message loss. By utilizing event-triggered and
distributed off-board computing, DMPC-SWARM supports nano UAVs, allowing them
to benefit from additional computational resources while retaining scalability
and fault tolerance. In a detailed theoretical analysis, we prove that
DMPC-SWARM guarantees collision avoidance under realistic conditions, including
communication delays and message loss. Finally, we present DMPC-SWARM's
implementation on a swarm of up to 16 nano-quadcopters, demonstrating the first
realization of these DMPC variants with computation distributed on multiple
physical devices interconnected by a real wireless mesh networks. A video
showcasing DMPC-SWARM is available at http://tiny.cc/DMPCSwarm.

### 9. [Minimizing AoI in Mobile Edge Computing: Nested Index Policy with Preemptive and Non-preemptive Structure](http://arxiv.org/pdf/2508.20564v1)

Authors: Ning Yang, Yibo Liu, Shuo Chen, Meng Zhang, Haijun Zhang

Mobile Edge Computing (MEC) leverages computational heterogeneity between
mobile devices and edge nodes to enable real-time applications requiring high
information freshness. The Age-of-Information (AoI) metric serves as a crucial
evaluator of information timeliness in such systems. Addressing AoI
minimization in multi-user MEC environments presents significant challenges due
to stochastic computing times. In this paper, we consider multiple users
offloading tasks to heterogeneous edge servers in an MEC system, focusing on
preemptive and non-preemptive task scheduling mechanisms. The problem is first
reformulated as a Restless Multi-Arm Bandit (RMAB) problem, with a multi-layer
Markov Decision Process (MDP) framework established to characterize AoI
dynamics in the MEC system. Based on the multi-layer MDP, we propose a nested
index framework and design a nested index policy with provably asymptotic
optimality. This establishes a theoretical framework adaptable to various
scheduling mechanisms, achieving efficient optimization through state
stratification and index design in both preemptive and non-preemptive modes.
Finally, the closed-form of the nested index is derived, facilitating
performance trade-offs between computational complexity and accuracy while
ensuring the universal applicability of the nested index policy across both
scheduling modes. The experimental results show that in non-preemptive
scheduling, compared with the benchmark method, the optimality gap is reduced
by 25.43%, while in preemptive scheduling, the gap has reduced by 61.84%. As
the system scale increases, it asymptotically converges in two scheduling modes
and especially provides near-optimal performance in non-preemptive structure.

### 10. [A Proposal for Yield Improvement with Power Tradeoffs in CMOS LNAs (English Version)](http://arxiv.org/pdf/2508.20611v1)

Authors: J. L. González, J. C. Cruz, R. L. Moreno, D. Vázquez

This paper studies an architecture with digitally controllable gain and power
consumption to mitigate the impact of process variations on CMOS low-noise
amplifiers (LNAs). A \SI{130}{nm}, \SI{1.2}{V} LNA implementing the proposed
architecture is designed based on an analysis of variability in traditional
LNAs under different bias currents and on the corresponding effects on the
performance of a complete receiver. Two different adjustment strategies are
evaluated, both of which are compatible with previously reported built-in
self-test (BIST) circuits. Results show that the proposed architecture enables
yield enhancement while keeping low-power operation compared with traditional
LNAs.

### Machine Learning (Statistics Category)

### 1. [Latent Factor Point Processes for Patient Representation in Electronic Health Records](http://arxiv.org/pdf/2508.20327v1)

Authors: Parker Knight, Doudou Zhou, Zongqi Xia, Tianxi Cai, Junwei Lu

Electronic health records (EHR) contain valuable longitudinal patient-level
information, yet most statistical methods reduce the irregular timing of EHR
codes into simple counts, thereby discarding rich temporal structure. Existing
temporal models often impose restrictive parametric assumptions or are tailored
to code level rather than patient-level tasks. We propose the latent factor
point process model, which represents code occurrences as a high-dimensional
point process whose conditional intensity is driven by a low dimensional latent
Poisson process. This low-rank structure reflects the clinical reality that
thousands of codes are governed by a small number of underlying disease
processes, while enabling statistically efficient estimation in high
dimensions. Building on this model, we introduce the Fourier-Eigen embedding, a
patient representation constructed from the spectral density matrix of the
observed process. We establish theoretical guarantees showing that these
embeddings efficiently capture subgroup-specific temporal patterns for
downstream classification and clustering. Simulations and an application to an
Alzheimer's disease EHR cohort demonstrate the practical advantages of our
approach in uncovering clinically meaningful heterogeneity.

### 2. [Unbiased Stochastic Optimization for Gaussian Processes on Finite Dimensional RKHS](http://arxiv.org/pdf/2508.20588v1)

Authors: Neta Shoham, Haim Avron

Current methods for stochastic hyperparameter learning in Gaussian Processes
(GPs) rely on approximations, such as computing biased stochastic gradients or
using inducing points in stochastic variational inference. However, when using
such methods we are not guaranteed to converge to a stationary point of the
true marginal likelihood. In this work, we propose algorithms for exact
stochastic inference of GPs with kernels that induce a Reproducing Kernel
Hilbert Space (RKHS) of moderate finite dimension. Our approach can also be
extended to infinite dimensional RKHSs at the cost of forgoing exactness. Both
for finite and infinite dimensional RKHSs, our method achieves better
experimental results than existing methods when memory resources limit the
feasible batch size and the possible number of inducing points.

### 3. [Dimension Agnostic Testing of Survey Data Credibility through the Lens of Regression](http://arxiv.org/pdf/2508.20616v1)

Authors: Debabrota Basu, Sourav Chakraborty, Debarshi Chanda, Buddha Dev Das, Arijit Ghosh, Arnab Ray

Assessing whether a sample survey credibly represents the population is a
critical question for ensuring the validity of downstream research. Generally,
this problem reduces to estimating the distance between two high-dimensional
distributions, which typically requires a number of samples that grows
exponentially with the dimension. However, depending on the model used for data
analysis, the conclusions drawn from the data may remain consistent across
different underlying distributions. In this context, we propose a task-based
approach to assess the credibility of sampled surveys. Specifically, we
introduce a model-specific distance metric to quantify this notion of
credibility. We also design an algorithm to verify the credibility of survey
data in the context of regression models. Notably, the sample complexity of our
algorithm is independent of the data dimension. This efficiency stems from the
fact that the algorithm focuses on verifying the credibility of the survey data
rather than reconstructing the underlying regression model. Furthermore, we
show that if one attempts to verify credibility by reconstructing the
regression model, the sample complexity scales linearly with the dimensionality
of the data. We prove the theoretical correctness of our algorithm and
numerically demonstrate our algorithm's performance.

### 4. [Supervised Stochastic Gradient Algorithms for Multi-Trial Source Separation](http://arxiv.org/pdf/2508.20618v1)

Authors: Ronak Mehta, Mateus Piovezan Otto, Noah Stanis, Azadeh Yazdan-Shahmorad, Zaid Harchaoui

We develop a stochastic algorithm for independent component analysis that
incorporates multi-trial supervision, which is available in many scientific
contexts. The method blends a proximal gradient-type algorithm in the space of
invertible matrices with joint learning of a prediction model through
backpropagation. We illustrate the proposed algorithm on synthetic and real
data experiments. In particular, owing to the additional supervision, we
observe an increased success rate of the non-convex optimization and the
improved interpretability of the independent components.

### 5. [Polynomial Chaos Expansion for Operator Learning](http://arxiv.org/pdf/2508.20886v1)

Authors: Himanshu Sharma, Lukáš Novák, Michael D. Shields

Operator learning (OL) has emerged as a powerful tool in scientific machine
learning (SciML) for approximating mappings between infinite-dimensional
functional spaces. One of its main applications is learning the solution
operator of partial differential equations (PDEs). While much of the progress
in this area has been driven by deep neural network-based approaches such as
Deep Operator Networks (DeepONet) and Fourier Neural Operator (FNO), recent
work has begun to explore traditional machine learning methods for OL. In this
work, we introduce polynomial chaos expansion (PCE) as an OL method. PCE has
been widely used for uncertainty quantification (UQ) and has recently gained
attention in the context of SciML. For OL, we establish a mathematical
framework that enables PCE to approximate operators in both purely data-driven
and physics-informed settings. The proposed framework reduces the task of
learning the operator to solving a system of equations for the PCE
coefficients. Moreover, the framework provides UQ by simply post-processing the
PCE coefficients, without any additional computational cost. We apply the
proposed method to a diverse set of PDE problems to demonstrate its
capabilities. Numerical results demonstrate the strong performance of the
proposed method in both OL and UQ tasks, achieving excellent numerical accuracy
and computational efficiency.

### 6. [Stochastic Gradients under Nuisances](http://arxiv.org/pdf/2508.20326v1)

Authors: Facheng Yu, Ronak Mehta, Alex Luedtke, Zaid Harchaoui

Stochastic gradient optimization is the dominant learning paradigm for a
variety of scenarios, from classical supervised learning to modern
self-supervised learning. We consider stochastic gradient algorithms for
learning problems whose objectives rely on unknown nuisance parameters, and
establish non-asymptotic convergence guarantees. Our results show that, while
the presence of a nuisance can alter the optimum and upset the optimization
trajectory, the classical stochastic gradient algorithm may still converge
under appropriate conditions, such as Neyman orthogonality. Moreover, even when
Neyman orthogonality is not satisfied, we show that an algorithm variant with
approximately orthogonalized updates (with an approximately orthogonalized
gradient oracle) may achieve similar convergence rates. Examples from
orthogonal statistical learning/double machine learning and causal inference
are discussed.

### 7. [Towards Trustworthy Amortized Bayesian Model Comparison](http://arxiv.org/pdf/2508.20614v1)

Authors: Šimon Kucharský, Aayush Mishra, Daniel Habermann, Stefan T. Radev, Paul-Christian Bürkner

Amortized Bayesian model comparison (BMC) enables fast probabilistic ranking
of models via simulation-based training of neural surrogates. However, the
reliability of neural surrogates deteriorates when simulation models are
misspecified - the very case where model comparison is most needed. Thus, we
supplement simulation-based training with a self-consistency (SC) loss on
unlabeled real data to improve BMC estimates under empirical distribution
shifts. Using a numerical experiment and two case studies with real data, we
compare amortized evidence estimates with and without SC against analytic or
bridge sampling benchmarks. SC improves calibration under model
misspecification when having access to analytic likelihoods. However, it offers
limited gains with neural surrogate likelihoods, making it most practical for
trustworthy BMC when likelihoods are exact.

### 8. [Provable Benefits of In-Tool Learning for Large Language Models](http://arxiv.org/pdf/2508.20755v1)

Authors: Sam Houliston, Ambroise Odonnat, Charles Arnal, Vivien Cabannes

Tool-augmented language models, equipped with retrieval, memory, or external
APIs, are reshaping AI, yet their theoretical advantages remain underexplored.
In this paper, we address this question by demonstrating the benefits of
in-tool learning (external retrieval) over in-weight learning (memorization)
for factual recall. We show that the number of facts a model can memorize
solely in its weights is fundamentally limited by its parameter count. In
contrast, we prove that tool-use enables unbounded factual recall via a simple
and efficient circuit construction. These results are validated in controlled
experiments, where tool-using models consistently outperform memorizing ones.
We further show that for pretrained large language models, teaching tool-use
and general rules is more effective than finetuning facts into memory. Our work
provides both a theoretical and empirical foundation, establishing why
tool-augmented workflows are not just practical, but provably more scalable.

### 9. [Fast Convergence Rates for Subsampled Natural Gradient Algorithms on Quadratic Model Problems](http://arxiv.org/pdf/2508.21022v1)

Authors: Gil Goldshlager, Jiang Hu, Lin Lin

Subsampled natural gradient descent (SNGD) has shown impressive results for
parametric optimization tasks in scientific machine learning, such as neural
network wavefunctions and physics-informed neural networks, but it has lacked a
theoretical explanation. We address this gap by analyzing the convergence of
SNGD and its accelerated variant, SPRING, for idealized parametric optimization
problems where the model is linear and the loss function is strongly convex and
quadratic. In the special case of a least-squares loss, namely the standard
linear least-squares problem, we prove that SNGD is equivalent to a regularized
Kaczmarz method while SPRING is equivalent to an accelerated regularized
Kaczmarz method. As a result, by leveraging existing analyses we obtain under
mild conditions (i) the first fast convergence rate for SNGD, (ii) the first
convergence guarantee for SPRING in any setting, and (iii) the first proof that
SPRING can accelerate SNGD. In the case of a general strongly convex quadratic
loss, we extend the analysis of the regularized Kaczmarz method to obtain a
fast convergence rate for SNGD under stronger conditions, providing the first
explanation for the effectiveness of SNGD outside of the least-squares setting.
Overall, our results illustrate how tools from randomized linear algebra can
shed new light on the interplay between subsampling and curvature-aware
optimization strategies.

### 10. [Transfer Learning for Classification under Decision Rule Drift with Application to Optimal Individualized Treatment Rule Estimation](http://arxiv.org/pdf/2508.20942v1)

Authors: Xiaohan Wang, Yang Ning

In this paper, we extend the transfer learning classification framework from
regression function-based methods to decision rules. We propose a novel
methodology for modeling posterior drift through Bayes decision rules. By
exploiting the geometric transformation of the Bayes decision boundary, our
method reformulates the problem as a low-dimensional empirical risk
minimization problem. Under mild regularity conditions, we establish the
consistency of our estimators and derive the risk bounds. Moreover, we
illustrate the broad applicability of our method by adapting it to the
estimation of optimal individualized treatment rules. Extensive simulation
studies and analyses of real-world data further demonstrate both superior
performance and robustness of our approach.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-29 PST.

### 1. [Geometric constraints and semantic optimization SLAM algorithm for dynamic scenarios](https://www.nature.com/articles/s41598-025-16714-x)

Authors: Yanli Liu et al.

### 2. [Proving vote correctness in the IVXV internet voting system](https://www.nature.com/articles/s41598-025-16764-1)

Authors: Taaniel Kraavi et al.

### 3. [Cyberattack event and arguments extraction based on feature interaction and few-shot learning](https://www.nature.com/articles/s41598-025-15138-x)

Authors: Yue Han et al.

### 4. [Evaluation of deep learning models using explainable AI with qualitative and quantitative analysis for rice leaf disease detection](https://www.nature.com/articles/s41598-025-14306-3)

Authors: Hari Kishan Kondaveeti et al.

### 5. [Personalized health monitoring using explainable AI: bridging trust in predictive healthcare](https://www.nature.com/articles/s41598-025-15867-z)

Authors: M. Sree Vani et al.

### 6. [Multimodal feature distinguishing and deep learning approach to detect lung disease from MRI images](https://www.nature.com/articles/s41598-025-17796-3)

Authors: Turki M. Alanazi

