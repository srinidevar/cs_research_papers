# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-15 17:00:26.899485 PST.

### Artificial Intelligence

### 1. [CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing](http://arxiv.org/pdf/2510.12033v1)

Authors: Chathurangi Shyalika, Aryaman Sharma, Fadi El Kalach, Utkarshani Jaimini, Cory Henson, Ramy Harik, Amit Sheth

Modern manufacturing environments demand not only accurate predictions but
also interpretable insights to process anomalies, root causes, and potential
interventions. Existing AI systems often function as isolated black boxes,
lacking the seamless integration of prediction, explanation, and causal
reasoning required for a unified decision-support solution. This fragmentation
limits their trustworthiness and practical utility in high-stakes industrial
environments. In this work, we present CausalTrace, a neurosymbolic causal
analysis module integrated into the SmartPilot industrial CoPilot. CausalTrace
performs data-driven causal analysis enriched by industrial ontologies and
knowledge graphs, including advanced functions such as causal discovery,
counterfactual reasoning, and root cause analysis (RCA). It supports real-time
operator interaction and is designed to complement existing agents by offering
transparent, explainable decision support. We conducted a comprehensive
evaluation of CausalTrace using multiple causal assessment methods and the C3AN
framework (i.e. Custom, Compact, Composite AI with Neurosymbolic Integration),
which spans principles of robustness, intelligence, and trustworthiness. In an
academic rocket assembly testbed, CausalTrace achieved substantial agreement
with domain experts (ROUGE-1: 0.91 in ontology QA) and strong RCA performance
(MAP@3: 94%, PR@2: 97%, MRR: 0.92, Jaccard: 0.92). It also attained 4.59/5 in
the C3AN evaluation, demonstrating precision and reliability for live
deployment.

### 2. [Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Response](http://arxiv.org/pdf/2510.12061v1)

Authors: Yiheng Chen, Lingyao Li, Zihui Ma, Qikai Hu, Yilun Zhu, Min Deng, Runlong Yu

Effective disaster response is essential for safeguarding lives and property.
Existing statistical approaches often lack semantic context, generalize poorly
across events, and offer limited interpretability. While Large language models
(LLMs) provide few-shot generalization, they remain text-bound and blind to
geography. To bridge this gap, we introduce a Geospatial Awareness Layer (GAL)
that grounds LLM agents in structured earth data. Starting from raw wildfire
detections, GAL automatically retrieves and integrates infrastructure,
demographic, terrain, and weather information from external geodatabases,
assembling them into a concise, unit-annotated perception script. This enriched
context enables agents to produce evidence-based resource-allocation
recommendations (e.g., personnel assignments, budget allocations), further
reinforced by historical analogs and daily change signals for incremental
updates. We evaluate the framework in real wildfire scenarios across multiple
LLM models, showing that geospatially grounded agents can outperform baselines.
The proposed framework can generalize to other hazards such as floods and
hurricanes.

### 3. [HiCoTraj:Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory](http://arxiv.org/pdf/2510.12067v1)

Authors: Junyi Xie, Yuankun Jiao, Jina Kim, Yao-Yi Chiang, Lingyi Zhao, Khurram Shafique

Inferring demographic attributes such as age, sex, or income level from human
mobility patterns enables critical applications such as targeted public health
interventions, equitable urban planning, and personalized transportation
services. Existing mobility-based demographic inference studies heavily rely on
large-scale trajectory data with demographic labels, leading to limited
interpretability and poor generalizability across different datasets and user
groups. We propose HiCoTraj (Zero-Shot Demographic Reasoning via Hierarchical
Chain-of-Thought Prompting from Trajectory), a framework that leverages LLMs'
zero-shot learning and semantic understanding capabilities to perform
demographic inference without labeled training data. HiCoTraj transforms
trajectories into semantically rich, natural language representations by
creating detailed activity chronicles and multi-scale visiting summaries. Then
HiCoTraj uses a novel hierarchical chain of thought reasoning to systematically
guide LLMs through three cognitive stages: factual feature extraction,
behavioral pattern analysis, and demographic inference with structured output.
This approach addresses the scarcity challenge of labeled demographic data
while providing transparent reasoning chains. Experimental evaluation on
real-world trajectory data demonstrates that HiCoTraj achieves competitive
performance across multiple demographic attributes in zero-shot scenarios.

### 4. [BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection for Human Mobility Data](http://arxiv.org/pdf/2510.12076v1)

Authors: Junyi Xie, Jina Kim, Yao-Yi Chiang, Lingyi Zhao, Khurram Shafique

Traditional anomaly detection in human mobility has primarily focused on
trajectory-level analysis, identifying statistical outliers or spatiotemporal
inconsistencies across aggregated movement traces. However, detecting
individual-level anomalies, i.e., unusual deviations in a person's mobility
behavior relative to their own historical patterns, within datasets
encompassing large populations remains a significant challenge. In this paper,
we present BeSTAD (Behavior-aware Spatio-Temporal Anomaly Detection for Human
Mobility Data), an unsupervised framework that captures individualized
behavioral signatures across large populations and uncovers fine-grained
anomalies by jointly modeling spatial context and temporal dynamics. BeSTAD
learns semantically enriched mobility representations that integrate location
meaning and temporal patterns, enabling the detection of subtle deviations in
individual movement behavior. BeSTAD further employs a behavior-cluster-aware
modeling mechanism that builds personalized behavioral profiles from normal
activity and identifies anomalies through cross-period behavioral comparison
with consistent semantic alignment. Building on prior work in mobility behavior
clustering, this approach enables not only the detection of behavioral shifts
and deviations from established routines but also the identification of
individuals exhibiting such changes within large-scale mobility datasets. By
learning individual behaviors directly from unlabeled data, BeSTAD advances
anomaly detection toward personalized and interpretable mobility analysis.

### 5. [Evaluating the Quality of Randomness and Entropy in Tasks Supported by Large Language Models](http://arxiv.org/pdf/2510.12080v1)

Authors: Rabimba Karanjai, Yang Lu, Ranjith Chodavarapu, Lei Xu, Weidong Shi

The rapid advancement of large language model (LLM) technology has led to
diverse applications, many of which inherently require randomness, such as
stochastic decision-making, gaming, scheduling, AI agents, and
cryptography-related tasks. However, the capabilities of LLMs in handling
randomness, particularly in generating and utilizing random numbers
effectively, remain unclear. This paper investigates the capacity of LLMs for
handling tasks that involve randomness through a series of experiments. We
designed a set of experiments that consider various factors that can influence
an LLM's performance in tasks involving randomness, such as accessibility to
external tools, types of tasks, model states (fresh vs. non-fresh), and
prompting strategies. The experiments cover a range of tasks, including
generating random numbers, generating random strings such as passwords,
shuffling items, and evaluating the quality of randomness using entropy and the
NIST randomness test-suite. Our findings reveal that while LLMs can generate
outputs that exhibit some degree of randomness, their performance is
inconsistent and often deviates significantly from the expected behavior. The
analysis of the experimental results highlights key limitations and areas where
improvement is needed for the LLMs to effectively handle tasks involving
randomness

### 6. [MatSciBench: Benchmarking the Reasoning Ability of Large Language Models in Materials Science](http://arxiv.org/pdf/2510.12171v1)

Authors: Junkai Zhang, Jingru Gan, Xiaoxuan Wang, Zian Jia, Changquan Gu, Jianpeng Chen, Yanqiao Zhu, Mingyu Derek Ma, Dawei Zhou, Ling Li, Wei Wang

Large Language Models (LLMs) have demonstrated remarkable abilities in
scientific reasoning, yet their reasoning capabilities in materials science
remain underexplored. To fill this gap, we introduce MatSciBench, a
comprehensive college-level benchmark comprising 1,340 problems that span the
essential subdisciplines of materials science. MatSciBench features a
structured and fine-grained taxonomy that categorizes materials science
questions into 6 primary fields and 31 sub-fields, and includes a three-tier
difficulty classification based on the reasoning length required to solve each
question. MatSciBench provides detailed reference solutions enabling precise
error analysis and incorporates multimodal reasoning through visual contexts in
numerous questions. Evaluations of leading models reveal that even the
highest-performing model, Gemini-2.5-Pro, achieves under 80% accuracy on
college-level materials science questions, highlighting the complexity of
MatSciBench. Our systematic analysis of different reasoning strategie--basic
chain-of-thought, tool augmentation, and self-correction--demonstrates that no
single method consistently excels across all scenarios. We further analyze
performance by difficulty level, examine trade-offs between efficiency and
accuracy, highlight the challenges inherent in multimodal reasoning tasks,
analyze failure modes across LLMs and reasoning methods, and evaluate the
influence of retrieval-augmented generation. MatSciBench thus establishes a
comprehensive and solid benchmark for assessing and driving improvements in the
scientific reasoning capabilities of LLMs within the materials science domain.

### 7. [ResearStudio: A Human-Intervenable Framework for Building Controllable Deep-Research Agents](http://arxiv.org/pdf/2510.12194v1)

Authors: Linyi Yang, Yixuan Weng

Current deep-research agents run in a ''fire-and-forget'' mode: once started,
they give users no way to fix errors or add expert knowledge during execution.
We present ResearStudio, the first open-source framework that places real-time
human control at its core. The system follows a Collaborative Workshop design.
A hierarchical Planner-Executor writes every step to a live
''plan-as-document,'' a fast communication layer streams each action, file
change, and tool call to a web interface. At any moment, the user can pause the
run, edit the plan or code, run custom commands, and resume -- switching
smoothly between AI-led, human-assisted and human-led, AI-assisted modes. In
fully autonomous mode, ResearStudio achieves state-of-the-art results on the
GAIA benchmark, surpassing systems like OpenAI's DeepResearch and Manus. These
results show that strong automated performance and fine-grained human control
can coexist. The full code, protocol, and evaluation scripts are available at
https://github.com/ResearAI/ResearStudio. We will continue to update the
repository to encourage further work on safe and controllable research agents.
Our live demo is publicly accessible at http://ai-researcher.net:3000/. We
support the development of DeepScientist, which can be accessed at
https://github.com/ResearAI/DeepScientist.

### 8. [On the Design and Evaluation of Human-centered Explainable AI Systems: A Systematic Review and Taxonomy](http://arxiv.org/pdf/2510.12201v1)

Authors: Aline Mangold, Juliane Zietz, Susanne Weinhold, Sebastian Pannasch

As AI becomes more common in everyday living, there is an increasing demand
for intelligent systems that are both performant and understandable.
Explainable AI (XAI) systems aim to provide comprehensible explanations of
decisions and predictions. At present, however, evaluation processes are rather
technical and not sufficiently focused on the needs of human users.
Consequently, evaluation studies involving human users can serve as a valuable
guide for conducting user studies. This paper presents a comprehensive review
of 65 user studies evaluating XAI systems across different domains and
application contexts. As a guideline for XAI developers, we provide a holistic
overview of the properties of XAI systems and evaluation metrics focused on
human users (human-centered). We propose objectives for the human-centered
design (design goals) of XAI systems. To incorporate users' specific
characteristics, design goals are adapted to users with different levels of AI
expertise (AI novices and data experts). In this regard, we provide an
extension to existing XAI evaluation and design frameworks. The first part of
our results includes the analysis of XAI system characteristics. An important
finding is the distinction between the core system and the XAI explanation,
which together form the whole system. Further results include the distinction
of evaluation metrics into affection towards the system, cognition, usability,
interpretability, and explanation metrics. Furthermore, the users, along with
their specific characteristics and behavior, can be assessed. For AI novices,
the relevant extended design goals include responsible use, acceptance, and
usability. For data experts, the focus is performance-oriented and includes
human-AI collaboration and system and user task performance.

### 9. [GOAT: A Training Framework for Goal-Oriented Agent with Tools](http://arxiv.org/pdf/2510.12218v1)

Authors: Hyunji Min, Sangwon Jung, Junyoung Sung, Dosung Lee, Leekyeung Han, Paul Hongsuck Seo

Large language models (LLMs) have recently been extended beyond traditional
text generation to serve as interactive agents capable of using external tools
based on user intent. However, current LLM agents still show limited ability to
handle goal-oriented queries, which require decomposing a high-level objective
into multiple interdependent API calls with correct planning and execution.
Current approaches mainly rely on zero-shot evaluation due to the absence of
training data. While proprietary closed-source models such as GPT-4 demonstrate
strong reasoning abilities, smaller open-source models struggle to perform
complex tool use effectively. Thus, we propose a novel training framework GOAT,
which enables fine-tuning of LLM agents in a human annotation-free setting.
GOAT automatically constructs synthetic datasets of goal-oriented API execution
tasks directly from given API documents, equipping models with the ability to
reason over interdependent calls and generate coherent responses. Through
extensive experiments, we show that GOAT-trained agents achieve
state-of-the-art performance across multiple existing goal-oriented benchmarks.
In addition, we introduce GOATBench, a new goal-oriented API execution
benchmark, and demonstrate that agents trained with GOAT also excel in this
setting. These results highlight GOAT as a practical path toward building
robust open-source LLM agents capable of complex reasoning and tool use.

### 10. [MedKGEval: A Knowledge Graph-Based Multi-Turn Evaluation Framework for Open-Ended Patient Interactions with Clinical LLMs](http://arxiv.org/pdf/2510.12224v1)

Authors: Yuechun Yu, Han Ying, Haoan Jin, Wenjian Jiang, Dong Xian, Binghao Wang, Zhou Yang, Mengyue Wu

The reliable evaluation of large language models (LLMs) in medical
applications remains an open challenge, particularly in capturing the
complexity of multi-turn doctor-patient interactions that unfold in real
clinical environments. Existing evaluation methods typically rely on post hoc
review of full conversation transcripts, thereby neglecting the dynamic,
context-sensitive nature of medical dialogues and the evolving informational
needs of patients. In this work, we present MedKGEval, a novel multi-turn
evaluation framework for clinical LLMs grounded in structured medical
knowledge. Our approach introduces three key contributions: (1) a knowledge
graph-driven patient simulation mechanism, where a dedicated control module
retrieves relevant medical facts from a curated knowledge graph, thereby
endowing the patient agent with human-like and realistic conversational
behavior. This knowledge graph is constructed by integrating open-source
resources with additional triples extracted from expert-annotated datasets; (2)
an in-situ, turn-level evaluation framework, where each model response is
assessed by a Judge Agent for clinical appropriateness, factual correctness,
and safety as the dialogue progresses using a suite of fine-grained,
task-specific metrics; (3) a comprehensive multi-turn benchmark of eight
state-of-the-art LLMs, demonstrating MedKGEval's ability to identify subtle
behavioral flaws and safety risks that are often overlooked by conventional
evaluation pipelines. Although initially designed for Chinese and English
medical applications, our framework can be readily extended to additional
languages by switching the input knowledge graphs, ensuring seamless bilingual
support and domain-specific applicability.

### Hardware Architecture

### 1. [A Direct Memory Access Controller (DMAC) for Irregular Data Transfers on RISC-V Linux Systems](http://arxiv.org/pdf/2510.12277v1)

Authors: Thomas Benz, Axel Vanoni, Michael Rogenmoser, Luca Benini

With the ever-growing heterogeneity in computing systems, driven by modern
machine learning applications, pressure is increasing on memory systems to
handle arbitrary and more demanding transfers efficiently. Descriptor-based
direct memory access controllers (DMACs) allow such transfers to be executed by
decoupling memory transfers from processing units. Classical descriptor-based
DMACs are inefficient when handling arbitrary transfers of small unit sizes.
Excessive descriptor size and the serialized nature of processing descriptors
employed by the DMAC lead to large static overheads when setting up transfers.
To tackle this inefficiency, we propose a descriptor-based DMAC optimized to
efficiently handle arbitrary transfers of small unit sizes. We implement a
lightweight descriptor format in an AXI4-based DMAC. We further increase
performance by implementing a low-overhead speculative descriptor prefetching
scheme without additional latency penalties in the case of a misprediction. Our
DMAC is integrated into a 64-bit Linux-capable RISC-V SoC and emulated on a
Kintex FPGA to evaluate its performance. Compared to an off-the-shelf
descriptor-based DMAC IP, we achieve 1.66x less latency launching transfers,
increase bus utilization up to 2.5x in an ideal memory system with
64-byte-length transfers while requiring 11% fewer lookup tables, 23% fewer
flip-flops, and no block RAMs. We can extend our lead in bus utilization to
3.6x with 64-byte-length transfers in deep memory systems. We synthesized our
DMAC in GlobalFoundries' GF12LP+ node, achieving a clock frequency of over 1.44
GHz while occupying only 49.5 kGE.

### 2. [Wavefront Coding for Accommodation-Invariant Near-Eye Displays](http://arxiv.org/pdf/2510.12778v1)

Authors: Ugur Akpinar, Erdem Sahin, Tina M. Hayward, Apratim Majumder, Rajesh Menon, Atanas Gotchev

We present a new computational near-eye display method that addresses the
vergence-accommodation conflict problem in stereoscopic displays through
accommodation-invariance. Our system integrates a refractive lens eyepiece with
a novel wavefront coding diffractive optical element, operating in tandem with
a pre-processing convolutional neural network. We employ end-to-end learning to
jointly optimize the wavefront-coding optics and the image pre-processing
module. To implement this approach, we develop a differentiable retinal image
formation model that accounts for limiting aperture and chromatic aberrations
introduced by the eye optics. We further integrate the neural transfer function
and the contrast sensitivity function into the loss model to account for
related perceptual effects. To tackle off-axis distortions, we incorporate
position dependency into the pre-processing module. In addition to conducting
rigorous analysis based on simulations, we also fabricate the designed
diffractive optical element and build a benchtop setup, demonstrating
accommodation-invariance for depth ranges of up to four diopters.

### Computational Complexity

### 1. [Exact Matching and Top-k Perfect Matching Parameterized by Neighborhood Diversity or Bandwidth](http://arxiv.org/pdf/2510.12552v1)

Authors: Nicolas El Maalouly, Kostas Lakis

The Exact Matching (EM) problem asks whether there exists a perfect matching
which uses a prescribed number of red edges in a red/blue edge-colored graph.
While there exists a randomized polynomial-time algorithm for the problem, only
some special cases admit a deterministic one so far, making it a natural
candidate for testing the P=RP hypothesis. A polynomial-time equivalent
problem, Top-k Perfect Matching (TkPM), asks for a perfect matching maximizing
the weight of the $k$ heaviest edges.
  We study the above problems, mainly the latter, in the scenario where the
input is a blown-up graph, meaning a graph which had its vertices replaced by
cliques or independent sets. We describe an FPT algorithm for TkPM
parameterized by $k$ and the neighborhood diversity of the input graph, which
is essentially the size of the graph before the blow-up; this graph is also
called the prototype. We extend this algorithm into an approximation scheme
with a much softer dependency on the aforementioned parameters, time-complexity
wise. Moreover, for prototypes with bounded bandwidth but unbounded size, we
develop a recursive algorithm that runs in subexponential time. Utilizing
another algorithm for EM on bounded neighborhood diversity graphs, we adapt
this recursive subexponential algorithm to EM.
  Our approach is similar to the use of dynamic programming on e.g. bounded
treewidth instances for various problems. The main point is that the existence
of many disjoint separators is utilized to avoid including in the separator any
of a set of ``bad'' vertices during the split phase.

### 2. [Tight Quantum Time-Space Tradeoffs for Permutation Inversion](http://arxiv.org/pdf/2510.12112v1)

Authors: Akshima, Tyler Besselman, Kai-Min Chung, Siyao Guo, Tzu-Yi Yang

In permutation inversion, we are given a permutation $\pi : [N] \rightarrow
[N]$, and want to prepare some advice of size $S$, such that we can efficiently
invert any image in time $T$. This is a fundamental cryptographic problem with
profound connections to communication complexity and circuit lower bounds.
  In the classical setting, a tight $ST = \tilde{\Theta}(N)$ bound has been
established since the seminal work of Hellman (1980) and Yao (1990). In the
quantum setting, a lower bound of $ST^2 = \tilde{\Omega}(N)$ is proved by
Nayebi, Aaronson, Belovs, and Trevisan (2015) against classical advice, and by
Hhan, Xagawa and Yamakawa (2019) against quantum advice. It left open an
intriguing possibility that Grover's search can be sped up to time
$\tilde{O}(\sqrt{N / S})$.
  In this work, we prove an $ST + T^2 = \Omega(N)$ lower bound for permutation
inversion with even quantum advice. This bound matches the best known attacks
and shows that Grover's search and the classical Hellman's algorithm cannot be
further sped up.
  Our proof combines recent techniques by Liu (2023) and by Rosmanis (2022).
Specifically, we first reduce the permutation inversion problem against quantum
advice to a variant by Liu's technique, then we analyze this variant via
representation theory inspired by Rosmanis (2022).

### 3. [Performance of Gaussian Boson Sampling on Planted Bipartite Clique Detection](http://arxiv.org/pdf/2510.12774v1)

Authors: Yu-Zhen Janice Chen, Laurent Massoulié, Don Towsley

We investigate whether Gaussian Boson Sampling (GBS) can provide a
computational advantage for solving the planted biclique problem, which is a
graph problem widely believed to be classically hard when the planted structure
is small. Although GBS has been heuristically and experimentally observed to
favor sampling dense subgraphs, its theoretical performance on this classically
hard problem remains largely unexplored. We focus on a natural statistic
derived from GBS output: the frequency with which a node appears in GBS
samples, referred to as the node weight. We rigorously analyze whether this
signal is strong enough to distinguish planted biclique nodes from background
nodes. Our analysis characterizes the distribution of node weights under GBS
and quantifies the bias introduced by the planted structure. The results reveal
a sharp limitation: when the planted biclique size falls within the conjectured
hard regime, the natural fluctuations in node weights dominate the bias signal,
making detection unreliable using simple ranking strategies. These findings
provide the first rigorous evidence that planted biclique detection may remain
computationally hard even under GBS-based quantum computing, and they motivate
further investigation into more advanced GBS-based algorithms or other quantum
approaches for this problem.

### Computational Engineering

### 1. [RAID-0e: A Resilient Striping Array Architecture for Balanced Performance and Availability](http://arxiv.org/pdf/2510.12139v1)

Authors: Yanzhao Jia, Zhaobo Wu, Zheyi Cao, Shihao Ji, Xu Tianhao, Zihui Song

This paper introduces a novel disk array architecture, designated RAID-0e
(Resilient Striping Array), designed to superimpose a low-overhead fault
tolerance layer upon traditional RAID 0 (striping). By employing a logically
and physically separate parity domain to protect a primary data domain, RAID-0e
mitigates the risk of array-wide data loss from common, non-catastrophic media
failures, such as isolated bad blocks, transient read errors, or sector-level
corruption. The architecture is engineered to preserve the intrinsic read
performance advantages of RAID 0 while significantly enhancing data
availability and operational resilience. This document provides a comprehensive
exposition of the architectural principles, operational workflows, performance
characteristics, failure mode analysis, and security considerations of RAID-0e.
It is presented as an experimental yet pragmatic solution for environments
seeking a new equilibrium between I/O performance, storage cost, and data
resilience, particularly where full drive failure is a secondary concern to
media degradation.

### 2. [Agent-Based Simulation of a Financial Market with Large Language Models](http://arxiv.org/pdf/2510.12189v1)

Authors: Ryuji Hashimoto, Takehiro Takayanagi, Masahiro Suzuki, Kiyoshi Izumi

In real-world stock markets, certain chart patterns -- such as price declines
near historical highs -- cannot be fully explained by fundamentals alone. These
phenomena suggest the presence of path dependence in price formation, where
investor decisions are influenced not only by current market conditions but
also by the trajectory of prices leading up to the present. Path dependence has
drawn attention in behavioral finance as a key mechanism behind such anomalies.
One plausible driver of path dependence is human loss aversion, anchored to
individual reference points like purchase prices or past peaks, which vary with
personal context. However, capturing such subtle behavioral tendencies in
traditional agent-based market simulations has remained a challenge. We propose
the Fundamental-Chartist-LLM-Agent (FCLAgent), which uses large language models
(LLMs) to emulate human-like trading decisions. In this framework, (1) buy/sell
decisions are made by LLMs based on individual situations, while (2) order
price and volume follow standard rule-based methods. Simulations show that
FCLAgents reproduce path-dependent patterns that conventional agents fail to
capture. Furthermore, an analysis of FCLAgents' behavior reveals that the
reference points guiding loss aversion vary with market trajectories,
highlighting the potential of LLM-based agents to model nuanced investor
behavior.

### 3. [Proceedings of the International Workshop on Verification of Scientific Software](http://arxiv.org/pdf/2510.12314v1)

Authors: Stephen F. Siegel, Ganesh Gopalakrishnan

This volume contains the proceedings of the Verification of Scientific
Software (VSS 2025) workshop, held on 4 May 2025 at McMaster University,
Canada, as part of ETAPS 2025. VSS brings together researchers in software
verification and scientific computing to address challenges in ensuring the
correctness and reliability of large-scale scientific codes. The program
featured five peer-reviewed papers, three invited contributions, and a set of
challenge problems, covering themes such as deductive verification,
floating-point error analysis, specification of coupled models, and
domain-aware testing. VSS builds on the Correctness Workshop series at
Supercomputing and the 2023 NSF/DOE report on scientific software correctness.
It serves as yet another snapshot of this important area, showcasing a wide
range of perspectives, problems and their solutions in progress, with the
challenge problems having the potential to bring together separate verification
tools into concerted action.

### 4. [Constrained Sensing and Reliable State Estimation with Shallow Recurrent Decoders on a TRIGA Mark II Reactor](http://arxiv.org/pdf/2510.12368v1)

Authors: Stefano Riva, Carolina Introini, Josè Nathan Kutz, Antonio Cammi

Shallow Recurrent Decoder networks are a novel data-driven methodology able
to provide accurate state estimation in engineering systems, such as nuclear
reactors. This deep learning architecture is a robust technique designed to map
the temporal trajectories of a few sparse measures to the full state space,
including unobservable fields, which is agnostic to sensor positions and able
to handle noisy data through an ensemble strategy, leveraging the short
training times and without the need for hyperparameter tuning. Following its
application to a novel reactor concept, this work investigates the performance
of Shallow Recurrent Decoders when applied to a real system. The underlying
model is represented by a fluid dynamics model of the TRIGA Mark II research
reactor; the architecture will use both synthetic temperature data coming from
the numerical model and leveraging experimental temperature data recorded
during a previous campaign. The objective of this work is, therefore, two-fold:
1) assessing if the architecture can reconstruct the full state of the system
(temperature, velocity, pressure, turbulence quantities) given sparse data
located in specific, low-dynamics channels and 2) assessing the correction
capabilities of the architecture (that is, given a discrepancy between model
and data, assessing if sparse measurements can provide some correction to the
architecture output). As will be shown, the accurate reconstruction of every
characteristic field, using both synthetic and experimental data, in real-time
makes this approach suitable for interpretable monitoring and control purposes
in the framework of a reactor digital twin.

### Computational Geometry

### 1. [Topological Signatures of ReLU Neural Network Activation Patterns](http://arxiv.org/pdf/2510.12700v1)

Authors: Vicente Bosca, Tatum Rask, Sunia Tanweer, Andrew R. Tawfeek, Branden Stone

This paper explores the topological signatures of ReLU neural network
activation patterns. We consider feedforward neural networks with ReLU
activation functions and analyze the polytope decomposition of the feature
space induced by the network. Mainly, we investigate how the Fiedler partition
of the dual graph and show that it appears to correlate with the decision
boundary -- in the case of binary classification. Additionally, we compute the
homology of the cellular decomposition -- in a regression task -- to draw
similar patterns in behavior between the training loss and polyhedral
cell-count, as the model is trained.

### Computation and Language

### 1. [Information Extraction from Conversation Transcripts: Neuro-Symbolic vs. LLM](http://arxiv.org/pdf/2510.12023v1)

Authors: Alice Saebom Kwak, Maria Alexeeva, Gus Hahn-Powell, Keith Alcock, Kevin McLaughlin, Doug McCorkle, Gabe McNunn, Mihai Surdeanu

The current trend in information extraction (IE) is to rely extensively on
large language models, effectively discarding decades of experience in building
symbolic or statistical IE systems. This paper compares a neuro-symbolic (NS)
and an LLM-based IE system in the agricultural domain, evaluating them on nine
interviews across pork, dairy, and crop subdomains. The LLM-based system
outperforms the NS one (F1 total: 69.4 vs. 52.7; core: 63.0 vs. 47.2), where
total includes all extracted information and core focuses on essential details.
However, each system has trade-offs: the NS approach offers faster runtime,
greater control, and high accuracy in context-free tasks but lacks
generalizability, struggles with contextual nuances, and requires significant
resources to develop and maintain. The LLM-based system achieves higher
performance, faster deployment, and easier maintenance but has slower runtime,
limited control, model dependency and hallucination risks. Our findings
highlight the "hidden cost" of deploying NLP systems in real-world
applications, emphasizing the need to balance performance, efficiency, and
control.

### 2. [On the Interplay between Human Label Variation and Model Fairness](http://arxiv.org/pdf/2510.12036v1)

Authors: Kemal Kurniawan, Meladel Mistica, Timothy Baldwin, Jey Han Lau

The impact of human label variation (HLV) on model fairness is an unexplored
topic. This paper examines the interplay by comparing training on majority-vote
labels with a range of HLV methods. Our experiments show that without explicit
debiasing, HLV training methods have a positive impact on fairness.

### 3. [Uncertainty Quantification for Hallucination Detection in Large Language Models: Foundations, Methodology, and Future Directions](http://arxiv.org/pdf/2510.12040v1)

Authors: Sungmin Kang, Yavuz Faruk Bakman, Duygu Nur Yaldiz, Baturalp Buyukates, Salman Avestimehr

The rapid advancement of large language models (LLMs) has transformed the
landscape of natural language processing, enabling breakthroughs across a wide
range of areas including question answering, machine translation, and text
summarization. Yet, their deployment in real-world applications has raised
concerns over reliability and trustworthiness, as LLMs remain prone to
hallucinations that produce plausible but factually incorrect outputs.
Uncertainty quantification (UQ) has emerged as a central research direction to
address this issue, offering principled measures for assessing the
trustworthiness of model generations. We begin by introducing the foundations
of UQ, from its formal definition to the traditional distinction between
epistemic and aleatoric uncertainty, and then highlight how these concepts have
been adapted to the context of LLMs. Building on this, we examine the role of
UQ in hallucination detection, where quantifying uncertainty provides a
mechanism for identifying unreliable generations and improving reliability. We
systematically categorize a wide spectrum of existing methods along multiple
dimensions and present empirical results for several representative approaches.
Finally, we discuss current limitations and outline promising future research
directions, providing a clearer picture of the current landscape of LLM UQ for
hallucination detection.

### 4. [Improving Text-to-Image Generation with Input-Side Inference-Time Scaling](http://arxiv.org/pdf/2510.12041v1)

Authors: Ruibo Chen, Jiacheng Pan, Heng Huang, Zhenheng Yang

Recent advances in text-to-image (T2I) generation have achieved impressive
results, yet existing models often struggle with simple or underspecified
prompts, leading to suboptimal image-text alignment, aesthetics, and quality.
We propose a prompt rewriting framework that leverages large language models
(LLMs) to refine user inputs before feeding them into T2I backbones. Our
approach introduces a carefully designed reward system and an iterative direct
preference optimization (DPO) training pipeline, enabling the rewriter to
enhance prompts without requiring supervised fine-tuning data. We evaluate our
method across diverse T2I models and benchmarks. Results show that our prompt
rewriter consistently improves image-text alignment, visual quality, and
aesthetics, outperforming strong baselines. Furthermore, we demonstrate strong
transferability by showing that a prompt rewriter trained on one T2I backbone
generalizes effectively to others without needing to be retrained. We also
systematically study scalability, evaluating how performance gains scale with
the capacity of the large LLM used as the rewriter. These findings highlight
that prompt rewriting is an effective, scalable, and practical model-agnostic
strategy for improving T2I systems. We plan to release the code and trained
prompt rewriters soon.

### 5. [Tracing Multilingual Knowledge Acquisition Dynamics in Domain Adaptation: A Case Study of English-Japanese Biomedical Adaptation](http://arxiv.org/pdf/2510.12115v1)

Authors: Xin Zhao, Naoki Yoshinaga, Yuma Tsuta, Akiko Aizawa

Multilingual domain adaptation (ML-DA) is widely used to learn new domain
knowledge across languages into large language models (LLMs). Although many
methods have been proposed to improve domain adaptation, the mechanisms of
multilingual knowledge acquisition, how domain knowledge is learned within a
language and transferred across languages, remain underexplored. This gap leads
to suboptimal performance, particularly in low-resource settings. This work
examines the learning dynamics of LLMs during ML-DA. Because prior ML-DA
studies often train and evaluate on datasets with mismatched knowledge
coverage, we propose AdaXEval, an adaptive evaluation method that builds
multiple-choice QA datasets from the same bilingual domain corpus used for
training, thereby directly studying multilingual knowledge acquisition. Through
continual training of LLMs with diverse data recipes, we track how LLMs acquire
domain facts and pinpoint the mechanism behind the transformation process from
domain training data to knowledge. Our experiments on a 13B English-Japanese
bilingual LLM reveal that cross-lingual transfer remains challenging despite a
high-quality bilingual corpus. The code has been released.

### 6. [A Survey on Parallel Reasoning](http://arxiv.org/pdf/2510.12164v1)

Authors: Ziqi Wang, Boye Niu, Zipeng Gao, Zhi Zheng, Tong Xu, Linghui Meng, Zhongli Li, Jing Liu, Yilong Chen, Chen Zhu, Hua Wu, Haifeng Wang, Enhong Chen

With the increasing capabilities of Large Language Models (LLMs), parallel
reasoning has emerged as a new inference paradigm that enhances reasoning
robustness by concurrently exploring multiple lines of thought before
converging on a final answer. It has become a significant trend to explore
parallel reasoning to overcome the fragility of standard sequential methods and
improve practical performance. In this paper, we aim to survey and summarize
the progress and challenges of parallel reasoning. We first present a formal
definition of parallel reasoning and clarify its distinction from related
concepts like Chain-of-Thought. Then, we organize and discuss advanced
techniques based on a novel taxonomy, including non-interactive reasoning,
interactive reasoning, and efficiency-focused decoding strategies.
Additionally, we explore various application scenarios, such as solving complex
problems and enhancing the reliability of LLM outputs.Finally, we highlight the
core challenges of parallel reasoning and suggest potential directions for
future research. We hope that our work can provide a useful roadmap for
beginners and encourage more research on improving parallel reasoning methods.
Related source can be avaliable in
https://github.com/PPPP-kaqiu/Awesome-Parallel-Reasoning.

### 7. [Towards Inference-time Scaling for Continuous Space Reasoning](http://arxiv.org/pdf/2510.12167v1)

Authors: Minghan Wang, Thuy-Trang Vu, Ehsan Shareghi, Gholamreza Haffari

Inference-time scaling through multiple sample generation in combination with
Process- or Outcome-Reward Model (PRM or ORM) re-ranking has proven effective
for text-based reasoning in large language models. This paper investigates
whether such established techniques can be successfully adapted to reasoning in
the continuous space, using COCONUT (Hao et al. 2024) continuous space
reasoning LM as the backbone. We demonstrate the feasibility of generating
diverse reasoning paths through dropout-based sampling. Our Pass@N analysis on
the generated samples reveals the potential that could enable a significant
gain in performance akin to observed gain in the discrete space. However, we
highlight unique challenges faced for materializing this gain in the continuous
thought space. In particular, working recipes for data generation and training
PRM and ORM models in the discrete space unlocks only marginal improvements in
the continuous space. Through probing various aspects including geometric
properties and trajectory dynamics we identify the underlying reasons that
prevent effective discrimination between correct and incorrect reasoning
(essential for the functioning of PRM and ORM). Our findings reveal that
current limitations stem from the absence of key inductive biases in continuous
thought representations. We argue that the training frameworks for continuous
reasoning LMs require not only to optimize for accuracy but also to explicitly
incorporate inductive biases that could be utilized during inference-time for
discrimination of correct and incorrect thoughts.\footnote{Our code and data
will be publicly available.}

### 8. [DPO-Tuned Large Language Models for Segmentation in Simultaneous Speech Translation](http://arxiv.org/pdf/2510.12195v1)

Authors: Zeyu Yang, Satoshi Nakamura

Simultaneous speech translation requires accurate segmentation to balance
translation quality and latency. Recent studies such as SHAS have introduced
pretrained segmentation models, achieving stronger performance than heuristic
rules. However, segmentation models such as SHAS, though pretrained and more
robust than heuristic methods, are still constrained by supervised learning
objectives and do not incorporate human preference alignment, which is crucial
for natural real-time interpretation. In this work, we propose a segmentation
framework based on large language models (LLMs) trained with Direct Preference
Optimization (DPO). By leveraging preference alignment, our method enables LLMs
to predict natural segmentation points that better meet the demands of
real-time translation. We evaluate the system on the ACL 60/60 corpus across
three language pairs (English-Japanese, Chinese, German), using SeamlessM4T v2
as the translation backbone. Experimental results show that our DPO-tuned LLM
achieves higher segmentation accuracy than SHAS and yields consistent
improvements in translation quality (BLEU, COMET) as well as latency (Average
Lagging). Furthermore, our system benefits from IWSLT baselines for direct
comparison. These findings highlight the potential of preference-tuned LLMs to
surpass existing pretrained segmentation models and advance adaptive,
human-aligned simultaneous interpretation.

### 9. [DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](http://arxiv.org/pdf/2510.12251v1)

Authors: Jiakai Li, Rongzheng Wang, Yizhuo Ma, Shuang Liang, Guangchun Luo, Ke Qin

While large language models (LLMs) show considerable promise across various
fields, they have notable limitations in handling multi-document question
answering (Multi-doc QA) tasks. The first challenge is long-range dependency
modeling, where LLMs struggle to focus on key information in long texts, which
weakens important semantic connections. Second, most LLMs suffer from the
''lost-in-the-middle'' issue, where they have difficulty processing information
in the middle of long inputs. Current solutions either truncate global
dependencies or demand costly finetuning, ultimately lacking a universal and
simple solution for these challenges. To resolve these limitations, we propose
Dual-Stage Adaptive Sharpening (DSAS) containing two modules. (i) The
Contextual Gate Weighting (CGW) module alleviates ''lost-in-the-middle'' by
assessing paragraph relevance through layer-wise attention tracking and
position-aware weighting. (ii) The Reciprocal Attention Suppression (RAS)
module enhances focus on critical paragraphs by suppressing information
exchange between key and irrelevant texts, thus mitigating the limitations in
long-range dependency modeling. Notably, DSAS functions as a plug-and-play
solution requiring no architectural modifications or extra training parameters.
Extensive experiments on four benchmarks demonstrate DSAS's efficacy across
mainstream LLMs (Llama, Qwen, Mistral, and Deepseek), with an average F1-score
improvement of 4.2% in Multi-doc QA tasks on Llama-3.1-8B-Instruct and
Qwen2.5-14B-Instruct. Ablation studies confirm the essential contributions of
both the CGW and RAS modules. In addition, detailed discussions in the Appendix
further validate the robustness and scalability of DSAS.

### 10. [A large-scale, unsupervised pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction](http://arxiv.org/pdf/2510.12306v1)

Authors: Cameron Morin, Matti Marttinen Larsson

As natural language corpora expand at an unprecedented rate, manual
annotation remains a significant methodological bottleneck in corpus linguistic
work. We address this challenge by presenting a scalable, unsupervised pipeline
for automating grammatical annotation in voluminous corpora using large
language models (LLMs). Unlike previous supervised and iterative approaches,
our method employs a four-phase workflow: prompt engineering, pre-hoc
evaluation, automated batch processing, and post-hoc validation. We demonstrate
the pipeline's accessibility and effectiveness through a diachronic case study
of variation in the English consider construction. Using GPT-5 through the
OpenAI API, we annotate 143,933 sentences from the Corpus of Historical
American English (COHA) in under 60 hours, achieving 98%+ accuracy on two
sophisticated annotation procedures. Our results suggest that LLMs can perform
a range of data preparation tasks at scale with minimal human intervention,
opening new possibilities for corpus-based research, though implementation
requires attention to costs, licensing, and other ethical considerations.

### Cryptography and Security

### 1. [Security and Privacy Assessment of U.S. and Non-U.S. Android E-Commerce Applications](http://arxiv.org/pdf/2510.12031v1)

Authors: Urvashi Kishnani, Sanchari Das

E-commerce mobile applications are central to global financial transactions,
making their security and privacy crucial. In this study, we analyze 92
top-grossing Android e-commerce apps (58 U.S.-based and 34 international) using
MobSF, AndroBugs, and RiskInDroid. Our analysis shows widespread SSL and
certificate weaknesses, with approximately 92% using unsecured HTTP connections
and an average MobSF security score of 40.92/100. Over-privileged permissions
were identified in 77 apps. While U.S. apps exhibited fewer manifest, code, and
certificate vulnerabilities, both groups showed similar network-related issues.
We advocate for the adoption of stronger, standardized, and user-focused
security practices across regions.

### 2. [Adding All Flavors: A Hybrid Random Number Generator for dApps and Web3](http://arxiv.org/pdf/2510.12062v1)

Authors: Ranjith Chodavarapu, Rabimba Karanjai, Xinxin Fan, Weidong Shi, Lei Xu

Random numbers play a vital role in many decentralized applications (dApps),
such as gaming and decentralized finance (DeFi) applications.
  Existing random number provision mechanisms can be roughly divided into two
categories, on-chain, and off-chain.
  On-chain approaches usually rely on the blockchain as the major input and all
computations are done by blockchain nodes.
  The major risk for this type of method is that the input itself is
susceptible to the adversary's influence.
  Off-chain approaches, as the name suggested, complete the generation without
the involvement of blockchain nodes and share the result directly with a dApp.
  These mechanisms usually have a strong security assumption and high
complexity.
  To mitigate these limitations and provide a framework that allows a dApp to
balance different factors involved in random number generation, we propose a
hybrid random number generation solution that leverages IoT devices equipped
with trusted execution environment (TEE) as the randomness sources, and then
utilizes a set of cryptographic tools to aggregate the multiple sources and
obtain the final random number that can be consumed by the dApp.
  The new approach only needs one honest random source to guarantee the
unbiasedness of the final random number and a user can configure the system to
tolerate malicious participants who can refuse to respond to avoid unfavored
results.
  We also provide a concrete construction that can further reduce the on-chain
computation complexity to lower the cost of the solution in practice.
  We evaluate the computation and gas costs to demonstrate the effectiveness of
the improvement.

### 3. [Elevating Medical Image Security: A Cryptographic Framework Integrating Hyperchaotic Map and GRU](http://arxiv.org/pdf/2510.12084v1)

Authors: Weixuan Li, Guang Yu, Quanjun Li, Junhua Zhou, Jiajun Chen, Yihang Dong, Mengqian Wang, Zimeng Li, Changwei Gong, Lin Tang, Xuhang Chen

Chaotic systems play a key role in modern image encryption due to their
sensitivity to initial conditions, ergodicity, and complex dynamics. However,
many existing chaos-based encryption methods suffer from vulnerabilities, such
as inadequate permutation and diffusion, and suboptimal pseudorandom
properties. This paper presents Kun-IE, a novel encryption framework designed
to address these issues. The framework features two key contributions: the
development of the 2D Sin-Cos Pi Hyperchaotic Map (2D-SCPHM), which offers a
broader chaotic range and superior pseudorandom sequence generation, and the
introduction of Kun-SCAN, a novel permutation strategy that significantly
reduces pixel correlations, enhancing resistance to statistical attacks. Kun-IE
is flexible and supports encryption for images of any size. Experimental
results and security analyses demonstrate its robustness against various
cryptanalytic attacks, making it a strong solution for secure image
communication. The code is available at this
\href{https://github.com/QuincyQAQ/Elevating-Medical-Image-Security-A-Cryptographic-Framework-Integrating-Hyperchaotic-Map-and-GRU}{link}.

### 4. [VeilAudit: Breaking the Deadlock Between Privacy and Accountability Across Blockchains](http://arxiv.org/pdf/2510.12153v1)

Authors: Minhao Qiao, Iqbal Gondal, Hai Dong

Cross chain interoperability in blockchain systems exposes a fundamental
tension between user privacy and regulatory accountability. Existing solutions
enforce an all or nothing choice between full anonymity and mandatory identity
disclosure, which limits adoption in regulated financial settings. We present
VeilAudit, a cross chain auditing framework that introduces Auditor Only
Linkability, which allows auditors to link transaction behaviors that originate
from the same anonymous entity without learning its identity. VeilAudit
achieves this with a user generated Linkable Audit Tag that embeds a zero
knowledge proof to attest to its validity without exposing the user master
wallet address, and with a special ciphertext that only designated auditors can
test for linkage. To balance privacy and compliance, VeilAudit also supports
threshold gated identity revelation under due process. VeilAudit further
provides a mechanism for building reputation in pseudonymous environments,
which enables applications such as cross chain credit scoring based on
verifiable behavioral history. We formalize the security guarantees and develop
a prototype that spans multiple EVM chains. Our evaluation shows that the
framework is practical for today multichain environments.

### 5. [Leaking Queries On Secure Stream Processing Systems](http://arxiv.org/pdf/2510.12172v1)

Authors: Hung Pham, Viet Vo, Tien Tuan Anh Dinh, Duc Tran, Shuhao Zhang

Stream processing systems are important in modern applications in which data
arrive continuously and need to be processed in real time. Because of their
resource and scalability requirements, many of these systems run on the cloud,
which is considered untrusted. Existing works on securing databases on the
cloud focus on protecting the data, and most systems leverage trusted hardware
for high performance. However, in stream processing systems, queries are as
sensitive as the data because they contain the application logics.
  We demonstrate that it is practical to extract the queries from stream
processing systems that use Intel SGX for securing the execution engine. The
attack performed by a malicious cloud provider is based on timing side
channels, and it works in two phases. In the offline phase, the attacker
profiles the execution time of individual stream operators, based on synthetic
data. This phase outputs a model that identifies individual stream operators.
In the online phase, the attacker isolates the operators that make up the
query, monitors its execution, and recovers the operators using the model in
the previous phase. We implement the attack based on popular data stream
benchmarks using SecureStream and NEXMark, and demonstrate attack success rates
of up to 92%. We further discuss approaches that can harden streaming
processing systems against our attacks without incurring high overhead.

### 6. [IP-Augmented Multi-Modal Malicious URL Detection Via Token-Contrastive Representation Enhancement and Multi-Granularity Fusion](http://arxiv.org/pdf/2510.12395v1)

Authors: Ye Tian, Yanqiu Yu, Liangliang Song, Zhiquan Liu, Yanbin Wang, Jianguo Sun

Malicious URL detection remains a critical cybersecurity challenge as
adversaries increasingly employ sophisticated evasion techniques including
obfuscation, character-level perturbations, and adversarial attacks. Although
pre-trained language models (PLMs) like BERT have shown potential for URL
analysis tasks, three limitations persist in current implementations: (1)
inability to effectively model the non-natural hierarchical structure of URLs,
(2) insufficient sensitivity to character-level obfuscation, and (3) lack of
mechanisms to incorporate auxiliary network-level signals such as IP
addresses-all essential for robust detection. To address these challenges, we
propose CURL-IP, an advanced multi-modal detection framework incorporating
three key innovations: (1) Token-Contrastive Representation Enhancer, which
enhances subword token representations through token-aware contrastive learning
to produce more discriminative and isotropic embeddings; (2) Cross-Layer
Multi-Scale Aggregator, employing hierarchical aggregation of Transformer
outputs via convolutional operations and gated MLPs to capture both local and
global semantic patterns across layers; and (3) Blockwise Multi-Modal Coupler
that decomposes URL-IP features into localized block units and computes
cross-modal attention weights at the block level, enabling fine-grained
inter-modal interaction. This architecture enables simultaneous preservation of
fine-grained lexical cues, contextual semantics, and integration of
network-level signals. Our evaluation on large-scale real-world datasets shows
the framework significantly outperforms state-of-the-art baselines across
binary and multi-class classification tasks.

### 7. [Attack-Specialized Deep Learning with Ensemble Fusion for Network Anomaly Detection](http://arxiv.org/pdf/2510.12455v1)

Authors: Nisith Dissanayake, Uthayasanker Thayasivam

The growing scale and sophistication of cyberattacks pose critical challenges
to network security, particularly in detecting diverse intrusion types within
imbalanced datasets. Traditional intrusion detection systems (IDS) often
struggle to maintain high accuracy across both frequent and rare attacks,
leading to increased false negatives for minority classes. To address this, we
propose a hybrid anomaly detection framework that integrates specialized deep
learning models with an ensemble meta-classifier. Each model is trained to
detect a specific attack category, enabling tailored learning of class-specific
patterns, while their collective outputs are fused by a Random Forest
meta-classifier to improve overall decision reliability. The framework is
evaluated on the NSL-KDD benchmark, demonstrating superior performance in
handling class imbalance compared to conventional monolithic models. Results
show significant improvements in precision, recall, and F1-score across all
attack categories, including rare classes such as User to Root (U2R). The
proposed system achieves near-perfect detection rates with minimal false
alarms, highlighting its robustness and generalizability. This work advances
the design of intrusion detection systems by combining specialization with
ensemble learning, providing an effective and scalable solution for
safeguarding modern networks.

### 8. [PromoGuardian: Detecting Promotion Abuse Fraud with Multi-Relation Fused Graph Neural Networks](http://arxiv.org/pdf/2510.12652v1)

Authors: Shaofei Li, Xiao Han, Ziqi Zhang, Minyao Hua, Shuli Gao, Zhenkai Liang, Yao Guo, Xiangqun Chen, Ding Li

As e-commerce platforms develop, fraudulent activities are increasingly
emerging, posing significant threats to the security and stability of these
platforms. Promotion abuse is one of the fastest-growing types of fraud in
recent years and is characterized by users exploiting promotional activities to
gain financial benefits from the platform. To investigate this issue, we
conduct the first study on promotion abuse fraud in e-commerce platforms
MEITUAN. We find that promotion abuse fraud is a group-based fraudulent
activity with two types of fraudulent activities: Stocking Up and Cashback
Abuse. Unlike traditional fraudulent activities such as fake reviews, promotion
abuse fraud typically involves ordinary customers conducting legitimate
transactions and these two types of fraudulent activities are often
intertwined. To address this issue, we propose leveraging additional
information from the spatial and temporal perspectives to detect promotion
abuse fraud. In this paper, we introduce PROMOGUARDIAN, a novel multi-relation
fused graph neural network that integrates the spatial and temporal information
of transaction data into a homogeneous graph to detect promotion abuse fraud.
We conduct extensive experiments on real-world data from MEITUAN, and the
results demonstrate that our proposed model outperforms state-of-the-art
methods in promotion abuse fraud detection, achieving 93.15% precision,
detecting 2.1 to 5.0 times more fraudsters, and preventing 1.5 to 8.8 times
more financial losses in production environments.

### 9. [Hash chaining degrades security at Facebook](http://arxiv.org/pdf/2510.12665v1)

Authors: Thomas Rivasseau

Modern web and digital application password storage relies on password
hashing for storage and security. Ad-hoc upgrade of password storage to keep up
with hash algorithm norms may be used to save costs but can introduce
unforeseen vulnerabilities. This is the case in the password storage scheme
used by Meta Platforms which services several billion monthly users worldwide.
In this paper we present the first example of an exploit which demonstrates the
security weakness of Facebook's password storage scheme, and discuss its
implications. Proper ethical disclosure guidelines and vendor notification were
followed.

### 10. [Over-Threshold Multiparty Private Set Intersection for Collaborative Network Intrusion Detection](http://arxiv.org/pdf/2510.12045v1)

Authors: Onur Eren Arpaci, Raouf Boutaba, Florian Kerschbaum

An important function of collaborative network intrusion detection is to
analyze the network logs of the collaborators for joint IP addresses. However,
sharing IP addresses in plain is sensitive and may be even subject to privacy
legislation as it is personally identifiable information. In this paper, we
present the privacy-preserving collection of IP addresses. We propose a single
collector, over-threshold private set intersection protocol. In this protocol
$N$ participants identify the IP addresses that appear in at least $t$
participant's sets without revealing any information about other IP addresses.
Using a novel hashing scheme, we reduce the computational complexity of the
previous state-of-the-art solution from $O(M(N \log{M}/t)^{2t})$ to
$O(t^2M\binom{N}{t})$, where $M$ denotes the dataset size. This reduction makes
it practically feasible to apply our protocol to real network logs. We test our
protocol using joint networks logs of multiple institutions. Additionally, we
present two deployment options: a collusion-safe deployment, which provides
stronger security guarantees at the cost of increased communication overhead,
and a non-interactive deployment, which assumes a non-colluding collector but
offers significantly lower communication costs and applicable to many use cases
of collaborative network intrusion detection similar to ours.

### Computer Vision and Pattern Recognition

### 1. [APGNet: Adaptive Prior-Guided for Underwater Camouflaged Object Detection](http://arxiv.org/pdf/2510.12056v1)

Authors: Xinxin Huang, Han Sun, Junmin Cai, Ningzhong Liu, Huiyu Zhou

Detecting camouflaged objects in underwater environments is crucial for
marine ecological research and resource exploration. However, existing methods
face two key challenges: underwater image degradation, including low contrast
and color distortion, and the natural camouflage of marine organisms.
Traditional image enhancement techniques struggle to restore critical features
in degraded images, while camouflaged object detection (COD) methods developed
for terrestrial scenes often fail to adapt to underwater environments due to
the lack of consideration for underwater optical characteristics.
  To address these issues, we propose APGNet, an Adaptive Prior-Guided Network,
which integrates a Siamese architecture with a novel prior-guided mechanism to
enhance robustness and detection accuracy. First, we employ the Multi-Scale
Retinex with Color Restoration (MSRCR) algorithm for data augmentation,
generating illumination-invariant images to mitigate degradation effects.
Second, we design an Extended Receptive Field (ERF) module combined with a
Multi-Scale Progressive Decoder (MPD) to capture multi-scale contextual
information and refine feature representations. Furthermore, we propose an
adaptive prior-guided mechanism that hierarchically fuses position and boundary
priors by embedding spatial attention in high-level features for coarse
localization and using deformable convolution to refine contours in low-level
features.
  Extensive experimental results on two public MAS datasets demonstrate that
our proposed method APGNet outperforms 15 state-of-art methods under widely
used evaluation metrics.

### 2. [VIDMP3: Video Editing by Representing Motion with Pose and Position Priors](http://arxiv.org/pdf/2510.12069v1)

Authors: Sandeep Mishra, Oindrila Saha, Alan C. Bovik

Motion-preserved video editing is crucial for creators, particularly in
scenarios that demand flexibility in both the structure and semantics of
swapped objects. Despite its potential, this area remains underexplored.
Existing diffusion-based editing methods excel in structure-preserving tasks,
using dense guidance signals to ensure content integrity. While some recent
methods attempt to address structure-variable editing, they often suffer from
issues such as temporal inconsistency, subject identity drift, and the need for
human intervention. To address these challenges, we introduce VidMP3, a novel
approach that leverages pose and position priors to learn a generalized motion
representation from source videos. Our method enables the generation of new
videos that maintain the original motion while allowing for structural and
semantic flexibility. Both qualitative and quantitative evaluations demonstrate
the superiority of our approach over existing methods. The code will be made
publicly available at https://github.com/sandeep-sm/VidMP3.

### 3. [Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback](http://arxiv.org/pdf/2510.12089v1)

Authors: Xingpei Ma, Shenneng Huang, Jiaran Cai, Yuansheng Guan, Shen Zheng, Hanfeng Zhao, Qiang Zhang, Shunsi Zhang

Recent advances in diffusion models have significantly improved audio-driven
human video generation, surpassing traditional methods in both quality and
controllability. However, existing approaches still face challenges in lip-sync
accuracy, temporal coherence for long video generation, and multi-character
animation. In this work, we propose a diffusion transformer (DiT)-based
framework for generating lifelike talking videos of arbitrary length, and
introduce a training-free method for multi-character audio-driven animation.
First, we employ a LoRA-based training strategy combined with a position shift
inference approach, which enables efficient long video generation while
preserving the capabilities of the foundation model. Moreover, we combine
partial parameter updates with reward feedback to enhance both lip
synchronization and natural body motion. Finally, we propose a training-free
approach, Mask Classifier-Free Guidance (Mask-CFG), for multi-character
animation, which requires no specialized datasets or model modifications and
supports audio-driven animation for three or more characters. Experimental
results demonstrate that our method outperforms existing state-of-the-art
approaches, achieving high-quality, temporally coherent, and multi-character
audio-driven video generation in a simple, efficient, and cost-effective
manner.

### 4. [IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation](http://arxiv.org/pdf/2510.12095v1)

Authors: Wenxu Zhou, Kaixuan Nie, Hang Du, Dong Yin, Wei Huang, Siqiang Guo, Xiaobo Zhang, Pengbo Hu

In this study, we present IL3D, a large-scale dataset meticulously designed
for large language model (LLM)-driven 3D scene generation, addressing the
pressing demand for diverse, high-quality training data in indoor layout
design. Comprising 27,816 indoor layouts across 18 prevalent room types and a
library of 29,215 high-fidelity 3D object assets, IL3D is enriched with
instance-level natural language annotations to support robust multimodal
learning for vision-language tasks. We establish rigorous benchmarks to
evaluate LLM-driven scene generation. Experimental results show that supervised
fine-tuning (SFT) of LLMs on IL3D significantly improves generalization and
surpasses the performance of SFT on other datasets. IL3D offers flexible
multimodal data export capabilities, including point clouds, 3D bounding boxes,
multiview images, depth maps, normal maps, and semantic masks, enabling
seamless adaptation to various visual tasks. As a versatile and robust
resource, IL3D significantly advances research in 3D scene generation and
embodied intelligence, by providing high-fidelity scene data to support
environment perception tasks of embodied agents.

### 5. [An Adaptive Edge-Guided Dual-Network Framework for Fast QR Code Motion Deblurring](http://arxiv.org/pdf/2510.12098v1)

Authors: Jianping Li, Dongyang Guo, Wenjie Li, Wei Zhao

Unlike general image deblurring that prioritizes perceptual quality, QR code
deblurring focuses on ensuring successful decoding. QR codes are characterized
by highly structured patterns with sharp edges, a robust prior for restoration.
Yet existing deep learning methods rarely exploit these priors explicitly. To
address this gap, we propose the Edge-Guided Attention Block (EGAB), which
embeds explicit edge priors into a Transformer architecture. Based on EGAB, we
develop Edge-Guided Restormer (EG-Restormer), an effective network that
significantly boosts the decoding rate of severely blurred QR codes. For mildly
blurred inputs, we design the Lightweight and Efficient Network (LENet) for
fast deblurring. We further integrate these two networks into an Adaptive
Dual-network (ADNet), which dynamically selects the suitable network based on
input blur severity, making it ideal for resource-constrained mobile devices.
Extensive experiments show that our EG-Restormer and ADNet achieve
state-of-the-art performance with a competitive speed. Project page:
https://github.com/leejianping/ADNet

### 6. [G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior](http://arxiv.org/pdf/2510.12099v1)

Authors: Junfeng Ni, Yixin Chen, Zhifei Yang, Yu Liu, Ruijie Lu, Song-Chun Zhu, Siyuan Huang

Despite recent advances in leveraging generative prior from pre-trained
diffusion models for 3D scene reconstruction, existing methods still face two
critical limitations. First, due to the lack of reliable geometric supervision,
they struggle to produce high-quality reconstructions even in observed regions,
let alone in unobserved areas. Second, they lack effective mechanisms to
mitigate multi-view inconsistencies in the generated images, leading to severe
shape-appearance ambiguities and degraded scene geometry. In this paper, we
identify accurate geometry as the fundamental prerequisite for effectively
exploiting generative models to enhance 3D scene reconstruction. We first
propose to leverage the prevalence of planar structures to derive accurate
metric-scale depth maps, providing reliable supervision in both observed and
unobserved regions. Furthermore, we incorporate this geometry guidance
throughout the generative pipeline to improve visibility mask estimation, guide
novel view selection, and enhance multi-view consistency when inpainting with
video diffusion models, resulting in accurate and consistent scene completion.
Extensive experiments on Replica, ScanNet++, and DeepBlending show that our
method consistently outperforms existing baselines in both geometry and
appearance reconstruction, particularly for unobserved regions. Moreover, our
method naturally supports single-view inputs and unposed videos, with strong
generalizability in both indoor and outdoor scenarios with practical real-world
applicability. The project page is available at
https://dali-jack.github.io/g4splat-web/.

### 7. [DRL: Discriminative Representation Learning with Parallel Adapters for Class Incremental Learning](http://arxiv.org/pdf/2510.12107v1)

Authors: Jiawei Zhan, Jun Liu, Jinlong Peng, Xiaochen Chen, Bin-Bin Gao, Yong Liu, Chengjie Wang

With the excellent representation capabilities of Pre-Trained Models (PTMs),
remarkable progress has been made in non-rehearsal Class-Incremental Learning
(CIL) research. However, it remains an extremely challenging task due to three
conundrums: increasingly large model complexity, non-smooth representation
shift during incremental learning and inconsistency between stage-wise
sub-problem optimization and global inference. In this work, we propose the
Discriminative Representation Learning (DRL) framework to specifically address
these challenges. To conduct incremental learning effectively and yet
efficiently, the DRL's network, called Incremental Parallel Adapter (IPA)
network, is built upon a PTM and increasingly augments the model by learning a
lightweight adapter with a small amount of parameter learning overhead in each
incremental stage. The adapter is responsible for adapting the model to new
classes, it can inherit and propagate the representation capability from the
current model through parallel connection between them by a transfer gate. As a
result, this design guarantees a smooth representation shift between different
incremental stages. Furthermore, to alleviate inconsistency and enable
comparable feature representations across incremental stages, we design the
Decoupled Anchor Supervision (DAS). It decouples constraints of positive and
negative samples by respectively comparing them with the virtual anchor. This
decoupling promotes discriminative representation learning and aligns the
feature spaces learned at different stages, thereby narrowing the gap between
stage-wise local optimization over a subset of data and global inference across
all classes. Extensive experiments on six benchmarks reveal that our DRL
consistently outperforms other state-of-the-art methods throughout the entire
CIL period while maintaining high efficiency in both training and inference
phases.

### 8. [Self-Supervised Selective-Guided Diffusion Model for Old-Photo Face Restoration](http://arxiv.org/pdf/2510.12114v1)

Authors: Wenjie Li, Xiangyi Wang, Heng Guo, Guangwei Gao, Zhanyu Ma

Old-photo face restoration poses significant challenges due to compounded
degradations such as breakage, fading, and severe blur. Existing pre-trained
diffusion-guided methods either rely on explicit degradation priors or global
statistical guidance, which struggle with localized artifacts or face color. We
propose Self-Supervised Selective-Guided Diffusion (SSDiff), which leverages
pseudo-reference faces generated by a pre-trained diffusion model under weak
guidance. These pseudo-labels exhibit structurally aligned contours and natural
colors, enabling region-specific restoration via staged supervision: structural
guidance applied throughout the denoising process and color refinement in later
steps, aligned with the coarse-to-fine nature of diffusion. By incorporating
face parsing maps and scratch masks, our method selectively restores breakage
regions while avoiding identity mismatch. We further construct VintageFace, a
300-image benchmark of real old face photos with varying degradation levels.
SSDiff outperforms existing GAN-based and diffusion-based methods in perceptual
quality, fidelity, and regional controllability. Code link:
https://github.com/PRIS-CV/SSDiff.

### 9. [ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](http://arxiv.org/pdf/2510.12119v1)

Authors: Ziyuan Luo, Yangyi Zhao, Ka Chun Cheung, Simon See, Renjie Wan

The widespread adoption of Retrieval-Augmented Image Generation (RAIG) has
raised significant concerns about the unauthorized use of private image
datasets. While these systems have shown remarkable capabilities in enhancing
generation quality through reference images, protecting visual datasets from
unauthorized use in such systems remains a challenging problem. Traditional
digital watermarking approaches face limitations in RAIG systems, as the
complex feature extraction and recombination processes fail to preserve
watermark signals during generation. To address these challenges, we propose
ImageSentinel, a novel framework for protecting visual datasets in RAIG. Our
framework synthesizes sentinel images that maintain visual consistency with the
original dataset. These sentinels enable protection verification through
randomly generated character sequences that serve as retrieval keys. To ensure
seamless integration, we leverage vision-language models to generate the
sentinel images. Experimental results demonstrate that ImageSentinel
effectively detects unauthorized dataset usage while preserving generation
quality for authorized applications. Code is available at
https://github.com/luo-ziyuan/ImageSentinel.

### 10. [Hardware-aware Coding Function Design for Compressive Single-Photon 3D Cameras](http://arxiv.org/pdf/2510.12123v1)

Authors: David Parra, Felipe Gutierrez-Barragan, Trevor Seets, Andreas Velten

Single-photon cameras are becoming increasingly popular in time-of-flight 3D
imaging because they can time-tag individual photons with extreme resolution.
However, their performance is susceptible to hardware limitations, such as
system bandwidth, maximum laser power, sensor data rates, and in-sensor memory
and compute resources. Compressive histograms were recently introduced as a
solution to the challenge of data rates through an online in-sensor compression
of photon timestamp data. Although compressive histograms work within limited
in-sensor memory and computational resources, they underperform when subjected
to real-world illumination hardware constraints. To address this, we present a
constrained optimization approach for designing practical coding functions for
compressive single-photon 3D imaging. Using gradient descent, we jointly
optimize an illumination and coding matrix (i.e., the coding functions) that
adheres to hardware constraints. We show through extensive simulations that our
coding functions consistently outperform traditional coding designs under both
bandwidth and peak power constraints. This advantage is particularly pronounced
in systems constrained by peak power. Finally, we show that our approach adapts
to arbitrary parameterized impulse responses by evaluating it on a real-world
system with a non-ideal impulse response function.

### Computers and Society

### 1. [Structure-aware Propagation Generation with Large Language Models for Fake News Detection](http://arxiv.org/pdf/2510.12125v1)

Authors: Mengyang Chen, Lingwei Wei, Wei Zhou, Songlin Hu

The spread of fake news on social media poses a serious threat to public
trust and societal stability. While propagation-based methods improve fake news
detection by modeling how information spreads, they often suffer from
incomplete propagation data. Recent work leverages large language models (LLMs)
to generate synthetic propagation, but typically overlooks the structural
patterns of real-world discussions. In this paper, we propose a novel
structure-aware synthetic propagation enhanced detection (StruSP) framework to
fully capture structural dynamics from real propagation. It enables LLMs to
generate realistic and structurally consistent propagation for better
detection. StruSP explicitly aligns synthetic propagation with real-world
propagation in both semantic and structural dimensions. Besides, we also design
a new bidirectional evolutionary propagation (BEP) learning strategy to better
align LLMs with structural patterns of propagation in the real world via
structure-aware hybrid sampling and masked propagation modeling objective.
Experiments on three public datasets demonstrate that StruSP significantly
improves fake news detection performance in various practical detection
scenarios. Further analysis indicates that BEP enables the LLM to generate more
realistic and diverse propagation semantically and structurally.

### 2. [From Delegates to Trustees: How Optimizing for Long-Term Interests Shapes Bias and Alignment in LLM](http://arxiv.org/pdf/2510.12689v1)

Authors: Suyash Fulay, Jocelyn Zhu, Michiel Bakker

Large language models (LLMs) have shown promising accuracy in predicting
survey responses and policy preferences, which has increased interest in their
potential to represent human interests in various domains. Most existing
research has focused on behavioral cloning, effectively evaluating how well
models reproduce individuals' expressed preferences. Drawing on theories of
political representation, we highlight an underexplored design trade-off:
whether AI systems should act as delegates, mirroring expressed preferences, or
as trustees, exercising judgment about what best serves an individual's
interests. This trade-off is closely related to issues of LLM sycophancy, where
models can encourage behavior or validate beliefs that may be aligned with a
user's short-term preferences, but is detrimental to their long-term interests.
Through a series of experiments simulating votes on various policy issues in
the U.S. context, we apply a temporal utility framework that weighs short and
long-term interests (simulating a trustee role) and compare voting outcomes to
behavior-cloning models (simulating a delegate). We find that trustee-style
predictions weighted toward long-term interests produce policy decisions that
align more closely with expert consensus on well-understood issues, but also
show greater bias toward models' default stances on topics lacking clear
agreement. These findings reveal a fundamental trade-off in designing AI
systems to represent human interests. Delegate models better preserve user
autonomy but may diverge from well-supported policy positions, while trustee
models can promote welfare on well-understood issues yet risk paternalism and
bias on subjective topics.

### 3. [Who is a Better Matchmaker? Human vs. Algorithmic Judge Assignment in a High-Stakes Startup Competition](http://arxiv.org/pdf/2510.12692v1)

Authors: Sarina Xi, Orelia Pi, Miaomiao Zhang, Becca Xiong, Jacqueline Ng Lane, Nihar B. Shah

There is growing interest in applying artificial intelligence (AI) to
automate and support complex decision-making tasks. However, it remains unclear
how algorithms compare to human judgment in contexts requiring semantic
understanding and domain expertise. We examine this in the context of the judge
assignment problem, matching submissions to suitably qualified judges.
Specifically, we tackled this problem at the Harvard President's Innovation
Challenge, the university's premier venture competition awarding over \$500,000
to student and alumni startups. This represents a real-world environment where
high-quality judge assignment is essential. We developed an AI-based
judge-assignment algorithm, Hybrid Lexical-Semantic Similarity Ensemble (HLSE),
and deployed it at the competition. We then evaluated its performance against
human expert assignments using blinded match-quality scores from judges on
$309$ judge-venture pairs. Using a Mann-Whitney U statistic based test, we
found no statistically significant difference in assignment quality between the
two approaches ($AUC=0.48, p=0.40$); on average, algorithmic matches are rated
$3.90$ and manual matches $3.94$ on a 5-point scale, where 5 indicates an
excellent match. Furthermore, manual assignments that previously required a
full week could be automated in several hours by the algorithm during
deployment. These results demonstrate that HLSE achieves human-expert-level
matching quality while offering greater scalability and efficiency,
underscoring the potential of AI-driven solutions to support and enhance human
decision-making for judge assignment in high-stakes settings.

### Databases

### 1. [Analysis and Evaluation of Using Microsecond-Latency Memory for In-Memory Indices and Caches in SSD-Based Key-Value Stores](http://arxiv.org/pdf/2510.12280v1)

Authors: Yosuke Bando, Akinobu Mita, Kazuhiro Hiwada, Shintaro Sano, Tomoya Suzuki, Yu Nakanishi, Kazutaka Tomida, Hirotsugu Kajihara, Akiyuki Kaneko, Daisuke Taki, Yukimasa Miyamoto, Tomokazu Yoshida, Tatsuo Shiozawa

When key-value (KV) stores use SSDs for storing a large number of items,
oftentimes they also require large in-memory data structures including indices
and caches to be traversed to reduce IOs. This paper considers offloading most
of such data structures from the costly host DRAM to secondary memory whose
latency is in the microsecond range, an order of magnitude longer than those of
currently available DIMM-mounted or CXL memory devices. While emerging
microsecond-latency memory is likely to cost much less than DRAM, it can
significantly slow down SSD-based KV stores if naively employed. This paper
analyzes and evaluates the impact of microsecond-level memory latency on the KV
operation throughput. Our analysis finds that a well-known latency-hiding
technique of software prefetching for long-latency memory from user-level
threads is effective. The novelty of our analysis lies in modeling how the
interplay between prefetching and IO affects performance, from which we derive
an equation that well explains the throughput degradation due to long memory
latency. The model tells us that the presence of IO significantly enhances the
tolerance to memory latency, leading to a finding that SSD-based KV stores can
be made latency-tolerant without devising new techniques for
microsecond-latency memory. To confirm this, we design a microbenchmark as well
as modify existing SSD-based KV stores so that they issue prefetches from
user-level threads, and run them while placing most of in-memory data
structures on FPGA-based memory with adjustable microsecond latency. The
results demonstrate that their KV operation throughputs can be well explained
by our model, and the modified KV stores achieve near-DRAM throughputs for up
to a memory latency of 5 microseconds. This suggests the possibility that
SSD-based KV stores can use microsecond-latency memory as a cost-effective
alternative to the host DRAM.

### 2. [Aixel: A Unified, Adaptive and Extensible System for AI-powered Data Analysis](http://arxiv.org/pdf/2510.12642v1)

Authors: Meihui Zhang, Liming Wang, Chi Zhang, Zhaojing Luo

A growing trend in modern data analysis is the integration of data management
with learning, guided by accuracy, latency, and cost requirements. In practice,
applications draw data of different formats from many sources. In the
meanwhile, the objectives and budgets change over time. Existing systems handle
these applications across databases, analysis libraries, and tuning services.
Such fragmentation leads to complex user interaction, limited adaptability,
suboptimal performance, and poor extensibility across components. To address
these challenges, we present Aixel, a unified, adaptive, and extensible system
for AI-powered data analysis. The system organizes work across four layers:
application, task, model, and data. The task layer provides a declarative
interface to capture user intent, which is parsed into an executable operator
plan. An optimizer compiles and schedules this plan to meet specified goals in
accuracy, latency, and cost. The task layer coordinates the execution of data
and model operators, with built-in support for reuse and caching to improve
efficiency. The model layer offers versioned storage for index, metadata,
tensors, and model artifacts. It supports adaptive construction, task-aligned
drift detection, and safe updates that reuse shared components. The data layer
provides unified data management capabilities, including indexing,
constraint-aware discovery, task-aligned selection, and comprehensive feature
management. With the above designed layers, Aixel delivers a user friendly,
adaptive, efficient, and extensible system.

### Distributed, Parallel, and Cluster Computing

### 1. [Comparing Cross-Platform Performance via Node-to-Node Scaling Studies](http://arxiv.org/pdf/2510.12166v1)

Authors: Kenneth Weiss, Thomas M. Stitt, Daryl Hawkins, Olga Pearce, Stephanie Brink, Robert N. Rieben

Due to the increasing diversity of high-performance computing architectures,
researchers and practitioners are increasingly interested in comparing a code's
performance and scalability across different platforms. However, there is a
lack of available guidance on how to actually set up and analyze such
cross-platform studies. In this paper, we contend that the natural base unit of
computing for such studies is a single compute node on each platform and offer
guidance in setting up, running, and analyzing node-to-node scaling studies. We
propose templates for presenting scaling results of these studies and provide
several case studies highlighting the benefits of this approach.

### 2. [GPU-Accelerated Algorithms for Process Mapping](http://arxiv.org/pdf/2510.12196v1)

Authors: Petr Samoldekin, Christian Schulz, Henning Woydt

Process mapping asks to assign vertices of a task graph to processing
elements of a supercomputer such that the computational workload is balanced
while the communication cost is minimized. Motivated by the recent success of
GPU-based graph partitioners, we propose two GPU-accelerated algorithms for
this optimization problem. The first algorithm employs hierarchical
multisection, which partitions the task graph alongside the hierarchy of the
supercomputer. The method utilizes GPU-based graph partitioners to accelerate
the mapping process. The second algorithm integrates process mapping directly
into the modern multilevel graph partitioning pipeline. Vital phases like
coarsening and refinement are accelerated by exploiting the parallelism of
GPUs. In our experiments, both methods achieve speedups exceeding 300 when
compared to state-of-the-art CPU-based algorithms. The first algorithm has, on
average, about 10 percent greater communication costs and thus remains
competitive to CPU algorithms. The second approach is much faster, with a
geometric mean speedup of 77.6 and peak speedup of 598 at the cost of lower
solution quality. To our knowledge, these are the first GPU-based algorithms
for process mapping.

### 3. [Metronome: Efficient Scheduling for Periodic Traffic Jobs with Network and Priority Awareness](http://arxiv.org/pdf/2510.12274v1)

Authors: Hao Jiang, Meng Qin, Ruijie Kuai, Dandan Liang

With the rapid growth in computing power demand, cloud native networks have
emerged as a promising solution to address the challenges of efficient resource
coordination, particularly in coping with the dynamic fluctuations of network
bandwidth in clusters. We propose Metronome, a network-aware and priority-aware
scheduling mechanism for cloud native networks. This mechanism is designed to
support jobs that exhibit periodic traffic patterns and dynamic bandwidth
demands, particularly in the context of distributed training. Specifically,
Metronome employs a time-division multiplexing approach that leverages job
traffic characteristics to construct an elastic network resource allocation
model, enabling efficient bandwidth sharing across multiple jobs. In addition,
it incorporates a multi-objective optimization strategy, jointly considering
latency and job priorities to achieve globally optimal as well as dynamic
resource allocation. Finally, Metronome adapts to the dynamic environment by
monitoring the cluster and performing reconfiguration operations. Extensive
experiments with 13 common machine learning models demonstrate that Metronome
can enhance cluster resource utilization while guaranteeing service
performance. Compared with the existing Kubernetes scheduling mechanisms across
multiple scenarios, Metronome reduces job completion time by up to 19.50% while
improving average bandwidth utilization by up to 23.20%.

### 4. [A Non-Intrusive Framework for Deferred Integration of Cloud Patterns in Energy-Efficient Data-Sharing Pipelines](http://arxiv.org/pdf/2510.12354v1)

Authors: Sepideh Masoudi, Mark Edward Michael Daly, Jannis Kiesel, Stefan Tai

As data mesh architectures gain traction in federated environments,
organizations are increasingly building consumer-specific data-sharing
pipelines using modular, cloud-native transformation services. Prior work has
shown that structuring these pipelines with reusable transformation stages
enhances both scalability and energy efficiency. However, integrating
traditional cloud design patterns into such pipelines poses a challenge:
predefining and embedding patterns can compromise modularity, reduce
reusability, and conflict with the pipelines dynamic, consumer-driven nature.
To address this, we introduce a Kubernetes-based tool that enables the deferred
and non-intrusive application of selected cloud design patterns without
requiring changes to service source code. The tool supports automated pattern
injection and collects energy consumption metrics, allowing developers to make
energy-aware decisions while preserving the flexible, composable structure of
reusable data-sharing pipelines.

### 5. [Low Latency, High Bandwidth Streaming of Experimental Data with EJFAT](http://arxiv.org/pdf/2510.12597v1)

Authors: Ilya Baldin, Michael Goodrich, Vardan Gyurjyan, Graham Heyes, Derek Howard, Yatish Kumar, David Lawrence, Brad Sawatzky, Stacey Sheldon, Carl Timmer

Thomas Jefferson National Accelerator Facility (JLab) has partnered with
Energy Sciences Network (ESnet) to define and implement an edge to compute
cluster computational load balancing acceleration architecture. The ESnet-JLab
FPGA Accelerated Transport (EJFAT) architecture focuses on FPGA acceleration to
address compression, fragmentation, UDP packet destination redirection (Network
Address Translation (NAT)) and decompression and reassembly.
  EJFAT seamlessly integrates edge and cluster computing to support direct
processing of streamed experimental data. This will directly benefit the JLab
science program as well as data centers of the future that require high
throughput and low latency for both time-critical data acquisition systems and
data center workflows.
  The EJFAT project will be presented along with how it is synergistic with
other DOE activities such as an Integrated Research Infrastructure (IRI), and
recent results using data sources at JLab, an EJFAT LB at ESnet, and
computational cluster resources at Lawrence Berkeley National Laboratory
(LBNL).

### 6. [TALP-Pages: An easy-to-integrate continuous performance monitoring framework](http://arxiv.org/pdf/2510.12436v1)

Authors: Valentin Seitz, Jordy Trilaksono, Marta Garcia-Gasulla

Ensuring good performance is a key aspect in the development of codes that
target HPC machines. As these codes are under active development, the necessity
to detect performance degradation early in the development process becomes
apparent. In addition, having meaningful insight into application scaling
behavior tightly coupled to the development workflow is helpful. In this paper,
we introduce TALP-Pages, an easy-to-integrate framework that enables developers
to get fast and in-repository feedback about their code performance using
established fundamental performance and scaling factors. The framework relies
on TALP, which enables the on-the-fly collection of these metrics. Based on a
folder structure suited for CI which contains the files generated by TALP,
TALP-Pages generates an HTML report with visualizations of the performance
factor regression as well as scaling-efficiency tables. We compare TALP-Pages
to tracing-based tools in terms of overhead and post-processing requirements
and find that TALP-Pages can produce the scaling-efficiency tables faster and
under tighter resource constraints. To showcase the ease of use and
effectiveness of this approach, we extend the current CI setup of GENE-X with
only minimal changes required and showcase the ability to detect and explain a
performance improvement.

### 7. [Proof of Cloud: Data Center Execution Assurance for Confidential VMs](http://arxiv.org/pdf/2510.12469v1)

Authors: Filip Rezabek, Moe Mahhouk, Andrew Miller, Stefan Genchev, Quintus Kilbourn, Georg Carle, Jonathan Passerat-Palmbach

Confidential Virtual Machines (CVMs) protect data in use by running workloads
inside hardware-isolated environments. In doing so, they also inherit the
limitations of the underlying hardware. Trusted Execution Environments (TEEs),
which enforce this isolation, explicitly exclude adversaries with physical
access from their threat model. Commercial TEEs, e.g., Intel TDX, thus assume
infrastructure providers do not physically exploit hardware and serve as
safeguards instead. This creates a tension: tenants must trust provider
integrity at the hardware layer, yet existing remote attestation offers no way
to verify that CVMs actually run on physically trusted platforms, leaving
today's CVM deployments unable to demonstrate that their guarantees align with
the TEE vendor's threat model.
  We bridge this confidence gap with Data Center Execution Assurance (DCEA), a
design generating "Proofs of Cloud". DCEA binds a CVM to its underlying
platform using vTPM-anchored measurements, ensuring CVM launch evidence and TPM
quotes refer to the same physical chassis.
  This takes advantage of the fact that data centers are often identifiable via
TPMs. Our approach applies to CVMs accessing vTPMs and running on top of
software stacks fully controlled by the cloud provider, as well as
single-tenant bare-metal deployments with discrete TPMs. We trust providers for
integrity (certificate issuance), but not for the confidentiality of
CVM-visible state. DCEA enables remote verification of a CVM's platform origin
and integrity, mitigating attacks like replay and attestation proxying. We
include a candidate implementation on Google Cloud and Intel TDX that leverages
Intel TXT for trusted launch. Our design refines CVMs' threat model and
provides a practical path for deploying high-assurance, confidential workloads
in minimally trusted environments.

### 8. [A GPU-resident Memory-Aware Algorithm for Accelerating Bidiagonalization of Banded Matrices](http://arxiv.org/pdf/2510.12705v1)

Authors: Evelyne Ringoot, Rabab Alomairy, Alan Edelman

The reduction of a banded matrix to a bidiagonal form is a crucial step in
the Singular Value Decomposition (SVD), a cornerstone of scientific computing
and AI. Despite being a highly parallel algorithm, it was previously believed
to be unsuitable for GPU computation because it is memory bandwidth-bound.
Recent developments in GPU hardware, including larger L1 memory per Streaming
Multiprocessor/Compute Unit, have changed that. We present the first GPU
algorithm for reducing a banded matrix to bidiagonal form as part of the
NextLA$.$jl open-source software package. Our algorithm is based on previous
CPU-based multicore parallel cache-efficient bulge chasing algorithms and
adapted to optimize for GPU throughput. We leverage Julia Language's Array
abstractions and KernelAbstractions to implement a single hardware- and data
precision-agnostic function on NVIDIA, AMD, Intel, and Apple Metal GPUs for
half, single, and double precision, and examine performance optimization across
hardware architectures and data precision. We also develop a hardware-aware
performance model and identify key hyperparameters, such as inner tilewidth and
block concurrency, that govern optimal GPU execution for bandwidth-bound
workloads. We demonstrate highly parallel bandwidth-bound algorithm on the GPU
can outperform CPU-based implementations: the GPU algorithm outperforms
multithreaded CPU High-Performance libraries PLASMA and SLATE as of matrix size
1024 x 1024 and by a factor over 100 for matrices of 32k x 32k. In addition,
the performance of the algorithm increases linearly with matrix bandwidth size,
making faster reduction of larger matrix bandwidths now also possible. With
this work, we break memory bandwidth barriers, as well as matrix bandwidth
barriers, resulting in orders-of-magnitude faster algorithms for the reduction
of banded matrices to bidiagonal form on the GPU.

### 9. [Personalized Federated Fine-Tuning of Vision Foundation Models for Healthcare](http://arxiv.org/pdf/2510.12741v1)

Authors: Adam Tupper, Christian Gagné

Foundation models open up new possibilities for the use of AI in healthcare.
However, even when pre-trained on health data, they still need to be fine-tuned
for specific downstream tasks. Furthermore, although foundation models reduce
the amount of training data required to achieve good performance, obtaining
sufficient data is still a challenge. This is due, in part, to restrictions on
sharing and aggregating data from different sources to protect patients'
privacy. One possible solution to this is to fine-tune foundation models via
federated learning across multiple participating clients (i.e., hospitals,
clinics, etc.). In this work, we propose a new personalized federated
fine-tuning method that learns orthogonal LoRA adapters to disentangle general
and client-specific knowledge, enabling each client to fully exploit both their
own data and the data of others. Our preliminary results on real-world
federated medical imaging tasks demonstrate that our approach is competitive
against current federated fine-tuning methods.

### 10. [nuGPR: GPU-Accelerated Gaussian Process Regression with Iterative Algorithms and Low-Rank Approximations](http://arxiv.org/pdf/2510.12128v1)

Authors: Ziqi Zhao, Vivek Sarin

Gaussian Process Regression (GPR) is an important type of supervised machine
learning model with inherent uncertainty measure in its predictions. We propose
a new framework, nuGPR, to address the well-known challenge of high computation
cost associated with GPR training. Our framework includes several ideas from
numerical linear algebra to reduce the amount of computation in key steps of
GPR, and we combine them to establish an end-to-end training algorithm.
Specifically, we leverage the preconditioned conjugate gradient method to
accelerate the convergence of the linear solves required in GPR. We exploit
clustering in the input data to identify block-diagonal structure of the
covariance matrix and subsequently construct low-rank approximations of the
off-diagonal blocks. These enhancements significantly reduce the time and space
complexity of our computations. In addition, unlike other frameworks that rely
on exact differentiation, we employ numerical gradients to optimize the
hyperparameters of our GPR model, further reducing the training cost by
eliminating the need for backpropagation. Lastly, we leverage the CUDA Toolkit
to efficiently parallelize the training procedure on NVIDIA GPUs. As a result,
nuGPR reduces total training time by up to 2x and peak memory consumption by up
to 12x on various synthetic and real-world datasets when compared to the best
existing GPU-based GPR implementation.

### Discrete Mathematics

### 1. [Thin Trees via $k$-Respecting Cut Identities](http://arxiv.org/pdf/2510.12050v1)

Authors: Mohit Daga

Thin spanning trees lie at the intersection of graph theory, approximation
algorithms, and combinatorial optimization. They are central to the
long-standing \emph{thin tree conjecture}, which asks whether every
$k$-edge-connected graph contains an $O(1/k)$-thin tree, and they underpin
algorithmic breakthroughs such as the $O(\log n/\log\log n)$-approximation for
ATSP. Yet even the basic algorithmic task of \emph{verifying} that a given tree
is thin has remained elusive: checking thinness requires reasoning about
exponentially many cuts, and no efficient certificates have been known.
  We introduce a new machinery of \emph{$k$-respecting cut identities}, which
express the weight of every cut that crosses a spanning tree in at most $k$
edges as a simple function of pairwise ($2$-respecting) cuts. This yields a
tree-local oracle that, after $O(n^2)$ preprocessing, evaluates such cuts in
$O_k(1)$ time. Building on this oracle, we give the first procedure to compute
the exact $k$-thinness certificate $\Theta_k(T)$ of any spanning tree for fixed
$k$ in time $\tilde O(n^2+n^k)$, outputting both the certificate value and a
witnessing cut.
  Beyond general graphs, our framework yields sharper guarantees in structured
settings. In planar graphs, duality with cycles and dual girth imply that every
spanning tree admits a verifiable certificate $\Theta_k(T)\le k/\lambda$ (hence
$O(1/\lambda)$ for constant $k$). In graphs embedded on a surface of genus
$\gamma$, refined counting gives certified (per-cut) bounds $O((\log
n+\gamma)/\lambda)$ via the same ensemble coverage.

### 2. [Unimodular toric ideals of graphs](http://arxiv.org/pdf/2510.12544v1)

Authors: Christos Tatakis

We give a necessary and sufficient graph-theoretic characterization of toric
ideals of graphs that are unimodular. As a direct consequence, we provide the
structure of unimodular graphs by proving that the incidence matrix of a graph
$G$ is unimodular if and only if any two odd cycles of $G$ intersect.

### Data Structures and Algorithms

### 1. [Engineering Dominating Patterns: A Fine-grained Case Study](http://arxiv.org/pdf/2510.12232v1)

Authors: Jonathan Dransfeld, Marvin Künnemann, Mirza Redzic, Marcus Wunderlich

The \emph{Dominating $H$-Pattern} problem generalizes the classical
$k$-Dominating Set problem: for a fixed \emph{pattern} $H$ and a given graph
$G$, the goal is to find an induced subgraph $S$ of $G$ such that (1) $S$ is
isomorphic to $H$, and (2) $S$ forms a dominating set in $G$. Fine-grained
complexity results show that on worst-case inputs, any significant improvement
over the naive brute-force algorithm is unlikely, as this would refute the
Strong Exponential Time Hypothesis. Nevertheless, a recent work by Dransfeld et
al. (ESA 2025) reveals some significant improvement potential particularly in
\emph{sparse} graphs.
  We ask: Can algorithms with conditionally almost-optimal worst-case
performance solve the Dominating $H$-Pattern, for selected patterns $H$,
efficiently on practical inputs? We develop and experimentally evaluate several
approaches on a large benchmark of diverse datasets, including baseline
approaches using the Glasgow Subgraph Solver (GSS), the SAT solver Kissat, and
the ILP solver Gurobi.
  Notably, while a straightforward implementation of the algorithms -- with
conditionally close-to-optimal worst-case guarantee -- performs comparably to
existing solvers, we propose a tailored Branch-\&-Bound approach --
supplemented with careful pruning techniques -- that achieves improvements of
up to two orders of magnitude on our test instances.

### 2. [Lossless Derandomization for Undirected Single-Source Shortest Paths and Approximate Distance Oracles](http://arxiv.org/pdf/2510.12598v1)

Authors: Shuyi Yan

A common step in algorithms related to shortest paths in undirected graphs is
that, we select a subset of vertices as centers, then grow a ball around each
vertex until a center is reached. We want the balls to be as small as possible.
A randomized algorithm can uniformly sample $r$ centers to achieve the optimal
(expected) ball size of $\Theta(n/r)$. A folklore derandomization is to use the
$O(\log n)$ approximation for the set cover problem in the hitting set version
where we want to hit all the balls with the centers.
  However, the extra $O(\log n)$ factor is sometimes too expensive. For
example, the recent $O(m\sqrt{\log n\log\log n})$ undirected single-source
shortest path algorithm [DMSY23] beats Dijkstra's algorithm in sparse graphs,
but the folklore derandomization would make it dominated by Dijkstra's.
  In this paper, we exploit the fact that the sizes of these balls can be
adaptively chosen by the algorithm instead of fixed by the input. We propose a
simple deterministic algorithm achieving the optimal ball size of $\Theta(n/r)$
on average. Furthermore, given any polynomially large cost function of the ball
size, we can still achieve the optimal cost on average. It allows us to
derandomize [DMSY23], resulting in a deterministic $O(m\sqrt{\log n\log\log
n})$ algorithm for undirected single-source shortest path.
  In addition, we show that the same technique can also be used to derandomize
the seminal Thorup-Zwick approximate distance oracle [TZ05], also without any
loss in the time/space complexity.

### 3. [Vizing's Theorem in Deterministic Almost-Linear Time](http://arxiv.org/pdf/2510.12619v1)

Authors: Sepehr Assadi, Soheil Behnezhad, Sayan Bhattacharya, Martín Costa, Shay Solomon, Tianyi Zhang

Vizing's theorem states that any $n$-vertex $m$-edge graph of maximum degree
$\Delta$ can be edge colored using at most $\Delta + 1$ different colors.
Vizing's original proof is easily translated into a deterministic $O(mn)$ time
algorithm. This deterministic time bound was subsequently improved to $\tilde
O(m \sqrt n)$ time, independently by [Arjomandi, 1982] and by [Gabow et al.,
1985].
  A series of recent papers improved the time bound of $\tilde O(m\sqrt{n})$
using randomization, culminating in the randomized near-linear time
$(\Delta+1)$-coloring algorithm by [Assadi, Behnezhad, Bhattacharya, Costa,
Solomon, and Zhang, 2025]. At the heart of all of these recent improvements,
there is some form of a sublinear time algorithm. Unfortunately, sublinear time
algorithms as a whole almost always require randomization. This raises a
natural question: can the deterministic time complexity of the problem be
reduced below the $\tilde O(m\sqrt{n})$ barrier?
  In this paper, we answer this question in the affirmative. We present a
deterministic almost-linear time $(\Delta+1)$-coloring algorithm, namely, an
algorithm running in $m \cdot 2^{O(\sqrt{\log \Delta})} \cdot \log n =
m^{1+o(1)}$ time. Our main technical contribution is to entirely forego
sublinear time algorithms. We do so by presenting a new deterministic
color-type sparsification approach that runs in almost-linear (instead of
sublinear) time, but can be used to color a much larger set of edges.

### 4. [Planted clique recovery in random geometric graphs](http://arxiv.org/pdf/2510.12365v1)

Authors: Konstantin Avrachenkov, Andrei Bobu, Nelly Litvak, Riccardo Michielan

We investigate the problem of identifying planted cliques in random geometric
graphs, focusing on two distinct algorithmic approaches: the first based on
vertex degrees (VD) and the other on common neighbors (CN). We analyze the
performance of these methods under varying regimes of key parameters, namely
the average degree of the graph and the size of the planted clique. We
demonstrate that exact recovery is achieved with high probability as the graph
size increases, in a specific set of parameters. Notably, our results reveal
that the CN-algorithm significantly outperforms the VD-algorithm. In
particular, in the connectivity regime, tiny planted cliques (even edges) are
correctly identified by the CN-algorithm, yielding a significant impact on
anomaly detection. Finally, our results are confirmed by a series of numerical
experiments, showing that the devised algorithms are effective in practice.

### 5. [Exact Matching and Top-k Perfect Matching Parameterized by Neighborhood Diversity or Bandwidth](http://arxiv.org/pdf/2510.12552v1)

Authors: Nicolas El Maalouly, Kostas Lakis

The Exact Matching (EM) problem asks whether there exists a perfect matching
which uses a prescribed number of red edges in a red/blue edge-colored graph.
While there exists a randomized polynomial-time algorithm for the problem, only
some special cases admit a deterministic one so far, making it a natural
candidate for testing the P=RP hypothesis. A polynomial-time equivalent
problem, Top-k Perfect Matching (TkPM), asks for a perfect matching maximizing
the weight of the $k$ heaviest edges.
  We study the above problems, mainly the latter, in the scenario where the
input is a blown-up graph, meaning a graph which had its vertices replaced by
cliques or independent sets. We describe an FPT algorithm for TkPM
parameterized by $k$ and the neighborhood diversity of the input graph, which
is essentially the size of the graph before the blow-up; this graph is also
called the prototype. We extend this algorithm into an approximation scheme
with a much softer dependency on the aforementioned parameters, time-complexity
wise. Moreover, for prototypes with bounded bandwidth but unbounded size, we
develop a recursive algorithm that runs in subexponential time. Utilizing
another algorithm for EM on bounded neighborhood diversity graphs, we adapt
this recursive subexponential algorithm to EM.
  Our approach is similar to the use of dynamic programming on e.g. bounded
treewidth instances for various problems. The main point is that the existence
of many disjoint separators is utilized to avoid including in the separator any
of a set of ``bad'' vertices during the split phase.

### 6. [Single-Deviation Stability in Additively Separable Hedonic Games with Constrained Coalition Sizes](http://arxiv.org/pdf/2510.12641v1)

Authors: Martin Bullinger, Adam Dunajski, Edith Elkind, Matan Gilboa

We study stability in additively separable hedonic games when coalition sizes
have to respect fixed size bounds. We consider four classic notions of
stability based on single-agent deviations, namely, Nash stability, individual
stability, contractual Nash stability, and contractual individual stability.
For each stability notion, we consider two variants: in one, the coalition left
behind by a deviator must still be of a valid size, and in the other there is
no such constraint. We provide a full picture of the existence of stable
outcomes with respect to given size parameters. Additionally, when there are
only upper bounds, we fully characterize the computational complexity of the
associated existence problem. In particular, we obtain polynomial-time
algorithms for contractual individual stability and contractual Nash stability,
where the latter requires an upper bound of 2. We obtain further results for
Nash stability and contractual individual stability, when the lower bound is at
least 2.

### 7. [Structure-Aware Spectral Sparsification via Uniform Edge Sampling](http://arxiv.org/pdf/2510.12669v1)

Authors: Kaiwen He, Petros Drineas, Rajiv Khanna

Spectral clustering is a fundamental method for graph partitioning, but its
reliance on eigenvector computation limits scalability to massive graphs.
Classical sparsification methods preserve spectral properties by sampling edges
proportionally to their effective resistances, but require expensive
preprocessing to estimate these resistances. We study whether uniform edge
sampling-a simple, structure-agnostic strategy-can suffice for spectral
clustering. Our main result shows that for graphs admitting a well-separated
$k$-clustering, characterized by a large structure ratio $\Upsilon(k) =
\lambda_{k+1} / \rho_G(k)$, uniform sampling preserves the spectral subspace
used for clustering. Specifically, we prove that uniformly sampling $O(\gamma^2
n \log n / \epsilon^2)$ edges, where $\gamma$ is the Laplacian condition
number, yields a sparsifier whose top $(n-k)$-dimensional eigenspace is
approximately orthogonal to the cluster indicators. This ensures that the
spectral embedding remains faithful, and clustering quality is preserved. Our
analysis introduces new resistance bounds for intra-cluster edges, a
rank-$(n-k)$ effective resistance formulation, and a matrix Chernoff bound
adapted to the dominant eigenspace. These tools allow us to bypass importance
sampling entirely. Conceptually, our result connects recent coreset-based
clustering theory to spectral sparsification, showing that under strong
clusterability, even uniform sampling is structure-aware. This provides the
first provable guarantee that uniform edge sampling suffices for
structure-preserving spectral clustering.

### 8. [Thin Trees via $k$-Respecting Cut Identities](http://arxiv.org/pdf/2510.12050v1)

Authors: Mohit Daga

Thin spanning trees lie at the intersection of graph theory, approximation
algorithms, and combinatorial optimization. They are central to the
long-standing \emph{thin tree conjecture}, which asks whether every
$k$-edge-connected graph contains an $O(1/k)$-thin tree, and they underpin
algorithmic breakthroughs such as the $O(\log n/\log\log n)$-approximation for
ATSP. Yet even the basic algorithmic task of \emph{verifying} that a given tree
is thin has remained elusive: checking thinness requires reasoning about
exponentially many cuts, and no efficient certificates have been known.
  We introduce a new machinery of \emph{$k$-respecting cut identities}, which
express the weight of every cut that crosses a spanning tree in at most $k$
edges as a simple function of pairwise ($2$-respecting) cuts. This yields a
tree-local oracle that, after $O(n^2)$ preprocessing, evaluates such cuts in
$O_k(1)$ time. Building on this oracle, we give the first procedure to compute
the exact $k$-thinness certificate $\Theta_k(T)$ of any spanning tree for fixed
$k$ in time $\tilde O(n^2+n^k)$, outputting both the certificate value and a
witnessing cut.
  Beyond general graphs, our framework yields sharper guarantees in structured
settings. In planar graphs, duality with cycles and dual girth imply that every
spanning tree admits a verifiable certificate $\Theta_k(T)\le k/\lambda$ (hence
$O(1/\lambda)$ for constant $k$). In graphs embedded on a surface of genus
$\gamma$, refined counting gives certified (per-cut) bounds $O((\log
n+\gamma)/\lambda)$ via the same ensemble coverage.

### 9. [Performance of Gaussian Boson Sampling on Planted Bipartite Clique Detection](http://arxiv.org/pdf/2510.12774v1)

Authors: Yu-Zhen Janice Chen, Laurent Massoulié, Don Towsley

We investigate whether Gaussian Boson Sampling (GBS) can provide a
computational advantage for solving the planted biclique problem, which is a
graph problem widely believed to be classically hard when the planted structure
is small. Although GBS has been heuristically and experimentally observed to
favor sampling dense subgraphs, its theoretical performance on this classically
hard problem remains largely unexplored. We focus on a natural statistic
derived from GBS output: the frequency with which a node appears in GBS
samples, referred to as the node weight. We rigorously analyze whether this
signal is strong enough to distinguish planted biclique nodes from background
nodes. Our analysis characterizes the distribution of node weights under GBS
and quantifies the bias introduced by the planted structure. The results reveal
a sharp limitation: when the planted biclique size falls within the conjectured
hard regime, the natural fluctuations in node weights dominate the bias signal,
making detection unreliable using simple ranking strategies. These findings
provide the first rigorous evidence that planted biclique detection may remain
computationally hard even under GBS-based quantum computing, and they motivate
further investigation into more advanced GBS-based algorithms or other quantum
approaches for this problem.

### Emerging Technologies

### 1. [Translating Milli/Microrobots with A Value-Centered Readiness Framework](http://arxiv.org/pdf/2510.12090v1)

Authors: Hakan Ceylan, Edoardo Sinibaldi, Sanjay Misra, Pankaj J. Pasricha, Dietmar W. Hutmacher

Untethered mobile milli/microrobots hold transformative potential for
interventional medicine by enabling more precise and entirely non-invasive
diagnosis and therapy. Realizing this promise requires bridging the gap between
groundbreaking laboratory demonstrations and successful clinical integration.
Despite remarkable technical progress over the past two decades, most
millirobots and microrobots remain confined to laboratory proof-of-concept
demonstrations, with limited real-world feasibility. In this Review, we
identify key factors that slow translation from bench to bedside, focusing on
the disconnect between technical innovation and real-world application. We
argue that the long-term impact and sustainability of the field depend on
aligning development with unmet medical needs, ensuring applied feasibility,
and integrating seamlessly into existing clinical workflows, which are
essential pillars for delivering meaningful patient outcomes. To support this
shift, we introduce a strategic milli/microrobot Technology Readiness Level
framework (mTRL), which maps system development from initial conceptualization
to clinical adoption through clearly defined milestones and their associated
stepwise activities. The mTRL model provides a structured gauge of
technological maturity, a common language for cross-disciplinary collaboration
and actionable guidance to accelerate translational development toward new,
safer and more efficient interventions.

### 2. [Quantum Annealing for Staff Scheduling in Educational Environments](http://arxiv.org/pdf/2510.12278v1)

Authors: Alessia Ciacco, Francesca Guerriero, Eneko Osaba

We address a novel staff allocation problem that arises in the organization
of collaborators among multiple school sites and educational levels. The
problem emerges from a real case study in a public school in Calabria, Italy,
where staff members must be distributed across kindergartens, primary, and
secondary schools under constraints of availability, competencies, and
fairness. To tackle this problem, we develop an optimization model and
investigate a solution approach based on quantum annealing. Our computational
experiments on real-world data show that quantum annealing is capable of
producing balanced assignments in short runtimes. These results provide
evidence of the practical applicability of quantum optimization methods in
educational scheduling and, more broadly, in complex resource allocation tasks.

### 3. [High-Parallel FPGA-Based Discrete Simulated Bifurcation for Large-Scale Optimization](http://arxiv.org/pdf/2510.12407v1)

Authors: Fabrizio Orlando, Deborah Volpe, Giacomo Orlandi, Mariagrazia Graziano, Fabrizio Riente, Marco Vacca

Combinatorial Optimization (CO) problems exhibit exponential complexity,
making their resolution challenging. Simulated Adiabatic Bifurcation (aSB) is a
quantum-inspired algorithm to obtain approximate solutions to largescale CO
problems written in the Ising form. It explores the solution space by emulating
the adiabatic evolution of a network of Kerr-nonlinear parametric oscillators
(KPOs), where each oscillator represents a variable in the problem. The optimal
solution corresponds to the ground state of this system. A key advantage of
this approach is the possibility of updating multiple variables simultaneously,
making it particularly suited for hardware implementation. To enhance solution
quality and convergence speed, variations of the algorithm have been proposed
in the literature, including ballistic (bSB), discrete (dSB), and thermal
(HbSB) versions. In this work, we have comprehensively analyzed dSB, bSB, and
HbSB using dedicated software models, evaluating the feasibility of using a
fixed-point representation for hardware implementation. We then present an
opensource hardware architecture implementing the dSB algorithm for
Field-Programmable Gate Arrays (FPGAs). The design allows users to adjust the
degree of algorithmic parallelization based on their specific requirements. A
proof-of-concept implementation that solves 256-variable problems was achieved
on an AMD Kria KV260 SoM, a low-tier FPGA, validated using well-known max-cut
and knapsack problems.

### 4. [ECMSim: A high-performance web simulation of cardiac ECM remodeling through integrated ODE-based signaling and diffusion](http://arxiv.org/pdf/2510.12577v1)

Authors: Hasi Hays, William Richardson

Extracellular matrix (ECM) remodeling is central to a wide variety of healthy
and diseased tissue processes. Unfortunately, predicting ECM remodeling under
various chemical and mechanical conditions has proven to be excessively
challenging, due in part to its complex regulation by intracellular and
extracellular molecular reaction networks that are spatially and temporally
dynamic. We introduce ECMSim, which is a highly interactive, real-time, and web
application designed to simulate heterogeneous matrix remodeling. The current
model simulates cardiac scar tissue with configurable input conditions using a
large-scale model of the cardiac fibroblast signaling network. Cardiac fibrosis
is a major component of many forms of heart failure. ECMSim simulates over 1.3
million equations simultaneously in real time that include more than 125
species and more than 200 edges in each cell in a 100*100 spatial array (10,000
cells), which accounts for inputs, receptors, intracellular signaling cascades,
ECM production, and feedback loops, as well as molecular diffusion. The
algorithm is represented by a set of ordinary differential equations (ODEs)
that are coupled with ECM molecular diffusion. The equations are solved on
demand using compiled C++ and the WebAssembly standard. The platform includes
brush-style cell selection to target a subset of cells with adjustable input
molecule concentrations, parameter sliders to adjust parameters on demand, and
multiple coupled real-time visualizations of network dynamics at multiple
scales. Implementing ECMSim in standard web technologies enables a fully
functional application that combines real-time simulation, visual interaction,
and model editing. The software enables the investigation of pathological or
experimental conditions, hypothetical scenarios, matrix remodeling, or the
testing of the effects of an experimental drug(s) with a target receptor.

### 5. [A Quantum Generative Framework for Modeling Single-Cell Transcriptomes with Gene-Gene and Cell-Cell Interactions](http://arxiv.org/pdf/2510.12776v1)

Authors: Selim Romero, Vignesh Kumar, Robert S. Chapkin, James J. Cai

Single-cell RNA sequencing (scRNA-seq) data simulation is limited by
classical methods that rely on linear correlations, failing to capture the
intrinsic, nonlinear dependencies and the simultaneous gene-gene and cell-cell
interactions. We introduce qSimCells, a novel hybrid quantum-classical
simulator that leverages quantum entanglement to model single-cell
transcriptomes. The core innovation is a quantum kernel that uses a
parameterized quantum circuit with CNOT gates to encode complex, nonlinear gene
regulatory network (GRN) and cell-cell communication topologies with explicit
directionality (causality). The synthetic data exhibits non-classical
dependencies that challenge standard analysis. We demonstrated that classical
correlation methods (Pearson and Spearman) failed to reconstruct the complete
programmed quantum causal paths, instead reporting spurious statistical
artifacts driven by high base-gene expression probabilities. Applying
CellChat2.0 to the simulated cell-cell communication validated the true
mechanistic links by showing a robust, relative increase in communication
probability (up to 75-fold) only when the quantum entanglement was active. This
work confirms that the quantum kernel is essential for creating high-fidelity
ground truth data, highlighting the need for advanced inference techniques to
capture the complex, non-classical dependencies inherent in gene regulation.

### Formal Languages and Automata Theory

### 1. [Bringing Algebraic Hierarchical Decompositions to Concatenative Functional Languages](http://arxiv.org/pdf/2510.12481v1)

Authors: Attila Egri-Nagy

Programming languages tend to evolve over time to use more and more concepts
from theoretical computer science. Still, there is a gap between programming
and pure mathematics. Not all theoretical results have realized their promising
applications. The algebraic decomposition of finite state automata
(Krohn-Rhodes Theory) constructs an emulating hierarchical structure from
simpler components for any computing device. These decompositions provide ways
to understand and control computational processes, but so far the applications
were limited to theoretical investigations. Here, we study how to apply
algebraic decompositions to programming languages. We use recent results on
generalizing the algebraic theory to the categorical level (from semigroups to
semigroupoids) and work with the special class of concatenative functional
programming languages. As a first application of semigroupoid decompositions,
we start to design a family of programming languages with an explicit
semigroupoid representation.

### 2. [Flavors of Quantifiers in Hyperlogics](http://arxiv.org/pdf/2510.12298v1)

Authors: Marek Chalupa, Thomas A. Henzinger, Ana Oliveira da Costa

Hypertrace logic is a sorted first-order logic with separate sorts for time
and execution traces. Its formulas specify hyperproperties, which are
properties relating multiple traces. In this work, we extend hypertrace logic
by introducing trace quantifiers that range over the set of all possible
traces. In this extended logic, formulas can quantify over two kinds of trace
variables: constrained trace variables, which range over a fixed set of traces
defined by the model, and unconstrained trace variables, which can be assigned
to any trace. In comparison, hyperlogics such as HyperLTL have only constrained
trace quantifiers. We use hypertrace logic to study how different quantifier
patterns affect the decidability of the satisfiability problem. We prove that
hypertrace logic without constrained trace quantifiers is equivalent to monadic
second-order logic of one successor (S1S), and therefore satisfiable, and that
the trace-prefixed fragment (all trace quantifiers precede all time
quantifiers) is equivalent to HyperQPTL. Moreover, we show that all hypertrace
formulas where the only alternation between constrained trace quantifiers is
from an existential to a universal quantifier are equisatisfiable to formulas
without constraints on their trace variables and, therefore, decidable as well.
Our framework allows us to study also time-prefixed hyperlogics, for which we
provide new decidability and undecidability results

### Graphics

### 1. [Coordinate Condensation: Subspace-Accelerated Coordinate Descent for Physics-Based Simulation](http://arxiv.org/pdf/2510.12053v1)

Authors: Ty Trusty

We introduce Coordinate Condensation, a variant of coordinate descent that
accelerates physics-based simulation by augmenting local coordinate updates
with a Schur-complement-based subspace correction. Recent work by Lan et al.
2025 (JGS2) uses perturbation subspaces to augment local solves to account for
global coupling, but their approach introduces damping that can degrade
convergence. We reuse this subspace but solve for local and subspace
displacements independently, eliminating this damping. For problems where the
subspace adequately captures global coupling, our method achieves near-Newton
convergence while retaining the efficiency and parallelism of coordinate
descent. Through experiments across varying material stiffnesses and mesh
resolutions, we show substantially faster convergence than both standard
coordinate descent and JGS2. We also characterize when subspace-based
coordinate methods succeed or fail, offering insights for future solver design.

### 2. [Can Representation Gaps Be the Key to Enhancing Robustness in Graph-Text Alignment?](http://arxiv.org/pdf/2510.12087v1)

Authors: Heng Zhang, Tianyi Zhang, Yuling Shi, Xiaodong Gu, Yaomin Shen, Zijian Zhang, Yilei Yuan, Hao Zhang, Jin Huang

Representation learning on text-attributed graphs (TAGs) integrates
structural connectivity with rich textual semantics, enabling applications in
diverse domains. Current methods largely rely on contrastive learning to
maximize cross-modal similarity, assuming tighter coupling between graph and
text representations improves transfer performance. However, our empirical
analysis reveals that both natural gap expansion and forced gap reduction
result in performance degradation by disrupting pre-trained knowledge
structures and impairing generalization. This arises from the geometric
incompatibility between encoders, where graph encoders capture topological
patterns, while text encoders capture semantic structures. Over-alignment
compresses these distinct spaces into shared subspaces, causing structure
collapse that diminishes both topological reasoning and semantic understanding.
We propose \textbf{LLM4GTA}, a gap-aware alignment framework that preserves
representation gaps as geometric necessities for maintaining modality-specific
knowledge and improving transfer performance. LLM4GTA includes an adaptive gap
preservation module to prevent over-alignment by monitoring similarity
evolution and an intra-modal compensation mechanism that boosts discriminative
power using auxiliary classifiers in graph space. Extensive experiments show
significant improvements over existing methods in zero-shot and few-shot
scenarios.

### 3. [SDGraph: Multi-Level Sketch Representation Learning by Sparse-Dense Graph Architecture](http://arxiv.org/pdf/2510.12192v1)

Authors: Xi Cheng, Pingfa Feng, Zhichao Liao, Mingyu Fan, Long Zeng

Freehand sketches exhibit unique sparsity and abstraction, necessitating
learning pipelines distinct from those designed for images. For sketch learning
methods, the central objective is to fully exploit the effective information
embedded in sketches. However, there is limited research on what constitutes
effective sketch information, which in turn constrains the performance of
existing approaches. To tackle this issue, we first proposed the Multi-Level
Sketch Representation Scheme to systematically identify the effective
information. The scheme organizes sketch representation into three levels:
sketch-level, stroke-level, and point-level. This design is based on the
granularity of analytical elements, from coarse (sketch-level) to fine
(point-level), thereby ensuring more comprehensive coverage of the sketch
information. For each level, we conducted theoretical analyses and experimental
evaluations to identify and validate the effective information. Building on the
above studies, we developed SDGraph, a deep learning architecture designed to
exploit the identified effective information across the three levels. SDGraph
comprises two complementary modules: a Sparse Graph that treats strokes as
nodes for sketch-level and stroke-level representation learning, and a Dense
Graph that treats points as nodes for sketch-level and point-level
representation learning. Both modules employ graph convolution along with
down-sampling and up-sampling operations, enabling them to function as both
encoder and decoder. Besides that, an information fusion module bridges the two
graphs to further enhance feature extraction. SDGraph supports a wide range of
sketch-related downstream tasks, achieving accuracy improvements of 1.15\% and
1.70\% over the state-of-the-art in classification and retrieval, respectively,
and 36.58\% improvement in vector sketch generation quality.

### 4. [GraphShaper: Geometry-aware Alignment for Improving Transfer Learning in Text-Attributed Graphs](http://arxiv.org/pdf/2510.12085v1)

Authors: Heng Zhang, Tianyi Zhang, Yuling Shi, Xiaodong Gu, Yaomin Shen, Haochen You, Zijian Zhang, Yilei Yuan, Jin Huang

Graph foundation models represent a transformative paradigm for learning
transferable representations across diverse graph domains. Recent methods
leverage large language models to unify graph and text modalities into a shared
representation space using contrastive learning. However, systematic
evaluations reveal significant performance degradation at structural boundaries
where distinct topological patterns converge, with accuracy losses exceeding 20
percentage points. This issue arises from a key limitation: current methods
assume all graph structures can be encoded within a single Euclidean space. In
reality, tree structures require hyperbolic geometry to preserve hierarchical
branching, while cyclic patterns depend on spherical geometry for closure
properties. At structural boundaries, nodes experience conflicting geometric
constraints that uniform encoding spaces cannot resolve. This raises a crucial
challenge: \textbf{Can alignment frameworks be designed to respect the
intrinsic geometric diversity of graph structures?} We introduce
\textbf{GraphShaper}, a geometry-aware framework that enhances graph encoding
through multi-geometric specialization. Our approach employs expert networks
tailored to different geometric spaces, dynamically computing fusion weights to
adaptively integrate geometric properties based on local structural
characteristics. This adaptive fusion preserves structural integrity before
alignment with text embeddings. Extensive experiments demonstrate that
GraphShaper achieves 9.47\% accuracy improvements on citation networks and
7.63\% on social networks in zero-shot settings.

### 5. [H4G: Unlocking Faithful Inference for Zero-Shot Graph Learning in Hyperbolic Space](http://arxiv.org/pdf/2510.12094v1)

Authors: Heng Zhang, Tianyi Zhang, Zijun Liu, Yuling Shi, Yaomin Shen, Haochen You, Haichuan Hu, Lubin Gan, Jin Huang

Text-attributed graphs are widely used across domains, offering rich
opportunities for zero-shot learning via graph-text alignment. However,
existing methods struggle with tasks requiring fine-grained pattern
recognition, particularly on heterophilic graphs. Through empirical and
theoretical analysis, we identify an \textbf{over-abstraction problem}: current
approaches operate at excessively large hyperbolic radii, compressing
multi-scale structural information into uniform high-level abstractions. This
abstraction-induced information loss obscures critical local patterns essential
for accurate predictions. By analyzing embeddings in hyperbolic space, we
demonstrate that optimal graph learning requires \textbf{faithful preservation}
of fine-grained structural details, better retained by representations
positioned closer to the origin. To address this, we propose \textbf{H4G}, a
framework that systematically reduces embedding radii using learnable
block-diagonal scaling matrices and M\"obius matrix multiplication. This
approach restores access to fine-grained patterns while maintaining global
receptive ability with minimal computational overhead. Experiments show H4G
achieves state-of-the-art zero-shot performance with \textbf{12.8\%}
improvement on heterophilic graphs and \textbf{8.4\%} on homophilic graphs,
confirming that radius reduction enables faithful multi-scale representation
for advancing zero-shot graph learning.

### 6. [Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](http://arxiv.org/pdf/2510.12768v1)

Authors: Fengzhi Guo, Chih-Chuan Hsu, Sihao Ding, Cheng Zhang

Reconstructing dynamic 3D scenes from monocular input is fundamentally
under-constrained, with ambiguities arising from occlusion and extreme novel
views. While dynamic Gaussian Splatting offers an efficient representation,
vanilla models optimize all Gaussian primitives uniformly, ignoring whether
they are well or poorly observed. This limitation leads to motion drifts under
occlusion and degraded synthesis when extrapolating to unseen views. We argue
that uncertainty matters: Gaussians with recurring observations across views
and time act as reliable anchors to guide motion, whereas those with limited
visibility are treated as less reliable. To this end, we introduce USplat4D, a
novel Uncertainty-aware dynamic Gaussian Splatting framework that propagates
reliable motion cues to enhance 4D reconstruction. Our key insight is to
estimate time-varying per-Gaussian uncertainty and leverages it to construct a
spatio-temporal graph for uncertainty-aware optimization. Experiments on
diverse real and synthetic datasets show that explicitly modeling uncertainty
consistently improves dynamic Gaussian Splatting models, yielding more stable
geometry under occlusion and high-quality synthesis at extreme viewpoints.

### 7. [MVP4D: Multi-View Portrait Video Diffusion for Animatable 4D Avatars](http://arxiv.org/pdf/2510.12785v1)

Authors: Felix Taubner, Ruihang Zhang, Mathieu Tuli, Sherwin Bahmani, David B. Lindell

Digital human avatars aim to simulate the dynamic appearance of humans in
virtual environments, enabling immersive experiences across gaming, film,
virtual reality, and more. However, the conventional process for creating and
animating photorealistic human avatars is expensive and time-consuming,
requiring large camera capture rigs and significant manual effort from
professional 3D artists. With the advent of capable image and video generation
models, recent methods enable automatic rendering of realistic animated avatars
from a single casually captured reference image of a target subject. While
these techniques significantly lower barriers to avatar creation and offer
compelling realism, they lack constraints provided by multi-view information or
an explicit 3D representation. So, image quality and realism degrade when
rendered from viewpoints that deviate strongly from the reference image. Here,
we build a video model that generates animatable multi-view videos of digital
humans based on a single reference image and target expressions. Our model,
MVP4D, is based on a state-of-the-art pre-trained video diffusion model and
generates hundreds of frames simultaneously from viewpoints varying by up to
360 degrees around a target subject. We show how to distill the outputs of this
model into a 4D avatar that can be rendered in real-time. Our approach
significantly improves the realism, temporal consistency, and 3D consistency of
generated avatars compared to previous methods.

### Computer Science and Game Theory

### 1. [Fair Division of Indivisible Items](http://arxiv.org/pdf/2510.12158v1)

Authors: Kevin Hsu

We study the fair division of indivisible items. In the general model, the
goal is to allocate $m$ indivisible items to $n$ agents while satisfying
fairness criteria such as MMS, EF1, and EFX. We also study a
recently-introduced graphical model that represents the fair division problem
as a multigraph, in which vertices correspond to agents and edges to items. The
graphical model stipulates that an item can have non-zero marginal utility to
an agent only if its corresponding edge is incident to the agent's
corresponding vertex. We study orientations (allocations that allocate each
edge to an endpoint) in this model, as they are particularly desirable.
  Our first contribution concerns MMS allocations of mixed manna (i.e. a
mixture of goods and chores) in the general model. It is known that MMS
allocations of goods exist when $m \leq n+5$. We generalize this and show that
when $m \leq n+5$, MMS allocations of mixed manna exist as long as $n \leq 3$,
there is an agent whose MMS threshold is non-negative, or every item is a
chore. Remarkably, our result leaves only the case where every agent has a
negative MMS threshold unanswered.
  Our second contribution concerns EFX orientations of multigraphs of goods. We
show that deciding whether EFX orientations exist for multigraphs is
NP-complete, even for symmetric bi-valued multigraphs. Complementarily, we show
symmetric bi-valued multigraphs that do not contain non-trivial odd multitrees
have EFX orientations that can be found in polynomial time.
  Our third contribution concerns EF1 and EFX orientations of graphs and
multigraphs of chores. We obtain polynomial-time algorithms for deciding
whether such graphs have EF1 and EFX orientations, resolving a previous
conjecture and showing a fundamental difference between goods and chores
division. In addition, we show that the analogous problems for multigraphs are
NP-hard.

### 2. [Perceived Fairness in Networks](http://arxiv.org/pdf/2510.12028v1)

Authors: Arthur Charpentier

The usual definitions of algorithmic fairness focus on population-level
statistics, such as demographic parity or equal opportunity. However, in many
social or economic contexts, fairness is not perceived globally, but locally,
through an individual's peer network and comparisons. We propose a theoretical
model of perceived fairness networks, in which each individual's sense of
discrimination depends on the local topology of interactions. We show that even
if a decision rule satisfies standard criteria of fairness, perceived
discrimination can persist or even increase in the presence of homophily or
assortative mixing. We propose a formalism for the concept of fairness
perception, linking network structure, local observation, and social
perception. Analytical and simulation results highlight how network topology
affects the divergence between objective fairness and perceived fairness, with
implications for algorithmic governance and applications in finance and
collaborative insurance.

### 3. [Single-Deviation Stability in Additively Separable Hedonic Games with Constrained Coalition Sizes](http://arxiv.org/pdf/2510.12641v1)

Authors: Martin Bullinger, Adam Dunajski, Edith Elkind, Matan Gilboa

We study stability in additively separable hedonic games when coalition sizes
have to respect fixed size bounds. We consider four classic notions of
stability based on single-agent deviations, namely, Nash stability, individual
stability, contractual Nash stability, and contractual individual stability.
For each stability notion, we consider two variants: in one, the coalition left
behind by a deviator must still be of a valid size, and in the other there is
no such constraint. We provide a full picture of the existence of stable
outcomes with respect to given size parameters. Additionally, when there are
only upper bounds, we fully characterize the computational complexity of the
associated existence problem. In particular, we obtain polynomial-time
algorithms for contractual individual stability and contractual Nash stability,
where the latter requires an upper bound of 2. We obtain further results for
Nash stability and contractual individual stability, when the lower bound is at
least 2.

### Human-Computer Interaction

### 1. [Social Simulation for Integrating Self-Care: Measuring the Effects of Contextual Environments in Augmented Reality for Mental Health Practice](http://arxiv.org/pdf/2510.12081v1)

Authors: Anna Fang, Jiayang Shi, Hriday Chhabria, Bosi Li, Haiyi Zhu

Despite growing interest in virtual and augmented reality (VR/AR) for mental
well-being, prior work using immersive interventions to teach mental health
skills has largely focused on calming or abstract settings. As a result, little
is known about how realistic social simulation may better support the transfer
and application of skills to in-person environments. In this work, we present a
14-day user study with 43-participants comparing an augmented reality
intervention simulating a realistic contextual environment against a matched
non-contextual control, applied to the public speaking context. We found that
participants who practice mental health skills in the contextual environment
showed significantly greater likelihood to apply self-care techniques and
greater physiological stress reduction when using skills in mock in-person
tasks. Overall, our work provides empirical evidence for the effects of
realistic stressor simulation, and offers design implications for mental health
technology that supports effective transfer of skills to the real-world.

### 2. [KnowledgeTrail: Generative Timeline for Exploration and Sensemaking of Historical Events and Knowledge Formation](http://arxiv.org/pdf/2510.12113v1)

Authors: Sangho Suh, Rahul Hingorani, Bryan Wang, Tovi Grossman

The landscape of interactive systems is shifting toward dynamic, generative
experiences that empower users to explore and construct knowledge in real time.
Yet, timelines -- a fundamental tool for representing historical and conceptual
development -- remain largely static, limiting user agency and curiosity. We
introduce the concept of a generative timeline: an AI-powered timeline that
adapts to users' evolving questions by expanding or contracting in response to
input. We instantiate this concept through KnowledgeTrail, a system that
enables users to co-construct timelines of historical events and knowledge
formation processes. Two user studies showed that KnowledgeTrail fosters
curiosity-driven exploration, serendipitous discovery, and the ability to trace
complex relationships between ideas and events, while citation features
supported verification yet revealed fragile trust shaped by perceptions of
source credibility. We contribute a vision for generative timelines as a new
class of exploratory interface, along with design insights for balancing
serendipity and credibility.

### 3. [Lowering Barriers to CAD Adoption: A Comparative Study of Augmented Reality-Based CAD (AR-CAD) and a Traditional CAD tool](http://arxiv.org/pdf/2510.12146v1)

Authors: Muhammad Talha, Abdullah Mohiuddin, Sehrish Javed, Ahmed Jawad Qureshi

The paper presents a comparative user study between an Augmented
Reality-based Computer-Aided Design (AR-CAD) system and a traditional
computer-based CAD modeling software, SolidWorks. Twenty participants of
varying skill levels performed 3D modeling tasks using both systems. The
results showed that while the average task completion time is comparable for
both groups, novice designers had a higher completion rate in AR-CAD than in
the traditional CAD interface, and experienced designers had a similar
completion rate in both systems. A statistical comparison of task completion
rate, time, and NASA Task Load Index (TLX) showed that AR-CAD slightly reduced
cognitive load while favoring a high task completion rate. Higher scores on the
System Usability Scale (SUS) by novices indicated that AR-CAD was superior and
worthwhile for reducing barriers to entering CAD. In contrast, the Traditional
CAD interface was favored by experienced users for its advanced capabilities,
while many viewed AR-CAD as a valid means for rapid concept development,
education, and an initial critique of designs. This opens up the need for
future research on the needed refinement of AR-CAD with a focus on
high-precision input tools and its evaluation of complex design processes. This
research highlights the potential for immersive interfaces to enhance design
practice, bridging the gap between novice and experienced CAD users.

### 4. [Embodied Natural Language Interaction (NLI): Speech Input Patterns in Immersive Analytics](http://arxiv.org/pdf/2510.12156v1)

Authors: Hyemi Song, Matthew Johnson, Kirsten Whitley, Eric Krokos, Amitabh Varshney

Embodiment shapes how users verbally express intent when interacting with
data through speech interfaces in immersive analytics. Despite growing interest
in Natural Language Interaction (NLI) for visual analytics in immersive
environments, users' speech patterns and their use of embodiment cues in speech
remain underexplored. Understanding their interplay is crucial to bridging the
gap between users' intent and an immersive analytic system. To address this, we
report the results from 15 participants in a user study conducted using the
Wizard of Oz method. We performed axial coding on 1,280 speech acts derived
from 734 utterances, examining how analysis tasks are carried out with
embodiment and linguistic features. Next, we measured speech input uncertainty
for each analysis task using the semantic entropy of utterances, estimating how
uncertain users' speech inputs appear to an analytic system. Through these
analyses, we identified five speech input patterns, showing that users
dynamically blend embodied and non-embodied speech acts depending on data
analysis tasks, phases, and embodiment reliance driven by the counts and types
of embodiment cues in each utterance. We then examined how these patterns align
with user reflections on factors that challenge speech interaction during the
study. Finally, we propose design implications aligned with the five patterns.

### 5. [How Far I'll Go: Imagining Futures of Conversational AI with People with Visual Impairments Through Design Fiction](http://arxiv.org/pdf/2510.12268v1)

Authors: Jeanne Choi, Dasom Choi, Sejun Jeong, Hwajung Hong, Joseph Seering

People with visual impairments (PVI) use a variety of assistive technologies
to navigate their daily lives, and conversational AI (CAI) tools are a growing
part of this toolset. Much existing HCI research has focused on the technical
capabilities of current CAI tools, but in this paper, we instead examine how
PVI themselves envision potential futures for living with CAI. We conducted a
study with 14 participants with visual impairments using an audio-based Design
Fiction probe featuring speculative dialogues between participants and a future
CAI. Participants imagined using CAI to expand their boundaries by exploring
new opportunities or places, but also voiced concerns about balancing reliance
on CAI with maintaining autonomy, the need to consider diverse levels of
vision-loss, and enhancing visibility of PVI for greater inclusion. We discuss
implications for designing CAI that support genuine agency for PVI based on the
future lives they envisioned.

### 6. [Hey Dashboard!: Supporting Voice, Text, and Pointing Modalities in Dashboard Onboarding](http://arxiv.org/pdf/2510.12386v1)

Authors: Vaishali Dhanoa, Gabriela Molina León, Eve Hoggan, Eduard Gröller, Marc Streit, Niklas Elmqvist

Visualization dashboards are regularly used for data exploration and
analysis, but their complex interactions and interlinked views often require
time-consuming onboarding sessions from dashboard authors. Preparing these
onboarding materials is labor-intensive and requires manual updates when
dashboards change. Recent advances in multimodal interaction powered by large
language models (LLMs) provide ways to support self-guided onboarding. We
present DIANA (Dashboard Interactive Assistant for Navigation and Analysis), a
multimodal dashboard assistant that helps users for navigation and guided
analysis through chat, audio, and mouse-based interactions. Users can choose
any interaction modality or a combination of them to onboard themselves on the
dashboard. Each modality highlights relevant dashboard features to support user
orientation. Unlike typical LLM systems that rely solely on text-based chat,
DIANA combines multiple modalities to provide explanations directly in the
dashboard interface. We conducted a qualitative user study to understand the
use of different modalities for different types of onboarding tasks and their
complexities.

### 7. [Gauging the Competition: Understanding Social Comparison and Anxiety through Eye-tracking in Virtual Reality Group Interview](http://arxiv.org/pdf/2510.12590v1)

Authors: Shi-Ting Ni, Kairong Fang, Yuyang Wang, Pan Hui

Virtual Reality (VR) is a promising tool for interview training, yet the
psychological dynamics of group interviews, such as social comparison, remain
underexplored. We investigate this phenomenon by developing an immersive VR
group interview system and conducting an eye-tracking study with 73
participants. We manipulated peer performance using ambiguous behavioral cues
(e.g., hand-raising) and objective information (public test scores) to measure
their effect on participants' attention and self-concept. Our results
demonstrate a "Big-Fish-Little-Pond Effect" in VR: an increase in
high-achieving peer behaviors heightened participants' processing of social
comparison information and significantly lowered their self-assessments. The
introduction of objective scores further intensified these comparative
behaviors. We also found that lower perceived realism of the VR environment
correlated with higher anxiety. These findings offer key insights and design
considerations for creating more effective and psychologically-aware virtual
training environments that account for complex social dynamics.

### 8. [CrisisNews: A Dataset Mapping Two Decades of News Articles on Online Problematic Behavior at Scale](http://arxiv.org/pdf/2510.12243v1)

Authors: Jeanne Choi, DongJae Kang, Yubin Choi, Juhoon Lee, Joseph Seering

As social media adoption grows globally, online problematic behaviors
increasingly escalate into large-scale crises, requiring an evolving set of
mitigation strategies. While HCI research often analyzes problematic behaviors
with pieces of user-generated content as the unit of analysis, less attention
has been given to event-focused perspectives that track how discrete events
evolve. In this paper, we examine 'social media crises': discrete patterns of
problematic behaviors originating and evolving within social media that cause
larger-scale harms. Using global news coverage, we present a dataset of 93,250
news articles covering social media-endemic crises from the past 20 years. We
analyze a representative subset to classify stakeholder roles, behavior types,
and outcomes, uncovering patterns that inform more nuanced classification of
social media crises beyond content-based descriptions. By adopting a wider
perspective, this research seeks to inform the design of safer platforms,
enabling proactive measures to mitigate crises and foster more trustworthy
online environments.

### 9. [Data-Model Co-Evolution: Growing Test Sets to Refine LLM Behavior](http://arxiv.org/pdf/2510.12728v1)

Authors: Minjae Lee, Minsuk Kahng

A long-standing challenge in machine learning has been the rigid separation
between data work and model refinement, enforced by slow fine-tuning cycles.
The rise of Large Language Models (LLMs) overcomes this historical barrier,
allowing applications developers to instantly govern model behavior by editing
prompt instructions. This shift enables a new paradigm: data-model
co-evolution, where a living test set and a model's instructions evolve in
tandem. We operationalize this paradigm in an interactive system designed to
address the critical challenge of encoding subtle, domain-specific policies
into prompt instructions. The system's structured workflow guides people to
discover edge cases, articulate rationales for desired behavior, and
iteratively evaluate instruction revisions against a growing test set. A user
study shows our workflow helps participants refine instructions systematically
and specify ambiguous policies more concretely. This work points toward more
robust and responsible LLM applications through human-in-the-loop development
aligned with local preferences and policies.

### 10. [(R)evolution of Programming: Vibe Coding as a Post-Coding Paradigm](http://arxiv.org/pdf/2510.12364v1)

Authors: Kevin Krings, Nino S. Bohn, Thomas Ludwig

Recent advancements in generative artificial intelligence (GenAI),
particularly large language models, have introduced new possibilities for
software development practices. In our paper we investigate the emerging Vibe
Coding (VC) paradigm that emphasizes intuitive, affect-driven, and
improvisational interactions between developers and AI systems. Building upon
the discourse of End-User Development (EUD), we explore how VC diverges from
conventional programming approaches such as those supported by tools like
GitHub Copilot. Through five semi-structured interview sessions with ten
experienced software practitioners, we identify five thematic dimensions:
creativity, sustainability, the future of programming, collaboration, and
criticism. Our analysis conceptualizes VC within the metaphor of co-drifting,
contrasting it with the prevalent co-piloting perspective of AI-assisted
development. We argue that VC reconfigures the developers role, blurring
boundaries between professional and non-developers. While VC enables novel
forms of expression and rapid prototyping, it also introduces challenges
regarding reproducibility, scalability, and inclusivity. We propose that VC
represents a meaningful shift in programming culture, warranting further
investigation within human-computer interaction (HCI) and software engineering
research.

### Information Retrieval

### 1. [Reinforced Preference Optimization for Recommendation](http://arxiv.org/pdf/2510.12211v1)

Authors: Junfei Tan, Yuxin Chen, An Zhang, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Xiang Wang

Recent breakthroughs in large language models (LLMs) have fundamentally
shifted recommender systems from discriminative to generative paradigms, where
user behavior modeling is achieved by generating target items conditioned on
historical interactions. Yet current generative recommenders still suffer from
two core limitations: the lack of high-quality negative modeling and the
reliance on implicit rewards. Reinforcement learning with verifiable rewards
(RLVR) offers a natural solution by enabling on-policy sampling of harder
negatives and grounding optimization in explicit reward signals. However,
applying RLVR to generative recommenders remains non-trivial. Its unique
generation space often leads to invalid or repetitive items that undermine
sampling efficiency, and ranking supervision is sparse since most items receive
identical zero rewards. To address these challenges, we propose Reinforced
Preference Optimization for Recommendation (ReRe), a reinforcement-based
paradigm tailored to LLM-based recommenders, an important direction in
generative recommendation. ReRe incorporates constrained beam search to improve
sampling efficiency and diversify hard negatives, while augmenting rule-based
accuracy rewards with auxiliary ranking rewards for finer-grained supervision.
Extensive experiments on three real-world datasets demonstrate that ReRe
consistently outperforms both traditional and LLM-based recommenders in ranking
performance. Further analysis shows that ReRe not only enhances performance
across both base and SFT-initialized models but also generalizes robustly
across different backbone families and scales. Beyond empirical gains, we
systematically investigate the design space of RLVR in recommendation across
generation, sampling strategy, reward modeling, and optimization algorithm,
offering insights for future research.

### 2. [An Empirical Study for Representations of Videos in Video Question Answering via MLLMs](http://arxiv.org/pdf/2510.12299v1)

Authors: Zhi Li, Yanan Wang, Hao Niu, Julio Vizcarra, Masato Taya

Multimodal large language models have recently achieved remarkable progress
in video question answering (VideoQA) by jointly processing visual, textual,
and audio information. However, it remains unclear which video representations
are most effective for MLLMs, and how different modalities balance task
accuracy against computational efficiency. In this work, we present a
comprehensive empirical study of video representation methods for VideoQA with
MLLMs. We systematically evaluate single modality inputs question only,
subtitles, visual frames, and audio signals as well as multimodal combinations,
on two widely used benchmarks: VideoMME and LongVideoBench. Our results show
that visual frames substantially enhance accuracy but impose heavy costs in GPU
memory and inference latency, while subtitles provide a lightweight yet
effective alternative, particularly for long videos. These findings highlight
clear trade-offs between effectiveness and efficiency and provide practical
insights for designing resource-aware MLLM-based VideoQA systems.

### 3. [A Hierarchical Quantized Tokenization Framework for Task-Adaptive Graph Representation Learning](http://arxiv.org/pdf/2510.12369v1)

Authors: Yang Xiang, Li Fan, Chenke Yin, Chengtao Ji

Recent progress in language and vision foundation models demonstrates the
importance of discrete token interfaces that transform complex inputs into
compact sequences for large-scale modeling. Extending this paradigm to graphs
requires a tokenization scheme that handles non-Euclidean structures and
multi-scale dependencies efficiently. Existing approaches to graph
tokenization, linearized, continuous, and quantized, remain limited in
adaptability and efficiency. In particular, most current quantization-based
tokenizers organize hierarchical information in fixed or task-agnostic ways,
which may either over-represent or under-utilize structural cues, and lack the
ability to dynamically reweight contributions from different levels without
retraining the encoder. This work presents a hierarchical quantization
framework that introduces a self-weighted mechanism for task-adaptive
aggregation across multiple scales. The proposed method maintains a frozen
encoder while modulating information flow through a lightweight gating process,
enabling parameter-efficient adaptation to diverse downstream tasks.
Experiments on benchmark datasets for node classification and link prediction
demonstrate consistent improvements over strong baselines under comparable
computational budgets.

### 4. [Leveraging Language Semantics for Collaborative Filtering with TextGCN and TextGCN-MLP: Zero-Shot vs In-Domain Performance](http://arxiv.org/pdf/2510.12461v1)

Authors: Andrei Chernov, Haroon Wahab, Oleg Novitskij

In recent years, various approaches have been proposed to leverage large
language models (LLMs) for incorporating textual information about items into
recommender systems. Existing methods primarily focus on either fine-tuning
LLMs to generate recommendations or integrating LLM-based embeddings into
downstream models. In this work, we follow the latter direction and propose
\textbf{TextGCN}, which applies parameter-free graph convolution layers
directly over LLM-based item-title embeddings, instead of learning ID-based
embeddings as in traditional methods. By combining language semantics with
graph message passing, this architecture achieves state-of-the-art zero-shot
performance, significantly outperforming prior approaches. Furthermore, we
introduce \textbf{TextGCN-MLP}, which extends TextGCN with a trainable
multilayer perceptron trained using a contrastive loss, achieving
state-of-the-art in-domain performance on recommendation benchmarks. However,
the zero-shot performance of TextGCN-MLP remains lower than that of TextGCN,
highlighting the trade-off between in-domain specialization and zero-shot
generalization. We release our code on github at
\href{https://github.com/ChernovAndrey/TFCE}{github.com/ChernovAndrey/TFCE}.

### 5. [MIARec: Mutual-influence-aware Heterogeneous Network Embedding for Scientific Paper Recommendation](http://arxiv.org/pdf/2510.12054v1)

Authors: Wenjin Xie, Tao Jia

With the rapid expansion of scientific literature, scholars increasingly
demand precise and high-quality paper recommendations. Among various
recommendation methodologies, graph-based approaches have garnered attention by
effectively exploiting the structural characteristics inherent in scholarly
networks. However, these methods often overlook the asymmetric academic
influence that is prevalent in scholarly networks when learning graph
representations. To address this limitation, this study proposes the
Mutual-Influence-Aware Recommendation (MIARec) model, which employs a
gravity-based approach to measure the mutual academic influence between
scholars and incorporates this influence into the feature aggregation process
during message propagation in graph representation learning. Additionally, the
model utilizes a multi-channel aggregation method to capture both individual
embeddings of distinct single relational sub-networks and their interdependent
embeddings, thereby enabling a more comprehensive understanding of the
heterogeneous scholarly network. Extensive experiments conducted on real-world
datasets demonstrate that the MIARec model outperforms baseline models across
three primary evaluation metrics, indicating its effectiveness in scientific
paper recommendation tasks.

### 6. [Causal Inspired Multi Modal Recommendation](http://arxiv.org/pdf/2510.12325v1)

Authors: Jie Yang, Chenyang Gu, Zixuan Liu

Multimodal recommender systems enhance personalized recommendations in
e-commerce and online advertising by integrating visual, textual, and user-item
interaction data. However, existing methods often overlook two critical biases:
(i) modal confounding, where latent factors (e.g., brand style or product
category) simultaneously drive multiple modalities and influence user
preference, leading to spurious feature-preference associations; (ii)
interaction bias, where genuine user preferences are mixed with noise from
exposure effects and accidental clicks. To address these challenges, we propose
a Causal-inspired multimodal Recommendation framework. Specifically, we
introduce a dual-channel cross-modal diffusion module to identify hidden modal
confounders, utilize back-door adjustment with hierarchical matching and
vector-quantized codebooks to block confounding paths, and apply front-door
adjustment combined with causal topology reconstruction to build a deconfounded
causal subgraph. Extensive experiments on three real-world e-commerce datasets
demonstrate that our method significantly outperforms state-of-the-art
baselines while maintaining strong interpretability.

### 7. [SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce SEarch](http://arxiv.org/pdf/2510.12604v1)

Authors: Qihang Zhao, Zhongbo Sun, Xiaoyang Zheng, Xian Guo, Siyuan Wang, Zihan Liang, Mingcan Peng, Ben Chen, Chenyi Lei

With the rise of modern search and recommendation platforms, insufficient
collaborative information of cold-start items exacerbates the Matthew effect of
existing platform items, challenging platform diversity and becoming a
longstanding issue. Existing methods align items' side content with
collaborative information to transfer collaborative signals from
high-popularity items to cold-start items. However, these methods fail to
account for the asymmetry between collaboration and content, nor the
fine-grained differences among items. To address these issues, we propose
SMILE, an item representation enhancement approach based on fused alignment of
semantic IDs. Specifically, we use RQ-OPQ encoding to quantize item content and
collaborative information, followed by a two-step alignment: RQ encoding
transfers shared collaborative signals across items, while OPQ encoding learns
differentiated information of items. Comprehensive offline experiments on
large-scale industrial datasets demonstrate superiority of SMILE, and rigorous
online A/B tests confirm statistically significant improvements: item CTR
+1.66%, buyers +1.57%, and order volume +2.17%.

### 8. [The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation](http://arxiv.org/pdf/2510.12668v1)

Authors: Minghao Tang, Shiyu Ni, Jingtong Wu, Zengxin Han, Keping Bi

Retrieval-augmented generation (RAG) enhances large language models (LLMs) by
retrieving external documents. As an emerging form of RAG, parametric
retrieval-augmented generation (PRAG) encodes documents as model parameters
(i.e., LoRA modules) and injects these representations into the model during
inference, enabling interaction between the LLM and documents at parametric
level. Compared with directly placing documents in the input context, PRAG is
more efficient and has the potential to offer deeper model-document
interaction. Despite its growing attention, the mechanism underlying parametric
injection remains poorly understood. In this work, we present a systematic
study of PRAG to clarify the role of parametric injection, showing that
parameterized documents capture only partial semantic information of documents,
and relying on them alone yields inferior performance compared to interaction
at text level. However, these parametric representations encode high-level
document information that can enhance the model's understanding of documents
within the input context. When combined parameterized documents with textual
documents, the model can leverage relevant information more effectively and
become more robust to noisy inputs, achieving better performance than either
source alone. We recommend jointly using parameterized and textual documents
and advocate for increasing the information content of parametric
representations to advance PRAG.

### 9. [SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Model](http://arxiv.org/pdf/2510.12709v1)

Authors: Lin Lin, Jiefeng Long, Zhihe Wan, Yuchi Wang, Dingkang Yang, Shuang Yang, Yueyang Yao, Xu Chen, Zirui Guo, Shengqiang Li, Weiran Li, Hanyu Li, Yaling Mou, Yan Qiu, Haiyang Yu, Xiao Liang, Hongsheng Li, Chao Feng

Multimodal embedding models aim to yield informative unified representations
that empower diverse cross-modal tasks. Despite promising developments in the
evolution from CLIP-based dual-tower architectures to large vision-language
models, prior works still face unavoidable challenges in real-world
applications and business scenarios, such as the limited modality support,
unstable training mechanisms, and industrial domain gaps. In this work, we
introduce SAIL-Embedding, an omni-modal embedding foundation model that
addresses these issues through tailored training strategies and architectural
design. In the optimization procedure, we propose a multi-stage training scheme
to boost the multifaceted effectiveness of representation learning.
Specifically, the content-aware progressive training aims to enhance the
model's adaptability to diverse downstream tasks and master enriched
cross-modal proficiency. The collaboration-aware recommendation enhancement
training further adapts multimodal representations for recommendation scenarios
by distilling knowledge from sequence-to-item and ID-to-item embeddings while
mining user historical interests. Concurrently, we develop the stochastic
specialization and dataset-driven pattern matching to strengthen model training
flexibility and generalizability. Experimental results show that SAIL-Embedding
achieves SOTA performance compared to other methods in different retrieval
tasks. In online experiments across various real-world scenarios integrated
with our model, we observe a significant increase in Lifetime (LT), which is a
crucial indicator for the recommendation experience. For instance, the model
delivers the 7-day LT gain of +0.158% and the 14-day LT gain of +0.144% in the
Douyin-Selected scenario. For the Douyin feed rank model, the match features
produced by SAIL-Embedding yield a +0.08% AUC gain.

### 10. [CTRL-Rec: Controlling Recommender Systems With Natural Language](http://arxiv.org/pdf/2510.12742v1)

Authors: Micah Carroll, Adeline Foote, Kevin Feng, Marcus Williams, Anca Dragan, W. Bradley Knox, Smitha Milli

When users are dissatisfied with recommendations from a recommender system,
they often lack fine-grained controls for changing them. Large language models
(LLMs) offer a solution by allowing users to guide their recommendations
through natural language requests (e.g., "I want to see respectful posts with a
different perspective than mine"). We propose a method, CTRL-Rec, that allows
for natural language control of traditional recommender systems in real-time
with computational efficiency. Specifically, at training time, we use an LLM to
simulate whether users would approve of items based on their language requests,
and we train embedding models that approximate such simulated judgments. We
then integrate these user-request-based predictions into the standard weighting
of signals that traditional recommender systems optimize. At deployment time,
we require only a single LLM embedding computation per user request, allowing
for real-time control of recommendations. In experiments with the MovieLens
dataset, our method consistently allows for fine-grained control across a
diversity of requests. In a study with 19 Letterboxd users, we find that
CTRL-Rec was positively received by users and significantly enhanced users'
sense of control and satisfaction with recommendations compared to traditional
controls.

### Machine Learning

### 1. [Influence Dynamics and Stagewise Data Attribution](http://arxiv.org/pdf/2510.12071v1)

Authors: Jin Hwa Lee, Matthew Smith, Maxwell Adam, Jesse Hoogland

Current training data attribution (TDA) methods treat the influence one
sample has on another as static, but neural networks learn in distinct stages
that exhibit changing patterns of influence. In this work, we introduce a
framework for stagewise data attribution grounded in singular learning theory.
We predict that influence can change non-monotonically, including sign flips
and sharp peaks at developmental transitions. We first validate these
predictions analytically and empirically in a toy model, showing that dynamic
shifts in influence directly map to the model's progressive learning of a
semantic hierarchy. Finally, we demonstrate these phenomena at scale in
language models, where token-level influence changes align with known
developmental stages.

### 2. [Rethinking the Role of Dynamic Sparse Training for Scalable Deep Reinforcement Learning](http://arxiv.org/pdf/2510.12096v1)

Authors: Guozheng Ma, Lu Li, Zilin Wang, Haoyu Wang, Shengchao Hu, Leszek Rutkowski, Dacheng Tao

Scaling neural networks has driven breakthrough advances in machine learning,
yet this paradigm fails in deep reinforcement learning (DRL), where larger
models often degrade performance due to unique optimization pathologies such as
plasticity loss. While recent works show that dynamically adapting network
topology during training can mitigate these issues, existing studies have three
critical limitations: (1) applying uniform dynamic training strategies across
all modules despite encoder, critic, and actor following distinct learning
paradigms, (2) focusing evaluation on basic architectures without clarifying
the relative importance and interaction between dynamic training and
architectural improvements, and (3) lacking systematic comparison between
different dynamic approaches including sparse-to-sparse, dense-to-sparse, and
sparse-to-dense. Through comprehensive investigation across modules and
architectures, we reveal that dynamic sparse training strategies provide
module-specific benefits that complement the primary scalability foundation
established by architectural improvements. We finally distill these insights
into Module-Specific Training (MST), a practical framework that further
exploits the benefits of architectural improvements and demonstrates
substantial scalability gains across diverse RL algorithms without algorithmic
modifications.

### 3. [Graph Few-Shot Learning via Adaptive Spectrum Experts and Cross-Set Distribution Calibration](http://arxiv.org/pdf/2510.12140v1)

Authors: Yonghao Liu, Yajun Wang, Chunli Guo, Wei Pang, Ximing Li, Fausto Giunchiglia, Xiaoyue Feng, Renchu Guan

Graph few-shot learning has attracted increasing attention due to its ability
to rapidly adapt models to new tasks with only limited labeled nodes. Despite
the remarkable progress made by existing graph few-shot learning methods,
several key limitations remain. First, most current approaches rely on
predefined and unified graph filters (e.g., low-pass or high-pass filters) to
globally enhance or suppress node frequency signals. Such fixed spectral
operations fail to account for the heterogeneity of local topological
structures inherent in real-world graphs. Moreover, these methods often assume
that the support and query sets are drawn from the same distribution. However,
under few-shot conditions, the limited labeled data in the support set may not
sufficiently capture the complex distribution of the query set, leading to
suboptimal generalization. To address these challenges, we propose GRACE, a
novel Graph few-shot leaRning framework that integrates Adaptive spectrum
experts with Cross-sEt distribution calibration techniques. Theoretically, the
proposed approach enhances model generalization by adapting to both local
structural variations and cross-set distribution calibration. Empirically,
GRACE consistently outperforms state-of-the-art baselines across a wide range
of experimental settings. Our code can be found here.

### 4. [Self-Verifying Reflection Helps Transformers with CoT Reasoning](http://arxiv.org/pdf/2510.12157v1)

Authors: Zhongwei Yu, Wannian Xia, Xue Yan, Bo Xu, Haifeng Zhang, Yali Du, Jun Wang

Advanced large language models (LLMs) frequently reflect in reasoning
chain-of-thoughts (CoTs), where they self-verify the correctness of current
solutions and explore alternatives. However, given recent findings that LLMs
detect limited errors in CoTs, how reflection contributes to empirical
improvements remains unclear. To analyze this issue, in this paper, we present
a minimalistic reasoning framework to support basic self-verifying reflection
for small transformers without natural language, which ensures analytic clarity
and reduces the cost of comprehensive experiments. Theoretically, we prove that
self-verifying reflection guarantees improvements if verification errors are
properly bounded. Experimentally, we show that tiny transformers, with only a
few million parameters, benefit from self-verification in both training and
reflective execution, reaching remarkable LLM-level performance in integer
multiplication and Sudoku. Similar to LLM results, we find that reinforcement
learning (RL) improves in-distribution performance and incentivizes frequent
reflection for tiny transformers, yet RL mainly optimizes shallow statistical
patterns without faithfully reducing verification errors. In conclusion,
integrating generative transformers with discriminative verification inherently
facilitates CoT reasoning, regardless of scaling and natural language.

### 5. [Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory](http://arxiv.org/pdf/2510.12220v1)

Authors: Hanru Bai, Weiyang Ding, Difan Zou

Diffusion models have achieved impressive success in high-fidelity image
generation but suffer from slow sampling due to their inherently iterative
denoising process. While recent one-step methods accelerate inference by
learning direct noise-to-image mappings, they sacrifice the interpretability
and fine-grained control intrinsic to diffusion dynamics, key advantages that
enable applications like editable generation. To resolve this dichotomy, we
introduce \textbf{Hierarchical Koopman Diffusion}, a novel framework that
achieves both one-step sampling and interpretable generative trajectories.
Grounded in Koopman operator theory, our method lifts the nonlinear diffusion
dynamics into a latent space where evolution is governed by globally linear
operators, enabling closed-form trajectory solutions. This formulation not only
eliminates iterative sampling but also provides full access to intermediate
states, allowing manual intervention during generation. To model the
multi-scale nature of images, we design a hierarchical architecture that
disentangles generative dynamics across spatial resolutions via scale-specific
Koopman subspaces, capturing coarse-to-fine details systematically. We
empirically show that the Hierarchical Koopman Diffusion not only achieves
competitive one-step generation performance but also provides a principled
mechanism for interpreting and manipulating the generative process through
spectral analysis. Our framework bridges the gap between fast sampling and
interpretability in diffusion models, paving the way for explainable image
synthesis in generative modeling.

### 6. [Unveiling the Vulnerability of Graph-LLMs: An Interpretable Multi-Dimensional Adversarial Attack on TAGs](http://arxiv.org/pdf/2510.12233v1)

Authors: Bowen Fan, Zhilin Guo, Xunkai Li, Yihan Zhou, Bing Zhou, Zhenjun Li, Rong-Hua Li, Guoren Wang

Graph Neural Networks (GNNs) have become a pivotal framework for modeling
graph-structured data, enabling a wide range of applications from social
network analysis to molecular chemistry. By integrating large language models
(LLMs), text-attributed graphs (TAGs) enhance node representations with rich
textual semantics, significantly boosting the expressive power of graph-based
learning. However, this sophisticated synergy introduces critical
vulnerabilities, as Graph-LLMs are susceptible to adversarial attacks on both
their structural topology and textual attributes. Although specialized attack
methods have been designed for each of these aspects, no work has yet unified
them into a comprehensive approach. In this work, we propose the Interpretable
Multi-Dimensional Graph Attack (IMDGA), a novel human-centric adversarial
attack framework designed to orchestrate multi-level perturbations across both
graph structure and textual features. IMDGA utilizes three tightly integrated
modules to craft attacks that balance interpretability and impact, enabling a
deeper understanding of Graph-LLM vulnerabilities. Through rigorous theoretical
analysis and comprehensive empirical evaluations on diverse datasets and
architectures, IMDGA demonstrates superior interpretability, attack
effectiveness, stealthiness, and robustness compared to existing methods. By
exposing critical weaknesses in TAG representation learning, this work uncovers
a previously underexplored semantic dimension of vulnerability in Graph-LLMs,
offering valuable insights for improving their resilience. Our code and
resources are publicly available at
https://anonymous.4open.science/r/IMDGA-7289.

### 7. [Optimal Regularization for Performative Learning](http://arxiv.org/pdf/2510.12249v1)

Authors: Edwige Cyffers, Alireza Mirrokni, Marco Mondelli

In performative learning, the data distribution reacts to the deployed model
- for example, because strategic users adapt their features to game it - which
creates a more complex dynamic than in classical supervised learning. One
should thus not only optimize the model for the current data but also take into
account that the model might steer the distribution in a new direction, without
knowing the exact nature of the potential shift. We explore how regularization
can help cope with performative effects by studying its impact in
high-dimensional ridge regression. We show that, while performative effects
worsen the test risk in the population setting, they can be beneficial in the
over-parameterized regime where the number of features exceeds the number of
samples. We show that the optimal regularization scales with the overall
strength of the performative effect, making it possible to set the
regularization in anticipation of this effect. We illustrate this finding
through empirical evaluations of the optimal regularization parameter on both
synthetic and real-world datasets.

### 8. [FedMMKT:Co-Enhancing a Server Text-to-Image Model and Client Task Models in Multi-Modal Federated Learning](http://arxiv.org/pdf/2510.12254v1)

Authors: Ningxin He, Yang Liu, Wei Sun, Xiaozhou Ye, Ye Ouyang, Tiegang Gao, Zehui Zhang

Text-to-Image (T2I) models have demonstrated their versatility in a wide
range of applications. However, adaptation of T2I models to specialized tasks
is often limited by the availability of task-specific data due to privacy
concerns. On the other hand, harnessing the power of rich multimodal data from
modern mobile systems and IoT infrastructures presents a great opportunity.
This paper introduces Federated Multi-modal Knowledge Transfer (FedMMKT), a
novel framework that enables co-enhancement of a server T2I model and client
task-specific models using decentralized multimodal data without compromising
data privacy.

### 9. [Multi-Action Self-Improvement for Neural Combinatorial Optimization](http://arxiv.org/pdf/2510.12273v1)

Authors: Laurin Luttmann, Lin Xie

Self-improvement has emerged as a state-of-the-art paradigm in Neural
Combinatorial Optimization (NCO), where models iteratively refine their
policies by generating and imitating high-quality solutions. Despite strong
empirical performance, existing methods face key limitations. Training is
computationally expensive, as policy updates require sampling numerous
candidate solutions per instance to extract a single expert trajectory. More
fundamentally, these approaches fail to exploit the structure of combinatorial
problems involving the coordination of multiple agents, such as vehicles in
min-max routing or machines in scheduling. By supervising on single-action
trajectories, they fail to exploit agent-permutation symmetries, where distinct
sequences of actions yield identical solutions, hindering generalization and
the ability to learn coordinated behavior.
  We address these challenges by extending self-improvement to operate over
joint multi-agent actions. Our model architecture predicts complete agent-task
assignments jointly at each decision step. To explicitly leverage symmetries,
we employ a set-prediction loss, which supervises the policy on multiple expert
assignments for any given state. This approach enhances sample efficiency and
the model's ability to learn coordinated behavior. Furthermore, by generating
multi-agent actions in parallel, it drastically accelerates the solution
generation phase of the self-improvement loop. Empirically, we validate our
method on several combinatorial problems, demonstrating consistent improvements
in the quality of the final solution and a reduced generation latency compared
to standard self-improvement.

### 10. [Leveraging Teleconnections with Physics-Informed Graph Attention Networks for Long-Range Extreme Rainfall Forecasting in Thailand](http://arxiv.org/pdf/2510.12328v1)

Authors: Kiattikun Chobtham, Kanoksri Sarinnapakorn, Kritanai Torsri, Prattana Deeprasertkul, Jirawan Kamma

Accurate rainfall forecasting, particularly for extreme events, remains a
significant challenge in climatology and the Earth system. This paper presents
novel physics-informed Graph Neural Networks (GNNs) combined with extreme-value
analysis techniques to improve gauge-station rainfall predictions across
Thailand. The model leverages a graph-structured representation of gauge
stations to capture complex spatiotemporal patterns, and it offers
explainability through teleconnections. We preprocess relevant climate indices
that potentially influence regional rainfall. The proposed Graph Attention
Network with Long Short-Term Memory (Attention-LSTM) applies the attention
mechanism using initial edge features derived from simple
orographic-precipitation physics formulation. The embeddings are subsequently
processed by LSTM layers. To address extremes, we perform Peak-Over-Threshold
(POT) mapping using the novel Spatial Season-aware Generalized Pareto
Distribution (GPD) method, which overcomes limitations of traditional
machine-learning models. Experiments demonstrate that our method outperforms
well-established baselines across most regions, including areas prone to
extremes, and remains strongly competitive with the state of the art. Compared
with the operational forecasting system SEAS5, our real-world application
improves extreme-event prediction and offers a practical enhancement to produce
fine-resolution maps that support decision-making in long-term water
management.

### Neural and Evolutionary Computing

### 1. [SpikePool: Event-driven Spiking Transformer with Pooling Attention](http://arxiv.org/pdf/2510.12102v1)

Authors: Donghyun Lee, Alex Sima, Yuhang Li, Panos Stinis, Priyadarshini Panda

Building on the success of transformers, Spiking Neural Networks (SNNs) have
increasingly been integrated with transformer architectures, leading to spiking
transformers that demonstrate promising performance on event-based vision
tasks. However, despite these empirical successes, there remains limited
understanding of how spiking transformers fundamentally process event-based
data. Current approaches primarily focus on architectural modifications without
analyzing the underlying signal processing characteristics. In this work, we
analyze spiking transformers through the frequency spectrum domain and discover
that they behave as high-pass filters, contrasting with Vision Transformers
(ViTs) that act as low-pass filters. This frequency domain analysis reveals why
certain designs work well for event-based data, which contains valuable
high-frequency information but is also sparse and noisy. Based on this
observation, we propose SpikePool, which replaces spike-based self-attention
with max pooling attention, a low-pass filtering operation, to create a
selective band-pass filtering effect. This design preserves meaningful
high-frequency content while capturing critical features and suppressing noise,
achieving a better balance for event-based data processing. Our approach
demonstrates competitive results on event-based datasets for both
classification and object detection tasks while significantly reducing training
and inference time by up to 42.5% and 32.8%, respectively.

### 2. [General Fourier Feature Physics-Informed Extreme Learning Machine (GFF-PIELM) for High-Frequency PDEs](http://arxiv.org/pdf/2510.12293v1)

Authors: Fei Ren, Sifan Wang, Pei-Zhi Zhuang, Hai-Sui Yu, He Yang

Conventional physics-informed extreme learning machine (PIELM) often faces
challenges in solving partial differential equations (PDEs) involving
high-frequency and variable-frequency behaviors. To address these challenges,
we propose a general Fourier feature physics-informed extreme learning machine
(GFF-PIELM). We demonstrate that directly concatenating multiple Fourier
feature mappings (FFMs) and an extreme learning machine (ELM) network makes it
difficult to determine frequency-related hyperparameters. Fortunately, we find
an alternative to establish the GFF-PIELM in three main steps. First, we
integrate a variation of FFM into ELM as the Fourier-based activation function,
so there is still one hidden layer in the GFF-PIELM framework. Second, we
assign a set of frequency coefficients to the hidden neurons, which enables ELM
network to capture diverse frequency components of target solutions. Finally,
we develop an innovative, straightforward initialization method for these
hyperparameters by monitoring the distribution of ELM output weights. GFF-PIELM
not only retains the high accuracy, efficiency, and simplicity of the PIELM
framework but also inherits the ability of FFMs to effectively handle
high-frequency problems. We carry out five case studies with a total of ten
numerical examples to highlight the feasibility and validity of the proposed
GFF-PIELM, involving high frequency, variable frequency, multi-scale behaviour,
irregular boundary and inverse problems. Compared to conventional PIELM, the
GFF-PIELM approach significantly improves predictive accuracy without
additional cost in training time and architecture complexity. Our results
confirm that that PIELM can be extended to solve high-frequency and
variable-frequency PDEs with high accuracy, and our initialization strategy may
further inspire advances in other physics-informed machine learning (PIML)
frameworks.

### 3. [Tensor Logic: The Language of AI](http://arxiv.org/pdf/2510.12269v1)

Authors: Pedro Domingos

Progress in AI is hindered by the lack of a programming language with all the
requisite features. Libraries like PyTorch and TensorFlow provide automatic
differentiation and efficient GPU implementation, but are additions to Python,
which was never intended for AI. Their lack of support for automated reasoning
and knowledge acquisition has led to a long and costly series of hacky attempts
to tack them on. On the other hand, AI languages like LISP an Prolog lack
scalability and support for learning. This paper proposes tensor logic, a
language that solves these problems by unifying neural and symbolic AI at a
fundamental level. The sole construct in tensor logic is the tensor equation,
based on the observation that logical rules and Einstein summation are
essentially the same operation, and all else can be reduced to them. I show how
to elegantly implement key forms of neural, symbolic and statistical AI in
tensor logic, including transformers, formal reasoning, kernel machines and
graphical models. Most importantly, tensor logic makes new directions possible,
such as sound reasoning in embedding space. This combines the scalability and
learnability of neural networks with the reliability and transparency of
symbolic reasoning, and is potentially a basis for the wider adoption of AI.

### Networking and Internet Architecture

### 1. [GeoPipe: a Geo-distributed LLM Training Framework with enhanced Pipeline Parallelism in a Lossless RDMA-enabled Datacenter Optical Transport Network](http://arxiv.org/pdf/2510.12064v1)

Authors: Jun Dai, Xiaorun Wang, Kexiong Fang, Zheng Yang, Yuefeng Ji, Jiawei Zhang

The proliferation of Large Language Models (LLMs) with exponentially growing
parameters is making cross-data center (DC) training an inevitable trend.
However, viable strategies for extending single-DC training frameworks to
multi-DC environments remain underdeveloped. We experimentally demonstrate, for
the first time, a high-performance geo-distributed LLMs training framework
across multiple DCs interconnected by a lossless, remote direct memory access
(RDMA) enabled Datacenter Optical Transport Network (DC-OTN). An enhanced
pipeline parallelism scheme is implemented within the Ascend full-stack
environment of Huawei, which effectively eliminates the impact of cross-DC
communication overhead on training efficiency. The overlapped computation and
cross-DC communication is achieved with constraint cross-DC bandwidth and High
Bandwidth Memory (HBM), reducing computation bubble ratio by up to 78.91%.

### 2. [A Network Digital Twin of a 5G Private Network: Designing a Proof-of-Concept from Theory to Practice](http://arxiv.org/pdf/2510.12458v1)

Authors: Cristina Emilia Costa, Tatenda Horiro Zhou, Fabrizio Granelli

Network Digital Twins represent a key technology in future networks, expected
to provide the capability to perform accurate analysis and predictions about
the behaviour of 6G mobile networks. However, despite the availability of
several theoretical works on the subject, still very few examples of actual
implementations of Network Digital Twin are available. This paper provides a
detailed description about the characteristics of Network Digital Twin and
provides a practical example about real deployment of the technology. The
considered network infrastructure is a real 5G private network running in a
lab. The Network Digital Twin is built based on open source network emulation
software and is available to the community as open source. Measurements on both
the physical infrastructure and the related Digital Twin demonstrate a high
accuracy in reproducing the state and behavior of the actual 5G system.

### 3. [AMHRP: Adaptive Multi-Hop Routing Protocol to Improve Network Lifetime for Multi-Hop Wireless Body Area Network](http://arxiv.org/pdf/2510.12698v1)

Authors: Muhammad Mateen Yaqoob, Kulsoom Fatima, Shahab Shamshirband, Amir Mosavi, Waqar Khurshid

This paper presents a protocol for enhancement of life time of WBAN network
as well other protocol related issues such as throughput, path loss, and
residual energy. Bio-sensors are used for deployment on human body. Poisson
distribution and equilibrium model techniques have been used for attaining the
required results. Multi-hop network topology and random network node deployment
used to achieve minimum energy consumption and longer network lifetime.

### 4. [Over-Threshold Multiparty Private Set Intersection for Collaborative Network Intrusion Detection](http://arxiv.org/pdf/2510.12045v1)

Authors: Onur Eren Arpaci, Raouf Boutaba, Florian Kerschbaum

An important function of collaborative network intrusion detection is to
analyze the network logs of the collaborators for joint IP addresses. However,
sharing IP addresses in plain is sensitive and may be even subject to privacy
legislation as it is personally identifiable information. In this paper, we
present the privacy-preserving collection of IP addresses. We propose a single
collector, over-threshold private set intersection protocol. In this protocol
$N$ participants identify the IP addresses that appear in at least $t$
participant's sets without revealing any information about other IP addresses.
Using a novel hashing scheme, we reduce the computational complexity of the
previous state-of-the-art solution from $O(M(N \log{M}/t)^{2t})$ to
$O(t^2M\binom{N}{t})$, where $M$ denotes the dataset size. This reduction makes
it practically feasible to apply our protocol to real network logs. We test our
protocol using joint networks logs of multiple institutions. Additionally, we
present two deployment options: a collusion-safe deployment, which provides
stronger security guarantees at the cost of increased communication overhead,
and a non-interactive deployment, which assumes a non-colluding collector but
offers significantly lower communication costs and applicable to many use cases
of collaborative network intrusion detection similar to ours.

### 5. [Noisy Neighbor: Exploiting RDMA for Resource Exhaustion Attacks in Containerized Clouds](http://arxiv.org/pdf/2510.12629v1)

Authors: Gunwoo Kim, Taejune Park, Jinwoo Kim

In modern containerized cloud environments, the adoption of RDMA (Remote
Direct Memory Access) has expanded to reduce CPU overhead and enable
high-performance data exchange. Achieving this requires strong performance
isolation to ensure that one container's RDMA workload does not degrade the
performance of others, thereby maintaining critical security assurances.
However, existing isolation techniques are difficult to apply effectively due
to the complexity of microarchitectural resource management within RDMA NICs
(RNICs). This paper experimentally analyzes two types of resource exhaustion
attacks on NVIDIA BlueField-3: (i) state saturation attacks and (ii) pipeline
saturation attacks. Our results show that state saturation attacks can cause up
to a 93.9% loss in bandwidth, a 1,117x increase in latency, and a 115% rise in
cache misses for victim containers, while pipeline saturation attacks lead to
severe link-level congestion and significant amplification, where small verb
requests result in disproportionately high resource consumption. To mitigate
these threats and restore predictable security assurances, we propose HT-Verbs,
a threshold-driven framework based on real-time per-container RDMA verb
telemetry and adaptive resource classification that partitions RNIC resources
into hot, warm, and cold tiers and throttles abusive workloads without
requiring hardware modifications.

### 6. [CAMNet: Leveraging Cooperative Awareness Messages for Vehicle Trajectory Prediction](http://arxiv.org/pdf/2510.12703v1)

Authors: Mattia Grasselli, Angelo Porrello, Carlo Augusto Grazia

Autonomous driving remains a challenging task, particularly due to safety
concerns. Modern vehicles are typically equipped with expensive sensors such as
LiDAR, cameras, and radars to reduce the risk of accidents. However, these
sensors face inherent limitations: their field of view and line of sight can be
obstructed by other vehicles, thereby reducing situational awareness. In this
context, vehicle-to-vehicle communication plays a crucial role, as it enables
cars to share information and remain aware of each other even when sensors are
occluded. One way to achieve this is through the use of Cooperative Awareness
Messages (CAMs). In this paper, we investigate the use of CAM data for vehicle
trajectory prediction. Specifically, we design and train a neural network,
Cooperative Awareness Message-based Graph Neural Network (CAMNet), on a widely
used motion forecasting dataset. We then evaluate the model on a second dataset
that we created from scratch using Cooperative Awareness Messages, in order to
assess whether this type of data can be effectively exploited. Our approach
demonstrates promising results, showing that CAMs can indeed support vehicle
trajectory prediction. At the same time, we discuss several limitations of the
approach, which highlight opportunities for future research.

### 7. [Human-in-the-Loop Bandwidth Estimation for Quality of Experience Optimization in Real-Time Video Communication](http://arxiv.org/pdf/2510.12265v1)

Authors: Sami Khairy, Gabriel Mittag, Vishak Gopal, Ross Cutler

The quality of experience (QoE) delivered by video conferencing systems is
significantly influenced by accurately estimating the time-varying available
bandwidth between the sender and receiver. Bandwidth estimation for real-time
communications remains an open challenge due to rapidly evolving network
architectures, increasingly complex protocol stacks, and the difficulty of
defining QoE metrics that reliably improve user experience. In this work, we
propose a deployed, human-in-the-loop, data-driven framework for bandwidth
estimation to address these challenges. Our approach begins with training
objective QoE reward models derived from subjective user evaluations to measure
audio and video quality in real-time video conferencing systems. Subsequently,
we collect roughly $1$M network traces with objective QoE rewards from
real-world Microsoft Teams calls to curate a bandwidth estimation training
dataset. We then introduce a novel distributional offline reinforcement
learning (RL) algorithm to train a neural-network-based bandwidth estimator
aimed at improving QoE for users. Our real-world A/B test demonstrates that the
proposed approach reduces the subjective poor call ratio by $11.41\%$ compared
to the baseline bandwidth estimator. Furthermore, the proposed offline RL
algorithm is benchmarked on D4RL tasks to demonstrate its generalization beyond
bandwidth estimation.

### Robotics

### 1. [Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications](http://arxiv.org/pdf/2510.12215v1)

Authors: Chanwoo Kim, Jihwan Yoon, Hyeonseong Kim, Taemoon Jeong, Changwoo Yoo, Seungbeen Lee, Soohwan Byeon, Hoon Chung, Matthew Pan, Jean Oh, Kyungjae Lee, Sungjoon Choi

Mobile robot navigation in dynamic human environments requires policies that
balance adaptability to diverse behaviors with compliance to safety
constraints. We hypothesize that integrating data-driven rewards with
rule-based objectives enables navigation policies to achieve a more effective
balance of adaptability and safety. To this end, we develop a framework that
learns a density-based reward from positive and negative demonstrations and
augments it with rule-based objectives for obstacle avoidance and goal
reaching. A sampling-based lookahead controller produces supervisory actions
that are both safe and adaptive, which are subsequently distilled into a
compact student policy suitable for real-time operation with uncertainty
estimates. Experiments in synthetic and elevator co-boarding simulations show
consistent gains in success rate and time efficiency over baselines, and
real-world demonstrations with human participants confirm the practicality of
deployment. A video illustrating this work can be found on our project page
https://chanwookim971024.github.io/PioneeR/.

### 2. [Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model](http://arxiv.org/pdf/2510.12276v1)

Authors: Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li

Vision-language-action (VLA) models have recently shown strong potential in
enabling robots to follow language instructions and execute precise actions.
However, most VLAs are built upon vision-language models pretrained solely on
2D data, which lack accurate spatial awareness and hinder their ability to
operate in the 3D physical world. Existing solutions attempt to incorporate
explicit 3D sensor inputs such as depth maps or point clouds, but these
approaches face challenges due to sensor noise, hardware heterogeneity, and
incomplete depth coverage in existing datasets. Alternative methods that
estimate 3D cues from 2D images also suffer from the limited performance of
depth estimators.We propose Spatial Forcing (SF), a simple yet effective
alignment strategy that implicitly forces VLA models to develop spatial
comprehension capabilities without relying on explicit 3D inputs or depth
estimators. SF aligns intermediate visual embeddings of VLAs with geometric
representations produced by pretrained 3D foundation models. By enforcing
alignment at intermediate layers, SF guides VLAs to encode richer spatial
representations that enhance action precision.Extensive experiments in
simulation and real-world environments demonstrate that SF achieves
state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further
accelerates training by up to 3.8x and improves data efficiency across diverse
robotic tasks. Project page is at https://spatial-forcing.github.io/

### 3. [Shape-Aware Whole-Body Control for Continuum Robots with Application in Endoluminal Surgical Robotics](http://arxiv.org/pdf/2510.12332v1)

Authors: Mohammadreza Kasaei, Mostafa Ghobadi, Mohsen Khadem

This paper presents a shape-aware whole-body control framework for
tendon-driven continuum robots with direct application to endoluminal surgical
navigation. Endoluminal procedures, such as bronchoscopy, demand precise and
safe navigation through tortuous, patient-specific anatomy where conventional
tip-only control often leads to wall contact, tissue trauma, or failure to
reach distal targets. To address these challenges, our approach combines a
physics-informed backbone model with residual learning through an Augmented
Neural ODE, enabling accurate shape estimation and efficient Jacobian
computation. A sampling-based Model Predictive Path Integral (MPPI) controller
leverages this representation to jointly optimize tip tracking, backbone
conformance, and obstacle avoidance under actuation constraints. A task manager
further enhances adaptability by allowing real-time adjustment of objectives,
such as wall clearance or direct advancement, during tele-operation. Extensive
simulation studies demonstrate millimeter-level accuracy across diverse
scenarios, including trajectory tracking, dynamic obstacle avoidance, and
shape-constrained reaching. Real-robot experiments on a bronchoscopy phantom
validate the framework, showing improved lumen-following accuracy, reduced wall
contacts, and enhanced adaptability compared to joystick-only navigation and
existing baselines. These results highlight the potential of the proposed
framework to increase safety, reliability, and operator efficiency in minimally
invasive endoluminal surgery, with broader applicability to other confined and
safety-critical environments.

### 4. [Achieving Meaningful Collaboration: Worker-centered Design of a Physical Human-Robot Collaborative Blending Task](http://arxiv.org/pdf/2510.12340v1)

Authors: Nicky Mol, Luka Peternel, Alessandro Ianniello, Denis Zatyagov, Auke Nachenius, Stephan Balvert, J. Micah Prendergast, Sara Muscolo, Olger Siebinga, Eva Verhoef, Deborah Forster, David A. Abbink

The use of robots in industrial settings continues to grow, driven by the
need to address complex societal challenges such as labor shortages, aging
populations, and ever-increasing production demands. In this abstract, we
advocate for (and demonstrate) a transdisciplinary approach when considering
robotics in the workplace. Transdisciplinarity emphasizes the integration of
academic research with pragmatic expertise and embodied experiential knowledge,
that prioritize values such as worker wellbeing and job attractiveness. In the
following, we describe an ongoing multi-pronged effort to explore the potential
of collaborative robots in the context of airplane engine repair and
maintenance operations.

### 5. [PolygMap: A Perceptive Locomotion Framework for Humanoid Robot Stair Climbing](http://arxiv.org/pdf/2510.12346v1)

Authors: Bingquan Li, Ning Wang, Tianwei Zhang, Zhicheng He, Yucong Wu

Recently, biped robot walking technology has been significantly developed,
mainly in the context of a bland walking scheme. To emulate human walking,
robots need to step on the positions they see in unknown spaces accurately. In
this paper, we present PolyMap, a perception-based locomotion planning
framework for humanoid robots to climb stairs. Our core idea is to build a
real-time polygonal staircase plane semantic map, followed by a footstep planar
using these polygonal plane segments. These plane segmentation and visual
odometry are done by multi-sensor fusion(LiDAR, RGB-D camera and IMUs). The
proposed framework is deployed on a NVIDIA Orin, which performs 20-30 Hz
whole-body motion planning output. Both indoor and outdoor real-scene
experiments indicate that our method is efficient and robust for humanoid robot
stair climbing.

### 6. [Controlling Intent Expressiveness in Robot Motion with Diffusion Models](http://arxiv.org/pdf/2510.12370v1)

Authors: Wenli Shi, Clemence Grislain, Olivier Sigaud, Mohamed Chetouani

Legibility of robot motion is critical in human-robot interaction, as it
allows humans to quickly infer a robot's intended goal. Although traditional
trajectory generation methods typically prioritize efficiency, they often fail
to make the robot's intentions clear to humans. Meanwhile, existing approaches
to legible motion usually produce only a single "most legible" trajectory,
overlooking the need to modulate intent expressiveness in different contexts.
In this work, we propose a novel motion generation framework that enables
controllable legibility across the full spectrum, from highly legible to highly
ambiguous motions. We introduce a modeling approach based on an Information
Potential Field to assign continuous legibility scores to trajectories, and
build upon it with a two-stage diffusion framework that first generates paths
at specified legibility levels and then translates them into executable robot
actions. Experiments in both 2D and 3D reaching tasks demonstrate that our
approach produces diverse and controllable motions with varying degrees of
legibility, while achieving performance comparable to SOTA. Code and project
page: https://legibility-modulator.github.io.

### 7. [M3D-skin: Multi-material 3D-printed Tactile Sensor with Hierarchical Infill Structures for Pressure Sensing](http://arxiv.org/pdf/2510.12419v1)

Authors: Shunnosuke Yoshimura, Kento Kawaharazuka, Kei Okada

Tactile sensors have a wide range of applications, from utilization in
robotic grippers to human motion measurement. If tactile sensors could be
fabricated and integrated more easily, their applicability would further
expand. In this study, we propose a tactile sensor-M3D-skin-that can be easily
fabricated with high versatility by leveraging the infill patterns of a
multi-material fused deposition modeling (FDM) 3D printer as the sensing
principle. This method employs conductive and non-conductive flexible filaments
to create a hierarchical structure with a specific infill pattern. The flexible
hierarchical structure deforms under pressure, leading to a change in
electrical resistance, enabling the acquisition of tactile information. We
measure the changes in characteristics of the proposed tactile sensor caused by
modifications to the hierarchical structure. Additionally, we demonstrate the
fabrication and use of a multi-tile sensor. Furthermore, as applications, we
implement motion pattern measurement on the sole of a foot, integration with a
robotic hand, and tactile-based robotic operations. Through these experiments,
we validate the effectiveness of the proposed tactile sensor.

### 8. [A Task-Efficient Reinforcement Learning Task-Motion Planner for Safe Human-Robot Cooperation](http://arxiv.org/pdf/2510.12477v1)

Authors: Gaoyuan Liu, Joris de Winter, Kelly Merckaert, Denis Steckelmacher, Ann Nowe, Bram Vanderborght

In a Human-Robot Cooperation (HRC) environment, safety and efficiency are the
two core properties to evaluate robot performance. However, safety mechanisms
usually hinder task efficiency since human intervention will cause backup
motions and goal failures of the robot. Frequent motion replanning will
increase the computational load and the chance of failure. In this paper, we
present a hybrid Reinforcement Learning (RL) planning framework which is
comprised of an interactive motion planner and a RL task planner. The RL task
planner attempts to choose statistically safe and efficient task sequences
based on the feedback from the motion planner, while the motion planner keeps
the task execution process collision-free by detecting human arm motions and
deploying new paths when the previous path is not valid anymore. Intuitively,
the RL agent will learn to avoid dangerous tasks, while the motion planner
ensures that the chosen tasks are safe. The proposed framework is validated on
the cobot in both simulation and the real world, we compare the planner with
hard-coded task motion planning methods. The results show that our planning
framework can 1) react to uncertain human motions at both joint and task
levels; 2) reduce the times of repeating failed goal commands; 3) reduce the
total number of replanning requests.

### 9. [Automated Behavior Planning for Fruit Tree Pruning via Redundant Robot Manipulators: Addressing the Behavior Planning Challenge](http://arxiv.org/pdf/2510.12509v1)

Authors: Gaoyuan Liu, Bas Boom, Naftali Slob, Yuri Durodié, Ann Nowé, Bram Vanderborght

Pruning is an essential agricultural practice for orchards. Proper pruning
can promote healthier growth and optimize fruit production throughout the
orchard's lifespan. Robot manipulators have been developed as an automated
solution for this repetitive task, which typically requires seasonal labor with
specialized skills. While previous research has primarily focused on the
challenges of perception, the complexities of manipulation are often
overlooked. These challenges involve planning and control in both joint and
Cartesian spaces to guide the end-effector through intricate, obstructive
branches. Our work addresses the behavior planning challenge for a robotic
pruning system, which entails a multi-level planning problem in environments
with complex collisions. In this paper, we formulate the planning problem for a
high-dimensional robotic arm in a pruning scenario, investigate the system's
intrinsic redundancies, and propose a comprehensive pruning workflow that
integrates perception, modeling, and holistic planning. In our experiments, we
demonstrate that more comprehensive planning methods can significantly enhance
the performance of the robotic manipulator. Finally, we implement the proposed
workflow on a real-world robot. As a result, this work complements previous
efforts on robotic pruning and motivates future research and development in
planning for pruning applications.

### 10. [Maximal Adaptation, Minimal Guidance: Permissive Reactive Robot Task Planning with Humans in the Loop](http://arxiv.org/pdf/2510.12662v1)

Authors: Oz Gitelson, Satya Prakash Nayak, Ritam Raha, Anne-Kathrin Schmuck

We present a novel framework for human-robot \emph{logical} interaction that
enables robots to reliably satisfy (infinite horizon) temporal logic tasks
while effectively collaborating with humans who pursue independent and unknown
tasks. The framework combines two key capabilities: (i) \emph{maximal
adaptation} enables the robot to adjust its strategy \emph{online} to exploit
human behavior for cooperation whenever possible, and (ii) \emph{minimal
tunable feedback} enables the robot to request cooperation by the human online
only when necessary to guarantee progress. This balance minimizes human-robot
interference, preserves human autonomy, and ensures persistent robot task
satisfaction even under conflicting human goals. We validate the approach in a
real-world block-manipulation task with a Franka Emika Panda robotic arm and in
the Overcooked-AI benchmark, demonstrating that our method produces rich,
\emph{emergent} cooperative behaviors beyond the reach of existing approaches,
while maintaining strong formal guarantees.

### Software Engineering

### 1. [Towards Engineering Multi-Agent LLMs: A Protocol-Driven Approach](http://arxiv.org/pdf/2510.12120v1)

Authors: Zhenyu Mao, Jacky Keung, Fengji Zhang, Shuo Liu, Yifei Wang, Jialong Li

The increasing demand for software development has driven interest in
automating software engineering (SE) tasks using Large Language Models (LLMs).
Recent efforts extend LLMs into multi-agent systems (MAS) that emulate
collaborative development workflows, but these systems often fail due to three
core deficiencies: under-specification, coordination misalignment, and
inappropriate verification, arising from the absence of foundational SE
structuring principles. This paper introduces Software Engineering Multi-Agent
Protocol (SEMAP), a protocol-layer methodology that instantiates three core SE
design principles for multi-agent LLMs: (1) explicit behavioral contract
modeling, (2) structured messaging, and (3) lifecycle-guided execution with
verification, and is implemented atop Google's Agent-to-Agent (A2A)
infrastructure. Empirical evaluation using the Multi-Agent System Failure
Taxonomy (MAST) framework demonstrates that SEMAP effectively reduces failures
across different SE tasks. In code development, it achieves up to a 69.6%
reduction in total failures for function-level development and 56.7% for
deployment-level development. For vulnerability detection, SEMAP reduces
failure counts by up to 47.4% on Python tasks and 28.2% on C/C++ tasks.

### 2. [iCodeReviewer: Improving Secure Code Review with Mixture of Prompts](http://arxiv.org/pdf/2510.12186v1)

Authors: Yun Peng, Kisub Kim, Linghan Meng, Kui Liu

Code review is an essential process to ensure the quality of software that
identifies potential software issues at an early stage of software development.
Among all software issues, security issues are the most important to identify,
as they can easily lead to severe software crashes and service disruptions.
Recent research efforts have been devoted to automated approaches to reduce the
manual efforts required in the secure code review process. Despite the
progress, current automated approaches on secure code review, including static
analysis, deep learning models, and prompting approaches, still face the
challenges of limited precision and coverage, and a lack of comprehensive
evaluation.
  To mitigate these challenges, we propose iCodeReviewer, which is an automated
secure code review approach based on large language models (LLMs).
iCodeReviewer leverages a novel mixture-of-prompts architecture that
incorporates many prompt experts to improve the coverage of security issues.
Each prompt expert is a dynamic prompt pipeline to check the existence of a
specific security issue. iCodeReviewer also implements an effective routing
algorithm to activate only necessary prompt experts based on the code features
in the input program, reducing the false positives induced by LLM
hallucination. Experiment results in our internal dataset demonstrate the
effectiveness of iCodeReviewer in security issue identification and
localization with an F1 of 63.98%. The review comments generated by
iCodeReviewer also achieve a high acceptance rate up to 84% when it is deployed
in production environments.

### 3. [Show Your Title! A Scoping Review on Verbalization in Software Engineering with LLM-Assisted Screening](http://arxiv.org/pdf/2510.12294v1)

Authors: Gergő Balogh, Dávid Kószó, Homayoun Safarpour Motealegh Mahalegi, László Tóth, Bence Szakács, Áron Búcsú

Understanding how software developers think, make decisions, and behave
remains a key challenge in software engineering (SE). Verbalization techniques
(methods that capture spoken or written thought processes) offer a lightweight
and accessible way to study these cognitive aspects. This paper presents a
scoping review of research at the intersection of SE and psychology (PSY),
focusing on the use of verbal data. To make large-scale interdisciplinary
reviews feasible, we employed a large language model (LLM)-assisted screening
pipeline using GPT to assess the relevance of over 9,000 papers based solely on
titles. We addressed two questions: what themes emerge from
verbalization-related work in SE, and how effective are LLMs in supporting
interdisciplinary review processes? We validated GPT's outputs against human
reviewers and found high consistency, with a 13\% disagreement rate. Prominent
themes mainly were tied to the craft of SE, while more human-centered topics
were underrepresented. The data also suggests that SE frequently draws on PSY
methods, whereas the reverse is rare.

### 4. [The EmpathiSEr: Development and Validation of Software Engineering Oriented Empathy Scales](http://arxiv.org/pdf/2510.12546v1)

Authors: Hashini Gunatilake, John Grundy, Rashina Hoda, Ingo Mueller

Empathy plays a critical role in software engineering (SE), influencing
collaboration, communication, and user-centred design. Although SE research has
increasingly recognised empathy as a key human aspect, there remains no
validated instrument specifically designed to measure it within the unique
socio-technical contexts of SE. Existing generic empathy scales, while
well-established in psychology and healthcare, often rely on language,
scenarios, and assumptions that are not meaningful or interpretable for
software practitioners. These scales fail to account for the diverse,
role-specific, and domain-bound expressions of empathy in SE, such as
understanding a non-technical user's frustrations or another practitioner's
technical constraints, which differ substantially from empathy in clinical or
everyday contexts. To address this gap, we developed and validated two
domain-specific empathy scales: EmpathiSEr-P, assessing empathy among
practitioners, and EmpathiSEr-U, capturing practitioner empathy towards users.
Grounded in a practitioner-informed conceptual framework, the scales encompass
three dimensions of empathy: cognitive empathy, affective empathy, and empathic
responses. We followed a rigorous, multi-phase methodology, including expert
evaluation, cognitive interviews, and two practitioner surveys. The resulting
instruments represent the first psychometrically validated empathy scales
tailored to SE, offering researchers and practitioners a tool for assessing
empathy and designing empathy-enhancing interventions in software teams and
user interactions.

### 5. [Evaluating End-User Device Energy Models in Sustainability Reporting of Browser-Based Web Services](http://arxiv.org/pdf/2510.12566v1)

Authors: Maja H. Kirkeby, Timmie Lagermann

Sustainability reporting in web-based services increasingly relies on
simplified energy and carbon models such as the Danish Agency of Digital
Government's Digst framework and the United Kingdom-based DIMPACT model.
Although these models are widely adopted, their accuracy and precision remain
underexplored. This paper presents an empirical study evaluating how well such
models reflect actual energy consumption during realistic user interactions
with common website categories. Energy use was measured across shopping,
booking, navigation, and news services using predefined user flows executed on
four laptop platforms. The results show that the commonly applied
constant-power approximation (P * t) can diverge substantially from measured
energy, depending on website category, device type, and task characteristics.
The findings demonstrate that model deviations are systematic rather than
random and highlight the need for category-aware and device-reflective power
parameters in reproducible sustainability reporting frameworks.

### 6. [Do Large Language Models Respect Contracts? Evaluating and Enforcing Contract-Adherence in Code Generation](http://arxiv.org/pdf/2510.12047v1)

Authors: Soohan Lim, Joonghyuk Hahn, Hyunwoo Park, Sang-Ki Ko, Yo-Sub Han

Prevailing code generation benchmarks, such as HumanEval+ and MBPP+,
primarily evaluate large language models (LLMs) with pass@k on functional
correctness using well-formed inputs. However, they ignore a crucial aspect of
real-world software: adherence to contracts-the preconditions and validity
constraints that dictate how ill-formed inputs must be rejected. This critical
oversight means that existing benchmarks fail to measure, and models
consequently fail to generate, truly robust and reliable code snippets. We
introduce PACT, a program assessment and contract-adherence evaluation
framework, to bridge this gap. PACT is the first framework designed to
systematically evaluate and enhance contract-adherence in LLM-generated code
snippets alongside functional correctness. PACT's contributions are threefold:
First, it provides a comprehensive test-suite corpus focused on contract
violations, extending HumanEval+ and MBPP+. Second, it enables a systematic
analysis of code generation under varied prompting conditions. This analysis
demonstrates that augmenting prompts with contract-violating test cases
significantly enhance a model's ability to respect contracts compared to using
contract description alone. Finally, it introduces novel metrics to rigorously
quantify contract adherence in both test generation and code generation. By
revealing critical errors that conventional benchmarks overlook, PACT provides
the rigorous and interpretable metrics to evaluate the robustness of
LLM-generated code snippets in both functionality and contract-adherence.Our
code and data are available at https://github.com/suhanmen/PACT.

### 7. [Enhancing Neural Code Representation with Additional Context](http://arxiv.org/pdf/2510.12082v1)

Authors: Huy Nguyen, Christoph Treude, Patanamon Thongtanunam

Automated program comprehension underpins many software engineering tasks,
from code summarisation to clone detection. Recent deep learning models achieve
strong results but typically rely on source code alone, overlooking contextual
information such as version history or structural relationships. This limits
their ability to capture how code evolves and operates. We conduct an empirical
study on how enriching code representations with such contextual signals
affects neural model performance on key comprehension tasks. Two downstream
tasks, code clone detection and code summarisation, are evaluated using SeSaMe
(1,679 Java methods) and CodeSearchNet (63,259 methods). Five representative
models (CodeBERT, GraphCodeBERT, CodeT5, PLBART, ASTNN) are fine-tuned under
code-only and context-augmented settings. Results show that context generally
improves performance: version history consistently boosts clone detection
(e.g., CodeT5 +15.92% F1) and summarisation (e.g., GraphCodeBERT +5.56%
METEOR), while call-graph effects vary by model and task. Combining multiple
contexts yields further gains (up to +21.48% macro-F1). Human evaluation on 100
Java snippets confirms that context-augmented summaries are significantly
preferred for Accuracy and Content Adequacy (p <= 0.026; |delta| up to 0.55).
These findings highlight the potential of contextual signals to enhance code
comprehension and open new directions for optimising contextual encoding in
neural SE models.

### 8. [Diff-XYZ: A Benchmark for Evaluating Diff Understanding](http://arxiv.org/pdf/2510.12487v1)

Authors: Evgeniy Glukhov, Michele Conti, Egor Bogomolov, Yaroslav Golubev, Alexander Bezzubov

Reliable handling of code diffs is central to agents that edit and refactor
repositories at scale. We introduce Diff-XYZ, a compact benchmark for code-diff
understanding with three supervised tasks: apply (old code $+$ diff
$\rightarrow$ new code), anti-apply (new code $-$ diff $\rightarrow$ old code),
and diff generation (new code $-$ old code $\rightarrow$ diff). Instances in
the benchmark are triples $\langle \textit{old code}, \textit{new code},
\textit{diff} \rangle$ drawn from real commits in CommitPackFT, paired with
automatic metrics and a clear evaluation protocol. We use the benchmark to do a
focused empirical study of the unified diff format and run a cross-format
comparison of different diff representations. Our findings reveal that
different formats should be used depending on the use case and model size. For
example, representing diffs in search-replace format is good for larger models
in the diff generation scenario, yet not suited well for diff analysis and
smaller models. The Diff-XYZ benchmark is a reusable foundation for assessing
and improving diff handling in LLMs that can aid future development of diff
formats and models editing code. The dataset is published on HuggingFace Hub:
https://huggingface.co/datasets/JetBrains-Research/diff-xyz.

### 9. [Runtime Composition in Dynamic System of Systems: A Systematic Review of Challenges, Solutions, Tools, and Evaluation Methods](http://arxiv.org/pdf/2510.12616v1)

Authors: Muhammad Ashfaq, Ahmed R. Sadik, Teerath Das, Muhammad Waseem, Niko Makitalo, Tommi Mikkonen

Context: Modern Systems of Systems (SoSs) increasingly operate in dynamic
environments (e.g., smart cities, autonomous vehicles) where runtime
composition -- the on-the-fly discovery, integration, and coordination of
constituent systems (CSs)--is crucial for adaptability. Despite growing
interest, the literature lacks a cohesive synthesis of runtime composition in
dynamic SoSs. Objective: This study synthesizes research on runtime composition
in dynamic SoSs and identifies core challenges, solution strategies, supporting
tools, and evaluation methods. Methods: We conducted a Systematic Literature
Review (SLR), screening 1,774 studies published between 2019 and 2024 and
selecting 80 primary studies for thematic analysis (TA). Results: Challenges
fall into four categories: modeling and analysis, resilient operations, system
orchestration, and heterogeneity of CSs. Solutions span seven areas:
co-simulation and digital twins, semantic ontologies, integration frameworks,
adaptive architectures, middleware, formal methods, and AI-driven resilience.
Service-oriented frameworks for composition and integration dominate tooling,
while simulation platforms support evaluation. Interoperability across tools,
limited cross-toolchain workflows, and the absence of standardized benchmarks
remain key gaps. Evaluation approaches include simulation-based,
implementation-driven, and human-centered studies, which have been applied in
domains such as smart cities, healthcare, defense, and industrial automation.
Conclusions: The synthesis reveals tensions, including autonomy versus
coordination, the modeling-reality gap, and socio-technical integration. It
calls for standardized evaluation metrics, scalable decentralized
architectures, and cross-domain frameworks. The analysis aims to guide
researchers and practitioners in developing and implementing dynamically
composable SoSs.

### 10. [(R)evolution of Programming: Vibe Coding as a Post-Coding Paradigm](http://arxiv.org/pdf/2510.12364v1)

Authors: Kevin Krings, Nino S. Bohn, Thomas Ludwig

Recent advancements in generative artificial intelligence (GenAI),
particularly large language models, have introduced new possibilities for
software development practices. In our paper we investigate the emerging Vibe
Coding (VC) paradigm that emphasizes intuitive, affect-driven, and
improvisational interactions between developers and AI systems. Building upon
the discourse of End-User Development (EUD), we explore how VC diverges from
conventional programming approaches such as those supported by tools like
GitHub Copilot. Through five semi-structured interview sessions with ten
experienced software practitioners, we identify five thematic dimensions:
creativity, sustainability, the future of programming, collaboration, and
criticism. Our analysis conceptualizes VC within the metaphor of co-drifting,
contrasting it with the prevalent co-piloting perspective of AI-assisted
development. We argue that VC reconfigures the developers role, blurring
boundaries between professional and non-developers. While VC enables novel
forms of expression and rapid prototyping, it also introduces challenges
regarding reproducibility, scalability, and inclusivity. We propose that VC
represents a meaningful shift in programming culture, warranting further
investigation within human-computer interaction (HCI) and software engineering
research.

### Social and Information Networks

### 1. [MOUFLON: Multi-group Modularity-based Fairness-aware Community Detection](http://arxiv.org/pdf/2510.12348v1)

Authors: Georgios Panayiotou, Anand Mathew Muthukulam Simon, Matteo Magnani, Ece Calikus

In this paper, we propose MOUFLON, a fairness-aware, modularity-based
community detection method that allows adjusting the importance of partition
quality over fairness outcomes. MOUFLON uses a novel proportional balance
fairness metric, providing consistent and comparable fairness scores across
multi-group and imbalanced network settings. We evaluate our method under both
synthetic and real network datasets, focusing on performance and the trade-off
between modularity and fairness in the resulting communities, along with the
impact of network characteristics such as size, density, and group
distribution. As structural biases can lead to strong alignment between
demographic groups and network structure, we also examine scenarios with highly
clustered homogeneous groups, to understand how such structures influence
fairness outcomes. Our findings showcase the effects of incorporating fairness
constraints into modularity-based community detection, and highlight key
considerations for designing and benchmarking fairness-aware social network
analysis methods.

### 2. [Timeliness, Consensus, and Composition of the Crowd: Community Notes on X](http://arxiv.org/pdf/2510.12559v1)

Authors: Olesya Razuvayevskaya, Adel Tayebi, Ulrikke Dybdal Sørensen, Kalina Bontcheva, Richard Rogers

This study presents the first large-scale quantitative analysis of the
efficiency of X's Community Notes, a crowdsourced moderation system for
identifying and contextualising potentially misleading content. Drawing on over
1.8 million notes, we examine three key dimensions of crowdsourced moderation:
participation inequality, consensus formation, and timeliness. Despite the
system's goal of collective moderation, we find substantial concentration
effect, with the top 10% of contributors producing 58% of all notes (Gini
Coefficient = 0.68). The observed consensus is rare-only 11.5% of notes reach
agreement on publication, while 69% of posts receive conflicting
classifications. A majority of noted posts (approximately 68%) are annotated as
"Note Not Needed", reflecting the repurposing of the platform for debate rather
than moderation. We found that such posts are paradoxically more likely to
yield published notes (OR = 3.12). Temporal analyses show that the notes, on
average, are published 65.7 hours after the original post, with longer delays
significantly reducing the likelihood of consensus. These results portray
Community Notes as a stratified, deliberative system dominated by a small
contributor elite, marked by persistent dissensus, and constrained by
timeliness. We conclude this study by outlining design strategies to promote
equity, faster consensus, and epistemic reliability in community-based
moderation.

### 3. [Structure-aware Propagation Generation with Large Language Models for Fake News Detection](http://arxiv.org/pdf/2510.12125v1)

Authors: Mengyang Chen, Lingwei Wei, Wei Zhou, Songlin Hu

The spread of fake news on social media poses a serious threat to public
trust and societal stability. While propagation-based methods improve fake news
detection by modeling how information spreads, they often suffer from
incomplete propagation data. Recent work leverages large language models (LLMs)
to generate synthetic propagation, but typically overlooks the structural
patterns of real-world discussions. In this paper, we propose a novel
structure-aware synthetic propagation enhanced detection (StruSP) framework to
fully capture structural dynamics from real propagation. It enables LLMs to
generate realistic and structurally consistent propagation for better
detection. StruSP explicitly aligns synthetic propagation with real-world
propagation in both semantic and structural dimensions. Besides, we also design
a new bidirectional evolutionary propagation (BEP) learning strategy to better
align LLMs with structural patterns of propagation in the real world via
structure-aware hybrid sampling and masked propagation modeling objective.
Experiments on three public datasets demonstrate that StruSP significantly
improves fake news detection performance in various practical detection
scenarios. Further analysis indicates that BEP enables the LLM to generate more
realistic and diverse propagation semantically and structurally.

### 4. [CrisisNews: A Dataset Mapping Two Decades of News Articles on Online Problematic Behavior at Scale](http://arxiv.org/pdf/2510.12243v1)

Authors: Jeanne Choi, DongJae Kang, Yubin Choi, Juhoon Lee, Joseph Seering

As social media adoption grows globally, online problematic behaviors
increasingly escalate into large-scale crises, requiring an evolving set of
mitigation strategies. While HCI research often analyzes problematic behaviors
with pieces of user-generated content as the unit of analysis, less attention
has been given to event-focused perspectives that track how discrete events
evolve. In this paper, we examine 'social media crises': discrete patterns of
problematic behaviors originating and evolving within social media that cause
larger-scale harms. Using global news coverage, we present a dataset of 93,250
news articles covering social media-endemic crises from the past 20 years. We
analyze a representative subset to classify stakeholder roles, behavior types,
and outcomes, uncovering patterns that inform more nuanced classification of
social media crises beyond content-based descriptions. By adopting a wider
perspective, this research seeks to inform the design of safer platforms,
enabling proactive measures to mitigate crises and foster more trustworthy
online environments.

### 5. [The Diameter of (Threshold) Geometric Inhomogeneous Random Graphs](http://arxiv.org/pdf/2510.12543v1)

Authors: Zylan Benjert, Kostas Lakis, Johannes Lengler, Raghu Raman Ravi

We prove that the diameter of threshold (zero temperature) Geometric
Inhomogeneous Random Graphs (GIRG) is $\Theta(\log n)$. This has strong
implications for the runtime of many distributed protocols on those graphs,
which often have runtimes bounded as a function of the diameter.
  The GIRG model exhibits many properties empirically found in real-world
networks, and the runtime of various practical algorithms has empirically been
found to scale in the same way for GIRG and for real-world networks, in
particular related to computing distances, diameter, clustering, cliques and
chromatic numbers. Thus the GIRG model is a promising candidate for deriving
insight about the performance of algorithms in real-world instances.
  The diameter was previously only known in the one-dimensional case, and the
proof relied very heavily on dimension one. Our proof employs a similar
Peierls-type argument alongside a novel renormalization scheme. Moreover,
instead of using topological arguments (which become complicated in high
dimensions) in establishing the connectivity of certain boundaries, we employ
some comparatively recent and clearer graph-theoretic machinery. The lower
bound is proven via a simple ad-hoc construction.

### 6. [Inclusive Fitness as a Key Step Towards More Advanced Social Behaviors in Multi-Agent Reinforcement Learning Settings](http://arxiv.org/pdf/2510.12555v1)

Authors: Andries Rosseau, Raphaël Avalos, Ann Nowé

The competitive and cooperative forces of natural selection have driven the
evolution of intelligence for millions of years, culminating in nature's vast
biodiversity and the complexity of human minds. Inspired by this process, we
propose a novel multi-agent reinforcement learning framework where each agent
is assigned a genotype and where reward functions are modelled after the
concept of inclusive fitness. An agent's genetic material may be shared with
other agents, and our inclusive reward function naturally accounts for this. We
study the resulting social dynamics in two types of network games with
prisoner's dilemmas and find that our results align with well-established
principles from biology, such as Hamilton's rule. Furthermore, we outline how
this framework can extend to more open-ended environments with spatial and
temporal structure, finite resources, and evolving populations. We hypothesize
the emergence of an arms race of strategies, where each new strategy is a
gradual improvement over earlier adaptations of other agents, effectively
producing a multi-agent autocurriculum analogous to biological evolution. In
contrast to the binary team-based structures prevalent in earlier research, our
gene-based reward structure introduces a spectrum of cooperation ranging from
full adversity to full cooperativeness based on genetic similarity, enabling
unique non team-based social dynamics. For example, one agent having a mutual
cooperative relationship with two other agents, while the two other agents
behave adversarially towards each other. We argue that incorporating inclusive
fitness in agents provides a foundation for the emergence of more strategically
advanced and socially intelligent agents.

### Systems and Control

### 1. [Sleepy Chauffeur Detection and Alert Techniques for Road Safety](http://arxiv.org/pdf/2510.12205v1)

Authors: Himel Ghosh, Sayak Chatterjee, Antik Ganguly, Shreetama Karmakar, Koushik Sarkar

The most startling of the contemporary problems is the sleepiness of
chauffeur which causes lots of car accidents. Prevention of those impending
accidents by detecting and alerting the sleepy chauffeur is vital, otherwise
that would lead to loss of lives and various traumas along with severe
injuries. The slumber or sleep may be caused by huge stress, pressure,
relentless work load or alcoholism, for which sleep deprivation occurs and the
chauffeur while driving gets drowsy. So far, considerable amount of systems has
been developed to detect drowsiness of drivers, most of which mainly depend on
image processing algorithms using cameras. Some of them also incorporate
artificial intelligence and machine learning based algorithms. This paper
presents a review of the existing systems and also proposes an easy and cheap
system using sensors and Arduino, capable of detecting sleepiness and generates
siren alarm and send alert message to take precautionary measures.

### 2. [Empowering Prosumers: Incentive Design for Local Electricity Markets Under Generalized Uncertainty and Grid Constraints](http://arxiv.org/pdf/2510.12318v1)

Authors: Pål Forr Austnes, Matthieu Jacobs, Lu Wang, Mario Paolone

Since the 1990s, widespread introduction of central (wholesale) electricity
markets has been seen across multiple continents, driven by the search for
efficient operation of the power grid through competition. The increase of
renewables has made significant impacts both on central electricity markets and
distribution-level grids as renewable power generation is often connected to
the latter. These stochastic renewable technologies have both advantages and
disadvantages. On one hand they offer very low marginal cost and carbon
emissions, while on the other hand, their output is uncertain, requiring
flexible backup power with high marginal cost. Flexibility from end-prosumers
or smaller market participants is therefore seen as a key enabler of
large-scale integration of renewables. However, current central electricity
markets do not directly include uncertainty into the market clearing and do not
account for physical constraints of distribution grids. In this paper we
propose a local electricity market framework based on probabilistic locational
marginal pricing, effectively accounting for uncertainties in production,
consumption and grid variables. The model includes a representation of the grid
using the lindistflow equations and accounts for the propagation of uncertainty
using general Polynomial Chaos (gPC). A two-stage convex model is proposed; in
the day-ahead stage, probability distributions of prices are calculated for
every timestep, where the expected values represent the day-ahead (spot)
prices. In the real-time stage, uncertainties are realized (measured) and a
trivial calculation reveals the real-time price. Through four instructive
case-studies we highlight the effectiveness of the method to incentivize
end-prosumers' participation in the market, while ensuring that their behavior
does not have an adverse impact on the operation of the grid.

### 3. [Physics-Informed Reinforcement Learning for Large-Scale EV Smart Charging Considering Distribution Network Voltage Constraints](http://arxiv.org/pdf/2510.12335v1)

Authors: Stavros Orfanoudakis, Frans Oliehoek, Peter Palesnky, Pedro P. Vergara

Electric Vehicles (EVs) offer substantial flexibility for grid services, yet
large-scale, uncoordinated charging can threaten voltage stability in
distribution networks. Existing Reinforcement Learning (RL) approaches for
smart charging often disregard physical grid constraints or have limited
performance for complex large-scale tasks, limiting their scalability and
real-world applicability. This paper introduces a physics-informed (PI) RL
algorithm that integrates a differentiable power flow model and voltage-based
reward design into the Twin Delayed Deep Deterministic Policy Gradient (TD3)
algorithm, enabling EVs to deliver real-time voltage support while meeting user
demands. The resulting PI-TD3 algorithm achieves faster convergence, improved
sample efficiency, and reliable voltage magnitude regulation under uncertain
and overloaded conditions. Benchmarks on the IEEE 34-bus and 123-bus networks
show that the proposed PI-TD3 outperforms both model-free RL and
optimization-based baselines in grid constraint management, user satisfaction,
and economic metrics, even as the system scales to hundreds of EVs. These
advances enable robust, scalable, and practical EV charging strategies that
enhance grid resilience and support distribution networks operation.

### 4. [Privacy-Preserving Distributed Estimation with Limited Data Rate](http://arxiv.org/pdf/2510.12549v1)

Authors: Jieming Ke, Jimin Wang, Ji-Feng Zhang

This paper focuses on the privacy-preserving distributed estimation problem
with a limited data rate, where the observations are the sensitive information.
Specifically, a binary-valued quantizer-based privacy-preserving distributed
estimation algorithm is developed, which improves the algorithm's
privacy-preserving capability and simultaneously reduces the communication
costs. The algorithm's privacy-preserving capability, measured by the Fisher
information matrix, is dynamically enhanced over time. Notably, the Fisher
information matrix of the output signals with respect to the sensitive
information converges to zero at a polynomial rate, and the improvement in
privacy brought by the quantizers is quantitatively characterized as a
multiplicative effect. Regarding the communication costs, each sensor transmits
only 1 bit of information to its neighbours at each time step. Additionally,
the assumption on the negligible quantization error for real-valued messages is
not required. While achieving the requirements of privacy preservation and
reducing communication costs, the algorithm ensures that its estimates converge
almost surely to the true value of the unknown parameter by establishing a
co-design guideline for the time-varying privacy noises and step-sizes. A
polynomial almost sure convergence rate is obtained, and then the trade-off
between privacy and convergence rate is established. Numerical examples
demonstrate the main results.

### 5. [Enhancing Robust Multi-Market Participation of Renewable-Based VPPs through Flexible Resources](http://arxiv.org/pdf/2510.12589v1)

Authors: Hadi Nemati, Álvaro Ortega, Pedro Sánchez-Martín, Lukas Sigrist, Luis Rouco, Ignacio Egido

In the transition toward a sustainable power system, renewable-based Virtual
Power Plants (RVPPs) have emerged as a promising solution to the challenges of
integrating renewable energy sources into electricity markets. Their viability,
however, depends on effective market participation strategies and the ability
to manage uncertainties while leveraging flexible resources. This paper
analyzes the impact of different flexible resources - such as concentrated
solar power plants, hydro plants, biomass plants, and flexible demand - on the
participation of RVPPs in energy and reserve markets. Multiple sources of
uncertainty in generation, consumption, and electricity prices are addressed
using a two-stage robust optimization approach. The contribution of different
technologies to RVPP profitability is evaluated through a marginal contribution
method, ensuring fair allocation of profits among them according to their
actual role in energy and reserve provision across markets. Simulations for an
RVPP in southern Spain demonstrate how strategic decisions and the availability
of flexible resources influence viability, market participation, and unit
scheduling.

### 6. [Hybrid Terrain-Aware Path Planning: Integrating VD--RRT\(^{*}\) Exploration and VD--D\(^{*}\) Lite Repair](http://arxiv.org/pdf/2510.12169v1)

Authors: Akshay Naik, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu

Autonomous ground vehicles operating off-road must plan curvature-feasible
paths while accounting for spatially varying soil strength and slope hazards in
real time. We present a continuous state--cost metric that combines a Bekker
pressure--sinkage model with elevation-derived slope and attitude penalties.
The resulting terrain cost field is analytic, bounded, and monotonic in soil
modulus and slope, ensuring well-posed discretization and stable updates under
sensor noise. This metric is evaluated on a lattice with exact steering
primitives: Dubins and Reeds--Shepp motions for differential drive and
time-parameterized bicycle arcs for Ackermann steering. Global exploration is
performed using Vehicle-Dynamics RRT\(^{*}\), while local repair is managed by
Vehicle-Dynamics D\(^{*}\) Lite, enabling millisecond-scale replanning without
heuristic smoothing. By separating the terrain--vehicle model from the planner,
the framework provides a reusable basis for deterministic, sampling-based, or
learning-driven planning in deformable terrain. Hardware trials on an off-road
platform demonstrate real-time navigation across soft soil and slope
transitions, supporting reliable autonomy in unstructured environments.

### 7. [Ultrafast Grid Impedance Identification in $dq$-Asymmetric Three-Phase Power Systems](http://arxiv.org/pdf/2510.12338v1)

Authors: Mohamed Abdalmoaty, Verena Häberle, Xiuqiang He, Florian Dörfler

We propose a non-parametric frequency-domain method to identify small-signal
$dq$-asymmetric grid impedances, over a wide frequency band, using
grid-connected converters. Existing identification methods are faced with
significant trade-offs: e.g., passive approaches rely on ambient harmonics and
rare grid events and thus can only provide estimates at a few frequencies,
while many active approaches that intentionally perturb grid operation require
long time series measurement and specialized equipment. Although active
time-domain methods reduce the measurement time, they either make crude
simplifying assumptions or require laborious model order tuning. Our approach
effectively addresses these challenges: it does not require specialized
excitation signals or hardware and achieves ultrafast ($<1$ s) identification,
drastically reducing measurement time. Being non-parametric, our approach also
makes no assumptions on the grid structure. A detailed electromagnetic
transient simulation is used to validate the method and demonstrate its clear
superiority over existing alternatives.

### 8. [A Unidirectionally Connected FAS Approach for 6-DOF Quadrotor Control](http://arxiv.org/pdf/2510.12360v1)

Authors: Weijie Ren, Haowen Liu, Guang-Ren Duan

This paper proposes a unidirectionally connected fully actuated system
(UC-FAS) approach for the sub-stabilization and tracking control of 6-DOF
quadrotors, tackling limitations both in state-space and FAS framework to some
extent. The framework systematically converts underactuated quadrotor dynamics
into a UC-FAS model, unifying the existing different FAS transformation ways.
By eliminating estimation of the high-order derivatives of control inputs, a
drawback of current methods, the UC-FAS model simplifies controller design and
enables direct eigenstructure assignment for closed-loop dynamics. Simulations
demonstrate precise 6-DOF tracking performance. This work bridges theoretical
FAS approach advancements with practical implementation needs, offering a
standardized paradigm for nonlinear quadrotor control.

### 9. [Pooling Probabilistic Forecasts for Cooperative Wind Power Offering](http://arxiv.org/pdf/2510.12382v1)

Authors: Honglin Wen, Pierre Pinson

Wind power producers can benefit from forming coalitions to participate
cooperatively in electricity markets. To support such collaboration, various
profit allocation rules rooted in cooperative game theory have been proposed.
However, existing approaches overlook the lack of coherence among producers
regarding forecast information, which may lead to ambiguity in offering and
allocations. In this paper, we introduce a ``reconcile-then-optimize''
framework for cooperative market offerings. This framework first aligns the
individual forecasts into a coherent joint forecast before determining market
offers. With such forecasts, we formulate and solve a two-stage stochastic
programming problem to derive both the aggregate offer and the corresponding
scenario-based dual values for each trading hour. Based on these dual values,
we construct a profit allocation rule that is budget-balanced and stable.
Finally, we validate the proposed method through empirical case studies,
demonstrating its practical effectiveness and theoretical soundness.

### 10. [High-Parallel FPGA-Based Discrete Simulated Bifurcation for Large-Scale Optimization](http://arxiv.org/pdf/2510.12407v1)

Authors: Fabrizio Orlando, Deborah Volpe, Giacomo Orlandi, Mariagrazia Graziano, Fabrizio Riente, Marco Vacca

Combinatorial Optimization (CO) problems exhibit exponential complexity,
making their resolution challenging. Simulated Adiabatic Bifurcation (aSB) is a
quantum-inspired algorithm to obtain approximate solutions to largescale CO
problems written in the Ising form. It explores the solution space by emulating
the adiabatic evolution of a network of Kerr-nonlinear parametric oscillators
(KPOs), where each oscillator represents a variable in the problem. The optimal
solution corresponds to the ground state of this system. A key advantage of
this approach is the possibility of updating multiple variables simultaneously,
making it particularly suited for hardware implementation. To enhance solution
quality and convergence speed, variations of the algorithm have been proposed
in the literature, including ballistic (bSB), discrete (dSB), and thermal
(HbSB) versions. In this work, we have comprehensively analyzed dSB, bSB, and
HbSB using dedicated software models, evaluating the feasibility of using a
fixed-point representation for hardware implementation. We then present an
opensource hardware architecture implementing the dSB algorithm for
Field-Programmable Gate Arrays (FPGAs). The design allows users to adjust the
degree of algorithmic parallelization based on their specific requirements. A
proof-of-concept implementation that solves 256-variable problems was achieved
on an AMD Kria KV260 SoM, a low-tier FPGA, validated using well-known max-cut
and knapsack problems.

### Machine Learning (Statistics Category)

### 1. [Mamaba Can Learn Low-Dimensional Targets In-Context via Test-Time Feature Learning](http://arxiv.org/pdf/2510.12026v1)

Authors: Junsoo Oh, Wei Huang, Taiji Suzuki

Mamba, a recently proposed linear-time sequence model, has attracted
significant attention for its computational efficiency and strong empirical
performance. However, a rigorous theoretical understanding of its underlying
mechanisms remains limited. In this work, we provide a theoretical analysis of
Mamba's in-context learning (ICL) capability by focusing on tasks defined by
low-dimensional nonlinear target functions. Specifically, we study in-context
learning of a single-index model $y \approx g_*(\langle \boldsymbol{\beta},
\boldsymbol{x} \rangle)$, which depends on only a single relevant direction
$\boldsymbol{\beta}$, referred to as feature. We prove that Mamba, pretrained
by gradient-based methods, can achieve efficient ICL via test-time feature
learning, extracting the relevant direction directly from context examples.
Consequently, we establish a test-time sample complexity that improves upon
linear Transformers -- analyzed to behave like kernel methods -- and is
comparable to nonlinear Transformers, which have been shown to surpass the
Correlational Statistical Query (CSQ) lower bound and achieve near
information-theoretically optimal rate in previous works. Our analysis reveals
the crucial role of the nonlinear gating mechanism in Mamba for feature
extraction, highlighting it as the fundamental driver behind Mamba's ability to
achieve both computational efficiency and high performance.

### 2. [Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory](http://arxiv.org/pdf/2510.12077v1)

Authors: Einar Urdshals, Edmund Lau, Jesse Hoogland, Stan van Wingerden, Daniel Murfet

We study neural network compressibility by using singular learning theory to
extend the minimum description length (MDL) principle to singular models like
neural networks. Through extensive experiments on the Pythia suite with
quantization, factorization, and other compression techniques, we find that
complexity estimates based on the local learning coefficient (LLC) are closely,
and in some cases, linearly correlated with compressibility. Our results
provide a path toward rigorously evaluating the limits of model compression.

### 3. [Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality](http://arxiv.org/pdf/2510.12152v1)

Authors: Chaiwon Kim, Jongyeong Lee, Min-hwan Oh

We study the decoupled multi-armed bandit (MAB) problem, where the learner
selects one arm for exploration and one arm for exploitation in each round. The
loss of the explored arm is observed but not counted, while the loss of the
exploited arm is incurred without being observed. We propose a policy within
the Follow-the-Perturbed-Leader (FTPL) framework using Pareto perturbations.
Our policy achieves (near-)optimal regret regardless of the environment, i.e.,
Best-of-Both-Worlds (BOBW): constant regret in the stochastic regime, improving
upon the optimal bound of the standard MABs, and minimax optimal regret in the
adversarial regime. Moreover, the practicality of our policy stems from
avoiding both the convex optimization step required by the previous BOBW
policy, Decoupled-Tsallis-INF (Rouyer & Seldin, 2020), and the resampling step
that is typically necessary in FTPL. Consequently, it achieves substantial
computational improvement, about $20$ times faster than Decoupled-Tsallis-INF,
while also demonstrating better empirical performance in both regimes. Finally,
we empirically show that our approach outperforms a pure exploration policy,
and that naively combining a pure exploration with a standard exploitation
policy is suboptimal.

### 4. [Sliding-Window Signatures for Time Series: Application to Electricity Demand Forecasting](http://arxiv.org/pdf/2510.12337v1)

Authors: Nina Drobac, Margaux Brégère, Joseph de Vilmarest, Olivier Wintenberger

Nonlinear and delayed effects of covariates often render time series
forecasting challenging. To this end, we propose a novel forecasting framework
based on ridge regression with signature features calculated on sliding
windows. These features capture complex temporal dynamics without relying on
learned or hand-crafted representations. Focusing on the discrete-time setting,
we establish theoretical guarantees, namely universality of approximation and
stationarity of signatures. We introduce an efficient sequential algorithm for
computing signatures on sliding windows. The method is evaluated on both
synthetic and real electricity demand data. Results show that signature
features effectively encode temporal and nonlinear dependencies, yielding
accurate forecasts competitive with those based on expert knowledge.

### 5. [Geopolitics, Geoeconomics and Risk:A Machine Learning Approach](http://arxiv.org/pdf/2510.12416v1)

Authors: Alvaro Ortiz, Tomasa Rodrigo

We introduce a novel high-frequency daily panel dataset of both markets and
news-based indicators -- including Geopolitical Risk, Economic Policy
Uncertainty, Trade Policy Uncertainty, and Political Sentiment -- for 42
countries across both emerging and developed markets. Using this dataset, we
study how sentiment dynamics shape sovereign risk, measured by Credit Default
Swap (CDS) spreads, and evaluate their forecasting value relative to
traditional drivers such as global monetary policy and market volatility. Our
horse-race analysis of forecasting models demonstrates that incorporating
news-based indicators significantly enhances predictive accuracy and enriches
the analysis, with non-linear machine learning methods -- particularly Random
Forests -- delivering the largest gains. Our analysis reveals that while global
financial variables remain the dominant drivers of sovereign risk, geopolitical
risk and economic policy uncertainty also play a meaningful role. Crucially,
their effects are amplified through non-linear interactions with global
financial conditions. Finally, we document pronounced regional heterogeneity,
as certain asset classes and emerging markets exhibit heightened sensitivity to
shocks in policy rates, global financial volatility, and geopolitical risk.

### 6. [CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling](http://arxiv.org/pdf/2510.12489v1)

Authors: Beibu Li, Qichao Shentu, Yang Shu, Hui Zhang, Ming Li, Ning Jin, Bin Yang, Chenjuan Guo

Time series anomaly detection plays a crucial role in a wide range of
real-world applications. Given that time series data can exhibit different
patterns at different sampling granularities, multi-scale modeling has proven
beneficial for uncovering latent anomaly patterns that may not be apparent at a
single scale. However, existing methods often model multi-scale information
independently or rely on simple feature fusion strategies, neglecting the
dynamic changes in cross-scale associations that occur during anomalies.
Moreover, most approaches perform multi-scale modeling based on fixed sliding
windows, which limits their ability to capture comprehensive contextual
information. In this work, we propose CrossAD, a novel framework for time
series Anomaly Detection that takes Cross-scale associations and Cross-window
modeling into account. We propose a cross-scale reconstruction that
reconstructs fine-grained series from coarser series, explicitly capturing
cross-scale associations. Furthermore, we design a query library and
incorporate global multi-scale context to overcome the limitations imposed by
fixed window sizes. Extensive experiments conducted on multiple real-world
datasets using nine evaluation metrics validate the effectiveness of CrossAD,
demonstrating state-of-the-art performance in anomaly detection.

### 7. [Universal Adaptive Environment Discovery](http://arxiv.org/pdf/2510.12547v1)

Authors: Madi Matymov, Ba-Hien Tran, Maurizio Filippone

An open problem in Machine Learning is how to avoid models to exploit
spurious correlations in the data; a famous example is the background-label
shortcut in the Waterbirds dataset. A common remedy is to train a model across
multiple environments; in the Waterbirds dataset, this corresponds to training
by randomizing the background. However, selecting the right environments is a
challenging problem, given that these are rarely known a priori. We propose
Universal Adaptive Environment Discovery (UAED), a unified framework that
learns a distribution over data transformations that instantiate environments,
and optimizes any robust objective averaged over this learned distribution.
UAED yields adaptive variants of IRM, REx, GroupDRO, and CORAL without
predefined groups or manual environment design. We provide a theoretical
analysis by providing PAC-Bayes bounds and by showing robustness to test
environment distributions under standard conditions. Empirically, UAED
discovers interpretable environment distributions and improves worst-case
accuracy on standard benchmarks, while remaining competitive on mean accuracy.
Our results indicate that making environments adaptive is a practical route to
out-of-distribution generalization.

### 8. [Learning Latent Energy-Based Models via Interacting Particle Langevin Dynamics](http://arxiv.org/pdf/2510.12311v1)

Authors: Joanna Marks, Tim Y. J. Wang, O. Deniz Akyildiz

We develop interacting particle algorithms for learning latent variable
models with energy-based priors. To do so, we leverage recent developments in
particle-based methods for solving maximum marginal likelihood estimation
(MMLE) problems. Specifically, we provide a continuous-time framework for
learning latent energy-based models, by defining stochastic differential
equations (SDEs) that provably solve the MMLE problem. We obtain a practical
algorithm as a discretisation of these SDEs and provide theoretical guarantees
for the convergence of the proposed algorithm. Finally, we demonstrate the
empirical effectiveness of our method on synthetic and image datasets.

### 9. [Cautious Weight Decay](http://arxiv.org/pdf/2510.12402v1)

Authors: Lizhang Chen, Jonathan Li, Kaizhao Liang, Baiyu Su, Cong Xie, Nuo Wang Pierse, Chen Liang, Ni Lao, Qiang Liu

We introduce Cautious Weight Decay (CWD), a one-line, optimizer-agnostic
modification that applies weight decay only to parameter coordinates whose
signs align with the optimizer update. Unlike standard decoupled decay, which
implicitly optimizes a regularized or constrained objective, CWD preserves the
original loss and admits a bilevel interpretation: it induces sliding-mode
behavior upon reaching the stationary manifold, allowing it to search for
locally Pareto-optimal stationary points of the unmodified objective. In
practice, CWD is a drop-in change for optimizers such as AdamW, Lion, and Muon,
requiring no new hyperparameters or additional tuning. For language model
pre-training and ImageNet classification, CWD consistently improves final loss
and accuracy at million- to billion-parameter scales.

### 10. [The Robustness of Differentiable Causal Discovery in Misspecified Scenarios](http://arxiv.org/pdf/2510.12503v1)

Authors: Huiyang Yi, Yanyan He, Duxin Chen, Mingyu Kang, He Wang, Wenwu Yu

Causal discovery aims to learn causal relationships between variables from
targeted data, making it a fundamental task in machine learning. However,
causal discovery algorithms often rely on unverifiable causal assumptions,
which are usually difficult to satisfy in real-world data, thereby limiting the
broad application of causal discovery in practical scenarios. Inspired by these
considerations, this work extensively benchmarks the empirical performance of
various mainstream causal discovery algorithms, which assume i.i.d. data, under
eight model assumption violations. Our experimental results show that
differentiable causal discovery methods exhibit robustness under the metrics of
Structural Hamming Distance and Structural Intervention Distance of the
inferred graphs in commonly used challenging scenarios, except for scale
variation. We also provide the theoretical explanations for the performance of
differentiable causal discovery methods. Finally, our work aims to
comprehensively benchmark the performance of recent differentiable causal
discovery methods under model assumption violations, and provide the standard
for reasonable evaluation of causal discovery, as well as to further promote
its application in real-world scenarios.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-15 PST.

### 1. [Image-based obstacle detection methods for the safe navigation of industrial unmanned aerial vehicles](https://www.nature.com/articles/s41598-025-19904-9)

Authors: Liang Wang et al.

### 2. [Computer vision assisted deep transfer learning model for accurate grading of renal cell carcinoma from kidney histopathology images](https://www.nature.com/articles/s41598-025-19930-7)

Authors: Mohammed Alghamdi et al.

### 3. [Child behavior recognition in social robot interaction using stacked deep neural networks and biomechanical signals](https://www.nature.com/articles/s41598-025-19728-7)

Authors: Sadiq J. Hamandi et al.

### 4. [A geography of indoors for analyzing global ways of living using computer vision](https://www.nature.com/articles/s41598-025-12198-x)

Authors: Martina Mazzarello et al.

### 5. [Anatomically informed deep learning framework for generating fast, low-dose synthetic CBCT for prostate radiotherapy](https://www.nature.com/articles/s41598-025-23781-7)

Authors: Mustafa Kadhim et al.

### 6. [Distinct hydrologic response patterns and trends worldwide revealed by physics-embedded learning](https://www.nature.com/articles/s41467-025-64367-1)

Authors: Haoyu Ji et al.

### 7. [A self-supervised group recommendation model with conformity awareness](https://www.nature.com/articles/s41598-025-03241-y)

Authors: Yue Kou et al.

### 8. [An innovative 3D attention mechanism for multi-label emotion classification](https://www.nature.com/articles/s41598-025-95804-2)

Authors: Haoran Luo et al.

### 9. [Efficient quantum thermal simulation](https://www.nature.com/articles/s41586-025-09583-x)

Authors: Chi-Fang Chen et al.

