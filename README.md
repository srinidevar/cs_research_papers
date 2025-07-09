# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-08 17:00:24.995291 PST.

### Artificial Intelligence

### 1. [DisMS-TS: Eliminating Redundant Multi-Scale Features for Time Series Classification](http://arxiv.org/pdf/2507.04600v1)

Authors: Zhipeng Liu, Peibo Duan, Binwu Wang, Xuan Tang, Qi Chu, Changsheng Zhang, Yongsheng Huang, Bin Zhang

Real-world time series typically exhibit complex temporal variations, making
the time series classification task notably challenging. Recent advancements
have demonstrated the potential of multi-scale analysis approaches, which
provide an effective solution for capturing these complex temporal patterns.
However, existing multi-scale analysis-based time series prediction methods
fail to eliminate redundant scale-shared features across multi-scale time
series, resulting in the model over- or under-focusing on scale-shared
features. To address this issue, we propose a novel end-to-end Disentangled
Multi-Scale framework for Time Series classification (DisMS-TS). The core idea
of DisMS-TS is to eliminate redundant shared features in multi-scale time
series, thereby improving prediction performance. Specifically, we propose a
temporal disentanglement module to capture scale-shared and scale-specific
temporal representations, respectively. Subsequently, to effectively learn both
scale-shared and scale-specific temporal representations, we introduce two
regularization terms that ensure the consistency of scale-shared
representations and the disparity of scale-specific representations across all
temporal scales. Extensive experiments conducted on multiple datasets validate
the superiority of DisMS-TS over its competitive baselines, with the accuracy
improvement up to 9.71%.

### 2. [Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message](http://arxiv.org/pdf/2507.04673v1)

Authors: Wei Duan, Li Qian

The rise of conversational interfaces has greatly enhanced LLM usability by
leveraging dialogue history for sophisticated reasoning. However, this reliance
introduces an unexplored attack surface. This paper introduces Trojan Horse
Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by
forging the model's own past utterances within the conversational history
provided to its API. A malicious payload is injected into a model-attributed
message, followed by a benign user prompt to trigger harmful content
generation. This vulnerability stems from Asymmetric Safety Alignment: models
are extensively trained to refuse harmful user requests but lack comparable
skepticism towards their own purported conversational history. This implicit
trust in its "past" creates a high-impact vulnerability. Experimental
validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan
Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than
established user-turn jailbreaking methods. These findings reveal a fundamental
flaw in modern conversational AI security, necessitating a paradigm shift from
input-level filtering to robust, protocol-level validation of conversational
context integrity.

### 3. [LumiCRS: Asymmetric Contrastive Prototype Learning for Long-Tail Conversational Movie Recommendation](http://arxiv.org/pdf/2507.04722v1)

Authors: Jinzhi Wang, Bin Li, Qingke Peng, Haozhou Li, Zeyuan Zeng, Ruimeng Li, Biyi Zhou

Conversational recommender systems (CRSs) often suffer from an extreme
long-tail distribution of dialogue data, causing a strong bias toward
head-frequency blockbusters that sacrifices diversity and exacerbates the
cold-start problem. An empirical analysis of DCRS and statistics on the REDIAL
corpus show that only 10% of head movies account for nearly half of all
mentions, whereas about 70% of tail movies receive merely 26% of the attention.
This imbalance gives rise to three critical challenges: head over-fitting, body
representation drift, and tail sparsity. To address these issues, we propose
LumiCRS, an end-to-end framework that mitigates long-tail imbalance through
three mutually reinforcing layers: (i) an Adaptive Comprehensive Focal Loss
(ACFL) that dynamically adjusts class weights and focusing factors to curb head
over-fitting and reduce popularity bias; (ii) Prototype Learning for Long-Tail
Recommendation, which selects semantic, affective, and contextual prototypes to
guide clustering and stabilize body and tail representations; and (iii) a
GPT-4o-driven prototype-guided dialogue augmentation module that automatically
generates diverse long-tail conversational snippets to alleviate tail sparsity
and distribution shift. Together, these strategies enable LumiCRS to markedly
improve recommendation accuracy, diversity, and fairness: on the REDIAL and
INSPIRED benchmarks, LumiCRS boosts Recall@10 and Tail-Recall@10 by 7-15% over
fifteen strong baselines, while human evaluations confirm superior fluency,
informativeness, and long-tail relevance. These results demonstrate the
effectiveness of multi-layer collaboration in building an efficient and fair
long-tail conversational recommender.

### 4. [LLM-based Question-Answer Framework for Sensor-driven HVAC System Interaction](http://arxiv.org/pdf/2507.04748v1)

Authors: Sungmin Lee, Minju Kang, Joonhee Lee, Seungyong Lee, Dongju Kim, Jingi Hong, Jun Shin, Pei Zhang, JeongGil Ko

Question-answering (QA) interfaces powered by large language models (LLMs)
present a promising direction for improving interactivity with HVAC system
insights, particularly for non-expert users. However, enabling accurate,
real-time, and context-aware interactions with HVAC systems introduces unique
challenges, including the integration of frequently updated sensor data,
domain-specific knowledge grounding, and coherent multi-stage reasoning. In
this paper, we present JARVIS, a two-stage LLM-based QA framework tailored for
sensor data-driven HVAC system interaction. JARVIS employs an Expert-LLM to
translate high-level user queries into structured execution instructions, and
an Agent that performs SQL-based data retrieval, statistical processing, and
final response generation. To address HVAC-specific challenges, JARVIS
integrates (1) an adaptive context injection strategy for efficient HVAC and
deployment-specific information integration, (2) a parameterized SQL builder
and executor to improve data access reliability, and (3) a bottom-up planning
scheme to ensure consistency across multi-stage response generation. We
evaluate JARVIS using real-world data collected from a commercial HVAC system
and a ground truth QA dataset curated by HVAC experts to demonstrate its
effectiveness in delivering accurate and interpretable responses across diverse
queries. Results show that JARVIS consistently outperforms baseline and
ablation variants in both automated and user-centered assessments, achieving
high response quality and accuracy.

### 5. [Application and Evaluation of Large Language Models for Forecasting the Impact of Traffic Incidents](http://arxiv.org/pdf/2507.04803v1)

Authors: George Jagadeesh, Srikrishna Iyer, Michal Polanowski, Kai Xin Thia

This study examines the feasibility of applying large language models (LLMs)
for forecasting the impact of traffic incidents on the traffic flow. The use of
LLMs for this task has several advantages over existing machine learning-based
solutions such as not requiring a large training dataset and the ability to
utilize free-text incident logs. We propose a fully LLM-based solution that
predicts the incident impact using a combination of traffic features and
LLM-extracted incident features. A key ingredient of this solution is an
effective method of selecting examples for the LLM's in-context learning. We
evaluate the performance of three advanced LLMs and two state-of-the-art
machine learning models on a real traffic incident dataset. The results show
that the best-performing LLM matches the accuracy of the most accurate machine
learning model, despite the former not having been trained on this prediction
task. The findings indicate that LLMs are a practically viable option for
traffic incident impact prediction.

### 6. [DoPI: Doctor-like Proactive Interrogation LLM for Traditional Chinese Medicine](http://arxiv.org/pdf/2507.04877v1)

Authors: Zewen Sun, Ruoxiang Huang, Jiahe Feng, Rundong Kong, Yuqian Wang, Hengyu Liu, Ziqi Gong, Yuyuan Qin, Yingxue Wang, Yu Wang

Enhancing interrogation capabilities in Traditional Chinese Medicine (TCM)
diagnosis through multi-turn dialogues and knowledge graphs presents a
significant challenge for modern AI systems. Current large language models
(LLMs), despite their advancements, exhibit notable limitations in medical
applications, particularly in conducting effective multi-turn dialogues and
proactive questioning. These shortcomings hinder their practical application
and effectiveness in simulating real-world diagnostic scenarios. To address
these limitations, we propose DoPI, a novel LLM system specifically designed
for the TCM domain. The DoPI system introduces a collaborative architecture
comprising a guidance model and an expert model. The guidance model conducts
multi-turn dialogues with patients and dynamically generates questions based on
a knowledge graph to efficiently extract critical symptom information.
Simultaneously, the expert model leverages deep TCM expertise to provide final
diagnoses and treatment plans. Furthermore, this study constructs a multi-turn
doctor-patient dialogue dataset to simulate realistic consultation scenarios
and proposes a novel evaluation methodology that does not rely on manually
collected real-world consultation data. Experimental results show that the DoPI
system achieves an accuracy rate of 84.68 percent in interrogation outcomes,
significantly enhancing the model's communication ability during diagnosis
while maintaining professional expertise.

### 7. [Supported Abstract Argumentation for Case-Based Reasoning](http://arxiv.org/pdf/2507.04994v1)

Authors: Adam Gould, Gabriel de Olim Gaul, Francesca Toni

We introduce Supported Abstract Argumentation for Case-Based Reasoning
(sAA-CBR), a binary classification model in which past cases engage in debates
by arguing in favour of their labelling and attacking or supporting those with
opposing or agreeing labels. With supports, sAA-CBR overcomes the limitation of
its precursor AA-CBR, which can contain extraneous cases (or spikes) that are
not included in the debates. We prove that sAA-CBR contains no spikes, without
trading off key model properties

### 8. [How Rules Represent Causal Knowledge: Causal Modeling with Abductive Logic Programs](http://arxiv.org/pdf/2507.05088v1)

Authors: Kilian Rückschloß, Felix Weitkämper

Pearl observes that causal knowledge enables predicting the effects of
interventions, such as actions, whereas descriptive knowledge only permits
drawing conclusions from observation. This paper extends Pearl's approach to
causality and interventions to the setting of stratified abductive logic
programs. It shows how stable models of such programs can be given a causal
interpretation by building on philosophical foundations and recent work by
Bochman and Eelink et al. In particular, it provides a translation of abductive
logic programs into causal systems, thereby clarifying the informal causal
reading of logic program rules and supporting principled reasoning about
external actions. The main result establishes that the stable model semantics
for stratified programs conforms to key philosophical principles of causation,
such as causal sufficiency, natural necessity, and irrelevance of unobserved
effects. This justifies the use of stratified abductive logic programs as a
framework for causal modeling and for predicting the effects of interventions

### 9. [Rule Learning for Knowledge Graph Reasoning under Agnostic Distribution Shift](http://arxiv.org/pdf/2507.05110v1)

Authors: Shixuan Liu, Yue He, Yunfei Wang, Hao Zou, Haoxiang Cheng, Wenjing Yang, Peng Cui, Zhong Liu

Knowledge graph (KG) reasoning remains a critical research area focused on
inferring missing knowledge by analyzing relationships among observed facts.
Despite its success, a key limitation of existing KG reasoning methods is their
dependence on the I.I.D assumption. This assumption can easily be violated due
to unknown sample selection bias during training or agnostic distribution
shifts during testing, significantly compromising model performance and
reliability. To facilitate the deployment of KG reasoning in wild environments,
this study investigates learning logical rules from KGs affected by unknown
selection bias. Additionally, we address test sets with agnostic distribution
shifts, formally defining this challenge as out-of-distribution (OOD) KG
reasoning-a previously underexplored problem. To solve the issue, we propose
the Stable Rule Learning (StableRule) framework, an end-to-end methodology that
integrates feature decorrelation with rule learning network, to enhance OOD
generalization performance. By leveraging feature decorrelation, the StableRule
framework mitigates the adverse effects of covariate shifts arising in OOD
scenarios, thereby improving the robustness of the rule learning component in
effectively deriving logical rules. Extensive experiments on seven benchmark
KGs demonstrate the framework's superior effectiveness and stability across
diverse heterogeneous environments, underscoring its practical significance for
real-world applications.

### 10. [GIST: Cross-Domain Click-Through Rate Prediction via Guided Content-Behavior Distillation](http://arxiv.org/pdf/2507.05142v1)

Authors: Wei Xu, Haoran Li, Baoyuan Ou, Lai Xu, Yingjie Qin, Ruilong Su, Ruiwen Xu

Cross-domain Click-Through Rate prediction aims to tackle the data sparsity
and the cold start problems in online advertising systems by transferring
knowledge from source domains to a target domain. Most existing methods rely on
overlapping users to facilitate this transfer, often focusing on joint training
or pre-training with fine-tuning approach to connect the source and target
domains. However, in real-world industrial settings, joint training struggles
to learn optimal representations with different distributions, and pre-training
with fine-tuning is not well-suited for continuously integrating new data. To
address these issues, we propose GIST, a cross-domain lifelong sequence model
that decouples the training processes of the source and target domains. Unlike
previous methods that search lifelong sequences in the source domains using
only content or behavior signals or their simple combinations, we innovatively
introduce a Content-Behavior Joint Training Module (CBJT), which aligns
content-behavior distributions and combines them with guided information to
facilitate a more stable representation. Furthermore, we develop an Asymmetric
Similarity Integration strategy (ASI) to augment knowledge transfer through
similarity computation. Extensive experiments demonstrate the effectiveness of
GIST, surpassing SOTA methods on offline evaluations and an online A/B test.
Deployed on the Xiaohongshu (RedNote) platform, GIST effectively enhances
online ads system performance at scale, serving hundreds of millions of daily
active users.

### Hardware Architecture

### 1. [NeuroPDE: A Neuromorphic PDE Solver Based on Spintronic and Ferroelectric Devices](http://arxiv.org/pdf/2507.04677v1)

Authors: Siqing Fu, Lizhou Wu, Tiejun Li, Chunyuan Zhang, Sheng Ma, Jianmin Zhang, Yuhan Tang, Jixuan Tang

In recent years, new methods for solving partial differential equations
(PDEs) such as Monte Carlo random walk methods have gained considerable
attention. However, due to the lack of hardware-intrinsic randomness in the
conventional von Neumann architecture, the performance of PDE solvers is
limited. In this paper, we introduce NeuroPDE, a hardware design for
neuromorphic PDE solvers that utilizes emerging spintronic and ferroelectric
devices. NeuroPDE incorporates spin neurons that are capable of probabilistic
transmission to emulate random walks, along with ferroelectric synapses that
store continuous weights non-volatilely. The proposed NeuroPDE achieves a
variance of less than 1e-2 compared to analytical solutions when solving
diffusion equations, demonstrating a performance advantage of 3.48x to 315x
speedup in execution time and an energy consumption advantage of 2.7x to 29.8x
over advanced CMOS-based neuromorphic chips. By leveraging the inherent
physical stochasticity of emerging devices, this study paves the way for future
probabilistic neuromorphic computing systems.

### 2. [Jack Unit: An Area- and Energy-Efficient Multiply-Accumulate (MAC) Unit Supporting Diverse Data Formats](http://arxiv.org/pdf/2507.04772v1)

Authors: Seock-Hwan Noh, Sungju Kim, Seohyun Kim, Daehoon Kim, Jaeha Kung, Yeseong Kim

In this work, we introduce an area- and energy-efficient multiply-accumulate
(MAC) unit, named Jack unit, that is a jack-of-all-trades, supporting various
data formats such as integer (INT), floating point (FP), and microscaling data
format (MX). It provides bit-level flexibility and enhances hardware efficiency
by i) replacing the carry-save multiplier (CSM) in the FP multiplier with a
precision-scalable CSM, ii) performing the adjustment of significands based on
the exponent differences within the CSM, and iii) utilizing 2D sub-word
parallelism. To assess effectiveness, we implemented the layout of the Jack
unit and three baseline MAC units. Additionally, we designed an AI accelerator
equipped with our Jack units to compare with a state-of-the-art AI accelerator
supporting various data formats. The proposed MAC unit occupies 1.17~2.01x
smaller area and consumes 1.05~1.84x lower power compared to the baseline MAC
units. On five AI benchmarks, the accelerator designed with our Jack units
improves energy efficiency by 1.32~5.41x over the baseline across various data
formats.

### 3. [Optimizing Scalable Multi-Cluster Architectures for Next-Generation Wireless Sensing and Communication](http://arxiv.org/pdf/2507.05012v1)

Authors: Samuel Riedel, Yichao Zhang, Marco Bertuletti, Luca Benini

Next-generation wireless technologies (for immersive-massive communication,
joint communication and sensing) demand highly parallel architectures for
massive data processing. A common architectural template scales up by grouping
tens to hundreds of cores into shared-memory clusters, which are then scaled
out as multi-cluster manycore systems. This hierarchical design, used in GPUs
and accelerators, requires a balancing act between fewer large clusters and
more smaller clusters, affecting design complexity, synchronization,
communication efficiency, and programmability. While all multi-cluster
architectures must balance these trade-offs, there is limited insight into
optimal cluster sizes. This paper analyzes various cluster configurations,
focusing on synchronization, data movement overhead, and programmability for
typical wireless sensing and communication workloads. We extend the open-source
shared-memory cluster MemPool into a multi-cluster architecture and propose a
novel double-buffering barrier that decouples processor and DMA. Our results
show a single 256-core cluster can be twice as fast as 16 16-core clusters for
memory-bound kernels and up to 24% faster for compute-bound kernels due to
reduced synchronization and communication overheads.

### 4. [ViPSN 2.0: A Reconfigurable Battery-free IoT Platform for Vibration Energy Harvesting](http://arxiv.org/pdf/2507.05081v1)

Authors: Xin Li, Mianxin Xiao, Xi Shen, Jiaqing Chu, Weifeng Huang, Jiashun Li, Yaoyi Li, Mingjing Cai, Jiaming Chen, Xinming Zhang, Daxing Zhang, Congsi Wang, Hong Tang, Bao Zhao, Qitao Lu, Yilong Wang, Jianjun Wang, Minyi Xu, Shitong Fang, Xuanyu Huang. Chaoyang Zhao, Zicheng Liu, Yaowen Yang, Guobiao Hu, Junrui Liang, Wei-Hsin Liao

Vibration energy harvesting is a promising solution for powering battery-free
IoT systems; however, the instability of ambient vibrations presents
significant challenges, such as limited harvested energy, intermittent power
supply, and poor adaptability to various applications. To address these
challenges, this paper proposes ViPSN2.0, a modular and reconfigurable IoT
platform that supports multiple vibration energy harvesters (piezoelectric,
electromagnetic, and triboelectric) and accommodates sensing tasks with varying
application requirements through standardized hot-swappable interfaces.
ViPSN~2.0 incorporates an energy-indication power management framework tailored
to various application demands, including light-duty discrete sampling,
heavy-duty high-power sensing, and complex-duty streaming tasks, thereby
effectively managing fluctuating energy availability. The platform's
versatility and robustness are validated through three representative
applications: ViPSN-Beacon, enabling ultra-low-power wireless beacon
transmission from a single transient fingertip press; ViPSN-LoRa, supporting
high-power, long-range wireless communication powered by wave vibrations in
actual marine environments; and ViPSN-Cam, enabling intermittent image capture
and wireless transfer. Experimental results demonstrate that ViPSN~2.0 can
reliably meet a wide range of requirements in practical battery-free IoT
deployments under energy-constrained conditions.

### 5. [ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven Reinforcement Learning](http://arxiv.org/pdf/2507.04736v1)

Authors: Zhirong Chen, Kaiyan Chang, Zhuolin Li, Xinyang He, Chujie Chen, Cangyuan Li, Mengdi Wang, Haobo Xu, Yinhe Han, Ying Wang

Large Language Models (LLMs) show significant potential for automating
Register-Transfer Level (RTL) code generation. However, current approaches face
a critical challenge: they can not simultaneously optimize for functional
correctness and hardware quality (Power, Performance, Area - PPA). Methods
based on supervised fine-tuning often generate functionally correct but
PPA-suboptimal code, lacking mechanisms to learn optimization principles. In
contrast, post-processing techniques that attempt to improve PPA metrics after
generation are often inefficient because they operate externally without
updating the LLM's parameters, thus failing to enhance the model's intrinsic
design capabilities.
  To bridge this gap, we introduce ChipSeek-R1, a hierarchical reward-driven
reinforcement learning framework to train LLMs to generate RTL code that
achieves both functional correctness and optimized PPA metrics. ChipSeek-R1
employs a hierarchical reward system, which incorporates direct feedback on
syntax, functional correctness (from simulators) and PPA metrics (from
synthesis tools) during reinforcement learning. This enables the model to learn
complex hardware design trade-offs via trial-and-error, generating RTL code
that is both functionally correct and PPA-optimized. Evaluating ChipSeek-R1 on
standard benchmarks (VerilogEval, RTLLM), we achieve state-of-the-art results
in functional correctness. Notably, on the RTLLM benchmark, ChipSeek-R1
generated 27 RTL designs surpassing the PPA metrics of the original
human-written code. Our findings demonstrate the effectiveness of integrating
toolchain feedback into LLM training and highlight the potential for
reinforcement learning to enable automated generation of human-surpassing RTL
code. We open-source our code in anonymous github.

### Computational Complexity

### 1. [Testing for Renamability to Classes of Clause Sets](http://arxiv.org/pdf/2507.05044v1)

Authors: Albert Brandl, Christian G. Fermüller, Gernot Salzer

This paper investigates the problem of testing clause sets for membership in
classes known from literature. In particular, we are interested in classes
defined via renaming: Is it possible to rename the predicates in a way such
that positive and negative literals satisfy certain conditions? We show that
for classes like Horn or OCC1N, the existence of such renamings can be decided
in polynomial time, whereas the same problem is NP-complete for class PVD. The
decision procedures are based on hyper-resolution; if a renaming exists, it can
be extracted from the final saturated clause set.

### Computational Engineering

### 1. [Operator-based machine learning framework for generalizable prediction of unsteady treatment dynamics in stormwater infrastructure](http://arxiv.org/pdf/2507.04682v1)

Authors: Mohamed Shatarah, Kai Liu, Haochen Li

Stormwater infrastructures are decentralized urban water-management systems
that face highly unsteady hydraulic and pollutant loadings from episodic
rainfall-runoff events. Accurately evaluating their in-situ treatment
performance is essential for cost-effective design and planning. Traditional
lumped dynamic models (e.g., continuously stirred tank reactor, CSTR) are
computationally efficient but oversimplify transport and reaction processes,
limiting predictive accuracy and insight. Computational fluid dynamics (CFD)
resolves detailed turbulent transport and pollutant fate physics but incurs
prohibitive computational cost for unsteady and long-term simulations. To
address these limitations, this study develops a composite operator-based
neural network (CPNN) framework that leverages state-of-the-art operator
learning to predict the spatial and temporal dynamics of hydraulics and
particulate matter (PM) in stormwater treatment. The framework is demonstrated
on a hydrodynamic separator (HS), a common urban treatment device. Results
indicate that the CPNN achieves R2 > 0.8 for hydraulic predictions in 95.2% of
test cases; for PM concentration predictions, R2 > 0.8 in 72.6% of cases and
0.4 < R2 < 0.8 in 22.6%. The analysis identifies challenges in capturing
dynamics under extreme low-flow conditions, owing to their lower contribution
to the training loss. Exploiting the automatic-differentiation capability of
the CPNN, sensitivity analyses quantify the influence of storm event loading on
PM transport. Finally, the potential of the CPNN framework for continuous,
long-term evaluation of stormwater infrastructure performance is discussed,
marking a step toward robust, climate-aware planning and implementation.

### 2. [From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems](http://arxiv.org/pdf/2507.04996v1)

Authors: Jiangbo Yu

Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity
to operate according to internal rules without external control. Accordingly,
autonomous vehicles (AuVs) are defined as systems capable of perceiving their
environment and executing preprogrammed tasks independently of external input.
However, both research and real-world deployments increasingly showcase
vehicles that demonstrate behaviors beyond this definition (including the SAE
levels 1 to 6), such as interaction with humans and machines, goal adaptation,
contextual reasoning, external tool use, and long-term planning, particularly
with the integration of large language models (LLMs) and agentic AI systems.
These developments reveal a conceptual gap between technical autonomy and the
broader cognitive and social capabilities needed for future human-centered
mobility systems. To address this, we introduce the concept of agentic vehicles
(AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and
interact within complex environments. This paper presents a systems-level
framework to characterize AgVs, focusing on their cognitive and communicative
layers and differentiating them from conventional AuVs. It synthesizes relevant
advances in agentic AI, robotics, multi-agent systems, and human-machine
interaction, and highlights how agentic AI, through high-level reasoning and
tool use, can function not merely as computational tools but as interactive
agents embedded in mobility ecosystems. The paper concludes by identifying key
challenges in the development and governance of AgVs, including safety,
real-time control, public acceptance, ethical alignment, and regulatory
frameworks.

### Computational Geometry

### 1. [Computing Largest Subsets of Points Whose Convex Hulls have Bounded Area and Diameter](http://arxiv.org/pdf/2507.04933v1)

Authors: Gianmarco Picarella, Marc van Kreveld, Frank Staals, Sjoerd de Vries

We study the problem of computing a convex region with bounded area and
diameter that contains the maximum number of points from a given point set $P$.
We show that this problem can be solved in $O(n^6k)$ time and $O(n^3k)$ space,
where $n$ is the size of $P$ and $k$ is the maximum number of points in the
found region. We experimentally compare this new algorithm with an existing
algorithm that does the same but without the diameter constraint, which runs in
$O(n^3k)$ time. For the new algorithm, we use different diameters. We use both
synthetic data and data from an application in cancer detection, which
motivated our research.

### 2. [Node-neighbor subnetworks and Hk-core decomposition](http://arxiv.org/pdf/2507.04948v1)

Authors: Dinghua Shi, Yang Zhao, Guanrong Chen

The network homology Hk-core decomposition proposed in this article is
similar to the k-core decomposition based on node degrees of the network. The
C. elegans neural network and the cat cortical network are used as examples to
reveal the symmetry of the deep structures of such networks. First, based on
the concept of neighborhood in mathematics, some new concepts are introduced,
including such as node-neighbor subnetwork and Betti numbers of the neighbor
subnetwork, among others. Then, the Betti numbers of the neighbor subnetwork of
each node are computed, which are used to perform Hk-core decomposition of the
network homology. The construction process is as follows: the initial network
is referred to as the H0-core; the H1-core is obtained from the H0-core by
deleting some nodes of certain properties; the H2-core is obtained from the
H1-core by deleting some nodes or edges of certain properties; the H3-core is
obtained from the H2-core by deleting some nodes of certain properties or by
retaining the nodes of certain properties, and so on, which will be described
in detail in the main text. Throughout the process, the index of node involved
in deleting edge needs to be updated in every step. The Hk-core decomposition
is easy to implement in parallel. It has a wide range of applications in many
fields such as network science, data science, computational topology, and
artificial intelligence. In this article, we also show how to use it to
simplify homology calculation, e.g. for the C. elegans neural network, whereas
the results of decomposition are the H1-core, the H2-core, and the H3-core.
Thus, the simplexes consisting of four highest-order cavities in the H3-core
subnetwork can also be directly obtained.

### 3. [Approximation and Hardness of Polychromatic TSP](http://arxiv.org/pdf/2507.04974v1)

Authors: Thomas Schibler, Subhash Suri, Jie Xue

We introduce the Polychromatic Traveling Salesman Problem (PCTSP), where the
input is an edge weighted graph whose vertices are partitioned into $k$
equal-sized color classes, and the goal is to find a minimum-length Hamiltonian
cycle that visits the classes in a fixed cyclic order. This generalizes the
Bipartite TSP (when $k = 2$) and the classical TSP (when $k = n$). We give a
polynomial-time $(3 - 2 * 10^{-36})$-approximation algorithm for metric PCTSP.
Complementing this, we show that Euclidean PCTSP is APX-hard even in $R^2$,
ruling out the existence of a PTAS unless P = NP.

### Computation and Language

### 1. [Retain or Reframe? A Computational Framework for the Analysis of Framing in News Articles and Reader Comments](http://arxiv.org/pdf/2507.04612v1)

Authors: Matteo Guida, Yulia Otmakhova, Eduard Hovy, Lea Frermann

When a news article describes immigration as an "economic burden" or a
"humanitarian crisis," it selectively emphasizes certain aspects of the issue.
Although \textit{framing} shapes how the public interprets such issues,
audiences do not absorb frames passively but actively reorganize the presented
information. While this relationship between source content and audience
response is well-documented in the social sciences, NLP approaches often ignore
it, detecting frames in articles and responses in isolation. We present the
first computational framework for large-scale analysis of framing across source
content (news articles) and audience responses (reader comments).
Methodologically, we refine frame labels and develop a framework that
reconstructs dominant frames in articles and comments from sentence-level
predictions, and aligns articles with topically relevant comments. Applying our
framework across eleven topics and two news outlets, we find that frame reuse
in comments correlates highly across outlets, while topic-specific patterns
vary. We release a frame classifier that performs well on both articles and
comments, a dataset of article and comment sentences manually labeled for
frames, and a large-scale dataset of articles and comments with predicted frame
labels.

### 2. [Put Teacher in Student's Shoes: Cross-Distillation for Ultra-compact Model Compression Framework](http://arxiv.org/pdf/2507.04636v1)

Authors: Maolin Wang, Jun Chu, Sicong Xie, Xiaoling Zang, Yao Zhao, Wenliang Zhong, Xiangyu Zhao

In the era of mobile computing, deploying efficient Natural Language
Processing (NLP) models in resource-restricted edge settings presents
significant challenges, particularly in environments requiring strict privacy
compliance, real-time responsiveness, and diverse multi-tasking capabilities.
These challenges create a fundamental need for ultra-compact models that
maintain strong performance across various NLP tasks while adhering to
stringent memory constraints. To this end, we introduce Edge ultra-lIte BERT
framework (EI-BERT) with a novel cross-distillation method. EI-BERT efficiently
compresses models through a comprehensive pipeline including hard token
pruning, cross-distillation and parameter quantization. Specifically, the
cross-distillation method uniquely positions the teacher model to understand
the student model's perspective, ensuring efficient knowledge transfer through
parameter integration and the mutual interplay between models. Through
extensive experiments, we achieve a remarkably compact BERT-based model of only
1.91 MB - the smallest to date for Natural Language Understanding (NLU) tasks.
This ultra-compact model has been successfully deployed across multiple
scenarios within the Alipay ecosystem, demonstrating significant improvements
in real-world applications. For example, it has been integrated into Alipay's
live Edge Recommendation system since January 2024, currently serving the app's
recommendation traffic across \textbf{8.4 million daily active devices}.

### 3. [R1-RE: Cross-Domain Relationship Extraction with RLVR](http://arxiv.org/pdf/2507.04642v1)

Authors: Runpeng Dai, Tong Zheng, Run Yang, Hongtu Zhu

Relationship extraction (RE) is a core task in natural language processing.
Traditional approaches typically frame RE as a supervised learning problem,
directly mapping context to labels-an approach that often suffers from poor
out-of-domain (OOD) generalization. Inspired by the workflow of human
annotators, we reframe RE as a reasoning task guided by annotation guidelines
and introduce R1-RE, the first reinforcement learning with verifiable reward
(RLVR) framework for RE tasks. Our method elicits the reasoning abilities of
small language models for annotation tasks, resulting in significantly improved
OOD robustness. We evaluate our approach on the public Sem-2010 dataset and a
private MDKG dataset. The R1-RE-7B model attains an average OOD accuracy of
approximately 70%, on par with leading proprietary models such as GPT-4o.
Additionally, our comprehensive analysis provides novel insights into the
training dynamics and emergent reasoning behaviors of the RLVR paradigm for RE.

### 4. [XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL](http://arxiv.org/pdf/2507.04701v1)

Authors: Yifu Liu, Yin Zhu, Yingqi Gao, Zhiling Luo, Xiaoxia Li, Xiaorong Shi, Yuntao Hong, Jinyang Gao, Yu Li, Bolin Ding, Jingren Zhou

To leverage the advantages of LLM in addressing challenges in the Text-to-SQL
task, we present XiYan-SQL, an innovative framework effectively generating and
utilizing multiple SQL candidates. It consists of three components: 1) a Schema
Filter module filtering and obtaining multiple relevant schemas; 2) a
multi-generator ensemble approach generating multiple highquality and diverse
SQL queries; 3) a selection model with a candidate reorganization strategy
implemented to obtain the optimal SQL query. Specifically, for the
multi-generator ensemble, we employ a multi-task fine-tuning strategy to
enhance the capabilities of SQL generation models for the intrinsic alignment
between SQL and text, and construct multiple generation models with distinct
generation styles by fine-tuning across different SQL formats. The experimental
results and comprehensive analysis demonstrate the effectiveness and robustness
of our framework. Overall, XiYan-SQL achieves a new SOTA performance of 75.63%
on the notable BIRD benchmark, surpassing all previous methods. It also attains
SOTA performance on the Spider test set with an accuracy of 89.65%.

### 5. [Why We Feel What We Feel: Joint Detection of Emotions and Their Opinion Triggers in E-commerce](http://arxiv.org/pdf/2507.04708v1)

Authors: Arnav Attri, Anuj Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera

Customer reviews on e-commerce platforms capture critical affective signals
that drive purchasing decisions. However, no existing research has explored the
joint task of emotion detection and explanatory span identification in
e-commerce reviews - a crucial gap in understanding what triggers customer
emotional responses. To bridge this gap, we propose a novel joint task unifying
Emotion detection and Opinion Trigger extraction (EOT), which explicitly models
the relationship between causal text spans (opinion triggers) and affective
dimensions (emotion categories) grounded in Plutchik's theory of 8 primary
emotions. In the absence of labeled data, we introduce EOT-X, a human-annotated
collection of 2,400 reviews with fine-grained emotions and opinion triggers. We
evaluate 23 Large Language Models (LLMs) and present EOT-DETECT, a structured
prompting framework with systematic reasoning and self-reflection. Our
framework surpasses zero-shot and chain-of-thought techniques, across
e-commerce domains.

### 6. [LOOM-Scope: a comprehensive and efficient LOng-cOntext Model evaluation framework](http://arxiv.org/pdf/2507.04723v1)

Authors: Zecheng Tang, Haitian Wang, Quantong Qiu, Baibei Ji, Ruoxi Sun, Keyan Zhou, Juntao Li, Min Zhang

Long-context processing has become a fundamental capability for large
language models~(LLMs). To assess model's long-context performance, numerous
long-context evaluation benchmarks have been proposed. However, variations in
evaluation settings across these benchmarks lead to inconsistent results,
making it difficult to draw reliable comparisons. Besides, the high
computational cost of long-context evaluation poses a significant barrier for
the community to conduct comprehensive assessments of long-context models. In
this paper, we propose LOOM-Scope, a comprehensive and efficient framework for
long-context evaluation. LOOM-Scope standardizes evaluation settings across
diverse benchmarks, supports deployment of efficient long-context inference
acceleration methods, and introduces a holistic yet lightweight benchmark suite
to evaluate models comprehensively. Homepage: https://loomscope.github.io

### 7. [A Tale of Two Scripts: Transliteration and Post-Correction for Judeo-Arabic](http://arxiv.org/pdf/2507.04746v1)

Authors: Juan Moreno Gonzalez, Bashar Alhafni, Nizar Habash

Judeo-Arabic refers to Arabic variants historically spoken by Jewish
communities across the Arab world, primarily during the Middle Ages. Unlike
standard Arabic, it is written in Hebrew script by Jewish writers and for
Jewish audiences. Transliterating Judeo-Arabic into Arabic script is
challenging due to ambiguous letter mappings, inconsistent orthographic
conventions, and frequent code-switching into Hebrew and Aramaic. In this
paper, we introduce a two-step approach to automatically transliterate
Judeo-Arabic into Arabic script: simple character-level mapping followed by
post-correction to address grammatical and orthographic errors. We also present
the first benchmark evaluation of LLMs on this task. Finally, we show that
transliteration enables Arabic NLP tools to perform morphosyntactic tagging and
machine translation, which would have not been feasible on the original texts.

### 8. [LLMs as Architects and Critics for Multi-Source Opinion Summarization](http://arxiv.org/pdf/2507.04751v1)

Authors: Anuj Attri, Arnav Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera

Multi-source Opinion Summarization (M-OS) extends beyond traditional opinion
summarization by incorporating additional sources of product metadata such as
descriptions, key features, specifications, and ratings, alongside reviews.
This integration results in comprehensive summaries that capture both
subjective opinions and objective product attributes essential for informed
decision-making. While Large Language Models (LLMs) have shown significant
success in various Natural Language Processing (NLP) tasks, their potential in
M-OS remains largely unexplored. Additionally, the lack of evaluation datasets
for this task has impeded further advancements. To bridge this gap, we
introduce M-OS-EVAL, a benchmark dataset for evaluating multi-source opinion
summaries across 7 key dimensions: fluency, coherence, relevance, faithfulness,
aspect coverage, sentiment consistency, specificity. Our results demonstrate
that M-OS significantly enhances user engagement, as evidenced by a user study
in which, on average, 87% of participants preferred M-OS over opinion
summaries. Our experiments demonstrate that factually enriched summaries
enhance user engagement. Notably, M-OS-PROMPTS exhibit stronger alignment with
human judgment, achieving an average Spearman correlation of \r{ho} = 0.74,
which surpasses the performance of previous methodologies.

### 9. [Spec-TOD: A Specialized Instruction-Tuned LLM Framework for Efficient Task-Oriented Dialogue Systems](http://arxiv.org/pdf/2507.04841v1)

Authors: Quang-Vinh Nguyen, Quang-Chieu Nguyen, Hoang Pham, Khac-Hoai Nam Bui

Task-oriented dialogue (TOD) systems facilitate goal-driven interactions
between users and machines. While recent advances in deep learning have
improved the performance, TOD systems often struggle in low-resource scenarios
with limited labeled data. To address this challenge, we propose Spec-TOD, a
novel framework designed to train an end-to-end TOD system with limited data.
Spec-TOD introduces two main innovations: (i) a novel specialized end-to-end
TOD framework that incorporates explicit task instructions for
instruction-tuned large language models (LLMs), and (ii) an efficient training
strategy that leverages lightweight, specialized LLMs to achieve strong
performance with minimal supervision. Experiments on the MultiWOZ dataset, a
widely used TOD benchmark, demonstrate that Spec-TOD achieves competitive
results while significantly reducing the need for labeled data. These findings
highlight the potential of the proposed framework in advancing efficient and
effective TOD systems in low-resource settings.

### 10. [Dialogue-Based Multi-Dimensional Relationship Extraction from Novels](http://arxiv.org/pdf/2507.04852v1)

Authors: Yuchen Yan, Hanjie Zhao, Senbin Zhu, Hongde Liu, Zhihong Zhang, Yuxiang Jia

Relation extraction is a crucial task in natural language processing, with
broad applications in knowledge graph construction and literary analysis.
However, the complex context and implicit expressions in novel texts pose
significant challenges for automatic character relationship extraction. This
study focuses on relation extraction in the novel domain and proposes a method
based on Large Language Models (LLMs). By incorporating relationship dimension
separation, dialogue data construction, and contextual learning strategies, the
proposed method enhances extraction performance. Leveraging dialogue structure
information, it improves the model's ability to understand implicit
relationships and demonstrates strong adaptability in complex contexts.
Additionally, we construct a high-quality Chinese novel relation extraction
dataset to address the lack of labeled resources and support future research.
Experimental results show that our method outperforms traditional baselines
across multiple evaluation metrics and successfully facilitates the automated
construction of character relationship networks in novels.

### Cryptography and Security

### 1. [FIDESlib: A Fully-Fledged Open-Source FHE Library for Efficient CKKS on GPUs](http://arxiv.org/pdf/2507.04775v1)

Authors: Carlos Agulló-Domingo, Óscar Vera-López, Seyda Guzelhan, Lohit Daksha, Aymane El Jerari, Kaustubh Shivdikar, Rashmi Agrawal, David Kaeli, Ajay Joshi, José L. Abellán

Word-wise Fully Homomorphic Encryption (FHE) schemes, such as CKKS, are
gaining significant traction due to their ability to provide
post-quantum-resistant, privacy-preserving approximate computing; an especially
desirable feature in Machine-Learning-as-a-Service (MLaaS) cloud-computing
paradigms. OpenFHE is a leading CPU-based FHE library with robust CKKS
operations, but its server-side performance is not yet sufficient for practical
cloud deployment. As GPU computing becomes more common in data centers, many
FHE libraries are adding GPU support. However, integrating an efficient GPU
backend into OpenFHE is challenging. While OpenFHE uses a Hardware Abstraction
Layer (HAL), its flexible architecture sacrifices performance due to the
abstraction layers required for multi-scheme and multi-backend compatibility.
In this work, we introduce FIDESlib, the first open-source server-side CKKS GPU
library that is fully interoperable with well-established client-side OpenFHE
operations. Unlike other existing open-source GPU libraries, FIDESlib provides
the first implementation featuring heavily optimized GPU kernels for all CKKS
primitives, including bootstrapping. Our library also integrates robust
benchmarking and testing, ensuring it remains adaptable to further
optimization. Furthermore, its software architecture is designed to support
extensions to a multi-GPU backend for enhanced acceleration. Our experiments
across various GPU systems and the leading open-source CKKS library to date,
Phantom, show that FIDESlib offers superior performance and scalability. For
bootstrapping, FIDESlib achieves no less than 70x speedup over the
AVX-optimized OpenFHE implementation.

### 2. [Hybrid Approach to Directed Fuzzing](http://arxiv.org/pdf/2507.04855v1)

Authors: Darya Parygina, Timofey Mezhuev, Daniil Kuts

Program analysis and automated testing have recently become an essential part
of SSDLC. Directed greybox fuzzing is one of the most popular automated testing
methods that focuses on error detection in predefined code regions. However, it
still lacks ability to overcome difficult program constraints. This problem can
be well addressed by symbolic execution, but at the cost of lower performance.
Thus, combining directed fuzzing and symbolic execution techniques can lead to
more efficient error detection.
  In this paper, we propose a hybrid approach to directed fuzzing with novel
seed scheduling algorithm, based on target-related interestingness and
coverage. The approach also performs minimization and sorting of objective
seeds according to a target-related information. We implement our approach in
Sydr-Fuzz tool using LibAFL-DiFuzz as directed fuzzer and Sydr as dynamic
symbolic executor. We evaluate our approach with Time to Exposure metric and
compare it with pure LibAFL-DiFuzz, AFLGo, BEACON, WAFLGo, WindRanger,
FishFuzz, and Prospector. The results show an improvement for 3 out of 7
examples with speedup up to 1.86 times over the second best result, as well as
a significant improvement for 3 out of 7 examples over the pure LibAFL-DiFuzz
fuzzer. Sydr-Fuzz hybrid approach to directed fuzzing shows high performance
and helps to improve directed fuzzing efficiency.

### 3. [LIFT: Automating Symbolic Execution Optimization with Large Language Models for AI Networks](http://arxiv.org/pdf/2507.04931v1)

Authors: Ruoxi Wang, Kun Li, Minghui Xu, Yue Zhang, Kaidi Xu, Chunchi Liu, Yinhao Xiao, Xiuzhen Cheng

Dynamic Symbolic Execution (DSE) is a key technique in program analysis,
widely used in software testing, vulnerability discovery, and formal
verification. In distributed AI systems, DSE plays a crucial role in
identifying hard-to-detect bugs, especially those arising from complex network
communication patterns. However, traditional approaches to symbolic execution
are often hindered by scalability issues and inefficiencies, particularly in
large-scale systems. This paper introduces LIFT (Large-language-model
Integrated Functional-equivalent-IR Transformation), a novel framework that
leverages Large Language Models (LLMs) to automate the optimization of
Intermediate Representations (IRs) in symbolic execution. LIFT addresses the
challenges of symbolic execution by providing a scalable, context-sensitive
solution for IR transformation. The framework consists of two phases: IR
Analysis and Optimization, where LLMs optimize time-intensive IR blocks, and
Symbolic Execution and Validation, which includes benchmarking and semantic
verification to ensure correctness and generalizability. Experiments on
real-world binaries demonstrated significant performance improvements,
including a 53.5\% reduction in execution time for bigtest and a 10.24\%
reduction for random, along with reductions in IR statements, PUT instructions,
and temporary variables. These results demonstrate that LLMs simplify IRs while
maintaining functional correctness, enhancing symbolic execution in distributed
AI systems.

### 4. [Extreme Learning Machine Based System for DDoS Attacks Detections on IoMT Devices](http://arxiv.org/pdf/2507.05132v1)

Authors: Nelly Elsayed, Lily Dzamesi, Zag ElSayed, Murat Ozer

The Internet of Medical Things (IoMT) represents a paradigm shift in the
healthcare sector, enabling the interconnection of medical devices, sensors,
and systems to enhance patient monitoring, diagnosis, and management. The rapid
evolution of IoMT presents significant benefits to the healthcare domains.
However, there is a rapid increase in distributed denial of service (DDoS)
attacks on the IoMT networks due to several vulnerabilities in the
IoMT-connected devices, which negatively impact patients' health and can even
lead to deaths. Thus, in this paper, we aim to save lives via investigating an
extreme learning machine for detecting DDoS attacks on IoMT devices. The
proposed approach achieves a high accuracy at a low implementation budget.
Thus, it can reduce the implementation cost of the DDoS detection system,
making the model capable of executing on the fog level.

### 5. [Hunting in the Dark: Metrics for Early Stage Traffic Discovery](http://arxiv.org/pdf/2507.05213v1)

Authors: Max Gao, Michael Collins, Ricky Mok, kc Claffy

Threat hunting is an operational security process where an expert analyzes
traffic, applying knowledge and lightweight tools on unlabeled data in order to
identify and classify previously unknown phenomena. In this paper, we examine
threat hunting metrics and practice by studying the detection of Crackonosh, a
cryptojacking malware package, has on various metrics for identifying its
behavior. Using a metric for discoverability, we model the ability of defenders
to measure Crackonosh traffic as the malware population decreases, evaluate the
strength of various detection methods, and demonstrate how different darkspace
sizes affect both the ability to track the malware, but enable emergent
behaviors by exploiting attacker mistakes.

### 6. [Efficient Unlearning with Privacy Guarantees](http://arxiv.org/pdf/2507.04771v1)

Authors: Josep Domingo-Ferrer, Najeeb Jebreel, David Sánchez

Privacy protection laws, such as the GDPR, grant individuals the right to
request the forgetting of their personal data not only from databases but also
from machine learning (ML) models trained on them. Machine unlearning has
emerged as a practical means to facilitate model forgetting of data instances
seen during training. Although some existing machine unlearning methods
guarantee exact forgetting, they are typically costly in computational terms.
On the other hand, more affordable methods do not offer forgetting guarantees
and are applicable only to specific ML models. In this paper, we present
\emph{efficient unlearning with privacy guarantees} (EUPG), a novel machine
unlearning framework that offers formal privacy guarantees to individuals whose
data are being unlearned. EUPG involves pre-training ML models on data
protected using privacy models, and it enables {\em efficient unlearning with
the privacy guarantees offered by the privacy models in use}. Through empirical
evaluation on four heterogeneous data sets protected with $k$-anonymity and
$\epsilon$-differential privacy as privacy models, our approach demonstrates
utility and forgetting effectiveness comparable to those of exact unlearning
methods, while significantly reducing computational and storage costs. Our code
is available at https://github.com/najeebjebreel/EUPG.

### 7. [Enabling Security on the Edge: A CHERI Compartmentalized Network Stack](http://arxiv.org/pdf/2507.04818v1)

Authors: Donato Ferraro, Andrea Bastoni, Alexander Zuepke, Andrea Marongiu

The widespread deployment of embedded systems in critical infrastructures,
interconnected edge devices like autonomous drones, and smart industrial
systems requires robust security measures. Compromised systems increase the
risks of operational failures, data breaches, and -- in safety-critical
environments -- potential physical harm to people. Despite these risks, current
security measures are often insufficient to fully address the attack surfaces
of embedded devices. CHERI provides strong security from the hardware level by
enabling fine-grained compartmentalization and memory protection, which can
reduce the attack surface and improve the reliability of such devices. In this
work, we explore the potential of CHERI to compartmentalize one of the most
critical and targeted components of interconnected systems: their network
stack. Our case study examines the trade-offs of isolating applications, TCP/IP
libraries, and network drivers on a CheriBSD system deployed on the Arm Morello
platform. Our results suggest that CHERI has the potential to enhance security
while maintaining performance in embedded-like environments.

### 8. [Cyclic Equalizability of Words and Its Application to Card-Based Cryptography](http://arxiv.org/pdf/2507.04916v1)

Authors: Kazumasa Shinagawa, Koji Nuida

Card-based cryptography is a research area to implement cryptographic
procedures using a deck of physical cards. In recent years, it has been found
to be related to finite group theory and algebraic combinatorics, and is
becoming more and more closely connected to the field of mathematics. In this
paper, we discuss the relationship between card-based cryptography and
combinatorics on words for the first time. In particular, we focus on cyclic
equality of words. We say that a set of words are cyclically equalizable if
they can be transformed to be cyclically equal by repeated simultaneous
insertion of letters. The main result of this paper is to show that two binary
words of equal length and equal Hamming weight are cyclically equalizable. As
applications of cyclic equalizability to card-based cryptography, we describe
its applications to the information erasure problem and to single-cut full-open
protocols.

### 9. [Bullshark on Narwhal: Implementation-level Workflow Analysis of Round-based DAG Consensus in Theory and Practice](http://arxiv.org/pdf/2507.04956v1)

Authors: Yusei Tanaka

Round-based DAGs enable high-performance Byzantine fault-tolerant consensus,
yet their technical advantages remain underutilized due to their short history.
While research on consensus protocols is active in both academia and industry,
many studies overlook implementation-level algorithms, leaving actual
performance unclear - particularly for theoretical protocols whose practical
performance cannot often be evaluated. Bullshark, a Round-based DAG BFT
protocol on Narwhal mempool, achieves optimal performance: 297,000 transactions
per second with 2-second latency. We analyze the algorithm's workflow, from
transaction submission to blockchain commitment, breaking it down layer by
layer at the functional level and delineating the key features and interactions
of the Bullshark and Narwhal components. Future work aims to improve
performance in Byzantine fault environments and optimize trade-offs in the CAP
theorem.

### 10. [The Hidden Threat in Plain Text: Attacking RAG Data Loaders](http://arxiv.org/pdf/2507.05093v1)

Authors: Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, Simeone Pizzi

Large Language Models (LLMs) have transformed human-machine interaction since
ChatGPT's 2022 debut, with Retrieval-Augmented Generation (RAG) emerging as a
key framework that enhances LLM outputs by integrating external knowledge.
However, RAG's reliance on ingesting external documents introduces new
vulnerabilities. This paper exposes a critical security gap at the data loading
stage, where malicious actors can stealthily corrupt RAG pipelines by
exploiting document ingestion.
  We propose a taxonomy of 9 knowledge-based poisoning attacks and introduce
two novel threat vectors -- Content Obfuscation and Content Injection --
targeting common formats (DOCX, HTML, PDF). Using an automated toolkit
implementing 19 stealthy injection techniques, we test five popular data
loaders, finding a 74.4% attack success rate across 357 scenarios. We further
validate these threats on six end-to-end RAG systems -- including white-box
pipelines and black-box services like NotebookLM and OpenAI Assistants --
demonstrating high success rates and critical vulnerabilities that bypass
filters and silently compromise output integrity. Our results emphasize the
urgent need to secure the document ingestion process in RAG systems against
covert content manipulations.

### Computer Vision and Pattern Recognition

### 1. [S$^2$Edit: Text-Guided Image Editing with Precise Semantic and Spatial Control](http://arxiv.org/pdf/2507.04584v1)

Authors: Xudong Liu, Zikun Chen, Ruowei Jiang, Ziyi Wu, Kejia Yin, Han Zhao, Parham Aarabi, Igor Gilitschenski

Recent advances in diffusion models have enabled high-quality generation and
manipulation of images guided by texts, as well as concept learning from
images. However, naive applications of existing methods to editing tasks that
require fine-grained control, e.g., face editing, often lead to suboptimal
solutions with identity information and high-frequency details lost during the
editing process, or irrelevant image regions altered due to entangled concepts.
In this work, we propose S$^2$Edit, a novel method based on a pre-trained
text-to-image diffusion model that enables personalized editing with precise
semantic and spatial control. We first fine-tune our model to embed the
identity information into a learnable text token. During fine-tuning, we
disentangle the learned identity token from attributes to be edited by
enforcing an orthogonality constraint in the textual feature space. To ensure
that the identity token only affects regions of interest, we apply object masks
to guide the cross-attention maps. At inference time, our method performs
localized editing while faithfully preserving the original identity with
semantically disentangled and spatially focused identity token learned.
Extensive experiments demonstrate the superiority of S$^2$Edit over
state-of-the-art methods both quantitatively and qualitatively. Additionally,
we showcase several compositional image editing applications of S$^2$Edit such
as makeup transfer.

### 2. [CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](http://arxiv.org/pdf/2507.04587v1)

Authors: Hanzhi Zhong, Zhiyu Xiang, Ruoyu Xu, Jingyun Fu, Peng Xu, Shaohong Wang, Zhihao Yang, Tianyu Pu, Eryun Liu

4D radar has received significant attention in autonomous driving thanks to
its robustness under adverse weathers. Due to the sparse points and noisy
measurements of the 4D radar, most of the research finish the 3D object
detection task by integrating images from camera and perform modality fusion in
BEV space. However, the potential of the radar and the fusion mechanism is
still largely unexplored, hindering the performance improvement. In this study,
we propose a cross-view two-stage fusion network called CVFusion. In the first
stage, we design a radar guided iterative (RGIter) BEV fusion module to
generate high-recall 3D proposal boxes. In the second stage, we aggregate
features from multiple heterogeneous views including points, image, and BEV for
each proposal. These comprehensive instance level features greatly help refine
the proposals and generate high-quality predictions. Extensive experiments on
public datasets show that our method outperforms the previous state-of-the-art
methods by a large margin, with 9.10% and 3.68% mAP improvements on
View-of-Delft (VoD) and TJ4DRadSet, respectively. Our code will be made
publicly available.

### 3. [QR-LoRA: Efficient and Disentangled Fine-tuning via QR Decomposition for Customized Generation](http://arxiv.org/pdf/2507.04599v1)

Authors: Jiahui Yang, Yongjia Ma, Donglin Di, Hao Li, Wei Chen, Yan Xie, Jianxun Cui, Xun Yang, Wangmeng Zuo

Existing text-to-image models often rely on parameter fine-tuning techniques
such as Low-Rank Adaptation (LoRA) to customize visual attributes. However,
when combining multiple LoRA models for content-style fusion tasks,
unstructured modifications of weight matrices often lead to undesired feature
entanglement between content and style attributes. We propose QR-LoRA, a novel
fine-tuning framework leveraging QR decomposition for structured parameter
updates that effectively separate visual attributes. Our key insight is that
the orthogonal Q matrix naturally minimizes interference between different
visual features, while the upper triangular R matrix efficiently encodes
attribute-specific transformations. Our approach fixes both Q and R matrices
while only training an additional task-specific $\Delta R$ matrix. This
structured design reduces trainable parameters to half of conventional LoRA
methods and supports effective merging of multiple adaptations without
cross-contamination due to the strong disentanglement properties between
$\Delta R$ matrices. Experiments demonstrate that QR-LoRA achieves superior
disentanglement in content-style fusion tasks, establishing a new paradigm for
parameter-efficient, disentangled fine-tuning in generative models.

### 4. [Learn 3D VQA Better with Active Selection and Reannotation](http://arxiv.org/pdf/2507.04630v1)

Authors: Shengli Zhou, Yang Liu, Feng Zheng

3D Visual Question Answering (3D VQA) is crucial for enabling models to
perceive the physical world and perform spatial reasoning. In 3D VQA, the
free-form nature of answers often leads to improper annotations that can
confuse or mislead models when training on the entire dataset. While other text
generation tasks can mitigate this issue by learning on large-scale datasets,
the scarcity of 3D scene data enlarges the negative effect of misleading
annotations. Although active learning strategies can select valuable instances
for training, they fail to identify and resolve misleading labels, which the
oracle inevitably provides in practice. To address this issue, we propose a
multi-turn interactive active learning strategy. This strategy selects data
based on models' semantic uncertainty to form a solid knowledge foundation more
effectively and actively requests reannotation from an oracle to resolve
potentially misleading labels. For uncertainty assessment, we utilize a
variance-based metric that takes semantic relationships between terms into
consideration, thus avoiding the uniform inter-class similarity assumption of
previous assessment metrics. Extensive experiments exhibit better model
performance and a substantial reduction in training costs, with a halving of
training costs for achieving relatively high accuracy. The code is available at
https://github.com/fz-zsl/AQuA.

### 5. [MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding](http://arxiv.org/pdf/2507.04635v1)

Authors: Zhicheng Zhang, Wuyou Xia, Chenxi Zhao, Zhou Yan, Xiaoqiang Liu, Yongjie Zhu, Wenyu Qin, Pengfei Wan, Di Zhang, Jufeng Yang

Multimodal large language models (MLLMs) recently showed strong capacity in
integrating data among multiple modalities, empowered by a generalizable
attention architecture. Advanced methods predominantly focus on
language-centric tuning while less exploring multimodal tokens mixed through
attention, posing challenges in high-level tasks that require fine-grained
cognition and emotion understanding. In this work, we identify the attention
deficit disorder problem in multimodal learning, caused by inconsistent
cross-modal attention and layer-by-layer decayed attention activation. To
address this, we propose a novel attention mechanism, termed MOdular Duplex
Attention (MODA), simultaneously conducting the inner-modal refinement and
inter-modal interaction. MODA employs a correct-after-align strategy to
effectively decouple modality alignment from cross-layer token mixing. In the
alignment phase, tokens are mapped to duplex modality spaces based on the basis
vectors, enabling the interaction between visual and language modality.
Further, the correctness of attention scores is ensured through adaptive masked
attention, which enhances the model's flexibility by allowing customizable
masking patterns for different modalities. Extensive experiments on 21
benchmark datasets verify the effectiveness of MODA in perception, cognition,
and emotion tasks. Source code and demo are available in
https://zzcheng.top/MODA.

### 6. [UGG-ReID: Uncertainty-Guided Graph Model for Multi-Modal Object Re-Identification](http://arxiv.org/pdf/2507.04638v1)

Authors: Xixi Wan, Aihua Zheng, Bo Jiang, Beibei Wang, Chenglong Li, Jin Tang

Multi-modal object Re-IDentification (ReID) has gained considerable attention
with the goal of retrieving specific targets across cameras using heterogeneous
visual data sources. Existing methods primarily aim to improve identification
performance, but often overlook the uncertainty arising from inherent defects,
such as intra-modal noise and inter-modal conflicts. This uncertainty is
particularly significant in the case of fine-grained local occlusion and frame
loss, which becomes a challenge in multi-modal learning. To address the above
challenge, we propose a robust approach named Uncertainty-Guided Graph model
for multi-modal object ReID (UGG-ReID). UGG-ReID is designed to mitigate noise
interference and facilitate effective multi-modal fusion by estimating both
local and sample-level aleatoric uncertainty and explicitly modeling their
dependencies. Specifically, we first propose the Gaussian patch-graph
representation model that leverages uncertainty to quantify fine-grained local
cues and capture their structural relationships. This process boosts the
expressiveness of modal-specific information, ensuring that the generated
embeddings are both more informative and robust. Subsequently, we design an
uncertainty-guided mixture of experts strategy that dynamically routes samples
to experts exhibiting low uncertainty. This strategy effectively suppresses
noise-induced instability, leading to enhanced robustness. Meanwhile, we design
an uncertainty-guided routing to strengthen the multi-modal interaction,
improving the performance. UGG-ReID is comprehensively evaluated on five
representative multi-modal object ReID datasets, encompassing diverse spectral
modalities. Experimental results show that the proposed method achieves
excellent performance on all datasets and is significantly better than current
methods in terms of noise immunity. Our code will be made public upon
acceptance.

### 7. [VectorLLM: Human-like Extraction of Structured Building Contours vis Multimodal LLMs](http://arxiv.org/pdf/2507.04664v1)

Authors: Tao Zhang, Shiqing Wei, Shihao Chen, Wenling Yu, Muying Luo, Shunping Ji

Automatically extracting vectorized building contours from remote sensing
imagery is crucial for urban planning, population estimation, and disaster
assessment. Current state-of-the-art methods rely on complex multi-stage
pipelines involving pixel segmentation, vectorization, and polygon refinement,
which limits their scalability and real-world applicability. Inspired by the
remarkable reasoning capabilities of Large Language Models (LLMs), we introduce
VectorLLM, the first Multi-modal Large Language Model (MLLM) designed for
regular building contour extraction from remote sensing images. Unlike existing
approaches, VectorLLM performs corner-point by corner-point regression of
building contours directly, mimicking human annotators' labeling process. Our
architecture consists of a vision foundation backbone, an MLP connector, and an
LLM, enhanced with learnable position embeddings to improve spatial
understanding capability. Through comprehensive exploration of training
strategies including pretraining, supervised fine-tuning, and preference
optimization across WHU, WHU-Mix, and CrowdAI datasets, VectorLLM significantly
outperformed the previous SOTA methods by 5.6 AP, 7.1 AP, 13.6 AP, respectively
in the three datasets. Remarkably, VectorLLM exhibits strong zero-shot
performance on unseen objects including aircraft, water bodies, and oil tanks,
highlighting its potential for unified modeling of diverse remote sensing
object contour extraction tasks. Overall, this work establishes a new paradigm
for vector extraction in remote sensing, leveraging the topological reasoning
capabilities of LLMs to achieve both high accuracy and exceptional
generalization. All the codes and weights will be published for promoting
community development.

### 8. [ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing](http://arxiv.org/pdf/2507.04678v1)

Authors: Zhenghui Zhao, Chen Wu, Di Wang, Hongruixuan Chen, Zhuo Zheng

Recent advancements in generative methods, especially diffusion models, have
made great progress in remote sensing image synthesis. Despite these
advancements, existing methods have not explored the simulation of future
scenarios based on given scenario images. This simulation capability has wide
applications for urban planning, land managementChangeBridge: Spatiotemporal
Image Generation with Multimodal Controls, and beyond. In this work, we propose
ChangeBridge, a conditional spatiotemporal diffusion model. Given pre-event
images and conditioned on multimodal spatial controls (e.g., text prompts,
instance layouts, and semantic maps), ChangeBridge can synthesize post-event
images. The core idea behind ChangeBridge is to modeling the noise-to-image
diffusion model, as a pre-to-post diffusion bridge. Conditioned on multimodal
controls, ChangeBridge leverages a stochastic Brownian-bridge diffusion,
directly modeling the spatiotemporal evolution between pre-event and post-event
states. To the best of our knowledge, ChangeBridge is the first spatiotemporal
generative model with multimodal controls for remote sensing. Experimental
results demonstrate that ChangeBridge can simulate high-fidelity future
scenarios aligned with given conditions, including event and event-driven
background variations. Code will be available.

### 9. [Colorectal Cancer Tumor Grade Segmentation in Digital Histopathology Images: From Giga to Mini Challenge](http://arxiv.org/pdf/2507.04681v1)

Authors: Alper Bahcekapili, Duygu Arslan, Umut Ozdemir, Berkay Ozkirli, Emre Akbas, Ahmet Acar, Gozde B. Akar, Bingdou He, Shuoyu Xu, Umit Mert Caglar, Alptekin Temizel, Guillaume Picaud, Marc Chaumont, Gérard Subsol, Luc Téot, Fahad Alsharekh, Shahad Alghannam, Hexiang Mao, Wenhua Zhang

Colorectal cancer (CRC) is the third most diagnosed cancer and the second
leading cause of cancer-related death worldwide. Accurate histopathological
grading of CRC is essential for prognosis and treatment planning but remains a
subjective process prone to observer variability and limited by global
shortages of trained pathologists. To promote automated and standardized
solutions, we organized the ICIP Grand Challenge on Colorectal Cancer Tumor
Grading and Segmentation using the publicly available METU CCTGS dataset. The
dataset comprises 103 whole-slide images with expert pixel-level annotations
for five tissue classes. Participants submitted segmentation masks via Codalab,
evaluated using metrics such as macro F-score and mIoU. Among 39 participating
teams, six outperformed the Swin Transformer baseline (62.92 F-score). This
paper presents an overview of the challenge, dataset, and the top-performing
methods

### 10. [TeethGenerator: A two-stage framework for paired pre- and post-orthodontic 3D dental data generation](http://arxiv.org/pdf/2507.04685v1)

Authors: Changsong Lei, Yaqian Liang, Shaofeng Wang, Jiajia Dai, Yong-Jin Liu

Digital orthodontics represents a prominent and critical application of
computer vision technology in the medical field. So far, the labor-intensive
process of collecting clinical data, particularly in acquiring paired 3D
orthodontic teeth models, constitutes a crucial bottleneck for developing tooth
arrangement neural networks. Although numerous general 3D shape generation
methods have been proposed, most of them focus on single-object generation and
are insufficient for generating anatomically structured teeth models, each
comprising 24-32 segmented teeth. In this paper, we propose TeethGenerator, a
novel two-stage framework designed to synthesize paired 3D teeth models pre-
and post-orthodontic, aiming to facilitate the training of downstream tooth
arrangement networks. Specifically, our approach consists of two key modules:
(1) a teeth shape generation module that leverages a diffusion model to learn
the distribution of morphological characteristics of teeth, enabling the
generation of diverse post-orthodontic teeth models; and (2) a teeth style
generation module that synthesizes corresponding pre-orthodontic teeth models
by incorporating desired styles as conditional inputs. Extensive qualitative
and quantitative experiments demonstrate that our synthetic dataset aligns
closely with the distribution of real orthodontic data, and promotes tooth
alignment performance significantly when combined with real data for training.
The code and dataset are available at
https://github.com/lcshhh/teeth_generator.

### Computers and Society

### 1. [Toward Valid Measurement Of (Un)fairness For Generative AI: A Proposal For Systematization Through The Lens Of Fair Equality of Chances](http://arxiv.org/pdf/2507.04641v1)

Authors: Kimberly Le Truong, Annette Zimmermann, Hoda Heidari

Disparities in the societal harms and impacts of Generative AI (GenAI)
systems highlight the critical need for effective unfairness measurement
approaches. While numerous benchmarks exist, designing valid measurements
requires proper systematization of the unfairness construct. Yet this process
is often neglected, resulting in metrics that may mischaracterize unfairness by
overlooking contextual nuances, thereby compromising the validity of the
resulting measurements. Building on established (un)fairness measurement
frameworks for predictive AI, this paper focuses on assessing and improving the
validity of the measurement task. By extending existing conceptual work in
political philosophy, we propose a novel framework for evaluating GenAI
unfairness measurement through the lens of the Fair Equality of Chances
framework. Our framework decomposes unfairness into three core constituents:
the harm/benefit resulting from the system outcomes, morally arbitrary factors
that should not lead to inequality in the distribution of harm/benefit, and the
morally decisive factors, which distinguish subsets that can justifiably
receive different treatments. By examining fairness through this structured
lens, we integrate diverse notions of (un)fairness while accounting for the
contextual dynamics that shape GenAI outcomes. We analyze factors contributing
to each component and the appropriate processes to systematize and measure each
in turn. This work establishes a foundation for developing more valid
(un)fairness measurements for GenAI systems.

### 2. [Real-Time AI-Driven Pipeline for Automated Medical Study Content Generation in Low-Resource Settings: A Kenyan Case Study](http://arxiv.org/pdf/2507.05212v1)

Authors: Emmanuel Korir, Eugene Wechuli

Juvenotes is a real-time AI-driven pipeline that automates the transformation
of academic documents into structured exam-style question banks, optimized for
low-resource medical education settings in Kenya. The system combines Azure
Document Intelligence for OCR and Azure AI Foundry (OpenAI o3-mini) for
question and answer generation in a microservices architecture, with a
Vue/TypeScript frontend and AdonisJS backend. Mobile-first design,
bandwidth-sensitive interfaces, institutional tagging, and offline features
address local challenges. Piloted over seven months at Kenyan medical
institutions, Juvenotes reduced content curation time from days to minutes and
increased daily active users by 40%. Ninety percent of students reported
improved study experiences. Key challenges included intermittent connectivity
and AI-generated errors, highlighting the need for offline sync and human
validation. Juvenotes shows that AI automation with contextual UX can enhance
access to quality study materials in low-resource settings.

### 3. [Perspectives on How Sociology Can Advance Theorizing about Human-Chatbot Interaction and Developing Chatbots for Social Good](http://arxiv.org/pdf/2507.05030v1)

Authors: Celeste Campos-Castillo, Xuan Kang, Linnea I. Laestadius

Recently, research into chatbots (also known as conversational agents, AI
agents, voice assistants), which are computer applications using artificial
intelligence to mimic human-like conversation, has grown sharply. Despite this
growth, sociology lags other disciplines (including computer science, medicine,
psychology, and communication) in publishing about chatbots. We suggest
sociology can advance understanding of human-chatbot interaction and offer four
sociological theories to enhance extant work in this field. The first two
theories (resource substitution theory, power-dependence theory) add new
insights to existing models of the drivers of chatbot use, which overlook
sociological concerns about how social structure (e.g., systemic
discrimination, the uneven distribution of resources within networks) inclines
individuals to use chatbots, including problematic levels of emotional
dependency on chatbots. The second two theories (affect control theory,
fundamental cause of disease theory) help inform the development of
chatbot-driven interventions that minimize safety risks and enhance equity by
leveraging sociological insights into how chatbot outputs could attend to
cultural contexts (e.g., affective norms) to promote wellbeing and enhance
communities (e.g., opportunities for civic participation). We discuss the value
of applying sociological theories for advancing theorizing about human-chatbot
interaction and developing chatbots for social good.

### 4. [SMART: Simulated Students Aligned with Item Response Theory for Question Difficulty Prediction](http://arxiv.org/pdf/2507.05129v1)

Authors: Alexander Scarlatos, Nigel Fernandez, Christopher Ormerod, Susan Lottridge, Andrew Lan

Item (question) difficulties play a crucial role in educational assessments,
enabling accurate and efficient assessment of student abilities and
personalization to maximize learning outcomes. Traditionally, estimating item
difficulties can be costly, requiring real students to respond to items,
followed by fitting an item response theory (IRT) model to get item difficulty
estimates. This approach cannot be applied to the cold-start setting for
previously unseen items either. In this work, we present SMART (Simulated
Students Aligned with IRT), a novel method for aligning simulated students with
instructed ability, which can then be used in simulations to predict the
difficulty of open-ended items. We achieve this alignment using direct
preference optimization (DPO), where we form preference pairs based on how
likely responses are under a ground-truth IRT model. We perform a simulation by
generating thousands of responses, evaluating them with an LLM-based scoring
model, and fit the resulting data to an IRT model to obtain item difficulty
estimates. Through extensive experiments on a real-world student response
dataset, we show that SMART outperforms other item difficulty prediction
methods by leveraging its improved ability alignment.

### 5. [From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems](http://arxiv.org/pdf/2507.04996v1)

Authors: Jiangbo Yu

Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity
to operate according to internal rules without external control. Accordingly,
autonomous vehicles (AuVs) are defined as systems capable of perceiving their
environment and executing preprogrammed tasks independently of external input.
However, both research and real-world deployments increasingly showcase
vehicles that demonstrate behaviors beyond this definition (including the SAE
levels 1 to 6), such as interaction with humans and machines, goal adaptation,
contextual reasoning, external tool use, and long-term planning, particularly
with the integration of large language models (LLMs) and agentic AI systems.
These developments reveal a conceptual gap between technical autonomy and the
broader cognitive and social capabilities needed for future human-centered
mobility systems. To address this, we introduce the concept of agentic vehicles
(AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and
interact within complex environments. This paper presents a systems-level
framework to characterize AgVs, focusing on their cognitive and communicative
layers and differentiating them from conventional AuVs. It synthesizes relevant
advances in agentic AI, robotics, multi-agent systems, and human-machine
interaction, and highlights how agentic AI, through high-level reasoning and
tool use, can function not merely as computational tools but as interactive
agents embedded in mobility ecosystems. The paper concludes by identifying key
challenges in the development and governance of AgVs, including safety,
real-time control, public acceptance, ethical alignment, and regulatory
frameworks.

### Databases

### 1. [AKEGEN: A LLM-based Tabular Corpus Generator for Evaluating Dataset Discovery in Data Lakes](http://arxiv.org/pdf/2507.04687v1)

Authors: Zhenwei Dai, Chuan Lei, Asterios Katsifodimos, Xiao Qin, Christos Faloutsos, Huzefa Rangwala

How to generate a large, realistic set of tables along with joinability
relationships, to stress-test dataset discovery methods? Dataset discovery
methods aim to automatically identify related data assets in a data lake. The
development and evaluation of such solutions for customers from a wide range of
business domains, relies on diverse, high quality and domain-specific tabular
benchmarks. Large language models (LLMs) are trained on a wide variety of text
data, which can provide a strong foundation of general and domain-specific
knowledge. In this paper, we ask the question -- \textit{can we leverage LLMs
to generate a tabular benchmark adequate for evaluating the dataset discovery
solutions?} In particular, we focus on the task of finding joinable tables
which is the cornerstone of virtually every dataset discovery method. Current
corpora for evaluating dataset discovery methods are mainly based on subsets of
open data, and they suffer from three important issues: $i)$ they focus on very
common and generic data types (e.g., address, id, name, etc.); $ii)$ they do
not contain human-annotated column pairs; instead, practitioners synthesize
ground truth using table splits (e.g., horizontal for table union search and
vertical ones for joinability) and $iii)$ they do not focus on semantic column
relationships.

### 2. [SHARP: Shared State Reduction for Efficient Matching of Sequential Patterns](http://arxiv.org/pdf/2507.04872v1)

Authors: Cong Yu, Tuo Shi, Matthias Weidlich, Bo Zhao

The detection of sequential patterns in data is a basic functionality of
modern data processing systems for complex event processing (CEP), OLAP, and
retrieval-augmented generation (RAG). In practice, pattern matching is
challenging, since common applications rely on a large set of patterns that
shall be evaluated with tight latency bounds. At the same time, matching needs
to maintain state, i.e., intermediate results, that grows exponentially in the
input size. Hence, systems turn to best-effort processing, striving for maximal
recall under a latency bound. Existing techniques, however, consider each
pattern in isolation, neglecting the optimization potential induced by state
sharing in pattern matching.
  In this paper, we present SHARP, a library that employs state reduction to
achieve efficient best-effort pattern matching. To this end, SHARP incorporates
state sharing between patterns through a new abstraction, coined
pattern-sharing degree (PSD). At runtime, this abstraction facilitates the
categorization and indexing of partial pattern matches. Based thereon, once a
latency bound is exceeded, SHARP realizes best-effort processing by selecting a
subset of partial matches for further processing in constant time. In
experiments with real-world data, SHARP achieves a recall of 97%, 96% and 73%
for pattern matching in CEP, OLAP, and RAG applications, under a bound of 50%
of the average processing latency.

### 3. [The Case for Instance-Optimized LLMs in OLAP Databases](http://arxiv.org/pdf/2507.04967v1)

Authors: Bardia Mohammadi, Laurent Bindschaedler

Large Language Models (LLMs) can enhance analytics systems with powerful data
summarization, cleaning, and semantic transformation capabilities. However,
deploying LLMs at scale -- processing millions to billions of rows -- remains
prohibitively expensive in computation and memory. We present IOLM-DB, a novel
system that makes LLM-enhanced database queries practical through
query-specific model optimization. Instead of using general-purpose LLMs,
IOLM-DB generates lightweight, specialized models tailored to each query's
specific needs using representative data samples. IOLM-DB reduces model
footprints by up to 76% and increases throughput by up to 3.31$\times$ while
maintaining accuracy through aggressive compression techniques, including
quantization, sparsification, and structural pruning. We further show how our
approach enables higher parallelism on existing hardware and seamlessly
supports caching and batching strategies to reduce overheads. Our prototype
demonstrates that leveraging LLM queries inside analytics systems is feasible
at scale, opening new possibilities for future OLAP applications.

### Distributed, Parallel, and Cluster Computing

### 1. [Communication Round and Computation Efficient Exclusive Prefix-Sums Algorithms (for MPI_Exscan)](http://arxiv.org/pdf/2507.04785v1)

Authors: Jesper Larsson Träff

Parallel scan primitives compute element-wise inclusive or exclusive prefix
sums of input vectors contributed by $p$ consecutively ranked processors under
an associative, binary operator $\oplus$. In message-passing systems with
bounded, one-ported communication capabilities, at least $\lceil\log_2 p\rceil$
or $\lceil\log_2 (p-1)\rceil$ communication rounds are required to perform the
scans. While there are well-known, simple algorithms for the inclusive scan
that solve the problem in $\lceil\log_2 p\rceil$ communication rounds with
$\lceil\log_2 p\rceil$ applications of $\oplus$ (which could be expensive), the
exclusive scan appears more difficult. Conventionally, the problem is solved
with either $\lceil\log_2 (p-1)\rceil+1$ communication rounds (e.g., by
shifting the input vectors), or in $\lceil\log_2 p\rceil$ communication rounds
with $2\lceil\log_2 p\rceil-1$ applications of $\oplus$ (by a modified
inclusive scan algorithm). We give a new, simple algorithm that computes the
exclusive prefix sums in $q=\lceil\log_2 (p-1)+\log_2\frac{4}{3}\rceil$
simultaneous send-receive communication rounds with $q-1$ applications of
$\oplus$. We compare the three algorithms implemented in MPI against the MPI
library native MPI\_Exscan primitive on a small, $36$-node cluster with a
state-of-the-art MPI library, indicating possible and worthwhile improvements
to standard implementations. The algorithms assume input vectors to be small so
that performance is dominated by the number of communication rounds. For large
input vectors, other (pipelined, fixed-degree tree) algorithms must be used.

### 2. [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](http://arxiv.org/pdf/2507.04786v1)

Authors: Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, Jeff Hammond, Torsten Hoefler

The NVIDIA Collective Communication Library (NCCL) is a critical software
layer enabling high-performance collectives on large-scale GPU clusters.
Despite being open source with a documented API, its internal design remains
largely opaque. The orchestration of communication channels, selection of
protocols, and handling of memory movement across devices and nodes are not
well understood, making it difficult to analyze performance or identify
bottlenecks. This paper presents a comprehensive analysis of NCCL, focusing on
its communication protocol variants (Simple, LL, and LL128), mechanisms
governing intra-node and inter-node data movement, and ring- and tree-based
collective communication algorithms. The insights obtained from this study
serve as the foundation for ATLAHS, an application-trace-driven network
simulation toolchain capable of accurately reproducing NCCL communication
patterns in large-scale AI training workloads. By demystifying NCCL's internal
architecture, this work provides guidance for system researchers and
performance engineers working to optimize or simulate collective communication
at scale.

### 3. [Silent Failures in Stateless Systems: Rethinking Anomaly Detection for Serverless Computing](http://arxiv.org/pdf/2507.04969v1)

Authors: Chanh Nguyen, Erik Elmroth, Monowar Bhuyan

Serverless computing has redefined cloud application deployment by
abstracting infrastructure and enabling on-demand, event-driven execution,
thereby enhancing developer agility and scalability. However, maintaining
consistent application performance in serverless environments remains a
significant challenge. The dynamic and transient nature of serverless functions
makes it difficult to distinguish between benign and anomalous behavior, which
in turn undermines the effectiveness of traditional anomaly detection methods.
These conventional approaches, designed for stateful and long-running services,
struggle in serverless settings where executions are short-lived, functions are
isolated, and observability is limited.
  In this first comprehensive vision paper on anomaly detection for serverless
systems, we systematically explore the unique challenges posed by this
paradigm, including the absence of persistent state, inconsistent monitoring
granularity, and the difficulty of correlating behaviors across distributed
functions. We further examine a range of threats that manifest as anomalies,
from classical Denial-of-Service (DoS) attacks to serverless-specific threats
such as Denial-of-Wallet (DoW) and cold start amplification. Building on these
observations, we articulate a research agenda for next-generation detection
frameworks that address the need for context-aware, multi-source data fusion,
real-time, lightweight, privacy-preserving, and edge-cloud adaptive
capabilities.
  Through the identification of key research directions and design principles,
we aim to lay the foundation for the next generation of anomaly detection in
cloud-native, serverless ecosystems.

### 4. [MoLink: Distributed and Efficient Serving Framework for Large Models](http://arxiv.org/pdf/2507.05043v1)

Authors: Lewei Jin, Yongqi Chen, Kui Zhang, Yifan Zhuo, Yi Gao, Bowei Yang, Zhengong Cai, Wei Dong

Large language models represent a groundbreaking shift in generative AI. Yet,
these advances come with a significant challenge: the high cost of model
serving. To mitigate these costs, consumer-grade GPUs emerge as a more
affordable alternative. This presents an opportunity for more cost-efficient
LLM serving by leveraging these GPUs.
  However, it is non-trivial to achieve high-efficiency LLM serving on
consumer-grade GPUs, mainly due to two challenges: 1) these GPUs are often
deployed in limited network conditions; 2) these GPUs often exhibit
heterogeneity in host systems. To address these challenges, we present MoLink,
a distributed LLM serving system for large models. It incorporates several key
techniques, enabling efficient LLM serving on heterogeneous and weakly
connected consumer-grade GPUs. Our experiments demonstrate that it achieves
throughput improvements of up to 458\% and cost-profit margin improvements of
up to 151\%, compared to state-of-the-art systems. MoLink allows users on
Windows, Linux, and containerized VMs to seamlessly integrate GPUs with just a
few lines of code over Ethernet or public networks. Currently, it supports 18
mainstream architectures of open-source large language models.

### 5. [Cooperative Gradient Coding](http://arxiv.org/pdf/2507.05230v1)

Authors: Shudi Weng, Ming Xiao, Chao Ren, Mikael Skoglund

This work studies gradient coding (GC) in the context of distributed training
problems with unreliable communication. We propose cooperative GC (CoGC), a
novel gradient-sharing-based GC framework that leverages cooperative
communication among clients. This approach ultimately eliminates the need for
dataset replication, making it both communication- and computation-efficient
and suitable for federated learning (FL). By employing the standard GC decoding
mechanism, CoGC yields strictly binary outcomes: either the global model is
exactly recovered, or the decoding fails entirely, with no intermediate
results. This characteristic ensures the optimality of the training and
demonstrates strong resilience to client-to-server communication failures when
the communication channels among clients are in good condition. However, it may
also result in communication inefficiency and hinder convergence due to its
lack of flexibility, especially when communication channels among clients are
in poor condition. To overcome this limitation and further harness the
potential of GC matrices, we propose a complementary decoding mechanism, termed
GC$^+$, which leverages information that would otherwise be discarded during GC
decoding failures. This approach significantly improves system reliability
under unreliable communication, as the full recovery of the global model
typically dominates in GC$^+$. To conclude, this work establishes solid
theoretical frameworks for both CoGC and GC$^+$. We provide complete outage
analyses for each decoding mechanism, along with a rigorous investigation of
how outages affect the structure and performance of GC matrices. Building on
these analyses, we derive convergence bounds for both decoding mechanisms.
Finally, the effectiveness of CoGC and GC$^+$ is validated through extensive
simulations.

### 6. [Bullshark on Narwhal: Implementation-level Workflow Analysis of Round-based DAG Consensus in Theory and Practice](http://arxiv.org/pdf/2507.04956v1)

Authors: Yusei Tanaka

Round-based DAGs enable high-performance Byzantine fault-tolerant consensus,
yet their technical advantages remain underutilized due to their short history.
While research on consensus protocols is active in both academia and industry,
many studies overlook implementation-level algorithms, leaving actual
performance unclear - particularly for theoretical protocols whose practical
performance cannot often be evaluated. Bullshark, a Round-based DAG BFT
protocol on Narwhal mempool, achieves optimal performance: 297,000 transactions
per second with 2-second latency. We analyze the algorithm's workflow, from
transaction submission to blockchain commitment, breaking it down layer by
layer at the functional level and delineating the key features and interactions
of the Bullshark and Narwhal components. Future work aims to improve
performance in Byzantine fault environments and optimize trade-offs in the CAP
theorem.

### 7. [Distributed Approximation Algorithms for Minimum Dominating Set in Locally Nice Graphs](http://arxiv.org/pdf/2507.04960v1)

Authors: Marthe Bonamy, Cyril Gavoille, Timothé Picavet, Alexandra Wesolek

We give a new, short proof that graphs embeddable in a given Euler genus-$g$
surface admit a simple $f(g)$-round $\alpha$-approximation distributed
algorithm for Minimum Dominating Set (MDS), where the approximation ratio
$\alpha \le 906$. Using tricks from Heydt et al. [European Journal of
Combinatorics (2025)], we in fact derive that $\alpha \le 34 +\varepsilon$,
therefore improving upon the current state of the art of $24g+O(1)$ due to
Amiri et al. [ACM Transactions on Algorithms (2019)]. It also improves the
approximation ratio of $91+\varepsilon$ due to Czygrinow et al. [Theoretical
Computer Science (2019)] in the particular case of orientable surfaces.
  All our distributed algorithms work in the deterministic LOCAL model. They do
not require any preliminary embedding of the graph and only rely on two things:
a LOCAL algorithm for MDS on planar graphs with ``uniform'' approximation
guarantees and the knowledge that graphs embeddable in bounded Euler genus
surfaces have asymptotic dimension $2$.
  More generally, our algorithms work in any graph class of bounded asymptotic
dimension where ``most vertices'' are locally in a graph class that admits a
LOCAL algorithm for MDS with uniform approximation guarantees.

### 8. [RAPTOR: Practical Numerical Profiling of Scientific Applications](http://arxiv.org/pdf/2507.04647v1)

Authors: Faveo Hoerold, Ivan R. Ivanov, Akash Dhruv, William S. Moses, Anshu Dubey, Mohamed Wahib, Jens Domke

The proliferation of low-precision units in modern high-performance
architectures increasingly burdens domain scientists. Historically, the choice
in HPC was easy: can we get away with 32 bit floating-point operations and
lower bandwidth requirements, or is FP64 necessary? Driven by Artificial
Intelligence, vendors introduced novel low-precision units for vector and
tensor operations, and FP64 capabilities stagnate or are reduced. This is
forcing scientists to re-evaluate their codes, but a trivial search-and-replace
approach to go from FP64 to FP16 will not suffice. We introduce RAPTOR: a
numerical profiling tool to guide scientists in their search for code regions
where precision lowering is feasible. Using LLVM, we transparently replace
high-precision computations using low-precision units, or emulate a
user-defined precision. RAPTOR is a novel, feature-rich approach -- with focus
on ease of use -- to change, profile, and reason about numerical requirements
and instabilities, which we demonstrate with four real-world multi-physics
Flash-X applications.

### 9. [Performance Evaluation of General Purpose Large Language Models for Basic Linear Algebra Subprograms Code Generation](http://arxiv.org/pdf/2507.04697v1)

Authors: Daichi Mukunoki, Shun-ichiro Hayashi, Tetsuya Hoshino, Takahiro Katagiri

Generative AI technology based on Large Language Models (LLM) has been
developed and applied to assist or automatically generate program codes. In
this paper, we evaluate the capability of existing general LLMs for Basic
Linear Algebra Subprograms (BLAS) code generation for CPUs. We use two LLMs
provided by OpenAI: GPT-4.1, a Generative Pre-trained Transformer (GPT) model,
and o4-mini, one of the o-series of Reasoning models. Both have been released
in April 2025. For the routines from level-1 to 3 BLAS, we tried to generate
(1) C code without optimization from routine name only, (2) C code with basic
performance optimizations (thread parallelization, SIMD vectorization, and
cache blocking) from routine name only, and (3) C code with basic performance
optimizations based on Fortran reference code. As a result, we found that
correct code can be generated in many cases even when only routine name are
given. We also confirmed that thread parallelization with OpenMP, SIMD
vectorization, and cache blocking can be implemented to some extent, and that
the code is faster than the reference code.

### 10. [BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning](http://arxiv.org/pdf/2507.04903v1)

Authors: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

Federated Learning (FL) systems are vulnerable to backdoor attacks, where
adversaries train their local models on poisoned data and submit poisoned model
updates to compromise the global model. Despite numerous proposed attacks and
defenses, divergent experimental settings, implementation errors, and
unrealistic assumptions hinder fair comparisons and valid conclusions about
their effectiveness in real-world scenarios. To address this, we introduce
BackFed - a comprehensive benchmark suite designed to standardize, streamline,
and reliably evaluate backdoor attacks and defenses in FL, with a focus on
practical constraints. Our benchmark offers key advantages through its
multi-processing implementation that significantly accelerates experimentation
and the modular design that enables seamless integration of new methods via
well-defined APIs. With a standardized evaluation pipeline, we envision BackFed
as a plug-and-play environment for researchers to comprehensively and reliably
evaluate new attacks and defenses. Using BackFed, we conduct large-scale
studies of representative backdoor attacks and defenses across both Computer
Vision and Natural Language Processing tasks with diverse model architectures
and experimental settings. Our experiments critically assess the performance of
proposed attacks and defenses, revealing unknown limitations and modes of
failures under practical conditions. These empirical insights provide valuable
guidance for the development of new methods and for enhancing the security of
FL systems. Our framework is openly available at
https://github.com/thinh-dao/BackFed.

### Discrete Mathematics

### 1. [Computing Expansions in Infinitely Many Cantor Real Bases via a Single Transducer](http://arxiv.org/pdf/2507.04848v1)

Authors: Émilie Charlier, Pierre Popoli, Michel Rigo

Representing real numbers using convenient numeration systems (integer bases,
$\beta$-numeration, Cantor bases, etc.) has been a longstanding mathematical
challenge. This paper focuses on Cantor real bases and, specifically, on
automatic Cantor real bases and the properties of expansions of real numbers in
this setting. We develop a new approach where a single transducer associated
with a fixed real number $r$, computes the $\mathbf{B}$-expansion of $r$ but
for an infinite family of Cantor real bases $\mathbf{B}$ given as input. This
point of view contrasts with traditional computational models for which the
numeration system is fixed. Under some assumptions on the finitely many Pisot
numbers occurring in the Cantor real base, we show that only a finite part of
the transducer is visited. We obtain fundamental results on the structure of
this transducer and on decidability problems about these expansions, proving
that for certain classes of Cantor real bases, key combinatorial properties
such as greediness of the expansion or periodicity can be decided
algorithmically.

### Data Structures and Algorithms

### 1. [Improved Algorithms for Effective Resistance Computation on Graphs](http://arxiv.org/pdf/2507.04674v1)

Authors: Yichun Yang, Rong-Hua Li, Meihao Liao, Guoren Wang

Effective Resistance (ER) is a fundamental tool in various graph learning
tasks. In this paper, we address the problem of efficiently approximating ER on
a graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ with $n$ vertices and $m$
edges. First, we focus on local online-computation algorithms for ER
approximation, aiming to improve the dependency on the approximation error
parameter $\epsilon$. Specifically, for a given vertex pair $(s,t)$, we propose
a local algorithm with a time complexity of $\tilde{O}(\sqrt{d}/\epsilon)$ to
compute an $\epsilon$-approximation of the $s,t$-ER value for expander graphs,
where $d=\min \{d_s,d_t\}$. This improves upon the previous state-of-the-art,
including an $\tilde{O}(1/\epsilon^2)$ time algorithm based on random walk
sampling by Andoni et al. (ITCS'19) and Peng et al. (KDD'21). Our method
achieves this improvement by combining deterministic search with random walk
sampling to reduce variance. Second, we establish a lower bound for ER
approximation on expander graphs. We prove that for any $\epsilon\in (0,1)$,
there exist an expander graph and a vertex pair $(s,t)$ such that any local
algorithm requires at least $\Omega(1/\epsilon)$ time to compute the
$\epsilon$-approximation of the $s,t$-ER value. Finally, we extend our
techniques to index-based algorithms for ER computation. We propose an
algorithm with $\tilde{O}(\min \{m+n/\epsilon^{1.5},\sqrt{nm}/\epsilon\})$
processing time, $\tilde{O}(n/\epsilon)$ space complexity and $O(1)$ query
complexity, which returns an $\epsilon$-approximation of the $s,t$-ER value for
any $s,t\in \mathcal{V}$ for expander graphs. Our approach improves upon the
state-of-the-art $\tilde{O}(m/\epsilon)$ processing time by Dwaraknath et al.
(NeurIPS'24) and the $\tilde{O}(m+n/\epsilon^2)$ processing time by Li and
Sachdeva (SODA'23).

### 2. [Truthful, Credible, and Optimal Auctions for Matroids via Blockchains and Commitments](http://arxiv.org/pdf/2507.04592v1)

Authors: Aadityan Ganesh, Qianfan Zhang

We consider a revenue-optimizing auctioneer in single-dimensional
environments with matroid feasibility constraints. Akbarpour and Li (2020)
argue that any revenue-optimal, truthful, and credible mechanism requires
unbounded communication. Recent works (Ferreira and Weinberg, 2020; Essaidi et
al., 2022; Chitra et al., 2024) circumvent their impossibility for the
single-item setting through the use of cryptographic commitments and
blockchains. We extend their results to matroid feasibility constraints.
  At a high level, the two-round Deferred-Revelation Auction (DRA) discussed by
Ferreira and Weinberg (2020) and Chitra et al., (2024) requires each bidder to
submit a deposit, which is slashed upon presenting verifiable evidence
indicating a deviation from the behaviour prescribed by the mechanism. We prove
that the DRA satisfies truthfulness, credibility and revenue-optimality for all
matroid environments when bidders' values are drawn from $\alpha$-strongly
regular distributions for $\alpha > 0$. Further, we argue that the DRA is not
credible for any feasibility constraint beyond matroids and for any smaller
deposits than suggested by previous literature even in single-item
environments.
  Finally, we modify the Ascending Deferred-Revelation Auction (ADRA) for
single-item settings proposed by Essaidi et al., (2022) for arbitrary bidder
value distributions. We implement a deferred-revelation variant of the
deferred-acceptance auction for matroids due to Bikhchandani et al., (2011),
which requires the same bounded communication as the ADRA.

### 3. [Liar's vertex-edge domination in subclasses of chordal graphs](http://arxiv.org/pdf/2507.04721v1)

Authors: Debojyoti Bhattacharya, Subhabrata Paul

Let $G=(V, E)$ be an undirected graph. The set $N_G[x]=\{y\in V|xy\in E\}\cup
\{x\}$ is called the closed neighbourhood of a vertex $x\in V$ and for an edge
$e=xy\in E$, the closed neighbourhood of $e$ is the set $N_G[x]\cup N_G[y]$,
which is denoted by $N_G[e]$ or $N_G[xy]$. A set $L\subseteq V$ is called
\emph{liar's vertex-edge dominating set} of a graph $G=(V,E)$ if for every
$e_i\in E$, $|N_G[e_i]\cap L|\geq 2$ and for every pair of distinct edges
$e_i,e_j\in E$, $|(N_G[e_i]\cup N_G[e_j])\cap L|\geq 3$. The notion of liar's
vertex-edge domination arises naturally from some applications in communication
networks. Given a graph $G$, the \textsc{Minimum Liar's Vertex-Edge Domination
Problem} (\textsc{MinLVEDP}) asks to find a liar's vertex-edge dominating set
of $G$ of minimum cardinality. In this paper, we study this problem from an
algorithmic point of view. We design two linear time algorithms for
\textsc{MinLVEDP} in block graphs and proper interval graphs, respectively. On
the negative side, we show that the decision version of liar's vertex-edge
domination problem is NP-complete for undirected path graphs.

### 4. [Distributed Approximation Algorithms for Minimum Dominating Set in Locally Nice Graphs](http://arxiv.org/pdf/2507.04960v1)

Authors: Marthe Bonamy, Cyril Gavoille, Timothé Picavet, Alexandra Wesolek

We give a new, short proof that graphs embeddable in a given Euler genus-$g$
surface admit a simple $f(g)$-round $\alpha$-approximation distributed
algorithm for Minimum Dominating Set (MDS), where the approximation ratio
$\alpha \le 906$. Using tricks from Heydt et al. [European Journal of
Combinatorics (2025)], we in fact derive that $\alpha \le 34 +\varepsilon$,
therefore improving upon the current state of the art of $24g+O(1)$ due to
Amiri et al. [ACM Transactions on Algorithms (2019)]. It also improves the
approximation ratio of $91+\varepsilon$ due to Czygrinow et al. [Theoretical
Computer Science (2019)] in the particular case of orientable surfaces.
  All our distributed algorithms work in the deterministic LOCAL model. They do
not require any preliminary embedding of the graph and only rely on two things:
a LOCAL algorithm for MDS on planar graphs with ``uniform'' approximation
guarantees and the knowledge that graphs embeddable in bounded Euler genus
surfaces have asymptotic dimension $2$.
  More generally, our algorithms work in any graph class of bounded asymptotic
dimension where ``most vertices'' are locally in a graph class that admits a
LOCAL algorithm for MDS with uniform approximation guarantees.

### 5. [Recent Advances in Maximum-Entropy Sampling](http://arxiv.org/pdf/2507.05066v1)

Authors: Marcia Fampa, Jon Lee

In 2022, we published a book, \emph{Maximum-Entropy Sampling: Algorithms and
Application (Springer)}. Since then, there have been several notable
advancements on this topic. In this manuscript, we survey some recent
highlights.

### Emerging Technologies

### 1. [Optimized Bistable Vortex Memory Arrays for Superconducting In-Memory Matrix-Vector Multiplication](http://arxiv.org/pdf/2507.04648v1)

Authors: Mustafa Altay Karamuftuoglu, Changxu Song, Beyza Zeynep Ucpinar, Sasan Razmkhah, Massoud Pedram

Building upon previously introduced Bistable Vortex Memory (BVM) as a novel,
nonvolatile, high-density, and scalable superconductor memory technology, this
work presents a methodology that uses BVM arrays to address challenges in
data-driven algorithms and neural networks, specifically focusing on
matrix-vector multiplication (MVM). The BVM approach introduces a novel
superconductor-based methodology for in-memory arithmetic, achieving
ultra-high-speed and energy-efficient computation by utilizing BVM arrays for
in-memory computation. The design employs a tiled multiplier structure where
BVM's inherent current summation capability is combined with Quantizer Buffer
(QB) cells to convert the analog accumulated current into a variable number of
digital Single Flux Quantum (SFQ) pulses. These pulses are then processed by T1
adder cells, which handle binary addition and carry propagation, thereby
forming a complete functional multiplier unit. This paper thus presents an
efficient MVM architecture that uses these BVM-based multipliers in a systolic
array configuration to enable parallel computation. A key innovation is an
optimized BVM array structure specifically tailored for multiplication
applications, involving a restructuring of Sense Lines (SLs) with diagonal
connections to reduce area and an adjusted input scheme to enhance
computational efficiency compared to the general-purpose BVM array design. We
demonstrate the efficacy of this approach with a 4-bit multiplier operating at
20 GHz with 50 ps latency and an MVM structure demonstrating operation at 20
GHz. Furthermore, we showcase how this multiplier design can be extended to
support Multiply-Accumulate (MAC) operations. This work paves the way for
power-efficient neural networks by enabling high-speed in-memory computation.

### 2. [Enabling Security on the Edge: A CHERI Compartmentalized Network Stack](http://arxiv.org/pdf/2507.04818v1)

Authors: Donato Ferraro, Andrea Bastoni, Alexander Zuepke, Andrea Marongiu

The widespread deployment of embedded systems in critical infrastructures,
interconnected edge devices like autonomous drones, and smart industrial
systems requires robust security measures. Compromised systems increase the
risks of operational failures, data breaches, and -- in safety-critical
environments -- potential physical harm to people. Despite these risks, current
security measures are often insufficient to fully address the attack surfaces
of embedded devices. CHERI provides strong security from the hardware level by
enabling fine-grained compartmentalization and memory protection, which can
reduce the attack surface and improve the reliability of such devices. In this
work, we explore the potential of CHERI to compartmentalize one of the most
critical and targeted components of interconnected systems: their network
stack. Our case study examines the trade-offs of isolating applications, TCP/IP
libraries, and network drivers on a CheriBSD system deployed on the Arm Morello
platform. Our results suggest that CHERI has the potential to enhance security
while maintaining performance in embedded-like environments.

### 3. [DYNAMO: Dynamic Neutral Atom Multi-programming Optimizer Towards Quantum Operating Systems](http://arxiv.org/pdf/2507.04874v1)

Authors: Wenjie Sun, Xiaoyu Li, Zhigang Wang, Geng Chen, Lianhui Yu, Guowu Yang

As quantum computing advances towards practical applications, quantum
operating systems become inevitable, where multi-programming -- the core
functionality of operating systems -- enables concurrent execution of multiple
quantum programs to enhance hardware utilization. However, most quantum
compilation work focuses solely on single-circuit execution, severely limiting
resource efficiency and hindering quantum operating system development. We
propose Dynamic Neutral Atom Multi-programming Optimizer (DYNAMO), a method
that realizes multi-programming on neutral atom quantum architectures through
parallel compilation and intelligent resource allocation across multiple
quantum processing units (QPUs). DYNAMO addresses two critical challenges:
inefficient and difficult resource partitioning, and complex scheduling
conflicts from concurrent program. Our method enables efficient spatial and
temporal resource sharing while maintaining circuit correctness and hardware
constraints. Experimental evaluation across circuits ranging from 12 to over
1200 gates demonstrates that DYNAMO achieves up to 14.39x compilation speedup
while reducing execution stages by an average of 50.47%. Furthermore, DYNAMO
successfully distributes workloads across multiple QPUs with balanced resource
utilization. By enabling efficient multi-programming capabilities, DYNAMO
establishes a critical foundation towards realizing practical quantum operating
systems.

### Formal Languages and Automata Theory

### 1. [A Note on Runtime Verification of Concurrent Systems](http://arxiv.org/pdf/2507.04830v1)

Authors: Martin Leucker

To maximize the information gained from a single execution when verifying a
concurrent system, one can derive all concurrency-aware equivalent executions
and check them against linear specifications. This paper offers an alternative
perspective on verification of concurrent systems by leveraging trace-based
logics rather than sequence-based formalisms. Linear Temporal Logic over
Mazurkiewicz Traces (LTrL) operates on partial-order representations of
executions, meaning that once a single execution is specified, all equivalent
interleavings are implicitly considered. This paper introduces a three valued
version of LTrL, indicating whether the so-far observed execution of the
concurrent system is one of correct, incorrect or inconclusive, together with a
suitable monitor synthesis procedure. To this end, the paper recalls a
construction of trace-consistent B\"uchi automata for LTrL formulas and
explains how to employ it in well-understood monitor synthesis procedures. In
this way, a monitor results that yields for any linearization of an observed
trace the same verification verdict.

### Graphics

### 1. [Neuralocks: Real-Time Dynamic Neural Hair Simulation](http://arxiv.org/pdf/2507.05191v1)

Authors: Gene Wei-Chin Lin, Egor Larionov, Hsiao-yu Chen, Doug Roble, Tuur Stuyck

Real-time hair simulation is a vital component in creating believable virtual
avatars, as it provides a sense of immersion and authenticity. The dynamic
behavior of hair, such as bouncing or swaying in response to character
movements like jumping or walking, plays a significant role in enhancing the
overall realism and engagement of virtual experiences. Current methods for
simulating hair have been constrained by two primary approaches: highly
optimized physics-based systems and neural methods. However, state-of-the-art
neural techniques have been limited to quasi-static solutions, failing to
capture the dynamic behavior of hair. This paper introduces a novel neural
method that breaks through these limitations, achieving efficient and stable
dynamic hair simulation while outperforming existing approaches. We propose a
fully self-supervised method which can be trained without any manual
intervention or artist generated training data allowing the method to be
integrated with hair reconstruction methods to enable automatic end-to-end
methods for avatar reconstruction. Our approach harnesses the power of compact,
memory-efficient neural networks to simulate hair at the strand level, allowing
for the simulation of diverse hairstyles without excessive computational
resources or memory requirements. We validate the effectiveness of our method
through a variety of hairstyle examples, showcasing its potential for
real-world applications.

### Computer Science and Game Theory

### 1. [A number game reconciliation](http://arxiv.org/pdf/2507.04717v1)

Authors: Prem Kant, Urban Larsson

Number games play a central role in alternating normal play combinatorial
game theory due to their real-number-like properties (Conway 1976). Here we
undertake a critical re-examination: we begin with integer and dyadic games and
identify subtle inconsistencies and oversights in the established literature
(e.g. Siegel 2013), most notably, the lack of distinction between a game being
a number and a game being equal to a number. After addressing this, we move to
the general theory of number games. We analyze Conway's original definition and
a later refinement by Siegel, and highlight conceptual gaps that have largely
gone unnoticed. Through a careful dissection of these issues, we propose a more
coherent and robust formulation. Specifically, we develop a refined
characterization of numbers, via several subclasses, dyadics, canonical forms,
their group theoretic closure and zugzwangs, that altogether better capture the
essence of number games. This reconciliation not only clarifies existing
ambiguities but also uncovers several open problems.

### 2. [Vector Cost Bimatrix Games with Applications to Autonomous Racing](http://arxiv.org/pdf/2507.05171v1)

Authors: Benjamin R. Toaz, Shaunak D. Bopardikar

We formulate a vector cost alternative to the scalarization method for
weighting and combining multi-objective costs. The algorithm produces solutions
to bimatrix games that are simultaneously pure, unique Nash equilibria and
Pareto optimal with guarantees for avoiding worst case outcomes. We achieve
this by enforcing exact potential game constraints to guide cost adjustments
towards equilibrium, while minimizing the deviation from the original cost
structure. The magnitude of this adjustment serves as a metric for
differentiating between Pareto optimal solutions. We implement this approach in
a racing competition between agents with heterogeneous cost structures,
resulting in fewer collision incidents with a minimal decrease in performance.
Code is available at https://github.com/toazbenj/race_simulation.

### 3. [Truthful, Credible, and Optimal Auctions for Matroids via Blockchains and Commitments](http://arxiv.org/pdf/2507.04592v1)

Authors: Aadityan Ganesh, Qianfan Zhang

We consider a revenue-optimizing auctioneer in single-dimensional
environments with matroid feasibility constraints. Akbarpour and Li (2020)
argue that any revenue-optimal, truthful, and credible mechanism requires
unbounded communication. Recent works (Ferreira and Weinberg, 2020; Essaidi et
al., 2022; Chitra et al., 2024) circumvent their impossibility for the
single-item setting through the use of cryptographic commitments and
blockchains. We extend their results to matroid feasibility constraints.
  At a high level, the two-round Deferred-Revelation Auction (DRA) discussed by
Ferreira and Weinberg (2020) and Chitra et al., (2024) requires each bidder to
submit a deposit, which is slashed upon presenting verifiable evidence
indicating a deviation from the behaviour prescribed by the mechanism. We prove
that the DRA satisfies truthfulness, credibility and revenue-optimality for all
matroid environments when bidders' values are drawn from $\alpha$-strongly
regular distributions for $\alpha > 0$. Further, we argue that the DRA is not
credible for any feasibility constraint beyond matroids and for any smaller
deposits than suggested by previous literature even in single-item
environments.
  Finally, we modify the Ascending Deferred-Revelation Auction (ADRA) for
single-item settings proposed by Essaidi et al., (2022) for arbitrary bidder
value distributions. We implement a deferred-revelation variant of the
deferred-acceptance auction for matroids due to Bikhchandani et al., (2011),
which requires the same bounded communication as the ADRA.

### Human-Computer Interaction

### 1. [Using Psychophysiological Insights to Evaluate the Impact of Loot Boxes on Arousal](http://arxiv.org/pdf/2507.04906v1)

Authors: Gianmarco Tedeschi, Rune Kristian Lundedal Nielsen, Paolo Burelli

This study investigates the psychophysiological effects of loot box
interactions in video games and their potential similarities to those recorded
during gambling interactions. Using electrodermal activity (EDA) measurements,
the research examines player arousal during loot box interactions and explores
the relationship between Internet Gaming Disorder (IGD) severity and loot box
interactions from a psychophysiological perspective. The study employs a
custom-designed game to control experimental conditions and standardise loot
box interactions. Participants' IGD severity is assessed using the Internet
Gaming Disorder Scale - Short Form (IGDS9-SF), while arousal is measured
through EDA, analysing both tonic and phasic components. The study contributes
to the ongoing debate surrounding gaming disorder and loot boxes, offering
insights for game developers and policymakers on the potential risks associated
with random reward mechanisms in video games.

### 2. [Cat Royale: An Artistic Inquiry into Trust in Robots](http://arxiv.org/pdf/2507.04970v1)

Authors: Matt Adams, Nick Tandavanitj, Steve Benford, Ayse Kucukyilmaz, Victor Ngo, Simon Castle-Green, Guido Salimberi, Pepita Bernard, Joel Fischer, Alan Chamberlain, Eike Schneiders, Clara Mancini

Cat Royale is an artwork created by the artists Blast Theory to explore the
question of whether we should trust robots to care for our loved ones. The
artists endeavoured to create a `Cat Utopia', a luxurious environment that was
inhabited by a family of three cats for six hours a day for twelve days, at the
centre of which a robot arm played with them by wielding toys. Behind the
scenes, the decision engine recommended games based on ongoing assessment of
their happiness. A video installation featuring an eight-hour movie of the
cats' exploits is currently touring worldwide, provoking audiences to engage
with the question of trust in autonomous systems.

### 3. [What Shapes User Trust in ChatGPT? A Mixed-Methods Study of User Attributes, Trust Dimensions, Task Context, and Societal Perceptions among University Students](http://arxiv.org/pdf/2507.05046v1)

Authors: Kadija Bouyzourn, Alexandra Birch

This mixed-methods inquiry examined four domains that shape university
students' trust in ChatGPT: user attributes, seven delineated trust dimensions,
task context, and perceived societal impact. Data were collected through a
survey of 115 UK undergraduate and postgraduate students and four complementary
semi-structured interviews. Behavioural engagement outweighed demographics:
frequent use increased trust, whereas self-reported understanding of
large-language-model mechanics reduced it. Among the dimensions, perceived
expertise and ethical risk were the strongest predictors of overall trust; ease
of use and transparency had secondary effects, while human-likeness and
reputation were non-significant. Trust was highly task-contingent; highest for
coding and summarising, lowest for entertainment and citation generation, yet
confidence in ChatGPT's referencing ability, despite known inaccuracies, was
the single strongest correlate of global trust, indicating automation bias.
Computer-science students surpassed peers only in trusting the system for
proofreading and writing, suggesting technical expertise refines rather than
inflates reliance. Finally, students who viewed AI's societal impact positively
reported the greatest trust, whereas mixed or negative outlooks dampened
confidence. These findings show that trust in ChatGPT hinges on task
verifiability, perceived competence, ethical alignment and direct experience,
and they underscore the need for transparency, accuracy cues and user education
when deploying LLMs in academic settings.

### 4. [Infrastructuring Contestability: A Framework for Community-Defined AI Value Pluralism](http://arxiv.org/pdf/2507.05187v1)

Authors: Andreas Mayer

The proliferation of AI-driven systems presents a fundamental challenge to
Human-Computer Interaction (HCI) and Computer-Supported Cooperative Work
(CSCW), often diminishing user agency and failing to account for value
pluralism. Current approaches to value alignment, which rely on centralized,
top-down definitions, lack the mechanisms for meaningful contestability. This
leaves users and communities unable to challenge or shape the values embedded
in the systems that govern their digital lives, creating a crisis of legitimacy
and trust. This paper introduces Community-Defined AI Value Pluralism (CDAVP),
a socio-technical framework that addresses this gap. It reframes the design
problem from achieving a single aligned state to infrastructuring a dynamic
ecosystem for value deliberation and application. At its core, CDAVP enables
diverse, self-organizing communities to define and maintain explicit value
profiles - rich, machine-readable representations that can encompass not only
preferences but also community-specific rights and duties. These profiles are
then contextually activated by the end-user, who retains ultimate control
(agency) over which values guide the AI's behavior. AI applications, in turn,
are designed to transparently interpret these profiles and moderate conflicts,
adhering to a set of non-negotiable, democratically-legitimated meta-rules. The
designer's role shifts from crafting static interfaces to becoming an architect
of participatory ecosystems. We argue that infrastructuring for pluralism is a
necessary pathway toward achieving robust algorithmic accountability and
genuinely contestable, human-centric AI.

### 5. [Perspectives on How Sociology Can Advance Theorizing about Human-Chatbot Interaction and Developing Chatbots for Social Good](http://arxiv.org/pdf/2507.05030v1)

Authors: Celeste Campos-Castillo, Xuan Kang, Linnea I. Laestadius

Recently, research into chatbots (also known as conversational agents, AI
agents, voice assistants), which are computer applications using artificial
intelligence to mimic human-like conversation, has grown sharply. Despite this
growth, sociology lags other disciplines (including computer science, medicine,
psychology, and communication) in publishing about chatbots. We suggest
sociology can advance understanding of human-chatbot interaction and offer four
sociological theories to enhance extant work in this field. The first two
theories (resource substitution theory, power-dependence theory) add new
insights to existing models of the drivers of chatbot use, which overlook
sociological concerns about how social structure (e.g., systemic
discrimination, the uneven distribution of resources within networks) inclines
individuals to use chatbots, including problematic levels of emotional
dependency on chatbots. The second two theories (affect control theory,
fundamental cause of disease theory) help inform the development of
chatbot-driven interventions that minimize safety risks and enhance equity by
leveraging sociological insights into how chatbot outputs could attend to
cultural contexts (e.g., affective norms) to promote wellbeing and enhance
communities (e.g., opportunities for civic participation). We discuss the value
of applying sociological theories for advancing theorizing about human-chatbot
interaction and developing chatbots for social good.

### 6. [From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems](http://arxiv.org/pdf/2507.04996v1)

Authors: Jiangbo Yu

Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity
to operate according to internal rules without external control. Accordingly,
autonomous vehicles (AuVs) are defined as systems capable of perceiving their
environment and executing preprogrammed tasks independently of external input.
However, both research and real-world deployments increasingly showcase
vehicles that demonstrate behaviors beyond this definition (including the SAE
levels 1 to 6), such as interaction with humans and machines, goal adaptation,
contextual reasoning, external tool use, and long-term planning, particularly
with the integration of large language models (LLMs) and agentic AI systems.
These developments reveal a conceptual gap between technical autonomy and the
broader cognitive and social capabilities needed for future human-centered
mobility systems. To address this, we introduce the concept of agentic vehicles
(AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and
interact within complex environments. This paper presents a systems-level
framework to characterize AgVs, focusing on their cognitive and communicative
layers and differentiating them from conventional AuVs. It synthesizes relevant
advances in agentic AI, robotics, multi-agent systems, and human-machine
interaction, and highlights how agentic AI, through high-level reasoning and
tool use, can function not merely as computational tools but as interactive
agents embedded in mobility ecosystems. The paper concludes by identifying key
challenges in the development and governance of AgVs, including safety,
real-time control, public acceptance, ethical alignment, and regulatory
frameworks.

### Information Retrieval

### 1. [Heterogeneous User Modeling for LLM-based Recommendation](http://arxiv.org/pdf/2507.04626v1)

Authors: Honghui Bao, Wenjie Wang, Xinyu Lin, Fengbin Zhu, Teng Sun, Fuli Feng, Tat-Seng Chua

Leveraging Large Language Models (LLMs) for recommendation has demonstrated
notable success in various domains, showcasing their potential for open-domain
recommendation. A key challenge to advancing open-domain recommendation lies in
effectively modeling user preferences from users' heterogeneous behaviors
across multiple domains. Existing approaches, including ID-based and
semantic-based modeling, struggle with poor generalization, an inability to
compress noisy interactions effectively, and the domain seesaw phenomenon. To
address these challenges, we propose a Heterogeneous User Modeling (HUM)
method, which incorporates a compression enhancer and a robustness enhancer for
LLM-based recommendation. The compression enhancer uses a customized prompt to
compress heterogeneous behaviors into a tailored token, while a masking
mechanism enhances cross-domain knowledge extraction and understanding. The
robustness enhancer introduces a domain importance score to mitigate the domain
seesaw phenomenon by guiding domain optimization. Extensive experiments on
heterogeneous datasets validate that HUM effectively models user heterogeneity
by achieving both high efficacy and robustness, leading to superior performance
in open-domain recommendation.

### 2. [FindRec: Stein-Guided Entropic Flow for Multi-Modal Sequential Recommendation](http://arxiv.org/pdf/2507.04651v1)

Authors: Maolin Wang, Yutian Xiao, Binhao Wang, Sheng Zhang, Shanshan Ye, Wanyu Wang, Hongzhi Yin, Ruocheng Guo, Zenglin Xu

Modern recommendation systems face significant challenges in processing
multimodal sequential data, particularly in temporal dynamics modeling and
information flow coordination. Traditional approaches struggle with
distribution discrepancies between heterogeneous features and noise
interference in multimodal signals. We propose \textbf{FindRec}~
(\textbf{F}lexible unified \textbf{in}formation \textbf{d}isentanglement for
multi-modal sequential \textbf{Rec}ommendation), introducing a novel
"information flow-control-output" paradigm. The framework features two key
innovations: (1) A Stein kernel-based Integrated Information Coordination
Module (IICM) that theoretically guarantees distribution consistency between
multimodal features and ID streams, and (2) A cross-modal expert routing
mechanism that adaptively filters and combines multimodal features based on
their contextual relevance. Our approach leverages multi-head subspace
decomposition for routing stability and RBF-Stein gradient for unbiased
distribution alignment, enhanced by linear-complexity Mamba layers for
efficient temporal modeling. Extensive experiments on three real-world datasets
demonstrate FindRec's superior performance over state-of-the-art baselines,
particularly in handling long sequences and noisy multimodal inputs. Our
framework achieves both improved recommendation accuracy and enhanced model
interpretability through its modular design. The implementation code is
available anonymously online for easy
reproducibility~\footnote{https://github.com/Applied-Machine-Learning-Lab/FindRec}.

### 3. [Harnessing Pairwise Ranking Prompting Through Sample-Efficient Ranking Distillation](http://arxiv.org/pdf/2507.04820v1)

Authors: Junru Wu, Le Yan, Zhen Qin, Honglei Zhuang, Paul Suganthan G. C., Tianqi Liu, Zhe Dong, Xuanhui Wang, Harrie Oosterhuis

While Pairwise Ranking Prompting (PRP) with Large Language Models (LLMs) is
one of the most effective zero-shot document ranking methods, it has a
quadratic computational complexity with respect to the number of documents to
be ranked, as it requires an enumeration over all possible document pairs.
Consequently, the outstanding ranking performance of PRP has remained
unreachable for most real-world ranking applications.
  In this work, we propose to harness the effectiveness of PRP through pairwise
distillation. Specifically, we distill a pointwise student ranker from pairwise
teacher labels generated by PRP, resulting in an efficient student model that
retains the performance of PRP with substantially lower computational costs.
Furthermore, we find that the distillation process can be made
sample-efficient: with only 2% of pairs, we are able to obtain the same
performance as using all pairs for teacher labels. Thus, our novel approach
provides a solution to harness the ranking performance of PRP without incurring
high computational costs during both distillation and serving.

### 4. [SimLab: A Platform for Simulation-based Evaluation of Conversational Information Access Systems](http://arxiv.org/pdf/2507.04888v1)

Authors: Nolwenn Bernard, Sharath Chandra Etagi Suresh, Krisztian Balog, ChengXiang Zhai

Research on interactive and conversational information access systems,
including search engines, recommender systems, and conversational assistants,
has been hindered by the difficulty in evaluating such systems with
reproducible experiments. User simulation provides a promising solution, but
there is a lack of infrastructure and tooling to support this kind of
evaluation. To facilitate simulation-based evaluation of conversational
information access systems, we introduce SimLab, the first cloud-based platform
to provide a centralized general solution for the community to benchmark both
conversational systems and user simulators in a controlled and reproducible
environment. We articulate requirements for such a platform and propose a
general infrastructure to address these requirements. We then present the
design and implementation of an initial version of SimLab and showcase its
features with an initial evaluation task of conversational movie
recommendation, which is made publicly available. Furthermore, we discuss the
sustainability of the platform and its future opportunities. This paper is a
call for the community to contribute to the platform to drive progress in the
field of conversational information access and user simulation.

### 5. [Hierarchical Intent-guided Optimization with Pluggable LLM-Driven Semantics for Session-based Recommendation](http://arxiv.org/pdf/2507.04623v1)

Authors: Jinpeng Chen, Jianxiang He, Huan Li, Senzhang Wang, Yuan Cao, Kaimin Wei, Zhenye Yang, Ye Ji

Session-based Recommendation (SBR) aims to predict the next item a user will
likely engage with, using their interaction sequence within an anonymous
session. Existing SBR models often focus only on single-session information,
ignoring inter-session relationships and valuable cross-session insights. Some
methods try to include inter-session data but struggle with noise and
irrelevant information, reducing performance. Additionally, most models rely on
item ID co-occurrence and overlook rich semantic details, limiting their
ability to capture fine-grained item features. To address these challenges, we
propose a novel hierarchical intent-guided optimization approach with pluggable
LLM-driven semantic learning for session-based recommendations, called HIPHOP.
First, we introduce a pluggable embedding module based on large language models
(LLMs) to generate high-quality semantic representations, enhancing item
embeddings. Second, HIPHOP utilizes graph neural networks (GNNs) to model item
transition relationships and incorporates a dynamic multi-intent capturing
module to address users' diverse interests within a session. Additionally, we
design a hierarchical inter-session similarity learning module, guided by user
intent, to capture global and local session relationships, effectively
exploring users' long-term and short-term interests. To mitigate noise, an
intent-guided denoising strategy is applied during inter-session learning.
Finally, we enhance the model's discriminative capability by using contrastive
learning to optimize session representations. Experiments on multiple datasets
show that HIPHOP significantly outperforms existing methods, demonstrating its
effectiveness in improving recommendation quality. Our code is available:
https://github.com/hjx159/HIPHOP.

### 6. ["This Suits You the Best": Query Focused Comparative Explainable Summarization](http://arxiv.org/pdf/2507.04733v1)

Authors: Arnav Attri, Anuj Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera

Product recommendations inherently involve comparisons, yet traditional
opinion summarization often fails to provide holistic comparative insights. We
propose the novel task of generating Query-Focused Comparative Explainable
Summaries (QF-CES) using Multi-Source Opinion Summarization (M-OS). To address
the lack of query-focused recommendation datasets, we introduce MS-Q2P,
comprising 7,500 queries mapped to 22,500 recommended products with metadata.
We leverage Large Language Models (LLMs) to generate tabular comparative
summaries with query-specific explanations. Our approach is personalized,
privacy-preserving, recommendation engine-agnostic, and category-agnostic. M-OS
as an intermediate step reduces inference latency approximately by 40% compared
to the direct input approach (DIA), which processes raw data directly. We
evaluate open-source and proprietary LLMs for generating and assessing QF-CES.
Extensive evaluations using QF-CES-PROMPT across 5 dimensions (clarity,
faithfulness, informativeness, format adherence, and query relevance) showed an
average Spearman correlation of 0.74 with human judgments, indicating its
potential for QF-CES evaluation.

### 7. [SIGIR 2025 -- LiveRAG Challenge Report](http://arxiv.org/pdf/2507.04942v1)

Authors: David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Oren Somekh, Ran Tavory

The LiveRAG Challenge at SIGIR 2025, held between March and May 2025,
provided a competitive platform for advancing Retrieval-Augmented Generation
(RAG) technologies. Participants from academia and industry were invited to
develop a RAG-based question-answering system using a fixed corpus
(Fineweb-10BT) and a common open-source LLM (Falcon3-10B-Instruct). The goal
was to facilitate challenging comparisons of retrieval and prompting
strategies. During the Live Challenge Day, 70 teams from 27 different countries
provided answers and supportive information to 500 unseen questions within a
strict two-hour time window. Evaluation was conducted in two stages: first an
automated LLM-as-a-judge approach was used to compute correctness and
faithfulness score, then a manual review of top ranked submissions was
conducted. The finalists were announced on June 12, 2025, with prizes awarded
during the LiveRAG Workshop at SIGIR 2025 in Padua, Italy.

### 8. [Interest Networks (iNETs) for Cities: Cross-Platform Insights and Urban Behavior Explanations](http://arxiv.org/pdf/2507.04995v1)

Authors: Gustavo H. Santos, Myriam Delgado, Thiago H. Silva

Location-Based Social Networks (LBSNs) provide a rich foundation for modeling
urban behavior through iNETs (Interest Networks), which capture how user
interests are distributed throughout urban spaces. This study compares iNETs
across platforms (Google Places and Foursquare) and spatial granularities,
showing that coarser levels reveal more consistent cross-platform patterns,
while finer granularities expose subtle, platform-specific behaviors. Our
analysis finds that, in general, user interest is primarily shaped by
geographic proximity and venue similarity, while socioeconomic and political
contexts play a lesser role. Building on these insights, we develop a
multi-level, explainable recommendation system that predicts high-interest
urban regions for different user types. The model adapts to behavior profiles
-- such as explorers, who are driven by proximity, and returners, who prefer
familiar venues -- and provides natural-language explanations using explainable
AI (XAI) techniques. To support our approach, we introduce h3-cities, a tool
for multi-scale spatial analysis, and release a public demo for interactively
exploring personalized urban recommendations. Our findings contribute to urban
mobility research by providing scalable, context-aware, and interpretable
recommendation systems.

### 9. [Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search](http://arxiv.org/pdf/2507.05006v1)

Authors: Matteo Attimonelli, Alessandro De Bellis, Claudio Pomo, Dietmar Jannach, Eugenio Di Sciascio, Tommaso Di Noia

Pre-trained language models (PLMs) are widely used to derive semantic
representations from item metadata in recommendation and search. In sequential
recommendation, PLMs enhance ID-based embeddings through textual metadata,
while in product search, they align item characteristics with user intent.
Recent studies suggest task and domain-specific fine-tuning are needed to
improve representational power. This paper challenges this assumption, showing
that Generalist Text Embedding Models (GTEs), pre-trained on large-scale
corpora, can guarantee strong zero-shot performance without specialized
adaptation. Our experiments demonstrate that GTEs outperform traditional and
fine-tuned models in both sequential recommendation and product search. We
attribute this to a superior representational power, as they distribute
features more evenly across the embedding space. Finally, we show that
compressing embedding dimensions by focusing on the most informative directions
(e.g., via PCA) effectively reduces noise and improves the performance of
specialized models. To ensure reproducibility, we provide our repository at
https://split.to/gte4ps.

### 10. [In-Context Learning as an Effective Estimator of Functional Correctness of LLM-Generated Code](http://arxiv.org/pdf/2507.05200v1)

Authors: Susmita Das, Madhusudan Ghosh, Priyanka Swami, Debasis Ganguly, Gul Calikli

When applying LLM-based code generation to software development projects that
follow a feature-driven or rapid application development approach, it becomes
necessary to estimate the functional correctness of the generated code in the
absence of test cases. Just as a user selects a relevant document from a ranked
list of retrieved ones, a software generation workflow requires a developer to
choose (and potentially refine) a generated solution from a ranked list of
alternative solutions, ordered by their posterior likelihoods. This implies
that estimating the quality of a ranked list -- akin to estimating "relevance"
for query performance prediction (QPP) in IR -- is also crucial for generative
software development, where quality is defined in terms of "functional
correctness". In this paper, we propose an in-context learning (ICL) based
approach for code quality estimation. Our findings demonstrate that providing
few-shot examples of functionally correct code from a training set enhances the
performance of existing QPP approaches as well as a zero-shot-based approach
for code quality estimation.

### Machine Learning

### 1. [A Lightweight Deep Learning Model for Automatic Modulation Classification using Dual Path Deep Residual Shrinkage Network](http://arxiv.org/pdf/2507.04586v1)

Authors: Prakash Suman, Yanzhen Qu

Efficient spectrum utilization is critical to meeting the growing data
demands of modern wireless communication networks. Automatic Modulation
Classification (AMC) plays a key role in enhancing spectrum efficiency by
accurately identifying modulation schemes in received signals-an essential
capability for dynamic spectrum allocation and interference mitigation,
particularly in cognitive radio (CR) systems. With the increasing deployment of
smart edge devices, such as IoT nodes with limited computational and memory
resources, there is a pressing need for lightweight AMC models that balance low
complexity with high classification accuracy. This paper proposes a
low-complexity, lightweight deep learning (DL) AMC model optimized for
resource-constrained edge devices. We introduce a dual-path deep residual
shrinkage network (DP-DRSN) with Garrote thresholding for effective signal
denoising and design a compact hybrid CNN-LSTM architecture comprising only
27,000 training parameters. The proposed model achieves average classification
accuracies of 61.20%, 63.78%, and 62.13% on the RML2016.10a, RML2016.10b, and
RML2018.01a datasets, respectively demonstrating a strong balance between model
efficiency and classification performance. These results underscore the model's
potential for enabling accurate and efficient AMC on-edge devices with limited
resources.

### 2. [Photon Splatting: A Physics-Guided Neural Surrogate for Real-Time Wireless Channel Prediction](http://arxiv.org/pdf/2507.04595v1)

Authors: Ge Cao, Gabriele Gradoni, Zhen Peng

We present Photon Splatting, a physics-guided neural surrogate model for
real-time wireless channel prediction in complex environments. The proposed
framework introduces surface-attached virtual sources, referred to as photons,
which carry directional wave signatures informed by the scene geometry and
transmitter configuration. At runtime, channel impulse responses (CIRs) are
predicted by splatting these photons onto the angular domain of the receiver
using a geodesic rasterizer. The model is trained to learn a physically
grounded representation that maps transmitter-receiver configurations to full
channel responses. Once trained, it generalizes to new transmitter positions,
antenna beam patterns, and mobile receivers without requiring model retraining.
We demonstrate the effectiveness of the framework through a series of
experiments, from canonical 3D scenes to a complex indoor cafe with 1,000
receivers. Results show 30 millisecond-level inference latency and accurate CIR
predictions across a wide range of configurations. The approach supports
real-time adaptability and interpretability, making it a promising candidate
for wireless digital twin platforms and future 6G network planning.

### 3. [SOSAE: Self-Organizing Sparse AutoEncoder](http://arxiv.org/pdf/2507.04644v1)

Authors: Sarthak Ketanbhai Modi, Zi Pong Lim, Yushi Cao, Yupeng Cheng, Yon Shin Teo, Shang-Wei Lin

The process of tuning the size of the hidden layers for autoencoders has the
benefit of providing optimally compressed representations for the input data.
However, such hyper-parameter tuning process would take a lot of computation
and time effort with grid search as the default option. In this paper, we
introduce the Self-Organization Regularization for Autoencoders that
dynamically adapts the dimensionality of the feature space to the optimal size.
Inspired by physics concepts, Self-Organizing Sparse AutoEncoder (SOSAE)
induces sparsity in feature space in a structured way that permits the
truncation of the non-active part of the feature vector without any loss of
information. This is done by penalizing the autoencoder based on the magnitude
and the positional index of the feature vector dimensions, which during
training constricts the feature space in both terms. Extensive experiments on
various datasets show that our SOSAE can tune the feature space dimensionality
up to 130 times lesser Floating-point Operations (FLOPs) than other baselines
while maintaining the same quality of tuning and performance.

### 4. [A Cycle-Consistency Constrained Framework for Dynamic Solution Space Reduction in Noninjective Regression](http://arxiv.org/pdf/2507.04659v1)

Authors: Hanzhang Jia, Yi Gao

To address the challenges posed by the heavy reliance of multi-output models
on preset probability distributions and embedded prior knowledge in
non-injective regression tasks, this paper proposes a cycle consistency-based
data-driven training framework. The method jointly optimizes a forward model
{\Phi}: X to Y and a backward model {\Psi}: Y to X, where the cycle consistency
loss is defined as L _cycleb equal L(Y reduce {\Phi}({\Psi}(Y))) (and vice
versa). By minimizing this loss, the framework establishes a closed-loop
mechanism integrating generation and validation phases, eliminating the need
for manual rule design or prior distribution assumptions. Experiments on
normalized synthetic and simulated datasets demonstrate that the proposed
method achieves a cycle reconstruction error below 0.003, achieving an
improvement of approximately 30% in evaluation metrics compared to baseline
models without cycle consistency. Furthermore, the framework supports
unsupervised learning and significantly reduces reliance on manual
intervention, demonstrating potential advantages in non-injective regression
tasks.

### 5. [Hybrid Adversarial Spectral Loss Conditional Generative Adversarial Networks for Signal Data Augmentation in Ultra-precision Machining Surface Roughness Prediction](http://arxiv.org/pdf/2507.04665v1)

Authors: Suiyan Shang, Chi Fai Cheung, Pai Zheng

Accurate surface roughness prediction in ultra-precision machining (UPM) is
critical for real-time quality control, but small datasets hinder model
performance. We propose HAS-CGAN, a Hybrid Adversarial Spectral Loss CGAN, for
effective UPM data augmentation. Among five CGAN variants tested, HAS-CGAN
excels in 1D force signal generation, particularly for high-frequency signals,
achieving >0.85 wavelet coherence through Fourier-domain optimization. By
combining generated signals with machining parameters, prediction accuracy
significantly improves. Experiments with traditional ML (SVR, RF, LSTM) and
deep learning models (BPNN, 1DCNN, CNN-Transformer) demonstrate that augmenting
training data with 520+ synthetic samples reduces prediction error from 31.4%
(original 52 samples) to ~9%, effectively addressing data scarcity in UPM
roughness prediction."

### 6. [Recovering Plasticity of Neural Networks via Soft Weight Rescaling](http://arxiv.org/pdf/2507.04683v1)

Authors: Seungwon Oh, Sangyeon Park, Isaac Han, Kyung-Joong Kim

Recent studies have shown that as training progresses, neural networks
gradually lose their capacity to learn new information, a phenomenon known as
plasticity loss. An unbounded weight growth is one of the main causes of
plasticity loss. Furthermore, it harms generalization capability and disrupts
optimization dynamics. Re-initializing the network can be a solution, but it
results in the loss of learned information, leading to performance drops. In
this paper, we propose Soft Weight Rescaling (SWR), a novel approach that
prevents unbounded weight growth without losing information. SWR recovers the
plasticity of the network by simply scaling down the weight at each step of the
learning process. We theoretically prove that SWR bounds weight magnitude and
balances weight magnitude between layers. Our experiment shows that SWR
improves performance on warm-start learning, continual learning, and
single-task learning setups on standard image classification benchmarks.

### 7. [Interpretable Reward Modeling with Active Concept Bottlenecks](http://arxiv.org/pdf/2507.04695v1)

Authors: Sonia Laguna, Katarzyna Kobalczyk, Julia E. Vogt, Mihaela Van der Schaar

We introduce Concept Bottleneck Reward Models (CB-RM), a reward modeling
framework that enables interpretable preference learning through selective
concept annotation. Unlike standard RLHF methods that rely on opaque reward
functions, CB-RM decomposes reward prediction into human-interpretable
concepts. To make this framework efficient in low-supervision settings, we
formalize an active learning strategy that dynamically acquires the most
informative concept labels. We propose an acquisition function based on
Expected Information Gain and show that it significantly accelerates concept
learning without compromising preference accuracy. Evaluated on the
UltraFeedback dataset, our method outperforms baselines in interpretability and
sample efficiency, marking a step towards more transparent, auditable, and
human-aligned reward models.

### 8. [Spooky Action at a Distance: Normalization Layers Enable Side-Channel Spatial Communication](http://arxiv.org/pdf/2507.04709v1)

Authors: Samuel Pfrommer, George Ma, Yixiao Huang, Somayeh Sojoudi

This work shows that normalization layers can facilitate a surprising degree
of communication across the spatial dimensions of an input tensor. We study a
toy localization task with a convolutional architecture and show that
normalization layers enable an iterative message passing procedure, allowing
information aggregation from well outside the local receptive field. Our
results suggest that normalization layers should be employed with caution in
applications such as diffusion-based trajectory generation, where maintaining a
spatially limited receptive field is crucial.

### 9. [FedPall: Prototype-based Adversarial and Collaborative Learning for Federated Learning with Feature Drift](http://arxiv.org/pdf/2507.04781v1)

Authors: Yong Zhang, Feng Liang, Guanghu Yuan, Min Yang, Chengming Li, Xiping Hu

Federated learning (FL) enables collaborative training of a global model in
the centralized server with data from multiple parties while preserving
privacy. However, data heterogeneity can significantly degrade the performance
of the global model when each party uses datasets from different sources to
train a local model, thereby affecting personalized local models. Among various
cases of data heterogeneity, feature drift, feature space difference among
parties, is prevalent in real-life data but remains largely unexplored. Feature
drift can distract feature extraction learning in clients and thus lead to poor
feature extraction and classification performance. To tackle the problem of
feature drift in FL, we propose FedPall, an FL framework that utilizes
prototype-based adversarial learning to unify feature spaces and collaborative
learning to reinforce class information within the features. Moreover, FedPall
leverages mixed features generated from global prototypes and local features to
enhance the global classifier with classification-relevant information from a
global perspective. Evaluation results on three representative feature-drifted
datasets demonstrate FedPall's consistently superior performance in
classification with feature-drifted data in the FL scenario.

### 10. [Machine Learning from Explanations](http://arxiv.org/pdf/2507.04788v1)

Authors: Jiashu Tao, Reza Shokri

Acquiring and training on large-scale labeled data can be impractical due to
cost constraints. Additionally, the use of small training datasets can result
in considerable variability in model outcomes, overfitting, and learning of
spurious correlations. A crucial shortcoming of data labels is their lack of
any reasoning behind a specific label assignment, causing models to learn any
arbitrary classification rule as long as it aligns data with labels. To
overcome these issues, we introduce an innovative approach for training
reliable classification models on smaller datasets, by using simple explanation
signals such as important input features from labeled data. Our method centers
around a two-stage training cycle that alternates between enhancing model
prediction accuracy and refining its attention to match the explanations. This
instructs models to grasp the rationale behind label assignments during their
learning phase. We demonstrate that our training cycle expedites the
convergence towards more accurate and reliable models, particularly for small,
class-imbalanced training data, or data with spurious features.

### Neural and Evolutionary Computing

### 1. [Bridging Expressivity and Scalability with Adaptive Unitary SSMs](http://arxiv.org/pdf/2507.05238v1)

Authors: Arjun Karuvally, Franz Nowak, Anderson T. Keller, Carmen Amo Alonso, Terrence J. Sejnowski, Hava T. Siegelmann

Recent work has revealed that state space models (SSMs), while efficient for
long-sequence processing, are fundamentally limited in their ability to
represent formal languages particularly due to time-invariant and real-valued
recurrence structures. In this work, we draw inspiration from adaptive and
structured dynamics observed in biological neural systems and introduce the
Adaptive Unitary State Space Model (AUSSM)- a novel class of SSMs that
leverages skew-symmetric, input-dependent recurrence to achieve unitary
evolution and high expressive power. Using algebraic automata theory, we prove
that AUSSM can perform modulo counting and simulate solvable group automata at
finite precision, enabling SSMs to model a broad class of regular languages
that are out of reach for other SSM architectures. To overcome the practical
inefficiencies of adaptive recurrence, we develop a separable convolution
formulation and a CUDA implementation that enables scalable parallel training.
Empirically, we show that AUSSM when interleaved with Mamba outperform prior
SSMs on formal algorithmic tasks such as parity and modular arithmetic, and
achieve competent performance on real-world long time-series classification
benchmarks. Our results demonstrate that adaptive unitary recurrence provides a
powerful and efficient inductive bias for both symbolic and continuous sequence
modeling.

### Networking and Internet Architecture

### 1. [Low-Latency Software Polar Encoders and Decoders for Short Blocklengths](http://arxiv.org/pdf/2507.04734v1)

Authors: Mathieu Leonardon, Mohammed El Houcine Ayoubi, Adrien Cassagne, Romain Tajan, Camille Leroux

This paper presents our low-latency Polar code encoders and decoders
developed for the 2025 International Symposium on Topics in Coding (ISTC 2025)
contest, which challenges participants to implement the fastest possible
channel code encoders and decoders in terms of average and maximum latency on a
CPU target. Our solution is based on Polar codes with an Adaptive Successive
Cancellation List (ASCL) decoder. We introduce a novel ASCL unrolled decoder
generator. We conduct an extensive exploration of the design space, including
code construction, CRC selection, and list size, to identify optimal trade-offs
between signal-to-noise ratio and decoding time across various operating
points. The considered operating points are frame error rates of 10^{-3} and
10^{-5}, information bit lengths of 64, 128, 256, and 512, and code rates of
1/4, 1/2, and 4/5. We also propose an optimized bit-packed encoder. All
implementations of the encoders and decoders, along with the code construction
and the unrolled decoders generator, are released as open source in the AFF3CT
toolbox.

### 2. [User Association in the Presence of Jamming in Wireless Networks Using the Whittle Index](http://arxiv.org/pdf/2507.04968v1)

Authors: Pramod N Chine, Suven Jagtiani, Mandar R Nalavade, Gaurav S Kasbekar

In wireless networks, algorithms for user association, i.e., the task of
choosing the base station (BS) that every arriving user should join,
significantly impact the network performance. A wireless network with multiple
BSs, operating on non-overlapping channels, is considered. The channels of the
BSs are susceptible to jamming by attackers. During every time slot, a user
arrives with a certain probability. There exists a holding cost in each slot
for every user associated with a BS. The goal here is to design a user
association scheme, which assigns a BS to each user upon arrival with the
objective of minimizing the long-run total average holding cost borne within
the network. This objective results in low average delays attained by the
users. This association problem is an instance of restless multi-armed bandit
problems, and is known to be hard to solve. By making use of the framework
presented by Whittle, the hard per-stage constraint that every arriving user
must connect to exactly one BS in a time slot is relaxed to a long-term
time-averaged constraint. Subsequently, we employ the Lagrangian multiplier
strategy to reformulate the problem into an unconstrained form and decompose it
into separate Markov Decision Processes at the BSs. Further, the problem is
proven to be Whittle indexable and a method for calculating the Whittle indices
corresponding to different BSs is presented. We design a user association
policy under which, upon arrival of a user in a time slot, it is assigned to
the BS having the least Whittle index in that slot. Through extensive
simulations, we show that our proposed association policy based on the Whittle
index outperforms various user association policies proposed in previous work
in terms of different metrics such as average cost, average delay, and Jain's
fairness index.

### 3. [On-Demand Multimedia Delivery in 6G: An Optimal-Cost Steiner Tree Approach](http://arxiv.org/pdf/2507.04589v1)

Authors: Zien Wang, Xiucheng Wang, Nan Cheng, Wenchao Xu, Wei Quan, Ruijin Sun, Conghao Zhou

The exponential growth of multimedia data traffic in 6G networks poses
unprecedented challenges for immersive communication, where
ultra-high-definition, multi-quality streaming must be delivered on demand
while minimizing network operational costs. Traditional routing approaches,
such as shortest-path algorithms, fail to optimize flow multiplexing across
multiple destinations, while conventional Steiner tree methods cannot
accommodate heterogeneous quality-of-service (QoS) requirements-a critical need
for 6G's personalized services. In this paper, we address a fundamental but
unsolved challenge: the minimum flow problem (MFP) with multi-destination,
heterogeneous outflow demands, which is pivotal for efficient multimedia
distribution such as adaptive-resolution video streaming. To overcome the
limitations of existing methods, we propose a two-stage dynamic
programming-enhanced On-demand Steiner Tree (OST) algorithm, the first approach
that jointly optimizes flow aggregation and QoS-aware path selection for
arbitrary outflow requirements. We rigorously prove the optimality of OST using
mathematical induction, demonstrating that it guarantees the minimum-cost
multicast flow under differentiated service constraints. Extensive experiments
in 6G-like multimedia transmission scenarios show that OST reduces total
network flow by over 10% compared to state-of-the-art methods while ensuring
on-demand QoS fulfillment. The complete code is available at
https://github.com/UNIC-Lab/OST.

### 4. [Multimodal LLM Integrated Semantic Communications for 6G Immersive Experiences](http://arxiv.org/pdf/2507.04621v1)

Authors: Yusong Zhang, Yuxuan Sun, Lei Guo, Wei Chen, Bo Ai, Deniz Gunduz

6G networks promise revolutionary immersive communication experiences
including augmented reality (AR), virtual reality (VR), and holographic
communications. These applications demand high-dimensional multimodal data
transmission and intelligent data processing in real-time, which is extremely
challenging over resource-limited wireless communication systems. Moreover, a
joint understanding of the environment, context, and user intent is essential
to deliver task-relevant content effectively. This article presents a novel
multimodal large language model (MLLM) integrated semantic communications
framework, termed MLLM-SC, which fully leverages reasoning and generative
capabilities of pre-trained foundation models for context-aware and
task-oriented wireless communication. The MLLM-SC framework adopts a
device-edge collaborative architecture. At the edge, MLLM-empowered semantic
guidance module analyzes multimodal inputs, user intents, and channel
conditions to generate importance-aware attention maps prioritizing
semantically critical information. An importance-aware semantic encoder and a
resource-adaptive semantic decoder are jointly designed and optimized, which
can utilize the semantic guidance for adaptive bandwidth allocation and
high-quality content reconstruction or generation. Extensive case studies on
visual question answering for AR/VR applications and diffusion-driven image
generation validate the effectiveness of MLLM-SC.

### 5. [Large Language Models for Network Intrusion Detection Systems: Foundations, Implementations, and Future Directions](http://arxiv.org/pdf/2507.04752v1)

Authors: Shuo Yang, Xinran Zheng, Xinchen Zhang, Jinfeng Xu, Jinze Li, Donglin Xie, Weicai Long, Edith C. H. Ngai

Large Language Models (LLMs) have revolutionized various fields with their
exceptional capabilities in understanding, processing, and generating
human-like text. This paper investigates the potential of LLMs in advancing
Network Intrusion Detection Systems (NIDS), analyzing current challenges,
methodologies, and future opportunities. It begins by establishing a
foundational understanding of NIDS and LLMs, exploring the enabling
technologies that bridge the gap between intelligent and cognitive systems in
AI-driven NIDS. While Intelligent NIDS leverage machine learning and deep
learning to detect threats based on learned patterns, they often lack
contextual awareness and explainability. In contrast, Cognitive NIDS integrate
LLMs to process both structured and unstructured security data, enabling deeper
contextual reasoning, explainable decision-making, and automated response for
intrusion behaviors. Practical implementations are then detailed, highlighting
LLMs as processors, detectors, and explainers within a comprehensive AI-driven
NIDS pipeline. Furthermore, the concept of an LLM-centered Controller is
proposed, emphasizing its potential to coordinate intrusion detection
workflows, optimizing tool collaboration and system performance. Finally, this
paper identifies critical challenges and opportunities, aiming to foster
innovation in developing reliable, adaptive, and explainable NIDS. By
presenting the transformative potential of LLMs, this paper seeks to inspire
advancement in next-generation network security systems.

### 6. [Age-Aware CSI Acquisition of a Finite-State Markovian Channel](http://arxiv.org/pdf/2507.05042v1)

Authors: Onur Ayan, Jiping Luo, Xueli An, Nikolaos Pappas

The Age of Information (AoI) has emerged as a critical metric for quantifying
information freshness; however, its interplay with channel estimation in
partially observable wireless systems remains underexplored. This work
considers a transmitter-receiver pair communicating over an unreliable channel
with time-varying reliability levels. The transmitter observes the
instantaneous link reliability through a channel state information acquisition
procedure, during which the data transmission is interrupted. This leads to a
fundamental trade-off between utilizing limited network resources for either
data transmission or channel state information acquisition to combat the
channel aging effect. Assuming the wireless channel is modeled as a
finite-state Markovian channel, we formulate an optimization problem as a
partially observable Markov decision process (POMDP), obtain the optimal policy
through the relative value iteration algorithm, and demonstrate the efficiency
of our solution through simulations. To the best of our knowledge, this is the
first work to aim for an optimal scheduling policy for data transmissions while
considering the effect of channel state information aging.

### Robotics

### 1. [DragonFly: Single mmWave Radar 3D Localization of Highly Dynamic Tags in GPS-Denied Environments](http://arxiv.org/pdf/2507.04602v1)

Authors: Skanda Harisha, Jimmy G. D. Hester, Aline Eid

The accurate localization and tracking of dynamic targets, such as equipment,
people, vehicles, drones, robots, and the assets that they interact with in
GPS-denied indoor environments is critical to enabling safe and efficient
operations in the next generation of spatially aware industrial facilities.
This paper presents DragonFly , a 3D localization system of highly dynamic
backscatter tags using a single MIMO mmWave radar. The system delivers the
first demonstration of a mmWave backscatter system capable of exploiting the
capabilities of MIMO radars for the 3D localization of mmID tags moving at high
speeds and accelerations at long ranges by introducing a critical Doppler
disambiguation algorithm and a fully integrated cross-polarized dielectric
lens-based mmID tag consuming a mere 68 uW. DragonFly was extensively evaluated
in static and dynamic configurations, including on a flying quadcopter, and
benchmarked against multiple baselines, demonstrating its ability to track the
positions of multiple tags with a median 3D accuracy of 12 cm at speeds and
acceleration on the order of 10 m/s-1 and 4 m/s-2 and at ranges of up to 50 m.

### 2. [IDAGC: Adaptive Generalized Human-Robot Collaboration via Human Intent Estimation and Multimodal Policy Learning](http://arxiv.org/pdf/2507.04620v1)

Authors: Haotian Liu, Yuchuang Tong, Guanchen Liu, Zhaojie Ju, Zhengtao Zhang

In Human-Robot Collaboration (HRC), which encompasses physical interaction
and remote cooperation, accurate estimation of human intentions and seamless
switching of collaboration modes to adjust robot behavior remain paramount
challenges. To address these issues, we propose an Intent-Driven Adaptive
Generalized Collaboration (IDAGC) framework that leverages multimodal data and
human intent estimation to facilitate adaptive policy learning across
multi-tasks in diverse scenarios, thereby facilitating autonomous inference of
collaboration modes and dynamic adjustment of robotic actions. This framework
overcomes the limitations of existing HRC methods, which are typically
restricted to a single collaboration mode and lack the capacity to identify and
transition between diverse states. Central to our framework is a predictive
model that captures the interdependencies among vision, language, force, and
robot state data to accurately recognize human intentions with a Conditional
Variational Autoencoder (CVAE) and automatically switch collaboration modes. By
employing dedicated encoders for each modality and integrating extracted
features through a Transformer decoder, the framework efficiently learns
multi-task policies, while force data optimizes compliance control and intent
estimation accuracy during physical interactions. Experiments highlights our
framework's practical potential to advance the comprehensive development of
HRC.

### 3. [PRISM: Pointcloud Reintegrated Inference via Segmentation and Cross-attention for Manipulation](http://arxiv.org/pdf/2507.04633v1)

Authors: Daqi Huang, Zhehao Cai, Yuzhi Hao, Zechen Li, Chee-Meng Chew

Robust imitation learning for robot manipulation requires comprehensive 3D
perception, yet many existing methods struggle in cluttered environments. Fixed
camera view approaches are vulnerable to perspective changes, and 3D point
cloud techniques often limit themselves to keyframes predictions, reducing
their efficacy in dynamic, contact-intensive tasks. To address these
challenges, we propose PRISM, designed as an end-to-end framework that directly
learns from raw point cloud observations and robot states, eliminating the need
for pretrained models or external datasets. PRISM comprises three main
components: a segmentation embedding unit that partitions the raw point cloud
into distinct object clusters and encodes local geometric details; a
cross-attention component that merges these visual features with processed
robot joint states to highlight relevant targets; and a diffusion module that
translates the fused representation into smooth robot actions. With training on
100 demonstrations per task, PRISM surpasses both 2D and 3D baseline policies
in accuracy and efficiency within our simulated environments, demonstrating
strong robustness in complex, object-dense scenarios. Code and some demos are
available on https://github.com/czknuaa/PRISM.

### 4. [Bio-Inspired Hybrid Map: Spatial Implicit Local Frames and Topological Map for Mobile Cobot Navigation](http://arxiv.org/pdf/2507.04649v1)

Authors: Tuan Dang, Manfred Huber

Navigation is a fundamental capacity for mobile robots, enabling them to
operate autonomously in complex and dynamic environments. Conventional
approaches use probabilistic models to localize robots and build maps
simultaneously using sensor observations. Recent approaches employ
human-inspired learning, such as imitation and reinforcement learning, to
navigate robots more effectively. However, these methods suffer from high
computational costs, global map inconsistency, and poor generalization to
unseen environments. This paper presents a novel method inspired by how humans
perceive and navigate themselves effectively in novel environments.
Specifically, we first build local frames that mimic how humans represent
essential spatial information in the short term. Points in local frames are
hybrid representations, including spatial information and learned features,
so-called spatial-implicit local frames. Then, we integrate spatial-implicit
local frames into the global topological map represented as a factor graph.
Lastly, we developed a novel navigation algorithm based on Rapid-Exploring
Random Tree Star (RRT*) that leverages spatial-implicit local frames and the
topological map to navigate effectively in environments. To validate our
approach, we conduct extensive experiments in real-world datasets and in-lab
environments. We open our source code at
https://github.com/tuantdang/simn}{https://github.com/tuantdang/simn.

### 5. [DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics](http://arxiv.org/pdf/2507.04661v1)

Authors: Yayu Long, Kewei Chen, Long Jin, Mingsheng Shang

We introduce Dynamic Retrieval-Augmented Expert Networks (DRAE), a
groundbreaking architecture that addresses the challenges of lifelong learning,
catastrophic forgetting, and task adaptation by combining the dynamic routing
capabilities of Mixture-of-Experts (MoE); leveraging the knowledge-enhancement
power of Retrieval-Augmented Generation (RAG); incorporating a novel
hierarchical reinforcement learning (RL) framework; and coordinating through
ReflexNet-SchemaPlanner-HyperOptima (RSHO).DRAE dynamically routes expert
models via a sparse MoE gating mechanism, enabling efficient resource
allocation while leveraging external knowledge through parametric retrieval
(P-RAG) to augment the learning process. We propose a new RL framework with
ReflexNet for low-level task execution, SchemaPlanner for symbolic reasoning,
and HyperOptima for long-term context modeling, ensuring continuous adaptation
and memory retention. Experimental results show that DRAE significantly
outperforms baseline approaches in long-term task retention and knowledge
reuse, achieving an average task success rate of 82.5% across a set of dynamic
robotic manipulation tasks, compared to 74.2% for traditional MoE models.
Furthermore, DRAE maintains an extremely low forgetting rate, outperforming
state-of-the-art methods in catastrophic forgetting mitigation. These results
demonstrate the effectiveness of our approach in enabling flexible, scalable,
and efficient lifelong learning for robotics.

### 6. [MOSU: Autonomous Long-range Robot Navigation with Multi-modal Scene Understanding](http://arxiv.org/pdf/2507.04686v1)

Authors: Jing Liang, Kasun Weerakoon, Daeun Song, Senthurbavan Kirubaharan, Xuesu Xiao, Dinesh Manocha

We present MOSU, a novel autonomous long-range navigation system that
enhances global navigation for mobile robots through multimodal perception and
on-road scene understanding. MOSU addresses the outdoor robot navigation
challenge by integrating geometric, semantic, and contextual information to
ensure comprehensive scene understanding. The system combines GPS and QGIS
map-based routing for high-level global path planning and multi-modal
trajectory generation for local navigation refinement. For trajectory
generation, MOSU leverages multi-modalities: LiDAR-based geometric data for
precise obstacle avoidance, image-based semantic segmentation for
traversability assessment, and Vision-Language Models (VLMs) to capture social
context and enable the robot to adhere to social norms in complex environments.
This multi-modal integration improves scene understanding and enhances
traversability, allowing the robot to adapt to diverse outdoor conditions. We
evaluate our system in real-world on-road environments and benchmark it on the
GND dataset, achieving a 10% improvement in traversability on navigable
terrains while maintaining a comparable navigation distance to existing global
navigation methods.

### 7. [Training-free Generation of Temporally Consistent Rewards from VLMs](http://arxiv.org/pdf/2507.04789v1)

Authors: Yinuo Zhao, Jiale Yuan, Zhiyuan Xu, Xiaoshuai Hao, Xinyi Zhang, Kun Wu, Zhengping Che, Chi Harold Liu, Jian Tang

Recent advances in vision-language models (VLMs) have significantly improved
performance in embodied tasks such as goal decomposition and visual
comprehension. However, providing accurate rewards for robotic manipulation
without fine-tuning VLMs remains challenging due to the absence of
domain-specific robotic knowledge in pre-trained datasets and high
computational costs that hinder real-time applicability. To address this, we
propose $\mathrm{T}^2$-VLM, a novel training-free, temporally consistent
framework that generates accurate rewards through tracking the status changes
in VLM-derived subgoals. Specifically, our method first queries the VLM to
establish spatially aware subgoals and an initial completion estimate before
each round of interaction. We then employ a Bayesian tracking algorithm to
update the goal completion status dynamically, using subgoal hidden states to
generate structured rewards for reinforcement learning (RL) agents. This
approach enhances long-horizon decision-making and improves failure recovery
capabilities with RL. Extensive experiments indicate that $\mathrm{T}^2$-VLM
achieves state-of-the-art performance in two robot manipulation benchmarks,
demonstrating superior reward accuracy with reduced computation consumption. We
believe our approach not only advances reward generation techniques but also
contributes to the broader field of embodied AI. Project website:
https://t2-vlm.github.io/.

### 8. [Safe Bimanual Teleoperation with Language-Guided Collision Avoidance](http://arxiv.org/pdf/2507.04791v1)

Authors: Dionis Totsila, Clemente Donoso, Enrico Mingo Hoffman, Jean-Baptiste Mouret, Serena Ivaldi

Teleoperating precise bimanual manipulations in cluttered environments is
challenging for operators, who often struggle with limited spatial perception
and difficulty estimating distances between target objects, the robot's body,
obstacles, and the surrounding environment. To address these challenges, local
robot perception and control should assist the operator during teleoperation.
In this work, we introduce a safe teleoperation system that enhances operator
control by preventing collisions in cluttered environments through the
combination of immersive VR control and voice-activated collision avoidance.
Using HTC Vive controllers, operators directly control a bimanual mobile
manipulator, while spoken commands such as "avoid the yellow tool" trigger
visual grounding and segmentation to build 3D obstacle meshes. These meshes are
integrated into a whole-body controller to actively prevent collisions during
teleoperation. Experiments in static, cluttered scenes demonstrate that our
system significantly improves operational safety without compromising task
efficiency.

### 9. [Dynamics and multi-stability of a rotor-actuated Twistcar robot with passive steering joint](http://arxiv.org/pdf/2507.04846v1)

Authors: Anna Zigelman, Zitao Yu, Rom Levy, Yizhar Or

The nonlinear dynamics of many under-actuated wheeled platforms are governed
by nonholonomic constraints of no-skid for passively rolling wheels, coupled
with momentum balance. In most of theoretical models, the shape variables, i.e.
joint angles, are directly prescribed as periodic inputs, such as steering
angle of the Twistcar. In this work, we study a variant of the Twistcar model
where the actuation input is periodic oscillations of an inertial rotor
attached to the main body, while the steering joint is passively free to
rotate. Remarkably, the dynamics of this model is extremely rich, and includes
multiplicity of periodic solutions, both symmetric and asymmetric, as well as
stability transitions and bifurcations. We conduct numerical simulations as
well as asymptotic analysis of the vehicle's reduced equations of motion. We
use perturbation expansion in order to obtain leading-order dynamics under
symmetric periodic solution. Then, we utilize harmonic balance and further
scaling assumptions in order to approximate the conditions for
symmetry-breaking pitchfork bifurcation and stability transition of the
symmetric periodic solution, as a function of actuation frequency and
structural parameters. The asymptotic results show good agreement with
numerical simulations. The results highlight the role of passive shape
variables in generating multi-stable periodic solutions for nonholonomic
systems of robotic locomotion.

### 10. [Automated UAV-based Wind Turbine Blade Inspection: Blade Stop Angle Estimation and Blade Detail Prioritized Exposure Adjustment](http://arxiv.org/pdf/2507.04922v1)

Authors: Yichuan Shi, Hao Liu, Haowen Zheng, Haowen Yu, Xianqi Liang, Jie Li, Minmin Ma, Ximin Lyu

Unmanned aerial vehicles (UAVs) are critical in the automated inspection of
wind turbine blades. Nevertheless, several issues persist in this domain.
Firstly, existing inspection platforms encounter challenges in meeting the
demands of automated inspection tasks and scenarios. Moreover, current blade
stop angle estimation methods are vulnerable to environmental factors,
restricting their robustness. Additionally, there is an absence of real-time
blade detail prioritized exposure adjustment during capture, where lost details
cannot be restored through post-optimization. To address these challenges, we
introduce a platform and two approaches. Initially, a UAV inspection platform
is presented to meet the automated inspection requirements. Subsequently, a
Fermat point based blade stop angle estimation approach is introduced,
achieving higher precision and success rates. Finally, we propose a blade
detail prioritized exposure adjustment approach to ensure appropriate
brightness and preserve details during image capture. Extensive tests,
comprising over 120 flights across 10 wind turbine models in 5 operational wind
farms, validate the effectiveness of the proposed approaches in enhancing
inspection autonomy.

### Software Engineering

### 1. [Supporting Software Formal Verification with Large Language Models: An Experimental Study](http://arxiv.org/pdf/2507.04857v1)

Authors: Weiqi Wang, Marie Farrell, Lucas C. Cordeiro, Liping Zhao

Formal methods have been employed for requirements verification for a long
time. However, it is difficult to automatically derive properties from natural
language requirements. SpecVerify addresses this challenge by integrating large
language models (LLMs) with formal verification tools, providing a more
flexible mechanism for expressing requirements. This framework combines Claude
3.5 Sonnet with the ESBMC verifier to form an automated workflow. Evaluated on
nine cyber-physical systems from Lockheed Martin, SpecVerify achieves 46.5%
verification accuracy, comparable to NASA's CoCoSim, but with lower false
positives. Our framework formulates assertions that extend beyond the
expressive power of LTL and identifies falsifiable cases that are missed by
more traditional methods. Counterexample analysis reveals CoCoSim's limitations
stemming from model connection errors and numerical approximation issues. While
SpecVerify advances verification automation, our comparative study of Claude,
ChatGPT, and Llama shows that high-quality requirements documentation and human
monitoring remain critical, as models occasionally misinterpret specifications.
Our results demonstrate that LLMs can significantly reduce the barriers to
formal verification, while highlighting the continued importance of
human-machine collaboration in achieving optimal results.

### 2. [Towards a Unifying Reference Model for Digital Twins of Cyber-Physical Systems](http://arxiv.org/pdf/2507.04871v1)

Authors: Jerome Pfeiffer, Jingxi Zhang, Benoit Combemale, Judith Michael, Bernhard Rumpe, Manuel Wimmer, Andreas Wortmann

Digital twins are sophisticated software systems for the representation,
monitoring, and control of cyber-physical systems, including automotive,
avionics, smart manufacturing, and many more. Existing definitions and
reference models of digital twins are overly abstract, impeding their
comprehensive understanding and implementation guidance. Consequently, a
significant gap emerges between abstract concepts and their industrial
implementations. We analyze popular reference models for digital twins and
combine these into a significantly detailed unifying reference model for
digital twins that reduces the concept-implementation gap to facilitate their
engineering in industrial practice. This enhances the understanding of the
concepts of digital twins and their relationships and guides developers to
implement digital twins effectively.

### 3. [Understanding Everything as Code: A Taxonomy and Conceptual Model](http://arxiv.org/pdf/2507.05100v1)

Authors: Haoran Wei, Nazim Madhavji, John Steinbacher

Background: Everything as Code (EaC) is an emerging paradigm aiming to codify
all aspects of modern software systems. Despite its growing popularity,
comprehensive industry standards and peer-reviewed research clarifying its
scope and guiding its adoption remain scarce. Aims: This study systematically
analyzes existing knowledge and perceptions of EaC, clarifies its scope and
boundaries, and provides structured guidance for researchers and practitioners.
Method: We conducted a large-scale multivocal literature review (MLR),
synthesizing academic and grey literature sources. Findings were analyzed
quantitatively and thematically. Based on this analysis, we developed a
taxonomy and conceptual model of EaC, validated through collaboration with
industry experts. Results: The resulting taxonomy comprises 25 distinct EaC
practices organized into six layers based on industry awareness and functional
roles. The conceptual model illustrates focus areas, overlaps, and interactions
among these EaC practices within the software delivery lifecycle. Additionally,
practical code examples demonstrating the implementation of these practices
were developed in collaboration with industry experts. Conclusions: This work
addresses the current scarcity of academic discourse on EaC by providing the
first comprehensive taxonomy and conceptual model. These contributions enhance
conceptual clarity, offer actionable guidance to practitioners, and lay the
groundwork for future research in this emerging domain.

### 4. [An Investigation into Maintenance Support for Neural Networks](http://arxiv.org/pdf/2507.05245v1)

Authors: Fatema Tuz Zohra, Brittany Johnson

As the potential for neural networks to augment our daily lives grows,
ensuring their quality through effective testing, debugging, and maintenance is
essential. This is especially the case as we acknowledge the prospects of
negative impacts from these technologies. Traditional software engineering
methods, such as testing and debugging, have proven effective in maintaining
software quality; however, they reveal significant research and practice gaps
in maintaining neural networks. In particular, there is a limited understanding
of how practitioners currently address challenges related to understanding and
mitigating undesirable behaviors in neural networks. In our ongoing research,
we explore the current state of research and practice in maintaining neural
networks by curating insights from practitioners through a preliminary study
involving interviews and supporting survey responses. Our findings thus far
indicate that existing tools primarily concentrate on building and training
models. While these tools can be beneficial, they often fall short of
supporting practitioners' understanding and addressing the underlying causes of
unexpected model behavior. By evaluating current procedures and identifying the
limitations of traditional methodologies, our study aims to offer a
developer-centric perspective on where current practices fall short and
highlight opportunities for improving maintenance support in neural networks.

### 5. [ArtifactsBench: Bridging the Visual-Interactive Gap in LLM Code Generation Evaluation](http://arxiv.org/pdf/2507.04952v1)

Authors: Chenchen Zhang, Yuhang Li, Can Xu, Jiaheng Liu, Ao Liu, Shihui Hu, Dengpeng Wu, Guanhua Huang, Kejiao Li, Qi Yi, Ruibin Xiong, Haotian Zhu, Yuanxing Zhang, Yuhao Jiang, Yue Zhang, Zenan Xu, Bohui Zhai, Guoxiang He, Hebin Li, Jie Zhao, Le Zhang, Lingyun Tan, Pengyu Guo, Xianshu Pang, Yang Ruan, Zhifeng Zhang, Zhonghu Wang, Ziyan Xu, Zuopu Yin, Wiggin Zhou, Chayse Zhou, Fengzong Lian

The generative capabilities of Large Language Models (LLMs) are rapidly
expanding from static code to dynamic, interactive visual artifacts. This
progress is bottlenecked by a critical evaluation gap: established benchmarks
focus on algorithmic correctness and are blind to the visual fidelity and
interactive integrity that define modern user experiences. To bridge this gap,
we introduce ArtifactsBench, a new benchmark and paradigm for the automated,
multimodal evaluation of visual code generation. Our framework programmatically
renders each generated artifact and captures its dynamic behavior through
temporal screenshots. This visual evidence, alongside the source code, is then
assessed by a Multimodal LLM (MLLM)-as-Judge, which is rigorously guided by a
fine-grained, per-task checklist to ensure holistic and reproducible scoring.
We construct a new benchmark of 1,825 diverse tasks and evaluate over 30
leading LLMs. Our automated evaluation achieves a striking 94.4% ranking
consistency with WebDev Arena, the gold-standard for human preference in web
development, and over 90% pairwise agreement with human experts. This
establishes ArtifactsBench as the first framework to reliably automate the
assessment of human-perceived quality at scale. Our analysis provides a
high-resolution map of the current SOTA, revealing that generalist models often
outperform domain-specific ones. We open-source ArtifactsBench, including the
benchmark, evaluation harness, and baseline results at
https://artifactsbenchmark.github.io/, to provide the community with a scalable
and accurate tool to accelerate the development of user-centric generative
models.

### 6. [AI for the Routine, Humans for the Complex: Accuracy-Driven Data Labelling with Mixed Integer Linear Programming](http://arxiv.org/pdf/2507.04990v1)

Authors: Mohammad Hossein Amini, Mehrdad Sabetzadeh, Shiva Nejati

The scarcity of accurately labelled data remains a major challenge in deep
learning (DL). Many DL approaches rely on semi-supervised methods, which focus
on constructing large datasets that require only a minimal amount of
human-labelled data. Since DL training algorithms can tolerate moderate label
noise, it has generally been acceptable for the accuracy of labels in large
training datasets to fall well short of a perfect 100%. However, when it comes
to testing DL models, achieving high label accuracy-as close to 100% as
possible-is paramount for reliable verification. In this article, we introduce
OPAL, a human-assisted labelling method that can be configured to target a
desired accuracy level while minimizing the manual effort required for
labelling. The main contribution of OPAL is a mixed-integer linear programming
(MILP) formulation that minimizes labelling effort subject to a specified
accuracy target. We evaluate OPAL for two tasks in the context of testing
vision systems: automatic labelling of test data and automated validation of
test data. Our evaluation, based on more than 2500 experiments performed on
seven datasets, comparing OPAL with eight baseline methods, shows that OPAL,
relying on its MILP formulation, achieves an average accuracy of 98.8%, just
1.2% below perfect accuracy, while cutting manual labelling by more than half.
Further, OPAL significantly outperforms automated labelling baselines in
labelling accuracy across all seven datasets, with large effect sizes, when all
methods are provided with the same manual-labelling budget. For automated
test-input validation, on average, OPAL reduces manual effort by 28.8% while
achieving 4.5% higher accuracy than the SOTA validation baselines. Finally, we
show that augmenting OPAL with an active learning loop leads to an additional
4.5% reduction in required manual labelling, without compromising accuracy.

### 7. [In-Context Learning as an Effective Estimator of Functional Correctness of LLM-Generated Code](http://arxiv.org/pdf/2507.05200v1)

Authors: Susmita Das, Madhusudan Ghosh, Priyanka Swami, Debasis Ganguly, Gul Calikli

When applying LLM-based code generation to software development projects that
follow a feature-driven or rapid application development approach, it becomes
necessary to estimate the functional correctness of the generated code in the
absence of test cases. Just as a user selects a relevant document from a ranked
list of retrieved ones, a software generation workflow requires a developer to
choose (and potentially refine) a generated solution from a ranked list of
alternative solutions, ordered by their posterior likelihoods. This implies
that estimating the quality of a ranked list -- akin to estimating "relevance"
for query performance prediction (QPP) in IR -- is also crucial for generative
software development, where quality is defined in terms of "functional
correctness". In this paper, we propose an in-context learning (ICL) based
approach for code quality estimation. Our findings demonstrate that providing
few-shot examples of functionally correct code from a training set enhances the
performance of existing QPP approaches as well as a zero-shot-based approach
for code quality estimation.

### 8. [React-tRace: A Semantics for Understanding React Hooks](http://arxiv.org/pdf/2507.05234v1)

Authors: Jay Lee, Joongwon Ahn, Kwangkeun Yi

React has become the most widely used web front-end framework, enabling the
creation of user interfaces in a declarative and compositional manner. Hooks
are a set of APIs that manage side effects in functional components in React.
However, their semantics are often seen as opaque to developers, leading to UI
bugs. In this paper, we formalize the semantics of the essence of React Hooks
we name React-tRace, providing a framework that clarifies their behavior. We
demonstrate that our model captures the behavior of React, by theoretically
showing that it embodies essential properties of Hooks and empirically
comparing our React-tRace-definitional interpreter against a test suite.
Furthermore, we showcase a practical visualization tool based on the
formalization to demonstrate how developers can better understand the semantics
of Hooks.

### 9. [A Note on Runtime Verification of Concurrent Systems](http://arxiv.org/pdf/2507.04830v1)

Authors: Martin Leucker

To maximize the information gained from a single execution when verifying a
concurrent system, one can derive all concurrency-aware equivalent executions
and check them against linear specifications. This paper offers an alternative
perspective on verification of concurrent systems by leveraging trace-based
logics rather than sequence-based formalisms. Linear Temporal Logic over
Mazurkiewicz Traces (LTrL) operates on partial-order representations of
executions, meaning that once a single execution is specified, all equivalent
interleavings are implicitly considered. This paper introduces a three valued
version of LTrL, indicating whether the so-far observed execution of the
concurrent system is one of correct, incorrect or inconclusive, together with a
suitable monitor synthesis procedure. To this end, the paper recalls a
construction of trace-consistent B\"uchi automata for LTrL formulas and
explains how to employ it in well-understood monitor synthesis procedures. In
this way, a monitor results that yields for any linearization of an observed
trace the same verification verdict.

### Social and Information Networks

### 1. [Investigating Algorithmic Bias in YouTube Shorts](http://arxiv.org/pdf/2507.04605v1)

Authors: Mert Can Cakmak, Nitin Agarwal, Diwash Poudel

The rapid growth of YouTube Shorts, now serving over 2 billion monthly users,
reflects a global shift toward short-form video as a dominant mode of online
content consumption. This study investigates algorithmic bias in YouTube
Shorts' recommendation system by analyzing how watch-time duration, topic
sensitivity, and engagement metrics influence content visibility and drift. We
focus on three content domains: the South China Sea dispute, the 2024 Taiwan
presidential election, and general YouTube Shorts content. Using generative AI
models, we classified 685,842 videos across relevance, topic category, and
emotional tone. Our results reveal a consistent drift away from politically
sensitive content toward entertainment-focused videos. Emotion analysis shows a
systematic preference for joyful or neutral content, while engagement patterns
indicate that highly viewed and liked videos are disproportionately promoted,
reinforcing popularity bias. This work provides the first comprehensive
analysis of algorithmic drift in YouTube Shorts based on textual content,
emotional tone, topic categorization, and varying watch-time conditions. These
findings offer new insights into how algorithmic design shapes content
exposure, with implications for platform transparency and information
diversity.

### 2. [Advancement of Circular Economy Through Interdisciplinary Collaboration: A Bibliometric Approach](http://arxiv.org/pdf/2507.04923v1)

Authors: Keita Nishimoto, Koji Kimita, Shinsuke Murakami, Yin Long, Kimitaka Asatani, Ichiro Sakata

Since the European Union introduced its Circular Economy (CE) Action Plan in
2015, CE research has expanded rapidly. However, the structure of this emerging
field - both in terms of its constituent disciplines and researcher dynamics -
remains poorly understood. To address this gap, we analyze over 25,000
CE-related publications from Scopus by combining conventional bibliometric
approaches with advanced machine learning techniques, including text embeddings
and clustering. This hybrid method enables both a macro-level mapping of
research domains and a micro-level investigation of individual researchers'
disciplinary backgrounds and collaborations.
  We classify CE research into 16 distinct clusters, identifying the original
disciplines of researchers and visualizing patterns of interdisciplinary
collaboration. Building on this foundation, we ask: Which CE-related research
domains receive the most attention in academic and policy contexts? And how are
different types of interdisciplinary collaboration associated with research
impact?
  Our findings show that research in business and management attracts
substantial academic and policy attention, while engineering research - though
less visible - tends to achieve higher funding success. This suggests a
positive dynamic in which the former draws attention to CE issues and the
latter secures the economic resources necessary to realize them.
  We further demonstrate that CE papers co-authored by researchers from
different disciplines tend to show higher research impact than
intradisciplinary work. Qualitative case analyses also highlight this tendency.
Centered particularly on collaborations between business-oriented and
engineering-oriented disciplines, our findings underscore the importance of
interdisciplinary efforts in CE research and offer insights for guiding future
cross-disciplinary engagement in the field.

### 3. [VaxPulse: Monitoring of Online Public Concerns to Enhance Post-licensure Vaccine Surveillance](http://arxiv.org/pdf/2507.04656v1)

Authors: Muhammad Javed, Sedigh Khademi, Joanne Hickman, Jim Buttery, Hazel Clothier, Gerardo Luis Dimaguila

The recent vaccine-related infodemic has amplified public concerns,
highlighting the need for proactive misinformation management. We describe how
we enhanced the reporting surveillance system of Victoria's vaccine safety
service, SAEFVIC, through the incorporation of new information sources for
public sentiment analysis, topics of discussion, and hesitancies about
vaccinations online. Using VaxPulse, a multi-step framework, we integrate
adverse events following immunisation (AEFI) with sentiment analysis,
demonstrating the importance of contextualising public concerns. Additionally,
we emphasise the need to address non-English languages to stratify concerns
across ethno-lingual communities, providing valuable insights for vaccine
uptake strategies and combating mis/disinformation. The framework is applied to
real-world examples and a case study on women's vaccine hesitancy, showcasing
its benefits and adaptability by identifying public opinion from online media.

### 4. [Interest Networks (iNETs) for Cities: Cross-Platform Insights and Urban Behavior Explanations](http://arxiv.org/pdf/2507.04995v1)

Authors: Gustavo H. Santos, Myriam Delgado, Thiago H. Silva

Location-Based Social Networks (LBSNs) provide a rich foundation for modeling
urban behavior through iNETs (Interest Networks), which capture how user
interests are distributed throughout urban spaces. This study compares iNETs
across platforms (Google Places and Foursquare) and spatial granularities,
showing that coarser levels reveal more consistent cross-platform patterns,
while finer granularities expose subtle, platform-specific behaviors. Our
analysis finds that, in general, user interest is primarily shaped by
geographic proximity and venue similarity, while socioeconomic and political
contexts play a lesser role. Building on these insights, we develop a
multi-level, explainable recommendation system that predicts high-interest
urban regions for different user types. The model adapts to behavior profiles
-- such as explorers, who are driven by proximity, and returners, who prefer
familiar venues -- and provides natural-language explanations using explainable
AI (XAI) techniques. To support our approach, we introduce h3-cities, a tool
for multi-scale spatial analysis, and release a public demo for interactively
exploring personalized urban recommendations. Our findings contribute to urban
mobility research by providing scalable, context-aware, and interpretable
recommendation systems.

### Systems and Control

### 1. [Risk-Aware Trajectory Optimization and Control for an Underwater Suspended Robotic System](http://arxiv.org/pdf/2507.04640v1)

Authors: Yuki Origane, Nicolas Hoischen, Tzu-Yuan Huang, Daisuke Kurabayashi, Stefan Sosnowski, Sandra Hirche

This paper focuses on the trajectory optimization of an underwater suspended
robotic system comprising an uncrewed surface vessel (USV) and an uncrewed
underwater vehicle (UUV) for autonomous litter collection. The key challenge
lies in the significant uncertainty in drag and weight parameters introduced by
the collected litter. We propose a dynamical model for the coupled UUV-USV
system in the primary plane of motion and a risk-aware optimization approach
incorporating parameter uncertainty and noise to ensure safe interactions with
the environment. A stochastic optimization problem is solved using a
conditional value-at-risk framework. Simulations demonstrate that our approach
reduces collision risks and energy consumption, highlighting its reliability
compared to existing control methods.

### 2. [Feature-Based Belief Aggregation for Partially Observable Markov Decision Problems](http://arxiv.org/pdf/2507.04646v1)

Authors: Yuchao Li, Kim Hammar, Dimitri Bertsekas

We consider a finite-state partially observable Markov decision problem
(POMDP) with an infinite horizon and a discounted cost, and we propose a new
method for computing a cost function approximation that is based on features
and aggregation. In particular, using the classical belief-space formulation,
we construct a related Markov decision problem (MDP) by first aggregating the
unobservable states into feature states, and then introducing representative
beliefs over these feature states. This two-stage aggregation approach
facilitates the use of dynamic programming methods for solving the aggregate
problem and provides additional design flexibility. The optimal cost function
of the aggregate problem can in turn be used within an on-line approximation in
value space scheme for the original POMDP. We derive a new bound on the
approximation error of our scheme. In addition, we establish conditions under
which the cost function approximation provides a lower bound for the optimal
cost. Finally, we present a biased aggregation approach, which leverages an
optimal cost function estimate to improve the quality of the approximation
error of the aggregate problem.

### 3. [Higher-Order Harmonics Reduction in Reset-Based Control Systems: Application to Precision Positioning Systems](http://arxiv.org/pdf/2507.04707v1)

Authors: S. Ali Hosseini, Nima Karbasizadeh, S. Hassan HosseiniNia

To address the limitations imposed by Bode's gain-phase relationship in
linear controllers, a reset-based filter called the Constant in gain- Lead in
phase (CgLp) filter has been introduced. This filter consists of a reset
element and a linear lead filter. However, the sequencing of these two
components has been a topic of debate. Positioning the lead filter before the
reset element in the loop leads to noise amplification in the reset signal,
whereas placing the lead filter after the reset element results in the
magnification of higher-order harmonics. This study introduces a tunable lead
CgLp structure in which the lead filter is divided into two segments, enabling
a balance between noise reduction and higher-order harmonics mitigation.
Additionally, a filtering technique is proposed, employing a
target-frequency-based approach to mitigate nonlinearity in reset control
systems in the presence of noise. The effectiveness of the proposed methods in
reducing nonlinearity is demonstrated through both frequency domain and
time-domain analyses using a simulated precision positioning system as a case
study.

### 4. [Multi-Objective Nonlinear Power Split Control For BESS With Real-Time Simulation Feedback](http://arxiv.org/pdf/2507.04800v1)

Authors: Vivek Teja Tanjavooru, Prashant Pant, Thomas Hamacher, Holger Hesse

This paper presents a mixed-integer, nonlinear, multi-objective optimization
strategy for optimal power allocation among parallel strings in Battery Energy
Storage Systems (BESS). High-fidelity control is achieved by co-simulating the
optimizer with a BESS electro-thermal simulation that models spatial thermal
dynamics of the battery, providing real-time State of Charge (SOC) and
temperature feedback. The optimizer prioritizes reliability by enforcing power
availability as a hard constraint and penalizing battery thermal derating.
Within these bounds, the controller performs a Pareto sweep on the relative
weights of inverter and battery losses to balance the trade-off between
inverter efficiency and battery efficiency. The inverter loss model is based on
an empirical lookup table (LUT) derived from a commercial inverter system,
while the battery thermal loss model uses SOC and temperature-dependent
internal resistance, with electric current computed from the battery Equivalent
Circuit Model (ECM). When the optimization was applied to a two-string BESS,
the competing effects of inverter and battery losses on system availability and
thermal derating were observed. The balanced operation yielded improvements of
1% in battery efficiency, 1.5% in inverter efficiency, and 2% in derating
efficiency, while maintaining higher availability. Additionally, a 5 degrees C
reduction in BESS peak temperature also suggests reduced thermal stress without
compromising availability.

### 5. [Accounting for Subsystem Aging Variability in Battery Energy Storage System Optimization](http://arxiv.org/pdf/2507.04813v1)

Authors: Melina Grane, Martin Cornejo, Holger Hesse, Andreas Jossen

This paper presents a degradation-cost-aware optimization framework for
multi-string battery energy storage systems, emphasizing the impact of
inhomogeneous subsystem-level aging in operational decision-making. We evaluate
four scenarios for an energy arbitrage scenario, that vary in model precision
and treatment of aging costs. Key performance metrics include operational
revenue, power schedule mismatch, missed revenues, capacity losses, and revenue
generated per unit of capacity loss. Our analysis reveals that ignoring
heterogeneity of subunits may lead to infeasible dispatch plans and reduced
revenues. In contrast, combining accurate representation of degraded subsystems
and the consideration of aging costs in the objective function improves
operational accuracy and economic efficiency of BESS with heterogeneous aged
subunits. The fully informed scenario, which combines aging-cost-aware
optimization with precise string-level modeling, achieves 21% higher revenue
per unit of SOH loss compared to the baseline scenario. These findings
highlight that modeling aging heterogeneity is not just a technical refinement
but may become a crucial enabler for maximizing both short-term profitability
and long-term asset value in particular for long BESS usage scenarios.

### 6. [Force-IMU Fusion-Based Sensing Acupuncture Needle and Quantitative Analysis System for Acupuncture Manipulations](http://arxiv.org/pdf/2507.04821v1)

Authors: Peng Tian, Kang Yu, Tianyun Jiang, Yuqi Wang, Haiying Zhang, Hao Yang, Yunfeng Wang, Jun Zhang, Shuo Gao, Junhong Gao

Acupuncture, one of the key therapeutic methods in Traditional Chinese
Medicine (TCM), has been widely adopted in various clinical fields.
Quantitative research on acupuncture manipulation parameters is critical to
achieve standardized techniques. However, quantitative mechanical detection of
acupuncture parameters remains limited. This study establishes a kinematic and
dynamic model of acupuncture, identifying key parameters such as
lifting-thrusting force, acceleration, velocity, displacement, as well as
twirling-rotating angular velocity and angle. To measure these critical
parameters, we propose a quantitative system comprising a sensing needle
equipped with a force sensor and an inertial measurement unit (IMU), as well as
an external camera module to capture image information. By fusing visual and
IMU data, we accurately identify the stationary or motion states of the needle,
enabling segmented computation of lifting-thrusting velocity and displacement.
The experimental results demonstrate that the sensing needle achieves
comprehensive detection with high precision, featuring a nonlinearity error of
0.45% in force measurement and an RMSE of 1.2 mm in displacement. The extracted
parameters provide an objective description of the operational characteristics
and motion patterns of the four basic acupuncture manipulations. These findings
provide valuable tools and methods for research in acupuncture standardization.

### 7. [A Corrective Frequency-Constrained Unit Commitment with Data-driven Estimation of Optimal UFLS in Island Power Systems](http://arxiv.org/pdf/2507.05062v1)

Authors: Miad Sarvarizadeh, Lukas Sigrist, Almudena Rouco, Mohammad Rajabdorri, Enrique Lobato

This paper presents a novel corrective \gls{fcuc} formulation for island
power systems by implementing data-driven constraint learning to estimate the
optimal \gls{ufls}. The Tobit model is presented to estimate the optimal amount
of \gls{ufls} using the initial rate of change of frequency. The proposed
formulation enables co-optimizing operation costs and \gls{ufls}. The aim is to
account for optimal \gls{ufls} occurrences during operation planning, without
increasing them. This would potentially reduce system operation costs by
relaxing the reserve requirement constraint. The performance of the proposed
formulation has been analyzed for a Spanish island power system through various
simulations. Different daily demand profiles are analyzed to demonstrate the
effectiveness of the proposed formulation. Additionally, a sensitivity analysis
is conducted to demonstrate the effects of changing the cost associated with
\gls{ufls}. The corrective \gls{fcuc} is shown to be capable of reducing system
operation costs without jeopardizing the quality of the frequency response in
terms of \gls{ufls} occurrence.

### 8. [A Comparative Study on Frequency-Constrained Unit Commitment Approaches in Island Power Systems](http://arxiv.org/pdf/2507.05079v1)

Authors: Miad Sarvarizadeh, Mohammad Rajabdorri, Enrique Lobato, Lukas Sigrist

The increasing penetration of renewable energy sources reduces rotating
inertia and even frequency control capacity, affecting frequency stability.
This challenge is significant in \gls{ips} that already suffer from low inertia
and frequency control capacity. This paper presents a comparative study on
different \gls{fcuc} formulations applied to \gls{ips}. Then, by considering
under-frequency load shedding as a significant measure of frequency stability
in \gls{ips}, two indices are presented to fully compare the formulations from
system benefits and computational burden perspectives. Simulations conducted on
a real Spanish island show that the data-driven corrective \gls{fcuc}
formulation has the most advantages among other formulations.

### 9. [On-Demand Multimedia Delivery in 6G: An Optimal-Cost Steiner Tree Approach](http://arxiv.org/pdf/2507.04589v1)

Authors: Zien Wang, Xiucheng Wang, Nan Cheng, Wenchao Xu, Wei Quan, Ruijin Sun, Conghao Zhou

The exponential growth of multimedia data traffic in 6G networks poses
unprecedented challenges for immersive communication, where
ultra-high-definition, multi-quality streaming must be delivered on demand
while minimizing network operational costs. Traditional routing approaches,
such as shortest-path algorithms, fail to optimize flow multiplexing across
multiple destinations, while conventional Steiner tree methods cannot
accommodate heterogeneous quality-of-service (QoS) requirements-a critical need
for 6G's personalized services. In this paper, we address a fundamental but
unsolved challenge: the minimum flow problem (MFP) with multi-destination,
heterogeneous outflow demands, which is pivotal for efficient multimedia
distribution such as adaptive-resolution video streaming. To overcome the
limitations of existing methods, we propose a two-stage dynamic
programming-enhanced On-demand Steiner Tree (OST) algorithm, the first approach
that jointly optimizes flow aggregation and QoS-aware path selection for
arbitrary outflow requirements. We rigorously prove the optimality of OST using
mathematical induction, demonstrating that it guarantees the minimum-cost
multicast flow under differentiated service constraints. Extensive experiments
in 6G-like multimedia transmission scenarios show that OST reduces total
network flow by over 10% compared to state-of-the-art methods while ensuring
on-demand QoS fulfillment. The complete code is available at
https://github.com/UNIC-Lab/OST.

### 10. [Exploring Core and Periphery Precepts in Biological and Artificial Intelligence: An Outcome-Based Perspective](http://arxiv.org/pdf/2507.04594v1)

Authors: Niloofar Shadab, Tyler Cody, Alejandro Salado, Taylan G. Topcu, Mohammad Shadab, Peter Beling

Engineering methodologies predominantly revolve around established principles
of decomposition and recomposition. These principles involve partitioning
inputs and outputs at the component level, ensuring that the properties of
individual components are preserved upon composition. However, this view does
not transfer well to intelligent systems, particularly when addressing the
scaling of intelligence as a system property. Our prior research contends that
the engineering of general intelligence necessitates a fresh set of overarching
systems principles. As a result, we introduced the "core and periphery"
principles, a novel conceptual framework rooted in abstract systems theory and
the Law of Requisite Variety. In this paper, we assert that these abstract
concepts hold practical significance. Through empirical evidence, we illustrate
their applicability to both biological and artificial intelligence systems,
bridging abstract theory with real-world implementations. Then, we expand on
our previous theoretical framework by mathematically defining core-dominant vs
periphery-dominant systems.

### Machine Learning (Statistics Category)

### 1. [A General Class of Model-Free Dense Precision Matrix Estimators](http://arxiv.org/pdf/2507.04663v1)

Authors: Mehmet Caner Agostino Capponi Mihailo Stojnic

We introduce prototype consistent model-free, dense precision matrix
estimators that have broad application in economics. Using quadratic form
concentration inequalities and novel algebraic characterizations of confounding
dimension reductions, we are able to: (i) obtain non-asymptotic bounds for
precision matrix estimation errors and also (ii) consistency in high
dimensions; (iii) uncover the existence of an intrinsic signal-to-noise --
underlying dimensions tradeoff; and (iv) avoid exact population sparsity
assumptions. In addition to its desirable theoretical properties, a thorough
empirical study of the S&P 500 index shows that a tuning parameter-free special
case of our general estimator exhibits a doubly ascending Sharpe Ratio pattern,
thereby establishing a link with the famous double descent phenomenon
dominantly present in recent statistical and machine learning literature.

### 2. [Intervening to learn and compose disentangled representations](http://arxiv.org/pdf/2507.04754v1)

Authors: Alex Markham, Jeri A. Chang, Isaac Hirsch, Liam Solus, Bryon Aragam

In designing generative models, it is commonly believed that in order to
learn useful latent structure, we face a fundamental tension between
expressivity and structure. In this paper we challenge this view by proposing a
new approach to training arbitrarily expressive generative models that
simultaneously learn disentangled latent structure. This is accomplished by
adding a simple decoder-only module to the head of an existing decoder block
that can be arbitrarily complex. The module learns to process concept
information by implicitly inverting linear representations from an encoder.
Inspired by the notion of intervention in causal graphical models, our module
selectively modifies its architecture during training, allowing it to learn a
compact joint model over different contexts. We show how adding this module
leads to disentangled representations that can be composed for
out-of-distribution generation. To further validate our proposed approach, we
prove a new identifiability result that extends existing work on identifying
structured representations in nonlinear models.

### 3. [Distribution-dependent Generalization Bounds for Tuning Linear Regression Across Tasks](http://arxiv.org/pdf/2507.05084v1)

Authors: Maria-Florina Balcan, Saumya Goyal, Dravyansh Sharma

Modern regression problems often involve high-dimensional data and a careful
tuning of the regularization hyperparameters is crucial to avoid overly complex
models that may overfit the training data while guaranteeing desirable
properties like effective variable selection. We study the recently introduced
direction of tuning regularization hyperparameters in linear regression across
multiple related tasks. We obtain distribution-dependent bounds on the
generalization error for the validation loss when tuning the L1 and L2
coefficients, including ridge, lasso and the elastic net. In contrast, prior
work develops bounds that apply uniformly to all distributions, but such bounds
necessarily degrade with feature dimension, d. While these bounds are shown to
be tight for worst-case distributions, our bounds improve with the "niceness"
of the data distribution. Concretely, we show that under additional assumptions
that instances within each task are i.i.d. draws from broad well-studied
classes of distributions including sub-Gaussians, our generalization bounds do
not get worse with increasing d, and are much sharper than prior work for very
large d. We also extend our results to a generalization of ridge regression,
where we achieve tighter bounds that take into account an estimate of the mean
of the ground truth distribution.

### 4. [DICE: Discrete inverse continuity equation for learning population dynamics](http://arxiv.org/pdf/2507.05107v1)

Authors: Tobias Blickhan, Jules Berman, Andrew Stuart, Benjamin Peherstorfer

We introduce the Discrete Inverse Continuity Equation (DICE) method, a
generative modeling approach that learns the evolution of a stochastic process
from given sample populations at a finite number of time points. Models learned
with DICE capture the typically smooth and well-behaved population dynamics,
rather than the dynamics of individual sample trajectories that can exhibit
complex or even chaotic behavior. The DICE loss function is developed
specifically to be invariant, even in discrete time, to spatially constant but
time-varying spurious constants that can emerge during training; this
invariance increases training stability and robustness. Generating a trajectory
of sample populations with DICE is fast because samples evolve directly in the
time interval over which the stochastic process is formulated, in contrast to
approaches that condition on time and then require multiple sampling steps per
time step. DICE is stable to train, in situations where other methods for
learning population dynamics fail, and DICE generates representative samples
with orders of magnitude lower costs than methods that have to condition on
time. Numerical experiments on a wide range of problems from random waves,
Vlasov-Poisson instabilities and high-dimensional chaos are included to justify
these assertions.

### 5. [QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions](http://arxiv.org/pdf/2507.05220v1)

Authors: Zhun Deng, Thomas P Zollo, Benjamin Eyre, Amogh Inamdar, David Madras, Richard Zemel

As machine learning models grow increasingly competent, their predictions can
supplement scarce or expensive data in various important domains. In support of
this paradigm, algorithms have emerged to combine a small amount of
high-fidelity observed data with a much larger set of imputed model outputs to
estimate some quantity of interest. Yet current hybrid-inference tools target
only means or single quantiles, limiting their applicability for many critical
domains and use cases. We present QuEst, a principled framework to merge
observed and imputed data to deliver point estimates and rigorous confidence
intervals for a wide family of quantile-based distributional measures. QuEst
covers a range of measures, from tail risk (CVaR) to population segments such
as quartiles, that are central to fields such as economics, sociology,
education, medicine, and more. We extend QuEst to multidimensional metrics, and
introduce an additional optimization technique to further reduce variance in
this and other hybrid estimators. We demonstrate the utility of our framework
through experiments in economic modeling, opinion polling, and language model
auto-evaluation.

### 6. [Optimal Model Selection for Conformalized Robust Optimization](http://arxiv.org/pdf/2507.04716v1)

Authors: Yajie Bao, Yang Hu, Haojie Ren, Peng Zhao, Changliang Zou

In decision-making under uncertainty, Contextual Robust Optimization (CRO)
provides reliability by minimizing the worst-case decision loss over a
prediction set, hedging against label variability. While recent advances use
conformal prediction to construct prediction sets for machine learning models,
the downstream decisions critically depend on model selection. This paper
introduces novel model selection frameworks for CRO that unify robustness
control with decision risk minimization. We first propose Conformalized Robust
Optimization with Model Selection (CROMS), which automatically selects models
to approximately minimize the average decision risk in CRO solutions. We
develop two algorithms: E-CROMS, which is computationally efficient, and
F-CROMS, which enjoys a marginal robustness guarantee in finite samples.
Further, we introduce Conformalized Robust Optimization with Individualized
Model Selection (CROiMS), which performs individualized model selection by
minimizing the conditional decision risk given the covariate of test data. This
framework advances conformal prediction methodology by enabling covariate-aware
model selection. Theoretically, CROiMS achieves asymptotic conditional
robustness and decision efficiency under mild assumptions. Numerical results
demonstrate significant improvements in decision efficiency and robustness
across diverse synthetic and real-world applications, outperforming baseline
approaches.

### 7. [Sure Convergence and Constructive Universal Approximation for Multi-Layer Neural Networks](http://arxiv.org/pdf/2507.04779v1)

Authors: Chien-Ming Chi

We propose a new neural network model, 01Neuro, built on indicator activation
neurons. Its boosted variant possesses two key statistical properties: (1) Sure
Convergence, where model optimization can be achieved with high probability
given sufficient computational resources; and (2) Constructive Universal
Approximation: In the infinite sample setting, the model can approximate any
finite sum of measurable functions, each depending on only k out of p input
features, provided the architecture is properly tuned. Unlike most universal
approximation results that are agnostic to training procedures, our guarantees
are directly tied to the model's explicit construction and optimization
algorithm. To improve prediction stability, we integrate stochastic training
and bagging into the boosted 01Neuro framework. Empirical evaluations on
simulated and real-world tabular datasets with small to medium sample sizes
highlight its strengths: effective approximation of interaction components
(multiplicative terms), stable prediction performance (comparable to Random
Forests), robustness to many noisy features, and insensitivity to feature
scaling. A major limitation of the current implementation of boosted 01Neuro is
its higher computational cost, which is approximately 5 to 30 times that of
Random Forests and XGBoost.

### 8. [Vecchia-Inducing-Points Full-Scale Approximations for Gaussian Processes](http://arxiv.org/pdf/2507.05064v1)

Authors: Tim Gyger, Reinhard Furrer, Fabio Sigrist

Gaussian processes are flexible, probabilistic, non-parametric models widely
used in machine learning and statistics. However, their scalability to large
data sets is limited by computational constraints. To overcome these
challenges, we propose Vecchia-inducing-points full-scale (VIF) approximations
combining the strengths of global inducing points and local Vecchia
approximations. Vecchia approximations excel in settings with low-dimensional
inputs and moderately smooth covariance functions, while inducing point methods
are better suited to high-dimensional inputs and smoother covariance functions.
Our VIF approach bridges these two regimes by using an efficient
correlation-based neighbor-finding strategy for the Vecchia approximation of
the residual process, implemented via a modified cover tree algorithm. We
further extend our framework to non-Gaussian likelihoods by introducing
iterative methods that substantially reduce computational costs for training
and prediction by several orders of magnitudes compared to Cholesky-based
computations when using a Laplace approximation. In particular, we propose and
compare novel preconditioners and provide theoretical convergence results.
Extensive numerical experiments on simulated and real-world data sets show that
VIF approximations are both computationally efficient as well as more accurate
and numerically stable than state-of-the-art alternatives. All methods are
implemented in the open source C++ library GPBoost with high-level Python and R
interfaces.

### 9. [Mutual Information Optimal Control of Discrete-Time Linear Systems](http://arxiv.org/pdf/2507.04712v1)

Authors: Shoju Enami, Kenji Kashima

In this paper, we formulate a mutual information optimal control problem
(MIOCP) for discrete-time linear systems. This problem can be regarded as an
extension of a maximum entropy optimal control problem (MEOCP). Differently
from the MEOCP where the prior is fixed to the uniform distribution, the MIOCP
optimizes the policy and prior simultaneously. As analytical results, under the
policy and prior classes consisting of Gaussian distributions, we derive the
optimal policy and prior of the MIOCP with the prior and policy fixed,
respectively. Using the results, we propose an alternating minimization
algorithm for the MIOCP. Through numerical experiments, we discuss how our
proposed algorithm works.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-08 PST.

### 1. [DSF-YOLO for robust multiscale traffic sign detection under adverse weather conditions](https://www.nature.com/articles/s41598-025-02877-0)

Authors: Jun Li et al.

### 2. [Graph-based vision transformer with sparsity for training on small datasets from scratch](https://www.nature.com/articles/s41598-025-10408-0)

Authors: Peng Li et al.

### 3. [Automated underwater plectropomus leopardus phenotype measurement through cylinder](https://www.nature.com/articles/s41598-025-08863-w)

Authors: Mengran Liu et al.

### 4. [Cross paradigm fusion of federated and continual learning on multilayer perceptron mixer architecture for incremental thoracic infection diagnosis](https://www.nature.com/articles/s41598-025-06077-8)

Authors: Tianshuo Zhou et al.

### 5. [An enhanced deep learning approach for speaker diarization using TitaNet, MarbelNet and time delay network](https://www.nature.com/articles/s41598-025-09385-1)

Authors: Muzamil Ahmed et al.

### 6. [A simple interpolation-based data augmentation method for implicit sentiment identification](https://www.nature.com/articles/s41598-025-00197-x)

Authors: Yuxia Zhao et al.

### 7. [Active learning algorithm for alleviating the user cold start problem of recommender systems](https://www.nature.com/articles/s41598-025-09708-2)

Authors: Toon De Pessemier et al.

### 8. [Batch gradient based smoothing L2/3 regularization for training pi-sigma higher-order networks](https://www.nature.com/articles/s41598-025-08324-4)

Authors: Khidir Shaib Mohamed et al.

### 9. [MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes](https://www.nature.com/articles/s41597-025-05493-9)

Authors: Shiyan Li et al.

### 10. [A directed greybox fuzzer for windows applications](https://www.nature.com/articles/s41598-025-09777-3)

Authors: Xin Ren et al.

### 11. [A novel model for expanding horizons in sign Language recognition](https://www.nature.com/articles/s41598-025-09643-2)

Authors: Esraa Hassan et al.

### 12. [SFMANet: A Spatial-Frequency multi-scale attention network for stroke lesion segmentation](https://www.nature.com/articles/s41598-025-10506-z)

Authors: Hualing Li et al.

