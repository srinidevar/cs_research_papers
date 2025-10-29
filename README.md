# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-28 17:00:30.478622 PST.

### Artificial Intelligence

### 1. [Exploring Structures of Inferential Mechanisms through Simplistic Digital Circuits](http://arxiv.org/pdf/2510.22883v1)

Authors: Giovanni Sileno, Jean-Louis Dessalles

Cognitive studies and artificial intelligence have developed distinct models
for various inferential mechanisms (categorization, induction, abduction,
causal inference, contrast, merge, ...). Yet, both natural and artificial views
on cognition lack apparently a unifying framework. This paper formulates a
speculative answer attempting to respond to this gap. To postulate on
higher-level activation processes from a material perspective, we consider
inferential mechanisms informed by symbolic AI modelling techniques, through
the simplistic lenses of electronic circuits based on logic gates. We observe
that a logic gate view entails a different treatment of implication and
negation compared to standard logic and logic programming. Then, by
combinatorial exploration, we identify four main forms of dependencies that can
be realized by these inferential circuits. Looking at how these forms are
generally used in the context of logic programs, we identify eight common
inferential patterns, exposing traditionally distinct inferential mechanisms in
an unifying framework. Finally, following a probabilistic interpretation of
logic programs, we unveil inner functional dependencies. The paper concludes
elaborating in what sense, even if our arguments are mostly informed by
symbolic means and digital systems infrastructures, our observations may
pinpoint to more generally applicable structures.

### 2. [ProfileXAI: User-Adaptive Explainable AI](http://arxiv.org/pdf/2510.22998v1)

Authors: Gilber A. Corrales, Carlos Andrés Ferro Sánchez, Reinel Tabares-Soto, Jesús Alfonso López Sotelo, Gonzalo A. Ruz, Johan Sebastian Piña Durán

ProfileXAI is a model- and domain-agnostic framework that couples post-hoc
explainers (SHAP, LIME, Anchor) with retrieval - augmented LLMs to produce
explanations for different types of users. The system indexes a multimodal
knowledge base, selects an explainer per instance via quantitative criteria,
and generates grounded narratives with chat-enabled prompting. On Heart Disease
and Thyroid Cancer datasets, we evaluate fidelity, robustness, parsimony, token
use, and perceived quality. No explainer dominates: LIME achieves the best
fidelity-robustness trade-off (Infidelity $\le 0.30$, $L<0.7$ on Heart
Disease); Anchor yields the sparsest, low-token rules; SHAP attains the highest
satisfaction ($\bar{x}=4.1$). Profile conditioning stabilizes tokens ($\sigma
\le 13\%$) and maintains positive ratings across profiles ($\bar{x}\ge 3.7$,
with domain experts at $3.77$), enabling efficient and trustworthy
explanations.

### 3. [From Prompt Optimization to Multi-Dimensional Credibility Evaluation: Enhancing Trustworthiness of Chinese LLM-Generated Liver MRI Reports](http://arxiv.org/pdf/2510.23008v1)

Authors: Qiuli Wang, Xiaoming Li, Jie Chen, Yongxu Liu, Xingpeng Zhang, Chen Liu, Wei Chen

Large language models (LLMs) have demonstrated promising performance in
generating diagnostic conclusions from imaging findings, thereby supporting
radiology reporting, trainee education, and quality control. However,
systematic guidance on how to optimize prompt design across different clinical
contexts remains underexplored. Moreover, a comprehensive and standardized
framework for assessing the trustworthiness of LLM-generated radiology reports
is yet to be established. This study aims to enhance the trustworthiness of
LLM-generated liver MRI reports by introducing a Multi-Dimensional Credibility
Assessment (MDCA) framework and providing guidance on institution-specific
prompt optimization. The proposed framework is applied to evaluate and compare
the performance of several advanced LLMs, including Kimi-K2-Instruct-0905,
Qwen3-235B-A22B-Instruct-2507, DeepSeek-V3, and
ByteDance-Seed-OSS-36B-Instruct, using the SiliconFlow platform.

### 4. [A Survey of AI Scientists: Surveying the automatic Scientists and Research](http://arxiv.org/pdf/2510.23045v1)

Authors: Guiyao Tie, Pan Zhou, Lichao Sun

Artificial intelligence is undergoing a profound transition from a
computational instrument to an autonomous originator of scientific knowledge.
This emerging paradigm, the AI scientist, is architected to emulate the
complete scientific workflow-from initial hypothesis generation to the final
synthesis of publishable findings-thereby promising to fundamentally reshape
the pace and scale of discovery. However, the rapid and unstructured
proliferation of these systems has created a fragmented research landscape,
obscuring overarching methodological principles and developmental trends. This
survey provides a systematic and comprehensive synthesis of this domain by
introducing a unified, six-stage methodological framework that deconstructs the
end-to-end scientific process into: Literature Review, Idea Generation,
Experimental Preparation, Experimental Execution, Scientific Writing, and Paper
Generation. Through this analytical lens, we chart the field's evolution from
early Foundational Modules (2022-2023) to integrated Closed-Loop Systems
(2024), and finally to the current frontier of Scalability, Impact, and
Human-AI Collaboration (2025-present). By rigorously synthesizing these
developments, this survey not only clarifies the current state of autonomous
science but also provides a critical roadmap for overcoming remaining
challenges in robustness and governance, ultimately guiding the next generation
of systems toward becoming trustworthy and indispensable partners in human
scientific inquiry.

### 5. [TLCD: A Deep Transfer Learning Framework for Cross-Disciplinary Cognitive Diagnosis](http://arxiv.org/pdf/2510.23062v1)

Authors: Zhifeng Wang, Meixin Su, Yang Yang, Chunyan Zeng, Lizhi Ye

Driven by the dual principles of smart education and artificial intelligence
technology, the online education model has rapidly emerged as an important
component of the education industry. Cognitive diagnostic technology can
utilize students' learning data and feedback information in educational
evaluation to accurately assess their ability level at the knowledge level.
However, while massive amounts of information provide abundant data resources,
they also bring about complexity in feature extraction and scarcity of
disciplinary data. In cross-disciplinary fields, traditional cognitive
diagnostic methods still face many challenges. Given the differences in
knowledge systems, cognitive structures, and data characteristics between
different disciplines, this paper conducts in-depth research on neural network
cognitive diagnosis and knowledge association neural network cognitive
diagnosis, and proposes an innovative cross-disciplinary cognitive diagnosis
method (TLCD). This method combines deep learning techniques and transfer
learning strategies to enhance the performance of the model in the target
discipline by utilizing the common features of the main discipline. The
experimental results show that the cross-disciplinary cognitive diagnosis model
based on deep learning performs better than the basic model in
cross-disciplinary cognitive diagnosis tasks, and can more accurately evaluate
students' learning situation.

### 6. [Lost in Tokenization: Context as the Key to Unlocking Biomolecular Understanding in Scientific LLMs](http://arxiv.org/pdf/2510.23127v1)

Authors: Kai Zhuang, Jiawei Zhang, Yumou Liu, Hanqun Cao, Chunbin Gu, Mengdi Liu, Zhangyang Gao, Zitong Jerry Wang, Xuanhe Zhou, Pheng-Ann Heng, Lijun Wu, Conghui He, Cheng Tan

Scientific Large Language Models (Sci-LLMs) have emerged as a promising
frontier for accelerating biological discovery. However, these models face a
fundamental challenge when processing raw biomolecular sequences: the
tokenization dilemma. Whether treating sequences as a specialized language,
risking the loss of functional motif information, or as a separate modality,
introducing formidable alignment challenges, current strategies fundamentally
limit their reasoning capacity. We challenge this sequence-centric paradigm by
positing that a more effective strategy is to provide Sci-LLMs with high-level
structured context derived from established bioinformatics tools, thereby
bypassing the need to interpret low-level noisy sequence data directly. Through
a systematic comparison of leading Sci-LLMs on biological reasoning tasks, we
tested three input modes: sequence-only, context-only, and a combination of
both. Our findings are striking: the context-only approach consistently and
substantially outperforms all other modes. Even more revealing, the inclusion
of the raw sequence alongside its high-level context consistently degrades
performance, indicating that raw sequences act as informational noise, even for
models with specialized tokenization schemes. These results suggest that the
primary strength of existing Sci-LLMs lies not in their nascent ability to
interpret biomolecular syntax from scratch, but in their profound capacity for
reasoning over structured, human-readable knowledge. Therefore, we argue for
reframing Sci-LLMs not as sequence decoders, but as powerful reasoning engines
over expert knowledge. This work lays the foundation for a new class of hybrid
scientific AI agents, repositioning the developmental focus from direct
sequence interpretation towards high-level knowledge synthesis. The code is
available at github.com/opendatalab-raise-dev/CoKE.

### 7. [Guiding Skill Discovery with Foundation Models](http://arxiv.org/pdf/2510.23167v1)

Authors: Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Vincent François-Lavet, Edward S. Hu

Learning diverse skills without hand-crafted reward functions could
accelerate reinforcement learning in downstream tasks. However, existing skill
discovery methods focus solely on maximizing the diversity of skills without
considering human preferences, which leads to undesirable behaviors and
possibly dangerous skills. For instance, a cheetah robot trained using previous
methods learns to roll in all directions to maximize skill diversity, whereas
we would prefer it to run without flipping or entering hazardous areas. In this
work, we propose a Foundation model Guided (FoG) skill discovery method, which
incorporates human intentions into skill discovery through foundation models.
Specifically, FoG extracts a score function from foundation models to evaluate
states based on human intentions, assigning higher values to desirable states
and lower to undesirable ones. These scores are then used to re-weight the
rewards of skill discovery algorithms. By optimizing the re-weighted skill
discovery rewards, FoG successfully learns to eliminate undesirable behaviors,
such as flipping or rolling, and to avoid hazardous areas in both state-based
and pixel-based tasks. Interestingly, we show that FoG can discover skills
involving behaviors that are difficult to define. Interactive visualisations
are available from https://sites.google.com/view/submission-fog.

### 8. [AUPO -- Abstracted Until Proven Otherwise: A Reward Distribution Based Abstraction Algorithm](http://arxiv.org/pdf/2510.23214v1)

Authors: Robin Schmöcker, Alexander Dockhorn, Bodo Rosenhahn

We introduce a novel, drop-in modification to Monte Carlo Tree Search's
(MCTS) decision policy that we call AUPO. Comparisons based on a range of IPPC
benchmark problems show that AUPO clearly outperforms MCTS. AUPO is an
automatic action abstraction algorithm that solely relies on reward
distribution statistics acquired during the MCTS. Thus, unlike other automatic
abstraction algorithms, AUPO requires neither access to transition
probabilities nor does AUPO require a directed acyclic search graph to build
its abstraction, allowing AUPO to detect symmetric actions that
state-of-the-art frameworks like ASAP struggle with when the resulting
symmetric states are far apart in state space. Furthermore, as AUPO only
affects the decision policy, it is not mutually exclusive with other
abstraction techniques that only affect the tree search.

### 9. [CNOT Minimal Circuit Synthesis: A Reinforcement Learning Approach](http://arxiv.org/pdf/2510.23304v1)

Authors: Riccardo Romanello, Daniele Lizzio Bosco, Jacopo Cossio, Dusan Sutulovic, Giuseppe Serra, Carla Piazza, Paolo Burelli

CNOT gates are fundamental to quantum computing, as they facilitate
entanglement, a crucial resource for quantum algorithms. Certain classes of
quantum circuits are constructed exclusively from CNOT gates. Given their
widespread use, it is imperative to minimise the number of CNOT gates employed.
This problem, known as CNOT minimisation, remains an open challenge, with its
computational complexity yet to be fully characterised. In this work, we
introduce a novel reinforcement learning approach to address this task. Instead
of training multiple reinforcement learning agents for different circuit sizes,
we use a single agent up to a fixed size $m$. Matrices of sizes different from
m are preprocessed using either embedding or Gaussian striping. To assess the
efficacy of our approach, we trained an agent with m = 8, and evaluated it on
matrices of size n that range from 3 to 15. The results we obtained show that
our method overperforms the state-of-the-art algorithm as the value of n
increases.

### 10. [Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens](http://arxiv.org/pdf/2510.23410v1)

Authors: Jiahao Ji, Tianyu Wang, Yeshu Li, Yushen Huo, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng

Auto-bidding is crucial in facilitating online advertising by automatically
providing bids for advertisers. While previous work has made great efforts to
model bidding environments for better ad performance, it has limitations in
generalizability across environments since these models are typically tailored
for specific bidding scenarios. To this end, we approach the
scenario-independent principles through a unified function that estimates the
achieved effect under specific bids, such as budget consumption, gross
merchandise volume (GMV), page views, etc. Then, we propose a bidding
foundation model Bid2X to learn this fundamental function from data in various
scenarios. Our Bid2X is built over uniform series embeddings that encode
heterogeneous data through tailored embedding methods. To capture complex
inter-variable and dynamic temporal dependencies in bidding data, we propose
two attention mechanisms separately treating embeddings of different variables
and embeddings at different times as attention tokens for representation
learning. On top of the learned variable and temporal representations, a
variable-aware fusion module is used to perform adaptive bidding outcome
prediction. To model the unique bidding data distribution, we devise a
zero-inflated projection module to incorporate the estimated non-zero
probability into its value prediction, which makes up a joint optimization
objective containing classification and regression. The objective is proven to
converge to the zero-inflated distribution. Our model has been deployed on the
ad platform in Taobao, one of the world's largest e-commerce platforms. Offline
evaluation on eight datasets exhibits Bid2X's superiority compared to various
baselines and its generality across different scenarios. Bid2X increased GMV by
4.65% and ROI by 2.44% in online A/B tests, paving the way for bidding
foundation model in computational advertising.

### Hardware Architecture

### 1. [Architecting Scalable Trapped Ion Quantum Computers using Surface Codes](http://arxiv.org/pdf/2510.23519v1)

Authors: Scott Jones, Prakash Murali

Trapped ion (TI) qubits are a leading quantum computing platform. Current TI
systems have less than 60 qubits, but a modular architecture known as the
Quantum Charge-Coupled Device (QCCD) is a promising path to scale up devices.
There is a large gap between the error rates of near-term systems ($10^{-3}$ to
$10^{-4}$) and the requirements of practical applications (below $10^{-9}$). To
bridge this gap, we require Quantum Error Correction (QEC) to build
\emph{logical qubits} that are composed of multiple physical qubits. While
logical qubits have been demonstrated on TI qubits, these demonstrations are
restricted to small codes and systems. There is no clarity on how QCCD systems
should be designed to implement practical-scale QEC. This paper studies how
surface codes, a standard QEC scheme, can be implemented efficiently on
QCCD-based systems. To examine how architectural parameters of a QCCD system
can be tuned for surface codes, we develop a near-optimal topology-aware
compilation method that outperforms existing QCCD compilers by an average of
3.8X in terms of logical clock speed. We use this compiler to examine how
hardware trap capacity, connectivity and electrode wiring choices can be
optimised for surface code implementation. In particular, we demonstrate that
small traps of two ions are surprisingly ideal from both a performance-optimal
and hardware-efficiency standpoint. This result runs counter to prior intuition
that larger traps (20-30 ions) would be preferable, and has the potential to
inform design choices for upcoming systems.

### 2. [BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement](http://arxiv.org/pdf/2510.23472v1)

Authors: Ke Xue, Ruo-Tong Chen, Rong-Xi Tan, Xi Lin, Yunqi Shi, Siyuan Xu, Mingxuan Yuan, Chao Qian

Chip placement is a vital stage in modern chip design as it has a substantial
impact on the subsequent processes and the overall quality of the final chip.
The use of black-box optimization (BBO) for chip placement has a history of
several decades. However, early efforts were limited by immature problem
formulations and inefficient algorithm designs. Recent progress has shown the
effectiveness and efficiency of BBO for chip placement, proving its potential
to achieve state-of-the-art results. Despite these advancements, the field
lacks a unified, BBO-specific benchmark for thoroughly assessing various
problem formulations and BBO algorithms. To fill this gap, we propose
BBOPlace-Bench, the first benchmark designed specifically for evaluating and
developing BBO algorithms for chip placement tasks. It integrates three problem
formulations of BBO for chip placement, and offers a modular, decoupled, and
flexible framework that enables users to seamlessly implement, test, and
compare their own algorithms. BBOPlace-Bench integrates a wide variety of
existing BBO algorithms, including simulated annealing (SA), evolutionary
algorithms (EAs), and Bayesian optimization (BO). Experimental results show
that the problem formulations of mask-guided optimization and hyperparameter
optimization exhibit superior performance than the sequence pair problem
formulation, while EAs demonstrate better overall performance than SA and BO,
especially in high-dimensional search spaces, and also achieve state-of-the-art
performance compared to the mainstream chip placement methods. BBOPlace-Bench
not only facilitates the development of efficient BBO-driven solutions for chip
placement but also broadens the practical application scenarios (which are
urgently needed) for the BBO community. The code of BBOPlace-Bench is available
at https://github.com/lamda-bbo/BBOPlace-Bench.

### Computational Complexity

### 1. [A Critique of Quigley's "A Polynomial Time Algorithm for 3SAT"](http://arxiv.org/pdf/2510.22985v1)

Authors: Nicholas DeJesse, Spencer Lyudovyk, Dhruv Pai

In this paper, we examine Quigley's "A Polynomial Time Algorithm for 3SAT"
[Qui24]. Quigley claims to construct an algorithm that runs in polynomial time
and determines whether a boolean formula in 3CNF form is satisfiable. Such a
result would prove that 3SAT $\in \text{P}$ and thus $\text{P} = \text{NP}$. We
show Quigley's argument is flawed by providing counterexamples to several
lemmas he attempts to use to justify the correctness of his algorithm. We also
provide an infinite class of 3CNF formulas that are unsatisfiable but are
classified as satisfiable by Quigley's algorithm. In doing so, we prove that
Quigley's algorithm fails on certain inputs, and thus his claim that $\text{P}
= \text{NP}$ is not established by his paper.

### 2. [Noisy nonlinear information and entropy numbers](http://arxiv.org/pdf/2510.23213v1)

Authors: David Krieg, Erich Novak, Leszek Plaskota, Mario Ullrich

It is impossible to recover a vector from $\mathbb{R}^m$ with less than $m$
linear measurements, even if the measurements are chosen adaptively. Recently,
it has been shown that one can recover vectors from $\mathbb{R}^m$ with
arbitrary precision using only $O(\log m)$ continuous (even Lipschitz) adaptive
measurements, resulting in an exponential speed-up of continuous information
compared to linear information for various approximation problems. In this
note, we characterize the quality of optimal (dis-)continuous information that
is disturbed by deterministic noise in terms of entropy numbers. This shows
that in the presence of noise the potential gain of continuous over linear
measurements is limited, but significant in some cases.

### Computational Engineering

### 1. [Capsule Network-Based Multimodal Fusion for Mortgage Risk Assessment from Unstructured Data Sources](http://arxiv.org/pdf/2510.22987v1)

Authors: Mahsa Tavakoli, Rohitash Chandra, Cristian Bravo

Mortgage risk assessment traditionally relies on structured financial data,
which is often proprietary, confidential, and costly. In this study, we propose
a novel multimodal deep learning framework that uses cost-free, publicly
available, unstructured data sources, including textual information, images,
and sentiment scores, to generate credit scores that approximate commercial
scorecards. Our framework adopts a two-phase approach. In the unimodal phase,
we identify the best-performing models for each modality, i.e. BERT for text,
VGG for image data, and a multilayer perceptron for sentiment-based features.
In the fusion phase, we introduce the capsule-based fusion network
(FusionCapsNet), a novel fusion strategy inspired by capsule networks, but
fundamentally redesigned for multimodal integration. Unlike standard capsule
networks, our method adapts a specific mechanism in capsule networks to each
modality and restructures the fusion process to preserve spatial, contextual,
and modality-specific information. It also enables adaptive weighting so that
stronger modalities dominate without ignoring complementary signals.
  Our framework incorporates sentiment analysis across distinct news categories
to capture borrower and market dynamics and employs GradCAM-based
visualizations as an interpretability tool. These components are designed
features of the framework, while our results later demonstrate that they
effectively enrich contextual understanding and highlight the influential
factors driving mortgage risk predictions. Our results show that our multimodal
FusionCapsNet framework not only exceeds individual unimodal models but also
outperforms benchmark fusion strategies such as addition, concatenation, and
cross attention in terms of AUC, partial AUC, and F1 score, demonstrating clear
gains in both predictive accuracy and interpretability for mortgage risk
assessment.

### 2. [P1GPT: a multi-agent LLM workflow module for multi-modal financial information analysis](http://arxiv.org/pdf/2510.23032v1)

Authors: Chen-Che Lu, Yun-Cheng Chou, Teng-Ruei Chen

Recent advances in large language models (LLMs) have enabled multi-agent
reasoning systems capable of collaborative decision-making. However, in
financial analysis, most frameworks remain narrowly focused on either isolated
single-agent predictors or loosely connected analyst ensembles, and they lack a
coherent reasoning workflow that unifies diverse data modalities. We introduce
P1GPT, a layered multi-agent LLM framework for multi-modal financial
information analysis and interpretable trading decision support. Unlike prior
systems that emulate trading teams through role simulation, P1GPT implements a
structured reasoning pipeline that systematically fuses technical, fundamental,
and news-based insights through coordinated agent communication and
integration-time synthesis. Backtesting on multi-modal datasets across major
U.S. equities demonstrates that P1GPT achieves superior cumulative and
risk-adjusted returns, maintains low drawdowns, and provides transparent causal
rationales. These findings suggest that structured reasoning workflows, rather
than agent role imitation, offer a scalable path toward explainable and
trustworthy financial AI systems.

### 3. [DRO-Based Computation Offloading and Trajectory Design for Low-Altitude Networks](http://arxiv.org/pdf/2510.23202v1)

Authors: Guanwang Jiang, Ziye Jia, Can Cui, Lijun He, Qiuming Zhu, Qihui Wu

The low-altitude networks (LANs) integrating unmanned aerial vehicles (UAVs)
and high-altitude platforms (HAPs) have become a promising solution for the
rising computation demands. However, the uncertain task sizes and high mobility
of UAVs pose great challenges to guarantee the quality of service. To address
these issues, we propose an LAN architecture where UAVs and HAPs
collaboratively provide computation offloading for ground users. Moreover, the
uncertainty sets are constructed to characterize the uncertain task size, and a
distributionally robust optimization problem is formulated to minimize the
worst-case delay by jointly optimizing the offloading decisions and UAV
trajectories. To solve the mixed-integer min-max optimization problem, we
design the distributionally robust computation offloading and trajectories
optimization algorithm. Specifically, the original problem is figured out by
iteratively solving the outerlayer and inner-layer problems. The convex
outer-layer problem with probability distributions is solved by the
optimization toolkit. As for the inner-layer mixed-integer problem, we employ
the Benders decomposition. The decoupled master problem concerning the binary
offloading decisions is solved by the integer solver, and UAV trajectories in
the sub-problem are optimized via the successive convex approximation.
Simulation results show the proposed algorithm outperforms traditional
optimization methods in balancing the worst-case delay and robustness.

### 4. [Learning the PTM Code through a Coarse-to-Fine, Mechanism-Aware Framework](http://arxiv.org/pdf/2510.23492v1)

Authors: Jingjie Zhang, Hanqun Cao, Zijun Gao, Yu Wang, Shaoning Li, Jun Xu, Cheng Tan, Jun Zhu, Chang-Yu Hsieh, Chunbin Gu, Pheng Ann Heng

Post-translational modifications (PTMs) form a combinatorial "code" that
regulates protein function, yet deciphering this code - linking modified sites
to their catalytic enzymes - remains a central unsolved problem in
understanding cellular signaling and disease. We introduce COMPASS-PTM, a
mechanism-aware, coarse-to-fine learning framework that unifies residue-level
PTM profiling with enzyme-substrate assignment. COMPASS-PTM integrates
evolutionary representations from protein language models with physicochemical
priors and a crosstalk-aware prompting mechanism that explicitly models
inter-PTM dependencies. This design allows the model to learn biologically
coherent patterns of cooperative and antagonistic modifications while
addressing the dual long-tail distribution of PTM data. Across multiple
proteome-scale benchmarks, COMPASS-PTM establishes new state-of-the-art
performance, including a 122% relative F1 improvement in multi-label site
prediction and a 54% gain in zero-shot enzyme assignment. Beyond accuracy, the
model demonstrates interpretable generalization, recovering canonical kinase
motifs and predicting disease-associated PTM rewiring caused by missense
variants. By bridging statistical learning with biochemical mechanism,
COMPASS-PTM unifies site-level and enzyme-level prediction into a single
framework that learns the grammar underlying protein regulation and signaling.

### 5. [GroupSHAP-Guided Integration of Financial News Keywords and Technical Indicators for Stock Price Prediction](http://arxiv.org/pdf/2510.23112v1)

Authors: Minjoo Kim, Jinwoong Kim, Sangjin Park

Recent advances in finance-specific language models such as FinBERT have
enabled the quantification of public sentiment into index-based measures, yet
compressing diverse linguistic signals into single metrics overlooks contextual
nuances and limits interpretability. To address this limitation, explainable AI
techniques, particularly SHAP (SHapley Additive Explanations), have been
employed to identify influential features. However, SHAP's computational cost
grows exponentially with input features, making it impractical for large-scale
text-based financial data. This study introduces a GRU-based forecasting
framework enhanced with GroupSHAP, which quantifies contributions of
semantically related keyword groups rather than individual tokens,
substantially reducing computational burden while preserving interpretability.
We employed FinBERT to embed news articles from 2015 to 2024, clustered them
into coherent semantic groups, and applied GroupSHAP to measure each group's
contribution to stock price movements. The resulting group-level SHAP variables
across multiple topics were used as input features for the prediction model.
Empirical results from one-day-ahead forecasting of the S&P 500 index
throughout 2024 demonstrate that our approach achieves a 32.2% reduction in MAE
and a 40.5% reduction in RMSE compared with benchmark models without the
GroupSHAP mechanism. This research presents the first application of GroupSHAP
in news-driven financial forecasting, showing that grouped sentiment
representations simultaneously enhance interpretability and predictive
performance.

### 6. [Common Task Framework For a Critical Evaluation of Scientific Machine Learning Algorithms](http://arxiv.org/pdf/2510.23166v1)

Authors: Philippe Martin Wyder, Judah Goldfeder, Alexey Yermakov, Yue Zhao, Stefano Riva, Jan P. Williams, David Zoro, Amy Sara Rude, Matteo Tomasetto, Joe Germany, Joseph Bakarji, Georg Maierhofer, Miles Cranmer, J. Nathan Kutz

Machine learning (ML) is transforming modeling and control in the physical,
engineering, and biological sciences. However, rapid development has outpaced
the creation of standardized, objective benchmarks - leading to weak baselines,
reporting bias, and inconsistent evaluations across methods. This undermines
reproducibility, misguides resource allocation, and obscures scientific
progress. To address this, we propose a Common Task Framework (CTF) for
scientific machine learning. The CTF features a curated set of datasets and
task-specific metrics spanning forecasting, state reconstruction, and
generalization under realistic constraints, including noise and limited data.
Inspired by the success of CTFs in fields like natural language processing and
computer vision, our framework provides a structured, rigorous foundation for
head-to-head evaluation of diverse algorithms. As a first step, we benchmark
methods on two canonical nonlinear systems: Kuramoto-Sivashinsky and Lorenz.
These results illustrate the utility of the CTF in revealing method strengths,
limitations, and suitability for specific classes of problems and diverse
objectives. Next, we are launching a competition around a global real world sea
surface temperature dataset with a true holdout dataset to foster community
engagement. Our long-term vision is to replace ad hoc comparisons with
standardized evaluations on hidden test sets that raise the bar for rigor and
reproducibility in scientific ML.

### 7. [Towards a Generalizable AI for Materials Discovery: Validation through Immersion Coolant Screening](http://arxiv.org/pdf/2510.23371v1)

Authors: Hyunseung Kim, Dae-Woong Jeong, Changyoung Park, Won-Ji Lee, Ha-Eun Lee, Ji-Hye Lee, Rodrigo Hormazabal, Sung Moon Ko, Sumin Lee, Soorin Yim, Chanhui Lee, Sehui Han, Sang-Ho Cha, Woohyung Lim

Artificial intelligence (AI) has emerged as a powerful accelerator of
materials discovery, yet most existing models remain problem-specific,
requiring additional data collection and retraining for each new property. Here
we introduce and validate GATE (Geometrically Aligned Transfer Encoder) -- a
generalizable AI framework that jointly learns 34 physicochemical properties
spanning thermal, electrical, mechanical, and optical domains. By aligning
these properties within a shared geometric space, GATE captures cross-property
correlations that reduce disjoint-property bias -- a key factor causing false
negatives in multi-criteria screening. To demonstrate its generalizability,
GATE -- without any problem-specific reconfiguration -- was directly applied to
the discovery of immersion cooling fluids for data centers, a stringent
real-world challenge defined by the Open Compute Project (OCP). Screening
billions of candidates, GATE identified 92,861 molecules as promising for
practical deployment. Four were experimentally or literarily validated, showing
strong agreement with wet-lab measurements and performance comparable to or
exceeding a commercial coolant. These results establish GATE as a ready-to-use,
generalizable AI platform readily applicable across diverse materials discovery
tasks.

### 8. [Tree-Cotree-Based IETI-DP for Eddy Current Problems in Time-Domain](http://arxiv.org/pdf/2510.23446v1)

Authors: Mario Mally, Rafael Vázquez, Sebastian Schöps

For low-frequency electromagnetic problems, where wave-propagation effects
can be neglected, eddy current formulations are commonly used as a
simplification of the full Maxwell's equations. In this setup, time-domain
simulations, needed to capture transient startup responses or nonlinear
behavior, are often computationally expensive. We propose a novel tearing and
interconnecting approach for eddy currents in time-domain and investigate its
scalability.

### Computational Geometry

### 1. [Online Hitting Set for Axis-Aligned Squares](http://arxiv.org/pdf/2510.23107v1)

Authors: Minati De, Satyam Singh, Csaba D. Tóth

We are given a set $P$ of $n$ points in the plane, and a sequence of
axis-aligned squares that arrive in an online fashion. The online hitting set
problem consists of maintaining, by adding new points if necessary, a set
$H\subseteq P$ that contains at least one point in each input square. We
present an $O(\log n)$-competitive deterministic algorithm for this problem.
The competitive ratio is the best possible, apart from constant factors. In
fact, this is the first $O(\log n)$-competitive algorithm for the online
hitting set problem that works for geometric objects of arbitrary sizes (i.e.,
arbitrary scaling factors) in the plane. We further generalize this result to
positive homothets of a polygon with $k\geq 3$ vertices in the plane and
provide an $O(k^2\log n)$-competitive algorithm.

### 2. [Expected Length of the Euclidean Minimum Spanning Tree and 1-norms of Chromatic Persistence Diagrams in the Plane](http://arxiv.org/pdf/2510.23373v1)

Authors: Ondřej Draganov, Herbert Edelsbrunner, Sophie Rosenmeier, Morteza Saghafian

Let $c$ be the constant such that the expected length of the Euclidean
minimum spanning tree of $n$ random points in the unit square is $c \sqrt{n}$
in the limit, when $n$ goes to infinity. We improve the prior best lower bound
of $0.6008 \leq c$ by Avram and Bertsimas to $0.6289 \leq c$. The proof is a
by-product of studying the persistent homology of randomly $2$-colored point
sets. Specifically, we consider the filtration induced by the inclusions of the
two mono-chromatic sublevel sets of the Euclidean distance function into the
bi-chromatic sublevel set of that function. Assigning colors randomly, and with
equal probability, we show that the expected $1$-norm of each chromatic
persistence diagram is a constant times $\sqrt{n}$ in the limit, and we
determine the constant in terms of $c$ and another constant, $c_L$, which
arises for a novel type of Euclidean minimum spanning tree of $2$-colored point
sets.

### 3. [Coresets for Clustering Under Stochastic Noise](http://arxiv.org/pdf/2510.23438v1)

Authors: Lingxiao Huang, Zhize Li, Nisheeth K. Vishnoi, Runkai Yang, Haoyu Zhao

We study the problem of constructing coresets for $(k, z)$-clustering when
the input dataset is corrupted by stochastic noise drawn from a known
distribution. In this setting, evaluating the quality of a coreset is
inherently challenging, as the true underlying dataset is unobserved. To
address this, we investigate coreset construction using surrogate error metrics
that are tractable and provably related to the true clustering cost. We analyze
a traditional metric from prior work and introduce a new error metric that more
closely aligns with the true cost. Although our metric is defined independently
of the noise distribution, it enables approximation guarantees that scale with
the noise level. We design a coreset construction algorithm based on this
metric and show that, under mild assumptions on the data and noise, enforcing
an $\varepsilon$-bound under our metric yields smaller coresets and tighter
guarantees on the true clustering cost than those obtained via classical
metrics. In particular, we prove that the coreset size can improve by a factor
of up to $\mathrm{poly}(k)$, where $n$ is the dataset size. Experiments on
real-world datasets support our theoretical findings and demonstrate the
practical advantages of our approach.

### 4. [T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](http://arxiv.org/pdf/2510.23484v1)

Authors: Julie Mordacq, David Loiseaux, Vicky Kalogeiton, Steve Oudot

Self-supervised learning (SSL) has emerged as a powerful paradigm for
learning representations without labeled data, often by enforcing invariance to
input transformations such as rotations or blurring. Recent studies have
highlighted two pivotal properties for effective representations: (i) avoiding
dimensional collapse-where the learned features occupy only a low-dimensional
subspace, and (ii) enhancing uniformity of the induced distribution. In this
work, we introduce T-REGS, a simple regularization framework for SSL based on
the length of the Minimum Spanning Tree (MST) over the learned representation.
We provide theoretical analysis demonstrating that T-REGS simultaneously
mitigates dimensional collapse and promotes distribution uniformity on
arbitrary compact Riemannian manifolds. Several experiments on synthetic data
and on classical SSL benchmarks validate the effectiveness of our approach at
enhancing representation quality.

### Computation and Language

### 1. [Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](http://arxiv.org/pdf/2510.22954v1)

Authors: Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, Yejin Choi

Language models (LMs) often struggle to generate diverse, human-like creative
content, raising concerns about the long-term homogenization of human thought
through repeated exposure to similar outputs. Yet scalable methods for
evaluating LM output diversity remain limited, especially beyond narrow tasks
such as random number or name generation, or beyond repeated sampling from a
single model. We introduce Infinity-Chat, a large-scale dataset of 26K diverse,
real-world, open-ended user queries that admit a wide range of plausible
answers with no single ground truth. We introduce the first comprehensive
taxonomy for characterizing the full spectrum of open-ended prompts posed to
LMs, comprising 6 top-level categories (e.g., brainstorm & ideation) that
further breaks down to 17 subcategories. Using Infinity-Chat, we present a
large-scale study of mode collapse in LMs, revealing a pronounced Artificial
Hivemind effect in open-ended generation of LMs, characterized by (1)
intra-model repetition, where a single model consistently generates similar
responses, and more so (2) inter-model homogeneity, where different models
produce strikingly similar outputs. Infinity-Chat also includes 31,250 human
annotations, across absolute ratings and pairwise preferences, with 25
independent human annotations per example. This enables studying collective and
individual-specific human preferences in response to open-ended queries. Our
findings show that LMs, reward models, and LM judges are less well calibrated
to human ratings on model generations that elicit differing idiosyncratic
annotator preferences, despite maintaining comparable overall quality. Overall,
INFINITY-CHAT presents the first large-scale resource for systematically
studying real-world open-ended queries to LMs, revealing critical insights to
guide future research for mitigating long-term AI safety risks posed by the
Artificial Hivemind.

### 2. [Knocking-Heads Attention](http://arxiv.org/pdf/2510.23052v1)

Authors: Zhanchao Zhou, Xiaodong Chen, Haoxing Chen, Zhenzhong Lan, Jianguo Li

Multi-head attention (MHA) has become the cornerstone of modern large
language models, enhancing representational capacity through parallel attention
heads. However, increasing the number of heads inherently weakens individual
head capacity, and existing attention mechanisms - whether standard MHA or its
variants like grouped-query attention (GQA) and grouped-tied attention (GTA) -
simply concatenate outputs from isolated heads without strong interaction. To
address this limitation, we propose knocking-heads attention (KHA), which
enables attention heads to "knock" on each other - facilitating cross-head
feature-level interactions before the scaled dot-product attention. This is
achieved by applying a shared, diagonally-initialized projection matrix across
all heads. The diagonal initialization preserves head-specific specialization
at the start of training while allowing the model to progressively learn
integrated cross-head representations. KHA adds only minimal parameters and
FLOPs and can be seamlessly integrated into MHA, GQA, GTA, and other attention
variants. We validate KHA by training a 6.1B parameter MoE model (1.01B
activated) on 1T high-quality tokens. Compared to baseline attention
mechanisms, KHA brings superior and more stable training dynamics, achieving
better performance across downstream tasks.

### 3. [A Survey on LLM Mid-training](http://arxiv.org/pdf/2510.23081v1)

Authors: Chengying Tu, Xuemiao Zhang, Rongxiang Weng, Rumei Li, Chen Zhang, Yang Bai, Hongfei Yan, Jingang Wang, Xunliang Cai

Recent advances in foundation models have highlighted the significant
benefits of multi-stage training, with a particular emphasis on the emergence
of mid-training as a vital stage that bridges pre-training and post-training.
Mid-training is distinguished by its use of intermediate data and computational
resources, systematically enhancing specified capabilities such as mathematics,
coding, reasoning, and long-context extension, while maintaining foundational
competencies. This survey provides a formal definition of mid-training for
large language models (LLMs) and investigates optimization frameworks that
encompass data curation, training strategies, and model architecture
optimization. We analyze mainstream model implementations in the context of
objective-driven interventions, illustrating how mid-training serves as a
distinct and critical stage in the progressive development of LLM capabilities.
By clarifying the unique contributions of mid-training, this survey offers a
comprehensive taxonomy and actionable insights, supporting future research and
innovation in the advancement of LLMs.

### 4. [MAP4TS: A Multi-Aspect Prompting Framework for Time-Series Forecasting with Large Language Models](http://arxiv.org/pdf/2510.23090v1)

Authors: Suchan Lee, Jihoon Choi, Sohyeon Lee, Minseok Song, Bong-Gyu Jang, Hwanjo Yu, Soyeon Caren Han

Recent advances have investigated the use of pretrained large language models
(LLMs) for time-series forecasting by aligning numerical inputs with LLM
embedding spaces. However, existing multimodal approaches often overlook the
distinct statistical properties and temporal dependencies that are fundamental
to time-series data. To bridge this gap, we propose MAP4TS, a novel
Multi-Aspect Prompting Framework that explicitly incorporates classical
time-series analysis into the prompt design. Our framework introduces four
specialized prompt components: a Global Domain Prompt that conveys
dataset-level context, a Local Domain Prompt that encodes recent trends and
series-specific behaviors, and a pair of Statistical and Temporal Prompts that
embed handcrafted insights derived from autocorrelation (ACF), partial
autocorrelation (PACF), and Fourier analysis. Multi-Aspect Prompts are combined
with raw time-series embeddings and passed through a cross-modality alignment
module to produce unified representations, which are then processed by an LLM
and projected for final forecasting. Extensive experiments across eight diverse
datasets show that MAP4TS consistently outperforms state-of-the-art LLM-based
methods. Our ablation studies further reveal that prompt-aware designs
significantly enhance performance stability and that GPT-2 backbones, when
paired with structured prompts, outperform larger models like LLaMA in
long-term forecasting tasks.

### 5. [Flexing in 73 Languages: A Single Small Model for Multilingual Inflection](http://arxiv.org/pdf/2510.23114v1)

Authors: Tomáš Sourada, Jana Straková

We present a compact, single-model approach to multilingual inflection, the
task of generating inflected word forms from base lemmas to express grammatical
categories. Our model, trained jointly on data from 73 languages, is
lightweight, robust to unseen words, and outperforms monolingual baselines in
most languages. This demonstrates the effectiveness of multilingual modeling
for inflection and highlights its practical benefits: simplifying deployment by
eliminating the need to manage and retrain dozens of separate monolingual
models. In addition to the standard SIGMORPHON shared task benchmarks, we
evaluate our monolingual and multilingual models on 73 Universal Dependencies
(UD) treebanks, extracting lemma-tag-form triples and their frequency counts.
To ensure realistic data splits, we introduce a novel frequency-weighted,
lemma-disjoint train-dev-test resampling procedure. Our work addresses the lack
of an open-source, general-purpose, multilingual morphological inflection
system capable of handling unseen words across a wide range of languages,
including Czech. All code is publicly released at:
https://github.com/tomsouri/multilingual-inflection.

### 6. [Corpus Frequencies in Morphological Inflection: Do They Matter?](http://arxiv.org/pdf/2510.23131v1)

Authors: Tomáš Sourada, Jana Straková

The traditional approach to morphological inflection (the task of modifying a
base word (lemma) to express grammatical categories) has been, for decades, to
consider lexical entries of lemma-tag-form triples uniformly, lacking any
information about their frequency distribution. However, in production
deployment, one might expect the user inputs to reflect a real-world
distribution of frequencies in natural texts. With future deployment in mind,
we explore the incorporation of corpus frequency information into the task of
morphological inflection along three key dimensions during system development:
(i) for train-dev-test split, we combine a lemma-disjoint approach, which
evaluates the model's generalization capabilities, with a frequency-weighted
strategy to better reflect the realistic distribution of items across different
frequency bands in training and test sets; (ii) for evaluation, we complement
the standard type accuracy (often referred to simply as accuracy), which treats
all items equally regardless of frequency, with token accuracy, which assigns
greater weight to frequent words and better approximates performance on running
text; (iii) for training data sampling, we introduce a method novel in the
context of inflection, frequency-aware training, which explicitly incorporates
word frequency into the sampling process. We show that frequency-aware training
outperforms uniform sampling in 26 out of 43 languages.

### 7. [ENTP: Enhancing Low-Quality SFT Data via Neural-Symbolic Text Purge-Mix](http://arxiv.org/pdf/2510.23160v1)

Authors: Zile Yang, Ling Li, Na Di, Jinlong Pang, Yao Zhou, Hao Cheng, Bo Han, Jiaheng Wei

Supervised Fine-Tuning (SFT) adapts pre-trained Large Language Models (LLMs)
to domain-specific instructions by training on a carefully curated subset of
high-quality instruction-response pairs, typically drawn from a larger dataset
that often contains many low-quality or noisy samples. However, existing
quality-first paradigms often overlook valuable signals in discarded
low-quality data and rely on imperfect quality filters. We introduce ENTP
(Enhancing low-quality SFT data via Neural-symbolic Text Purge-Mix), a
framework that revitalizes low-quality corpora through symbolic purification
and neural reconstruction. The symbolic module identifies and prunes noisy
samples based on statistical priors, while the neural component synthesizes
enriched instruction-response pairs by leveraging latent representations and
model knowledge. This neural-symbolic synergy enhances data informativeness and
diversity. Experiments show that ENTP-augmented datasets, constructed
exclusively from low-quality data, outperform 13 established data-selection
baselines across five instruction-following benchmarks, and even surpass
fine-tuning on the full original dataset (approximately 300K examples). Our
results highlight the untapped potential of low-quality data and underscore the
importance of intelligent purification and synthesis for efficient instruction
alignment.

### 8. [SI-Bench: Benchmarking Social Intelligence of Large Language Models in Human-to-Human Conversations](http://arxiv.org/pdf/2510.23182v1)

Authors: Shuai Huang, Wenxuan Zhao, Jun Gao

As large language models (LLMs) develop anthropomorphic abilities, they are
increasingly being deployed as autonomous agents to interact with humans.
However, evaluating their performance in realistic and complex social
interactions remains a significant challenge. Most previous research built
datasets through simulated agent-to-agent interactions, which fails to capture
the authentic linguistic styles and relational dynamics found in real human
conversations. To address this gap, we introduce SI-Bench, a novel benchmark
designed to evaluate aspects of social intelligence in LLMs. Grounded in broad
social science theories, SI-Bench contains 2,221 authentic multi-turn dialogues
collected from a social networking application. We further selected a subset of
312 dialogues for manual annotation across 8 major models. The experiments show
that SOTA models have surpassed the human expert in process reasoning under
complex social situations, yet they still fall behind humans in reply quality.
Moreover, introducing Chain-of-Thought (CoT) reasoning may degrade the
performance of LLMs in social dialogue tasks. All datasets are openly available
at https://github.com/SI-Bench/SI-Bench.git.

### 9. [Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages?](http://arxiv.org/pdf/2510.23252v1)

Authors: Tawsif Tashwar Dipto, Azmol Hossain, Rubayet Sabbir Faruque, Md. Rezuwan Hassan, Kanij Fatema, Tanmoy Shome, Ruwad Naswan, Md. Foriduzzaman Zihad, Mohaymen Ul Anam, Nazia Tasnim, Hasan Mahmud, Md Kamrul Hasan, Md. Mehedi Hasan Shawon, Farig Sadeque, Tahsin Reasat

Conventional research on speech recognition modeling relies on the canonical
form for most low-resource languages while automatic speech recognition (ASR)
for regional dialects is treated as a fine-tuning task. To investigate the
effects of dialectal variations on ASR we develop a 78-hour annotated Bengali
Speech-to-Text (STT) corpus named Ben-10. Investigation from linguistic and
data-driven perspectives shows that speech foundation models struggle heavily
in regional dialect ASR, both in zero-shot and fine-tuned settings. We observe
that all deep learning methods struggle to model speech data under dialectal
variations but dialect specific model training alleviates the issue. Our
dataset also serves as a out of-distribution (OOD) resource for ASR modeling
under constrained resources in ASR algorithms. The dataset and code developed
for this project are publicly available

### 10. [Mubeen AI: A Specialized Arabic Language Model for Heritage Preservation and User Intent Understanding](http://arxiv.org/pdf/2510.23271v1)

Authors: Mohammed Aljafari, Ismail Alturki, Ahmed Mori, Yehya Kadumi

Mubeen is a proprietary Arabic language model developed by MASARAT SA,
optimized for deep understanding of Arabic linguistics, Islamic studies, and
cultural heritage. Trained on an extensive collection of authentic Arabic
sources significantly expanded by digitizing historical manuscripts via a
proprietary Arabic OCR engine, the model incorporates seminal scholarly works
in linguistics, jurisprudence, hadith, and Quranic exegesis, alongside
thousands of academic theses and peer-reviewed research papers. Conditioned
through a deep linguistic engineering framework, Mubeen masters not just the
meaning but the eloquence of Arabic, enabling precise understanding across
classical texts, contemporary writing, and regional dialects with focus on
comprehending user intent and delivering accurate, contextually relevant
responses. Unlike other Arabic models relying on translated English data that
often fail in intent detection or retrieval-augmented generation (RAG), Mubeen
uses native Arabic sources to ensure cultural authenticity and accuracy. Its
core innovation is the Practical Closure Architecture, designed to solve the
"Utility Gap Crisis" where factually correct answers fail to resolve users'
core needs, forcing them into frustrating cycles of re-prompting. By
prioritizing clarity and decisive guidance, Mubeen transforms from an
information repository into a decisive guide, aligning with Saudi Vision 2030.
The model's architecture combines deep heritage specialization with
multi-disciplinary expert modules, enabling robust performance across both
cultural preservation and general knowledge domains.

### Cryptography and Security

### 1. [QuantumShield: Multilayer Fortification for Quantum Federated Learning](http://arxiv.org/pdf/2510.22945v1)

Authors: Dev Gurung, Shiva Raj Pokhrel

In this paper, we propose a groundbreaking quantum-secure federated learning
(QFL) framework designed to safeguard distributed learning systems against the
emerging threat of quantum-enabled adversaries. As classical cryptographic
methods become increasingly vulnerable to quantum attacks, our framework
establishes a resilient security architecture that remains robust even in the
presence of quantum-capable attackers. We integrate and rigorously evaluate
advanced quantum and post-quantum protocols including Quantum Key Distribution
(QKD), Quantum Teleportation, Key Encapsulation Mechanisms (KEM) and
Post-Quantum Cryptography (PQC) to fortify the QFL process against both
classical and quantum threats. These mechanisms are systematically analyzed and
implemented to demonstrate their seamless interoperability within a secure and
scalable QFL ecosystem. Through comprehensive theoretical modeling and
experimental validation, this work provides a detailed security and performance
assessment of the proposed framework. Our findings lay a strong foundation for
next-generation federated learning systems that are inherently secure in the
quantum era.

### 2. [Advancing Honeywords for Real-World Authentication Security](http://arxiv.org/pdf/2510.22971v1)

Authors: Sudiksha Das, Ashish Kundu

Introduced by Juels and Rivest in 2013, Honeywords, which are decoy passwords
stored alongside a real password, appear to be a proactive method to help
detect password credentials misuse. However, despite over a decade of research,
this technique has not been adopted by major authentication platforms. This
position paper argues that the core concept of Honeywords has potential but
requires more research on issues such as flatness, integration, and
reliability, in order to be a practical deployable solution. This paper
examines the current work on Honeyword generation, attacker modeling, and
honeychecker architecture, analyzing the subproblems that have been addressed
and ongoing issues that prevent this system from being more widely used. The
paper then suggests a deployable framework that combines the
attacker-resilient, context-aware decoy creation that Honeywords provide with
easy integration into existing systems. Honeywords will only move from an
academic idea to a practical security tool if technical advances are paired
with secure and straightforward architectures, along with adaptive response
handling and detailed configuration checks.

### 3. [KAPG: Adaptive Password Guessing via Knowledge-Augmented Generation](http://arxiv.org/pdf/2510.23036v1)

Authors: Xudong Yang, Jincheng Li, Kaiwen Xing, Zhenjia Xiao, Mingjian Duan, Weili Han, Hu Xiong

As the primary mechanism of digital authentication, user-created passwords
exhibit common patterns and regularities that can be learned from leaked
datasets. Password choices are profoundly shaped by external factors, including
social contexts, cultural trends, and popular vocabulary. Prevailing password
guessing models primarily emphasize patterns derived from leaked passwords,
while neglecting these external influences -- a limitation that hampers their
adaptability to emerging password trends and erodes their effectiveness over
time.
  To address these challenges, we propose KAPG, a knowledge-augmented password
guessing framework that adaptively integrates external lexical knowledge into
the guessing process. KAPG couples internal statistical knowledge learned from
leaked passwords with external information that reflects real-world trends. By
using password prefixes as anchors for knowledge lookup, it dynamically injects
relevant external cues during generation while preserving the structural
regularities of authentic passwords. Experiments on twelve leaked datasets show
that KnowGuess achieves average improvements of 36.5\% and 74.7\% over
state-of-the-art models in intra-site and cross-site scenarios, respectively.
Further analyses of password overlap and model efficiency highlight its
robustness and computational efficiency. To counter these attacks, we further
develop KAPSM, a trend-aware and site-specific password strength meter.
Experiments demonstrate that KAPSM significantly outperforms existing tools in
accuracy across diverse evaluation settings.

### 4. [Optimizing Optimism: Up to 6.5x Faster zkVM Validty Proofs via Sparse Derivation](http://arxiv.org/pdf/2510.23172v1)

Authors: Mohsen Ahmadvand, Pedro Souto

The Optimism derivation pipeline is engineered for correctness and liveness,
not for succinct validity proofs. A straightforward port to a zkVM imposes
significant overheads, making validity proofs significantly more costly than
necessary. We systematically identify inefficiencies in the current design,
analyze their impact on proving costs, and provide a soundness-preserving
redesign tailored to zk proving. Our redesign achieves up to 6.5x faster
derivation inside zkVMs (3.5x overall speedup) while maintaining identical
safety guarantees.

### 5. [Network Intrusion Detection: Evolution from Conventional Approaches to LLM Collaboration and Emerging Risks](http://arxiv.org/pdf/2510.23313v1)

Authors: Yaokai Feng, Kouichi Sakurai

This survey systematizes the evolution of network intrusion detection systems
(NIDS), from conventional methods such as signature-based and neural network
(NN)-based approaches to recent integrations with large language models (LLMs).
It clearly and concisely summarizes the current status, strengths, and
limitations of conventional techniques, and explores the practical benefits of
integrating LLMs into NIDS. Recent research on the application of LLMs to NIDS
in diverse environments is reviewed, including conventional network
infrastructures, autonomous vehicle environments and IoT environments.
  From this survey, readers will learn that: 1) the earliest methods,
signature-based IDSs, continue to make significant contributions to modern
systems, despite their well-known weaknesses; 2) NN-based detection, although
considered promising and under development for more than two decades, and
despite numerous related approaches, still faces significant challenges in
practical deployment; 3) LLMs are useful for NIDS in many cases, and a number
of related approaches have been proposed; however, they still face significant
challenges in practical applications. Moreover, they can even be exploited as
offensive tools, such as for generating malware, crafting phishing messages, or
launching cyberattacks. Recently, several studies have been proposed to address
these challenges, which are also reviewed in this survey; and 4) strategies for
constructing domain-specific LLMs have been proposed and are outlined in this
survey, as it is nearly impossible to train a NIDS-specific LLM from scratch.

### 6. [Authentication Against Insecure Bootstrapping for 5G Networks: Feasibility, Resiliency, and Transitional Solutions in Post-Quantum Era](http://arxiv.org/pdf/2510.23457v1)

Authors: Saleh Darzi, Mirza Masfiqur Rahman, Imtiaz Karim, Rouzbeh Behnia, Attila A Yavuz, Elisa Bertino

The 5G protocol lacks a robust base station authentication mechanism during
the initial bootstrapping phase, leaving it susceptible to threats such as fake
base station attacks. Conventional solutions, including digital signatures
based on Public Key Infrastructures (PKIs) and identity-based signatures, are
inadequate against quantum-capable adversaries. While integrating NIST's
Post-Quantum Cryptography (PQC) standards is a leading approach for quantum
resistance, their suitability for 5G base station authentication remains
unexplored. Moreover, current solutions are predominantly centralized and lack
security features such as distributed authentication. This work presents, to
our knowledge, the first comprehensive network-level performance
characterization of integrating NIST-PQC standards and conventional digital
signatures (including threshold and identity-based schemes) into 5G base
station authentication. Our findings reveal significant feasibility concerns,
with direct PQC adoption hindered by protocol constraints and large signature
sizes. We also highlight the performance limitations of conventional methods
due to the overhead of certificate chains. To mitigate these challenges, we
propose BORG, a transitional authentication solution based on a Hierarchical
Identity-Based Threshold Signature scheme with a Fail-Stop property. BORG
offers post-mortem post-quantum forgery detection and distributed trust via
threshold and compact signatures, well-suited for 5G's stringent requirements.
Our performance analysis underscores an important warning on the infeasibility
of direct PQC integration and positions BORG as an effective transitional
solution toward future quantum-resilient 5G authentication.

### 7. [Towards a Functionally Complete and Parameterizable TFHE Processor](http://arxiv.org/pdf/2510.23483v1)

Authors: Valentin Reyes Häusler, Gabriel Ott, Aruna Jayasena, Andreas Peter

Fully homomorphic encryption allows the evaluation of arbitrary functions on
encrypted data. It can be leveraged to secure outsourced and multiparty
computation. TFHE is a fast torus-based fully homomorphic encryption scheme
that allows both linear operations, as well as the evaluation of arbitrary
non-linear functions. It currently provides the fastest bootstrapping operation
performance of any other FHE scheme. Despite its fast performance, TFHE suffers
from a considerably higher computational overhead for the evaluation of
homomorphic circuits. Computations in the encrypted domain are orders of
magnitude slower than their unencrypted equivalents. This bottleneck hinders
the widespread adoption of (T)FHE for the protection of sensitive data. While
state-of-the-art implementations focused on accelerating and outsourcing single
operations, their scalability and practicality are constrained by high memory
bandwidth costs. In order to overcome this, we propose an FPGA-based hardware
accelerator for the evaluation of homomorphic circuits. Specifically, we design
a functionally complete TFHE processor for FPGA hardware capable of processing
instructions on the data completely on the FPGA. In order to achieve a higher
throughput from our TFHE processor, we implement an improved programmable
bootstrapping module which outperforms the current state-of-the-art by 240\% to
480\% more bootstrappings per second. Our efficient, compact, and scalable
design lays the foundation for implementing complete FPGA-based TFHE processor
architectures.

### 8. [Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies](http://arxiv.org/pdf/2510.22944v1)

Authors: Bin Wang, YiLu Zhong, MiDi Wan, WenJie Yu, YuanBing Ouyang, Yenan Huang, Hui Li

Large language models (LLMs) have become indispensable for automated code
generation, yet the quality and security of their outputs remain a critical
concern. Existing studies predominantly concentrate on adversarial attacks or
inherent flaws within the models. However, a more prevalent yet underexplored
issue concerns how the quality of a benign but poorly formulated prompt affects
the security of the generated code. To investigate this, we first propose an
evaluation framework for prompt quality encompassing three key dimensions: goal
clarity, information completeness, and logical consistency. Based on this
framework, we construct and publicly release CWE-BENCH-PYTHON, a large-scale
benchmark dataset containing tasks with prompts categorized into four distinct
levels of normativity (L0-L3). Extensive experiments on multiple
state-of-the-art LLMs reveal a clear correlation: as prompt normativity
decreases, the likelihood of generating insecure code consistently and markedly
increases. Furthermore, we demonstrate that advanced prompting techniques, such
as Chain-of-Thought and Self-Correction, effectively mitigate the security
risks introduced by low-quality prompts, substantially improving code safety.
Our findings highlight that enhancing the quality of user prompts constitutes a
critical and effective strategy for strengthening the security of AI-generated
code.

### 9. [CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents](http://arxiv.org/pdf/2510.22963v1)

Authors: Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She

LLM-powered agents often use prompt compression to reduce inference costs,
but this introduces a new security risk. Compression modules, which are
optimized for efficiency rather than safety, can be manipulated by adversarial
inputs, causing semantic drift and altering LLM behavior. This work identifies
prompt compression as a novel attack surface and presents CompressionAttack,
the first framework to exploit it. CompressionAttack includes two strategies:
HardCom, which uses discrete adversarial edits for hard compression, and
SoftCom, which performs latent-space perturbations for soft compression.
Experiments on multiple LLMs show up to 80% attack success and 98% preference
flips, while remaining highly stealthy and transferable. Case studies in VSCode
Cline and Ollama confirm real-world impact, and current defenses prove
ineffective, highlighting the need for stronger protections.

### 10. [A Multi-Store Privacy Measurement of Virtual Reality App Ecosystem](http://arxiv.org/pdf/2510.23024v1)

Authors: Chuan Yan, Zeng Li, Kunlin Cai, Liuhuo Wan, Ruomai Ren, Yiran Shen, Guangdong Bai

Virtual Reality (VR) has gained increasing traction among various domains in
recent years, with major companies such as Meta, Pico, and Microsoft launching
their application stores to support third-party developers in releasing their
applications (or simply apps). These apps offer rich functionality but
inherently collect privacy-sensitive data, such as user biometrics, behaviors,
and the surrounding environment. Nevertheless, there is still a lack of
domain-specific regulations to govern the data handling of VR apps, resulting
in significant variations in their privacy practices among app stores.
  In this work, we present the first comprehensive multi-store study of privacy
practices in the current VR app ecosystem, covering a large-scale dataset
involving 6,565 apps collected from five major app stores. We assess both
declarative and behavioral privacy practices of VR apps, using a multi-faceted
approach based on natural language processing, reverse engineering, and static
analysis. Our assessment reveals significant privacy compliance issues across
all stores, underscoring the premature status of privacy protection in this
rapidly growing ecosystem. For instance, one third of apps fail to declare
their use of sensitive data, and 21.5\% of apps neglect to provide valid
privacy policies. Our work sheds light on the status quo of privacy protection
within the VR app ecosystem for the first time. Our findings should raise an
alert to VR app developers and users, and encourage store operators to
implement stringent regulations on privacy compliance among VR apps.

### Computer Vision and Pattern Recognition

### 1. [Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture](http://arxiv.org/pdf/2510.22916v1)

Authors: Qiyu Liao, Dadong Wang, Rebecca Haling, Jiajun Liu, Xun Li, Martyna Plomecka, Andrew Robson, Matthew Pringle, Rhys Pirie, Megan Walker, Joshua Whelan

Accurate estimation of pasture biomass is important for decision-making in
livestock production systems. Estimates of pasture biomass can be used to
manage stocking rates to maximise pasture utilisation, while minimising the
risk of overgrazing and promoting overall system health. We present a
comprehensive dataset of 1,162 annotated top-view images of pastures collected
across 19 locations in Australia. The images were taken across multiple seasons
and include a range of temperate pasture species. Each image captures a 70cm *
30cm quadrat and is paired with on-ground measurements including biomass sorted
by component (green, dead, and legume fraction), vegetation height, and
Normalized Difference Vegetation Index (NDVI) from Active Optical Sensors
(AOS). The multidimensional nature of the data, which combines visual,
spectral, and structural information, opens up new possibilities for advancing
the use of precision grazing management. The dataset is released and hosted in
a Kaggle competition that challenges the international Machine Learning
community with the task of pasture biomass estimation. The dataset is available
on the official Kaggle webpage:
https://www.kaggle.com/competitions/csiro-biomass

### 2. [Positional Preservation Embedding for Multimodal Large Language Models](http://arxiv.org/pdf/2510.22936v1)

Authors: Mouxiao Huang, Borui Jiang, Dehua Zheng, Hailin Hu, Kai Han, Xinghao Chen

Multimodal large language models (MLLMs) have achieved strong performance on
vision-language tasks, yet often suffer from inefficiencies due to redundant
visual tokens. Existing token merging methods reduce sequence length but
frequently disrupt spatial layouts and temporal continuity by disregarding
positional relationships. In this work, we propose a novel encoding operator
dubbed as \textbf{P}ositional \textbf{P}reservation \textbf{E}mbedding
(\textbf{PPE}), which has the main hallmark of preservation of spatiotemporal
structure during visual token compression. PPE explicitly introduces the
disentangled encoding of 3D positions in the token dimension, enabling each
compressed token to encapsulate different positions from multiple original
tokens. Furthermore, we show that PPE can effectively support cascade
clustering -- a progressive token compression strategy that leads to better
performance retention. PPE is a parameter-free and generic operator that can be
seamlessly integrated into existing token merging methods without any
adjustments. Applied to state-of-the-art token merging framework, PPE achieves
consistent improvements of $2\%\sim5\%$ across multiple vision-language
benchmarks, including MMBench (general vision understanding), TextVQA (layout
understanding) and VideoMME (temporal understanding). These results demonstrate
that preserving positional cues is critical for efficient and effective MLLM
reasoning.

### 3. [Switchable Token-Specific Codebook Quantization For Face Image Compression](http://arxiv.org/pdf/2510.22943v1)

Authors: Yongbo Wang, Haonan Wang, Guodong Mu, Ruixin Zhang, Jiaqi Chen, Jingyun Zhang, Jun Wang, Yuan Xie, Zhizhong Zhang, Shouhong Ding

With the ever-increasing volume of visual data, the efficient and lossless
transmission, along with its subsequent interpretation and understanding, has
become a critical bottleneck in modern information systems. The emerged
codebook-based solution utilize a globally shared codebook to quantize and
dequantize each token, controlling the bpp by adjusting the number of tokens or
the codebook size. However, for facial images, which are rich in attributes,
such global codebook strategies overlook both the category-specific
correlations within images and the semantic differences among tokens, resulting
in suboptimal performance, especially at low bpp. Motivated by these
observations, we propose a Switchable Token-Specific Codebook Quantization for
face image compression, which learns distinct codebook groups for different
image categories and assigns an independent codebook to each token. By
recording the codebook group to which each token belongs with a small number of
bits, our method can reduce the loss incurred when decreasing the size of each
codebook group. This enables a larger total number of codebooks under a lower
overall bpp, thereby enhancing the expressive capability and improving
reconstruction performance. Owing to its generalizable design, our method can
be integrated into any existing codebook-based representation learning approach
and has demonstrated its effectiveness on face recognition datasets, achieving
an average accuracy of 93.51% for reconstructed images at 0.05 bpp.

### 4. [LightBagel: A Light-weighted, Double Fusion Framework for Unified Multimodal Understanding and Generation](http://arxiv.org/pdf/2510.22946v1)

Authors: Zeyu Wang, Zilong Chen, Chenhui Gou, Feng Li, Chaorui Deng, Deyao Zhu, Kunchang Li, Weihao Yu, Haoqin Tu, Haoqi Fan, Cihang Xie

Unified multimodal models have recently shown remarkable gains in both
capability and versatility, yet most leading systems are still trained from
scratch and require substantial computational resources. In this paper, we show
that competitive performance can be obtained far more efficiently by
strategically fusing publicly available models specialized for either
generation or understanding. Our key design is to retain the original blocks
while additionally interleaving multimodal self-attention blocks throughout the
networks. This double fusion mechanism (1) effectively enables rich multi-modal
fusion while largely preserving the original strengths of the base models, and
(2) catalyzes synergistic fusion of high-level semantic representations from
the understanding encoder with low-level spatial signals from the generation
encoder. By training with only ~ 35B tokens, this approach achieves strong
results across multiple benchmarks: 0.91 on GenEval for compositional
text-to-image generation, 82.16 on DPG-Bench for complex text-to-image
generation, 6.06 on GEditBench, and 3.77 on ImgEdit-Bench for image editing. By
fully releasing the entire suite of code, model weights, and datasets, we hope
to support future research on unified multimodal modeling.

### 5. [Survey of Multimodal Geospatial Foundation Models: Techniques, Applications, and Challenges](http://arxiv.org/pdf/2510.22964v1)

Authors: Liling Yang, Ning Chen, Jun Yue, Yidan Liu, Jiayi Ma, Pedram Ghamisi, Antonio Plaza, Leyuan Fang

Foundation models have transformed natural language processing and computer
vision, and their impact is now reshaping remote sensing image analysis. With
powerful generalization and transfer learning capabilities, they align
naturally with the multimodal, multi-resolution, and multi-temporal
characteristics of remote sensing data. To address unique challenges in the
field, multimodal geospatial foundation models (GFMs) have emerged as a
dedicated research frontier. This survey delivers a comprehensive review of
multimodal GFMs from a modality-driven perspective, covering five core visual
and vision-language modalities. We examine how differences in imaging physics
and data representation shape interaction design, and we analyze key techniques
for alignment, integration, and knowledge transfer to tackle modality
heterogeneity, distribution shifts, and semantic gaps. Advances in training
paradigms, architectures, and task-specific adaptation strategies are
systematically assessed alongside a wealth of emerging benchmarks.
Representative multimodal visual and vision-language GFMs are evaluated across
ten downstream tasks, with insights into their architectures, performance, and
application scenarios. Real-world case studies, spanning land cover mapping,
agricultural monitoring, disaster response, climate studies, and geospatial
intelligence, demonstrate the practical potential of GFMs. Finally, we outline
pressing challenges in domain generalization, interpretability, efficiency, and
privacy, and chart promising avenues for future research.

### 6. [VALA: Learning Latent Anchors for Training-Free and Temporally Consistent](http://arxiv.org/pdf/2510.22970v1)

Authors: Zhangkai Wu, Xuhui Fan, Zhongyuan Xie, Kaize Shi, Longbing Cao

Recent advances in training-free video editing have enabled lightweight and
precise cross-frame generation by leveraging pre-trained text-to-image
diffusion models. However, existing methods often rely on heuristic frame
selection to maintain temporal consistency during DDIM inversion, which
introduces manual bias and reduces the scalability of end-to-end inference. In
this paper, we propose~\textbf{VALA} (\textbf{V}ariational \textbf{A}lignment
for \textbf{L}atent \textbf{A}nchors), a variational alignment module that
adaptively selects key frames and compresses their latent features into
semantic anchors for consistent video editing. To learn meaningful assignments,
VALA propose a variational framework with a contrastive learning objective.
Therefore, it can transform cross-frame latent representations into compressed
latent anchors that preserve both content and temporal coherence. Our method
can be fully integrated into training-free text-to-image based video editing
models. Extensive experiments on real-world video editing benchmarks show that
VALA achieves state-of-the-art performance in inversion fidelity, editing
quality, and temporal consistency, while offering improved efficiency over
prior methods.

### 7. [Scaling Up Occupancy-centric Driving Scene Generation: Dataset and Method](http://arxiv.org/pdf/2510.22973v1)

Authors: Bohan Li, Xin Jin, Hu Zhu, Hongsi Liu, Ruikai Li, Jiazhe Guo, Kaiwen Cai, Chao Ma, Yueming Jin, Hao Zhao, Xiaokang Yang, Wenjun Zeng

Driving scene generation is a critical domain for autonomous driving,
enabling downstream applications, including perception and planning evaluation.
Occupancy-centric methods have recently achieved state-of-the-art results by
offering consistent conditioning across frames and modalities; however, their
performance heavily depends on annotated occupancy data, which still remains
scarce. To overcome this limitation, we curate Nuplan-Occ, the largest semantic
occupancy dataset to date, constructed from the widely used Nuplan benchmark.
Its scale and diversity facilitate not only large-scale generative modeling but
also autonomous driving downstream applications. Based on this dataset, we
develop a unified framework that jointly synthesizes high-quality semantic
occupancy, multi-view videos, and LiDAR point clouds. Our approach incorporates
a spatio-temporal disentangled architecture to support high-fidelity spatial
expansion and temporal forecasting of 4D dynamic occupancy. To bridge modal
gaps, we further propose two novel techniques: a Gaussian splatting-based
sparse point map rendering strategy that enhances multi-view video generation,
and a sensor-aware embedding strategy that explicitly models LiDAR sensor
properties for realistic multi-LiDAR simulation. Extensive experiments
demonstrate that our method achieves superior generation fidelity and
scalability compared to existing approaches, and validates its practical value
in downstream tasks. Repo:
https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2

### 8. [SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency](http://arxiv.org/pdf/2510.22994v1)

Authors: Quanjian Song, Donghao Zhou, Jingyu Lin, Fei Shen, Jiaze Wang, Xiaowei Hu, Cunjian Chen, Pheng-Ann Heng

Recent text-to-image models have revolutionized image generation, but they
still struggle with maintaining concept consistency across generated images.
While existing works focus on character consistency, they often overlook the
crucial role of scenes in storytelling, which restricts their creativity in
practice. This paper introduces scene-oriented story generation, addressing two
key challenges: (i) scene planning, where current methods fail to ensure
scene-level narrative coherence by relying solely on text descriptions, and
(ii) scene consistency, which remains largely unexplored in terms of
maintaining scene consistency across multiple stories. We propose
SceneDecorator, a training-free framework that employs VLM-Guided Scene
Planning to ensure narrative coherence across different scenes in a
``global-to-local'' manner, and Long-Term Scene-Sharing Attention to maintain
long-term scene consistency and subject diversity across generated stories.
Extensive experiments demonstrate the superior performance of SceneDecorator,
highlighting its potential to unleash creativity in the fields of arts, films,
and games.

### 9. [LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](http://arxiv.org/pdf/2510.22995v1)

Authors: Md Mostafijur Rahman, Radu Marculescu

U-shaped networks output logits at multiple spatial scales, each capturing a
different blend of coarse context and fine detail. Yet, training still treats
these logits in isolation - either supervising only the final,
highest-resolution logits or applying deep supervision with identical loss
weights at every scale - without exploring mixed-scale combinations.
Consequently, the decoder output misses the complementary cues that arise only
when coarse and fine predictions are fused. To address this issue, we introduce
LoMix (Logits Mixing), a NAS-inspired, differentiable plug-and-play module that
generates new mixed-scale outputs and learns how exactly each of them should
guide the training process. More precisely, LoMix mixes the multi-scale decoder
logits with four lightweight fusion operators: addition, multiplication,
concatenation, and attention-based weighted fusion, yielding a rich set of
synthetic mutant maps. Every original or mutant map is given a softplus loss
weight that is co-optimized with network parameters, mimicking a one-step
architecture search that automatically discovers the most useful scales,
mixtures, and operators. Plugging LoMix into recent U-shaped architectures
(i.e., PVT-V2-B2 backbone with EMCAD decoder) on Synapse 8-organ dataset
improves DICE by +4.2% over single-output supervision, +2.2% over deep
supervision, and +1.5% over equally weighted additive fusion, all with zero
inference overhead. When training data are scarce (e.g., one or two labeled
scans), the advantage grows to +9.23%, underscoring LoMix's data efficiency.
Across four benchmarks and diverse U-shaped networks, LoMiX improves DICE by up
to +13.5% over single-output supervision, confirming that learnable weighted
mixed-scale fusion generalizes broadly while remaining data efficient, fully
interpretable, and overhead-free at inference. Our code is available at
https://github.com/SLDGroup/LoMix.

### 10. [CoMo: Compositional Motion Customization for Text-to-Video Generation](http://arxiv.org/pdf/2510.23007v1)

Authors: Youcan Xu, Zhen Wang, Jiaxin Shi, Kexin Li, Feifei Shao, Jun Xiao, Yi Yang, Jun Yu, Long Chen

While recent text-to-video models excel at generating diverse scenes, they
struggle with precise motion control, particularly for complex, multi-subject
motions. Although methods for single-motion customization have been developed
to address this gap, they fail in compositional scenarios due to two primary
challenges: motion-appearance entanglement and ineffective multi-motion
blending. This paper introduces CoMo, a novel framework for
$\textbf{compositional motion customization}$ in text-to-video generation,
enabling the synthesis of multiple, distinct motions within a single video.
CoMo addresses these issues through a two-phase approach. First, in the
single-motion learning phase, a static-dynamic decoupled tuning paradigm
disentangles motion from appearance to learn a motion-specific module. Second,
in the multi-motion composition phase, a plug-and-play divide-and-merge
strategy composes these learned motions without additional training by
spatially isolating their influence during the denoising process. To facilitate
research in this new domain, we also introduce a new benchmark and a novel
evaluation metric designed to assess multi-motion fidelity and blending.
Extensive experiments demonstrate that CoMo achieves state-of-the-art
performance, significantly advancing the capabilities of controllable video
generation. Our project page is at https://como6.github.io/.

### Computers and Society

### 1. [How Can AI Augment Access to Justice? Public Defenders' Perspectives on AI Adoption](http://arxiv.org/pdf/2510.22933v1)

Authors: Inyoung Cheong, Patty Liu, Dominik Stammbach, Peter Henderson

Public defenders are asked to do more with less: representing clients
deserving of adequate counsel while facing overwhelming caseloads and scarce
resources. While artificial intelligence (AI) and large language models (LLMs)
are promoted as tools to alleviate this burden, such proposals are detached
from the lived realities of public defenders. This study addresses that gap
through semi-structured interviews with fourteen practitioners across the
United States to examine their experiences with AI, anticipated applications,
and ethical concerns. We find that AI adoption is constrained by costs,
restrictive office norms, confidentiality risks, and unsatisfactory tool
quality. To clarify where AI can and cannot contribute, we propose a task-level
map of public defense. Public defenders view AI as most useful for evidence
investigation to analyze overwhelming amounts of digital records, with narrower
roles in legal research & writing, and client communication. Courtroom
representation and defense strategy are considered least compatible with AI
assistance, as they depend on contextual judgment and trust. Public defenders
emphasize safeguards for responsible use, including mandatory human
verification, limits on overreliance, and the preservation of relational aspect
of lawyering. Building on these findings, we outline a research agenda that
promotes equitable access to justice by prioritizing open-source models,
domain-specific datasets and evaluation, and participatory design that
incorporates defenders' perspectives into system development.

### 2. [From Perceived Effectiveness to Measured Impact: Identity-Aware Evaluation of Automated Counter-Stereotypes](http://arxiv.org/pdf/2510.23523v1)

Authors: Svetlana Kiritchenko, Anna Kerkhof, Isar Nejadgholi, Kathleen C. Fraser

We investigate the effect of automatically generated counter-stereotypes on
gender bias held by users of various demographics on social media. Building on
recent NLP advancements and social psychology literature, we evaluate two
counter-stereotype strategies -- counter-facts and broadening universals (i.e.,
stating that anyone can have a trait regardless of group membership) -- which
have been identified as the most potentially effective in previous studies. We
assess the real-world impact of these strategies on mitigating gender bias
across user demographics (gender and age), through the Implicit Association
Test and the self-reported measures of explicit bias and perceived utility. Our
findings reveal that actual effectiveness does not align with perceived
effectiveness, and the former is a nuanced and sometimes divergent phenomenon
across demographic groups. While overall bias reduction was limited, certain
groups (e.g., older, male participants) exhibited measurable improvements in
implicit bias in response to some interventions. Conversely, younger
participants, especially women, showed increasing bias in response to the same
interventions. These results highlight the complex and identity-sensitive
nature of stereotype mitigation and call for dynamic and context-aware
evaluation and mitigation strategies.

### 3. [LangLingual: A Personalised, Exercise-oriented English Language Learning Tool Leveraging Large Language Models](http://arxiv.org/pdf/2510.23011v1)

Authors: Sammriddh Gupta, Sonit Singh, Aditya Joshi, Mira Kim

Language educators strive to create a rich experience for learners, while
they may be restricted in the extend of feedback and practice they can provide.
We present the design and development of LangLingual, a conversational agent
built using the LangChain framework and powered by Large Language Models. The
system is specifically designed to provide real-time, grammar-focused feedback,
generate context-aware language exercises and track learner proficiency over
time. The paper discusses the architecture, implementation and evaluation of
LangLingual in detail. The results indicate strong usability, positive learning
outcomes and encouraging learner engagement.

### 4. [Modeling Political Discourse with Sentence-BERT and BERTopic](http://arxiv.org/pdf/2510.22904v1)

Authors: Margarida Mendonca, Alvaro Figueira

Social media has reshaped political discourse, offering politicians a
platform for direct engagement while reinforcing polarization and ideological
divides. This study introduces a novel topic evolution framework that
integrates BERTopic-based topic modeling with Moral Foundations Theory (MFT) to
analyze the longevity and moral dimensions of political topics in Twitter
activity during the 117th U.S. Congress. We propose a methodology for tracking
dynamic topic shifts over time and measuring their association with moral
values and quantifying topic persistence. Our findings reveal that while
overarching themes remain stable, granular topics tend to dissolve rapidly,
limiting their long-term influence. Moreover, moral foundations play a critical
role in topic longevity, with Care and Loyalty dominating durable topics, while
partisan differences manifest in distinct moral framing strategies. This work
contributes to the field of social network analysis and computational political
discourse by offering a scalable, interpretable approach to understanding
moral-driven topic evolution on social media.

### Databases

### 1. [A Survey of Data Agents: Emerging Paradigm or Overstated Hype?](http://arxiv.org/pdf/2510.23587v1)

Authors: Yizhang Zhu, Liangwei Wang, Chenyu Yang, Xiaotian Lin, Boyan Li, Wei Zhou, Xinyu Liu, Zhangyang Peng, Tianqi Luo, Yu Li, Chengliang Chai, Chong Chen, Shimin Di, Ju Fan, Ji Sun, Nan Tang, Fugee Tsung, Jiannan Wang, Chenglin Wu, Yanwei Xu, Shaolei Zhang, Yong Zhang, Xuanhe Zhou, Guoliang Li, Yuyu Luo

The rapid advancement of large language models (LLMs) has spurred the
emergence of data agents--autonomous systems designed to orchestrate Data + AI
ecosystems for tackling complex data-related tasks. However, the term "data
agent" currently suffers from terminological ambiguity and inconsistent
adoption, conflating simple query responders with sophisticated autonomous
architectures. This terminological ambiguity fosters mismatched user
expectations, accountability challenges, and barriers to industry growth.
Inspired by the SAE J3016 standard for driving automation, this survey
introduces the first systematic hierarchical taxonomy for data agents,
comprising six levels that delineate and trace progressive shifts in autonomy,
from manual operations (L0) to a vision of generative, fully autonomous data
agents (L5), thereby clarifying capability boundaries and responsibility
allocation. Through this lens, we offer a structured review of existing
research arranged by increasing autonomy, encompassing specialized data agents
for data management, preparation, and analysis, alongside emerging efforts
toward versatile, comprehensive systems with enhanced autonomy. We further
analyze critical evolutionary leaps and technical gaps for advancing data
agents, especially the ongoing L2-to-L3 transition, where data agents evolve
from procedural execution to autonomous orchestration. Finally, we conclude
with a forward-looking roadmap, envisioning the advent of proactive, generative
data agents.

### Distributed, Parallel, and Cluster Computing

### 1. [Sentinel: Dynamic Knowledge Distillation for Personalized Federated Intrusion Detection in Heterogeneous IoT Networks](http://arxiv.org/pdf/2510.23019v1)

Authors: Gurpreet Singh, Keshav Sood, P. Rajalakshmi, Yong Xiang

Federated learning (FL) offers a privacy-preserving paradigm for machine
learning, but its application in intrusion detection systems (IDS) within IoT
networks is challenged by severe class imbalance, non-IID data, and high
communication overhead.These challenges severely degrade the performance of
conventional FL methods in real-world network traffic classification. To
overcome these limitations, we propose Sentinel, a personalized federated IDS
(pFed-IDS) framework that incorporates a dual-model architecture on each
client, consisting of a personalized teacher and a lightweight shared student
model. This design effectively balances deep local adaptation with efficient
global model consensus while preserving client privacy by transmitting only the
compact student model, thus reducing communication costs. Sentinel integrates
three key mechanisms to ensure robust performance: bidirectional knowledge
distillation with adaptive temperature scaling, multi-faceted feature
alignment, and class-balanced loss functions. Furthermore, the server employs
normalized gradient aggregation with equal client weighting to enhance fairness
and mitigate client drift. Extensive experiments on the IoTID20 and 5GNIDD
benchmark datasets demonstrate that Sentinel significantly outperforms
state-of-the-art federated methods, establishing a new performance benchmark,
especially under extreme data heterogeneity, while maintaining communication
efficiency.

### 2. [AirFed: Federated Graph-Enhanced Multi-Agent Reinforcement Learning for Multi-UAV Cooperative Mobile Edge Computing](http://arxiv.org/pdf/2510.23053v1)

Authors: Zhiyu Wang, Suman Raj, Rajkumar Buyya

Multiple Unmanned Aerial Vehicles (UAVs) cooperative Mobile Edge Computing
(MEC) systems face critical challenges in coordinating trajectory planning,
task offloading, and resource allocation while ensuring Quality of Service
(QoS) under dynamic and uncertain environments. Existing approaches suffer from
limited scalability, slow convergence, and inefficient knowledge sharing among
UAVs, particularly when handling large-scale IoT device deployments with
stringent deadline constraints. This paper proposes AirFed, a novel federated
graph-enhanced multi-agent reinforcement learning framework that addresses
these challenges through three key innovations. First, we design dual-layer
dynamic Graph Attention Networks (GATs) that explicitly model spatial-temporal
dependencies among UAVs and IoT devices, capturing both service relationships
and collaborative interactions within the network topology. Second, we develop
a dual-Actor single-Critic architecture that jointly optimizes continuous
trajectory control and discrete task offloading decisions. Third, we propose a
reputation-based decentralized federated learning mechanism with
gradient-sensitive adaptive quantization, enabling efficient and robust
knowledge sharing across heterogeneous UAVs. Extensive experiments demonstrate
that AirFed achieves 42.9% reduction in weighted cost compared to
state-of-the-art baselines, attains over 99% deadline satisfaction and 94.2%
IoT device coverage rate, and reduces communication overhead by 54.5%.
Scalability analysis confirms robust performance across varying UAV numbers,
IoT device densities, and system scales, validating AirFed's practical
applicability for large-scale UAV-MEC deployments.

### 3. [Rethinking Inference Placement for Deep Learning across Edge and Cloud Platforms: A Multi-Objective Optimization Perspective and Future Directions](http://arxiv.org/pdf/2510.22909v1)

Authors: Zongshun Zhang, Ibrahim Matta

Edge intelligent applications like VR/AR and language model based chatbots
have become widespread with the rapid expansion of IoT and mobile devices.
However, constrained edge devices often cannot serve the increasingly large and
complex deep learning (DL) models. To mitigate these challenges, researchers
have proposed optimizing and offloading partitions of DL models among user
devices, edge servers, and the cloud. In this setting, users can take advantage
of different services to support their intelligent applications. For example,
edge resources offer low response latency. In contrast, cloud platforms provide
low monetary cost computation resources for computation-intensive workloads.
However, communication between DL model partitions can introduce transmission
bottlenecks and pose risks of data leakage. Recent research aims to balance
accuracy, computation delay, transmission delay, and privacy concerns. They
address these issues with model compression, model distillation, transmission
compression, and model architecture adaptations, including internal
classifiers. This survey contextualizes the state-of-the-art model offloading
methods and model adaptation techniques by studying their implication to a
multi-objective optimization comprising inference latency, data privacy, and
resource monetary cost.

### 4. [CodeAD: Synthesize Code of Rules for Log-based Anomaly Detection with LLMs](http://arxiv.org/pdf/2510.22986v1)

Authors: Junjie Huang, Minghua He, Jinyang Liu, Yintong Huo, Domenico Bianculli, Michael R. Lyu

Log-based anomaly detection (LogAD) is critical for maintaining the
reliability and availability of large-scale online service systems. While
machine learning, deep learning, and large language models (LLMs)-based methods
have advanced the LogAD, they often suffer from limited interpretability, high
inference costs, and extensive preprocessing requirements, limiting their
practicality for real-time, high-volume log analysis. In contrast, rule-based
systems offer efficiency and transparency, but require significant manual
effort and are difficult to scale across diverse and evolving environments. In
this paper, We present CodeAD, a novel framework that automatically synthesizes
lightweight Python rule functions for LogAD using LLMs. CodeAD introduces a
hierarchical clustering and anchor-grounded sampling strategy to construct
representative contrastive log windows, enabling LLMs to discern discriminative
anomaly patterns. To ensure robustness and generalizability, CodeAD employs an
agentic workflow that iteratively generates, tests, repairs, and refines the
rules until it meets correctness and abstraction requirements. The synthesized
rules are interpretable, lightweight, and directly executable on raw logs,
supporting efficient and transparent online anomaly detection. Our
comprehensive experiments on three public datasets (BGL, Hadoop, Thunderbird)
demonstrate that CodeAD achieves an average absolute improvement of 3.6% F1
score over the state-of-the-art baselines, while processing large datasets up
to 4x faster and at a fraction of the cost (total LLM invocation cost under 4
USD per dataset). These results highlight CodeAD as a practical and scalable
solution for online monitoring systems, enabling interpretable, efficient, and
automated LogAD in real-world environment.

### 5. [The First Star-by-star $N$-body/Hydrodynamics Simulation of Our Galaxy Coupling with a Surrogate Model](http://arxiv.org/pdf/2510.23330v1)

Authors: Keiya Hirashima, Michiko S. Fujii, Takayuki R. Saitoh, Naoto Harada, Kentaro Nomura, Kohji Yoshikawa, Yutaka Hirai, Tetsuro Asano, Kana Moriwaki, Masaki Iwasawa, Takashi Okamoto, Junichiro Makino

A major goal of computational astrophysics is to simulate the Milky Way
Galaxy with sufficient resolution down to individual stars. However, the
scaling fails due to some small-scale, short-timescale phenomena, such as
supernova explosions. We have developed a novel integration scheme of
$N$-body/hydrodynamics simulations working with machine learning. This approach
bypasses the short timesteps caused by supernova explosions using a surrogate
model, thereby improving scalability. With this method, we reached 300 billion
particles using 148,900 nodes, equivalent to 7,147,200 CPU cores, breaking
through the billion-particle barrier currently faced by state-of-the-art
simulations. This resolution allows us to perform the first star-by-star galaxy
simulation, which resolves individual stars in the Milky Way Galaxy. The
performance scales over $10^4$ CPU cores, an upper limit in the current
state-of-the-art simulations using both A64FX and X86-64 processors and NVIDIA
CUDA GPUs.

### 6. [Bayes-Split-Edge: Bayesian Optimization for Constrained Collaborative Inference in Wireless Edge Systems](http://arxiv.org/pdf/2510.23503v1)

Authors: Fatemeh Zahra Safaeipour, Jacob Chakareski, Morteza Hashemi

Mobile edge devices (e.g., AR/VR headsets) typically need to complete timely
inference tasks while operating with limited on-board computing and energy
resources. In this paper, we investigate the problem of collaborative inference
in wireless edge networks, where energy-constrained edge devices aim to
complete inference tasks within given deadlines. These tasks are carried out
using neural networks, and the edge device seeks to optimize inference
performance under energy and delay constraints. The inference process can be
split between the edge device and an edge server, thereby achieving
collaborative inference over wireless networks. We formulate an inference
utility optimization problem subject to energy and delay constraints, and
propose a novel solution called Bayes-Split-Edge, which leverages Bayesian
optimization for collaborative split inference over wireless edge networks. Our
solution jointly optimizes the transmission power and the neural network split
point. The Bayes-Split-Edge framework incorporates a novel hybrid acquisition
function that balances inference task utility, sample efficiency, and
constraint violation penalties. We evaluate our approach using the VGG19 model
on the ImageNet-Mini dataset, and Resnet101 on Tiny-ImageNet, and real-world
mMobile wireless channel datasets. Numerical results demonstrate that
Bayes-Split-Edge achieves up to 2.4x reduction in evaluation cost compared to
standard Bayesian optimization and achieves near-linear convergence. It also
outperforms several baselines, including CMA-ES, DIRECT, exhaustive search, and
Proximal Policy Optimization (PPO), while matching exhaustive search
performance under tight constraints. These results confirm that the proposed
framework provides a sample-efficient solution requiring maximum 20 function
evaluations and constraint-aware optimization for wireless split inference in
edge computing systems.

### 7. [AutoStreamPipe: LLM Assisted Automatic Generation of Data Stream Processing Pipelines](http://arxiv.org/pdf/2510.23408v1)

Authors: Abolfazl Younesi, Zahra Najafabadi Samani, Thomas Fahringer

Data pipelines are essential in stream processing as they enable the
efficient collection, processing, and delivery of real-time data, supporting
rapid data analysis. In this paper, we present AutoStreamPipe, a novel
framework that employs Large Language Models (LLMs) to automate the design,
generation, and deployment of stream processing pipelines. AutoStreamPipe
bridges the semantic gap between high-level user intent and platform-specific
implementations across distributed stream processing systems for structured
multi-agent reasoning by integrating a Hypergraph of Thoughts (HGoT) as an
extended version of GoT. AutoStreamPipe combines resilient execution
strategies, advanced query analysis, and HGoT to deliver pipelines with good
accuracy. Experimental evaluations on diverse pipelines demonstrate that
AutoStreamPipe significantly reduces development time (x6.3) and error rates
(x5.19), as measured by a novel Error-Free Score (EFS), compared to LLM
code-generation methods.

### Digital Libraries

### 1. [Web Archives for Verifying Attribution in Twitter Screenshots](http://arxiv.org/pdf/2510.22939v1)

Authors: Tarannum Zaki, Michael L. Nelson, Michele C. Weigle

Screenshots of social media posts are a common approach for information
sharing. Unfortunately, before sharing a screenshot, users rarely verify
whether the attribution of the post is fake or real. There are numerous
legitimate reasons to share screenshots. However, sharing screenshots of social
media posts is also a vector for mis-/disinformation spread on social media. We
are exploring methods to verify the attribution of a social media post shown in
a screenshot, using resources found on the live web and in web archives. We
focus on the use of web archives, since the attribution of non-deleted posts
can be relatively easily verified using the live web. We show how information
from a Twitter screenshot (Twitter handle, timestamp, and tweet text) can be
extracted and used for locating potential archived tweets in the Internet
Archive's Wayback Machine. We evaluate our method on a dataset of 1,571 single
tweet screenshots.

### 2. [Fake scientific journals are here to stay](http://arxiv.org/pdf/2510.23146v1)

Authors: Enrique Orduña-Malea

Scientific publishing is facing an alarming proliferation of fraudulent
practices that threaten the integrity of research communication. The production
and dissemination of fake research have become a profitable business,
undermining trust in scientific journals and distorting the evaluation
processes that depend on them. This brief piece examines the problem of fake
journals through a three-level typology. The first level concerns predatory
journals, which prioritise financial gain over scholarly quality by charging
authors publication fees while providing superficial or fabricated peer review.
The second level analyses hijacked journals, in which counterfeit websites
impersonate legitimate titles to deceive authors into submitting and paying for
publication. The third level addresses hacked journals, where legitimate
platforms are compromised through cyberattacks or internal manipulation,
enabling the distortion of review and publication processes. Together, these
forms of misconduct expose deep vulnerabilities in the scientific communication
ecosystem, exacerbated by the pressure to publish and the marketisation of
research outputs. The manuscript concludes that combating these practices
requires structural reforms in scientific evaluation and governance. Only by
reducing the incentives that sustain the business of fraudulent publishing can
the scholarly community restore credibility and ensure that scientific
communication fulfils the essential purpose of reliable advancement of
knowledge.

### Data Structures and Algorithms

### 1. [Multi-Way Co-Ranking: Index-Space Partitioning of Sorted Sequences Without Merge](http://arxiv.org/pdf/2510.22882v1)

Authors: Amit Joshi

We present a merge-free algorithm for multi-way co-ranking, the problem of
computing cut indices $i_1,\dots,i_m$ that partition each of the $m$ sorted
sequences such that all prefix segments together contain exactly $K$ elements.
Our method extends two-list co-ranking to arbitrary $m$, maintaining
per-sequence bounds that converge to a consistent global frontier without
performing any multi-way merge or value-space search. Rather, we apply binary
search to \emph{index-space}. The algorithm runs in $O(\log(\sum_t n_t)\,\log
m)$ time and $O(m)$ space, independent of $K$. We prove correctness via an
exchange argument and discuss applications to distributed fractional knapsack,
parallel merge partitioning, and multi-stream joins.
  Keywords: Co-ranking \sep partitioning \sep Merge-free algorithms \sep
Index-space optimization \sep Selection and merging \sep Data structures

### 2. [Sublinear Sketches for Approximate Nearest Neighbor and Kernel Density Estimation](http://arxiv.org/pdf/2510.23039v1)

Authors: Ved Danait, Srijan Das, Sujoy Bhore

Approximate Nearest Neighbor (ANN) search and Approximate Kernel Density
Estimation (A-KDE) are fundamental problems at the core of modern machine
learning, with broad applications in data analysis, information systems, and
large-scale decision making. In massive and dynamic data streams, a central
challenge is to design compact sketches that preserve essential structural
properties of the data while enabling efficient queries.
  In this work, we develop new sketching algorithms that achieve sublinear
space and query time guarantees for both ANN and A-KDE for a dynamic stream of
data. For ANN in the streaming model, under natural assumptions, we design a
sublinear sketch that requires only $\mathcal{O}(n^{1+\rho-\eta})$ memory by
storing only a sublinear ($n^{-\eta}$) fraction of the total inputs, where
$\rho$ is a parameter of the LSH family, and $0<\eta<1$. Our method supports
sublinear query time, batch queries, and extends to the more general Turnstile
model. While earlier works have focused on Exact NN, this is the first result
on ANN that achieves near-optimal trade-offs between memory size and
approximation error.
  Next, for A-KDE in the Sliding-Window model, we propose a sketch of size
$\mathcal{O}\left(RW \cdot \frac{1}{\sqrt{1+\epsilon} - 1} \log^2 N\right)$,
where $R$ is the number of sketch rows, $W$ is the LSH range, $N$ is the window
size, and $\epsilon$ is the approximation error. This, to the best of our
knowledge, is the first theoretical sublinear sketch guarantee for A-KDE in the
Sliding-Window model.
  We complement our theoretical results with experiments on various real-world
datasets, which show that the proposed sketches are lightweight and achieve
consistently low error in practice.

### 3. [Coresets for Clustering Under Stochastic Noise](http://arxiv.org/pdf/2510.23438v1)

Authors: Lingxiao Huang, Zhize Li, Nisheeth K. Vishnoi, Runkai Yang, Haoyu Zhao

We study the problem of constructing coresets for $(k, z)$-clustering when
the input dataset is corrupted by stochastic noise drawn from a known
distribution. In this setting, evaluating the quality of a coreset is
inherently challenging, as the true underlying dataset is unobserved. To
address this, we investigate coreset construction using surrogate error metrics
that are tractable and provably related to the true clustering cost. We analyze
a traditional metric from prior work and introduce a new error metric that more
closely aligns with the true cost. Although our metric is defined independently
of the noise distribution, it enables approximation guarantees that scale with
the noise level. We design a coreset construction algorithm based on this
metric and show that, under mild assumptions on the data and noise, enforcing
an $\varepsilon$-bound under our metric yields smaller coresets and tighter
guarantees on the true clustering cost than those obtained via classical
metrics. In particular, we prove that the coreset size can improve by a factor
of up to $\mathrm{poly}(k)$, where $n$ is the dataset size. Experiments on
real-world datasets support our theoretical findings and demonstrate the
practical advantages of our approach.

### Emerging Technologies

### 1. [IoT-Driven Smart Management in Broiler Farming: Simulation of Remote Sensing and Control Systems](http://arxiv.org/pdf/2510.23356v1)

Authors: Sandra Coello Suarez, V. Sanchez Padilla, Ronald Ponguillo-Intriago, Albert Espinal

Parameter monitoring and control systems are crucial in the industry as they
enable automation processes that improve productivity and resource
optimization. These improvements also help to manage environmental factors and
the complex interactions between multiple inputs and outputs required for
production management. This paper proposes an automation system for broiler
management based on a simulation scenario that involves sensor networks and
embedded systems. The aim is to create a transmission network for monitoring
and controlling broiler temperature and feeding using the Internet of Things
(IoT), complemented by a dashboard and a cloud-based service database to track
improvements in broiler management. We look forward this work will serve as a
guide for stakeholders and entrepreneurs in the animal production industry,
fostering sustainable development through simple and cost-effective automation
solutions. The goal is for them to scale and integrate these recommendations
into their existing operations, leading to more efficient decision-making at
the management level.

### 2. [Probabilistic Computing Optimization of Complex Spin-Glass Topologies](http://arxiv.org/pdf/2510.23419v1)

Authors: Fredrik Hasselgren, Max O. Al-Hasso, Amy Searle, Joseph Tindall, Marko von der Leyen

Spin glass systems as lattices of disordered magnets with random interactions
have important implications within the theory of magnetization and applications
to a wide-range of hard combinatorial optimization problems. Nevertheless,
despite sustained efforts, algorithms that attain both high accuracy and
efficiency remain elusive. Due to their topologies being low-$k$-partite such
systems are well suited to a probabilistic computing (PC) approach using
probabilistic bits (P-bits). Here we present complex spin glass topologies
solved on a simulated PC realization of an Ising machine. First, we considered
a number of three dimensional Edwards-Anderson cubic spin-glasses randomly
generated as well as found in the literature as a benchmark. Second, biclique
topologies were identified as a likely candidate for a comparative advantage
compared to other state-of-the-art techniques, with a range of sizes simulated.
We find that the number of iterations necessary to find solutions of a given
quality has constant scaling with system size past a saturation point if one
assumes perfect parallelization of the hardware. Therefore a PC architecture
can trade the computational depth of other methods for parallelized width by
connecting a number of P-bits that scales linearly in system size. This
constant scaling is shown to persist across a number of solution qualities, up
to a certain limit beyond which resource constraints limited further
investigation. The saturation point varies between topologies and qualities and
becomes exponentially hard in the limit of finding the ground truth.
Furthermore we demonstrate that our PC architecture can solve spin-glass
topologies to the same quality as the most advanced quantum annealer in
minutes, making modest assumptions about their implementation on hardware.

### 3. [AutoStreamPipe: LLM Assisted Automatic Generation of Data Stream Processing Pipelines](http://arxiv.org/pdf/2510.23408v1)

Authors: Abolfazl Younesi, Zahra Najafabadi Samani, Thomas Fahringer

Data pipelines are essential in stream processing as they enable the
efficient collection, processing, and delivery of real-time data, supporting
rapid data analysis. In this paper, we present AutoStreamPipe, a novel
framework that employs Large Language Models (LLMs) to automate the design,
generation, and deployment of stream processing pipelines. AutoStreamPipe
bridges the semantic gap between high-level user intent and platform-specific
implementations across distributed stream processing systems for structured
multi-agent reasoning by integrating a Hypergraph of Thoughts (HGoT) as an
extended version of GoT. AutoStreamPipe combines resilient execution
strategies, advanced query analysis, and HGoT to deliver pipelines with good
accuracy. Experimental evaluations on diverse pipelines demonstrate that
AutoStreamPipe significantly reduces development time (x6.3) and error rates
(x5.19), as measured by a novel Error-Free Score (EFS), compared to LLM
code-generation methods.

### Formal Languages and Automata Theory

### 1. [Proceedings of the Combined 32nd International Workshop on Expressiveness in Concurrency and 22nd Workshop on Structural Operational Semantics](http://arxiv.org/pdf/2510.23211v1)

Authors: Cinzia Di Giusto, Giorgio Bacci

This volume contains the proceedings of EXPRESS/SOS 2025: the Combined 32nd
International Workshop on Expressiveness in Concurrency and the 22nd Workshop
on Structural Operational Semantics, which was held in Aarhus, Denmark, as an
affiliated workshop of CONFEST 2025. The EXPRESS/SOS workshop series aims at
bringing together researchers interested in the formal semantics of systems and
programming concepts, and in the expressiveness of computational models.

### 2. [Are Agents Just Automata? On the Formal Equivalence Between Agentic AI and the Chomsky Hierarchy](http://arxiv.org/pdf/2510.23487v1)

Authors: Roham Koohestani, Ziyou Li, Anton Podkopaev, Maliheh Izadi

This paper establishes a formal equivalence between the architectural classes
of modern agentic AI systems and the abstract machines of the Chomsky
hierarchy. We posit that the memory architecture of an AI agent is the
definitive feature determining its computational power and that it directly
maps it to a corresponding class of automaton. Specifically, we demonstrate
that simple reflex agents are equivalent to Finite Automata, hierarchical
task-decomposition agents are equivalent to Pushdown Automata, and agents
employing readable/writable memory for reflection are equivalent to TMs. This
Automata-Agent Framework provides a principled methodology for right-sizing
agent architectures to optimize computational efficiency and cost. More
critically, it creates a direct pathway to formal verification, enables the
application of mature techniques from automata theory to guarantee agent safety
and predictability. By classifying agents, we can formally delineate the
boundary between verifiable systems and those whose behavior is fundamentally
undecidable. We address the inherent probabilistic nature of LLM-based agents
by extending the framework to probabilistic automata that allow quantitative
risk analysis. The paper concludes by outlining an agenda for developing static
analysis tools and grammars for agentic frameworks.

### General Literature

### 1. [Education Paradigm Shift To Maintain Human Competitive Advantage Over AI](http://arxiv.org/pdf/2510.23436v1)

Authors: Stanislav Selitskiy, Chihiro Inoue

Discussion about the replacement of intellectual human labour by ``thinking
machines'' has been present in the public and expert discourse since the
creation of Artificial Intelligence (AI) as an idea and terminology since the
middle of the twentieth century. Until recently, it was more of a hypothetical
concern. However, in recent years, with the rise of Generative AI, especially
Large Language Models (LLM), and particularly with the widespread popularity of
the ChatGPT model, that concern became practical. Many domains of human
intellectual labour have to adapt to the new AI tools that give humans new
functionality and opportunity, but also question the viability and necessity of
some human work that used to be considered intellectual yet has now become an
easily automatable commodity. Education, unexpectedly, has now become burdened
by an especially crucial role of charting long-range strategies for discovering
viable human skills that would guarantee their place in the world of the
ubiquitous use of AI in the intellectual sphere. We highlight weaknesses of the
current AI and, especially, of its LLM-based core, show that root causes of
LLMs' weaknesses are unfixable by the current technologies, and propose
directions in the constructivist paradigm for the changes in Education that
ensure long-term advantages of humans over AI tools.

### Graphics

### 1. [FlowCapX: Physics-Grounded Flow Capture with Long-Term Consistency](http://arxiv.org/pdf/2510.23122v1)

Authors: Ningxiao Tao, Liru Zhang, Xingyu Ni, Mengyu Chu, Baoquan Chen

We present FlowCapX, a physics-enhanced framework for flow reconstruction
from sparse video inputs, addressing the challenge of jointly optimizing
complex physical constraints and sparse observational data over long time
horizons. Existing methods often struggle to capture turbulent motion while
maintaining physical consistency, limiting reconstruction quality and
downstream tasks. Focusing on velocity inference, our approach introduces a
hybrid framework that strategically separates representation and supervision
across spatial scales. At the coarse level, we resolve sparse-view ambiguities
via a novel optimization strategy that aligns long-term observation with
physics-grounded velocity fields. By emphasizing vorticity-based physical
constraints, our method enhances physical fidelity and improves optimization
stability. At the fine level, we prioritize observational fidelity to preserve
critical turbulent structures. Extensive experiments demonstrate
state-of-the-art velocity reconstruction, enabling velocity-aware downstream
tasks, e.g., accurate flow analysis, scene augmentation with tracer
visualization and re-simulation.

### 2. [Yesnt: Are Diffusion Relighting Models Ready for Capture Stage Compositing? A Hybrid Alternative to Bridge the Gap](http://arxiv.org/pdf/2510.23494v1)

Authors: Elisabeth Jüttner, Leona Krath, Stefan Korfhage, Hannah Dröge, Matthias B. Hullin, Markus Plack

Volumetric video relighting is essential for bringing captured performances
into virtual worlds, but current approaches struggle to deliver temporally
stable, production-ready results. Diffusion-based intrinsic decomposition
methods show promise for single frames, yet suffer from stochastic noise and
instability when extended to sequences, while video diffusion models remain
constrained by memory and scale. We propose a hybrid relighting framework that
combines diffusion-derived material priors with temporal regularization and
physically motivated rendering. Our method aggregates multiple stochastic
estimates of per-frame material properties into temporally consistent shading
components, using optical-flow-guided regularization. For indirect effects such
as shadows and reflections, we extract a mesh proxy from Gaussian Opacity
Fields and render it within a standard graphics pipeline. Experiments on real
and synthetic captures show that this hybrid strategy achieves substantially
more stable relighting across sequences than diffusion-only baselines, while
scaling beyond the clip lengths feasible for video diffusion. These results
indicate that hybrid approaches, which balance learned priors with physically
grounded constraints, are a practical step toward production-ready volumetric
video relighting.

### 3. [VoMP: Predicting Volumetric Mechanical Property Fields](http://arxiv.org/pdf/2510.22975v1)

Authors: Rishit Dagli, Donglai Xiang, Vismay Modi, Charles Loop, Clement Fuji Tsang, Anka He Chen, Anita Hu, Gavriel State, David I. W. Levin, Maria Shugrina

Physical simulation relies on spatially-varying mechanical properties, often
laboriously hand-crafted. VoMP is a feed-forward method trained to predict
Young's modulus ($E$), Poisson's ratio ($\nu$), and density ($\rho$) throughout
the volume of 3D objects, in any representation that can be rendered and
voxelized. VoMP aggregates per-voxel multi-view features and passes them to our
trained Geometry Transformer to predict per-voxel material latent codes. These
latents reside on a manifold of physically plausible materials, which we learn
from a real-world dataset, guaranteeing the validity of decoded per-voxel
materials. To obtain object-level training data, we propose an annotation
pipeline combining knowledge from segmented 3D datasets, material databases,
and a vision-language model, along with a new benchmark. Experiments show that
VoMP estimates accurate volumetric properties, far outperforming prior art in
accuracy and speed.

### 4. [Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling](http://arxiv.org/pdf/2510.23605v1)

Authors: Shuhong Zheng, Ashkan Mirzaei, Igor Gilitschenski

Current 3D/4D generation methods are usually optimized for photorealism,
efficiency, and aesthetics. However, they often fail to preserve the semantic
identity of the subject across different viewpoints. Adapting generation
methods with one or few images of a specific subject (also known as
Personalization or Subject-driven generation) allows generating visual content
that align with the identity of the subject. However, personalized 3D/4D
generation is still largely underexplored. In this work, we introduce TIRE
(Track, Inpaint, REsplat), a novel method for subject-driven 3D/4D generation.
It takes an initial 3D asset produced by an existing 3D generative model as
input and uses video tracking to identify the regions that need to be modified.
Then, we adopt a subject-driven 2D inpainting model for progressively infilling
the identified regions. Finally, we resplat the modified 2D multi-view
observations back to 3D while still maintaining consistency. Extensive
experiments demonstrate that our approach significantly improves identity
preservation in 3D/4D generation compared to state-of-the-art methods. Our
project website is available at
https://zsh2000.github.io/track-inpaint-resplat.github.io/.

### Computer Science and Game Theory

### 1. [Feedback in Dynamic Contests: Theory and Experiment](http://arxiv.org/pdf/2510.23178v1)

Authors: Sumit Goel, Yiqing Yan, Jeffrey Zeidel

We study the effect of interim feedback policies in a dynamic all-pay auction
where two players bid over two stages to win a common-value prize. We show that
sequential equilibrium outcomes are characterized by Cheapest Signal
Equilibria, wherein stage 1 bids are such that one player bids zero while the
other chooses a cheapest bid consistent with some signal. Equilibrium payoffs
for both players are always zero, and the sum of expected total bids equals the
value of the prize. We conduct an experiment with four natural feedback policy
treatments -- full, rank, and two cutoff policies -- and while the bidding
behavior deviates from equilibrium, we fail to reject the hypothesis of no
treatment effect on total bids. Further, stage 1 bids induce sunk costs and
head starts, and we test for the resulting sunk cost and discouragement effects
in stage 2 bidding.

### Human-Computer Interaction

### 1. [Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces](http://arxiv.org/pdf/2510.22922v1)

Authors: Runtao Zhou, Giang Nguyen, Nikita Kharya, Anh Nguyen, Chirag Agarwal

The reasoning capabilities of Large Language Models (LLMs) have led to their
increasing employment in several critical applications, particularly education,
where they support problem-solving, tutoring, and personalized study. While
there are a plethora of works showing the effectiveness of LLMs in generating
step-by-step solutions through chain-of-thought (CoT) reasoning on reasoning
benchmarks, little is understood about whether the generated CoT is helpful for
end-users in improving their ability to comprehend mathematical reasoning
problems and detect errors/hallucinations in LLM-generated solutions. To
address this gap and contribute to understanding how reasoning can improve
human-AI interaction, we present three new interactive reasoning interfaces:
interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive
Graph (iGraph), and a novel framework that generates the LLM's reasoning from
traditional CoT to alternative, interactive formats. Across 125 participants,
we found that interactive interfaces significantly improved performance.
Specifically, the iGraph interface yielded the highest clarity and error
detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all
outperforming standard CoT (73.5%). Interactive interfaces also led to faster
response times, where participants using iGraph were fastest (57.9 secs),
compared to iCoT and iPoT (60 secs), and the standard CoT baseline (64.7 secs).
Furthermore, participants preferred the iGraph reasoning interface, citing its
superior ability to enable users to follow the LLM's reasoning process. We
discuss the implications of these results and provide recommendations for the
future design of reasoning models.

### 2. [Reasoning About Reasoning: Towards Informed and Reflective Use of LLM Reasoning in HCI](http://arxiv.org/pdf/2510.22978v1)

Authors: Ramaravind Kommiya Mothilal, Sally Zhang, Syed Ishtiaque Ahmed, Shion Guha

Reasoning is a distinctive human-like characteristic attributed to LLMs in
HCI due to their ability to simulate various human-level tasks. However, this
work argues that the reasoning behavior of LLMs in HCI is often
decontextualized from the underlying mechanics and subjective decisions that
condition the emergence and human interpretation of this behavior. Through a
systematic survey of 258 CHI papers from 2020-2025 on LLMs, we discuss how HCI
hardly perceives LLM reasoning as a product of sociotechnical orchestration and
often references it as an object of application. We argue that such abstraction
leads to oversimplification of reasoning methodologies from NLP/ML and results
in a distortion of LLMs' empirically studied capabilities and (un)known
limitations. Finally, drawing on literature from both NLP/ML and HCI, as a
constructive step forward, we develop reflection prompts to support HCI
practitioners engage with LLM reasoning in an informed and reflective way.

### 3. [Moderating Role of Presence in EEG Responses to Visuo-haptic Prediction Error in Virtual Reality](http://arxiv.org/pdf/2510.23262v1)

Authors: Lukas Gehrke, Leonie Terfurth, Klaus Gramann

Virtual reality (VR) can create compelling experiences that evoke presence,
the sense of ``being there.'' However, problems in rendering can create
sensorimotor disruptions that undermine presence and task performance. Presence
is typically assessed with post-hoc questionnaires, but their coarse temporal
resolution limits insight into how sensorimotor disruptions shape user
experience. Here, we combined questionnaires with electroencephalography (EEG)
to identify neural markers of presence-affecting prediction error in immersive
VR. Twenty-five participants performed a grasp-and-place task under two levels
of immersion (visual-only vs.~visuo-haptic). Occasional oddball-like
sensorimotor disruptions introduced premature feedback to elicit prediction
errors. Overall, higher immersion enhanced self-presence but not physical
presence, while accuracy and speed improved over time irrespective of
immersion. At the neural level, sensorimotor disruptions elicited robust
event-related potential effects at FCz and Pz, accompanied by increases in
frontal midline $\theta$ and posterior $\alpha$ suppression. Through source
analyses localized to anterior- and posterior cingulate cortex (ACC/PCC) we
found that PCC $\alpha$ activity showed heightened sensitivity to disruptions
exclusively in visuo-haptic immersion. Exploratory moderation analyses by
presence scores revealed no consistent patterns. Together, these results
suggest that higher immersion amplifies both the benefits and costs of
sensorimotor coherence.

### 4. [Partnering with Generative AI: Experimental Evaluation of Human-Led and Model-Led Interaction in Human-AI Co-Creation](http://arxiv.org/pdf/2510.23324v1)

Authors: Sebastian Maier, Manuel Schneider, Stefan Feuerriegel

Large language models (LLMs) show strong potential to support creative tasks,
but the role of the interface design is poorly understood. In particular, the
effect of different modes of collaboration between humans and LLMs on
co-creation outcomes is unclear. To test this, we conducted a randomized
controlled experiment ($N = 486$) comparing: (a) two variants of reflective,
human-led modes in which the LLM elicits elaboration through suggestions or
questions, against (b) a proactive, model-led mode in which the LLM
independently rewrites ideas. By assessing the effects on idea quality,
diversity, and perceived ownership, we found that the model-led mode
substantially improved idea quality but reduced idea diversity and users'
perceived idea ownership. The reflective, human-led mode also improved idea
quality, yet while preserving diversity and ownership. Our findings highlight
the importance of designing interactions with generative AI systems as
reflective thought partners that complement human strengths and augment
creative processes.

### 5. [Reciprocity Deficits: Observing AI in the street with everyday publics](http://arxiv.org/pdf/2510.23342v1)

Authors: Alex S. Taylor, Noortje Marres, Mercedes Bunz, Thao Phan, Maya Indira Ganesh, Dominique Barron, Yasmine Boudiaf, Rachel Coldicutt, Iain Emsley, Beatrice Gobbo, Louise Hickman, Bettina Nissen, Mukul Patel, Luis Soares

The street has emerged as a primary site where everyday publics are
confronted with AI as an infrastructural phenomenon, as machine learning-based
systems are now commonly deployed in this setting in the form of automated
cars, facial recognition, smart billboards and the like. While these
deployments of AI in the street have attracted significant media attention and
public controversy in recent years, the presence of AI in the street often
remains inscrutable, and many everyday publics are unaware of it. In this
paper, we explore the challenges and possibilities of everyday public
engagement with AI in the situated environment of city streets under these
paradoxical conditions. Combining perspectives and approaches from social and
cultural studies of AI, Design Research and Science and Technology Studies
(STS), we explore the affordances of the street as a site for 'material
participation' in AI through design-based interventions: the creation of
'everyday AI observatories.' We narrate and reflect on our participatory
observations of AI in five city streets in the UK and Australia and highlight a
set of tensions that emerged from them: 1) the framing of the street as a
transactional environment, 2) the designed invisibility of AI and its publics
in the street 3) the stratification of street environments through statistical
governance. Based on this discussion and drawing on Jane Jacobs' notion of
"eyes on the street," we put forward the relational notion of "reciprocity
deficits" between AI infrastructures and everyday publics in the street. The
conclusion reflects on the consequences of this form of social invisibility of
AI for situated engagement with AI by everyday publics in the street and for
public trust in urban governance.

### 6. [Shareholder Democracy with AI Representatives](http://arxiv.org/pdf/2510.23475v1)

Authors: Suyash Fulay, Sercan Demir, Galen Hines-Pierce, Hélène Landemore, Michiel Bakker

A large share of retail investors hold public equities through mutual funds,
yet lack adequate control over these investments. Indeed, mutual funds
concentrate voting power in the hands of a few asset managers. These managers
vote on behalf of shareholders despite having limited insight into their
individual preferences, leaving them exposed to growing political and
regulatory pressures, particularly amid rising shareholder activism.
Pass-through voting has been proposed as a way to empower retail investors and
provide asset managers with clearer guidance, but it faces challenges such as
low participation rates and the difficulty of capturing highly individualized
shareholder preferences for each specific vote. Randomly selected assemblies of
shareholders, or ``investor assemblies,'' have also been proposed as more
representative proxies than asset managers. As a third alternative, we propose
artificial intelligence (AI) enabled representatives trained on individual
shareholder preferences to act as proxies and vote on their behalf. Over time,
these models could not only predict how retail investors would vote at any
given moment but also how they might vote if they had significantly more time,
knowledge, and resources to evaluate each proposal, leading to better overall
decision-making. We argue that shareholder democracy offers a compelling
real-world test bed for AI-enabled representation, providing valuable insights
into both the potential benefits and risks of this approach more generally.

### 7. [If They Disagree, Will You Conform? Exploring the Role of Robots' Value Awareness in a Decision-Making Task](http://arxiv.org/pdf/2510.23204v1)

Authors: Giulia Pusceddu, Giulio Antonio Abbo, Francesco Rea, Tony Belpaeme, Alessandra Sciutti

This study investigates whether the opinions of robotic agents are more
likely to influence human decision-making when the robots are perceived as
value-aware (i.e., when they display an understanding of human principles). We
designed an experiment in which participants interacted with two Furhat robots
- one programmed to be Value-Aware and the other Non-Value-Aware - during a
labeling task for images representing human values. Results indicate that
participants distinguished the Value-Aware robot from the Non-Value-Aware one.
Although their explicit choices did not indicate a clear preference for one
robot over the other, participants directed their gaze more toward the
Value-Aware robot. Additionally, the Value-Aware robot was perceived as more
loyal, suggesting that value awareness in a social robot may enhance its
perceived commitment to the group. Finally, when both robots disagreed with the
participant, conformity occurred in about one out of four trials, and
participants took longer to confirm their responses, suggesting that two robots
expressing dissent may introduce hesitation in decision-making. On one hand,
this highlights the potential risk that robots, if misused, could manipulate
users for unethical purposes. On the other hand, it reinforces the idea that
social robots might encourage reflection in ambiguous situations and help users
avoid scams.

### 8. [Multi-Stakeholder Alignment in LLM-Powered Collaborative AI Systems: A Multi-Agent Framework for Intelligent Tutoring](http://arxiv.org/pdf/2510.23245v1)

Authors: Alexandre P Uchoa, Carlo E T Oliveira, Claudia L R Motta, Daniel Schneider

The integration of Large Language Models into Intelligent Tutoring Systems
pre-sents significant challenges in aligning with diverse and often conflicting
values from students, parents, teachers, and institutions. Existing
architectures lack for-mal mechanisms for negotiating these multi-stakeholder
tensions, creating risks in accountability and bias. This paper introduces the
Advisory Governance Layer (AGL), a non-intrusive, multi-agent framework
designed to enable distributed stakeholder participation in AI governance. The
AGL employs specialized agents representing stakeholder groups to evaluate
pedagogical actions against their spe-cific policies in a privacy-preserving
manner, anticipating future advances in per-sonal assistant technology that
will enhance stakeholder value expression. Through a novel policy taxonomy and
conflict-resolution protocols, the frame-work provides structured, auditable
governance advice to the ITS without altering its core pedagogical
decision-making. This work contributes a reference architec-ture and technical
specifications for aligning educational AI with multi-stakeholder values,
bridging the gap between high-level ethical principles and practical
implementation.

### 9. [Education Paradigm Shift To Maintain Human Competitive Advantage Over AI](http://arxiv.org/pdf/2510.23436v1)

Authors: Stanislav Selitskiy, Chihiro Inoue

Discussion about the replacement of intellectual human labour by ``thinking
machines'' has been present in the public and expert discourse since the
creation of Artificial Intelligence (AI) as an idea and terminology since the
middle of the twentieth century. Until recently, it was more of a hypothetical
concern. However, in recent years, with the rise of Generative AI, especially
Large Language Models (LLM), and particularly with the widespread popularity of
the ChatGPT model, that concern became practical. Many domains of human
intellectual labour have to adapt to the new AI tools that give humans new
functionality and opportunity, but also question the viability and necessity of
some human work that used to be considered intellectual yet has now become an
easily automatable commodity. Education, unexpectedly, has now become burdened
by an especially crucial role of charting long-range strategies for discovering
viable human skills that would guarantee their place in the world of the
ubiquitous use of AI in the intellectual sphere. We highlight weaknesses of the
current AI and, especially, of its LLM-based core, show that root causes of
LLMs' weaknesses are unfixable by the current technologies, and propose
directions in the constructivist paradigm for the changes in Education that
ensure long-term advantages of humans over AI tools.

### 10. [Emotion-Coherent Reasoning for Multimodal LLMs via Emotional Rationale Verifier](http://arxiv.org/pdf/2510.23506v1)

Authors: Hyeongseop Rha, Jeong Hun Yeo, Yeonju Kim, Yong Man Ro

The recent advancement of Multimodal Large Language Models (MLLMs) is
transforming human-computer interaction (HCI) from surface-level exchanges into
more nuanced and emotionally intelligent communication. To realize this shift,
emotion understanding becomes essential allowing systems to capture subtle cues
underlying user intent. Furthermore, providing faithful explanations for
predicted emotions is crucial to ensure interpretability and build user trust.
However, current MLLM-based methods often generate emotion explanations that
diverge from the target labels and sometimes even contradict their own
predicted emotions. This inconsistency poses a critical risk for
misunderstanding and erodes reliability in interactive settings. To address
this, we propose a novel approach: the Emotional Rationale Verifier (ERV) and
an Explanation Reward. Our method guides the model to produce reasoning that is
explicitly consistent with the target emotion during multimodal emotion
recognition without modifying the model architecture or requiring additional
paired video-description annotations. Our method significantly improves
faithful explanation-prediction consistency and explanation emotion accuracy on
the MAFW and DFEW datasets. Through extensive experiments and human
evaluations, we show that our approach not only enhances alignment between
explanation and prediction but also empowers MLLMs to deliver emotionally
coherent, trustworthy interactions, marking a key step toward truly human-like
HCI systems.

### Information Retrieval

### 1. [MGFRec: Towards Reinforced Reasoning Recommendation with Multiple Groundings and Feedback](http://arxiv.org/pdf/2510.22888v1)

Authors: Shihao Cai, Chongming Gao, Haoyan Liu, Wentao Shi, Jianshan Sun, Ruiming Tang, Fuli Feng

The powerful reasoning and generative capabilities of large language models
(LLMs) have inspired researchers to apply them to reasoning-based
recommendation tasks, which require in-depth reasoning about user interests and
the generation of recommended items. However, previous reasoning-based
recommendation methods have typically performed inference within the language
space alone, without incorporating the actual item space. This has led to
over-interpreting user interests and deviating from real items. Towards this
research gap, we propose performing multiple rounds of grounding during
inference to help the LLM better understand the actual item space, which could
ensure that its reasoning remains aligned with real items. Furthermore, we
introduce a user agent that provides feedback during each grounding step,
enabling the LLM to better recognize and adapt to user interests. Comprehensive
experiments conducted on three Amazon review datasets demonstrate the
effectiveness of incorporating multiple groundings and feedback. These findings
underscore the critical importance of reasoning within the actual item space,
rather than being confined to the language space, for recommendation tasks.

### 2. [Improving Product Search Relevance with EAR-MP: A Solution for the CIKM 2025 AnalytiCup](http://arxiv.org/pdf/2510.23018v1)

Authors: JaeEun Lim, Soomin Kim, Jaeyong Seo, Iori Ono, Qimu Ran, Jae-woong Lee

Multilingual e-commerce search is challenging due to linguistic diversity and
the noise inherent in user-generated queries. This paper documents the solution
employed by our team (EAR-MP) for the CIKM 2025 AnalytiCup, which addresses two
core tasks: Query-Category (QC) relevance and Query-Item (QI) relevance. Our
approach first normalizes the multilingual dataset by translating all text into
English, then mitigates noise through extensive data cleaning and
normalization. For model training, we build on DeBERTa-v3-large and improve
performance with label smoothing, self-distillation, and dropout. In addition,
we introduce task-specific upgrades, including hierarchical token injection for
QC and a hybrid scoring mechanism for QI. Under constrained compute, our method
achieves competitive results, attaining an F1 score of 0.8796 on QC and 0.8744
on QI. These findings underscore the importance of systematic data
preprocessing and tailored training strategies for building robust,
resource-efficient multilingual relevance systems.

### 3. [Multi-Stage Field Extraction of Financial Documents with OCR and Compact Vision-Language Models](http://arxiv.org/pdf/2510.23066v1)

Authors: Yichao Jin, Yushuo Wang, Qishuai Zhong, Kent Chiu Jin-Chun, Kenneth Zhu Ke, Donald MacDonald

Financial documents are essential sources of information for regulators,
auditors, and financial institutions, particularly for assessing the wealth and
compliance of Small and Medium-sized Businesses. However, SMB documents are
often difficult to parse. They are rarely born digital and instead are
distributed as scanned images that are none machine readable. The scans
themselves are low in resolution, affected by skew or rotation, and often
contain noisy backgrounds. These documents also tend to be heterogeneous,
mixing narratives, tables, figures, and multilingual content within the same
report. Such characteristics pose major challenges for automated information
extraction, especially when relying on end to end large Vision Language Models,
which are computationally expensive, sensitive to noise, and slow when applied
to files with hundreds of pages.
  We propose a multistage pipeline that leverages traditional image processing
models and OCR extraction, together with compact VLMs for structured field
extraction of large-scale financial documents. Our approach begins with image
pre-processing, including segmentation, orientation detection, and size
normalization. Multilingual OCR is then applied to recover page-level text.
Upon analyzing the text information, pages are retrieved for coherent sections.
Finally, compact VLMs are operated within these narrowed-down scopes to extract
structured financial indicators.
  Our approach is evaluated using an internal corpus of multi-lingual, scanned
financial documents. The results demonstrate that compact VLMs, together with a
multistage pipeline, achieves 8.8 times higher field level accuracy relative to
directly feeding the whole document into large VLMs, only at 0.7 percent of the
GPU cost and 92.6 percent less end-to-end service latency.

### 4. [GTR-Mamba: Geometry-to-Tangent Routing for Hyperbolic POI Recommendation](http://arxiv.org/pdf/2510.22942v1)

Authors: Zhuoxuan Li, Jieyuan Pei, Tangwei Ye, Zhongyuan Lai, Zihan Liu, Fengyuan Xu, Qi Zhang, Liang Hu

Next Point-of-Interest (POI) recommendation is a critical task in modern
Location-Based Social Networks (LBSNs), aiming to model the complex
decision-making process of human mobility to provide personalized
recommendations for a user's next check-in location. Existing POI
recommendation models, predominantly based on Graph Neural Networks and
sequential models, have been extensively studied. However, these models face a
fundamental limitation: they struggle to simultaneously capture the inherent
hierarchical structure of spatial choices and the dynamics and irregular shifts
of user-specific temporal contexts. To overcome this limitation, we propose
GTR-Mamba, a novel framework for cross-manifold conditioning and routing.
GTR-Mamba leverages the distinct advantages of different mathematical spaces
for different tasks: it models the static, tree-like preference hierarchies in
hyperbolic geometry, while routing the dynamic sequence updates to a novel
Mamba layer in the computationally stable and efficient Euclidean tangent
space. This process is coordinated by a cross-manifold channel that fuses
spatio-temporal information to explicitly steer the State Space Model (SSM),
enabling flexible adaptation to contextual changes. Extensive experiments on
three real-world datasets demonstrate that GTR-Mamba consistently outperforms
state-of-the-art baseline models in next POI recommendation.

### 5. [Tagging-Augmented Generation: Assisting Language Models in Finding Intricate Knowledge In Long Contexts](http://arxiv.org/pdf/2510.22956v1)

Authors: Anwesan Pal, Karen Hovsepian, Tinghao Guo, Mengnan Zhao, Somendra Tripathi, Nikos Kanakaris, George Mihaila, Sumit Nigam

Recent investigations into effective context lengths of modern flagship large
language models (LLMs) have revealed major limitations in effective question
answering (QA) and reasoning over long and complex contexts for even the
largest and most impressive cadre of models. While approaches like
retrieval-augmented generation (RAG) and chunk-based re-ranking attempt to
mitigate this issue, they are sensitive to chunking, embedding and retrieval
strategies and models, and furthermore, rely on extensive pre-processing,
knowledge acquisition and indexing steps. In this paper, we propose
Tagging-Augmented Generation (TAG), a lightweight data augmentation strategy
that boosts LLM performance in long-context scenarios, without degrading and
altering the integrity and composition of retrieved documents. We validate our
hypothesis by augmenting two challenging and directly relevant
question-answering benchmarks -- NoLima and NovelQA -- and show that tagging
the context or even just adding tag definitions into QA prompts leads to
consistent performance gains over the baseline -- up to 17% for 32K token
contexts, and 2.9% in complex reasoning question-answering for multi-hop
queries requiring knowledge across a wide span of text. Additional details are
available at https://sites.google.com/view/tag-emnlp.

### 6. [Think before Recommendation: Autonomous Reasoning-enhanced Recommender](http://arxiv.org/pdf/2510.23077v1)

Authors: Xiaoyu Kong, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Jiancan Wu, Xiang Wang

The core task of recommender systems is to learn user preferences from
historical user-item interactions. With the rapid development of large language
models (LLMs), recent research has explored leveraging the reasoning
capabilities of LLMs to enhance rating prediction tasks. However, existing
distillation-based methods suffer from limitations such as the teacher model's
insufficient recommendation capability, costly and static supervision, and
superficial transfer of reasoning ability. To address these issues, this paper
proposes RecZero, a reinforcement learning (RL)-based recommendation paradigm
that abandons the traditional multi-model and multi-stage distillation
approach. Instead, RecZero trains a single LLM through pure RL to autonomously
develop reasoning capabilities for rating prediction. RecZero consists of two
key components: (1) "Think-before-Recommendation" prompt construction, which
employs a structured reasoning template to guide the model in step-wise
analysis of user interests, item features, and user-item compatibility; and (2)
rule-based reward modeling, which adopts group relative policy optimization
(GRPO) to compute rewards for reasoning trajectories and optimize the LLM.
Additionally, the paper explores a hybrid paradigm, RecOne, which combines
supervised fine-tuning with RL, initializing the model with cold-start
reasoning samples and further optimizing it with RL. Experimental results
demonstrate that RecZero and RecOne significantly outperform existing baseline
methods on multiple benchmark datasets, validating the superiority of the RL
paradigm in achieving autonomous reasoning-enhanced recommender systems.

### 7. [Accurate and Scalable Multimodal Pathology Retrieval via Attentive Vision-Language Alignment](http://arxiv.org/pdf/2510.23224v1)

Authors: Hongyi Wang, Zhengjie Zhu, Jiabo Ma, Fang Wang, Yue Shi, Bo Luo, Jili Wang, Qiuyu Cai, Xiuming Zhang, Yen-Wei Chen, Lanfen Lin, Hao Chen

The rapid digitization of histopathology slides has opened up new
possibilities for computational tools in clinical and research workflows. Among
these, content-based slide retrieval stands out, enabling pathologists to
identify morphologically and semantically similar cases, thereby supporting
precise diagnoses, enhancing consistency across observers, and assisting
example-based education. However, effective retrieval of whole slide images
(WSIs) remains challenging due to their gigapixel scale and the difficulty of
capturing subtle semantic differences amid abundant irrelevant content. To
overcome these challenges, we present PathSearch, a retrieval framework that
unifies fine-grained attentive mosaic representations with global-wise slide
embeddings aligned through vision-language contrastive learning. Trained on a
corpus of 6,926 slide-report pairs, PathSearch captures both fine-grained
morphological cues and high-level semantic patterns to enable accurate and
flexible retrieval. The framework supports two key functionalities: (1)
mosaic-based image-to-image retrieval, ensuring accurate and efficient slide
research; and (2) multi-modal retrieval, where text queries can directly
retrieve relevant slides. PathSearch was rigorously evaluated on four public
pathology datasets and three in-house cohorts, covering tasks including
anatomical site retrieval, tumor subtyping, tumor vs. non-tumor discrimination,
and grading across diverse organs such as breast, lung, kidney, liver, and
stomach. External results show that PathSearch outperforms traditional
image-to-image retrieval frameworks. A multi-center reader study further
demonstrates that PathSearch improves diagnostic accuracy, boosts confidence,
and enhances inter-observer agreement among pathologists in real clinical
scenarios. These results establish PathSearch as a scalable and generalizable
retrieval solution for digital pathology.

### 8. [LimRank: Less is More for Reasoning-Intensive Information Reranking](http://arxiv.org/pdf/2510.23544v1)

Authors: Tingyu Song, Yilun Zhao, Siyue Zhang, Chen Zhao, Arman Cohan

Existing approaches typically rely on large-scale fine-tuning to adapt LLMs
for information reranking tasks, which is computationally expensive. In this
work, we demonstrate that modern LLMs can be effectively adapted using only
minimal, high-quality supervision. To enable this, we design
LIMRANK-SYNTHESIZER, a reusable and open-source pipeline for generating
diverse, challenging, and realistic reranking examples. Using this synthetic
data, we fine-tune our reranker model, LIMRANK. We evaluate LIMRANK on two
challenging benchmarks, i.e., BRIGHT for reasoning-intensive retrieval and
FollowIR for instruction-following retrieval. Our experiments demonstrate that
LIMRANK achieves competitive performance, while being trained on less than 5%
of the data typically used in prior work. Further ablation studies demonstrate
the effectiveness of LIMRANK-SYNTHESIZER and the strong generalization
capabilities of LIMRANK across downstream tasks, including scientific
literature search and retrieval-augmented generation for knowledge-intensive
problem solving.

### 9. [Leveraging Hierarchical Organization for Medical Multi-document Summarization](http://arxiv.org/pdf/2510.23104v1)

Authors: Yi-Li Hsu, Katelyn X. Mei, Lucy Lu Wang

Medical multi-document summarization (MDS) is a complex task that requires
effectively managing cross-document relationships. This paper investigates
whether incorporating hierarchical structures in the inputs of MDS can improve
a model's ability to organize and contextualize information across documents
compared to traditional flat summarization methods. We investigate two ways of
incorporating hierarchical organization across three large language models
(LLMs), and conduct comprehensive evaluations of the resulting summaries using
automated metrics, model-based metrics, and domain expert evaluation of
preference, understandability, clarity, complexity, relevance, coverage,
factuality, and coherence. Our results show that human experts prefer
model-generated summaries over human-written summaries. Hierarchical approaches
generally preserve factuality, coverage, and coherence of information, while
also increasing human preference for summaries. Additionally, we examine
whether simulated judgments from GPT-4 align with human judgments, finding
higher agreement along more objective evaluation facets. Our findings
demonstrate that hierarchical structures can improve the clarity of medical
summaries generated by models while maintaining content coverage, providing a
practical way to improve human preference for generated summaries.

### Machine Learning

### 1. [Limits of Generative Pre-Training in Structured EMR Trajectories with Irregular Sampling](http://arxiv.org/pdf/2510.22878v1)

Authors: Nicholas I-Hsien Kuo, Blanca Gallego, Louisa Jorm

Foundation models refer to architectures trained on vast datasets using
autoregressive pre-training from natural language processing to capture
intricate patterns and motifs. They were originally developed to transfer such
learned knowledge to downstream predictive tasks. Recently, however, some
studies repurpose these learned representations for phenotype discovery without
rigorous validation, risking superficially realistic but clinically incoherent
embeddings. To test this mismatch, we trained two autoregressive models -- a
sequence-to-sequence LSTM and a reduced Transformer -- on longitudinal ART for
HIV and Acute Hypotension datasets. Controlled irregularity was added during
training via random inter-visit gaps, while test sequences stayed complete.
Patient-trajectory synthesis evaluated distributional and correlational
fidelity. Both reproduced feature distributions but failed to preserve
cross-feature structure -- showing that generative pre-training yields local
realism but limited clinical coherence. These results highlight the need for
domain-specific evaluation and support trajectory synthesis as a practical
probe before fine-tuning or deployment.

### 2. [AI based signage classification for linguistic landscape studies](http://arxiv.org/pdf/2510.22885v1)

Authors: Yuqin Jiang, Song Jiang, Jacob Algrim, Trevor Harms, Maxwell Koenen, Xinya Lan, Xingyu Li, Chun-Han Lin, Jia Liu, Jiayang Sun, Henry Zenger

Linguistic Landscape (LL) research traditionally relies on manual photography
and annotation of public signages to examine distribution of languages in urban
space. While such methods yield valuable findings, the process is
time-consuming and difficult for large study areas. This study explores the use
of AI powered language detection method to automate LL analysis. Using Honolulu
Chinatown as a case study, we constructed a georeferenced photo dataset of
1,449 images collected by researchers and applied AI for optical character
recognition (OCR) and language classification. We also conducted manual
validations for accuracy checking. This model achieved an overall accuracy of
79%. Five recurring types of mislabeling were identified, including distortion,
reflection, degraded surface, graffiti, and hallucination. The analysis also
reveals that the AI model treats all regions of an image equally, detecting
peripheral or background texts that human interpreters typically ignore.
Despite these limitations, the results demonstrate the potential of integrating
AI-assisted workflows into LL research to reduce such time-consuming processes.
However, due to all the limitations and mis-labels, we recognize that AI cannot
be fully trusted during this process. This paper encourages a hybrid approach
combining AI automation with human validation for a more reliable and efficient
workflow.

### 3. [Transforming volcanic monitoring: A dataset and benchmark for onboard volcano activity detection](http://arxiv.org/pdf/2510.22889v1)

Authors: Darshana Priyasad, Tharindu Fernando, Maryam Haghighat, Harshala Gammulle, Clinton Fookes

Natural disasters, such as volcanic eruptions, pose significant challenges to
daily life and incur considerable global economic losses. The emergence of
next-generation small-satellites, capable of constellation-based operations,
offers unparalleled opportunities for near-real-time monitoring and onboard
processing of such events. However, a major bottleneck remains the lack of
extensive annotated datasets capturing volcanic activity, which hinders the
development of robust detection systems. This paper introduces a novel dataset
explicitly designed for volcanic activity and eruption detection, encompassing
diverse volcanoes worldwide. The dataset provides binary annotations to
identify volcanic anomalies or non-anomalies, covering phenomena such as
temperature anomalies, eruptions, and volcanic ash emissions. These annotations
offer a foundational resource for developing and evaluating detection models,
addressing a critical gap in volcanic monitoring research. Additionally, we
present comprehensive benchmarks using state-of-the-art models to establish
baselines for future studies. Furthermore, we explore the potential for
deploying these models onboard next-generation satellites. Using the Intel
Movidius Myriad X VPU as a testbed, we demonstrate the feasibility of volcanic
activity detection directly onboard. This capability significantly reduces
latency and enhances response times, paving the way for advanced early warning
systems. This paves the way for innovative solutions in volcanic disaster
management, encouraging further exploration and refinement of onboard
monitoring technologies.

### 4. [Charting the Design Space of Neural Graph Representations for Subgraph Matching](http://arxiv.org/pdf/2510.22897v1)

Authors: Vaibhav Raj, Indradyumna Roy, Ashwin Ramachandran, Soumen Chakrabarti, Abir De

Subgraph matching is vital in knowledge graph (KG) question answering,
molecule design, scene graph, code and circuit search, etc. Neural methods have
shown promising results for subgraph matching. Our study of recent systems
suggests refactoring them into a unified design space for graph matching
networks. Existing methods occupy only a few isolated patches in this space,
which remains largely uncharted. We undertake the first comprehensive
exploration of this space, featuring such axes as attention-based vs. soft
permutation-based interaction between query and corpus graphs, aligning nodes
vs. edges, and the form of the final scoring network that integrates neural
representations of the graphs. Our extensive experiments reveal that judicious
and hitherto-unexplored combinations of choices in this space lead to large
performance benefits. Beyond better performance, our study uncovers valuable
insights and establishes general design principles for neural graph
representation and interaction, which may be of wider interest.

### 5. [Simple Denoising Diffusion Language Models](http://arxiv.org/pdf/2510.22926v1)

Authors: Huaisheng Zhu, Zhengyu Chen, Shijie Zhou, Zhihui Xie, Yige Yuan, Zhimeng Guo, Siyuan Xu, Hangfan Zhang, Vasant Honavar, Teng Xiao

Diffusion models have recently been extended to language generation through
Masked Diffusion Language Models (MDLMs), which achieve performance competitive
with strong autoregressive models. However, MDLMs tend to degrade in the
few-step regime and cannot directly adopt existing few-step distillation
methods designed for continuous diffusion models, as they lack the intrinsic
property of mapping from noise to data. Recent Uniform-state Diffusion Models
(USDMs), initialized from a uniform prior, alleviate some limitations but still
suffer from complex loss formulations that hinder scalability. In this work, we
propose a simplified denoising-based loss for USDMs that optimizes only
noise-replaced tokens, stabilizing training and matching ELBO-level
performance. Furthermore, by framing denoising as self-supervised learning, we
introduce a simple modification to our denoising loss with contrastive-inspired
negative gradients, which is practical and yield additional improvements in
generation quality.

### 6. [Diffuse to Detect: A Generalizable Framework for Anomaly Detection with Diffusion Models Applications to UAVs and Beyond](http://arxiv.org/pdf/2510.22928v1)

Authors: Mingze Gong, Juan Du, Jianbang You

Anomaly detection in complex, high-dimensional data, such as UAV sensor
readings, is essential for operational safety but challenging for existing
methods due to their limited sensitivity, scalability, and inability to capture
intricate dependencies. We propose the Diffuse to Detect (DTD) framework, a
novel approach that innovatively adapts diffusion models for anomaly detection,
diverging from their conventional use in generative tasks with high inference
time. By comparison, DTD employs a single-step diffusion process to predict
noise patterns, enabling rapid and precise identification of anomalies without
reconstruction errors. This approach is grounded in robust theoretical
foundations that link noise prediction to the data distribution's score
function, ensuring reliable deviation detection. By integrating Graph Neural
Networks to model sensor relationships as dynamic graphs, DTD effectively
captures spatial (inter-sensor) and temporal anomalies. Its two-branch
architecture, with parametric neural network-based energy scoring for
scalability and nonparametric statistical methods for interpretability,
provides flexible trade-offs between computational efficiency and transparency.
Extensive evaluations on UAV sensor data, multivariate time series, and images
demonstrate DTD's superior performance over existing methods, underscoring its
generality across diverse data modalities. This versatility, combined with its
adaptability, positions DTD as a transformative solution for safety-critical
applications, including industrial monitoring and beyond.

### 7. [RL-AUX: Reinforcement Learning for Auxiliary Task Generation](http://arxiv.org/pdf/2510.22940v1)

Authors: Judah Goldfeder, Matthew So, Hod Lipson

Auxiliary Learning (AL) is a special case of Multi-task Learning (MTL) in
which a network trains on auxiliary tasks to improve performance on its main
task. This technique is used to improve generalization and, ultimately,
performance on the network's main task. AL has been demonstrated to improve
performance across multiple domains, including navigation, image
classification, and natural language processing. One weakness of AL is the need
for labeled auxiliary tasks, which can require human effort and domain
expertise to generate. Meta Learning techniques have been used to solve this
issue by learning an additional auxiliary task generation network that can
create helpful tasks for the primary network. The most prominent techniques
rely on Bi-Level Optimization, which incurs computational cost and increased
code complexity. To avoid the need for Bi-Level Optimization, we present an
RL-based approach to dynamically create auxiliary tasks. In this framework, an
RL agent is tasked with selecting auxiliary labels for every data point in a
training set. The agent is rewarded when their selection improves the
performance on the primary task. We also experiment with learning optimal
strategies for weighing the auxiliary loss per data point. On the 20-Superclass
CIFAR100 problem, our RL approach outperforms human-labeled auxiliary tasks and
performs as well as a prominent Bi-Level Optimization technique. Our weight
learning approaches significantly outperform all of these benchmarks. For
example, a Weight-Aware RL-based approach helps the VGG16 architecture achieve
80.9% test accuracy while the human-labeled auxiliary task setup achieved
75.53%. The goal of this work is to (1) prove that RL is a viable approach to
dynamically generate auxiliary tasks and (2) demonstrate that per-sample
auxiliary task weights can be learned alongside the auxiliary task labels and
can achieve strong results.

### 8. [Hazard-Responsive Digital Twin for Climate-Driven Urban Resilience and Equity](http://arxiv.org/pdf/2510.22941v1)

Authors: Zhenglai Shen, Hongyu Zhou

Compounding climate hazards, such as wildfire-induced outages and urban
heatwaves, challenge the stability and equity of cities. We present a
Hazard-Responsive Digital Twin (H-RDT) that combines physics-informed neural
network modeling, multimodal data fusion, and equity-aware risk analytics for
urban-scale response. In a synthetic district with diverse building archetypes
and populations, a simulated wildfire-outage-heatwave cascade shows that H-RDT
maintains stable indoor temperature predictions (approximately 31 to 33 C)
under partial sensor loss, reproducing outage-driven surges and recovery. The
reinforcement learning based fusion module adaptively reweights IoT, UAV, and
satellite inputs to sustain spatiotemporal coverage, while the equity-adjusted
mapping isolates high-vulnerability clusters (schools, clinics, low-income
housing). Prospective interventions, such as preemptive cooling-center
activation and microgrid sharing, reduce population-weighted thermal risk by 11
to 13 percent, shrink the 95th-percentile (tail) risk by 7 to 17 percent, and
cut overheating hours by up to 9 percent. Beyond the synthetic demonstration,
the framework establishes a transferable foundation for real-city
implementation, linking physical hazard modeling with social equity and
decision intelligence. The H-RDT advances digital urban resilience toward
adaptive, learning-based, and equity-centered decision support for climate
adaptation.

### 9. [SARNet: A Spike-Aware consecutive validation Framework for Accurate Remaining Useful Life Prediction](http://arxiv.org/pdf/2510.22955v1)

Authors: Junhao Fan, Wenrui Liang, Wei-Qiang Zhang

Accurate prediction of remaining useful life (RUL) is essential to enhance
system reliability and reduce maintenance risk. Yet many strong contemporary
models are fragile around fault onset and opaque to engineers: short,
high-energy spikes are smoothed away or misread, fixed thresholds blunt
sensitivity, and physics-based explanations are scarce. To remedy this, we
introduce SARNet (Spike-Aware Consecutive Validation Framework), which builds
on a Modern Temporal Convolutional Network (ModernTCN) and adds spike-aware
detection to provide physics-informed interpretability. ModernTCN forecasts
degradation-sensitive indicators; an adaptive consecutive threshold validates
true spikes while suppressing noise. Failure-prone segments then receive
targeted feature engineering (spectral slopes, statistical derivatives, energy
ratios), and the final RUL is produced by a stacked RF--LGBM regressor. Across
benchmark-ported datasets under an event-triggered protocol, SARNet
consistently lowers error compared to recent baselines (RMSE 0.0365, MAE
0.0204) while remaining lightweight, robust, and easy to deploy.

### 10. [QoSGMAA: A Robust Multi-Order Graph Attention and Adversarial Framework for Sparse QoS Prediction](http://arxiv.org/pdf/2510.22982v1)

Authors: Guanchen Du, Jianlong Xu, Mingtong Li, Ruiqi Wang, Qianqing Guo, Caiyi Chen, Qingcao Dai, Yuxiang Zeng

With the rapid advancement of internet technologies, network services have
become critical for delivering diverse and reliable applications to users.
However, the exponential growth in the number of available services has
resulted in many similar offerings, posing significant challenges in selecting
optimal services. Predicting Quality of Service (QoS) accurately thus becomes a
fundamental prerequisite for ensuring reliability and user satisfaction.
However, existing QoS prediction methods often fail to capture rich contextual
information and exhibit poor performance under extreme data sparsity and
structural noise. To bridge this gap, we propose a novel architecture, QoSMGAA,
specifically designed to enhance prediction accuracy in complex and noisy
network service environments. QoSMGAA integrates a multi-order attention
mechanism to aggregate extensive contextual data and predict missing QoS values
effectively. Additionally, our method incorporates adversarial neural networks
to perform autoregressive supervised learning based on transformed interaction
matrices. To capture complex, higher-order interactions among users and
services, we employ a discrete sampling technique leveraging the Gumbel-Softmax
method to generate informative negative samples. Comprehensive experimental
validation conducted on large-scale real-world datasets demonstrates that our
proposed model significantly outperforms existing baseline methods,
highlighting its strong potential for practical deployment in service selection
and recommendation scenarios.

### Neural and Evolutionary Computing

### 1. [One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons](http://arxiv.org/pdf/2510.23383v1)

Authors: Qiuyang Chen, Huiqi Yang, Qingyan Meng, Zhengyu Ma

Spiking Neural Networks (SNNs) are gaining attention as energy-efficient
alternatives to Artificial Neural Networks (ANNs), especially in
resource-constrained settings. While ANN-to-SNN conversion (ANN2SNN) achieves
high accuracy without end-to-end SNN training, existing methods rely on large
time steps, leading to high inference latency and computational cost. In this
paper, we propose a theoretical and practical framework for single-timestep
ANN2SNN. We establish the Temporal-to-Spatial Equivalence Theory, proving that
multi-timestep integrate-and-fire (IF) neurons can be equivalently replaced by
single-timestep multi-threshold neurons (MTN). Based on this theory, we
introduce the Scale-and-Fire Neuron (SFN), which enables effective
single-timestep ($T=1$) spiking through adaptive scaling and firing.
Furthermore, we develop the SFN-based Spiking Transformer (SFormer), a
specialized instantiation of SFN within Transformer architectures, where spike
patterns are aligned with attention distributions to mitigate the
computational, energy, and hardware overhead of the multi-threshold design.
Extensive experiments on image classification, object detection, and instance
segmentation demonstrate that our method achieves state-of-the-art performance
under single-timestep inference. Notably, we achieve 88.8% top-1 accuracy on
ImageNet-1K at $T=1$, surpassing existing conversion methods.

### 2. [Multi-Task Surrogate-Assisted Search with Bayesian Competitive Knowledge Transfer for Expensive Optimization](http://arxiv.org/pdf/2510.23407v1)

Authors: Yi Lu, Xiaoming Xue, Kai Zhang, Liming Zhang, Guodong Chen, Chenming Cao, Piyang Liu, Kay Chen Tan

Expensive optimization problems (EOPs) present significant challenges for
traditional evolutionary optimization due to their limited evaluation calls.
Although surrogate-assisted search (SAS) has become a popular paradigm for
addressing EOPs, it still suffers from the cold-start issue. In response to
this challenge, knowledge transfer has been gaining popularity for its ability
to leverage search experience from potentially related instances, ultimately
facilitating head-start optimization for more efficient decision-making.
However, the curse of negative transfer persists when applying knowledge
transfer to EOPs, primarily due to the inherent limitations of existing methods
in assessing knowledge transferability. On the one hand, a priori
transferability assessment criteria are intrinsically inaccurate due to their
imprecise understandings. On the other hand, a posteriori methods often
necessitate sufficient observations to make correct inferences, rendering them
inefficient when applied to EOPs. Considering the above, this paper introduces
a Bayesian competitive knowledge transfer (BCKT) method developed to improve
multi-task SAS (MSAS) when addressing multiple EOPs simultaneously.
Specifically, the transferability of knowledge is estimated from a Bayesian
perspective that accommodates both prior beliefs and empirical evidence,
enabling accurate competition between inner-task and inter-task solutions,
ultimately leading to the adaptive use of promising solutions while effectively
suppressing inferior ones. The effectiveness of our method in boosting various
SAS algorithms for both multi-task and many-task problems is empirically
validated, complemented by comparative studies that demonstrate its superiority
over peer algorithms and its applicability to real-world scenarios. The source
code of our method is available at https://github.com/XmingHsueh/MSAS-BCKT.

### 3. [Equivariant Neural Networks for General Linear Symmetries on Lie Algebras](http://arxiv.org/pdf/2510.22984v1)

Authors: Chankyo Kim, Sicheng Zhao, Minghan Zhu, Tzu-Yuan Lin, Maani Ghaffari

Encoding symmetries is a powerful inductive bias for improving the
generalization of deep neural networks. However, most existing equivariant
models are limited to simple symmetries like rotations, failing to address the
broader class of general linear transformations, GL(n), that appear in many
scientific domains. We introduce Reductive Lie Neurons (ReLNs), a novel neural
network architecture exactly equivariant to these general linear symmetries.
ReLNs are designed to operate directly on a wide range of structured inputs,
including general n-by-n matrices. ReLNs introduce a novel adjoint-invariant
bilinear layer to achieve stable equivariance for both Lie-algebraic features
and matrix-valued inputs, without requiring redesign for each subgroup. This
architecture overcomes the limitations of prior equivariant networks that only
apply to compact groups or simple vector data. We validate ReLNs' versatility
across a spectrum of tasks: they outperform existing methods on algebraic
benchmarks with sl(3) and sp(4) symmetries and achieve competitive results on a
Lorentz-equivariant particle physics task. In 3D drone state estimation with
geometric uncertainty, ReLNs jointly process velocities and covariances,
yielding significant improvements in trajectory accuracy. ReLNs provide a
practical and general framework for learning with broad linear group symmetries
on Lie algebras and matrix-valued data. Project page:
https://reductive-lie-neuron.github.io/

### 4. [Sequential Multi-Agent Dynamic Algorithm Configuration](http://arxiv.org/pdf/2510.23535v1)

Authors: Chen Lu, Ke Xue, Lei Yuan, Yao Wang, Yaoyuan Wang, Sheng Fu, Chao Qian

Dynamic algorithm configuration (DAC) is a recent trend in automated machine
learning, which can dynamically adjust the algorithm's configuration during the
execution process and relieve users from tedious trial-and-error tuning tasks.
Recently, multi-agent reinforcement learning (MARL) approaches have improved
the configuration of multiple heterogeneous hyperparameters, making various
parameter configurations for complex algorithms possible. However, many complex
algorithms have inherent inter-dependencies among multiple parameters (e.g.,
determining the operator type first and then the operator's parameter), which
are, however, not considered in previous approaches, thus leading to
sub-optimal results. In this paper, we propose the sequential multi-agent DAC
(Seq-MADAC) framework to address this issue by considering the inherent
inter-dependencies of multiple parameters. Specifically, we propose a
sequential advantage decomposition network, which can leverage action-order
information through sequential advantage decomposition. Experiments from
synthetic functions to the configuration of multi-objective optimization
algorithms demonstrate Seq-MADAC's superior performance over state-of-the-art
MARL methods and show strong generalization across problem classes. Seq-MADAC
establishes a new paradigm for the widespread dependency-aware automated
algorithm configuration. Our code is available at
https://github.com/lamda-bbo/seq-madac.

### 5. [Symbolic Neural Generation with Applications to Lead Discovery in Drug Design](http://arxiv.org/pdf/2510.23379v1)

Authors: Ashwin Srinivasan, A Baskar, Tirtharaj Dash, Michael Bain, Sanjay Kumar Dey, Mainak Banerjee

We investigate a relatively underexplored class of hybrid neurosymbolic
models integrating symbolic learning with neural reasoning to construct data
generators meeting formal correctness criteria. In \textit{Symbolic Neural
Generators} (SNGs), symbolic learners examine logical specifications of
feasible data from a small set of instances -- sometimes just one. Each
specification in turn constrains the conditional information supplied to a
neural-based generator, which rejects any instance violating the symbolic
specification. Like other neurosymbolic approaches, SNG exploits the
complementary strengths of symbolic and neural methods. The outcome of an SNG
is a triple $(H, X, W)$, where $H$ is a symbolic description of feasible
instances constructed from data, $X$ a set of generated new instances that
satisfy the description, and $W$ an associated weight. We introduce a semantics
for such systems, based on the construction of appropriate \textit{base} and
\textit{fibre} partially-ordered sets combined into an overall partial order,
and outline a probabilistic extension relevant to practical applications. In
this extension, SNGs result from searching over a weighted partial ordering. We
implement an SNG combining a restricted form of Inductive Logic Programming
(ILP) with a large language model (LLM) and evaluate it on early-stage drug
design. Our main interest is the description and the set of potential inhibitor
molecules generated by the SNG. On benchmark problems -- where drug targets are
well understood -- SNG performance is statistically comparable to
state-of-the-art methods. On exploratory problems with poorly understood
targets, generated molecules exhibit binding affinities on par with leading
clinical candidates. Experts further find the symbolic specifications useful as
preliminary filters, with several generated molecules identified as viable for
synthesis and wet-lab testing.

### 6. [BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement](http://arxiv.org/pdf/2510.23472v1)

Authors: Ke Xue, Ruo-Tong Chen, Rong-Xi Tan, Xi Lin, Yunqi Shi, Siyuan Xu, Mingxuan Yuan, Chao Qian

Chip placement is a vital stage in modern chip design as it has a substantial
impact on the subsequent processes and the overall quality of the final chip.
The use of black-box optimization (BBO) for chip placement has a history of
several decades. However, early efforts were limited by immature problem
formulations and inefficient algorithm designs. Recent progress has shown the
effectiveness and efficiency of BBO for chip placement, proving its potential
to achieve state-of-the-art results. Despite these advancements, the field
lacks a unified, BBO-specific benchmark for thoroughly assessing various
problem formulations and BBO algorithms. To fill this gap, we propose
BBOPlace-Bench, the first benchmark designed specifically for evaluating and
developing BBO algorithms for chip placement tasks. It integrates three problem
formulations of BBO for chip placement, and offers a modular, decoupled, and
flexible framework that enables users to seamlessly implement, test, and
compare their own algorithms. BBOPlace-Bench integrates a wide variety of
existing BBO algorithms, including simulated annealing (SA), evolutionary
algorithms (EAs), and Bayesian optimization (BO). Experimental results show
that the problem formulations of mask-guided optimization and hyperparameter
optimization exhibit superior performance than the sequence pair problem
formulation, while EAs demonstrate better overall performance than SA and BO,
especially in high-dimensional search spaces, and also achieve state-of-the-art
performance compared to the mainstream chip placement methods. BBOPlace-Bench
not only facilitates the development of efficient BBO-driven solutions for chip
placement but also broadens the practical application scenarios (which are
urgently needed) for the BBO community. The code of BBOPlace-Bench is available
at https://github.com/lamda-bbo/BBOPlace-Bench.

### Networking and Internet Architecture

### 1. [Exploring LR-FHSS Modulation for Enhanced IoT Connectivity: A Measurement Campaign](http://arxiv.org/pdf/2510.23152v1)

Authors: Alexis Delplace, Samer Lahoud, Kinda Khawam

This paper presents the first comprehensive real-world measurement campaign
comparing LR-FHSS and LoRa modulations within LoRaWAN networks in urban
environments. Conducted in Halifax, Canada, the campaign used a LoRaWAN
platform capable of operating both modulations in the FCC-regulated US915 band.
Real-world measurements are crucial for capturing the effects of urban topology
and signal propagation challenges, which are difficult to fully replicate in
simulations. Results show that LR-FHSS can achieve up to a 20% improvement in
Packet Reception Rate (PRR) over traditional LoRa in dense urban areas.
Additionally, the study investigated path loss and Received Signal Strength
Indicator (RSSI), finding that LR-FHSS achieved a minimum RSSI of -138 dBm
compared to LoRa's -120 dBm. The findings demonstrate that the introduction of
LR-FHSS enhances communication robustness and reliability under regulatory
limitations and suggest promising applications in LoRaWAN networks.

### 2. [Trajectory-Aware Air-to-Ground Channel Characterization for Low-Altitude UAVs Using MaMIMO Measurements](http://arxiv.org/pdf/2510.23465v1)

Authors: Abdul Saboor, Zhuangzhuang Cui, Achiel Colpaert, Evgenii Vinogradov, Wout Joseph, Sofie Pollin

This paper presents a comprehensive measurement-based trajectory-aware
characterization of low-altitude Air-to-Ground (A2G) channels in a suburban
environment. A 64-element Massive Multi-Input Multi-Output (MaMIMO) array was
used to capture channels for three trajectories of an Uncrewed Aerial Vehicle
(UAV), including two horizontal zig-zag flights at fixed altitudes and one
vertical ascent, chosen to emulate AUE operations and to induce controlled
azimuth and elevation sweeps for analyzing geometry-dependent propagation
dynamics. We examine large-scale power variations and their correlation with
geometric features, such as elevation, azimuth, and 3D distance, followed by an
analysis of fading behavior through distribution fitting and Rician K-factor
estimation. Furthermore, temporal non-stationarity is quantified using the
Correlation Matrix Distance (CMD), and angular stationarity spans are utilized
to demonstrate how channel characteristics change with the movement of the UAV.
We also analyze Spectral Efficiency (SE) in relation to K-factor and Root Mean
Square (RMS) delay spread, highlighting their combined influence on link
performance. The results show that the elevation angle is the strongest
predictor of the received power, with a correlation of more than 0.77 for each
trajectory, while the Nakagami model best fits the small-scale fading. The
K-factor increases from approximately 5 dB at low altitudes to over 15 dB at
higher elevations, indicating stronger LoS dominance. Non-stationarity patterns
are highly trajectory- and geometry-dependent, with azimuth most affected in
horizontal flights and elevation during vertical flight. These findings offer
valuable insights for modeling and improving UAV communication channels in 6G
Non-Terrestrial Networks (NTNs).

### 3. [How to build a sovereign network? -- A proposal to measure network sovereignty](http://arxiv.org/pdf/2510.23510v1)

Authors: Shakthivelu Janardhanan, Ritanshi Agarwal, Wolfgang Kellerer, Carmen Mas-Machuca

Network sovereignty is a network operator's ability to reduce the dependency
on component manufacturers to minimize the impact of manufacturer failures.
Network operators now face new design challenges to increase network
sovereignty and avoid vendor lock-in problems because a high dependency on a
manufacturer corresponds to low survivability if that manufacturer is
unavailable. The main contribution of this work is the proposal of a novel
metric to measure network sovereignty, the Cut Set Coloring (CSC) score. Based
on the CSC core metric CSC-ILP, our Integer Linear Program formulation is
presented to maximize network sovereignty. We compare CSC-ILP's performance
with state of the art manufacturer assignment strategies.

### 4. [PASS-Enhanced MEC: Joint Optimization of Task Offloading and Uplink PASS Beamforming](http://arxiv.org/pdf/2510.22948v1)

Authors: Zhaoming Hu, Ruikang Zhong, Xidong Mu, Dengao Li, Yuanwei Liu

A pinching-antenna system (PASS)-enhanced mobile edge computing (MEC)
architecture is investigated to improve the task offloading efficiency and
latency performance in dynamic wireless environments. By leveraging dielectric
waveguides and flexibly adjustable pinching antennas, PASS establishes
short-distance line-of-sight (LoS) links while effectively mitigating the
significant path loss and potential signal blockage, making it a promising
solution for high-frequency MEC systems. We formulate a network latency
minimization problem to joint optimize uplink PASS beamforming and task
offloading. The resulting problem is modeled as a Markov decision process (MDP)
and solved via the deep reinforcement learning (DRL) method. To address the
instability introduced by the $\max$ operator in the objective function, we
propose a load balancing-aware proximal policy optimization (LBPPO) algorithm.
LBPPO incorporates both node-level and waveguide-level load balancing
information into the policy design, maintaining computational and transmission
delay equilibrium, respectively. Simulation results demonstrate that the
proposed PASS-enhanced MEC with adaptive uplink PASS beamforming exhibit
stronger convergence capability than fixed-PA baselines and conventional
MIMO-assisted MEC, especially in scenarios with a large number of UEs or high
transmit power.

### Robotics

### 1. [ManiDP: Manipulability-Aware Diffusion Policy for Posture-Dependent Bimanual Manipulation](http://arxiv.org/pdf/2510.23016v1)

Authors: Zhuo Li, Junjia Liu, Dianxi Li, Tao Teng, Miao Li, Sylvain Calinon, Darwin Caldwell, Fei Chen

Recent work has demonstrated the potential of diffusion models in robot
bimanual skill learning. However, existing methods ignore the learning of
posture-dependent task features, which are crucial for adapting dual-arm
configurations to meet specific force and velocity requirements in dexterous
bimanual manipulation. To address this limitation, we propose
Manipulability-Aware Diffusion Policy (ManiDP), a novel imitation learning
method that not only generates plausible bimanual trajectories, but also
optimizes dual-arm configurations to better satisfy posture-dependent task
requirements. ManiDP achieves this by extracting bimanual manipulability from
expert demonstrations and encoding the encapsulated posture features using
Riemannian-based probabilistic models. These encoded posture features are then
incorporated into a conditional diffusion process to guide the generation of
task-compatible bimanual motion sequences. We evaluate ManiDP on six real-world
bimanual tasks, where the experimental results demonstrate a 39.33$\%$ increase
in average manipulation success rate and a 0.45 improvement in task
compatibility compared to baseline methods. This work highlights the importance
of integrating posture-relevant robotic priors into bimanual skill diffusion to
enable human-like adaptability and dexterity.

### 2. [Awakening Facial Emotional Expressions in Human-Robot](http://arxiv.org/pdf/2510.23059v1)

Authors: Yongtong Zhu, Lei Li, Iggy Qian, WenBin Zhou, Ye Yuan, Qingdu Li, Na Liu, Jianwei Zhang

The facial expression generation capability of humanoid social robots is
critical for achieving natural and human-like interactions, playing a vital
role in enhancing the fluidity of human-robot interactions and the accuracy of
emotional expression. Currently, facial expression generation in humanoid
social robots still relies on pre-programmed behavioral patterns, which are
manually coded at high human and time costs. To enable humanoid robots to
autonomously acquire generalized expressive capabilities, they need to develop
the ability to learn human-like expressions through self-training. To address
this challenge, we have designed a highly biomimetic robotic face with
physical-electronic animated facial units and developed an end-to-end learning
framework based on KAN (Kolmogorov-Arnold Network) and attention mechanisms.
Unlike previous humanoid social robots, we have also meticulously designed an
automated data collection system based on expert strategies of facial motion
primitives to construct the dataset. Notably, to the best of our knowledge,
this is the first open-source facial dataset for humanoid social robots.
Comprehensive evaluations indicate that our approach achieves accurate and
diverse facial mimicry across different test subjects.

### 3. [Breaking the Circle: An Autonomous Control-Switching Strategy for Stable Orographic Soaring in MAVs](http://arxiv.org/pdf/2510.23084v1)

Authors: Sunyou Hwang, Christophe De Wagter, Bart Remes, Guido de Croon

Orographic soaring can significantly extend the endurance of micro aerial
vehicles (MAVs), but circling behavior, arising from control conflicts between
the longitudinal and vertical axes, increases energy consumption and the risk
of divergence. We propose a control switching method, named SAOS: Switched
Control for Autonomous Orographic Soaring, which mitigates circling behavior by
selectively controlling either the horizontal or vertical axis, effectively
transforming the system from underactuated to fully actuated during soaring.
Additionally, the angle of attack is incorporated into the INDI controller to
improve force estimation. Simulations with randomized initial positions and
wind tunnel experiments on two MAVs demonstrate that the SAOS improves position
convergence, reduces throttle usage, and mitigates roll oscillations caused by
pitch-roll coupling. These improvements enhance energy efficiency and flight
stability in constrained soaring environments.

### 4. [An Automated Tape Laying System Employing a Uniaxial Force Control Device](http://arxiv.org/pdf/2510.23109v1)

Authors: Bernhard Rameder, Hubert Gattringer, Ronald Naderer, Andreas Mueller

This paper deals with the design of a cost effective automated tape laying
system (ATL system) with integrated uniaxial force control to ensure the
necessary compaction forces as well as with an accurate temperature control to
guarantee the used tape being melted appropriate. It is crucial to control the
substrate and the oncoming tape onto a specific temperature level to ensure an
optimal consolidation between the different layers of the product. Therefore,
it takes several process steps from the spooled tape on the coil until it is
finally tacked onto the desired mold. The different modules are divided into
the tape storage spool, a tape-guiding roller, a tape processing unit, a
heating zone and the consolidation unit. Moreover, a special robot control
concept for testing the ATL system is presented. In contrast to many other
systems, with this approach, the tape laying device is spatially fixed and the
shape is moved accordingly by the robot, which allows for handling of rather
compact and complex shapes. The functionality of the subsystems and the taping
process itself was finally approved in experimental results using a carbon
fiber reinforced HDPE tape.

### 5. [OmniDexGrasp: Generalizable Dexterous Grasping via Foundation Model and Force Feedback](http://arxiv.org/pdf/2510.23119v1)

Authors: Yi-Lin Wei, Zhexi Luo, Yuhao Lin, Mu Lin, Zhizhao Liang, Shuoyu Chen, Wei-Shi Zheng

Enabling robots to dexterously grasp and manipulate objects based on human
commands is a promising direction in robotics. However, existing approaches are
challenging to generalize across diverse objects or tasks due to the limited
scale of semantic dexterous grasp datasets. Foundation models offer a new way
to enhance generalization, yet directly leveraging them to generate feasible
robotic actions remains challenging due to the gap between abstract model
knowledge and physical robot execution. To address these challenges, we propose
OmniDexGrasp, a generalizable framework that achieves omni-capabilities in user
prompting, dexterous embodiment, and grasping tasks by combining foundation
models with the transfer and control strategies. OmniDexGrasp integrates three
key modules: (i) foundation models are used to enhance generalization by
generating human grasp images supporting omni-capability of user prompt and
task; (ii) a human-image-to-robot-action transfer strategy converts human
demonstrations into executable robot actions, enabling omni dexterous
embodiment; (iii) force-aware adaptive grasp strategy ensures robust and stable
grasp execution. Experiments in simulation and on real robots validate the
effectiveness of OmniDexGrasp on diverse user prompts, grasp task and dexterous
hands, and further results show its extensibility to dexterous manipulation
tasks.

### 6. [Reliable Robotic Task Execution in the Face of Anomalies](http://arxiv.org/pdf/2510.23121v1)

Authors: Bharath Santhanam, Alex Mitrevski, Santosh Thoduka, Sebastian Houben, Teena Hassan

Learned robot policies have consistently been shown to be versatile, but they
typically have no built-in mechanism for handling the complexity of open
environments, making them prone to execution failures; this implies that
deploying policies without the ability to recognise and react to failures may
lead to unreliable and unsafe robot behaviour. In this paper, we present a
framework that couples a learned policy with a method to detect visual
anomalies during policy deployment and to perform recovery behaviours when
necessary, thereby aiming to prevent failures. Specifically, we train an
anomaly detection model using data collected during nominal executions of a
trained policy. This model is then integrated into the online policy execution
process, so that deviations from the nominal execution can trigger a
three-level sequential recovery process that consists of (i) pausing the
execution temporarily, (ii) performing a local perturbation of the robot's
state, and (iii) resetting the robot to a safe state by sampling from a learned
execution success model. We verify our proposed method in two different
scenarios: (i) a door handle reaching task with a Kinova Gen3 arm using a
policy trained in simulation and transferred to the real robot, and (ii) an
object placing task with a UFactory xArm 6 using a general-purpose policy
model. Our results show that integrating policy execution with anomaly
detection and recovery increases the execution success rate in environments
with various anomalies, such as trajectory deviations and adversarial human
interventions.

### 7. [Workspace Registration and Collision Detection for Industrial Robotics Applications](http://arxiv.org/pdf/2510.23227v1)

Authors: Klaus Zauner, Josef El Dib, Hubert Gattringer, Andreas Mueller

Motion planning for robotic manipulators relies on precise knowledge of the
environment in order to be able to define restricted areas and to take
collision objects into account. To capture the workspace, point clouds of the
environment are acquired using various sensors. The collision objects are
identified by region growing segmentation and VCCS algorithm. Subsequently the
point clusters are approximated. The aim of the present paper is to compare
different sensors, to illustrate the process from detection to the finished
collision environment and to detect collisions between the robot and this
environment.

### 8. [Optimal Dimensioning of Elastic-Link Manipulators regarding Lifetime Estimation](http://arxiv.org/pdf/2510.23234v1)

Authors: Klaus Zauner, Hubert Gattringer, Andreas Mueller

Resourceful operation and design of robots is key for sustainable industrial
automation. This will be enabled by lightweight design along with time and
energy optimal control of robotic manipulators. Design and control of such
systems is intertwined as the control must take into account inherent
mechanical compliance while the design must accommodate the dynamic
requirements demanded by the control. As basis for such design optimization, a
method for estimating the lifetime of elastic link robotic manipulators is
presented. This is applied to the geometry optimization of flexible serial
manipulators performing pick-and-place operations, where the optimization
objective is a combination of overall weight and vibration amplitudes. The
lifetime estimation draws from a fatigue analysis combining the rainflow
counting algorithm and the method of critical cutting plane. Tresca hypothesis
is used to formulate an equivalent stress, and linear damage accumulation is
assumed. The final robot geometry is selected from a Pareto front as a tradeoff
of lifetime and vibration characteristic. The method is illustrated for a three
degrees of freedom articulated robotic manipulator.

### 9. [Precise Time Delay Measurement and Compensation for Tightly Coupled Underwater SINS/piUSBL Navigation](http://arxiv.org/pdf/2510.23286v1)

Authors: Jin Huang, Yingqiang Wang, Haoda Li, Zichen Liu, Zhikun Wang, Ying Chen

In multi-sensor systems, time synchronization between sensors is a
significant challenge, and this issue is particularly pronounced in underwater
integrated navigation systems incorporating acoustic positioning. Such systems
are highly susceptible to time delay, which can significantly degrade accuracy
when measurement and fusion moments are misaligned. To address this challenge,
this paper introduces a tightly coupled navigation framework that integrates a
passive inverted ultra-short baseline (piUSBL) acoustic positioning system, a
strapdown inertial navigation system (SINS), and a depth gauge under precise
time synchronization. The framework fuses azimuth and slant range from the
piUSBL with depth data, thereby avoiding poor vertical-angle observability in
planar arrays. A novel delay measurement strategy is introduced, combining
synchronized timing with acoustic signal processing, which redefines
delay-traditionally an unobservable error-into a quantifiable parameter,
enabling explicit estimation of both acoustic propagation and system processing
delays. Simulations and field experiments confirm the feasibility of the
proposed method, with delay-compensated navigation reducing RMSE by 40.45% and
maximum error by 32.55%. These findings show that precise delay measurement and
compensation not only enhance underwater navigation accuracy but also establish
a generalizable framework for acoustic positioning integration, offering
valuable insights into time alignment and data fusion in latency-sensitive
multi-sensor systems.

### 10. [Transferable Deep Reinforcement Learning for Cross-Domain Navigation: from Farmland to the Moon](http://arxiv.org/pdf/2510.23329v1)

Authors: Shreya Santra, Thomas Robbins, Kazuya Yoshida

Autonomous navigation in unstructured environments is essential for field and
planetary robotics, where robots must efficiently reach goals while avoiding
obstacles under uncertain conditions. Conventional algorithmic approaches often
require extensive environment-specific tuning, limiting scalability to new
domains. Deep Reinforcement Learning (DRL) provides a data-driven alternative,
allowing robots to acquire navigation strategies through direct interactions
with their environment. This work investigates the feasibility of DRL policy
generalization across visually and topographically distinct simulated domains,
where policies are trained in terrestrial settings and validated in a zero-shot
manner in extraterrestrial environments. A 3D simulation of an agricultural
rover is developed and trained using Proximal Policy Optimization (PPO) to
achieve goal-directed navigation and obstacle avoidance in farmland settings.
The learned policy is then evaluated in a lunar-like simulated environment to
assess transfer performance. The results indicate that policies trained under
terrestrial conditions retain a high level of effectiveness, achieving close to
50\% success in lunar simulations without the need for additional training and
fine-tuning. This underscores the potential of cross-domain DRL-based policy
transfer as a promising approach to developing adaptable and efficient
autonomous navigation for future planetary exploration missions, with the added
benefit of minimizing retraining costs.

### Software Engineering

### 1. [TALM: Dynamic Tree-Structured Multi-Agent Framework with Long-Term Memory for Scalable Code Generation](http://arxiv.org/pdf/2510.23010v1)

Authors: Ming-Tung Shen, Yuh-Jzer Joung

Agentic code generation requires large language models (LLMs) capable of
complex context management and multi-step reasoning. Prior multi-agent
frameworks attempt to address these challenges through collaboration, yet they
often suffer from rigid workflows and high reasoning recovery costs. To
overcome these limitations, we propose TALM (Tree-Structured Multi-Agent
Framework with Long-Term Memory), a dynamic framework that integrates
structured task decomposition, localized re-reasoning, and long-term memory
mechanisms. TALM employs an extensible tree-based collaboration structure. The
parent-child relationships, when combined with a divide-and-conquer strategy,
enhance reasoning flexibility and enable efficient error correction across
diverse task scopes. Furthermore, a long-term memory module enables semantic
querying and integration of prior knowledge, supporting implicit
self-improvement through experience reuse. Experimental results on HumanEval,
BigCodeBench, and ClassEval benchmarks demonstrate that TALM consistently
delivers strong reasoning performance and high token efficiency, highlighting
its robustness and practical utility in complex code generation tasks.

### 2. [From Online User Feedback to Requirements: Evaluating Large Language Models for Classification and Specification Tasks](http://arxiv.org/pdf/2510.23055v1)

Authors: Manjeshwar Aniruddh Mallya, Alessio Ferrari, Mohammad Amin Zadenoori, Jacek Dąbrowski

[Context and Motivation] Online user feedback provides valuable information
to support requirements engineering (RE). However, analyzing online user
feedback is challenging due to its large volume and noise. Large language
models (LLMs) show strong potential to automate this process and outperform
previous techniques. They can also enable new tasks, such as generating
requirements specifications.
  [Question-Problem] Despite their potential, the use of LLMs to analyze user
feedback for RE remains underexplored. Existing studies offer limited empirical
evidence, lack thorough evaluation, and rarely provide replication packages,
undermining validity and reproducibility.
  [Principal Idea-Results] We evaluate five lightweight open-source LLMs on
three RE tasks: user request classification, NFR classification, and
requirements specification generation. Classification performance was measured
on two feedback datasets, and specification quality via human evaluation. LLMs
achieved moderate-to-high classification accuracy (F1 ~ 0.47-0.68) and
moderately high specification quality (mean ~ 3/5).
  [Contributions] We newly explore lightweight LLMs for feedback-driven
requirements development. Our contributions are: (i) an empirical evaluation of
lightweight LLMs on three RE tasks, (ii) a replication package, and (iii)
insights into their capabilities and limitations for RE.

### 3. [Checkstyle+: Reducing Technical Debt Through The Use of Linters with LLMs](http://arxiv.org/pdf/2510.23068v1)

Authors: Ella Dodor, Cristina V. Lopes

Good code style improves program readability, maintainability, and
collaboration, and is an integral component of software quality. Developers,
however, often cut corners when following style rules, leading to the wide
adoption of tools such as linters in professional software development
projects. Traditional linters like Checkstyle operate using rigid, rule-based
mechanisms that effectively detect many surface-level violations. However, in
most programming languages, there is a subset of style rules that require a
more nuanced understanding of code, and fall outside the scope of such static
analysis. In this paper, we propose Checkstyle+, a hybrid approach that
augments Checkstyle with large language model (LLM) capabilities, to identify
style violations that elude the conventional rule-based analysis. Checkstyle+
is evaluated on a sample of 380 Java code files, drawn from a broader dataset
of 30,800 real-world Java programs sourced from accepted Codeforces
submissions. The results show that Checkstyle+ achieves superior performance
over standard Checkstyle in detecting violations of the semantically nuanced
rules.

### 4. [Validating Formal Specifications with LLM-generated Test Cases](http://arxiv.org/pdf/2510.23350v1)

Authors: Alcino Cunha, Nuno Macedo

Validation is a central activity when developing formal specifications.
Similarly to coding, a possible validation technique is to define upfront test
cases or scenarios that a future specification should satisfy or not.
Unfortunately, specifying such test cases is burdensome and error prone, which
could cause users to skip this validation task. This paper reports the results
of an empirical evaluation of using pre-trained large language models (LLMs) to
automate the generation of test cases from natural language requirements. In
particular, we focus on test cases for structural requirements of simple domain
models formalized in the Alloy specification language. Our evaluation focuses
on the state-of-art GPT-5 model, but results from other closed- and open-source
LLMs are also reported. The results show that, in this context, GPT-5 is
already quite effective at generating positive (and negative) test cases that
are syntactically correct and that satisfy (or not) the given requirement, and
that can detect many wrong specifications written by humans.

### 5. [Tracing Distribution Shifts with Causal System Maps](http://arxiv.org/pdf/2510.23528v1)

Authors: Joran Leest, Ilias Gerostathopoulos, Patricia Lago, Claudia Raibulet

Monitoring machine learning (ML) systems is hard, with standard practice
focusing on detecting distribution shifts rather than their causes. Root-cause
analysis often relies on manual tracing to determine whether a shift is caused
by software faults, data-quality issues, or natural change. We propose ML
System Maps -- causal maps that, through layered views, make explicit the
propagation paths between the environment and the ML system's internals,
enabling systematic attribution of distribution shifts. We outline the approach
and a research agenda for its development and evaluation.

### 6. [On Generalization in Agentic Tool Calling: CoreThink Agentic Reasoner and MAVEN Dataset](http://arxiv.org/pdf/2510.22898v1)

Authors: Vishvesh Bhat, Omkar Ghugarkar, Julian McAuley

Generalization across Agentic tool-calling environments remains a key
unsolved challenge in developing reliable agentic reasoning systems. While
large language models (LLMs) demonstrate strong performance on isolated
benchmarks, their ability to transfer reasoning strategies and co-ordinate
tools across diverse domains is poorly understood. In this work, we conduct a
large-scale evaluation of state-of-the-art LLMs on multiple tool-calling
benchmarksBFCL v3, TauBench, Tau2Bench, and AceBenchand introduce MAVEN (Math &
Physics Adversarial Verification & Evaluation Network), a new out of
distribution (OOD) benchmark designed to stress-test multi-step reasoning
through explicit verification and adversarial task composition. Our results
show that most current models achieve below 50% accuracy on MAVEN, revealing a
significant generalization gap across tool-use settings.
  To address this, we present the CoreThink Agentic Reasoner, a framework that
augments LLMs with a lightweight symbolic reasoning layer for structured
decomposition and adaptive tool orchestration. Without additional training, it
generalizes across all benchmarks, achieving state-of-the-art performance with
530% improvements over existing baselines at roughly one-tenth the
computational cost.

### 7. [A Multi-Store Privacy Measurement of Virtual Reality App Ecosystem](http://arxiv.org/pdf/2510.23024v1)

Authors: Chuan Yan, Zeng Li, Kunlin Cai, Liuhuo Wan, Ruomai Ren, Yiran Shen, Guangdong Bai

Virtual Reality (VR) has gained increasing traction among various domains in
recent years, with major companies such as Meta, Pico, and Microsoft launching
their application stores to support third-party developers in releasing their
applications (or simply apps). These apps offer rich functionality but
inherently collect privacy-sensitive data, such as user biometrics, behaviors,
and the surrounding environment. Nevertheless, there is still a lack of
domain-specific regulations to govern the data handling of VR apps, resulting
in significant variations in their privacy practices among app stores.
  In this work, we present the first comprehensive multi-store study of privacy
practices in the current VR app ecosystem, covering a large-scale dataset
involving 6,565 apps collected from five major app stores. We assess both
declarative and behavioral privacy practices of VR apps, using a multi-faceted
approach based on natural language processing, reverse engineering, and static
analysis. Our assessment reveals significant privacy compliance issues across
all stores, underscoring the premature status of privacy protection in this
rapidly growing ecosystem. For instance, one third of apps fail to declare
their use of sensitive data, and 21.5\% of apps neglect to provide valid
privacy policies. Our work sheds light on the status quo of privacy protection
within the VR app ecosystem for the first time. Our findings should raise an
alert to VR app developers and users, and encourage store operators to
implement stringent regulations on privacy compliance among VR apps.

### 8. [MATCH: Task-Driven Code Evaluation through Contrastive Learning](http://arxiv.org/pdf/2510.23169v1)

Authors: Marah Ghoummaid, Vladimir Tchuiev, Ofek Glick, Michal Moschkovitz, Dotan Di Castro

AI-based code generation is increasingly prevalent, with GitHub Copilot
estimated to generate 46% of the code on GitHub. Accurately evaluating how well
generated code aligns with developer intent remains a critical challenge.
Traditional evaluation methods, such as unit tests, are often unscalable and
costly. Syntactic similarity metrics (e.g., BLEU, ROUGE) fail to capture code
functionality, and metrics like CodeBERTScore require reference code, which is
not always available. To address the gap in reference-free evaluation, with few
alternatives such as ICE-Score, this paper introduces MATCH, a novel
reference-free metric. MATCH uses Contrastive Learning to generate meaningful
embeddings for code and natural language task descriptions, enabling similarity
scoring that reflects how well generated code implements the task. We show that
MATCH achieves stronger correlations with functional correctness and human
preference than existing metrics across multiple programming languages.

### 9. [Language Server CLI Empowers Language Agents with Process Rewards](http://arxiv.org/pdf/2510.22907v1)

Authors: Yifan Zhang, Lanser Contributors

Large language models routinely hallucinate APIs and mislocalize edits, while
language servers compute verified, IDE-grade facts about real code. We present
Lanser-CLI, a CLI-first orchestration layer that pins and mediates a Language
Server Protocol (LSP) server for coding agents and CI, exposing deterministic,
replayable workflows. Our position is that language servers provide not only
structural information (definitions, references, types, diagnostics) but also
an actionable process reward: machine-checked, step-wise signals that align an
agent's planning loop with program reality. In this work, Lanser-CLI
contributes: (i) a robust addressing scheme beyond brittle "file:line:col" via
a Selector DSL (symbolic, AST-path, and content-anchored selectors) with a
principled relocation algorithm; (ii) deterministic Analysis Bundles that
normalize Language Server responses and capture environment/capability metadata
with stable content hashes; (iii) a safety envelope for mutating operations
(rename, code actions) with preview, workspace jails, and Git-aware,
transactional apply; and (iv) a process-reward functional derived from Language
Server facts (diagnostic deltas, disambiguation confidence, and safe-apply
checks) that is computable online and replayable offline. We formalize
determinism under frozen snapshots and establish a monotonicity property for
the process reward, making it suitable for process supervision and
counterfactual analysis. Project Page:
https://github.com/yifanzhang-pro/lanser-cli

### 10. [CodeAD: Synthesize Code of Rules for Log-based Anomaly Detection with LLMs](http://arxiv.org/pdf/2510.22986v1)

Authors: Junjie Huang, Minghua He, Jinyang Liu, Yintong Huo, Domenico Bianculli, Michael R. Lyu

Log-based anomaly detection (LogAD) is critical for maintaining the
reliability and availability of large-scale online service systems. While
machine learning, deep learning, and large language models (LLMs)-based methods
have advanced the LogAD, they often suffer from limited interpretability, high
inference costs, and extensive preprocessing requirements, limiting their
practicality for real-time, high-volume log analysis. In contrast, rule-based
systems offer efficiency and transparency, but require significant manual
effort and are difficult to scale across diverse and evolving environments. In
this paper, We present CodeAD, a novel framework that automatically synthesizes
lightweight Python rule functions for LogAD using LLMs. CodeAD introduces a
hierarchical clustering and anchor-grounded sampling strategy to construct
representative contrastive log windows, enabling LLMs to discern discriminative
anomaly patterns. To ensure robustness and generalizability, CodeAD employs an
agentic workflow that iteratively generates, tests, repairs, and refines the
rules until it meets correctness and abstraction requirements. The synthesized
rules are interpretable, lightweight, and directly executable on raw logs,
supporting efficient and transparent online anomaly detection. Our
comprehensive experiments on three public datasets (BGL, Hadoop, Thunderbird)
demonstrate that CodeAD achieves an average absolute improvement of 3.6% F1
score over the state-of-the-art baselines, while processing large datasets up
to 4x faster and at a fraction of the cost (total LLM invocation cost under 4
USD per dataset). These results highlight CodeAD as a practical and scalable
solution for online monitoring systems, enabling interpretable, efficient, and
automated LogAD in real-world environment.

### Social and Information Networks

### 1. [Modeling Political Discourse with Sentence-BERT and BERTopic](http://arxiv.org/pdf/2510.22904v1)

Authors: Margarida Mendonca, Alvaro Figueira

Social media has reshaped political discourse, offering politicians a
platform for direct engagement while reinforcing polarization and ideological
divides. This study introduces a novel topic evolution framework that
integrates BERTopic-based topic modeling with Moral Foundations Theory (MFT) to
analyze the longevity and moral dimensions of political topics in Twitter
activity during the 117th U.S. Congress. We propose a methodology for tracking
dynamic topic shifts over time and measuring their association with moral
values and quantifying topic persistence. Our findings reveal that while
overarching themes remain stable, granular topics tend to dissolve rapidly,
limiting their long-term influence. Moreover, moral foundations play a critical
role in topic longevity, with Care and Loyalty dominating durable topics, while
partisan differences manifest in distinct moral framing strategies. This work
contributes to the field of social network analysis and computational political
discourse by offering a scalable, interpretable approach to understanding
moral-driven topic evolution on social media.

### 2. [Grassmanian Interpolation of Low-Pass Graph Filters: Theory and Applications](http://arxiv.org/pdf/2510.23235v1)

Authors: Anton Savostianov, Michael T. Schaub, Benjamin Stamm

Low-pass graph filters are fundamental for signal processing on graphs and
other non-Euclidean domains. However, the computation of such filters for
parametric graph families can be prohibitively expensive as computation of the
corresponding low-frequency subspaces, requires the repeated solution of an
eigenvalue problem. We suggest a novel algorithm of low-pass graph filter
interpolation based on Riemannian interpolation in normal coordinates on the
Grassmann manifold. We derive an error bound estimate for the subspace
interpolation and suggest two possible applications for induced parametric
graph families. First, we argue that the temporal evolution of the node
features may be translated to the evolving graph topology via a similarity
correction to adjust the homophily degree of the network. Second, we suggest a
dot product graph family induced by a given static graph which allows to infer
improved message passing scheme for node classification facilitated by the
filter interpolation.

### Systems and Control

### 1. [NeuroDOB: A Deep Neural Observer-Based Controller for Vehicle Lateral Dynamics](http://arxiv.org/pdf/2510.23067v1)

Authors: Sangmin Kim, Taehun Kim, Guntae Kim, Chang Mook Kang

This paper proposes NeuroDOB, a deep neural network based observer controller
for vehicle lateral dynamics, which replaces the conventional disturbance
observer (DOB) with a deep neural network (DNN) to enhance personalized lateral
control. Unlike conventional DOBs that compensate for general disturbances such
as road friction variation and crosswind, NeuroDOB explicitly addresses
unmodeled vehicle dynamics and driver-specific behaviors by learning the
steering compensation signal from driver-in-the-loop simulations using CarSim's
embedded controller as a surrogate driver. The proposed architecture integrates
NeuroDOB with a linear quadratic regulator (LQR), where the DNN outputs a delta
error correction added to the baseline LQR steering input to produce the final
control command. Input features to the DNN include lateral position and yaw
angle errors, and the LQR control input. Experimental validation using a
lateral dynamic bicycle model within CarSim demonstrates that NeuroDOB
effectively adapts to individual driving habits, improving lateral control
performance beyond what conventional LQR controllers achieve. The results
indicate the potential of deep neural network based observer to enable
personalized and adaptive autonomous vehicle control. In cognitive terms, the
proposed architecture can be viewed as a dual-system control structure. The
baseline LQR corresponds to System 1, a model-based, fast, and analytic
reasoning layer ensuring stability. The NeuroDOB acts as System 2, a
reflective, data-driven layer that learns compensation from experience and
corrects the analytical bias of System 1. Together, they form an integrated
decision process analogous to human intuition-reflection interaction, enabling
both stability and adaptability in lateral control.

### 2. [Context-awareness for Dependable Low-Power IoT](http://arxiv.org/pdf/2510.23125v1)

Authors: David E. Ruiz-Guirola, Prasoon Raghuwanshi, Gabriel M. de Jesus, Mateen Ashraf, Onel L. A. López

Dependability is the ability to consistently deliver trusted and
uninterrupted service in the face of operational uncertainties. Ensuring
dependable operation in large-scale, energy-constrained Internet of Things
(IoT) deployments is as crucial as challenging, and calls for context-aware
protocols where context refers to situational or state information. In this
paper, we identify four critical context dimensions for IoT networks, namely
energy status, information freshness, task relevance, and physical/medium
conditions, and show how each one underpins core dependability attributes.
Building on these insights, we propose a two-step protocol design framework
that incorporates operation-specific context fields. Through three
representative use cases, we demonstrate how context awareness can
significantly enhance system dependability while imposing only minimal
control-plane overhead.

### 3. [Embroidery Actuator Utilizing Embroidery Patterns to Generate Diverse Fabric Deformations](http://arxiv.org/pdf/2510.23188v1)

Authors: Yuki Ota, Yuki Funabora

This paper presents a novel Embroidery Actuator, a fabric-integrated
pneumatic actuator that enables diverse and controllable deformations through
embroidery pattern design. Unlike conventional fabric actuators that rely on
fiber- or thread-shaped actuators, the proposed actuator is fabricated by
directly stitching an inflatable tube onto the fabric using a cord-embroidery
technique. The embroidered thread and the fabric jointly form a sleeve that
constrains the expansion of the inflatable tube, converting internal pressure
into targeted bending or stretching deformations. By varying the embroidery
pattern, such as zigzag or cross configurations, different geometric
constraints can be realized, allowing for flexible control of deformation
direction and magnitude. Analytical deformation models based on the Neo-Hookean
model and Lagrange's equations were developed to predict the relationship
between pneumatic pressure and bending angle, and were experimentally validated
using motion-capture measurements. The results demonstrated that the actuator
achieves strong agreement with the analytical deformation model.

### 4. [Neural Networks for AC Optimal Power Flow: Improving Worst-Case Guarantees during Training](http://arxiv.org/pdf/2510.23196v1)

Authors: Bastien Giraud, Rahul Nellikath, Johanna Vorwerk, Maad Alowaifeer, Spyros Chatzivasileiadis

The AC Optimal Power Flow (AC-OPF) problem is central to power system
operation but challenging to solve efficiently due to its nonconvex and
nonlinear nature. Neural networks (NNs) offer fast surrogates, yet their
black-box behavior raises concerns about constraint violations that can
compromise safety. We propose a verification-informed NN framework that
incorporates worst-case constraint violations directly into training, producing
models that are both accurate and provably safer. Through post-hoc
verification, we achieve substantial reductions in worst-case violations and,
for the first time, verify all operational constraints of large-scale AC-OPF
proxies. Practical feasibility is further enhanced via restoration and
warm-start strategies for infeasible operating points. Experiments on systems
ranging from 57 to 793 buses demonstrate scalability, speed, and reliability,
bridging the gap between ML acceleration and safe, real-time deployment of
AC-OPF solutions - and paving the way toward data-driven optimal control.

### 5. [Inertia Partitioning Modular Control Framework for Reconfigurable Multibody Systems](http://arxiv.org/pdf/2510.23226v1)

Authors: Mohammad Dastranj, Jouni Mattila

A novel modular control framework for reconfigurable rigid multibody systems
is proposed, motivated by the challenges of modular control of systems with
closed kinematic chains. In the framework, modularity is defined in the sense
of degrees of freedom, and the inertial properties of each body are partitioned
with respect to how they are reflected in the kinetic energy of the system
through the motion induced by each degree of freedom. This approach inherently
handles closed chains in the same manner as tree-like structures, eliminating
the need for explicit constraint force calculations or formulations based on
differential-algebraic equations. The proposed framework is implemented via
simulation on a three-degree-of-freedom series-parallel manipulator, with the
results being consistent with the expected stability and tracking performance,
and indicating the framework's potential for scalability in trajectory-tracking
control of multibody systems.

### 6. [Towards Stochastic (N-1)-Secure Redispatch](http://arxiv.org/pdf/2510.23551v1)

Authors: Oleksii Molodchyk, Hendrik Drögehorn, Martin Lindner, Mario Kendziorski, Timm Faulwasser

The intermittent nature of renewable power availability is one of the major
sources of uncertainty in power systems. While markets can guarantee that the
demand is covered by the available generation, transmission system operators
have to often intervene via economic redispatch to ensure that the physical
constraints of the network are satisfied. To account for uncertainty, the
underlying optimal power flow (OPF) routines have to be modified. Recently,
polynomial chaos expansion (PCE) has been suggested in the literature as a tool
for stochastic OPF problems. However, the usage of PCE-based methods in
security-constrained OPF for (N-1)-secure operations has not yet been explored.
In this paper, we propose a procedure that iteratively solves a PCE-overloaded
stochastic OPF problem by including line outage constraints until an
(N-1)-secure solution is achieved. We demonstrate the efficacy of our method by
comparing it with a Monte-Carlo simulation on a 118-bus example system.

### 7. [Never Too Rigid to Reach: Adaptive Virtual Model Control with LLM- and Lyapunov-Based Reinforcement Learning](http://arxiv.org/pdf/2510.22892v1)

Authors: Jingzehua Xu, Yangyang Li, Yangfei Chen, Guanwen Xie, Shuai Zhang

Robotic arms are increasingly deployed in uncertain environments, yet
conventional control pipelines often become rigid and brittle when exposed to
perturbations or incomplete information. Virtual Model Control (VMC) enables
compliant behaviors by embedding virtual forces and mapping them into joint
torques, but its reliance on fixed parameters and limited coordination among
virtual components constrains adaptability and may undermine stability as task
objectives evolve. To address these limitations, we propose Adaptive VMC with
Large Language Model (LLM)- and Lyapunov-Based Reinforcement Learning (RL),
which preserves the physical interpretability of VMC while supporting
stability-guaranteed online adaptation. The LLM provides structured priors and
high-level reasoning that enhance coordination among virtual components,
improve sample efficiency, and facilitate flexible adjustment to varying task
requirements. Complementarily, Lyapunov-based RL enforces theoretical stability
constraints, ensuring safe and reliable adaptation under uncertainty. Extensive
simulations on a 7-DoF Panda arm demonstrate that our approach effectively
balances competing objectives in dynamic tasks, achieving superior performance
while highlighting the synergistic benefits of LLM guidance and
Lyapunov-constrained adaptation.

### 8. [End-to-End Design and Validation of a Low-Cost Stewart Platform with Nonlinear Estimation and Control](http://arxiv.org/pdf/2510.22949v1)

Authors: Benedictus C. G. Cinun, Tua A. Tamba, Immanuel R. Santjoko, Xiaofeng Wang, Michael A. Gunarso, Bin Hu

This paper presents the complete design, control, and experimental validation
of a low-cost Stewart platform prototype developed as an affordable yet capable
robotic testbed for research and education. The platform combines off the shelf
components with 3D printed and custom fabricated parts to deliver full six
degrees of freedom motions using six linear actuators connecting a moving
platform to a fixed base. The system software integrates dynamic modeling, data
acquisition, and real time control within a unified framework. A robust
trajectory tracking controller based on feedback linearization, augmented with
an LQR scheme, compensates for the platform's nonlinear dynamics to achieve
precise motion control. In parallel, an Extended Kalman Filter fuses IMU and
actuator encoder feedback to provide accurate and reliable state estimation
under sensor noise and external disturbances. Unlike prior efforts that
emphasize only isolated aspects such as modeling or control, this work delivers
a complete hardware-software platform validated through both simulation and
experiments on static and dynamic trajectories. Results demonstrate effective
trajectory tracking and real-time state estimation, highlighting the platform's
potential as a cost effective and versatile tool for advanced research and
educational applications.

### 9. [An Intelligent Water-Saving Irrigation System Based on Multi-Sensor Fusion and Visual Servoing Control](http://arxiv.org/pdf/2510.23003v1)

Authors: ZhengKai Huang, YiKun Wang, ChenYu Hui, XiaoCheng

This paper introduces an intelligent water-saving irrigation system designed
to address critical challenges in precision agriculture, such as inefficient
water use and poor terrain adaptability. The system integrates advanced
computer vision, robotic control, and real-time stabilization technologies via
a multi-sensor fusion approach. A lightweight YOLO model, deployed on an
embedded vision processor (K210), enables real-time plant container detection
with over 96% accuracy under varying lighting conditions. A simplified hand-eye
calibration algorithm-designed for 'handheld camera' robot arm
configurations-ensures that the end effector can be precisely positioned, with
a success rate exceeding 90%. The active leveling system, driven by the
STM32F103ZET6 main control chip and JY901S inertial measurement data, can
stabilize the irrigation platform on slopes up to 10 degrees, with a response
time of 1.8 seconds. Experimental results across three simulated agricultural
environments (standard greenhouse, hilly terrain, complex lighting) demonstrate
a 30-50% reduction in water consumption compared to conventional flood
irrigation, with water use efficiency exceeding 92% in all test cases.

### 10. [Planning Oriented Integrated Sensing and Communication](http://arxiv.org/pdf/2510.23021v1)

Authors: Xibin Jin, Guoliang Li, Shuai Wang, Fan Liu, Miaowen Wen, Huseyin Arslan, Derrick Wing Kwan Ng, Chengzhong Xu

Integrated sensing and communication (ISAC) enables simultaneous
localization, environment perception, and data exchange for connected
autonomous vehicles. However, most existing ISAC designs prioritize sensing
accuracy and communication throughput, treating all targets uniformly and
overlooking the impact of critical obstacles on motion efficiency. To overcome
this limitation, we propose a planning-oriented ISAC (PISAC) framework that
reduces the sensing uncertainty of planning-bottleneck obstacles and expands
the safe navigable path for the ego-vehicle, thereby bridging the gap between
physical-layer optimization and motion-level planning. The core of PISAC lies
in deriving a closed-form safety bound that explicitly links ISAC transmit
power to sensing uncertainty, based on the Cram\'er-Rao Bound and occupancy
inflation principles. Using this model, we formulate a bilevel power allocation
and motion planning (PAMP) problem, where the inner layer optimizes the ISAC
beam power distribution and the outer layer computes a collision-free
trajectory under uncertainty-aware safety constraints. Comprehensive
simulations in high-fidelity urban driving environments demonstrate that PISAC
achieves up to 40% higher success rates and over 5% shorter traversal times
than existing ISAC-based and communication-oriented benchmarks, validating its
effectiveness in enhancing both safety and efficiency.

### Machine Learning (Statistics Category)

### 1. [On the Anisotropy of Score-Based Generative Models](http://arxiv.org/pdf/2510.22899v1)

Authors: Andreas Floros, Seyed-Mohsen Moosavi-Dezfooli, Pier Luigi Dragotti

We investigate the role of network architecture in shaping the inductive
biases of modern score-based generative models. To this end, we introduce the
Score Anisotropy Directions (SADs), architecture-dependent directions that
reveal how different networks preferentially capture data structure. Our
analysis shows that SADs form adaptive bases aligned with the architecture's
output geometry, providing a principled way to predict generalization ability
in score models prior to training. Through both synthetic data and standard
image benchmarks, we demonstrate that SADs reliably capture fine-grained model
behavior and correlate with downstream performance, as measured by Wasserstein
metrics. Our work offers a new lens for explaining and predicting directional
biases of generative models.

### 2. [Towards Personalized Treatment Plan: Geometrical Model-Agnostic Approach to Counterfactual Explanations](http://arxiv.org/pdf/2510.22911v1)

Authors: Daniel Sin, Milad Toutounchian

In our article, we describe a method for generating counterfactual
explanations in high-dimensional spaces using four steps that involve fitting
our dataset to a model, finding the decision boundary, determining constraints
on the problem, and computing the closest point (counterfactual explanation)
from that boundary. We propose a discretized approach where we find many
discrete points on the boundary and then identify the closest feasible
counterfactual explanation. This method, which we later call $\textit{Segmented
Sampling for Boundary Approximation}$ (SSBA), applies binary search to find
decision boundary points and then searches for the closest boundary point.
Across four datasets of varying dimensionality, we show that our method can
outperform current methods for counterfactual generation with reductions in
distance between $5\%$ to $50\%$ in terms of the $L_2$ norm. Our method can
also handle real-world constraints by restricting changes to immutable and
categorical features, such as age, gender, sex, height, and other related
characteristics such as the case for a health-based dataset. In terms of
runtime, the SSBA algorithm generates decision boundary points on multiple
orders of magnitude in the same given time when we compare to a grid-based
approach. In general, our method provides a simple and effective model-agnostic
method that can compute nearest feasible (i.e. realistic with constraints)
counterfactual explanations. All of our results and our code can be found here
at this link:
$\href{https://github.com/dsin85691/SSBA_For_Counterfactuals}{https://github.com/
dsin85691/SSBA\_For\_Counterfactuals}$

### 3. [How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data](http://arxiv.org/pdf/2510.22980v1)

Authors: Bhavya Vasudeva, Puneesh Deora, Yize Zhao, Vatsal Sharan, Christos Thrampoulidis

The growing adoption of spectrum-aware matrix-valued optimizers such as Muon
and Shampoo in deep learning motivates a systematic study of their
generalization properties and, in particular, when they might outperform
competitive algorithms. We approach this question by introducing appropriate
simplifying abstractions as follows: First, we use imbalanced data as a
testbed. Second, we study the canonical form of such optimizers, which is
Spectral Gradient Descent (SpecGD) -- each update step is $UV^T$ where $U\Sigma
V^T$ is the truncated SVD of the gradient. Third, within this framework we
identify a canonical setting for which we precisely quantify when SpecGD
outperforms vanilla Euclidean GD. For a Gaussian mixture data model and both
linear and bilinear models, we show that unlike GD, which prioritizes learning
dominant principal components of the data first, SpecGD learns all principal
components of the data at equal rates. We demonstrate how this translates to a
growing gap in balanced accuracy favoring SpecGD early in training and further
show that the gap remains consistent even when the GD counterpart uses adaptive
step-sizes via normalization. By extending the analysis to deep linear models,
we show that depth amplifies these effects. We empirically verify our
theoretical findings on a variety of imbalanced datasets. Our experiments
compare practical variants of spectral methods, like Muon and Shampoo, against
their Euclidean counterparts and Adam. The results validate our findings that
these spectral optimizers achieve superior generalization by promoting a more
balanced learning of the data's underlying components.

### 4. [Adaptive Forests For Classification](http://arxiv.org/pdf/2510.22991v1)

Authors: Dimitris Bertsimas, Yubing Cui

Random Forests (RF) and Extreme Gradient Boosting (XGBoost) are two of the
most widely used and highly performing classification and regression models.
They aggregate equally weighted CART trees, generated randomly in RF or
sequentially in XGBoost. In this paper, we propose Adaptive Forests (AF), a
novel approach that adaptively selects the weights of the underlying CART
models. AF combines (a) the Optimal Predictive-Policy Trees (OP2T) framework to
prescribe tailored, input-dependent unequal weights to trees and (b) Mixed
Integer Optimization (MIO) to refine weight candidates dynamically, enhancing
overall performance. We demonstrate that AF consistently outperforms RF,
XGBoost, and other weighted RF in binary and multi-class classification
problems over 20+ real-world datasets.

### 5. [Coupled Flow Matching](http://arxiv.org/pdf/2510.23015v1)

Authors: Wenxi Cai, Yuheng Wang, Naichen Shi

We introduce Coupled Flow Matching (CPFM), a framework that integrates
controllable dimensionality reduction and high-fidelity reconstruction. CPFM
learns coupled continuous flows for both the high-dimensional data x and the
low-dimensional embedding y, which enables sampling p(y|x) via a latent-space
flow and p(x|y) via a data-space flow. Unlike classical dimension-reduction
methods, where information discarded during compression is often difficult to
recover, CPFM preserves the knowledge of residual information within the
weights of a flow network. This design provides bespoke controllability: users
may decide which semantic factors to retain explicitly in the latent space,
while the complementary information remains recoverable through the flow
network. Coupled flow matching builds on two components: (i) an extended
Gromov-Wasserstein optimal transport objective that establishes a probabilistic
correspondence between data and embeddings, and (ii) a dual-conditional
flow-matching network that extrapolates the correspondence to the underlying
space. Experiments on multiple benchmarks show that CPFM yields semantically
rich embeddings and reconstructs data with higher fidelity than existing
baselines.

### 6. [The Benchmarking Epistemology: Construct Validity for Evaluating Machine Learning Models](http://arxiv.org/pdf/2510.23191v1)

Authors: Timo Freiesleben, Sebastian Zezulka

Predictive benchmarking, the evaluation of machine learning models based on
predictive performance and competitive ranking, is a central epistemic practice
in machine learning research and an increasingly prominent method for
scientific inquiry. Yet, benchmark scores alone provide at best measurements of
model performance relative to an evaluation dataset and a concrete learning
problem. Drawing substantial scientific inferences from the results, say about
theoretical tasks like image classification, requires additional assumptions
about the theoretical structure of the learning problems, evaluation functions,
and data distributions. We make these assumptions explicit by developing
conditions of construct validity inspired by psychological measurement theory.
We examine these assumptions in practice through three case studies, each
exemplifying a typical intended inference: measuring engineering progress in
computer vision with ImageNet; evaluating policy-relevant weather predictions
with WeatherBench; and examining limitations of the predictability of life
events with the Fragile Families Challenge. Our framework clarifies the
conditions under which benchmark scores can support diverse scientific claims,
bringing predictive benchmarking into perspective as an epistemological
practice and a key site of conceptual and theoretical reasoning in machine
learning.

### 7. [Rate-optimal Design for Anytime Best Arm Identification](http://arxiv.org/pdf/2510.23199v1)

Authors: Junpei Komiyama, Kyoungseok Jang, Junya Honda

We consider the best arm identification problem, where the goal is to
identify the arm with the highest mean reward from a set of $K$ arms under a
limited sampling budget. This problem models many practical scenarios such as
A/B testing. We consider a class of algorithms for this problem, which is
provably minimax optimal up to a constant factor. This idea is a generalization
of existing works in fixed-budget best arm identification, which are limited to
a particular choice of risk measures. Based on the framework, we propose Almost
Tracking, a closed-form algorithm that has a provable guarantee on the popular
risk measure $H_1$. Unlike existing algorithms, Almost Tracking does not
require the total budget in advance nor does it need to discard a significant
part of samples, which gives a practical advantage. Through experiments on
synthetic and real-world datasets, we show that our algorithm outperforms
existing anytime algorithms as well as fixed-budget algorithms.

### 8. [GCAO: Group-driven Clustering via Gravitational Attraction and Optimization](http://arxiv.org/pdf/2510.23259v1)

Authors: Qi Li, Jun Wang

Traditional clustering algorithms often struggle with high-dimensional and
non-uniformly distributed data, where low-density boundary samples are easily
disturbed by neighboring clusters, leading to unstable and distorted clustering
results. To address this issue, we propose a Group-driven Clustering via
Gravitational Attraction and Optimization (GCAO) algorithm. GCAO introduces a
group-level optimization mechanism that aggregates low-density boundary points
into collaboratively moving groups, replacing the traditional point-based
contraction process. By combining local density estimation with neighborhood
topology, GCAO constructs effective gravitational interactions between groups
and their surroundings, enhancing boundary clarity and structural consistency.
Using groups as basic motion units, a gravitational contraction strategy
ensures globally stable and directionally consistent convergence. Experiments
on multiple high-dimensional datasets demonstrate that GCAO outperforms 11
representative clustering methods, achieving average improvements of 37.13%,
52.08%, 44.98%, and 38.81% in NMI, ARI, Homogeneity, and ACC, respectively,
while maintaining competitive efficiency and scalability. These results
highlight GCAO's superiority in preserving cluster integrity, enhancing
boundary separability, and ensuring robust performance on complex data
distributions.

### 9. [Robust Non-negative Proximal Gradient Algorithm for Inverse Problems](http://arxiv.org/pdf/2510.23362v1)

Authors: Hanzhang Wang, Zonglin Liu, Jingyi Xu, Chenyang Wang, Zhiwei Zhong, Qiangqiang Shen

Proximal gradient algorithms (PGA), while foundational for inverse problems
like image reconstruction, often yield unstable convergence and suboptimal
solutions by violating the critical non-negativity constraint. We identify the
gradient descent step as the root cause of this issue, which introduces
negative values and induces high sensitivity to hyperparameters. To overcome
these limitations, we propose a novel multiplicative update proximal gradient
algorithm (SSO-PGA) with convergence guarantees, which is designed for
robustness in non-negative inverse problems. Our key innovation lies in
superseding the gradient descent step with a learnable sigmoid-based operator,
which inherently enforces non-negativity and boundedness by transforming
traditional subtractive updates into multiplicative ones. This design,
augmented by a sliding parameter for enhanced stability and convergence, not
only improves robustness but also boosts expressive capacity and noise
immunity. We further formulate a degradation model for multi-modal restoration
and derive its SSO-PGA-based optimization algorithm, which is then unfolded
into a deep network to marry the interpretability of optimization with the
power of deep learning. Extensive numerical and real-world experiments
demonstrate that our method significantly surpasses traditional PGA and other
state-of-the-art algorithms, ensuring superior performance and stability.

### 10. [An Information-Theoretic Analysis of Out-of-Distribution Generalization in Meta-Learning with Applications to Meta-RL](http://arxiv.org/pdf/2510.23448v1)

Authors: Xingtu Liu

In this work, we study out-of-distribution generalization in meta-learning
from an information-theoretic perspective. We focus on two scenarios: (i) when
the testing environment mismatches the training environment, and (ii) when the
training environment is broader than the testing environment. The first
corresponds to the standard distribution mismatch setting, while the second
reflects a broad-to-narrow training scenario. We further formalize the
generalization problem in meta-reinforcement learning and establish
corresponding generalization bounds. Finally, we analyze the generalization
performance of a gradient-based meta-reinforcement learning algorithm.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-28 PST.

### 1. [Spintronic digital compute-in-memory macro for efficient artificial intelligence](https://www.nature.com/articles/s41928-025-01480-5)

Authors: 

### 2. [Open-source protein structure AI aims to match AlphaFold](https://www.nature.com/articles/d41586-025-03546-y)

Authors: Miryam Naddaf

### 3. [Estimation of protein content in wheat samples using NIR hyperspectral imaging and 1D-CNN](https://www.nature.com/articles/s41598-025-15408-8)

Authors: Apurva Sharma et al.

### 4. [A swarm intelligence-driven hybrid framework for brain tumor classification with enhanced deep features](https://www.nature.com/articles/s41598-025-23820-3)

Authors: Aynur Yonar

### 5. [Transformer-based representation learning for robust gene expression modeling and cancer prognosis](https://www.nature.com/articles/s41598-025-14949-2)

Authors: Shuai Jiang et al.

### 6. [Quantum adaptive clonal genetic algorithm for low-energy clustering in agricultural WSNs](https://www.nature.com/articles/s41598-025-21501-9)

Authors: Jiawei Zhao et al.

### 7. [A dual-channel hyperspectral classification method based on NAS and transformer](https://www.nature.com/articles/s41598-025-21399-3)

Authors: Wenbin Liu et al.

