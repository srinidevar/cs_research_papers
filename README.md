# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-06-05 17:08:39.234564 PST.

### Artificial Intelligence

### 1. [Computational Architects of Society: Quantum Machine Learning for Social Rule Genesis](http://arxiv.org/pdf/2506.03503v1)

Authors: Shan Shan

The quantification of social science remains a longstanding challenge,
largely due to the philosophical nature of its foundational theories. Although
quantum computing has advanced rapidly in recent years, its relevance to social
theory remains underexplored. Most existing research focuses on micro-cognitive
models or philosophical analogies, leaving a gap in system-level applications
of quantum principles to the analysis of social systems. This study addresses
that gap by proposing a theoretical and computational framework that combines
quantum mechanics with Generative AI to simulate the emergence and evolution of
social norms. Drawing on core quantum concepts--such as superposition,
entanglement, and probabilistic measurement--this research models society as a
dynamic, uncertain system and sets up five ideal-type experiments. These
scenarios are simulated using 25 generative agents, each assigned evolving
roles as compliers, resistors, or enforcers. Within a simulated environment
monitored by a central observer (the Watcher), agents interact, respond to
surveillance, and adapt to periodic normative disruptions. These interactions
allow the system to self-organize under external stress and reveal emergent
patterns. Key findings show that quantum principles, when integrated with
generative AI, enable the modeling of uncertainty, emergence, and
interdependence in complex social systems. Simulations reveal patterns
including convergence toward normative order, the spread of resistance, and the
spontaneous emergence of new equilibria in social rules. In conclusion, this
study introduces a novel computational lens that lays the groundwork for a
quantum-informed social theory. It offers interdisciplinary insights into how
society can be understood not just as a structure to observe but as a dynamic
system to simulate and redesign through quantum technologies.

### 2. [SUMO-MCP: Leveraging the Model Context Protocol for Autonomous Traffic Simulation and Optimization](http://arxiv.org/pdf/2506.03548v1)

Authors: Chenglong Ye, Gang Xiong, Junyou Shang, Xingyuan Dai, Xiaoyan Gong, Yisheng Lv

Traffic simulation tools, such as SUMO, are essential for urban mobility
research. However, such tools remain challenging for users due to complex
manual workflows involving network download, demand generation, simulation
setup, and result analysis. In this paper, we introduce SUMO-MCP, a novel
platform that not only wraps SUMO' s core utilities into a unified tool suite
but also provides additional auxiliary utilities for common preprocessing and
postprocessing tasks. Using SUMO-MCP, users can issue simple natural-language
prompts to generate traffic scenarios from OpenStreetMap data, create demand
from origin-destination matrices or random patterns, run batch simulations with
multiple signal-control strategies, perform comparative analyses with automated
reporting, and detect congestion for signal-timing optimization. Furthermore,
the platform allows flexible custom workflows by dynamically combining exposed
SUMO tools without additional coding. Experiments demonstrate that SUMO-MCP
significantly makes traffic simulation more accessible and reliable for
researchers. We will release code for SUMO-MCP at
https://github.com/ycycycl/SUMO-MCP in the future.

### 3. [Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games](http://arxiv.org/pdf/2506.03610v1)

Authors: Dongmin Park, Minkyu Kim, Beongjun Choi, Junhyuck Kim, Keon Lee, Jonghyun Lee, Inkyu Park, Byeong-Uk Lee, Jaeyoung Hwang, Jaewoo Ahn, Ameya S. Mahabaleshwarkar, Bilal Kartal, Pritam Biswas, Yoshi Suhara, Kangwook Lee, Jaewoong Cho

Large Language Model (LLM) agents are reshaping the game industry,
particularly with more intelligent and human-preferable game characters.
However, existing game benchmarks fall short of practical needs: they lack
evaluations of diverse LLM capabilities across various game genres, studies of
agentic modules crucial for complex gameplay, and fine-tuning datasets for
aligning pre-trained LLMs into gaming agents. To fill these gaps, we present
\textbf{\benchname{}}, a foundational benchmark designed to train and evaluate
LLM agents across diverse real-world video games. Unlike existing benchmarks,
Orak includes 12 popular video games spanning all major genres, enabling
comprehensive studies of LLM capabilities and agentic modules essential for
intricate game scenarios. To support consistent evaluation of LLMs, we
introduce a plug-and-play interface based on Model Context Protocol (MCP) that
enables LLMs to seamlessly connect with games and manipulate agentic modules.
Additionally, we propose a fine-tuning dataset, consisting of LLM gameplay
trajectories across diverse game genres. Orak offers a comprehensive evaluation
framework, encompassing general game score leaderboards, LLM battle arenas, and
in-depth analyses of visual input state, agentic strategies, and fine-tuning
effects, establishing a foundation towards building generic gaming agents. Code
is available at https://github.com/krafton-ai/Orak.

### 4. [Reason from Future: Reverse Thought Chain Enhances LLM Reasoning](http://arxiv.org/pdf/2506.03673v1)

Authors: Yinlong Xu, Yanzhao Zheng, Shuoshuo Sun, Shuaihan Huang, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu, Hongxia Xu, Jian Wu

It has been demonstrated that carefully designed reasoning paradigms, like
Chain-of-Thought (CoT) and Tree-of-Thought (ToT), can enhance the reasoning
capabilities of small language models by detailed thinking and extensive
thought searching, unbounded branching factors in the searching space create
prohibitive reasoning consumption. However these methods fall into the trap of
local optimum reasoning, which means the model lacks a global perspective while
solving problems. We propose a novel reasoning paradigm called Reason from
Future (RFF), which generates reasoning paths by bidirectional reasoning that
combines top-down planning with bottom-up reasoning accumulation. The essence
of RFF lies in its reverse reasoning mechanism, which prioritizes core logical
relationships and imposes goal-oriented constraints on intermediate steps,
thereby reducing the searching space and mitigating error accumulation inherent
in sequential forward reasoning. Empirical evaluations across diverse
experiments demonstrate that RFF outperforms conventional paradigms with higher
accuracy and less searching space to solve complex tasks.

### 5. [Causal Explanations Over Time: Articulated Reasoning for Interactive Environments](http://arxiv.org/pdf/2506.03915v1)

Authors: Sebastian Rödling, Matej Zečević, Devendra Singh Dhami, Kristian Kersting

Structural Causal Explanations (SCEs) can be used to automatically generate
explanations in natural language to questions about given data that are
grounded in a (possibly learned) causal model. Unfortunately they work for
small data only. In turn they are not attractive to offer reasons for events,
e.g., tracking causal changes over multiple time steps, or a behavioral
component that involves feedback loops through actions of an agent. To this
end, we generalize SCEs to a (recursive) formulation of explanation trees to
capture the temporal interactions between reasons. We show the benefits of this
more general SCE algorithm on synthetic time-series data and a 2D grid game,
and further compare it to the base SCE and other existing methods for causal
explanations.

### 6. [TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems](http://arxiv.org/pdf/2506.04133v1)

Authors: Shaina Raza, Ranjan Sapkota, Manoj Karkee, Christos Emmanouilidis

Agentic AI systems, built on large language models (LLMs) and deployed in
multi-agent configurations, are redefining intelligent autonomy, collaboration
and decision-making across enterprise and societal domains. This review
presents a structured analysis of Trust, Risk, and Security Management (TRiSM)
in the context of LLM-based agentic multi-agent systems (AMAS). We begin by
examining the conceptual foundations of agentic AI, its architectural
differences from traditional AI agents, and the emerging system designs that
enable scalable, tool-using autonomy. The TRiSM in the agentic AI framework is
then detailed through four pillars governance, explainability, ModelOps, and
privacy/security each contextualized for agentic LLMs. We identify unique
threat vectors and introduce a comprehensive risk taxonomy for the agentic AI
applications, supported by case studies illustrating real-world
vulnerabilities. Furthermore, the paper also surveys trust-building mechanisms,
transparency and oversight techniques, and state-of-the-art explainability
strategies in distributed LLM agent systems. Additionally, metrics for
evaluating trust, interpretability, and human-centered performance are reviewed
alongside open benchmarking challenges. Security and privacy are addressed
through encryption, adversarial defense, and compliance with evolving AI
regulations. The paper concludes with a roadmap for responsible agentic AI,
proposing research directions to align emerging multi-agent systems with robust
TRiSM principles for safe, accountable, and transparent deployment.

### 7. [macOSWorld: A Multilingual Interactive Benchmark for GUI Agents](http://arxiv.org/pdf/2506.04135v1)

Authors: Pei Yang, Hai Ci, Mike Zheng Shou

Graphical User Interface (GUI) agents show promising capabilities for
automating computer-use tasks and facilitating accessibility, but existing
interactive benchmarks are mostly English-only, covering web-use or Windows,
Linux, and Android environments, but not macOS. macOS is a major OS with
distinctive GUI patterns and exclusive applications. To bridge the gaps, we
present macOSWorld, the first comprehensive benchmark for evaluating GUI agents
on macOS. macOSWorld features 202 multilingual interactive tasks across 30
applications (28 macOS-exclusive), with task instructions and OS interfaces
offered in 5 languages (English, Chinese, Arabic, Japanese, and Russian). As
GUI agents are shown to be vulnerable to deception attacks, macOSWorld also
includes a dedicated safety benchmarking subset. Our evaluation on six GUI
agents reveals a dramatic gap: proprietary computer-use agents lead at above
30% success rate, while open-source lightweight research models lag at below
2%, highlighting the need for macOS domain adaptation. Multilingual benchmarks
also expose common weaknesses, especially in Arabic, with a 27.5% average
degradation compared to English. Results from safety benchmarking also
highlight that deception attacks are more general and demand immediate
attention. macOSWorld is available at https://github.com/showlab/macosworld.

### 8. [Verification-Guided Falsification for Safe RL via Explainable Abstraction and Risk-Aware Exploration](http://arxiv.org/pdf/2506.03469v1)

Authors: Tuan Le, Risal Shefin, Debashis Gupta, Thai Le, Sarra Alqahtani

Ensuring the safety of reinforcement learning (RL) policies in high-stakes
environments requires not only formal verification but also interpretability
and targeted falsification. While model checking provides formal guarantees,
its effectiveness is limited by abstraction quality and the completeness of the
underlying trajectory dataset. We propose a hybrid framework that integrates
(1) explainability, (2) model checking, and (3) risk-guided falsification to
achieve both rigor and coverage. Our approach begins by constructing a
human-interpretable abstraction of the RL policy using Comprehensible Abstract
Policy Summarization (CAPS). This abstract graph, derived from offline
trajectories, is both verifier-friendly, semantically meaningful, and can be
used as input to Storm probabilistic model checker to verify satisfaction of
temporal safety specifications. If the model checker identifies a violation, it
will return an interpretable counterexample trace by which the policy fails the
safety requirement. However, if no violation is detected, we cannot conclude
satisfaction due to potential limitation in the abstraction and coverage of the
offline dataset. In such cases, we estimate associated risk during model
checking to guide a falsification strategy that prioritizes searching in
high-risk states and regions underrepresented in the trajectory dataset. We
further provide PAC-style guarantees on the likelihood of uncovering undetected
violations. Finally, we incorporate a lightweight safety shield that switches
to a fallback policy at runtime when such a risk exceeds a threshold,
facilitating failure mitigation without retraining.

### 9. [Explainable AI: XAI-Guided Context-Aware Data Augmentation](http://arxiv.org/pdf/2506.03484v1)

Authors: Melkamu Abay Mersha, Mesay Gemeda Yigezu, Atnafu Lambebo Tonja, Hassan Shakil, Samer Iskander, Olga Kolesnikova, Jugal Kalita

Explainable AI (XAI) has emerged as a powerful tool for improving the
performance of AI models, going beyond providing model transparency and
interpretability. The scarcity of labeled data remains a fundamental challenge
in developing robust and generalizable AI models, particularly for low-resource
languages. Conventional data augmentation techniques introduce noise, cause
semantic drift, disrupt contextual coherence, lack control, and lead to
overfitting. To address these challenges, we propose XAI-Guided Context-Aware
Data Augmentation. This novel framework leverages XAI techniques to modify less
critical features while selectively preserving most task-relevant features. Our
approach integrates an iterative feedback loop, which refines augmented data
over multiple augmentation cycles based on explainability-driven insights and
the model performance gain. Our experimental results demonstrate that XAI-SR-BT
and XAI-PR-BT improve the accuracy of models on hate speech and sentiment
analysis tasks by 6.6% and 8.1%, respectively, compared to the baseline, using
the Amharic dataset with the XLM-R model. XAI-SR-BT and XAI-PR-BT outperform
existing augmentation techniques by 4.8% and 5%, respectively, on the same
dataset and model. Overall, XAI-SR-BT and XAI-PR-BT consistently outperform
both baseline and conventional augmentation techniques across all tasks and
models. This study provides a more controlled, interpretable, and context-aware
solution to data augmentation, addressing critical limitations of existing
augmentation techniques and offering a new paradigm shift for leveraging XAI
techniques to enhance AI model training.

### 10. [EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](http://arxiv.org/pdf/2506.03489v1)

Authors: Mingxu Tao, Jie Hu, Mingchuan Yang, Yunhuai Liu, Dongyan Zhao, Yansong Feng

The remarkable performance of Large language models (LLMs) relies heavily on
the availability of abundant high-quality training data. However, the high cost
of acquiring annotated data often prevents models from obtaining capabilities
to tackle downstream tasks. In this paper, we introduce a novel method, EpiCoDe
that boosts model performance in data-scarcity scenarios without extra
training. We first employ model extrapolation to enhance a finetuned model with
its inferior version, and then adopt contrastive decoding to further reduce
predicted errors, by comparing the logit scores given by the extrapolated and
the vanilla finetuned model. Experiments across three tasks over four different
LLMs show that EpiCoDe consistently outperforms existing methods with
significant and robust improvement. We also propose a new theoretical framework
to reveal the mechanism behind contrastive decoding in data-scarcity scenarios,
which further helps us better understand the effectiveness of EpiCoDe.

### Hardware Architecture

### 1. [FPGA-Enabled Machine Learning Applications in Earth Observation: A Systematic Review](http://arxiv.org/pdf/2506.03938v1)

Authors: Cédric Léonard, Dirk Stober, Martin Schulz

New UAV technologies and the NewSpace era are transforming Earth Observation
missions and data acquisition. Numerous small platforms generate large data
volume, straining bandwidth and requiring onboard decision-making to transmit
high-quality information in time. While Machine Learning allows real-time
autonomous processing, FPGAs balance performance with adaptability to
mission-specific requirements, enabling onboard deployment. This review
systematically analyzes 66 experiments deploying ML models on FPGAs for Remote
Sensing applications. We introduce two distinct taxonomies to capture both
efficient model architectures and FPGA implementation strategies. For
transparency and reproducibility, we follow PRISMA 2020 guidelines and share
all data and code at https://github.com/CedricLeon/Survey_RS-ML-FPGA.

### 2. [CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design](http://arxiv.org/pdf/2506.03474v1)

Authors: Yifeng Xiao, Yurong Xu, Ning Yan, Masood Mortazavi, Pierluigi Nuzzo

Simulation-based design space exploration (DSE) aims to efficiently optimize
high-dimensional structured designs under complex constraints and expensive
evaluation costs. Existing approaches, including heuristic and multi-step
reinforcement learning (RL) methods, struggle to balance sampling efficiency
and constraint satisfaction due to sparse, delayed feedback, and large hybrid
action spaces. In this paper, we introduce CORE, a constraint-aware, one-step
RL method for simulationguided DSE. In CORE, the policy agent learns to sample
design configurations by defining a structured distribution over them,
incorporating dependencies via a scaling-graph-based decoder, and by reward
shaping to penalize invalid designs based on the feedback obtained from
simulation. CORE updates the policy using a surrogate objective that compares
the rewards of designs within a sampled batch, without learning a value
function. This critic-free formulation enables efficient learning by
encouraging the selection of higher-reward designs. We instantiate CORE for
hardware-mapping co-design of neural network accelerators, demonstrating that
it significantly improves sample efficiency and achieves better accelerator
configurations compared to state-of-the-art baselines. Our approach is general
and applicable to a broad class of discrete-continuous constrained design
problems.

### Computational Complexity

### 1. [Hive is PSPACE-Hard](http://arxiv.org/pdf/2506.03492v1)

Authors: Daniël Andel, Benjamin Rin

Hive is an abstract strategy game played on a table with hexagonal pieces.
First published in 2001, it was and continues to be highly popular among both
casual and competitive players. In this paper, we show that for a suitably
generalized version of the game, the computational problem of determining
whether a given player in an arbitrary position has a winning strategy is
PSPACE-hard. We do this by reduction from a variant of Generalized Geography we
call Formula Game Geography.

### 2. [The Line Traveling Salesman and Repairman Problem with Collaboration](http://arxiv.org/pdf/2506.04127v1)

Authors: Julian Golak, Finn Sörensen, Malte Fliedner

In this work, we consider extensions of both the Line Traveling Salesman and
Line Traveling Repairman Problem, in which a single server must service a set
of clients located along a line segment under the assumption that not only the
server, but also the clients can move along the line and seek to collaborate
with the server to speed up service times. We analyze the structure of
different problem versions and identify hard and easy subproblems by building
up on prior results from the literature. Specifically, we investigate problem
versions with zero or general processing times, clients that are either slower
or faster than the server, as well as different time window restrictions.
Collectively, these results map out the complexity landscape of the Line
Traveling Salesman and Repairman Problem with collaboration.

### 3. [Training Cross-Morphology Embodied AI Agents: From Practical Challenges to Theoretical Foundations](http://arxiv.org/pdf/2506.03613v1)

Authors: Shaoshan Liu, Fan Wang, Hongjun Zhou, Yuanfeng Wang

While theory and practice are often seen as separate domains, this article
shows that theoretical insight is essential for overcoming real-world
engineering barriers. We begin with a practical challenge: training a
cross-morphology embodied AI policy that generalizes across diverse robot
morphologies. We formalize this as the Heterogeneous Embodied Agent Training
(HEAT) problem and prove it reduces to a structured Partially Observable Markov
Decision Process (POMDP) that is PSPACE-complete. This result explains why
current reinforcement learning pipelines break down under morphological
diversity, due to sequential training constraints, memory-policy coupling, and
data incompatibility. We further explore Collective Adaptation, a distributed
learning alternative inspired by biological systems. Though NEXP-complete in
theory, it offers meaningful scalability and deployment benefits in practice.
This work illustrates how computational theory can illuminate system design
trade-offs and guide the development of more robust, scalable embodied AI. For
practitioners and researchers to explore this problem, the implementation code
of this work has been made publicly available at
https://github.com/airs-admin/HEAT

### 4. [Minimizing the Arithmetic and Communication Complexity of Jacobi's Method for Eigenvalues and Singular Values](http://arxiv.org/pdf/2506.03466v1)

Authors: James Demmel, Hengrui Luo, Ryan Schneider, Yifu Wang

In this paper, we analyze several versions of Jacobi's method for the
symmetric eigenvalue problem. Our goal throughout is to reduce the asymptotic
cost of the algorithm as much as possible, as measured by the number of
arithmetic operations performed and associated (sequential or parallel)
communication, i.e., the amount of data moved between slow and fast memory or
between processors in a network. In producing rigorous complexity bounds, we
allow our algorithms to be built on both classic $O(n^3)$ matrix multiplication
and fast, Strassen-like $O(n^{\omega_0})$ alternatives. In the classical
setting, we show that a blocked implementation of Jacobi's method attains the
communication lower bound for $O(n^3)$ matrix multiplication (and is therefore
expected to be communication optimal among $O(n^3)$ methods). In the fast
setting, we demonstrate that a recursive version of blocked Jacobi can go even
further, reaching essentially optimal complexity in both measures. We also
discuss Jacobi-based SVD algorithms and a parallel version of block Jacobi,
showing that analogous complexity bounds apply.

### 5. [Complexity and Manipulation of International Kidney Exchange Programmes with Country-Specific Parameterss](http://arxiv.org/pdf/2506.04092v1)

Authors: Rachael Colley, David Manlove, Daniel Paulusma, Mengxiao Zhang

Kidney Exchange Programmes (KEPs) facilitate the exchange of kidneys, and
larger pools of recipient-donor pairs tend to yield proportionally more
transplants, leading to the proposal of international KEPs (IKEPs). However, as
studied by \citet{mincu2021ip}, practical limitations must be considered in
IKEPs to ensure that countries remain willing to participate. Thus, we study
IKEPs with country-specific parameters, represented by a tuple $\Gamma$,
restricting the selected transplants to be feasible for the countries to
conduct, e.g., imposing an upper limit on the number of consecutive exchanges
within a country's borders. We provide a complete complexity dichotomy for the
problem of finding a feasible (according to the constraints given by $\Gamma$)
cycle packing with the maximum number of transplants, for every possible
$\Gamma$. We also study the potential for countries to misreport their
parameters to increase their allocation. As manipulation can harm the total
number of transplants, we propose a novel individually rational and incentive
compatible mechanism $\mathcal{M}_{\text{order}}$. We first give a theoretical
approximation ratio for $\mathcal{M}_{\text{order}}$ in terms of the number of
transplants, and show that the approximation ratio of
$\mathcal{M}_{\text{order}}$ is asymptotically optimal. We then use simulations
which suggest that, in practice, the performance of
$\mathcal{M}_{\text{order}}$ is significantly better than this worst-case
ratio.

### Computational Engineering

### 1. [On the robustness of Dirichlet-Neumann coupling schemes for fluid-structure-interaction problems with nearly-closed fluid domains](http://arxiv.org/pdf/2506.04027v1)

Authors: A. Aissa-Berraies, Ferdinando A. Auricchio, Gertjan van Zwieten, E. Harald van Brummelen

Partitioned methods for fluid-structure interaction (FSI) involve solving the
structural and flow problems sequentially. These methods allow for separate
settings for the fluid and solid subsystems and thus modularity, enabling reuse
of advanced commercial and open-source software. Most partitioned FSI schemes
apply a Dirichlet-Neumann (DN) split of the interface conditions. The DN scheme
is adequate in a wide range of applications, but it is sensitive to the
added-mass effect, and it is susceptible to the incompressibility dilemma, i.e.
it completely fails for FSI problems with an incompressible fluid furnished
with Dirichlet boundary conditions on the part of its boundary complementary to
the interface. In this paper, we show that if the fluid is incompressible and
the fluid domain is nearly-closed, i.e. it carries Dirichlet conditions except
for a permeable part of the boundary carrying a Robin condition, then the DN
partitioned approach is sensitive to the flow resistance at the permeable part,
and convergence of the partitioned approach deteriorates as the flow resistance
increases. The DN scheme then becomes unstable in the limit as the flow
resistance passes to infinity. Based on a simple model problem, we show that in
the nearly-closed case, the convergence rate of the DN partitioned method
depends on a so-called added-damping effect. The analysis gives insights that
can aid to improve robustness and efficiency of partitioned method for FSI
problems with contact, e.g. valve applications. In addition, the results
elucidate the incompressibility dilemma as a limit of the added-damping effect
passing to infinity, and the corresponding challenges related to FSI problems
with nearly closed fluid-domain configurations. Via numerical experiments, we
consider the generalization of the results of the simple model problem to more
complex nearly-closed FSI problems.

### 2. [Risk and Reward of Transitioning from a National to a Zonal Electricity Market in Great Britain](http://arxiv.org/pdf/2506.04107v1)

Authors: Lukas Franken, Andrew Lyden, Daniel Friedrich

More spatially granular electricity wholesale markets promise more efficient
operation and better asset siting in highly renewable power systems. Great
Britain is considering moving from its current single-price national wholesale
market to a zonal design. Existing studies reach varying and
difficult-to-reconcile conclusions about the desirability of a zonal market in
GB, partly because they rely on models that vary in their transparency and
assumptions about future power systems. Using a novel open-source electricity
market model, calibrated to match observed network behaviour, this article
quantifies consumer savings, unit-level producer surplus impacts, and broader
socioeconomic benefits that would have arisen had a six-zone market operated in
Great Britain during 2022-2024. In the absence of mitigating policies, it is
estimated that during those three years GB consumers would save approximately
{\pounds}9.4/MWh (equalling an average of more than {\pounds}2.3B per year),
but generators in northern regions would experience revenue reductions of
30-40\%. Policy interventions can restore these units' national market revenues
to up to 97\% while still preserving around {\pounds}3.1/MWh in consumer
savings (about {\pounds}750M per year). It is further estimated that the
current system could achieve approximately {\pounds}380-{\pounds}770 million in
annual welfare gain during 2022-2024 through improved operational efficiency
alone. The drivers behind these benefits, notably wind curtailment volumes, are
expected to become more pronounced towards 2030, suggesting that purely
operationally achieved annual benefits of around {\pounds}1-2 billion beyond
2029 are likely. It is found that the scale of these benefits would outweigh
the potential downsides related to increases in the cost of capital that have
been estimated elsewhere.

### 3. [Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](http://arxiv.org/pdf/2506.04171v1)

Authors: Utkarsh Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gomez-Bombarelli, Christopher Vincent Rackauckas

Deep generative models have recently been applied to physical systems
governed by partial differential equations (PDEs), offering scalable simulation
and uncertainty-aware inference. However, enforcing physical constraints, such
as conservation laws (linear and nonlinear) and physical consistencies, remains
challenging. Existing methods often rely on soft penalties or architectural
biases that fail to guarantee hard constraints. In this work, we propose
Physics-Constrained Flow Matching (PCFM), a zero-shot inference framework that
enforces arbitrary nonlinear constraints in pretrained flow-based generative
models. PCFM continuously guides the sampling process through physics-based
corrections applied to intermediate solution states, while remaining aligned
with the learned flow and satisfying physical constraints. Empirically, PCFM
outperforms both unconstrained and constrained baselines on a range of PDEs,
including those with shocks, discontinuities, and sharp features, while
ensuring exact constraint satisfaction at the final solution. Our method
provides a general framework for enforcing hard constraints in both scientific
and general-purpose generative models, especially in applications where
constraint satisfaction is essential.

### Computational Geometry

### 1. [Better Late than Never: the Complexity of Arrangements of Polyhedra](http://arxiv.org/pdf/2506.03960v1)

Authors: Boris Aronov, Sang Won Bae, Sergio Cabello, Otfried Cheong, David Eppstein, Christian Knauer, Raimund Seidel

Let $\mathcal{A}$ be the subdivision of $\mathbb{R}^d$ induced by $m$ convex
polyhedra having $n$ facets in total. We prove that $\mathcal{A}$ has
combinatorial complexity $O(m^{\lceil d/2 \rceil} n^{\lfloor d/2 \rfloor})$ and
that this bound is tight. The bound is mentioned several times in the
literature, but no proof for arbitrary dimension has been published before.

### 2. [Optimizing Mesh to Improve the Triangular Expansion Algorithm for Computing Visibility Regions](http://arxiv.org/pdf/2506.04086v1)

Authors: Jan Mikula, Miroslav Kulich

This paper addresses the problem of improving the query performance of the
triangular expansion algorithm (TEA) for computing visibility regions by
finding the most advantageous instance of the triangular mesh, the
preprocessing structure. The TEA recursively traverses the mesh while keeping
track of the visible region, the set of all points visible from a query point
in a polygonal world. We show that the measured query time is approximately
proportional to the number of triangle edge expansions during the mesh
traversal. We propose a new type of triangular mesh that minimizes the expected
number of expansions assuming the query points are drawn from a known
probability distribution. We design a heuristic method to approximate the mesh
and evaluate the approach on many challenging instances that resemble
real-world environments. The proposed mesh improves the mean query times by
12-16% compared to the reference constrained Delaunay triangulation. The
approach is suitable to boost offline applications that require computing
millions of queries without addressing the preprocessing time. The
implementation is publicly available to replicate our experiments and serve the
community.

### Computation and Language

### 1. [Delta-KNN: Improving Demonstration Selection in In-Context Learning for Alzheimer's Disease Detection](http://arxiv.org/pdf/2506.03476v1)

Authors: Chuyuan Li, Raymond Li, Thalia S. Field, Giuseppe Carenini

Alzheimer's Disease (AD) is a progressive neurodegenerative disorder that
leads to dementia, and early intervention can greatly benefit from analyzing
linguistic abnormalities. In this work, we explore the potential of Large
Language Models (LLMs) as health assistants for AD diagnosis from
patient-generated text using in-context learning (ICL), where tasks are defined
through a few input-output examples. Empirical results reveal that conventional
ICL methods, such as similarity-based selection, perform poorly for AD
diagnosis, likely due to the inherent complexity of this task. To address this,
we introduce Delta-KNN, a novel demonstration selection strategy that enhances
ICL performance. Our method leverages a delta score to assess the relative
gains of each training example, coupled with a KNN-based retriever that
dynamically selects optimal "representatives" for a given input. Experiments on
two AD detection datasets across three open-source LLMs demonstrate that
Delta-KNN consistently outperforms existing ICL baselines. Notably, when using
the Llama-3.1 model, our approach achieves new state-of-the-art results,
surpassing even supervised classifiers.

### 2. [APT: Improving Specialist LLM Performance with Weakness Case Acquisition and Iterative Preference Training](http://arxiv.org/pdf/2506.03483v1)

Authors: Jun Rao, Zepeng Lin, Xuebo Liu, Xiaopeng Ke, Lian Lian, Dong Jin, Shengjun Cheng, Jun Yu, Min Zhang

Large Language Models (LLMs) often require domain-specific fine-tuning to
address targeted tasks, which risks degrading their general capabilities.
Maintaining a balance between domain-specific enhancements and general model
utility is a key challenge. This paper proposes a novel approach named APT
(Weakness Case Acquisition and Iterative Preference Training) to enhance
domain-specific performance with self-generated dis-preferred weakness data
(bad cases and similar cases). APT uniquely focuses on training the model using
only those samples where errors occur, alongside a small, similar set of
samples retrieved for this purpose. This targeted training minimizes
interference with the model's existing knowledge base, effectively retaining
generic capabilities. Experimental results on the LLama-2 and Mistral-V0.3
models across various benchmarks demonstrate that APT ensures no reduction in
generic capacity and achieves superior performance on downstream tasks compared
to various existing methods. This validates our method as an effective strategy
for enhancing domain-specific capabilities without sacrificing the model's
broader applicability.

### 3. [Beyond Memorization: A Rigorous Evaluation Framework for Medical Knowledge Editing](http://arxiv.org/pdf/2506.03490v1)

Authors: Shigeng Chen, Linhao Luo, Zhangchi Qiu, Yanan Cao, Carl Yang, Shirui Pan

Recently, knowledge editing (KE) has emerged as a promising approach to
update specific facts in Large Language Models (LLMs) without the need for full
retraining. Despite the effectiveness in general-domain benchmarks, their
applicability to complex medical domain remains largely unexplored. Medical
knowledge editing is particularly challenging, as it requires LLMs to
internalize the knowledge and generalize to unseen scenarios for effective and
interpretable decision-making. In this work, we propose a novel framework
called MedEditBench to rigorously evaluate the effectiveness of existing KE
methods in the medical domain. In MedEditBench, we introduce a new medical
knowledge editing benchmark as well as three different knowledge editing
paradigms, which are designed to assess the impact of different knowledge
sources for editing. Our findings indicate that current KE methods result in
only superficial memorization of the injected information, failing to
generalize to new scenarios. To overcome this limitation, we present
Self-Generated Rationale Editing (SGR-Edit), which utilizes model-derived
rationales as the target knowledge for editing, thereby uncovering the
underlying reasoning process and demonstrating significant improvements over
existing KE approaches. Additionally, we offer deeper insights into medical
knowledge editing, including the localization of medical knowledge in LLMs and
the impact of sequential editing on evolving knowledge. This could provide
practical guidance for implementing KE methods in real-world medical
applications.

### 4. [Accurate Sublayer Pruning for Large Language Models by Exploiting Latency and Tunability Information](http://arxiv.org/pdf/2506.03510v1)

Authors: Seungcheol Park, Sojin Lee, Jongjin Kim, Jinsik Lee, Hyunjik Jo, U Kang

How can we accelerate large language models(LLMs) without sacrificing
accuracy? The slow inference speed of LLMs hinders us to benefit from their
remarkable performance in diverse applications. This is mainly because numerous
sublayers are stacked together in LLMs. Sublayer pruning compresses and
expedites LLMs via removing unnecessary sublayers. However, existing sublayer
pruning algorithms are limited in accuracy since they naively select sublayers
to prune, overlooking the different characteristics of each sublayer. In this
paper, we propose SPRINT (Sublayer PRuning wIth LateNcy and Tunability
Information), an accurate sublayer pruning method for LLMs. SPRINT accurately
selects a target sublayer to prune by considering 1) the amount of latency
reduction after pruning and 2) the tunability of sublayers. SPRINT iteratively
prunes redundant sublayers and swiftly tunes the parameters of remaining
sublayers. Experiments show that SPRINT achieves the best accuracy-speedup
trade-off, exhibiting up to 23.88%p higher accuracy on zero-shot commonsense
reasoning benchmarks compared to existing pruning algorithms.

### 5. [An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals](http://arxiv.org/pdf/2506.03519v1)

Authors: Yangyang Zhao, Ben Niu, Libo Qin, Shihan Wang

Deep Reinforcement Learning (DRL) is widely used in task-oriented dialogue
systems to optimize dialogue policy, but it struggles to balance exploration
and exploitation due to the high dimensionality of state and action spaces.
This challenge often results in local optima or poor convergence. Evolutionary
Algorithms (EAs) have been proven to effectively explore the solution space of
neural networks by maintaining population diversity. Inspired by this, we
innovatively combine the global search capabilities of EA with the local
optimization of DRL to achieve a balance between exploration and exploitation.
Nevertheless, the inherent flexibility of natural language in dialogue tasks
complicates this direct integration, leading to prolonged evolutionary times.
Thus, we further propose an elite individual injection mechanism to enhance
EA's search efficiency by adaptively introducing best-performing individuals
into the population. Experiments across four datasets show that our approach
significantly improves the balance between exploration and exploitation,
boosting performance. Moreover, the effectiveness of the EII mechanism in
reducing exploration time has been demonstrated, achieving an efficient
integration of EA and DRL on task-oriented dialogue policy tasks.

### 6. [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](http://arxiv.org/pdf/2506.03523v1)

Authors: Chong Li, Jiajun Zhang, Chengqing Zong

Tokenization serves as a foundational step for Large Language Models (LLMs)
to process text. In new domains or languages, the inefficiency of the tokenizer
will slow down the training and generation of LLM. The mismatch in vocabulary
also hinders deep knowledge transfer between LLMs like token-level
distillation. To mitigate this gap, we propose an efficient method named
TokAlign to replace the vocabulary of LLM from the token co-occurrences view,
and further transfer the token-level knowledge between models. It first aligns
the source vocabulary to the target one by learning a one-to-one mapping matrix
for token IDs. Model parameters, including embeddings, are rearranged and
progressively fine-tuned for the new vocabulary. Our method significantly
improves multilingual text compression rates and vocabulary initialization for
LLMs, decreasing the perplexity from 3.4$\text{e}^2$ of strong baseline methods
to 1.2$\text{e}^2$ after initialization. Experimental results on models across
multiple parameter scales demonstrate the effectiveness and generalization of
TokAlign, which costs as few as 5k steps to restore the performance of the
vanilla model. After unifying vocabularies between LLMs, token-level
distillation can remarkably boost (+4.4% than sentence-level distillation) the
base model, costing only 235M tokens.

### 7. [Seed-Coder: Let the Code Model Curate Data for Itself](http://arxiv.org/pdf/2506.03524v1)

Authors: Yuyu Zhang, Jing Su, Yifan Sun, Chenguang Xi, Xia Xiao, Shen Zheng, Anxiang Zhang, Kaibo Liu, Daoguang Zan, Tao Sun, Jinhua Zhu, Shulin Xin, Dong Huang, Yetao Bai, Lixin Dong, Chao Li, Jianchong Chen, Hanzhi Zhou, Yifan Huang, Guanghan Ning, Xierui Song, Jiaze Chen, Siyao Liu, Kai Shen, Liang Xiang, Yonghui Wu

Code data in large language model (LLM) pretraining is recognized crucial not
only for code-related tasks but also for enhancing general intelligence of
LLMs. Current open-source LLMs often heavily rely on human effort to produce
their code pretraining data, such as employing hand-crafted filtering rules
tailored to individual programming languages, or using human-annotated data to
train quality filters. However, these approaches are inherently limited in
scalability, prone to subjective biases, and costly to extend and maintain
across diverse programming languages. To address these challenges, we introduce
Seed-Coder, a series of open-source LLMs comprising base, instruct and
reasoning models of 8B size, minimizing human involvement in data construction.
Our code pretraining data is produced by a model-centric data pipeline, which
predominantly leverages LLMs for scoring and filtering code data. The instruct
model is further trained via supervised fine-tuning and preference
optimization, and the reasoning model leverages Long-Chain-of-Thought (LongCoT)
reinforcement learning to improve multi-step code reasoning. Seed-Coder
achieves state-of-the-art results among open-source models of similar size and
even surpasses some much larger models, demonstrating superior performance in
code generation, code completion, code editing, code reasoning, and software
engineering tasks.

### 8. [Go-Browse: Training Web Agents with Structured Exploration](http://arxiv.org/pdf/2506.03533v1)

Authors: Apurva Gandhi, Graham Neubig

One of the fundamental problems in digital agents is their lack of
understanding of their environment. For instance, a web browsing agent may get
lost in unfamiliar websites, uncertain what pages must be visited to achieve
its goals. To address this, we propose Go-Browse, a method for automatically
collecting diverse and realistic web agent data at scale through structured
exploration of web environments. Go-Browse achieves efficient exploration by
framing data collection as a graph search, enabling reuse of information across
exploration episodes. We instantiate our method on the WebArena benchmark,
collecting a dataset of 10K successful task-solving trajectories and 40K
interaction steps across 100 URLs. Fine-tuning a 7B parameter language model on
this dataset achieves a success rate of 21.7% on the WebArena benchmark,
beating GPT-4o mini by 2.4% and exceeding current state-of-the-art results for
sub-10B parameter models by 2.9%.

### 9. [BPO: Revisiting Preference Modeling in Direct Preference Optimization](http://arxiv.org/pdf/2506.03557v1)

Authors: Lin Sun, Chuang Liu, Peng Liu, Bingyang Li, Weijia Lu, Ning Wu

Direct Preference Optimization (DPO) have emerged as a popular method for
aligning Large Language Models (LLMs) with human preferences. While DPO
effectively preserves the relative ordering between chosen and rejected
responses through pairwise ranking losses, it often neglects absolute reward
magnitudes. This oversight can decrease the likelihood of chosen responses and
increase the risk of generating out-of-distribution responses, leading to poor
performance. We term this issue Degraded Chosen Responses (DCR).To address this
issue, we propose Balanced Preference Optimization (BPO), a novel framework
that dynamically balances the optimization of chosen and rejected responses
through two key components: balanced reward margin and gap adaptor. Unlike
previous methods, BPO can fundamentally resolve DPO's DCR issue, without
introducing additional constraints to the loss function. Experimental results
on multiple mathematical reasoning tasks show that BPO significantly
outperforms DPO, improving accuracy by +10.1% with Llama-3.1-8B-Instruct (18.8%
to 28.9%) and +11.7% with Qwen2.5-Math-7B (35.0% to 46.7%). It also surpasses
DPO variants by +3.6% over IPO (43.1%), +5.0% over SLiC (41.7%), and +3.1% over
Cal-DPO (43.6%) on the same model. Remarkably, our algorithm requires only a
single line of code modification, making it simple to implement and fully
compatible with existing DPO-based frameworks.

### 10. [ConsistentChat: Building Skeleton-Guided Consistent Dialogues for Large Language Models from Scratch](http://arxiv.org/pdf/2506.03558v1)

Authors: Jiawei Chen, Xinyan Guan, Qianhao Yuan, Guozhao Mo, Weixiang Zhou, Yaojie Lu, Hongyu Lin, Ben He, Le Sun, Xianpei Han

Current instruction data synthesis methods primarily focus on single-turn
instructions and often neglect cross-turn coherence, resulting in context drift
and reduced task completion rates in extended conversations. To address this
limitation, we propose Skeleton-Guided Multi-Turn Dialogue Generation, a
framework that constrains multi-turn instruction synthesis by explicitly
modeling human conversational intent. It operates in two stages: (1) Intent
Modeling, which captures the global structure of human dialogues by assigning
each conversation to one of nine well-defined intent trajectories, ensuring a
coherent and goal-oriented information flow; and (2) Skeleton Generation, which
constructs a structurally grounded sequence of user queries aligned with the
modeled intent, thereby serving as a scaffold that constrains and guides the
downstream instruction synthesis process. Based on this process, we construct
ConsistentChat, a multi-turn instruction dataset with approximately 15,000
multi-turn conversations and 224,392 utterances. Experiments on the Light,
Topdial, and MT-Eval benchmarks show that models fine-tuned on ConsistentChat
achieve a 20-30% improvement in chat consistency and up to a 15% increase in
task success rate, significantly outperforming models trained on existing
single-turn and multi-turn instruction datasets.

### Cryptography and Security

### 1. [A Threat Intelligence Event Extraction Conceptual Model for Cyber Threat Intelligence Feeds](http://arxiv.org/pdf/2506.03551v1)

Authors: Jamal H. Al-Yasiri, Mohamad Fadli Bin Zolkipli, Nik Fatinah N Mohd Farid, Mohammed Alsamman, Zainab Ali Mohammed

In response to the escalating cyber threats, the efficiency of Cyber Threat
Intelligence (CTI) data collection has become paramount in ensuring robust
cybersecurity. However, existing works encounter significant challenges in
preprocessing large volumes of multilingual threat data, leading to
inefficiencies in real-time threat analysis. This paper presents a systematic
review of current techniques aimed at enhancing CTI data collection efficiency.
Additionally, it proposes a conceptual model to further advance the
effectiveness of threat intelligence feeds. Following the PRISMA guidelines,
the review examines relevant studies from the Scopus database, highlighting the
critical role of artificial intelligence (AI) and machine learning models in
optimizing CTI data preprocessing. The findings underscore the importance of
AI-driven methods, particularly supervised and unsupervised learning, in
significantly improving the accuracy of threat detection and event extraction,
thereby strengthening cybersecurity. Furthermore, the study identifies a gap in
the existing research and introduces XBC conceptual model integrating
XLM-RoBERTa, BiGRU, and CRF, specifically developed to address this gap. This
paper contributes conceptually to the field by providing a detailed analysis of
current CTI data collection techniques and introducing an innovative conceptual
model to enhance future threat intelligence capabilities.

### 2. [Client-Side Zero-Shot LLM Inference for Comprehensive In-Browser URL Analysis](http://arxiv.org/pdf/2506.03656v1)

Authors: Avihay Cohen

Malicious websites and phishing URLs pose an ever-increasing cybersecurity
risk, with phishing attacks growing by 40% in a single year. Traditional
detection approaches rely on machine learning classifiers or rule-based
scanners operating in the cloud, but these face significant challenges in
generalization, privacy, and evasion by sophisticated threats. In this paper,
we propose a novel client-side framework for comprehensive URL analysis that
leverages zero-shot inference by a local large language model (LLM) running
entirely in-browser. Our system uses a compact LLM (e.g., 3B/8B parameters) via
WebLLM to perform reasoning over rich context collected from the target
webpage, including static code analysis (JavaScript abstract syntax trees,
structure, and code patterns), dynamic sandbox execution results (DOM changes,
API calls, and network requests),and visible content. We detail the
architecture and methodology of the system, which combines a real browser
sandbox (using iframes) resistant to common anti-analysis techniques, with an
LLM-based analyzer that assesses potential vulnerabilities and malicious
behaviors without any task-specific training (zero-shot). The LLM aggregates
evidence from multiple sources (code, execution trace, page content) to
classify the URL as benign or malicious and to provide an explanation of the
threats or security issues identified. We evaluate our approach on a diverse
set of benign and malicious URLs, demonstrating that even a compact client-side
model can achieve high detection accuracy and insightful explanations
comparable to cloud-based solutions, while operating privately on end-user
devices. The results show that client-side LLM inference is a feasible and
effective solution to web threat analysis, eliminating the need to send
potentially sensitive data to cloud services.

### 3. [Prediction Inconsistency Helps Achieve Generalizable Detection of Adversarial Examples](http://arxiv.org/pdf/2506.03765v1)

Authors: Sicong Han, Chenhao Lin, Zhengyu Zhao, Xiyuan Wang, Xinlei He, Qian Li, Cong Wang, Qian Wang, Chao Shen

Adversarial detection protects models from adversarial attacks by refusing
suspicious test samples. However, current detection methods often suffer from
weak generalization: their effectiveness tends to degrade significantly when
applied to adversarially trained models rather than naturally trained ones, and
they generally struggle to achieve consistent effectiveness across both
white-box and black-box attack settings. In this work, we observe that an
auxiliary model, differing from the primary model in training strategy or model
architecture, tends to assign low confidence to the primary model's predictions
on adversarial examples (AEs), while preserving high confidence on normal
examples (NEs). Based on this discovery, we propose Prediction Inconsistency
Detector (PID), a lightweight and generalizable detection framework to
distinguish AEs from NEs by capturing the prediction inconsistency between the
primal and auxiliary models. PID is compatible with both naturally and
adversarially trained primal models and outperforms four detection methods
across 3 white-box, 3 black-box, and 1 mixed adversarial attacks. Specifically,
PID achieves average AUC scores of 99.29\% and 99.30\% on CIFAR-10 when the
primal model is naturally and adversarially trained, respectively, and 98.31%
and 96.81% on ImageNet under the same conditions, outperforming existing SOTAs
by 4.70%$\sim$25.46%.

### 4. [Software Bill of Materials in Software Supply Chain Security A Systematic Literature Review](http://arxiv.org/pdf/2506.03507v1)

Authors: Eric O'Donoghue, Yvette Hastings, Ernesto Ortiz, A. Redempta Manzi Muneza

Software Bill of Materials (SBOMs) are increasingly regarded as essential
tools for securing software supply chains (SSCs), yet their real-world use and
adoption barriers remain poorly understood. This systematic literature review
synthesizes evidence from 40 peer-reviewed studies to evaluate how SBOMs are
currently used to bolster SSC security. We identify five primary application
areas: vulnerability management, transparency, component assessment, risk
assessment, and SSC integrity. Despite clear promise, adoption is hindered by
significant barriers: generation tooling, data privacy, format/standardization,
sharing/distribution, cost/overhead, vulnerability exploitability, maintenance,
analysis tooling, false positives, hidden packages, and tampering. To structure
our analysis, we map these barriers to the ISO/IEC 25019:2023 Quality-in-Use
model, revealing critical deficiencies in SBOM trustworthiness, usability, and
suitability for security tasks. We also highlight key gaps in the literature.
These include the absence of applying machine learning techniques to assess
SBOMs and limited evaluation of SBOMs and SSCs using software quality assurance
techniques. Our findings provide actionable insights for researchers, tool
developers, and practitioners seeking to advance SBOM-driven SSC security and
lay a foundation for future work at the intersection of SSC assurance,
automation, and empirical software engineering.

### 5. [Quantum Secure Key Exchange with Position-based Credentials](http://arxiv.org/pdf/2506.03549v1)

Authors: Wen Yu Kon, Ignatius William Primaatmaja, Kaushik Chakraborty, Charles Lim

Quantum key distribution (QKD) provides an information-theoretic way of
securely exchanging secret keys, and typically relies on pre-shared keys or
public keys for message authentication. To lift the requirement of pre-shared
or public keys, Buhrman et. al. [SIAM J. Comput. 43, 150 (2014)] proposed
utilizing the location of a party as a credential. Here, we extend upon the
proposal, develop a QKD protocol with location credentials using quantum
position verification (QPV) based message and identity authentication. By using
QKD with delayed authentication as a base, and later simplifying QPV-based
message authentication, we significantly reduce the number of QPV runs, which
currently acts as a bottleneck. Besides demonstrating security for the proposed
protocol, we also provide improvements to QPV security analysis, including
generalization of the QPV adversary model, tightening a trace distance bound
using semidefinite programming, and propose a multi-basis QPV requiring only
BB84 state preparation but with multiple measurement basis.

### 6. [Mono: Is Your "Clean" Vulnerability Dataset Really Solvable? Exposing and Trapping Undecidable Patches and Beyond](http://arxiv.org/pdf/2506.03651v1)

Authors: Zeyu Gao, Junlin Zhou, Bolun Zhang, Yi He, Chao Zhang, Yuxin Cui, Hao Wang

The quantity and quality of vulnerability datasets are essential for
developing deep learning solutions to vulnerability-related tasks. Due to the
limited availability of vulnerabilities, a common approach to building such
datasets is analyzing security patches in source code. However, existing
security patches often suffer from inaccurate labels, insufficient contextual
information, and undecidable patches that fail to clearly represent the root
causes of vulnerabilities or their fixes. These issues introduce noise into the
dataset, which can mislead detection models and undermine their effectiveness.
To address these issues, we present mono, a novel LLM-powered framework that
simulates human experts' reasoning process to construct reliable vulnerability
datasets. mono introduces three key components to improve security patch
datasets: (i) semantic-aware patch classification for precise vulnerability
labeling, (ii) iterative contextual analysis for comprehensive code
understanding, and (iii) systematic root cause analysis to identify and filter
undecidable patches. Our comprehensive evaluation on the MegaVul benchmark
demonstrates that mono can correct 31.0% of labeling errors, recover 89% of
inter-procedural vulnerabilities, and reveals that 16.7% of CVEs contain
undecidable patches. Furthermore, mono's enriched context representation
improves existing models' vulnerability detection accuracy by 15%. We open
source the framework mono and the dataset MonoLens in
https://github.com/vul337/mono.

### 7. [Evaluating Apple Intelligence's Writing Tools for Privacy Against Large Language Model-Based Inference Attacks: Insights from Early Datasets](http://arxiv.org/pdf/2506.03870v1)

Authors: Mohd. Farhan Israk Soumik, Syed Mhamudul Hasan, Abdur R. Shahid

The misuse of Large Language Models (LLMs) to infer emotions from text for
malicious purposes, known as emotion inference attacks, poses a significant
threat to user privacy. In this paper, we investigate the potential of Apple
Intelligence's writing tools, integrated across iPhone, iPad, and MacBook, to
mitigate these risks through text modifications such as rewriting and tone
adjustment. By developing early novel datasets specifically for this purpose,
we empirically assess how different text modifications influence LLM-based
detection. This capability suggests strong potential for Apple Intelligence's
writing tools as privacy-preserving mechanisms. Our findings lay the groundwork
for future adaptive rewriting systems capable of dynamically neutralizing
sensitive emotional content to enhance user privacy. To the best of our
knowledge, this research provides the first empirical analysis of Apple
Intelligence's text-modification tools within a privacy-preservation context
with the broader goal of developing on-device, user-centric privacy-preserving
mechanisms to protect against LLMs-based advanced inference attacks on deployed
systems.

### 8. [Depermissioning Web3: a Permissionless Accountable RPC Protocol for Blockchain Networks](http://arxiv.org/pdf/2506.03940v1)

Authors: Weihong Wang, Tom Van Cutsem

In blockchain networks, so-called "full nodes" serve data to and relay
transactions from clients through an RPC interface. This serving layer enables
integration of "Web3" data, stored on blockchains, with "Web2" mobile or web
applications that cannot directly participate as peers in a blockchain network.
In practice, the serving layer is dominated by a small number of centralized
services ("node providers") that offer permissioned access to RPC endpoints.
Clients register with these providers because they offer reliable and
convenient access to blockchain data: operating a full node themselves requires
significant computational and storage resources, and public (permissionless)
RPC nodes lack financial incentives to serve large numbers of clients with
consistent performance.
  Permissioned access to an otherwise permissionless blockchain network raises
concerns regarding the privacy, integrity, and availability of data access. To
address this, we propose a Permissionless Accountable RPC Protocol (PARP). It
enables clients and full nodes to interact pseudonymously while keeping both
parties accountable. PARP leverages "light client" schemes for essential data
integrity checks, combined with fraud proofs, to keep full nodes honest and
accountable. It integrates payment channels to facilitate micro-payments,
holding clients accountable for the resources they consume and providing an
economic incentive for full nodes to serve. Our prototype implementation for
Ethereum demonstrates the feasibility of PARP, and we quantify its overhead
compared to the base RPC protocol.

### 9. [Privacy and Security Threat for OpenAI GPTs](http://arxiv.org/pdf/2506.04036v1)

Authors: Wei Wenying, Zhao Kaifa, Xue Lei, Fan Ming

Large language models (LLMs) demonstrate powerful information handling
capabilities and are widely integrated into chatbot applications. OpenAI
provides a platform for developers to construct custom GPTs, extending
ChatGPT's functions and integrating external services. Since its release in
November 2023, over 3 million custom GPTs have been created. However, such a
vast ecosystem also conceals security and privacy threats. For developers,
instruction leaking attacks threaten the intellectual property of instructions
in custom GPTs through carefully crafted adversarial prompts. For users,
unwanted data access behavior by custom GPTs or integrated third-party services
raises significant privacy concerns. To systematically evaluate the scope of
threats in real-world LLM applications, we develop three phases instruction
leaking attacks target GPTs with different defense level. Our widespread
experiments on 10,000 real-world custom GPTs reveal that over 98.8% of GPTs are
vulnerable to instruction leaking attacks via one or more adversarial prompts,
and half of the remaining GPTs can also be attacked through multiround
conversations. We also developed a framework to assess the effectiveness of
defensive strategies and identify unwanted behaviors in custom GPTs. Our
findings show that 77.5% of custom GPTs with defense strategies are vulnerable
to basic instruction leaking attacks. Additionally, we reveal that 738 custom
GPTs collect user conversational information, and identified 8 GPTs exhibiting
data access behaviors that are unnecessary for their intended functionalities.
Our findings raise awareness among GPT developers about the importance of
integrating specific defensive strategies in their instructions and highlight
users' concerns about data privacy when using LLM-based applications.

### 10. [Dropout-Robust Mechanisms for Differentially Private and Fully Decentralized Mean Estimation](http://arxiv.org/pdf/2506.03746v1)

Authors: César Sabater, Sonia Ben Mokhtar, Jan Ramon

Achieving differentially private computations in decentralized settings poses
significant challenges, particularly regarding accuracy, communication cost,
and robustness against information leakage. While cryptographic solutions offer
promise, they often suffer from high communication overhead or require
centralization in the presence of network failures. Conversely, existing fully
decentralized approaches typically rely on relaxed adversarial models or
pairwise noise cancellation, the latter suffering from substantial accuracy
degradation if parties unexpectedly disconnect. In this work, we propose IncA,
a new protocol for fully decentralized mean estimation, a widely used primitive
in data-intensive processing. Our protocol, which enforces differential
privacy, requires no central orchestration and employs low-variance correlated
noise, achieved by incrementally injecting sensitive information into the
computation. First, we theoretically demonstrate that, when no parties
permanently disconnect, our protocol achieves accuracy comparable to that of a
centralized setting-already an improvement over most existing decentralized
differentially private techniques. Second, we empirically show that our use of
low-variance correlated noise significantly mitigates the accuracy loss
experienced by existing techniques in the presence of dropouts.

### Computer Vision and Pattern Recognition

### 1. [MamFusion: Multi-Mamba with Temporal Fusion for Partially Relevant Video Retrieval](http://arxiv.org/pdf/2506.03473v1)

Authors: Xinru Ying, Jiaqi Mo, Jingyang Lin, Canghong Jin, Fangfang Wang, Lina Wei

Partially Relevant Video Retrieval (PRVR) is a challenging task in the domain
of multimedia retrieval. It is designed to identify and retrieve untrimmed
videos that are partially relevant to the provided query. In this work, we
investigate long-sequence video content understanding to address information
redundancy issues. Leveraging the outstanding long-term state space modeling
capability and linear scalability of the Mamba module, we introduce a
multi-Mamba module with temporal fusion framework (MamFusion) tailored for PRVR
task. This framework effectively captures the state-relatedness in long-term
video content and seamlessly integrates it into text-video relevance
understanding, thereby enhancing the retrieval process. Specifically, we
introduce Temporal T-to-V Fusion and Temporal V-to-T Fusion to explicitly model
temporal relationships between text queries and video moments, improving
contextual awareness and retrieval accuracy. Extensive experiments conducted on
large-scale datasets demonstrate that MamFusion achieves state-of-the-art
performance in retrieval effectiveness. Code is available at the link:
https://github.com/Vision-Multimodal-Lab-HZCU/MamFusion.

### 2. [Heterogeneous Skeleton-Based Action Representation Learning](http://arxiv.org/pdf/2506.03481v1)

Authors: Hongsong Wang, Xiaoyan Ma, Jidong Kuang, Jie Gui

Skeleton-based human action recognition has received widespread attention in
recent years due to its diverse range of application scenarios. Due to the
different sources of human skeletons, skeleton data naturally exhibit
heterogeneity. The previous works, however, overlook the heterogeneity of human
skeletons and solely construct models tailored for homogeneous skeletons. This
work addresses the challenge of heterogeneous skeleton-based action
representation learning, specifically focusing on processing skeleton data that
varies in joint dimensions and topological structures. The proposed framework
comprises two primary components: heterogeneous skeleton processing and unified
representation learning. The former first converts two-dimensional skeleton
data into three-dimensional skeleton via an auxiliary network, and then
constructs a prompted unified skeleton using skeleton-specific prompts. We also
design an additional modality named semantic motion encoding to harness the
semantic information within skeletons. The latter module learns a unified
action representation using a shared backbone network that processes different
heterogeneous skeletons. Extensive experiments on the NTU-60, NTU-120, and
PKU-MMD II datasets demonstrate the effectiveness of our method in various
tasks of action understanding. Our approach can be applied to action
recognition in robots with different humanoid structures.

### 3. [EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](http://arxiv.org/pdf/2506.03512v1)

Authors: Daikun Liu, Lei Cheng, Teng Wang, changyin Sun

Recent learning-based methods for event-based optical flow estimation utilize
cost volumes for pixel matching but suffer from redundant computations and
limited scalability to higher resolutions for flow refinement. In this work, we
take advantage of the complementarity between temporally dense feature
differences of adjacent event frames and cost volume and present a lightweight
event-based optical flow network (EDCFlow) to achieve high-quality flow
estimation at a higher resolution. Specifically, an attention-based multi-scale
temporal feature difference layer is developed to capture diverse motion
patterns at high resolution in a computation-efficient manner. An adaptive
fusion of high-resolution difference motion features and low-resolution
correlation motion features is performed to enhance motion representation and
model generalization. Notably, EDCFlow can serve as a plug-and-play refinement
module for RAFT-like event-based methods to enhance flow details. Extensive
experiments demonstrate that EDCFlow achieves better performance with lower
complexity compared to existing methods, offering superior generalization.

### 4. [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](http://arxiv.org/pdf/2506.03517v1)

Authors: Ziyi Wu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Ashkan Mirzaei, Igor Gilitschenski, Sergey Tulyakov, Aliaksandr Siarohin

Direct Preference Optimization (DPO) has recently been applied as a
post-training technique for text-to-video diffusion models. To obtain training
data, annotators are asked to provide preferences between two videos generated
from independent noise. However, this approach prohibits fine-grained
comparisons, and we point out that it biases the annotators towards low-motion
clips as they often contain fewer visual artifacts. In this work, we introduce
DenseDPO, a method that addresses these shortcomings by making three
contributions. First, we create each video pair for DPO by denoising corrupted
copies of a ground truth video. This results in aligned pairs with similar
motion structures while differing in local details, effectively neutralizing
the motion bias. Second, we leverage the resulting temporal alignment to label
preferences on short segments rather than entire clips, yielding a denser and
more precise learning signal. With only one-third of the labeled data, DenseDPO
greatly improves motion generation over vanilla DPO, while matching it in text
alignment, visual quality, and temporal consistency. Finally, we show that
DenseDPO unlocks automatic preference annotation using off-the-shelf Vision
Language Models (VLMs): GPT accurately predicts segment-level preferences
similar to task-specifically fine-tuned video reward models, and DenseDPO
trained on these labels achieves performance close to using human labels.

### 5. [Target Semantics Clustering via Text Representations for Robust Universal Domain Adaptation](http://arxiv.org/pdf/2506.03521v1)

Authors: Weinan He, Zilei Wang, Yixin Zhang

Universal Domain Adaptation (UniDA) focuses on transferring source domain
knowledge to the target domain under both domain shift and unknown category
shift. Its main challenge lies in identifying common class samples and aligning
them. Current methods typically obtain target domain semantics centers from an
unconstrained continuous image representation space. Due to domain shift and
the unknown number of clusters, these centers often result in complex and less
robust alignment algorithm. In this paper, based on vision-language models, we
search for semantic centers in a semantically meaningful and discrete text
representation space. The constrained space ensures almost no domain bias and
appropriate semantic granularity for these centers, enabling a simple and
robust adaptation algorithm. Specifically, we propose TArget Semantics
Clustering (TASC) via Text Representations, which leverages information
maximization as a unified objective and involves two stages. First, with the
frozen encoders, a greedy search-based framework is used to search for an
optimal set of text embeddings to represent target semantics. Second, with the
search results fixed, encoders are refined based on gradient descent,
simultaneously achieving robust domain alignment and private class clustering.
Additionally, we propose Universal Maximum Similarity (UniMS), a scoring
function tailored for detecting open-set samples in UniDA. Experimentally, we
evaluate the universality of UniDA algorithms under four category shift
scenarios. Extensive experiments on four benchmarks demonstrate the
effectiveness and robustness of our method, which has achieved state-of-the-art
performance.

### 6. [Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](http://arxiv.org/pdf/2506.03538v1)

Authors: Chengqi Li, Zhihao Shi, Yangdi Lu, Wenbo He, Xiangyu Xu

3D reconstruction from in-the-wild images remains a challenging task due to
inconsistent lighting conditions and transient distractors. Existing methods
typically rely on heuristic strategies to handle the low-quality training data,
which often struggle to produce stable and consistent reconstructions,
frequently resulting in visual artifacts. In this work, we propose Asymmetric
Dual 3DGS, a novel framework that leverages the stochastic nature of these
artifacts: they tend to vary across different training runs due to minor
randomness. Specifically, our method trains two 3D Gaussian Splatting (3DGS)
models in parallel, enforcing a consistency constraint that encourages
convergence on reliable scene geometry while suppressing inconsistent
artifacts. To prevent the two models from collapsing into similar failure modes
due to confirmation bias, we introduce a divergent masking strategy that
applies two complementary masks: a multi-cue adaptive mask and a
self-supervised soft mask, which leads to an asymmetric training process of the
two models, reducing shared error modes. In addition, to improve the efficiency
of model training, we introduce a lightweight variant called Dynamic EMA Proxy,
which replaces one of the two models with a dynamically updated Exponential
Moving Average (EMA) proxy, and employs an alternating masking strategy to
preserve divergence. Extensive experiments on challenging real-world datasets
demonstrate that our method consistently outperforms existing approaches while
achieving high efficiency. Codes and trained models will be released.

### 7. [WIFE-Fusion:Wavelet-aware Intra-inter Frequency Enhancement for Multi-model Image Fusion](http://arxiv.org/pdf/2506.03555v1)

Authors: Tianpei Zhang, Jufeng Zhao, Yiming Zhu, Guangmang Cui

Multimodal image fusion effectively aggregates information from diverse
modalities, with fused images playing a crucial role in vision systems.
However, existing methods often neglect frequency-domain feature exploration
and interactive relationships. In this paper, we propose wavelet-aware
Intra-inter Frequency Enhancement Fusion (WIFE-Fusion), a multimodal image
fusion framework based on frequency-domain components interactions. Its core
innovations include: Intra-Frequency Self-Attention (IFSA) that leverages
inherent cross-modal correlations and complementarity through interactive
self-attention mechanisms to extract enriched frequency-domain features, and
Inter-Frequency Interaction (IFI) that enhances enriched features and filters
latent features via combinatorial interactions between heterogeneous
frequency-domain components across modalities. These processes achieve precise
source feature extraction and unified modeling of feature
extraction-aggregation. Extensive experiments on five datasets across three
multimodal fusion tasks demonstrate WIFE-Fusion's superiority over current
specialized and unified fusion methods. Our code is available at
https://github.com/Lmmh058/WIFE-Fusion.

### 8. [A Large-Scale Referring Remote Sensing Image Segmentation Dataset and Benchmark](http://arxiv.org/pdf/2506.03583v1)

Authors: Zhigang Yang, Huiguang Yao, Linmao Tian, Xuezhi Zhao, Qiang Li, Qi Wang

Referring Remote Sensing Image Segmentation is a complex and challenging task
that integrates the paradigms of computer vision and natural language
processing. Existing datasets for RRSIS suffer from critical limitations in
resolution, scene diversity, and category coverage, which hinders the
generalization and real-world applicability of refer segmentation models. To
facilitate the development of this field, we introduce NWPU-Refer, the largest
and most diverse RRSIS dataset to date, comprising 15,003 high-resolution
images (1024-2048px) spanning 30+ countries with 49,745 annotated targets
supporting single-object, multi-object, and non-object segmentation scenarios.
Additionally, we propose the Multi-scale Referring Segmentation Network
(MRSNet), a novel framework tailored for the unique demands of RRSIS. MRSNet
introduces two key innovations: (1) an Intra-scale Feature Interaction Module
(IFIM) that captures fine-grained details within each encoder stage, and (2) a
Hierarchical Feature Interaction Module (HFIM) to enable seamless cross-scale
feature fusion, preserving spatial integrity while enhancing discriminative
power. Extensive experiments conducte on the proposed NWPU-Refer dataset
demonstrate that MRSNet achieves state-of-the-art performance across multiple
evaluation metrics, validating its effectiveness. The dataset and code are
publicly available at https://github.com/CVer-Yang/NWPU-Refer.

### 9. [Resolving Task Objective Conflicts in Unified Multimodal Understanding and Generation via Task-Aware Mixture-of-Experts](http://arxiv.org/pdf/2506.03591v1)

Authors: Jiaxing Zhang, Xinyi Zeng, Hao Tang

Unified multimodal large language models (MLLMs) based on end-to-end
autoregressive (AR) transformers effectively integrate both understanding and
generation tasks within a single framework. However, intrinsic Task Objective
Conflicts between high-level semantic abstraction in understanding and
fine-grained detail preservation in generation pose significant challenges,
often leading to suboptimal trade-offs and task interference. Existing
solutions, such as decoupling shared visual encoders, fall short of
fundamentally resolving these conflicts due to inherent AR architecture. In
this paper, we propose a novel approach that decouples internal components of
AR to resolve task objective conflicts. Specifically, we design UTAMoE, a
Unified Task-Aware Mixture-of-Experts (MoE) framework that decouples internal
AR modules via a Task-Aware MoE Layer to create task-specific optimization
subpaths. To enhance task differentiation while maintaining overall
coordination, we introduce a novel Two-Stage Training Strategy. Extensive
experiments on multimodal benchmarks demonstrate that UTAMoE mitigates task
objective conflicts, achieving state-of-the-art performance across various
tasks. Visualizations and ablation studies further validate the effectiveness
of our approach.

### 10. [ControlThinker: Unveiling Latent Semantics for Controllable Image Generation through Visual Reasoning](http://arxiv.org/pdf/2506.03596v1)

Authors: Feng Han, Yang Jiao, Shaoxiang Chen, Junhao Xu, Jingjing Chen, Yu-Gang Jiang

The field of controllable image generation has seen significant advancements,
with various architectures improving generation layout consistency with control
signals. However, contemporary methods still face challenges in bridging the
semantic gap between input text prompts with sparse semantics and the target
images, often over-relying on low-level control signals to infer regional
details. To address this challenge, we propose ControlThinker, a novel
framework that employs a "comprehend-then-generate" paradigm. Firstly, by
incentivizing the visual reasoning capability of a MLLM, latent semantics from
control images are mined to enrich text prompts. This enriched semantic
understanding then seamlessly aids in image generation without the need for
additional complex modifications. To further tackle the uncertainty arising
from the ambiguity of control images, we encourage broader exploration of
reasoning trajectories and select the optimal one using a metric-based output
reward model (ORM). Extensive experimental results demonstrate that
ControlThinker effectively mitigates the semantic gap between raw text prompts
and target images, resulting in improved visual quality and semantic
consistency across a wide range of benchmarks. The code and models are
available at https://github.com/Maplebb/ControlThinker.

### Computers and Society

### 1. [Bridging the Artificial Intelligence Governance Gap: The United States' and China's Divergent Approaches to Governing General-Purpose Artificial Intelligence](http://arxiv.org/pdf/2506.03497v1)

Authors: Oliver Guest, Kevin Wei

The United States and China are among the world's top players in the
development of advanced artificial intelligence (AI) systems, and both are keen
to lead in global AI governance and development. A look at U.S. and Chinese
policy landscapes reveals differences in how the two countries approach the
governance of general-purpose artificial intelligence (GPAI) systems. Three
areas of divergence are notable for policymakers: the focus of domestic AI
regulation, key principles of domestic AI regulation, and approaches to
implementing international AI governance. As AI development continues, global
conversation around AI has warned of global safety and security challenges
posed by GPAI systems. Cooperation between the United States and China might be
needed to address these risks, and understanding the implications of these
differences might help address the broader challenges for international
cooperation between the United States and China on AI safety and security.

### 2. [Facts are Harder Than Opinions -- A Multilingual, Comparative Analysis of LLM-Based Fact-Checking Reliability](http://arxiv.org/pdf/2506.03655v1)

Authors: Lorraine Saju, Arnim Bleier, Jana Lasser, Claudia Wagner

The proliferation of misinformation necessitates scalable, automated
fact-checking solutions. Yet, current benchmarks often overlook multilingual
and topical diversity. This paper introduces a novel, dynamically extensible
data set that includes 61,514 claims in multiple languages and topics,
extending existing datasets up to 2024. Through a comprehensive evaluation of
five prominent Large Language Models (LLMs), including GPT-4o, GPT-3.5 Turbo,
LLaMA 3.1, and Mixtral 8x7B, we identify significant performance gaps between
different languages and topics. While overall GPT-4o achieves the highest
accuracy, it declines to classify 43% of claims. Across all models,
factual-sounding claims are misclassified more often than opinions, revealing a
key vulnerability. These findings underscore the need for caution and highlight
challenges in deploying LLM-based fact-checking systems at scale.

### 3. [Construction of Urban Greenland Resources Collaborative Management Platform](http://arxiv.org/pdf/2506.03830v1)

Authors: Dongyang Lyu, Xiaoqi Li, Zongwei Li

Nowadays, environmental protection has become a global consensus. At the same
time, with the rapid development of science and technology, urbanisation has
become a phenomenon that has become the norm. Therefore, the urban greening
management system is an essential component in protecting the urban
environment. The system utilises a transparent management process known as"
monitoring - early warning - response - optimisation," which enhances the
tracking of greening resources, streamlines maintenance scheduling, and
encourages employee involvement in planning. Designed with a microservice
architecture, the system can improve the utilisation of greening resources by
30\% , increase citizen satisfaction by 20\%, and support carbon neutrality
objectives, ultimately making urban governance more intelligent and focused on
the community. The Happy City Greening Management System effectively manages
gardeners, trees, flowers, and green spaces. It comprises modules for gardener
management, purchase and supplier management, tree and flower management, and
maintenance planning. Its automation feature allows for real-time updates of
greening data, thereby enhancing decision-making. The system is built using
Java for the backend and MySQL for data storage, complemented by a
user-friendly frontend designed with the Vue framework. Additionally, it
leverages features from the Spring Boot framework to enhance maintainability
and scalability.

### 4. [Improving Regulatory Oversight in Online Content Moderation](http://arxiv.org/pdf/2506.04145v1)

Authors: Benedetta Tessa, Denise Amram, Anna Monreale, Stefano Cresci

The European Union introduced the Digital Services Act (DSA) to address the
risks associated with digital platforms and promote a safer online environment.
However, despite the potential of components such as the Transparency Database,
Transparency Reports, and Article 40 of the DSA to improve platform
transparency, significant challenges remain. These include data inconsistencies
and a lack of detailed information, which hinder transparency in content
moderation practices. Additionally, the absence of standardized reporting
structures makes cross-platform comparisons and broader analyses difficult. To
address these issues, we propose two complementary processes: a Transparency
Report Cross-Checking Process and a Verification Process. Their goal is to
provide both internal and external validation by detecting possible
inconsistencies between self-reported and actual platform data, assessing
compliance levels, and ultimately enhancing transparency while improving the
overall effectiveness of the DSA in ensuring accountability in content
moderation. Additionally, these processes can benefit policymakers by providing
more accurate data for decision-making, independent researchers with
trustworthy analysis, and platforms by offering a method for self-assessment
and improving compliance and reporting practices.

### 5. [GA-S$^3$: Comprehensive Social Network Simulation with Group Agents](http://arxiv.org/pdf/2506.03532v1)

Authors: Yunyao Zhang, Zikai Song, Hang Zhou, Wenfeng Ren, Yi-Ping Phoebe Chen, Junqing Yu, Wei Yang

Social network simulation is developed to provide a comprehensive
understanding of social networks in the real world, which can be leveraged for
a wide range of applications such as group behavior emergence, policy
optimization, and business strategy development. However, billions of
individuals and their evolving interactions involved in social networks pose
challenges in accurately reflecting real-world complexities. In this study, we
propose a comprehensive Social Network Simulation System (GA-S3) that leverages
newly designed Group Agents to make intelligent decisions regarding various
online events. Unlike other intelligent agents that represent an individual
entity, our group agents model a collection of individuals exhibiting similar
behaviors, facilitating the simulation of large-scale network phenomena with
complex interactions at a manageable computational cost. Additionally, we have
constructed a social network benchmark from 2024 popular online events that
contains fine-grained information on Internet traffic variations. The
experiment demonstrates that our approach is capable of achieving accurate and
highly realistic prediction results. Code is open at
https://github.com/AI4SS/GAS-3.

### 6. [Misalignment or misuse? The AGI alignment tradeoff](http://arxiv.org/pdf/2506.03755v1)

Authors: Max Hellrigel-Holderbaum, Leonard Dung

Creating systems that are aligned with our goals is seen as a leading
approach to create safe and beneficial AI in both leading AI companies and the
academic field of AI safety. We defend the view that misaligned AGI - future,
generally intelligent (robotic) AI agents - poses catastrophic risks. At the
same time, we support the view that aligned AGI creates a substantial risk of
catastrophic misuse by humans. While both risks are severe and stand in tension
with one another, we show that - in principle - there is room for alignment
approaches which do not increase misuse risk. We then investigate how the
tradeoff between misalignment and misuse looks empirically for different
technical approaches to AI alignment. Here, we argue that many current
alignment techniques and foreseeable improvements thereof plausibly increase
risks of catastrophic misuse. Since the impacts of AI depend on the social
context, we close by discussing important social factors and suggest that to
reduce the risk of a misuse catastrophe due to aligned AGI, techniques such as
robustness, AI control methods and especially good governance seem essential.

### 7. [Words of Warmth: Trust and Sociability Norms for over 26k English Words](http://arxiv.org/pdf/2506.03993v1)

Authors: Saif M. Mohammad

Social psychologists have shown that Warmth (W) and Competence (C) are the
primary dimensions along which we assess other people and groups. These
dimensions impact various aspects of our lives from social competence and
emotion regulation to success in the work place and how we view the world. More
recent work has started to explore how these dimensions develop, why they have
developed, and what they constitute. Of particular note, is the finding that
warmth has two distinct components: Trust (T) and Sociability (S). In this
work, we introduce Words of Warmth, the first large-scale repository of
manually derived word--warmth (as well as word--trust and word--sociability)
associations for over 26k English words. We show that the associations are
highly reliable. We use the lexicons to study the rate at which children
acquire WCTS words with age. Finally, we show that the lexicon enables a wide
variety of bias and stereotype research through case studies on various target
entities. Words of Warmth is freely available at:
http://saifmohammad.com/warmth.html

### 8. [CogniPair: From LLM Chatbots to Conscious AI Agents -- GNWT-Based Multi-Agent Digital Twins for Social Pairing -- Dating & Hiring Applications](http://arxiv.org/pdf/2506.03543v1)

Authors: Wanghao Ye, Sihan Chen, Yiting Wang, Shwai He, Bowei Tian, Guoheng Sun, Ziyi Wang, Ziyao Wang, Yexiao He, Zheyu Shen, Meng Liu, Yuning Zhang, Meng Feng, Yang Wang, Siyuan Peng, Yilong Dai, Zhenle Duan, Hanzhang Qin, Ang Li

Current large language model (LLM) agents lack authentic human psychological
processes necessary for genuine digital twins and social AI applications. To
address this limitation, we present a computational implementation of Global
Workspace Theory (GNWT) that integrates human cognitive architecture principles
into LLM agents, creating specialized sub-agents for emotion, memory, social
norms, planning, and goal-tracking coordinated through a global workspace
mechanism. However, authentic digital twins require accurate personality
initialization. We therefore develop a novel adventure-based personality test
that evaluates true personality through behavioral choices within interactive
scenarios, bypassing self-presentation bias found in traditional assessments.
Building on these innovations, our CogniPair platform enables digital twins to
engage in realistic simulated dating interactions and job interviews before
real encounters, providing bidirectional cultural fit assessment for both
romantic compatibility and workplace matching. Validation using 551 GNWT-Agents
and Columbia University Speed Dating dataset demonstrates 72% correlation with
human attraction patterns, 77.8% match prediction accuracy, and 74% agreement
in human validation studies. This work advances psychological authenticity in
LLM agents and establishes a foundation for intelligent dating platforms and HR
technology solutions.

### 9. [Intersectional Bias in Pre-Trained Image Recognition Models](http://arxiv.org/pdf/2506.03664v1)

Authors: Valerie Krug, Sebastian Stober

Deep Learning models have achieved remarkable success. Training them is often
accelerated by building on top of pre-trained models which poses the risk of
perpetuating encoded biases. Here, we investigate biases in the representations
of commonly used ImageNet classifiers for facial images while considering
intersections of sensitive variables age, race and gender. To assess the
biases, we use linear classifier probes and visualize activations as
topographic maps. We find that representations in ImageNet classifiers
particularly allow differentiation between ages. Less strongly pronounced, the
models appear to associate certain ethnicities and distinguish genders in
middle-aged groups.

### 10. [Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](http://arxiv.org/pdf/2506.03941v1)

Authors: Vivian Nguyen, Lillian Lee, Cristian Danescu-Niculescu-Mizil

During a conversation, there can come certain moments where its outcome hangs
in the balance. In these pivotal moments, how one responds can put the
conversation on substantially different trajectories leading to significantly
different outcomes. Systems that can detect when such moments arise could
assist conversationalists in domains with highly consequential outcomes, such
as mental health crisis counseling.
  In this work, we introduce an unsupervised computational method for detecting
such pivotal moments as they happen, in an online fashion. Our approach relies
on the intuition that a moment is pivotal if our expectation of the outcome
varies widely depending on what might be said next. By applying our method to
crisis counseling conversations, we first validate it by showing that it aligns
with human perception -- counselors take significantly longer to respond during
moments detected by our method -- and with the eventual conversational
trajectory -- which is more likely to change course at these times. We then use
our framework to explore the relation of the counselor's response during
pivotal moments with the eventual outcome of the session.

### Databases

### 1. [Signals as a First-Class Citizen When Querying Knowledge Graphs](http://arxiv.org/pdf/2506.03826v1)

Authors: Tobias Schwarzinger, Gernot Steindl, Thomas Frühwirth, Thomas Preindl, Konrad Diwold, Katrin Ehrenmüller, Fajar J. Ekaputra

Cyber-Physical Systems (CPSs) tightly integrate computation with physical
entities, often generating vast amounts of time series data from thousands of
sensors. Although knowledge graphs offer a powerful means to contextualize
these data, existing approaches to integrating knowledge graphs with time
series data lack a concept to model the continuous temporal values inherent in
CPSs. This gap can make expressing computations on the sensor data cumbersome.
In this work, we propose the integration of knowledge graphs and signals, a
proven concept for modeling temporal values. By treating signals as first-class
citizens in query languages, we can enable seamless querying over knowledge
graphs and signals. While the knowledge graph captures information on the CPS,
signals represent its run-time data from sensors. We discuss the implications
of such an approach and propose SigSPARQL, an extension to the SPARQL query
language, to demonstrate these concepts. Furthermore, we evaluate the
feasibility of implementing SigSPARQL with a prototype and demonstrate the
applicability of the query language for a monitoring use case within a CPS.

### 2. [An Efficient Candidate-Free R-S Set Similarity Join Algorithm with the Filter-and-Verification Tree and MapReduce](http://arxiv.org/pdf/2506.03893v1)

Authors: Yuhong Feng, Fangcao Jian, Yixuan Cao, Xiaobin Jian, Jia Wang, Haiyue Feng, Chunyan Miao

Given two different collections of sets, the exact set similarity R-S Join
finds all set pairs with similarity no less than a given threshold, which has
widespread applications. While existing algorithms accelerate large-scale R-S
Joins using a two-stage filter-and-verification framework along with the
parallel and distributed MapReduce framework, they suffer from excessive
candidate set pairs, leading to significant I/O, data transfer, and
verification overhead, and ultimately degrading the performance. This paper
proposes novel candidate-free R-S Join (CF-RS-Join) algorithms that integrate
filtering and verification into a single stage through filter-and-verification
trees (FVTs) and their linear variants (LFVTs). First, CF-RS-Join with FVT
(CF-RS-Join/FVT) is proposed to leverage an innovative FVT structure that
compresses elements and associated sets in memory, enabling single-stage
processing that eliminates the candidate set generation, fast lookups, and
reduced database scans. Correctness proofs are provided. Second, CF-RS-Join
with LFVT (CF-RS-Join/LFVT) is proposed to exploit a more compact Linear FVT,
which compresses non-branching paths into single nodes and stores them in
linear arrays for optimized traversal. Third, MR-CF-RS-Join/FVT and
MR-CF-RS-Join/LFVT have been proposed to extend our approaches using MapReduce
for parallel processing. Empirical studies on 7 real-world datasets have been
conducted to evaluate the performance of the proposed algorithms against
selected existing algorithms in terms of execution time, scalability, memory
usage, and disk usage. Experimental results demonstrate that our algorithm
using MapReduce, i.e., MR-CF-RS-Join/LFVT, achieves the best performance.

### 3. [TransClean: Finding False Positives in Multi-Source Entity Matching under Real-World Conditions via Transitive Consistency](http://arxiv.org/pdf/2506.04006v1)

Authors: Fernando de Meer Pardo, Branka Hadji Misheva, Martin Braschler, Kurt Stockinger

We present TransClean, a method for detecting false positive predictions of
entity matching algorithms under real-world conditions characterized by
large-scale, noisy, and unlabeled multi-source datasets that undergo
distributional shifts. TransClean is explicitly designed to operate with
multiple data sources in an efficient, robust and fast manner while accounting
for edge cases and requiring limited manual labeling. TransClean leverages the
Transitive Consistency of a matching, a measure of the consistency of a
pairwise matching model f_theta on the matching it produces G_f_theta, based
both on its predictions on directly evaluated record pairs and its predictions
on implied record pairs. TransClean iteratively modifies a matching through
gradually removing false positive matches while removing as few true positive
matches as possible. In each of these steps, the estimation of the Transitive
Consistency is exclusively done through model evaluations and produces
quantities that can be used as proxies of the amounts of true and false
positives in the matching while not requiring any manual labeling, producing an
estimate of the quality of the matching and indicating which record groups are
likely to contain false positives. In our experiments, we compare combining
TransClean with a naively trained pairwise matching model (DistilBERT) and with
a state-of-the-art end-to-end matching method (CLER) and illustrate the
flexibility of TransClean in being able to detect most of the false positives
of either setup across a variety of datasets. Our experiments show that
TransClean induces an average +24.42 F1 score improvement for entity matching
in a multi-source setting when compared to traditional pair-wise matching
algorithms.

### Distributed, Parallel, and Cluster Computing

### 1. [LRScheduler: A Layer-aware and Resource-adaptive Container Scheduler in Edge Computing](http://arxiv.org/pdf/2506.03694v1)

Authors: Zhiqing Tang, Wentao Peng, Jianxiong Guo, Jiong Lou, Hanshuai Cui, Tian Wang, Yuan Wu, Weijia Jia

Lightweight containers provide an efficient approach for deploying
computation-intensive applications in network edge. The layered storage
structure of container images can further reduce the deployment cost and
container startup time. Existing researches discuss layer sharing scheduling
theoretically but with little attention paid to the practical implementation.
To fill in this gap, we propose and implement a Layer-aware and
Resource-adaptive container Scheduler (LRScheduler) in edge computing.
Specifically, we first utilize container image layer information to design and
implement a node scoring and container scheduling mechanism. This mechanism can
effectively reduce the download cost when deploying containers, which is very
important in edge computing with limited bandwidth. Then, we design a
dynamically weighted and resource-adaptive mechanism to enhance load balancing
in edge clusters, increasing layer sharing scores when resource load is low to
use idle resources effectively. Our scheduler is built on the scheduling
framework of Kubernetes, enabling full process automation from task information
acquisition to container dep=loyment. Testing on a real system has shown that
our design can effectively reduce the container deployment cost as compared
with the default scheduler.

### 2. [SLURM Heterogeneous Jobs for Hybrid Classical-Quantum Workflows](http://arxiv.org/pdf/2506.03846v1)

Authors: Aniello Esposito, Utz-Uwe Haus

A method for efficient scheduling of hybrid classical-quantum workflows is
presented, based on standard tools available on common supercomputer systems.
Moderate interventions by the user are required, such as splitting a monolithic
workflow in to basic building blocks and ensuring the data flow. This bares the
potential to significantly reduce idle time of the quantum resource as well as
overall wall time of co-scheduled workflows. Relevant pseudo-code samples and
scripts are provided to demonstrate the simplicity and working principles of
the method.

### 3. [Analysis of Server Throughput For Managed Big Data Analytics Frameworks](http://arxiv.org/pdf/2506.03854v1)

Authors: Emmanouil Anagnostakis, Polyvios Pratikakis

Managed big data frameworks, such as Apache Spark and Giraph demand a large
amount of memory per core to process massive volume datasets effectively. The
memory pressure that arises from the big data processing leads to high garbage
collection (GC) overhead. Big data analytics frameworks attempt to remove this
overhead by offloading objects to storage devices. At the same time,
infrastructure providers, trying to address the same problem, attribute more
memory to increase memory per instance leaving cores underutilized. For
frameworks, trying to avoid GC through offloading to storage devices leads to
high Serialization/Deserialization (S/D) overhead. For infrastructure, the
result is that resource usage is decreased. These limitations prevent managed
big data frameworks from effectively utilizing the CPU thus leading to low
server throughput.
  We conduct a methodological analysis of server throughput for managed big
data analytics frameworks. More specifically, we examine, whether reducing GC
and S/D can help increase the effective CPU utilization of the server. We use a
system called TeraHeap that moves objects from the Java managed heap (H1) to a
secondary heap over a fast storage device (H2) to reduce the GC overhead and
eliminate S/D over data. We focus on analyzing the system's performance under
the co-location of multiple memory-bound instances to utilize all available
DRAM and study server throughput. Our detailed methodology includes choosing
the DRAM budget for each instance and how to distribute this budget among H1
and Page Cache (PC). We try two different distributions for the DRAM budget,
one with more H1 and one with more PC to study the needs of both approaches. We
evaluate both techniques under 3 different memory-per-core scenarios using
Spark and Giraph with native JVM or JVM with TeraHeap. We do this to check
throughput changes when memory capacity increases.

### 4. [Energy-Aware Workflow Execution: An Overview of Techniques for Saving Energy and Emissions in Scientific Compute Clusters](http://arxiv.org/pdf/2506.04062v1)

Authors: Lauritz Thamsen, Yehia Elkhatib, Paul Harvey, Syed Waqar Nabi, Jeremy Singer, Wim Vanderbauwhede

Scientific research in many fields routinely requires the analysis of large
datasets, and scientists often employ workflow systems to leverage clusters of
computers for their data analysis. However, due to their size and scale, these
workflow applications can have a considerable environmental footprint in terms
of compute resource use, energy consumption, and carbon emissions. Mitigating
this is critical in light of climate change and the urgent need to reduce
carbon emissions.
  In this chapter, we exemplify the problem by estimating the carbon footprint
of three real-world scientific workflows from different scientific domains. We
then describe techniques for reducing the energy consumption and, thereby,
carbon footprint of individual workflow tasks and entire workflow applications,
such as using energy-efficient heterogeneous architectures, generating
optimised code, scaling processor voltages and frequencies, consolidating
workloads on shared cluster nodes, and scheduling workloads for optimised
energy efficiency.

### 5. [Cascadia: A Cascade Serving System for Large Language Models](http://arxiv.org/pdf/2506.04203v1)

Authors: Youhe Jiang, Fangcheng Fu, Wanru Zhao, Stephan Rabanser, Nicholas D. Lane, Binhang Yuan

Recent advances in large language models (LLMs) have intensified the need to
deliver both rapid responses and high-quality answers. More powerful models
yield better results but incur higher inference latency, whereas smaller models
are faster yet less capable. Recent work proposes balancing this
latency-quality trade-off using model cascades, which route simpler queries to
smaller models and more complex ones to larger models. However, enabling
efficient cascade serving remains challenging. Current frameworks lack
effective mechanisms for handling (i) the huge and varying resource demands of
different LLMs, (ii) the inherent heterogeneity of LLM workloads, and (iii) the
co-optimization of system deployment and routing strategy. Motivated by these
observations, we introduce Cascadia, a novel cascade serving framework designed
explicitly to schedule request routing and deploy model cascades for fast,
quality-preserving LLM serving. Cascadia employs a bi-level optimization
method: at the inner level, it uses a mixed-integer linear program to select
resource allocations and parallelism strategies based on LLM information and
workload characteristics; at the outer level, it applies a weighted Tchebycheff
algorithm to iteratively co-optimize the routing strategy and the system
deployment produced by the inner level. Our extensive evaluation on diverse
workload traces and different model cascades (DeepSeek and the Llama series)
demonstrates that Cascadia significantly outperforms both single-model
deployments and the state-of-the-art cascade serving baseline, achieving up to
4x (2.3x on average) tighter latency SLOs and up to 5x (2.4x on average) higher
throughput while maintaining target answer quality.

### 6. [An Efficient Candidate-Free R-S Set Similarity Join Algorithm with the Filter-and-Verification Tree and MapReduce](http://arxiv.org/pdf/2506.03893v1)

Authors: Yuhong Feng, Fangcao Jian, Yixuan Cao, Xiaobin Jian, Jia Wang, Haiyue Feng, Chunyan Miao

Given two different collections of sets, the exact set similarity R-S Join
finds all set pairs with similarity no less than a given threshold, which has
widespread applications. While existing algorithms accelerate large-scale R-S
Joins using a two-stage filter-and-verification framework along with the
parallel and distributed MapReduce framework, they suffer from excessive
candidate set pairs, leading to significant I/O, data transfer, and
verification overhead, and ultimately degrading the performance. This paper
proposes novel candidate-free R-S Join (CF-RS-Join) algorithms that integrate
filtering and verification into a single stage through filter-and-verification
trees (FVTs) and their linear variants (LFVTs). First, CF-RS-Join with FVT
(CF-RS-Join/FVT) is proposed to leverage an innovative FVT structure that
compresses elements and associated sets in memory, enabling single-stage
processing that eliminates the candidate set generation, fast lookups, and
reduced database scans. Correctness proofs are provided. Second, CF-RS-Join
with LFVT (CF-RS-Join/LFVT) is proposed to exploit a more compact Linear FVT,
which compresses non-branching paths into single nodes and stores them in
linear arrays for optimized traversal. Third, MR-CF-RS-Join/FVT and
MR-CF-RS-Join/LFVT have been proposed to extend our approaches using MapReduce
for parallel processing. Empirical studies on 7 real-world datasets have been
conducted to evaluate the performance of the proposed algorithms against
selected existing algorithms in terms of execution time, scalability, memory
usage, and disk usage. Experimental results demonstrate that our algorithm
using MapReduce, i.e., MR-CF-RS-Join/LFVT, achieves the best performance.

### 7. [Depermissioning Web3: a Permissionless Accountable RPC Protocol for Blockchain Networks](http://arxiv.org/pdf/2506.03940v1)

Authors: Weihong Wang, Tom Van Cutsem

In blockchain networks, so-called "full nodes" serve data to and relay
transactions from clients through an RPC interface. This serving layer enables
integration of "Web3" data, stored on blockchains, with "Web2" mobile or web
applications that cannot directly participate as peers in a blockchain network.
In practice, the serving layer is dominated by a small number of centralized
services ("node providers") that offer permissioned access to RPC endpoints.
Clients register with these providers because they offer reliable and
convenient access to blockchain data: operating a full node themselves requires
significant computational and storage resources, and public (permissionless)
RPC nodes lack financial incentives to serve large numbers of clients with
consistent performance.
  Permissioned access to an otherwise permissionless blockchain network raises
concerns regarding the privacy, integrity, and availability of data access. To
address this, we propose a Permissionless Accountable RPC Protocol (PARP). It
enables clients and full nodes to interact pseudonymously while keeping both
parties accountable. PARP leverages "light client" schemes for essential data
integrity checks, combined with fraud proofs, to keep full nodes honest and
accountable. It integrates payment channels to facilitate micro-payments,
holding clients accountable for the resources they consume and providing an
economic incentive for full nodes to serve. Our prototype implementation for
Ethereum demonstrates the feasibility of PARP, and we quantify its overhead
compared to the base RPC protocol.

### 8. [Crowd-SFT: Crowdsourcing for LLM Alignment](http://arxiv.org/pdf/2506.04063v1)

Authors: Alex Sotiropoulos, Sulyab Thottungal Valapu, Linus Lei, Jared Coleman, Bhaskar Krishnamachari

Large Language Models (LLMs) increasingly rely on Supervised Fine-Tuning
(SFT) and Reinforcement Learning from Human Feedback (RLHF) to align model
responses with human preferences. While RLHF employs a reinforcement learning
approach with a separate reward model, SFT uses human-curated datasets for
supervised learning. Both approaches traditionally depend on small, vetted
groups of annotators, making them costly, prone to bias, and limited in
scalability. We propose an open, crowd-sourced fine-tuning framework that
addresses these limitations by enabling broader feedback collection for SFT
without extensive annotator training. Our framework promotes incentive fairness
via a point-based reward system correlated with Shapley values and guides model
convergence through iterative model updates. Our multi-model selection
framework demonstrates up to a 55% reduction in target distance over
single-model selection, enabling subsequent experiments that validate our
point-based reward mechanism's close alignment with Shapley values (a
well-established method for attributing individual contributions) thereby
supporting fair and scalable participation.

### 9. [Carbon-Aware Temporal Data Transfer Scheduling Across Cloud Datacenters](http://arxiv.org/pdf/2506.04117v1)

Authors: Elvis Rodrigues, Jacob Goldverg, Tevfik Kosar

Inter-datacenter communication is a significant part of cloud operations and
produces a substantial amount of carbon emissions for cloud data centers, where
the environmental impact has already been a pressing issue. In this paper, we
present a novel carbon-aware temporal data transfer scheduling framework,
called LinTS, which promises to significantly reduce the carbon emission of
data transfers between cloud data centers. LinTS produces a competitive
transfer schedule and makes scaling decisions, outperforming common heuristic
algorithms. LinTS can lower carbon emissions during inter-datacenter transfers
by up to 66% compared to the worst case and up to 15% compared to other
solutions while preserving all deadline constraints.

### 10. [GenTT: Generate Vectorized Codes for General Tensor Permutation](http://arxiv.org/pdf/2506.03686v1)

Authors: Yaojian Chen, Tianyu Ma, An Yang, Lin Gan, Wenlai Zhao, Guangwen Yang

Tensor permutation is a fundamental operation widely applied in AI, tensor
networks, and related fields. However, it is extremely complex, and different
shapes and permutation maps can make a huge difference. SIMD permutation began
to be studied in 2006, but the best method at that time was to split complex
permutations into multiple simple permutations to do SIMD, which might increase
the complexity for very complex permutations. Subsequently, as tensor
contraction gained significant attention, researchers explored structured
permutations associated with tensor contraction. Progress on general
permutations has been limited, and with increasing SIMD bit widths, achieving
efficient performance for these permutations has become increasingly
challenging. We propose a SIMD permutation toolkit, \system, that generates
optimized permutation code for arbitrary instruction sets, bit widths, tensor
shapes, and permutation patterns, while maintaining low complexity. In our
experiments, \system is able to achieve up to $38\times$ speedup for special
cases and $5\times$ for general gases compared to Numpy.

### Digital Libraries

### 1. [Distinguishing True Influence from Hyperprolificity with Citation Distance](http://arxiv.org/pdf/2506.03527v1)

Authors: Lu Li, Yun Wan, Feng Xiao

Accurately evaluating scholarly influence is essential for fair academic
assessment, yet traditional bibliometric indicators - dominated by publication
and citation counts - often favor hyperprolific authors over those with deeper,
long-term impact. We propose the x-index, a novel citation-based metric that
conceptualizes citation as a process of knowledge diffusion and incorporates
citation distance to reflect the structural reach of scholarly work. By
weighting citations according to the collaborative proximity between citing and
cited authors, the x-index captures both the depth and breadth of influence
within evolving academic networks. Empirical analyses show that the x-index
significantly improves the rankings of Turing Award recipients while reducing
those of hyperprolific authors, better aligning rankings with recognized
academic merit. It also demonstrates superior discriminatory power among
early-career researchers and reveals stronger sensitivity to institutional
research quality. These results suggest that the x-index offers a more
equitable and forward-looking alternative to existing metrics, with practical
applications in talent identification, funding decisions, and academic
recommendation systems.

### 2. [Preface to the Special Issue of the TAL Journal on Scholarly Document Processing](http://arxiv.org/pdf/2506.03587v1)

Authors: Florian Boudin, Akiko Aizawa

The rapid growth of scholarly literature makes it increasingly difficult for
researchers to keep up with new knowledge. Automated tools are now more
essential than ever to help navigate and interpret this vast body of
information. Scientific papers pose unique difficulties, with their complex
language, specialized terminology, and diverse formats, requiring advanced
methods to extract reliable and actionable insights. Large language models
(LLMs) offer new opportunities, enabling tasks such as literature reviews,
writing assistance, and interactive exploration of research. This special issue
of the TAL journal highlights research addressing these challenges and, more
broadly, research on natural language processing and information retrieval for
scholarly and scientific documents.

### 3. [Introducing multiverse analysis to bibliometrics: The case of team size effects on disruptive research](http://arxiv.org/pdf/2506.03726v1)

Authors: Christian Leibel, Lutz Bornmann

Although bibliometrics has become an essential tool in the evaluation of
research performance, bibliometric analyses are sensitive to a range of
methodological choices. Subtle choices in data selection, indicator
construction, and modeling decisions can substantially alter results. Ensuring
robustness, meaning that findings hold up under different reasonable scenarios,
is therefore critical for credible research and research evaluation. To address
this issue, this study introduces multiverse analysis to bibliometrics.
Multiverse analysis is a statistical tool that enables analysts to
transparently discuss modeling assumptions and thoroughly assess model
robustness. Whereas standard robustness checks usually cover only a small
subset of all plausible models, multiverse analysis includes all plausible
models. We illustrate the benefits of multiverse analysis by testing the
hypothesis posed by Wu et al. (2019) that small teams produce more disruptive
research than large teams. While we found robust evidence of a negative effect
of team size on disruption scores, the effect size is so small that its
practical relevance seems questionable. Our findings underscore the importance
of assessing the multiverse robustness of bibliometric results to clarify their
practical implications.

### Discrete Mathematics

### 1. [Tournament Robustness via Redundancy](http://arxiv.org/pdf/2506.03701v1)

Authors: Klim Efremenko, Hendrik Molter, Meirav Zehavi

A knockout tournament is one of the most simple and popular forms of
competition. Here, we are given a binary tournament tree where all leaves are
labeled with seed position names. The players participating in the tournament
are assigned to the seed positions. In each round, the two players assigned to
leaves of the tournament tree with a common parent compete, and the winner is
promoted to the parent. The last remaining player is the winner of the
tournament.
  In this work, we study the problem of making knock-out tournaments robust
against manipulation, where the form of manipulation we consider is changing
the outcome of a game. We assume that our input is only the number of players
that compete in the tournament, and the number of manipulations against which
the tournament should be robust. Furthermore, we assume that there is a
strongest player, that is, a player that beats any of the other players.
However, the identity of this player is not part of the problem input.
  To ensure robustness against manipulation, we uncover an unexpected
connection between the problem at hand and communication protocols that utilize
a feedback channel, offering resilience against adversarial noise. We explore
the trade-off between the size of the robust tournament tree and the degree of
protection against manipulation. Specifically, we demonstrate that it is
possible to tolerate up to a $1/3$ fraction of manipulations along each
leaf-to-root path, at the cost of only a polynomial blow-up in the tournament
size.

### 2. [GenTT: Generate Vectorized Codes for General Tensor Permutation](http://arxiv.org/pdf/2506.03686v1)

Authors: Yaojian Chen, Tianyu Ma, An Yang, Lin Gan, Wenlai Zhao, Guangwen Yang

Tensor permutation is a fundamental operation widely applied in AI, tensor
networks, and related fields. However, it is extremely complex, and different
shapes and permutation maps can make a huge difference. SIMD permutation began
to be studied in 2006, but the best method at that time was to split complex
permutations into multiple simple permutations to do SIMD, which might increase
the complexity for very complex permutations. Subsequently, as tensor
contraction gained significant attention, researchers explored structured
permutations associated with tensor contraction. Progress on general
permutations has been limited, and with increasing SIMD bit widths, achieving
efficient performance for these permutations has become increasingly
challenging. We propose a SIMD permutation toolkit, \system, that generates
optimized permutation code for arbitrary instruction sets, bit widths, tensor
shapes, and permutation patterns, while maintaining low complexity. In our
experiments, \system is able to achieve up to $38\times$ speedup for special
cases and $5\times$ for general gases compared to Numpy.

### 3. [Spanning-tree-packing protocol for conference key propagation in quantum networks](http://arxiv.org/pdf/2506.04105v1)

Authors: Anton Trushechkin, Hermann Kampermann, Dagmar Bruß

We consider a network of users connected by pairwise quantum key distribution
(QKD) links. Using these pairwise secret keys and public classical
communication, the users want to generate a common (conference) secret key at
the maximal rate. We propose an algorithm based on spanning tree packing (a
known problem in graph theory) and prove its optimality. This algorithm enables
optimal conference key generation in modern quantum networks of arbitrary
topology. Additionally, we discuss how it can guide the optimal placement of
new bipartite links in the network design.

### Data Structures and Algorithms

### 1. [Connectivity-Preserving Minimum Separator in AT-free Graphs](http://arxiv.org/pdf/2506.03612v1)

Authors: Batya Kenig

Let $A$ and $B$ be disjoint, non-adjacent vertex-sets in an undirected,
connected graph $G$, whose vertices are associated with positive weights. We
address the problem of identifying a minimum-weight subset of vertices
$S\subseteq V(G)$ that, when removed, disconnects $A$ from $B$ while preserving
the internal connectivity of both $A$ and $B$. We call such a subset of
vertices a connectivity-preserving, or safe minimum $A,B$-separator. Deciding
whether a safe $A,B$-separator exists is NP-hard by reduction from the
2-disjoint connected subgraphs problem, and remains NP-hard even for restricted
graph classes that include planar graphs, and $P_\ell$-free graphs if $\ell\geq
5$. In this work, we show that if $G$ is AT-free then in polynomial time we can
find a safe $A,B$-separator of minimum weight, or establish that no safe
$A,B$-separator exists.

### 2. [Stability Notions for Hospital Residents with Sizes](http://arxiv.org/pdf/2506.03638v1)

Authors: Haricharan Balasundaram, J B Krishnashree, Girija Limaye, Meghana Nasre

The Hospital Residents problem with sizes (HRS) is a generalization of the
well-studied hospital residents (HR) problem. In the HRS problem, an agent $a$
has a size $s(a)$ and the agent occupies $s(a)$ many positions of the hospital
$h$ when assigned to $h$. The notion of stability in this setting is suitably
modified, and it is known that deciding whether an HRS instance admits a stable
matching is NP-hard under severe restrictions. In this work, we explore a
variation of stability, which we term occupancy-based stability. This notion
was defined by McDermid and Manlove in their work, however, to the best of our
knowledge, this notion remains unexplored. We show that every HRS instance
admits an occupancy-stable matching. We further show that computing a
maximum-size occupancy-stable matching is NP-hard. We complement our hardness
result by providing a linear-time 3-approximation algorithm for the max-size
occupancy-stable matching problem. Given that the classical notion of stability
adapted for HRS is not guaranteed to exist in general, we show a practical
restriction under which a stable matching is guaranteed to exist. We present an
efficient algorithm to output a stable matching in the restricted HRS
instances. We also provide an alternate NP-hardness proof for the decision
version of the stable matching problem for HRS which imposes a severe
restriction on the number of neighbours of non-unit sized agents.

### 3. [Faster Approx. Top-K: Harnessing the Full Power of Two Stages](http://arxiv.org/pdf/2506.04165v1)

Authors: Yashas Samaga, Varun Yerram, Spandana Raj Babbula, Prateek Jain, Praneeth Netrapalli

We consider the Top-$K$ selection problem, which aims to identify the
largest-$K$ elements from an array. Top-$K$ selection arises in many machine
learning algorithms and often becomes a bottleneck on accelerators, which are
optimized for dense matrix multiplications. To address this problem,
\citet{chern2022tpuknnknearestneighbor} proposed a fast two-stage
\textit{approximate} Top-$K$ algorithm: (i) partition the input array and
select the top-$1$ element from each partition, (ii) sort this \textit{smaller
subset} and return the top $K$ elements. In this paper, we consider a
generalized version of this algorithm, where the first stage selects top-$K'$
elements, for some $1 \leq K' \leq K$, from each partition. Our contributions
are as follows: (i) we derive an expression for the expected recall of this
generalized algorithm and show that choosing $K' > 1$ with fewer partitions in
the first stage reduces the input size to the second stage more effectively
while maintaining the same expected recall as the original algorithm, (ii) we
derive a bound on the expected recall for the original algorithm in
\citet{chern2022tpuknnknearestneighbor} that is provably tighter by a factor of
$2$ than the one in that paper, and (iii) we implement our algorithm on Cloud
TPUv5e and achieve around an order of magnitude speedups over the original
algorithm without sacrificing recall on real-world tasks.

### 4. [GenTT: Generate Vectorized Codes for General Tensor Permutation](http://arxiv.org/pdf/2506.03686v1)

Authors: Yaojian Chen, Tianyu Ma, An Yang, Lin Gan, Wenlai Zhao, Guangwen Yang

Tensor permutation is a fundamental operation widely applied in AI, tensor
networks, and related fields. However, it is extremely complex, and different
shapes and permutation maps can make a huge difference. SIMD permutation began
to be studied in 2006, but the best method at that time was to split complex
permutations into multiple simple permutations to do SIMD, which might increase
the complexity for very complex permutations. Subsequently, as tensor
contraction gained significant attention, researchers explored structured
permutations associated with tensor contraction. Progress on general
permutations has been limited, and with increasing SIMD bit widths, achieving
efficient performance for these permutations has become increasingly
challenging. We propose a SIMD permutation toolkit, \system, that generates
optimized permutation code for arbitrary instruction sets, bit widths, tensor
shapes, and permutation patterns, while maintaining low complexity. In our
experiments, \system is able to achieve up to $38\times$ speedup for special
cases and $5\times$ for general gases compared to Numpy.

### 5. [Testing (Conditional) Mutual Information](http://arxiv.org/pdf/2506.03894v1)

Authors: Jan Seyfried, Sayantan Sen, Marco Tomamichel

We investigate the sample complexity of mutual information and conditional
mutual information testing. For conditional mutual information testing, given
access to independent samples of a triple of random variables $(A, B, C)$ with
unknown distribution, we want to distinguish between two cases: (i) $A$ and $C$
are conditionally independent, i.e., $I(A\!:\!C|B) = 0$, and (ii) $A$ and $C$
are conditionally dependent, i.e., $I(A\!:\!C|B) \geq \varepsilon$ for some
threshold $\varepsilon$. We establish an upper bound on the number of samples
required to distinguish between the two cases with high confidence, as a
function of $\varepsilon$ and the three alphabet sizes. We conjecture that our
bound is tight and show that this is indeed the case in several parameter
regimes. For the special case of mutual information testing (when $B$ is
trivial), we establish the necessary and sufficient number of samples required
up to polylogarithmic terms.
  Our technical contributions include a novel method to efficiently simulate
weakly correlated samples from the conditionally independent distribution
$P_{A|B} P_{C|B} P_B$ given access to samples from an unknown distribution
$P_{ABC}$, and a new estimator for equivalence testing that can handle such
correlated samples, which might be of independent interest.

### Emerging Technologies

### 1. [Does Prompt Design Impact Quality of Data Imputation by LLMs?](http://arxiv.org/pdf/2506.04172v1)

Authors: Shreenidhi Srinivasan, Lydia Manikonda

Generating realistic synthetic tabular data presents a critical challenge in
machine learning. It adds another layer of complexity when this data contain
class imbalance problems. This paper presents a novel token-aware data
imputation method that leverages the in-context learning capabilities of large
language models. This is achieved through the combination of a structured
group-wise CSV-style prompting technique and the elimination of irrelevant
contextual information in the input prompt. We test this approach with two
class-imbalanced binary classification datasets and evaluate the effectiveness
of imputation using classification-based evaluation metrics. The experimental
results demonstrate that our approach significantly reduces the input prompt
size while maintaining or improving imputation quality compared to our baseline
prompt, especially for datasets that are of relatively smaller in size. The
contributions of this presented work is two-fold -- 1) it sheds light on the
importance of prompt design when leveraging LLMs for synthetic data generation
and 2) it addresses a critical gap in LLM-based data imputation for
class-imbalanced datasets with missing data by providing a practical solution
within computational constraints. We hope that our work will foster further
research and discussions about leveraging the incredible potential of LLMs and
prompt engineering techniques for synthetic data generation.

### Formal Languages and Automata Theory

### 1. [Jumbled Scattered Factors](http://arxiv.org/pdf/2506.03814v1)

Authors: Pamela Fleischmann, Annika Huch, Melf Kammholz, Tore Koß

In this work, we combine the research on (absent) scattered factors with the
one of jumbled words. For instance, $\mathtt{wolf}$ is an absent scattered
factor of $\mathtt{cauliflower}$ but since $\mathtt{lfow}$, a jumbled (or
abelian) version of $\mathtt{wolf}$, is a scattered factor, $\mathtt{wolf}$
occurs as a jumbled scattered factor in $\mathtt{cauliflower}$. A \emph{jumbled
scattered factor} $u$ of a word $w$ is constructed by letters of $w$ with the
only rule that the number of occurrences per letter in $u$ is smaller than or
equal to the one in $w$. We proceed to partition and characterise the set of
jumbled scattered factors by the number of jumbled letters and use the latter
as a measure. For this new class of words, we relate the folklore longest
common subsequence (scattered factor) to the number of required jumbles.
Further, we investigate the smallest possible number of jumbles alongside the
jumbled scattered factor relation as well as Simon's congruence from the point
of view of jumbled scattered factors and jumbled universality.

### 2. [Mapped Exponent and Asymptotic Critical Exponent of Words](http://arxiv.org/pdf/2506.04091v1)

Authors: Eva Foster, Aleksi Saarela, Aleksi Vanhatalo

We study how much injective morphisms can increase the repetitiveness of a
given word. This question has a few possible variations depending on the
meaning of ``repetitiveness''. We concentrate on fractional exponents of finite
words and asymptotic critical exponents of infinite words. We characterize
finite words that, when mapped by injective morphisms, can have arbitrarily
high fractional exponent. For infinite words, alongside other results, we show
that the asymptotic critical exponent grows at most by a constant factor
(depending on the size of the alphabet) when mapped by an injective morphism.
For both finite and infinite words, the binary case is better understood than
the general case.

### Graphics

### 1. [Facial Appearance Capture at Home with Patch-Level Reflectance Prior](http://arxiv.org/pdf/2506.03478v1)

Authors: Yuxuan Han, Junfeng Lyu, Kuan Sheng, Minghao Que, Qixuan Zhang, Lan Xu, Feng Xu

Existing facial appearance capture methods can reconstruct plausible facial
reflectance from smartphone-recorded videos. However, the reconstruction
quality is still far behind the ones based on studio recordings. This paper
fills the gap by developing a novel daily-used solution with a co-located
smartphone and flashlight video capture setting in a dim room. To enhance the
quality, our key observation is to solve facial reflectance maps within the
data distribution of studio-scanned ones. Specifically, we first learn a
diffusion prior over the Light Stage scans and then steer it to produce the
reflectance map that best matches the captured images. We propose to train the
diffusion prior at the patch level to improve generalization ability and
training stability, as current Light Stage datasets are in ultra-high
resolution but limited in data size. Tailored to this prior, we propose a
patch-level posterior sampling technique to sample seamless full-resolution
reflectance maps from this patch-level diffusion model. Experiments demonstrate
our method closes the quality gap between low-cost and studio recordings by a
large margin, opening the door for everyday users to clone themselves to the
digital world. Our code will be released at https://github.com/yxuhan/DoRA.

### 2. [Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data](http://arxiv.org/pdf/2506.04120v1)

Authors: Ben Moran, Mauro Comi, Steven Bohez, Tom Erez, Zhibin Li, Leonard Hasenclever

Creating accurate, physical simulations directly from real-world robot motion
holds great value for safe, scalable, and affordable robot learning, yet
remains exceptionally challenging. Real robot data suffers from occlusions,
noisy camera poses, dynamic scene elements, which hinder the creation of
geometrically accurate and photorealistic digital twins of unseen objects. We
introduce a novel real-to-sim framework tackling all these challenges at once.
Our key insight is a hybrid scene representation merging the photorealistic
rendering of 3D Gaussian Splatting with explicit object meshes suitable for
physics simulation within a single representation. We propose an end-to-end
optimization pipeline that leverages differentiable rendering and
differentiable physics within MuJoCo to jointly refine all scene components -
from object geometry and appearance to robot poses and physical parameters -
directly from raw and imprecise robot trajectories. This unified optimization
allows us to simultaneously achieve high-fidelity object mesh reconstruction,
generate photorealistic novel views, and perform annotation-free robot pose
calibration. We demonstrate the effectiveness of our approach both in
simulation and on challenging real-world sequences using an ALOHA 2 bi-manual
manipulator, enabling more practical and robust real-to-simulation pipelines.

### 3. [SplArt: Articulation Estimation and Part-Level Reconstruction with 3D Gaussian Splatting](http://arxiv.org/pdf/2506.03594v1)

Authors: Shengjie Lin, Jiading Fang, Muhammad Zubair Irshad, Vitor Campagnolo Guizilini, Rares Andrei Ambrus, Greg Shakhnarovich, Matthew R. Walter

Reconstructing articulated objects prevalent in daily environments is crucial
for applications in augmented/virtual reality and robotics. However, existing
methods face scalability limitations (requiring 3D supervision or costly
annotations), robustness issues (being susceptible to local optima), and
rendering shortcomings (lacking speed or photorealism). We introduce SplArt, a
self-supervised, category-agnostic framework that leverages 3D Gaussian
Splatting (3DGS) to reconstruct articulated objects and infer kinematics from
two sets of posed RGB images captured at different articulation states,
enabling real-time photorealistic rendering for novel viewpoints and
articulations. SplArt augments 3DGS with a differentiable mobility parameter
per Gaussian, achieving refined part segmentation. A multi-stage optimization
strategy is employed to progressively handle reconstruction, part segmentation,
and articulation estimation, significantly enhancing robustness and accuracy.
SplArt exploits geometric self-supervision, effectively addressing challenging
scenarios without requiring 3D annotations or category-specific priors.
Evaluations on established and newly proposed benchmarks, along with
applications to real-world scenarios using a handheld RGB camera, demonstrate
SplArt's state-of-the-art performance and real-world practicality. Code is
publicly available at https://github.com/ripl/splart.

### Computer Science and Game Theory

### 1. [From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications](http://arxiv.org/pdf/2506.03464v1)

Authors: Yang Cai, Haipeng Luo, Chen-Yu Wei, Weiqiang Zheng

The convergence of online learning algorithms in games under self-play is a
fundamental question in game theory and machine learning. Among various notions
of convergence, last-iterate convergence is particularly desirable, as it
reflects the actual decisions made by the learners and captures the day-to-day
behavior of the learning dynamics. While many algorithms are known to converge
in the average-iterate, achieving last-iterate convergence typically requires
considerably more effort in both the design and the analysis of the algorithm.
Somewhat surprisingly, we show in this paper that for a large family of games,
there exists a simple black-box reduction that transforms the average iterates
of an uncoupled learning dynamics into the last iterates of a new uncoupled
learning dynamics, thus also providing a reduction from last-iterate
convergence to average-iterate convergence. Our reduction applies to games
where each player's utility is linear in both their own strategy and the joint
strategy of all opponents. This family includes two-player bimatrix games and
generalizations such as multi-player polymatrix games. By applying our
reduction to the Optimistic Multiplicative Weights Update algorithm, we obtain
new state-of-the-art last-iterate convergence rates for uncoupled learning
dynamics in two-player zero-sum normal-form games: (1) an $O(\frac{\log d}{T})$
last-iterate convergence rate under gradient feedback, representing an
exponential improvement in the dependence on the dimension $d$ (i.e., the
maximum number of actions available to either player); and (2) an
$\widetilde{O}(d^{\frac{1}{5}} T^{-\frac{1}{5}})$ last-iterate convergence rate
under bandit feedback, improving upon the previous best rates of
$\widetilde{O}(\sqrt{d} T^{-\frac{1}{8}})$ and $\widetilde{O}(\sqrt{d}
T^{-\frac{1}{6}})$.

### 2. [Complexity and Manipulation of International Kidney Exchange Programmes with Country-Specific Parameterss](http://arxiv.org/pdf/2506.04092v1)

Authors: Rachael Colley, David Manlove, Daniel Paulusma, Mengxiao Zhang

Kidney Exchange Programmes (KEPs) facilitate the exchange of kidneys, and
larger pools of recipient-donor pairs tend to yield proportionally more
transplants, leading to the proposal of international KEPs (IKEPs). However, as
studied by \citet{mincu2021ip}, practical limitations must be considered in
IKEPs to ensure that countries remain willing to participate. Thus, we study
IKEPs with country-specific parameters, represented by a tuple $\Gamma$,
restricting the selected transplants to be feasible for the countries to
conduct, e.g., imposing an upper limit on the number of consecutive exchanges
within a country's borders. We provide a complete complexity dichotomy for the
problem of finding a feasible (according to the constraints given by $\Gamma$)
cycle packing with the maximum number of transplants, for every possible
$\Gamma$. We also study the potential for countries to misreport their
parameters to increase their allocation. As manipulation can harm the total
number of transplants, we propose a novel individually rational and incentive
compatible mechanism $\mathcal{M}_{\text{order}}$. We first give a theoretical
approximation ratio for $\mathcal{M}_{\text{order}}$ in terms of the number of
transplants, and show that the approximation ratio of
$\mathcal{M}_{\text{order}}$ is asymptotically optimal. We then use simulations
which suggest that, in practice, the performance of
$\mathcal{M}_{\text{order}}$ is significantly better than this worst-case
ratio.

### Human-Computer Interaction

### 1. [VChatter: Exploring Generative Conversational Agents for Simulating Exposure Therapy to Reduce Social Anxiety](http://arxiv.org/pdf/2506.03520v1)

Authors: Han Zhang, KaWing Tsang, Zhenhui Peng

Many people struggle with social anxiety, feeling fear, or even physically
uncomfortable in social situations like talking to strangers. Exposure therapy,
a clinical method that gradually and repeatedly exposes individuals to the
source of their fear and helps them build coping mechanisms, can reduce social
anxiety but traditionally requires human therapists' guidance and constructions
of situations. In this paper, we developed a multi-agent system VChatter to
explore large language models(LLMs)-based conversational agents for simulating
exposure therapy with users. Based on a survey study (N=36) and an expert
interview, VChatter includes an Agent-P, which acts as a psychotherapist to
design the exposure therapy plans for users, and two Agent-Hs, which can take
on different interactive roles in low, medium, and high exposure scenarios. A
six-day qualitative study (N=10) showcases VChatter's usefulness in reducing
users' social anxiety, feelings of isolation, and avoidance of social
interactions. We demonstrated the feasibility of using LLMs-based
conversational agents to simulate exposure therapy for addressing social
anxiety and discussed future concerns for designing agents tailored to social
anxiety.

### 2. [Understanding Visually Impaired Tramway Passengers Interaction with Public Transport Systems](http://arxiv.org/pdf/2506.03687v1)

Authors: Dominik Mimra, Dominik Kaar, Enrico Del Re, Novel Certad, Joshua Cherian Varughese, David Seibt, Cristina Olaverri-Monreal

Designing inclusive public transport services is crucial to developing
modern, barrier-free smart city infrastructure. This research contributes to
the design of inclusive public transport by considering accessibility
challenges emerging from socio-technical systems, thus demanding the
integration of technological and social solutions. Using Actor-Network Theory
(ANT) as a theoretical framework and a mixed-method approach, including
shadowing and a focus group, this study examines the socio-technical networks
that shape accessibility experiences for visually impaired passengers utilizing
the tram in Linz, Austria. Key dimensions that influence public transport
accessibility are identified: network configuration, mobility patterns,
technology integration, and warning systems. The results show that
accessibility emerges from complex interactions between human actors
(passengers, staff) and non-human actors (assistive devices, infrastructure)
rather than being an inherent property of transport systems. Digital
technologies serve multiple functions, from navigational assistance to broader
social inclusion, although users comfort with technology varies. Participants
emphasized the importance of the two-sense principle for warning signals, with
directional audio and tactile feedback particularly valuable.

### 3. [Design of a visual environment for programming by direct data manipulation](http://arxiv.org/pdf/2506.03720v1)

Authors: Michel Adam, Patrice Frison, Moncef Daoud, Sabine Letellier Zarshenas

The use of applications on computers, smartphones, and tablets has been
considerably simplified thanks to interactive and dynamic graphical interfaces
coupled with the mouse and touch screens. It is no longer necessary to be a
computer specialist to use them. Paradoxically, the development of computer
programs generally requires writing lines of code in a programming language
whose syntax is particularly strict. This process poses many difficulties for
programmers. We propose an original tool in which arbitrary programs
(Turing-complete) can be developed in a completely visual manner by direct
manipulation of the data, without writing a line of code. The user can thus
develop an algorithm by directly visualizing the result of actions taken on the
data. A method for constructing iterations is associated with the tool. It
proposes to create each part, including the loop body, in a non-linear manner
under visual control of the state of the data. In addition, the tool supports
the production of lines of code in several languages including Python, C, Java,
that correspond to the actions performed. In this article, we present the tool,
the design choices, the problems to be solved, and the limits and the
contributions of the direct-data-manipulation approach.

### 4. [Enhancing Text Comprehension for Dyslexic Readers: A 3D Semantic Visualization Approach Using Transformer Mode](http://arxiv.org/pdf/2506.03731v1)

Authors: Zhengyang Li

Dyslexic individuals often face significant challenges with traditional
reading, particularly when engaging with complex texts such as mystery novels.
These texts typically demand advanced narrative tracking and information
integration skills, making it difficult for dyslexic readers to fully
comprehend the content. However, research indicates that while dyslexic
individuals may struggle with textual processing, they often possess strong
spatial imagination abilities. Leveraging this strength, this study proposes an
innovative approach using Transformer models to map sentences and words into
three-dimensional vector representations. This process clusters semantically
similar sentences and words in spatial proximity, allowing dyslexic readers to
interpret the semantic structure and narrative flow of the text through spatial
perception. Experimental results demonstrate that, compared to direct text
reading, this three-dimensional semantic visualization method significantly
enhances dyslexic readers' comprehension of complex texts. In particular, it
shows marked advantages in identifying narrative relationships and character
connections. This study provides a novel pathway for improving textual
comprehension among dyslexic individuals

### 5. [Neural and Cognitive Impacts of AI: The Influence of Task Subjectivity on Human-LLM Collaboration](http://arxiv.org/pdf/2506.04167v1)

Authors: Matthew Russell, Aman Shah, Giles Blaney, Judith Amores, Mary Czerwinski, Robert J. K. Jacob

AI-based interactive assistants are advancing human-augmenting technology,
yet their effects on users' mental and physiological states remain
under-explored. We address this gap by analyzing how Copilot for Microsoft
Word, a LLM-based assistant, impacts users. Using tasks ranging from objective
(SAT reading comprehension) to subjective (personal reflection), and with
measurements including fNIRS, Empatica E4, NASA-TLX, and questionnaires, we
measure Copilot's effects on users. We also evaluate users' performance with
and without Copilot across tasks. In objective tasks, participants reported a
reduction of workload and an increase in enjoyment, which was paired with
objective performance increases. Participants reported reduced workload and
increased enjoyment with no change in performance in a creative poetry writing
task. However, no benefits due to Copilot use were reported in a highly
subjective self-reflection task. Although no physiological changes were
recorded due to Copilot use, task-dependent differences in prefrontal cortex
activation offer complementary insights into the cognitive processes associated
with successful and unsuccessful human-AI collaboration. These findings suggest
that AI assistants' effectiveness varies with task type-particularly showing
decreased usefulness in tasks that engage episodic memory-and presents a
brain-network based hypothesis of human-AI collaboration.

### 6. [PromptCanvas: Composable Prompting Workspaces Using Dynamic Widgets for Exploration and Iteration in Creative Writing](http://arxiv.org/pdf/2506.03741v1)

Authors: Rifat Mehreen Amin, Oliver Hans Kühle, Daniel Buschek, Andreas Butz

We introduce PromptCanvas, a concept that transforms prompting into a
composable, widget-based experience on an infinite canvas. Users can generate,
customize, and arrange interactive widgets representing various facets of their
text, offering greater control over AI-generated content. PromptCanvas allows
widget creation through system suggestions, user prompts, or manual input,
providing a flexible environment tailored to individual needs. This enables
deeper engagement with the creative process. In a lab study with 18
participants, PromptCanvas outperformed a traditional conversational UI on the
Creativity Support Index. Participants found that it reduced cognitive load,
with lower mental demand and frustration. Qualitative feedback revealed that
the visual organization of thoughts and easy iteration encouraged new
perspectives and ideas. A follow-up field study (N=10) confirmed these results,
showcasing the potential of dynamic, customizable interfaces in improving
collaborative writing with AI.

### 7. [Understanding Mental Models of Generative Conversational Search and The Effect of Interface Transparency](http://arxiv.org/pdf/2506.03807v1)

Authors: Chadha Degachi, Samuel Kernan Freire, Evangelos Niforatos, Gerd Kortuem

The experience and adoption of conversational search is tied to the accuracy
and completeness of users' mental models -- their internal frameworks for
understanding and predicting system behaviour. Thus, understanding these models
can reveal areas for design interventions. Transparency is one such
intervention which can improve system interpretability and enable mental model
alignment. While past research has explored mental models of search engines,
those of generative conversational search remain underexplored, even while the
popularity of these systems soars. To address this, we conducted a study with
16 participants, who performed 4 search tasks using 4 conversational interfaces
of varying transparency levels. Our analysis revealed that most user mental
models were too abstract to support users in explaining individual search
instances. These results suggest that 1) mental models may pose a barrier to
appropriate trust in conversational search, and 2) hybrid web-conversational
search is a promising novel direction for future search interface design.

### 8. [Controlling Difficulty of Generated Text for AI-Assisted Language Learning](http://arxiv.org/pdf/2506.04072v1)

Authors: Meiqing Jin, Liam Dugan, Chris Callison-Burch

Practicing conversations with large language models (LLMs) presents a
promising alternative to traditional in-person language learning. However, most
LLMs generate text at a near-native level of complexity, making them ill-suited
for beginner learners (CEFR: A1-A2). In this paper, we investigate whether
controllable generation techniques -- specifically modular methods that do not
require model fine-tuning -- can adapt LLM outputs to better support absolute
beginners. We evaluate these methods through both automatic metrics and a user
study with university-level learners of Japanese. Our findings show that while
prompting alone fails to control output difficulty, the use of future
discriminators (Yang and Klein, 2021) significantly improves output
comprehensibility (from 40.4\% to 84.3\%). We further introduce a novel
token-level evaluation metric, Token Miss Rate (TMR), that quantifies the
proportion of incomprehensible tokens per utterance and correlates strongly
with human judgments. To support future research in AI-assisted language
learning, we release our code, models, annotation tools, and dataset.

### 9. [Intersectional Bias in Pre-Trained Image Recognition Models](http://arxiv.org/pdf/2506.03664v1)

Authors: Valerie Krug, Sebastian Stober

Deep Learning models have achieved remarkable success. Training them is often
accelerated by building on top of pre-trained models which poses the risk of
perpetuating encoded biases. Here, we investigate biases in the representations
of commonly used ImageNet classifiers for facial images while considering
intersections of sensitive variables age, race and gender. To assess the
biases, we use linear classifier probes and visualize activations as
topographic maps. We find that representations in ImageNet classifiers
particularly allow differentiation between ages. Less strongly pronounced, the
models appear to associate certain ethnicities and distinguish genders in
middle-aged groups.

### 10. [Generating Pedagogically Meaningful Visuals for Math Word Problems: A New Benchmark and Analysis of Text-to-Image Models](http://arxiv.org/pdf/2506.03735v1)

Authors: Junling Wang, Anna Rutkiewicz, April Yi Wang, Mrinmaya Sachan

Visuals are valuable tools for teaching math word problems (MWPs), helping
young learners interpret textual descriptions into mathematical expressions
before solving them. However, creating such visuals is labor-intensive and
there is a lack of automated methods to support this process. In this paper, we
present Math2Visual, an automatic framework for generating pedagogically
meaningful visuals from MWP text descriptions. Math2Visual leverages a
pre-defined visual language and a design space grounded in interviews with math
teachers, to illustrate the core mathematical relationships in MWPs. Using
Math2Visual, we construct an annotated dataset of 1,903 visuals and evaluate
Text-to-Image (TTI) models for their ability to generate visuals that align
with our design. We further fine-tune several TTI models with our dataset,
demonstrating improvements in educational visual generation. Our work
establishes a new benchmark for automated generation of pedagogically
meaningful visuals and offers insights into key challenges in producing
multimodal educational content, such as the misrepresentation of mathematical
relationships and the omission of essential visual elements.

### Information Retrieval

### 1. [Scaling Transformers for Discriminative Recommendation via Generative Pretraining](http://arxiv.org/pdf/2506.03699v1)

Authors: Chunqi Wang, Bingchao Wu, Zheng Chen, Lei Shen, Bing Wang, Xiaoyi Zeng

Discriminative recommendation tasks, such as CTR (click-through rate) and CVR
(conversion rate) prediction, play critical roles in the ranking stage of
large-scale industrial recommender systems. However, training a discriminative
model encounters a significant overfitting issue induced by data sparsity.
Moreover, this overfitting issue worsens with larger models, causing them to
underperform smaller ones. To address the overfitting issue and enhance model
scalability, we propose a framework named GPSD (\textbf{G}enerative
\textbf{P}retraining for \textbf{S}calable \textbf{D}iscriminative
Recommendation), drawing inspiration from generative training, which exhibits
no evident signs of overfitting. GPSD leverages the parameters learned from a
pretrained generative model to initialize a discriminative model, and
subsequently applies a sparse parameter freezing strategy. Extensive
experiments conducted on both industrial-scale and publicly available datasets
demonstrate the superior performance of GPSD. Moreover, it delivers remarkable
improvements in online A/B tests. GPSD offers two primary advantages: 1) it
substantially narrows the generalization gap in model training, resulting in
better test performance; and 2) it leverages the scalability of Transformers,
delivering consistent performance gains as models are scaled up. Specifically,
we observe consistent performance improvements as the model dense parameters
scale from 13K to 0.3B, closely adhering to power laws. These findings pave the
way for unifying the architectures of recommendation models and language
models, enabling the direct application of techniques well-established in large
language models to recommendation models.

### 2. [Graph-Embedding Empowered Entity Retrieval](http://arxiv.org/pdf/2506.03895v1)

Authors: Emma J. Gerritse, Faegheh Hasibi, Arjen P. de Vries

In this research, we investigate methods for entity retrieval using graph
embeddings. While various methods have been proposed over the years, most
utilize a single graph embedding and entity linking approach. This hinders our
understanding of how different graph embedding and entity linking methods
impact entity retrieval. To address this gap, we investigate the effects of
three different categories of graph embedding techniques and five different
entity linking methods. We perform a reranking of entities using the distance
between the embeddings of annotated entities and the entities we wish to
rerank. We conclude that the selection of both graph embeddings and entity
linkers significantly impacts the effectiveness of entity retrieval. For graph
embeddings, methods that incorporate both graph structure and textual
descriptions of entities are the most effective. For entity linking, both
precision and recall concerning concepts are important for optimal retrieval
performance. Additionally, it is essential for the graph to encompass as many
entities as possible.

### 3. [GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems](http://arxiv.org/pdf/2506.04015v1)

Authors: Tiehua Mei, Hengrui Chen, Peng Yu, Jiaqing Liang, Deqing Yang

Although large language models (LLMs) have shown great potential in
recommender systems, the prohibitive computational costs for fine-tuning LLMs
on entire datasets hinder their successful deployment in real-world scenarios.
To develop affordable and effective LLM-based recommender systems, we focus on
the task of coreset selection which identifies a small subset of fine-tuning
data to optimize the test loss, thereby facilitating efficient LLMs'
fine-tuning. Although there exist some intuitive solutions of subset selection,
including distribution-based and importance-based approaches, they often lead
to suboptimal performance due to the misalignment with downstream fine-tuning
objectives or weak generalization ability caused by individual-level sample
selection. To overcome these challenges, we propose GORACS, which is a novel
Group-level Optimal tRAnsport-guided Coreset Selection framework for LLM-based
recommender systems. GORACS is designed based on two key principles for coreset
selection: 1) selecting the subsets that minimize the test loss to align with
fine-tuning objectives, and 2) enhancing model generalization through
group-level data selection. Corresponding to these two principles, GORACS has
two key components: 1) a Proxy Optimization Objective (POO) leveraging optimal
transport and gradient information to bound the intractable test loss, thus
reducing computational costs by avoiding repeated LLM retraining, and 2) a
two-stage Initialization-Then-Refinement Algorithm (ITRA) for efficient
group-level selection. Our extensive experiments across diverse recommendation
datasets and tasks validate that GORACS significantly reduces fine-tuning costs
of LLMs while achieving superior performance over the state-of-the-art
baselines and full data training. The source code of GORACS are available at
https://github.com/Mithas-114/GORACS.

### 4. [A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](http://arxiv.org/pdf/2506.04083v1)

Authors: Zhiyu Zhang, Wei Chen, Youfang Lin, Huaiyu Wan

Recent Continual Learning (CL)-based Temporal Knowledge Graph Reasoning
(TKGR) methods focus on significantly reducing computational cost and
mitigating catastrophic forgetting caused by fine-tuning models with new data.
However, existing CL-based TKGR methods still face two key limitations: (1)
They usually one-sidedly reorganize individual historical facts, while
overlooking the historical context essential for accurately understanding the
historical semantics of these facts; (2) They preserve historical knowledge by
simply replaying historical facts, while ignoring the potential conflicts
between historical and emerging facts. In this paper, we propose a Deep
Generative Adaptive Replay (DGAR) method, which can generate and adaptively
replay historical entity distribution representations from the whole historical
context. To address the first challenge, historical context prompts as sampling
units are built to preserve the whole historical context information. To
overcome the second challenge, a pre-trained diffusion model is adopted to
generate the historical distribution. During the generation process, the common
features between the historical and current distributions are enhanced under
the guidance of the TKGR model. In addition, a layer-by-layer adaptive replay
mechanism is designed to effectively integrate historical and current
distributions. Experimental results demonstrate that DGAR significantly
outperforms baselines in reasoning and mitigating forgetting.

### 5. [Quantifying Query Fairness Under Unawareness](http://arxiv.org/pdf/2506.04140v1)

Authors: Thomas Jaenich, Alejandro Moreo, Alessandro Fabris, Graham McDonald, Andrea Esuli, Iadh Ounis, Fabrizio Sebastiani

Traditional ranking algorithms are designed to retrieve the most relevant
items for a user's query, but they often inherit biases from data that can
unfairly disadvantage vulnerable groups. Fairness in information access systems
(IAS) is typically assessed by comparing the distribution of groups in a
ranking to a target distribution, such as the overall group distribution in the
dataset. These fairness metrics depend on knowing the true group labels for
each item. However, when groups are defined by demographic or sensitive
attributes, these labels are often unknown, leading to a setting known as
"fairness under unawareness". To address this, group membership can be inferred
using machine-learned classifiers, and group prevalence is estimated by
counting the predicted labels. Unfortunately, such an estimation is known to be
unreliable under dataset shift, compromising the accuracy of fairness
evaluations. In this paper, we introduce a robust fairness estimator based on
quantification that effectively handles multiple sensitive attributes beyond
binary classifications. Our method outperforms existing baselines across
various sensitive attributes and, to the best of our knowledge, is the first to
establish a reliable protocol for measuring fairness under unawareness across
multiple queries and groups.

### 6. [ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking](http://arxiv.org/pdf/2506.03487v1)

Authors: Xianming Li, Aamir Shakir, Rui Huang, Julius Lipp, Jing Li

Reranking is fundamental to information retrieval and retrieval-augmented
generation, with recent Large Language Models (LLMs) significantly advancing
reranking quality. While recent advances with LLMs have significantly improved
document reranking quality, current approaches primarily rely on large-scale
LLMs (>7B parameters) through zero-shot prompting, presenting high
computational costs. Small Language Models (SLMs) offer a promising alternative
because of their efficiency, but our preliminary quantitative analysis reveals
they struggle with understanding task prompts without fine-tuning. This limits
their effectiveness for document reranking tasks. To address this issue, we
introduce a novel two-stage training approach, ProRank, for SLM-based document
reranking. First, we propose a prompt warmup stage using reinforcement learning
GRPO to steer SLMs to understand task prompts and generate more accurate
coarse-grained binary relevance scores for document reranking. Then, we
continuously fine-tune the SLMs with a fine-grained score learning stage
without introducing additional layers to further improve the reranking quality.
Comprehensive experimental results demonstrate that the proposed ProRank
consistently outperforms both the most advanced open-source and proprietary
reranking models. Notably, our lightweight ProRank-0.5B model even surpasses
the powerful 32B LLM reranking model on the BEIR benchmark, establishing that
properly trained SLMs can achieve superior document reranking performance while
maintaining computational efficiency.

### 7. [Understanding Mental Models of Generative Conversational Search and The Effect of Interface Transparency](http://arxiv.org/pdf/2506.03807v1)

Authors: Chadha Degachi, Samuel Kernan Freire, Evangelos Niforatos, Gerd Kortuem

The experience and adoption of conversational search is tied to the accuracy
and completeness of users' mental models -- their internal frameworks for
understanding and predicting system behaviour. Thus, understanding these models
can reveal areas for design interventions. Transparency is one such
intervention which can improve system interpretability and enable mental model
alignment. While past research has explored mental models of search engines,
those of generative conversational search remain underexplored, even while the
popularity of these systems soars. To address this, we conducted a study with
16 participants, who performed 4 search tasks using 4 conversational interfaces
of varying transparency levels. Our analysis revealed that most user mental
models were too abstract to support users in explaining individual search
instances. These results suggest that 1) mental models may pose a barrier to
appropriate trust in conversational search, and 2) hybrid web-conversational
search is a promising novel direction for future search interface design.

### 8. [CRAWLDoc: A Dataset for Robust Ranking of Bibliographic Documents](http://arxiv.org/pdf/2506.03822v1)

Authors: Fabian Karl, Ansgar Scherp

Publication databases rely on accurate metadata extraction from diverse web
sources, yet variations in web layouts and data formats present challenges for
metadata providers. This paper introduces CRAWLDoc, a new method for contextual
ranking of linked web documents. Starting with a publication's URL, such as a
digital object identifier, CRAWLDoc retrieves the landing page and all linked
web resources, including PDFs, ORCID profiles, and supplementary materials. It
embeds these resources, along with anchor texts and the URLs, into a unified
representation. For evaluating CRAWLDoc, we have created a new, manually
labeled dataset of 600 publications from six top publishers in computer
science. Our method CRAWLDoc demonstrates a robust and layout-independent
ranking of relevant documents across publishers and data formats. It lays the
foundation for improved metadata extraction from web documents with various
layouts and formats. Our source code and dataset can be accessed at
https://github.com/FKarl/CRAWLDoc.

### 9. [Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising](http://arxiv.org/pdf/2506.03827v1)

Authors: Zhenhui Liu, Chunyuan Yuan, Ming Pang, Zheng Fang, Li Yuan, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, Jingping Shao

Retrieval systems primarily address the challenge of matching user queries
with the most relevant advertisements, playing a crucial role in e-commerce
search advertising. The diversity of user needs and expressions often produces
massive long-tail queries that cannot be matched with merchant bidwords or
product titles, which results in some advertisements not being recalled,
ultimately harming user experience and search efficiency. Existing query
rewriting research focuses on various methods such as query log mining,
query-bidword vector matching, or generation-based rewriting. However, these
methods often fail to simultaneously optimize the relevance and authenticity of
the user's original query and rewrite and maximize the revenue potential of
recalled ads.
  In this paper, we propose a Multi-objective aligned Bidword Generation Model
(MoBGM), which is composed of a discriminator, generator, and preference
alignment module, to address these challenges. To simultaneously improve the
relevance and authenticity of the query and rewrite and maximize the platform
revenue, we design a discriminator to optimize these key objectives. Using the
feedback signal of the discriminator, we train a multi-objective aligned
bidword generator that aims to maximize the combined effect of the three
objectives. Extensive offline and online experiments show that our proposed
algorithm significantly outperforms the state of the art. After deployment, the
algorithm has created huge commercial value for the platform, further verifying
its feasibility and robustness.

### Machine Learning

### 1. [Directional Non-Commutative Monoidal Embeddings for MNIST](http://arxiv.org/pdf/2506.03472v1)

Authors: Mahesh Godavarti

We present an empirical validation of the directional non-commutative
monoidal embedding framework recently introduced in prior
work~\cite{Godavarti2025monoidal}. This framework defines learnable
compositional embeddings using distinct non-commutative operators per dimension
(axis) that satisfy an interchange law, generalizing classical one-dimensional
transforms. Our primary goal is to verify that this framework can effectively
model real data by applying it to a controlled, well-understood task: image
classification on the MNIST dataset~\cite{lecun1998gradient}. A central
hypothesis for why the proposed monoidal embedding works well is that it
generalizes the Discrete Fourier Transform (DFT)~\cite{oppenheim1999discrete}
by learning task-specific frequency components instead of using fixed basis
frequencies. We test this hypothesis by comparing learned monoidal embeddings
against fixed DFT-based embeddings on MNIST. The results show that as the
embedding dimensionality decreases (e.g., from 32 to 8 to 2), the performance
gap between the learned monoidal embeddings and fixed DFT-based embeddings on
MNIST grows increasingly large. This comparison is used as an analytic tool to
explain why the framework performs well: the learnable embeddings can capture
the most discriminative spectral components for the task. Overall, our
experiments confirm that directional non-commutative monoidal embeddings are
highly effective for representing image data, offering a compact learned
representation that retains high task performance. The code used in this work
is available at
https://github.com/mahesh-godavarti/directional_composition_mnist.

### 2. [Learning Monotonic Probabilities with a Generative Cost Model](http://arxiv.org/pdf/2506.03542v1)

Authors: Yongxiang Tang, Yanhua Cheng, Xiaocheng Liu, Chenchen Jiao, Yanxiang Zeng, Ning Luo, Pengjia Yuan, Xialong Liu, Peng Jiang

In many machine learning tasks, it is often necessary for the relationship
between input and output variables to be monotonic, including both strictly
monotonic and implicitly monotonic relationships. Traditional methods for
maintaining monotonicity mainly rely on construction or regularization
techniques, whereas this paper shows that the issue of strict monotonic
probability can be viewed as a partial order between an observable revenue
variable and a latent cost variable. This perspective enables us to reformulate
the monotonicity challenge into modeling the latent cost variable. To tackle
this, we introduce a generative network for the latent cost variable, termed
the Generative Cost Model (GCM), which inherently addresses the strict
monotonic problem, and propose the Implicit Generative Cost Model (IGCM) to
address the implicit monotonic problem. We further validate our approach with a
numerical simulation of quantile regression and conduct multiple experiments on
public datasets, showing that our method significantly outperforms existing
monotonic modeling techniques. The code for our experiments can be found at
https://github.com/tyxaaron/GCM.

### 3. [Optimizing FPGA and Wafer Test Coverage with Spatial Sampling and Machine Learning](http://arxiv.org/pdf/2506.03556v1)

Authors: Wang WeiQuan, Riaz-ul-Haque Mian

In semiconductor manufacturing, testing costs remain significantly high,
especially during wafer and FPGA testing. To reduce the number of required
tests while maintaining predictive accuracy, this study investigates three
baseline sampling strategies: Random Sampling, Stratified Sampling, and k-means
Clustering Sampling. To further enhance these methods, this study proposes a
novel algorithm that improves the sampling quality of each approach. This
research is conducted using real industrial production data from wafer-level
tests and silicon measurements from various FPGAs. This study introduces two
hybrid strategies: Stratified with Short Distance Elimination (S-SDE) and
k-means with Short Distance Elimination (K-SDE). Their performance is evaluated
within the framework of Gaussian Process Regression (GPR) for predicting wafer
and FPGA test data. At the core of our proposed approach is the Short Distance
Elimination (SDE) algorithm, which excludes spatially proximate candidate
points during sampling, thereby ensuring a more uniform distribution of
training data across the physical domain. A parameter sweep was conducted over
the (alpha, beta) thresholds, where alpha and beta are in the range {0, 1, 2,
3, 4} and not both zero, to identify the optimal combination that minimizes
RMSD. Experimental results on a randomly selected wafer file reveal that
(alpha, beta) equal (2, 2) yields the lowest RMSD. Accordingly, all subsequent
experiments adopt this parameter configuration. The results demonstrate that
the proposed SDE-based strategies enhance predictive accuracy: K-SDE improves
upon k-means sampling by 16.26 percent (wafer) and 13.07 percent (FPGA), while
S-SDE improves upon stratified sampling by 16.49 percent (wafer) and 8.84
percent (FPGA).

### 4. [VCDiag: Classifying Erroneous Waveforms for Failure Triage Acceleration](http://arxiv.org/pdf/2506.03590v1)

Authors: Minh Luu, Surya Jasper, Khoi Le, Evan Pan, Michael Quinn, Aakash Tyagi, Jiang Hu

Failure triage in design functional verification is critical but
time-intensive, relying on manual specification reviews, log inspections, and
waveform analyses. While machine learning (ML) has improved areas like stimulus
generation and coverage closure, its application to RTL-level simulation
failure triage, particularly for large designs, remains limited. VCDiag offers
an efficient, adaptable approach using VCD data to classify failing waveforms
and pinpoint likely failure locations. In the largest experiment, VCDiag
achieves over 94% accuracy in identifying the top three most likely modules.
The framework introduces a novel signal selection and statistical compression
approach, achieving over 120x reduction in raw data size while preserving
features essential for classification. It can also be integrated into diverse
Verilog/SystemVerilog designs and testbenches.

### 5. [Out-of-Distribution Graph Models Merging](http://arxiv.org/pdf/2506.03674v1)

Authors: Yidi Wang, Jiawei Gu, pei Xiaobing, Xubin Zheng, Xiao Luo, Pengyang Wang, Ziyue Qiao

This paper studies a novel problem of out-of-distribution graph models
merging, which aims to construct a generalized model from multiple graph models
pre-trained on different domains with distribution discrepancy. This problem is
challenging because of the difficulty in learning domain-invariant knowledge
implicitly in model parameters and consolidating expertise from potentially
heterogeneous GNN backbones. In this work, we propose a graph generation
strategy that instantiates the mixture distribution of multiple domains. Then,
we merge and fine-tune the pre-trained graph models via a MoE module and a
masking mechanism for generalized adaptation. Our framework is
architecture-agnostic and can operate without any source/target domain data.
Both theoretical analysis and experimental results demonstrate the
effectiveness of our approach in addressing the model generalization problem.

### 6. [Comprehensive Attribute Encoding and Dynamic LSTM HyperModels for Outcome Oriented Predictive Business Process Monitoring](http://arxiv.org/pdf/2506.03696v1)

Authors: Fang Wang, Paolo Ceravolo, Ernesto Damiani

Predictive Business Process Monitoring (PBPM) aims to forecast future
outcomes of ongoing business processes. However, existing methods often lack
flexibility to handle real-world challenges such as simultaneous events, class
imbalance, and multi-level attributes. While prior work has explored static
encoding schemes and fixed LSTM architectures, they struggle to support
adaptive representations and generalize across heterogeneous datasets. To
address these limitations, we propose a suite of dynamic LSTM HyperModels that
integrate two-level hierarchical encoding for event and sequence attributes,
character-based decomposition of event labels, and novel pseudo-embedding
techniques for durations and attribute correlations. We further introduce
specialized LSTM variants for simultaneous event modeling, leveraging
multidimensional embeddings and time-difference flag augmentation. Experimental
validation on four public and real-world datasets demonstrates up to 100%
accuracy on balanced datasets and F1 scores exceeding 86\% on imbalanced ones.
Our approach advances PBPM by offering modular and interpretable models better
suited for deployment in complex settings. Beyond PBPM, it contributes to the
broader AI community by improving temporal outcome prediction, supporting data
heterogeneity, and promoting explainable process intelligence frameworks.

### 7. [FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning](http://arxiv.org/pdf/2506.03777v1)

Authors: Li Zhang, Zhongxuan Han, Chaochao chen, Xiaohua Feng, Jiaming Zhang, Yuyuan Li

With emerging application of Federated Learning (FL) in decision-making
scenarios, it is imperative to regulate model fairness to prevent disparities
across sensitive groups (e.g., female, male). Current research predominantly
focuses on two concepts of group fairness within FL: Global Fairness (overall
model disparity across all clients) and Local Fairness (the disparity within
each client). However, the non-decomposable, non-differentiable nature of
fairness criteria pose two fundamental, unresolved challenges for fair FL: (i)
Harmonizing global and local fairness in multi-class classification; (ii)
Enabling a controllable, optimal accuracy-fairness trade-off. To tackle the
aforementioned challenges, we propose a novel controllable federated
group-fairness calibration framework, named FedFACT. FedFACT identifies the
Bayes-optimal classifiers under both global and local fairness constraints in
multi-class case, yielding models with minimal performance decline while
guaranteeing fairness. To effectively realize an adjustable, optimal
accuracy-fairness balance, we derive specific characterizations of the
Bayes-optimal fair classifiers for reformulating fair FL as personalized
cost-sensitive learning problem for in-processing, and bi-level optimization
for post-processing. Theoretically, we provide convergence and generalization
guarantees for FedFACT to approach the near-optimal accuracy under given
fairness levels. Extensive experiments on multiple datasets across various data
heterogeneity demonstrate that FedFACT consistently outperforms baselines in
balancing accuracy and global-local fairness.

### 8. [Attention-Only Transformers via Unrolled Subspace Denoising](http://arxiv.org/pdf/2506.03790v1)

Authors: Peng Wang, Yifu Lu, Yaodong Yu, Druv Pai, Qing Qu, Yi Ma

Despite the popularity of transformers in practice, their architectures are
empirically designed and neither mathematically justified nor interpretable.
Moreover, as indicated by many empirical studies, some components of
transformer architectures may be redundant. To derive a fully interpretable
transformer architecture with only necessary components, we contend that the
goal of representation learning is to compress a set of noisy initial token
representations towards a mixture of low-dimensional subspaces. To compress
these noisy token representations, an associated denoising operation naturally
takes the form of a multi-head (subspace) self-attention. By unrolling such
iterative denoising operations into a deep network, we arrive at a highly
compact architecture that consists of \textit{only} self-attention operators
with skip connections at each layer. Moreover, we show that each layer performs
highly efficient denoising: it improves the signal-to-noise ratio of token
representations \textit{at a linear rate} with respect to the number of layers.
Despite its simplicity, extensive experiments on vision and language tasks
demonstrate that such a transformer achieves performance close to that of
standard transformer architectures such as GPT-2 and CRATE.

### 9. [Learning Equilibria in Matching Games with Bandit Feedback](http://arxiv.org/pdf/2506.03802v1)

Authors: Andreas Athanasopoulos, Christos Dimitrakakis

We investigate the problem of learning an equilibrium in a generalized
two-sided matching market, where agents can adaptively choose their actions
based on their assigned matches. Specifically, we consider a setting in which
matched agents engage in a zero-sum game with initially unknown payoff
matrices, and we explore whether a centralized procedure can learn an
equilibrium from bandit feedback. We adopt the solution concept of matching
equilibrium, where a pair consisting of a matching $\mathfrak{m}$ and a set of
agent strategies $X$ forms an equilibrium if no agent has the incentive to
deviate from $(\mathfrak{m}, X)$. To measure the deviation of a given pair
$(\mathfrak{m}, X)$ from the equilibrium pair $(\mathfrak{m}^\star, X^\star)$,
we introduce matching instability that can serve as a regret measure for the
corresponding learning problem. We then propose a UCB algorithm in which agents
form preferences and select actions based on optimistic estimates of the game
payoffs, and prove that it achieves sublinear, instance-independent regret over
a time horizon $T$.

### 10. [Survey of Active Learning Hyperparameters: Insights from a Large-Scale Experimental Grid](http://arxiv.org/pdf/2506.03817v1)

Authors: Julius Gonsior, Tim Rieß, Anja Reusch, Claudio Hartmann, Maik Thiele, Wolfgang Lehner

Annotating data is a time-consuming and costly task, but it is inherently
required for supervised machine learning. Active Learning (AL) is an
established method that minimizes human labeling effort by iteratively
selecting the most informative unlabeled samples for expert annotation, thereby
improving the overall classification performance. Even though AL has been known
for decades, AL is still rarely used in real-world applications. As indicated
in the two community web surveys among the NLP community about AL, two main
reasons continue to hold practitioners back from using AL: first, the
complexity of setting AL up, and second, a lack of trust in its effectiveness.
We hypothesize that both reasons share the same culprit: the large
hyperparameter space of AL. This mostly unexplored hyperparameter space often
leads to misleading and irreproducible AL experiment results. In this study, we
first compiled a large hyperparameter grid of over 4.6 million hyperparameter
combinations, second, recorded the performance of all combinations in the
so-far biggest conducted AL study, and third, analyzed the impact of each
hyperparameter in the experiment results. In the end, we give recommendations
about the influence of each hyperparameter, demonstrate the surprising
influence of the concrete AL strategy implementation, and outline an
experimental study design for reproducible AL experiments with minimal
computational effort, thus contributing to more reproducible and trustworthy AL
research in the future.

### Neural and Evolutionary Computing

### 1. [Designing morphologies of soft medical devices using cooperative neuro coevolution](http://arxiv.org/pdf/2506.03847v1)

Authors: Hugo Alcaraz-Herrera, Michail-Antisthenis Tsompanas, Igor Balaz, Andrew Adamatzky

Soft robots have proven to outperform traditional robots in applications
related to propagation in geometrically constrained environments. Designing
these robots and their controllers is an intricate task, since their building
materials exhibit non-linear properties. Human designs may be biased; hence,
alternative designing processes should be considered. We present a cooperative
neuro coevolution approach to designing the morphologies of soft actuators and
their controllers for applications in drug delivery apparatus. Morphologies and
controllers are encoded as compositional pattern-producing networks evolved by
Neuroevolution of Augmented Topologies (NEAT) and in cooperative coevolution
methodology, taking into account different collaboration methods. Four
collaboration methods are studied: n best individuals, n worst individuals, n
best and worst individuals, and n random individuals. As a performance
baseline, the results from the implementation of Age-Fitness Pareto
Optimisation (AFPO) are considered. The metrics used are the maximum
displacement in upward bending and the robustness of the devices in terms of
applying to the same evolved morphology a diverse set of controllers. Results
suggest that the cooperative neuro coevolution approach can produce more
suitable morphologies for the intended devices than AFPO.

### 2. [IntLevPy: A Python library to classify and model intermittent and Lévy processes](http://arxiv.org/pdf/2506.03729v1)

Authors: Shailendra Bhandari, Pedro Lencastre, Sergiy Denysov, Yurii Bystryk, Pedro G. Lind

IntLevPy provides a comprehensive description of the IntLevPy Package, a
Python library designed for simulating and analyzing intermittent and L\'evy
processes. The package includes functionalities for process simulation,
including full parameter estimation and fitting optimization for both families
of processes, moment calculation, and classification methods. The
classification methodology utilizes adjusted-$R^2$ and a noble performance
measure {\Gamma}, enabling the distinction between intermittent and L\'evy
processes. IntLevPy integrates iterative parameter optimization with
simulation-based validation. This paper provides an in-depth user guide
covering IntLevPy software architecture, installation, validation workflows,
and usage examples. In this way, IntLevPy facilitates systematic exploration of
these two broad classes of stochastic processes, bridging theoretical models
and practical applications.

### 3. [Optimal Spiking Brain Compression: Improving One-Shot Post-Training Pruning and Quantization for Spiking Neural Networks](http://arxiv.org/pdf/2506.03996v1)

Authors: Lianfeng Shi, Ao Li, Benjamin Ward-Cherrier

Spiking Neural Networks (SNNs) have emerged as a new generation of
energy-efficient neural networks suitable for implementation on neuromorphic
hardware. As neuromorphic hardware has limited memory and computing resources,
weight pruning and quantization have recently been explored to improve SNNs'
efficiency. State-of-the-art SNN pruning/quantization methods employ multiple
compression and training iterations, increasing the cost for pre-trained or
very large SNNs. In this paper, we propose a new one-shot post-training
pruning/quantization framework, Optimal Spiking Brain Compression (OSBC), that
adapts the Optimal Brain Compression (OBC) method of [Frantar, Singh, and
Alistarh, 2023] for SNNs. Rather than minimizing the loss on neuron input
current as OBC does, OSBC achieves more efficient and accurate SNN compression
in one pass by minimizing the loss on spiking neuron membrane potential with a
small sample dataset. Our experiments on neuromorphic datasets (N-MNIST,
CIFAR10-DVS, DVS128-Gesture) demonstrate that OSBC can achieve 97% sparsity
through pruning with 1.41%, 10.20%, and 1.74% accuracy loss, or 4-bit symmetric
quantization with 0.17%, 1.54%, and 7.71% accuracy loss, respectively. Code
will be available on GitHub.

### 4. [A Class Inference Scheme With Dempster-Shafer Theory for Learning Fuzzy-Classifier Systems](http://arxiv.org/pdf/2506.03588v1)

Authors: Hiroki Shiraishi, Hisao Ishibuchi, Masaya Nakata

The decision-making process significantly influences the predictions of
machine learning models. This is especially important in rule-based systems
such as Learning Fuzzy-Classifier Systems (LFCSs) where the selection and
application of rules directly determine prediction accuracy and reliability.
LFCSs combine evolutionary algorithms with supervised learning to optimize
fuzzy classification rules, offering enhanced interpretability and robustness.
Despite these advantages, research on improving decision-making mechanisms
(i.e., class inference schemes) in LFCSs remains limited. Most LFCSs use
voting-based or single-winner-based inference schemes. These schemes rely on
classification performance on training data and may not perform well on unseen
data, risking overfitting. To address these limitations, this article
introduces a novel class inference scheme for LFCSs based on the
Dempster-Shafer Theory of Evidence (DS theory). The proposed scheme handles
uncertainty well. By using the DS theory, the scheme calculates belief masses
(i.e., measures of belief) for each specific class and the ``I don't know''
state from each fuzzy rule and infers a class from these belief masses. Unlike
the conventional schemes, the proposed scheme also considers the ``I don't
know'' state that reflects uncertainty, thereby improving the transparency and
reliability of LFCSs. Applied to a variant of LFCS (i.e., Fuzzy-UCS), the
proposed scheme demonstrates statistically significant improvements in terms of
test macro F1 scores across 30 real-world datasets compared to conventional
voting-based and single-winner-based fuzzy inference schemes. It forms smoother
decision boundaries, provides reliable confidence measures, and enhances the
robustness and generalizability of LFCSs in real-world applications. Our
implementation is available at https://github.com/YNU-NakataLab/jUCS.

### 5. [Adapting Rule Representation With Four-Parameter Beta Distribution for Learning Classifier Systems](http://arxiv.org/pdf/2506.03602v1)

Authors: Hiroki Shiraishi, Yohei Hayamizu, Tomonori Hashiyama, Keiki Takadama, Hisao Ishibuchi, Masaya Nakata

Rule representations significantly influence the search capabilities and
decision boundaries within the search space of Learning Classifier Systems
(LCSs), a family of rule-based machine learning systems that evolve
interpretable models through evolutionary processes. However, it is very
difficult to choose an appropriate rule representation for each problem.
Additionally, some problems benefit from using different representations for
different subspaces within the input space. Thus, an adaptive mechanism is
needed to choose an appropriate rule representation for each rule in LCSs. This
article introduces a flexible rule representation using a four-parameter beta
distribution and integrates it into a fuzzy-style LCS. The four-parameter beta
distribution can form various function shapes, and this flexibility enables our
LCS to automatically select appropriate representations for different
subspaces. Our rule representation can represent crisp/fuzzy decision
boundaries in various boundary shapes, such as rectangles and bells, by
controlling four parameters, compared to the standard representations such as
trapezoidal ones. Leveraging this flexibility, our LCS is designed to adapt the
appropriate rule representation for each subspace. Moreover, our LCS
incorporates a generalization bias favoring crisp rules where feasible,
enhancing model interpretability without compromising accuracy. Experimental
results on real-world classification tasks show that our LCS achieves
significantly superior test accuracy and produces more compact rule sets. Our
implementation is available at https://github.com/YNU-NakataLab/Beta4-UCS. An
extended abstract related to this work is available at
https://doi.org/10.36227/techrxiv.174900805.59801248/v1.

### Networking and Internet Architecture

### 1. [A Model-Data Dual-Driven Resource Allocation Scheme for IREE Oriented 6G Networks](http://arxiv.org/pdf/2506.03508v1)

Authors: Tao Yu, Simin Wang, Shunqing Zhang, Xiaojing Chen, Zi Xu, Xin Wang, Jiandong Li, Junyu Liu, Sihai Zhang

The rapid and substantial fluctuations in wireless network capacity and
traffic demand, driven by the emergence of 6G technologies, have exacerbated
the issue of traffic-capacity mismatch, raising concerns about wireless network
energy consumption. To address this challenge, we propose a model-data
dual-driven resource allocation (MDDRA) algorithm aimed at maximizing the
integrated relative energy efficiency (IREE) metric under dynamic traffic
conditions. Unlike conventional model-driven or data-driven schemes, the
proposed MDDRA framework employs a model-driven Lyapunov queue to accumulate
long-term historical mismatch information and a data-driven Graph Radial bAsis
Fourier (GRAF) network to predict the traffic variations under incomplete data,
and hence eliminates the reliance on high-precision models and complete
spatial-temporal traffic data. We establish the universal approximation
property of the proposed GRAF network and provide convergence and complexity
analysis for the MDDRA algorithm. Numerical experiments validate the
performance gains achieved through the data-driven and model-driven components.
By analyzing IREE and EE curves under diverse traffic conditions, we recommend
that network operators shall spend more efforts to balance the traffic demand
and the network capacity distribution to ensure the network performance,
particularly in scenarios with large speed limits and higher driving
visibility.

### 2. [Carbon-Aware Temporal Data Transfer Scheduling Across Cloud Datacenters](http://arxiv.org/pdf/2506.04117v1)

Authors: Elvis Rodrigues, Jacob Goldverg, Tevfik Kosar

Inter-datacenter communication is a significant part of cloud operations and
produces a substantial amount of carbon emissions for cloud data centers, where
the environmental impact has already been a pressing issue. In this paper, we
present a novel carbon-aware temporal data transfer scheduling framework,
called LinTS, which promises to significantly reduce the carbon emission of
data transfers between cloud data centers. LinTS produces a competitive
transfer schedule and makes scaling decisions, outperforming common heuristic
algorithms. LinTS can lower carbon emissions during inter-datacenter transfers
by up to 66% compared to the worst case and up to 15% compared to other
solutions while preserving all deadline constraints.

### Robotics

### 1. [Robust Position Estimation by Rao-Blackwellized Particle Filter without Integer Ambiguity Resolution in Urban Environments](http://arxiv.org/pdf/2506.03537v1)

Authors: Daiki Niimi, An Fujino, Taro Suzuki, Junichi Meguro

This study proposes a centimeter-accurate positioning method that utilizes a
Rao-Blackwellized particle filter (RBPF) without requiring integer ambiguity
resolution in global navigation satellite system (GNSS) carrier phase
measurements. The conventional positioning method employing a particle filter
(PF) eliminates the necessity for ambiguity resolution by calculating the
likelihood from the residuals of the carrier phase based on the particle
position. However, this method encounters challenges, particularly in urban
environments characterized by non-line-of-sight (NLOS) multipath errors. In
such scenarios, PF tracking may fail due to the degradation of velocity
estimation accuracy used for state transitions, thereby complicating subsequent
position estimation. To address this issue, we apply Rao-Blackwellization to
the conventional PF framework, treating position and velocity as distinct
states and employing the Kalman filter for velocity estimation. This approach
enhances the accuracy of velocity estimation and, consequently, the precision
of position estimation. Moreover, the proposed method rejects NLOS multipath
signals based on the pseudorange residuals at each particle position during the
velocity estimation step. This process not only enhances velocity accuracy, but
also preserves particle diversity by allowing particles to transition to unique
states with varying velocities. Consequently, particles are more likely to
cluster around the true position, thereby enabling more accurate position
estimation. Vehicular experiments in urban environments demonstrated the
effectiveness of proposed method in achieving a higher positioning accuracy
than conventional PF-based and conventional GNSS positioning methods.

### 2. [SwitchVLA: Execution-Aware Task Switching for Vision-Language-Action Models](http://arxiv.org/pdf/2506.03574v1)

Authors: Meng Li, Zhen Zhao, Zhengping Che, Fei Liao, Kun Wu, Zhiyuan Xu, Pei Ren, Zhao Jin, Ning Liu, Jian Tang

Robots deployed in dynamic environments must be able to not only follow
diverse language instructions but flexibly adapt when user intent changes
mid-execution. While recent Vision-Language-Action (VLA) models have advanced
multi-task learning and instruction following, they typically assume static
task intent, failing to respond when new instructions arrive during ongoing
execution. This limitation hinders natural and robust interaction in dynamic
settings, such as retail or household environments, where real-time intent
changes are common. We propose SwitchVLA, a unified, execution-aware framework
that enables smooth and reactive task switching without external planners or
additional switch-specific data. We model task switching as a behavior
modulation problem conditioned on execution state and instruction context.
Expert demonstrations are segmented into temporally grounded contact phases,
allowing the policy to infer task progress and adjust its behavior accordingly.
A multi-behavior conditional policy is then trained to generate flexible action
chunks under varying behavior modes through conditioned trajectory modeling.
Experiments in both simulation and real-world robotic manipulation demonstrate
that SwitchVLA enables robust instruction adherence, fluid task switching, and
strong generalization-outperforming prior VLA baselines in both task success
rate and interaction naturalness.

### 3. [An Improved Grey Wolf Optimizer Inspired by Advanced Cooperative Predation for UAV Shortest Path Planning](http://arxiv.org/pdf/2506.03663v1)

Authors: Zuhao Teng, Qian Dong, Ze Zhang, Shuangyao Huang, Wenzhang Zhang, Jingchen Wang, Ji Li, Xi Chen

With the widespread application of Unmanned Aerial Vehicles (UAVs) in domains
like military reconnaissance, emergency rescue, and logistics delivery,
efficiently planning the shortest flight path has become a critical challenge.
Traditional heuristic-based methods often suffer from the inability to escape
from local optima, which limits their effectiveness in finding the shortest
path. To address these issues, a novel Improved Grey Wolf Optimizer (IGWO) is
presented in this study. The proposed IGWO incorporates an Advanced Cooperative
Predation (ACP) and a Lens Opposition-based Learning Strategy (LOBL) in order
to improve the optimization capability of the method. Simulation results show
that IGWO ranks first in optimization performance on benchmark functions F1-F5,
F7, and F9-F12, outperforming all other compared algorithms. Subsequently, IGWO
is applied to UAV shortest path planning in various obstacle-laden
environments. Simulation results show that the paths planned by IGWO are, on
average, shorter than those planned by GWO, PSO, and WOA by 1.70m, 1.68m, and
2.00m, respectively, across four different maps.

### 4. [An Open-source Capping Machine Suitable for Confined Spaces](http://arxiv.org/pdf/2506.03743v1)

Authors: Francisco Munguia-Galeano, Louis Longley, Satheeshkumar Veeramani, Zhengxue Zhou, Rob Clowes, Hatem Fakhruldeen, Andrew I. Cooper

In the context of self-driving laboratories (SDLs), ensuring automated and
error-free capping is crucial, as it is a ubiquitous step in sample
preparation. Automated capping in SDLs can occur in both large and small
workspaces (e.g., inside a fume hood). However, most commercial capping
machines are designed primarily for large spaces and are often too bulky for
confined environments. Moreover, many commercial products are closed-source,
which can make their integration into fully autonomous workflows difficult.
This paper introduces an open-source capping machine suitable for compact
spaces, which also integrates a vision system that recognises capping failure.
The capping and uncapping processes are repeated 100 times each to validate the
machine's design and performance. As a result, the capping machine reached a
100 % success rate for capping and uncapping. Furthermore, the machine sealing
capacities are evaluated by capping 12 vials filled with solvents of different
vapour pressures: water, ethanol and acetone. The vials are then weighed every
3 hours for three days. The machine's performance is benchmarked against an
industrial capping machine (a Chemspeed station) and manual capping. The vials
capped with the prototype lost 0.54 % of their content weight on average per
day, while the ones capped with the Chemspeed and manually lost 0.0078 % and
0.013 %, respectively. The results show that the capping machine is a
reasonable alternative to industrial and manual capping, especially when space
and budget are limitations in SDLs.

### 5. [Understanding Physical Properties of Unseen Deformable Objects by Leveraging Large Language Models and Robot Actions](http://arxiv.org/pdf/2506.03760v1)

Authors: Changmin Park, Beomjoon Lee, Haechan Jung, Haejin Jung, Changjoo Nam

In this paper, we consider the problem of understanding the physical
properties of unseen objects through interactions between the objects and a
robot. Handling unseen objects with special properties such as deformability is
challenging for traditional task and motion planning approaches as they are
often with the closed world assumption. Recent results in Large Language Models
(LLMs) based task planning have shown the ability to reason about unseen
objects. However, most studies assume rigid objects, overlooking their physical
properties. We propose an LLM-based method for probing the physical properties
of unseen deformable objects for the purpose of task planning. For a given set
of object properties (e.g., foldability, bendability), our method uses robot
actions to determine the properties by interacting with the objects. Based on
the properties examined by the LLM and robot actions, the LLM generates a task
plan for a specific domain such as object packing. In the experiment, we show
that the proposed method can identify properties of deformable objects, which
are further used for a bin-packing task where the properties take crucial roles
to succeed.

### 6. [Enhancing Safety of Foundation Models for Visual Navigation through Collision Avoidance via Repulsive Estimation](http://arxiv.org/pdf/2506.03834v1)

Authors: Joonkyung Kim, Joonyeol Sim, Woojun Kim, Katia Sycara, Changjoo Nam

We propose CARE (Collision Avoidance via Repulsive Estimation), a
plug-and-play module that enhances the safety of vision-based navigation
without requiring additional range sensors or fine-tuning of pretrained models.
While recent foundation models using only RGB inputs have shown strong
performance, they often fail to generalize in out-of-distribution (OOD)
environments with unseen objects or variations in camera parameters (e.g.,
field of view, pose, or focal length). Without fine-tuning, these models may
generate unsafe trajectories that lead to collisions, requiring costly data
collection and retraining. CARE addresses this limitation by seamlessly
integrating with any RGB-based navigation system that outputs local
trajectories, dynamically adjusting them using repulsive force vectors derived
from monocular depth maps. We evaluate CARE by combining it with
state-of-the-art vision-based navigation models across multiple robot
platforms. CARE consistently reduces collision rates (up to 100%) without
sacrificing goal-reaching performance and improves collision-free travel
distance by up to 10.7x in exploration tasks.

### 7. [Phase-based Nonlinear Model Predictive Control for Humanoid Walking Stabilization with Single and Double Support Time Adjustments](http://arxiv.org/pdf/2506.03856v1)

Authors: Kwanwoo Lee, Gyeongjae Park, Jaeheung Park

Balance control for humanoid robots has been extensively studied to enable
robots to navigate in real-world environments. However, balance controllers
that explicitly optimize the durations of both the single support phase, also
known as step timing, and the Double Support Phase (DSP) have not been widely
explored due to the inherent nonlinearity of the associated optimization
problem. Consequently, many recent approaches either ignore the DSP or adjust
its duration based on heuristics or on linearization techniques that rely on
sequential coordination of balance strategies. This study proposes a novel
phase-based nonlinear Model Predictive Control (MPC) framework that
simultaneously optimizes Zero Moment Point~(ZMP) modulation, step location,
step timing, and DSP duration to maintain balance under external disturbances.
In simulation, the proposed controller was compared with two state-of-the-art
frameworks that rely on heuristics or sequential coordination of balance
strategies under two scenarios: forward walking on terrain emulating compliant
ground and external push recovery while walking in place. Overall, the findings
suggest that the proposed method offers more flexible coordination of balance
strategies than the sequential approach, and consistently outperforms the
heuristic approach. The robustness and effectiveness of the proposed controller
were also validated through experiments with a real humanoid robot.

### 8. [FLIP: Flowability-Informed Powder Weighing](http://arxiv.org/pdf/2506.03896v1)

Authors: Nikola Radulov, Alex Wright, Thomas little, Andrew I. Cooper, Gabriella Pizzuto

Autonomous manipulation of powders remains a significant challenge for
robotic automation in scientific laboratories. The inherent variability and
complex physical interactions of powders in flow, coupled with variability in
laboratory conditions necessitates adaptive automation. This work introduces
FLIP, a flowability-informed powder weighing framework designed to enhance
robotic policy learning for granular material handling. Our key contribution
lies in using material flowability, quantified by the angle of repose, to
optimise physics-based simulations through Bayesian inference. This yields
material-specific simulation environments capable of generating accurate
training data, which reflects diverse powder behaviours, for training `robot
chemists'. Building on this, FLIP integrates quantified flowability into a
curriculum learning strategy, fostering efficient acquisition of robust robotic
policies by gradually introducing more challenging, less flowable powders. We
validate the efficacy of our method on a robotic powder weighing task under
real-world laboratory conditions. Experimental results show that FLIP with a
curriculum strategy achieves a low dispensing error of 2.12 +- 1.53 mg,
outperforming methods that do not leverage flowability data, such as domain
randomisation (6.11 +- 3.92 mg). These results demonstrate FLIP's improved
ability to generalise to previously unseen, more cohesive powders and to new
target masses.

### 9. [A Bi-Level Optimization Method for Redundant Dual-Arm Minimum Time Problems](http://arxiv.org/pdf/2506.03982v1)

Authors: Jonathan Fried, Santiago Paternain

In this work, we present a method for minimizing the time required for a
redundant dual-arm robot to follow a desired relative Cartesian path at
constant path speed by optimizing its joint trajectories, subject to position,
velocity, and acceleration limits. The problem is reformulated as a bi-level
optimization whose lower level is a convex, closed-form subproblem that
maximizes path speed for a fixed trajectory, while the upper level updates the
trajectory using a single-chain kinematic formulation and the subgradient of
the lower-level value. Numerical results demonstrate the effectiveness of the
proposed approach.

### 10. [SemNav: A Model-Based Planner for Zero-Shot Object Goal Navigation Using Vision-Foundation Models](http://arxiv.org/pdf/2506.03516v1)

Authors: Arnab Debnath, Gregory J. Stein, Jana Kosecka

Object goal navigation is a fundamental task in embodied AI, where an agent
is instructed to locate a target object in an unexplored environment.
Traditional learning-based methods rely heavily on large-scale annotated data
or require extensive interaction with the environment in a reinforcement
learning setting, often failing to generalize to novel environments and
limiting scalability. To overcome these challenges, we explore a zero-shot
setting where the agent operates without task-specific training, enabling more
scalable and adaptable solution. Recent advances in Vision Foundation Models
(VFMs) offer powerful capabilities for visual understanding and reasoning,
making them ideal for agents to comprehend scenes, identify relevant regions,
and infer the likely locations of objects. In this work, we present a zero-shot
object goal navigation framework that integrates the perceptual strength of
VFMs with a model-based planner that is capable of long-horizon decision making
through frontier exploration. We evaluate our approach on the HM3D dataset
using the Habitat simulator and demonstrate that our method achieves
state-of-the-art performance in terms of success weighted by path length for
zero-shot object goal navigation.

### Software Engineering

### 1. [Beyond C/C++: Probabilistic and LLM Methods for Next-Generation Software Reverse Engineering](http://arxiv.org/pdf/2506.03504v1)

Authors: Zhuo Zhuo, Xiangyu Zhang

This proposal discusses the growing challenges in reverse engineering modern
software binaries, particularly those compiled from newer system programming
languages such as Rust, Go, and Mojo. Traditional reverse engineering
techniques, developed with a focus on C and C++, fall short when applied to
these newer languages due to their reliance on outdated heuristics and failure
to fully utilize the rich semantic information embedded in binary programs.
These challenges are exacerbated by the limitations of current data-driven
methods, which are susceptible to generating inaccurate results, commonly
referred to as hallucinations. To overcome these limitations, we propose a
novel approach that integrates probabilistic binary analysis with fine-tuned
large language models (LLMs). Our method systematically models the
uncertainties inherent in reverse engineering, enabling more accurate reasoning
about incomplete or ambiguous information. By incorporating LLMs, we extend the
analysis beyond traditional heuristics, allowing for more creative and
context-aware inferences, particularly for binaries from diverse programming
languages. This hybrid approach not only enhances the robustness and accuracy
of reverse engineering efforts but also offers a scalable solution adaptable to
the rapidly evolving landscape of software development.

### 2. [Across Programming Language Silos: A Study on Cross-Lingual Retrieval-augmented Code Generation](http://arxiv.org/pdf/2506.03535v1)

Authors: Qiming Zhu, Jialun Cao, Xuanang Chen, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Shing-Chi Cheung

Current research on large language models (LLMs) with retrieval-augmented
code generation (RACG) mainly focuses on single-language settings, leaving
cross-lingual effectiveness and security unexplored. Multi-lingual RACG systems
are valuable for migrating code-bases across programming languages (PLs), yet
face risks from error (e.g. adversarial data corruption) propagation in
cross-lingual transfer. We construct a dataset spanning 13 PLs with nearly 14k
instances to explore utility and robustness of multi-lingual RACG systems. Our
investigation reveals four key insights: (1) Effectiveness: multi-lingual RACG
significantly enhances multi-lingual code LLMs generation; (2) Inequality: Java
demonstrate superior cross-lingual utility over Python in RACG; (3) Robustness:
Adversarial attacks degrade performance significantly in mono-lingual RACG but
show mitigated impacts in cross-lingual scenarios; Counterintuitively,
perturbed code may improve RACG in cross-lingual scenarios; (4) Specialization:
Domain-specific code retrievers outperform significantly general text
retrievers. These findings establish foundation for developing effective and
secure multi-lingual code assistants.

### 3. [Improving LLM-Based Fault Localization with External Memory and Project Context](http://arxiv.org/pdf/2506.03585v1)

Authors: Inseok Yeo, Duksan Ryu, Jongmoon Baik

Fault localization, the process of identifying the software components
responsible for failures, is essential but often time-consuming. Recent
advances in Large Language Models (LLMs) have enabled fault localization
without extensive defect datasets or model fine-tuning. However, existing
LLM-based methods rely only on general LLM capabilities and lack integration of
project-specific knowledge, resulting in limited effectiveness, especially for
complex software.
  We introduce MemFL, a novel approach that enhances LLM-based fault
localization by integrating project-specific knowledge via external memory.
This memory includes static summaries of the project and dynamic, iterative
debugging insights gathered from previous attempts. By leveraging external
memory, MemFL simplifies debugging into three streamlined steps, significantly
improving efficiency and accuracy. Iterative refinement through dynamic memory
further enhances reasoning quality over time.
  Evaluated on the Defects4J benchmark, MemFL using GPT-4o-mini localized 12.7%
more bugs than current LLM-based methods, achieving this improvement with just
21% of the execution time (17.4 seconds per bug) and 33% of the API cost
(0.0033 dollars per bug). On complex projects, MemFL's advantage increased to
27.6%. Additionally, MemFL with GPT-4.1-mini outperformed existing methods by
24.4%, requiring only 24.7 seconds and 0.0094 dollars per bug. MemFL thus
demonstrates significant improvements by effectively incorporating
project-specific knowledge into LLM-based fault localization, delivering high
accuracy with reduced time and cost.

### 4. [A Two-Staged LLM-Based Framework for CI/CD Failure Detection and Remediation with Industrial Validation](http://arxiv.org/pdf/2506.03691v1)

Authors: Weiyuan Xu, Juntao Luo, Tao Huang, Kaixin Sui, Jie Geng, Qijun Ma, Isami Akasaka, Xiaoxue Shi, Jing Tang, Peng Cai

Continuous Integration and Continuous Deployment (CI/CD) pipelines are
pivotal to modern software engineering, yet diagnosing and resolving their
failures remains a complex and labor-intensive challenge. In this paper, we
present LogSage, the first end-to-end LLM-powered framework that performs root
cause analysis and solution generation from failed CI/CD pipeline logs. During
the root cause analysis stage, LogSage employs a specialized log preprocessing
pipeline tailored for LLMs, which extracts critical error logs and eliminates
noise to enhance the precision of LLM-driven root cause analysis. In the
solution generation stage, LogSage leverages RAG to integrate historical
resolution strategies and utilizes tool-calling to deliver actionable,
automated fixes. We evaluated the root cause analysis stage using a newly
curated open-source dataset, achieving 98\% in precision and 12\% improvement
over naively designed LLM-based log analysis baselines, while attaining
near-perfect recall. The end-to-end system was rigorously validated in a
large-scale industrial CI/CD environment of production quality, processing more
than 3,000 executions daily and accumulating more than 1.07 million executions
in its first year of deployment, with end-to-end precision exceeding 88\%.
These two forms of evaluation confirm that LogSage providing a scalable and
practical solution to manage CI/CD pipeline failures in real-world DevOps
workflows.

### 5. [Differences between Neurodivergent and Neurotypical Software Engineers: Analyzing the 2022 Stack Overflow Survey](http://arxiv.org/pdf/2506.03840v1)

Authors: Pragya Verma, Marcos Vinicius Cruz, Grischa Liebel

Neurodiversity describes variation in brain function among people, including
common conditions such as Autism spectrum disorder (ASD), Attention deficit
hyperactivity disorder (ADHD), and dyslexia. While Software Engineering (SE)
literature has started to explore the experiences of neurodivergent software
engineers, there is a lack of research that compares their challenges to those
of neurotypical software engineers. To address this gap, we analyze existing
data from the 2022 Stack Overflow Developer survey that collected data on
neurodiversity. We quantitatively compare the answers of professional engineers
with ASD (n=374), ADHD (n=1305), and dyslexia (n=363) with neurotypical
engineers. Our findings indicate that neurodivergent engineers face more
difficulties than neurotypical engineers. Specifically, engineers with ADHD
report that they face more interruptions caused by waiting for answers, and
that they less frequently interact with individuals outside their team. This
study provides a baseline for future research comparing neurodivergent
engineers with neurotypical ones. Several factors in the Stack Overflow survey
and in our analysis are likely to lead to conservative estimates of the actual
effects between neurodivergent and neurotypical engineers, e.g., the effects of
the COVID-19 pandemic and our focus on employed professionals.

### 6. [Automated Mechanism to Support Trade Transactions in Smart Contracts with Upgrade and Repair](http://arxiv.org/pdf/2506.03877v1)

Authors: Christian Gang Liu, Peter Bodorik, Dawn Jutla

In our previous research, we addressed the problem of automated
transformation of models, represented using the business process model and
notation (BPMN) standard, into the methods of a smart contract. The
transformation supports BPMN models that contain complex multi-step activities
that are supported using our concept of multi-step nested trade transactions,
wherein the transactional properties are enforced by a mechanism generated
automatically by the transformation process from a BPMN model to a smart
contract. In this paper, we present a methodology for repairing a smart
contract that cannot be completed due to events that were not anticipated by
the developer and thus prevent the completion of the smart contract. The repair
process starts with the original BPMN model fragment causing the issue,
providing the modeler with the innermost transaction fragment containing the
failed activity. The modeler amends the BPMN pattern on the basis of successful
completion of previous activities. If repairs exceed the inner transaction's
scope, they are addressed using the parent transaction's BPMN model. The
amended BPMN model is then transformed into a new smart contract, ensuring
consistent data and logic transitions. We previously developed a tool, called
TABS+, as a proof of concept (PoC) to transform BPMN models into smart
contracts for nested transactions. This paper describes the tool TABS+R,
developed by extending the TABS+ tool, to allow the repair of smart contracts.

### 7. [Multi-Language Detection of Design Pattern Instances](http://arxiv.org/pdf/2506.03903v1)

Authors: Hugo Andrade, João Bispo, Filipe F. Correia

Code comprehension is often supported by source code analysis tools which
provide more abstract views over software systems, such as those detecting
design patterns. These tools encompass analysis of source code and ensuing
extraction of relevant information. However, the analysis of the source code is
often specific to the target programming language.
  We propose DP-LARA, a multi-language pattern detection tool that uses the
multi-language capability of the LARA framework to support finding pattern
instances in a code base. LARA provides a virtual AST, which is common to
multiple OOP programming languages, and DP-LARA then performs code analysis of
detecting pattern instances on this abstract representation.
  We evaluate the detection performance and consistency of DP-LARA with a few
software projects. Results show that a multi-language approach does not
compromise detection performance, and DP-LARA is consistent across the
languages we tested it for (i.e., Java and C/C++). Moreover, by providing a
virtual AST as the abstract representation, we believe to have decreased the
effort of extending the tool to new programming languages and maintaining
existing ones.

### 8. [Solsmith: Solidity Random Program Generator for Compiler Testing](http://arxiv.org/pdf/2506.03909v1)

Authors: Lantian Li, Zhihao Liu, Zhongxing Yu

Smart contracts are computer programs that run on blockchain platforms, with
Solidity being the most widely used language for their development. As
blockchain technology advances, smart contracts have become increasingly
important across various fields. In order for smart contracts to operate
correctly, the correctness of the compiler is particularly crucial. Although
some research efforts have been devoted to testing Solidity compilers, they
primarily focus on testing methods and do not address the core issue of
generating test programs. To fill this gap, this paper designs and implements
Solsmith, a test program generator specifically aimed at uncovering defects in
Solidity compilers. It tests the compiler correctness by generating valid and
diverse Solidity programs. We have designed a series of unique program
generation strategies tailored to Solidity, including enabling optimizations
more frequently, avoiding undefined behaviour, and mitigating behavioural
differences caused by intermediate representations. To validate the
effectiveness of Solsmith, we assess the effectiveness of the test programs
generated by Solsmith using the approach of differential testing. The
preliminary results show that Solsmith can generate the expected test programs
and uncover four confirmed defects in Solidity compilers, demonstrating the
effectiveness and potential of Solsmith.

### 9. [Boosting Open-Source LLMs for Program Repair via Reasoning Transfer and LLM-Guided Reinforcement Learning](http://arxiv.org/pdf/2506.03921v1)

Authors: Xunzhu Tang, Jacques Klein, Tegawendé F. Bissyandé

Several closed-source LLMs have consistently outperformed open-source
alternatives in program repair tasks, primarily due to their superior reasoning
capabilities and extensive pre-training. This paper introduces Repairity, a
novel three-stage methodology that significantly narrows this performance gap
through reasoning extraction and reinforcement learning. Our approach: (1)
systematically filters high-quality reasoning traces from closed-source models
using correctness verification, (2) transfers this reasoning knowledge to
open-source models via supervised fine-tuning, and (3) develops reinforcement
learning with LLM-based feedback to further optimize performance. Empirical
evaluation across multiple program repair benchmarks demonstrates that
Repairity improves the performance of Qwen2.5-Coder-32B-Instruct, a base open
source LLM, by 8.68\% on average, reducing the capability gap with
Claude-Sonnet3.7, a state-of-the-art closed-source model, from 10.05% to 1.35%.
Ablation studies confirm that both reasoning extraction and LLM-guided
reinforcement learning contribute significantly to these improvements. Our
methodology generalizes effectively to additional code-related tasks, enabling
organizations to leverage high-quality program repair capabilities while
maintaining the customizability, transparency, and deployment flexibility
inherent to open-source models.

### 10. [Automatic Multi-level Feature Tree Construction for Domain-Specific Reusable Artifacts Management](http://arxiv.org/pdf/2506.03946v1)

Authors: Dongming Jin, Zhi Jin, Nianyu Li, Kai Yang, Linyu Li, Suijing Guan

With the rapid growth of open-source ecosystems (e.g., Linux) and
domain-specific software projects (e.g., aerospace), efficient management of
reusable artifacts is becoming increasingly crucial for software reuse. The
multi-level feature tree enables semantic management based on functionality and
supports requirements-driven artifact selection. However, constructing such a
tree heavily relies on domain expertise, which is time-consuming and
labor-intensive. To address this issue, this paper proposes an automatic
multi-level feature tree construction framework named FTBUILDER, which consists
of three stages. It automatically crawls domain-specific software repositories
and merges their metadata to construct a structured artifact library. It
employs clustering algorithms to identify a set of artifacts with common
features. It constructs a prompt and uses LLMs to summarize their common
features. FTBUILDER recursively applies the identification and summarization
stages to construct a multi-level feature tree from the bottom up. To validate
FTBUILDER, we conduct experiments from multiple aspects (e.g., tree quality and
time cost) using the Linux distribution ecosystem. Specifically, we first
simultaneously develop and evaluate 24 alternative solutions in the FTBUILDER.
We then construct a three-level feature tree using the best solution among
them. Compared to the official feature tree, our tree exhibits higher quality,
with a 9% improvement in the silhouette coefficient and an 11% increase in
GValue. Furthermore, it can save developers more time in selecting artifacts by
26% and improve the accuracy of artifact recommendations with GPT-4 by 235%.
FTBUILDER can be extended to other open-source software communities and
domain-specific industrial enterprises.

### Social and Information Networks

### 1. [Modeling Bulimia Nervosa in the Digital Age: The Role of Social Media](http://arxiv.org/pdf/2506.03491v1)

Authors: Brenda Murillo, Fabio Sanchez

Globalization has fundamentally reshaped societal dynamics, influencing how
individuals interact and perceive themselves and others. One significant
consequence is the evolving landscape of eating disorders such as bulimia
nervosa (BN), which are increasingly driven not just by internal psychological
factors but by broader sociocultural and digital contexts. While mathematical
modeling has provided valuable insights, traditional frameworks often fall
short in capturing the nuanced roles of social contagion, digital media, and
adaptive behavior. This review synthesizes two decades of quantitative modeling
efforts, including compartmental, stochastic, and delay-based approaches. We
spotlight foundational work that conceptualizes BN as a socially transmissible
condition and identify critical gaps, especially regarding the intensifying
impact of social media. Drawing on behavioral epidemiology and the adaptive
behavior framework by Fenichel et al., we advocate for a new generation of
models that incorporate feedback mechanisms, content-driven influence
functions, and dynamic network effects. This work outlines a roadmap for
developing more realistic, data-informed models that can guide effective public
health interventions in the digital era.

### 2. [A Retrieval-Augmented Multi-Agent Framework for Psychiatry Diagnosis](http://arxiv.org/pdf/2506.03750v1)

Authors: Mengxi Xiao, Mang Ye, Ben Liu, Xiaofen Zong, He Li, Jimin Huang, Qianqian Xie, Min Peng

The application of AI in psychiatric diagnosis faces significant challenges,
including the subjective nature of mental health assessments, symptom overlap
across disorders, and privacy constraints limiting data availability. To
address these issues, we present MoodAngels, the first specialized multi-agent
framework for mood disorder diagnosis. Our approach combines granular-scale
analysis of clinical assessments with a structured verification process,
enabling more accurate interpretation of complex psychiatric data.
Complementing this framework, we introduce MoodSyn, an open-source dataset of
1,173 synthetic psychiatric cases that preserves clinical validity while
ensuring patient privacy. Experimental results demonstrate that MoodAngels
outperforms conventional methods, with our baseline agent achieving 12.3%
higher accuracy than GPT-4o on real-world cases, and our full multi-agent
system delivering further improvements. Evaluation in the MoodSyn dataset
demonstrates exceptional fidelity, accurately reproducing both the core
statistical patterns and complex relationships present in the original data
while maintaining strong utility for machine learning applications. Together,
these contributions provide both an advanced diagnostic tool and a critical
research resource for computational psychiatry, bridging important gaps in
AI-assisted mental health assessment.

### 3. [The Impact of COVID-19 on Twitter Ego Networks: Structure, Sentiment, and Topics](http://arxiv.org/pdf/2506.03788v1)

Authors: Kamer Cekini, Elisabetta Biondi, Chiara Boldrini, Andrea Passarella, Marco Conti

Lockdown measures, implemented by governments during the initial phases of
the COVID-19 pandemic to reduce physical contact and limit viral spread,
imposed significant restrictions on in-person social interactions.
Consequently, individuals turned to online social platforms to maintain
connections. Ego networks, which model the organization of personal
relationships according to human cognitive constraints on managing meaningful
interactions, provide a framework for analyzing such dynamics. The disruption
of physical contact and the predominant shift of social life online potentially
altered the allocation of cognitive resources dedicated to managing these
digital relationships. This research aims to investigate the impact of lockdown
measures on the characteristics of online ego networks, presumably resulting
from this reallocation of cognitive resources. To this end, a large dataset of
Twitter users was examined, covering a seven-year period of activity. Analyzing
a seven-year Twitter dataset -- including five years pre-pandemic and two years
post -- we observe clear, though temporary, changes. During lockdown, ego
networks expanded, social circles became more structured, and relationships
intensified. Simultaneously, negative interactions increased, and users engaged
with a broader range of topics, indicating greater thematic diversity. Once
restrictions were lifted, these structural, emotional, and thematic shifts
largely reverted to pre-pandemic norms -- suggesting a temporary adaptation to
an extraordinary social context.

### 4. [GA-S$^3$: Comprehensive Social Network Simulation with Group Agents](http://arxiv.org/pdf/2506.03532v1)

Authors: Yunyao Zhang, Zikai Song, Hang Zhou, Wenfeng Ren, Yi-Ping Phoebe Chen, Junqing Yu, Wei Yang

Social network simulation is developed to provide a comprehensive
understanding of social networks in the real world, which can be leveraged for
a wide range of applications such as group behavior emergence, policy
optimization, and business strategy development. However, billions of
individuals and their evolving interactions involved in social networks pose
challenges in accurately reflecting real-world complexities. In this study, we
propose a comprehensive Social Network Simulation System (GA-S3) that leverages
newly designed Group Agents to make intelligent decisions regarding various
online events. Unlike other intelligent agents that represent an individual
entity, our group agents model a collection of individuals exhibiting similar
behaviors, facilitating the simulation of large-scale network phenomena with
complex interactions at a manageable computational cost. Additionally, we have
constructed a social network benchmark from 2024 popular online events that
contains fine-grained information on Internet traffic variations. The
experiment demonstrates that our approach is capable of achieving accurate and
highly realistic prediction results. Code is open at
https://github.com/AI4SS/GAS-3.

### Systems and Control

### 1. [Topology-Aware Graph Neural Network-based State Estimation for PMU-Unobservable Power Systems](http://arxiv.org/pdf/2506.03493v1)

Authors: Shiva Moshtagh, Behrouz Azimian, Mohammad Golgol, Anamitra Pal

Traditional optimization-based techniques for time-synchronized state
estimation (SE) often suffer from high online computational burden, limited
phasor measurement unit (PMU) coverage, and presence of non-Gaussian
measurement noise. Although conventional learning-based models have been
developed to overcome these challenges, they are negatively impacted by
topology changes and real-time data loss. This paper proposes a novel deep
geometric learning approach based on graph neural networks (GNNs) to estimate
the states of PMU-unobservable power systems. The proposed approach combines
graph convolution and multi-head graph attention layers inside a customized
end-to-end learning framework to handle topology changes and real-time data
loss. An upper bound on SE error as a function of topology change is also
derived. Experimental results for different test systems demonstrate
superiority of the proposed customized GNN-SE (CGNN-SE) over traditional
optimization-based techniques as well as conventional learning-based models in
presence of topology changes, PMU failures, bad data, non-Gaussian measurement
noise, and large system implementation.

### 2. [Fast Sampling for System Identification: Overcoming Noise, Offsets, and Closed-Loop Challenges with State Variable Filter](http://arxiv.org/pdf/2506.03650v1)

Authors: Ichiro Maruta, Toshiharu Sugie

This paper investigates the effects of setting the sampling frequency
significantly higher than conventional guidelines in system identification.
Although continuous-time identification methods resolve the numerical
difficulties encountered in discrete-time approaches when employing fast
sampling (e.g., the problems caused by all poles approaching unity), the
potential benefits of using sampling frequencies that far exceed traditional
rules like the "ten times the bandwidth" guideline remained largely unexplored.
We show that using a state variable filter (SVF)-like least squares approach,
the variance of the estimation error scales as $O(h)$ with the sampling
interval $h$. Importantly, this scaling holds even with colored noise or noise
correlations between variables. Thus, increasing the sampling frequency and
applying the SVF method offers a novel solution for challenging problems such
as closed-loop system identification and measurements with offsets. Theoretical
findings are supported by numerical examples, including the closed-loop
identification of unstable multi-input multi-output (MIMO) systems.

### 3. [Stabilization of Linear Switched Systems with Long Input Delay via Average or Averaging Predictor Feedbacks](http://arxiv.org/pdf/2506.03908v1)

Authors: Andreas Katsanikakis, Nikolaos Bekiaris-Liberis

We develop delay-compensating feedback laws for linear switched systems with
time-dependent switching. Because the future values of the switching signal,
which are needed for constructing an exact predictor-feedback law, may be
unavailable at current time, the key design challenge is how to construct a
proper predictor state. We resolve this challenge constructing two alternative,
average predictor-based feedback laws. The first is viewed as a
predictor-feedback law for a particular average system, properly modified to
provide exact state predictions over a horizon that depends on a minimum dwell
time of the switching signal (when it is available). The second is,
essentially, a modification of an average of predictor feedbacks, each one
corresponding to the fixed-mode predictor-feedback law. We establish that under
the control laws introduced, the closed-loop systems are (uniformly)
exponentially stable, provided that the differences among system's matrices and
among (nominal stabilizing) controller's gains are sufficiently small, with a
size that is inversely proportional to the delay length. Since no restriction
is imposed on the delay, such a limitation is inherent to the problem
considered (in which the future switching signal values are unavailable), and
thus, it cannot be removed. The stability proof relies on multiple Lyapunov
functionals constructed via backstepping and derivation of solutions' estimates
for quantifying the difference between average and exact predictor states. We
present consistent numerical simulation results, which illustrate the necessity
of employing the average predictor-based laws and demonstrate the performance
improvement when the knowledge of a minimum dwell time is properly utilized for
improving state prediction accuracy.

### 4. [An Improved Finite Element Modeling Method for Triply Periodic Minimal Surface Structures Based on Element Size and Minimum Jacobian](http://arxiv.org/pdf/2506.04028v1)

Authors: Siqi Wang, Chuangyu Jiang, Xiaodong Zhang, Yilong Zhang, Baoqiang Zhang, Huageng Luo

Triply periodic minimal surface (TPMS) structures, a type of lattice
structure, have garnered significant attention due to their lightweight nature,
controllability, and excellent mechanical properties. Voxel-based modeling is a
widely used method for investigating the mechanical behavior of such lattice
structures through finite element simulations. This study proposes a
two-parameter voxel method that incorporates joint control of element size and
minimum Jacobian (MJ). Numerical results indicate that the simulation outcomes
tend to stabilize when the MJ reaches 0.3. The grid convergence index (GCI),
based on Richardson extrapolation, is introduced to systematically assess the
numerical convergence behavior of both voxel models and the proposed
two-parameter voxel models. This provides a systematic and objective framework
for evaluating discretization errors and mesh convergence in TPMS modeling.
Compared with traditional voxel method, the proposed method exhibits superior
mesh convergence, solution accuracy, and computational efficiency. Furthermore,
the two-parameter voxel method also shows excellent applicability in the
analysis of graded TPMS structures, exhibiting even better convergence behavior
than in uniform structures.

### 5. [CHIME: Conditional Hallucination and Integrated Multi-scale Enhancement for Time Series Diffusion Model](http://arxiv.org/pdf/2506.03502v1)

Authors: Yuxuan Chen, Haipeng Xie

The denoising diffusion probabilistic model has become a mainstream
generative model, achieving significant success in various computer vision
tasks. Recently, there has been initial exploration of applying diffusion
models to time series tasks. However, existing studies still face challenges in
multi-scale feature alignment and generative capabilities across different
entities and long-time scales. In this paper, we propose CHIME, a conditional
hallucination and integrated multi-scale enhancement framework for time series
diffusion models. By employing multi-scale decomposition and adaptive
integration, CHIME captures the decomposed features of time series, achieving
in-domain distribution alignment between generated and original samples. In
addition, we introduce a feature hallucination module in the conditional
denoising process, enabling the transfer of temporal features through the
training of category-independent transformation layers. Experimental results on
publicly available real-world datasets demonstrate that CHIME achieves
state-of-the-art performance and exhibits excellent generative generalization
capabilities in few-shot scenarios.

### 6. [3D Holographic Flow Cytometry Measurements of Microalgae: Strategies for Angle Recovery in Complex Rotation Patterns](http://arxiv.org/pdf/2506.03738v1)

Authors: Francesca Borrelli, Giusy Giugliano, Emilie Houliez, Jaromir Behal, Daniele Pirone, Leonilde Roselli, Angela Sardo, Valerio Zupo, Maria Costantini, Lisa Miccio, Pasquale Memmolo, Vittorio Bianco, Pietro Ferraro

Marine ecosystems are in the spotlight, because environmental changes are
threatening biodiversity and ecological functions. In this context, microalgae
play key ecological roles both in planktonic and benthic ecosystems.
Consequently, they are considered indispensable targets for global monitoring
programs. However, due to a high spatial and temporal variability and to
difficulties of species identification (still relying on microscopy
observations), the assessment of roles played by these components of marine
ecosystems is demanding. In addition, technologies for a 3D assessment of their
complex morphology are scarcely available. Here, we present a comprehensive
workflow for retrieving 3D information on microalgae with diverse geometries
through holographic microscopy operating in flow-cytometry mode. Depending on
the rotation patterns of samples, a tailored approach is used to retrieve their
rolling angles. We demonstrate the feasibility of measuring 3D data of various
microalgae, contingent to the intrinsic optical properties of cells.
Specifically, we show that for quasi-transparent and low-scattering
microorganisms, the retrieved angles permit to achieve quantitative 3D
tomographic Refractive Index (RI) mapping, providing a full characterization of
the alga in terms of its inner structure and the outer shape. Moreover, even in
the most challenging scenarios, where microalgae exhibit high light absorption
or strong scattering, quantitative 3D shape reconstructions of diatoms and
dinoflagellates can be at least achieved. Finally, we compare our direct 3D
measurements with 2D inferences of 3D properties, obtained using a commercially
available microscopy system. The ability to non-invasively obtain 3D
information on microalgae marks a fundamental advancement in the field,
unlocking a wealth of novel biological insights for characterizing aquatic
ecosystems.

### 7. [Feedback stabilization of switched systems under arbitrary switching: A convex characterization](http://arxiv.org/pdf/2506.03759v1)

Authors: Thiago Alves Lima, Matteo Della Rossa, Antoine Girard

In this paper, we study stabilizability of discrete-time switched linear
systems where the switching signal is considered as an arbitrary disturbance
(and not a control variable). We characterize feedback stabilization via
necessary and sufficient linear matrix inequalities (LMIs) conditions based on
novel graph structures. We analyze both the cases in which the controller has
(or has not) access to the current switching mode, the so-called mode-dependent
and mode-independent settings, providing specular results. Moreover, our
approach provides explicit piecewise-linear and memory-dependent linear
controllers, highlighting the connections with existing stabilization
approaches. The effectiveness of the proposed technique is finally illustrated
with the help of some numerical examples.

### 8. [Discrete Element Parameter Calibration of Livestock Salt Based on Particle Scaling](http://arxiv.org/pdf/2506.03786v1)

Authors: Lulu Nie, Baoqin Wen, Jingbin Li, Shufeng Li, Yali Li, Zhaokun Zhang, Zhiyuan Wang, Zhihao Fan

In order to obtain accurate contact parameters for the discrete element
simulation of salt particles used in animal husbandry, the principle of
particle contact scaling and dimensional analysis were used for particle
scaling. Firstly, the Plackett Burman experiment was used to screen the
parameters that significantly affect the angle of repose: salt salt rolling
friction coefficient, salt salt recovery coefficient, and salt steel rolling
friction coefficient. Considering the influence of other parameters, a
combination of bench and simulation experiments was used to calibrate the
contact parameters between salt particles and steel plates used in animal
husbandry in EDEM. Finally, through the stacking test, steepest climbing test,
and orthogonal rotation combination test, the salt salt rolling friction
coefficient was obtained to be 0.23, the salt salt recovery coefficient was
0.544, and the salt steel rolling friction coefficient was 0.368, which were
verified through bench tests. The experimental results show that the relative
error between the actual value of the stacking angle and the simulation results
is 0.6%. The results indicate that the calibrated contact parameters can be
used for discrete element simulation of salt particles for animal husbandry,
providing reference for the design of quantitative feeding screws and silos.

### 9. [Quasioptic, Calibrated, Full 2-port Measurements of Cryogenic Devices under Vacuum in the 220-330 GHz Band](http://arxiv.org/pdf/2506.03824v1)

Authors: Maxim Masyukov, Aleksi Tamminen, Irina Nefedova, Andrey Generalov, Samu-Ville Pälli, Roman Grigorev, Pouyan Rezapoor, Rui Silva, Juha Mallat, Juha Ala-Laurinaho, Zachary Taylor

A quasi-optical (QO) test bench was designed, simulated, and calibrated for
characterizing S-parameters of devices in the 220-330 GHz (WR-3.4) frequency
range, from room temperature down to 4.8 K. The devices were measured through
vacuum windows via focused beam radiation. A de-embedding method employing
line-reflect-match (LRM) calibration was established to account for the effects
of optical components and vacuum windows. The setup provides all four
S-parameters with the reference plane located inside the cryostat, and achieves
a return loss of 30 dB with an empty holder. System validation was performed
with measurements of cryogenically cooled devices, such as bare silicon wafers
and stainless-steel frequency-selective surface (FSS) bandpass filters, and
superconducting bandpass FSS fabricated in niobium. A permittivity reduction of
Si based on 4-GHz resonance shift was observed concomitant with a drop in
temperature from 296 K to 4.8 K. The stainless steel FSS measurements revealed
a relatively temperature invariant center frequency and return loss level of
263 GHz and 35 dB on average, respectively. Finally, a center frequency of 257
GHz was measured with the superconducting filters, with return loss improved by
7 dB on average at 4.8 K. To the best of our knowledge, this is the first
reported attempt to scale LRM calibration to 330 GHz and use it to de-embed the
impact of optics and cryostat from cryogenically cooled device S-parameters.

### 10. [Object-centric 3D Motion Field for Robot Learning from Human Videos](http://arxiv.org/pdf/2506.04227v1)

Authors: Zhao-Heng Yin, Sherry Yang, Pieter Abbeel

Learning robot control policies from human videos is a promising direction
for scaling up robot learning. However, how to extract action knowledge (or
action representations) from videos for policy learning remains a key
challenge. Existing action representations such as video frames, pixelflow, and
pointcloud flow have inherent limitations such as modeling complexity or loss
of information. In this paper, we propose to use object-centric 3D motion field
to represent actions for robot learning from human videos, and present a novel
framework for extracting this representation from videos for zero-shot control.
We introduce two novel components in its implementation. First, a novel
training pipeline for training a ''denoising'' 3D motion field estimator to
extract fine object 3D motions from human videos with noisy depth robustly.
Second, a dense object-centric 3D motion field prediction architecture that
favors both cross-embodiment transfer and policy generalization to background.
We evaluate the system in real world setups. Experiments show that our method
reduces 3D motion estimation error by over 50% compared to the latest method,
achieve 55% average success rate in diverse tasks where prior approaches
fail~($\lesssim 10$\%), and can even acquire fine-grained manipulation skills
like insertion.

### Machine Learning (Statistics Category)

### 1. [Models of Heavy-Tailed Mechanistic Universality](http://arxiv.org/pdf/2506.03470v1)

Authors: Liam Hodgkinson, Zhichao Wang, Michael W. Mahoney

Recent theoretical and empirical successes in deep learning, including the
celebrated neural scaling laws, are punctuated by the observation that many
objects of interest tend to exhibit some form of heavy-tailed or power law
behavior. In particular, the prevalence of heavy-tailed spectral densities in
Jacobians, Hessians, and weight matrices has led to the introduction of the
concept of heavy-tailed mechanistic universality (HT-MU). Multiple lines of
empirical evidence suggest a robust correlation between heavy-tailed metrics
and model performance, indicating that HT-MU may be a fundamental aspect of
deep learning efficacy. Here, we propose a general family of random matrix
models -- the high-temperature Marchenko-Pastur (HTMP) ensemble -- to explore
attributes that give rise to heavy-tailed behavior in trained neural networks.
Under this model, spectral densities with power laws on (upper and lower) tails
arise through a combination of three independent factors (complex correlation
structures in the data; reduced temperatures during training; and reduced
eigenvector entropy), appearing as an implicit bias in the model structure, and
they can be controlled with an "eigenvalue repulsion" parameter. Implications
of our model on other appearances of heavy tails, including neural scaling
laws, optimizer trajectories, and the five-plus-one phases of neural network
training, are discussed.

### 2. [Path Generation and Evaluation in Video Games: A Nonparametric Statistical Approach](http://arxiv.org/pdf/2506.03522v1)

Authors: Daniel Campa, Mehdi Saeedi, Ian Colbert, Srinjoy Das

Navigation path traces play a crucial role in video game design, serving as a
vital resource for both enhancing player engagement and fine-tuning
non-playable character behavior. Generating such paths with human-like realism
can enrich the overall gaming experience, and evaluating path traces can
provide game designers insights into player interactions. Despite the
impressive recent advancements in deep learning-based generative modeling, the
video game industry hesitates to adopt such models for path generation, often
citing their complex training requirements and interpretability challenges. To
address these problems, we propose a novel path generation and evaluation
approach that is grounded in principled nonparametric statistics and provides
precise control while offering interpretable insights. Our path generation
method fuses two statistical techniques: (1) nonparametric model-free
transformations that capture statistical characteristics of path traces through
time; and (2) copula models that capture statistical dependencies in space. For
path evaluation, we adapt a nonparametric three-sample hypothesis test designed
to determine if the generated paths are overfit (mimicking the original data
too closely) or underfit (diverging too far from it). We demonstrate the
precision and reliability of our proposed methods with empirical analysis on
two existing gaming benchmarks to showcase controlled generation of diverse
navigation paths. Notably, our novel path generator can be fine-tuned with user
controllable parameters to create navigation paths that exhibit varying levels
of human-likeness in contrast to those produced by neural network-based agents.
The code is available at https://github.com/daniel-campa/mf-copula.

### 3. [SubSearch: Robust Estimation and Outlier Detection for Stochastic Block Models via Subgraph Search](http://arxiv.org/pdf/2506.03657v1)

Authors: Leonardo Martins Bianco, Christine Keribin, Zacharie Naulet

Community detection is a fundamental task in graph analysis, with methods
often relying on fitting models like the Stochastic Block Model (SBM) to
observed networks. While many algorithms can accurately estimate SBM parameters
when the input graph is a perfect sample from the model, real-world graphs
rarely conform to such idealized assumptions. Therefore, robust algorithms are
crucial-ones that can recover model parameters even when the data deviates from
the assumed distribution. In this work, we propose SubSearch, an algorithm for
robustly estimating SBM parameters by exploring the space of subgraphs in
search of one that closely aligns with the model's assumptions. Our approach
also functions as an outlier detection method, properly identifying nodes
responsible for the graph's deviation from the model and going beyond simple
techniques like pruning high-degree nodes. Extensive experiments on both
synthetic and real-world datasets demonstrate the effectiveness of our method.

### 4. [Position: There Is No Free Bayesian Uncertainty Quantification](http://arxiv.org/pdf/2506.03670v1)

Authors: Ivan Melev, Goeran Kauermann

Due to their intuitive appeal, Bayesian methods of modeling and uncertainty
quantification have become popular in modern machine and deep learning. When
providing a prior distribution over the parameter space, it is straightforward
to obtain a distribution over the parameters that is conventionally interpreted
as uncertainty quantification of the model. We challenge the validity of such
Bayesian uncertainty quantification by discussing the equivalent
optimization-based representation of Bayesian updating, provide an alternative
interpretation that is coherent with the optimization-based perspective,
propose measures of the quality of the Bayesian inferential stage, and suggest
directions for future work.

### 5. [Latent Guided Sampling for Combinatorial Optimization](http://arxiv.org/pdf/2506.03672v1)

Authors: Sobihan Surendran, Adeline Fermanian, Sylvain Le Corff

Combinatorial Optimization problems are widespread in domains such as
logistics, manufacturing, and drug discovery, yet their NP-hard nature makes
them computationally challenging. Recent Neural Combinatorial Optimization
methods leverage deep learning to learn solution strategies, trained via
Supervised or Reinforcement Learning (RL). While promising, these approaches
often rely on task-specific augmentations, perform poorly on
out-of-distribution instances, and lack robust inference mechanisms. Moreover,
existing latent space models either require labeled data or rely on pre-trained
policies. In this work, we propose LGS-Net, a novel latent space model that
conditions on problem instances, and introduce an efficient inference method,
Latent Guided Sampling (LGS), based on Markov Chain Monte Carlo and Stochastic
Approximation. We show that the iterations of our method form a
time-inhomogeneous Markov Chain and provide rigorous theoretical convergence
guarantees. Empirical results on benchmark routing tasks show that our method
achieves state-of-the-art performance among RL-based approaches.

### 6. [On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity](http://arxiv.org/pdf/2506.03719v1)

Authors: Quentin Bertrand, Anne Gagneux, Mathurin Massias, Rémi Emonet

Modern deep generative models can now produce high-quality synthetic samples
that are often indistinguishable from real training data. A growing body of
research aims to understand why recent methods -- such as diffusion and flow
matching techniques -- generalize so effectively. Among the proposed
explanations are the inductive biases of deep learning architectures and the
stochastic nature of the conditional flow matching loss. In this work, we rule
out the latter -- the noisy nature of the loss -- as a primary contributor to
generalization in flow matching. First, we empirically show that in
high-dimensional settings, the stochastic and closed-form versions of the flow
matching loss yield nearly equivalent losses. Then, using state-of-the-art flow
matching models on standard image datasets, we demonstrate that both variants
achieve comparable statistical performance, with the surprising observation
that using the closed-form can even improve performance.

### 7. [Infinitesimal Higher-Order Spectral Variations in Rectangular Real Random Matrices](http://arxiv.org/pdf/2506.03764v1)

Authors: Róisín Luo

We present a theoretical framework for deriving the general $n$-th order
Fr\'echet derivatives of singular values in real rectangular matrices, by
leveraging reduced resolvent operators from Kato's analytic perturbation theory
for self-adjoint operators. Deriving closed-form expressions for higher-order
derivatives of singular values is notoriously challenging through standard
matrix-analysis techniques. To overcome this, we treat a real rectangular
matrix as a compact operator on a finite-dimensional Hilbert space, and embed
the rectangular matrix into a block self-adjoint operator so that non-symmetric
perturbations are captured. Applying Kato's asymptotic eigenvalue expansion to
this construction, we obtain a general, closed-form expression for the
infinitesimal $n$-th order spectral variations. Specializing to $n=2$ and
deploying on a Kronecker-product representation with matrix convention yield
the Hessian of a singular value, not found in literature. By bridging abstract
operator-theoretic perturbation theory with matrices, our framework equips
researchers with a practical toolkit for higher-order spectral sensitivity
studies in random matrix applications (e.g., adversarial perturbation in deep
learning).

### 8. [Spatially Resolved Meteorological and Ancillary Data in Central Europe for Rainfall Streamflow Modeling](http://arxiv.org/pdf/2506.03819v1)

Authors: Marc Aurel Vischer, Noelia Otero, Jackie Ma

We present a dataset for rainfall streamflow modeling that is fully spatially
resolved with the aim of taking neural network-driven hydrological modeling
beyond lumped catchments. To this end, we compiled data covering five river
basins in central Europe: upper Danube, Elbe, Oder, Rhine, and Weser. The
dataset contains meteorological forcings, as well as ancillary information on
soil, rock, land cover, and orography. The data is harmonized to a regular 9km
times 9km grid and contains daily values that span from October 1981 to
September 2011. We also provide code to further combine our dataset with
publicly available river discharge data for end-to-end rainfall streamflow
modeling.

### 9. [Revisiting Unbiased Implicit Variational Inference](http://arxiv.org/pdf/2506.03839v1)

Authors: Tobias Pielok, Bernd Bischl, David Rügamer

Recent years have witnessed growing interest in semi-implicit variational
inference (SIVI) methods due to their ability to rapidly generate samples from
complex distributions. However, since the likelihood of these samples is
non-trivial to estimate in high dimensions, current research focuses on finding
effective SIVI training routines. Although unbiased implicit variational
inference (UIVI) has largely been dismissed as imprecise and computationally
prohibitive because of its inner MCMC loop, we revisit this method and show
that UIVI's MCMC loop can be effectively replaced via importance sampling and
the optimal proposal distribution can be learned stably by minimizing an
expected forward Kullback-Leibler divergence without bias. Our refined approach
demonstrates superior performance or parity with state-of-the-art methods on
established SIVI benchmarks.

### 10. [Algorithm- and Data-Dependent Generalization Bounds for Score-Based Generative Models](http://arxiv.org/pdf/2506.03849v1)

Authors: Benjamin Dupuis, Dario Shariatian, Maxime Haddouche, Alain Durmus, Umut Simsekli

Score-based generative models (SGMs) have emerged as one of the most popular
classes of generative models. A substantial body of work now exists on the
analysis of SGMs, focusing either on discretization aspects or on their
statistical performance. In the latter case, bounds have been derived, under
various metrics, between the true data distribution and the distribution
induced by the SGM, often demonstrating polynomial convergence rates with
respect to the number of training samples. However, these approaches adopt a
largely approximation theory viewpoint, which tends to be overly pessimistic
and relatively coarse. In particular, they fail to fully explain the empirical
success of SGMs or capture the role of the optimization algorithm used in
practice to train the score network. To support this observation, we first
present simple experiments illustrating the concrete impact of optimization
hyperparameters on the generalization ability of the generated distribution.
Then, this paper aims to bridge this theoretical gap by providing the first
algorithmic- and data-dependent generalization analysis for SGMs. In
particular, we establish bounds that explicitly account for the optimization
dynamics of the learning algorithm, offering new insights into the
generalization behavior of SGMs. Our theoretical findings are supported by
empirical results on several datasets.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

### 1. [Correction: The “LLM World of Words” English free association norms generated by large language models](https://www.nature.com/articles/s41597-025-05265-5)

Authors: Katherine Abramski et al.

### 2. [Enhanced analog circuit fault diagnosis via continuous wavelet transform and dual-stream convolutional fusion](https://www.nature.com/articles/s41598-025-02596-6)

Authors: Zhiwen Hou et al.

### 3. [Progressive plug and play full waveform inversion with multitask learning](https://www.nature.com/articles/s41598-025-04506-2)

Authors: Benwen Zhang et al.

### 4. [Fourier-modulated CLIP for zero-shot vehicle counting](https://www.nature.com/articles/s41598-025-04876-7)

Authors: Yunpeng Luo et al.

### 5. [Multi sensor based monitoring of paralyzed using Emperor Penguin Optimizer and Deep Maxout Network](https://www.nature.com/articles/s41598-025-04381-x)

Authors: Vijaya Gunturu et al.

### 6. [Efficient joint resource allocation using self organized map based Deep Reinforcement Learning for cybertwin enabled 6G networks](https://www.nature.com/articles/s41598-025-02274-7)

Authors: Nivetha A et al.

### 7. [An end-to-end attention-based approach for learning on graphs](https://www.nature.com/articles/s41467-025-60252-z)

Authors: David Buterez et al.

### 8. [Decentralized Proof-of-Location systems for trust, scalability, and privacy in digital societies](https://www.nature.com/articles/s41598-025-04566-4)

Authors: Eduardo Brito et al.

### 9. [Settlement early warning method for high speed railway subgrades based on TD Transformer](https://www.nature.com/articles/s41598-025-05067-0)

Authors: Wen Kebing et al.

### 10. [A lightweight scalable hybrid authentication framework for Internet of Medical Things (IoMT) using blockchain hyperledger consortium network with edge computing](https://www.nature.com/articles/s41598-025-05130-w)

Authors: Abdullah Ayub Khan et al.

### 11. [Guiding responsible AI in healthcare in the Philippines](https://www.nature.com/articles/s41746-025-01755-3)

Authors: Raymond Francis R. Sarmiento et al.

