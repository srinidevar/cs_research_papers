# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-23 17:00:25.360372 PST.

### Artificial Intelligence

### 1. [WebGraphEval: Multi-Turn Trajectory Evaluation for Web Agents using Graph Representation](http://arxiv.org/pdf/2510.19205v1)

Authors: Yaoyao Qian, Yuanli Wang, Jinda Zhang, Yun Zong, Meixu Chen, Hanhan Zhou, Jindan Huang, Yifan Zeng, Xinyu Hu, Chan Hee Song, Danqing Zhang

Current evaluation of web agents largely reduces to binary success metrics or
conformity to a single reference trajectory, ignoring the structural diversity
present in benchmark datasets. We present WebGraphEval, a framework that
abstracts trajectories from multiple agents into a unified, weighted action
graph. This representation is directly compatible with benchmarks such as
WebArena, leveraging leaderboard runs and newly collected trajectories without
modifying environments. The framework canonically encodes actions, merges
recurring behaviors, and applies structural analyses including reward
propagation and success-weighted edge statistics. Evaluations across thousands
of trajectories from six web agents show that the graph abstraction captures
cross-model regularities, highlights redundancy and inefficiency, and
identifies critical decision points overlooked by outcome-based metrics. By
framing web interaction as graph-structured data, WebGraphEval establishes a
general methodology for multi-path, cross-agent, and efficiency-aware
evaluation of web agents.

### 2. [ChatGPT Unveils Its Limits: Principles of Law Deliver Checkmate](http://arxiv.org/pdf/2510.19261v1)

Authors: Marianna Molinari, Ilaria Angela Amantea, Marinella Quaranta, Guido Governatori

This study examines the performance of ChatGPT with an experiment in the
legal domain. We compare the outcome with it a baseline using regular
expressions (Regex), rather than focusing solely on the assessment against
human performance. The study reveals that even if ChatGPT has access to the
necessary knowledge and competencies, it is unable to assemble them, reason
through, in a way that leads to an exhaustive result. This unveils a major
limitation of ChatGPT. Intelligence encompasses the ability to break down
complex issues and address them according to multiple required competencies,
providing a unified and comprehensive solution. In the legal domain, one of the
most crucial tasks is reading legal decisions and extracting key passages
condensed from principles of law (PoLs), which are then incorporated into
subsequent rulings by judges or defense documents by lawyers. In performing
this task, artificial intelligence lacks an all-encompassing understanding and
reasoning, which makes it inherently limited. Genuine intelligence, remains a
uniquely human trait, at least in this particular field.

### 3. [An Argumentative Explanation Framework for Generalized Reason Model with Inconsistent Precedents](http://arxiv.org/pdf/2510.19263v1)

Authors: Wachara Fungwacharakorn, Gauvain Bourgne, Ken Satoh

Precedential constraint is one foundation of case-based reasoning in AI and
Law. It generally assumes that the underlying set of precedents must be
consistent. To relax this assumption, a generalized notion of the reason model
has been introduced. While several argumentative explanation approaches exist
for reasoning with precedents based on the traditional consistent reason model,
there has been no corresponding argumentative explanation method developed for
this generalized reasoning framework accommodating inconsistent precedents. To
address this question, this paper examines an extension of the derivation state
argumentation framework (DSA-framework) to explain the reasoning according to
the generalized notion of the reason model.

### 4. [Continual Knowledge Adaptation for Reinforcement Learning](http://arxiv.org/pdf/2510.19314v1)

Authors: Jinwu Hu, Zihao Lian, Zhiquan Wen, Chenghao Li, Guohao Chen, Xutao Wen, Bin Xiao, Mingkui Tan

Reinforcement Learning enables agents to learn optimal behaviors through
interactions with environments. However, real-world environments are typically
non-stationary, requiring agents to continuously adapt to new tasks and
changing conditions. Although Continual Reinforcement Learning facilitates
learning across multiple tasks, existing methods often suffer from catastrophic
forgetting and inefficient knowledge utilization. To address these challenges,
we propose Continual Knowledge Adaptation for Reinforcement Learning (CKA-RL),
which enables the accumulation and effective utilization of historical
knowledge. Specifically, we introduce a Continual Knowledge Adaptation
strategy, which involves maintaining a task-specific knowledge vector pool and
dynamically using historical knowledge to adapt the agent to new tasks. This
process mitigates catastrophic forgetting and enables efficient knowledge
transfer across tasks by preserving and adapting critical model parameters.
Additionally, we propose an Adaptive Knowledge Merging mechanism that combines
similar knowledge vectors to address scalability challenges, reducing memory
requirements while ensuring the retention of essential knowledge. Experiments
on three benchmarks demonstrate that the proposed CKA-RL outperforms
state-of-the-art methods, achieving an improvement of 4.20% in overall
performance and 8.02% in forward transfer. The source code is available at
https://github.com/Fhujinwu/CKA-RL.

### 5. [MSC-Bench: A Rigorous Benchmark for Multi-Server Tool Orchestration](http://arxiv.org/pdf/2510.19423v1)

Authors: Jia-Kai Dong, I-Wei Huang, Chun-Tin Wu, Yi-Tien Tsai

We introduce MSC-Bench, a large-scale benchmark for evaluating multi-hop,
end-to-end tool orchestration by LLM agents in a hierarchical Model-Context
Protocol (MCP) ecosystem. Existing benchmarks often evaluate tools in
isolation, ignoring challenges such as functional overlap and cross-server
orchestration, leading to overly optimistic assessments. MSC-Bench addresses
these gaps by constructing ground truth through 'equal function sets', allowing
objective metrics such as F1 score and reducing the dependency on
LLM-as-a-judge evaluation. Organized as a five-level curriculum, it
systematically tests agent capabilities from single-tool orchestration to
complex cross-server planning, and robustness to out-of-scope requests.
Experiments reveal that rigid hierarchies can hinder performance without
co-designed strategies, and even state-of-the-art agents exhibit systemic
weaknesses in robustness. MSC-Bench provides a diagnostic framework to expose
these limitations and guide the development of more capable and efficient
tool-using agents. The benchmark and resources are publicly available at
https://github.com/snooow1029/MSC_Bench.

### 6. [NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](http://arxiv.org/pdf/2510.19429v1)

Authors: Wonje Choi, Jooyoung Kim, Honguk Woo

We address the challenge of adopting language models (LMs) for embodied tasks
in dynamic environments, where online access to large-scale inference engines
or symbolic planners is constrained due to latency, connectivity, and resource
limitations. To this end, we present NeSyPr, a novel embodied reasoning
framework that compiles knowledge via neurosymbolic proceduralization, thereby
equipping LM-based agents with structured, adaptive, and timely reasoning
capabilities. In NeSyPr, task-specific plans are first explicitly generated by
a symbolic tool leveraging its declarative knowledge. These plans are then
transformed into composable procedural representations that encode the plans'
implicit production rules, enabling the resulting composed procedures to be
seamlessly integrated into the LM's inference process. This neurosymbolic
proceduralization abstracts and generalizes multi-step symbolic structured
path-finding and reasoning into single-step LM inference, akin to human
knowledge compilation. It supports efficient test-time inference without
relying on external symbolic guidance, making it well suited for deployment in
latency-sensitive and resource-constrained physical systems. We evaluate NeSyPr
on the embodied benchmarks PDDLGym, VirtualHome, and ALFWorld, demonstrating
its efficient reasoning capabilities over large-scale reasoning models and a
symbolic planner, while using more compact LMs.

### 7. [DAIL: Beyond Task Ambiguity for Language-Conditioned Reinforcement Learning](http://arxiv.org/pdf/2510.19562v1)

Authors: Runpeng Xie, Quanwei Wang, Hao Hu, Zherui Zhou, Ni Mu, Xiyun Li, Yiqin Yang, Shuang Xu, Qianchuan Zhao, Bo XU

Comprehending natural language and following human instructions are critical
capabilities for intelligent agents. However, the flexibility of linguistic
instructions induces substantial ambiguity across language-conditioned tasks,
severely degrading algorithmic performance. To address these limitations, we
present a novel method named DAIL (Distributional Aligned Learning), featuring
two key components: distributional policy and semantic alignment. Specifically,
we provide theoretical results that the value distribution estimation mechanism
enhances task differentiability. Meanwhile, the semantic alignment module
captures the correspondence between trajectories and linguistic instructions.
Extensive experimental results on both structured and visual observation
benchmarks demonstrate that DAIL effectively resolves instruction ambiguities,
achieving superior performance to baseline methods. Our implementation is
available at https://github.com/RunpengXie/Distributional-Aligned-Learning.

### 8. [AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing](http://arxiv.org/pdf/2510.19661v1)

Authors: Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang

Web-based participatory urban sensing has emerged as a vital approach for
modern urban management by leveraging mobile individuals as distributed
sensors. However, existing urban sensing systems struggle with limited
generalization across diverse urban scenarios and poor interpretability in
decision-making. In this work, we introduce AgentSense, a hybrid, training-free
framework that integrates large language models (LLMs) into participatory urban
sensing through a multi-agent evolution system. AgentSense initially employs
classical planner to generate baseline solutions and then iteratively refines
them to adapt sensing task assignments to dynamic urban conditions and
heterogeneous worker preferences, while producing natural language explanations
that enhance transparency and trust. Extensive experiments across two
large-scale mobility datasets and seven types of dynamic disturbances
demonstrate that AgentSense offers distinct advantages in adaptivity and
explainability over traditional methods. Furthermore, compared to single-agent
LLM baselines, our approach outperforms in both performance and robustness,
while delivering more reasonable and transparent explanations. These results
position AgentSense as a significant advancement towards deploying adaptive and
explainable urban sensing systems on the web.

### 9. [A Graph Engine for Guitar Chord-Tone Soloing Education](http://arxiv.org/pdf/2510.19666v1)

Authors: Matthew Keating, Michael Casey

We present a graph-based engine for computing chord tone soloing suggestions
for guitar students. Chord tone soloing is a fundamental practice for
improvising over a chord progression, where the instrumentalist uses only the
notes contained in the current chord. This practice is a building block for all
advanced jazz guitar theory but is difficult to learn and practice. First, we
discuss methods for generating chord-tone arpeggios. Next, we construct a
weighted graph where each node represents a chord tone arpeggio for a chord in
the progression. Then, we calculate the edge weight between each consecutive
chord's nodes in terms of optimal transition tones. We then find the shortest
path through this graph and reconstruct a chord-tone soloing line. Finally, we
discuss a user-friendly system to handle input and output to this engine for
guitar students to practice chord tone soloing.

### 10. [Explainable e-sports win prediction through Machine Learning classification in streaming](http://arxiv.org/pdf/2510.19671v1)

Authors: Silvia García-Méndez, Francisco de Arriba-Pérez

The increasing number of spectators and players in e-sports, along with the
development of optimized communication solutions and cloud computing
technology, has motivated the constant growth of the online game industry. Even
though Artificial Intelligence-based solutions for e-sports analytics are
traditionally defined as extracting meaningful patterns from related data and
visualizing them to enhance decision-making, most of the effort in professional
winning prediction has been focused on the classification aspect from a batch
perspective, also leaving aside the visualization techniques. Consequently,
this work contributes to an explainable win prediction classification solution
in streaming in which input data is controlled over several sliding windows to
reflect relevant game changes. Experimental results attained an accuracy higher
than 90 %, surpassing the performance of competing solutions in the literature.
Ultimately, our system can be leveraged by ranking and recommender systems for
informed decision-making, thanks to the explainability module, which fosters
trust in the outcome predictions.

### Hardware Architecture

### 1. [gem5 Co-Pilot: AI Assistant Agent for Architectural Design Space Exploration](http://arxiv.org/pdf/2510.19577v1)

Authors: Zuoming Fu, Alex Manley, Mohammad Alian

Generative AI is increasing the productivity of software and hardware
development across many application domains. In this work, we utilize the power
of Large Language Models (LLMs) to develop a co-pilot agent for assisting gem5
users with automating design space exploration. Computer architecture design
space exploration is complex and time-consuming, given that numerous parameter
settings and simulation statistics must be analyzed before improving the
current design. The emergence of LLMs has significantly accelerated the
analysis of long-text data as well as smart decision making, two key functions
in a successful design space exploration task. In this project, we first build
gem5 Co-Pilot, an AI agent assistant for gem5, which comes with a webpage-GUI
for smooth user interaction, agent automation, and result summarization. We
also implemented a language for design space exploration, as well as a Design
Space Database (DSDB). With DSDB, gem5 Co-Pilot effectively implements a
Retrieval Augmented Generation system for gem5 design space exploration. We
experiment on cost-constraint optimization with four cost ranges and compare
our results with two baseline models. Results show that gem5 Co-Pilot can
quickly identify optimal parameters for specific design constraints based on
performance and cost, with limited user interaction.

### 2. [Res-DPU: Resource-shared Digital Processing-in-memory Unit for Edge-AI Workloads](http://arxiv.org/pdf/2510.19260v1)

Authors: Mukul Lokhande, Narendra Singh Dhakad, Seema Chouhan, Akash Sankhe, Santosh Kumar Vishvakarma

Processing-in-memory (PIM) has emerged as the go to solution for addressing
the von Neumann bottleneck in edge AI accelerators. However, state-of-the-art
(SoTA) digital PIM approaches suffer from low compute density, primarily due to
the use of bulky bit cells and transistor-heavy adder trees, which impose
limitations on macro scalability and energy efficiency. This work introduces
Res-DPU, a resource-shared digital PIM unit, with a dual-port 5T SRAM latch and
shared 2T AND compute logic. This reflects the per-bit multiplication cost to
just 5.25T and reduced the transistor count of the PIM array by up to 56% over
the SoTA works. Furthermore, a Transistor-Reduced 2D Interspersed Adder Tree
(TRAIT) with FA-7T and PG-FA-26T helps reduce the power consumption of the
adder tree by up to 21.35% and leads to improved energy efficiency by 59%
compared to conventional 28T RCA designs. We propose a Cycle-controlled
Iterative Approximate-Accurate Multiplication (CIA2M) approach, enabling
run-time accuracy-latency trade-offs without requiring error-correction
circuitry. The 16 KB REP-DPIM macro achieves 0.43 TOPS throughput and 87.22
TOPS/W energy efficiency in TSMC 65nm CMOS, with 96.85% QoR for ResNet-18 or
VGG-16 on CIFAR-10, including 30% pruning. The proposed results establish a
Res-DPU module for highly scalable and energy-efficient real-time edge AI
accelerators.

### 3. [QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](http://arxiv.org/pdf/2510.19296v1)

Authors: Yang Zhang, Rui Zhang, Jiaming Guo, Lei Huang, Di Huang, Yunpu Zhao, Shuyao Cheng, Pengwei Jin, Chongxiao Li, Zidong Du, Xing Hu, Qi Guo, Yunji Chen

The remarkable progress of Large Language Models (LLMs) presents promising
opportunities for Verilog code generation which is significantly important for
automated circuit design. The lacking of meaningful functional rewards hinders
the preference optimization based on Reinforcement Learning (RL) for producing
functionally correct Verilog code. In this paper, we propose Signal-Aware
Learning for Verilog code generation (QiMeng-SALV) by leveraging code segments
of functionally correct output signal to optimize RL training. Considering
Verilog code specifies the structural interconnection of hardware gates and
wires so that different output signals are independent, the key insight of
QiMeng-SALV is to extract verified signal-aware implementations in partially
incorrect modules, so as to enhance the extraction of meaningful functional
rewards. Roughly, we verify the functional correctness of signals in generated
module by comparing with that of reference module in the training data. Then
abstract syntax tree (AST) is employed to identify signal-aware code segments
which can provide meaningful functional rewards from erroneous modules.
Finally, we introduce signal-aware DPO which is optimized on the correct
signal-level code segments, thereby preventing noise and interference from
incorrect signals. The proposed QiMeng-SALV underscores the paradigm shift from
conventional module-level to fine-grained signal-level optimization in Verilog
code generation, addressing the issue of insufficient functional rewards.
Experiments demonstrate that our method achieves state-of-the-art performance
on VerilogEval and RTLLM, with a 7B parameter model matching the performance of
the DeepSeek v3 671B model and significantly outperforming the leading
open-source model CodeV trained on the same dataset. Our code is available at
https://github.com/zy1xxx/SALV.

### Computational Complexity

### 1. [Problems from Optimization and Computational Algebra Equivalent to Hilbert's Nullstellensatz](http://arxiv.org/pdf/2510.19704v1)

Authors: Markus Bläser, Sagnik Dutta, Gorav Jindal

Efficient algorithms for many problems in optimization and computational
algebra often arise from casting them as systems of polynomial equations. Blum,
Shub, and Smale formalized this as Hilbert's Nullstellensatz Problem $HN_R$:
given multivariate polynomials over a ring $R$, decide whether they have a
common solution in $R$. We can also view $HN_R$ as a complexity class by taking
the downward closure of the problem $HN_R$ under polynomial-time many-one
reductions. In this work, we show that many important problems from
optimization and algebra are complete or hard for this class.
  We first consider the Affine Polynomial Projection Problem: given polynomials
$f,g$, does an affine projection of the variables transform $f$ into $g$? We
show that this problem is at least as hard as $HN_F$ for any field $F$. Then we
consider the Sparse Shift Problem: given a polynomial, can its number of
monomials be reduced by an affine shift of the variables? Prior $HN_R$-hardness
for this problem was known for non-field integral domains $R$, which we extend
to fields.
  For the special case of the real field, HN captures the existential theory of
the reals and its complement captures the universal theory of the reals. We
prove that the problems of deciding real stability, convexity, and
hyperbolicity of a given polynomial are all complete for the universal theory
of the reals, thereby pinning down their exact complexity.

### 2. [On Minimal Achievable Quotas in Multiwinner Voting](http://arxiv.org/pdf/2510.19620v1)

Authors: Patrick Becker, Fabian Frank

Justified representation (JR) and extended justified representation (EJR) are
well-established proportionality axioms in approval-based multiwinner voting.
Both axioms are always satisfiable, but they rely on a fixed quota (typically
Hare or Droop), with the Droop quota being the smallest one that guarantees
existence across all instances. With this observation in mind, we take a first
step beyond the fixed-quota paradigm and introduce proportionality notions
where the quota is instance-dependent. We demonstrate that all commonly studied
voting rules can have an additive distance to the optimum of
$\frac{k^2}{(k+1)^2}$. Moreover, we look into the computational aspects of our
instance-dependent quota and prove that determining the optimal value of
$\alpha$ for a given approval profile satisfying $\alpha$-JR is NP-complete. To
address this, we introduce an integer linear programming (ILP) formulation for
computing committees that satisfy $\alpha$-JR, and we provide positive results
in the voter interval (VI) and candidate interval (CI) domains.

### 3. [A simplified version of the quantum OTOC$^{(2)}$ problem](http://arxiv.org/pdf/2510.19751v1)

Authors: Robbie King, Robin Kothari, Ryan Babbush, Sergio Boixo, Kostyantyn Kechedzhi, Thomas E. O'Brien, Vadim Smelyanskiy

This note presents a simplified version of the OTOC$^{(2)}$ problem that was
recently experimentally implemented by Google Quantum AI and collaborators. We
present a formulation of the problem for growing input size and hope this spurs
further theoretical work on the problem.

### 4. [Query-Efficient Zeroth-Order Algorithms for Nonconvex Optimization](http://arxiv.org/pdf/2510.19165v1)

Authors: Ruiyang Jin, Yuke Zhou, Yujie Tang, Jie Song, Siyang Gao

Zeroth-order optimization (ZO) has been a powerful framework for solving
black-box problems, which estimates gradients using zeroth-order data to update
variables iteratively. The practical applicability of ZO critically depends on
the efficiency of single-step gradient estimation and the overall query
complexity. However, existing ZO algorithms cannot achieve efficiency on both
simultaneously. In this work, we consider a general constrained optimization
model with black-box objective and constraint functions. To solve it, we
propose novel algorithms that can achieve the state-of-the-art overall query
complexity bound of $\mathcal{O}(d/\epsilon^4)$ to find an
$\epsilon$-stationary solution ($d$ is the dimension of variable space), while
reducing the queries for estimating a single-step gradient from
$\mathcal{O}(d)$ to $\mathcal{O}(1)$. Specifically, we integrate block updates
with gradient descent ascent and a block gradient estimator, which leads to two
algorithms, ZOB-GDA and ZOB-SGDA, respectively. Instead of constructing full
gradients, they estimate only partial gradients along random blocks of
dimensions, where the adjustable block sizes enable high single-step efficiency
without sacrificing convergence guarantees. Our theoretical results establish
the finite-sample convergence of the proposed algorithms for nonconvex
optimization. Finally, numerical experiments on a practical problem demonstrate
that our algorithms require over ten times fewer queries than existing methods.

### Computational Engineering

### 1. [Parameter Estimation in River Transport Models With Immobile Phase Exchange Using Dimensional Analysis and Reduced-Order Models](http://arxiv.org/pdf/2510.19664v1)

Authors: Manuel M. Reyna, Alexandre M. Tartakovsky

We propose a framework for parameter estimation in river transport models
using breakthrough curve data, which we refer to as Dimensionless Synthetic
Transport Estimation (DSTE). We utilize this framework to parameterize the
one-dimensional advection-dispersion equation model, incorporating immobile
phase exchange through a memory function. We solve the governing equation
analytically in the Laplace domain and numerically invert it to generate
synthetic breakthrough curves for different memory functions and boundary
conditions. A dimensionless formulation enables decoupling the estimation of
advection velocity from other parameters, significantly reducing the number of
required forward solutions. To improve computational efficiency, we apply a
Karhunen-Loeve (KL) expansion to transform the synthetic dataset into a
reduced-order space. Given a measured breakthrough curve, we estimate the
advection velocity by minimizing the distance from the measurement to the
synthetic data in KL space, and infer the remaining dimensionless parameters by
Projected Barycentric Interpolation (PBI). We benchmark our method against
several alternatives, including Laplace domain fitting, moment matching, global
random optimization, and variations of the DSTE framework using
nearest-neighbor interpolation and neural network-based estimation. Applied to
295 breakthrough curves from 54 tracer tests in 25 rivers, DSTE delivers
accurate parameter estimates. The resulting labeled dataset allows researchers
to link transport parameters with hydraulic conditions, site characteristics,
and measured concentrations. The synthetic dataset can be leveraged for the
analysis of new breakthrough curves, eliminating the need for additional
forward simulations.

### 2. [Wind Variability and Its Effect on Transmission Line Capacity Estimation](http://arxiv.org/pdf/2510.19433v1)

Authors: Nika Mlinarič Hribar, Matjaž Depolli, Gregor Kosec

This study investigates the impact of wind velocity averaging on Dynamic
Thermal Rating (DTR) calculations. It is based on a high-temporal-resolution (1
second) wind measurements obtained from a transmission line in Slovenia,
Europe. Wind speed and direction variability are analysed, and two averaging
methods, namely vector averaging, where velocity is averaged as vector, and
hybrid averaging, where speed is averaged as scalar, are employed. DTR
calculations are performed on both high-resolution data and averaged data (5
minute averaging window). It is demonstrated that averaging has a significant
effect on both Nusselt number and ampacity, and the effect exhibits a strong
angular dependency on the relative angle of the wind to the line. Therefore,
two limit cases are studied: in the case of parallel wind, averaged data
underestimates the ampacity, and there is a significant amount of cases where
the underestimation is larger than 10 %. In the case of perpendicular wind, the
two averaging methods affect the results in different ways, but both result in
a substantial amount of cases where ampacity is overestimated, potentially
leading to unsafe operation. The main takeaway of the study is that averaging
wind velocity has a significant impact on DTR results, and special emphasis
should be given to the averaging method, as different methods affect the
results in different ways.

### Computation and Language

### 1. [Tibetan Language and AI: A Comprehensive Survey of Resources, Methods and Challenges](http://arxiv.org/pdf/2510.19144v1)

Authors: Cheng Huang, Nyima Tashi, Fan Gao, Yutong Liu, Jiahao Li, Hao Tian, Siyang Jiang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Jin Zhang, Xiao Feng, Hao Wang, Jie Tang, Guojie Tang, Xiangxiang Wang, Jia Zhang, Tsengdar Lee, Yongbin Yu

Tibetan, one of the major low-resource languages in Asia, presents unique
linguistic and sociocultural characteristics that pose both challenges and
opportunities for AI research. Despite increasing interest in developing AI
systems for underrepresented languages, Tibetan has received limited attention
due to a lack of accessible data resources, standardized benchmarks, and
dedicated tools. This paper provides a comprehensive survey of the current
state of Tibetan AI in the AI domain, covering textual and speech data
resources, NLP tasks, machine translation, speech recognition, and recent
developments in LLMs. We systematically categorize existing datasets and tools,
evaluate methods used across different tasks, and compare performance where
possible. We also identify persistent bottlenecks such as data sparsity,
orthographic variation, and the lack of unified evaluation metrics.
Additionally, we discuss the potential of cross-lingual transfer, multi-modal
learning, and community-driven resource creation. This survey aims to serve as
a foundational reference for future work on Tibetan AI research and encourages
collaborative efforts to build an inclusive and sustainable AI ecosystem for
low-resource languages.

### 2. ["You Are Rejected!": An Empirical Study of Large Language Models Taking Hiring Evaluations](http://arxiv.org/pdf/2510.19167v1)

Authors: Dingjie Fu, Dianxing Shi

With the proliferation of the internet and the rapid advancement of
Artificial Intelligence, leading technology companies face an urgent annual
demand for a considerable number of software and algorithm engineers. To
efficiently and effectively identify high-potential candidates from thousands
of applicants, these firms have established a multi-stage selection process,
which crucially includes a standardized hiring evaluation designed to assess
job-specific competencies. Motivated by the demonstrated prowess of Large
Language Models (LLMs) in coding and reasoning tasks, this paper investigates a
critical question: Can LLMs successfully pass these hiring evaluations? To this
end, we conduct a comprehensive examination of a widely used professional
assessment questionnaire. We employ state-of-the-art LLMs to generate responses
and subsequently evaluate their performance. Contrary to any prior expectation
of LLMs being ideal engineers, our analysis reveals a significant inconsistency
between the model-generated answers and the company-referenced solutions. Our
empirical findings lead to a striking conclusion: All evaluated LLMs fails to
pass the hiring evaluation.

### 3. [Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](http://arxiv.org/pdf/2510.19171v1)

Authors: Jihwan Bang, Juntae Lee, Seunghan Yang, Sungha Choi

Multi-hop retrieval-augmented generation (RAG) is a promising strategy for
complex reasoning, yet existing iterative prompting approaches remain
inefficient. They often regenerate predictable token sequences at every step
and rely on stochastic stopping, leading to excessive token usage and unstable
termination. We propose TSSS (Think Straight, Stop Smart), a structured
multi-hop RAG framework designed for efficiency. TSSS introduces (i) a
template-based reasoning that caches recurring prefixes and anchors sub-queries
to the main question, reducing token generation cost while promoting stable
reasoning, and (ii) a retriever-based terminator, which deterministically halts
reasoning once additional sub-queries collapse into repetition. This separation
of structured reasoning and termination control enables both faster inference
and more reliable answers. On HotpotQA, 2WikiMultiHop, and MuSiQue, TSSS
achieves state-of-the-art accuracy and competitive efficiency among RAG-CoT
approaches, highlighting its effectiveness in efficiency-constrained scenarios
such as on-device inference.

### 4. [Multi-Faceted Evaluation of Tool-Augmented Dialogue Systems](http://arxiv.org/pdf/2510.19186v1)

Authors: Zhaoyi Joey Hou, Tanya Shourya, Yingfan Wang, Shamik Roy, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah

Evaluating conversational AI systems that use external tools is challenging,
as errors can arise from complex interactions among user, agent, and tools.
While existing evaluation methods assess either user satisfaction or agents'
tool-calling capabilities, they fail to capture critical errors in multi-turn
tool-augmented dialogues-such as when agents misinterpret tool results yet
appear satisfactory to users. We introduce TRACE, a benchmark of systematically
synthesized tool-augmented conversations covering diverse error cases, and
SCOPE, an evaluation framework that automatically discovers diverse error
patterns and evaluation rubrics in tool-augmented dialogues. Experiments show
SCOPE significantly outperforms the baseline, particularly on challenging cases
where user satisfaction signals are misleading.

### 5. [DiSRouter: Distributed Self-Routing for LLM Selections](http://arxiv.org/pdf/2510.19208v1)

Authors: Hang Zheng, Hongshen Xu, Yongkai Lin, Shuai Fan, Lu Chen, Kai Yu

The proliferation of Large Language Models (LLMs) has created a diverse
ecosystem of models with highly varying performance and costs, necessitating
effective query routing to balance performance and expense. Current routing
systems often rely on a centralized external router trained on a fixed set of
LLMs, making them inflexible and prone to poor performance since the small
router can not fully understand the knowledge boundaries of different LLMs. We
introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts
from centralized control to distributed routing. In DiSRouter, a query
traverses a network of LLM agents, each independently deciding whether to
answer or route to other agents based on its own self-awareness, its ability to
judge its competence. This distributed design offers superior flexibility,
scalability, and generalizability. To enable this, we propose a two-stage
Self-Awareness Training pipeline that enhances each LLM's self-awareness.
Extensive experiments demonstrate that DiSRouter significantly outperforms
existing routing methods in utility across various scenarios, effectively
distinguishes between easy and hard queries, and shows strong generalization to
out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic
self-awareness is more effective than external assessment, paving the way for
more modular and efficient multi-agent systems.

### 6. [Modality Matching Matters: Calibrating Language Distances for Cross-Lingual Transfer in URIEL+](http://arxiv.org/pdf/2510.19217v1)

Authors: York Hay Ng, Aditya Khan, Xiang Lu, Matteo Salloum, Michael Zhou, Phuong H. Hoang, A. Seza Doğruöz, En-Shiun Annie Lee

Existing linguistic knowledge bases such as URIEL+ provide valuable
geographic, genetic and typological distances for cross-lingual transfer but
suffer from two key limitations. One, their one-size-fits-all vector
representations are ill-suited to the diverse structures of linguistic data,
and two, they lack a principled method for aggregating these signals into a
single, comprehensive score. In this paper, we address these gaps by
introducing a framework for type-matched language distances. We propose novel,
structure-aware representations for each distance type: speaker-weighted
distributions for geography, hyperbolic embeddings for genealogy, and a latent
variables model for typology. We unify these signals into a robust,
task-agnostic composite distance. In selecting transfer languages, our
representations and composite distances consistently improve performance across
a wide range of NLP tasks, providing a more principled and effective toolkit
for multilingual research.

### 7. [SheetBrain: A Neuro-Symbolic Agent for Accurate Reasoning over Complex and Large Spreadsheets](http://arxiv.org/pdf/2510.19247v1)

Authors: Ziwei Wang, Jiayuan Su, Mengyu Zhou, Huaxing Zeng, Mengni Jia, Xiao Lv, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang

Understanding and reasoning over complex spreadsheets remain fundamental
challenges for large language models (LLMs), which often struggle with
accurately capturing the complex structure of tables and ensuring reasoning
correctness. In this work, we propose SheetBrain, a neuro-symbolic dual
workflow agent framework designed for accurate reasoning over tabular data,
supporting both spreadsheet question answering and manipulation tasks.
SheetBrain comprises three core modules: an understanding module, which
produces a comprehensive overview of the spreadsheet - including sheet summary
and query-based problem insight to guide reasoning; an execution module, which
integrates a Python sandbox with preloaded table-processing libraries and an
Excel helper toolkit for effective multi-turn reasoning; and a validation
module, which verifies the correctness of reasoning and answers, triggering
re-execution when necessary. We evaluate SheetBrain on multiple public tabular
QA and manipulation benchmarks, and introduce SheetBench, a new benchmark
targeting large, multi-table, and structurally complex spreadsheets.
Experimental results show that SheetBrain significantly improves accuracy on
both existing benchmarks and the more challenging scenarios presented in
SheetBench. Our code is publicly available at
https://github.com/microsoft/SheetBrain.

### 8. [Difficulty-Controllable Multiple-Choice Question Generation Using Large Language Models and Direct Preference Optimization](http://arxiv.org/pdf/2510.19265v1)

Authors: Yuto Tomikawa, Masaki Uto

Difficulty-controllable question generation for reading comprehension has
gained significant attention in the field of education as a fundamental tool
for adaptive learning support. Although several neural question generation
methods have recently succeeded in controlling difficulty, conventional
approaches still face two major limitations. First, they cannot directly
generate multiple-choice questions, which are the most widely used question
type in educational contexts. Second, they are not explicitly trained to
optimize the accuracy of difficulty control, leaving room for further
improvement in difficulty controllability. To address these limitations, this
study proposes a novel difficulty-controllable multiple-choice question
generation method for reading comprehension which leverages a large language
model trained using a direct preference optimization technique to improve the
accuracy of difficulty control.

### 9. [TheMCPCompany: Creating General-purpose Agents with Task-specific Tools](http://arxiv.org/pdf/2510.19286v1)

Authors: Reza Esfandiarpoor, Vishwas Suryanarayanan, Stephen H. Bach, Vishal Chowdhary, Anthony Aue

Since the introduction of the Model Context Protocol (MCP), the number of
available tools for Large Language Models (LLMs) has increased significantly.
These task-specific tool sets offer an alternative to general-purpose tools
such as web browsers, while being easier to develop and maintain than GUIs.
However, current general-purpose agents predominantly rely on web browsers for
interacting with the environment. Here, we introduce TheMCPCompany, a benchmark
for evaluating tool-calling agents on tasks that involve interacting with
various real-world services. We use the REST APIs of these services to create
MCP servers, which include over 18,000 tools. We also provide manually
annotated ground-truth tools for each task. In our experiments, we use the
ground truth tools to show the potential of tool-calling agents for both
improving performance and reducing costs assuming perfect tool retrieval. Next,
we explore agent performance using tool retrieval to study the real-world
practicality of tool-based agents. While all models with tool retrieval perform
similarly or better than browser-based agents, smaller models cannot take full
advantage of the available tools through retrieval. On the other hand, GPT-5's
performance with tool retrieval is very close to its performance with
ground-truth tools. Overall, our work shows that the most advanced reasoning
models are effective at discovering tools in simpler environments, but
seriously struggle with navigating complex enterprise environments.
TheMCPCompany reveals that navigating tens of thousands of tools and combining
them in non-trivial ways to solve complex problems is still a challenging task
for current models and requires both better reasoning and better retrieval
models.

### 10. [JointCQ: Improving Factual Hallucination Detection with Joint Claim and Query Generation](http://arxiv.org/pdf/2510.19310v1)

Authors: Fan Xu, Huixuan Zhang, Zhenliang Zhang, Jiahao Wang, Xiaojun Wan

Current large language models (LLMs) often suffer from hallucination issues,
i,e, generating content that appears factual but is actually unreliable. A
typical hallucination detection pipeline involves response decomposition (i.e.,
claim extraction), query generation, evidence collection (i.e., search or
retrieval), and claim verification. However, existing methods exhibit
limitations in the first two stages, such as context loss during claim
extraction and low specificity in query generation, resulting in degraded
performance across the hallucination detection pipeline. In this work, we
introduce JointCQ https://github.com/pku0xff/JointCQ, a joint claim-and-query
generation framework designed to construct an effective and efficient
claim-query generator. Our framework leverages elaborately designed evaluation
criteria to filter synthesized training data, and finetunes a language model
for joint claim extraction and query generation, providing reliable and
informative inputs for downstream search and verification. Experimental results
demonstrate that our method outperforms previous methods on multiple
open-domain QA hallucination detection benchmarks, advancing the goal of more
trustworthy and transparent language model systems.

### Cryptography and Security

### 1. [Defending Against Prompt Injection with DataFilter](http://arxiv.org/pdf/2510.19207v1)

Authors: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

When large language model (LLM) agents are increasingly deployed to automate
tasks and interact with untrusted external data, prompt injection emerges as a
significant security threat. By injecting malicious instructions into the data
that LLMs access, an attacker can arbitrarily override the original user task
and redirect the agent toward unintended, potentially harmful actions. Existing
defenses either require access to model weights (fine-tuning), incur
substantial utility loss (detection-based), or demand non-trivial system
redesign (system-level). Motivated by this, we propose DataFilter, a test-time
model-agnostic defense that removes malicious instructions from the data before
it reaches the backend LLM. DataFilter is trained with supervised fine-tuning
on simulated injections and leverages both the user's instruction and the data
to selectively strip adversarial content while preserving benign information.
Across multiple benchmarks, DataFilter consistently reduces the prompt
injection attack success rates to near zero while maintaining the LLMs'
utility. DataFilter delivers strong security, high utility, and plug-and-play
deployment, making it a strong practical defense to secure black-box commercial
LLMs against prompt injection. Our DataFilter model is released at
https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to
reproduce our results at https://github.com/yizhu-joy/DataFilter.

### 2. [Reliability and Resilience of AI-Driven Critical Network Infrastructure under Cyber-Physical Threats](http://arxiv.org/pdf/2510.19295v1)

Authors: Konstantinos A. Lizos, Leandros Maglaras, Elena Petrovik, Saied M. Abd El-atty, Georgios Tsachtsiris, Mohamed Amine Ferrag

The increasing reliance on AI-driven 5G/6G network infrastructures for
mission-critical services highlights the need for reliability and resilience
against sophisticated cyber-physical threats. These networks are highly exposed
to novel attack surfaces due to their distributed intelligence, virtualized
resources, and cross-domain integration. This paper proposes a fault-tolerant
and resilience-aware framework that integrates AI-driven anomaly detection,
adaptive routing, and redundancy mechanisms to mitigate cascading failures
under cyber-physical attack conditions. A comprehensive validation is carried
out using NS-3 simulations, where key performance indicators such as
reliability, latency, resilience index, and packet loss rate are analyzed under
various attack scenarios. The deduced results demonstrate that the proposed
framework significantly improves fault recovery, stabilizes packet delivery,
and reduces service disruption compared to baseline approaches.

### 3. [An Adaptive Intelligent Thermal-Aware Routing Protocol for Wireless Body Area Networks](http://arxiv.org/pdf/2510.19300v1)

Authors: Abdollah Rahimi, Mehdi Jafari Shahbazzadeh, Amid Khatibi

Wireless Body Area Networks (WBANs) have gained significant attention due to
their applications in healthcare monitoring, sports, military communication,
and remote patient care. These networks consist of wearable or implanted
sensors that continuously collect and transmit physiological data, requiring
efficient and reliable communication. However, WBANs face challenges such as
limited energy, dynamic topology, and sensitivity to node temperature, which
demand specialized routing strategies. Traditional shortest-path routing often
causes congestion and overheating in specific nodes, leading to early failures.
To address these problems, this paper proposes an intelligent temperature-aware
and reliability-based routing approach that enhances WBAN performance. The
proposed method works in two phases: (1) network setup and intelligent path
selection, and (2) dynamic traffic management and hotspot avoidance. In the
first phase, nodes share information such as residual energy, temperature, link
reliability, and delay to build an optimized topology using a multi-criteria
decision algorithm. The second phase continuously monitors real-time conditions
and reroutes traffic away from overheated or depleted nodes. Simulation results
show that the proposed approach improves throughput by 13 percent, reduces
end-to-end delay by 10 percent, decreases energy consumption by 25 percent, and
lowers routing load by 30 percent compared to existing methods.

### 4. [Authorization of Knowledge-base Agents in an Intent-based Management Function](http://arxiv.org/pdf/2510.19324v1)

Authors: Loay Abdelrazek, Leyli Karaçay, Marin Orlic

As networks move toward the next-generation 6G, Intent-based Management (IbM)
systems are increasingly adopted to simplify and automate network management by
translating high-level intents into low-level configurations. Within these
systems, agents play a critical role in monitoring current state of the
network, gathering data, and enforcing actions across the network to fulfill
the intent. However, ensuring secure and fine-grained authorization of agents
remains a significant challenge, especially in dynamic and multi-tenant
environments. Traditional models such as Role-Based Access Control (RBAC),
Attribute-Based Access Control (ABAC) and Relational-Based Access Control
(RelBAC) often lack the flexibility to accommodate the evolving context and
granularity required by intentbased operations. In this paper, we propose an
enhanced authorization framework that integrates contextual and functional
attributes with agent roles to achieve dynamic, policy-driven access control.
By analyzing agent functionalities, our approach ensures that agents are
granted only the minimal necessary privileges towards knowledge graphs.

### 5. [Transmitter Identification via Volterra Series Based Radio Frequency Fingerprint](http://arxiv.org/pdf/2510.19440v1)

Authors: Rundong Jiang, Jun Hu, Zhiyuan Xie, Yunqi Song, Shiyou Xu

The growing number of wireless devices increases the need for secure network
access. Radio Frequency Fingerprinting (RFF), a physical-layer authentication
method, offers a promising solution as it requires no cryptography and resists
spoofing. However, existing RFF approaches often lack a unified theory and
effective feature extraction. Many methods use handcrafted signal features or
direct neural network classification, leading to limited generalization and
interpretability. In this work, we model the transmitter as a black box and
analyze its impact on transmitted signals. By treating the deviation from an
ideal signal as hardware-induced distortion, we represent the received signal
using a Volterra series, using its kernels to capture linear and nonlinear
hardware traits. To manage the high dimensionality of these kernels, we
approximate them via wavelet decomposition and estimate coefficients through
least-squares fitting. The resulting wavelet coefficients provide compact yet
informative hardware representations, which are classified using a
complex-valued neural network. Experiments on a public LoRa dataset show
state-of-the-art performance, with over 98% accuracy in static channels and
above 90% under multipath and Doppler effects. The proposed approach improves
both interpretability and generalization across varying channel conditions.

### 6. [AegisMCP: Online Graph Intrusion Detection for Tool-Augmented LLMs on Edge Devices](http://arxiv.org/pdf/2510.19462v1)

Authors: Zhonghao Zhan, Amir Al Sadi, Krinos Li, Hamed Haddadi

In this work, we study security of Model Context Protocol (MCP) agent
toolchains and their applications in smart homes. We introduce AegisMCP, a
protocol-level intrusion detector. Our contributions are: (i) a minimal attack
suite spanning instruction-driven escalation, chain-of-tool exfiltration,
malicious MCP server registration, and persistence; (ii) NEBULA-Schema
(Network-Edge Behavioral Learning for Untrusted LLM Agents), a reusable
protocol-level instrumentation that represents MCP activity as a streaming
heterogeneous temporal graph over agents, MCP servers, tools, devices, remotes,
and sessions; and (iii) a CPU-only streaming detector that fuses novelty,
session-DAG structure, and attribute cues for near-real-time edge inference,
with optional fusion of local prompt-guardrail signals. On an emulated
smart-home testbed spanning multiple MCP stacks and a physical bench, AegisMCP
achieves sub-second per-window model inference and end-to-end alerting. The
latency of AegisMCP is consistently sub-second on Intel N150-class edge
hardware, while outperforming traffic-only and sequence baselines; ablations
confirm the importance of DAG and install/permission signals. We release code,
schemas, and generators for reproducible evaluation.

### 7. [Cross-Chain Sealed-Bid Auctions Using Confidential Compute Blockchains](http://arxiv.org/pdf/2510.19491v1)

Authors: Jonas Gebele, Timm Mutzel, Burak Oez, Florian Matthes

Sealed-bid auctions ensure fair competition and efficient allocation but are
often deployed on centralized infrastructure, enabling opaque manipulation.
Public blockchains eliminate central control, yet their inherent transparency
conflicts with the confidentiality required for sealed bidding. Prior attempts
struggle to reconcile privacy, verifiability, and scalability without relying
on trusted intermediaries, multi-round protocols, or expensive cryptography. We
present a sealed-bid auction protocol that executes sensitive bidding logic on
a Trusted Execution Environment (TEE)-backed confidential compute blockchain
while retaining settlement and enforcement on a public chain. Bidders commit
funds to enclave-generated escrow addresses, ensuring confidentiality and
binding commitments. After the deadline, any party can trigger resolution: the
confidential blockchain determines the winner through verifiable off-chain
computation and issues signed settlement transactions for execution on the
public chain. Our design provides security, privacy, and scalability without
trusted third parties or protocol modifications. We implement it on SUAVE with
Ethereum settlement, evaluate its scalability and trust assumptions, and
demonstrate deployment with minimal integration on existing infrastructure

### 8. [Privacy-Preserving Spiking Neural Networks: A Deep Dive into Encryption Parameter Optimisation](http://arxiv.org/pdf/2510.19537v1)

Authors: Mahitha Pulivathi, Ana Fontes Rodrigues, Isibor Kennedy Ihianle, Andreas Oikonomou, Srinivas Boppu, Pedro Machado

Deep learning is widely applied to modern problems through neural networks,
but the growing computational and energy demands of these models have driven
interest in more efficient approaches. Spiking Neural Networks (SNNs), the
third generation of neural networks, mimic the brain's event-driven behaviour,
offering improved performance and reduced power use. At the same time, concerns
about data privacy during cloud-based model execution have led to the adoption
of cryptographic methods. This article introduces BioEncryptSNN, a spiking
neural network based encryption-decryption framework for secure and
noise-resilient data protection. Unlike conventional algorithms, BioEncryptSNN
converts ciphertext into spike trains and exploits temporal neural dynamics to
model encryption and decryption, optimising parameters such as key length,
spike timing, and synaptic connectivity. Benchmarked against AES-128, RSA-2048,
and DES, BioEncryptSNN preserved data integrity while achieving up to 4.1x
faster encryption and decryption than PyCryptodome's AES implementation. The
framework demonstrates scalability and adaptability across symmetric and
asymmetric ciphers, positioning SNNs as a promising direction for secure,
energy-efficient computing.

### 9. [CircuitGuard: Mitigating LLM Memorization in RTL Code Generation Against IP Leakage](http://arxiv.org/pdf/2510.19676v1)

Authors: Nowfel Mashnoor, Mohammad Akyash, Hadi Kamali, Kimia Azar

Large Language Models (LLMs) have achieved remarkable success in generative
tasks, including register-transfer level (RTL) hardware synthesis. However,
their tendency to memorize training data poses critical risks when proprietary
or security-sensitive designs are unintentionally exposed during inference.
While prior work has examined memorization in natural language, RTL introduces
unique challenges: In RTL, structurally different implementations (e.g.,
behavioral vs. gate-level descriptions) can realize the same hardware, leading
to intellectual property (IP) leakage (full or partial) even without verbatim
overlap. Conversely, even small syntactic variations (e.g., operator precedence
or blocking vs. non-blocking assignments) can drastically alter circuit
behavior, making correctness preservation especially challenging. In this work,
we systematically study memorization in RTL code generation and propose
CircuitGuard, a defense strategy that balances leakage reduction with
correctness preservation. CircuitGuard (1) introduces a novel RTL-aware
similarity metric that captures both structural and functional equivalence
beyond surface-level overlap, and (2) develops an activation-level steering
method that identifies and attenuates transformer components most responsible
for memorization. Our empirical evaluation demonstrates that CircuitGuard
identifies (and isolates) 275 memorization-critical features across layers
18-28 of Llama 3.1-8B model, achieving up to 80% reduction in semantic
similarity to proprietary patterns while maintaining generation quality.
CircuitGuard further shows 78-85% cross-domain transfer effectiveness, enabling
robust memorization mitigation across circuit categories without retraining.

### 10. [Under Pressure: Security Analysis and Process Impacts of a Commercial Smart Air Compressor](http://arxiv.org/pdf/2510.19772v1)

Authors: Jad Zarzour, Matthew Jablonski

The integration of Industrial Internet of Things (IIoT) devices into
manufacturing environments has accelerated the transition to Industry 4.0, but
has also introduced new cybersecurity risks. This paper conducts a
comprehensive security analysis of a commercial smart air compressor, revealing
critical vulnerabilities including hardcoded credentials, unauthenticated APIs,
and an insecure update mechanism. It includes a formal threat model,
demonstrates practical attack scenarios in a testbed environment, and evaluates
their subsequent impact on an industrial process, leading to denial of service
and the corruption of critical process telemetry. In addition, an analysis of
the device's supply chain reveals how product integration from multiple vendors
and limited security considerations can expose a device to threats. The
findings underscore the necessity of incorporating cybersecurity principles
into both IIoT device design and supply chain governance to enhance resilience
against emerging industrial cyber threats.

### Computer Vision and Pattern Recognition

### 1. [FootFormer: Estimating Stability from Visual Input](http://arxiv.org/pdf/2510.19170v1)

Authors: Keaton Kraiger, Jingjing Li, Skanda Bharadwaj, Jesse Scott, Robert T. Collins, Yanxi Liu

We propose FootFormer, a cross-modality approach for jointly predicting human
motion dynamics directly from visual input. On multiple datasets, FootFormer
achieves statistically significantly better or equivalent estimates of foot
pressure distributions, foot contact maps, and center of mass (CoM), as
compared with existing methods that generate one or two of those measures.
Furthermore, FootFormer achieves SOTA performance in estimating
stability-predictive components (CoP, CoM, BoS) used in classic kinesiology
metrics. Code and data are available at
https://github.com/keatonkraiger/Vision-to-Stability.git.

### 2. [Malaria Detection from Blood Cell Images Using XceptionNet](http://arxiv.org/pdf/2510.19182v1)

Authors: Warisa Nusrat, Mostafijur Rahman, Ayatullah Faruk Mollah

Malaria, which primarily spreads with the bite of female anopheles mosquitos,
often leads to death of people - specifically children in the age-group of 0-5
years. Clinical experts identify malaria by observing RBCs in blood smeared
images with a microscope. Lack of adequate professional knowledge and skills,
and most importantly manual involvement may cause incorrect diagnosis.
Therefore, computer aided automatic diagnosis stands as a preferred substitute.
In this paper, well-demonstrated deep networks have been applied to extract
deep intrinsic features from blood cell images and thereafter classify them as
malaria infected or healthy cells. Among the six deep convolutional networks
employed in this work viz. AlexNet, XceptionNet, VGG-19, Residual Attention
Network, DenseNet-121 and Custom-CNN. Residual Attention Network and
XceptionNet perform relatively better than the rest on a publicly available
malaria cell image dataset. They yield an average accuracy of 97.28% and 97.55%
respectively, that surpasses other related methods on the same dataset. These
findings highly encourage the reality of deep learning driven method for
automatic and reliable detection of malaria while minimizing direct manual
involvement.

### 3. [Video Consistency Distance: Enhancing Temporal Consistency for Image-to-Video Generation via Reward-Based Fine-Tuning](http://arxiv.org/pdf/2510.19193v1)

Authors: Takehiro Aoshima, Yusuke Shinohara, Park Byeongseon

Reward-based fine-tuning of video diffusion models is an effective approach
to improve the quality of generated videos, as it can fine-tune models without
requiring real-world video datasets. However, it can sometimes be limited to
specific performances because conventional reward functions are mainly aimed at
enhancing the quality across the whole generated video sequence, such as
aesthetic appeal and overall consistency. Notably, the temporal consistency of
the generated video often suffers when applying previous approaches to
image-to-video (I2V) generation tasks. To address this limitation, we propose
Video Consistency Distance (VCD), a novel metric designed to enhance temporal
consistency, and fine-tune a model with the reward-based fine-tuning framework.
To achieve coherent temporal consistency relative to a conditioning image, VCD
is defined in the frequency space of video frame features to capture frame
information effectively through frequency-domain analysis. Experimental results
across multiple I2V datasets demonstrate that fine-tuning a video generation
model with VCD significantly enhances temporal consistency without degrading
other performance compared to the previous method.

### 4. [MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting](http://arxiv.org/pdf/2510.19210v1)

Authors: In-Hwan Jin, Hyeongju Mun, Joonsoo Kim, Kugjin Yun, Kyeongbo Kong

Recent advances in dynamic scene reconstruction have significantly benefited
from 3D Gaussian Splatting, yet existing methods show inconsistent performance
across diverse scenes, indicating no single approach effectively handles all
dynamic challenges. To overcome these limitations, we propose Mixture of
Experts for Dynamic Gaussian Splatting (MoE-GS), a unified framework
integrating multiple specialized experts via a novel Volume-aware Pixel Router.
Our router adaptively blends expert outputs by projecting volumetric
Gaussian-level weights into pixel space through differentiable weight
splatting, ensuring spatially and temporally coherent results. Although MoE-GS
improves rendering quality, the increased model capacity and reduced FPS are
inherent to the MoE architecture. To mitigate this, we explore two
complementary directions: (1) single-pass multi-expert rendering and gate-aware
Gaussian pruning, which improve efficiency within the MoE framework, and (2) a
distillation strategy that transfers MoE performance to individual experts,
enabling lightweight deployment without architectural changes. To the best of
our knowledge, MoE-GS is the first approach incorporating Mixture-of-Experts
techniques into dynamic Gaussian splatting. Extensive experiments on the N3V
and Technicolor datasets demonstrate that MoE-GS consistently outperforms
state-of-the-art methods with improved efficiency. Video demonstrations are
available at https://anonymous.4open.science/w/MoE-GS-68BA/.

### 5. [SFGFusion: Surface Fitting Guided 3D Object Detection with 4D Radar and Camera Fusion](http://arxiv.org/pdf/2510.19215v1)

Authors: Xiaozhi Li, Huijun Di, Jian Li, Feng Liu, Wei Liang

3D object detection is essential for autonomous driving. As an emerging
sensor, 4D imaging radar offers advantages as low cost, long-range detection,
and accurate velocity measurement, making it highly suitable for object
detection. However, its sparse point clouds and low resolution limit object
geometric representation and hinder multi-modal fusion. In this study, we
introduce SFGFusion, a novel camera-4D imaging radar detection network guided
by surface fitting. By estimating quadratic surface parameters of objects from
image and radar data, the explicit surface fitting model enhances spatial
representation and cross-modal interaction, enabling more reliable prediction
of fine-grained dense depth. The predicted depth serves two purposes: 1) in an
image branch to guide the transformation of image features from perspective
view (PV) to a unified bird's-eye view (BEV) for multi-modal fusion, improving
spatial mapping accuracy; and 2) in a surface pseudo-point branch to generate
dense pseudo-point cloud, mitigating the radar point sparsity. The original
radar point cloud is also encoded in a separate radar branch. These two point
cloud branches adopt a pillar-based method and subsequently transform the
features into the BEV space. Finally, a standard 2D backbone and detection head
are used to predict object labels and bounding boxes from BEV features.
Experimental results show that SFGFusion effectively fuses camera and 4D radar
features, achieving superior performance on the TJ4DRadSet and view-of-delft
(VoD) object detection benchmarks.

### 6. [Space Object Detection using Multi-frame Temporal Trajectory Completion Method](http://arxiv.org/pdf/2510.19220v1)

Authors: Xiaoqing Lan, Biqiao Xin, Bingshu Wang, Han Zhang, Laixian Zhang

Space objects in Geostationary Earth Orbit (GEO) present significant
detection challenges in optical imaging due to weak signals, complex stellar
backgrounds, and environmental interference. In this paper, we enhance
high-frequency features of GEO targets while suppressing background noise at
the single-frame level through wavelet transform. Building on this, we propose
a multi-frame temporal trajectory completion scheme centered on the Hungarian
algorithm for globally optimal cross-frame matching. To effectively mitigate
missing and false detections, a series of key steps including temporal matching
and interpolation completion, temporal-consistency-based noise filtering, and
progressive trajectory refinement are designed in the post-processing pipeline.
Experimental results on the public SpotGEO dataset demonstrate the
effectiveness of the proposed method, achieving an F_1 score of 90.14%.

### 7. [Advances in 4D Representation: Geometry, Motion, and Interaction](http://arxiv.org/pdf/2510.19255v1)

Authors: Mingrui Zhao, Sauradip Nag, Kai Wang, Aditya Vora, Guangda Ji, Peter Chun, Ali Mahdavi-Amiri, Hao Zhang

We present a survey on 4D generation and reconstruction, a fast-evolving
subfield of computer graphics whose developments have been propelled by recent
advances in neural fields, geometric and motion deep learning, as well 3D
generative artificial intelligence (GenAI). While our survey is not the first
of its kind, we build our coverage of the domain from a unique and distinctive
perspective of 4D representations\/}, to model 3D geometry evolving over time
while exhibiting motion and interaction. Specifically, instead of offering an
exhaustive enumeration of many works, we take a more selective approach by
focusing on representative works to highlight both the desirable properties and
ensuing challenges of each representation under different computation,
application, and data scenarios. The main take-away message we aim to convey to
the readers is on how to select and then customize the appropriate 4D
representations for their tasks. Organizationally, we separate the 4D
representations based on three key pillars: geometry, motion, and interaction.
Our discourse will not only encompass the most popular representations of
today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS),
but also bring attention to relatively under-explored representations in the 4D
context, such as structured models and long-range motions. Throughout our
survey, we will reprise the role of large language models (LLMs) and video
foundational models (VFMs) in a variety of 4D applications, while steering our
discussion towards their current limitations and how they can be addressed. We
also provide a dedicated coverage on what 4D datasets are currently available,
as well as what is lacking, in driving the subfield forward. Project
page:https://mingrui-zhao.github.io/4DRep-GMI/

### 8. [SCEESR: Semantic-Control Edge Enhancement for Diffusion-Based Super-Resolution](http://arxiv.org/pdf/2510.19272v1)

Authors: Yun Kai Zhuang

Real-world image super-resolution (Real-ISR) must handle complex degradations
and inherent reconstruction ambiguities. While generative models have improved
perceptual quality, a key trade-off remains with computational cost. One-step
diffusion models offer speed but often produce structural inaccuracies due to
distillation artifacts. To address this, we propose a novel SR framework that
enhances a one-step diffusion model using a ControlNet mechanism for semantic
edge guidance. This integrates edge information to provide dynamic structural
control during single-pass inference. We also introduce a hybrid loss combining
L2, LPIPS, and an edge-aware AME loss to optimize for pixel accuracy,
perceptual quality, and geometric precision. Experiments show our method
effectively improves structural integrity and realism while maintaining the
efficiency of one-step generation, achieving a superior balance between output
quality and inference speed. The results of test datasets will be published at
https://drive.google.com/drive/folders/1amddXQ5orIyjbxHgGpzqFHZ6KTolinJF?usp=drive_link
and the related code will be published at
https://github.com/ARBEZ-ZEBRA/SCEESR.

### 9. [MobiAct: Efficient MAV Action Recognition Using MobileNetV4 with Contrastive Learning and Knowledge Distillation](http://arxiv.org/pdf/2510.19273v1)

Authors: Zhang Nengbo, Ho Hann Woei

Accurate and efficient recognition of Micro Air Vehicle (MAV) motion is
essential for enabling real-time perception and coordination in autonomous
aerial swarm. However, most existing approaches rely on large, computationally
intensive models that are unsuitable for resource-limited MAV platforms, which
results in a trade-off between recognition accuracy and inference speed. To
address these challenges, this paper proposes a lightweight MAV action
recognition framework, MobiAct, designed to achieve high accuracy with low
computational cost. Specifically, MobiAct adopts MobileNetV4 as the backbone
network and introduces a Stage-wise Orthogonal Knowledge Distillation (SOKD)
strategy to effectively transfer MAV motion features from a teacher network
(ResNet18) to a student network, thereby enhancing knowledge transfer
efficiency. Furthermore, a parameter-free attention mechanism is integrated
into the architecture to improve recognition accuracy without increasing model
complexity. In addition, a hybrid loss training strategy is developed to
combine multiple loss objectives, which ensures stable and robust optimization
during training. Experimental results demonstrate that the proposed MobiAct
achieves low-energy and low-computation MAV action recognition, while
maintaining the fastest action decoding speed among compared methods. Across
all three self-collected datasets, MobiAct achieves an average recognition
accuracy of 92.12%, while consuming only 136.16 pJ of energy and processing
recognition at a rate of 8.84 actions per second. Notably, MobiAct decodes
actions up to 2 times faster than the leading method, with highly comparable
recognition accuracy, highlighting its superior efficiency in MAV action
recognition.

### 10. [D2D: Detector-to-Differentiable Critic for Improved Numeracy in Text-to-Image Generation](http://arxiv.org/pdf/2510.19278v1)

Authors: Nobline Yoo, Olga Russakovsky, Ye Zhu

Text-to-image (T2I) diffusion models have achieved strong performance in
semantic alignment, yet they still struggle with generating the correct number
of objects specified in prompts. Existing approaches typically incorporate
auxiliary counting networks as external critics to enhance numeracy. However,
since these critics must provide gradient guidance during generation, they are
restricted to regression-based models that are inherently differentiable, thus
excluding detector-based models with superior counting ability, whose
count-via-enumeration nature is non-differentiable. To overcome this
limitation, we propose Detector-to-Differentiable (D2D), a novel framework that
transforms non-differentiable detection models into differentiable critics,
thereby leveraging their superior counting ability to guide numeracy
generation. Specifically, we design custom activation functions to convert
detector logits into soft binary indicators, which are then used to optimize
the noise prior at inference time with pre-trained T2I models. Our extensive
experiments on SDXL-Turbo, SD-Turbo, and Pixart-DMD across four benchmarks of
varying complexity (low-density, high-density, and multi-object scenarios)
demonstrate consistent and substantial improvements in object counting accuracy
(e.g., boosting up to 13.7% on D2D-Small, a 400-prompt, low-density benchmark),
with minimal degradation in overall image quality and computational overhead.

### Computers and Society

### 1. [Integration of AI in STEM Education, Addressing Ethical Challenges in K-12 Settings](http://arxiv.org/pdf/2510.19196v1)

Authors: Shaouna Shoaib Lodhi, Shoaib Lodhi

The rapid integration of Artificial Intelligence (AI) into K-12 STEM
education presents transformative opportunities alongside significant ethical
challenges. While AI-powered tools such as Intelligent Tutoring Systems (ITS),
automated assessments, and predictive analytics enhance personalized learning
and operational efficiency, they also risk perpetuating algorithmic bias,
eroding student privacy, and exacerbating educational inequities. This paper
examines the dual-edged impact of AI in STEM classrooms, analyzing its benefits
(e.g., adaptive learning, real-time feedback) and drawbacks (e.g., surveillance
risks, pedagogical limitations) through an ethical lens. We identify critical
gaps in current AI education research, particularly the lack of
subject-specific frameworks for responsible integration and propose a
three-phased implementation roadmap paired with a tiered professional
development model for educators. Our framework emphasizes equity-centered
design, combining technical AI literacy with ethical reasoning to foster
critical engagement among students. Key recommendations include mandatory bias
audits, low-resource adaptation strategies, and policy alignment to ensure AI
serves as a tool for inclusive, human-centered STEM education. By bridging
theory and practice, this work advances a research-backed approach to AI
integration that prioritizes pedagogical integrity, equity, and student agency
in an increasingly algorithmic world. Keywords: Artificial Intelligence, STEM
education, algorithmic bias, ethical AI, K-12 pedagogy, equity in education

### 2. [A Design Science Blueprint for an Orchestrated AI Assistant in Doctoral Supervision](http://arxiv.org/pdf/2510.19227v1)

Authors: Teo Susnjak, Timothy R. McIntosh, Tong Liu, Paul Watters

This study presents a design science blueprint for an orchestrated AI
assistant and co-pilot in doctoral supervision that acts as a socio-technical
mediator. Design requirements are derived from Stakeholder Theory and bounded
by Academic Integrity. We consolidated recent evidence on supervision gaps and
student wellbeing, then mapped issues to adjacent large language model
capabilities using a transparent severity-mitigability triage. The artefact
assembles existing capabilities into one accountable agentic AI workflow that
proposes retrieval-augmented generation and temporal knowledge graphs, as well
as mixture-of-experts routing as a solution stack of technologies to address
existing doctoral supervision pain points. Additionally, a student context
store is proposed, which introduces behaviour patches that turn tacit guidance
into auditable practice and student-set thresholds that trigger progress
summaries, while keeping authorship and final judgement with people. We specify
a student-initiated moderation loop in which assistant outputs are routed to a
supervisor for review and patching, and we analyse a reconfigured stakeholder
ecosystem that makes information explicit and accountable. Risks in such a
system exist, and among others, include AI over-reliance and the potential for
the illusion of learning, while guardrails are proposed. The contribution is an
ex ante, literature-grounded design with workflow and governance rules that
institutions can implement and trial across disciplines.

### 3. [Designing Knowledge Tools: How Students Transition from Using to Creating Generative AI in STEAM classroom](http://arxiv.org/pdf/2510.19405v1)

Authors: Qian Huang, Nachamma Sockalingam, Thijs Willems, King Wang Poon

This study explores how graduate students in an urban planning program
transitioned from passive users of generative AI to active creators of custom
GPT-based knowledge tools. Drawing on Self-Determination Theory (SDT), which
emphasizes the psychological needs of autonomy, competence, and relatedness as
foundations for intrinsic motivation, the research investigates how the act of
designing AI tools influences students' learning experiences, identity
formation, and engagement with knowledge. The study is situated within a
two-term curriculum, where students first used instructor-created GPTs to
support qualitative research tasks and later redesigned these tools to create
their own custom applications, including the Interview Companion GPT. Using
qualitative thematic analysis of student slide presentations and focus group
interviews, the findings highlight a marked transformation in students' roles
and mindsets. Students reported feeling more autonomous as they chose the
functionality, design, and purpose of their tools, more competent through the
acquisition of AI-related skills such as prompt engineering and iterative
testing, and more connected to peers through team collaboration and a shared
sense of purpose. The study contributes to a growing body of evidence that
student agency can be powerfully activated when learners are invited to
co-design the very technologies they use. The shift from AI tool users to AI
tool designers reconfigures students' relationships with technology and
knowledge, transforming them from consumers into co-creators in an evolving
educational landscape.

### 4. [IoT-Enabled Sleep Monitoring and Cognitive Assessment for Evaluating Teacher Well-Being](http://arxiv.org/pdf/2510.19269v1)

Authors: Anwar Ahmed Khan, Shama Siddiqui, Mehar Ullah, Indrakshi Dey

Sleep quality is an important indicator of the efficient cognitive function
for high school teachers. Due to the high work stress and multi-tasking
expectations, the teachers often face issues with their sleep quality and
cognitive function, which has a clearly negative influence on their teaching
abilities. In this work, we propose a unique but simple method of deploying
Internet of Things (IoT) technology to monitor the sleep quality of high school
teachers at Pakistan. Smart watches embedded with pulse rate and SpO2 sensors
were used to collect data and categorize the sleep quality as "poor", "fair" or
"good". Moreover, we used a psychological tool, Cognitive Assessment
Questionnaire (CAQ) for the self-assessment of teachers' cognitive function.
The study was conducted over 208 high school teachers from across Pakistan. It
has been found that most of the teachers had a poor sleep quality and cognitive
function; The link between these two variables indicate that the workload and
other factors must be improved for the teachers to ensure their well-being,
which will in turn have a positive impact on their teaching quality.

### 5. [Social World Model-Augmented Mechanism Design Policy Learning](http://arxiv.org/pdf/2510.19270v1)

Authors: Xiaoyuan Zhang, Yizhe Huang, Chengdong Ma, Zhixun Chen, Long Ma, Yali Du, Song-Chun Zhu, Yaodong Yang, Xue Feng

Designing adaptive mechanisms to align individual and collective interests
remains a central challenge in artificial social intelligence. Existing methods
often struggle with modeling heterogeneous agents possessing persistent latent
traits (e.g., skills, preferences) and dealing with complex multi-agent system
dynamics. These challenges are compounded by the critical need for high sample
efficiency due to costly real-world interactions. World Models, by learning to
predict environmental dynamics, offer a promising pathway to enhance mechanism
design in heterogeneous and complex systems. In this paper, we introduce a
novel method named SWM-AP (Social World Model-Augmented Mechanism Design Policy
Learning), which learns a social world model hierarchically modeling agents'
behavior to enhance mechanism design. Specifically, the social world model
infers agents' traits from their interaction trajectories and learns a
trait-based model to predict agents' responses to the deployed mechanisms. The
mechanism design policy collects extensive training trajectories by interacting
with the social world model, while concurrently inferring agents' traits online
during real-world interactions to further boost policy learning efficiency.
Experiments in diverse settings (tax policy design, team coordination, and
facility location) demonstrate that SWM-AP outperforms established model-based
and model-free RL baselines in cumulative rewards and sample efficiency.

### 6. [Code Sharing in Healthcare Research: A Practical Guide and Recommendations for Good Practice](http://arxiv.org/pdf/2510.19279v1)

Authors: Lukas Hughes-Noehrer, Matthew J Parkes, Andrew Stewart, Anthony J Wilson, Gary S Collins, Richard D Riley, Maya Mathur, Matthew P Fox, Nazrul Islam, Paul N Zivich, Timothy J Feeney

As computational analysis becomes increasingly more complex in health
research, transparent sharing of analytical code is vital for reproducibility
and trust. This practical guide, aligned to open science practices, outlines
actionable recommendations for code sharing in healthcare research. Emphasising
the FAIR (Findable, Accessible, Interoperable, Reusable) principles, the
authors address common barriers and provide clear guidance to help make code
more robust, reusable, and scrutinised as part of the scientific record. This
supports better science and more reliable evidence for computationally-driven
practice and helps to adhere to new standards and guidelines of codesharing
mandated by publishers and funding bodies.

### 7. [Mapping the AI Divide in Undergraduate Education: Community Detection in Disciplinary Networks and Survey Evidence](http://arxiv.org/pdf/2510.19288v1)

Authors: Liwen Zhang, Wei Si, Ke-ke Shang, Jiangli Zhu, Xiaomin Ji

As artificial intelligence-generated content (AIGC) reshapes knowledge
acquisition, higher education faces growing inequities that demand systematic
mapping and intervention. We map the AI divide in undergraduate education by
combining network science with survey evidence from 301 students at Nanjing
University, one of China's leading institutions in AI education. Drawing on
course enrolment patterns to construct a disciplinary network, we identify four
distinct student communities: science dominant, science peripheral, social
sciences & science, and humanities and social sciences. Survey results reveal
significant disparities in AIGC literacy and motivational efficacy, with
science dominant students outperforming humanities and social sciences peers.
Ordinary least squares (OLS) regression shows that motivational
efficacy--particularly skill efficacy--partially mediates this gap, whereas
usage efficacy does not mediate at the evaluation level, indicating a
dissociation between perceived utility and critical engagement. Our findings
demonstrate that curriculum structure and cross-disciplinary integration are
key determinants of technological fluency. This work provides a scalable
framework for diagnosing and addressing the AI divide through institutional
design.

### 8. [Algorithmic Fairness in NLP: Persona-Infused LLMs for Human-Centric Hate Speech Detection](http://arxiv.org/pdf/2510.19331v1)

Authors: Ewelina Gajewska, Arda Derbent, Jaroslaw A Chudziak, Katarzyna Budzynska

In this paper, we investigate how personalising Large Language Models
(Persona-LLMs) with annotator personas affects their sensitivity to hate
speech, particularly regarding biases linked to shared or differing identities
between annotators and targets. To this end, we employ Google's Gemini and
OpenAI's GPT-4.1-mini models and two persona-prompting methods: shallow persona
prompting and a deeply contextualised persona development based on
Retrieval-Augmented Generation (RAG) to incorporate richer persona profiles. We
analyse the impact of using in-group and out-group annotator personas on the
models' detection performance and fairness across diverse social groups. This
work bridges psychological insights on group identity with advanced NLP
techniques, demonstrating that incorporating socio-demographic attributes into
LLMs can address bias in automated hate speech detection. Our results highlight
both the potential and limitations of persona-based approaches in reducing
bias, offering valuable insights for developing more equitable hate speech
detection systems.

### 9. [To Use or to Refuse? Re-Centering Student Agency with Generative AI in Engineering Design Education](http://arxiv.org/pdf/2510.19342v1)

Authors: Thijs Willems, Sumbul Khan, Qian Huang, Bradley Camburn, Nachamma Sockalingam, King Wang Poon

This pilot study traces students' reflections on the use of AI in a 13-week
foundational design course enrolling over 500 first-year engineering and
architecture students at the Singapore University of Technology and Design. The
course was an AI-enhanced design course, with several interventions to equip
students with AI based design skills. Students were required to reflect on
whether the technology was used as a tool (instrumental assistant), a teammate
(collaborative partner), or neither (deliberate non-use). By foregrounding this
three-way lens, students learned to use AI for innovation rather than just
automation and to reflect on agency, ethics, and context rather than on prompt
crafting alone. Evidence stems from coursework artefacts: thirteen structured
reflection spreadsheets and eight illustrated briefs submitted, combined with
notes of teachers and researchers. Qualitative coding of these materials
reveals shared practices brought about through the inclusion of Gen-AI,
including accelerated prototyping, rapid skill acquisition, iterative prompt
refinement, purposeful "switch-offs" during user research, and emergent
routines for recognizing hallucinations. Unexpectedly, students not only
harnessed Gen-AI for speed but (enabled by the tool-teammate-neither triage)
also learned to reject its outputs, invent their own hallucination fire-drills,
and divert the reclaimed hours into deeper user research, thereby transforming
efficiency into innovation. The implications of the approach we explore shows
that: we can transform AI uptake into an assessable design habit; that
rewarding selective non-use cultivates hallucination-aware workflows; and,
practically, that a coordinated bundle of tool access, reflection, role
tagging, and public recognition through competition awards allows AI based
innovation in education to scale without compromising accountability.

### 10. [Cultural Dimensions of Artificial Intelligence Adoption: Empirical Insights for Wave 1 from a Multinational Longitudinal Pilot Study](http://arxiv.org/pdf/2510.19743v1)

Authors: Michelle J. Cummings-Koether, Franziska Durner, Theophile Shyiramunda, Matthias Huemmer

The swift diffusion of artificial intelligence (AI) raises critical questions
about how cultural contexts shape adoption patterns and their consequences for
human daily life. This study investigates the cultural dimensions of AI
adoption and their influence on cognitive strategies across nine national
contexts in Europe, Africa, Asia, and South America. Drawing on survey data
from a diverse pilot sample (n = 21) and guided by cross-cultural psychology,
digital ethics, and sociotechnical systems theory, we examine how demographic
variables (age, gender, professional role) and cultural orientations (language,
values, and institutional exposure) mediate perceptions of trust, ethical
acceptability, and reliance on AI. Results reveal two key findings: First,
cultural factors, particularly language and age, significantly affect AI
adoption and perceptions of reliability with older participants reporting
higher engagement with AI for educational purposes. Second, ethical judgment
about AI use varied across domains, with professional contexts normalizing its
role as a pragmatic collaborator while academic settings emphasized risks of
plagiarism. These findings extend prior research on culture and technology
adoption by demonstrating that AI use is neither universal nor neutral but
culturally contingent, domain-specific, and ethically situated. The study
highlights implications for AI use in education, professional practice, and
global technology policy, pointing at actions that enable usage of AI in a way
that is both culturally adaptive and ethically robust.

### Databases

### 1. [Fine-Grained Dichotomies for Conjunctive Queries with Minimum or Maximum](http://arxiv.org/pdf/2510.19197v1)

Authors: Nofar Carmeli, Nikolaos Tziavelis

We investigate the fine-grained complexity of direct access to Conjunctive
Query (CQ) answers according to their position, ordered by the minimum (or
maximum) value between attributes. We further use the tools we develop to
explore a wealth of related tasks. We consider the task of ranked enumeration
under min/max orders, as well as tasks concerning CQs with predicates of the
form x <= min X , where X is a set of variables and x is a single variable:
counting, enumeration, direct access, and predicate elimination (i.e.,
transforming the pair of query and database to an equivalent pair without
min-predicates). For each task, we establish a complete dichotomy for
self-join-free CQs, precisely identifying the cases that are solvable in
near-ideal time, i.e., (quasi)linear preprocessing time followed by constant or
logarithmic time per output.

### 2. [Next Generation Cloud-native In-Memory Stores: From Redis to Valkey and Beyond](http://arxiv.org/pdf/2510.19805v1)

Authors: Carl-Johan Fauvelle Munck af Rosensch"old, Feras M. Awaysheh, Ahmad Awad

In-memory key-value datastores have become indispensable building blocks of
modern cloud-native infrastructures, yet their evolution faces scalability,
compatibility, and sustainability constraints. The current literature lacks an
experimental evaluation of state-of-the-art tools in the domain. This study
addressed this timely gap by benchmarking Redis alternatives and systematically
evaluating Valkey, KeyDB, and Garnet under realistic workloads within
Kubernetes deployments. The results demonstrate clear trade-offs among the
benchmarked data systems. Our study presents a comprehensive performance and
viability assessment of the emerging in-memory key-value stores. Metrics
include throughput, tail latency, CPU and memory efficiency, and migration
complexity. We highlight trade-offs between performance, compatibility, and
long-term viability, including project maturity, community support, and
sustained development.

### Distributed, Parallel, and Cluster Computing

### 1. [FLASH Viterbi: Fast and Adaptive Viterbi Decoding for Modern Data Systems](http://arxiv.org/pdf/2510.19301v1)

Authors: Ziheng Deng, Xue Liu, Jiantong Jiang, Yankai Li, Qingxu Deng, Xiaochun Yang

The Viterbi algorithm is a key operator for structured sequence inference in
modern data systems, with applications in trajectory analysis, online
recommendation, and speech recognition. As these workloads increasingly migrate
to resource-constrained edge platforms, standard Viterbi decoding remains
memory-intensive and computationally inflexible. Existing methods typically
trade decoding time for space efficiency, but often incur significant runtime
overhead and lack adaptability to various system constraints. This paper
presents FLASH Viterbi, a Fast, Lightweight, Adaptive, and Hardware-Friendly
Viterbi decoding operator that enhances adaptability and resource efficiency.
FLASH Viterbi combines a non-recursive divide-and-conquer strategy with pruning
and parallelization techniques to enhance both time and memory efficiency,
making it well-suited for resource-constrained data systems.To further decouple
space complexity from the hidden state space size, we present FLASH-BS Viterbi,
a dynamic beam search variant built on a memory-efficient data structure. Both
proposed algorithms exhibit strong adaptivity to diverse deployment scenarios
by dynamically tuning internal parameters.To ensure practical deployment on
edge devices, we also develop FPGA-based hardware accelerators for both
algorithms, demonstrating high throughput and low resource usage. Extensive
experiments show that our algorithms consistently outperform existing baselines
in both decoding time and memory efficiency, while preserving adaptability and
hardware-friendly characteristics essential for modern data systems. All codes
are publicly available at https://github.com/Dzh-16/FLASH-Viterbi.

### 2. [Propius: A Platform for Collaborative Machine Learning across the Edge and the Cloud](http://arxiv.org/pdf/2510.19617v1)

Authors: Eric Ding

Collaborative Machine Learning is a paradigm in the field of distributed
machine learning, designed to address the challenges of data privacy,
communication overhead, and model heterogeneity. There have been significant
advancements in optimization and communication algorithm design and ML hardware
that enables fair, efficient and secure collaborative ML training. However,
less emphasis is put on collaborative ML infrastructure development. Developers
and researchers often build server-client systems for a specific collaborative
ML use case, which is not scalable and reusable. As the scale of collaborative
ML grows, the need for a scalable, efficient, and ideally multi-tenant resource
management system becomes more pressing. We propose a novel system, Propius,
that can adapt to the heterogeneity of client machines, and efficiently manage
and control the computation flow between ML jobs and edge resources in a
scalable fashion. Propius is comprised of a control plane and a data plane. The
control plane enables efficient resource sharing among multiple collaborative
ML jobs and supports various resource sharing policies, while the data plane
improves the scalability of collaborative ML model sharing and result
collection. Evaluations show that Propius outperforms existing resource
management techniques and frameworks in terms of resource utilization (up to
$1.88\times$), throughput (up to $2.76$), and job completion time (up to
$1.26\times$).

### 3. [On the Randomized Locality of Matching Problems in Regular Graphs](http://arxiv.org/pdf/2510.19151v1)

Authors: Seri Khoury, Manish Purohit, Aaron Schild, Joshua Wang

The main goal in distributed symmetry-breaking is to understand the locality
of problems; i.e., the radius of the neighborhood that a node needs to explore
in order to arrive at its part of a global solution. In this work, we study the
locality of matching problems in the family of regular graphs, which is one of
the main benchmarks for establishing lower bounds on the locality of
symmetry-breaking problems, as well as for obtaining classification results.
For approximate matching, we develop randomized algorithms to show that $(1 +
\epsilon)$-approximate matching in regular graphs is truly local; i.e., the
locality depends only on $\epsilon$ and is independent of all other graph
parameters. Furthermore, as long as the degree $\Delta$ is not very small
(namely, as long as $\Delta \geq \text{poly}(1/\epsilon)$), this dependence is
only logarithmic in $1/\epsilon$. This stands in sharp contrast to maximal
matching in regular graphs which requires some dependence on the number of
nodes $n$ or the degree $\Delta$. We show matching lower bounds for both
results. For maximal matching, our techniques further allow us to establish a
strong separation between the node-averaged complexity and worst-case
complexity of maximal matching in regular graphs, by showing that the former is
only $O(1)$. Central to our main technical contribution is a novel
martingale-based analysis for the $\approx 40$-year-old algorithm by Luby. In
particular, our analysis shows that applying one round of Luby's algorithm on
the line graph of a $\Delta$-regular graph results in an almost
$\Delta/2$-regular graph.

### 4. [RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs](http://arxiv.org/pdf/2510.19225v1)

Authors: Yongji Wu, Xueshen Liu, Haizhong Zheng, Juncheng Gu, Beidi Chen, Z. Morley Mao, Arvind Krishnamurthy, Ion Stoica

Reinforcement learning (RL) has become essential for unlocking advanced
reasoning capabilities in large language models (LLMs). RL workflows involve
interleaving rollout and training stages with fundamentally different resource
requirements. Rollout typically dominates overall execution time, yet scales
efficiently through multiple independent instances. In contrast, training
requires tightly-coupled GPUs with full-mesh communication. Existing RL
frameworks fall into two categories: co-located and disaggregated
architectures. Co-located ones fail to address this resource tension by forcing
both stages to share the same GPUs. Disaggregated architectures, without
modifications of well-established RL algorithms, suffer from resource
under-utilization. Meanwhile, preemptible GPU resources, i.e., spot instances
on public clouds and spare capacity in production clusters, present significant
cost-saving opportunities for accelerating RL workflows, if efficiently
harvested for rollout.
  In this paper, we present RLBoost, a systematic solution for cost-efficient
RL training that harvests preemptible GPU resources. Our key insight is that
rollout's stateless and embarrassingly parallel nature aligns perfectly with
preemptible and often fragmented resources. To efficiently utilize these
resources despite frequent and unpredictable availability changes, RLBoost
adopts a hybrid architecture with three key techniques: (1) adaptive rollout
offload to dynamically adjust workloads on the reserved (on-demand) cluster,
(2) pull-based weight transfer that quickly provisions newly available
instances, and (3) token-level response collection and migration for efficient
preemption handling and continuous load balancing. Extensive experiments show
RLBoost increases training throughput by 1.51x-1.97x while improving cost
efficiency by 28%-49% compared to using only on-demand GPU resources.

### 5. [RailS: Load Balancing for All-to-All Communication in Distributed Mixture-of-Experts Training](http://arxiv.org/pdf/2510.19262v1)

Authors: Heng Xu, Zhiwei Yu, Chengze Du, Ying Zhou, Letian Li, Haojie Wang, Weiqiang Cheng, Jialong Li

Training Mixture-of-Experts (MoE) models introduces sparse and highly
imbalanced all-to-all communication that dominates iteration time. Conventional
load-balancing methods fail to exploit the deterministic topology of Rail
architectures, leaving multi-NIC bandwidth underutilized. We present RailS, a
distributed load-balancing framework that minimizes all-to-all completion time
in MoE training. RailS leverages the Rail topology's symmetry to prove that
uniform sending ensures uniform receiving, transforming global coordination
into local scheduling. Each node independently executes a Longest Processing
Time First (LPT) spraying scheduler to proactively balance traffic using local
information. RailS activates N parallel rails for fine-grained, topology-aware
multipath transmission. Across synthetic and real-world MoE workloads, RailS
improves bus bandwidth by 20%--78% and reduces completion time by 17%--78%. For
Mixtral workloads, it shortens iteration time by 18%--40% and achieves
near-optimal load balance, fully exploiting architectural parallelism in
distributed training.

### 6. [CommonSense: Efficient Set Intersection (SetX) Protocol Based on Compressed Sensing](http://arxiv.org/pdf/2510.19725v1)

Authors: Jingfan Meng, Tianji Yang, Jun Xu

In the set reconciliation (\textsf{SetR}) problem, two parties Alice and Bob,
holding sets $\mathsf{A}$ and $\mathsf{B}$, communicate to learn the symmetric
difference $\mathsf{A} \Delta \mathsf{B}$. In this work, we study a related but
under-explored problem: set intersection (\textsf{SetX})~\cite{Ozisik2019},
where both parties learn $\mathsf{A} \cap \mathsf{B}$ instead. However,
existing solutions typically reuse \textsf{SetR} protocols due to the absence
of dedicated \textsf{SetX} protocols and the misconception that \textsf{SetR}
and \textsf{SetX} have comparable costs. Observing that \textsf{SetX} is
fundamentally cheaper than \textsf{SetR}, we developed a multi-round
\textsf{SetX} protocol that outperforms the information-theoretic lower bound
of \textsf{SetR} problem. In our \textsf{SetX} protocol, Alice sends Bob a
compressed sensing (CS) sketch of $\mathsf{A}$ to help Bob identify his unique
elements (those in $\mathsf{B \setminus A}$). This solves the \textsf{SetX}
problem, if $\mathsf{A} \subseteq \mathsf{B}$. Otherwise, Bob sends a CS sketch
of the residue (a set of elements he cannot decode) back to Alice for her to
decode her unique elements (those in $\mathsf{A \setminus B}$). As such, Alice
and Bob communicate back and forth %with a set membership filter (SMF) of
estimated $\mathsf{B \setminus A}$. Alice updates $\mathsf{A}$ and
communication repeats until both parties agrees on $\mathsf{A} \cap
\mathsf{B}$. On real world datasets, experiments show that our $\mathsf{SetX}$
protocol reduces the communication cost by 8 to 10 times compared to the
IBLT-based $\mathsf{SetR}$ protocol.

### 7. [Next Generation Cloud-native In-Memory Stores: From Redis to Valkey and Beyond](http://arxiv.org/pdf/2510.19805v1)

Authors: Carl-Johan Fauvelle Munck af Rosensch"old, Feras M. Awaysheh, Ahmad Awad

In-memory key-value datastores have become indispensable building blocks of
modern cloud-native infrastructures, yet their evolution faces scalability,
compatibility, and sustainability constraints. The current literature lacks an
experimental evaluation of state-of-the-art tools in the domain. This study
addressed this timely gap by benchmarking Redis alternatives and systematically
evaluating Valkey, KeyDB, and Garnet under realistic workloads within
Kubernetes deployments. The results demonstrate clear trade-offs among the
benchmarked data systems. Our study presents a comprehensive performance and
viability assessment of the emerging in-memory key-value stores. Metrics
include throughput, tail latency, CPU and memory efficiency, and migration
complexity. We highlight trade-offs between performance, compatibility, and
long-term viability, including project maturity, community support, and
sustained development.

### 8. [Enabling Reconfiguration-Communication Overlap for Collective Communication in Optical Networks](http://arxiv.org/pdf/2510.19322v1)

Authors: Changbo Wu, Zhuolong Yu, Gongming Zhao, Hongli Xu

Collective communication (CC) is widely adopted for large-scale distributed
machine learning (DML) training workloads. DML's predictable traffic pattern
provides a great oppotunity for applying optical network technology. Existing
optical interconnects-based CC schemes adopt ``one-shot network
reconfiguration'', which provisions static high-capacity topologies for an
entire collective operation -- sometimes for a full training iteration.
However, this approach faces significant scalability limitations when
supporting more complex and efficient CC algorithms required for modern
workloads: the ``one-shot'' strategies either demand excessive resource
overprovisioning or suffer performance degradation due to rigid resource
allocation.
  To address these challenges, we propose SWOT, a demand-aware optical network
framework. SWOT employs ``intra-collective reconfiguration'' and can
dynamically align network resources with CC traffic patterns. SWOT incorporates
a novel scheduling technique that overlaps optical switch reconfigurations with
ongoing transmissions, and improves communication efficiency. SWOT introduce a
lightweight collective communication shim that enables coordinated optical
network configuration and transmission scheduling while supporting seamless
integration with existing CC libraries. Our simulation results demonstrate
SWOT's significant performance improvements.

### 9. [HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid Expert/Data Transmission](http://arxiv.org/pdf/2510.19470v1)

Authors: Weihao Yang, Hao Huang, Donglei Wu, Ningke Li, Yanqi Pan, Qiyang Zheng, Wen Xia, Shiyi Li, Qiang Wang

Mixture-of-Experts (MoE) has become a popular architecture for scaling large
models. However, the rapidly growing scale outpaces model training on a single
DC, driving a shift toward a more flexible, cross-DC training paradigm. Under
this, Expert Parallelism (EP) of MoE faces significant scalability issues due
to the limited cross-DC bandwidth. Specifically, existing EP optimizations
attempt to overlap data communication and computation, which has little benefit
in low-bandwidth scenarios due to a much longer data communication time.
Therefore, the trends of cross-DC EP scaling is fast becoming a critical
roadblock to the continued growth of MoE models.
  To address this, we propose HybridEP, a modeling-guided framework to optimize
EP under constrained bandwidth. Our key idea is to dynamically transform the
spatial placement of experts to reduce data communication traffic and
frequency, thereby minimizing EP's communication overheads. However, it is
non-trivial to find the optimal solution because it complicates the original
communication pattern by mixing data and expert communication. We therefore
build a stream-based model to determine the optimal transmission ratio. Guided
by this, we incorporate two techniques: (1) domain-based partition to construct
the mapping between hybrid patterns and specific communication topology at GPU
level, and (2) parameter-efficient migration to further refine this topology by
reducing expert transmission overhead and enlarging the domain size. Combining
all these designs, HybridEP can be considered as a more general EP with better
scalability. Experimental results show that HybridEP outperforms existing
state-of-the-art MoE training systems by up to 5.6x under constrained
bandwidth. We further compare HybridEP and EP on large-scale simulations.
HybridEP achieves up to 1.45x speedup with 1k DCs under different bandwidths.

### 10. [Serverless GPU Architecture for Enterprise HR Analytics: A Production-Scale BDaaS Implementation](http://arxiv.org/pdf/2510.19689v1)

Authors: Guilin Zhang, Wulan Guo, Ziqi Tan, Srinivas Vippagunta, Suchitra Raman, Shreeshankar Chatterjee, Ju Lin, Shang Liu, Mary Schladenhauffen, Jeffrey Luo, Hailong Jiang

Industrial and government organizations increasingly depend on data-driven
analytics for workforce, finance, and regulated decision processes, where
timeliness, cost efficiency, and compliance are critical. Distributed
frameworks such as Spark and Flink remain effective for massive-scale batch or
streaming analytics but introduce coordination complexity and auditing
overheads that misalign with moderate-scale, latency-sensitive inference.
Meanwhile, cloud providers now offer serverless GPUs, and models such as TabNet
enable interpretable tabular ML, motivating new deployment blueprints for
regulated environments. In this paper, we present a production-oriented Big
Data as a Service (BDaaS) blueprint that integrates a single-node serverless
GPU runtime with TabNet. The design leverages GPU acceleration for throughput,
serverless elasticity for cost reduction, and feature-mask interpretability for
IL4/FIPS compliance. We conduct benchmarks on the HR, Adult, and BLS datasets,
comparing our approach against Spark and CPU baselines. Our results show that
GPU pipelines achieve up to 4.5x higher throughput, 98x lower latency, and 90%
lower cost per 1K inferences compared to Spark baselines, while compliance
mechanisms add only ~5.7 ms latency with p99 < 22 ms. Interpretability remains
stable under peak load, ensuring reliable auditability. Taken together, these
findings provide a compliance-aware benchmark, a reproducible Helm-packaged
blueprint, and a decision framework that demonstrate the practicality of
secure, interpretable, and cost-efficient serverless GPU analytics for
regulated enterprise and government settings.

### Digital Libraries

### 1. [Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark](http://arxiv.org/pdf/2510.19585v1)

Authors: Yu Wu, Ke Shu, Jonas Fischer, Lidia Pivovarova, David Rosson, Eetu Mäkelä, Mikko Tolonen

This paper presents a novel task of extracting Latin fragments from
mixed-language historical documents with varied layouts. We benchmark and
evaluate the performance of large foundation models against a multimodal
dataset of 724 annotated pages. The results demonstrate that reliable Latin
detection with contemporary models is achievable. Our study provides the first
comprehensive analysis of these models' capabilities and limits for this task.

### Discrete Mathematics

### 1. [The vertex visibility number of graphs](http://arxiv.org/pdf/2510.19452v1)

Authors: Dhanya Roy, Gabriele Di Stefano, Sandi Klavžar, Aparna Lakshmanan S

If $x\in V(G)$, then $S\subseteq V(G)\setminus\{x\}$ is an $x$-visibility set
if for any $y\in S$ there exists a shortest $x,y$-path avoiding $S$. The
$x$-visibility number $v_x(G)$ is the maximum cardinality of an $x$-visibility
set, and the maximum value of $v_x(G)$ among all vertices $x$ of $G$ is the
vertex visibility number ${\rm vv}(G)$ of $G$. It is proved that ${\rm vv}(G)$
is equal to the largest possible number of leaves of a shortest-path tree of
$G$. Deciding whether $v_x(G) \ge k$ holds for given $G$, a vertex $x\in V(G)$,
and a positive integer $k$ is NP-complete even for graphs of diameter $2$.
Several general sharp lower and upper bounds on the vertex visibility number
are proved. The vertex visibility number of Cartesian products is also bounded
from below and above, and the exact value of the vertex visibility number is
determined for square grids, square prisms, and square toruses.

### 2. [Burling graphs in graphs with large chromatic number](http://arxiv.org/pdf/2510.19650v1)

Authors: Tara Abrishami, Marcin Briański, James Davies, Xiying Du, Jana Masaříková, Paweł Rzążewski, Bartosz Walczak

A graph class is $\chi$-bounded if the only way to force large chromatic
number in graphs from the class is by forming a large clique. In the 1970s,
Erd\H{o}s conjectured that intersection graphs of straight-line segments in the
plane are $\chi$-bounded, but this was disproved by Pawlik et al. (2014), who
showed another way to force large chromatic number in this class -- by
triangle-free graphs $B_k$ with $\chi(B_k)=k$ constructed by Burling (1965).
This also disproved the celebrated conjecture of Scott (1997) that classes of
graphs excluding induced subdivisions of a fixed graph are $\chi$-bounded.
  We prove that in broad classes of graphs excluding induced subdivisions of a
fixed graph, including the increasingly more general classes of segment
intersection graphs, string graphs, region intersection graphs, and hereditary
classes of graphs with finite asymptotic dimension, large chromatic number can
be forced only by large cliques or large graphs $B_k$.
  One corollary is that the hereditary closure of $\{B_k\colon k\geq 1\}$ forms
a minimal hereditary graph class with unbounded chromatic number -- the second
known graph class with this property after the class of complete graphs.
Another corollary is that the decision variant of approximate coloring in the
aforementioned graph classes can be solved in polynomial time by exhaustively
searching for a sufficiently large clique or copy of $B_k$. We also discuss how
our results along with some results of Chudnovsky, Scott, and Seymour on the
existence of colorings can be turned into polynomial-time algorithms for the
search variant of approximate coloring in string graphs (with intersection
model in the input) and other aforementioned graph classes. Such an algorithm
has not yet been known for any graph class that is not $\chi$-bounded.

### 3. [Recognizing Leaf Powers and Pairwise Compatibility Graphs is NP-Complete](http://arxiv.org/pdf/2510.19763v1)

Authors: Max Dupré la Tour, Manuel Lafond, Ndiamé Ndiaye

Leaf powers and pairwise compatibility graphs were introduced over twenty
years ago as simplified graph models for phylogenetic trees. Despite
significant research, several properties of these graph classes remain poorly
understood. In this paper, we establish that the recognition problem for both
classes is NP-complete. We extend this hardness result to a broader hierarchy
of graph classes, including pairwise compatibility graphs and their
generalizations, multi interval pairwise compatibility graphs.

### 4. [Supermodular Maximization with Cardinality Constraints](http://arxiv.org/pdf/2510.19191v1)

Authors: Xujin Chen, Xiaodong Hu, Changjun Wang, Qingjie Ye

Let $V$ be a finite set of $n$ elements, $f: 2^V \rightarrow \mathbb{R}_+$ be
a nonnegative monotone supermodular function, and $k$ be a positive integer no
greater than $n$. This paper addresses the problem of maximizing $f(S)$ over
all subsets $S \subseteq V$ subject to the cardinality constraint $|S| = k$ or
$|S|\le k$.
  Let $r$ be a constant integer. The function $f$ is assumed to be {\em
$r$-decomposable}, meaning there exist $m\,(\ge1)$ subsets $V_1, \dots, V_m$ of
$V$, each with a cardinality at most $r$, and a corresponding set of
nonnegative supermodular functions $f_i : 2^{V_i} \rightarrow \mathbb{R}_+$,
$i=1,\ldots,m$ such that $f(S) =\sum_{i=1}^m f_i(S \cap V_i)$ holds for each $S
\subseteq V$. Given $r$ as an input, we present a polynomial-time
$O(n^{(r-1)/2})$-approximation algorithm for this maximization problem, which
does not require prior knowledge of the specific decomposition.
  When the decomposition $(V_i,f_i)_{i=1}^m$ is known, an additional
connectivity requirement is introduced to the problem. Let $G$ be the graph
with vertex set $V$ and edge set $\cup_{i=1}^m \{uv:u,v\in V_i,u\neq v\}$. The
cardinality constrained solution set $S$ is required to induce a connected
subgraph in $G$. This model generalizes the well-known problem of finding the
densest connected $k$-subgraph. We propose a polynomial time
$O(n^{(r-1)/2})$-approximation algorithm for this generalization. Notably, this
algorithm gives an $O(n^{1/2})$-approximation for the densest connected
$k$-subgraph problem, improving upon the previous best-known approximation
ratio of $O(n^{2/3})$.

### 5. [String graphs are quasi-isometric to planar graphs](http://arxiv.org/pdf/2510.19602v1)

Authors: James Davies

We prove that for every countable string graph $S$, there is a planar graph
$G$ with $V(G)=V(S)$ such that \[ \frac{1}{23660800}d_S(u,v) \le d_G(u,v) \le
162 d_S(u,v) \] for all $u,v\in V(S)$, where $d_S(u,v)$, $d_G(u,v)$ denotes the
distance between $u$ and $v$ in $S$ and $G$ respectively. In other words,
string graphs are quasi-isometric to planar graphs.
  This theorem lifts a number of theorems from planar graphs to string graphs,
we give some examples. String graphs have Assouad-Nagata (and asymptotic
dimension) at most 2. Connected, locally finite, quasi-transitive string graphs
are accessible. A finitely generated group $\Gamma$ is virtually a free product
of free and surface groups if and only if $\Gamma$ is quasi-isometric to a
string graph.
  Two further corollaries are that countable planar metric graphs and complete
Riemannian planes are also quasi-isometric to planar graphs, which answers a
question of Georgakopoulos and Papasoglu. For finite string graphs and planar
metric graphs, our proofs yield polynomial time (for string graphs, this is in
terms of the size of a representation given in the input) algorithms for
generating such quasi-isometric planar graphs.

### 6. [The maximal hard-core model as a recoverable system: Gibbs measures and phase coexistence](http://arxiv.org/pdf/2510.19746v1)

Authors: Geyang Wang, Alexander Barg, Navin Kashyap

Recoverable systems provide coarse models of data storage on the
two-dimensional square lattice, where each site reconstructs its value from
neighboring sites according to a specified local rule. To study the typical
behavior of recoverable patterns, this work introduces an interaction potential
on the local recovery regions of the lattice, which defines a corresponding
interaction model. We establish uniqueness of the Gibbs measure at high
temperature and derive bounds on the entropy in the zero- and low-temperature
regimes.
  For the recovery rule under consideration, exactly recoverable configurations
coincide with maximal independent sets of the grid. Relying on methods
developed for the standard hard-core model, we show phase coexistence at high
activity in the maximal case. Unlike the standard hard-core model, however, the
maximal version admits nontrivial ground states even at low activity, and we
manage to classify them explicitly. We further verify the Peierls condition for
the associated contour model. Combined with the Pirogov-Sinai theory, this
shows that each ground state gives rise to an extremal Gibbs measure, proving
phase coexistence at low activity.

### Data Structures and Algorithms

### 1. [Succinct Dynamic Rank/Select: Bypassing the Tree-Structure Bottleneck](http://arxiv.org/pdf/2510.19175v1)

Authors: William Kuszmaul, Jingxun Liang, Renfei Zhou

We show how to construct a dynamic ordered dictionary, supporting
insert/delete/rank/select on a set of $n$ elements from a universe of size $U$,
that achieves the optimal amortized expected time complexity of $O(1 + \log n /
\log \log U)$, while achieving a nearly optimal space consumption of $\log
\binom{U}{n} + n / 2^{(\log n)^{\Omega(1)}} + \text{polylog}\, U$ bits in the
regime where $U = \text{poly}(n)$. This resolves an open question by Pibiri and
Venturini as to whether a redundancy (a.k.a. space overhead) of $o(n)$ bits is
possible, and is the first dynamic solution to bypass the so-called
tree-structure bottleneck, in which the bits needed to encode some dynamic tree
structure are themselves enough to force a redundancy of
$\widetilde{\Omega}(n)$ bits. Our main technical building block is a dynamic
balanced binary search tree, which we call the compressed tabulation-weighted
treap, that itself achieves a surprising time/space tradeoff. The tree supports
$\text{polylog}\, n$-time operations and requires a static lookup table of size
$\text{poly}(n) + \text{polylog}\, U$ -- but, in exchange for these, the tree
is able to achieve a remarkable space guarantee. Its total space redundancy is
$O(\log U)$ bits. In fact, if the tree is given $n$ and $U$ for free, then the
redundancy further drops to $O(1)$ bits.

### 2. [Optimal Random Access and Conditional Lower Bounds for 2D Compressed Strings](http://arxiv.org/pdf/2510.19750v1)

Authors: Rajat De, Dominik Kempa

Compressed indexing is a powerful technique that enables efficient querying
over data stored in compressed form, significantly reducing memory usage and
often accelerating computation. While extensive progress has been made for
one-dimensional strings, many real-world datasets (such as images, maps, and
adjacency matrices) are inherently two-dimensional and highly compressible.
Unfortunately, naively applying 1D techniques to 2D data leads to suboptimal
results, as fundamental structural repetition is lost during linearization.
This motivates the development of native 2D compressed indexing schemes that
preserve both compression and query efficiency.
  We present three main contributions that advance the theory of compressed
indexing for 2D strings: (1) We design the first data structure that supports
optimal-time random access to a 2D string compressed by a 2D grammar.
Specifically, for a 2D string $T\in\Sigma^{r\times c}$ compressed by a 2D
grammar $G$ and any constant $\epsilon>0$, we achieve $O(\log n/\log \log n)$
query time and $O(|G|\log^{2+\epsilon}n)$ space, where $n=\max(r,c)$. (2) We
prove conditional lower bounds for pattern matching over 2D-grammar compressed
strings. Assuming the Orthogonal Vectors Conjecture, no algorithm can solve
this problem in time $O(|G|^{2-\epsilon}\cdot |P|^{O(1)})$ for any
$\epsilon>0$, demonstrating a separation from the 1D case, where optimal
solutions exist. (3) We show that several fundamental 2D queries, such as the
2D longest common extension, rectangle sum, and equality, cannot be supported
efficiently under hardness assumptions for rank and symbol occurrence queries
on 1D grammar-compressed strings. This is the first evidence connecting the
complexity of 2D compressed indexing to long-standing open problems in the 1D
setting.

### 3. [Strongly Polynomial Parallel Work-Depth Tradeoffs for Directed SSSP](http://arxiv.org/pdf/2510.19780v1)

Authors: Adam Karczmarz, Wojciech Nadara, Marek Sokołowski

In this paper, we show new strongly polynomial work-depth tradeoffs for
computing single-source shortest paths (SSSP) in non-negatively weighted
directed graphs in parallel. Most importantly, we prove that directed SSSP can
be solved within $\tilde{O}(m+n^{2-\epsilon})$ work and
$\tilde{O}(n^{1-\epsilon})$ depth for some positive $\epsilon>0$. In
particular, for dense graphs with non-negative real weights, we provide the
first nearly work-efficient strongly polynomial algorithm with sublinear depth.
  Our result immediately yields improved strongly polynomial parallel
algorithms for min-cost flow and the assignment problem. It also leads to the
first non-trivial strongly polynomial dynamic algorithm for minimum mean cycle.
Moreover, we develop efficient parallel algorithms in the Word RAM model for
several variants of SSSP in graphs with exponentially large edge weights.

### 4. [Explaining the Inherent Tradeoffs for Suffix Array Functionality: Equivalences between String Problems and Prefix Range Queries](http://arxiv.org/pdf/2510.19815v1)

Authors: Dominik Kempa, Tomasz Kociumaka

We study the fundamental question of how efficiently suffix array entries can
be accessed when the array cannot be stored explicitly. The suffix array
$SA_T[1..n]$ of a text $T$ of length $n$ encodes the lexicographic order of its
suffixes and underlies numerous applications in pattern matching, data
compression, and bioinformatics. Previous work established one-way reductions
showing how suffix array queries can be answered using, for example, rank
queries on the Burrows-Wheeler Transform. More recently, a new class of prefix
queries was introduced, together with reductions that, among others, transform
a simple tradeoff for prefix-select queries into a suffix array tradeoff
matching state-of-the-art space and query-time bounds, while achieving
sublinear construction time. For binary texts, the resulting data structure
achieves space $O(n)$ bits, preprocessing time $O(n / \sqrt{\log n})$,
preprocessing space of $O(n)$ bits, and query time $O(\log^{\epsilon} n)$ for
any constant $\epsilon > 0$. However, whether these bounds could be improved
using different techniques has remained open.
  We resolve this question by presenting the first bidirectional reduction
showing that suffix array queries are, up to an additive $O(\log\log n)$ term
in query time, equivalent to prefix-select queries in all parameters. This
result unifies prior approaches and shows that essentially all efficient suffix
array representations can be expressed via prefix-select structures. Moreover,
we prove analogous equivalences for inverse suffix array queries, pattern
ranking, lexicographic range, and SA-interval queries, identifying six core
problem pairs that connect string and prefix query models. Our framework thus
provides a unified foundation for analyzing and improving the efficiency of
fundamental string-processing problems through the lens of prefix queries.

### 5. [Tight Lower Bounds for Central String Queries in Compressed Space](http://arxiv.org/pdf/2510.19820v1)

Authors: Dominik Kempa, Tomasz Kociumaka

In this work, we study the limits of compressed data structures, i.e.,
structures that support various queries on an input text $T\in\Sigma^n$ using
space proportional to the size of $T$ in compressed form. Nearly all
fundamental queries can currently be efficiently supported in
$O(\delta(T)\log^{O(1)}n)$ space, where $\delta(T)$ is the substring
complexity, a strong compressibility measure that lower-bounds the optimal
space to represent the text [Kociumaka, Navarro, Prezza, IEEE Trans. Inf.
Theory 2023]. However, optimal query time has been characterized only for
random access.
  We address this gap by developing tight lower bounds for nearly all other
fundamental queries: (1) We prove that suffix array (SA), inverse suffix array
(SA$^{-1}$), longest common prefix (LCP) array, and longest common extension
(LCE) queries all require $\Omega(\log n/\log\log n)$ time within
$O(\delta(T)\log^{O(1)}n)$ space, matching known upper bounds. (2) We further
show that other common queries, currently supported in $O(\log\log n)$ time and
$O(\delta(T)\log^{O(1)}n)$ space, including the Burrows-Wheeler Transform
(BWT), permuted longest common prefix (PLCP) array, Last-to-First (LF), inverse
LF, lexicographic predecessor ($\Phi$), and inverse $\Phi$ queries, all require
$\Omega(\log\log n)$ time, yielding another set of tight bounds.
  Our lower bounds hold even for texts over a binary alphabet. This work
establishes a clean dichotomy: the optimal time complexity to support central
string queries in compressed space is either $\Theta(\log n/\log\log n)$ or
$\Theta(\log\log n)$. This completes the theoretical foundation of compressed
indexing, closing a crucial gap between upper and lower bounds and providing a
clear target for future data structures: seeking either the optimal time in the
smallest space or the fastest time in the optimal space, both of which are now
known for central string queries.

### 6. [On the Randomized Locality of Matching Problems in Regular Graphs](http://arxiv.org/pdf/2510.19151v1)

Authors: Seri Khoury, Manish Purohit, Aaron Schild, Joshua Wang

The main goal in distributed symmetry-breaking is to understand the locality
of problems; i.e., the radius of the neighborhood that a node needs to explore
in order to arrive at its part of a global solution. In this work, we study the
locality of matching problems in the family of regular graphs, which is one of
the main benchmarks for establishing lower bounds on the locality of
symmetry-breaking problems, as well as for obtaining classification results.
For approximate matching, we develop randomized algorithms to show that $(1 +
\epsilon)$-approximate matching in regular graphs is truly local; i.e., the
locality depends only on $\epsilon$ and is independent of all other graph
parameters. Furthermore, as long as the degree $\Delta$ is not very small
(namely, as long as $\Delta \geq \text{poly}(1/\epsilon)$), this dependence is
only logarithmic in $1/\epsilon$. This stands in sharp contrast to maximal
matching in regular graphs which requires some dependence on the number of
nodes $n$ or the degree $\Delta$. We show matching lower bounds for both
results. For maximal matching, our techniques further allow us to establish a
strong separation between the node-averaged complexity and worst-case
complexity of maximal matching in regular graphs, by showing that the former is
only $O(1)$. Central to our main technical contribution is a novel
martingale-based analysis for the $\approx 40$-year-old algorithm by Luby. In
particular, our analysis shows that applying one round of Luby's algorithm on
the line graph of a $\Delta$-regular graph results in an almost
$\Delta/2$-regular graph.

### 7. [Fine-Grained Dichotomies for Conjunctive Queries with Minimum or Maximum](http://arxiv.org/pdf/2510.19197v1)

Authors: Nofar Carmeli, Nikolaos Tziavelis

We investigate the fine-grained complexity of direct access to Conjunctive
Query (CQ) answers according to their position, ordered by the minimum (or
maximum) value between attributes. We further use the tools we develop to
explore a wealth of related tasks. We consider the task of ranked enumeration
under min/max orders, as well as tasks concerning CQs with predicates of the
form x <= min X , where X is a set of variables and x is a single variable:
counting, enumeration, direct access, and predicate elimination (i.e.,
transforming the pair of query and database to an equivalent pair without
min-predicates). For each task, we establish a complete dichotomy for
self-join-free CQs, precisely identifying the cases that are solvable in
near-ideal time, i.e., (quasi)linear preprocessing time followed by constant or
logarithmic time per output.

### 8. [Online Two-Stage Submodular Maximization](http://arxiv.org/pdf/2510.19480v1)

Authors: Iasonas Nikolaou, Miltiadis Stouras, Stratis Ioannidis, Evimaria Terzi

Given a collection of monotone submodular functions, the goal of Two-Stage
Submodular Maximization (2SSM) [Balkanski et al., 2016] is to restrict the
ground set so an objective selected u.a.r. from the collection attains a high
maximal value, on average, when optimized over the restricted ground set. We
introduce the Online Two-Stage Submodular Maximization (O2SSM) problem, in
which the submodular objectives are revealed in an online fashion. We study
this problem for weighted threshold potential functions, a large and important
subclass of monotone submodular functions that includes influence maximization,
data summarization, and facility location, to name a few. We design an
algorithm that achieves sublinear $(1 - 1/e)^2$-regret under general matroid
constraints and $(1 - 1/e)(1-e^{-k}k^k/k!)$-regret in the case of uniform
matroids of rank $k$; the latter also yields a state-of-the-art bound for the
(offline) 2SSM problem. We empirically validate the performance of our online
algorithm with experiments on real datasets.

### 9. [A Logic-based Algorithmic Meta-Theorem for Treedepth: Single Exponential FPT Time and Polynomial Space](http://arxiv.org/pdf/2510.19793v1)

Authors: Benjamin Bergougnoux, Vera Chekan, Giannos Stamoulis

For a graph $G$, the parameter treedepth measures the minimum depth among all
forests $F$, called elimination forests, such that $G$ is a subgraph of the
ancestor-descendant closure of $F$. We introduce a logic, called neighborhood
operator logic with acyclicity, connectivity and clique constraints
($\mathsf{NEO}_2[\mathsf{FRec}]+\mathsf{ACK}$ for short), that captures all
NP-hard problems$\unicode{x2013}$like Independent Set or Hamiltonian
Cycle$\unicode{x2013}$that are known to be tractable in time
$2^{\mathcal{O}(k)}n^{\mathcal{O}(1)}$ and space $n^{\mathcal{O}(1)}$ on
$n$-vertex graphs provided with elimination forests of depth $k$. We provide a
model checking algorithm for $\mathsf{NEO}_2[\mathsf{FRec}]+\mathsf{ACK}$ with
such complexity that unifies and extends these results. For
$\mathsf{NEO}_2[\mathsf{FRec}]+\mathsf{k}$, the fragment of the above logic
that does not use acyclicity and connectivity constraints, we get a
strengthening of this result, where the space complexity is reduced to
$\mathcal{O}(k\log(n))$.
  With a similar mechanism as the distance neighborhood logic introduced in
[Bergougnoux, Dreier and Jaffke, SODA 2023], the logic
$\mathsf{NEO}_2[\mathsf{FRec}]+\mathsf{ACK}$ is an extension of the
fully-existential $\mathsf{MSO}_2$ with predicates for (1) querying
generalizations of the neighborhoods of vertex sets, (2) verifying the
connectivity and acyclicity of vertex and edge sets, and (3) verifying that a
vertex set induces a clique. Our results provide
$2^{\mathcal{O}(k)}n^{\mathcal{O}(1)}$ time and $n^{\mathcal{O}(1)}$ space
algorithms for problems for which the existence of such algorithms was
previously unknown. In particular, $\mathsf{NEO}_2[\mathsf{FRec}]$ captures
CNF-SAT via the incidence graphs associated to CNF formulas, and it also
captures several modulo counting problems like Odd Dominating Set.

### 10. [Supermodular Maximization with Cardinality Constraints](http://arxiv.org/pdf/2510.19191v1)

Authors: Xujin Chen, Xiaodong Hu, Changjun Wang, Qingjie Ye

Let $V$ be a finite set of $n$ elements, $f: 2^V \rightarrow \mathbb{R}_+$ be
a nonnegative monotone supermodular function, and $k$ be a positive integer no
greater than $n$. This paper addresses the problem of maximizing $f(S)$ over
all subsets $S \subseteq V$ subject to the cardinality constraint $|S| = k$ or
$|S|\le k$.
  Let $r$ be a constant integer. The function $f$ is assumed to be {\em
$r$-decomposable}, meaning there exist $m\,(\ge1)$ subsets $V_1, \dots, V_m$ of
$V$, each with a cardinality at most $r$, and a corresponding set of
nonnegative supermodular functions $f_i : 2^{V_i} \rightarrow \mathbb{R}_+$,
$i=1,\ldots,m$ such that $f(S) =\sum_{i=1}^m f_i(S \cap V_i)$ holds for each $S
\subseteq V$. Given $r$ as an input, we present a polynomial-time
$O(n^{(r-1)/2})$-approximation algorithm for this maximization problem, which
does not require prior knowledge of the specific decomposition.
  When the decomposition $(V_i,f_i)_{i=1}^m$ is known, an additional
connectivity requirement is introduced to the problem. Let $G$ be the graph
with vertex set $V$ and edge set $\cup_{i=1}^m \{uv:u,v\in V_i,u\neq v\}$. The
cardinality constrained solution set $S$ is required to induce a connected
subgraph in $G$. This model generalizes the well-known problem of finding the
densest connected $k$-subgraph. We propose a polynomial time
$O(n^{(r-1)/2})$-approximation algorithm for this generalization. Notably, this
algorithm gives an $O(n^{1/2})$-approximation for the densest connected
$k$-subgraph problem, improving upon the previous best-known approximation
ratio of $O(n^{2/3})$.

### Emerging Technologies

### 1. [Machine Olfaction and Embedded AI Are Shaping the New Global Sensing Industry](http://arxiv.org/pdf/2510.19660v1)

Authors: Andreas Mershin, Nikolas Stefanou, Adan Rotteveel, Matthew Kung, George Kung, Alexandru Dan, Howard Kivell, Zoia Okulova, Zoi Kountouri, Paul Pu Liang

Machine olfaction is rapidly emerging as a transformative capability, with
applications spanning non-invasive medical diagnostics, industrial monitoring,
agriculture, and security and defense. Recent advances in stabilizing mammalian
olfactory receptors and integrating them into biophotonic and bioelectronic
systems have enabled detection at near single-molecule resolution thus placing
machines on par with trained detection dogs. As this technology converges with
multimodal AI and distributed sensor networks imbued with embedded AI, it
introduces a new, biochemical layer to a sensing ecosystem currently dominated
by machine vision and audition. This review and industry roadmap surveys the
scientific foundations, technological frontiers, and strategic applications of
machine olfaction making the case that we are currently witnessing the rise of
a new industry that brings with it a global chemosensory infrastructure. We
cover exemplary industrial, military and consumer applications and address some
of the ethical and legal concerns arising. We find that machine olfaction is
poised to bring forth a planet-wide molecular awareness tech layer with the
potential of spawning vast emerging markets in health, security, and
environmental sensing via scent.

### 2. [Res-DPU: Resource-shared Digital Processing-in-memory Unit for Edge-AI Workloads](http://arxiv.org/pdf/2510.19260v1)

Authors: Mukul Lokhande, Narendra Singh Dhakad, Seema Chouhan, Akash Sankhe, Santosh Kumar Vishvakarma

Processing-in-memory (PIM) has emerged as the go to solution for addressing
the von Neumann bottleneck in edge AI accelerators. However, state-of-the-art
(SoTA) digital PIM approaches suffer from low compute density, primarily due to
the use of bulky bit cells and transistor-heavy adder trees, which impose
limitations on macro scalability and energy efficiency. This work introduces
Res-DPU, a resource-shared digital PIM unit, with a dual-port 5T SRAM latch and
shared 2T AND compute logic. This reflects the per-bit multiplication cost to
just 5.25T and reduced the transistor count of the PIM array by up to 56% over
the SoTA works. Furthermore, a Transistor-Reduced 2D Interspersed Adder Tree
(TRAIT) with FA-7T and PG-FA-26T helps reduce the power consumption of the
adder tree by up to 21.35% and leads to improved energy efficiency by 59%
compared to conventional 28T RCA designs. We propose a Cycle-controlled
Iterative Approximate-Accurate Multiplication (CIA2M) approach, enabling
run-time accuracy-latency trade-offs without requiring error-correction
circuitry. The 16 KB REP-DPIM macro achieves 0.43 TOPS throughput and 87.22
TOPS/W energy efficiency in TSMC 65nm CMOS, with 96.85% QoR for ResNet-18 or
VGG-16 on CIFAR-10, including 30% pruning. The proposed results establish a
Res-DPU module for highly scalable and energy-efficient real-time edge AI
accelerators.

### 3. [A Probabilistic Computing Approach to the Closest Vector Problem for Lattice-Based Factoring](http://arxiv.org/pdf/2510.19390v1)

Authors: Max O. Al-Hasso, Marko von der Leyen

The closest vector problem (CVP) is a fundamental optimization problem in
lattice-based cryptography and its conjectured hardness underpins the security
of lattice-based cryptosystems. Furthermore, Schnorr's lattice-based factoring
algorithm reduces integer factoring (the foundation of current cryptosystems,
including RSA) to the CVP. Recent work has investigated the inclusion of a
heuristic CVP approximation `refinement' step in the lattice-based factoring
algorithm, using quantum variational algorithms to perform the heuristic
optimization. This coincides with the emergence of probabilistic computing as a
hardware accelerator for randomized algorithms including tasks in combinatorial
optimization. In this work we investigate the application of probabilistic
computing to the heuristic optimization task of CVP approximation refinement in
lattice-based factoring. We present the design of a probabilistic computing
algorithm for this task, a discussion of `prime lattice' parameters, and
experimental results showing the efficacy of probabilistic computing for
solving the CVP as well as its efficacy as a subroutine for lattice-based
factoring. The main results found that (a) this approach is capable of finding
the maximal available CVP approximation refinement in time linear in problem
size and (b) probabilistic computing used in conjunction with the lattice
parameters presented can find the composite prime factors of a semiprime number
using up to 100x fewer lattice instances than similar quantum and classical
methods.

### Formal Languages and Automata Theory

### 1. [Stochastic Languages at Sub-stochastic Cost](http://arxiv.org/pdf/2510.19276v1)

Authors: Smayan Agarwal, Aalok Thakkar

When does a deterministic computational model define a probability
distribution? What are its properties? This work formalises and settles this
stochasticity problem for weighted automata, and its generalisation cost
register automata (CRA).
  We show that checking stochasticity is undecidable for CRAs in general. This
motivates the study of the fully linear fragment, where a complete and
tractable theory is established. For this class, stochasticity becomes
decidable in polynomial time via spectral methods, and every stochastic linear
CRA admits an equivalent model with locally sub-stochastic update functions.
This provides a local syntactic characterisation of the semantics of the
quantitative model.
  This local characterisation allows us to provide an algebraic
Kleene-Schutzenberger characterisation for stochastic languages. The class of
rational stochastic languages is the smallest class containing finite support
distributions, which is closed under convex combination, Cauchy product, and
discounted Kleene star. We also introduce Stochastic Regular Expressions as a
complete and composable grammar for this class.
  Our framework provides the foundations for a formal theory of probabilistic
computation, with immediate consequences for approximation, sampling, and
distribution testing.

### 2. [Transformers are Inherently Succinct](http://arxiv.org/pdf/2510.19315v1)

Authors: Pascal Bergsträßer, Ryan Cotterell, Anthony W. Lin

We propose succinctness as a measure of the expressive power of a transformer
in describing a concept. To this end, we prove that transformers are highly
expressive in that they can represent formal languages substantially more
succinctly than standard representations of formal languages like finite
automata and Linear Temporal Logic (LTL) formulas. As a by-product of this
expressivity, we show that verifying properties of transformers is provably
intractable (i.e. EXPSPACE-complete).

### Graphics

### 1. [A New Type of Adversarial Examples](http://arxiv.org/pdf/2510.19347v1)

Authors: Xingyang Nie, Guojie Xiao, Su Pan, Biao Wang, Huilin Ge, Tao Fang

Most machine learning models are vulnerable to adversarial examples, which
poses security concerns on these models. Adversarial examples are crafted by
applying subtle but intentionally worst-case modifications to examples from the
dataset, leading the model to output a different answer from the original
example. In this paper, adversarial examples are formed in an exactly opposite
manner, which are significantly different from the original examples but result
in the same answer. We propose a novel set of algorithms to produce such
adversarial examples, including the negative iterative fast gradient sign
method (NI-FGSM) and the negative iterative fast gradient method (NI-FGM),
along with their momentum variants: the negative momentum iterative fast
gradient sign method (NMI-FGSM) and the negative momentum iterative fast
gradient method (NMI-FGM). Adversarial examples constructed by these methods
could be used to perform an attack on machine learning systems in certain
occasions. Moreover, our results show that the adversarial examples are not
merely distributed in the neighbourhood of the examples from the dataset;
instead, they are distributed extensively in the sample space.

### Computer Science and Game Theory

### 1. [Autobidding Arena: unified evaluation of the classical and RL-based autobidding algorithms](http://arxiv.org/pdf/2510.19357v1)

Authors: Andrey Pudovikov, Alexandra Khirianova, Ekaterina Solodneva, Aleksandr Katrutsa, Egor Samosvat, Yuriy Dorn

Advertisement auctions play a crucial role in revenue generation for
e-commerce companies. To make the bidding procedure scalable to thousands of
auctions, the automatic bidding (autobidding) algorithms are actively developed
in the industry. Therefore, the fair and reproducible evaluation of autobidding
algorithms is an important problem. We present a standardized and transparent
evaluation protocol for comparing classical and reinforcement learning (RL)
autobidding algorithms. We consider the most efficient autobidding algorithms
from different classes, e.g., ones based on the controllers, RL, optimal
formulas, etc., and benchmark them in the bidding environment. We utilize the
most recent open-source environment developed in the industry, which accurately
emulates the bidding process. Our work demonstrates the most promising use
cases for the considered autobidding algorithms, highlights their surprising
drawbacks, and evaluates them according to multiple metrics. We select the
evaluation metrics that illustrate the performance of the autobidding
algorithms, the corresponding costs, and track the budget pacing. Such a choice
of metrics makes our results applicable to the broad range of platforms where
autobidding is effective. The presented comparison results help practitioners
to evaluate the candidate autobidding algorithms from different perspectives
and select ones that are efficient according to their companies' targets.

### 2. [Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](http://arxiv.org/pdf/2510.19591v1)

Authors: Marius Potfer, Vianney Perchet

Repeated multi-unit auctions, where a seller allocates multiple identical
items over many rounds, are common mechanisms in electricity markets and
treasury auctions. We compare the two predominant formats: uniform-price and
discriminatory auctions, focusing on the perspective of a single bidder
learning to bid against stochastic adversaries. We characterize the learning
difficulty in each format, showing that the regret scales similarly for both
auction formats under both full-information and bandit feedback, as
$\tilde{\Theta} ( \sqrt{T} )$ and $\tilde{\Theta} ( T^{2/3} )$, respectively.
However, analysis beyond worst-case regret reveals structural differences:
uniform-price auctions may admit faster learning rates, with regret scaling as
$\tilde{\Theta} ( \sqrt{T} )$ in settings where discriminatory auctions remain
at $\tilde{\Theta} ( T^{2/3} )$. Finally, we provide a specific analysis for
auctions in which the other participants are symmetric and have unit-demand,
and show that in these instances, a similar regret rate separation appears.

### 3. [On Minimal Achievable Quotas in Multiwinner Voting](http://arxiv.org/pdf/2510.19620v1)

Authors: Patrick Becker, Fabian Frank

Justified representation (JR) and extended justified representation (EJR) are
well-established proportionality axioms in approval-based multiwinner voting.
Both axioms are always satisfiable, but they rely on a fixed quota (typically
Hare or Droop), with the Droop quota being the smallest one that guarantees
existence across all instances. With this observation in mind, we take a first
step beyond the fixed-quota paradigm and introduce proportionality notions
where the quota is instance-dependent. We demonstrate that all commonly studied
voting rules can have an additive distance to the optimum of
$\frac{k^2}{(k+1)^2}$. Moreover, we look into the computational aspects of our
instance-dependent quota and prove that determining the optimal value of
$\alpha$ for a given approval profile satisfying $\alpha$-JR is NP-complete. To
address this, we introduce an integer linear programming (ILP) formulation for
computing committees that satisfy $\alpha$-JR, and we provide positive results
in the voter interval (VI) and candidate interval (CI) domains.

### Human-Computer Interaction

### 1. [LLMartini: Seamless and Interactive Leveraging of Multiple LLMs through Comparison and Composition](http://arxiv.org/pdf/2510.19252v1)

Authors: Yingtian Shi, Jinda Yang, Yuhan Wang, Yiwen Yin, Haoyu Li, Kunyu Gao, Chun Yu

The growing diversity of large language models (LLMs) means users often need
to compare and combine outputs from different models to obtain higher-quality
or more comprehensive responses. However, switching between separate interfaces
and manually integrating outputs is inherently inefficient, leading to a high
cognitive burden and fragmented workflows. To address this, we present
LLMartini, a novel interactive system that supports seamless comparison,
selection, and intuitive cross-model composition tools. The system decomposes
responses into semantically aligned segments based on task-specific criteria,
automatically merges consensus content, and highlights model differences
through color coding while preserving unique contributions. In a user study
(N=18), LLMartini significantly outperformed conventional manual methods across
all measured metrics, including task completion time, cognitive load, and user
satisfaction. Our work highlights the importance of human-centered design in
enhancing the efficiency and creativity of multi-LLM interactions and offers
practical implications for leveraging the complementary strengths of various
language models.

### 2. [Design Considerations for Human Oversight of AI: Insights from Co-Design Workshops and Work Design Theory](http://arxiv.org/pdf/2510.19512v1)

Authors: Cedric Faas, Sophie Kerstan, Richard Uth, Markus Langer, Anna Maria Feit

As AI systems become increasingly capable and autonomous, domain experts'
roles are shifting from performing tasks themselves to overseeing AI-generated
outputs. Such oversight is critical, as undetected errors can have serious
consequences or undermine the benefits of AI. Effective oversight, however,
depends not only on detecting and correcting AI errors but also on the
motivation and engagement of the oversight personnel and the meaningfulness
they see in their work. Yet little is known about how domain experts approach
and experience the oversight task and what should be considered to design
effective and motivational interfaces that support human oversight. To address
these questions, we conducted four co-design workshops with domain experts from
psychology and computer science. We asked them to first oversee an AI-based
grading system, and then discuss their experiences and needs during oversight.
Finally, they collaboratively prototyped interfaces that could support them in
their oversight task. Our thematic analysis revealed four key user
requirements: understanding tasks and responsibilities, gaining insight into
the AI's decision-making, contributing meaningfully to the process, and
collaborating with peers and the AI. We integrated these empirical insights
with the SMART model of work design to develop a generalizable framework of
twelve design considerations. Our framework links interface characteristics and
user requirements to the psychological processes underlying effective and
satisfying work. Being grounded in work design theory, we expect these
considerations to be applicable across domains and discuss how they extend
existing guidelines for human-AI interaction and theoretical frameworks for
effective human oversight by providing concrete guidance on the design of
engaging and meaningful interfaces that support human oversight of AI systems.

### 3. [Unmanned Aerial Vehicles Control in a Digital Twin: Exploring the Effect of Different Points of View on User Experience in Virtual Reality](http://arxiv.org/pdf/2510.19604v1)

Authors: Francesco Vona, Mohamed Amer, Omar Abdellatif, Michelle Celina Hallmann, Maximilian Warsinke, Adriana-Simona Mihaita, Jan-Niklas Voigt-Antons

Controlling Unmanned Aerial Vehicles (UAVs) is a cognitively demanding task,
with accidents often arising from insufficient situational awareness,
inadequate training, and poor user experiences. Providing more intuitive and
immersive visual feedback, particularly through Digital Twin technologies,
offers new opportunities to enhance pilot awareness and overall experience
quality. In this study, we investigate how different virtual points of view
(POVs) influence user experience and performance during UAV piloting in Virtual
Reality (VR), utilizing a digital twin that faithfully replicates the
real-world flight environment. We developed a VR application that enables
participants to control a physical DJI Mini 4 Pro drone while immersed in a
digital twin with four distinct camera perspectives: Baseline View (static
external), First-Person View, Chase View, and Third-Person View. Nineteen
participants completed a series of ring-based obstacle courses from each
perspective. In addition to objective flight data, we collected standardized
subjective assessments of user experience, presence, workload, cybersickness,
and situational awareness. Quantitative analyses revealed that the First-Person
View was associated with significantly higher mental demand and effort, greater
trajectory deviation, but smoother control inputs compared to the Third-Person
and Chase perspectives. Complementing these findings, preference data indicated
that the Third-Person View was most consistently favored, whereas the
First-Person View elicited polarized reactions.

### 4. [Sentiment Analysis of Social Media Data for Predicting Consumer Behavior Trends Using Machine Learning](http://arxiv.org/pdf/2510.19656v1)

Authors: S M Rakib Ul Karim, Rownak Ara Rasul, Tunazzina Sultana

In the era of rapid technological advancement, social media platforms such as
Twitter (X) have emerged as indispensable tools for gathering consumer
insights, capturing diverse opinions, and understanding public attitudes. This
research applies advanced machine learning methods for sentiment analysis on
Twitter data, with a focus on predicting consumer trends. Using the
Sentiment140 dataset, the study detects evolving patterns in consumer
preferences with "car" as an example. A structured workflow was used to clean
and prepare data for analysis. Machine learning models, including Support
Vector Machines (SVM), Naive Bayes, Long Short-Term Memory (LSTM) networks, and
Bidirectional Encoder Representations from Transformers (BERT), were employed
to classify sentiments and predict trends. Model performance was measured using
accuracy, precision, recall, and F1 score, with BERT achieving the highest
results (Accuracy: 83.48%, Precision: 79.37%, Recall: 90.60%, F1: 84.61).
Results show that LSTM and BERT effectively capture linguistic and contextual
patterns, improving prediction accuracy and providing insights into consumer
behavior. Temporal analysis revealed sentiment shifts across time, while Named
Entity Recognition (NER) identified related terms and themes. This research
addresses challenges like sarcasm detection and multilingual data processing,
offering a scalable framework for generating actionable consumer insights.

### 5. [Interactive visualization of kidney micro-compartmental segmentations and associated pathomics on whole slide images](http://arxiv.org/pdf/2510.19499v1)

Authors: Mark S. Keller, Nicholas Lucarelli, Yijiang Chen, Samuel Border, Andrew Janowczyk, Jonathan Himmelfarb, Matthias Kretzler, Jeffrey Hodgin, Laura Barisoni, Dawit Demeke, Leal Herlitz, Gilbert Moeckel, Avi Z. Rosenberg, Yanli Ding, Pinaki Sarder, Nils Gehlenborg

Application of machine learning techniques enables segmentation of functional
tissue units in histology whole-slide images (WSIs). We built a pipeline to
apply previously validated segmentation models of kidney structures and extract
quantitative features from these structures. Such quantitative analysis also
requires qualitative inspection of results for quality control, exploration,
and communication. We extend the Vitessce web-based visualization tool to
enable visualization of segmentations of multiple types of functional tissue
units, such as, glomeruli, tubules, arteries/arterioles in the kidney.
Moreover, we propose a standard representation for files containing multiple
segmentation bitmasks, which we define polymorphically, such that existing
formats including OME-TIFF, OME-NGFF, AnnData, MuData, and SpatialData can be
used. We demonstrate that these methods enable researchers and the broader
public to interactively explore datasets containing multiple segmented entities
and associated features, including for exploration of renal morphometry of
biopsies from the Kidney Precision Medicine Project (KPMP) and the Human
Biomolecular Atlas Program (HuBMAP).

### 6. [EasyVitessce: auto-magically adding interactivity to Scverse single-cell and spatial biology plots](http://arxiv.org/pdf/2510.19532v1)

Authors: Selena Luo, Mark S. Keller, Tabassum Kakar, Lisa Choy, Nils Gehlenborg

EasyVitessce is a Python package that turns existing static Scanpy and
SpatialData plots into interactive visualizations by virtue of adding a single
line of Python code. The package uses Vitessce internally to render interactive
plots, and abstracts away technical details involved with configuration of
Vitessce. The resulting interactive plots can be viewed in computational
notebook environments or their configurations can be exported for usage in
other contexts such as web applications, enhancing the utility of popular
Scverse Python plotting APIs. EasyVitessce is released under the MIT License
and available on the Python Package Index (PyPI). The source code is publicly
available on GitHub.

### 7. [Directive, Metacognitive or a Blend of Both? A Comparison of AI-Generated Feedback Types on Student Engagement, Confidence, and Outcomes](http://arxiv.org/pdf/2510.19685v1)

Authors: Omar Alsaiari, Nilufar Baghaei, Jason M. Lodge, Omid Noroozi, Dragan Gašević, Marie Boden, Hassan Khosravi

Feedback is one of the most powerful influences on student learning, with
extensive research examining how best to implement it in educational settings.
Increasingly, feedback is being generated by artificial intelligence (AI),
offering scalable and adaptive responses. Two widely studied approaches are
directive feedback, which gives explicit explanations and reduces cognitive
load to speed up learning, and metacognitive feedback which prompts learners to
reflect, track their progress, and develop self-regulated learning (SRL)
skills. While both approaches have clear theoretical advantages, their
comparative effects on engagement, confidence, and quality of work remain
underexplored. This study presents a semester-long randomised controlled trial
with 329 students in an introductory design and programming course using an
adaptive educational platform. Participants were assigned to receive directive,
metacognitive, or hybrid AI-generated feedback that blended elements of both
directive and metacognitive feedback. Results showed that revision behaviour
differed across feedback conditions, with Hybrid prompting the most revisions
compared to Directive and Metacognitive. Confidence ratings were uniformly
high, and resource quality outcomes were comparable across conditions. These
findings highlight the promise of AI in delivering feedback that balances
clarity with reflection. Hybrid approaches, in particular, show potential to
combine actionable guidance for immediate improvement with opportunities for
self-reflection and metacognitive growth.

### 8. [Cultural Dimensions of Artificial Intelligence Adoption: Empirical Insights for Wave 1 from a Multinational Longitudinal Pilot Study](http://arxiv.org/pdf/2510.19743v1)

Authors: Michelle J. Cummings-Koether, Franziska Durner, Theophile Shyiramunda, Matthias Huemmer

The swift diffusion of artificial intelligence (AI) raises critical questions
about how cultural contexts shape adoption patterns and their consequences for
human daily life. This study investigates the cultural dimensions of AI
adoption and their influence on cognitive strategies across nine national
contexts in Europe, Africa, Asia, and South America. Drawing on survey data
from a diverse pilot sample (n = 21) and guided by cross-cultural psychology,
digital ethics, and sociotechnical systems theory, we examine how demographic
variables (age, gender, professional role) and cultural orientations (language,
values, and institutional exposure) mediate perceptions of trust, ethical
acceptability, and reliance on AI. Results reveal two key findings: First,
cultural factors, particularly language and age, significantly affect AI
adoption and perceptions of reliability with older participants reporting
higher engagement with AI for educational purposes. Second, ethical judgment
about AI use varied across domains, with professional contexts normalizing its
role as a pragmatic collaborator while academic settings emphasized risks of
plagiarism. These findings extend prior research on culture and technology
adoption by demonstrating that AI use is neither universal nor neutral but
culturally contingent, domain-specific, and ethically situated. The study
highlights implications for AI use in education, professional practice, and
global technology policy, pointing at actions that enable usage of AI in a way
that is both culturally adaptive and ethically robust.

### 9. [Learning To Defer To A Population With Limited Demonstrations](http://arxiv.org/pdf/2510.19351v1)

Authors: Nilesh Ramgolam, Gustavo Carneiro, Hsiang-Ting, Chen

This paper addresses the critical data scarcity that hinders the practical
deployment of learning to defer (L2D) systems to the population. We introduce a
context-aware, semi-supervised framework that uses meta-learning to generate
expert-specific embeddings from only a few demonstrations. We demonstrate the
efficacy of a dual-purpose mechanism, where these embeddings are used first to
generate a large corpus of pseudo-labels for training, and subsequently to
enable on-the-fly adaptation to new experts at test-time. The experiment
results on three different datasets confirm that a model trained on these
synthetic labels rapidly approaches oracle-level performance, validating the
data efficiency of our approach. By resolving a key training bottleneck, this
work makes adaptive L2D systems more practical and scalable, paving the way for
human-AI collaboration in real-world environments. To facilitate
reproducibility and address implementation details not covered in the main
text, we provide our source code and training configurations at
https://github.com/nil123532/learning-to-defer-to-a-population-with-limited-demonstrations.

### 10. [From Prototypes to Sparse ECG Explanations: SHAP-Driven Counterfactuals for Multivariate Time-Series Multi-class Classification](http://arxiv.org/pdf/2510.19514v1)

Authors: Maciej Mozolewski, Betül Bayrak, Kerstin Bach, Grzegorz J. Nalepa

In eXplainable Artificial Intelligence (XAI), instance-based explanations for
time series have gained increasing attention due to their potential for
actionable and interpretable insights in domains such as healthcare. Addressing
the challenges of explainability of state-of-the-art models, we propose a
prototype-driven framework for generating sparse counterfactual explanations
tailored to 12-lead ECG classification models. Our method employs SHAP-based
thresholds to identify critical signal segments and convert them into interval
rules, uses Dynamic Time Warping (DTW) and medoid clustering to extract
representative prototypes, and aligns these prototypes to query R-peaks for
coherence with the sample being explained. The framework generates
counterfactuals that modify only 78% of the original signal while maintaining
81.3% validity across all classes and achieving 43% improvement in temporal
stability. We evaluate three variants of our approach, Original, Sparse, and
Aligned Sparse, with class-specific performance ranging from 98.9% validity for
myocardial infarction (MI) to challenges with hypertrophy (HYP) detection
(13.2%). This approach supports near realtime generation (< 1 second) of
clinically valid counterfactuals and provides a foundation for interactive
explanation platforms. Our findings establish design principles for
physiologically-aware counterfactual explanations in AI-based diagnosis systems
and outline pathways toward user-controlled explanation interfaces for clinical
deployment.

### Information Retrieval

### 1. [C2T-ID: Converting Semantic Codebooks to Textual Document Identifiers for Generative Search](http://arxiv.org/pdf/2510.19221v1)

Authors: Yingchen Zhang, Ruqing Zhang, Jiafeng Guo, Wenjun Peng, Sen Li, Fuyu Lv, Xueqi Cheng

Designing document identifiers (docids) that carry rich semantic information
while maintaining tractable search spaces is a important challenge in
generative retrieval (GR). Popular codebook methods address this by building a
hierarchical semantic tree and constraining generation to its child nodes, yet
their numeric identifiers cannot leverage the large language model's pretrained
natural language understanding. Conversely, using text as docid provides more
semantic expressivity but inflates the decoding space, making the system
brittle to early-step errors. To resolve this trade-off, we propose C2T-ID: (i)
first construct semantic numerical docid via hierarchical clustering; (ii) then
extract high-frequency metadata keywords and iteratively replace each numeric
label with its cluster's top-K keywords; and (iii) an optional two-level
semantic smoothing step further enhances the fluency of C2T-ID. Experiments on
Natural Questions and Taobao's product search demonstrate that C2T-ID
significantly outperforms atomic, semantic codebook, and pure-text docid
baselines, demonstrating its effectiveness in balancing semantic expressiveness
with search space constraints.

### 2. [CoRECT: A Framework for Evaluating Embedding Compression Techniques at Scale](http://arxiv.org/pdf/2510.19340v1)

Authors: L. Caspari, M. Dinzinger, K. Gosh Dastidar, C. Fellicious, J. Mitrović, M. Granitzer

Dense retrieval systems have proven to be effective across various
benchmarks, but require substantial memory to store large search indices.
Recent advances in embedding compression show that index sizes can be greatly
reduced with minimal loss in ranking quality. However, existing studies often
overlook the role of corpus complexity -- a critical factor, as recent work
shows that both corpus size and document length strongly affect dense retrieval
performance. In this paper, we introduce CoRECT (Controlled Retrieval
Evaluation of Compression Techniques), a framework for large-scale evaluation
of embedding compression methods, supported by a newly curated dataset
collection. To demonstrate its utility, we benchmark eight representative types
of compression methods. Notably, we show that non-learned compression achieves
substantial index size reduction, even on up to 100M passages, with
statistically insignificant performance loss. However, selecting the optimal
compression method remains challenging, as performance varies across models.
Such variability highlights the necessity of CoRECT to enable consistent
comparison and informed selection of compression methods. All code, data, and
results are available on GitHub and HuggingFace.

### 3. [Top-P Masking for Cross Language Information Retrieval](http://arxiv.org/pdf/2510.19758v1)

Authors: Joseph Casale, Andrew Silverschotz, Joseph DeSimone

Top-K masking schemes have been proposed as a method to promote sparse
representations in Information Retrieval (IR) tasks, as a simple alternative to
Floating Point Operations per Second (FLOPS) regularization. Algorithms such as
Bilingual Lexical and Document Expansion Model (BLADE), adopt this approach as
a post-processing stage. We propose using Top-P Dynamic Masking similar to
Nucleus Sampling in Large Language Models, and demonstrate better performance
than Top-K masking. Specifically, we evaluate our methods in the domain of
Cross Language Information Retrieval (CLIR)

### 4. [ToolDreamer: Instilling LLM Reasoning Into Tool Retrievers](http://arxiv.org/pdf/2510.19791v1)

Authors: Saptarshi Sengupta, Zhengyu Zhou, Jun Araki, Xingbo Wang, Bingqing Wang, Suhang Wang, Zhe Feng

Tool calling has become increasingly popular for Large Language Models
(LLMs). However, for large tool sets, the resulting tokens would exceed the
LLM's context window limit, making it impossible to include every tool. Hence,
an external retriever is used to provide LLMs with the most relevant tools for
a query. Existing retrieval models rank tools based on the similarity between a
user query and a tool description (TD). This leads to suboptimal retrieval as
user requests are often poorly aligned with the language of TD. To remedy the
issue, we propose ToolDreamer, a framework to condition retriever models to
fetch tools based on hypothetical (synthetic) TD generated using an LLM, i.e.,
description of tools that the LLM feels will be potentially useful for the
query. The framework enables a more natural alignment between queries and tools
within the language space of TD's. We apply ToolDreamer on the ToolRet dataset
and show that our method improves the performance of sparse and dense
retrievers with and without training, thus showcasing its flexibility. Through
our proposed framework, our aim is to offload a portion of the reasoning burden
to the retriever so that the LLM may effectively handle a large collection of
tools without inundating its context window.

### 5. [Metadata Extraction Leveraging Large Language Models](http://arxiv.org/pdf/2510.19334v1)

Authors: Cuize Han, Sesh Jalagam

The advent of Large Language Models has revolutionized tasks across domains,
including the automation of legal document analysis, a critical component of
modern contract management systems. This paper presents a comprehensive
implementation of LLM-enhanced metadata extraction for contract review,
focusing on the automatic detection and annotation of salient legal clauses.
Leveraging both the publicly available Contract Understanding Atticus Dataset
(CUAD) and proprietary contract datasets, our work demonstrates the integration
of advanced LLM methodologies with practical applications. We identify three
pivotal elements for optimizing metadata extraction: robust text conversion,
strategic chunk selection, and advanced LLM-specific techniques, including
Chain of Thought (CoT) prompting and structured tool calling. The results from
our experiments highlight the substantial improvements in clause identification
accuracy and efficiency. Our approach shows promise in reducing the time and
cost associated with contract review while maintaining high accuracy in legal
clause identification. The results suggest that carefully optimized LLM systems
could serve as valuable tools for legal professionals, potentially increasing
access to efficient contract review services for organizations of all sizes.

### 6. [The Massive Legal Embedding Benchmark (MLEB)](http://arxiv.org/pdf/2510.19365v1)

Authors: Umar Butler, Abdur-Rahman Butler, Adrian Lucas Malec

We present the Massive Legal Embedding Benchmark (MLEB), the largest, most
diverse, and most comprehensive open-source benchmark for legal information
retrieval to date. MLEB consists of ten expert-annotated datasets spanning
multiple jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore),
document types (cases, legislation, regulatory guidance, contracts, and
literature), and task types (search, zero-shot classification, and question
answering). Seven of the datasets in MLEB were newly constructed in order to
fill domain and jurisdictional gaps in the open-source legal information
retrieval landscape. We document our methodology in building MLEB and creating
the new constituent datasets, and release our code, results, and data openly to
assist with reproducible evaluations.

### 7. [A Matter of Time: Revealing the Structure of Time in Vision-Language Models](http://arxiv.org/pdf/2510.19559v1)

Authors: Nidham Tekaya, Manuela Waldner, Matthias Zeppelzauer

Large-scale vision-language models (VLMs) such as CLIP have gained popularity
for their generalizable and expressive multimodal representations. By
leveraging large-scale training data with diverse textual metadata, VLMs
acquire open-vocabulary capabilities, solving tasks beyond their training
scope. This paper investigates the temporal awareness of VLMs, assessing their
ability to position visual content in time. We introduce TIME10k, a benchmark
dataset of over 10,000 images with temporal ground truth, and evaluate the
time-awareness of 37 VLMs by a novel methodology. Our investigation reveals
that temporal information is structured along a low-dimensional, non-linear
manifold in the VLM embedding space. Based on this insight, we propose methods
to derive an explicit ``timeline'' representation from the embedding space.
These representations model time and its chronological progression and thereby
facilitate temporal reasoning tasks. Our timeline approaches achieve
competitive to superior accuracy compared to a prompt-based baseline while
being computationally efficient. All code and data are available at
https://tekayanidham.github.io/timeline-page/.

### Machine Learning

### 1. [Subliminal Corruption: Mechanisms, Thresholds, and Interpretability](http://arxiv.org/pdf/2510.19152v1)

Authors: Reya Vir, Sarvesh Bhatnagar

As machine learning models are increasingly fine-tuned on synthetic data,
there is a critical risk of subtle misalignments spreading through
interconnected AI systems. This paper investigates subliminal corruption, which
we define as undesirable traits are transmitted through semantically neutral
data, bypassing standard safety checks. While this phenomenon has been
identified, a quantitative understanding of its dynamics is missing. To address
this gap, we present a systematic study of the scaling laws, thresholds, and
mechanisms of subliminal corruption using a teacher-student setup with GPT-2.
Our experiments reveal three key findings: (1) subliminal corruption causes
behavioral crossover, degrading the model's overall alignment, not just the
targeted trait; (2) alignment fails in a sharp phase transition at a critical
threshold of poisoned data, rather than degrading gradually; and (3)
interpretability analysis shows the corruption mechanism mimics the model's
natural fine-tuning process, making it difficult to detect. These results
demonstrate a critical vulnerability in AI systems that rely on synthetic data
and highlight the need for new safety protocols that can account for latent
threats.

### 2. [Feature Space Adaptation for Robust Model Fine-Tuning](http://arxiv.org/pdf/2510.19155v1)

Authors: Peng Wang, Minghao Gu, Qiang Huang

Catastrophic forgetting is a common issue in model fine-tuning, especially
when the downstream domain contains limited labeled data or differs greatly
from the pre-training distribution. Existing parameter-efficient fine-tuning
methods operate in the weight space by modifying or augmenting the pre-trained
model's parameters, which can yield models overly specialized to the available
downstream data. To mitigate the risk of overwriting pre-trained knowledge and
enhance robustness, we propose to fine-tune the pre-trained model in the
feature space. Two new fine-tuning methods are proposed: LoRFA (Low-Rank
Feature Adaptation) and VeFA (Vector-Based Feature Adaptation). Feature space
adaptation is inspired by the idea of effect equivalence modeling (EEM) of
downstream lurking variables causing distribution shifts, which posits that
unobserved factors can be represented as the total equivalent amount on
observed features. By compensating for the effects of downstream lurking
variables via a lightweight feature-level transformation, the pre-trained
representations can be preserved, which improves model generalization under
distribution shift. We evaluate LoRFA and VeFA versus LoRA on image
classification, NLU, and NLG, covering both standard fine-tuning metrics and
robustness. Feature space adaptation achieves comparable fine-tuning results
and consistently stronger robustness.

### 3. [Instance-Dependent Regret Bounds for Nonstochastic Linear Partial Monitoring](http://arxiv.org/pdf/2510.19158v1)

Authors: Federico Di Gennaro, Khaled Eldowa, Nicolò Cesa-Bianchi

In contrast to the classic formulation of partial monitoring, linear partial
monitoring can model infinite outcome spaces, while imposing a linear structure
on both the losses and the observations. This setting can be viewed as a
generalization of linear bandits where loss and feedback are decoupled in a
flexible manner. In this work, we address a nonstochastic (adversarial),
finite-actions version of the problem through a simple instance of the
exploration-by-optimization method that is amenable to efficient
implementation. We derive regret bounds that depend on the game structure in a
more transparent manner than previous theoretical guarantees for this paradigm.
Our bounds feature instance-specific quantities that reflect the degree of
alignment between observations and losses, and resemble known guarantees in the
stochastic setting. Notably, they achieve the standard $\sqrt{T}$ rate in easy
(locally observable) games and $T^{2/3}$ in hard (globally observable) games,
where $T$ is the time horizon. We instantiate these bounds in a selection of
old and new partial information settings subsumed by this model, and illustrate
that the achieved dependence on the game structure can be tight in interesting
cases.

### 4. [Preliminary Use of Vision Language Model Driven Extraction of Mouse Behavior Towards Understanding Fear Expression](http://arxiv.org/pdf/2510.19160v1)

Authors: Paimon Goulart, Jordan Steinhauser, Kylene Shuler, Edward Korzus, Jia Chen, Evangelos E. Papalexakis

Integration of diverse data will be a pivotal step towards improving
scientific explorations in many disciplines. This work establishes a
vision-language model (VLM) that encodes videos with text input in order to
classify various behaviors of a mouse existing in and engaging with their
environment. Importantly, this model produces a behavioral vector over time for
each subject and for each session the subject undergoes. The output is a
valuable dataset that few programs are able to produce with as high accuracy
and with minimal user input. Specifically, we use the open-source Qwen2.5-VL
model and enhance its performance through prompts, in-context learning (ICL)
with labeled examples, and frame-level preprocessing. We found that each of
these methods contributes to improved classification, and that combining them
results in strong F1 scores across all behaviors, including rare classes like
freezing and fleeing, without any model fine-tuning. Overall, this model will
support interdisciplinary researchers studying mouse behavior by enabling them
to integrate diverse behavioral features, measured across multiple time points
and environments, into a comprehensive dataset that can address complex
research questions.

### 5. [Enhancing Graph Neural Networks: A Mutual Learning Approach](http://arxiv.org/pdf/2510.19223v1)

Authors: Paul Agbaje, Akajyoti Mitra, Afia Anjum, Pranali Khose, Ebelechukwu Nwafor, Habeeb Olufowobi

Knowledge distillation (KD) techniques have emerged as a powerful tool for
transferring expertise from complex teacher models to lightweight student
models, particularly beneficial for deploying high-performance models in
resource-constrained devices. This approach has been successfully applied to
graph neural networks (GNNs), harnessing their expressive capabilities to
generate node embeddings that capture structural and feature-related
information. In this study, we depart from the conventional KD approach by
exploring the potential of collaborative learning among GNNs. In the absence of
a pre-trained teacher model, we show that relatively simple and shallow GNN
architectures can synergetically learn efficient models capable of performing
better during inference, particularly in tackling multiple tasks. We propose a
collaborative learning framework where ensembles of student GNNs mutually teach
each other throughout the training process. We introduce an adaptive logit
weighting unit to facilitate efficient knowledge exchange among models and an
entropy enhancement technique to improve mutual learning. These components
dynamically empower the models to adapt their learning strategies during
training, optimizing their performance for downstream tasks. Extensive
experiments conducted on three datasets each for node and graph classification
demonstrate the effectiveness of our approach.

### 6. [Brain-Inspired Perspective on Configurations: Unsupervised Similarity and Early Cognition](http://arxiv.org/pdf/2510.19229v1)

Authors: Juntang Wang, Yihan Wang, Hao Wu, Dongmian Zou, Shixin Xu

Infants discover categories, detect novelty, and adapt to new contexts
without supervision -- a challenge for current machine learning. We present a
brain-inspired perspective on configurations, a finite-resolution clustering
framework that uses a single resolution parameter and attraction-repulsion
dynamics to yield hierarchical organization, novelty sensitivity, and flexible
adaptation. To evaluate these properties, we introduce mheatmap, which provides
proportional heatmaps and a reassignment algorithm to fairly assess
multi-resolution and dynamic behavior. Across datasets, configurations are
competitive on standard clustering metrics, achieve 87% AUC in novelty
detection, and show 35% better stability during dynamic category evolution.
These results position configurations as a principled computational model of
early cognitive categorization and a step toward brain-inspired AI.

### 7. [Understanding the Implicit Biases of Design Choices for Time Series Foundation Models](http://arxiv.org/pdf/2510.19236v1)

Authors: Annan Yu, Danielle C. Maddix, Boran Han, Xiyuan Zhang, Abdul Fatir Ansari, Oleksandr Shchur, Christos Faloutsos, Andrew Gordon Wilson, Michael W. Mahoney, Yuyang Wang

Time series foundation models (TSFMs) are a class of potentially powerful,
general-purpose tools for time series forecasting and related temporal tasks,
but their behavior is strongly shaped by subtle inductive biases in their
design. Rather than developing a new model and claiming that it is better than
existing TSFMs, e.g., by winning on existing well-established benchmarks, our
objective is to understand how the various ``knobs'' of the training process
affect model quality. Using a mix of theory and controlled empirical
evaluation, we identify several design choices (patch size, embedding choice,
training objective, etc.) and show how they lead to implicit biases in
fundamental model properties (temporal behavior, geometric structure, how
aggressively or not the model regresses to the mean, etc.); and we show how
these biases can be intuitive or very counterintuitive, depending on properties
of the model and data. We also illustrate in a case study on outlier handling
how multiple biases can interact in complex ways; and we discuss implications
of our results for learning the bitter lesson and building TSFMs.

### 8. [Interpret Policies in Deep Reinforcement Learning using SILVER with RL-Guided Labeling: A Model-level Approach to High-dimensional and Multi-action Environments](http://arxiv.org/pdf/2510.19244v1)

Authors: Yiyu Qian, Su Nguyen, Chao Chen, Qinyue Zhou, Liyuan Zhao

Deep reinforcement learning (RL) achieves remarkable performance but lacks
interpretability, limiting trust in policy behavior. The existing SILVER
framework (Li, Siddique, and Cao 2025) explains RL policy via Shapley-based
regression but remains restricted to low-dimensional, binary-action domains. We
propose SILVER with RL-guided labeling, an enhanced variant that extends SILVER
to multi-action and high-dimensional environments by incorporating the RL
policy's own action outputs into the boundary points identification. Our method
first extracts compact feature representations from image observations,
performs SHAP-based feature attribution, and then employs RL-guided labeling to
generate behaviorally consistent boundary datasets. Surrogate models, such as
decision trees and regression-based functions, are subsequently trained to
interpret RL policy's decision structure. We evaluate the proposed framework on
two Atari environments using three deep RL algorithms and conduct human-subject
study to assess the clarity and trustworthiness of the derived interpretable
policy. Results show that our approach maintains competitive task performance
while substantially improving transparency and human understanding of agent
behavior. This work advances explainable RL by transforming SILVER into a
scalable and behavior-aware framework for interpreting deep RL agents in
high-dimensional, multi-action settings.

### 9. [Mixing Configurations for Downstream Prediction](http://arxiv.org/pdf/2510.19248v1)

Authors: Juntang Wang, Hao Wu, Runkun Guo, Yihan Wang, Dongmian Zou, Shixin Xu

Humans possess an innate ability to group objects by similarity, a cognitive
mechanism that clustering algorithms aim to emulate. Recent advances in
community detection have enabled the discovery of configurations -- valid
hierarchical clusterings across multiple resolution scales -- without requiring
labeled data. In this paper, we formally characterize these configurations and
identify similar emergent structures in register tokens within Vision
Transformers. Unlike register tokens, configurations exhibit lower redundancy
and eliminate the need for ad hoc selection. They can be learned through
unsupervised or self-supervised methods, yet their selection or composition
remains specific to the downstream task and input. Building on these insights,
we introduce GraMixC, a plug-and-play module that extracts configurations,
aligns them using our Reverse Merge/Split (RMS) technique, and fuses them via
attention heads before forwarding them to any downstream predictor. On the DSN1
16S rRNA cultivation-media prediction task, GraMixC improves the R2 score from
0.6 to 0.9 across multiple methods, setting a new state of the art. We further
validate GraMixC on standard tabular benchmarks, where it consistently
outperforms single-resolution and static-feature baselines.

### 10. [Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge](http://arxiv.org/pdf/2510.19266v1)

Authors: Penghao Wang, Yuhao Zhou, Mengxuan Wu, Panpan Zhang, Zhangyang Wang, Kai Wang

State-space models (SSMs) have emerged as efficient alternatives to
Transformers for sequence modeling, offering superior scalability through
recurrent structures. However, their training remains costly and the ecosystem
around them is far less mature than that of Transformers. Moreover, the
structural heterogeneity between SSMs and Transformers makes it challenging to
efficiently distill knowledge from pretrained attention models. In this work,
we propose Cross-architecture distillation via Attention Bridge (CAB), a novel
data-efficient distillation framework that efficiently transfers attention
knowledge from Transformer teachers to state-space student models. Unlike
conventional knowledge distillation that transfers knowledge only at the output
level, CAB enables token-level supervision via a lightweight bridge and
flexible layer-wise alignment, improving both efficiency and transferability.
We further introduce flexible layer-wise alignment strategies to accommodate
architectural discrepancies between teacher and student. Extensive experiments
across vision and language domains demonstrate that our method consistently
improves the performance of state-space models, even under limited training
data, outperforming both standard and cross-architecture distillation methods.
Our findings suggest that attention-based knowledge can be efficiently
transferred to recurrent models, enabling rapid utilization of Transformer
expertise for building a stronger SSM community.

### Neural and Evolutionary Computing

### 1. [A flexible framework for structural plasticity in GPU-accelerated sparse spiking neural networks](http://arxiv.org/pdf/2510.19764v1)

Authors: James C. Knight, Johanna Senk, Thomas Nowotny

The majority of research in both training Artificial Neural Networks (ANNs)
and modeling learning in biological brains focuses on synaptic plasticity,
where learning equates to changing the strength of existing connections.
However, in biological brains, structural plasticity - where new connections
are created and others removed - is also vital, not only for effective learning
but also for recovery from damage and optimal resource usage. Inspired by
structural plasticity, pruning is often used in machine learning to remove weak
connections from trained models to reduce the computational requirements of
inference. However, the machine learning frameworks typically used for
backpropagation-based training of both ANNs and Spiking Neural Networks (SNNs)
are optimized for dense connectivity, meaning that pruning does not help reduce
the training costs of ever-larger models. The GeNN simulator already supports
efficient GPU-accelerated simulation of sparse SNNs for computational
neuroscience and machine learning. Here, we present a new flexible framework
for implementing GPU-accelerated structural plasticity rules and demonstrate
this first using the e-prop supervised learning rule and DEEP R to train
efficient, sparse SNN classifiers and then, in an unsupervised learning
context, to learn topographic maps. Compared to baseline dense models, our
sparse classifiers reduce training time by up to 10x while the DEEP R rewiring
enables them to perform as well as the original models. We demonstrate
topographic map formation in faster-than-realtime simulations, provide insights
into the connectivity evolution, and measure simulation speed versus network
size. The proposed framework will enable further research into achieving and
maintaining sparsity in network structure and neural communication, as well as
exploring the computational benefits of sparsity in a range of neuromorphic
applications.

### Networking and Internet Architecture

### 1. [RailS: Load Balancing for All-to-All Communication in Distributed Mixture-of-Experts Training](http://arxiv.org/pdf/2510.19262v1)

Authors: Heng Xu, Zhiwei Yu, Chengze Du, Ying Zhou, Letian Li, Haojie Wang, Weiqiang Cheng, Jialong Li

Training Mixture-of-Experts (MoE) models introduces sparse and highly
imbalanced all-to-all communication that dominates iteration time. Conventional
load-balancing methods fail to exploit the deterministic topology of Rail
architectures, leaving multi-NIC bandwidth underutilized. We present RailS, a
distributed load-balancing framework that minimizes all-to-all completion time
in MoE training. RailS leverages the Rail topology's symmetry to prove that
uniform sending ensures uniform receiving, transforming global coordination
into local scheduling. Each node independently executes a Longest Processing
Time First (LPT) spraying scheduler to proactively balance traffic using local
information. RailS activates N parallel rails for fine-grained, topology-aware
multipath transmission. Across synthetic and real-world MoE workloads, RailS
improves bus bandwidth by 20%--78% and reduces completion time by 17%--78%. For
Mixtral workloads, it shortens iteration time by 18%--40% and achieves
near-optimal load balance, fully exploiting architectural parallelism in
distributed training.

### 2. [CommonSense: Efficient Set Intersection (SetX) Protocol Based on Compressed Sensing](http://arxiv.org/pdf/2510.19725v1)

Authors: Jingfan Meng, Tianji Yang, Jun Xu

In the set reconciliation (\textsf{SetR}) problem, two parties Alice and Bob,
holding sets $\mathsf{A}$ and $\mathsf{B}$, communicate to learn the symmetric
difference $\mathsf{A} \Delta \mathsf{B}$. In this work, we study a related but
under-explored problem: set intersection (\textsf{SetX})~\cite{Ozisik2019},
where both parties learn $\mathsf{A} \cap \mathsf{B}$ instead. However,
existing solutions typically reuse \textsf{SetR} protocols due to the absence
of dedicated \textsf{SetX} protocols and the misconception that \textsf{SetR}
and \textsf{SetX} have comparable costs. Observing that \textsf{SetX} is
fundamentally cheaper than \textsf{SetR}, we developed a multi-round
\textsf{SetX} protocol that outperforms the information-theoretic lower bound
of \textsf{SetR} problem. In our \textsf{SetX} protocol, Alice sends Bob a
compressed sensing (CS) sketch of $\mathsf{A}$ to help Bob identify his unique
elements (those in $\mathsf{B \setminus A}$). This solves the \textsf{SetX}
problem, if $\mathsf{A} \subseteq \mathsf{B}$. Otherwise, Bob sends a CS sketch
of the residue (a set of elements he cannot decode) back to Alice for her to
decode her unique elements (those in $\mathsf{A \setminus B}$). As such, Alice
and Bob communicate back and forth %with a set membership filter (SMF) of
estimated $\mathsf{B \setminus A}$. Alice updates $\mathsf{A}$ and
communication repeats until both parties agrees on $\mathsf{A} \cap
\mathsf{B}$. On real world datasets, experiments show that our $\mathsf{SetX}$
protocol reduces the communication cost by 8 to 10 times compared to the
IBLT-based $\mathsf{SetR}$ protocol.

### 3. [On the Power Saving in High-Speed Ethernet-based Networks for Supercomputers and Data Centers](http://arxiv.org/pdf/2510.19783v1)

Authors: Miguel Sánchez de la Rosa, Francisco J. andújar, Jesus Escudero-Sahuquillo, José L. Sánchez, Francisco J. Alfaro-Cortés

The increase in computation and storage has led to a significant growth in
the scale of systems powering applications and services, raising concerns about
sustainability and operational costs. In this paper, we explore power-saving
techniques in high-performance computing (HPC) and datacenter networks, and
their relation with performance degradation. From this premise, we propose
leveraging Energy Efficient Ethernet (EEE), with the flexibility to extend to
conventional Ethernet or upcoming Ethernet-derived interconnect versions of BXI
and Omnipath.
  We analyze the PerfBound proposal, identifying possible improvements and
modeling it into a simulation framework. Through different experiments, we
examine its impact on performance and determine the most appropriate
interconnect. We also study traffic patterns generated by selected HPC and
machine learning applications to evaluate the behavior of power-saving
techniques.
  From these experiments, we provide an analysis of how applications affect
system and network energy consumption. Based on this, we disclose the weakness
of dynamic power-down mechanisms and propose an approach that improves energy
reduction with minimal or no performance penalty. To our knowledge, this is the
first power management proposal tailored to future Ethernet-based HPC
architectures, with promising results.

### 4. [LAPRAD: LLM-Assisted PRotocol Attack Discovery](http://arxiv.org/pdf/2510.19264v1)

Authors: R. Can Aygun, Yehuda Afek, Anat Bremler-Barr, Leonard Kleinrock

With the goal of improving the security of Internet protocols, we seek
faster, semi-automatic methods to discover new vulnerabilities in protocols
such as DNS, BGP, and others. To this end, we introduce the LLM-Assisted
Protocol Attack Discovery (LAPRAD) methodology, enabling security researchers
with some DNS knowledge to efficiently uncover vulnerabilities that would
otherwise be hard to detect.
  LAPRAD follows a three-stage process. In the first, we consult an LLM
(GPT-o1) that has been trained on a broad corpus of DNS-related sources and
previous DDoS attacks to identify potential exploits. In the second stage, a
different LLM automatically constructs the corresponding attack configurations
using the ReACT approach implemented via LangChain (DNS zone file generation).
Finally, in the third stage, we validate the attack's functionality and
effectiveness.
  Using LAPRAD, we uncovered three new DDoS attacks on the DNS protocol and
rediscovered two recently reported ones that were not included in the LLM's
training data. The first new attack employs a bait-and-switch technique to
trick resolvers into caching large, bogus DNSSEC RRSIGs, reducing their serving
capacity to as little as 6%. The second exploits large DNSSEC encryption
algorithms (RSA-4096) with multiple keys, thereby bypassing a recently
implemented default RRSet limit. The third leverages ANY-type responses to
produce a similar effect.
  These variations of a cache-flushing DDoS attack, called SigCacheFlush,
circumvent existing patches, severely degrade resolver query capacity, and
impact the latest versions of major DNS resolver implementations.

### 5. [Enabling Reconfiguration-Communication Overlap for Collective Communication in Optical Networks](http://arxiv.org/pdf/2510.19322v1)

Authors: Changbo Wu, Zhuolong Yu, Gongming Zhao, Hongli Xu

Collective communication (CC) is widely adopted for large-scale distributed
machine learning (DML) training workloads. DML's predictable traffic pattern
provides a great oppotunity for applying optical network technology. Existing
optical interconnects-based CC schemes adopt ``one-shot network
reconfiguration'', which provisions static high-capacity topologies for an
entire collective operation -- sometimes for a full training iteration.
However, this approach faces significant scalability limitations when
supporting more complex and efficient CC algorithms required for modern
workloads: the ``one-shot'' strategies either demand excessive resource
overprovisioning or suffer performance degradation due to rigid resource
allocation.
  To address these challenges, we propose SWOT, a demand-aware optical network
framework. SWOT employs ``intra-collective reconfiguration'' and can
dynamically align network resources with CC traffic patterns. SWOT incorporates
a novel scheduling technique that overlaps optical switch reconfigurations with
ongoing transmissions, and improves communication efficiency. SWOT introduce a
lightweight collective communication shim that enables coordinated optical
network configuration and transmission scheduling while supporting seamless
integration with existing CC libraries. Our simulation results demonstrate
SWOT's significant performance improvements.

### Robotics

### 1. [TARMAC: A Taxonomy for Robot Manipulation in Chemistry](http://arxiv.org/pdf/2510.19289v1)

Authors: Kefeng Huang, Jonathon Pipe, Alice E. Martin, Tianyuan Wang, Barnabas A. Franklin, Andy M. Tyrrell, Ian J. S. Fairlamb, Jihong Zhu

Chemistry laboratory automation aims to increase throughput, reproducibility,
and safety, yet many existing systems still depend on frequent human
intervention. Advances in robotics have reduced this dependency, but without a
structured representation of the required skills, autonomy remains limited to
bespoke, task-specific solutions with little capacity to transfer beyond their
initial design. Current experiment abstractions typically describe
protocol-level steps without specifying the robotic actions needed to execute
them. This highlights the lack of a systematic account of the manipulation
skills required for robots in chemistry laboratories. To address this gap, we
introduce TARMAC - a Taxonomy for Robot Manipulation in Chemistry - a
domain-specific framework that defines and organizes the core manipulations
needed in laboratory practice. Based on annotated teaching-lab demonstrations
and supported by experimental validation, TARMAC categorizes actions according
to their functional role and physical execution requirements. Beyond serving as
a descriptive vocabulary, TARMAC can be instantiated as robot-executable
primitives and composed into higher-level macros, enabling skill reuse and
supporting scalable integration into long-horizon workflows. These
contributions provide a structured foundation for more flexible and autonomous
laboratory automation. More information is available at
https://tarmac-paper.github.io/

### 2. [Imitation Learning Policy based on Multi-Step Consistent Integration Shortcut Model](http://arxiv.org/pdf/2510.19356v1)

Authors: Yu Fang, Xinyu Wang, Xuehe Zhang, Wanli Xue, Mingwei Zhang, Shengyong Chen, Jie Zhao

The wide application of flow-matching methods has greatly promoted the
development of robot imitation learning. However, these methods all face the
problem of high inference time. To address this issue, researchers have
proposed distillation methods and consistency methods, but the performance of
these methods still struggles to compete with that of the original diffusion
models and flow-matching models. In this article, we propose a one-step
shortcut method with multi-step integration for robot imitation learning. To
balance the inference speed and performance, we extend the multi-step
consistency loss on the basis of the shortcut model, split the one-step loss
into multi-step losses, and improve the performance of one-step inference.
Secondly, to solve the problem of unstable optimization of the multi-step loss
and the original flow-matching loss, we propose an adaptive gradient allocation
method to enhance the stability of the learning process. Finally, we evaluate
the proposed method in two simulation benchmarks and five real-world
environment tasks. The experimental results verify the effectiveness of the
proposed algorithm.

### 3. [ProTerrain: Probabilistic Physics-Informed Rough Terrain World Modeling](http://arxiv.org/pdf/2510.19364v1)

Authors: Golnaz Raja, Ruslan Agishev, Miloš Prágr, Joni Pajarinen, Karel Zimmermann, Arun Kumar Singh, Reza Ghabcheloo

Uncertainty-aware robot motion prediction is crucial for downstream
traversability estimation and safe autonomous navigation in unstructured,
off-road environments, where terrain is heterogeneous and perceptual
uncertainty is high. Most existing methods assume deterministic or spatially
independent terrain uncertainties, ignoring the inherent local correlations of
3D spatial data and often producing unreliable predictions. In this work, we
introduce an efficient probabilistic framework that explicitly models spatially
correlated aleatoric uncertainty over terrain parameters as a probabilistic
world model and propagates this uncertainty through a differentiable physics
engine for probabilistic trajectory forecasting. By leveraging structured
convolutional operators, our approach provides high-resolution multivariate
predictions at manageable computational cost. Experimental evaluation on a
publicly available dataset shows significantly improved uncertainty estimation
and trajectory prediction accuracy over aleatoric uncertainty estimation
baselines.

### 4. [Optimizing Prosthetic Wrist Movement: A Model Predictive Control Approach](http://arxiv.org/pdf/2510.19541v1)

Authors: Francesco Schetter, Shifa Sulaiman, Shoby George, Paolino De Risi, Fanny Ficuciello

The integration of advanced control strategies into prosthetic hands is
essential to improve their adaptability and performance. In this study, we
present an implementation of a Model Predictive Control (MPC) strategy to
regulate the motions of a soft continuum wrist section attached to a
tendon-driven prosthetic hand with less computational effort. MPC plays a
crucial role in enhancing the functionality and responsiveness of prosthetic
hands. By leveraging predictive modeling, this approach enables precise
movement adjustments while accounting for dynamic user interactions. This
advanced control strategy allows for the anticipation of future movements and
adjustments based on the current state of the prosthetic device and the
intentions of the user. Kinematic and dynamic modelings are performed using
Euler-Bernoulli beam and Lagrange methods respectively. Through simulation and
experimental validations, we demonstrate the effectiveness of MPC in optimizing
wrist articulation and user control. Our findings suggest that this technique
significantly improves the prosthetic hand dexterity, making movements more
natural and intuitive. This research contributes to the field of robotics and
biomedical engineering by offering a promising direction for intelligent
prosthetic systems.

### 5. [LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments](http://arxiv.org/pdf/2510.19655v1)

Authors: Hongyu Ding, Ziming Xu, Yudong Fang, You Wu, Zixuan Chen, Jieqi Shi, Jing Huo, Yifan Zhang, Yang Gao

Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE)
requires an agent to navigate unseen environments based on natural language
instructions without any prior training. Current methods face a critical
trade-off: either rely on environment-specific waypoint predictors that limit
scene generalization, or underutilize the reasoning capabilities of large
models during navigation. We introduce LaViRA, a simple yet effective zero-shot
framework that addresses this dilemma by decomposing action into a
coarse-to-fine hierarchy: Language Action for high-level planning, Vision
Action for perceptual grounding, and Robot Action for robust navigation. This
modular decomposition allows us to leverage the distinct strengths of different
scales of Multimodal Large Language Models (MLLMs) at each stage, creating a
system that is powerful in its reasoning, grounding and practical control.
LaViRA significantly outperforms existing state-of-the-art methods on the
VLN-CE benchmark, demonstrating superior generalization capabilities in unseen
environments, while maintaining transparency and efficiency for real-world
deployment.

### 6. [Fast Marker Detection for UV-Based Visual Relative Localisation in Agile UAV Swarms](http://arxiv.org/pdf/2510.19663v1)

Authors: Vojtěch Vrba, Viktor Walter, Petr Štěpán, Martin Saska

A novel approach for the fast onboard detection of isolated markers for
visual relative localisation of multiple teammates in agile UAV swarms is
introduced in this paper. As the detection forms a key component of real-time
localisation systems, a three-fold innovation is presented, consisting of an
optimised procedure for CPUs, a GPU shader program, and a functionally
equivalent FPGA streaming architecture. For the proposed CPU and GPU solutions,
the mean processing time per pixel of input camera frames was accelerated by
two to three orders of magnitude compared to the state of the art. For the
localisation task, the proposed FPGA architecture offered the most significant
overall acceleration by minimising the total delay from camera exposure to
detection results. Additionally, the proposed solutions were evaluated on
various 32-bit and 64-bit embedded platforms to demonstrate their efficiency,
as well as their feasibility for applications using low-end UAVs and MAVs.
Thus, it has become a crucial enabling technology for agile UAV swarming.

### 7. [SEA: Semantic Map Prediction for Active Exploration of Uncertain Areas](http://arxiv.org/pdf/2510.19766v1)

Authors: Hongyu Ding, Xinyue Liang, Yudong Fang, You Wu, Jieqi Shi, Jing Huo, Wenbin Li, Jing Wu, Yu-Kun Lai, Yang Gao

In this paper, we propose SEA, a novel approach for active robot exploration
through semantic map prediction and a reinforcement learning-based hierarchical
exploration policy. Unlike existing learning-based methods that rely on
one-step waypoint prediction, our approach enhances the agent's long-term
environmental understanding to facilitate more efficient exploration. We
propose an iterative prediction-exploration framework that explicitly predicts
the missing areas of the map based on current observations. The difference
between the actual accumulated map and the predicted global map is then used to
guide exploration. Additionally, we design a novel reward mechanism that
leverages reinforcement learning to update the long-term exploration
strategies, enabling us to construct an accurate semantic map within limited
steps. Experimental results demonstrate that our method significantly
outperforms state-of-the-art exploration strategies, achieving superior
coverage ares of the global map within the same time constraints.

### 8. [GRASPLAT: Enabling dexterous grasping through novel view synthesis](http://arxiv.org/pdf/2510.19200v1)

Authors: Matteo Bortolon, Nuno Ferreira Duarte, Plinio Moreno, Fabio Poiesi, José Santos-Victor, Alessio Del Bue

Achieving dexterous robotic grasping with multi-fingered hands remains a
significant challenge. While existing methods rely on complete 3D scans to
predict grasp poses, these approaches face limitations due to the difficulty of
acquiring high-quality 3D data in real-world scenarios. In this paper, we
introduce GRASPLAT, a novel grasping framework that leverages consistent 3D
information while being trained solely on RGB images. Our key insight is that
by synthesizing physically plausible images of a hand grasping an object, we
can regress the corresponding hand joints for a successful grasp. To achieve
this, we utilize 3D Gaussian Splatting to generate high-fidelity novel views of
real hand-object interactions, enabling end-to-end training with RGB data.
Unlike prior methods, our approach incorporates a photometric loss that refines
grasp predictions by minimizing discrepancies between rendered and real images.
We conduct extensive experiments on both synthetic and real-world grasping
datasets, demonstrating that GRASPLAT improves grasp success rates up to 36.9%
over existing image-based methods. Project page:
https://mbortolon97.github.io/grasplat/

### 9. [Background Fades, Foreground Leads: Curriculum-Guided Background Pruning for Efficient Foreground-Centric Collaborative Perception](http://arxiv.org/pdf/2510.19250v1)

Authors: Yuheng Wu, Xiangbo Gao, Quang Tau, Zhengzhong Tu, Dongman Lee

Collaborative perception enhances the reliability and spatial coverage of
autonomous vehicles by sharing complementary information across vehicles,
offering a promising solution to long-tail scenarios that challenge
single-vehicle perception. However, the bandwidth constraints of vehicular
networks make transmitting the entire feature map impractical. Recent methods,
therefore, adopt a foreground-centric paradigm, transmitting only predicted
foreground-region features while discarding the background, which encodes
essential context. We propose FadeLead, a foreground-centric framework that
overcomes this limitation by learning to encapsulate background context into
compact foreground features during training. At the core of our design is a
curricular learning strategy that leverages background cues early on but
progressively prunes them away, forcing the model to internalize context into
foreground representations without transmitting background itself. Extensive
experiments on both simulated and real-world benchmarks show that FadeLead
outperforms prior methods under different bandwidth settings, underscoring the
effectiveness of context-enriched foreground sharing.

### 10. [Hierarchical DLO Routing with Reinforcement Learning and In-Context Vision-language Models](http://arxiv.org/pdf/2510.19268v1)

Authors: Mingen Li, Houjian Yu, Yixuan Huang, Youngjin Hong, Changhyun Choi

Long-horizon routing tasks of deformable linear objects (DLOs), such as
cables and ropes, are common in industrial assembly lines and everyday life.
These tasks are particularly challenging because they require robots to
manipulate DLO with long-horizon planning and reliable skill execution.
Successfully completing such tasks demands adapting to their nonlinear
dynamics, decomposing abstract routing goals, and generating multi-step plans
composed of multiple skills, all of which require accurate high-level reasoning
during execution. In this paper, we propose a fully autonomous hierarchical
framework for solving challenging DLO routing tasks. Given an implicit or
explicit routing goal expressed in language, our framework leverages
vision-language models~(VLMs) for in-context high-level reasoning to synthesize
feasible plans, which are then executed by low-level skills trained via
reinforcement learning. To improve robustness in long horizons, we further
introduce a failure recovery mechanism that reorients the DLO into
insertion-feasible states. Our approach generalizes to diverse scenes involving
object attributes, spatial descriptions, as well as implicit language commands.
It outperforms the next best baseline method by nearly 50% and achieves an
overall success rate of 92.5% across long-horizon routing scenarios.

### Software Engineering

### 1. [Automated Concern Extraction from Textual Requirements of Cyber-Physical Systems: A Multi-solution Study](http://arxiv.org/pdf/2510.19237v1)

Authors: Dongming Jin, Zhi Jin, Xiaohong Chen, Zheng Fang, Linyu Li, Shengxin Zhao, Chuihui Wang, Hongbin Xiao

Cyber-physical systems (CPSs) are characterized by a deep integration of the
information space and the physical world, which makes the extraction of
requirements concerns more challenging. Some automated solutions for
requirements concern extraction have been proposed to alleviate the burden on
requirements engineers. However, evaluating the effectiveness of these
solutions, which relies on fair and comprehensive benchmarks, remains an open
question. To address this gap, we propose ReqEBench, a new CPSs requirements
concern extraction benchmark, which contains 2,721 requirements from 12
real-world CPSs. ReqEBench offers four advantages. It aligns with real-world
CPSs requirements in multiple dimensions, e.g., scale and complexity. It covers
comprehensive concerns related to CPSs requirements. It undergoes a rigorous
annotation process. It covers multiple application domains of CPSs, e.g.,
aerospace and healthcare. We conducted a comparative study on three types of
automated requirements concern extraction solutions and revealed their
performance in real-world CPSs using our ReqEBench. We found that the highest
F1 score of GPT-4 is only 0.24 in entity concern extraction. We further analyze
failure cases of popular LLM-based solutions, summarize their shortcomings, and
provide ideas for improving their capabilities. We believe ReqEBench will
facilitate the evaluation and development of automated requirements concern
extraction.

### 2. [A General Solution for the Implementation of CI/CD in Embedded Linux Development](http://arxiv.org/pdf/2510.19240v1)

Authors: Behnam Agahi, Hamed Farbeh

With the growing use of embedded systems in various industries, the need for
automated platforms for the development and deployment of customized
Linux-based operating systems has become more important. This research was
conducted with the aim of designing and implementing an integrated and
reproducible infrastructure for the development, building, and testing of a
Linux-based operating system using the Yocto Project. The proposed structure
was implemented based on a three-layer architecture consisting of the main
Yocto repositories, a custom layer (meta-custom), and a coordinating manifest
layer to ensure version synchronization, scalability, and reproducibility.
Three sample projects, including libhelloworld, helloworld, and the kernel
module hello mod, were developed and integrated into the build process.
Continuous Integration and Continuous Deployment pipelines were implemented
with GitLab CI and combined with an isolated Docker environment to automate and
streamline the build and testing workflows. Using a local cache server
containing hashserv, downloads and sstate cache significantly reduced the build
time. The functionality and stability of the system were verified through six
boot test scenarios in the QEMU simulator. The results show that the proposed
design not only ensures reproducibility but also can be extended to advanced
applications such as continuous deployment of real-time Linux versions. Future
recommendations include expanding automated tests, implementing system
monitoring with Prometheus and Grafana, using distributed builds, optimizing
with Docker multi-stage builds, and enabling continuous deployment of real-time
Linux changes to provide a stable and scalable model for industrial and
research projects in embedded systems with a rapid and reliable development
cycle.

### 3. [Trace: Securing Smart Contract Repository Against Access Control Vulnerability](http://arxiv.org/pdf/2510.19254v1)

Authors: Chong Chen, Jiachi Chen, Lingfeng Bao, David Lo, Yanlin Wang, Zhenyu Shan, Ting Chen, Guangqiang Yin, Jianxing Yu, Zibin Zheng

Smart contract vulnerabilities, particularly improper Access Control that
allows unauthorized execution of restricted functions, have caused billions of
dollars in losses. GitHub hosts numerous smart contract repositories containing
source code, documentation, and configuration files-these serve as intermediate
development artifacts that must be compiled and packaged before deployment.
Third-party developers often reference, reuse, or fork code from these
repositories during custom development. However, if the referenced code
contains vulnerabilities, it can introduce significant security risks. Existing
tools for detecting smart contract vulnerabilities are limited in their ability
to handle complex repositories, as they typically require the target contract
to be compilable to generate an abstract representation for further analysis.
This paper presents TRACE, a tool designed to secure non-compilable smart
contract repositories against access control vulnerabilities. TRACE employs
LLMs to locate sensitive functions involving critical operations (e.g.,
transfer) within the contract and subsequently completes function snippets into
a fully compilable contract. TRACE constructs a function call graph from the
abstract syntax tree (AST) of the completed contract. It uses the control flow
graph (CFG) of each function as node information. The nodes of the sensitive
functions are then analyzed to detect Access Control vulnerabilities.
Experimental results demonstrate that TRACE outperforms state-of-the-art tools
on an open-sourced CVE dataset, detecting 14 out of 15 CVEs. In addition, it
achieves 89.2% precision on 5,000 recent on-chain contracts, far exceeding the
best existing tool at 76.9%. On 83 real-world repositories, TRACE achieves
87.0% precision, significantly surpassing DeepSeek-R1's 14.3%.

### 4. [From Specification to Service: Accelerating API-First Development Using Multi-Agent Systems](http://arxiv.org/pdf/2510.19274v1)

Authors: Saurabh Chauhan, Zeeshan Rasheed, Malik Abdul Sami, Kai-Kristian Kemell, Muhammad Waseem, Zheying Zhang, Jussi Rasku, Mika Saari, Pekka Abrahamsson

This paper presents a system that uses Large Language Models (LLMs)-based
agents to automate the API-first development of RESTful microservices. This
system helps to create an OpenAPI specification, generate server code from it,
and refine the code through a feedback loop that analyzes execution logs and
error messages. The integration of log analysis enables the LLM to detect and
address issues efficiently, reducing the number of iterations required to
produce functional and robust services. This study's main goal is to advance
API-first development automation for RESTful web services and test the
capability of LLM-based multi-agent systems in supporting the API-first
development approach. To test the proposed system's potential, we utilized the
PRAB benchmark. The results indicate that if we keep the OpenAPI specification
small and focused, LLMs are capable of generating complete functional code with
business logic that aligns to the specification. The code for the system is
publicly available at https://github.com/sirbh/code-gen

### 5. [AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems](http://arxiv.org/pdf/2510.19438v1)

Authors: Linfeng Liang, Chenkai Tan, Yao Deng, Yingfeng Cai, T. Y Chen, Xi Zheng

Autonomous Driving Systems (ADS) are safety-critical, where failures can be
severe. While Metamorphic Testing (MT) is effective for fault detection in ADS,
existing methods rely heavily on manual effort and lack automation. We present
AutoMT, a multi-agent MT framework powered by Large Language Models (LLMs) that
automates the extraction of Metamorphic Relations (MRs) from local traffic
rules and the generation of valid follow-up test cases. AutoMT leverages LLMs
to extract MRs from traffic rules in Gherkin syntax using a predefined
ontology. A vision-language agent analyzes scenarios, and a search agent
retrieves suitable MRs from a RAG-based database to generate follow-up cases
via computer vision. Experiments show that AutoMT achieves up to 5 x higher
test diversity in follow-up case generation compared to the best baseline
(manual expert-defined MRs) in terms of validation rate, and detects up to
20.55% more behavioral violations. While manual MT relies on a fixed set of
predefined rules, AutoMT automatically extracts diverse metamorphic relations
that augment real-world datasets and help uncover corner cases often missed
during in-field testing and data collection. Its modular architecture
separating MR extraction, filtering, and test generation supports integration
into industrial pipelines and potentially enables simulation-based testing to
systematically cover underrepresented or safety-critical scenarios.

### 6. [Mapping and Evolving Interoperability Testing in European Energy Systems: The int:net Perspective](http://arxiv.org/pdf/2510.19460v1)

Authors: Thomas I. Strasser, Edmund Widl, Carlos Ayon Mac Gregor, Mirko Ginocchi, Rene Kuchenbuch

The ongoing transformation of the European energy landscape, driven by the
integration of renewable energy sources, digital technologies, and
decentralized systems, requires a high degree of interoperability across
diverse components and systems. Ensuring that these elements can exchange
information and operate together reliably is essential for achieving a secure,
flexible, and efficient energy supply infrastructure. While several initiatives
have contributed to the development of smart grid testing infrastructures, they
do not provide a dedicated or comprehensive focus on interoperability testing.
A structured and harmonized overview of interoperability testing capabilities
across Europe is therefore still missing. This work therefore presents a novel
contribution by analyzing the European interoperability testing facility
landscape through a structured survey of 30 facilities. It provides a
categorized inventory of testing infrastructures, applied methodologies, and
reference test cases, and introduces a blueprint for the development of future
testing environments. The findings contribute to the establishment of a
coordinated European ecosystem for interoperability testing, supporting
collaboration, innovation, and alignment with the goals of the energy
transition.

### 7. [Review of Tools for Zero-Code LLM Based Application Development](http://arxiv.org/pdf/2510.19747v1)

Authors: Priyaranjan Pattnayak, Hussain Bohra

Large Language Models (LLMs) are transforming software creation by enabling
zero code development platforms. Our survey reviews recent platforms that let
users build applications without writing code, by leveraging LLMs as the brains
of the development process. We adopt a broad survey methodology, categorizing
platforms based on key dimensions such as interface style, backend integration,
output type, and extensibility. We analyze both dedicated LLM based app
builders (OpenAI's custom GPTs, Bolt.new, Dust.tt, Flowise, Cognosys) and
general no code platforms (e.g., Bubble, Glide) that integrate LLM
capabilities. We present a taxonomy categorizing these platforms by their
interface (conversational, visual, etc.), supported LLM backends, output type
(chatbot, full application, workflow), and degree of extensibility. Core
features such as autonomous agents, memory management, workflow orchestration,
and API integrations are in scope of the survey. We provide a detailed
comparison, highlighting each platform's strengths and limitations. Trade offs
(customizability, scalability, vendor lock-in) are discussed in comparison with
traditional and low code development approaches. Finally, we outline future
directions, including multimodal interfaces, on device LLMs, and improved
orchestration for democratizing app creation with AI. Our findings indicate
that while zero code LLM platforms greatly reduce the barrier to creating AI
powered applications, they still face challenges in flexibility and
reliability. Overall, the landscape is rapidly evolving, offering exciting
opportunities to empower non programmers to create sophisticated software.

### 8. [BOSQTGEN: Breaking the Sound Barrier in Test Generation](http://arxiv.org/pdf/2510.19777v1)

Authors: S M Sadrul Islam Asif, James Chen, Earl T. Barr, Mark Marron

Modern software is increasingly built by composing APIs, elevating the API
contract to a critical role. Inadequate contracts, however, lead to mismatched
expectations and failures, creating a pressing need for robust conformance
testing. Current test generation techniques are hindered by key challenges:
polyglot systems, source code inaccessibility, a cost-reliability trade-off,
and, most critically, the difficulty of generating structured inputs.
  We introduce BOSQTGEN, a novel black-box methodology and tool for API test
generation. BOSQTGEN utilizes a novel approach for decomposing API
specifications into primitives, using LLMs to suggest coherent strata for them,
and employing combinatorial testing to efficiently sample over these values.
This approach ensures coverage of critical interactions while avoiding the
redundancy of random sampling.
  The resulting BOSQTGEN system achieves an average of 82% code coverage on
RESTful benchmarks, often a 20% or more increase over prior state-of-the-art
systems and nearing parity with hand-written test suites. Providing a fully
API-driven approach to test generation, enables developers to automatically
create high-quality test cases for validation or test-driven development.

### 9. [Bytecode-centric Detection of Known-to-be-vulnerable Dependencies in Java Projects](http://arxiv.org/pdf/2510.19393v1)

Authors: Stefan Schott, Serena Elisa Ponta, Wolfram Fischer, Jonas Klauke, Eric Bodden

On average, 71% of the code in typical Java projects comes from open-source
software (OSS) dependencies, making OSS dependencies the dominant component of
modern software code bases. This high degree of OSS reliance comes with a
considerable security risk of adding known security vulnerabilities to a code
base. To remedy this risk, researchers and companies have developed various
dependency scanners, which try to identify inclusions of known-to-be-vulnerable
OSS dependencies. However, there are still challenges that modern dependency
scanners do not overcome, especially when it comes to dependency modifications,
such as re-compilations, re-bundlings or re-packagings, which are common in the
Java ecosystem. To overcome these challenges, we present Jaralyzer, a
bytecode-centric dependency scanner for Java. Jaralyzer does not rely on the
metadata or the source code of the included OSS dependencies being available
but directly analyzes a dependency's bytecode. Our evaluation across 56 popular
OSS components demonstrates that Jaralyzer outperforms other popular dependency
scanners in detecting vulnerabilities within modified dependencies. It is the
only scanner capable of identifying vulnerabilities across all the above
mentioned types of modifications. But even when applied to unmodified
dependencies, Jaralyzer outperforms the current state-of-the-art code-centric
scanner Eclipse Steady by detecting 28 more true vulnerabilities and yielding
29 fewer false warnings.

### 10. [A Goal-Driven Survey on Root Cause Analysis](http://arxiv.org/pdf/2510.19593v1)

Authors: Aoyang Fang, Haowen Yang, Haoze Dong, Qisheng Lu, Junjielong Xu, Pinjia He

Root Cause Analysis (RCA) is a crucial aspect of incident management in
large-scale cloud services. While the term root cause analysis or RCA has been
widely used, different studies formulate the task differently. This is because
the term "RCA" implicitly covers tasks with distinct underlying goals. For
instance, the goal of localizing a faulty service for rapid triage is
fundamentally different from identifying a specific functional bug for a
definitive fix. However, previous surveys have largely overlooked these
goal-based distinctions, conventionally categorizing papers by input data types
(e.g., metric-based vs. trace-based methods). This leads to the grouping of
works with disparate objectives, thereby obscuring the true progress and gaps
in the field. Meanwhile, the typical audience of an RCA survey is either laymen
who want to know the goals and big picture of the task or RCA researchers who
want to figure out past research under the same task formulation. Thus, an RCA
survey that organizes the related papers according to their goals is in high
demand. To this end, this paper presents a goal-driven framework that
effectively categorizes and integrates 135 papers on RCA in the context of
cloud incident management based on their diverse goals, spanning the period
from 2014 to 2025. In addition to the goal-driven categorization, it discusses
the ultimate goal of all RCA papers as an umbrella covering different RCA
formulations. Moreover, the paper discusses open challenges and future
directions in RCA.

### Social and Information Networks

### 1. [From Newborn to Impact: Bias-Aware Citation Prediction](http://arxiv.org/pdf/2510.19246v1)

Authors: Mingfei Lu, Mengjia Wu, Jiawei Xu, Weikai Li, Feng Liu, Ying Ding, Yizhou Sun, Jie Lu, Yi Zhang

As a key to accessing research impact, citation dynamics underpins research
evaluation, scholarly recommendation, and the study of knowledge diffusion.
Citation prediction is particularly critical for newborn papers, where early
assessment must be performed without citation signals and under highly
long-tailed distributions. We identify two key research gaps: (i) insufficient
modeling of implicit factors of scientific impact, leading to reliance on
coarse proxies; and (ii) a lack of bias-aware learning that can deliver stable
predictions on lowly cited papers. We address these gaps by proposing a
Bias-Aware Citation Prediction Framework, which combines multi-agent feature
extraction with robust graph representation learning. First, a multi-agent x
graph co-learning module derives fine-grained, interpretable signals, such as
reproducibility, collaboration network, and text quality, from metadata and
external resources, and fuses them with heterogeneous-network embeddings to
provide rich supervision even in the absence of early citation signals. Second,
we incorporate a set of robust mechanisms: a two-stage forward process that
routes explicit factors through an intermediate exposure estimate, GroupDRO to
optimize worst-case group risk across environments, and a regularization head
that performs what-if analyses on controllable factors under monotonicity and
smoothness constraints. Comprehensive experiments on two real-world datasets
demonstrate the effectiveness of our proposed model. Specifically, our model
achieves around a 13% reduction in error metrics (MALE and RMSLE) and a notable
5.5% improvement in the ranking metric (NDCG) over the baseline methods.

### 2. [From Substitution to Complement? Uncovering the Evolving Interplay between Ride-hailing Services and Public Transit](http://arxiv.org/pdf/2510.19745v1)

Authors: Zhicheng Jin, Xiaotong Sun, Li Zhen, Weihua Gu, Huizhao Tu

The literature on transportation network companies (TNCs), also known as
ride-hailing services, has often characterized these service providers as
predominantly substitutive to public transit (PT). However, as TNC markets
expand and mature, the complementary and substitutive relationships with PT may
shift. To explore whether such a transformation is occurring, this study
collected travel data from 96,716 ride-hailing vehicles during September 2022
in Shanghai, a city characterized by an increasingly saturated TNC market. An
enhanced data-driven framework is proposed to classify TNC-PT relationships
into four types: first-mile complementary, last-mile complementary,
substitutive, and independent. Our findings indicate comparable ratios of
complementary trips (9.22%) and substitutive trips (9.06%), contrasting sharply
with the findings of prior studies. Furthermore, to examine the nonlinear
impact of various influential factors on these ratios, a machine learning
method integrating categorical boosting (CatBoost) and Shapley additive
explanations (SHAP) is proposed. The results show significant nonlinear effects
in some variables, including the distance to the nearest metro station and the
density of bus stops. Moreover, metro hubs and regular single-line stations
exhibit distinct effects on first- or last-mile complementary ratios. These
ratios' relation to the distance to single-line stations shows an inverted
U-shaped pattern, with effects rising sharply within 1.5 km, remaining at the
peak between 1.5 and 3 km, and then declining as the distance increases to
about 15 km.

### 3. [Belief propagation for finite networks using a symmetry-breaking source node](http://arxiv.org/pdf/2510.19231v1)

Authors: Seongmin Kim, Alec Kirkley

Belief Propagation (BP) is an efficient message-passing algorithm widely used
for inference in graphical models and for solving various problems in
statistical physics. However, BP often yields inaccurate estimates of order
parameters and their susceptibilities in finite systems, particularly in sparse
networks with few loops. Here, we show for both percolation and Ising models
that fixing the state of a single well-connected "source" node to break global
symmetry substantially improves inference accuracy and captures finite-size
effects across a broad range of networks, especially tree-like ones, at no
additional computational cost.

### 4. [Unfair Mistakes on Social Media: How Demographic Characteristics influence Authorship Attribution](http://arxiv.org/pdf/2510.19708v1)

Authors: Jasmin Wyss, Rebekah Overdorf

Authorship attribution techniques are increasingly being used in online
contexts such as sock puppet detection, malicious account linking, and
cross-platform account linking. Yet, it is unknown whether these models perform
equitably across different demographic groups. Bias in such techniques could
lead to false accusations, account banning, and privacy violations
disproportionately impacting users from certain demographics. In this paper, we
systematically audit authorship attribution for bias with respect to gender,
native language, and age. We evaluate fairness in 3 ways. First, we evaluate
how the proportion of users with a certain demographic characteristic impacts
the overall classifier performance. Second, we evaluate if a user's demographic
characteristics influence the probability that their texts are misclassified.
Our analysis indicates that authorship attribution does not demonstrate bias
across demographic groups in the closed-world setting. Third, we evaluate the
types of errors that occur when the true author is removed from the suspect
set, thereby forcing the classifier to choose an incorrect author. Unlike the
first two settings, this analysis demonstrates a tendency to attribute
authorship to users who share the same demographic characteristic as the true
author. Crucially, these errors do not only include texts that deviate from a
user's usual style, but also those that are very close to the author's average.
Our results highlight that though a model may appear fair in the closed-world
setting for a performant classifier, this does not guarantee fairness when
errors are inevitable.

### 5. [Learning to Make Friends: Coaching LLM Agents toward Emergent Social Ties](http://arxiv.org/pdf/2510.19299v1)

Authors: Philipp J. Schneider, Lin Tian, Marian-Andrei Rizoiu

Can large language model (LLM) agents reproduce the complex social dynamics
that characterize human online behavior -- shaped by homophily, reciprocity,
and social validation -- and what memory and learning mechanisms enable such
dynamics to emerge? We present a multi-agent LLM simulation framework in which
agents repeatedly interact, evaluate one another, and adapt their behavior
through in-context learning accelerated by a coaching signal. To model human
social behavior, we design behavioral reward functions that capture core
drivers of online engagement, including social interaction, information
seeking, self-presentation, coordination, and emotional support. These rewards
align agent objectives with empirically observed user motivations, enabling the
study of how network structures and group formations emerge from individual
decision-making. Our experiments show that coached LLM agents develop stable
interaction patterns and form emergent social ties, yielding network structures
that mirror properties of real online communities. By combining behavioral
rewards with in-context adaptation, our framework establishes a principled
testbed for investigating collective dynamics in LLM populations and reveals
how artificial agents may approximate or diverge from human-like social
behavior.

### Systems and Control

### 1. [Spatiotemporal Tubes based Control of Unknown Multi-Agent Systems for Temporal Reach-Avoid-Stay Tasks](http://arxiv.org/pdf/2510.19232v1)

Authors: Ahan Basu, Ratnangshu Das, Pushpak Jagtap

The paper focuses on designing a controller for unknown dynamical multi-agent
systems to achieve temporal reach-avoid-stay tasks for each agent while
preventing inter-agent collisions. The main objective is to generate a
spatiotemporal tube (STT) for each agent and thereby devise a closed-form,
approximation-free, and decentralized control strategy that ensures the system
trajectory reaches the target within a specific time while avoiding
time-varying unsafe sets and collisions with other agents. In order to achieve
this, the requirements of STTs are formulated as a robust optimization problem
(ROP) and solved using a sampling-based scenario optimization problem (SOP) to
address the issue of infeasibility caused by the infinite number of constraints
in ROP. The STTs are generated by solving the SOP, and the corresponding
closed-form control is designed to fulfill the specified task. Finally, the
effectiveness of our approach is demonstrated through two case studies, one
involving omnidirectional robots and the other involving multiple drones
modelled as Euler-Lagrange systems.

### 2. [Managing Charging Induced Grid Stress and Battery Degradation in Electric Taxi Fleets](http://arxiv.org/pdf/2510.19293v1)

Authors: Michael Yuhas, Rajesh K. Ahir, Laksamana Vixell Tanjaya Hartono, Muhammad Dzaki Dwi Putranto, Arvind Easwaran, Suhono Harso Supangkat

Operating fleets of electric vehicles (EVs) introduces several challenges,
some of which are borne by the fleet operator, and some of which are borne by
the power grid. To maximize short-term profit a fleet operator could always
charge EVs at the maximum rate to ensure vehicles are ready to service ride
demand. However, due to the stochastic nature of electricity demand, charging
EVs at their maximum rate may potentially increase the grid stress and lead to
overall instability. Furthermore, high-rate charging of EVs can accelerate
battery degradation, thereby reducing the service lifespan of the fleet. This
study aims to reconcile the conflicting incentives of fleet longevity,
short-term profitability, and grid stability by simulating a taxi fleet
throughout its lifespan in relation to its charging policies and service
conditions. We develop an EV fleet simulator to evaluate the battery
degradation due to unpredictable charging and ride demand. Consequently, the
impact on the power grid through the charging infrastructure is assessed due to
these activities. This simulation utilizes publicly accessible real-world
travel data from the NYC taxi dataset. We compare a baseline 80-20 fleet
charging policy with a reinforcement learning-based policy designed to prolong
the fleet's service life and alleviate grid stress. We monitor grid stress,
battery degradation, and profitability over five years and find that our
learned policy outperforms the baseline. This simulator enables fleet operators
to assess the impact of different charging policies on these indicators to make
informed decisions in the future.

### 3. [Multi-UAV Flood Monitoring via CVT with Gaussian Mixture of Density Functions for Coverage Control](http://arxiv.org/pdf/2510.19548v1)

Authors: Jie Song, Yang Bai, Mikhail Svinin, Naoki Wakamiya

This study presents a control strategy for coordinating multiple unmanned
aerial vehicles (UAVs) to monitor unknown flood regions and estimate the extent
of inundation. The proposed method adopts a density-driven coverage framework
based on Centroidal Voronoi Tessellation (CVT), in which the density function
is modeled using a Gaussian Mixture of Density Functions (GMDF). This
formulation provides a more accurate characterization of inundated areas
compared to conventional axis-aligned Gaussian models. The performance of the
two density modeling approaches is systematically evaluated under different UAV
fleet sizes (16, 20, and 24), with multiple simulation trials conducted in the
ROS/Gazebo environment. The results show that the GMDF-based formulation
consistently achieves higher coverage rates, demonstrating its effectiveness in
enhancing flood monitoring and improving UAV spatial distribution.

### 4. [Control Barrier Functions for the Full Class of Signal Temporal Logic Tasks using Spatiotemporal Tubes](http://arxiv.org/pdf/2510.19595v1)

Authors: Ratnangshu Das, Subhodeep Choudhury, Pushpak Jagtap

This paper introduces a new framework for synthesizing time-varying control
barrier functions (TV-CBFs) for general Signal Temporal Logic (STL)
specifications using spatiotemporal tubes (STT). We first formulate the STT
synthesis as a robust optimization problem (ROP) and solve it through a
scenario optimization problem (SOP), providing formal guarantees that the
resulting tubes capture the given STL specifications. These STTs are then used
to construct TV-CBFs, ensuring that under any control law rendering them
invariant, the system satisfies the STL tasks. We demonstrate the framework
through case studies on a differential-drive mobile robot and a quadrotor, and
provide a comparative analysis showing improved efficiency over existing
approaches.

### 5. [Optimal Kron-based Reduction of Networks (Opti-KRON) for Three-phase Distribution Feeders](http://arxiv.org/pdf/2510.19608v1)

Authors: Omid Mokhtari, Samuel Chevalier, Mads Almassalkhi

This paper presents a novel structure-preserving, Kron-based reduction
framework for unbalanced distribution feeders. The method aggregates
electrically similar nodes within a mixed-integer optimization (MIP) problem to
produce reduced networks that optimally reproduce the voltage profiles of the
original full network. To overcome computational bottlenecks of MIP
formulations, we propose an exhaustive-search formulation to identify optimal
aggregation decisions while enforcing voltage margin limits. The proposed
exhaustive network reduction algorithm is parallelizable on GPUs, which enables
scalable network reduction. The resulting reduced networks approximate the full
system's voltage profiles with low errors and are suitable for steady-state
analysis and optimal power flow studies. The framework is validated on two real
utility distribution feeders with 5,991 and 8,381 nodes. The reduced models
achieve up to 90% and 80% network reduction, respectively, while the maximum
voltage-magnitude error remains below 0.003 p.u. Furthermore, on a 1000-node
version of the network, the GPU-accelerated reduction algorithm runs up to 15x
faster than its CPU-based counterpart.

### 6. [Policy Gradient Method for LQG Control via Input-Output-History Representation: Convergence to $O(ε)$-Stationary Points](http://arxiv.org/pdf/2510.19141v1)

Authors: Tomonori Sadamoto, Takashi Tanaka

We study the policy gradient method (PGM) for the linear quadratic Gaussian
(LQG) dynamic output-feedback control problem using an input-output-history
(IOH) representation of the closed-loop system. First, we show that any dynamic
output-feedback controller is equivalent to a static partial-state feedback
gain for a new system representation characterized by a finite-length IOH.
Leveraging this equivalence, we reformulate the search for an optimal dynamic
output feedback controller as an optimization problem over the corresponding
partial-state feedback gain. Next, we introduce a relaxed version of the
IOH-based LQG problem by incorporating a small process noise with covariance
$\epsilon I$ into the new system to ensure coerciveness, a key condition for
establishing gradient-based convergence guarantees. Consequently, we show that
a vanilla PGM for the relaxed problem converges to an
$\mathcal{O}(\epsilon)$-stationary point, i.e., $\overline{K}$ satisfying
$\|\nabla J(\overline{K})\|_F \leq \mathcal{O}(\epsilon)$, where $J$ denotes
the original LQG cost. Numerical experiments empirically indicate convergence
to the vicinity of the globally optimal LQG controller.

### 7. [Query-Efficient Zeroth-Order Algorithms for Nonconvex Optimization](http://arxiv.org/pdf/2510.19165v1)

Authors: Ruiyang Jin, Yuke Zhou, Yujie Tang, Jie Song, Siyang Gao

Zeroth-order optimization (ZO) has been a powerful framework for solving
black-box problems, which estimates gradients using zeroth-order data to update
variables iteratively. The practical applicability of ZO critically depends on
the efficiency of single-step gradient estimation and the overall query
complexity. However, existing ZO algorithms cannot achieve efficiency on both
simultaneously. In this work, we consider a general constrained optimization
model with black-box objective and constraint functions. To solve it, we
propose novel algorithms that can achieve the state-of-the-art overall query
complexity bound of $\mathcal{O}(d/\epsilon^4)$ to find an
$\epsilon$-stationary solution ($d$ is the dimension of variable space), while
reducing the queries for estimating a single-step gradient from
$\mathcal{O}(d)$ to $\mathcal{O}(1)$. Specifically, we integrate block updates
with gradient descent ascent and a block gradient estimator, which leads to two
algorithms, ZOB-GDA and ZOB-SGDA, respectively. Instead of constructing full
gradients, they estimate only partial gradients along random blocks of
dimensions, where the adjustable block sizes enable high single-step efficiency
without sacrificing convergence guarantees. Our theoretical results establish
the finite-sample convergence of the proposed algorithms for nonconvex
optimization. Finally, numerical experiments on a practical problem demonstrate
that our algorithms require over ten times fewer queries than existing methods.

### 8. [Magnetic field estimation using Gaussian process regression for interactive wireless power system design](http://arxiv.org/pdf/2510.19277v1)

Authors: Yuichi Honjo, Cedric Caremel, Ken Takaki, Yuta Noma, Yoshihiro Kawahara, Takuya Sasatani

Wireless power transfer (WPT) with coupled resonators offers a promising
solution for the seamless powering of electronic devices. Interactive design
approaches that visualize the magnetic field and power transfer efficiency
based on system geometry adjustments can facilitate the understanding and
exploration of the behavior of these systems for dynamic applications. However,
typical electromagnetic field simulation methods, such as the Method of Moments
(MoM), require significant computational resources, limiting the rate at which
computation can be performed for acceptable interactivity. Furthermore, the
system's sensitivity to positional and geometrical changes necessitates a large
number of simulations, and structures such as ferromagnetic shields further
complicate these simulations. Here, we introduce a machine learning approach
using Gaussian Process Regression (GPR), demonstrating for the first time the
rapid estimation of the entire magnetic field and power transfer efficiency for
near-field coupled systems. To achieve quick and accurate estimation, we
develop 3D adaptive grid systems and an active learning strategy to effectively
capture the nonlinear interactions between complex system geometries and
magnetic fields. By training a regression model, our approach achieves magnetic
field computation with sub-second latency and with an average error of less
than 6% when validated against independent electromagnetic simulation results.

### 9. [Risk Assessment of an Autonomous Underwater Snake Robot in Confined Operations](http://arxiv.org/pdf/2510.19415v1)

Authors: Abdelrahman Sayed Sayed

The growing interest in ocean discovery imposes a need for inspection and
intervention in confined and demanding environments. Eely's slender shape, in
addition to its ability to change its body configurations, makes articulated
underwater robots an adequate option for such environments. However, operation
of Eely in such environments imposes demanding requirements on the system, as
it must deal with uncertain and unstructured environments, extreme
environmental conditions, and reduced navigational capabilities. This paper
proposes a Bayesian approach to assess the risks of losing Eely during two
mission scenarios. The goal of this work is to improve Eely's performance and
the likelihood of mission success. Sensitivity analysis results are presented
in order to demonstrate the causes having the highest impact on losing Eely.

### 10. [Bridging Earth and Space: A Survey on HAPS for Non-Terrestrial Networks](http://arxiv.org/pdf/2510.19731v1)

Authors: G. Svistunov, A. Akhtarshenas, D. López-Pérez, M. Giordani, G. Geraci, H. Yanikomeroglu

HAPS are emerging as key enablers in the evolution of 6G wireless networks,
bridging terrestrial and non-terrestrial infrastructures. Operating in the
stratosphere, HAPS can provide wide-area coverage, low-latency,
energy-efficient broadband communications with flexible deployment options for
diverse applications. This survey delivers a comprehensive overview of HAPS use
cases, technologies, and integration strategies within the 6G ecosystem. The
roles of HAPS in extending connectivity to underserved regions, supporting
dynamic backhauling, enabling massive IoT, and delivering reliable low-latency
communications for autonomous and immersive services are discussed. The paper
reviews state-of-the-art architectures for terrestrial and non-terrestrial
network integration, highlights recent field trials. Furthermore, key enabling
technologies such as channel modeling, AI-driven resource allocation,
interference control, mobility management, and energy-efficient communications
are examined. The paper also outlines open research challenges. By addressing
existing gaps in the literature, this survey positions HAPS as a foundational
component of globally integrated, resilient, and sustainable 6G networks.

### Machine Learning (Statistics Category)

### 1. [Scalable LinUCB: Low-Rank Design Matrix Updates for Recommenders with Large Action Spaces](http://arxiv.org/pdf/2510.19349v1)

Authors: Evgenia Shustova, Marina Sheshukova, Sergey Samsonov, Evgeny Frolov

Linear contextual bandits, especially LinUCB, are widely used in recommender
systems. However, its training, inference, and memory costs grow with feature
dimensionality and the size of the action space. The key bottleneck becomes the
need to update, invert and store a design matrix that absorbs contextual
information from interaction history. In this paper, we introduce Scalable
LinUCB, the algorithm that enables fast and memory efficient operations with
the inverse regularized design matrix. We achieve this through a dynamical
low-rank parametrization of its inverse Cholesky-style factors. We derive
numerically stable rank-1 and batched updates that maintain the inverse without
directly forming the entire matrix. To control memory growth, we employ a
projector-splitting integrator for dynamical low-rank approximation, yielding
average per-step update cost $O(dr)$ and memory $O(dr)$ for approximation rank
$r$. Inference complexity of the suggested algorithm is $O(dr)$ per action
evaluation. Experiments on recommender system datasets demonstrate the
effectiveness of our algorithm.

### 2. [On the hardness of RL with Lookahead](http://arxiv.org/pdf/2510.19372v1)

Authors: Corentin Pla, Hugo Richard, Marc Abeille, Nadav Merlis, Vianney Perchet

We study reinforcement learning (RL) with transition look-ahead, where the
agent may observe which states would be visited upon playing any sequence of
$\ell$ actions before deciding its course of action. While such predictive
information can drastically improve the achievable performance, we show that
using this information optimally comes at a potentially prohibitive
computational cost. Specifically, we prove that optimal planning with one-step
look-ahead ($\ell=1$) can be solved in polynomial time through a novel linear
programming formulation. In contrast, for $\ell \geq 2$, the problem becomes
NP-hard. Our results delineate a precise boundary between tractable and
intractable cases for the problem of planning with transition look-ahead in
reinforcement learning.

### 3. [Square root Cox's survival analysis by the fittest linear and neural networks model](http://arxiv.org/pdf/2510.19374v1)

Authors: Maxime van Cutsem, Sylvain Sardy

We revisit Cox's proportional hazard models and LASSO in the aim of improving
feature selection in survival analysis. Unlike traditional methods relying on
cross-validation or BIC, the penalty parameter $\lambda$ is directly tuned for
feature selection and is asymptotically pivotal thanks to taking the square
root of Cox's partial likelihood. Substantially improving over both
cross-validation LASSO and BIC subset selection, our approach has a phase
transition on the probability of retrieving all and only the good features,
like in compressed sensing. The method can be employed by linear models but
also by artificial neural networks.

### 4. [A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond](http://arxiv.org/pdf/2510.19382v1)

Authors: Nikos Tsikouras, Yorgos Pantis, Ioannis Mitliagkas, Christos Tzamos

Understanding the dynamics of feature learning in neural networks (NNs)
remains a significant challenge. The work of (Mousavi-Hosseini et al., 2023)
analyzes a multiple index teacher-student setting and shows that a two-layer
student attains a low-rank structure in its first-layer weights when trained
with stochastic gradient descent (SGD) and a strong regularizer. This
structural property is known to reduce sample complexity of generalization.
Indeed, in a second step, the same authors establish algorithm-specific
learning guarantees under additional assumptions. In this paper, we focus
exclusively on the structure discovery aspect and study it under weaker
assumptions, more specifically: we allow (a) NNs of arbitrary size and depth,
(b) with all parameters trainable, (c) under any smooth loss function, (d) tiny
regularization, and (e) trained by any method that attains a second-order
stationary point (SOSP), e.g.\ perturbed gradient descent (PGD). At the core of
our approach is a key $\textit{derandomization}$ lemma, which states that
optimizing the function $\mathbb{E}_{\mathbf{x}}
\left[g_{\theta}(\mathbf{W}\mathbf{x} + \mathbf{b})\right]$ converges to a
point where $\mathbf{W} = \mathbf{0}$, under mild conditions. The fundamental
nature of this lemma directly explains structure discovery and has immediate
applications in other domains including an end-to-end approximation for MAXCUT,
and computing Johnson-Lindenstrauss embeddings.

### 5. [Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](http://arxiv.org/pdf/2510.19591v1)

Authors: Marius Potfer, Vianney Perchet

Repeated multi-unit auctions, where a seller allocates multiple identical
items over many rounds, are common mechanisms in electricity markets and
treasury auctions. We compare the two predominant formats: uniform-price and
discriminatory auctions, focusing on the perspective of a single bidder
learning to bid against stochastic adversaries. We characterize the learning
difficulty in each format, showing that the regret scales similarly for both
auction formats under both full-information and bandit feedback, as
$\tilde{\Theta} ( \sqrt{T} )$ and $\tilde{\Theta} ( T^{2/3} )$, respectively.
However, analysis beyond worst-case regret reveals structural differences:
uniform-price auctions may admit faster learning rates, with regret scaling as
$\tilde{\Theta} ( \sqrt{T} )$ in settings where discriminatory auctions remain
at $\tilde{\Theta} ( T^{2/3} )$. Finally, we provide a specific analysis for
auctions in which the other participants are symmetric and have unit-demand,
and show that in these instances, a similar regret rate separation appears.

### 6. [Shrinkage to Infinity: Reducing Test Error by Inflating the Minimum Norm Interpolator in Linear Models](http://arxiv.org/pdf/2510.19206v1)

Authors: Jake Freeman

Hastie et al. (2022) found that ridge regularization is essential in high
dimensional linear regression $y=\beta^Tx + \epsilon$ with isotropic
co-variates $x\in \mathbb{R}^d$ and $n$ samples at fixed $d/n$. However, Hastie
et al. (2022) also notes that when the co-variates are anisotropic and $\beta$
is aligned with the top eigenvalues of population covariance, the "situation is
qualitatively different." In the present article, we make precise this
observation for linear regression with highly anisotropic covariances and
diverging $d/n$. We find that simply scaling up (or inflating) the minimum
$\ell_2$ norm interpolator by a constant greater than one can improve the
generalization error. This is in sharp contrast to traditional
regularization/shrinkage prescriptions. Moreover, we use a data-splitting
technique to produce consistent estimators that achieve generalization error
comparable to that of the optimally inflated minimum-norm interpolator. Our
proof relies on apparently novel matching upper and lower bounds for
expectations of Gaussian random projections for a general class of anisotropic
covariance matrices when $d/n\to \infty$.

### 7. [Error Analysis of Triangular Optimal Transport Maps for Filtering](http://arxiv.org/pdf/2510.19283v1)

Authors: Mohammad Al-Jarrah, Bamdad Hosseini, Niyizhen Jin, Michele Martino, Amirhossein Taghvaei

We present a systematic analysis of estimation errors for a class of optimal
transport based algorithms for filtering and data assimilation. Along the way,
we extend previous error analyses of Brenier maps to the case of conditional
Brenier maps that arise in the context of simulation based inference. We then
apply these results in a filtering scenario to analyze the optimal transport
filtering algorithm of Al-Jarrah et al. (2024, ICML). An extension of that
algorithm along with numerical benchmarks on various non-Gaussian and
high-dimensional examples are provided to demonstrate its effectiveness and
practical potential.

### 8. [Metadata Extraction Leveraging Large Language Models](http://arxiv.org/pdf/2510.19334v1)

Authors: Cuize Han, Sesh Jalagam

The advent of Large Language Models has revolutionized tasks across domains,
including the automation of legal document analysis, a critical component of
modern contract management systems. This paper presents a comprehensive
implementation of LLM-enhanced metadata extraction for contract review,
focusing on the automatic detection and annotation of salient legal clauses.
Leveraging both the publicly available Contract Understanding Atticus Dataset
(CUAD) and proprietary contract datasets, our work demonstrates the integration
of advanced LLM methodologies with practical applications. We identify three
pivotal elements for optimizing metadata extraction: robust text conversion,
strategic chunk selection, and advanced LLM-specific techniques, including
Chain of Thought (CoT) prompting and structured tool calling. The results from
our experiments highlight the substantial improvements in clause identification
accuracy and efficiency. Our approach shows promise in reducing the time and
cost associated with contract review while maintaining high accuracy in legal
clause identification. The results suggest that carefully optimized LLM systems
could serve as valuable tools for legal professionals, potentially increasing
access to efficient contract review services for organizations of all sizes.

### 9. [Learning Upper Lower Value Envelopes to Shape Online RL: A Principled Approach](http://arxiv.org/pdf/2510.19528v1)

Authors: Sebastian Reboul, Hélène Halconruy, Randal Douc

We investigate the fundamental problem of leveraging offline data to
accelerate online reinforcement learning - a direction with strong potential
but limited theoretical grounding. Our study centers on how to learn and apply
value envelopes within this context. To this end, we introduce a principled
two-stage framework: the first stage uses offline data to derive upper and
lower bounds on value functions, while the second incorporates these learned
bounds into online algorithms. Our method extends prior work by decoupling the
upper and lower bounds, enabling more flexible and tighter approximations. In
contrast to approaches that rely on fixed shaping functions, our envelopes are
data-driven and explicitly modeled as random variables, with a filtration
argument ensuring independence across phases. The analysis establishes
high-probability regret bounds determined by two interpretable quantities,
thereby providing a formal bridge between offline pre-training and online
fine-tuning. Empirical results on tabular MDPs demonstrate substantial regret
reductions compared with both UCBVI and prior methods.

### 10. [Policy Learning with Abstention](http://arxiv.org/pdf/2510.19672v1)

Authors: Ayush Sawarni, Jikai Jin, Justin Whitehouse, Vasilis Syrgkanis

Policy learning algorithms are widely used in areas such as personalized
medicine and advertising to develop individualized treatment regimes. However,
most methods force a decision even when predictions are uncertain, which is
risky in high-stakes settings. We study policy learning with abstention, where
a policy may defer to a safe default or an expert. When a policy abstains, it
receives a small additive reward on top of the value of a random guess. We
propose a two-stage learner that first identifies a set of near-optimal
policies and then constructs an abstention rule from their disagreements. We
establish fast O(1/n)-type regret guarantees when propensities are known, and
extend these guarantees to the unknown-propensity case via a doubly robust (DR)
objective. We further show that abstention is a versatile tool with direct
applications to other core problems in policy learning: it yields improved
guarantees under margin conditions without the common realizability assumption,
connects to distributionally robust policy learning by hedging against small
data shifts, and supports safe policy improvement by ensuring improvement over
a baseline policy with high probability.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-23 PST.

### 1. [Optimizing YOLOv11 for automated classification of breast cancer in medical images](https://www.nature.com/articles/s41598-025-24850-7)

Authors: Tarek Abd El-Hafeez et al.

### 2. [Enhancing pumping unit diagnosis with similarity splicing data augmentation and wavelet denoising](https://www.nature.com/articles/s41598-025-20819-8)

Authors: Xinlong Tan et al.

### 3. [A forest fire identification and monitoring model based on improved YOLOv8](https://www.nature.com/articles/s41598-025-17893-3)

Authors: Yunchang Zheng et al.

### 4. [Quantum-resilient and adaptive multi-region data aggregation for IoMT using zero-knowledge proofs and edge intelligence](https://www.nature.com/articles/s41598-025-22457-6)

Authors: Soufiane Ben Othman et al.

### 5. [A high precision and speed question answering system about the post-COVID-19](https://www.nature.com/articles/s41598-025-20088-5)

Authors: Ziang Zheng

### 6. [Automating wastewater characteristic parameter quantitation using neural architecture search in AutoML systems on spectral reflectance data](https://www.nature.com/articles/s41598-025-21069-4)

Authors: Shilpa Ankalaki

