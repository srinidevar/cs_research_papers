# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-16 18:12:52.504989 PST.

### Artificial Intelligence

### 1. ["There Is No Such Thing as a Dumb Question," But There Are Good Ones](http://arxiv.org/pdf/2505.09923v1)

Authors: Minjung Shin, Donghyun Kim, Jeh-Kwang Ryu

Questioning has become increasingly crucial for both humans and artificial
intelligence, yet there remains limited research comprehensively assessing
question quality. In response, this study defines good questions and presents a
systematic evaluation framework. We propose two key evaluation dimensions:
appropriateness (sociolinguistic competence in context) and effectiveness
(strategic competence in goal achievement). Based on these foundational
dimensions, a rubric-based scoring system was developed. By incorporating
dynamic contextual variables, our evaluation framework achieves structure and
flexibility through semi-adaptive criteria. The methodology was validated using
the CAUS and SQUARE datasets, demonstrating the ability of the framework to
access both well-formed and problematic questions while adapting to varied
contexts. As we establish a flexible and comprehensive framework for question
evaluation, this study takes a significant step toward integrating questioning
behavior with structured analytical methods grounded in the intrinsic nature of
questioning.

### 2. [Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents](http://arxiv.org/pdf/2505.09970v1)

Authors: Mrinal Rawat, Ambuje Gupta, Rushil Goomer, Alessandro Di Bari, Neha Gupta, Roberto Pieraccini

The ReAct (Reasoning + Action) capability in large language models (LLMs) has
become the foundation of modern agentic systems. Recent LLMs, such as
DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through
the generation of ample intermediate tokens, which help build a strong premise
before producing the final output tokens. In this paper, we introduce Pre-Act,
a novel approach that enhances the agent's performance by creating a multi-step
execution plan along with the detailed reasoning for the given user input. This
plan incrementally incorporates previous steps and tool outputs, refining
itself after each step execution until the final response is obtained. Our
approach is applicable to both conversational and non-conversational agents. To
measure the performance of task-oriented agents comprehensively, we propose a
two-level evaluation framework: (1) turn level and (2) end-to-end. Our
turn-level evaluation, averaged across five models, shows that our approach,
Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While
this approach is effective for larger models, smaller models crucial for
practical applications, where latency and cost are key constraints, often
struggle with complex reasoning tasks required for agentic systems. To address
this limitation, we fine-tune relatively small models such as Llama 3.1 (8B &
70B) using the proposed Pre-Act approach. Our experiments show that the
fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action
accuracy (turn-level) and a 28% improvement in goal completion rate
(end-to-end) on the Almita (out-of-domain) dataset.

### 3. [The First MPDD Challenge: Multimodal Personality-aware Depression Detection](http://arxiv.org/pdf/2505.10034v1)

Authors: Changzeng Fu, Zelin Fu, Xinhe Kuang, Jiacheng Dong, Qi Zhang, Kaifeng Su, Yikai Su, Wenbo Shi, Junfeng Yao, Yuliang Zhao, Shiqi Zhao, Jiadong Wang, Siyang Song, Chaoran Liu, Yuichiro Yoshikawa, Björn Schuller, Hiroshi Ishiguro

Depression is a widespread mental health issue affecting diverse age groups,
with notable prevalence among college students and the elderly. However,
existing datasets and detection methods primarily focus on young adults,
neglecting the broader age spectrum and individual differences that influence
depression manifestation. Current approaches often establish a direct mapping
between multimodal data and depression indicators, failing to capture the
complexity and diversity of depression across individuals. This challenge
includes two tracks based on age-specific subsets: Track 1 uses the
MPDD-Elderly dataset for detecting depression in older adults, and Track 2 uses
the MPDD-Young dataset for detecting depression in younger participants. The
Multimodal Personality-aware Depression Detection (MPDD) Challenge aims to
address this gap by incorporating multimodal data alongside individual
difference factors. We provide a baseline model that fuses audio and video
modalities with individual difference information to detect depression
manifestations in diverse populations. This challenge aims to promote the
development of more personalized and accurate de pression detection methods,
advancing mental health research and fostering inclusive detection systems.
More details are available on the official challenge website:
https://hacilab.github.io/MPDDChallenge.github.io.

### 4. [A User Study Evaluating Argumentative Explanations in Diagnostic Decision Support](http://arxiv.org/pdf/2505.10188v1)

Authors: Felix Liedeker, Olivia Sanchez-Graillet, Moana Seidler, Christian Brandt, Jörg Wellmer, Philipp Cimiano

As the field of healthcare increasingly adopts artificial intelligence, it
becomes important to understand which types of explanations increase
transparency and empower users to develop confidence and trust in the
predictions made by machine learning (ML) systems. In shared decision-making
scenarios where doctors cooperate with ML systems to reach an appropriate
decision, establishing mutual trust is crucial. In this paper, we explore
different approaches to generating explanations in eXplainable AI (XAI) and
make their underlying arguments explicit so that they can be evaluated by
medical experts. In particular, we present the findings of a user study
conducted with physicians to investigate their perceptions of various types of
AI-generated explanations in the context of diagnostic decision support. The
study aims to identify the most effective and useful explanations that enhance
the diagnostic process. In the study, medical doctors filled out a survey to
assess different types of explanations. Further, an interview was carried out
post-survey to gain qualitative insights on the requirements of explanations
incorporated in diagnostic decision support. Overall, the insights gained from
this study contribute to understanding the types of explanations that are most
effective.

### 5. [MASS: Multi-Agent Simulation Scaling for Portfolio Construction](http://arxiv.org/pdf/2505.10278v1)

Authors: Taian Guo, Haiyang Shen, Jinsheng Huang, Zhengyang Mao, Junyu Luo, Zhuoru Chen, Xuhui Liu, Bingyu Xia, Luchen Liu, Yun Ma, Ming Zhang

LLM-based multi-agent has gained significant attention for their potential in
simulation and enhancing performance. However, existing works are limited to
pure simulations or are constrained by predefined workflows, restricting their
applicability and effectiveness. In this paper, we introduce the Multi-Agent
Scaling Simulation (MASS) for portfolio construction. MASS achieves stable and
continuous excess returns by progressively increasing the number of agents for
large-scale simulations to gain a superior understanding of the market and
optimizing agent distribution end-to-end through a reverse optimization
process, rather than relying on a fixed workflow. We demonstrate its
superiority through performance experiments, ablation studies, backtesting
experiments, experiments on updated data and stock pools, scaling experiments,
parameter sensitivity experiments, and visualization experiments, conducted in
comparison with 6 state-of-the-art baselines on 3 challenging A-share stock
pools. We expect the paradigm established by MASS to expand to other tasks with
similar characteristics. The implementation of MASS has been open-sourced at
https://github.com/gta0804/MASS.

### 6. [AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenge](http://arxiv.org/pdf/2505.10468v1)

Authors: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee

This study critically distinguishes between AI Agents and Agentic AI,
offering a structured conceptual taxonomy, application mapping, and challenge
analysis to clarify their divergent design philosophies and capabilities. We
begin by outlining the search strategy and foundational definitions,
characterizing AI Agents as modular systems driven by Large Language Models
(LLMs) and Large Image Models (LIMs) for narrow, task-specific automation.
Generative AI is positioned as a precursor, with AI Agents advancing through
tool integration, prompt engineering, and reasoning enhancements. In contrast,
Agentic AI systems represent a paradigmatic shift marked by multi-agent
collaboration, dynamic task decomposition, persistent memory, and orchestrated
autonomy. Through a sequential evaluation of architectural evolution,
operational mechanisms, interaction styles, and autonomy levels, we present a
comparative analysis across both paradigms. Application domains such as
customer support, scheduling, and data summarization are contrasted with
Agentic AI deployments in research automation, robotic coordination, and
medical decision support. We further examine unique challenges in each paradigm
including hallucination, brittleness, emergent behavior, and coordination
failure and propose targeted solutions such as ReAct loops, RAG, orchestration
layers, and causal modeling. This work aims to provide a definitive roadmap for
developing robust, scalable, and explainable AI agent and Agentic AI-driven
systems. >AI Agents, Agent-driven, Vision-Language-Models, Agentic AI Decision
Support System, Agentic-AI Applications

### 7. [Reinforced Interactive Continual Learning via Real-time Noisy Human Feedback](http://arxiv.org/pdf/2505.09925v1)

Authors: Yutao Yang, Jie Zhou, Junsong Li, Qianjun Pan, Bihao Zhan, Qin Chen, Xipeng Qiu, Liang He

This paper introduces an interactive continual learning paradigm where AI
models dynamically learn new skills from real-time human feedback while
retaining prior knowledge. This paradigm distinctively addresses two major
limitations of traditional continual learning: (1) dynamic model updates using
streaming, real-time human-annotated data, rather than static datasets with
fixed labels, and (2) the assumption of clean labels, by explicitly handling
the noisy feedback common in real-world interactions. To tackle these problems,
we propose RiCL, a Reinforced interactive Continual Learning framework
leveraging Large Language Models (LLMs) to learn new skills effectively from
dynamic feedback. RiCL incorporates three key components: a temporal
consistency-aware purifier to automatically discern clean from noisy samples in
data streams; an interaction-aware direct preference optimization strategy to
align model behavior with human intent by reconciling AI-generated and
human-provided feedback; and a noise-resistant contrastive learning module that
captures robust representations by exploiting inherent data relationships, thus
avoiding reliance on potentially unreliable labels. Extensive experiments on
two benchmark datasets (FewRel and TACRED), contaminated with realistic noise
patterns, demonstrate that our RiCL approach substantially outperforms existing
combinations of state-of-the-art online continual learning and noisy-label
learning methods.

### 8. [AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection](http://arxiv.org/pdf/2505.09926v1)

Authors: Bin-Bin Gao, Yue Zhu, Jiangtao Yan, Yuezhi Cai, Weixi Zhang, Meng Wang, Jun Liu, Yong Liu, Lei Wang, Chengjie Wang

Universal visual anomaly detection aims to identify anomalies from novel or
unseen vision domains without additional fine-tuning, which is critical in open
scenarios. Recent studies have demonstrated that pre-trained vision-language
models like CLIP exhibit strong generalization with just zero or a few normal
images. However, existing methods struggle with designing prompt templates,
complex token interactions, or requiring additional fine-tuning, resulting in
limited flexibility. In this work, we present a simple yet effective method
called AdaptCLIP based on two key insights. First, adaptive visual and textual
representations should be learned alternately rather than jointly. Second,
comparative learning between query and normal image prompt should incorporate
both contextual and aligned residual features, rather than relying solely on
residual features. AdaptCLIP treats CLIP models as a foundational service,
adding only three simple adapters, visual adapter, textual adapter, and
prompt-query adapter, at its input or output ends. AdaptCLIP supports
zero-/few-shot generalization across domains and possesses a training-free
manner on target domains once trained on a base dataset. AdaptCLIP achieves
state-of-the-art performance on 12 anomaly detection benchmarks from industrial
and medical domains, significantly outperforming existing competitive methods.
We will make the code and model of AdaptCLIP available at
https://github.com/gaobb/AdaptCLIP.

### 9. [Task-Core Memory Management and Consolidation for Long-term Continual Learning](http://arxiv.org/pdf/2505.09952v1)

Authors: Tianyu Huai, Jie Zhou, Yuxuan Cai, Qin Chen, Wen Wu, Xingjiao Wu, Xipeng Qiu, Liang He

In this paper, we focus on a long-term continual learning (CL) task, where a
model learns sequentially from a stream of vast tasks over time, acquiring new
knowledge while retaining previously learned information in a manner akin to
human learning. Unlike traditional CL settings, long-term CL involves handling
a significantly larger number of tasks, which exacerbates the issue of
catastrophic forgetting. Our work seeks to address two critical questions: 1)
How do existing CL methods perform in the context of long-term CL? and 2) How
can we mitigate the catastrophic forgetting that arises from prolonged
sequential updates? To tackle these challenges, we propose a novel framework
inspired by human memory mechanisms for long-term continual learning (Long-CL).
Specifically, we introduce a task-core memory management strategy to
efficiently index crucial memories and adaptively update them as learning
progresses. Additionally, we develop a long-term memory consolidation mechanism
that selectively retains hard and discriminative samples, ensuring robust
knowledge retention. To facilitate research in this area, we construct and
release two multi-modal and textual benchmarks, MMLongCL-Bench and
TextLongCL-Bench, providing a valuable resource for evaluating long-term CL
approaches. Experimental results show that Long-CL outperforms the previous
state-of-the-art by 7.4\% and 6.5\% AP on the two benchmarks, respectively,
demonstrating the effectiveness of our approach.

### 10. [TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation](http://arxiv.org/pdf/2505.09955v1)

Authors: Jaeho Kim, Seulki Lee

Unsupervised domain adaptation (UDA) for time series data remains a critical
challenge in deep learning, with traditional pseudo-labeling strategies failing
to capture temporal patterns and channel-wise shifts between domains, producing
sub-optimal pseudo-labels. As such, we introduce TransPL, a novel approach that
addresses these limitations by modeling the joint distribution $P(\mathbf{X},
y)$ of the source domain through code transition matrices, where the codes are
derived from vector quantization (VQ) of time series patches. Our method
constructs class- and channel-wise code transition matrices from the source
domain and employs Bayes' rule for target domain adaptation, generating
pseudo-labels based on channel-wise weighted class-conditional likelihoods.
TransPL offers three key advantages: explicit modeling of temporal transitions
and channel-wise shifts between different domains, versatility towards
different UDA scenarios (e.g., weakly-supervised UDA), and explainable
pseudo-label generation. We validate TransPL's effectiveness through extensive
analysis on four time series UDA benchmarks and confirm that it consistently
outperforms state-of-the-art pseudo-labeling methods by a strong margin (6.1%
accuracy improvement, 4.9% F1 improvement), while providing interpretable
insights into the domain adaptation process through its learned code transition
matrices.

### Hardware Architecture

### 1. [Basilisk: A 34 mm2 End-to-End Open-Source 64-bit Linux-Capable RISC-V SoC in 130nm BiCMOS](http://arxiv.org/pdf/2505.10060v1)

Authors: Philippe Sauter, Thomas Benz, Paul Scheffler, Martin Povišer, Frank K. Gürkaynak, Luca Benini

End-to-end open-source electronic design automation (OSEDA) enables a
collaborative approach to chip design conducive to supply chain diversification
and zero-trust step-by-step design verification. However, existing end-to-end
OSEDA flows have mostly been demonstrated on small designs and have not yet
enabled large, industry-grade chips such as Linux-capable systems-on-chip
(SoCs). This work presents Basilisk, the largest end-to-end open-source SoC to
date. Basilisk's 34 mm2, 2.7 MGE design features a 64-bit Linux-capable RISC-V
core, a lightweight 124 MB/s DRAM controller, and extensive IO, including a USB
1.1 host, a video output, and a fully digital 62 Mb/s chip-to-chip (C2C) link.
We implement Basilisk in IHP's open 130 nm BiCMOS technology, significantly
improving on the state-of-the-art (SoA) OSEDA flow. Our enhancements of the
Yosys-based synthesis flow improve design timing and area by 2.3x and 1.6x,
respectively, while consuming significantly less system resources. By tuning
OpenROAD place and route (P&R) to our design and technology, we decrease the
die size by 12%. The fabricated Basilisk chip reaches 62 MHz at its nominal 1.2
V core voltage and up to 102 MHz at 1.64 V. It achieves a peak energy
efficiency of 18.9 DP MFLOP/s/W at 0.88 V.

### 2. [An Integrated UVM-TLM Co-Simulation Framework for RISC-V Functional Verification and Performance Evaluation](http://arxiv.org/pdf/2505.10145v1)

Authors: Ruizhi Qiu, Yang Liu

The burgeoning RISC-V ecosystem necessitates efficient verification
methodologies for complex processors. Traditional approaches often struggle to
concurrently evaluate functional correctness and performance, or balance
simulation speed with modeling accuracy. This paper introduces an integrated
co-simulation framework leveraging Universal Verification Methodology (UVM) and
Transaction-Level Modeling (TLM) for RISC-V processor validation. We present a
configurable UVM-TLM model (vmodel) of a superscalar, out-of-order RISC-V core,
featuring key microarchitectural modeling techniques such as credit-based
pipeline flow control. This environment facilitates unified functional
verification via co-simulation against the Spike ISA simulator and enables
early-stage performance assessment using benchmarks like CoreMark, orchestrated
within UVM. The methodology prioritizes integration, simulation efficiency, and
acceptable fidelity for architectural exploration over cycle-level precision.
Experimental results validate functional correctness and significant simulation
speedup over RTL approaches, accelerating design iterations and enhancing
verification coverage.

### 3. [Enabling Syscall Intercept for RISC-V](http://arxiv.org/pdf/2505.10217v1)

Authors: Petar Andrić, Aaron Call, Ramon Nou

The European Union technological sovereignty strategy centers around the
RISC-V Instruction Set Architecture, with the European Processor Initiative
leading efforts to build production-ready processors. Focusing on realizing a
functional RISC-V ecosystem, the BZL initiative (www.bzl.es) is making an
effort to create a software stack along with the hardware. In this work, we
detail the efforts made in porting a widely used syscall interception library,
mainly used on AdHocFS (i.e., DAOS, GekkoFS), to RISC-V and how we overcame
some of the limitations encountered.

### 4. [Scalable 28nm IC implementation of coupled oscillator network featuring tunable topology and complexity](http://arxiv.org/pdf/2505.10248v1)

Authors: S. Y. Neyaz, A. Ashok, M. Schiek, C. Grewing, A. Zambanini, S. van Waasen

Integrated circuit implementations of coupled oscillator networks have
recently gained increased attention. The focus is usually on using these
networks for analogue computing, for example for solving computational
optimization tasks. For use within analog computing, these networks are run
close to critical dynamics. On the other hand, such networks are also used as
an analogy of transport networks such as electrical power grids to answer the
question of how exactly such critical dynamic states can be avoided. However,
simulating large network of coupled oscillators is computationally intensive,
with specifc regards to electronic ones. We have developed an integrated
circuit using integrated Phase-Locked Loop (PLL) with modifications, that
allows to flexibly vary the topology as well as a complexity parameter of the
network during operation. The proposed architecture, inspired by the brain,
employs a clustered architecture, with each cluster containing 7 PLLs featuring
programmable coupling mechanisms. Additionally, the inclusion of a RISC-V
processor enables future algorithmic implementations. Thus, we provide a
practical alternative for large-scale network simulations both in the field of
analog computing and transport network stability research.

### Computational Complexity

### 1. [A Fine-Grained Complexity View on Propositional Abduction -- Algorithms and Lower Bounds](http://arxiv.org/pdf/2505.10201v1)

Authors: Victor Lagerkvist, Mohamed Maizia, Johannes Schmidt

The Boolean satisfiability problem (SAT) is a well-known example of monotonic
reasoning, of intense practical interest due to fast solvers, complemented by
rigorous fine-grained complexity results. However, for non-monotonic reasoning,
e.g., abductive reasoning, comparably little is known outside classic
complexity theory. In this paper we take a first step of bridging the gap
between monotonic and non-monotonic reasoning by analyzing the complexity of
intractable abduction problems under the seemingly overlooked but natural
parameter n: the number of variables in the knowledge base. We obtain several
positive results for $\Sigma^P_2$- as well as NP- and coNP-complete fragments,
which implies the first example of beating exhaustive search for a
$\Sigma^P_2$-complete problem (to the best of our knowledge). We complement
this with lower bounds and for many fragments rule out improvements under the
(strong) exponential-time hypothesis.

### 2. [On the quantum computational complexity of classical linear dynamics with geometrically local interactions: Dequantization and universality](http://arxiv.org/pdf/2505.10445v1)

Authors: Kazuki Sakamoto, Keisuke Fujii

The simulation of large-scale classical systems in exponentially small space
on quantum computers has gained attention. The prior work demonstrated that a
quantum algorithm offers an exponential speedup over any classical algorithm in
simulating classical dynamics with long-range interactions. However, many
real-world classical systems, such as those arising from partial differential
equations, exhibit only local interactions. The question remains whether
quantum algorithms can still provide exponential speedup under this condition.
In this work, we thoroughly characterize the computational complexity of
quantum algorithms for simulating such geometrically local systems. First, we
dequantize the quantum algorithm for simulating short-time (polynomial-time)
dynamics of such systems. This implies that the problem of simulating this
dynamics does not yield any exponential quantum advantage. Second, we show that
quantum algorithms for short-time dynamics have the same computational
complexity as polynomial-time probabilistic classical computation. Third, we
show that the computational complexity of quantum algorithms for long-time
(exponential-time) dynamics is captured by exponential-time and
polynomial-space quantum computation. This suggests a super-polynomial time
advantage when restricting the computation to polynomial-space, or an
exponential space advantage otherwise. This work offers new insights into the
complexity of classical dynamics governed by partial differential equations,
providing a pathway for achieving quantum advantage in practical problems.

### 3. [Multi-Agent Path Finding For Large Agents Is Intractable](http://arxiv.org/pdf/2505.10387v1)

Authors: Artem Agafonov, Konstantin Yakovlev

The multi-agent path finding (MAPF) problem asks to find a set of paths on a
graph such that when synchronously following these paths the agents never
encounter a conflict. In the most widespread MAPF formulation, the so-called
Classical MAPF, the agents sizes are neglected and two types of conflicts are
considered: occupying the same vertex or using the same edge at the same time
step. Meanwhile in numerous practical applications, e.g. in robotics, taking
into account the agents' sizes is vital to ensure that the MAPF solutions can
be safely executed. Introducing large agents yields an additional type of
conflict arising when one agent follows an edge and its body overlaps with the
body of another agent that is actually not using this same edge (e.g. staying
still at some distinct vertex of the graph). Until now it was not clear how
harder the problem gets when such conflicts are to be considered while
planning. Specifically, it was known that Classical MAPF problem on an
undirected graph can be solved in polynomial time, however no complete
polynomial-time algorithm was presented to solve MAPF with large agents. In
this paper we, for the first time, establish that the latter problem is NP-hard
and, thus, if P!=NP no polynomial algorithm for it can, unfortunately, be
presented. Our proof is based on the prevalent in the field technique of
reducing the seminal 3SAT problem (which is known to be an NP-complete problem)
to the problem at hand. In particular, for an arbitrary 3SAT formula we
procedurally construct a dedicated graph with specific start and goal vertices
and show that the given 3SAT formula is satisfiable iff the corresponding path
finding instance has a solution.

### Computational Engineering

### 1. [Promise of Data-Driven Modeling and Decision Support for Precision Oncology and Theranostics](http://arxiv.org/pdf/2505.09899v1)

Authors: Binesh Sadanandan, Vahid Behzadan

Cancer remains a leading cause of death worldwide, necessitating personalized
treatment approaches to improve outcomes. Theranostics, combining
molecular-level imaging with targeted therapy, offers potential for precision
oncology but requires optimized, patient-specific care plans. This paper
investigates state-of-the-art data-driven decision support applications with a
reinforcement learning focus in precision oncology. We review current
applications, training environments, state-space representation, performance
evaluation criteria, and measurement of risk and reward, highlighting key
challenges. We propose a framework integrating data-driven modeling with
reinforcement learning-based decision support to optimize radiopharmaceutical
therapy dosing, addressing identified challenges and setting directions for
future research. The framework leverages Neural Ordinary Differential Equations
and Physics-Informed Neural Networks to enhance Physiologically Based
Pharmacokinetic models while applying reinforcement learning algorithms to
iteratively refine treatment policies based on patient-specific data.

### 2. [Physical regularized Hierarchical Generative Model for Metallic Glass Structural Generation and Energy Prediction](http://arxiv.org/pdf/2505.09977v1)

Authors: Qiyuan Chen, Ajay Annamareddy, Ying-Fei Li, Dane Morgan, Bu Wang

Disordered materials such as glasses, unlike crystals, lack long range atomic
order and have no periodic unit cells, yielding a high dimensional
configuration space with widely varying properties. The complexity not only
increases computational costs for atomistic simulations but also makes it
difficult for generative AI models to deliver accurate property predictions and
realistic structure generation. In this work, we introduce GlassVAE, a
hierarchical graph variational autoencoder that uses graph representations to
learn compact, rotation, translation, and permutation invariant embeddings of
atomic configurations. The resulting structured latent space not only enables
efficient generation of novel, physically plausible structures but also
supports exploration of the glass energy landscape. To enforce structural
realism and physical fidelity, we augment GlassVAE with two physics informed
regularizers, a radial distribution function (RDF) loss that captures
characteristic short and medium range ordering and an energy regression loss
that reflects the broad configurational energetics. Both theoretical analysis
and experimental results highlight the critical impact of these regularizers.
By encoding high dimensional atomistic data into a compact latent vector and
decoding it into structures with accurate energy predictions, GlassVAE provides
a fast, physics aware path for modeling and designing disordered materials.

### 3. [Knowledge-Based Aerospace Engineering -- A Systematic Literature Review](http://arxiv.org/pdf/2505.10142v1)

Authors: Tim Wittenborg, Ildar Baimuratov, Ludvig Knöös Franzén, Ingo Staack, Ulrich Römer, Sören Auer

The aerospace industry operates at the frontier of technological innovation
while maintaining high standards regarding safety and reliability. In this
environment, with an enormous potential for re-use and adaptation of existing
solutions and methods, Knowledge-Based Engineering (KBE) has been applied for
decades. The objective of this study is to identify and examine
state-of-the-art knowledge management practices in the field of aerospace
engineering. Our contributions include: 1) A SWARM-SLR of over 1,000 articles
with qualitative analysis of 164 selected articles, supported by two aerospace
engineering domain expert surveys. 2) A knowledge graph of over 700
knowledge-based aerospace engineering processes, software, and data, formalized
in the interoperable Web Ontology Language (OWL) and mapped to Wikidata entries
where possible. The knowledge graph is represented on the Open Research
Knowledge Graph (ORKG), and an aerospace Wikibase, for reuse and continuation
of structuring aerospace engineering knowledge exchange. 3) Our resulting
intermediate and final artifacts of the knowledge synthesis, available as a
Zenodo dataset. This review sets a precedent for structured, semantic-based
approaches to managing aerospace engineering knowledge. By advancing these
principles, research, and industry can achieve more efficient design processes,
enhanced collaboration, and a stronger commitment to sustainable aviation.

### 4. [Space-Time Multigrid Methods Suitable for Topology Optimisation of Transient Heat Conduction](http://arxiv.org/pdf/2505.10168v1)

Authors: Magnus Appel, Joe Alexandersen

This paper presents Space-Time MultiGrid (STMG) methods which are suitable
for performing topology optimisation of transient heat conduction problems. The
proposed methods use a pointwise smoother and uniform Cartesian space-time
meshes. For problems with high contrast in the diffusivity, it was found that
it is beneficial to define a coarsening strategy based on the geometric mean of
the minimum and maximum diffusivity. However, other coarsening strategies may
be better for other smoothers. Several methods of discretising the coarse
levels were tested. Of these, it was best to use a method which averages the
thermal resistivities on the finer levels. However, this was likely a
consequence of the fact that only one spatial dimension was considered for the
test problems. A second coarsening strategy was proposed which ensures spatial
resolution on the coarse grids. Mixed results were found for this strategy. The
proposed STMG methods were used as a solver for a one-dimensional topology
optimisation problem. In this context, the adjoint problem was also solved
using the STMG methods. The STMG methods were sufficiently robust for this
application, since they converged during every optimisation cycle. It was found
that the STMG methods also work for the adjoint problem when the prolongation
operator only sends information forwards in time, even although the direction
of time for the adjoint problem is backwards.

### 5. [Avocado Price Prediction Using a Hybrid Deep Learning Model: TCN-MLP-Attention Architecture](http://arxiv.org/pdf/2505.09907v1)

Authors: Linwei Zhang, LuFeng, Ruijia Liang

With the growing demand for healthy foods, agricultural product price
forecasting has become increasingly important. Hass avocados, as a high-value
crop, exhibit complex price fluctuations influenced by factors such as
seasonality, region, and weather. Traditional prediction models often struggle
with highly nonlinear and dynamic data. To address this, we propose a hybrid
deep learning model, TCN-MLP-Attention Architecture, combining Temporal
Convolutional Networks (TCN) for sequential feature extraction, Multi-Layer
Perceptrons (MLP) for nonlinear interactions, and an Attention mechanism for
dynamic feature weighting. The dataset used covers over 50,000 records of Hass
avocado sales across the U.S. from 2015 to 2018, including variables such as
sales volume, average price, time, region, weather, and variety type, collected
from point-of-sale systems and the Hass Avocado Board. After systematic
preprocessing, including missing value imputation and feature normalization,
the proposed model was trained and evaluated. Experimental results demonstrate
that the TCN-MLP-Attention model achieves excellent predictive performance,
with an RMSE of 1.23 and an MSE of 1.51, outperforming traditional methods.
This research provides a scalable and effective approach for time series
forecasting in agricultural markets and offers valuable insights for
intelligent supply chain management and price strategy optimization.

### Computational Geometry

### 1. [Topology-driven identification of repetitions in multi-variate time series](http://arxiv.org/pdf/2505.10004v1)

Authors: Simon Schindler, Elias Steffen Reich, Saverio Messineo, Simon Hoher, Stefan Huber

Many multi-variate time series obtained in the natural sciences and
engineering possess a repetitive behavior, as for instance state-space
trajectories of industrial machines in discrete automation. Recovering the
times of recurrence from such a multi-variate time series is of a fundamental
importance for many monitoring and control tasks. For a periodic time series
this is equivalent to determining its period length. In this work we present a
persistent homology framework to estimate recurrence times in multi-variate
time series with different generalizations of cyclic behavior (periodic,
repetitive, and recurring). To this end, we provide three specialized methods
within our framework that are provably stable and validate them using
real-world data, including a new benchmark dataset from an injection molding
machine.

### Computation and Language

### 1. [Crossing Borders Without Crossing Boundaries: How Sociolinguistic Awareness Can Optimize User Engagement with Localized Spanish AI Models Across Hispanophone Countries](http://arxiv.org/pdf/2505.09902v1)

Authors: Martin Capdevila, Esteban Villa Turek, Ellen Karina Chumbe Fernandez, Luis Felipe Polo Galvez, Luis Cadavid, Andrea Marroquin, Rebeca Vargas Quesada, Johanna Crew, Nicole Vallejo Galarraga, Christopher Rodriguez, Diego Gutierrez, Radhi Datla

Large language models are, by definition, based on language. In an effort to
underscore the critical need for regional localized models, this paper examines
primary differences between variants of written Spanish across Latin America
and Spain, with an in-depth sociocultural and linguistic contextualization
therein. We argue that these differences effectively constitute significant
gaps in the quotidian use of Spanish among dialectal groups by creating
sociolinguistic dissonances, to the extent that locale-sensitive AI models
would play a pivotal role in bridging these divides. In doing so, this approach
informs better and more efficient localization strategies that also serve to
more adequately meet inclusivity goals, while securing sustainable active daily
user growth in a major low-risk investment geographic area. Therefore,
implementing at least the proposed five sub variants of Spanish addresses two
lines of action: to foment user trust and reliance on AI language models while
also demonstrating a level of cultural, historical, and sociolinguistic
awareness that reflects positively on any internationalization strategy.

### 2. [Rethinking Prompt Optimizers: From Prompt Merits to Optimization](http://arxiv.org/pdf/2505.09930v1)

Authors: Zixiao Zhu, Hanzhang Zhou, Zijian Feng, Tianjiao Li, Chua Jia Jim Deryl, Mak Lee Onn, Gee Wah Ng, Kezhi Mao

Prompt optimization (PO) offers a practical alternative to fine-tuning large
language models (LLMs), enabling performance improvements without altering
model weights. Existing methods typically rely on advanced, large-scale LLMs
like GPT-4 to generate optimized prompts. However, due to limited downward
compatibility, verbose, instruction-heavy prompts from advanced LLMs can
overwhelm lightweight inference models and degrade response quality. In this
work, we rethink prompt optimization through the lens of interpretable design.
We first identify a set of model-agnostic prompt quality merits and empirically
validate their effectiveness in enhancing prompt and response quality. We then
introduce MePO, a merit-guided, lightweight, and locally deployable prompt
optimizer trained on our preference dataset built from merit-aligned prompts
generated by a lightweight LLM. Unlike prior work, MePO avoids online
optimization reliance, reduces cost and privacy concerns, and, by learning
clear, interpretable merits, generalizes effectively to both large-scale and
lightweight inference models. Experiments demonstrate that MePO achieves better
results across diverse tasks and model types, offering a scalable and robust
solution for real-world deployment. Our model and dataset are available at:
https://github.com/MidiyaZhu/MePO

### 3. [DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs](http://arxiv.org/pdf/2505.10013v1)

Authors: Lake Yin, Fan Huang

As Large Language Models (LLMs) have risen in prominence over the past few
years, there has been concern over the potential biases in LLMs inherited from
the training data. Previous studies have examined how LLMs exhibit implicit
bias, such as when response generation changes when different social contexts
are introduced. We argue that this implicit bias is not only an ethical, but
also a technical issue, as it reveals an inability of LLMs to accommodate
extraneous information. However, unlike other measures of LLM intelligence,
there are no standard methods to benchmark this specific subset of LLM bias. To
bridge this gap, we developed a method for calculating an easily interpretable
benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM
logic and math problem datasets with sociodemographic personas. We demonstrate
that this method can statistically validate the presence of implicit bias in
LLM behavior and find an inverse trend between question answering accuracy and
implicit bias, supporting our argument.

### 4. [CAFE: Retrieval Head-based Coarse-to-Fine Information Seeking to Enhance Multi-Document QA Capability](http://arxiv.org/pdf/2505.10063v1)

Authors: Han Peng, Jinhao Jiang, Zican Dong, Wayne Xin Zhao, Lei Fang

Advancements in Large Language Models (LLMs) have extended their input
context length, yet they still struggle with retrieval and reasoning in
long-context inputs. Existing methods propose to utilize the prompt strategy
and retrieval head to alleviate this limitation. However, they still face
challenges in balancing retrieval precision and recall, impacting their
efficacy in answering questions. To address this, we introduce $\textbf{CAFE}$,
a two-stage coarse-to-fine method to enhance multi-document question-answering
capacities. By gradually eliminating the negative impacts of background and
distracting documents, CAFE makes the responses more reliant on the evidence
documents. Initially, a coarse-grained filtering method leverages retrieval
heads to identify and rank relevant documents. Then, a fine-grained steering
method guides attention to the most relevant content. Experiments across
benchmarks show CAFE outperforms baselines, achieving up to 22.1% and 13.7%
SubEM improvement over SFT and RAG methods on the Mistral model, respectively.

### 5. [Designing and Contextualising Probes for African Languages](http://arxiv.org/pdf/2505.10081v1)

Authors: Wisdom Aduah, Francois Meyer

Pretrained language models (PLMs) for African languages are continually
improving, but the reasons behind these advances remain unclear. This paper
presents the first systematic investigation into probing PLMs for linguistic
knowledge about African languages. We train layer-wise probes for six
typologically diverse African languages to analyse how linguistic features are
distributed. We also design control tasks, a way to interpret probe
performance, for the MasakhaPOS dataset. We find PLMs adapted for African
languages to encode more linguistic information about target languages than
massively multilingual PLMs. Our results reaffirm previous findings that
token-level syntactic information concentrates in middle-to-last layers, while
sentence-level semantic information is distributed across all layers. Through
control tasks and probing baselines, we confirm that performance reflects the
internal knowledge of PLMs rather than probe memorisation. Our study applies
established interpretability techniques to African-language PLMs. In doing so,
we highlight the internal mechanisms underlying the success of strategies like
active learning and multilingual adaptation.

### 6. [XRAG: Cross-lingual Retrieval-Augmented Generation](http://arxiv.org/pdf/2505.10089v1)

Authors: Wei Liu, Sony Trenous, Leonardo F. R. Ribeiro, Bill Byrne, Felix Hieber

We propose XRAG, a novel benchmark designed to evaluate the generation
abilities of LLMs in cross-lingual Retrieval-Augmented Generation (RAG)
settings where the user language does not match the retrieval results. XRAG is
constructed from recent news articles to ensure that its questions require
external knowledge to be answered. It covers the real-world scenarios of
monolingual and multilingual retrieval, and provides relevancy annotations for
each retrieved document. Our novel dataset construction pipeline results in
questions that require complex reasoning, as evidenced by the significant gap
between human and LLM performance. Consequently, XRAG serves as a valuable
benchmark for studying LLM reasoning abilities, even before considering the
additional cross-lingual complexity. Experimental results on five LLMs uncover
two previously unreported challenges in cross-lingual RAG: 1) in the
monolingual retrieval setting, all evaluated models struggle with response
language correctness; 2) in the multilingual retrieval setting, the main
challenge lies in reasoning over retrieved information across languages rather
than generation of non-English text.

### 7. [What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs](http://arxiv.org/pdf/2505.10113v1)

Authors: Xinlan Yan, Di Wu, Yibin Lei, Christof Monz, Iacer Calixto

In this paper, we introduce S-MedQA, an English medical question-answering
(QA) dataset for benchmarking large language models in fine-grained clinical
specialties. We use S-MedQA to check the applicability of a popular hypothesis
related to knowledge injection in the knowledge-intense scenario of medical QA,
and show that: 1) training on data from a speciality does not necessarily lead
to best performance on that specialty and 2) regardless of the specialty
fine-tuned on, token probabilities of clinically relevant terms for all
specialties increase consistently. Thus, we believe improvement gains come
mostly from domain shifting (e.g., general to medical) rather than knowledge
injection and suggest rethinking the role of fine-tuning data in the medical
domain. We release S-MedQA and all code needed to reproduce all our experiments
to the research community.

### 8. [GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs](http://arxiv.org/pdf/2505.10143v1)

Authors: Longchao Da, Parth Mitesh Shah, Kuan-Ru Liou, Jiaxing Zhang, Hua Wei

Large Language Models are now key assistants in human decision-making
processes. However, a common note always seems to follow: "LLMs can make
mistakes. Be careful with important info." This points to the reality that not
all outputs from LLMs are dependable, and users must evaluate them manually.
The challenge deepens as hallucinated responses, often presented with seemingly
plausible explanations, create complications and raise trust issues among
users. To tackle such issue, this paper proposes GE-Chat, a knowledge Graph
enhanced retrieval-augmented generation framework to provide Evidence-based
response generation. Specifically, when the user uploads a material document, a
knowledge graph will be created, which helps construct a retrieval-augmented
agent, enhancing the agent's responses with additional knowledge beyond its
training corpus. Then we leverage Chain-of-Thought (CoT) logic generation,
n-hop sub-graph searching, and entailment-based sentence generation to realize
accurate evidence retrieval. We demonstrate that our method improves the
existing models' performance in terms of identifying the exact evidence in a
free-form context, providing a reliable way to examine the resources of LLM's
conclusion and help with the judgment of the trustworthiness.

### 9. [VQ-Logits: Compressing the Output Bottleneck of Large Language Models via Vector Quantized Logits](http://arxiv.org/pdf/2505.10202v1)

Authors: Jintian Shao, Hongyi Huang, Jiayi Wu, YiMing Cheng, ZhiYu Wu, You Shan, MingKai Zheng

Large Language Models (LLMs) have achieved remarkable success but face
significant computational and memory challenges, particularly due to their
extensive output vocabularies. The final linear projection layer, mapping
hidden states to vocabulary-sized logits, often constitutes a substantial
portion of the model's parameters and computational cost during inference.
Existing methods like adaptive softmax or hierarchical softmax introduce
structural complexities. In this paper, we propose VQ-Logits, a novel approach
that leverages Vector Quantization (VQ) to drastically reduce the parameter
count and computational load of the LLM output layer. VQ-Logits replaces the
large V * dmodel output embedding matrix with a small, shared codebook of K
embedding vectors (K << V ). Each token in the vocabulary is mapped to one of
these K codebook vectors. The LLM predicts logits over this compact codebook,
which are then efficiently "scattered" to the full vocabulary space using the
learned or preassigned mapping. We demonstrate through extensive experiments on
standard language modeling benchmarks (e.g., WikiText-103, C4) that VQ-Logits
can achieve up to 99% parameter reduction in the output layer and 6x speedup in
logit computation, with only a marginal 4% increase in perplexity compared to
full softmax baselines. We further provide detailed ablation studies on
codebook size, initialization, and learning strategies, showcasing the
robustness and effectiveness of our approach.

### 10. [RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward](http://arxiv.org/pdf/2505.10218v1)

Authors: Zongsheng Wang, Kaili Sun, Bowen Wu, Qun Yu, Ying Li, Baoxun Wang

Role-playing conversational agents (RPCAs) face persistent challenges in
maintaining role consistency. To address this, we propose RAIDEN-R1, a novel
reinforcement learning framework that integrates Verifiable Role-Awareness
Reward (VRAR). The method introduces both singular and multi-term mining
strategies to generate quantifiable rewards by assessing role-specific keys.
Additionally, we construct a high-quality, role-aware Chain-of-Thought dataset
through multi-LLM collaboration, and implement experiments to enhance reasoning
coherence. Experiments on the RAIDEN benchmark demonstrate RAIDEN-R1's
superiority: our 14B-GRPO model achieves 88.04% and 88.65% accuracy on
Script-Based Knowledge and Conversation Memory metrics, respectively,
outperforming baseline models while maintaining robustness. Case analyses
further reveal the model's enhanced ability to resolve conflicting contextual
cues and sustain first-person narrative consistency. This work bridges the
non-quantifiability gap in RPCA training and provides insights into role-aware
reasoning patterns, advancing the development of RPCAs.

### Cryptography and Security

### 1. [Correlating Account on Ethereum Mixing Service via Domain-Invariant feature learning](http://arxiv.org/pdf/2505.09892v1)

Authors: Zheng Che, Taoyu Li, Meng Shen, Hanbiao Du, Liehuang Zhu

The untraceability of transactions facilitated by Ethereum mixing services
like Tornado Cash poses significant challenges to blockchain security and
financial regulation. Existing methods for correlating mixing accounts suffer
from limited labeled data and vulnerability to noisy annotations, which
restrict their practical applicability. In this paper, we propose StealthLink,
a novel framework that addresses these limitations through cross-task
domain-invariant feature learning. Our key innovation lies in transferring
knowledge from the well-studied domain of blockchain anomaly detection to the
data-scarce task of mixing transaction tracing. Specifically, we design a
MixFusion module that constructs and encodes mixing subgraphs to capture local
transactional patterns, while introducing a knowledge transfer mechanism that
aligns discriminative features across domains through adversarial discrepancy
minimization. This dual approach enables robust feature learning under label
scarcity and distribution shifts. Extensive experiments on real-world mixing
transaction datasets demonstrate that StealthLink achieves state-of-the-art
performance, with 96.98\% F1-score in 10-shot learning scenarios. Notably, our
framework shows superior generalization capability in imbalanced data
conditions than conventional supervised methods. This work establishes the
first systematic approach for cross-domain knowledge transfer in blockchain
forensics, providing a practical solution for combating privacy-enhanced
financial crimes in decentralized ecosystems.

### 2. [DeFeed: Secure Decentralized Cross-Contract Data Feed in Web 3.0 for Connected Autonomous Vehicles](http://arxiv.org/pdf/2505.09928v1)

Authors: Xingchen Sun, Runhua Xu, Wei Ni, Li Duan, Chao Li

Smart contracts have been a topic of interest in blockchain research and are
a key enabling technology for Connected Autonomous Vehicles (CAVs) in the era
of Web 3.0. These contracts enable trustless interactions without the need for
intermediaries, as they operate based on predefined rules encoded on the
blockchain. However, smart contacts face significant challenges in
cross-contract communication and information sharing, making it difficult to
establish seamless connectivity and collaboration among CAVs with Web 3.0. In
this paper, we propose DeFeed, a novel secure protocol that incorporates
various gas-saving functions for CAVs, originated from in-depth research into
the interaction among smart contracts for decentralized cross-contract data
feed in Web 3.0. DeFeed allows smart contracts to obtain information from other
contracts efficiently in a single click, without complicated operations. We
judiciously design and complete various functions with DeFeed, including a pool
function and a cache function for gas optimization, a subscribe function for
facilitating data access, and an update function for the future iteration of
our protocol. Tailored for CAVs with Web 3.0 use cases, DeFeed enables
efficient data feed between smart contracts underpinning decentralized
applications and vehicle coordination. Implemented and tested on the Ethereum
official test network, DeFeed demonstrates significant improvements in contract
interaction efficiency, reducing computational complexity and gas costs. Our
solution represents a critical step towards seamless, decentralized
communication in Web 3.0 ecosystems.

### 3. [Security and Privacy Measurement on Chinese Consumer IoT Traffic based on Device Lifecycle](http://arxiv.org/pdf/2505.09929v1)

Authors: Chenghua Jin, Yan Jia, Yuxin Song, Qingyin Tan, Rui Yang, Zheli Liu

In recent years, consumer Internet of Things (IoT) devices have become widely
used in daily life. With the popularity of devices, related security and
privacy risks arise at the same time as they collect user-related data and
transmit it to various service providers. Although China accounts for a larger
share of the consumer IoT industry, current analyses on consumer IoT device
traffic primarily focus on regions such as Europe, the United States, and
Australia. Research on China, however, is currently rather rare. This study
constructs the first large-scale dataset about consumer IoT device traffic in
China. Specifically, we propose a fine-grained traffic collection guidance
covering the entire lifecycle of consumer IoT devices, gathering traffic from
70 devices spanning 36 brands and 8 device categories. Based on this dataset,
we analyze traffic destinations and encryption practices across different
device types during the entire lifecycle and compare the findings with the
results of other regions. Compared to other regions, our results show that
consumer IoT devices in China rely more on domestic services and overally
perform better in terms of encryption practices. However, there are still 20/35
devices improperly conduct certificate validation, and 5/70 devices use
insecure encryption protocols. To facilitate future research, we open-source
our traffic collection guidance and make our dataset publicly available.

### 4. [When Mitigations Backfire: Timing Channel Attacks and Defense for PRAC-Based RowHammer Mitigations](http://arxiv.org/pdf/2505.10111v1)

Authors: Jeonghyun Woo, Joyce Qu, Gururaj Saileshwar, Prashant J. Nair

Per Row Activation Counting (PRAC) has emerged as a robust framework for
mitigating RowHammer (RH) vulnerabilities in modern DRAM systems. However, we
uncover a critical vulnerability: a timing channel introduced by the Alert
Back-Off (ABO) protocol and Refresh Management (RFM) commands. We present
PRACLeak, a novel attack that exploits these timing differences to leak
sensitive information, such as secret keys from vulnerable AES implementations,
by monitoring memory access latencies.
  To counter this, we propose Timing-Safe PRAC (TPRAC), a defense that
eliminates PRAC-induced timing channels without compromising RH mitigation
efficacy. TPRAC uses Timing-Based RFMs, issued periodically and independent of
memory activity. It requires only a single-entry in-DRAM mitigation queue per
DRAM bank and is compatible with existing DRAM standards. Our evaluations
demonstrate that TPRAC closes timing channels while incurring only 3.4%
performance overhead at the RH threshold of 1024.

### 5. [One For All: Formally Verifying Protocols which use Aggregate Signatures (extended version)](http://arxiv.org/pdf/2505.10316v1)

Authors: Xenia Hofmeier, Andrea Raguso, Ralf Sasse, Dennis Jackson, David Basin

Aggregate signatures are digital signatures that compress multiple signatures
from different parties into a single signature, thereby reducing storage and
bandwidth requirements. BLS aggregate signatures are a popular kind of
aggregate signature, deployed by Ethereum, Dfinity, and Cloudflare amongst
others, currently undergoing standardization at the IETF. However, BLS
aggregate signatures are difficult to use correctly, with nuanced requirements
that must be carefully handled by protocol developers.
  In this work, we design the first models of aggregate signatures that enable
formal verification tools, such as Tamarin and ProVerif, to be applied to
protocols using these signatures. We introduce general models that are based on
the cryptographic security definition of generic aggregate signatures, allowing
the attacker to exploit protocols where the security requirements are not
satisfied. We also introduce a second family of models formalizing BLS
aggregate signatures in particular. We demonstrate our approach's practical
relevance by modelling and analyzing in Tamarin a device attestation protocol
called SANA. Despite SANA's claimed correctness proof, with Tamarin we uncover
undocumented assumptions that, when omitted, lead to attacks.

### 6. [Locally Differentially Private Frequency Estimation via Joint Randomized Response](http://arxiv.org/pdf/2505.10349v1)

Authors: Ye Zheng, Shafizur Rahman Seeam, Yidan Hu, Rui Zhang, Yanchao Zhang

Local Differential Privacy (LDP) has been widely recognized as a powerful
tool for providing a strong theoretical guarantee of data privacy to data
contributors against an untrusted data collector. Under a typical LDP scheme,
each data contributor independently randomly perturbs their data before
submitting them to the data collector, which in turn infers valuable statistics
about the original data from received perturbed data. Common to existing LDP
mechanisms is an inherent trade-off between the level of privacy protection and
data utility in the sense that strong data privacy often comes at the cost of
reduced data utility. Frequency estimation based on Randomized Response (RR) is
a fundamental building block of many LDP mechanisms. In this paper, we propose
a novel Joint Randomized Response (JRR) mechanism based on correlated data
perturbations to achieve locally differentially private frequency estimation.
JRR divides data contributors into disjoint groups of two members and lets
those in the same group jointly perturb their binary data to improve
frequency-estimation accuracy and achieve the same level of data privacy by
hiding the group membership information in contrast to the classical RR
mechanism. Theoretical analysis and detailed simulation studies using both real
and synthetic datasets show that JRR achieves the same level of data privacy as
the classical RR mechanism while improving the frequency-estimation accuracy in
the overwhelming majority of the cases by up to two orders of magnitude.

### 7. [The Ephemeral Threat: Assessing the Security of Algorithmic Trading Systems powered by Deep Learning](http://arxiv.org/pdf/2505.10430v1)

Authors: Advije Rizvani, Giovanni Apruzzese, Pavel Laskov

We study the security of stock price forecasting using Deep Learning (DL) in
computational finance. Despite abundant prior research on the vulnerability of
DL to adversarial perturbations, such work has hitherto hardly addressed
practical adversarial threat models in the context of DL-powered algorithmic
trading systems (ATS). Specifically, we investigate the vulnerability of ATS to
adversarial perturbations launched by a realistically constrained attacker. We
first show that existing literature has paid limited attention to DL security
in the financial domain, which is naturally attractive for adversaries. Then,
we formalize the concept of ephemeral perturbations (EP), which can be used to
stage a novel type of attack tailored for DL-based ATS. Finally, we carry out
an end-to-end evaluation of our EP against a profitable ATS. Our results reveal
that the introduction of small changes to the input stock prices not only (i)
induces the DL model to behave incorrectly but also (ii) leads the whole ATS to
make suboptimal buy/sell decisions, resulting in a worse financial performance
of the targeted ATS.

### 8. [S3C2 Summit 2024-09: Industry Secure Software Supply Chain Summit](http://arxiv.org/pdf/2505.10538v1)

Authors: Imranur Rahman, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

While providing economic and software development value, software supply
chains are only as strong as their weakest link. Over the past several years,
there has been an exponential increase in cyberattacks, specifically targeting
vulnerable links in critical software supply chains. These attacks disrupt the
day-to-day functioning and threaten the security of nearly everyone on the
internet, from billion-dollar companies and government agencies to hobbyist
open-source developers. The ever-evolving threat of software supply chain
attacks has garnered interest from the software industry and the US government
in improving software supply chain security.
  On September 20, 2024, three researchers from the NSF-backed Secure Software
Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with
a diverse set of 12 practitioners from 9 companies. The goals of the Summit
were to: (1) to enable sharing between individuals from different companies
regarding practical experiences and challenges with software supply chain
security, (2) to help form new collaborations, (3) to share our observations
from our previous summits with industry, and (4) to learn about practitioners'
challenges to inform our future research direction. The summit consisted of
discussions of six topics relevant to the companies represented, including
updating vulnerable dependencies, component and container choice, malicious
commits, building infrastructure, large language models, and reducing entire
classes of vulnerabilities.

### 9. [PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization](http://arxiv.org/pdf/2505.09921v1)

Authors: Yidan Wang, Yanan Cao, Yubing Ren, Fang Fang, Zheng Lin, Binxing Fang

Large Language Models (LLMs) excel in various domains but pose inherent
privacy risks. Existing methods to evaluate privacy leakage in LLMs often use
memorized prefixes or simple instructions to extract data, both of which
well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM
safety mechanisms to generate harmful content, but their role in privacy
scenarios remains underexplored. In this paper, we examine the effectiveness of
jailbreak attacks in extracting sensitive information, bridging privacy leakage
and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework
targeting Personally Identifiable Information (PII) and addressing the
limitations of current jailbreak methods. Specifically, PIG identifies PII
entities and their types in privacy queries, uses in-context learning to build
a privacy context, and iteratively updates it with three gradient-based
strategies to elicit target PII. We evaluate PIG and existing jailbreak methods
using two privacy-related datasets. Experiments on four white-box and two
black-box LLMs show that PIG outperforms baseline methods and achieves
state-of-the-art (SoTA) results. The results underscore significant privacy
risks in LLMs, emphasizing the need for stronger safeguards. Our code is
availble at
\href{https://github.com/redwyd/PrivacyJailbreak}{https://github.com/redwyd/PrivacyJailbreak}.

### 10. [From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](http://arxiv.org/pdf/2505.09924v1)

Authors: Yidan Wang, Yubing Ren, Yanan Cao, Binxing Fang

The rise of Large Language Models (LLMs) has heightened concerns about the
misuse of AI-generated text, making watermarking a promising solution.
Mainstream watermarking schemes for LLMs fall into two categories: logits-based
and sampling-based. However, current schemes entail trade-offs among
robustness, text quality, and security. To mitigate this, we integrate
logits-based and sampling-based schemes, harnessing their respective strengths
to achieve synergy. In this paper, we propose a versatile symbiotic
watermarking framework with three strategies: serial, parallel, and hybrid. The
hybrid framework adaptively embeds watermarks using token entropy and semantic
entropy, optimizing the balance between detectability, robustness, text
quality, and security. Furthermore, we validate our approach through
comprehensive experiments on various datasets and models. Experimental results
indicate that our method outperforms existing baselines and achieves
state-of-the-art (SOTA) performance. We believe this framework provides novel
insights into diverse watermarking paradigms. Our code is available at
\href{https://github.com/redwyd/SymMark}{https://github.com/redwyd/SymMark}.

### Computer Vision and Pattern Recognition

### 1. [DDFP: Data-dependent Frequency Prompt for Source Free Domain Adaptation of Medical Image Segmentation](http://arxiv.org/pdf/2505.09927v1)

Authors: Siqi Yin, Shaolei Liu, Manning Wang

Domain adaptation addresses the challenge of model performance degradation
caused by domain gaps. In the typical setup for unsupervised domain adaptation,
labeled data from a source domain and unlabeled data from a target domain are
used to train a target model. However, access to labeled source domain data,
particularly in medical datasets, can be restricted due to privacy policies. As
a result, research has increasingly shifted to source-free domain adaptation
(SFDA), which requires only a pretrained model from the source domain and
unlabeled data from the target domain data for adaptation. Existing SFDA
methods often rely on domain-specific image style translation and
self-supervision techniques to bridge the domain gap and train the target
domain model. However, the quality of domain-specific style-translated images
and pseudo-labels produced by these methods still leaves room for improvement.
Moreover, training the entire model during adaptation can be inefficient under
limited supervision. In this paper, we propose a novel SFDA framework to
address these challenges. Specifically, to effectively mitigate the impact of
domain gap in the initial training phase, we introduce preadaptation to
generate a preadapted model, which serves as an initialization of target model
and allows for the generation of high-quality enhanced pseudo-labels without
introducing extra parameters. Additionally, we propose a data-dependent
frequency prompt to more effectively translate target domain images into a
source-like style. To further enhance adaptation, we employ a style-related
layer fine-tuning strategy, specifically designed for SFDA, to train the target
model using the prompted target domain images and pseudo-labels. Extensive
experiments on cross-modality abdominal and cardiac SFDA segmentation tasks
demonstrate that our proposed method outperforms existing state-of-the-art
methods.

### 2. [CSPENet: Contour-Aware and Saliency Priors Embedding Network for Infrared Small Target Detection](http://arxiv.org/pdf/2505.09943v1)

Authors: Jiakun Deng, Kexuan Li, Xingye Cui, Jiaxuan Li, Chang Long, Tian Pu, Zhenming Peng

Infrared small target detection (ISTD) plays a critical role in a wide range
of civilian and military applications. Existing methods suffer from
deficiencies in the localization of dim targets and the perception of contour
information under dense clutter environments, severely limiting their detection
performance. To tackle these issues, we propose a contour-aware and saliency
priors embedding network (CSPENet) for ISTD. We first design a
surround-convergent prior extraction module (SCPEM) that effectively captures
the intrinsic characteristic of target contour pixel gradients converging
toward their center. This module concurrently extracts two collaborative
priors: a boosted saliency prior for accurate target localization and
multi-scale structural priors for comprehensively enriching contour detail
representation. Building upon this, we propose a dual-branch priors embedding
architecture (DBPEA) that establishes differentiated feature fusion pathways,
embedding these two priors at optimal network positions to achieve performance
enhancement. Finally, we develop an attention-guided feature enhancement module
(AGFEM) to refine feature representations and improve saliency estimation
accuracy. Experimental results on public datasets NUDT-SIRST, IRSTD-1k, and
NUAA-SIRST demonstrate that our CSPENet outperforms other state-of-the-art
methods in detection performance. The code is available at
https://github.com/IDIP2025/CSPENet.

### 3. [MambaControl: Anatomy Graph-Enhanced Mamba ControlNet with Fourier Refinement for Diffusion-Based Disease Trajectory Prediction](http://arxiv.org/pdf/2505.09965v1)

Authors: Hao Yang, Tao Tan, Shuai Tan, Weiqin Yang, Kunyan Cai, Calvin Chen, Yue Sun

Modelling disease progression in precision medicine requires capturing
complex spatio-temporal dynamics while preserving anatomical integrity.
Existing methods often struggle with longitudinal dependencies and structural
consistency in progressive disorders. To address these limitations, we
introduce MambaControl, a novel framework that integrates selective state-space
modelling with diffusion processes for high-fidelity prediction of medical
image trajectories. To better capture subtle structural changes over time while
maintaining anatomical consistency, MambaControl combines Mamba-based
long-range modelling with graph-guided anatomical control to more effectively
represent anatomical correlations. Furthermore, we introduce Fourier-enhanced
spectral graph representations to capture spatial coherence and multiscale
detail, enabling MambaControl to achieve state-of-the-art performance in
Alzheimer's disease prediction. Quantitative and regional evaluations
demonstrate improved progression prediction quality and anatomical fidelity,
highlighting its potential for personalised prognosis and clinical decision
support.

### 4. [TKFNet: Learning Texture Key Factor Driven Feature for Facial Expression Recognition](http://arxiv.org/pdf/2505.09967v1)

Authors: Liqian Deng

Facial expression recognition (FER) in the wild remains a challenging task
due to the subtle and localized nature of expression-related features, as well
as the complex variations in facial appearance. In this paper, we introduce a
novel framework that explicitly focuses on Texture Key Driver Factors (TKDF),
localized texture regions that exhibit strong discriminative power across
emotional categories. By carefully observing facial image patterns, we identify
that certain texture cues, such as micro-changes in skin around the brows,
eyes, and mouth, serve as primary indicators of emotional dynamics. To
effectively capture and leverage these cues, we propose a FER architecture
comprising a Texture-Aware Feature Extractor (TAFE) and Dual Contextual
Information Filtering (DCIF). TAFE employs a ResNet-based backbone enhanced
with multi-branch attention to extract fine-grained texture representations,
while DCIF refines these features by filtering context through adaptive pooling
and attention mechanisms. Experimental results on RAF-DB and KDEF datasets
demonstrate that our method achieves state-of-the-art performance, verifying
the effectiveness and robustness of incorporating TKDFs into FER pipelines.

### 5. [APCoTTA: Continual Test-Time Adaptation for Semantic Segmentation of Airborne LiDAR Point Clouds](http://arxiv.org/pdf/2505.09971v1)

Authors: Yuan Gao, Shaobo Xia, Sheng Nie, Cheng Wang, Xiaohuan Xi, Bisheng Yang

Airborne laser scanning (ALS) point cloud segmentation is a fundamental task
for large-scale 3D scene understanding. In real-world applications, models are
typically fixed after training. However, domain shifts caused by changes in the
environment, sensor types, or sensor degradation often lead to a decline in
model performance. Continuous Test-Time Adaptation (CTTA) offers a solution by
adapting a source-pretrained model to evolving, unlabeled target domains.
Despite its potential, research on ALS point clouds remains limited, facing
challenges such as the absence of standardized datasets and the risk of
catastrophic forgetting and error accumulation during prolonged adaptation. To
tackle these challenges, we propose APCoTTA, the first CTTA method tailored for
ALS point cloud semantic segmentation. We propose a dynamic trainable layer
selection module. This module utilizes gradient information to select
low-confidence layers for training, and the remaining layers are kept frozen,
mitigating catastrophic forgetting. To further reduce error accumulation, we
propose an entropy-based consistency loss. By losing such samples based on
entropy, we apply consistency loss only to the reliable samples, enhancing
model stability. In addition, we propose a random parameter interpolation
mechanism, which randomly blends parameters from the selected trainable layers
with those of the source model. This approach helps balance target adaptation
and source knowledge retention, further alleviating forgetting. Finally, we
construct two benchmarks, ISPRSC and H3DC, to address the lack of CTTA
benchmarks for ALS point cloud segmentation. Experimental results demonstrate
that APCoTTA achieves the best performance on two benchmarks, with mIoU
improvements of approximately 9% and 14% over direct inference. The new
benchmarks and code are available at https://github.com/Gaoyuan2/APCoTTA.

### 6. [PointArena: Probing Multimodal Grounding Through Language-Guided Pointing](http://arxiv.org/pdf/2505.09990v1)

Authors: Long Cheng, Jiafei Duan, Yi Ru Wang, Haoquan Fang, Boyang Li, Yushan Huang, Elvis Wang, Ainaz Eftekhar, Jason Lee, Wentao Yuan, Rose Hendrix, Noah A. Smith, Fei Xia, Dieter Fox, Ranjay Krishna

Pointing serves as a fundamental and intuitive mechanism for grounding
language within visual contexts, with applications spanning robotics, assistive
technologies, and interactive AI systems. While recent multimodal models have
started to support pointing capabilities, existing benchmarks typically focus
only on referential object localization tasks. We introduce PointArena, a
comprehensive platform for evaluating multimodal pointing across diverse
reasoning scenarios. PointArena comprises three components: (1) Point-Bench, a
curated dataset containing approximately 1,000 pointing tasks across five
reasoning categories; (2) Point-Battle, an interactive, web-based arena
facilitating blind, pairwise model comparisons, which has already gathered over
4,500 anonymized votes; and (3) Point-Act, a real-world robotic manipulation
system allowing users to directly evaluate multimodal model pointing
capabilities in practical settings. We conducted extensive evaluations of both
state-of-the-art open-source and proprietary multimodal models. Results
indicate that Molmo-72B consistently outperforms other models, though
proprietary models increasingly demonstrate comparable performance.
Additionally, we find that supervised training specifically targeting pointing
tasks significantly enhances model performance. Across our multi-stage
evaluation pipeline, we also observe strong correlations, underscoring the
critical role of precise pointing capabilities in enabling multimodal models to
effectively bridge abstract reasoning with concrete, real-world actions.
Project page: https://pointarena.github.io/

### 7. [Descriptive Image-Text Matching with Graded Contextual Similarity](http://arxiv.org/pdf/2505.09997v1)

Authors: Jinhyun Jang, Jiyeong Lee, Kwanghoon Sohn

Image-text matching aims to build correspondences between visual and textual
data by learning their pairwise similarities. Most existing approaches have
adopted sparse binary supervision, indicating whether a pair of images and
sentences matches or not. However, such sparse supervision covers a limited
subset of image-text relationships, neglecting their inherent many-to-many
correspondences; an image can be described in numerous texts at different
descriptive levels. Moreover, existing approaches overlook the implicit
connections from general to specific descriptions, which form the underlying
rationale for the many-to-many relationships between vision and language. In
this work, we propose descriptive image-text matching, called DITM, to learn
the graded contextual similarity between image and text by exploring the
descriptive flexibility of language. We formulate the descriptiveness score of
each sentence with cumulative term frequency-inverse document frequency
(TF-IDF) to balance the pairwise similarity according to the keywords in the
sentence. Our method leverages sentence descriptiveness to learn robust
image-text matching in two key ways: (1) to refine the false negative labeling,
dynamically relaxing the connectivity between positive and negative pairs, and
(2) to build more precise matching, aligning a set of relevant sentences in a
generic-to-specific order. By moving beyond rigid binary supervision, DITM
enhances the discovery of both optimal matches and potential positive pairs.
Extensive experiments on MS-COCO, Flickr30K, and CxC datasets demonstrate the
effectiveness of our method in representing complex image-text relationships
compared to state-of-the-art approaches. In addition, DITM enhances the
hierarchical reasoning ability of the model, supported by the extensive
analysis on HierarCaps benchmark.

### 8. [From Air to Wear: Personalized 3D Digital Fashion with AR/VR Immersive 3D Sketching](http://arxiv.org/pdf/2505.09998v1)

Authors: Ying Zang, Yuanqi Hu, Xinyu Chen, Yuxia Xu, Suhui Wang, Chunan Yu, Lanyun Zhu, Deyi Ji, Xin Xu, Tianrun Chen

In the era of immersive consumer electronics, such as AR/VR headsets and
smart devices, people increasingly seek ways to express their identity through
virtual fashion. However, existing 3D garment design tools remain inaccessible
to everyday users due to steep technical barriers and limited data. In this
work, we introduce a 3D sketch-driven 3D garment generation framework that
empowers ordinary users - even those without design experience - to create
high-quality digital clothing through simple 3D sketches in AR/VR environments.
By combining a conditional diffusion model, a sketch encoder trained in a
shared latent space, and an adaptive curriculum learning strategy, our system
interprets imprecise, free-hand input and produces realistic, personalized
garments. To address the scarcity of training data, we also introduce
KO3DClothes, a new dataset of paired 3D garments and user-created sketches.
Extensive experiments and user studies confirm that our method significantly
outperforms existing baselines in both fidelity and usability, demonstrating
its promise for democratized fashion design on next-generation consumer
platforms.

### 9. [Exploring the Deep Fusion of Large Language Models and Diffusion Transformers for Text-to-Image Synthesis](http://arxiv.org/pdf/2505.10046v1)

Authors: Bingda Tang, Boyang Zheng, Xichen Pan, Sayak Paul, Saining Xie

This paper does not describe a new method; instead, it provides a thorough
exploration of an important yet understudied design space related to recent
advances in text-to-image synthesis -- specifically, the deep fusion of large
language models (LLMs) and diffusion transformers (DiTs) for multi-modal
generation. Previous studies mainly focused on overall system performance
rather than detailed comparisons with alternative methods, and key design
details and training recipes were often left undisclosed. These gaps create
uncertainty about the real potential of this approach. To fill these gaps, we
conduct an empirical study on text-to-image generation, performing controlled
comparisons with established baselines, analyzing important design choices, and
providing a clear, reproducible recipe for training at scale. We hope this work
offers meaningful data points and practical guidelines for future research in
multi-modal generation.

### 10. [Advances in Radiance Field for Dynamic Scene: From Neural Field to Gaussian Field](http://arxiv.org/pdf/2505.10049v1)

Authors: Jinlong Fan, Xuepu Zeng, Jing Zhang, Mingming Gong, Yuxiang Yang, Dacheng Tao

Dynamic scene representation and reconstruction have undergone transformative
advances in recent years, catalyzed by breakthroughs in neural radiance fields
and 3D Gaussian splatting techniques. While initially developed for static
environments, these methodologies have rapidly evolved to address the
complexities inherent in 4D dynamic scenes through an expansive body of
research. Coupled with innovations in differentiable volumetric rendering,
these approaches have significantly enhanced the quality of motion
representation and dynamic scene reconstruction, thereby garnering substantial
attention from the computer vision and graphics communities. This survey
presents a systematic analysis of over 200 papers focused on dynamic scene
representation using radiance field, spanning the spectrum from implicit neural
representations to explicit Gaussian primitives. We categorize and evaluate
these works through multiple critical lenses: motion representation paradigms,
reconstruction techniques for varied scene dynamics, auxiliary information
integration strategies, and regularization approaches that ensure temporal
consistency and physical plausibility. We organize diverse methodological
approaches under a unified representational framework, concluding with a
critical examination of persistent challenges and promising research
directions. By providing this comprehensive overview, we aim to establish a
definitive reference for researchers entering this rapidly evolving field while
offering experienced practitioners a systematic understanding of both
conceptual principles and practical frontiers in dynamic scene reconstruction.

### Computers and Society

### 1. [To what extent can current French mobile network support agricultural robots?](http://arxiv.org/pdf/2505.10044v1)

Authors: Pierre La Rocca, Gaël Guennebaud, Aurélie Bugeau

The large-scale integration of robots in agriculture offers many promises for
enhancing sustainability and increasing food production. The numerous
applications of agricultural robots rely on the transmission of data via mobile
network, with the amount of data depending on the services offered by the
robots and the level of on-board technology. Nevertheless, infrastructure
required to deploy these robots, as well as the related energy and
environmental consequences, appear overlooked in the digital agriculture
literature. In this study, we propose a method for assessing the additional
energy consumption and carbon footprint induced by a large-scale deployment of
agricultural robots. Our method also estimates the share of agricultural area
that can be managed by the deployed robots with respect to network
infrastructure constraints. We have applied this method to metropolitan France
mobile network and agricultural parcels for five different robotic scenarios.
Our results show that increasing the robot's bitrate needs leads to significant
additional impacts, which increase at a pace that is poorly captured by
classical linear extrapolation methods. When constraining the network to the
existing sites, increased bitrate needs also comes with a rapidly decreasing
manageable agricultural area.

### 2. [Top-Down vs. Bottom-Up Approaches for Automatic Educational Knowledge Graph Construction in CourseMapper](http://arxiv.org/pdf/2505.10069v1)

Authors: Qurat Ul Ain, Mohamed Amine Chatti, Amr Shakhshir, Jean Qussa, Rawaa Alatrash, Shoeb Joarder

The automatic construction of Educational Knowledge Graphs (EduKGs) is
crucial for modeling domain knowledge in digital learning environments,
particularly in Massive Open Online Courses (MOOCs). However, identifying the
most effective approach for constructing accurate EduKGs remains a challenge.
This study compares Top-down and Bottom-up approaches for automatic EduKG
construction, evaluating their effectiveness in capturing and structuring
knowledge concepts from learning materials in our MOOC platform CourseMapper.
Through a user study and expert validation using Simple Random Sampling (SRS),
results indicate that the Bottom-up approach outperforms the Top-down approach
in accurately identifying and mapping key knowledge concepts. To further
enhance EduKG accuracy, we integrate a Human-in-the-Loop approach, allowing
course moderators to review and refine the EduKG before publication. This
structured comparison provides a scalable framework for improving knowledge
representation in MOOCs, ultimately supporting more personalized and adaptive
learning experiences.

### 3. [Formalising Human-in-the-Loop: Computational Reductions, Failure Modes, and Legal-Moral Responsibility](http://arxiv.org/pdf/2505.10426v1)

Authors: Maurice Chiodo, Dennis Müller, Paul Siewert, Jean-Luc Wetherall, Zoya Yasmine, John Burden

The legal compliance and safety of different Human-in-the-loop (HITL) setups
for AI can vary greatly. This manuscript aims to identify new ways of choosing
between such setups, and shows that there is an unavoidable trade-off between
the attribution of legal responsibility and the technical explainability of AI.
We begin by using the notion of oracle machines from computability theory to
formalise different HITL setups, distinguishing between trivial human
monitoring, single endpoint human action, and highly involved interaction
between the human(s) and the AI. These correspond to total functions, many-one
reductions, and Turing reductions respectively. A taxonomy categorising HITL
failure modes is then presented, highlighting the limitations on what any HITL
setup can actually achieve. Our approach then identifies oversights from UK and
EU legal frameworks, which focus on certain HITL setups which may not always
achieve the desired ethical, legal, and sociotechnical outcomes. We suggest
areas where the law should recognise the effectiveness of different HITL setups
and assign responsibility in these contexts, avoiding unnecessary and
unproductive human "scapegoating". Overall, we show how HITL setups involve
many technical design decisions, and can be prone to failures which are often
out of the humans' control. This opens up a new analytic perspective on the
challenges arising in the creation of HITL setups, helping inform AI developers
and lawmakers on designing HITL to better achieve their desired outcomes.

### 4. [Campus AI vs Commercial AI: A Late-Breaking Study on How LLM As-A-Service Customizations Shape Trust and Usage Patterns](http://arxiv.org/pdf/2505.10490v1)

Authors: Leon Hannig, Annika Bush, Meltem Aksoy, Steffen Becker, Greta Ontrup

As the use of Large Language Models (LLMs) by students, lecturers and
researchers becomes more prevalent, universities - like other organizations -
are pressed to develop coherent AI strategies. LLMs as-a-Service (LLMaaS) offer
accessible pre-trained models, customizable to specific (business) needs. While
most studies prioritize data, model, or infrastructure adaptations (e.g., model
fine-tuning), we focus on user-salient customizations, like interface changes
and corporate branding, which we argue influence users' trust and usage
patterns. This study serves as a functional prequel to a large-scale field
study in which we examine how students and employees at a German university
perceive and use their institution's customized LLMaaS compared to ChatGPT. The
goals of this prequel are to stimulate discussions on psychological effects of
LLMaaS customizations and refine our research approach through feedback. Our
forthcoming findings will deepen the understanding of trust dynamics in LLMs,
providing practical guidance for organizations considering LLMaaS deployment.

### 5. [Determining Absence of Unreasonable Risk: Approval Guidelines for an Automated Driving System Release](http://arxiv.org/pdf/2505.09880v1)

Authors: Francesca Favaro, Scott Schnelle, Laura Fraade-Blanar, Trent Victor, Mauricio Peña, Nick Webb, Holland Broce, Craig Paterson, Dan Smith

This paper provides an overview of how the determination of absence of
unreasonable risk can be operationalized. It complements previous theoretical
work published by existing developers of Automated Driving Systems (ADS) on the
overall engineering practices and methodologies for readiness determination.
Readiness determination is, at its core, a risk assessment process. It is aimed
at evaluating the residual risk associated with the deployment of a new
software release candidate. The paper proposes methodological criteria to
ground the readiness review process for an ADS release. While informed by
Waymo's experience in this domain, the criteria presented are agnostic of any
specific ADS technological solution and/or architectural choice, to support
broad implementation by others in the industry. The paper continues with a
discussion on governance and decision-making toward approval of a new software
release candidate for the ADS. The implementation of the presented criteria
requires the existence of appropriate safety management practices in addition
to many other cultural, procedural, and operational considerations. As such,
the paper is concluded by a statement of limitations for those wishing to
replicate part or all of its content.

### 6. [Leveraging Graph Retrieval-Augmented Generation to Support Learners' Understanding of Knowledge Concepts in MOOCs](http://arxiv.org/pdf/2505.10074v1)

Authors: Mohamed Abdelmagied, Mohamed Amine Chatti, Shoeb Joarder, Qurat Ul Ain, Rawaa Alatrash

Massive Open Online Courses (MOOCs) lack direct interaction between learners
and instructors, making it challenging for learners to understand new knowledge
concepts. Recently, learners have increasingly used Large Language Models
(LLMs) to support them in acquiring new knowledge. However, LLMs are prone to
hallucinations which limits their reliability. Retrieval-Augmented Generation
(RAG) addresses this issue by retrieving relevant documents before generating a
response. However, the application of RAG across different MOOCs is limited by
unstructured learning material. Furthermore, current RAG systems do not
actively guide learners toward their learning needs. To address these
challenges, we propose a Graph RAG pipeline that leverages Educational
Knowledge Graphs (EduKGs) and Personal Knowledge Graphs (PKGs) to guide
learners to understand knowledge concepts in the MOOC platform CourseMapper.
Specifically, we implement (1) a PKG-based Question Generation method to
recommend personalized questions for learners in context, and (2) an
EduKG-based Question Answering method that leverages the relationships between
knowledge concepts in the EduKG to answer learner selected questions. To
evaluate both methods, we conducted a study with 3 expert instructors on 3
different MOOCs in the MOOC platform CourseMapper. The results of the
evaluation show the potential of Graph RAG to empower learners to understand
new knowledge concepts in a personalized learning experience.

### 7. [Digital Natives, Digital Activists: Youth, Social Media and the Rise of Environmental Sustainability Movements](http://arxiv.org/pdf/2505.10158v1)

Authors: Manya Pandit, Triveni Magadum, Harshit Mittal, Omkar Kushwaha

The research examines the challenges revolving around young people's social
movements, activism regarding sustainability, as well as the accompanying
social media aspect, and how social media impacts environmental action. This
study focuses on the environmental craze on social media platforms and its
impact on young activists aged 16-25. With the advancement of social media, new
avenues have opened for participation in sustainability issues, especially for
the marginalized, as information moved through transnational networks at
lightning speed. Along with specific Formative Visual Storytelling methods, the
young leaders of the movement deploy hashtags and other online tools to capture
the attention of their peers and decision makers. Challenges persist with
"clicktivism" fatigue from the internet, and site limitations. This article
contributes to insights on emerging forms of civic activism by explaining how
digital natives adapt technology to reframe green activism. The research
suggests that effective digital environmental movements integrate online and
offline action, make it simple for individuals to get involved, and promote
tolerance to algorithmic modifications and climate care among participants.

### 8. [Lost in Models? Structuring Managerial Decision Support in Process Mining with Multi-criteria Decision Making](http://arxiv.org/pdf/2505.10236v1)

Authors: Rob H. Bemthuis

Process mining is increasingly adopted in modern organizations, producing
numerous process models that, while valuable, can lead to model overload and
decision-making complexity. This paper explores a multi-criteria
decision-making (MCDM) approach to evaluate and prioritize process models by
incorporating both quantitative metrics (e.g., fitness, precision) and
qualitative factors (e.g., cultural fit). An illustrative logistics example
demonstrates how MCDM, specifically the Analytic Hierarchy Process (AHP),
facilitates trade-off analysis and promotes alignment with managerial
objectives. Initial insights suggest that the MCDM approach enhances
context-sensitive decision-making, as selected models address both operational
metrics and broader managerial needs. While this study is an early-stage
exploration, it provides an initial foundation for deeper exploration of
MCDM-driven strategies to enhance the role of process mining in complex
organizational settings.

### 9. [Which Demographic Features Are Relevant for Individual Fairness Evaluation of U.S. Recidivism Risk Assessment Tools?](http://arxiv.org/pdf/2505.09868v1)

Authors: Tin Trung Nguyen, Jiannan Xu, Phuong-Anh Nguyen-Le, Jonathan Lazar, Donald Braman, Hal Daumé III, Zubin Jelveh

Despite its U.S. constitutional foundation, the technical ``individual
fairness'' criterion has not been operationalized in state or federal
statutes/regulations. We conduct a human subjects experiment to address this
gap, evaluating which demographic features are relevant for individual fairness
evaluation of recidivism risk assessment (RRA) tools. Our analyses conclude
that the individual similarity function should consider age and sex, but it
should ignore race.

### Databases

### 1. [Approximation-First Timeseries Monitoring Query At Scale](http://arxiv.org/pdf/2505.10560v1)

Authors: Zeying Zhu, Jonathan Chamberlain, Kenny Wu, David Starobinski, Zaoxing Liu

Timeseries monitoring systems such as Prometheus play a crucial role in
gaining observability of the underlying system components. These systems
collect timeseries metrics from various system components and perform
monitoring queries over periodic window-based aggregations (i.e., rule
queries). However, despite wide adoption, the operational costs and query
latency of rule queries remain high. In this paper, we identify major
bottlenecks associated with repeated data scans and query computations
concerning window overlaps in rule queries, and present PromSketch, an
approximation-first query framework as intermediate caches for monitoring
systems. It enables low operational costs and query latency, by combining
approximate window-based query frameworks and sketch-based precomputation.
PromSketch is implemented as a standalone module that can be integrated into
Prometheus and VictoriaMetrics, covering 70% of Prometheus' aggregation over
time queries. Our evaluation shows that PromSketch achieves up to a two orders
of magnitude reduction in query latency over Prometheus and VictoriaMetrics,
while lowering operational dollar costs of query processing by two orders of
magnitude compared to Prometheus and by at least 4x compared to VictoriaMetrics
with at most 5% average errors across statistics. The source code has been made
available at https://github.com/Froot-NetSys/promsketch.

### 2. [Lost in Models? Structuring Managerial Decision Support in Process Mining with Multi-criteria Decision Making](http://arxiv.org/pdf/2505.10236v1)

Authors: Rob H. Bemthuis

Process mining is increasingly adopted in modern organizations, producing
numerous process models that, while valuable, can lead to model overload and
decision-making complexity. This paper explores a multi-criteria
decision-making (MCDM) approach to evaluate and prioritize process models by
incorporating both quantitative metrics (e.g., fitness, precision) and
qualitative factors (e.g., cultural fit). An illustrative logistics example
demonstrates how MCDM, specifically the Analytic Hierarchy Process (AHP),
facilitates trade-off analysis and promotes alignment with managerial
objectives. Initial insights suggest that the MCDM approach enhances
context-sensitive decision-making, as selected models address both operational
metrics and broader managerial needs. While this study is an early-stage
exploration, it provides an initial foundation for deeper exploration of
MCDM-driven strategies to enhance the role of process mining in complex
organizational settings.

### 3. [Inconsistency Handling in DatalogMTL](http://arxiv.org/pdf/2505.10394v1)

Authors: Meghyn Bienvenu, Camille Bourgaux, Atefe Khodadaditaghanaki

In this paper, we explore the issue of inconsistency handling in DatalogMTL,
an extension of Datalog with metric temporal operators. Since facts are
associated with time intervals, there are different manners to restore
consistency when they contradict the rules, such as removing facts or modifying
their time intervals. Our first contribution is the definition of relevant
notions of conflicts (minimal explanations for inconsistency) and repairs
(possible ways of restoring consistency) for this setting and the study of the
properties of these notions and the associated inconsistency-tolerant
semantics. Our second contribution is a data complexity analysis of the tasks
of generating a single conflict / repair and query entailment under
repair-based semantics.

### Distributed, Parallel, and Cluster Computing

### 1. [ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production](http://arxiv.org/pdf/2505.09999v1)

Authors: Yuxing Xiang, Xue Li, Kun Qian, Wenyuan Yu, Ennan Zhai, Xin Jin

With the widespread adoption of Large Language Models (LLMs), serving LLM
inference requests has become an increasingly important task, attracting active
research advancements. Practical workloads play an essential role in this
process: they are critical for motivating and benchmarking serving techniques
and systems. However, the existing understanding of real-world LLM serving
workloads is limited due to the lack of a comprehensive workload
characterization. Prior analyses remain insufficient in scale and scope, thus
failing to fully capture intricate workload characteristics.
  In this paper, we fill the gap with an in-depth characterization of LLM
serving workloads collected from our worldwide cloud inference serving service,
covering not only language models but also emerging multimodal and reasoning
models, and unveiling important new findings in each case. Moreover, based on
our findings, we propose ServeGen, a principled framework for generating
realistic LLM serving workloads by composing them on a per-client basis. A
practical use case in production validates that ServeGen avoids 50%
under-provisioning compared to naive workload generation, demonstrating
ServeGen's advantage in performance benchmarking. We will open-source ServeGen
to foster future research.

### 2. [A categorical and logical framework for iterated protocols](http://arxiv.org/pdf/2505.10071v1)

Authors: Eric Goubault, Bernardo Hummes Flores, Roman Kniazev, Jeremy Ledent, Sergio Rajsbaum

In this article, we show that the now classical protocol complex approach to
distributed task solvability of Herlihy et al. can be understood in standard
categorical terms. First, protocol complexes are functors, from chromatic
(semi-) simplicial sets to chromatic simplicial sets, that naturally give rise
to algebras. These algebras describe the next state operator for the
corresponding distributed systems. This is constructed for semi-synchronous
distributed systems with general patterns of communication for which we show
that these functors are always Yoneda extensions of simpler functors, implying
a number of interesting properties. Furthermore, for these protocol complex
functors, we prove the existence of a free algebra on any initial chromatic
simplicial complex, modeling iterated protocol complexes. Under this
categorical formalization, protocol complexes are seen as transition systems,
where states are structured as chromatic simplicial sets. We exploit the
epistemic interpretation of chromatic simplicial sets and the underlying
transition system (or algebra) structure to introduce a temporal-epistemic
logic and its semantics on all free algebras on chromatic simplicial sets. We
end up by giving hints on how to extend this framework to more general dynamic
network graphs and state-dependent protocols, and give example in
fault-tolerant distributed systems and mobile robotics.

### 3. [KAITIAN: A Unified Communication Framework for Enabling Efficient Collaboration Across Heterogeneous Accelerators in Embodied AI Systems](http://arxiv.org/pdf/2505.10183v1)

Authors: Jieke Lin, Wanyu Wang, Longxiang Yin, Yinhe Han

Embodied Artificial Intelligence (AI) systems, such as autonomous robots and
intelligent vehicles, are increasingly reliant on diverse heterogeneous
accelerators (e.g., GPGPUs, NPUs, FPGAs) to meet stringent real-time processing
and energy-efficiency demands. However, the proliferation of vendor-specific
proprietary communication libraries creates significant interoperability
barriers, hindering seamless collaboration between different accelerator types
and leading to suboptimal resource utilization and performance bottlenecks in
distributed AI workloads. This paper introduces KAITIAN, a novel distributed
communication framework designed to bridge this gap. KAITIAN provides a unified
abstraction layer that intelligently integrates vendor-optimized communication
libraries for intra-group efficiency with general-purpose communication
protocols for inter-group interoperability. Crucially, it incorporates a
load-adaptive scheduling mechanism that dynamically balances computational
tasks across heterogeneous devices based on their real-time performance
characteristics. Implemented as an extension to PyTorch and rigorously
evaluated on a testbed featuring NVIDIA GPUs and Cambricon MLUs, KAITIAN
demonstrates significant improvements in resource utilization and scalability
for distributed training tasks. Experimental results show that KAITIAN can
accelerate training time by up to 42% compared to baseline homogeneous systems,
while incurring minimal communication overhead (2.8--4.3%) and maintaining
model accuracy. KAITIAN paves the way for more flexible and powerful
heterogeneous computing in complex embodied AI applications.

### 4. [AI Greenferencing: Routing AI Inferencing to Green Modular Data Centers with Heron](http://arxiv.org/pdf/2505.09989v1)

Authors: Tella Rajashekhar Reddy, Palak, Rohan Gandhi, Anjaly Parayil, Chaojie Zhang, Mike Shepperd, Liangcheng Yu, Jayashree Mohan, Srinivasan Iyengar, Shivkumar Kalyanaraman, Debopam Bhattacherjee

AI power demand is growing unprecedentedly thanks to the high power density
of AI compute and the emerging inferencing workload. On the supply side,
abundant wind power is waiting for grid access in interconnection queues. In
this light, this paper argues bringing AI workload to modular compute clusters
co-located in wind farms. Our deployment right-sizing strategy makes it
economically viable to deploy more than 6 million high-end GPUs today that
could consume cheap, green power at its source. We built Heron, a cross-site
software router, that could efficiently leverage the complementarity of power
generation across wind farms by routing AI inferencing workload around power
drops. Using 1-week ofcoding and conversation production traces from Azure and
(real) variable wind power traces, we show how Heron improves aggregate goodput
of AI compute by up to 80% compared to the state-of-the-art.

### Digital Libraries

### 1. [A Survey on Open-Source Edge Computing Simulators and Emulators: The Computing and Networking Convergence Perspective](http://arxiv.org/pdf/2505.09995v1)

Authors: Jianpeng Qi, Chao Liu, Xiao Zhang, Lei Wang, Rui Wang, Junyu Dong, Yanwei Yu

Edge computing, with its low latency, dynamic scalability, and location
awareness, along with the convergence of computing and communication paradigms,
has been successfully applied in critical domains such as industrial IoT, smart
healthcare, smart homes, and public safety. This paper provides a comprehensive
survey of open-source edge computing simulators and emulators, presented in our
GitHub repository (https://github.com/qijianpeng/awesome-edge-computing),
emphasizing the convergence of computing and networking paradigms. By examining
more than 40 tools, including CloudSim, NS-3, and others, we identify the
strengths and limitations in simulating and emulating edge environments. This
survey classifies these tools into three categories: packet-level,
application-level, and emulators. Furthermore, we evaluate them across five
dimensions, ranging from resource representation to resource utilization. The
survey highlights the integration of different computing paradigms, packet
processing capabilities, support for edge environments, user-defined metric
interfaces, and scenario visualization. The findings aim to guide researchers
in selecting appropriate tools for developing and validating advanced computing
and networking technologies.

### Discrete Mathematics

### 1. [How to Color Temporal Graphs to Ensure Proper Transitions](http://arxiv.org/pdf/2505.10207v1)

Authors: Allen Ibiapina, Minh Hang Nguyen, Mikaël Rabie, Cléophée Robin

Graph Coloring consists in assigning colors to vertices ensuring that two
adjacent vertices do not have the same color. In dynamic graphs, this notion is
not well defined, as we need to decide if different colors for adjacent
vertices must happen all the time or not, and how to go from a coloring in one
time to the next one.
  In this paper, we define a coloring notion for Temporal Graphs where at each
step, the coloring must be proper. It uses a notion of compatibility between
two consecutive snapshots that implies that the coloring stays proper while the
transition happens. Given a graph, the minimum number of colors needed to
ensure that such coloring exists is the \emph{Temporal Chromatic Number} of
this graph.
  With those notions, we provide some lower and upper bounds for the temporal
chromatic number in the general case. We then dive into some specific classes
of graphs such as trees, graphs with bounded degree or bounded degeneracy.
Finally, we consider temporal graphs where grow pace is one, that is, a single
edge can be added and a single other one can be removed between two time steps.
In that case, we consider bipartite and bounded degree graphs.
  Even though the problem is defined with full knowledge of the temporal graph,
our results also work in the case where future snapshots are given online: we
need to choose the coloring of the next snapshot after having computed the
current one, not knowing what

### Data Structures and Algorithms

### 1. [Improved Rank Aggregation under Fairness Constraint](http://arxiv.org/pdf/2505.10006v1)

Authors: Alvin Hong Yao Yan, Diptarka Chakraborty, Himika Das, Sanjana Dey

Aggregating multiple input rankings into a consensus ranking is essential in
various fields such as social choice theory, hiring, college admissions, web
search, and databases. A major challenge is that the optimal consensus ranking
might be biased against individual candidates or groups, especially those from
marginalized communities. This concern has led to recent studies focusing on
fairness in rank aggregation. The goal is to ensure that candidates from
different groups are fairly represented in the top-$k$ positions of the
aggregated ranking.
  We study this fair rank aggregation problem by considering the Kendall tau as
the underlying metric. While we know of a polynomial-time approximation scheme
(PTAS) for the classical rank aggregation problem, the corresponding fair
variant only possesses a quite straightforward 3-approximation algorithm due to
Wei et al., SIGMOD'22, and Chakraborty et al., NeurIPS'22, which finds closest
fair ranking for each input ranking and then simply outputs the best one.
  In this paper, we first provide a novel algorithm that achieves
$(2+\epsilon)$-approximation (for any $\epsilon > 0$), significantly improving
over the 3-approximation bound. Next, we provide a $2.881$-approximation fair
rank aggregation algorithm that works irrespective of the fairness notion,
given one can find a closest fair ranking, beating the 3-approximation bound.
We complement our theoretical guarantee by performing extensive experiments on
various real-world datasets to establish the effectiveness of our algorithm
further by comparing it with the performance of state-of-the-art algorithms.

### 2. [Simpler and Faster Directed Low-Diameter Decompositions](http://arxiv.org/pdf/2505.10244v1)

Authors: Jason Li

We present a simpler and faster algorithm for low-diameter decompositions on
directed graphs, matching the $O(\log m\log\log m)$ loss factor from Bringmann,
Fischer, Haeupler, and Latypov (ICALP 2025) and improving the running time to
$O((m+n\log\log n)\log^2m\log\log m)$.

### 3. [Price of Anarchy for Congestion and Scheduling Games via Vector Fitting](http://arxiv.org/pdf/2505.10082v1)

Authors: Danish Kashaev

We provide a dual fitting technique on a semidefinite program yielding simple
proofs of tight bounds for the robust price of anarchy of several congestion
and scheduling games under the sum of weighted completion times objective. The
same approach also allows to bound the approximation ratio of local search
algorithms for the scheduling problem $R || \sum w_j C_j$. All of our results
are obtained through a simple unified dual fitting argument on the same
semidefinite programming relaxation, which can essentially be obtained through
the first round of the Lasserre/Sum of Squares hierarchy.
  As our main application, we show that the known coordination ratio bounds of
respectively $4, (3 + \sqrt{5})/2 \approx 2.618,$ and $32/15 \approx 2.133$ for
the scheduling game $R || \sum w_j C_j$ under the coordination mechanisms
Smith's Rule, Proportional Sharing and Rand (STOC 2011) can be extended to
congestion games and obtained through this approach. For the natural
restriction where the weight of each player is proportional to its processing
time on every resource, we show that the last bound can be improved from 2.133
to 2. This improvement can also be made for general instances when considering
the price of anarchy of the game, rather than the coordination ratio. As a
further application of the technique, we show that it recovers the tight bound
of $(3 + \sqrt{5})/2$ for the price of anarchy of weighted affine congestion
games and the Kawaguchi-Kyan bound of $(1+ \sqrt{2})/2$ for the pure price of
anarchy of $P || \sum w_j C_j$. In addition, this approach recovers the known
tight approximation ratio of $(3 + \sqrt{5})/2 \approx 2.618$ for a natural
local search algorithm for $R || \sum w_j C_j$, as well as the best currently
known combinatorial approximation algorithm for this problem achieving an
approximation ratio of $(5 + \sqrt{5})/4 + \varepsilon \approx 1.809 +
\varepsilon$.

### Emerging Technologies

### 1. [Scalable 28nm IC implementation of coupled oscillator network featuring tunable topology and complexity](http://arxiv.org/pdf/2505.10248v1)

Authors: S. Y. Neyaz, A. Ashok, M. Schiek, C. Grewing, A. Zambanini, S. van Waasen

Integrated circuit implementations of coupled oscillator networks have
recently gained increased attention. The focus is usually on using these
networks for analogue computing, for example for solving computational
optimization tasks. For use within analog computing, these networks are run
close to critical dynamics. On the other hand, such networks are also used as
an analogy of transport networks such as electrical power grids to answer the
question of how exactly such critical dynamic states can be avoided. However,
simulating large network of coupled oscillators is computationally intensive,
with specifc regards to electronic ones. We have developed an integrated
circuit using integrated Phase-Locked Loop (PLL) with modifications, that
allows to flexibly vary the topology as well as a complexity parameter of the
network during operation. The proposed architecture, inspired by the brain,
employs a clustered architecture, with each cluster containing 7 PLLs featuring
programmable coupling mechanisms. Additionally, the inclusion of a RISC-V
processor enables future algorithmic implementations. Thus, we provide a
practical alternative for large-scale network simulations both in the field of
analog computing and transport network stability research.

### 2. [Demystifying AI Agents: The Final Generation of Intelligence](http://arxiv.org/pdf/2505.09932v1)

Authors: Kevin J McNamara, Rhea Pritham Marpu

The trajectory of artificial intelligence (AI) has been one of relentless
acceleration, evolving from rudimentary rule-based systems to sophisticated,
autonomous agents capable of complex reasoning and interaction. This whitepaper
chronicles this remarkable journey, charting the key technological
milestones--advancements in prompting, training methodologies, hardware
capabilities, and architectural innovations--that have converged to create the
AI agents of today. We argue that these agents, exemplified by systems like
OpenAI's ChatGPT with plugins and xAI's Grok, represent a culminating phase in
AI development, potentially constituting the "final generation" of intelligence
as we currently conceive it. We explore the capabilities and underlying
technologies of these agents, grounded in practical examples, while also
examining the profound societal implications and the unprecedented pace of
progress that suggests intelligence is now doubling approximately every six
months. The paper concludes by underscoring the critical need for wisdom and
foresight in navigating the opportunities and challenges presented by this
powerful new era of intelligence.

### 3. [Optimal normalization in quantum-classical hybrid models for anti-cancer drug response prediction](http://arxiv.org/pdf/2505.10037v1)

Authors: Takafumi Ito, Lysenko Artem, Tatsuhiko Tsunoda

Quantum-classical Hybrid Machine Learning (QHML) models are recognized for
their robust performance and high generalization ability even for relatively
small datasets. These qualities offer unique advantages for anti-cancer drug
response prediction, where the number of available samples is typically small.
However, such hybrid models appear to be very sensitive to the data encoding
used at the interface of a neural network and a quantum circuit, with
suboptimal choices leading to stability issues. To address this problem, we
propose a novel strategy that uses a normalization function based on a
moderated gradient version of the $\tanh$. This method transforms the outputs
of the neural networks without concentrating them at the extreme value ranges.
Our idea was evaluated on a dataset of gene expression and drug response
measurements for various cancer cell lines, where we compared the prediction
performance of a classical deep learning model and several QHML models. These
results confirmed that QHML performed better than the classical models when
data was optimally normalized. This study opens up new possibilities for
biomedical data analysis using quantum computers.

### 4. [Unlocking Innate Computing Abilities in Electric Grids](http://arxiv.org/pdf/2505.10382v1)

Authors: Yubo Song, Subham Sahoo

High energy consumption of artificial intelligence has gained momentum
worldwide, which necessitates major investments on expanding efficient and
carbon-neutral generation and data center infrastructure in electric power
grids. Going beyond the conventional ideation, this article unleashes innate
computational abilities in the power grid network circuits itself. By
programming power electronic converters (PECs) to mimic biological neurons, we
sustainably transform power grids into a neural network and enable it to
optimize, compute and make data-driven decisions using distributed PECs.
Instead of seen merely as an energy delivery platform, this article
conceptualizes a novel application for electric grid to be used as a computing
asset without affecting its operation. To illustrate its computational
abilities, we solve a affine transformation task in a microgrid with five PECs.
By encoding the digital data into the control of PECs, our preliminary results
conclude that computing using electric grids does not disturb its operation.
From a scientific perspective, this work fundamentally merges energy and
computing optimization theories by harnessing inherent high-dimensional
computational relationships in electric grids.

### Formal Languages and Automata Theory

### 1. [Büchi-Elgot-Trakhtenbrot Theorem for Higher-Dimensional Automata](http://arxiv.org/pdf/2505.10461v1)

Authors: Amazigh Amrane, Hugo Bazille, Emily Clement, Uli Fahrenberg, Marie Fortin, Krzysztof Ziemiański

In this paper we explore languages of higher-dimensional automata (HDAs) from
an algebraic and logical point of view. Such languages are sets of finite
width-bounded interval pomsets with interfaces (ipomsets) closed under order
extension. We show that ipomsets can be represented as equivalence classes of
words over a particular alphabet, called step sequences. We introduce an
automaton model that recognize such languages. Doing so allows us to lift the
classical B\"uchi-Elgot-Trakhtenbrot Theorem to languages of HDAs: we prove
that a set of interval ipomsets is the language of an HDA if and only if it is
simultaneously MSO-definable, of bounded width, and closed under order
refinement.

### 2. [Probabilistic Bisimulation for Parameterized Anonymity and Uniformity Verification](http://arxiv.org/pdf/2505.09963v1)

Authors: Chih-Duo Hong, Anthony W. Lin, Philipp Rümmer, Rupak Majumdar

Bisimulation is crucial for verifying process equivalence in probabilistic
systems. This paper presents a novel logical framework for analyzing
bisimulation in probabilistic parameterized systems, namely, infinite families
of finite-state probabilistic systems. Our framework is built upon the
first-order theory of regular structures, which provides a decidable logic for
reasoning about these systems. We show that essential properties like anonymity
and uniformity can be encoded and verified within this framework in a manner
aligning with the principles of deductive software verification, where systems,
properties, and proofs are expressed in a unified decidable logic. By
integrating language inference techniques, we achieve full automation in
synthesizing candidate bisimulation proofs for anonymity and uniformity. We
demonstrate the efficacy of our approach by addressing several challenging
examples, including cryptographic protocols and randomized algorithms that were
previously beyond the reach of fully automated methods.

### 3. [Deconstructing Subset Construction -- Reducing While Determinizing](http://arxiv.org/pdf/2505.10319v1)

Authors: John Nicol, Markus Frohme

We present a novel perspective on the NFA canonization problem, which
introduces intermediate minimization steps to reduce the exploration space
on-the-fly. Essential to our approach are so-called equivalence registries
which manage information about equivalent states and allow for incorporating
further optimization techniques such as convexity closures or simulation to
boost performance. Due to the generality of our approach, these concepts can be
embedded in classic subset construction or Brzozowski's approach. We evaluate
our approach on a set of real-world examples from automatic sequences and
observe that we are able to improve especially worst-case scenarios. We
implement our approach in an open-source library for users to experiment with.

### 4. [An algebraic theory of ω-regular languages, via μν-expressions](http://arxiv.org/pdf/2505.10303v1)

Authors: Anupam Das, Abhishek De

Alternating parity automata (APAs) provide a robust formalism for modelling
infinite behaviours and play a central role in formal verification. Despite
their widespread use, the algebraic theory underlying APAs has remained largely
unexplored. In recent work, a notation for non-deterministic finite automata
(NFAs) was introduced, along with a sound and complete axiomatisation of their
equational theory via right-linear algebras. In this paper, we extend that line
of work, in particular to the setting of infinite words. We present a dualised
syntax, yielding a notation for APAs based on right-linear lattice expressions,
and provide a natural axiomatisation of their equational theory with respect to
the standard language model of {\omega}-regular languages. The design of this
axiomatisation is guided by the theory of fixed point logics; in fact, the
completeness factors cleanly through the completeness of the linear-time
{\mu}-calculus.

### Graphics

### 1. [VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality](http://arxiv.org/pdf/2505.10144v1)

Authors: Xuechang Tu, Lukas Radl, Michael Steiner, Markus Steinberger, Bernhard Kerbl, Fernando de la Torre

3D Gaussian Splatting (3DGS) has rapidly become a leading technique for
novel-view synthesis, providing exceptional performance through efficient
software-based GPU rasterization. Its versatility enables real-time
applications, including on mobile and lower-powered devices. However, 3DGS
faces key challenges in virtual reality (VR): (1) temporal artifacts, such as
popping during head movements, (2) projection-based distortions that result in
disturbing and view-inconsistent floaters, and (3) reduced framerates when
rendering large numbers of Gaussians, falling below the critical threshold for
VR. Compared to desktop environments, these issues are drastically amplified by
large field-of-view, constant head movements, and high resolution of
head-mounted displays (HMDs). In this work, we introduce VRSplat: we combine
and extend several recent advancements in 3DGS to address challenges of VR
holistically. We show how the ideas of Mini-Splatting, StopThePop, and Optimal
Projection can complement each other, by modifying the individual techniques
and core 3DGS rasterizer. Additionally, we propose an efficient foveated
rasterizer that handles focus and peripheral areas in a single GPU launch,
avoiding redundant computations and improving GPU utilization. Our method also
incorporates a fine-tuning step that optimizes Gaussian parameters based on
StopThePop depth evaluations and Optimal Projection. We validate our method
through a controlled user study with 25 participants, showing a strong
preference for VRSplat over other configurations of Mini-Splatting. VRSplat is
the first, systematically evaluated 3DGS approach capable of supporting modern
VR applications, achieving 72+ FPS while eliminating popping and
stereo-disrupting floaters.

### 2. [Style Customization of Text-to-Vector Generation with Image Diffusion Priors](http://arxiv.org/pdf/2505.10558v1)

Authors: Peiying Zhang, Nanxuan Zhao, Jing Liao

Scalable Vector Graphics (SVGs) are highly favored by designers due to their
resolution independence and well-organized layer structure. Although existing
text-to-vector (T2V) generation methods can create SVGs from text prompts, they
often overlook an important need in practical applications: style
customization, which is vital for producing a collection of vector graphics
with consistent visual appearance and coherent aesthetics. Extending existing
T2V methods for style customization poses certain challenges.
Optimization-based T2V models can utilize the priors of text-to-image (T2I)
models for customization, but struggle with maintaining structural regularity.
On the other hand, feed-forward T2V models can ensure structural regularity,
yet they encounter difficulties in disentangling content and style due to
limited SVG training data.
  To address these challenges, we propose a novel two-stage style customization
pipeline for SVG generation, making use of the advantages of both feed-forward
T2V models and T2I image priors. In the first stage, we train a T2V diffusion
model with a path-level representation to ensure the structural regularity of
SVGs while preserving diverse expressive capabilities. In the second stage, we
customize the T2V diffusion model to different styles by distilling customized
T2I models. By integrating these techniques, our pipeline can generate
high-quality and diverse SVGs in custom styles based on text prompts in an
efficient feed-forward manner. The effectiveness of our method has been
validated through extensive experiments. The project page is
https://customsvg.github.io.

### 3. [CartoAgent: a multimodal large language model-powered multi-agent cartographic framework for map style transfer and evaluation](http://arxiv.org/pdf/2505.09936v1)

Authors: Chenglong Wang, Yuhao Kang, Zhaoya Gong, Pengjun Zhao, Yu Feng, Wenjia Zhang, Ge Li

The rapid development of generative artificial intelligence (GenAI) presents
new opportunities to advance the cartographic process. Previous studies have
either overlooked the artistic aspects of maps or faced challenges in creating
both accurate and informative maps. In this study, we propose CartoAgent, a
novel multi-agent cartographic framework powered by multimodal large language
models (MLLMs). This framework simulates three key stages in cartographic
practice: preparation, map design, and evaluation. At each stage, different
MLLMs act as agents with distinct roles to collaborate, discuss, and utilize
tools for specific purposes. In particular, CartoAgent leverages MLLMs' visual
aesthetic capability and world knowledge to generate maps that are both
visually appealing and informative. By separating style from geographic data,
it can focus on designing stylesheets without modifying the vector-based data,
thereby ensuring geographic accuracy. We applied CartoAgent to a specific task
centered on map restyling-namely, map style transfer and evaluation. The
effectiveness of this framework was validated through extensive experiments and
a human evaluation study. CartoAgent can be extended to support a variety of
cartographic design decisions and inform future integrations of GenAI in
cartography.

### 4. [LAV: Audio-Driven Dynamic Visual Generation with Neural Compression and StyleGAN2](http://arxiv.org/pdf/2505.10101v1)

Authors: Jongmin Jung, Dasaem Jeong

This paper introduces LAV (Latent Audio-Visual), a system that integrates
EnCodec's neural audio compression with StyleGAN2's generative capabilities to
produce visually dynamic outputs driven by pre-recorded audio. Unlike previous
works that rely on explicit feature mappings, LAV uses EnCodec embeddings as
latent representations, directly transformed into StyleGAN2's style latent
space via randomly initialized linear mapping. This approach preserves semantic
richness in the transformation, enabling nuanced and semantically coherent
audio-visual translations. The framework demonstrates the potential of using
pretrained audio compression models for artistic and computational
applications.

### Computer Science and Game Theory

### 1. [Variety-Seeking Jump Games on Graphs](http://arxiv.org/pdf/2505.10005v1)

Authors: Lata Narayanan, Jaroslav Opatrny, Shanmukha Tummala, Alexandros A. Voudouris

We consider a class of jump games in which agents of different types occupy
the nodes of a graph aiming to maximize the variety of types in their
neighborhood. In particular, each agent derives a utility equal to the number
of types different from its own in its neighborhood. We show that the jump game
induced by the strategic behavior of the agents (who aim to maximize their
utility) may in general have improving response cycles, but is a potential game
under any of the following four conditions: there are only two types of agents;
or exactly one empty node; or the graph is of degree at most 2; or the graph is
3-regular and there are two empty nodes. Additionally, we show that on trees,
cylinder graphs, and tori, there is always an equilibrium. Finally, we show
tight bounds on the price of anarchy with respect to two different measures of
diversity: the social welfare (the total utility of the agents) and the number
of colorful edges (that connect agents of different types).

### 2. [The Art of Two-Round Voting](http://arxiv.org/pdf/2505.10377v1)

Authors: Qishen Han, Grant Schoenebeck, Biaoshuai Tao, Lirong Xia

We study the voting problem with two alternatives where voters' preferences
depend on a not-directly-observable state variable. While equilibria in the
one-round voting mechanisms lead to a good decision, they are usually hard to
compute and follow. We consider the two-round voting mechanism where the first
round serves as a polling stage and the winning alternative only depends on the
outcome of the second round. We show that the two-round voting mechanism is a
powerful tool for making collective decisions. Firstly, every (approximated)
equilibrium in the two-round voting mechanisms (asymptotically) leads to the
decision preferred by the majority as if the state of the world were revealed
to the voters. Moreover, there exist natural equilibria in the two-round game
following intuitive behaviors such as informative voting, sincere voting
[Austen-Smith and Banks, 1996], and the surprisingly popular strategy [Prelec
et al., 2017]. This sharply contrasts with the one-round voting mechanisms in
the previous literature, where no simple equilibrium is known. Finally, we show
that every equilibrium in the standard one-round majority vote mechanism gives
an equilibrium in the two-round mechanisms that is not more complicated than
the one-round equilibrium. Therefore, the two-round voting mechanism provides a
natural equilibrium in every instance, including those where one-round voting
fails to have a natural solution, and it can reach an informed majority
decision whenever one-round voting can. Our experiments on generative AI voters
also imply that two-round voting leads to the correct outcome more often than
one-round voting under some circumstances.

### 3. [Simultaneous Best-Response Dynamics in Random Potential Games](http://arxiv.org/pdf/2505.10378v1)

Authors: Galit Ashkenazi-Golan, Domenico Mergoni Cecchelli, Edward Plumb

This paper examines the convergence behaviour of simultaneous best-response
dynamics in random potential games. We provide a theoretical result showing
that, for two-player games with sufficiently many actions, the dynamics
converge quickly to a cycle of length two. This cycle lies within the
intersection of the neighbourhoods of two distinct Nash equilibria. For three
players or more, simulations show that the dynamics converge quickly to a Nash
equilibrium with high probability. Furthermore, we show that all these results
are robust, in the sense that they hold in non-potential games, provided the
players' payoffs are sufficiently correlated. We also compare these dynamics to
gradient-based learning methods in near-potential games with three players or
more, and observe that simultaneous best-response dynamics converge to a Nash
equilibrium of comparable payoff substantially faster.

### 4. [Aggregating Information and Preferences with Bounded-Size Deviations](http://arxiv.org/pdf/2505.10388v1)

Authors: Qishen Han, Grant Schoenebeck, Biaoshuai Tao, Lirong Xia

We investigate a voting scenario with two groups of agents whose preferences
depend on a ground truth that cannot be directly observed. The majority's
preferences align with the ground truth, while the minorities disagree.
Focusing on strategic behavior, we analyze situations where agents can form
coalitions up to a certain capacity and adopt the concept of ex-ante Bayesian
$k$-strong equilibrium, in which no group of at most $k$ agents has an
incentive to deviate. Our analysis provides a complete characterization of the
region where equilibria exist and yield the majority-preferred outcome when the
ground truth is common knowledge. This region is defined by two key parameters:
the size of the majority group and the maximum coalition capacity. When agents
cannot coordinate beyond a certain threshold determined by these parameters, a
stable outcome supporting the informed majority emerges. The boundary of this
region exhibits several distinct segments, notably including a surprising
non-linear relationship between majority size and deviation capacity. Our
results reveal the complexity of the strategic behaviors in this type of voting
game, which in turn demonstrate the capability of the ex-ante Bayesian
$k$-strong equilibrium to provide a more detailed analysis.

### 5. [Bridging Theory and Perception in Fair Division: A Study on Comparative and Fair Share Notions](http://arxiv.org/pdf/2505.10433v1)

Authors: Hadi Hosseini, Joshua Kavner, Samarth Khanna, Sujoy Sikdar, Lirong Xia

The allocation of resources among multiple agents is a fundamental problem in
both economics and computer science. In these settings, fairness plays a
crucial role in ensuring social acceptability and practical implementation of
resource allocation algorithms. Traditional fair division solutions have given
rise to a variety of approximate fairness notions, often as a response to the
challenges posed by non-existence or computational intractability of exact
solutions. However, the inherent incompatibility among these notions raises a
critical question: which concept of fairness is most suitable for practical
applications? In this paper, we examine two broad frameworks -- threshold-based
and comparison-based fairness notions -- and evaluate their perceived fairness
through a comprehensive human subject study. Our findings uncover novel
insights into the interplay between perception of fairness, theoretical
guarantees, the role of externalities and subjective valuations, and underlying
cognitive processes, shedding light on the theory and practice of fair
division.

### 6. [Price of Anarchy for Congestion and Scheduling Games via Vector Fitting](http://arxiv.org/pdf/2505.10082v1)

Authors: Danish Kashaev

We provide a dual fitting technique on a semidefinite program yielding simple
proofs of tight bounds for the robust price of anarchy of several congestion
and scheduling games under the sum of weighted completion times objective. The
same approach also allows to bound the approximation ratio of local search
algorithms for the scheduling problem $R || \sum w_j C_j$. All of our results
are obtained through a simple unified dual fitting argument on the same
semidefinite programming relaxation, which can essentially be obtained through
the first round of the Lasserre/Sum of Squares hierarchy.
  As our main application, we show that the known coordination ratio bounds of
respectively $4, (3 + \sqrt{5})/2 \approx 2.618,$ and $32/15 \approx 2.133$ for
the scheduling game $R || \sum w_j C_j$ under the coordination mechanisms
Smith's Rule, Proportional Sharing and Rand (STOC 2011) can be extended to
congestion games and obtained through this approach. For the natural
restriction where the weight of each player is proportional to its processing
time on every resource, we show that the last bound can be improved from 2.133
to 2. This improvement can also be made for general instances when considering
the price of anarchy of the game, rather than the coordination ratio. As a
further application of the technique, we show that it recovers the tight bound
of $(3 + \sqrt{5})/2$ for the price of anarchy of weighted affine congestion
games and the Kawaguchi-Kyan bound of $(1+ \sqrt{2})/2$ for the pure price of
anarchy of $P || \sum w_j C_j$. In addition, this approach recovers the known
tight approximation ratio of $(3 + \sqrt{5})/2 \approx 2.618$ for a natural
local search algorithm for $R || \sum w_j C_j$, as well as the best currently
known combinatorial approximation algorithm for this problem achieving an
approximation ratio of $(5 + \sqrt{5})/4 + \varepsilon \approx 1.809 +
\varepsilon$.

### Human-Computer Interaction

### 1. [Electrodermal Insights into Stress Dynamics of AR-Assisted Safety Warnings in Virtual Roadway Work Zone Environments](http://arxiv.org/pdf/2505.09867v1)

Authors: Fatemeh Banani Ardecani, Omidreza Shoghli

This study examines stress levels in roadway workers utilizing AR-assisted
multi-sensory warning systems under varying work intensities. A high-fidelity
Virtual Reality environment was used to replicate real-world scenarios,
allowing safe exploration of high-risk situations while focusing on the
physiological impacts of work conditions. Wearable sensors were used to
continuously and non-invasively collect physiological data, including
electrodermal activity to monitor stress responses. Analysis of data from 18
participants revealed notable differences in EDR between light- and
medium-intensity activities, reflecting variations in autonomic nervous system
activity under stress. Also, a feature importance analysis revealed that peak
and central tendency metrics of EDR were robust indicators of physiological
responses, between light- and medium-intensity activities. The findings
emphasize the relationship between AR-enabled warnings, work intensity, and
worker stress, offering an approach to active stress monitoring and improved
safety practices. By leveraging real-time physiological insights, this
methodology has the potential to support better stress management and the
development of more effective safety warning systems for roadway work zones.
This research also provides valuable guidance for designing interventions to
enhance worker safety, productivity, and well-being in high-risk settings.

### 2. [Context-AI Tunes: Context-Aware AI-Generated Music for Stress Reduction](http://arxiv.org/pdf/2505.09872v1)

Authors: Xiaoyan Wei, Zebang Zhang, Zijian Yue, Hsiang-Ting Chen

Music plays a critical role in emotional regulation and stress relief;
however, individuals often need different types of music tailored to their
unique stress levels or surrounding environment. Choosing the right music can
be challenging due to the overwhelming number of options and the time-consuming
trial-and-error process. To address this, we propose Context-AI Tune (CAT), a
system that generates personalized music based on environmental inputs and the
user's self-assessed stress level. A 2x2 within-subject experiment (N=26) was
conducted with two independent variables: AI (AI, NoAI) and Environment (Busy
Hub, Quiet Library). CAT's effectiveness in reducing stress was evaluated using
the Visual Analog Scale for Stress (VAS-S). Results show that CAT is more
effective than manually chosen music in reducing stress by adapting to user
context.

### 3. [Characterizing Unintended Consequences in Human-GUI Agent Collaboration for Web Browsing](http://arxiv.org/pdf/2505.09875v1)

Authors: Shuning Zhang, Jingruo Chen, Jiajing Gao, Zhiqi Gao, Xin Yi, Hewu Li

The proliferation of Large Language Model (LLM)-based Graphical User
Interface (GUI) agents in web browsing scenarios present complex unintended
consequences (UCs). This paper characterizes three UCs from three perspectives:
phenomena, influence and mitigation, drawing on social media analysis (N=221
posts) and semi-structured interviews (N=14). Key phenomenon for UCs include
agents' deficiencies in comprehending instructions and planning tasks,
challenges in executing accurate GUI interactions and adapting to dynamic
interfaces, the generation of unreliable or misaligned outputs, and
shortcomings in error handling and feedback processing. These phenomena
manifest as influences from unanticipated actions and user frustration, to
privacy violations and security vulnerabilities, and further to eroded trust
and wider ethical concerns. Our analysis also identifies user-initiated
mitigation, such as technical adjustments and manual oversight, and provides
implications for designing future LLM-based GUI agents that are robust,
user-centric, and transparent, fostering a crucial balance between automation
and human oversight.

### 4. [Post-Post-API Age: Studying Digital Platforms in Scant Data Access Times](http://arxiv.org/pdf/2505.09877v1)

Authors: Kayo Mimizuka, Megan A Brown, Kai-Cheng Yang, Josephine Lukito

Over the past decade, data provided by digital platforms has informed
substantial research in HCI to understand online human interaction and
communication. Following the closure of major social media APIs that previously
provided free access to large-scale data (the "post-API age"), emerging data
access programs required by the European Union's Digital Services Act (DSA)
have sparked optimism about increased platform transparency and renewed
opportunities for comprehensive research on digital platforms, leading to the
"post-post-API age." However, it remains unclear whether platforms provide
adequate data access in practice. To assess how platforms make data available
under the DSA, we conducted a comprehensive survey followed by in-depth
interviews with 19 researchers to understand their experiences with data access
in this new era. Our findings reveal significant challenges in accessing social
media data, with researchers facing multiple barriers including complex API
application processes, difficulties obtaining credentials, and limited API
usability. These challenges have exacerbated existing institutional, regional,
and financial inequities in data access. Based on these insights, we provide
actionable recommendations for platforms, researchers, and policymakers to
foster more equitable and effective data access, while encouraging broader
dialogue within the CSCW community around interdisciplinary and
multi-stakeholder solutions.

### 5. [SnapNCode: An Integrated Development Environment for Programming Physical Objects Interactions](http://arxiv.org/pdf/2505.09882v1)

Authors: Xiaoyan Wei, Zijian Yue, Hsiang-Ting Chen

Spatial computing technologies have the potential to revolutionize how we
interact with the world around us. However, most modern integrated development
environments (IDEs) have not fully adapted to this paradigm shift. For example,
physical 3D objects in the real world are still represented as 2D text
variables in code, creating a significant perceptual distance between these
representations. In response to this challenge, we introduce SnapNCode, a novel
IDE for spatial programming. SnapNCode enables programmers to capture various
states of physical objects through live video streams from cameras and directly
insert these visual representations into their code. Moreover, users can
augment physical objects by attaching code snippets onto objects, which are
opportunistically triggered when observed by cameras. We conducted a user study
(N=12) to assess the usability of SnapNCode. Feedback from participants
indicates that the system is easy-to-use and holds promise for daily casual
uses and integration into a broader range of workflows.

### 6. [Design and Evaluation of Generative Agent-based Platform for Human-Assistant Interaction Research: A Tale of 10 User Studies](http://arxiv.org/pdf/2505.09938v1)

Authors: Ziyi Xuan, Yiwen Wu, Xuhai Xu, Vinod Namboodiri, Mooi Choo Chuah, Yu Yang

Designing and evaluating personalized and proactive assistant agents remains
challenging due to the time, cost, and ethical concerns associated with
human-in-the-loop experimentation. Existing Human-Computer Interaction (HCI)
methods often require extensive physical setup and human participation, which
introduces privacy concerns and limits scalability. Simulated environments
offer a partial solution but are typically constrained by rule-based scenarios
and still depend heavily on human input to guide interactions and interpret
results. Recent advances in large language models (LLMs) have introduced the
possibility of generative agents that can simulate realistic human behavior,
reasoning, and social dynamics. However, their effectiveness in modeling
human-assistant interactions remains largely unexplored. To address this gap,
we present a generative agent-based simulation platform designed to simulate
human-assistant interactions. We identify ten prior studies on assistant agents
that span different aspects of interaction design and replicate these studies
using our simulation platform. Our results show that fully simulated
experiments using generative agents can approximate key aspects of
human-assistant interactions. Based on these simulations, we are able to
replicate the core conclusions of the original studies. Our work provides a
scalable and cost-effective approach for studying assistant agent design
without requiring live human subjects. We will open source both the platform
and collected results from the experiments on our website:
https://dash-gidea.github.io/.

### 7. [Exploring Large Quantities of Secondary Data from High-Resolution Synchrotron X-ray Computed Tomography Scans Using AccuStripes](http://arxiv.org/pdf/2505.10098v1)

Authors: Anja Heim, Thomas Lang, Christoph Heinzl

The analysis of secondary quantitative data extracted from high-resolution
synchrotron X-ray computed tomography scans represents a significant challenge
for users. While a number of methods have been introduced for processing large
three-dimensional images in order to generate secondary data, there are only a
few techniques available for simple and intuitive visualization of such data in
their entirety. This work employs the AccuStripes visualization technique for
that purpose, which enables the visual analysis of secondary data represented
by an ensemble of univariate distributions. It supports different schemes for
adaptive histogram binnings in combination with several ways of rendering
aggregated data and it allows the interactive selection of optimal visual
representations depending on the data and the use case. We demonstrate the
usability of AccuStripes on a high-resolution synchrotron scan of a
particle-reinforced metal matrix composite sample, containing more than 20
million particles. Through AccuStripes, detailed insights are facilitated into
distributions of derived particle characteristics of the entire sample.
Furthermore, research questions such as how the overall shape of the particles
is or how homogeneously they are distributed across the sample can be answered.

### 8. [Using Virtual Reality in Museums to Bridge the Gap Between Material Heritage and the Interpretation of Its Immaterial Context](http://arxiv.org/pdf/2505.10412v1)

Authors: Carlos R. Cunha, Vítor Mendonça, André Moreira, João Pedro Gomes, Aida Carvalho

Material heritage typically has a whole set of associated immaterial
heritage, which is essential to pass on to the visitor as a cultural mission of
the destinations and those who manage them. In this sense, the interpretation
of material heritage is a complex process that is not a fully efficient process
with the mere observation of physical artifacts. In this context, it emerges as
fundamental to provide visitors with a set of tools that allow them to
correctly interpret the artifacts that come to fully understand the cultural
dimension of the destinations and their heritage. Accordingly, the role of
virtual reality can leverage the creation of innovative and immersive solutions
that allow the visitor to understand and feel part of their own heritage and
its ancestral component that defines the sociocultural roots of destinations
and their civilizational traditions. This article, after dissecting and
substantiating the role of virtual reality in the interpretation of heritage,
presents a conceptual model, based on the use of virtual reality, which was, in
part, prototyped in the scenario of the Portuguese Museum in the city of
Miranda do Douro. This proposal is an ongoing contribution to the creation of
innovative and immersive tools for the interpretation of heritage.

### 9. [Influence of prior and task generated emotions on XAI explanation retention and understanding](http://arxiv.org/pdf/2505.10427v1)

Authors: Birte Richter, Christian Schütze, Anna Aksonova, Britta Wrede

The explanation of AI results and how they are received by users is an
increasingly active research field. However, there is a surprising lack of
knowledge about how social factors such as emotions affect the process of
explanation by a decision support system (DSS). While previous research has
shown effects of emotions on DSS supported decision-making, it remains unknown
in how far emotions affect cognitive processing during an explanation. In this
study, we, therefore, investigated the influence of prior emotions and
task-related arousal on the retention and understanding of explained feature
relevance. To investigate the influence of prior emotions, we induced happiness
and fear prior to the decision support interaction. Before emotion induction,
user characteristics to assess their risk type were collected via a
questionnaire. To identify emotional reactions to the explanations of the
relevance of different features, we observed heart rate variability (HRV),
facial expressions, and self-reported emotions of the explainee while observing
and listening to the explanation and assessed their retention of the features
as well as their influence on the outcome of the decision task. Results
indicate that (1) task-unrelated prior emotions do not affected the ratantion
but may affect the understanding of the relevance of certain features in the
sense of an emotion-induced confirmation bias, (2) certain features related to
personal attitudes yielded arousal in individual participants, (3) this arousal
affected the understanding of these variables.

### 10. [Emotion-sensitive Explanation Model](http://arxiv.org/pdf/2505.10454v1)

Authors: Christian Schütze, Birte Richter, Britta Wrede

Explainable AI (XAI) research has traditionally focused on rational users,
aiming to improve understanding and reduce cognitive biases. However, emotional
factors play a critical role in how explanations are perceived and processed.
Prior work shows that prior and task-generated emotions can negatively impact
the understanding of explanation. Building on these insights, we propose a
three-stage model for emotion-sensitive explanation grounding: (1) emotional or
epistemic arousal, (2) understanding, and (3) agreement. This model provides a
conceptual basis for developing XAI systems that dynamically adapt explanation
strategies to users emotional states, ultimately supporting more effective and
user-centered decision-making.

### Information Retrieval

### 1. [Boosting Text-to-Chart Retrieval through Training with Synthesized Semantic Insights](http://arxiv.org/pdf/2505.10043v1)

Authors: Yifan Wu, Lutao Yan, Yizhang Zhu, Yinan Mei, Jiannan Wang, Nan Tang, Yuyu Luo

Charts are crucial for data analysis and decision-making.Text-to-chart
retrieval systems have become increasingly important for Business Intelligence
(BI), where users need to find relevant charts that match their analytical
needs. These needs can be categorized into precise queries that are
well-specified and fuzzy queries that are more exploratory -- both require
understanding the semantics and context of the charts. However, existing
text-to-chart retrieval solutions often fail to capture the semantic content
and contextual information of charts, primarily due to the lack of
comprehensive metadata (or semantic insights). To address this limitation, we
propose a training data development pipeline that automatically synthesizes
hierarchical semantic insights for charts, covering visual patterns
(visual-oriented), statistical properties (statistics-oriented), and practical
applications (task-oriented), which produces 207,498 semantic insights for
69,166 charts. Based on these, we train a CLIP-based model named ChartFinder to
learn better representations of charts for text-to-chart retrieval. Our method
leverages rich semantic insights during the training phase to develop a model
that understands both visual and semantic aspects of charts.To evaluate
text-to-chart retrieval performance, we curate the first benchmark, CRBench,
for this task with 21,862 charts and 326 text queries from real-world BI
applications, with ground-truth labels verified by the crowd
workers.Experiments show that ChartFinder significantly outperforms existing
methods in text-to-chart retrieval tasks across various settings. For precise
queries, ChartFinder achieves up to 66.9% NDCG@10, which is 11.58% higher than
state-of-the-art models. In fuzzy query tasks, our method also demonstrates
consistent improvements, with an average increase of 5% across nearly all
metrics.

### 2. [Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M](http://arxiv.org/pdf/2505.10212v1)

Authors: Dario Di Palma, Felice Antonio Merra, Maurizio Sfilio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia

Large Language Models (LLMs) have become increasingly central to
recommendation scenarios due to their remarkable natural language understanding
and generation capabilities. Although significant research has explored the use
of LLMs for various recommendation tasks, little effort has been dedicated to
verifying whether they have memorized public recommendation dataset as part of
their training data. This is undesirable because memorization reduces the
generalizability of research findings, as benchmarking on memorized datasets
does not guarantee generalization to unseen datasets. Furthermore, memorization
can amplify biases, for example, some popular items may be recommended more
frequently than others.
  In this work, we investigate whether LLMs have memorized public
recommendation datasets. Specifically, we examine two model families (GPT and
Llama) across multiple sizes, focusing on one of the most widely used dataset
in recommender systems: MovieLens-1M. First, we define dataset memorization as
the extent to which item attributes, user profiles, and user-item interactions
can be retrieved by prompting the LLMs. Second, we analyze the impact of
memorization on recommendation performance. Lastly, we examine whether
memorization varies across model families and model sizes. Our results reveal
that all models exhibit some degree of memorization of MovieLens-1M, and that
recommendation performance is related to the extent of memorization. We have
made all the code publicly available at:
https://github.com/sisinflab/LLM-MemoryInspector

### Machine Learning

### 1. [BINGO: A Novel Pruning Mechanism to Reduce the Size of Neural Networks](http://arxiv.org/pdf/2505.09864v1)

Authors: Aditya Panangat

Over the past decade, the use of machine learning has increased
exponentially. Models are far more complex than ever before, growing to
gargantuan sizes and housing millions of weights. Unfortunately, the fact that
large models have become the state of the art means that it often costs
millions of dollars to train and operate them. These expenses not only hurt
companies but also bar non-wealthy individuals from contributing to new
developments and force consumers to pay greater prices for AI. Current methods
used to prune models, such as iterative magnitude pruning, have shown great
accuracy but require an iterative training sequence that is incredibly
computationally and environmentally taxing. To solve this problem, BINGO is
introduced. BINGO, during the training pass, studies specific subsets of a
neural network one at a time to gauge how significant of a role each weight
plays in contributing to a network's accuracy. By the time training is done,
BINGO generates a significance score for each weight, allowing for
insignificant weights to be pruned in one shot. BINGO provides an
accuracy-preserving pruning technique that is less computationally intensive
than current methods, allowing for a world where AI growth does not have to
mean model growth, as well.

### 2. [Improving the Euclidean Diffusion Generation of Manifold Data by Mitigating Score Function Singularity](http://arxiv.org/pdf/2505.09922v1)

Authors: Zichen Liu, Wei Zhang, Tiejun Li

Euclidean diffusion models have achieved remarkable success in generative
modeling across diverse domains, and they have been extended to manifold case
in recent advances. Instead of explicitly utilizing the structure of special
manifolds as studied in previous works, we investigate direct sampling of the
Euclidean diffusion models for general manifold-constrained data in this paper.
We reveal the multiscale singularity of the score function in the embedded
space of manifold, which hinders the accuracy of diffusion-generated samples.
We then present an elaborate theoretical analysis of the singularity structure
of the score function by separating it along the tangential and normal
directions of the manifold. To mitigate the singularity and improve the
sampling accuracy, we propose two novel methods: (1) Niso-DM, which introduces
non-isotropic noise along the normal direction to reduce scale discrepancies,
and (2) Tango-DM, which trains only the tangential component of the score
function using a tangential-only loss function. Numerical experiments
demonstrate that our methods achieve superior performance on distributions over
various manifolds with complex geometries.

### 3. [Approximated Behavioral Metric-based State Projection for Federated Reinforcement Learning](http://arxiv.org/pdf/2505.09959v1)

Authors: Zengxia Guo, Bohui An, Zhongqi Lu

Federated reinforcement learning (FRL) methods usually share the encrypted
local state or policy information and help each client to learn from others
while preserving everyone's privacy. In this work, we propose that sharing the
approximated behavior metric-based state projection function is a promising way
to enhance the performance of FRL and concurrently provides an effective
protection of sensitive information. We introduce FedRAG, a FRL framework to
learn a computationally practical projection function of states for each client
and aggregating the parameters of projection functions at a central server. The
FedRAG approach shares no sensitive task-specific information, yet provides
information gain for each client. We conduct extensive experiments on the
DeepMind Control Suite to demonstrate insightful results.

### 4. [Sybil-based Virtual Data Poisoning Attacks in Federated Learning](http://arxiv.org/pdf/2505.09983v1)

Authors: Changxun Zhu, Qilong Wu, Lingjuan Lyu, Shibei Xue

Federated learning is vulnerable to poisoning attacks by malicious
adversaries. Existing methods often involve high costs to achieve effective
attacks. To address this challenge, we propose a sybil-based virtual data
poisoning attack, where a malicious client generates sybil nodes to amplify the
poisoning model's impact. To reduce neural network computational complexity, we
develop a virtual data generation method based on gradient matching. We also
design three schemes for target model acquisition, applicable to online local,
online global, and offline scenarios. In simulation, our method outperforms
other attack algorithms since our method can obtain a global target model under
non-independent uniformly distributed data.

### 5. [ImagineBench: Evaluating Reinforcement Learning with Large Language Model Rollouts](http://arxiv.org/pdf/2505.10010v1)

Authors: Jing-Cheng Pang, Kaiyuan Li, Yidi Wang, Si-Hang Yang, Shengyi Jiang, Yang Yu

A central challenge in reinforcement learning (RL) is its dependence on
extensive real-world interaction data to learn task-specific policies. While
recent work demonstrates that large language models (LLMs) can mitigate this
limitation by generating synthetic experience (noted as imaginary rollouts) for
mastering novel tasks, progress in this emerging field is hindered due to the
lack of a standard benchmark. To bridge this gap, we introduce ImagineBench,
the first comprehensive benchmark for evaluating offline RL algorithms that
leverage both real rollouts and LLM-imaginary rollouts. The key features of
ImagineBench include: (1) datasets comprising environment-collected and
LLM-imaginary rollouts; (2) diverse domains of environments covering
locomotion, robotic manipulation, and navigation tasks; and (3) natural
language task instructions with varying complexity levels to facilitate
language-conditioned policy learning. Through systematic evaluation of
state-of-the-art offline RL algorithms, we observe that simply applying
existing offline RL algorithms leads to suboptimal performance on unseen tasks,
achieving 35.44% success rate in hard tasks in contrast to 64.37% of method
training on real rollouts for hard tasks. This result highlights the need for
algorithm advancements to better leverage LLM-imaginary rollouts. Additionally,
we identify key opportunities for future research: including better utilization
of imaginary rollouts, fast online adaptation and continual learning, and
extension to multi-modal tasks. Our code is publicly available at
https://github.com/LAMDA-RL/ImagineBench.

### 6. [Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](http://arxiv.org/pdf/2505.10039v1)

Authors: Hang Chen, Jiaying Zhu, Xinyu Yang, Wenya Wang

Circuit discovery has gradually become one of the prominent methods for
mechanistic interpretability, and research on circuit completeness has also
garnered increasing attention. Methods of circuit discovery that do not
guarantee completeness not only result in circuits that are not fixed across
different runs but also cause key mechanisms to be omitted. The nature of
incompleteness arises from the presence of OR gates within the circuit, which
are often only partially detected in standard circuit discovery methods. To
this end, we systematically introduce three types of logic gates: AND, OR, and
ADDER gates, and decompose the circuit into combinations of these logical
gates. Through the concept of these gates, we derive the minimum requirements
necessary to achieve faithfulness and completeness. Furthermore, we propose a
framework that combines noising-based and denoising-based interventions, which
can be easily integrated into existing circuit discovery methods without
significantly increasing computational complexity. This framework is capable of
fully identifying the logic gates and distinguishing them within the circuit.
In addition to the extensive experimental validation of the framework's ability
to restore the faithfulness, completeness, and sparsity of circuits, using this
framework, we uncover fundamental properties of the three logic gates, such as
their proportions and contributions to the output, and explore how they behave
among the functionalities of language models.

### 7. [Instance-Prototype Affinity Learning for Non-Exemplar Continual Graph Learning](http://arxiv.org/pdf/2505.10040v1)

Authors: Lei Song, Jiaxing Li, Shihan Guan, Youyong Kong

Graph Neural Networks (GNN) endure catastrophic forgetting, undermining their
capacity to preserve previously acquired knowledge amid the assimilation of
novel information. Rehearsal-based techniques revisit historical examples,
adopted as a principal strategy to alleviate this phenomenon. However, memory
explosion and privacy infringements impose significant constraints on their
utility. Non-Exemplar methods circumvent the prior issues through Prototype
Replay (PR), yet feature drift presents new challenges. In this paper, our
empirical findings reveal that Prototype Contrastive Learning (PCL) exhibits
less pronounced drift than conventional PR. Drawing upon PCL, we propose
Instance-Prototype Affinity Learning (IPAL), a novel paradigm for Non-Exemplar
Continual Graph Learning (NECGL). Exploiting graph structural information, we
formulate Topology-Integrated Gaussian Prototypes (TIGP), guiding feature
distributions towards high-impact nodes to augment the model's capacity for
assimilating new knowledge. Instance-Prototype Affinity Distillation (IPAD)
safeguards task memory by regularizing discontinuities in class relationships.
Moreover, we embed a Decision Boundary Perception (DBP) mechanism within PCL,
fostering greater inter-class discriminability. Evaluations on four node
classification benchmark datasets demonstrate that our method outperforms
existing state-of-the-art methods, achieving a better trade-off between
plasticity and stability.

### 8. [JointDistill: Adaptive Multi-Task Distillation for Joint Depth Estimation and Scene Segmentation](http://arxiv.org/pdf/2505.10057v1)

Authors: Tiancong Cheng, Ying Zhang, Yuxuan Liang, Roger Zimmermann, Zhiwen Yu, Bin Guo

Depth estimation and scene segmentation are two important tasks in
intelligent transportation systems. A joint modeling of these two tasks will
reduce the requirement for both the storage and training efforts. This work
explores how the multi-task distillation could be used to improve such unified
modeling. While existing solutions transfer multiple teachers' knowledge in a
static way, we propose a self-adaptive distillation method that can dynamically
adjust the knowledge amount from each teacher according to the student's
current learning ability. Furthermore, as multiple teachers exist, the
student's gradient update direction in the distillation is more prone to be
erroneous where knowledge forgetting may occur. To avoid this, we propose a
knowledge trajectory to record the most essential information that a model has
learnt in the past, based on which a trajectory-based distillation loss is
designed to guide the student to follow the learning curve similarly in a
cost-effective way. We evaluate our method on multiple benchmarking datasets
including Cityscapes and NYU-v2. Compared to the state-of-the-art solutions,
our method achieves a clearly improvement. The code is provided in the
supplementary materials.

### 9. [ChronoSteer: Bridging Large Language Model and Time Series Foundation Model via Synthetic Data](http://arxiv.org/pdf/2505.10083v1)

Authors: Chengsen Wang, Qi Qi, Zhongwen Rao, Lujia Pan, Jingyu Wang, Jianxin Liao

Conventional forecasting methods rely on unimodal time series data, limiting
their ability to exploit rich textual information. Recently, large language
models (LLMs) and time series foundation models (TSFMs) have demonstrated
powerful capability in textual reasoning and temporal modeling, respectively.
Integrating the strengths of both to construct a multimodal model that
concurrently leverages both temporal and textual information for future
inference has emerged as a critical research challenge. To address the scarcity
of event-series paired data, we propose a decoupled framework: an LLM is
employed to transform textual events into revision instructions, which are then
used to steer the output of TSFM. To implement this framework, we introduce
ChronoSteer, a multimodal TSFM that can be steered through textual revision
instructions, effectively bridging LLM and TSFM. Moreover, to mitigate the
shortage of cross-modal instruction-series paired data, we devise a two-stage
training strategy based on synthetic data. In addition, we also construct a
high-quality multimodal time series forecasting benchmark to address the
information leakage concerns during evaluation. After integrating with an LLM,
ChronoSteer, which is trained exclusively on synthetic data, achieves a 25.7%
improvement in prediction accuracy compared to the unimodal backbone and a
22.5% gain over the previous state-of-the-art multimodal method.

### 10. [Enhancing the Performance of Global Model by Improving the Adaptability of Local Models in Federated Learning](http://arxiv.org/pdf/2505.10125v1)

Authors: Wujun Zhou, Shu Ding, ZeLin Li, Wei Wang

Federated learning enables the clients to collaboratively train a global
model, which is aggregated from local models. Due to the heterogeneous data
distributions over clients and data privacy in federated learning, it is
difficult to train local models to achieve a well-performed global model. In
this paper, we introduce the adaptability of local models, i.e., the average
performance of local models on data distributions over clients, and enhance the
performance of the global model by improving the adaptability of local models.
Since each client does not know the data distributions over other clients, the
adaptability of the local model cannot be directly optimized. First, we provide
the property of an appropriate local model which has good adaptability on the
data distributions over clients. Then, we formalize the property into the local
training objective with a constraint and propose a feasible solution to train
the local model. Extensive experiments on federated learning benchmarks
demonstrate that our method significantly improves the adaptability of local
models and achieves a well-performed global model that consistently outperforms
the baseline methods.

### Neural and Evolutionary Computing

### 1. [Incorporating brain-inspired mechanisms for multimodal learning in artificial intelligence](http://arxiv.org/pdf/2505.10176v1)

Authors: Xiang He, Dongcheng Zhao, Yang Li, Qingqun Kong, Xin Yang, Yi Zeng

Multimodal learning enhances the perceptual capabilities of cognitive systems
by integrating information from different sensory modalities. However, existing
multimodal fusion research typically assumes static integration, not fully
incorporating key dynamic mechanisms found in the brain. Specifically, the
brain exhibits an inverse effectiveness phenomenon, wherein weaker unimodal
cues yield stronger multisensory integration benefits; conversely, when
individual modal cues are stronger, the effect of fusion is diminished. This
mechanism enables biological systems to achieve robust cognition even with
scarce or noisy perceptual cues. Inspired by this biological mechanism, we
explore the relationship between multimodal output and information from
individual modalities, proposing an inverse effectiveness driven multimodal
fusion (IEMF) strategy. By incorporating this strategy into neural networks, we
achieve more efficient integration with improved model performance and
computational efficiency, demonstrating up to 50% reduction in computational
cost across diverse fusion methods. We conduct experiments on audio-visual
classification, continual learning, and question answering tasks to validate
our method. Results consistently demonstrate that our method performs
excellently in these tasks. To verify universality and generalization, we also
conduct experiments on Artificial Neural Networks (ANN) and Spiking Neural
Networks (SNN), with results showing good adaptability to both network types.
Our research emphasizes the potential of incorporating biologically inspired
mechanisms into multimodal networks and provides promising directions for the
future development of multimodal artificial intelligence. The code is available
at https://github.com/Brain-Cog-Lab/IEMF.

### 2. [ILIF: Temporal Inhibitory Leaky Integrate-and-Fire Neuron for Overactivation in Spiking Neural Networks](http://arxiv.org/pdf/2505.10371v1)

Authors: Kai Sun, Peibo Duan, Levin Kuhlmann, Beilun Wang, Bin Zhang

The Spiking Neural Network (SNN) has drawn increasing attention for its
energy-efficient, event-driven processing and biological plausibility. To train
SNNs via backpropagation, surrogate gradients are used to approximate the
non-differentiable spike function, but they only maintain nonzero derivatives
within a narrow range of membrane potentials near the firing threshold,
referred to as the surrogate gradient support width gamma. We identify a major
challenge, termed the dilemma of gamma: a relatively large gamma leads to
overactivation, characterized by excessive neuron firing, which in turn
increases energy consumption, whereas a small gamma causes vanishing gradients
and weakens temporal dependencies. To address this, we propose a temporal
Inhibitory Leaky Integrate-and-Fire (ILIF) neuron model, inspired by biological
inhibitory mechanisms. This model incorporates interconnected inhibitory units
for membrane potential and current, effectively mitigating overactivation while
preserving gradient propagation. Theoretical analysis demonstrates ILIF
effectiveness in overcoming the gamma dilemma, and extensive experiments on
multiple datasets show that ILIF improves energy efficiency by reducing firing
rates, stabilizes training, and enhances accuracy. The code is available at
github.com/kaisun1/ILIF.

### 3. [Role of scrambling and noise in temporal information processing with quantum systems](http://arxiv.org/pdf/2505.10080v1)

Authors: Weijie Xiong, Zoë Holmes, Armando Angrisani, Yudai Suzuki, Thiparat Chotibut, Supanut Thanasilp

Scrambling quantum systems have been demonstrated as effective substrates for
temporal information processing. While their role in providing rich feature
maps has been widely studied, a theoretical understanding of their performance
in temporal tasks is still lacking. Here we consider a general quantum
reservoir processing framework that captures a broad range of physical
computing models with quantum systems. We examine the scalability and memory
retention of the model with scrambling reservoirs modelled by high-order
unitary designs in both noiseless and noisy settings. In the former regime, we
show that measurement readouts become exponentially concentrated with
increasing reservoir size, yet strikingly do not worsen with the reservoir
iterations. Thus, while repeatedly reusing a small scrambling reservoir with
quantum data might be viable, scaling up the problem size deteriorates
generalization unless one can afford an exponential shot overhead. In contrast,
the memory of early inputs and initial states decays exponentially in both
reservoir size and reservoir iterations. In the noisy regime, we also prove
exponential memory decays with iterations for local noisy channels. Proving
these results required us to introduce new proof techniques for bounding
concentration in temporal quantum learning models.

### Networking and Internet Architecture

### 1. [Solar-CSK: Decoding Color Coded Visible Light Communications using Solar Cells](http://arxiv.org/pdf/2505.10226v1)

Authors: Yanxiang Wang, Yihe Yan, Jiawei Hu, Cheng Jiang, Brano Kusy, Ashraf Uddin, Mahbub Hassan, Wen Hu

Visible Light Communication (VLC) provides an energy-efficient wireless
solution by using existing LED-based illumination for high-speed data
transmissions. Although solar cells offer the advantage of simultaneous energy
harvesting and data reception, their broadband nature hinders accurate decoding
of color-coded signals like Color Shift Keying (CSK). In this paper, we propose
a novel approach exploiting the concept of tandem solar cells, multi-layer
devices with partial wavelength selectivity, to capture coarse color
information without resorting to energy-limiting color filters. To address the
residual spectral overlap, we develop a bidirectional LSTM-based machine
learning framework that infers channel characteristics by comparing solar
cells' photovoltaic signals with pilot-based anchor data. Our commercial
off-the-shelf (COTS) solar prototype achieves robust performance across varying
distances and ambient lighting levels, significantly reducing bit error rates
compared to conventional channel estimation methods. These findings mark a step
toward sustainable, high-performance VLC systems powered by the multi-layer
solar technologies.

### 2. [A Survey on Open-Source Edge Computing Simulators and Emulators: The Computing and Networking Convergence Perspective](http://arxiv.org/pdf/2505.09995v1)

Authors: Jianpeng Qi, Chao Liu, Xiao Zhang, Lei Wang, Rui Wang, Junyu Dong, Yanwei Yu

Edge computing, with its low latency, dynamic scalability, and location
awareness, along with the convergence of computing and communication paradigms,
has been successfully applied in critical domains such as industrial IoT, smart
healthcare, smart homes, and public safety. This paper provides a comprehensive
survey of open-source edge computing simulators and emulators, presented in our
GitHub repository (https://github.com/qijianpeng/awesome-edge-computing),
emphasizing the convergence of computing and networking paradigms. By examining
more than 40 tools, including CloudSim, NS-3, and others, we identify the
strengths and limitations in simulating and emulating edge environments. This
survey classifies these tools into three categories: packet-level,
application-level, and emulators. Furthermore, we evaluate them across five
dimensions, ranging from resource representation to resource utilization. The
survey highlights the integration of different computing paradigms, packet
processing capabilities, support for edge environments, user-defined metric
interfaces, and scenario visualization. The findings aim to guide researchers
in selecting appropriate tools for developing and validating advanced computing
and networking technologies.

### 3. [Energy-Efficient and Reliable Data Collection in Receiver-Initiated Wake-up Radio Enabled IoT Networks](http://arxiv.org/pdf/2505.10122v1)

Authors: Syed Luqman Shah, Ziaul Haq Abbas, Ghulam Abbas, Nurul Huda Mahmood

In unmanned aerial vehicle (UAV)-assisted wake-up radio (WuR)-enabled
internet of things (IoT) networks, UAVs can instantly activate the main radios
(MRs) of the sensor nodes (SNs) with a wake-up call (WuC) for efficient data
collection in mission-driven data collection scenarios. However, the
spontaneous response of numerous SNs to the UAV's WuC can lead to significant
packet loss and collisions, as WuR does not exhibit its superiority for
high-traffic loads. To address this challenge, we propose an innovative
receiver-initiated WuR UAV-assisted clustering (RI-WuR-UAC) medium access
control (MAC) protocol to achieve low latency and high reliability in ultra-low
power consumption applications. We model the proposed protocol using the
$M/G/1/2$ queuing framework and derive expressions for key performance metrics,
i.e., channel busyness probability, probability of successful clustering,
average SN energy consumption, and average transmission delay. The RI-WuR-UAC
protocol employs three distinct data flow models, tailored to different network
traffic conditions, which perform three MAC mechanisms: channel assessment
(CCA) clustering for light traffic loads, backoff plus CCA clustering for dense
and heavy traffic, and adaptive clustering for variable traffic loads.
Simulation results demonstrate that the RI-WuR-UAC protocol significantly
outperforms the benchmark sub-carrier modulation clustering protocol. By
varying the network load, we capture the trade-offs among the performance
metrics, showcasing the superior efficiency and reliability of the RI-WuR-UAC
protocol.

### 4. [LibIQ: Toward Real-Time Spectrum Classification in O-RAN dApps](http://arxiv.org/pdf/2505.10537v1)

Authors: Filippo Olimpieri, Noemi Giustini, Andrea Lacava, Salvatore D'Oro, Tommaso Melodia, Francesca Cuomo

The O-RAN architecture is transforming cellular networks by adopting RAN
softwarization and disaggregation concepts to enable data-driven monitoring and
control of the network. Such management is enabled by RICs, which facilitate
near-real-time and non-real-time network control through xApps and rApps.
However, they face limitations, including latency overhead in data exchange
between the RAN and RIC, restricting real-time monitoring, and the inability to
access user plain data due to privacy and security constraints, hindering use
cases like beamforming and spectrum classification. In this paper, we leverage
the dApps concept to enable real-time RF spectrum classification with LibIQ, a
novel library for RF signals that facilitates efficient spectrum monitoring and
signal classification by providing functionalities to read I/Q samples as
time-series, create datasets and visualize time-series data through plots and
spectrograms. Thanks to LibIQ, I/Q samples can be efficiently processed to
detect external RF signals, which are subsequently classified using a CNN
inside the library. To achieve accurate spectrum analysis, we created an
extensive dataset of time-series-based I/Q samples, representing distinct
signal types captured using a custom dApp running on a 5G deployment over the
Colosseum network emulator and an OTA testbed. We evaluate our model by
deploying LibIQ in heterogeneous scenarios with varying center frequencies,
time windows, and external RF signals. In real-time analysis, the model
classifies the processed I/Q samples, achieving an average accuracy of
approximately 97.8\% in identifying signal types across all scenarios. We
pledge to release both LibIQ and the dataset created as a publicly available
framework upon acceptance.

### 5. [AI Greenferencing: Routing AI Inferencing to Green Modular Data Centers with Heron](http://arxiv.org/pdf/2505.09989v1)

Authors: Tella Rajashekhar Reddy, Palak, Rohan Gandhi, Anjaly Parayil, Chaojie Zhang, Mike Shepperd, Liangcheng Yu, Jayashree Mohan, Srinivasan Iyengar, Shivkumar Kalyanaraman, Debopam Bhattacherjee

AI power demand is growing unprecedentedly thanks to the high power density
of AI compute and the emerging inferencing workload. On the supply side,
abundant wind power is waiting for grid access in interconnection queues. In
this light, this paper argues bringing AI workload to modular compute clusters
co-located in wind farms. Our deployment right-sizing strategy makes it
economically viable to deploy more than 6 million high-end GPUs today that
could consume cheap, green power at its source. We built Heron, a cross-site
software router, that could efficiently leverage the complementarity of power
generation across wind farms by routing AI inferencing workload around power
drops. Using 1-week ofcoding and conversation production traces from Azure and
(real) variable wind power traces, we show how Heron improves aggregate goodput
of AI compute by up to 80% compared to the state-of-the-art.

### 6. [AttentionGuard: Transformer-based Misbehavior Detection for Secure Vehicular Platoons](http://arxiv.org/pdf/2505.10273v1)

Authors: Hexu Li, Konstantinos Kalogiannis, Ahmed Mohamed Hussain, Panos Papadimitratos

Vehicle platooning, with vehicles traveling in close formation coordinated
through Vehicle-to-Everything (V2X) communications, offers significant benefits
in fuel efficiency and road utilization. However, it is vulnerable to
sophisticated falsification attacks by authenticated insiders that can
destabilize the formation and potentially cause catastrophic collisions. This
paper addresses this challenge: misbehavior detection in vehicle platooning
systems. We present AttentionGuard, a transformer-based framework for
misbehavior detection that leverages the self-attention mechanism to identify
anomalous patterns in mobility data. Our proposal employs a multi-head
transformer-encoder to process sequential kinematic information, enabling
effective differentiation between normal mobility patterns and falsification
attacks across diverse platooning scenarios, including steady-state
(no-maneuver) operation, join, and exit maneuvers. Our evaluation uses an
extensive simulation dataset featuring various attack vectors (constant,
gradual, and combined falsifications) and operational parameters (controller
types, vehicle speeds, and attacker positions). Experimental results
demonstrate that AttentionGuard achieves up to 0.95 F1-score in attack
detection, with robust performance maintained during complex maneuvers.
Notably, our system performs effectively with minimal latency (100ms decision
intervals), making it suitable for real-time transportation safety
applications. Comparative analysis reveals superior detection capabilities and
establishes the transformer-encoder as a promising approach for securing
Cooperative Intelligent Transport Systems (C-ITS) against sophisticated insider
threats.

### Robotics

### 1. [Unsupervised Radar Point Cloud Enhancement via Arbitrary LiDAR Guided Diffusion Prior](http://arxiv.org/pdf/2505.09887v1)

Authors: Yanlong Yang, Jianan Liu, Guanxiong Luo, Hao Li, Euijoon Ahn, Mostafa Rahimi Azghadi, Tao Huang

In industrial automation, radar is a critical sensor in machine perception.
However, the angular resolution of radar is inherently limited by the Rayleigh
criterion, which depends on both the radar's operating wavelength and the
effective aperture of its antenna array.To overcome these hardware-imposed
limitations, recent neural network-based methods have leveraged high-resolution
LiDAR data, paired with radar measurements, during training to enhance radar
point cloud resolution. While effective, these approaches require extensive
paired datasets, which are costly to acquire and prone to calibration error.
These challenges motivate the need for methods that can improve radar
resolution without relying on paired high-resolution ground-truth data. Here,
we introduce an unsupervised radar points enhancement algorithm that employs an
arbitrary LiDAR-guided diffusion model as a prior without the need for paired
training data. Specifically, our approach formulates radar angle estimation
recovery as an inverse problem and incorporates prior knowledge through a
diffusion model with arbitrary LiDAR domain knowledge. Experimental results
demonstrate that our method attains high fidelity and low noise performance
compared to traditional regularization techniques. Additionally, compared to
paired training methods, it not only achieves comparable performance but also
offers improved generalization capability. To our knowledge, this is the first
approach that enhances radar points output by integrating prior knowledge via a
diffusion model rather than relying on paired training data. Our code is
available at https://github.com/yyxr75/RadarINV.

### 2. [Diffusion-SAFE: Shared Autonomy Framework with Diffusion for Safe Human-to-Robot Driving Handover](http://arxiv.org/pdf/2505.09889v1)

Authors: Yunxin Fan, Monroe Kennedy III

Safe handover in shared autonomy for vehicle control is well-established in
modern vehicles. However, avoiding accidents often requires action several
seconds in advance. This necessitates understanding human driver behavior and
an expert control strategy for seamless intervention when a collision or unsafe
state is predicted. We propose Diffusion-SAFE, a closed-loop shared autonomy
framework leveraging diffusion models to: (1) predict human driving behavior
for detection of potential risks, (2) generate safe expert trajectories, and
(3) enable smooth handovers by blending human and expert policies over a short
time horizon. Unlike prior works which use engineered score functions to rate
driving performance, our approach enables both performance evaluation and
optimal action sequence generation from demonstrations. By adjusting the
forward and reverse processes of the diffusion-based copilot, our method
ensures a gradual transition of control authority, by mimicking the drivers'
behavior before intervention, which mitigates abrupt takeovers, leading to
smooth transitions. We evaluated Diffusion-SAFE in both simulation
(CarRacing-v0) and real-world (ROS-based race car), measuring human-driving
similarity, safety, and computational efficiency. Results demonstrate a 98.5\%
successful handover rate, highlighting the framework's effectiveness in
progressively correcting human actions and continuously sampling optimal robot
actions.

### 3. [Learning Diverse Natural Behaviors for Enhancing the Agility of Quadrupedal Robots](http://arxiv.org/pdf/2505.09979v1)

Authors: Huiqiao Fu, Haoyu Dong, Wentao Xu, Zhehao Zhou, Guizhou Deng, Kaiqiang Tang, Daoyi Dong, Chunlin Chen

Achieving animal-like agility is a longstanding goal in quadrupedal robotics.
While recent studies have successfully demonstrated imitation of specific
behaviors, enabling robots to replicate a broader range of natural behaviors in
real-world environments remains an open challenge. Here we propose an
integrated controller comprising a Basic Behavior Controller (BBC) and a
Task-Specific Controller (TSC) which can effectively learn diverse natural
quadrupedal behaviors in an enhanced simulator and efficiently transfer them to
the real world. Specifically, the BBC is trained using a novel semi-supervised
generative adversarial imitation learning algorithm to extract diverse
behavioral styles from raw motion capture data of real dogs, enabling smooth
behavior transitions by adjusting discrete and continuous latent variable
inputs. The TSC, trained via privileged learning with depth images as input,
coordinates the BBC to efficiently perform various tasks. Additionally, we
employ evolutionary adversarial simulator identification to optimize the
simulator, aligning it closely with reality. After training, the robot exhibits
diverse natural behaviors, successfully completing the quadrupedal agility
challenge at an average speed of 1.1 m/s and achieving a peak speed of 3.2 m/s
during hurdling. This work represents a substantial step toward animal-like
agility in quadrupedal robots, opening avenues for their deployment in
increasingly complex real-world environments.

### 4. [LEMON-Mapping: Loop-Enhanced Large-Scale Multi-Session Point Cloud Merging and Optimization for Globally Consistent Mapping](http://arxiv.org/pdf/2505.10018v1)

Authors: Lijie Wang, Xiaoyi Zhong, Ziyi Xu, Kaixin Chai, Anke Zhao, Tianyu Zhao, Qianhao Wang, Fei Gao

With the rapid development of robotics, multi-robot collaboration has become
critical and challenging. One key problem is integrating data from multiple
robots to build a globally consistent and accurate map for robust cooperation
and precise localization. While traditional multi-robot pose graph optimization
(PGO) maintains basic global consistency, it focuses primarily on pose
optimization and ignores the geometric structure of the map. Moreover, PGO only
uses loop closure as a constraint between two nodes, failing to fully exploit
its capability to maintaining local consistency of multi-robot maps. Therefore,
PGO-based multi-robot mapping methods often suffer from serious map divergence
and blur, especially in regions with overlapping submaps. To address this
issue, we propose Lemon-Mapping, a loop-enhanced framework for large-scale
multi-session point cloud map fusion and optimization, which reasonably
utilizes loop closure and improves the geometric quality of the map. We
re-examine the role of loops for multi-robot mapping and introduce three key
innovations. First, we develop a robust loop processing mechanism that
effectively rejects outliers and a novel loop recall strategy to recover
mistakenly removed loops. Second, we introduce a spatial bundle adjustment
method for multi-robot maps that significantly reduces the divergence in
overlapping regions and eliminates map blur. Third, we design a PGO strategy
that leverages the refined constraints of bundle adjustment to extend the local
accuracy to the global map. We validate our framework on several public
datasets and a self-collected dataset. Experimental results demonstrate that
our method outperforms traditional map merging approaches in terms of mapping
accuracy and reduction of map divergence. Scalability experiments also
demonstrate the strong capability of our framework to handle scenarios
involving numerous robots.

### 5. [APEX: Action Priors Enable Efficient Exploration for Skill Imitation on Articulated Robots](http://arxiv.org/pdf/2505.10022v1)

Authors: Shivam Sood, Laukik B Nakhwa, Yuhong Cao, Sun Ge, Guillaume Sartoretti

Learning by imitation provides an effective way for robots to develop
well-regulated complex behaviors and directly benefit from natural
demonstrations. State-of-the-art imitation learning (IL) approaches typically
leverage Adversarial Motion Priors (AMP), which, despite their impressive
results, suffer from two key limitations. They are prone to mode collapse,
which often leads to overfitting to the simulation environment and thus
increased sim-to-real gap, and they struggle to learn diverse behaviors
effectively. To overcome these limitations, we introduce APEX (Action Priors
enable Efficient eXploration): a simple yet versatile imitation learning
framework that integrates demonstrations directly into reinforcement learning
(RL), maintaining high exploration while grounding behavior with
expert-informed priors. We achieve this through a combination of decaying
action priors, which initially bias exploration toward expert demonstrations
but gradually allow the policy to explore independently. This is complemented
by a multi-critic RL framework that effectively balances stylistic consistency
with task performance. Our approach achieves sample-efficient imitation
learning and enables the acquisition of diverse skills within a single policy.
APEX generalizes to varying velocities and preserves reference-like styles
across complex tasks such as navigating rough terrain and climbing stairs,
utilizing only flat-terrain kinematic motion data as a prior. We validate our
framework through extensive hardware experiments on the Unitree Go2 quadruped.
There, APEX yields diverse and agile locomotion gaits, inherent gait
transitions, and the highest reported speed for the platform to the best of our
knowledge (peak velocity of ~3.3 m/s on hardware). Our results establish APEX
as a compelling alternative to existing IL methods, offering better efficiency,
adaptability, and real-world performance.

### 6. [Training People to Reward Robots](http://arxiv.org/pdf/2505.10151v1)

Authors: Endong Sun, Yuqing Zhu, Matthew Howard

Learning from demonstration (LfD) is a technique that allows expert teachers
to teach task-oriented skills to robotic systems. However, the most effective
way of guiding novice teachers to approach expert-level demonstrations
quantitatively for specific teaching tasks remains an open question. To this
end, this paper investigates the use of machine teaching (MT) to guide novice
teachers to improve their teaching skills based on reinforcement learning from
demonstration (RLfD). The paper reports an experiment in which novices receive
MT-derived guidance to train their ability to teach a given motor skill with
only 8 demonstrations and generalise this to previously unseen ones. Results
indicate that the MT-guidance not only enhances robot learning performance by
89% on the training skill but also causes a 70% improvement in robot learning
performance on skills not seen by subjects during training. These findings
highlight the effectiveness of MT-guidance in upskilling human teaching
behaviours, ultimately improving demonstration quality in RLfD.

### 7. [Towards Safe Robot Foundation Models Using Inductive Biases](http://arxiv.org/pdf/2505.10219v1)

Authors: Maximilian Tölle, Theo Gruner, Daniel Palenicek, Tim Schneider, Jonas Günster, Joe Watson, Davide Tateo, Puze Liu, Jan Peters

Safety is a critical requirement for the real-world deployment of robotic
systems. Unfortunately, while current robot foundation models show promising
generalization capabilities across a wide variety of tasks, they fail to
address safety, an important aspect for ensuring long-term operation. Current
robot foundation models assume that safe behavior should emerge by learning
from a sufficiently large dataset of demonstrations. However, this approach has
two clear major drawbacks. Firstly, there are no formal safety guarantees for a
behavior cloning policy trained using supervised learning. Secondly, without
explicit knowledge of any safety constraints, the policy may require an
unreasonable number of additional demonstrations to even approximate the
desired constrained behavior. To solve these key issues, we show how we can
instead combine robot foundation models with geometric inductive biases using
ATACOM, a safety layer placed after the foundation policy that ensures safe
state transitions by enforcing action constraints. With this approach, we can
ensure formal safety guarantees for generalist policies without providing
extensive demonstrations of safe behavior, and without requiring any specific
fine-tuning for safety. Our experiments show that our approach can be
beneficial both for classical manipulation tasks, where we avoid unwanted
collisions with irrelevant objects, and for dynamic tasks, such as the robot
air hockey environment, where we can generate fast trajectories respecting
complex tasks and joint space constraints.

### 8. [Force-Driven Validation for Collaborative Robotics in Automated Avionics Testing](http://arxiv.org/pdf/2505.10224v1)

Authors: Pietro Dardano, Paolo Rocco, David Frisini

ARTO is a project combining collaborative robots (cobots) and Artificial
Intelligence (AI) to automate functional test procedures for civilian and
military aircraft certification. This paper proposes a Deep Learning (DL) and
eXplainable AI (XAI) approach, equipping ARTO with interaction analysis
capabilities to verify and validate the operations on cockpit components.
During these interactions, forces, torques, and end effector poses are recorded
and preprocessed to filter disturbances caused by low performance force
controllers and embedded Force Torque Sensors (FTS). Convolutional Neural
Networks (CNNs) then classify the cobot actions as Success or Fail, while also
identifying and reporting the causes of failure. To improve interpretability,
Grad CAM, an XAI technique for visual explanations, is integrated to provide
insights into the models decision making process. This approach enhances the
reliability and trustworthiness of the automated testing system, facilitating
the diagnosis and rectification of errors that may arise during testing.

### 9. [Quad-LCD: Layered Control Decomposition Enables Actuator-Feasible Quadrotor Trajectory Planning](http://arxiv.org/pdf/2505.10228v1)

Authors: Anusha Srikanthan, Hanli Zhang, Spencer Folk, Vijay Kumar, Nikolai Matni

In this work, we specialize contributions from prior work on data-driven
trajectory generation for a quadrotor system with motor saturation constraints.
When motors saturate in quadrotor systems, there is an ``uncontrolled drift" of
the vehicle that results in a crash. To tackle saturation, we apply a control
decomposition and learn a tracking penalty from simulation data consisting of
low, medium and high-cost reference trajectories. Our approach reduces crash
rates by around $49\%$ compared to baselines on aggressive maneuvers in
simulation. On the Crazyflie hardware platform, we demonstrate feasibility
through experiments that lead to successful flights. Motivated by the growing
interest in data-driven methods to quadrotor planning, we provide open-source
lightweight code with an easy-to-use abstraction of hardware platforms.

### 10. [Context-aware collaborative pushing of heavy objects using skeleton-based intention prediction](http://arxiv.org/pdf/2505.10239v1)

Authors: Gokhan Solak, Gustavo J. G. Lahr, Idil Ozdamar, Arash Ajoudani

In physical human-robot interaction, force feedback has been the most common
sensing modality to convey the human intention to the robot. It is widely used
in admittance control to allow the human to direct the robot. However, it
cannot be used in scenarios where direct force feedback is not available since
manipulated objects are not always equipped with a force sensor. In this work,
we study one such scenario: the collaborative pushing and pulling of heavy
objects on frictional surfaces, a prevalent task in industrial settings. When
humans do it, they communicate through verbal and non-verbal cues, where body
poses, and movements often convey more than words. We propose a novel
context-aware approach using Directed Graph Neural Networks to analyze
spatio-temporal human posture data to predict human motion intention for
non-verbal collaborative physical manipulation. Our experiments demonstrate
that robot assistance significantly reduces human effort and improves task
efficiency. The results indicate that incorporating posture-based context
recognition, either together with or as an alternative to force sensing,
enhances robot decision-making and control efficiency.

### Software Engineering

### 1. [Advancing Mobile UI Testing by Learning Screen Usage Semantics](http://arxiv.org/pdf/2505.09894v1)

Authors: Safwat Ali Khan

The demand for quality in mobile applications has increased greatly given
users' high reliance on them for daily tasks. Developers work tirelessly to
ensure that their applications are both functional and user-friendly. In
pursuit of this, Automated Input Generation (AIG) tools have emerged as a
promising solution for testing mobile applications by simulating user
interactions and exploring app functionalities. However, these tools face
significant challenges in navigating complex Graphical User Interfaces (GUIs),
and developers often have trouble understanding their output. More
specifically, AIG tools face difficulties in navigating out of certain screens,
such as login pages and advertisements, due to a lack of contextual
understanding which leads to suboptimal testing coverage. Furthermore, while
AIG tools can provide interaction traces consisting of action and screen
details, there is limited understanding of its coverage of higher level
functionalities, such as logging in, setting alarms, or saving notes.
Understanding these covered use cases are essential to ensure comprehensive
test coverage of app functionalities. Difficulty in testing mobile UIs can lead
to the design of complex interfaces, which can adversely affect users of
advanced age who often face usability barriers due to small buttons, cluttered
layouts, and unintuitive navigation. There exists many studies that highlight
these issues, but automated solutions for improving UI accessibility needs more
attention. This research seeks to enhance automated UI testing techniques by
learning the screen usage semantics of mobile apps and helping them navigate
more efficiently, offer more insights about tested functionalities and also
improve the usability of a mobile app's interface by identifying and mitigating
UI design issues.

### 2. [UICopilot: Automating UI Synthesis via Hierarchical Code Generation from Webpage Designs](http://arxiv.org/pdf/2505.09904v1)

Authors: Yi Gui, Yao Wan, Zhen Li, Zhongyi Zhang, Dongping Chen, Hongyu Zhang, Yi Su, Bohua Chen, Xing Zhou, Wenbin Jiang, Xiangliang Zhang

Automating the synthesis of User Interfaces (UIs) plays a crucial role in
enhancing productivity and accelerating the development lifecycle, reducing
both development time and manual effort. Recently, the rapid development of
Multimodal Large Language Models (MLLMs) has made it possible to generate
front-end Hypertext Markup Language (HTML) code directly from webpage designs.
However, real-world webpages encompass not only a diverse array of HTML tags
but also complex stylesheets, resulting in significantly lengthy code. The
lengthy code poses challenges for the performance and efficiency of MLLMs,
especially in capturing the structural information of UI designs. To address
these challenges, this paper proposes UICopilot, a novel approach to automating
UI synthesis via hierarchical code generation from webpage designs. The core
idea of UICopilot is to decompose the generation process into two stages:
first, generating the coarse-grained HTML hierarchical structure, followed by
the generation of fine-grained code. To validate the effectiveness of
UICopilot, we conduct experiments on a real-world dataset, i.e., WebCode2M.
Experimental results demonstrate that UICopilot significantly outperforms
existing baselines in both automatic evaluation metrics and human evaluations.
Specifically, statistical analysis reveals that the majority of human
annotators prefer the webpages generated by UICopilot over those produced by
GPT-4V.

### 3. [SVA-ICL: Improving LLM-based Software Vulnerability Assessment via In-Context Learning and Information Fusion](http://arxiv.org/pdf/2505.10008v1)

Authors: Chaoyang Gao, Xiang Chen, Guangbei Zhang

Context: Software vulnerability assessment (SVA) is critical for identifying,
evaluating, and prioritizing security weaknesses in software applications.
Objective: Despite the increasing application of large language models (LLMs)
in various software engineering tasks, their effectiveness in SVA remains
underexplored. Method: To address this gap, we introduce a novel approach
SVA-ICL, which leverages in-context learning (ICL) to enhance LLM performance.
Our approach involves the selection of high-quality demonstrations for ICL
through information fusion, incorporating both source code and vulnerability
descriptions. For source code, we consider semantic, lexical, and syntactic
similarities, while for vulnerability descriptions, we focus on textual
similarity. Based on the selected demonstrations, we construct context prompts
and consider DeepSeek-V2 as the LLM for SVA-ICL. Results: We evaluate the
effectiveness of SVA-ICL using a large-scale dataset comprising 12,071 C/C++
vulnerabilities. Experimental results demonstrate that SVA-ICL outperforms
state-of-the-art SVA baselines in terms of Accuracy, F1-score, and MCC
measures. Furthermore, ablation studies highlight the significance of component
customization in SVA-ICL, such as the number of demonstrations, the
demonstration ordering strategy, and the optimal fusion ratio of different
modalities. Conclusion: Our findings suggest that leveraging ICL with
information fusion can effectively improve the effectiveness of LLM-based SVA,
warranting further research in this direction.

### 4. [GBM Returns the Best Prediction Performance among Regression Approaches: A Case Study of Stack Overflow Code Quality](http://arxiv.org/pdf/2505.10019v1)

Authors: Sherlock A. Licorish, Brendon Woodford, Lakmal Kiyaduwa Vithanage, Osayande Pascal Omondiagbe

Practitioners are increasingly dependent on publicly available resources for
supporting their knowledge needs during software development. This has thus
caused a spotlight to be paced on these resources, where researchers have
reported mixed outcomes around the quality of these resources. Stack Overflow,
in particular, has been studied extensively, with evidence showing that code
resources on this platform can be of poor quality at times. Limited research
has explored the variables or factors that predict code quality on Stack
Overflow, but instead has focused on ranking content, identifying defects and
predicting future content. In many instances approaches used for prediction are
not evaluated to identify the best techniques. Contextualizing the Stack
Overflow code quality problem as regression-based, we examined the variables
that predict Stack Overflow (Java) code quality, and the regression approach
that provides the best predictive power. Six approaches were considered in our
evaluation, where Gradient Boosting Machine (GBM) stood out. In addition,
longer Stack Overflow code tended to have more code violations, questions that
were scored higher also attracted more views and the more answers that are
added to questions on Stack Overflow the more errors were typically observed in
the code that was provided. Outcomes here point to the value of the GBM
ensemble learning mechanism, and the need for the practitioner community to be
prudent when contributing and reusing Stack Overflow Java coding resource.

### 5. [Cross-Functional AI Task Forces (X-FAITs) for AI Transformation of Software Organizations](http://arxiv.org/pdf/2505.10021v1)

Authors: Lucas Gren, Robert Feldt

This experience report introduces the Cross-Functional AI Task Force (X-FAIT)
framework to bridge the gap between strategic AI ambitions and operational
execution within software-intensive organizations. Drawing from an Action
Research case study at a global Swedish enterprise, we identify and address
critical barriers such as departmental fragmentation, regulatory constraints,
and organizational inertia that can impede successful AI transformation. X-FAIT
employs force field analysis, executive sponsorship, cross-functional
integration, and systematic risk assessment strategies to coordinate efforts
across organizational boundaries, facilitating knowledge sharing and ensuring
AI initiatives align with objectives. The framework provides both theoretical
insights into AI-driven organizational transformation and practical guidance
for software organizations aiming to effectively integrate AI into their daily
workflows and, longer-term, products.

### 6. [Determining Absence of Unreasonable Risk: Approval Guidelines for an Automated Driving System Release](http://arxiv.org/pdf/2505.09880v1)

Authors: Francesca Favaro, Scott Schnelle, Laura Fraade-Blanar, Trent Victor, Mauricio Peña, Nick Webb, Holland Broce, Craig Paterson, Dan Smith

This paper provides an overview of how the determination of absence of
unreasonable risk can be operationalized. It complements previous theoretical
work published by existing developers of Automated Driving Systems (ADS) on the
overall engineering practices and methodologies for readiness determination.
Readiness determination is, at its core, a risk assessment process. It is aimed
at evaluating the residual risk associated with the deployment of a new
software release candidate. The paper proposes methodological criteria to
ground the readiness review process for an ADS release. While informed by
Waymo's experience in this domain, the criteria presented are agnostic of any
specific ADS technological solution and/or architectural choice, to support
broad implementation by others in the industry. The paper continues with a
discussion on governance and decision-making toward approval of a new software
release candidate for the ADS. The implementation of the presented criteria
requires the existence of appropriate safety management practices in addition
to many other cultural, procedural, and operational considerations. As such,
the paper is concluded by a statement of limitations for those wishing to
replicate part or all of its content.

### 7. [Probabilistic Bisimulation for Parameterized Anonymity and Uniformity Verification](http://arxiv.org/pdf/2505.09963v1)

Authors: Chih-Duo Hong, Anthony W. Lin, Philipp Rümmer, Rupak Majumdar

Bisimulation is crucial for verifying process equivalence in probabilistic
systems. This paper presents a novel logical framework for analyzing
bisimulation in probabilistic parameterized systems, namely, infinite families
of finite-state probabilistic systems. Our framework is built upon the
first-order theory of regular structures, which provides a decidable logic for
reasoning about these systems. We show that essential properties like anonymity
and uniformity can be encoded and verified within this framework in a manner
aligning with the principles of deductive software verification, where systems,
properties, and proofs are expressed in a unified decidable logic. By
integrating language inference techniques, we achieve full automation in
synthesizing candidate bisimulation proofs for anonymity and uniformity. We
demonstrate the efficacy of our approach by addressing several challenging
examples, including cryptographic protocols and randomized algorithms that were
previously beyond the reach of fully automated methods.

### 8. [Digital Natives, Digital Activists: Youth, Social Media and the Rise of Environmental Sustainability Movements](http://arxiv.org/pdf/2505.10158v1)

Authors: Manya Pandit, Triveni Magadum, Harshit Mittal, Omkar Kushwaha

The research examines the challenges revolving around young people's social
movements, activism regarding sustainability, as well as the accompanying
social media aspect, and how social media impacts environmental action. This
study focuses on the environmental craze on social media platforms and its
impact on young activists aged 16-25. With the advancement of social media, new
avenues have opened for participation in sustainability issues, especially for
the marginalized, as information moved through transnational networks at
lightning speed. Along with specific Formative Visual Storytelling methods, the
young leaders of the movement deploy hashtags and other online tools to capture
the attention of their peers and decision makers. Challenges persist with
"clicktivism" fatigue from the internet, and site limitations. This article
contributes to insights on emerging forms of civic activism by explaining how
digital natives adapt technology to reframe green activism. The research
suggests that effective digital environmental movements integrate online and
offline action, make it simple for individuals to get involved, and promote
tolerance to algorithmic modifications and climate care among participants.

### 9. [Are Large Language Models Robust in Understanding Code Against Semantics-Preserving Mutations?](http://arxiv.org/pdf/2505.10443v1)

Authors: Pedro Orvalho, Marta Kwiatkowska

Understanding the reasoning and robustness of Large Language Models (LLMs) is
critical for their reliable use in programming tasks. While recent studies have
assessed LLMs' ability to predict program outputs, most focus solely on the
accuracy of those predictions, without evaluating the reasoning behind them.
Moreover, it has been observed on mathematical reasoning tasks that LLMs can
arrive at correct answers through flawed logic, raising concerns about similar
issues in code understanding.
  In this work, we evaluate whether state-of-the-art LLMs with up to 8B
parameters can reason about Python programs or are simply guessing. We apply
five semantics-preserving code mutations: renaming variables, mirroring
comparison expressions, swapping if-else branches, converting for loops to
while, and loop unrolling. These mutations maintain program semantics while
altering its syntax. We evaluated six LLMs and performed a human expert
analysis using LiveCodeBench to assess whether the correct predictions are
based on sound reasoning. We also evaluated prediction stability across
different code mutations on LiveCodeBench and CruxEval. Our findings show that
some LLMs, such as Llama3.2, produce correct predictions based on flawed
reasoning in up to 61% of cases. Furthermore, LLMs often change predictions in
response to our code mutations, indicating limited robustness in their semantic
understanding.

### 10. [Are Sparse Autoencoders Useful for Java Function Bug Detection?](http://arxiv.org/pdf/2505.10375v1)

Authors: Rui Melo, Claudia Mamede, Andre Catarino, Rui Abreu, Henrique Lopes Cardoso

Software vulnerabilities such as buffer overflows and SQL injections are a
major source of security breaches. Traditional methods for vulnerability
detection remain essential but are limited by high false positive rates,
scalability issues, and reliance on manual effort. These constraints have
driven interest in AI-based approaches to automated vulnerability detection and
secure code generation. While Large Language Models (LLMs) have opened new
avenues for classification tasks, their complexity and opacity pose challenges
for interpretability and deployment. Sparse Autoencoder offer a promising
solution to this problem. We explore whether SAEs can serve as a lightweight,
interpretable alternative for bug detection in Java functions. We evaluate the
effectiveness of SAEs when applied to representations from GPT-2 Small and
Gemma 2B, examining their capacity to highlight buggy behaviour without
fine-tuning the underlying LLMs. We found that SAE-derived features enable bug
detection with an F1 score of up to 89%, consistently outperforming fine-tuned
transformer encoder baselines. Our work provides the first empirical evidence
that SAEs can be used to detect software bugs directly from the internal
representations of pretrained LLMs, without any fine-tuning or task-specific
supervision.

### Social and Information Networks

### 1. [Community Fact-Checks Do Not Break Follower Loyalty](http://arxiv.org/pdf/2505.10254v1)

Authors: Michelle Bobek, Nicolas Pröllochs

Major social media platforms increasingly adopt community-based fact-checking
to address misinformation on their platforms. While previous research has
largely focused on its effect on engagement (e.g., reposts, likes), an
understanding of how fact-checking affects a user's follower base is missing.
In this study, we employ quasi-experimental methods to causally assess whether
users lose followers after their posts are corrected via community fact-checks.
Based on time-series data on follower counts for N=3516 community fact-checked
posts from X, we find that community fact-checks do not lead to meaningful
declines in the follower counts of users who post misleading content. This
suggests that followers of spreaders of misleading posts tend to remain loyal
and do not view community fact-checks as a sufficient reason to disengage. Our
findings underscore the need for complementary interventions to more
effectively disincentivize the production of misinformation on social media.

### 2. [Characterizing AI-Generated Misinformation on Social Media](http://arxiv.org/pdf/2505.10266v1)

Authors: Chiara Drolsbach, Nicolas Pröllochs

AI-generated misinformation (e.g., deepfakes) poses a growing threat to
information integrity on social media. However, prior research has largely
focused on its potential societal consequences rather than its real-world
prevalence. In this study, we conduct a large-scale empirical analysis of
AI-generated misinformation on the social media platform X. Specifically, we
analyze a dataset comprising N=91,452 misleading posts, both AI-generated and
non-AI-generated, that have been identified and flagged through X's Community
Notes platform. Our analysis yields four main findings: (i) AI-generated
misinformation is more often centered on entertaining content and tends to
exhibit a more positive sentiment than conventional forms of misinformation,
(ii) it is more likely to originate from smaller user accounts, (iii) despite
this, it is significantly more likely to go viral, and (iv) it is slightly less
believable and harmful compared to conventional misinformation. Altogether, our
findings highlight the unique characteristics of AI-generated misinformation on
social media. We discuss important implications for platforms and future
research.

### 3. [Scalable Approximate Biclique Counting over Large Bipartite Graphs](http://arxiv.org/pdf/2505.10471v1)

Authors: Jingbang Chen, Weinuo Li, Yingli Zhou, Hangrui Zhou, Qiuyang Mang, Can Wang, Yixiang Fang, Chenhao Ma

Counting $(p,q)$-bicliques in bipartite graphs is crucial for a variety of
applications, from recommendation systems to cohesive subgraph analysis. Yet,
it remains computationally challenging due to the combinatorial explosion to
exactly count the $(p,q)$-bicliques. In many scenarios, e.g., graph kernel
methods, however, exact counts are not strictly required. To design a scalable
and high-quality approximate solution, we novelly resort to $(p,q)$-broom, a
special spanning tree of the $(p,q)$-biclique, which can be counted via graph
coloring and efficient dynamic programming. Based on the intermediate results
of the dynamic programming, we propose an efficient sampling algorithm to
derive the approximate $(p,q)$-biclique count from the $(p,q)$-broom counts.
Theoretically, our method offers unbiased estimates with provable error
guarantees. Empirically, our solution outperforms existing approximation
techniques in both accuracy (up to 8$\times$ error reduction) and runtime (up
to 50$\times$ speedup) on nine real-world bipartite networks, providing a
scalable solution for large-scale $(p,q)$-biclique counting.

### 4. [Advancing Community Detection with Graph Convolutional Neural Networks: Bridging Topological and Attributive Cohesion](http://arxiv.org/pdf/2505.10197v1)

Authors: Anjali de Silva, Gang Chen, Hui Ma, Seyed Mohammad Nekooei, Xingquan Zuo

Community detection, a vital technology for real-world applications, uncovers
cohesive node groups (communities) by leveraging both topological and attribute
similarities in social networks. However, existing Graph Convolutional Networks
(GCNs) trained to maximize modularity often converge to suboptimal solutions.
Additionally, directly using human-labeled communities for training can
undermine topological cohesiveness by grouping disconnected nodes based solely
on node attributes. We address these issues by proposing a novel Topological
and Attributive Similarity-based Community detection (TAS-Com) method. TAS-Com
introduces a novel loss function that exploits the highly effective and
scalable Leiden algorithm to detect community structures with global optimal
modularity. Leiden is further utilized to refine human-labeled communities to
ensure connectivity within each community, enabling TAS-Com to detect community
structures with desirable trade-offs between modularity and compliance with
human labels. Experimental results on multiple benchmark networks confirm that
TAS-Com can significantly outperform several state-of-the-art algorithms.

### 5. [Empirically evaluating commonsense intelligence in large language models with large-scale human judgments](http://arxiv.org/pdf/2505.10309v1)

Authors: Tuan Dung Nguyen, Duncan J. Watts, Mark E. Whiting

Commonsense intelligence in machines is often assessed by static benchmarks
that compare a model's output against human-prescribed correct labels. An
important, albeit implicit, assumption of these labels is that they accurately
capture what any human would think, effectively treating human common sense as
homogeneous. However, recent empirical work has shown that humans vary
enormously in what they consider commonsensical; thus what appears self-evident
to one benchmark designer may not be so to another. Here, we propose a novel
method for evaluating common sense in artificial intelligence (AI),
specifically in large language models (LLMs), that incorporates empirically
observed heterogeneity among humans by measuring the correspondence between a
model's judgment and that of a human population. We first find that, when
treated as independent survey respondents, most LLMs remain below the human
median in their individual commonsense competence. Second, when used as
simulators of a hypothetical population, LLMs correlate with real humans only
modestly in the extent to which they agree on the same set of statements. In
both cases, smaller, open-weight models are surprisingly more competitive than
larger, proprietary frontier models. Our evaluation framework, which ties
commonsense intelligence to its cultural basis, contributes to the growing call
for adapting AI models to human collectivities that possess different, often
incompatible, social stocks of knowledge.

### 6. [Reproducing the first and second moment of empirical degree distributions](http://arxiv.org/pdf/2505.10373v1)

Authors: Mattia Marzi, Francesca Giuffrida, Diego Garlaschelli, Tiziano Squartini

The study of probabilistic models for the analysis of complex networks
represents a flourishing research field. Among the former, Exponential Random
Graphs (ERGs) have gained increasing attention over the years. So far, only
linear ERGs have been extensively employed to gain insight into the structural
organisation of real-world complex networks. None, however, is capable of
accounting for the variance of the empirical degree distribution. To this aim,
non-linear ERGs must be considered. After showing that the usual mean-field
approximation forces the degree-corrected version of the two-star model to
degenerate, we define a fitness-induced variant of it. Such a `softened' model
is capable of reproducing the sample variance, while retaining the explanatory
power of its linear counterpart, within a purely canonical framework.

### Systems and Control

### 1. [The Path Integral Bottleneck: Exploring the Control-Compute Tradeoff](http://arxiv.org/pdf/2505.09896v1)

Authors: Justin Ting, Jing Shuang Li

Executing a control sequence requires some computation effort. Intuitively, a
high-effort, fine-grained computation should result in better control (e.g.
lower cost), whereas little to no computation effort would lead to worse
control. To quantify and explore the tradeoff between control performance and
compute effort, we present the Path Integral Bottleneck (PIB), a fusion of the
Path Integral (PI) optimal control and Information Bottleneck (IB) frameworks.
Both frameworks provide flexible and probabilistic descriptions of control. The
PI does not limit itself to a particular control law, and the IB is not bound
to any specific state encoding. Combining the generality of both frameworks
enables us to produce an analytical description of the control-compute
tradeoff. We provide PIB formulations for both continuous and discrete random
variables. With these formulations, we can plot a tradeoff curve between
performance and computation effort for any given plant description and control
cost function. Simulations of a cart-pole for both the continuous and discrete
variable cases reveal fundamental control-compute tradeoffs, exposing regions
where the task performance-per-compute is higher than others.

### 2. [Stability and Convergence Analysis of Multi-Agent Consensus with Communication Delays: A Lambert W Function Approach](http://arxiv.org/pdf/2505.09897v1)

Authors: Layan Badran, Kiarash Aryankia, Rastko R. Selmic

This paper investigates the effect of constant time delay in weakly connected
multi-agent systems modeled by double integrator dynamics. A novel analytical
approach is proposed to establish an upper bound on the permissible time delay
that ensures stability and consensus convergence. The analysis employs the
Lambert W function method in higher-dimensional systems to derive explicit
conditions under which consensus is achieved. The theoretical results are
rigorously proven and provide insight into the allowable delay margins. The
analysis applies to general leaderless undirected network topologies. The
framework also accounts for complex and realistic delays, including
non-commensurate communication delays. Numerical examples are provided to
demonstrate the effectiveness of the proposed method.

### 3. [Event-Triggered Synergistic Controllers with Dwell-Time Transmission](http://arxiv.org/pdf/2505.09980v1)

Authors: Xuanzhi Zhu, Pedro Casau, Carlos Silvestre

We propose novel event-triggered synergistic controllers for nonlinear
continuous-time plants by incorporating event-triggered control into
stabilizing synergistic controllers. We highlight that a naive application of
common event-triggering conditions may not ensure dwell-time transmission due
to the joint jumping dynamics of the closed-loop system. Under mild conditions,
we develop a suite of event-triggered synergistic controllers that guarantee
both dwell-time transmission and global asymptotic stability. Through numerical
simulations, we demonstrate the effectiveness of our controller applied to the
problem of rigid body attitude stabilization.

### 4. [Planar Herding of Multiple Evaders with a Single Herder](http://arxiv.org/pdf/2505.10048v1)

Authors: Rishabh Kumar Singh, Debraj Chakraborty

A planar herding problem is considered, where a superior pursuer herds a
flock of non-cooperative, inferior evaders around a predefined target point. An
inverse square law of repulsion is assumed between the pursuer and each evader.
Two classes of pursuer trajectories are proposed: (i) a constant
angular-velocity spiral, and (ii) a constant angular-velocity circle, both
centered around the target point. For the spiraling pursuer, the radial
velocity is dynamically adjusted based on a feedback law that depends on the
instantaneous position of the evader, which is located at the farthest distance
from the target at the start of the game. It is shown that, under suitable
choices of the model parameters, all the evaders are herded into an arbitrarily
small limit cycle around the target point. Meanwhile, the pursuer also
converges onto a circular trajectory around the target. The conditions for the
stability of these limit cycles are derived. For the circling pursuer, similar
guarantees are provided along with explicit formulas for the radii of the limit
cycles.

### 5. [Improving Power Systems Controllability via Edge Centrality Measures](http://arxiv.org/pdf/2505.10059v1)

Authors: MirSaleh Bahavarnia, Muhammad Nadeem, Ahmad F. Taha

Improving the controllability of power networks is crucial as they are highly
complex networks operating in synchrony; even minor perturbations can cause
desynchronization and instability. To that end, one needs to assess the
criticality of key network components (buses and lines) in terms of their
impact on system performance. Traditional methods to identify the key
nodes/edges in power networks often rely on static centrality measures based on
the network's topological structure ignoring the network's dynamic behavior. In
this paper, using multi-machine power network models and a new
control-theoretic edge centrality matrix (ECM) approach, we: (i) quantify the
influence of edges (i.e., the line susceptances) in terms of controllability
performance metrics, (ii) identify the most influential lines, and (iii)
compute near-optimal edge modifications that improve the power network
controllability. Employing various IEEE power network benchmarks, we validate
the effectiveness of the ECM-based algorithm and demonstrate improvements in
system reachability, control, and damping performance.

### 6. [DB InfraGO's Automated Dispatching Assistant ADA-PMB](http://arxiv.org/pdf/2505.10085v1)

Authors: Stephan Zieger, Hannah Richta

As railway infrastructure manager, DB InfraGO AG is faced with the challenge
of offering fluid and punctual operation despite rising demand and increased
construction activity. The high capacity utilisation, especially in the core
network sections, causes delays to be propagated quickly and widely across the
entire network. Up to now, conflicts between train runs can be identified
automatically, but dispatching measures have been based on past human
experience.
  An automated dispatching assistance system is currently being piloted to
provide support for train dispatchers in their work. The aim is to offer them
helpful dispatching recommendations, particularly in stressful situations with
a high conflict density in the network section under consideration, in order to
ensure the most efficient operation of the system.
  The recommendations are currently displayed separately alongside the central
control system. In future, they will be integrated into the central control
system, which will significantly simplify communication between the train
dispatcher and signal setter. Further development steps for the integration
process are also presented and discussed.

### 7. [Can On Body Sensing Be Spatial Adaptive?](http://arxiv.org/pdf/2505.10546v1)

Authors: Shubham Rohal, Dong Yoon Lee, Phuc Nguyen, Shijia Pan

Wearable sensors are typically affixed to specific locations on the human
body, and their position remains static, only changing unintentionally due to
motion artifacts. This static configuration introduces significant limitations.
As a result, current systems miss the opportunity to capture dynamic
physiological data from diverse body regions. This research investigates the
potential of developing movable sensors that adaptively reposition themselves
to sample different areas of interest on the body, addressing gaps in spatial
coverage. We designed, developed, and fabricated a 3 x 3 matrix platform to
support moving sensors from one location to another. We validated the
feasibility through simulations on a matrix of up to 9 x 9 locations with up to
16 concurrent sensors and real-world prototype characterization.

### 8. [Dynamic Beam-Stabilized, Additive-Printed Flexible Antenna Arrays with On-Chip Rapid Insight Generation](http://arxiv.org/pdf/2505.09870v1)

Authors: Sreeni Poolakkal, Abdullah Islam, Arpit Rao, Shrestha Bansal, Ted Dabrowski, Kalsi Kwan, Zhongxuan Wang, Amit Kumar Mishra, Julio Navarro, Shenqiang Ren, John Williams, Sudip Shekhar, Subhanshu Gupta

Conformal phased arrays promise shape-changing properties, multiple degrees
of freedom to the scan angle, and novel applications in wearables, aerospace,
defense, vehicles, and ships. However, they have suffered from two critical
limitations. (1) Although most applications require on-the-move communication
and sensing, prior conformal arrays have suffered from dynamic
deformation-induced beam pointing errors. We introduce a Dynamic
Beam-Stabilized (DBS) processor capable of beam adaptation through on-chip
real-time control of fundamental gain, phase, and delay for each element. (2)
Prior conformal arrays have leveraged additive printing to enhance flexibility,
but conventional printable inks based on silver are expensive, and those based
on copper suffer from spontaneous metal oxidation that alters trace impedance
and degrades beamforming performance. We instead leverage a low-cost Copper
Molecular Decomposition (CuMOD) ink with < 0.1% variation per degree C with
temperature and strain and correct any residual deformity in real-time using
the DBS processor. Demonstrating unified material and physical deformation
correction, our CMOS DBS processor is low power, low-area, and easily scalable
due to a tile architecture, thereby ideal for on-device implementations.

### 9. [Hyper Yoshimura: How a slight tweak on a classical folding pattern unleashes meta-stability for deployable robots](http://arxiv.org/pdf/2505.09919v1)

Authors: Ziyang Zhou, Yogesh Phalak, Vishrut Deshpande, Ian Walker, Suyi Li

Deployable structures inspired by origami offer lightweight, compact, and
reconfigurable solutions for robotic and architectural applications. We present
a geometric and mechanical framework for Yoshimura-Ori modules that supports a
diverse set of metastable states, including newly identified asymmetric
"pop-out" and "hyperfolded" configurations. These states are governed by three
parameters -- tilt angle, phase shift, and slant height -- and enable discrete,
programmable transformations. Using this model, we develop forward and inverse
kinematic strategies to stack modules into deployable booms that approximate
complex 3D shapes. We validate our approach through mechanical tests and
demonstrate a tendon- and pneumatically-actuated Yoshimura Space Crane capable
of object manipulation, solar tracking, and high load-bearing performance. A
meter-scale solar charging station further illustrates the design's
scalability. These results establish Yoshimura-Ori structures as a promising
platform for adaptable, multifunctional deployable systems in both terrestrial
and space environments.

### 10. [Offline Reinforcement Learning for Microgrid Voltage Regulation](http://arxiv.org/pdf/2505.09920v1)

Authors: Shan Yang, Yongli Zhu

This paper presents a study on using different offline reinforcement learning
algorithms for microgrid voltage regulation with solar power penetration. When
environment interaction is unviable due to technical or safety reasons, the
proposed approach can still obtain an applicable model through offline-style
training on a previously collected dataset, lowering the negative impact of
lacking online environment interactions. Experiment results on the IEEE 33-bus
system demonstrate the feasibility and effectiveness of the proposed approach
on different offline datasets, including the one with merely low-quality
experience.

### Machine Learning (Statistics Category)

### 1. [Path Gradients after Flow Matching](http://arxiv.org/pdf/2505.10139v1)

Authors: Lorenz Vaitl, Leon Klein

Boltzmann Generators have emerged as a promising machine learning tool for
generating samples from equilibrium distributions of molecular systems using
Normalizing Flows and importance weighting. Recently, Flow Matching has helped
speed up Continuous Normalizing Flows (CNFs), scale them to more complex
molecular systems, and minimize the length of the flow integration
trajectories. We investigate the benefits of using path gradients to fine-tune
CNFs initially trained by Flow Matching, in the setting where a target energy
is known. Our experiments show that this hybrid approach yields up to a
threefold increase in sampling efficiency for molecular systems, all while
using the same model, a similar computational budget and without the need for
additional sampling. Furthermore, by measuring the length of the flow
trajectories during fine-tuning, we show that path gradients largely preserve
the learned structure of the flow.

### 2. [One-Stage Top-$k$ Learning-to-Defer: Score-Based Surrogates with Theoretical Guarantees](http://arxiv.org/pdf/2505.10160v1)

Authors: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

We introduce the first one-stage Top-$k$ Learning-to-Defer framework, which
unifies prediction and deferral by learning a shared score-based model that
selects the $k$ most cost-effective entities-labels or experts-per input. While
existing one-stage L2D methods are limited to deferring to a single expert, our
approach jointly optimizes prediction and deferral across multiple entities
through a single end-to-end objective. We define a cost-sensitive loss and
derive a novel convex surrogate that is independent of the cardinality
parameter $k$, enabling generalization across Top-$k$ regimes without
retraining. Our formulation recovers the Top-1 deferral policy of prior
score-based methods as a special case, and we prove that our surrogate is both
Bayes-consistent and $\mathcal{H}$-consistent under mild assumptions. We
further introduce an adaptive variant, Top-$k(x)$, which dynamically selects
the number of consulted entities per input to balance predictive accuracy and
consultation cost. Experiments on CIFAR-10 and SVHN confirm that our one-stage
Top-$k$ method strictly outperforms Top-1 deferral, while Top-$k(x)$ achieves
superior accuracy-cost trade-offs by tailoring allocations to input complexity.

### 3. [Efficient MCMC Sampling with Expensive-to-Compute and Irregular Likelihoods](http://arxiv.org/pdf/2505.10448v1)

Authors: Conor Rosato, Harvinder Lehal, Simon Maskell, Lee Devlin, Malcolm Strens

Bayesian inference with Markov Chain Monte Carlo (MCMC) is challenging when
the likelihood function is irregular and expensive to compute. We explore
several sampling algorithms that make use of subset evaluations to reduce
computational overhead. We adapt the subset samplers for this setting where
gradient information is not available or is unreliable. To achieve this, we
introduce data-driven proxies in place of Taylor expansions and define a novel
computation-cost aware adaptive controller. We undertake an extensive
evaluation for a challenging disease modelling task and a configurable task
with similar irregularity in the likelihood surface. We find our improved
version of Hierarchical Importance with Nested Training Samples (HINTS), with
adaptive proposals and a data-driven proxy, obtains the best sampling error in
a fixed computational budget. We conclude that subset evaluations can provide
cheap and naturally-tempered exploration, while a data-driven proxy can
pre-screen proposals successfully in explored regions of the state space. These
two elements combine through hierarchical delayed acceptance to achieve
efficient, exact sampling.

### 4. [Topology-driven identification of repetitions in multi-variate time series](http://arxiv.org/pdf/2505.10004v1)

Authors: Simon Schindler, Elias Steffen Reich, Saverio Messineo, Simon Hoher, Stefan Huber

Many multi-variate time series obtained in the natural sciences and
engineering possess a repetitive behavior, as for instance state-space
trajectories of industrial machines in discrete automation. Recovering the
times of recurrence from such a multi-variate time series is of a fundamental
importance for many monitoring and control tasks. For a periodic time series
this is equivalent to determining its period length. In this work we present a
persistent homology framework to estimate recurrence times in multi-variate
time series with different generalizations of cyclic behavior (periodic,
repetitive, and recurring). To this end, we provide three specialized methods
within our framework that are provably stable and validate them using
real-world data, including a new benchmark dataset from an injection molding
machine.

### 5. [Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](http://arxiv.org/pdf/2505.10007v1)

Authors: Zijun Chen, Shengbo Wang, Nian Si

Motivated by practical applications where stable long-term performance is
critical-such as robotics, operations research, and healthcare-we study the
problem of distributionally robust (DR) average-reward reinforcement learning.
We propose two algorithms that achieve near-optimal sample complexity. The
first reduces the problem to a DR discounted Markov decision process (MDP),
while the second, Anchored DR Average-Reward MDP, introduces an anchoring state
to stabilize the controlled transition kernels within the uncertainty set.
Assuming the nominal MDP is uniformly ergodic, we prove that both algorithms
attain a sample complexity of $\widetilde{O}\left(|\mathbf{S}||\mathbf{A}|
t_{\mathrm{mix}}^2\varepsilon^{-2}\right)$ for estimating the optimal policy as
well as the robust average reward under KL and $f_k$-divergence-based
uncertainty sets, provided the uncertainty radius is sufficiently small. Here,
$\varepsilon$ is the target accuracy, $|\mathbf{S}|$ and $|\mathbf{A}|$ denote
the sizes of the state and action spaces, and $t_{\mathrm{mix}}$ is the mixing
time of the nominal MDP. This represents the first finite-sample convergence
guarantee for DR average-reward reinforcement learning. We further validate the
convergence rates of our algorithms through numerical experiments.

### 6. [A Scalable Gradient-Based Optimization Framework for Sparse Minimum-Variance Portfolio Selection](http://arxiv.org/pdf/2505.10099v1)

Authors: Sarat Moka, Matias Quiroz, Vali Asimit, Samuel Muller

Portfolio optimization involves selecting asset weights to minimize a
risk-reward objective, such as the portfolio variance in the classical
minimum-variance framework. Sparse portfolio selection extends this by imposing
a cardinality constraint: only $k$ assets from a universe of $p$ may be
included. The standard approach models this problem as a mixed-integer
quadratic program and relies on commercial solvers to find the optimal
solution. However, the computational costs of such methods increase
exponentially with $k$ and $p$, making them too slow for problems of even
moderate size. We propose a fast and scalable gradient-based approach that
transforms the combinatorial sparse selection problem into a constrained
continuous optimization task via Boolean relaxation, while preserving
equivalence with the original problem on the set of binary points. Our
algorithm employs a tunable parameter that transmutes the auxiliary objective
from a convex to a concave function. This allows a stable convex starting
point, followed by a controlled path toward a sparse binary solution as the
tuning parameter increases and the objective moves toward concavity. In
practice, our method matches commercial solvers in asset selection for most
instances and, in rare instances, the solution differs by a few assets whilst
showing a negligible error in portfolio variance.

### 7. [Whitened Score Diffusion: A Structured Prior for Imaging Inverse Problems](http://arxiv.org/pdf/2505.10311v1)

Authors: Jeffrey Alido, Tongyu Li, Yu Sun, Lei Tian

Conventional score-based diffusion models (DMs) may struggle with anisotropic
Gaussian diffusion processes due to the required inversion of covariance
matrices in the denoising score matching training objective
\cite{vincent_connection_2011}. We propose Whitened Score (WS) diffusion
models, a novel SDE-based framework that learns the Whitened Score function
instead of the standard score. This approach circumvents covariance inversion,
extending score-based DMs by enabling stable training of DMs on arbitrary
Gaussian forward noising processes. WS DMs establish equivalence with FM for
arbitrary Gaussian noise, allow for tailored spectral inductive biases, and
provide strong Bayesian priors for imaging inverse problems with structured
noise. We experiment with a variety of computational imaging tasks using the
CIFAR and CelebA ($64\times64$) datasets and demonstrate that WS diffusion
priors trained on anisotropic Gaussian noising processes consistently
outperform conventional diffusion priors based on isotropic Gaussian noise.

### 8. [PIF: Anomaly detection via preference embedding](http://arxiv.org/pdf/2505.10441v1)

Authors: Filippo Leveni, Luca Magri, Giacomo Boracchi, Cesare Alippi

We address the problem of detecting anomalies with respect to structured
patterns. To this end, we conceive a novel anomaly detection method called PIF,
that combines the advantages of adaptive isolation methods with the flexibility
of preference embedding. Specifically, we propose to embed the data in a high
dimensional space where an efficient tree-based method, PI-Forest, is employed
to compute an anomaly score. Experiments on synthetic and real datasets
demonstrate that PIF favorably compares with state-of-the-art anomaly detection
techniques, and confirm that PI-Forest is better at measuring arbitrary
distances and isolate points in the preference space.

### 9. [FlowVAT: Normalizing Flow Variational Inference with Affine-Invariant Tempering](http://arxiv.org/pdf/2505.10466v1)

Authors: Juehang Qin, Shixiao Liang, Christopher Tunnell

Multi-modal and high-dimensional posteriors present significant challenges
for variational inference, causing mode-seeking behavior and collapse despite
the theoretical expressiveness of normalizing flows. Traditional annealing
methods require temperature schedules and hyperparameter tuning, falling short
of the goal of truly black-box variational inference. We introduce FlowVAT, a
conditional tempering approach for normalizing flow variational inference that
addresses these limitations. Our method tempers both the base and target
distributions simultaneously, maintaining affine-invariance under tempering. By
conditioning the normalizing flow on temperature, we leverage overparameterized
neural networks' generalization capabilities to train a single flow
representing the posterior across a range of temperatures. This preserves modes
identified at higher temperatures when sampling from the variational posterior
at $T = 1$, mitigating standard variational methods' mode-seeking behavior. In
experiments with 2, 10, and 20 dimensional multi-modal distributions, FlowVAT
outperforms traditional and adaptive annealing methods, finding more modes and
achieving better ELBO values, particularly in higher dimensions where existing
approaches fail. Our method requires minimal hyperparameter tuning and does not
require an annealing schedule, advancing toward fully-automatic black-box
variational inference for complicated posteriors.

### 10. [Neural Thermodynamic Laws for Large Language Model Training](http://arxiv.org/pdf/2505.10559v1)

Authors: Ziming Liu, Yizhou Liu, Jeff Gore, Max Tegmark

Beyond neural scaling laws, little is known about the laws underlying large
language models (LLMs). We introduce Neural Thermodynamic Laws (NTL) -- a new
framework that offers fresh insights into LLM training dynamics. On the
theoretical side, we demonstrate that key thermodynamic quantities (e.g.,
temperature, entropy, heat capacity, thermal conduction) and classical
thermodynamic principles (e.g., the three laws of thermodynamics and the
equipartition theorem) naturally emerge under river-valley loss landscape
assumptions. On the practical side, this scientific perspective yields
intuitive guidelines for designing learning rate schedules.

