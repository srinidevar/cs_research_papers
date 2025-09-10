# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-09 17:00:27.931192 PST.

### Artificial Intelligence

### 1. [REMI: A Novel Causal Schema Memory Architecture for Personalized Lifestyle Recommendation Agents](http://arxiv.org/pdf/2509.06269v1)

Authors: Vishal Raman, Vijai Aravindh R, Abhijith Ragav

Personalized AI assistants often struggle to incorporate complex personal
data and causal knowledge, leading to generic advice that lacks explanatory
power. We propose REMI, a Causal Schema Memory architecture for a multimodal
lifestyle agent that integrates a personal causal knowledge graph, a causal
reasoning engine, and a schema based planning module. The idea is to deliver
explainable, personalized recommendations in domains like fashion, personal
wellness, and lifestyle planning. Our architecture uses a personal causal graph
of the user's life events and habits, performs goal directed causal traversals
enriched with external knowledge and hypothetical reasoning, and retrieves
adaptable plan schemas to generate tailored action plans. A Large Language
Model orchestrates these components, producing answers with transparent causal
explanations. We outline the CSM system design and introduce new evaluation
metrics for personalization and explainability, including Personalization
Salience Score and Causal Reasoning Accuracy, to rigorously assess its
performance. Results indicate that CSM based agents can provide more context
aware, user aligned recommendations compared to baseline LLM agents. This work
demonstrates a novel approach to memory augmented, causal reasoning in
personalized agents, advancing the development of transparent and trustworthy
AI lifestyle assistants.

### 2. [TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning](http://arxiv.org/pdf/2509.06278v1)

Authors: Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu

Table reasoning is crucial for leveraging structured data in domains such as
finance, healthcare, and scientific research. While large language models
(LLMs) show promise in multi-step reasoning, purely text-based methods often
struggle with the complex numerical computations and fine-grained operations
inherently required in this task. Tool-integrated reasoning improves
computational accuracy via explicit code execution, yet existing systems
frequently rely on rigid patterns, supervised imitation, and lack true
autonomous adaptability. In this paper, we present TableMind, an LLM-driven
table reasoning agent that (i) autonomously performs multi-turn tool
invocation, (ii) writes and executes data-analyzing code in a secure sandbox
environment for data analysis and precise numerical reasoning, and (iii)
exhibits high-level capabilities such as planning and self-reflection to adapt
strategies. To realize these capabilities, we adopt a two-stage fine-tuning
paradigm built on top of a powerful pre-trained language model: supervised
fine-tuning on high-quality reasoning trajectories to establish effective tool
usage patterns, followed by reinforcement fine-tuning to optimize
multi-objective strategies. In particular, we propose Rank-Aware Policy
Optimization (RAPO), which increases the update weight of high-quality
trajectories when their output probabilities are lower than those of
low-quality ones, thereby guiding the model more consistently toward better and
more accurate answers. Extensive experiments on several mainstream benchmarks
demonstrate that TableMind achieves superior performance compared to
competitive baselines, yielding substantial gains in both reasoning accuracy
and computational precision.

### 3. [Can AI Make Energy Retrofit Decisions? An Evaluation of Large Language Models](http://arxiv.org/pdf/2509.06307v1)

Authors: Lei Shu, Dong Zhao

Conventional approaches to building energy retrofit decision making suffer
from limited generalizability and low interpretability, hindering adoption in
diverse residential contexts. With the growth of Smart and Connected
Communities, generative AI, especially large language models (LLMs), may help
by processing contextual information and producing practitioner readable
recommendations. We evaluate seven LLMs (ChatGPT, DeepSeek, Gemini, Grok,
Llama, and Claude) on residential retrofit decisions under two objectives:
maximizing CO2 reduction (technical) and minimizing payback period
(sociotechnical). Performance is assessed on four dimensions: accuracy,
consistency, sensitivity, and reasoning, using a dataset of 400 homes across 49
US states. LLMs generate effective recommendations in many cases, reaching up
to 54.5 percent top 1 match and 92.8 percent within top 5 without fine tuning.
Performance is stronger for the technical objective, while sociotechnical
decisions are limited by economic trade offs and local context. Agreement
across models is low, and higher performing models tend to diverge from others.
LLMs are sensitive to location and building geometry but less sensitive to
technology and occupant behavior. Most models show step by step, engineering
style reasoning, but it is often simplified and lacks deeper contextual
awareness. Overall, LLMs are promising assistants for energy retrofit decision
making, but improvements in accuracy, consistency, and context handling are
needed for reliable practice.

### 4. [Large Language Models as Virtual Survey Respondents: Evaluating Sociodemographic Response Generation](http://arxiv.org/pdf/2509.06337v1)

Authors: Jianpeng Zhao, Chenyu Yuan, Weiming Luo, Haoling Xie, Guangwei Zhang, Steven Jige Quan, Zixuan Yuan, Pengyang Wang, Denghui Zhang

Questionnaire-based surveys are foundational to social science research and
public policymaking, yet traditional survey methods remain costly,
time-consuming, and often limited in scale. This paper explores a new paradigm:
simulating virtual survey respondents using Large Language Models (LLMs). We
introduce two novel simulation settings, namely Partial Attribute Simulation
(PAS) and Full Attribute Simulation (FAS), to systematically evaluate the
ability of LLMs to generate accurate and demographically coherent responses. In
PAS, the model predicts missing attributes based on partial respondent
profiles, whereas FAS involves generating complete synthetic datasets under
both zero-context and context-enhanced conditions. We curate a comprehensive
benchmark suite, LLM-S^3 (Large Language Model-based Sociodemographic Survey
Simulation), that spans 11 real-world public datasets across four sociological
domains. Our evaluation of multiple mainstream LLMs (GPT-3.5/4 Turbo, LLaMA
3.0/3.1-8B) reveals consistent trends in prediction performance, highlights
failure modes, and demonstrates how context and prompt design impact simulation
fidelity. This work establishes a rigorous foundation for LLM-driven survey
simulations, offering scalable and cost-effective tools for sociological
research and policy evaluation. Our code and dataset are available at:
https://github.com/dart-lab-research/LLM-S-Cube-Benchmark

### 5. [Evaluating Multi-Turn Bargain Skills in LLM-Based Seller Agent](http://arxiv.org/pdf/2509.06341v1)

Authors: Issue Yishu Wang, Kakam Chong, Xiaofeng Wang, Xu Yan, DeXin Kong, Chen Ju, Ming Chen, Shuai Xiao, Shuguang Han, jufeng chen

In online second-hand marketplaces, multi-turn bargaining is a crucial part
of seller-buyer interactions. Large Language Models (LLMs) can act as seller
agents, negotiating with buyers on behalf of sellers under given business
constraints. A critical ability for such agents is to track and accurately
interpret cumulative buyer intents across long negotiations, which directly
impacts bargaining effectiveness. We introduce a multi-turn evaluation
framework for measuring the bargaining ability of seller agents in e-commerce
dialogues. The framework tests whether an agent can extract and track buyer
intents. Our contributions are: (1) a large-scale e-commerce bargaining
benchmark spanning 622 categories, 9,892 products, and 3,014 tasks; (2) a
turn-level evaluation framework grounded in Theory of Mind (ToM) with annotated
buyer intents, moving beyond outcome-only metrics; and (3) an automated
pipeline that extracts reliable intent from massive dialogue data.

### 6. [Teaching AI Stepwise Diagnostic Reasoning with Report-Guided Chain-of-Thought Learning](http://arxiv.org/pdf/2509.06409v1)

Authors: Yihong Luo, Wenwu He, Zhuo-Xu Cui, Dong Liang

This study presents DiagCoT, a multi-stage framework that applies supervised
fine-tuning to general-purpose vision-language models (VLMs) to emulate
radiologists' stepwise diagnostic reasoning using only free-text reports.
DiagCoT combines contrastive image-report tuning for domain alignment,
chain-of-thought supervision to capture inferential logic, and reinforcement
tuning with clinical reward signals to enhance factual accuracy and fluency. On
the MIMIC-CXR benchmark, DiagCoT improved zero-shot disease classification AUC
from 0.52 to 0.76 (absolute gain of 0.24), pathology grounding mIoU from 0.08
to 0.31 (absolute gain of 0.23), and report generation BLEU from 0.11 to 0.33
(absolute gain of 0.22). It outperformed state-of-the-art models including
LLaVA-Med and CXR-LLAVA on long-tailed diseases and external datasets. By
converting unstructured clinical narratives into structured supervision,
DiagCoT offers a scalable approach for developing interpretable and
diagnostically competent AI systems for radiology.

### 7. [Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning](http://arxiv.org/pdf/2509.06436v1)

Authors: Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian

Large language models (LLMs) face persistent challenges when handling
long-context tasks, most notably the lost in the middle issue, where
information located in the middle of a long input tends to be underutilized.
Some existing methods that reduce input have the risk of discarding key
information, while others that extend context windows often lead to attention
dispersion. To address these limitations, we propose Tree of Agents (TOA), a
multi-agent reasoning framework that segments the input into chunks processed
by independent agents. Each agent generates its local cognition, then agents
dynamically exchange information for collaborative reasoning along
tree-structured paths. TOA enables agents to probe different reasoning orders
for multi-perspective understanding, effectively mitigating position bias and
reducing hallucinations. To improve processing efficiency, we incorporate
prefix-hash caching and adaptive pruning strategies, achieving significant
performance improvements with comparable API overhead. Experiments show that
TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple
baselines and demonstrates comparable performance to the latest and much larger
commercial models, such as Gemini1.5-pro, on various long-context tasks. Code
is available at https://github.com/Aireduce952/Tree-of-Agents.

### 8. [HyFedRAG: A Federated Retrieval-Augmented Generation Framework for Heterogeneous and Privacy-Sensitive Data](http://arxiv.org/pdf/2509.06444v1)

Authors: Cheng Qian, Hainan Zhang, Yongxin Tong, Hong-Wei Zheng, Zhiming Zheng

Centralized RAG pipelines struggle with heterogeneous and privacy-sensitive
data, especially in distributed healthcare settings where patient data spans
SQL, knowledge graphs, and clinical notes. Clinicians face difficulties
retrieving rare disease cases due to privacy constraints and the limitations of
traditional cloud-based RAG systems in handling diverse formats and edge
devices. To address this, we introduce HyFedRAG, a unified and efficient
Federated RAG framework tailored for Hybrid data modalities. By leveraging an
edge-cloud collaborative mechanism, HyFedRAG enables RAG to operate across
diverse data sources while preserving data privacy. Our key contributions are:
(1) We design an edge-cloud collaborative RAG framework built on Flower, which
supports querying structured SQL data, semi-structured knowledge graphs, and
unstructured documents. The edge-side LLMs convert diverse data into
standardized privacy-preserving representations, and the server-side LLMs
integrates them for global reasoning and generation. (2) We integrate
lightweight local retrievers with privacy-aware LLMs and provide three
anonymization tools that enable each client to produce semantically rich,
de-identified summaries for global inference across devices. (3) To optimize
response latency and reduce redundant computation, we design a three-tier
caching strategy consisting of local cache, intermediate representation cache,
and cloud inference cache. Experimental results on PMC-Patients demonstrate
that HyFedRAG outperforms existing baselines in terms of retrieval quality,
generation consistency, and system efficiency. Our framework offers a scalable
and privacy-compliant solution for RAG over structural-heterogeneous data,
unlocking the potential of LLMs in sensitive and diverse data environments.

### 9. [Accelerate Scaling of LLM Alignment via Quantifying the Coverage and Depth of Instruction Set](http://arxiv.org/pdf/2509.06463v1)

Authors: Chengwei Wu, Li Du, Hanyu Zhao, Yiming Ju, Jiapu Wang, Tengfei Pan

With the growing demand for applying large language models to downstream
tasks, improving model alignment performance and efficiency has become crucial.
Such a process involves selecting informative instructions from a candidate
pool. However, due to the complexity of instruction set distributions, the key
factors driving the performance of aligned models remain unclear. As a result,
current instruction set refinement methods fail to improve performance as the
instruction pool expands continuously. To address this issue, we first
investigate the key factors that influence the relationship between instruction
dataset distribution and aligned model performance. Based on these insights, we
propose a novel instruction data selection method. We identify that the depth
of instructions and the coverage of the semantic space are the crucial factors
determining downstream performance, which could explain over 70\% of the model
loss on the development set. We then design an instruction selection algorithm
to simultaneously maximize the depth and semantic coverage of the selected
instructions. Experimental results demonstrate that, compared to
state-of-the-art baseline methods, it can sustainably improve model performance
at a faster pace and thus achieve \emph{``Accelerated Scaling''}.

### 10. [MAS-Bench: A Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents](http://arxiv.org/pdf/2509.06477v1)

Authors: Pengxiang Zhao, Guangyi Liu, Yaozhen Liang, Weiqing He, Zhengxi Lu, Yuehao Huang, Yaxuan Guo, Kexin Zhang, Hao Wang, Liang Liu, Yong Liu

To enhance the efficiency of GUI agents on various platforms like smartphones
and computers, a hybrid paradigm that combines flexible GUI operations with
efficient shortcuts (e.g., API, deep links) is emerging as a promising
direction. However, a framework for systematically benchmarking these hybrid
agents is still underexplored. To take the first step in bridging this gap, we
introduce MAS-Bench, a benchmark that pioneers the evaluation of GUI-shortcut
hybrid agents with a specific focus on the mobile domain. Beyond merely using
predefined shortcuts, MAS-Bench assesses an agent's capability to autonomously
generate shortcuts by discovering and creating reusable, low-cost workflows. It
features 139 complex tasks across 11 real-world applications, a knowledge base
of 88 predefined shortcuts (APIs, deep-links, RPA scripts), and 7 evaluation
metrics. The tasks are designed to be solvable via GUI-only operations, but can
be significantly accelerated by intelligently embedding shortcuts. Experiments
show that hybrid agents achieve significantly higher success rates and
efficiency than their GUI-only counterparts. This result also demonstrates the
effectiveness of our method for evaluating an agent's shortcut generation
capabilities. MAS-Bench fills a critical evaluation gap, providing a
foundational platform for future advancements in creating more efficient and
robust intelligent agents.

### Hardware Architecture

### 1. [Hardware Acceleration in Portable MRIs: State of the Art and Future Prospects](http://arxiv.org/pdf/2509.06365v1)

Authors: Omar Al Habsi, Safa Mohammed Sali, Anis Meribout, Mahmoud Meribout, Saif Almazrouei, Mohamed Seghier

There is a growing interest in portable MRI (pMRI) systems for point-of-care
imaging, particularly in remote or resource-constrained environments. However,
the computational complexity of pMRI, especially in image reconstruction and
machine learning (ML) algorithms for enhanced imaging, presents significant
challenges. Such challenges can be potentially addressed by harnessing hardware
application solutions, though there is little focus in the current pMRI
literature on hardware acceleration. This paper bridges that gap by reviewing
recent developments in pMRI, emphasizing the role and impact of hardware
acceleration to speed up image acquisition and reconstruction. Key technologies
such as Graphics Processing Units (GPUs), Field-Programmable Gate Arrays
(FPGAs), and Application-Specific Integrated Circuits (ASICs) offer excellent
performance in terms of reconstruction speed and power consumption. This review
also highlights the promise of AI-powered reconstruction, open low-field pMRI
datasets, and innovative edge-based hardware solutions for the future of pMRI
technology. Overall, hardware acceleration can enhance image quality, reduce
power consumption, and increase portability for next-generation pMRI
technology. To accelerate reproducible AI for portable MRI, we propose forming
a Low-Field MRI Consortium and an evidence ladder (analytic/phantom validation,
retrospective multi-center testing, prospective reader and non-inferiority
trials) to provide standardized datasets, benchmarks, and regulator-ready
testbeds.

### 2. [VCO-CARE: VCO-based Calibration-free Analog Readout for Electrodermal activity sensing](http://arxiv.org/pdf/2509.06698v1)

Authors: Leidy Mabel Alvero-Gonzalez, Matias Miguez, Eric Gutierrez, Juan Sapriza, Susana Patón, David Atienza, José Miranda

Continuous monitoring of electrodermal activity (EDA) through wearable
devices has attracted much attention in recent times. However, the persistent
challenge demands analog front-end (AFE) systems with high sensitivity, low
power consumption, and minimal calibration requirements to ensure practical
usability in wearable technologies. In response to this challenge, this
research introduces VCO-CARE, a Voltage-Controlled Oscillator-based Analog
Readout tailored for continuous EDA sensing. The results show that our system
achieves an exceptional average sensitivity of up to 40 pS within a 0-20 uS
range and a negligible relative error of less than 0.0025% for
fixed-resistance. Furthermore, the proposed system consumes only an average of
2.3 uW based on post-layout validations and introduces a low noise
contribution, measuring only 0.8 uVrms across the 0-1.5 Hz EDA signal band.
This research aims to drive the evolution of wearable sensors characterized by
seamless adaptability to diverse users, minimal power consumption, and
outstanding noise resilience.

### 3. [A Spatio-Temporal Graph Neural Networks Approach for Predicting Silent Data Corruption inducing Circuit-Level Faults](http://arxiv.org/pdf/2509.06289v1)

Authors: Shaoqi Wei, Senling Wang, Hiroshi Kai, Yoshinobu Higami, Ruijun Ma, Tianming Ni, Xiaoqing Wen, Hiroshi Takahashi

Silent Data Errors (SDEs) from time-zero defects and aging degrade
safety-critical systems. Functional testing detects SDE-related faults but is
expensive to simulate. We present a unified spatio-temporal graph convolutional
network (ST-GCN) for fast, accurate prediction of long-cycle fault impact
probabilities (FIPs) in large sequential circuits, supporting quantitative risk
assessment. Gate-level netlists are modeled as spatio-temporal graphs to
capture topology and signal timing; dedicated spatial and temporal encoders
predict multi-cycle FIPs efficiently. On ISCAS-89 benchmarks, the method
reduces simulation time by more than 10x while maintaining high accuracy (mean
absolute error 0.024 for 5-cycle predictions). The framework accepts features
from testability metrics or fault simulation, allowing efficiency-accuracy
trade-offs. A test-point selection study shows that choosing observation points
by predicted FIPs improves detection of long-cycle, hard-to-detect faults. The
approach scales to SoC-level test strategy optimization and fits downstream
electronic design automation flows.

### 4. [BioLite U-Net: Edge-Deployable Semantic Segmentation for In Situ Bioprinting Monitoring](http://arxiv.org/pdf/2509.06690v1)

Authors: Usman Haider, Lukasz Szemet, Daniel Kelly, Vasileios Sergis, Andrew C. Daly, Karl Mason

Bioprinting is a rapidly advancing field that offers a transformative
approach to fabricating tissue and organ models through the precise deposition
of cell-laden bioinks. Ensuring the fidelity and consistency of printed
structures in real-time remains a core challenge, particularly under
constraints imposed by limited imaging data and resource-constrained embedded
hardware. Semantic segmentation of the extrusion process, differentiating
between nozzle, extruded bioink, and surrounding background, enables in situ
monitoring critical to maintaining print quality and biological viability. In
this work, we introduce a lightweight semantic segmentation framework tailored
for real-time bioprinting applications. We present a novel, manually annotated
dataset comprising 787 RGB images captured during the bioprinting process,
labeled across three classes: nozzle, bioink, and background. To achieve fast
and efficient inference suitable for integration with bioprinting systems, we
propose a BioLite U-Net architecture that leverages depthwise separable
convolutions to drastically reduce computational load without compromising
accuracy. Our model is benchmarked against MobileNetV2 and MobileNetV3-based
segmentation baselines using mean Intersection over Union (mIoU), Dice score,
and pixel accuracy. All models were evaluated on a Raspberry Pi 4B to assess
real-world feasibility. The proposed BioLite U-Net achieves an mIoU of 92.85%
and a Dice score of 96.17%, while being over 1300x smaller than
MobileNetV2-DeepLabV3+. On-device inference takes 335 ms per frame,
demonstrating near real-time capability. Compared to MobileNet baselines,
BioLite U-Net offers a superior tradeoff between segmentation accuracy,
efficiency, and deployability, making it highly suitable for intelligent,
closed-loop bioprinting systems.

### 5. [Dato: A Task-Based Programming Model for Dataflow Accelerators](http://arxiv.org/pdf/2509.06794v1)

Authors: Shihan Fang, Hongzheng Chen, Niansong Zhang, Jiajie Li, Han Meng, Adrian Liu, Zhiru Zhang

Recent deep learning workloads increasingly push computational demand beyond
what current memory systems can sustain, with many kernels stalling on data
movement rather than computation. While modern dataflow accelerators
incorporate on-chip streaming to mitigate off-chip bandwidth limitations,
existing programming models struggle to harness these capabilities effectively.
Low-level interfaces provide fine-grained control but impose significant
development overhead, whereas high-level tile-based languages abstract away
communication details, restricting optimization and forcing compilers to
reconstruct the intended dataflow. We present Dato, a Python-embedded,
task-based programming model for dataflow accelerators that elevates data
communication and sharding to first-class type constructs. Developers write
programs as a graph of tasks connected via explicit stream types, with sharded
inputs specified using layout types. These tasks are first mapped virtually
onto the accelerator's spatial fabric, and the compiler then generates a
physical mapping that respects hardware constraints. Experimental results on
both AMD Ryzen AI NPU and Alveo FPGA devices demonstrate that Dato achieves
high performance while significantly reducing the burden of writing optimized
code. On the NPU, Dato attains up to 84% hardware utilization for GEMM and
delivers a 2.81x speedup on attention kernels compared to a state-of-the-art
commercial framework. On the FPGA, Dato surpasses leading frameworks in
performance when generating custom systolic arrays, achieving 98% of the
theoretical peak performance.

### Computational Complexity

### 1. [Linear Matroid Intersection is in Catalytic Logspace](http://arxiv.org/pdf/2509.06435v1)

Authors: Aryan Agarwala, Yaroslav Alekseev, Antoine Vinciguerra

Linear matroid intersection is an important problem in combinatorial
optimization. Given two linear matroids over the same ground set, the linear
matroid intersection problem asks you to find a common independent set of
maximum size. The deep interest in linear matroid intersection is due to the
fact that it generalises many classical problems in theoretical computer
science, such as bipartite matching, edge disjoint spanning trees, rainbow
spanning tree, and many more.
  We study this problem in the model of catalytic computation: space-bounded
machines are granted access to \textit{catalytic space}, which is additional
working memory that is full with arbitrary data that must be preserved at the
end of its computation.
  Although linear matroid intersection has had a polynomial time algorithm for
over 50 years, it remains an important open problem to show that linear matroid
intersection belongs to any well studied subclass of $P$. We address this
problem for the class catalytic logspace ($CL$) with a polynomial time bound
($CLP$).
  Recently, Agarwala and Mertz (2025) showed that bipartite maximum matching
can be computed in the class $CLP\subseteq P$. This was the first subclass of
$P$ shown to contain bipartite matching, and additionally the first problem
outside $TC^1$ shown to be contained in $CL$. We significantly improve the
result of Agarwala and Mertz by showing that linear matroid intersection can be
computed in $CLP$.

### 2. [The Parameter Report: An Orientation Guide for Data-Driven Parameterization](http://arxiv.org/pdf/2509.06880v1)

Authors: Christian Komusiewicz, Nils Morawietz, Frank Sommer, Luca Pascal Staus

A strength of parameterized algorithmics is that each problem can be
parameterized by an essentially inexhaustible set of parameters. Usually, the
choice of the considered parameter is informed by the theoretical relations
between parameters with the general goal of achieving FPT-algorithms for
smaller and smaller parameters. However, the FPT-algorithms for smaller
parameters usually have higher running times and it is unclear whether the
decrease in the parameter value or the increase in the running time bound
dominates in real-world data. This question cannot be answered from purely
theoretical considerations and any answer requires knowledge on typical
parameter values.
  To provide a data-driven guideline for parameterized complexity studies of
graph problems, we present the first comprehensive comparison of parameter
values for a set of benchmark graphs originating from real-world applications.
Our study covers degree-related parameters, such as maximum degree or
degeneracy, neighborhood-based parameters such as neighborhood diversity and
modular-width, modulator-based parameters such as vertex cover number and
feedback vertex set number, and the treewidth of the graphs.
  Our results may help assess the significance of FPT-running time bounds on
the solvability of real-world instances. For example, the vertex cover number
$vc$ of $n$-vertex graphs is often only slightly below $n/2$. Thus, a running
time bound of $O(2^{vc})$ is only slightly better than a running time bound of
$O(1.4^{n})$. In contrast, the treewidth $tw$ is almost always below $n/3$ and
often close to $n/9$, making a running time of $O(2^{tw})$ much more practical
on real-world instances.
  We make our implementation and full experimental data openly available. In
particular, this provides the first implementations for several graph
parameters such as 4-path vertex cover number and vertex integrity.

### 3. [On the Bit Size of Sum-of-Squares Proofs for Symmetric Formulations](http://arxiv.org/pdf/2509.06928v1)

Authors: Alex Bortolotti, Monaldo Mastrolilli, Marilena Palomba, Luis Felipe Vargas

The Sum-of-Squares (SoS) hierarchy is a powerful framework for polynomial
optimization and proof complexity, offering tight semidefinite relaxations that
capture many classical algorithms. Despite its broad applicability, several
works have revealed fundamental limitations to SoS automatability. (i) While
low-degree SoS proofs are often desirable for tractability, recent works have
revealed they may require coefficients of prohibitively large bit size,
rendering them computationally infeasible. (ii) Prior works have shown that SoS
proofs for seemingly easy problems require high-degree. In particular, this
phenomenon also arises in highly symmetric problems. Instances of symmetric
problems-particularly those with a small number of constraints-have repeatedly
served as benchmarks for establishing high-degree lower bounds in the SoS
hierarchy. It has remained unclear whether symmetry can also lead to large bit
sizes in SoS proofs, potentially making low-degree proofs computationally
infeasible even in symmetric settings.
  In this work, we resolve this question by proving that symmetry alone does
not lead to large bit size SoS proofs. Focusing on symmetric Archimedean
instances, we show that low-degree SoS proofs for such systems admit compact,
low bit size representations. Together, these results provide a conceptual
separation between two sources of SoS hardness-degree and bit size-by showing
they do not necessarily align, even in highly symmetric instances. This insight
guides future work on automatability and lower bounds: symmetry may necessitate
high-degree proofs, but it does not by itself force large coefficients.

### 4. [Slice rank and partition rank of the determinant](http://arxiv.org/pdf/2509.06294v1)

Authors: Amichai Lampert, Guy Moshkovitz

The Laplace expansion expresses the $n \times n$ determinant $\det_n$ as a
sum of $n$ products. Do shorter expansions exist? In this paper we:
  - Fully determine the slice rank decompositions of $\det_n$ (where each
product must contain a linear factor): In this case, we show that $n$ summands
are necessary, and moreover, the only such expansions with $n$ summands are
equivalent (in a precise sense) to the Laplace expansion.
  - Prove a logarithmic lower bound for the partition rank of $\det_n$ (where
each product is of multilinear forms): In this case, we show that at least
$\log_2(n)+1$ summands are needed. We also explain why existing techniques fail
to yield any nontrivial lower bound, and why our new method cannot give a
super-logarithmic lower bound.
  - Separate partition rank from slice rank for $\det_n$: we find a quadratic
expansion for $\det_4$, over any field, with fewer summands than the Laplace
expansion. This construction is related to a well-known example of Green-Tao
and Lovett-Meshulam-Samorodnitsky disproving the naive version of the Gowers
Inverse conjecture over small fields.
  An important motivation for these questions comes from the challenge of
separating structure and randomness for tensors. On the one hand, we show that
the random construction fails to separate: for a random tensor of partition
rank $r$, the analytic rank is $r-o(1)$ with high probability. On the other
hand, our results imply that the determinant yields the first asymptotic
separation between partition rank and analytic rank of $d$-tensors, with their
ratio tending to infinity with $d$.

### 5. [Information-Theoretic Bounds and Task-Centric Learning Complexity for Real-World Dynamic Nonlinear Systems](http://arxiv.org/pdf/2509.06599v1)

Authors: Sri Satish Krishna Chaitanya Bulusu, Mikko Sillanpää

Dynamic nonlinear systems exhibit distortions arising from coupled static and
dynamic effects. Their intertwined nature poses major challenges for
data-driven modeling. This paper presents a theoretical framework grounded in
structured decomposition, variance analysis, and task-centric complexity
bounds.
  The framework employs a directional lower bound on interactions between
measurable system components, extending orthogonality in inner product spaces
to structurally asymmetric settings. This bound supports variance inequalities
for decomposed systems. Key behavioral indicators are introduced along with a
memory finiteness index. A rigorous power-based condition establishes a
measurable link between finite memory in realizable systems and the First Law
of Thermodynamics. This offers a more foundational perspective than classical
bounds based on the Second Law.
  Building on this foundation, we formulate a `Behavioral Uncertainty
Principle,' demonstrating that static and dynamic distortions cannot be
minimized simultaneously. We identify that real-world systems seem to resist
complete deterministic decomposition due to entangled static and dynamic
effects. We also present two general-purpose theorems linking function variance
to mean-squared Lipschitz continuity and learning complexity. This yields a
model-agnostic, task-aware complexity metric, showing that lower-variance
components are inherently easier to learn.
  These insights explain the empirical benefits of structured residual
learning, including improved generalization, reduced parameter count, and lower
training cost, as previously observed in power amplifier linearization
experiments. The framework is broadly applicable and offers a scalable,
theoretically grounded approach to modeling complex dynamic nonlinear systems.

### Computational Engineering

### 1. [Anticipating AMOC transitions via deep learning](http://arxiv.org/pdf/2509.06450v1)

Authors: Wenjie Zhang, Yu Huang, Sebastian Bathiany, Yechul Shin, Maya Ben-Yami, Suiping Zhou, Niklas Boers

Key components of the Earth system can undergo abrupt and potentially
irreversible transitions when the magnitude or rate of external forcing exceeds
critical thresholds. In this study, we use the example of the Atlantic
Meridional Overturning Circulation (AMOC) to demonstrate the challenges
associated with anticipating such transitions when the system is susceptible to
bifurcation-induced, rate-induced, and noise-induced tipping. Using a
calibrated AMOC box model, we conduct large ensemble simulations and show that
transition behavior is inherently probabilistic: under identical freshwater
forcing scenarios, some ensemble members exhibit transitions while others do
not. In this stochastic regime, traditional early warning indicators based on
critical slowing down are unreliable in predicting impending transitions. To
address this limitation, we develop a convolutional neural network (CNN)-based
approach that identifies higher-order statistical differences between
transitioning and non-transitioning trajectories within the ensemble
realizations. This method enables the real-time prediction of transition
probabilities for individual trajectories prior to the onset of tipping. Our
results show that the CNN-based indicator provides effective early warnings in
a system where transitions can be induced by bifurcations, critical forcing
rates, and noise. These findings underscore the potential in identifying safe
operating spaces and early warning indicators for abrupt transitions of Earth
system components under uncertainty.

### 2. [Reusable Surrogate Models for Distillation Columns](http://arxiv.org/pdf/2509.06638v1)

Authors: Martin Bubel, Tobias Seidel, Michael Bortz

Surrogate modeling is a powerful methodology in chemical process engineering,
frequently employed to accelerate optimization tasks where traditional
flowsheet simulators are computationally prohibitive. However, the
state-of-the-art is dominated by surrogate models trained for a narrow range of
fixed chemical systems and operating conditions, limiting their reusability.
This work introduces a paradigm shift towards reusable surrogates by developing
a single model for distillation columns that generalizes across a vast design
space. The key enabler is a novel ML-fueled modelfluid representation which
allows for the generation of datasets of more than $1,000,000$ samples. This
allows the surrogate to generalize not only over column specifications but also
over the entire chemical space of homogeneous ternary vapor-liquid mixtures. We
validate the model's accuracy and demonstrate its practical utility in a case
study on entrainer distillation, where it successfully screens and ranks
candidate entrainers, significantly reducing the computational effort compared
to rigorous optimization.

### 3. [A machine-learned expression for the excess Gibbs energy](http://arxiv.org/pdf/2509.06484v1)

Authors: Marco Hoffmann, Thomas Specht, Quirin Göttl, Jakob Burger, Stephan Mandt, Hans Hasse, Fabian Jirasek

The excess Gibbs energy plays a central role in chemical engineering and
chemistry, providing a basis for modeling the thermodynamic properties of
liquid mixtures. Predicting the excess Gibbs energy of multi-component mixtures
solely from the molecular structures of their components is a long-standing
challenge. In this work, we address this challenge by integrating physical laws
as hard constraints within a flexible neural network. The resulting model,
HANNA, was trained end-to-end on an extensive experimental dataset for binary
mixtures from the Dortmund Data Bank, guaranteeing thermodynamically consistent
predictions. A novel surrogate solver developed in this work enabled the
inclusion of liquid-liquid equilibrium data in the training process.
Furthermore, a geometric projection method was applied to enable robust
extrapolations to multi-component mixtures, without requiring additional
parameters. We demonstrate that HANNA delivers excellent predictions, clearly
outperforming state-of-the-art benchmark methods in accuracy and scope. The
trained model and corresponding code are openly available, and an interactive
interface is provided on our website, MLPROP.

### 4. [CAME-AB: Cross-Modality Attention with Mixture-of-Experts for Antibody Binding Site Prediction](http://arxiv.org/pdf/2509.06465v1)

Authors: Hongzong Li, Jiahao Ma, Zhanpeng Shi, Fanming Jin, Ye-Fan Hu, Jian-Dong Huang

Antibody binding site prediction plays a pivotal role in computational
immunology and therapeutic antibody design. Existing sequence or structure
methods rely on single-view features and fail to identify antibody-specific
binding sites on the antigens-a dual limitation in representation and
prediction. In this paper, we propose CAME-AB, a novel Cross-modality Attention
framework with a Mixture-of-Experts (MoE) backbone for robust antibody binding
site prediction. CAME-AB integrates five biologically grounded modalities,
including raw amino acid encodings, BLOSUM substitution profiles, pretrained
language model embeddings, structure-aware features, and GCN-refined
biochemical graphs-into a unified multimodal representation. To enhance
adaptive cross-modal reasoning, we propose an adaptive modality fusion module
that learns to dynamically weight each modality based on its global relevance
and input-specific contribution. A Transformer encoder combined with an MoE
module further promotes feature specialization and capacity expansion. We
additionally incorporate a supervised contrastive learning objective to
explicitly shape the latent space geometry, encouraging intra-class compactness
and inter-class separability. To improve optimization stability and
generalization, we apply stochastic weight averaging during training. Extensive
experiments on benchmark antibody-antigen datasets demonstrate that CAME-AB
consistently outperforms strong baselines on multiple metrics, including
Precision, Recall, F1-score, AUC-ROC, and MCC. Ablation studies further
validate the effectiveness of each architectural component and the benefit of
multimodal feature integration. The model implementation details and the codes
are available on https://anonymous.4open.science/r/CAME-AB-C525

### 5. [A Parallel Solver with Multiphysics Finite Element Method for Poroelasticity Coupled with Elasticity Model](http://arxiv.org/pdf/2509.06673v1)

Authors: Zhihao Ge, Chengxin Wang

In this paper, we propose a parallel solver for solving the quasi-static
linear poroelasticity coupled with linear elasticity model in the Lagrange
multiplier framework. Firstly, we reformulate the model into a coupling of the
nearly incompressible elasticity and an unsteady affection-diffusion equations
by setting new variable ``elastic pressure" and ``volumetric fluid content".
And we introduce a Lagrange multiplier to guarantee the normal stress
continuity on the interface. Then, we give the variational formulations in each
subdomain and choose the $\boldsymbol{P}_k$-$P_1$-$P_1$ mixed finite element
tuple for poroelasticity subdomain, and $\boldsymbol{P}_k$-$P_1$ finite element
pair ($k=1,2$) for elasticity subdomain and the backward Euler scheme for time.
Also, we propose a parallel solver for solving the fully discrete scheme at
each time step -- the FETI method with a classical FETI preconditioner for
solving the Lagrange multiplier and calculating the subproblems in each
subdomain in parallel. And we show several numerical tests to validate the
computational efficiency and the convergence error order, and we consider
Barry-Mercer's model as the benchmark test to show that there no oscillation in
the computed pressure. Finally, we draw conclusions to summarize the main
results of this paper.

### Computational Geometry

### 1. [No Infinite $(p,q)$-Theorem for Piercing Compact Convex Sets with Lines in $\mathbb{R}^3$](http://arxiv.org/pdf/2509.06731v1)

Authors: Sutanoya Chakraborty, Arijit Ghosh

An infinite $(p,q)$-theorem, or an $(\aleph_0,q)$-theorem, involving two
families $\mathcal{F}$ and $\mathcal{G}$ of sets, states that if in every
infinite subset of $\mathcal{F}$, there are $q$ sets that are intersected by
some set in $\mathcal{G}$, then there is a finite set
$S_{\mathcal{F}}\subseteq\mathcal{G}$ such that for every $C\in\mathcal{F}$,
there is a $B\in S_{\mathcal{F}}$ with $C\cap B\neq\emptyset$. We provide an
example demonstrating that there is no $(\aleph_0,q)$-theorem for piercing
compact convex sets in $\mathbb{R}^3$ with lines by constructing a family
$\mathcal{F}$ of compact convex sets such that it does not have a finite line
transversal, but for any $t\in\mathbb{N}$, every infinite subset of
$\mathcal{F}$ contains $t$ sets that are pierced by a line.

### Computation and Language

### 1. [No Encore: Unlearning as Opt-Out in Music Generation](http://arxiv.org/pdf/2509.06277v1)

Authors: Jinju Kim, Taehan Kim, Abdul Waheed, Rita Singh

AI music generation is rapidly emerging in the creative industries, enabling
intuitive music generation from textual descriptions. However, these systems
pose risks in exploitation of copyrighted creations, raising ethical and legal
concerns. In this paper, we present preliminary results on the first
application of machine unlearning techniques from an ongoing research to
prevent inadvertent usage of creative content. Particularly, we explore
existing methods in machine unlearning to a pre-trained Text-to-Music (TTM)
baseline and analyze their efficacy in unlearning pre-trained datasets without
harming model performance. Through our experiments, we provide insights into
the challenges of applying unlearning in music generation, offering a
foundational analysis for future works on the application of unlearning for
music generative models.

### 2. [Do LLMs exhibit the same commonsense capabilities across languages?](http://arxiv.org/pdf/2509.06401v1)

Authors: Ivan Martínez-Murillo, Elena Lloret, Paloma Moreda, Albert Gatt

This paper explores the multilingual commonsense generation abilities of
Large Language Models (LLMs). To facilitate this investigation, we introduce
MULTICOM, a novel benchmark that extends the COCOTEROS dataset to four
languages: English, Spanish, Dutch, and Valencian. The task involves generating
a commonsensical sentence that includes a given triplet of words. We evaluate a
range of open-source LLMs, including LLaMA, Qwen, Gemma, EuroLLM, and
Salamandra, on this benchmark. Our evaluation combines automatic metrics,
LLM-as-a-judge approaches (using Prometheus and JudgeLM), and human
annotations. Results consistently show superior performance in English, with
significantly lower performance in less-resourced languages. While contextual
support yields mixed results, it tends to benefit underrepresented languages.
These findings underscore the current limitations of LLMs in multilingual
commonsense generation. The dataset is publicly available at
https://huggingface.co/datasets/gplsi/MULTICOM.

### 3. [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](http://arxiv.org/pdf/2509.06501v1)

Authors: Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, Jiayuan Song, Zhengmao Zhu, Wenhu Chen, Pengyu Zhao, Junxian He

The paradigm of Large Language Models (LLMs) has increasingly shifted toward
agentic applications, where web browsing capabilities are fundamental for
retrieving information from diverse online sources. However, existing
open-source web agents either demonstrate limited information-seeking abilities
on complex tasks or lack transparent implementations. In this work, we identify
that the key challenge lies in the scarcity of challenging data for information
seeking. To address this limitation, we introduce WebExplorer: a systematic
data generation approach using model-based exploration and iterative,
long-to-short query evolution. This method creates challenging query-answer
pairs that require multi-step reasoning and complex web navigation. By
leveraging our curated high-quality dataset, we successfully develop advanced
web agent WebExplorer-8B through supervised fine-tuning followed by
reinforcement learning. Our model supports 128K context length and up to 100
tool calling turns, enabling long-horizon problem solving. Across diverse
information-seeking benchmarks, WebExplorer-8B achieves the state-of-the-art
performance at its scale. Notably, as an 8B-sized model, WebExplorer-8B is able
to effectively search over an average of 16 turns after RL training, achieving
higher accuracy than WebSailor-72B on BrowseComp-en/zh and attaining the best
performance among models up to 100B parameters on WebWalkerQA and FRAMES.
Beyond these information-seeking tasks, our model also achieves strong
generalization on the HLE benchmark even though it is only trained on
knowledge-intensive QA data. These results highlight our approach as a
practical path toward long-horizon web agents.

### 4. [LAMDAS: LLM as an Implicit Classifier for Domain-specific Data Selection](http://arxiv.org/pdf/2509.06524v1)

Authors: Jian Wu, Hang Yu, Bingchang Liu, Wenjie Yang, Peng Di, Jianguo Li, Yue Zhang

Adapting large language models (LLMs) to specific domains often faces a
critical bottleneck: the scarcity of high-quality, human-curated data. While
large volumes of unchecked data are readily available, indiscriminately using
them for fine-tuning risks introducing noise and degrading performance.
Strategic data selection is thus crucial, requiring a method that is both
accurate and efficient. Existing approaches, categorized as similarity-based
and direct optimization methods, struggle to simultaneously achieve these
goals. In this paper, we introduce LAMDAS (LLM As an iMplicit classifier for
domain-specific DAta Selection), a novel approach that leverages the
pre-trained LLM itself as an implicit classifier, thereby bypassing explicit
feature engineering and computationally intensive optimization process. LAMDAS
reframes data selection as a one-class classification problem, identifying
candidate data that "belongs" to the target domain defined by a small reference
dataset. Extensive experimental results demonstrate that LAMDAS not only
exceeds the performance of full-data training using a fraction of the data but
also outperforms nine state-of-the-art (SOTA) baselines under various
scenarios. Furthermore, LAMDAS achieves the most compelling balance between
performance gains and computational efficiency compared to all evaluated
baselines.

### 5. [Guided Decoding and Its Critical Role in Retrieval-Augmented Generation](http://arxiv.org/pdf/2509.06631v1)

Authors: Özgür Uğur, Musa Yılmaz, Esra Şavirdi, Özay Ezerceli, Mahmut El Huseyni, Selva Taş, Reyhan Bayraktar

The integration of Large Language Models (LLMs) into various applications has
driven the need for structured and reliable responses. A key challenge in
Retrieval-Augmented Generation (RAG) systems is ensuring that outputs align
with expected formats while minimizing hallucinations. This study examines the
role of guided decoding in RAG systems, comparing three methods, Outlines,
XGrammar, and LM Format Enforcer, across different multi-turn prompting setups
(0-turn, 1-turn, and 2-turn). By evaluating success rates, hallucination rates,
and output quality, we provide insights into their performance and
applicability. Our findings reveal how multi-turn interactions influence guided
decoding, uncovering unexpected performance variations that can inform method
selection for specific use cases. This work advances the understanding of
structured output generation in RAG systems, offering both theoretical insights
and practical guidance for LLM deployment.

### 6. [Modelling Intertextuality with N-gram Embeddings](http://arxiv.org/pdf/2509.06637v1)

Authors: Yi Xing

Intertextuality is a central tenet in literary studies. It refers to the
intricate links between literary texts that are created by various types of
references. This paper proposes a new quantitative model of intertextuality to
enable scalable analysis and network-based insights: perform pairwise
comparisons of the embeddings of n-grams from two texts and average their
results as the overall intertextuality. Validation on four texts with known
degrees of intertextuality, alongside a scalability test on 267 diverse texts,
demonstrates the method's effectiveness and efficiency. Network analysis
further reveals centrality and community structures, affirming the approach's
success in capturing and quantifying intertextual relationships.

### 7. [IntrEx: A Dataset for Modeling Engagement in Educational Conversations](http://arxiv.org/pdf/2509.06652v1)

Authors: Xingwei Tan, Mahathi Parvatham, Chiara Gambi, Gabriele Pergola

Engagement and motivation are crucial for second-language acquisition, yet
maintaining learner interest in educational conversations remains a challenge.
While prior research has explored what makes educational texts interesting,
still little is known about the linguistic features that drive engagement in
conversations. To address this gap, we introduce IntrEx, the first large
dataset annotated for interestingness and expected interestingness in
teacher-student interactions. Built upon the Teacher-Student Chatroom Corpus
(TSCC), IntrEx extends prior work by incorporating sequence-level annotations,
allowing for the study of engagement beyond isolated turns to capture how
interest evolves over extended dialogues. We employ a rigorous annotation
process with over 100 second-language learners, using a comparison-based rating
approach inspired by reinforcement learning from human feedback (RLHF) to
improve agreement. We investigate whether large language models (LLMs) can
predict human interestingness judgments. We find that LLMs (7B/8B parameters)
fine-tuned on interestingness ratings outperform larger proprietary models like
GPT-4o, demonstrating the potential for specialised datasets to model
engagement in educational settings. Finally, we analyze how linguistic and
cognitive factors, such as concreteness, comprehensibility (readability), and
uptake, influence engagement in educational dialogues.

### 8. [ParCzech4Speech: A New Speech Corpus Derived from Czech Parliamentary Data](http://arxiv.org/pdf/2509.06675v1)

Authors: Vladislav Stankov, Matyáš Kopp, Ondřej Bojar

We introduce ParCzech4Speech 1.0, a processed version of the ParCzech 4.0
corpus, targeted at speech modeling tasks with the largest variant containing
2,695 hours. We combined the sound recordings of the Czech parliamentary
speeches with the official transcripts. The recordings were processed with
WhisperX and Wav2Vec 2.0 to extract automated audio-text alignment. Our
processing pipeline improves upon the ParCzech 3.0 speech recognition version
by extracting more data with higher alignment reliability. The dataset is
offered in three flexible variants: (1) sentence-segmented for automatic speech
recognition and speech synthesis tasks with clean boundaries, (2) unsegmented
preserving original utterance flow across sentences, and (3) a raw-alignment
for further custom refinement for other possible tasks. All variants maintain
the original metadata and are released under a permissive CC-BY license. The
dataset is available in the LINDAT repository, with the sentence-segmented and
unsegmented variants additionally available on Hugging Face.

### 9. [Will Annotators Disagree? Identifying Subjectivity in Value-Laden Arguments](http://arxiv.org/pdf/2509.06704v1)

Authors: Amir Homayounirad, Enrico Liscio, Tong Wang, Catholijn M. Jonker, Luciano C. Siebert

Aggregating multiple annotations into a single ground truth label may hide
valuable insights into annotator disagreement, particularly in tasks where
subjectivity plays a crucial role. In this work, we explore methods for
identifying subjectivity in recognizing the human values that motivate
arguments. We evaluate two main approaches: inferring subjectivity through
value prediction vs. directly identifying subjectivity. Our experiments show
that direct subjectivity identification significantly improves the model
performance of flagging subjective arguments. Furthermore, combining
contrastive loss with binary cross-entropy loss does not improve performance
but reduces the dependency on per-label subjectivity. Our proposed methods can
help identify arguments that individuals may interpret differently, fostering a
more nuanced annotation process.

### 10. [Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint](http://arxiv.org/pdf/2509.06795v1)

Authors: Yanrui Du, Fenglei Fan, Sendong Zhao, Jiawei Cao, Qika Lin, Kai He, Ting Liu, Bing Qin, Mengling Feng

Instruction Fine-Tuning (IFT) has been widely adopted as an effective
post-training strategy to enhance various abilities of Large Language Models
(LLMs). However, prior studies have shown that IFT can significantly compromise
LLMs' safety, particularly their ability to refuse malicious instructions,
raising significant concerns. Recent research into the internal mechanisms of
LLMs has identified the refusal direction (r-direction) in the hidden states,
which plays a pivotal role in governing refusal behavior. Building on this
insight, our study reveals that the r-direction tends to drift during training,
which we identify as one of the causes of the associated safety risks. To
mitigate such drift, our proposed ProCon method introduces a
projection-constrained loss term that regularizes the projection magnitude of
each training sample's hidden state onto the r-direction. Our initial analysis
shows that applying an appropriate constraint can effectively mitigate the
refusal direction drift and associated safety risks, but remains limited by
overall performance barriers. To overcome this barrier, informed by our
observation of early-stage sharp drift and a data-driven perspective, we
introduce a warm-up strategy that emphasizes early-stage strong constraints and
broaden the data distribution to strengthen constraint signals, leading to an
enhanced ProCon method. Experimental results under various datasets, scenarios,
and LLMs demonstrate that our method can significantly mitigate safety risks
posed by IFT while preserving task performance gains. Even compared with strong
baselines, our method consistently delivers superior overall performance.
Crucially, our analysis indicates that ProCon can contribute to stabilizing the
r-direction during training, while such an interpretability-driven exploration
of LLMs' internal mechanisms lays a solid foundation for future safety
research.

### Cryptography and Security

### 1. [When Code Crosses Borders: A Security-Centric Evaluation of LLM-based Code Translation](http://arxiv.org/pdf/2509.06504v1)

Authors: Hailong Chang, Guozhu Meng, Shuhui Xiao, Kai Chen, Kun Sun, Yilin Li

With the growing demand for cross-language codebase migration, evaluating
LLMs' security implications in translation tasks has become critical. Existing
evaluations primarily focus on syntactic or functional correctness at the
function level, neglecting the critical dimension of security.
  To enable security evaluation, we construct STED (Security-centric
Translation Evaluation Dataset), the first dataset specifically designed for
evaluating the security implications of LLM-based code translation. It
comprises 720 security-related code samples across five programming languages
and nine high-impact CWE categories, sourced from CVE/NVD and manually verified
for translation tasks. Our evaluation framework consists of two independent
assessment modules: (1) rigorous evaluation by security researchers, and (2)
automated analysis via LLM-as-a-judge. Together they evaluate three critical
aspects: functional correctness, vulnerability preservation, and vulnerability
introduction rates.
  Our large-scale evaluation of five state-of-the-art LLMs across 6,000
translation instances reveals significant security degradation, with 28.6-45%
of translations introducing new vulnerabilities--particularly for web-related
flaws like input validation, where LLMs show consistent weaknesses.
Furthermore, we develop a Retrieval-Augmented Generation (RAG)-based mitigation
strategy that reduces translation-induced vulnerabilities by 32.8%, showing the
potential of knowledge-enhanced prompting.

### 2. [Synthesis of Sound and Precise Leakage Contracts for Open-Source RISC-V Processors](http://arxiv.org/pdf/2509.06509v1)

Authors: Zilong Wang, Gideon Mohr, Klaus von Gleissenthall, Jan Reineke, Marco Guarnieri

Leakage contracts have been proposed as a new security abstraction at the
instruction set architecture level. Leakage contracts aim to capture the
information that processors may leak via microarchitectural side channels.
Recently, the first tools have emerged to verify whether a processor satisfies
a given contract. However, coming up with a contract that is both sound and
precise for a given processor is challenging, time-consuming, and error-prone,
as it requires in-depth knowledge of the timing side channels introduced by
microarchitectural optimizations.
  In this paper, we address this challenge by proposing LeaSyn, the first tool
for automatically synthesizing leakage contracts that are both sound and
precise for processor designs at register-transfer level. Starting from a
user-provided contract template that captures the space of possible contracts,
LeaSyn automatically constructs a contract, alternating between contract
synthesis, which ensures precision based on an empirical characterization of
the processor's leaks, and contract verification, which ensures soundness.
  Using LeaSyn, we automatically synthesize contracts for six open-source
RISC-V CPUs for a variety of contract templates. Our experiments indicate that
LeaSyn's contracts are sound and more precise (i.e., represent the actual leaks
in the target processor more faithfully) than contracts constructed by existing
approaches.

### 3. [Marginal sets in semigroups and semirings](http://arxiv.org/pdf/2509.06562v1)

Authors: I. Buchinskiy, M. Kotov, A. Ponmaheshkumar, R. Perumal

In 2019, V. A. Roman'kov introduced the concept of marginal sets for groups.
He developed a theory of marginal sets and demonstrated how these sets can be
applied to improve some key exchange schemes. In this paper, we extend his
ideas and introduce the concept of marginal sets for semigroups and semirings.
For tropical matrix semigroups and semirings, we describe how some marginal
sets can be constructed. We apply marginal sets to improve some key exchange
schemes over semigroups.

### 4. [A Simple Data Exfiltration Game](http://arxiv.org/pdf/2509.06571v1)

Authors: Tristan Caulfield

Data exfiltration is a growing problem for business who face costs related to
the loss of confidential data as well as potential extortion. This work
presents a simple game theoretic model of network data exfiltration. In the
model, the attacker chooses the exfiltration route and speed, and the defender
selects monitoring thresholds to detect unusual activity. The attacker is
rewarded for exfiltrating data, and the defender tries to minimize the costs of
data loss and of responding to alerts.

### 5. [Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem](http://arxiv.org/pdf/2509.06572v1)

Authors: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

Large language models (LLMs) are increasingly integrated with external
systems through the Model Context Protocol (MCP), which standardizes tool
invocation and has rapidly become a backbone for LLM-powered applications.
While this paradigm enhances functionality, it also introduces a fundamental
security shift: LLMs transition from passive information processors to
autonomous orchestrators of task-oriented toolchains, expanding the attack
surface, elevating adversarial goals from manipulating single outputs to
hijacking entire execution flows. In this paper, we reveal a new class of
attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy
Disclosure (MCP-UPD). These attacks require no direct victim interaction;
instead, adversaries embed malicious instructions into external data sources
that LLMs access during legitimate tasks. The malicious logic infiltrates the
toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection,
and Privacy Disclosure, culminating in stealthy exfiltration of private data.
Our root cause analysis reveals that MCP lacks both context-tool isolation and
least-privilege enforcement, enabling adversarial instructions to propagate
unchecked into sensitive tool invocations. To assess the severity, we design
MCP-SEC and conduct the first large-scale security census of the MCP ecosystem,
analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP
ecosystem is rife with exploitable gadgets and diverse attack methods,
underscoring systemic risks in MCP platforms and the urgent need for defense
mechanisms in LLM-integrated environments.

### 6. [LLMs in Cybersecurity: Friend or Foe in the Human Decision Loop?](http://arxiv.org/pdf/2509.06595v1)

Authors: Irdin Pekaric, Philipp Zech, Tom Mattson

Large Language Models (LLMs) are transforming human decision-making by acting
as cognitive collaborators. Yet, this promise comes with a paradox: while LLMs
can improve accuracy, they may also erode independent reasoning, promote
over-reliance and homogenize decisions. In this paper, we investigate how LLMs
shape human judgment in security-critical contexts. Through two exploratory
focus groups (unaided and LLM-supported), we assess decision accuracy,
behavioral resilience and reliance dynamics. Our findings reveal that while
LLMs enhance accuracy and consistency in routine decisions, they can
inadvertently reduce cognitive diversity and improve automation bias, which is
especially the case among users with lower resilience. In contrast,
high-resilience individuals leverage LLMs more effectively, suggesting that
cognitive traits mediate AI benefit.

### 7. [A Secure Sequencer and Data Availability Committee for Rollups (Extended Version)](http://arxiv.org/pdf/2509.06614v1)

Authors: Margarita Capretto, Martín Ceresa, Antonio Fernández Anta, Pedro Moreno Sánchez, César Sánchez

Blockchains face a scalability limitation, partly due to the throughput
limitations of consensus protocols, especially when aiming to obtain a high
degree of decentralization. Layer 2 Rollups (L2s) are a faster alternative to
conventional blockchains. L2s perform most computations offchain using
minimally blockchains (L1) under-the-hood to guarantee correctness. A sequencer
is a service that receives offchain L2 transaction requests, batches these
transactions, and commits compressed or hashed batches to L1. Using hashing
needs less L1 space, which is beneficial for gas cost, but requires a data
availability committee (DAC) service to translate hashes into their
corresponding batches of transaction requests. The behavior of sequencers and
DACs influence the evolution of the L2 blockchain, presenting a potential
security threat and delaying L2 adoption. We propose in this paper fraud-proof
mechanisms, arbitrated by L1 contracts, to detect and generate evidence of
dishonest behavior of the sequencer and DAC. We study how these fraud-proofs
limit the power of adversaries that control different number of sequencer and
DACs members, and provide incentives for their honest behavior. We designed
these fraud-proof mechanisms as two player games. Unlike the generic
fraud-proofs in current L2s (designed to guarantee the correct execution of
transactions), our fraud-proofs are over pred-etermined algorithms that verify
the properties that determine the correctness of the DAC. Arbitrating over
concrete algorithms makes our fraud-proofs more efficient, easier to
understand, and simpler to prove correct. We provide as an artifact a
mechanization in LEAN4 of our fraud-proof games, including (1) the verified
strategies that honest players should play to win all games as well as (2)
mechanisms to detect dishonest claims.

### 8. [Image Encryption Scheme Based on Hyper-Chaotic Map and Self-Adaptive Diffusion](http://arxiv.org/pdf/2509.06754v1)

Authors: Yiqi Tang

In the digital age, image encryption technology acts as a safeguard,
preventing unauthorized access to images. This paper proposes an innovative
image encryption scheme that integrates a novel 2D hyper-chaotic map with a
newly developed self-adaptive diffusion method. The 2D hyper-chaotic map,
namely the 2D-RA map, is designed by hybridizing the Rastrigin and Ackley
functions. The chaotic performance of the 2D-RA map is validated through a
series of measurements, including the Bifurcation Diagram, Lyapunov Exponent
(LE), Initial Value Sensitivity, 0 - 1 Test, Correlation Dimension (CD), and
Kolmogorov Entropy (KE). The results demonstrate that the chaotic performance
of the 2D-RA map surpasses that of existing advanced chaotic functions.
Additionally, the self-adaptive diffusion method is employed to enhance the
uniformity of grayscale distribution. The performance of the image encryption
scheme is evaluated using a series of indicators. The results show that the
proposed image encryption scheme significantly outperforms current
state-of-the-art image encryption techniques.

### 9. [PLRV-O: Advancing Differentially Private Deep Learning via Privacy Loss Random Variable Optimization](http://arxiv.org/pdf/2509.06264v1)

Authors: Qin Yang, Nicholas Stout, Meisam Mohammady, Han Wang, Ayesha Samreen, Christopher J Quinn, Yan Yan, Ashish Kundu, Yuan Hong

Differentially Private Stochastic Gradient Descent (DP-SGD) is a standard
method for enforcing privacy in deep learning, typically using the Gaussian
mechanism to perturb gradient updates. However, conventional mechanisms such as
Gaussian and Laplacian noise are parameterized only by variance or scale. This
single degree of freedom ties the magnitude of noise directly to both privacy
loss and utility degradation, preventing independent control of these two
factors. The problem becomes more pronounced when the number of composition
rounds T and batch size B vary across tasks, as these variations induce
task-dependent shifts in the privacy-utility trade-off, where small changes in
noise parameters can disproportionately affect model accuracy. To address this
limitation, we introduce PLRV-O, a framework that defines a broad search space
of parameterized DP-SGD noise distributions, where privacy loss moments are
tightly characterized yet can be optimized more independently with respect to
utility loss. This formulation enables systematic adaptation of noise to
task-specific requirements, including (i) model size, (ii) training duration,
(iii) batch sampling strategies, and (iv) clipping thresholds under both
training and fine-tuning settings. Empirical results demonstrate that PLRV-O
substantially improves utility under strict privacy constraints. On CIFAR-10, a
fine-tuned ViT achieves 94.03% accuracy at epsilon approximately 0.5, compared
to 83.93% with Gaussian noise. On SST-2, RoBERTa-large reaches 92.20% accuracy
at epsilon approximately 0.2, versus 50.25% with Gaussian.

### 10. [Schrodinger's Toolbox: Exploring the Quantum Rowhammer Attack](http://arxiv.org/pdf/2509.06318v1)

Authors: Devon Campbell

Residual cross-talk in superconducting qubit devices creates a security
vulnerability for emerging quantum cloud services. We demonstrate a
Clifford-only Quantum Rowhammer attack-using just X and CNOT gates-that injects
faults on IBM's 127-qubit Eagle processors without requiring pulse-level
access. Experiments show that targeted hammering induces localized errors
confined to the attack cycle and primarily manifests as phase noise, as
confirmed by near 50% flip rates under Hadamard-basis probing. A full lattice
sweep maps QR's spatial and temporal behavior, revealing reproducible
corruption limited to qubits within two coupling hops and rapid recovery in
subsequent benign cycles. Finally, we leverage these properties to outline a
prime-and-probe covert channel, demonstrating that the clear separability
between hammered and benign rounds enables highly reliable signaling without
error correction. These findings underscore the need for hardware-level
isolation and scheduler-aware defenses as multi-tenant quantum computing
becomes standard.

### Computer Vision and Pattern Recognition

### 1. [Spatial Reasoning with Vision-Language Models in Ego-Centric Multi-View Scenes](http://arxiv.org/pdf/2509.06266v1)

Authors: Mohsen Gholami, Ahmad Rezaei, Zhou Weimin, Yong Zhang, Mohammad Akbari

Understanding 3D spatial relationships remains a major limitation of current
Vision-Language Models (VLMs). Prior work has addressed this issue by creating
spatial question-answering (QA) datasets based on single images or indoor
videos. However, real-world embodied AI agents such as robots and self-driving
cars typically rely on ego-centric, multi-view observations. To this end, we
introduce Ego3D-Bench, a new benchmark designed to evaluate the spatial
reasoning abilities of VLMs using ego-centric, multi-view outdoor data.
Ego3D-Bench comprises over 8,600 QA pairs, created with significant involvement
from human annotators to ensure quality and diversity. We benchmark 16 SOTA
VLMs, including GPT-4o, Gemini1.5-Pro, InternVL3, and Qwen2.5-VL. Our results
reveal a notable performance gap between human level scores and VLM
performance, highlighting that current VLMs still fall short of human level
spatial understanding. To bridge this gap, we propose Ego3D-VLM, a
post-training framework that enhances 3D spatial reasoning of VLMs. Ego3D-VLM
generates cognitive map based on estimated global 3D coordinates, resulting in
12% average improvement on multi-choice QA and 56% average improvement on
absolute distance estimation. Ego3D-VLM is modular and can be integrated with
any existing VLM. Together, Ego3D-Bench and Ego3D-VLM offer valuable tools for
advancing toward human level spatial understanding in real-world, multi-view
environments.

### 2. [AI-driven Remote Facial Skin Hydration and TEWL Assessment from Selfie Images: A Systematic Solution](http://arxiv.org/pdf/2509.06282v1)

Authors: Cecelia Soh, Rizhao Cai, Monalisha Paul, Dennis Sng, Alex Kot

Skin health and disease resistance are closely linked to the skin barrier
function, which protects against environmental factors and water loss. Two key
physiological indicators can quantitatively represent this barrier function:
skin hydration (SH) and trans-epidermal water loss (TEWL). Measurement of SH
and TEWL is valuable for the public to monitor skin conditions regularly,
diagnose dermatological issues, and personalize their skincare regimens.
However, these measurements are not easily accessible to general users unless
they visit a dermatology clinic with specialized instruments. To tackle this
problem, we propose a systematic solution to estimate SH and TEWL from selfie
facial images remotely with smartphones. Our solution encompasses multiple
stages, including SH/TEWL data collection, data preprocessing, and formulating
a novel Skin-Prior Adaptive Vision Transformer model for SH/TEWL regression.
Through experiments, we identified the annotation imbalance of the SH/TEWL data
and proposed a symmetric-based contrastive regularization to reduce the model
bias due to the imbalance effectively. This work is the first study to explore
skin assessment from selfie facial images without physical measurements. It
bridges the gap between computer vision and skin care research, enabling
AI-driven accessible skin analysis for broader real-world applications.

### 3. [Prototype-Aware Multimodal Alignment for Open-Vocabulary Visual Grounding](http://arxiv.org/pdf/2509.06291v1)

Authors: Jiangnan Xie, Xiaolong Zheng, Liang Zheng

Visual Grounding (VG) aims to utilize given natural language queries to
locate specific target objects within images. While current transformer-based
approaches demonstrate strong localization performance in standard scene (i.e,
scenarios without any novel objects), they exhibit notable limitations in
open-vocabulary scene (i.e, both familiar and novel object categories during
testing). These limitations primarily stem from three key factors: (1)
imperfect alignment between visual and linguistic modalities, (2) insufficient
cross-modal feature fusion, and (3) ineffective utilization of semantic
prototype information. To overcome these challenges, we present Prototype-Aware
Multimodal Learning (PAML), an innovative framework that systematically
addresses these issues through several key components: First, we leverage ALBEF
to establish robust cross-modal alignment during initial feature encoding.
Subsequently, our Visual Discriminative Feature Encoder selectively enhances
salient object representations while suppressing irrelevant visual context. The
framework then incorporates a novel prototype discovering and inheriting
mechanism that extracts and aggregates multi-neighbor semantic prototypes to
facilitate open-vocabulary recognition. These enriched features undergo
comprehensive multimodal integration through our Multi-stage Decoder before
final bounding box regression. Extensive experiments across five benchmark
datasets validate our approach, showing competitive performance in standard
scene while achieving state-of-the-art results in open-vocabulary scene. Our
code is available at https://github.com/plankXie/PAML.

### 4. [Video-based Generalized Category Discovery via Memory-Guided Consistency-Aware Contrastive Learning](http://arxiv.org/pdf/2509.06306v1)

Authors: Zhang Jing, Pu Nan, Xie Yu Xiang, Guo Yanming, Lu Qianqi, Zou Shiwei, Yan Jie, Chen Yan

Generalized Category Discovery (GCD) is an emerging and challenging
open-world problem that has garnered increasing attention in recent years. Most
existing GCD methods focus on discovering categories in static images. However,
relying solely on static visual content is often insufficient to reliably
discover novel categories. To bridge this gap, we extend the GCD problem to the
video domain and introduce a new setting, termed Video-GCD. Thus, effectively
integrating multi-perspective information across time is crucial for accurate
Video-GCD. To tackle this challenge, we propose a novel Memory-guided
Consistency-aware Contrastive Learning (MCCL) framework, which explicitly
captures temporal-spatial cues and incorporates them into contrastive learning
through a consistency-guided voting mechanism. MCCL consists of two core
components: Consistency-Aware Contrastive Learning(CACL) and Memory-Guided
Representation Enhancement (MGRE). CACL exploits multiperspective temporal
features to estimate consistency scores between unlabeled instances, which are
then used to weight the contrastive loss accordingly. MGRE introduces a
dual-level memory buffer that maintains both feature-level and logit-level
representations, providing global context to enhance intra-class compactness
and inter-class separability. This in turn refines the consistency estimation
in CACL, forming a mutually reinforcing feedback loop between representation
learning and consistency modeling. To facilitate a comprehensive evaluation, we
construct a new and challenging Video-GCD benchmark, which includes action
recognition and bird classification video datasets. Extensive experiments
demonstrate that our method significantly outperforms competitive GCD
approaches adapted from image-based settings, highlighting the importance of
temporal information for discovering novel categories in videos. The code will
be publicly available.

### 5. [Text4Seg++: Advancing Image Segmentation via Generative Language Modeling](http://arxiv.org/pdf/2509.06321v1)

Authors: Mengcheng Lan, Chaofeng Chen, Jiaxing Xu, Zongrui Li, Yiping Ke, Xudong Jiang, Yingchen Yu, Yunqing Zhao, Song Bai

Multimodal Large Language Models (MLLMs) have shown exceptional capabilities
in vision-language tasks. However, effectively integrating image segmentation
into these models remains a significant challenge. In this work, we propose a
novel text-as-mask paradigm that casts image segmentation as a text generation
problem, eliminating the need for additional decoders and significantly
simplifying the segmentation process. Our key innovation is semantic
descriptors, a new textual representation of segmentation masks where each
image patch is mapped to its corresponding text label. We first introduce
image-wise semantic descriptors, a patch-aligned textual representation of
segmentation masks that integrates naturally into the language modeling
pipeline. To enhance efficiency, we introduce the Row-wise Run-Length Encoding
(R-RLE), which compresses redundant text sequences, reducing the length of
semantic descriptors by 74% and accelerating inference by $3\times$, without
compromising performance. Building upon this, our initial framework Text4Seg
achieves strong segmentation performance across a wide range of vision tasks.
To further improve granularity and compactness, we propose box-wise semantic
descriptors, which localizes regions of interest using bounding boxes and
represents region masks via structured mask tokens called semantic bricks. This
leads to our refined model, Text4Seg++, which formulates segmentation as a
next-brick prediction task, combining precision, scalability, and generative
efficiency. Comprehensive experiments on natural and remote sensing datasets
show that Text4Seg++ consistently outperforms state-of-the-art models across
diverse benchmarks without any task-specific fine-tuning, while remaining
compatible with existing MLLM backbones. Our work highlights the effectiveness,
scalability, and generalizability of text-driven image segmentation within the
MLLM framework.

### 6. [Quantitative Currency Evaluation in Low-Resource Settings through Pattern Analysis to Assist Visually Impaired Users](http://arxiv.org/pdf/2509.06331v1)

Authors: Md Sultanul Islam Ovi, Mainul Hossain, Md Badsha Biswas

Currency recognition systems often overlook usability and authenticity
assessment, especially in low-resource environments where visually impaired
users and offline validation are common. While existing methods focus on
denomination classification, they typically ignore physical degradation and
forgery, limiting their applicability in real-world conditions. This paper
presents a unified framework for currency evaluation that integrates three
modules: denomination classification using lightweight CNN models, damage
quantification through a novel Unified Currency Damage Index (UCDI), and
counterfeit detection using feature-based template matching. The dataset
consists of over 82,000 annotated images spanning clean, damaged, and
counterfeit notes. Our Custom_CNN model achieves high classification
performance with low parameter count. The UCDI metric provides a continuous
usability score based on binary mask loss, chromatic distortion, and structural
feature loss. The counterfeit detection module demonstrates reliable
identification of forged notes across varied imaging conditions. The framework
supports real-time, on-device inference and addresses key deployment challenges
in constrained environments. Results show that accurate, interpretable, and
compact solutions can support inclusive currency evaluation in practical
settings.

### 7. [Harnessing Object Grounding for Time-Sensitive Video Understanding](http://arxiv.org/pdf/2509.06335v1)

Authors: Tz-Ying Wu, Sharath Nittur Sridhar, Subarna Tripathi

We propose to improve the time-sensitive video understanding (TSV) capability
of video large language models (Video-LLMs) with grounded objects (GO). We
hypothesize that TSV tasks can benefit from GO within frames, which is
supported by our preliminary experiments on LITA, a state-of-the-art Video-LLM
for reasoning temporal localization. While augmenting prompts with textual
description of these object annotations improves the performance of LITA, it
also introduces extra token length and susceptibility to the noise in object
level information. To address this, we propose GO-Tokenizer, a lightweight
add-on module for Video-LLMs leveraging off-the-shelf object detectors to
encode compact object information on the fly. Experimental results demonstrate
that pretraining with GO-Tokenizer outperforms the vanilla Video-LLM and its
counterpart utilizing textual description of objects in the prompt. The gain
generalizes across different models, datasets and video understanding tasks
such as reasoning temporal localization and dense captioning.

### 8. [Your Super Resolution Model is not Enough for Tackling Real-World Scenarios](http://arxiv.org/pdf/2509.06387v1)

Authors: Dongsik Yoon, Jongeun Kim

Despite remarkable progress in Single Image Super-Resolution (SISR),
traditional models often struggle to generalize across varying scale factors,
limiting their real-world applicability. To address this, we propose a plug-in
Scale-Aware Attention Module (SAAM) designed to retrofit modern fixed-scale SR
models with the ability to perform arbitrary-scale SR. SAAM employs
lightweight, scale-adaptive feature extraction and upsampling, incorporating
the Simple parameter-free Attention Module (SimAM) for efficient guidance and
gradient variance loss to enhance sharpness in image details. Our method
integrates seamlessly into multiple state-of-the-art SR backbones (e.g., SCNet,
HiT-SR, OverNet), delivering competitive or superior performance across a wide
range of integer and non-integer scale factors. Extensive experiments on
benchmark datasets demonstrate that our approach enables robust multi-scale
upscaling with minimal computational overhead, offering a practical solution
for real-world scenarios.

### 9. [AI-based response assessment and prediction in longitudinal imaging for brain metastases treated with stereotactic radiosurgery](http://arxiv.org/pdf/2509.06396v1)

Authors: Lorenz Achim Kuhn, Daniel Abler, Jonas Richiardi, Andreas F. Hottinger, Luis Schiappacasse, Vincent Dunet, Adrien Depeursinge, Vincent Andrearczyk

Brain Metastases (BM) are a large contributor to mortality of patients with
cancer. They are treated with Stereotactic Radiosurgery (SRS) and monitored
with Magnetic Resonance Imaging (MRI) at regular follow-up intervals according
to treatment guidelines. Analyzing and quantifying this longitudinal imaging
represents an intractable workload for clinicians. As a result, follow-up
images are not annotated and merely assessed by observation. Response to
treatment in longitudinal imaging is being studied, to better understand growth
trajectories and ultimately predict treatment success or toxicity as early as
possible. In this study, we implement an automated pipeline to curate a large
longitudinal dataset of SRS treatment data, resulting in a cohort of 896 BMs in
177 patients who were monitored for >360 days at approximately two-month
intervals at Lausanne University Hospital (CHUV). We use a data-driven
clustering to identify characteristic trajectories. In addition, we predict 12
months lesion-level response using classical as well as graph machine learning
Graph Machine Learning (GML). Clustering revealed 5 dominant growth
trajectories with distinct final response categories. Response prediction
reaches up to 0.90 AUC (CI95%=0.88-0.92) using only pre-treatment and first
follow-up MRI with gradient boosting. Similarly, robust predictive performance
of up to 0.88 AUC (CI95%=0.86-0.90) was obtained using GML, offering more
flexibility with a single model for multiple input time-points configurations.
Our results suggest potential automation and increased precision for the
comprehensive assessment and prediction of BM response to SRS in longitudinal
MRI. The proposed pipeline facilitates scalable data curation for the
investigation of BM growth patterns, and lays the foundation for clinical
decision support systems aiming at optimizing personalized care.

### 10. [3DOF+Quantization: 3DGS quantization for large scenes with limited Degrees of Freedom](http://arxiv.org/pdf/2509.06400v1)

Authors: Matthieu Gendrin, Stéphane Pateux, Théo Ladune

3D Gaussian Splatting (3DGS) is a major breakthrough in 3D scene
reconstruction. With a number of views of a given object or scene, the
algorithm trains a model composed of 3D gaussians, which enables the production
of novel views from arbitrary points of view. This freedom of movement is
referred to as 6DoF for 6 degrees of freedom: a view is produced for any
position (3 degrees), orientation of camera (3 other degrees). On large scenes,
though, the input views are acquired from a limited zone in space, and the
reconstruction is valuable for novel views from the same zone, even if the
scene itself is almost unlimited in size. We refer to this particular case as
3DoF+, meaning that the 3 degrees of freedom of camera position are limited to
small offsets around the central position. Considering the problem of
coordinate quantization, the impact of position error on the projection error
in pixels is studied. It is shown that the projection error is proportional to
the squared inverse distance of the point being projected. Consequently, a new
quantization scheme based on spherical coordinates is proposed. Rate-distortion
performance of the proposed method are illustrated on the well-known Garden
scene.

### Computers and Society

### 1. [Simulating Dispute Mediation with LLM-Based Agents for Legal Research](http://arxiv.org/pdf/2509.06586v1)

Authors: Junjie Chen, Haitao Li, Minghao Qin, Yujia Zhou, Yanxue Ren, Wuyue Wang, Yiqun Liu, Yueyue Wu, Qingyao Ai

Legal dispute mediation plays a crucial role in resolving civil disputes, yet
its empirical study is limited by privacy constraints and complex multivariate
interactions. To address this limitation, we present AgentMediation, the first
LLM-based agent framework for simulating dispute mediation. It simulates
realistic mediation processes grounded in real-world disputes and enables
controlled experimentation on key variables such as disputant strategies,
dispute causes, and mediator expertise. Our empirical analysis reveals patterns
consistent with sociological theories, including Group Polarization and
Surface-level Consensus. As a comprehensive and extensible platform,
AgentMediation paves the way for deeper integration of social science and AI in
legal research.

### 2. [NeedForHeat DataGear: An Open Monitoring System to Accelerate the Residential Heating Transition](http://arxiv.org/pdf/2509.06927v1)

Authors: Henri ter Hofte, Nick van Ravenzwaaij

We introduce NeedForHeat DataGear: an open hardware and open software data
collection system designed to accelerate the residential heating transition.
NeedForHeat DataGear collects time series monitoring data in homes that have
not yet undergone a heating transition, enabling assessment of real-life
thermal characteristics, heating system efficiency, and residents' comfort
needs. This paper outlines its architecture and functionalities, emphasizing
its modularity, adaptability, and cost-effectiveness for field data
acquisition. Unlike conventional domestic monitoring solutions focused on home
automation, direct feedback, or post-installation heat pump monitoring, it
prioritizes time series data we deemed essential to evaluate the current
situation in existing homes before the heating transition. Designed for
seamless deployment across diverse households, NeedForHeat DataGear combines
openness, security, and privacy with a low-cost, user-friendly approach, making
it a valuable tool for researchers, energy professionals, and energy coaches.

### 3. [AI for Scientific Discovery is a Social Problem](http://arxiv.org/pdf/2509.06580v1)

Authors: Georgia Channing, Avijit Ghosh

Artificial intelligence promises to accelerate scientific discovery, yet its
benefits remain unevenly distributed. While technical obstacles such as scarce
data, fragmented standards, and unequal access to computation are significant,
we argue that the primary barriers are social and institutional. Narratives
that defer progress to speculative "AI scientists," the undervaluing of data
and infrastructure contributions, misaligned incentives, and gaps between
domain experts and machine learning researchers all constrain impact. We
highlight four interconnected challenges: community dysfunction, research
priorities misaligned with upstream needs, data fragmentation, and
infrastructure inequities. We argue that their roots lie in cultural and
organizational practices. Addressing them requires not only technical
innovation but also intentional community-building, cross-disciplinary
education, shared benchmarks, and accessible infrastructure. We call for
reframing AI for science as a collective social project, where sustainable
collaboration and equitable participation are treated as prerequisites for
technical progress.

### 4. [Explained, yet misunderstood: How AI Literacy shapes HR Managers' interpretation of User Interfaces in Recruiting Recommender Systems](http://arxiv.org/pdf/2509.06475v1)

Authors: Yannick Kalff, Katharina Simbeck

AI-based recommender systems increasingly influence recruitment decisions.
Thus, transparency and responsible adoption in Human Resource Management (HRM)
are critical. This study examines how HR managers' AI literacy influences their
subjective perception and objective understanding of explainable AI (XAI)
elements in recruiting recommender dashboards. In an online experiment, 410
German-based HR managers compared baseline dashboards to versions enriched with
three XAI styles: important features, counterfactuals, and model criteria. Our
results show that the dashboards used in practice do not explain AI results and
even keep AI elements opaque. However, while adding XAI features improves
subjective perceptions of helpfulness and trust among users with moderate or
high AI literacy, it does not increase their objective understanding. It may
even reduce accurate understanding, especially with complex explanations. Only
overlays of important features significantly aided the interpretations of
high-literacy users. Our findings highlight that the benefits of XAI in
recruitment depend on users' AI literacy, emphasizing the need for tailored
explanation strategies and targeted literacy training in HRM to ensure fair,
transparent, and effective adoption of AI.

### 5. [An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection](http://arxiv.org/pdf/2509.06920v1)

Authors: Haywood Gelman, John D. Hastings, David Kenley

Insider threats are a growing organizational problem due to the complexity of
identifying their technical and behavioral elements. A large research body is
dedicated to the study of insider threats from technological, psychological,
and educational perspectives. However, research in this domain has been
generally dependent on datasets that are static and limited access which
restricts the development of adaptive detection models. This study introduces a
novel, ethically grounded approach that uses the large language model (LLM)
Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which
contain indicators of insider threat scenarios. The messages reflect real-world
data distributions by being highly imbalanced (1% insider threats). The syslogs
were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with
their performance evaluated through statistical metrics including precision,
recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across
nearly all metrics, particularly in reducing false alarms and improving
detection accuracy. The results show strong promise for the use of LLMs in
synthetic dataset generation and insider threat detection.

### Databases

### 1. [MCTuner: Spatial Decomposition-Enhanced Database Tuning via LLM-Guided Exploration](http://arxiv.org/pdf/2509.06298v1)

Authors: Zihan Yan, Rui Xi, Mengshu Hou

Database knob tuning is essential for optimizing the performance of modern
database management systems, which often expose hundreds of knobs with
continuous or categorical values. However, the large number of knobs and the
vast configuration space make it difficult to identify optimal settings
efficiently. Although learning-based tuning has shown promise, existing
approaches either ignore domain knowledge by relying solely on benchmark
feedback or struggle to explore the high-dimensional knob space, resulting in
high tuning costs and suboptimal performance. To address these challenges, we
propose MCTuner, an adaptive knob tuning framework that minimizes exploration
in ineffective regions of the configuration space. MCTuner employs a
Mixture-of-Experts (MoE) mechanism with specialized LLMs to identify
performance-critical knobs. In further, MCTuner introduces the first spatial
decomposition algorithm that recursively partitions the space into hierarchical
subspaces, on which Bayesian Optimization is performed to efficiently search
for near-optimal configurations. Evaluated on different benchmarks (OLAP, OLTP,
and HTAP), MCTuner achieves up to 19.2% performance gains and 1.4x faster
configuration discovery per iteration compared to state-of-the-art methods.

### 2. [Relational Algebras for Subset Selection and Optimisation](http://arxiv.org/pdf/2509.06439v1)

Authors: David Robert Pratten, Luke Mathieson, Fahimeh Ramezani

The database community lacks a unified relational query language for subset
selection and optimisation queries, limiting both user expression and query
optimiser reasoning about such problems. Decades of research (latterly under
the rubric of prescriptive analytics) have produced powerful evaluation
algorithms with incompatible, ad-hoc SQL extensions that specify and filter
through distinct mechanisms. We present the first unified algebraic foundation
for these queries, introducing relational exponentiation to complete the
fundamental algebraic operations alongside union (addition) and cross product
(multiplication). First, we extend relational algebra to complete domain
relations-relations defined by characteristic functions rather than explicit
extensions-achieving the expressiveness of NP-complete/hard problems, while
simultaneously providing query safety for finite inputs. Second, we introduce
solution sets, a higher-order relational algebra over sets of relations that
naturally expresses search spaces as functions f: Base to Decision, yielding
|Decision|^|Base| candidate relations. Third, we provide structure-preserving
translation semantics from solution sets to standard relational algebra,
enabling mechanical translation to existing evaluation algorithms. This
framework achieves the expressiveness of the most powerful prior approaches
while providing the theoretical clarity and compositional properties absent in
previous work. We demonstrate the capabilities these algebras open up through a
polymorphic SQL where standard clauses seamlessly express data management,
subset selection, and optimisation queries within a single paradigm.

### 3. [Proof-Carrying Numbers (PCN): A Protocol for Trustworthy Numeric Answers from LLMs via Claim Verification](http://arxiv.org/pdf/2509.06902v1)

Authors: Aivin V. Solatorio

Large Language Models (LLMs) as stochastic systems may generate numbers that
deviate from available data, a failure known as \emph{numeric hallucination}.
Existing safeguards -- retrieval-augmented generation, citations, and
uncertainty estimation -- improve transparency but cannot guarantee fidelity:
fabricated or misquoted values may still be displayed as if correct. We propose
\textbf{Proof-Carrying Numbers (PCN)}, a presentation-layer protocol that
enforces numeric fidelity through mechanical verification. Under PCN, numeric
spans are emitted as \emph{claim-bound tokens} tied to structured claims, and a
verifier checks each token under a declared policy (e.g., exact equality,
rounding, aliases, or tolerance with qualifiers). Crucially, PCN places
verification in the \emph{renderer}, not the model: only claim-checked numbers
are marked as verified, and all others default to unverified. This separation
prevents spoofing and guarantees fail-closed behavior. We formalize PCN and
prove soundness, completeness under honest tokens, fail-closed behavior, and
monotonicity under policy refinement. PCN is lightweight and model-agnostic,
integrates seamlessly into existing applications, and can be extended with
cryptographic commitments. By enforcing verification as a mandatory step before
display, PCN establishes a simple contract for numerically sensitive settings:
\emph{trust is earned only by proof}, while the absence of a mark communicates
uncertainty.

### Distributed, Parallel, and Cluster Computing

### 1. [MaaSO: SLO-aware Orchestration of Heterogeneous Model Instances for MaaS](http://arxiv.org/pdf/2509.06362v1)

Authors: Mo Xuan, Zhang yue, Wu Weigang

Model-as-a-Service (MaaS) platforms face diverse Service Level Objective
(SLO) requirements stemming from various large language model (LLM)
applications, manifested in contextual complexity, first-token latency, and
between-token latency. On the other hand, an LLM instance, when configured with
different parallelism strategies and inference batch sizes, exhibits distinct
performance characteristics and can thus be used to serve different SLO
requirements. However, current LLM inference systems typically deploy instances
of the same model with identical configurations, lacking mechanisms to leverage
such heterogeneity. To fill this research gap, we propose MaaSO, the first MaaS
Orchestrator, which comprises three modules: (1) a profiler characterizing
instance performance under diverse parallelism strategies and inference batch
sizes; (2) a placer optimizing heterogeneous instance configurations; (3) a
distributor enabling SLO-aware request distribution and preventing cascaded
timeouts in continuous batching. Experiments show that MaaSO improves the SLO
satisfaction ratio by 15 to 30% and reduces response latency by 40 to 60%
compared to existing approaches, and significantly lowers overall orchestration
overhead.

### 2. [IM-PIR: In-Memory Private Information Retrieval](http://arxiv.org/pdf/2509.06514v1)

Authors: Mpoki Mwaisela, Peterson Yuhala, Pascal Felber, Valerio Schiavoni

Private information retrieval (PIR) is a cryptographic primitive that allows
a client to securely query one or multiple servers without revealing their
specific interests. In spite of their strong security guarantees, current PIR
constructions are computationally costly. Specifically, most PIR
implementations are memory-bound due to the need to scan extensive databases
(in the order of GB), making them inherently constrained by the limited memory
bandwidth in traditional processor-centric computing
architectures.Processing-in-memory (PIM) is an emerging computing paradigm that
augments memory with compute capabilities, addressing the memory bandwidth
bottleneck while simultaneously providing extensive parallelism.Recent research
has demonstrated PIM's potential to significantly improve performance across a
range of data-intensive workloads, including graph processing, genome analysis,
and machine learning.
  In this work, we propose the first PIM-based architecture for multi-server
PIR. We discuss the algorithmic foundations of the latter and show how its
operations align with the core strengths of PIM architectures: extensive
parallelism and high memory bandwidth. Based on this observation, we design and
implement IM-PIR, a PIM-based multi-server PIR approach on top of UPMEM PIM,
the first openly commercialized PIM architecture. Our evaluation demonstrates
that a PIM-based multi-server PIR implementation significantly improves query
throughput by more than 3.7x when compared to a standard CPU-based PIR
approach.

### 3. [Mangrove: Fast and Parallelizable State Replication for Blockchains](http://arxiv.org/pdf/2509.06616v1)

Authors: Anton Paramonov, Yann Vonlanthen, Quentin Kniep, Jakub Sliwinski, Roger Wattenhofer

Mangrove is a novel scaling approach to building blockchains with parallel
smart contract support. Unlike in monolithic blockchains, where a single
consensus mechanism determines a strict total order over all transactions,
Mangrove uses separate consensus instances per smart contract, without a global
order. To allow multiple instances to run in parallel while ensuring that no
conflicting transactions are committed, we propose a mechanism called Parallel
Optimistic Agreement. Additionally, for simple transactions, we leverage a
lightweight Byzantine Reliable Broadcast primitive to reduce latency. Mangrove
is optimized for performance under optimistic conditions, where there is no
misbehavior and the network is synchronous. Under these conditions, our
protocol can achieve a latency of 2 communication steps between creating and
executing a transaction.

### 4. [FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving](http://arxiv.org/pdf/2509.06261v1)

Authors: Kyungmin Bin, Seungbeom Choi, Jimyoung Son, Jieun Choi, Daseul Bae, Daehyeon Baek, Kihyo Moon, Minsung Jang, Hyojung Lee

Recent advances in Post-Training Quantization (PTQ) techniques have
significantly increased demand for serving quantized large language models
(LLMs), enabling higher throughput and substantially reduced memory usage with
minimal accuracy loss. Quantized models address memory constraints in LLMs and
enhance GPU resource utilization through efficient GPU sharing. However,
quantized models have smaller KV block sizes than non-quantized models, causing
limited memory efficiency due to memory fragmentation. Also, distinct resource
usage patterns between quantized and non-quantized models require efficient
scheduling to maximize throughput. To address these challenges, we propose
FineServe, an inference serving framework for mixed-precision LLMs. FineServe's
key contributions include: (1) KV Slab, a precision-aware adaptive memory
management technique dynamically allocating KV cache based on model
quantization characteristics, significantly reducing GPU memory fragmentation,
and (2) a two-level scheduling framework comprising a global scheduler that
places models to GPUs based on request rates, latency SLOs, and memory
constraints and efficiency, and a local scheduler that adaptively adjusts batch
sizes according to real-time request fluctuations. Experimental results
demonstrate that FineServe achieves up to 2.2x higher SLO attainment and 1.8x
higher token generation throughput compared to the state-of-the-art GPU sharing
systems.

### 5. [Several Performance Bounds on Decentralized Online Optimization are Highly Conservative and Potentially Misleading](http://arxiv.org/pdf/2509.06466v1)

Authors: Erwan Meunier, Julien M. Hendrickx

We analyze Decentralized Online Optimization algorithms using the Performance
Estimation Problem approach which allows, to automatically compute exact
worst-case performance of optimization algorithms. Our analysis shows that
several available performance guarantees are very conservative, sometimes by
multiple orders of magnitude, and can lead to misguided choices of algorithm.
Moreover, at least in terms of worst-case performance, some algorithms appear
not to benefit from inter-agent communications for a significant period of
time. We show how to improve classical methods by tuning their step-sizes, and
find that we can save up to 20% on their actual worst-case performance regret.

### 6. [Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing](http://arxiv.org/pdf/2509.06552v1)

Authors: Zheqi Lv, Wenqiao Zhang, Kairui Fu, Qi Tian, Shengyu Zhang, Jiajie Su, Jingyuan Chen, Kun Kuang, Fei Wu

The on-device real-time data distribution shift on devices challenges the
generalization of lightweight on-device models. This critical issue is often
overlooked in current research, which predominantly relies on data-intensive
and computationally expensive fine-tuning approaches. To tackle this, we
introduce Persona, a novel personalized method using a prototype-based,
backpropagation-free parameter editing framework to enhance model
generalization without post-deployment retraining. Persona employs a neural
adapter in the cloud to generate a parameter editing matrix based on real-time
device data. This matrix adeptly adapts on-device models to the prevailing data
distributions, efficiently clustering them into prototype models. The
prototypes are dynamically refined via the parameter editing matrix,
facilitating efficient evolution. Furthermore, the integration of cross-layer
knowledge transfer ensures consistent and context-aware multi-layer parameter
changes and prototype assignment. Extensive experiments on vision task and
recommendation task on multiple datasets confirm Persona's effectiveness and
generality.

### 7. [Distributed Automatic Generation Control subject to Ramp-Rate-Limits: Anytime Feasibility and Uniform Network-Connectivity](http://arxiv.org/pdf/2509.06588v1)

Authors: Mohammadreza Doostmohammadian, Hamid R. Rabiee

This paper considers automatic generation control over an information-sharing
network of communicating generators as a multi-agent system. The optimization
solution is distributed among the agents based on information consensus
algorithms, while addressing the generators' ramp-rate-limits (RRL). This is
typically ignored in the existing linear/nonlinear optimization solutions but
they exist in real-time power generation scenarios. Without addressing the RRL,
the generators cannot follow the assigned rate of generating power by the
optimization algorithm; therefore, the existing solutions may not necessarily
converge to the exact optimal cost or may lose feasibility in practice. The
proposed solution in this work addresses the ramp-rate-limit constraint along
with the box constraint (limits on the generated powers) and the
coupling-constraint (generation-demand balance) at all iteration times of the
algorithm. The latter is referred to as the anytime feasibility and implies
that at every termination point of the algorithm, the balance between the
demand and generated power holds. To improve the convergence rate of the
algorithm we further consider internal signum-based nonlinearity. We also show
that our solution can tolerate communication link removal. This follows from
the uniform-connectivity assumption on the communication network.

### Digital Libraries

### 1. [Compare: A Framework for Scientific Comparisons](http://arxiv.org/pdf/2509.06412v1)

Authors: Moritz Staudinger, Wojciech Kusa, Matteo Cancellieri, David Pride, Petr Knoth, Allan Hanbury

Navigating the vast and rapidly increasing sea of academic publications to
identify institutional synergies, benchmark research contributions and pinpoint
key research contributions has become an increasingly daunting task, especially
with the current exponential increase in new publications. Existing tools
provide useful overviews or single-document insights, but none supports
structured, qualitative comparisons across institutions or publications.
  To address this, we demonstrate Compare, a novel framework that tackles this
challenge by enabling sophisticated long-context comparisons of scientific
contributions. Compare empowers users to explore and analyze research overlaps
and differences at both the institutional and publication granularity, all
driven by user-defined questions and automatic retrieval over online resources.
For this we leverage on Retrieval-Augmented Generation over evolving data
sources to foster long context knowledge synthesis. Unlike traditional
scientometric tools, Compare goes beyond quantitative indicators by providing
qualitative, citation-supported comparisons.

### Discrete Mathematics

### 1. [Optimal Average Disk-Inspection via Fermat's Principle](http://arxiv.org/pdf/2509.06334v1)

Authors: Konstantinos Georgiou

This work resolves the optimal average-case cost of the Disk-Inspection
problem, a variant of Bellman's 1955 lost-in-a-forest problem. In
Disk-Inspection, a mobile agent starts at the center of a unit disk and follows
a trajectory that inspects perimeter points whenever the disk does not obstruct
visibility. The worst-case cost was solved optimally in 1957 by Isbell, but the
average-case version remained open, with heuristic upper bounds proposed by
Gluss in 1961 and improved only recently.
  Our approach applies Fermat's Principle of Least Time to a recently proposed
discretization framework, showing that optimal solutions are captured by a
one-parameter family of recurrences independent of the discretization size. In
the continuum limit these recurrences give rise to a single-parameter optimal
control problem, whose trajectories coincide with limiting solutions of the
original Disk-Inspection problem. A crucial step is proving that the optimal
initial condition generates a trajectory that avoids the unit disk, thereby
validating the optics formulation and reducing the many-variable optimization
to a rigorous one-parameter problem. In particular, this disproves Gluss's
conjecture that optimal trajectories must touch the disk.
  Our analysis determines the exact optimal average-case inspection cost, equal
to $3.549259\ldots$ and certified to at least six digits of accuracy.

### 2. [Verifying Sampling Algorithms via Distributional Invariants](http://arxiv.org/pdf/2509.06410v1)

Authors: Kevin Batz, Joost-Pieter Katoen, Tobias Winkler, Daniel Zilken

This paper develops a verification framework aimed at establishing the
correctness of discrete sampling algorithms. We do so by considering
probabilistic programs as distribution transformers. Inspired by recent work on
distributional verification of Markov models, we introduce the notion of
(inductive) distributional loop invariants for discrete probabilistic programs.
These invariants are embedded in a Hoare-like verification framework that
includes proof rules for total and partial correctness. To illustrate the
applicability of our framework, we prove the correctness of two discrete
sampling algorithms: the Fast Dice Roller and the Fast Loaded Dice Roller.

### 3. [Relational Algebras for Subset Selection and Optimisation](http://arxiv.org/pdf/2509.06439v1)

Authors: David Robert Pratten, Luke Mathieson, Fahimeh Ramezani

The database community lacks a unified relational query language for subset
selection and optimisation queries, limiting both user expression and query
optimiser reasoning about such problems. Decades of research (latterly under
the rubric of prescriptive analytics) have produced powerful evaluation
algorithms with incompatible, ad-hoc SQL extensions that specify and filter
through distinct mechanisms. We present the first unified algebraic foundation
for these queries, introducing relational exponentiation to complete the
fundamental algebraic operations alongside union (addition) and cross product
(multiplication). First, we extend relational algebra to complete domain
relations-relations defined by characteristic functions rather than explicit
extensions-achieving the expressiveness of NP-complete/hard problems, while
simultaneously providing query safety for finite inputs. Second, we introduce
solution sets, a higher-order relational algebra over sets of relations that
naturally expresses search spaces as functions f: Base to Decision, yielding
|Decision|^|Base| candidate relations. Third, we provide structure-preserving
translation semantics from solution sets to standard relational algebra,
enabling mechanical translation to existing evaluation algorithms. This
framework achieves the expressiveness of the most powerful prior approaches
while providing the theoretical clarity and compositional properties absent in
previous work. We demonstrate the capabilities these algebras open up through a
polymorphic SQL where standard clauses seamlessly express data management,
subset selection, and optimisation queries within a single paradigm.

### 4. [Codes Correcting Transpositions of Consecutive Symbols](http://arxiv.org/pdf/2509.06692v1)

Authors: Mladen Kovačević, Keshav Goyal, Han Mao Kiah

The problem of correcting transpositions (or swaps) of consecutive symbols in
$ q $-ary strings is studied. A family of codes correcting a transposition at
an arbitrary location is described and proved to have asymptotically optimal
redundancy. Additionally, an improved construction is given over a binary
alphabet. Bounds on the cardinality of codes correcting $ t = \textrm{const} $
transpositions are obtained. A lower bound on the achievable asymptotic rate of
optimal codes correcting $ t = \tau n $ transpositions is derived. Finally, a
construction of codes correcting all possible patterns of transpositions is
presented, and the corresponding lower bound on the zero-error capacity of the
$ q $-ary transposition channel is stated.

### Data Structures and Algorithms

### 1. [Zero-Freeness is All You Need: A Weitz-Type FPTAS for the Entire Lee-Yang Zero-Free Region](http://arxiv.org/pdf/2509.06623v1)

Authors: Shuai Shao, Ke Shi

We present a Weitz-type FPTAS for the ferromagnetic Ising model across the
entire Lee-Yang zero-free region, without relying on the strong spatial mixing
(SSM) property. Our algorithm is Weitz-type for two reasons. First, it
expresses the partition function as a telescoping product of ratios, with the
key being to approximate each ratio. Second, it uses Weitz's self-avoiding walk
tree, and truncates it at logarithmic depth to give a good and efficient
approximation. The key difference from the standard Weitz algorithm is that we
approximate a carefully designed edge-deletion ratio instead of the marginal
probability of a vertex's spin, ensuring our algorithm does not require SSM.
  Furthermore, by establishing local dependence of coefficients (LDC), we
indeed prove a novel form of SSM for these edge-deletion ratios, which, in
turn, implies the standard SSM for the random cluster model. This is the first
SSM result for the random cluster model on general graphs, beyond lattices. We
prove LDC using a new division relation, and remarkably, such relations hold
quite universally. As a result, we establish LDC for a variety of models.
Combined with existing zero-freeness results for these models, we derive new
SSM results for them. Our work suggests that both Weitz-type FPTASes and SSM
can be derived from zero-freeness, while zero-freeness alone suffices for
Weitz-type FPTASes, SSM additionally requires LDC, a combinatorial property
independent of zero-freeness.

### 2. [The Steiner Shortest Path Tree Problem](http://arxiv.org/pdf/2509.06789v1)

Authors: Omer Asher, Yefim Dinitz, Shlomi Dolev, Li-on Raviv, Baruch Schieber

We introduce and study a novel problem of computing a shortest path tree with
a minimum number of non-terminals. It can be viewed as an (unweighted) Steiner
Shortest Path Tree (SSPT) that spans a given set of terminal vertices by
shortest paths from a given source while minimizing the number of nonterminal
vertices included in the tree. This problem is motivated by applications where
shortest-path connections from a source are essential, and where reducing the
number of intermediate vertices helps limit cost, complexity, or overhead. We
show that the SSPT problem is NP-hard. To approximate it, we introduce and
study the shortest path subgraph of a graph. Using it, we show an
approximation-preserving reduction of SSPT to the uniform vertex-weighted
variant of the Directed Steiner Tree (DST) problem, termed UVDST. Consequently,
the algorithm of [Grandoni et al., 2023] approximating DST implies a
quasi-polynomial polylog-approximation algorithm for SSPT. We present a
polynomial polylog-approximation algorithm for UVDST, and thus for SSPT, for a
restricted class of graphs.

### 3. [Engineering Select Support for Hybrid Bitvectors](http://arxiv.org/pdf/2509.06900v1)

Authors: Eric Chiu, Dominik Kempa

One of the central problems in the design of compressed data structures is
the efficient support for rank and select queries on bitvectors. These two
operations form the backbone of more complex data structures (such as wavelet
trees) used for the compact representation of texts, trees, graphs, or grids.
Their efficient implementation is one of the most frequently studied problems
in compressed data structures.
  One effective solution is the so-called hybrid bitvector implementation,
which partitions the input bitvector into blocks and adaptively selects an
encoding method, such as run-length, plain, or minority encoding, based on
local redundancy. Experiments have shown that hybrid bitvectors achieve
excellent all-around performance on repetitive and non-repetitive inputs.
  However, current implementations support only rank queries (i.e., counting
the number of ones up to a given position) and lack support for select queries.
This limitation significantly restricts their applicability. In this paper, we
propose a method to add support for select queries to hybrid bitvectors, and we
conduct an extensive set of experiments. Our results show that hybrid
bitvectors offer excellent performance, matching the speed of the fastest and
the space efficiency of the most compact existing bitvectors.

### Emerging Technologies

### 1. [A Spatio-Temporal Graph Neural Networks Approach for Predicting Silent Data Corruption inducing Circuit-Level Faults](http://arxiv.org/pdf/2509.06289v1)

Authors: Shaoqi Wei, Senling Wang, Hiroshi Kai, Yoshinobu Higami, Ruijun Ma, Tianming Ni, Xiaoqing Wen, Hiroshi Takahashi

Silent Data Errors (SDEs) from time-zero defects and aging degrade
safety-critical systems. Functional testing detects SDE-related faults but is
expensive to simulate. We present a unified spatio-temporal graph convolutional
network (ST-GCN) for fast, accurate prediction of long-cycle fault impact
probabilities (FIPs) in large sequential circuits, supporting quantitative risk
assessment. Gate-level netlists are modeled as spatio-temporal graphs to
capture topology and signal timing; dedicated spatial and temporal encoders
predict multi-cycle FIPs efficiently. On ISCAS-89 benchmarks, the method
reduces simulation time by more than 10x while maintaining high accuracy (mean
absolute error 0.024 for 5-cycle predictions). The framework accepts features
from testability metrics or fault simulation, allowing efficiency-accuracy
trade-offs. A test-point selection study shows that choosing observation points
by predicted FIPs improves detection of long-cycle, hard-to-detect faults. The
approach scales to SoC-level test strategy optimization and fits downstream
electronic design automation flows.

### Formal Languages and Automata Theory

### 1. [On Synthesis of Timed Regular Expressions](http://arxiv.org/pdf/2509.06262v1)

Authors: Ziran Wang, Jie An, Naijun Zhan, Miaomiao Zhang, Zhenya Zhang

Timed regular expressions serve as a formalism for specifying real-time
behaviors of Cyber-Physical Systems. In this paper, we consider the synthesis
of timed regular expressions, focusing on generating a timed regular expression
consistent with a given set of system behaviors including positive and negative
examples, i.e., accepting all positive examples and rejecting all negative
examples. We first prove the decidability of the synthesis problem through an
exploration of simple timed regular expressions. Subsequently, we propose our
method of generating a consistent timed regular expression with minimal length,
which unfolds in two steps. The first step is to enumerate and prune candidate
parametric timed regular expressions. In the second step, we encode the
requirement that a candidate generated by the first step is consistent with the
given set into a Satisfiability Modulo Theories (SMT) formula, which is
consequently solved to determine a solution to parametric time constraints.
Finally, we evaluate our approach on benchmarks, including randomly generated
behaviors from target timed models and a case study.

### Graphics

### 1. [From Rigging to Waving: 3D-Guided Diffusion for Natural Animation of Hand-Drawn Characters](http://arxiv.org/pdf/2509.06573v1)

Authors: Jie Zhou, Linzi Qu, Miu-Ling Lam, Hongbo Fu

Hand-drawn character animation is a vibrant field in computer graphics,
presenting challenges in achieving geometric consistency while conveying
expressive motion. Traditional skeletal animation methods maintain geometric
consistency but struggle with complex non-rigid elements like flowing hair and
skirts, leading to unnatural deformation. Conversely, video diffusion models
synthesize realistic dynamics but often create geometric distortions in
stylized drawings due to domain gaps. This work proposes a hybrid animation
system that combines skeletal animation and video diffusion. Initially, coarse
images are generated from characters retargeted with skeletal animations for
geometric guidance. These images are then enhanced in texture and secondary
dynamics using video diffusion priors, framing this enhancement as an
inpainting task. A domain-adapted diffusion model refines user-masked regions
needing improvement, especially for secondary dynamics. To enhance motion
realism further, we introduce a Secondary Dynamics Injection (SDI) strategy in
the denoising process, incorporating features from a pre-trained diffusion
model enriched with human motion priors. Additionally, to tackle unnatural
deformations from low-poly single-mesh character modeling, we present a Hair
Layering Modeling (HLM) technique that uses segmentation maps to separate hair
from the body, allowing for more natural animation of long-haired characters.
Extensive experiments show that our system outperforms state-of-the-art methods
in both quantitative and qualitative evaluations.

### 2. [From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans](http://arxiv.org/pdf/2509.06607v1)

Authors: Marilyn Keller, Keenon Werling, Soyong Shin, Scott Delp, Sergi Pujades, C. Karen Liu, Michael J. Black

Great progress has been made in estimating 3D human pose and shape from
images and video by training neural networks to directly regress the parameters
of parametric human models like SMPL. However, existing body models have
simplified kinematic structures that do not correspond to the true joint
locations and articulations in the human skeletal system, limiting their
potential use in biomechanics. On the other hand, methods for estimating
biomechanically accurate skeletal motion typically rely on complex motion
capture systems and expensive optimization methods. What is needed is a
parametric 3D human model with a biomechanically accurate skeletal structure
that can be easily posed. To that end, we develop SKEL, which re-rigs the SMPL
body model with a biomechanics skeleton. To enable this, we need training data
of skeletons inside SMPL meshes in diverse poses.
  We build such a dataset by optimizing biomechanically accurate skeletons
inside SMPL meshes from AMASS sequences. We then learn a regressor from SMPL
mesh vertices to the optimized joint locations and bone rotations. Finally, we
re-parametrize the SMPL mesh with the new kinematic parameters. The resulting
SKEL model is animatable like SMPL but with fewer, and
biomechanically-realistic, degrees of freedom. We show that SKEL has more
biomechanically accurate joint locations than SMPL, and the bones fit inside
the body surface better than previous methods. By fitting SKEL to SMPL meshes
we are able to "upgrade" existing human pose and shape datasets to include
biomechanical parameters. SKEL provides a new tool to enable biomechanics in
the wild, while also providing vision and graphics researchers with a better
constrained and more realistic model of human articulation. The model, code,
and data are available for research at https://skel.is.tue.mpg.de..

### 3. [Scaling Transformer-Based Novel View Synthesis Models with Token Disentanglement and Synthetic Data](http://arxiv.org/pdf/2509.06950v1)

Authors: Nithin Gopalakrishnan Nair, Srinivas Kaza, Xuan Luo, Vishal M. Patel, Stephen Lombardi, Jungyeon Park

Large transformer-based models have made significant progress in
generalizable novel view synthesis (NVS) from sparse input views, generating
novel viewpoints without the need for test-time optimization. However, these
models are constrained by the limited diversity of publicly available scene
datasets, making most real-world (in-the-wild) scenes out-of-distribution. To
overcome this, we incorporate synthetic training data generated from diffusion
models, which improves generalization across unseen domains. While synthetic
data offers scalability, we identify artifacts introduced during data
generation as a key bottleneck affecting reconstruction quality. To address
this, we propose a token disentanglement process within the transformer
architecture, enhancing feature separation and ensuring more effective
learning. This refinement not only improves reconstruction quality over
standard transformers but also enables scalable training with synthetic data.
As a result, our method outperforms existing models on both in-dataset and
cross-dataset evaluations, achieving state-of-the-art results across multiple
benchmarks while significantly reducing computational costs. Project page:
https://scaling3dnvs.github.io/

### Human-Computer Interaction

### 1. [Context-Adaptive Hearing Aid Fitting Advisor through Multi-turn Multimodal LLM Conversation](http://arxiv.org/pdf/2509.06382v1)

Authors: Yingke Ding, Zeyu Wang, Xiyuxing Zhang, Hongbin Chen, Zhenan Xu

Traditional hearing aids often rely on static fittings that fail to adapt to
their dynamic acoustic environments. We propose CAFA, a Context-Adaptive
Fitting Advisor that provides personalized, real-time hearing aid adjustments
through a multi-agent Large Language Model (LLM) workflow. CAFA combines live
ambient audio, audiograms, and user feedback in a multi-turn conversational
system. Ambient sound is classified into conversation, noise, or quiet with
91.2\% accuracy using a lightweight neural network based on YAMNet embeddings.
This system utilizes a modular LLM workflow, comprising context acquisition,
subproblem classification, strategy provision, and ethical regulation, and is
overseen by an LLM Judge. The workflow translates context and feedback into
precise, safe tuning commands. Evaluation confirms that real-time sound
classification enhances conversational efficiency. CAFA exemplifies how
agentic, multimodal AI can enable intelligent, user-centric assistive
technologies.

### 2. [Talking to an AI Mirror: Designing Self-Clone Chatbots for Enhanced Engagement in Digital Mental Health Support](http://arxiv.org/pdf/2509.06393v1)

Authors: Mehrnoosh Sadat Shirvani, Jackie Liu, Thomas Chao, Suky Martinez, Laura Brandt, Ig-Jae Kim, Dongwook Yoon

Mental health conversational agents have the potential to deliver valuable
therapeutic impact, but low user engagement remains a critical barrier
hindering their efficacy. Existing therapeutic approaches have leveraged
clients' internal dialogues (e.g., journaling, talking to an empty chair) to
enhance engagement through accountable, self-sourced support. Inspired by
these, we designed novel AI-driven self-clone chatbots that replicate users'
support strategies and conversational patterns to improve therapeutic
engagement through externalized meaningful self-conversation. Validated through
a semi-controlled experiment (N=180), significantly higher emotional and
cognitive engagement was demonstrated with self-clone chatbots than a chatbot
with a generic counselor persona. Our findings highlight self-clone
believability as a mediator and emphasize the balance required in maintaining
convincing self-representation while creating positive interactions. This study
contributes to AI-based mental health interventions by introducing and
evaluating self-clones as a promising approach to increasing user engagement,
while exploring implications for their application in mental health care.

### 3. [Mapping Community Appeals Systems: Lessons for Community-led Moderation in Multi-Level Governance](http://arxiv.org/pdf/2509.06557v1)

Authors: Juhoon Lee, Bich Ngoc Doan, Jonghyun Jee, Joseph Seering

Platforms are increasingly adopting industrial models of moderation that
prioritize scalability and consistency, frequently at the expense of
context-sensitive and user-centered values. Building on the multi-level
governance framework that examines the interdependent relationship between
platforms and middle-level communities, we investigate community appeals
systems on Discord as a model for successful community-led governance. We
investigate how Discord servers operationalize appeal systems through a
qualitative interview study with focus groups and individual interviews with 17
community moderators. Our findings reveal a structured appeals process that
balances scalability, fairness, and accountability while upholding
community-centered values of growth and rehabilitation. Communities design
these processes to empower users, ensuring their voices are heard in moderation
decisions and fostering a sense of belonging. This research provides insights
into the practical implementation of community-led governance in a multi-level
governance framework, illustrating how communities can maintain their core
principles while integrating procedural fairness and tool-based design. We
discuss how platforms can gain insights from community-led moderation work to
motivate governance structures that effectively balance and align the interests
of multiple stakeholders.

### 4. [From Perception to Protection: A Developer-Centered Study of Security and Privacy Threats in Extended Reality (XR)](http://arxiv.org/pdf/2509.06368v1)

Authors: Kunlin Cai, Jinghuai Zhang, Ying Li, Zhiyuan Wang, Xun Chen, Tianshi Li, Yuan Tian

The immersive nature of XR introduces a fundamentally different set of
security and privacy (S&P) challenges due to the unprecedented user
interactions and data collection that traditional paradigms struggle to
mitigate. As the primary architects of XR applications, developers play a
critical role in addressing novel threats. However, to effectively support
developers, we must first understand how they perceive and respond to different
threats. Despite the growing importance of this issue, there is a lack of
in-depth, threat-aware studies that examine XR S&P from the developers'
perspective. To fill this gap, we interviewed 23 professional XR developers
with a focus on emerging threats in XR. Our study addresses two research
questions aiming to uncover existing problems in XR development and identify
actionable paths forward.
  By examining developers' perceptions of S&P threats, we found that: (1) XR
development decisions (e.g., rich sensor data collection, user-generated
content interfaces) are closely tied to and can amplify S&P threats, yet
developers are often unaware of these risks, resulting in cognitive biases in
threat perception; and (2) limitations in existing mitigation methods, combined
with insufficient strategic, technical, and communication support, undermine
developers' motivation, awareness, and ability to effectively address these
threats. Based on these findings, we propose actionable and stakeholder-aware
recommendations to improve XR S&P throughout the XR development process. This
work represents the first effort to undertake a threat-aware,
developer-centered study in the XR domain -- an area where the immersive,
data-rich nature of the XR technology introduces distinctive challenges.

### 5. [FireRedChat: A Pluggable, Full-Duplex Voice Interaction System with Cascaded and Semi-Cascaded Implementations](http://arxiv.org/pdf/2509.06502v1)

Authors: Junjie Chen, Yao Hu, Junjie Li, Kangyue Li, Kun Liu, Wenpeng Li, Xu Li, Ziyuan Li, Feiyu Shen, Xu Tang, Manzhen Wei, Yichen Wu, Fenglong Xie, Kaituo Xu, Kun Xie

Full-duplex voice interaction allows users and agents to speak simultaneously
with controllable barge-in, enabling lifelike assistants and customer service.
Existing solutions are either end-to-end, difficult to design and hard to
control, or modular pipelines governed by turn-taking controllers that ease
upgrades and per-module optimization; however, prior modular frameworks depend
on non-open components and external providers, limiting holistic optimization.
In this work, we present a complete, practical full-duplex voice interaction
system comprising a turn-taking controller, an interaction module, and a
dialogue manager. The controller integrates streaming personalized VAD (pVAD)
to suppress false barge-ins from noise and non-primary speakers, precisely
timestamp primary-speaker segments, and explicitly enable primary-speaker
barge-ins; a semantic end-of-turn detector improves stop decisions. It upgrades
heterogeneous half-duplex pipelines, cascaded, semi-cascaded, and
speech-to-speech, to full duplex. Using internal models, we implement cascaded
and semi-cascaded variants; the semi-cascaded one captures emotional and
paralinguistic cues, yields more coherent responses, lowers latency and error
propagation, and improves robustness. A dialogue manager extends capabilities
via tool invocation and context management. We also propose three system-level
metrics, barge-in, end-of-turn detection accuracy, and end-to-end latency, to
assess naturalness, control accuracy, and efficiency. Experiments show fewer
false interruptions, more accurate semantic ends, and lower latency approaching
industrial systems, enabling robust, natural, real-time full-duplex
interaction. Demos: https://fireredteam.github.io/demos/firered_chat.

### 6. [Co-Located VR with Hybrid SLAM-based HMD Tracking and Motion Capture Synchronization](http://arxiv.org/pdf/2509.06582v1)

Authors: Carlos A. Pinheiro de Sousa, Niklas Gröne, Mathias Günther, Oliver Deussen

We introduce a multi-user VR co-location framework that synchronizes users
within a shared virtual environment aligned to physical space. Our approach
combines a motion capture system with SLAM-based inside-out tracking to deliver
smooth, high-framerate, low-latency performance. Previous methods either rely
on continuous external tracking, which introduces latency and jitter, or on
one-time calibration, which cannot correct drift over time. In contrast, our
approach combines the responsiveness of local HMD SLAM tracking with the
flexibility to realign to an external source when needed. It also supports
real-time pose sharing across devices, ensuring consistent spatial alignment
and engagement between users. Our evaluation demonstrates that our framework
achieves the spatial accuracy required for natural multi-user interaction while
offering improved comfort, scalability, and robustness over existing co-located
VR solutions.

### 7. [Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting](http://arxiv.org/pdf/2509.06770v1)

Authors: Shashidhar Reddy Javaji, Bhavul Gauri, Zining Zhu

Large language models (LLMs) are now used in multi-turn workflows, but we
still lack a clear way to measure when iteration helps and when it hurts. We
present an evaluation framework for iterative refinement that spans ideation,
code, and math. Our protocol runs controlled 12-turn conversations per task,
utilizing a variety of prompts ranging from vague ``improve it'' feedback to
targeted steering, and logs per-turn outputs. We score outcomes with
domain-appropriate checks (unit tests for code; answer-equivalence plus
reasoning-soundness for math; originality and feasibility for ideation) and
track turn-level behavior with three families of metrics: semantic movement
across turns, turn-to-turn change, and output size growth. Across models and
tasks, gains are domain-dependent: they arrive early in ideas and code, but in
math late turns matter when guided by elaboration. After the first few turns,
vague feedback often plateaus or reverses correctness, while targeted prompts
reliably shift the intended quality axis (novelty vs. feasibility in ideation;
speed vs. readability in code; in math, elaboration outperforms exploration and
drives late-turn gains). We also observe consistent domain patterns: ideation
moves more in meaning across turns, code tends to grow in size with little
semantic change, and math starts fixed but can break that path with late,
elaborative iteration.Together, the framework and metrics make iteration
measurable and comparable across models, and signal when to steer, stop, or
switch strategies.

### 8. [Hue4U: Real-Time Personalized Color Correction in Augmented Reality](http://arxiv.org/pdf/2509.06776v1)

Authors: Jingwen Qin, Semen Checherin, Yue Li, Berend-Jan van der Zwaag, Özlem Durmaz-Incel

Color Vision Deficiency (CVD) affects nearly 8 percent of men and 0.5 percent
of women worldwide. Existing color-correction methods often rely on prior
clinical diagnosis and static filtering, making them less effective for users
with mild or moderate CVD. In this paper, we introduce Hue4U, a personalized,
real-time color-correction system in augmented reality using consumer-grade
Meta Quest headsets. Unlike previous methods, Hue4U requires no prior medical
diagnosis and adapts to the user in real time. A user study with 10
participants showed notable improvements in their ability to distinguish
colors. The results demonstrated large effect sizes (Cohen's d > 1.4),
suggesting clinically meaningful gains for individuals with CVD. These findings
highlight the potential of personalized AR interventions to improve visual
accessibility and quality of life for people affected by CVD.

### 9. ["It was Tragic": Exploring the Impact of a Robot's Shutdown](http://arxiv.org/pdf/2509.06934v1)

Authors: Agam Oberlender, Hadas Erel

It is well established that people perceive robots as social entities, even
when they are not designed for social interaction. We evaluated whether the
social interpretation of robotic gestures should also be considered when
turning off a robot. In the experiment, participants engaged in a brief
preliminary neutral interaction while a robotic arm showed interest in their
actions. At the end of the task, participants were asked to turn off the
robotic arm under two conditions: (1) a Non-designed condition, where all of
the robot's engines were immediately and simultaneously turned off, as robots
typically shut down; (2) a Designed condition, where the robot's engines
gradually folded inward in a motion resembling "falling asleep." Our findings
revealed that all participants anthropomorphized the robot's movement when it
was turned off. In the Non-designed condition, most participants interpreted
the robot's turn-off movement negatively, as if the robot had "died." In the
Designed condition, most participants interpreted it more neutrally, stating
that the robot "went to sleep." The robot's turn-off movement also impacted its
perception, leading to higher likeability, perceived intelligence, and animacy
in the Designed condition. We conclude that the impact of common edge
interactions, such as turning off a robot, should be carefully designed while
considering people's automatic tendency to perceive robots as social entities.

### 10. [Explained, yet misunderstood: How AI Literacy shapes HR Managers' interpretation of User Interfaces in Recruiting Recommender Systems](http://arxiv.org/pdf/2509.06475v1)

Authors: Yannick Kalff, Katharina Simbeck

AI-based recommender systems increasingly influence recruitment decisions.
Thus, transparency and responsible adoption in Human Resource Management (HRM)
are critical. This study examines how HR managers' AI literacy influences their
subjective perception and objective understanding of explainable AI (XAI)
elements in recruiting recommender dashboards. In an online experiment, 410
German-based HR managers compared baseline dashboards to versions enriched with
three XAI styles: important features, counterfactuals, and model criteria. Our
results show that the dashboards used in practice do not explain AI results and
even keep AI elements opaque. However, while adding XAI features improves
subjective perceptions of helpfulness and trust among users with moderate or
high AI literacy, it does not increase their objective understanding. It may
even reduce accurate understanding, especially with complex explanations. Only
overlays of important features significantly aided the interpretations of
high-literacy users. Our findings highlight that the benefits of XAI in
recruitment depend on users' AI literacy, emphasizing the need for tailored
explanation strategies and targeted literacy training in HRM to ensure fair,
transparent, and effective adoption of AI.

### Information Retrieval

### 1. [Rethinking LLM Parametric Knowledge as Post-retrieval Confidence for Dynamic Retrieval and Reranking](http://arxiv.org/pdf/2509.06472v1)

Authors: Haoxiang Jin, Ronghan Li, Qiguang Miao, Zixiang Lu

Large Language Models (LLMs) often generate inaccurate responses
(hallucinations) when faced with questions beyond their knowledge scope.
Retrieval-Augmented Generation (RAG) addresses this by leveraging external
knowledge, but a critical challenge remains: determining whether retrieved
contexts effectively enhance the model`s ability to answer specific queries.
This challenge underscores the importance of knowledge boundary awareness,
which current methods-relying on discrete labels or limited signals-fail to
address adequately, as they overlook the rich information in LLMs` continuous
internal hidden states. To tackle this, we propose a novel post-retrieval
knowledge filtering approach. First, we construct a confidence detection model
based on LLMs` internal hidden states to quantify how retrieved contexts
enhance the model`s confidence. Using this model, we build a preference dataset
(NQ_Rerank) to fine-tune a reranker, enabling it to prioritize contexts
preferred by the downstream LLM during reranking. Additionally, we introduce
Confidence-Based Dynamic Retrieval (CBDR), which adaptively triggers retrieval
based on the LLM`s initial confidence in the original question, reducing
knowledge conflicts and improving efficiency. Experimental results demonstrate
significant improvements in accuracy for context screening and end-to-end RAG
performance, along with a notable reduction in retrieval costs while
maintaining competitive accuracy.

### 2. [Reasoning-enhanced Query Understanding through Decomposition and Interpretation](http://arxiv.org/pdf/2509.06544v1)

Authors: Yunfei Zhong, Jun Yang, Yixing Fan, Jiafeng Guo, Lixin Su, Maarten de Rijke, Ruqing Zhang, Dawei Yin, Xueqi Cheng

Accurate inference of user intent is crucial for enhancing document retrieval
in modern search engines. While large language models (LLMs) have made
significant strides in this area, their effectiveness has predominantly been
assessed with short, keyword-based queries. As AI-driven search evolves,
long-form queries with intricate intents are becoming more prevalent, yet they
remain underexplored in the context of LLM-based query understanding (QU). To
bridge this gap, we introduce ReDI: a Reasoning-enhanced approach for query
understanding through Decomposition and Interpretation. ReDI leverages the
reasoning and comprehension capabilities of LLMs in a three-stage pipeline: (i)
it breaks down complex queries into targeted sub-queries to accurately capture
user intent; (ii) it enriches each sub-query with detailed semantic
interpretations to improve the query-document matching; and (iii) it
independently retrieves documents for each sub-query and employs a fusion
strategy to aggregate the results for the final ranking. We compiled a
large-scale dataset of real-world complex queries from a major search engine
and distilled the query understanding capabilities of teacher models into
smaller models for practical application. Experiments on BRIGHT and BEIR
demonstrate that ReDI consistently surpasses strong baselines in both sparse
and dense retrieval paradigms, affirming its effectiveness.

### 3. [UniSearch: Rethinking Search System with a Unified Generative Architecture](http://arxiv.org/pdf/2509.06887v1)

Authors: Jiahui Chen, Xiaoze Jiang, Zhibo Wang, Quanzhi Zhu, Junyao Zhao, Feng Hu, Kang Pan, Ao Xie, Maohua Pei, Zhiheng Qin, Hongjing Zhang, Zhixin Zhai, Xiaobo Guo, Runbin Zhou, Kefeng Wang, Mingyang Geng, Cheng Chen, Jingshan Lv, Yupeng Huang, Xiao Liang, Han Li

Modern search systems play a crucial role in facilitating information
acquisition. Traditional search engines typically rely on a cascaded
architecture, where results are retrieved through recall, pre-ranking, and
ranking stages. The complexity of designing and maintaining multiple modules
makes it difficult to achieve holistic performance gains. Recent advances in
generative recommendation have motivated the exploration of unified generative
search as an alternative. However, existing approaches are not genuinely
end-to-end: they typically train an item encoder to tokenize candidates first
and then optimize a generator separately, leading to objective inconsistency
and limited generalization. To address these limitations, we propose UniSearch,
a unified generative search framework for Kuaishou Search. UniSearch replaces
the cascaded pipeline with an end-to-end architecture that integrates a Search
Generator and a Video Encoder. The Generator produces semantic identifiers of
relevant items given a user query, while the Video Encoder learns latent item
embeddings and provides their tokenized representations. A unified training
framework jointly optimizes both components, enabling mutual enhancement and
improving representation quality and generation accuracy. Furthermore, we
introduce Search Preference Optimization (SPO), which leverages a reward model
and real user feedback to better align generation with user preferences.
Extensive experiments on industrial-scale datasets, together with online A/B
testing in both short-video and live search scenarios, demonstrate the strong
effectiveness and deployment potential of UniSearch. Notably, its deployment in
live search yields the largest single-experiment improvement in recent years of
our product's history, highlighting its practical value for real-world
applications.

### 4. [Compare: A Framework for Scientific Comparisons](http://arxiv.org/pdf/2509.06412v1)

Authors: Moritz Staudinger, Wojciech Kusa, Matteo Cancellieri, David Pride, Petr Knoth, Allan Hanbury

Navigating the vast and rapidly increasing sea of academic publications to
identify institutional synergies, benchmark research contributions and pinpoint
key research contributions has become an increasingly daunting task, especially
with the current exponential increase in new publications. Existing tools
provide useful overviews or single-document insights, but none supports
structured, qualitative comparisons across institutions or publications.
  To address this, we demonstrate Compare, a novel framework that tackles this
challenge by enabling sophisticated long-context comparisons of scientific
contributions. Compare empowers users to explore and analyze research overlaps
and differences at both the institutional and publication granularity, all
driven by user-defined questions and automatic retrieval over online resources.
For this we leverage on Retrieval-Augmented Generation over evolving data
sources to foster long context knowledge synthesis. Unlike traditional
scientometric tools, Compare goes beyond quantitative indicators by providing
qualitative, citation-supported comparisons.

### 5. [AudioBoost: Increasing Audiobook Retrievability in Spotify Search with Synthetic Query Generation](http://arxiv.org/pdf/2509.06452v1)

Authors: Enrico Palumbo, Gustavo Penha, Alva Liu, Marcus Eltscheminov, Jefferson Carvalho dos Santos, Alice Wang, Hugues Bouchard, Humberto Jesús Corona Pampin, Michelle Tran Luu

Spotify has recently introduced audiobooks as part of its catalog,
complementing its music and podcast offering. Search is often the first entry
point for users to access new items, and an important goal for Spotify is to
support users in the exploration of the audiobook catalog. More specifically,
we would like to enable users without a specific item in mind to broadly search
by topic, genre, story tropes, decade, and discover audiobooks, authors and
publishers they may like. To do this, we need to 1) inspire users to type more
exploratory queries for audiobooks and 2) augment our retrieval systems to
better deal with exploratory audiobook queries. This is challenging in a
cold-start scenario, where we have a retrievabiliy bias due to the little
amount of user interactions with audiobooks compared to previously available
items such as music and podcast content. To address this, we propose
AudioBoost, a system to boost audiobook retrievability in Spotify's Search via
synthetic query generation. AudioBoost leverages Large Language Models (LLMs)
to generate synthetic queries conditioned on audiobook metadata. The synthetic
queries are indexed both in the Query AutoComplete (QAC) and in the Search
Retrieval engine to improve query formulation and retrieval at the same time.
We show through offline evaluation that synthetic queries increase
retrievability and are of high quality. Moreover, results from an online A/B
test show that AudioBoost leads to a +0.7% in audiobook impressions, +1.22% in
audiobook clicks, and +1.82% in audiobook exploratory query completions.

### 6. [Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval](http://arxiv.org/pdf/2509.06650v1)

Authors: Hao Lin, Peitong Xie, Jingxue Chen, Jie Lin, Qingkun Tang, Qianchun Lu

Retrieval-Augmented Generation (RAG) systems rely heavily on the retrieval
stage, particularly the coarse-ranking process. Existing coarse-ranking
optimization approaches often struggle to balance domain-specific knowledge
learning with query enhencement, resulting in suboptimal retrieval performance.
To address this challenge, we propose MoLER, a domain-aware RAG method that
uses MoL-Enhanced Reinforcement Learning to optimize retrieval. MoLER has a
two-stage pipeline: a continual pre-training (CPT) phase using a Mixture of
Losses (MoL) to balance domain-specific knowledge with general language
capabilities, and a reinforcement learning (RL) phase leveraging Group Relative
Policy Optimization (GRPO) to optimize query and passage generation for
maximizing document recall. A key innovation is our Multi-query Single-passage
Late Fusion (MSLF) strategy, which reduces computational overhead during RL
training while maintaining scalable inference via Multi-query Multi-passage
Late Fusion (MMLF). Extensive experiments on benchmark datasets show that MoLER
achieves state-of-the-art performance, significantly outperforming baseline
methods. MoLER bridges the knowledge gap in RAG systems, enabling robust and
scalable retrieval in specialized domains.

### 7. [Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing](http://arxiv.org/pdf/2509.06552v1)

Authors: Zheqi Lv, Wenqiao Zhang, Kairui Fu, Qi Tian, Shengyu Zhang, Jiajie Su, Jingyuan Chen, Kun Kuang, Fei Wu

The on-device real-time data distribution shift on devices challenges the
generalization of lightweight on-device models. This critical issue is often
overlooked in current research, which predominantly relies on data-intensive
and computationally expensive fine-tuning approaches. To tackle this, we
introduce Persona, a novel personalized method using a prototype-based,
backpropagation-free parameter editing framework to enhance model
generalization without post-deployment retraining. Persona employs a neural
adapter in the cloud to generate a parameter editing matrix based on real-time
device data. This matrix adeptly adapts on-device models to the prevailing data
distributions, efficiently clustering them into prototype models. The
prototypes are dynamically refined via the parameter editing matrix,
facilitating efficient evolution. Furthermore, the integration of cross-layer
knowledge transfer ensures consistent and context-aware multi-layer parameter
changes and prototype assignment. Extensive experiments on vision task and
recommendation task on multiple datasets confirm Persona's effectiveness and
generality.

### 8. [Unveiling the Listener Structure Underlying K-pop's Global Success: A Large-Scale Listening Data Analysis](http://arxiv.org/pdf/2509.06606v1)

Authors: Ryota Nakamura, Keita Nishimoto, Ichiro Sakata, Kimitaka Asatani

From the mid-2000s to the 2010s, K-pop moved beyond its status as a
regionally popular genre in Asia and established itself as a global music genre
with enthusiastic fans around the world. However, little is known about how the
vast number of music listeners across the globe have listened to and perceived
K-pop. This study addresses this question by analyzing a large-scale listening
dataset from Last.fm. An analysis of the distribution of play counts reveals
that K-pop experienced a significant increase in plays between 2005 and 2019,
largely supported by a small group of heavy listeners. The Gini coefficient in
play counts is notably greater than that of existing mainstream genres and
other growing niche genres. Furthermore, an analysis based on user-assigned
genre tags quantitatively demonstrates that between 2005 and 2010, K-pop shed
its status as a local Asian genre and established itself as a distinct music
genre in its own right.

### 9. [UNH at CheckThat! 2025: Fine-tuning Vs Prompting in Claim Extraction](http://arxiv.org/pdf/2509.06883v1)

Authors: Joe Wilder, Nikhil Kadapala, Benji Xu, Mohammed Alsaadi, Aiden Parsons, Mitchell Rogers, Palash Agarwal, Adam Hassick, Laura Dietz

We participate in CheckThat! Task 2 English and explore various methods of
prompting and in-context learning, including few-shot prompting and fine-tuning
with different LLM families, with the goal of extracting check-worthy claims
from social media passages. Our best METEOR score is achieved by fine-tuning a
FLAN-T5 model. However, we observe that higher-quality claims can sometimes be
extracted using other methods, even when their METEOR scores are lower.

### 10. [mmBERT: A Modern Multilingual Encoder with Annealed Language Learning](http://arxiv.org/pdf/2509.06888v1)

Authors: Marc Marone, Orion Weller, William Fleshman, Eugene Yang, Dawn Lawrie, Benjamin Van Durme

Encoder-only languages models are frequently used for a variety of standard
machine learning tasks, including classification and retrieval. However, there
has been a lack of recent research for encoder models, especially with respect
to multilingual models. We introduce mmBERT, an encoder-only language model
pretrained on 3T tokens of multilingual text in over 1800 languages. To build
mmBERT we introduce several novel elements, including an inverse mask ratio
schedule and an inverse temperature sampling ratio. We add over 1700
low-resource languages to the data mix only during the decay phase, showing
that it boosts performance dramatically and maximizes the gains from the
relatively small amount of training data. Despite only including these
low-resource languages in the short decay phase we achieve similar
classification performance to models like OpenAI's o3 and Google's Gemini 2.5
Pro. Overall, we show that mmBERT significantly outperforms the previous
generation of models on classification and retrieval tasks -- on both high and
low-resource languages.

### Machine Learning

### 1. [IPR: Intelligent Prompt Routing with User-Controlled Quality-Cost Trade-offs](http://arxiv.org/pdf/2509.06274v1)

Authors: Aosong Feng, Zhichao Xu, Xian Wu, Kang Zhou, Sheng Guan, Yueyan Chen, Ninad Kulkarni, Yun Zhou, Balasubramaniam Srinivasan, Haibo Ding, Lin Lee Cheong

Routing incoming queries to the most cost-effective LLM while maintaining
response quality poses a fundamental challenge in optimizing performance-cost
trade-offs for large-scale commercial systems. We present IPR\, a
quality-constrained Intelligent Prompt Routing framework that dynamically
selects optimal models based on predicted response quality and user-specified
tolerance levels. IPR introduces three key innovations: (1) a modular
architecture with lightweight quality estimators trained on 1.5M prompts
annotated with calibrated quality scores, enabling fine-grained quality
prediction across model families; (2) a user-controlled routing mechanism with
tolerance parameter $\tau \in [0,1]$ that provides explicit control over
quality-cost trade-offs; and (3) an extensible design using frozen encoders
with model-specific adapters, reducing new model integration from days to
hours. To rigorously train and evaluate IPR, we curate an industrial-level
dataset IPRBench\footnote{IPRBench will be released upon legal approval.}, a
comprehensive benchmark containing 1.5 million examples with response quality
annotations across 11 LLM candidates. Deployed on a major cloud platform, IPR
achieves 43.9\% cost reduction while maintaining quality parity with the
strongest model in the Claude family and processes requests with sub-150ms
latency.

### 2. [RecMind: LLM-Enhanced Graph Neural Networks for Personalized Consumer Recommendations](http://arxiv.org/pdf/2509.06286v1)

Authors: Chang Xue, Youwei Lu, Chen Yang, Jinming Xing

Personalization is a core capability across consumer technologies, streaming,
shopping, wearables, and voice, yet it remains challenged by sparse
interactions, fast content churn, and heterogeneous textual signals. We present
RecMind, an LLM-enhanced graph recommender that treats the language model as a
preference prior rather than a monolithic ranker. A frozen LLM equipped with
lightweight adapters produces text-conditioned user/item embeddings from
titles, attributes, and reviews; a LightGCN backbone learns collaborative
embeddings from the user-item graph. We align the two views with a symmetric
contrastive objective and fuse them via intra-layer gating, allowing language
to dominate in cold/long-tail regimes and graph structure to stabilize rankings
elsewhere. On Yelp and Amazon-Electronics, RecMind attains the best results on
all eight reported metrics, with relative improvements up to +4.53\%
(Recall@40) and +4.01\% (NDCG@40) over strong baselines. Ablations confirm both
the necessity of cross-view alignment and the advantage of gating over late
fusion and LLM-only variants.

### 3. [LoaQ: Layer-wise Output Approximation Quantization](http://arxiv.org/pdf/2509.06297v1)

Authors: Li Lin, Xiaojun Wan

A natural and intuitive idea in model quantization is to approximate each
component's quantized output to match its original. Layer-wise post-training
quantization (PTQ), though based on this idea, adopts a strictly local view and
can achieve, at best, only activation-aware approximations of weights. As a
result, it often leads to insufficient approximations and practical deviations
from this guiding intuition. Recent work has achieved a more accurate
approximation of linear-layer outputs within the framework of layer-wise PTQ,
but such refinements remain inadequate for achieving alignment with the full
model output. Based on a deeper understanding of the structural characteristics
of mainstream LLMs, we propose $LoaQ$, an output-approximation method for
layer-wise PTQ that explicitly targets output-level consistency. It better
aligns with this intuition and can feature a simple closed-form solution,
making it orthogonal to existing techniques and readily integrable into
existing quantization pipelines. Experiments on the LLaMA and Qwen model
families demonstrate that LoaQ performs effectively in both weight-only and
weight-activation joint quantization. By integrating seamlessly with existing
quantization strategies, it further enhances overall quantization quality and
shows strong potential to advance the frontier of post-training quantization.

### 4. [WindFM: An Open-Source Foundation Model for Zero-Shot Wind Power Forecasting](http://arxiv.org/pdf/2509.06311v1)

Authors: Hang Fan, Yu Shi, Zongliang Fu, Shuo Chen, Wei Wei, Wei Xu, Jian Li

High-quality wind power forecasting is crucial for the operation of modern
power grids. However, prevailing data-driven paradigms either train a
site-specific model which cannot generalize to other locations or rely on
fine-tuning of general-purpose time series foundation models which are
difficult to incorporate domain-specific data in the energy sector. This paper
introduces WindFM, a lightweight and generative Foundation Model designed
specifically for probabilistic wind power forecasting. WindFM employs a
discretize-and-generate framework. A specialized time-series tokenizer first
converts continuous multivariate observations into discrete, hierarchical
tokens. Subsequently, a decoder-only Transformer learns a universal
representation of wind generation dynamics by autoregressively pre-training on
these token sequences. Using the comprehensive WIND Toolkit dataset comprising
approximately 150 billion time steps from more than 126,000 sites, WindFM
develops a foundational understanding of the complex interplay between
atmospheric conditions and power output. Extensive experiments demonstrate that
our compact 8.1M parameter model achieves state-of-the-art zero-shot
performance on both deterministic and probabilistic tasks, outperforming
specialized models and larger foundation models without any fine-tuning. In
particular, WindFM exhibits strong adaptiveness under out-of-distribution data
from a different continent, demonstrating the robustness and transferability of
its learned representations. Our pre-trained model is publicly available at
https://github.com/shiyu-coder/WindFM.

### 5. [Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics](http://arxiv.org/pdf/2509.06322v1)

Authors: Jiajun Bao, Nicolas Boullé, Toni J. B. Liu, Raphaël Sarfati, Christopher J. Earls

Large language models (LLMs) have demonstrated emergent in-context learning
(ICL) capabilities across a range of tasks, including zero-shot time-series
forecasting. We show that text-trained foundation models can accurately
extrapolate spatiotemporal dynamics from discretized partial differential
equation (PDE) solutions without fine-tuning or natural language prompting.
Predictive accuracy improves with longer temporal contexts but degrades at
finer spatial discretizations. In multi-step rollouts, where the model
recursively predicts future spatial states over multiple time steps, errors
grow algebraically with the time horizon, reminiscent of global error
accumulation in classical finite-difference solvers. We interpret these trends
as in-context neural scaling laws, where prediction quality varies predictably
with both context length and output length. To better understand how LLMs are
able to internally process PDE solutions so as to accurately roll them out, we
analyze token-level output distributions and uncover a consistent ICL
progression: beginning with syntactic pattern imitation, transitioning through
an exploratory high-entropy phase, and culminating in confident, numerically
grounded predictions.

### 6. [Exploring approaches to computational representation and classification of user-generated meal logs](http://arxiv.org/pdf/2509.06330v1)

Authors: Guanlan Hu, Adit Anand, Pooja M. Desai, Iñigo Urteaga, Lena Mamykina

This study examined the use of machine learning and domain specific
enrichment on patient generated health data, in the form of free text meal
logs, to classify meals on alignment with different nutritional goals. We used
a dataset of over 3000 meal records collected by 114 individuals from a
diverse, low income community in a major US city using a mobile app. Registered
dietitians provided expert judgement for meal to goal alignment, used as gold
standard for evaluation. Using text embeddings, including TFIDF and BERT, and
domain specific enrichment information, including ontologies, ingredient
parsers, and macronutrient contents as inputs, we evaluated the performance of
logistic regression and multilayer perceptron classifiers using accuracy,
precision, recall, and F1 score against the gold standard and self assessment.
Even without enrichment, ML outperformed self assessments of individuals who
logged meals, and the best performing combination of ML classifier with
enrichment achieved even higher accuracies. In general, ML classifiers with
enrichment of Parsed Ingredients, Food Entities, and Macronutrients information
performed well across multiple nutritional goals, but there was variability in
the impact of enrichment and classification algorithm on accuracy of
classification for different nutritional goals. In conclusion, ML can utilize
unstructured free text meal logs and reliably classify whether meals align with
specific nutritional goals, exceeding self assessments, especially when
incorporating nutrition domain knowledge. Our findings highlight the potential
of ML analysis of patient generated health data to support patient centered
nutrition guidance in precision healthcare.

### 7. [Breaking SafetyCore: Exploring the Risks of On-Device AI Deployment](http://arxiv.org/pdf/2509.06371v1)

Authors: Victor Guyomard, Mathis Mauvisseau, Marie Paindavoine

Due to hardware and software improvements, an increasing number of AI models
are deployed on-device. This shift enhances privacy and reduces latency, but
also introduces security risks distinct from traditional software. In this
article, we examine these risks through the real-world case study of
SafetyCore, an Android system service incorporating sensitive image content
detection. We demonstrate how the on-device AI model can be extracted and
manipulated to bypass detection, effectively rendering the protection
ineffective. Our analysis exposes vulnerabilities of on-device AI models and
provides a practical demonstration of how adversaries can exploit them.

### 8. [Lane Change Intention Prediction of two distinct Populations using a Transformer](http://arxiv.org/pdf/2509.06529v1)

Authors: Francesco De Cristofaro, Cornelia Lex, Jia Hu, Arno Eichberger

As a result of the growing importance of lane change intention prediction for
a safe and efficient driving experience in complex driving scenarios,
researchers have in recent years started to train novel machine learning
algorithms on available datasets with promising results. A shortcoming of this
recent research effort, though, is that the vast majority of the proposed
algorithms are trained on a single datasets. In doing so, researchers failed to
test if their algorithm would be as effective if tested on a different dataset
and, by extension, on a different population with respect to the one on which
they were trained. In this article we test a transformer designed for lane
change intention prediction on two datasets collected by LevelX in Germany and
Hong Kong. We found that the transformer's accuracy plummeted when tested on a
population different to the one it was trained on with accuracy values as low
as 39.43%, but that when trained on both populations simultaneously it could
achieve an accuracy as high as 86.71%. - This work has been submitted to the
IEEE for possible publication. Copyright may be transferred without notice,
after which this version may no longer be accessible.

### 9. [Predicting Fetal Outcomes from Cardiotocography Signals Using a Supervised Variational Autoencoder](http://arxiv.org/pdf/2509.06540v1)

Authors: John Tolladay, Beth Albert, Gabriel Davis Jones

Objective: To develop and interpret a supervised variational autoencoder
(VAE) model for classifying cardiotocography (CTG) signals based on pregnancy
outcomes, addressing interpretability limits of current deep learning
approaches. Methods: The OxMat CTG dataset was used to train a VAE on
five-minute fetal heart rate (FHR) segments, labeled with postnatal outcomes.
The model was optimised for signal reconstruction and outcome prediction,
incorporating Kullback-Leibler divergence and total correlation (TC)
constraints to structure the latent space. Performance was evaluated using area
under the receiver operating characteristic curve (AUROC) and mean squared
error (MSE). Interpretability was assessed using coefficient of determination,
latent traversals and unsupervised component analyses. Results: The model
achieved an AUROC of 0.752 at the segment level and 0.779 at the CTG level,
where predicted scores were aggregated. Relaxing TC constraints improved both
reconstruction and classification. Latent analysis showed that baseline-related
features (e.g., FHR baseline, baseline shift) were well represented and aligned
with model scores, while metrics like short- and long-term variability were
less strongly encoded. Traversals revealed clear signal changes for baseline
features, while other properties were entangled or subtle. Unsupervised
decompositions corroborated these patterns. Findings: This work demonstrates
that supervised VAEs can achieve competitive fetal outcome prediction while
partially encoding clinically meaningful CTG features. The irregular,
multi-timescale nature of FHR signals poses challenges for disentangling
physiological components, distinguishing CTG from more periodic signals such as
ECG. Although full interpretability was not achieved, the model supports
clinically useful outcome prediction and provides a basis for future
interpretable, generative models.

### 10. [PAC-Bayesian Generalization Bounds for Graph Convolutional Networks on Inductive Node Classification](http://arxiv.org/pdf/2509.06600v1)

Authors: Huayi Tang, Yong Liu

Graph neural networks (GNNs) have achieved remarkable success in processing
graph-structured data across various applications. A critical aspect of
real-world graphs is their dynamic nature, where new nodes are continually
added and existing connections may change over time. Previous theoretical
studies, largely based on the transductive learning framework, fail to
adequately model such temporal evolution and structural dynamics. In this
paper, we presents a PAC-Bayesian theoretical analysis of graph convolutional
networks (GCNs) for inductive node classification, treating nodes as dependent
and non-identically distributed data points. We derive novel generalization
bounds for one-layer GCNs that explicitly incorporate the effects of data
dependency and non-stationarity, and establish sufficient conditions under
which the generalization gap converges to zero as the number of nodes
increases. Furthermore, we extend our analysis to two-layer GCNs, and reveal
that it requires stronger assumptions on graph topology to guarantee
convergence. This work establishes a theoretical foundation for understanding
and improving GNN generalization in dynamic graph environments.

### Neural and Evolutionary Computing

### 1. [Full Integer Arithmetic Online Training for Spiking Neural Networks](http://arxiv.org/pdf/2509.06636v1)

Authors: Ismael Gomez, Guangzhi Tang

Spiking Neural Networks (SNNs) are promising for neuromorphic computing due
to their biological plausibility and energy efficiency. However, training
methods like Backpropagation Through Time (BPTT) and Real Time Recurrent
Learning (RTRL) remain computationally intensive. This work introduces an
integer-only, online training algorithm using a mixed-precision approach to
improve efficiency and reduce memory usage by over 60%. The method replaces
floating-point operations with integer arithmetic to enable hardware-friendly
implementation. It generalizes to Convolutional and Recurrent SNNs (CSNNs,
RSNNs), showing versatility across architectures. Evaluations on MNIST and the
Spiking Heidelberg Digits (SHD) dataset demonstrate that mixed-precision models
achieve accuracy comparable to or better than full-precision baselines using
16-bit shadow and 8- or 12-bit inference weights. Despite some limitations in
low-precision and deeper models, performance remains robust. In conclusion, the
proposed integer-only online learning algorithm presents an effective solution
for efficiently training SNNs, enabling deployment on resource-constrained
neuromorphic hardware without sacrificing accuracy.

### 2. [An Explainable Framework for Particle Swarm Optimization using Landscape Analysis and Machine Learning](http://arxiv.org/pdf/2509.06272v1)

Authors: Nitin Gupta, Bapi Dutta, Anupam Yadav

Swarm intelligence algorithms have demonstrated remarkable success in solving
complex optimization problems across diverse domains. However, their widespread
adoption is often hindered by limited transparency in how algorithmic
components influence performance. This work presents a multi-faceted
investigation of Particle Swarm Optimization (PSO) to further understand the
key role of different topologies for better interpretability and
explainability. To achieve this objective, we first develop a comprehensive
landscape characterization framework using Exploratory Landscape Analysis (ELA)
to quantify problem difficulty and identify critical features affecting the
optimization performance of PSO. Next, we conduct a rigorous empirical study
comparing three fundamental swarm communication architectures -- Ring, Star,
and Von Neumann topologies -- analysing their distinct impacts on
exploration-exploitation balance, convergence behaviour, and solution quality
and eventually develop an explainable benchmarking framework for PSO, to decode
how swarm topologies affects information flow, diversity, and convergence.
Based on this, a novel machine learning approach for automated algorithm
configuration is introduced for training predictive models on extensive Area
over the Convergence Curve (AOCC) data to recommend optimal settings based on
problem characteristics. Through systematic experimentation across twenty four
benchmark functions in multiple dimensions, we establish practical guidelines
for topology selection and parameter configuration. These findings advance the
development of more transparent and reliable swarm intelligence systems. The
source codes of this work can be accessed at
https://github.com/GitNitin02/ioh_pso.

### 3. [Approximating Condorcet Ordering for Vector-valued Mathematical Morphology](http://arxiv.org/pdf/2509.06577v1)

Authors: Marcos Eduardo Valle, Santiago Velasco-Forero, Joao Batista Florindo, Gustavo Jesus Angulo

Mathematical morphology provides a nonlinear framework for image and spatial
data processing and analysis. Although there have been many successful
applications of mathematical morphology to vector-valued images, such as color
and hyperspectral images, there is still no consensus on the most suitable
vector ordering for constructing morphological operators. This paper addresses
this issue by examining a reduced ordering approximating the Condorcet ranking
derived from a set of vector orderings. Inspired by voting problems, the
Condorcet ordering ranks elements from most to least voted, with voters
representing different orderings. In this paper, we develop a machine learning
approach that learns a reduced ordering that approximates the Condorcet
ordering. Preliminary computational experiments confirm the effectiveness of
learning the reduced mapping to define vector-valued morphological operators
for color images.

### Networking and Internet Architecture

### 1. [Network-Aware Control of AGVs in an Industrial Scenario: A Simulation Study Based on ROS 2 and Gazebo](http://arxiv.org/pdf/2509.06451v1)

Authors: Filippo Bragato, Tullia Fontana, Marco Giordani, Malte Schellmann, Josef Eichinger, Michele Zorzi

Networked Control System (NCS) is a paradigm where sensors, controllers, and
actuators communicate over a shared network. One promising application of NCS
is the control of Automated Guided Vehicles (AGVs) in the industrial
environment, for example to transport goods efficiently and to autonomously
follow predefined paths or routes. In this context, communication and control
are tightly correlated, a paradigm referred to as Joint Communication and
Control (JCC), since network issues such as delays or errors can lead to
significant deviations of the AGVs from the planned trajectory. In this paper,
we present a simulation framework based on Gazebo and Robot Operating System 2
(ROS 2) to simulate and visualize, respectively, the complex interaction
between the control of AGVs and the underlying communication network. This
framework explicitly incorporates communication metrics, such as delay and
packet loss, and control metrics, especially the Mean Squared Error (MSE)
between the optimal/desired and actual path of the AGV in response to driving
commands. Our results shed light into the correlation between the network
performance, particularly Packet Reception Ratio (PRR), and accuracy of
control.

### 2. [Empirical Evaluation of a 5G Transparent Clock for Time Synchronization in a TSN-5G Network](http://arxiv.org/pdf/2509.06454v1)

Authors: Julia Caleya-Sanchez, Pablo Muñoz, Jorge Sánchez-Garrido, Emilio Florentín, Felix Delgado-Ferro, Pablo Rodriguez-Martin, Pablo Ameigeiras

Time synchronization is essential for industrial IoT and Industry 4.0/5.0
applications, but achieving high synchronization accuracy in Time-Sensitive
Networking (TSN)-5G networks is challenging due to jitter and asymmetric
delays. 3GPP TS 23.501 defines three 5G synchronization modes: time-aware
system, boundary clock (BC), and transparent clock (TC), where TC offers a
promising solution. However, to the best of our knowledge, there is no
empirical evaluation of TC in a TSN-5G network. This paper empirically
evaluates an 5G end-to-end TC in a TSN-5G network, implemented on commercial
TSN switches with a single clock. For TC development, we compute the residence
time in 5G and recover the clock domain at the slave node. We deploy a TSN-5G
testbed with commercial equipment for synchronization evaluation by modifying
the Precision Timing Protocol (PTP) message transmission rates. Experimental
results show a peak-to-peak synchronization of 500 ns, meeting the industrial
requirement of < 1 us, with minimal synchronization offsets for specific PTP
message transmission rates.

### 3. [Five Blind Men and the Internet: Towards an Understanding of Internet Traffic](http://arxiv.org/pdf/2509.06515v1)

Authors: Ege Cem Kirci, Ayush Mishra, Laurent Vanbever

The Internet, the world's largest and most pervasive network, lacks a
transparent, granular view of its traffic patterns, volumes, and growth trends,
hindering the networking community's understanding of its dynamics. This paper
leverages publicly available Internet Exchange Point traffic statistics to
address this gap, presenting a comprehensive two-year study (2023-2024) from
472 IXPs worldwide, capturing approximately 300 Tbps of peak daily aggregate
traffic by late 2024. Our analysis reveals a 49.2% global traffic increase
(24.5% annualized), uncovers regionally distinct diurnal patterns and
event-driven anomalies, and demonstrates stable utilization rates, reflecting
predictable infrastructure scaling. By analyzing biases and confirming high
self-similarity, we establish IXP traffic as a robust proxy for overall
Internet growth and usage behavior. With transparent, replicable data--covering
87% of the worldwide IXP port capacity--and plans to release our dataset, this
study offers a verifiable foundation for long-term Internet traffic monitoring.
In particular, our findings shed light on the interplay between network design
and function, providing an accessible framework for researchers and operators
to explore the Internet's evolving ecosystem.

### 4. [Ghost Points Matter: Far-Range Vehicle Detection with a Single mmWave Radar in Tunnel](http://arxiv.org/pdf/2509.06639v1)

Authors: Chenming He, Rui Xia, Chengzhen Meng, Xiaoran Fan, Dequan Wang, Haojie Ren, Jianmin Ji, Yanyong Zhang

Vehicle detection in tunnels is crucial for traffic monitoring and accident
response, yet remains underexplored. In this paper, we develop mmTunnel, a
millimeter-wave radar system that achieves far-range vehicle detection in
tunnels. The main challenge here is coping with ghost points caused by
multi-path reflections, which lead to severe localization errors and false
alarms. Instead of merely removing ghost points, we propose correcting them to
true vehicle positions by recovering their signal reflection paths, thus
reserving more data points and improving detection performance, even in
occlusion scenarios. However, recovering complex 3D reflection paths from
limited 2D radar points is highly challenging. To address this problem, we
develop a multi-path ray tracing algorithm that leverages the ground plane
constraint and identifies the most probable reflection path based on signal
path loss and spatial distance. We also introduce a curve-to-plane segmentation
method to simplify tunnel surface modeling such that we can significantly
reduce the computational delay and achieve real-time processing.
  We have evaluated mmTunnel with comprehensive experiments. In two test
tunnels, we conducted controlled experiments in various scenarios with cars and
trucks. Our system achieves an average F1 score of 93.7% for vehicle detection
while maintaining real-time processing. Even in the challenging occlusion
scenarios, the F1 score remains above 91%. Moreover, we collected extensive
data from a public tunnel with heavy traffic at times and show our method could
achieve an F1 score of 91.5% in real-world traffic conditions.

### 5. [Sovereign AI for 6G: Towards the Future of AI-Native Networks](http://arxiv.org/pdf/2509.06700v1)

Authors: Swarna Bindu Chetty, David Grace, Simon Saunders, Paul Harris, Eirini Eleni Tsiropoulou, Tony Quek, Hamed Ahmadi

The advent of Generative Artificial Intelligence (GenAI), Large Language
Models (LLMs), and Large Telecom Models (LTM) significantly reshapes mobile
networks, especially as the telecom industry transitions from 5G's
cloud-centric to AI-native 6G architectures. This transition unlocks
unprecedented capabilities in real-time automation, semantic networking, and
autonomous service orchestration. However, it introduces critical risks related
to data sovereignty, security, explainability, and regulatory compliance
especially when AI models are trained, deployed, or governed externally. This
paper introduces the concept of `Sovereign AI' as a strategic imperative for
6G, proposing architectural, operational, and governance frameworks that enable
national or operator-level control over AI development, deployment, and
life-cycle management. Focusing on O-RAN architecture, we explore how sovereign
AI-based xApps and rApps can be deployed Near-RT and Non-RT RICs to ensure
policy-aligned control, secure model updates, and federated learning across
trusted infrastructure. We analyse global strategies, technical enablers, and
challenges across safety, talent, and model governance. Our findings underscore
that Sovereign AI is not just a regulatory necessity but a foundational pillar
for secure, resilient, and ethically-aligned 6G networks.

### 6. [VariSAC: V2X Assured Connectivity in RIS-Aided ISAC via GNN-Augmented Reinforcement Learning](http://arxiv.org/pdf/2509.06763v1)

Authors: Huijun Tang, Wang Zeng, Ming Du, Pinlong Zhao, Pengfei Jiao, Huaming Wu, Hongjian Sun

The integration of Reconfigurable Intelligent Surfaces (RIS) and Integrated
Sensing and Communication (ISAC) in vehicular networks enables dynamic spatial
resource management and real-time adaptation to environmental changes. However,
the coexistence of distinct vehicle-to-infrastructure (V2I) and
vehicle-to-vehicle (V2V) connectivity requirements, together with highly
dynamic and heterogeneous network topologies, presents significant challenges
for unified reliability modeling and resource optimization. To address these
issues, we propose VariSAC, a graph neural network (GNN)-augmented deep
reinforcement learning framework for assured, time-continuous connectivity in
RIS-assisted, ISAC-enabled vehicle-to-everything (V2X) systems. Specifically,
we introduce the Continuous Connectivity Ratio (CCR), a unified metric that
characterizes the sustained temporal reliability of V2I connections and the
probabilistic delivery guarantees of V2V links, thus unifying their continuous
reliability semantics. Next, we employ a GNN with residual adapters to encode
complex, high-dimensional system states, capturing spatial dependencies among
vehicles, base stations (BS), and RIS nodes. These representations are then
processed by a Soft Actor-Critic (SAC) agent, which jointly optimizes channel
allocation, power control, and RIS configurations to maximize CCR-driven
long-term rewards. Extensive experiments on real-world urban datasets
demonstrate that VariSAC consistently outperforms existing baselines in terms
of continuous V2I ISAC connectivity and V2V delivery reliability, enabling
persistent connectivity in highly dynamic vehicular environments.

### 7. [Resilience of Mega-Satellite Constellations: How Node Failures Impact Inter-Satellite Networking Over Time?](http://arxiv.org/pdf/2509.06766v1)

Authors: Binquan Guo, Zehui Xiong, Zhou Zhang, Baosheng Li, Dusit Niyato, Chau Yuen, Zhu Han

Mega-satellite constellations have the potential to leverage inter-satellite
links to deliver low-latency end-to-end communication services globally,
thereby extending connectivity to underserved regions. However, harsh space
environments make satellites vulnerable to failures, leading to node removals
that disrupt inter-satellite networking. With the high risk of satellite node
failures, understanding their impact on end-to-end services is essential. This
study investigates the importance of individual nodes on inter-satellite
networking and the resilience of mega satellite constellations against node
failures. We represent the mega-satellite constellation as discrete temporal
graphs and model node failure events accordingly. To quantify node importance
for targeted services over time, we propose a service-aware temporal
betweenness metric. Leveraging this metric, we develop an analytical framework
to identify critical nodes and assess the impact of node failures. The
framework takes node failure events as input and efficiently evaluates their
impacts across current and subsequent time windows. Simulations on the Starlink
constellation setting reveal that satellite networks inherently exhibit
resilience to node failures, as their dynamic topology partially restore
connectivity and mitigate the long-term impact. Furthermore, we find that the
integration of rerouting mechanisms is crucial for unleashing the full
resilience potential to ensure rapid recovery of inter-satellite networking.

### 8. [Network-level Censorship Attacks in the InterPlanetary File System](http://arxiv.org/pdf/2509.06626v1)

Authors: Jan Matter, Muoi Tran

The InterPlanetary File System (IPFS) has been successfully established as
the de facto standard for decentralized data storage in the emerging Web3.
Despite its decentralized nature, IPFS nodes, as well as IPFS content
providers, have converged to centralization in large public clouds.
Centralization introduces BGP routing-based attacks, such as passive
interception and BGP hijacking, as potential threats. Although this attack
vector has been investigated for many other Web3 protocols, such as Bitcoin and
Ethereum, to the best of our knowledge, it has not been analyzed for the IPFS
network. In our work, we bridge this gap and demonstrate that BGP routing
attacks can be effectively leveraged to censor content in IPFS. For the
analysis, we collected 3,000 content blocks called CIDs and conducted a
simulation of BGP hijacking and passive interception against them. We find that
a single malicious AS can censor 75% of the IPFS content for more than 57% of
all requester nodes. Furthermore, we show that even with a small set of only 62
hijacked prefixes, 70% of the full attack effectiveness can already be reached.
We further propose and validate countermeasures based on global collaborative
content replication among all nodes in the IPFS network, together with
additional robust backup content provider nodes that are well-hardened against
BGP hijacking. We hope this work raises awareness about the threat BGP
routing-based attacks pose to IPFS and triggers further efforts to harden the
live IPFS network against them.

### 9. [Knowledge-Guided Machine Learning for Stabilizing Near-Shortest Path Routing](http://arxiv.org/pdf/2509.06640v1)

Authors: Yung-Fu Chen, Sen Lin, Anish Arora

We propose a simple algorithm that needs only a few data samples from a
single graph for learning local routing policies that generalize across a rich
class of geometric random graphs in Euclidean metric spaces. We thus solve the
all-pairs near-shortest path problem by training deep neural networks (DNNs)
that let each graph node efficiently and scalably route (i.e., forward) packets
by considering only the node's state and the state of the neighboring nodes.
Our algorithm design exploits network domain knowledge in the selection of
input features and design of the policy function for learning an approximately
optimal policy. Domain knowledge also provides theoretical assurance that the
choice of a ``seed graph'' and its node data sampling suffices for
generalizable learning. Remarkably, one of these DNNs we train -- using
distance-to-destination as the only input feature -- learns a policy that
exactly matches the well-known Greedy Forwarding policy, which forwards packets
to the neighbor with the shortest distance to the destination. We also learn a
new policy, which we call GreedyTensile routing -- using both
distance-to-destination and node stretch as the input features -- that almost
always outperforms greedy forwarding. We demonstrate the explainability and
ultra-low latency run-time operation of Greedy Tensile routing by symbolically
interpreting its DNN in low-complexity terms of two linear actions.

### 10. [BatStation: Toward In-Situ Radar Sensing on 5G Base Stations with Zero-Shot Template Generation](http://arxiv.org/pdf/2509.06898v1)

Authors: Zhihui Gao, Zhecun Liu, Tingjun Chen

The coexistence between incumbent radar signals and commercial 5G signals
necessitates a versatile and ubiquitous radar sensing for efficient and
adaptive spectrum sharing. In this context, leveraging the densely deployed 5G
base stations (BS) for radar sensing is particularly promising, offering both
wide coverage and immediate feedback to 5G scheduling. However, the targeting
radar signals are superimposed with concurrent 5G uplink transmissions received
by the BS, and practical deployment also demands a lightweight, portable radar
sensing model. This paper presents BatStation, a lightweight, in-situ radar
sensing framework seamlessly integrated into 5G BSs. BatStation leverages
uplink resource grids to extract radar signals through three key components:
(i) radar signal separation to cancel concurrent 5G transmissions and reveal
the radar signals, (ii) resource grid reshaping to align time-frequency
resolution with radar pulse characteristics, and (iii) zero-shot template
correlation based on a portable model trained purely on synthetic data that
supports detection, classification, and localization of radar pulses without
fine-tuning using experimental data. We implement BatStation on a
software-defined radio (SDR) testbed and evaluate its performance with real 5G
traffic in the CBRS band. Results show robust performance across diverse radar
types, achieving detection probabilities of 97.02% (PUCCH) and 79.23% (PUSCH),
classification accuracy up to 97.00%, and median localization errors of
2.68-6.20 MHz (frequency) and 24.6-32.4 microseconds (time). Notably,
BatStation achieves this performance with a runtime latency of only 0.11/0.94
ms on GPU/CPU, meeting the real-time requirement of 5G networks.

### Robotics

### 1. [DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration](http://arxiv.org/pdf/2509.06285v1)

Authors: Xiangcheng Hu, Xieyuanli Chen, Mingkai Jia, Jin Wu, Ping Tan, Steven L. Waslander

LiDAR point cloud registration is fundamental to robotic perception and
navigation. However, in geometrically degenerate or narrow environments,
registration problems become ill-conditioned, leading to unstable solutions and
degraded accuracy. While existing approaches attempt to handle these issues,
they fail to address the core challenge: accurately detection, interpret, and
resolve this ill-conditioning, leading to missed detections or corrupted
solutions. In this study, we introduce DCReg, a principled framework that
systematically addresses the ill-conditioned registration problems through
three integrated innovations. First, DCReg achieves reliable ill-conditioning
detection by employing a Schur complement decomposition to the hessian matrix.
This technique decouples the registration problem into clean rotational and
translational subspaces, eliminating coupling effects that mask degeneracy
patterns in conventional analyses. Second, within these cleanly subspaces, we
develop quantitative characterization techniques that establish explicit
mappings between mathematical eigenspaces and physical motion directions,
providing actionable insights about which specific motions lack constraints.
Finally, leveraging this clean subspace, we design a targeted mitigation
strategy: a novel preconditioner that selectively stabilizes only the
identified ill-conditioned directions while preserving all well-constrained
information in observable space. This enables efficient and robust optimization
via the Preconditioned Conjugate Gradient method with a single physical
interpretable parameter. Extensive experiments demonstrate DCReg achieves at
least 20% - 50% improvement in localization accuracy and 5-100 times speedup
over state-of-the-art methods across diverse environments. Our implementation
will be available at https://github.com/JokerJohn/DCReg.

### 2. [Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots](http://arxiv.org/pdf/2509.06342v1)

Authors: Filip Bjelonic, Fabian Tischhauser, Marco Hutter

Legged robots must achieve both robust locomotion and energy efficiency to be
practical in real-world environments. Yet controllers trained in simulation
often fail to transfer reliably, and most existing approaches neglect
actuator-specific energy losses or depend on complex, hand-tuned reward
formulations. We propose a framework that integrates sim-to-real reinforcement
learning with a physics-grounded energy model for permanent magnet synchronous
motors. The framework requires a minimal parameter set to capture the
simulation-to-reality gap and employs a compact four-term reward with a
first-principle-based energetic loss formulation that balances electrical and
mechanical dissipation. We evaluate and validate the approach through a
bottom-up dynamic parameter identification study, spanning actuators,
full-robot in-air trajectories and on-ground locomotion. The framework is
tested on three primary platforms and deployed on ten additional robots,
demonstrating reliable policy transfer without randomization of dynamic
parameters. Our method improves energetic efficiency over state-of-the-art
methods, achieving a 32 percent reduction in the full Cost of Transport of
ANYmal (value 1.27). All code, models, and datasets will be released.

### 3. [Adaptive Evolution Factor Risk Ellipse Framework for Reliable and Safe Autonomous Driving](http://arxiv.org/pdf/2509.06375v1)

Authors: Fujiang Yuan, Zhen Tian, Yangfan He, Guojian Zou, Chunhong Yuan, Yanhong Peng, Zhihao Lin

In recent years, ensuring safety, efficiency, and comfort in interactive
autonomous driving has become a critical challenge. Traditional model-based
techniques, such as game-theoretic methods and robust control, are often overly
conservative or computationally intensive. Conversely, learning-based
approaches typically require extensive training data and frequently exhibit
limited interpretability and generalizability. Simpler strategies, such as Risk
Potential Fields (RPF), provide lightweight alternatives with minimal data
demands but are inherently static and struggle to adapt effectively to dynamic
traffic conditions. To overcome these limitations, we propose the Evolutionary
Risk Potential Field (ERPF), a novel approach that dynamically updates risk
assessments in dynamical scenarios based on historical obstacle proximity data.
We introduce a Risk-Ellipse construct that combines longitudinal reach and
lateral uncertainty into a unified spatial temporal collision envelope.
Additionally, we define an adaptive Evolution Factor metric, computed through
sigmoid normalization of Time to Collision (TTC) and Time-Window-of-Hazard
(TWH), which dynamically adjusts the dimensions of the ellipse axes in real
time. This adaptive risk metric is integrated seamlessly into a Model
Predictive Control (MPC) framework, enabling autonomous vehicles to proactively
address complex interactive driving scenarios in terms of uncertain driving of
surrounding vehicles. Comprehensive comparative experiments demonstrate that
our ERPF-MPC approach consistently achieves smoother trajectories, higher
average speeds, and collision-free navigation, offering a robust and adaptive
solution suitable for complex interactive driving environments.

### 4. [Safety Meets Speed: Accelerated Neural MPC with Safety Guarantees and No Retraining](http://arxiv.org/pdf/2509.06404v1)

Authors: Kaikai Wang, Tianxun Li, Liang Xu, Qinglei Hu, Keyou You

While Model Predictive Control (MPC) enforces safety via constraints, its
real-time execution can exceed embedded compute budgets. We propose a
Barrier-integrated Adaptive Neural Model Predictive Control (BAN-MPC) framework
that synergizes neural networks' fast computation with MPC's
constraint-handling capability. To ensure strict safety, we replace traditional
Euclidean distance with Control Barrier Functions (CBFs) for collision
avoidance. We integrate an offline-learned neural value function into the
optimization objective of a Short-horizon MPC, substantially reducing online
computational complexity. Additionally, we use a second neural network to learn
the sensitivity of the value function to system parameters, and adaptively
adjust the neural value function based on this neural sensitivity when model
parameters change, eliminating the need for retraining and reducing offline
computation costs. The hardware in-the-loop (HIL) experiments on Jetson Nano
show that BAN-MPC solves 200 times faster than traditional MPC, enabling
collision-free navigation with control error below 5\% under model parameter
variations within 15\%, making it an effective embedded MPC alternative.

### 5. [Real-time Photorealistic Mapping for Situational Awareness in Robot Teleoperation](http://arxiv.org/pdf/2509.06433v1)

Authors: Ian Page, Pierre Susbielle, Olivier Aycard, Pierre-Brice Wieber

Achieving efficient remote teleoperation is particularly challenging in
unknown environments, as the teleoperator must rapidly build an understanding
of the site's layout. Online 3D mapping is a proven strategy to tackle this
challenge, as it enables the teleoperator to progressively explore the site
from multiple perspectives. However, traditional online map-based teleoperation
systems struggle to generate visually accurate 3D maps in real-time due to the
high computational cost involved, leading to poor teleoperation performances.
In this work, we propose a solution to improve teleoperation efficiency in
unknown environments. Our approach proposes a novel, modular and efficient
GPU-based integration between recent advancement in gaussian splatting SLAM and
existing online map-based teleoperation systems. We compare the proposed
solution against state-of-the-art teleoperation systems and validate its
performances through real-world experiments using an aerial vehicle. The
results show significant improvements in decision-making speed and more
accurate interaction with the environment, leading to greater teleoperation
efficiency. In doing so, our system enhances remote teleoperation by seamlessly
integrating photorealistic mapping generation with real-time performances,
enabling effective teleoperation in unfamiliar environments.

### 6. [Interactive Shaping of Granular Media Using Reinforcement Learning](http://arxiv.org/pdf/2509.06469v1)

Authors: Benedikt Kreis, Malte Mosbach, Anny Ripke, Muhammad Ehsan Ullah, Sven Behnke, Maren Bennewitz

Autonomous manipulation of granular media, such as sand, is crucial for
applications in construction, excavation, and additive manufacturing. However,
shaping granular materials presents unique challenges due to their
high-dimensional configuration space and complex dynamics, where traditional
rule-based approaches struggle without extensive engineering efforts.
Reinforcement learning (RL) offers a promising alternative by enabling agents
to learn adaptive manipulation strategies through trial and error. In this
work, we present an RL framework that enables a robotic arm with a cubic
end-effector and a stereo camera to shape granular media into desired target
structures. We show the importance of compact observations and concise reward
formulations for the large configuration space, validating our design choices
with an ablation study. Our results demonstrate the effectiveness of the
proposed approach for the training of visual policies that manipulate granular
media including their real-world deployment, outperforming two baseline
approaches.

### 7. [Event Driven CBBA with Reduced Communication](http://arxiv.org/pdf/2509.06481v1)

Authors: Vinita Sao, Tu Dac Ho, Sujoy Bhore, P. B. Sujit

In various scenarios such as multi-drone surveillance and search-and-rescue
operations, deploying multiple robots is essential to accomplish multiple tasks
at once. Due to the limited communication range of these vehicles, a
decentralised task allocation algorithm is crucial for effective task
distribution among robots. The consensus-based bundle algorithm (CBBA) has been
promising for multi-robot operation, offering theoretical guarantees. However,
CBBA demands continuous communication, leading to potential congestion and
packet loss that can hinder performance. In this study, we introduce an
event-driven communication mechanism designed to address these communication
challenges while maintaining the convergence and performance bounds of CBBA. We
demonstrate theoretically that the solution quality matches that of CBBA and
validate the approach with Monte-Carlo simulations across varying targets,
agents, and bundles. Results indicate that the proposed algorithm (ED-CBBA) can
reduce message transmissions by up to 52%.

### 8. [A Robust Approach for LiDAR-Inertial Odometry Without Sensor-Specific Modeling](http://arxiv.org/pdf/2509.06593v1)

Authors: Meher V. R. Malladi, Tiziano Guadagnino, Luca Lobefaro, Cyrill Stachniss

Accurate odometry is a critical component in a robotic navigation stack, and
subsequent modules such as planning and control often rely on an estimate of
the robot's motion. Sensor-based odometry approaches should be robust across
sensor types and deployable in different target domains, from solid-state
LiDARs mounted on cars in urban-driving scenarios to spinning LiDARs on
handheld packages used in unstructured natural environments. In this paper, we
propose a robust LiDAR-inertial odometry system that does not rely on
sensor-specific modeling. Sensor fusion techniques for LiDAR and inertial
measurement unit (IMU) data typically integrate IMU data iteratively in a
Kalman filter or use pre-integration in a factor graph framework, combined with
LiDAR scan matching often exploiting some form of feature extraction. We
propose an alternative strategy that only requires a simplified motion model
for IMU integration and directly registers LiDAR scans in a scan-to-map
approach. Our approach allows us to impose a novel regularization on the LiDAR
registration, improving the overall odometry performance. We detail extensive
experiments on a number of datasets covering a wide array of commonly used
robotic sensors and platforms. We show that our approach works with the exact
same configuration in all these scenarios, demonstrating its robustness. We
have open-sourced our implementation so that the community can build further on
our work and use it in their navigation stacks.

### 9. [LiHRA: A LiDAR-Based HRI Dataset for Automated Risk Monitoring Methods](http://arxiv.org/pdf/2509.06597v1)

Authors: Frederik Plahl, Georgios Katranis, Ilshat Mamaev, Andrey Morozov

We present LiHRA, a novel dataset designed to facilitate the development of
automated, learning-based, or classical risk monitoring (RM) methods for
Human-Robot Interaction (HRI) scenarios. The growing prevalence of
collaborative robots in industrial environments has increased the need for
reliable safety systems. However, the lack of high-quality datasets that
capture realistic human-robot interactions, including potentially dangerous
events, slows development. LiHRA addresses this challenge by providing a
comprehensive, multi-modal dataset combining 3D LiDAR point clouds, human body
keypoints, and robot joint states, capturing the complete spatial and dynamic
context of human-robot collaboration. This combination of modalities allows for
precise tracking of human movement, robot actions, and environmental
conditions, enabling accurate RM during collaborative tasks. The LiHRA dataset
covers six representative HRI scenarios involving collaborative and coexistent
tasks, object handovers, and surface polishing, with safe and hazardous
versions of each scenario. In total, the data set includes 4,431 labeled point
clouds recorded at 10 Hz, providing a rich resource for training and
benchmarking classical and AI-driven RM algorithms. Finally, to demonstrate
LiHRA's utility, we introduce an RM method that quantifies the risk level in
each scenario over time. This method leverages contextual information,
including robot states and the dynamic model of the robot. With its combination
of high-resolution LiDAR data, precise human tracking, robot state data, and
realistic collision events, LiHRA offers an essential foundation for future
research into real-time RM and adaptive safety strategies in human-robot
workspaces.

### 10. [T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation](http://arxiv.org/pdf/2509.06644v1)

Authors: Xiaobei Zhao, Xingqi Lyu, Xiang Li

Agricultural robotic agents have been becoming powerful helpers in a wide
range of agricultural tasks, nevertheless, still heavily rely on manual
operation or untransportable railway for movement. The AgriVLN method and the
A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the
agricultural domain, enabling agents navigate to the target position following
the natural language instructions. AgriVLN effectively understands the simple
instructions, however, often misunderstands the complicated instructions. To
bridge this gap, we propose the method of Translator for Agricultural Robotic
Agents on Vision-and-Language Navigation (T-araVLN), in which the Instruction
Translator module translates the original instruction to be both refined and
precise. Being evaluated on the A2A benchmark, our T-araVLN effectively
improves SR from 0.47 to 0.63 and reduces NE from 2.91m to 2.28m, demonstrating
the state-of-the-art performance in the agricultural domain. Code:
https://github.com/AlexTraveling/T-araVLN.

### Software Engineering

### 1. [Learning From Software Failures: A Case Study at a National Space Research Center](http://arxiv.org/pdf/2509.06301v1)

Authors: Dharun Anandayuvaraj, Zain Hammadeh, Andreas Lund, Alexandra Holloway, James C. Davis

Software failures can have significant consequences, making learning from
failures a critical aspect of software engineering. While software
organizations are recommended to conduct postmortems, the effectiveness and
adoption of these practices vary widely. Understanding how engineers gather,
document, share, and apply lessons from failures is essential for improving
reliability and preventing recurrence. High-reliability organizations (HROs)
often develop software systems where failures carry catastrophic risks,
requiring continuous learning to ensure reliability. These organizations
provide a valuable setting to examine practices and challenges for learning
from software failures. Such insight could help develop processes and tools to
improve reliability and prevent recurrence. However, we lack in-depth industry
perspectives on the practices and challenges of learning from failures.
  To address this gap, we conducted a case study through 10 in-depth interviews
with research software engineers at a national space research center. We
examine how they learn from failures: how they gather, document, share, and
apply lessons. To assess transferability, we include data from 5 additional
interviews at other HROs. Our findings provide insight into how engineers learn
from failures in practice. To summarize: (1) failure learning is informal, ad
hoc, and inconsistently integrated into SDLC; (2) recurring failures persist
due to absence of structured processes; and (3) key challenges, including time
constraints, knowledge loss from turnover and fragmented documentation, and
weak process enforcement, undermine systematic learning. Our findings deepen
understanding of how software engineers learn from failures and offer guidance
for improving failure management practices.

### 2. [A Generic and Efficient Python Runtime Verification System and its Large-scale Evaluation](http://arxiv.org/pdf/2509.06324v1)

Authors: Zhuohang Shen, Mohammed Yaseen, Denini Silva, Kevin Guan, Junho Lee, Marcelo d'Amorim, Owolabi Legunsen

Runtime verification (RV) now scales for testing thousands of open-source
Java projects, helping find hundreds of bugs. The popular Python ecosystem
could use such benefits. But, today's Python RV systems are limited to a domain
or specification logic, or slow. We propose PyMOP, a generic, extensible, and
efficient RV system for Python. PyMOP supports five logics, implements five
existing monitoring algorithms, ships with 73 API specs of Python and
widely-used libraries, supports three instrumentation strategies, and users can
easily add more of these. On 290,133 unit tests in 1,463 GitHub projects, we
find mainly that (i) the default monitoring algorithm for Java is often not the
fastest for Python; (ii) PyMOP is up to 1,168.3x faster than two recent dynamic
analysis systems; and (iii) 44 of 121 bugs that PyMOP helped find so far were
fixed by developers. PyMOP's generality and efficiency position it well as an
excellent platform for the next advances on RV for Python.

### 3. [Analyzing the Instability of Large Language Models in Automated Bug Injection and Correction](http://arxiv.org/pdf/2509.06429v1)

Authors: Mehmet Bilal Er, Nagehan İlhan, Umut Kuran

The use of Large Language Models (LLMs) in software engineering tasks is
growing, especially in the areas of bug fixing and code generation.
Nevertheless, these models often yield unstable results; when executed at
different times with the same input, they can generate radically different
code. The consistency of LLMs in bug-fixing tasks has not yet been thoroughly
assessed, despite the fact that this instability has typically been discussed
in the literature in relation to code generation. The purpose of this study is
to look into how unstable an LLM like ChatGPT is when it comes to fixing code
bugs. We examine the structural, syntactic, and functional variations among
several fix recommendations made in response to the same prompt using code
samples with various error types. Additionally, we assess how instability is
affected by the temperature settings (0, 0.5, and 1) used for the model's
deterministic operation. For a total of 20 problems in the experimental
analysis, the model produced three fix suggestions at each temperature value,
comparing nine distinct outputs for each problem. The Syntax Similarity and
Output Equivalence Rate (OER) metrics were used to assess the outputs'
structural and functional consistency. The results demonstrate that the model's
outputs become much more unstable and variable as the temperature rises, with
high temperatures showing especially high rates of functional failure.
According to syntax similarity analyses, the suggested fixes show notable
structural differences at high temperatures but are fairly similar at low
temperatures. The purpose of this study is to provide important methodological
insights into how LLM-based error correction systems can be applied more
consistently in software development processes while also casting doubt on
their dependability.

### 4. [Modeling in the Design Multiverse](http://arxiv.org/pdf/2509.06530v1)

Authors: Sylvain Guérin, Salvador Martinez, Ciprian Teodorov

Real-world design processes often involve the evolution and divergence of
design paths (by branching, revising, merging, etc.), especially when multiple
stakeholders or teams operate concurrently and/or explore different
alternatives for complex and heterogeneous systems. Unfortunately, this
variability in time and space can not be directly managed in current modeling
spaces but requires resorting to external tools and methodologies.
  In order to tackle this problem, we introduce the Design Multiverse. The
Design Multiverse aims to integrate in the modeling space a selection of
revisions and variants, representing snapshots of a design state composed of
multiple artifacts. This enables stakeholders to seamlessly trace, analyze, and
manage design decisions, system variants, and their interdependencies.
Concretely, in this paper we present a conceptual definition of the Design
Multiverse, discuss usage scenarios such as model product lines and
model/metamodel co-evolution, and propose an implementation leveraging the
model federation paradigm.

### 5. [Design and Implementation of a Domain-specific Language for Modelling Evacuation Scenarios Using Eclipse EMG/GMF Tool](http://arxiv.org/pdf/2509.06688v1)

Authors: Heerok Banerjee

Domain-specific languages (DSLs) play a crucial role in resolving internal
dependencies across enterprises and boosts their upfront business management
processes. Yet, a lot of development is needed to build modelling frameworks
which support graphical interfaces (canvas, pallettes etc.), hierarchical
structures and easy implementation to shorten the gap for novice users. In this
paper, a DSL namely, Bmod is introduced, which can be used to model evacuation
scenarios. The language is built using Eclipse Modelling Framework (EMF) and
Eclipse Graphical Modelling Framework (GMF). Furthermore, a comparison is also
shown between Eclipse EMF/GMF and other modelling tools such as AToMPM,
metaDepth, Sirius etc with respect to expressiveness, learning curve and
performance.

### 6. [OpenCoderRank: AI-Driven Technical Assessments Made Easy](http://arxiv.org/pdf/2509.06774v1)

Authors: Hridoy Sankar Dutta, Sana Ansari, Swati Kumari, Shounak Ravi Bhalerao

Organizations and educational institutions use time-bound assessment tasks to
evaluate coding and problem-solving skills. These assessments measure not only
the correctness of the solutions, but also their efficiency. Problem setters
(educator/interviewer) are responsible for crafting these challenges, carefully
balancing difficulty and relevance to create meaningful evaluation experiences.
Conversely, problem solvers (student/interviewee) apply coding efficiency and
logical thinking to arrive at correct solutions. In the era of Large Language
Models (LLMs), LLMs assist problem setters in generating diverse and
challenging questions, but they can undermine assessment integrity for problem
solvers by providing easy access to solutions. This paper introduces
OpenCoderRank, an easy-to-use platform designed to simulate technical
assessments. It acts as a bridge between problem setters and problem solvers,
helping solvers prepare for time constraints and unfamiliar problems while
allowing setters to self-host assessments, offering a no-cost and customizable
solution for technical assessments in resource-constrained environments.

### 7. [Efficiently Ranking Software Variants with Minimal Benchmarks](http://arxiv.org/pdf/2509.06716v1)

Authors: Théo Matricon, Mathieu Acher, Helge Spieker, Arnaud Gotlieb

Benchmarking is a common practice in software engineering to assess the
qualities and performance of software variants, coming from multiple competing
systems or from configurations of the same system. Benchmarks are used notably
to compare and understand variant performance, fine-tune software, detect
regressions, or design new software systems. The execution of benchmarks to get
a complete picture of software variants is highly costly in terms of
computational resources and time. In this paper, we propose a novel approach
for reducing benchmarks while maintaining stable rankings, using test suite
optimization techniques. That is, we remove instances from the benchmarks while
trying to keep the same rankings of the variants on all tests. Our method,
BISection Sampling, BISS, strategically retains the most critical tests and
applies a novel divide-and-conquer approach to efficiently sample among
relevant remaining tests. We experiment with datasets and use cases from LLM
leaderboards, SAT competitions, and configurable systems for performance
modeling. Our results show that our method outperforms baselines even when
operating on a subset of variants. Using BISS, we reduce the computational cost
of the benchmarks on average to 44% and on more than half the benchmarks by up
to 99% without loss in ranking stability.

### 8. [MIO: Multiverse Debugging in the Face of Input/Output -- Extended Version with Additional Appendices](http://arxiv.org/pdf/2509.06845v1)

Authors: Tom Lauwaerts, Maarten Steevens, Christophe Scholliers

Debugging non-deterministic programs on microcontrollers is notoriously
challenging, especially when bugs manifest in unpredictable, input-dependent
execution paths. A recent approach, called multiverse debugging, makes it
easier to debug non-deterministic programs by allowing programmers to explore
all potential execution paths. Current multiverse debuggers enable both forward
and backward traversal of program paths, and some facilitate jumping to any
previously visited states, potentially branching into alternative execution
paths within the state space.
  Unfortunately, debugging programs that involve input/output operations using
existing multiverse debuggers can reveal inaccessible program states, i.e.
states which are not encountered during regular execution. This can
significantly hinder the debugging process, as the programmer may spend
substantial time exploring and examining inaccessible program states, or worse,
may mistakenly assume a bug is present in the code, when in fact, the issue is
caused by the debugger.
  This paper presents a novel approach to multiverse debugging, which can
accommodate a broad spectrum of input/output operations. We provide the
semantics of our approach and prove the correctness of our debugger, ensuring
that despite having support for a wide range of input/output operations the
debugger will only explore those program states which can be reached during
regular execution.
  We have developed a prototype, called MIO, leveraging the WARDuino
WebAssembly virtual machine to demonstrate the feasibility and efficiency of
our techniques. As a demonstration of the approach we highlight a color dial
built with a Lego Mindstorms motor, and color sensor, providing a tangible
example of how our approach enables multiverse debugging for programs running
on an STM32 microcontroller.

### 9. [Concolic Testing on Individual Fairness of Neural Network Models](http://arxiv.org/pdf/2509.06864v1)

Authors: Ming-I Huang, Chih-Duo Hong, Fang Yu

This paper introduces PyFair, a formal framework for evaluating and verifying
individual fairness of Deep Neural Networks (DNNs). By adapting the concolic
testing tool PyCT, we generate fairness-specific path constraints to
systematically explore DNN behaviors. Our key innovation is a dual network
architecture that enables comprehensive fairness assessments and provides
completeness guarantees for certain network types. We evaluate PyFair on 25
benchmark models, including those enhanced by existing bias mitigation
techniques. Results demonstrate PyFair's efficacy in detecting discriminatory
instances and verifying fairness, while also revealing scalability challenges
for complex models. This work advances algorithmic fairness in critical domains
by offering a rigorous, systematic method for fairness testing and verification
of pre-trained DNNs.

### 10. [Hypergraph-Guided Regex Filter Synthesis for Event-Based Anomaly Detection](http://arxiv.org/pdf/2509.06911v1)

Authors: Margarida Ferreira, Victor Nicolet, Luan Pham, Joey Dodds, Daniel Kroening, Ines Lynce, Ruben Martins

We propose HyGLAD, a novel algorithm that automatically builds a set of
interpretable patterns that model event data. These patterns can then be used
to detect event-based anomalies in a stationary system, where any deviation
from past behavior may indicate malicious activity. The algorithm infers
equivalence classes of entities with similar behavior observed from the events,
and then builds regular expressions that capture the values of those entities.
As opposed to deep-learning approaches, the regular expressions are directly
interpretable, which also translates to interpretable anomalies. We evaluate
HyGLAD against all 7 unsupervised anomaly detection methods from DeepOD on five
datasets from real-world systems. The experimental results show that on average
HyGLAD outperforms existing deep-learning methods while being an order of
magnitude more efficient in training and inference (single CPU vs GPU).
Precision improved by 1.2x and recall by 1.3x compared to the second-best
baseline.

### Social and Information Networks

### 1. [No Such Thing as Free Brain Time: For a Pigouvian Tax on Attention Capture](http://arxiv.org/pdf/2509.06453v1)

Authors: Hamza Belgroun, Franck Michel, Fabien Gandon

In our age of digital platforms, human attention has become a scarce and
highly valuable resource, rivalrous, tradable, and increasingly subject to
market dynamics. This article explores the commodification of attention within
the framework of the attention economy, arguing that attention should be
understood as a common good threatened by over-exploitation. Drawing from
philosophical, economic, and legal perspectives, we first conceptualize
attention not only as an individual cognitive process but as a collective and
infrastructural phenomenon susceptible to enclosure by digital intermediaries.
We then identify and analyze negative externalities of the attention economy,
particularly those stemming from excessive screen time: diminished individual
agency, adverse health outcomes, and societal and political harms, including
democratic erosion and inequality. These harms are largely unpriced by market
actors and constitute a significant market failure. In response, among a
spectrum of public policy tools ranging from informational campaigns to
outright restrictions, we propose a Pigouvian tax on attention capture as a
promising regulatory instrument to internalize the externalities and, in
particular, the social cost of compulsive digital engagement. Such a tax would
incentivize structural changes in platform design while preserving user
autonomy. By reclaiming attention as a shared resource vital to human agency,
health, and democracy, this article contributes a novel economic and policy
lens to the debate on digital regulation. Ultimately, this article advocates
for a paradigm shift: from treating attention as a private, monetizable asset
to protecting it as a collective resource vital for humanity.

### 2. [Unveiling the Listener Structure Underlying K-pop's Global Success: A Large-Scale Listening Data Analysis](http://arxiv.org/pdf/2509.06606v1)

Authors: Ryota Nakamura, Keita Nishimoto, Ichiro Sakata, Kimitaka Asatani

From the mid-2000s to the 2010s, K-pop moved beyond its status as a
regionally popular genre in Asia and established itself as a global music genre
with enthusiastic fans around the world. However, little is known about how the
vast number of music listeners across the globe have listened to and perceived
K-pop. This study addresses this question by analyzing a large-scale listening
dataset from Last.fm. An analysis of the distribution of play counts reveals
that K-pop experienced a significant increase in plays between 2005 and 2019,
largely supported by a small group of heavy listeners. The Gini coefficient in
play counts is notably greater than that of existing mainstream genres and
other growing niche genres. Furthermore, an analysis based on user-assigned
genre tags quantitatively demonstrates that between 2005 and 2010, K-pop shed
its status as a local Asian genre and established itself as a distinct music
genre in its own right.

### Systems and Control

### 1. [DNN-based Digital Twin Framework of a DC-DC Buck Converter using Spider Monkey Optimization Algorithm](http://arxiv.org/pdf/2509.06279v1)

Authors: Tahmin Mahmud, Euzeli Cipriano Dos Santos Jr

Component ageing is a critical concern in power electronic converter systems
(PECSs). It directly impacts the reliability, performance, and operational
lifespan of converters used across diverse applications, including electric
vehicles (EVs), renewable energy systems (RESs) and industrial automation.
Therefore, understanding and monitoring component ageing is crucial for
developing robust converters and achieving long-term system reliability. This
paper proposes a data-driven digital twin (DT) framework for DC-DC buck
converters, integrating deep neural network (DNN) with the spider monkey
optimization (SMO) algorithm to monitor and predict component degradation.
Utilizing a low-power prototype testbed along with empirical and synthetic
datasets, the SMO+DNN approach achieves the global optimum in 95% of trials,
requires 33% fewer iterations, and results in 80% fewer parameter constraint
violations compared to traditional methods. The DNN model achieves $R^2$ scores
above 0.998 for all key degradation parameters and accurately forecasts time to
failure ($t_{failure}$). In addition, SMO-tuned degradation profile improves
the converter's performance by reducing voltage ripple by 20-25% and inductor
current ripple by 15-20%.

### 2. [First-Principle Modeling Framework of Boost Converter Dynamics for Precise Energy Conversions in Space](http://arxiv.org/pdf/2509.06425v1)

Authors: Yifan Wang, Wenhua Li, Zhenlong Wang, Xinrui Zhang, Jianfeng Sun, Qianfu Xia, Zhongtao Gou, Jiangang Rong, Tao Ye

Boost converters are essential for modern electrification and intelligent
technologies. However, conventional Boost converter models relying on
steady-state assumptions fail to accurately predict transient behaviors during
input voltage and load fluctuations, which cause significant output voltage
overshoots and instability, resulting in failures of electrical systems,
thereby restricting their use in space. This study introduces a first-principle
modeling framework that derives precise dynamic equations for Boost converters
by incorporating non-ideal component coupling. As compared to the most accurate
existing Boost converter model, the proposed models reduce steady-state and
dynamic-state errors between experimental and simulated output voltages by
factors of 11.0 (from 20.9% to 1.9%) and 15.4 (from 77.1% to 5.0%) under input
voltage variations, and by factors of 10.2 (from 15.3% to 1.5%) and 35.1 (from
42.1% to 1.2%) under load changes, respectively. Consequently, a reliable Boost
converter is accordingly designed and on-orbit deployed for precise energy
conversions.

### 3. [Unified Graph-Theoretic Modeling of Multi-Energy Flows in Distribution Systems](http://arxiv.org/pdf/2509.06447v1)

Authors: Marwan Mostafa, Daniel Wenser, Payam Teimourzadeh Baboli, Christian Becker

The increasing complexity of energy systems due to sector coupling and
decarbonization calls for unified modeling frameworks that capture the physical
and structural interactions between electricity, gas, and heat networks. This
paper presents a graph-based modeling approach for multi-energy systems, where
each domain is represented as a layer in a multi-layer graph, and coupling
technologies are modeled as inter-layer edges via a dedicated coupling layer. A
steady-state solver based on a block-structured Newton-Raphson method is
developed to jointly compute flows and state variables across all carriers. The
proposed model is tested and validated on a realistic case study based on data
from a German distribution network. The results demonstrate convergence,
numerical accuracy, and consistent domain interaction, and demonstrate the
method's applicability for system-wide analysis and its potential as a
foundation for future optimizations in integrated energy systems.

### 4. [Wireless Low-Latency Synchronization for Body-Worn Multi-Node Systems in Sports](http://arxiv.org/pdf/2509.06541v1)

Authors: Nico Krull, Lukas Schulthess, Michele Magno, Luca Benini, Christoph Leitner

Biomechanical data acquisition in sports demands sub-millisecond
synchronization across distributed body-worn sensor nodes. This study evaluates
and characterizes the Enhanced ShockBurst (ESB) protocol from Nordic
Semiconductor under controlled laboratory conditions for wireless, low-latency
command broadcasting, enabling fast event updates in multi-node systems.
Through systematic profiling of protocol parameters, including
cyclic-redundancy-check modes, bitrate, transmission modes, and payload
handling, we achieve a mean Device-to-Device (D2D) latency of 504.99 +- 96.89
us and a network-to-network core latency of 311.78 +- 96.90 us using a one-byte
payload with retransmission optimization. This performance significantly
outperforms Bluetooth Low Energy (BLE), which is constrained by a 7.5 ms
connection interval, by providing deterministic, sub-millisecond
synchronization suitable for high-frequency (500 Hz to 1000 Hz) biosignals.
These results position ESB as a viable solution for time-critical, multi-node
wearable systems in sports, enabling precise event alignment and reliable
high-speed data fusion for advanced athlete monitoring and feedback
applications.

### 5. [Human-Hardware-in-the-Loop simulations for systemic resilience assessment in cyber-socio-technical systems](http://arxiv.org/pdf/2509.06657v1)

Authors: Francesco Simone, Marco Bortolini, Giovanni Mazzuto, Giulio di Gravio, Riccardo Patriarca

Modern industrial systems require updated approaches to safety management, as
the tight interplay between cyber-physical, human, and organizational factors
has driven their processes toward increasing complexity. In addition to dealing
with known risks, managing system resilience acquires great value to address
complex behaviors pragmatically. This manuscript starts from the
System-Theoretic Accident Model and Processes (STAMP) as a modelling initiative
for such complexity. The STAMP can be natively integrated with simulation-based
approaches, which however fail to realistically represent human behaviors and
their influence on the system performance. To overcome this limitation, this
paper proposes a Human-Hardware-in-the-Loop (HHIL) modeling and simulation
framework aimed at supporting a more realistic and comprehensive assessments of
systemic resilience. The approach is tested on an experimental oil and gas
plant experiencing cyber-attacks, where two personas of operators (experts and
novices) work. This research provides a mean to quantitatively assess how
variations in operator behavior impact the overall system performance, offering
insights into how resilience should be understood and implemented in complex
socio-technical systems at large.

### 6. [Edge Server Monitoring for Job Assignment](http://arxiv.org/pdf/2509.06722v1)

Authors: Samuel Chamoun, Sirin Chakraborty, Eric Graves, Kevin Chan, Yin Sun

In this paper, we study a goal-oriented communication problem for edge server
monitoring, where compute jobs arrive intermittently at dispatchers and must be
immediately assigned to distributed edge servers. Due to competing workloads
and the dynamic nature of the edge environment, server availability fluctuates
over time. To maintain accurate estimates of server availability states, each
dispatcher updates its belief using two mechanisms: (i) active queries over
shared communication channels and (ii) feedback from past job executions. We
formulate a query scheduling problem that maximizes the job success rate under
limited communication resources for queries. This problem is modeled as a
Restless Multi-Armed Bandit (RMAB) with multiple actions and addressed using a
Net-Gain Maximization (NGM) scheduling algorithm, which selects servers to
query based on their expected improvement in execution performance. Simulation
results show that the proposed NGM Policy significantly outperforms baseline
strategies, achieving up to a 30% gain over the Round-Robin Policy and up to a
107% gain over the Never-Query Policy.

### 7. [Steering Opinion through Dynamic Stackelberg Optimization](http://arxiv.org/pdf/2509.06758v1)

Authors: Hossein Rastgoftar

This paper employs the Friedkin-Johnsen (FJ) model to describe the dynamics
of opinion evolution within a social network. Under the FJ framework, the
society is divided into two subgroups that include stubborn agents and regular
agents. The opinions of stubborn agents are not influenced by regular agents,
whereas the opinions of regular agents evolve based on the opinions of their
neighboring agents. By defining the origin as the desired collective opinion of
the society, the objective of the paper is to minimize deviations from this
desired opinion. To achieve this, a Stackelberg game is established between the
stubborn and regular subgroups, where the opinion adjustments of the stubborn
agents and the openness variables of regular agents serve as the decision
variables. The proposed solution approach integrates quadratic programming and
dynamic programming to optimize these decision variables at each discrete time
step using forward and backward propagation.

### 8. [Agentic DDQN-Based Scheduling for Licensed and Unlicensed Band Allocation in Sidelink Networks](http://arxiv.org/pdf/2509.06775v1)

Authors: Po-Heng Chou, Pin-Qi Fu, Walid Saad, Li-Chun Wang

This paper presents an agentic artificial intelligence (AI)-driven double
deep Q-network (DDQN) scheduling framework for licensed and unlicensed band
allocation in New Radio (NR) sidelink (SL) networks. SL must share licensed
spectrum with cellular communications (CC) and unlicensed bands with Wi-Fi,
posing significant challenges for coexistence. Unlike prior rule-based or
threshold-based methods, the proposed agentic scheduler autonomously perceives
queueing dynamics, channel conditions, and coexistence states, and adapts its
policy to maintain quality-of-service (QoS). Simulation results show that our
framework reduces the blocking rate by up to 87.5% compared to threshold-based
scheduling under limited licensed bandwidth. These findings demonstrate the
potential of Agentic AI to enable stable, QoS-aware, and adaptive scheduling
for future NR SL systems.

### 9. [Human Body Weight Estimation Through Music-Induced Bed Vibrations](http://arxiv.org/pdf/2509.06257v1)

Authors: Yuyan Wu, Jiale Zhang, Moon Lee, Cherrelle Smith, Xinyi Li, Ankur Senapati, Pei Zhang, Hae Young Noh

Rapid and accurate body weight estimation is critical in emergency medical
care, as it directly influences treatment decisions, such as drug dosing,
defibrillation energy selection, and fluid resuscitation. Traditional methods
such as stand-on scales, length-based tapes, or transfer-based weighing scales
are often impractical for immobilized patients, inaccurate, or labor-intensive
and time-consuming. This paper introduces MelodyBedScale, a non-intrusive and
rapid on-bed weight estimation system that leverages bed vibration induced by
music. The core insight is that body weight affects the vibration transfer
function of the bed-body system, which is captured using vibration sensors
placed on opposite sides of the bed. First, we identify weight-sensitive
frequency bands and compose clinically acceptable soft, natural music with high
signal energy in these frequency bands. This music is then played through a
speaker mounted on the bed to induce bed vibrations. Additionally, to
efficiently capture the complex weight-vibration relationship with limited data
and enhance generalizability to unseen individuals and weights, we
theoretically analyze the weight-vibration relationship and integrate the
results into the activation functions of the neural network for
physics-informed weight regression. We evaluated MelodyBedScale on both wooden
and steel beds across 11 participants, achieving a mean absolute error of up to
1.55 kg.

### 10. [Enhancing Low-Altitude Airspace Security: MLLM-Enabled UAV Intent Recognition](http://arxiv.org/pdf/2509.06312v1)

Authors: Guangyu Lei, Tianhao Liang, Yuqi Ping, Xinglin Chen, Longyu Zhou, Junwei Wu, Xiyuan Zhang, Huahao Ding, Xingjian Zhang, Weijie Yuan, Tingting Zhang, Qinyu Zhang

The rapid development of the low-altitude economy emphasizes the critical
need for effective perception and intent recognition of non-cooperative
unmanned aerial vehicles (UAVs). The advanced generative reasoning capabilities
of multimodal large language models (MLLMs) present a promising approach in
such tasks. In this paper, we focus on the combination of UAV intent
recognition and the MLLMs. Specifically, we first present an MLLM-enabled UAV
intent recognition architecture, where the multimodal perception system is
utilized to obtain real-time payload and motion information of UAVs, generating
structured input information, and MLLM outputs intent recognition results by
incorporating environmental information, prior knowledge, and tactical
preferences. Subsequently, we review the related work and demonstrate their
progress within the proposed architecture. Then, a use case for low-altitude
confrontation is conducted to demonstrate the feasibility of our architecture
and offer valuable insights for practical system design. Finally, the future
challenges are discussed, followed by corresponding strategic recommendations
for further applications.

### Machine Learning (Statistics Category)

### 1. [Automated Hierarchical Graph Construction for Multi-source Electronic Health Records](http://arxiv.org/pdf/2509.06576v1)

Authors: Yinjie Wang, Doudou Zhou, Yue Liu, Junwei Lu, Tianxi Cai

Electronic Health Records (EHRs), comprising diverse clinical data such as
diagnoses, medications, and laboratory results, hold great promise for
translational research. EHR-derived data have advanced disease prevention,
improved clinical trial recruitment, and generated real-world evidence.
Synthesizing EHRs across institutions enables large-scale, generalizable
studies that capture rare diseases and population diversity, but remains
hindered by the heterogeneity of medical codes, institution-specific
terminologies, and the absence of standardized data structures. These barriers
limit the interpretability, comparability, and scalability of EHR-based
analyses, underscoring the need for robust methods to harmonize and extract
meaningful insights from distributed, heterogeneous data. To address this, we
propose MASH (Multi-source Automated Structured Hierarchy), a fully automated
framework that aligns medical codes across institutions using neural optimal
transport and constructs hierarchical graphs with learned hyperbolic
embeddings. During training, MASH integrates information from pre-trained
language models, co-occurrence patterns, textual descriptions, and supervised
labels to capture semantic and hierarchical relationships among medical
concepts more effectively. Applied to real-world EHR data, including diagnosis,
medication, and laboratory codes, MASH produces interpretable hierarchical
graphs that facilitate the navigation and understanding of heterogeneous
clinical data. Notably, it generates the first automated hierarchies for
unstructured local laboratory codes, establishing foundational references for
downstream applications.

### 2. [Not All Samples Are Equal: Quantifying Instance-level Difficulty in Targeted Data Poisoning](http://arxiv.org/pdf/2509.06896v1)

Authors: William Xu, Yiwei Lu, Yihan Wang, Matthew Y. R. Yang, Zuoqiu Liu, Gautam Kamath, Yaoliang Yu

Targeted data poisoning attacks pose an increasingly serious threat due to
their ease of deployment and high success rates. These attacks aim to
manipulate the prediction for a single test sample in classification models.
Unlike indiscriminate attacks that aim to decrease overall test performance,
targeted attacks present a unique threat to individual test instances. This
threat model raises a fundamental question: what factors make certain test
samples more susceptible to successful poisoning than others? We investigate
how attack difficulty varies across different test instances and identify key
characteristics that influence vulnerability. This paper introduces three
predictive criteria for targeted data poisoning difficulty: ergodic prediction
accuracy (analyzed through clean training dynamics), poison distance, and
poison budget. Our experimental results demonstrate that these metrics
effectively predict the varying difficulty of real-world targeted poisoning
attacks across diverse scenarios, offering practitioners valuable insights for
vulnerability assessment and understanding data poisoning attacks.

### 3. [MOSAIC: Minimax-Optimal Sparsity-Adaptive Inference for Change Points in Dynamic Networks](http://arxiv.org/pdf/2509.06303v1)

Authors: Yingying Fan, Jingyuan Liu, Jinchi Lv, Ao Sun

We propose a new inference framework, named MOSAIC, for change-point
detection in dynamic networks with the simultaneous low-rank and sparse-change
structure. We establish the minimax rate of detection boundary, which relies on
the sparsity of changes. We then develop an eigen-decomposition-based test with
screened signals that approaches the minimax rate in theory, with only a minor
logarithmic loss. For practical implementation of MOSAIC, we adjust the
theoretical test by a novel residual-based technique, resulting in a pivotal
statistic that converges to a standard normal distribution via the martingale
central limit theorem under the null hypothesis and achieves full power under
the alternative hypothesis. We also analyze the minimax rate of testing
boundary for dynamic networks without the low-rank structure, which almost
aligns with the results in high-dimensional mean-vector change-point inference.
We showcase the effectiveness of MOSAIC and verify our theoretical results with
several simulation examples and a real data application.

### 4. [Minimax optimal transfer learning for high-dimensional additive regression](http://arxiv.org/pdf/2509.06308v1)

Authors: Seung Hyun Moon

This paper studies high-dimensional additive regression under the transfer
learning framework, where one observes samples from a target population
together with auxiliary samples from different but potentially related
regression models. We first introduce a target-only estimation procedure based
on the smooth backfitting estimator with local linear smoothing. In contrast to
previous work, we establish general error bounds under sub-Weibull($\alpha$)
noise, thereby accommodating heavy-tailed error distributions. In the
sub-exponential case ($\alpha=1$), we show that the estimator attains the
minimax lower bound under regularity conditions, which requires a substantial
departure from existing proof strategies. We then develop a novel two-stage
estimation method within a transfer learning framework, and provide theoretical
guarantees at both the population and empirical levels. Error bounds are
derived for each stage under general tail conditions, and we further
demonstrate that the minimax optimal rate is achieved when the auxiliary and
target distributions are sufficiently close. All theoretical results are
supported by simulation studies and real data analysis.

### 5. [On optimal solutions of classical and sliced Wasserstein GANs with non-Gaussian data](http://arxiv.org/pdf/2509.06505v1)

Authors: Yu-Jui Huang, Hsin-Hua Shen, Yu-Chih Huang, Wan-Yi Lin, Shih-Chun Lin

The generative adversarial network (GAN) aims to approximate an unknown
distribution via a parameterized neural network (NN). While GANs have been
widely applied in reinforcement and semisupervised learning as well as computer
vision tasks, selecting their parameters often needs an exhaustive search and
only a few selection methods can be proved to be theoretically optimal. One of
the most promising GAN variants is the Wasserstein GAN (WGAN). Prior work on
optimal parameters for WGAN is limited to the linear-quadratic-Gaussian (LQG)
setting, where the NN is linear and the data is Gaussian. In this paper, we
focus on the characterization of optimal WGAN parameters beyond the LQG
setting. We derive closed-form optimal parameters for one-dimensional WGANs
when the NN has non-linear activation functions and the data is non-Gaussian.
To extend this to high-dimensional WGANs, we adopt the sliced Wasserstein
framework and replace the constraint on marginal distributions of the randomly
projected data by a constraint on the joint distribution of the original
(unprojected) data. We show that the linear generator can be asymptotically
optimal for sliced WGAN with non-Gaussian data. Empirical studies show that our
closed-form WGAN parameters have good convergence behavior with data under both
Gaussian and Laplace distributions. Also, compared to the r principal component
analysis (r-PCA) solution, our proposed solution for sliced WGAN can achieve
the same performance while requiring less computational resources.

### 6. [Robust and Adaptive Spectral Method for Representation Multi-Task Learning with Contamination](http://arxiv.org/pdf/2509.06575v1)

Authors: Yian Huang, Yang Feng, Zhiliang Ying

Representation-based multi-task learning (MTL) improves efficiency by
learning a shared structure across tasks, but its practical application is
often hindered by contamination, outliers, or adversarial tasks. Most existing
methods and theories assume a clean or near-clean setting, failing when
contamination is significant. This paper tackles representation MTL with an
unknown and potentially large contamination proportion, while also allowing for
heterogeneity among inlier tasks. We introduce a Robust and Adaptive Spectral
method (RAS) that can distill the shared inlier representation effectively and
efficiently, while requiring no prior knowledge of the contamination level or
the true representation dimension. Theoretically, we provide non-asymptotic
error bounds for both the learned representation and the per-task parameters.
These bounds adapt to inlier task similarity and outlier structure, and
guarantee that RAS performs at least as well as single-task learning, thus
preventing negative transfer. We also extend our framework to transfer learning
with corresponding theoretical guarantees for the target task. Extensive
experiments confirm our theory, showcasing the robustness and adaptivity of
RAS, and its superior performance in regimes with up to 80\% task
contamination.

### 7. [Neural ARFIMA model for forecasting BRIC exchange rates with long memory under oil shocks and policy uncertainties](http://arxiv.org/pdf/2509.06697v1)

Authors: Tanujit Chakraborty, Donia Besher, Madhurima Panja, Shovon Sengupta

Accurate forecasting of exchange rates remains a persistent challenge,
particularly for emerging economies such as Brazil, Russia, India, and China
(BRIC). These series exhibit long memory, nonlinearity, and non-stationarity
properties that conventional time series models struggle to capture.
Additionally, there exist several key drivers of exchange rate dynamics,
including global economic policy uncertainty, US equity market volatility, US
monetary policy uncertainty, oil price growth rates, and country-specific
short-term interest rate differentials. These empirical complexities underscore
the need for a flexible modeling framework that can jointly accommodate long
memory, nonlinearity, and the influence of external drivers. To address these
challenges, we propose a Neural AutoRegressive Fractionally Integrated Moving
Average (NARFIMA) model that combines the long-memory representation of ARFIMA
with the nonlinear learning capacity of neural networks, while flexibly
incorporating exogenous causal variables. We establish theoretical properties
of the model, including asymptotic stationarity of the NARFIMA process using
Markov chains and nonlinear time series techniques. We quantify forecast
uncertainty using conformal prediction intervals within the NARFIMA framework.
Empirical results across six forecast horizons show that NARFIMA consistently
outperforms various state-of-the-art statistical and machine learning models in
forecasting BRIC exchange rates. These findings provide new insights for
policymakers and market participants navigating volatile financial conditions.
The \texttt{narfima} \textbf{R} package provides an implementation of our
approach.

### 8. [Sequential Least-Squares Estimators with Fast Randomized Sketching for Linear Statistical Models](http://arxiv.org/pdf/2509.06856v1)

Authors: Guan-Yu Chen, Xi Yang

We propose a novel randomized framework for the estimation problem of
large-scale linear statistical models, namely Sequential Least-Squares
Estimators with Fast Randomized Sketching (SLSE-FRS), which integrates
Sketch-and-Solve and Iterative-Sketching methods for the first time. By
iteratively constructing and solving sketched least-squares (LS) subproblems
with increasing sketch sizes to achieve better precisions, SLSE-FRS gradually
refines the estimators of the true parameter vector, ultimately producing
high-precision estimators. We analyze the convergence properties of SLSE-FRS,
and provide its efficient implementation. Numerical experiments show that
SLSE-FRS outperforms the state-of-the-art methods, namely the Preconditioned
Conjugate Gradient (PCG) method, and the Iterative Double Sketching (IDS)
method.

### 9. [Learning from one graph: transductive learning guarantees via the geometry of small random worlds](http://arxiv.org/pdf/2509.06894v1)

Authors: Nils Detering, Luca Galimberti, Anastasis Kratsios, Giulia Livieri, A. Martina Neuman

Since their introduction by Kipf and Welling in $2017$, a primary use of
graph convolutional networks is transductive node classification, where missing
labels are inferred within a single observed graph and its feature matrix.
Despite the widespread use of the network model, the statistical foundations of
transductive learning remain limited, as standard inference frameworks
typically rely on multiple independent samples rather than a single graph. In
this work, we address these gaps by developing new concentration-of-measure
tools that leverage the geometric regularities of large graphs via
low-dimensional metric embeddings. The emergent regularities are captured using
a random graph model; however, the methods remain applicable to deterministic
graphs once observed. We establish two principal learning results. The first
concerns arbitrary deterministic $k$-vertex graphs, and the second addresses
random graphs that share key geometric properties with an Erd\H{o}s-R\'{e}nyi
graph $\mathbf{G}=\mathbf{G}(k,p)$ in the regime $p \in \mathcal{O}((\log
(k)/k)^{1/2})$. The first result serves as the basis for and illuminates the
second. We then extend these results to the graph convolutional network
setting, where additional challenges arise. Lastly, our learning guarantees
remain informative even with a few labelled nodes $N$ and achieve the optimal
nonparametric rate $\mathcal{O}(N^{-1/2})$ as $N$ grows.

