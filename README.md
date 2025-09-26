# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-25 17:00:26.434756 PST.

### Artificial Intelligence

### 1. [Calibrated Reasoning: An Explanatory Verifier for Dynamic and Efficient Problem-Solving](http://arxiv.org/pdf/2509.19681v1)

Authors: Anisha Garg, Engin Tekin, Yash More, David Bick, Nishit Neema, Ganesh Venkatesh

Advanced test-time computing strategies are essential for scaling reasoning
models, but their effectiveness is capped by the models' poor self-evaluation.
We propose a pairwise Explanatory Verifier, trained via reinforcement learning
(GRPO), that produces calibrated confidence scores and associated natural
language reasoning for generated solutions. Our verifier improves the accuracy
and efficiency of test-time strategies like best-of-n and self-reflection.
Crucially, it excels at identifying challenging failure modes, such as when
both candidate solutions are identically incorrect, succeeding where standard
methods like majority voting fail.

### 2. [The Conductor and the Engine: A Path Towards Co-Designed Reasoning](http://arxiv.org/pdf/2509.19762v1)

Authors: Yuanxin Wang, Pawel Filipczuk, Anisha Garg, Amaan Dhada, Mohammad Hassanpour, David Bick, Ganesh Venkatesh

Modern LLM reasoning relies on extensive test-time computation, driven by
internal model training and external agentic orchestration. However, this
synergy is often inefficient, as model verbosity and poor instruction following
lead to wasted compute. We analyze this capability-cost trade-off and introduce
an optimized reasoning workflow (\cepo) that empowers smaller open-source
models to outperform models multiple times their size. We will open-source this
workflow to enable further research. Our work demonstrates a clear path toward
co-designing orchestration frameworks with the underlying model capabilities to
unlock powerful reasoning in small-to-medium sized models.

### 3. [Analysis of approximate linear programming solution to Markov decision problem with log barrier function](http://arxiv.org/pdf/2509.19800v1)

Authors: Donghwan Lee, Hyukjun Yang, Bum Geun Park

There are two primary approaches to solving Markov decision problems (MDPs):
dynamic programming based on the Bellman equation and linear programming (LP).
Dynamic programming methods are the most widely used and form the foundation of
both classical and modern reinforcement learning (RL). By contrast, LP-based
methods have been less commonly employed, although they have recently gained
attention in contexts such as offline RL. The relative underuse of the LP-based
methods stems from the fact that it leads to an inequality-constrained
optimization problem, which is generally more challenging to solve effectively
compared with Bellman-equation-based methods. The purpose of this paper is to
establish a theoretical foundation for solving LP-based MDPs in a more
effective and practical manner. Our key idea is to leverage the log-barrier
function, widely used in inequality-constrained optimization, to transform the
LP formulation of the MDP into an unconstrained optimization problem. This
reformulation enables approximate solutions to be obtained easily via gradient
descent. While the method may appear simple, to the best of our knowledge, a
thorough theoretical interpretation of this approach has not yet been
developed. This paper aims to bridge this gap.

### 4. [LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](http://arxiv.org/pdf/2509.19839v1)

Authors: Huizhen Shu, Xuying Li, Zhuo Li

Achieving robust safety alignment in large language models (LLMs) while
preserving their utility remains a fundamental challenge. Existing approaches
often struggle to balance comprehensive safety with fine-grained
controllability at the representation level. We introduce LATENTGUARD, a novel
three-stage framework that combines behavioral alignment with supervised latent
space control for interpretable and precise safety steering. Our approach
begins by fine-tuning an LLM on rationalized datasets containing both
reasoning-enhanced refusal responses to adversarial prompts and
reasoning-enhanced normal responses to benign queries, establishing robust
behavioral priors across both safety-critical and utility-preserving scenarios.
We then train a structured variational autoencoder (VAE) on intermediate MLP
activations, supervised by multi-label annotations including attack types,
attack methods, and benign indicators. This supervision enables the VAE to
learn disentangled latent representations that capture distinct adversarial
characteristics while maintaining semantic interpretability. Through targeted
manipulation of learned latent dimensions, LATENTGUARD achieves selective
refusal behavior, effectively blocking harmful requests while preserving
helpfulness for legitimate use cases. Experiments on Qwen3-8B demonstrate
significant improvements in both safety controllability and response
interpretability without compromising utility. Cross-architecture validation on
Mistral-7B confirms the generalizability of our latent steering approach,
showing consistent effectiveness across different model families. Our results
suggest that structured representation-level intervention offers a promising
pathway toward building safer yet practical LLM systems.

### 5. [CON-QA: Privacy-Preserving QA using cloud LLMs in Contract Domain](http://arxiv.org/pdf/2509.19925v1)

Authors: Ajeet Kumar Singh, Rajsabi Surya, Anurag Tripathi, Santanu Choudhury, Sudhir Bisane

As enterprises increasingly integrate cloud-based large language models
(LLMs) such as ChatGPT and Gemini into their legal document workflows,
protecting sensitive contractual information - including Personally
Identifiable Information (PII) and commercially sensitive clauses - has emerged
as a critical challenge. In this work, we propose CON-QA, a hybrid
privacy-preserving framework designed specifically for secure question
answering over enterprise contracts, effectively combining local and
cloud-hosted LLMs. The CON-QA framework operates through three stages: (i)
semantic query decomposition and query-aware document chunk retrieval using a
locally deployed LLM analysis, (ii) anonymization of detected sensitive
entities via a structured one-to-many mapping scheme, ensuring semantic
coherence while preventing cross-session entity inference attacks, and (iii)
anonymized response generation by a cloud-based LLM, with accurate
reconstruction of the original answer locally using a session-consistent
many-to-one reverse mapping. To rigorously evaluate CON-QA, we introduce
CUAD-QA, a corpus of 85k question-answer pairs generated over 510 real-world
CUAD contract documents, encompassing simple, complex, and summarization-style
queries. Empirical evaluations, complemented by detailed human assessments,
confirm that CON-QA effectively maintains both privacy and utility, preserves
answer quality, maintains fidelity to legal clause semantics, and significantly
mitigates privacy risks, demonstrating its practical suitability for secure,
enterprise-level contract documents.

### 6. [MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM](http://arxiv.org/pdf/2509.20067v1)

Authors: Wenliang Li, Rui Yan, Xu Zhang, Li Chen, Hongji Zhu, Jing Zhao, Junjun Li, Mengru Li, Wei Cao, Zihang Jiang, Wei Wei, Kun Zhang, Shaohua Kevin Zhou

Large language models (LLMs) have demonstrated notable potential in medical
applications, yet they face substantial challenges in handling complex
real-world clinical diagnoses using conventional prompting methods. Current
prompt engineering and multi-agent approaches typically optimize isolated
inferences, neglecting the accumulation of reusable clinical experience. To
address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD)
framework, which allows LLMs to self-learn clinical knowledge via a multi-agent
pipeline that summarizes, refines, and applies diagnostic insights. It mirrors
how physicians develop expertise through experience, enabling more focused and
accurate diagnosis on key disease-specific cues. We further extend it to a
MACD-human collaborative workflow, where multiple LLM-based diagnostician
agents engage in iterative consultations, supported by an evaluator agent and
human oversight for cases where agreement is not reached. Evaluated on 4,390
real-world patient cases across seven diseases using diverse open-source LLMs
(Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves
primary diagnostic accuracy, outperforming established clinical guidelines with
gains up to 22.3% (MACD). On the subset of the data, it achieves performance on
par with or exceeding that of human physicians (up to 16% improvement over
physicians-only diagnosis). Additionally, on the MACD-human workflow, it
achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover,
self-learned knowledge exhibits strong cross-model stability, transferability,
and model-specific personalization, while the system can generate traceable
rationales, enhancing explainability. Consequently, this work presents a
scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap
between the intrinsic knowledge of LLMs and real-world clinical practice.

### 7. [From Pheromones to Policies: Reinforcement Learning for Engineered Biological Swarms](http://arxiv.org/pdf/2509.20095v1)

Authors: Aymeric Vellinger, Nemanja Antonic, Elio Tuci

Swarm intelligence emerges from decentralised interactions among simple
agents, enabling collective problem-solving. This study establishes a
theoretical equivalence between pheromone-mediated aggregation in \celeg\ and
reinforcement learning (RL), demonstrating how stigmergic signals function as
distributed reward mechanisms. We model engineered nematode swarms performing
foraging tasks, showing that pheromone dynamics mathematically mirror
cross-learning updates, a fundamental RL algorithm. Experimental validation
with data from literature confirms that our model accurately replicates
empirical \celeg\ foraging patterns under static conditions. In dynamic
environments, persistent pheromone trails create positive feedback loops that
hinder adaptation by locking swarms into obsolete choices. Through
computational experiments in multi-armed bandit scenarios, we reveal that
introducing a minority of exploratory agents insensitive to pheromones restores
collective plasticity, enabling rapid task switching. This behavioural
heterogeneity balances exploration-exploitation trade-offs, implementing
swarm-level extinction of outdated strategies. Our results demonstrate that
stigmergic systems inherently encode distributed RL processes, where
environmental signals act as external memory for collective credit assignment.
By bridging synthetic biology with swarm robotics, this work advances
programmable living systems capable of resilient decision-making in volatile
environments.

### 8. [Steerable Adversarial Scenario Generation through Test-Time Preference Alignment](http://arxiv.org/pdf/2509.20102v1)

Authors: Tong Nie, Yuewen Mei, Yihong Tang, Junlin He, Jie Sun, Haotian Shi, Wei Ma, Jian Sun

Adversarial scenario generation is a cost-effective approach for safety
assessment of autonomous driving systems. However, existing methods are often
constrained to a single, fixed trade-off between competing objectives such as
adversariality and realism. This yields behavior-specific models that cannot be
steered at inference time, lacking the efficiency and flexibility to generate
tailored scenarios for diverse training and testing requirements. In view of
this, we reframe the task of adversarial scenario generation as a
multi-objective preference alignment problem and introduce a new framework
named \textbf{S}teerable \textbf{A}dversarial scenario \textbf{GE}nerator
(SAGE). SAGE enables fine-grained test-time control over the trade-off between
adversariality and realism without any retraining. We first propose
hierarchical group-based preference optimization, a data-efficient offline
alignment method that learns to balance competing objectives by decoupling hard
feasibility constraints from soft preferences. Instead of training a fixed
model, SAGE fine-tunes two experts on opposing preferences and constructs a
continuous spectrum of policies at inference time by linearly interpolating
their weights. We provide theoretical justification for this framework through
the lens of linear mode connectivity. Extensive experiments demonstrate that
SAGE not only generates scenarios with a superior balance of adversariality and
realism but also enables more effective closed-loop training of driving
policies. Project page: https://tongnie.github.io/SAGE/.

### 9. [PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs](http://arxiv.org/pdf/2509.20105v1)

Authors: Venkat Margapuri, Garik Kazanjian, Naren Kosaraju

Large Language Models (LLMs) often struggle with maintaining coherent
multi-step reasoning traces, particularly in tasks that require a structured
logical flow. This work introduces a quantum-inspired approach to address the
challenge by incorporating a fidelity-based reward derived from Projected
Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior
approaches that use direct supervision or contrastive objectives, the proposed
method guides learning through structural consistency, offering a novel
approach to enforce global coherence in generated reasoning traces. The
proposed framework is evaluated using multiple coherence-determining metrics on
diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning
arithmetic, intuitive, and entailment-based reasoning. Results show that the
proposed quantum-inspired approach offers significant improvements over
supervised, contrastive, and pretrained baseline approaches, highlighting the
effectiveness of quantum-inspired fidelity as a foundation to improve reasoning
trace coherence in LLMs.

### 10. [Formal Verification of Minimax Algorithms](http://arxiv.org/pdf/2509.20138v1)

Authors: Wieger Wesselink, Kees Huizing, Huub van de Wetering

Using the Dafny verification system, we formally verify a range of minimax
search algorithms, including variations with alpha-beta pruning and
transposition tables. For depth-limited search with transposition tables, we
introduce a witness-based correctness criterion and apply it to two
representative algorithms. All verification artifacts, including proofs and
Python implementations, are publicly available.

### Hardware Architecture

### 1. [Open-source Stand-Alone Versatile Tensor Accelerator](http://arxiv.org/pdf/2509.19790v1)

Authors: Anthony Faure-Gignoux, Kevin Delmas, Adrien Gauffriau, Claire Pagetti

Machine Learning (ML) applications demand significant computational
resources, posing challenges for safety-critical domains like aeronautics. The
Versatile Tensor Accelerator (VTA) is a promising FPGA-based solution, but its
adoption was hindered by its dependency on the TVM compiler and by other code
non-compliant with certification requirements. This paper presents an
open-source, standalone Python compiler pipeline for the VTA, developed from
scratch and designed with certification requirements, modularity, and
extensibility in mind. The compiler's effectiveness is demonstrated by
compiling and executing LeNet-5 Convolutional Neural Network (CNN) using the
VTA simulators, and preliminary results indicate a strong potential for scaling
its capabilities to larger CNN architectures. All contributions are publicly
available.

### 2. [SpecMamba: Accelerating Mamba Inference on FPGA with Speculative Decoding](http://arxiv.org/pdf/2509.19873v1)

Authors: Linfeng Zhong, Songqiang Xu, Huifeng Wen, Tong Xie, Qingyu Guo, Yuan Wang, Meng Li

The growing demand for efficient long-sequence modeling on edge devices has
propelled widespread adoption of State Space Models (SSMs) like Mamba, due to
their superior computational efficiency and scalability. As its autoregressive
generation process remains memory-bound, speculative decoding has been proposed
that incorporates draft model generation and target model verification.
However, directly applying speculative decoding to SSMs faces three key
challenges: (1) hidden state backtracking difficulties, (2) tree-based parallel
verification incompatibility, and (3) hardware workload mismatch. To address
these challenges, we propose SpecMamba, the first FPGA-based accelerator for
Mamba with speculative decoding, which features system, algorithm, and hardware
co-design. At the system level, we present a memory-aware hybrid backtracking
strategy to coordinate both models. At the algorithm level, we propose
first-in-first-out (FIFO)-based tree verification with tiling to minimize
memory access. At the hardware level, we customize a dataflow that computes
linear layers in parallel and SSM layers in series to enable maximal
overlapping. Implemented on AMD FPGA platforms (VHK158 and VCK190), SpecMamba
achieves a 2.27x speedup over GPU baselines and a 2.85x improvement compared to
prior FPGA solutions, while demonstrating 5.41x and 1.26x higher energy
efficiency, respectively.

### 3. [OpenGL GPU-Based Rowhammer Attack (Work in Progress)](http://arxiv.org/pdf/2509.19959v1)

Authors: Antoine Plin, Frédéric Fauberteau, Nga Nguyen

Rowhammer attacks have emerged as a significant threat to modern DRAM-based
memory systems, leveraging frequent memory accesses to induce bit flips in
adjacent memory cells. This work-in-progress paper presents an adaptive,
many-sided Rowhammer attack utilizing GPU compute shaders to systematically
achieve high-frequency memory access patterns. Our approach employs statistical
distributions to optimize row targeting and avoid current mitigations. The
methodology involves initializing memory with known patterns, iteratively
hammering victim rows, monitoring for induced errors, and dynamically adjusting
parameters to maximize success rates. The proposed attack exploits the parallel
processing capabilities of GPUs to accelerate hammering operations, thereby
increasing the probability of successful bit flips within a constrained
timeframe. By leveraging OpenGL compute shaders, our implementation achieves
highly efficient row hammering with minimal software overhead. Experimental
results on a Raspberry Pi 4 demonstrate that the GPU-based approach attains a
high rate of bit flips compared to traditional CPU-based hammering, confirming
its effectiveness in compromising DRAM integrity. Our findings align with
existing research on microarchitectural attacks in heterogeneous systems that
highlight the susceptibility of GPUs to security vulnerabilities. This study
contributes to the understanding of GPU-assisted fault-injection attacks and
underscores the need for improved mitigation strategies in future memory
architectures.

### 4. [Automated Multi-Agent Workflows for RTL Design](http://arxiv.org/pdf/2509.20182v1)

Authors: Amulya Bhattaram, Janani Ramamoorthy, Ranit Gupta, Diana Marculescu, Dimitrios Stamoulis

The rise of agentic AI workflows unlocks novel opportunities for computer
systems design and optimization. However, for specialized domains such as
program synthesis, the relative scarcity of HDL and proprietary EDA resources
online compared to more common programming tasks introduces challenges, often
necessitating task-specific fine-tuning, high inference costs, and
manually-crafted agent orchestration. In this work, we present VeriMaAS, a
multi-agent framework designed to automatically compose agentic workflows for
RTL code generation. Our key insight is to integrate formal verification
feedback from HDL tools directly into workflow generation, reducing the cost of
gradient-based updates or prolonged reasoning traces. Our method improves
synthesis performance by 5-7% for pass@k over fine-tuned baselines, while
requiring only a few hundred training examples, representing an
order-of-magnitude reduction in supervision cost.

### 5. [The Cream Rises to the Top: Efficient Reranking Method for Verilog Code Generation](http://arxiv.org/pdf/2509.20215v1)

Authors: Guang Yang, Wei Zheng, Xiang Chen, Yifan Sun, Fengji Zhang, Terry Yue Zhuo

LLMs face significant challenges in Verilog generation due to limited
domain-specific knowledge. While sampling techniques improve pass@k metrics,
hardware engineers need one trustworthy solution rather than uncertain
candidates. To bridge this gap, we formulate it as a semantic alignment problem
between requirements and Verilog implementations, and propose VCD-RNK, a
discriminator model tailored for efficient Verilog code reranking.
Specifically, VCD-RNKincorporates Verilog-specific reasoning by distilling
expert knowledge across three dimensions: code semantic analysis, test case
generation, and functional correctness assessment. By explicitly simulating the
above reasoning processes during inference, VCD-RNK effectively avoids
computationally intensive test execution in existing methods.

### 6. [Design Insights and Comparative Evaluation of a Hardware-Based Cooperative Perception Architecture for Lane Change Prediction](http://arxiv.org/pdf/2509.20218v1)

Authors: Mohamed Manzour, Catherine M. Elias, Omar M. Shehata, Rubén Izquierdo, Miguel Ángel Sotelo

Research on lane change prediction has gained attention in the last few
years. Most existing works in this area have been conducted in simulation
environments or with pre-recorded datasets, these works often rely on
simplified assumptions about sensing, communication, and traffic behavior that
do not always hold in practice. Real-world deployments of lane-change
prediction systems are relatively rare, and when they are reported, the
practical challenges, limitations, and lessons learned are often
under-documented. This study explores cooperative lane-change prediction
through a real hardware deployment in mixed traffic and shares the insights
that emerged during implementation and testing. We highlight the practical
challenges we faced, including bottlenecks, reliability issues, and operational
constraints that shaped the behavior of the system. By documenting these
experiences, the study provides guidance for others working on similar
pipelines.

### 7. [Digital Signal Processing from Classical Coherent Systems to Continuous-Variable QKD: A Review of Cross-Domain Techniques, Applications, and Challenges](http://arxiv.org/pdf/2509.20141v1)

Authors: Davi Juvêncio Gomes de Sousa, Caroline da Silva Morais Alves, Valéria Loureiro da Silva, Nelson Alves Ferreira Neto

This systematic review investigates the application of digital signal
processing (DSP) techniques -- originally developed for coherent optical
communication systems to continuous-variable quantum key distribution (CV-QKD).
The convergence of these domains has enabled significant advances in CV-QKD
performance, particularly in phase synchronization, polarization tracking, and
excess noise mitigation. To provide a comprehensive and reproducible synthesis
of this emerging field, we employed the APISSER methodology, a task-oriented
framework adapted from the PRISMA protocol. A structured search across IEEE
Xplore and Web of Science databases (2021-2025) yielded 220 relevant
publications, which were screened, classified, and analyzed to address six
research questions. Our findings highlight that many classical DSP algorithms,
such as Kalman filtering, carrier recovery, adaptive equalization, and
machine-learning-assisted signal estimation, have been successfully adapted to
the quantum regime, often requiring modifications to meet security and noise
constraints. We also identify a range of recent DSP innovations in coherent
optical communication systems with high potential for future CV-QKD
integration, including neural equalization, probabilistic shaping, and joint
retiming-equalization filters. Despite these advances, challenges remain in
achieving robust phase tracking under ultra-low Signal-to-Noise Ratio (SNR)
conditions, real-time polarization compensation, and secure co-existence with
classical channels. This review maps current trends, technical barriers, and
emerging opportunities at the intersection of signal processing for quantum and
classical communication, supporting the development of scalable and resilient
CV-QKD systems.

### Computational Complexity

### 1. [Nonlocal Games and Self-tests in the Presence of Noise](http://arxiv.org/pdf/2509.20350v1)

Authors: Honghao Fu, Minglong Qin, Haochen Xu, Penghui Yao

Self-testing is a key characteristic of certain nonlocal games, which allow
one to uniquely determine the underlying quantum state and measurement
operators used by the players, based solely on their observed input-output
correlations [MY04]. Motivated by the limitations of current quantum devices,
we study self-testing in the high-noise regime, where the two players are
restricted to sharing many copies of a noisy entangled state with an arbitrary
constant noise rate. In this setting, many existing self-tests fail to certify
any nontrivial structure. We first characterize the maximal winning
probabilities of the CHSH game [CHSH69], the Magic Square game [Mer90a], and
the 2-out-of-n CHSH game [CRSV18] as functions of the noise rate, under the
assumption that players use traceless observables. These results enable the
construction of device-independent protocols for estimating the noise rate.
Building on this analysis, we show that these three games--together with an
additional test enforcing the tracelessness of binary observables--can
self-test one, two, and n pairs of anticommuting Pauli operators, respectively.
These are the first known self-tests that are robust in the high-noise regime
and remain sound even when the players' measurements are noisy. Our proofs rely
on Sum-of-Squares (SoS) decompositions and Pauli analysis techniques developed
in the contexts of quantum proof systems and quantum learning theory.

### 2. [Dequantization and Hardness of Spectral Sum Estimation](http://arxiv.org/pdf/2509.20183v1)

Authors: Roman Edenhofer, Atsuya Hasegawa, François Le Gall

We give new dequantization and hardness results for estimating spectral sums
of matrices, such as the log-determinant. Recent quantum algorithms have
demonstrated that the logarithm of the determinant of sparse, well-conditioned,
positive matrices can be approximated to $\varepsilon$-relative accuracy in
time polylogarithmic in the dimension $N$, specifically in time
$\mathrm{poly}(\mathrm{log}(N), s, \kappa, 1/\varepsilon)$, where $s$ is the
sparsity and $\kappa$ the condition number of the input matrix. We provide a
simple dequantization of these techniques that preserves the polylogarithmic
dependence on the dimension. Our classical algorithm runs in time
$\mathrm{polylog}(N)\cdot s^{O(\sqrt{\kappa}\log \kappa/\varepsilon)}$ which
constitutes an exponential improvement over previous classical algorithms in
certain parameter regimes.
  We complement our classical upper bound with $\mathsf{DQC1}$-completeness
results for estimating specific spectral sums such as the trace of the inverse
and the trace of matrix powers for log-local Hamiltonians, with parameter
scalings analogous to those of known quantum algorithms. Assuming
$\mathsf{BPP}\subsetneq\mathsf{DQC1}$, this rules out classical algorithms with
the same scalings. It also resolves a main open problem of Cade and Montanaro
(TQC 2018) concerning the complexity of Schatten-$p$ norm estimation. We
further analyze a block-encoding input model, where instead of a classical
description of a sparse matrix, we are given a block-encoding of it. We show
$\mathsf{DQC}1$-completeness in a very general way in this model for estimating
$\mathrm{tr}[f(A)]$ whenever $f$ and $f^{-1}$ are sufficiently smooth.
  We conclude our work with $\mathsf{BQP}$-hardness and
$\mathsf{PP}$-completeness results for high-accuracy log-determinant
estimation.

### Computational Engineering

### 1. [Characterizing failure morphologies in fiber-reinforced composites via k-means clustering based multiscale framework](http://arxiv.org/pdf/2509.20011v1)

Authors: Harpreet Singh

A novel homogenization methodology is proposed for analyzing the failure of
fiber-reinforced composite materials, utilizing elastic and eigen influence
tensors within a damage informed transformation field analysis (D-TFA)
framework. This approach includes a technique for calculating macroscopic
damage under uniform stress and strain conditions, offering more realistic
simulations. Computational efficiency is enhanced through a reduced-order
modeling strategy, while elastic and eigen strain distribution driven k-means
clustering methods are employed to partition the microscale domain. The model's
performance is assessed by simulating the response of a representative volume
element (RVE) treated as a homogenized continuum. Subsequently, a comparative
assessment is carried out to check the efficacy of two clustering schemes.
Damage morphologies are calculated using proposed framework and compared with
predictions obtained using finite element method. Furthermore, open-hole
specimen tests are simulated and failure paths are predicted for the domains
with different fiber layups. Ultimately, we show that D-TFA can accurately
capture damage patterns and directional strengths, providing improved
predictions of the mechanical behavior of composite materials. It has been
demonstrated that higher cluster counts are crucial for capturing a more
accurate stress-strain response, especially for complex microstructures.

### 2. [Efficient Multi-Objective Constrained Bayesian Optimization of Bridge Girder](http://arxiv.org/pdf/2509.20161v1)

Authors: Heine Havneraas Røstum, Joseph Morlier, Sebastien Gros, Ketil Aas-Jakobsen

The buildings and construction sector is a significant source of greenhouse
gas emissions, with cement production alone contributing 7~\% of global
emissions and the industry as a whole accounting for approximately 37~\%.
Reducing emissions by optimizing structural design can achieve significant
global benefits. This article introduces an efficient multi-objective
constrained Bayesian optimization approach to address this challenge. Rather
than attempting to determine the full set of non-dominated solutions with
arbitrary trade-offs, the approach searches for a solution matching a specified
trade-off. Structural design is typically conducted using computationally
expensive finite element simulations, whereas Bayesian optimization offers an
efficient approach for optimizing problems that involve such high-cost
simulations. The proposed method integrates proper orthogonal decomposition for
dimensionality reduction of simulation results with Kriging partial least
squares to enhance efficiency. Constrained expected improvement is used as an
acquisition function for Bayesian optimization. The approach is demonstrated
through a case study of a two-lane, three-span post-tensioned concrete bridge
girder, incorporating fifteen design variables and nine constraints. A
comparison with conventional design methods demonstrates the potential of this
optimization approach to achieve substantial cost reductions, with savings of
approximately 10\% to 15\% in financial costs and about 20\% in environmental
costs for the case study, while ensuring structural integrity.

### 3. [Enabling Multi-Species Bird Classification on Low-Power Bioacoustic Loggers](http://arxiv.org/pdf/2509.20103v1)

Authors: Stefano Ciapponi, Leonardo Mannini, Jarek Scanferla, Matteo Anderle, Elisabetta Farella

This paper introduces WrenNet, an efficient neural network enabling real-time
multi-species bird audio classification on low-power microcontrollers for
scalable biodiversity monitoring. We propose a semi-learnable spectral feature
extractor that adapts to avian vocalizations, outperforming standard mel-scale
and fully-learnable alternatives. On an expert-curated 70-species dataset,
WrenNet achieves up to 90.8\% accuracy on acoustically distinctive species and
70.1\% on the full task. When deployed on an AudioMoth device ($\leq$1MB RAM),
it consumes only 77mJ per inference. Moreover, the proposed model is over 16x
more energy-efficient compared to Birdnet when running on a Raspberry Pi 3B+.
This work demonstrates the first practical framework for continuous,
multi-species acoustic monitoring on low-power edge devices.

### 4. [An Overview of Meshfree Collocation Methods](http://arxiv.org/pdf/2509.20056v1)

Authors: Tomas Halada, Serhii Yaskovets, Abhinav Singh, Ludek Benes, Pratik Suchde, Ivo F. Sbalzarini

We provide a comprehensive overview of meshfree collocation methods for
numerically approximating differential operators on continuously labeled
unstructured point clouds. Meshfree collocation methods do not require a
computational grid or mesh. Instead, they approximate smooth functions and
their derivatives at potentially irregularly distributed collocation points,
often called particles, to a desired order of consistency. We review several
meshfree collocation methods from the literature, trace the historical
development of key concepts, and propose a classification of methods according
to their principle of derivation. Although some of the methods reviewed are
similar or identical, there are subtle yet important differences between many,
which we highlight and discuss. We present a unifying formulation of meshfree
collocation methods that renders these differences apparent and show how each
method can be derived from this formulation. Finally, we propose a generalized
derivation for meshfree collocation methods going forward.

### Computational Geometry

### 1. [MeshMosaic: Scaling Artist Mesh Generation via Local-to-Global Assembly](http://arxiv.org/pdf/2509.19995v1)

Authors: Rui Xu, Tianyang Xue, Qiujie Dong, Le Wan, Zhe Zhu, Peng Li, Zhiyang Dou, Cheng Lin, Shiqing Xin, Yuan Liu, Wenping Wang, Taku Komura

Scaling artist-designed meshes to high triangle numbers remains challenging
for autoregressive generative models. Existing transformer-based methods suffer
from long-sequence bottlenecks and limited quantization resolution, primarily
due to the large number of tokens required and constrained quantization
granularity. These issues prevent faithful reproduction of fine geometric
details and structured density patterns. We introduce MeshMosaic, a novel
local-to-global framework for artist mesh generation that scales to over 100K
triangles--substantially surpassing prior methods, which typically handle only
around 8K faces. MeshMosaic first segments shapes into patches, generating each
patch autoregressively and leveraging shared boundary conditions to promote
coherence, symmetry, and seamless connectivity between neighboring regions.
This strategy enhances scalability to high-resolution meshes by quantizing
patches individually, resulting in more symmetrical and organized mesh density
and structure. Extensive experiments across multiple public datasets
demonstrate that MeshMosaic significantly outperforms state-of-the-art methods
in both geometric fidelity and user preference, supporting superior detail
representation and practical mesh generation for real-world applications.

### Computation and Language

### 1. [Personality Vector: Modulating Personality of Large Language Models by Model Merging](http://arxiv.org/pdf/2509.19727v1)

Authors: Seungjong Sun, Seo Yeon Baek, Jang Hyun Kim

Driven by the demand for personalized AI systems, there is growing interest
in aligning the behavior of large language models (LLMs) with human traits such
as personality. Previous attempts to induce personality in LLMs have shown
promising results, but they struggle to capture the continuous and
multidimensional nature of human traits. In this work, we propose a novel
method for personality modulation in LLMs via model merging. Specifically, we
construct personality vectors by subtracting the weights of a pre-trained model
from those of the fine-tuned model on a given personality trait. By merging
personality vectors, we enable LLMs to exhibit desired personality traits
without additional training. Extensive experiments show that personality
vectors enable continuous control over trait intensity and support the
composition of multiple traits. Furthermore, personality vectors transfer
across diverse downstream models, suggesting that they encode generalizable
representations of personality. Our code is available at here.

### 2. [EnAnchored-X2X: English-Anchored Optimization for Many-to-Many Translation](http://arxiv.org/pdf/2509.19770v1)

Authors: Sen Yang, Yu Bao, Yu Lu, Jiajun Chen, Shujian Huang, Shanbo Cheng

Large language models (LLMs) have demonstrated strong machine translation
capabilities for English-centric language pairs but underperform in direct
non-English (x2x) translation. This work addresses this limitation through a
synthetic data generation framework that leverages models' established
English-to-x (en2x) capabilities. By extending English parallel corpora into
omnidirectional datasets and developing an English-referenced quality
evaluation proxy, we enable effective collection of high-quality x2x training
data. Combined with preference-based optimization, our method achieves
significant improvement across 72 x2x directions for widely used LLMs, while
generalizing to enhance en2x performance. The results demonstrate that
strategic exploitation of English-centric strengths can bootstrap comprehensive
multilingual translation capabilities in LLMs. We release codes, datasets, and
model checkpoints at https://github.com/NJUNLP/EAX

### 3. [Mahānāma: A Unique Testbed for Literary Entity Discovery and Linking](http://arxiv.org/pdf/2509.19844v1)

Authors: Sujoy Sarkar, Gourav Sarkar, Manoj Balaji Jagadeeshan, Jivnesh Sandhan, Amrith Krishna, Pawan Goyal

High lexical variation, ambiguous references, and long-range dependencies
make entity resolution in literary texts particularly challenging. We present
Mah\={a}n\={a}ma, the first large-scale dataset for end-to-end Entity Discovery
and Linking (EDL) in Sanskrit, a morphologically rich and under-resourced
language. Derived from the Mah\={a}bh\={a}rata, the world's longest epic, the
dataset comprises over 109K named entity mentions mapped to 5.5K unique
entities, and is aligned with an English knowledge base to support
cross-lingual linking. The complex narrative structure of Mah\={a}n\={a}ma,
coupled with extensive name variation and ambiguity, poses significant
challenges to resolution systems. Our evaluation reveals that current
coreference and entity linking models struggle when evaluated on the global
context of the test set. These results highlight the limitations of current
approaches in resolving entities within such complex discourse. Mah\=an\=ama
thus provides a unique benchmark for advancing entity resolution, especially in
literary domains.

### 4. [Benchmarking Gaslighting Attacks Against Speech Large Language Models](http://arxiv.org/pdf/2509.19858v1)

Authors: Jinyang Wu, Bin Zhu, Xiandong Zou, Qiquan Zhang, Xu Fang, Pan Zhou

As Speech Large Language Models (Speech LLMs) become increasingly integrated
into voice-based applications, ensuring their robustness against manipulative
or adversarial input becomes critical. Although prior work has studied
adversarial attacks in text-based LLMs and vision-language models, the unique
cognitive and perceptual challenges of speech-based interaction remain
underexplored. In contrast, speech presents inherent ambiguity, continuity, and
perceptual diversity, which make adversarial attacks more difficult to detect.
In this paper, we introduce gaslighting attacks, strategically crafted prompts
designed to mislead, override, or distort model reasoning as a means to
evaluate the vulnerability of Speech LLMs. Specifically, we construct five
manipulation strategies: Anger, Cognitive Disruption, Sarcasm, Implicit, and
Professional Negation, designed to test model robustness across varied tasks.
It is worth noting that our framework captures both performance degradation and
behavioral responses, including unsolicited apologies and refusals, to diagnose
different dimensions of susceptibility. Moreover, acoustic perturbation
experiments are conducted to assess multi-modal robustness. To quantify model
vulnerability, comprehensive evaluation across 5 Speech and multi-modal LLMs on
over 10,000 test samples from 5 diverse datasets reveals an average accuracy
drop of 24.3% under the five gaslighting attacks, indicating significant
behavioral vulnerability. These findings highlight the need for more resilient
and trustworthy speech-based AI systems.

### 5. [SINAI at eRisk@CLEF 2025: Transformer-Based and Conversational Strategies for Depression Detection](http://arxiv.org/pdf/2509.19861v1)

Authors: Alba Maria Marmol-Romero, Manuel Garcia-Vega, Miguel Angel Garcia-Cumbreras, Arturo Montejo-Raez

This paper describes the participation of the SINAI-UJA team in the
eRisk@CLEF 2025 lab. Specifically, we addressed two of the proposed tasks: (i)
Task 2: Contextualized Early Detection of Depression, and (ii) Pilot Task:
Conversational Depression Detection via LLMs. Our approach for Task 2 combines
an extensive preprocessing pipeline with the use of several transformer-based
models, such as RoBERTa Base or MentalRoBERTA Large, to capture the contextual
and sequential nature of multi-user conversations. For the Pilot Task, we
designed a set of conversational strategies to interact with LLM-powered
personas, focusing on maximizing information gain within a limited number of
dialogue turns. In Task 2, our system ranked 8th out of 12 participating teams
based on F1 score. However, a deeper analysis revealed that our models were
among the fastest in issuing early predictions, which is a critical factor in
real-world deployment scenarios. This highlights the trade-off between early
detection and classification accuracy, suggesting potential avenues for
optimizing both jointly in future work. In the Pilot Task, we achieved 1st
place out of 5 teams, obtaining the best overall performance across all
evaluation metrics: DCHR, ADODL and ASHR. Our success in this task demonstrates
the effectiveness of structured conversational design when combined with
powerful language models, reinforcing the feasibility of deploying LLMs in
sensitive mental health assessment contexts.

### 6. [SwissGPC v1.0 -- The Swiss German Podcasts Corpus](http://arxiv.org/pdf/2509.19866v1)

Authors: Samuel Stucki, Mark Cieliebak, Jan Deriu

We present SwissGPC v1.0, the first mid-to-large-scale corpus of spontaneous
Swiss German speech, developed to support research in ASR, TTS, dialect
identification, and related fields. The dataset consists of links to talk shows
and podcasts hosted on Schweizer Radio und Fernsehen and YouTube, which contain
approximately 5400 hours of raw audio. After segmentation and weak annotation,
nearly 5000 hours of speech were retained, covering the seven major Swiss
German dialect regions alongside Standard German. We describe the corpus
construction methodology, including an automated annotation pipeline, and
provide statistics on dialect distribution, token counts, and segmentation
characteristics. Unlike existing Swiss German speech corpora, which primarily
feature controlled speech, this corpus captures natural, spontaneous
conversations, making it a valuable resource for real-world speech
applications.

### 7. [Future Policy Aware Preference Learning for Mathematical Reasoning](http://arxiv.org/pdf/2509.19893v1)

Authors: Minjae Oh, Yunho Choi, Dongmin Choi, Yohan Jo

Preference learning methods such as Direct Preference Optimization (DPO) have
become standard for Large Language Model (LLM) post-training, yet they are
often ineffective for mathematical reasoning. A key challenge is the large
token overlap between preferred and dispreferred trajectories; lowering the
probability of dispreferred trajectories also reduces the probability of shared
useful tokens, leading to over-penalization and overall performance collapse.
As a mitigation, existing algorithms include the probability of a trajectory
under the current policy as a regularization term, which decreases the effect
of the gradient when the probability is low. However, by the time this effect
takes hold, useful tokens may have already been over-penalized as the model has
begun to degrade. To address this, we propose Future Policy Aware (FPA)
preference learning, which replaces the current policy with a future policy in
the regularization term. This future policy is estimated via lightweight,
logit-space extrapolation from a reference model toward the current model. FPA
enables safer training by preemptively regularizing potentially problematic
gradients. We apply FPA to DPO, RPO, and SimPER and evaluate them on the MATH
and GSM8K benchmarks. FPA yields consistent performance gains, with the largest
improvements observed with SimPER, achieving gains of up to 5.75%. We
demonstrate that FPA provides proactive regularization while preserving the
probability of shared, useful mathematical tokens, and enables longer,
degradation-free training with negligible computational overhead. We will
release our code publicly upon publication.

### 8. [WEST: LLM based Speech Toolkit for Speech Understanding, Generation, and Interaction](http://arxiv.org/pdf/2509.19902v1)

Authors: Binbin Zhang, Chengdong Liang, Shuai Wang, Xuelong Geng, Zhao Guo, Haoyu Li, Hao Yin, Xipeng Yang, Pengshen Zhang, Changwei Ma, Lei Xie

In this paper, we present WEST(WE Speech Toolkit), a speech toolkit based on
a large language model (LLM) for speech understanding, generation, and
interaction. There are three key features of WEST: 1) Fully LLM-based: Standing
on the shoulders of giants by reusing mature architectures, ecosystems (e.g.,
Hugging Face), and methods (e.g., sequence packing) from large models. 2)
Full-stack: Supports tasks such as recognition, synthesis, understanding,
dialogue, and multimodal capabilities, with extensibility to incorporate
open-source models. 3) Simple and Stupid: A simple and stupid speech toolkit
that everyone can Touch. In addition, WEST provides two types of recipes,
models, and experimental results. The first is entirely based on open-source
models and open-source data, allowing users to fully reproduce the experiments
in this paper and serving as a verification system or minimal system baseline.
The second is trained on massive data, offering superior performance so the
user can directly apply it out of the box. WEST is publicly avilable at
https://github.com/wenet-e2e/west/

### 9. [DiffNator: Generating Structured Explanations of Time-Series Differences](http://arxiv.org/pdf/2509.20007v1)

Authors: Kota Dohi, Tomoya Nishida, Harsh Purohit, Takashi Endo, Yohei Kawaguchi

In many IoT applications, the central interest lies not in individual sensor
signals but in their differences, yet interpreting such differences requires
expert knowledge. We propose DiffNator, a framework for structured explanations
of differences between two time series. We first design a JSON schema that
captures the essential properties of such differences. Using the Time-series
Observations of Real-world IoT (TORI) dataset, we generate paired sequences and
train a model that combine a time-series encoder with a frozen LLM to output
JSON-formatted explanations. Experimental results show that DiffNator generates
accurate difference explanations and substantially outperforms both a visual
question answering (VQA) baseline and a retrieval method using a pre-trained
time-series encoder.

### 10. [From Input Perception to Predictive Insight: Modeling Model Blind Spots Before They Become Errors](http://arxiv.org/pdf/2509.20065v1)

Authors: Maggie Mi, Aline Villavicencio, Nafise Sadat Moosavi

Language models often struggle with idiomatic, figurative, or
context-sensitive inputs, not because they produce flawed outputs, but because
they misinterpret the input from the outset. We propose an input-only method
for anticipating such failures using token-level likelihood features inspired
by surprisal and the Uniform Information Density hypothesis. These features
capture localized uncertainty in input comprehension and outperform standard
baselines across five linguistically challenging datasets. We show that
span-localized features improve error detection for larger models, while
smaller models benefit from global patterns. Our method requires no access to
outputs or hidden activations, offering a lightweight and generalizable
approach to pre-generation error prediction.

### Cryptography and Security

### 1. [Unmasking Fake Careers: Detecting Machine-Generated Career Trajectories via Multi-layer Heterogeneous Graphs](http://arxiv.org/pdf/2509.19677v1)

Authors: Michiharu Yamashita, Thanh Tran, Delvin Ce Zhang, Dongwon Lee

The rapid advancement of Large Language Models (LLMs) has enabled the
generation of highly realistic synthetic data. We identify a new vulnerability,
LLMs generating convincing career trajectories in fake resumes and explore
effective detection methods. To address this challenge, we construct a dataset
of machine-generated career trajectories using LLMs and various methods, and
demonstrate that conventional text-based detectors perform poorly on structured
career data. We propose CareerScape, a novel heterogeneous, hierarchical
multi-layer graph framework that models career entities and their relations in
a unified global graph built from genuine resumes. Unlike conventional
classifiers that treat each instance independently, CareerScape employs a
structure-aware framework that augments user-specific subgraphs with trusted
neighborhood information from a global graph, enabling the model to capture
both global structural patterns and local inconsistencies indicative of
synthetic career paths. Experimental results show that CareerScape outperforms
state-of-the-art baselines by 5.8-85.0% relatively, highlighting the importance
of structure-aware detection for machine-generated content.

### 2. [chainScale: Secure Functionality-oriented Scalability for Decentralized Resource Markets](http://arxiv.org/pdf/2509.20356v1)

Authors: Mohamed E. Najd, Ghada Almashaqbeh

Decentralized resource markets are Web 3.0 applications that build
open-access platforms for trading digital resources among users without any
central management. They promise cost reduction, transparency, and flexible
service provision. However, these markets usually have large workload that must
be processed in a timely manner, leading to serious scalability problems.
Despite the large amount of work on blockchain scalability, existing solutions
are ineffective as they do not account for these markets' work models and
traffic patterns.
  We introduce chainScale, a secure hybrid sidechain-sharding solution that
aims to boost throughput of decentralized resource markets and reduce their
latency and storage footprint. At its core, chainScale leverages dependent
sidechains and functionality-oriented workload splitting to parallelize traffic
processing by having each market module assigned to a sidechain. Different from
sharding, chainScale does not incur any cross-sidechain transactions that tend
to be costly. chainScale introduces several techniques, including hierarchical
workload sharing that further sub-divides overloaded modules, and weighted
miner assignment that assigns miners with vested interest in the system to
critical modules' sidechains. Furthermore, chainScale employs sidechain syncing
to maintain the mainchain as the single truth of system state, and pruning to
discard stale records. Beside analyzing security, we build a proof-of-concept
implementation for a distributed file storage market as a use case. Our
experiments show that, compared to a single sidechain-based prior solution,
chainScale boosts throughput by 4x and reduces confirmation latency by 5x.
Also, they show that chainScale outperforms sharding by 2.5x in throughput and
3.5x in latency.

### 3. [FlyTrap: Physical Distance-Pulling Attack Towards Camera-based Autonomous Target Tracking Systems](http://arxiv.org/pdf/2509.20362v1)

Authors: Shaoyuan Xie, Mohamad Habib Fakih, Junchi Lu, Fayzah Alshammari, Ningfei Wang, Takami Sato, Halima Bouzidi, Mohammad Abdullah Al Faruque, Qi Alfred Chen

Autonomous Target Tracking (ATT) systems, especially ATT drones, are widely
used in applications such as surveillance, border control, and law enforcement,
while also being misused in stalking and destructive actions. Thus, the
security of ATT is highly critical for real-world applications. Under the
scope, we present a new type of attack: distance-pulling attacks (DPA) and a
systematic study of it, which exploits vulnerabilities in ATT systems to
dangerously reduce tracking distances, leading to drone capturing, increased
susceptibility to sensor attacks, or even physical collisions. To achieve these
goals, we present FlyTrap, a novel physical-world attack framework that employs
an adversarial umbrella as a deployable and domain-specific attack vector.
FlyTrap is specifically designed to meet key desired objectives in attacking
ATT drones: physical deployability, closed-loop effectiveness, and
spatial-temporal consistency. Through novel progressive distance-pulling
strategy and controllable spatial-temporal consistency designs, FlyTrap
manipulates ATT drones in real-world setups to achieve significant system-level
impacts. Our evaluations include new datasets, metrics, and closed-loop
experiments on real-world white-box and even commercial ATT drones, including
DJI and HoverAir. Results demonstrate FlyTrap's ability to reduce tracking
distances within the range to be captured, sensor attacked, or even directly
crashed, highlighting urgent security risks and practical implications for the
safe deployment of ATT systems.

### 4. [A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers](http://arxiv.org/pdf/2509.19947v1)

Authors: Zhixiao Wu, Yao Lu, Jie Wen, Hao Sun, Qi Zhou, Guangming Lu

Poison-only Clean-label Backdoor Attacks aim to covertly inject
attacker-desired behavior into DNNs by merely poisoning the dataset without
changing the labels. To effectively implant a backdoor, multiple
\textbf{triggers} are proposed for various attack requirements of Attack
Success Rate (ASR) and stealthiness. Additionally, sample selection enhances
clean-label backdoor attacks' ASR by meticulously selecting ``hard'' samples
instead of random samples to poison. Current methods 1) usually handle the
sample selection and triggers in isolation, leading to severely limited
improvements on both ASR and stealthiness. Consequently, attacks exhibit
unsatisfactory performance on evaluation metrics when converted to PCBAs via a
mere stacking of methods. Therefore, we seek to explore the bidirectional
collaborative relations between the sample selection and triggers to address
the above dilemma. 2) Since the strong specificity within triggers, the simple
combination of sample selection and triggers fails to substantially enhance
both evaluation metrics, with generalization preserved among various attacks.
Therefore, we seek to propose a set of components to significantly improve both
stealthiness and ASR based on the commonalities of attacks. Specifically,
Component A ascertains two critical selection factors, and then makes them an
appropriate combination based on the trigger scale to select more reasonable
``hard'' samples for improving ASR. Component B is proposed to select samples
with similarities to relevant trigger implanted samples to promote
stealthiness. Component C reassigns trigger poisoning intensity on RGB colors
through distinct sensitivity of the human visual system to RGB for higher ASR,
with stealthiness ensured by sample selection, including Component B.
Furthermore, all components can be strategically integrated into diverse PCBAs.

### 5. [OpenGL GPU-Based Rowhammer Attack (Work in Progress)](http://arxiv.org/pdf/2509.19959v1)

Authors: Antoine Plin, Frédéric Fauberteau, Nga Nguyen

Rowhammer attacks have emerged as a significant threat to modern DRAM-based
memory systems, leveraging frequent memory accesses to induce bit flips in
adjacent memory cells. This work-in-progress paper presents an adaptive,
many-sided Rowhammer attack utilizing GPU compute shaders to systematically
achieve high-frequency memory access patterns. Our approach employs statistical
distributions to optimize row targeting and avoid current mitigations. The
methodology involves initializing memory with known patterns, iteratively
hammering victim rows, monitoring for induced errors, and dynamically adjusting
parameters to maximize success rates. The proposed attack exploits the parallel
processing capabilities of GPUs to accelerate hammering operations, thereby
increasing the probability of successful bit flips within a constrained
timeframe. By leveraging OpenGL compute shaders, our implementation achieves
highly efficient row hammering with minimal software overhead. Experimental
results on a Raspberry Pi 4 demonstrate that the GPU-based approach attains a
high rate of bit flips compared to traditional CPU-based hammering, confirming
its effectiveness in compromising DRAM integrity. Our findings align with
existing research on microarchitectural attacks in heterogeneous systems that
highlight the susceptibility of GPUs to security vulnerabilities. This study
contributes to the understanding of GPU-assisted fault-injection attacks and
underscores the need for improved mitigation strategies in future memory
architectures.

### 6. [Learning Robust Penetration-Testing Policies under Partial Observability: A systematic evaluation](http://arxiv.org/pdf/2509.20008v1)

Authors: Raphael Simon, Pieter Libin, Wim Mees

Penetration testing, the simulation of cyberattacks to identify security
vulnerabilities, presents a sequential decision-making problem well-suited for
reinforcement learning (RL) automation. Like many applications of RL to
real-world problems, partial observability presents a major challenge, as it
invalidates the Markov property present in Markov Decision Processes (MDPs).
Partially Observable MDPs require history aggregation or belief state
estimation to learn successful policies. We investigate stochastic, partially
observable penetration testing scenarios over host networks of varying size,
aiming to better reflect real-world complexity through more challenging and
representative benchmarks. This approach leads to the development of more
robust and transferable policies, which are crucial for ensuring reliable
performance across diverse and unpredictable real-world environments. Using
vanilla Proximal Policy Optimization (PPO) as a baseline, we compare a
selection of PPO variants designed to mitigate partial observability, including
frame-stacking, augmenting observations with historical information, and
employing recurrent or transformer-based architectures. We conduct a systematic
empirical analysis of these algorithms across different host network sizes. We
find that this task greatly benefits from history aggregation. Converging three
times faster than other approaches. Manual inspection of the learned policies
by the algorithms reveals clear distinctions and provides insights that go
beyond quantitative results.

### 7. [CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning](http://arxiv.org/pdf/2509.20166v1)

Authors: Lauren Deason, Adam Bali, Ciprian Bejean, Diana Bolocan, James Crnkovich, Ioana Croitoru, Krishna Durai, Chase Midler, Calin Miron, David Molnar, Brad Moon, Bruno Ostarcevic, Alberto Peltea, Matt Rosenberg, Catalin Sandu, Arthur Saputkin, Sagar Shah, Daniel Stan, Ernest Szocs, Shengye Wan, Spencer Whitman, Sven Krasser, Joshua Saxe

Today's cyber defenders are overwhelmed by a deluge of security alerts,
threat intelligence signals, and shifting business context, creating an urgent
need for AI systems to enhance operational security work. While Large Language
Models (LLMs) have the potential to automate and scale Security Operations
Center (SOC) operations, existing evaluations do not fully assess the scenarios
most relevant to real-world defenders. This lack of informed evaluation impacts
both AI developers and those applying LLMs to SOC automation. Without clear
insight into LLM performance in real-world security scenarios, developers lack
a north star for development, and users cannot reliably select the most
effective models. Meanwhile, malicious actors are using AI to scale cyber
attacks, highlighting the need for open source benchmarks to drive adoption and
community-driven improvement among defenders and model developers. To address
this, we introduce CyberSOCEval, a new suite of open source benchmarks within
CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in
two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive
domains with inadequate coverage in current benchmarks. Our evaluations show
that larger, more modern LLMs tend to perform better, confirming the training
scaling laws paradigm. We also find that reasoning models leveraging test time
scaling do not achieve the same boost as in coding and math, suggesting these
models have not been trained to reason about cybersecurity analysis, and
pointing to a key opportunity for improvement. Finally, current LLMs are far
from saturating our evaluations, showing that CyberSOCEval presents a
significant challenge for AI developers to improve cyber defense capabilities.

### 8. [STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation](http://arxiv.org/pdf/2509.20190v1)

Authors: Tanmay Khule, Stefan Marksteiner, Jose Alguindigue, Hannes Fuchs, Sebastian Fischmeister, Apurva Narayan

In modern automotive development, security testing is critical for
safeguarding systems against increasingly advanced threats. Attack trees are
widely used to systematically represent potential attack vectors, but
generating comprehensive test cases from these trees remains a labor-intensive,
error-prone task that has seen limited automation in the context of testing
vehicular systems. This paper introduces STAF (Security Test Automation
Framework), a novel approach to automating security test case generation.
Leveraging Large Language Models (LLMs) and a four-step self-corrective
Retrieval-Augmented Generation (RAG) framework, STAF automates the generation
of executable security test cases from attack trees, providing an end-to-end
solution that encompasses the entire attack surface. We particularly show the
elements and processes needed to provide an LLM to actually produce sensible
and executable automotive security test suites, along with the integration with
an automated testing framework. We further compare our tailored approach with
general purpose (vanilla) LLMs and the performance of different LLMs (namely
GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our
operation step-by-step in a concrete case study. Our results show significant
improvements in efficiency, accuracy, scalability, and easy integration in any
workflow, marking a substantial advancement in automating automotive security
testing methodologies. Using TARAs as an input for verfication tests, we create
synergies by connecting two vital elements of a secure automotive development
process.

### 9. [Investigating Security Implications of Automatically Generated Code on the Software Supply Chain](http://arxiv.org/pdf/2509.20277v1)

Authors: Xiaofan Li, Xing Gao

In recent years, various software supply chain (SSC) attacks have posed
significant risks to the global community. Severe consequences may arise if
developers integrate insecure code snippets that are vulnerable to SSC attacks
into their products. Particularly, code generation techniques, such as large
language models (LLMs), have been widely utilized in the developer community.
However, LLMs are known to suffer from inherent issues when generating code,
including fabrication, misinformation, and reliance on outdated training data,
all of which can result in serious software supply chain threats. In this
paper, we investigate the security threats to the SSC that arise from these
inherent issues. We examine three categories of threats, including eleven
potential SSC-related threats, related to external components in source code,
and continuous integration configuration files. We find some threats in
LLM-generated code could enable attackers to hijack software and workflows,
while some others might cause potential hidden threats that compromise the
security of the software over time. To understand these security impacts and
severity, we design a tool, SSCGuard, to generate 439,138 prompts based on
SSC-related questions collected online, and analyze the responses of four
popular LLMs from GPT and Llama. Our results show that all identified
SSC-related threats persistently exist. To mitigate these risks, we propose a
novel prompt-based defense mechanism, namely Chain-of-Confirmation, to reduce
fabrication, and a middleware-based defense that informs users of various SSC
threats.

### 10. [RAG Security and Privacy: Formalizing the Threat Model and Attack Surface](http://arxiv.org/pdf/2509.20324v1)

Authors: Atousa Arzanipour, Rouzbeh Behnia, Reza Ebrahimi, Kaushik Dutta

Retrieval-Augmented Generation (RAG) is an emerging approach in natural
language processing that combines large language models (LLMs) with external
document retrieval to produce more accurate and grounded responses. While RAG
has shown strong potential in reducing hallucinations and improving factual
consistency, it also introduces new privacy and security challenges that differ
from those faced by traditional LLMs. Existing research has demonstrated that
LLMs can leak sensitive information through training data memorization or
adversarial prompts, and RAG systems inherit many of these vulnerabilities. At
the same time, reliance of RAG on an external knowledge base opens new attack
surfaces, including the potential for leaking information about the presence or
content of retrieved documents, or for injecting malicious content to
manipulate model behavior. Despite these risks, there is currently no formal
framework that defines the threat landscape for RAG systems. In this paper, we
address a critical gap in the literature by proposing, to the best of our
knowledge, the first formal threat model for retrieval-RAG systems. We
introduce a structured taxonomy of adversary types based on their access to
model components and data, and we formally define key threat vectors such as
document-level membership inference and data poisoning, which pose serious
privacy and integrity risks in real-world deployments. By establishing formal
definitions and attack models, our work lays the foundation for a more rigorous
and principled understanding of privacy and security in RAG systems.

### Computer Vision and Pattern Recognition

### 1. [Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment](http://arxiv.org/pdf/2509.19659v1)

Authors: Aravind Narayanan, Vahid Reza Khazaie, Shaina Raza

Large vision-language models (VLMs) can jointly interpret images and text,
but they are also prone to absorbing and reproducing harmful social stereotypes
when visual cues such as age, gender, race, clothing, or occupation are
present. To investigate these risks, we introduce a news-image benchmark
consisting of 1,343 image-question pairs drawn from diverse outlets, which we
annotated with ground-truth answers and demographic attributes (age, gender,
race, occupation, and sports). We evaluate a range of state-of-the-art VLMs and
employ a large language model (LLM) as judge, with human verification. Our
findings show that: (i) visual context systematically shifts model outputs in
open-ended settings; (ii) bias prevalence varies across attributes and models,
with particularly high risk for gender and occupation; and (iii) higher
faithfulness does not necessarily correspond to lower bias. We release the
benchmark prompts, evaluation rubric, and code to support reproducible and
fairness-aware multimodal assessment.

### 2. [Enhancing Transformer-Based Vision Models: Addressing Feature Map Anomalies Through Novel Optimization Strategies](http://arxiv.org/pdf/2509.19687v1)

Authors: Sumit Mamtani

Vision Transformers (ViTs) have demonstrated superior performance across a
wide range of computer vision tasks. However, structured noise artifacts in
their feature maps hinder downstream applications such as segmentation and
depth estimation. We propose two novel and lightweight optimisation techniques-
Structured Token Augmentation (STA) and Adaptive Noise Filtering (ANF)- to
improve interpretability and mitigate these artefacts. STA enhances token
diversity through spatial perturbations during tokenisation, while ANF applies
learnable inline denoising between transformer layers. These methods are
architecture-agnostic and evaluated across standard benchmarks, including
ImageNet, Ade20k, and NYUv2. Experimental results show consistent improvements
in visual quality and task performance, highlighting the practical
effectiveness of our approach.

### 3. [From Prompt to Progression: Taming Video Diffusion Models for Seamless Attribute Transition](http://arxiv.org/pdf/2509.19690v1)

Authors: Ling Lo, Kelvin C. K. Chan, Wen-Huang Cheng, Ming-Hsuan Yang

Existing models often struggle with complex temporal changes, particularly
when generating videos with gradual attribute transitions. The most common
prompt interpolation approach for motion transitions often fails to handle
gradual attribute transitions, where inconsistencies tend to become more
pronounced. In this work, we propose a simple yet effective method to extend
existing models for smooth and consistent attribute transitions, through
introducing frame-wise guidance during the denoising process. Our approach
constructs a data-specific transitional direction for each noisy latent,
guiding the gradual shift from initial to final attributes frame by frame while
preserving the motion dynamics of the video. Moreover, we present the
Controlled-Attribute-Transition Benchmark (CAT-Bench), which integrates both
attribute and motion dynamics, to comprehensively evaluate the performance of
different models. We further propose two metrics to assess the accuracy and
smoothness of attribute transitions. Experimental results demonstrate that our
approach performs favorably against existing baselines, achieving visual
fidelity, maintaining alignment with text prompts, and delivering seamless
attribute transitions. Code and CATBench are released:
https://github.com/lynn-ling-lo/Prompt2Progression.

### 4. [Anatomically Constrained Transformers for Cardiac Amyloidosis Classification](http://arxiv.org/pdf/2509.19691v1)

Authors: Alexander Thorley, Agis Chartsias, Jordan Strom, Roberto Lang, Jeremy Slivnick, Jamie O'Driscoll, Rajan Sharma, Dipak Kotecha, Jinming Duan, Alberto Gomez

Cardiac amyloidosis (CA) is a rare cardiomyopathy, with typical abnormalities
in clinical measurements from echocardiograms such as reduced global
longitudinal strain of the myocardium. An alternative approach for detecting CA
is via neural networks, using video classification models such as convolutional
neural networks. These models process entire video clips, but provide no
assurance that classification is based on clinically relevant features known to
be associated with CA. An alternative paradigm for disease classification is to
apply models to quantitative features such as strain, ensuring that the
classification relates to clinically relevant features. Drawing inspiration
from this approach, we explicitly constrain a transformer model to the
anatomical region where many known CA abnormalities occur -- the myocardium,
which we embed as a set of deforming points and corresponding sampled image
patches into input tokens. We show that our anatomical constraint can also be
applied to the popular self-supervised learning masked autoencoder
pre-training, where we propose to mask and reconstruct only anatomical patches.
We show that by constraining both the transformer and pre-training task to the
myocardium where CA imaging features are localized, we achieve increased
performance on a CA classification task compared to full video transformers.
Our model provides an explicit guarantee that the classification is focused on
only anatomical regions of the echo, and enables us to visualize transformer
attention scores over the deforming myocardium.

### 5. [Learning to Stop: Reinforcement Learning for Efficient Patient-Level Echocardiographic Classification](http://arxiv.org/pdf/2509.19694v1)

Authors: Woo-Jin Cho Kim, Jorge Oliveira, Arian Beqiri, Alex Thorley, Jordan Strom, Jamie O'Driscoll, Rajan Sharma, Jeremy Slivnick, Roberto Lang, Alberto Gomez, Agisilaos Chartsias

Guidelines for transthoracic echocardiographic examination recommend the
acquisition of multiple video clips from different views of the heart,
resulting in a large number of clips. Typically, automated methods, for
instance disease classifiers, either use one clip or average predictions from
all clips. Relying on one clip ignores complementary information available from
other clips, while using all clips is computationally expensive and may be
prohibitive for clinical adoption.
  To select the optimal subset of clips that maximize performance for a
specific task (image-based disease classification), we propose a method
optimized through reinforcement learning. In our method, an agent learns to
either keep processing view-specific clips to reduce the disease classification
uncertainty, or stop processing if the achieved classification confidence is
sufficient. Furthermore, we propose a learnable attention-based aggregation
method as a flexible way of fusing information from multiple clips. The
proposed method obtains an AUC of 0.91 on the task of detecting cardiac
amyloidosis using only 30% of all clips, exceeding the performance achieved
from using all clips and from other benchmarks.

### 6. [Towards Robust In-Context Learning for Medical Image Segmentation via Data Synthesis](http://arxiv.org/pdf/2509.19711v1)

Authors: Jiesi Hu, Yanwu Yang, Zhiyu Ye, Chenfei Ye, Hanyang Peng, Jianfeng Cao, Ting Ma

The rise of In-Context Learning (ICL) for universal medical image
segmentation has introduced an unprecedented demand for large-scale, diverse
datasets for training, exacerbating the long-standing problem of data scarcity.
While data synthesis offers a promising solution, existing methods often fail
to simultaneously achieve both high data diversity and a domain distribution
suitable for medical data. To bridge this gap, we propose \textbf{SynthICL}, a
novel data synthesis framework built upon domain randomization. SynthICL
ensures realism by leveraging anatomical priors from real-world datasets,
generates diverse anatomical structures to cover a broad data distribution, and
explicitly models inter-subject variations to create data cohorts suitable for
ICL. Extensive experiments on four held-out datasets validate our framework's
effectiveness, showing that models trained with our data achieve performance
gains of up to 63\% in average Dice and substantially enhanced generalization
to unseen anatomical domains. Our work helps mitigate the data bottleneck for
ICL-based segmentation, paving the way for robust models. Our code and the
generated dataset are publicly available at
https://github.com/jiesihu/Neuroverse3D.

### 7. [Frequency-domain Multi-modal Fusion for Language-guided Medical Image Segmentation](http://arxiv.org/pdf/2509.19719v1)

Authors: Bo Yu, Jianhua Yang, Zetao Du, Yan Huang, Chenglong Li, Liang Wang

Automatically segmenting infected areas in radiological images is essential
for diagnosing pulmonary infectious diseases. Recent studies have demonstrated
that the accuracy of the medical image segmentation can be improved by
incorporating clinical text reports as semantic guidance. However, the complex
morphological changes of lesions and the inherent semantic gap between
vision-language modalities prevent existing methods from effectively enhancing
the representation of visual features and eliminating semantically irrelevant
information, ultimately resulting in suboptimal segmentation performance. To
address these problems, we propose a Frequency-domain Multi-modal Interaction
model (FMISeg) for language-guided medical image segmentation. FMISeg is a late
fusion model that establishes interaction between linguistic features and
frequency-domain visual features in the decoder. Specifically, to enhance the
visual representation, our method introduces a Frequency-domain Feature
Bidirectional Interaction (FFBI) module to effectively fuse frequency-domain
features. Furthermore, a Language-guided Frequency-domain Feature Interaction
(LFFI) module is incorporated within the decoder to suppress semantically
irrelevant visual features under the guidance of linguistic information.
Experiments on QaTa-COV19 and MosMedData+ demonstrated that our method
outperforms the state-of-the-art methods qualitatively and quantitatively.

### 8. [PolGS: Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction](http://arxiv.org/pdf/2509.19726v1)

Authors: Yufei Han, Bowen Tie, Heng Guo, Youwei Lyu, Si Li, Boxin Shi, Yunpeng Jia, Zhanyu Ma

Efficient shape reconstruction for surfaces with complex reflectance
properties is crucial for real-time virtual reality. While 3D Gaussian
Splatting (3DGS)-based methods offer fast novel view rendering by leveraging
their explicit surface representation, their reconstruction quality lags behind
that of implicit neural representations, particularly in the case of recovering
surfaces with complex reflective reflectance. To address these problems, we
propose PolGS, a Polarimetric Gaussian Splatting model allowing fast reflective
surface reconstruction in 10 minutes. By integrating polarimetric constraints
into the 3DGS framework, PolGS effectively separates specular and diffuse
components, enhancing reconstruction quality for challenging reflective
materials. Experimental results on the synthetic and real-world dataset
validate the effectiveness of our method.

### 9. [CAMILA: Context-Aware Masking for Image Editing with Language Alignment](http://arxiv.org/pdf/2509.19731v1)

Authors: Hyunseung Kim, Chiho Choi, Srikanth Malla, Sai Prahladh Padmanabhan, Saurabh Bagchi, Joon Hee Choi

Text-guided image editing has been allowing users to transform and synthesize
images through natural language instructions, offering considerable
flexibility. However, most existing image editing models naively attempt to
follow all user instructions, even if those instructions are inherently
infeasible or contradictory, often resulting in nonsensical output. To address
these challenges, we propose a context-aware method for image editing named as
CAMILA (Context-Aware Masking for Image Editing with Language Alignment).
CAMILA is designed to validate the contextual coherence between instructions
and the image, ensuring that only relevant edits are applied to the designated
regions while ignoring non-executable instructions. For comprehensive
evaluation of this new method, we constructed datasets for both single- and
multi-instruction image editing, incorporating the presence of infeasible
requests. Our method achieves better performance and higher semantic alignment
than state-of-the-art models, demonstrating its effectiveness in handling
complex instruction challenges while preserving image integrity.

### 10. [Robust RGB-T Tracking via Learnable Visual Fourier Prompt Fine-tuning and Modality Fusion Prompt Generation](http://arxiv.org/pdf/2509.19733v1)

Authors: Hongtao Yang, Bineng Zhong, Qihua Liang, Zhiruo Zhu, Yaozong Zheng, Ning Li

Recently, visual prompt tuning is introduced to RGB-Thermal (RGB-T) tracking
as a parameter-efficient finetuning (PEFT) method. However, these PEFT-based
RGB-T tracking methods typically rely solely on spatial domain information as
prompts for feature extraction. As a result, they often fail to achieve optimal
performance by overlooking the crucial role of frequency-domain information in
prompt learning. To address this issue, we propose an efficient Visual Fourier
Prompt Tracking (named VFPTrack) method to learn modality-related prompts via
Fast Fourier Transform (FFT). Our method consists of symmetric feature
extraction encoder with shared parameters, visual fourier prompts, and Modality
Fusion Prompt Generator that generates bidirectional interaction prompts
through multi-modal feature fusion. Specifically, we first use a frozen feature
extraction encoder to extract RGB and thermal infrared (TIR) modality features.
Then, we combine the visual prompts in the spatial domain with the frequency
domain prompts obtained from the FFT, which allows for the full extraction and
understanding of modality features from different domain information. Finally,
unlike previous fusion methods, the modality fusion prompt generation module we
use combines features from different modalities to generate a fused modality
prompt. This modality prompt is interacted with each individual modality to
fully enable feature interaction across different modalities. Extensive
experiments conducted on three popular RGB-T tracking benchmarks show that our
method demonstrates outstanding performance.

### Computers and Society

### 1. [DSA, AIA, and LLMs: Approaches to conceptualizing and auditing moderation in LLM-based chatbots across languages and interfaces in the electoral contexts](http://arxiv.org/pdf/2509.19890v1)

Authors: Natalia Stanusch, Raziye Buse Cetin, Salvatore Romano, Miazia Schueler, Meret Baumgartner, Bastian August, Alexandra Rosca

The integration of Large Language Models (LLMs) into chatbot-like search
engines poses new challenges for governing, assessing, and scrutinizing the
content output by these online entities, especially in light of the Digital
Service Act (DSA). In what follows, we first survey the regulation landscape in
which we can situate LLM-based chatbots and the notion of moderation. Second,
we outline the methodological approaches to our study: a mixed-methods audit
across chatbots, languages, and elections. We investigated Copilot, ChatGPT,
and Gemini across ten languages in the context of the 2024 European
Parliamentary Election and the 2024 US Presidential Election. Despite the
uncertainty in regulatory frameworks, we propose a set of solutions on how to
situate, study, and evaluate chatbot moderation.

### 2. [The three main doctrines on the future of AI](http://arxiv.org/pdf/2509.20050v1)

Authors: Alex Amadori, Eva Behrens, Gabriel Alfour, Andrea Miotti

This paper develops a taxonomy of expert perspectives on the risks and likely
consequences of artificial intelligence, with particular focus on Artificial
General Intelligence (AGI) and Artificial Superintelligence (ASI). Drawing from
primary sources, we identify three predominant doctrines: (1) The dominance
doctrine, which predicts that the first actor to create sufficiently advanced
AI will attain overwhelming strategic superiority sufficient to cheaply
neutralize its opponents' defenses; (2) The extinction doctrine, which
anticipates that humanity will likely lose control of ASI, leading to the
extinction of the human species or its permanent disempowerment; (3) The
replacement doctrine, which forecasts that AI will automate a large share of
tasks currently performed by humans, but will not be so transformative as to
fundamentally reshape or bring an end to human civilization. We examine the
assumptions and arguments underlying each doctrine, including expectations
around the pace of AI progress and the feasibility of maintaining advanced AI
under human control. While the boundaries between doctrines are sometimes
porous and many experts hedge across them, this taxonomy clarifies the core
axes of disagreement over the anticipated scale and nature of the consequences
of AI development.

### 3. [Current and Future Directions for Responsible Quantum Technologies: A ResQT Community Perspective](http://arxiv.org/pdf/2509.19815v1)

Authors: Adrian Schmidt, Alexandre Artaud, Arsev Umur Aydinoglu, Astrid Bötticher, Rodrigo Araiza Bravo, Marilu Chiofalo, Rebecca Coates, Ilke Ercan, Alexei Grinbaum, Emily Haworth, Carolyn Ten Holter, Eline de Jong, Bart Karstens, Matthias C. Kettemann, Anna Knörr, Clarissa Ai Ling Lee, Fabienne Marco, Wenzel Mehnert, Josephine C. Meyer, Shantanu Sharma, Pieter Vermaas, Carrie Weidner, Barbara Wellmann, Mira L. Wolf-Bauwens, Zeki C. Seskir

Quantum technologies (QT) are advancing rapidly, promising advancements
across a wide spectrum of applications but also raising significant ethical,
societal, and geopolitical impacts, including dual-use capabilities, varying
levels of access, and impending quantum divide(s). To address these, the
Responsible Quantum Technologies (ResQT) community was established to share
knowledge, perspectives, and best practices across various disciplines. Its
mission is to ensure QT developments align with ethical principles, promote
equity, and mitigate unintended consequences. Initial progress has been made,
as scholars and policymakers increasingly recognize principles of responsible
QT. However, more widespread dissemination is needed, and as QT matures, so
must responsible QT. This paper provides a comprehensive overview of the ResQT
community's current work and states necessary future directions. Drawing on
historical lessons from artificial intelligence and nanotechnology, actions
targeting the quantum divide(s) are addressed, including the implementation of
responsible research and innovation, fostering wider stakeholder engagement,
and sustainable development. These actions aim to build trust and engagement,
facilitating the participatory and responsible development of QT. The ResQT
community advocates that responsible QT should be an integral part of quantum
development rather than an afterthought so that quantum technologies evolve
toward a future that is technologically advanced and beneficial for all.

### 4. [Choosing to Be Green: Advancing Green AI via Dynamic Model Selection](http://arxiv.org/pdf/2509.19996v1)

Authors: Emilio Cruciani, Roberto Verdecchia

Artificial Intelligence is increasingly pervasive across domains, with ever
more complex models delivering impressive predictive performance. This fast
technological advancement however comes at a concerning environmental cost,
with state-of-the-art models - particularly deep neural networks and large
language models - requiring substantial computational resources and energy. In
this work, we present the intuition of Green AI dynamic model selection, an
approach based on dynamic model selection that aims at reducing the
environmental footprint of AI by selecting the most sustainable model while
minimizing potential accuracy loss. Specifically, our approach takes into
account the inference task, the environmental sustainability of available
models, and accuracy requirements to dynamically choose the most suitable
model. Our approach presents two different methods, namely Green AI dynamic
model cascading and Green AI dynamic model routing. We demonstrate the
effectiveness of our approach via a proof of concept empirical example based on
a real-world dataset. Our results show that Green AI dynamic model selection
can achieve substantial energy savings (up to ~25%) while substantially
retaining the accuracy of the most energy greedy solution (up to ~95%). As
conclusion, our preliminary findings highlight the potential that hybrid,
adaptive model selection strategies withhold to mitigate the energy demands of
modern AI systems without significantly compromising accuracy requirements.

### 5. [Cascade! Human in the loop shortcomings can increase the risk of failures in recommender systems](http://arxiv.org/pdf/2509.20099v1)

Authors: Wm. Matthew Kennedy, Nishanshi Shukla, Cigdem Patlak, Blake Chambers, Theodora Skeadas, Tuesday, Kingsley Owadara, Aayush Dhanotiya

Recommender systems are among the most commonly deployed systems today.
Systems design approaches to AI-powered recommender systems have done well to
urge recommender system developers to follow more intentional data collection,
curation, and management procedures. So too has the "human-in-the-loop"
paradigm been widely adopted, primarily to address the issue of accountability.
However, in this paper, we take the position that human oversight in
recommender system design also entails novel risks that have yet to be fully
described. These risks are "codetermined" by the information context in which
such systems are often deployed. Furthermore, new knowledge of the shortcomings
of "human-in-the-loop" practices to deliver meaningful oversight of other AI
systems suggest that they may also be inadequate for achieving socially
responsible recommendations. We review how the limitations of human oversight
may increase the chances of a specific kind of failure: a "cascade" or
"compound" failure. We then briefly explore how the unique dynamics of three
common deployment contexts can make humans in the loop more likely to fail in
their oversight duties. We then conclude with two recommendations.

### 6. [Affective Computing and Emotional Data: Challenges and Implications in Privacy Regulations, The AI Act, and Ethics in Large Language Models](http://arxiv.org/pdf/2509.20153v1)

Authors: Nicola Fabiano

This paper examines the integration of emotional intelligence into artificial
intelligence systems, with a focus on affective computing and the growing
capabilities of Large Language Models (LLMs), such as ChatGPT and Claude, to
recognize and respond to human emotions. Drawing on interdisciplinary research
that combines computer science, psychology, and neuroscience, the study
analyzes foundational neural architectures - CNNs for processing facial
expressions and RNNs for sequential data, such as speech and text - that enable
emotion recognition. It examines the transformation of human emotional
experiences into structured emotional data, addressing the distinction between
explicit emotional data collected with informed consent in research settings
and implicit data gathered passively through everyday digital interactions.
That raises critical concerns about lawful processing, AI transparency, and
individual autonomy over emotional expressions in digital environments. The
paper explores implications across various domains, including healthcare,
education, and customer service, while addressing challenges of cultural
variations in emotional expression and potential biases in emotion recognition
systems across different demographic groups. From a regulatory perspective, the
paper examines emotional data in the context of the GDPR and the EU AI Act
frameworks, highlighting how emotional data may be considered sensitive
personal data that requires robust safeguards, including purpose limitation,
data minimization, and meaningful consent mechanisms.

### Databases

### 1. [Output-Sensitive Evaluation of Acyclic Conjunctive Regular Path Queries](http://arxiv.org/pdf/2509.20204v1)

Authors: Mahmoud Abo Khamis, Alexandru-Mihai Hurjui, Ahmet Kara, Dan Olteanu, Dan Suciu, Zilu Tian

Conjunctive Regular Path Queries, or CRPQs for short, are an essential
construct in graph query languages. In this paper, we propose the first
output-sensitive algorithm for evaluating acyclic CRPQs. It is output-sensitive
in the sense that its complexity is a function of the sizes of the input graph
and of the query output. In particular, it does not depend on the output sizes
of the regular expressions that appear in the query, as these sizes can be much
larger than the query output size.
  Our algorithm proceeds in two stages. In the first stage, it contracts the
given query into a free-connex acyclic one such that the output of the original
query can be obtained from the output of the contracted one. This contraction
removes bound variables by composing regular expressions or by promoting bound
variables to free ones. The minimum necessary number of promoted bound
variables gives the contraction width, which is a novel parameter specific to
CRPQs. In the second stage, our algorithm evaluates the free-connex acyclic
CRPQ and projects away the columns of the promoted bound variables. It ensures
output-sensitivity by computing the calibrated outputs of the regular
expressions appearing in the free-connex acyclic CRPQ in time proportional to
their sizes.
  Our algorithm has lower complexity than the state-of-the-art approaches for
problem instances where (i) the query output is asymptotically smaller than the
worst-case output size or (ii) the largest output size of any of the regular
expression in the query.

### 2. [ARCADE: A Real-Time Data System for Hybrid and Continuous Query Processing across Diverse Data Modalities](http://arxiv.org/pdf/2509.19757v1)

Authors: Jingyi Yang, Songsong Mo, Jiachen Shi, Zihao Yu, Kunhao Shi, Xuchen Ding, Gao Cong

The explosive growth of multimodal data - spanning text, image, video,
spatial, and relational modalities, coupled with the need for real-time
semantic search and retrieval over these data - has outpaced the capabilities
of existing multimodal and real-time database systems, which either lack
efficient ingestion and continuous query capability, or fall short in
supporting expressive hybrid analytics. We introduce ARCADE, a real-time data
system that efficiently supports high-throughput ingestion and expressive
hybrid and continuous query processing across diverse data types. ARCADE
introduces unified disk-based secondary index on LSM-based storage for vector,
spatial, and text data modalities, a comprehensive cost-based query optimizer
for hybrid queries, and an incremental materialized view framework for
efficient continuous queries. Built on open-source RocksDB storage and MySQL
query engine, ARCADE outperforms leading multimodal data systems by up to 7.4x
on read-heavy and 1.4x on write-heavy workloads.

### 3. [FusedANN: Convexified Hybrid ANN via Attribute-Vector Fusion](http://arxiv.org/pdf/2509.19767v1)

Authors: Alireza Heidari, Wei Zhang, Ying Xiong

Vector search powers transformers technology, but real-world use demands
hybrid queries that combine vector similarity with attribute filters (e.g.,
"top document in category X, from 2023"). Current solutions trade off recall,
speed, and flexibility, relying on fragile index hacks that don't scale. We
introduce FusedANN (Fused Attribute-Vector Nearest Neighbor), a geometric
framework that elevates filtering to ANN optimization constraints and
introduces a convex fused space via a Lagrangian-like relaxation. Our method
jointly embeds attributes and vectors through transformer-based
convexification, turning hard filters into continuous, weighted penalties that
preserve top-k semantics while enabling efficient approximate search. We prove
that FusedANN reduces to exact filtering under high selectivity, gracefully
relaxes to semantically nearest attributes when exact matches are insufficient,
and preserves downstream ANN alpha-approximation guarantees. Empirically,
FusedANN improves query throughput by eliminating brittle filtering stages,
achieving superior recall-latency tradeoffs on standard hybrid benchmarks
without specialized index hacks, delivering up to 3 times higher throughput and
better recall than state-of-the-art hybrid and graph-based systems.
Theoretically, we provide explicit error bounds and parameter selection rules
that make FusedANN practical for production. This establishes a principled,
scalable, and verifiable bridge between symbolic constraints and vector
similarity, unlocking a new generation of filtered retrieval systems for large,
hybrid, and dynamic NLP/ML workloads.

### 4. [Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs](http://arxiv.org/pdf/2509.20208v1)

Authors: Parker Glenn, Alfy Samuel, Daben Liu

Integrating LLM powered operators in declarative query languages allows for
the combination of cheap and interpretable functions with powerful,
generalizable language model reasoning. However, in order to benefit from the
optimized execution of a database query language like SQL, generated outputs
must align with the rules enforced by both type checkers and database contents.
Current approaches address this challenge with orchestrations consisting of
many LLM-based post-processing calls to ensure alignment between generated
outputs and database values, introducing performance bottlenecks. We perform a
study on the ability of various sized open-source language models to both parse
and execute functions within a query language based on SQL, showing that small
language models can excel as function executors over hybrid data sources. Then,
we propose an efficient solution to enforce the well-typedness of LLM
functions, demonstrating 7% accuracy improvement on a multi-hop question
answering dataset with 53% improvement in latency over comparable solutions. We
make our implementation available at https://github.com/parkervg/blendsql

### Distributed, Parallel, and Cluster Computing

### 1. [Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference](http://arxiv.org/pdf/2509.19729v1)

Authors: Haoyu Chen, Xue Li, Kun Qian, Yu Guan, Jin Zhao, Xin Wang

Efficiently processing the dynamics of requests, especially the context
length variance, is important in Large Language Model (LLM) serving scenarios.
However, there is an intrinsic trade-off: while leveraging parallelism
strategies, such as Tensor Parallelism (TP), can coordinate multiple GPUs to
accommodate larger context lengths, it inevitably results in degraded overall
throughput. In this paper, we propose Cross-Instance Parallelism Transformation
(Gyges), which adaptively adjusts the parallelism strategies of running
instances to align with the dynamics of incoming requests. We design (1) a
page-friendly, header-centric layout to accelerate KV cache transformations;
(2) dedicated weight padding to accelerate model weight transformations; and
(3) a transformation-aware scheduler to cooperatively schedule requests and
parallelism transformations, optimizing the overall performance. Evaluations
using real-world traces show that Gyges improves throughput by 1.75x-6.57x
compared to state-of-the-art solutions.

### 2. [BurstEngine: an Efficient Distributed Framework for Training Transformers on Extremely Long Sequences of over 1M Tokens](http://arxiv.org/pdf/2509.19836v1)

Authors: Ao Sun, Weilin Zhao, Xu Han, Cheng Yang, Zhiyuan Liu, Chuan Shi, Maosong sun

Existing methods for training LLMs on long-sequence data, such as Tensor
Parallelism and Context Parallelism, exhibit low Model FLOPs Utilization as
sequence lengths and number of GPUs increase, especially when sequence lengths
exceed 1M tokens. To address these challenges, we propose BurstEngine, an
efficient framework designed to train LLMs on long-sequence data. BurstEngine
introduces BurstAttention, an optimized distributed attention with lower
communication cost than RingAttention. BurstAttention leverages topology-aware
ring communication to fully utilize network bandwidth and incorporates
fine-grained communication-computation overlap. Furthermore, BurstEngine
introduces sequence-level selective checkpointing and fuses the language
modeling head with the loss function to reduce memory cost. Additionally,
BurstEngine introduces workload balance optimization for various types of
attention masking. By integrating these optimizations, BurstEngine achieves a
$1.2\times$ speedup with much lower memory overhead than the state-of-the-art
baselines when training LLMs on extremely long sequences of over 1M tokens. We
have made our code publicly available on GitHub:
https://github.com/thunlp/BurstEngine.

### 3. [Characterizing the Performance of Accelerated Jetson Edge Devices for Training Deep Learning Models](http://arxiv.org/pdf/2509.20160v1)

Authors: Prashanthi S. K., Sai Anuroop Kesanapalli, Yogesh Simmhan

Deep Neural Networks (DNNs) have had a significant impact on domains like
autonomous vehicles and smart cities through low-latency inferencing on edge
computing devices close to the data source. However, DNN training on the edge
is poorly explored. Techniques like federated learning and the growing capacity
of GPU-accelerated edge devices like NVIDIA Jetson motivate the need for a
holistic characterization of DNN training on the edge. Training DNNs is
resource-intensive and can stress an edge's GPU, CPU, memory and storage
capacities. Edge devices also have different resources compared to workstations
and servers, such as slower shared memory and diverse storage media. Here, we
perform a principled study of DNN training on individual devices of three
contemporary Jetson device types: AGX Xavier, Xavier NX and Nano for three
diverse DNN model--dataset combinations. We vary device and training parameters
such as I/O pipelining and parallelism, storage media, mini-batch sizes and
power modes, and examine their effect on CPU and GPU utilization, fetch stalls,
training time, energy usage, and variability. Our analysis exposes several
resource inter-dependencies and counter-intuitive insights, while also helping
quantify known wisdom. Our rigorous study can help tune the training
performance on the edge, trade-off time and energy usage on constrained
devices, and even select an ideal edge hardware for a DNN workload, and, in
future, extend to federated learning too. As an illustration, we use these
results to build a simple model to predict the training time and energy per
epoch for any given DNN across different power modes, with minimal additional
profiling.

### 4. [Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators](http://arxiv.org/pdf/2509.20189v1)

Authors: Prashanthi S. K., Kunal Kumar Sahoo, Amartya Ranjan Saikia, Pranav Gupta, Atharva Vinay Joshi, Priyanshu Pansari, Yogesh Simmhan

Edge accelerators such as Nvidia Jetsons are becoming an integral part of the
computing continuum, and are often used for DNN inferencing and training.
Nvidia Jetson edge devices have $2000$+ CUDA cores within a $70$W power
envelope and offer $1000$s of power modes to customize CPU, GPU and memory
frequencies. Their widely varying power--performance trade-offs can be
exploited for energy and power-constrained deployments. While data-driven
methods to predict the power and latency of DNN workloads for edge devices
exist, there is a lack of principled study to understand why edge accelerators
and their power modes perform the way they do. We develop a time roofline and a
novel energy roofline model for the Jetson Orin AGX for diverse power modes,
and couple it with an analytical model of the compute (FLOP) and memory access
(bytes) for DNN inference workloads to analyze them from first principles.
These reveal unique, sometimes counter-intuitive, insights into the power and
performance behavior of DNN workloads on edge accelerators, e.g., the default
power mode MAXN is not the most energy efficient and time efficiency implies
energy efficiency for all power modes. We also extend our analytical roofline
models to DNN training. Finally, we apply these methods to tune the power mode
(and hence the roofline) of the edge device to optimize the latency and energy
for DNN inference, with up to $15\%$ lower energy and minimal degradation in
inference time.

### 5. [Fulcrum: Optimizing Concurrent DNN Training and Inferencing on Edge Accelerators](http://arxiv.org/pdf/2509.20205v1)

Authors: Prashanthi S. K., Saisamarth Taluri, Pranav Gupta, Amartya Ranjan Saikia, Kunal Kumar Sahoo, Atharva Vinay Joshi, Lakshya Karwa, Kedar Dhule, Yogesh Simmhan

The proliferation of GPU accelerated edge devices like Nvidia Jetsons and the
rise in privacy concerns are placing an emphasis on concurrent DNN training and
inferencing on edge devices. Inference and training have different computing
and QoS goals. But edge accelerators like Jetson do not support native GPU
sharing and expose 1000s of power modes. This requires careful time-sharing of
concurrent workloads to meet power--performance goals, while limiting costly
profiling. In this paper, we design an intelligent time-slicing approach for
concurrent DNN training and inferencing on Jetsons. We formulate an
optimization problem to interleave training and inferencing minibatches, and
decide the device power mode and inference minibatch size, while maximizing the
training throughput and staying within latency and power budgets, with modest
profiling costs. We propose GMD, an efficient multi-dimensional gradient
descent search which profiles just $15$ power modes; and ALS, an Active
Learning technique which identifies reusable Pareto-optimal power modes, but
profiles $50$--$150$ power modes. We evaluate these within our Fulcrum
scheduler for $273,000+$ configurations across $15$ DNN workloads. We also
evaluate our strategies on dynamic arrival inference and concurrent inferences.
ALS and GMD outperform simpler and more complex baselines with larger-scale
profiling. Their solutions satisfy the latency and power budget for $>97\%$ of
our runs, and on average are within $7\%$ of the optimal throughput.

### 6. [An Empirical Analysis of Secure Federated Learning for Autonomous Vehicle Applications](http://arxiv.org/pdf/2509.20223v1)

Authors: Md Jueal Mia, M. Hadi Amini

Federated Learning lends itself as a promising paradigm in enabling
distributed learning for autonomous vehicles applications and ensuring data
privacy while enhancing and refining predictive model performance through
collaborative training on edge client vehicles. However, it remains vulnerable
to various categories of cyber-attacks, necessitating more robust security
measures to effectively mitigate potential threats. Poisoning attacks and
inference attacks are commonly initiated within the federated learning
environment to compromise secure system performance. Secure aggregation can
limit the disclosure of sensitive information from outsider and insider
attackers of the federated learning environment. In this study, our aim is to
conduct an empirical analysis on the transportation image dataset (e.g., LISA
traffic light) using various secure aggregation techniques and multiparty
computation in the presence of diverse categories of cyber-attacks. Multiparty
computation serves as a state-of-the-art security mechanism, offering standard
privacy for secure aggregation of edge autonomous vehicles local model updates
through various security protocols. The presence of adversaries can mislead the
autonomous vehicle learning model, leading to the misclassification of traffic
lights, and resulting in detrimental impacts. This empirical study explores the
resilience of various secure federated learning aggregation techniques and
multiparty computation in safeguarding autonomous vehicle applications against
various cyber threats during both training and inference times.

### 7. [xGFabric: Coupling Sensor Networks and HPC Facilities with Private 5G Wireless Networks for Real-Time Digital Agriculture](http://arxiv.org/pdf/2509.20340v1)

Authors: Liubov Kurafeeva, Alan Subedi, Ryan Hartung, Michael Fay, Avhishek Biswas, Shantenu Jha, Ozgur O. Kilic, Chandra Krintz, Andre Merzky, Douglas Thain, Mehmet C. Vuran, Rich Wolski

Advanced scientific applications require coupling distributed sensor networks
with centralized high-performance computing facilities. Citrus Under Protective
Screening (CUPS) exemplifies this need in digital agriculture, where citrus
research facilities are instrumented with numerous sensors monitoring
environmental conditions and detecting protective screening damage. CUPS
demands access to computational fluid dynamics codes for modeling environmental
conditions and guiding real-time interventions like water application or
robotic repairs. These computing domains have contrasting properties: sensor
networks provide low-performance, limited-capacity, unreliable data access,
while high-performance facilities offer enormous computing power through
high-latency batch processing. Private 5G networks present novel capabilities
addressing this challenge by providing low latency, high throughput, and
reliability necessary for near-real-time coupling of edge sensor networks with
HPC simulations. This work presents xGFabric, an end-to-end system coupling
sensor networks with HPC facilities through Private 5G networks. The prototype
connects remote sensors via 5G network slicing to HPC systems, enabling
real-time digital agriculture simulation.

### 8. [Characterizing Adaptive Mesh Refinement on Heterogeneous Platforms with Parthenon-VIBE](http://arxiv.org/pdf/2509.19701v1)

Authors: Akash Poptani, Alireza Khadem, Scott Mahlke, Jonah Miller, Joshua Dolence, Reetuparna Das

Hero-class HPC simulations rely on Adaptive Mesh Refinement (AMR) to reduce
compute and memory demands while maintaining accuracy. This work analyzes the
performance of Parthenon, a block-structured AMR benchmark, on CPU-GPU systems.
We show that smaller mesh blocks and deeper AMR levels degrade GPU performance
due to increased communication, serial overheads, and inefficient GPU
utilization. Through detailed profiling, we identify inefficiencies, low
occupancy, and memory access bottlenecks. We further analyze rank scalability
and memory constraints, and propose optimizations to improve GPU throughput and
reduce memory footprint. Our insights can inform future AMR deployments on
Department of Energy's upcoming heterogeneous supercomputers.

### 9. [Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute](http://arxiv.org/pdf/2509.20241v1)

Authors: Felipe Oviedo, Fiodar Kazhamiaka, Esha Choukse, Allen Kim, Amy Luers, Melanie Nakagawa, Ricardo Bianchini, Juan M. Lavista Ferres

As AI inference scales to billions of queries and emerging reasoning and
agentic workflows increase token demand, reliable estimates of per-query energy
use are increasingly important for capacity planning, emissions accounting, and
efficiency prioritization. Many public estimates are inconsistent and overstate
energy use, because they extrapolate from limited benchmarks and fail to
reflect efficiency gains achievable at scale. In this perspective, we introduce
a bottom-up methodology to estimate the per-query energy of large-scale LLM
systems based on token throughput. For models running on an H100 node under
realistic workloads, GPU utilization and PUE constraints, we estimate a median
energy per query of 0.34 Wh (IQR: 0.18-0.67) for frontier-scale models (>200
billion parameters). These results are consistent with measurements using
production-scale configurations and show that non-production estimates and
assumptions can overstate energy use by 4-20x. Extending to test-time scaling
scenarios with 15x more tokens per typical query, the median energy rises 13x
to 4.32 Wh, indicating that targeting efficiency in this regime will deliver
the largest fleet-wide savings. We quantify achievable efficiency gains at the
model, serving platform, and hardware levels, finding individual median
reductions of 1.5-3.5x in energy per query, while combined advances can
plausibly deliver 8-20x reductions. To illustrate the system-level impact, we
estimate the baseline daily energy use of a deployment serving 1 billion
queries to be 0.8 GWh/day. If 10% are long queries, demand could grow to 1.8
GWh/day. With targeted efficiency interventions, it falls to 0.9 GWh/day,
similar to the energy footprint of web search at that scale. This echoes how
data centers historically tempered energy growth through efficiency gains
during the internet and cloud build-up.

### Digital Libraries

### 1. [Polarity Detection of Sustainable Detection Goals in News Text](http://arxiv.org/pdf/2509.19833v1)

Authors: Andrea Cadeddua, Alessandro Chessa, Vincenzo De Leo, Gianni Fenu, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino, Luca Secchi

The United Nations' Sustainable Development Goals (SDGs) provide a globally
recognised framework for addressing critical societal, environmental, and
economic challenges. Recent developments in natural language processing (NLP)
and large language models (LLMs) have facilitated the automatic classification
of textual data according to their relevance to specific SDGs. Nevertheless, in
many applications, it is equally important to determine the directionality of
this relevance; that is, to assess whether the described impact is positive,
neutral, or negative. To tackle this challenge, we propose the novel task of
SDG polarity detection, which assesses whether a text segment indicates
progress toward a specific SDG or conveys an intention to achieve such
progress. To support research in this area, we introduce SDG-POD, a benchmark
dataset designed specifically for this task, combining original and
synthetically generated data. We perform a comprehensive evaluation using six
state-of-the-art large LLMs, considering both zero-shot and fine-tuned
configurations. Our results suggest that the task remains challenging for the
current generation of LLMs. Nevertheless, some fine-tuned models, particularly
QWQ-32B, achieve good performance, especially on specific Sustainable
Development Goals such as SDG-9 (Industry, Innovation and Infrastructure),
SDG-12 (Responsible Consumption and Production), and SDG-15 (Life on Land).
Furthermore, we demonstrate that augmenting the fine-tuning dataset with
synthetically generated examples yields improved model performance on this
task. This result highlights the effectiveness of data enrichment techniques in
addressing the challenges of this resource-constrained domain. This work
advances the methodological toolkit for sustainability monitoring and provides
actionable insights into the development of efficient, high-performing polarity
detection systems.

### Discrete Mathematics

### 1. [There is no prime functional digraph: Seifert's proof revisited](http://arxiv.org/pdf/2509.19940v1)

Authors: Adrien Richard

A functional digraph is a finite digraph in which each vertex has a unique
out-neighbor. Considered up to isomorphism and endowed with the directed sum
and product, functional digraphs form a semigroup that has recently attracted
significant attention, particularly regarding its multiplicative structure. In
this context, a functional digraph $X$ divides a functional digraph $A$ if
there exists a functional digraph $Y$ such that $XY$ is isomorphic to $A$. The
digraph $X$ is said to be prime if it is not the identity for the product, and
if, for all functional digraphs $A$ and $B$, the fact that $X$ divides $AB$
implies that $X$ divides $A$ or $B$. In 2020, Antonio E. Porreca asked whether
prime functional digraphs exist, and in 2023, his work led him to conjecture
that they do not. However, in 2024, Barbora Hudcov\'a discovered that this
result had already been proved by Ralph Seifert in 1971, in a somewhat
forgotten paper. The terminology in that work differs significantly from that
used in recent studies, the framework is more general, and the non-existence of
prime functional digraphs appears only as a part of broader results, relying on
(overly) technical lemmas developed within this general setting. The aim of
this note is to present a much more accessible version of Seifert's proof $-$
that no prime functional digraph exists $-$ by using the current language and
simplifying each step as much as possible.

### Data Structures and Algorithms

### 1. [A Better-Than-$5/4$-Approximation for Two-Edge Connectivity](http://arxiv.org/pdf/2509.19655v1)

Authors: Felix Hommelsheim, Alexander Lindermayr, Zhenwei Liu

The 2-Edge-Connected Spanning Subgraph Problem (2ECSS) is a fundamental
problem in survivable network design. Given an undirected $2$-edge-connected
graph, the goal is to find a $2$-edge-connected spanning subgraph with the
minimum number of edges; a graph is 2-edge-connected if it is connected after
the removal of any single edge. 2ECSS is APX-hard and has been extensively
studied in the context of approximation algorithms. Very recently, Bosch-Calvo,
Garg, Grandoni, Hommelsheim, Jabal Ameli, and Lindermayr showed the currently
best-known approximation ratio of $\frac{5}{4}$ [STOC 2025]. This factor is
tight for many of their techniques and arguments, and it was not clear whether
$\frac{5}{4}$ can be improved.
  We break this natural barrier and present a $(\frac{5}{4} -
\eta)$-approximation algorithm, for some constant $\eta \geq 10^{-6}$. On a
high level, we follow the approach of previous works: take a triangle-free
$2$-edge cover and transform it into a 2-edge-connected spanning subgraph by
adding only a few additional edges. For $\geq \frac{5}{4}$-approximations, one
can heavily exploit that a $4$-cycle in the 2-edge cover can ``buy'' one
additional edge. This enables simple and nice techniques, but immediately fails
for our improved approximation ratio. To overcome this, we design two
complementary algorithms that perform well for different scenarios: one for few
$4$-cycles and one for many $4$-cycles. Besides this, there appear more
obstructions when breaching $\frac54$, which we surpass via new techniques such
as colorful bridge covering, rich vertices, and branching gluing paths.

### 2. [Non-Clairvoyant Scheduling with Progress Bars](http://arxiv.org/pdf/2509.19662v1)

Authors: Ziyad Benomar, Romain Cosson, Alexander Lindermayr, Jens Schlöter

In non-clairvoyant scheduling, the goal is to minimize the total job
completion time without prior knowledge of individual job processing times.
This classical online optimization problem has recently gained attention
through the framework of learning-augmented algorithms. We introduce a natural
setting in which the scheduler receives continuous feedback in the form of
progress bars: estimates of the fraction of each job completed over time. We
design new algorithms for both adversarial and stochastic progress bars and
prove strong competitive bounds. Our results in the adversarial case
surprisingly induce improved guarantees for learning-augmented scheduling with
job size predictions. We also introduce a general method for combining
scheduling algorithms, yielding further insights in scheduling with
predictions. Finally, we propose a stochastic model of progress bars as a more
optimistic alternative to conventional worst-case models, and present an
asymptotically optimal scheduling algorithm in this setting.

### 3. [SS-GUMAP, SL-GUMAP, SSSL-GUMAP: Fast UMAP Algorithms for Large Graph Drawing](http://arxiv.org/pdf/2509.19703v1)

Authors: Amyra Meidiana, Seok-Hee Hong

UMAP is a popular neighborhood-preserving dimension reduction (DR) algorithm.
However, its application for graph drawing has not been evaluated. Moreover, a
naive application of UMAP to graph drawing would include O(nm) time all-pair
shortest path computation, which is not scalable to visualizing large graphs.
  In this paper, we present fast UMAP-based for graph drawing. Specifically, we
present three fast UMAP-based algorithms for graph drawing: (1) The SS-GUMAP
algorithm utilizes spectral sparsification to compute a subgraph G' preserving
important properties of a graph G, reducing the O(nm) component of the runtime
to O(n^2 log n) runtime; (2) The SSL-GUMAP algorithm reduces the kNN (k-Nearest
Neighbors) graph computation from $O(n \log n)$ time to linear time using
partial BFS (Breadth First Search), and the cost optimization runtime from O(n)
time to sublinear time using edge sampling; (3) The SSSL-GUMAP algorithm
combines both approaches, for an overall O(n) runtime.
  Experiments demonstrate that SS-GUMAP runs 28% faster than GUMAP, a naive
application of UMAP to graph drawing, with similar quality metrics, while
SL-GUMAP and SSSL-GUMAP run over 80% faster than GUMAP with less than 15%
difference on average for all quality metrics.
  We also present an evaluation of GUMAP to tsNET, a graph layout based on the
popular DR algorithm t-SNE. GUMAP runs 90% faster than tsNET with similar
neighborhood preservation and, on average, 10% better on quality metrics such
as stress, edge crossing, and shape-based metrics, validating the effectiveness
of UMAP for graph drawing.

### 4. [Geometric Interpretation of 3-SAT and Phase Transition](http://arxiv.org/pdf/2509.19740v1)

Authors: Frederic Gillet

Interpretation of 3-SAT as a volume filling problem, and its use to explore
the SAT/UNSAT phase transition.

### 5. [BH-tsNET, FIt-tsNET, L-tsNET: Fast tsNET Algorithms for Large Graph Drawing](http://arxiv.org/pdf/2509.19785v1)

Authors: Amyra Meidiana, Seok-Hee Hong, Kwan-Liu Ma

The tsNET algorithm utilizes t-SNE to compute high-quality graph drawings,
preserving the neighborhood and clustering structure. We present three fast
algorithms for reducing the time complexity of tsNET algorithm from O(nm) time
to O(n log n) time and O(n) time. To reduce the runtime of tsNET, there are
three components that need to be reduced: (C0) computation of high-dimensional
probabilities, (C1) computation of KL divergence gradient, and (C2) entropy
computation. Specifically, we reduce the overall runtime of tsNET, integrating
our new fast approaches for C0 and C2 with fast t-SNE algorithms for C1. We
first present O(n log n)-time BH-tsNET, based on (C0) new O(n)-time partial
BFS-based high-dimensional probability computation and (C2) new O(n log n)-time
quadtree-based entropy computation, integrated with (C1) O(n log n)-time
quadtree-based KL divergence computation of BH-SNE. We next present faster O(n
log n)-time FIt-tsNET, using (C0) O(n)-time partial BFS-based high-dimensional
probability computation and (C2) quadtree-based O(n log n)-time entropy
computation, integrated with (C1) O(n)-time interpolation-based KL divergence
computation of FIt-SNE. Finally, we present the O(n)-time L-tsNET, integrating
(C2) new O(n)-time FFT-accelerated interpolation-based entropy computation with
(C0) O(n)-time partial BFS-based high-dimensional probability computation, and
(C1) O(n)-time interpolation-based KL divergence computation of FIt-SNE.
Extensive experiments using benchmark data sets confirm that BH-tsNET,
FIt-tsNET, and L-tsNET outperform tsNET, running 93.5%, 96%, and 98.6% faster
while computing similar quality drawings in terms of quality metrics
(neighborhood preservation, stress, edge crossing, and shape-based metrics) and
visual comparison. We also present a comparison between our algorithms and
DRGraph, another dimension reduction-based graph drawing algorithm.

### 6. [Stealing From the Dragon's Hoard: Online Unbounded Knapsack With Removal](http://arxiv.org/pdf/2509.19914v1)

Authors: Matthias Gehnen, Moritz Stocker

We introduce the Online Unbounded Knapsack Problem with Removal, a variation
of the well-known Online Knapsack Problem. Items, each with a weight and value,
arrive online and an algorithm must decide on whether or not to pack them into
a knapsack with a fixed weight limit. An item may be packed an arbitrary number
of times and items may be removed from the knapsack at any time without cost.
The goal is to maximize the total value of items packed, while respecting a
weight limit. We show that this is one of the very few natural online knapsack
variants that allow for competitive deterministic algorithms in the general
setting, by providing an algorithm with competitivity 1.6911. We complement
this with a lower bound of 1.5877.
  We also analyze the proportional setting, where the weight and value of any
single item agree, and show that deterministic algorithms can be exactly
3/2-competitive. Lastly, we give lower and upper bounds of 6/5 and 4/3 on the
competitivity of randomized algorithms in this setting.

### 7. [Testable algorithms for approximately counting edges and triangles in sublinear time and space](http://arxiv.org/pdf/2509.20351v1)

Authors: Talya Eden, Ronitt Rubinfeld, Arsen Vasilyan

We consider the fundamental problems of approximately counting the numbers of
edges and triangles in a graph in sublinear time. Previous algorithms for these
tasks are significantly more efficient under a promise that the arboricity of
the graph is bounded by some parameter $\overline{\alpha}$. However, when this
promise is violated, the estimates given by these algorithms are no longer
guaranteed to be correct.
  For the triangle counting task, we give an algorithm that requires no promise
on the input graph $G$, and computes a $(1\pm \epsilon)$-approximation for the
number of triangles $t$ in $G$ in time $O^*\left( \frac{m\cdot \alpha(G)}{t} +
\frac{m}{t^{2/3}}
  \right)$, where $\alpha(G)$ is the arboricity of the graph. The algorithm can
be used on any graph $G$ (no prior knowledge the arboricity $\alpha(G)$ is
required), and the algorithm adapts its run-time on the fly based on the graph
$G$.
  We accomplish this by trying a sequence of candidate values $\tilde{\alpha}$
for $\alpha(G)$ and using a novel algorithm in the framework of testable
algorithms. This ensures that wrong candidates $\tilde{\alpha}$ cannot lead to
incorrect estimates: as long as the advice is incorrect, the algorithm detects
it and continues with a new candidate. Once the algorithm accepts the
candidate, its output is guaranteed to be correct with high probability.
  We prove that this approach preserves - up to an additive overhead - the
dramatic efficiency gains obtainable when good arboricity bounds are known in
advance, while ensuring robustness against misleading advice. We further
complement this result with a lower bound, showing that such an overhead is
unavoidable whenever the advice may be faulty.
  We further demonstrate implications of our results for triangle counting in
the streaming model.

### 8. [ALNS for Tugboat Scheduling in Inland Waterway](http://arxiv.org/pdf/2509.19718v1)

Authors: Zihang Ma

This paper focuses on the barges shipping problem, also known as the tugboats
scheduling problem, within the context of a scenario where a single tugboat has
the capacity to tow multiple barges and conduct multiple trips in a
drop-and-pull mode during a daily work shift. The problem is mathematically
formalized as mixed-integer programming models. To tackle real-world-sized
problem instances, an adaptive large neighborhood search (ALNS) algorithm
integrated with a decoding mathematical model is proposed. When applied to
large-scale instances, the ALNS algorithm showcases performance superiority
over the strengthened mathematical model.

### 9. [No Quantum Advantage in Decoded Quantum Interferometry for MaxCut](http://arxiv.org/pdf/2509.19966v1)

Authors: Ojas Parekh

Decoded Quantum Interferometry (DQI) is a framework for approximating special
kinds of discrete optimization problems that relies on problem structure in a
way that sets it apart from other classical or quantum approaches. We show that
the instances of MaxCut on which DQI attains a nontrivial asymptotic
approximation guarantee are solvable exactly in classical polynomial time. We
include a streamlined exposition of DQI tailored for MaxCut that relies on
elementary graph theory instead of coding theory to motivate and explain the
algorithm.

### 10. [Ads that Stick: Near-Optimal Ad Optimization through Psychological Behavior Models](http://arxiv.org/pdf/2509.20304v1)

Authors: Kailash Gopal Darmasubramanian, Akash Pareek, Arindam Khan, Arpit Agarwal

Optimizing the timing and frequency of ads is a central problem in digital
advertising, with significant economic consequences. Existing scheduling
policies rely on simple heuristics, such as uniform spacing and frequency caps,
that overlook long-term user interest. However, it is well-known that users'
long-term interest and engagement result from the interplay of several
psychological effects (Curmei, Haupt, Recht, Hadfield-Menell, ACM CRS, 2022).
  In this work, we model change in user interest upon showing ads based on
three key psychological principles: mere exposure, hedonic adaptation, and
operant conditioning. The first two effects are modeled using a concave
function of user interest with repeated exposure, while the third effect is
modeled using a temporal decay function, which explains the decline in user
interest due to overexposure. Under our psychological behavior model, we ask
the following question: Given a continuous time interval $T$, how many ads
should be shown, and at what times, to maximize the user interest towards the
ads?
  Towards answering this question, we first show that, if the number of
displayed ads is fixed, then the optimal ad-schedule only depends on the
operant conditioning function. Our main result is a quasi-linear time algorithm
that outputs a near-optimal ad-schedule, i.e., the difference in the
performance of our schedule and the optimal schedule is exponentially small.
Our algorithm leads to significant insights about optimal ad placement and
shows that simple heuristics such as uniform spacing are sub-optimal under many
natural settings. The optimal number of ads to display, which also depends on
the mere exposure and hedonistic adaptation functions, can be found through a
simple linear search given the above algorithm. We further support our findings
with experimental results, demonstrating that our strategy outperforms various
baselines.

### Emerging Technologies

### 1. [Digital Signal Processing from Classical Coherent Systems to Continuous-Variable QKD: A Review of Cross-Domain Techniques, Applications, and Challenges](http://arxiv.org/pdf/2509.20141v1)

Authors: Davi Juvêncio Gomes de Sousa, Caroline da Silva Morais Alves, Valéria Loureiro da Silva, Nelson Alves Ferreira Neto

This systematic review investigates the application of digital signal
processing (DSP) techniques -- originally developed for coherent optical
communication systems to continuous-variable quantum key distribution (CV-QKD).
The convergence of these domains has enabled significant advances in CV-QKD
performance, particularly in phase synchronization, polarization tracking, and
excess noise mitigation. To provide a comprehensive and reproducible synthesis
of this emerging field, we employed the APISSER methodology, a task-oriented
framework adapted from the PRISMA protocol. A structured search across IEEE
Xplore and Web of Science databases (2021-2025) yielded 220 relevant
publications, which were screened, classified, and analyzed to address six
research questions. Our findings highlight that many classical DSP algorithms,
such as Kalman filtering, carrier recovery, adaptive equalization, and
machine-learning-assisted signal estimation, have been successfully adapted to
the quantum regime, often requiring modifications to meet security and noise
constraints. We also identify a range of recent DSP innovations in coherent
optical communication systems with high potential for future CV-QKD
integration, including neural equalization, probabilistic shaping, and joint
retiming-equalization filters. Despite these advances, challenges remain in
achieving robust phase tracking under ultra-low Signal-to-Noise Ratio (SNR)
conditions, real-time polarization compensation, and secure co-existence with
classical channels. This review maps current trends, technical barriers, and
emerging opportunities at the intersection of signal processing for quantum and
classical communication, supporting the development of scalable and resilient
CV-QKD systems.

### Formal Languages and Automata Theory

### 1. [Scalable and Approximation-free Symbolic Control for Unknown Euler-Lagrange Systems](http://arxiv.org/pdf/2509.19859v1)

Authors: Ratnangshu Das, Shubham Sawarkar, Pushpak Jagtap

We propose a novel symbolic control framework for enforcing temporal logic
specifications in Euler-Lagrange systems that addresses the key limitations of
traditional abstraction-based approaches. Unlike existing methods that require
exact system models and provide guarantees only at discrete sampling instants,
our approach relies only on bounds on system parameters and input constraints,
and ensures correctness for the full continuous-time trajectory. The framework
combines scalable abstraction of a simplified virtual system with a
closed-form, model-free controller that guarantees trajectories satisfy the
original specification while respecting input bounds and remaining robust to
unknown but bounded disturbances. We provide feasibility conditions for the
construction of confinement regions and analyze the trade-off between
efficiency and conservatism. Case studies on pendulum dynamics, a two-link
manipulator, and multi-agent systems, including hardware experiments,
demonstrate that the proposed approach ensures both correctness and safety
while significantly reducing computational time and memory requirements. These
results highlight its scalability and practicality for real-world robotic
systems where precise models are unavailable and continuous-time guarantees are
essential.

### Graphics

### 1. [LidarScout: Direct Out-of-Core Rendering of Massive Point Clouds](http://arxiv.org/pdf/2509.20198v1)

Authors: Philipp Erler, Lukas Herzberger, Michael Wimmer, Markus Schütz

Large-scale terrain scans are the basis for many important tasks, such as
topographic mapping, forestry, agriculture, and infrastructure planning. The
resulting point cloud data sets are so massive in size that even basic tasks
like viewing take hours to days of pre-processing in order to create
level-of-detail structures that allow inspecting the data set in their entirety
in real time. In this paper, we propose a method that is capable of instantly
visualizing massive country-sized scans with hundreds of billions of points.
Upon opening the data set, we first load a sparse subsample of points and
initialize an overview of the entire point cloud, immediately followed by a
surface reconstruction process to generate higher-quality, hole-free
heightmaps. As users start navigating towards a region of interest, we continue
to prioritize the heightmap construction process to the user's viewpoint. Once
a user zooms in closely, we load the full-resolution point cloud data for that
region and update the corresponding height map textures with the
full-resolution data. As users navigate elsewhere, full-resolution point data
that is no longer needed is unloaded, but the updated heightmap textures are
retained as a form of medium level of detail. Overall, our method constitutes a
form of direct out-of-core rendering for massive point cloud data sets
(terabytes, compressed) that requires no preprocessing and no additional disk
space. Source code, executable, pre-trained model, and dataset are available
at: https://github.com/cg-tuwien/lidarscout

### 2. [AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](http://arxiv.org/pdf/2509.19939v1)

Authors: Hyunjin Cho, Giyun Choi, Jongwon Choi

Existing human mesh recovery methods assume a standard human body structure,
overlooking diverse anatomical conditions such as limb loss. This assumption
introduces bias when applied to individuals with amputations - a limitation
further exacerbated by the scarcity of suitable datasets. To address this gap,
we propose Amputated Joint Aware 3D Human Mesh Recovery (AJAHR), which is an
adaptive pose estimation framework that improves mesh reconstruction for
individuals with limb loss. Our model integrates a body-part amputation
classifier, jointly trained with the mesh recovery network, to detect potential
amputations. We also introduce Amputee 3D (A3D), which is a synthetic dataset
offering a wide range of amputee poses for robust training. While maintaining
competitive performance on non-amputees, our approach achieves state-of-the-art
results for amputated individuals. Additional materials can be found at the
project webpage.

### 3. [MeshMosaic: Scaling Artist Mesh Generation via Local-to-Global Assembly](http://arxiv.org/pdf/2509.19995v1)

Authors: Rui Xu, Tianyang Xue, Qiujie Dong, Le Wan, Zhe Zhu, Peng Li, Zhiyang Dou, Cheng Lin, Shiqing Xin, Yuan Liu, Wenping Wang, Taku Komura

Scaling artist-designed meshes to high triangle numbers remains challenging
for autoregressive generative models. Existing transformer-based methods suffer
from long-sequence bottlenecks and limited quantization resolution, primarily
due to the large number of tokens required and constrained quantization
granularity. These issues prevent faithful reproduction of fine geometric
details and structured density patterns. We introduce MeshMosaic, a novel
local-to-global framework for artist mesh generation that scales to over 100K
triangles--substantially surpassing prior methods, which typically handle only
around 8K faces. MeshMosaic first segments shapes into patches, generating each
patch autoregressively and leveraging shared boundary conditions to promote
coherence, symmetry, and seamless connectivity between neighboring regions.
This strategy enhances scalability to high-resolution meshes by quantizing
patches individually, resulting in more symmetrical and organized mesh density
and structure. Extensive experiments across multiple public datasets
demonstrate that MeshMosaic significantly outperforms state-of-the-art methods
in both geometric fidelity and user preference, supporting superior detail
representation and practical mesh generation for real-world applications.

### 4. [KSDiff: Keyframe-Augmented Speech-Aware Dual-Path Diffusion for Facial Animation](http://arxiv.org/pdf/2509.20128v1)

Authors: Tianle Lyu, Junchuan Zhao, Ye Wang

Audio-driven facial animation has made significant progress in multimedia
applications, with diffusion models showing strong potential for talking-face
synthesis. However, most existing works treat speech features as a monolithic
representation and fail to capture their fine-grained roles in driving
different facial motions, while also overlooking the importance of modeling
keyframes with intense dynamics. To address these limitations, we propose
KSDiff, a Keyframe-Augmented Speech-Aware Dual-Path Diffusion framework.
Specifically, the raw audio and transcript are processed by a Dual-Path Speech
Encoder (DPSE) to disentangle expression-related and head-pose-related
features, while an autoregressive Keyframe Establishment Learning (KEL) module
predicts the most salient motion frames. These components are integrated into a
Dual-path Motion generator to synthesize coherent and realistic facial motions.
Extensive experiments on HDTF and VoxCeleb demonstrate that KSDiff achieves
state-of-the-art performance, with improvements in both lip synchronization
accuracy and head-pose naturalness. Our results highlight the effectiveness of
combining speech disentanglement with keyframe-aware diffusion for talking-head
generation.

### Computer Science and Game Theory

### 1. [A Novel Framework for Honey-X Deception in Zero-Sum Games](http://arxiv.org/pdf/2509.20329v1)

Authors: Brendan Gould, Kyriakos Vamvoudakis

In this paper, we present a novel, game-theoretic model of deception in
two-player, zero-sum games. Our framework leverages an information asymmetry:
one player (the deceiver) has access to accurate payoff information, while the
other (the victim) observes a modified version of these payoffs due to the
deception strategy employed. The deceiver's objective is to choose a
deception-action pair that optimally exploits the victim's best response to the
altered payoffs, subject to a constraint on the deception's magnitude. We
characterize the optimal deceptive strategy as the solution to a bi-level
optimization problem, and we provide both an exact solution and an efficient
method for computing a high-quality feasible point. Finally, we demonstrate the
effectiveness of our approach on numerical examples inspired by honeypot
deception.

### 2. [On the Fragility of Contribution Score Computation in Federated Learning](http://arxiv.org/pdf/2509.19921v1)

Authors: Balazs Pejo, Marcell Frank, Krisztian Varga, Peter Veliczky

This paper investigates the fragility of contribution evaluation in federated
learning, a critical mechanism for ensuring fairness and incentivizing
participation. We argue that contribution scores are susceptible to significant
distortions from two fundamental perspectives: architectural sensitivity and
intentional manipulation. First, we explore how different model aggregation
methods impact these scores. While most research assumes a basic averaging
approach, we demonstrate that advanced techniques, including those designed to
handle unreliable or diverse clients, can unintentionally yet significantly
alter the final scores. Second, we explore vulnerabilities posed by poisoning
attacks, where malicious participants strategically manipulate their model
updates to inflate their own contribution scores or reduce the importance of
other participants. Through extensive experiments across diverse datasets and
model architectures, implemented within the Flower framework, we rigorously
show that both the choice of aggregation method and the presence of attackers
are potent vectors for distorting contribution scores, highlighting a critical
need for more robust evaluation schemes.

### 3. [Pure Exploration via Frank-Wolfe Self-Play](http://arxiv.org/pdf/2509.19901v1)

Authors: Xinyu Liu, Chao Qin, Wei You

We study pure exploration in structured stochastic multi-armed bandits,
aiming to efficiently identify the correct hypothesis from a finite set of
alternatives. For a broad class of tasks, asymptotic analyses reduce to a
maximin optimization that admits a two-player zero-sum game interpretation
between an experimenter and a skeptic: the experimenter allocates measurements
to rule out alternatives while the skeptic proposes alternatives. We
reformulate the game by allowing the skeptic to adopt a mixed strategy,
yielding a concave-convex saddle-point problem. This viewpoint leads to
Frank-Wolfe Self-Play (FWSP): a projection-free, regularization-free,
tuning-free method whose one-hot updates on both sides match the bandit
sampling paradigm. However, structural constraints introduce sharp pathologies
that complicate algorithm design and analysis: our linear-bandit case study
exhibits nonunique optima, optimal designs with zero mass on the best arm,
bilinear objectives, and nonsmoothness at the boundary. We address these
challenges via a differential-inclusion argument, proving convergence of the
game value for best-arm identification in linear bandits. Our analysis proceeds
through a continuous-time limit: a differential inclusion with a Lyapunov
function that decays exponentially, implying a vanishing duality gap and
convergence to the optimal value. Although Lyapunov analysis requires
differentiability of the objective, which is not guaranteed on the boundary, we
show that along continuous trajectories the algorithm steers away from
pathological nonsmooth points and achieves uniform global convergence to the
optimal game value. We then embed the discrete-time updates into a perturbed
flow and show that the discrete game value also converges. Building on FWSP, we
further propose a learning algorithm based on posterior sampling. Numerical
experiments demonstrate a vanishing duality gap.

### 4. [Choose Your Battles: Distributed Learning Over Multiple Tug of War Games](http://arxiv.org/pdf/2509.20147v1)

Authors: Siddharth Chandak, Ilai Bistritz, Nicholas Bambos

Consider N players and K games taking place simultaneously. Each of these
games is modeled as a Tug-of-War (ToW) game where increasing the action of one
player decreases the reward for all other players. Each player participates in
only one game at any given time. At each time step, a player decides the game
in which they wish to participate in and the action they take in that game.
Their reward depends on the actions of all players that are in the same game.
This system of K games is termed `Meta Tug-of-War' (Meta-ToW) game. These games
can model scenarios such as power control, distributed task allocation, and
activation in sensor networks. We propose the Meta Tug-of-Peace algorithm, a
distributed algorithm where the action updates are done using a simple
stochastic approximation algorithm, and the decision to switch games is made
using an infrequent 1-bit communication between the players. We prove that in
Meta-ToW games, our algorithm converges to an equilibrium that satisfies a
target Quality of Service reward vector for the players. We then demonstrate
the efficacy of our algorithm through simulations for the scenarios mentioned
above.

### Human-Computer Interaction

### 1. [Interactive Semantic Segmentation for Phosphene Vision Neuroprosthetics](http://arxiv.org/pdf/2509.19957v1)

Authors: Eleftherios Papadopoulos, Yagmur Güçlütürk

Visual impairments present significant challenges to individuals worldwide,
impacting daily activities and quality of life. Visual neuroprosthetics offer a
promising solution, leveraging advancements in technology to provide a
simplified visual sense through devices comprising cameras, computers, and
implanted electrodes. This study investigates user-centered design principles
for a phosphene vision algorithm, utilizing feedback from visually impaired
individuals to guide the development of a gaze-controlled semantic segmentation
system. We conducted interviews revealing key design principles. These
principles informed the implementation of a gaze-guided semantic segmentation
algorithm using the Segment Anything Model (SAM). In a simulated phosphene
vision environment, participants performed object detection tasks under SAM,
edge detection, and normal vision conditions. SAM improved identification
accuracy over edge detection, remained effective in complex scenes, and was
particularly robust for specific object shapes. These findings demonstrate the
value of user feedback and the potential of gaze-guided semantic segmentation
to enhance neuroprosthetic vision.

### 2. [Investigating the Effect of Prior Exposure and Fidelity on Quality and Realism Perception of VR Digital Twins](http://arxiv.org/pdf/2509.20106v1)

Authors: Maximilian Warsinke, Maurizio Vergari, Tanja Kojić, Daniel Nikulin, Sebastian Möller

This study explores how prior exposure to physical objects influences the
quality and realism perception of Digital Twins (DT) with varying levels of
fidelity in Virtual Reality (VR). In a mixed experimental design, 24
participants were divided into two equal groups: an exposure group, in which
members were shown physical objects before inspecting and rating their replicas
in VR, and a control group without prior knowledge. Three objects were
presented, each under four fidelity conditions with varying texture resolution
and geometric detail. Participants rated perceived quality and realism through
in-VR self-reports. Statistical analysis revealed that texture resolution
significantly affected realism and quality perception, whereas geometric detail
only influenced quality ratings. Investigating the between-factor, no
significant effect of exposure on quality and realism perception was found.
These findings raise important questions about the cognitive relationship
between physical objects and their digital counterparts and how fidelity
influences the perception of DTs in VR.

### 3. [Visual Tools for Input and Reflection in Social Work](http://arxiv.org/pdf/2509.20307v1)

Authors: Alexander Rind, Julia Boeck

Social workers need visual tools to collect information about their client's
life situation, so that they can reflect it together and choose tailored
interventions. easyNWK and easyBiograph are two visual tools for the client's
social network and life history. We recently redesigned both tools in a
participatory design project with social work faculty and professionals. In
this short paper we discuss these tools from perspective of input visualization
systems.

### 4. [Governing Together: Toward Infrastructure for Community-Run Social Media](http://arxiv.org/pdf/2509.19653v1)

Authors: Sohyeon Hwang, Sophie Rollins, Thatiany Andrade Nunes, Yuhan Liu, Richmond Wong, Aaron Shaw, Andrés Monroy-Hernández

Decentralizing the governance of social computing systems to communities
promises to empower them to make independent decisions, with nuance and in
accordance with their values. Yet, communities do not govern in isolation. Many
problems communities face are common, or move across their boundaries. We
therefore propose designing for "inter-community governance:" mechanisms that
support relationships and interactions between communities to coordinate on
governance issues. Drawing from workshops with 24 individuals on decentralized,
community-run social media, we present six challenges in designing for
inter-community governance surfaced through ideas proposed in workshops.
Together, these ideas come together as an ecosystem of resources,
infrastructures, and tools that highlight three key principles for designing
for inter-community governance: modularity, forkability, and polycentricity. We
end with a discussion of how the ideas proposed in workshops might be
implemented in future work aiming to support community governance in social
computing systems broadly.

### 5. [PolicyPad: Collaborative Prototyping of LLM Policies](http://arxiv.org/pdf/2509.19680v1)

Authors: K. J. Kevin Feng, Tzu-Sheng Kuo, Quan Ze, Chen, Inyoung Cheong, Kenneth Holstein, Amy X. Zhang

As LLMs gain adoption in high-stakes domains like mental health, domain
experts are increasingly consulted to provide input into policies governing
their behavior. From an observation of 19 policymaking workshops with 9 experts
over 15 weeks, we identified opportunities to better support rapid
experimentation, feedback, and iteration for collaborative policy design
processes. We present PolicyPad, an interactive system that facilitates the
emerging practice of LLM policy prototyping by drawing from established UX
prototyping practices, including heuristic evaluation and storyboarding. Using
PolicyPad, policy designers can collaborate on drafting a policy in real time
while independently testing policy-informed model behavior with usage
scenarios. We evaluate PolicyPad through workshops with 8 groups of 22 domain
experts in mental health and law, finding that PolicyPad enhanced collaborative
dynamics during policy design, enabled tight feedback loops, and led to novel
policy contributions. Overall, our work paves participatory paths for advancing
AI alignment and safety.

### 6. [How People Manage Knowledge in their "Second Brains"- A Case Study with Industry Researchers Using Obsidian](http://arxiv.org/pdf/2509.20187v1)

Authors: Juliana Jansen Ferreira, Vinícius Segura, Joana Gabriela Souza, Joao Henrique Gallas Brasil

People face overwhelming information during work activities, necessitating
effective organization and management strategies. Even in personal lives,
individuals must keep, annotate, organize, and retrieve knowledge from daily
routines. The collection of records for future reference is known as a personal
knowledge base. Note-taking applications are valuable tools for building and
maintaining these bases, often called a ''second brain''. This paper presents a
case study on how people build and explore personal knowledge bases for various
purposes. We selected the note-taking tool Obsidian and researchers from a
Brazilian lab for an in-depth investigation. Our investigation reveals
interesting findings about how researchers build and explore their personal
knowledge bases. A key finding is that participants' knowledge retrieval
strategy influences how they build and maintain their content. We suggest
potential features for an AI system to support this process.

### 7. [Agentic Metacognition: Designing a "Self-Aware" Low-Code Agent for Failure Prediction and Human Handoff](http://arxiv.org/pdf/2509.19783v1)

Authors: Jiexi Xu

The inherent non-deterministic nature of autonomous agents, particularly
within low-code/no-code (LCNC) environments, presents significant reliability
challenges. Agents can become trapped in unforeseen loops, generate inaccurate
outputs, or encounter unrecoverable failures, leading to user frustration and a
breakdown of trust. This report proposes a novel architectural pattern to
address these issues: the integration of a secondary, "metacognitive" layer
that actively monitors the primary LCNC agent. Inspired by human introspection,
this layer is designed to predict impending task failures based on a defined
set of triggers, such as excessive latency or repetitive actions. Upon
predicting a failure, the metacognitive agent proactively initiates a human
handoff, providing the user with a clear summary of the agent's "thought
process" and a detailed explanation of why it could not proceed. An empirical
analysis of a prototype system demonstrates that this approach significantly
increases the overall task success rate. However, this performance gain comes
with a notable increase in computational overhead. The findings reframe human
handoffs not as an admission of defeat but as a core design feature that
enhances system resilience, improves user experience, and builds trust by
providing transparency into the agent's internal state. The report discusses
the practical and ethical implications of this approach and identifies key
directions for future research.

### 8. [Queryable 3D Scene Representation: A Multi-Modal Framework for Semantic Reasoning and Robotic Task Planning](http://arxiv.org/pdf/2509.20077v1)

Authors: Xun Li, Rodrigo Santa Cruz, Mingze Xi, Hu Zhang, Madhawa Perera, Ziwei Wang, Ahalya Ravendran, Brandon J. Matthews, Feng Xu, Matt Adcock, Dadong Wang, Jiajun Liu

To enable robots to comprehend high-level human instructions and perform
complex tasks, a key challenge lies in achieving comprehensive scene
understanding: interpreting and interacting with the 3D environment in a
meaningful way. This requires a smart map that fuses accurate geometric
structure with rich, human-understandable semantics. To address this, we
introduce the 3D Queryable Scene Representation (3D QSR), a novel framework
built on multimedia data that unifies three complementary 3D representations:
(1) 3D-consistent novel view rendering and segmentation from panoptic
reconstruction, (2) precise geometry from 3D point clouds, and (3) structured,
scalable organization via 3D scene graphs. Built on an object-centric design,
the framework integrates with large vision-language models to enable semantic
queryability by linking multimodal object embeddings, and supporting
object-level retrieval of geometric, visual, and semantic information. The
retrieved data are then loaded into a robotic task planner for downstream
execution. We evaluate our approach through simulated robotic task planning
scenarios in Unity, guided by abstract language instructions and using the
indoor public dataset Replica. Furthermore, we apply it in a digital duplicate
of a real wet lab environment to test QSR-supported robotic task planning for
emergency response. The results demonstrate the framework's ability to
facilitate scene understanding and integrate spatial and semantic reasoning,
effectively translating high-level human instructions into precise robotic task
planning in complex 3D environments.

### 9. [Into the Void: Understanding Online Health Information in Low-Web Data Languages](http://arxiv.org/pdf/2509.20245v1)

Authors: Hellina Hailu Nigatu, Nuredin Ali Abdelkadir, Fiker Tewelde, Stevie Chancellor, Daricia Wilkinson

Data voids--areas of the internet where reliable information is scarce or
absent--pose significant challenges to online health information seeking,
particularly for users operating in low-web data languages. These voids are
increasingly encountered not on traditional search engines alone, but on social
media platforms, which have gradually morphed into informal search engines for
millions of people. In this paper, we introduce the phenomenon of data
horizons: a critical boundary where algorithmic structures begin to degrade the
relevance and reliability of search results. Unlike the core of a data void,
which is often exploited by bad actors to spread misinformation, the data
horizon marks the critical space where systemic factors, such as linguistic
underrepresentation, algorithmic amplification, and socio-cultural mismatch,
create conditions of informational instability. Focusing on Tigrinya and
Amharic as languages of study, we evaluate (1) the common characteristics of
search results for health queries, (2) the quality and credibility of health
information, and (3) characteristics of search results that diverge from their
queries. We find that search results for health queries in low-web data
languages may not always be in the language of search and may be dominated by
nutritional and religious advice. We show that search results that diverge from
their queries in low-resourced languages are due to algorithmic failures,
(un)intentional manipulation, or active manipulation by content creators. We
use our findings to illustrate how a data horizon manifests under several
interacting constraints on information availability.

### 10. [Muse-it: A Tool for Analyzing Music Discourse on Reddit](http://arxiv.org/pdf/2509.20228v1)

Authors: Jatin Agarwala, George Paul, Nemani Harsha Vardhan, Vinoo Alluri

Music engagement spans diverse interactions with music, from selection and
emotional response to its impact on behavior, identity, and social connections.
Social media platforms provide spaces where such engagement can be observed in
natural, unprompted conversations. Advances in natural language processing
(NLP) and big data analytics make it possible to analyze these discussions at
scale, extending music research to broader contexts. Reddit, in particular,
offers anonymity that encourages diverse participation and yields rich
discourse on music in ecological settings. Yet the scale of this data requires
tools to extract, process, and analyze it effectively. We present Muse-it, a
platform that retrieves comprehensive Reddit data centered on user-defined
queries. It aggregates posts from across subreddits, supports topic modeling,
temporal trend analysis, and clustering, and enables efficient study of
large-scale discourse. Muse-it also identifies music-related hyperlinks (e.g.,
Spotify), retrieves track-level metadata such as artist, album, release date,
genre, popularity, and lyrics, and links these to the discussions. An
interactive interface provides dynamic visualizations of the collected data.
Muse-it thus offers an accessible way for music researchers to gather and
analyze big data, opening new avenues for understanding music engagement as it
naturally unfolds online.

### Information Retrieval

### 1. [Learning Contextual Retrieval for Robust Conversational Search](http://arxiv.org/pdf/2509.19700v1)

Authors: Seunghan Yang, Juntae Lee, Jihwan Bang, Kyuhong Shim, Minsoo Kim, Simyung Chang

Effective conversational search demands a deep understanding of user intent
across multiple dialogue turns. Users frequently use abbreviations and shift
topics in the middle of conversations, posing challenges for conventional
retrievers. While query rewriting techniques improve clarity, they often incur
significant computational cost due to additional autoregressive steps.
Moreover, although LLM-based retrievers demonstrate strong performance, they
are not explicitly optimized to track user intent in multi-turn settings, often
failing under topic drift or contextual ambiguity. To address these
limitations, we propose ContextualRetriever, a novel LLM-based retriever that
directly incorporates conversational context into the retrieval process. Our
approach introduces: (1) a context-aware embedding mechanism that highlights
the current query within the dialogue history; (2) intent-guided supervision
based on high-quality rewritten queries; and (3) a training strategy that
preserves the generative capabilities of the base LLM. Extensive evaluations
across multiple conversational search benchmarks demonstrate that
ContextualRetriever significantly outperforms existing methods while incurring
no additional inference overhead.

### 2. [Adaptive User Interest Modeling via Conditioned Denoising Diffusion For Click-Through Rate Prediction](http://arxiv.org/pdf/2509.19876v1)

Authors: Qihang Zhao, Xiaoyang Zheng, Ben Chen, Zhongbo Sun, Chenyi Lei

User behavior sequences in search systems resemble "interest fossils",
capturing genuine intent yet eroded by exposure bias, category drift, and
contextual noise. Current methods predominantly follow an "identify-aggregate"
paradigm, assuming sequences immutably reflect user preferences while
overlooking the organic entanglement of noise and genuine interest. Moreover,
they output static, context-agnostic representations, failing to adapt to
dynamic intent shifts under varying Query-User-Item-Context conditions.
  To resolve this dual challenge, we propose the Contextual Diffusion Purifier
(CDP). By treating category-filtered behaviors as "contaminated observations",
CDP employs a forward noising and conditional reverse denoising process guided
by cross-interaction features (Query x User x Item x Context), controllably
generating pure, context-aware interest representations that dynamically evolve
with scenarios. Extensive offline/online experiments demonstrate the
superiority of CDP over state-of-the-art methods.

### 3. [Documentation Retrieval Improves Planning Language Generation](http://arxiv.org/pdf/2509.19931v1)

Authors: Renxiang Wang, Li Zhang

Certain strong LLMs have shown promise for zero-shot formal planning by
generating planning languages like PDDL. Yet, performance of most open-source
models under 50B parameters has been reported to be close to zero due to the
low-resource nature of these languages. We significantly improve their
performance via a series of lightweight pipelines that integrates documentation
retrieval with modular code generation and error refinement. With models like
Llama-4-Maverick, our best pipeline improves plan correctness from 0\% to over
80\% on the common BlocksWorld domain. However, while syntactic errors are
substantially reduced, semantic errors persist in more challenging domains,
revealing fundamental limitations in current models' reasoning
capabilities.\footnote{Our code and data can be found at
https://github.com/Nangxxxxx/PDDL-RAG

### 4. [Multimodal-enhanced Federated Recommendation: A Group-wise Fusion Approach](http://arxiv.org/pdf/2509.19955v1)

Authors: Chunxu Zhang, Weipeng Zhang, Guodong Long, Zhiheng Xue, Riting Xia, Bo Yang

Federated Recommendation (FR) is a new learning paradigm to tackle the
learn-to-rank problem in a privacy-preservation manner. How to integrate
multi-modality features into federated recommendation is still an open
challenge in terms of efficiency, distribution heterogeneity, and fine-grained
alignment. To address these challenges, we propose a novel multimodal fusion
mechanism in federated recommendation settings (GFMFR). Specifically, it
offloads multimodal representation learning to the server, which stores item
content and employs a high-capacity encoder to generate expressive
representations, alleviating client-side overhead. Moreover, a group-aware item
representation fusion approach enables fine-grained knowledge sharing among
similar users while retaining individual preferences. The proposed fusion loss
could be simply plugged into any existing federated recommender systems
empowering their capability by adding multi-modality features. Extensive
experiments on five public benchmark datasets demonstrate that GFMFR
consistently outperforms state-of-the-art multimodal FR baselines.

### 5. [Cascade! Human in the loop shortcomings can increase the risk of failures in recommender systems](http://arxiv.org/pdf/2509.20099v1)

Authors: Wm. Matthew Kennedy, Nishanshi Shukla, Cigdem Patlak, Blake Chambers, Theodora Skeadas, Tuesday, Kingsley Owadara, Aayush Dhanotiya

Recommender systems are among the most commonly deployed systems today.
Systems design approaches to AI-powered recommender systems have done well to
urge recommender system developers to follow more intentional data collection,
curation, and management procedures. So too has the "human-in-the-loop"
paradigm been widely adopted, primarily to address the issue of accountability.
However, in this paper, we take the position that human oversight in
recommender system design also entails novel risks that have yet to be fully
described. These risks are "codetermined" by the information context in which
such systems are often deployed. Furthermore, new knowledge of the shortcomings
of "human-in-the-loop" practices to deliver meaningful oversight of other AI
systems suggest that they may also be inadequate for achieving socially
responsible recommendations. We review how the limitations of human oversight
may increase the chances of a specific kind of failure: a "cascade" or
"compound" failure. We then briefly explore how the unique dynamics of three
common deployment contexts can make humans in the loop more likely to fail in
their oversight duties. We then conclude with two recommendations.

### 6. [Intelligent Algorithm Selection for Recommender Systems: Meta-Learning via in-depth algorithm feature engineering](http://arxiv.org/pdf/2509.20134v1)

Authors: Jarne Mathi Decker

The "No Free Lunch" theorem dictates that no single recommender algorithm is
optimal for all users, creating a significant Algorithm Selection Problem.
Standard meta-learning approaches aim to solve this by selecting an algorithm
based on user features, but treat the fundamentally diverse algorithms
themselves as equivalent, "black-box" choices. This thesis investigates the
impact of overcoming this limitation by engineering a comprehensive feature set
to explicitly characterize the algorithms themselves. We combine static code
metrics, Abstract Syntax Tree properties, behavioral performance landmarks, and
high-level conceptual features. We evaluate two meta-learners across five
datasets: a baseline using only user features and our proposed model using both
user and algorithm features. Our results show that the meta-learner augmented
with algorithm features achieves an average NDCG@10 of 0.143, a statistically
significant improvement of 11.7% over the Single Best Algorithm baseline
(0.128). However, we found that the inclusion of algorithm features did not
lead to an improvement in overall NDCG@10 over the meta learner using only user
features (0.144). While adding algorithm features to the meta-learner did
improve its Top-1 selection accuracy (+16.1%), this was counterbalanced by
leading to a lower Top-3 accuracy (-10.7%). We conclude that for the per-user
algorithm selection task in recommender systems, the predictive power of user
features is overwhelmingly dominant. While algorithm features improve selection
precision, unlocking their potential to boost overall performance remains a
non-trivial challenge.

### 7. [Multimodal Representation-disentangled Information Bottleneck for Multimodal Recommendation](http://arxiv.org/pdf/2509.20225v1)

Authors: Hui Wang, Jinghui Qin, Wushao Wen, Qingling Li, Shanshan Zhong, Zhongzhan Huang

Multimodal data has significantly advanced recommendation systems by
integrating diverse information sources to model user preferences and item
characteristics. However, these systems often struggle with redundant and
irrelevant information, which can degrade performance. Most existing methods
either fuse multimodal information directly or use rigid architectural
separation for disentanglement, failing to adequately filter noise and model
the complex interplay between modalities. To address these challenges, we
propose a novel framework, the Multimodal Representation-disentangled
Information Bottleneck (MRdIB). Concretely, we first employ a Multimodal
Information Bottleneck to compress the input representations, effectively
filtering out task-irrelevant noise while preserving rich semantic information.
Then, we decompose the information based on its relationship with the
recommendation target into unique, redundant, and synergistic components. We
achieve this decomposition with a series of constraints: a unique information
learning objective to preserve modality-unique signals, a redundant information
learning objective to minimize overlap, and a synergistic information learning
objective to capture emergent information. By optimizing these objectives,
MRdIB guides a model to learn more powerful and disentangled representations.
Extensive experiments on several competitive models and three benchmark
datasets demonstrate the effectiveness and versatility of our MRdIB in
enhancing multimodal recommendation.

### 8. [DyBBT: Dynamic Balance via Bandit inspired Targeting for Dialog Policy with Cognitive Dual-Systems](http://arxiv.org/pdf/2509.19695v1)

Authors: Shuyu Zhang, Yifan Wei, Jialuo Yuan, Xinru Wang, Yanmin Zhu, Bin Li

Task oriented dialog systems often rely on static exploration strategies that
do not adapt to dynamic dialog contexts, leading to inefficient exploration and
suboptimal performance. We propose DyBBT, a novel dialog policy learning
framework that formalizes the exploration challenge through a structured
cognitive state space capturing dialog progression, user uncertainty, and slot
dependency. DyBBT proposes a bandit inspired meta-controller that dynamically
switches between a fast intuitive inference (System 1) and a slow deliberative
reasoner (System 2) based on real-time cognitive states and visitation counts.
Extensive experiments on single- and multi-domain benchmarks show that DyBBT
achieves state-of-the-art performance in success rate, efficiency, and
generalization, with human evaluations confirming its decisions are well
aligned with expert judgment. Code is available at
https://github.com/carsonz/DyBBT.

### 9. [HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST](http://arxiv.org/pdf/2509.19742v1)

Authors: Shuyu Zhang, Yifan Wei, Xinru Wang, Yanmin Zhu, Yangfan He, Yixuan Weng, Bin Li

Zero-shot Dialog State Tracking (zs-DST) is essential for enabling
Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly
data annotation. A central challenge lies in the semantic misalignment between
dynamic dialog contexts and static prompts, leading to inflexible cross-layer
coordination, domain interference, and catastrophic forgetting. To tackle this,
we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a
framework that enhances zero-shot slot inference through robust prompt
alignment. It features a hierarchical LoRA architecture for dynamic
layer-specific processing (combining lower-layer heuristic grouping and
higher-layer full interaction), integrates Spectral Joint Domain-Slot
Clustering to identify transferable associations (feeding an Adaptive Linear
Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization
(SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain
datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving
SOTA in zs-DST. Code is available at https://github.com/carsonz/HiCoLoRA.

### 10. [FusedANN: Convexified Hybrid ANN via Attribute-Vector Fusion](http://arxiv.org/pdf/2509.19767v1)

Authors: Alireza Heidari, Wei Zhang, Ying Xiong

Vector search powers transformers technology, but real-world use demands
hybrid queries that combine vector similarity with attribute filters (e.g.,
"top document in category X, from 2023"). Current solutions trade off recall,
speed, and flexibility, relying on fragile index hacks that don't scale. We
introduce FusedANN (Fused Attribute-Vector Nearest Neighbor), a geometric
framework that elevates filtering to ANN optimization constraints and
introduces a convex fused space via a Lagrangian-like relaxation. Our method
jointly embeds attributes and vectors through transformer-based
convexification, turning hard filters into continuous, weighted penalties that
preserve top-k semantics while enabling efficient approximate search. We prove
that FusedANN reduces to exact filtering under high selectivity, gracefully
relaxes to semantically nearest attributes when exact matches are insufficient,
and preserves downstream ANN alpha-approximation guarantees. Empirically,
FusedANN improves query throughput by eliminating brittle filtering stages,
achieving superior recall-latency tradeoffs on standard hybrid benchmarks
without specialized index hacks, delivering up to 3 times higher throughput and
better recall than state-of-the-art hybrid and graph-based systems.
Theoretically, we provide explicit error bounds and parameter selection rules
that make FusedANN practical for production. This establishes a principled,
scalable, and verifiable bridge between symbolic constraints and vector
similarity, unlocking a new generation of filtered retrieval systems for large,
hybrid, and dynamic NLP/ML workloads.

### Machine Learning

### 1. [Symbol-Temporal Consistency Self-supervised Learning for Robust Time Series Classification](http://arxiv.org/pdf/2509.19654v1)

Authors: Kevin Garcia, Cassandra Garza, Brooklyn Berry, Yifeng Gao

The surge in the significance of time series in digital health domains
necessitates advanced methodologies for extracting meaningful patterns and
representations. Self-supervised contrastive learning has emerged as a
promising approach for learning directly from raw data. However, time series
data in digital health is known to be highly noisy, inherently involves concept
drifting, and poses a challenge for training a generalizable deep learning
model. In this paper, we specifically focus on data distribution shift caused
by different human behaviors and propose a self-supervised learning framework
that is aware of the bag-of-symbol representation. The bag-of-symbol
representation is known for its insensitivity to data warping, location shifts,
and noise existed in time series data, making it potentially pivotal in guiding
deep learning to acquire a representation resistant to such data shifting. We
demonstrate that the proposed method can achieve significantly better
performance where significant data shifting exists.

### 2. [Consistent Estimation of Numerical Distributions under Local Differential Privacy by Wavelet Expansion](http://arxiv.org/pdf/2509.19661v1)

Authors: Puning Zhao, Zhikun Zhang, Bo Sun, Li Shen, Liang Zhang, Shaowei Wang, Zhe Liu

Distribution estimation under local differential privacy (LDP) is a
fundamental and challenging task. Significant progresses have been made on
categorical data. However, due to different evaluation metrics, these methods
do not work well when transferred to numerical data. In particular, we need to
prevent the probability mass from being misplaced far away. In this paper, we
propose a new approach that express the sample distribution using wavelet
expansions. The coefficients of wavelet series are estimated under LDP. Our
method prioritizes the estimation of low-order coefficients, in order to ensure
accurate estimation at macroscopic level. Therefore, the probability mass is
prevented from being misplaced too far away from its ground truth. We establish
theoretical guarantees for our methods. Experiments show that our wavelet
expansion method significantly outperforms existing solutions under Wasserstein
and KS distances.

### 3. [Revisiting Performance Claims for Chest X-Ray Models Using Clinical Context](http://arxiv.org/pdf/2509.19671v1)

Authors: Andrew Wang, Jiashuo Zhang, Michael Oberst

Public healthcare datasets of Chest X-Rays (CXRs) have long been a popular
benchmark for developing computer vision models in healthcare. However, strong
average-case performance of machine learning (ML) models on these datasets is
insufficient to certify their clinical utility. In this paper, we use clinical
context, as captured by prior discharge summaries, to provide a more holistic
evaluation of current ``state-of-the-art'' models for the task of CXR
diagnosis. Using discharge summaries recorded prior to each CXR, we derive a
``prior'' or ``pre-test'' probability of each CXR label, as a proxy for
existing contextual knowledge available to clinicians when interpreting CXRs.
Using this measure, we demonstrate two key findings: First, for several
diagnostic labels, CXR models tend to perform best on cases where the pre-test
probability is very low, and substantially worse on cases where the pre-test
probability is higher. Second, we use pre-test probability to assess whether
strong average-case performance reflects true diagnostic signal, rather than an
ability to infer the pre-test probability as a shortcut. We find that
performance drops sharply on a balanced test set where this shortcut does not
exist, which may indicate that much of the apparent diagnostic power derives
from inferring this clinical context. We argue that this style of analysis,
using context derived from clinical notes, is a promising direction for more
rigorous and fine-grained evaluation of clinical vision models.

### 4. [Faster, Smaller, and Smarter: Task-Aware Expert Merging for Online MoE Inference](http://arxiv.org/pdf/2509.19781v1)

Authors: Ziyi Han, Xutong Liu, Ruiting Zhou, Xiangxiang Dai, John C. S. Lui

Sparse Mixture of Experts (SMoE) has become a preferred architecture for
scaling Transformer capacity without increasing computational cost, as it
activates only a small subset of experts for each input. However, deploying
such an approach for \textit{online inference} remains challenging due to the
large size of a full SMoE model and the complexity of expert routing,
especially in resource-constrained edge networks. Moreover, during the online
inference, task information is often unavailable, making the task-level routing
error-prone. In this work, we propose a novel tree-structured adaptive neural
bandit router, \texttt{Tanbr}, to enable efficient and reliable online MoE
inference. Instead of relying on explicit task tags, \texttt{Tanbr} estimates
the task distribution over time from historical data and uses it to guide
task-aware expert merging within a given pre-trained MoE. To handle the large
continuous space of merging weights, \texttt{Tanbr} employs a binary tree to
progressively partition the space and generate finer candidate weights. It then
applies a neural bandit to learn the non-linear mapping from merging weight to
model performance and decides optimal expert merging. We prove that
\texttt{Tanbr} achieves a sublinear regret bound of {\small
$\mathcal{O}(\sqrt{T} \log(T))$} over {\small $T$} rounds, despite operating
over a continuous decision space, matching regret bounds compared to existing
methods. Extensive experiments show that \texttt{Tanbr} reduces inference
latency by at least {\small $45\%$} and memory usage by up to {\small $25\%$},
while maintaining a high accuracy compared to many state-of-the-art methods.

### 5. [An Efficient Conditional Score-based Filter for High Dimensional Nonlinear Filtering Problems](http://arxiv.org/pdf/2509.19816v1)

Authors: Zhijun Zeng, Weiye Gan, Junqing Chen, Zuoqiang Shi

In many engineering and applied science domains, high-dimensional nonlinear
filtering is still a challenging problem. Recent advances in score-based
diffusion models offer a promising alternative for posterior sampling but
require repeated retraining to track evolving priors, which is impractical in
high dimensions. In this work, we propose the Conditional Score-based Filter
(CSF), a novel algorithm that leverages a set-transformer encoder and a
conditional diffusion model to achieve efficient and accurate posterior
sampling without retraining. By decoupling prior modeling and posterior
sampling into offline and online stages, CSF enables scalable score-based
filtering across diverse nonlinear systems. Extensive experiments on benchmark
problems show that CSF achieves superior accuracy, robustness, and efficiency
across diverse nonlinear filtering scenarios.

### 6. [BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management](http://arxiv.org/pdf/2509.19846v1)

Authors: Kevin Bradley Dsouza, Enoch Ofosu, Daniel Chukwuemeka Amaogu, Jérôme Pigeon, Richard Boudreault, Pooneh Maghoul, Juan Moreno-Cruz, Yuri Leonenko

Boreal forests store 30-40% of terrestrial carbon, much in climate-vulnerable
permafrost soils, making their management critical for climate mitigation.
However, optimizing forest management for both carbon sequestration and
permafrost preservation presents complex trade-offs that current tools cannot
adequately address. We introduce $\textbf{BoreaRL}$, the first multi-objective
reinforcement learning environment for climate-adaptive boreal forest
management, featuring a physically-grounded simulator of coupled energy,
carbon, and water fluxes. BoreaRL supports two training paradigms:
site-specific mode for controlled studies and generalist mode for learning
robust policies under environmental stochasticity. Through evaluation of
multi-objective RL algorithms, we reveal a fundamental asymmetry in learning
difficulty: carbon objectives are significantly easier to optimize than thaw
(permafrost preservation) objectives, with thaw-focused policies showing
minimal learning progress across both paradigms. In generalist settings,
standard preference-conditioned approaches fail entirely, while a naive
curriculum learning approach achieves superior performance by strategically
selecting training episodes. Analysis of learned strategies reveals distinct
management philosophies, where carbon-focused policies favor aggressive
high-density coniferous stands, while effective multi-objective policies
balance species composition and density to protect permafrost while maintaining
carbon gains. Our results demonstrate that robust climate-adaptive forest
management remains challenging for current MORL methods, establishing BoreaRL
as a valuable benchmark for developing more effective approaches. We
open-source BoreaRL to accelerate research in multi-objective RL for climate
applications.

### 7. [Oversampling and Downsampling with Core-Boundary Awareness: A Data Quality-Driven Approach](http://arxiv.org/pdf/2509.19856v1)

Authors: Samir Brahim Belhaouari, Yunis Carreon Kahalan, Humaira Shaffique, Ismael Belhaouari, Ashhadul Islam

The effectiveness of machine learning models, particularly in unbalanced
classification tasks, is often hindered by the failure to differentiate between
critical instances near the decision boundary and redundant samples
concentrated in the core of the data distribution. In this paper, we propose a
method to systematically identify and differentiate between these two types of
data. Through extensive experiments on multiple benchmark datasets, we show
that the boundary data oversampling method improves the F1 score by up to 10\%
on 96\% of the datasets, whereas our core-aware reduction method compresses
datasets up to 90\% while preserving their accuracy, making it 10 times more
powerful than the original dataset. Beyond imbalanced classification, our
method has broader implications for efficient model training, particularly in
computationally expensive domains such as Large Language Model (LLM) training.
By prioritizing high-quality, decision-relevant data, our approach can be
extended to text, multimodal, and self-supervised learning scenarios, offering
a pathway to faster convergence, improved generalization, and significant
computational savings. This work paves the way for future research in
data-efficient learning, where intelligent sampling replaces brute-force
expansion, driving the next generation of AI advancements. Our code is
available as a Python package at https://pypi.org/project/adaptive-resampling/ .

### 8. [MCGrad:: Multicalibration at Web Scale](http://arxiv.org/pdf/2509.19884v1)

Authors: Lorenzo Perini, Daniel Haimovich, Fridolin Linder, Niek Tax, Dima Karamshuk, Milan Vojnovic, Nastaran Okati, Pavlos Athanasios Apostolopoulos

We propose MCGrad, a novel and scalable multicalibration algorithm.
Multicalibration - calibration in sub-groups of the data - is an important
property for the performance of machine learning-based systems. Existing
multicalibration methods have thus far received limited traction in industry.
We argue that this is because existing methods (1) require such subgroups to be
manually specified, which ML practitioners often struggle with, (2) are not
scalable, or (3) may harm other notions of model performance such as log loss
and Area Under the Precision-Recall Curve (PRAUC). MCGrad does not require
explicit specification of protected groups, is scalable, and often improves
other ML evaluation metrics instead of harming them. MCGrad has been in
production at Meta, and is now part of hundreds of production models. We
present results from these deployments as well as results on public datasets.

### 9. [Latent Iterative Refinement Flow: A Geometric-Constrained Approach for Few-Shot Generation](http://arxiv.org/pdf/2509.19903v1)

Authors: Songtao Li, Zhenyu Liao, Tianqi Hou, Ting Gao

Few-shot generation, the synthesis of high-quality and diverse samples from
limited training data, remains a significant challenge in generative modeling.
Existing methods trained from scratch often fail to overcome overfitting and
mode collapse, and fine-tuning large models can inherit biases while neglecting
the crucial geometric structure of the latent space. To address these
limitations, we introduce Latent Iterative Refinement Flow (LIRF), a novel
approach that reframes few-shot generation as the progressive densification of
geometrically structured manifold. LIRF establishes a stable latent space using
an autoencoder trained with our novel \textbf{manifold-preservation loss}
$L_{\text{manifold}}$. This loss ensures that the latent space maintains the
geometric and semantic correspondence of the input data. Building on this, we
propose an iterative generate-correct-augment cycle. Within this cycle,
candidate samples are refined by a geometric \textbf{correction operator}, a
provably contractive mapping that pulls samples toward the data manifold while
preserving diversity. We also provide the \textbf{Convergence Theorem}
demonstrating a predictable decrease in Hausdorff distance between generated
and true data manifold. We also demonstrate the framework's scalability by
generating coherent, high-resolution images on AFHQ-Cat. Ablation studies
confirm that both the manifold-preserving latent space and the contractive
correction mechanism are critical components of this success. Ultimately, LIRF
provides a solution for data-scarce generative modeling that is not only
theoretically grounded but also highly effective in practice.

### 10. [MMSE-Calibrated Few-Shot Prompting for Alzheimer's Detection](http://arxiv.org/pdf/2509.19926v1)

Authors: Jana Sweidan, Mounim A. El-Yacoubi, Nasredine Semmar

Prompting large language models is a training-free method for detecting
Alzheimer's disease from speech transcripts. Using the ADReSS dataset, we
revisit zero-shot prompting and study few-shot prompting with a class-balanced
protocol using nested interleave and a strict schema, sweeping up to 20
examples per class. We evaluate two variants achieving state-of-the-art
prompting results. (i) MMSE-Proxy Prompting: each few-shot example carries a
probability anchored to Mini-Mental State Examination bands via a deterministic
mapping, enabling AUC computing; this reaches 0.82 accuracy and 0.86 AUC (ii)
Reasoning-augmented Prompting: few-shot examples pool is generated with a
multimodal LLM (GPT-5) that takes as input the Cookie Theft image, transcript,
and MMSE to output a reasoning and MMSE-aligned probability; evaluation remains
transcript-only and reaches 0.82 accuracy and 0.83 AUC. To our knowledge, this
is the first ADReSS study to anchor elicited probabilities to MMSE and to use
multimodal construction to improve interpretability.

### Neural and Evolutionary Computing

### 1. [Fully Tensorized GPU-accelerated Multi-population Evolutionary Algorithm for Constrained Multiobjective Optimization Problems](http://arxiv.org/pdf/2509.19821v1)

Authors: Weixiong Huang, Rui Wang, Wenhua Li, Sheng Qi, Tianyu Luo, Delong Chen, Tao Zhang, Ling Wang

Real world constrained multiobjective optimization problems (CMOPs) are
prevalent and often come with stringent time-sensitive requirements. However,
most contemporary constrained multiobjective evolutionary algorithms (CMOEAs)
suffer from a number of drawbacks, including complex designs, low computational
efficiency, and long convergence times, which are particularly pronounced when
addressing time-sensitive CMOPs. Although research on accelerating evolutionary
algorithms using GPU parallelism has advanced, existing CMOEAs still face
significant limitations within GPU frameworks. To overcome these challenges,
this paper proposes a GPU-accelerated multi-population evolutionary algorithm,
termed GMPEA. We first systematically analyze the performance bottlenecks of
representative CMOEAs when implemented in a GPU environment. To address the
trade-off between computational speed and solution performance, GMPEA
introduces a decomposition-based multi-population approach that is fully
parallelized across its entire workflow. We conducted comparative experiments
on various benchmark tests and real world applications: the Weapon Target
Assignment Problems. The results demonstrate that GMPEA achieves competitive
performance even without time constraints, while its computational speed
significantly surpasses that of the compared algorithms. More critically, under
a strict time limit, the performance of GMPEA drastically outperforms its
counterparts. This work provides compelling evidence of GMPEA's superiority in
solving time-sensitive CMOPs.

### 2. [Biologically Plausible Learning via Bidirectional Spike-Based Distillation](http://arxiv.org/pdf/2509.20284v1)

Authors: Changze Lv, Yifei Wang, Yanxun Zhang, Yiyang Lu, Jingwen Xu, Di Yu, Xin Du, Xuanjing Huang, Xiaoqing Zheng

Developing biologically plausible learning algorithms that can achieve
performance comparable to error backpropagation remains a longstanding
challenge. Existing approaches often compromise biological plausibility by
entirely avoiding the use of spikes for error propagation or relying on both
positive and negative learning signals, while the question of how spikes can
represent negative values remains unresolved. To address these limitations, we
introduce Bidirectional Spike-based Distillation (BSD), a novel learning
algorithm that jointly trains a feedforward and a backward spiking network. We
formulate learning as a transformation between two spiking representations
(i.e., stimulus encoding and concept encoding) so that the feedforward network
implements perception and decision-making by mapping stimuli to actions, while
the backward network supports memory recall by reconstructing stimuli from
concept representations. Extensive experiments on diverse benchmarks, including
image recognition, image generation, and sequential regression, show that BSD
achieves performance comparable to networks trained with classical error
backpropagation. These findings represent a significant step toward
biologically grounded, spike-driven learning in neural networks.

### 3. [Projective Kolmogorov Arnold Neural Networks (P-KANs): Entropy-Driven Functional Space Discovery for Interpretable Machine Learning](http://arxiv.org/pdf/2509.20049v1)

Authors: Alastair Poole, Stig McArthur, Saravan Kumar

Kolmogorov-Arnold Networks (KANs) relocate learnable nonlinearities from
nodes to edges, demonstrating remarkable capabilities in scientific machine
learning and interpretable modeling. However, current KAN implementations
suffer from fundamental inefficiencies due to redundancy in high-dimensional
spline parameter spaces, where numerous distinct parameterisations yield
functionally equivalent behaviors. This redundancy manifests as a "nuisance
space" in the model's Jacobian, leading to susceptibility to overfitting and
poor generalization. We introduce Projective Kolmogorov-Arnold Networks
(P-KANs), a novel training framework that guides edge function discovery
towards interpretable functional representations through entropy-minimisation
techniques from signal analysis and sparse dictionary learning. Rather than
constraining functions to predetermined spaces, our approach maintains spline
space flexibility while introducing "gravitational" terms that encourage
convergence towards optimal functional representations. Our key insight
recognizes that optimal representations can be identified through entropy
analysis of projection coefficients, compressing edge functions to
lower-parameter projective spaces (Fourier, Chebyshev, Bessel). P-KANs
demonstrate superior performance across multiple domains, achieving up to 80%
parameter reduction while maintaining representational capacity, significantly
improved robustness to noise compared to standard KANs, and successful
application to industrial automated fiber placement prediction. Our approach
enables automatic discovery of mixed functional representations where different
edges converge to different optimal spaces, providing both compression benefits
and enhanced interpretability for scientific machine learning applications.

### 4. [Predictive Coding-based Deep Neural Network Fine-tuning for Computationally Efficient Domain Adaptation](http://arxiv.org/pdf/2509.20269v1)

Authors: Matteo Cardoni, Sam Leroux

As deep neural networks are increasingly deployed in dynamic, real-world
environments, relying on a single static model is often insufficient. Changes
in input data distributions caused by sensor drift or lighting variations
necessitate continual model adaptation. In this paper, we propose a hybrid
training methodology that enables efficient on-device domain adaptation by
combining the strengths of Backpropagation and Predictive Coding. The method
begins with a deep neural network trained offline using Backpropagation to
achieve high initial performance. Subsequently, Predictive Coding is employed
for online adaptation, allowing the model to recover accuracy lost due to
shifts in the input data distribution. This approach leverages the robustness
of Backpropagation for initial representation learning and the computational
efficiency of Predictive Coding for continual learning, making it particularly
well-suited for resource-constrained edge devices or future neuromorphic
accelerators. Experimental results on the MNIST and CIFAR-10 datasets
demonstrate that this hybrid strategy enables effective adaptation with a
reduced computational overhead, offering a promising solution for maintaining
model performance in dynamic environments.

### Networking and Internet Architecture

### 1. [RIS-assisted Data Collection and Wireless Power Transfer in Low-altitude Wireless Networks](http://arxiv.org/pdf/2509.19651v1)

Authors: Wenwen Xie, Geng Sun, Jiahui Li, Jiacheng Wang, Yinqiu Liu, Dusit Niyato, Dong In Kim, Shiwen Mao

Low-altitude wireless networks (LAWNs) have become effective solutions for
collecting data from low-power Internet-of-Things devices (IoTDs) in remote
areas with limited communication infrastructure. However, some outdoor IoTDs
deployed in such areas face both energy constraints and low-channel quality
challenges, making it challenging to ensure timely data collection from these
IoTDs in LAWNs. In this work, we investigate a reconfigurable intelligent
surface (RIS)-assisted uncrewed aerial vehicle (UAV)-enabled data collection
and wireless power transfer system in LAWN. Specifically, IoTDs first harvest
energy from a low-altitude UAV, and then upload their data to the UAV by
applying the time division multiple access (TDMA) protocol, supported by an RIS
to improve the channel quality. To maintain satisfactory data freshness of the
IoTDs and save energy for an energy-constrained UAV, we aim to minimize the age
of information (AoI) and energy consumption of the UAV by jointly optimizing
the RIS phase shits, UAV trajectory, charging time allocation, and binary IoTD
scheduling. We propose a deep reinforcement learning (DRL)-based approach,
namely the alternating optimization-improved parameterized deep Q-network
(AO-IPDQN). Specifically, considering that RIS typically contains a large
number of reflecting elements, we first adopt an alternating optimization (AO)
method to optimize the RIS phase shifts to reduce the dimension of the action
space. Then, we propose the improved parameterized deep Q-network (IPDQN)
method to deal with the hybrid action space. Simulation results indicate that
AO-IPDQN approach achieves excellent performance relative to multiple
comparison methods across various simulation scenarios.

### 2. [SPARQ: An Optimization Framework for the Distribution of AI-Intensive Applications under Non-Linear Delay Constraints](http://arxiv.org/pdf/2509.19913v1)

Authors: Pietro Spadaccino, Paolo Di Lorenzo, Sergio Barbarossa, Antonia M. Tulino, Jaime Llorca

Next-generation real-time compute-intensive applications, such as extended
reality, multi-user gaming, and autonomous transportation, are increasingly
composed of heterogeneous AI-intensive functions with diverse resource
requirements and stringent latency constraints. While recent advances have
enabled very efficient algorithms for joint service placement, routing, and
resource allocation for increasingly complex applications, current models fail
to capture the non-linear relationship between delay and resource usage that
becomes especially relevant in AI-intensive workloads. In this paper, we extend
the cloud network flow optimization framework to support queuing-delay-aware
orchestration of distributed AI applications over edge-cloud infrastructures.
We introduce two execution models, Guaranteed-Resource (GR) and Shared-Resource
(SR), that more accurately capture how computation and communication delays
emerge from system-level resource constraints. These models incorporate M/M/1
and M/G/1 queue dynamics to represent dedicated and shared resource usage,
respectively. The resulting optimization problem is non-convex due to the
non-linear delay terms. To overcome this, we develop SPARQ, an iterative
approximation algorithm that decomposes the problem into two convex
sub-problems, enabling joint optimization of service placement, routing, and
resource allocation under nonlinear delay constraints. Simulation results
demonstrate that the SPARQ not only offers a more faithful representation of
system delays, but also substantially improves resource efficiency and the
overall cost-delay tradeoff compared to existing state-of-the-art methods.

### 3. [Can LLMs Forecast Internet Traffic from Social Media?](http://arxiv.org/pdf/2509.20123v1)

Authors: Jonatan Langlet, Mariano Scazzariello, Flavio Luciani, Marta Burocchi, Dejan Kostić, Marco Chiesa

Societal events shape the Internet's behavior. The death of a prominent
public figure, a software launch, or a major sports match can trigger sudden
demand surges that overwhelm peering points and content delivery networks.
Although these events fall outside regular traffic patterns, forecasting
systems still rely solely on those patterns and therefore miss these critical
anomalies.
  Thus, we argue for socio-technical systems that supplement technical
measurements with an active understanding of the underlying drivers, including
how events and collective behavior shape digital demands. We propose traffic
forecasting using signals from public discourse, such as headlines, forums, and
social media, as early demand indicators.
  To validate our intuition, we present a proof-of-concept system that
autonomously scrapes online discussions, infers real-world events, clusters and
enriches them semantically, and correlates them with traffic measurements at a
major Internet Exchange Point. This prototype predicted between 56-92% of
society-driven traffic spikes after scraping a moderate amount of online
discussions.
  We believe this approach opens new research opportunities in cross-domain
forecasting, scheduling, demand anticipation, and society-informed decision
making.

### 4. [Games Are Not Equal: Classifying Cloud Gaming Contexts for Effective User Experience Measurement](http://arxiv.org/pdf/2509.19669v1)

Authors: Yifan Wang, Minzhao Lyu, Vijay Sivaraman

To tap into the growing market of cloud gaming, whereby game graphics is
rendered in the cloud and streamed back to the user as a video feed, network
operators are creating monetizable assurance services that dynamically
provision network resources. However, without accurately measuring cloud gaming
user experience, they cannot assess the effectiveness of their provisioning
methods. Basic measures such as bandwidth and frame rate by themselves do not
suffice, and can only be interpreted in the context of the game played and the
player activity within the game. This paper equips the network operator with a
method to obtain a real-time measure of cloud gaming experience by analyzing
network traffic, including contextual factors such as the game title and player
activity stage. Our method is able to classify the game title within the first
five seconds of game launch, and continuously assess the player activity stage
as being active, passive, or idle. We deploy it in an ISP hosting NVIDIA cloud
gaming servers for the region. We provide insights from hundreds of thousands
of cloud game streaming sessions over a three-month period into the dependence
of bandwidth consumption and experience level on the gameplay contexts.

### 5. [Joint Ex-Post Location Calibration and Radio Map Construction under Biased Positioning Errors](http://arxiv.org/pdf/2509.20059v1)

Authors: Koki Kanzaki, Koya Sato

This paper proposes a high-accuracy radio map construction method tailored
for environments where location information is affected by bursty errors. Radio
maps are an effective tool for visualizing wireless environments. Although
extensive research has been conducted on accurate radio map construction, most
existing approaches assume noise-free location information during sensing. In
practice, however, positioning errors ranging from a few to several tens of
meters can arise due to device-based positioning systems (e.g., GNSS). Ignoring
such errors during inference can lead to significant degradation in radio map
accuracy. This study highlights that these errors often tend to be biased when
using mobile devices as sensors. We introduce a novel framework that models
these errors together with spatial correlation in radio propagation by
embedding them as tunable parameters in the marginal log-likelihood function.
This enables ex-post calibration of location uncertainty during radio map
construction. Numerical results based on practical human mobility data
demonstrate that the proposed method can limit RMSE degradation to
approximately 0.25-0.29 dB, compared with Gaussian process regression using
noise-free location data, whereas baseline methods suffer performance losses
exceeding 1 dB.

### 6. [CollaPipe: Adaptive Segment-Optimized Pipeline Parallelism for Collaborative LLM Training in Heterogeneous Edge Networks](http://arxiv.org/pdf/2509.19855v1)

Authors: Jiewei Chen, Xiumei Deng, Zehui Xiong, Shaoyong Guo, Xuesong Qiu, Ping Wang, Dusit Niyato

The increasing demand for intelligent mobile applications has made
multi-agent collaboration with Transformer-based large language models (LLMs)
essential in mobile edge computing (MEC) networks. However, training LLMs in
such environments remains challenging due to heavy computation, high end-to-end
latency, and limited model generalization. We introduce CollaPipe, a hybrid
distributed learning framework that integrates collaborative pipeline
parallelism with federated aggregation to support self-evolving intelligent
networks. In CollaPipe, the encoder part is adaptively partitioned into
variable-sized segments and deployed across mobile devices for
pipeline-parallel training, while the decoder is deployed on edge servers to
handle generative tasks. Then we perform global model update via federated
aggregation. To enhance training efficiency, we formulate a joint optimization
problem that adaptively allocates model segments, micro-batches, bandwidth, and
transmission power. We derive and use a closed-form convergence bound to design
an Dynamic Segment Scheduling and Resource Allocation (DSSDA) algorithm based
on Lyapunov optimization, ensuring system stability under long-term
constraints. Extensive experiments on downstream tasks with Transformer and
BERT models show that CollaPipe improves computation efficiency by up to
15.09%, reduces end-to-end latency by at least 48.98%, and cuts single device
memory usage by more than half, enabling online learning in heterogeneous and
dynamic communication environments.

### 7. [A Novel Short-Term Anomaly Prediction for IIoT with Software Defined Twin Network](http://arxiv.org/pdf/2509.20068v1)

Authors: Bilal Dalgic, Betul Sen, Muge Erel-Ozcevik

Secure monitoring and dynamic control in an IIoT environment are major
requirements for current development goals. We believe that dynamic, secure
monitoring of the IIoT environment can be achieved through integration with the
Software-Defined Network (SDN) and Digital Twin (DT) paradigms. The current
literature lacks implementation details for SDN-based DT and time-aware
intelligent model training for short-term anomaly detection against IIoT
threats. Therefore, we have proposed a novel framework for short-term anomaly
detection that uses an SDN-based DT. Using a comprehensive dataset, time-aware
labeling of features, and a comprehensive evaluation of various machine
learning models, we propose a novel SD-TWIN-based anomaly detection algorithm.
According to the performance of a new real-time SD-TWIN deployment, the GPU-
accelerated LightGBM model is particularly effective, achieving a balance of
high recall and strong classification performance.

### Robotics

### 1. [TopoCut: Learning Multi-Step Cutting with Spectral Rewards and Discrete Diffusion Policies](http://arxiv.org/pdf/2509.19712v1)

Authors: Liquan Wang, Jiangjie Bian, Eric Heiden, Animesh Garg

Robotic manipulation tasks involving cutting deformable objects remain
challenging due to complex topological behaviors, difficulties in perceiving
dense object states, and the lack of efficient evaluation methods for cutting
outcomes. In this paper, we introduce TopoCut, a comprehensive benchmark for
multi-step robotic cutting tasks that integrates a cutting environment and
generalized policy learning. TopoCut is built upon three core components: (1)
We introduce a high-fidelity simulation environment based on a particle-based
elastoplastic solver with compliant von Mises constitutive models, augmented by
a novel damage-driven topology discovery mechanism that enables accurate
tracking of multiple cutting pieces. (2) We develop a comprehensive reward
design that integrates the topology discovery with a pose-invariant spectral
reward model based on Laplace-Beltrami eigenanalysis, facilitating consistent
and robust assessment of cutting quality. (3) We propose an integrated policy
learning pipeline, where a dynamics-informed perception module predicts
topological evolution and produces particle-wise, topology-aware embeddings to
support PDDP (Particle-based Score-Entropy Discrete Diffusion Policy) for
goal-conditioned policy learning. Extensive experiments demonstrate that
TopoCut supports trajectory generation, scalable learning, precise evaluation,
and strong generalization across diverse object geometries, scales, poses, and
cutting goals.

### 2. [Towards Autonomous Robotic Electrosurgery via Thermal Imaging](http://arxiv.org/pdf/2509.19725v1)

Authors: Naveed D. Riaziat, Joseph Chen, Axel Krieger, Jeremy D. Brown

Electrosurgery is a surgical technique that can improve tissue cutting by
reducing cutting force and bleeding. However, electrosurgery adds a risk of
thermal injury to surrounding tissue. Expert surgeons estimate desirable
cutting velocities based on experience but have no quantifiable reference to
indicate if a particular velocity is optimal. Furthermore, prior demonstrations
of autonomous electrosurgery have primarily used constant tool velocity, which
is not robust to changes in electrosurgical tissue characteristics, power
settings, or tool type. Thermal imaging feedback provides information that can
be used to reduce thermal injury while balancing cutting force by controlling
tool velocity. We introduce Thermography for Electrosurgical Rate Modulation
via Optimization (ThERMO) to autonomously reduce thermal injury while balancing
cutting force by intelligently controlling tool velocity. We demonstrate ThERMO
in tissue phantoms and compare its performance to the constant velocity
approach. Overall, ThERMO improves cut success rate by a factor of three and
can reduce peak cutting force by a factor of two. ThERMO responds to varying
environmental disturbances, reduces damage to tissue, and completes cutting
tasks that would otherwise result in catastrophic failure for the constant
velocity approach.

### 3. [Simultaneous estimation of contact position and tool shape with high-dimensional parameters using force measurements and particle filtering](http://arxiv.org/pdf/2509.19732v1)

Authors: Kyo Kutsuzawa, Mitsuhiro Hayashibe

Estimating the contact state between a grasped tool and the environment is
essential for performing contact tasks such as assembly and object
manipulation. Force signals are valuable for estimating the contact state, as
they can be utilized even when the contact location is obscured by the tool.
Previous studies proposed methods for estimating contact positions using
force/torque signals; however, most methods require the geometry of the tool
surface to be known. Although several studies have proposed methods that do not
require the tool shape, these methods require considerable time for estimation
or are limited to tools with low-dimensional shape parameters. Here, we propose
a method for simultaneously estimating the contact position and tool shape,
where the tool shape is represented by a grid, which is high-dimensional (more
than 1000 dimensional). The proposed method uses a particle filter in which
each particle has individual tool shape parameters, thereby to avoid directly
handling a high-dimensional parameter space. The proposed method is evaluated
through simulations and experiments using tools with curved shapes on a plane.
Consequently, the proposed method can estimate the shape of the tool
simultaneously with the contact positions, making the contact-position
estimation more accurate.

### 4. [Trajectory Planning Using Safe Ellipsoidal Corridors as Projections of Orthogonal Trust Regions](http://arxiv.org/pdf/2509.19734v1)

Authors: Akshay Jaitly, Jon Arrizabalaga, Guanrui Li

Planning collision free trajectories in complex environments remains a core
challenge in robotics. Existing corridor based planners which rely on
decomposition of the free space into collision free subsets scale poorly with
environmental complexity and require explicit allocations of time windows to
trajectory segments. We introduce a new trajectory parameterization that
represents trajectories in a nonconvex collision free corridor as being in a
convex cartesian product of balls. This parameterization allows us to decouple
problem size from geometric complexity of the solution and naturally avoids
explicit time allocation by allowing trajectories to evolve continuously inside
ellipsoidal corridors. Building on this representation, we formulate the
Orthogonal Trust Region Problem (Orth-TRP), a specialized convex program with
separable block constraints, and develop a solver that exploits this parallel
structure and the unique structure of each parallel subproblem for efficient
optimization. Experiments on a quadrotor trajectory planning benchmark show
that our approach produces smoother trajectories and lower runtimes than
state-of-the-art corridor based planners, especially in highly complicated
environments.

### 5. [Beyond Human Demonstrations: Diffusion-Based Reinforcement Learning to Generate Data for VLA Training](http://arxiv.org/pdf/2509.19752v1)

Authors: Rushuai Yang, Hangxing Wei, Ran Zhang, Zhiyuan Feng, Xiaoyu Chen, Tong Li, Chuheng Zhang, Li Zhao, Jiang Bian, Xiu Su, Yi Chen

Vision-language-action (VLA) models have shown strong generalization across
tasks and embodiments; however, their reliance on large-scale human
demonstrations limits their scalability owing to the cost and effort of manual
data collection. Reinforcement learning (RL) offers a potential alternative to
generate demonstrations autonomously, yet conventional RL algorithms often
struggle on long-horizon manipulation tasks with sparse rewards. In this paper,
we propose a modified diffusion policy optimization algorithm to generate
high-quality and low-variance trajectories, which contributes to a diffusion
RL-powered VLA training pipeline. Our algorithm benefits from not only the high
expressiveness of diffusion models to explore complex and diverse behaviors but
also the implicit regularization of the iterative denoising process to yield
smooth and consistent demonstrations. We evaluate our approach on the LIBERO
benchmark, which includes 130 long-horizon manipulation tasks, and show that
the generated trajectories are smoother and more consistent than both human
demonstrations and those from standard Gaussian RL policies. Further, training
a VLA model exclusively on the diffusion RL-generated data achieves an average
success rate of 81.9%, which outperforms the model trained on human data by
+5.3% and that on Gaussian RL-generated data by +12.6%. The results highlight
our diffusion RL as an effective alternative for generating abundant,
high-quality, and low-variance demonstrations for VLA models.

### 6. [DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations](http://arxiv.org/pdf/2509.19804v1)

Authors: Sowoo Lee, Dongyun Kang, Jaehyun Park, Hae-Won Park

This paper introduces DynaFlow, a novel framework that embeds a
differentiable simulator directly into a flow matching model. By generating
trajectories in the action space and mapping them to dynamically feasible state
trajectories via the simulator, DynaFlow ensures all outputs are physically
consistent by construction. This end-to-end differentiable architecture enables
training on state-only demonstrations, allowing the model to simultaneously
generate physically consistent state trajectories while inferring the
underlying action sequences required to produce them. We demonstrate the
effectiveness of our approach through quantitative evaluations and showcase its
real-world applicability by deploying the generated actions onto a physical Go1
quadruped robot. The robot successfully reproduces diverse gait present in the
dataset, executes long-horizon motions in open-loop control and translates
infeasible kinematic demonstrations into dynamically executable, stylistic
behaviors. These hardware experiments validate that DynaFlow produces
deployable, highly effective motions on real-world hardware from state-only
demonstrations, effectively bridging the gap between kinematic data and
real-world execution.

### 7. [Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments](http://arxiv.org/pdf/2509.19851v1)

Authors: Benjamin Bogenberger, Oliver Harrison, Orrin Dahanaggamaarachchi, Lukas Brunke, Jingxing Qian, Siqi Zhou, Angela P. Schoellig

Robots deployed in real-world environments, such as homes, must not only
navigate safely but also understand their surroundings and adapt to environment
changes. To perform tasks efficiently, they must build and maintain a semantic
map that accurately reflects the current state of the environment. Existing
research on semantic exploration largely focuses on static scenes without
persistent object-level instance tracking. A consistent map is, however,
crucial for real-world robotic applications where objects in the environment
can be removed, reintroduced, or shifted over time. In this work, to close this
gap, we propose an open-vocabulary, semantic exploration system for semi-static
environments. Our system maintains a consistent map by building a probabilistic
model of object instance stationarity, systematically tracking semi-static
changes, and actively exploring areas that have not been visited for a
prolonged period of time. In addition to active map maintenance, our approach
leverages the map's semantic richness with LLM-based reasoning for
open-vocabulary object-goal navigation. This enables the robot to search more
efficiently by prioritizing contextually relevant areas. We evaluate our
approach across multiple real-world semi-static environments. Our system
detects 95% of map changes on average, improving efficiency by more than 29% as
compared to random and patrol baselines. Overall, our approach achieves a
mapping precision within 2% of a fully rebuilt map while requiring
substantially less exploration and further completes object goal navigation
tasks about 14% faster than the next-best tested strategy (coverage
patrolling). A video of our work can be found at
http://tiny.cc/sem-explor-semi-static .

### 8. [SAGE:State-Aware Guided End-to-End Policy for Multi-Stage Sequential Tasks via Hidden Markov Decision Process](http://arxiv.org/pdf/2509.19853v1)

Authors: BinXu Wu, TengFei Zhang, Chen Yang, JiaHao Wen, HaoCheng Li, JingTian Ma, Zhen Chen, JingYuan Wang

Multi-stage sequential (MSS) robotic manipulation tasks are prevalent and
crucial in robotics. They often involve state ambiguity, where visually similar
observations correspond to different actions. We present SAGE, a state-aware
guided imitation learning framework that models tasks as a Hidden Markov
Decision Process (HMDP) to explicitly capture latent task stages and resolve
ambiguity. We instantiate the HMDP with a state transition network that infers
hidden states, and a state-aware action policy that conditions on both
observations and hidden states to produce actions, thereby enabling
disambiguation across task stages. To reduce manual annotation effort, we
propose a semi-automatic labeling pipeline combining active learning and soft
label interpolation. In real-world experiments across multiple complex MSS
tasks with state ambiguity, SAGE achieved 100% task success under the standard
evaluation protocol, markedly surpassing the baselines. Ablation studies
further show that such performance can be maintained with manual labeling for
only about 13% of the states, indicating its strong effectiveness.

### 9. [D3Grasp: Diverse and Deformable Dexterous Grasping for General Objects](http://arxiv.org/pdf/2509.19892v1)

Authors: Keyu Wang, Bingcong Lu, Zhengxue Cheng, Hengdi Zhang, Li Song

Achieving diverse and stable dexterous grasping for general and deformable
objects remains a fundamental challenge in robotics, due to high-dimensional
action spaces and uncertainty in perception. In this paper, we present D3Grasp,
a multimodal perception-guided reinforcement learning framework designed to
enable Diverse and Deformable Dexterous Grasping. We firstly introduce a
unified multimodal representation that integrates visual and tactile perception
to robustly grasp common objects with diverse properties. Second, we propose an
asymmetric reinforcement learning architecture that exploits privileged
information during training while preserving deployment realism, enhancing both
generalization and sample efficiency. Third, we meticulously design a training
strategy to synthesize contact-rich, penetration-free, and kinematically
feasible grasps with enhanced adaptability to deformable and contact-sensitive
objects. Extensive evaluations confirm that D3Grasp delivers highly robust
performance across large-scale and diverse object categories, and substantially
advances the state of the art in dexterous grasping for deformable and
compliant objects, even under perceptual uncertainty and real-world
disturbances. D3Grasp achieves an average success rate of 95.1% in real-world
trials,outperforming prior methods on both rigid and deformable objects
benchmarks.

### 10. [GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference](http://arxiv.org/pdf/2509.19916v1)

Authors: Zijun Che, Yinghong Zhang, Shengyi Liang, Boyu Zhou, Jun Ma, Jinni Zhou

Autonomous exploration in structured and complex indoor environments remains
a challenging task, as existing methods often struggle to appropriately model
unobserved space and plan globally efficient paths. To address these
limitations, we propose GUIDE, a novel exploration framework that
synergistically combines global graph inference with diffusion-based
decision-making. We introduce a region-evaluation global graph representation
that integrates both observed environmental data and predictions of unexplored
areas, enhanced by a region-level evaluation mechanism to prioritize reliable
structural inferences while discounting uncertain predictions. Building upon
this enriched representation, a diffusion policy network generates stable,
foresighted action sequences with significantly reduced denoising steps.
Extensive simulations and real-world deployments demonstrate that GUIDE
consistently outperforms state-of-the-art methods, achieving up to 18.3% faster
coverage completion and a 34.9% reduction in redundant movements.

### Software Engineering

### 1. [Assertion Messages with Large Language Models (LLMs) for Code](http://arxiv.org/pdf/2509.19673v1)

Authors: Ahmed Aljohani, Anamul Haque Mollah, Hyunsook Do

Assertion messages significantly enhance unit tests by clearly explaining the
reasons behind test failures, yet they are frequently omitted by developers and
automated test-generation tools. Despite recent advancements, Large Language
Models (LLMs) have not been systematically evaluated for their ability to
generate informative assertion messages. In this paper, we introduce an
evaluation of four state-of-the-art Fill-in-the-Middle (FIM) LLMs -
Qwen2.5-Coder-32B, Codestral-22B, CodeLlama-13B, and StarCoder - on a dataset
of 216 Java test methods containing developer-written assertion messages. We
find that Codestral-22B achieves the highest quality score of 2.76 out of 5
using a human-like evaluation approach, compared to 3.24 for manually written
messages. Our ablation study shows that including descriptive test comments
further improves Codestral's performance to 2.97, highlighting the critical
role of context in generating clear assertion messages. Structural analysis
demonstrates that all models frequently replicate developers' preferred
linguistic patterns. We discuss the limitations of the selected models and
conventional text evaluation metrics in capturing diverse assertion message
structures. Our benchmark, evaluation results, and discussions provide an
essential foundation for advancing automated, context-aware generation of
assertion messages in test code. A replication package is available at
https://doi.org/10.5281/zenodo.15293133

### 2. [Beyond Language Barriers: Multi-Agent Coordination for Multi-Language Code Generation](http://arxiv.org/pdf/2509.19918v1)

Authors: Micheline Bénédicte Moumoula, Serge Lionel Nikiema, Albérick Euraste Djire, Abdoul Kader Kabore, Jacques Klein, Tegawendé F. Bissyande

Producing high-quality code across multiple programming languages is
increasingly important as today's software systems are built on heterogeneous
stacks. Large language models (LLMs) have advanced the state of automated
programming, yet their proficiency varies sharply between languages, especially
those with limited training data such as Rust, Perl, OCaml, and Erlang. Many
current solutions including language-specific fine-tuning, multi-agent
orchestration, transfer learning, and intermediate-representation pipelines
still approach each target language in isolation, missing opportunities to
share knowledge or exploit recurring cross-language patterns.
  XL-CoGen tackles this challenge with a coordinated multi-agent architecture
that integrates intermediate representation, code generation, translation, and
automated repair. Its distinguishing feature is a data-driven mechanism for
selecting bridging languages: empirically derived transfer matrices identify
the best intermediate languages based on demonstrated translation success
rather than raw generation accuracy. The system performs early output
validation, iteratively corrects errors, and reuses intermediate artifacts as
contextual scaffolds for subsequent translations.
  Extensive experiments show that XL-CoGen yields notable improvements with 13
percentage-point gains over the strongest fine-tuned baseline and as much as 30
percentage points over existing single-language multi-agent methods. Ablation
studies further demonstrate that compatibility-guided bridging significantly
outperforms LLM-based heuristics, confirming the value of cumulative
cross-language knowledge transfer.

### 3. [Demystifying the Evolution of Neural Networks with BOM Analysis: Insights from a Large-Scale Study of 55,997 GitHub Repositories](http://arxiv.org/pdf/2509.20010v1)

Authors: Xiaoning Ren, Yuhang Ye, Xiongfei Wu, Yueming Wu, Yinxing Xue

Neural networks have become integral to many fields due to their exceptional
performance. The open-source community has witnessed a rapid influx of neural
network (NN) repositories with fast-paced iterations, making it crucial for
practitioners to analyze their evolution to guide development and stay ahead of
trends. While extensive research has explored traditional software evolution
using Software Bill of Materials (SBOMs), these are ill-suited for NN software,
which relies on pre-defined modules and pre-trained models (PTMs) with distinct
component structures and reuse patterns. Conceptual AI Bills of Materials
(AIBOMs) also lack practical implementations for large-scale evolutionary
analysis. To fill this gap, we introduce the Neural Network Bill of Material
(NNBOM), a comprehensive dataset construct tailored for NN software. We create
a large-scale NNBOM database from 55,997 curated PyTorch GitHub repositories,
cataloging their TPLs, PTMs, and modules. Leveraging this database, we conduct
a comprehensive empirical study of neural network software evolution across
software scale, component reuse, and inter-domain dependency, providing
maintainers and developers with a holistic view of its long-term trends.
Building on these findings, we develop two prototype applications,
\textit{Multi repository Evolution Analyzer} and \textit{Single repository
Component Assessor and Recommender}, to demonstrate the practical value of our
analysis.

### 4. [V-GameGym: Visual Game Generation for Code Large Language Models](http://arxiv.org/pdf/2509.20136v1)

Authors: Wei Zhang, Jack Yang, Renshuai Tao, Lingzheng Chai, Shawn Guo, Jiajun Wu, Xiaoming Chen, Ganqu Cui, Ning Ding, Xander Xu, Hu Wei, Bowen Zhou

Code large language models have demonstrated remarkable capabilities in
programming tasks, yet current benchmarks primarily focus on single modality
rather than visual game development. Most existing code-related benchmarks
evaluate syntax correctness and execution accuracy, overlooking critical
game-specific metrics such as playability, visual aesthetics, and user
engagement that are essential for real-world deployment. To address the gap
between current LLM capabilities in algorithmic problem-solving and competitive
programming versus the comprehensive requirements of practical game
development, we present V-GameGym, a comprehensive benchmark comprising 2,219
high-quality samples across 100 thematic clusters derived from real-world
repositories, adopting a novel clustering-based curation methodology to ensure
both diversity and structural completeness. Further, we introduce a multimodal
evaluation framework with an automated LLM-driven pipeline for visual code
synthesis using complete UI sandbox environments. Our extensive analysis
reveals that V-GameGym effectively bridges the gap between code generation
accuracy and practical game development workflows, providing quantifiable
quality metrics for visual programming and interactive element generation.

### 5. [Enhancing Requirement Traceability through Data Augmentation Using Large Language Models](http://arxiv.org/pdf/2509.20149v1)

Authors: Jianzhang Zhang, Jialong Zhou, Nan Niu, Chuang Liu

Requirements traceability is crucial in software engineering to ensure
consistency between requirements and code. However, existing automated
traceability methods are constrained by the scarcity of training data and
challenges in bridging the semantic gap between artifacts. This study aims to
address the data scarcity problem in requirements traceability by employing
large language models (LLMs) for data augmentation. We propose a novel approach
that utilizes prompt-based techniques with LLMs to generate augmented
requirement-to-code trace links, thereby enhancing the training dataset. Four
LLMs (Gemini 1.5 Pro, Claude 3, GPT-3.5, and GPT-4) were used, employing both
zero-shot and few-shot templates. Moreover, we optimized the encoder component
of the tracing model to improve its efficiency and adaptability to augmented
data. The key contributions of this paper are: (1) proposing and evaluating
four prompt templates for data augmentation; (2) providing a comparative
analysis of four LLMs for generating trace links; (3) enhancing the model's
encoder for improved adaptability to augmented datasets. Experimental results
show that our approach significantly enhances model performance, achieving an
F1 score improvement of up to 28.59%, thus demonstrating its effectiveness and
potential for practical application.

### 6. [Confidentiality-Preserving Verifiable Business Processes through Zero-Knowledge Proofs](http://arxiv.org/pdf/2509.20300v1)

Authors: Jannis Kiesel, Jonathan Heiss

Ensuring the integrity of business processes without disclosing confidential
business information is a major challenge in inter-organizational processes.
This paper introduces a zero-knowledge proof (ZKP)-based approach for the
verifiable execution of business processes while preserving confidentiality. We
integrate ZK virtual machines (zkVMs) into business process management engines
through a comprehensive system architecture and a prototypical implementation.
Our approach supports chained verifiable computations through proof
compositions. On the example of product carbon footprinting, we model
sequential footprinting activities and demonstrate how organizations can prove
and verify the integrity of verifiable processes without exposing sensitive
information. We assess different ZKP proving variants within process models for
their efficiency in proving and verifying, and discuss the practical
integration of ZKPs throughout the Business Process Management (BPM) lifecycle.
Our experiment-driven evaluation demonstrates the automation of process
verification under given confidentiality constraints.

### 7. [Protocol Testing with I/O Grammars](http://arxiv.org/pdf/2509.20308v1)

Authors: Alexander Liggesmeyer, José Antonio Zamudio Amaya, Andreas Zeller

Generating software tests faces two fundamental problems. First, one needs to
_generate inputs_ that are syntactically and semantically correct, yet
sufficiently diverse to cover behavior. Second, one needs an _oracle_ to _check
outputs_ whether a test case is correct or not. Both problems become apparent
in _protocol testing_, where inputs are messages exchanged between parties, and
outputs are the responses of these parties.
  In this paper, we propose a novel approach to protocol testing that combines
input generation and output checking in a single framework. We introduce _I/O
grammars_ as the first means to _completely_ specify the syntax and semantics
of protocols, including messages, states, and interactions. Our implementation,
based on the FANDANGO framework, takes a single I/O grammar, and can act as a
_test generator_, as a _mock object_, and as an _oracle_ for a _client_, a
_server_, or both (or actually any number of parties), a versatility not found
in any existing tool or formalism. User-defined _constraints}_can have the
generator focus on arbitrary protocol features; $k$-path guidance
systematically covers states, messages, responses, and value alternatives in a
unified fashion.
  We evaluate the effectiveness of our approach by applying it to several
protocols, including DNS, FTP, and SMTP. We demonstrate that I/O grammars can
specify advanced protocol features correctly and completely, while also
enabling output validation of the programs under test. In its evaluation, we
find that systematic coverage of the I/O grammar results in much quicker
coverage of the input and response spaces (and thus functionality) compared to
the random-based state-of-the-art approaches.

### 8. [Developer Productivity With and Without GitHub Copilot: A Longitudinal Mixed-Methods Case Study](http://arxiv.org/pdf/2509.20353v1)

Authors: Viktoria Stray, Elias Goldmann Brandtzæg, Viggo Tellefsen Wivestad, Astri Barbala, Nils Brede Moe

This study investigates the real-world impact of the generative AI (GenAI)
tool GitHub Copilot on developer activity and perceived productivity. We
conducted a mixed-methods case study in NAV IT, a large public sector agile
organization. We analyzed 26,317 unique non-merge commits from 703 of NAV IT's
GitHub repositories over a two-year period, focusing on commit-based activity
metrics from 25 Copilot users and 14 non-users. The analysis was complemented
by survey responses on their roles and perceived productivity, as well as 13
interviews. Our analysis of activity metrics revealed that individuals who used
Copilot were consistently more active than non-users, even prior to Copilot's
introduction. We did not find any statistically significant changes in
commit-based activity for Copilot users after they adopted the tool, although
minor increases were observed. This suggests a discrepancy between changes in
commit-based metrics and the subjective experience of productivity.

### 9. [Benchmarking Web API Integration Code Generation](http://arxiv.org/pdf/2509.20172v1)

Authors: Daniel Maninger, Leon Chemnitz, Amir Molzam Sharifloo, Jannis Brugger, Mira Mezini

API integration is a cornerstone of our digital infrastructure, enabling
software systems to connect and interact. However, as shown by many studies,
writing or generating correct code to invoke APIs, particularly web APIs, is
challenging. Although large language models~(LLMs) have become popular in
software development, their effectiveness in automating the generation of web
API integration code remains unexplored. In order to address this, we present a
dataset and evaluation pipeline designed to assess the ability of LLMs to
generate web API invocation code. Our experiments with several open-source LLMs
reveal that generating API invocations poses a significant challenge, resulting
in hallucinated endpoints, incorrect argument usage, and other errors. None of
the evaluated open-source models were able to solve more than 40% of the tasks.

### 10. [Intuition to Evidence: Measuring AI's True Impact on Developer Productivity](http://arxiv.org/pdf/2509.19708v1)

Authors: Anand Kumar, Vishal Khare, Deepak Sharma, Satyam Kumar, Vijay Saini, Anshul Yadav, Sachendra Jain, Ankit Rana, Pratham Verma, Vaibhav Meena, Avinash Edubilli

We present a comprehensive real-world evaluation of AI-assisted software
development tools deployed at enterprise scale. Over one year, 300 engineers
across multiple teams integrated an in-house AI platform (DeputyDev) that
combines code generation and automated review capabilities into their daily
workflows. Through rigorous cohort analysis, our study demonstrates
statistically significant productivity improvements, including an overall 31.8%
reduction in PR review cycle time.
  Developer adoption was strong, with 85% satisfaction for code review features
and 93% expressing a desire to continue using the platform. Adoption patterns
showed systematic scaling from 4% engagement in month 1 to 83% peak usage by
month 6, stabilizing at 60% active engagement. Top adopters achieved a 61%
increase in code volume pushed to production, contributing to approximately 30
to 40% of code shipped to production through this tool, accounting for an
overall 28% increase in code shipment volume.
  Unlike controlled benchmark evaluations, our longitudinal analysis provides
empirical evidence from production environments, revealing both the
transformative potential and practical deployment challenges of integrating AI
into enterprise software development workflows.

### Social and Information Networks

### 1. [Governing Together: Toward Infrastructure for Community-Run Social Media](http://arxiv.org/pdf/2509.19653v1)

Authors: Sohyeon Hwang, Sophie Rollins, Thatiany Andrade Nunes, Yuhan Liu, Richmond Wong, Aaron Shaw, Andrés Monroy-Hernández

Decentralizing the governance of social computing systems to communities
promises to empower them to make independent decisions, with nuance and in
accordance with their values. Yet, communities do not govern in isolation. Many
problems communities face are common, or move across their boundaries. We
therefore propose designing for "inter-community governance:" mechanisms that
support relationships and interactions between communities to coordinate on
governance issues. Drawing from workshops with 24 individuals on decentralized,
community-run social media, we present six challenges in designing for
inter-community governance surfaced through ideas proposed in workshops.
Together, these ideas come together as an ecosystem of resources,
infrastructures, and tools that highlight three key principles for designing
for inter-community governance: modularity, forkability, and polycentricity. We
end with a discussion of how the ideas proposed in workshops might be
implemented in future work aiming to support community governance in social
computing systems broadly.

### 2. [Deterministic Frequency--Domain Inference of Network Topology and Hidden Components via Structure--Behavior Scaling](http://arxiv.org/pdf/2509.19857v1)

Authors: Xiaoxiao Liang, Tianlong Fan, Linyuan Lü

Hidden interactions and components in complex systems-ranging from covert
actors in terrorist networks to unobserved brain regions and molecular
regulators-often manifest only through indirect behavioral signals. Inferring
the underlying network structure from such partial observations remains a
fundamental challenge, particularly under nonlinear dynamics. We uncover a
robust linear relationship between the spectral strength of a node's behavioral
time series under evolutionary game dynamics and its structural degree, $S
\propto k$, a structural-behavioral scaling that holds across network types and
scales, revealing a universal correspondence between local connectivity and
dynamic energy. Leveraging this insight, we develop a deterministic,
frequency-domain inference framework based on the discrete Fourier transform
(DFT) that reconstructs network topology directly from payoff sequences-without
prior knowledge of the network or internal node strategies-by selectively
perturbing node dynamics. The framework simultaneously localizes individual
hidden nodes or identifies all edges connected to multiple hidden nodes, and
estimates tight bounds on the number of hidden nodes. Extensive experiments on
synthetic and real-world networks demonstrate that our method consistently
outperforms state-of-the-art baselines in both topology reconstruction and
hidden component detection. Moreover, it scales efficiently to large networks,
offering robustness to stochastic fluctuations and overcoming the size
limitations of existing techniques. Our work establishes a principled
connection between local dynamic observables and global structural inference,
enabling accurate topology recovery in complex systems with hidden elements.

### 3. [Large Language Models for Pedestrian Safety: An Application to Predicting Driver Yielding Behavior at Unsignalized Intersections](http://arxiv.org/pdf/2509.19657v1)

Authors: Yicheng Yang, Zixian Li, Jean Paul Bizimana, Niaz Zafri, Yongfeng Dong, Tianyi Li

Pedestrian safety is a critical component of urban mobility and is strongly
influenced by the interactions between pedestrian decision-making and driver
yielding behavior at crosswalks. Modeling driver--pedestrian interactions at
intersections requires accurately capturing the complexity of these behaviors.
Traditional machine learning models often struggle to capture the nuanced and
context-dependent reasoning required for these multifactorial interactions, due
to their reliance on fixed feature representations and limited
interpretability. In contrast, large language models (LLMs) are suited for
extracting patterns from heterogeneous traffic data, enabling accurate modeling
of driver-pedestrian interactions. Therefore, this paper leverages multimodal
LLMs through a novel prompt design that incorporates domain-specific knowledge,
structured reasoning, and few-shot prompting, enabling interpretable and
context-aware inference of driver yielding behavior, as an example application
of modeling pedestrian--driver interaction. We benchmarked state-of-the-art
LLMs against traditional classifiers, finding that GPT-4o consistently achieves
the highest accuracy and recall, while Deepseek-V3 excels in precision. These
findings highlight the critical trade-offs between model performance and
computational efficiency, offering practical guidance for deploying LLMs in
real-world pedestrian safety systems.

### 4. [Into the Void: Understanding Online Health Information in Low-Web Data Languages](http://arxiv.org/pdf/2509.20245v1)

Authors: Hellina Hailu Nigatu, Nuredin Ali Abdelkadir, Fiker Tewelde, Stevie Chancellor, Daricia Wilkinson

Data voids--areas of the internet where reliable information is scarce or
absent--pose significant challenges to online health information seeking,
particularly for users operating in low-web data languages. These voids are
increasingly encountered not on traditional search engines alone, but on social
media platforms, which have gradually morphed into informal search engines for
millions of people. In this paper, we introduce the phenomenon of data
horizons: a critical boundary where algorithmic structures begin to degrade the
relevance and reliability of search results. Unlike the core of a data void,
which is often exploited by bad actors to spread misinformation, the data
horizon marks the critical space where systemic factors, such as linguistic
underrepresentation, algorithmic amplification, and socio-cultural mismatch,
create conditions of informational instability. Focusing on Tigrinya and
Amharic as languages of study, we evaluate (1) the common characteristics of
search results for health queries, (2) the quality and credibility of health
information, and (3) characteristics of search results that diverge from their
queries. We find that search results for health queries in low-web data
languages may not always be in the language of search and may be dominated by
nutritional and religious advice. We show that search results that diverge from
their queries in low-resourced languages are due to algorithmic failures,
(un)intentional manipulation, or active manipulation by content creators. We
use our findings to illustrate how a data horizon manifests under several
interacting constraints on information availability.

### 5. [Muse-it: A Tool for Analyzing Music Discourse on Reddit](http://arxiv.org/pdf/2509.20228v1)

Authors: Jatin Agarwala, George Paul, Nemani Harsha Vardhan, Vinoo Alluri

Music engagement spans diverse interactions with music, from selection and
emotional response to its impact on behavior, identity, and social connections.
Social media platforms provide spaces where such engagement can be observed in
natural, unprompted conversations. Advances in natural language processing
(NLP) and big data analytics make it possible to analyze these discussions at
scale, extending music research to broader contexts. Reddit, in particular,
offers anonymity that encourages diverse participation and yields rich
discourse on music in ecological settings. Yet the scale of this data requires
tools to extract, process, and analyze it effectively. We present Muse-it, a
platform that retrieves comprehensive Reddit data centered on user-defined
queries. It aggregates posts from across subreddits, supports topic modeling,
temporal trend analysis, and clustering, and enables efficient study of
large-scale discourse. Muse-it also identifies music-related hyperlinks (e.g.,
Spotify), retrieves track-level metadata such as artist, album, release date,
genre, popularity, and lyrics, and links these to the discussions. An
interactive interface provides dynamic visualizations of the collected data.
Muse-it thus offers an accessible way for music researchers to gather and
analyze big data, opening new avenues for understanding music engagement as it
naturally unfolds online.

### Systems and Control

### 1. [Dispersion Formation Control: from Geometry to Distribution](http://arxiv.org/pdf/2509.19784v1)

Authors: Jin Chen, Jesus Bautista Villar, Bayu Jayawardhana, Hector Garcia de Marina

We introduce and develop the concept of dispersion formation control,
bridging a gap between shape-assembly studies in physics and biology and
formation control theory. In current formation control studies, the control
objectives typically focus on achieving desired local geometric properties,
such as inter-agent distances, bearings, or relative positions. In contrast,
our dispersion formation control approach enables agents to directly regulate
the dispersion of their spatial distribution, a global variable associated with
a covariance matrix. Specifically, we introduce the notion of covariance
similarity to define the target spatial dispersion of agents. Building on this
framework, we propose two control strategies: a centralized approach to
illustrate the key ideas, and a distributed approach that enables agents to
control the global dispersion but using only local information. Our stability
analysis demonstrates that both strategies ensure exponential convergence of
the agents' distribution to the desired dispersion. Notably, controlling a
global variable rather than multiple local ones enhances the resiliency of the
system, particularly against malfunctioning agents. Simulations validate the
effectiveness of the proposed dispersion formation control.

### 2. [Zonotope-Based Elastic Tube Model Predictive Control](http://arxiv.org/pdf/2509.19824v1)

Authors: Sabin Diaconescu, Florin Stoican, Bogdan D. Ciubotaru, Sorin Olaru

Tube-based Model Predictive Control (MPC) is a widely adopted robust control
framework for constrained linear systems under additive disturbance. The paper
is focused on reducing the numerical complexity associated with the tube
parameterization, described as a sequence of elastically-scaled zonotopic sets.
A new class of scaled-zonotope inclusion conditions is proposed, alleviating
the need for a priori specification of certain set-containment constraints and
achieving significant reductions in complexity. A comprehensive complexity
analysis is provided for both the polyhedral and the zonotopic setting,
illustrating the trade-off between an enlarged domain of attraction and the
required computational effort. The proposed approach is validated through
extensive numerical experiments.

### 3. [An early termination strategy for the distributed biased min-consensus protocol under disturbances](http://arxiv.org/pdf/2509.19832v1)

Authors: Zicheng Huang, Wangzhi Zhou, Yuanqiu Mo

The distributed biased min-consensus (DBMC) protocol is an iterative scheme
that solves the shortest path problem asymptotically, requiring only local
information exchange between neighboring nodes. By appropriately designing the
gain function, prior work [1] proposed a DBMC-based system that ensures
convergence within a pre-specified time interval. However, this guarantee
assumes the absence of disturbances. In this paper, we study the DBMC-based
system under disturbances affecting the edge weights. We first establish
rigorous error bounds on the resulting state estimates. Building on this
analysis, we then propose a practical early termination strategy to prevent
potential singularities, specifically, unbounded gain, that may arise in the
presence of disturbances, while still ensuring that the shortest paths are
correctly identified.Simulations are performed to validate and illustrate the
theoretical results.

### 4. [Control and Navigation of a 2-D Electric Rocket](http://arxiv.org/pdf/2509.19970v1)

Authors: André Fonte, Pedro Santos, Paulo Oliveira

This work addresses the control and navigation of a simulated two-dimensional
electric rocket. The model provides a simplified framework that neglects
actuator dynamics and aerodynamic effects while capturing the complexities of
underactuation and state coupling. Trajectory tracking is achieved through a
modularized and layered control architecture, with employement of a Linear
Quadratic Regulator (LQR) and Lyapunov theory. Full-state estimation is
achieved through Kalman filtering techniques, part of the navigation module.
The solutions are thoroughly evaluated in a custom-built MATLAB/Simulink
testbed, simulating real-world conditions while maintaining a simplified setup.
The results reveal limitations along the lateral axis, whose resolution is
suggested for future work.

### 5. [Distributed Koopman Operator Learning from Sequential Observations](http://arxiv.org/pdf/2509.20071v1)

Authors: Ali Azarbahram, Shenyu Liu, Gian Paolo Incremona

This paper presents a distributed Koopman operator learning framework for
modeling unknown nonlinear dynamics using sequential observations from multiple
agents. Each agent estimates a local Koopman approximation based on lifted data
and collaborates over a communication graph to reach exponential consensus on a
consistent distributed approximation. The approach supports distributed
computation under asynchronous and resource-constrained sensing. Its
performance is demonstrated through simulation results, validating convergence
and predictive accuracy under sensing-constrained scenarios and limited
communication.

### 6. [Koopman-Operator-Based Model Predictive Control for Drag-free Satellite](http://arxiv.org/pdf/2509.20100v1)

Authors: Yankai Wang, Ti Chen

This paper presents a data-driven modelling method for nonlinear dynamics of
drag-free satellite based on Koopman operator theory, and a model predictive
controller is designed based on the identified model. The nonlinear dynamics of
drag-free satellite are identified and controlled based on Sparse
Identification of Nonlinear Dynamics (SINDy). Using the manually constructed
nonlinear function dictionary as observables, the system approximation is
obtained by SINDy algorithm, and a linear Model Predictive Control (MPC)
controller is designed for test mass capture based on the SINDy model. Finally,
the effectiveness of MPC control is verified by numerical examples.

### 7. [Certified Learning-Enabled Noise-Aware Motion Planning for Urban Air Mobility](http://arxiv.org/pdf/2509.20306v1)

Authors: Jaejeong Park, Mahmoud Elfar, Cody Fleming, Yasser Shoukry

Urban Air Mobility (UAM) has emerged as a promising solution to alleviate
urban congestion and transportation challenges. Nevertheless, the noise
generated by eVTOL aircrafts poses a significant barrier to public acceptance
and regulatory approval, potentially limiting the operational scope and
scalability of UAM systems. Hence, the successful adoption of UAM systems
hinges on the ability to predict generated noise levels, and further develop
motion planning strategies that comply with community-level noise regulations
while maintaining operational efficiency. To this end, this paper proposes a
novel noise-aware motion planning framework for UAM systems that ensures
compliance with noise regulations. We first develop a certifiable neural
network model to accurately predict eVTOL noise propagation patterns in urban
environments, providing provable bounds on its correctness. To achieve a
desired level of accuracy, we propose an active sampling strategy to
efficiently build the dataset used to train and test the noise model. Next, we
develop a noise-aware motion planning algorithm that utilizes the noise model
to generate eVTOL trajectories that guarantee compliance with community noise
regulations. The algorithm exploits the monotonic structure of the noise model
to efficiently sample the configuration space, ensuring that the generated
trajectories are both noise-compliant and operationally efficient. We
demonstrate the effectiveness of the proposed framework through a number of
experiments for Vahana eVTOLs. The results show that the framework can generate
noise-compliant flight plans for a fleet of eVTOLs that adhere to community
noise regulations while optimizing operational efficiency.

### 8. [Adversarial Pursuits in Cislunar Space](http://arxiv.org/pdf/2509.20330v1)

Authors: Filippos Fotiadis, Quentin Rommel, Gregory Falco, Ufuk Topcu

Cislunar space is becoming a critical domain for future lunar and
interplanetary missions, yet its remoteness, sparse infrastructure, and
unstable dynamics create single points of failure. Adversaries in cislunar
orbits can exploit these vulnerabilities to pursue and jam co-located
communication relays, potentially severing communications between lunar
missions and the Earth. We study a pursuit-evasion scenario between two
spacecraft in a cislunar orbit, where the evader must avoid a pursuer-jammer
while remaining close to its nominal trajectory. We model the evader-pursuer
interaction as a zero-sum adversarial differential game cast in the circular
restricted three-body problem. This formulation incorporates critical aspects
of cislunar orbital dynamics, including autonomous adjustment of the reference
orbit phasing to enable aggressive evading maneuvers, and shaping of the
evader's cost with the orbit's stable and unstable manifolds. We solve the
resulting nonlinear game locally using a continuous-time differential dynamic
programming variant, which iteratively applies linear-quadratic approximations
to the Hamilton-Jacobi-Isaacs equation. We simulate the evader's behavior
against both a worst-case and a linear-quadratic pursuer. Our results pave the
way for securing future missions in cislunar space against emerging cyber
threats.

### 9. [Approximately Optimal Toll Design for Efficiency and Equity in Arc-Based Traffic Assignment Models](http://arxiv.org/pdf/2509.20355v1)

Authors: Chih-Yuan Chiu

Congestion pricing policies have emerged as promising traffic management
tools to alleviate traffic congestion caused by travelers' selfish routing
behaviors. The core principle behind deploying tolls is to impose monetary
costs on frequently overcrowded routes, to incentivize self-interested
travelers to select less easily congested routes. Recent literature has focused
on toll design based on arc-based traffic assignment models (TAMs), which
characterize commuters as traveling through a traffic network by successively
selecting an outgoing arc from every intermediate node along their journey.
However, existing tolling mechanisms predicated on arc-based TAMs often target
the design of a single congestion-minimizing toll, ignoring crucial fairness
considerations, such as the financial impact of high congestion fees on
low-income travelers. To address these shortcomings, in this paper, we pose the
dual considerations of efficiency and equity in traffic routing as bilevel
optimization problems. Since such problems are in general computationally
intractable to solve precisely, we construct a linear program approximation by
introducing a polytope approximation for the set of all tolls that induce
congestion-minimizing traffic flow patterns. Finally, we provide numerical
results that validate our theoretical conclusions.

### 10. [SoK: A Systematic Review of Malware Ontologies and Taxonomies and Implications for the Quantum Era](http://arxiv.org/pdf/2509.19650v1)

Authors: Dehinde Molade, Dave Ormrod, Mamello Thinyane, Nalin Arachchilage, Jill Slay

The threat of quantum malware is real and a growing security concern that
will have catastrophic scientific and technological impacts, if not addressed
early. If weaponised or exploited especially by the wrong hands, malware will
undermine highly sophisticated critical systems supported by next-generation
quantum architectures, for example, in defence, communications, energy, and
space. This paper explores the fundamental nature and implications of quantum
malware to enable the future development of appropriate mitigations and
defences, thereby protecting critical infrastructure. By conducting a
systematic literature review (SLR) that draws on knowledge frameworks such as
ontologies and taxonomies to explore malware, this provides insights into how
malicious behaviours can be translated into attacks on quantum technologies,
thereby providing a lens to analyse the severity of malware against quantum
technologies. This study employs the European Competency Framework for Quantum
Technologies (CFQT) as a guide to map malware behaviour to several competency
layers, creating a foundation in this emerging field.

### Machine Learning (Statistics Category)

### 1. [Convex Regression with a Penalty](http://arxiv.org/pdf/2509.19788v1)

Authors: Eunji Lim

A common way to estimate an unknown convex regression function $f_0: \Omega
\subset \mathbb{R}^d \rightarrow \mathbb{R}$ from a set of $n$ noisy
observations is to fit a convex function that minimizes the sum of squared
errors. However, this estimator is known for its tendency to overfit near the
boundary of $\Omega$, posing significant challenges in real-world applications.
In this paper, we introduce a new estimator of $f_0$ that avoids this
overfitting by minimizing a penalty on the subgradient while enforcing an upper
bound $s_n$ on the sum of squared errors. The key advantage of this method is
that $s_n$ can be directly estimated from the data. We establish the uniform
almost sure consistency of the proposed estimator and its subgradient over
$\Omega$ as $n \rightarrow \infty$ and derive convergence rates. The
effectiveness of our estimator is illustrated through its application to
estimating waiting times in a single-server queue.

### 2. [Learnable Sampler Distillation for Discrete Diffusion Models](http://arxiv.org/pdf/2509.19962v1)

Authors: Feiyang Fu, Tongxian Guo, Zhaoqiang Liu

Discrete diffusion models (DDMs) have shown powerful generation ability for
discrete data modalities like text and molecules. However, their practical
application is hindered by inefficient sampling, requiring a large number of
sampling steps. Accelerating DDMs by using larger step sizes typically
introduces significant problems in generation quality, as it amplifies the
impact of both the compounding decoding error due to factorized predictions and
discretization error from numerical approximations, leading to a significant
decrease in sampling quality. To address these challenges, we propose learnable
sampler distillation (LSD), a novel approach to train fast and high-fidelity
samplers for DDMs. LSD employs a distillation approach where a student sampler
with a few steps learns to align its intermediate score trajectory with that of
a high-quality teacher sampler with numerous steps. This alignment is achieved
by optimizing learnable sampler coefficients that adaptively adjust sampling
dynamics. Additionally, we further propose LSD+, which also learns time
schedules that allocate steps non-uniformly. Experiments across text
generation, image generation, and synthetic tasks demonstrate that our proposed
approaches outperform existing samplers for DDMs, achieving substantially
higher sampling quality with significantly fewer sampling steps. Our code is
available at
\href{https://github.com/feiyangfu/LSD}{https://github.com/feiyangfu/LSD}.

### 3. [Diffusion and Flow-based Copulas: Forgetting and Remembering Dependencies](http://arxiv.org/pdf/2509.19707v1)

Authors: David Huk, Theodoros Damoulas

Copulas are a fundamental tool for modelling multivariate dependencies in
data, forming the method of choice in diverse fields and applications. However,
the adoption of existing models for multimodal and high-dimensional
dependencies is hindered by restrictive assumptions and poor scaling. In this
work, we present methods for modelling copulas based on the principles of
diffusions and flows. We design two processes that progressively forget
inter-variable dependencies while leaving dimension-wise distributions
unaffected, provably defining valid copulas at all times. We show how to obtain
copula models by learning to remember the forgotten dependencies from each
process, theoretically recovering the true copula at optimality. The first
instantiation of our framework focuses on direct density estimation, while the
second specialises in expedient sampling. Empirically, we demonstrate the
superior performance of our proposed methods over state-of-the-art copula
approaches in modelling complex and high-dimensional dependencies from
scientific datasets and images. Our work enhances the representational power of
copula models, empowering applications and paving the way for their adoption on
larger scales and more challenging domains.

### 4. [Hierarchical Bayesian Operator-induced Symbolic Regression Trees for Structural Learning of Scientific Expressions](http://arxiv.org/pdf/2509.19710v1)

Authors: Somjit Roy, Pritam Dey, Debdeep Pati, Bani K. Mallick

The advent of Scientific Machine Learning has heralded a transformative era
in scientific discovery, driving progress across diverse domains. Central to
this progress is uncovering scientific laws from experimental data through
symbolic regression. However, existing approaches are dominated by heuristic
algorithms or data-hungry black-box methods, which often demand low-noise
settings and lack principled uncertainty quantification. Motivated by
interpretable Statistical Artificial Intelligence, we develop a hierarchical
Bayesian framework for symbolic regression that represents scientific laws as
ensembles of tree-structured symbolic expressions endowed with a regularized
tree prior. This coherent probabilistic formulation enables full posterior
inference via an efficient Markov chain Monte Carlo algorithm, yielding a
balance between predictive accuracy and structural parsimony. To guide symbolic
model selection, we develop a marginal posterior-based criterion adhering to
the Occam's window principle and further quantify structural fidelity to ground
truth through a tailored expression-distance metric. On the theoretical front,
we establish near-minimax rate of Bayesian posterior concentration, providing
the first rigorous guarantee in context of symbolic regression. Empirical
evaluation demonstrates robust performance of our proposed methodology against
state-of-the-art competing modules on a simulated example, a suite of canonical
Feynman equations, and single-atom catalysis dataset.

### 5. [High-Dimensional Statistical Process Control via Manifold Fitting and Learning](http://arxiv.org/pdf/2509.19820v1)

Authors: Burak I. Tas, Enrique del Castillo

We address the Statistical Process Control (SPC) of high-dimensional, dynamic
industrial processes from two complementary perspectives: manifold fitting and
manifold learning, both of which assume data lies on an underlying nonlinear,
lower dimensional space. We propose two distinct monitoring frameworks for
online or 'phase II' Statistical Process Control (SPC). The first method
leverages state-of-the-art techniques in manifold fitting to accurately
approximate the manifold where the data resides within the ambient
high-dimensional space. It then monitors deviations from this manifold using a
novel scalar distribution-free control chart. In contrast, the second method
adopts a more traditional approach, akin to those used in linear dimensionality
reduction SPC techniques, by first embedding the data into a lower-dimensional
space before monitoring the embedded observations. We prove how both methods
provide a controllable Type I error probability, after which they are
contrasted for their corresponding fault detection ability. Extensive numerical
experiments on a synthetic process and on a replicated Tennessee Eastman
Process show that the conceptually simpler manifold-fitting approach achieves
performance competitive with, and sometimes superior to, the more classical
lower-dimensional manifold monitoring methods. In addition, we demonstrate the
practical applicability of the proposed manifold-fitting approach by
successfully detecting surface anomalies in a real image dataset of electrical
commutators.

### 6. [On the Rate of Convergence of Kolmogorov-Arnold Network Regression Estimators](http://arxiv.org/pdf/2509.19830v1)

Authors: Wei Liu, Eleni Chatzi, Zhilu Lai

Kolmogorov-Arnold Networks (KANs) offer a structured and interpretable
framework for multivariate function approximation by composing univariate
transformations through additive or multiplicative aggregation. This paper
establishes theoretical convergence guarantees for KANs when the univariate
components are represented by B-splines. We prove that both additive and hybrid
additive-multiplicative KANs attain the minimax-optimal convergence rate
$O(n^{-2r/(2r+1)})$ for functions in Sobolev spaces of smoothness $r$. We
further derive guidelines for selecting the optimal number of knots in the
B-splines. The theory is supported by simulation studies that confirm the
predicted convergence rates. These results provide a theoretical foundation for
using KANs in nonparametric regression and highlight their potential as a
structured alternative to existing methods.

### 7. [Generalized Nonnegative Structured Kruskal Tensor Regression](http://arxiv.org/pdf/2509.19900v1)

Authors: Xinjue Wang, Esa Ollila, Sergiy A. Vorobyov, Ammar Mian

This paper introduces Generalized Nonnegative Structured Kruskal Tensor
Regression (NS-KTR), a novel tensor regression framework that enhances
interpretability and performance through mode-specific hybrid regularization
and nonnegativity constraints. Our approach accommodates both linear and
logistic regression formulations for diverse response variables while
addressing the structural heterogeneity inherent in multidimensional tensor
data. We integrate fused LASSO, total variation, and ridge regularizers, each
tailored to specific tensor modes, and develop an efficient alternating
direction method of multipliers (ADMM) based algorithm for parameter
estimation. Comprehensive experiments on synthetic signals and real
hyperspectral datasets demonstrate that NS-KTR consistently outperforms
conventional tensor regression methods. The framework's ability to preserve
distinct structural characteristics across tensor dimensions while ensuring
physical interpretability makes it especially suitable for applications in
signal processing and hyperspectral image analysis.

### 8. [Geometric Autoencoder Priors for Bayesian Inversion: Learn First Observe Later](http://arxiv.org/pdf/2509.19929v1)

Authors: Arnaud Vadeboncoeur, Gregory Duthé, Mark Girolami, Eleni Chatzi

Uncertainty Quantification (UQ) is paramount for inference in engineering
applications. A common inference task is to recover full-field information of
physical systems from a small number of noisy observations, a usually highly
ill-posed problem. Critically, engineering systems often have complicated and
variable geometries prohibiting the use of standard Bayesian UQ. In this work,
we introduce Geometric Autoencoders for Bayesian Inversion (GABI), a framework
for learning geometry-aware generative models of physical responses that serve
as highly informative geometry-conditioned priors for Bayesian inversion.
Following a ''learn first, observe later'' paradigm, GABI distills information
from large datasets of systems with varying geometries, without requiring
knowledge of governing PDEs, boundary conditions, or observation processes,
into a rich latent prior. At inference time, this prior is seamlessly combined
with the likelihood of the specific observation process, yielding a
geometry-adapted posterior distribution. Our proposed framework is architecture
agnostic. A creative use of Approximate Bayesian Computation (ABC) sampling
yields an efficient implementation that utilizes modern GPU hardware. We test
our method on: steady-state heat over rectangular domains; Reynold-Averaged
Navier-Stokes (RANS) flow around airfoils; Helmholtz resonance and source
localization on 3D car bodies; RANS airflow over terrain. We find: the
predictive accuracy to be comparable to deterministic supervised learning
approaches in the restricted setting where supervised learning is applicable;
UQ to be well calibrated and robust on challenging problems with complex
geometries. The method provides a flexible geometry-aware
train-once-use-anywhere foundation model which is independent of any particular
observation process.

### 9. [How deep is your network? Deep vs. shallow learning of transfer operators](http://arxiv.org/pdf/2509.19930v1)

Authors: Mohammad Tabish, Benedict Leimkuhler, Stefan Klus

We propose a randomized neural network approach called RaNNDy for learning
transfer operators and their spectral decompositions from data. The weights of
the hidden layers of the neural network are randomly selected and only the
output layer is trained. The main advantage is that without a noticeable
reduction in accuracy, this approach significantly reduces the training time
and resources while avoiding common problems associated with deep learning such
as sensitivity to hyperparameters and slow convergence. Additionally, the
proposed framework allows us to compute a closed-form solution for the output
layer which directly represents the eigenfunctions of the operator. Moreover,
it is possible to estimate uncertainties associated with the computed spectral
properties via ensemble learning. We present results for different dynamical
operators, including Koopman and Perron-Frobenius operators, which have
important applications in analyzing the behavior of complex dynamical systems,
and the Schr\"odinger operator. The numerical examples, which highlight the
strengths but also weaknesses of the proposed framework, include several
stochastic dynamical systems, protein folding processes, and the quantum
harmonic oscillator.

### 10. [BioBO: Biology-informed Bayesian Optimization for Perturbation Design](http://arxiv.org/pdf/2509.19988v1)

Authors: Yanke Li, Tianyu Cui, Tommaso Mansi, Mangal Prakash, Rui Liao

Efficient design of genomic perturbation experiments is crucial for
accelerating drug discovery and therapeutic target identification, yet
exhaustive perturbation of the human genome remains infeasible due to the vast
search space of potential genetic interactions and experimental constraints.
Bayesian optimization (BO) has emerged as a powerful framework for selecting
informative interventions, but existing approaches often fail to exploit
domain-specific biological prior knowledge. We propose Biology-Informed
Bayesian Optimization (BioBO), a method that integrates Bayesian optimization
with multimodal gene embeddings and enrichment analysis, a widely used tool for
gene prioritization in biology, to enhance surrogate modeling and acquisition
strategies. BioBO combines biologically grounded priors with acquisition
functions in a principled framework, which biases the search toward promising
genes while maintaining the ability to explore uncertain regions. Through
experiments on established public benchmarks and datasets, we demonstrate that
BioBO improves labeling efficiency by 25-40%, and consistently outperforms
conventional BO by identifying top-performing perturbations more effectively.
Moreover, by incorporating enrichment analysis, BioBO yields pathway-level
explanations for selected perturbations, offering mechanistic interpretability
that links designs to biologically coherent regulatory circuits.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-09-25 PST.

### 1. [A Multimodal Optical Dataset for Underwater Image Enhancement, Detection, Segmentation, and Reconstruction](https://www.nature.com/articles/s41597-025-05797-w)

Authors: Xuanhe Chu et al.

### 2. [Python, the movie! The programming language’s origin story comes to the silver screen](https://www.nature.com/articles/d41586-025-02903-1)

Authors: Jeffrey M. Perkel

### 3. [A deep learning approach for improving spatiotemporal resolution of numerical weather prediction forecasts](https://www.nature.com/articles/s41598-025-17867-5)

Authors: Décio Alves et al.

### 4. [A scalable benchmark to evaluate the robustness of image stitching under simulated distortions](https://www.nature.com/articles/s41598-025-17730-7)

Authors: Yiding Liu et al.

### 5. [A hybrid multi-node QKD-ECC architecture for securing IoT networks](https://www.nature.com/articles/s41598-025-17184-x)

Authors: Rajnish Chaturvedi et al.

### 6. [Mutual information maximizing quantum generative adversarial networks](https://www.nature.com/articles/s41598-025-18476-y)

Authors: Mingyu Lee et al.

### 7. [Acute myeloid leukemia classification using ReLViT and detection with YOLO enhanced by adversarial networks on bone marrow images](https://www.nature.com/articles/s41598-025-17891-5)

Authors: Madiha Hameed et al.

### 8. [Large lithium-ion battery model for secure shared electric bike battery in smart cities](https://www.nature.com/articles/s41467-025-63678-7)

Authors: Donghui Ding et al.

### 9. [Variational quantum recommendation system with embedded latent vectors](https://www.nature.com/articles/s41598-025-15869-x)

Authors: Shlomi Debi et al.

### 10. [Fidelity assessment of synthetic images with multi-criteria combination under adverse weather conditions](https://www.nature.com/articles/s41598-025-15480-0)

Authors: Alexandra Duminil et al.

### 11. [Hierarchical reinforcement learning-based traffic signal control](https://www.nature.com/articles/s41598-025-18449-1)

Authors: Jiajing Shen

### 12. [Steel surface defect detection algorithm based on improved YOLOv10](https://www.nature.com/articles/s41598-025-16725-8)

Authors: Laomo Zhang et al.

### 13. [Advanced gesture recognition in Indian sign language using a synergistic combination of YOLOv10 with Swin Transformer model](https://www.nature.com/articles/s41598-025-18496-8)

Authors: Umang Rastogi et al.

### 14. [Determining the minimum urban fleet for a valet style autonomous mobility service using real trip data](https://www.nature.com/articles/s41598-025-17607-9)

Authors: Antonio Pagliaroli et al.

### 15. [Instance mask alignment for object detection knowledge distillation](https://www.nature.com/articles/s41598-025-03100-w)

Authors: Zhen Guo et al.

### 16. [Optimization of traction power conservation and energy efficiency in agricultural mobile robots using the TECS algorithm](https://www.nature.com/articles/s41598-025-03204-3)

Authors: Yavuz Bahadır Koca et al.

