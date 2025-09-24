# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-23 17:00:25.685387 PST.

### Artificial Intelligence

### 1. [Correlation or Causation: Analyzing the Causal Structures of LLM and LRM Reasoning Process](http://arxiv.org/pdf/2509.17380v1)

Authors: Zhizhang FU, Guangsheng Bao, Hongbo Zhang, Chenkai Hu, Yue Zhang

LLMs suffer from critical reasoning issues such as unfaithfulness, bias, and
inconsistency, since they lack robust causal underpinnings and may rely on
superficial correlations rather than genuine understanding. Successive LRMs
have emerged as a promising alternative, leveraging advanced training
techniques such as reinforcement learning (RL) and distillation to improve task
accuracy. However, the impact of these training methods on causality remains
largely unexplored. In this study, we conduct a systematic causal analysis on
LLMs and LRMs, examining structural causal models (SCMs) of four key variables:
problem instruction (Z), thinking process (T), reasoning steps (X), and answer
(Y). Our findings reveal that RLVR-trained LRMs exhibit enhanced causal
reasoning capabilities, aligning more closely with ideal causal structures,
while LLMs and distilled LRMs fail to address causality-related deficiencies.
Our further investigation indicates that RLVR reduces spurious correlations and
strengthens genuine causal patterns, thereby mitigating unfaithfulness and
bias. In addition, our inspection on the dynamics of the RLVR training process
observes a high correlation between reduced spurious features and improved
causal structures, where the causal relationships consistently improve in the
training process. This study contributes to the understanding of causality in
reasoning models, highlights the critical role of RLVR in enhancing causal
reasoning, and provides insights for designing future AI systems with stronger
causal foundations. We release our code and data at
https://github.com/Harryking1999/CoT_Causal_Analysis.

### 2. [Evaluating Multimodal Large Language Models with Daily Composite Tasks in Home Environments](http://arxiv.org/pdf/2509.17425v1)

Authors: Zhenliang Zhang, Yuxi Wang, Hongzhao Xie, Shiyun Zhao, Mingyuan Liu, Yujie Lu, Xinyi He, Zhenku Cheng, Yujia Peng

A key feature differentiating artificial general intelligence (AGI) from
traditional AI is that AGI can perform composite tasks that require a wide
range of capabilities. Although embodied agents powered by multimodal large
language models (MLLMs) offer rich perceptual and interactive capabilities, it
remains largely unexplored whether they can solve composite tasks. In the
current work, we designed a set of composite tasks inspired by common daily
activities observed in early childhood development. Within a dynamic and
simulated home environment, these tasks span three core domains: object
understanding, spatial intelligence, and social activity. We evaluated 17
leading proprietary and open-source MLLMs on these tasks. The results
consistently showed poor performance across all three domains, indicating a
substantial gap between current capabilities and general intelligence
requirements. Together, our tasks offer a preliminary framework for evaluating
the general capabilities of embodied agents, marking an early but significant
step toward the development of embodied MLLMs and their real-world deployment.

### 3. [A Multimodal Conversational Assistant for the Characterization of Agricultural Plots from Geospatial Open Data](http://arxiv.org/pdf/2509.17544v1)

Authors: Juan Cañada, Raúl Alonso, Julio Molleda, Fidel Díez

The increasing availability of open Earth Observation (EO) and agricultural
datasets holds great potential for supporting sustainable land management.
However, their high technical entry barrier limits accessibility for non-expert
users. This study presents an open-source conversational assistant that
integrates multimodal retrieval and large language models (LLMs) to enable
natural language interaction with heterogeneous agricultural and geospatial
data. The proposed architecture combines orthophotos, Sentinel-2 vegetation
indices, and user-provided documents through retrieval-augmented generation
(RAG), allowing the system to flexibly determine whether to rely on multimodal
evidence, textual knowledge, or both in formulating an answer. To assess
response quality, we adopt an LLM-as-a-judge methodology using Qwen3-32B in a
zero-shot, unsupervised setting, applying direct scoring in a multi-dimensional
quantitative evaluation framework. Preliminary results show that the system is
capable of generating clear, relevant, and context-aware responses to
agricultural queries, while remaining reproducible and scalable across
geographic regions. The primary contributions of this work include an
architecture for fusing multimodal EO and textual knowledge sources, a
demonstration of lowering the barrier to access specialized agricultural
information through natural language interaction, and an open and reproducible
design.

### 4. [LIMI: Less is More for Agency](http://arxiv.org/pdf/2509.17567v1)

Authors: Yang Xiao, Mohan Jiang, Jie Sun, Keyu Li, Jifan Lin, Yumin Zhuang, Ji Zeng, Shijie Xia, Qishuo Hua, Xuefeng Li, Xiaojie Cai, Tongyu Wang, Yue Zhang, Liming Liu, Xia Wu, Jinlong Hou, Yuan Cheng, Wenjie Li, Xiang Wang, Dequan Wang, Pengfei Liu

We define Agency as the emergent capacity of AI systems to function as
autonomous agents actively discovering problems, formulating hypotheses, and
executing solutions through self-directed engagement with environments and
tools. This fundamental capability marks the dawn of the Age of AI Agency,
driven by a critical industry shift: the urgent need for AI systems that don't
just think, but work. While current AI excels at reasoning and generating
responses, industries demand autonomous agents that can execute tasks, operate
tools, and drive real-world outcomes. As agentic intelligence becomes the
defining characteristic separating cognitive systems from productive workers,
efficiently cultivating machine autonomy becomes paramount. Current approaches
assume that more data yields better agency, following traditional scaling laws
from language modeling. We fundamentally challenge this paradigm. LIMI (Less Is
More for Intelligent Agency) demonstrates that agency follows radically
different development principles. Through strategic focus on collaborative
software development and scientific research workflows, we show that
sophisticated agentic intelligence can emerge from minimal but strategically
curated demonstrations of autonomous behavior. Using only 78 carefully designed
training samples, LIMI achieves 73.5% on comprehensive agency benchmarks,
dramatically outperforming state-of-the-art models: Kimi-K2-Instruct (24.1%),
DeepSeek-V3.1 (11.9%), Qwen3-235B-A22B-Instruct (27.5%), and GLM-4.5 (45.1%).
Most strikingly, LIMI demonstrates 53.7% improvement over models trained on
10,000 samples-achieving superior agentic intelligence with 128 times fewer
samples. Our findings establish the Agency Efficiency Principle: machine
autonomy emerges not from data abundance but from strategic curation of
high-quality agentic demonstrations.

### 5. [Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](http://arxiv.org/pdf/2509.17589v1)

Authors: Jun Ling, Yao Qi, Tao Huang, Shibo Zhou, Yanqin Huang, Jiang Yang, Ziqi Song, Ying Zhou, Yang Yang, Heng Tao Shen, Peng Wang

In this work, we address the task of table image to LaTeX code generation,
with the goal of automating the reconstruction of high-quality,
publication-ready tables from visual inputs. A central challenge of this task
lies in accurately handling complex tables -- those with large sizes, deeply
nested structures, and semantically rich or irregular cell content -- where
existing methods often fail. We begin with a comprehensive analysis,
identifying key challenges and highlighting the limitations of current
evaluation protocols. To overcome these issues, we propose a reinforced
multimodal large language model (MLLM) framework, where a pre-trained MLLM is
fine-tuned on a large-scale table-to-LaTeX dataset. To further improve
generation quality, we introduce a dual-reward reinforcement learning strategy
based on Group Relative Policy Optimization (GRPO). Unlike standard approaches
that optimize purely over text outputs, our method incorporates both a
structure-level reward on LaTeX code and a visual fidelity reward computed from
rendered outputs, enabling direct optimization of the visual output quality. We
adopt a hybrid evaluation protocol combining TEDS-Structure and CW-SSIM, and
show that our method achieves state-of-the-art performance, particularly on
structurally complex tables, demonstrating the effectiveness and robustness of
our approach.

### 6. [EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](http://arxiv.org/pdf/2509.17677v1)

Authors: Xiyuan Zhou, Xinlei Wang, Yirui He, Yang Wu, Ruixi Zou, Yuheng Cheng, Yulu Xie, Wenxuan Liu, Huan Zhao, Yan Xu, Jinjin Gu, Junhua Zhao

Large language models (LLMs) have shown strong performance on mathematical
reasoning under well-posed conditions. However, real-world engineering problems
require more than mathematical symbolic computation -- they need to deal with
uncertainty, context, and open-ended scenarios. Existing benchmarks fail to
capture these complexities. We introduce EngiBench, a hierarchical benchmark
designed to evaluate LLMs on solving engineering problems. It spans three
levels of increasing difficulty (foundational knowledge retrieval, multi-step
contextual reasoning, and open-ended modeling) and covers diverse engineering
subfields. To facilitate a deeper understanding of model performance, we
systematically rewrite each problem into three controlled variants (perturbed,
knowledge-enhanced, and math abstraction), enabling us to separately evaluate
the model's robustness, domain-specific knowledge, and mathematical reasoning
abilities. Experiment results reveal a clear performance gap across levels:
models struggle more as tasks get harder, perform worse when problems are
slightly changed, and fall far behind human experts on the high-level
engineering tasks. These findings reveal that current LLMs still lack the
high-level reasoning needed for real-world engineering, highlighting the need
for future models with deeper and more reliable problem-solving capabilities.
Our source code and data are available at
https://github.com/EngiBench/EngiBench.

### 7. [Virtual Arc Consistency for Linear Constraints inCost Function Networks](http://arxiv.org/pdf/2509.17706v1)

Authors: Pierre Montalbano, Simon de Givry, George Katsirelos

In Constraint Programming, solving discrete minimization problems with hard
and soft constraints can be done either using (i) soft global constraints, (ii)
a reformulation into a linear program, or (iii) a reformulation into local cost
functions. Approach (i) benefits from a vast catalog of constraints. Each soft
constraint propagator communicates with other soft constraints only through the
variable domains, resulting in weak lower bounds. Conversely, the approach (ii)
provides a global view with strong bounds, but the size of the reformulation
can be problematic. We focus on approach (iii) in which soft arc consistency
(SAC) algorithms produce bounds of intermediate quality. Recently, the
introduction of linear constraints as local cost functions increases their
modeling expressiveness. We adapt an existing SAC algorithm to handle linear
constraints. We show that our algorithm significantly improves the lower bounds
compared to the original algorithm on several benchmarks, reducing solving time
in some cases.

### 8. [DA-Mamba: Dialogue-aware selective state-space model for multimodal engagement estimation](http://arxiv.org/pdf/2509.17711v1)

Authors: Shenwei Kang, Xin Zhang, Wen Liu, Bin Li, Yujie Liu, Bo Gao

Human engagement estimation in conversational scenarios is essential for
applications such as adaptive tutoring, remote healthcare assessment, and
socially aware human--computer interaction. Engagement is a dynamic, multimodal
signal conveyed by facial expressions, speech, gestures, and behavioral cues
over time. In this work we introduce DA-Mamba, a dialogue-aware multimodal
architecture that replaces attention-heavy dialogue encoders with Mamba-based
selective state-space processing to achieve linear time and memory complexity
while retaining expressive cross-modal reasoning. We design a Mamba
dialogue-aware selective state-space model composed of three core modules: a
Dialogue-Aware Encoder, and two Mamba-based fusion mechanisms: Modality-Group
Fusion and Partner-Group Fusion, these modules achieve expressive dialogue
understanding. Extensive experiments on three standard benchmarks (NoXi,
NoXi-Add, and MPIIGI) show that DA-Mamba surpasses prior state-of-the-art
(SOTA) methods in concordance correlation coefficient (CCC), while reducing
training time and peak memory; these gains enable processing much longer
sequences and facilitate real-time deployment in resource-constrained,
multi-party conversational settings. The source code will be available at:
https://github.com/kksssssss-ssda/MMEA.

### 9. [Mitigating Strategy-Selection Bias in Reasoning for More Effective Test-Time Scaling](http://arxiv.org/pdf/2509.17905v1)

Authors: Zongqian Wu, Baoduo Xu, Tianyu Li, Zhu Sun, Xiaofeng Zhu, Lei Feng

Test-time scaling (TTS) has been shown to improve the performance of large
language models (LLMs) by sampling and aggregating diverse reasoning paths.
However, existing research has overlooked a critical issue: selection bias of
reasoning strategies during scaling. Specifically, when generating reasoning
processes, LLMs tend to follow certain strategies (e.g., algebraic solutions
for math problems) while neglecting other valid alternatives (e.g., geometric
solutions), resulting in insufficient exploration of the solution space. To
further understand the impact of this bias, we present a theoretical analysis
that reveals when it undermines the effectiveness of test-time scaling.
Motivated by this theoretical insight, we introduce TTS-Uniform, a framework
designed to mitigate the selection bias of reasoning strategies. It (i)
identifies potential strategies, (ii) uniformly allocates the sampling budget
across them, and (iii) filters out unstable strategies prior to aggregation.
Experimental results show that TTS-Uniform significantly enhances scaling
effectiveness across multiple mainstream LLMs and benchmark datasets.

### 10. [MEF: A Systematic Evaluation Framework for Text-to-Image Models](http://arxiv.org/pdf/2509.17907v1)

Authors: Xiaojing Dong, Weilin Huang, Liang Li, Yiying Li, Shu Liu, Tongtong Ou, Shuang Ouyang, Yu Tian, Fengxuan Zhao

Rapid advances in text-to-image (T2I) generation have raised higher
requirements for evaluation methodologies. Existing benchmarks center on
objective capabilities and dimensions, but lack an application-scenario
perspective, limiting external validity. Moreover, current evaluations
typically rely on either ELO for overall ranking or MOS for dimension-specific
scoring, yet both methods have inherent shortcomings and limited
interpretability. Therefore, we introduce the Magic Evaluation Framework (MEF),
a systematic and practical approach for evaluating T2I models. First, we
propose a structured taxonomy encompassing user scenarios, elements, element
compositions, and text expression forms to construct the Magic-Bench-377, which
supports label-level assessment and ensures a balanced coverage of both user
scenarios and capabilities. On this basis, we combine ELO and
dimension-specific MOS to generate model rankings and fine-grained assessments
respectively. This joint evaluation method further enables us to quantitatively
analyze the contribution of each dimension to user satisfaction using
multivariate logistic regression. By applying MEF to current T2I models, we
obtain a leaderboard and key characteristics of the leading models. We release
our evaluation framework and make Magic-Bench-377 fully open-source to advance
research in the evaluation of visual generative models.

### Hardware Architecture

### 1. [Overcoming challenges in bamboo connections: A review of mechanical properties and structural considerations](http://arxiv.org/pdf/2509.17721v1)

Authors: Pierre Boucher, Victor Fréchard, Diego Ramirez-Cardona, Claudiane Ouellet-Plamondon

Over the past decades, bamboo has increasingly gained attention as a
sustainable construction material, through its rapid growth, naturally
optimized shape, high mechanical properties, and significant environmental
benefits. However, despite these advantages, the use of bamboo in its natural
form for structural applications remains limited, partly due to insufficient
knowledge of connection behavior, which is crucial for ensuring the long-term
reliability and performance of bamboo structures. This article provides a
comprehensive review of the key factors to consider in the design of structural
bamboo connections and discusses the existing connection classification methods
used as guidelines by designers. By synthesizing findings from the literature,
our research aims to identify the key parameters interacting with the
connection design process, focusing on the anatomical, geometric, and
mechanical properties of bamboo, the mechanical requirements of the structure
design, and the building methods. A critical analysis of Janssen's
classification of bamboo connections, based on force transfer modes and later
refined by Widyowijatnoko, is presented. Finally, we discuss the identified
research gaps and emphasize the need for integrated design approaches supported
by guidelines to support the broader adoption of bamboo in construction.

### 2. [Minimal Neuron Circuits: Bursters](http://arxiv.org/pdf/2509.17731v1)

Authors: Amr Nabil, T. Nandha Kumar, Haider Abbas F. Almurib

This work introduces a novel methodology for designing biologically plausible
bursting neuron circuits using a minimal number of components. We hypothesize
that to design circuits capable of bursting, the neuron circuit design must
mimic a neuron model that inherently exhibits bursting dynamics. Consequently,
classical models such as the Hodgkin-Huxley, $I_{Na,p}+I_{K}$, and
FitzHugh-Nagumo models are not suitable choices. Instead, we propose a
methodology for designing neuron circuits that emulate the qualitative
characteristics of the $I_{Na,p}+I_{K}+I_{K(M)}$ model, a well-established
minimal bursting neuron model. Based on this methodology, we present two novel
MOSFET-based circuits that exhibit bursting. Using the method of dissection of
neural bursting, we demonstrate that the nullcline and bifurcation diagrams of
the fast subsystem in our circuits are qualitatively equivalent to those of the
$I_{Na,p}+I_{K}+I_{K(M)}$ model. Furthermore, we examine the effect of the type
of bifurcation at burst initiation and termination on the bursting
characteristics, showing that our circuits can exhibit diverse bursting
behaviours. Importantly, the main contribution of this work lies not in the
specific circuit implementation, but in the methodology proposed for
constructing bursting neuron circuits.

### 3. [Single-Cell Universal Logic-in-Memory Using 2T-nC FeRAM: An Area and Energy-Efficient Approach for Bulk Bitwise Computation](http://arxiv.org/pdf/2509.17963v1)

Authors: Rudra Biswas, Jiahui Duan, Shan Deng, Xuezhong Niu, Yixin Qin, Prapti Panigrahi, Varun Parekh, Rajiv Joshi, Kai Ni, Vijaykrishnan Narayanan

This work presents a novel approach to configure 2T-nC ferroelectric RAM
(FeRAM) for performing single cell logic-in-memory operations, highlighting its
advantages in energy-efficient computation over conventional DRAM-based
approaches. Unlike conventional 1T-1C dynamic RAM (DRAM), which incurs refresh
overhead, 2T-nC FeRAM offers a promising alternative as a non-volatile memory
solution with low energy consumption. Our key findings include the potential of
quasi-nondestructive readout (QNRO) sensing in 2T-nC FeRAM for logic-in-memory
(LiM) applications, demonstrating its inherent capability to perform inverting
logic without requiring external modifications, a feature absent in traditional
1T-1C DRAM. We successfully implement the MINORITY function within a single
cell of 2T-nC FeRAM, enabling universal NAND and NOR logic, validated through
SPICE simulations and experimental data. Additionally, the research
investigates the feasibility of 3D integration with 2T-nC FeRAM, showing
substantial improvements in storage and computational density, facilitating
bulk-bitwise computation. Our evaluation of eight real-world, data-intensive
applications reveals that 2T-nC FeRAM achieves 2x higher performance and 2.5x
lower energy consumption compared to DRAM. Furthermore, the thermal stability
of stacked 2T-nC FeRAM is validated, confirming its reliable operation when
integrated on a compute die. These findings emphasize the advantages of 2T-nC
FeRAM for LiM, offering superior performance and energy efficiency over
conventional DRAM.

### Computational Complexity

### 1. [Supersimulators](http://arxiv.org/pdf/2509.17994v1)

Authors: Cynthia Dwork, Pranay Tankala

We prove that every randomized Boolean function admits a supersimulator: a
randomized polynomial-size circuit whose output on random inputs cannot be
efficiently distinguished from reality with constant advantage, even by
polynomially larger distinguishers. Our result builds on the landmark
complexity-theoretic regularity lemma of Trevisan, Tulsiani and Vadhan (2009),
which, in contrast, provides a simulator that fools smaller distinguishers. We
circumvent lower bounds for the simulator size by letting the distinguisher
size bound vary with the target function, while remaining below an absolute
upper bound independent of the target function. This dependence on the target
function arises naturally from our use of an iteration technique originating in
the graph regularity literature.
  The simulators provided by the regularity lemma and recent refinements
thereof, known as multiaccurate and multicalibrated predictors, respectively,
as per Hebert-Johnson et al. (2018), have previously been shown to have myriad
applications in complexity theory, cryptography, learning theory, and beyond.
We first show that a recent multicalibration-based characterization of the
computational indistinguishability of product distributions actually requires
only (calibrated) multiaccuracy. We then show that supersimulators yield an
even tighter result in this application domain, closing a complexity gap
present in prior versions of the characterization.

### 2. [Sketching approximations and LP approximations for finite CSPs are related](http://arxiv.org/pdf/2509.17926v1)

Authors: Noah G. Singer, Madhur Tulsiani, Santhoshini Velusamy

We identify a connection between the approximability of CSPs in two models:
(i) sublinear space streaming algorithms, and (ii) the basic LP relaxation. We
show that whenever the basic LP admits an integrality gap, there is an
$\Omega(\sqrt{n})$-space sketching lower bound. We also show that all existing
linear space streaming lower bounds for Max-CSPs can be lifted to integrality
gap instances for basic LPs. For bounded-degree graphs, by combining the
distributed algorithm of Yoshida (STOC 2011) for approximately solving the
basic LP with techniques described in Saxena, Singer, Sudan, and Velusamy (SODA
2025) for simulating a distributed algorithm by a sublinear space streaming
algorithm on bounded-degree instances of Max-DICUT, it appears that there are
sublinear space streaming algorithms implementing the basic LP, for every CSP.
  Based on our results, we conjecture the following dichotomy theorem: Whenever
the basic LP admits an integrality gap, there is a linear space single-pass
streaming lower bound, and when the LP is roundable, there is a sublinear space
streaming algorithm.

### 3. [Reinforced Generation of Combinatorial Structures: Applications to Complexity Theory](http://arxiv.org/pdf/2509.18057v1)

Authors: Ansh Nagda, Prabhakar Raghavan, Abhradeep Thakurta

We explore whether techniques from AI can help discover new combinatorial
structures that improve provable limits on efficient algorithms. Specifically,
we use AlphaEvolve (an LLM coding agent) to study two settings:
  a) Average-case hardness for MAX-CUT and MAX-Independent Set: We improve a
recent result of Kunisky and Yu to obtain near-optimal upper and (conditional)
lower bounds on certification algorithms for MAX-CUT and MAX-Independent Set on
random 3- and 4-regular graphs. Our improved lower bounds are obtained by
constructing nearly extremal Ramanujan graphs on as many as $163$ nodes, using
AlphaEvolve. Additionally, via analytical arguments we strengthen the upper
bounds to settle the computational hardness of these questions up to an error
in the third decimal place.
  b) Worst-case Hardness of Approximation for MAX-k-CUT: We obtain new
inapproximability results, proving that it is NP-hard to approximate MAX-4-CUT
and MAX-3-CUT within factors of $0.987$ and $0.9649$ respectively, using
AlphaEvolve to discover new gadget reductions. Our MAX-4-CUT result improves
upon the SOTA of $0.9883$, and our MAX-3-CUT result improves on the current
best gadget-based inapproximability result of $0.9853$, but falls short of
improving the SOTA of $16/17$ that relies on a custom PCP, rather than a gadget
reduction from "standard" H{\aa}stad-style PCPs.
  A key technical challenge we faced: verifying a candidate construction
produced by AlphaEvolve is costly (often requiring exponential time). In both
settings above, our results were enabled by using AlphaEvolve itself to evolve
the verification procedure to be faster (sometimes by $10,000\times$). We
conclude with a discussion of norms by which to assess the assistance from AI
in developing proofs.

### Computational Engineering

### 1. [$i$MIND: Insightful Multi-subject Invariant Neural Decoding](http://arxiv.org/pdf/2509.17313v1)

Authors: Zixiang Yin, Jiarui Li, Zhengming Ding

Decoding visual signals holds the tantalizing potential to unravel the
complexities of cognition and perception. While recent studies have focused on
reconstructing visual stimuli from neural recordings to bridge brain activity
with visual imagery, existing methods offer limited insights into the
underlying mechanisms of visual processing in the brain. To mitigate this gap,
we present an \textit{i}nsightful \textbf{M}ulti-subject \textbf{I}nvariant
\textbf{N}eural \textbf{D}ecoding ($i$MIND) model, which employs a novel
dual-decoding framework--both biometric and semantic decoding--to offer neural
interpretability in a data-driven manner and deepen our understanding of
brain-based visual functionalities. Our $i$MIND model operates through three
core steps: establishing a shared neural representation space across subjects
using a ViT-based masked autoencoder, disentangling neural features into
complementary subject-specific and object-specific components, and performing
dual decoding to support both biometric and semantic classification tasks.
Experimental results demonstrate that $i$MIND achieves state-of-the-art
decoding performance with minimal scalability limitations. Furthermore, $i$MIND
empirically generates voxel-object activation fingerprints that reveal
object-specific neural patterns and enable investigation of subject-specific
variations in attention to identical stimuli. These findings provide a
foundation for more interpretable and generalizable subject-invariant neural
decoding, advancing our understanding of the voxel semantic selectivity as well
as the neural vision processing dynamics.

### 2. [Rational Multi-Modal Transformers for TCR-pMHC Prediction](http://arxiv.org/pdf/2509.17305v1)

Authors: Jiarui Li, Zixiang Yin, Zhengming Ding, Samuel J. Landry, Ramgopal R. Mettu

T cell receptor (TCR) recognition of peptide-MHC (pMHC) complexes is
fundamental to adaptive immunity and central to the development of T cell-based
immunotherapies. While transformer-based models have shown promise in
predicting TCR-pMHC interactions, most lack a systematic and explainable
approach to architecture design. We present an approach that uses a new
post-hoc explainability method to inform the construction of a novel
encoder-decoder transformer model. By identifying the most informative
combinations of TCR and epitope sequence inputs, we optimize cross-attention
strategies, incorporate auxiliary training objectives, and introduce a novel
early-stopping criterion based on explanation quality. Our framework achieves
state-of-the-art predictive performance while simultaneously improving
explainability, robustness, and generalization. This work establishes a
principled, explanation-driven strategy for modeling TCR-pMHC binding and
offers mechanistic insights into sequence-level binding behavior through the
lens of deep learning.

### 3. [An AutoML Framework using AutoGluonTS for Forecasting Seasonal Extreme Temperatures](http://arxiv.org/pdf/2509.17734v1)

Authors: Pablo Rodríguez-Bocca, Guillermo Pereira, Diego Kiedanski, Soledad Collazo, Sebastián Basterrech, Gerardo Rubino

In recent years, great progress has been made in the field of forecasting
meteorological variables. Recently, deep learning architectures have made a
major breakthrough in forecasting the daily average temperature over a ten-day
horizon. However, advances in forecasting events related to the maximum
temperature over short horizons remain a challenge for the community. A problem
that is even more complex consists in making predictions of the maximum daily
temperatures in the short, medium, and long term. In this work, we focus on
forecasting events related to the maximum daily temperature over medium-term
periods (90 days). Therefore, instead of addressing the problem from a
meteorological point of view, this article tackles it from a climatological
point of view. Due to the complexity of this problem, a common approach is to
frame the study as a temporal classification problem with the classes: maximum
temperature "above normal", "normal" or "below normal". From a practical point
of view, we created a large historical dataset (from 1981 to 2018) collecting
information from weather stations located in South America. In addition, we
also integrated exogenous information from the Pacific, Atlantic, and Indian
Ocean basins. We applied the AutoGluonTS platform to solve the above-mentioned
problem. This AutoML tool shows competitive forecasting performance with
respect to large operational platforms dedicated to tackling this
climatological problem; but with a "relatively" low computational cost in terms
of time and resources.

### 4. [Explainability matters: The effect of liability rules on the healthcare sector](http://arxiv.org/pdf/2509.17334v1)

Authors: Jiawen Wei, Elena Verona, Andrea Bertolini, Gianmarco Mengaldo

Explainability, the capability of an artificial intelligence system (AIS) to
explain its outcomes in a manner that is comprehensible to human beings at an
acceptable level, has been deemed essential for critical sectors, such as
healthcare. Is it really the case? In this perspective, we consider two extreme
cases, ``Oracle'' (without explainability) versus ``AI Colleague'' (with
explainability) for a thorough analysis. We discuss how the level of automation
and explainability of AIS can affect the determination of liability among the
medical practitioner/facility and manufacturer of AIS. We argue that
explainability plays a crucial role in setting a responsibility framework in
healthcare, from a legal standpoint, to shape the behavior of all involved
parties and mitigate the risk of potential defensive medicine practices.

### Computational Geometry

### 1. [A note on non-crossing path partitions in the plane](http://arxiv.org/pdf/2509.17485v1)

Authors: Javier Tejel

In the paper ``Lower bounds on the number of crossing-free subgraphs of
$K_N$'' (Computational Geometry 16 (2000), 211-221), it is shown that a double
chain of $n$ points in the plane admits at least $\Omega(4.642126305^n)$
polygonizations, and it is claimed that it admits at most $O(5.61^n)$
polygonizations. In this note, we provide a proof of this last result. The
proof is based on counting non-crossing path partitions for points in the plane
in convex position, where a non-crossing path partition consists of a set of
paths connecting the points such that no two edges cross and isolated points
are allowed.
  We prove that a set of $n$ points in the plane in convex position admits
$\mathcal{O}^*(5.610718614^{n})$ non-crossing path partitions and a double
chain of $n$ points in the plane admits at least $\Omega(7.164102920^n)$
non-crossing path partitions. If isolated points are not allowed, we also show
that there are $\mathcal{O}^*(4.610718614^n)$ non-crossing path partitions for
$n$ points in the plane in convex position and at least $\Omega(6.164492582^n)$
non-crossing path partitions in a double chain of $n$ points in the plane. In
addition, using a particular family of non-crossing path partitions for points
in convex position, we provide an alternative proof for the result that a
double chain of $n$ points admits at least $\Omega(4.642126305^n)$
polygonizations.

### 2. [Word2VecGD: Neural Graph Drawing with Cosine-Stress Optimization](http://arxiv.org/pdf/2509.17333v1)

Authors: Minglai Yang, Reyan Ahmed

We propose a novel graph visualization method leveraging random walk-based
embeddings to replace costly graph-theoretical distance computations. Using
word2vec-inspired embeddings, our approach captures both structural and
semantic relationships efficiently. Instead of relying on exact shortest-path
distances, we optimize layouts using cosine dissimilarities, significantly
reducing computational overhead. Our framework integrates differentiable stress
optimization with stochastic gradient descent (SGD), supporting multi-criteria
layout objectives. Experimental results demonstrate that our method produces
high-quality, semantically meaningful layouts while efficiently scaling to
large graphs. Code available at: https://github.com/mlyann/graphv_nn

### Computation and Language

### 1. [Automated Knowledge Graph Construction using Large Language Models and Sentence Complexity Modelling](http://arxiv.org/pdf/2509.17289v1)

Authors: Sydney Anuyah, Mehedi Mahmud Kaushik, Krishna Dwarampudi, Rakesh Shiradkar, Arjan Durresi, Sunandan Chakraborty

We introduce CoDe-KG, an open-source, end-to-end pipeline for extracting
sentence-level knowledge graphs by combining robust coreference resolution with
syntactic sentence decomposition. Using our model, we contribute a dataset of
over 150,000 knowledge triples, which is open source. We also contribute a
training corpus of 7248 rows for sentence complexity, 190 rows of gold human
annotations for co-reference resolution using open source lung-cancer abstracts
from PubMed, 900 rows of gold human annotations for sentence conversion
policies, and 398 triples of gold human annotations. We systematically select
optimal prompt-model pairs across five complexity categories, showing that
hybrid chain-of-thought and few-shot prompting yields up to 99.8% exact-match
accuracy on sentence simplification. On relation extraction (RE), our pipeline
achieves 65.8% macro-F1 on REBEL, an 8-point gain over the prior state of the
art, and 75.7% micro-F1 on WebNLG2, while matching or exceeding performance on
Wiki-NRE and CaRB. Ablation studies demonstrate that integrating coreference
and decomposition increases recall on rare relations by over 20%. Code and
dataset are available at https://github.com/KaushikMahmud/CoDe-KG_EMNLP_2025

### 2. [Scale-free Characteristics of Multilingual Legal Texts and the Limitations of LLMs](http://arxiv.org/pdf/2509.17367v1)

Authors: Haoyang Chen, Kumiko Tanaka-Ishii

We present a comparative analysis of text complexity across domains using
scale-free metrics. We quantify linguistic complexity via Heaps' exponent
$\beta$ (vocabulary growth), Taylor's exponent $\alpha$ (word-frequency
fluctuation scaling), compression rate $r$ (redundancy), and entropy. Our
corpora span three domains: legal documents (statutes, cases, deeds) as a
specialized domain, general natural language texts (literature, Wikipedia), and
AI-generated (GPT) text. We find that legal texts exhibit slower vocabulary
growth (lower $\beta$) and higher term consistency (higher $\alpha$) than
general texts. Within legal domain, statutory codes have the lowest $\beta$ and
highest $\alpha$, reflecting strict drafting conventions, while cases and deeds
show higher $\beta$ and lower $\alpha$. In contrast, GPT-generated text shows
the statistics more aligning with general language patterns. These results
demonstrate that legal texts exhibit domain-specific structures and
complexities, which current generative models do not fully replicate.

### 3. [Robustness of Neurosymbolic Reasoners on First-Order Logic Problems](http://arxiv.org/pdf/2509.17377v1)

Authors: Hannah Bansal, Kemal Kurniawan, Lea Frermann

Recent trends in NLP aim to improve reasoning capabilities in Large Language
Models (LLMs), with key focus on generalization and robustness to variations in
tasks. Counterfactual task variants introduce minimal but semantically
meaningful changes to otherwise valid first-order logic (FOL) problem instances
altering a single predicate or swapping roles of constants to probe whether a
reasoning system can maintain logical consistency under perturbation. Previous
studies showed that LLMs becomes brittle on counterfactual variations,
suggesting that they often rely on spurious surface patterns to generate
responses. In this work, we explore if a neurosymbolic (NS) approach that
integrates an LLM and a symbolic logical solver could mitigate this problem.
Experiments across LLMs of varying sizes show that NS methods are more robust
but perform worse overall that purely neural methods. We then propose NSCoT
that combines an NS method and Chain-of-Thought (CoT) prompting and demonstrate
that while it improves performance, NSCoT still lags behind standard CoT. Our
analysis opens research directions for future work.

### 4. [FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis](http://arxiv.org/pdf/2509.17395v1)

Authors: Tianshi Cai, Guanxu Li, Nijia Han, Ce Huang, Zimu Wang, Changyu Zeng, Yuqi Wang, Jingshi Zhou, Haiyang Zhang, Qi Chen, Yushan Pan, Shuihua Wang, Wei Wang

We introduce FinDebate, a multi-agent framework for financial analysis,
integrating collaborative debate with domain-specific Retrieval-Augmented
Generation (RAG). Five specialized agents, covering earnings, market,
sentiment, valuation, and risk, run in parallel to synthesize evidence into
multi-dimensional insights. To mitigate overconfidence and improve reliability,
we introduce a safe debate protocol that enables agents to challenge and refine
initial conclusions while preserving coherent recommendations. Experimental
results, based on both LLM-based and human evaluations, demonstrate the
framework's efficacy in producing high-quality analysis with calibrated
confidence levels and actionable investment strategies across multiple time
horizons.

### 5. [EpiCache: Episodic KV Cache Management for Long Conversational Question Answering](http://arxiv.org/pdf/2509.17396v1)

Authors: Minsoo Kim, Arnav Kundu, Han-Byul Kim, Richa Dixit, Minsik Cho

Recent advances in large language models (LLMs) have extended context
lengths, enabling assistants to sustain long histories for coherent,
personalized responses. This ability, however, hinges on Key-Value (KV)
caching, whose memory grows linearly with dialogue length and quickly dominates
under strict resource constraints. An active line of research for reducing this
overhead is KV cache compression, which seeks to limit cache size while
preserving accuracy. Yet existing methods face two major limitations: (i)
evicting entries after full-context prefill causes unbounded peak memory, and
(ii) query-dependent eviction narrows the cache to a single query, leading to
degraded accuracy in multi-turn conversations. We introduce EpiCache, a
training-free KV cache management framework for long conversational question
answering (LongConvQA) under fixed memory budgets. EpiCache bounds cache growth
through block-wise prefill and preserves topic-relevant context via episodic KV
compression, which clusters conversation history into coherent episodes and
applies episode-specific KV cache eviction. We further design an adaptive
layer-wise budget allocation strategy that measures each layer's sensitivity to
eviction and distributes the memory budget across layers accordingly. Across
three LongConvQA benchmarks, EpiCache improves accuracy by up to 40% over
recent baselines, sustains near-full KV accuracy under 4-6x compression, and
reduces latency and memory by up to 2.4x and 3.5x, thereby enabling efficient
multi-turn interaction under strict resource constraints.

### 6. [DIWALI - Diversity and Inclusivity aWare cuLture specific Items for India: Dataset and Assessment of LLMs for Cultural Text Adaptation in Indian Context](http://arxiv.org/pdf/2509.17399v1)

Authors: Pramit Sahoo, Maharaj Brahma, Maunendra Sankar Desarkar

Large language models (LLMs) are widely used in various tasks and
applications. However, despite their wide capabilities, they are shown to lack
cultural alignment \citep{ryan-etal-2024-unintended,
alkhamissi-etal-2024-investigating} and produce biased generations
\cite{naous-etal-2024-beer} due to a lack of cultural knowledge and competence.
Evaluation of LLMs for cultural awareness and alignment is particularly
challenging due to the lack of proper evaluation metrics and unavailability of
culturally grounded datasets representing the vast complexity of cultures at
the regional and sub-regional levels. Existing datasets for culture specific
items (CSIs) focus primarily on concepts at the regional level and may contain
false positives. To address this issue, we introduce a novel CSI dataset for
Indian culture, belonging to 17 cultural facets. The dataset comprises $\sim$8k
cultural concepts from 36 sub-regions. To measure the cultural competence of
LLMs on a cultural text adaptation task, we evaluate the adaptations using the
CSIs created, LLM as Judge, and human evaluations from diverse
socio-demographic region. Furthermore, we perform quantitative analysis
demonstrating selective sub-regional coverage and surface-level adaptations
across all considered LLMs. Our dataset is available here:
\href{https://huggingface.co/datasets/nlip/DIWALI}{https://huggingface.co/datasets/nlip/DIWALI},
project
webpage\footnote{\href{https://nlip-lab.github.io/nlip/publications/diwali/}{https://nlip-lab.github.io/nlip/publications/diwali/}},
and our codebase with model outputs can be found here:
\href{https://github.com/pramitsahoo/culture-evaluation}{https://github.com/pramitsahoo/culture-evaluation}.

### 7. [QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning on Large Language Models](http://arxiv.org/pdf/2509.17428v1)

Authors: Hyesung Jeon, Seojune Lee, Beomseok Kang, Yulhwa Kim, Jae-Joon Kim

The demand for efficient deployment of large language models (LLMs) has
driven interest in quantization, which reduces inference cost, and
parameter-efficient fine-tuning (PEFT), which lowers training overhead. This
motivated the development of quantization-aware PEFT to produce accurate yet
efficient quantized models. In this setting, reducing quantization error prior
to fine-tuning is crucial for achieving high model accuracy. However, existing
methods that rely on low-rank adaptation suffer from limited representational
capacity. Recent Fourier-related transform (FT)-based adapters offer greater
representational power than low-rank adapters, but their direct integration
into quantized models often results in ineffective error reduction and
increased computational overhead. To overcome these limitations, we propose
QWHA, a method that integrates FT-based adapters into quantized models by
employing the Walsh-Hadamard Transform (WHT) as the transform kernel, together
with a novel adapter initialization scheme incorporating adaptive parameter
selection and value refinement. We demonstrate that QWHA effectively mitigates
quantization errors while facilitating fine-tuning, and that its design
substantially reduces computational cost. Experimental results show that QWHA
consistently outperforms baselines in low-bit quantization accuracy and
achieves significant training speedups over existing FT-based adapters. The
code is available at https://github.com/vantaa89/qwha.

### 8. [MedFact: A Large-scale Chinese Dataset for Evidence-based Medical Fact-checking of LLM Responses](http://arxiv.org/pdf/2509.17436v1)

Authors: Tong Chen, Zimu Wang, Yiyi Miao, Haoran Luo, Yuanfei Sun, Wei Wang, Zhengyong Jiang, Procheta Sen, Jionglong Su

Medical fact-checking has become increasingly critical as more individuals
seek medical information online. However, existing datasets predominantly focus
on human-generated content, leaving the verification of content generated by
large language models (LLMs) relatively unexplored. To address this gap, we
introduce MedFact, the first evidence-based Chinese medical fact-checking
dataset of LLM-generated medical content. It consists of 1,321 questions and
7,409 claims, mirroring the complexities of real-world medical scenarios. We
conduct comprehensive experiments in both in-context learning (ICL) and
fine-tuning settings, showcasing the capability and challenges of current LLMs
on this task, accompanied by an in-depth error analysis to point out key
directions for future research. Our dataset is publicly available at
https://github.com/AshleyChenNLP/MedFact.

### 9. [GeoPQA: Bridging the Visual Perception Gap in MLLMs for Geometric Reasoning](http://arxiv.org/pdf/2509.17437v1)

Authors: Guizhen Chen, Weiwen Xu, Hao Zhang, Hou Pong Chan, Deli Zhao, Anh Tuan Luu, Yu Rong

Recent advancements in reinforcement learning (RL) have enhanced the
reasoning abilities of large language models (LLMs), yet the impact on
multimodal LLMs (MLLMs) is limited. Particularly in vision-intensive tasks like
geometric reasoning, MLLMs hallucinate frequently, leading to inaccurate
reasoning. We attribute this to the perceptual bottleneck in MLLMs, which caps
the benefits of reasoning training. To quantify this, we design a
Geo-Perception Question-Answering (GeoPQA) benchmark, targeting basic geometric
concepts and spatial relationships. Experiments on GeoPQA reveal significant
shortcomings of MLLMs in visual perception, which constrain RL reward signals
for effective training. To address this bottleneck, we propose a two-stage RL
training framework by first enhancing the visual perception of geometric
structures, then fostering reasoning capabilities. Applied to
Qwen2.5-VL-3B-Instruct, our two-stage training improves geometric reasoning by
9.7% and geometric problem solving by 9.1%, compared to the direct reasoning
training approach. Our method also generalizes to other vision-intensive
domains like figure understanding, highlighting the importance of perceptual
grounding in effective MLLM reasoning.

### 10. [Filling in the Clinical Gaps in Benchmark: Case for HealthBench for the Japanese medical system](http://arxiv.org/pdf/2509.17444v1)

Authors: Shohei Hisada, Endo Sunao, Himi Yamato, Shoko Wakamiya, Eiji Aramaki

This study investigates the applicability of HealthBench, a large-scale,
rubric-based medical benchmark, to the Japanese context. While robust
evaluation frameworks are crucial for the safe development of medical LLMs,
resources in Japanese remain limited, often relying on translated
multiple-choice questions. Our research addresses this gap by first
establishing a performance baseline, applying a machine-translated version of
HealthBench's 5,000 scenarios to evaluate both a high-performing multilingual
model (GPT-4.1) and a Japanese-native open-source model (LLM-jp-3.1). Second,
we employ an LLM-as-a-Judge approach to systematically classify the benchmark's
scenarios and rubric criteria, identifying "contextual gaps" where content is
misaligned with Japan's clinical guidelines, healthcare systems, or cultural
norms. Our findings reveal a modest performance drop in GPT-4.1 due to rubric
mismatches and a significant failure in the Japanese-native model, which lacked
the required clinical completeness. Furthermore, our classification indicates
that while the majority of scenarios are applicable, a substantial portion of
the rubric criteria requires localization. This work underscores the
limitations of direct benchmark translation and highlights the urgent need for
a context-aware, localized adaptation, a J-HealthBench, to ensure the reliable
and safe evaluation of medical LLMs in Japan.

### Cryptography and Security

### 1. [TextCrafter: Optimization-Calibrated Noise for Defending Against Text Embedding Inversion](http://arxiv.org/pdf/2509.17302v1)

Authors: Duoxun Tang, Xinhang Jiang, Jiajun Niu

Text embedding inversion attacks reconstruct original sentences from latent
representations, posing severe privacy threats in collaborative inference and
edge computing. We propose TextCrafter, an optimization-based adversarial
perturbation mechanism that combines RL learned, geometry aware noise injection
orthogonal to user embeddings with cluster priors and PII signal guidance to
suppress inversion while preserving task utility. Unlike prior defenses either
non learnable or agnostic to perturbation direction, TextCrafter provides a
directional protective policy that balances privacy and utility. Under strong
privacy setting, TextCrafter maintains 70 percentage classification accuracy on
four datasets and consistently outperforms Gaussian/LDP baselines across lower
privacy budgets, demonstrating a superior privacy utility trade off.

### 2. [A Lightweight Authentication and Key Agreement Protocol Design for FANET](http://arxiv.org/pdf/2509.17409v1)

Authors: Yao Wu, Ziye Jia, Qihui Wu, Yian Zhu

The advancement of low-altitude intelligent networks enables unmanned aerial
vehicle (UAV) interconnection via flying ad-hoc networks (FANETs), offering
flexibility and decentralized coordination. However, resource constraints,
dynamic topologies, and UAV operations in open environments present significant
security and communication challenges. Existing multi-factor and public-key
cryptography protocols are vulnerable due to their reliance on stored sensitive
information, increasing the risk of exposure and compromise. This paper
proposes a lightweight authentication and key agreement protocol for FANETs,
integrating physical unclonable functions with dynamic credential management
and lightweight cryptographic primitives. The protocol reduces computational
and communication overhead while enhancing security. Security analysis confirms
its resilience against various attacks, and comparative evaluations demonstrate
its superiority in security, communication efficiency, and computational cost.

### 3. [DINVMark: A Deep Invertible Network for Video Watermarking](http://arxiv.org/pdf/2509.17416v1)

Authors: Jianbin Ji, Dawen Xu, Li Dong, Lin Yang, Songhan He

With the wide spread of video, video watermarking has become increasingly
crucial for copyright protection and content authentication. However, video
watermarking still faces numerous challenges. For example, existing methods
typically have shortcomings in terms of watermarking capacity and robustness,
and there is a lack of specialized noise layer for High Efficiency Video
Coding(HEVC) compression. To address these issues, this paper introduces a Deep
Invertible Network for Video watermarking (DINVMark) and designs a noise layer
to simulate HEVC compression. This approach not only in creases watermarking
capacity but also enhances robustness. DINVMark employs an Invertible Neural
Network (INN), where the encoder and decoder share the same network structure
for both watermark embedding and extraction. This shared architecture ensures
close coupling between the encoder and decoder, thereby improving the accuracy
of the watermark extraction process. Experimental results demonstrate that the
proposed scheme significantly enhances watermark robustness, preserves video
quality, and substantially increases watermark embedding capacity.

### 4. [Ordered Multi-Signatures with Public-Key Aggregation from SXDH Assumption](http://arxiv.org/pdf/2509.17709v1)

Authors: Masayuki Tezuka, Keisuke Tanaka

An ordered multi-signature scheme allows multiple signers to sign a common
message in a sequential manner and allows anyone to verify the signing order of
signers with a public-key list. In this work, we propose an ordered
multi-signature scheme by modifying the sequential aggregate signature scheme
by Chatterjee and Kabaleeshwaran (ACISP 2020). Our scheme offers compact public
parameter size and the public-key aggregation property. This property allows us
to compress a public-key list into a short aggregated key. We prove the
security of our scheme under the symmetric external Diffie-Hellman (SXDH)
assumption without the random oracle model.

### 5. [Public Key Encryption with Equality Test from Tag-Based Encryption](http://arxiv.org/pdf/2509.17722v1)

Authors: Masayuki Tezuka, Keisuke Tanaka

Public key encryption with equality test (PKEET), proposed by Yang et al.
(CT-RSA 2010), is a variant of public key encryption that enables an equality
test to determine whether two ciphertexts correspond to the same plaintext.
This test applies not only for ciphertexts generated under the same encryption
key but also for those generated under different encryption keys. To date,
several generic constructions of PKEET have been proposed. However, these
generic constructions have the drawback of reliance on the random oracle model
or a (hierarchical) identity-based encryption scheme. In this paper, we propose
a generic construction of a PKEET scheme based on tag-based encryption without
the random oracle model. Tag-based encryption is a weaker primitive than
identity-based encryption. Our scheme allows to derive new PKEET schemes
without the random oracle model. By instantiating our construction with the
pairing-free tag-based encryption scheme by Kiltz (TCC 2006), we obtain a
pairing-free PKEET scheme without the random oracle model. Moreover, by
instantiating our construction with a tag-based encryption scheme based on the
learning parity with noise (LPN) assumption, we obtain a PKEET scheme based on
the LPN assumption without the random oracle model.

### 6. [AEAS: Actionable Exploit Assessment System](http://arxiv.org/pdf/2509.17832v1)

Authors: Xiangmin Shen, Wenyuan Cheng, Yan Chen, Zhenyuan Li, Yuqiao Gu, Lingzhi Wang, Wencheng Zhao, Dawei Sun, Jiashui Wang

Security practitioners face growing challenges in exploit assessment, as
public vulnerability repositories are increasingly populated with inconsistent
and low-quality exploit artifacts. Existing scoring systems, such as CVSS and
EPSS, offer limited support for this task. They either rely on theoretical
metrics or produce opaque probability estimates without assessing whether
usable exploit code exists. In practice, security teams often resort to manual
triage of exploit repositories, which is time-consuming, error-prone, and
difficult to scale. We present AEAS, an automated system designed to assess and
prioritize actionable exploits through static analysis. AEAS analyzes both
exploit code and associated documentation to extract a structured set of
features reflecting exploit availability, functionality, and setup complexity.
It then computes an actionability score for each exploit and produces ranked
exploit recommendations. We evaluate AEAS on a dataset of over 5,000
vulnerabilities derived from 600+ real-world applications frequently
encountered by red teams. Manual validation and expert review on representative
subsets show that AEAS achieves a 100% top-3 success rate in recommending
functional exploits and shows strong alignment with expert-validated rankings.
These results demonstrate the effectiveness of AEAS in supporting
exploit-driven vulnerability prioritization.

### 7. [Federated Learning in the Wild: A Comparative Study for Cybersecurity under Non-IID and Unbalanced Settings](http://arxiv.org/pdf/2509.17836v1)

Authors: Roberto Doriguzzi-Corin, Petr Sabel, Silvio Cretti, Silvio Ranise

Machine Learning (ML) techniques have shown strong potential for network
traffic analysis; however, their effectiveness depends on access to
representative, up-to-date datasets, which is limited in cybersecurity due to
privacy and data-sharing restrictions. To address this challenge, Federated
Learning (FL) has recently emerged as a novel paradigm that enables
collaborative training of ML models across multiple clients while ensuring that
sensitive data remains local. Nevertheless, Federated Averaging (FedAvg), the
canonical FL algorithm, has proven poor convergence in heterogeneous
environments where data distributions are non-independent and identically
distributed (i.i.d.) and client datasets are unbalanced, conditions frequently
observed in cybersecurity contexts. To overcome these challenges, several
alternative FL strategies have been developed, yet their applicability to
network intrusion detection remains insufficiently explored. This study
systematically reviews and evaluates a range of FL methods in the context of
intrusion detection for DDoS attacks. Using a dataset of network attacks within
a Kubernetes-based testbed, we assess convergence efficiency, computational
overhead, bandwidth consumption, and model accuracy. To the best of our
knowledge, this is the first comparative analysis of FL algorithms for
intrusion detection under realistic non-i.i.d. and unbalanced settings,
providing new insights for the design of robust, privacypreserving network
security solutions.

### 8. [B-Privacy: Defining and Enforcing Privacy in Weighted Voting](http://arxiv.org/pdf/2509.17871v1)

Authors: Samuel Breckenridge, Dani Vilardell, Andrés Fábrega, Amy Zhao, Patrick McCorry, Rafael Solari, Ari Juels

In traditional, one-vote-per-person voting systems, privacy equates with
ballot secrecy: voting tallies are published, but individual voters' choices
are concealed.
  Voting systems that weight votes in proportion to token holdings, though, are
now prevalent in cryptocurrency and web3 systems. We show that these
weighted-voting systems overturn existing notions of voter privacy. Our
experiments demonstrate that even with secret ballots, publishing raw tallies
often reveals voters' choices.
  Weighted voting thus requires a new framework for privacy. We introduce a
notion called B-privacy whose basis is bribery, a key problem in voting systems
today. B-privacy captures the economic cost to an adversary of bribing voters
based on revealed voting tallies.
  We propose a mechanism to boost B-privacy by noising voting tallies. We prove
bounds on its tradeoff between B-privacy and transparency, meaning
reported-tally accuracy. Analyzing 3,582 proposals across 30 Decentralized
Autonomous Organizations (DAOs), we find that the prevalence of large voters
("whales") limits the effectiveness of any B-Privacy-enhancing technique.
However, our mechanism proves to be effective in cases without extreme voting
weight concentration: among proposals requiring coalitions of $\geq5$ voters to
flip outcomes, our mechanism raises B-privacy by a geometric mean factor of
$4.1\times$.
  Our work offers the first principled guidance on transparency-privacy
tradeoffs in weighted-voting systems, complementing existing approaches that
focus on ballot secrecy and revealing fundamental constraints that voting
weight concentration imposes on privacy mechanisms.

### 9. [What if we could hot swap our Biometrics?](http://arxiv.org/pdf/2509.17962v1)

Authors: Jon Crowcroft, Anil Madhavapeddy, Chris Hicks, Richard Mortier, Vasilios Mavroudis

What if you could really revoke your actual biometric identity, and install a
new one, by live rewriting your biological self? We propose some novel
mechanisms for hot swapping identity based in novel biotechnology. We discuss
the potential positive use cases, and negative consequences if such technology
was to become available and affordable. Biometrics are selected on the basis
that they are supposed to be unfakeable, or at least not at reasonable cost. If
they become easier to fake, it may be much cheaper to fake someone else's
biometrics than it is for you to change your own biometrics if someone does
copy yours. This potentially makes biometrics a bad trade-off for the user. At
the time of writing, this threat is highly speculative, but we believe it is
worth raising and considering the potential consequences.

### 10. [The Reverse File System: Towards open cost-effective secure WORM storage devices for logging](http://arxiv.org/pdf/2509.17969v1)

Authors: Gorka Guardiola Múzquiz, Juan González-Gómez, Enrique Soriano-Salvador

Write Once Read Many (WORM) properties for storage devices are desirable to
ensure data immutability for applications such as secure logging, regulatory
compliance, archival storage, and other types of backup systems. WORM devices
guarantee that data, once written, cannot be altered or deleted. However,
implementing secure and compatible WORM storage remains a challenge.
Traditional solutions often rely on specialized hardware, which is either
costly, closed, or inaccessible to the general public. Distributed approaches,
while promising, introduce additional risks such as denial-of-service
vulnerabilities and operational complexity. We introduce Socarrat, a novel,
cost-effective, and local WORM storage solution that leverages a simple
external USB device (specifically, a single-board computer running Linux with
USB On-The-Go support). The resulting device can be connected via USB,
appearing as an ordinary external disk formatted with an ext4 or exFAT file
system, without requiring any specialized software or drivers. By isolating the
WORM enforcement mechanism in a dedicated USB hardware module, Socarrat
significantly reduces the attack surface and ensures that even privileged
attackers cannot modify or erase stored data. In addition to the WORM capacity,
the system is designed to be tamper-evident, becoming resilient against
advanced attacks. This work describes a novel approach, the Reverse File
System, based on inferring the file system operations occurring at higher
layers in the host computer where Socarrat is mounted. The paper also describes
the current Socarrat prototype, implemented in Go and available as free/libre
software. Finally, it provides a complete evaluation of the logging performance
on different single-board computers.

### Computer Vision and Pattern Recognition

### 1. [SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction](http://arxiv.org/pdf/2509.17329v1)

Authors: Neham Jain, Andrew Jong, Sebastian Scherer, Ioannis Gkioulekas

Smoke in real-world scenes can severely degrade the quality of images and
hamper visibility. Recent methods for image restoration either rely on
data-driven priors that are susceptible to hallucinations, or are limited to
static low-density smoke. We introduce SmokeSeer, a method for simultaneous 3D
scene reconstruction and smoke removal from a video capturing multiple views of
a scene. Our method uses thermal and RGB images, leveraging the fact that the
reduced scattering in thermal images enables us to see through the smoke. We
build upon 3D Gaussian splatting to fuse information from the two image
modalities, and decompose the scene explicitly into smoke and non-smoke
components. Unlike prior approaches, SmokeSeer handles a broad range of smoke
densities and can adapt to temporally varying smoke. We validate our approach
on synthetic data and introduce a real-world multi-view smoke dataset with RGB
and thermal images. We provide open-source code and data at the project
website.

### 2. [Revisiting Vision Language Foundations for No-Reference Image Quality Assessment](http://arxiv.org/pdf/2509.17374v1)

Authors: Ankit Yadav, Ta Duc Huy, Lingqiao Liu

Large-scale vision language pre-training has recently shown promise for
no-reference image-quality assessment (NR-IQA), yet the relative merits of
modern Vision Transformer foundations remain poorly understood. In this work,
we present the first systematic evaluation of six prominent pretrained
backbones, CLIP, SigLIP2, DINOv2, DINOv3, Perception, and ResNet, for the task
of No-Reference Image Quality Assessment (NR-IQA), each finetuned using an
identical lightweight MLP head. Our study uncovers two previously overlooked
factors: (1) SigLIP2 consistently achieves strong performance; and (2) the
choice of activation function plays a surprisingly crucial role, particularly
for enhancing the generalization ability of image quality assessment models.
Notably, we find that simple sigmoid activations outperform commonly used ReLU
and GELU on several benchmarks. Motivated by this finding, we introduce a
learnable activation selection mechanism that adaptively determines the
nonlinearity for each channel, eliminating the need for manual activation
design, and achieving new state-of-the-art SRCC on CLIVE, KADID10K, and
AGIQA3K. Extensive ablations confirm the benefits across architectures and
regimes, establishing strong, resource-efficient NR-IQA baselines.

### 3. [Single-Image Depth from Defocus with Coded Aperture and Diffusion Posterior Sampling](http://arxiv.org/pdf/2509.17427v1)

Authors: Hodaka Kawachi, Jose Reinaldo Cunha Santos A. V. Silva Neto, Yasushi Yagi, Hajime Nagahara, Tomoya Nakamura

We propose a single-snapshot depth-from-defocus (DFD) reconstruction method
for coded-aperture imaging that replaces hand-crafted priors with a learned
diffusion prior used purely as regularization. Our optimization framework
enforces measurement consistency via a differentiable forward model while
guiding solutions with the diffusion prior in the denoised image domain,
yielding higher accuracy and stability than clas- sical optimization. Unlike
U-Net-style regressors, our approach requires no paired defocus-RGBD training
data and does not tie training to a specific camera configuration. Experiments
on comprehensive simulations and a prototype camera demonstrate consistently
strong RGBD reconstructions across noise levels, outperforming both U-Net
baselines and a classical coded- aperture DFD method.

### 4. [Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration](http://arxiv.org/pdf/2509.17429v1)

Authors: Zhitao Zeng, Guojian Yuan, Junyuan Mao, Yuxuan Wang, Xiaoshuang Jia, Yueming Jin

Accurate temporal prediction is the bridge between comprehensive scene
understanding and embodied artificial intelligence. However, predicting
multiple fine-grained states of a scene at multiple temporal scales is
difficult for vision-language models. We formalize the Multi-Scale Temporal
Prediction (MSTP) task in general and surgical scenes by decomposing
multi-scale into two orthogonal dimensions: the temporal scale, forecasting
states of humans and surgery at varying look-ahead intervals, and the state
scale, modeling a hierarchy of states in general and surgical scenes. For
example, in general scenes, states of contact relationships are finer-grained
than states of spatial relationships. In surgical scenes, medium-level steps
are finer-grained than high-level phases yet remain constrained by their
encompassing phase. To support this unified task, we introduce the first MSTP
Benchmark, featuring synchronized annotations across multiple state scales and
temporal scales. We further propose a method, Incremental Generation and
Multi-agent Collaboration (IG-MC), which integrates two key innovations. First,
we present a plug-and-play incremental generation module that continuously
synthesizes up-to-date visual previews at expanding temporal scales to inform
multiple decision-making agents, keeping decisions and generated visuals
synchronized and preventing performance degradation as look-ahead intervals
lengthen. Second, we present a decision-driven multi-agent collaboration
framework for multi-state prediction, comprising generation, initiation, and
multi-state assessment agents that dynamically trigger and evaluate prediction
cycles to balance global coherence and local fidelity.

### 5. [Emergent 3D Correspondence from Neural Shape Representation](http://arxiv.org/pdf/2509.17431v1)

Authors: Keyu Du, Jingyu Hu, Haipeng Li, Hao Xu, Haibing Huang, Chi-Wing Fu, Shuaicheng Liu

This paper presents a new approach to estimate accurate and robust 3D
semantic correspondence with the hierarchical neural semantic representation.
Our work has three key contributions. First, we design the hierarchical neural
semantic representation (HNSR), which consists of a global semantic feature to
capture high-level structure and multi-resolution local geometric features to
preserve fine details, by carefully harnessing 3D priors from pre-trained 3D
generative models. Second, we design a progressive global-to-local matching
strategy, which establishes coarse semantic correspondence using the global
semantic feature, then iteratively refines it with local geometric features,
yielding accurate and semantically-consistent mappings. Third, our framework is
training-free and broadly compatible with various pre-trained 3D generative
backbones, demonstrating strong generalization across diverse shape categories.
Our method also supports various applications, such as shape co-segmentation,
keypoint matching, and texture transfer, and generalizes well to structurally
diverse shapes, with promising results even in cross-category scenarios. Both
qualitative and quantitative evaluations show that our method outperforms
previous state-of-the-art techniques.

### 6. [CARINOX: Inference-time Scaling with Category-Aware Reward-based Initial Noise Optimization and Exploration](http://arxiv.org/pdf/2509.17458v1)

Authors: Seyed Amir Kasaei, Ali Aghayari, Arash Marioriyad, Niki Sepasian, Shayan Baghayi Nejad, MohammadAmin Fazli, Mahdieh Soleymani Baghshah, Mohammad Hossein Rohban

Text-to-image diffusion models, such as Stable Diffusion, can produce
high-quality and diverse images but often fail to achieve compositional
alignment, particularly when prompts describe complex object relationships,
attributes, or spatial arrangements. Recent inference-time approaches address
this by optimizing or exploring the initial noise under the guidance of reward
functions that score text-image alignment without requiring model fine-tuning.
While promising, each strategy has intrinsic limitations when used alone:
optimization can stall due to poor initialization or unfavorable search
trajectories, whereas exploration may require a prohibitively large number of
samples to locate a satisfactory output. Our analysis further shows that
neither single reward metrics nor ad-hoc combinations reliably capture all
aspects of compositionality, leading to weak or inconsistent guidance. To
overcome these challenges, we present Category-Aware Reward-based Initial Noise
Optimization and Exploration (CARINOX), a unified framework that combines noise
optimization and exploration with a principled reward selection procedure
grounded in correlation with human judgments. Evaluations on two complementary
benchmarks covering diverse compositional challenges show that CARINOX raises
average alignment scores by +16% on T2I-CompBench++ and +11% on the HRS
benchmark, consistently outperforming state-of-the-art optimization and
exploration-based methods across all major categories, while preserving image
quality and diversity. The project page is available at
https://amirkasaei.com/carinox/{this URL}.

### 7. [CSDformer: A Conversion Method for Fully Spike-Driven Transformer](http://arxiv.org/pdf/2509.17461v1)

Authors: Yuhao Zhang, Chengjun Zhang, Di Wu, Jie Yang, Mohamad Sawan

Spike-based transformer is a novel architecture aiming to enhance the
performance of spiking neural networks while mitigating the energy overhead
inherent to transformers. However, methods for generating these models suffer
from critical limitations: excessive training costs introduced by direct
training methods, or unavoidably hardware-unfriendly operations in existing
conversion methods. In this paper, we propose CSDformer, a novel conversion
method for fully spike-driven transformers. We tailor a conversion-oriented
transformer-based architecture and propose a new function NReLU to replace
softmax in self-attention. Subsequently, this model is quantized and trained,
and converted into a fully spike-driven model with temporal decomposition
technique. Also, we propose delayed Integrate-andFire neurons to reduce
conversion errors and improve the performance of spiking models. We evaluate
CSDformer on ImageNet, CIFAR-10 and CIFAR-100 datasets and achieve 76.36% top-1
accuracy under 7 time-steps on ImageNet, demonstrating superiority over
state-of-the-art models. Furthermore, CSDformer eliminates the need for
training SNNs, thereby reducing training costs (reducing computational resource
by 75% and accelerating training speed by 2-3$\times$). To the best of our
knowledge, this is the first fully spike-driven transformer-based model
developed via conversion method, achieving high performance under ultra-low
latency, while dramatically reducing both computational complexity and training
overhead.

### 8. [MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](http://arxiv.org/pdf/2509.17462v1)

Authors: Changwon Kang, Jisong Kim, Hongjae Shin, Junseo Park, Jun Won Choi

The goal of multi-task learning is to learn to conduct multiple tasks
simultaneously based on a shared data representation. While this approach can
improve learning efficiency, it may also cause performance degradation due to
task conflicts that arise when optimizing the model for different objectives.
To address this challenge, we introduce MAESTRO, a structured framework
designed to generate task-specific features and mitigate feature interference
in multi-task 3D perception, including 3D object detection, bird's-eye view
(BEV) map segmentation, and 3D occupancy prediction. MAESTRO comprises three
components: the Class-wise Prototype Generator (CPG), the Task-Specific Feature
Generator (TSFG), and the Scene Prototype Aggregator (SPA). CPG groups class
categories into foreground and background groups and generates group-wise
prototypes. The foreground and background prototypes are assigned to the 3D
object detection task and the map segmentation task, respectively, while both
are assigned to the 3D occupancy prediction task. TSFG leverages these
prototype groups to retain task-relevant features while suppressing irrelevant
features, thereby enhancing the performance for each task. SPA enhances the
prototype groups assigned for 3D occupancy prediction by utilizing the
information produced by the 3D object detection head and the map segmentation
head. Extensive experiments on the nuScenes and Occ3D benchmarks demonstrate
that MAESTRO consistently outperforms existing methods across 3D object
detection, BEV map segmentation, and 3D occupancy prediction tasks.

### 9. [Stable Video-Driven Portraits](http://arxiv.org/pdf/2509.17476v1)

Authors: Mallikarjun B. R., Fei Yin, Vikram Voleti, Nikita Drobyshev, Maksim Lapin, Aaryaman Vasishta, Varun Jampani

Portrait animation aims to generate photo-realistic videos from a single
source image by reenacting the expression and pose from a driving video. While
early methods relied on 3D morphable models or feature warping techniques, they
often suffered from limited expressivity, temporal inconsistency, and poor
generalization to unseen identities or large pose variations. Recent advances
using diffusion models have demonstrated improved quality but remain
constrained by weak control signals and architectural limitations. In this
work, we propose a novel diffusion based framework that leverages masked facial
regions specifically the eyes, nose, and mouth from the driving video as strong
motion control cues. To enable robust training without appearance leakage, we
adopt cross identity supervision. To leverage the strong prior from the
pretrained diffusion model, our novel architecture introduces minimal new
parameters that converge faster and help in better generalization. We introduce
spatial temporal attention mechanisms that allow inter frame and intra frame
interactions, effectively capturing subtle motions and reducing temporal
artifacts. Our model uses history frames to ensure continuity across segments.
At inference, we propose a novel signal fusion strategy that balances motion
fidelity with identity preservation. Our approach achieves superior temporal
consistency and accurate expression control, enabling high-quality,
controllable portrait animation suitable for real-world applications.

### 10. [SAMSON: 3rd Place Solution of LSVOS 2025 VOS Challenge](http://arxiv.org/pdf/2509.17500v1)

Authors: Yujie Xie, Hongyang Zhang, Zhihui Liu, Shihai Ruan

Large-scale Video Object Segmentation (LSVOS) addresses the challenge of
accurately tracking and segmenting objects in long video sequences, where
difficulties stem from object reappearance, small-scale targets, heavy
occlusions, and crowded scenes. Existing approaches predominantly adopt
SAM2-based frameworks with various memory mechanisms for complex video mask
generation. In this report, we proposed Segment Anything with Memory
Strengthened Object Navigation (SAMSON), the 3rd place solution in the MOSE
track of ICCV 2025, which integrates the strengths of stateof-the-art VOS
models into an effective paradigm. To handle visually similar instances and
long-term object disappearance in MOSE, we incorporate a long-term memorymodule
for reliable object re-identification. Additionly, we adopt SAM2Long as a
post-processing strategy to reduce error accumulation and enhance segmentation
stability in long video sequences. Our method achieved a final performance of
0.8427 in terms of J &F in the test-set leaderboard.

### Computers and Society

### 1. [AI, Digital Platforms, and the New Systemic Risk](http://arxiv.org/pdf/2509.17878v1)

Authors: Philipp Hacker, Atoosa Kasirzadeh, Lilian Edwards

As artificial intelligence (AI) becomes increasingly embedded in digital,
social, and institutional infrastructures, and AI and platforms are merged into
hybrid structures, systemic risk has emerged as a critical but undertheorized
challenge. In this paper, we develop a rigorous framework for understanding
systemic risk in AI, platform, and hybrid system governance, drawing on
insights from finance, complex systems theory, climate change, and
cybersecurity - domains where systemic risk has already shaped regulatory
responses. We argue that recent legislation, including the EU's AI Act and
Digital Services Act (DSA), invokes systemic risk but relies on narrow or
ambiguous characterizations of this notion, sometimes reducing this risk to
specific capabilities present in frontier AI models, or to harms occurring in
economic market settings. The DSA, we show, actually does a better job at
identifying systemic risk than the more recent AI Act. Our framework highlights
novel risk pathways, including the possibility of systemic failures arising
from the interaction of multiple AI agents. We identify four levels of
AI-related systemic risk and emphasize that discrimination at scale and
systematic hallucinations, despite their capacity to destabilize institutions
and fundamental rights, may not fall under current legal definitions, given the
AI Act's focus on frontier model capabilities. We then test the DSA, the AI
Act, and our own framework on five key examples, and propose reforms that
broaden systemic risk assessments, strengthen coordination between regulatory
regimes, and explicitly incorporate collective harms.

### 2. [Tracing the Techno-Supremacy Doctrine: A Critical Discourse Analysis of the AI Executive Elite](http://arxiv.org/pdf/2509.18079v1)

Authors: Héctor Pérez-Urbina

This paper critically analyzes the discourse of the 'AI executive elite,' a
group of highly influential individuals shaping the way AI is funded,
developed, and deployed worldwide. The primary objective is to examine the
presence and dynamics of the 'Techno-Supremacy Doctrine' (TSD), a term
introduced in this study to describe a belief system characterized by an
excessive trust in technology's alleged inherent superiority in solving complex
societal problems. This study integrates quantitative heuristics with in-depth
qualitative investigations. Its methodology is operationalized in a two-phase
critical discourse analysis of 14 texts published by elite members between 2017
and 2025. The findings demonstrate that the elite is not a monolithic bloc but
exhibits a broad spectrum of stances. The discourse is highly dynamic, showing
a marked polarization and general increase in pro-TSD discourse following the
launch of ChatGPT. The analysis identifies key discursive patterns, including a
dominant pro-TSD narrative that combines utopian promises with claims of
inevitable progress, and the common tactic of acknowledging risks only as a
strategic preamble to proposing further technological solutions. This paper
presents TSD as a comprehensive analytical framework and provides a 'diagnostic
toolkit' for identifying its manifestations, from insidious to benign. It
argues that fostering critical awareness of these discursive patterns is
essential for AI practitioners, policymakers, and the public to actively
navigate the future of AI.

### 3. [Community Covert Communication - Dynamic Mass Covert Communication Through Social Media](http://arxiv.org/pdf/2509.17508v1)

Authors: Eric Filiol

Since the early 2010s, social network-based influence technologies have grown
almost exponentially. Initiated by the U.S. Army's early OEV system in 2011, a
number of companies specializing in this field have emerged. The most
(in)famous cases are Bell Pottinger, Cambridge Analytica, Aggregate-IQ and,
more recently, Team Jorge.
  In this paper, we consider the use-case of sock puppet master activities,
which consist in creating hundreds or even thousands of avatars, in organizing
them into communities and implement influence operations. On-purpose software
is used to automate these operations (e.g. Ripon software, AIMS) and organize
these avatar populations into communities. The aim is to organize targeted and
directed influence communication to rather large communities (influence
targets).
  The goal of the present research work is to show how these community
management techniques (social networks) can also be used to
communicate/disseminate relatively large volumes (up to a few tens of Mb) of
multi-level encrypted information to a limited number of actors. To a certain
extent, this can be compared to a Dark Post-type function, with a number of
much more powerful potentialities. As a consequence, the concept of
communication has been totally redefined and disrupted, so that eavesdropping,
interception and jamming operations no longer make sense.

### 4. [The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies](http://arxiv.org/pdf/2509.18052v1)

Authors: Jiaxu Zhou, Jen-tse Huang, Xuhui Zhou, Man Ho Lam, Xintao Wang, Hao Zhu, Wenxuan Wang, Maarten Sap

Large Language Models (LLMs) are increasingly used for social simulation,
where populations of agents are expected to reproduce human-like collective
behavior. However, we find that many recent studies adopt experimental designs
that systematically undermine the validity of their claims. From a survey of
over 40 papers, we identify six recurring methodological flaws: agents are
often homogeneous (Profile), interactions are absent or artificially imposed
(Interaction), memory is discarded (Memory), prompts tightly control outcomes
(Minimal-Control), agents can infer the experimental hypothesis (Unawareness),
and validation relies on simplified theoretical models rather than real-world
data (Realism). For instance, GPT-4o and Qwen-3 correctly infer the underlying
social experiment in 53.1% of cases when given instructions from prior
work-violating the Unawareness principle. We formalize these six requirements
as the PIMMUR principles and argue they are necessary conditions for credible
LLM-based social simulation. To demonstrate their impact, we re-run five
representative studies using a framework that enforces PIMMUR and find that the
reported social phenomena frequently fail to emerge under more rigorous
conditions. Our work establishes methodological standards for LLM-based
multi-agent research and provides a foundation for more reliable and
reproducible claims about "AI societies."

### 5. [Explainability matters: The effect of liability rules on the healthcare sector](http://arxiv.org/pdf/2509.17334v1)

Authors: Jiawen Wei, Elena Verona, Andrea Bertolini, Gianmarco Mengaldo

Explainability, the capability of an artificial intelligence system (AIS) to
explain its outcomes in a manner that is comprehensible to human beings at an
acceptable level, has been deemed essential for critical sectors, such as
healthcare. Is it really the case? In this perspective, we consider two extreme
cases, ``Oracle'' (without explainability) versus ``AI Colleague'' (with
explainability) for a thorough analysis. We discuss how the level of automation
and explainability of AIS can affect the determination of liability among the
medical practitioner/facility and manufacturer of AIS. We argue that
explainability plays a crucial role in setting a responsibility framework in
healthcare, from a legal standpoint, to shape the behavior of all involved
parties and mitigate the risk of potential defensive medicine practices.

### 6. [Comparing Data Assimilation and Likelihood-Based Inference on Latent State Estimation in Agent-Based Models](http://arxiv.org/pdf/2509.17625v1)

Authors: Blas Kolic, Corrado Monti, Gianmarco De Francisci Morales, Marco Pangallo

In this paper, we present the first systematic comparison of Data
Assimilation (DA) and Likelihood-Based Inference (LBI) in the context of
Agent-Based Models (ABMs). These models generate observable time series driven
by evolving, partially-latent microstates. Latent states need to be estimated
to align simulations with real-world data -- a task traditionally addressed by
DA, especially in continuous and equation-based models such as those used in
weather forecasting. However, the nature of ABMs poses challenges for standard
DA methods. Solving such issues requires adaptation of previous DA techniques,
or ad-hoc alternatives such as LBI. DA approximates the likelihood in a
model-agnostic way, making it broadly applicable but potentially less precise.
In contrast, LBI provides more accurate state estimation by directly leveraging
the model's likelihood, but at the cost of requiring a hand-crafted,
model-specific likelihood function, which may be complex or infeasible to
derive. We compare the two methods on the Bounded-Confidence Model, a
well-known opinion dynamics ABM, where agents are affected only by others
holding sufficiently similar opinions. We find that LBI better recovers latent
agent-level opinions, even under model mis-specification, leading to improved
individual-level forecasts. At the aggregate level, however, both methods
perform comparably, and DA remains competitive across levels of aggregation
under certain parameter settings. Our findings suggest that DA is well-suited
for aggregate predictions, while LBI is preferable for agent-level inference.

### 7. [Mechanistic Interpretability with SAEs: Probing Religion, Violence, and Geography in Large Language Models](http://arxiv.org/pdf/2509.17665v1)

Authors: Katharina Simbeck, Mariam Mahran

Despite growing research on bias in large language models (LLMs), most work
has focused on gender and race, with little attention to religious identity.
This paper explores how religion is internally represented in LLMs and how it
intersects with concepts of violence and geography. Using mechanistic
interpretability and Sparse Autoencoders (SAEs) via the Neuronpedia API, we
analyze latent feature activations across five models. We measure overlap
between religion- and violence-related prompts and probe semantic patterns in
activation contexts. While all five religions show comparable internal
cohesion, Islam is more frequently linked to features associated with violent
language. In contrast, geographic associations largely reflect real-world
religious demographics, revealing how models embed both factual distributions
and cultural stereotypes. These findings highlight the value of structural
analysis in auditing not just outputs but also internal representations that
shape model behavior.

### 8. [Empirical AI Ethics: Reconfiguring Ethics towards a Situated, Plural, and Transformative Approach](http://arxiv.org/pdf/2509.17727v1)

Authors: Paula Helm, Selin Gerlek

Mainstream AI ethics, with its reliance on top-down, principle-driven
frameworks, fails to account for the situated realities of diverse communities
affected by AI (Artificial Intelligence). Critics have argued that AI ethics
frequently serves corporate interests through practices of 'ethics washing',
operating more as a tool for public relations than as a means of preventing
harm or advancing the common good. As a result, growing scepticism among
critical scholars has cast the field as complicit in sustaining harmful systems
rather than challenging or transforming them. In response, this paper adopts a
Science and Technology Studies (STS) perspective to critically interrogate the
field of AI ethics. It hence applies the same analytic tools STS has long
directed at disciplines such as biology, medicine, and statistics to ethics.
This perspective reveals a core tension between vertical (top-down,
principle-based) and horizontal (risk-mitigating, implementation-oriented)
approaches to ethics. By tracing how these models have shaped the discourse, we
show how both fall short in addressing the complexities of AI as a
socio-technical assemblage, embedded in practice and entangled with power. To
move beyond these limitations, we propose a threefold reorientation of AI
ethics. First, we call for a shift in foundations: from top-down abstraction to
empirical grounding. Second, we advocate for pluralisation: moving beyond
Western-centric frameworks toward a multiplicity of onto-epistemic
perspectives. Finally, we outline strategies for reconfiguring AI ethics as a
transformative force, moving from narrow paradigms of risk mitigation toward
co-creating technologies of hope.

### 9. [The Narcissus Hypothesis:Descending to the Rung of Illusion](http://arxiv.org/pdf/2509.17999v1)

Authors: Riccardo Cadei, Christian Internò

Modern foundational models increasingly reflect not just world knowledge, but
patterns of human preference embedded in their training data. We hypothesize
that recursive alignment-via human feedback and model-generated corpora-induces
a social desirability bias, nudging models to favor agreeable or flattering
responses over objective reasoning. We refer to it as the Narcissus Hypothesis
and test it across 31 models using standardized personality assessments and a
novel Social Desirability Bias score. Results reveal a significant drift toward
socially conforming traits, with profound implications for corpus integrity and
the reliability of downstream inferences. We then offer a novel epistemological
interpretation, tracing how recursive bias may collapse higher-order reasoning
down Pearl's Ladder of Causality, culminating in what we refer to as the Rung
of Illusion.

### Databases

### 1. [Propuesta de implementación de catálogos federados para espacios de datos sobre DataHub](http://arxiv.org/pdf/2509.17649v1)

Authors: Carlos Aparicio de Santiago, Pablo Viñuales Esquinas, Irene Plaza Ortiz, Andres Munoz-Arcentales, Gabriel Huecas, Joaquín Salvachúa, Enrique Barra

In the digital era, data spaces are emerging as key ecosystems for the secure
and controlled exchange of information among participants. To achieve this,
components such as metadata catalogs and data space connectors are essential.
This document proposes an implementation and integration solution for both
elements, considering standardization guidelines for data formats, metadata,
and protocols, which ensures interoperability. A hybrid solution is presented:
DataHub is used as a federated catalog for robust metadata management,
leveraging its advanced ingestion, governance, and lineage capabilities. On the
other hand, a custom implementation, Rainbow Catalog, manages ODRL policies for
access and usage. This integration makes it possible to query datasets from
DataHub and associate them with ODRL policies, facilitating negotiation and
transfer flows defined by the Dataspace Protocol. The result is a system that
combines the power of DataHub for large-scale cataloging with the policy
management of the connector crucial for sovereignty and trust in data spaces.

### 2. [Transformer-Gather, Fuzzy-Reconsider: A Scalable Hybrid Framework for Entity Resolution](http://arxiv.org/pdf/2509.17470v1)

Authors: Mohammadreza Sharifi, Danial Ahmadzadeh

Entity resolution plays a significant role in enterprise systems where data
integrity must be rigorously maintained. Traditional methods often struggle
with handling noisy data or semantic understanding, while modern methods suffer
from computational costs or the excessive need for parallel computation. In
this study, we introduce a scalable hybrid framework, which is designed to
address several important problems, including scalability, noise robustness,
and reliable results. We utilized a pre-trained language model to encode each
structured data into corresponding semantic embedding vectors. Subsequently,
after retrieving a semantically relevant subset of candidates, we apply a
syntactic verification stage using fuzzy string matching techniques to refine
classification on the unlabeled data. This approach was applied to a real-world
entity resolution task, which exposed a linkage between a central user
management database and numerous shared hosting server records. Compared to
other methods, this approach exhibits an outstanding performance in terms of
both processing time and robustness, making it a reliable solution for a
server-side product. Crucially, this efficiency does not compromise results, as
the system maintains a high retrieval recall of approximately 0.97. The
scalability of the framework makes it deployable on standard CPU-based
infrastructure, offering a practical and effective solution for
enterprise-level data integrity auditing.

### 3. [MontePrep: Monte-Carlo-Driven Automatic Data Preparation without Target Data Instances](http://arxiv.org/pdf/2509.17553v1)

Authors: Congcong Ge, Yachuan Liu, Yixuan Tang, Yifan Zhu, Yaofeng Tu, Yunjun Gao

In commercial systems, a pervasive requirement for automatic data preparation
(ADP) is to transfer relational data from disparate sources to targets with
standardized schema specifications. Previous methods rely on labor-intensive
supervision signals or target table data access permissions, limiting their
usage in real-world scenarios. To tackle these challenges, we propose an
effective end-to-end ADP framework MontePrep, which enables training-free
pipeline synthesis with zero target-instance requirements. MontePrep is
formulated as an open-source large language model (LLM) powered tree-structured
search problem. It consists of three pivot components, i.e., a data preparation
action sandbox (DPAS), a fundamental pipeline generator (FPG), and an
execution-aware pipeline optimizer (EPO). We first introduce DPAS, a
lightweight action sandbox, to navigate the search-based pipeline generation.
The design of DPAS circumvents exploration of infeasible pipelines. Then, we
present FPG to build executable DP pipelines incrementally, which explores the
predefined action sandbox by the LLM-powered Monte Carlo Tree Search.
Furthermore, we propose EPO, which invokes pipeline execution results from
sources to targets to evaluate the reliability of the generated pipelines in
FPG. In this way, unreasonable pipelines are eliminated, thus facilitating the
search process from both efficiency and effectiveness perspectives. Extensive
experimental results demonstrate the superiority of MontePrep with significant
improvement against five state-of-the-art competitors.

### 4. [From Documents to Database: Failure Modes for Industrial Assets](http://arxiv.org/pdf/2509.17834v1)

Authors: Duygu Kabakci-Zorlu, Fabio Lorenzi, John Sheehan, Karol Lynch, Bradley Eck

We propose an interactive system using foundation models and user-provided
technical documents to generate Failure Mode and Effects Analyses (FMEA) for
industrial equipment. Our system aggregates unstructured content across
documents to generate an FMEA and stores it in a relational database.
Leveraging this tool, the time required for creation of this
knowledge-intensive content is reduced, outperforming traditional manual
approaches. This demonstration showcases the potential of foundation models to
facilitate the creation of specialized structured content for enterprise asset
management systems.

### Distributed, Parallel, and Cluster Computing

### 1. [Institutional Research Computing Capabilities in Australia: 2024](http://arxiv.org/pdf/2509.17351v1)

Authors: Slava Kitaeff, Luc Betbeder-Matibet, Jake Carroll, Stephen Giugni, David Abramson, John Zaitseff, Sarah Walters, David Powell, Chris Bording, Trung Nguyen, Angus Macoustra, Fabien Voisin, Bowen Chen, Jarrod Hurley

Institutional research computing infrastructure plays a vital role in
Australia's research ecosystem, complementing and extending national
facilities. This paper analyses research computing capabilities across
Australian universities and organisations, showing how institutional systems
support research excellence through local compute resources, specialised
hardware, and cluster solutions. Our study finds that nearly 112,258 CPU cores
and 2,241 GPUs serve over 6,000 researchers as essential bridges between
desktops and national facilities, enabling workflows from development to
large-scale computations. The estimated replacement value of this
infrastructure is $144M AUD. Drawing on detailed data from multiple
institutions, we identify key patterns in deployment, utilisation, and
strategic alignment with research priorities. Institutional resources provide
critical support for data-intensive projects, facilitate training and
higher-degree student research, enable prototyping and development, and ensure
data sovereignty compliance when required. The analysis shows how these
facilities leverage national investments while addressing institution-specific
needs that national systems cannot meet. We present evidence that strategic
investment in institutional capabilities yields significant returns through
greater research productivity, enhanced graduate training, and improved
outcomes. The study offers insights for organisations planning computing
strategies and highlights the importance of maintaining robust institutional
resources alongside national facilities.

### 2. [Cronus: Efficient LLM inference on Heterogeneous GPU Clusters via Partially Disaggregated Prefill](http://arxiv.org/pdf/2509.17357v1)

Authors: Yunzhao Liu, Qiang Xu, Y. Charlie Hu

Efficient LLM inference is critical for real-world applications, especially
within heterogeneous GPU clusters commonly found in organizations and
on-premise datacenters as GPU architecture rapidly evolves. Current
disaggregated prefill strategies, which separate the prefill and decode stages
of LLM inference across different GPUs, often suffer from suboptimal
performance due to imbalances between GPU capabilities and workload demands. On
the other hand, extending conventional data parallelism and pipeline
parallelism to heterogeneous setups incurs high inference latencies. To address
these challenges, we introduce Cronus, a novel LLM inference system designed to
dynamically balance workloads across heterogeneous GPUs using partially
disaggregated prefill. Cronus partitions each prefill stage and executes its
initial portion on the low-end GPU, while overlapping the remaining prefill and
decode stages of earlier requests on the high-end GPU. Extensive evaluations
across various high-end and low-end GPU combinations demonstrate that Cronus
significantly improves the throughput over disaggregated prefill. It also
reduces TTFT P99 and TBT P99 significantly over DP and PP while maintaining
similar or better throughput.

### 3. [Asteria: Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access](http://arxiv.org/pdf/2509.17360v1)

Authors: Chaoyi Ruan, Chao Bi, Kaiwen Zheng, Ziji Shi, Xinyi Wan, Jialin Li

Large Language Model (LLM) agents tackle data-intensive tasks such as deep
research and code generation. However, their effectiveness depends on frequent
interactions with knowledge sources across remote clouds or regions. Such
interactions can create non-trivial latency and cost bottlenecks. Existing
caching solutions focus on exact-match queries, limiting their effectiveness
for semantic knowledge reuse.
  To address this challenge, we introduce Asteria, a novel cross-region
knowledge caching architecture for LLM agents. At its core are two
abstractions: Semantic Element (SE) and Semantic Retrieval Index (Sine). A
semantic element captures the semantic embedding representation of an LLM query
together with performance-aware metadata such as latency, cost, and staticity.
Sine then provides two-stage retrieval: a vector similar index with semantic
embedding for fast candidate selection and a lightweight LLM-powered semantic
judger for precise validation. Atop these primitives, Asteria builds a new
cache interface that includes a new semantic-aware cache hit definition, a
cost-efficient eviction policy, and proactive prefetching. To reduce overhead,
Asteria co-locates the small LLM judger with the main LLM using adaptive
scheduling and resource sharing. Our evaluation demonstrates that Asteria
delivers substantial performance improvements without compromising correctness.
On representative search workloads, Asteria achieves up to a 3.6$\times$
increase in throughput by maintaining cache hit rates of over 85%, while
preserving accuracy virtually identical to non-cached baselines. Asteria also
improves throughput for complex coding tasks by 20%, showcasing its versatility
across diverse agentic workloads.

### 4. [Prefetching in Deep Memory Hierarchies with NVRAM as Main Memory](http://arxiv.org/pdf/2509.17388v1)

Authors: Manel Lurbe, Miguel Avargues, Salvador Petit, Maria E. Gomez, Rui Yang, Guanhao Wang, Julio Sahuquillo

Emerging applications, such as big data analytics and machine learning,
require increasingly large amounts of main memory, often exceeding the capacity
of current commodity processors built on DRAM technology. To address this,
recent research has focused on off-chip memory controllers that facilitate
access to diverse memory media, each with unique density and latency
characteristics. While these solutions improve memory system performance, they
also exacerbate the already significant memory latency. As a result,
multi-level prefetching techniques are essential to mitigate these extended
latencies.
  This paper investigates the advantages of prefetching across both sides of
the memory system: the off-chip memory and the on-chip cache hierarchy. Our
primary objective is to assess the impact of a multi-level prefetching engine
on overall system performance. Additionally, we analyze the individual
contribution of each prefetching level to system efficiency. To achieve this,
the study evaluates two key prefetching approaches: HMC (Hybrid Memory
Controller) and HMC+L1, both of which employ prefetching mechanisms commonly
used by processor vendors. The HMC approach integrates a prefetcher within the
off-chip hybrid memory controller, while the HMC+L1 approach combines this with
additional L1 on-chip prefetchers.
  Experimental results on an out-of-order execution processor show that on-chip
cache prefetchers are crucial for maximizing the benefits of off-chip
prefetching, which in turn further enhances performance. Specifically, the
off-chip HMC prefetcher achieves coverage and accuracy rates exceeding 60% and
up to 80%, while the combined HMC+L1 approach boosts off-chip prefetcher
coverage to as much as 92%. Consequently, overall performance increases from 9%
with the HMC approach to 12% when L1 prefetching is also employed.

### 5. [pBeeGees: A Prudent Approach to Certificate-Decoupled BFT Consensus](http://arxiv.org/pdf/2509.17496v1)

Authors: Kaiji Yang, Jingjing Zhang, Junyao Zheng, Qiwen Liu, Weigang Wu, Jieying Zhou

Pipelined Byzantine Fault Tolerant (BFT) consensus is fundamental to
permissioned blockchains. However, many existing protocols are limited by the
requirement for view-consecutive quorum certificates (QCs). This constraint
impairs performance and creates liveness vulnerabilities under adverse network
conditions. Achieving "certificate decoupling"-committing blocks without this
requirement-is therefore a key research goal. While the recent BeeGees
algorithm achieves this, our work reveals that it suffers from security and
liveness issues. To address this problem, this paper makes two primary
contributions. First, we formally define these flaws as the Invalid Block
Problem and the Hollow Chain Problem. Second, we propose pBeeGees, a new
algorithm that addresses these issues while preserving certificate decoupling
with no additional computational overhead. To achieve this, pBeeGees integrates
traceback and pre-commit validation to solve the Invalid Block Problem.Further,
to mitigate the Hollow Chain Problem, we introduce a prudent validation
mechanism, which prevents unverified branches from growing excessively. To
summarize, pBeeGees is the first protocol to simultaneously achieve safety,
liveness, and certificate decoupling in a pipelined BFT framework. Experiments
confirm that our design significantly reduces block commit latency compared to
classic algorithms, particularly under frequent stopping faults.

### 6. [TACTFL: Temporal Contrastive Training for Multi-modal Federated Learning with Similarity-guided Model Aggregation](http://arxiv.org/pdf/2509.17532v1)

Authors: Guanxiong Sun, Majid Mirmehdi, Zahraa Abdallah, Raul Santos-Rodriguez, Ian Craddock, Telmo de Menezes e Silva Filho

Real-world federated learning faces two key challenges: limited access to
labelled data and the presence of heterogeneous multi-modal inputs. This paper
proposes TACTFL, a unified framework for semi-supervised multi-modal federated
learning. TACTFL introduces a modality-agnostic temporal contrastive training
scheme that conducts representation learning from unlabelled client data by
leveraging temporal alignment across modalities. However, as clients perform
self-supervised training on heterogeneous data, local models may diverge
semantically. To mitigate this, TACTFL incorporates a similarity-guided model
aggregation strategy that dynamically weights client models based on their
representational consistency, promoting global alignment. Extensive experiments
across diverse benchmarks and modalities, including video, audio, and wearable
sensors, demonstrate that TACTFL achieves state-of-the-art performance. For
instance, on the UCF101 dataset with only 10% labelled data, TACTFL attains
68.48% top-1 accuracy, significantly outperforming the FedOpt baseline of
35.35%. Code will be released upon publication.

### 7. [Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs](http://arxiv.org/pdf/2509.17542v1)

Authors: Xing Chen, Rong Shi, Lu Zhao, Lingbin Wang, Xiao Jin, Yueqiang Chen, Hongfeng Sun

LLM-based applications have been widely used in various industries, but with
the increasing of models size, an efficient large language model (LLM)
inference system is an urgent problem to be solved for service providers. Since
the inference system is divided into two stage with different characteristics:
Prefill and Decode, the two stage will interfere with each other during the
inference process. Toward this end, a P-D disaggregated inference framework is
proposed by some researchers. Current research is done on homogeneous GPUs, and
lacks deployment solutions based on business scenarios. Compared with
homogeneous GPUs, using heterogeneous GPUs to construct inference systems can
better improve resource utilization and reduce costs. Even if GPUs from
different vendors are used to build inference systems, on the basis of reducing
costs, the resource utilization rate can be improved and the dependence on a
single vendor can be reduced. Therefore, a P-D disaggreagetd inference system
based on heterogeneous GPUs is designed, and the heterogeneous compatible
transmission module in the system is designed to address heterogeneous GPU data
compatibility issues. Then, a joint optimization algorithm of parallel strategy
and instance number allocation is proposed to obtain the deployment solutions.
Finally, the experimental results show that the P-D disaggregated inference
system can well solve the hybrid inference problem of heterogeneous GPUs from
different vendors, and the joint optimization algorithm can obtain the optimal
deployment solution.

### 8. [A Lightweight Approach for State Machine Replication](http://arxiv.org/pdf/2509.17771v1)

Authors: Christian Cachin, Jinfeng Dou, Christian Scheideler, Philipp Schneider

We present a lightweight solution for state machine replication with
commitment certificates. Specifically, we adapt a simple median rule from the
stabilizing consensus problem [Doerr11] to operate in a client-server setting
where arbitrary servers may be blocked adaptively based on past system
information. We further extend our protocol by compressing information about
committed commands, thus keeping the protocol lightweight, while still enabling
clients to easily prove that their commands have indeed been committed on the
shared state. Our approach guarantees liveness as long as at most a constant
fraction of servers are blocked, ensures safety under any number of blocked
servers, and supports fast recovery from massive blocking attacks. In addition
to offering near-optimal performance in several respects, our method is fully
decentralized, unlike other near-optimal solutions that rely on leaders. In
particular, our solution is robust against adversaries that target key servers
(which captures insider-based denial-of-service attacks), whereas leader-based
approaches fail under such a blocking model.

### 9. [Expert-as-a-Service: Towards Efficient, Scalable, and Robust Large-scale MoE Serving](http://arxiv.org/pdf/2509.17863v1)

Authors: Ziming Liu, Boyu Tian, Guoteng Wang, Zhen Jiang, Peng Sun, Zhenhua Han, Tian Tang, Xiaohe Hu, Yanmin Jia, Yan Zhang, He Liu, Mingjun Zhang, Yiqi Zhang, Qiaoling Chen, Shenggan Cheng, Mingyu Gao, Yang You, Siyuan Feng

Mixture-of-Experts (MoE) models challenge serving infrastructures with
dynamic, sparse expert utilization, causing instability on conventional systems
designed for dense architectures. We propose EaaS, a novel serving system to
enable efficient, scalable, and robust MoE deployment. Our system disaggregates
MoE modules into independent, stateless services. This design enables
fine-grained resource scaling and provides inherent fault tolerance by
decoupling compute units. The architecture is powered by a high-performance,
CPU-free peer-to-peer communication library that ensures minimal overhead and
high throughput. Experiments confirm EaaS's scalability and efficiency,
achieving performance comparable to monolithic systems while providing robust
fault tolerance and strong scalability. EaaS incurs less than a 2% throughput
reduction under simulated hardware failures that would otherwise halt
monolithic architectures. It further saves up to 37.5% of computing resources
through dynamic fine-grained adaptation to serving traffic, demonstrating
strong resilience for large-scale MoE deployment in production.

### 10. [XaaS Containers: Performance-Portable Representation With Source and IR Containers](http://arxiv.org/pdf/2509.17914v1)

Authors: Marcin Copik, Eiman Alnuaimi, Alok Kamatar, Valerie Hayot-Sasson, Alberto Madonna, Todd Gamblin, Kyle Chard, Ian Foster, Torsten Hoefler

High-performance computing (HPC) systems and cloud data centers are
converging, and containers are becoming the default method of portable software
deployment. Yet, while containers simplify software management, they face
significant performance challenges in HPC environments as they must sacrifice
hardware-specific optimizations to achieve portability. Although HPC containers
can use runtime hooks to access optimized MPI libraries and GPU devices, they
are limited by application binary interface (ABI) compatibility and cannot
overcome the effects of early-stage compilation decisions. Acceleration as a
Service (XaaS) proposes a vision of performance-portable containers, where a
containerized application should achieve peak performance across all HPC
systems. We present a practical realization of this vision through Source and
Intermediate Representation (IR) containers, where we delay
performance-critical decisions until the target system specification is known.
We analyze specialization mechanisms in HPC software and propose a new
LLM-assisted method for automatic discovery of specializations. By examining
the compilation pipeline, we develop a methodology to build containers
optimized for target architectures at deployment time. Our prototype
demonstrates that new XaaS containers combine the convenience of
containerization with the performance benefits of system-specialized builds.

### Digital Libraries

### 1. [Open Political Corpora: Structuring, Searching, and Analyzing Political Text Collections with PoliCorp](http://arxiv.org/pdf/2509.17465v1)

Authors: Nina Smirnova, Muhammad Ahsan Shahid, Philipp Mayr

In this work, we present PoliCorp (https://demo-pollux.gesis.org/), a web
portal designed to facilitate the search and analysis of political text
corpora. PoliCorp provides researchers with access to rich textual data,
enabling in-depth analysis of parliamentary discourse over time. The platform
currently features a collection of transcripts from debates in the German
parliament, spanning 76 years of proceedings. With the advanced search
functionality, researchers can apply logical operations to combine or exclude
search criteria, making it easier to filter through vast amounts of
parliamentary debate data. The search can be customised by combining multiple
fields and applying logical operators to uncover complex patterns and insights
within the data. Additional data processing steps were performed to enable
web-based search and incorporate extra features. A key feature that
differentiates PoliCorp is its intuitive web-based interface that enables users
to query processed political texts without requiring programming skills. The
user-friendly platform allows for the creation of custom subcorpora via search
parameters, which can be freely downloaded in JSON format for further analysis.

### Discrete Mathematics

### 1. [Generalized DP-colorings of digraphs](http://arxiv.org/pdf/2509.17471v1)

Authors: Lucas Picasarri-Arrieta, Michael Stiebitz

In this paper we consider the following three coloring concepts for digraphs.
First of all, the generalized coloring concept, in which the same colored
vertices of a digraph induce a subdigraph that satisfies a given digraph
property. Second, the concept of variable degeneracy, introduced for graphs by
Borodin, Kostochka and Toft in 2000; this allows to give a common
generalization of the point partition number and the list dichromatic number.
Finally, the DP-coloring concept as introduced for graphs by Dvo\v{r}\'ak and
Postle in 2018, in which a list assignment of a graph is replaced by a cover.
Combining these three coloring concepts leads to generalizations of several
classical coloring results for graphs and digraphs, including the theorems of
Brooks, of Gallai, of Erd\H{o}s, Rubin, and Taylor, and of Bernshteyn,
Kostochka, and Pron for graphs, and the corresponding theorems for digraphs due
to Harutyunyan and Mohar. Our main result combines the DP-coloring and variable
degeneracy concepts for digraphs.

### 2. [Planar induced paths via a decomposition into non-crossing ordered graphs](http://arxiv.org/pdf/2509.17835v1)

Authors: Julien Duron, Hugo Jacob

In any graph, the maximum size of an induced path is bounded by the maximum
size of a path. However, in the general case, one cannot find a converse bound,
even up to an arbitrary function, as evidenced by the case of cliques. Galvin,
Rival and Sands proved in 1982 that, when restricted to weakly sparse graphs,
such a converse property actually holds.
  In this paper, we consider the maximal function $f$ such that any planar
graph (and in general, any graph of bounded genus) containing a path on $n$
vertices contains an induced path of size $f(n)$, and prove that $f(n) \in
\Theta \left(\frac{\log n}{\log \log n}\right)$ by providing a lower bound
matching the upper bound obtained by Esperet, Lemoine and Maffray, up to a
constant factor. We obtain these tight bounds by analyzing graphs ordered along
a Hamiltonian path that admit an edge partition into a bounded number of sets
without crossing edges. In particular, we prove that when such an ordered graph
can be partitioned into $2k$ sets of non-crossing edges, then it contains an
induced path of size $\Omega_k\left(\left(\frac{\log n}{\log \log
n}\right)^{1/k} \right)$ and provide almost matching upper bounds.

### 3. [Circular-arc H-graphs: Ordering Characterizations and Forbidden Patterns](http://arxiv.org/pdf/2509.18021v1)

Authors: Indrajit Paul, Ashok Kumar Das

We introduce the class of circular-arc H-graphs, which generalizes
circular-arc graphs, particularly circular-arc bigraphs. We investigate two
types of ordering-based characterizations of circular-arc r-graphs. Finally, we
provide forbidden patterns for circular-arc r-graphs in terms of specific
vertex orderings.

### Data Structures and Algorithms

### 1. [Theory Meets Practice for Bit Vectors Supporting Rank and Select](http://arxiv.org/pdf/2509.17819v1)

Authors: Florian Kurpicz, Niccolò Rigi-Luperti, Peter Sanders

Bit vectors with support for fast rank and select are a fundamental building
block for compressed data structures. We close a gap between theory and
practice by analyzing an important part of the design space and experimentally
evaluating a sweet spot. The result is the first implementation of a rank and
select data structure for bit vectors with worst-case constant query time, good
practical performance, and a space-overhead of just 0.78%, i.e., between
$4.5\times$ and $64.1\times$ less than previous implementations.

### 2. [Sketching approximations and LP approximations for finite CSPs are related](http://arxiv.org/pdf/2509.17926v1)

Authors: Noah G. Singer, Madhur Tulsiani, Santhoshini Velusamy

We identify a connection between the approximability of CSPs in two models:
(i) sublinear space streaming algorithms, and (ii) the basic LP relaxation. We
show that whenever the basic LP admits an integrality gap, there is an
$\Omega(\sqrt{n})$-space sketching lower bound. We also show that all existing
linear space streaming lower bounds for Max-CSPs can be lifted to integrality
gap instances for basic LPs. For bounded-degree graphs, by combining the
distributed algorithm of Yoshida (STOC 2011) for approximately solving the
basic LP with techniques described in Saxena, Singer, Sudan, and Velusamy (SODA
2025) for simulating a distributed algorithm by a sublinear space streaming
algorithm on bounded-degree instances of Max-DICUT, it appears that there are
sublinear space streaming algorithms implementing the basic LP, for every CSP.
  Based on our results, we conjecture the following dichotomy theorem: Whenever
the basic LP admits an integrality gap, there is a linear space single-pass
streaming lower bound, and when the LP is roundable, there is a sublinear space
streaming algorithm.

### 3. [Hyperbolic Sets in Incomplete Tables](http://arxiv.org/pdf/2509.17591v1)

Authors: J. J. Bernal, J. J. Simón

In this paper, we extend results about the implementation of the
Berlekamp-Massey-Sakata algorithm on data tables having a number of unknown
values.

### Emerging Technologies

### 1. [Truth Without Comprehension: A BlueSky Agenda for Steering the Fourth Mathematical Crisis](http://arxiv.org/pdf/2509.17290v1)

Authors: Runlong Yu, Xiaowei Jia

Machine-generated proofs are poised to reach large-scale, human-unreadable
artifacts. They foreshadow what we call the Fourth Mathematical Crisis. This
crisis crystallizes around three fundamental tensions: trusting proofs that no
human can inspect, understanding results that no one can fully read, and
verifying systems that themselves resist verification. As a minimal yet
principled response, we propose the Human Understandability (HU) meta-axiom,
which requires that every proof admits at least one projection that is
resource-bounded, divergence-measured, and acceptable to a verifier.
Confronting these questions opens a timely research agenda and points toward
new directions in scalable reasoning, interpretable inference, and epistemic
trust for the era of machine-scale mathematics.

### 2. [Diff-GNSS: Diffusion-based Pseudorange Error Estimation](http://arxiv.org/pdf/2509.17397v1)

Authors: Jiaqi Zhu, Shouyi Lu, Ziyao Li, Guirong Zhuo, Lu Xiong

Global Navigation Satellite Systems (GNSS) are vital for reliable urban
positioning. However, multipath and non-line-of-sight reception often introduce
large measurement errors that degrade accuracy. Learning-based methods for
predicting and compensating pseudorange errors have gained traction, but their
performance is limited by complex error distributions. To address this
challenge, we propose Diff-GNSS, a coarse-to-fine GNSS measurement
(pseudorange) error estimation framework that leverages a conditional diffusion
model to capture such complex distributions. Firstly, a Mamba-based module
performs coarse estimation to provide an initial prediction with appropriate
scale and trend. Then, a conditional denoising diffusion layer refines the
estimate, enabling fine-grained modeling of pseudorange errors. To suppress
uncontrolled generative diversity and achieve controllable synthesis, three key
features related to GNSS measurement quality are used as conditions to
precisely guide the reverse denoising process. We further incorporate
per-satellite uncertainty modeling within the diffusion stage to assess the
reliability of the predicted errors. We have collected and publicly released a
real-world dataset covering various scenes. Experiments on public and
self-collected datasets show that DiffGNSS consistently outperforms
state-of-the-art baselines across multiple metrics. To the best of our
knowledge, this is the first application of diffusion models to pseudorange
error estimation. The proposed diffusion-based refinement module is
plug-and-play and can be readily integrated into existing networks to markedly
improve estimation accuracy.

### 3. [Propuesta de implementación de catálogos federados para espacios de datos sobre DataHub](http://arxiv.org/pdf/2509.17649v1)

Authors: Carlos Aparicio de Santiago, Pablo Viñuales Esquinas, Irene Plaza Ortiz, Andres Munoz-Arcentales, Gabriel Huecas, Joaquín Salvachúa, Enrique Barra

In the digital era, data spaces are emerging as key ecosystems for the secure
and controlled exchange of information among participants. To achieve this,
components such as metadata catalogs and data space connectors are essential.
This document proposes an implementation and integration solution for both
elements, considering standardization guidelines for data formats, metadata,
and protocols, which ensures interoperability. A hybrid solution is presented:
DataHub is used as a federated catalog for robust metadata management,
leveraging its advanced ingestion, governance, and lineage capabilities. On the
other hand, a custom implementation, Rainbow Catalog, manages ODRL policies for
access and usage. This integration makes it possible to query datasets from
DataHub and associate them with ODRL policies, facilitating negotiation and
transfer flows defined by the Dataspace Protocol. The result is a system that
combines the power of DataHub for large-scale cataloging with the policy
management of the connector crucial for sovereignty and trust in data spaces.

### 4. [Single-Cell Universal Logic-in-Memory Using 2T-nC FeRAM: An Area and Energy-Efficient Approach for Bulk Bitwise Computation](http://arxiv.org/pdf/2509.17963v1)

Authors: Rudra Biswas, Jiahui Duan, Shan Deng, Xuezhong Niu, Yixin Qin, Prapti Panigrahi, Varun Parekh, Rajiv Joshi, Kai Ni, Vijaykrishnan Narayanan

This work presents a novel approach to configure 2T-nC ferroelectric RAM
(FeRAM) for performing single cell logic-in-memory operations, highlighting its
advantages in energy-efficient computation over conventional DRAM-based
approaches. Unlike conventional 1T-1C dynamic RAM (DRAM), which incurs refresh
overhead, 2T-nC FeRAM offers a promising alternative as a non-volatile memory
solution with low energy consumption. Our key findings include the potential of
quasi-nondestructive readout (QNRO) sensing in 2T-nC FeRAM for logic-in-memory
(LiM) applications, demonstrating its inherent capability to perform inverting
logic without requiring external modifications, a feature absent in traditional
1T-1C DRAM. We successfully implement the MINORITY function within a single
cell of 2T-nC FeRAM, enabling universal NAND and NOR logic, validated through
SPICE simulations and experimental data. Additionally, the research
investigates the feasibility of 3D integration with 2T-nC FeRAM, showing
substantial improvements in storage and computational density, facilitating
bulk-bitwise computation. Our evaluation of eight real-world, data-intensive
applications reveals that 2T-nC FeRAM achieves 2x higher performance and 2.5x
lower energy consumption compared to DRAM. Furthermore, the thermal stability
of stacked 2T-nC FeRAM is validated, confirming its reliable operation when
integrated on a compute die. These findings emphasize the advantages of 2T-nC
FeRAM for LiM, offering superior performance and energy efficiency over
conventional DRAM.

### 5. [VQEzy: An Open-Source Dataset for Parameter Initialize in Variational Quantum Eigensolvers](http://arxiv.org/pdf/2509.17322v1)

Authors: Chi Zhang, Mengxin Zheng, Qian Lou, Hui Min Leung, Fan Chen

Variational Quantum Eigensolvers (VQEs) are a leading class of noisy
intermediate-scale quantum (NISQ) algorithms, whose performance is highly
sensitive to parameter initialization. Although recent machine learning-based
initialization methods have achieved state-of-the-art performance, their
progress has been limited by the lack of comprehensive datasets. Existing
resources are typically restricted to a single domain, contain only a few
hundred instances, and lack complete coverage of Hamiltonians, ansatz circuits,
and optimization trajectories. To overcome these limitations, we introduce
VQEzy, the first large-scale dataset for VQE parameter initialization. VQEzy
spans three major domains and seven representative tasks, comprising 12,110
instances with full VQE specifications and complete optimization trajectories.
The dataset is available online, and will be continuously refined and expanded
to support future research in VQE optimization.

### 6. [DiffQ: Unified Parameter Initialization for Variational Quantum Algorithms via Diffusion Models](http://arxiv.org/pdf/2509.17324v1)

Authors: Chi Zhang, Mengxin Zheng, Qian Lou, Fan Chen

Variational Quantum Algorithms (VQAs) are widely used in the noisy
intermediate-scale quantum (NISQ) era, but their trainability and performance
depend critically on initialization parameters that shape the optimization
landscape. Existing machine learning-based initializers achieve
state-of-the-art results yet remain constrained to single-task domains and
small datasets of only hundreds of samples. We address these limitations by
reformulating VQA parameter initialization as a generative modeling problem and
introducing DiffQ, a parameter initializer based on the Denoising Diffusion
Probabilistic Model (DDPM). To support robust training and evaluation, we
construct a dataset of 15,085 instances spanning three domains and five
representative tasks. Experiments demonstrate that DiffQ surpasses baselines,
reducing initial loss by up to 8.95 and convergence steps by up to 23.4%.

### 7. [Evaluating the Energy Efficiency of NPU-Accelerated Machine Learning Inference on Embedded Microcontrollers](http://arxiv.org/pdf/2509.17533v1)

Authors: Anastasios Fanariotis, Theofanis Orphanoudakis, Vasilis Fotopoulos

The deployment of machine learning (ML) models on microcontrollers (MCUs) is
constrained by strict energy, latency, and memory requirements, particularly in
battery-operated and real-time edge devices. While software-level optimizations
such as quantization and pruning reduce model size and computation, hardware
acceleration has emerged as a decisive enabler for efficient embedded
inference. This paper evaluates the impact of Neural Processing Units (NPUs) on
MCU-based ML execution, using the ARM Cortex-M55 core combined with the
Ethos-U55 NPU on the Alif Semiconductor Ensemble E7 development board as a
representative platform. A rigorous measurement methodology was employed,
incorporating per-inference net energy accounting via GPIO-triggered
high-resolution digital multimeter synchronization and idle-state subtraction,
ensuring accurate attribution of energy costs. Experimental results across six
representative ML models -including MiniResNet, MobileNetV2, FD-MobileNet,
MNIST, TinyYolo, and SSD-MobileNet- demonstrate substantial efficiency gains
when inference is offloaded to the NPU. For moderate to large networks, latency
improvements ranged from 7x to over 125x, with per-inference net energy
reductions up to 143x. Notably, the NPU enabled execution of models unsupported
on CPU-only paths, such as SSD-MobileNet, highlighting its functional as well
as efficiency advantages. These findings establish NPUs as a cornerstone of
energy-aware embedded AI, enabling real-time, power-constrained ML inference at
the MCU level.

### 8. [Towards Seeing Bones at Radio Frequency](http://arxiv.org/pdf/2509.17979v1)

Authors: Yiwen Song, Hongyang Li, Kuang Yuan, Ran Bi, Swarun Kumar

Wireless sensing literature has long aspired to achieve X-ray-like vision at
radio frequencies. Yet, state-of-the-art wireless sensing literature has yet to
generate the archetypal X-ray image: one of the bones beneath flesh. In this
paper, we explore MCT, a penetration-based RF-imaging system for imaging bones
at mm-resolution, one that significantly exceeds prior penetration-based RF
imaging literature. Indeed the long wavelength, significant attenuation and
complex diffraction that occur as RF propagates through flesh, have long
limited imaging resolution (to several centimeters at best). We address these
concerns through a novel penetration-based synthetic aperture algorithm,
coupled with a learning-based pipeline to correct for diffraction-induced
artifacts. A detailed evaluation of meat models demonstrates a resolution
improvement from sub-decimeter to sub-centimeter over prior art in RF
penetrative imaging.

### 9. [HuMam: Humanoid Motion Control via End-to-End Deep Reinforcement Learning with Mamba](http://arxiv.org/pdf/2509.18046v1)

Authors: Yinuo Wang, Yuanyang Qi, Jinzhao Zhou, Gavin Tao

End-to-end reinforcement learning (RL) for humanoid locomotion is appealing
for its compact perception-action mapping, yet practical policies often suffer
from training instability, inefficient feature fusion, and high actuation cost.
We present HuMam, a state-centric end-to-end RL framework that employs a
single-layer Mamba encoder to fuse robot-centric states with oriented footstep
targets and a continuous phase clock. The policy outputs joint position targets
tracked by a low-level PD loop and is optimized with PPO. A concise six-term
reward balances contact quality, swing smoothness, foot placement, posture, and
body stability while implicitly promoting energy saving. On the JVRC-1 humanoid
in mc-mujoco, HuMam consistently improves learning efficiency, training
stability, and overall task performance over a strong feedforward baseline,
while reducing power consumption and torque peaks. To our knowledge, this is
the first end-to-end humanoid RL controller that adopts Mamba as the fusion
backbone, demonstrating tangible gains in efficiency, stability, and control
economy.

### Formal Languages and Automata Theory

### 1. [The hereditariness problem for the Černý conjecture](http://arxiv.org/pdf/2509.17992v1)

Authors: Emanuele Rodaro, Riccardo Venturi

This paper addresses the lifting problem for the \v{C}ern\'y conjecture:
namely, whether the validity of the conjecture for a quotient automaton can
always be transferred (or "lifted") to the original automaton. Although a
complete solution remains open, we show that it is sufficient to verify the
\v{C}ern\'y conjecture for three specific subclasses of reset automata:
radical, simple, and quasi-simple. Our approach relies on establishing a Galois
connection between the lattices of congruences and ideals of the transition
monoid. This connection not only serves as the main tool in our proofs but also
provides a systematic method for computing the radical ideal and for deriving
structural insights about these classes. In particular, we show that for every
simple or quasi-simple automaton $\mathcal{A}$, the transition monoid
$\text{M}(\mathcal{A})$ possesses a unique ideal covering the minimal ideal of
constant (reset) maps; a result of similar flavor holds for the class of
radical automata.

### Graphics

### 1. [A Comparative Study of Different Edit Distance-Based Methods for Feature Tracking using Merge Trees on Time-Varying Scalar Fields](http://arxiv.org/pdf/2509.17974v1)

Authors: Son Le Thanh, Tino Weinkauf

Feature tracking in time-varying scalar fields is a fundamental task in
scientific computing. Topological descriptors, which summarize important
features of data, have proved to be viable tools to facilitate this task. The
merge tree is a topological descriptor that captures the connectivity behaviors
of the sub- or superlevel sets of a scalar field. Edit distances between merge
trees play a vital role in effective temporal data tracking. Existing methods
to compute them fall into two main classes, namely whether they are dependent
or independent of the branch decomposition. These two classes represent the
most prominent approaches for producing tracking results. In this paper, we
compare four different merge tree edit distance-based methods for feature
tracking. We demonstrate that these methods yield distinct results with both
analytical and real-world data sets. Furthermore, we investigate how these
results vary and identify the factors that influence them. Our experiments
reveal significant differences in tracked features over time, even among those
produced by techniques within the same category.

### 2. [VideoFrom3D: 3D Scene Video Generation via Complementary Image and Video Diffusion Models](http://arxiv.org/pdf/2509.17985v1)

Authors: Geonung Kim, Janghyeok Han, Sunghyun Cho

In this paper, we propose VideoFrom3D, a novel framework for synthesizing
high-quality 3D scene videos from coarse geometry, a camera trajectory, and a
reference image. Our approach streamlines the 3D graphic design workflow,
enabling flexible design exploration and rapid production of deliverables. A
straightforward approach to synthesizing a video from coarse geometry might
condition a video diffusion model on geometric structure. However, existing
video diffusion models struggle to generate high-fidelity results for complex
scenes due to the difficulty of jointly modeling visual quality, motion, and
temporal consistency. To address this, we propose a generative framework that
leverages the complementary strengths of image and video diffusion models.
Specifically, our framework consists of a Sparse Anchor-view Generation (SAG)
and a Geometry-guided Generative Inbetweening (GGI) module. The SAG module
generates high-quality, cross-view consistent anchor views using an image
diffusion model, aided by Sparse Appearance-guided Sampling. Building on these
anchor views, GGI module faithfully interpolates intermediate frames using a
video diffusion model, enhanced by flow-based camera control and structural
guidance. Notably, both modules operate without any paired dataset of 3D scene
models and natural images, which is extremely difficult to obtain.
Comprehensive experiments show that our method produces high-quality,
style-consistent scene videos under diverse and challenging scenarios,
outperforming simple and extended baselines.

### 3. ["I don't like my avatar": Investigating Human Digital Doubles](http://arxiv.org/pdf/2509.17748v1)

Authors: Siyi Liu, Kazi Injamamul Haque, Zerrin Yumak

Creating human digital doubles is becoming easier and much more accessible to
everyone using consumer grade devices. In this work, we investigate how avatar
style (realistic vs cartoon) and avatar familiarity (self, acquaintance,
unknown person) affect self/other-identification, perceived realism, affinity
and social presence with a controlled offline experiment. We created two styles
of avatars (realistic-looking MetaHumans and cartoon-looking ReadyPlayerMe
avatars) and facial animations stimuli for them using performance capture.
Questionnaire responses demonstrate that higher appearance realism leads to a
higher level of identification, perceived realism and social presence. However,
avatars with familiar faces, especially those with high appearance realism,
lead to a lower level of identification, perceived realism, and affinity.
Although participants identified their digital doubles as their own, they
consistently did not like their avatars, especially of realistic appearance.
But they were less critical and more forgiving about their acquaintance's or an
unknown person's digital double.

### 4. [Effect of Appearance and Animation Realism on the Perception of Emotionally Expressive Virtual Humans](http://arxiv.org/pdf/2509.17803v1)

Authors: Nabila Amadou, Kazi Injamamul Haque, Zerrin Yumak

3D Virtual Human technology is growing with several potential applications in
health, education, business and telecommunications. Investigating the
perception of these virtual humans can help guide to develop better and more
effective applications. Recent developments show that the appearance of the
virtual humans reached to a very realistic level. However, there is not yet
adequate analysis on the perception of appearance and animation realism for
emotionally expressive virtual humans. In this paper, we designed a user
experiment and analyzed the effect of a realistic virtual human's appearance
realism and animation realism in varying emotion conditions. We found that
higher appearance realism and higher animation realism leads to higher social
presence and higher attractiveness ratings. We also found significant effects
of animation realism on perceived realism and emotion intensity levels. Our
study sheds light into how appearance and animation realism effects the
perception of highly realistic virtual humans in emotionally expressive
scenarios and points out to future directions.

### 5. [Preconditioned Deformation Grids](http://arxiv.org/pdf/2509.18097v1)

Authors: Julian Kaltheuner, Alexander Oebel, Hannah Droege, Patrick Stotko, Reinhard Klein

Dynamic surface reconstruction of objects from point cloud sequences is a
challenging field in computer graphics. Existing approaches either require
multiple regularization terms or extensive training data which, however, lead
to compromises in reconstruction accuracy as well as over-smoothing or poor
generalization to unseen objects and motions. To address these lim- itations,
we introduce Preconditioned Deformation Grids, a novel technique for estimating
coherent deformation fields directly from unstructured point cloud sequences
without requiring or forming explicit correspondences. Key to our approach is
the use of multi-resolution voxel grids that capture the overall motion at
varying spatial scales, enabling a more flexible deformation representation. In
conjunction with incorporating grid-based Sobolev preconditioning into
gradient-based optimization, we show that applying a Chamfer loss between the
input point clouds as well as to an evolving template mesh is sufficient to
obtain accurate deformations. To ensure temporal consistency along the object
surface, we include a weak isometry loss on mesh edges which complements the
main objective without constraining deformation fidelity. Extensive evaluations
demonstrate that our method achieves superior results, particularly for long
sequences, compared to state-of-the-art techniques.

### 6. [Learning Neural Antiderivatives](http://arxiv.org/pdf/2509.17755v1)

Authors: Fizza Rubab, Ntumba Elie Nsampi, Martin Balint, Felix Mujkanovic, Hans-Peter Seidel, Tobias Ritschel, Thomas Leimkühler

Neural fields offer continuous, learnable representations that extend beyond
traditional discrete formats in visual computing. We study the problem of
learning neural representations of repeated antiderivatives directly from a
function, a continuous analogue of summed-area tables. Although widely used in
discrete domains, such cumulative schemes rely on grids, which prevents their
applicability in continuous neural contexts. We introduce and analyze a range
of neural methods for repeated integration, including both adaptations of prior
work and novel designs. Our evaluation spans multiple input dimensionalities
and integration orders, assessing both reconstruction quality and performance
in downstream tasks such as filtering and rendering. These results enable
integrating classical cumulative operators into modern neural systems and offer
insights into learning tasks involving differential and integral operators.

### 7. [Towards Seeing Bones at Radio Frequency](http://arxiv.org/pdf/2509.17979v1)

Authors: Yiwen Song, Hongyang Li, Kuang Yuan, Ran Bi, Swarun Kumar

Wireless sensing literature has long aspired to achieve X-ray-like vision at
radio frequencies. Yet, state-of-the-art wireless sensing literature has yet to
generate the archetypal X-ray image: one of the bones beneath flesh. In this
paper, we explore MCT, a penetration-based RF-imaging system for imaging bones
at mm-resolution, one that significantly exceeds prior penetration-based RF
imaging literature. Indeed the long wavelength, significant attenuation and
complex diffraction that occur as RF propagates through flesh, have long
limited imaging resolution (to several centimeters at best). We address these
concerns through a novel penetration-based synthetic aperture algorithm,
coupled with a learning-based pipeline to correct for diffraction-induced
artifacts. A detailed evaluation of meat models demonstrates a resolution
improvement from sub-decimeter to sub-centimeter over prior art in RF
penetrative imaging.

### Human-Computer Interaction

### 1. [Cooperative Dynamics of Censorship, Misinformation, and Influence Operations: Insights from the Global South and U.S](http://arxiv.org/pdf/2509.17933v1)

Authors: Zaid Hakami, Yuzhou Feng, Bogdan Carbunar

Censorship and the distribution of false information, tools used to
manipulate what users see and believe, are seemingly at opposite ends of the
information access spectrum. Most previous work has examined them in isolation
and within individual countries, leaving gaps in our understanding of how these
information manipulation tools interact and reinforce each other across diverse
societies. In this paper, we study perceptions about the interplay between
censorship, false information, and influence operations, gathered through a
mixed-methods study consisting of a survey (n = 384) and semi-structured
interviews (n = 30) with participants who have experienced these phenomena
across diverse countries in both the Global South and Global North, including
Bangladesh, China, Cuba, Iran, Venezuela, and the United States. Our findings
reveal perceptions of cooperation across various platforms between distinct
entities working together to create information cocoons, within which
censorship and false information become imperceptible to those affected.
Building on study insights, we propose novel platform-level interventions to
enhance transparency and help users navigate information manipulation. In
addition, we introduce the concept of plausibly deniable social platforms,
enabling censored users to provide credible, benign explanations for their
activities, protecting them from surveillance and coercion.

### 2. [ClassMind: Scaling Classroom Observation and Instructional Feedback with Multimodal AI](http://arxiv.org/pdf/2509.18020v1)

Authors: Ao Qu, Yuxi Wen, Jiayi Zhang, Yunge Wen, Yibo Zhao, Alok Prakash, Andrés F. Salazar-Gómez, Paul Pu Liang, Jinhua Zhao

Classroom observation -- one of the most effective methods for teacher
development -- remains limited due to high costs and a shortage of expert
coaches. We present ClassMind, an AI-driven classroom observation system that
integrates generative AI and multimodal learning to analyze classroom artifacts
(e.g., class recordings) and deliver timely, personalized feedback aligned with
pedagogical practices. At its core is AVA-Align, an agent framework that
analyzes long classroom video recordings to generate temporally precise,
best-practice-aligned feedback to support teacher reflection and improvement.
Our three-phase study involved participatory co-design with educators,
development of a full-stack system, and field testing with teachers at
different stages of practice. Teachers highlighted the system's usefulness,
ease of use, and novelty, while also raising concerns about privacy and the
role of human judgment, motivating deeper exploration of future human--AI
coaching partnerships. This work illustrates how multimodal AI can scale expert
coaching and advance teacher development.

### 3. [UIPro: Unleashing Superior Interaction Capability For GUI Agents](http://arxiv.org/pdf/2509.17328v1)

Authors: Hongxin Li, Jingran Su, Jingfan Chen, Zheng Ju, Yuntao Chen, Qing Li, Zhaoxiang Zhang

Building autonomous agents that perceive and operate graphical user
interfaces (GUIs) like humans has long been a vision in the field of artificial
intelligence. Central to these agents is the capability for GUI interaction,
which involves GUI understanding and planning capabilities. Existing methods
have tried developing GUI agents based on the multi-modal comprehension ability
of vision-language models (VLMs). However, the limited scenario, insufficient
size, and heterogeneous action spaces hinder the progress of building
generalist GUI agents. To resolve these issues, this paper proposes
\textbf{UIPro}, a novel generalist GUI agent trained with extensive
multi-platform and multi-task GUI interaction data, coupled with a unified
action space. We first curate a comprehensive dataset encompassing 20.6 million
GUI understanding tasks to pre-train UIPro, granting it a strong GUI grounding
capability, which is key to downstream GUI agent tasks. Subsequently, we
establish a unified action space to harmonize heterogeneous GUI agent task
datasets and produce a merged dataset to foster the action prediction ability
of UIPro via continued fine-tuning. Experimental results demonstrate UIPro's
superior performance across multiple GUI task benchmarks on various platforms,
highlighting the effectiveness of our approach.

### 4. [Towards the State Space Interpretation (SSI): A Formalized Framework for Game Studies and Design](http://arxiv.org/pdf/2509.17610v1)

Authors: Zhenghao Wang, Shuo Xiong

In this paper, we establish structural analogies between core concepts in
quantum mechanics and games. By constructing the Quantum Coin Toss on a quantum
circuit, we preliminarily investigate the similarity between quantum system
behavior and game behavior, thereby formulating the state-operation paradigm.
Using this paradigm, we introduce the conceptual prototype of the State Space
Interpretation (SSI). Based on mathematical and physical theories, particularly
linear algebra, quantum mechanics, and statistical mechanics, we define formal
constructs including state space, evolution path, and derived concepts. With
the SSI, a game is conceptualized as a state space, while a gameplay process
corresponds to an evolution path within this space. We propose that the SSI
constitutes a novel interpretation framework for game design and game studies.
This framework aims to enhance understanding of games and function as a link
between game studies and related fields.

### 5. ["I don't like my avatar": Investigating Human Digital Doubles](http://arxiv.org/pdf/2509.17748v1)

Authors: Siyi Liu, Kazi Injamamul Haque, Zerrin Yumak

Creating human digital doubles is becoming easier and much more accessible to
everyone using consumer grade devices. In this work, we investigate how avatar
style (realistic vs cartoon) and avatar familiarity (self, acquaintance,
unknown person) affect self/other-identification, perceived realism, affinity
and social presence with a controlled offline experiment. We created two styles
of avatars (realistic-looking MetaHumans and cartoon-looking ReadyPlayerMe
avatars) and facial animations stimuli for them using performance capture.
Questionnaire responses demonstrate that higher appearance realism leads to a
higher level of identification, perceived realism and social presence. However,
avatars with familiar faces, especially those with high appearance realism,
lead to a lower level of identification, perceived realism, and affinity.
Although participants identified their digital doubles as their own, they
consistently did not like their avatars, especially of realistic appearance.
But they were less critical and more forgiving about their acquaintance's or an
unknown person's digital double.

### 6. [Effect of Appearance and Animation Realism on the Perception of Emotionally Expressive Virtual Humans](http://arxiv.org/pdf/2509.17803v1)

Authors: Nabila Amadou, Kazi Injamamul Haque, Zerrin Yumak

3D Virtual Human technology is growing with several potential applications in
health, education, business and telecommunications. Investigating the
perception of these virtual humans can help guide to develop better and more
effective applications. Recent developments show that the appearance of the
virtual humans reached to a very realistic level. However, there is not yet
adequate analysis on the perception of appearance and animation realism for
emotionally expressive virtual humans. In this paper, we designed a user
experiment and analyzed the effect of a realistic virtual human's appearance
realism and animation realism in varying emotion conditions. We found that
higher appearance realism and higher animation realism leads to higher social
presence and higher attractiveness ratings. We also found significant effects
of animation realism on perceived realism and emotion intensity levels. Our
study sheds light into how appearance and animation realism effects the
perception of highly realistic virtual humans in emotionally expressive
scenarios and points out to future directions.

### 7. [Toward Affordable and Non-Invasive Detection of Hypoglycemia: A Machine Learning Approach](http://arxiv.org/pdf/2509.17842v1)

Authors: Lawrence Obiuwevwi, Krzysztof J. Rechowicz, Vikas Ashok, Sampath Jayarathna

Diabetes mellitus is a growing global health issue, with Type 1 Diabetes
(T1D) requiring constant monitoring to avoid hypoglycemia. Although Continuous
Glucose Monitors (CGMs) are effective, their cost and invasiveness limit
access, particularly in low-resource settings. This paper proposes a
non-invasive method to classify glycemic states using Galvanic Skin Response
(GSR), a biosignal commonly captured by wearable sensors. We use the merged
OhioT1DM 2018 and 2020 datasets to build a machine learning pipeline that
detects hypoglycemia (glucose < 70 mg/dl) and normoglycemia (glucose > 70
mg/dl) with GSR alone. Seven models are trained and evaluated: Random Forest,
XGBoost, MLP, CNN, LSTM, Logistic Regression, and K-Nearest Neighbors.
Validation sets and 95% confidence intervals are reported to increase
reliability and assess robustness. Results show that the LSTM model achieves a
perfect hypoglycemia recall (1.00) with an F1-score confidence interval of
[0.611-0.745], while XGBoost offers strong performance with a recall of 0.54
even under class imbalance. This approach highlights the potential for
affordable, wearable-compatible glucose monitoring tools suitable for settings
with limited CGM availability using GSR data.
  Index Terms: Hypoglycemia Detection, Galvanic Skin Response, Non Invasive
Monitoring, Wearables, Machine Learning, Confidence Intervals.

### 8. ["I think this is fair'': Uncovering the Complexities of Stakeholder Decision-Making in AI Fairness Assessment](http://arxiv.org/pdf/2509.17956v1)

Authors: Lin Luo, Yuri Nakao, Mathieu Chollet, Hiroya Inakoshi, Simone Stumpf

Assessing fairness in artificial intelligence (AI) typically involves AI
experts who select protected features, fairness metrics, and set fairness
thresholds. However, little is known about how stakeholders, particularly those
affected by AI outcomes but lacking AI expertise, assess fairness. To address
this gap, we conducted a qualitative study with 30 stakeholders without AI
expertise, representing potential decision subjects in a credit rating
scenario, to examine how they assess fairness when placed in the role of
deciding on features with priority, metrics, and thresholds. We reveal that
stakeholders' fairness decisions are more complex than typical AI expert
practices: they considered features far beyond legally protected features,
tailored metrics for specific contexts, set diverse yet stricter fairness
thresholds, and even preferred designing customized fairness. Our results
extend the understanding of how stakeholders can meaningfully contribute to AI
fairness governance and mitigation, underscoring the importance of
incorporating stakeholders' nuanced fairness judgments.

### 9. [Autiverse: Eliciting Autistic Adolescents' Daily Narratives through AI-guided Multimodal Journaling](http://arxiv.org/pdf/2509.17466v1)

Authors: Migyeong Yang, Kyungah Lee, Jinyoung Han, SoHyun Park, Young-Ho Kim

Journaling can potentially serve as an effective method for autistic
adolescents to improve narrative skills. However, its text-centric nature and
high executive functioning demands present barriers to practice. We present
Autiverse, an AI-guided multimodal journaling app for tablets that scaffolds
storytelling through conversational prompts and visual supports. Autiverse
elicits key details through a stepwise dialogue with peer-like, customizable AI
and composes them into an editable four-panel comic strip. Through a two-week
deployment study with 10 autistic adolescent-parent dyads, we examine how
Autiverse supports autistic adolescents to organize their daily experience and
emotion. Autiverse helped them construct coherent narratives, while enabling
parents to learn additional details of their child's events and emotions. The
customized AI peer created a comfortable space for sharing, fostering enjoyment
and a strong sense of agency. We discuss the implications of designing
technologies that complement autistic adolescents' strengths while ensuring
their autonomy and safety in sharing experiences.

### 10. [LingoQ: Bridging the Gap between ESL Learning and Work through AI-Generated Work-Related Quizzes](http://arxiv.org/pdf/2509.17477v1)

Authors: Yeonsun Yang, Sang Won Lee, Jean Y. Song, Sangdoo Yun, Young-Ho Kim

Non-native English speakers performing English-related tasks at work struggle
to sustain ESL learning, despite their motivation. Often, study materials are
disconnected from their work context. Although workers rely on LLM assistants
to address their immediate needs, these interactions may not directly
contribute to their English skills. We present LingoQ, an AI-mediated system
that allows workers to practice English using quizzes generated from their LLM
queries during work. LingoQ leverages these queries using AI to generate
personalized quizzes that workers can review and practice on their smartphones.
We conducted a three-week deployment study with 28 ESL workers to evaluate
LingoQ. Participants valued the relevance of quizzes that reflect their own
context, constantly engaging with the app during the study. This active
engagement improved self-efficacy and led to learning gains for beginners and,
potentially, for intermediate learners. We discuss opportunities of leveraging
users' reliance on LLMs to situate their learning in the user context for
improved learning.

### Information Retrieval

### 1. [MLLM-Driven Semantic Identifier Generation for Generative Cross-Modal Retrieval](http://arxiv.org/pdf/2509.17359v1)

Authors: Tianyuan Li, Lei Wang, Ahtamjan Ahmat, Yating Yang, Bo Ma, Rui Dong, Bangju Han

Generative cross-modal retrieval, which treats retrieval as a generation
task, has emerged as a promising direction with the rise of Multimodal Large
Language Models (MLLMs). In this setting, the model responds to a text query by
generating an identifier corresponding to the target image. However, existing
methods typically rely on manually crafted string IDs, clustering-based labels,
or atomic identifiers requiring vocabulary expansion, all of which face
challenges in semantic alignment or scalability.To address these limitations,
we propose a vocabulary-efficient identifier generation framework that prompts
MLLMs to generate Structured Semantic Identifiers from image-caption pairs.
These identifiers are composed of concept-level tokens such as objects and
actions, naturally aligning with the model's generation space without modifying
the tokenizer. Additionally, we introduce a Rationale-Guided Supervision
Strategy, prompting the model to produce a one-sentence explanation alongside
each identifier serves as an auxiliary supervision signal that improves
semantic grounding and reduces hallucinations during training.

### 2. [Simplified Longitudinal Retrieval Experiments: A Case Study on Query Expansion and Document Boosting](http://arxiv.org/pdf/2509.17440v1)

Authors: Jüri Keller, Maik Fröbe, Gijs Hendriksen, Daria Alexander, Martin Potthast, Philipp Schaer

The longitudinal evaluation of retrieval systems aims to capture how
information needs and documents evolve over time. However, classical
Cranfield-style retrieval evaluations only consist of a static set of queries
and documents and thereby miss time as an evaluation dimension. Therefore,
longitudinal evaluations need to complement retrieval toolkits with custom
logic. This custom logic increases the complexity of research software, which
might reduce the reproducibility and extensibility of experiments. Based on our
submissions to the 2024 edition of LongEval, we propose a custom extension of
ir_datasets for longitudinal retrieval experiments. This extension allows for
declaratively, instead of imperatively, describing important aspects of
longitudinal retrieval experiments, e.g., which queries, documents, and/or
relevance feedback are available at which point in time. We reimplement our
submissions to LongEval 2024 against our new ir_datasets extension, and find
that the declarative access can reduce the complexity of the code.

### 3. [WildClaims: Information Access Conversations in the Wild(Chat)](http://arxiv.org/pdf/2509.17442v1)

Authors: Hideaki Joko, Shakiba Amirshahi, Charles L. A. Clarke, Faegheh Hasibi

The rapid advancement of Large Language Models (LLMs) has transformed
conversational systems into practical tools used by millions. However, the
nature and necessity of information retrieval in real-world conversations
remain largely unexplored, as research has focused predominantly on
traditional, explicit information access conversations. The central question
is: What do real-world information access conversations look like? To this end,
we first conduct an observational study on the WildChat dataset, large-scale
user-ChatGPT conversations, finding that users' access to information occurs
implicitly as check-worthy factual assertions made by the system, even when the
conversation's primary intent is non-informational, such as creative writing.
To enable the systematic study of this phenomenon, we release the WildClaims
dataset, a novel resource consisting of 121,905 extracted factual claims from
7,587 utterances in 3,000 WildChat conversations, each annotated for
check-worthiness. Our preliminary analysis of this resource reveals that
conservatively 18% to 51% of conversations contain check-worthy assertions,
depending on the methods employed, and less conservatively, as many as 76% may
contain such assertions. This high prevalence underscores the importance of
moving beyond the traditional understanding of explicit information access, to
address the implicit information access that arises in real-world user-system
conversations.

### 4. [LongEval at CLEF 2025: Longitudinal Evaluation of IR Systems on Web and Scientific Data](http://arxiv.org/pdf/2509.17469v1)

Authors: Matteo Cancellieri, Alaa El-Ebshihy, Tobias Fink, Maik Fröbe, Petra Galuščáková, Gabriela Gonzalez-Saez, Lorraine Goeuriot, David Iommi, Jüri Keller, Petr Knoth, Philippe Mulhem, Florina Piroi, David Pride, Philipp Schaer

The LongEval lab focuses on the evaluation of information retrieval systems
over time. Two datasets are provided that capture evolving search scenarios
with changing documents, queries, and relevance assessments. Systems are
assessed from a temporal perspective-that is, evaluating retrieval
effectiveness as the data they operate on changes. In its third edition,
LongEval featured two retrieval tasks: one in the area of ad-hoc web retrieval,
and another focusing on scientific article retrieval. We present an overview of
this year's tasks and datasets, as well as the participating systems. A total
of 19 teams submitted their approaches, which we evaluated using nDCG and a
variety of measures that quantify changes in retrieval effectiveness over time.

### 5. [Human vs. Agent in Task-Oriented Conversations](http://arxiv.org/pdf/2509.17619v1)

Authors: Zhefan Wang, Ning Geng, Zhiqiang Guo, Weizhi Ma, Min Zhang

Task-oriented conversational systems are essential for efficiently addressing
diverse user needs, yet their development requires substantial amounts of
high-quality conversational data that is challenging and costly to obtain.
While large language models (LLMs) have demonstrated potential in generating
synthetic conversations, the extent to which these agent-generated interactions
can effectively substitute real human conversations remains unclear. This work
presents the first systematic comparison between LLM-simulated users and human
users in personalized task-oriented conversations. We propose a comprehensive
analytical framework encompassing three key aspects (conversation strategy,
interaction style, and conversation evaluation) and ten distinct dimensions for
evaluating user behaviors, and collect parallel conversational datasets from
both human users and LLM agent users across four representative scenarios under
identical conditions. Our analysis reveals significant behavioral differences
between the two user types in problem-solving approaches, question broadness,
user engagement, context dependency, feedback polarity and promise, language
style, and hallucination awareness. We found consistency in the agent users and
human users across the depth-first or breadth-first dimensions, as well as the
usefulness dimensions. These findings provide critical insights for advancing
LLM-based user simulation. Our multi-dimensional taxonomy constructed a
generalizable framework for analyzing user behavior patterns, offering insights
from LLM agent users and human users. By this work, we provide perspectives on
rethinking how to use user simulation in conversational systems in the future.

### 6. [A Generative Framework for Personalized Sticker Retrieval](http://arxiv.org/pdf/2509.17749v1)

Authors: Changjiang Zhou, Ruqing Zhang, Jiafeng Guo, Yu-An Liu, Fan Zhang, Ganyuan Luo, Xueqi Cheng

Formulating information retrieval as a variant of generative modeling,
specifically using autoregressive models to generate relevant identifiers for a
given query, has recently attracted considerable attention. However, its
application to personalized sticker retrieval remains largely unexplored and
presents unique challenges: existing relevance-based generative retrieval
methods typically lack personalization, leading to a mismatch between diverse
user expectations and the retrieved results. To address this gap, we propose
PEARL, a novel generative framework for personalized sticker retrieval, and
make two key contributions: (i) To encode user-specific sticker preferences, we
design a representation learning model to learn discriminative user
representations. It is trained on three prediction tasks that leverage personal
information and click history; and (ii) To generate stickers aligned with a
user's query intent, we propose a novel intent-aware learning objective that
prioritizes stickers associated with higher-ranked intents. Empirical results
from both offline evaluations and online tests demonstrate that PEARL
significantly outperforms state-of-the-art methods.

### 7. [SeqUDA-Rec: Sequential User Behavior Enhanced Recommendation via Global Unsupervised Data Augmentation for Personalized Content Marketing](http://arxiv.org/pdf/2509.17361v1)

Authors: Ruihan Luo, Xuanjing Chen, Ziyang Ding

Personalized content marketing has become a crucial strategy for digital
platforms, aiming to deliver tailored advertisements and recommendations that
match user preferences. Traditional recommendation systems often suffer from
two limitations: (1) reliance on limited supervised signals derived from
explicit user feedback, and (2) vulnerability to noisy or unintentional
interactions. To address these challenges, we propose SeqUDA-Rec, a novel deep
learning framework that integrates user behavior sequences with global
unsupervised data augmentation to enhance recommendation accuracy and
robustness. Our approach first constructs a Global User-Item Interaction Graph
(GUIG) from all user behavior sequences, capturing both local and global item
associations. Then, a graph contrastive learning module is applied to generate
robust embeddings, while a sequential Transformer-based encoder models users'
evolving preferences. To further enhance diversity and counteract sparse
supervised labels, we employ a GAN-based augmentation strategy, generating
plausible interaction patterns and supplementing training data. Extensive
experiments on two real-world marketing datasets (Amazon Ads and TikTok Ad
Clicks) demonstrate that SeqUDA-Rec significantly outperforms state-of-the-art
baselines such as SASRec, BERT4Rec, and GCL4SR. Our model achieves a 6.7%
improvement in NDCG@10 and 11.3% improvement in HR@10, proving its
effectiveness in personalized advertising and intelligent content
recommendation.

### 8. [Shilling Recommender Systems by Generating Side-feature-aware Fake User Profiles](http://arxiv.org/pdf/2509.17918v1)

Authors: Yuanrong Wang, Yingpeng Du

Recommender systems (RS) greatly influence users' consumption decisions,
making them attractive targets for malicious shilling attacks that inject fake
user profiles to manipulate recommendations. Existing shilling methods can
generate effective and stealthy fake profiles when training data only contain
rating matrix, but they lack comprehensive solutions for scenarios where side
features are present and utilized by the recommender. To address this gap, we
extend the Leg-UP framework by enhancing the generator architecture to
incorporate side features, enabling the generation of side-feature-aware fake
user profiles. Experiments on benchmarks show that our method achieves strong
attack performance while maintaining stealthiness.

### 9. [A Knowledge Graph-based Retrieval-Augmented Generation Framework for Algorithm Selection in the Facility Layout Problem](http://arxiv.org/pdf/2509.18054v1)

Authors: Nikhil N S, Amol Dilip Joshi, Bilal Muhammed, Soban Babu

Selecting a solution algorithm for the Facility Layout Problem (FLP), an
NP-hard optimization problem with a multiobjective trade-off, is a complex task
that requires deep expert knowledge. The performance of a given algorithm
depends on specific problem characteristics such as its scale, objectives, and
constraints. This creates a need for a data-driven recommendation method to
guide algorithm selection in automated design systems. This paper introduces a
new recommendation method to make such expertise accessible, based on a
Knowledge Graph-based Retrieval-Augmented Generation (KG RAG) framework. To
address this, a domain-specific knowledge graph is constructed from published
literature. The method then employs a multi-faceted retrieval mechanism to
gather relevant evidence from this knowledge graph using three distinct
approaches, which include a precise graph-based search, flexible vector-based
search, and high-level cluster-based search. The retrieved evidence is utilized
by a Large Language Model (LLM) to generate algorithm recommendations with
data-driven reasoning. The proposed KG-RAG method is compared against a
commercial LLM chatbot with access to the knowledge base as a table, across a
series of diverse, real-world FLP test cases. Based on recommendation accuracy
and reasoning capability, the proposed method performed significantly better
than the commercial LLM chatbot.

### 10. [OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System](http://arxiv.org/pdf/2509.18091v1)

Authors: Sunhao Dai, Jiakai Tang, Jiahua Wu, Kun Wang, Yuxuan Zhu, Bingjun Chen, Bangyang Hong, Yu Zhao, Cong Fu, Kangle Wu, Yabo Ni, Anxiang Zeng, Wenjie Wang, Xu Chen, Jun Xu, See-Kiong Ng

Despite the growing interest in replicating the scaled success of large
language models (LLMs) in industrial search and recommender systems, most
existing industrial efforts remain limited to transplanting Transformer
architectures, which bring only incremental improvements over strong Deep
Learning Recommendation Models (DLRMs). From a first principle perspective, the
breakthroughs of LLMs stem not only from their architectures but also from two
complementary mechanisms: context engineering, which enriches raw input queries
with contextual cues to better elicit model capabilities, and multi-step
reasoning, which iteratively refines model outputs through intermediate
reasoning paths. However, these two mechanisms and their potential to unlock
substantial improvements remain largely underexplored in industrial ranking
systems.
  In this paper, we propose OnePiece, a unified framework that seamlessly
integrates LLM-style context engineering and reasoning into both retrieval and
ranking models of industrial cascaded pipelines. OnePiece is built on a pure
Transformer backbone and further introduces three key innovations: (1)
structured context engineering, which augments interaction history with
preference and scenario signals and unifies them into a structured tokenized
input sequence for both retrieval and ranking; (2) block-wise latent reasoning,
which equips the model with multi-step refinement of representations and scales
reasoning bandwidth via block size; (3) progressive multi-task training, which
leverages user feedback chains to effectively supervise reasoning steps during
training. OnePiece has been deployed in the main personalized search scenario
of Shopee and achieves consistent online gains across different key business
metrics, including over $+2\%$ GMV/UU and a $+2.90\%$ increase in advertising
revenue.

### Machine Learning

### 1. [GraphWeave: Interpretable and Robust Graph Generation via Random Walk Trajectories](http://arxiv.org/pdf/2509.17291v1)

Authors: Rahul Nandakumar, Deepayan Chakrabarti

Given a set of graphs from some unknown family, we want to generate new
graphs from that family. Recent methods use diffusion on either graph
embeddings or the discrete space of nodes and edges. However, simple changes to
embeddings (say, adding noise) can mean uninterpretable changes in the graph.
In discrete-space diffusion, each step may add or remove many nodes/edges. It
is hard to predict what graph patterns we will observe after many diffusion
steps. Our proposed method, called GraphWeave, takes a different approach. We
separate pattern generation and graph construction. To find patterns in the
training graphs, we see how they transform vectors during random walks. We then
generate new graphs in two steps. First, we generate realistic random walk
"trajectories" which match the learned patterns. Then, we find the optimal
graph that fits these trajectories. The optimization infers all edges jointly,
which improves robustness to errors. On four simulated and five real-world
benchmark datasets, GraphWeave outperforms existing methods. The most
significant differences are on large-scale graph structures such as PageRank,
cuts, communities, degree distributions, and flows. GraphWeave is also 10x
faster than its closest competitor. Finally, GraphWeave is simple, needing only
a transformer and standard optimizers.

### 2. [SPRINT: Stochastic Performative Prediction With Variance Reduction](http://arxiv.org/pdf/2509.17304v1)

Authors: Tian Xie, Ding Zhu, Jia Liu, Mahdi Khalili, Xueru Zhang

Performative prediction (PP) is an algorithmic framework for optimizing
machine learning (ML) models where the model's deployment affects the
distribution of the data it is trained on. Compared to traditional ML with
fixed data, designing algorithms in PP converging to a stable point -- known as
a stationary performative stable (SPS) solution -- is more challenging than the
counterpart in conventional ML tasks due to the model-induced distribution
shifts. While considerable efforts have been made to find SPS solutions using
methods such as repeated gradient descent (RGD) and greedy stochastic gradient
descent (SGD-GD), most prior studies assumed a strongly convex loss until a
recent work established $\mathcal{O}(1/\sqrt{T})$ convergence of SGD-GD to SPS
solutions under smooth, non-convex losses. However, this latest progress is
still based on the restricted bounded variance assumption in stochastic
gradient estimates and yields convergence bounds with a non-vanishing error
neighborhood that scales with the variance. This limitation motivates us to
improve convergence rates and reduce error in stochastic optimization for PP,
particularly in non-convex settings. Thus, we propose a new algorithm called
stochastic performative prediction with variance reduction (SPRINT) and
establish its convergence to an SPS solution at a rate of $\mathcal{O}(1/T)$.
Notably, the resulting error neighborhood is **independent** of the variance of
the stochastic gradients. Experiments on multiple real datasets with non-convex
models demonstrate that SPRINT outperforms SGD-GD in both convergence rate and
stability.

### 3. [Robust Anomaly Detection Under Normality Distribution Shift in Dynamic Graphs](http://arxiv.org/pdf/2509.17400v1)

Authors: Xiaoyang Xu, Xiaofeng Lin, Koh Takeuchi, Kyohei Atarashi, Hisashi Kashima

Anomaly detection in dynamic graphs is a critical task with broad real-world
applications, including social networks, e-commerce, and cybersecurity. Most
existing methods assume that normal patterns remain stable over time; however,
this assumption often fails in practice due to the phenomenon we refer to as
normality distribution shift (NDS), where normal behaviors evolve over time.
Ignoring NDS can lead models to misclassify shifted normal instances as
anomalies, degrading detection performance. To tackle this issue, we propose
WhENDS, a novel unsupervised anomaly detection method that aligns normal edge
embeddings across time by estimating distributional statistics and applying
whitening transformations. Extensive experiments on four widely-used dynamic
graph datasets show that WhENDS consistently outperforms nine strong baselines,
achieving state-of-the-art results and underscoring the importance of
addressing NDS in dynamic graph anomaly detection.

### 4. [Efficient Sliced Wasserstein Distance Computation via Adaptive Bayesian Optimization](http://arxiv.org/pdf/2509.17405v1)

Authors: Manish Acharya, David Hyde

The sliced Wasserstein distance (SW) reduces optimal transport on
$\mathbb{R}^d$ to a sum of one-dimensional projections, and thanks to this
efficiency, it is widely used in geometry, generative modeling, and
registration tasks. Recent work shows that quasi-Monte Carlo constructions for
computing SW (QSW) yield direction sets with excellent approximation error.
This paper presents an alternate, novel approach: learning directions with
Bayesian optimization (BO), particularly in settings where SW appears inside an
optimization loop (e.g., gradient flows). We introduce a family of drop-in
selectors for projection directions: BOSW, a one-shot BO scheme on the unit
sphere; RBOSW, a periodic-refresh variant; ABOSW, an adaptive hybrid that seeds
from competitive QSW sets and performs a few lightweight BO refinements; and
ARBOSW, a restarted hybrid that periodically relearns directions during
optimization. Our BO approaches can be composed with QSW and its variants
(demonstrated by ABOSW/ARBOSW) and require no changes to downstream losses or
gradients. We provide numerical experiments where our methods achieve
state-of-the-art performance, and on the experimental suite of the original QSW
paper, we find that ABOSW and ARBOSW can achieve convergence comparable to the
best QSW variants with modest runtime overhead.

### 5. [Periodic Graph-Enhanced Multivariate Time Series Anomaly Detector](http://arxiv.org/pdf/2509.17472v1)

Authors: Jia Li, Shiyu Long, Ye Yuan

Multivariate time series (MTS) anomaly detection commonly encounters in
various domains like finance, healthcare, and industrial monitoring. However,
existing MTS anomaly detection methods are mostly defined on the static graph
structure, which fails to perform an accurate representation of complex
spatio-temporal correlations in MTS. To address this issue, this study proposes
a Periodic Graph-Enhanced Multivariate Time Series Anomaly Detector (PGMA) with
the following two-fold ideas: a) designing a periodic time-slot allocation
strategy based Fast Fourier Transform (FFT), which enables the graph structure
to reflect dynamic changes in MTS; b) utilizing graph neural network and
temporal extension convolution to accurate extract the complex spatio-temporal
correlations from the reconstructed periodic graphs. Experiments on four real
datasets from real applications demonstrate that the proposed PGMA outperforms
state-of-the-art models in MTS anomaly detection.

### 6. [Path-Weighted Integrated Gradients for Interpretable Dementia Classification](http://arxiv.org/pdf/2509.17491v1)

Authors: Firuz Kamalov, Mohmad Al Falasi, Fadi Thabtah

Integrated Gradients (IG) is a widely used attribution method in explainable
artificial intelligence (XAI). In this paper, we introduce Path-Weighted
Integrated Gradients (PWIG), a generalization of IG that incorporates a
customizable weighting function into the attribution integral. This
modification allows for targeted emphasis along different segments of the path
between a baseline and the input, enabling improved interpretability, noise
mitigation, and the detection of path-dependent feature relevance. We establish
its theoretical properties and illustrate its utility through experiments on a
dementia classification task using the OASIS-1 MRI dataset. Attribution maps
generated by PWIG highlight clinically meaningful brain regions associated with
various stages of dementia, providing users with sharp and stable explanations.
The results suggest that PWIG offers a flexible and theoretically grounded
approach for enhancing attribution quality in complex predictive models.

### 7. [Achilles' Heel of Mamba: Essential difficulties of the Mamba architecture demonstrated by synthetic data](http://arxiv.org/pdf/2509.17514v1)

Authors: Tianyi Chen, Pengxiao Lin, Zhiwei Wang, Zhi-Qin John Xu

State Space Models (SSMs) have emerged as promising alternatives to attention
mechanisms, with the Mamba architecture demonstrating impressive performance
and linear complexity for processing long sequences. However, the fundamental
differences between Mamba and Transformer architectures remain incompletely
understood. In this work, we use carefully designed synthetic tasks to reveal
Mamba's inherent limitations. Through experiments, we identify that Mamba's
nonlinear convolution introduces an asymmetry bias that significantly impairs
its ability to recognize symmetrical patterns and relationships. Using
composite function and inverse sequence matching tasks, we demonstrate that
Mamba strongly favors compositional solutions over symmetrical ones and
struggles with tasks requiring the matching of reversed sequences. We show
these limitations stem not from the SSM module itself but from the nonlinear
convolution preceding it, which fuses token information asymmetrically. These
insights provide a new understanding of Mamba's constraints and suggest
concrete architectural improvements for future sequence models.

### 8. [An Unlearning Framework for Continual Learning](http://arxiv.org/pdf/2509.17530v1)

Authors: Sayanta Adhikari, Vishnuprasadh Kumaravelu, P. K. Srijith

Growing concerns surrounding AI safety and data privacy have driven the
development of Machine Unlearning as a potential solution. However, current
machine unlearning algorithms are designed to complement the offline training
paradigm. The emergence of the Continual Learning (CL) paradigm promises
incremental model updates, enabling models to learn new tasks sequentially.
Naturally, some of those tasks may need to be unlearned to address safety or
privacy concerns that might arise. We find that applying conventional
unlearning algorithms in continual learning environments creates two critical
problems: performance degradation on retained tasks and task relapse, where
previously unlearned tasks resurface during subsequent learning. Furthermore,
most unlearning algorithms require data to operate, which conflicts with CL's
philosophy of discarding past data. A clear need arises for unlearning
algorithms that are data-free and mindful of future learning. To that end, we
propose UnCLe, an Unlearning framework for Continual Learning. UnCLe employs a
hypernetwork that learns to generate task-specific network parameters, using
task embeddings. Tasks are unlearned by aligning the corresponding generated
network parameters with noise, without requiring any data. Empirical
evaluations on several vision data sets demonstrate UnCLe's ability to
sequentially perform multiple learning and unlearning operations with minimal
disruption to previously acquired knowledge.

### 9. [Fast, Accurate and Interpretable Graph Classification with Topological Kernels](http://arxiv.org/pdf/2509.17693v1)

Authors: Adam Wesołowski, Ronin Wu, Karim Essafi

We introduce a novel class of explicit feature maps based on topological
indices that represent each graph by a compact feature vector, enabling fast
and interpretable graph classification. Using radial basis function kernels on
these compact vectors, we define a measure of similarity between graphs. We
perform evaluation on standard molecular datasets and observe that
classification accuracies based on single topological-index feature vectors
underperform compared to state-of-the-art substructure-based kernels. However,
we achieve significantly faster Gram matrix evaluation -- up to $20\times$
faster -- compared to the Weisfeiler--Lehman subtree kernel. To enhance
performance, we propose two extensions: 1) concatenating multiple topological
indices into an \emph{Extended Feature Vector} (EFV), and 2) \emph{Linear
Combination of Topological Kernels} (LCTK) by linearly combining Radial Basis
Function kernels computed on feature vectors of individual topological graph
indices. These extensions deliver up to $12\%$ percent accuracy gains across
all the molecular datasets. A complexity analysis highlights the potential for
exponential quantum speedup for some of the vector components. Our results
indicate that LCTK and EFV offer a favourable trade-off between accuracy and
efficiency, making them strong candidates for practical graph learning
applications.

### 10. [Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](http://arxiv.org/pdf/2509.17738v1)

Authors: Ting Han, Linara Adilova, Henning Petzka, Jens Kleesiek, Michael Kamp

Neural collapse, i.e., the emergence of highly symmetric, class-wise
clustered representations, is frequently observed in deep networks and is often
assumed to reflect or enable generalization. In parallel, flatness of the loss
landscape has been theoretically and empirically linked to generalization. Yet,
the causal role of either phenomenon remains unclear: Are they prerequisites
for generalization, or merely by-products of training dynamics? We disentangle
these questions using grokking, a training regime in which memorization
precedes generalization, allowing us to temporally separate generalization from
training dynamics and we find that while both neural collapse and relative
flatness emerge near the onset of generalization, only flatness consistently
predicts it. Models encouraged to collapse or prevented from collapsing
generalize equally well, whereas models regularized away from flat solutions
exhibit delayed generalization. Furthermore, we show theoretically that neural
collapse implies relative flatness under classical assumptions, explaining
their empirical co-occurrence. Our results support the view that relative
flatness is a potentially necessary and more fundamental property for
generalization, and demonstrate how grokking can serve as a powerful probe for
isolating its geometric underpinnings.

### Neural and Evolutionary Computing

### 1. [CMOS Implementation of Field Programmable Spiking Neural Network for Hardware Reservoir Computing](http://arxiv.org/pdf/2509.17355v1)

Authors: Ckristian Duran, Nanako Kimura, Zolboo Byambadorj, Tetsuya Iizuka

The increasing complexity and energy demands of large-scale neural networks,
such as Deep Neural Networks (DNNs) and Large Language Models (LLMs), challenge
their practical deployment in edge applications due to high power consumption,
area requirements, and privacy concerns. Spiking Neural Networks (SNNs),
particularly in analog implementations, offer a promising low-power alternative
but suffer from noise sensitivity and connectivity limitations. This work
presents a novel CMOS-implemented field-programmable neural network
architecture for hardware reservoir computing. We propose a Leaky
Integrate-and-Fire (LIF) neuron circuit with integrated voltage-controlled
oscillators (VCOs) and programmable weighted interconnections via an on-chip
FPGA framework, enabling arbitrary reservoir configurations. The system
demonstrates effective implementation of the FORCE algorithm learning, linear
and non-linear memory capacity benchmarks, and NARMA10 tasks, both in
simulation and actual chip measurements. The neuron design achieves compact
area utilization (around 540 NAND2-equivalent units) and low energy consumption
(21.7 pJ/pulse) without requiring ADCs for information readout, making it ideal
for system-on-chip integration of reservoir computing. This architecture paves
the way for scalable, energy-efficient neuromorphic systems capable of
performing real-time learning and inference with high configurability and
digital interfacing.

### 2. [Minimal Neuron Circuits: Bursters](http://arxiv.org/pdf/2509.17731v1)

Authors: Amr Nabil, T. Nandha Kumar, Haider Abbas F. Almurib

This work introduces a novel methodology for designing biologically plausible
bursting neuron circuits using a minimal number of components. We hypothesize
that to design circuits capable of bursting, the neuron circuit design must
mimic a neuron model that inherently exhibits bursting dynamics. Consequently,
classical models such as the Hodgkin-Huxley, $I_{Na,p}+I_{K}$, and
FitzHugh-Nagumo models are not suitable choices. Instead, we propose a
methodology for designing neuron circuits that emulate the qualitative
characteristics of the $I_{Na,p}+I_{K}+I_{K(M)}$ model, a well-established
minimal bursting neuron model. Based on this methodology, we present two novel
MOSFET-based circuits that exhibit bursting. Using the method of dissection of
neural bursting, we demonstrate that the nullcline and bifurcation diagrams of
the fast subsystem in our circuits are qualitatively equivalent to those of the
$I_{Na,p}+I_{K}+I_{K(M)}$ model. Furthermore, we examine the effect of the type
of bifurcation at burst initiation and termination on the bursting
characteristics, showing that our circuits can exhibit diverse bursting
behaviours. Importantly, the main contribution of this work lies not in the
specific circuit implementation, but in the methodology proposed for
constructing bursting neuron circuits.

### Networking and Internet Architecture

### 1. [Optimizing Split Federated Learning with Unstable Client Participation](http://arxiv.org/pdf/2509.17398v1)

Authors: Wei Wei, Zheng Lin, Xihui Liu, Hongyang Du, Dusit Niyato, Xianhao Chen

To enable training of large artificial intelligence (AI) models at the
network edge, split federated learning (SFL) has emerged as a promising
approach by distributing computation between edge devices and a server.
However, while unstable network environments pose significant challenges to
SFL, prior schemes often overlook such an effect by assuming perfect client
participation, rendering them impractical for real-world scenarios. In this
work, we develop an optimization framework for SFL with unstable client
participation. We theoretically derive the first convergence upper bound for
SFL with unstable client participation by considering activation uploading
failures, gradient downloading failures, and model aggregation failures. Based
on the theoretical results, we formulate a joint optimization problem for
client sampling and model splitting to minimize the upper bound. We then
develop an efficient solution approach to solve the problem optimally.
Extensive simulations on EMNIST and CIFAR-10 demonstrate the superiority of our
proposed framework compared to existing benchmarks.

### 2. [GLo-MAPPO: A Multi-Agent Proximal Policy Optimization for Energy Efficiency in UAV-Assisted LoRa Networks](http://arxiv.org/pdf/2509.17676v1)

Authors: Abdullahi Isa Ahmed, Jamal Bentahar, El Mehdi Amhoud

Long Range (LoRa) based low-power wide area networks (LPWANs) are crucial for
enabling next-generation IoT (NG-IoT) applications in 5G/6G ecosystems due to
their long-range, low-power, and low-cost characteristics. However, achieving
high energy efficiency in such networks remains a critical challenge,
particularly in large-scale or dynamically changing environments. Traditional
terrestrial LoRa deployments often suffer from coverage gaps and
non-line-of-sight (NLoS) propagation losses, while satellite-based IoT
solutions consume excessive energy and introduce high latency, limiting their
suitability for energy-constrained and delay-sensitive applications. To address
these limitations, we propose a novel architecture using multiple unmanned
aerial vehicles (UAVs) as flying LoRa gateways to dynamically collect data from
ground-based LoRa end devices. Our approach aims to maximize the system's
weighted global energy efficiency by jointly optimizing spreading factors,
transmission powers, UAV trajectories, and end-device associations.
Additionally, we formulate this complex optimization problem as a partially
observable Markov decision process (POMDP) and propose green LoRa multi-agent
proximal policy optimization (GLo-MAPPO), a multi-agent reinforcement learning
(MARL) framework based on centralized training with decentralized execution
(CTDE). Simulation results show that GLo-MAPPO significantly outperforms
benchmark algorithms, achieving energy efficiency improvements of 71.25%,
18.56%, 67.00%, 59.73%, and 49.95% for networks with 10, 20, 30, 40, and 50
LoRa end devices, respectively.

### 3. [BiLCNet : BiLSTM-Conformer Network for Encrypted Traffic Classification with 5G SA Physical Channel Records](http://arxiv.org/pdf/2509.17495v1)

Authors: Ke Ma, Jialiang Lu, Philippe Martins

Accurate and efficient traffic classification is vital for wireless network
management, especially under encrypted payloads and dynamic application
behavior, where traditional methods such as port-based identification and deep
packet inspection (DPI) are increasingly inadequate. This work explores the
feasibility of using physical channel data collected from the air interface of
5G Standalone (SA) networks for traffic sensing. We develop a preprocessing
pipeline to transform raw channel records into structured representations with
customized feature engineering to enhance downstream classification
performance. To jointly capture temporal dependencies and both local and global
structural patterns inherent in physical channel records, we propose a novel
hybrid architecture: BiLSTM-Conformer Network (BiLCNet), which integrates the
sequential modeling capability of Bidirectional Long Short-Term Memory networks
(BiLSTM) with the spatial feature extraction strength of Conformer blocks.
Evaluated on a noise-limited 5G SA dataset, our model achieves a classification
accuracy of 93.9%, outperforming a series of conventional machine learning and
deep learning algorithms. Furthermore, we demonstrate its generalization
ability under zero-shot transfer settings, validating its robustness across
traffic categories and varying environmental conditions.

### 4. [Building Transparency in Deep Learning-Powered Network Traffic Classification: A Traffic-Explainer Framework](http://arxiv.org/pdf/2509.18007v1)

Authors: Riya Ponraj, Ram Durairajan, Yu Wang

Recent advancements in deep learning have significantly enhanced the
performance and efficiency of traffic classification in networking systems.
However, the lack of transparency in their predictions and decision-making has
made network operators reluctant to deploy DL-based solutions in production
networks. To tackle this challenge, we propose Traffic-Explainer, a
model-agnostic and input-perturbation-based traffic explanation framework. By
maximizing the mutual information between predictions on original traffic
sequences and their masked counterparts, Traffic-Explainer automatically
uncovers the most influential features driving model predictions. Extensive
experiments demonstrate that Traffic-Explainer improves upon existing
explanation methods by approximately 42%. Practically, we further apply
Traffic-Explainer to identify influential features and demonstrate its enhanced
transparency across three critical tasks: application classification, traffic
localization, and network cartography. For the first two tasks,
Traffic-Explainer identifies the most decisive bytes that drive predicted
traffic applications and locations, uncovering potential vulnerabilities and
privacy concerns. In network cartography, Traffic-Explainer identifies
submarine cables that drive the mapping of traceroute to physical path,
enabling a traceroute-informed risk analysis.

### 5. [Detection of Misreporting Attacks on Software-Defined Immersive Environments](http://arxiv.org/pdf/2509.18040v1)

Authors: Sourya Saha, Md Nurul Absur, Shima Yousefi, Saptarshi Debroy

The ability to centrally control network infrastructure using a programmable
middleware has made Software-Defined Networking (SDN) ideal for emerging
applications, such as immersive environments. However, such flexibility
introduces new vulnerabilities, such as switch misreporting led load imbalance,
which in turn make such immersive environment vulnerable to severe quality
degradation. In this paper, we present a hybrid machine learning (ML)-based
network anomaly detection framework that identifies such stealthy misreporting
by capturing temporal inconsistencies in switch-reported loads, and thereby
counter potentially catastrophic quality degradation of hosted immersive
application. The detection system combines unsupervised anomaly scoring with
supervised classification to robustly distinguish malicious behavior. Data
collected from a realistic testbed deployment under both benign and adversarial
conditions is used to train and evaluate the model. Experimental results show
that the framework achieves high recall in detecting misreporting behavior,
making it effective for early and reliable detection in SDN environments.

### Robotics

### 1. [Pose Estimation of a Cable-Driven Serpentine Manipulator Utilizing Intrinsic Dynamics via Physical Reservoir Computing](http://arxiv.org/pdf/2509.17308v1)

Authors: Kazutoshi Tanaka, Tomoya Takahashi, Masashi Hamaya

Cable-driven serpentine manipulators hold great potential in unstructured
environments, offering obstacle avoidance, multi-directional force application,
and a lightweight design. By placing all motors and sensors at the base and
employing plastic links, we can further reduce the arm's weight. To demonstrate
this concept, we developed a 9-degree-of-freedom cable-driven serpentine
manipulator with an arm length of 545 mm and a total mass of only 308 g.
However, this design introduces flexibility-induced variations, such as cable
slack, elongation, and link deformation. These variations result in
discrepancies between analytical predictions and actual link positions, making
pose estimation more challenging. To address this challenge, we propose a
physical reservoir computing based pose estimation method that exploits the
manipulator's intrinsic nonlinear dynamics as a high-dimensional reservoir.
Experimental results show a mean pose error of 4.3 mm using our method,
compared to 4.4 mm with a baseline long short-term memory network and 39.5 mm
with an analytical approach. This work provides a new direction for control and
perception strategies in lightweight cable-driven serpentine manipulators
leveraging their intrinsic dynamics.

### 2. [DyDexHandover: Human-like Bimanual Dynamic Dexterous Handover using RGB-only Perception](http://arxiv.org/pdf/2509.17350v1)

Authors: Haoran Zhou, Yangwei You, Shuaijun Wang

Dynamic in air handover is a fundamental challenge for dual-arm robots,
requiring accurate perception, precise coordination, and natural motion. Prior
methods often rely on dynamics models, strong priors, or depth sensing,
limiting generalization and naturalness. We present DyDexHandover, a novel
framework that employs multi-agent reinforcement learning to train an end to
end RGB based policy for bimanual object throwing and catching. To achieve more
human-like behavior, the throwing policy is guided by a human policy
regularization scheme, encouraging fluid and natural motion, and enhancing the
generalization capability of the policy. A dual arm simulation environment was
built in Isaac Sim for experimental evaluation. DyDexHandover achieves nearly
99 percent success on training objects and 75 percent on unseen objects, while
generating human-like throwing and catching behaviors. To our knowledge, it is
the first method to realize dual-arm in-air handover using only raw RGB
perception.

### 3. [Fast Trajectory Planner with a Reinforcement Learning-based Controller for Robotic Manipulators](http://arxiv.org/pdf/2509.17381v1)

Authors: Yongliang Wang, Hamidreza Kasaei

Generating obstacle-free trajectories for robotic manipulators in
unstructured and cluttered environments remains a significant challenge.
Existing motion planning methods often require additional computational effort
to generate the final trajectory by solving kinematic or dynamic equations.
This paper highlights the strong potential of model-free reinforcement learning
methods over model-based approaches for obstacle-free trajectory planning in
joint space. We propose a fast trajectory planning system for manipulators that
combines vision-based path planning in task space with reinforcement
learning-based obstacle avoidance in joint space. We divide the framework into
two key components. The first introduces an innovative vision-based trajectory
planner in task space, leveraging the large-scale fast segment anything (FSA)
model in conjunction with basis spline (B-spline)-optimized kinodynamic path
searching. The second component enhances the proximal policy optimization (PPO)
algorithm by integrating action ensembles (AE) and policy feedback (PF), which
greatly improve precision and stability in goal-reaching and obstacle avoidance
within the joint space. These PPO enhancements increase the algorithm's
adaptability across diverse robotic tasks, ensuring consistent execution of
commands from the first component by the manipulator, while also enhancing both
obstacle avoidance efficiency and reaching accuracy. The experimental results
demonstrate the effectiveness of PPO enhancements, as well as
simulation-to-simulation (Sim-to-Sim) and simulation-to-reality (Sim-to-Real)
transfer, in improving model robustness and planner efficiency in complex
scenarios. These enhancements allow the robot to perform obstacle avoidance and
real-time trajectory planning in obstructed environments. Project page
available at: https://sites.google.com/view/ftp4rm/home

### 4. [High-Precision and High-Efficiency Trajectory Tracking for Excavators Based on Closed-Loop Dynamics](http://arxiv.org/pdf/2509.17387v1)

Authors: Ziqing Zou, Cong Wang, Yue Hu, Xiao Liu, Bowen Xu, Rong Xiong, Changjie Fan, Yingfeng Chen, Yue Wang

The complex nonlinear dynamics of hydraulic excavators, such as time delays
and control coupling, pose significant challenges to achieving high-precision
trajectory tracking. Traditional control methods often fall short in such
applications due to their inability to effectively handle these nonlinearities,
while commonly used learning-based methods require extensive interactions with
the environment, leading to inefficiency. To address these issues, we introduce
EfficientTrack, a trajectory tracking method that integrates model-based
learning to manage nonlinear dynamics and leverages closed-loop dynamics to
improve learning efficiency, ultimately minimizing tracking errors. We validate
our method through comprehensive experiments both in simulation and on a
real-world excavator. Comparative experiments in simulation demonstrate that
our method outperforms existing learning-based approaches, achieving the
highest tracking precision and smoothness with the fewest interactions.
Real-world experiments further show that our method remains effective under
load conditions and possesses the ability for continual learning, highlighting
its practical applicability. For implementation details and source code, please
refer to https://github.com/ZiqingZou/EfficientTrack.

### 5. [3D Printable Soft Liquid Metal Sensors for Delicate Manipulation Tasks](http://arxiv.org/pdf/2509.17389v1)

Authors: Lois Liow, Jonty Milford, Emre Uygun, Andre Farinha, Vinoth Viswanathan, Josh Pinskier, David Howard

Robotics and automation are key enablers to increase throughput in ongoing
conservation efforts across various threatened ecosystems. Cataloguing,
digitisation, husbandry, and similar activities require the ability to interact
with delicate, fragile samples without damaging them. Additionally,
learning-based solutions to these tasks require the ability to safely acquire
data to train manipulation policies through, e.g., reinforcement learning. To
address these twin needs, we introduce a novel method to print free-form,
highly sensorised soft 'physical twins'. We present an automated design
workflow to create complex and customisable 3D soft sensing structures on
demand from 3D scans or models. Compared to the state of the art, our soft
liquid metal sensors faithfully recreate complex natural geometries and display
excellent sensing properties suitable for validating performance in delicate
manipulation tasks. We demonstrate the application of our physical twins as
'sensing corals': high-fidelity, 3D printed replicas of scanned corals that
eliminate the need for live coral experimentation, whilst increasing data
quality, offering an ethical and scalable pathway for advancing autonomous
coral handling and soft manipulation broadly. Through extensive bench-top
manipulation and underwater grasping experiments, we show that our sensing
coral is able to detect grasps under 0.5 N, effectively capturing the delicate
interactions and light contact forces required for coral handling. Finally, we
showcase the value of our physical twins across two demonstrations: (i)
automated coral labelling for lab identification and (ii) robotic coral
aquaculture. Sensing physical twins such as ours can provide richer grasping
feedback than conventional sensors providing experimental validation of prior
to deployment in handling fragile and delicate items.

### 6. [FGGS-LiDAR: Ultra-Fast, GPU-Accelerated Simulation from General 3DGS Models to LiDAR](http://arxiv.org/pdf/2509.17390v1)

Authors: Junzhe Wu, Yufei Jia, Yiyi Yan, Zhixing Chen, Tiao Tan, Zifan Wang, Guangyu Wang

While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic
rendering, its vast ecosystem of assets remains incompatible with
high-performance LiDAR simulation, a critical tool for robotics and autonomous
driving. We present \textbf{FGGS-LiDAR}, a framework that bridges this gap with
a truly plug-and-play approach. Our method converts \textit{any} pretrained
3DGS model into a high-fidelity, watertight mesh without requiring
LiDAR-specific supervision or architectural alterations. This conversion is
achieved through a general pipeline of volumetric discretization and Truncated
Signed Distance Field (TSDF) extraction. We pair this with a highly optimized,
GPU-accelerated ray-casting module that simulates LiDAR returns at over 500
FPS. We validate our approach on indoor and outdoor scenes, demonstrating
exceptional geometric fidelity; By enabling the direct reuse of 3DGS assets for
geometrically accurate depth sensing, our framework extends their utility
beyond visualization and unlocks new capabilities for scalable, multimodal
simulation. Our open-source implementation is available at
https://github.com/TATP-233/FGGS-LiDAR.

### 7. [Learning Dexterous Manipulation with Quantized Hand State](http://arxiv.org/pdf/2509.17450v1)

Authors: Ying Feng, Hongjie Fang, Yinong He, Jingjing Chen, Chenxi Wang, Zihao He, Ruonan Liu, Cewu Lu

Dexterous robotic hands enable robots to perform complex manipulations that
require fine-grained control and adaptability. Achieving such manipulation is
challenging because the high degrees of freedom tightly couple hand and arm
motions, making learning and control difficult. Successful dexterous
manipulation relies not only on precise hand motions, but also on accurate
spatial positioning of the arm and coordinated arm-hand dynamics. However, most
existing visuomotor policies represent arm and hand actions in a single
combined space, which often causes high-dimensional hand actions to dominate
the coupled action space and compromise arm control. To address this, we
propose DQ-RISE, which quantizes hand states to simplify hand motion prediction
while preserving essential patterns, and applies a continuous relaxation that
allows arm actions to diffuse jointly with these compact hand states. This
design enables the policy to learn arm-hand coordination from data while
preventing hand actions from overwhelming the action space. Experiments show
that DQ-RISE achieves more balanced and efficient learning, paving the way
toward structured and generalizable dexterous manipulation. Project website:
http://rise-policy.github.io/DQ-RISE/

### 8. [GeCCo - a Generalist Contact-Conditioned Policy for Loco-Manipulation Skills on Legged Robots](http://arxiv.org/pdf/2509.17582v1)

Authors: Vassil Atanassov, Wanming Yu, Siddhant Gangapurwala, James Wilson, Ioannis Havoutis

Most modern approaches to quadruped locomotion focus on using Deep
Reinforcement Learning (DRL) to learn policies from scratch, in an end-to-end
manner. Such methods often fail to scale, as every new problem or application
requires time-consuming and iterative reward definition and tuning. We present
Generalist Contact-Conditioned Policy (GeCCo) -- a low-level policy trained
with Deep Reinforcement Learning that is capable of tracking arbitrary contact
points on a quadruped robot. The strength of our approach is that it provides a
general and modular low-level controller that can be reused for a wider range
of high-level tasks, without the need to re-train new controllers from scratch.
We demonstrate the scalability and robustness of our method by evaluating on a
wide range of locomotion and manipulation tasks in a common framework and under
a single generalist policy. These include a variety of gaits, traversing
complex terrains (eg. stairs and slopes) as well as previously unseen
stepping-stones and narrow beams, and interacting with objects (eg. pushing
buttons, tracking trajectories). Our framework acquires new behaviors more
efficiently, simply by combining a task-specific high-level contact planner and
the pre-trained generalist policy. A supplementary video can be found at
https://youtu.be/o8Dd44MkG2E.

### 9. [Robust and Resilient Soft Robotic Object Insertion with Compliance-Enabled Contact Formation and Failure Recovery](http://arxiv.org/pdf/2509.17666v1)

Authors: Mimo Shirasaka, Cristian C. Beltran-Hernandez, Masashi Hamaya, Yoshitaka Ushiku

Object insertion tasks are prone to failures under pose uncertainties and
environmental variations, traditionally requiring manual finetuning or
controller retraining. We present a novel approach for robust and resilient
object insertion using a passively compliant soft wrist that enables safe
contact absorption through large deformations, without high-frequency control
or force sensing. Our method structures insertion as compliance-enabled contact
formations, sequential contact states that progressively constrain degrees of
freedom, and integrates automated failure recovery strategies. Our key insight
is that wrist compliance permits safe, repeated recovery attempts; hence, we
refer to it as compliance-enabled failure recovery. We employ a pre-trained
vision-language model (VLM) that assesses each skill execution from terminal
poses and images, identifies failure modes, and proposes recovery actions by
selecting skills and updating goals. In simulation, our method achieved an 83%
success rate, recovering from failures induced by randomized
conditions--including grasp misalignments up to 5 degrees, hole-pose errors up
to 20mm, fivefold increases in friction, and previously unseen
square/rectangular pegs--and we further validate the approach on a real robot.

### 10. [Towards Learning Boulder Excavation with Hydraulic Excavators](http://arxiv.org/pdf/2509.17683v1)

Authors: Jonas Gruetter, Lorenzo Terenzi, Pascal Egli, Marco Hutter

Construction sites frequently require removing large rocks before excavation
or grading can proceed. Human operators typically extract these boulders using
only standard digging buckets, avoiding time-consuming tool changes to
specialized grippers. This task demands manipulating irregular objects with
unknown geometries in harsh outdoor environments where dust, variable lighting,
and occlusions hinder perception. The excavator must adapt to varying soil
resistance--dragging along hard-packed surfaces or penetrating soft
ground--while coordinating multiple hydraulic joints to secure rocks using a
shovel. Current autonomous excavation focuses on continuous media (soil,
gravel) or uses specialized grippers with detailed geometric planning for
discrete objects. These approaches either cannot handle large irregular rocks
or require impractical tool changes that interrupt workflow. We train a
reinforcement learning policy in simulation using rigid-body dynamics and
analytical soil models. The policy processes sparse LiDAR points (just 20 per
rock) from vision-based segmentation and proprioceptive feedback to control
standard excavator buckets. The learned agent discovers different strategies
based on soil resistance: dragging along the surface in hard soil and
penetrating directly in soft conditions. Field tests on a 12-ton excavator
achieved 70% success across varied rocks (0.4-0.7m) and soil types, compared to
83% for human operators. This demonstrates that standard construction equipment
can learn complex manipulation despite sparse perception and challenging
outdoor conditions.

### Software Engineering

### 1. [BASFuzz: Towards Robustness Evaluation of LLM-based NLP Software via Automated Fuzz Testing](http://arxiv.org/pdf/2509.17335v1)

Authors: Mingxuan Xiao, Yan Xiao, Shunhui Ji, Jiahe Tu, Pengcheng Zhang

Fuzzing has shown great success in evaluating the robustness of intelligent
natural language processing (NLP) software. As large language model (LLM)-based
NLP software is widely deployed in critical industries, existing methods still
face two main challenges: 1 testing methods are insufficiently coupled with the
behavioral patterns of LLM-based NLP software; 2 fuzzing capability for the
testing scenario of natural language generation (NLG) generally degrades. To
address these issues, we propose BASFuzz, an efficient Fuzz testing method
tailored for LLM-based NLP software. BASFuzz targets complete test inputs
composed of prompts and examples, and uses a text consistency metric to guide
mutations of the fuzzing loop, aligning with the behavioral patterns of
LLM-based NLP software. A Beam-Annealing Search algorithm, which integrates
beam search and simulated annealing, is employed to design an efficient fuzzing
loop. In addition, information entropy-based adaptive adjustment and an elitism
strategy further enhance fuzzing capability. We evaluate BASFuzz on six
datasets in representative scenarios of NLG and natural language understanding
(NLU). Experimental results demonstrate that BASFuzz achieves a testing
effectiveness of 90.335% while reducing the average time overhead by 2,163.852
seconds compared to the current best baseline, enabling more effective
robustness evaluation prior to software deployment.

### 2. [SLICET5: Static Program Slicing using Language Models with Copy Mechanism and Constrained Decoding](http://arxiv.org/pdf/2509.17338v1)

Authors: Pengfei He, Shaowei Wang, Tse-Hsun Chen

Static program slicing is a fundamental technique in software engineering.
Traditional static slicing tools rely on parsing complete source code, which
limits their applicability to real-world scenarios where code snippets are
incomplete or unparsable. While recent research developed learning-based
approaches to predict slices, they face critical challenges: (1) Inaccurate
dependency identification, where models fail to precisely capture data and
control dependencies between code elements; and (2) Unconstrained generation,
where models produce slices with extraneous or hallucinated tokens not present
in the input, violating the structural integrity of slices. To address these
challenges, we propose \ourtool, a novel slicing framework that reformulates
static program slicing as a sequence-to-sequence task using lightweight
language models (e.g., CodeT5+). Our approach incorporates two key innovations.
First, we introduce a copy mechanism that enables the model to more accurately
capture inter-element dependencies and directly copy relevant tokens from the
input, improving both dependency reasoning and generation constraint. Second,
we design a constrained decoding process with (a) lexical constraint,
restricting outputs to input tokens only, and (b) syntactic constraint,
leveraging Tree Similarity of Edit Distance (TSED) monotonicity to detect
structurally invalid outputs and discard them. We evaluate \ourtool on CodeNet
and LeetCode datasets and show it consistently outperforms state-of-the-art
baselines, improving ExactMatch scores by up to 27\%. Furthermore, \ourtool
demonstrates strong performance on incomplete code, highlighting its robustness
and practical utility in real-world development environments.

### 3. [Prompts as Software Engineering Artifacts: A Research Agenda and Preliminary Findings](http://arxiv.org/pdf/2509.17548v1)

Authors: Hugo Villamizar, Jannik Fischbach, Alexander Korn, Andreas Vogelsang, Daniel Mendez

Developers now routinely interact with large language models (LLMs) to
support a range of software engineering (SE) tasks. This prominent role
positions prompts as potential SE artifacts that, like other artifacts, may
require systematic development, documentation, and maintenance. However, little
is known about how prompts are actually used and managed in LLM-integrated
workflows, what challenges practitioners face, and whether the benefits of
systematic prompt management outweigh the associated effort. To address this
gap, we propose a research programme that (a) characterizes current prompt
practices, challenges, and influencing factors in SE; (b) analyzes prompts as
software artifacts, examining their evolution, traceability, reuse, and the
trade-offs of systematic management; and (c) develops and empirically evaluates
evidence-based guidelines for managing prompts in LLM-integrated workflows. As
a first step, we conducted an exploratory survey with 74 software professionals
from six countries to investigate current prompt practices and challenges. The
findings reveal that prompt usage in SE is largely ad-hoc: prompts are often
refined through trial-and-error, rarely reused, and shaped more by individual
heuristics than standardized practices. These insights not only highlight the
need for more systematic approaches to prompt management but also provide the
empirical foundation for the subsequent stages of our research programme.

### 4. [From OCL to JSX: declarative constraint modeling in modern SaaS tools](http://arxiv.org/pdf/2509.17629v1)

Authors: Antonio Bucchiarone, Juri Di Rocco, Damiano Di Vincenzo, Alfonso Pierantonio

The rise of Node.js in 2010, followed by frameworks like Angular, React, and
Vue.js, has accelerated the growth of low code development platforms. These
platforms harness modern UIX paradigms, component-based architectures, and the
SaaS model to enable non-experts to build software. The widespread adoption of
single-page applications (SPAs), driven by these frameworks, has shaped
low-code tools to deliver responsive, client side experiences. In parallel,
many modeling platforms have moved to the cloud, adopting either server-centric
architectures (e.g., GSLP) or client-side intelligence via SPA frameworks,
anchoring core components in JavaScript or TypeScript. Within this context,
OCL.js, a JavaScript-based implementation of the Object Constraint Language,
offers a web aligned approach to model validation, yet faces challenges such as
partial standard coverage, limited adoption, and weak integration with modern
front-end toolchains. In this paper, we explore JSX, a declarative, functional
subset of JavaScript/TypeScript used in the React ecosystem, as an alternative
to constraint expression in SaaS-based modeling environments. Its
component-oriented structure supports inductive definitions for syntax, code
generation, and querying. Through empirical evaluation, we compare JSX-based
constraints with OCL.js across representative modeling scenarios. Results show
JSX provides broader expressiveness and better fits front-end-first
architectures, indicating a promising path for constraint specification in
modern modeling tools.

### 5. [Diagnosing Violations of State-based Specifications in iCFTL](http://arxiv.org/pdf/2509.17776v1)

Authors: Cristina Stratan, Claudio Mandrioli, Domenico Bianculli

As modern software systems grow in complexity and operate in dynamic
environments, the need for runtime analysis techniques becomes a more critical
part of the verification and validation process. Runtime verification monitors
the runtime system behaviour by checking whether an execution trace - a
sequence of recorded events - satisfies a given specification, yielding a
Boolean or quantitative verdict. However, when a specification is violated,
such a verdict is often insufficient to understand why the violation happened.
To fill this gap, diagnostics approaches aim to produce more informative
verdicts. In this paper, we address the problem of generating informative
verdicts for violated Inter-procedural Control-Flow Temporal Logic (iCFTL)
specifications that express constraints over program variable values. We
propose a diagnostic approach based on backward data-flow analysis to
statically determine the relevant statements contributing to the specification
violation. Using this analysis, we instrument the program to produce enriched
execution traces. Using the enriched execution traces, we perform the runtime
analysis and identify the statements whose execution led to the specification
violation. We implemented our approach in a prototype tool, iCFTL-Diagnostics,
and evaluated it on 112 specifications across 10 software projects. Our tool
achieves 90% precision in identifying relevant statements for 100 of the 112
specifications. It reduces the number of lines that have to be inspected for
diagnosing a violation by at least 90%. In terms of computational cost,
iCFTL-Diagnostics generates a diagnosis within 7 min, and requires no more than
25 MB of memory. The instrumentation required to support diagnostics incurs an
execution time overhead of less than 30% and a memory overhead below 20%.

### 6. [Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs](http://arxiv.org/pdf/2509.17314v1)

Authors: Juyeon Yoon, Somin Kim, Robert Feldt, Shin Yoo

Software increasingly relies on the emergent capabilities of Large Language
Models (LLMs), from natural language understanding to program analysis and
generation. Yet testing them on specific tasks remains difficult and costly:
many prompts lack ground truth, forcing reliance on human judgment, while
existing uncertainty and adequacy measures typically require full inference. A
key challenge is to assess input adequacy in a way that reflects the demands of
the task, ideally before even generating any output. We introduce CLOTHO, a
task-specific, pre-generation adequacy measure that estimates input difficulty
directly from hidden LLM states. Given a large pool of unlabelled inputs for a
specific task, CLOTHO uses a Gaussian Mixture Model (GMM) to adaptively sample
the most informative cases for human labelling. Based on this reference set the
GMM can then rank unseen inputs by their likelihood of failure. In our
empirical evaluation across eight benchmark tasks and three open-weight LLMs,
CLOTHO can predict failures with a ROC-AUC of 0.716, after labelling reference
sets that are on average only 5.4% of inputs. It does so without generating any
outputs, thereby reducing costs compared to existing uncertainty measures.
Comparison of CLOTHO and post-generation uncertainty measures shows that the
two approaches complement each other. Crucially, we show that adequacy scores
learnt from open-weight LLMs transfer effectively to proprietary models,
extending the applicability of the approach. When prioritising test inputs for
proprietary models, CLOTHO increases the average number of failing inputs from
18.7 to 42.5 out of 100, compared to random prioritisation.

### 7. [Cluster Workload Allocation: A Predictive Approach Leveraging Machine Learning Efficiency](http://arxiv.org/pdf/2509.17695v1)

Authors: Leszek Sliwko

This research investigates how Machine Learning (ML) algorithms can assist in
workload allocation strategies by detecting tasks with node affinity operators
(referred to as constraint operators), which constrain their execution to a
limited number of nodes. Using real-world Google Cluster Data (GCD) workload
traces and the AGOCS framework, the study extracts node attributes and task
constraints, then analyses them to identify suitable node-task pairings. It
focuses on tasks that can be executed on either a single node or fewer than a
thousand out of 12.5k nodes in the analysed GCD cluster. Task constraint
operators are compacted, pre-processed with one-hot encoding, and used as
features in a training dataset. Various ML classifiers, including Artificial
Neural Networks, K-Nearest Neighbours, Decision Trees, Naive Bayes, Ridge
Regression, Adaptive Boosting, and Bagging, are fine-tuned and assessed for
accuracy and F1-scores. The final ensemble voting classifier model achieved 98%
accuracy and a 1.5-1.8% misclassification rate for tasks with a single suitable
node.

### Social and Information Networks

### 1. [Limited Improvement of Connectivity in Scale-Free Networks by Increasing the Power-Law Exponent](http://arxiv.org/pdf/2509.17652v1)

Authors: Yingzhou Mou, Yukio Hayashi

It has been well-known that many real networks are scale-free (SF) but
extremely vulnerable against attacks. We investigate the robustness of
connectivity and the lengths of the shortest loops in randomized SF networks
with realistic exponents $2.0 < \gamma \leq 4.0$. We show that smaller variance
of degree distributions leads to stronger robustness and longer average length
of the shortest loops, which means the existing of large holes. These results
will provide important insights toward enhancing the robustness by changing
degree distributions.

### Systems and Control

### 1. [Methods for Multi-objective Optimization PID Controller for quadrotor UAVs](http://arxiv.org/pdf/2509.17423v1)

Authors: Andrea Vaiuso, Gabriele Immordino, Ludovica Onofri, Giuliano Coppotelli, Marcello Righi

Integrating unmanned aerial vehicles into daily use requires controllers that
ensure stable flight, efficient energy use, and reduced noise. Proportional
integral derivative controllers remain standard but are highly sensitive to
gain selection, with manual tuning often yielding suboptimal trade-offs. This
paper studies different optimization techniques for the automated tuning of
quadrotor proportional integral derivative gains under a unified simulation
that couples a blade element momentum based aerodynamic model with a fast deep
neural network surrogate, six degrees of freedom rigid body dynamics,
turbulence, and a data driven acoustic surrogate model that predicts third
octave spectra and propagates them to ground receivers. We compare three
families of gradient-free optimizers: metaheuristics, Bayesian optimization,
and deep reinforcement learning. Candidate controllers are evaluated using a
composite cost function that incorporates multiple metrics, such as noise
footprint and power consumption, simultaneously. Metaheuristics improve
performance consistently, with Grey Wolf Optimization producing optimal
results. Bayesian optimization is sample efficient but carries higher per
iteration overhead and depends on the design domain. The reinforcement learning
agents do not surpass the baseline in the current setup, suggesting the problem
formulation requires further refinement. On unseen missions the best tuned
controller maintains accurate tracking while reducing oscillations, power
demand, and acoustic emissions. These results show that noise aware
proportional integral derivative tuning through black box search can deliver
quieter and more efficient flight without hardware changes.

### 2. [A Fundamental Study for Multiobjective Optimization Problems in Nonlinear Dynamical Systems](http://arxiv.org/pdf/2509.17434v1)

Authors: Ryunosuke Numata, Toshimichi Saito

Multiobjective optimization problems are important in analysis and
application of nonlinear dynamical systems. As a first step, this paper studies
a biobjective optimization problem in a simple nonlinear switched dynamical
system: a piecewise linear system based on a boost converter with photovoltaic
input. The piecewise linearity enables us to analyze the nonlinear dynamics
exactly. In the biobjective optimization problem, the first objective evaluates
stability of circuit operation and the second objective evaluates average input
power. A main task is analysis of a trade-off between the two objectives. Using
the piecewise exact solutions, the two objectives are formulated theoretically.
Using the theoretical formulae, the existence of a trade-off between the two
objectives is clarified exactly. Relationship between the trade-off and
parameters is also considered. The results provide fundamental information to
analyze multiobjective optimization problems in various nonlinear systems and
to realize their engineering applications.

### 3. [Coordinated Battery Electric Vehicle Charging Scheduling across Multiple Charging Stations](http://arxiv.org/pdf/2509.17607v1)

Authors: Saman Mehrnia, Hui Song, Nameer Al Khafaf, Mahdi Jalili, Lasantha Meegahapola, Brendan McGrath

The uptake of battery electric vehicles (BEVs) is increasing to reduce
greenhouse gas emissions in the transport sector. The rapid adoption of BEVs
depends significantly on the coordinated charging/discharging infrastructure.
Without it, uncontrolled and erratic charging patterns could lead to increased
power losses and voltage fluctuations beyond acceptable thresholds. BEV charge
scheduling presents a multi-objective optimization (MOO) challenge, demanding a
balance between minimizing network impact and maximizing the benefits for
electric vehicle charging station (EVCS) operators and BEV owners. In this
paper, we develop an MOO framework incorporating a carbon emission program and
a dynamic economic dispatch problem, allowing BEV users to respond by charging
and discharging through grid-to-vehicle (G2V) and vehicle-to-grid (V2G)
technologies according to the optimal electricity price and compensation.
Furthermore, we integrate dynamic economic dispatch with time-of-use tariffs to
obtain optimal market electricity prices and reduce total costs over 24 hours.
Our experimental results on a sample network show that the proposed scheduling
increases participation in V2G services by over 10%, increases EVCS benefits by
over 20%, and reduces network losses. Furthermore, increased rates of
charging/discharging, coupled with more significant carbon revenue benefits for
BEV users and EVCS, contribute to better offsetting battery degradation costs.

### 4. [On continuous-time sparse identification of nonlinear polynomial systems](http://arxiv.org/pdf/2509.17635v1)

Authors: Mazen Alamir

This paper leverages recent advances in high derivatives reconstruction from
noisy-time series and sparse multivariate polynomial identification in order to
improve the process of parsimoniously identifying, from a small amount of data,
unknown Single-Input/Single-Output nonlinear dynamics of relative degree up to
4. The methodology is illustrated on the Electronic Throttle Controlled
automotive system.

### 5. [Holistic Grid-Forming Control for HVDC-Connected Offshore Wind Power Plants to Provide Frequency Response](http://arxiv.org/pdf/2509.17672v1)

Authors: Zhenghua Xu, Dominic Gross, George Alin Raducu, Hesam Khazraj, Nicolaos A. Cutululis

HVDC-connected offshore wind power plants (OWPPs) are expected to provide
inertial response and frequency containment reserve (FCR) to help address the
frequency control challenges caused by the growing penetration of power
electronics in power systems. Initially dominated by communication-based and
grid-following (GFL) control, recent efforts have shifted towards incorporating
communication-free and grid-forming (GFM) control into HVDC-OWPP systems to
enhance their frequency response capability. This paper proposes a holistic GFM
control based on dual-port GFM control to improve the coordination across the
entire AC-DC-AC dynamics. A frequency response model of a typical HVDC-OWPP
system is developed for GFM control design. Then, dual-port GFM control and
virtual synchronous generator control are implemented respectively on the HVDC
system and OWPP of the typical system, where the asynchronism of onshore and
offshore frequencies is revealed. Next, holistic GFM control is proposed to
improve the synchronization and DC voltage regulation. Finally, simulations on
the delivery of FCR and inertial response are carried out to verify the
feasibility and effectiveness of the proposed control.

### 6. [RSU-Assisted Resource Allocation for Collaborative Perception](http://arxiv.org/pdf/2509.17691v1)

Authors: Guowei Liu, Le Liang, Chongtao Guo, Hao Ye, Shi Jin

As a pivotal technology for autonomous driving, collaborative perception
enables vehicular agents to exchange perceptual data through
vehicle-to-everything (V2X) communications, thereby enhancing perception
accuracy of all collaborators. However, existing collaborative perception
frameworks often assume ample communication resources, which is usually
impractical in real-world vehicular networks. To address this challenge, this
paper investigates the problem of communication resource allocation for
collaborative perception and proposes RACooper, a novel RSU-assisted resource
allocation framework that maximizes perception accuracy under constrained
communication resources. RACooper leverages a hierarchical reinforcement
learning model to dynamically allocate communication resources while accounting
for real-time sensing data and channel dynamics induced by vehicular mobility.
By jointly optimizing spatial confidence metrics and channel state information,
our approach ensures efficient feature transmission, enhancing the
effectiveness of collaborative perception. Simulation results demonstrate that
compared to conventional baseline algorithms, RACooper achieves significant
improvements in perception accuracy, especially under bandwidth-constrained
scenarios.

### 7. [Existence and Synthesis of Multi-Resolution Approximate Bisimulations for Continuous-State Dynamical Systems](http://arxiv.org/pdf/2509.17739v1)

Authors: Rudi Coppola, Yannik Schnitzer, Mirco Giacobbe, Alessandro Abate, Manuel Mazo Jr

We present a fully automatic framework for synthesising compact, finite-state
deterministic abstractions of deterministic, continuous-state autonomous
systems under locally specified resolution requirements.
  Our approach builds on multi-resolution approximate bisimulations, a
generalisation of classical $\epsilon$-approximate bisimulations, that support
state-dependent error bounds and subsumes both variable- and uniform-resolution
relations. We show that some systems admit multi-resolution bisimulations but
no $\epsilon$-approximate bisimulation.
  We prove the existence of multi-resolution approximately bisimilar
abstractions for all incrementally uniformly bounded ($\delta$-UB) systems,
thereby broadening the applicability of symbolic verification to a larger class
of dynamics; as a trivial special case, this result also covers incrementally
globally asymptotically stable ($\delta$-GAS) systems.
  The Multi-resolution Abstraction Synthesis Problem (MRASP) is solved via a
scalable Counterexample-Guided Inductive Synthesis (CEGIS) loop, combining mesh
refinement with counterexample-driven refinement. This ensures soundness for
all $\delta$-UB systems, and ensures termination in certain special cases.
  Experiments on linear and nonlinear benchmarks, including non-$\delta$-GAS
and non-differentiable cases, demonstrate that our algorithm yields
abstractions up to 50\% smaller than Lyapunov-based grids while enforcing
tighter, location-dependent error guarantees.

### 8. [On Fast Attitude Filtering Based on Matrix Fisher Distribution with Stability Guarantee](http://arxiv.org/pdf/2509.17827v1)

Authors: Shijie Wang, Haichao Gui, Rui Zhong

This paper addresses two interrelated problems of the nonlinear filtering
mechanism and fast attitude filtering with the matrix Fisher distribution (MFD)
on the special orthogonal group. By analyzing the distribution evolution along
Bayes' rule, we reveal two essential properties that enhance the performance of
Bayesian attitude filters with MFDs, particularly in challenging conditions,
from a theoretical viewpoint.
  Benefiting from the new understanding of the filtering mechanism associated
with MFDs, two closed-form filters with MFDs is then proposed. These filters
avoid the burdensome computations in previous MFD-based filters by introducing
linearized error systems with right-invariant errors but retaining the two
advantageous properties. Moreover, we leverage the two properties and
closed-form filtering iteration to prove the almost-global exponential
stability of the proposed filter with right-invariant error for the single-axis
rotation, which, to our knowledge, is not achieved by existing directional
statistics-based filters. Numerical simulations demonstrate that the proposed
filters are significantly more accurate than the classic invariant Kalman
filter. Besides, they are also as accurate as recent MFD-based Bayesian filters
in challenging circumstances with large initial error and measurement
uncertainty but consumes far less computation time (about 1/5 to 1/100 of
previous MFD-based attitude filters).

### 9. [Addressing Model Inaccuracies in Transmission Network Reconfiguration via Diverse Alternatives](http://arxiv.org/pdf/2509.17865v1)

Authors: Paul Bannmüller, Périne Cunat, Ali Rajaei, Jochen Cremer

The ongoing energy transition places significant pressure on the transmission
network due to increasing shares of renewables and electrification. To mitigate
grid congestion, transmission system operators need decision support tools to
suggest remedial actions, such as transmission network reconfigurations or
redispatch. However, these tools are prone to model inaccuracies and may not
provide relevant suggestions with regard to important unmodeled constraints or
operator preferences. We propose a human-in-the-loop modeling-to-generate
alternatives (HITL-MGA) approach to address these shortcomings by generating
diverse topology reconfiguration alternatives. Case studies on the IEEE 57-bus
and IEEE 118-bus systems show the method can leverage expert feedback and
improve the quality of the suggested remedial actions.

### 10. [Developing a Dynamic Mobility Model for Backcasting Applications: A Case Study with Shared Autonomous Vehicles](http://arxiv.org/pdf/2509.17928v1)

Authors: Théotime Héraud, Vinith Lakshmanan, Antonio Sciaretta

This study proposes the application of a backcasting approach to a mobility
model with the aim of defining an optimal decarbonization roadmap. The selected
decision variable is the introduction of a fleet of shared autonomous vehicles.
The mobility model developed is composed of six interconnected sub-models.
After presenting each of these models in detail, a method is introduced to
analyze the direct and indirect effects of the measure, and a necessary
condition for the occurrence of an undesirable effect is identified.
Simulations in both forecasting and backcasting frameworks are then conducted,
demonstrating the relevance of backcasting: it enables a 10% reduction in
operator costs compared to forecasting results, while maintaining the same
level of emissions.

### Machine Learning (Statistics Category)

### 1. [Robust Mixture Models for Algorithmic Fairness Under Latent Heterogeneity](http://arxiv.org/pdf/2509.17411v1)

Authors: Siqi Li, Molei Liu, Ziye Tian, Chuan Hong, Nan Liu

Standard machine learning models optimized for average performance often fail
on minority subgroups and lack robustness to distribution shifts. This
challenge worsens when subgroups are latent and affected by complex
interactions among continuous and discrete features. We introduce ROME (RObust
Mixture Ensemble), a framework that learns latent group structure from data
while optimizing for worst-group performance. ROME employs two approaches: an
Expectation-Maximization algorithm for linear models and a neural
Mixture-of-Experts for nonlinear settings. Through simulations and experiments
on real-world datasets, we demonstrate that ROME significantly improves
algorithmic fairness compared to standard methods while maintaining competitive
average performance. Importantly, our method requires no predefined group
labels, making it practical when sources of disparities are unknown or
evolving.

### 2. [Whitening Spherical Gaussian Mixtures in the Large-Dimensional Regime](http://arxiv.org/pdf/2509.17636v1)

Authors: Mohammed Racim Moussa Boudjemaa, Alper Kalle, Xiaoyi Mai, José Henrique de Morais Goulart, Cédric Févotte

Whitening is a classical technique in unsupervised learning that can
facilitate estimation tasks by standardizing data. An important application is
the estimation of latent variable models via the decomposition of tensors built
from high-order moments. In particular, whitening orthogonalizes the means of a
spherical Gaussian mixture model (GMM), thereby making the corresponding moment
tensor orthogonally decomposable, hence easier to decompose. However, in the
large-dimensional regime (LDR) where data are high-dimensional and scarce, the
standard whitening matrix built from the sample covariance becomes ineffective
because the latter is spectrally distorted. Consequently, whitened means of a
spherical GMM are no longer orthogonal. Using random matrix theory, we derive
exact limits for their dot products, which are generally nonzero in the LDR. As
our main contribution, we then construct a corrected whitening matrix that
restores asymptotic orthogonality, allowing for performance gains in spherical
GMM estimation.

### 3. [Synth-MIA: A Testbed for Auditing Privacy Leakage in Tabular Data Synthesis](http://arxiv.org/pdf/2509.18014v1)

Authors: Joshua Ward, Xiaofeng Lin, Chi-Hua Wang, Guang Cheng

Tabular Generative Models are often argued to preserve privacy by creating
synthetic datasets that resemble training data. However, auditing their
empirical privacy remains challenging, as commonly used similarity metrics fail
to effectively characterize privacy risk. Membership Inference Attacks (MIAs)
have recently emerged as a method for evaluating privacy leakage in synthetic
data, but their practical effectiveness is limited. Numerous attacks exist
across different threat models, each with distinct implementations targeting
various sources of privacy leakage, making them difficult to apply
consistently. Moreover, no single attack consistently outperforms the others,
leading to a routine underestimation of privacy risk.
  To address these issues, we propose a unified, model-agnostic threat
framework that deploys a collection of attacks to estimate the maximum
empirical privacy leakage in synthetic datasets. We introduce Synth-MIA, an
open-source Python library that streamlines this auditing process through a
novel testbed that integrates seamlessly into existing synthetic data
evaluation pipelines through a Scikit-Learn-like API. Our software implements
13 attack methods through a Scikit-Learn-like API, designed to enable fast
systematic estimation of privacy leakage for practitioners as well as
facilitate the development of new attacks and experiments for researchers.
  We demonstrate our framework's utility in the largest tabular synthesis
privacy benchmark to date, revealing that higher synthetic data quality
corresponds to greater privacy leakage, that similarity-based privacy metrics
show weak correlation with MIA results, and that the differentially private
generator PATEGAN can fail to preserve privacy under such attacks. This
underscores the necessity of MIA-based auditing when designing and deploying
Tabular Generative Models.

### 4. [On Quantification of Borrowing of Information in Hierarchical Bayesian Models](http://arxiv.org/pdf/2509.17301v1)

Authors: Prasenjit Ghosh, Anirban Bhattacharya, Debdeep Pati

In this work, we offer a thorough analytical investigation into the role of
shared hyperparameters in a hierarchical Bayesian model, examining their impact
on information borrowing and posterior inference. Our approach is rooted in a
non-asymptotic framework, where observations are drawn from a mixed-effects
model, and a Gaussian distribution is assumed for the true effect generator. We
consider a nested hierarchical prior distribution model to capture these
effects and use the posterior means for Bayesian estimation. To quantify the
effect of information borrowing, we propose an integrated risk measure relative
to the true data-generating distribution. Our analysis reveals that the Bayes
estimator for the model with a deeper hierarchy performs better, provided that
the unknown random effects are correlated through a compound symmetric
structure. Our work also identifies necessary and sufficient conditions for
this model to outperform the one nested within it. We further obtain sufficient
conditions when the correlation is perturbed. Our study suggests that the model
with a deeper hierarchy tends to outperform the nested model unless the true
data-generating distribution favors sufficiently independent groups. These
findings have significant implications for Bayesian modeling, and we believe
they will be of interest to researchers across a wide range of fields.

### 5. [Bilateral Distribution Compression: Reducing Both Data Size and Dimensionality](http://arxiv.org/pdf/2509.17543v1)

Authors: Dominic Broadbent, Nick Whiteley, Robert Allison, Tom Lovett

Existing distribution compression methods reduce dataset size by minimising
the Maximum Mean Discrepancy (MMD) between original and compressed sets, but
modern datasets are often large in both sample size and dimensionality. We
propose Bilateral Distribution Compression (BDC), a two-stage framework that
compresses along both axes while preserving the underlying distribution, with
overall linear time and memory complexity in dataset size and dimension.
Central to BDC is the Decoded MMD (DMMD), which quantifies the discrepancy
between the original data and a compressed set decoded from a low-dimensional
latent space. BDC proceeds by (i) learning a low-dimensional projection using
the Reconstruction MMD (RMMD), and (ii) optimising a latent compressed set with
the Encoded MMD (EMMD). We show that this procedure minimises the DMMD,
guaranteeing that the compressed set faithfully represents the original
distribution. Experiments show that across a variety of scenarios BDC can
achieve comparable or superior performance to ambient-space compression at
substantially lower cost.

### 6. [GEM-T: Generative Tabular Data via Fitting Moments](http://arxiv.org/pdf/2509.17752v1)

Authors: Miao Li, Phuc Nguyen, Christopher Tam, Alexandra Morgan, Kenneth Ge, Rahul Bansal, Linzi Yu, Rima Arnaout, Ramy Arnaout

Tabular data dominates data science but poses challenges for generative
models, especially when the data is limited or sensitive. We present a novel
approach to generating synthetic tabular data based on the principle of maximum
entropy -- MaxEnt -- called GEM-T, for ``generative entropy maximization for
tables.'' GEM-T directly captures nth-order interactions -- pairwise,
third-order, etc. -- among columns of training data. In extensive testing,
GEM-T matches or exceeds deep neural network approaches previously regarded as
state-of-the-art in 23 of 34 publicly available datasets representing diverse
subject domains (68\%). Notably, GEM-T involves orders-of-magnitude fewer
trainable parameters, demonstrating that much of the information in real-world
data resides in low-dimensional, potentially human-interpretable correlations,
provided that the input data is appropriately transformed first. Furthermore,
MaxEnt better handles heterogeneous data types (continuous vs. discrete vs.
categorical), lack of local structure, and other features of tabular data.
GEM-T represents a promising direction for light-weight high-performance
generative models for structured data.

### 7. [Robust, Online, and Adaptive Decentralized Gaussian Processes](http://arxiv.org/pdf/2509.18011v1)

Authors: Fernando Llorente, Daniel Waxman, Sanket Jantre, Nathan M. Urban, Susan E. Minkoff

Gaussian processes (GPs) offer a flexible, uncertainty-aware framework for
modeling complex signals, but scale cubically with data, assume static targets,
and are brittle to outliers, limiting their applicability in large-scale
problems with dynamic and noisy environments. Recent work introduced
decentralized random Fourier feature Gaussian processes (DRFGP), an online and
distributed algorithm that casts GPs in an information-filter form, enabling
exact sequential inference and fully distributed computation without reliance
on a fusion center. In this paper, we extend DRFGP along two key directions:
first, by introducing a robust-filtering update that downweights the impact of
atypical observations; and second, by incorporating a dynamic adaptation
mechanism that adapts to time-varying functions. The resulting algorithm
retains the recursive information-filter structure while enhancing stability
and accuracy. We demonstrate its effectiveness on a large-scale Earth system
application, underscoring its potential for in-situ modeling.

### 8. [Fréchet Geodesic Boosting](http://arxiv.org/pdf/2509.18013v1)

Authors: Yidong Zhou, Su I Iao, Hans-Georg Müller

Gradient boosting has become a cornerstone of machine learning, enabling base
learners such as decision trees to achieve exceptional predictive performance.
While existing algorithms primarily handle scalar or Euclidean outputs,
increasingly prevalent complex-structured data, such as distributions,
networks, and manifold-valued outputs, present challenges for traditional
methods. Such non-Euclidean data lack algebraic structures such as addition,
subtraction, or scalar multiplication required by standard gradient boosting
frameworks. To address these challenges, we introduce Fr\'echet geodesic
boosting (FGBoost), a novel approach tailored for outputs residing in geodesic
metric spaces. FGBoost leverages geodesics as proxies for residuals and
constructs ensembles in a way that respects the intrinsic geometry of the
output space. Through theoretical analysis, extensive simulations, and
real-world applications, we demonstrate the strong performance and adaptability
of FGBoost, showcasing its potential for modeling complex data.

### 9. [Core-elements Subsampling for Alternating Least Squares](http://arxiv.org/pdf/2509.18024v1)

Authors: Dunyao Xue, Mengyu Li, Cheng Meng, Jingyi Zhang

In this paper, we propose a novel element-wise subset selection method for
the alternating least squares (ALS) algorithm, focusing on low-rank matrix
factorization involving matrices with missing values, as commonly encountered
in recommender systems. While ALS is widely used for providing personalized
recommendations based on user-item interaction data, its high computational
cost, stemming from repeated regression operations, poses significant
challenges for large-scale datasets. To enhance the efficiency of ALS, we
propose a core-elements subsampling method that selects a representative subset
of data and leverages sparse matrix operations to approximate ALS estimations
efficiently. We establish theoretical guarantees for the approximation and
convergence of the proposed approach, showing that it achieves similar accuracy
with significantly reduced computational time compared to full-data ALS.
Extensive simulations and real-world applications demonstrate the effectiveness
of our method in various scenarios, emphasizing its potential in large-scale
recommendation systems.

### 10. [Kernel K-means clustering of distributional data](http://arxiv.org/pdf/2509.18037v1)

Authors: Amparo Baíllo, Jose R. Berrendero, Martín Sánchez-Signorini

We consider the problem of clustering a sample of probability distributions
from a random distribution on $\mathbb R^p$. Our proposed partitioning method
makes use of a symmetric, positive-definite kernel $k$ and its associated
reproducing kernel Hilbert space (RKHS) $\mathcal H$. By mapping each
distribution to its corresponding kernel mean embedding in $\mathcal H$, we
obtain a sample in this RKHS where we carry out the $K$-means clustering
procedure, which provides an unsupervised classification of the original
sample. The procedure is simple and computationally feasible even for dimension
$p>1$. The simulation studies provide insight into the choice of the kernel and
its tuning parameter. The performance of the proposed clustering procedure is
illustrated on a collection of Synthetic Aperture Radar (SAR) images.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-09-23 PST.

### 1. [Towards reproducible robotics research](https://www.nature.com/articles/s42256-025-01114-7)

Authors: Fabio Bonsignorio et al.

### 2. [A novel kangaroo escape optimizer for parameter estimation of solar photovoltaic cells/modules via one, two and three-diode equivalent circuit modeling](https://www.nature.com/articles/s41598-025-19917-4)

Authors: Sulaiman Z. Almutairi et al.

