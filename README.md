# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-17 17:00:25.862158 PST.

### Artificial Intelligence

### 1. [zELO: ELO-inspired Training Method for Rerankers and Embedding Models](http://arxiv.org/pdf/2509.12541v1)

Authors: Nicholas Pipitone, Ghita Houir Alami, Advaith Avadhanam, Anton Kaminskyi, Ashley Khoo

We introduce a novel training methodology named zELO, which optimizes
retrieval performance via the analysis that ranking tasks are statically
equivalent to a Thurstone model. Based on the zELO method, we use unsupervised
data in order train a suite of state-of-the-art open-weight reranker models:
zerank-1 and zerank-1-small. These models achieve the highest retrieval scores
in multiple domains, including finance, legal, code, and STEM, outperforming
closed-source proprietary rerankers on both NDCG@10 and Recall. These models
also demonstrate great versatility, maintaining their 0-shot performance on
out-of-domain and private customer datasets. The training data included 112,000
queries and 100 documents per query, and was trained end-to-end from
unannotated queries and documents in less than 10,000 H100-hours.

### 2. [Redefining CX with Agentic AI: Minerva CQ Case Study](http://arxiv.org/pdf/2509.12589v1)

Authors: Garima Agrawal, Riccardo De Maria, Kiran Davuluri, Daniele Spera, Charlie Read, Cosimo Spera, Jack Garrett, Don Miller

Despite advances in AI for contact centers, customer experience (CX)
continues to suffer from high average handling time (AHT), low first-call
resolution, and poor customer satisfaction (CSAT). A key driver is the
cognitive load on agents, who must navigate fragmented systems, troubleshoot
manually, and frequently place customers on hold. Existing AI-powered
agent-assist tools are often reactive driven by static rules, simple prompting,
or retrieval-augmented generation (RAG) without deeper contextual reasoning. We
introduce Agentic AI goal-driven, autonomous, tool-using systems that
proactively support agents in real time. Unlike conventional approaches,
Agentic AI identifies customer intent, triggers modular workflows, maintains
evolving context, and adapts dynamically to conversation state. This paper
presents a case study of Minerva CQ, a real-time Agent Assist product deployed
in voice-based customer support. Minerva CQ integrates real-time transcription,
intent and sentiment detection, entity recognition, contextual retrieval,
dynamic customer profiling, and partial conversational summaries enabling
proactive workflows and continuous context-building. Deployed in live
production, Minerva CQ acts as an AI co-pilot, delivering measurable
improvements in agent efficiency and customer experience across multiple
deployments.

### 3. [Analogy-Driven Financial Chain-of-Thought (AD-FCoT): A Prompting Approach for Financial Sentiment Analysis](http://arxiv.org/pdf/2509.12611v1)

Authors: Anmol Singhal Navya Singhal

Financial news sentiment analysis is crucial for anticipating market
movements. With the rise of AI techniques such as Large Language Models (LLMs),
which demonstrate strong text understanding capabilities, there has been
renewed interest in enhancing these systems. Existing methods, however, often
struggle to capture the complex economic context of news and lack transparent
reasoning, which undermines their reliability. We propose Analogy-Driven
Financial Chain-of-Thought (AD-FCoT), a prompting framework that integrates
analogical reasoning with chain-of-thought (CoT) prompting for sentiment
prediction on historical financial news. AD-FCoT guides LLMs to draw parallels
between new events and relevant historical scenarios with known outcomes,
embedding these analogies into a structured, step-by-step reasoning chain. To
our knowledge, this is among the first approaches to explicitly combine
analogical examples with CoT reasoning in finance. Operating purely through
prompting, AD-FCoT requires no additional training data or fine-tuning and
leverages the model's internal financial knowledge to generate rationales that
mirror human analytical reasoning. Experiments on thousands of news articles
show that AD-FCoT outperforms strong baselines in sentiment classification
accuracy and achieves substantially higher correlation with market returns. Its
generated explanations also align with domain expertise, providing
interpretable insights suitable for real-world financial analysis.

### 4. [GBV-SQL: Guided Generation and SQL2Text Back-Translation Validation for Multi-Agent Text2SQL](http://arxiv.org/pdf/2509.12612v1)

Authors: Daojun Chen, Xi Wang, Shenyuan Ren, Qingzhi Ma, Pengpeng Zhao, An Liu

While Large Language Models have significantly advanced Text2SQL generation,
a critical semantic gap persists where syntactically valid queries often
misinterpret user intent. To mitigate this challenge, we propose GBV-SQL, a
novel multi-agent framework that introduces Guided Generation with SQL2Text
Back-translation Validation. This mechanism uses a specialized agent to
translate the generated SQL back into natural language, which verifies its
logical alignment with the original question. Critically, our investigation
reveals that current evaluation is undermined by a systemic issue: the poor
quality of the benchmarks themselves. We introduce a formal typology for "Gold
Errors", which are pervasive flaws in the ground-truth data, and demonstrate
how they obscure true model performance. On the challenging BIRD benchmark,
GBV-SQL achieves 63.23% execution accuracy, a 5.8% absolute improvement. After
removing flawed examples, GBV-SQL achieves 96.5% (dev) and 97.6% (test)
execution accuracy on the Spider benchmark. Our work offers both a robust
framework for semantic validation and a critical perspective on benchmark
integrity, highlighting the need for more rigorous dataset curation.

### 5. [ECG-aBcDe: Overcoming Model Dependence, Encoding ECG into a Universal Language for Any LLM](http://arxiv.org/pdf/2509.12625v1)

Authors: Yong Xia, Jingxuan Li, YeTeng Sun, Jiarui Bu

Large Language Models (LLMs) hold significant promise for electrocardiogram
(ECG) analysis, yet challenges remain regarding transferability, time-scale
information learning, and interpretability. Current methods suffer from
model-specific ECG encoders, hindering transfer across LLMs. Furthermore, LLMs
struggle to capture crucial time-scale information inherent in ECGs due to
Transformer limitations. And their black-box nature limits clinical adoption.
To address these limitations, we introduce ECG-aBcDe, a novel ECG encoding
method that transforms ECG signals into a universal ECG language readily
interpretable by any LLM. By constructing a hybrid dataset of ECG language and
natural language, ECG-aBcDe enables direct fine-tuning of pre-trained LLMs
without architectural modifications, achieving "construct once, use anywhere"
capability. Moreover, the bidirectional convertibility between ECG and ECG
language of ECG-aBcDe allows for extracting attention heatmaps from ECG
signals, significantly enhancing interpretability. Finally, ECG-aBcDe
explicitly represents time-scale information, mitigating Transformer
limitations. This work presents a new paradigm for integrating ECG analysis
with LLMs. Compared with existing methods, our method achieves competitive
performance on ROUGE-L and METEOR. Notably, it delivers significant
improvements in the BLEU-4, with improvements of 2.8 times and 3.9 times in
in-dataset and cross-dataset evaluations, respectively, reaching scores of
42.58 and 30.76. These results provide strong evidence for the feasibility of
the new paradigm.

### 6. [Learn to Relax with Large Language Models: Solving Nonlinear Combinatorial Optimization Problems via Bidirectional Coevolution](http://arxiv.org/pdf/2509.12643v1)

Authors: Beidan Liu, Zhengqiu Zhu, Chen Gao, Yong Zhao, Wei Qi, Quanjun Yin

Nonlinear Combinatorial Optimization Problems (NCOPs) present a formidable
computational hurdle in practice, as their nonconvex nature gives rise to
multi-modal solution spaces that defy efficient optimization. Traditional
constraint relaxation approaches rely heavily on expert-driven, iterative
design processes that lack systematic automation and scalable adaptability.
While recent Large Language Model (LLM)-based optimization methods show promise
for autonomous problem-solving, they predominantly function as passive
constraint validators rather than proactive strategy architects, failing to
handle the sophisticated constraint interactions inherent to NCOPs.To address
these limitations, we introduce the first end-to-end \textbf{Auto}mated
\textbf{C}onstraint \textbf{O}ptimization (AutoCO) method, which revolutionizes
NCOPs resolution through learning to relax with LLMs.Specifically, we leverage
structured LLM reasoning to generate constraint relaxation strategies, which
are dynamically evolving with algorithmic principles and executable code
through a unified triple-representation scheme. We further establish a novel
bidirectional (global-local) coevolution mechanism that synergistically
integrates Evolutionary Algorithms for intensive local refinement with Monte
Carlo Tree Search for systematic global strategy space exploration, ensuring
optimal balance between intensification and diversification in fragmented
solution spaces. Finally, comprehensive experiments on three challenging NCOP
benchmarks validate AutoCO's consistent effectiveness and superior performance
over the baselines.

### 7. [H$^2$R: Hierarchical Hindsight Reflection for Multi-Task LLM Agents](http://arxiv.org/pdf/2509.12810v1)

Authors: Shicheng Ye, Chao Yu, Kaiqiang Ke, Chengdong Xu, Yinqi Wei

Large language model (LLM)-based agents have shown strong potential in
multi-task scenarios, owing to their ability to transfer knowledge across
diverse tasks. However, existing approaches often treat prior experiences and
knowledge as monolithic units, leading to inefficient and coarse-grained
knowledge transfer. In this work, we propose a novel hierarchical memory
architecture that enables fine-grained knowledge transfer by decoupling
high-level planning memory from low-level execution memory. To construct and
refine these hierarchical memories, we introduce Hierarchical Hindsight
Reflection (H$^2$R), a mechanism that distills reusable and hierarchical
knowledge from past agent-environment interactions. At test time, H$^2$R
performs retrievals of high-level and low-level memories separately, allowing
LLM-based agents to efficiently access and utilize task-relevant knowledge for
new tasks.Experimental results across two benchmarks demonstrate that H$^2$R
can improve generalization and decision-making performance, outperforming prior
baselines such as Expel.

### 8. [LTA-thinker: Latent Thought-Augmented Training Framework for Large Language Models on Complex Reasoning](http://arxiv.org/pdf/2509.12875v1)

Authors: Jiaqi Wang, Binquan Ji, Haibo Luo, Yiyang Qi, Ruiting Li, Huiyan Wang, Yuantao Han, Cangyi Yang, jiaxu Zhang, Feiliang Ren

Complex Reasoning in Large Language Models can be dynamically optimized using
Test-Time Scaling (TTS) to mitigate Overthinking. Methods such as Coconut,
SoftCoT and its variant are effective in continuous latent space inference, the
core bottleneck still lies in the efficient generation and utilization of
high-quality Latent Thought. Drawing from the theory of SoftCoT++ that a larger
variance in the generated Latent Thought distribution more closely approximates
the golden truth distribution, we propose a Latent Thought-Augmented Training
Framework--LTA-Thinker, which improves distributional variance and enhances
reasoning performance from two perspectives. First, LTA-Thinker constructs a
Latent Thought generation architecture based on a learnable prior. This
architecture aims to increase the variance distribution of generated Latent
Thought Vectors in order to simplify the overall structure and raise the
performance ceiling. Second, LTA-Thinker introduces a distribution-based
directional optimization paradigm that jointly constrains both distribution
locality and distribution scale. This mechanism improves information efficiency
and computational cost through a multi-objective co-training strategy, which
combines standard Supervised Fine-Tuning (SFT) loss with two novel losses:
Semantic Alignment Loss, which utilizes KL divergence to ensure that the Latent
Thought is highly relevant to the semantics of the question; Reasoning Focus
Loss, which utilizes a contrastive learning mechanism to guide the model to
focus on the most critical reasoning steps. Experiments show that LTA-thinker
achieves state-of-the-art (SOTA) performance among various baselines and
demonstrates a higher performance ceiling and better scaling effects.

### 9. [Stochastic Streets: A Walk Through Random LLM Address Generation in four European Cities](http://arxiv.org/pdf/2509.12914v1)

Authors: Tairan Fu, David Campo-Nazareno, Javier Coronado-Blázquez, Javier Conde, Pedro Reviriego, Fabrizio Lombardi

Large Language Models (LLMs) are capable of solving complex math problems or
answer difficult questions on almost any topic, but can they generate random
street addresses for European cities?

### 10. [Population Estimation using Deep Learning over Gandhinagar Urban Area](http://arxiv.org/pdf/2509.12926v1)

Authors: Jai Singla, Peal Jotania, Keivalya Pandya

Population estimation is crucial for various applications, from resource
allocation to urban planning. Traditional methods such as surveys and censuses
are expensive, time-consuming and also heavily dependent on human resources,
requiring significant manpower for data collection and processing. In this
study a deep learning solution is proposed to estimate population using high
resolution (0.3 m) satellite imagery, Digital Elevation Models (DEM) of 0.5m
resolution and vector boundaries. Proposed method combines Convolution Neural
Network (CNN) architecture for classification task to classify buildings as
residential and non-residential and Artificial Neural Network (ANN)
architecture to estimate the population. Approx. 48k building footprints over
Gandhinagar urban area are utilized containing both residential and
non-residential, with residential categories further used for building-level
population estimation. Experimental results on a large-scale dataset
demonstrate the effectiveness of our model, achieving an impressive overall
F1-score of 0.9936. The proposed system employs advanced geospatial analysis
with high spatial resolution to estimate Gandhinagar population at 278,954. By
integrating real-time data updates, standardized metrics, and infrastructure
planning capabilities, this automated approach addresses critical limitations
of conventional census-based methodologies. The framework provides
municipalities with a scalable and replicable tool for optimized resource
management in rapidly urbanizing cities, showcasing the efficiency of AI-driven
geospatial analytics in enhancing data-driven urban governance.

### Hardware Architecture

### 1. [HPIM: Heterogeneous Processing-In-Memory-based Accelerator for Large Language Models Inference](http://arxiv.org/pdf/2509.12993v1)

Authors: Cenlin Duan, Jianlei Yang, Rubing Yang, Yikun Wang, Yiou Wang, Lingkun Long, Yingjie Qi, Xiaolin He, Ao Zhou, Xueyan Wang, Weisheng Zhao

The deployment of large language models (LLMs) presents significant
challenges due to their enormous memory footprints, low arithmetic intensity,
and stringent latency requirements, particularly during the autoregressive
decoding stage. Traditional compute-centric accelerators, such as GPUs, suffer
from severe resource underutilization and memory bandwidth bottlenecks in these
memory-bound workloads. To overcome these fundamental limitations, we propose
HPIM, the first memory-centric heterogeneous Processing-In-Memory (PIM)
accelerator that integrates SRAM-PIM and HBM-PIM subsystems designed
specifically for LLM inference. HPIM employs a software-hardware co-design
approach that combines a specialized compiler framework with a heterogeneous
hardware architecture. It intelligently partitions workloads based on their
characteristics: latency-critical attention operations are mapped to the
SRAM-PIM subsystem to exploit its ultra-low latency and high computational
flexibility, while weight-intensive GEMV computations are assigned to the
HBM-PIM subsystem to leverage its high internal bandwidth and large storage
capacity. Furthermore, HPIM introduces a tightly coupled pipeline strategy
across SRAM-PIM and HBM-PIM subsystems to maximize intra-token parallelism,
thereby significantly mitigating serial dependency of the autoregressive
decoding stage. Comprehensive evaluations using a cycle-accurate simulator
demonstrate that HPIM significantly outperforms state-of-the-art accelerators,
achieving a peak speedup of up to 22.8x compared to the NVIDIA A100 GPU.
Moreover, HPIM exhibits superior performance over contemporary PIM-based
accelerators, highlighting its potential as a highly practical and scalable
solution for accelerating large-scale LLM inference.

### 2. [Orthrus: Dual-Loop Automated Framework for System-Technology Co-Optimization](http://arxiv.org/pdf/2509.13029v1)

Authors: Yi Ren, Baokang Peng, Chenhao Xue, Kairong Guo, Yukun Wang, Guoyao Cheng, Yibo Lin, Lining Zhang, Guangyu Sun

With the diminishing return from Moore's Law, system-technology
co-optimization (STCO) has emerged as a promising approach to sustain the
scaling trends in the VLSI industry. By bridging the gap between system
requirements and technology innovations, STCO enables customized optimizations
for application-driven system architectures. However, existing research lacks
sufficient discussion on efficient STCO methodologies, particularly in
addressing the information gap across design hierarchies and navigating the
expansive cross-layer design space. To address these challenges, this paper
presents Orthrus, a dual-loop automated framework that synergizes system-level
and technology-level optimizations. At the system level, Orthrus employs a
novel mechanism to prioritize the optimization of critical standard cells using
system-level statistics. It also guides technology-level optimization via the
normal directions of the Pareto frontier efficiently explored by Bayesian
optimization. At the technology level, Orthrus leverages system-aware insights
to optimize standard cell libraries. It employs a neural network-assisted
enhanced differential evolution algorithm to efficiently optimize technology
parameters. Experimental results on 7nm technology demonstrate that Orthrus
achieves 12.5% delay reduction at iso-power and 61.4% power savings at
iso-delay over the baseline approaches, establishing new Pareto frontiers in
STCO.

### 3. [A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption](http://arxiv.org/pdf/2509.12676v1)

Authors: Jiaao Ma, Ceyu Xu, Lisa Wu Wills

In the era of cloud computing, privacy-preserving computation offloading is
crucial for safeguarding sensitive data. Fully Homomorphic Encryption (FHE)
enables secure processing of encrypted data, but the inherent computational
complexity of FHE operations introduces significant computational overhead on
the server side. FHE schemes often face a tradeoff between efficiency and
versatility. While the CKKS scheme is highly efficient for polynomial
operations, it lacks the flexibility of the binary TFHE (Torus-FHE) scheme,
which offers greater versatility but at the cost of efficiency. The recent
multi-bit TFHE extension offers greater flexibility and performance by
supporting native non-polynomial operations and efficient integer processing.
However, current implementations of multi-bit TFHE are constrained by its
narrower numeric representation, which prevents its adoption in applications
requiring wider numeric representations.
  To address this challenge, we introduce Taurus, a hardware accelerator
designed to enhance the efficiency of multi-bit TFHE computations. Taurus
supports ciphertexts up to 10 bits by leveraging novel FFT units and optimizing
memory bandwidth through key reuse strategies. We also propose a compiler with
operation deduplication to improve memory utilization. Our experiment results
demonstrate that Taurus achieves up to 2600x speedup over a CPU, 1200x speedup
over a GPU, and up to 7x faster compared to the previous state-of-the-art TFHE
accelerator. Moreover, Taurus is the first accelerator to demonstrate
privacy-preserving inference with large language models such as GPT-2. These
advancements enable more practical and scalable applications of
privacy-preserving computation in cloud environments.

### Computational Complexity

### 1. [On the Hardness of Order Finding and Equivalence Testing for ROABPs](http://arxiv.org/pdf/2509.13238v1)

Authors: C. Ramya, Pratik Shastri

The complexity of representing a polynomial by a Read-Once Oblivious
Algebraic Branching Program (ROABP) is highly dependent on the chosen variable
ordering. Bhargava et al. prove that finding the optimal ordering is NP-hard,
and provide some evidence (based on the Small Set Expansion hypothesis) that it
is also hard to approximate the optimal ROABP width. In another work, Baraskar
et al. show that it is NP-hard to test whether a polynomial is in the
$\mathrm{GL}_n$ orbit of a polynomial of sparsity at most $s$. Building upon
these works, we show the following results: first, we prove that approximating
the minimum ROABP width up to any constant factor is NP-hard, when the input is
presented as a circuit. This removes the reliance on stronger conjectures in
the previous work. Second, we show that testing if an input polynomial given in
the sparse representation is in the affine $\mathrm{GL}_n$ orbit of a width-$w$
ROABP is NP-hard. Furthermore, we show that over fields of characteristic $0$,
the problem is NP-hard even when the input polynomial is homogeneous. This
provides the first NP-hardness results for membership testing for a dense
subclass of polynomial sized algebraic branching programs (VBP). Finally, we
locate the source of hardness for the order finding problem at the lowest
possible non-trivial degree, proving that the problem is NP-hard even for
quadratic forms.

### 2. [Deterministic polynomial factorisation modulo many primes](http://arxiv.org/pdf/2509.12705v1)

Authors: Daniel Altman

Designing a deterministic polynomial time algorithm for factoring univariate
polynomials over finite fields remains a notorious open problem. In this paper,
we present an unconditional deterministic algorithm that takes as input an
irreducible polynomial $f \in \mathbb{Z}[x]$, and computes the factorisation of
its reductions modulo $p$ for all primes $p$ up to a prescribed bound $N$. The
\emph{average running time per prime} is polynomial in the size of the input
and the degree of the splitting field of $f$ over $\mathbb{Q}$. In particular,
if $f$ is Galois, we succeed in factoring in (amortised) deterministic
polynomial time.

### 3. [An elementary proof that linking problems are hard](http://arxiv.org/pdf/2509.13120v1)

Authors: Shannon Cheng, Anna Chlopecki, Saarah Nazar, Eric Samperton

We give a new, elementary proof of what we believe is the simplest known
example of a ``natural'' problem in computational 3-dimensional topology that
is $\mathsf{NP}$-hard -- namely, the \emph{Trivial Sublink Problem}: given a
diagram $L$ of a link in $S^3$ and a positive integer $k$, decide if $L$
contains a $k$ component sublink that is trivial. This problem was previously
shown to be $\mathsf{NP}$-hard in independent works of Koenig-Tsvietkova and de
Mesmay-Rieck-Sedgwick-Tancer, both of which used reductions from
$\mathsf{3SAT}$. The reduction we describe instead starts with the Independent
Set Problem, and allows us to avoid the use of Brunnian links such as the
Borromean rings. On the technical level, this entails a new conceptual insight:
the Trivial Sublink Problem is hard entirely due to mod 2 pairwise linking,
with no need for integral or higher order linking. On the pedagogical level,
the reduction we describe is entirely elementary, and thus suitable for
introducing undergraduates and non-experts to complexity-theoretic
low-dimensional topology. To drive this point home, in this work we assume no
familiarity with low-dimensional topology, and -- other than Reidemeister's
Theorem and Karp's result that the Clique Problem is $\mathsf{NP}$-hard -- we
provide more-or-less complete definitions and proofs. We have also constructed
a web app that accompanies this work and allows a user to visualize the new
reduction interactively.

### Computational Engineering

### 1. [Impact of Geometric Uncertainty on the Computation of Abdominal Aortic Aneurysm Wall Strain](http://arxiv.org/pdf/2509.12550v1)

Authors: Saeideh Sekhavat, Mostafa Jamshidian, Adam Wittek, Karol Miller

Abdominal aortic aneurysm (AAA) is a life-threatening condition characterized
by permanent enlargement of the aorta, often detected incidentally during
imaging for unrelated conditions. Current management relies primarily on
aneurysm diameter and growth rate, which may not reliably predict
patient-specific rupture risk. Computation of AAA wall stress and strain has
the potential to improve individualized risk assessment, but these analyses
depend on image-derived geometry, which is subject to segmentation uncertainty
and lacks a definitive ground truth for the wall boundary. While the effect of
geometric uncertainty on wall stress has been studied, its influence on wall
strain remains unclear. In this study, we assessed the impact of geometric
uncertainty on AAA wall strain computed using deformable image registration of
time-resolved 3D computed tomography angiography (4D-CTA). Controlled
perturbations were applied to the wall geometry along the surface normal,
parameterized by the standard deviation for random variation and the mean for
systematic inward or outward bias, both scaled relative to wall thickness.
Results show that uncertainties in AAA wall geometry reduce the accuracy of
computed strain, with inward bias (toward the blood lumen and intraluminal
thrombus) consistently causing greater deviations than outward bias (toward
regions external to the aortic wall). Peak strain is more sensitive but less
robust, whereas the 99th percentile strain remains more stable under
perturbations. We concluded that, for sufficiently accurate strain estimation,
geometric uncertainty should remain within one wall thickness (typically 1.5
mm).

### 2. [FinSentLLM: Multi-LLM and Structured Semantic Signals for Enhanced Financial Sentiment Forecasting](http://arxiv.org/pdf/2509.12638v1)

Authors: Zijian Zhang, Rong Fu, Yangfan He, Xinze Shen, Yanlong Wang, Xiaojing Du, Haochen You, Jiazhao Shi, Simon Fong

Financial sentiment analysis (FSA) has attracted significant attention, and
recent studies increasingly explore large language models (LLMs) for this
field. Yet most work evaluates only classification metrics, leaving unclear
whether sentiment signals align with market behavior. We propose FinSentLLM, a
lightweight multi-LLM framework that integrates an expert panel of sentiment
forecasting LLMs, and structured semantic financial signals via a compact
meta-classifier. This design captures expert complementarity, semantic
reasoning signal, and agreement/divergence patterns without costly retraining,
yielding consistent 3-6% gains over strong baselines in accuracy and F1-score
on the Financial PhraseBank dataset. In addition, we also provide econometric
evidence that financial sentiment and stock markets exhibit statistically
significant long-run comovement, applying Dynamic Conditional Correlation GARCH
(DCC-GARCH) and the Johansen cointegration test to daily sentiment scores
computed from the FNSPID dataset and major stock indices. Together, these
results demonstrate that FinSentLLM delivers superior forecasting accuracy for
financial sentiment and further establish that sentiment signals are robustly
linked to long-run equity market dynamics.

### 3. [A Computational Pipeline for Patient-Specific Modeling of Thoracic Aortic Aneurysm: From Medical Image to Finite Element Analysis](http://arxiv.org/pdf/2509.12596v1)

Authors: Jiasong Chen, Linchen Qian, Ruonan Gong, Christina Sun, Tongran Qin, Thuy Pham, Caitlin Martin, Mohammad Zafar, John Elefteriades, Wei Sun, Liang Liang

The aorta is the body's largest arterial vessel, serving as the primary
pathway for oxygenated blood within the systemic circulation. Aortic aneurysms
consistently rank among the top twenty causes of mortality in the United
States. Thoracic aortic aneurysm (TAA) arises from abnormal dilation of the
thoracic aorta and remains a clinically significant disease, ranking as one of
the leading causes of death in adults. A thoracic aortic aneurysm ruptures when
the integrity of all aortic wall layers is compromised due to elevated blood
pressure. Currently, three-dimensional computed tomography (3D CT) is
considered the gold standard for diagnosing TAA. The geometric characteristics
of the aorta, which can be quantified from medical imaging, and stresses on the
aortic wall, which can be obtained by finite element analysis (FEA), are
critical in evaluating the risk of rupture and dissection. Deep learning based
image segmentation has emerged as a reliable method for extracting anatomical
regions of interest from medical images. Voxel based segmentation masks of
anatomical structures are typically converted into structured mesh
representation to enable accurate simulation. Hexahedral meshes are commonly
used in finite element simulations of the aorta due to their computational
efficiency and superior simulation accuracy. Due to anatomical variability,
patient specific modeling enables detailed assessment of individual anatomical
and biomechanics behaviors, supporting precise simulations, accurate diagnoses,
and personalized treatment strategies. Finite element (FE) simulations provide
valuable insights into the biomechanical behaviors of tissues and organs in
clinical studies. Developing accurate FE models represents a crucial initial
step in establishing a patient-specific, biomechanically based framework for
predicting the risk of TAA.

### 4. [Large Language Model Scaling Laws for Neural Quantum States in Quantum Chemistry](http://arxiv.org/pdf/2509.12679v1)

Authors: Oliver Knitter, Dan Zhao, Stefan Leichenauer, Shravan Veerapaneni

Scaling laws have been used to describe how large language model (LLM)
performance scales with model size, training data size, or amount of
computational resources. Motivated by the fact that neural quantum states (NQS)
has increasingly adopted LLM-based components, we seek to understand NQS
scaling laws, thereby shedding light on the scalability and optimal
performance--resource trade-offs of NQS ansatze. In particular, we identify
scaling laws that predict the performance, as measured by absolute error and
V-score, for transformer-based NQS as a function of problem size in
second-quantized quantum chemistry applications. By performing analogous
compute-constrained optimization of the obtained parametric curves, we find
that the relationship between model size and training time is highly dependent
on loss metric and ansatz, and does not follow the approximately linear
relationship found for language models.

### 5. [A Graph Machine Learning Approach for Detecting Topological Patterns in Transactional Graphs](http://arxiv.org/pdf/2509.12730v1)

Authors: Francesco Zola, Jon Ander Medina, Andrea Venturi, Amaia Gil, Raul Orduna

The rise of digital ecosystems has exposed the financial sector to evolving
abuse and criminal tactics that share operational knowledge and techniques both
within and across different environments (fiat-based, crypto-assets, etc.).
Traditional rule-based systems lack the adaptability needed to detect
sophisticated or coordinated criminal behaviors (patterns), highlighting the
need for strategies that analyze actors' interactions to uncover suspicious
activities and extract their modus operandi. For this reason, in this work, we
propose an approach that integrates graph machine learning and network analysis
to improve the detection of well-known topological patterns within
transactional graphs. However, a key challenge lies in the limitations of
traditional financial datasets, which often provide sparse, unlabeled
information that is difficult to use for graph-based pattern analysis.
Therefore, we firstly propose a four-step preprocessing framework that involves
(i) extracting graph structures, (ii) considering data temporality to manage
large node sets, (iii) detecting communities within, and (iv) applying
automatic labeling strategies to generate weak ground-truth labels. Then, once
the data is processed, Graph Autoencoders are implemented to distinguish among
the well-known topological patterns. Specifically, three different GAE variants
are implemented and compared in this analysis. Preliminary results show that
this pattern-focused, topology-driven method is effective for detecting complex
financial crime schemes, offering a promising alternative to conventional
rule-based detection systems.

### Computational Geometry

### 1. [Efficient Enumeration of At Most $k$-Out Polygons](http://arxiv.org/pdf/2509.12696v1)

Authors: Waseem Akram, Katsuhisa Yamanaka

Let $S$ be a set of $n$ points in the Euclidean plane and general position
i.e., no three points are collinear. An \emph{at most $k$-out polygon of $S$}
is a simple polygon such that each vertex is a point in $S$ and there are at
most $k$ points outside the polygon. In this paper, we consider the problem of
enumerating all the at most $k$-out polygon of $S$. We propose a new
enumeration algorithm for the at most $k$-out polygons of a point set. Our
algorithm enumerates all the at most $k$-out polygons in $\mathcal{O}(n^2
\log{n})$ delay, while the running time of an existing algorithm is
$\mathcal{O}(n^3 \log{n})$ delay.

### 2. [An elementary proof that linking problems are hard](http://arxiv.org/pdf/2509.13120v1)

Authors: Shannon Cheng, Anna Chlopecki, Saarah Nazar, Eric Samperton

We give a new, elementary proof of what we believe is the simplest known
example of a ``natural'' problem in computational 3-dimensional topology that
is $\mathsf{NP}$-hard -- namely, the \emph{Trivial Sublink Problem}: given a
diagram $L$ of a link in $S^3$ and a positive integer $k$, decide if $L$
contains a $k$ component sublink that is trivial. This problem was previously
shown to be $\mathsf{NP}$-hard in independent works of Koenig-Tsvietkova and de
Mesmay-Rieck-Sedgwick-Tancer, both of which used reductions from
$\mathsf{3SAT}$. The reduction we describe instead starts with the Independent
Set Problem, and allows us to avoid the use of Brunnian links such as the
Borromean rings. On the technical level, this entails a new conceptual insight:
the Trivial Sublink Problem is hard entirely due to mod 2 pairwise linking,
with no need for integral or higher order linking. On the pedagogical level,
the reduction we describe is entirely elementary, and thus suitable for
introducing undergraduates and non-experts to complexity-theoretic
low-dimensional topology. To drive this point home, in this work we assume no
familiarity with low-dimensional topology, and -- other than Reidemeister's
Theorem and Karp's result that the Clique Problem is $\mathsf{NP}$-hard -- we
provide more-or-less complete definitions and proofs. We have also constructed
a web app that accompanies this work and allows a user to visualize the new
reduction interactively.

### Computation and Language

### 1. [MAGIC-Enhanced Keyword Prompting for Zero-Shot Audio Captioning with CLIP Models](http://arxiv.org/pdf/2509.12591v1)

Authors: Vijay Govindarajan, Pratik Patel, Sahil Tripathi, Md Azizul Hoque, Gautam Siddharth Kashyap

Automated Audio Captioning (AAC) generates captions for audio clips but faces
challenges due to limited datasets compared to image captioning. To overcome
this, we propose the zero-shot AAC system that leverages pre-trained models,
eliminating the need for extensive training. Our approach uses a pre-trained
audio CLIP model to extract auditory features and generate a structured prompt,
which guides a Large Language Model (LLM) in caption generation. Unlike
traditional greedy decoding, our method refines token selection through the
audio CLIP model, ensuring alignment with the audio content. Experimental
results demonstrate a 35% improvement in NLG mean score (from 4.7 to 7.3) using
MAGIC search with the WavCaps model. The performance is heavily influenced by
the audio-text matching model and keyword selection, with optimal results
achieved using a single keyword prompt, and a 50% performance drop when no
keyword list is used.

### 2. [Mitigating Strategy Preference Bias in Emotional Support Conversation via Uncertainty Estimations](http://arxiv.org/pdf/2509.12661v1)

Authors: Yougen Zhou, Qin Chen, Ningning Zhou, Jie Zhou, Xingjiao Wu, Liang He

Emotional support conversation (ESC) aims to alleviate distress through
empathetic dialogue, yet large language models (LLMs) face persistent
challenges in delivering effective ESC due to low accuracy in strategy
planning. Moreover, there is a considerable preference bias towards specific
strategies. Prior methods using fine-tuned strategy planners have shown
potential in reducing such bias, while the underlying causes of the preference
bias in LLMs have not well been studied. To address these issues, we first
reveal the fundamental causes of the bias by identifying the knowledge
boundaries of LLMs in strategy planning. Then, we propose an approach to
mitigate the bias by reinforcement learning with a dual reward function, which
optimizes strategy planning via both accuracy and entropy-based confidence for
each region according to the knowledge boundaries. Experiments on the ESCov and
ExTES datasets with multiple LLM backbones show that our approach outperforms
the baselines, confirming the effectiveness of our approach.

### 3. [Chat-Driven Text Generation and Interaction for Person Retrieval](http://arxiv.org/pdf/2509.12662v1)

Authors: Zequn Xie, Chuxin Wang, Sihang Cai, Yeqiang Wang, Shulei Wang, Tao Jin

Text-based person search (TBPS) enables the retrieval of person images from
large-scale databases using natural language descriptions, offering critical
value in surveillance applications. However, a major challenge lies in the
labor-intensive process of obtaining high-quality textual annotations, which
limits scalability and practical deployment. To address this, we introduce two
complementary modules: Multi-Turn Text Generation (MTG) and Multi-Turn Text
Interaction (MTI). MTG generates rich pseudo-labels through simulated dialogues
with MLLMs, producing fine-grained and diverse visual descriptions without
manual supervision. MTI refines user queries at inference time through dynamic,
dialogue-based reasoning, enabling the system to interpret and resolve vague,
incomplete, or ambiguous descriptions - characteristics often seen in
real-world search scenarios. Together, MTG and MTI form a unified and
annotation-free framework that significantly improves retrieval accuracy,
robustness, and usability. Extensive evaluations demonstrate that our method
achieves competitive or superior results while eliminating the need for manual
captions, paving the way for scalable and practical deployment of TBPS systems.

### 4. [Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content](http://arxiv.org/pdf/2509.12672v1)

Authors: Shaz Furniturewala, Arkaitz Zubiaga

The volume of machine-generated content online has grown dramatically due to
the widespread use of Large Language Models (LLMs), leading to new challenges
for content moderation systems. Conventional content moderation classifiers,
which are usually trained on text produced by humans, suffer from
misclassifications due to LLM-generated text deviating from their training data
and adversarial attacks that aim to avoid detection. Present-day defence
tactics are reactive rather than proactive, since they rely on adversarial
training or external detection models to identify attacks. In this work, we aim
to identify the vulnerable components of toxicity classifiers that contribute
to misclassification, proposing a novel strategy based on mechanistic
interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa
classifiers, testing on diverse datasets spanning a variety of minority groups.
We use adversarial attacking techniques to identify vulnerable circuits.
Finally, we suppress these vulnerable circuits, improving performance against
adversarial attacks. We also provide demographic-level insights into these
vulnerable circuits, exposing fairness and robustness gaps in model training.
We find that models have distinct heads that are either crucial for performance
or vulnerable to attack and suppressing the vulnerable heads improves
performance on adversarial input. We also find that different heads are
responsible for vulnerability across different demographic groups, which can
inform more inclusive development of toxicity detection models.

### 5. [Case-Based Decision-Theoretic Decoding with Quality Memories](http://arxiv.org/pdf/2509.12677v1)

Authors: Hiroyuki Deguchi, Masaaki Nagata

Minimum Bayes risk (MBR) decoding is a decision rule of text generation,
which selects the hypothesis that maximizes the expected utility and robustly
generates higher-quality texts than maximum a posteriori (MAP) decoding.
However, it depends on sample texts drawn from the text generation model; thus,
it is difficult to find a hypothesis that correctly captures the knowledge or
information of out-of-domain. To tackle this issue, we propose case-based
decision-theoretic (CBDT) decoding, another method to estimate the expected
utility using examples of domain data. CBDT decoding not only generates
higher-quality texts than MAP decoding, but also the combination of MBR and
CBDT decoding outperformed MBR decoding in seven domain De--En and
Ja$\leftrightarrow$En translation tasks and image captioning tasks on MSCOCO
and nocaps datasets.

### 6. [HistoryBankQA: Multilingual Temporal Question Answering on Historical Events](http://arxiv.org/pdf/2509.12720v1)

Authors: Biswadip Mandal, Anant Khandelwal, Manish Gupta

Temporal reasoning about historical events is a critical skill for NLP tasks
like event extraction, historical entity linking, temporal question answering,
timeline summarization, temporal event clustering and temporal natural language
inference. Yet efforts on benchmarking temporal reasoning capabilities of large
language models (LLMs) are rather limited. Existing temporal reasoning datasets
are limited in scale, lack multilingual coverage and focus more on contemporary
events. To address these limitations, we present HistoryBank, a multilingual
database of 10M+ historical events extracted from Wikipedia timeline pages and
article infoboxes. Our database provides unprecedented coverage in both
historical depth and linguistic breadth with 10 languages. Additionally, we
construct a comprehensive question answering benchmark for temporal reasoning
across all languages. This benchmark covers a diverse set of 6 temporal QA
reasoning tasks, and we evaluate a suite of popular language models
(LLaMA-3-8B, Mistral-7B, Gemma-2-9b, Qwen3-8B, GPT4o) to assess their
performance on these tasks. As expected GPT4o performs best across all answer
types and languages; Gemma-2 outperforms the other small language models. Our
work aims to provide a comprehensive resource for advancing multilingual and
temporally-aware natural language understanding of historical events. To
facilitate further research, we will make our code and datasets publicly
available upon acceptance of this paper.

### 7. [Contrastive Learning with Enhanced Abstract Representations using Grouped Loss of Abstract Semantic Supervision](http://arxiv.org/pdf/2509.12771v1)

Authors: Omri Suissa, Muhiim Ali, Shengmai Chen, Yinuo Cai, Shekhar Pradhan

Humans can recognize an image as an instance of a general concept, beyond
simply identifying its objects and their relationships. In this paper, we
investigate 1. The extent to which VLMs have this concept abstraction capacity,
and 2. Strategies for encoding the sort of higher-concept information in images
that would enable the resulting VLM model (CLEAR GLASS model) to have this
capability to a greater degree. To this end, we introduce a grouped
image-caption dataset (MAGIC), which consists of several groups of image
captions and for each group a set of associated images and higher-level
conceptual labels. We use a novel contrastive loss technique to induce the
model to encode in the representation of each image (caption) in a group the
information that is common to all members of the image-caption group. Our main
contribution is a grouped contrastive loss function based on text-image
contrastive groups (outer contrastive loss) as well as an inner loss which
measures the distances between image-caption instances in the group. Our
training methodology results in the CLEAR GLASS model having the concept
abstraction capacity as an emergent capacity because the model is not exposed
to the higher-level concepts associated with each group. Instead, the training
forces the model to create for each image-caption group a semantic
representation that brings it closer to the semantic representation of the
higher-level concepts in the latent semantic space. Our experiments show that
this training methodology results in a model which shows improvement in
abstract concept recognition compared to SOTA models.

### 8. [ConvergeWriter: Data-Driven Bottom-Up Article Construction](http://arxiv.org/pdf/2509.12811v1)

Authors: Binquan Ji, Jiaqi Wang, Ruiting Li, Xingchen Han, Yiyang Qi, Shichao Wang, Yifei Lu, Yuantao Han, Feiliang Ren

Large Language Models (LLMs) have shown remarkable prowess in text
generation, yet producing long-form, factual documents grounded in extensive
external knowledge bases remains a significant challenge. Existing "top-down"
methods, which first generate a hypothesis or outline and then retrieve
evidence, often suffer from a disconnect between the model's plan and the
available knowledge, leading to content fragmentation and factual inaccuracies.
To address these limitations, we propose a novel "bottom-up," data-driven
framework that inverts the conventional generation pipeline. Our approach is
predicated on a "Retrieval-First for Knowledge, Clustering for Structure"
strategy, which first establishes the "knowledge boundaries" of the source
corpus before any generative planning occurs. Specifically, we perform
exhaustive iterative retrieval from the knowledge base and then employ an
unsupervised clustering algorithm to organize the retrieved documents into
distinct "knowledge clusters." These clusters form an objective, data-driven
foundation that directly guides the subsequent generation of a hierarchical
outline and the final document content. This bottom-up process ensures that the
generated text is strictly constrained by and fully traceable to the source
material, proactively adapting to the finite scope of the knowledge base and
fundamentally mitigating the risk of hallucination. Experimental results on
both 14B and 32B parameter models demonstrate that our method achieves
performance comparable to or exceeding state-of-the-art baselines, and is
expected to demonstrate unique advantages in knowledge-constrained scenarios
that demand high fidelity and structural coherence. Our work presents an
effective paradigm for generating reliable, structured, long-form documents,
paving the way for more robust LLM applications in high-stakes,
knowledge-intensive domains.

### 9. [Data Augmentation for Maltese NLP using Transliterated and Machine Translated Arabic Data](http://arxiv.org/pdf/2509.12853v1)

Authors: Kurt Micallef, Nizar Habash, Claudia Borg

Maltese is a unique Semitic language that has evolved under extensive
influence from Romance and Germanic languages, particularly Italian and
English. Despite its Semitic roots, its orthography is based on the Latin
script, creating a gap between it and its closest linguistic relatives in
Arabic. In this paper, we explore whether Arabic-language resources can support
Maltese natural language processing (NLP) through cross-lingual augmentation
techniques. We investigate multiple strategies for aligning Arabic textual data
with Maltese, including various transliteration schemes and machine translation
(MT) approaches. As part of this, we also introduce novel transliteration
systems that better represent Maltese orthography. We evaluate the impact of
these augmentations on monolingual and mutlilingual models and demonstrate that
Arabic-based augmentation can significantly benefit Maltese NLP tasks.

### 10. [Do LLMs Understand Wine Descriptors Across Cultures? A Benchmark for Cultural Adaptations of Wine Reviews](http://arxiv.org/pdf/2509.12961v1)

Authors: Chenye Zou, Xingyue Wen, Tianyi Hu, Qian Janice Wang, Daniel Hershcovich

Recent advances in large language models (LLMs) have opened the door to
culture-aware language tasks. We introduce the novel problem of adapting wine
reviews across Chinese and English, which goes beyond literal translation by
incorporating regional taste preferences and culture-specific flavor
descriptors. In a case study on cross-cultural wine review adaptation, we
compile the first parallel corpus of professional reviews, containing 8k
Chinese and 16k Anglophone reviews. We benchmark both
neural-machine-translation baselines and state-of-the-art LLMs with automatic
metrics and human evaluation. For the latter, we propose three culture-oriented
criteria -- Cultural Proximity, Cultural Neutrality, and Cultural Genuineness
-- to assess how naturally a translated review resonates with target-culture
readers. Our analysis shows that current models struggle to capture cultural
nuances, especially in translating wine descriptions across different cultures.
This highlights the challenges and limitations of translation models in
handling cultural content.

### Cryptography and Security

### 1. [Exploiting Timing Side-Channels in Quantum Circuits Simulation Via ML-Based Methods](http://arxiv.org/pdf/2509.12535v1)

Authors: Ben Dong, Hui Feng, Qian Wang

As quantum computing advances, quantum circuit simulators serve as critical
tools to bridge the current gap caused by limited quantum hardware
availability. These simulators are typically deployed on cloud platforms, where
users submit proprietary circuit designs for simulation. In this work, we
demonstrate a novel timing side-channel attack targeting cloud-based quantum
simulators. A co-located malicious process can observe fine-grained execution
timing patterns to extract sensitive information about concurrently running
quantum circuits. We systematically analyze simulator behavior using the
QASMBench benchmark suite, profiling timing and memory characteristics across
various circuit executions. Our experimental results show that timing profiles
exhibit circuit-dependent patterns that can be effectively classified using
pattern recognition techniques, enabling the adversary to infer circuit
identities and compromise user confidentiality. We were able to achieve 88% to
99.9% identification rate of quantum circuits based on different datasets. This
work highlights previously unexplored security risks in quantum simulation
environments and calls for stronger isolation mechanisms to protect user
workloads

### 2. [Hardened CTIDH: Dummy-Free and Deterministic CTIDH](http://arxiv.org/pdf/2509.12877v1)

Authors: Gustavo Banegas, Andreas Hellenbrand, Matheus Saldanha

Isogeny-based cryptography has emerged as a promising postquantum
alternative, with CSIDH and its constant-time variants CTIDH and dCTIDH
offering efficient group-action protocols. However, CTIDH and dCTIDH rely on
dummy operations in differential addition chains (DACs) and Matryoshka, which
can be exploitable by fault-injection attacks. In this work, we present the
first dummy-free implementation of dCTIDH. Our approach combines two recent
ideas: DACsHUND, which enforces equal-length DACs within each batch without
padding, and a reformulated Matryoshka structure that removes dummy
multiplications and validates all intermediate points. Our analysis shows that
small primes such as 3, 5, and 7 severely restrict feasible DACsHUND
configurations, motivating new parameter sets that exclude them. We implement
dummy-free dCTIDH-2048-194 and dCTIDH-2048-205, achieving group action costs of
roughly 357,000-362,000 Fp-multiplications, with median evaluation times of
1.59-1.60 (Gcyc). These results do not surpass dC-TIDH, but they outperform
CTIDH by roughly 5% while eliminating dummy operations entirely. Compared to
dCSIDH, our construction is more than 4x faster. To the best of our knowledge,
this is the first efficient implementation of a CSIDH-like protocol that is
simultaneously deterministic, constant-time, and fully dummy-free.

### 3. [A Fault Analysis on SNOVA](http://arxiv.org/pdf/2509.12879v1)

Authors: Gustavo Banegas, Ricardo Villanueva-Polanco

SNOVA is a post-quantum cryptographic signature scheme known for its
efficiency and compact key sizes, making it a second-round candidate in the
NIST post-quantum cryptography standardization process. This paper presents a
comprehensive fault analysis of SNOVA, focusing on both permanent and transient
faults during signature generation. We introduce several fault injection
strategies that exploit SNOVA's structure to recover partial or complete secret
keys with limited faulty signatures. Our analysis reveals that as few as 22 to
68 faulty signatures, depending on the security level, can suffice for key
recovery. We propose a novel fault-assisted reconciliation attack,
demonstrating its effectiveness in extracting the secret key space via solving
a quadratic polynomial system. Simulations show transient faults in key
signature generation steps can significantly compromise SNOVA's security. To
address these vulnerabilities, we propose a lightweight countermeasure to
reduce the success of fault attacks without adding significant overhead. Our
results highlight the importance of fault-resistant mechanisms in post-quantum
cryptographic schemes like SNOVA to ensure robustness.

### 4. [EByFTVeS: Efficient Byzantine Fault Tolerant-based Verifiable Secret-sharing in Distributed Privacy-preserving Machine Learning](http://arxiv.org/pdf/2509.12899v1)

Authors: Zhen Li, Zijian Zhang, Wenjin Yang, Pengbo Wang, Zhaoqi Wang, Meng Li, Yan Wu, Xuyang Liu, Jing Sun, Liehuang Zhu

Verifiable Secret Sharing (VSS) has been widespread in Distributed
Privacy-preserving Machine Learning (DPML), because invalid shares from
malicious dealers or participants can be recognized by verifying the commitment
of the received shares for honest participants. However, the consistency and
the computation and communitation burden of the VSS-based DPML schemes are
still two serious challenges. Although Byzantine Fault Tolerance (BFT) system
has been brought to guarantee the consistency and improve the efficiency of the
existing VSS-based DPML schemes recently, we explore an Adaptive Share Delay
Provision (ASDP) strategy, and launch an ASDP-based Customized Model Poisoning
Attack (ACuMPA) for certain participants in this paper. We theoretically
analyzed why the ASDP strategy and the ACuMPA algorithm works to the existing
schemes. Next, we propose an [E]fficient [By]zantine [F]ault [T]olerant-based
[Ve]rifiable [S]ecret-sharing (EByFTVeS) scheme. Finally, the validity,
liveness, consistency and privacy of the EByFTVeS scheme are theoretically
analyzed, while the efficiency of the EByFTVeS scheme outperforms that of
the-state-of-art VSS scheme according to comparative experiment results.

### 5. [xRWA: A Cross-Chain Framework for Interoperability of Real-World Assets](http://arxiv.org/pdf/2509.12957v1)

Authors: Yihao Guo, Haoming Zhu, Minghui Xu, Xiuzhen Cheng, Bin Xiao

Real-World Assets (RWAs) have recently attracted increasing attention as a
means of bridging traditional financial instruments with decentralized
infrastructures. By representing assets such as bonds, commodities, and real
estate on blockchains, RWAs can enhance liquidity, broaden accessibility, and
extend the scope of decentralized finance. Industry forecasts further suggest
rapid growth of tokenized RWAs in the coming years, underscoring their
potential role in the evolution of digital financial markets. However, when
deployed across multiple blockchains, RWAs face challenges such as repeated
authentication on different chains and inefficiency caused by multi-step
settlement protocols. To address these issues, we present a cross-chain
framework for RWAs that emphasizes identity management, authentication, and
interaction. The framework integrates Decentralized Identifiers and Verifiable
Credentials with customized attributes to support decentralized identification,
and incorporates an authentication protocol based on Simplified Payment
Verification to avoid redundant verification across chains. Furthermore, we
design a cross-chain channel that enables the settlement of RWAs without
requiring channel closure, thereby improving operational efficiency. We
implement the framework and evaluate it through simulations, which confirm its
feasibility and demonstrate improvements in efficiency for RWAs in cross-chain
settings.

### 6. [Universal share based quantum multi secret image sharing scheme](http://arxiv.org/pdf/2509.12979v1)

Authors: Dipak K. Rabari, Yogesh K. Meghrajani, Laxmi S. Desai

Image security for information has become increasingly critical as internet
become more prevalent due to hacking and unauthorized access. To ensure the
security of confidential image data, image encryption using visual cryptography
plays a crucial role. To share multiple images using visual cryptography, the
company organizer utilizes the concept of a universal or common share.
Likewise, quantum computing is an emerging technology that facilitates secure
communication. The ability of quantum computers to solve certain mathematical
problems efficiently threatens the security of many current encryption
algorithms. Hence, to leverage the strengths of quantum computing and visual
cryptography, this research introduces a novel universal share-based quantum
multi-secret sharing technique for secure image communication. Quantum
computing enables the scheme to exhibit high resilience to different
eavesdropping threats. Consequently, the proposed method offers robust security
solution for sharing confidential images across a range of applications,
including enterprise data access and military communications.

### 7. [Bridging Threat Models and Detections: Formal Verification via CADP](http://arxiv.org/pdf/2509.13035v1)

Authors: Dumitru-Bogdan Prelipcean, Cătălin Dima

Threat detection systems rely on rule-based logic to identify adversarial
behaviors, yet the conformance of these rules to high-level threat models is
rarely verified formally. We present a formal verification framework that
models both detection logic and attack trees as labeled transition systems
(LTSs), enabling automated conformance checking via bisimulation and weak trace
inclusion. Detection rules specified in the Generic Threat Detection Language
(GTDL, a general-purpose detection language we formalize in this work) are
assigned a compositional operational semantics, and threat models expressed as
attack trees are interpreted as LTSs through a structural trace semantics. Both
representations are translated to LNT, a modeling language supported by the
CADP toolbox. This common semantic domain enables systematic and automated
verification of detection coverage. We evaluate our approach on real-world
malware scenarios such as LokiBot and Emotet and provide scalability analysis
through parametric synthetic models. Results confirm that our methodology
identifies semantic mismatches between threat models and detection rules,
supports iterative refinement, and scales to realistic threat landscapes.

### 8. [SLasH-DSA: Breaking SLH-DSA Using an Extensible End-To-End Rowhammer Framework](http://arxiv.org/pdf/2509.13048v1)

Authors: Jeremy Boy, Antoon Purnal, Anna Pätschke, Luca Wilke, Thomas Eisenbarth

As quantum computing advances, PQC schemes are adopted to replace classical
algorithms. Among them is the SLH-DSA that was recently standardized by NIST
and is favored for its conservative security foundations.
  In this work, we present the first software-only universal forgery attack on
SLH-DSA, leveraging Rowhammer-induced bit flips to corrupt the internal state
and forge signatures. While prior work targeted embedded systems and required
physical access, our attack is software-only, targeting commodity desktop and
server hardware, significantly broadening the threat model. We demonstrate a
full end-to-end attack against all security levels of SLH-DSA in OpenSSL 3.5.1,
achieving universal forgery for the highest security level after eight hours of
hammering and 36 seconds of post-processing. Our post-processing is informed by
a novel complexity analysis that, given a concrete set of faulty signatures,
identifies the most promising computational path to pursue.
  To enable the attack, we introduce Swage, a modular and extensible framework
for implementing end-to-end Rowhammer-based fault attacks. Swage abstracts and
automates key components of practical Rowhammer attacks. Unlike prior tooling,
Swage is untangled from the attacked code, making it reusable and suitable for
frictionless analysis of different targets. Our findings highlight that even
theoretically sound PQC schemes can fail under real-world conditions,
underscoring the need for additional implementation hardening or hardware
defenses against Rowhammer.

### 9. [Digital Sovereignty Control Framework for Military AI-based Cyber Security](http://arxiv.org/pdf/2509.13072v1)

Authors: Clara Maathuis, Kasper Cools

In today's evolving threat landscape, ensuring digital sovereignty has become
mandatory for military organizations, especially given their increased
development and investment in AI-driven cyber security solutions. To this end,
a multi-angled framework is proposed in this article in order to define and
assess digital sovereign control of data and AI-based models for military cyber
security. This framework focuses on aspects such as context, autonomy,
stakeholder involvement, and mitigation of risks in this domain. Grounded on
the concepts of digital sovereignty and data sovereignty, the framework aims to
protect sensitive defence assets against threats such as unauthorized access,
ransomware, and supply-chain attacks. This approach reflects the multifaceted
nature of digital sovereignty by preserving operational autonomy, assuring
security and safety, securing privacy, and fostering ethical compliance of both
military systems and decision-makers. At the same time, the framework addresses
interoperability challenges among allied forces, strategic and legal
considerations, and the integration of emerging technologies by considering a
multidisciplinary approach that enhances the resilience and preservation of
control over (critical) digital assets. This is done by adopting a design
oriented research where systematic literature review is merged with critical
thinking and analysis of field incidents in order to assure the effectivity and
realism of the framework proposed.

### 10. [Characterizing Phishing Pages by JavaScript Capabilities](http://arxiv.org/pdf/2509.13186v1)

Authors: Aleksandr Nahapetyan, Kanv Khare, Kevin Schwarz, Bradley Reaves, Alexandros Kapravelos

In 2024, the Anti-Phishing Work Group identified over one million phishing
pages. Phishers achieve this scale by using phishing kits -- ready-to-deploy
phishing websites -- to rapidly deploy phishing campaigns with specific data
exfiltration, evasion, or mimicry techniques. In contrast, researchers and
defenders continue to fight phishing on a page-by-page basis and rely on manual
analysis to recognize static features for kit identification.
  This paper aims to aid researchers and analysts by automatically
differentiating groups of phishing pages based on the underlying kit,
automating a previously manual process, and enabling us to measure how popular
different client-side techniques are across these groups. For kit detection,
our system has an accuracy of 97% on a ground-truth dataset of 548 kit families
deployed across 4,562 phishing URLs. On an unlabeled dataset, we leverage the
complexity of 434,050 phishing pages' JavaScript logic to group them into
11,377 clusters, annotating the clusters with what phishing techniques they
employ. We find that UI interactivity and basic fingerprinting are universal
techniques, present in 90% and 80% of the clusters, respectively. On the other
hand, mouse detection via the browser's mouse API is among the rarest
behaviors, despite being used in a deployment of a 7-year-old open-source
phishing kit. Our methods and findings provide new ways for researchers and
analysts to tackle the volume of phishing pages.

### Computer Vision and Pattern Recognition

### 1. [Neural Collapse-Inspired Multi-Label Federated Learning under Label-Distribution Skew](http://arxiv.org/pdf/2509.12544v1)

Authors: Can Peng, Yuyuan Liu, Yingyu Yang, Pramit Saha, Qianye Yang, J. Alison Noble

Federated Learning (FL) enables collaborative model training across
distributed clients while preserving data privacy. However, the performance of
deep learning often deteriorates in FL due to decentralized and heterogeneous
data. This challenge is further amplified in multi-label scenarios, where data
exhibit complex characteristics such as label co-occurrence, inter-label
dependency, and discrepancies between local and global label relationships.
While most existing FL research primarily focuses on single-label
classification, many real-world applications, particularly in domains such as
medical imaging, often involve multi-label settings. In this paper, we address
this important yet underexplored scenario in FL, where clients hold multi-label
data with skewed label distributions. Neural Collapse (NC) describes a
geometric structure in the latent feature space where features of each class
collapse to their class mean with vanishing intra-class variance, and the class
means form a maximally separated configuration. Motivated by this theory, we
propose a method to align feature distributions across clients and to learn
high-quality, well-clustered representations. To make the NC-structure
applicable to multi-label settings, where image-level features may contain
multiple semantic concepts, we introduce a feature disentanglement module that
extracts semantically specific features. The clustering of these disentangled
class-wise features is guided by a predefined shared NC structure, which
mitigates potential conflicts between client models due to diverse local data
distributions. In addition, we design regularisation losses to encourage
compact clustering in the latent feature space. Experiments conducted on four
benchmark datasets across eight diverse settings demonstrate that our approach
outperforms existing methods, validating its effectiveness in this challenging
FL scenario.

### 2. [Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection](http://arxiv.org/pdf/2509.12546v1)

Authors: Yingxin Lai, Zitong Yu, Jun Wang, Linlin Shen, Yong Xu, Xiaochun Cao

Face forgery detection faces a critical challenge: a persistent gap between
offline benchmarks and real-world efficacy,which we attribute to the ecological
invalidity of training data.This work introduces Agent4FaceForgery to address
two fundamental problems: (1) how to capture the diverse intents and iterative
processes of human forgery creation, and (2) how to model the complex, often
adversarial, text-image interactions that accompany forgeries in social media.
To solve this,we propose a multi-agent framework where LLM-poweredagents,
equipped with profile and memory modules, simulate the forgery creation
process. Crucially, these agents interact in a simulated social environment to
generate samples labeled for nuanced text-image consistency, moving beyond
simple binary classification. An Adaptive Rejection Sampling (ARS) mechanism
ensures data quality and diversity. Extensive experiments validate that the
data generated by our simulationdriven approach brings significant performance
gains to detectors of multiple architectures, fully demonstrating the
effectiveness and value of our framework.

### 3. [Explicit Multimodal Graph Modeling for Human-Object Interaction Detection](http://arxiv.org/pdf/2509.12554v1)

Authors: Wenxuan Ji, Haichao Shi, Xiao-Yu zhang

Transformer-based methods have recently become the prevailing approach for
Human-Object Interaction (HOI) detection. However, the Transformer architecture
does not explicitly model the relational structures inherent in HOI detection,
which impedes the recognition of interactions. In contrast, Graph Neural
Networks (GNNs) are inherently better suited for this task, as they explicitly
model the relationships between human-object pairs. Therefore, in this paper,
we propose \textbf{M}ultimodal \textbf{G}raph \textbf{N}etwork
\textbf{M}odeling (MGNM) that leverages GNN-based relational structures to
enhance HOI detection. Specifically, we design a multimodal graph network
framework that explicitly models the HOI task in a four-stage graph structure.
Furthermore, we introduce a multi-level feature interaction mechanism within
our graph network. This mechanism leverages multi-level vision and language
features to enhance information propagation across human-object pairs.
Consequently, our proposed MGNM achieves state-of-the-art performance on two
widely used benchmarks: HICO-DET and V-COCO. Moreover, when integrated with a
more advanced object detector, our method demonstrates a significant
performance gain and maintains an effective balance between rare and non-rare
classes.

### 4. [VQT-Light:Lightweight HDR Illumination Map Prediction with Richer Texture.pdf](http://arxiv.org/pdf/2509.12556v1)

Authors: Kunliang Xie

Accurate lighting estimation is a significant yet challenging task in
computer vision and graphics. However, existing methods either struggle to
restore detailed textures of illumination map, or face challenges in running
speed and texture fidelity. To tackle this problem, we propose a novel
framework (VQT-Light) based on VQVAE and ViT architecture. VQT-Light includes
two modules: feature extraction and lighting estimation. First, we take
advantages of VQVAE to extract discrete features of illumination map rather
than continuous features to avoid "posterior collapse". Second, we capture
global context and dependencies of input image through ViT rather than CNNs to
improve the prediction of illumination outside the field of view. Combining the
above two modules, we formulate the lighting estimation as a multiclass
classification task, which plays a key role in our pipeline. As a result, our
model predicts light map with richer texture and better fidelity while keeping
lightweight and fast. VQT-Light achieves an inference speed of 40FPS and
improves multiple evaluation metrics. Qualitative and quantitative experiments
demonstrate that the proposed method realizes superior results compared to
existing state-of-the-art methods.

### 5. [Exploring Spectral Characteristics for Single Image Reflection Removal](http://arxiv.org/pdf/2509.12627v1)

Authors: Pengbo Guo, Chengxu Liu, Guoshuai Zhao, Xingsong Hou, Jialie Shen, Xueming Qian

Eliminating reflections caused by incident light interacting with reflective
medium remains an ill-posed problem in the image restoration area. The primary
challenge arises from the overlapping of reflection and transmission components
in the captured images, which complicates the task of accurately distinguishing
and recovering the clean background. Existing approaches typically address
reflection removal solely in the image domain, ignoring the spectral property
variations of reflected light, which hinders their ability to effectively
discern reflections. In this paper, we start with a new perspective on spectral
learning, and propose the Spectral Codebook to reconstruct the optical spectrum
of the reflection image. The reflections can be effectively distinguished by
perceiving the wavelength differences between different light sources in the
spectrum. To leverage the reconstructed spectrum, we design two spectral prior
refinement modules to re-distribute pixels in the spatial dimension and
adaptively enhance the spectral differences along the wavelength dimension.
Furthermore, we present the Spectrum-Aware Transformer to jointly recover the
transmitted content in spectral and pixel domains. Experimental results on
three different reflection benchmarks demonstrate the superiority and
generalization ability of our method compared to state-of-the-art models.

### 6. [Maps for Autonomous Driving: Full-process Survey and Frontiers](http://arxiv.org/pdf/2509.12632v1)

Authors: Pengxin Chen, Zhipeng Luo, Xiaoqi Jiang, Zhangcai Yin, Jonathan Li

Maps have always been an essential component of autonomous driving. With the
advancement of autonomous driving technology, both the representation and
production process of maps have evolved substantially. The article categorizes
the evolution of maps into three stages: High-Definition (HD) maps, Lightweight
(Lite) maps, and Implicit maps. For each stage, we provide a comprehensive
review of the map production workflow, with highlighting technical challenges
involved and summarizing relevant solutions proposed by the academic community.
Furthermore, we discuss cutting-edge research advances in map representations
and explore how these innovations can be integrated into end-to-end autonomous
driving frameworks.

### 7. [StereoCarla: A High-Fidelity Driving Dataset for Generalizable Stereo](http://arxiv.org/pdf/2509.12683v1)

Authors: Xianda Guo, Chenming Zhang, Ruilin Wang, Youmin Zhang, Wenzhao Zheng, Matteo Poggi, Hao Zhao, Qin Zou, Long Chen

Stereo matching plays a crucial role in enabling depth perception for
autonomous driving and robotics. While recent years have witnessed remarkable
progress in stereo matching algorithms, largely driven by learning-based
methods and synthetic datasets, the generalization performance of these models
remains constrained by the limited diversity of existing training data. To
address these challenges, we present StereoCarla, a high-fidelity synthetic
stereo dataset specifically designed for autonomous driving scenarios. Built on
the CARLA simulator, StereoCarla incorporates a wide range of camera
configurations, including diverse baselines, viewpoints, and sensor placements
as well as varied environmental conditions such as lighting changes, weather
effects, and road geometries. We conduct comprehensive cross-domain experiments
across four standard evaluation datasets (KITTI2012, KITTI2015, Middlebury,
ETH3D) and demonstrate that models trained on StereoCarla outperform those
trained on 11 existing stereo datasets in terms of generalization accuracy
across multiple benchmarks. Furthermore, when integrated into multi-dataset
training, StereoCarla contributes substantial improvements to generalization
accuracy, highlighting its compatibility and scalability. This dataset provides
a valuable benchmark for developing and evaluating stereo algorithms under
realistic, diverse, and controllable settings, facilitating more robust depth
perception systems for autonomous vehicles. Code can be available at
https://github.com/XiandaGuo/OpenStereo, and data can be available at
https://xiandaguo.net/StereoCarla.

### 8. [SmokeBench: A Real-World Dataset for Surveillance Image Desmoking in Early-Stage Fire Scenes](http://arxiv.org/pdf/2509.12701v1)

Authors: Wenzhuo Jin, Qianfeng Yang, Xianhao Wu, Hongming Chen, Pengpeng Li, Xiang Chen

Early-stage fire scenes (0-15 minutes after ignition) represent a crucial
temporal window for emergency interventions. During this stage, the smoke
produced by combustion significantly reduces the visibility of surveillance
systems, severely impairing situational awareness and hindering effective
emergency response and rescue operations. Consequently, there is an urgent need
to remove smoke from images to obtain clear scene information. However, the
development of smoke removal algorithms remains limited due to the lack of
large-scale, real-world datasets comprising paired smoke-free and
smoke-degraded images. To address these limitations, we present a real-world
surveillance image desmoking benchmark dataset named SmokeBench, which contains
image pairs captured under diverse scenes setup and smoke concentration. The
curated dataset provides precisely aligned degraded and clean images, enabling
supervised learning and rigorous evaluation. We conduct comprehensive
experiments by benchmarking a variety of desmoking methods on our dataset. Our
dataset provides a valuable foundation for advancing robust and practical image
desmoking in real-world fire scenes. This dataset has been released to the
public and can be downloaded from https://github.com/ncfjd/SmokeBench.

### 9. [RIS-FUSION: Rethinking Text-Driven Infrared and Visible Image Fusion from the Perspective of Referring Image Segmentation](http://arxiv.org/pdf/2509.12710v1)

Authors: Siju Ma, Changsiyu Gong, Xiaofeng Fan, Yong Ma, Chengjie Jiang

Text-driven infrared and visible image fusion has gained attention for
enabling natural language to guide the fusion process. However, existing
methods lack a goal-aligned task to supervise and evaluate how effectively the
input text contributes to the fusion outcome. We observe that referring image
segmentation (RIS) and text-driven fusion share a common objective:
highlighting the object referred to by the text. Motivated by this, we propose
RIS-FUSION, a cascaded framework that unifies fusion and RIS through joint
optimization. At its core is the LangGatedFusion module, which injects textual
features into the fusion backbone to enhance semantic alignment. To support
multimodal referring image segmentation task, we introduce MM-RIS, a
large-scale benchmark with 12.5k training and 3.5k testing triplets, each
consisting of an infrared-visible image pair, a segmentation mask, and a
referring expression. Extensive experiments show that RIS-FUSION achieves
state-of-the-art performance, outperforming existing methods by over 11% in
mIoU. Code and dataset will be released at
https://github.com/SijuMa2003/RIS-FUSION.

### 10. [Learning by Imagining: Debiased Feature Augmentation for Compositional Zero-Shot Learning](http://arxiv.org/pdf/2509.12711v1)

Authors: Haozhe Zhang, Chenchen Jing, Mingyu Liu, Qingsheng Wang, Hao Chen

Compositional Zero-Shot Learning (CZSL) aims to recognize unseen
attribute-object compositions by learning prior knowledge of seen primitives,
\textit{i.e.}, attributes and objects. Learning generalizable compositional
representations in CZSL remains challenging due to the entangled nature of
attributes and objects as well as the prevalence of long-tailed distributions
in real-world data. Inspired by neuroscientific findings that imagination and
perception share similar neural processes, we propose a novel approach called
Debiased Feature Augmentation (DeFA) to address these challenges. The proposed
DeFA integrates a disentangle-and-reconstruct framework for feature
augmentation with a debiasing strategy. DeFA explicitly leverages the prior
knowledge of seen attributes and objects by synthesizing high-fidelity
composition features to support compositional generalization. Extensive
experiments on three widely used datasets demonstrate that DeFA achieves
state-of-the-art performance in both \textit{closed-world} and
\textit{open-world} settings.

### Computers and Society

### 1. [An AI-Powered Framework for Analyzing Collective Idea Evolution in Deliberative Assemblies](http://arxiv.org/pdf/2509.12577v1)

Authors: Elinor Poole-Dayan, Deb Roy, Jad Kabbara

In an era of increasing societal fragmentation, political polarization, and
erosion of public trust in institutions, representative deliberative assemblies
are emerging as a promising democratic forum for developing effective policy
outcomes on complex global issues. Despite theoretical attention, there remains
limited empirical work that systematically traces how specific ideas evolve,
are prioritized, or are discarded during deliberation to form policy
recommendations. Addressing these gaps, this work poses two central questions:
(1) How might we trace the evolution and distillation of ideas into concrete
recommendations within deliberative assemblies? (2) How does the deliberative
process shape delegate perspectives and influence voting dynamics over the
course of the assembly? To address these questions, we develop LLM-based
methodologies for empirically analyzing transcripts from a tech-enhanced
in-person deliberative assembly. The framework identifies and visualizes the
space of expressed suggestions. We also empirically reconstruct each delegate's
evolving perspective throughout the assembly. Our methods contribute novel
empirical insights into deliberative processes and demonstrate how LLMs can
surface high-resolution dynamics otherwise invisible in traditional assembly
outputs.

### 2. [Designing the Hybrid Cooperative: A Socio-Technical Architecture for Scalable, Global Coordination Using Blockchain](http://arxiv.org/pdf/2509.13156v1)

Authors: Henrik Axelsen, Jan Damsgaard

Blockchain has been promoted as a remedy for coordination in fragmented,
multi-stakeholder ecosystems, yet many projects stall at pilot stage. Using a
design-science approach, we develop the Hybrid Cooperative (HC), a digitally
native governance architecture that combines smart-contract coordination with a
minimal, code-deferent legal interface and jurisdictional modules. This
selective decentralization decentralizes rules where programmability lowers
agency and verification costs, and centralizes only what is needed for
enforceability. A post-case evaluation against two traceability initiatives in
supply chains illustrates how the HC improves distributed task management,
verifiable information, incentive alignment, institutional interoperability,
and scalable, contestable governance. The paper contributes to Information
Systems by specifying a socio-technical model for scalable, multi-stakeholder
coordination across regulatory and organizational boundaries.

### 3. [Beyond Private or Public: Large Language Models as Quasi-Public Goods in the AI Economy](http://arxiv.org/pdf/2509.13265v1)

Authors: Yukun Zhang, TianYang Zhang

This paper conceptualizes Large Language Models (LLMs) as a form of mixed
public goods within digital infrastructure, analyzing their economic properties
through a comprehensive theoretical framework. We develop mathematical models
to quantify the non-rivalry characteristics, partial excludability, and
positive externalities of LLMs. Through comparative analysis of open-source and
closed-source development paths, we identify systematic differences in resource
allocation efficiency, innovation trajectories, and access equity. Our
empirical research evaluates the spillover effects and network externalities of
LLMs across different domains, including knowledge diffusion, innovation
acceleration, and industry transformation. Based on these findings, we propose
policy recommendations for balancing innovation incentives with equitable
access, including public-private partnership mechanisms, computational resource
democratization, and governance structures that optimize social welfare. This
interdisciplinary approach contributes to understanding the economic nature of
foundation AI models and provides policy guidance for their development as
critical digital infrastructure

### 4. [Introducing the A2AJ's Canadian Legal Data: An open-source alternative to CanLII for the era of computational law](http://arxiv.org/pdf/2509.13032v1)

Authors: Simon Wallace, Sean Rehaag

The Access to Algorithmic Justice project (A2AJ) is an open-source
alternative to the Canadian Legal Information Institute (CanLII). At a moment
when technology promises to enable new ways of working with law, CanLII is
becoming an impediment to the free access of law and access to justice
movements because it restricts bulk and programmatic access to Canadian legal
data. This means that Canada is staring down a digital divide: well-resourced
actors have the best new technological tools and, because CanLII has disclaimed
leadership, the public only gets second-rate tools. This article puts CanLII in
its larger historical context and shows how long and deep efforts to
democratize access to Canadian legal data are, and how often they are thwarted
by private industry. We introduce the A2AJ's Canadian Legal Data project, which
provides open access to over 116,000 court decisions and 5,000 statutes through
multiple channels including APIs, machine learning datasets, and AI integration
protocols. Through concrete examples, we demonstrate how open legal data
enables courts to conduct evidence-based assessments and allows developers to
create tools for practitioners serving low-income communities.

### 5. [CattleSense -- A Multisensory Approach to Optimize Cattle Well-Being](http://arxiv.org/pdf/2509.12617v1)

Authors: Srijesh Pillai, M. I. Jawid Nazir

CattleSense is an innovative application of Internet of Things (IoT)
technology for the comprehensive monitoring and management of cattle
well-being. This research paper outlines the design and implementation of a
sophisticated system using a Raspberry Pi Module 4B, RFID Card Reader, Electret
Arduino Microphone Module, DHT11 Sensor, Arduino UNO, Neo-6M GPS Sensor, and
Heartbeat Sensor. The system aims to provide real-time surveillance of the
environment in which Cows are present and individual Cow parameters such as
location, milking frequency, and heartbeat fluctuations. The primary objective
is to simplify managing the Cattle in the shed, ensuring that the Cattle are
healthy and safe.

### 6. [DoubleAgents: Exploring Mechanisms of Building Trust with Proactive AI](http://arxiv.org/pdf/2509.12626v1)

Authors: Tao Long, Xuanming Zhang, Sitong Wang, Zhou Yu, Lydia B Chilton

Agentic workflows promise efficiency, but adoption hinges on whether people
actually trust systems that act on their behalf. We present DoubleAgents, an
agentic planning tool that embeds transparency and control through user
intervention, value-reflecting policies, rich state visualizations, and
uncertainty flagging for human coordination tasks. A built-in respondent
simulation generates realistic scenarios, allowing users to rehearse, refine
policies, and calibrate their reliance before live use. We evaluate
DoubleAgents in a two-day lab study (n=10), two deployments (n=2), and a
technical evaluation. Results show that participants initially hesitated to
delegate but grew more reliant as they experienced transparency, control, and
adaptive learning during simulated cases. Deployment results demonstrate
DoubleAgents' real-world relevance and usefulness, showing that the effort
required scaled appropriately with task complexity and contextual data. We
contribute trust-by-design patterns and mechanisms for proactive AI --
consistency, controllability, and explainability -- along with simulation as a
safe path to build and calibrate trust over time.

### 7. [Podcasts as a Medium for Participation in Collective Action: A Case Study of Black Lives Matter](http://arxiv.org/pdf/2509.13197v1)

Authors: Theodora Moldovan, Arianna Pera, Davide Vega, Luca Maria Aiello

We study how participation in collective action is articulated in podcast
discussions, using the Black Lives Matter (BLM) movement as a case study. While
research on collective action discourse has primarily focused on text-based
content, this study takes a first step toward analyzing audio formats by using
podcast transcripts. Using the Structured Podcast Research Corpus (SPoRC), we
investigated spoken language expressions of participation in collective action,
categorized as problem-solution, call-to-action, intention, and execution. We
identified podcast episodes discussing racial justice after important
BLM-related events in May and June of 2020, and extracted participatory
statements using a layered framework adapted from prior work on social media.
We examined the emotional dimensions of these statements, detecting eight key
emotions and their association with varying stages of activism. We found that
emotional profiles vary by stage, with different positive emotions standing out
during calls-to-action, intention, and execution. We detected negative
associations between collective action and negative emotions, contrary to
theoretical expectations. Our work contributes to a better understanding of how
activism is expressed in spoken digital discourse and how emotional framing may
depend on the format of the discussion.

### Databases

### 1. [ScaleDoc: Scaling LLM-based Predicates over Large Document Collections](http://arxiv.org/pdf/2509.12610v1)

Authors: Hengrui Zhang, Yulong Hui, Yihao Liu, Huanchen Zhang

Predicates are foundational components in data analysis systems. However,
modern workloads increasingly involve unstructured documents, which demands
semantic understanding, beyond traditional value-based predicates. Given
enormous documents and ad-hoc queries, while Large Language Models (LLMs)
demonstrate powerful zero-shot capabilities, their high inference cost leads to
unacceptable overhead. Therefore, we introduce \textsc{ScaleDoc}, a novel
system that addresses this by decoupling predicate execution into an offline
representation phase and an optimized online filtering phase. In the offline
phase, \textsc{ScaleDoc} leverages a LLM to generate semantic representations
for each document. Online, for each query, it trains a lightweight proxy model
on these representations to filter the majority of documents, forwarding only
the ambiguous cases to the LLM for final decision. Furthermore,
\textsc{ScaleDoc} proposes two core innovations to achieve significant
efficiency: (1) a contrastive-learning-based framework that trains the proxy
model to generate reliable predicating decision scores; (2) an adaptive cascade
mechanism that determines the effective filtering policy while meeting specific
accuracy targets. Our evaluations across three datasets demonstrate that
\textsc{ScaleDoc} achieves over a 2$\times$ end-to-end speedup and reduces
expensive LLM invocations by up to 85\%, making large-scale semantic analysis
practical and efficient.

### Distributed, Parallel, and Cluster Computing

### 1. [Analysis and Optimization of Wireless Multimodal Federated Learning on Modal Heterogeneity](http://arxiv.org/pdf/2509.12930v1)

Authors: Xuefeng Han, Wen Chen, Jun Li, Ming Ding, Qingqing Wu, Kang Wei, Xiumei Deng, Yumeng Shao, Qiong Wu

Multimodal federated learning (MFL) is a distributed framework for training
multimodal models without uploading local multimodal data of clients, thereby
effectively protecting client privacy. However, multimodal data is commonly
heterogeneous across diverse clients, where each client possesses only a subset
of all modalities, renders conventional analysis results and optimization
methods in unimodal federated learning inapplicable. In addition, fixed latency
demand and limited communication bandwidth pose significant challenges for
deploying MFL in wireless scenarios. To optimize the wireless MFL performance
on modal heterogeneity, this paper proposes a joint client scheduling and
bandwidth allocation (JCSBA) algorithm based on a decision-level fusion
architecture with adding a unimodal loss function. Specifically, with the
decision results, the unimodal loss functions are added to both the training
objective and local update loss functions to accelerate multimodal convergence
and improve unimodal performance. To characterize MFL performance, we derive a
closed-form upper bound related to client and modality scheduling and minimize
the derived bound under the latency, energy, and bandwidth constraints through
JCSBA. Experimental results on multimodal datasets demonstrate that the JCSBA
algorithm improves the multimodal accuracy and the unimodal accuracy by 4.06%
and 2.73%, respectively, compared to conventional algorithms.

### 2. [Asymmetric Grid Quorum Systems for Heterogeneous Processes](http://arxiv.org/pdf/2509.12942v1)

Authors: Michael Senn, Christian Cachin

Quorum systems are a common way to formalize failure assumptions in
distributed systems. Traditionally, these assumptions are shared by all
involved processes. More recently, systems have emerged which allow processes
some freedom in choosing their own, subjective or asymmetric, failure
assumptions. For such a system to work, individual processes' assumptions must
be compatible. However, this leads to a Catch-22-style scenario: How can
processes collaborate to agree on compatible failure assumptions when they have
no compatible failure assumptions to start with?
  We introduce asymmetric grid quorum systems that allow a group of processes
to specify heterogeneous trust assumptions independently of each other and
without coordination. They are based on qualitative attributes describing how
the processes differ. Each process may select a quorum system from this class
that aligns best with its subjective view. The available choices are designed
to be compatible by definition, thereby breaking the cycling dependency.
Asymmetric grid quorum systems have many applications that range from cloud
platforms to blockchain networks.

### 3. [Space-Time Trade-off in Bounded Iterated Memory](http://arxiv.org/pdf/2509.13157v1)

Authors: Guillermo Toyos-Marfurt, Petr Kuznetsov

The celebrated asynchronous computability theorem (ACT) characterizes tasks
solvable in the read-write shared-memory model using the unbounded
full-information protocol, where in every round of computation, each process
shares its complete knowledge of the system with the other processes.
Therefore, ACT assumes shared-memory variables of unbounded capacity. It has
been recently shown that boundedvariables can achieve the same computational
power at the expense of extra rounds. However, the exact relationship between
the bit capacity of the shared memory and the number of rounds required in
order to implement one round of the full-information protocol remained unknown.
  In this paper, we focus on the asymptotic round complexity of bounded
iterated shared-memory algorithms that simulate, up to isomorphism, the
unbounded full-information protocol. We relate the round complexity to the
number of processes $n$, the number of iterations of the full information
protocol $r$, and the bit size per shared-memory entry $b$. By analyzing the
corresponding protocol complex, a combinatorial structure representing
reachable states, we derive necessary conditions and present a bounded
full-information algorithm tailored to the bits available $b$ per shared memory
entry. We show that for $n>2$, the round complexity required to implement the
full-information protocol satisfies $\Omega((n!)^{r-1} \cdot 2^{n-b})$. Our
results apply to a range of iterated shared-memory models, from regular
read-write registers to atomic and immediate snapshots. Moreover, our bounded
full-information algorithm is asymptotically optimal for the iterated collect
model and within a linear factor $n$ of optimal for the snapshot-based models.

### 4. [Scaling Up Throughput-oriented LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management](http://arxiv.org/pdf/2509.13201v1)

Authors: Thanh Son Phung, Douglas Thain

The widespread growth in LLM developments increasingly demands more
computational power from clusters than what they can supply. Traditional LLM
applications inherently require huge static resource allocations, which force
users to either wait in a long job queue and accept progress delay, or buy
expensive hardware to fulfill their needs and exacerbate the demand-supply
problem. However, not all LLM applications are latency-sensitive and can
instead be executed in a throughput-oriented way. This throughput orientation
allows a dynamic allocation that opportunistically pools available resources
over time, avoiding both the long queue and expensive GPU purchases.
Effectively utilizing opportunistic resources brings numerous challenges
nevertheless. Our solution, pervasive context management, exploits the common
computational context in LLM applications and provides mechanisms and policies
that allow seamless context reuse on opportunistic resources. Our evaluation
shows an LLM application with pervasive context management on opportunistic
resources reduces its execution time by 98.1%.

### 5. [Energy-Efficient Quantized Federated Learning for Resource-constrained IoT devices](http://arxiv.org/pdf/2509.12814v1)

Authors: Wilfrid Sougrinoma Compaoré, Yaya Etiabi, El Mehdi Amhoud, Mohamad Assaad

Federated Learning (FL) has emerged as a promising paradigm for enabling
collaborative machine learning while preserving data privacy, making it
particularly suitable for Internet of Things (IoT) environments. However,
resource-constrained IoT devices face significant challenges due to limited
energy,unreliable communication channels, and the impracticality of assuming
infinite blocklength transmission. This paper proposes a federated learning
framework for IoT networks that integrates finite blocklength transmission,
model quantization, and an error-aware aggregation mechanism to enhance energy
efficiency and communication reliability. The framework also optimizes uplink
transmission power to balance energy savings and model performance. Simulation
results demonstrate that the proposed approach significantly reduces energy
consumption by up to 75\% compared to a standard FL model, while maintaining
robust model accuracy, making it a viable solution for FL in real-world IoT
scenarios with constrained resources. This work paves the way for efficient and
reliable FL implementations in practical IoT deployments. Index Terms:
Federated learning, IoT, finite blocklength, quantization, energy efficiency.

### 6. [AI Factories: It's time to rethink the Cloud-HPC divide](http://arxiv.org/pdf/2509.12849v1)

Authors: Pedro Garcia Lopez, Daniel Barcelona Pons, Marcin Copik, Torsten Hoefler, Eduardo Quiñones, Maciej Malawski, Peter Pietzutch, Alberto Marti, Thomas Ohlson Timoudas, Aleksander Slominski

The strategic importance of artificial intelligence is driving a global push
toward Sovereign AI initiatives. Nationwide governments are increasingly
developing dedicated infrastructures, called AI Factories (AIF), to achieve
technological autonomy and secure the resources necessary to sustain robust
local digital ecosystems.
  In Europe, the EuroHPC Joint Undertaking is investing hundreds of millions of
euros into several AI Factories, built atop existing high-performance computing
(HPC) supercomputers. However, while HPC systems excel in raw performance, they
are not inherently designed for usability, accessibility, or serving as
public-facing platforms for AI services such as inference or agentic
applications. In contrast, AI practitioners are accustomed to cloud-native
technologies like Kubernetes and object storage, tools that are often difficult
to integrate within traditional HPC environments.
  This article advocates for a dual-stack approach within supercomputers:
integrating both HPC and cloud-native technologies. Our goal is to bridge the
divide between HPC and cloud computing by combining high performance and
hardware acceleration with ease of use and service-oriented front-ends. This
convergence allows each paradigm to amplify the other. To this end, we will
study the cloud challenges of HPC (Serverless HPC) and the HPC challenges of
cloud technologies (High-performance Cloud).

### 7. [Emergent complexity and rhythms in evoked and spontaneous dynamics of human whole-brain models after tuning through analysis tools](http://arxiv.org/pdf/2509.12873v1)

Authors: Gianluca Gaglioti, Alessandra Cardinale, Cosimo Lupo, Thierry Nieus, Federico Marmoreo, Robin Gutzen, Michael Denker, Andrea Pigorini, Marcello Massimini, Simone Sarasso, Pier Stanislao Paolucci, Giulia De Bonis

The simulation of whole-brain dynamics should reproduce realistic spontaneous
and evoked neural activity across different scales, including emergent rhythms,
spatio-temporal activation patterns, and macroscale complexity. Once a
mathematical model is selected, its configuration must be determined by
properly setting its parameters. A critical preliminary step in this process is
defining an appropriate set of observables to guide the selection of model
configurations (parameter tuning), laying the groundwork for quantitative
calibration of accurate whole-brain models. Here, we address this challenge by
presenting a framework that integrates two complementary tools: The Virtual
Brain (TVB) platform for simulating whole-brain dynamics, and the Collaborative
Brain Wave Analysis Pipeline (Cobrawap) for analyzing the simulations using a
set of standardized metrics. We apply this framework to a 998-node human
connectome, using two configurations of the Larter-Breakspear neural mass
model: one with the TVB default parameters, the other tuned using Cobrawap. The
results reveal that the tuned configuration exhibits several biologically
relevant features, absent in the default model for both spontaneous and evoked
dynamics. In response to external perturbations, the tuned model generates
non-stereotyped, complex spatio-temporal activity, as measured by the
perturbational complexity index. In spontaneous activity, it displays robust
alpha-band oscillations, infra-slow rhythms, scale-free characteristics,
greater spatio-temporal heterogeneity, and asymmetric functional connectivity.
This work demonstrates the potential of combining TVB and Cobrawap to guide
parameter tuning and lays the groundwork for data-driven calibration and
validation of accurate whole-brain models.

### Digital Libraries

### 1. [Layout-Aware OCR for Black Digital Archives with Unsupervised Evaluation](http://arxiv.org/pdf/2509.13236v1)

Authors: Fitsum Sileshi Beyene, Christopher L. Dancy

Despite their cultural and historical significance, Black digital archives
continue to be a structurally underrepresented area in AI research and
infrastructure. This is especially evident in efforts to digitize historical
Black newspapers, where inconsistent typography, visual degradation, and
limited annotated layout data hinder accurate transcription, despite the
availability of various systems that claim to handle optical character
recognition (OCR) well. In this short paper, we present a layout-aware OCR
pipeline tailored for Black newspaper archives and introduce an unsupervised
evaluation framework suited to low-resource archival contexts. Our approach
integrates synthetic layout generation, model pretraining on augmented data,
and a fusion of state-of-the-art You Only Look Once (YOLO) detectors. We used
three annotation-free evaluation metrics, the Semantic Coherence Score (SCS),
Region Entropy (RE), and Textual Redundancy Score (TRS), which quantify
linguistic fluency, informational diversity, and redundancy across OCR regions.
Our evaluation on a 400-page dataset from ten Black newspaper titles
demonstrates that layout-aware OCR improves structural diversity and reduces
redundancy compared to full-page baselines, with modest trade-offs in
coherence. Our results highlight the importance of respecting cultural layout
logic in AI-driven document understanding and lay the foundation for future
community-driven and ethically grounded archival AI systems.

### 2. [Automated Generation of Research Workflows from Academic Papers: A Full-text Mining Framework](http://arxiv.org/pdf/2509.12955v1)

Authors: Heng Zhang, Chengzhi Zhang

The automated generation of research workflows is essential for improving the
reproducibility of research and accelerating the paradigm of "AI for Science".
However, existing methods typically extract merely fragmented procedural
components and thus fail to capture complete research workflows. To address
this gap, we propose an end-to-end framework that generates comprehensive,
structured research workflows by mining full-text academic papers. As a case
study in the Natural Language Processing (NLP) domain, our paragraph-centric
approach first employs Positive-Unlabeled (PU) Learning with SciBERT to
identify workflow-descriptive paragraphs, achieving an F1-score of 0.9772.
Subsequently, we utilize Flan-T5 with prompt learning to generate workflow
phrases from these paragraphs, yielding ROUGE-1, ROUGE-2, and ROUGE-L scores of
0.4543, 0.2877, and 0.4427, respectively. These phrases are then systematically
categorized into data preparation, data processing, and data analysis stages
using ChatGPT with few-shot learning, achieving a classification precision of
0.958. By mapping categorized phrases to their document locations in the
documents, we finally generate readable visual flowcharts of the entire
research workflows. This approach facilitates the analysis of workflows derived
from an NLP corpus and reveals key methodological shifts over the past two
decades, including the increasing emphasis on data analysis and the transition
from feature engineering to ablation studies. Our work offers a validated
technical framework for automated workflow generation, along with a novel,
process-oriented perspective for the empirical investigation of evolving
scientific paradigms. Source code and data are available at:
https://github.com/ZH-heng/research_workflow.

### Discrete Mathematics

### 1. [The Power Contamination Problem on Grids Revisited: Optimality, Combinatorics, and Links to Integer Sequences](http://arxiv.org/pdf/2509.12756v1)

Authors: El-Mehdi Mehiri, Mohammed L. Nadji

This paper presents a combinatorial study of the power contamination problem,
a dynamic variant of power domination modeled on grid graphs. We resolve a
conjecture posed by Ainouche and Bouroubi (2021) by proving it is false and
instead establish the exact value of the power contamination number on grid
graphs. Furthermore, we derive recurrence relations for this number and
initiate the enumeration of optimal contamination sets. We prove that the
number of optimal solutions for specific grid families corresponds to
well-known integer sequences, including those counting ternary words with
forbidden subwords and the large Schr\"oder numbers. This work settles the
fundamental combinatorial questions of the power contamination problem on grids
and reveals its rich connections to classical combinatorics.

### 2. [An involution for trivariate symmetries of vincular patterns](http://arxiv.org/pdf/2509.13097v1)

Authors: Joanna N. Chen, Shishuo Fu, Jiang Zeng

We provide a bijective proof of the equidistribution of two pairs of vincular
patterns in permutations, thereby resolving a recent open problem of Bitonti,
Deb, and Sokal (arXiv:2412.10214). Since the bijection is involutive, we also
confirm their conjecture on the equidistribution of triple vincular patterns.
Somewhat unexpectedly, we show that this involution is closed on the set of
Baxter permutations, thereby implying another trivariate symmetries of vincular
patterns. The proof of this second result requires a variant of a
characterization of Baxter permutations in terms of restricted Laguerre
histories, first given by Viennot using the Fran\c{c}on-Viennot bijection.

### Data Structures and Algorithms

### 1. [Efficient Enumeration of At Most $k$-Out Polygons](http://arxiv.org/pdf/2509.12696v1)

Authors: Waseem Akram, Katsuhisa Yamanaka

Let $S$ be a set of $n$ points in the Euclidean plane and general position
i.e., no three points are collinear. An \emph{at most $k$-out polygon of $S$}
is a simple polygon such that each vertex is a point in $S$ and there are at
most $k$ points outside the polygon. In this paper, we consider the problem of
enumerating all the at most $k$-out polygon of $S$. We propose a new
enumeration algorithm for the at most $k$-out polygons of a point set. Our
algorithm enumerates all the at most $k$-out polygons in $\mathcal{O}(n^2
\log{n})$ delay, while the running time of an existing algorithm is
$\mathcal{O}(n^3 \log{n})$ delay.

### 2. [TimeCluster with PCA is Equivalent to Subspace Identification of Linear Dynamical Systems](http://arxiv.org/pdf/2509.12895v1)

Authors: Christian L. Hines, Samuel Spillard, Daniel P. Martin

TimeCluster is a visual analytics technique for discovering structure in long
multivariate time series by projecting overlapping windows of data into a
low-dimensional space. We show that, when Principal Component Analysis (PCA) is
chosen as the dimensionality reduction technique, this procedure is
mathematically equivalent to classical linear subspace identification
(block-Hankel matrix plus Singular Vector Decomposition (SVD)). In both
approaches, the same low-dimensional linear subspace is extracted from the time
series data. We first review the TimeCluster method and the theory of subspace
system identification. Then we show that forming the sliding-window matrix of a
time series yields a Hankel matrix, so applying PCA (via SVD) to this matrix
recovers the same principal directions as subspace identification. Thus the
cluster coordinates from TimeCluster coincide with the subspace identification
methods. We present experiments on synthetic and real dynamical signals
confirming that the two embeddings coincide. Finally, we explore and discuss
future opportunities enabled by this equivalence, including forecasting from
the identified state space, streaming/online extensions, incorporating and
visualising external inputs and robust techniques for displaying underlying
trends in corrupted data.

### 3. [Protecting participants or population? Comparison of k-anonymous Origin-Destination matrices](http://arxiv.org/pdf/2509.12950v1)

Authors: Pietro Armenante, Kai Huang, Nikhil Jha, Luca Vassio

Origin-Destination (OD) matrices are a core component of research on users'
mobility and summarize how individuals move between geographical regions. These
regions should be small enough to be representative of user mobility, without
incurring substantial privacy risks. There are two added values of the
NetMob2025 challenge dataset. Firstly, the data is extensive and contains a lot
of socio-demographic information that can be used to create multiple OD
matrices, based on the segments of the population. Secondly, a participant is
not merely a record in the data, but a statistically weighted proxy for a
segment of the real population. This opens the door to a fundamental shift in
the anonymization paradigm. A population-based view of privacy is central to
our contribution. By adjusting our anonymization framework to account for
representativeness, we are also protecting the inferred identity of the actual
population, rather than survey participants alone. The challenge addressed in
this work is to produce and compare OD matrices that are k-anonymous for survey
participants and for the whole population. We compare several traditional
methods of anonymization to k-anonymity by generalizing geographical areas.
These include generalization over a hierarchy (ATG and OIGH) and the classical
Mondrian. To this established toolkit, we add a novel method, i.e., ODkAnon, a
greedy algorithm aiming at balancing speed and quality. Unlike previous
approaches, which primarily address the privacy aspects of the given datasets,
we aim to contribute to the generation of privacy-preserving OD matrices
enriched with socio-demographic segmentation that achieves k-anonymity on the
actual population.

### 4. [Sublinear-Time Algorithms for Diagonally Dominant Systems and Applications to the Friedkin-Johnsen Model](http://arxiv.org/pdf/2509.13112v1)

Authors: Weiming Feng, Zelin Li, Pan Peng

We study sublinear-time algorithms for solving linear systems $Sz = b$, where
$S$ is a diagonally dominant matrix, i.e., $|S_{ii}| \geq \delta + \sum_{j \ne
i} |S_{ij}|$ for all $i \in [n]$, for some $\delta \geq 0$. We present
randomized algorithms that, for any $u \in [n]$, return an estimate $z_u$ of
$z^*_u$ with additive error $\varepsilon$ or $\varepsilon \lVert
z^*\rVert_\infty$, where $z^*$ is some solution to $Sz^* = b$, and the
algorithm only needs to read a small portion of the input $S$ and $b$. For
example, when the additive error is $\varepsilon$ and assuming $\delta>0$, we
give an algorithm that runs in time $O\left( \frac{\|b\|_\infty^2
S_{\max}}{\delta^3 \varepsilon^2} \log \frac{\| b \|_\infty}{\delta
\varepsilon} \right)$, where $S_{\max} = \max_{i \in [n]} |S_{ii}|$. We also
prove a matching lower bound, showing that the linear dependence on $S_{\max}$
is optimal. Unlike previous sublinear-time algorithms, which apply only to
symmetric diagonally dominant matrices with non-negative diagonal entries, our
algorithm works for general strictly diagonally dominant matrices ($\delta >
0$) and a broader class of non-strictly diagonally dominant matrices $(\delta =
0)$. Our approach is based on analyzing a simple probabilistic recurrence
satisfied by the solution. As an application, we obtain an improved
sublinear-time algorithm for opinion estimation in the Friedkin--Johnsen model.

### Emerging Technologies

### 1. [An Analysis of Resource Allocation and User Association Strategies in Space-Air-Ground Integrated Networks](http://arxiv.org/pdf/2509.12657v1)

Authors: Siri Vennela Geddam, Sruthi Ilapuram, Kamesh Namuduri, K L V Sai Prakash Sakuru

Space-Air-Ground-Integrated Networks (SAGIN) enable seamless data
connectivity for applications such as smart transport, healthcare, smart
cities, and disaster response through the coordinated use of low-earth orbit
(LEO) satellites, base stations mounted with uncrewed aerial vehicles (UAV),
and terrestrial infrastructure. This paper provides a detailed analysis of
resource management frameworks, reviews the literature, and evaluates key
methods such as alternating optimization (AO), damped iterative water filling
(DIWF), and genetic algorithms (GA) for resource allocation. MATLAB simulation
results benchmark these algorithms across 10,000 trials, demonstrating robust,
fair, and low-latency resource allocation. In addition, this paper also
analyzes strategies for user association with terrestrial and aerial base
stations during emergencies and network overloads. The main contributions
include a comparative assessment of resource allocation strategies in SAGIN and
an in-depth analysis of user association policies for emergency scenarios. The
study provides guidance for designing resilient and efficient next-generation
networks. Potential future research directions include investigating satellite
handover and multi-domain orchestration for SAGIN deployments.

### 2. [DoubleAgents: Exploring Mechanisms of Building Trust with Proactive AI](http://arxiv.org/pdf/2509.12626v1)

Authors: Tao Long, Xuanming Zhang, Sitong Wang, Zhou Yu, Lydia B Chilton

Agentic workflows promise efficiency, but adoption hinges on whether people
actually trust systems that act on their behalf. We present DoubleAgents, an
agentic planning tool that embeds transparency and control through user
intervention, value-reflecting policies, rich state visualizations, and
uncertainty flagging for human coordination tasks. A built-in respondent
simulation generates realistic scenarios, allowing users to rehearse, refine
policies, and calibrate their reliance before live use. We evaluate
DoubleAgents in a two-day lab study (n=10), two deployments (n=2), and a
technical evaluation. Results show that participants initially hesitated to
delegate but grew more reliant as they experienced transparency, control, and
adaptive learning during simulated cases. Deployment results demonstrate
DoubleAgents' real-world relevance and usefulness, showing that the effort
required scaled appropriately with task complexity and contextual data. We
contribute trust-by-design patterns and mechanisms for proactive AI --
consistency, controllability, and explainability -- along with simulation as a
safe path to build and calibrate trust over time.

### 3. [Low-Altitude UAV Tracking via Sensing-Assisted Predictive Beamforming](http://arxiv.org/pdf/2509.12698v1)

Authors: Yifan Jiang, Qingqing Wu, Hongxun Hui, Wen Chen, Derrick Wing Kwan Ng

Sensing-assisted predictive beamforming, as one of the enabling technologies
for emerging integrated sensing and communication (ISAC) paradigm, shows
significant promise for enhancing various future unmanned aerial vehicle (UAV)
applications. However, current works predominately emphasized on spectral
efficiency enhancement, while the impact of such beamforming techniques on the
communication reliability was largely unexplored and challenging to
characterize. To fill this research gap and tackle this issue, this paper
investigates outage capacity maximization for UAV tracking under the
sensing-assisted predictive beamforming scheme. Specifically, a
cellular-connected UAV tracking scheme is proposed leveraging extended Kalman
filtering (EKF), where the predicted UAV trajectory, sensing duration ratio,
and target constant received signal-to-noise ratio (SNR) are jointly optimized
to maximize the outage capacity at each time slot. To address the implicit
nature of the objective function, closed-form approximations of the outage
probabilities (OPs) at both prediction and measurement stages of each time slot
are proposed based on second-order Taylor expansions, providing an efficient
and full characterization of outage capacity. Subsequently, an efficient
algorithm is proposed based on a combination of bisection search and successive
convex approximation (SCA) to address the non-convex optimization problem with
guaranteed convergence. To further reduce computational complexity, a second
efficient algorithm is developed based on alternating optimization (AO).
Simulation results validate the accuracy of the derived OP approximations, the
effectiveness of the proposed algorithms, and the significant outage capacity
enhancement over various benchmarks, while also indicating a trade-off between
decreasing path loss and enjoying wide beam coverage for outage capacity
maximization.

### 4. [Deep Learning for Model-Free Prediction of Thermal States of Robot Joint Motors](http://arxiv.org/pdf/2509.12739v1)

Authors: Trung Kien La, Eric Guiffo Kaigom

In this work, deep neural networks made up of multiple hidden Long Short-Term
Memory (LSTM) and Feedforward layers are trained to predict the thermal
behavior of the joint motors of robot manipulators. A model-free and scalable
approach is adopted. It accommodates complexity and uncertainty challenges
stemming from the derivation, identification, and validation of a large number
of parameters of an approximation model that is hardly available. To this end,
sensed joint torques are collected and processed to foresee the thermal
behavior of joint motors. Promising prediction results of the machine learning
based capture of the temperature dynamics of joint motors of a redundant robot
with seven joints are presented.

### 5. [Deep Generative and Discriminative Digital Twin endowed with Variational Autoencoder for Unsupervised Predictive Thermal Condition Monitoring of Physical Robots in Industry 6.0 and Society 6.0](http://arxiv.org/pdf/2509.12740v1)

Authors: Eric Guiffo Kaigom

Robots are unrelentingly used to achieve operational efficiency in Industry
4.0 along with symbiotic and sustainable assistance for the work-force in
Industry 5.0. As resilience, robustness, and well-being are required in
anti-fragile manufacturing and human-centric societal tasks, an autonomous
anticipation and adaption to thermal saturation and burns due to motors
overheating become instrumental for human safety and robot availability. Robots
are thereby expected to self-sustain their performance and deliver user
experience, in addition to communicating their capability to other agents in
advance to ensure fully automated thermally feasible tasks, and prolong their
lifetime without human intervention. However, the traditional robot shutdown,
when facing an imminent thermal saturation, inhibits productivity in factories
and comfort in the society, while cooling strategies are hard to implement
after the robot acquisition. In this work, smart digital twins endowed with
generative AI, i.e., variational autoencoders, are leveraged to manage
thermally anomalous and generate uncritical robot states. The notion of thermal
difficulty is derived from the reconstruction error of variational
autoencoders. A robot can use this score to predict, anticipate, and share the
thermal feasibility of desired motion profiles to meet requirements from
emerging applications in Industry 6.0 and Society 6.0.

### Formal Languages and Automata Theory

### 1. [A Variety of Request-Response Specifications](http://arxiv.org/pdf/2509.13078v1)

Authors: Daichi Aiba, Masaki Waga, Hiroya Fujinami, Koko Muroya, Shutaro Ouchi, Naoki Ueda, Yosuke Yokoyama, Yuta Wada, Ichiro Hasuo

We find, motivated by real-world applications, that the well-known
request-response specification comes with multiple variations, and that these
variations should be distinguished. As the first main contribution, we
introduce a classification of those variations into six types, and present it
as a decision tree, where a user is led to the type that is suited for their
application by answering a couple of questions. Our second main contribution is
the formalization of those six types in various formalisms such as temporal
logics, grammars, and automata; here, two types out of the six are non-regular
specifications and their formalization requires extended formalisms. We also
survey tools for monitoring these specifications to cater for practitioners'
needs.

### 2. [It Takes a Village: Bridging the Gaps between Current and Formal Specifications for Protocols](http://arxiv.org/pdf/2509.13208v1)

Authors: David Basin, Nate Foster, Kenneth L. McMillan, Kedar S. Namjoshi, Cristina Nita-Rotaru, Jonathan M. Smith, Pamela Zave, Lenore D. Zuck

Formal specifications have numerous benefits for both designers and users of
network protocols. They provide clear, unambiguous representations, which are
useful as documentation and for testing. They can help reveal disagreements
about what a protocol "is" and identify areas where further work is needed to
resolve ambiguities or internal inconsistencies. They also provide a foundation
for formal reasoning, making it possible to establish important security and
correctness guarantees on all inputs and in every environment. Despite these
advantages, formal methods are not widely used to design, implement, and
validate network protocols today. Instead, Internet protocols are usually
described in informal documents, such as IETF Requests for Comments (RFCs) or
IEEE standards. These documents primarily consist of lengthy prose
descriptions, accompanied by pseudocode, header descriptions, state machine
diagrams, and reference implementations which are used for interoperability
testing. So, while RFCs and reference implementations were only intended to
help guide the social process used by protocol designers, they have evolved
into the closest things to formal specifications the Internet community has. In
this paper, we discuss the different roles that specifications play in the
networking and formal methods communities. We then illustrate the potential
benefits of specifying protocols formally, presenting highlights from several
recent success stories. Finally, we identify key differences between how formal
specifications are understood by the two communities and suggest possible
strategies to bridge the gaps.

### Graphics

### 1. [Temporally Smooth Mesh Extraction for Procedural Scenes with Long-Range Camera Trajectories using Spacetime Octrees](http://arxiv.org/pdf/2509.13306v1)

Authors: Zeyu Ma, Adam Finkelstein, Jia Deng

The procedural occupancy function is a flexible and compact representation
for creating 3D scenes. For rasterization and other tasks, it is often
necessary to extract a mesh that represents the shape. Unbounded scenes with
long-range camera trajectories, such as flying through a forest, pose a unique
challenge for mesh extraction. A single static mesh representing all the
geometric detail necessary for the full camera path can be prohibitively large.
Therefore, independent meshes can be extracted for different camera views, but
this approach may lead to popping artifacts during transitions. We propose a
temporally coherent method for extracting meshes suitable for long-range camera
trajectories in unbounded scenes represented by an occupancy function. The key
idea is to perform 4D mesh extraction using a new spacetime tree structure
called a binary-octree. Experiments show that, compared to existing baseline
methods, our method offers superior visual consistency at a comparable cost.
The code and the supplementary video for this paper are available at
https://github.com/princeton-vl/BinocMesher.

### Computer Science and Game Theory

### 1. [Between proportionnality and envy-freeness: k-proportionality](http://arxiv.org/pdf/2509.12903v1)

Authors: Guillaume Chèze

This article deals with the cake cutting problem. In this setting, there
exists two notions of fair division: proportional division (when there are n
players, each player thinks to get at least 1/n of the cake) and envy-free
division (each player wants to keep his or her share because he or she does not
envy the portion given to another player). Some results are valid for
proportional division but not for envy-free division. Here, we introduce and
study a scale between the proportional division and the envy-free division. The
goal is to understand where is the gap between statements about proportional
division and envy-free division. This scale comes from the notion introduced in
this article: k-proportionality. When k = n this notion corresponds to the
proportional division and when k = 2 it corresponds to envy-free division. With
k-proportionality we can understand where some difficulties in fair division
lie. First, we show that there are situations in which there is no
k-proportional and equitable division of a pie with connected pieces when k
$\le$ n -1. This result was known only for envy-free division, ie k = 2. Next,
we prove that there are situations in which there is no Pareto-optimal
k-proportional division of a cake with connected pieces when k $\le$ n -1. This
result was known only for k = 2. These theorems say that we can get an
impossibility result even if we do not consider an envy-free division but a
weaker notion. Finally, k-proportionality allows to give a generalization with
a uniform statement of theorems about strong envy-free and strong proportional
divisions.

### 2. [HLSMAC: A New StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making](http://arxiv.org/pdf/2509.12927v1)

Authors: Xingxing Hong, Yungong Wang, Dexin Jin, Ye Yuan, Ximing Huang, Zijian Wu, Wenxin Li

Benchmarks are crucial for assessing multi-agent reinforcement learning
(MARL) algorithms. While StarCraft II-related environments have driven
significant advances in MARL, existing benchmarks like SMAC focus primarily on
micromanagement, limiting comprehensive evaluation of high-level strategic
intelligence. To address this, we introduce HLSMAC, a new cooperative MARL
benchmark with 12 carefully designed StarCraft II scenarios based on classical
stratagems from the Thirty-Six Stratagems. Each scenario corresponds to a
specific stratagem and is designed to challenge agents with diverse strategic
elements, including tactical maneuvering, timing coordination, and deception,
thereby opening up avenues for evaluating high-level strategic decision-making
capabilities. We also propose novel metrics across multiple dimensions beyond
conventional win rate, such as ability utilization and advancement efficiency,
to assess agents' overall performance within the HLSMAC environment. We
integrate state-of-the-art MARL algorithms and LLM-based agents with our
benchmark and conduct comprehensive experiments. The results demonstrate that
HLSMAC serves as a robust testbed for advancing multi-agent strategic
decision-making.

### Human-Computer Interaction

### 1. [Conflect: Designing Reflective Thinking-Based Contextual Privacy Policy for Mobile Applications](http://arxiv.org/pdf/2509.12578v1)

Authors: Shuning Zhang, Sixing Tao, Eve He, Yuting Yang, Ying Ma, Ailei Wang, Xin Yi, Hewu Li

Privacy policies are lengthy and complex, leading to user neglect. While
contextual privacy policies (CPPs) present information at the point of risk,
they may lack engagement and disrupt tasks. We propose Conflect, an interactive
CPP for mobile apps, guided by a reflective thinking framework. Through three
workshops with experienced designers and researchers, we constructed the design
space of reflective thinking-based CPP design, and identified the disconnect
between context and action as the most critical problem. Based on participants'
feedback, we designed Conflect to use sidebar alerts, allowing users to reflect
on contextualized risks and fostering their control. Our system contextually
detects privacy risks, extracts policy segments, and automatically generates
risk descriptions with 94.0% policy extraction accuracy on CPP4APP dataset and
a 4.35s latency. A user study (N=28) demonstrated that Conflect improves user
understanding, trust, and satisfaction while lowering cognitive load compared
to CPPs, privacy policies and privacy labels.

### 2. [DPCheatSheet: Using Worked and Erroneous LLM-usage Examples to Scaffold Differential Privacy Implementation](http://arxiv.org/pdf/2509.12590v1)

Authors: Shao-Yu Chu, Yuhe Tian, Yu-Xiang Wang, Haojian Jin

This paper explores how programmers without specialized expertise in
differential privacy (DP) (i.e., novices) can leverage LLMs to implement DP
programs with minimal training. We first conducted a need-finding study with 6
novices and 3 experts to understand how they utilize LLMs in DP implementation.
While DP experts can implement correct DP analyses through a few prompts,
novices struggle to articulate their requirements in prompts and lack the
skills to verify the correctness of the generated code. We then developed
DPCheatSheet, an instructional tool that helps novices implement DP using LLMs.
DPCheatSheet combines two learning concepts: it annotates an expert's workflow
with LLMs as a worked example to bridge the expert mindset to novices, and it
presents five common mistakes in LLM-based DP code generation as erroneous
examples to support error-driven learning. We demonstrated the effectiveness of
DPCheatSheet with an error identification study and an open-ended DP
implementation study.

### 3. [Harnessing the Power of AI in Qualitative Research: Role Assignment, Engagement, and User Perceptions of AI-Generated Follow-Up Questions in Semi-Structured Interviews](http://arxiv.org/pdf/2509.12709v1)

Authors: He Zhang, Yueyan Liu, Xin Guan, Jie Cai, John M. Carroll

Semi-structured interviews highly rely on the quality of follow-up questions,
yet interviewers' knowledge and skills may limit their depth and potentially
affect outcomes. While many studies have shown the usefulness of large language
models (LLMs) for qualitative analysis, their possibility in the data
collection process remains underexplored. We adopt an AI-driven "Wizard-of-Oz"
setup to investigate how real-time LLM support in generating follow-up
questions shapes semi-structured interviews. Through a study with 17
participants, we examine the value of LLM-generated follow-up questions, the
evolving division of roles, relationships, collaborative behaviors, and
responsibilities between interviewers and AI. Our findings (1) provide
empirical evidence of the strengths and limitations of AI-generated follow-up
questions (AGQs); (2) introduce a Human-AI collaboration framework in this
interview context; and (3) propose human-centered design guidelines for
AI-assisted interviewing. We position LLMs as complements, not replacements, to
human judgment, and highlight pathways for integrating AI into qualitative data
collection.

### 4. [Participatory AI: A Scandinavian Approach to Human-Centered AI](http://arxiv.org/pdf/2509.12752v1)

Authors: Niklas Elmqvist, Eve Hoggan, Hans-Jörg Schulz, Marianne Graves Petersen, Peter Dalsgaard, Ira Assent, Olav W. Bertelsen, Akhil Arora, Kaj Grønbæk, Susanne Bødker, Clemens Nylandsted Klokmose, Rachel Charlotte Smith, Sebastian Hubenschmid, Christoph A. Johns, Gabriela Molina León, Anton Wolter, Johannes Ellemose, Vaishali Dhanoa, Simon Aagaard Enni, Mille Skovhus Lunding, Karl-Emil Kjær Bilstrup, Juan Sánchez Esquivel, Luke Connelly, Rafael Pablos Sarabia, Morten Birk, Joachim Nyborg, Stefanie Zollmann, Tobias Langlotz, Meredith Siang-Yun Chou, Jens Emil Sloth Grønbæk, Michael Wessely, Yijing Jiang, Caroline Berger, Duosi Dai, Michael Mose Biskjaer, Germán Leiva, Jonas Frich, Eva Eriksson, Kim Halskov, Thorbjørn Mikkelsen, Nearchos Potamitis, Michel Yildirim, Arvind Srinivasan, Jeanette Falk, Nanna Inie, Ole Sejer Iversen, Hugo Andersson

AI's transformative impact on work, education, and everyday life makes it as
much a political artifact as a technological one. Current AI models are opaque,
centralized, and overly generic. The algorithmic automation they provide
threatens human agency and democratic values in both workplaces and daily life.
To confront such challenges, we turn to Scandinavian Participatory Design (PD),
which was devised in the 1970s to face a similar threat from mechanical
automation. In the PD tradition, technology is seen not just as an artifact,
but as a locus of democracy. Drawing from this tradition, we propose
Participatory AI as a PD approach to human-centered AI that applies five PD
principles to four design challenges for algorithmic automation. We use
concrete case studies to illustrate how to treat AI models less as proprietary
products and more as shared socio-technical systems that enhance rather than
diminish human agency, human dignity, and human values.

### 5. [PLUTO: A Public Value Assessment Tool](http://arxiv.org/pdf/2509.12773v1)

Authors: Laura Koesten, Péter Ferenc Gyarmati, Connor Hogan, Bernhard Jordan, Seliem El-Sayed, Barbara Prainsack, Torsten Möller

We present PLUTO (Public VaLUe Assessment TOol), a framework for assessing
the public value of specific instances of data use. Grounded in the concept of
data solidarity, PLUTO aims to empower diverse stakeholders - including
regulatory bodies, private enterprises, NGOs, and individuals - to critically
engage with data projects through a structured assessment of the risks and
benefits of data use, and by encouraging critical reflection. This paper
discusses the theoretical foundation, development process, and initial user
experiences with PLUTO. Key challenges include translating qualitative
assessments of benefits and risks into actionable quantitative metrics while
maintaining inclusivity and transparency. Initial feedback highlights PLUTO's
potential to foster responsible decision-making and shared accountability in
data practices.

### 6. [The Impact of Automation on Risk-Taking: The Role of Sense of Agency](http://arxiv.org/pdf/2509.12794v1)

Authors: Yang Chen, Zhijun Zhang

Automation significantly alters human behavior, particularly risk-taking.
Previous researches have paid limited attention to the underlying
characteristics of automation and their mechanisms of influence on risk-taking.
This study investigated how automation affects risk-taking and examined the
role of sense of agency therein. By quantifying sense of agency through
subjective ratings, this research explored the impact of automation level and
reliability level on risk-taking. The results of three experiments indicated
that automation reduced the level of risk-taking; higher automation level was
associated with lower sense of agency and lower risk-taking, with sense of
agency playing a complete mediating role; higher automation reliability was
associated with higher sense of agency and higher risk-taking, with sense of
agency playing a partial mediating role. The study concludes that automation
influences risk-taking, such that higher automation level or lower reliability
is associated with a lower likelihood of risk-taking. Sense of agency mediates
the impact of automation on risk-taking, and automation level and reliability
have different effects on risk-taking.

### 7. [More than Meets the Eye: Understanding the Effect of Individual Objects on Perceived Visual Privacy](http://arxiv.org/pdf/2509.13051v1)

Authors: Mete Harun Akcay, Siddharth Prakash Rao, Alexandros Bakas, Buse Gul Atli

User-generated content, such as photos, comprises the majority of online
media content and drives engagement due to the human ability to process visual
information quickly. Consequently, many online platforms are designed for
sharing visual content, with billions of photos posted daily. However, photos
often reveal more than they intended through visible and contextual cues,
leading to privacy risks. Previous studies typically treat privacy as a
property of the entire image, overlooking individual objects that may carry
varying privacy risks and influence how users perceive it. We address this gap
with a mixed-methods study (n = 92) to understand how users evaluate the
privacy of images containing multiple sensitive objects. Our results reveal
mental models and nuanced patterns that uncover how granular details, such as
photo-capturing context and co-presence of other objects, affect privacy
perceptions. These novel insights could enable personalized, context-aware
privacy protection designs on social media and future technologies.

### 8. [Patient Perspectives on Telemonitoring during Colorectal Cancer Surgery Prehabilitation](http://arxiv.org/pdf/2509.13064v1)

Authors: Irina Bianca Serban, Dimitra Dritsa, David ten Cate, Loes Janssen, Margot Heijmans, Sara Colombo, Aarnout Brombacher, Steven Houben

Multimodal prehabilitation for colorectal cancer (CRC) surgery aims to
optimize patient fitness and reduce postoperative complications. While
telemonitoring's clinical value in supporting decision-making is recognized,
patient perspectives on its use in prehabilitation remain underexplored,
particularly compared to its related clinical context, rehabilitation. To
address this gap, we conducted interviews with five patients who completed a
four-week CRC prehabilitation program incorporating continuous telemonitoring.
Our findings reveal patients' willingness to engage with telemonitoring, shaped
by their motivations, perceived benefits, and concerns. We outline design
considerations for patient-centered systems and offer a foundation for further
research on telemonitoring in CRC prehabilitation.

### 9. [Towards an Embodied Composition Framework for Organizing Immersive Computational Notebooks](http://arxiv.org/pdf/2509.13291v1)

Authors: Sungwon In, Eric Krokos, Kirsten Whitley, Chris North, Yalong Yang

As immersive technologies evolve, immersive computational notebooks offer new
opportunities for interacting with code, data, and outputs. However, scaling
these environments remains a challenge, particularly when analysts manually
arrange large numbers of cells to maintain both execution logic and visual
coherence. To address this, we introduce an embodied composition framework,
facilitating organizational processes in the context of immersive computational
notebooks. To evaluate the effectiveness of the embodied composition framework,
we conducted a controlled user study comparing manual and embodied composition
frameworks in an organizational process. The results show that embodied
composition frameworks significantly reduced user effort and decreased
completion time. However, the design of the triggering mechanism requires
further refinement. Our findings highlight the potential of embodied
composition frameworks to enhance the scalability of the organizational process
in immersive computational notebooks.

### 10. [Investigating Seamless Transitions Between Immersive Computational Notebooks and Embodied Data Interactions](http://arxiv.org/pdf/2509.13295v1)

Authors: Sungwon In, Eric Krokos, Kirsten Whitley, Chris North, Yalong Yang

A growing interest in Immersive Analytics (IA) has led to the extension of
computational notebooks (e.g., Jupyter Notebook) into an immersive environment
to enhance analytical workflows. However, existing solutions rely on the WIMP
(windows, icons, menus, pointer) metaphor, which remains impractical for
complex data exploration. Although embodied interaction offers a more intuitive
alternative, immersive computational notebooks and embodied data exploration
systems are implemented as standalone tools. This separation requires analysts
to invest considerable effort to transition from one environment to an entirely
different one during analytical workflows. To address this, we introduce ICoN,
a prototype that facilitates a seamless transition between computational
notebooks and embodied data explorations within a unified, fully immersive
environment. Our findings reveal that unification improves transition
efficiency and intuitiveness during analytical workflows, highlighting its
potential for seamless data analysis.

### Information Retrieval

### 1. [DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval](http://arxiv.org/pdf/2509.12824v1)

Authors: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

Deep hashing models have been widely adopted to tackle the challenges of
large-scale image retrieval. However, these approaches face serious security
risks due to their vulnerability to adversarial examples. Despite the
increasing exploration of targeted attacks on deep hashing models, existing
approaches still suffer from a lack of multimodal guidance, reliance on
labeling information and dependence on pixel-level operations for attacks. To
address these limitations, we proposed DiffHash, a novel diffusion-based
targeted attack for deep hashing. Unlike traditional pixel-based attacks that
directly modify specific pixels and lack multimodal guidance, our approach
focuses on optimizing the latent representations of images, guided by text
information generated by a Large Language Model (LLM) for the target image.
Furthermore, we designed a multi-space hash alignment network to align the
high-dimension image space and text space to the low-dimension binary hash
space. During reconstruction, we also incorporated text-guided attention
mechanisms to refine adversarial examples, ensuring them aligned with the
target semantics while maintaining visual plausibility. Extensive experiments
have demonstrated that our method outperforms state-of-the-art (SOTA) targeted
attack methods, achieving better black-box transferability and offering more
excellent stability across datasets.

### 2. [A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System](http://arxiv.org/pdf/2509.12948v1)

Authors: Chao Xiong, Xianwen Yu, Wei Xu, Lei Cheng, Chuan Yuan, Linjian Mo

Pre-ranking plays a crucial role in large-scale recommender systems by
significantly improving the efficiency and scalability within the constraints
of providing high-quality candidate sets in real time. The two-tower model is
widely used in pre-ranking systems due to a good balance between efficiency and
effectiveness with decoupled architecture, which independently processes user
and item inputs before calculating their interaction (e.g. dot product or
similarity measure). However, this independence also leads to the lack of
information interaction between the two towers, resulting in less
effectiveness. In this paper, a novel architecture named learnable Fully
Interacted Two-tower Model (FIT) is proposed, which enables rich information
interactions while ensuring inference efficiency. FIT mainly consists of two
parts: Meta Query Module (MQM) and Lightweight Similarity Scorer (LSS).
Specifically, MQM introduces a learnable item meta matrix to achieve expressive
early interaction between user and item features. Moreover, LSS is designed to
further obtain effective late interaction between the user and item towers.
Finally, experimental results on several public datasets show that our proposed
FIT significantly outperforms the state-of-the-art baseline pre-ranking models.

### 3. [Green Recommender Systems: Understanding and Minimizing the Carbon Footprint of AI-Powered Personalization](http://arxiv.org/pdf/2509.13001v1)

Authors: Lukas Wegmeth, Tobias Vente, Alan Said, Joeran Beel

As global warming soars, the need to assess and reduce the environmental
impact of recommender systems is becoming increasingly urgent. Despite this,
the recommender systems community hardly understands, addresses, and evaluates
the environmental impact of their work. In this study, we examine the
environmental impact of recommender systems research by reproducing typical
experimental pipelines. Based on our results, we provide guidelines for
researchers and practitioners on how to minimize the environmental footprint of
their work and implement green recommender systems - recommender systems
designed to minimize their energy consumption and carbon footprint. Our
analysis covers 79 papers from the 2013 and 2023 ACM RecSys conferences,
comparing traditional "good old-fashioned AI" models with modern deep learning
models. We designed and reproduced representative experimental pipelines for
both years, measuring energy consumption using a hardware energy meter and
converting it into CO2 equivalents. Our results show that papers utilizing deep
learning models emit approximately 42 times more CO2 equivalents than papers
using traditional models. On average, a single deep learning-based paper
generates 2,909 kilograms of CO2 equivalents - more than the carbon emissions
of a person flying from New York City to Melbourne or the amount of CO2
sequestered by one tree over 260 years. This work underscores the urgent need
for the recommender systems and wider machine learning communities to adopt
green AI principles, balancing algorithmic advancements and environmental
responsibility to build a sustainable future with AI-powered personalization.

### 4. [Timbre-Adaptive Transcription: A Lightweight Architecture with Associative Memory for Dynamic Instrument Separation](http://arxiv.org/pdf/2509.12712v1)

Authors: Ruigang Li, Yongxu Zhu

Existing multi-timbre transcription models struggle with generalization
beyond pre-trained instruments and rigid source-count constraints. We address
these limitations with a lightweight deep clustering solution featuring: 1) a
timbre-agnostic backbone achieving state-of-the-art performance with only half
the parameters of comparable models, and 2) a novel associative memory
mechanism that mimics human auditory cognition to dynamically encode unseen
timbres via attention-based clustering. Our biologically-inspired framework
enables adaptive polyphonic separation with minimal training data (12.5
minutes), supported by a new synthetic dataset method offering cost-effective,
high-precision multi-timbre generation. Experiments show the timbre-agnostic
transcription model outperforms existing models on public benchmarks, while the
separation module demonstrates promising timbre discrimination. This work
provides an efficient framework for timbre-related music transcription and
explores new directions for timbre-aware separation through cognitive-inspired
architectures.

### 5. [Protecting participants or population? Comparison of k-anonymous Origin-Destination matrices](http://arxiv.org/pdf/2509.12950v1)

Authors: Pietro Armenante, Kai Huang, Nikhil Jha, Luca Vassio

Origin-Destination (OD) matrices are a core component of research on users'
mobility and summarize how individuals move between geographical regions. These
regions should be small enough to be representative of user mobility, without
incurring substantial privacy risks. There are two added values of the
NetMob2025 challenge dataset. Firstly, the data is extensive and contains a lot
of socio-demographic information that can be used to create multiple OD
matrices, based on the segments of the population. Secondly, a participant is
not merely a record in the data, but a statistically weighted proxy for a
segment of the real population. This opens the door to a fundamental shift in
the anonymization paradigm. A population-based view of privacy is central to
our contribution. By adjusting our anonymization framework to account for
representativeness, we are also protecting the inferred identity of the actual
population, rather than survey participants alone. The challenge addressed in
this work is to produce and compare OD matrices that are k-anonymous for survey
participants and for the whole population. We compare several traditional
methods of anonymization to k-anonymity by generalizing geographical areas.
These include generalization over a hierarchy (ATG and OIGH) and the classical
Mondrian. To this established toolkit, we add a novel method, i.e., ODkAnon, a
greedy algorithm aiming at balancing speed and quality. Unlike previous
approaches, which primarily address the privacy aspects of the given datasets,
we aim to contribute to the generation of privacy-preserving OD matrices
enriched with socio-demographic segmentation that achieves k-anonymity on the
actual population.

### 6. [Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM](http://arxiv.org/pdf/2509.13179v1)

Authors: Yushang Zhao, Xinyue Han, Qian Leng, Qianyi Sun, Haotian Lyu, Chengrui Zhou

The cold-start issue is the challenge when we talk about recommender systems,
especially in the case when we do not have the past interaction data of new
users or new items. Content-based features or hybrid solutions are common as
conventional solutions, but they can only work in a sparse metadata environment
with shallow patterns. In this paper, the efficient cold-start recommendation
strategy is presented, which is based on the sub word-level representations by
applying Byte Pair Encoding (BPE) tokenization and pre-trained Large Language
Model (LLM) embedding in the initialization procedure. We obtain fine-grained
token-level vectors that are aligned with the BPE vocabulary as opposed to
using coarse-grained sentence embeddings. Together, these token embeddings can
be used as dense semantic priors on unseen entities, making immediate
recommendation performance possible without user-item interaction history. Our
mechanism can be compared to collaborative filtering systems and tested over
benchmark datasets with stringent cold-start assumptions. Experimental findings
show that the given BPE-LLM method achieves higher Recall@k, NDCG@k, and Hit
Rate measurements compared to the standard baseline and displays the same
capability of sufficient computational performance. Furthermore, we demonstrate
that using subword-aware embeddings yields better generalizability and is more
interpretable, especially within a multilingual and sparse input setting. The
practical application of token-level semantic initialization as a lightweight,
but nevertheless effective extension to modern recommender systems in the
zero-shot setting is indicated within this work.

### 7. [LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations](http://arxiv.org/pdf/2509.12539v1)

Authors: Robin Vujanic, Thomas Rueckstiess

We present LEAF ("Lightweight Embedding Alignment Framework"), a knowledge
distillation framework for text embedding models. A key distinguishing feature
is that our distilled leaf models are aligned to their teacher. In the context
of information retrieval, this allows for flexible asymmetric architectures
where documents are encoded with the larger teacher model, while queries can be
served with the smaller leaf models. We also show that leaf models
automatically inherit MRL and robustness to output quantization whenever these
properties are present in the teacher model, without explicitly training for
them. To demonstrate the capability of our framework we publish leaf-ir, a 23M
parameters information retrieval oriented text embedding model trained using
LEAF, which sets a new state-of-the-art (SOTA) on BEIR, ranking #1 on the
public leaderboard for this benchmark and for models of its size. When run in
asymmetric mode, its retrieval performance is further increased. Our scheme is
however not restricted to the information retrieval setting, and we demonstrate
its wider applicability by synthesizing the multi-task leaf-mt model. This also
sets a new SOTA, ranking #1 on the public MTEB v2 (English) leaderboard for its
size. LEAF is applicable to black-box models and in contrast to other embedding
model training frameworks, it does not require judgments nor hard negatives,
and training can be conducted using small batch sizes. Thus, dataset and
training infrastructure requirements for our framework are modest. We make our
models publicly available under a permissive Apache 2.0 license.

### 8. [InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering](http://arxiv.org/pdf/2509.12765v1)

Authors: Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei, Yuqing Ding, Han Li

Retrieval-Augmented Generation (RAG) has emerged as a promising approach to
address key limitations of Large Language Models (LLMs), such as hallucination,
outdated knowledge, and lacking reference. However, current RAG frameworks
often struggle with identifying whether retrieved documents meaningfully
contribute to answer generation. This shortcoming makes it difficult to filter
out irrelevant or even misleading content, which notably impacts the final
performance. In this paper, we propose Document Information Gain (DIG), a novel
metric designed to quantify the contribution of retrieved documents to correct
answer generation. DIG measures a document's value by computing the difference
of LLM's generation confidence with and without the document augmented.
Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to
train a specialized reranker, which prioritizes each retrieved document from
exact distinguishing and accurate sorting perspectives. This approach can
effectively filter out irrelevant documents and select the most valuable ones
for better answer generation. Extensive experiments across various models and
benchmarks demonstrate that InfoGain-RAG can significantly outperform existing
approaches, on both single and multiple retrievers paradigm. Specifically on
NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match
accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG
respectively, and even an average of 15.3% increment on advanced proprietary
model GPT-4o across all datasets. These results demonstrate the feasibility of
InfoGain-RAG as it can offer a reliable solution for RAG in multiple
applications.

### 9. [Automated Generation of Research Workflows from Academic Papers: A Full-text Mining Framework](http://arxiv.org/pdf/2509.12955v1)

Authors: Heng Zhang, Chengzhi Zhang

The automated generation of research workflows is essential for improving the
reproducibility of research and accelerating the paradigm of "AI for Science".
However, existing methods typically extract merely fragmented procedural
components and thus fail to capture complete research workflows. To address
this gap, we propose an end-to-end framework that generates comprehensive,
structured research workflows by mining full-text academic papers. As a case
study in the Natural Language Processing (NLP) domain, our paragraph-centric
approach first employs Positive-Unlabeled (PU) Learning with SciBERT to
identify workflow-descriptive paragraphs, achieving an F1-score of 0.9772.
Subsequently, we utilize Flan-T5 with prompt learning to generate workflow
phrases from these paragraphs, yielding ROUGE-1, ROUGE-2, and ROUGE-L scores of
0.4543, 0.2877, and 0.4427, respectively. These phrases are then systematically
categorized into data preparation, data processing, and data analysis stages
using ChatGPT with few-shot learning, achieving a classification precision of
0.958. By mapping categorized phrases to their document locations in the
documents, we finally generate readable visual flowcharts of the entire
research workflows. This approach facilitates the analysis of workflows derived
from an NLP corpus and reveals key methodological shifts over the past two
decades, including the increasing emphasis on data analysis and the transition
from feature engineering to ablation studies. Our work offers a validated
technical framework for automated workflow generation, along with a novel,
process-oriented perspective for the empirical investigation of evolving
scientific paradigms. Source code and data are available at:
https://github.com/ZH-heng/research_workflow.

### 10. [ResidualViT for Efficient Temporally Dense Video Encoding](http://arxiv.org/pdf/2509.13255v1)

Authors: Mattia Soldan, Fabian Caba Heilbron, Bernard Ghanem, Josef Sivic, Bryan Russell

Several video understanding tasks, such as natural language temporal video
grounding, temporal activity localization, and audio description generation,
require "temporally dense" reasoning over frames sampled at high temporal
resolution. However, computing frame-level features for these tasks is
computationally expensive given the temporal resolution requirements. In this
paper, we make three contributions to reduce the cost of computing features for
temporally dense tasks. First, we introduce a vision transformer (ViT)
architecture, dubbed ResidualViT, that leverages the large temporal redundancy
in videos to efficiently compute temporally dense frame-level features. Our
architecture incorporates (i) learnable residual connections that ensure
temporal consistency across consecutive frames and (ii) a token reduction
module that enhances processing speed by selectively discarding temporally
redundant information while reusing weights of a pretrained foundation model.
Second, we propose a lightweight distillation strategy to approximate the
frame-level features of the original foundation model. Finally, we evaluate our
approach across four tasks and five datasets, in both zero-shot and fully
supervised settings, demonstrating significant reductions in computational cost
(up to 60%) and improvements in inference speed (up to 2.5x faster), all while
closely approximating the accuracy of the original foundation model.

### Machine Learning

### 1. [Graph Homophily Booster: Rethinking the Role of Discrete Features on Heterophilic Graphs](http://arxiv.org/pdf/2509.12530v1)

Authors: Ruizhong Qiu, Ting-Wei Li, Gaotang Li, Hanghang Tong

Graph neural networks (GNNs) have emerged as a powerful tool for modeling
graph-structured data. However, existing GNNs often struggle with heterophilic
graphs, where connected nodes tend to have dissimilar features or labels. While
numerous methods have been proposed to address this challenge, they primarily
focus on architectural designs without directly targeting the root cause of the
heterophily problem. These approaches still perform even worse than the
simplest MLPs on challenging heterophilic datasets. For instance, our
experiments show that 21 latest GNNs still fall behind the MLP on the Actor
dataset. This critical challenge calls for an innovative approach to addressing
graph heterophily beyond architectural designs. To bridge this gap, we propose
and study a new and unexplored paradigm: directly increasing the graph
homophily via a carefully designed graph transformation. In this work, we
present a simple yet effective framework called GRAPHITE to address graph
heterophily. To the best of our knowledge, this work is the first method that
explicitly transforms the graph to directly improve the graph homophily.
Stemmed from the exact definition of homophily, our proposed GRAPHITE creates
feature nodes to facilitate homophilic message passing between nodes that share
similar features. Furthermore, we both theoretically and empirically show that
our proposed GRAPHITE significantly increases the homophily of originally
heterophilic graphs, with only a slight increase in the graph size. Extensive
experiments on challenging datasets demonstrate that our proposed GRAPHITE
significantly outperforms state-of-the-art methods on heterophilic graphs while
achieving comparable accuracy with state-of-the-art methods on homophilic
graphs.

### 2. [Cross-Modal Deep Metric Learning for Time Series Anomaly Detection](http://arxiv.org/pdf/2509.12540v1)

Authors: Wei Li, Zheze Yang

To effectively address the issues of low sensitivity and high time
consumption in time series anomaly detection, we propose an anomaly detection
method based on cross-modal deep metric learning. A cross-modal deep metric
learning feature clustering model is constructed, composed of an input layer, a
triplet selection layer, and a loss function computation layer. The squared
Euclidean distances between cluster centers are calculated, and a stochastic
gradient descent strategy is employed to optimize the model and classify
different time series features. The inner product of principal component
direction vectors is used as a metric for anomaly measurement. The von
Mises-Fisher (vMF) distribution is applied to describe the directional
characteristics of time series data, and historical data is used to train and
obtain evaluation parameters. By comparing the principal component direction
vector of actual time series data with the threshold, anomaly detection is
performed. Experimental results demonstrate that the proposed method accurately
classifies time series data with different attributes, exhibits high
sensitivity to anomalies, and achieves high detection accuracy, fast detection
speed, and strong robustness.

### 3. [Exploring Training Data Attribution under Limited Access Constraints](http://arxiv.org/pdf/2509.12581v1)

Authors: Shiyuan Zhang, Junwei Deng, Juhan Bae, Jiaqi Ma

Training data attribution (TDA) plays a critical role in understanding the
influence of individual training data points on model predictions.
Gradient-based TDA methods, popularized by \textit{influence function} for
their superior performance, have been widely applied in data selection, data
cleaning, data economics, and fact tracing. However, in real-world scenarios
where commercial models are not publicly accessible and computational resources
are limited, existing TDA methods are often constrained by their reliance on
full model access and high computational costs. This poses significant
challenges to the broader adoption of TDA in practical applications.
  In this work, we present a systematic study of TDA methods under various
access and resource constraints. We investigate the feasibility of performing
TDA under varying levels of access constraints by leveraging appropriately
designed solutions such as proxy models. Besides, we demonstrate that
attribution scores obtained from models without prior training on the target
dataset remain informative across a range of tasks, which is useful for
scenarios where computational resources are limited. Our findings provide
practical guidance for deploying TDA in real-world environments, aiming to
improve feasibility and efficiency under limited access.

### 4. [High-Energy Concentration for Federated Learning in Frequency Domain](http://arxiv.org/pdf/2509.12630v1)

Authors: Haozhi Shi, Weiying Xie, Hangyu Ye, Daixun Li, Jitao Ma, Leyuan Fang

Federated Learning (FL) presents significant potential for collaborative
optimization without data sharing. Since synthetic data is sent to the server,
leveraging the popular concept of dataset distillation, this FL framework
protects real data privacy while alleviating data heterogeneity. However, such
methods are still challenged by the redundant information and noise in entire
spatial-domain designs, which inevitably increases the communication burden. In
this paper, we propose a novel Frequency-Domain aware FL method with
high-energy concentration (FedFD) to address this problem. Our FedFD is
inspired by the discovery that the discrete cosine transform predominantly
distributes energy to specific regions, referred to as high-energy
concentration. The principle behind FedFD is that low-energy like
high-frequency components usually contain redundant information and noise, thus
filtering them helps reduce communication costs and optimize performance. Our
FedFD is mathematically formulated to preserve the low-frequency components
using a binary mask, facilitating an optimal solution through frequency-domain
distribution alignment. In particular, real data-driven synthetic
classification is imposed into the loss to enhance the quality of the
low-frequency components. On five image and speech datasets, FedFD achieves
superior performance than state-of-the-art methods while reducing communication
costs. For example, on the CIFAR-10 dataset with Dirichlet coefficient $\alpha
= 0.01$, FedFD achieves a minimum reduction of 37.78\% in the communication
cost, while attaining a 10.88\% performance gain.

### 5. [ZTree: A Subgroup Identification Based Decision Tree Learning Framework](http://arxiv.org/pdf/2509.12688v1)

Authors: Eric Cheng, Jie Cheng

Decision trees are a commonly used class of machine learning models valued
for their interpretability and versatility, capable of both classification and
regression. We propose ZTree, a novel decision tree learning framework that
replaces CART's traditional purity based splitting with statistically
principled subgroup identification. At each node, ZTree applies hypothesis
testing (e.g., z-tests, t-tests, Mann-Whitney U, log-rank) to assess whether a
candidate subgroup differs meaningfully from the complement. To adjust for the
complication of multiple testing, we employ a cross-validation-based approach
to determine if further node splitting is needed. This robust stopping
criterion eliminates the need for post-pruning and makes the test threshold
(z-threshold) the only parameter for controlling tree complexity. Because of
the simplicity of the tree growing procedure, once a detailed tree is learned
using the most lenient z-threshold, all simpler trees can be derived by simply
removing nodes that do not meet the larger z-thresholds. This makes parameter
tuning intuitive and efficient. Furthermore, this z-threshold is essentially a
p-value, allowing users to easily plug in appropriate statistical tests into
our framework without adjusting the range of parameter search. Empirical
evaluation on five large-scale UCI datasets demonstrates that ZTree
consistently delivers strong performance, especially at low data regimes.
Compared to CART, ZTree also tends to grow simpler trees without sacrificing
performance. ZTree introduces a statistically grounded alternative to
traditional decision tree splitting by leveraging hypothesis testing and a
cross-validation approach to multiple testing correction, resulting in an
efficient and flexible framework.

### 6. [Soft Graph Transformer for MIMO Detection](http://arxiv.org/pdf/2509.12694v1)

Authors: Jiadong Hong, Lei Liu, Xinyu Bian, Wenjie Wang, Zhaoyang Zhang

We propose the Soft Graph Transformer (SGT), a Soft-Input-Soft-Output neural
architecture tailored for MIMO detection. While Maximum Likelihood (ML)
detection achieves optimal accuracy, its prohibitive exponential complexity
renders it impractical for real-world systems. Conventional message passing
algorithms offer tractable alternatives but rely on large-system asymptotics
and random matrix assumptions, both of which break down under practical
implementations. Prior Transformer-based detectors, on the other hand, fail to
incorporate the MIMO factor graph structure and cannot utilize decoder-side
soft information, limiting their standalone performance and their applicability
in iterative detection-decoding (IDD). To overcome these limitations, SGT
integrates message passing directly into a graph-aware attention mechanism and
supports decoder-informed updates through soft-input embeddings. This design
enables effective soft-output generation while preserving computational
efficiency. As a standalone detector, SGT closely approaches ML performance and
surpasses prior Transformer-based approaches.

### 7. [Bi-level Personalization for Federated Foundation Models: A Task-vector Aggregation Approach](http://arxiv.org/pdf/2509.12697v1)

Authors: Yiyuan Yang, Guodong Long, Qinghua Lu, Liming Zhu, Jing Jiang

Federated foundation models represent a new paradigm to jointly fine-tune
pre-trained foundation models across clients. It is still a challenge to
fine-tune foundation models for a small group of new users or specialized
scenarios, which typically involve limited data compared to the large-scale
data used in pre-training. In this context, the trade-off between
personalization and federation becomes more sensitive. To tackle these, we
proposed a bi-level personalization framework for federated fine-tuning on
foundation models. Specifically, we conduct personalized fine-tuning on the
client-level using its private data, and then conduct a personalized
aggregation on the server-level using similar users measured by client-specific
task vectors. Given the personalization information gained from client-level
fine-tuning, the server-level personalized aggregation can gain group-wise
personalization information while mitigating the disturbance of irrelevant or
interest-conflict clients with non-IID data. The effectiveness of the proposed
algorithm has been demonstrated by extensive experimental analysis in benchmark
datasets.

### 8. [NORA: A Nephrology-Oriented Representation Learning Approach Towards Chronic Kidney Disease Classification](http://arxiv.org/pdf/2509.12704v1)

Authors: Mohammad Abdul Hafeez Khan, Twisha Bhattacharyya, Omar Khan, Noorah Khan, Alina Aziz Fatima Khan, Mohammed Qutub Khan, Sujoy Ghosh Hajra

Chronic Kidney Disease (CKD) affects millions of people worldwide, yet its
early detection remains challenging, especially in outpatient settings where
laboratory-based renal biomarkers are often unavailable. In this work, we
investigate the predictive potential of routinely collected non-renal clinical
variables for CKD classification, including sociodemographic factors, comorbid
conditions, and urinalysis findings. We introduce the Nephrology-Oriented
Representation leArning (NORA) approach, which combines supervised contrastive
learning with a nonlinear Random Forest classifier. NORA first derives
discriminative patient representations from tabular EHR data, which are then
used for downstream CKD classification. We evaluated NORA on a clinic-based EHR
dataset from Riverside Nephrology Physicians. Our results demonstrated that
NORA improves class separability and overall classification performance,
particularly enhancing the F1-score for early-stage CKD. Additionally, we
assessed the generalizability of NORA on the UCI CKD dataset, demonstrating its
effectiveness for CKD risk stratification across distinct patient cohorts.

### 9. [Safe Reinforcement Learning using Action Projection: Safeguard the Policy or the Environment?](http://arxiv.org/pdf/2509.12833v1)

Authors: Hannah Markgraf, Shamburaj Sawant, Hanna Krasowski, Lukas Schäfer, Sebastien Gros, Matthias Althoff

Projection-based safety filters, which modify unsafe actions by mapping them
to the closest safe alternative, are widely used to enforce safety constraints
in reinforcement learning (RL). Two integration strategies are commonly
considered: Safe environment RL (SE-RL), where the safeguard is treated as part
of the environment, and safe policy RL (SP-RL), where it is embedded within the
policy through differentiable optimization layers. Despite their practical
relevance in safety-critical settings, a formal understanding of their
differences is lacking. In this work, we present a theoretical comparison of
SE-RL and SP-RL. We identify a key distinction in how each approach is affected
by action aliasing, a phenomenon in which multiple unsafe actions are projected
to the same safe action, causing information loss in the policy gradients. In
SE-RL, this effect is implicitly approximated by the critic, while in SP-RL, it
manifests directly as rank-deficient Jacobians during backpropagation through
the safeguard. Our contributions are threefold: (i) a unified formalization of
SE-RL and SP-RL in the context of actor-critic algorithms, (ii) a theoretical
analysis of their respective policy gradient estimates, highlighting the role
of action aliasing, and (iii) a comparative study of mitigation strategies,
including a novel penalty-based improvement for SP-RL that aligns with
established SE-RL practices. Empirical results support our theoretical
predictions, showing that action aliasing is more detrimental for SP-RL than
for SE-RL. However, with appropriate improvement strategies, SP-RL can match or
outperform improved SE-RL across a range of environments. These findings
provide actionable insights for choosing and refining projection-based safe RL
methods based on task characteristics.

### 10. [BAPFL: Exploring Backdoor Attacks Against Prototype-based Federated Learning](http://arxiv.org/pdf/2509.12964v1)

Authors: Honghong Zeng, Jiong Lou, Zhe Wang, Hefeng Zhou, Chentao Wu, Wei Zhao, Jie Li

Prototype-based federated learning (PFL) has emerged as a promising paradigm
to address data heterogeneity problems in federated learning, as it leverages
mean feature vectors as prototypes to enhance model generalization. However,
its robustness against backdoor attacks remains largely unexplored. In this
paper, we identify that PFL is inherently resistant to existing backdoor
attacks due to its unique prototype learning mechanism and local data
heterogeneity. To further explore the security of PFL, we propose BAPFL, the
first backdoor attack method specifically designed for PFL frameworks. BAPFL
integrates a prototype poisoning strategy with a trigger optimization
mechanism. The prototype poisoning strategy manipulates the trajectories of
global prototypes to mislead the prototype training of benign clients, pushing
their local prototypes of clean samples away from the prototypes of
trigger-embedded samples. Meanwhile, the trigger optimization mechanism learns
a unique and stealthy trigger for each potential target label, and guides the
prototypes of trigger-embedded samples to align closely with the global
prototype of the target label. Experimental results across multiple datasets
and PFL variants demonstrate that BAPFL achieves a 35\%-75\% improvement in
attack success rate compared to traditional backdoor attacks, while preserving
main task accuracy. These results highlight the effectiveness, stealthiness,
and adaptability of BAPFL in PFL.

### Neural and Evolutionary Computing

### 1. [A Neuromorphic Model of Learning Meaningful Sequences with Long-Term Memory](http://arxiv.org/pdf/2509.12850v1)

Authors: Laxmi R. Iyer, Ali A. Minai

Learning meaningful sentences is different from learning a random set of
words. When humans understand the meaning, the learning occurs relatively
quickly. What mechanisms enable this to happen? In this paper, we examine the
learning of novel sequences in familiar situations. We embed the Small World of
Words (SWOW-EN), a Word Association Norms (WAN) dataset, in a spiking neural
network based on the Hierarchical Temporal Memory (HTM) model to simulate
long-term memory. Results show that in the presence of SWOW-EN, there is a
clear difference in speed between the learning of meaningful sentences and
random noise. For example, short poems are learned much faster than sequences
of random words. In addition, the system initialized with SWOW-EN weights shows
greater tolerance to noise.

### 2. [Large Language Model-assisted Meta-optimizer for Automated Design of Constrained Evolutionary Algorithm](http://arxiv.org/pdf/2509.13251v1)

Authors: Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Weixiong Huang

Meta-black-box optimization has been significantly advanced through the use
of large language models (LLMs), yet in fancy on constrained evolutionary
optimization. In this work, AwesomeDE is proposed that leverages LLMs as the
strategy of meta-optimizer to generate update rules for constrained
evolutionary algorithm without human intervention. On the meanwhile, $RTO^2H$
framework is introduced for standardize prompt design of LLMs. The
meta-optimizer is trained on a diverse set of constrained optimization
problems. Key components, including prompt design and iterative refinement, are
systematically analyzed to determine their impact on design quality.
Experimental results demonstrate that the proposed approach outperforms
existing methods in terms of computational efficiency and solution accuracy.
Furthermore, AwesomeDE is shown to generalize well across distinct problem
domains, suggesting its potential for broad applicability. This research
contributes to the field by providing a scalable and data-driven methodology
for automated constrained algorithm design, while also highlighting limitations
and directions for future work.

### 3. [Efficient lattice field theory simulation using adaptive normalizing flow on a resistive memory-based neural differential equation solver](http://arxiv.org/pdf/2509.12812v1)

Authors: Meng Xu, Jichang Yang, Ning Lin, Qundao Xu, Siqi Tang, Han Wang, Xiaojuan Qi, Zhongrui Wang, Ming Xu

Lattice field theory (LFT) simulations underpin advances in classical
statistical mechanics and quantum field theory, providing a unified
computational framework across particle, nuclear, and condensed matter physics.
However, the application of these methods to high-dimensional systems remains
severely constrained by several challenges, including the prohibitive
computational cost and limited parallelizability of conventional sampling
algorithms such as hybrid Monte Carlo (HMC), the substantial training expense
associated with traditional normalizing flow models, and the inherent energy
inefficiency of digital hardware architectures. Here, we introduce a
software-hardware co-design that integrates an adaptive normalizing flow (ANF)
model with a resistive memory-based neural differential equation solver,
enabling efficient generation of LFT configurations. Software-wise, ANF enables
efficient parallel generation of statistically independent configurations,
thereby reducing computational costs, while low-rank adaptation (LoRA) allows
cost-effective fine-tuning across diverse simulation parameters. Hardware-wise,
in-memory computing with resistive memory substantially enhances both
parallelism and energy efficiency. We validate our approach on the scalar phi4
theory and the effective field theory of graphene wires, using a hybrid
analog-digital neural differential equation solver equipped with a 180 nm
resistive memory in-memory computing macro. Our co-design enables low-cost
computation, achieving approximately 8.2-fold and 13.9-fold reductions in
integrated autocorrelation time over HMC, while requiring fine-tuning of less
than 8% of the weights via LoRA. Compared to state-of-the-art GPUs, our
co-design achieves up to approximately 16.1- and 17.0-fold speedups for the two
tasks, as well as 73.7- and 138.0-fold improvements in energy efficiency.

### 4. [Traces Propagation: Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks](http://arxiv.org/pdf/2509.13053v1)

Authors: Lorenzo Pes, Bojian Yin, Sander Stuijk, Federico Corradi

Spiking Neural Networks (SNNs) provide an efficient framework for processing
dynamic spatio-temporal signals and for investigating the learning principles
underlying biological neural systems. A key challenge in training SNNs is to
solve both spatial and temporal credit assignment. The dominant approach for
training SNNs is Backpropagation Through Time (BPTT) with surrogate gradients.
However, BPTT is in stark contrast with the spatial and temporal locality
observed in biological neural systems and leads to high computational and
memory demands, limiting efficient training strategies and on-device learning.
Although existing local learning rules achieve local temporal credit assignment
by leveraging eligibility traces, they fail to address the spatial credit
assignment without resorting to auxiliary layer-wise matrices, which increase
memory overhead and hinder scalability, especially on embedded devices. In this
work, we propose Traces Propagation (TP), a forward-only, memory-efficient,
scalable, and fully local learning rule that combines eligibility traces with a
layer-wise contrastive loss without requiring auxiliary layer-wise matrices. TP
outperforms other fully local learning rules on NMNIST and SHD datasets. On
more complex datasets such as DVS-GESTURE and DVS-CIFAR10, TP showcases
competitive performance and scales effectively to deeper SNN architectures such
as VGG-9, while providing favorable memory scaling compared to prior fully
local scalable rules, for datasets with a significant number of classes.
Finally, we show that TP is well suited for practical fine-tuning tasks, such
as keyword spotting on the Google Speech Commands dataset, thus paving the way
for efficient learning at the edge.

### Networking and Internet Architecture

### 1. [A Unified Learning-based Optimization Framework for 0-1 Mixed Problems in Wireless Networks](http://arxiv.org/pdf/2509.12664v1)

Authors: Kairong Ma, Yao Sun, Shuheng Hua, Muhammad Ali Imran, Walid Saad

Several wireless networking problems are often posed as 0-1 mixed
optimization problems, which involve binary variables (e.g., selection of
access points, channels, and tasks) and continuous variables (e.g., allocation
of bandwidth, power, and computing resources). Traditional optimization methods
as well as reinforcement learning (RL) algorithms have been widely exploited to
solve these problems under different network scenarios. However, solving such
problems becomes more challenging when dealing with a large network scale,
multi-dimensional radio resources, and diversified service requirements. To
this end, in this paper, a unified framework that combines RL and optimization
theory is proposed to solve 0-1 mixed optimization problems in wireless
networks. First, RL is used to capture the process of solving binary variables
as a sequential decision-making task. During the decision-making steps, the
binary (0-1) variables are relaxed and, then, a relaxed problem is solved to
obtain a relaxed solution, which serves as prior information to guide RL
searching policy. Then, at the end of decision-making process, the search
policy is updated via suboptimal objective value based on decisions made. The
performance bound and convergence guarantees of the proposed framework are then
proven theoretically. An extension of this approach is provided to solve
problems with a non-convex objective function and/or non-convex constraints.
Numerical results show that the proposed approach reduces the convergence time
by about 30% over B&B in small-scale problems with slightly higher objective
values. In large-scale scenarios, it can improve the normalized objective
values by 20% over RL with a shorter convergence time.

### 2. [State Aware Traffic Generation for Real-Time Network Digital Twins](http://arxiv.org/pdf/2509.12860v1)

Authors: Enes Koktas, Peter Rost

Digital twins (DTs) enable smarter, self-optimizing mobile networks, but they
rely on a steady supply of real world data. Collecting and transferring
complete traces in real time is a significant challenge. We present a compact
traffic generator that combines hidden Markov model, capturing the broad
rhythms of buffering, streaming and idle periods, with a small feed forward
mixture density network that generates realistic payload sizes and
inter-arrival times to be fed to the DT. This traffic generator trains in
seconds on a server GPU, runs in real time and can be fine tuned inside the DT
whenever the statistics of the generated data do not match the actual traffic.
This enables operators to keep their DT up to date without causing overhead to
the operational network. The results show that the traffic generator presented
is able to derive realistic packet traces of payload length and inter-arrival
time across various metrics that assess distributional fidelity, diversity, and
temporal correlation of the synthetic trace.

### 3. [Secure and Efficient Out-of-band Call Metadata Transmission](http://arxiv.org/pdf/2509.12582v1)

Authors: David Adei, Varun Madathil, Nithin Shyam S., Bradley Reaves

The STIR/SHAKEN (S/S) attestation Framework mandated by the United States,
Canada, and France to combat pervasive telephone abuse has not achieved its
goals, partly because legacy non-VoIP infrastructure could not participate. The
industry solution to extend S/S broadcasts sensitive metadata of every non-VoIP
call in plaintext to every third party required to facilitate the system. It
has no mechanism to determine whether a provider's request for call data is
appropriate, nor can it ensure that every copy of that call data is unavailable
after its specified expiration. It threatens subscriber privacy and provider
confidentiality.
  In this paper, we present Sidecar, a distributed, privacy-preserving system
with tunable decentralization that securely extends S/S across all telephone
network technologies. We introduce the notion of secure out-of-band signaling
for telephony and formalize its system and security requirements. We then
design novel, scalable protocols that realize these requirements and prove
their security within the Universal Composability framework. Finally, we
demonstrate Sidecar's efficiency with our open-sourced reference
implementation. Compared to the current solution, Sidecar 1) protects the
confidentiality of subscriber identity and provider trade secrets, 2)
guarantees record expiration as long as a single node handling a record is
honest, 3) reduces resource requirements while providing virtually identical
call-setup times and equivalent or better uptimes, and 4) enables secure
pay-per-use billing and integrates mechanisms to mitigate and detect
misbehavior. Moreover, Sidecar can be extended to provide the same security
guarantees for arbitrary call metadata. Not only is Sidecar a superior
approach, it is also a transformative tool to retrofit fragmented global
telephony and enable future improvements, such as stronger call authentication
and Branded Calling.

### 4. [Joint AoI and Handover Optimization in Space-Air-Ground Integrated Network](http://arxiv.org/pdf/2509.12716v1)

Authors: Zifan Lang, Guixia Liu, Geng Sun, Jiahui Li, Jiacheng Wang, Weijie Yuan, Dusit Niyato, Dong In Kim

Despite the widespread deployment of terrestrial networks, providing reliable
communication services to remote areas and maintaining connectivity during
emergencies remains challenging. Low Earth orbit (LEO) satellite constellations
offer promising solutions with their global coverage capabilities and reduced
latency, yet struggle with intermittent coverage and limited communication
windows due to orbital dynamics. This paper introduces an age of information
(AoI)-aware space-air-ground integrated network (SAGIN) architecture that
leverages a high-altitude platform (HAP) as intelligent relay between the LEO
satellites and ground terminals. Our three-layer design employs hybrid
free-space optical (FSO) links for high-capacity satellite-to-HAP communication
and reliable radio frequency (RF) links for HAP-to-ground transmission, and
thus addressing the temporal discontinuity in LEO satellite coverage while
serving diverse user priorities. Specifically, we formulate a joint
optimization problem to simultaneously minimize the AoI and satellite handover
frequency through optimal transmit power distribution and satellite selection
decisions. This highly dynamic, non-convex problem with time-coupled
constraints presents significant computational challenges for traditional
approaches. To address these difficulties, we propose a novel diffusion model
(DM)-enhanced dueling double deep Q-network with action decomposition and state
transformer encoder (DD3QN-AS) algorithm that incorporates transformer-based
temporal feature extraction and employs a DM-based latent prompt generative
module to refine state-action representations through conditional denoising.
Simulation results highlight the superior performance of the proposed approach
compared with policy-based methods and some other deep reinforcement learning
(DRL) benchmarks.

### 5. [It Takes a Village: Bridging the Gaps between Current and Formal Specifications for Protocols](http://arxiv.org/pdf/2509.13208v1)

Authors: David Basin, Nate Foster, Kenneth L. McMillan, Kedar S. Namjoshi, Cristina Nita-Rotaru, Jonathan M. Smith, Pamela Zave, Lenore D. Zuck

Formal specifications have numerous benefits for both designers and users of
network protocols. They provide clear, unambiguous representations, which are
useful as documentation and for testing. They can help reveal disagreements
about what a protocol "is" and identify areas where further work is needed to
resolve ambiguities or internal inconsistencies. They also provide a foundation
for formal reasoning, making it possible to establish important security and
correctness guarantees on all inputs and in every environment. Despite these
advantages, formal methods are not widely used to design, implement, and
validate network protocols today. Instead, Internet protocols are usually
described in informal documents, such as IETF Requests for Comments (RFCs) or
IEEE standards. These documents primarily consist of lengthy prose
descriptions, accompanied by pseudocode, header descriptions, state machine
diagrams, and reference implementations which are used for interoperability
testing. So, while RFCs and reference implementations were only intended to
help guide the social process used by protocol designers, they have evolved
into the closest things to formal specifications the Internet community has. In
this paper, we discuss the different roles that specifications play in the
networking and formal methods communities. We then illustrate the potential
benefits of specifying protocols formally, presenting highlights from several
recent success stories. Finally, we identify key differences between how formal
specifications are understood by the two communities and suggest possible
strategies to bridge the gaps.

### 6. [CattleSense -- A Multisensory Approach to Optimize Cattle Well-Being](http://arxiv.org/pdf/2509.12617v1)

Authors: Srijesh Pillai, M. I. Jawid Nazir

CattleSense is an innovative application of Internet of Things (IoT)
technology for the comprehensive monitoring and management of cattle
well-being. This research paper outlines the design and implementation of a
sophisticated system using a Raspberry Pi Module 4B, RFID Card Reader, Electret
Arduino Microphone Module, DHT11 Sensor, Arduino UNO, Neo-6M GPS Sensor, and
Heartbeat Sensor. The system aims to provide real-time surveillance of the
environment in which Cows are present and individual Cow parameters such as
location, milking frequency, and heartbeat fluctuations. The primary objective
is to simplify managing the Cattle in the shed, ensuring that the Cattle are
healthy and safe.

### Robotics

### 1. [Robust Online Residual Refinement via Koopman-Guided Dynamics Modeling](http://arxiv.org/pdf/2509.12562v1)

Authors: Zhefei Gong, Shangke Lyu, Pengxiang Ding, Wei Xiao, Donglin Wang

Imitation learning (IL) enables efficient skill acquisition from
demonstrations but often struggles with long-horizon tasks and high-precision
control due to compounding errors. Residual policy learning offers a promising,
model-agnostic solution by refining a base policy through closed-loop
corrections. However, existing approaches primarily focus on local corrections
to the base policy, lacking a global understanding of state evolution, which
limits robustness and generalization to unseen scenarios. To address this, we
propose incorporating global dynamics modeling to guide residual policy
updates. Specifically, we leverage Koopman operator theory to impose linear
time-invariant structure in a learned latent space, enabling reliable state
transitions and improved extrapolation for long-horizon prediction and unseen
environments. We introduce KORR (Koopman-guided Online Residual Refinement), a
simple yet effective framework that conditions residual corrections on
Koopman-predicted latent states, enabling globally informed and stable action
refinement. We evaluate KORR on long-horizon, fine-grained robotic furniture
assembly tasks under various perturbations. Results demonstrate consistent
gains in performance, robustness, and generalization over strong baselines. Our
findings further highlight the potential of Koopman-based modeling to bridge
modern learning methods with classical control theory.

### 2. [PerchMobi^3: A Multi-Modal Robot with Power-Reuse Quad-Fan Mechanism for Air-Ground-Wall Locomotion](http://arxiv.org/pdf/2509.12620v1)

Authors: Yikai Chen, Zhi Zheng, Jin Wang, Bingye He, Xiangyu Xu, Jialu Zhang, Huan Yu, Guodong Lu

Achieving seamless integration of aerial flight, ground driving, and wall
climbing within a single robotic platform remains a major challenge, as
existing designs often rely on additional adhesion actuators that increase
complexity, reduce efficiency, and compromise reliability. To address these
limitations, we present PerchMobi^3, a quad-fan, negative-pressure,
air-ground-wall robot that implements a propulsion-adhesion power-reuse
mechanism. By repurposing four ducted fans to simultaneously provide aerial
thrust and negative-pressure adhesion, and integrating them with four actively
driven wheels, PerchMobi^3 eliminates dedicated pumps while maintaining a
lightweight and compact design. To the best of our knowledge, this is the first
quad-fan prototype to demonstrate functional power reuse for multi-modal
locomotion. A modeling and control framework enables coordinated operation
across ground, wall, and aerial domains with fan-assisted transitions. The
feasibility of the design is validated through a comprehensive set of
experiments covering ground driving, payload-assisted wall climbing, aerial
flight, and cross-mode transitions, demonstrating robust adaptability across
locomotion scenarios. These results highlight the potential of PerchMobi^3 as a
novel design paradigm for multi-modal robotic mobility, paving the way for
future extensions toward autonomous and application-oriented deployment.

### 3. [Safety filtering of robotic manipulation under environment uncertainty: a computational approach](http://arxiv.org/pdf/2509.12674v1)

Authors: Anna Johansson, Daniel Lindmark, Viktor Wiberg, Martin Servin

Robotic manipulation in dynamic and unstructured environments requires safety
mechanisms that exploit what is known and what is uncertain about the world.
Existing safety filters often assume full observability, limiting their
applicability in real-world tasks. We propose a physics-based safety filtering
scheme that leverages high-fidelity simulation to assess control policies under
uncertainty in world parameters. The method combines dense rollout with nominal
parameters and parallelizable sparse re-evaluation at critical
state-transitions, quantified through generalized factors of safety for stable
grasping and actuator limits, and targeted uncertainty reduction through
probing actions. We demonstrate the approach in a simulated bimanual
manipulation task with uncertain object mass and friction, showing that unsafe
trajectories can be identified and filtered efficiently. Our results highlight
physics-based sparse safety evaluation as a scalable strategy for safe robotic
manipulation under uncertainty.

### 4. [UDON: Uncertainty-weighted Distributed Optimization for Multi-Robot Neural Implicit Mapping under Extreme Communication Constraints](http://arxiv.org/pdf/2509.12702v1)

Authors: Hongrui Zhao, Xunlan Zhou, Boris Ivanovic, Negar Mehr

Multi-robot mapping with neural implicit representations enables the compact
reconstruction of complex environments. However, it demands robustness against
communication challenges like packet loss and limited bandwidth. While prior
works have introduced various mechanisms to mitigate communication disruptions,
performance degradation still occurs under extremely low communication success
rates. This paper presents UDON, a real-time multi-agent neural implicit
mapping framework that introduces a novel uncertainty-weighted distributed
optimization to achieve high-quality mapping under severe communication
deterioration. The uncertainty weighting prioritizes more reliable portions of
the map, while the distributed optimization isolates and penalizes mapping
disagreement between individual pairs of communicating agents. We conduct
extensive experiments on standard benchmark datasets and real-world robot
hardware. We demonstrate that UDON significantly outperforms existing
baselines, maintaining high-fidelity reconstructions and consistent scene
representations even under extreme communication degradation (as low as 1%
success rate).

### 5. [NAMOUnc: Navigation Among Movable Obstacles with Decision Making on Uncertainty Interval](http://arxiv.org/pdf/2509.12723v1)

Authors: Kai Zhang, Eric Lucet, Julien Alexandre Dit Sandretto, Shoubin Chen, David Filait

Navigation among movable obstacles (NAMO) is a critical task in robotics,
often challenged by real-world uncertainties such as observation noise, model
approximations, action failures, and partial observability. Existing solutions
frequently assume ideal conditions, leading to suboptimal or risky decisions.
This paper introduces NAMOUnc, a novel framework designed to address these
uncertainties by integrating them into the decision-making process. We first
estimate them and compare the corresponding time cost intervals for removing
and bypassing obstacles, optimizing both the success rate and time efficiency,
ensuring safer and more efficient navigation. We validate our method through
extensive simulations and real-world experiments, demonstrating significant
improvements over existing NAMO frameworks. More details can be found in our
website: https://kai-zhang-er.github.io/namo-uncertainty/

### 6. [NavMoE: Hybrid Model- and Learning-based Traversability Estimation for Local Navigation via Mixture of Experts](http://arxiv.org/pdf/2509.12747v1)

Authors: Botao He, Amir Hossein Shahidzadeh, Yu Chen, Jiayi Wu, Tianrui Guan, Guofei Chen, Howie Choset, Dinesh Manocha, Glen Chou, Cornelia Fermuller, Yiannis Aloimonos

This paper explores traversability estimation for robot navigation. A key
bottleneck in traversability estimation lies in efficiently achieving reliable
and robust predictions while accurately encoding both geometric and semantic
information across diverse environments. We introduce Navigation via Mixture of
Experts (NAVMOE), a hierarchical and modular approach for traversability
estimation and local navigation. NAVMOE combines multiple specialized models
for specific terrain types, each of which can be either a classical model-based
or a learning-based approach that predicts traversability for specific terrain
types. NAVMOE dynamically weights the contributions of different models based
on the input environment through a gating network. Overall, our approach offers
three advantages: First, NAVMOE enables traversability estimation to adaptively
leverage specialized approaches for different terrains, which enhances
generalization across diverse and unseen environments. Second, our approach
significantly improves efficiency with negligible cost of solution quality by
introducing a training-free lazy gating mechanism, which is designed to
minimize the number of activated experts during inference. Third, our approach
uses a two-stage training strategy that enables the training for the gating
networks within the hybrid MoE method that contains nondifferentiable modules.
Extensive experiments show that NAVMOE delivers a better efficiency and
performance balance than any individual expert or full ensemble across
different domains, improving cross- domain generalization and reducing average
computational cost by 81.2% via lazy gating, with less than a 2% loss in path
quality.

### 7. [Integrating Trajectory Optimization and Reinforcement Learning for Quadrupedal Jumping with Terrain-Adaptive Landing](http://arxiv.org/pdf/2509.12776v1)

Authors: Renjie Wang, Shangke Lyu, Xin Lang, Wei Xiao, Donglin Wang

Jumping constitutes an essential component of quadruped robots' locomotion
capabilities, which includes dynamic take-off and adaptive landing. Existing
quadrupedal jumping studies mainly focused on the stance and flight phase by
assuming a flat landing ground, which is impractical in many real world cases.
This work proposes a safe landing framework that achieves adaptive landing on
rough terrains by combining Trajectory Optimization (TO) and Reinforcement
Learning (RL) together. The RL agent learns to track the reference motion
generated by TO in the environments with rough terrains. To enable the learning
of compliant landing skills on challenging terrains, a reward relaxation
strategy is synthesized to encourage exploration during landing recovery
period. Extensive experiments validate the accurate tracking and safe landing
skills benefiting from our proposed method in various scenarios.

### 8. [A Novel Skill Modeling Approach: Integrating Vergnaud's Scheme with Cognitive Architectures](http://arxiv.org/pdf/2509.12851v1)

Authors: Antoine Lénat, Olivier Cheminat, Damien Chablat, Camilo Charron

Human-machine interaction is increasingly important in industry, and this
trend will only intensify with the rise of Industry 5.0. Human operators have
skills that need to be adapted when using machines to achieve the best results.
It is crucial to highlight the operator's skills and understand how they use
and adapt them [18]. A rigorous description of these skills is necessary to
compare performance with and without robot assistance. Predicate logic, used by
Vergnaud within Piaget's scheme concept, offers a promising approach. However,
this theory doesn't account for cognitive system constraints, such as the
timing of actions, the limitation of cognitive resources, the parallelization
of tasks, or the activation of automatic gestures contrary to optimal
knowledge. Integrating these constraints is essential for representing agent
skills understanding skill transfer between biological and mechanical
structures. Cognitive architectures models [2] address these needs by
describing cognitive structure and can be combined with the scheme for mutual
benefit. Welding provides a relevant case study, as it highlights the
challenges faced by operators, even highly skilled ones. Welding's complexity
stems from the need for constant skill adaptation to variable parameters like
part position and process. This adaptation is crucial, as weld quality, a key
factor, is only assessed afterward via destructive testing. Thus, the welder is
confronted with a complex perception-decision-action cycle, where the
evaluation of the impact of his actions is delayed and where errors are
definitive. This dynamic underscores the importance of understanding and
modeling the skills of operators.

### 9. [Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion](http://arxiv.org/pdf/2509.12858v1)

Authors: Yidan Lu, Rurui Yang, Qiran Kou, Mengting Chen, Tao Fan, Peter Cui, Yinzhao Dong, Peng Lu

Reinforcement learning has produced remarkable advances in humanoid
locomotion, yet a fundamental dilemma persists for real-world deployment:
policies must choose between the robustness of reactive proprioceptive control
or the proactivity of complex, fragile perception-driven systems. This paper
resolves this dilemma by introducing a paradigm that imbues a purely
proprioceptive policy with proactive capabilities, achieving the foresight of
perception without its deployment-time costs. Our core contribution is a
contrastive learning framework that compels the actor's latent state to encode
privileged environmental information from simulation. Crucially, this
``distilled awareness" empowers an adaptive gait clock, allowing the policy to
proactively adjust its rhythm based on an inferred understanding of the
terrain. This synergy resolves the classic trade-off between rigid, clocked
gaits and unstable clock-free policies. We validate our approach with zero-shot
sim-to-real transfer to a full-sized humanoid, demonstrating highly robust
locomotion over challenging terrains, including 30 cm high steps and 26.5{\deg}
slopes, proving the effectiveness of our method. Website:
https://lu-yidan.github.io/cra-loco.

### 10. [GRATE: a Graph transformer-based deep Reinforcement learning Approach for Time-efficient autonomous robot Exploration](http://arxiv.org/pdf/2509.12863v1)

Authors: Haozhan Ni, Jingsong Liang, Chenyu He, Yuhong Cao, Guillaume Sartoretti

Autonomous robot exploration (ARE) is the process of a robot autonomously
navigating and mapping an unknown environment. Recent Reinforcement Learning
(RL)-based approaches typically formulate ARE as a sequential decision-making
problem defined on a collision-free informative graph. However, these methods
often demonstrate limited reasoning ability over graph-structured data.
Moreover, due to the insufficient consideration of robot motion, the resulting
RL policies are generally optimized to minimize travel distance, while
neglecting time efficiency. To overcome these limitations, we propose GRATE, a
Deep Reinforcement Learning (DRL)-based approach that leverages a Graph
Transformer to effectively capture both local structure patterns and global
contextual dependencies of the informative graph, thereby enhancing the model's
reasoning capability across the entire environment. In addition, we deploy a
Kalman filter to smooth the waypoint outputs, ensuring that the resulting path
is kinodynamically feasible for the robot to follow. Experimental results
demonstrate that our method exhibits better exploration efficiency (up to 21.5%
in distance and 21.3% in time to complete exploration) than state-of-the-art
conventional and learning-based baselines in various simulation benchmarks. We
also validate our planner in real-world scenarios.

### Software Engineering

### 1. [Ensembling Large Language Models for Code Vulnerability Detection: An Empirical Evaluation](http://arxiv.org/pdf/2509.12629v1)

Authors: Zhihong Sun, Jia Li, Yao Wan, Chuanyi Li, Hongyu Zhang, Zhi jin, Ge Li, Hong Liu, Chen Lyu, Songlin Hu

Code vulnerability detection is crucial for ensuring the security and
reliability of modern software systems. Recently, Large Language Models (LLMs)
have shown promising capabilities in this domain. However, notable
discrepancies in detection results often arise when analyzing identical code
segments across different training stages of the same model or among
architecturally distinct LLMs. While such inconsistencies may compromise
detection stability, they also highlight a key opportunity: the latent
complementarity among models can be harnessed through ensemble learning to
create more robust vulnerability detection systems. In this study, we explore
the potential of ensemble learning to enhance the performance of LLMs in source
code vulnerability detection. We conduct comprehensive experiments involving
five LLMs (i.e., DeepSeek-Coder-6.7B, CodeLlama-7B, CodeLlama-13B,
CodeQwen1.5-7B, and StarCoder2-15B), using three ensemble strategies (i.e.,
Bagging, Boosting, and Stacking). These experiments are carried out across
three widely adopted datasets (i.e., Devign, ReVeal, and BigVul). Inspired by
Mixture of Experts (MoE) techniques, we further propose Dynamic Gated Stacking
(DGS), a Stacking variant tailored for vulnerability detection. Our results
demonstrate that ensemble approaches can significantly improve detection
performance, with Boosting excelling in scenarios involving imbalanced
datasets. Moreover, DGS consistently outperforms traditional Stacking,
particularly in handling class imbalance and multi-class classification tasks.
These findings offer valuable insights into building more reliable and
effective LLM-based vulnerability detection systems through ensemble learning.

### 2. [When Large Language Models Meet UAVs: How Far Are We?](http://arxiv.org/pdf/2509.12795v1)

Authors: Yihua Chen, Xingle Que, Jiashuo Zhang, Ting Chen, Guangshun Li, Jiachi Chen

The integration of unmanned aerial vehicles (UAVs) and large language models
(LLMs) has emerged as a research direction of growing interest, with the
potential to address challenges in autonomous decision-making, human-UAV
interaction, and real-time adaptability. However, existing studies have
remained largely in preliminary exploration with a limited understanding of
real-world practice, risking a misalignment between academic research and
practical needs and hindering the translation of results. To examine and
address these potential challenges, we conducted an empirical study of 74
selected papers and 56 public GitHub projects, identified nine task types for
LLMs in UAV systems, and quantified their distribution. Our findings show that
academic research emphasizes theoretical modeling and task optimization with
dispersed attention across tasks. In contrast, industrial projects focus on
flight control, task planning, and human-machine interaction, prioritizing
operability and efficiency. To further capture industry perspectives, we
distributed an online questionnaire. We obtained 52 valid responses: 40.4% of
practitioners have attempted to apply LLMs to UAV tasks. We further identify
factors that impede real-world integration, including technological maturity,
performance, safety, cost, and other considerations. Finally, we highlight
challenges for future development and provide recommendations.

### 3. [SateLight: A Satellite Application Update Framework for Satellite Computing](http://arxiv.org/pdf/2509.12809v1)

Authors: Jinfeng Wen, Jianshu Zhao, Zixi Zhu, Xiaomin Zhang, Qi Liang, Ao Zhou, Shangguang Wang

Satellite computing is an emerging paradigm that empowers satellites to
perform onboard processing tasks (i.e., \textit{satellite applications}),
thereby reducing reliance on ground-based systems and improving responsiveness.
However, enabling application software updates in this context remains a
fundamental challenge due to application heterogeneity, limited
ground-to-satellite bandwidth, and harsh space conditions. Existing software
update approaches, designed primarily for terrestrial systems, fail to address
these constraints, as they assume abundant computational capacity and stable
connectivity.
  To address this gap, we propose SateLight, a practical and effective
satellite application update framework tailored for satellite computing.
SateLight leverages containerization to encapsulate heterogeneous applications,
enabling efficient deployment and maintenance. SateLight further integrates
three capabilities: (1) a content-aware differential strategy that minimizes
communication data volume, (2) a fine-grained onboard update design that
reconstructs target applications, and (3) a layer-based fault-tolerant recovery
mechanism to ensure reliability under failure-prone space conditions.
Experimental results on a satellite simulation environment with 10
representative satellite applications demonstrate that SateLight reduces
transmission latency by up to 91.18% (average 56.54%) compared to the best
currently available baseline. It also consistently ensures 100% update
correctness across all evaluated applications. Furthermore, a case study on a
real-world in-orbit satellite demonstrates the practicality of our approach.

### 4. [Evaluating Large Language Models for Code Translation: Effects of Prompt Language and Prompt Design](http://arxiv.org/pdf/2509.12973v1)

Authors: Aamer Aljagthami, Mohammed Banabila, Musab Alshehri, Mohammed Kabini, Mohammad D. Alahmadi

Large language models (LLMs) have shown promise for automated source-code
translation, a capability critical to software migration, maintenance, and
interoperability. Yet comparative evidence on how model choice, prompt design,
and prompt language shape translation quality across multiple programming
languages remains limited. This study conducts a systematic empirical
assessment of state-of-the-art LLMs for code translation among C++, Java,
Python, and C#, alongside a traditional baseline (TransCoder). Using BLEU and
CodeBLEU, we quantify syntactic fidelity and structural correctness under two
prompt styles (concise instruction and detailed specification) and two prompt
languages (English and Arabic), with direction-aware evaluation across language
pairs. Experiments show that detailed prompts deliver consistent gains across
models and translation directions, and English prompts outperform Arabic by
13-15%. The top-performing model attains the highest CodeBLEU on challenging
pairs such as Java to C# and Python to C++. Our evaluation shows that each LLM
outperforms TransCoder across the benchmark. These results demonstrate the
value of careful prompt engineering and prompt language choice, and provide
practical guidance for software modernization and cross-language
interoperability.

### 5. [Automating Code Generation for Semiconductor Equipment Control from Developer Utterances with LLMs](http://arxiv.org/pdf/2509.13055v1)

Authors: Youngkyoung Kim, Sanghyeok Park, Misoo Kim, Gangho Yoon, Eunseok Lee, Simon S. Woo

Semiconductors form the backbone of modern electronics, with their
manufacturing and testing relying on highly specialized equipment and
domain-specific programming languages. Equipment languages such as the
Algorithmic Pattern Generator (ALPG) are critical for precise hardware control
but are challenging to program due to their low-level syntax and steep learning
curve. While large language models (LLMs) have shown promise in generating
high-level code from natural language, their effectiveness on low-level
equipment languages remains limited. To address this, we propose Progressive
Knowledge Enhancement (PKE), a novel multi-stage prompting framework that
progressively extracts and activates the latent knowledge within LLMs, guiding
them from simple to complex examples without extensive fine-tuning. Empirical
evaluation on an industrial ALPG dataset shows that PKE significantly
outperforms standard prompting and surpasses state-of-the-art methods in
generating correct ALPG code, achieving 11.1\% and 15.2\% higher exact match
scores compared to the second-best technique. Further analysis of individual
components confirms that progressive knowledge extraction based on difficulty
enhances accuracy. Our study offer a practical approach to boosting LLM
capabilities for specialized low-level programming, supporting greater
productivity in semiconductor software development.

### 6. [Accelerating Discovery: Rapid Literature Screening with LLMs](http://arxiv.org/pdf/2509.13103v1)

Authors: Santiago Matalonga, Domenico Amalfitano, Jean Carlo Rossa Hauck, Martín Solari, Guilherme H. Travassos

Background: Conducting Multi Vocal Literature Reviews (MVLRs) is often time
and effort-intensive. Researchers must review and filter a large number of
unstructured sources, which frequently contain sparse information and are
unlikely to be included in the final study. Our experience conducting an MVLR
on Context-Aware Software Systems (CASS) Testing in the avionics domain
exemplified this challenge, with over 8,000 highly heterogeneous documents
requiring review. Therefore, we developed a Large Language Model (LLM)
assistant to support the search and filtering of documents. Aims: To develop
and validate an LLM based tool that can support researchers in performing the
search and filtering of documents for an MVLR without compromising the rigor of
the research protocol. Method: We applied sound engineering practices to
develop an on-premises LLM-based tool incorporating Retrieval Augmented
Generation (RAG) to process candidate sources. Progress towards the aim was
quantified using the Positive Percent Agreement (PPA) as the primary metric to
ensure the performance of the LLM based tool. Convenience sampling, supported
by human judgment and statistical sampling, were used to verify and validate
the tool's quality-in-use. Results: The tool currently demonstrates a PPA
agreement with human researchers of 90% for sources that are not relevant to
the study. Development details are shared to support domain-specific adaptation
of the tool. Conclusions: Using LLM-based tools to support academic researchers
in rigorous MVLR is feasible. These tools can free valuable time for
higher-level, abstract tasks. However, researcher participation remains
essential to ensure that the tool supports thorough research.

### 7. [Optimizing Code Embeddings and ML Classifiers for Python Source Code Vulnerability Detection](http://arxiv.org/pdf/2509.13134v1)

Authors: Talaya Farasat, Joachim Posegga

In recent years, the growing complexity and scale of source code have
rendered manual software vulnerability detection increasingly impractical. To
address this challenge, automated approaches leveraging machine learning and
code embeddings have gained substantial attention. This study investigates the
optimal combination of code embedding techniques and machine learning
classifiers for vulnerability detection in Python source code. We evaluate
three embedding techniques, i.e., Word2Vec, CodeBERT, and GraphCodeBERT
alongside two deep learning classifiers, i.e., Bidirectional Long Short-Term
Memory (BiLSTM) networks and Convolutional Neural Networks (CNN). While CNN
paired with GraphCodeBERT exhibits strong performance, the BiLSTM model using
Word2Vec consistently achieves superior overall results. These findings suggest
that, despite the advanced architectures of recent models like CodeBERT and
GraphCodeBERT, classical embeddings such as Word2Vec, when used with
sequence-based models like BiLSTM, can offer a slight yet consistent
performance advantage. The study underscores the critical importance of
selecting appropriate combinations of embeddings and classifiers to enhance the
effectiveness of automated vulnerability detection systems, particularly for
Python source code.

### 8. [Towards the Next Generation of Software: Insights from Grey Literature on AI-Native Applications](http://arxiv.org/pdf/2509.13144v1)

Authors: Lingli Cao, Shanshan Li, Ying Fan, Danyang Li, Chenxing Zhong

Background: The rapid advancement of large language models (LLMs) has given
rise to AI-native applications, a new paradigm in software engineering that
fundamentally redefines how software is designed, developed, and evolved.
Despite their growing prominence, AI-native applications still lack a unified
engineering definition and architectural blueprint, leaving practitioners
without systematic guidance for system design, quality assurance, and
technology selection.
  Objective: This study seeks to establish a comprehensive understanding of
AI-native applications by identifying their defining characteristics, key
quality attributes, and typical technology stacks, as well as by clarifying the
opportunities and challenges they present.
  Method: We conducted a grey literature review, integrating conceptual
perspectives retrieved from targeted Google and Bing searches with practical
insights derived from leading open-source projects on GitHub. A structured
protocol encompassing source selection, quality assessment, and thematic
analysis was applied to synthesize findings across heterogeneous sources.
  Results: We finally identified 106 studies based on the selection criteria.
The analysis reveals that AI-native applications are distinguished by two core
pillars: the central role of AI as the system's intelligence paradigm and their
inherently probabilistic, non-deterministic nature. Critical quality attributes
include reliability, usability, performance efficiency, and AI-specific
observability. In addition, a typical technology stack has begun to emerge,
comprising LLM orchestration frameworks, vector databases, and AI-native
observability platforms. These systems emphasize response quality,
cost-effectiveness, and outcome predictability, setting them apart from
conventional software systems.
  Conclusion: This study is the first to propose a dual-layered engineering
blueprint...

### 9. [LLM-Based Approach for Enhancing Maintainability of Automotive Architectures](http://arxiv.org/pdf/2509.12798v1)

Authors: Nenad Petrovic, Lukasz Mazur, Alois Knoll

There are many bottlenecks that decrease the flexibility of automotive
systems, making their long-term maintenance, as well as updates and extensions
in later lifecycle phases increasingly difficult, mainly due to long
re-engineering, standardization, and compliance procedures, as well as
heterogeneity and numerosity of devices and underlying software components
involved. In this paper, we explore the potential of Large Language Models
(LLMs) when it comes to the automation of tasks and processes that aim to
increase the flexibility of automotive systems. Three case studies towards
achieving this goal are considered as outcomes of early-stage research: 1)
updates, hardware abstraction, and compliance, 2) interface compatibility
checking, and 3) architecture modification suggestions. For proof-of-concept
implementation, we rely on OpenAI's GPT-4o model.

### 10. [Validating Solidity Code Defects using Symbolic and Concrete Execution powered by Large Language Models](http://arxiv.org/pdf/2509.13023v1)

Authors: Ştefan-Claudiu Susan, Andrei Arusoaie, Dorel Lucanu

The high rate of false alarms from static analysis tools and Large Language
Models (LLMs) complicates vulnerability detection in Solidity Smart Contracts,
demanding methods that can formally or empirically prove the presence of
defects. This paper introduces a novel detection pipeline that integrates
custom Slither-based detectors, LLMs, Kontrol, and Forge. Our approach is
designed to reliably detect defects and generate proofs. We currently perform
experiments with promising results for seven types of critical defects. We
demonstrate the pipeline's efficacy by presenting our findings for three
vulnerabilities -- Reentrancy, Complex Fallback, and Faulty Access Control
Policies -- that are challenging for current verification solutions, which
often generate false alarms or fail to detect them entirely. We highlight the
potential of either symbolic or concrete execution in correctly classifying
such code faults. By chaining these instruments, our method effectively
validates true positives, significantly reducing the manual verification
burden. Although we identify potential limitations, such as the inconsistency
and the cost of LLMs, our findings establish a robust framework for combining
heuristic analysis with formal verification to achieve more reliable and
automated smart contract auditing.

### Social and Information Networks

### 1. [Ketto and the Science of Giving: A Data-Driven Investigation of Crowdfunding for India](http://arxiv.org/pdf/2509.12616v1)

Authors: Karuna Chandra, Akshay Menon, Lydia Manikonda, Ponnurangam Kumaraguru

The main goal of this paper is to investigate an up and coming crowdfunding
platform used to raise funds for social causes in India called Ketto. Despite
the growing usage of this platform, there is insufficient understanding in
terms of why users choose this platform when there are other popular platforms
such as GoFundMe. Using a dataset comprising of 119,493 Ketto campaigns, our
research conducts an in-depth investigation into different aspects of how the
campaigns on Ketto work with a specific focus on medical campaigns, which make
up the largest percentage of social causes in the dataset. We also perform
predictive modeling to identify the factors that contribute to the success of
campaigns on this platform. We use several features such as the campaign
metadata, description, geolocation, donor behaviors, and campaign-related
features to learn about the platform and its components. Our results suggest
that majority of the campaigns for medical causes seek funds to address chronic
health conditions, yet medical campaigns have the least success rate. Most of
the campaigns originate from the most populous states and major metropolitan
cities in India. Our analysis also indicates that factors such as online
engagement on the platform in terms of the number of comments, duration of the
campaign, and frequent updates on a campaign positively influence the funds
being raised. Overall, this preliminary work sheds light on the importance of
investigating various dynamics around crowdfunding for India-focused
community-driven needs.

### 2. [Extending the BEND Framework to Webgraphs](http://arxiv.org/pdf/2509.13212v1)

Authors: Evan M. Williams, Peter Carragher, Kathleen M. Carley

Attempts to manipulate webgraphs can have many downstream impacts, but
analysts lack shared quantitative metrics to characterize actions taken to
manipulate information environments at this level. We demonstrate how the BEND
framework can be used to characterize attempts to manipulate webgraph
information environments, and propose quantitative metrics for BEND community
maneuvers. We demonstrate the face validity of our proposed Webgraph BEND
metrics by using them to characterize two small web-graphs containing
SEO-boosted Kremlin-aligned websites. We demonstrate how our proposed metrics
improve BEND scores in webgraph settings and demonstrate the usefulness of our
metrics in characterizing webgraph information environments. These metrics
offer analysts a systematic and standardized way to characterize attempts to
manipulate webgraphs using common Search Engine Optimization tactics.

### 3. [A Pressure-Based Diffusion Model for Influence Maximization on Social Networks](http://arxiv.org/pdf/2509.12822v1)

Authors: Curt Stutsman, Eliot W. Robson, Abhishek K. Umrawal

In many real-world scenarios, an individual's local social network carries
significant influence over the opinions they form and subsequently propagate to
others. In this paper, we propose a novel diffusion model -- the Pressure
Threshold model (PT) -- for dynamically simulating the spread of influence
through a social network. This new model extends the popular Linear Threshold
Model (LT) by adjusting a node's outgoing influence proportional to the
influence it receives from its activated neighbors. We address the Influence
Maximization (IM) problem, which involves selecting the most effective seed
nodes to achieve maximal graph coverage after a diffusion process, and how the
problem manifests with the PT Model. Experiments conducted on real-world
networks, facilitated by enhancements to the open-source network-diffusion
Python library, CyNetDiff, demonstrate unique seed node selection for the PT
Model when compared to the LT Model. Moreover, analyses demonstrate that
densely connected networks amplify pressure effects more significantly than
sparse networks.

### 4. [Fast Unbiased Sampling of Networks with Given Expected Degrees and Strengths](http://arxiv.org/pdf/2509.13230v1)

Authors: Xuanchi Li, Xin Wang, Sadamori Kojaku

The configuration model is a cornerstone of statistical assessment of network
structure. While the Chung-Lu model is among the most widely used configuration
models, it systematically oversamples edges between large-degree nodes, leading
to inaccurate statistical conclusions. Although the maximum entropy principle
offers unbiased configuration models, its high computational cost has hindered
widespread adoption, making the Chung-Lu model an inaccurate yet persistently
practical choice. Here, we propose fast and efficient sampling algorithms for
the max-entropy-based models by adapting the Miller-Hagberg algorithm.
Evaluation on 103 empirical networks demonstrates 10-1000 times speedup, making
theoretically rigorous configuration models practical and contributing to a
more accurate understanding of network structure.

### 5. [Topology and Fragility of European High-Voltage Networks: A Cross-Country Comparative Analysis](http://arxiv.org/pdf/2509.12900v1)

Authors: Bálint Hartmann, Michelle T. Cirunay

Reliable electricity supply depends on the seamless operation of high-voltage
grid infrastructure spanning both transmission and sub-transmission levels.
Beneath this apparent uniformity lies a striking structural diversity, which
leaves a clear imprint on system vulnerability. In this paper, we present
harmonized topological models of the high-voltage grids of 15 European
countries, integrating all elements at voltage levels above 110 kV. Topological
analysis of these networks reveals a simple yet robust pattern: node degree
distributions consistently follow an exponential decay, but the rate of decay
varies significantly across countries. Through a detailed and systematic
evaluation of network tolerance to node and edge removals, we show that the
decay rate delineates the boundary between systems that are more resilient to
failures and those that are prone to large-scale disruptions. Furthermore, we
demonstrate that this numerical boundary is highly sensitive to which layers of
the infrastructure are included in the models. To our knowledge, this study
provides the first quantitative cross-country comparison of 15 European
high-voltage networks, linking topological properties with vulnerability
characteristics.

### 6. [Sublinear-Time Algorithms for Diagonally Dominant Systems and Applications to the Friedkin-Johnsen Model](http://arxiv.org/pdf/2509.13112v1)

Authors: Weiming Feng, Zelin Li, Pan Peng

We study sublinear-time algorithms for solving linear systems $Sz = b$, where
$S$ is a diagonally dominant matrix, i.e., $|S_{ii}| \geq \delta + \sum_{j \ne
i} |S_{ij}|$ for all $i \in [n]$, for some $\delta \geq 0$. We present
randomized algorithms that, for any $u \in [n]$, return an estimate $z_u$ of
$z^*_u$ with additive error $\varepsilon$ or $\varepsilon \lVert
z^*\rVert_\infty$, where $z^*$ is some solution to $Sz^* = b$, and the
algorithm only needs to read a small portion of the input $S$ and $b$. For
example, when the additive error is $\varepsilon$ and assuming $\delta>0$, we
give an algorithm that runs in time $O\left( \frac{\|b\|_\infty^2
S_{\max}}{\delta^3 \varepsilon^2} \log \frac{\| b \|_\infty}{\delta
\varepsilon} \right)$, where $S_{\max} = \max_{i \in [n]} |S_{ii}|$. We also
prove a matching lower bound, showing that the linear dependence on $S_{\max}$
is optimal. Unlike previous sublinear-time algorithms, which apply only to
symmetric diagonally dominant matrices with non-negative diagonal entries, our
algorithm works for general strictly diagonally dominant matrices ($\delta >
0$) and a broader class of non-strictly diagonally dominant matrices $(\delta =
0)$. Our approach is based on analyzing a simple probabilistic recurrence
satisfied by the solution. As an application, we obtain an improved
sublinear-time algorithm for opinion estimation in the Friedkin--Johnsen model.

### 7. [Podcasts as a Medium for Participation in Collective Action: A Case Study of Black Lives Matter](http://arxiv.org/pdf/2509.13197v1)

Authors: Theodora Moldovan, Arianna Pera, Davide Vega, Luca Maria Aiello

We study how participation in collective action is articulated in podcast
discussions, using the Black Lives Matter (BLM) movement as a case study. While
research on collective action discourse has primarily focused on text-based
content, this study takes a first step toward analyzing audio formats by using
podcast transcripts. Using the Structured Podcast Research Corpus (SPoRC), we
investigated spoken language expressions of participation in collective action,
categorized as problem-solution, call-to-action, intention, and execution. We
identified podcast episodes discussing racial justice after important
BLM-related events in May and June of 2020, and extracted participatory
statements using a layered framework adapted from prior work on social media.
We examined the emotional dimensions of these statements, detecting eight key
emotions and their association with varying stages of activism. We found that
emotional profiles vary by stage, with different positive emotions standing out
during calls-to-action, intention, and execution. We detected negative
associations between collective action and negative emotions, contrary to
theoretical expectations. Our work contributes to a better understanding of how
activism is expressed in spoken digital discourse and how emotional framing may
depend on the format of the discussion.

### Systems and Control

### 1. [Nonlinear Sampled-data Systems--A Lifting Framework](http://arxiv.org/pdf/2509.12681v1)

Authors: Yutaka Yamamoto, Kaoru Yamamoto

This short note gives a new framework for dealing with nonlinear sampled-data
systems. We introduce a new idea of lifting, which is well known for linear
systems, but not successfully generalized to nonlinear systems. This paper
introduces a new lifting technique for nonlinear, time-invariant systems, which
are different from the linear counterpart as developed in [Bamieh et al. 1991,
Yamamoto 1994], etc. The main difficulty is that the direct feedthrough term
effective in the linear case cannot be generalized to the nonlinear case.
Instead, we will further lift the state trajectory, and obtain an equivalent
time-invariant discrete-time system with function-space input and output
spaces. The basic framework, as well as the closed-loop equation with a
discrete-time controller, is given. As an application of this framework, we
give a representation for the Koopman operator derived from the given original
nonlinear system.

### 2. [MAPS: A Mode-Aware Probabilistic Scheduling Framework for LPV-Based Adaptive Control](http://arxiv.org/pdf/2509.12695v1)

Authors: Taehun Kim, Guntae Kim, Cheolmin Jeong, Chang Mook Kang

This paper proposes Mode-Aware Probabilistic Scheduling (MAPS), a novel
adaptive control framework tailored for DC motor systems experiencing varying
friction. MAPS uniquely integrates an Interacting Multiple Model (IMM)
estimator with a Linear Parameter-Varying (LPV) based control strategy,
leveraging real-time mode probability estimates to perform probabilistic gain
scheduling. A key innovation of MAPS lies in directly using the updated mode
probabilities as the interpolation weights for online gain synthesis in the LPV
controller, thereby tightly coupling state estimation with adaptive control.
This seamless integration enables the controller to dynamically adapt control
gains in real time, effectively responding to changes in frictional operating
modes without requiring explicit friction model identification. Validation on a
Hardware-in-the-Loop Simulation (HILS) environment demonstrates that MAPS
significantly enhances both state estimation accuracy and reference tracking
performance compared to Linear Quadratic Regulator (LQR) controllers relying on
predefined scheduling variables. These results establish MAPS as a robust,
generalizable solution for friction-aware adaptive control in uncertain,
time-varying environments, with practical real-time applicability.

### 3. [Towards Native AI in 6G Standardization: The Roadmap of Semantic Communication](http://arxiv.org/pdf/2509.12758v1)

Authors: Ping Zhang, Xiaodong Xu, Mengying Sun, Haixiao Gao, Nan Ma, Xiaoyun Wang, Ruichen Zhang, Jiacheng Wang, Dusit Niyato

Semantic communication (SemCom) has emerged as a transformative paradigm for
future 6G networks, offering task-oriented and meaning-aware transmission that
fundamentally redefines traditional bit-centric design. Recognized by leading
standardization bodies including the institute of electrical and electronics
engineers (IEEE) and the international telecommunication union (ITU), and
actively discussed within the 3rd generation partnership project (3GPP) working
groups, SemCom is rapidly gaining traction as a foundational enabler for
native-AI 6G. This paper presents a comprehensive overview of recent progress
in SemCom from both academic and industrial perspectives, with a focus on its
ongoing and upcoming standardization activities. We systematically examine
advances in representative application scenarios, architectural design,
semantic-traditional system compatibility, unified evaluation metrics, and
validation methodologies. Furthermore, we highlight several key enabling
technologies, such as joint source-channel coding (JSCC), SemCom-based multiple
access (MA) technologies such as model division MA (MDMA), and semantic
knowledge base (KB), that support the practical implementation of SemCom in
standard-compliant systems. Additionally, we present a case study for channel
state information (CSI) feedback, illustrating the concrete performance gains
of SemCom under 3GPP-compliant fading channels. Finally, we discuss emerging
challenges and research opportunities for incorporating semantic-native
mechanisms into the evolving 6G standardization landscape, and provide
forward-looking insights into its development and global adoption.

### 4. [Spatial Correlation and Degrees of Freedom in Arched HMIMO Arrays: A Closed-Form Analysis](http://arxiv.org/pdf/2509.12839v1)

Authors: Liuxun Xue, Shu Sun, Hangsong Yan

This paper presents a closed-form analysis of spatial correlation and degrees
of freedom (DoF) for arched holographic multiple-input multiple-output (HMIMO)
arrays, which can be viewed as a special form of fluid antenna systems (FAS)
when their geometry is fluidically adaptable. Unlike traditional planar
configurations, practical HMIMO surfaces may exhibit curvature, significantly
influencing their spatial characteristics and performance. We derive exact
correlation expressions for both arched uniform linear arrays and arched
uniform rectangular arrays, capturing curvature effects under far field
propagation. Our results reveal that isotropic scattering results in DoF being
dominated by the maximum span of the HMIMO array, such that shape effects are
weakened, and bending does not significantly reduce the available spatial DoF.
Numerical simulations validate the accuracy of the closed-form formulas and
demonstrate the robustness of DoF against curvature variations, supporting
flexible array designs. These findings offer fundamental insights into
geometry-aware optimization for next-generation HMIMO/FAS systems and pave the
way for practical implementations of curved HMIMO arrays.

### 5. [Grid-informed Sharing Coefficients in Renewable Energy Communities](http://arxiv.org/pdf/2509.12847v1)

Authors: Alireza Shooshtari, Antonio Pepiciello, José Luis Domínguez-García

The role of energy communities in grid operations is highly dependent on the
spatial distribution of their participants. In particular, when local energy
producers and consumers are concentrated in different feeders, economic
incentives from energy communities have the potential to affect local grid
congestion. To address this challenge, we propose a feeder-aware allocation
strategy that reflects grid topology in energy sharing. This strategy
prioritizes energy sharing within the same feeder, thus incentivizing local
generation-demand balance and improving grid operation. Different sharing
coefficients are tested, such as equal, proportional, and rank-based, in both
static and dynamic formulations. The proposed strategy is tested on data from a
real energy community, whose participants are assumed to be distributed across
four feeders. The analysis is carried out from the perspectives of the
community as a whole, individual feeders, and single participants. Simulation
results show that the feeder-aware strategy, in addition to promoting local
energy balance, leads to higher and more stable revenues for most participants.

### 6. [Momentum-Based Access and Speed Control for Improved Safety in Heterogeneous Road Networks](http://arxiv.org/pdf/2509.12944v1)

Authors: Felix Wieberneit, Emanuele Crisostomi, Wynita Griggs, Robert Shorten

The increasing variety of means of transportation, including light vehicles
like e-scooters and e-bikes, together with the increasing weight of
conventional vehicles due to electrification and consumer preferences for SUVs,
are raising serious concerns regarding the safety of road networks. In this
paper we design a two-level control algorithm to improve the safety of
heterogeneous networks: first, an access control strategy decreases the
heterogeneity of the network depending on actual traffic conditions; then, a
speed control strategy mitigates the probability of serious injuries in
potential collisions. Both control strategies are designed based on momentum
considerations, as this is regarded as the most influential variable to assess
injury risk. The road network mobility simulator SUMO is adopted to implement
and validate our proposed control strategies.

### 7. [CattleSense -- A Multisensory Approach to Optimize Cattle Well-Being](http://arxiv.org/pdf/2509.12617v1)

Authors: Srijesh Pillai, M. I. Jawid Nazir

CattleSense is an innovative application of Internet of Things (IoT)
technology for the comprehensive monitoring and management of cattle
well-being. This research paper outlines the design and implementation of a
sophisticated system using a Raspberry Pi Module 4B, RFID Card Reader, Electret
Arduino Microphone Module, DHT11 Sensor, Arduino UNO, Neo-6M GPS Sensor, and
Heartbeat Sensor. The system aims to provide real-time surveillance of the
environment in which Cows are present and individual Cow parameters such as
location, milking frequency, and heartbeat fluctuations. The primary objective
is to simplify managing the Cattle in the shed, ensuring that the Cattle are
healthy and safe.

### 8. [Loss-aware distributionally robust optimization via trainable optimal transport ambiguity sets](http://arxiv.org/pdf/2509.12689v1)

Authors: Jonas Ohnemus, Marta Fochesato, Riccardo Zuliani, John Lygeros

Optimal-Transport Distributionally Robust Optimization (OT-DRO) robustifies
data-driven decision-making under uncertainty by capturing the sampling-induced
statistical error via optimal transport ambiguity sets. The standard OT-DRO
pipeline consists of a two-step procedure, where the ambiguity set is first
designed and subsequently embedded into the downstream OT-DRO problem. However,
this separation between uncertainty quantification and optimization might
result in excessive conservatism. We introduce an end-to-end pipeline to
automatically learn decision-focused ambiguity sets for OT-DRO problems, where
the loss function informs the shape of the optimal transport ambiguity set,
leading to less conservative yet distributionally robust decisions. We
formulate the learning problem as a bilevel optimization program and solve it
via a hypergradient-based method. By leveraging the recently introduced
nonsmooth conservative implicit function theorem, we establish convergence to a
critical point of the bilevel problem. We present experiments validating our
method on standard portfolio optimization and linear regression tasks.

### 9. [Differentiable by Design Nonlinear Optimization and its application to Model Predictive Control](http://arxiv.org/pdf/2509.12692v1)

Authors: Riccardo Zuliani, Efe Balta, John Lygeros

Nonlinear optimization-based policies have seen large success in recent
years, primarily due to the incredible capabilities of nonlinear Model
Predictive Control (nMPC). These policies require solving computationally
demanding nonlinear optimization programs (NLP) online at each time-step. The
solution map of these NLPs, viewed as a function of the measured state of the
system and design parameters, may not be differentiable, which poses
significant challenges if the policy is designed with a policy optimization
scheme. In this paper, we propose a principled way to regularize NLPs to obtain
a surrogate derivative even if the NLP is not differentiable. The surrogate
problem is differentiable by design and its solution map coincides with the
solution of the unregularized problem. We demonstrate the effectiveness of our
approach in a free-final-time optimal control problem and a receding-horizon
nonlinear MPC example.

### 10. [Ellipsoidal partitions for improved multi-stage robust model predictive control](http://arxiv.org/pdf/2509.12792v1)

Authors: Moritz Heinlein, Florian Messerer, Moritz Diehl, Sergio Lucia

Ellipsoidal tube-based model predictive control methods effectively account
for the propagation of the reachable set, typically employing linear feedback
policies. In contrast, scenario-based approaches offer more flexibility in the
feedback structure by considering different control actions for different
branches of a scenario tree. However, they face challenges in ensuring rigorous
guarantees. This work aims to integrate the strengths of both methodologies by
enhancing ellipsoidal tube-based MPC with a scenario tree formulation. The
uncertainty ellipsoids are partitioned by halfspaces such that each partitioned
set can be controlled independently. The proposed ellipsoidal multi-stage
approach is demonstrated in a human-robot system, highlighting its advantages
in handling uncertainty while maintaining computational tractability.

### Machine Learning (Statistics Category)

### 1. [Selective Risk Certification for LLM Outputs via Information-Lift Statistics: PAC-Bayes, Robustness, and Skeleton Design](http://arxiv.org/pdf/2509.12527v1)

Authors: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma

Large language models often produce plausible but incorrect outputs. Existing
heuristics such as HallBayes lack formal guarantees. We develop the first
comprehensive theory of \emph{information-lift certificates} under selective
classification. Our contributions are: (i) a PAC-Bayes \emph{sub-gamma}
analysis extending beyond standard Bernstein bounds; (ii) explicit skeleton
sensitivity theorems quantifying robustness to misspecification; (iii)
failure-mode guarantees under assumption violations; and (iv) a principled
variational method for skeleton construction. Across six datasets and multiple
model families, we validate assumptions empirically, reduce abstention by
12--15\% at the same risk, and maintain runtime overhead below 20\% (further
reduced via batching).

### 2. [Modeling nonstationary spatial processes with normalizing flows](http://arxiv.org/pdf/2509.12884v1)

Authors: Pratik Nag, Andrew Zammit-Mangion, Ying Sun

Nonstationary spatial processes can often be represented as stationary
processes on a warped spatial domain. Selecting an appropriate spatial warping
function for a given application is often difficult and, as a result of this,
warping methods have largely been limited to two-dimensional spatial domains.
In this paper, we introduce a novel approach to modeling nonstationary,
anisotropic spatial processes using neural autoregressive flows (NAFs), a class
of invertible mappings capable of generating complex, high-dimensional
warpings. Through simulation studies we demonstrate that a NAF-based model has
greater representational capacity than other commonly used spatial process
models. We apply our proposed modeling framework to a subset of the 3D Argo
Floats dataset, highlighting the utility of our framework in real-world
applications.

### 3. [Reversible Deep Equilibrium Models](http://arxiv.org/pdf/2509.12917v1)

Authors: Sam McCallum, Kamran Arora, James Foster

Deep Equilibrium Models (DEQs) are an interesting class of implicit model
where the model output is implicitly defined as the fixed point of a learned
function. These models have been shown to outperform explicit (fixed-depth)
models in large-scale tasks by trading many deep layers for a single layer that
is iterated many times. However, gradient calculation through DEQs is
approximate. This often leads to unstable training dynamics and requires
regularisation or many function evaluations to fix. Here, we introduce
Reversible Deep Equilibrium Models (RevDEQs) that allow for exact gradient
calculation, no regularisation and far fewer function evaluations than DEQs. We
show that RevDEQs achieve state-of-the-art performance on language modelling
and image classification tasks against comparable implicit and explicit models.

### 4. [Causal Discovery via Quantile Partial Effect](http://arxiv.org/pdf/2509.12981v1)

Authors: Yikang Chen, Xingzhe Sun, Dehui Du

Quantile Partial Effect (QPE) is a statistic associated with conditional
quantile regression, measuring the effect of covariates at different levels.
Our theory demonstrates that when the QPE of cause on effect is assumed to lie
in a finite linear span, cause and effect are identifiable from their
observational distribution. This generalizes previous identifiability results
based on Functional Causal Models (FCMs) with additive, heteroscedastic noise,
etc. Meanwhile, since QPE resides entirely at the observational level, this
parametric assumption does not require considering mechanisms, noise, or even
the Markov assumption, but rather directly utilizes the asymmetry of shape
characteristics in the observational distribution. By performing basis function
tests on the estimated QPE, causal directions can be distinguished, which is
empirically shown to be effective in experiments on a large number of bivariate
causal discovery datasets. For multivariate causal discovery, leveraging the
close connection between QPE and score functions, we find that Fisher
Information is sufficient as a statistical measure to determine causal order
when assumptions are made about the second moment of QPE. We validate the
feasibility of using Fisher Information to identify causal order on multiple
synthetic and real-world multivariate causal discovery datasets.

### 5. [Learning Discrete Bayesian Networks with Hierarchical Dirichlet Shrinkage](http://arxiv.org/pdf/2509.13267v1)

Authors: Alexander Dombowsky, David B. Dunson

Discrete Bayesian networks (DBNs) provide a broadly useful framework for
modeling dependence structures in multivariate categorical data. There is a
vast literature on methods for inferring conditional probabilities and
graphical structure in DBNs, but data sparsity and parametric assumptions are
major practical issues. In this article, we detail a comprehensive Bayesian
framework for learning DBNs. First, we propose a hierarchical prior for the
conditional probabilities that enables complicated interactions between parent
variables and stability in sparse regimes. We give a novel Markov chain Monte
Carlo (MCMC) algorithm utilizing parallel Langevin proposals to generate exact
posterior samples, avoiding the pitfalls of variational approximations.
Moreover, we verify that the full conditional distribution of the concentration
parameters is log-concave under mild conditions, facilitating efficient
sampling. We then propose two methods for learning network structures,
including parent sets, Markov blankets, and DAGs, from categorical data. The
first cycles through individual edges each MCMC iteration, whereas the second
updates the entire structure as a single step. We evaluate the accuracy, power,
and MCMC performance of our methods on several simulation studies. Finally, we
apply our methodology to uncover prognostic network structure from primary
breast cancer samples.

### 6. [PBPK-iPINNs : Inverse Physics-Informed Neural Networks for Physiologically Based Pharmacokinetic Brain Models](http://arxiv.org/pdf/2509.12666v1)

Authors: Charuka D. Wickramasinghe, Krishanthi C. Weerasinghe, Pradeep K. Ranaweera

Physics-Informed Neural Networks (PINNs) leverage machine learning with
differential equations to solve direct and inverse problems, ensuring
predictions follow physical laws. Physiologically based pharmacokinetic (PBPK)
modeling advances beyond classical compartmental approaches by using a
mechanistic, physiology focused framework. A PBPK model is based on a system of
ODEs, with each equation representing the mass balance of a drug in a
compartment, such as an organ or tissue. These ODEs include parameters that
reflect physiological, biochemical, and drug-specific characteristics to
simulate how the drug moves through the body. In this paper, we introduce
PBPK-iPINN, a method to estimate drug-specific or patient-specific parameters
and drug concentration profiles in PBPK brain compartment models using inverse
PINNs. We demonstrate that, for the inverse problem to converge to the correct
solution, the loss function components (data loss, initial conditions loss, and
residual loss) must be appropriately weighted, and parameters (including number
of layers, number of neurons, activation functions, learning rate, optimizer,
and collocation points) must be carefully tuned. The performance of the
PBPK-iPINN approach is then compared with established traditional numerical and
statistical methods.

### 7. [Fast reconstruction of degenerate populations of conductance-based neuron models from spike times](http://arxiv.org/pdf/2509.12783v1)

Authors: Julien Brandoit, Damien Ernst, Guillaume Drion, Arthur Fyon

Neurons communicate through spikes, and spike timing is a crucial part of
neuronal processing. Spike times can be recorded experimentally both
intracellularly and extracellularly, and are the main output of
state-of-the-art neural probes. On the other hand, neuronal activity is
controlled at the molecular level by the currents generated by many different
transmembrane proteins called ion channels. Connecting spike timing to ion
channel composition remains an arduous task to date. To address this challenge,
we developed a method that combines deep learning with a theoretical tool
called Dynamic Input Conductances (DICs), which reduce the complexity of ion
channel interactions into three interpretable components describing how neurons
spike. Our approach uses deep learning to infer DICs directly from spike times
and then generates populations of "twin" neuron models that replicate the
observed activity while capturing natural variability in membrane channel
composition. The method is fast, accurate, and works using only spike
recordings. We also provide open-source software with a graphical interface,
making it accessible to researchers without programming expertise.

### 8. [Gaussian Mixture Model with unknown diagonal covariances via continuous sparse regularization](http://arxiv.org/pdf/2509.12889v1)

Authors: Romane Giard, Yohann de Castro, Clément Marteau

This paper addresses the statistical estimation of Gaussian Mixture Models
(GMMs) with unknown diagonal covariances from independent and identically
distributed samples. We employ the Beurling-LASSO (BLASSO), a convex
optimization framework that promotes sparsity in the space of measures, to
simultaneously estimate the number of components and their parameters. Our main
contribution extends the BLASSO methodology to multivariate GMMs with
component-specific unknown diagonal covariance matrices-a significantly more
flexible setting than previous approaches requiring known and identical
covariances. We establish non-asymptotic recovery guarantees with nearly
parametric convergence rates for component means, diagonal covariances, and
weights, as well as for density prediction. A key theoretical contribution is
the identification of an explicit separation condition on mixture components
that enables the construction of non-degenerate dual certificates-essential
tools for establishing statistical guarantees for the BLASSO. Our analysis
leverages the Fisher-Rao geometry of the statistical model and introduces a
novel semi-distance adapted to our framework, providing new insights into the
interplay between component separation, parameter space geometry, and
achievable statistical recovery.

### 9. [Optimal Conformal Prediction, E-values, Fuzzy Prediction Sets and Subsequent Decisions](http://arxiv.org/pdf/2509.13130v1)

Authors: Nick W. Koning, Sam van Meer

We make three contributions to conformal prediction. First, we propose fuzzy
conformal confidence sets that offer a degree of exclusion, generalizing beyond
the binary inclusion/exclusion offered by classical confidence sets. We connect
fuzzy confidence sets to e-values to show this degree of exclusion is
equivalent to an exclusion at different confidence levels, capturing precisely
what e-values bring to conformal prediction. We show that a fuzzy confidence
set is a predictive distribution with a more appropriate error guarantee.
Second, we derive optimal conformal confidence sets by interpreting the
minimization of the expected measure of the confidence set as an optimal
testing problem against a particular alternative. We use this to characterize
exactly in what sense traditional conformal prediction is optimal. Third, we
generalize the inheritance of guarantees by subsequent minimax decisions from
confidence sets to fuzzy confidence sets. All our results generalize beyond the
exchangeable conformal setting to prediction sets for arbitrary models. In
particular, we find that any valid test (e-value) for a hypothesis
automatically defines a (fuzzy) prediction confidence set.

### 10. [SURGIN: SURrogate-guided Generative INversion for subsurface multiphase flow with quantified uncertainty](http://arxiv.org/pdf/2509.13189v1)

Authors: Zhao Feng, Bicheng Yan, Luanxiao Zhao, Xianda Shen, Renyu Zhao, Wenhao Wang, Fengshou Zhang

We present a direct inverse modeling method named SURGIN, a SURrogate-guided
Generative INversion framework tailed for subsurface multiphase flow data
assimilation. Unlike existing inversion methods that require adaptation for
each new observational configuration, SURGIN features a zero-shot conditional
generation capability, enabling real-time assimilation of unseen monitoring
data without task-specific retraining. Specifically, SURGIN synergistically
integrates a U-Net enhanced Fourier Neural Operator (U-FNO) surrogate with a
score-based generative model (SGM), framing the conditional generation as a
surrogate prediction-guidance process in a Bayesian perspective. Instead of
directly learning the conditional generation of geological parameters, an
unconditional SGM is first pretrained in a self-supervised manner to capture
the geological prior, after which posterior sampling is performed by leveraging
a differentiable U-FNO surrogate to enable efficient forward evaluations
conditioned on unseen observations. Extensive numerical experiments demonstrate
SURGIN's capability to decently infer heterogeneous geological fields and
predict spatiotemporal flow dynamics with quantified uncertainty across diverse
measurement settings. By unifying generative learning with surrogate-guided
Bayesian inference, SURGIN establishes a new paradigm for inverse modeling and
uncertainty quantification in parametric functional spaces.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-09-17 PST.

### 1. [Bring us your LLMs: why peer review is good for AI models](https://www.nature.com/articles/d41586-025-02979-9)

Authors: 

### 2. [People are more likely to cheat when they delegate tasks to AI](https://www.nature.com/articles/d41586-025-02819-w)

Authors: Shoko Suzuki

### 3. [AI can learn to show its workings through trial and error](https://www.nature.com/articles/d41586-025-02703-7)

Authors: Daphne Ippolito et al.

### 4. [Secrets of DeepSeek AI model revealed in landmark paper](https://www.nature.com/articles/d41586-025-03015-6)

Authors: Elizabeth Gibney

### 5. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning](https://www.nature.com/articles/s41586-025-09422-z)

Authors: Daya Guo et al.

